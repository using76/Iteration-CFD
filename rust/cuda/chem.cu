// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
/*
  ==========================================================================
  chem.cu - the chemistry device kernels, SPEC-LIT 132.9. One thread per
  cell: the molar production rates omega_k(T, p, Y) of (130.1), and the
  constant-pressure reactor's right-hand side (132.7)-(132.9) with its
  one-sided finite-difference Jacobian (132.3).
  ==========================================================================

  Written from: SPEC-LIT.md §130.7, §132.3, §132.8, §132.9
  No GPL-licensed source was consulted.

  Every kernel computes in `typedef double ofchem` in both builds (132.6),
  reproducing the f64 host twins - src/chem/kinetics.rs's
  production_rates_tpy and src/chem/reactor.rs's ConstPressureReactor::rhs
  plus src/chem/rosenbrock.rs's fd_jacobian - choice for choice and in the
  hosts' operation order (132.9 D1). The production kernel's inputs and
  outputs are ofscalar; the Jacobian's buffers are double, internal to the
  chemistry step CHR-07b integrates in double. No atomics, no allocation,
  no sync: every thread touches only its own cell column c of the SoA
  arrays (stride n), so a launch is pure device work and capturable.

  ------------------------------------------------------------------------
  The flat mechanism (132.9 D2). The Rust twin of this layout is
  src/chem/device.rs::flatten; the two field lists must stay identical.
  ------------------------------------------------------------------------
  `mi` header, 11 ints:
    [0]  K            species count
    [1]  I            reaction count
    [2]  n_tb         dense efficiency rows
    [3]  base_r       offset into mf
    [4]  base_o       offset into mf
    [5]  base_s       offset into mf
    [6]  base_e       offset into mf
    [7]  base_need_g  offset into mi
    [8]  base_rx      offset into mi
    [9]  base_ord_sp  offset into mi
    [10] base_sto_sp  offset into mi

  `mf`: the species block at 0, 18 doubles per species k at 18k - W
  (kg/mol), t_low, t_mid, t_high, lower a1..a7, upper a1..a7; then at 18K
  R_GAS and p_ref, taken from the host so that no constant is spelled
  twice; then at base_r, 14 doubles per reaction i at base_r + 14i - A,
  beta, theta (the reaction rate), revA, revBeta, revTheta (the REV rate,
  or 0), lowA, lowBeta, lowTheta (the fall-off LOW rate, or 0), q0..q4
  (Troe: a, T***, T*, T**, 0 - T** 0 when absent; SRI: a, b, c, d, e;
  otherwise five 0s); then at base_o one order value per order entry;
  then at base_s one nu per stoichiometry entry; then at base_e n_tb rows
  of K alphas each, alpha = the listed efficiency, else 1.0.

  `mi` at base_need_g: K ints, 1 when species k appears in the reactants
  or the products of a reaction that is reversible without REV (the only
  time G/RT is read), else 0. At base_rx, 13 ints per reaction i at
  base_rx + 13i: kind (2 fall-off, else 1 third body, else 0), rev_kind
  (0 irreversible, 2 explicit REV, else 1 from ln K_c), tb_row (the dense
  row of the reaction's third body or (+M) collider, else -1),
  collider_species (the (+SPECIES) index, else -1), blending (-1 none,
  0 Lindemann, 1 Troe without T**, 2 Troe with T**, 3 SRI), fo_begin,
  fo_end (the reaction's range of the forward-order list), ro_begin,
  ro_end (the reverse-order list), re_begin, re_end (the reactant list),
  pr_begin, pr_end (the product list), all in stored order. At
  base_ord_sp: the species index of each order entry, the same index
  space as base_o's values. At base_sto_sp: the species index of each
  stoichiometry entry, the same index space as base_s's values.

  ------------------------------------------------------------------------
  The one-f64 control array of chemIntegrate (SPEC-LIT 132.10), uploaded
  once per DeviceReactor. The Rust twin of this layout is
  DeviceReactor::new in src/chem/device.rs; the two field lists must stay
  identical. The coefficients come from the set's own Coefficients, so no
  constant is spelled twice.
    [0]      gamma
    [1..17)  alpha 4x4 row-major
    [17..33) gamma_ij 4x4 row-major
    [33..37) b
    [37..41) b_hat
    [41]     stages
    [42]     embedded_order
    [43]     rtol
    [44]     atol_y
    [45]     atol_t
    [46]     h_max (+inf allowed)
    [47]     max_steps
  ------------------------------------------------------------------------

  CHR-14 adds the TCI closures of SPEC-LIT 134.6, still one thread per
  cell in ofchem double in both builds: chemEdmSource evaluates (134.2)'s
  rate of progress per global reaction and the per-species EDM source
  W_m omega-bar_m, and chemPasrScale the PaSR frozen-mean scale (134.13)
  with (134.14)'s mixing times and (134.15)'s reacting-fraction edges.
  Their flat EDM layout (ef, ei) is stated in the chemEdmSource comment
  below; the Rust twin of that layout is src/chem/tci.rs
  EdmMechanism::flatten, and the two field lists must stay identical.
  Their statuses extend 132.9's codes and never reuse them: 3 a non-finite
  Y, 11 BAD_MIXING, 12 BAD_DENSITY, 13 BAD_RATE; a refused cell writes
  NaN to every one of its wdot entries.
  ------------------------------------------------------------------------

  CHR-15 adds EDC (SPEC-LIT 134.7): chemEdcIntegrate integrates each
  cell's fine-structure reactor, (134.9) with (134.16)'s energy mixing
  term, from (Ybar, Tbar) over psrSpan tau_e by a copy of chemIntegrate's
  step loop whose right-hand side is chemEdcRhs - chemIntegrate itself is
  not edited, so its SASS is unchanged - then writes (134.17)'s residual
  verdict into the flag bits and (134.10)'s mean rate. Its host twin is
  src/chem/tci.rs's edc_host.
  ------------------------------------------------------------------------
*/

#include "ofgpu_device.cuh"

//- Everything below runs in double in both builds (132.6) -
typedef double ofchem;

//- (130.4): k = A T^beta exp(-theta/T), as kinetics.rs arrhenius -
OFGPU_DEV ofchem chemArrhenius(const ofchem* a, ofchem t)
{
    return a[0] * pow(t, a[1]) * exp(-a[2] / t);
}

//- (130.15)'s core: ln|A| + beta ln T - theta/T -
OFGPU_DEV ofchem chemLnArrhenius(const ofchem* a, ofchem t)
{
    return log(fabs(a[0])) + a[1] * log(t) - a[2] / t;
}

//- (128.1): cp/R on one NASA-7 set, in thermo.rs nasa7_cp_r's order -
OFGPU_DEV ofchem chemCpR(const ofchem* a, ofchem t)
{
    return a[0] + t * (a[1] + t * (a[2] + t * (a[3] + t * a[4])));
}

//- (128.2): H/(RT), in thermo.rs nasa7_h_rt's order -
OFGPU_DEV ofchem chemHrt(const ofchem* a, ofchem t)
{
    const ofchem t2 = t * t;
    return a[0] + a[1] * t / 2.0 + a[2] * t2 / 3.0 + a[3] * t2 * t / 4.0
        + a[4] * t2 * t2 / 5.0 + a[5] / t;
}

//- (128.3): S/R, in thermo.rs nasa7_s_r's order -
OFGPU_DEV ofchem chemSr(const ofchem* a, ofchem t)
{
    const ofchem t2 = t * t;
    return a[0] * log(t) + a[1] * t + a[2] * t2 / 2.0 + a[3] * t2 * t / 3.0
        + a[4] * t2 * t2 / 4.0 + a[6];
}

//- (128.4) as thermo.rs g_rt forms it: H/(RT) - S/R -
OFGPU_DEV ofchem chemGrt(const ofchem* a, ofchem t)
{
    return chemHrt(a, t) - chemSr(a, t);
}

//- A species record: sp[0] W, sp[1] t_low, sp[2] t_mid, sp[3] t_high,
//  sp[4..10] the lower set, sp[11..17] the upper set (128.5). The range
//  check is the record's own (128.1): 5 outside it, never extrapolated. -
OFGPU_DEV int chemRange(const ofchem* sp, ofchem t)
{
    if (!isfinite(t) || t < sp[1] || t > sp[3]) return 5;
    return 0;
}

//- (128.5)'s interval: the lower set at or below t_mid -
OFGPU_DEV const ofchem* chemCoeffs(const ofchem* sp, ofchem t)
{
    return (t <= sp[2]) ? sp + 4 : sp + 11;
}

//- (130.9)-(130.11): F_cent, the third term only when T** was given -
OFGPU_DEV ofchem chemFCent(const ofchem* q, ofchem t, int blending)
{
    ofchem f_cent = (1.0 - q[0]) * exp(-t / q[1]) + q[0] * exp(-t / q[2]);
    if (blending == 2) f_cent += exp(-q[3] / t);
    return f_cent;
}

//- (130.9)-(130.10) with d = 0.14; powi(2) as a product, 10^x as
//  pow(10.0, x), as kinetics.rs troe_f forms them -
OFGPU_DEV ofchem chemTroeF(ofchem f_cent, ofchem pr)
{
    const ofchem log_pr = log10(pr);
    const ofchem log_fc = log10(f_cent);
    const ofchem c = -0.4 - 0.67 * log_fc;
    const ofchem nn = 0.75 - 1.27 * log_fc;
    const ofchem x = log_pr + c;
    const ofchem r = x / (nn - 0.14 * x);
    return pow(10.0, log_fc / (1.0 + r * r));
}

//- (130.12), kinetics.rs sri_f's order -
OFGPU_DEV ofchem chemSriF(const ofchem* q, ofchem t, ofchem pr)
{
    const ofchem lp = log10(pr);
    const ofchem x = 1.0 / (1.0 + lp * lp);
    return q[3] * pow(q[0] * exp(-q[1] / t) + exp(-t / q[2]), x) * pow(t, q[4]);
}

//- (130.8)'s blending factor F: Lindemann's 1, else Troe's or SRI's -
OFGPU_DEV ofchem chemBlending(int blending, const ofchem* q, ofchem t, ofchem pr)
{
    if (blending == 0) return 1.0;
    if (blending == 3) return chemSriF(q, t, pr);
    return chemTroeF(chemFCent(q, t, blending), pr);
}

//- (130.1) for one cell, kinetics.rs production_rates in its own order:
//  G/RT for the need_g species (range 5 on the first miss), then the
//  reactions in mechanism order, each one progress() with the forward
//  term direct, the reverse one exp of the log sum (130.16)/(130.17),
//  and the accumulation -nu' q then +nu'' q into the ofchem scratch
//  column omg. T is already finite and above 0. Returns 0 or 5. -
OFGPU_DEV int chemOmega(ofchem t, const ofchem* conc, const ofchem* mf,
                        const int* mi, ofchem* g, ofchem* omg, oflabel n)
{
    const int K = mi[0];
    const int I = mi[1];
    const int base_r = mi[3];
    const int base_o = mi[4];
    const int base_s = mi[5];
    const int base_e = mi[6];
    const int base_rx = mi[8];
    const ofchem rg = mf[18 * K];
    const ofchem p_ref = mf[18 * K + 1];

    for (int k = 0; k < K; ++k)
    {
        g[k * n] = 0.0;
        omg[k * n] = 0.0;
    }
    //- Only the species of reversible reactions without REV are read, so
    //  only theirs is a range the T must be inside (130.7). -
    for (int k = 0; k < K; ++k)
    {
        if (mi[mi[7] + k] == 0) continue;
        const ofchem* sp = mf + 18 * k;
        const int st = chemRange(sp, t);
        if (st != 0) return st;
        g[k * n] = chemGrt(chemCoeffs(sp, t), t);
    }

    for (int i = 0; i < I; ++i)
    {
        const int* rx = mi + base_rx + 13 * i;
        const int kind = rx[0];
        const int rev_kind = rx[1];
        const ofchem* rt = mf + base_r + 14 * i;
        ofchem q_f = 0.0;
        ofchem q_r = 0.0;
        ofchem c_i = 1.0;
        ofchem ln_k_f = 0.0;
        int live = 1;
        if (kind == 2)
        {
            //- Fall-off (130.8): Pr = k_0 [M] / k_inf; the collider's [M]
            //  is one species' concentration, or (130.3)'s sum. -
            ofchem m;
            if (rx[3] >= 0)
            {
                m = conc[rx[3] * n];
            }
            else
            {
                const ofchem* al = mf + base_e + rx[2] * K;
                m = 0.0;
                for (int k = 0; k < K; ++k) m += al[k] * conc[k * n];
            }
            const ofchem pr = chemArrhenius(rt + 6, t) * m / chemArrhenius(rt, t);
            if (pr == 0.0)
            {
                live = 0; //- q_f = q_r = 0 exactly, log10 Pr never formed -
            }
            else
            {
                const ofchem p_fac = isfinite(pr) ? pr / (1.0 + pr) : 1.0;
                const ofchem f = chemBlending(rx[4], rt + 9, t, pr);
                ln_k_f = chemLnArrhenius(rt, t) + log(p_fac) + log(f);
                q_f = c_i * (chemArrhenius(rt, t) * p_fac * f);
            }
        }
        else
        {
            if (kind == 1)
            {
                const ofchem* al = mf + base_e + rx[2] * K;
                c_i = 0.0;
                for (int k = 0; k < K; ++k) c_i += al[k] * conc[k * n];
            }
            q_f = c_i * chemArrhenius(rt, t);
            ln_k_f = chemLnArrhenius(rt, t);
        }
        if (live)
        {
            //- (130.14)'s product; an order-0 factor is skipped on both
            //  sides, so 0 * ln 0 is never formed (130.7). -
            for (int e = rx[5]; e < rx[6]; ++e)
            {
                const ofchem o = mf[base_o + e];
                if (o == 0.0) continue;
                q_f *= pow(conc[mi[mi[9] + e] * n], o);
            }
            if (rev_kind != 0)
            {
                ofchem ln_c = 0.0;
                for (int e = rx[7]; e < rx[8]; ++e)
                {
                    const ofchem o = mf[base_o + e];
                    if (o == 0.0) continue;
                    ln_c += o * log(conc[mi[mi[9] + e] * n]);
                }
                if (rev_kind == 2)
                {
                    //- (130.17): the explicit reverse Arrhenius -
                    q_r = c_i * copysign(1.0, rt[3])
                        * exp(chemLnArrhenius(rt + 3, t) + ln_c);
                }
                else
                {
                    //- (130.16): one exp of the whole log sum, ln K_c of
                    //  (130.6) summed over the products then the
                    //  reactants -
                    ofchem acc = 0.0;
                    ofchem nu_sum = 0.0;
                    for (int e = rx[11]; e < rx[12]; ++e)
                    {
                        const ofchem nu = mf[base_s + e];
                        acc -= nu * g[mi[mi[10] + e] * n];
                        nu_sum += nu;
                    }
                    for (int e = rx[9]; e < rx[10]; ++e)
                    {
                        const ofchem nu = mf[base_s + e];
                        acc += nu * g[mi[mi[10] + e] * n];
                        nu_sum -= nu;
                    }
                    const ofchem ln_kc = acc + nu_sum * log(p_ref / (rg * t));
                    q_r = c_i * copysign(1.0, rt[0]) * exp(ln_k_f - ln_kc + ln_c);
                }
            }
        }
        //- (130.1): -nu' q to each reactant, +nu'' q to each product -
        const ofchem net = q_f - q_r;
        for (int e = rx[9]; e < rx[10]; ++e)
            omg[mi[mi[10] + e] * n] -= mf[base_s + e] * net;
        for (int e = rx[11]; e < rx[12]; ++e)
            omg[mi[mi[10] + e] * n] += mf[base_s + e] * net;
    }
    return 0;
}
//- (132.7)-(132.9) at one cell, ConstPressureReactor::rhs with R8's clip
//  in its own order: T refused (2), Yp = max(Y, 0), S = sum Yp/W refused
//  (4), rho = p/(R T S), c = rho Yp/W, omega, cp over Yp != 0 (5 on a
//  range miss), dY = W omega/rho, then the enthalpy sum over
//  omega != 0 (5) and dT = -sum/(rho cp). Returns 0 or the first refusal. -
OFGPU_DEV int chemRhs(const ofchem* y, ofchem p, const ofchem* mf, const int* mi,
                      ofchem* yp, ofchem* conc, ofchem* g, ofchem* omg,
                      ofchem* dydt, oflabel n)
{
    const int K = mi[0];
    const ofchem rg = mf[18 * K];
    const ofchem t = y[K * n];
    if (!isfinite(t) || t <= 0.0) return 2;
    for (int k = 0; k < K; ++k) yp[k * n] = fmax(y[k * n], 0.0);
    ofchem s = 0.0;
    for (int k = 0; k < K; ++k) s += yp[k * n] / mf[18 * k];
    if (isnan(s) || s <= 0.0) return 4;
    const ofchem rho = p / (rg * t * s);
    for (int k = 0; k < K; ++k) conc[k * n] = rho * yp[k * n] / mf[18 * k];
    const int st = chemOmega(t, conc, mf, mi, g, omg, n);
    if (st != 0) return st;
    ofchem cp = 0.0;
    for (int k = 0; k < K; ++k)
    {
        if (yp[k * n] == 0.0) continue;
        const ofchem* sp = mf + 18 * k;
        const int miss = chemRange(sp, t);
        if (miss != 0) return miss;
        cp += yp[k * n] * (rg * chemCpR(chemCoeffs(sp, t), t)) / mf[18 * k];
    }
    for (int l = 0; l < K; ++l) dydt[l * n] = mf[18 * l] * omg[l * n] / rho;
    ofchem sum = 0.0;
    for (int l = 0; l < K; ++l)
    {
        if (omg[l * n] == 0.0) continue;
        const ofchem* sp = mf + 18 * l;
        const int miss = chemRange(sp, t);
        if (miss != 0) return miss;
        sum += (rg * t * chemHrt(chemCoeffs(sp, t), t)) * omg[l * n];
    }
    dydt[K * n] = -sum / (rho * cp);
    return 0;
}
//- chemProductionRates: (130.1) from (T, p, Y) through (128.9), which is
//  production_rates_tpy of SPEC-LIT 130.7. Status (132.9 D3), in the
//  hosts' order of checks: 1 p not finite or not above 0, 2 T, 3 a mass
//  fraction negative or not finite, 4 no species present, 5 a T outside a
//  needed record's range. A refused cell writes NaN to every omega_k. -
extern "C" __global__ void chemProductionRates
(
    const ofscalar* __restrict__ t,
    const ofscalar* __restrict__ p,
    const ofscalar* __restrict__ y,
    ofscalar* __restrict__ omega,
    int* __restrict__ status,
    ofchem* __restrict__ conc,
    ofchem* __restrict__ g,
    ofchem* __restrict__ omg,
    const ofchem* __restrict__ mf,
    const int* __restrict__ mi,
    oflabel n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;
    const int K = mi[0];
    int st = 0;
    const ofscalar t_c = t[c];
    const ofscalar p_c = p[c];
    if (!isfinite(p_c) || p_c <= (ofscalar)0) st = 1;
    else if (!isfinite(t_c) || t_c <= (ofscalar)0) st = 2;
    if (st == 0)
    {
        for (int k = 0; k < K; ++k)
        {
            const ofscalar v = y[k * n + c];
            if (!isfinite(v) || v < (ofscalar)0) { st = 3; break; }
        }
    }
    ofchem s = 0.0;
    if (st == 0)
    {
        for (int k = 0; k < K; ++k) s += (ofchem)y[k * n + c] / mf[18 * k];
        if (isnan(s) || s <= 0.0) st = 4;
    }
    if (st == 0)
    {
        //- (128.9): x_k = Y_k (1/S) / W_k, c_k = p x_k / (R T), exactly
        //  thermo.rs mole_fractions and concentrations -
        const ofchem t_d = (ofchem)t_c;
        const ofchem p_d = (ofchem)p_c;
        const ofchem w = 1.0 / s;
        const ofchem rg = mf[18 * K];
        for (int k = 0; k < K; ++k)
            conc[k * n + c] = p_d * ((ofchem)y[k * n + c] * w / mf[18 * k]) / (rg * t_d);
        st = chemOmega(t_d, conc + c, mf, mi, g + c, omg + c, n);
        if (st == 0)
            for (int k = 0; k < K; ++k) omega[k * n + c] = (ofscalar)omg[k * n + c];
    }
    if (st != 0)
    {
        for (int k = 0; k < K; ++k) omega[k * n + c] = (ofscalar)nan("");
    }
    status[c] = st;
}
//- chemReactorJacobian: f0 = rhs(y) and 132.3's one-sided difference
//  column by column, delta_j = 2^-26 max(|y_j|, atol_j) (2^-26 is
//  f64::EPSILON.sqrt() exactly) - fd_jacobian of SPEC-LIT 132.7. The
//  status is the first refusal met at y or at any perturbed column, 1 p
//  first; a refused cell writes NaN to its f0 and every J entry. -
extern "C" __global__ void chemReactorJacobian
(
    const ofchem* __restrict__ state,
    const ofchem* __restrict__ p,
    const ofchem* __restrict__ atol,
    ofchem* __restrict__ f0,
    ofchem* __restrict__ jac,
    int* __restrict__ status,
    ofchem* __restrict__ conc,
    ofchem* __restrict__ g,
    ofchem* __restrict__ omg,
    ofchem* __restrict__ yp,
    ofchem* __restrict__ ypert,
    ofchem* __restrict__ fpert,
    const ofchem* __restrict__ mf,
    const int* __restrict__ mi,
    oflabel n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;
    const int K = mi[0];
    const int m = K + 1;
    int st = 0;
    const ofchem p_c = p[c];
    if (!isfinite(p_c) || p_c <= 0.0) st = 1;
    if (st == 0)
        st = chemRhs(state + c, p_c, mf, mi, yp + c, conc + c, g + c, omg + c,
                     f0 + c, n);
    for (int j = 0; j < m && st == 0; ++j)
    {
        const ofchem y_j = state[j * n + c];
        const ofchem delta = 1.4901161193847656e-08 * fmax(fabs(y_j), atol[j]);
        for (int i = 0; i < m; ++i) ypert[i * n + c] = state[i * n + c];
        ypert[j * n + c] = y_j + delta;
        st = chemRhs(ypert + c, p_c, mf, mi, yp + c, conc + c, g + c, omg + c,
                     fpert + c, n);
        if (st == 0)
        {
            for (int i = 0; i < m; ++i)
                jac[(i * m + j) * n + c] = (fpert[i * n + c] - f0[i * n + c]) / delta;
        }
    }
    if (st != 0)
    {
        for (int i = 0; i < m; ++i)
        {
            f0[i * n + c] = nan("");
            for (int j = 0; j < m; ++j) jac[(i * m + j) * n + c] = nan("");
        }
    }
    status[c] = st;
}
//- R1 (SPEC-LIT 132.7): in-place LU with partial pivoting of the row-major
//  m x m matrix in one cell's column of a stride-n buffer, taken as the
//  column pointer chemRhs takes (a[k*n] is entry (k/n, k%n) of the cell) -
//  the largest abs in the column, the FIRST one on a tie, a[i,k] /= a[k,k],
//  the row update skipped when the multiplier is exactly 0. piv[k] is the
//  row swapped into k at step k. Returns 0, or 1 when a pivot is exactly 0
//  or not finite (a singular matrix). rosenbrock.rs's lu_factor, one for
//  one. -
OFGPU_DEV int chemLuFactor(ofchem* a, int* piv, int m, oflabel n)
{
    for (int k = 0; k < m; ++k)
    {
        int sel = k;
        ofchem best = fabs(a[(k * m + k) * n]);
        for (int i = k + 1; i < m; ++i)
        {
            const ofchem v = fabs(a[(i * m + k) * n]);
            if (v > best) { best = v; sel = i; }
        }
        if (!isfinite(best) || best == 0.0) return 1;
        piv[k * n] = sel;
        if (sel != k)
        {
            for (int j = 0; j < m; ++j)
            {
                const ofchem tmp = a[(k * m + j) * n];
                a[(k * m + j) * n] = a[(sel * m + j) * n];
                a[(sel * m + j) * n] = tmp;
            }
        }
        for (int i = k + 1; i < m; ++i)
        {
            a[(i * m + k) * n] /= a[(k * m + k) * n];
            const ofchem mult = a[(i * m + k) * n];
            if (mult != 0.0)
            {
                for (int j = k + 1; j < m; ++j)
                    a[(i * m + j) * n] = __dsub_rn(a[(i * m + j) * n],
                        __dmul_rn(mult, a[(k * m + j) * n]));
            }
        }
    }
    return 0;
}
//- Solves A x = b with the factors chemLuFactor left in a, applying the
//  same row swaps to b and overwriting it with x, in rosenbrock.rs's
//  lu_solve order: the swaps, the forward substitution, the back
//  substitution with the diagonal division last. Column pointers, as
//  chemRhs takes them. -
OFGPU_DEV void chemLuSolve(const ofchem* a, const int* piv, int m, ofchem* b,
                           oflabel n)
{
    for (int k = 0; k < m; ++k)
    {
        const int pk = piv[k * n];
        const ofchem tmp = b[k * n];
        b[k * n] = b[pk * n];
        b[pk * n] = tmp;
    }
    for (int i = 0; i < m; ++i)
    {
        ofchem s = b[i * n];
        for (int j = 0; j < i; ++j)
            s = __dsub_rn(s, __dmul_rn(a[(i * m + j) * n], b[j * n]));
        b[i * n] = s;
    }
    for (int i = m - 1; i >= 0; --i)
    {
        ofchem s = b[i * n];
        for (int j = i + 1; j < m; ++j)
            s = __dsub_rn(s, __dmul_rn(a[(i * m + j) * n], b[j * n]));
        b[i * n] = s / a[(i * m + i) * n];
    }
}
//- chemIntegrate: one thread per cell advances the host twin
//  ConstPressureReactor::advance (SPEC-LIT 132.4) over one flow step dt -
//  rosenbrock.rs's integrate_observed with its ros_step, lu_factor,
//  lu_solve, fd_jacobian and error_norm, and advance's observer (R10's
//  clip / refusal) - in the host's operation order. Every multiply, add,
//  subtract and divide of the integrator itself goes through the
//  round-to-nearest intrinsics so nvcc cannot contract it into an FMA the
//  host's uncontracted arithmetic would disagree with; chemRhs keeps its
//  frozen results. state[i*n+c] holds
//  (Y_1..Y_K, T) in place; p[c]; h[c] in = the first trial step (advance's
//  h0), out = the step proposed after the last accepted step
//  (RosStats::h_next); counts[0*n+c] accepted, [1*n+c] rejected,
//  [2*n+c] clipped. Status, in the input checks' order: 1 p not finite or
//  not above 0 (R7), 3 a mass fraction negative or not finite (R9),
//  2 T not finite or not above 0 (R9), 9 h not finite or not above 0 (R5's
//  h0); then during the run any f evaluation chemRhs refuses ends the cell
//  with that status (2, 4, 5), 6 a singular I - gamma h J, 7 more step
//  attempts than max_steps, 8 a Y below -atol_y at the observer. A refused
//  cell writes NaN to its whole state column and to h[c]; its counts hold
//  what stood at the refusal. -
extern "C" __global__ void chemIntegrate
(
    ofchem* __restrict__ state,
    const ofchem* __restrict__ p,
    ofchem* __restrict__ h,
    int* __restrict__ status,
    int* __restrict__ counts,
    const ofchem* __restrict__ ctrl,
    ofchem* __restrict__ vecs,
    ofchem* __restrict__ jac,
    ofchem* __restrict__ lu,
    int* __restrict__ piv,
    ofchem* __restrict__ conc,
    ofchem* __restrict__ g,
    ofchem* __restrict__ omg,
    ofchem* __restrict__ yp,
    ofchem* __restrict__ ypert,
    ofchem* __restrict__ fpert,
    const ofchem* __restrict__ mf,
    const int* __restrict__ mi,
    const ofchem dt,
    oflabel n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;
    const int K = mi[0];
    const int m = K + 1;
    //- The control array, in the layout the header states. -
    const ofchem gamma = ctrl[0];
    const ofchem* alpha = ctrl + 1;
    const ofchem* gij = ctrl + 17;
    const ofchem* b = ctrl + 33;
    const ofchem* bhat = ctrl + 37;
    const int stages = (int)ctrl[41];
    const int emb_order = (int)ctrl[42];
    const ofchem rtol = ctrl[43];
    const ofchem atol_y = ctrl[44];
    const ofchem atol_t = ctrl[45];
    const ofchem h_max = ctrl[46];
    const int max_steps = (int)ctrl[47];
    //- The cell's columns of the ten SoA work vectors: f0, y_new, est,
    //  arg, rhs, v, then the four stage vectors; element l of each is at
    //  [l * n]. -
    ofchem* f0 = vecs + c;
    ofchem* ynew = vecs + m * n + c;
    ofchem* est = vecs + 2 * m * n + c;
    ofchem* arg = vecs + 3 * m * n + c;
    ofchem* rhs = vecs + 4 * m * n + c;
    ofchem* v = vecs + 5 * m * n + c;
    ofchem* kj[4];
    for (int i = 0; i < 4; ++i) kj[i] = vecs + (6 + i) * m * n + c;

    int st = 0;
    int accepted = 0;
    int rejected = 0;
    int clipped = 0;
    const ofchem p_c = p[c];
    //- The input checks, in advance's and integrate_observed's order: the
    //  pressure (R7), the mass fractions (R9), T (R9), the first trial
    //  step (R5's h0). -
    if (!isfinite(p_c) || p_c <= 0.0) st = 1;
    for (int k = 0; k < K && st == 0; ++k)
    {
        const ofchem yk = state[k * n + c];
        if (!isfinite(yk) || yk < 0.0) st = 3;
    }
    if (st == 0)
    {
        const ofchem t_c = state[K * n + c];
        if (!isfinite(t_c) || t_c <= 0.0) st = 2;
    }
    ofchem hstep = h[c];
    if (st == 0 && (!isfinite(hstep) || hstep <= 0.0)) st = 9;

    if (st == 0)
    {
        //- R3: h = min(min(h_in, h_max), dt); advance's h_min is 0. -
        hstep = fmin(fmin(hstep, h_max), dt);
        ofchem fac_max = 10.0;
        int accepted_any = 0;
        int have_jac = 0;
        int attempts = 0;
        const ofchem inv_p1 = 1.0 / ((ofchem)emb_order + 1.0);
        ofchem t_run = 0.0;
        while (t_run < dt && st == 0)
        {
            //- R3: the end point; the shortened last step. -
            const int last = __dadd_rn(t_run, hstep) >= dt;
            if (last) hstep = __dsub_rn(dt, t_run);
            //- R2: f0 and the FD Jacobian at y, once per accepted point; a
            //  retry after a rejection reuses it. Any chemRhs refusal ends
            //  the cell with that status, as the host's ? does. -
            if (!have_jac)
            {
                st = chemRhs(state + c, p_c, mf, mi, yp + c, conc + c, g + c,
                             omg + c, f0, n);
                for (int j = 0; j < m && st == 0; ++j)
                {
                    const ofchem y_j = state[j * n + c];
                    const ofchem atol_j = (j < K) ? atol_y : atol_t;
                    const ofchem delta = __dmul_rn(1.4901161193847656e-08,
                                                   fmax(fabs(y_j), atol_j));
                    for (int i = 0; i < m; ++i)
                        ypert[i * n + c] = state[i * n + c];
                    ypert[j * n + c] = __dadd_rn(y_j, delta);
                    st = chemRhs(ypert + c, p_c, mf, mi, yp + c, conc + c,
                                 g + c, omg + c, fpert + c, n);
                    for (int i = 0; i < m && st == 0; ++i)
                        jac[(i * m + j) * n + c] = __ddiv_rn(
                            __dsub_rn(fpert[i * n + c], f0[i * n]), delta);
                }
                if (st != 0) break;
                have_jac = 1;
            }
            attempts++;
            if (attempts > max_steps) { st = 7; break; }
            //- The matrix lu = (-gamma h) J, then the +1 diagonal
            //  (rosenbrock.rs's ros_step). -
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < m; ++j)
                    lu[(i * m + j) * n + c] =
                        __dmul_rn(__dmul_rn(-gamma, hstep),
                                  jac[(i * m + j) * n + c]);
                lu[(i * m + i) * n + c] =
                    __dadd_rn(lu[(i * m + i) * n + c], 1.0);
            }
            if (chemLuFactor(lu + c, piv + c, m, n)) { st = 6; break; }
            //- The stages of (132.1): the argument y + sum alpha_ij k_j,
            //  f there, then rhs = h f + h J v with v = sum gamma_ij k_j,
            //  then the solve that gives k_i. -
            for (int i = 0; i < stages && st == 0; ++i)
            {
                for (int l = 0; l < m; ++l) arg[l * n] = state[l * n + c];
                for (int j = 0; j < i; ++j)
                {
                    const ofchem a_ij = alpha[i * 4 + j];
                    if (a_ij == 0.0) continue;
                    for (int l = 0; l < m; ++l)
                        arg[l * n] = __dadd_rn(arg[l * n],
                            __dmul_rn(a_ij, kj[j][l * n]));
                }
                st = chemRhs(arg, p_c, mf, mi, yp + c, conc + c, g + c,
                             omg + c, rhs, n);
                if (st != 0) break;
                for (int l = 0; l < m; ++l) v[l * n] = 0.0;
                for (int j = 0; j < i; ++j)
                {
                    const ofchem g_ij = gij[i * 4 + j];
                    if (g_ij == 0.0) continue;
                    for (int l = 0; l < m; ++l)
                        v[l * n] = __dadd_rn(v[l * n],
                            __dmul_rn(g_ij, kj[j][l * n]));
                }
                for (int l = 0; l < m; ++l)
                {
                    ofchem jv = 0.0;
                    for (int q = 0; q < m; ++q)
                        jv = __dadd_rn(jv, __dmul_rn(jac[(l * m + q) * n + c],
                                                     v[q * n]));
                    rhs[l * n] = __dadd_rn(__dmul_rn(hstep, rhs[l * n]),
                                           __dmul_rn(hstep, jv));
                }
                chemLuSolve(lu + c, piv + c, m, rhs, n);
                for (int l = 0; l < m; ++l) kj[i][l * n] = rhs[l * n];
            }
            if (st != 0) break;
            //- (132.2): y_new = y + sum b_i k_i, est = sum (b_i - bhat_i)
            //  k_i, both sums in i order, one loop. -
            for (int l = 0; l < m; ++l)
            {
                ofchem sy = 0.0;
                ofchem se = 0.0;
                for (int i = 0; i < stages; ++i)
                {
                    sy = __dadd_rn(sy, __dmul_rn(b[i], kj[i][l * n]));
                    se = __dadd_rn(se, __dmul_rn(__dsub_rn(b[i], bhat[i]),
                                                 kj[i][l * n]));
                }
                ynew[l * n] = __dadd_rn(state[l * n + c], sy);
                est[l * n] = se;
            }
            //- (132.5): Err = sqrt(sum (est/(atol + rtol|y_new|))^2 / m),
            //  m = K + 1. -
            ofchem err_sum = 0.0;
            int finite = 1;
            for (int l = 0; l < m; ++l)
            {
                const ofchem y_l = ynew[l * n];
                if (!isfinite(y_l)) finite = 0;
                const ofchem atol_l = (l < K) ? atol_y : atol_t;
                const ofchem e = __ddiv_rn(est[l * n],
                    __dadd_rn(atol_l, __dmul_rn(rtol, fabs(y_l))));
                err_sum = __dadd_rn(err_sum, __dmul_rn(e, e));
            }
            const ofchem err = sqrt(__ddiv_rn(err_sum, (ofchem)m));
            if (!isfinite(err)) finite = 0;
            const ofchem fac = fmin(fac_max,
                fmax(0.1, __ddiv_rn(0.9, pow(err, inv_p1))));
            if (finite && err < 1.0)
            {
                //- Accept; on the last step t is dt exactly (R3). -
                t_run = last ? dt : __dadd_rn(t_run, hstep);
                for (int l = 0; l < m; ++l) state[l * n + c] = ynew[l * n];
                accepted++;
                accepted_any = 1;
                fac_max = 10.0;
                hstep = fmin(__dmul_rn(hstep, fac), h_max);
                have_jac = 0;
                //- advance's observer (R10): below -atol_y refuses, in
                //  [-atol_y, 0) clips to 0, in species order. -
                for (int l = 0; l < K; ++l)
                {
                    if (state[l * n + c] < -atol_y) { st = 8; break; }
                    if (state[l * n + c] < 0.0)
                    {
                        state[l * n + c] = 0.0;
                        clipped++;
                    }
                }
            }
            else
            {
                //- Reject; R4's non-finite step always divides h by 10. -
                rejected++;
                fac_max = 1.0;
                if (!finite) hstep = __ddiv_rn(hstep, 10.0);
                else if (accepted_any) hstep = __dmul_rn(hstep, fac);
                else hstep = __ddiv_rn(hstep, 10.0);
            }
        }
    }
    if (st != 0)
    {
        for (int l = 0; l < m; ++l) state[l * n + c] = nan("");
        h[c] = nan("");
    }
    else
    {
        h[c] = hstep;
    }
    counts[c] = accepted;
    counts[n + c] = rejected;
    counts[2 * n + c] = clipped;
    status[c] = st;
}
//- chemPackState: the ofscalar crossing of integrate_fields (SPEC-LIT
//  132.6) - t[c], y[k*n+c], p[c] widened exactly into the f64 state and
//  pressure the chemIntegrate launch runs on. -
extern "C" __global__ void chemPackState
(
    const ofscalar* __restrict__ t,
    const ofscalar* __restrict__ p,
    const ofscalar* __restrict__ y,
    ofchem* __restrict__ state,
    ofchem* __restrict__ pv,
    const int* __restrict__ mi,
    oflabel n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;
    const int K = mi[0];
    pv[c] = (ofchem)p[c];
    state[K * n + c] = (ofchem)t[c];
    for (int k = 0; k < K; ++k) state[k * n + c] = (ofchem)y[k * n + c];
}
//- chemUnpackState: the integrated state rounded once into t and y. -
extern "C" __global__ void chemUnpackState
(
    const ofchem* __restrict__ state,
    ofscalar* __restrict__ t,
    ofscalar* __restrict__ y,
    const int* __restrict__ mi,
    oflabel n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;
    const int K = mi[0];
    t[c] = (ofscalar)state[K * n + c];
    for (int k = 0; k < K; ++k) y[k * n + c] = (ofscalar)state[k * n + c];
}

/*
  ------------------------------------------------------------------------
  The flat transport array of the three transport kernels (SPEC-LIT
  133.4), uploaded once per DeviceTransport. The Rust twin of this layout
  is flatten_transport in src/chem/device.rs; the two field lists must
  stay identical. 40 doubles per species k at 40k:
    [0]      W (kg/mol)
    [1]      n_v (1..3, as a double)
    [2..20)  three viscosity fits of 6 doubles (t_lo, t_hi, a, b, c, d),
             the unused slots 0.0
    [20]     n_c
    [21..39) three conductivity fits of 6 doubles, the unused slots 0.0
    [39]     0.0
  The species count K is not in the array; the launches carry it as an
  argument, like the array length they already know.
  ------------------------------------------------------------------------
*/

//- TM-4513 eq. (10) on one fit f: ln q = a ln T + b/T + c/T^2 + d, in
//  transport.rs TransportFit::ln_value's order, left to right. -
OFGPU_DEV ofchem chemFitLn(const ofchem* f, ofchem t)
{
    return f[2] * log(t) + f[3] / t + f[4] / (t * t) + f[5];
}

//- The lowest accepted T, K: transport.rs's TRANSPORT_FLOOR_K (SPEC-LIT
//  133.7 T1, the user's decision of 2026-10-14). -
#define CHEM_TRANSPORT_FLOOR_K 200.0

//- The FIRST fit with t_lo <= T <= t_hi; else fit 0, extrapolated, when
//  CHEM_TRANSPORT_FLOOR_K <= T < fit 0's t_lo; else -1 - transport.rs's
//  interval choice. n is the fit count (the fits ascend, contiguous). -
OFGPU_DEV int chemFitPick(const ofchem* fits, int n, ofchem t)
{
    for (int i = 0; i < n; ++i)
    {
        const ofchem* f = fits + 6 * i;
        if (f[0] <= t && t <= f[1]) return i;
    }
    if ((ofchem)CHEM_TRANSPORT_FLOOR_K <= t && t < fits[0]) return 0;
    return -1;
}

//- chemMixtureTransport: the Wilke mixture viscosity (133.8) with (133.10)
//  and the mixture conductivity (133.9) with (133.10) and (133.11), from
//  the TM-4513 eq. (10) fits of route (a) - transport.rs's mixture_ty of
//  SPEC-LIT 133.4, choice for choice and in the host's operation order.
//  Status, in the host's order of checks: 2 T not finite or not above 0,
//  3 a mass fraction negative or not finite, 4 no species present, 10 a
//  present species' T outside its viscosity or conductivity accepted
//  range (chemFitPick, 133.7 T1). A
//  refused cell writes NaN to its mu and lambda. The sums run over the
//  present species (y != 0) in index order exactly as the host: den = x_i,
//  then j != i with y_j != 0, den += x_j phi_ij, then acc += x_i mu_i/den;
//  the conductivity's psi (133.11) rides on the same viscosity phi
//  (133.10), as the host calls it. -
extern "C" __global__ void chemMixtureTransport
(
    const ofscalar* __restrict__ t,
    const ofscalar* __restrict__ y,
    ofscalar* __restrict__ mu,
    ofscalar* __restrict__ lam,
    int* __restrict__ status,
    ofchem* __restrict__ x,
    ofchem* __restrict__ muv,
    ofchem* __restrict__ lamv,
    const ofchem* __restrict__ tf,
    const int K,
    oflabel n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;
    int st = 0;
    const ofscalar t_c = t[c];
    if (!isfinite(t_c) || t_c <= (ofscalar)0) st = 2;
    if (st == 0)
    {
        for (int k = 0; k < K; ++k)
        {
            const ofscalar v = y[k * n + c];
            if (!isfinite(v) || v < (ofscalar)0) { st = 3; break; }
        }
    }
    ofchem s = 0.0;
    if (st == 0)
    {
        for (int k = 0; k < K; ++k) s += (ofchem)y[k * n + c] / tf[40 * k];
        if (isnan(s) || s <= 0.0) st = 4;
    }
    if (st == 0)
    {
        //- mole_fractions (x_k = y_k (1/s) / W_k), and the pure-species
        //  values of every present species, whose eq. (10) range is
        //  checked on the FIRST fitting interval -
        const ofchem t_d = (ofchem)t_c;
        const ofchem w = 1.0 / s;
        for (int k = 0; k < K; ++k)
        {
            const ofchem* rec = tf + 40 * k;
            x[k * n + c] = (ofchem)y[k * n + c] * w / rec[0];
            if ((ofchem)y[k * n + c] == 0.0) continue;
            const int iv = chemFitPick(rec + 2, (int)rec[1], t_d);
            const int ic = chemFitPick(rec + 21, (int)rec[20], t_d);
            if (iv < 0 || ic < 0) { st = 10; break; }
            muv[k * n + c] = exp(chemFitLn(rec + 2 + 6 * iv, t_d)) * 1.0e-7;
            lamv[k * n + c] = exp(chemFitLn(rec + 21 + 6 * ic, t_d)) * 1.0e-4;
        }
    }
    if (st == 0)
    {
        //- (133.8): mu_mix = sum_i x_i mu_i / (x_i + sum_{j!=i} x_j phi_ij) -
        ofchem acc = 0.0;
        for (int i = 0; i < K; ++i)
        {
            if ((ofchem)y[i * n + c] == 0.0) continue;
            const ofchem xi = x[i * n + c];
            const ofchem mui = muv[i * n + c];
            const ofchem wi = tf[40 * i];
            ofchem den = xi;
            for (int j = 0; j < K; ++j)
            {
                if (j == i || (ofchem)y[j * n + c] == 0.0) continue;
                const ofchem wj = tf[40 * j];
                const ofchem r = sqrt(mui / muv[j * n + c]) * pow(wj / wi, 0.25);
                const ofchem a = 1.0 + r;
                const ofchem phi = 0.25 * a * a * sqrt(2.0 * wj / (wi + wj));
                den += x[j * n + c] * phi;
            }
            acc += xi * mui / den;
        }
        mu[c] = (ofscalar)acc;
    }
    if (st == 0)
    {
        //- (133.9) with (133.11): psi_ij = phi_ij [1 + 2.41 (W_i-W_j)
        //  (W_i-0.142 W_j)/(W_i+W_j)^2] on the same viscosity phi -
        ofchem acc = 0.0;
        for (int i = 0; i < K; ++i)
        {
            if ((ofchem)y[i * n + c] == 0.0) continue;
            const ofchem xi = x[i * n + c];
            const ofchem mui = muv[i * n + c];
            const ofchem lami = lamv[i * n + c];
            const ofchem wi = tf[40 * i];
            ofchem den = xi;
            for (int j = 0; j < K; ++j)
            {
                if (j == i || (ofchem)y[j * n + c] == 0.0) continue;
                const ofchem wj = tf[40 * j];
                const ofchem r = sqrt(mui / muv[j * n + c]) * pow(wj / wi, 0.25);
                const ofchem a = 1.0 + r;
                const ofchem phi = 0.25 * a * a * sqrt(2.0 * wj / (wi + wj));
                const ofchem m = (wi - wj) * (wi - 0.142 * wj)
                    / ((wi + wj) * (wi + wj));
                den += x[j * n + c] * (phi * (1.0 + 2.41 * m));
            }
            acc += xi * lami / den;
        }
        lam[c] = (ofscalar)acc;
    }
    if (st != 0)
    {
        mu[c] = (ofscalar)nan("");
        lam[c] = (ofscalar)nan("");
    }
    status[c] = st;
}

//- chemPrandtlConductivity: the Lewis route's lambda = mu cp / Pr (SPEC-LIT
//  133.4), (mu * cp) / pr exactly as the host computes it, in double in
//  both builds and rounded once on the way out. -
extern "C" __global__ void chemPrandtlConductivity
(
    const ofscalar* __restrict__ mu,
    const ofscalar* __restrict__ cp,
    ofscalar* __restrict__ lam,
    const ofchem pr,
    oflabel n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;
    lam[c] = (ofscalar)(((ofchem)mu[c] * (ofchem)cp[c]) / pr);
}

//- chemLewisDiffusivity: the Lewis route's D_m = lambda / (rho cp Le_m)
//  (SPEC-LIT 133.4), lambda / ((rho * cp) * le_k) exactly as the host
//  computes it, in double in both builds, rounded once on the way out. -
extern "C" __global__ void chemLewisDiffusivity
(
    const ofscalar* __restrict__ lam,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ cp,
    ofscalar* __restrict__ d,
    const ofchem* __restrict__ le,
    const int K,
    oflabel n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;
    for (int k = 0; k < K; ++k)
    {
        d[k * n + c] = (ofscalar)((ofchem)lam[c]
            / (((ofchem)rho[c] * (ofchem)cp[c]) * le[k]));
    }
}

//- chemEdmSource: the EDM source of SPEC-LIT 134.6 - (134.2)'s rates of
//  progress q_r = ((A r) rho) lim_r, lim_r the reactant minimum in list
//  order and then B's product-limiting argument, and the per-species
//  accumulation W_m acc_m - exactly the host twin
//  src/chem/tci.rs EdmMechanism::progress/source in the same order.
//  arm: 0 Epsilon (m0 = k, m1 = epsilon, r = eps/k), 1 Omega (m0 = omega,
//  r = beta_star*omega). The flat layout, whose Rust twin is
//  EdmMechanism::flatten (the two field lists must stay identical):
//    ei: [0] K, [1] R (1 or 2), then 4 ints per reaction j at 2+4j -
//        re_begin, re_end, pr_begin, pr_end (half-open entry indices, the
//        reactant entries of j then its product entries, reactions in
//        order), then at 2+4R one species index per entry.
//    ef: [0..K) W_k (kg/mol), then at K+e the coefficient (nu' or nu'')
//        of entry e.
//  Statuses: 3 a non-finite Y, 12 rho not finite or <= 0, 11 a rate that
//  is not finite or is negative; a refused cell writes NaN to every wdot. -
extern "C" __global__ void chemEdmSource
(
    const int arm,
    const ofscalar* __restrict__ m0,
    const ofscalar* __restrict__ m1,
    const double beta_star,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ y,
    ofscalar* __restrict__ wdot,
    int* __restrict__ status,
    const double* __restrict__ ef,
    const int* __restrict__ ei,
    const double a,
    const double b,
    const int has_b,
    const int n
)
{
    const int c = OFGPU_TID;
    if (c >= n) return;
    const int K = ei[0];
    const int R = ei[1];
    int st = 0;
    for (int k = 0; k < K; ++k)
    {
        if (!isfinite((ofchem)y[k * n + c])) { st = 3; break; }
    }
    ofchem rho_c = 0.0;
    if (st == 0)
    {
        rho_c = (ofchem)rho[c];
        if (!isfinite(rho_c) || rho_c <= 0.0) st = 12;
    }
    ofchem q[2];
    if (st == 0)
    {
        ofchem r = 0.0;
        if (arm == 0) r = (ofchem)m1[c] / (ofchem)m0[c];
        else r = beta_star * (ofchem)m0[c];
        if (!isfinite(r) || r < 0.0) st = 11;
        for (int j = 0; j < R && st == 0; ++j)
        {
            const int re0 = ei[2 + 4 * j];
            const int re1 = ei[3 + 4 * j];
            const int pr0 = ei[4 + 4 * j];
            const int pr1 = ei[5 + 4 * j];
            const int s0 = ei[2 + 4 * R + re0];
            ofchem lim = (ofchem)y[s0 * n + c] / (ef[K + re0] * ef[s0]);
            for (int e = re0 + 1; e < re1; ++e)
            {
                const int s = ei[2 + 4 * R + e];
                lim = fmin(lim, (ofchem)y[s * n + c] / (ef[K + e] * ef[s]));
            }
            if (has_b != 0)
            {
                ofchem sy = 0.0;
                ofchem sw = 0.0;
                for (int e = pr0; e < pr1; ++e)
                {
                    const int s = ei[2 + 4 * R + e];
                    sy += (ofchem)y[s * n + c];
                    sw += ef[K + e] * ef[s];
                }
                lim = fmin(lim, b * sy / sw);
            }
            q[j] = ((a * r) * rho_c) * lim;
        }
    }
    if (st == 0)
    {
        for (int m = 0; m < K; ++m)
        {
            ofchem acc = 0.0;
            for (int j = 0; j < R; ++j)
            {
                for (int e = ei[2 + 4 * j]; e < ei[3 + 4 * j]; ++e)
                {
                    if (ei[2 + 4 * R + e] == m) acc -= ef[K + e] * q[j];
                }
                for (int e = ei[4 + 4 * j]; e < ei[5 + 4 * j]; ++e)
                {
                    if (ei[2 + 4 * R + e] == m) acc += ef[K + e] * q[j];
                }
            }
            wdot[m * n + c] = (ofscalar)(ef[m] * acc);
        }
    }
    if (st != 0)
    {
        for (int m = 0; m < K; ++m) wdot[m * n + c] = (ofscalar)nan("");
    }
    status[c] = st;
}

//- chemPasrScale: PaSR's frozen-mean rate of SPEC-LIT 134.6,
//  W_i omega-bar_i = kappa (W_i omega_i) - (134.14)'s mixing time
//  (tau_mix 0 integral c_mix tau_i, 1 kolmogorov sqrt(nu/eps),
//  2 geometric sqrt(tau_i sqrt(nu/eps)), 3 strain 1/|S|) and (134.15)'s
//  edges, tau_c from Piscopo eqs. (5)-(7) over the species with
//  |ydot| >= 1e-8, exactly the host twin src/chem/tci.rs's
//  tau_mix/tau_chem/kappa/pasr_source in the same order. arm: 0 Epsilon
//  (m0 = k, m1 = epsilon), 1 Omega (m0 = omega, kf = k), 2 Strain
//  (m0 = |S|). Statuses: 3 a non-finite Y, 12 rho not finite or <= 0,
//  11 a tau_m that is NaN or negative (tau_m = +inf is allowed and gives
//  kappa 0), 13 a non-finite omega; a refused cell writes NaN to every
//  wdot. -
extern "C" __global__ void chemPasrScale
(
    const int arm,
    const ofscalar* __restrict__ m0,
    const ofscalar* __restrict__ m1,
    const ofscalar* __restrict__ kf,
    const double beta_star,
    const ofscalar* __restrict__ nu,
    const int tau_mix,
    const double c_mix,
    const int tau_chem,
    const double* __restrict__ w,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ y,
    const ofscalar* __restrict__ omega,
    ofscalar* __restrict__ wdot,
    int* __restrict__ status,
    const int K,
    const int n
)
{
    const int c = OFGPU_TID;
    if (c >= n) return;
    int st = 0;
    for (int k = 0; k < K; ++k)
    {
        if (!isfinite((ofchem)y[k * n + c])) { st = 3; break; }
    }
    ofchem rho_c = 0.0;
    if (st == 0)
    {
        rho_c = (ofchem)rho[c];
        if (!isfinite(rho_c) || rho_c <= 0.0) st = 12;
    }
    ofchem tau_m = 0.0;
    if (st == 0)
    {
        //- (134.14) by arm: tau_i and eps, then the named mixing time -
        ofchem tau_i = 0.0;
        ofchem eps = 0.0;
        if (arm == 0)
        {
            tau_i = (ofchem)m0[c] / (ofchem)m1[c];
            eps = (ofchem)m1[c];
        }
        else if (arm == 1)
        {
            tau_i = 1.0 / (beta_star * (ofchem)m0[c]);
            eps = (beta_star * (ofchem)kf[c]) * (ofchem)m0[c];
        }
        if (tau_mix == 0) tau_m = c_mix * tau_i;
        else if (tau_mix == 1) tau_m = sqrt((ofchem)nu[c] / eps);
        else if (tau_mix == 2) tau_m = sqrt(tau_i * sqrt((ofchem)nu[c] / eps));
        else tau_m = 1.0 / (ofchem)m0[c];
        if (isnan(tau_m) || tau_m < 0.0) st = 11;
    }
    if (st == 0)
    {
        for (int k = 0; k < K; ++k)
        {
            if (!isfinite((ofchem)omega[k * n + c])) { st = 13; break; }
        }
    }
    ofchem kap = 1.0;
    if (st == 0)
    {
        //- Piscopo eqs. (5)-(7): the chemical time from the ratios -
        ofchem sfr = 0.0;
        ofchem ffr = 0.0;
        int present = 0;
        for (int k = 0; k < K; ++k)
        {
            const ofchem wl = w[k] * (ofchem)omega[k * n + c];
            const ofchem yd = wl / rho_c;
            if (fabs(yd) < 1.0e-8) continue;
            const ofchem ratio = (ofchem)y[k * n + c] / fabs(yd);
            if (present == 0)
            {
                sfr = ratio;
                ffr = ratio;
                present = 1;
            }
            else
            {
                if (ratio > sfr) sfr = ratio;
                if (ratio < ffr) ffr = ratio;
            }
        }
        //- (134.15) at its edges -
        if (tau_m == 0.0 || present == 0)
        {
            kap = 1.0;
        }
        else
        {
            const ofchem tau_c = tau_chem == 0 ? sfr
                : (tau_chem == 1 ? ffr : sqrt(sfr * ffr));
            kap = tau_c / (tau_c + tau_m);
        }
        for (int k = 0; k < K; ++k)
        {
            const ofchem wl = w[k] * (ofchem)omega[k * n + c];
            wdot[k * n + c] = (ofscalar)(kap * wl);
        }
    }
    if (st != 0)
    {
        for (int k = 0; k < K; ++k) wdot[k * n + c] = (ofscalar)nan("");
    }
    status[c] = st;
}

//- chemEdcRhs: the fine-structure reactor's right-hand side of SPEC-LIT
//  134.7 - (134.9)'s chemistry part from chemRhs at (Y*, T*), then the
//  mixing terms - (Y*_k - Ybar_k)/tau_e per species and (134.16)'s
//  - (sum Ybar_k h_k(T*) - hbar)/(cp(T*, Y*+) tau_e) - exactly the host
//  twin edc_host's rhs closure, in its order. hbar is
//  sum Ybar_k h_k(Tbar), computed once per cell by the caller. The
//  pointers are cell-column pointers, as chemRhs takes them (element k at
//  [k*n]). cp runs over yp != 0 as chemRhs's does, hs over Ybar != 0;
//  a chemRange miss returns its code (5). Returns chemRhs's code, or the
//  range miss, or 0. -
OFGPU_DEV int chemEdcRhs(const ofchem* y, ofchem p, const ofchem* ybar, ofchem hbar, ofchem te,
                         const ofchem* mf, const int* mi, ofchem* yp, ofchem* conc, ofchem* g,
                         ofchem* omg, ofchem* dydt, oflabel n)
{
    const int K = mi[0];
    const int st = chemRhs(y, p, mf, mi, yp, conc, g, omg, dydt, n);
    if (st != 0) return st;
    const ofchem rg = mf[18 * K];
    const ofchem t = y[K * n];
    for (int k = 0; k < K; ++k)
    {
        dydt[k * n] = __dsub_rn(dydt[k * n],
            __ddiv_rn(__dsub_rn(y[k * n], ybar[k * n]), te));
    }
    ofchem cp = 0.0;
    for (int k = 0; k < K; ++k)
    {
        if (yp[k * n] == 0.0) continue;
        const ofchem* sp = mf + 18 * k;
        const int miss = chemRange(sp, t);
        if (miss != 0) return miss;
        cp = __dadd_rn(cp, __ddiv_rn(
            __dmul_rn(yp[k * n], __dmul_rn(rg, chemCpR(chemCoeffs(sp, t), t))), mf[18 * k]));
    }
    ofchem hs = 0.0;
    for (int k = 0; k < K; ++k)
    {
        if (ybar[k * n] == 0.0) continue;
        const ofchem* sp = mf + 18 * k;
        const int miss = chemRange(sp, t);
        if (miss != 0) return miss;
        hs = __dadd_rn(hs, __ddiv_rn(
            __dmul_rn(ybar[k * n], __dmul_rn(__dmul_rn(rg, t), chemHrt(chemCoeffs(sp, t), t))),
            mf[18 * k]));
    }
    dydt[K * n] = __dsub_rn(dydt[K * n], __ddiv_rn(__dsub_rn(hs, hbar), __dmul_rn(cp, te)));
    return 0;
}

//- chemEdcIntegrate: EDC's device source of SPEC-LIT 134.7 - one thread
//  per cell integrates the fine-structure reactor, (134.9) with
//  (134.16)'s energy mixing term, from (Ybar, Tbar) over psrSpan tau_e by
//  a copy of chemIntegrate's step loop whose every chemRhs call is
//  chemEdcRhs and whose duration is span; then (134.17)'s residual in
//  (132.5)'s weights - a cell with r_res > 1, or a non-finite one, is
//  flagged UNCONVERGED and still written - and (134.10)'s mean rate
//  W_k omega-bar_k = ((rho gamma*) (Y*_k - Ybar_k)) / tau_e. gamma_lambda
//  capped at gamma_max flags CAPPED. gamma_star = 0 (eps = 0 or nu = 0)
//  means no fine structures: Y* = Ybar, T* = Tbar, W omega-bar = 0
//  exactly, nothing integrated, the carried step untouched. arm:
//  0 Epsilon (m0 = k, m1 = eps), 1 Omega (m0 = omega, kf = k,
//  eps = (beta_star k) omega); cube 1 = 1989 (gamma* = gl^3), 0 = 2005
//  (gl^2). Statuses, in the input checks' order: 1 p, 3 a Y negative or
//  not finite, 2 Tbar, 12 rho, 11 the mixing inputs (k <= 0, eps < 0,
//  nu < 0, omega < 0, or one of them not finite); then, only when
//  gamma* > 0, 5 a Tbar outside a record hbar needs, 9 a carried step
//  that is not finite or not above 0, then the run codes the copy
//  carries (2, 4, 5 from chemEdcRhs, 6 a singular I - gamma h J,
//  7 max_steps, 8 the observer's Y below -atol_y). A refused cell writes
//  NaN to its wdot and its fine state, leaves its carried step, and has
//  flags 0; on success h[c] holds the proposed step, as chemIntegrate's
//  does. -
extern "C" __global__ void chemEdcIntegrate
(
    const int arm,
    const ofscalar* __restrict__ m0,
    const ofscalar* __restrict__ m1,
    const ofscalar* __restrict__ kf,
    const double beta_star,
    const ofscalar* __restrict__ nu,
    const int cube,
    const double c_gamma,
    const double c_tau,
    const double gamma_max,
    const double psr_span,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ t,
    const ofscalar* __restrict__ p,
    const ofscalar* __restrict__ y,
    ofscalar* __restrict__ wdot,
    int* __restrict__ status,
    int* __restrict__ flags,
    ofchem* __restrict__ state,
    ofchem* __restrict__ ybar,
    ofchem* __restrict__ h,
    int* __restrict__ counts,
    const ofchem* __restrict__ ctrl,
    ofchem* __restrict__ vecs,
    ofchem* __restrict__ jac,
    ofchem* __restrict__ lu,
    int* __restrict__ piv,
    ofchem* __restrict__ conc,
    ofchem* __restrict__ g,
    ofchem* __restrict__ omg,
    ofchem* __restrict__ yp,
    ofchem* __restrict__ ypert,
    ofchem* __restrict__ fpert,
    const ofchem* __restrict__ mf,
    const int* __restrict__ mi,
    oflabel n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;
    const int K = mi[0];
    const int m = K + 1;
    //- The control array, in the layout the header states. -
    const ofchem gamma = ctrl[0];
    const ofchem* alpha = ctrl + 1;
    const ofchem* gij = ctrl + 17;
    const ofchem* b = ctrl + 33;
    const ofchem* bhat = ctrl + 37;
    const int stages = (int)ctrl[41];
    const int emb_order = (int)ctrl[42];
    const ofchem rtol = ctrl[43];
    const ofchem atol_y = ctrl[44];
    const ofchem atol_t = ctrl[45];
    const ofchem h_max = ctrl[46];
    const int max_steps = (int)ctrl[47];
    //- The cell's columns of the ten SoA work vectors: f0, y_new, est,
    //  arg, rhs, v, then the four stage vectors; element l at [l * n]. -
    ofchem* f0 = vecs + c;
    ofchem* ynew = vecs + m * n + c;
    ofchem* est = vecs + 2 * m * n + c;
    ofchem* arg = vecs + 3 * m * n + c;
    ofchem* rhs = vecs + 4 * m * n + c;
    ofchem* v = vecs + 5 * m * n + c;
    ofchem* kj[4];
    for (int i = 0; i < 4; ++i) kj[i] = vecs + (6 + i) * m * n + c;

    int st = 0;
    int accepted = 0;
    int rejected = 0;
    int clipped = 0;
    int fl = 0;
    //- Widen the inputs and write ybar and the fine state's start
    //  (Ybar, Tbar) first; the refused cell overwrites it with NaN. -
    for (int k = 0; k < K; ++k)
    {
        const ofchem yk = (ofchem)y[k * n + c];
        ybar[k * n + c] = yk;
        state[k * n + c] = yk;
    }
    const ofchem t_d = (ofchem)t[c];
    state[K * n + c] = t_d;
    const ofchem p_c = (ofchem)p[c];
    const ofchem rho_c = (ofchem)rho[c];
    const ofchem nu_c = (ofchem)nu[c];
    //- The input checks, in 134.7's status order. -
    if (!isfinite(p_c) || p_c <= 0.0) st = 1;
    for (int k = 0; k < K && st == 0; ++k)
    {
        if (!isfinite(state[k * n + c]) || state[k * n + c] < 0.0) st = 3;
    }
    if (st == 0 && (!isfinite(t_d) || t_d <= 0.0)) st = 2;
    if (st == 0 && (!isfinite(rho_c) || rho_c <= 0.0)) st = 12;
    ofchem kk = 0.0;
    ofchem eps = 0.0;
    if (st == 0)
    {
        if (arm == 0)
        {
            kk = (ofchem)m0[c];
            eps = (ofchem)m1[c];
            if (!isfinite(kk) || kk <= 0.0 || !isfinite(eps) || eps < 0.0) st = 11;
        }
        else
        {
            const ofchem om = (ofchem)m0[c];
            kk = (ofchem)kf[c];
            eps = (beta_star * kk) * om;
            if (!isfinite(kk) || kk <= 0.0 || !isfinite(eps) || eps < 0.0
                || !isfinite(om) || om < 0.0) st = 11;
        }
        if (st == 0 && (!isfinite(nu_c) || nu_c < 0.0)) st = 11;
    }
    ofchem te = 0.0;
    ofchem span = 0.0;
    ofchem gs = 0.0;
    ofchem hbar = 0.0;
    ofchem hstep = 0.0;
    if (st == 0)
    {
        //- The scales of 134.7's order: gamma_lambda, the cap, gamma*,
        //  tau_e and the span. -
        ofchem gl = __dmul_rn(c_gamma,
            pow(__ddiv_rn(__dmul_rn(nu_c, eps), __dmul_rn(kk, kk)), 0.25));
        if (gl > gamma_max) { gl = gamma_max; fl |= 1; }
        gs = (cube != 0) ? __dmul_rn(__dmul_rn(gl, gl), gl) : __dmul_rn(gl, gl);
        te = __dmul_rn(__dsub_rn(1.0, gs), __dmul_rn(c_tau, sqrt(__ddiv_rn(nu_c, eps))));
        span = __dmul_rn(psr_span, te);
        if (gs == 0.0)
        {
            //- No fine structures: Y* = Ybar, T* = Tbar (already
            //  written), W omega-bar = 0 exactly, nothing integrated. -
            for (int k = 0; k < K; ++k) wdot[k * n + c] = (ofscalar)0.0;
            counts[c] = 0;
            counts[n + c] = 0;
            counts[2 * n + c] = 0;
            flags[c] = 0;
            status[c] = 0;
            return;
        }
        //- hbar = sum Ybar_k h_k(Tbar) over Ybar != 0; a range miss is 5. -
        const ofchem rg = mf[18 * K];
        for (int k = 0; k < K; ++k)
        {
            const ofchem yb = ybar[k * n + c];
            if (yb == 0.0) continue;
            const ofchem* sp = mf + 18 * k;
            const int miss = chemRange(sp, t_d);
            if (miss != 0) { st = miss; break; }
            hbar = __dadd_rn(hbar, __ddiv_rn(__dmul_rn(yb,
                __dmul_rn(__dmul_rn(rg, t_d), chemHrt(chemCoeffs(sp, t_d), t_d))),
                mf[18 * k]));
        }
    }
    if (st == 0)
    {
        hstep = h[c];
        if (!isfinite(hstep) || hstep <= 0.0) st = 9;
    }
    if (st == 0)
    {
        //- R3: h = min(min(h_in, h_max), span); advance's h_min is 0. -
        hstep = fmin(fmin(hstep, h_max), span);
        ofchem fac_max = 10.0;
        int accepted_any = 0;
        int have_jac = 0;
        int attempts = 0;
        const ofchem inv_p1 = 1.0 / ((ofchem)emb_order + 1.0);
        ofchem t_run = 0.0;
        while (t_run < span && st == 0)
        {
            //- R3: the end point; the shortened last step. -
            const int last = __dadd_rn(t_run, hstep) >= span;
            if (last) hstep = __dsub_rn(span, t_run);
            //- R2: f0 and the FD Jacobian at y, once per accepted point; a
            //  retry after a rejection reuses it. Any chemEdcRhs refusal
            //  ends the cell with that status, as the host's ? does. -
            if (!have_jac)
            {
                st = chemEdcRhs(state + c, p_c, ybar + c, hbar, te, mf, mi, yp + c,
                                conc + c, g + c, omg + c, f0, n);
                for (int j = 0; j < m && st == 0; ++j)
                {
                    const ofchem y_j = state[j * n + c];
                    const ofchem atol_j = (j < K) ? atol_y : atol_t;
                    const ofchem delta = __dmul_rn(1.4901161193847656e-08,
                                                   fmax(fabs(y_j), atol_j));
                    for (int i = 0; i < m; ++i) ypert[i * n + c] = state[i * n + c];
                    ypert[j * n + c] = __dadd_rn(y_j, delta);
                    st = chemEdcRhs(ypert + c, p_c, ybar + c, hbar, te, mf, mi, yp + c,
                                    conc + c, g + c, omg + c, fpert + c, n);
                    for (int i = 0; i < m && st == 0; ++i)
                        jac[(i * m + j) * n + c] = __ddiv_rn(
                            __dsub_rn(fpert[i * n + c], f0[i * n]), delta);
                }
                if (st != 0) break;
                have_jac = 1;
            }
            attempts++;
            if (attempts > max_steps) { st = 7; break; }
            //- The matrix lu = (-gamma h) J, then the +1 diagonal
            //  (rosenbrock.rs's ros_step). -
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < m; ++j)
                    lu[(i * m + j) * n + c] =
                        __dmul_rn(__dmul_rn(-gamma, hstep),
                                  jac[(i * m + j) * n + c]);
                lu[(i * m + i) * n + c] =
                    __dadd_rn(lu[(i * m + i) * n + c], 1.0);
            }
            if (chemLuFactor(lu + c, piv + c, m, n)) { st = 6; break; }
            //- The stages of (132.1): the argument y + sum alpha_ij k_j,
            //  f there, then rhs = h f + h J v with v = sum gamma_ij k_j,
            //  then the solve that gives k_i. -
            for (int i = 0; i < stages && st == 0; ++i)
            {
                for (int l = 0; l < m; ++l) arg[l * n] = state[l * n + c];
                for (int j = 0; j < i; ++j)
                {
                    const ofchem a_ij = alpha[i * 4 + j];
                    if (a_ij == 0.0) continue;
                    for (int l = 0; l < m; ++l)
                        arg[l * n] = __dadd_rn(arg[l * n],
                            __dmul_rn(a_ij, kj[j][l * n]));
                }
                st = chemEdcRhs(arg, p_c, ybar + c, hbar, te, mf, mi, yp + c,
                                conc + c, g + c, omg + c, rhs, n);
                if (st != 0) break;
                for (int l = 0; l < m; ++l) v[l * n] = 0.0;
                for (int j = 0; j < i; ++j)
                {
                    const ofchem g_ij = gij[i * 4 + j];
                    if (g_ij == 0.0) continue;
                    for (int l = 0; l < m; ++l)
                        v[l * n] = __dadd_rn(v[l * n],
                            __dmul_rn(g_ij, kj[j][l * n]));
                }
                for (int l = 0; l < m; ++l)
                {
                    ofchem jv = 0.0;
                    for (int q = 0; q < m; ++q)
                        jv = __dadd_rn(jv, __dmul_rn(jac[(l * m + q) * n + c],
                                                     v[q * n]));
                    rhs[l * n] = __dadd_rn(__dmul_rn(hstep, rhs[l * n]),
                                           __dmul_rn(hstep, jv));
                }
                chemLuSolve(lu + c, piv + c, m, rhs, n);
                for (int l = 0; l < m; ++l) kj[i][l * n] = rhs[l * n];
            }
            if (st != 0) break;
            //- (132.2): y_new = y + sum b_i k_i, est = sum (b_i - bhat_i)
            //  k_i, both sums in i order, one loop. -
            for (int l = 0; l < m; ++l)
            {
                ofchem sy = 0.0;
                ofchem se = 0.0;
                for (int i = 0; i < stages; ++i)
                {
                    sy = __dadd_rn(sy, __dmul_rn(b[i], kj[i][l * n]));
                    se = __dadd_rn(se, __dmul_rn(__dsub_rn(b[i], bhat[i]),
                                                 kj[i][l * n]));
                }
                ynew[l * n] = __dadd_rn(state[l * n + c], sy);
                est[l * n] = se;
            }
            //- (132.5): Err = sqrt(sum (est/(atol + rtol|y_new|))^2 / m),
            //  m = K + 1. -
            ofchem err_sum = 0.0;
            int finite = 1;
            for (int l = 0; l < m; ++l)
            {
                const ofchem y_l = ynew[l * n];
                if (!isfinite(y_l)) finite = 0;
                const ofchem atol_l = (l < K) ? atol_y : atol_t;
                const ofchem e = __ddiv_rn(est[l * n],
                    __dadd_rn(atol_l, __dmul_rn(rtol, fabs(y_l))));
                err_sum = __dadd_rn(err_sum, __dmul_rn(e, e));
            }
            const ofchem err = sqrt(__ddiv_rn(err_sum, (ofchem)m));
            if (!isfinite(err)) finite = 0;
            const ofchem fac = fmin(fac_max,
                fmax(0.1, __ddiv_rn(0.9, pow(err, inv_p1))));
            if (finite && err < 1.0)
            {
                //- Accept; on the last step t is span exactly (R3). -
                t_run = last ? span : __dadd_rn(t_run, hstep);
                for (int l = 0; l < m; ++l) state[l * n + c] = ynew[l * n];
                accepted++;
                accepted_any = 1;
                fac_max = 10.0;
                hstep = fmin(__dmul_rn(hstep, fac), h_max);
                have_jac = 0;
                //- advance's observer (R10): below -atol_y refuses, in
                //  [-atol_y, 0) clips to 0, in species order. -
                for (int l = 0; l < K; ++l)
                {
                    if (state[l * n + c] < -atol_y) { st = 8; break; }
                    if (state[l * n + c] < 0.0)
                    {
                        state[l * n + c] = 0.0;
                        clipped++;
                    }
                }
            }
            else
            {
                //- Reject; R4's non-finite step always divides h by 10. -
                rejected++;
                fac_max = 1.0;
                if (!finite) hstep = __ddiv_rn(hstep, 10.0);
                else if (accepted_any) hstep = __dmul_rn(hstep, fac);
                else hstep = __ddiv_rn(hstep, 10.0);
            }
        }
    }
    if (st == 0)
    {
        //- (134.17): the residual of (134.8) after the span, in the
        //  integrator's own weights, with f at the final state in the f0
        //  column. -
        st = chemEdcRhs(state + c, p_c, ybar + c, hbar, te, mf, mi, yp + c,
                        conc + c, g + c, omg + c, f0, n);
        if (st == 0)
        {
            ofchem rsum = 0.0;
            for (int l = 0; l < m; ++l)
            {
                const ofchem atol_l = (l < K) ? atol_y : atol_t;
                const ofchem e = __ddiv_rn(__dmul_rn(te, f0[l * n]),
                    __dadd_rn(atol_l, __dmul_rn(rtol, fabs(state[l * n + c]))));
                rsum = __dadd_rn(rsum, __dmul_rn(e, e));
            }
            const ofchem res = sqrt(__ddiv_rn(rsum, (ofchem)m));
            if (!isfinite(res) || res > 1.0) fl |= 2;
        }
    }
    if (st == 0)
    {
        //- (134.10) with chi = 1: W_k omega-bar_k =
        //  ((rho gamma*) (Y*_k - Ybar_k)) / tau_e. -
        const ofchem base = __dmul_rn(rho_c, gs);
        for (int k = 0; k < K; ++k)
        {
            wdot[k * n + c] = (ofscalar)(__ddiv_rn(
                __dmul_rn(base, __dsub_rn(state[k * n + c], ybar[k * n + c])), te));
        }
    }
    if (st != 0)
    {
        for (int k = 0; k < K; ++k) wdot[k * n + c] = (ofscalar)nan("");
        for (int l = 0; l < m; ++l) state[l * n + c] = nan("");
        flags[c] = 0;
    }
    else
    {
        flags[c] = fl;
        h[c] = hstep;
    }
    counts[c] = accepted;
    counts[n + c] = rejected;
    counts[2 * n + c] = clipped;
    status[c] = st;
}

//- chemHeatRelease: SPEC-LIT 134.8 (134.18), the heat release of a mean
//  rate - Q = - sum_k h_k(T) W_k omega-bar_k, with h_k the species'
//  specific enthalpy carrying its formation enthalpy (J/kg) =
//  R T (H/RT)_k / W_k, and W_k omega-bar_k = wdot_k, the (134.10) mean
//  rate in mechanism order, kg/(m3 s). One thread per cell, the
//  chemProductionRates launch style: t/q/status[c], wdot[k*n+c]. The sum
//  runs in mechanism order over wdot != 0, term for term as the host twin
//  edc_heat_release_host runs it, so the two agree to round-off in f64.
//  Status (132.9 D3's codes, in the hosts' order of checks): 2 T not
//  finite or not above 0, 13 a non-finite wdot_k, 5 a T outside a needed
//  record's range; a refused cell writes NaN to its q, nothing else. -
extern "C" __global__ void chemHeatRelease
(
    const ofscalar* __restrict__ t,
    const ofscalar* __restrict__ wdot,
    ofscalar* __restrict__ q,
    int* __restrict__ status,
    const ofchem* __restrict__ mf,
    const int* __restrict__ mi,
    oflabel n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;
    const int K = mi[0];
    const ofchem rg = mf[18 * K];
    const ofchem tc = (ofchem)t[c];
    if (!isfinite(tc) || tc <= 0.0)
    {
        q[c] = (ofscalar)nan("");
        status[c] = 2;
        return;
    }
    ofchem sum = 0.0;
    for (int k = 0; k < K; ++k)
    {
        const ofchem w = (ofchem)wdot[k * n + c];
        if (!isfinite(w))
        {
            q[c] = (ofscalar)nan("");
            status[c] = 13;
            return;
        }
        if (w == 0.0) continue;
        const ofchem* sp = mf + 18 * k;
        if (chemRange(sp, tc) != 0)
        {
            q[c] = (ofscalar)nan("");
            status[c] = 5;
            return;
        }
        sum += (rg * tc * chemHrt(chemCoeffs(sp, tc), tc)) * w / sp[0];
    }
    q[c] = (ofscalar)(-sum);
    status[c] = 0;
}
