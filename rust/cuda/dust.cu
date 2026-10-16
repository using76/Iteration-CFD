// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
/*
  ==========================================================================
  The FSC and TFC closures of the dust flame on the device. SPEC-LIT 154
  (docs/17 DST-03).
  ==========================================================================

  Written from:
    ofgpu SPEC-LIT.md §154.2 - the developing diffusivity and burning
      velocity (154.2)-(154.3), and the series of the bracket below x = 0.5
    ofgpu SPEC-LIT.md §154.3 - U_t,ISP and Da (154.4), tau_L (154.5)
    ofgpu SPEC-LIT.md §154.4 - the k-omega mapping of (154.6)
    ofgpu SPEC-LIT.md §154.5 - the extended FSC (154.7)
    ofgpu SPEC-LIT.md §154.6 - the TFC limit (154.9)
    ofgpu SPEC-LIT.md §154.7 - the f32-safe rate constant (154.10) and its
      zero rule; §118.11 - why K_b is formed once in f64
    ofgpu SPEC-LIT.md §155.1-§155.6 - the progress-variable kernels (DST-04):
      the sources of (155.3) with the split (154.11), the ignition bracket
      (155.4), the flame-Courant number (155.9), the Reynolds mean (155.2)
      and the bounds counter
    C. Huang, A. N. Lipatnikov & K. Nessvi, J. Loss Prev. Process Ind. 67
      (2020) 104237, DOI 10.1016/j.jlp.2020.104237 - eqs. (3)-(7), (15)-(17),
      and eqs. (1)-(2), (14) for the progress-variable kernels
    C. Huang, M. Bloching & A. N. Lipatnikov, J. Loss Prev. Process Ind. 75
      (2022) 104707, DOI 10.1016/j.jlp.2021.104707 - eqs. (7)-(11)
    ofgpu SPEC-LIT.md (152.12) - the heat release of the equivalent gas
    ofgpu SPEC-LIT.md (133.17), (133.18) - the enthalpy diffusion and the
      mixture c_p, with Y_B = c~ and the inert U, as DST-06 part 2 forms
      them (the three dust thermo kernels at the end of this file)
  The specification was informed by a review of FSCDustFoam (Huang &
  Lipatnikov, GPL) and no code was copied.
  No GPL-licensed source was consulted.

  One thread per cell. Every intermediate is `ofacc` (double) in BOTH builds
  and each output is rounded once at its store (§154.10, decision D1): an
  f32 exponent near -70 is spaced 7.6e-6 apart, so an E formed in float
  could not hold the round-off class its gate demands. The host twins in
  src/dust/fsc.rs make the same operations in the same order. The only
  products that could be contracted into a fused multiply-add are the
  series' running terms, and those are written with __dmul_rn, which nvcc
  never contracts; every other expression has no a*b + c in it. What is
  left between device and host is the last bit of exp, expm1 and sqrt.

  The zero rule (§154.7): E and K are written as exactly +0 and the cell is
  counted when the exponent is below ezero = -102 ln 2 (E < FLT_MIN 2^24),
  and K is also written as +0 and counted when K_b E / (1 + D_t/kappa_b)
  falls below FLT_MIN (decision D2). No output word is ever subnormal in
  the float build, and the f64 build zeroes the same cells.
*/

#include "ofgpu_device.cuh"

// FLT_MIN = 2^-126, as a double in both builds.
#define OFGPU_DUST_FLT_MIN 1.1754943508222875e-38

//- (154.3)'s bracket T1(x) = 1 - 1/x + exp(-x)/x for x >= 0: the series
//  sum_{j>=1} (-1)^{j+1} x^j/(j+1)! below x = 0.5, summed until a term is
//  below 1e-17 of the sum (at most 25 terms), the closed form above.
OFGPU_DEV ofacc dustBracket(const ofacc x)
{
    if (x < 0.5)
    {
        ofacc term = 0.5 * x;
        ofacc sum = term;
        for (int j = 2; j <= 25; ++j)
        {
            term = __dmul_rn(term, -x / (ofacc)(j + 1));
            sum = sum + term;
            if (fabs(term) < 1.0e-17 * fabs(sum)) break;
        }
        return sum;
    }
    // At x = +inf (tau_L = 0, no nu_t) this is exactly 1.
    return (1.0 - 1.0 / x) + exp(-x) / x;
}

//- (154.10)'s bounded exponent with the zero rule: returns 0 and sets
//  *zeroed when theta_b (1 - T_b/T) < ezero (or is not a number).
OFGPU_DEV ofacc dustArrheniusE
(
    const ofacc T,
    const ofacc thetab,
    const ofacc tb,
    const ofacc ezero,
    bool* zeroed
)
{
    const ofacc arg = thetab * (1.0 - tb / T);
    if (!(arg >= ezero))
    {
        *zeroed = true;
        return 0.0;
    }
    *zeroed = false;
    return exp(arg);
}

//- The closures of §154, per cell. model = 0 is FSC, 1 is TFC. tfd is the
//  flame development time t_fd and tign the time since ignition t of
//  (154.7), both on the global clock. tflacc <= 0 turns the extended FSC
//  off. tauf = kappa_u / (S_L S_L) and kb = exp(-theta_b)/t_r arrive formed
//  on the host in f64.
extern "C" __global__ void dustFsc
(
    ofscalar* __restrict__ dt,
    ofscalar* __restrict__ ut,
    ofscalar* __restrict__ utisp,
    ofscalar* __restrict__ taul,
    ofscalar* __restrict__ krate,
    unsigned long long* __restrict__ nzero,
    const ofscalar* __restrict__ T,
    const ofscalar* __restrict__ k,
    const ofscalar* __restrict__ omega,
    const ofscalar* __restrict__ nut,
    oflabel n,
    oflabel model,
    ofacc tfd,
    ofacc tign,
    ofacc tflacc,
    ofacc a,
    ofacc tauf,
    ofacc prt,
    ofacc cd,
    ofacc betastar,
    ofacc kb,
    ofacc thetab,
    ofacc tb,
    ofacc kappab,
    ofacc ezero
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;

    const ofacc ki = (ofacc)k[i];
    const ofacc wi = (ofacc)omega[i];

    // (154.4)-(154.6), §154.4's k-omega mapping. No turbulence, no
    // turbulent closure (decision D5).
    ofacc dinf = 0.0, u2 = 0.0, up = 0.0, isp = 0.0, tl = 0.0;
    if (ki > 0.0 && wi > 0.0)
    {
        u2 = (2.0 * ki) / 3.0;
        up = sqrt(u2);
        const ofacc l = (cd * sqrt(ki)) / (betastar * wi);
        dinf = (ofacc)nut[i] / prt;
        tl = dinf / u2;
        const ofacc da = (l / up) / tauf;
        isp = (a * up) * sqrt(sqrt(da));
    }

    ofacc d = dinf, u = isp;
    if (model == 0)
    {
        // (154.2)-(154.3), x = t_fd / tau_L; t_fd <= 0 is x = 0.
        const ofacc x = (tfd > 0.0) ? tfd / tl : 0.0;
        d = dinf * (-expm1(-x));
        u = isp * sqrt(dustBracket(x));
        // (154.7): T2 is not evaluated before t_flacc.
        if (tflacc > 0.0 && tign >= tflacc)
        {
            u = u * sqrt(tign / tflacc);
        }
    }

    ofacc kr = 0.0;
    if (model == 0)
    {
        bool zeroed = false;
        const ofacc e = dustArrheniusE((ofacc)T[i], thetab, tb, ezero, &zeroed);
        if (!zeroed)
        {
            kr = (kb * e) / (1.0 + d / kappab);
            if (kr < OFGPU_DUST_FLT_MIN)
            {
                kr = 0.0;
                zeroed = true;
            }
        }
        if (zeroed) atomicAdd(nzero, 1ULL);
    }

    dt[i] = (ofscalar)d;
    ut[i] = (ofscalar)u;
    utisp[i] = (ofscalar)isp;
    taul[i] = (ofscalar)tl;
    krate[i] = (ofscalar)kr;
}

//- (154.10)'s E alone, with the zero rule and its counter: the f32-safe
//  Arrhenius row of §154.9.
extern "C" __global__ void dustArrhenius
(
    ofscalar* __restrict__ e,
    unsigned long long* __restrict__ nzero,
    const ofscalar* __restrict__ T,
    oflabel n,
    ofacc thetab,
    ofacc tb,
    ofacc ezero
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    bool zeroed = false;
    const ofacc v = dustArrheniusE((ofacc)T[i], thetab, tb, ezero, &zeroed);
    if (zeroed) atomicAdd(nzero, 1ULL);
    e[i] = (ofscalar)v;
}

// ==========================================================================
//  The progress-variable equation (SPEC-LIT 155, docs/17 DST-04). Appended
//  after the DST-03 kernels, which are unchanged. The same rule holds: every
//  intermediate is `ofacc` in both builds and each output is rounded once at
//  its store; the host twins are in src/dust/{progress,ignition,courant}.rs.
// ==========================================================================

//- The sources of (155.3) as one S_U + S_P c~ pair per cell (155.2):
//  S_U = rho_u U_t |grad c~| + rho K + rho W and S_P' = rho K + rho W, the
//  diagonal coefficient fvm_sp adds (S_P = -S_P'). K is (154.11)'s rate, W
//  the bracket of (155.4) (`hasIgn` = 0 leaves it out, `wign` unread), and
//  rhou the CURRENT unburned density rho_u(p) of (155.10).
extern "C" __global__ void dustProgressSources
(
    ofscalar* __restrict__ su,
    ofscalar* __restrict__ sp,
    const ofvec3* __restrict__ gradc,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ ut,
    const ofscalar* __restrict__ krate,
    const ofscalar* __restrict__ wign,
    oflabel n,
    oflabel hasIgn,
    ofacc rhou
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofacc gx = (ofacc)gradc[i].x;
    const ofacc gy = (ofacc)gradc[i].y;
    const ofacc gz = (ofacc)gradc[i].z;
    const ofacc mag = sqrt(__dadd_rn(__dadd_rn(__dmul_rn(gx, gx), __dmul_rn(gy, gy)), __dmul_rn(gz, gz)));
    const ofacc r = (ofacc)rho[i];
    ofacc rate = (ofacc)krate[i];
    if (hasIgn) rate = rate + (ofacc)wign[i];
    const ofacc imp = __dmul_rn(r, rate);
    const ofacc prop = __dmul_rn(__dmul_rn(rhou, (ofacc)ut[i]), mag);
    su[i] = (ofscalar)(prop + imp);
    sp[i] = (ofscalar)imp;
}

//- The bracket of (155.4), W = W_0 exp{-[(r/sigma_r)^2 + H(t_0 - t)
//  ((t - t_0)/sigma_t)^2]}, 1/s. r is formed from LOCAL positions: the cell
//  centres as stored and the ignition point already shifted by the same
//  origin on the host (SPEC-LIT 118.7), so the difference has no
//  cancellation in either build. A W below FLT_MIN is written as +0 (the
//  float store would be subnormal), the same cells in both builds.
extern "C" __global__ void dustIgnition
(
    ofscalar* __restrict__ w,
    const ofvec3* __restrict__ c,
    oflabel n,
    ofacc px,
    ofacc py,
    ofacc pz,
    ofacc sigr,
    ofacc w0,
    ofacc t,
    ofacc t0,
    ofacc sigt
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofacc dx = (ofacc)c[i].x - px;
    const ofacc dy = (ofacc)c[i].y - py;
    const ofacc dz = (ofacc)c[i].z - pz;
    const ofacc r2 = __dadd_rn(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)), __dmul_rn(dz, dz));
    ofacc arg = r2 / __dmul_rn(sigr, sigr);
    if (t < t0)
    {
        const ofacc s = (t - t0) / sigt;
        arg = arg + __dmul_rn(s, s);
    }
    ofacc v = __dmul_rn(w0, exp(-arg));
    if (!(v >= OFGPU_DUST_FLT_MIN)) v = 0.0;
    w[i] = (ofscalar)v;
}

//- The flame-Courant number (155.9) per cell:
//  Co_f = dt (|u| + rho_u U_t / rho) / V^{1/3}.
extern "C" __global__ void dustFlameCourant
(
    ofscalar* __restrict__ co,
    const ofvec3* __restrict__ u,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ ut,
    const ofscalar* __restrict__ v,
    oflabel n,
    ofacc rhou,
    ofacc dt
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofacc ux = (ofacc)u[i].x;
    const ofacc uy = (ofacc)u[i].y;
    const ofacc uz = (ofacc)u[i].z;
    const ofacc mag = sqrt(__dadd_rn(__dadd_rn(__dmul_rn(ux, ux), __dmul_rn(uy, uy)), __dmul_rn(uz, uz)));
    const ofacc w = __dmul_rn(rhou, (ofacc)ut[i]) / (ofacc)rho[i];
    co[i] = (ofscalar)(__dmul_rn(dt, mag + w) / cbrt((ofacc)v[i]));
}

//- The Reynolds mean (155.2): cbar = rho c~ / rho_b(p).
extern "C" __global__ void dustReynoldsMean
(
    ofscalar* __restrict__ cbar,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ c,
    oflabel n,
    ofacc rhob
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    cbar[i] = (ofscalar)(__dmul_rn((ofacc)rho[i], (ofacc)c[i]) / rhob);
}

//- The bounds counter of SPEC-LIT 155.2: every cell with c~ < -tol,
//  c~ > 1 + tol or not a number is added to nout[0], tol the round-off
//  margin of the build (64 epsilon). Nothing is clipped.
extern "C" __global__ void dustCountOutOfBounds
(
    unsigned long long* __restrict__ nout,
    const ofscalar* __restrict__ c,
    oflabel n,
    ofscalar tol
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofscalar v = c[i];
    if (!(v >= -tol && v <= (ofscalar)1 + tol)) atomicAdd(nout, 1ULL);
}

// ==========================================================================
//  The equivalent-gas thermo of one dust step (SPEC-LIT (152.12) and the
//  SPEC-LIT (133.17), (133.18) forms with Y_B = c~, docs/17 DST-06 part 2).
//  Appended after the DST-03/DST-04 kernels, which are unchanged. The same
//  rule holds: every intermediate is `ofacc` in both builds and each output
//  is rounded once at its store. One `ofacc` row per pseudo-species holds
//  [R0/W, t_mid, lower a1..a5, lower a6, upper a1..a5, upper a6] (128.5);
//  `thermo_row` in src/dust/device.rs forms it from the record.
// ==========================================================================

//- (128.1) on one row, cp/R: the interval is the lower set when
//  T <= the row's t_mid, else the upper set (SPEC-LIT 128.1). The Horner
//  chain is written with __dmul_rn/__dadd_rn, which nvcc never contracts,
//  so the host twin (SPEC-LIT 128.1's nasa7_cp_r) and this agree bit for
//  bit in f64.
OFGPU_DEV ofacc dustRowCpR(const ofacc* r, const ofacc T)
{
    const ofacc* a = (T <= r[1]) ? &r[2] : &r[8];
    const ofacc p4 = __dmul_rn(T, a[4]);
    const ofacc p3 = __dmul_rn(T, __dadd_rn(a[3], p4));
    const ofacc p2 = __dmul_rn(T, __dadd_rn(a[2], p3));
    const ofacc p1 = __dmul_rn(T, __dadd_rn(a[1], p2));
    return __dadd_rn(a[0], p1);
}

//- (128.2) on one row, H/(RT): a1 + a2 T/2 + a3 T^2/3 + a4 T^3/4 +
//  a5 T^4/5 + a6/T. The sum runs left to right and every quotient is a
//  quotient, both as the host twin's nasa7_h_rt evaluates them, so the two
//  agree bit for bit in f64. A row's a[5] IS its a6: the lower set starts
//  at r[2] (a6 at r[7]) and the upper at r[8] (a6 at r[13]).
OFGPU_DEV ofacc dustRowHrT(const ofacc* r, const ofacc T)
{
    const ofacc* a = (T <= r[1]) ? &r[2] : &r[8];
    const ofacc t2 = __dmul_rn(T, T);
    const ofacc s1 = __dadd_rn(a[0], __ddiv_rn(__dmul_rn(a[1], T), 2.0));
    const ofacc s2 = __dadd_rn(s1, __ddiv_rn(__dmul_rn(a[2], t2), 3.0));
    const ofacc s3 = __dadd_rn(s2, __ddiv_rn(__dmul_rn(__dmul_rn(a[3], t2), T), 4.0));
    const ofacc s4 = __dadd_rn(s3, __ddiv_rn(__dmul_rn(__dmul_rn(a[4], t2), t2), 5.0));
    return __dadd_rn(s4, __ddiv_rn(a[5], T));
}

//- (152.4)/(133.18): the mixture c_p per kg on every cell,
//  cp = (1 - c) cp_U(T) + c cp_B(T), with cp_X = (R0/W_X) cp/R of (128.1) -
//  row[0] IS R0/W_X, so a row's cp_X is row[0] * cp/R.
extern "C" __global__ void dustMixtureCp
(
    ofscalar* __restrict__ cp,
    const ofscalar* __restrict__ t,
    const ofscalar* __restrict__ c,
    const ofacc* __restrict__ rowu,
    const ofacc* __restrict__ rowb,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofacc T = (ofacc)t[i];
    const ofacc ci = (ofacc)c[i];
    const ofacc cpu = __dmul_rn(rowu[0], dustRowCpR(rowu, T));
    const ofacc cpb = __dmul_rn(rowb[0], dustRowCpR(rowb, T));
    cp[i] = (ofscalar)__dadd_rn(__dmul_rn(1.0 - ci, cpu), __dmul_rn(ci, cpb));
}

//- (152.12): the heat release qdot = (h_U(T) - h_B(T)) wdot_c, W/m3, with
//  h_X = (R0/W_X) T H/(RT) J/kg - row[0] T H/(RT), formed in that order.
extern "C" __global__ void dustHeatRelease
(
    ofscalar* __restrict__ q,
    const ofscalar* __restrict__ t,
    const ofscalar* __restrict__ wdot,
    const ofacc* __restrict__ rowu,
    const ofacc* __restrict__ rowb,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofacc T = (ofacc)t[i];
    const ofacc hu = __dmul_rn(__dmul_rn(rowu[0], T), dustRowHrT(rowu, T));
    const ofacc hb = __dmul_rn(__dmul_rn(rowb[0], T), dustRowHrT(rowb, T));
    q[i] = (ofscalar)__dmul_rn(hu - hb, (ofacc)wdot[i]);
}

//- (133.17) with Y_B = c~ and the inert U: Q_hd = rho Gamma (cp_B(T) -
//  cp_U(T)) (grad c~ . grad T), W/m3. Le = 1 makes Gamma the SAME scalar
//  the c~ equation diffuses with, so c~ and T diffuse alike (DST-06's
//  choice); kappa = k / (rho cp(T, c~)) is the molecular half of it.
extern "C" __global__ void dustEnthalpyDiffusion
(
    ofscalar* __restrict__ q,
    const ofscalar* __restrict__ t,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ gamma,
    const ofvec3* __restrict__ gradc,
    const ofvec3* __restrict__ gradt,
    const ofacc* __restrict__ rowu,
    const ofacc* __restrict__ rowb,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofacc T = (ofacc)t[i];
    const ofacc dcp = __dmul_rn(rowb[0], dustRowCpR(rowb, T))
                    - __dmul_rn(rowu[0], dustRowCpR(rowu, T));
    const ofacc gx = __dmul_rn((ofacc)gradc[i].x, (ofacc)gradt[i].x);
    const ofacc gy = __dmul_rn((ofacc)gradc[i].y, (ofacc)gradt[i].y);
    const ofacc gz = __dmul_rn((ofacc)gradc[i].z, (ofacc)gradt[i].z);
    const ofacc dot = __dadd_rn(__dadd_rn(gx, gy), gz);
    const ofacc rg = __dmul_rn((ofacc)rho[i], (ofacc)gamma[i]);
    q[i] = (ofscalar)__dmul_rn(__dmul_rn(rg, dcp), dot);
}
