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
      zero rule; §118.7 - why K_b is formed once in f64
    C. Huang, A. N. Lipatnikov & K. Nessvi, J. Loss Prev. Process Ind. 67
      (2020) 104237, DOI 10.1016/j.jlp.2020.104237 - eqs. (3)-(7), (15)-(17)
    C. Huang, M. Bloching & A. N. Lipatnikov, J. Loss Prev. Process Ind. 75
      (2022) 104707, DOI 10.1016/j.jlp.2021.104707 - eqs. (7)-(11)
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
