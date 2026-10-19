// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
/*
  ==========================================================================
  The reacting dust parcel on the device. SPEC-LIT 159 (docs/17 DST-12,
  DST-13).
  ==========================================================================

  Written from:
    ofgpu SPEC-LIT.md §159.1 - (159.1) the particle energy, (159.2)
      Ranz-Marshall, (159.3) Bird's correction and its series
    ofgpu SPEC-LIT.md §159.2 - (159.4) SFOM devolatilisation
    ofgpu SPEC-LIT.md §159.4 - (159.7) the radiation temperature
    ofgpu SPEC-LIT.md §159.7 - the stages, the segment integrator with
      phi1/g2 and the accumulators
    ofgpu SPEC-LIT.md §159.2, §159.8 - (159.5) the char rate with the heat
      retention h_s, its exact shrinkage (159.10) and burnout time, the
      element ratios (159.11) and the stop on a particle with no mass left
    ofgpu SPEC-LIT.md §76.1, §76.4, §76.5, §76.9 - the moisture: (76.1)'s
      Ranz-Marshall rung, the film, Watson's h_v, D(T), Y_s and the boiling
      branch; (S54.3) of §54 - Hyland-Wexler p_ws; §77.1 - (77.5)
    A. Islas, A. Rodriguez Fernandez, C. Betegon, E. Martinez-Paneda,
      A. Pandal, arXiv:2207.13149 (2022) - eqs. (13)-(17), (20)-(23)
  The specification was informed by a review of FSCDustFoam (Huang &
  Lipatnikov, GPL) and no code was copied.
  No GPL-licensed source was consulted.

  One thread per parcel. Every intermediate is `ofacc` (double) in BOTH
  builds and each state word is rounded once at its store; the eleven
  accumulators are `ofacc` and added to. The char mass is rounded where the
  char segment ends, so the carbon given is the difference of two stored
  words (§159.8); the host twin src/dust/char.rs rounds it there too. The
  unit is compiled with -fmad=false (build.rs FMAD_OFF_UNITS), so every
  product and sum rounds as the host twin's does: the host twins
  src/dust/particle.rs and src/dust/char.rs make the same operations in the
  same order, and what is left between them is the last bit of exp, expm1,
  log, log1p, pow and cbrt.
*/

#include "ofgpu_device.cuh"

#define DUSTP_PI          3.14159265358979323846
#define DUSTP_SIGMA       5.670374419e-8
#define DUSTP_BIRD_SERIES 1.0e-3
#define DUSTP_STAGE_END   1.0e-6
// An all-volatile parcel never meets DUSTP_STAGE_END's fraction (its
// remaining mass IS m_v), so the volatile stage also ends when the remnant
// drops below the smallest normal number of the stored word: the smallest
// normal double in the f64 build; under OFGPU_SINGLE the stored m_v is a
// float, so FLT_MIN (SPEC-LIT 159.11).
#ifdef OFGPU_SINGLE
#define DUSTP_MASS_FLOOR  1.17549435e-38
#else
#define DUSTP_MASS_FLOOR  2.2250738585072014e-308
#endif
#define DUSTP_MAX_SEG     8
#define DUSTP_PHI_SERIES  1.0

//- (159.3)'s factor beta/(e^beta - 1), with its series below 1e-3.
OFGPU_DEV ofacc dustpBird(const ofacc beta)
{
    if (fabs(beta) < DUSTP_BIRD_SERIES)
    {
        const ofacc b2 = beta * beta;
        return 1.0 - 0.5 * beta + b2 / 12.0 - b2 * b2 / 720.0;
    }
    return beta / expm1(beta);
}

//- phi1(x) = (1 - e^-x)/x and g2(x) = (x - 1 + e^-x)/x^2 of §159.7.
OFGPU_DEV void dustpPhiG(const ofacc x, ofacc* p1, ofacc* g2)
{
    if (x < DUSTP_PHI_SERIES)
    {
        ofacc t1 = 1.0, s1 = 1.0, t2 = 0.5, s2 = 0.5;
        for (int j = 1; j < 30; ++j)
        {
            t1 = t1 * (-x) / ((ofacc)j + 1.0);
            t2 = t2 * (-x) / ((ofacc)j + 2.0);
            s1 = s1 + t1;
            s2 = s2 + t2;
            if (fabs(t1) < 1.0e-17 * fabs(s1) && fabs(t2) < 1.0e-17 * fabs(s2)) break;
        }
        *p1 = s1;
        *g2 = s2;
        return;
    }
    const ofacc em = expm1(-x);
    *p1 = -em / x;
    *g2 = (x + em) / (x * x);
}

//- Hyland-Wexler (S54.3), over water at and above 273.15 K, over ice below.
OFGPU_DEV ofacc dustpPws(const ofacc t)
{
    const ofacc l = log(t);
    if (t < 273.15)
    {
        return exp(-5.6745359e3 / t + 6.3925247 + -9.677843e-3 * t + 6.2215701e-7 * t * t
                   + 2.0747825e-9 * t * t * t
                   + -9.484024e-13 * t * t * t * t
                   + 4.1635019 * l);
    }
    return exp(-5.8002206e3 / t + 1.3914993 + -4.8640239e-2 * t + 4.1764768e-5 * t * t
               + -1.4452093e-8 * t * t * t
               + 6.5459673 * l);
}

//- Watson's h_v (§76.4), zero at and above T_crit.
OFGPU_DEV ofacc dustpLatent(const ofacc t, const ofacc hvBoil, const ofacc tBoil, const ofacc tCrit)
{
    if (t >= tCrit) return 0.0;
    return hvBoil * pow((tCrit - t) / (tCrit - tBoil), 0.38);
}

//- One frozen-rate segment of (159.1) (§159.7).
OFGPU_DEV void dustpSegment
(
    const ofacc a, const ofacc b, const ofacc rad0, const ofacc s,
    const ofacc tg, const ofacc t0, const ofacc tseg, const ofacc mc,
    ofacc* t1, ofacc* qc, ofacc* qr
)
{
    const ofacc num = a * (tg - t0) + rad0 + s;
    const ofacc lam = a + b;
    const ofacc x = lam * tseg / mc;
    ofacc p1, g2;
    dustpPhiG(x, &p1, &g2);
    const ofacc tt = tseg / mc;
    const ofacc w = num * tseg * tt * g2;
    *t1 = t0 + num * tt * p1;
    *qc = a * ((tg - t0) * tseg - w);
    *qr = rad0 * tseg - b * w;
}

//- One char segment of at most rem seconds (§159.8), the twin of
//  dust::char::burn. Returns false when nothing burns.
OFGPU_DEV bool dustpCharBurn
(
    const ofacc mc, const ofacc ma, const ofacc d0, const ofacc tp, const ofacc tg,
    const ofacc pO2, const ofacc rem, const ofacc c0, const ofacc chA, const ofacc chTheta,
    ofacc* tseg, bool* out, ofacc* mc1, ofacc* d1, ofacc* dm
)
{
    if (!(mc > 0.0 && pO2 > 0.0)) return false;
    const ofacc m0 = mc + ma;
    const ofacc rho = 6.0 * m0 / (DUSTP_PI * d0 * d0 * d0);
    const ofacc cd = c0 * pow(0.5 * (tg + tp), 0.75);
    const ofacc rk = chA * exp(-chTheta / tg);
    const ofacc kk = 2.0 * pO2 * cd * rk / rho;
    if (!(kk > 0.0)) return false;
    const ofacc dEnd = d0 * cbrt(ma / m0);
    const ofacc dmax = d0 - dEnd;
    const ofacc tOut = dmax * (cd + rk * 0.5 * (d0 + dEnd)) / kk;
    if (!(tOut > 0.0) || tOut <= rem)
    {
        // Burnout inside the segment, or a remnant that leaves at once.
        *tseg = (tOut > 0.0) ? tOut : 0.0;
        *out = true;
        *mc1 = 0.0;
        *d1 = dEnd;
        *dm = mc;
        return true;
    }
    // (159.10), the exact shrinkage at frozen rates and constant density.
    const ofacc bb = cd + rk * d0;
    const ofacc sh = 2.0 * kk * rem / (bb + sqrt(fmax(bb * bb - 2.0 * rk * kk * rem, 0.0)));
    const ofacc dn = d0 - sh;
    const ofacc dmRaw = m0 * sh * (d0 * d0 + d0 * dn + dn * dn) / (d0 * d0 * d0);
    const ofacc m1 = (ofacc)(ofscalar)fmax(mc - dmRaw, 0.0);
    *tseg = rem;
    *out = false;
    *mc1 = m1;
    *d1 = d0 * cbrt((m1 + ma) / m0);
    *dm = mc - m1;
    return true;
}

extern "C" __global__ void dustParticleStep
(
    ofscalar* __restrict__ mw,
    ofscalar* __restrict__ mv,
    ofscalar* __restrict__ mch,
    const ofscalar* __restrict__ ma,
    ofscalar* __restrict__ tp,
    ofscalar* __restrict__ dp,
    ofacc* __restrict__ accMw,
    ofacc* __restrict__ accMv,
    ofacc* __restrict__ accMtw,
    ofacc* __restrict__ accMtv,
    ofacc* __restrict__ accQc,
    ofacc* __restrict__ accQr,
    ofacc* __restrict__ accMc,
    ofacc* __restrict__ accMo2,
    ofacc* __restrict__ accMco2,
    ofacc* __restrict__ accMtc,
    ofacc* __restrict__ accQch,
    const oflabel* __restrict__ cell,
    const ofscalar* __restrict__ urel,
    const ofscalar* __restrict__ Tg,
    const ofscalar* __restrict__ Yw,
    const ofscalar* __restrict__ rhog,
    const ofscalar* __restrict__ G,
    const ofscalar* __restrict__ pO2,
    oflabel n,
    ofacc dt,
    ofacc cp, ofacc eps, ofacc devA, ofacc theta, ofacc devDh, ofacc cpVol,
    ofacc mu, ofacc kg, ofacc cbrtPr,
    ofacc wVap, ofacc tBoil, ofacc pBoil, ofacc hvBoil, ofacc tCrit,
    ofacc dVap, ofacc tRefD, ofacc dExp, ofacc cpVap, ofacc wCar, ofacc pAmb, ofacc tb,
    ofacc chC0, ofacc chA, ofacc chTheta, ofacc chHc, ofacc chHs, ofacc chGas,
    ofacc o2PerC, ofacc co2PerC
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    ofacc m_w = mw[i], m_v = mv[i], m_c = mch[i];
    const ofacc m_a = ma[i];
    ofacc t = tp[i];
    ofacc d = dp[i];
    if (!(m_w + m_v + m_c + m_a > 0.0 && d > 0.0)) return;
    const oflabel c = cell[i];
    const ofacc tgas = Tg[c], ygas = Yw[c], rho = rhog[c], po2 = pO2[c];
    const ofacc ur = urel[i];
    const ofacc g4 = 0.25 * (ofacc)G[c];
    ofacc exMw = 0.0, exMv = 0.0, exQc = 0.0, exQr = 0.0;
    ofacc exMc = 0.0, exMo2 = 0.0, exMco2 = 0.0, exQch = 0.0;
    ofacc rem = dt;
    for (int seg = 0; seg < DUSTP_MAX_SEG; ++seg)
    {
        // A particle with no mass left has nothing to heat (§159.8).
        if (rem <= 0.0 || !(m_w + m_v + m_c + m_a > 0.0)) break;
        // The diameter moves only in the char stage.
        const ofacc re = rho * ur * d / mu;
        const ofacc sqre = sqrt(re);
        const ofacc nu0 = 2.0 + 0.6 * sqre * cbrtPr;
        const ofacc a0 = DUSTP_PI * d * kg * nu0;
        const ofacc ar = DUSTP_PI * d * d * eps;
        const ofacc mc = (m_w + m_v + m_c + m_a) * cp;
        const ofacc t0 = t;
        const ofacc t03 = t0 * t0 * t0;
        const ofacc rad0 = ar * (g4 - DUSTP_SIGMA * t03 * t0);
        const ofacc b = 4.0 * ar * DUSTP_SIGMA * t03;
        if (m_w > 0.0)
        {
            const ofacc hv = dustpLatent(fmin(t0, tb), hvBoil, tBoil, tCrit);
            const ofacc bnum = cpVap * (tgas - tb) / hv;
            const ofacc fb = (bnum > 0.0) ? log1p(bnum) / bnum : 1.0;
            const ofacc ab = a0 * fb;
            const ofacc tb3 = tb * tb * tb;
            const ofacc radB = ar * (g4 - DUSTP_SIGMA * tb3 * tb);
            const ofacc qB = ab * (tgas - tb) + radB;
            if (t0 >= tb && qB > 0.0)
            {
                // (76.9): heat-limited, T_p held at T_b.
                const ofacc mdot = qB / hv;
                const ofacc tOut = m_w / mdot;
                ofacc tseg, dm;
                bool out;
                if (tOut <= rem) { tseg = tOut; dm = m_w; out = true; }
                else { tseg = rem; dm = mdot * rem; out = false; }
                out = out || dm >= m_w;
                if (out) dm = m_w;
                m_w = out ? 0.0 : m_w - dm;
                exMw = exMw + dm;
                exQc = exQc + ab * (tgas - tb) * tseg;
                exQr = exQr + radB * tseg;
                t = tb;
                rem = out ? rem - tseg : 0.0;
                continue;
            }
            // (76.1), the Ranz-Marshall rung, with the (76.4) film.
            const ofacc tpc = fmin(t0, tb);
            const ofacc tf = tpc + (tgas - tpc) / 3.0;
            const ofacc rhof = rho * tgas / tf;
            const ofacc df = dVap * pow(tf / tRefD, dExp) * (pBoil / pAmb);
            const ofacc sc = mu / (rhof * df);
            const ofacc sh = 2.0 + 0.6 * sqre * cbrt(sc);
            ofacc x = dustpPws(tpc) / pAmb;
            if (x < 0.0) x = 0.0;
            if (x > 1.0) x = 1.0;
            const ofacc ys = x * wVap / (wCar + x * (wVap - wCar));
            const ofacc mdot = fmax(DUSTP_PI * d * rhof * df * sh * (ys - ygas), 0.0);
            const ofacc a = a0 * dustpBird(cpVap * mdot / a0);
            const ofacc sv = -mdot * hv;
            ofacc tseg = rem;
            bool out = false;
            if (mdot > 0.0)
            {
                const ofacc tOut = m_w / mdot;
                if (tOut <= rem) { tseg = tOut; out = true; }
            }
            bool hit = false;
            const ofacc num = a * (tgas - t0) + rad0 + sv;
            const ofacc lam = a + b;
            if (num > 0.0 && t0 < tb)
            {
                const ofacc r = lam * (tb - t0) / num;
                if (r < 1.0)
                {
                    const ofacc tHit = -log1p(-r) * mc / lam;
                    if (tHit < tseg) { tseg = tHit; out = false; hit = true; }
                }
            }
            ofacc t1, qc, qr;
            dustpSegment(a, b, rad0, sv, tgas, t0, tseg, mc, &t1, &qc, &qr);
            const ofacc dm = out ? m_w : fmin(mdot * tseg, m_w);
            out = out || dm >= m_w;
            m_w = out ? 0.0 : m_w - dm;
            exMw = exMw + dm;
            exQc = exQc + qc;
            exQr = exQr + qr;
            t = hit ? tb : t1;
            rem = (out || hit) ? rem - tseg : 0.0;
        }
        else if (m_v > 0.0)
        {
            // (159.4) at the frozen T_p.
            const ofacc k = devA * exp(-theta / t0);
            const ofacc e = -expm1(-k * rem);
            ofacc dm = m_v * e;
            ofacc mvn = m_v - dm;
            if (mvn <= DUSTP_STAGE_END * ((m_w + m_v + m_c + m_a) - dm) || mvn < DUSTP_MASS_FLOOR) { dm = m_v; mvn = 0.0; }
            const ofacc mdot = dm / rem;
            const ofacc a = a0 * dustpBird(cpVol * mdot / a0);
            const ofacc sv = -mdot * devDh;
            ofacc t1, qc, qr;
            dustpSegment(a, b, rad0, sv, tgas, t0, rem, mc, &t1, &qc, &qr);
            m_v = mvn;
            exMv = exMv + dm;
            exQc = exQc + qc;
            exQr = exQr + qr;
            t = t1;
            rem = 0.0;
        }
        else
        {
            ofacc tsc, mcn, dn, dmc;
            bool outc;
            if (dustpCharBurn(m_c, m_a, d, t0, tgas, po2, rem, chC0, chA, chTheta,
                              &tsc, &outc, &mcn, &dn, &dmc))
            {
                // (159.5): the char burns; h_s of its heat stays in the particle.
                const ofacc q = dmc * chHc;
                if (tsc > 0.0)
                {
                    const ofacc sv = chHs * q / tsc;
                    ofacc t1, qc, qr;
                    dustpSegment(a0, b, rad0, sv, tgas, t0, tsc, mc, &t1, &qc, &qr);
                    exQc = exQc + qc;
                    exQr = exQr + qr;
                    t = t1;
                }
                else
                {
                    t = t0 + chHs * q / mc;
                }
                exMc = exMc + dmc;
                exMo2 = exMo2 + dmc * o2PerC;
                exMco2 = exMco2 + dmc * co2PerC;
                exQch = exQch + chGas * q;
                m_c = mcn;
                d = dn;
                rem = outc ? rem - tsc : 0.0;
                continue;
            }
            // Ash, char with nothing to burn it, or a particle that is out.
            ofacc t1, qc, qr;
            dustpSegment(a0, b, rad0, 0.0, tgas, t0, rem, mc, &t1, &qc, &qr);
            exQc = exQc + qc;
            exQr = exQr + qr;
            t = t1;
            rem = 0.0;
        }
    }
    // (77.5): the vapour leaves at the end-of-step T_p as the state stores it.
    const ofscalar ts = (ofscalar)t;
    mw[i] = (ofscalar)m_w;
    mv[i] = (ofscalar)m_v;
    mch[i] = (ofscalar)m_c;
    dp[i] = (ofscalar)d;
    tp[i] = ts;
    accMw[i] = accMw[i] + exMw;
    accMv[i] = accMv[i] + exMv;
    accMtw[i] = accMtw[i] + exMw * (ofacc)ts;
    accMtv[i] = accMtv[i] + exMv * (ofacc)ts;
    accQc[i] = accQc[i] + exQc;
    accQr[i] = accQr[i] + exQr;
    accMc[i] = accMc[i] + exMc;
    accMo2[i] = accMo2[i] + exMo2;
    accMco2[i] = accMco2[i] + exMco2;
    accMtc[i] = accMtc[i] + exMc * (ofacc)ts;
    accQch[i] = accQch[i] + exQch;
}
