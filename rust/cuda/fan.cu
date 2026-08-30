// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
/*
  ==========================================================================
  Fan curves and porous jumps. SPEC-LIT S52 and S53.
  ==========================================================================

  Written from:
    ofgpu SPEC-LIT.md S4    - the universal Robin triple these kernels rewrite
    ofgpu SPEC-LIT.md S52.2 - the rank-1 downdate derivation, and S52.3's
      row-sum-preserving lumping fr = 1/(1 + S SIGMA_D)
    ofgpu SPEC-LIT.md S52.5 - the monotone Hermite curve and (S52.13)'s
      density and speed corrections
    ofgpu SPEC-LIT.md S52.7 - why every reduction is a gather into a compact
      buffer followed by the existing two-stage `device_sum`
    ofgpu SPEC-LIT.md S53.2 - the porous jump as a per-face division of
      THREE arrays by one number
    F. N. Fritsch, R. E. Carlson, SIAM J. Numer. Anal. 17 (1980) 238-246 -
      the monotone slope limiter of fanCurveEval
    FDS (NIST, US Government public domain; reference/fds/LICENSE.md read
      verbatim) - the DISCIPLINE that a fan curve is scaled by rho/rho_curve
      at every evaluation, and the WARNING that its tabulated branch resolves
      the operating point by a bisection with a data-dependent trip count,
      which is uncapturable here. Its source was read for those two points
      only; nothing here is transcribed from it.
  No GPL-licensed source was consulted. OpenFOAM's `fanPressure` and
  `porousBafflePressure` were not opened.

  ==========================================================================
  What is here, and what is deliberately not
  ==========================================================================

  NO f64 ATOMIC ANYWHERE. Every sum in S52 and S55 is a GATHER: one thread
  per item writes its own contribution into a compact buffer, and
  `solver::device_sum` reduces that buffer with a partition that is a pure
  function of n (S8.4). That is what makes the fan operating point bitwise
  reproducible across runs and across schedulings.

  EVERY LOOP HAS A FIXED TRIP COUNT. The curve scan is nPoints iterations
  whatever the answer; there is no bisection and no convergence test inside a
  kernel. That is what keeps the whole fan update CUDA-Graph capturable
  (S52.7), and it is the one place FDS's shape is deliberately not followed.

  NO HOST READBACK. Q, Phi and SIGMA_D stay device scalars from the reduction
  to the triple. The curve evaluation, the slope, the under-relaxation and
  the triple rewrite are kernels reading those scalars.
*/

#include "ofgpu_device.cuh"

//  The single/double math wrappers, the same shape `cuda/energy.cu` uses.
#ifdef OFGPU_SINGLE
OFGPU_DEV ofscalar fanAbs_(ofscalar a)  { return fabsf(a); }
OFGPU_DEV ofscalar fanLog_(ofscalar a)  { return logf(a); }
OFGPU_DEV ofscalar fanExp_(ofscalar a)  { return expf(a); }
#else
OFGPU_DEV ofscalar fanAbs_(ofscalar a)  { return fabs(a); }
OFGPU_DEV ofscalar fanLog_(ofscalar a)  { return log(a); }
OFGPU_DEV ofscalar fanExp_(ofscalar a)  { return exp(a); }
#endif


// ==========================================================================
//  S52.7  The three gathers
// ==========================================================================

//- Gather one patch's face contributions into the front of three compact
//  buffers, ready for three `device_sum` calls of length n.
//
//      q[i]     = phi[start + i]                     the conservative flux
//      ph[i]    = phiHbyA[start + i]                 the Rhie-Chow flux
//      sd[i]    = rAU_f a_f Delta_f = bGammaMagSf[bf] * bDeltaCoeffs[bf]
//
//  `bGammaMagSf` is `momentum::pressure_laplacian_coeffs().1`, i.e.
//  rAU_b |Sf_b|, so the product with the delta coefficient is exactly the
//  boundary Laplacian conductance D_f of (S52.6). Nothing is recomputed
//  from rAU and the area separately: the assembly's own coefficient is the
//  one the boundary condition must be built from, or the two would differ
//  in the last bit and the flow rate the triple imposes would not be the
//  flow rate the matrix delivers.
extern "C" __global__ void fanGatherPatch
(
    ofscalar* __restrict__ q,
    ofscalar* __restrict__ ph,
    ofscalar* __restrict__ sd,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ phiHbyA,
    const ofscalar* __restrict__ bGammaMagSf,
    const ofscalar* __restrict__ bDeltaCoeffs,
    oflabel start,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const oflabel bf = start + i;

    q[i]  = phi[bf];
    ph[i] = phiHbyA[bf];
    sd[i] = bGammaMagSf[bf]*bDeltaCoeffs[bf];
}


//- Gather |phi_f| and |phi_f| psi_f over a patch, for S55.2's flux-weighted
//  patch mean. Two buffers, one launch, so the two sums see exactly the same
//  |phi| - a second launch recomputing it would be free to differ in the
//  last bit and the ratio would stop being a weighted mean of the values it
//  was formed from.
extern "C" __global__ void fanGatherFluxWeighted
(
    ofscalar* __restrict__ w,
    ofscalar* __restrict__ wpsi,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ psi,
    oflabel start,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const oflabel bf = start + i;

    const ofscalar a = fanAbs_(phi[bf]);
    w[i]    = a;
    wpsi[i] = a*psi[bf];
}


//- Gather a patch's signed flux and, separately, only the part of it that
//  flows INTO the domain (phi < 0 with an outward Sf). S55.3's rack inlet
//  wants the entering mass flow, not the net.
extern "C" __global__ void fanGatherInflow
(
    ofscalar* __restrict__ w,
    ofscalar* __restrict__ wpsi,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ psi,
    oflabel start,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const oflabel bf = start + i;

    const ofscalar f = phi[bf];
    const ofscalar a = (f < 0) ? -f : (ofscalar)0;
    w[i]    = a;
    wpsi[i] = a*psi[bf];
}


//- Copy three reduced scalars into one fan's slot.
//
//  NOT a reduction. `solver::device_sum` writes its answer to element 0 of
//  a buffer, and `fanOperatingPoint` wants the per-patch answers in an array
//  indexed by patch, so one thread moves three numbers. The reduction itself
//  is the existing two-stage one, unmodified - S52.7's whole point is that
//  no new reduction is written for this section.
extern "C" __global__ void fanStoreScalar3
(
    ofscalar* __restrict__ a,
    ofscalar* __restrict__ b,
    ofscalar* __restrict__ c,
    const ofscalar* __restrict__ sa,
    const ofscalar* __restrict__ sb,
    const ofscalar* __restrict__ sc,
    oflabel slot
)
{
    if (OFGPU_TID != 0) return;
    a[slot] = sa[0];
    b[slot] = sb[0];
    c[slot] = sc[0];
}


// ==========================================================================
//  S52.5  The curve, on the device
// ==========================================================================

//  Curve kinds. Mirrored by `crate::fan::CurveKind`, and pinned to these
//  numbers by `fan::tests::curve_kind_values_match_the_device`.
#define OFGPU_FAN_CONSTANT  0
#define OFGPU_FAN_QUADRATIC 1
#define OFGPU_FAN_TABLE     2

//  The largest table a curve may carry. A fixed bound is what lets the scan
//  below be a fixed-trip-count loop over a compile-time maximum, which is
//  what CUDA Graph capture needs (S52.7).
#define OFGPU_FAN_MAX_POINTS 64


//- (S52.13)'s corrections, and the value/slope of the curve at one flow.
//
//  Returns dp (Pa) in *dp and -d(dp)/dQ (Pa per m^3/s) in *slope. BOTH are
//  in Pa here; the caller divides by rho_ref once, at the end, because
//  S52.2's F and S are the kinematic forms and doing the division in one
//  place is what keeps the two consistent.
//
//  The affinity laws are applied by mapping the requested flow BACK to the
//  curve's own speed, evaluating there, and scaling the result:
//
//      dp(Q; rho, N) = dp_curve(Q N_curve/N) (rho/rho_curve) (N/N_curve)^2
//
//  so the slope picks up one more factor of N_curve/N from the chain rule.
OFGPU_DEV void fanCurveAt
(
    ofscalar Q,
    oflabel kind,
    ofscalar dpMax,
    ofscalar qMax,
    const ofscalar* __restrict__ tq,
    const ofscalar* __restrict__ tdp,
    const ofscalar* __restrict__ tm,
    oflabel nPoints,
    ofscalar rhoRatio,      // rho/rho_curve
    ofscalar speedRatio,    // N/N_curve
    ofscalar* dp,
    ofscalar* slope
)
{
    // Map to the curve's own speed. speedRatio > 0 is enforced on the host.
    const ofscalar qc = Q/speedRatio;

    ofscalar v = 0;
    ofscalar s = 0;   // d(dp_curve)/dQ_curve

    if (kind == OFGPU_FAN_CONSTANT)
    {
        v = dpMax;
        s = 0;
    }
    else if (kind == OFGPU_FAN_QUADRATIC)
    {
        // dp = dpMax [1 - Q|Q|/QMax^2], NOT dpMax [1 - (Q/QMax)^2].
        //
        // The textbook form is EVEN in Q, so on the reverse branch it FALLS
        // as the machine is pushed backwards: dp' = -2 dpMax Q/QMax^2 is
        // POSITIVE there, S is negative, and a fan being driven backwards
        // develops ever more pressure pushing the reversal along. That is a
        // positive feedback loop and it destroyed a first draft of
        // `cases/coldAisle.dc.jsonc` - Q went 3.0, -4.6, -33, -90, -1692 in
        // five outer iterations. S52.5 refuses a non-monotone TABLE for
        // exactly this reason; the quadratic has to be written so it cannot
        // happen.
        //
        // Q|Q| is identical to Q^2 for Q >= 0, so the forward branch - and
        // every gate written against it, including S52.12's closed form and
        // the FDS cross-check - is UNCHANGED. On the reverse branch dp now
        // GROWS, which is a fan opposing the flow being forced through it,
        // and S = 2 dpMax |Q|/QMax^2 >= 0 on the whole line.
        v = dpMax*(1 - qc*fanAbs_(qc)/(qMax*qMax));
        s = -2*dpMax*fanAbs_(qc)/(qMax*qMax);
    }
    else
    {
        // ---- monotone Hermite through the table (S52.5) ------------------
        //
        // A FIXED-TRIP-COUNT scan over the compile-time maximum, not a
        // binary search and not a break: the trip count must not depend on
        // the data if the launch is to be graph-capturable. nPoints <= 64,
        // so this is 64 iterations of three compares.
        const ofscalar q0 = tq[0];
        const ofscalar qN = tq[nPoints - 1];

        if (qc <= q0 || qc >= qN)
        {
            // ONE expression for BOTH tails (S52.5).
            //
            //     dp = dp_end + m_end d - k d|d| ,   d = Q - Q_end
            //     S  = -dp'   = -m_end + 2 k |d|
            //
            // The linear part carries the join slope, so dp' is continuous at
            // the end point; the -k d|d| part adds curvature of the SAME sign
            // in both directions, so dp falls faster above free delivery and
            // RISES below shut-off. Both give S > 0 and growing, which is
            // what bounds an excursion instead of leaving the patch as a
            // fixedValue at whatever the end point held - and a fixedValue at
            // shut-off is the stiffest condition a curve has.
            const int e = (qc <= q0) ? 0 : (int)(nPoints - 1);
            const ofscalar mE = tm[e];
            const ofscalar d  = qc - tq[e];
            const ofscalar qref = ofmax_(fanAbs_(qN - q0), (ofscalar)1e-30);
            const ofscalar kA = fanAbs_(mE)/qref;
            const ofscalar kB = fanAbs_(tdp[0])/(qref*qref);
            const ofscalar k  = ofmax_(ofmax_(kA, kB), (ofscalar)1e-300);
            v = tdp[e] + mE*d - k*d*fanAbs_(d);
            s = mE - 2*k*fanAbs_(d);
        }
        else
        {
            ofscalar acc_v = 0;
            ofscalar acc_s = 0;
            for (int k = 0; k < OFGPU_FAN_MAX_POINTS - 1; ++k)
            {
                const int kk = (k < nPoints - 1) ? k : (int)(nPoints - 2);
                const ofscalar a = tq[kk];
                const ofscalar b = tq[kk + 1];
                const bool in = (k < nPoints - 1) && (qc >= a) && (qc < b);

                const ofscalar h = b - a;
                const ofscalar t = (qc - a)/h;
                const ofscalar t2 = t*t;
                const ofscalar t3 = t2*t;

                // Hermite basis (Numerical Recipes S3.3).
                const ofscalar h00 =  2*t3 - 3*t2 + 1;
                const ofscalar h10 =    t3 - 2*t2 + t;
                const ofscalar h01 = -2*t3 + 3*t2;
                const ofscalar h11 =    t3 -   t2;

                const ofscalar y0 = tdp[kk],     y1 = tdp[kk + 1];
                const ofscalar m0 = tm[kk],      m1 = tm[kk + 1];

                const ofscalar val = h00*y0 + h10*h*m0 + h01*y1 + h11*h*m1;

                const ofscalar d00 = ( 6*t2 - 6*t)/h;
                const ofscalar d10 = ( 3*t2 - 4*t + 1)/h;
                const ofscalar d01 = (-6*t2 + 6*t)/h;
                const ofscalar d11 = ( 3*t2 - 2*t)/h;
                const ofscalar der = d00*y0 + d10*h*m0 + d01*y1 + d11*h*m1;

                acc_v += in ? val : (ofscalar)0;
                acc_s += in ? der : (ofscalar)0;
            }
            v = acc_v;
            s = acc_s;
        }
    }

    const ofscalar sp2 = speedRatio*speedRatio;
    *dp    = v*rhoRatio*sp2;
    // d/dQ = d/dQ_curve * dQ_curve/dQ = s * (1/speedRatio)
    *slope = -s*rhoRatio*sp2/speedRatio;   // S = -d(dp)/dQ
}


// ==========================================================================
//  S52.3  The operating point and the triple
// ==========================================================================

//- One thread per fan patch: read (Q, Phi, SIGMA_D), under-relax the
//  operating point (S52.14), evaluate the curve, and write the three patch
//  scalars the stamp needs.
//
//  out[0] = fr        = 1/(1 + S SIGMA_D)                          (S52.10)
//  out[1] = refValue  = c + S Phi,  c = p_a - sigma F* - S Q*      (S52.11)
//  out[2] = Q*        the relaxed operating point, carried to the next call
//  out[3] = S         reported
//  out[4] = dp        reported, Pa
//  out[5] = Q         the raw patch flow this iteration, reported
//
//  `first` is 1 on the very first update, where there is no previous
//  operating point to relax against and Q* is seeded from Q directly.
//  Without that seed the first iteration would linearise about whatever Q*
//  was initialised to, which is a different (and arbitrary) fixed point to
//  start from - the answer is the same at convergence but the path is not,
//  and a path that depends on an uninitialised value is not reproducible.
extern "C" __global__ void fanOperatingPoint
(
    ofscalar* __restrict__ out,        // [6*nPatches]
    const ofscalar* __restrict__ qSum, // [nPatches]
    const ofscalar* __restrict__ phSum,
    const ofscalar* __restrict__ sdSum,
    const oflabel*  __restrict__ kind,
    const ofscalar* __restrict__ dpMax,
    const ofscalar* __restrict__ qMax,
    const ofscalar* __restrict__ pAmb,     // kinematic, m^2/s^2
    const ofscalar* __restrict__ sigma,    // +1 outflow, -1 inflow
    const ofscalar* __restrict__ rhoRatio,
    const ofscalar* __restrict__ speedRatio,
    const ofscalar* __restrict__ alpha,    // S52.14 under-relaxation
    const ofscalar* __restrict__ qSeed,    // sigma * free delivery
    const ofscalar* __restrict__ tq,       // [nPatches*OFGPU_FAN_MAX_POINTS]
    const ofscalar* __restrict__ tdp,
    const ofscalar* __restrict__ tm,
    const oflabel*  __restrict__ nPoints,
    ofscalar rhoRef,
    oflabel first,
    oflabel nPatches
)
{
    const oflabel j = OFGPU_TID;
    if (j >= nPatches) return;

    const ofscalar Q   = qSum[j];
    const ofscalar Phi = phSum[j];
    const ofscalar SD  = sdSum[j];

    // ---- S52.14: under-relax the operating point ------------------------
    //
    // On the FIRST update with no flux yet, linearise about FREE DELIVERY
    // rather than about the measured Q = 0. Shut-off is where the pressure is
    // maximal, and on a quadratic curve it is also where S = 0, so the patch
    // would start life as a fixedValue at the full shut-off pressure - the
    // stiffest linearisation the curve has, and one that drove a first draft
    // of `cases/coldAisle.dc.jsonc` to 135 m^3/s through a 35 m^3 room on its
    // second iteration. Free delivery starts at dp = 0 and the iteration
    // walks DOWN. Where a flux DOES exist, the measured Q is used, because
    // then there is a real operating point to linearise about.
    //
    // The branch is on a VALUE, not a trip count: the launch shape is
    // unchanged and the kernel stays graph-capturable (S52.7).
    const ofscalar qOld = out[6*j + 2];
    const ofscalar a    = alpha[j];
    const ofscalar qFirst = (Q == (ofscalar)0) ? qSeed[j] : Q;
    const ofscalar qStar = (first != 0) ? qFirst : (qOld + a*(Q - qOld));

    // ---- the curve at Q_dev = sigma Q* ----------------------------------
    const ofscalar sg = sigma[j];
    ofscalar dp = 0, S = 0;
    fanCurveAt(
        sg*qStar, kind[j], dpMax[j], qMax[j],
        tq + (size_t)j*OFGPU_FAN_MAX_POINTS,
        tdp + (size_t)j*OFGPU_FAN_MAX_POINTS,
        tm + (size_t)j*OFGPU_FAN_MAX_POINTS,
        nPoints[j], rhoRatio[j], speedRatio[j], &dp, &S);

    // Kinematic. S is already -d(dp)/dQ_dev and is therefore >= 0 on a
    // falling curve WHICHEVER WAY THE PATCH FACES (S52.4's derivation): the
    // direction enters only through c.
    const ofscalar F  = dp/rhoRef;
    ofscalar Sk = S/rhoRef;
    // A curve the host declared monotone cannot produce S < 0, but the
    // extrapolation tails and a f64 round-off at a breakpoint can produce a
    // -0.0. Clamping at zero is what keeps fr in (0, 1]; it is not a
    // fallback for a rising curve, which is refused on the host by name.
    Sk = (Sk > 0) ? Sk : (ofscalar)0;

    const ofscalar c = pAmb[j] - sg*F - Sk*qStar;

    // (S52.10)/(S52.11). At Sk == 0 exactly: beta = 0.0, fr = 1.0/1.0 = 1.0
    // and refValue = c + 0.0*Phi = c - bitwise the fixedValue triple.
    const ofscalar beta = Sk*SD;
    out[6*j + 0] = (ofscalar)1/(1 + beta);
    out[6*j + 1] = c + Sk*Phi;
    out[6*j + 2] = qStar;
    out[6*j + 3] = Sk;
    out[6*j + 4] = dp;
    out[6*j + 5] = Q;
}


//- One thread per fan-patch face: stamp (fr, refValue, refGrad).
//
//  Every face of one patch gets the SAME three numbers, which is (S52.11)'s
//  whole content - the fan sets one pressure for the patch, not a profile.
extern "C" __global__ void fanStampTriple
(
    ofscalar* __restrict__ fr,
    ofscalar* __restrict__ refValue,
    ofscalar* __restrict__ refGrad,
    const ofscalar* __restrict__ out,   // [6*nPatches]
    oflabel patch,
    oflabel start,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const oflabel bf = start + i;

    fr[bf]       = out[6*patch + 0];
    refValue[bf] = out[6*patch + 1];
    refGrad[bf]  = 0;
}


// ==========================================================================
//  S53  The porous jump
// ==========================================================================

//- (S53.2)'s resistance at one face.
//
//      R = ( r_visc + r_inert |phi|/a ) / a
//
//  r_visc = nu t_m/alpha, r_inert = C2 t_m/2 = K/2. Both are non-negative by
//  construction on the host, so R >= 0 with no sign branch - the same
//  argument S18 makes for the volumetric drag.
OFGPU_DEV ofscalar fanJumpR(ofscalar rVisc, ofscalar rInert, ofscalar phi, ofscalar area)
{
    const ofscalar u = fanAbs_(phi)/area;
    return (rVisc + rInert*u)/area;
}


//- S53.2: divide the THREE arrays that carry rAU_f into the pressure
//  equation by the same (1 + R D_f), on the listed INTERNAL faces.
//
//  D_f is taken from the UNMODIFIED gammaMagSf, so the kernel is idempotent
//  in the sense that matters: it reads what rhie_chow just wrote and is
//  called exactly once per corrector.
//
//  Dividing only gammaMagSf - which is what the design note this was written
//  from names - would leave the sheet resisting the pressure gradient but
//  not the momentum flux through it, which is not (S53.4).
extern "C" __global__ void fanJumpInternal
(
    ofscalar* __restrict__ gammaMagSf,   // rAU_f |Sf|      : the matrix
    ofscalar* __restrict__ rauf,         // rAU_f           : the corrector
    ofscalar* __restrict__ phiHbyA,      // phi_HbyA        : the RHS
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ magSf,
    const ofscalar* __restrict__ deltaCoeffs,
    const oflabel*  __restrict__ face,
    const ofscalar* __restrict__ rVisc,
    const ofscalar* __restrict__ rInert,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const oflabel f = face[i];

    const ofscalar g = gammaMagSf[f];
    const ofscalar D = g*deltaCoeffs[f];
    const ofscalar R = fanJumpR(rVisc[i], rInert[i], phi[f], magSf[f]);

    // R == 0 gives 1 + 0*D == 1.0 and x/1.0 == x BITWISE, for all three.
    const ofscalar den = 1 + R*D;

    gammaMagSf[f] = g/den;
    rauf[f]       = rauf[f]/den;
    phiHbyA[f]    = phiHbyA[f]/den;
}


//- S53.3: the boundary form. The same denominator, applied to the boundary
//  arrays, plus the Robin triple against the plenum pressure.
//
//  This is S52's triple with S SIGMA_D -> R D_f, which is why it lives in
//  the same translation unit: one algebra, two entry points.
extern "C" __global__ void fanJumpBoundary
(
    // Read, not written: `D` is formed from the assembly's own coefficient so
    // the two cannot drift (S53.3).
    const ofscalar* __restrict__ bGammaMagSf,
    const ofscalar* __restrict__ brauf,
    ofscalar* __restrict__ bPhiHbyA,
    ofscalar* __restrict__ fr,
    ofscalar* __restrict__ refValue,
    ofscalar* __restrict__ refGrad,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ bMagSf,
    const ofscalar* __restrict__ bDeltaCoeffs,
    const oflabel*  __restrict__ face,
    const ofscalar* __restrict__ rVisc,
    const ofscalar* __restrict__ rInert,
    const ofscalar* __restrict__ pPlenum,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const oflabel bf = face[i];

    const ofscalar g = bGammaMagSf[bf];
    const ofscalar D = g*bDeltaCoeffs[bf];
    const ofscalar R = fanJumpR(rVisc[i], rInert[i], phi[bf], bMagSf[bf]);
    (void)brauf;

    const ofscalar den = 1 + R*D;

    // ONLY phi_HbyA is divided here, and that is the whole difference from
    // the internal form.
    //
    // At a BOUNDARY the resistance is carried by `fr`, and `fr` is already a
    // factor in both places the coefficient reaches: the assembly forms
    // `bGammaMagSf * bDeltaCoeffs * fr` and the flux corrector forms
    // `rauf_b * |Sf| * fr * Delta`. Dividing the coefficient as well would
    // apply `1/(1 + R D)` TWICE and give an effective conductance of
    // `D/(1 + R D)^2` - a tile more restrictive than the one the case asked
    // for, by exactly the factor `1 + R D`. An early draft did that, and
    // S53.7's series-law gate for the boundary form is what now catches it;
    // the gate for the INTERNAL form could not, because there `fr` does not
    // exist and all three arrays genuinely must be divided (S53.2).
    bPhiHbyA[bf] = bPhiHbyA[bf]/den;

    // R == 0 -> den == 1.0 -> fr == 1.0 exactly: a plain fixedValue at the
    // plenum pressure, bitwise (S53.3).
    fr[bf]       = (ofscalar)1/den;
    refValue[bf] = pPlenum[i];
    refGrad[bf]  = 0;
}


// ==========================================================================
//  S54  Psychrometrics, and the virtual temperature
// ==========================================================================

//  Hyland & Wexler (1983) / ASHRAE Fundamentals eq. (5)/(6). Mirrored
//  EXACTLY by `crate::psychro::p_ws` on the host, and the two are pinned
//  together to 1e-14 by `psychro::tests::the_device_mirrors_the_host`.
#define OFGPU_HW_C1  (-5.6745359e3)
#define OFGPU_HW_C2  ( 6.3925247)
#define OFGPU_HW_C3  (-9.677843e-3)
#define OFGPU_HW_C4  ( 6.2215701e-7)
#define OFGPU_HW_C5  ( 2.0747825e-9)
#define OFGPU_HW_C6  (-9.484024e-13)
#define OFGPU_HW_C7  ( 4.1635019)
#define OFGPU_HW_C8  (-5.8002206e3)
#define OFGPU_HW_C9  ( 1.3914993)
#define OFGPU_HW_C10 (-4.8640239e-2)
#define OFGPU_HW_C11 ( 4.1764768e-5)
#define OFGPU_HW_C12 (-1.4452093e-8)
#define OFGPU_HW_C13 ( 6.5459673)

//- M_w/M_a, Gatley, Herrmann & Kretzschmar (2008).
#define OFGPU_PSY_EPS (0.621945)


OFGPU_DEV ofscalar psyPws(ofscalar T)
{
    const ofscalar l = fanLog_(T);
    if (T < (ofscalar)273.15)
    {
        return fanExp_(OFGPU_HW_C1/T + OFGPU_HW_C2 + OFGPU_HW_C3*T
                   + OFGPU_HW_C4*T*T + OFGPU_HW_C5*T*T*T
                   + OFGPU_HW_C6*T*T*T*T + OFGPU_HW_C7*l);
    }
    return fanExp_(OFGPU_HW_C8/T + OFGPU_HW_C9 + OFGPU_HW_C10*T
               + OFGPU_HW_C11*T*T + OFGPU_HW_C12*T*T*T + OFGPU_HW_C13*l);
}


//- (S54.2)/(S54.4): the whole psychrometric state of a cell, from
//  (T, Y_v, p_atm). One thread per cell, closed form, nothing cached -
//  EnergyPlus caches these because they are hot on a CPU; on a GPU the
//  polynomial is cheaper than the table lookup it would replace (S54).
extern "C" __global__ void psyState
(
    ofscalar* __restrict__ w,      // humidity ratio, kg/kg dry air
    ofscalar* __restrict__ rh,     // relative humidity, 0..1
    ofscalar* __restrict__ h,      // specific enthalpy, kJ/kg dry air
    ofscalar* __restrict__ v,      // specific volume, m^3/kg dry air
    const ofscalar* __restrict__ T,
    const ofscalar* __restrict__ yv,
    ofscalar pAtm,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;

    const ofscalar y = yv[i];
    const ofscalar Ti = T[i];

    // Y_v -> W. Y_v is bounded in [0,1] by S19's clip; the guard is for the
    // exact endpoint, where W is infinite and rh is undefined.
    const ofscalar den = 1 - y;
    const ofscalar W = (den > (ofscalar)0) ? y/den : (ofscalar)0;

    const ofscalar pw  = pAtm*W/(OFGPU_PSY_EPS + W);
    const ofscalar pws = psyPws(Ti);

    const ofscalar t = Ti - (ofscalar)273.15;

    w[i]  = W;
    rh[i] = (pws > (ofscalar)0) ? pw/pws : (ofscalar)0;
    h[i]  = (ofscalar)1.006*t + W*((ofscalar)2501 + (ofscalar)1.86*t);
    v[i]  = (ofscalar)0.287042*Ti*(1 + (ofscalar)1.607858*W)
            /(pAtm/(ofscalar)1000);
}


//- (S54.7): the virtual temperature.
//
//      T_v = T (1 + (1/eps - 1) Y_v)
//
//  At Y_v == 0 this is T*(1.0 + c*0.0) = T*1.0 = T, BITWISE, because
//  multiplication by 1.0 is exact in IEEE 754. That is what makes S54.4's
//  "the default is unmoved" a property of the arithmetic and not of a
//  branch: `momentum::update_buoyancy` is handed this field and is itself
//  unmodified.
extern "C" __global__ void psyVirtualTemperature
(
    ofscalar* __restrict__ tv,
    const ofscalar* __restrict__ T,
    const ofscalar* __restrict__ yv,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    tv[i] = T[i]*(1 + (ofscalar)((ofscalar)1/OFGPU_PSY_EPS - 1)*yv[i]);
}


//- The boundary half of the same map, so the buoyancy flux at a boundary
//  face reads the virtual temperature there too.
extern "C" __global__ void psyVirtualTemperatureBoundary
(
    ofscalar* __restrict__ btv,
    const ofscalar* __restrict__ bT,
    const ofscalar* __restrict__ byv,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    btv[i] = bT[i]*(1 + (ofscalar)((ofscalar)1/OFGPU_PSY_EPS - 1)*byv[i]);
}


//- Supersaturation, per cell, for S54.5's report: max(0, Y_v - Y_v,sat).
//  Reduced by the existing `device_sum`/`device_max_mag`; NOT clipped and
//  NOT condensed, because field-level condensation is a different model and
//  is refused by name.
extern "C" __global__ void psySupersaturation
(
    ofscalar* __restrict__ excess,
    const ofscalar* __restrict__ T,
    const ofscalar* __restrict__ yv,
    ofscalar pAtm,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;

    const ofscalar pws = psyPws(T[i]);
    // W_s = eps pws/(p - pws); Y_v,sat = W_s/(1 + W_s). Guard the
    // super-atmospheric branch, where saturated air is not a state this
    // model describes.
    const ofscalar dp = pAtm - pws;
    const ofscalar ws = (dp > (ofscalar)0) ? OFGPU_PSY_EPS*pws/dp : (ofscalar)1e30;
    const ofscalar ysat = ws/(1 + ws);

    const ofscalar e = yv[i] - ysat;
    excess[i] = (e > 0) ? e : (ofscalar)0;
}


// ==========================================================================
//  S55  Metric contributions
// ==========================================================================

//- S55.1: the two RCI excesses at a list of sample values.
//
//  Gathered into two compact buffers and reduced by `device_sum`. The
//  samples are cell values picked by index, so the SAME kernel serves the
//  `faces` and the `thirds` sample sets - which set was used is decided on
//  the host, at setup, and printed.
extern "C" __global__ void dcRciExcess
(
    ofscalar* __restrict__ hi,
    ofscalar* __restrict__ lo,
    const ofscalar* __restrict__ psi,
    const oflabel*  __restrict__ idx,
    ofscalar tHiRec,
    ofscalar tLoRec,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;

    const ofscalar t = psi[idx[i]];
    const ofscalar a = t - tHiRec;
    const ofscalar b = tLoRec - t;
    hi[i] = (a > 0) ? a : (ofscalar)0;
    lo[i] = (b > 0) ? b : (ofscalar)0;
}


//- S18's heat-release total, gathered per cell: V_P q'''_P.
extern "C" __global__ void dcZoneHeat
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ v,
    const oflabel*  __restrict__ cell,
    ofscalar qVol,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    out[i] = v[cell[i]]*qVol;
}
