// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  gmtrans.cu - the ONE-equation gamma transition model of Menter, Smirnov,
  Liu & Avancha (2015) on the k-omega SST background (SPEC-LIT S90).

  The device half of src/models/menter_gamma.rs. Every closed form below is
  the CPU twin's twin: same order, same constants, same floors, so the
  host-vs-device test can measure the two against each other and the last
  bit is decided by operation order alone.

  Written from:
    ofgpu SPEC-LIT.md S90 - the equations, the discretisation, the guards,
      the couplings and the gates
    NASA / TMBWG, "Turbulence Modeling Resource - SST-2003-Menter-Gamma-2015",
      <https://tmbwg.github.io/turbmodels/menter_gamma_3eqn.html> - US
      government-authored DOCUMENTATION, not source. FETCHED AND READ
      2026-09-05; every coefficient below is transcribed from it to the
      printed digit, including C_PG3, which the page prints in no equation
      and which SPEC-LIT §90.3 carries on the min[lambda + 0.0681, 0] term.
    Menter, F. R., Smirnov, P. E., Liu, T. & Avancha, R., "A One-Equation
      Local Correlation-Based Transition Model", Flow Turbul. Combust. 95
      (2015) 583-619 - the model's primary reference, PAYWALLED AND NOT
      READ; cited as such and never quoted.
    F. R. Menter, M. Kuntz, R. B. Langtry (2003) - the SST-2003 background
      this model couples to (SPEC-LIT S6.3).
    S. V. Patankar, "Numerical Heat Transfer and Fluid Flow" (1980) S4.2 -
      the linearisation the gamma source below is emitted through.
  No GPL-licensed source was consulted. OpenFOAM's and SU2's transition
  implementations were not opened, searched or quoted.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

/*---------------------------------------------------------------------------*\
  ------------------------------------------------------------------------
  Shape
  ------------------------------------------------------------------------
  One thread per cell, reading only that cell's own entries. No neighbour
  access, no reduction, no atomic, no host decision. The launch sequence is
  fixed - the same six kernels in the same order every outer iteration - so
  a whole transitional `correct` captures into a CUDA graph (SPEC-LIT S81).

  No kernel here loops, and none branches on anything but its own cell's
  values. This model has no fixed-point sweep (the correlation Re_thetac
  needs is in LOCAL quantities, SPEC-LIT §90.1) and no gamma_eff, no
  separation branch: every stamp reads gamma itself, and P_k^lim is what
  the 2015 model puts where LM2009 had gamma_sep.

  This file owns its helpers under the OFGM_ prefix. Each .cu compiles to
  its own CUBIN, so nothing here is shared with cuda/lmtrans.cu and nothing
  there moves.
\*---------------------------------------------------------------------------*/

#ifdef OFGPU_SINGLE
OFGPU_DEV ofscalar ofgmExp_(ofscalar a)  { return expf(a); }
OFGPU_DEV ofscalar ofgmSqrt_(ofscalar a) { return sqrtf(a); }
#else
OFGPU_DEV ofscalar ofgmExp_(ofscalar a)  { return exp(a); }
OFGPU_DEV ofscalar ofgmSqrt_(ofscalar a) { return sqrt(a); }
#endif

//- Integer powers, written out. `pow(x, 4.0)` and (x*x)*(x*x) are not the
//  same number in IEEE-754, and every exponent in this model is an integer.
OFGPU_DEV ofscalar ofgmSq_(ofscalar a)   { return a*a; }
OFGPU_DEV ofscalar ofgmCube_(ofscalar a) { return a*a*a; }
OFGPU_DEV ofscalar ofgmP4_(ofscalar a)   { const ofscalar b = a*a; return b*b; }
OFGPU_DEV ofscalar ofgmP8_(ofscalar a)   { const ofscalar b = a*a, c = b*b; return c*c; }

//- Clamp on the base of every integer power that feeds an exp(-x). exp
//  underflows to exactly 0.0 well before 1e6^8, so clamping the base at 1e6
//  changes no representable result and removes every path to infinity - and
//  hence every path to a NaN through inf*0.
#define OFGM_ARG_CLAMP ((ofscalar)1e6)

//- The floor under a magnitude that only ever divides. Never a branch: the
//  limit each floor produces is named at its use site.
#define OFGM_TINY ((ofscalar)1e-30)

//- Scalar::MIN_POSITIVE, in the precision this unit is compiled for - the
//  threshold Vec3::normalised() divides above and zeroes below.
#ifdef OFGPU_SINGLE
#define OFGM_MIN_POSITIVE ((ofscalar)1.1754943508222875e-38)
#else
#define OFGM_MIN_POSITIVE ((ofscalar)2.2250738585072014e-308)
#endif


// ==========================================================================
//  §90.4 - the local inputs. One OFGPU_DEV function per host twin, in the
//  SAME ORDER as the free functions at the top of src/models/menter_gamma.rs:
//  operation order decides the last bit, and the host-vs-device test reads
//  it. None of these reads a velocity magnitude, which is why the model is
//  Galilean invariant (SPEC-LIT §90.7, Gate 90-G).
// ==========================================================================

//- `Tu_L = min(100 sqrt(2k/3)/(omega d_w), 100)` (90.9), a percentage, with
//  the page's own cap. k is floored at zero and omega and d_w at OFGM_TINY,
//  so a degenerate cell reads the cap rather than an infinity.
OFGPU_DEV ofscalar gmTuL(ofscalar k, ofscalar omega, ofscalar d)
{
    const ofscalar num =
        (ofscalar)100*ofgmSqrt_((ofscalar)2*ofmax_(k, (ofscalar)0)/(ofscalar)3);
    const ofscalar den = ofmax_(omega, OFGM_TINY)*ofmax_(d, OFGM_TINY);
    return ofmin_(num/den, (ofscalar)100);
}

//- (90.10) before (90.8)'s limiter: `-7.57e-3 (dV/dy) d_w^2/nu + 0.0128`.
OFGPU_DEV ofscalar gmLambdaThetaLRaw(ofscalar dvDy, ofscalar d, ofscalar nu)
{
    return -(ofscalar)7.57e-3*dvDy*(d*d)/ofmax_(nu, OFGM_TINY) + (ofscalar)0.0128;
}

//- `lambda_thL` (90.10) with the published limiter `-1 <= lambda_thL <= 1`
//  (90.8) applied.
OFGPU_DEV ofscalar gmLambdaThetaL(ofscalar dvDy, ofscalar d, ofscalar nu)
{
    return ofmin_(ofmax_(gmLambdaThetaLRaw(dvDy, d, nu), -(ofscalar)1), (ofscalar)1);
}

//- The wall normal `n` of (90.11): grad(y)/|grad y|, normalised per cell.
//  Where |grad y| is zero (no wall in the mesh at all) n is zero, so
//  dV/dy is zero and lambda_thL is exactly the clean-air 0.0128 rather
//  than a NaN (SPEC-LIT §90.4).
OFGPU_DEV ofvec3 gmWallNormal(ofvec3 g)
{
    const ofscalar m = ofgmSqrt_(dot3(g, g));
    if (m > OFGM_MIN_POSITIVE)
    {
        return mkvec(g.x/m, g.y/m, g.z/m);
    }
    return mkvec((ofscalar)0, (ofscalar)0, (ofscalar)0);
}

//- `dV/dy ~= n_i g_ij n_j` with g_ij = dU_j/dx_i (90.11), the layout
//  gradU holds. Freezing n under the derivative - dropping grad(n) - is
//  SPEC-LIT §90.4's DESIGN choice, recorded there.
OFGPU_DEV ofscalar gmDvDy(ofvec3 n, oftensor g)
{
    return  n.x*(g.xx*n.x + g.xy*n.y + g.xz*n.z)
          + n.y*(g.yx*n.x + g.yy*n.y + g.yz*n.z)
          + n.z*(g.zx*n.x + g.zy*n.y + g.zz*n.z);
}


// ==========================================================================
//  §90.3 - Re_thetac and F_PG
// ==========================================================================

//- `F_PG(lambda_thL)` (90.7), then (90.8)'s `F_PG = max(F_PG, 0)`. C_PG3
//  sits on the `min[lambda + 0.0681, 0]` term - SPEC-LIT §90.3's reading of
//  the constant the page prints in no equation; at the printed C_PG3 = 0.00
//  the term vanishes.
OFGPU_DEV ofscalar gmFPG
(
    ofscalar lambda,
    ofscalar cPg1, ofscalar cPg2, ofscalar cPg3,
    ofscalar cPg1Lim, ofscalar cPg2Lim
)
{
    ofscalar f;
    if (lambda >= (ofscalar)0)
    {
        f = ofmin_((ofscalar)1 + cPg1*lambda, cPg1Lim);
    }
    else
    {
        f = ofmin_
        (
            (ofscalar)1 + cPg2*lambda
                        + cPg3*ofmin_(lambda + (ofscalar)0.0681, (ofscalar)0),
            cPg2Lim
        );
    }
    return ofmax_(f, (ofscalar)0);
}

//- `Re_thetac(Tu_L, lambda_thL)` (90.6). At lambda = 0 this is the
//  two-constant form C_TU1 + C_TU2 exp(-C_TU3 Tu_L); bounded below by
//  C_TU1 wherever (90.8) holds, so no floor is carried (SPEC-LIT §90.8).
OFGPU_DEV ofscalar gmReThetac
(
    ofscalar tuL, ofscalar lambda,
    ofscalar cTu1, ofscalar cTu2, ofscalar cTu3,
    ofscalar cPg1, ofscalar cPg2, ofscalar cPg3,
    ofscalar cPg1Lim, ofscalar cPg2Lim
)
{
    return cTu1
         + cTu2*ofgmExp_
           (
               -(cTu3*tuL*gmFPG(lambda, cPg1, cPg2, cPg3, cPg1Lim, cPg2Lim))
           );
}


// ==========================================================================
//  §90.2 - the onset and the gates
// ==========================================================================

//- `F_onset = max(F_onset2 - F_onset3, 0)` (90.4), with the page's 2.2,
//  2.0 and 3.5. F_onset2 is the plain min against (88.3)'s fourth power -
//  the 2015 model dropped it (SPEC-LIT §90.2). reThetac carries the same
//  tiny floor the host twin gives its divisor.
OFGPU_DEV ofscalar gmFOnset(ofscalar reV, ofscalar reThetac, ofscalar rT)
{
    const ofscalar fo1 = reV/((ofscalar)2.2*ofmax_(reThetac, OFGM_TINY));
    const ofscalar fo2 = ofmin_(fo1, (ofscalar)2.0);
    const ofscalar fo3 = ofmax_((ofscalar)1 - ofgmCube_(rT/(ofscalar)3.5), (ofscalar)0);
    return ofmax_(fo2 - fo3, (ofscalar)0);
}

//- `F_turb = exp(-(R_T/2)^4)` (90.5) - HALF (88.8)'s R_T. The base is
//  clamped the way every exponential argument here is clamped.
OFGPU_DEV ofscalar gmFTurb(ofscalar rT)
{
    return ofgmExp_(-ofgmP4_(ofmin_(rT/(ofscalar)2, OFGM_ARG_CLAMP)));
}

//- `F_on^lim = min[max(Re_V/(2.2 Re_thetac_lim) - 1, 0), 3]` (90.16) - the
//  onset switch P_k^lim reads, with the coupling's OWN limiter constant.
OFGPU_DEV ofscalar gmFOnLim(ofscalar reV, ofscalar reThetacLim)
{
    return ofmin_
    (
        ofmax_(reV/((ofscalar)2.2*ofmax_(reThetacLim, OFGM_TINY)) - (ofscalar)1, (ofscalar)0),
        (ofscalar)3
    );
}

//- `F_3 = exp(-(R_y/120)^8)` (90.17), (88.6)'s to the character.
OFGPU_DEV ofscalar gmF3(ofscalar rY)
{
    return ofgmExp_(-ofgmP8_(ofmin_(rY/(ofscalar)120, OFGM_ARG_CLAMP)));
}


// ==========================================================================
//  §90.2-§90.4 - every cell-local field the correlations and the SST
//  coupling read, in ONE kernel
//
//  Seven outputs, one launch, one pass over the inputs - (88.3)'s one-
//  kernel shape, with two equations' worth of fields gone: there is no
//  transported Re_theta~ and no gamma_eff (SPEC-LIT §90.1, §90.6).
//
//  Outputs
//    fOnset     max(F_onset2 - F_onset3, 0)   gates gamma production
//    fTurb      exp(-(R_T/2)^4)               gates gamma destruction
//    reThetac   the critical momentum-thickness Reynolds number
//    fOnLim     the onset switch P_k^lim reads
//    f3         exp(-(R_y/120)^8)             what SST's F1 is raised to
//    tuL        Tu_L, a percentage            both are diagnostics a test
//    lambdaThL  lambda_thL, clipped           reads, and the correlations'
//                                             two local inputs
//
//  gamma is an input for completeness of the cell state and NO output
//  reads it: (90.4)-(90.11) are gamma-free, which is what makes them
//  correlations rather than transport (SPEC-LIT §90.1).
// ==========================================================================

extern "C" __global__ void gmFields
(
    ofscalar* __restrict__ fOnset,
    ofscalar* __restrict__ fTurb,
    ofscalar* __restrict__ reThetac,
    ofscalar* __restrict__ fOnLim,
    ofscalar* __restrict__ f3,
    ofscalar* __restrict__ tuL,
    ofscalar* __restrict__ lambdaThL,
    const ofscalar* __restrict__ gamma,
    const ofscalar* __restrict__ k,
    const ofscalar* __restrict__ omega,
    const ofscalar* __restrict__ s,
    const ofscalar* __restrict__ omegaMag,
    const ofscalar* __restrict__ y,
    const ofvec3*   __restrict__ gradY,
    const oftensor* __restrict__ gradU,
    ofscalar nu,
    ofscalar cTu1,
    ofscalar cTu2,
    ofscalar cTu3,
    ofscalar cPg1,
    ofscalar cPg2,
    ofscalar cPg3,
    ofscalar cPg1Lim,
    ofscalar cPg2Lim,
    ofscalar reThetacLim,
    oflabel  nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar kc = ofmax_(k[c], (ofscalar)0);
    const ofscalar w  = ofmax_(omega[c], OFGM_TINY);
    const ofscalar d  = y[c];
    const ofscalar sc = s[c];
    const ofscalar om = omegaMag[c];

    //- lambda_thL, from the wall normal and the wall-normal strain.
    const ofvec3  n   = gmWallNormal(gradY[c]);
    const ofscalar lam = gmLambdaThetaL(gmDvDy(n, gradU[c]), d, nu);

    //- The two correlation inputs, then Re_thetac from them.
    const ofscalar tu = gmTuL(kc, w, d);
    const ofscalar rc = gmReThetac
    (
        tu, lam, cTu1, cTu2, cTu3, cPg1, cPg2, cPg3, cPg1Lim, cPg2Lim
    );

    //- The three wall-scaled Reynolds numbers, and R_T. R_T reads nu*omega,
    //  NOT nu_t: it is k/(nu omega), the turbulence Reynolds number (the
    //  transcription warning is (88.3)'s and it applies here verbatim).
    const ofscalar rT = kc/(nu*w);
    const ofscalar rV = sc*d*d/nu;
    const ofscalar rY = d*ofgmSqrt_(kc)/nu;

    fOnset[c]    = gmFOnset(rV, rc, rT);
    fTurb[c]     = gmFTurb(rT);
    reThetac[c]  = rc;
    fOnLim[c]    = gmFOnLim(rV, reThetacLim);
    f3[c]        = gmF3(rY);
    tuL[c]       = tu;
    lambdaThL[c] = lam;
}


// ==========================================================================
//  §90.5 - the gamma equation's source, and the split it is emitted through
//
//      P_gamma - E_gamma = (A + B) gamma - (A + B c_e2) gamma^2     (90.12)
//      A = F_length S F_onset >= 0,   B = c_a2 Omega F_turb >= 0
//
//  emitted, under the crate's sign convention
//  `ddt + div - laplacian + Sp psi = Su`, as
//
//      Su   = 0                      the equation has no explicit source
//      Sp   = (A + B c_e2) gamma     a sink, diagonal, >= 0 at every state
//      Susp = -(A + B)               a source proportional to gamma, which
//                                    fvm_susp moves back with Patankar's rule
//
//  The split never divides by gamma anywhere, which is what makes gamma = 0
//  an honest part of the operating envelope - the absorbing state
//  (SPEC-LIT §90.5) - rather than a singularity.
// ==========================================================================

extern "C" __global__ void gmGammaSources
(
    ofscalar* __restrict__ su,
    ofscalar* __restrict__ sp,
    ofscalar* __restrict__ susp,
    const ofscalar* __restrict__ gamma,
    const ofscalar* __restrict__ fOnset,
    const ofscalar* __restrict__ fTurb,
    const ofscalar* __restrict__ s,
    const ofscalar* __restrict__ omegaMag,
    ofscalar fLength,
    ofscalar ca2,
    ofscalar ce2,
    oflabel  nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar g = gamma[c];
    const ofscalar a = fLength*s[c]*fOnset[c];
    const ofscalar b = ca2*omegaMag[c]*fTurb[c];

    su[c]   = (ofscalar)0;
    sp[c]   = (a + b*ce2)*g;
    susp[c] = -(a + b);
}


// ==========================================================================
//  §90.6 - the production that is not SST's
//
//  (90.13): `P_k = nu_t S Omega`, the KATO-LAUNDER production - the page's
//  own note says so - which replaces SST's production in BOTH the k and the
//  omega equation. `g` is what sstKSources will read as the k production
//  and `p` what the omega equation reads as production per unit nu_t
//  (`alpha P` is `alpha g/nu_t`).
//
//  SST's own production limiter `min(G, c1 beta* k omega)` is NOT applied
//  here: `sstKSources` limits whatever production it is handed, and it is
//  handed this one (SPEC-LIT §90.6). Stamping happens BEFORE
//  the omega equation assembles, so the limiter lands on the replaced G.
// ==========================================================================

extern "C" __global__ void gmStampProduction
(
    ofscalar* __restrict__ g,
    ofscalar* __restrict__ p,
    const ofscalar* __restrict__ nut,
    const ofscalar* __restrict__ s,
    const ofscalar* __restrict__ omegaMag,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    g[c] = nut[c]*s[c]*omegaMag[c];
    p[c] = s[c]*omegaMag[c];
}


// ==========================================================================
//  §90.6 - the coupling into SST, by STAMPING, after sstKSources
//
//      gLim <- gamma gLim + P_k^lim         (90.13) on the LIMITED production
//      sp   <- max(gamma, 0.1) sp           (90.14)
//
//      P_k^lim = 5 C_k max(gamma - 0.2, 0)(1 - gamma) F_on^lim
//                * max(3 C_sep nu - nu_t, 0) S Omega                (90.15)
//
//  Every stamp reads gamma ITSELF - this model has no gamma_eff and no
//  separation branch; P_k^lim is what replaces them (SPEC-LIT §90.2). No
//  upper clamp on either stamp: gamma never leaves its own bounds (90.8),
//  and the model has no gamma_sep that could push it past 1.
// ==========================================================================

extern "C" __global__ void gmStampKSources
(
    ofscalar* __restrict__ gLim,
    ofscalar* __restrict__ sp,
    const ofscalar* __restrict__ gamma,
    const ofscalar* __restrict__ fOnLim,
    const ofscalar* __restrict__ nut,
    const ofscalar* __restrict__ s,
    const ofscalar* __restrict__ omegaMag,
    ofscalar nu,
    ofscalar ck,
    ofscalar csep,
    oflabel  nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar g  = gamma[c];
    const ofscalar pk = (ofscalar)5*ck*ofmax_(g - (ofscalar)0.2, (ofscalar)0)
                            *((ofscalar)1 - g)
                            *fOnLim[c]
                            *ofmax_((ofscalar)3*csep*nu - nut[c], (ofscalar)0)
                            *s[c]
                            *omegaMag[c];

    gLim[c] = g*gLim[c] + pk;
    sp[c]   = ofmax_(g, (ofscalar)0.1)*sp[c];
}


//- F_1 = max(F_1,SST, F_3) (90.17). Stamped between `sstBlending` and
//  `sstBlendCoeffs`, so the four blended coefficient fields are built from
//  the RAISED F_1 - the page's "a modification to SST F_1 blending function
//  is required with the gamma transition model" (SPEC-LIT §90.6).
extern "C" __global__ void gmStampF1
(
    ofscalar* __restrict__ f1,
    const ofscalar* __restrict__ f3,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    f1[c] = ofmax_(f1[c], f3[c]);
}


//- gamma into [lo, hi] (SPEC-LIT §90.8) - OURS, the page publishing no
//  bounds. The two may be equal: that freezes the intermittency, and
//  gammaMin = gammaMax = 1 is Gate 90-R's fully-turbulent limit.
extern "C" __global__ void gmBoundGamma
(
    ofscalar* __restrict__ gamma,
    ofscalar lo,
    ofscalar hi,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    gamma[c] = ofmin_(ofmax_(gamma[c], lo), hi);
}
