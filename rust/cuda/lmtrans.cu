// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  lmtrans.cu - the Langtry-Menter gamma-Re_theta transition model, four
  equations on the k-omega SST background (SPEC-LIT S88).

  A transition model exists to say WHERE a boundary layer stops being laminar.
  Everything in this file is in service of one scalar - the intermittency
  gamma, which multiplies SST's own k production - and of the transported
  onset Reynolds number Re_theta~ that the correlations turn into a critical
  momentum-thickness Reynolds number.

  Written from:
    ofgpu SPEC-LIT.md S88 - the equations, the discretisation, the fixed
      point, the guards and the gates
    R. B. Langtry, F. R. Menter, "Correlation-Based Transition Modeling for
      Unstructured Parallelized Computational Fluid Dynamics Codes",
      AIAA Journal 47 (2009) 2894-2906 - THE paper: the 2006 pair withholds
      the correlations and is not enough to write the model.
    NASA / Turbulence Modeling Benchmarking Working Group, "Turbulence
      Modeling Resource - Langtry-Menter 4-equation Transitional SST Model
      (SST-2003-LM2009)",
      <https://tmbwg.github.io/turbmodels/langtrymenter_4eqn.html> - US
      government-authored DOCUMENTATION, not source. FETCHED AND READ while
      writing this file; every coefficient below is transcribed from it to
      the printed digit, and the piecewise breakpoints and the three
      numerical limits (lambda in [-0.1, 0.1], Tu >= 0.027,
      Re_theta_eq >= 20) come from there.
    F. R. Menter, R. B. Langtry, S. R. Likki, Y. B. Suzen, P. G. Huang,
      S. Voelker, "A Correlation-Based Transition Model Using Local
      Variables - Part I: Model Formulation", J. Turbomach. 128 (2006)
      413-422 - the model's structure
    S. V. Patankar, "Numerical Heat Transfer and Fluid Flow" (1980) S4.2 -
      the linearisation every source below is emitted through
  No GPL-licensed source was consulted. OpenFOAM's and SU2's transition
  implementations were not opened, searched or quoted.

  ------------------------------------------------------------------------
  Shape
  ------------------------------------------------------------------------

  One thread per cell, reading only that cell's own entries. No neighbour
  access, no reduction, no atomic. The piecewise polynomials branch
  per-thread on the cell's own Re_theta~, never on the host, so the launch
  sequence is identical every outer iteration and a whole transitional
  `correct` captures into a CUDA graph (SPEC-LIT S81).

  The one loop in the model - the fixed point Re_theta_eq sits inside,
  because the momentum thickness it needs is built from Re_theta_eq itself -
  runs a FIXED number of sweeps, `nSweeps`, passed in as a launch parameter.
  A convergence test would make the sweep count depend on a floating-point
  comparison, which is a warp-divergence problem and, far worse for this
  crate, destroys bitwise reproducibility. SPEC-LIT §88.4.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"


#ifdef OFGPU_SINGLE
OFGPU_DEV ofscalar ofsqrt_(ofscalar a) { return sqrtf(a); }
OFGPU_DEV ofscalar ofexp_(ofscalar a)  { return expf(a); }
OFGPU_DEV ofscalar ofpow_(ofscalar a, ofscalar b) { return powf(a, b); }
#else
OFGPU_DEV ofscalar ofsqrt_(ofscalar a) { return sqrt(a); }
OFGPU_DEV ofscalar ofexp_(ofscalar a)  { return exp(a); }
OFGPU_DEV ofscalar ofpow_(ofscalar a, ofscalar b) { return pow(a, b); }
#endif

//- Integer powers, written out. `pow(x, 4.0)` and `x*x*x*x` are not the same
//  number in IEEE-754, and every exponent in this model except two is an
//  integer. The two that are not - (Tu/1.5)^1.5 and (Tu - 0.5658)^-0.671 -
//  go through `ofpow_` and are marked where they occur.
OFGPU_DEV ofscalar ofsq_(ofscalar a)   { return a*a; }
OFGPU_DEV ofscalar ofcube_(ofscalar a) { return a*a*a; }
OFGPU_DEV ofscalar ofp4_(ofscalar a)   { const ofscalar b = a*a; return b*b; }
OFGPU_DEV ofscalar ofp8_(ofscalar a)   { const ofscalar b = a*a, c = b*b; return c*c; }

//- Clamp on the base of every integer power that feeds an exp(-x). exp
//  underflows to exactly 0.0 well before 1e6^8 = 1e48, so clamping the base
//  at 1e6 changes no representable result and removes every path to
//  infinity - and hence every path to a NaN through inf*0.
#define OFLM_ARG_CLAMP ((ofscalar)1e6)

//- The floor under a magnitude that only ever divides. Never a branch: the
//  limit each floor produces is named at its use site.
#define OFLM_TINY ((ofscalar)1e-30)


// ==========================================================================
//  §88.2 - the two published correlations, digit for digit
//
//  Re_thetac is the CRITICAL momentum-thickness Reynolds number - where the
//  intermittency starts to grow - and Re_theta~ is the TRANSITION-ONSET one
//  the second transport equation carries. They are not the same number and
//  Re_thetac < Re_theta~ over the whole fitted range: the correlation is a
//  fit to the distance between "the model starts producing gamma" and "the
//  experiment records transition".
//
//  Both are quoted here in the TMR's own EXPANDED form rather than the
//  paper's nested `Re_tt - f(Re_tt)` form. They are the same polynomial
//  algebraically; `tests::the_two_forms_of_re_thetac_agree` measures what
//  the rearrangement costs in round-off and reports it.
// ==========================================================================

OFGPU_DEV ofscalar lmReThetac(ofscalar r)
{
    if (r <= (ofscalar)1870)
    {
        //- TMR: -396.035e-2 + 10120.656e-4 R - 868.230e-6 R^2
        //       + 696.506e-9 R^3 - 174.105e-12 R^4
        return  -(ofscalar)3.96035
              +  (ofscalar)1.0120656*r
              -  (ofscalar)868.230e-6*r*r
              +  (ofscalar)696.506e-9*r*r*r
              -  (ofscalar)174.105e-12*r*r*r*r;
    }
    return r - ((ofscalar)593.11 + (ofscalar)0.482*(r - (ofscalar)1870));
}

OFGPU_DEV ofscalar lmFLength1(ofscalar r)
{
    if (r < (ofscalar)400)
    {
        return  (ofscalar)39.8189
              - (ofscalar)119.270e-4*r
              - (ofscalar)132.567e-6*r*r;
    }
    if (r < (ofscalar)596)
    {
        return  (ofscalar)263.404
              - (ofscalar)123.939e-2*r
              + (ofscalar)194.548e-5*r*r
              - (ofscalar)101.695e-8*r*r*r;
    }
    if (r < (ofscalar)1200)
    {
        return (ofscalar)0.5 - (ofscalar)3.0e-4*(r - (ofscalar)596);
    }
    return (ofscalar)0.3188;
}


// ==========================================================================
//  §88.4 - Re_theta_eq, and the fixed point it sits inside
//
//      Re_eq   = f(Tu, lambda)
//      theta   = Re_eq nu/U
//      lambda  = clamp((theta^2/nu) dU/ds, -0.1, +0.1)
//
//  so Re_eq appears inside its own argument. Langtry & Menter prescribe
//  iterating to convergence; this runs a FIXED `nSweeps` from the
//  zero-pressure-gradient value, for the reason in the file header.
// ==========================================================================

OFGPU_DEV ofscalar lmReThetaEqRaw(ofscalar tu, ofscalar lambda)
{
    //- F(lambda), the pressure-gradient factor.
    ofscalar f;
    if (lambda <= (ofscalar)0)
    {
        const ofscalar e = ofexp_(-ofpow_(tu/(ofscalar)1.5, (ofscalar)1.5));
        f = (ofscalar)1
          + ( (ofscalar)12.986*lambda
            + (ofscalar)123.66*lambda*lambda
            + (ofscalar)405.689*lambda*lambda*lambda)*e;
    }
    else
    {
        f = (ofscalar)1
          + (ofscalar)0.275*((ofscalar)1 - ofexp_(-(ofscalar)35*lambda))
            *ofexp_(-tu/(ofscalar)0.5);
    }

    ofscalar re;
    if (tu <= (ofscalar)1.3)
    {
        re = ((ofscalar)1173.51 - (ofscalar)589.428*tu + (ofscalar)0.2196/(tu*tu))*f;
    }
    else
    {
        re = (ofscalar)331.50*ofpow_(tu - (ofscalar)0.5658, -(ofscalar)0.671)*f;
    }

    //- TMR: Re_theta_eq >= 20. A numerical limit published with the model.
    return ofmax_(re, (ofscalar)20);
}

OFGPU_DEV ofscalar lmTu(ofscalar k, ofscalar uMag)
{
    const ofscalar tu =
        (ofscalar)100*ofsqrt_((ofscalar)2*ofmax_(k, (ofscalar)0)/(ofscalar)3)
        /ofmax_(uMag, OFLM_TINY);
    //- TMR: Tu >= 0.027. Below it the 0.2196/Tu^2 term runs away.
    return ofmax_(tu, (ofscalar)0.027);
}

OFGPU_DEV ofscalar lmReThetaEq
(
    ofscalar tu,
    ofscalar dUds,
    ofscalar nu,
    ofscalar uMag,
    oflabel  nSweeps
)
{
    ofscalar re = lmReThetaEqRaw(tu, (ofscalar)0);

    for (oflabel i = 0; i < nSweeps; ++i)
    {
        const ofscalar theta = re*nu/ofmax_(uMag, OFLM_TINY);
        ofscalar lambda = theta*theta*dUds/ofmax_(nu, OFLM_TINY);
        lambda = ofmin_(ofmax_(lambda, -(ofscalar)0.1), (ofscalar)0.1);
        re = lmReThetaEqRaw(tu, lambda);
    }

    return re;
}


// ==========================================================================
//  §88.3 - every cell-local field the two equations and the SST coupling
//  read, in ONE kernel
//
//  Eight outputs, one launch, one pass over the inputs. They are written
//  together rather than in eight kernels because every one of them is a
//  closed form of the same handful of cell values, and because SPEC-LIT S81
//  counts launches: a transitional `correct` adds five kernels to SST's, not
//  twelve.
//
//  Outputs
//    fOnset    max(F_onset2 - F_onset3, 0)      gates gamma production
//    fTurb     exp(-(R_T/4)^4)                  gates gamma destruction
//    fLength   the transition-length correlation, sublayer-blended
//    reThetac  the critical momentum-thickness Reynolds number
//    fThetat   the "inside a switched boundary layer" blend
//    gammaEff  max(gamma, gamma_sep)            what SST's k equation sees
//    reThetaEq the fixed point above
//    f3        exp(-(R_y/120)^8)                what SST's F1 is raised to
// ==========================================================================

extern "C" __global__ void lmFields
(
    ofscalar* __restrict__ fOnset,
    ofscalar* __restrict__ fTurb,
    ofscalar* __restrict__ fLength,
    ofscalar* __restrict__ reThetac,
    ofscalar* __restrict__ fThetat,
    ofscalar* __restrict__ gammaEff,
    ofscalar* __restrict__ reThetaEq,
    ofscalar* __restrict__ f3,
    const ofscalar* __restrict__ gamma,
    const ofscalar* __restrict__ reThetat,
    const ofscalar* __restrict__ k,
    const ofscalar* __restrict__ omega,
    const ofscalar* __restrict__ s,
    const ofscalar* __restrict__ omegaMag,
    const ofscalar* __restrict__ y,
    const ofvec3*   __restrict__ u,
    const oftensor* __restrict__ gradU,
    ofscalar nu,
    ofscalar ce2,
    ofscalar s1,
    oflabel  nSweeps,
    oflabel  nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar kc = ofmax_(k[c], (ofscalar)0);
    const ofscalar w  = ofmax_(omega[c], OFLM_TINY);
    const ofscalar d  = y[c];
    const ofscalar sc = s[c];
    const ofscalar om = omegaMag[c];
    const ofscalar rt = reThetat[c];
    const ofscalar g  = gamma[c];

    const ofvec3 uc = u[c];
    const ofscalar uMag = ofsqrt_(dot3(uc, uc));

    //- The three wall-scaled Reynolds numbers, and R_T. R_T reads nu*omega,
    //  NOT nu_t: it is k/(nu omega), the turbulence Reynolds number, and
    //  substituting nu_t/nu for it is the classic transcription error here
    //  because the two agree only where SST's own eddy-viscosity limiter is
    //  inactive.
    const ofscalar rT  = kc/(nu*w);
    const ofscalar rV  = sc*d*d/nu;
    const ofscalar reW = w*d*d/nu;
    const ofscalar rY  = d*ofsqrt_(kc)/nu;

    //- F_length, blended into 40 through the viscous sublayer.
    const ofscalar fSub = ofexp_(-ofsq_(ofmin_(reW/(ofscalar)200, OFLM_ARG_CLAMP)));
    fLength[c] = lmFLength1(rt)*((ofscalar)1 - fSub) + (ofscalar)40*fSub;

    //- Re_thetac, floored so it can only ever divide.
    const ofscalar rc = ofmax_(lmReThetac(rt), OFLM_TINY);
    reThetac[c] = rc;

    //- The onset switch. 2.193 is the ratio of the MAXIMUM vorticity Reynolds
    //  number across a Blasius profile to that profile's momentum-thickness
    //  Reynolds number; it is what makes the strictly local R_V a stand-in
    //  for a quantity that is an integral across the layer.
    const ofscalar fo1 = rV/((ofscalar)2.193*rc);
    const ofscalar fo2 =
        ofmin_(ofmax_(fo1, ofp4_(ofmin_(fo1, OFLM_ARG_CLAMP))), (ofscalar)2);
    const ofscalar fo3 = ofmax_((ofscalar)1 - ofcube_(rT/(ofscalar)2.5), (ofscalar)0);
    fOnset[c] = ofmax_(fo2 - fo3, (ofscalar)0);

    fTurb[c] = ofexp_(-ofp4_(ofmin_(rT/(ofscalar)4, OFLM_ARG_CLAMP)));

    //- F_thetat: 0 in the free stream, 1 inside a boundary layer whose
    //  intermittency has already switched. `delta` enters as the RATIO
    //  d/delta directly, so a quiescent cell - where delta is 0/0 - gives
    //  the ratio 0 and exp(0) = 1 rather than a NaN.
    const ofscalar fWake = ofexp_(-ofsq_(ofmin_(reW/(ofscalar)1e5, OFLM_ARG_CLAMP)));
    const ofscalar dOverDelta = ofmin_
    (
        d*uMag*uMag/ofmax_((ofscalar)375*om*nu*rt, OFLM_TINY),
        OFLM_ARG_CLAMP
    );
    const ofscalar gTerm = (ce2*g - (ofscalar)1)/(ce2 - (ofscalar)1);
    const ofscalar ft = ofmin_
    (
        ofmax_(fWake*ofexp_(-ofp4_(dOverDelta)), (ofscalar)1 - gTerm*gTerm),
        (ofscalar)1
    );
    fThetat[c] = ft;

    //- Separation-induced transition, and the effective intermittency.
    const ofscalar fReattach = ofexp_(-ofp4_(ofmin_(rT/(ofscalar)20, OFLM_ARG_CLAMP)));
    const ofscalar gSep = ofmin_
    (
        s1*ofmax_(rV/((ofscalar)3.235*rc) - (ofscalar)1, (ofscalar)0)*fReattach,
        (ofscalar)2
    )*ft;
    gammaEff[c] = ofmax_(g, gSep);

    //- dU/ds = (u_m u_n/U^2) du_m/dx_n, with gradU laid out g_ij = dU_j/dx_i,
    //  so du_m/dx_n is g_{nm}.
    const oftensor gr = gradU[c];
    const ofscalar u2 = ofmax_(uMag*uMag, OFLM_TINY);
    const ofscalar dUds =
    (
        uc.x*uc.x*gr.xx + uc.x*uc.y*gr.yx + uc.x*uc.z*gr.zx
      + uc.y*uc.x*gr.xy + uc.y*uc.y*gr.yy + uc.y*uc.z*gr.zy
      + uc.z*uc.x*gr.xz + uc.z*uc.y*gr.yz + uc.z*uc.z*gr.zz
    )/u2;

    reThetaEq[c] = lmReThetaEq(lmTu(kc, uMag), dUds, nu, uMag, nSweeps);

    f3[c] = ofexp_(-ofp8_(ofmin_(rY/(ofscalar)120, OFLM_ARG_CLAMP)));
}


// ==========================================================================
//  §88.5 - the intermittency equation's sources
//
//      P_g - E_g = A - A c_e1 gamma - B c_e2 gamma^2 + B gamma
//      A = F_length c_a1 S sqrt(gamma F_onset)
//      B = c_a2 Omega F_turb
//
//  emitted, under the crate's sign convention
//  `ddt + div - laplacian + Sp psi = Su`, as
//
//      Su   = A
//      Sp   = A c_e1 + B c_e2 gamma          both sinks, both >= 0
//      Susp = -B                             a source proportional to gamma
//
//  which is a Patankar split with a non-negative diagonal contribution,
//  rather than the single lumped Susp of -(P_g - E_g)/gamma the design note
//  proposed: that form divides by gamma, and gamma is zero in every cell of
//  a laminar initial field.
// ==========================================================================

extern "C" __global__ void lmGammaSources
(
    ofscalar* __restrict__ su,
    ofscalar* __restrict__ sp,
    ofscalar* __restrict__ susp,
    const ofscalar* __restrict__ gamma,
    const ofscalar* __restrict__ fOnset,
    const ofscalar* __restrict__ fTurb,
    const ofscalar* __restrict__ fLength,
    const ofscalar* __restrict__ s,
    const ofscalar* __restrict__ omegaMag,
    ofscalar ca1,
    ofscalar ca2,
    ofscalar ce1,
    ofscalar ce2,
    oflabel  nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar g = ofmax_(gamma[c], (ofscalar)0);

    const ofscalar a = fLength[c]*ca1*s[c]*ofsqrt_(g*fOnset[c]);
    const ofscalar b = ca2*omegaMag[c]*fTurb[c];

    su[c]   = a;
    sp[c]   = a*ce1 + b*ce2*g;
    susp[c] = -b;
}


// ==========================================================================
//  §88.5 - the Re_theta~ equation's source
//
//      P_tt = c_tt (1/T)(Re_eq - Re_theta~)(1 - F_thetat) ,  T = 500 nu/U^2
//
//  linear in the unknown and sign-definite in both halves, so it splits
//  exactly: Su = c_tt (1/T)(1 - F) Re_eq, Sp = c_tt (1/T)(1 - F).
//
//  `1/T` is formed as U^2/(500 nu) rather than as a reciprocal of T, so a
//  quiescent cell gives EXACTLY zero production instead of dividing by zero.
// ==========================================================================

extern "C" __global__ void lmReThetatSources
(
    ofscalar* __restrict__ su,
    ofscalar* __restrict__ sp,
    const ofscalar* __restrict__ reThetaEq,
    const ofscalar* __restrict__ fThetat,
    const ofvec3*   __restrict__ u,
    ofscalar nu,
    ofscalar ctt,
    oflabel  nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofvec3 uc = u[c];
    const ofscalar invT = dot3(uc, uc)/((ofscalar)500*nu);

    const ofscalar coeff = ctt*invT*((ofscalar)1 - fThetat[c]);

    su[c] = coeff*reThetaEq[c];
    sp[c] = coeff;
}


// ==========================================================================
//  §88.6 - the coupling into SST, by STAMPING
//
//  `cuda/sst.cu` is not touched by this file. `sstKSources` writes its three
//  buffers exactly as it always did and this kernel then multiplies two of
//  them, which is why a case that names no transition model is unmoved BY
//  CONSTRUCTION rather than by a tolerance: with no model attached the
//  kernel is not launched at all.
//
//      P~_k = gamma_eff P_k
//      D~_k = min(max(gamma_eff, 0.1), 1) D_k
//
//  The third buffer, `susp`, carries S6.3's Favre dilatation -(2/3)(div u)k.
//  Langtry & Menter say nothing about it, so it is left alone rather than
//  scaled by a factor they did not write (SPEC-LIT S13.4).
// ==========================================================================

extern "C" __global__ void lmStampKSources
(
    ofscalar* __restrict__ gLim,
    ofscalar* __restrict__ sp,
    const ofscalar* __restrict__ gammaEff,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar ge = gammaEff[c];

    gLim[c] *= ge;
    sp[c]   *= ofmin_(ofmax_(ge, (ofscalar)0.1), (ofscalar)1);
}


//- F_1 = max(F_1,SST, F_3). Stamped between `sstBlending` and
//  `sstBlendCoeffs`, so the four blended coefficient fields are built from
//  the RAISED F_1 - which is the whole point of F_3: it keeps the model on
//  the k-omega branch through a laminar boundary layer, where SST's own F_1
//  would have handed the near-wall region to the transformed k-epsilon.
extern "C" __global__ void lmStampF1
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


// ==========================================================================
//  §88.8 - the two bounds, which are OURS
//
//  gamma into [0, 1] and Re_theta~ at or above 20. Neither is in Langtry &
//  Menter; the second is the same floor the TMR puts on Re_theta_eq, applied
//  to the TRANSPORTED field so that `lmReThetac` can never be handed an
//  argument outside the range its polynomial was fitted over. SPEC-LIT
//  §88.8 says so, and both are case-settable.
// ==========================================================================

extern "C" __global__ void lmBoundGamma
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

extern "C" __global__ void lmBoundReThetat
(
    ofscalar* __restrict__ reThetat,
    ofscalar floorValue,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    reThetat[c] = ofmax_(reThetat[c], floorValue);
}
