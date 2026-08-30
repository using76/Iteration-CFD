// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  sa.cu - the Spalart-Allmaras one-equation model and its negative
  continuation (SPEC-LIT S56).

  Written from:
    ofgpu SPEC-LIT.md S56 - the equation, the three invariants that are not
      the same number, the S~ positivity fix, the log-layer identity that
      verifies the whole model, the negative continuation and the c_n1 bound
    S. R. Allmaras, F. T. Johnson, P. R. Spalart, "Modifications and
      Clarifications for the Implementation of the Spalart-Allmaras
      Turbulence Model", ICCFD7-1902 (2012),
      https://www.iccfd.org/iccfd7/assets/pdf/papers/ICCFD7-1902_paper.pdf -
      a freely distributed conference paper, read in full. Eqs. (1)-(8)
      restate the baseline, (11)-(13) give the S~ positivity fix, (14), (21),
      (22) give SA-neg. THIS is the implementation reference.
    P. R. Spalart, S. R. Allmaras, "A one-equation turbulence model for
      aerodynamic flows", AIAA Paper 92-0439 (1992); La Recherche
      Aerospatiale 1 (1994) 5-21 - the original.
    NASA / Turbulence Modeling Benchmarking Working Group, "Turbulence
      Modeling Resource - The Spalart-Allmaras Turbulence Model",
      https://tmbwg.github.io/turbmodels/spalart.html - US government-authored
      DOCUMENTATION, not source. Read. It states SA-noft2 and SA-neg to the
      printed digit, it is where `r = 10 when S~ = 0 with Omega identically
      zero` comes from, and it publishes nu_t/nu = 0.210438 and 1.294234 at
      the two ends of the recommended far-field range - the two numbers
      saNut is gated against.
    S. V. Patankar, Numerical Heat Transfer and Fluid Flow (1980) S4.2 - the
      S = S_u + S_p psi linearisation the sources are emitted through.
  No GPL-licensed source was consulted. OpenFOAM and SU2 were not opened,
  searched or quoted.

  ------------------------------------------------------------------------
  Shape
  ------------------------------------------------------------------------

  Every kernel here is one thread per cell (or per face for the diffusivity
  pair), reads only that entity's own entries and writes only its own. No
  neighbour scatter, no reduction, no atomic, no order-dependent
  accumulation - so two runs of the same build on the same device are bitwise
  equal and a whole `correct` captures into a CUDA graph unchanged.

  The two branches this model has - the sign of nu~, and the S~ positivity
  fix - are PER-THREAD branches inside one kernel, never a host-side choice
  between two launchers. That is what keeps the launch sequence identical
  whatever the data (SPEC-LIT S56.9).

  ------------------------------------------------------------------------
  Sign convention of the source coefficients
  ------------------------------------------------------------------------

  As turbulence.cu and ke_variants.cu:

      ddt(psi) + div(phi, psi) - laplacian(Gamma_eff, psi) + Sp*psi = Su

  so a physical SINK arrives as a POSITIVE Sp on the diagonal. SPEC-LIT
  (56.15) writes the whole right-hand side of the nu~ equation as A*nu~ and
  emits `susp = -A`, so Patankar's rule decides per cell which side of the
  equation it belongs on. All four sign cases - production-dominant,
  destruction-dominant, the negative branch, and a negative destruction
  bracket under f_t2 - come out of that one line.
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


// ==========================================================================
//  The closed forms, as device inlines
//
//  They are inlines rather than repeated expressions because the host has an
//  independent transcription of each of them (src/models/spalart_allmaras.rs)
//  and the tests compare the two; a copy-paste divergence inside this file
//  would then show up as a host/device mismatch rather than as nothing.
// ==========================================================================

//- f_v1 = chi^3/(chi^3 + c_v1^3)                            SPEC-LIT (56.1)
OFGPU_DEV ofscalar saFv1(ofscalar chi, ofscalar cv1)
{
    const ofscalar c3 = chi*chi*chi;
    return c3/(c3 + cv1*cv1*cv1);
}

//- f_v2 = 1 - chi/(1 + chi f_v1)                            SPEC-LIT (56.3)
OFGPU_DEV ofscalar saFv2(ofscalar chi, ofscalar fv1)
{
    return (ofscalar)1 - chi/((ofscalar)1 + chi*fv1);
}

//- The S~ positivity fix, Allmaras et al. (11)-(13).        SPEC-LIT (56.9)
//
//  Identical to Omega + Sbar wherever Sbar >= -c_v2 Omega, and asymptotes to
//  (1 - c_v3) Omega = 0.1 Omega as Sbar/Omega -> -inf, so S~ is strictly
//  positive wherever Omega is. Without it f_v2 < 0 drives S~ through zero on
//  a coarse mesh and r, g and f_w all follow it.
OFGPU_DEV ofscalar saStilde(ofscalar omega, ofscalar sbar, ofscalar cv2, ofscalar cv3)
{
    if (sbar >= -cv2*omega)
    {
        return omega + sbar;
    }
    const ofscalar num = cv2*cv2*omega + cv3*sbar;
    const ofscalar den = (cv3 - (ofscalar)2*cv2)*omega - sbar;
    return omega + omega*num/den;
}

//- f_w = g[(1 + c_w3^6)/(g^6 + c_w3^6)]^(1/6)               SPEC-LIT (56.4)
//
//  f_w(1) = 1 exactly, which is the log-layer value, and f_w is bounded above
//  by (1 + c_w3^6)^(1/6) = 65^(1/6) = 2.0051747 as r -> inf. r is capped at
//  r_lim = 10 before this is called, so g <= 10 + 0.3(10^6 - 10) = 300007 and
//  g^6 ~ 3e32, comfortably inside double range.
OFGPU_DEV ofscalar saFw(ofscalar r, ofscalar cw2, ofscalar cw3)
{
    const ofscalar r2 = r*r;
    const ofscalar r6 = r2*r2*r2;
    const ofscalar g  = r + cw2*(r6 - r);
    const ofscalar g2 = g*g;
    const ofscalar g6 = g2*g2*g2;
    const ofscalar c6 = cw3*cw3*cw3*cw3*cw3*cw3;
    return g*ofpow_(((ofscalar)1 + c6)/(g6 + c6), (ofscalar)1/(ofscalar)6);
}

//- f_n = (c_n1 + chi^3)/(c_n1 - chi^3) for chi < 0, and exactly 1 otherwise.
//                                                          SPEC-LIT (56.12)
//
//  Exactly 1 on the positive branch is what makes the two variants run the
//  SAME diffusivity kernel: the positive-only model never sees a negative
//  nu~, so the branch costs one comparison and changes nothing (S56.6).
OFGPU_DEV ofscalar saFn(ofscalar chi, ofscalar cn1)
{
    if (chi >= (ofscalar)0) return (ofscalar)1;
    const ofscalar c3 = chi*chi*chi;
    return (cn1 + c3)/(cn1 - c3);
}


// ==========================================================================
//  Section 56.1 - the eddy viscosity
//
//      nu_t = nu~ f_v1 ,   f_v1 = chi^3/(chi^3 + c_v1^3)
//      nu_t = 0                                   wherever nu~ < 0
//
//  The second line is (56.13) and is EXACTLY zero, not a small number: the
//  negative continuation's whole point is that a negative nu~ contributes no
//  eddy viscosity at all.
//
//  The cap at nutMax is *DESIGN*, the same one S6.1 applies for the same
//  reason: an eddy viscosity a hundred thousand times the molecular one is a
//  diverging solve, not a turbulent flow. It is inactive on any converged
//  field.
// ==========================================================================

extern "C" __global__ void saNut
(
    ofscalar* __restrict__ nut,
    const ofscalar* __restrict__ nuTilda,
    ofscalar nu,
    ofscalar cv1,
    ofscalar nutMax,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar nt = nuTilda[c];
    if (nt < (ofscalar)0)
    {
        nut[c] = (ofscalar)0;
        return;
    }
    const ofscalar chi = nt/nu;
    nut[c] = ofmin_(nt*saFv1(chi, cv1), nutMax);
}


// ==========================================================================
//  Section 56.6 - the source coefficients
//
//  SPEC-LIT (56.15). The whole right-hand side is written as A*nu~:
//
//    nu~ >= 0:  A = c_b1 (1 - f_t2) S~
//                 - (c_w1 f_w - (c_b1/kappa^2) f_t2) nu~/dtil^2
//    nu~ <  0:  A = c_b1 (1 - c_t3neg) Omega  +  c_w1 nu~/dtil^2
//
//  and susp = -A is handed to fvm_susp, which sends the stabilising half to
//  the diagonal and the rest to the right-hand side. su carries the one
//  explicit term, (c_b2/sigma)|grad nu~|^2, which is non-negative always.
//
//  d is the TRUE wall distance and is what S~ and r read; dtil is S57's
//  hybrid length scale and appears in the destruction term ALONE. A pure
//  RANS run passes the same buffer for both, so plain SA is the hybrid with
//  the substitution not made rather than a separate code path. Neither
//  pointer is __restrict__ for exactly that reason.
//
//  ct3Pos is the positive branch's c_t3 - zero under SA-noft2, the default -
//  and ct3Neg is the negative branch's, which must exceed 1 for P_n >= 0 and
//  is therefore 1.2 even when ct3Pos is 0. SPEC-LIT S56.5: this is the one
//  place the two branches must not share a constant.
// ==========================================================================

extern "C" __global__ void saSources
(
    ofscalar* __restrict__ su,
    ofscalar* __restrict__ sp,
    ofscalar* __restrict__ susp,
    const ofscalar* __restrict__ nuTilda,
    const ofvec3* __restrict__ gradNuTilda,
    const ofscalar* __restrict__ omega,
    const ofscalar* d,
    const ofscalar* dtil,
    ofscalar nu,
    ofscalar cb1,
    ofscalar cb2,
    ofscalar cv1,
    ofscalar cv2,
    ofscalar cv3,
    ofscalar cw1,
    ofscalar cw2,
    ofscalar cw3,
    ofscalar ct3Pos,
    ofscalar ct3Neg,
    ofscalar ct4,
    ofscalar sigma,
    ofscalar kappa,
    ofscalar rlim,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar nt   = nuTilda[c];
    const ofscalar om   = omega[c];
    const ofvec3   gnt  = gradNuTilda[c];
    const ofscalar dw   = d[c];
    const ofscalar dt   = dtil[c];
    const ofscalar k2d2 = kappa*kappa*dw*dw;

    //- The one explicit source: (c_b2/sigma) |grad nu~|^2, non-negative.
    su[c] = (cb2/sigma)*dot3(gnt, gnt);

    //- Nothing goes on sp: (56.15) routes production AND destruction through
    //  one susp so that the sign branch never changes which launcher runs.
    sp[c] = (ofscalar)0;

    ofscalar A;

    if (nt >= (ofscalar)0)
    {
        const ofscalar chi = nt/nu;
        const ofscalar fv1 = saFv1(chi, cv1);
        const ofscalar fv2 = saFv2(chi, fv1);
        const ofscalar sbar = nt*fv2/k2d2;
        const ofscalar stil = saStilde(om, sbar, cv2, cv3);

        //- r = min(nu~/(S~ kappa^2 d^2), r_lim), and the TMR's own rule for
        //  the S~ = 0 corner: set r = r_lim. Reaching it means Omega is
        //  identically zero and Sbar is negative, where the quotient is 0/0.
        const ofscalar r = (stil > (ofscalar)0)
            ? ofmin_(nt/(stil*k2d2), rlim)
            : rlim;

        const ofscalar fw  = saFw(r, cw2, cw3);
        const ofscalar ft2 = ct3Pos*ofexp_(-ct4*chi*chi);

        A = cb1*((ofscalar)1 - ft2)*stil
          - (cw1*fw - (cb1/(kappa*kappa))*ft2)*nt/(dt*dt);
    }
    else
    {
        //- Allmaras et al. S3.2, SPEC-LIT (56.11). Production on Omega, not
        //  S~; the destruction term changes SIGN and becomes a source pushing
        //  nu~ back toward zero. Both halves of A are negative here (nt < 0),
        //  so susp = -A > 0 and both land on the diagonal - which is the
        //  "energy stable" property Allmaras et al. name as a design goal,
        //  obtained from the ordinary Patankar split rather than a special
        //  case.
        A = cb1*((ofscalar)1 - ct3Neg)*om + cw1*nt/(dt*dt);
    }

    susp[c] = -A;
}


// ==========================================================================
//  Section 56.6 - the face diffusivity, built from the TRANSPORTED field
//
//      Gamma_eff |Sf| = ((nu + nu~ f_n)/sigma) |Sf|
//
//  turbulence.cu's turbGammaInternal/turbGammaBoundary all read nut; this
//  reads nuTilda. Same shape otherwise, and deliberately so: interpolate the
//  DIFFUSIVITY rather than reconstructing it from an interpolated field (the
//  expression is linear in nu~ apart from f_n, so the two differ only where
//  f_n does), and multiply by |Sf| here because that product - not the bare
//  Gamma - is what fvm_laplacian takes.
//
//  f_n is exactly 1 for nu~ >= 0, so the positive-only variant and the
//  negative continuation run the same kernel.
// ==========================================================================

extern "C" __global__ void saGammaInternal
(
    ofscalar* __restrict__ gammaMagSf,
    const ofscalar* __restrict__ nuTilda,
    const ofscalar* __restrict__ weights,
    const ofscalar* __restrict__ magSf,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    ofscalar nu,
    ofscalar sigma,
    ofscalar cn1,
    oflabel nInternalFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nInternalFaces) return;

    const ofscalar w  = weights[f];

    const ofscalar np = nuTilda[owner[f]];
    const ofscalar nn = nuTilda[neighbour[f]];

    const ofscalar gP = (nu + np*saFn(np/nu, cn1))/sigma;
    const ofscalar gN = (nu + nn*saFn(nn/nu, cn1))/sigma;

    gammaMagSf[f] = (w*gP + ((ofscalar)1 - w)*gN)*magSf[f];
}


extern "C" __global__ void saGammaBoundary
(
    ofscalar* __restrict__ bGammaMagSf,
    const ofscalar* __restrict__ nuTildaB,
    const ofscalar* __restrict__ bMagSf,
    ofscalar nu,
    ofscalar sigma,
    ofscalar cn1,
    oflabel nBoundaryFaces
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nBoundaryFaces) return;

    const ofscalar nb = nuTildaB[i];
    bGammaMagSf[i] = ((nu + nb*saFn(nb/nu, cn1))/sigma)*bMagSf[i];
}


// ==========================================================================
//  Section 56.5 - the lower bound, and what it means that it is optional
//
//      nu~ <- max(nu~, 0)
//
//  *DESIGN*, and named as ours, exactly as S6.1's bound_k is. It is launched
//  ONLY by the variants without the negative continuation; with it, nu~ is
//  not bounded at all and (56.11) does the work. The two are different
//  models and the case says which (S56.8) - which is why this is a separate
//  kernel and not a clamp inside saSources.
// ==========================================================================

extern "C" __global__ void saBoundNuTilda
(
    ofscalar* __restrict__ nuTilda,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;
    nuTilda[c] = ofmax_(nuTilda[c], (ofscalar)0);
}


// ==========================================================================
//  Section 56.4 - the log-layer residual, as a field
//
//  The three terms of (56.2) evaluated at a given nu~, Omega, d and the
//  DIFFUSION supplied by the caller (which has it from fvc_laplacian of the
//  same field), written out separately so a failure says WHICH term failed.
//  This is a diagnostic kernel: nothing in `correct` calls it. It exists
//  because S56.4's identity is the model's own defining balance and a gate
//  that reports one number cannot say what moved.
// ==========================================================================

extern "C" __global__ void saLogLayerTerms
(
    ofscalar* __restrict__ prod,
    ofscalar* __restrict__ dest,
    const ofscalar* __restrict__ nuTilda,
    const ofscalar* __restrict__ omega,
    const ofscalar* __restrict__ d,
    ofscalar nu,
    ofscalar cb1,
    ofscalar cv1,
    ofscalar cv2,
    ofscalar cv3,
    ofscalar cw1,
    ofscalar cw2,
    ofscalar cw3,
    ofscalar ct3Pos,
    ofscalar ct4,
    ofscalar kappa,
    ofscalar rlim,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar nt   = ofmax_(nuTilda[c], (ofscalar)0);
    const ofscalar om   = omega[c];
    const ofscalar dw   = d[c];
    const ofscalar k2d2 = kappa*kappa*dw*dw;

    const ofscalar chi  = nt/nu;
    const ofscalar fv1  = saFv1(chi, cv1);
    const ofscalar fv2  = saFv2(chi, fv1);
    const ofscalar sbar = nt*fv2/k2d2;
    const ofscalar stil = saStilde(om, sbar, cv2, cv3);
    const ofscalar r    = (stil > (ofscalar)0)
        ? ofmin_(nt/(stil*k2d2), rlim)
        : rlim;
    const ofscalar fw   = saFw(r, cw2, cw3);
    const ofscalar ft2  = ct3Pos*ofexp_(-ct4*chi*chi);

    prod[c] = cb1*((ofscalar)1 - ft2)*stil*nt;
    dest[c] = (cw1*fw - (cb1/(kappa*kappa))*ft2)*(nt/dw)*(nt/dw);
}
