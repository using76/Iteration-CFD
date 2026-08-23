// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  sst.cu - the arithmetic that makes Menter's k-omega SST different from the
  two models it blends: the blending functions F1 and F2, the coefficient sets
  they interpolate between, the shear-stress-limited eddy viscosity, the
  production limiter, and the cross-diffusion term.

  Written from:
    Menter, "Two-equation eddy-viscosity turbulence models for engineering
      applications", AIAA J. 32 (1994) 1598-1605
    Menter, Kuntz & Langtry, "Ten years of industrial experience with the SST
      turbulence model", Turbulence, Heat and Mass Transfer 4 (2003) 625-632
    Patankar, "Numerical Heat Transfer and Fluid Flow" (1980) section 4.2 -
      the S = S_u + S_p psi linearisation these coefficients feed
    ofgpu SPEC-LIT.md section 6.3, which tabulates every coefficient below,
      and section 6.6 for the wall distance the blending functions consume
  No GPL-licensed source was consulted.

  ------------------------------------------------------------------------
  WHICH VARIANT
  ------------------------------------------------------------------------

  The 2003 revision, and SPEC-LIT 6.3 asks for the choice to be stated. The
  two published forms differ in exactly two places, and this file takes the
  later one in both:

    nu_t          1994 puts the vorticity magnitude Omega in the limiter's
                  denominator; 2003 puts the strain-rate magnitude
                  S = sqrt(2 symm(grad U) : symm(grad U)). This file uses S,
                  which is what SPEC-LIT 6.3 writes as sqrt(S^2) with
                  S^2 = 2|symm(grad U)|^2.

    production    1994 limits the k production against the dissipation with a
                  large factor on a slightly different expression; 2003 uses
                  min(G, c1 beta* k omega) with c1 = 10, which is what
                  SPEC-LIT 6.3 writes and what is implemented here.

  Everything else - the two coefficient sets, F1, F2, the cross-diffusion
  term - is common to both papers.

  ------------------------------------------------------------------------
  Sign convention
  ------------------------------------------------------------------------

  As everywhere else in this crate, the equation is assembled as

      ddt(psi) + div(phi, psi) - laplacian(Gamma_eff, psi) + Sp*psi = Su

  so a physical sink arrives as a POSITIVE Sp on the diagonal (Patankar's
  rule), and a source of unknown sign is emitted as Susp - a coefficient
  MULTIPLYING psi, which fvm_susp then splits between the diagonal and the
  right-hand side depending on which way it stabilises the matrix.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

#ifdef OFGPU_SINGLE
OFGPU_DEV ofscalar ofsqrt_(ofscalar a) { return sqrtf(a); }
OFGPU_DEV ofscalar oftanh_(ofscalar a) { return tanhf(a); }
#else
OFGPU_DEV ofscalar ofsqrt_(ofscalar a) { return sqrt(a); }
OFGPU_DEV ofscalar oftanh_(ofscalar a) { return tanh(a); }
#endif

//- The floor CD_kw is clipped at inside arg1, from SPEC-LIT 6.3 verbatim.
//  It is not a stabilisation constant we chose: it is what makes the third
//  branch of arg1 finite where grad k and grad omega happen to be orthogonal,
//  and Menter states it as 1e-10.
#define OFGPU_SST_CD_FLOOR ((ofscalar)1e-10)

//- Guards against a division by a quantity the solver bounds away from zero
//  anyway. omega is bounded below by `bound_omega` before anything here runs
//  and y is a length, so these only ever bite on a degenerate input; they are
//  here so that such an input produces a large-but-finite blending argument
//  rather than a NaN that propagates into nu_t and is then invisible.
#define OFGPU_SST_TINY ((ofscalar)1e-300)


// ==========================================================================
//  F1 and F2 - SPEC-LIT 6.3
//
//      CD_kw  = 2 sigma_w2 (grad k . grad omega)/omega
//      CD_kw+ = max(CD_kw, 1e-10)
//
//      arg1 = min( min( max( sqrt(k)/(beta* omega y), 500 nu/(y^2 omega) ),
//                       4 sigma_w2 k/(CD_kw+ y^2) ), 10 )
//      F1   = tanh(arg1^4)
//
//      arg2 = min( max( 2 sqrt(k)/(beta* omega y), 500 nu/(y^2 omega) ), 100 )
//      F2   = tanh(arg2^2)
//
//  The two outer clips (10 and 100) are SPEC-LIT's, and they are not cosmetic:
//  tanh(arg^4) saturates at 1 to within double-precision round-off by arg = 5,
//  so past the clip the function is constant anyway, while arg itself grows
//  like 1/y and would overflow when raised to the fourth power in the first
//  cell of a fine mesh. Clipping the argument rather than the result keeps the
//  expression finite everywhere without changing a single value tanh can
//  distinguish.
// ==========================================================================

extern "C" __global__ void sstBlending
(
    ofscalar* __restrict__ F1,
    ofscalar* __restrict__ F2,
    const ofscalar* __restrict__ k,
    const ofscalar* __restrict__ omega,
    const ofvec3* __restrict__ gradK,
    const ofvec3* __restrict__ gradOmega,
    const ofscalar* __restrict__ y,
    ofscalar nu,
    ofscalar betaStar,
    ofscalar sigmaW2,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar kc = ofmax_(k[c], (ofscalar)0);
    const ofscalar w  = ofmax_(omega[c], OFGPU_SST_TINY);
    const ofscalar yc = ofmax_(y[c], OFGPU_SST_TINY);
    const ofscalar y2 = yc*yc;

    const ofscalar cdkw =
        (ofscalar)2*sigmaW2*dot3(gradK[c], gradOmega[c])/w;
    const ofscalar cdPos = ofmax_(cdkw, OFGPU_SST_CD_FLOOR);

    const ofscalar sqrtK   = ofsqrt_(kc);
    const ofscalar turbLen = sqrtK/(betaStar*w*yc);   //- turbulent length / y
    const ofscalar viscous = (ofscalar)500*nu/(y2*w); //- viscous sublayer term

    ofscalar a1 = ofmax_(turbLen, viscous);
    a1 = ofmin_(a1, (ofscalar)4*sigmaW2*kc/(cdPos*y2));
    a1 = ofmin_(a1, (ofscalar)10);

    const ofscalar a1sq = a1*a1;
    F1[c] = oftanh_(a1sq*a1sq);

    ofscalar a2 = ofmax_((ofscalar)2*turbLen, viscous);
    a2 = ofmin_(a2, (ofscalar)100);

    F2[c] = oftanh_(a2*a2);
}


// ==========================================================================
//  The blended coefficient sets - SPEC-LIT 6.3
//
//      blend(phi) = F1 phi_1 + (1 - F1) phi_2
//
//  Four of them at once because they are read together and share the load of
//  F1. sigma_k and sigma_w are MULTIPLIERS on nu_t, not divisors: SPEC-LIT
//  writes the diffusivity as (nu + blend(sigma) nu_t), which is the SST
//  convention and the opposite of k-epsilon's nu + nu_t/sigma_k. They are
//  therefore handed straight to turbGammaInternalCell, whose rSigma argument
//  is a multiplier for exactly this reason.
// ==========================================================================

extern "C" __global__ void sstBlendCoeffs
(
    ofscalar* __restrict__ sigmaK,
    ofscalar* __restrict__ sigmaW,
    ofscalar* __restrict__ gammaB,
    ofscalar* __restrict__ betaB,
    const ofscalar* __restrict__ F1,
    ofscalar sigmaK1, ofscalar sigmaK2,
    ofscalar sigmaW1, ofscalar sigmaW2,
    ofscalar gamma1,  ofscalar gamma2,
    ofscalar beta1,   ofscalar beta2,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar f = F1[c];
    const ofscalar g = (ofscalar)1 - f;

    sigmaK[c] = f*sigmaK1 + g*sigmaK2;
    sigmaW[c] = f*sigmaW1 + g*sigmaW2;
    gammaB[c] = f*gamma1  + g*gamma2;
    betaB[c]  = f*beta1   + g*beta2;
}


// ==========================================================================
//  The eddy viscosity - SPEC-LIT 6.3
//
//      nu_t = a1 k / max(a1 omega, b1 F2 sqrt(S^2))
//
//  This is the whole point of SST. In a boundary layer under an adverse
//  pressure gradient the ordinary k/omega overpredicts the shear stress,
//  because it lets the stress grow with k while Bradshaw's observation is that
//  tau ~ a1 k. The denominator switches to b1 F2 S wherever the strain rate
//  says the layer is out of equilibrium, and F2 confines that switch to the
//  boundary layer so a free shear flow keeps k/omega.
//
//  Capped at nutMax like every other nu_t in this crate (*DESIGN*, SPEC-LIT
//  6.1), so that a transient with a collapsed omega cannot hand the momentum
//  equation an unbounded viscosity.
// ==========================================================================

extern "C" __global__ void sstNut
(
    ofscalar* __restrict__ nut,
    const ofscalar* __restrict__ k,
    const ofscalar* __restrict__ omega,
    const ofscalar* __restrict__ F2,
    const ofscalar* __restrict__ S,
    ofscalar a1,
    ofscalar b1,
    ofscalar nutMax,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar kc = ofmax_(k[c], (ofscalar)0);
    const ofscalar den = ofmax_(a1*omega[c], b1*F2[c]*S[c]);

    const ofscalar v = (den > (ofscalar)0) ? a1*kc/den : nutMax;

    nut[c] = ofmin_(v, nutMax);
}


// ==========================================================================
//  Production per unit eddy viscosity
//
//      P = dev(2 symm(grad U)) : grad U
//        = (grad U + grad U^T) : grad U - (2/3) tr(grad U)^2
//
//  the same expression fvProduction evaluates, without the nu_t factor.
//
//  The omega equation's source is blend(gamma) (G/nu_t), and forming that as
//  a division by nu_t would be a division by a quantity that is legitimately
//  zero - a laminar patch, the first iteration of a run, a cell where the
//  limiter has just clipped k. So P is computed directly from grad U instead,
//  which is the same number wherever both are defined and finite where the
//  quotient is not.
// ==========================================================================

extern "C" __global__ void sstProductionByNut
(
    ofscalar* __restrict__ P,
    const oftensor* __restrict__ gradU,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const oftensor g = gradU[c];
    const ofscalar tr = g.xx + g.yy + g.zz;

    const ofscalar dd =
          (g.xx + g.xx)*g.xx + (g.xy + g.yx)*g.xy + (g.xz + g.zx)*g.xz
        + (g.yx + g.xy)*g.yx + (g.yy + g.yy)*g.yy + (g.yz + g.zy)*g.yz
        + (g.zx + g.xz)*g.zx + (g.zy + g.yz)*g.zy + (g.zz + g.zz)*g.zz;

    P[c] = dd - ((ofscalar)2/(ofscalar)3)*tr*tr;
}


// ==========================================================================
//  The k equation's sources - SPEC-LIT 6.3
//
//      Dk/Dt = ... + min(G, c1 beta* k omega) - beta* k omega
//
//  so the limited production is an explicit Su and the destruction is an
//  implicit Sp = beta* omega, which is the sign that keeps k positive.
//
//  The production limiter is the 2003 revision's. Without it a stagnation
//  point - where the strain rate is large and the turbulence is not - builds
//  k without bound, and the error is then convected downstream over the whole
//  body. c1 = 10 makes the limit ten times the local dissipation, which is far
//  above anything an equilibrium layer produces and far below what a
//  stagnation point would.
//
//  Susp carries the dilatation term -(2/3)(div u) k of the Favre-averaged
//  form, exactly as turbKOmegaKSources does, and is identically zero for a
//  discretely conservative flux.
// ==========================================================================

extern "C" __global__ void sstKSources
(
    ofscalar* __restrict__ gLim,
    ofscalar* __restrict__ sp,
    ofscalar* __restrict__ susp,
    const ofscalar* __restrict__ G,
    const ofscalar* __restrict__ k,
    const ofscalar* __restrict__ omega,
    const ofscalar* __restrict__ divU,
    ofscalar betaStar,
    ofscalar c1,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar w  = omega[c];
    const ofscalar kc = ofmax_(k[c], (ofscalar)0);

    gLim[c] = ofmin_(G[c], c1*betaStar*kc*w);
    sp[c]   = betaStar*w;
    susp[c] = ((ofscalar)2/(ofscalar)3)*divU[c];
}


// ==========================================================================
//  The omega equation's sources - SPEC-LIT 6.3
//
//      Domega/Dt = ... + blend(gamma) (G/nu_t)
//                      - blend(beta) omega^2
//                      + 2 (1 - F1) sigma_w2 (grad k . grad omega)/omega
//
//  The first two are the ordinary pair: an explicit production and an implicit
//  destruction Sp = blend(beta) omega.
//
//  The third is the cross-diffusion term, and it is the one that needs care.
//  It is what is left of the k-epsilon equation when it is transformed into
//  omega form, so it is the term that MAKES the outer layer k-epsilon, and it
//  can take either sign - grad k and grad omega point in opposite directions
//  through most of a boundary layer. Writing it as
//
//      2 (1 - F1) sigma_w2 (grad k . grad omega)/omega = X * omega ,
//      X = 2 (1 - F1) sigma_w2 (grad k . grad omega)/omega^2
//
//  makes it exactly linear in omega, so it can be emitted as a Susp of -X and
//  handed to Patankar's rule, which puts it on the diagonal when it is a sink
//  and on the right-hand side when it is a source. The sign is -X because a
//  term on the RIGHT of the equation is a NEGATIVE Sp on the left.
// ==========================================================================

extern "C" __global__ void sstOmegaSources
(
    ofscalar* __restrict__ su,
    ofscalar* __restrict__ sp,
    ofscalar* __restrict__ susp,
    const ofscalar* __restrict__ P,
    const ofscalar* __restrict__ omega,
    const ofvec3* __restrict__ gradK,
    const ofvec3* __restrict__ gradOmega,
    const ofscalar* __restrict__ F1,
    const ofscalar* __restrict__ gammaB,
    const ofscalar* __restrict__ betaB,
    ofscalar sigmaW2,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar w = ofmax_(omega[c], OFGPU_SST_TINY);

    su[c] = gammaB[c]*P[c];
    sp[c] = betaB[c]*omega[c];

    const ofscalar cross =
        (ofscalar)2*((ofscalar)1 - F1[c])*sigmaW2
      * dot3(gradK[c], gradOmega[c])/(w*w);

    susp[c] = -cross;
}
