// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  marangoni.cu - the tangential interfacial stress grad_s(sigma), SPEC-LIT S87.

  Written from:
    ofgpu SPEC-LIT.md S87 (all of it), S20.4 (the CSF normal term this sits
      beside), S20.1 (n_hat and its epsN), S5.1 (why the normal force is on
      faces and why the tangential one may not be)
    J. U. Brackbill, D. B. Kothe, C. Zemach, J. Comput. Phys. 100 (1992)
      335-354 - the continuum-surface-force regularisation. The tangential
      term is in the original paper; S20.4 implemented only the normal one.
    C. Ma, D. Bothe, Int. J. Multiphase Flow 37 (2011) 1045-1058 - the
      VOF-specific discretisation of the tangential stress
    N. O. Young, J. S. Goldstein, M. J. Block, J. Fluid Mech. 6 (1959)
      350-356 - the closed-form terminal velocity §87.9 gates against
  No GPL-licensed source was consulted.

  ------------------------------------------------------------------------
  Why this is a separate translation unit
  ------------------------------------------------------------------------
  §87.5's bitwise claim is that a case configuring no Marangoni model runs
  EXACTLY the arithmetic it ran before. The strongest available proof of that
  is not a measurement, it is that cuda/vof.cu was not edited: the default
  body force is still vofBodyForceFlux, compiled from an unchanged source
  file into an unchanged cubin, and the host launches it from the same call
  site with the same arguments. Everything S87 adds is in THIS file and is
  reached only through an Option being Some.

  That is also why marBodyForceFlux below duplicates four lines of
  vofBodyForceFlux rather than sharing a __device__ helper with it. Sharing
  would have meant editing vof.cu, and four duplicated lines are cheaper than
  re-deriving the bitwise argument every time nvcc changes its mind about
  contraction.

  ------------------------------------------------------------------------
  Units
  ------------------------------------------------------------------------
  This solver two-phase momentum equation is DIMENSIONAL, not kinematic:
  S20.3 assembles d(rho U)/dt + div(rho_phi, U) - laplacian(mu, U), whose
  rows are in newtons. So the cell force this file produces is a force
  DENSITY,

      [f_M] = (N/m)/m * (1/m) = N/m^3

  and it is NOT divided by rho. momSolveSource multiplies it by V_P.
\*---------------------------------------------------------------------------*/

#include "ofgpu_device.cuh"

// --------------------------------------------------------------------------
//  PatchKind, restated
//
//  These mirror `PatchKind` in src/mesh.rs and are the same numbering
//  cuda/vof.cu and cuda/momentum.cu each restate for themselves - the
//  constants are per-translation-unit defines in this codebase, not a shared
//  header, so a third file restating them is the established pattern rather
//  than a new liberty. cuda/momentum.cu carries the warning in full: a
//  PatchKind must never be compared against a BcKind discriminant, because
//  the two enums number the same names differently and the comparison
//  compiles, runs, and is silently wrong.
//
//  `marangoni_restates_the_prescribed_flux_predicate` in src/vof.rs pins the
//  branch below against vofFluxIsPrescribed by measurement, so the copy
//  cannot drift from the original it was taken from.
// --------------------------------------------------------------------------
#define OFPATCH_EMPTY     2
#define OFPATCH_SYMMETRY  3

//- The component view of a vector, as cuda/momentum.cu defines it: `cmpt` is
//  0, 1 or 2, and anything else selects z, which is what `Vec3::component`
//  does on the host too.
OFGPU_DEV ofscalar marVecCmpt(const ofvec3& v, oflabel c)
{
    return (c == 0) ? v.x : ((c == 1) ? v.y : v.z);
}


// ==========================================================================
//  §87.2  sigma as a field
// ==========================================================================

//- sigma = sigma0 + (d sigma/dT)(T - T0), elementwise.
//
//  Runs over cells and over boundary faces alike: it reads one array and
//  writes one, and knows nothing about which. sigma boundary values are
//  needed because the Green-Gauss gather in fvc_grad_scalar reads them, and
//  the closure is algebraic in T, so there is no boundary CONDITION on sigma
//  to evaluate - only the same line applied to T own boundary values.
//
//  The linear closure is §87.2, and it is the one Young, Goldstein & Block
//  assume. A non-linear sigma(T) would change this kernel and nothing else.
extern "C" __global__ void marSigmaFromT
(
    ofscalar* __restrict__ sigma,
    const ofscalar* __restrict__ T,
    ofscalar sigma0,
    ofscalar dSigmaDT,
    ofscalar T0,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;

    sigma[i] = sigma0 + dSigmaDT*(T[i] - T0);
}


//- The floor: a surface tension that has gone negative is not physics.
//
//  sigma0 + (d sigma/dT)(T - T0) is linear and unbounded below, so a large
//  enough temperature excursion drives it through zero, at which point the
//  continuum problem is ill-posed too - the interface wants infinite area.
//  §87.2 asks for the clip and for the number of clipped cells to be
//  reportable, which is what the second output is: 1 where the floor bit, 0
//  where it did not. The host reduces it. No atomics, one flag per cell.
extern "C" __global__ void marSigmaFloor
(
    ofscalar* __restrict__ sigma,
    ofscalar* __restrict__ clipped,
    ofscalar sigmaMin,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;

    const ofscalar s = sigma[i];
    if (s < sigmaMin)
    {
        sigma[i] = sigmaMin;
        clipped[i] = 1;
    }
    else
    {
        clipped[i] = 0;
    }
}


// ==========================================================================
//  §87.3  The tangential force
// ==========================================================================

//- f_M = ( grad(sigma) - n_hat (n_hat . grad(sigma)) ) |grad(alpha)|
//
//  §87.3 derives this from the interfacial stress balance
//
//      [[ -p I + tau ]] . n_hat = sigma kappa n_hat + grad_s(sigma)
//
//  under the Brackbill, Kothe & Zemach regularisation delta_s ~
//  |grad(alpha)|, n_hat = grad(alpha)/|grad(alpha)|. The first term on the
//  right is the S20.4 face flux and is untouched; the second is this kernel,
//  and it is the reason S87 exists at all - a face SCALAR flux carries one
//  degree of freedom per face, always along Sf, while grad_s(sigma) is by
//  construction orthogonal to the interface normal. No rewrite of phib can
//  hold it.
//
//  epsN is the S20.1 one, passed in from the host as 1e-8/L: eight orders
//  below the interface signal |grad alpha| ~ 1/L and eight orders above the
//  round-off |grad alpha| ~ 1e-16/L of a pure phase. Inside a pure phase
//  n_hat is then a zero vector rather than a random direction, and the |grad
//  alpha| factor is zero to round-off, so f_M is zero twice over.
//
//  The projector is applied with n_hat, a UNIT vector, rather than by the
//  algebraically equal g (g.gs)/|g|^2, because the second form squares the
//  small quantity: at |grad alpha| ~ 1e-16/L it underflows the ratio, while
//  the first form merely divides by epsN.
//
//  ORDER: reads grad(alpha) and grad(sigma), both of which the host has
//  already gathered this iteration. Pure gather, one thread per cell, no
//  atomics; S81 capture stance is unaffected, the launch count being fixed.
extern "C" __global__ void marTangentialForce
(
    ofvec3* __restrict__ fM,
    const ofvec3* __restrict__ gradSigma,
    const ofvec3* __restrict__ gradAlpha,
    ofscalar epsN,
    oflabel n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;

    const ofvec3 g = gradAlpha[c];
    const ofscalar mag = sqrt(dot3(g, g));
    const ofscalar r = 1/(mag + epsN);

    const ofvec3 nh = mkvec(g.x*r, g.y*r, g.z*r);
    const ofvec3 gs = gradSigma[c];
    const ofscalar gn = dot3(nh, gs);

    fM[c] = mkvec
    (
        (gs.x - nh.x*gn)*mag,
        (gs.y - nh.y*gn)*mag,
        (gs.z - nh.z*gn)*mag
    );
}


// ==========================================================================
//  §87.4  The normal term, with sigma a field
// ==========================================================================

//- phi_b = -(g.x)_f |Sf| snGrad(rho) + sigma_f kappa_f |Sf| snGrad(alpha).
//
//  Identical to vofBodyForceFlux in cuda/vof.cu except that sigma arrives per
//  face instead of as one scalar. The whole of the S20.4 and S5.1 reasoning
//  carries over unchanged: the capillary force stays in the balanced-force
//  face representation, built from the same |Sf| snGrad(.) coefficients as
//  the pressure gradient it is balanced against, so the two still cancel face
//  by face and a sealed tank of stratified fluid still stays at rest.
//
//  This kernel runs ONLY when a Marangoni model is configured. §87.5
//  measures why that matters: sigma_f = w sigma_P + (1-w) sigma_N does not
//  reliably return a UNIFORM sigma field unchanged. It always does at
//  w = 1/2, and whether it does elsewhere depends on sigma's mantissa -
//  three quarters of the plausible band is moved by one ulp at some weight -
//  so routing a d sigma/dT = 0 case through here would move a VOF
//  measurement the case never asked to change.
extern "C" __global__ void marBodyForceFlux
(
    ofscalar* __restrict__ phib,
    const ofscalar* __restrict__ snGradRhoMagSf,
    const ofscalar* __restrict__ snGradAlphaMagSf,
    const ofscalar* __restrict__ kappaf,
    const ofscalar* __restrict__ sigmaf,
    const ofvec3* __restrict__ Cf,
    ofscalar gx,
    ofscalar gy,
    ofscalar gz,
    oflabel n
)
{
    const oflabel f = OFGPU_TID;
    if (f >= n) return;

    const ofvec3 c = Cf[f];
    const ofscalar gh = gx*c.x + gy*c.y + gz*c.z;

    phib[f] = -gh*snGradRhoMagSf[f] + sigmaf[f]*kappaf[f]*snGradAlphaMagSf[f];
}


//- The same on the boundary, zero wherever the flux is prescribed.
//
//  The predicate is the vofFluxIsPrescribed one, restated here because the
//  §87.5 by-construction bitwise argument turns on cuda/vof.cu not being
//  edited, and hoisting a shared helper into a header would have been an
//  edit. A wall flux is U_b . Sf and the pressure equation may not move it;
//  giving such a face a body force would put a source into a cell whose flux
//  cannot answer it, and a sealed box would drift.
//
//  `marangoni_restates_the_prescribed_flux_predicate` pins the restatement
//  against vofFluxIsPrescribed so the copy cannot drift from the original.
extern "C" __global__ void marBodyForceFluxBoundary
(
    ofscalar* __restrict__ bphib,
    const ofscalar* __restrict__ snGradRhoMagSf,
    const ofscalar* __restrict__ snGradAlphaMagSf,
    const ofscalar* __restrict__ kappaf,
    const ofscalar* __restrict__ sigmaf,
    const ofvec3* __restrict__ bCf,
    const oflabel* __restrict__ bKind,
    const ofscalar* __restrict__ fr,
    ofscalar gx,
    ofscalar gy,
    ofscalar gz,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    const oflabel k = bKind[i];
    if (k == OFPATCH_EMPTY || k == OFPATCH_SYMMETRY || fr[i] >= (ofscalar)1)
    {
        bphib[i] = 0;
        return;
    }

    const ofvec3 c = bCf[i];
    const ofscalar gh = gx*c.x + gy*c.y + gz*c.z;

    bphib[i] = -gh*snGradRhoMagSf[i] + sigmaf[i]*kappaf[i]*snGradAlphaMagSf[i];
}


// ==========================================================================
//  §87.6  The diagnostic the §87.9 gates read
// ==========================================================================

//- The volume integral of one component of f_M, per cell: V_P f_M,P[cmpt].
//
//  Gate 87-C compares SUM_c V_c f_M,c against the closed-form surface
//  integral of grad_s(sigma) over the interface. The sum itself is the host
//  one - crate::exactsum, the same deterministic reduction every other budget
//  in this project is measured with - so this kernel only forms the per-cell
//  term. One thread per cell, no atomics, nothing order-dependent.
extern "C" __global__ void marForceIntegrand
(
    ofscalar* __restrict__ out,
    const ofvec3* __restrict__ fM,
    const ofscalar* __restrict__ V,
    oflabel cmpt,
    oflabel n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;

    out[c] = V[c]*marVecCmpt(fM[c], cmpt);
}
