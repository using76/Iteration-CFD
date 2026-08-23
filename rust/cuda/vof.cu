// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  vof.cu - volume of fluid, two immiscible phases. Device side.

  Written from:
    C. W. Hirt, B. D. Nichols, J. Comput. Phys. 39 (1981) 201-225 - the volume
      of fluid method and the phase-fraction equation
    O. Ubbink, "Numerical prediction of two fluid systems with sharp
      interfaces", PhD thesis, Imperial College London (1997) - the
      interface-compressed finite-volume form on an unstructured mesh
    H. Rusche, "Computational fluid dynamics of dispersed two-phase flows at
      high phase fractions", PhD thesis, Imperial College London (2002) - the
      compression velocity tied to the local flux
    S. T. Zalesak, J. Comput. Phys. 31 (1979) 335-362 - the flux-corrected
      transport limiter that keeps alpha in [0, 1]
    J. U. Brackbill, D. B. Kothe, C. Zemach, J. Comput. Phys. 100 (1992)
      335-354 - the continuum surface force
    J. H. Ferziger, M. Peric, "Computational Methods for Fluid Dynamics",
      S7.5 - body forces on faces
  and from rust/SPEC-LIT.md S20 (all five subsections), which cites all of the
  above, together with S2.4 for the corrected face gradient and S5.1 for why
  every body force here is applied as a FACE flux.

  No GPL-licensed source was consulted.

  ---------------------------------------------------------------------------
  The two structural rules of cuda/fv.cu hold here without exception:

  1. GATHER, NEVER SCATTER. Every per-cell accumulation walks that cell's own
     faces through the cell -> face CSR. No atomics on a double, one fixed
     summation order, bitwise reproducible.

  2. A kernel handed the MESH's b_kind compares it against OFPATCH_*, never
     against the OFGPU_BC_* of a FIELD's bc_kind. The two enums number their
     shared names differently and mixing them compiles, runs, and is wrong.
        PatchKind   Generic 0  Wall 1  Empty 2  Symmetry 3  Cyclic 4  Proc 5
        BcKind      ... Calculated 4  Empty 5  Symmetry 6  Cyclic 7  ...
     Only vofBodyForceFluxBoundary takes anything from a field, and what it
     takes is `fr`, a number, not a kind.
\*---------------------------------------------------------------------------*/

#include "ofgpu_device.cuh"

// --------------------------------------------------------------------------
//  Patch kinds - PatchKind in src/mesh.rs, NOT BcKind in src/field.rs.
// --------------------------------------------------------------------------
#define OFPATCH_GENERIC   0
#define OFPATCH_WALL      1
#define OFPATCH_EMPTY     2
#define OFPATCH_SYMMETRY  3
#define OFPATCH_CYCLIC    4
#define OFPATCH_PROCESSOR 5

OFGPU_DEV ofscalar vofAbs(ofscalar a) { return a < 0 ? -a : a; }

//- The face where the flux is not the pressure equation's to choose.
//
//  Identical in meaning to momFluxIsPrescribed in cuda/momentum.cu, and
//  deliberately so: this module builds the same phi_HbyA those kernels do,
//  only with gravity and surface tension in place of the plume's buoyancy, so
//  the two must agree face for face about which boundary faces carry a body
//  force at all.
OFGPU_DEV bool vofFluxIsPrescribed(oflabel kind, ofscalar fr)
{
    return kind == OFPATCH_EMPTY
        || kind == OFPATCH_SYMMETRY
        || fr >= (ofscalar)1;
}


// ==========================================================================
//  S20.3  Mixture properties
// ==========================================================================

//- rho = alpha rho1 + (1 - alpha) rho2,  mu = alpha mu1 + (1 - alpha) mu2.
//
//  Volume-weighted for both, exactly as SPEC-LIT S20.3 writes them, and
//  written in that form rather than the algebraically equal
//  `rho2 + alpha (rho1 - rho2)` so that alpha = 1 gives rho1 and alpha = 0
//  gives rho2 to the last bit rather than to within one rounding of a
//  difference.
//
//  Runs over cells and over boundary faces alike - it reads one array and
//  writes two, and knows nothing about which.
extern "C" __global__ void vofMixture
(
    ofscalar* __restrict__ rho,
    ofscalar* __restrict__ mu,
    const ofscalar* __restrict__ alpha,
    ofscalar rho1,
    ofscalar rho2,
    ofscalar mu1,
    ofscalar mu2,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;

    const ofscalar a = alpha[i];
    rho[i] = a*rho1 + (1 - a)*rho2;
    mu[i]  = a*mu1  + (1 - a)*mu2;
}


// ==========================================================================
//  S20.1 / S20.4  The interface normal, on FACES
//
//  One array serves both sections, which is the point of computing it here
//  rather than twice:
//
//    S20.1 needs   n_f . Sf   to build the compression flux
//                  phi_r = c_alpha |phi_f/|Sf|| (n_f . Sf) ;
//    S20.4 needs   div(n_hat) "from the FACE normals, not by taking a cell
//                  divergence of a cell field", and the Gauss divergence of a
//                  face-normalised n_hat is exactly (1/V) sum_f (+-n_f . Sf).
//
//  So `nHatf[f] = n_hat_f . Sf` is computed once and read by both.
//
//  The face gradient is the plain interpolated cell gradient,
//
//      g_f = w grad_P + (1 - w) grad_N
//
//  and NOT that gradient with its normal component replaced by the two-point
//  face difference snGrad(alpha)_f.
//
//  DERIVED, and then measured. SPEC-LIT S20.4 says to normalise on faces and
//  does not say which face gradient to use. Replacing the normal component
//  with the face difference is the natural reading of S2.4 and it is what this
//  kernel did first; it makes the curvature MUCH worse, and the reason is a
//  scale argument. snGrad is a two-point difference, so its error is
//  (h^2/6) d3(alpha)/dn3, and across an interface resolved over about two
//  cells the third derivative of the profile is enormous - the relative error
//  is O((h/w)^2) with w the interface thickness, which at w = 2h is tens of
//  per cent. The interpolated cell gradient is a symmetric four-point estimate
//  of the same quantity and its leading error term partly cancels between the
//  two cells.
//
//  Measured on a smooth radial profile of thickness 2h against the analytic
//  (d - 1)/r, worst relative error over the interface band:
//
//      face-difference normal component   0.81 (2-D)   0.24 (3-D)
//      interpolated gradient              0.11 (2-D)   0.12 (3-D)
//
//  so the "improvement" was a factor of seven the wrong way. The test that
//  found it is `the_curvature_of_a_circular_interface_converges_to_one_over_r`
//  in src/vof.rs, and it is a convergence test rather than a threshold so that
//  a future change of this kind is caught by its ORDER and not by a tuned
//  tolerance.
//
//  `epsN` stabilises the normalisation. SPEC-LIT S20.1 asks for "a small
//  fraction of 1/(mean cell size)" and for its value to be stated: the host
//  passes 1e-8/L with L the cube root of the mean cell volume. Across an
//  interface |grad alpha| ~ 1/L, so epsN is eight orders below the signal;
//  inside a pure phase |grad alpha| is zero to round-off, ~1e-16/L, so epsN is
//  eight orders ABOVE the noise and n_hat comes out as zero rather than as a
//  random direction.
// ==========================================================================

extern "C" __global__ void vofFaceUnitNormal
(
    ofscalar* __restrict__ nHatf,
    const ofvec3* __restrict__ gradAlpha,
    const ofscalar* __restrict__ w,
    const ofvec3* __restrict__ Sf,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    ofscalar epsN,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;

    const ofscalar wf = w[f];
    const ofvec3 gP = gradAlpha[owner[f]];
    const ofvec3 gN = gradAlpha[neighbour[f]];

    const ofvec3 g = mkvec
    (
        wf*gP.x + (1 - wf)*gN.x,
        wf*gP.y + (1 - wf)*gN.y,
        wf*gP.z + (1 - wf)*gN.z
    );

    const ofscalar mag = sqrt(dot3(g, g));

    nHatf[f] = dot3(g, Sf[f])/(mag + epsN);
}


//- The same on a boundary face.
//
//  A cyclic face is an interior face in disguise and gets the full treatment
//  across the couple. Every other boundary face gets ZERO, and that is a
//  modelling statement, not a shortcut: n_hat . Sf = 0 says the interface
//  normal is tangential to the boundary, i.e. the interface meets it at ninety
//  degrees. That is the no-wall-adhesion contact angle, and it is *DESIGN* -
//  SPEC-LIT S20 specifies no contact-angle model, so this solver makes the one
//  choice that adds no unstated physics. A case needing a real contact angle
//  needs a model this file does not have.
//
//  It also keeps the curvature gather honest: a wall-adjacent interface cell
//  then sees curvature from its interior faces alone rather than from a normal
//  the boundary cannot define.
extern "C" __global__ void vofFaceUnitNormalBoundary
(
    ofscalar* __restrict__ bNHatf,
    const ofvec3* __restrict__ gradAlpha,
    const ofscalar* __restrict__ bw,
    const ofvec3* __restrict__ bSf,
    const oflabel* __restrict__ bFaceCells,
    const oflabel* __restrict__ bNbrCell,
    const oflabel* __restrict__ bKind,
    ofscalar epsN,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    if (bKind[i] != OFPATCH_CYCLIC)
    {
        bNHatf[i] = 0;
        return;
    }

    const oflabel N = bNbrCell[i];
    if (N < 0)
    {
        bNHatf[i] = 0;
        return;
    }

    const ofscalar wf = bw[i];
    const ofvec3 gP = gradAlpha[bFaceCells[i]];
    const ofvec3 gN = gradAlpha[N];

    const ofvec3 g = mkvec
    (
        wf*gP.x + (1 - wf)*gN.x,
        wf*gP.y + (1 - wf)*gN.y,
        wf*gP.z + (1 - wf)*gN.z
    );

    const ofscalar mag = sqrt(dot3(g, g));

    bNHatf[i] = dot3(g, bSf[i])/(mag + epsN);
}


//- kappa_P = -div(n_hat)_P = -(1/V_P) sum_f (+- n_hat_f . Sf).
//
//  SPEC-LIT S20.4. The sign is the section's: kappa is positive where the
//  phase-1 region is convex, so sigma kappa grad(alpha) points into it and a
//  drop is squeezed rather than blown apart.
extern "C" __global__ void vofCurvature
(
    ofscalar* __restrict__ kappa,
    const ofscalar* __restrict__ nHatf,
    const ofscalar* __restrict__ bNHatf,
    const ofscalar* __restrict__ V,
    const oflabel* __restrict__ bKind,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofscalar acc = 0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const ofscalar v = nHatf[cfFace[j]];
        acc += cfOwn[j] ? v : -v;
    }

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        const oflabel b = bcfFace[j];
        if (bKind[b] == OFPATCH_EMPTY) continue;
        acc += bNHatf[b];
    }

    kappa[c] = -acc/V[c];
}


// ==========================================================================
//  S20.1  The compression flux
// ==========================================================================

//- phi_r = c_alpha |phi_f / |Sf|| (n_f . Sf).
//
//  SPEC-LIT S20.1 verbatim. The magnitude is tied to the local flux, so the
//  compression can never exceed the flow it corrects (Rusche 2002): where the
//  fluid is at rest the interface is not being smeared and there is nothing to
//  compress.
//
//  c_alpha = 0 switches it off, 1 is conservative compression, > 1 enhances
//  it. Whatever it is, the term is multiplied by alpha_f (1 - alpha_f) in
//  vofAlphaFlux and so vanishes identically in both pure phases.
extern "C" __global__ void vofCompressionFlux
(
    ofscalar* __restrict__ phir,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ nHatf,
    const ofscalar* __restrict__ magSf,
    ofscalar cAlpha,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;

    const ofscalar ms = magSf[f];
    const ofscalar rms = ms > 0 ? 1/ms : (ofscalar)0;

    phir[f] = cAlpha*vofAbs(phi[f])*rms*nHatf[f];
}


// ==========================================================================
//  S20.2  Flux-corrected transport (Zalesak 1979)
// ==========================================================================

//- Steps 1 to 3: the low-order flux, and the antidiffusive flux on top of it.
//
//      phi_L = phi_f alpha_upwind                      bounded, diffusive
//      phi_H = phi_f alpha_f + phi_r alpha_f(1 - alpha_f)
//      A     = phi_H - phi_L
//
//  alpha_f is the linear interpolate. Any high-order face value would do -
//  that is the whole point of FCT, the limiter and not the interpolation is
//  what enforces boundedness - and the linear one is chosen because it is the
//  least diffusive of them and carries no gradient of its own to evaluate.
//
//  The compression term is formed as alpha_f (1 - alpha_f) from the
//  INTERPOLATED face value, not as the interpolate of the cell product. Across
//  a sharp interface alpha_P = 0 and alpha_N = 1 make the cell product zero in
//  both cells, so interpolating it would switch the compression off exactly
//  where SPEC-LIT S20.1 wants it hardest; forming it at the face gives the
//  maximum 1/4 there instead.
extern "C" __global__ void vofAlphaFlux
(
    ofscalar* __restrict__ phiL,
    ofscalar* __restrict__ anti,
    const ofscalar* __restrict__ alpha,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ phir,
    const ofscalar* __restrict__ w,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;

    const ofscalar aP = alpha[owner[f]];
    const ofscalar aN = alpha[neighbour[f]];
    const ofscalar ph = phi[f];

    const ofscalar lo = (ph >= 0) ? ph*aP : ph*aN;

    const ofscalar wf = w[f];
    const ofscalar af = wf*aP + (1 - wf)*aN;
    const ofscalar hi = ph*af + phir[f]*af*(1 - af);

    phiL[f] = lo;
    anti[f] = hi - lo;
}


//- The boundary half. Upwind, and with NO antidiffusive part at all.
//
//  There is nothing to correct. Where the flux leaves the domain the upwind
//  value IS the cell value and every scheme agrees; where it enters, the
//  boundary condition supplies the value and no interpolation is involved.
//  Compression is left off a boundary face on purpose: the interface normal
//  there is the contact-angle model this solver does not have (see
//  vofFaceUnitNormalBoundary), and an unbounded guess at it would be corrected
//  by nothing, since a boundary face has only one cell to draw room from.
//
//  An empty patch carries no flux, exactly as everywhere else in this crate.
extern "C" __global__ void vofAlphaFluxBoundary
(
    ofscalar* __restrict__ bPhiL,
    const ofscalar* __restrict__ alpha,
    const ofscalar* __restrict__ balpha,
    const ofscalar* __restrict__ bphi,
    const oflabel* __restrict__ bFaceCells,
    const oflabel* __restrict__ bNbrCell,
    const oflabel* __restrict__ bKind,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    const oflabel kind = bKind[i];
    if (kind == OFPATCH_EMPTY)
    {
        bPhiL[i] = 0;
        return;
    }

    const oflabel P = bFaceCells[i];
    const ofscalar ph = bphi[i];

    if (kind == OFPATCH_CYCLIC)
    {
        const oflabel N = bNbrCell[i];
        if (N >= 0)
        {
            bPhiL[i] = (ph >= 0) ? ph*alpha[P] : ph*alpha[N];
            return;
        }
    }

    bPhiL[i] = (ph >= 0) ? ph*alpha[P] : ph*balpha[i];
}


//- The explicit update: alpha -= (dtau/V) sum_f (+-F_f).
//
//  IN PLACE, and it has to be: the low-order solution and each of the limiter
//  iterations are the same arithmetic applied to a different flux, and holding
//  a separate old level for each would be three extra copies of the field for
//  no arithmetic difference. Each thread reads and writes only its own cell.
extern "C" __global__ void vofAdvance
(
    ofscalar* __restrict__ alpha,
    const ofscalar* __restrict__ F,
    const ofscalar* __restrict__ bF,
    const ofscalar* __restrict__ V,
    const oflabel* __restrict__ bKind,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    ofscalar dtau,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofscalar acc = 0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const ofscalar v = F[cfFace[j]];
        acc += cfOwn[j] ? v : -v;
    }

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        const oflabel b = bcfFace[j];
        if (bKind[b] == OFPATCH_EMPTY) continue;
        acc += bF[b];
    }

    alpha[c] -= dtau*acc/V[c];
}


//- Zalesak steps 5 and 6, per cell: how much room is left, and therefore what
//  fraction of the antidiffusive flux reaching this cell it can absorb.
//
//      P+ = sum of the POSITIVE antidiffusive contributions entering the cell
//      P- = sum of the magnitudes of the negative ones
//      Q+ = (alphaMax - alpha) V / dtau        room before alpha exceeds 1
//      Q- = (alpha - alphaMin) V / dtau        room before it drops below 0
//      R+ = min(1, Q+/P+)   or 0 if P+ = 0
//      R- = min(1, Q-/P-)   or 0 if P- = 0
//
//  alphaMin and alphaMax are the GLOBAL bounds 0 and 1, which is what
//  SPEC-LIT S20.2 step 5 asks for in as many words ("the most A can add before
//  alpha exceeds 1, and the most it can remove before alpha drops below 0").
//  Zalesak's paper also offers the tighter local-extremum bound; the global one
//  is used here because [0, 1] is the physical statement - a phase fraction
//  outside it gives a negative density - and because it is the bound the test
//  in SPEC-LIT S22 measures.
//
//  Q is clamped at zero. It is already non-negative for any alpha the previous
//  stage produced, and a round-off-negative Q would otherwise flip the sense of
//  the ratio and let the limiter ADD flux to a cell that has no room.
//
//  Boundary faces contribute nothing: vofAlphaFluxBoundary gives them no
//  antidiffusive flux at all.
extern "C" __global__ void vofLimiterRoom
(
    ofscalar* __restrict__ rPlus,
    ofscalar* __restrict__ rMinus,
    const ofscalar* __restrict__ alpha,
    const ofscalar* __restrict__ anti,
    const ofscalar* __restrict__ V,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    ofscalar alphaMin,
    ofscalar alphaMax,
    ofscalar dtau,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofscalar pPlus = 0;
    ofscalar pMinus = 0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const ofscalar a = anti[cfFace[j]];

        // What the cell GAINS: the owner loses a positive flux, the
        // neighbour gains it.
        const ofscalar in = cfOwn[j] ? -a : a;

        if (in > 0) pPlus += in;
        else        pMinus -= in;
    }

    const ofscalar rdt = V[c]/dtau;
    const ofscalar a = alpha[c];

    ofscalar qPlus = (alphaMax - a)*rdt;
    ofscalar qMinus = (a - alphaMin)*rdt;
    if (qPlus < 0) qPlus = 0;
    if (qMinus < 0) qMinus = 0;

    rPlus[c]  = (pPlus  > 0) ? ofmin_((ofscalar)1, qPlus/pPlus)   : (ofscalar)0;
    rMinus[c] = (pMinus > 0) ? ofmin_((ofscalar)1, qMinus/pMinus) : (ofscalar)0;
}


//- Zalesak step 7, per face: the limiter that satisfies BOTH cells.
//
//      A >= 0   flows owner -> neighbour: the owner must have room to give
//               (R-) and the neighbour room to take (R+)
//      A <  0   the other way round
//
//  and then three things at once, because they are the same read:
//
//      dF    = lambda A       what this iteration applies
//      A    -= dF             what is left for the next one
//      phiA += dF             the running total the mass flux must reuse
//
//  Iterating - recomputing the room left after applying the current limiter -
//  tightens the answer towards the least diffusive bounded solution.
//  SPEC-LIT S20.2 marks the count *DESIGN*; the host runs three.
extern "C" __global__ void vofApplyLimiter
(
    ofscalar* __restrict__ dF,
    ofscalar* __restrict__ anti,
    ofscalar* __restrict__ phiAlpha,
    const ofscalar* __restrict__ rPlus,
    const ofscalar* __restrict__ rMinus,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;

    const oflabel P = owner[f];
    const oflabel N = neighbour[f];
    const ofscalar a = anti[f];

    const ofscalar lam = (a >= 0)
        ? ofmin_(rMinus[P], rPlus[N])
        : ofmin_(rPlus[P], rMinus[N]);

    const ofscalar d = lam*a;

    dF[f] = d;
    anti[f] = a - d;
    phiAlpha[f] += d;
}


// ==========================================================================
//  S20.3  The mass flux, from the SAME limited fluxes
// ==========================================================================

//- sum += weight * x. The sub-cycle accumulator.
//
//  SPEC-LIT S20.2: "accumulating the flux so the momentum equation sees a
//  consistent one". The weight is dtau_i/dt, so what accumulates is the
//  time-average of the sub-cycle fluxes over the whole momentum step.
extern "C" __global__ void vofAccumulate
(
    ofscalar* __restrict__ sum,
    const ofscalar* __restrict__ x,
    ofscalar weight,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    sum[i] += weight*x[i];
}


//- rho_phi = rho2 phi + (rho1 - rho2) phi_alpha.
//
//  SPEC-LIT S20.3: the mass flux must come from the same limited fluxes that
//  advanced alpha, "not from re-interpolating rho". This IS that statement:
//  rho = alpha rho1 + (1 - alpha) rho2 is affine in alpha, so the flux of rho
//  is the same affine function of the flux of alpha, and using the accumulated
//  phi_alpha here makes
//
//      (rho - rho0) V/dt + sum_f (+-rho_phi_f) = 0
//
//  hold to round-off for exactly the same reason the alpha update does. If the
//  two disagreed, mass and momentum would be advected inconsistently and the
//  interface would generate velocity out of nothing.
extern "C" __global__ void vofRhoPhi
(
    ofscalar* __restrict__ rhoPhi,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ phiAlpha,
    ofscalar rho1,
    ofscalar rho2,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    rhoPhi[i] = rho2*phi[i] + (rho1 - rho2)*phiAlpha[i];
}


// ==========================================================================
//  |Sf| snGrad(psi) - the face difference both body-force terms are built on
//
//  Deliberately NOT fv::sn_grad_flux. That function rebuilds the boundary half
//  from the (fr, refValue, refGrad) triple, which is right for a field the
//  matrix solves for; rho and mu are DERIVED from alpha and carry no triple of
//  their own, only an evaluated boundary value. The two agree wherever both
//  are defined - fr = 0, refGrad = 0 gives zero from either route, and fr = 1
//  gives Delta_b (refValue - psi_P) from either - and the internal halves are
//  identical arithmetic on identical arrays.
//
//  What matters is that the SAME deltaCoeffs and the SAME |Sf| appear here and
//  in fvm_laplacian's pressure coefficient, because that is what makes the
//  gravity flux and the pressure gradient cancel face by face rather than to
//  truncation error.
// ==========================================================================

extern "C" __global__ void vofSnGradMagSf
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ psi,
    const ofscalar* __restrict__ deltaCoeffs,
    const ofscalar* __restrict__ magSf,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;

    out[f] = deltaCoeffs[f]*(psi[neighbour[f]] - psi[owner[f]])*magSf[f];
}


//- The same on the boundary. An empty patch has no surface integral at all.
extern "C" __global__ void vofSnGradMagSfBoundary
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ psi,
    const ofscalar* __restrict__ bpsi,
    const ofscalar* __restrict__ bDeltaCoeffs,
    const ofscalar* __restrict__ bMagSf,
    const oflabel* __restrict__ bFaceCells,
    const oflabel* __restrict__ bKind,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    if (bKind[i] == OFPATCH_EMPTY)
    {
        out[i] = 0;
        return;
    }

    out[i] = bDeltaCoeffs[i]*(bpsi[i] - psi[bFaceCells[i]])*bMagSf[i];
}


// ==========================================================================
//  S20.4 and S20.5  The body force, on FACES
// ==========================================================================

//- phi_b = -(g.x)_f |Sf| snGrad(rho)  +  sigma kappa_f |Sf| snGrad(alpha).
//
//  The first term is SPEC-LIT S20.5's gravity flux and the second S20.4's
//  continuum surface force, and both are here rather than in a cell field for
//  the reason S5.1 gives about buoyancy: a body force interpolated from cell
//  values checkerboards on a collocated grid exactly as an interpolated
//  pressure gradient does. For surface tension the symptom has its own name -
//  spurious interface currents - and it is the classic CSF failure mode.
//
//  Both arguments arrive as |Sf| snGrad(.), straight out of fv::sn_grad_flux
//  with gamma = 1, so the pressure gradient this is balanced against is built
//  from the same coefficients in the same order and the two cancel face by
//  face rather than merely to truncation error. That exact cancellation is
//  what makes a sealed tank of stratified fluid stay at rest.
//
//  (g.x)_f is evaluated at the FACE CENTRE. The origin is free: shifting it by
//  x0 shifts p_rgh by rho (g.x0), and the identity
//  -grad(p) + rho g = -grad(p_rgh) - (g.x) grad(rho) is unchanged.
extern "C" __global__ void vofBodyForceFlux
(
    ofscalar* __restrict__ phib,
    const ofscalar* __restrict__ snGradRhoMagSf,
    const ofscalar* __restrict__ snGradAlphaMagSf,
    const ofscalar* __restrict__ kappaf,
    const ofvec3* __restrict__ Cf,
    ofscalar gx,
    ofscalar gy,
    ofscalar gz,
    ofscalar sigma,
    oflabel n
)
{
    const oflabel f = OFGPU_TID;
    if (f >= n) return;

    const ofvec3 c = Cf[f];
    const ofscalar gh = gx*c.x + gy*c.y + gz*c.z;

    phib[f] = -gh*snGradRhoMagSf[f] + sigma*kappaf[f]*snGradAlphaMagSf[f];
}


//- The same on the boundary, zero wherever the flux is prescribed.
//
//  A wall's flux is U_b . Sf and the pressure equation may not move it; giving
//  such a face a body force would put a source into a cell whose flux cannot
//  answer it, and a sealed box would drift. Identical in intent to
//  momBuoyancyFluxBoundary in cuda/momentum.cu.
extern "C" __global__ void vofBodyForceFluxBoundary
(
    ofscalar* __restrict__ bphib,
    const ofscalar* __restrict__ snGradRhoMagSf,
    const ofscalar* __restrict__ snGradAlphaMagSf,
    const ofscalar* __restrict__ kappaf,
    const ofvec3* __restrict__ bCf,
    const oflabel* __restrict__ bKind,
    const ofscalar* __restrict__ fr,
    ofscalar gx,
    ofscalar gy,
    ofscalar gz,
    ofscalar sigma,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    if (vofFluxIsPrescribed(bKind[i], fr[i]))
    {
        bphib[i] = 0;
        return;
    }

    const ofvec3 c = bCf[i];
    const ofscalar gh = gx*c.x + gy*c.y + gz*c.z;

    bphib[i] = -gh*snGradRhoMagSf[i] + sigma*kappaf[i]*snGradAlphaMagSf[i];
}


// ==========================================================================
//  S20.2  Sub-cycling: the Courant number the alpha equation is limited by
// ==========================================================================

//- Co_P/dt = (1/2) sum_f |phi_f| / V_P.
//
//  The explicit update moves at most the outgoing flux out of a cell in one
//  step, and the outgoing flux is half the total when the flux is solenoidal -
//  which is where the 1/2 comes from and why this is the right quantity to
//  compare against 1. The host multiplies by dt, reduces the maximum over the
//  mesh, and takes n = ceil(Co/Co_max) sub-cycles.
//
//  The compression flux is not summed here. |phi_r| <= c_alpha |phi_f| by
//  construction (S20.1: |n_f . Sf| <= |Sf|), so the host bounds the total by
//  (1 + c_alpha) times this and never underestimates it.
extern "C" __global__ void vofCourant
(
    ofscalar* __restrict__ co,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ bphi,
    const ofscalar* __restrict__ V,
    const oflabel* __restrict__ bKind,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofscalar acc = 0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        acc += vofAbs(phi[cfFace[j]]);
    }

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        const oflabel b = bcfFace[j];
        if (bKind[b] == OFPATCH_EMPTY) continue;
        acc += vofAbs(bphi[b]);
    }

    co[c] = (ofscalar)0.5*acc/V[c];
}


//- alpha V per cell, so a reduction over it gives the phase volume.
//
//  Conservation is the one property FCT does not have to buy: the limiter
//  scales FLUXES, and a flux leaving one cell is the flux entering the next
//  with the same lambda, so the scheme is conservative by construction. This
//  exists so a test can say so with a number.
extern "C" __global__ void vofPhaseVolume
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ alpha,
    const ofscalar* __restrict__ V,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    out[i] = alpha[i]*V[i];
}
