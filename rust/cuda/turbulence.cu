// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  turbulence.cu - the model-independent arithmetic of an eddy-viscosity RAS
  closure: face diffusivities, the eddy viscosity itself, the linearised
  source coefficients of the k, epsilon and omega equations, and the bounding
  that keeps all three positive.

  Written from:
    Launder & Spalding, "The numerical computation of turbulent flows",
      Comput. Methods Appl. Mech. Eng. 3 (1974) 269-289
    Wilcox, "Turbulence Modeling for CFD", DCW Industries - the 1988 k-omega
      form, and section 5.4 for the Favre-averaged dilatation terms
    Patankar, "Numerical Heat Transfer and Fluid Flow" (1980) section 4.2,
      for the S = S_u + S_p psi linearisation rule these coefficients feed
    ofgpu SPEC-LIT.md sections 6, 6.1 and 6.2. The bounding of k, epsilon and
      omega is marked *DESIGN* there and is ours; it is documented as such at
      each kernel below.
    ofgpu SPEC-LIT.md section 15.2 - nutLowReWallFunction is nu_t = 0 at the
      wall, and section 15.5 - each field's own patch type decides what
      happens to it there
    Menter, "Two-equation eddy-viscosity turbulence models for engineering
      applications", AIAA J. 32 (1994) 1598-1605, and Menter, Kuntz & Langtry,
      Turbulence, Heat and Mass Transfer 4 (2003) - the blended diffusivity at
      the bottom of this file, SPEC-LIT.md section 6.3
    Tucker, Applied Mathematical Modelling 22 (1998) 293-305 - the Poisson
      wall distance, SPEC-LIT.md section 6.6
    Rodi, J. Geophys. Res. 92 (1987) 5305-5328, and Henkes, van der Vlugt &
      Hoogendoorn, Int. J. Heat Mass Transfer 34 (1991) 377-388 - the
      buoyancy production G_b and the C_3 convention, SPEC-LIT.md section 17
  No GPL-licensed source was consulted.

  ------------------------------------------------------------------------
  What is here and what is not
  ------------------------------------------------------------------------

  The production term G = nu_t (dev(2 symm(grad U)) : grad U) is NOT here: it
  is `fvProduction` in fv.cu, next to the gradient operator that makes its
  argument. This file consumes G.

  Every kernel is one thread per cell or one thread per face, reads only its
  own item, and writes only its own item. There is no accumulation anywhere,
  so there are no atomics and the result is bitwise reproducible.

  ------------------------------------------------------------------------
  Sign convention of the source coefficients
  ------------------------------------------------------------------------

  The transport equations are assembled as

      ddt(psi) + div(phi, psi) - laplacian(Gamma_eff, psi) + Sp*psi = Su

  so a physical SINK on the right-hand side, -c*psi, arrives here as a
  POSITIVE Sp: that is the sign that lands on the diagonal and keeps the
  matrix diagonally dominant, which is Patankar's rule (section 4.2). The
  kernels below therefore emit magnitudes, and the caller passes them to
  fvm_sp with sign +1.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

//- BcKind values this file needs; mirrored from src/field.rs, and pinned by
//  `bc_kind_values_match_the_device` in src/turbulence.rs.
#define OFGPU_BC_CALCULATED 4
//- nutLowReWallFunction. SPEC-LIT 15.2: nu_t,w = 0, and that is the whole
//  model. The name declares that the mesh resolves the viscous sublayer, so
//  no wall function is wanted and the molecular viscosity alone carries the
//  wall shear.
#define OFGPU_BC_NUT_LOW_RE 22
//- Every wall-function kind is >= this. A patch marked with one of them has
//  its value written by a model, exactly like CALCULATED.
#define OFGPU_BC_WALLFN_FIRST 20

//- PatchKind (mesh b_kind), NOT BcKind (field bc_kind) - the two are never
//  compared across each other. This is the mesh-level "empty" patch (a 2-D
//  prismatic mesh's front/back plane), the same constant fv.cu's own
//  OFPATCH_EMPTY names; turbLsGradGradUMagSqr below needs it for the same
//  reason fvGradScalar/fvGradVector do - see their own definitions.
#define OFPATCH_EMPTY 2

#ifdef OFGPU_SINGLE
OFGPU_DEV ofscalar ofsqrt_(ofscalar a) { return sqrtf(a); }
OFGPU_DEV ofscalar oftanh_(ofscalar a) { return tanhf(a); }
OFGPU_DEV ofscalar ofexp_(ofscalar a)  { return expf(a); }
OFGPU_DEV ofscalar offma_(ofscalar a, ofscalar b, ofscalar c) { return fmaf(a, b, c); }
#else
OFGPU_DEV ofscalar ofsqrt_(ofscalar a) { return sqrt(a); }
OFGPU_DEV ofscalar oftanh_(ofscalar a) { return tanh(a); }
OFGPU_DEV ofscalar ofexp_(ofscalar a)  { return exp(a); }
OFGPU_DEV ofscalar offma_(ofscalar a, ofscalar b, ofscalar c) { return fma(a, b, c); }
#endif


// ==========================================================================
//  Effective diffusivity on the faces
//
//  Gamma_eff = nu + nu_t/sigma, interpolated to the face and multiplied by
//  |Sf| there, because that product - and not the bare Gamma - is what
//  fvm_laplacian takes (see the note on its signature in src/fv.rs).
//
//  Interpolating the DIFFUSIVITY rather than reconstructing it from an
//  interpolated nu_t is deliberate: the two differ only by the linearity of
//  the expression, which is exact here, so this is the cheaper of two
//  identical answers.
// ==========================================================================

extern "C" __global__ void turbGammaInternal
(
    ofscalar* __restrict__ gammaMagSf,
    const ofscalar* __restrict__ nut,
    const ofscalar* __restrict__ weights,
    const ofscalar* __restrict__ magSf,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    ofscalar nu,
    ofscalar rSigma,
    oflabel nInternalFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nInternalFaces) return;

    const ofscalar w = weights[f];
    const ofscalar gP = nu + rSigma*nut[owner[f]];
    const ofscalar gN = nu + rSigma*nut[neighbour[f]];

    gammaMagSf[f] = (w*gP + ((ofscalar)1 - w)*gN)*magSf[f];
}


extern "C" __global__ void turbGammaBoundary
(
    ofscalar* __restrict__ bGammaMagSf,
    const ofscalar* __restrict__ nutB,
    const ofscalar* __restrict__ bMagSf,
    ofscalar nu,
    ofscalar rSigma,
    oflabel nBoundaryFaces
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nBoundaryFaces) return;

    bGammaMagSf[i] = (nu + rSigma*nutB[i])*bMagSf[i];
}


// ==========================================================================
//  The eddy viscosity
// ==========================================================================

//- nu_t = C_mu k^2 / epsilon   (Launder & Spalding 1974)
//
//  *DESIGN* (SPEC-LIT 6.1): the quotient is capped at nutMax rather than
//  guarded with an epsilon in the denominator. A cap is a statement about the
//  physics - an eddy viscosity a hundred thousand times the molecular one is
//  not a turbulent flow, it is a diverging solve - whereas a denominator
//  epsilon is a statement about floating point and silently changes the
//  answer near the wall, where epsilon is genuinely small and k with it. The
//  companion kernel turbBoundEpsilon raises epsilon to match, so that the two
//  fields stay mutually consistent rather than the cap being applied to nut
//  alone.
extern "C" __global__ void turbNutKEpsilon
(
    ofscalar* __restrict__ nut,
    const ofscalar* __restrict__ k,
    const ofscalar* __restrict__ epsilon,
    ofscalar cmu,
    ofscalar nutMax,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar kc = ofmax_(k[c], (ofscalar)0);
    const ofscalar ec = epsilon[c];

    //- epsilon has already been bounded away from zero from below by
    //  turbBoundEpsilon; the test is what makes this kernel safe to call on a
    //  field that has not been, e.g. straight off a case file.
    const ofscalar v = (ec > (ofscalar)0) ? cmu*kc*kc/ec : nutMax;

    nut[c] = ofmin_(v, nutMax);
}


//- nu_t = k / omega   (Wilcox 1988)
extern "C" __global__ void turbNutKOmega
(
    ofscalar* __restrict__ nut,
    const ofscalar* __restrict__ k,
    const ofscalar* __restrict__ omega,
    ofscalar nutMax,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar kc = ofmax_(k[c], (ofscalar)0);
    const ofscalar wc = omega[c];

    const ofscalar v = (wc > (ofscalar)0) ? kc/wc : nutMax;

    nut[c] = ofmin_(v, nutMax);
}


//- nu_t on the boundary faces whose value is the MODEL's to write.
//
//  A CALCULATED face (and every wall-function kind, which behaves the same
//  way) carries no (fr, refValue, refGrad) triple that fldCorrectBcScalar
//  could evaluate, so something has to put a number there. Zero gradient is
//  the right default: nu_t is a property of the local turbulence and there is
//  no transport equation for it to satisfy at the face. Wall faces are
//  overwritten immediately afterwards by wfNutWall, which is the whole point
//  of a wall function.
extern "C" __global__ void turbNutBoundary
(
    ofscalar* __restrict__ nutB,
    const ofscalar* __restrict__ nut,
    const oflabel* __restrict__ bFaceCells,
    const oflabel* __restrict__ bcKind,
    oflabel nBoundaryFaces
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nBoundaryFaces) return;

    const oflabel kind = bcKind[i];
    if (kind != OFGPU_BC_CALCULATED && kind < OFGPU_BC_WALLFN_FIRST) return;

    //- SPEC-LIT 15.2. Three lines, and the whole point of the name: a
    //  resolved sublayer has no modelled eddy viscosity at the wall. Zero
    //  gradient here instead would hand the wall the first cell's nu_t, and
    //  the mesh is already resolving that stress.
    if (kind == OFGPU_BC_NUT_LOW_RE)
    {
        nutB[i] = (ofscalar)0;
        return;
    }

    nutB[i] = nut[bFaceCells[i]];
}


// ==========================================================================
//  Bounding
//
//  *DESIGN* (SPEC-LIT 6.1). k, epsilon and omega are positive by definition
//  and a discrete solve does not know that. The choices below are ours:
//
//    * k is clipped at kMin, a small absolute floor. It is the only one of
//      the three with no other quantity to be consistent with.
//
//    * epsilon is bounded from BELOW by the value that makes nu_t equal to
//      nutMax, epsilon >= C_mu k^2 / nutMax, and additionally by an absolute
//      epsMin. Bounding epsilon rather than clipping nu_t means the field
//      that enters the k equation as a sink is the same field that produced
//      the capped nu_t; clip nu_t alone and the two disagree, which shows up
//      as a k equation that will not converge.
//
//    * omega is bounded the same way, omega >= k/nutMax, for the same reason.
//
//  Both floors are applied to the internal field only; the boundary values
//  are regenerated from the triple afterwards, and the ones a model writes
//  (the wall cells) are physical values that need no floor.
// ==========================================================================

extern "C" __global__ void turbBoundK
(
    ofscalar* __restrict__ k,
    ofscalar kMin,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;
    k[c] = ofmax_(k[c], kMin);
}


extern "C" __global__ void turbBoundEpsilon
(
    ofscalar* __restrict__ epsilon,
    const ofscalar* __restrict__ k,
    ofscalar cmu,
    ofscalar nutMax,
    ofscalar epsMin,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar kc = ofmax_(k[c], (ofscalar)0);
    const ofscalar floorNut = (nutMax > (ofscalar)0) ? cmu*kc*kc/nutMax : (ofscalar)0;

    epsilon[c] = ofmax_(epsilon[c], ofmax_(epsMin, floorNut));
}


extern "C" __global__ void turbBoundOmega
(
    ofscalar* __restrict__ omega,
    const ofscalar* __restrict__ k,
    ofscalar nutMax,
    ofscalar omegaMin,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar kc = ofmax_(k[c], (ofscalar)0);
    const ofscalar floorNut = (nutMax > (ofscalar)0) ? kc/nutMax : (ofscalar)0;

    omega[c] = ofmax_(omega[c], ofmax_(omegaMin, floorNut));
}


// ==========================================================================
//  Linearised sources - k-epsilon (Launder & Spalding 1974)
//
//      Dk/Dt   = div((nu + nu_t/sigma_k) grad k)
//                + G - epsilon - (2/3)(div u) k
//      Deps/Dt = div((nu + nu_t/sigma_eps) grad eps)
//                + C_1 (eps/k) G - C_2 eps^2/k - (2/3 C_1 - C_3)(div u) eps
//
//  The dissipation terms are written as coefficients OF THE UNKNOWN, so they
//  land on the diagonal:
//
//      -epsilon        =  -(epsilon/k)  * k
//      -C_2 eps^2/k    =  -(C_2 eps/k)  * eps
//
//  which is Patankar's rule and is what SPEC-LIT 6.1 prescribes ("epsilon/k
//  and C_2 epsilon/k as implicit sinks on the diagonal, so that both
//  quantities stay positive"). Both are strictly positive given a bounded k,
//  so both stabilise the matrix unconditionally.
//
//  The dilatation terms are the Favre-averaged ones of Wilcox section 5.4 and
//  vanish identically when the discrete flux is solenoidal. Their sign is not
//  known in advance, so they are emitted as a `susp` coefficient and split by
//  fvm_susp; see SPEC-LIT 3.4.
// ==========================================================================

extern "C" __global__ void turbKSources
(
    ofscalar* __restrict__ sp,
    ofscalar* __restrict__ susp,
    const ofscalar* __restrict__ k,
    const ofscalar* __restrict__ epsilon,
    const ofscalar* __restrict__ divU,
    ofscalar kMin,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar kc = ofmax_(k[c], kMin);

    sp[c] = epsilon[c]/kc;
    susp[c] = ((ofscalar)2/(ofscalar)3)*divU[c];
}


extern "C" __global__ void turbEpsilonSources
(
    ofscalar* __restrict__ su,
    ofscalar* __restrict__ sp,
    ofscalar* __restrict__ susp,
    const ofscalar* __restrict__ g,
    const ofscalar* __restrict__ k,
    const ofscalar* __restrict__ epsilon,
    const ofscalar* __restrict__ divU,
    ofscalar c1,
    ofscalar c2,
    ofscalar c3,
    ofscalar kMin,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar kc = ofmax_(k[c], kMin);
    const ofscalar ec = epsilon[c];
    const ofscalar rTau = ec/kc;              // 1 / turbulent time scale

    su[c] = c1*rTau*g[c];
    sp[c] = c2*rTau;
    susp[c] = (((ofscalar)2/(ofscalar)3)*c1 - c3)*divU[c];
}


// ==========================================================================
//  SPEC-LIT section 33 - Launder-Sharma low-Reynolds-number k-epsilon
//
//  Written from:
//    Launder & Sharma, "Application of the energy-dissipation model of
//      turbulence to the calculation of flow near a spinning disc", Letters
//      in Heat and Mass Transfer 1 (1974) 131-138
//    Patel, Rodi & Scheuerer, "Turbulence models for near-wall and low
//      Reynolds number flows: a review", AIAA J. 23 (1985) 1308-1319
//    ofgpu SPEC-LIT.md section 33
//  No GPL-licensed source was consulted.
//
//  The model solves for epsilon_tilde, the ISOTROPIC dissipation, which
//  (unlike epsilon) is exactly zero at a solid wall:
//
//      epsilon      = epsilon_tilde + D,     D = 2 nu |grad(sqrt k)|^2
//      nu_t         = C_mu f_mu k^2 / epsilon_tilde
//      Re_t         = k^2 / (nu epsilon_tilde)
//      f_mu         = exp( -3.4 / (1 + Re_t/50)^2 )
//      f_2          = 1 - 0.3 exp(-Re_t^2)
//      E            = 2 nu nu_t |grad(grad U)|^2
//
//      Dk/Dt  = div((nu + nu_t/sigma_k) grad k)   + G - epsilon_tilde - D
//      De~/Dt = div((nu + nu_t/sigma_e) grad e~)  + C_1 (e~/k) G
//                                                  - C_2 f_2 e~^2/k + E
//
//  C_mu, C_1, C_2, sigma_k, sigma_eps are UNCHANGED from section 6.1 - the
//  model modifies section 6.1 with f_mu, f_2, D and E, not with new
//  constants, and the two sections are checked against each other for
//  exactly that reason (turbNutLaunderSharma/turbLsEpsilonSources reduce to
//  turbNutKEpsilon/turbEpsilonSources's shear-production term at
//  f_mu, f_2 -> 1). Section 33.1 gives no dilatation term for this model -
//  unlike section 6.1's Favre-averaged extension - so none is added here;
//  the k equation's Sp = epsilon_tilde/k sink is exactly turbKSources's own
//  formula (called with epsilon_tilde standing in for epsilon), and the
//  model's own doc says why that reuse is exact rather than approximate.
//
//  DESIGN - grad(grad U). SPEC-LIT section 33.1 marks the second-derivative
//  term as *DESIGN*: take the Gauss gradient of the ALREADY-COMPUTED cell
//  velocity gradient (fvGradVector's own output), once per outer iteration,
//  reused by both the k and epsilon_tilde equations. turbLsGradGradUMagSqr
//  below is that gradient, specialised to emit only the scalar magnitude
//  the E term needs (27 second-derivative components per cell, squared and
//  summed) rather than materialising the full third-order tensor - same
//  gather cost as one more fvGradVector-shaped pass, three times the
//  arithmetic per face for the extra six components, paid once per outer
//  iteration. grad U carries no boundary field of its own (unlike U), so
//  the boundary contribution extrapolates the owner cell's gradient
//  (zero-gradient) rather than reading one that does not exist; this is
//  exact wherever grad U is itself uniform near the face, which is what
//  `E = 0 on a linear velocity field` (SPEC-LIT section 33.3) checks.
// ==========================================================================

//- nu_t = C_mu f_mu k^2 / epsilon_tilde   (Launder & Sharma 1974)
//
//  Same cap-rather-than-guard *DESIGN* as turbNutKEpsilon; f_mu <= 1 always,
//  so this nu_t never exceeds the standard model's for the same k and
//  epsilon_tilde, and the epsilon_tilde bound turbBoundEpsilon already
//  enforces (reused unchanged - SPEC-LIT 33.1's coefficients are the same)
//  is conservative for the capped nu_t here too.
extern "C" __global__ void turbNutLaunderSharma
(
    ofscalar* __restrict__ nut,
    const ofscalar* __restrict__ k,
    const ofscalar* __restrict__ epsTilde,
    ofscalar nu,
    ofscalar cmu,
    ofscalar nutMax,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar kc = ofmax_(k[c], (ofscalar)0);
    const ofscalar ec = epsTilde[c];

    if (!(ec > (ofscalar)0))
    {
        nut[c] = (ofscalar)0;
        return;
    }

    const ofscalar reT   = kc*kc/(nu*ec);
    const ofscalar denom = (ofscalar)1 + reT/(ofscalar)50;
    const ofscalar fmu   = ofexp_(-(ofscalar)3.4/(denom*denom));

    const ofscalar v = cmu*fmu*kc*kc/ec;
    nut[c] = ofmin_(v, nutMax);
}


//- out = sqrt(max(in, 0)), elementwise - the scratch `sqrt(k)` field the D
//  term's gradient is taken of, built once for the interior cells and once
//  for the boundary faces (two launches, one kernel: see
//  turbulence::ls_sqrt_positive).
extern "C" __global__ void turbLsSqrtPositive
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ in,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    out[i] = ofsqrt_(ofmax_(in[i], (ofscalar)0));
}


//- D = 2 nu |grad(sqrt k)|^2 - the extra sink in the k equation that
//  survives the epsilon -> epsilon_tilde substitution (SPEC-LIT 33.1).
//  `gradSqrtK` is the Gauss gradient of the field turbLsSqrtPositive built;
//  see `fv::fvc_grad_scalar`.
extern "C" __global__ void turbLsDTerm
(
    ofscalar* __restrict__ d,
    const ofvec3* __restrict__ gradSqrtK,
    ofscalar nu,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofvec3 g = gradSqrtK[c];
    d[c] = (ofscalar)2*nu*dot3(g, g);
}


//- |grad(grad U)|^2 - the Gauss gradient of the cell velocity-gradient
//  TENSOR field, specialised to its scalar magnitude squared (27
//  second-derivative components, summed) rather than the full third-order
//  tensor. See this section's own header for the boundary DESIGN note and
//  the cost this pays once per outer iteration.
//
//  Structurally this is fvGradVector's gather with 9 field components
//  instead of 3: three accumulators (a*, b*, c*), one per differentiation
//  direction x/y/z, each a 3x3 running sum over the SAME 9 components of
//  gradU that fvGradVector accumulates 3 of.
extern "C" __global__ void turbLsGradGradUMagSqr
(
    ofscalar* __restrict__ out,
    const oftensor* __restrict__ gradU,
    const ofscalar* __restrict__ w,
    const ofvec3* __restrict__ Sf,
    const ofvec3* __restrict__ bSf,
    const ofscalar* __restrict__ V,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ bFaceCells,
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

    // d(gradU_**)/dx
    ofscalar axx=0,axy=0,axz=0, ayx=0,ayy=0,ayz=0, azx=0,azy=0,azz=0;
    // d(gradU_**)/dy
    ofscalar bxx=0,bxy=0,bxz=0, byx=0,byy=0,byz=0, bzx=0,bzy=0,bzz=0;
    // d(gradU_**)/dz
    ofscalar bcxx=0,bcxy=0,bcxz=0, bcyx=0,bcyy=0,bcyz=0, bczx=0,bczy=0,bczz=0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const ofscalar wf = w[f];
        const ofscalar omw = (ofscalar)1 - wf;

        const oftensor go = gradU[owner[f]];
        const oftensor gn = gradU[neighbour[f]];

        const ofscalar txx = wf*go.xx + omw*gn.xx;
        const ofscalar txy = wf*go.xy + omw*gn.xy;
        const ofscalar txz = wf*go.xz + omw*gn.xz;
        const ofscalar tyx = wf*go.yx + omw*gn.yx;
        const ofscalar tyy = wf*go.yy + omw*gn.yy;
        const ofscalar tyz = wf*go.yz + omw*gn.yz;
        const ofscalar tzx = wf*go.zx + omw*gn.zx;
        const ofscalar tzy = wf*go.zy + omw*gn.zy;
        const ofscalar tzz = wf*go.zz + omw*gn.zz;

        const ofvec3 s0 = Sf[f];
        const ofscalar sg = cfOwn[j] ? (ofscalar)1 : (ofscalar)-1;
        const ofscalar sx = sg*s0.x, sy = sg*s0.y, sz = sg*s0.z;

        axx += sx*txx; axy += sx*txy; axz += sx*txz;
        ayx += sx*tyx; ayy += sx*tyy; ayz += sx*tyz;
        azx += sx*tzx; azy += sx*tzy; azz += sx*tzz;

        bxx += sy*txx; bxy += sy*txy; bxz += sy*txz;
        byx += sy*tyx; byy += sy*tyy; byz += sy*tyz;
        bzx += sy*tzx; bzy += sy*tzy; bzz += sy*tzz;

        bcxx += sz*txx; bcxy += sz*txy; bcxz += sz*txz;
        bcyx += sz*tyx; bcyy += sz*tyy; bcyz += sz*tyz;
        bczx += sz*tzx; bczy += sz*tzy; bczz += sz*tzz;
    }

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        const oflabel b = bcfFace[j];
        if (bKind[b] == OFPATCH_EMPTY) continue;

        // DESIGN: zero-gradient extrapolation - see this section's header.
        const oftensor t = gradU[bFaceCells[b]];
        const ofvec3 s = bSf[b];

        axx += s.x*t.xx; axy += s.x*t.xy; axz += s.x*t.xz;
        ayx += s.x*t.yx; ayy += s.x*t.yy; ayz += s.x*t.yz;
        azx += s.x*t.zx; azy += s.x*t.zy; azz += s.x*t.zz;

        bxx += s.y*t.xx; bxy += s.y*t.xy; bxz += s.y*t.xz;
        byx += s.y*t.yx; byy += s.y*t.yy; byz += s.y*t.yz;
        bzx += s.y*t.zx; bzy += s.y*t.zy; bzz += s.y*t.zz;

        bcxx += s.z*t.xx; bcxy += s.z*t.xy; bcxz += s.z*t.xz;
        bcyx += s.z*t.yx; bcyy += s.z*t.yy; bcyz += s.z*t.yz;
        bczx += s.z*t.zx; bczy += s.z*t.zy; bczz += s.z*t.zz;
    }

    const ofscalar rv = 1/V[c];
    axx*=rv; axy*=rv; axz*=rv; ayx*=rv; ayy*=rv; ayz*=rv; azx*=rv; azy*=rv; azz*=rv;
    bxx*=rv; bxy*=rv; bxz*=rv; byx*=rv; byy*=rv; byz*=rv; bzx*=rv; bzy*=rv; bzz*=rv;
    bcxx*=rv; bcxy*=rv; bcxz*=rv; bcyx*=rv; bcyy*=rv; bcyz*=rv; bczx*=rv; bczy*=rv; bczz*=rv;

    out[c] =
        axx*axx + axy*axy + axz*axz + ayx*ayx + ayy*ayy + ayz*ayz + azx*azx + azy*azy + azz*azz +
        bxx*bxx + bxy*bxy + bxz*bxz + byx*byx + byy*byy + byz*byz + bzx*bzx + bzy*bzy + bzz*bzz +
        bcxx*bcxx + bcxy*bcxy + bcxz*bcxz + bcyx*bcyx + bcyy*bcyy + bcyz*bcyz +
        bczx*bczx + bczy*bczy + bczz*bczz;
}


//- E = 2 nu nu_t |grad(grad U)|^2 (SPEC-LIT 33.1) - `gradGradUMagSqr` is
//  turbLsGradGradUMagSqr's output, `nut` the PREVIOUS outer iteration's eddy
//  viscosity (the same production-term lag every source in this crate uses).
extern "C" __global__ void turbLsETerm
(
    ofscalar* __restrict__ e,
    const ofscalar* __restrict__ gradGradUMagSqr,
    const ofscalar* __restrict__ nut,
    ofscalar nu,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;
    e[c] = (ofscalar)2*nu*nut[c]*gradGradUMagSqr[c];
}


//- The epsilon_tilde equation's linearised sources (SPEC-LIT 33.1):
//
//      Su = C_1 (e~/k) G + E
//      Sp = C_2 f_2 e~/k
//
//  f_2 = 1 - 0.3 exp(-Re_t^2) is computed inline rather than cached: it is
//  three flops given Re_t, and Re_t itself needs epsilon_tilde, which this
//  kernel already reads. `ec <= 0` defaults f_2 to its Re_t -> infinity
//  limit (1), which is the value bounding epsilon_tilde away from zero
//  would otherwise have produced - not a special case, the limit itself.
//  No dilatation term: SPEC-LIT 33.1 gives none for this model - see this
//  section's own header.
extern "C" __global__ void turbLsEpsilonSources
(
    ofscalar* __restrict__ su,
    ofscalar* __restrict__ sp,
    const ofscalar* __restrict__ g,
    const ofscalar* __restrict__ k,
    const ofscalar* __restrict__ epsTilde,
    const ofscalar* __restrict__ eTerm,
    ofscalar nu,
    ofscalar c1,
    ofscalar c2,
    ofscalar kMin,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar kc = ofmax_(k[c], kMin);
    const ofscalar ec = epsTilde[c];
    const ofscalar rTau = ec/kc;

    ofscalar f2 = (ofscalar)1;
    if (ec > (ofscalar)0)
    {
        const ofscalar reT = kc*kc/(nu*ec);
        f2 = (ofscalar)1 - (ofscalar)0.3*ofexp_(-reT*reT);
    }

    su[c] = c1*rTau*g[c] + eTerm[c];
    sp[c] = c2*f2*rTau;
}


// ==========================================================================
//  Linearised sources - Wilcox k-omega (1988)
//
//      Dk/Dt = div((nu + alpha_k nu_t) grad k)
//              + G - beta* k omega - (2/3)(div u) k
//      Dw/Dt = div((nu + alpha_w nu_t) grad omega)
//              + gamma (omega/k) G - beta omega^2
//
//  The k sink is -beta* omega k, so Sp = beta* omega with no division at all,
//  which is one of the reasons the omega form is better behaved than the
//  epsilon form at a wall.
//
//  The omega production is written exactly as SPEC-LIT 6.2 gives it,
//  gamma (omega/k) G, rather than as the algebraically equal gamma G/nu_t.
//  They differ once nu_t has been CAPPED by turbNutKOmega: omega/k is then
//  still the model's own inverse time scale, whereas G/nu_t would silently
//  inflate the production by the ratio the cap removed.
// ==========================================================================

extern "C" __global__ void turbKOmegaKSources
(
    ofscalar* __restrict__ sp,
    ofscalar* __restrict__ susp,
    const ofscalar* __restrict__ omega,
    const ofscalar* __restrict__ divU,
    ofscalar betaStar,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    sp[c] = betaStar*omega[c];
    susp[c] = ((ofscalar)2/(ofscalar)3)*divU[c];
}


extern "C" __global__ void turbOmegaSources
(
    ofscalar* __restrict__ su,
    ofscalar* __restrict__ sp,
    const ofscalar* __restrict__ g,
    const ofscalar* __restrict__ k,
    const ofscalar* __restrict__ omega,
    ofscalar gamma,
    ofscalar beta,
    ofscalar kMin,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar kc = ofmax_(k[c], kMin);
    const ofscalar wc = omega[c];

    su[c] = gamma*(wc/kc)*g[c];
    sp[c] = beta*wc;
}


// ==========================================================================
//  Convergence measure
// ==========================================================================

//- out = |a - b|, so that a max-magnitude reduction over `out` gives the
//  largest change of the field between two outer iterations. Kept as its own
//  kernel rather than folded into a fused reduction because it is called once
//  every `convergenceCheckEvery` iterations and never inside a captured
//  graph, so its cost is irrelevant and its clarity is not.
extern "C" __global__ void turbAbsDiff
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ a,
    const ofscalar* __restrict__ b,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;

    const ofscalar d = a[i] - b[i];
    out[i] = (d < (ofscalar)0) ? -d : d;
}


// ==========================================================================
//  Diagnostics
// ==========================================================================

//- The strain-rate magnitude sqrt(2 |symm(grad U)|^2) per cell.
//
//  Not used by the two models in this crate - both take their production from
//  fvProduction - but it is what an SST nu_t limiter and every LES delta need
//  (SPEC-LIT 6.3, 6.5), and it is one line given grad U, so it lives here
//  rather than being rederived when those arrive.
extern "C" __global__ void turbStrainRateMag
(
    ofscalar* __restrict__ out,
    const oftensor* __restrict__ gradU,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const oftensor g = gradU[c];

    //- twoSymm(g) = g + g^T
    const ofscalar sxx = (ofscalar)2*g.xx;
    const ofscalar syy = (ofscalar)2*g.yy;
    const ofscalar szz = (ofscalar)2*g.zz;
    const ofscalar sxy = g.xy + g.yx;
    const ofscalar sxz = g.xz + g.zx;
    const ofscalar syz = g.yz + g.zy;

    //- S^2 = 2 symm(g) : symm(g) = 0.5 twoSymm(g) : twoSymm(g)
    const ofscalar dd =
        sxx*sxx + syy*syy + szz*szz
      + (ofscalar)2*(sxy*sxy + sxz*sxz + syz*syz);

    out[c] = ofsqrt_((ofscalar)0.5*dd);
}


// ==========================================================================
//  Two more invariants of grad U - SPEC-LIT S56.2
//
//  turbStrainRateMag above gives S = sqrt(2 S_ij S_ij). The Spalart-Allmaras
//  model (S56) is calibrated on the VORTICITY magnitude, and the DES-family
//  shielding functions (S57) read the Frobenius norm of the FULL gradient.
//  Three different numbers, and confusing any pair of them is a silent error,
//  so all three live together here rather than being rederived where they are
//  used:
//
//      S     = sqrt(2 S_ij S_ij)          turbStrainRateMag
//      Omega = sqrt(2 W_ij W_ij)          turbVorticityMag
//      F     = sqrt(sum_ij g_ij^2)        turbGradFrobenius
//
//  They satisfy F^2 = (S^2 + Omega^2)/2 identically, because S_ij : W_ij = 0
//  for a symmetric tensor contracted with an antisymmetric one. F is
//  nevertheless written out from the nine components below, NOT rebuilt from
//  the other two, so that the identity stays something to check rather than
//  something assumed twice (SPEC-LIT S56.10).
//
//  There is no `dev` here. S40.2's deviatoric correction belongs to the
//  realizability construction, which is about the normal stresses; a
//  vorticity and a norm of the raw gradient are neither, and subtracting a
//  trace from either would be wrong.
//
//  Written from:
//    ofgpu SPEC-LIT.md S56.2, S57.3
//    P. R. Spalart, S. R. Allmaras, AIAA Paper 92-0439 (1992) - Omega
//    P. R. Spalart et al., Theor. Comput. Fluid Dyn. 20 (2006) 181-195 - the
//      full-gradient norm in r_d
//  No GPL-licensed source was consulted.
// ==========================================================================

extern "C" __global__ void turbVorticityMag
(
    ofscalar* __restrict__ out,
    const oftensor* __restrict__ gradU,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const oftensor g = gradU[c];

    //- 2 W_ij = g_ij - g_ji, three independent components. Then
    //  W_ij W_ij = (wxy^2 + wxz^2 + wyz^2)/2, so Omega^2 = 2 W_ij W_ij is
    //  exactly that sum.
    const ofscalar wxy = g.xy - g.yx;
    const ofscalar wxz = g.xz - g.zx;
    const ofscalar wyz = g.yz - g.zy;

    out[c] = ofsqrt_(wxy*wxy + wxz*wxz + wyz*wyz);
}


extern "C" __global__ void turbGradFrobenius
(
    ofscalar* __restrict__ out,
    const oftensor* __restrict__ gradU,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const oftensor g = gradU[c];

    out[c] = ofsqrt_(
        g.xx*g.xx + g.xy*g.xy + g.xz*g.xz
      + g.yx*g.yx + g.yy*g.yy + g.yz*g.yz
      + g.zx*g.zx + g.zy*g.zy + g.zz*g.zz);
}


// ==========================================================================
//  Blended diffusivity - SPEC-LIT 6.3
//
//  Extended from:
//    Menter, "Two-equation eddy-viscosity turbulence models for engineering
//      applications", AIAA J. 32 (1994) 1598-1605
//    Menter, Kuntz & Langtry, "Ten years of industrial experience with the
//      SST turbulence model", Turbulence, Heat and Mass Transfer 4 (2003)
//    Tucker, "Assessment of geometric multilevel convergence and a wall
//      distance method for flows with multiple internal boundaries",
//      Applied Mathematical Modelling 22 (1998) 293-305
//    ofgpu SPEC-LIT.md sections 6.3 and 6.6
//  No GPL-licensed source was consulted.
//
//  SST blends its own sigma between two coefficient sets with F1, so the
//  diffusivity multiplier is a FIELD, not a constant. These two kernels are
//  turbGammaInternal/turbGammaBoundary with rSigma read per cell instead of
//  passed by value; everything else about them - interpolating the
//  diffusivity rather than nu_t, multiplying by |Sf| here rather than in
//  fvm_laplacian - is unchanged, and deliberately so: the two pairs must
//  agree exactly when rSigma happens to be uniform, which is what
//  `blended_diffusivity_matches_the_uniform_one` in src/turbulence.rs
//  measures.
// ==========================================================================

extern "C" __global__ void turbGammaInternalCell
(
    ofscalar* __restrict__ gammaMagSf,
    const ofscalar* __restrict__ nut,
    const ofscalar* __restrict__ rSigma,
    const ofscalar* __restrict__ weights,
    const ofscalar* __restrict__ magSf,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    ofscalar nu,
    oflabel nInternalFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nInternalFaces) return;

    const oflabel P = owner[f];
    const oflabel N = neighbour[f];

    const ofscalar w = weights[f];
    const ofscalar gP = nu + rSigma[P]*nut[P];
    const ofscalar gN = nu + rSigma[N]*nut[N];

    gammaMagSf[f] = (w*gP + ((ofscalar)1 - w)*gN)*magSf[f];
}


//- The boundary face takes its owner cell's blending factor. There is no
//  second cell to interpolate against, and F1 -> 1 at a wall in any case, so
//  the two coefficient sets agree there to the accuracy of the first cell.
extern "C" __global__ void turbGammaBoundaryCell
(
    ofscalar* __restrict__ bGammaMagSf,
    const ofscalar* __restrict__ nutB,
    const ofscalar* __restrict__ rSigma,
    const oflabel* __restrict__ faceCells,
    const ofscalar* __restrict__ bMagSf,
    ofscalar nu,
    oflabel nBoundaryFaces
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nBoundaryFaces) return;

    bGammaMagSf[i] = (nu + rSigma[faceCells[i]]*nutB[i])*bMagSf[i];
}


// ==========================================================================
//  Affine diffusivity - SPEC-LIT 41.2
//
//  Extended from:
//    V. Yakhot, S. A. Orszag, S. Thangam, T. B. Gatski, C. G. Speziale,
//      ICASE Report 91-65 / NASA CR-187611 (1991); Phys. Fluids A 4 (1992)
//      1510-1520 - the RNG inverse Prandtl numbers alpha_k, alpha_eps
//    ofgpu SPEC-LIT.md section 41.2
//  No GPL-licensed source was consulted.
//
//  turbGammaInternal computes Gamma_eff = nu + rSigma nu_t: the coefficient
//  multiplies the TURBULENT viscosity alone. The RNG model (section 41) wants
//  Gamma_eff = alpha (nu + nu_t) - the inverse Prandtl number multiplies the
//  EFFECTIVE viscosity, molecular part included. At high Reynolds number the
//  two are indistinguishable; in the first cell off a wall they are not, and
//  folding alpha into rSigma to get nu + alpha nu_t would be wrong there and
//  silently so.
//
//  So: one kernel taking BOTH coefficients, Gamma_eff = a nu + b nu_t.
//  turbGammaInternal is the a = 1 case of it, and multiplication by an exact
//  1.0 is exact in IEEE-754, so the two agree BIT FOR BIT - which is what
//  `the_affine_diffusivity_reduces_to_the_plain_one_bitwise` in
//  src/turbulence.rs measures rather than assumes. Section 40 calls it with
//  (1, 1/sigma) and section 41 with (alpha, alpha), so the kernel added here
//  once is used by both.
// ==========================================================================

extern "C" __global__ void turbGammaInternalAffine
(
    ofscalar* __restrict__ gammaMagSf,
    const ofscalar* __restrict__ nut,
    const ofscalar* __restrict__ weights,
    const ofscalar* __restrict__ magSf,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    ofscalar nu,
    ofscalar a,
    ofscalar b,
    oflabel nInternalFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nInternalFaces) return;

    const ofscalar w = weights[f];

    //- EXPLICIT fma, not `a*nu + b*nut[P]`. Written the obvious way the
    //  expression has TWO multiplies and one add, nvcc is free to fuse either,
    //  and it fuses the wrong one - which rounds the `b*nut` product before
    //  adding and leaves the answer ONE ULP from turbGammaInternal's. One ULP
    //  is enough to make `affine(1, r)` a different kernel from
    //  `face_diffusivity(r)`, and sections 6.1, 6.2 and 6.3's recorded results
    //  are stated against the latter. Hoisting `a*nu` onto its own line is not
    //  enough either (measured). Naming the fused operation removes the
    //  compiler's discretion entirely: `a*nu` is exact at a = 1, and what is
    //  left is the single fused multiply-add nvcc emits for the plain kernel's
    //  `nu + rSigma*nut[P]`.
    //  `the_affine_diffusivity_reduces_to_the_plain_one_bitwise` in
    //  src/turbulence.rs is what found this and is what keeps it true.
    const ofscalar aNu = a*nu;
    const ofscalar gP = offma_(b, nut[owner[f]], aNu);
    const ofscalar gN = offma_(b, nut[neighbour[f]], aNu);

    gammaMagSf[f] = (w*gP + ((ofscalar)1 - w)*gN)*magSf[f];
}


extern "C" __global__ void turbGammaBoundaryAffine
(
    ofscalar* __restrict__ bGammaMagSf,
    const ofscalar* __restrict__ nutB,
    const ofscalar* __restrict__ bMagSf,
    ofscalar nu,
    ofscalar a,
    ofscalar b,
    oflabel nBoundaryFaces
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nBoundaryFaces) return;

    const ofscalar aNu = a*nu;
    bGammaMagSf[i] = offma_(b, nutB[i], aNu)*bMagSf[i];
}


// ==========================================================================
//  Wall distance from the Poisson potential - SPEC-LIT 6.6
//
//  Tucker (1998). Having solved
//
//      laplacian(phi) = -1 ,  phi = 0 on walls,  dphi/dn = 0 elsewhere
//
//  the distance follows point by point from
//
//      y = -|grad phi| + sqrt( |grad phi|^2 + 2 phi )
//
//  which is exact wherever the problem is locally one-dimensional: with
//  phi = y(L-y)/2 between two walls a distance L apart, |grad phi| = |L/2 - y|
//  and the square root collapses to L/2, leaving y = min(y, L-y) identically.
//  Away from that limit it is an approximation that keeps the right zero at
//  the wall and the right unit slope leaving it, which is what the models
//  that consume it need.
//
//  phi is clamped at zero before the square root. The solve returns a
//  strictly positive phi in exact arithmetic, but a wall-adjacent cell can
//  come back very slightly negative on a badly non-orthogonal mesh, and a NaN
//  distance would then propagate into nu_t rather than announcing itself.
// ==========================================================================

extern "C" __global__ void turbWallDistance
(
    ofscalar* __restrict__ y,
    const ofvec3* __restrict__ gradPhi,
    const ofscalar* __restrict__ phi,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofvec3 g = gradPhi[c];
    const ofscalar gg = dot3(g, g);
    const ofscalar p  = ofmax_(phi[c], (ofscalar)0);

    const ofscalar d = ofsqrt_(gg + (ofscalar)2*p) - ofsqrt_(gg);

    y[c] = ofmax_(d, (ofscalar)0);
}


// ==========================================================================
//  Buoyancy production - SPEC-LIT section 17
//
//  Rodi (1987) writes the term as
//
//      G_b = -(nu_t/Pr_t) g . grad(rho) / rho
//
//  and with the ideal-gas density of SPEC-LIT section 9, rho ~ 1/T at
//  constant pressure, grad(rho)/rho = -grad(T)/T, so
//
//      G_b = (nu_t/Pr_t) g . grad(T) / T
//
//  THE SIGN, which is the first thing to check and the only thing that
//  cannot be fixed later: in a stably stratified layer grad(T) points UP and
//  g points DOWN, so g.grad(T) < 0 and G_b < 0 - buoyancy destroys
//  turbulence. Above a heat source grad(T) points DOWN, g.grad(T) > 0 and
//  G_b > 0 - buoyancy makes it. Both are what the physics says.
//
//  C_3, the coefficient the epsilon equation multiplies G_b by, is the one
//  genuinely unsettled constant in the model. SPEC-LIT section 17 gives two
//  conventions and defaults to the second:
//
//      mode 0: C_3 = the constant the case supplied (0 ignores G_b in eps)
//      mode 1: C_3 = tanh |u_parallel_to_g / u_normal_to_g|   Henkes (1991)
//
//  The Henkes form goes to 1 in a vertical shear layer - a plume - and to 0
//  in a horizontal one, which is the behaviour the data supports.
//
//  T is clamped away from zero: it is an absolute temperature and a case that
//  has 0 K in it has worse problems than this term, but a division by it
//  would put a NaN into k and epsilon rather than announcing itself.
// ==========================================================================

extern "C" __global__ void turbBuoyancyProduction
(
    ofscalar* __restrict__ gb,
    ofscalar* __restrict__ c3,
    const ofscalar* __restrict__ nut,
    const ofvec3*  __restrict__ gradT,
    const ofscalar* __restrict__ t,
    const ofvec3*  __restrict__ u,
    ofscalar gx,
    ofscalar gy,
    ofscalar gz,
    ofscalar rPrt,
    oflabel  c3Mode,
    ofscalar c3Const,
    ofscalar tMin,
    oflabel  nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofvec3 g  = mkvec(gx, gy, gz);
    const ofvec3 gt = gradT[c];
    const ofscalar tc = ofmax_(t[c], tMin);

    gb[c] = nut[c]*rPrt*dot3(g, gt)/tc;

    if (c3Mode == 0)
    {
        c3[c] = c3Const;
        return;
    }

    // Henkes et al. (1991): the velocity is split into the component along
    // gravity and what is left, and C_3 = tanh of their ratio. |g| == 0 has
    // no axis to split along, and there is no buoyancy either, so C_3 there
    // is arbitrary; zero is the value that adds nothing.
    const ofscalar gmag = ofsqrt_(dot3(g, g));
    if (!(gmag > (ofscalar)0))
    {
        c3[c] = (ofscalar)0;
        return;
    }

    const ofvec3 e = mkvec(g.x/gmag, g.y/gmag, g.z/gmag);
    const ofvec3 uc = u[c];
    const ofscalar upar = dot3(uc, e);
    const ofscalar u2 = dot3(uc, uc);
    // |u_normal|^2 = |u|^2 - u_parallel^2, clamped: round-off can make it
    // very slightly negative when the velocity is exactly along g.
    const ofscalar un2 = ofmax_(u2 - upar*upar, (ofscalar)0);
    const ofscalar un = ofsqrt_(un2);

    // u_normal == 0 is a purely vertical velocity: the ratio is infinite and
    // tanh of it is 1, which is the limit and not a special case.
    if (!(un > (ofscalar)0))
    {
        c3[c] = (upar != (ofscalar)0) ? (ofscalar)1 : (ofscalar)0;
        return;
    }

    ofscalar r = upar/un;
    if (r < (ofscalar)0) r = -r;
    c3[c] = oftanh_(r);
}


// ==========================================================================
//  G_b into the k equation - SPEC-LIT section 17
//
//  k gets + G_b, of either sign. Patankar section 4.2 wants a source that is
//  a sink linearised onto the diagonal, so the term is split:
//
//      G_b >= 0:  su += G_b                      a source, explicit
//      G_b <  0:  sp += -G_b/max(k, kMin)        a sink, on the diagonal
//
//  The two agree at the current k - sp*k = -G_b exactly - so this is a
//  stability choice and not an approximation, the same argument fvm_susp
//  rests on. Splitting rather than using fvm_susp directly is what lets the
//  epsilon equation apply a DIFFERENT rule to the same two branches; see
//  below.
//
//  `sp` is ACCUMULATED into, because the dissipation sink epsilon/k is
//  already there. `su` is WRITTEN: the k equation's own explicit source is
//  the shear production G, which the model hands to fvm_su as a separate
//  array, so this buffer holds the buoyant part alone and starts each call
//  with whatever the previous equation left in it.
// ==========================================================================

extern "C" __global__ void turbAddBuoyancyToK
(
    ofscalar* __restrict__ su,
    ofscalar* __restrict__ sp,
    const ofscalar* __restrict__ gb,
    const ofscalar* __restrict__ k,
    ofscalar kMin,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar b = gb[c];
    if (b >= (ofscalar)0)
    {
        su[c] = b;
    }
    else
    {
        su[c] = (ofscalar)0;
        sp[c] += -b/ofmax_(k[c], kMin);
    }
}


// ==========================================================================
//  G_b into the epsilon equation - SPEC-LIT section 17
//
//      epsilon : + C_1 (eps/k) C_3 G_b
//
//  split by sign exactly as the k equation's is:
//
//      term >= 0:  su += C_1 (eps/k) C_3 G_b
//      term <  0:  sp += -C_1 C_3 G_b / k       (dividing the sink by eps)
//
//  `stableBranch` selects which branches are included, which SPEC-LIT
//  section 17 asks to be stated rather than assumed:
//
//      0  the UNSTABLE branch only (G_b > 0) - the default
//      1  both branches
//
//  *DESIGN.* The default is 0. Section 17 says the unstable branch belongs in
//  both equations and that the stable branch is often kept in k alone, and
//  that is the combination implemented here; a case that wants the stable
//  branch in epsilon too can ask for it. With the Henkes C_3 the difference
//  is small in a plume (C_3 -> 1 where the shear is vertical, and there the
//  branch is unstable anyway) and largest in a stratified horizontal layer,
//  which is where the evidence for including it is weakest.
// ==========================================================================

extern "C" __global__ void turbAddBuoyancyToEpsilon
(
    ofscalar* __restrict__ su,
    ofscalar* __restrict__ sp,
    const ofscalar* __restrict__ gb,
    const ofscalar* __restrict__ c3,
    const ofscalar* __restrict__ k,
    const ofscalar* __restrict__ epsilon,
    ofscalar c1,
    ofscalar kMin,
    oflabel  stableBranch,
    oflabel  nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar b = gb[c];
    if (b < (ofscalar)0 && stableBranch == 0) return;

    const ofscalar kc = ofmax_(k[c], kMin);
    const ofscalar term = c1*c3[c]*b;          // the (eps/k) factor is below

    if (term >= (ofscalar)0)
    {
        su[c] += term*epsilon[c]/kc;
    }
    else
    {
        // The sink linearised on epsilon: sp*eps = -term*eps/k, so
        // sp = -term/k. No epsilon appears, which is what makes the
        // linearisation exact at the current value.
        sp[c] += -term/kc;
    }
}


// ==========================================================================
//  G_b into the omega equation - SPEC-LIT section 17
//
//      omega : + (gamma/nu_t) G_b
//
//  the same production route the shear production G takes in section 6.2,
//  where the omega equation's source is gamma*G/nu_t written as
//  gamma (omega/k) G. Here it is written against nu_t directly because that
//  is the form section 17 gives, and nu_t is floored so a laminar cell - where
//  nu_t is zero and there is no eddy transport of buoyancy either -
//  contributes nothing instead of infinity.
//
//  Split by sign as above, the sink linearised on omega.
// ==========================================================================

extern "C" __global__ void turbAddBuoyancyToOmega
(
    ofscalar* __restrict__ su,
    ofscalar* __restrict__ sp,
    const ofscalar* __restrict__ gb,
    const ofscalar* __restrict__ nut,
    const ofscalar* __restrict__ omega,
    ofscalar gamma,
    ofscalar nutMin,
    oflabel  stableBranch,
    oflabel  nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar b = gb[c];
    if (b < (ofscalar)0 && stableBranch == 0) return;

    const ofscalar nt = nut[c];
    if (!(nt > nutMin)) return;

    const ofscalar term = gamma*b/nt;

    if (term >= (ofscalar)0)
    {
        su[c] += term;
    }
    else
    {
        const ofscalar w = omega[c];
        if (w > (ofscalar)0) sp[c] += -term/w;
    }
}


// ==========================================================================
//  G_b into the omega equation, gamma read PER CELL - SPEC-LIT section 17
//  and section 30.2 ("k-omega and SST take the same production route
//  (gamma/nu_t) G_b in the omega equation").
//
//  Identical to turbAddBuoyancyToOmega above except that SST's gamma is not
//  one constant but F1's blend of gamma_1 and gamma_2 (SPEC-LIT section 6.3),
//  already built per cell by sstBlendCoeffs before this kernel runs - the
//  same relationship turbGammaInternalCell bears to turbGammaInternal.
// ==========================================================================

extern "C" __global__ void turbAddBuoyancyToOmegaCell
(
    ofscalar* __restrict__ su,
    ofscalar* __restrict__ sp,
    const ofscalar* __restrict__ gb,
    const ofscalar* __restrict__ nut,
    const ofscalar* __restrict__ omega,
    const ofscalar* __restrict__ gamma,
    ofscalar nutMin,
    oflabel  stableBranch,
    oflabel  nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar b = gb[c];
    if (b < (ofscalar)0 && stableBranch == 0) return;

    const ofscalar nt = nut[c];
    if (!(nt > nutMin)) return;

    const ofscalar term = gamma[c]*b/nt;

    if (term >= (ofscalar)0)
    {
        su[c] += term;
    }
    else
    {
        const ofscalar w = omega[c];
        if (w > (ofscalar)0) sp[c] += -term/w;
    }
}
