// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  fv.cu - the discrete finite-volume operators, device side.

  Written from:
    H. Jasak, "Error Analysis and Estimation for the Finite Volume Method with
      Applications to Fluid Flows", PhD thesis, Imperial College (1996), ch. 3
      - Gauss convection and diffusion, the over-relaxed non-orthogonal
        splitting (Jasak S3.4.2), the Green-Gauss gradient (Jasak S3.3), the
        TVD ratio on an unstructured mesh (Jasak S3.5)
    F. Moukalled, L. Mangani, M. Darwish, "The Finite Volume Method in
      Computational Fluid Dynamics", Springer (2016), ch. 8, 11, 12 and S15.4
      - the bounded convection correction
    S. V. Patankar, "Numerical Heat Transfer and Fluid Flow", Hemisphere (1980),
      ch. 4-6 - source linearisation with S_p <= 0
    J. H. Ferziger, M. Peric, "Computational Methods for Fluid Dynamics",
      S6.3.2 - three-level backward differencing (BDF2)
    P. K. Sweby, SIAM J. Numer. Anal. 21 (1984) 995 - the TVD framework
    B. van Leer, J. Comput. Phys. 23 (1977) 276 - the van Leer limiter
    G. D. van Albada, B. van Leer, W. W. Roberts, Astron. Astrophys. 108 (1982)
      76 - the van Albada limiter
    P. L. Roe, Ann. Rev. Fluid Mech. 18 (1986) 337 - minmod and Superbee
    B. van Leer, J. Comput. Phys. 32 (1979) 101 - MUSCL
    M. Darwish, F. Moukalled, Int. J. Heat Mass Transfer 46 (2003) 599 - the
      gradient ratio r for an unstructured mesh
    K. C. Khosla, S. G. Rubin, Computers & Fluids 2 (1974) 207 - the deferred
      correction pattern every scheme of S11 is assembled with
    R. F. Warming, R. M. Beam, AIAA J. 14 (1976) 1241 - second-order upwind
    B. P. Leonard, Comput. Methods Appl. Mech. Eng. 19 (1979) 59 - QUICK
    H. Jasak, H. G. Weller, A. D. Gosman, Int. J. Numer. Methods Fluids 31
      (1999) 431 - the Gamma NVD scheme
    W. H. Press et al., "Numerical Recipes", S3.3 - Hermite cubic interpolation
    T. J. Barth, D. C. Jespersen, AIAA 89-0366 (1989) - the cell-limited
      gradient
    V. Venkatakrishnan, AIAA 93-0880 (1993) - its differentiable variant
  and from rust/SPEC-LIT.md sections 2, 3, 4, 6, 7, 11 and 12, which cite all
  of the above. No GPL-licensed source was consulted.

  ---------------------------------------------------------------------------
  TWO STRUCTURAL RULES, which the rest of this file obeys without exception.

  1. GATHER, NEVER SCATTER. A cell's diagonal and its explicit sources are
     accumulated by one thread that walks that cell's own faces through the
     cell -> face CSR. No atomics on a double, one fixed summation order, and
     therefore a bitwise reproducible answer. Face coefficients (upper/lower)
     are written one thread per face, which collides with nothing either.

  2. EVERY OPERATOR'S DIAGONAL IS RECOMPUTED FROM ITS OWN INPUTS.
     Both the convection and the diffusion operator want
     "diag -= the sum of my own off-diagonal coefficients". The obvious way to
     get that is to read upper[]/lower[] back in the per-cell pass - and it is
     wrong, because by then a previously applied operator has already added its
     own coefficients to the same arrays and they would be subtracted a second
     time. So fvDivDiag recomputes w*phi and fvLapDiag recomputes
     gammaMagSf*deltaCoeffs from the same arrays the face pass read. It costs a
     multiply per face and it removes all cross-operator coupling: the
     operators may be applied in any order, any number of times.
\*---------------------------------------------------------------------------*/

#include "ofgpu_device.cuh"

// --------------------------------------------------------------------------
//  Patch kinds.
//
//  CAREFUL: these mirror `PatchKind` in src/mesh.rs, NOT `BcKind` in
//  src/field.rs. Every kernel below is handed the MESH's `b_kind`, which is
//  topology - what the patch IS - and the two enums number their shared names
//  differently:
//
//      PatchKind   Generic 0  Wall 1  Empty 2  Symmetry 3  Cyclic 4  Processor 5
//      BcKind      ... Calculated 4  Empty 5  Symmetry 6  Cyclic 7  InletOutlet 8
//
//  Comparing a `PatchKind` value against a `BcKind` discriminant compiles,
//  runs, and is silently wrong: `PatchKind::Processor` (5) would read as
//  `BcKind::Empty`, and a real empty patch (2) would never match at all - so
//  a 2-D case would integrate flux through its front and back planes.
//
//  `validate` has a check for exactly this ("an empty patch carries no flux"),
//  which is how the mismatch was found. Do not renumber either enum without
//  it.
//
//  Kernels that take a FIELD's `bc_kind` use the `OFGPU_BC_*` constants
//  instead; see cuda/field.cu. No kernel in this file takes one.
// --------------------------------------------------------------------------
#define OFPATCH_GENERIC   0
#define OFPATCH_WALL      1
#define OFPATCH_EMPTY     2
#define OFPATCH_SYMMETRY  3
#define OFPATCH_CYCLIC    4
#define OFPATCH_PROCESSOR 5
//- SPEC-LIT S47.4: a conformal conjugate interface. Topologically a cyclic
//  couple with a zero transform, so fvLapBoundary treats it EXACTLY as one
//  (the coupled coefficient of S47.9). Everywhere else it deliberately falls
//  into the UNCOUPLED branch and is read from the evaluated face value, which
//  is the only representation that can carry the contact-resistance jump -
//  the cyclic branch's geometric interpolation of psi[nbr] cannot. And
//  fvLapNonOrth skips it outright; see the note there.
#define OFPATCH_INTERFACE 6

// --------------------------------------------------------------------------
//  The single mixed boundary form - SPEC-LIT S4, which is our own design.
//
//      psi_b = fr*refValue + (1 - fr)*(psi_P + refGrad/Delta_b)
//
//  Differentiating that one expression in psi_P gives all four coefficients,
//  so a Dirichlet, a Neumann, a Robin and a wall function are the same three
//  lines of arithmetic with different numbers in (fr, refValue, refGrad).
// --------------------------------------------------------------------------

//- d(psi_b)/d(psi_P)
OFGPU_DEV ofscalar bcValueInternal(ofscalar fr)
{
    return 1 - fr;
}

//- the part of psi_b that does not depend on psi_P
OFGPU_DEV ofscalar bcValueBoundary
(
    ofscalar fr, ofscalar refValue, ofscalar refGrad, ofscalar delta
)
{
    return fr*refValue + (1 - fr)*refGrad/delta;
}

//- d(snGrad_b)/d(psi_P), where snGrad_b = Delta_b*(psi_b - psi_P)
OFGPU_DEV ofscalar bcGradInternal(ofscalar fr, ofscalar delta)
{
    return -fr*delta;
}

//- the part of snGrad_b that does not depend on psi_P
OFGPU_DEV ofscalar bcGradBoundary
(
    ofscalar fr, ofscalar refValue, ofscalar refGrad, ofscalar delta
)
{
    return fr*delta*refValue + (1 - fr)*refGrad;
}

OFGPU_DEV ofscalar ofabs_(ofscalar a) { return a < 0 ? -a : a; }


//- The over-relaxed correction vector of a BOUNDARY face.
//
//  Internal faces carry k_f in the mesh; boundary faces do not, so it is
//  rebuilt here from exactly the same definition (SPEC-LIT S2.4) with
//  d_b = Cf - C_P, the vector the boundary delta coefficient was formed from:
//
//      k_b = nf - d_b*Delta_b ,   nf = Sf/|Sf|
//
//  Without this the boundary flux estimate is Delta_b*(psi_b - psi_P), which
//  on a non-orthogonal mesh is not nf . grad psi at all, and the whole solve
//  drops to first order however well the interior is corrected. With it, a
//  linear field gives nf . grad psi exactly on the boundary as well as inside:
//
//      Delta_b (a.d_b) + (nf - d_b Delta_b).a = nf.a
OFGPU_DEV ofvec3 boundaryCorrVector
(
    const ofvec3& sf, ofscalar mag, const ofvec3& cf, const ofvec3& cp,
    ofscalar delta
)
{
    const ofscalar rm = mag > 0 ? 1/mag : (ofscalar)0;
    const ofvec3 d = mkvec(cf.x - cp.x, cf.y - cp.y, cf.z - cp.z);
    return mkvec
    (
        sf.x*rm - d.x*delta,
        sf.y*rm - d.y*delta,
        sf.z*rm - d.z*delta
    );
}


//- The limited surface-normal gradient of SPEC-LIT S12.3.
//
//  DESIGN - the expression is ours, not the literature's. S2.4 splits snGrad
//  into an implicit orthogonal part and an explicit correction. On a badly
//  non-orthogonal mesh the correction can exceed the orthogonal part, and the
//  laplacian then stops being diagonally dominant in that cell. Cap it at a
//  multiple of the orthogonal part:
//
//      scale = min(1, alpha |orth| / (|corr| + eps))
//
//  alpha < 0 is `corrected`   - unlimited, and the branch returns 1 with no
//                               arithmetic at all;
//  alpha = 0 is `uncorrected` - scale is EXACTLY zero, so `limited 0` and
//                               `uncorrected` produce bit-identical matrices;
//  alpha = 1 caps the correction at the orthogonal part.
//
//  Jasak (1996) S3.4.2 discusses the trade-off this parameterises.
OFGPU_DEV ofscalar snGradLimitScale(ofscalar alpha, ofscalar orth, ofscalar corr)
{
    if (alpha < 0) return 1;             // corrected: no limit
    if (alpha == 0) return 0;            // uncorrected, exactly

    const ofscalar eps = (ofscalar)1e-30;
    const ofscalar s = alpha*ofabs_(orth)/(ofabs_(corr) + eps);

    return s < 1 ? s : (ofscalar)1;
}


//- The interpolated face gradient, w*grad_P + (1-w)*grad_N.
OFGPU_DEV ofvec3 faceGrad(const ofvec3& gP, const ofvec3& gN, ofscalar w)
{
    return mkvec
    (
        w*gP.x + (1 - w)*gN.x,
        w*gP.y + (1 - w)*gN.y,
        w*gP.z + (1 - w)*gN.z
    );
}


// ==========================================================================
//  S3.3  Temporal derivative
// ==========================================================================

//- Euler implicit, Patankar S4.2 / SPEC-LIT S3.3.
//
//      V*(psi^n - psi^{n-1})/dt
//      diag += V*rDeltaT ;  source += V*rDeltaT*psi^{n-1}
//
//  `sign` is the factor the operator is added with, so a caller writing
//  "- ddt(psi)" passes -1 rather than negating anything by hand.
extern "C" __global__ void fvDdtEuler
(
    ofscalar* __restrict__ diag,
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ V,
    const ofscalar* __restrict__ psi0,
    ofscalar rDeltaT,
    ofscalar sign,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar a = sign*V[c]*rDeltaT;
    diag[c]   += a;
    source[c] += a*psi0[c];
}


//- Euler implicit with a density, d(rho psi)/dt. rho0 is the OLD-time density,
//  which is what makes the discrete form conserve rho*psi rather than psi.
extern "C" __global__ void fvDdtEulerRho
(
    ofscalar* __restrict__ diag,
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ V,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ rho0,
    const ofscalar* __restrict__ psi0,
    ofscalar rDeltaT,
    ofscalar sign,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar a = sign*V[c]*rDeltaT;
    diag[c]   += a*rho[c];
    source[c] += a*rho0[c]*psi0[c];
}


//- Second-order backward differencing, Ferziger & Peric S6.3.2, constant dt:
//
//      (3 psi^n - 4 psi^{n-1} + psi^{n-2})/(2 dt)
//      diag   += 3/2 V rDeltaT
//      source += V rDeltaT (2 psi^{n-1} - 1/2 psi^{n-2})
//
//  The first step of a run has no psi^{n-2} and must be taken with Euler; that
//  decision belongs to the caller, which knows the step number.
extern "C" __global__ void fvDdtBdf2
(
    ofscalar* __restrict__ diag,
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ V,
    const ofscalar* __restrict__ psi0,
    const ofscalar* __restrict__ psi00,
    ofscalar rDeltaT,
    ofscalar sign,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar a = sign*V[c]*rDeltaT;
    diag[c]   += (ofscalar)1.5*a;
    source[c] += a*((ofscalar)2*psi0[c] - (ofscalar)0.5*psi00[c]);
}


extern "C" __global__ void fvDdtBdf2Rho
(
    ofscalar* __restrict__ diag,
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ V,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ rho0,
    const ofscalar* __restrict__ rho00,
    const ofscalar* __restrict__ psi0,
    const ofscalar* __restrict__ psi00,
    ofscalar rDeltaT,
    ofscalar sign,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar a = sign*V[c]*rDeltaT;
    diag[c]   += (ofscalar)1.5*a*rho[c];
    source[c] += a*((ofscalar)2*rho0[c]*psi0[c] - (ofscalar)0.5*rho00[c]*psi00[c]);
}


// ==========================================================================
//  S7  Convection weights - central, upwind, and the TVD limiter family
// ==========================================================================

#define OFLIM_CENTRAL   0
#define OFLIM_UPWIND    1
#define OFLIM_MINMOD    2
#define OFLIM_VANLEER   3
#define OFLIM_VANALBADA 4
#define OFLIM_SUPERBEE  5
#define OFLIM_MUSCL     6
#define OFLIM_SWEBY     7

// SPEC-LIT S11: schemes that are still expressible as a limiter Psi(r), and so
// need no new assembly path - only a new branch in limiterPsi.
#define OFLIM_QUICK     8   // S11.3, clipped into the TVD region
#define OFLIM_QUICKU    9   // S11.3, unlimited: Psi = (3 + r)/4 as written
#define OFLIM_GAMMA     10  // S11.6, the NVD scheme of Jasak et al. (1999)
#define OFLIM_BLENDED   11  // S11.5, a constant central/upwind blend

//- Psi(r), the flux limiter. SPEC-LIT S7 table.
//
//  Two properties are required of every entry and are enforced here rather
//  than assumed:
//
//    Psi(r) = 0 for r <= 0   is what makes the scheme TVD. Of the six only van
//                            Albada fails to give it from its formula alone -
//                            (r^2+r)/(r^2+1) turns positive again below
//                            r = -1 - so the branch is taken for all of them.
//    Psi(1) = 1              is what makes it second order on smooth data.
//
//  r is clamped before use. Every limiter in the table has a finite limit as
//  r -> infinity (1, 2, 1, 2, 2, beta respectively), but van Leer and van
//  Albada reach theirs as inf/inf, which evaluates to NaN. Clamping to a large
//  finite ratio gives the limit to more digits than a double carries and
//  cannot produce a NaN.
OFGPU_DEV ofscalar limiterPsi(oflabel code, ofscalar r, ofscalar beta)
{
    // Two schemes are NOT limiters of the TVD family and must be answered
    // before the r <= 0 guard, because they are deliberately unbounded:
    //
    //   OFLIM_BLENDED  a constant blend (SPEC-LIT S11.5). Psi is the blend
    //                  factor gamma whatever r is; gamma = 1 is pure central,
    //                  0 pure upwind. It never reads r at all.
    //   OFLIM_QUICKU   unlimited QUICK (S11.3). Psi = (3 + r)/4 "as written",
    //                  which is the whole point of asking for the unlimited
    //                  form; clipping it at r <= 0 would make it the limited
    //                  one under a different name.
    if (code == OFLIM_BLENDED) return beta;

    if (code == OFLIM_QUICKU)
    {
        if (r != r) return 0;            // a NaN ratio must not propagate
        const ofscalar RC = (ofscalar)1e12;
        const ofscalar rc = r >  RC ?  RC : (r < -RC ? -RC : r);
        return (3 + rc)/4;
    }

    if (!(r > 0)) return 0;              // also catches NaN

    const ofscalar RMAX = (ofscalar)1e12;
    if (r > RMAX) r = RMAX;

    switch (code)
    {
        case OFLIM_MINMOD:
            return ofmin_(1, r);

        case OFLIM_VANLEER:
            return 2*r/(1 + r);          // (r + |r|)/(1 + |r|) for r > 0

        case OFLIM_VANALBADA:
            return (r*r + r)/(r*r + 1);

        case OFLIM_SUPERBEE:
            return ofmax_(ofmin_(2*r, 1), ofmin_(r, 2));

        case OFLIM_MUSCL:
            return ofmin_(ofmin_(2*r, (ofscalar)0.5*(r + 1)), 2);

        case OFLIM_SWEBY:
            return ofmax_(ofmin_(beta*r, 1), ofmin_(r, beta));

        // SPEC-LIT S11.3. QUICK expressed in TVD form is Psi = (3 + r)/4;
        // clipping it into the TVD region max(0, min(Psi, 2r, 2)) is what
        // makes it bounded, and a bare `QUICK` in fvSchemes selects THIS one
        // - see the DESIGN note in SPEC-LIT S11.3.
        case OFLIM_QUICK:
            return ofmax_(0, ofmin_(ofmin_((3 + r)/4, 2*r), 2));

        // SPEC-LIT S11.6, Jasak, Weller & Gosman (1999). The paper states the
        // scheme in the normalised variable
        //
        //     psi~ = 1 - (psi_N - psi_P)/(2 d . grad psi_U)
        //
        // and this file's limiters are stated in r = 2 d.grad psi_U/(psi_N -
        // psi_P) - 1. The two are the same number twice: writing
        // g = 2 d.grad psi_U and e = psi_N - psi_P, r = g/e - 1 so
        // g/e = 1 + r, and psi~ = 1 - e/g = 1 - 1/(1 + r) = r/(1 + r).
        // DERIVED here from the two definitions; the algebra is ours, the
        // scheme is the paper's.
        //
        // r > 0 is the whole of the NVD "bounded" window: r <= 0 covers both
        // the psi~ <= 0 exit (upwind cell gradient opposing the P->N
        // difference) and the psi~ >= 1 exit (an over-steep upwind gradient),
        // and both of those are upwind in the NVD diagram, which is what the
        // guard above already returns.
        case OFLIM_GAMMA:
        {
            const ofscalar psit = r/(1 + r);       // in (0,1) for r > 0
            if (psit >= beta) return 1;            // central
            return psit/beta;                      // blend, gamma = psi~/beta_m
        }

        default:
            return 1;                    // central; unreachable from the host
    }
}


//- The limited weight, as a weight rather than as a face value.
//
//  SPEC-LIT S7 writes the face value as upwind plus a limited correction,
//
//      psi_f = psi_U + Psi(r)*(psi_f,central - psi_U)
//
//  and this assembly wants w with psi_f = w*psi_P + (1-w)*psi_N. Substituting
//  psi_f,central = wc*psi_P + (1-wc)*psi_N and psi_U = psi_P (phi >= 0) or
//  psi_N (phi < 0) and collecting terms gives, in both cases,
//
//      w = w_upwind + Psi(r)*(wc - w_upwind)
//
//  so the limiter interpolates between the two schemes it is built from, and
//  Psi = 0 / Psi = 1 recover them exactly.
OFGPU_DEV ofscalar limitedWeight
(
    oflabel code, ofscalar beta,
    ofscalar phi, ofscalar wc,
    ofscalar psiP, ofscalar psiN,
    const ofvec3& d, const ofvec3& gradU
)
{
    const ofscalar wUp = phi >= 0 ? (ofscalar)1 : (ofscalar)0;

    const ofscalar den = psiN - psiP;

    // A face across which psi does not change at all: upwind and central give
    // the same face value, so the weight is immaterial, and central is the one
    // that keeps the operator second order where the field happens to be flat.
    if (den == 0) return wc;

    // r = 2 (d . grad psi_U)/(psi_N - psi_P) - 1, Jasak S3.5 / Darwish &
    // Moukalled (2003). d = C_N - C_P for BOTH upwind directions: flipping the
    // upwind cell flips the sign of d and of the denominator together, and the
    // two cancel, which is why this expression needs no branch on phi.
    const ofscalar r = 2*dot3(d, gradU)/den - 1;

    return wUp + limiterPsi(code, r, beta)*(wc - wUp);
}


//- Central, upwind, or a constant blend of the two, on the internal faces.
//  No gradient needed by any of the three.
//
//  The blend is SPEC-LIT S11.5: psi_f = (1-gamma) psi_f,upwind + gamma
//  psi_f,central. In weight form that is w = w_upwind + gamma (wc - w_upwind),
//  exactly the shape a limiter takes with Psi = gamma, which is why it lives
//  in this kernel rather than needing one of its own.
extern "C" __global__ void fvWeightsUnlimited
(
    ofscalar* __restrict__ w,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ wCentral,
    oflabel code,
    ofscalar beta,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;

    if (code == OFLIM_CENTRAL) { w[f] = wCentral[f]; return; }

    const ofscalar wUp = phi[f] >= 0 ? (ofscalar)1 : (ofscalar)0;

    w[f] = (code == OFLIM_BLENDED) ? wUp + beta*(wCentral[f] - wUp) : wUp;
}


//- TVD-limited weights on the internal faces.
//
//  This is the kernel that needs the UPWIND CELL GRADIENT, and the reason
//  div_scheme_weights on the host carries an optional grad(psi) at all: a
//  limited scheme cannot be assembled without it, and quietly falling back to
//  upwind when it is missing would turn a second-order run into a first-order
//  one with no diagnostic anywhere.
extern "C" __global__ void fvWeightsLimited
(
    ofscalar* __restrict__ w,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ wCentral,
    const ofscalar* __restrict__ psi,
    const ofvec3* __restrict__ gradPsi,
    const ofvec3* __restrict__ C,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    oflabel code,
    ofscalar beta,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;

    const oflabel o = owner[f];
    const oflabel n = neighbour[f];

    const ofvec3 cO = C[o];
    const ofvec3 cN = C[n];
    const ofvec3 d = mkvec(cN.x - cO.x, cN.y - cO.y, cN.z - cO.z);

    const ofvec3 gU = phi[f] >= 0 ? gradPsi[o] : gradPsi[n];

    w[f] = limitedWeight(code, beta, phi[f], wCentral[f], psi[o], psi[n], d, gU);
}


//- Boundary weights.
//
//  A weight only means anything on a COUPLED face, where there really are two
//  cells to interpolate between. On every other patch the face value comes
//  from (fr, refValue, refGrad) and the weight is never read, so 1 is written
//  there - "the face value is entirely the internal side's business", which is
//  what the uncoupled branch of fvDivBoundary effectively assumes.
extern "C" __global__ void fvWeightsBoundary
(
    ofscalar* __restrict__ bw,
    const ofscalar* __restrict__ bphi,
    const ofscalar* __restrict__ bWeights,
    const oflabel* __restrict__ bKind,
    oflabel code,
    ofscalar beta,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    if (bKind[i] != OFPATCH_CYCLIC)
    {
        bw[i] = 1;
        return;
    }

    if (code == OFLIM_CENTRAL) { bw[i] = bWeights[i]; return; }

    const ofscalar wUp = bphi[i] >= 0 ? (ofscalar)1 : (ofscalar)0;

    bw[i] = (code == OFLIM_BLENDED) ? wUp + beta*(bWeights[i] - wUp) : wUp;
}


//- Boundary weights for a limited scheme.
//
//  DESIGN. A cyclic couple has no centre-to-centre vector in this mesh's
//  arrays: the neighbour cell is physically somewhere else in the domain, so
//  C[nbr] - C[own] is the wrong vector entirely. The separation along the face
//  normal is however exactly 1/Delta_b by the definition of the delta
//  coefficient, so d_b = nf/Delta_b is used here, with nf = Sf/|Sf|. On a
//  matched cyclic pair - the only kind this solver builds - that IS the
//  couple's d; on a mismatched one it is its normal component, which is the
//  part the limiter's ratio is sensitive to. The choice is ours, not the
//  literature's.
extern "C" __global__ void fvWeightsBoundaryLimited
(
    ofscalar* __restrict__ bw,
    const ofscalar* __restrict__ bphi,
    const ofscalar* __restrict__ bWeights,
    const ofscalar* __restrict__ psi,
    const ofvec3* __restrict__ gradPsi,
    const ofvec3* __restrict__ bSf,
    const ofscalar* __restrict__ bMagSf,
    const ofscalar* __restrict__ bDeltaCoeffs,
    const oflabel* __restrict__ bFaceCells,
    const oflabel* __restrict__ bNbrCell,
    const oflabel* __restrict__ bKind,
    oflabel code,
    ofscalar beta,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    const oflabel nbr = bNbrCell[i];
    if (bKind[i] != OFPATCH_CYCLIC || nbr < 0)
    {
        bw[i] = 1;
        return;
    }

    const oflabel o = bFaceCells[i];

    const ofscalar mag = bMagSf[i];
    const ofscalar delta = bDeltaCoeffs[i];
    if (!(mag > 0) || !(delta > 0))
    {
        bw[i] = bWeights[i];
        return;
    }

    const ofvec3 s = bSf[i];
    const ofscalar sc = 1/(mag*delta);
    const ofvec3 d = mkvec(s.x*sc, s.y*sc, s.z*sc);

    const ofvec3 gU = bphi[i] >= 0 ? gradPsi[o] : gradPsi[nbr];

    bw[i] = limitedWeight
    (
        code, beta, bphi[i], bWeights[i], psi[o], psi[nbr], d, gU
    );
}


// ==========================================================================
//  S3.1  Convection - Gauss with face weights
// ==========================================================================

//- Off-diagonal coefficients, one thread per internal face.
//
//      psi_f = w psi_P + (1-w) psi_N
//      row P gains +phi_f psi_f, row N gains -phi_f psi_f
//
//  so A(P,N) = upper = (1-w) phi and A(N,P) = lower = -w phi.
extern "C" __global__ void fvDivFaces
(
    ofscalar* __restrict__ upper,
    ofscalar* __restrict__ lower,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ w,
    ofscalar sign,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;

    const ofscalar p = phi[f];
    const ofscalar wf = w[f];

    upper[f] += sign*(1 - wf)*p;
    lower[f] += sign*(-wf*p);
}


//- Diagonal, one thread per cell, RECOMPUTED from phi and w.
//
//  A(P,P) gains +w phi from a face it owns and -(1-w) phi from a face it
//  neighbours. Reading upper[]/lower[] instead would pick up whatever another
//  operator has already put there; see the header note.
extern "C" __global__ void fvDivDiag
(
    ofscalar* __restrict__ diag,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ w,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    ofscalar sign,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofscalar acc = 0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const ofscalar p = phi[f];
        const ofscalar wf = w[f];

        acc += cfOwn[j] ? wf*p : -(1 - wf)*p;
    }

    diag[c] += sign*acc;
}


//- Boundary contribution of the convection operator.
//
//  Row P gains phi_b*psi_b. With the mixed form psi_b = vic*psi_P + vbc that
//  splits into an implicit part and a known one:
//
//      internalCoeffs =  sign*phi_b*vic          folded into diag
//      boundaryCoeffs = -sign*phi_b*vbc          folded into source
//
//  On a coupled face psi_b = w*psi_P + (1-w)*psi_nbr instead, and the
//  neighbour term stays in the matrix: amul applies it as
//  Apsi[P] -= boundaryCoeffs*psi[nbr], hence the extra minus.
extern "C" __global__ void fvDivBoundary
(
    ofscalar* __restrict__ internalCoeffs,
    ofscalar* __restrict__ boundaryCoeffs,
    const ofscalar* __restrict__ bphi,
    const ofscalar* __restrict__ bw,
    const ofscalar* __restrict__ fr,
    const ofscalar* __restrict__ refValue,
    const ofscalar* __restrict__ refGrad,
    const ofscalar* __restrict__ bDeltaCoeffs,
    const oflabel* __restrict__ bKind,
    ofscalar sign,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    const oflabel kind = bKind[i];
    if (kind == OFPATCH_EMPTY) return;

    const ofscalar p = bphi[i];

    if (kind == OFPATCH_CYCLIC)
    {
        const ofscalar wf = bw[i];
        internalCoeffs[i] += sign*p*wf;
        boundaryCoeffs[i] += -sign*p*(1 - wf);
        return;
    }

    const ofscalar f0 = fr[i];
    internalCoeffs[i] += sign*p*bcValueInternal(f0);
    boundaryCoeffs[i] +=
        -sign*p*bcValueBoundary(f0, refValue[i], refGrad[i], bDeltaCoeffs[i]);
}


//- The bounded correction, Moukalled et al. S15.4 / SPEC-LIT S3.1.
//
//  Part-way through a pressure-velocity iteration the discrete flux is not
//  solenoidal, and the convection operator then injects a spurious source
//  proportional to psi*(sum_f phi_f). Subtracting V_P (div u)_P from the
//  diagonal removes it and restores boundedness; when phi IS conservative the
//  correction is identically zero and costs only the pass.
extern "C" __global__ void fvDivBoundedDiag
(
    ofscalar* __restrict__ diag,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ bphi,
    const oflabel* __restrict__ bKind,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    ofscalar sign,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofscalar acc = 0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        acc += cfOwn[j] ? phi[f] : -phi[f];
    }

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        const oflabel b = bcfFace[j];
        if (bKind[b] == OFPATCH_EMPTY) continue;
        acc += bphi[b];
    }

    diag[c] -= sign*acc;
}


// ==========================================================================
//  S11  Deferred correction - the explicit half of a higher-order scheme
// ==========================================================================
//
//  SPEC-LIT S11.1 (Khosla & Rubin 1974; Ferziger & Peric S5.6). A scheme that
//  cannot be written as a face weight is assembled as
//
//      implicit : upwind or central weights, which keep the matrix bounded
//      explicit : (psi_f,scheme - psi_f,implicit)*phi_f, into the source
//
//  At convergence the explicit term is evaluated with the same field the
//  matrix solves for, so the converged solution satisfies the full scheme
//  while the matrix keeps the diagonal dominance of its implicit base.
//
//  Sign convention, S11.1 verbatim: the correction enters the source as
//  -phi_f*corr_f for the OWNER and +phi_f*corr_f for the NEIGHBOUR, with
//  corr_f = psi_f,scheme - psi_f,implicit. Gathered per cell, as everything
//  else in this file is.

#define OFDIVCORR_NONE          0
#define OFDIVCORR_LINEARUPWIND  1   // SPEC-LIT S11.2
#define OFDIVCORR_CUBIC         2   // SPEC-LIT S11.4

//- corr_f for one face, given the two cells, their gradients, and the face
//  centre offset from each of them.
//
//  linearUpwind (S11.2, Warming & Beam 1976). The implicit base is upwind, so
//  psi_f,implicit = psi_U and the correction is exactly the extrapolation
//  term:
//
//      corr_f = (C_f - C_U) . grad(psi)_U
//
//  `coef` scales it, which is what makes the central/second-order-upwind blend
//  of S11.5 fall out of the same kernel: that blend is
//  psi_f = (1-gamma) psi_f,linearUpwind + gamma psi_f,central, whose implicit
//  part is the blended weight (handled in fvWeightsUnlimited) and whose
//  explicit part is (1-gamma) times this correction.
//
//  cubic (S11.4, Hermite). The implicit base is CENTRAL and the correction is
//  the Hermite bracket:
//
//      corr_f = [ d . grad(psi)_P - d . grad(psi)_N ] / 8,   d = C_N - C_P
OFGPU_DEV ofscalar divCorrection
(
    oflabel code, ofscalar coef, ofscalar phi,
    const ofvec3& dP, const ofvec3& dN, const ofvec3& d,
    const ofvec3& gP, const ofvec3& gN
)
{
    if (code == OFDIVCORR_LINEARUPWIND)
    {
        return phi >= 0 ? coef*dot3(dP, gP) : coef*dot3(dN, gN);
    }

    if (code == OFDIVCORR_CUBIC)
    {
        return coef*(dot3(d, gP) - dot3(d, gN))/8;
    }

    return 0;
}


//- The explicit correction of a convection scheme, gathered into the source.
//
//  Cyclic couples are included. They have no centre-to-centre vector in this
//  mesh's arrays - the neighbour cell is physically elsewhere - so the same
//  DESIGN convention fvWeightsBoundaryLimited uses is applied here:
//  d_b = nf/Delta_b, the separation along the face normal, which IS the
//  couple's d on the matched pairs this solver builds. The face centre then
//  sits at (1-w_b) d_b from the owner and -w_b d_b from the neighbour, which
//  is what the two offsets below say. Every other patch takes its face value
//  from the (fr, refValue, refGrad) triple of S4 and has no interpolation to
//  correct.
extern "C" __global__ void fvDivCorrection
(
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ bphi,
    const ofvec3* __restrict__ gradPsi,
    const ofvec3* __restrict__ C,
    const ofvec3* __restrict__ Cf,
    const ofvec3* __restrict__ bSf,
    const ofscalar* __restrict__ bMagSf,
    const ofscalar* __restrict__ bDeltaCoeffs,
    const ofscalar* __restrict__ bWeights,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ bFaceCells,
    const oflabel* __restrict__ bNbrCell,
    const oflabel* __restrict__ bKind,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    oflabel code,
    ofscalar coef,
    ofscalar sign,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofscalar acc = 0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];

        const oflabel o = owner[f];
        const oflabel n = neighbour[f];

        const ofvec3 cO = C[o];
        const ofvec3 cN = C[n];
        const ofvec3 cf = Cf[f];

        const ofvec3 dP = mkvec(cf.x - cO.x, cf.y - cO.y, cf.z - cO.z);
        const ofvec3 dN = mkvec(cf.x - cN.x, cf.y - cN.y, cf.z - cN.z);
        const ofvec3 d  = mkvec(cN.x - cO.x, cN.y - cO.y, cN.z - cO.z);

        const ofscalar corr = divCorrection
        (
            code, coef, phi[f], dP, dN, d, gradPsi[o], gradPsi[n]
        );

        const ofscalar t = phi[f]*corr;

        // -phi*corr for the owner, +phi*corr for the neighbour; `acc` is
        // subtracted from the source at the end, so the owner accumulates +t.
        acc += cfOwn[j] ? t : -t;
    }

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        const oflabel b = bcfFace[j];
        const oflabel nbr = bNbrCell[b];
        if (bKind[b] != OFPATCH_CYCLIC || nbr < 0) continue;

        const ofscalar mag = bMagSf[b];
        const ofscalar delta = bDeltaCoeffs[b];
        if (!(mag > 0) || !(delta > 0)) continue;

        const ofvec3 sb = bSf[b];
        const ofscalar sc = 1/(mag*delta);
        const ofvec3 d = mkvec(sb.x*sc, sb.y*sc, sb.z*sc);

        const ofscalar wb = bWeights[b];
        const ofvec3 dP = mkvec((1 - wb)*d.x, (1 - wb)*d.y, (1 - wb)*d.z);
        const ofvec3 dN = mkvec(-wb*d.x, -wb*d.y, -wb*d.z);

        const oflabel o = bFaceCells[b];

        const ofscalar corr = divCorrection
        (
            code, coef, bphi[b], dP, dN, d, gradPsi[o], gradPsi[nbr]
        );

        // The cell is always the owner of its own boundary face.
        acc += bphi[b]*corr;
    }

    source[c] -= sign*acc;
}


//- The face value a convection scheme actually forms, as a face field.
//
//  psi_f = w psi_P + (1 - w) psi_N + corr_f
//
//  i.e. the implicit interpolation the weights describe PLUS the deferred
//  correction of S11.1, which is the whole scheme and not half of it. Nothing
//  in the assembly needs this - fvDivFaces and fvDivCorrection between them
//  put the same arithmetic into the matrix and the source - but the order of
//  accuracy of a scheme is a statement about THIS number, and a test that
//  measures it against the exact value at the face centre is measuring the
//  scheme rather than the quadrature rule the divergence is built on.
extern "C" __global__ void fvDivSchemeFaceValue
(
    ofscalar* __restrict__ psif,
    const ofscalar* __restrict__ psi,
    const ofscalar* __restrict__ w,
    const ofscalar* __restrict__ phi,
    const ofvec3* __restrict__ gradPsi,
    const ofvec3* __restrict__ C,
    const ofvec3* __restrict__ Cf,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    oflabel code,
    ofscalar coef,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;

    const oflabel o = owner[f];
    const oflabel n = neighbour[f];

    const ofvec3 cO = C[o];
    const ofvec3 cN = C[n];
    const ofvec3 cf = Cf[f];

    const ofvec3 dP = mkvec(cf.x - cO.x, cf.y - cO.y, cf.z - cO.z);
    const ofvec3 dN = mkvec(cf.x - cN.x, cf.y - cN.y, cf.z - cN.z);
    const ofvec3 d  = mkvec(cN.x - cO.x, cN.y - cO.y, cN.z - cO.z);

    const ofscalar corr = divCorrection
    (
        code, coef, phi[f], dP, dN, d, gradPsi[o], gradPsi[n]
    );

    psif[f] = w[f]*psi[o] + (1 - w[f])*psi[n] + corr;
}


// ==========================================================================
//  S3.2  Diffusion - Gauss laplacian
// ==========================================================================

//- Off-diagonal coefficients. The implicit, orthogonal part of
//  snGrad = Delta*(psi_N - psi_P) + k . (grad psi)_f is symmetric:
//
//      upper[f] = lower[f] = gamma_f |Sf| Delta_f
extern "C" __global__ void fvLapFaces
(
    ofscalar* __restrict__ upper,
    ofscalar* __restrict__ lower,
    const ofscalar* __restrict__ gammaMagSf,
    const ofscalar* __restrict__ deltaCoeffs,
    ofscalar sign,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;

    const ofscalar coef = sign*gammaMagSf[f]*deltaCoeffs[f];

    upper[f] += coef;
    lower[f] += coef;
}


//- Diagonal, RECOMPUTED from gammaMagSf and deltaCoeffs. Both rows of a face
//  lose the same coefficient, so no owner/neighbour branch is needed - but
//  cfOwn stays in the signature because the CSR argument order is fixed
//  crate-wide.
extern "C" __global__ void fvLapDiag
(
    ofscalar* __restrict__ diag,
    const ofscalar* __restrict__ gammaMagSf,
    const ofscalar* __restrict__ deltaCoeffs,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    ofscalar sign,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofscalar acc = 0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        acc += gammaMagSf[f]*deltaCoeffs[f];
    }

    diag[c] -= sign*acc;

    (void)cfOwn;
}


//- Boundary contribution. Row P gains gamma_b |Sf_b| snGrad_b, and snGrad_b
//  differentiates into (gic, gbc) exactly as SPEC-LIT S4 sets out.
extern "C" __global__ void fvLapBoundary
(
    ofscalar* __restrict__ internalCoeffs,
    ofscalar* __restrict__ boundaryCoeffs,
    const ofscalar* __restrict__ bGammaMagSf,
    const ofscalar* __restrict__ bDeltaCoeffs,
    const ofscalar* __restrict__ fr,
    const ofscalar* __restrict__ refValue,
    const ofscalar* __restrict__ refGrad,
    const oflabel* __restrict__ bKind,
    ofscalar sign,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    const oflabel kind = bKind[i];
    if (kind == OFPATCH_EMPTY) return;

    const ofscalar g = bGammaMagSf[i];
    const ofscalar delta = bDeltaCoeffs[i];

    if (kind == OFPATCH_CYCLIC)
    {
        // snGrad = Delta_b (psi_nbr - psi_P) across the couple. The neighbour
        // term stays implicit, and amul applies it as -boundaryCoeffs*psi_nbr.
        const ofscalar coef = sign*g*delta;
        internalCoeffs[i] += -coef;
        boundaryCoeffs[i] += -coef;
        return;
    }

    if (kind == OFPATCH_INTERFACE)
    {
        // SPEC-LIT S47.3. Structurally the same coupled entry as the cyclic
        // branch above, with the series conductance h_G |Sf| of (S47.9) in
        // place of a harmonic interpolation - and taken DIRECTLY, not as
        // g*delta.
        //
        // WHY THE DELTA IS NOT APPLIED HERE, and it is not a shortcut.
        // S47.2 consequence 2 requires the two faces of a couple to carry the
        // BITWISE SAME coefficient, or the matrix is asymmetric and PCG and
        // DIC - which S48.3's check exists to guard - are being run on a
        // system that has no symmetry. Writing bGammaMagSf as h_G|Sf|/Delta_i
        // and multiplying it back by Delta_i here is a division and a
        // multiplication by two DIFFERENT numbers on the two sides, and
        // x/y*y is not x in floating point: measured, the two entries then
        // differed by about one ulp. Passing the coefficient itself removes
        // the round trip and makes the equality exact by construction, which
        // is what the design claimed and what `ofgpu-validate` checks.
        //
        // So on an interface face bGammaMagSf holds h_G |Sf| (W/K) - the
        // coefficient - rather than gamma |Sf|. That is a different quantity
        // from every other face's, and it is safe because exactly one thing
        // reads it there: this branch. fvLapNonOrth skips interface faces
        // outright (see the note there), and nothing else in the crate is
        // handed a conjugate mesh's bGammaMagSf.
        const ofscalar coef = sign*g;
        internalCoeffs[i] += -coef;
        boundaryCoeffs[i] += -coef;
        return;
    }

    const ofscalar f0 = fr[i];
    internalCoeffs[i] += sign*g*bcGradInternal(f0, delta);
    boundaryCoeffs[i] += -sign*g*bcGradBoundary(f0, refValue[i], refGrad[i], delta);
}


//- The explicit non-orthogonal correction, Jasak S3.4.3.
//
//      source_P -= sign * [ sum_f  (+-1) gamma_f |Sf| ( k_f . (grad psi)_f )
//                         + sum_b  fr_b  gamma_b |Sf_b| ( k_b . (grad psi)_P ) ]
//
//  with k the over-relaxed correction vector of SPEC-LIT S2.4 and
//  (grad psi)_f the linear interpolation of the two cell gradients. Deferred
//  to the source and iterated: with nNonOrthogonalCorrectors extra passes,
//  grad psi is recomputed from the latest solution each pass.
//
//  The BOUNDARY term is scaled by the valueFraction fr, and that factor is the
//  whole of the reasoning. In the mixed form of S4,
//
//      snGrad_b = fr*Delta_b*(refValue - psi_P) + (1 - fr)*refGrad
//
//  the fr part is an ESTIMATE of the normal gradient obtained by differencing
//  across d_b, and on a non-orthogonal mesh that estimate is wrong by exactly
//  k_b . grad psi. The (1 - fr) part is a PRESCRIBED normal gradient, which is
//  already the normal gradient and has nothing to correct. Leaving the
//  boundary term out entirely - which is tempting, because the mixed form
//  "already gives snGrad" - costs a full order of accuracy on a
//  non-orthogonal mesh; the test
//  fv::tests::the_non_orthogonal_correction_restores_second_order measures it.
//
//  Coupled (cyclic) faces are NOT skipped: SPEC-LIT S2.4's over-relaxed split
//  applies to them exactly as it does to an internal face - a matched cyclic
//  pair IS one internal face folded in half - using `bNonOrthCorr`, which the
//  host geometry sweep (`mesh/geometry.rs::compute`) builds from the `d` that
//  spans the periodic image, not from `Cf - C_P`. `boundaryCorrVector` below
//  remains correct for an UNCOUPLED patch, where the condition is imposed
//  directly on the face and `d = Cf - C_P` really is the right separation.
extern "C" __global__ void fvLapNonOrth
(
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ gammaMagSf,
    const ofvec3* __restrict__ nonOrthCorr,
    const ofscalar* __restrict__ w,
    const ofscalar* __restrict__ deltaCoeffs,
    const ofscalar* __restrict__ psi,
    const ofvec3* __restrict__ gradPsi,
    const ofscalar* __restrict__ bGammaMagSf,
    const ofvec3* __restrict__ bSf,
    const ofscalar* __restrict__ bMagSf,
    const ofvec3* __restrict__ bCf,
    const ofscalar* __restrict__ bDeltaCoeffs,
    const ofvec3* __restrict__ bNonOrthCorr,
    const oflabel* __restrict__ bNbrCell,
    const ofscalar* __restrict__ bWeights,
    const ofscalar* __restrict__ fr,
    const ofscalar* __restrict__ refValue,
    const ofscalar* __restrict__ refGrad,
    const ofvec3* __restrict__ C,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ bKind,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    ofscalar alpha,
    ofscalar sign,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofscalar acc = 0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];

        const oflabel o = owner[f];
        const oflabel n = neighbour[f];

        const ofvec3 gf = faceGrad(gradPsi[o], gradPsi[n], w[f]);

        const ofscalar corr = dot3(nonOrthCorr[f], gf);
        const ofscalar orth = deltaCoeffs[f]*(psi[n] - psi[o]);

        const ofscalar t =
            gammaMagSf[f]*snGradLimitScale(alpha, orth, corr)*corr;

        acc += cfOwn[j] ? t : -t;
    }

    const ofvec3 gP = gradPsi[c];

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        const oflabel b = bcfFace[j];
        const oflabel kind = bKind[b];
        if (kind == OFPATCH_EMPTY) continue;

        //- SPEC-LIT S47.3. A conjugate interface is SKIPPED, deliberately and
        //  with the accuracy cost stated. Across it `kappa` is discontinuous
        //  and, with a contact resistance, so is `T` itself: interpolating
        //  the two cells' gradients - what the cyclic branch below does - has
        //  no physical meaning there, and using only the owner-side gradient
        //  is inconsistent with the two-point flux the interface assembles.
        //  Neither is defensible, so v1 computes neither: the host reports
        //  the interface non-orthogonality at setup and REFUSES above a
        //  threshold rather than adding a term that is wrong. On an
        //  orthogonal interface - the only kind S47.4's pairing accepts - the
        //  correction is zero anyway, so this changes no supported answer.
        if (kind == OFPATCH_INTERFACE) continue;

        ofscalar corr;
        ofscalar orth;

        if (kind == OFPATCH_CYCLIC)
        {
            const oflabel n = bNbrCell[b];
            if (n < 0) continue;   // named but not yet paired: no couple yet

            // The couple is one internal face folded in half, so the
            // gradient is interpolated across it exactly as an internal
            // face's is, with `bWeights[b]` this face's share of the split.
            const ofvec3 gf = faceGrad(gP, gradPsi[n], bWeights[b]);

            corr = dot3(bNonOrthCorr[b], gf);
            orth = bDeltaCoeffs[b]*(psi[n] - psi[c]);
        }
        else
        {
            const ofvec3 kb = boundaryCorrVector
            (
                bSf[b], bMagSf[b], bCf[b], C[c], bDeltaCoeffs[b]
            );

            corr = fr[b]*dot3(kb, gP);

            // The orthogonal part of snGrad_b implied by the mixed form of
            // S4, which is what S12.3 compares the correction against.
            orth =
                bcGradInternal(fr[b], bDeltaCoeffs[b])*psi[c]
              + bcGradBoundary(fr[b], refValue[b], refGrad[b], bDeltaCoeffs[b]);
        }

        acc += bGammaMagSf[b]*snGradLimitScale(alpha, orth, corr)*corr;
    }

    source[c] -= sign*acc;
}


//- The explicit SKEWNESS correction of the laplacian - SPEC-LIT S74.4.
//
//      source_P -= sign * sum_f (+-1) gamma_f |Sf| Delta_f
//                         ( s_f . [ (grad psi)_N - (grad psi)_P ] )
//
//  with s_f = Cf - (C_P + (1 - w) d) the skewness vector of SPEC-LIT S2.5.
//
//  WHY this is the term. `fvLapFaces` differences psi along d and `fvLapNonOrth`
//  rotates the result onto the face normal; between them they give the normal
//  gradient on the line P-N, which pierces the face plane at C_P + (1 - w) d.
//  On a skewed face that is NOT the face centroid, and the midpoint rule the
//  whole finite-volume statement rests on wants the centroid. Shifting BOTH
//  cell values onto a line through the centroid, psi_P -> psi_P + s.(grad psi)_P
//  and psi_N -> psi_N + s.(grad psi)_N - Ferziger & Peric S8.6 - changes the
//  differenced pair by exactly the expression above and leaves d, Delta and k
//  alone. It is therefore additive on top of the non-orthogonal correction and
//  not a replacement for it.
//
//  Internal faces ONLY. An uncoupled boundary face's value is imposed AT the
//  face, at Cf, so there is nothing to move; a cyclic couple can be skewed and
//  is NOT corrected here - SPEC-LIT S74.7 refuses that combination by name.
//
//  Unlimited on purpose: `skewCorrected` is the unlimited member of S12.3's
//  family, and a limiter on a term whose orthogonal counterpart is a different
//  face's would have nothing to compare against.
extern "C" __global__ void fvLapSkew
(
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ gammaMagSf,
    const ofvec3* __restrict__ skewCorr,
    const ofscalar* __restrict__ deltaCoeffs,
    const ofvec3* __restrict__ gradPsi,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    ofscalar sign,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofscalar acc = 0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];

        const ofvec3 s  = skewCorr[f];
        const ofvec3 gO = gradPsi[owner[f]];
        const ofvec3 gN = gradPsi[neighbour[f]];

        const ofscalar jump =
            s.x*(gN.x - gO.x) + s.y*(gN.y - gO.y) + s.z*(gN.z - gO.z);

        const ofscalar t = gammaMagSf[f]*deltaCoeffs[f]*jump;

        acc += cfOwn[j] ? t : -t;
    }

    source[c] -= sign*acc;
}


// ==========================================================================
//  S3.4  Source terms - Patankar's linearisation
// ==========================================================================

//- An implicit sink whose sign the caller has already decided:
//      diag += sign * V * sp
extern "C" __global__ void fvSp
(
    ofscalar* __restrict__ diag,
    const ofscalar* __restrict__ V,
    const ofscalar* __restrict__ sp,
    ofscalar sign,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    diag[c] += sign*V[c]*sp[c];
}


//- Patankar's rule for a source of unknown sign (S4.2, SPEC-LIT S3.4):
//  whichever part stabilises the matrix goes on the diagonal, the rest goes to
//  the right-hand side evaluated at the current psi.
//
//      diag   += sign * V * max(S, 0)
//      source -= sign * V * min(S, 0) * psi_P
//
//  The two branches agree to the last bit when S >= 0, and when S < 0 the
//  explicit branch is exactly what a fully implicit treatment would have moved
//  across - it is a stability choice, not an approximation.
extern "C" __global__ void fvSusp
(
    ofscalar* __restrict__ diag,
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ V,
    const ofscalar* __restrict__ susp,
    const ofscalar* __restrict__ psi,
    ofscalar sign,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar s = susp[c];
    const ofscalar v = sign*V[c];

    diag[c]   += v*ofmax_(s, 0);
    source[c] -= v*ofmin_(s, 0)*psi[c];
}


//- A wholly explicit source: source += sign * V * su
extern "C" __global__ void fvSu
(
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ V,
    const ofscalar* __restrict__ su,
    ofscalar sign,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    source[c] += sign*V[c]*su[c];
}


// ==========================================================================
//  S3.5  Explicit operators
// ==========================================================================

//- Green-Gauss gradient of a cell scalar field, Jasak S3.3:
//
//      (grad psi)_P = (1/V_P) sum_f (+-Sf) psi_f
//
//  Empty faces contribute nothing: they are the 2-D front and back, and their
//  area vectors point along the direction the case does not resolve. On a
//  prismatic 2-D mesh every remaining face has zero area component in that
//  direction, so the corresponding gradient component comes out exactly zero
//  rather than merely small.
extern "C" __global__ void fvGradScalar
(
    ofvec3* __restrict__ g,
    const ofscalar* __restrict__ psi,
    const ofscalar* __restrict__ bpsi,
    const ofscalar* __restrict__ w,
    const ofvec3* __restrict__ Sf,
    const ofvec3* __restrict__ bSf,
    const ofscalar* __restrict__ V,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
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

    ofscalar ax = 0, ay = 0, az = 0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const ofscalar wf = w[f];
        const ofscalar pf = wf*psi[owner[f]] + (1 - wf)*psi[neighbour[f]];

        const ofvec3 s = Sf[f];
        const ofscalar v = cfOwn[j] ? pf : -pf;

        ax += s.x*v; ay += s.y*v; az += s.z*v;
    }

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        const oflabel b = bcfFace[j];
        if (bKind[b] == OFPATCH_EMPTY) continue;

        const ofvec3 s = bSf[b];
        const ofscalar pf = bpsi[b];

        ax += s.x*pf; ay += s.y*pf; az += s.z*pf;
    }

    const ofscalar rv = 1/V[c];
    g[c] = mkvec(ax*rv, ay*rv, az*rv);
}


//- The deferred SKEWNESS correction of a Green-Gauss SCALAR gradient -
//  SPEC-LIT S74.4.
//
//      (grad psi)_P += (1/V_P) sum_f (+-Sf) [ (grad psi)_f . s_f ]
//
//  Green-Gauss is exact for a linear field only when psi_f is the value at the
//  face CENTROID. `fvGradScalar` places it where the face plane cuts P-N, so
//  on a skewed face it is short by (grad psi).s_f. Adding that back is a
//  deferred correction - it reads a gradient to compute a gradient - and one
//  pass is enough for a linear field, which is the property the correction
//  exists to restore.
//
//  ADDS to `g`; it does not overwrite it. Boundary faces contribute nothing,
//  for the same reason as in `fvLapSkew`.
extern "C" __global__ void fvGradScalarSkew
(
    ofvec3* __restrict__ g,
    const ofvec3* __restrict__ gradPrev,
    const ofvec3* __restrict__ skewCorr,
    const ofscalar* __restrict__ w,
    const ofvec3* __restrict__ Sf,
    const ofscalar* __restrict__ V,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofscalar ax = 0, ay = 0, az = 0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];

        const ofvec3 gf =
            faceGrad(gradPrev[owner[f]], gradPrev[neighbour[f]], w[f]);

        const ofscalar dpsi = dot3(gf, skewCorr[f]);

        const ofvec3 s = Sf[f];
        const ofscalar v = cfOwn[j] ? dpsi : -dpsi;

        ax += s.x*v; ay += s.y*v; az += s.z*v;
    }

    const ofscalar rv = 1/V[c];
    const ofvec3 g0 = g[c];
    g[c] = mkvec(g0.x + ax*rv, g0.y + ay*rv, g0.z + az*rv);
}


//- The same for a cell VECTOR field, whose gradient is a tensor.
//
//  The face value is short by (s_f . grad U), a VECTOR whose j-th component is
//  s_i (grad U)_ij - the area vector supplies the first index, SPEC-LIT S1 -
//  and the correction to the tensor is Sf (x) that vector.
extern "C" __global__ void fvGradVectorSkew
(
    oftensor* __restrict__ g,
    const oftensor* __restrict__ gradPrev,
    const ofvec3* __restrict__ skewCorr,
    const ofscalar* __restrict__ w,
    const ofvec3* __restrict__ Sf,
    const ofscalar* __restrict__ V,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    oftensor t;
    t.xx = 0; t.xy = 0; t.xz = 0;
    t.yx = 0; t.yy = 0; t.yz = 0;
    t.zx = 0; t.zy = 0; t.zz = 0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const ofscalar wf = w[f];

        const oftensor a = gradPrev[owner[f]];
        const oftensor b = gradPrev[neighbour[f]];
        const ofvec3 sk = skewCorr[f];

        const ofscalar gxx = wf*a.xx + (1 - wf)*b.xx;
        const ofscalar gxy = wf*a.xy + (1 - wf)*b.xy;
        const ofscalar gxz = wf*a.xz + (1 - wf)*b.xz;
        const ofscalar gyx = wf*a.yx + (1 - wf)*b.yx;
        const ofscalar gyy = wf*a.yy + (1 - wf)*b.yy;
        const ofscalar gyz = wf*a.yz + (1 - wf)*b.yz;
        const ofscalar gzx = wf*a.zx + (1 - wf)*b.zx;
        const ofscalar gzy = wf*a.zy + (1 - wf)*b.zy;
        const ofscalar gzz = wf*a.zz + (1 - wf)*b.zz;

        const ofvec3 du = mkvec
        (
            sk.x*gxx + sk.y*gyx + sk.z*gzx,
            sk.x*gxy + sk.y*gyy + sk.z*gzy,
            sk.x*gxz + sk.y*gyz + sk.z*gzz
        );

        const ofvec3 s0 = Sf[f];
        const ofscalar sg = cfOwn[j] ? (ofscalar)1 : (ofscalar)-1;
        const ofvec3 s = mkvec(sg*s0.x, sg*s0.y, sg*s0.z);

        t.xx += s.x*du.x; t.xy += s.x*du.y; t.xz += s.x*du.z;
        t.yx += s.y*du.x; t.yy += s.y*du.y; t.yz += s.y*du.z;
        t.zx += s.z*du.x; t.zy += s.z*du.y; t.zz += s.z*du.z;
    }

    const ofscalar rv = 1/V[c];
    oftensor g0 = g[c];

    g0.xx += t.xx*rv; g0.xy += t.xy*rv; g0.xz += t.xz*rv;
    g0.yx += t.yx*rv; g0.yy += t.yy*rv; g0.yz += t.yz*rv;
    g0.zx += t.zx*rv; g0.zy += t.zy*rv; g0.zz += t.zz*rv;

    g[c] = g0;
}


//- Green-Gauss gradient of a cell vector field:
//
//      (grad U)_P = (1/V_P) sum_f (+-Sf) (x) U_f
//
//  Component (i,j) is dU_j/dx_i, because the area vector supplies the first
//  index - SPEC-LIT S1, and the convention src/types.rs is pinned to.
extern "C" __global__ void fvGradVector
(
    oftensor* __restrict__ g,
    const ofvec3* __restrict__ u,
    const ofvec3* __restrict__ bu,
    const ofscalar* __restrict__ w,
    const ofvec3* __restrict__ Sf,
    const ofvec3* __restrict__ bSf,
    const ofscalar* __restrict__ V,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
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

    oftensor t;
    t.xx = 0; t.xy = 0; t.xz = 0;
    t.yx = 0; t.yy = 0; t.yz = 0;
    t.zx = 0; t.zy = 0; t.zz = 0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const ofscalar wf = w[f];

        const ofvec3 uo = u[owner[f]];
        const ofvec3 un = u[neighbour[f]];
        const ofvec3 uf = mkvec
        (
            wf*uo.x + (1 - wf)*un.x,
            wf*uo.y + (1 - wf)*un.y,
            wf*uo.z + (1 - wf)*un.z
        );

        const ofvec3 s0 = Sf[f];
        const ofscalar sg = cfOwn[j] ? (ofscalar)1 : (ofscalar)-1;
        const ofvec3 s = mkvec(sg*s0.x, sg*s0.y, sg*s0.z);

        t.xx += s.x*uf.x; t.xy += s.x*uf.y; t.xz += s.x*uf.z;
        t.yx += s.y*uf.x; t.yy += s.y*uf.y; t.yz += s.y*uf.z;
        t.zx += s.z*uf.x; t.zy += s.z*uf.y; t.zz += s.z*uf.z;
    }

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        const oflabel b = bcfFace[j];
        if (bKind[b] == OFPATCH_EMPTY) continue;

        const ofvec3 s = bSf[b];
        const ofvec3 uf = bu[b];

        t.xx += s.x*uf.x; t.xy += s.x*uf.y; t.xz += s.x*uf.z;
        t.yx += s.y*uf.x; t.yy += s.y*uf.y; t.yz += s.y*uf.z;
        t.zx += s.z*uf.x; t.zy += s.z*uf.y; t.zz += s.z*uf.z;
    }

    const ofscalar rv = 1/V[c];

    t.xx *= rv; t.xy *= rv; t.xz *= rv;
    t.yx *= rv; t.yy *= rv; t.yz *= rv;
    t.zx *= rv; t.zy *= rv; t.zz *= rv;

    g[c] = t;
}


// ==========================================================================
//  S12.1  Least-squares gradient
// ==========================================================================
//
//  SPEC-LIT S3.5 / S12.1 (Jasak 1996 S3.3.2; Moukalled et al. S9.3). Per cell,
//  the gradient that best reproduces the neighbour differences in the
//  inverse-distance-weighted least-squares sense:
//
//      G_P = sum_N w_N^2 d_N (x) d_N          3x3 symmetric
//      (grad psi)_P = G_P^{-1} . sum_N w_N^2 d_N (psi_N - psi_P)
//      w_N = 1/|d_N|
//
//  DESIGN, two points, both ours:
//
//  1. SPEC-LIT says "invert once at setup". G is rebuilt and inverted inside
//     this kernel on every call instead. It is 30-odd flops against a gather
//     that already touches every neighbour, it needs no new mesh state and no
//     new invalidation rule, and it keeps a moving mesh correct for free. If
//     the profile ever says otherwise, the cached form is a drop-in.
//
//  2. EMPTY boundary faces are INCLUDED, with a value difference of zero. On a
//     2-D prismatic mesh the front and back planes are the only faces with a
//     separation component in the unresolved direction; leaving them out makes
//     G exactly singular. Including them with zero difference is the statement
//     "psi does not vary in that direction", which is precisely what an empty
//     patch means, and it makes the component come out exactly zero rather
//     than merely small - the same argument fvReconstruct makes below.

//- The symmetric 3x3 inverse, by cofactors. Returns 0 when the matrix is
//  singular, which on a real mesh means a cell with no usable neighbour at
//  all; the caller then writes a zero gradient rather than a NaN one.
OFGPU_DEV int invSym3
(
    ofscalar xx, ofscalar xy, ofscalar xz,
    ofscalar yy, ofscalar yz, ofscalar zz,
    ofscalar* inv                       // 6 entries: xx xy xz yy yz zz
)
{
    const ofscalar cxx = yy*zz - yz*yz;
    const ofscalar cxy = xz*yz - xy*zz;
    const ofscalar cxz = xy*yz - xz*yy;

    const ofscalar det = xx*cxx + xy*cxy + xz*cxz;
    if (!(det > 0) && !(det < 0)) return 0;

    const ofscalar rd = 1/det;

    inv[0] = cxx*rd;
    inv[1] = cxy*rd;
    inv[2] = cxz*rd;
    inv[3] = (xx*zz - xz*xz)*rd;
    inv[4] = (xy*xz - xx*yz)*rd;
    inv[5] = (xx*yy - xy*xy)*rd;

    return 1;
}

OFGPU_DEV ofvec3 applySym3(const ofscalar* inv, ofscalar bx, ofscalar by, ofscalar bz)
{
    return mkvec
    (
        inv[0]*bx + inv[1]*by + inv[2]*bz,
        inv[1]*bx + inv[3]*by + inv[4]*bz,
        inv[2]*bx + inv[4]*by + inv[5]*bz
    );
}


//- The separation vector this cell should use for one of its boundary faces.
//
//  A CYCLIC couple's neighbour cell is physically somewhere else in the
//  domain, so C[nbr] - C[P] is the wrong vector entirely; the separation along
//  the face normal is exactly 1/Delta_b by the definition of the boundary
//  delta coefficient, which is the same DESIGN convention
//  fvWeightsBoundaryLimited uses. Every other patch differences against the
//  face centre, where its value lives.
OFGPU_DEV ofvec3 boundarySeparation
(
    oflabel kind, oflabel nbr,
    const ofvec3& sf, ofscalar mag, ofscalar delta,
    const ofvec3& cf, const ofvec3& cp
)
{
    if (kind == OFPATCH_CYCLIC && nbr >= 0 && mag > 0 && delta > 0)
    {
        const ofscalar sc = 1/(mag*delta);
        return mkvec(sf.x*sc, sf.y*sc, sf.z*sc);
    }

    return mkvec(cf.x - cp.x, cf.y - cp.y, cf.z - cp.z);
}


extern "C" __global__ void fvGradScalarLeastSquares
(
    ofvec3* __restrict__ g,
    const ofscalar* __restrict__ psi,
    const ofscalar* __restrict__ bpsi,
    const ofvec3* __restrict__ C,
    const ofvec3* __restrict__ bSf,
    const ofscalar* __restrict__ bMagSf,
    const ofvec3* __restrict__ bCf,
    const ofscalar* __restrict__ bDeltaCoeffs,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ bNbrCell,
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

    const ofvec3 cp = C[c];
    const ofscalar psiP = psi[c];

    ofscalar xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
    ofscalar bx = 0, by = 0, bz = 0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const oflabel nb = cfOwn[j] ? neighbour[f] : owner[f];

        const ofvec3 cn = C[nb];
        const ofvec3 d = mkvec(cn.x - cp.x, cn.y - cp.y, cn.z - cp.z);

        const ofscalar d2 = dot3(d, d);
        if (!(d2 > 0)) continue;
        const ofscalar w2 = 1/d2;                 // w = 1/|d|

        xx += w2*d.x*d.x; xy += w2*d.x*d.y; xz += w2*d.x*d.z;
        yy += w2*d.y*d.y; yz += w2*d.y*d.z; zz += w2*d.z*d.z;

        const ofscalar dv = w2*(psi[nb] - psiP);
        bx += dv*d.x; by += dv*d.y; bz += dv*d.z;
    }

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        const oflabel b = bcfFace[j];
        const oflabel kind = bKind[b];
        const oflabel nbr = bNbrCell[b];

        const ofvec3 d = boundarySeparation
        (
            kind, nbr, bSf[b], bMagSf[b], bDeltaCoeffs[b], bCf[b], cp
        );

        const ofscalar d2 = dot3(d, d);
        if (!(d2 > 0)) continue;
        const ofscalar w2 = 1/d2;

        xx += w2*d.x*d.x; xy += w2*d.x*d.y; xz += w2*d.x*d.z;
        yy += w2*d.y*d.y; yz += w2*d.y*d.z; zz += w2*d.z*d.z;

        // An empty patch contributes the constraint "no variation in this
        // direction", i.e. a difference of exactly zero - see the note above.
        ofscalar diff = 0;
        if (kind != OFPATCH_EMPTY)
        {
            diff = (kind == OFPATCH_CYCLIC && nbr >= 0) ? psi[nbr] - psiP
                                                        : bpsi[b] - psiP;
        }

        const ofscalar dv = w2*diff;
        bx += dv*d.x; by += dv*d.y; bz += dv*d.z;
    }

    ofscalar inv[6];
    if (!invSym3(xx, xy, xz, yy, yz, zz, inv))
    {
        g[c] = mkvec(0, 0, 0);
        return;
    }

    g[c] = applySym3(inv, bx, by, bz);
}


//- The same for a vector field; component (i,j) of the result is dU_j/dx_i,
//  the convention of SPEC-LIT S1. One G, three right-hand sides.
extern "C" __global__ void fvGradVectorLeastSquares
(
    oftensor* __restrict__ g,
    const ofvec3* __restrict__ u,
    const ofvec3* __restrict__ bu,
    const ofvec3* __restrict__ C,
    const ofvec3* __restrict__ bSf,
    const ofscalar* __restrict__ bMagSf,
    const ofvec3* __restrict__ bCf,
    const ofscalar* __restrict__ bDeltaCoeffs,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ bNbrCell,
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

    const ofvec3 cp = C[c];
    const ofvec3 uP = u[c];

    ofscalar xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
    ofvec3 bX = mkvec(0, 0, 0);          // rhs for component x
    ofvec3 bY = mkvec(0, 0, 0);
    ofvec3 bZ = mkvec(0, 0, 0);

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const oflabel nb = cfOwn[j] ? neighbour[f] : owner[f];

        const ofvec3 cn = C[nb];
        const ofvec3 d = mkvec(cn.x - cp.x, cn.y - cp.y, cn.z - cp.z);

        const ofscalar d2 = dot3(d, d);
        if (!(d2 > 0)) continue;
        const ofscalar w2 = 1/d2;

        xx += w2*d.x*d.x; xy += w2*d.x*d.y; xz += w2*d.x*d.z;
        yy += w2*d.y*d.y; yz += w2*d.y*d.z; zz += w2*d.z*d.z;

        const ofvec3 un = u[nb];
        const ofscalar ax = w2*(un.x - uP.x);
        const ofscalar ay = w2*(un.y - uP.y);
        const ofscalar az = w2*(un.z - uP.z);

        bX.x += ax*d.x; bX.y += ax*d.y; bX.z += ax*d.z;
        bY.x += ay*d.x; bY.y += ay*d.y; bY.z += ay*d.z;
        bZ.x += az*d.x; bZ.y += az*d.y; bZ.z += az*d.z;
    }

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        const oflabel b = bcfFace[j];
        const oflabel kind = bKind[b];
        const oflabel nbr = bNbrCell[b];

        const ofvec3 d = boundarySeparation
        (
            kind, nbr, bSf[b], bMagSf[b], bDeltaCoeffs[b], bCf[b], cp
        );

        const ofscalar d2 = dot3(d, d);
        if (!(d2 > 0)) continue;
        const ofscalar w2 = 1/d2;

        xx += w2*d.x*d.x; xy += w2*d.x*d.y; xz += w2*d.x*d.z;
        yy += w2*d.y*d.y; yz += w2*d.y*d.z; zz += w2*d.z*d.z;

        ofvec3 un = uP;                  // empty patch: zero difference
        if (kind != OFPATCH_EMPTY)
        {
            un = (kind == OFPATCH_CYCLIC && nbr >= 0) ? u[nbr] : bu[b];
        }

        const ofscalar ax = w2*(un.x - uP.x);
        const ofscalar ay = w2*(un.y - uP.y);
        const ofscalar az = w2*(un.z - uP.z);

        bX.x += ax*d.x; bX.y += ax*d.y; bX.z += ax*d.z;
        bY.x += ay*d.x; bY.y += ay*d.y; bY.z += ay*d.z;
        bZ.x += az*d.x; bZ.y += az*d.y; bZ.z += az*d.z;
    }

    oftensor t;
    ofscalar inv[6];
    if (!invSym3(xx, xy, xz, yy, yz, zz, inv))
    {
        t.xx = 0; t.xy = 0; t.xz = 0;
        t.yx = 0; t.yy = 0; t.yz = 0;
        t.zx = 0; t.zy = 0; t.zz = 0;
        g[c] = t;
        return;
    }

    const ofvec3 gx = applySym3(inv, bX.x, bX.y, bX.z);   // grad of U_x
    const ofvec3 gy = applySym3(inv, bY.x, bY.y, bY.z);
    const ofvec3 gz = applySym3(inv, bZ.x, bZ.y, bZ.z);

    t.xx = gx.x; t.xy = gy.x; t.xz = gz.x;
    t.yx = gx.y; t.yy = gy.y; t.yz = gz.y;
    t.zx = gx.z; t.zy = gy.z; t.zz = gz.z;

    g[c] = t;
}


// ==========================================================================
//  S12.2  Cell-limited and face-limited gradients
// ==========================================================================
//
//  Barth & Jespersen, AIAA 89-0366 (1989); Venkatakrishnan, AIAA 93-0880
//  (1993). An unlimited gradient can extrapolate a face value outside the
//  range of the cell and its neighbours, which is exactly how a second-order
//  scheme manufactures a new extremum. Scale the whole gradient down until it
//  cannot:
//
//      for each face f of P:
//          d_f = (C_f - C_P) . grad(psi)_P
//          y = (psi_max - psi_P)/d_f   for d_f > 0
//          y = (psi_min - psi_P)/d_f   for d_f < 0
//          limiter = min(limiter, Phi(y))
//      grad(psi)_P *= limiter
//
//  cellLimited takes psi_min/psi_max over P and ALL its neighbours;
//  faceLimited - DERIVED, SPEC-LIT S12.2 gives only the cell-limited
//  algorithm - takes them over just the two cells of each face in turn, which
//  is the same statement made face by face and is therefore never weaker.

#define OFGRADLIM_BJ      0   // Phi(y) = min(1, y)
#define OFGRADLIM_VENKAT  1   // Phi(y) = (y^2 + 2y)/(y^2 + y + 2)

#define OFGRADMODE_CELL   0
#define OFGRADMODE_FACE   1

OFGPU_DEV ofscalar gradLimiterPhi(oflabel kind, ofscalar y)
{
    if (!(y > 0)) return 0;              // also catches NaN

    const ofscalar YMAX = (ofscalar)1e12;
    if (y > YMAX) return 1;              // both forms are 1 in the limit

    if (kind == OFGRADLIM_VENKAT)
    {
        // Differentiable, which is what stops the limiter chattering between
        // iterations and lets a steady solve actually converge.
        return (y*y + 2*y)/(y*y + y + 2);
    }

    return y < 1 ? y : (ofscalar)1;      // Barth-Jespersen
}


//- One face's contribution to the limiter factor.
OFGPU_DEV ofscalar gradLimitFace
(
    oflabel kind, ofscalar psiP, ofscalar lo, ofscalar hi, ofscalar df
)
{
    if (df > 0) return gradLimiterPhi(kind, (hi - psiP)/df);
    if (df < 0) return gradLimiterPhi(kind, (lo - psiP)/df);
    return 1;                            // the face value is psi_P exactly
}


//- Relax the cell bounds by (1 - coeff) of the local range.
//
//  SPEC-LIT S12.2: coeff 1 applies the limiter fully, 0 disables it, and
//  intermediate values relax psi_min/psi_max by that fraction of the local
//  range. coeff <= 0 is handled by the caller, which skips the pass entirely.
OFGPU_DEV void relaxBounds(ofscalar coeff, ofscalar* lo, ofscalar* hi)
{
    const ofscalar rng = (1 - coeff)*(*hi - *lo);
    *lo -= rng;
    *hi += rng;
}


extern "C" __global__ void fvGradLimitScalar
(
    ofvec3* __restrict__ g,
    const ofscalar* __restrict__ psi,
    const ofscalar* __restrict__ bpsi,
    const ofvec3* __restrict__ C,
    const ofvec3* __restrict__ Cf,
    const ofvec3* __restrict__ bCf,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ bNbrCell,
    const oflabel* __restrict__ bKind,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    oflabel limKind,
    oflabel mode,
    ofscalar coeff,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofvec3 cp = C[c];
    const ofscalar psiP = psi[c];
    const ofvec3 gP = g[c];

    ofscalar lo = psiP, hi = psiP;

    if (mode == OFGRADMODE_CELL)
    {
        for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
        {
            const oflabel f = cfFace[j];
            const ofscalar v = psi[cfOwn[j] ? neighbour[f] : owner[f]];
            lo = ofmin_(lo, v); hi = ofmax_(hi, v);
        }
        for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
        {
            const oflabel b = bcfFace[j];
            if (bKind[b] == OFPATCH_EMPTY) continue;
            const oflabel nbr = bNbrCell[b];
            const ofscalar v =
                (bKind[b] == OFPATCH_CYCLIC && nbr >= 0) ? psi[nbr] : bpsi[b];
            lo = ofmin_(lo, v); hi = ofmax_(hi, v);
        }
        relaxBounds(coeff, &lo, &hi);
    }

    ofscalar lim = 1;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const ofvec3 cf = Cf[f];
        const ofscalar df =
            gP.x*(cf.x - cp.x) + gP.y*(cf.y - cp.y) + gP.z*(cf.z - cp.z);

        ofscalar flo = lo, fhi = hi;
        if (mode == OFGRADMODE_FACE)
        {
            const ofscalar v = psi[cfOwn[j] ? neighbour[f] : owner[f]];
            flo = ofmin_(psiP, v); fhi = ofmax_(psiP, v);
            relaxBounds(coeff, &flo, &fhi);
        }

        lim = ofmin_(lim, gradLimitFace(limKind, psiP, flo, fhi, df));
    }

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        const oflabel b = bcfFace[j];
        if (bKind[b] == OFPATCH_EMPTY) continue;

        const ofvec3 cf = bCf[b];
        const ofscalar df =
            gP.x*(cf.x - cp.x) + gP.y*(cf.y - cp.y) + gP.z*(cf.z - cp.z);

        ofscalar flo = lo, fhi = hi;
        if (mode == OFGRADMODE_FACE)
        {
            const oflabel nbr = bNbrCell[b];
            const ofscalar v =
                (bKind[b] == OFPATCH_CYCLIC && nbr >= 0) ? psi[nbr] : bpsi[b];
            flo = ofmin_(psiP, v); fhi = ofmax_(psiP, v);
            relaxBounds(coeff, &flo, &fhi);
        }

        lim = ofmin_(lim, gradLimitFace(limKind, psiP, flo, fhi, df));
    }

    g[c] = mkvec(gP.x*lim, gP.y*lim, gP.z*lim);
}


//- The same for a tensor gradient.
//
//  SPEC-LIT S12.2, DESIGN: each component is limited with its own factor and
//  the MINIMUM across the three is applied to the whole tensor, so the limited
//  gradient stays frame-consistent rather than deforming the tensor.
extern "C" __global__ void fvGradLimitVector
(
    oftensor* __restrict__ g,
    const ofvec3* __restrict__ u,
    const ofvec3* __restrict__ bu,
    const ofvec3* __restrict__ C,
    const ofvec3* __restrict__ Cf,
    const ofvec3* __restrict__ bCf,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ bNbrCell,
    const oflabel* __restrict__ bKind,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    oflabel limKind,
    oflabel mode,
    ofscalar coeff,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofvec3 cp = C[c];
    const ofvec3 uP = u[c];
    oftensor t = g[c];

    const ofscalar pv[3] = { uP.x, uP.y, uP.z };

    // Column j of the tensor is grad(U_j) - component (i,j) is dU_j/dx_i.
    const ofvec3 gc[3] =
    {
        mkvec(t.xx, t.yx, t.zx),
        mkvec(t.xy, t.yy, t.zy),
        mkvec(t.xz, t.yz, t.zz)
    };

    ofscalar lo[3] = { uP.x, uP.y, uP.z };
    ofscalar hi[3] = { uP.x, uP.y, uP.z };

    if (mode == OFGRADMODE_CELL)
    {
        for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
        {
            const oflabel f = cfFace[j];
            const ofvec3 v = u[cfOwn[j] ? neighbour[f] : owner[f]];
            const ofscalar vv[3] = { v.x, v.y, v.z };
            for (int k = 0; k < 3; ++k)
            {
                lo[k] = ofmin_(lo[k], vv[k]); hi[k] = ofmax_(hi[k], vv[k]);
            }
        }
        for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
        {
            const oflabel b = bcfFace[j];
            if (bKind[b] == OFPATCH_EMPTY) continue;
            const oflabel nbr = bNbrCell[b];
            const ofvec3 v =
                (bKind[b] == OFPATCH_CYCLIC && nbr >= 0) ? u[nbr] : bu[b];
            const ofscalar vv[3] = { v.x, v.y, v.z };
            for (int k = 0; k < 3; ++k)
            {
                lo[k] = ofmin_(lo[k], vv[k]); hi[k] = ofmax_(hi[k], vv[k]);
            }
        }
        for (int k = 0; k < 3; ++k) relaxBounds(coeff, &lo[k], &hi[k]);
    }

    ofscalar lim = 1;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const ofvec3 cf = Cf[f];
        const ofvec3 dr = mkvec(cf.x - cp.x, cf.y - cp.y, cf.z - cp.z);

        ofscalar flo[3], fhi[3];
        if (mode == OFGRADMODE_FACE)
        {
            const ofvec3 v = u[cfOwn[j] ? neighbour[f] : owner[f]];
            const ofscalar vv[3] = { v.x, v.y, v.z };
            for (int k = 0; k < 3; ++k)
            {
                flo[k] = ofmin_(pv[k], vv[k]); fhi[k] = ofmax_(pv[k], vv[k]);
                relaxBounds(coeff, &flo[k], &fhi[k]);
            }
        }
        else
        {
            for (int k = 0; k < 3; ++k) { flo[k] = lo[k]; fhi[k] = hi[k]; }
        }

        for (int k = 0; k < 3; ++k)
        {
            lim = ofmin_
            (
                lim,
                gradLimitFace(limKind, pv[k], flo[k], fhi[k], dot3(dr, gc[k]))
            );
        }
    }

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        const oflabel b = bcfFace[j];
        if (bKind[b] == OFPATCH_EMPTY) continue;

        const ofvec3 cf = bCf[b];
        const ofvec3 dr = mkvec(cf.x - cp.x, cf.y - cp.y, cf.z - cp.z);

        ofscalar flo[3], fhi[3];
        if (mode == OFGRADMODE_FACE)
        {
            const oflabel nbr = bNbrCell[b];
            const ofvec3 v =
                (bKind[b] == OFPATCH_CYCLIC && nbr >= 0) ? u[nbr] : bu[b];
            const ofscalar vv[3] = { v.x, v.y, v.z };
            for (int k = 0; k < 3; ++k)
            {
                flo[k] = ofmin_(pv[k], vv[k]); fhi[k] = ofmax_(pv[k], vv[k]);
                relaxBounds(coeff, &flo[k], &fhi[k]);
            }
        }
        else
        {
            for (int k = 0; k < 3; ++k) { flo[k] = lo[k]; fhi[k] = hi[k]; }
        }

        for (int k = 0; k < 3; ++k)
        {
            lim = ofmin_
            (
                lim,
                gradLimitFace(limKind, pv[k], flo[k], fhi[k], dot3(dr, gc[k]))
            );
        }
    }

    t.xx *= lim; t.xy *= lim; t.xz *= lim;
    t.yx *= lim; t.yy *= lim; t.yz *= lim;
    t.zx *= lim; t.zy *= lim; t.zz *= lim;

    g[c] = t;
}


//- Divergence of a face flux: (div phi)_P = (1/V_P) sum_f (+-phi_f).
//
//  Volumetric, so that fvm::Sp(fvc::div(phi), psi) - the bounded convection
//  correction - reproduces sum_f (+-phi_f) exactly after the V multiply.
extern "C" __global__ void fvDivSurface
(
    ofscalar* __restrict__ d,
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
        const oflabel f = cfFace[j];
        acc += cfOwn[j] ? phi[f] : -phi[f];
    }

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        const oflabel b = bcfFace[j];
        if (bKind[b] == OFPATCH_EMPTY) continue;
        acc += bphi[b];
    }

    d[c] = acc/V[c];
}


//- Linear interpolation of a cell scalar onto the internal faces.
extern "C" __global__ void fvInterpolateLinear
(
    ofscalar* __restrict__ f,
    const ofscalar* __restrict__ psi,
    const ofscalar* __restrict__ w,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    oflabel nFaces
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nFaces) return;

    const ofscalar wf = w[i];
    f[i] = wf*psi[owner[i]] + (1 - wf)*psi[neighbour[i]];
}


//- The boundary half of the same: the face value IS the boundary value.
extern "C" __global__ void fvInterpolateBoundary
(
    ofscalar* __restrict__ bf,
    const ofscalar* __restrict__ bpsi,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    bf[i] = bpsi[i];
}


//- phi_f = interpolate(U)_f . Sf, on the internal faces.
//
//  NOT a conservative flux - see src/potential_flow.rs for why - but the
//  starting point a SIMPLE loop needs, with the Rhie-Chow correction applied
//  to it rather than replacing it.
extern "C" __global__ void fvFluxInternal
(
    ofscalar* __restrict__ phi,
    const ofvec3* __restrict__ u,
    const ofscalar* __restrict__ w,
    const ofvec3* __restrict__ Sf,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    oflabel nFaces
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nFaces) return;

    const ofscalar wf = w[i];
    const ofvec3 uo = u[owner[i]];
    const ofvec3 un = u[neighbour[i]];

    const ofvec3 uf = mkvec
    (
        wf*uo.x + (1 - wf)*un.x,
        wf*uo.y + (1 - wf)*un.y,
        wf*uo.z + (1 - wf)*un.z
    );

    phi[i] = dot3(uf, Sf[i]);
}


//- phi_b = U_b . Sf_b, zero on an empty patch, which carries no flux at all.
extern "C" __global__ void fvFluxBoundary
(
    ofscalar* __restrict__ bphi,
    const ofvec3* __restrict__ bu,
    const ofvec3* __restrict__ bSf,
    const oflabel* __restrict__ bKind,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    bphi[i] = (bKind[i] == OFPATCH_EMPTY) ? (ofscalar)0 : dot3(bu[i], bSf[i]);
}


//- The diffusive face flux gamma_f |Sf| snGrad(psi)_f, internal faces.
//
//  Written with the same coefficients in the same multiplication order as
//  fvLapFaces, so that a flux read off after a laplacian solve satisfies the
//  discrete conservation statement the matrix enforced, to the last bit.
extern "C" __global__ void fvSnGradFluxInternal
(
    ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ psi,
    const ofscalar* __restrict__ gammaMagSf,
    const ofscalar* __restrict__ deltaCoeffs,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;

    const ofscalar coef = gammaMagSf[f]*deltaCoeffs[f];
    phi[f] = coef*(psi[neighbour[f]] - psi[owner[f]]);
}


//- The same on the boundary, rebuilt from (fr, refValue, refGrad) rather than
//  from an evaluated boundary value, so the flux matches what fvLapBoundary
//  put in the matrix whether or not the field's faces have been corrected.
extern "C" __global__ void fvSnGradFluxBoundary
(
    ofscalar* __restrict__ bphi,
    const ofscalar* __restrict__ psi,
    const ofscalar* __restrict__ bGammaMagSf,
    const ofscalar* __restrict__ bDeltaCoeffs,
    const ofscalar* __restrict__ fr,
    const ofscalar* __restrict__ refValue,
    const ofscalar* __restrict__ refGrad,
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
        bphi[i] = 0;
        return;
    }

    const oflabel o = bFaceCells[i];
    const ofscalar g = bGammaMagSf[i];
    const ofscalar delta = bDeltaCoeffs[i];

    if (kind == OFPATCH_CYCLIC)
    {
        const oflabel n = bNbrCell[i];
        const ofscalar coef = g*delta;
        bphi[i] = (n >= 0) ? coef*(psi[n] - psi[o]) : (ofscalar)0;
        return;
    }

    const ofscalar f0 = fr[i];

    // g*(gic*psi_P + gbc), expanded so each product appears in the same order
    // as it does in fvLapBoundary.
    bphi[i] = g*bcGradInternal(f0, delta)*psi[o]
            + g*bcGradBoundary(f0, refValue[i], refGrad[i], delta);
}


//- The non-orthogonal correction of the same flux, ADDED to an existing phi.
//
//  fvSnGradFluxInternal / fvSnGradFluxBoundary reproduce fvm_laplacian exactly;
//  these two reproduce fvm_laplacian_non_orth_correction exactly. Apply both
//  pairs and the flux is the full operator, so a cell's fluxes sum to the
//  residual of the row the matrix solved - which is what keeps
//  phi = phi_HbyA - rAUf |Sf| snGrad(p) conservative on a non-orthogonal mesh.
extern "C" __global__ void fvSnGradCorrInternal
(
    ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ gammaMagSf,
    const ofvec3* __restrict__ nonOrthCorr,
    const ofscalar* __restrict__ w,
    const ofscalar* __restrict__ deltaCoeffs,
    const ofscalar* __restrict__ psi,
    const ofvec3* __restrict__ gradPsi,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    ofscalar alpha,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;

    const oflabel o = owner[f];
    const oflabel n = neighbour[f];

    const ofvec3 gf = faceGrad(gradPsi[o], gradPsi[n], w[f]);

    const ofscalar corr = dot3(nonOrthCorr[f], gf);
    const ofscalar orth = deltaCoeffs[f]*(psi[n] - psi[o]);

    phi[f] += gammaMagSf[f]*snGradLimitScale(alpha, orth, corr)*corr;
}


//- Coupled (cyclic) faces are NOT skipped, for the same reason as
//  `fvLapNonOrth`: this must reproduce `fvm_laplacian_non_orth_correction`
//  exactly (see the pair's doc above), and that function no longer skips
//  them either. `bNonOrthCorr` carries the correction through the periodic
//  image; `bWeights`/`bNbrCell` let the face gradient be interpolated
//  between the two coupled cells exactly as an internal face's is.
extern "C" __global__ void fvSnGradCorrBoundary
(
    ofscalar* __restrict__ bphi,
    const ofscalar* __restrict__ bGammaMagSf,
    const ofvec3* __restrict__ bSf,
    const ofscalar* __restrict__ bMagSf,
    const ofvec3* __restrict__ bCf,
    const ofscalar* __restrict__ bDeltaCoeffs,
    const ofvec3* __restrict__ bNonOrthCorr,
    const oflabel* __restrict__ bNbrCell,
    const ofscalar* __restrict__ bWeights,
    const ofscalar* __restrict__ fr,
    const ofscalar* __restrict__ refValue,
    const ofscalar* __restrict__ refGrad,
    const ofscalar* __restrict__ psi,
    const ofvec3* __restrict__ C,
    const ofvec3* __restrict__ gradPsi,
    const oflabel* __restrict__ bFaceCells,
    const oflabel* __restrict__ bKind,
    ofscalar alpha,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    const oflabel kind = bKind[i];
    if (kind == OFPATCH_EMPTY) return;

    const oflabel o = bFaceCells[i];

    ofscalar corr;
    ofscalar orth;

    if (kind == OFPATCH_CYCLIC)
    {
        const oflabel n = bNbrCell[i];
        if (n < 0) return;   // named but not yet paired: no couple yet

        const ofvec3 gf = faceGrad(gradPsi[o], gradPsi[n], bWeights[i]);
        corr = dot3(bNonOrthCorr[i], gf);
        orth = bDeltaCoeffs[i]*(psi[n] - psi[o]);
    }
    else
    {
        const ofvec3 kb = boundaryCorrVector
        (
            bSf[i], bMagSf[i], bCf[i], C[o], bDeltaCoeffs[i]
        );

        corr = fr[i]*dot3(kb, gradPsi[o]);

        orth =
            bcGradInternal(fr[i], bDeltaCoeffs[i])*psi[o]
          + bcGradBoundary(fr[i], refValue[i], refGrad[i], bDeltaCoeffs[i]);
    }

    bphi[i] += bGammaMagSf[i]*snGradLimitScale(alpha, orth, corr)*corr;
}


//- Reconstruct a cell vector from a face flux.
//
//  DERIVED, because SPEC-LIT does not give a formula for it. We want the cell
//  vector U_P whose own face fluxes best reproduce the given ones. Weighting
//  each face residual by 1/|Sf|, so a large face does not dominate a small one
//  purely by area,
//
//      minimise  sum_f (1/|Sf|) (U . Sf - phi_f)^2
//
//  and the normal equations of that least-squares problem are
//
//      [ sum_f (Sf (x) Sf)/|Sf| ] . U = sum_f (Sf/|Sf|) phi_f
//
//  The owner/neighbour sign cancels on both sides - flipping Sf flips phi_f
//  with it - so no face sign appears below. Empty faces are deliberately
//  INCLUDED: on a 2-D mesh they are the only faces with an area component in
//  the unresolved direction, and without them the 3x3 system is singular.
//  Their flux is zero, so what they contribute is precisely the constraint
//  U . n = 0 in that direction.
//- The SKEWNESS correction of the diffusive face flux - SPEC-LIT S74.4, the
//  face-indexed twin of `fvLapSkew`. Written in the same multiplication order,
//  so a flux read off after a skewCorrected solve matches the row the matrix
//  enforced to the last bit.
//
//  Internal faces only, and there is deliberately no boundary twin: neither
//  `fvLapSkew` nor this one touches a boundary face.
extern "C" __global__ void fvSnGradSkewInternal
(
    ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ gammaMagSf,
    const ofvec3* __restrict__ skewCorr,
    const ofscalar* __restrict__ deltaCoeffs,
    const ofvec3* __restrict__ gradPsi,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    oflabel nFaces
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nFaces) return;

    const ofvec3 s  = skewCorr[i];
    const ofvec3 gO = gradPsi[owner[i]];
    const ofvec3 gN = gradPsi[neighbour[i]];

    const ofscalar jump =
        s.x*(gN.x - gO.x) + s.y*(gN.y - gO.y) + s.z*(gN.z - gO.z);

    phi[i] += gammaMagSf[i]*deltaCoeffs[i]*jump;
}


extern "C" __global__ void fvReconstruct
(
    ofvec3* __restrict__ u,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ bphi,
    const ofvec3* __restrict__ Sf,
    const ofscalar* __restrict__ magSf,
    const ofvec3* __restrict__ bSf,
    const ofscalar* __restrict__ bMagSf,
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

    ofscalar txx = 0, txy = 0, txz = 0, tyy = 0, tyz = 0, tzz = 0;
    ofscalar rx = 0, ry = 0, rz = 0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const ofscalar m = magSf[f];
        if (!(m > 0)) continue;

        const ofvec3 s = Sf[f];
        const ofscalar rm = 1/m;

        txx += s.x*s.x*rm; txy += s.x*s.y*rm; txz += s.x*s.z*rm;
        tyy += s.y*s.y*rm; tyz += s.y*s.z*rm; tzz += s.z*s.z*rm;

        const ofscalar p = phi[f]*rm;
        rx += s.x*p; ry += s.y*p; rz += s.z*p;
    }

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        const oflabel b = bcfFace[j];
        const ofscalar m = bMagSf[b];
        if (!(m > 0)) continue;

        const ofvec3 s = bSf[b];
        const ofscalar rm = 1/m;

        txx += s.x*s.x*rm; txy += s.x*s.y*rm; txz += s.x*s.z*rm;
        tyy += s.y*s.y*rm; tyz += s.y*s.z*rm; tzz += s.z*s.z*rm;

        const ofscalar p = bphi[b]*rm;
        rx += s.x*p; ry += s.y*p; rz += s.z*p;
    }

    // Symmetric 3x3 inverse by cofactors.
    const ofscalar cxx = tyy*tzz - tyz*tyz;
    const ofscalar cxy = txz*tyz - txy*tzz;
    const ofscalar cxz = txy*tyz - txz*tyy;
    const ofscalar cyy = txx*tzz - txz*txz;
    const ofscalar cyz = txy*txz - txx*tyz;
    const ofscalar czz = txx*tyy - txy*txy;

    const ofscalar det = txx*cxx + txy*cxy + txz*cxz;

    // A cell whose faces do not span three directions cannot have a vector
    // reconstructed at all; zero is the only honest answer, and far better
    // than the infinity the division would otherwise produce.
    const ofscalar scale = txx + tyy + tzz;
    if (!(ofabs_(det) > (ofscalar)1e-30*scale*scale*scale))
    {
        u[c] = mkvec(0, 0, 0);
        return;
    }

    const ofscalar rd = 1/det;

    u[c] = mkvec
    (
        (cxx*rx + cxy*ry + cxz*rz)*rd,
        (cxy*rx + cyy*ry + cyz*rz)*rd,
        (cxz*rx + cyz*ry + czz*rz)*rd
    );
}


// ==========================================================================
//  S6  Turbulence production
// ==========================================================================

//- G = nu_t ( dev(2 symm(grad U)) : grad U )
//    = nu_t ( (grad U + grad U^T) : grad U  -  (2/3) tr(grad U)^2 )
//
//  SPEC-LIT S6 says to implement the second form because it avoids building
//  the deviatoric tensor; it follows from dev(A):B = A:B - tr(A)tr(B)/3 and
//  tr(2 symm(grad U)) = 2 tr(grad U). Evaluated in the same term order as
//  Tensor::g_by_nut in src/types.rs, so the host and the device agree.
extern "C" __global__ void fvProduction
(
    ofscalar* __restrict__ G,
    const ofscalar* __restrict__ nut,
    const oftensor* __restrict__ gradU,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const oftensor g = gradU[c];

    const ofscalar tr = g.xx + g.yy + g.zz;

    // twoSymm(g) = g + g^T, contracted with g
    const ofscalar dd =
          (g.xx + g.xx)*g.xx + (g.xy + g.yx)*g.xy + (g.xz + g.zx)*g.xz
        + (g.yx + g.xy)*g.yx + (g.yy + g.yy)*g.yy + (g.yz + g.zy)*g.yz
        + (g.zx + g.xz)*g.zx + (g.zy + g.yz)*g.zy + (g.zz + g.zz)*g.zz;

    G[c] = nut[c]*(dd - ((ofscalar)2/(ofscalar)3)*tr*tr);
}
