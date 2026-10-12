// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  field.cu - boundary-condition evaluation and the elementwise field algebra.

  Written from:
    ofgpu SPEC-LIT.md section 4 (the single mixed boundary representation,
      marked DESIGN there - it is ours), section 2.4 (Delta_b, the
      non-orthogonal delta coefficient) and section 6.1 (bounding k and
      epsilon, also DESIGN)
    Hirsch, "Numerical Computation of Internal and External Flows", 2nd ed.
      (2007), on Robin/mixed boundary treatment
    Jasak (1996) section 3.2 for the symmetry-plane condition
    ofgpu SPEC-LIT.md section 13.4 - the flux-switched block below is a RANGE
      rather than one value because the turbulence inlets and
      pressureInletOutletVelocity all switch on the sign of the same flux and
      differ only in the value they switch to; see src/field.rs
  No GPL-licensed source was consulted.

  ------------------------------------------------------------------------
  One boundary condition, one expression
  ------------------------------------------------------------------------

  Every scalar boundary condition in this solver is a triple
  (fr, refValue, refGrad) evaluated by

      psi_b = fr*refValue + (1 - fr)*(psi_P + refGrad/Delta_b)

  which specialises to fr=1 Dirichlet, fr=0 g=0 zero-gradient, fr=0 g!=0
  Neumann, and 0<fr<1 Robin (SPEC-LIT section 4). A wall function is then a
  kernel that rewrites the triple on the faces it owns; nothing here needs to
  know which one wrote it.

  Consequently bcKind is consulted for exactly four things, and the value of
  every other kind comes out of the expression above:

    * CALCULATED - the value is written by whichever model owns the field
      (nut from a wall function, for instance) and must not be overwritten;
    * CYCLIC - the face value is an interpolation across the couple, and no
      triple can express "the cell on the other side";
    * SYMMETRY on a VECTOR - the condition is tensorial, U_b = (I - n n).U_P,
      and a scalar fr cannot express it either. For a scalar, symmetry IS
      zero-gradient and needs no branch;
    * DIRECTION_MIXED on a VECTOR (SPEC-LIT 120.3) - the tensorial mixed
      form U_b = F.r + (I - F).(U_P + g/Delta_b), eqn (120.3a) there, whose
      per-face F, r and g live in the field's dm block. What the matrix
      sees is (120.3h): fr = f_t = (tr F - n.F.n)/2, refValue = r,
      refGrad = g - the TANGENTIAL part of (120.3a) enters the matrix
      implicitly, through the one triple of section 4; this branch writes
      bf only, the face value every operator reads.

  ------------------------------------------------------------------------
  Symmetry, and what the matrix sees
  ------------------------------------------------------------------------

  A symmetry plane mirrors the flow: the mirrored velocity is
  U_P - 2 n (n.U_P), and the face value is the average of the two, hence
  U_b = U_P - n (n.U_P) - the tangential part, with the normal component
  removed. That is what this file evaluates.

  The matrix coefficients derived from the triple see fr = 0, i.e. plain
  zero-gradient, because a per-component scalar fr cannot carry the projection
  n (x) n. That is the standard segregated treatment: the normal component is
  removed explicitly, once per outer iteration, rather than implicitly. It is
  a consequence of the single-triple design of SPEC-LIT section 4 and is
  recorded here so it is a known approximation rather than a surprise.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

// --------------------------------------------------------------------------
//  BcKind, mirroring the enum in src/field.rs. `bc_kind_values_match_the_
//  device` in src/field_ops.rs pins these to the Rust side so the two cannot
//  drift apart silently.
// --------------------------------------------------------------------------
#define OFGPU_BC_CALCULATED 4
#define OFGPU_BC_SYMMETRY   6
#define OFGPU_BC_CYCLIC     7
//- SPEC-LIT 120.3: the tensorial mixed condition of the wall and slip
//  family (directionMixed, partialSlip, fixedNormalSlip, fixedShearStress).
#define OFGPU_BC_DIRECTION_MIXED 37

//- The FLUX-SWITCHED block: every kind in [FIRST, LAST] is Dirichlet where
//  the face flux is inward and zero-gradient where it is outward, and they
//  differ only in where the inflow value comes from - inletOutlet takes it
//  from the file, turbulentIntensityKineticEnergyInlet from 3/2 (I|U|)^2,
//  turbulentMixingLength* from k and a length, pressureInletOutletVelocity
//  from the flux itself. All of them switch on the same test, so it is a
//  range and not a list.
#define OFGPU_BC_INLET_OUTLET_FIRST 8
#define OFGPU_BC_INLET_OUTLET_LAST  12


//- The mixed expression of SPEC-LIT section 4, for one component.
//
//  Delta_b is the boundary delta coefficient of SPEC-LIT section 2.4 and is
//  strictly positive for any cell of finite size. The guard costs one
//  predicate and turns a degenerate face into a zero-gradient one instead of
//  poisoning the whole field with a NaN: with fr = 1 the (1-fr) factor would
//  otherwise multiply an infinity and give 0*inf.
OFGPU_DEV ofscalar fldMixed
(
    ofscalar fr,
    ofscalar refValue,
    ofscalar refGrad,
    ofscalar psiP,
    ofscalar deltaCoeff
)
{
    const ofscalar g = (deltaCoeff != (ofscalar)0) ? refGrad/deltaCoeff : (ofscalar)0;
    return fr*refValue + ((ofscalar)1 - fr)*(psiP + g);
}


// ==========================================================================
//  correctBoundaryConditions
// ==========================================================================

//- Evaluate every boundary face of a scalar field.
extern "C" __global__ void fldCorrectBcScalar
(
    ofscalar* __restrict__ bf,
    const ofscalar* __restrict__ psi,
    const ofscalar* __restrict__ fr,
    const ofscalar* __restrict__ refValue,
    const ofscalar* __restrict__ refGrad,
    const oflabel* __restrict__ bcKind,
    const oflabel* __restrict__ bFaceCells,
    const ofscalar* __restrict__ bDeltaCoeffs,
    const oflabel* __restrict__ bNbrCell,
    const ofscalar* __restrict__ bWeights,
    oflabel nBoundaryFaces
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nBoundaryFaces) return;

    const oflabel kind = bcKind[i];

    //- Written by a model, not by this kernel.
    if (kind == OFGPU_BC_CALCULATED) return;

    const oflabel c = bFaceCells[i];
    const ofscalar psiP = psi[c];

    if (kind == OFGPU_BC_CYCLIC)
    {
        const oflabel nbr = bNbrCell[i];
        if (nbr >= 0)
        {
            //- The face sits between two cells exactly as an internal face
            //  does, so it is interpolated exactly as one: SPEC-LIT 2.3, with
            //  bWeights the weight of THIS side.
            const ofscalar w = bWeights[i];
            bf[i] = w*psiP + ((ofscalar)1 - w)*psi[nbr];
        }
        else
        {
            //- Marked cyclic with nothing on the other side. Degrading to
            //  zero-gradient keeps the field finite; the mesh check is what
            //  reports the pairing failure.
            bf[i] = psiP;
        }
        return;
    }

    bf[i] = fldMixed(fr[i], refValue[i], refGrad[i], psiP, bDeltaCoeffs[i]);
}


//- Evaluate every boundary face of a vector field.
//
//  Componentwise the same expression, with the two tensorial conditions -
//  symmetry, and DirectionMixed (SPEC-LIT 120.3) - handled separately; see
//  the file header. DirectionMixed's face value is (120.3a) and this kernel
//  writes it into bf ONLY: the matrix sees (120.3h) - fr = f_t, refValue =
//  r, refGrad = g, the tangential part implicitly - and refValue is read
//  here as an input, never rewritten.
extern "C" __global__ void fldCorrectBcVector
(
    ofvec3* __restrict__ bf,
    const ofvec3* __restrict__ psi,
    const ofscalar* __restrict__ fr,
    const ofvec3* __restrict__ refValue,
    const ofvec3* __restrict__ refGrad,
    const ofscalar* __restrict__ dm,
    const oflabel* __restrict__ bcKind,
    const oflabel* __restrict__ bFaceCells,
    const ofscalar* __restrict__ bDeltaCoeffs,
    const ofvec3* __restrict__ bSf,
    const oflabel* __restrict__ bNbrCell,
    const ofscalar* __restrict__ bWeights,
    oflabel nBoundaryFaces
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nBoundaryFaces) return;

    const oflabel kind = bcKind[i];

    if (kind == OFGPU_BC_CALCULATED) return;

    const oflabel c = bFaceCells[i];
    const ofvec3 uP = psi[c];

    if (kind == OFGPU_BC_DIRECTION_MIXED)
    {
        //- SPEC-LIT 120.3, eqn (120.3a):
        //
        //    U_b = F.r + (I - F).(U_P + g/Delta_b)
        //
        //  in the operation order 120.3 fixes: q = U_P + g/Delta_b (the term
        //  0 where Delta_b = 0, the guard of SPEC-LIT section 4), a = F.r,
        //  b = F.q, U_b = a + (q - b), each matrix-vector product summed
        //  F_x0 v_0 + F_x1 v_1 + F_x2 v_2 left to right. The host twin is
        //  field::direction_mixed_value in src/field.rs, the same order; FMA
        //  contraction is on, so the two agree to round-off, not bitwise.
        const ofscalar dc = bDeltaCoeffs[i];
        const ofscalar* d = dm + 12*i;   // Fxx Fxy Fxz Fyy Fyz Fzz rx ry rz gx gy gz

        const ofscalar gdx = (dc != (ofscalar)0) ? d[ 9]/dc : (ofscalar)0;
        const ofscalar gdy = (dc != (ofscalar)0) ? d[10]/dc : (ofscalar)0;
        const ofscalar gdz = (dc != (ofscalar)0) ? d[11]/dc : (ofscalar)0;
        const ofvec3 q = mkvec(uP.x + gdx, uP.y + gdy, uP.z + gdz);

        const ofvec3 a = mkvec
        (
            d[0]*d[6] + d[1]*d[7] + d[2]*d[8],
            d[1]*d[6] + d[3]*d[7] + d[4]*d[8],
            d[2]*d[6] + d[4]*d[7] + d[5]*d[8]
        );
        const ofvec3 b = mkvec
        (
            d[0]*q.x + d[1]*q.y + d[2]*q.z,
            d[1]*q.x + d[3]*q.y + d[4]*q.z,
            d[2]*q.x + d[4]*q.y + d[5]*q.z
        );

        const ofvec3 ub = mkvec(a.x + (q.x - b.x), a.y + (q.y - b.y), a.z + (q.z - b.z));
        bf[i] = ub;
        return;
    }

    if (kind == OFGPU_BC_SYMMETRY)
    {
        //- U_b = U_P - n (n.U_P): the mirror plane leaves the tangential
        //  component alone and cancels the normal one.
        const ofvec3 sf = bSf[i];
        const ofscalar magSqr = dot3(sf, sf);

        if (magSqr > (ofscalar)0)
        {
            const ofscalar un = dot3(sf, uP)/magSqr;    // (n.U)/|Sf|
            bf[i] = mkvec
            (
                uP.x - sf.x*un,
                uP.y - sf.y*un,
                uP.z - sf.z*un
            );
        }
        else
        {
            bf[i] = uP;                                  // degenerate face
        }
        return;
    }

    if (kind == OFGPU_BC_CYCLIC)
    {
        const oflabel nbr = bNbrCell[i];
        if (nbr >= 0)
        {
            const ofscalar w = bWeights[i];
            const ofscalar w1 = (ofscalar)1 - w;
            const ofvec3 uN = psi[nbr];
            bf[i] = mkvec
            (
                w*uP.x + w1*uN.x,
                w*uP.y + w1*uN.y,
                w*uP.z + w1*uN.z
            );
        }
        else
        {
            bf[i] = uP;
        }
        return;
    }

    const ofscalar f = fr[i];
    const ofscalar dc = bDeltaCoeffs[i];
    const ofvec3 rv = refValue[i];
    const ofvec3 rg = refGrad[i];

    bf[i] = mkvec
    (
        fldMixed(f, rv.x, rg.x, uP.x, dc),
        fldMixed(f, rv.y, rg.y, uP.y, dc),
        fldMixed(f, rv.z, rg.z, uP.z, dc)
    );
}


//- Regenerate the value fraction of an inletOutlet face from the flux.
//
//  Its definition (src/field.rs) is "fr = 1 where the face flux is inward,
//  else 0": an inflow carries information from outside the domain and must be
//  told what it is bringing, an outflow carries it out and must not be told
//  anything. Sf points OUT of the domain, so an inward flux is a negative
//  phi. Faces of any other kind are left alone, which is what lets this run
//  over the whole boundary once per outer iteration.
//
//  Every kind in the flux-switched range shares this switch; only the
//  refValue they switch TO differs, and that is set once at setup by
//  src/field_setup.rs.
extern "C" __global__ void fldInletOutletFraction
(
    ofscalar* __restrict__ fr,
    const oflabel* __restrict__ bcKind,
    const ofscalar* __restrict__ phiB,
    oflabel nBoundaryFaces
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nBoundaryFaces) return;

    const oflabel kind = bcKind[i];

    if (kind >= OFGPU_BC_INLET_OUTLET_FIRST && kind <= OFGPU_BC_INLET_OUTLET_LAST)
    {
        fr[i] = (phiB[i] < (ofscalar)0) ? (ofscalar)1 : (ofscalar)0;
    }
}


// ==========================================================================
//  Elementwise algebra
//
//  Deliberately one kernel per operation rather than a fused expression
//  evaluator: these are memory-bound, so the only thing that matters is that
//  each touches its arrays exactly once.
// ==========================================================================

extern "C" __global__ void fldCopy
(
    ofscalar* __restrict__ dst,
    const ofscalar* __restrict__ src,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    dst[i] = src[i];
}


extern "C" __global__ void fldCopyVector
(
    ofvec3* __restrict__ dst,
    const ofvec3* __restrict__ src,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    dst[i] = src[i];
}


extern "C" __global__ void fldSet
(
    ofscalar* __restrict__ dst,
    ofscalar value,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    dst[i] = value;
}


extern "C" __global__ void fldSetVector
(
    ofvec3* __restrict__ dst,
    ofscalar x,
    ofscalar y,
    ofscalar z,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    dst[i] = mkvec(x, y, z);
}


//- x = max(x, lo). SPEC-LIT 6.1 requires k and epsilon to stay positive; the
//  choice of clip is ours and is documented there as DESIGN.
extern "C" __global__ void fldBound
(
    ofscalar* __restrict__ x,
    ofscalar lo,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofscalar v = x[i];
    x[i] = (v < lo) ? lo : v;
}


//- x = min(max(x, lo), hi). The two-sided form, for the limiters that need
//  a ceiling as well as a floor (SPEC-LIT 6.1 bounds nu_t from above).
extern "C" __global__ void fldClamp
(
    ofscalar* __restrict__ x,
    ofscalar lo,
    ofscalar hi,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    ofscalar v = x[i];
    v = (v < lo) ? lo : v;
    v = (v > hi) ? hi : v;
    x[i] = v;
}


extern "C" __global__ void fldMultiply
(
    ofscalar* __restrict__ dst,
    const ofscalar* __restrict__ src,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    dst[i] *= src[i];
}


//- dst += src. The partner of fldMultiply, and the second half of SPEC-LIT
//  S59.1's blend `dst = dst*mask + other`: on the side the mask keeps, the
//  pair is `x*1 + 0`, which is x in every bit; on the side it drops, it is
//  `x*0 + y`, which is y in every bit. That exactness is the whole reason the
//  blend is written as two elementwise kernels rather than as one fused one
//  with a branch - a branch would be a second code path, and S59.5 has to
//  prove there is only one.
extern "C" __global__ void fldAdd
(
    ofscalar* __restrict__ dst,
    const ofscalar* __restrict__ src,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    dst[i] += src[i];
}


//- dst /= src. No epsilon in the denominator: a zero divisor here means the
//  caller handed over a field it should have bounded first, and hiding that
//  behind a regularisation would hide the bug with it.
//- out += in, elementwise, on a vector field.
//
//  The vector twin of `fldAdd`, and the accumulate SPEC-LIT S18's whole-field
//  registries are built out of: `MomentumSources` sums the contributions of
//  everything that pushes a body force into one array before the assembly
//  reads it once.
extern "C" __global__ void fldAddVector
(
    ofvec3* __restrict__ out,
    const ofvec3* __restrict__ in,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofvec3 a = out[i];
    const ofvec3 b = in[i];
    out[i] = mkvec(a.x + b.x, a.y + b.y, a.z + b.z);
}

extern "C" __global__ void fldDivide
(
    ofscalar* __restrict__ dst,
    const ofscalar* __restrict__ src,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    dst[i] /= src[i];
}


extern "C" __global__ void fldScale
(
    ofscalar* __restrict__ dst,
    ofscalar s,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    dst[i] *= s;
}
