// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  momentum.cu - the pieces of the momentum equation and of the Rhie-Chow
  face flux that are not already discrete operators.

  Written from:
    C. M. Rhie, W. L. Chow, AIAA J. 21 (1983) 1525-1532 - the momentum
      interpolation that removes the collocated-grid checkerboard mode
    S. V. Patankar, D. B. Spalding, Int. J. Heat Mass Transfer 15 (1972) 1787
      and S. V. Patankar, "Numerical Heat Transfer and Fluid Flow" (1980),
      ch. 6 - SIMPLE, and the under-relaxation of ch. 4.9
    J. P. Van Doormaal, G. D. Raithby, Numer. Heat Transfer 7 (1984) 147-163 -
      SIMPLEC's consistent neighbour-correction coefficient
    F. Moukalled, L. Mangani, M. Darwish, "The Finite Volume Method in
      Computational Fluid Dynamics", Springer (2016), S15.6, and
      J. H. Ferziger, M. Peric, "Computational Methods for Fluid Dynamics",
      S7.5 - the body force treated on FACES, for the same reason the pressure
      gradient is
    E. E. Spiegel, G. Veronis, ApJ 131 (1960) 442 - why Boussinesq needs
      dT/T << 1, i.e. why it is NOT used here
    ofgpu SPEC-LIT.md sections 5 (Rhie-Chow, SIMPLE, SIMPLEC) and 9 (buoyancy)
  No GPL-licensed source was consulted.

  ---------------------------------------------------------------------------
  WHY THE BODY FORCE AND THE PRESSURE GRADIENT LIVE ON FACES

  On a collocated grid a pressure field that alternates cell to cell has zero
  central-difference gradient. The momentum equation cannot see it, so nothing
  damps it and the solution checkerboards. Rhie and Chow (1983) remove the mode
  by building the face flux from a FACE pressure difference,

      phi_f = interpolate(HbyA)_f . Sf - rAU_f |Sf| snGrad(p)_f

  in which the alternating mode has a large face difference and is therefore
  visible - and damped.

  Exactly the same argument applies to any other force that appears only
  through its gradient balance. A buoyancy force interpolated from cell values
  reintroduces the very mode Rhie-Chow removed, and a hydrostatic case then
  grows a sawtooth pressure field that still integrates to the right total and
  still looks plausible in a contour plot. So the body force is built here as a
  face quantity,

      phib_f = (Tref/T_f - 1) * (g . Sf)

  with T_f the interpolated FACE temperature, and enters phi_HbyA on faces.
  Its cell-centred counterpart is never formed at all.

  ---------------------------------------------------------------------------
  WHY A PRESCRIBED-VELOCITY FACE CARRIES NO CORRECTION

  Where the velocity is given (a wall, an inlet, a symmetry plane) the flux
  through the face is already known and the pressure equation must not change
  it: p carries zeroGradient there, so snGrad(p) = 0, and phi_b has to be
  U_b . Sf and nothing else. Adding the buoyancy flux on such a face would put
  a force through a wall - which is precisely how a sealed box at rest acquires
  a spurious circulation. `momFluxIsPrescribed` is the single predicate that
  decides this, and every kernel that touches a boundary face asks it.
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

//- True when the volumetric flux through this boundary face is fixed by the
//  velocity boundary condition rather than by the pressure equation.
//
//  * empty     - a 2-D front or back; it carries no flux by construction
//  * symmetry  - the mirror condition removes the normal component exactly
//  * fr >= 1   - a Dirichlet velocity: U_b is given, so U_b . Sf is given
//
//  A cyclic face is deliberately NOT prescribed: it is an interior face that
//  happens to be stored in the boundary arrays.
OFGPU_DEV bool momFluxIsPrescribed(oflabel kind, ofscalar fr)
{
    return kind == OFPATCH_EMPTY
        || kind == OFPATCH_SYMMETRY
        || fr >= (ofscalar)1;
}


// ==========================================================================
//  Component views of a vector field
//
//  The momentum equation is one matrix and three right-hand sides. Everything
//  the scalar operators in fv.cu need is therefore extracted component by
//  component into plain scalar arrays, and the answer is written back the same
//  way. `cmpt` is 0, 1 or 2; anything else selects z, which is what the host
//  side's `Vec3::component` does too.
// ==========================================================================

OFGPU_DEV ofscalar vecCmpt(const ofvec3& v, oflabel c)
{
    return (c == 0) ? v.x : ((c == 1) ? v.y : v.z);
}

OFGPU_DEV void setVecCmpt(ofvec3& v, oflabel c, ofscalar s)
{
    if (c == 0)      v.x = s;
    else if (c == 1) v.y = s;
    else             v.z = s;
}

extern "C" __global__ void momVecComponent
(
    ofscalar* __restrict__ out,
    const ofvec3* __restrict__ in,
    oflabel cmpt,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    out[i] = vecCmpt(in[i], cmpt);
}


extern "C" __global__ void momSetComponent
(
    ofvec3* __restrict__ out,
    const ofscalar* __restrict__ in,
    oflabel cmpt,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    ofvec3 v = out[i];
    setVecCmpt(v, cmpt, in[i]);
    out[i] = v;
}


//- bcKind is an oflabel array and field.cu's copy kernel only moves scalars.
extern "C" __global__ void momCopyLabel
(
    oflabel* __restrict__ dst,
    const oflabel* __restrict__ src,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    dst[i] = src[i];
}


// ==========================================================================
//  Small elementwise helpers
// ==========================================================================

//- out = in + c.
//
//  This BUILT nuEff = nu + nut, on cells and on boundary faces, until
//  SPEC-LIT S38.5(i) made the laminar viscosity a field and momAdd below took
//  over. It is deliberately still here, and still compiled: it is the
//  REFERENCE the regression test measures against.
//  `the_uniform_buffer_is_the_scalar_bitwise` in src/momentum.rs launches
//  both and requires the two results to be bit-identical, which is the whole
//  claim that a Newtonian case keeps every measurement it had before S38.
//  Delete it and that claim becomes an assertion again.
extern "C" __global__ void momAddConst
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ in,
    ofscalar c,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    out[i] = in[i] + c;
}


//- out = a + b, elementwise - SPEC-LIT S38.5(i).
//
//  This replaced `momAddConst(out, in, c)` when the laminar viscosity became
//  a FIELD. It is deliberately the same arithmetic in the same order:
//  `a[i] + b[i]` with every `b[i]` equal to the old scalar `c` is BITWISE
//  `in[i] + c`, because IEEE-754 addition does not care how the second
//  operand was delivered. `the_uniform_buffer_is_the_scalar_bitwise` in
//  src/momentum.rs measures exactly that, which is what lets a Newtonian
//  case keep every result it had before S38.
extern "C" __global__ void momAdd
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ a,
    const ofscalar* __restrict__ b,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    out[i] = a[i] + b[i];
}


//- out = a*b, elementwise.
extern "C" __global__ void momMul
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ a,
    const ofscalar* __restrict__ b,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    out[i] = a[i]*b[i];
}


//- The SIMPLE velocity correction:  U = HbyA + rAU*force.
//
//  `force` is the CELL force, reconstruct(b_f.Sf - |Sf| snGrad p), and rAU is
//  the CELL coefficient - and both of those words matter.
//
//  The momentum predictor solved  diag*u = su + V*force, which is
//
//      u* = HbyA + rAU*force
//
//  with exactly this rAU and exactly this force. Writing the correction the
//  same way makes the corrected velocity and the predicted one the same vector
//  once the pressure stops changing, so the SIMPLE iteration has a genuine
//  fixed point and the momentum residual falls to the linear solver's
//  tolerance rather than stalling at the difference between two spellings of
//  the same term.
//
//  Reconstructing the face-scaled flux instead - reconstruct(rAU_f*forceFlux) -
//  looks more Rhie-Chow but is NOT the predictor's own expression: rAU varies
//  from cell to cell, so the face-weighted version differs from rAU*force by
//  that variation, and the difference reappears as a floor under the momentum
//  residual that no number of iterations removes.
//
//  The face flux keeps its own rAU_f (see momCorrectFlux). That the cell
//  velocity and the face flux are not exactly each other's interpolation is
//  the whole content of Rhie-Chow, not an inconsistency in it.
extern "C" __global__ void momCorrectVelocity
(
    ofvec3* __restrict__ u,
    const ofvec3* __restrict__ hbya,
    const ofscalar* __restrict__ rau,
    const ofvec3* __restrict__ force,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;

    const ofvec3 h = hbya[i];
    const ofvec3 f = force[i];
    const ofscalar r = rau[i];

    u[i] = mkvec(h.x + r*f.x, h.y + r*f.y, h.z + r*f.z);
}


//- The magnitude of a cell vector field.
//
//  The limited convection schemes of SPEC-LIT S7 need a SCALAR to form the
//  gradient ratio r from, and the three momentum components share one matrix,
//  so they must share one set of face weights. |U| is that scalar - see the
//  note on the host side, where the choice is recorded as ours.
extern "C" __global__ void momMag
(
    ofscalar* __restrict__ out,
    const ofvec3* __restrict__ u,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofvec3 v = u[i];
    out[i] = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}


// ==========================================================================
//  A, H, rAU and HbyA  -  SPEC-LIT S5.1
//
//      A_P    = diag_P / V_P
//      H_P    = ( b_P - sum_N a_PN u_N ) / V_P
//      rAU    = 1/A_P = V_P/diag_P
//      HbyA   = rAU * H = ( b_P - sum_N a_PN u_N ) / diag_P
//
//  The neighbour sum is not walked here. The host hands over `Au = A.u`, the
//  full matrix-vector product the linear solver already knows how to form -
//  including the coupled cyclic term, which is an off-diagonal entry against a
//  cell that is not a face neighbour and which a CSR walk over faces would
//  silently drop. Then
//
//      sum_N a_PN u_N = (A.u)_P - diag_P u_P
//
//  exactly, whatever the connectivity.
// ==========================================================================

//- rAU = V/diag, guarded against a zero diagonal.
//
//  A zero diagonal means the row constrains nothing - an isolated cell, or a
//  mesh with a zero-volume cell - and 1/0 would poison every face of it. Zero
//  is returned instead, which makes that cell contribute nothing to the flux
//  rather than infinity.
extern "C" __global__ void momRau
(
    ofscalar* __restrict__ rau,
    const ofscalar* __restrict__ V,
    const ofscalar* __restrict__ diag,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofscalar d = diag[i];
    rau[i] = (d != (ofscalar)0) ? V[i]/d : (ofscalar)0;
}


//- SIMPLEC's coefficient - Van Doormaal & Raithby (1984), SPEC-LIT S5.3.
//
//  SIMPLE drops the neighbour velocity corrections entirely; SIMPLEC assumes
//  they equal the local one, u'_N = u'_P, which turns
//
//      a_P u'_P + sum_N a_N u'_N = -V grad p'
//
//  into (a_P + sum_N a_N) u'_P = -V grad p'. In Van Doormaal & Raithby's own
//  notation, where a_P u_P = sum_nb a_nb u_nb + b and a_nb = -a_N, that
//  denominator is a_P - sum_nb a_nb - which is how SPEC-LIT S5.3 writes it.
//  Here a_N is the raw matrix off-diagonal, so the sign is a plus.
//
//  `rowSum` is (A.1)_P, so sum_N a_N = rowSum - diag.
//
//  DESIGN - the floor. With no time derivative and no relaxation the
//  convection-diffusion operator satisfies a_P = -sum_N a_N exactly and the
//  denominator is zero: rAtU is unbounded and the correction diverges. Under-
//  relaxation by alpha < 1 leaves diag/alpha and rescues it, but nothing in
//  the discretisation guarantees that. We therefore floor the denominator at
//  `floorFrac` times the diagonal, which caps rAtU at rAU/floorFrac. The
//  choice of limiter is ours; the host records the value it passes.
extern "C" __global__ void momRatU
(
    ofscalar* __restrict__ rau,
    const ofscalar* __restrict__ V,
    const ofscalar* __restrict__ diag,
    const ofscalar* __restrict__ rowSum,
    ofscalar floorFrac,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;

    const ofscalar d = diag[i];
    if (d == (ofscalar)0)
    {
        rau[i] = 0;
        return;
    }

    // rowSum[i] = diag + sum_N a_N, so this IS a_P + sum_N a_N.
    ofscalar den = rowSum[i];

    // The floor is applied with the diagonal's own sign, so a matrix assembled
    // with the opposite global sign convention is not silently inverted.
    const ofscalar lo = floorFrac*d;
    if (d > (ofscalar)0) { if (den < lo) den = lo; }
    else                 { if (den > lo) den = lo; }

    rau[i] = V[i]/den;
}


//- One component of HbyA = ( b - sum_N a_N u_N ) / diag.
extern "C" __global__ void momHbyA
(
    ofvec3* __restrict__ hbya,
    const ofvec3* __restrict__ su,
    const ofscalar* __restrict__ au,
    const ofscalar* __restrict__ diag,
    const ofscalar* __restrict__ u,
    oflabel cmpt,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;

    const ofscalar d = diag[i];
    // b - sum_N a_N u_N = b - (A.u - diag u)
    const ofscalar h = vecCmpt(su[i], cmpt) - au[i] + d*u[i];

    ofvec3 v = hbya[i];
    setVecCmpt(v, cmpt, (d != (ofscalar)0) ? h/d : (ofscalar)0);
    hbya[i] = v;
}


//- source = su_cmpt + V * force_cmpt.
//
//  `su` is the momentum source WITHOUT the pressure gradient and the body
//  force; those two are the `force` the Rhie-Chow reconstruction supplies, and
//  they are kept out of `su` because H must not contain them (SPEC-LIT S5.1).
//  Adding them here, after the relaxation increment has already been written
//  by `ldu_ops::relax`, changes nothing about the relaxation: that increment
//  is (diag' - diag)*psi and does not depend on the source at all.
extern "C" __global__ void momSolveSource
(
    ofscalar* __restrict__ source,
    const ofvec3* __restrict__ su,
    const ofvec3* __restrict__ force,
    const ofscalar* __restrict__ V,
    oflabel cmpt,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    source[i] = vecCmpt(su[i], cmpt) + V[i]*vecCmpt(force[i], cmpt);
}


// ==========================================================================
//  The variable-viscosity part of the stress divergence
//
//  The full incompressible stress divergence is
//
//      div( nuEff (grad U + grad U^T) )
//        = div(nuEff grad U)  +  div(nuEff grad U^T)
//
//  The first term is fvm::laplacian and is implicit. The second is
//
//      d/dx_i ( nuEff dU_i/dx_j )
//        = (d nuEff/dx_i)(dU_i/dx_j) + nuEff d/dx_j (div U)
//
//  and the last term vanishes for a solenoidal velocity, which is what the
//  pressure equation enforces. What is left is a plain tensor-vector product:
//  with the crate's convention T_ij = dU_j/dx_i (SPEC-LIT S1), dU_i/dx_j is
//  T_ji, so
//
//      out_j = sum_i T_ji (grad nuEff)_i
//
//  i.e. `out = gradU . grad(nuEff)` read as an ordinary matrix acting on a
//  vector. It is identically zero for a uniform viscosity, which is why a
//  laminar run never notices it and a run with a strongly varying nu_t does.
//
//  SPEC-LIT is silent on this term; it is derived above rather than copied.
// ==========================================================================
extern "C" __global__ void momStressCorrection
(
    ofvec3* __restrict__ out,
    const oftensor* __restrict__ gradU,
    const ofvec3* __restrict__ gradNu,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;

    const oftensor t = gradU[i];
    const ofvec3 g = gradNu[i];

    out[i] = mkvec
    (
        t.xx*g.x + t.xy*g.y + t.xz*g.z,
        t.yx*g.x + t.yy*g.y + t.yz*g.z,
        t.zx*g.x + t.zy*g.y + t.zz*g.z
    );
}


// ==========================================================================
//  Face interpolation of a cell scalar
//
//  fv.cu's `fvInterpolateLinear` writes a whole surface field including the
//  boundary half taken from the field's evaluated faces. rAU is not a field
//  and has no boundary values, so these two do the same job from the cell
//  array alone.
// ==========================================================================

extern "C" __global__ void momFaceInterp
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ x,
    const ofscalar* __restrict__ w,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;
    const ofscalar wf = w[f];
    out[f] = wf*x[owner[f]] + (1 - wf)*x[neighbour[f]];
}


//- The boundary half. A cyclic face really does have two cells and is
//  interpolated; every other boundary face takes the adjacent cell's value,
//  which is what "zero gradient of rAU at the boundary" means.
extern "C" __global__ void momFaceInterpBoundary
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ x,
    const ofscalar* __restrict__ bw,
    const oflabel* __restrict__ bFaceCells,
    const oflabel* __restrict__ bNbrCell,
    const oflabel* __restrict__ bKind,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    const oflabel o = bFaceCells[i];

    if (bKind[i] == OFPATCH_CYCLIC)
    {
        const oflabel nb = bNbrCell[i];
        if (nb >= 0)
        {
            const ofscalar wf = bw[i];
            out[i] = wf*x[o] + (1 - wf)*x[nb];
            return;
        }
    }

    out[i] = x[o];
}


// ==========================================================================
//  Buoyancy  -  SPEC-LIT S9
//
//      rho/rho_ref = Tref/T        ideal gas at constant pressure
//      b           = g (Tref/T - 1)
//
//  NOT Boussinesq. A hot plume at 1173 K against 293 K ambient has
//  dT/T ~ 3, and the linearised beta*(T - Tref) is then wrong by a factor of
//  three (Spiegel & Veronis 1960 give the dT/T << 1 condition Boussinesq
//  needs). The density ratio above is exact for an ideal gas at constant
//  pressure and costs one divide.
//
//  What is built here is the FACE flux of that force,
//
//      phib_f = (Tref/T_f - 1) (g . Sf)
//
//  from the interpolated face temperature - never from a cell value. At
//  T_f = Tref it is exactly zero, so an isothermal case is undisturbed to the
//  last bit.
//
//  T is floored at `tMin` before the divide: a corrupted or uninitialised zero
//  would otherwise put an infinite force on one face and destroy the whole
//  pressure field. The floor is the host's, and is documented there.
// ==========================================================================

extern "C" __global__ void momBuoyancyFlux
(
    ofscalar* __restrict__ phib,
    const ofscalar* __restrict__ tf,
    const ofvec3* __restrict__ sf,
    ofscalar gx, ofscalar gy, ofscalar gz,
    ofscalar tRef,
    ofscalar tMin,
    oflabel n
)
{
    const oflabel f = OFGPU_TID;
    if (f >= n) return;

    ofscalar t = tf[f];
    if (!(t > tMin)) t = tMin;          // catches NaN as well as a low value

    const ofvec3 s = sf[f];
    phib[f] = (tRef/t - 1)*(gx*s.x + gy*s.y + gz*s.z);
}


//- The same on the boundary, with the prescribed-flux faces zeroed.
//
//  A wall does not let a body force through it any more than it lets mass
//  through it: whatever force the fluid exerts there is taken by the wall. If
//  this flux were left in, a sealed box under gravity would find a net force
//  on its boundary cells that no pressure field can balance, and would drift.
extern "C" __global__ void momBuoyancyFluxBoundary
(
    ofscalar* __restrict__ phib,
    const ofscalar* __restrict__ tf,
    const ofvec3* __restrict__ sf,
    const oflabel* __restrict__ bKind,
    const ofscalar* __restrict__ fr,
    ofscalar gx, ofscalar gy, ofscalar gz,
    ofscalar tRef,
    ofscalar tMin,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    if (momFluxIsPrescribed(bKind[i], fr[i]))
    {
        phib[i] = 0;
        return;
    }

    ofscalar t = tf[i];
    if (!(t > tMin)) t = tMin;

    const ofvec3 s = sf[i];
    phib[i] = (tRef/t - 1)*(gx*s.x + gy*s.y + gz*s.z);
}


// ==========================================================================
//  phi_HbyA  -  SPEC-LIT S5.1
//
//      phi_HbyA = interpolate(HbyA) . Sf  +  rAU_f (b_f . Sf)
//
//  The second term is `rauf*phib` and is the whole reason the body force does
//  not checkerboard.
// ==========================================================================

extern "C" __global__ void momPhiHbyA
(
    ofscalar* __restrict__ phi,
    const ofvec3* __restrict__ hbya,
    const ofscalar* __restrict__ rauf,
    const ofscalar* __restrict__ phib,
    const ofscalar* __restrict__ w,
    const ofvec3* __restrict__ sf,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    oflabel nFaces
)
{
    const oflabel f = OFGPU_TID;
    if (f >= nFaces) return;

    const ofscalar wf = w[f];
    const ofvec3 ho = hbya[owner[f]];
    const ofvec3 hn = hbya[neighbour[f]];
    const ofvec3 s = sf[f];

    const ofscalar hf = (wf*ho.x + (1 - wf)*hn.x)*s.x
                      + (wf*ho.y + (1 - wf)*hn.y)*s.y
                      + (wf*ho.z + (1 - wf)*hn.z)*s.z;

    phi[f] = hf + rauf[f]*phib[f];
}


//- phi_HbyA on the boundary.
//
//  Three cases, and the middle one is the one that keeps a sealed box sealed:
//
//    prescribed  phi_b = U_b . Sf, exactly, with no buoyancy and no HbyA. The
//                pressure equation sees zeroGradient p there and leaves it
//                alone, so this IS the final flux through the face.
//    cyclic      an interior face in disguise: interpolate across the couple.
//    otherwise   the adjacent cell's HbyA, plus the face body force. This is
//                the outlet case, where p is given and the flux is not.
extern "C" __global__ void momPhiHbyABoundary
(
    ofscalar* __restrict__ bphi,
    const ofvec3* __restrict__ hbya,
    const ofvec3* __restrict__ bu,
    const ofscalar* __restrict__ rau,
    const ofscalar* __restrict__ phib,
    const ofvec3* __restrict__ bsf,
    const ofscalar* __restrict__ bw,
    const oflabel* __restrict__ bFaceCells,
    const oflabel* __restrict__ bNbrCell,
    const oflabel* __restrict__ bKind,
    const ofscalar* __restrict__ fr,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    const oflabel kind = bKind[i];
    const ofvec3 s = bsf[i];

    if (kind == OFPATCH_EMPTY)
    {
        bphi[i] = 0;
        return;
    }

    if (momFluxIsPrescribed(kind, fr[i]))
    {
        bphi[i] = dot3(bu[i], s);
        return;
    }

    const oflabel o = bFaceCells[i];

    if (kind == OFPATCH_CYCLIC)
    {
        const oflabel nb = bNbrCell[i];
        if (nb >= 0)
        {
            const ofscalar wf = bw[i];
            const ofvec3 ho = hbya[o];
            const ofvec3 hn = hbya[nb];
            const ofscalar hf = (wf*ho.x + (1 - wf)*hn.x)*s.x
                              + (wf*ho.y + (1 - wf)*hn.y)*s.y
                              + (wf*ho.z + (1 - wf)*hn.z)*s.z;
            const ofscalar rf = wf*rau[o] + (1 - wf)*rau[nb];
            bphi[i] = hf + rf*phib[i];
            return;
        }
    }

    bphi[i] = dot3(hbya[o], s) + rau[o]*phib[i];
}


// ==========================================================================
//  The face force, and the two things built from it
// ==========================================================================

//- forceFlux = phib - |Sf| snGrad(p), on the internal faces.
//
//  Everything the momentum equation feels other than convection and diffusion,
//  expressed as a flux through a face. Reconstructed to a cell vector it is
//  the source of the next momentum predictor; multiplied by rAU_f it is the
//  correction that turns phi_HbyA into phi.
//
//  In a hydrostatic balance the two terms cancel face by face and this is
//  exactly zero - which is the discrete statement that nothing moves.
extern "C" __global__ void momForceFlux
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ phib,
    const ofscalar* __restrict__ snGradP,
    oflabel n
)
{
    const oflabel f = OFGPU_TID;
    if (f >= n) return;
    out[f] = phib[f] - snGradP[f];
}


//- The same on the boundary, zero wherever the flux is prescribed.
//
//  `phib` is already zero on those faces (momBuoyancyFluxBoundary saw to it),
//  but snGrad(p) need not be - a case that puts a fixedValue p on a wall is
//  over-specified, and this is where that mistake stops rather than where it
//  starts pushing the near-wall cell.
extern "C" __global__ void momForceFluxBoundary
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ phib,
    const ofscalar* __restrict__ snGradP,
    const oflabel* __restrict__ bKind,
    const ofscalar* __restrict__ fr,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    out[i] = momFluxIsPrescribed(bKind[i], fr[i])
           ? (ofscalar)0
           : phib[i] - snGradP[i];
}


//- phi = phi_HbyA - rAU_f |Sf| snGrad(p), on the internal faces.
extern "C" __global__ void momCorrectFlux
(
    ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ phiHbyA,
    const ofscalar* __restrict__ rauf,
    const ofscalar* __restrict__ snGradP,
    oflabel n
)
{
    const oflabel f = OFGPU_TID;
    if (f >= n) return;
    phi[f] = phiHbyA[f] - rauf[f]*snGradP[f];
}


//- The same on the boundary. A prescribed face keeps phi_HbyA untouched: that
//  value is U_b . Sf and no pressure solve may move it.
extern "C" __global__ void momCorrectFluxBoundary
(
    ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ phiHbyA,
    const ofscalar* __restrict__ rauf,
    const ofscalar* __restrict__ snGradP,
    const oflabel* __restrict__ bKind,
    const ofscalar* __restrict__ fr,
    oflabel nbf
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nbf) return;

    phi[i] = momFluxIsPrescribed(bKind[i], fr[i])
           ? phiHbyA[i]
           : phiHbyA[i] - rauf[i]*snGradP[i];
}
