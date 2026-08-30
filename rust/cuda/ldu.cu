// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  ldu.cu - operations on the lower/diagonal/upper linear system.

  Written from:
    Jasak, "Error Analysis and Estimation for the Finite Volume Method with
      Applications to Fluid Flows", PhD thesis, Imperial College (1996), ch. 3
    Patankar, "Numerical Heat Transfer and Fluid Flow" (1980), sections 4.2-4.9
    Moukalled, Mangani & Darwish, "The Finite Volume Method in Computational
      Fluid Dynamics" (2016), ch. 8
    Saad, "Iterative Methods for Sparse Linear Systems", 2nd ed. (2003), 3.4
      (compressed sparse row storage)
    ofgpu SPEC-LIT.md sections 1, 3, 4 and 5.2
  No GPL-licensed source was consulted.

  ------------------------------------------------------------------------
  Storage (SPEC-LIT section 1, marked DESIGN there)
  ------------------------------------------------------------------------

      diag  [nCells]   A(c,c)
      upper [nFaces]   A(owner[f], neighbour[f])
      lower [nFaces]   A(neighbour[f], owner[f])
      source[nCells]   right-hand side of A.psi = b

  plus, per boundary face, the pair that SPEC-LIT section 4 derives by
  differentiating the single mixed boundary expression:

      internalCoeffs[bf]   multiplies psi in the OWN cell -> folds into diag
      boundaryCoeffs[bf]   the known part                 -> folds into source

  A COUPLED face (cyclic; a face whose bNbrCell names a real cell) is the one
  exception. Its "known part" is not known: it is the value in the cell across
  the couple. Folding it into the source would freeze that value at its
  previous iterate, so it stays in the matrix and multiplies the live
  neighbour value inside Amul, exactly as an internal face's off-diagonal
  does. Its sign follows from moving the term from the right-hand side to the
  left:

      b_P       += boundaryCoeffs[bf]*psi_N      if it were explicit
      (A.psi)_P -= boundaryCoeffs[bf]*psi_N      implicit, which is what we do

  so the effective off-diagonal entry of a coupled face is -boundaryCoeffs.

  ------------------------------------------------------------------------
  Why every kernel is one thread per CELL
  ------------------------------------------------------------------------

  The natural face-based loop scatters into diag[owner[f]] and
  diag[neighbour[f]], which on a GPU needs atomicAdd on a double: slow, and
  non-deterministic in its rounding because the order of the additions varies
  run to run. Walking the cell->face CSR instead lets each thread accumulate
  its own row in a register, in the fixed ascending-face order the CSR was
  built with. No atomics, and bitwise reproducible results.

  Two kernels below (lduSetValues, lduCsrFill) do write face- or slot-indexed
  memory. Neither is a scatter: upper[f] is written only by the thread that
  owns f, lower[f] only by the thread that neighbours it, and the CSR slots
  are a permutation. No two threads ever touch the same address.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

//- |a|, spelled out rather than pulled from a header, so this translation
//  unit needs nothing but ofgpu_device.cuh in either precision.
OFGPU_DEV ofscalar lduAbs(ofscalar a) { return a < (ofscalar)0 ? -a : a; }


// ==========================================================================
//  negSumDiag - the diagonal that makes a uniform field a null vector
// ==========================================================================

//- diag[c] -= sum of column c's off-diagonal entries.
//
//  Both operators that need this need it for the same reason. Convection
//  (SPEC-LIT 3.1) writes lower[f] = -w*phi, upper[f] = (1-w)*phi and needs
//  diag[P] += w*phi = -lower[f], diag[N] -= (1-w)*phi = -upper[f]. Diffusion
//  (SPEC-LIT 3.2) writes upper[f] = lower[f] = gamma*|Sf|*Delta and needs
//  diag[P] -= lower[f], diag[N] -= upper[f]. In both cases the contribution
//  to cell c is minus the entry that sits in COLUMN c - the entry in the
//  other cell's row - which is what makes the row sum of the pure internal
//  operator vanish, and therefore makes a uniform field produce zero
//  convective divergence and zero diffusion.
//
//  For cell c owning f the column-c entry is A(N,P) = lower[f]; where c is
//  the neighbour it is A(P,N) = upper[f]. Hence the swap below, which is the
//  detail that separates a column sum from a row sum and only shows up once
//  the matrix is asymmetric.
extern "C" __global__ void lduNegSumDiag
(
    ofscalar* __restrict__ diag,
    const ofscalar* __restrict__ upper,
    const ofscalar* __restrict__ lower,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofscalar s = 0;

    const oflabel begin = cfOffset[c];
    const oflabel end   = cfOffset[c + 1];
    for (oflabel j = begin; j < end; ++j)
    {
        const oflabel f = cfFace[j];
        if (f < 0) continue;                 // dropped by a corrupt mesh
        s += cfOwn[j] ? lower[f] : upper[f];
    }

    diag[c] -= s;
}


// ==========================================================================
//  addBoundaryContributions - fold the boundary pair into diag and source
// ==========================================================================

//- diag[c] += sum internalCoeffs ;  source[c] += sum boundaryCoeffs.
//
//  The boundary half is skipped on a coupled face, whose coefficient stays in
//  the matrix for lduAmul to apply against the live neighbour value; see the
//  file header. internalCoeffs is folded on EVERY face, coupled included: it
//  multiplies this cell's own value and so belongs on the diagonal whatever
//  is on the other side.
extern "C" __global__ void lduAddBoundaryContributions
(
    ofscalar* __restrict__ diag,
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ internalCoeffs,
    const ofscalar* __restrict__ boundaryCoeffs,
    const oflabel* __restrict__ bNbrCell,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofscalar d = 0;
    ofscalar s = 0;

    const oflabel begin = bcfOffset[c];
    const oflabel end   = bcfOffset[c + 1];
    for (oflabel j = begin; j < end; ++j)
    {
        const oflabel bf = bcfFace[j];
        if (bf < 0) continue;

        d += internalCoeffs[bf];
        if (bNbrCell[bf] < 0)
        {
            s += boundaryCoeffs[bf];
        }
    }

    diag[c]   += d;
    source[c] += s;
}


// ==========================================================================
//  Amul - the sparse matrix-vector product
// ==========================================================================

//- Apsi = A.psi, including the coupled-interface term.
//
//  Row c is the diagonal, one entry per incident internal face, and one entry
//  per incident coupled boundary face:
//
//      (A.psi)_c = diag[c]*psi[c]
//                + sum_f  upper[f]*psi[nei[f]]      c owns f
//                + sum_f  lower[f]*psi[own[f]]      c neighbours f
//                - sum_bf boundaryCoeffs[bf]*psi[bNbrCell[bf]]
//
//  Call it on a matrix whose boundary pair has already been folded by
//  lduAddBoundaryContributions: the coupled term is the only boundary
//  contribution left to apply here, because it is the only one that is not
//  constant.
extern "C" __global__ void lduAmul
(
    ofscalar* __restrict__ Apsi,
    const ofscalar* __restrict__ psi,
    const ofscalar* __restrict__ diag,
    const ofscalar* __restrict__ upper,
    const ofscalar* __restrict__ lower,
    const ofscalar* __restrict__ boundaryCoeffs,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ bNbrCell,
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

    ofscalar sum = diag[c]*psi[c];

    const oflabel begin = cfOffset[c];
    const oflabel end   = cfOffset[c + 1];
    for (oflabel j = begin; j < end; ++j)
    {
        const oflabel f = cfFace[j];
        if (f < 0) continue;

        if (cfOwn[j])
        {
            sum += upper[f]*psi[neighbour[f]];
        }
        else
        {
            sum += lower[f]*psi[owner[f]];
        }
    }

    const oflabel bbegin = bcfOffset[c];
    const oflabel bend   = bcfOffset[c + 1];
    for (oflabel j = bbegin; j < bend; ++j)
    {
        const oflabel bf = bcfFace[j];
        if (bf < 0) continue;

        const oflabel nbr = bNbrCell[bf];
        if (nbr >= 0)
        {
            sum -= boundaryCoeffs[bf]*psi[nbr];
        }
    }

    Apsi[c] = sum;
}


// ==========================================================================
//  relax - implicit under-relaxation, SPEC-LIT 5.2 (Patankar 1980, 4.9)
// ==========================================================================

//- Make the diagonal dominant, divide it by alpha, put the difference in the
//  source:
//
//      diag' = max(diag, sum|off-diagonal|) / alpha
//      b'    = b + (diag' - diag)*psi_current
//
//  The fixed point is untouched: at convergence psi_current IS the solution,
//  so A'.psi = b' and A.psi = b have the same root. All that changes is how
//  far one iteration is allowed to move the answer, which is the entire
//  purpose.
//
//  Three things here are decisions rather than transcription, and are made
//  explicitly rather than left to be discovered:
//
//  1. ONE per-cell kernel, no scratch array. The unrelaxed diagonal is wanted
//     twice - to test dominance and to size the source correction - and a
//     cell has everything it needs for both, so it lives in a register for
//     the four lines it is needed and no second nCells array is allocated,
//     written, read back and freed.
//
//  2. The diagonal it tests is the FOLDED one, diag + sum internalCoeffs,
//     because that is the diagonal the solver will actually see; and the
//     off-diagonal sum includes |boundaryCoeffs| on coupled faces, because
//     those are genuine off-diagonal entries (see the file header). Both
//     halves of a coupled face are therefore counted, which is what makes
//     relax(alpha = 1) an exact no-op on an already-dominant cyclic mesh -
//     counting the coupled off-diagonal while ignoring the coupled diagonal
//     would invent a source there out of nothing.
//
//     The correction is then applied as an INCREMENT to diag and to source,
//     so that folding afterwards lands on exactly max(|D|,sumOff)/alpha.
//
//     THIS KERNEL THEREFORE RUNS BEFORE lduAddBoundaryContributions, and it
//     is the one operation here that cares. Run after the fold it would add
//     the internal coefficients to a diagonal that already contains them and
//     over-estimate the dominance test. The launcher's documentation says so
//     too; the assembly order in src/ldu_ops.rs is the definitive statement.
//
//  3. The sign of the diagonal is preserved. SPEC-LIT writes
//     max(diag, sum|off|), which is right for the positive-diagonal
//     convention Patankar assumes (a_P > 0, a_N < 0); applied to a matrix
//     assembled with the opposite overall sign, the bare max would flip the
//     row and turn relaxation into divergence. sign(diag)*max(|diag|,
//     sum|off|) is identical on the intended convention and merely safe off
//     it.
extern "C" __global__ void lduRelax
(
    ofscalar* __restrict__ diag,
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ upper,
    const ofscalar* __restrict__ lower,
    const ofscalar* __restrict__ internalCoeffs,
    const ofscalar* __restrict__ boundaryCoeffs,
    const ofscalar* __restrict__ psi,
    const oflabel* __restrict__ bNbrCell,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    ofscalar alpha,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    //- The diagonal as the solver will see it, in a register.
    ofscalar d = diag[c];
    ofscalar sumOff = 0;

    const oflabel begin = cfOffset[c];
    const oflabel end   = cfOffset[c + 1];
    for (oflabel j = begin; j < end; ++j)
    {
        const oflabel f = cfFace[j];
        if (f < 0) continue;
        sumOff += lduAbs(cfOwn[j] ? upper[f] : lower[f]);
    }

    const oflabel bbegin = bcfOffset[c];
    const oflabel bend   = bcfOffset[c + 1];
    for (oflabel j = bbegin; j < bend; ++j)
    {
        const oflabel bf = bcfFace[j];
        if (bf < 0) continue;

        d += internalCoeffs[bf];
        if (bNbrCell[bf] >= 0)
        {
            sumOff += lduAbs(boundaryCoeffs[bf]);
        }
    }

    const ofscalar magD = lduAbs(d);
    const ofscalar dominant = (magD > sumOff ? magD : sumOff)/alpha;
    const ofscalar relaxed = (d < (ofscalar)0) ? -dominant : dominant;

    //- Applied as an increment, so that folding and relaxing commute.
    const ofscalar delta = relaxed - d;

    diag[c]   += delta;
    source[c] += delta*psi[c];
}


// ==========================================================================
//  setValues - pin cells to a value
// ==========================================================================

//- Constrain psi[c] = fixedValue[c] wherever isFixed[c] != 0.
//
//  Row c becomes diag[c]*psi[c] = diag[c]*value, which is the constraint
//  itself and nothing else; every off-diagonal entry of that row is dropped.
//  Used for the wall-adjacent cells a wall function prescribes (SPEC-LIT 6.4,
//  "the relations prescribe values at the first cell rather than at the
//  face") and for pinning a pressure reference.
//
//  DESIGN - the COLUMN is eliminated too, not just the row. Every other
//  cell's coefficient against a fixed cell multiplies a known value, so it
//  moves to that cell's source and is zeroed:
//
//      source[c] -= A(c,fixed)*value
//
//  This costs nothing - the same thread is already walking the same faces -
//  and buys two things: a matrix that stays symmetric when it started
//  symmetric, which is what lets the pressure equation keep using conjugate
//  gradients, and a residual that is not polluted by a column the solver
//  cannot change. The solution is identical either way, because psi at a
//  fixed cell IS the value that was substituted.
//
//  The boundary pair on a fixed cell is zeroed as well, so folding the
//  boundary contributions afterwards adds nothing to a row that is already
//  final. That makes this kernel independent of whether it runs before or
//  after lduAddBoundaryContributions.
//
//  Race freedom: thread c writes upper[f] only for faces it owns, lower[f]
//  only for faces it neighbours, and boundary coefficients only on its own
//  boundary faces. Every address has exactly one writer.
extern "C" __global__ void lduSetValues
(
    ofscalar* __restrict__ diag,
    ofscalar* __restrict__ source,
    ofscalar* __restrict__ upper,
    ofscalar* __restrict__ lower,
    ofscalar* __restrict__ internalCoeffs,
    ofscalar* __restrict__ boundaryCoeffs,
    const oflabel* __restrict__ isFixed,
    const ofscalar* __restrict__ fixedValue,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ bNbrCell,
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

    const bool fixed = (isFixed[c] != 0);
    ofscalar s = source[c];

    const oflabel begin = cfOffset[c];
    const oflabel end   = cfOffset[c + 1];
    for (oflabel j = begin; j < end; ++j)
    {
        const oflabel f = cfFace[j];
        if (f < 0) continue;

        const bool own = (cfOwn[j] != 0);

        if (fixed)
        {
            // Drop this row's entry against its neighbour.
            if (own) upper[f] = 0; else lower[f] = 0;
        }
        else
        {
            const oflabel other = own ? neighbour[f] : owner[f];
            if (other >= 0 && isFixed[other] != 0)
            {
                // Known column: move it to the right-hand side and drop it.
                const ofscalar a = own ? upper[f] : lower[f];
                s -= a*fixedValue[other];
                if (own) upper[f] = 0; else lower[f] = 0;
            }
        }
    }

    const oflabel bbegin = bcfOffset[c];
    const oflabel bend   = bcfOffset[c + 1];
    for (oflabel j = bbegin; j < bend; ++j)
    {
        const oflabel bf = bcfFace[j];
        if (bf < 0) continue;

        if (fixed)
        {
            internalCoeffs[bf] = 0;
            boundaryCoeffs[bf] = 0;
        }
        else
        {
            const oflabel nbr = bNbrCell[bf];
            if (nbr >= 0 && isFixed[nbr] != 0)
            {
                // Amul applies this term as -boundaryCoeffs*psi_N, so moving
                // a known psi_N to the source ADDS it; see the sign
                // derivation in the file header.
                s += boundaryCoeffs[bf]*fixedValue[nbr];
                boundaryCoeffs[bf] = 0;
            }
        }
    }

    if (fixed)
    {
        // The pinned row must be invertible. An assembled diagonal never
        // vanishes - there is always a ddt, a relaxation or a boundary term -
        // but a matrix that has only had its off-diagonals written would
        // otherwise leave a zero row here, which is a singular system rather
        // than a constraint.
        ofscalar d = diag[c];
        if (d == (ofscalar)0)
        {
            d = 1;
            diag[c] = d;
        }
        source[c] = d*fixedValue[c];
    }
    else
    {
        source[c] = s;
    }
}


// ==========================================================================
//  csrFill - gather the LDU values into the prebuilt CSR
// ==========================================================================

//- val[diagSlot[c]] = diag[c], val[upperSlot[f]] = upper[f], and likewise
//  lower.
//
//  The pattern and the LDU-entry -> slot permutation were built once on the
//  host (src/ldu.rs), because neither changes for a static mesh. Refilling is
//  then just this: a permutation write, one thread per cell and per face, no
//  conflicts - which is what lets an external solver (AMGX, cuSPARSE, cuDSS)
//  be handed the matrix without the host ever seeing a coefficient.
//
//  SPEC-LIT S48.2. A COUPLED boundary face's boundaryCoeffs is an
//  off-diagonal against a cell that is not a face neighbour, and the pattern
//  now carries a column for it (`coupledSlot`, -1 on an uncoupled face). The
//  SIGN is the one thing to get right, and it follows from lduAmul, which
//  applies the coupled term as
//
//      sum -= boundaryCoeffs[bf]*psi[nbr]
//
//  so the matrix ENTRY is -boundaryCoeffs[bf] and the export negates. Before
//  S48 this write did not exist at all, and the exported matrix was therefore
//  a different operator from the one amul applies on every mesh with a cyclic
//  patch - which is why the AMGX backend refused them.
//
//  One launch covers all three loops: the grid is sized to the largest of the
//  three counts and each write is guarded, which costs two predicates and
//  saves two kernel launches.
extern "C" __global__ void lduCsrFill
(
    ofscalar* __restrict__ val,
    const ofscalar* __restrict__ diag,
    const ofscalar* __restrict__ upper,
    const ofscalar* __restrict__ lower,
    const ofscalar* __restrict__ boundaryCoeffs,
    const oflabel* __restrict__ diagSlot,
    const oflabel* __restrict__ upperSlot,
    const oflabel* __restrict__ lowerSlot,
    const oflabel* __restrict__ coupledSlot,
    oflabel nCells,
    oflabel nFaces,
    oflabel nbf
)
{
    const oflabel t = OFGPU_TID;

    if (t < nCells)
    {
        val[diagSlot[t]] = diag[t];
    }

    if (t < nFaces)
    {
        val[upperSlot[t]] = upper[t];
        val[lowerSlot[t]] = lower[t];
    }

    if (t < nbf)
    {
        const oflabel s = coupledSlot[t];
        if (s >= 0) val[s] = -boundaryCoeffs[t];
    }
}
