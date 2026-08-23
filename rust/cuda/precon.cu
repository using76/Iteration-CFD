// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  precon.cu - multi-colour incomplete factorisation (DIC / DILU).

  Written from:
    Saad, Iterative Methods for Sparse Linear Systems, 2nd ed. (2003),
      ch. 10 (ILU(0) / IC(0)) and S12.4 (multicolour ILU)
    ofgpu SPEC-LIT.md S21, which specifies the colouring, the per-colour
      kernel and the forward/backward sweep order.
  No GPL-licensed source was consulted.

  DEVICE CODE ONLY.

  ------------------------------------------------------------------------
  The factorisation
  ------------------------------------------------------------------------
  Write A = L + D + U with L and U the strict lower/upper triangles in some
  ordering. The no-fill diagonal-based incomplete factorisation is

      M = (Dt + L) Dt^-1 (Dt + U),      Dt chosen so diag(M) = diag(A)

  which gives, for every cell v,

      Dt_v = A_vv - sum_{u < v} A_vu A_uv / Dt_u                       (1)

  and the ORDERING "u < v" is the only thing that has to be sequential. Saad
  S12.4's answer, and SPEC-LIT S21's, is to take the ordering from a colouring
  of the matrix graph: cells of colour 0 first, then colour 1, and so on. No
  two neighbours share a colour, so within a colour every cell in (1) reads
  only Dt of STRICTLY EARLIER colours - the cells of its own colour are not
  its neighbours and never appear in the sum. One kernel per colour is
  therefore correct with no ordering inside the launch, and the factorisation
  is schedule-independent: exactly the property the sequential sweep lacks on
  a GPU, and the reason SPEC-LIT S8.3 refused to ship one without this.

  DIC is the symmetric case and DILU the asymmetric one. In the LDU storage of
  this crate A_vu A_uv is `upper[f]*lower[f]` for the face f joining u and v,
  whichever way round they are; when the matrix is symmetric upper == lower and
  that product IS upper^2, so the same arithmetic computes the Cholesky-form
  factor exactly. Requesting DIC on an asymmetric matrix is rejected on the
  host (src/solver.rs) rather than quietly computing DILU here.

  ------------------------------------------------------------------------
  What is left out, and why that is still a preconditioner
  ------------------------------------------------------------------------
  Coupled (cyclic) interface coefficients live in `boundary_coeffs`, not in
  upper/lower, and are not factorised. An incomplete factorisation is an
  approximation by construction; dropping the couples drops a few entries more.
  It changes the iteration count, never the answer - the preconditioner appears
  only as M^-1 applied to a vector, and the Krylov method's residual is always
  measured against the true A.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"


//- The reciprocal diagonal of a cell whose factorisation broke down.
//
//  (1) can produce zero or a sign flip on a matrix that is not diagonally
//  dominant. Falling back to the cell's own diagonal keeps M non-singular and
//  degrades that row to Jacobi; falling back to 1 keeps it defined even when
//  the diagonal itself is zero. Neither can produce a wrong ANSWER - only a
//  worse preconditioner - because M never enters the residual.
OFGPU_DEV ofscalar pcSafeReciprocal(ofscalar d, ofscalar fallback)
{
    if (d != 0) return (ofscalar)1/d;
    if (fallback != 0) return (ofscalar)1/fallback;
    return (ofscalar)1;
}


// ==========================================================================
//  1. Factorise one colour
//
//      Dt_v = A_vv - sum_{colour(u) < colour(v)} upper[f]*lower[f]*rD_u
//      rD_v = 1/Dt_v
//
//  `cells[start .. start+count)` are the cells of this colour; every one of
//  them is independent of every other, so there is no ordering inside the
//  launch. `rD` of the earlier colours is already final because their kernels
//  have already run on the same stream.
//
//  `symmetric != 0` uses upper[f]^2 in place of upper[f]*lower[f], which is
//  the DIC (Cholesky) form written out rather than inferred.
// ==========================================================================

extern "C" __global__ void pcFactorColour
(
    ofscalar* __restrict__ rD,
    const ofscalar* __restrict__ diag,
    const ofscalar* __restrict__ upper,
    const ofscalar* __restrict__ lower,
    const oflabel* __restrict__ colour,
    const oflabel* __restrict__ cells,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    oflabel symmetric,
    oflabel start,
    oflabel count
)
{
    const oflabel t = OFGPU_TID;
    if (t >= count) return;

    const oflabel c = cells[start + t];
    const oflabel myColour = colour[c];

    ofscalar d = diag[c];

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const oflabel nbr = (cfOwn[j] != 0) ? neighbour[f] : owner[f];

        if (colour[nbr] < myColour)
        {
            const ofscalar u = upper[f];
            const ofscalar prod = (symmetric != 0) ? u*u : u*lower[f];
            d -= prod*rD[nbr];
        }
    }

    rD[c] = pcSafeReciprocal(d, diag[c]);
}


// ==========================================================================
//  2. Forward sweep, colours in ASCENDING order
//
//      y_v = rD_v * ( y_v - sum_{colour(u) < colour(v)} A_vu y_u )
//
//  which is the forward substitution of (Dt + L) w = x with y holding x on
//  entry and w on exit. A_vu is upper[f] when v owns f and lower[f] when it
//  does not, because upper[f] = A(owner, neighbour).
// ==========================================================================

extern "C" __global__ void pcForwardColour
(
    ofscalar* __restrict__ y,
    const ofscalar* __restrict__ rD,
    const ofscalar* __restrict__ upper,
    const ofscalar* __restrict__ lower,
    const oflabel* __restrict__ colour,
    const oflabel* __restrict__ cells,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    oflabel start,
    oflabel count
)
{
    const oflabel t = OFGPU_TID;
    if (t >= count) return;

    const oflabel c = cells[start + t];
    const oflabel myColour = colour[c];

    ofscalar acc = 0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const int isOwner = (cfOwn[j] != 0);
        const oflabel nbr = isOwner ? neighbour[f] : owner[f];

        if (colour[nbr] < myColour)
        {
            const ofscalar a = isOwner ? upper[f] : lower[f];
            acc += a*y[nbr];
        }
    }

    y[c] = rD[c]*(y[c] - acc);
}


// ==========================================================================
//  3. Backward sweep, colours in DESCENDING order
//
//      y_v = y_v - rD_v * sum_{colour(u) > colour(v)} A_vu y_u
//
//  which is the back substitution of (Dt + U) y = Dt w.
// ==========================================================================

extern "C" __global__ void pcBackwardColour
(
    ofscalar* __restrict__ y,
    const ofscalar* __restrict__ rD,
    const ofscalar* __restrict__ upper,
    const ofscalar* __restrict__ lower,
    const oflabel* __restrict__ colour,
    const oflabel* __restrict__ cells,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    oflabel start,
    oflabel count
)
{
    const oflabel t = OFGPU_TID;
    if (t >= count) return;

    const oflabel c = cells[start + t];
    const oflabel myColour = colour[c];

    ofscalar acc = 0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const int isOwner = (cfOwn[j] != 0);
        const oflabel nbr = isOwner ? neighbour[f] : owner[f];

        if (colour[nbr] > myColour)
        {
            const ofscalar a = isOwner ? upper[f] : lower[f];
            acc += a*y[nbr];
        }
    }

    y[c] -= rD[c]*acc;
}
