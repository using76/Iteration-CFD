// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  species.cu - the three operations multi-species transport needs that a
  single passive scalar does not: bounding a mass fraction into [0, 1],
  accumulating the sum of the solved fractions, and closing the inert one.

  Written from:
    ofgpu SPEC-LIT.md section 19 - the N-1 formulation, the boundedness
      requirement and the sum-to-one constraint
    Patankar, "Numerical Heat Transfer and Fluid Flow" (1980) section 4.2 -
      the source linearisation the reaction rates use, via sources.cu
  No GPL-licensed source was consulted.

  ------------------------------------------------------------------------
  Why the inert species is not solved
  ------------------------------------------------------------------------

  N independent advection-diffusion solves each satisfy their own equation
  and, in general, do NOT satisfy sum_i Y_i = 1: the discretisation error of
  each is different, the linear solver stops at a different residual, and the
  bounding clip of one has no idea what the others did. SPEC-LIT section 19
  therefore solves N-1 and defines the last one by

      Y_N = 1 - sum_{i<N} Y_i

  which makes the constraint an identity rather than a hope. The error that
  would have shown up as a sum-to-one violation shows up instead in the inert
  species, where it belongs - it is the one nobody is claiming to have solved
  for.

  ------------------------------------------------------------------------
  Why the accumulation is a separate kernel
  ------------------------------------------------------------------------

  One thread per cell, walking the species one at a time in host-issued
  launches, rather than a single kernel over a strided array of them: the
  species count is a run-time number and the fields are separate allocations,
  and passing an array of pointers to a kernel would defeat the per-buffer
  stream dependency tracking cudarc does for every argument.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"


// ==========================================================================
//  Clip a mass fraction into [0, 1] - SPEC-LIT section 19, requirement 1
//
//  *DESIGN.* A hard clip, applied after each solve, on top of a limited
//  convection scheme. The limiter is what keeps the field from going out of
//  bounds in the first place; the clip is what guarantees it, because the
//  temporal and source terms can still push a cell past the ends and a
//  negative mass fraction is not an approximation of anything.
//
//  It is deliberately NOT conservative: clipping creates or destroys a little
//  mass. Section 19 asks for boundedness and for sum-to-one, and with the
//  inert species closed by 1 - sum the clip's error lands there rather than
//  breaking the constraint.
// ==========================================================================

extern "C" __global__ void spcBound
(
    ofscalar* __restrict__ y,
    oflabel n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;

    ofscalar v = y[c];
    if (v < (ofscalar)0) v = (ofscalar)0;
    if (v > (ofscalar)1) v = (ofscalar)1;
    y[c] = v;
}


// ==========================================================================
//  sum += y   -  accumulate one solved fraction into the running total
// ==========================================================================

extern "C" __global__ void spcAccumulate
(
    ofscalar* __restrict__ sum,
    const ofscalar* __restrict__ y,
    oflabel n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;

    sum[c] += y[c];
}


// ==========================================================================
//  The inert species: Y_N = 1 - sum_{i<N} Y_i, then clipped into [0, 1].
//
//  The clip here is a diagnostic as much as a guard: if it ever fires, the
//  solved fractions have summed to more than one, and that is a statement
//  about the solution and not about round-off. `Species::inert_deficit`
//  reports how far it went so a run can say so rather than absorb it.
// ==========================================================================

extern "C" __global__ void spcCloseInert
(
    ofscalar* __restrict__ yInert,
    const ofscalar* __restrict__ sum,
    oflabel n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;

    ofscalar v = (ofscalar)1 - sum[c];
    if (v < (ofscalar)0) v = (ofscalar)0;
    if (v > (ofscalar)1) v = (ofscalar)1;
    yInert[c] = v;
}


// ==========================================================================
//  |1 - sum_i Y_i| per cell, INERT INCLUDED - the check of SPEC-LIT section
//  22: "species: sum of mass fractions -> exactly 1".
//
//  `sum` arrives holding the solved fractions' total and this adds the inert
//  one back, which is the number a reader cares about. Written to a
//  whole-mesh array so the ordinary device max-reduction can find the worst
//  cell without a second pass.
// ==========================================================================

extern "C" __global__ void spcSumError
(
    ofscalar* __restrict__ err,
    const ofscalar* __restrict__ sum,
    const ofscalar* __restrict__ yInert,
    oflabel n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;

    const ofscalar total = sum[c] + yInert[c];
    const ofscalar d = total - (ofscalar)1;
    err[c] = d < (ofscalar)0 ? -d : d;
}
