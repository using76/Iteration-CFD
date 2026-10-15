// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
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
    ofgpu SPEC-LIT.md section 120.15 - the two species boundary conditions
      as triple rewrites (spcTransferTriple; Bird, Stewart & Lightfoot,
      "Transport Phenomena", 2nd ed., section 22.1, for the film coefficient)
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
//  S86.9  What the boundedness clip did to the MASS, cell by cell
// ==========================================================================
//
//  SPEC-LIT S19 requirement 1 clips every solved fraction into [0, 1] after
//  its solve, and until S86 nothing measured what that costs. On the
//  constant-density equation there was nothing to measure it IN: the clip
//  moves `Y`, and a budget in `Y` is not a budget in kilograms. On S86's
//  mass-weighted equation the currency exists, so the clip's ledger is
//
//      acc[P] += rho_P V_P (Y_P,after - Y_P,before)
//
//  accumulated over every step of the run and reduced once at the end. It is
//  a SOURCE of species mass sitting outside the transport equation, and
//  (86.5)'s conservation identity is a statement about the equation and not
//  about the clip.
extern "C" __global__ void spcClipLedger
(
    ofscalar* __restrict__ acc,
    const ofscalar* __restrict__ y,
    const ofscalar* __restrict__ before,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ V,
    oflabel n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;

    acc[c] += rho[c]*V[c]*(y[c] - before[c]);
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


// ==========================================================================
//  The species boundary conditions as a triple rewrite - SPEC-LIT 120.15
//
//  One rule (one patch) per launch, one thread per face of it. The face's
//  outward flux of the transported quantity is the film law (120.15a/b),
//
//      n_f = c (Y_P - Y_N)        Y_N the paired baffle face's cell value
//      n_f = c (Y_P - Y_inf)      specieTransfer (nbrStart < 0)
//
//  and the section 4 triple that makes the laplacian apply exactly that is
//  (120.15c): fr = c / (Gamma_b Delta_b) on the constant-density equation,
//  refValue = Y_N (or Y_inf),
//  refGrad = 0, with Gamma_b = nu + rSigma nut_b written as turbGammaBoundary
//  writes it, so Gamma_b |Sf| snGrad_b = c |Sf| (Y_N - Y_P). On the
//  mass-weighted equation (weighted != 0) the assembly multiplies Gamma_b by
//  the face density rho_b, and the flux is rho_b c (Y_P - Y_N) at a
//  specieTransfer face; at a baffle face fr carries rho_m/rho_b as well,
//  rho_m = (rho_b + rho_b,partner)/2 the same number on both sides, so the
//  two sides' fluxes are rho_m c (Y_P - Y_N) and its negative. c = 0 gives
//  fr = 0 and refGrad = 0: zero gradient, and a flux of exactly zero.
//
//  The neighbour value is the one the paired cell holds when the rewrite
//  runs (before this species' solve), the explicit half of section 4's
//  rewrite; the cell's own value stays implicit.
// ==========================================================================

extern "C" __global__ void spcTransferTriple
(
    ofscalar* __restrict__ fr,
    ofscalar* __restrict__ refValue,
    ofscalar* __restrict__ refGrad,
    const ofscalar* __restrict__ psi,
    const oflabel* __restrict__ bFaceCells,
    const ofscalar* __restrict__ nutB,
    const ofscalar* __restrict__ bDeltaCoeffs,
    const ofscalar* __restrict__ rhoB,
    oflabel weighted,
    oflabel start,
    oflabel nbrStart,
    oflabel n,
    ofscalar c,
    ofscalar yInf,
    ofscalar nu,
    ofscalar rSigma
)
{
    const oflabel j = OFGPU_TID;
    if (j >= n) return;

    const oflabel i = start + j;
    const ofscalar gamma = nu + rSigma*nutB[i];
    ofscalar ratio = (ofscalar)1;
    if (weighted != 0 && nbrStart >= 0)
    {
        const ofscalar rhoP = rhoB[i];
        ratio = (ofscalar)0.5*(rhoP + rhoB[nbrStart + j])/rhoP;
    }
    fr[i] = c*ratio/(gamma*bDeltaCoeffs[i]);
    refValue[i] = nbrStart >= 0 ? psi[bFaceCells[nbrStart + j]] : yInf;
    refGrad[i] = (ofscalar)0;
}
