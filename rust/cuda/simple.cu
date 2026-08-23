// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  simple.cu - the pieces of the SIMPLE outer loop that are not operators:
  the discrete divergence of a face flux, the field under-relaxation of the
  pressure, and the reference level of a pressure that has no Dirichlet
  boundary anywhere.

  Written from:
    S. V. Patankar, D. B. Spalding, Int. J. Heat Mass Transfer 15 (1972) 1787
      and S. V. Patankar, "Numerical Heat Transfer and Fluid Flow" (1980),
      ch. 6 - SIMPLE, and S6.7 for the relaxation of the pressure as a FIELD
      rather than as an equation
    C. M. Rhie, W. L. Chow, AIAA J. 21 (1983) 1525 - the flux the divergence
      below is taken of
    ofgpu SPEC-LIT.md section 5.2, and section 8.5's note that an all-Neumann
      Poisson problem has the constant in its null space
  No GPL-licensed source was consulted.
\*---------------------------------------------------------------------------*/

#include "ofgpu_device.cuh"


//- sum_f (+-phi_f) over one cell: the volume integral of div(phi).
//
//  GATHER, not scatter: one thread walks its own cell's faces through the
//  cell -> face CSR, so there are no atomics on a double and the summation
//  order is fixed. Two cells that see the same face see the same product with
//  opposite signs, which is what makes the total over a closed domain cancel.
//
//  This is both the right-hand side of the pressure equation - the equation
//  IS "make this zero" - and, applied to the corrected flux afterwards, the
//  continuity error that says whether it worked.
//
//  `accumulate != 0` adds into `out` instead of overwriting it, because the
//  pressure source already carries the explicit non-orthogonal correction by
//  the time this runs.
extern "C" __global__ void smpFaceFluxSum
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ bphi,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    oflabel accumulate,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofscalar s = 0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const ofscalar f = phi[cfFace[j]];
        s += cfOwn[j] ? f : -f;
    }

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        s += bphi[bcfFace[j]];
    }

    out[c] = accumulate ? out[c] + s : s;
}


//- Explicit under-relaxation of a FIELD: psi = psi_old + alpha (psi - psi_old).
//
//  Patankar S6.7. The pressure is relaxed this way and not by the implicit
//  matrix relaxation of S4.9, because the pressure equation has no
//  under-relaxable diagonal of its own to speak of - it is a pure Poisson
//  operator, and inflating its diagonal would change the operator rather than
//  the step length. SPEC-LIT S5.2 puts this step after the flux and velocity
//  correction, so the flux is corrected with the pressure that actually
//  satisfies continuity and only the field carried into the next momentum
//  predictor is relaxed.
extern "C" __global__ void smpRelaxField
(
    ofscalar* __restrict__ psi,
    const ofscalar* __restrict__ psiOld,
    ofscalar alpha,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofscalar o = psiOld[i];
    psi[i] = o + alpha*(psi[i] - o);
}


//- Copy one element to a device scalar.
//
//  Separate from `smpSubScalar` on purpose. Subtracting psi[ref] in place with
//  every thread reading psi[ref] itself is a race: the thread that owns `ref`
//  writes it while the others are still reading. Reading it out first costs
//  one extra launch and removes the race entirely.
extern "C" __global__ void smpPickValue
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ x,
    oflabel idx
)
{
    if (OFGPU_TID != 0) return;
    out[0] = x[idx];
}


//- x -= scale * (*v), with v a device scalar.
//
//  Two jobs, both to do with the constant that an all-Neumann Poisson problem
//  leaves undetermined (SPEC-LIT S8.5).
//
//  1. `scale = 1`, `v = psi[ref]` - fix the LEVEL of the solved pressure. The
//     matrix is deliberately left alone: forcing one row would stop it being
//     the separable Poisson operator the direct cuFFT backend recognises, and
//     would cost that backend for no gain. Subtracting a constant afterwards
//     changes no gradient, and therefore no flux and no velocity.
//
//  2. `scale = 1/nCells`, `v = sum(source)` - make the right-hand side
//     CONSISTENT. A singular system A p = b is solvable only when b is
//     orthogonal to the null space, here the constant vector. The physical
//     source is: the interior face fluxes cancel in pairs and a sealed domain
//     has no boundary flux. In floating point they cancel only to round-off,
//     and a Krylov solver cannot reduce the residual below the part of b that
//     lies along the null space - it stalls there and reports a residual that
//     never improves. Removing the mean removes exactly that part.
//
//     This must NOT be done when the pressure has a Dirichlet boundary
//     anywhere: the system is then non-singular, every component of b is
//     meaningful, and subtracting a constant would solve a different problem.
//     The host checks.
extern "C" __global__ void smpSubScalar
(
    ofscalar* __restrict__ x,
    const ofscalar* __restrict__ v,
    ofscalar scale,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    x[i] -= scale*v[0];
}
