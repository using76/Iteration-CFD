// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  timescheme.cu - the time schemes of SPEC-LIT S13.

  Written from:
    Crank & Nicolson, Proc. Camb. Phil. Soc. 43 (1947) 50-67       (S13.1)
    Ferziger & Peric, Computational Methods for Fluid Dynamics, S6.3
                                                            (BDF2, theta)
    Patankar, Numerical Heat Transfer and Fluid Flow (1980) S4.2  (Euler)
    ofgpu SPEC-LIT.md S13.1, S13.2, S13.3 - S13.2's smoothing ratio,
      sweep count and damping are marked *DESIGN* there and are ours.
  No GPL-licensed source was consulted.

  DEVICE CODE ONLY.

  Everything here gathers over the cell->face CSR: one thread per cell walking
  its own faces, so there are no atomics, the summation order is fixed, and the
  result is bitwise reproducible.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"


// ==========================================================================
//  S3.3 / S13.3  The general three-level implicit time derivative
//
//  Every scheme in this file that is not the theta method writes
//
//      d(psi)/dt ~= aN*psi^n + a0*psi^{n-1} + a00*psi^{n-2}
//
//  which covers Euler (aN = 1/dt, a0 = -1/dt, a00 = 0) and BDF2 at both
//  constant and variable dt (SPEC-LIT S13.3). Keeping one kernel for both
//  means an adaptive run cannot silently fall back to the constant-dt
//  coefficients: the host computes the three numbers and the device does not
//  know which scheme produced them.
//
//      diag[P]   += sign*V_P*aN
//      source[P] -= sign*V_P*(a0*psi0_P + a00*psi00_P)
// ==========================================================================

extern "C" __global__ void tsDdtGeneral
(
    ofscalar* __restrict__ diag,
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ V,
    const ofscalar* __restrict__ psi0,
    const ofscalar* __restrict__ psi00,
    ofscalar aN,
    ofscalar a0,
    ofscalar a00,
    ofscalar sign,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar sv = sign*V[c];
    diag[c]   += sv*aN;
    source[c] -= sv*(a0*psi0[c] + a00*psi00[c]);
}


//- The same, for d(rho psi)/dt. Each level carries its own density, which is
//  what makes the discrete form conserve rho*psi rather than psi.
extern "C" __global__ void tsDdtGeneralRho
(
    ofscalar* __restrict__ diag,
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ V,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ rho0,
    const ofscalar* __restrict__ rho00,
    const ofscalar* __restrict__ psi0,
    const ofscalar* __restrict__ psi00,
    ofscalar aN,
    ofscalar a0,
    ofscalar a00,
    ofscalar sign,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar sv = sign*V[c];
    diag[c]   += sv*aN*rho[c];
    source[c] -= sv*(a0*rho0[c]*psi0[c] + a00*rho00[c]*psi00[c]);
}


//- SPEC-LIT S86.4: the ddt half of the DISCRETE continuity residual,
//
//      contDdt[P] = aN*rho_P + a0*rho0_P + a00*rho00_P
//
//  which is exactly what tsDdtGeneralRho above contributes to row P when the
//  transported field is 1 everywhere, divided by V_P. The bounded correction
//  of a mass-weighted equation subtracts psi_P times this (through fvSp) as
//  well as psi_P*sum_f(+-phi_m,f), because on a variable-density equation the
//  continuity residual has two halves and S3.1's correction only ever saw
//  one. Written as its own kernel rather than three field_ops passes so that
//  no scratch cell field has to exist to hold an intermediate.
extern "C" __global__ void tsDdtRhoContinuity
(
    ofscalar* __restrict__ contDdt,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ rho0,
    const ofscalar* __restrict__ rho00,
    ofscalar aN,
    ofscalar a0,
    ofscalar a00,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    contDdt[c] = aN*rho[c] + a0*rho0[c] + a00*rho00[c];
}


// ==========================================================================
//  S13.2  Local time stepping - Euler with a PER-CELL reciprocal step
// ==========================================================================

//- Euler implicit where rDeltaT is a field rather than a scalar.
//
//      diag[P]   += sign*V_P*rDeltaT_P
//      source[P] += sign*V_P*rDeltaT_P*psi0_P
//
//  Nothing about this is physical: it is a preconditioner wearing a time
//  derivative's clothes (SPEC-LIT S13.2), and the converged steady answer must
//  not depend on rDeltaT at all.
extern "C" __global__ void tsDdtLocal
(
    ofscalar* __restrict__ diag,
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ V,
    const ofscalar* __restrict__ rDeltaT,
    const ofscalar* __restrict__ psi0,
    ofscalar sign,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar a = sign*V[c]*rDeltaT[c];
    diag[c]   += a;
    source[c] += a*psi0[c];
}


// ==========================================================================
//  S13.1  The theta method
//
//  With M the assembled spatial matrix and b its source, the semi-discrete
//  system is  V dpsi/dt = b - M.psi = L(psi), and the theta scheme
//
//      V (psi^n - psi^{n-1})/dt = theta L(psi^n) + (1 - theta) L(psi^{n-1})
//
//  rearranges to
//
//      M' = theta*M
//      b' = b - (1 - theta)*M.psi^{n-1}
//
//  after which the Euler ddt (V/dt on the diagonal, V psi^{n-1}/dt on the
//  source) is added exactly as usual. theta = 1 leaves M and b untouched, so a
//  Crank-Nicolson code path that is asked for Euler really is Euler.
//
//  `apsi0` is M.psi^{n-1} formed BEFORE the scaling, with the same matrix -
//  SPEC-LIT S13.1 marks re-applying the current operator (rather than keeping
//  the old-time matrix) as the *DESIGN* choice, because a second matrix would
//  double the largest allocation in the solver.
// ==========================================================================

extern "C" __global__ void tsThetaCells
(
    ofscalar* __restrict__ diag,
    ofscalar* __restrict__ source,
    const ofscalar* __restrict__ apsi0,
    ofscalar theta,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    diag[c]   *= theta;
    source[c] -= ((ofscalar)1 - theta)*apsi0[c];
}


//- Scale a face or boundary coefficient array in place. The off-diagonals and
//  the coupled-interface coefficients are part of M and must be scaled with
//  it; forgetting one of them leaves an operator that is theta-weighted in
//  some directions and not in others, which is second order in nothing.
extern "C" __global__ void tsScale
(
    ofscalar* __restrict__ x,
    ofscalar factor,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    x[i] *= factor;
}


// ==========================================================================
//  S13.2  The local time step field
//
//      rDeltaT_P = max( 1/dt_max , (1/2) sum_f |phi_f| / (Co_max V_P) )
//
//  The 1/2 is there because sum_f |phi_f| counts every unit of volume flux
//  twice - once entering the cell and once leaving it.
// ==========================================================================

extern "C" __global__ void tsLtsRDeltaT
(
    ofscalar* __restrict__ rDeltaT,
    const ofscalar* __restrict__ phi,
    const ofscalar* __restrict__ bphi,
    const ofscalar* __restrict__ V,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    const oflabel* __restrict__ bcfOffset,
    const oflabel* __restrict__ bcfFace,
    ofscalar coMax,
    ofscalar rDeltaTMax,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofscalar sumPhi = 0;

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        // cfOwn is read but not used: |phi| is orientation-independent. It is
        // still walked through the same CSR so the gather order matches every
        // other operator in the crate.
        (void)cfOwn[j];
        sumPhi += fabs(phi[cfFace[j]]);
    }

    for (oflabel j = bcfOffset[c]; j < bcfOffset[c + 1]; ++j)
    {
        sumPhi += fabs(bphi[bcfFace[j]]);
    }

    const ofscalar vc = V[c];
    const ofscalar local =
        (vc > 0 && coMax > 0) ? (ofscalar)0.5*sumPhi/(coMax*vc) : (ofscalar)0;

    rDeltaT[c] = ofmax_(rDeltaTMax, local);
}


//- One smoothing sweep, as a GATHER.
//
//  SPEC-LIT S13.2: the raw field varies too abruptly between neighbours to be
//  stable, so the largest value is propagated outward until no cell exceeds
//  its neighbour by more than `ratio`. Written as
//
//      out_P = max( in_P , max_N in_N / ratio )
//
//  which raises the small values rather than lowering the large ones - the
//  safe direction, since a smaller local time step is always stable.
//
//  Separate in/out buffers, deliberately: an in-place sweep would make the
//  answer depend on which blocks happened to run first, and this whole file
//  exists to keep that from happening.
extern "C" __global__ void tsLtsSmooth
(
    ofscalar* __restrict__ out,
    const ofscalar* __restrict__ in,
    const oflabel* __restrict__ owner,
    const oflabel* __restrict__ neighbour,
    const oflabel* __restrict__ cfOffset,
    const oflabel* __restrict__ cfFace,
    const oflabel* __restrict__ cfOwn,
    ofscalar ratio,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    ofscalar best = in[c];

    for (oflabel j = cfOffset[c]; j < cfOffset[c + 1]; ++j)
    {
        const oflabel f = cfFace[j];
        const oflabel nbr = (cfOwn[j] != 0) ? neighbour[f] : owner[f];
        best = ofmax_(best, in[nbr]/ratio);
    }

    out[c] = best;
}


//- Optional damping between outer iterations (SPEC-LIT S13.2, *DESIGN*).
//
//      rDeltaT = rDeltaT_old + damping*(rDeltaT_new - rDeltaT_old)
//
//  damping = 1 is no damping at all, which is the default; smaller values stop
//  the local step from collapsing the moment a transient sweeps through.
extern "C" __global__ void tsLtsDamp
(
    ofscalar* __restrict__ rDeltaT,
    const ofscalar* __restrict__ rDeltaTOld,
    ofscalar damping,
    oflabel nCells
)
{
    const oflabel c = OFGPU_TID;
    if (c >= nCells) return;

    const ofscalar o = rDeltaTOld[c];
    rDeltaT[c] = o + damping*(rDeltaT[c] - o);
}
