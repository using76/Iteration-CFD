// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  sources.cu - volumetric source terms over a cell set.

  Written from:
    Patankar, "Numerical Heat Transfer and Fluid Flow" (1980) section 4.2 -
      the S = S_u + S_p psi linearisation and the requirement S_p <= 0
    Ward, J. Hydraul. Div. ASCE 90 (1964) 1-12 - the Darcy-Forchheimer
      resistance of a porous medium
    Kays & Crawford, "Convective Heat and Mass Transfer", 3rd ed. (1993)
      ch. 9, and Patankar, Liu & Sparrow, ASME J. Heat Transfer 99 (1977)
      180-186 - the periodic-fully-developed decomposition whose
      compensating source is proportional to the LOCAL streamwise mass flux,
      which is what srcThermostatMassFluxWeight forms (SPEC-LIT 35.3)
    ofgpu SPEC-LIT.md sections 3.4, 18 and 35.3. The geometric cell-set
      selection of section 18 is marked *DESIGN* there and is ours; it lives
      on the host, in src/sources.rs, and reaches these kernels as a list of
      cell indices. The discrete weighting of section 35.3.3 is ours too.
  No GPL-licensed source was consulted.

  ------------------------------------------------------------------------
  Why a cell LIST and not a mask
  ------------------------------------------------------------------------

  A source occupies a heater, a fan or a filter: a few thousand cells out of a
  few million. Launching one thread per cell in the mesh and having almost
  all of them return would cost the whole mesh in bandwidth to touch nothing.
  One thread per SELECTED cell costs the zone.

  Every thread reads and writes exactly one cell, and the list holds each
  cell at most once - src/sources.rs sorts and deduplicates it - so there is
  no accumulation, no atomic, and the result is bitwise reproducible whatever
  order the blocks retire in.

  ------------------------------------------------------------------------
  The sign convention, which is the whole of section 3.4
  ------------------------------------------------------------------------

  For A psi = b with A's diagonal positive:

      explicit source S_u :  source[P] += V_P S_u
      implicit sink   S_p :  diag[P]   += V_P |S_p|      (S_p <= 0)
      unknown sign    S   :  diag[P]   += V_P max(S, 0)
                             source[P] -= V_P min(S, 0) psi_P

  The mixed form puts whichever half stabilises the matrix on the diagonal
  and evaluates the rest at the current psi. The two branches agree exactly
  at psi = psi_P, so it is a stability choice and not an approximation.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

#ifdef OFGPU_SINGLE
OFGPU_DEV ofscalar ofsqrt_(ofscalar a) { return sqrtf(a); }
#else
OFGPU_DEV ofscalar ofsqrt_(ofscalar a) { return sqrt(a); }
#endif


// ==========================================================================
//  Explicit source, one constant over the zone: source[P] += V_P S_u
// ==========================================================================

extern "C" __global__ void srcExplicitConst
(
    ofscalar* __restrict__ source,
    const oflabel* __restrict__ cells,
    const ofscalar* __restrict__ v,
    ofscalar su,
    oflabel nSel
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nSel) return;

    const oflabel c = cells[i];
    source[c] += v[c]*su;
}


// ==========================================================================
//  Explicit source from a field: source[P] += V_P su[P]
//
//  The field is indexed by CELL, not by position in the list, so a caller can
//  hand over a whole-mesh array and let the zone pick out of it.
// ==========================================================================

extern "C" __global__ void srcExplicitField
(
    ofscalar* __restrict__ source,
    const oflabel* __restrict__ cells,
    const ofscalar* __restrict__ v,
    const ofscalar* __restrict__ su,
    ofscalar scale,
    oflabel nSel
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nSel) return;

    const oflabel c = cells[i];
    source[c] += scale*v[c]*su[c];
}


// ==========================================================================
//  Implicit sink of known sign: diag[P] += V_P spMag,  spMag >= 0
//
//  The caller has already taken the magnitude, which is what makes this the
//  stabilising branch by construction (Patankar section 4.2). A source that
//  might be positive must go through srcMixedConst instead.
// ==========================================================================

extern "C" __global__ void srcImplicitConst
(
    ofscalar* __restrict__ diag,
    const oflabel* __restrict__ cells,
    const ofscalar* __restrict__ v,
    ofscalar spMag,
    oflabel nSel
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nSel) return;

    const oflabel c = cells[i];
    diag[c] += v[c]*spMag;
}


// ==========================================================================
//  Mixed sign - Patankar's linearisation, SPEC-LIT section 3.4
// ==========================================================================

extern "C" __global__ void srcMixedConst
(
    ofscalar* __restrict__ diag,
    ofscalar* __restrict__ source,
    const oflabel* __restrict__ cells,
    const ofscalar* __restrict__ v,
    const ofscalar* __restrict__ psi,
    ofscalar s,
    oflabel nSel
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nSel) return;

    const oflabel c = cells[i];
    const ofscalar vc = v[c];

    if (s >= (ofscalar)0)
    {
        diag[c] += vc*s;
    }
    else
    {
        source[c] -= vc*s*psi[c];
    }
}


// ==========================================================================
//  Darcy-Forchheimer porous drag - Ward (1964), SPEC-LIT section 18
//
//      S_p = -(mu/K + rho C_F |U|/2)          per unit volume
//
//  divided through by rho, because this solver's momentum equation is the
//  kinematic one and carries p/rho:
//
//      S_p = -(d + f |U|/2),   d = nu/K [1/s],  f = C_F [1/m]
//
//  The implicit part is negative BY CONSTRUCTION - both d and f are positive
//  properties of the medium and |U| >= 0 - which is exactly what makes a
//  porous zone unconditionally stable however large the resistance is. It
//  goes on the diagonal and never on the right-hand side; there is no branch
//  here because there is no sign to test.
//
//  |U| is the velocity the last momentum solve produced, which lags the
//  Forchheimer term by one outer iteration. That is the same segregated lag
//  the convection coefficients carry and it vanishes on convergence.
// ==========================================================================

extern "C" __global__ void srcDarcyForchheimer
(
    ofscalar* __restrict__ diag,
    const oflabel* __restrict__ cells,
    const ofscalar* __restrict__ v,
    const ofvec3*  __restrict__ u,
    ofscalar d,
    ofscalar f,
    oflabel nSel
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nSel) return;

    const oflabel c = cells[i];
    const ofvec3 uc = u[c];
    const ofscalar umag = ofsqrt_(dot3(uc, uc));

    diag[c] += v[c]*(d + (ofscalar)0.5*f*umag);
}


// ==========================================================================
//  Flag a zone for lduSetValues - SPEC-LIT section 18's fixed-value
//  constraint, which is section 3's setValues.
//
//  Writes the flag and the value; the elimination itself is lduSetValues in
//  ldu.cu, which owns the row and column bookkeeping.
// ==========================================================================

extern "C" __global__ void srcFlagFixed
(
    oflabel* __restrict__ isFixed,
    ofscalar* __restrict__ fixedValue,
    const oflabel* __restrict__ cells,
    ofscalar value,
    oflabel nSel
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nSel) return;

    const oflabel c = cells[i];
    isFixed[c] = 1;
    fixedValue[c] = value;
}


// ==========================================================================
//  The zone's total volume-weighted source, per cell, for the accounting
//  check of SPEC-LIT section 22: "a heat source of known power raises the
//  domain enthalpy by exactly that much".
//
//  Writes V_P S_u into a whole-mesh array that the caller has zeroed, so the
//  ordinary device reduction can sum it. Nothing in the time loop calls this;
//  it exists so a test can measure what was actually injected rather than
//  what was asked for.
// ==========================================================================

extern "C" __global__ void srcZoneWeight
(
    ofscalar* __restrict__ out,
    const oflabel* __restrict__ cells,
    const ofscalar* __restrict__ v,
    ofscalar su,
    oflabel nSel
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nSel) return;

    const oflabel c = cells[i];
    out[c] = v[c]*su;
}


// ==========================================================================
//  SPEC-LIT section 35.3: the mass-flux weight of the thermostat.
//
//      w[c]    = (rho u)_c . e_hat            kg/(m2 s)
//      wAbs[c] = |w[c]|
//
//  Both are written in one pass because the host needs BOTH reductions -
//  the signed sum W = sum_c w_c V_c is the normalisation, and the gross sum
//  W_abs = sum_c |w_c| V_c is what section 35.3.4's degenerate guard
//  compares it against. One launch, one read of rho and U, two writes.
//
//  Whole-mesh, not zone-indexed: a thermostat always selects `all`
//  (section 35.1), so there is no cell list to gather through and one thread
//  per cell is exactly one thread per selected cell.
// ==========================================================================

extern "C" __global__ void srcThermostatMassFluxWeight
(
    ofscalar* __restrict__ w,
    ofscalar* __restrict__ wAbs,
    const ofscalar* __restrict__ rho,
    const ofvec3* __restrict__ u,
    ofscalar ex,
    ofscalar ey,
    ofscalar ez,
    oflabel n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;

    const ofvec3 uc = u[c];
    const ofscalar wc = rho[c]*(uc.x*ex + uc.y*ey + uc.z*ez);
    w[c] = wc;
    wAbs[c] = wc < (ofscalar)0 ? -wc : wc;
}
