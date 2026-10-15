// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

/*---------------------------------------------------------------------------*\
  energy.cu - the three fused elementwise kernels the low-Mach energy
  equation needs that the generic field algebra of cuda/field.cu does not
  provide (it has copy/set/bound/clamp/multiply/divide/scale, but no
  add-two-buffers and no add-a-constant, and materialising either as an extra
  pass over every cell would cost more than writing the one formula wanted).

  Written from:
    R. G. Rehm, H. R. Baum, J. Res. Natl. Bur. Stand. 83 (1978) 297-308 -
      the divergence constraint, SPEC-LIT S25.1, kernel energyTargetDivergence
    ofgpu SPEC-LIT.md S26 - k_eff = k + rho*cp*nu_t/Pr_t, kernel energyKEff
    ofgpu SPEC-LIT.md S18 - the source registry an explicit contribution is
      summed into, kernel energyAccumulate
    M. S. Day, J. B. Bell, Combust. Theory Model. 4 (2000) 535-556, eqs.
      (1)-(6) - the species' enthalpy flux in the temperature equation,
      SPEC-LIT 133.9, kernels energyMixtureCpAdd, energyEnthalpyDiffusionAdd
  No GPL-licensed source was consulted.

  Every other piece of the energy equation - the ideal-gas density, the
  rho*cp-weighted convection flux, ddt itself - reuses cuda/field.cu's
  multiply/divide/scale and cuda/fv.cu / cuda/timescheme.cu's existing
  operators directly from the Rust side; nothing here duplicates them.
\*---------------------------------------------------------------------------*/
#include "ofgpu_device.cuh"

//- dst[i] += src[i]. The one generic accumulate cuda/field.cu is missing -
//  used by energy::EnergySources to sum whatever a volumetric heat model
//  registered this iteration, and nowhere else needs it enough to justify
//  adding it to the shared file.
extern "C" __global__ void energyAccumulate
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

//- k_eff on a face (SPEC-LIT S26): k_eff = k + rho_f*cp*nu_t_f/Pr_t.
//
//  Takes the already-interpolated FACE rho and FACE nu_t (SPEC-LIT S25.3:
//  "rho_f by linear interpolation of cell rho" - the same convention used for
//  every other face diffusivity in this crate) and the constant molecular
//  conductivity `kMol` and `cpOverPrt = cp/Pr_t`, so the call site building
//  the pressure-equation-scale (W/m/K) diffusivity needs no scratch buffer of
//  its own beyond the two face fields it already had to interpolate.
//
//  Used unchanged for both the internal-face pass (n = nInternalFaces) and
//  the boundary-face pass (n = nBoundaryFaces), exactly like
//  turbGammaBoundary reuses turbGammaInternal's formula in cuda/turbulence.cu.
extern "C" __global__ void energyKEff
(
    ofscalar* __restrict__ dst,
    const ofscalar* __restrict__ rhoF,
    const ofscalar* __restrict__ nutF,
    ofscalar kMol,
    ofscalar cpOverPrt,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    dst[i] = kMol + rhoF[i]*nutF[i]*cpOverPrt;
}

//- Kays-Crawford's variable turbulent Prandtl number, SPEC-LIT S37.
//
//  Written from:
//    W. M. Kays, "Turbulent Prandtl number - where are we?", ASME J. Heat
//      Transfer 116 (1994) 284-295, and Kays & Crawford, Convective Heat and
//      Mass Transfer, 4th ed., ch. 13 - the correlation itself.
//  No GPL-licensed source was consulted.
//
//  This is the DEVICE TWIN of src/energy.rs's `kays_crawford_prt`, evaluated
//  in the rearranged form S37.2 derives (the same function, and the one that
//  survives floating point):
//
//      Pe_t = (nu_t/nu) Pr
//      u    = 1/(C Pe_t sqrt(Pr_t_inf))
//      h(u) = (exp(-u) + u - 1)/u^2          -> 1/2 as u -> 0, 0 as u -> inf
//      Pr_t = Pr_t_inf/(1/2 + h(u))
//
//  with the same two end branches, for the same two reasons: at Pe_t = 0
//  (a resolved wall face under a low-Re model, where nu_t is pinned to zero)
//  `u` is +inf and the direct form evaluates inf/inf = NaN; at small `u`
//  (large Pe_t) `exp(-u) + u - 1` is a difference of numbers near 1 whose
//  true value is u^2/2, so it is summed as its Taylor series instead.
#ifdef OFGPU_SINGLE
OFGPU_DEV ofscalar keffExp_(ofscalar a)  { return expf(a); }
OFGPU_DEV ofscalar keffSqrt_(ofscalar a) { return sqrtf(a); }
#else
OFGPU_DEV ofscalar keffExp_(ofscalar a)  { return exp(a); }
OFGPU_DEV ofscalar keffSqrt_(ofscalar a) { return sqrt(a); }
#endif

OFGPU_DEV ofscalar kaysCrawfordPrt
(
    ofscalar peT,
    ofscalar c,
    ofscalar prtInf,
    ofscalar eps
)
{
    const ofscalar a = keffSqrt_(prtInf);
    const ofscalar x = c*peT;

    // The Pe_t -> 0 branch, written as the NOT of the positive test so a NaN
    // takes it too rather than propagating.
    if (!((ofscalar)2*x*a > eps)) return (ofscalar)2*prtInf;

    const ofscalar u = (ofscalar)1/(x*a);
    ofscalar h;
    if (u < (ofscalar)1e-2)
    {
        // h(u) = sum_{k>=0} (-u)^k/(k+2)!
        h = (ofscalar)0.5 - u/(ofscalar)6 + u*u/(ofscalar)24
          - u*u*u/(ofscalar)120 + u*u*u*u/(ofscalar)720;
    }
    else
    {
        h = (keffExp_(-u) + u - (ofscalar)1)/(u*u);
    }
    return prtInf/((ofscalar)0.5 + h);
}

//- k_eff on a face with a LOCAL Pr_t (SPEC-LIT S37.3):
//
//      k_eff = kMol + rho_f cp nu_t_f / Pr_t(Pe_t),   Pe_t = (nu_t_f/nu) Pr
//
//  Same two passes as energyKEff above (internal faces, then boundary faces),
//  same inputs plus the three numbers the correlation needs - `nu` to form
//  Pe_t, `pr`, and `prtInf` which is the case's own `Prt` read as the
//  free-stream asymptote. `cp` arrives whole rather than as `cp/Pr_t`,
//  because there is no longer one `Pr_t` to fold it into.
extern "C" __global__ void energyKEffKaysCrawford
(
    ofscalar* __restrict__ dst,
    const ofscalar* __restrict__ rhoF,
    const ofscalar* __restrict__ nutF,
    ofscalar kMol,
    ofscalar cp,
    ofscalar nu,
    ofscalar pr,
    ofscalar prtInf,
    ofscalar c,
    ofscalar eps,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofscalar nut = nutF[i];
    const ofscalar prt = kaysCrawfordPrt(nut*pr/nu, c, prtInf, eps);
    dst[i] = kMol + rhoF[i]*nut*cp/prt;
}

//- The target divergence of SPEC-LIT S25.1:
//
//      (div u)_target = Q/(rho*cp*T) - (1/(gamma*p0))*dp0dt
//
//  `q` is the volumetric heat-release rate registered on the energy equation
//  (S18), W/m3; `invGammaP0 = 1/(gamma*p0)` and `dp0dt` are the two halves of
//  the uniform second term, folded into one multiply here because p0 and its
//  derivative are single numbers the whole domain shares - there is nothing
//  per-cell about them, so there is nothing to interpolate.
//
//  `q` is the S18 registry - a heater's q''', a parcel cloud's convective
//  exchange, the S35 thermostat - and `qCond` is the CONDUCTION half of
//  S25.1's own `Q`,
//  div(k_eff grad T), which Energy::update_conduction_source forms off the
//  same face flux fvm_laplacian assembles. The two are separate arguments
//  rather than one summed field because they are accumulated by different
//  owners and because a reader of this kernel should be able to see that
//  BOTH halves of S25.1's `Q` are here: leaving `qCond` out prescribes the
//  wrong dilatation, and SPEC-LIT S26.1 measures what that cost.
extern "C" __global__ void energyTargetDivergence
(
    ofscalar* __restrict__ dst,
    const ofscalar* __restrict__ q,
    const ofscalar* __restrict__ qCond,
    const ofscalar* __restrict__ rho,
    const ofscalar* __restrict__ t,
    ofscalar cp,
    ofscalar invGammaP0,
    ofscalar dp0dt,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    dst[i] = (q[i] + qCond[i])/(rho[i]*cp*t[i]) - invGammaP0*dp0dt;
}

//- SPEC-LIT S32.2's fixed wall heat flux: rewrite the fixedGradient-shaped
//  ({fr = 0}) Robin triple on every `fixedFluxTemperature` face so that
//  k_eff_wall*refGrad reproduces `q` EXACTLY, whatever k_eff_wall is - see
//  src/field.rs's `BcKind::FixedFluxTemperature` doc for why one condition
//  serves both a wall-function mesh (k_eff_wall carries the momentum wall
//  function's eddy diffusivity, refreshed here every outer iteration as it
//  evolves) and a resolved/lowRe mesh (k_eff_wall is the constant molecular
//  k there, so this reproduces a plain steady fixedGradient exactly). `q` is
//  read from refValue, never written - crate::field_setup seeded it from the
//  field file's own `q` entry, and this kernel leaves it there so it survives
//  being read again next iteration, exactly like ThermalWallFunction's T_w.
//
//  One thread per owned face - no wall-adjacent-CELL constraint, same shape
//  as wfThermalWall in cuda/wallfunctions.cu (T is not pinned, only its
//  triple is rewritten, so there is no CSR and no scatter).
extern "C" __global__ void energyFixedFluxTemperature
(
    ofscalar* __restrict__ fr,
    ofscalar* __restrict__ refGrad,
    const ofscalar* __restrict__ refValue,
    const ofscalar* __restrict__ kEffWall,
    const oflabel* __restrict__ face,
    oflabel nFaces
)
{
    const oflabel i = OFGPU_TID;
    if (i >= nFaces) return;

    const oflabel bf = face[i];
    const ofscalar keff = kEffWall[bf];
    if (!(keff > (ofscalar)0)) return;

    fr[bf] = (ofscalar)0;
    refGrad[bf] = refValue[bf]/keff;
}

//- SPEC-LIT 120.8's thermal wall family: rewrite the Robin triple on every
//  face `field_setup::thermal_walls` read from T's OWN patch types, so that
//  C_b (T_P - T_b) = H (T_b - T_ref) - q_in holds AT EVERY ITERATE, with
//  C_b = k_eff,b*Delta_b the face's CURRENT conductance - the S98 ambient
//  triple carried from the conjugate path onto any fluid wall. The host twin
//  of this exact operation order is cht::ambient::ThermalFace::triple;
//  tests/bc_thermal.rs holds the two within tol(1e-14) relative (exact for
//  a zero), not bit for bit.
//
//  Four names ride it, distinguished by their per-face parameters:
//    externalWallHeatFluxTemperature  - flux/power (h = 0, qIn given) or
//                                       coefficient (h, Ta, eps, layers)
//    turbulentHeatFluxTemperature     - rides 32.2's kind instead
//    lumpedMassTemperature            - h and tInf rewritten per STEP by
//                                       Energy::advance_lumped_masses
//    totalFlowRateAdvectiveDiffusive  - advective[s] = 1: H from (120.8h),
//                                       off phiConvB = cp*rho_f*phi_b
//
//  A guard that fails (kEff or C_b not positive, a radiating face's T_o*
//  not positive, a Danckwerts face with no area) returns WITHOUT writing -
//  the convention of energyFixedFluxTemperature above: the face keeps
//  whatever triple it had. Every input is widened to ofacc, the arithmetic
//  of (120.8b)-(120.8f)/(120.8h) runs in f64 in the order written, and each
//  of the three outputs is rounded ONCE back to ofscalar (SPEC-LIT 118.2).
//  The five parameter arrays are f64 in BOTH builds - Energy uploads them
//  widened once, so no per-iteration cast can drift.
//
//  One thread per listed face - no wall-adjacent-CELL constraint, same
//  shape as energyFixedFluxTemperature (T is not pinned, only its triple is
//  rewritten). No GPL-licensed source was consulted.
extern "C" __global__ void energyThermalWall
(
    ofscalar* __restrict__ fr,
    ofscalar* __restrict__ refValue,
    ofscalar* __restrict__ refGrad,
    const ofscalar* __restrict__ tb,
    const ofscalar* __restrict__ psi,
    const oflabel* __restrict__ bFaceCells,
    const ofscalar* __restrict__ kEffWall,
    const ofscalar* __restrict__ bDeltaCoeffs,
    const ofscalar* __restrict__ phiConvB,
    const ofscalar* __restrict__ bMagSf,
    const oflabel* __restrict__ face,
    const ofacc* __restrict__ h,
    const ofacc* __restrict__ tInf,
    const ofacc* __restrict__ eps,
    const ofacc* __restrict__ rL,
    const ofacc* __restrict__ qIn,
    const oflabel* __restrict__ advective,
    ofacc sigma,
    oflabel nFaces
)
{
    const oflabel s = OFGPU_TID;
    if (s >= nFaces) return;

    const oflabel bf = face[s];
    const ofacc keff = (ofacc)kEffWall[bf];
    if (!(keff > (ofacc)0)) return;
    const ofacc c = keff * (ofacc)bDeltaCoeffs[bf];
    if (!(c > (ofacc)0)) return;

    const ofacc tbv = (ofacc)tb[bf];
    const ofacc tp = (ofacc)psi[bFaceCells[bf]];

    ofacc hh, tRef;
    if (advective[s]) {
        // (120.8h): phiConv is positive OUT, so an inflow face carries
        // H = -phiConv/|Sf| and an outflow face is zero-gradient (hh = 0
        // gives fr = 0 below). refGrad stays qIn/kEff = 0.
        const ofacc msf = (ofacc)bMagSf[bf];
        if (!(msf > (ofacc)0)) return;
        const ofacc pc = (ofacc)phiConvB[bf];
        hh = pc < (ofacc)0 ? -pc/msf : (ofacc)0;
        tRef = tInf[s];
    } else {
        // (120.8b): what the last solve conducted from the cell to the
        // face; (120.8c): the outer temperature that iterate implies.
        const ofacc qc = c*(tp - tbv);
        const ofacc to = tbv - rL[s]*qc;
        if (eps[s] > (ofacc)0) {
            // (120.8d): (S98.2)'s tangent about the OUTER surface.
            if (!(to > (ofacc)0)) return;
            const ofacc es = eps[s]*sigma;
            const ofacc b = (ofacc)4*es*to*to*to;
            const ofacc ta4 = tInf[s]*tInf[s]*tInf[s]*tInf[s];
            const ofacc a = es*(to*to*to*to - ta4) - b*to;
            const ofacc ho = h[s] + b;
            tRef = (h[s]*tInf[s] - a)/ho;
            hh = ho;
        } else {
            // Taken, not computed - (S98.1) in every bit.
            hh = h[s];
            tRef = tInf[s];
        }
        // (120.8e): the layers in series with the outer film.
        hh = hh/((ofacc)1 + rL[s]*hh);
    }

    fr[bf] = (ofscalar)(hh/(hh + c));
    refValue[bf] = (ofscalar)tRef;
    refGrad[bf] = (ofscalar)(qIn[s]/keff);
}

// --------------------------------------------------------------------------
//  SPEC-LIT 133.9 (CHR-10b): the species' enthalpies in the temperature
//  equation. Each species is one row of 12 ofacc in the table
//  chem::source::SpeciesEnthalpy uploads: R/W (J/(kg K)), t_mid, the lower
//  set a1..a5, the upper set a1..a5 (128.1)/(128.5). Every sum is formed in
//  ofacc (double in both builds, SPEC-LIT 118.1) and rounded once by
//  energyRoundAcc.
// --------------------------------------------------------------------------

//- cp_m(T) per unit mass, (128.1) times R/W_m, the lower set at or below
//  t_mid (128.5), in thermo.rs nasa7_cp_r's operation order -
OFGPU_DEV ofacc energySpeciesCp(const ofacc* row, ofacc t)
{
    const ofacc* a = (t <= row[1]) ? row + 2 : row + 7;
    return row[0] * (a[0] + t * (a[1] + t * (a[2] + t * (a[3] + t * a[4]))));
}

//- (133.18), one member of the set: acc[i] += Y_m[i] cp_m(T[i]). A member
//  with Y_m == 0 adds exactly nothing, as ThermoSet::cp_mass skips it -
extern "C" __global__ void energyMixtureCpAdd
(
    ofacc* __restrict__ acc,
    const ofscalar* __restrict__ t,
    const ofscalar* __restrict__ y,
    const ofacc* __restrict__ row,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    const ofacc yv = (ofacc)y[i];
    if (yv == (ofacc)0) return;
    acc[i] += yv * energySpeciesCp(row, (ofacc)t[i]);
}

//- (133.17), one solved species k against the inert N:
//  acc[c] += rho_c D_k (cp_k(T_c) - cp_N(T_c)) (grad Y_k)_c . (grad T)_c -
extern "C" __global__ void energyEnthalpyDiffusionAdd
(
    ofacc* __restrict__ acc,
    const ofscalar* __restrict__ t,
    const ofscalar* __restrict__ rho,
    const ofvec3* __restrict__ gradY,
    const ofvec3* __restrict__ gradT,
    const ofacc* __restrict__ rowK,
    const ofacc* __restrict__ rowN,
    const ofacc dK,
    oflabel n
)
{
    const oflabel c = OFGPU_TID;
    if (c >= n) return;
    const ofacc tc = (ofacc)t[c];
    const ofvec3 gy = gradY[c];
    const ofvec3 gt = gradT[c];
    const ofacc dot = (ofacc)gy.x * (ofacc)gt.x + (ofacc)gy.y * (ofacc)gt.y
        + (ofacc)gy.z * (ofacc)gt.z;
    const ofacc dcp = energySpeciesCp(rowK, tc) - energySpeciesCp(rowN, tc);
    acc[c] += (ofacc)rho[c] * dK * dcp * dot;
}

//- out[i] = acc[i], the one rounding to the build's scalar -
extern "C" __global__ void energyRoundAcc
(
    ofscalar* __restrict__ out,
    const ofacc* __restrict__ acc,
    oflabel n
)
{
    const oflabel i = OFGPU_TID;
    if (i >= n) return;
    out[i] = (ofscalar)acc[i];
}
