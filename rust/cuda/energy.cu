// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
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
