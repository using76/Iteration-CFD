// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The one-equation `gamma` transition model of Menter, Smirnov, Liu &
//! Avancha (2015) on the SST background - SPEC-LIT §90.
//!
//! Written from:
//!   NASA / TMBWG, *Turbulence Modeling Resource - SST-2003-Menter-Gamma-2015*,
//!     <https://tmbwg.github.io/turbmodels/menter_gamma_3eqn.html> - US
//!     government-authored DOCUMENTATION, not source, **fetched and read
//!     2026-09-05**. The page is marked "under construction" and carries the
//!     equations and the constants in full; every digit below is the page's,
//!     and where it under-determines the model §90.3 and §90.4 record the
//!     reading chosen here rather than one borrowed from a source that was
//!     not read.
//!   Menter, F. R., Smirnov, P. E., Liu, T. & Avancha, R., "A One-Equation
//!     Local Correlation-Based Transition Model", *Flow Turbul. Combust.* 95
//!     (2015) 583-619 - the model's primary reference, **paywalled and NOT
//!     read**; cited as such and never quoted.
//!   Menter, Kuntz & Langtry (2003) - already the source of §6.3; the
//!     SST-2003 background this model couples to, and the paper the page
//!     itself names as that baseline's reference.
//!   ofgpu `SPEC-LIT.md` §90, and §6.3 for the background model.
//! No GPL-licensed source was consulted. OpenFOAM's and SU2's transition
//! implementations were not opened, searched or quoted.
//!
//! # The host half of the model
//!
//! This module holds what the CPU can hold of §90: the closed forms
//! (90.4)-(90.17) as free functions, [`GammaCoeffs`], and the `gamma`
//! equation's own `system/` settings. It is the shape of
//! [`crate::models::transition`] with one equation gone: §90.1's whole point
//! is that `Re_thetac` is correlated in quantities that ARE local - `Tu_L`
//! from `k`, `omega` and the wall distance (90.9), `lambda_thL` from a
//! wall-normal strain (90.10) - so the second transported field LM2009 pays
//! for disappears, and with it §88.4's sweeps and §88.8's floors.
//!
//! No function below reads a velocity magnitude, which is the whole reason
//! §90.7 gets to quote the page's own line - **"This model is Galilean
//! invariant"** - and Gate 90-G holds it bitwise rather than merely small,
//! where §88.9 measured LM2009's `Re_theta_eq` moving +82.4 % under a frame
//! shift of +5 m/s.
//!
//! Nothing here is launched by a case that names no transition model: the
//! device twins live in `cuda/gmtrans.cu`, the model struct
//! [`MenterGamma`] below launches them, and the SST hook that would attach
//! one is not wired yet - the whole of this file is dead code to a run
//! until `k_omega_sst.rs` grows the `Option<MenterGamma>` slot.
//!
//! The page prints `C_PG2` twice in (90.7)'s negative branch and `C_PG3` in
//! no equation at all; §90.3 records the reading implemented here and the
//! pair test that proves the choice live in both directions.

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::GpuScalarField;
use crate::field_ops::{
    advance_time_levels, correct_boundary_conditions, copy_field, copy_field_vector, FieldKernels,
};
use crate::fv::{fvm_sp, fvm_su, fvm_susp};
use crate::io::case::SolverControls;
use crate::io::schemes::DivEntry;
use crate::mesh::GpuMesh;
use crate::solver::SolverPerformance;
use crate::turbulence::{vorticity_mag, FlowState, RasCore, TurbKernels};
use crate::{Scalar, Tensor, Vec3};

use cudarc::driver::{CudaFunction, PushKernelArg};

// ==========================================================================
//  90.4  The local inputs, as host functions
//
//  Every one of these is the CPU twin of a device function in
//  `cuda/gmtrans.cu`, written in the same order with the same constants.
//  None of them reads a velocity
//  magnitude - the property Gate 90-G holds bitwise.
// ==========================================================================

/// `Tu_L = min(100 sqrt(2k/3)/(omega d_w), 100)` (90.9) - the local
/// turbulence intensity the correlation reads, a percentage throughout, and
/// the `min(..., 100)` cap the page's own.
///
/// `k` is floored at zero and `omega` and `d_w` at the tiny positive floor
/// `lmFields` guards every divisor with, so a degenerate cell reads the cap
/// rather than an infinity.
#[must_use]
pub fn tu_l(k: Scalar, omega: Scalar, d: Scalar) -> Scalar {
    let num = 100.0 * (2.0 * k.max(0.0) / 3.0).sqrt();
    let den = omega.max(1e-30) * d.max(1e-30);
    (num / den).min(100.0)
}

/// (90.10) before (90.8)'s limiter: `-7.57e-3 (dV/dy) d_w^2/nu + 0.0128`.
///
/// Exposed because the reference digits pin the RAW value at states whose
/// clipped value is the same `1`, and because the 0.0128 offset is the
/// correlation's own clean-air constant, visible only before the clip.
#[must_use]
pub fn lambda_theta_l_raw(dv_dy: Scalar, d: Scalar, nu: Scalar) -> Scalar {
    -7.57e-3 * dv_dy * (d * d) / nu.max(1e-30) + 0.0128
}

/// `lambda_thL` (90.10) with the published limiter `-1 <= lambda_thL <= 1`
/// (90.8) applied.
#[must_use]
pub fn lambda_theta_l(dv_dy: Scalar, d: Scalar, nu: Scalar) -> Scalar {
    lambda_theta_l_raw(dv_dy, d, nu).clamp(-1.0, 1.0)
}

/// The wall normal `n` of (90.11): `grad(y)/|grad y|`, **normalised per
/// cell** (90.11, the reading §90.4 records).
///
/// `walldistance`'s `grad_y` is a direction and not a unit vector - §57.6
/// measured `|grad y|` leaving 1 by 0.495 over one block - so feeding it to
/// (90.11) unnormalised would put that error into every `lambda_thL` a
/// mesh's interior carries. Where there is no wall in the mesh at all,
/// `|grad y|` is zero and so is `n`: (90.11) is then zero and `lambda_thL`
/// is exactly the clean-air `0.0128` rather than a NaN (§90.4).
#[must_use]
pub fn wall_normal(grad_y: Vec3) -> Vec3 {
    grad_y.normalised()
}

/// `dV/dy ~= n_i g_ij n_j` with `g_ij = dU_j/dx_i` (90.11) - the layout
/// [`crate::types::Tensor`] holds and `RasCore::grad_u` carries, the one
/// `lmFields` reads `gradU` with for `dU/ds`.
///
/// Freezing `n` under the derivative - dropping the `grad(n)` term of the
/// page's continuous `grad(n . V) . n` - is §90.4's DESIGN choice, recorded
/// there and labelled here.
#[must_use]
pub fn dv_dy(n: Vec3, g: Tensor) -> Scalar {
    n.x * (g.xx * n.x + g.xy * n.y + g.xz * n.z)
        + n.y * (g.yx * n.x + g.yy * n.y + g.yz * n.z)
        + n.z * (g.zx * n.x + g.zy * n.y + g.zz * n.z)
}

// ==========================================================================
//  90.3  Re_thetac and F_PG
// ==========================================================================

/// `F_PG(lambda_thL)` (90.7), then (90.8)'s `F_PG = max(F_PG, 0)`.
///
/// The two branches with [`GammaCoeffs::c_pg3`] on the
/// `min[lambda_thL + 0.0681, 0]` term - §90.3's reading of the constant the
/// page prints in no equation, which at the printed `C_PG3 = 0.00` makes
/// the term vanish. The `[-1, 1]` clip on the argument is (90.8)'s too and
/// belongs to [`lambda_theta_l`]; this function caps its own result at the
/// printed `max(F_PG, 0)`.
#[must_use]
pub fn f_pg(lambda: Scalar, c: &GammaCoeffs) -> Scalar {
    let f = if lambda >= 0.0 {
        (1.0 + c.c_pg1 * lambda).min(c.c_pg1_lim)
    } else {
        (1.0 + c.c_pg2 * lambda + c.c_pg3 * (lambda + 0.0681).min(0.0))
            .min(c.c_pg2_lim)
    };
    f.max(0.0)
}

/// `Re_thetac(Tu_L, lambda_thL)` (90.6) - the critical momentum-thickness
/// Reynolds number, correlated in local quantities alone (§90.1).
///
/// At `lambda_thL = 0`, `F_PG = 1` exactly and this is the two-constant form
/// `C_TU1 + C_TU2 exp(-C_TU3 Tu_L)`; §90.3 records both facts and why a
/// pressure-gradient cap below 1 would silently deform the first of them.
#[must_use]
pub fn re_thetac(tu_l: Scalar, lambda: Scalar, c: &GammaCoeffs) -> Scalar {
    c.c_tu1 + c.c_tu2 * (-(c.c_tu3 * tu_l * f_pg(lambda, c))).exp()
}

/// `F_onset = max(F_onset2 - F_onset3, 0)` (90.4) - the page's four-step
/// switch with `2.2`, `2.0` and `3.5`, which are not §88.3's `2.193` and
/// `2.5` (§90.2 names every place the calibration moved).
///
/// `F_onset2` is (90.4)'s plain `min(F_onset1, 2.0)`: §88.3's fourth power
/// is one of the things the 2015 model dropped. `re_thetac` carries the same
/// tiny floor `lmFields` guards its divisor with; (90.6) bounds it below by
/// `C_TU1` wherever (90.8) holds, so the floor never bites in a live state
/// (§90.8).
#[must_use]
pub fn f_onset(re_v: Scalar, re_thetac: Scalar, r_t: Scalar) -> Scalar {
    let fo1 = re_v / (2.2 * re_thetac.max(1e-30));
    let fo2 = fo1.min(2.0);
    let fo3 = (1.0 - (r_t / 3.5).powi(3)).max(0.0);
    (fo2 - fo3).max(0.0)
}

/// `F_turb = exp(-(R_T/2)^4)` (90.5) - the fully-turbulent gate closing at
/// HALF §88.8's `R_T`: `F_turb = e^-1` at `R_T = 2` here, at `R_T = 4`
/// there (§90.2). The argument is clamped the way `lmFields` clamps every
/// exponential argument (`OFLM_ARG_CLAMP`).
#[must_use]
pub fn f_turb(r_t: Scalar) -> Scalar {
    (-(r_t / 2.0).min(1e6).powi(4)).exp()
}

/// `F_on^lim = min[max(Re_V/(2.2 Re_thetac_lim) - 1, 0), 3]` (90.16) - the
/// onset switch `P_k^lim` reads, with the coupling's OWN limiter constant
/// and not a `Re_thetac` anywhere in it (§90.6). Clipped at 3, pinned by
/// test.
///
/// The `max`/`min` chain is kept rather than `clamp` deliberately, lint
/// notwithstanding: it is (90.16)'s printed shape, and on a garbage cell
/// that reaches here as NaN it reads 0.0 - the switch OFF, the safe
/// degenerate answer - where `clamp` would hand NaN to `P_k^lim`.
#[must_use]
#[allow(clippy::manual_clamp)]
pub fn f_on_lim(re_v: Scalar, re_thetac_lim: Scalar) -> Scalar {
    ((re_v / (2.2 * re_thetac_lim.max(1e-30)) - 1.0).max(0.0)).min(3.0)
}

/// `P_k^lim` (90.15) - the page's additional production, whose own note says
/// it exists "to help with the proper generation of `k` at the transition
/// location for arbitrary low values of `Tu`" (§90.6).
///
/// Every factor is non-negative at a live state: the two `max`s by
/// construction, `(1 - gamma)` on `gamma`'s own bounds (§90.8), and
/// `F_on^lim` by (90.16). `nu_t` enters with the sign the page prints -
/// through `max(3 C_sep nu - nu_t, 0)` - so the term switches itself off in
/// the turbulent log layer rather than being clamped there.
#[must_use]
pub fn p_k_lim(
    gamma: Scalar,
    f_on_lim: Scalar,
    nu: Scalar,
    nu_t: Scalar,
    s: Scalar,
    omega_mag: Scalar,
    c: &GammaCoeffs,
) -> Scalar {
    5.0 * c.c_k * (gamma - 0.2).max(0.0) * (1.0 - gamma) * f_on_lim
        * (3.0 * c.c_sep * nu - nu_t).max(0.0)
        * s
        * omega_mag
}

/// `F_3 = exp(-(R_y/120)^8)` (90.17) - §88.6's to the character, `R_y`
/// included; written again here rather than imported from
/// [`crate::models::transition`], so the two models stay independent of each
/// other's private surface. The 2015 model adds to the blending only the
/// `max` that lets SST's own `F_1` win where it is already larger (§90.6).
#[must_use]
pub fn f3(r_y: Scalar) -> Scalar {
    (-(r_y / 120.0).min(1e6).powi(8)).exp()
}

// ==========================================================================
//  90.5  The source, and the split it is emitted through
// ==========================================================================

/// The Patankar split of (90.12), §90.5's table.
///
/// With `A = F_length S F_onset >= 0` and `B = c_a2 Omega F_turb >= 0`
/// handed in as `a` and `b` at the lagged state,
///
/// ```text
/// P_gamma - E_gamma = (A + B) gamma - (A + B c_e2) gamma^2            (90.12)
/// ```
///
/// is emitted as `su = 0`, `sp = (A + B c_e2) gamma_lagged >= 0` and
/// `susp = -(A + B) <= 0`. The square goes on the diagonal as a sink,
/// non-negative at every state; the linear half goes through
/// [`crate::fv::fvm_susp`], which moves a NON-POSITIVE coefficient to the
/// source (`source -= min(S, 0) psi`), so the emitted source is
/// `-susp gamma`. The split never divides by `gamma` anywhere, which is
/// what makes `gamma = 0` an honest part of the operating envelope rather
/// than a singularity (§90.5) - the absorbing state.
#[must_use]
pub fn gamma_source_split(
    gamma: Scalar,
    a: Scalar,
    b: Scalar,
    ce2: Scalar,
) -> (Scalar, Scalar, Scalar) {
    (0.0, (a + b * ce2) * gamma, -(a + b))
}

// ==========================================================================
//  90.2  Coefficients
// ==========================================================================

/// Every constant of the model, to the printed digit (§90.2's table), and
/// the two that are OURS (§90.8).
///
/// The dictionary keys these reach the solver through are the ones §91.1
/// writes: `Flength ce2 ca2 sigmaGamma CTU1 CTU2 CTU3 CPG1 CPG2 CPG3
/// CPG1lim CPG2lim ReThetacLim Ck Csep` and OURS `gammaMin gammaMax`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaCoeffs {
    /// `F_length`: the transition-length factor, ONE constant here against
    /// §88.2's four-piece correlation - whose 0.019 % and 0.810 %
    /// discontinuities the 2015 model does not carry over (§90.2).
    pub f_length: Scalar,
    /// `c_e2`: the destruction term (90.3) is §88.6's verbatim, and the
    /// laminar state is its fixed point `gamma = 1/c_e2` (§90.5).
    pub ce2: Scalar,
    pub ca2: Scalar,
    /// `sigma_gamma`: the `gamma` equation's diffusivity is
    /// `nu + nu_t/sigma_gamma`, the standard shape - §88's `sigma_tt`
    /// multiplied the molecular viscosity too, and this is not that
    /// (§90.5).
    pub sigma_gamma: Scalar,
    pub c_tu1: Scalar,
    pub c_tu2: Scalar,
    pub c_tu3: Scalar,
    pub c_pg1: Scalar,
    pub c_pg2: Scalar,
    /// `C_PG3`: at the printed `0.00` the term it carries in (90.7)
    /// vanishes; §90.3 records the reading that gives it a home at all.
    pub c_pg3: Scalar,
    pub c_pg1_lim: Scalar,
    pub c_pg2_lim: Scalar,
    /// `Re_thetac_lim`: (90.16)'s divisor, the coupling's own limiter
    /// constant - at the printed digit it is also `C_TU1 + C_TU2`, which is
    /// a coincidence §90.3 states and nothing exploits.
    pub re_thetac_lim: Scalar,
    pub c_k: Scalar,
    pub c_sep: Scalar,
    /// **OURS (§90.8).** `gamma` is bounded into `[gamma_min, gamma_max]`
    /// after its solve. The two may be equal: that freezes the
    /// intermittency, and `gammaMin = gammaMax = 1` is Gate 90-R's
    /// fully-turbulent limit.
    pub gamma_min: Scalar,
    pub gamma_max: Scalar,
}

impl Default for GammaCoeffs {
    fn default() -> Self {
        Self {
            f_length: 100.0,
            ce2: 50.0,
            ca2: 0.06,
            sigma_gamma: 1.0,
            c_tu1: 100.0,
            c_tu2: 1000.0,
            c_tu3: 1.0,
            c_pg1: 14.68,
            c_pg2: -7.34,
            c_pg3: 0.0,
            c_pg1_lim: 1.5,
            c_pg2_lim: 3.0,
            re_thetac_lim: 1100.0,
            c_k: 1.0,
            c_sep: 1.0,
            gamma_min: 0.0,
            gamma_max: 1.0,
        }
    }
}

impl GammaCoeffs {
    /// The checks that stop a coefficient set from being quietly impossible
    /// (§90.9's refusal list, the coefficient half of it). A hybrid or
    /// gravity under this model is refused where the model is ATTACHED, not
    /// here, for the same reason §88's is.
    pub fn check(&self) -> Result<()> {
        if self.ce2 <= 1.0 {
            return Err(Error::Config(format!(
                "momentumTransport/RAS/ce2 = {}: the laminar fixed point \
                 1/ce2 leaves [0, 1] and (ce2 gamma - 1) in E_gamma (90.3) \
                 goes negative on the whole operating range, turning \
                 destruction into source (SPEC-LIT 90.9). Menter et al.'s \
                 value is 50",
                self.ce2
            )));
        }
        if self.sigma_gamma <= 0.0 {
            return Err(Error::Config(format!(
                "momentumTransport/RAS/sigmaGamma = {}: gamma's diffusivity \
                 is nu + nu_t/sigma_gamma (SPEC-LIT 90.5) and a non-positive \
                 divisor makes the laplacian anti-diffusive. Menter et al.'s \
                 value is 1.0",
                self.sigma_gamma
            )));
        }
        if self.f_length < 0.0 {
            return Err(Error::Config(format!(
                "momentumTransport/RAS/Flength = {}: P_gamma (90.2) changes \
                 sign with it, so the onset machinery would drive gamma DOWN \
                 exactly where it should drive it up (SPEC-LIT 90.9). Menter \
                 et al.'s value is 100",
                self.f_length
            )));
        }
        if self.c_tu2 < 0.0 {
            return Err(Error::Config(format!(
                "momentumTransport/RAS/CTU2 = {}: the correlation's \
                 amplitude - with it negative Re_thetac (90.6) RISES with \
                 Tu_L and reaches C_TU1 + C_TU2 at clean air, non-positive \
                 from C_TU2 = -100 down, where F_onset1 divides by it \
                 (SPEC-LIT 90.9). Menter et al.'s value is 1000",
                self.c_tu2
            )));
        }
        if self.re_thetac_lim <= 0.0 {
            return Err(Error::Config(format!(
                "momentumTransport/RAS/ReThetacLim = {}: F_on^lim (90.16) \
                 divides by it, and a non-positive divisor is a NaN or a \
                 sign flip in the term that caps k's production at \
                 transition (SPEC-LIT 90.9). Menter et al.'s value is 1100",
                self.re_thetac_lim
            )));
        }
        if self.gamma_min > self.gamma_max {
            return Err(Error::Config(format!(
                "momentumTransport/RAS/gammaMin = {} is above gammaMax = {} \
                 (SPEC-LIT 90.8). The two may be EQUAL - that FREEZES the \
                 intermittency, and gammaMin = gammaMax = 1 is the \
                 fully-turbulent limit Gate 90-R runs the bitwise reduction \
                 on",
                self.gamma_min, self.gamma_max
            )));
        }
        if self.c_pg1_lim < 1.0 || self.c_pg2_lim < 1.0 {
            return Err(Error::Config(format!(
                "momentumTransport/RAS/CPG1lim = {} or CPG2lim = {}: either \
                 cap below 1 bites at lambda_thL = 0, where F_PG (90.7) \
                 would stop being 1 and Re_thetac stops being the \
                 two-constant form C_TU1 + C_TU2 exp(-C_TU3 Tu_L) (SPEC-LIT \
                 90.3, 90.9). Menter et al.'s values are 1.5 and 3.0",
                self.c_pg1_lim, self.c_pg2_lim
            )));
        }
        Ok(())
    }

    /// What the run banner prints. §90.7 promises the last words: after
    /// §88.9 measured LM2009's frame dependence, a line like this is a
    /// claim with a gate behind it.
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "Flength {} ce2 {} ca2 {} sigmaGamma {} CTU1 {} CTU2 {} CTU3 {} \
             CPG1 {} CPG2 {} CPG3 {} CPG1lim {} CPG2lim {} ReThetacLim {} \
             Ck {} Csep {} | OURS: gamma in [{}, {}] | Galilean invariant \
             (Gate 90-G)",
            self.f_length,
            self.ce2,
            self.ca2,
            self.sigma_gamma,
            self.c_tu1,
            self.c_tu2,
            self.c_tu3,
            self.c_pg1,
            self.c_pg2,
            self.c_pg3,
            self.c_pg1_lim,
            self.c_pg2_lim,
            self.re_thetac_lim,
            self.c_k,
            self.c_sep,
            self.gamma_min,
            self.gamma_max
        )
    }
}

/// The `gamma` equation's own `system/` settings, in the shape of
/// [`crate::models::transition::LmControls`]'s `gamma` half: `solvers/gamma`,
/// `relaxationFactors/equations/gamma` and `divSchemes/div(phi,gamma)` each
/// reach their own equation (§91.1), and each is pinned by a pair test when
/// the registry lands.
///
/// A separate struct rather than a third overload of
/// [`crate::io::case::TurbulenceControls`]' `epsilon_solver` slot, whose
/// doc already carries one "also used for" - §13.4.1's pair tests exist
/// because that kind of drift is otherwise silent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaControls {
    pub gamma_solver: SolverControls,
    pub gamma_relax: Scalar,
    pub gamma_conv: DivEntry,
}

impl Default for GammaControls {
    fn default() -> Self {
        let base = crate::io::case::TurbulenceControls::default();
        Self {
            gamma_solver: base.k_solver,
            gamma_relax: base.k_relax,
            gamma_conv: base.k_conv(),
        }
    }
}

// ==========================================================================
//  The device half
// ==========================================================================

/// Every entry point in `cuda/gmtrans.cu`, resolved once.
pub struct GmKernels {
    fields: CudaFunction,
    gamma_sources: CudaFunction,
    stamp_production: CudaFunction,
    stamp_k_sources: CudaFunction,
    stamp_f1: CudaFunction,
    bound_gamma: CudaFunction,
}

impl GmKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::GMTRANS)?;
        Ok(Self {
            fields: k.func("gmFields")?,
            gamma_sources: k.func("gmGammaSources")?,
            stamp_production: k.func("gmStampProduction")?,
            stamp_k_sources: k.func("gmStampKSources")?,
            stamp_f1: k.func("gmStampF1")?,
            bound_gamma: k.func("gmBoundGamma")?,
        })
    }
}

/// The one transported field, the seven closed forms it stands on, and the
/// three stamps that reach the background model - the one-equation twin of
/// [`crate::models::transition::LangtryMenter`], with the second equation
/// gone (SPEC-LIT §90.1) and the production stamp the Kato-Launder coupling
/// adds (§90.6).
///
/// Owns its buffers and allocates nothing in an outer iteration. Every
/// launch is guarded by `n == 0` the way LM's is, and nothing here reads a
/// velocity: `update_fields` takes no `U` at all, which is what makes
/// Gate 90-G's claim structural rather than measured.
pub struct MenterGamma {
    kern: GmKernels,
    fld: FieldKernels,
    coeffs: GammaCoeffs,
    ctrl: GammaControls,

    gamma: GpuScalarField,

    /// `[n_cells]` the wall distance of SPEC-LIT §6.6, and its gradient.
    /// Both are COPIED at construction, for the reason
    /// [`crate::models::transition::LangtryMenter`] copies `y`: they are
    /// computed once at setup and the model outlives the `WallDistance`
    /// that produced them. `grad_y` is a DIRECTION, not a unit vector -
    /// §57.6 measured `|grad y|` leaving 1 by 0.495 - and (90.11)
    /// normalises it per cell.
    y: DevBuf<Scalar>,
    grad_y: DevBuf<Vec3>,

    /// `[n_cells]` the vorticity magnitude `sqrt(2 W_ij W_ij)`, formed here
    /// and not by SST: (90.2) reads `S`, (90.3) reads `Omega`, and the two
    /// agree only in a pure shear (§40.2's warning, §90.2's).
    omega_mag: DevBuf<Scalar>,

    f_onset: DevBuf<Scalar>,
    f_turb: DevBuf<Scalar>,
    re_thetac: DevBuf<Scalar>,
    f_on_lim: DevBuf<Scalar>,
    f3: DevBuf<Scalar>,

    /// The two correlation inputs (90.9), (90.10), kept as diagnostics a
    /// test - and Gate 90-G's device leg - read.
    tu_l: DevBuf<Scalar>,
    lambda_thl: DevBuf<Scalar>,

    /// Gate 90-R (i)'s instrument: the Kato-Launder production stamp
    /// switched OFF, so the bitwise reduction to plain SST can be run
    /// without it. Test-only, like §88's `seed_stamp_inputs` - never a
    /// case setting (§90.8).
    #[cfg(test)]
    kato_launder: bool,
}

impl MenterGamma {
    /// `y` is the wall distance of SPEC-LIT §6.6 and `grad_y` its gradient;
    /// both are copied, not borrowed.
    pub fn new(
        gpu: &Gpu,
        mesh: &GpuMesh,
        coeffs: GammaCoeffs,
        ctrl: GammaControls,
        y: &DevBuf<Scalar>,
        grad_y: &DevBuf<Vec3>,
    ) -> Result<Self> {
        coeffs.check()?;
        if mesh.n_cells > 0 && (y.len() != mesh.n_cells || grad_y.len() != mesh.n_cells) {
            return Err(Error::Config(format!(
                "MenterGamma::new: the wall distance has {} entries and its \
                 gradient {} against {} cells (SPEC-LIT §6.6)",
                y.len(),
                grad_y.len(),
                mesh.n_cells
            )));
        }
        let nc = mesh.n_cells.max(1);
        let fld = FieldKernels::new(gpu)?;
        let mut y_own: DevBuf<Scalar> = gpu.zeros(nc)?;
        copy_field(gpu, &fld, &mut y_own, y, mesh.n_cells)?;
        let mut gy_own: DevBuf<Vec3> = gpu.zeros(nc)?;
        copy_field_vector(gpu, &fld, &mut gy_own, grad_y, mesh.n_cells)?;

        Ok(Self {
            kern: GmKernels::new(gpu)?,
            fld,
            coeffs,
            ctrl,
            gamma: GpuScalarField::zeros(gpu, mesh, "gamma")?,
            y: y_own,
            grad_y: gy_own,
            omega_mag: gpu.zeros(nc)?,
            f_onset: gpu.zeros(nc)?,
            f_turb: gpu.zeros(nc)?,
            re_thetac: gpu.zeros(nc)?,
            f_on_lim: gpu.zeros(nc)?,
            f3: gpu.zeros(nc)?,
            tu_l: gpu.zeros(nc)?,
            lambda_thl: gpu.zeros(nc)?,
            #[cfg(test)]
            kato_launder: true,
        })
    }

    #[must_use]
    pub fn coeffs(&self) -> &GammaCoeffs {
        &self.coeffs
    }
    #[must_use]
    pub fn controls(&self) -> &GammaControls {
        &self.ctrl
    }
    #[must_use]
    pub fn gamma(&self) -> &GpuScalarField {
        &self.gamma
    }
    pub fn gamma_mut(&mut self) -> &mut GpuScalarField {
        &mut self.gamma
    }
    #[must_use]
    pub fn f_onset(&self) -> &DevBuf<Scalar> {
        &self.f_onset
    }
    #[must_use]
    pub fn f_turb_field(&self) -> &DevBuf<Scalar> {
        &self.f_turb
    }
    #[must_use]
    pub fn re_thetac_field(&self) -> &DevBuf<Scalar> {
        &self.re_thetac
    }
    #[must_use]
    pub fn f_on_lim_field(&self) -> &DevBuf<Scalar> {
        &self.f_on_lim
    }
    #[must_use]
    pub fn f3_field(&self) -> &DevBuf<Scalar> {
        &self.f3
    }
    #[must_use]
    pub fn tu_l_field(&self) -> &DevBuf<Scalar> {
        &self.tu_l
    }
    #[must_use]
    pub fn lambda_thl_field(&self) -> &DevBuf<Scalar> {
        &self.lambda_thl
    }
    #[must_use]
    pub fn vorticity(&self) -> &DevBuf<Scalar> {
        &self.omega_mag
    }
    #[must_use]
    pub fn wall_distance(&self) -> &DevBuf<Scalar> {
        &self.y
    }

    /// The `0/` file a driver has to find for this model, beyond `k` and
    /// `omega`: one, against LM's two (§90.1 - the second equation is the
    /// one that is gone).
    #[must_use]
    pub fn named_fields(&self) -> Vec<(&'static str, &GpuScalarField)> {
        vec![("gamma", &self.gamma)]
    }

    pub fn named_fields_mut(&mut self) -> Vec<(&'static str, &mut GpuScalarField)> {
        vec![("gamma", &mut self.gamma)]
    }

    /// `psi^{n-2} <- psi^{n-1} <- psi`, once per step.
    ///
    /// Called from [`crate::models::KOmegaSst::correct`] beside `k`'s and
    /// `omega`'s, so all three equations see the same time levels.
    pub fn advance_time_levels(&mut self, gpu: &Gpu) -> Result<()> {
        advance_time_levels(gpu, &self.fld, &mut self.gamma)
    }

    /// Bound `gamma` and evaluate its boundary values, without solving.
    ///
    /// The counterpart of [`crate::models::KOmegaSst::initialise`]: a driver
    /// that has just read `0/gamma` calls this so the first `correct` sees a
    /// field the correlations can be evaluated on.
    pub fn initialise(&mut self, gpu: &Gpu, mesh: &GpuMesh) -> Result<()> {
        let n = mesh.n_cells;
        self.bound(gpu, n)?;
        correct_boundary_conditions(gpu, &self.fld, &mut self.gamma, mesh)
    }

    fn bound(&mut self, gpu: &Gpu, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let nl = n as crate::Label;
        let (lo, hi) = (self.coeffs.gamma_min, self.coeffs.gamma_max);
        let f = self.kern.bound_gamma.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.gamma.f)
                .arg(&lo)
                .arg(&hi)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// §90.2-§90.4: the seven closed forms, one launch, from the fields the
    /// previous outer iteration left.
    ///
    /// `s` is the strain-rate magnitude the background model has already
    /// formed; the vorticity magnitude is formed here, because SST does not
    /// need one and a buffer nobody reads is a buffer that goes stale.
    ///
    /// **There is no velocity argument, and nothing here reads `U`.** Every
    /// input is a gradient, a turbulence quantity or the wall distance - the
    /// structural form of Gate 90-G's claim that a frame shift moves
    /// nothing (§90.7).
    #[allow(clippy::too_many_arguments)]
    pub fn update_fields(
        &mut self,
        gpu: &Gpu,
        turb: &TurbKernels,
        k: &DevBuf<Scalar>,
        omega: &DevBuf<Scalar>,
        s: &DevBuf<Scalar>,
        grad_u: &DevBuf<Tensor>,
        nu: Scalar,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        vorticity_mag(gpu, turb, &mut self.omega_mag, grad_u, n)?;

        let nl = n as crate::Label;
        let c = self.coeffs;
        let f = self.kern.fields.clone();

        let Self {
            f_onset,
            f_turb,
            re_thetac,
            f_on_lim,
            f3,
            tu_l,
            lambda_thl,
            gamma,
            y,
            grad_y,
            omega_mag,
            ..
        } = self;

        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(f_onset)
                .arg(f_turb)
                .arg(re_thetac)
                .arg(f_on_lim)
                .arg(f3)
                .arg(tu_l)
                .arg(lambda_thl)
                .arg(&gamma.f)
                .arg(k)
                .arg(omega)
                .arg(s)
                .arg(&*omega_mag)
                .arg(&*y)
                .arg(&*grad_y)
                .arg(grad_u)
                .arg(&nu)
                .arg(&c.c_tu1)
                .arg(&c.c_tu2)
                .arg(&c.c_tu3)
                .arg(&c.c_pg1)
                .arg(&c.c_pg2)
                .arg(&c.c_pg3)
                .arg(&c.c_pg1_lim)
                .arg(&c.c_pg2_lim)
                .arg(&c.re_thetac_lim)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// §90.6: `F_1 <- max(F_1, F_3)`, stamped between `sstBlending` and
    /// `sstBlendCoeffs`.
    pub fn stamp_f1(&self, gpu: &Gpu, f1: &mut DevBuf<Scalar>, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let nl = n as crate::Label;
        let f = self.kern.stamp_f1.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(f1)
                .arg(&self.f3)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// §90.6: the production replacement - `g <- nu_t S Omega` (what the
    /// `k` equation reads) and `p <- S Omega` (what the `omega` equation
    /// reads per unit `nu_t`), stamped in `update_blending` BEFORE the
    /// `omega` equation assembles. SST's own production limiter is NOT
    /// applied here: `sstKSources` limits whatever it is handed, and it is
    /// handed this (§90.6).
    ///
    /// A no-op while the `kato_launder` instrument is false - Gate 90-R
    /// (i)'s instrument, test-only.
    pub fn stamp_production(
        &self,
        gpu: &Gpu,
        g: &mut DevBuf<Scalar>,
        p: &mut DevBuf<Scalar>,
        nut: &DevBuf<Scalar>,
        s: &DevBuf<Scalar>,
        n: usize,
    ) -> Result<()> {
        #[cfg(test)]
        {
            if !self.kato_launder {
                return Ok(());
            }
        }
        if n == 0 {
            return Ok(());
        }
        let nl = n as crate::Label;
        let f = self.kern.stamp_production.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(g)
                .arg(p)
                .arg(nut)
                .arg(s)
                .arg(&self.omega_mag)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// §90.6: `g_lim <- gamma g_lim + P_k^lim` and `sp <- max(gamma, 0.1)
    /// sp`, stamped over what `sstKSources` has just written - the limiter
    /// has already been applied to the production this model replaced
    /// (§90.6).
    #[allow(clippy::too_many_arguments)]
    pub fn stamp_k_sources(
        &self,
        gpu: &Gpu,
        g_lim: &mut DevBuf<Scalar>,
        sp: &mut DevBuf<Scalar>,
        nut: &DevBuf<Scalar>,
        s: &DevBuf<Scalar>,
        nu: Scalar,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let nl = n as crate::Label;
        let (ck, csep) = (self.coeffs.c_k, self.coeffs.c_sep);
        let f = self.kern.stamp_k_sources.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(g_lim)
                .arg(sp)
                .arg(&self.gamma.f)
                .arg(&self.f_on_lim)
                .arg(nut)
                .arg(s)
                .arg(&self.omega_mag)
                .arg(&nu)
                .arg(&ck)
                .arg(&csep)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// §90.5: assemble and solve `gamma` - the ONE equation, where §88.5
    /// had two.
    ///
    /// It runs through the background model's own [`RasCore`] - the same
    /// matrix, the same workspace, the same `ddt + div - laplacian` - with
    /// `r_sigma = 1/sigma_gamma`, the plain entry point: the page's
    /// `mu + mu_t/sigma_gamma` is the standard shape, and the kinematic
    /// form divides it straight through (§90.5, against §88's affine one).
    ///
    /// ONE [`SolverPerformance`]: `gamma`'s solve is the whole of it. The
    /// performance is not returned from `correct` - that signature is
    /// §6.3's and belongs to the two equations SST owns.
    pub fn solve(
        &mut self,
        gpu: &Gpu,
        core: &mut RasCore<'_>,
        flow: &FlowState,
        s: &DevBuf<Scalar>,
    ) -> Result<SolverPerformance> {
        let n = core.mesh.n_cells;
        let c = self.coeffs;

        core.assemble_transport(gpu, flow, &self.gamma, self.ctrl.gamma_conv, 1.0 / c.sigma_gamma)?;

        {
            let Self {
                kern,
                gamma,
                f_onset,
                f_turb,
                omega_mag,
                ..
            } = self;
            let RasCore { su, sp, susp, .. } = core;
            let nl = n as crate::Label;
            let f = kern.gamma_sources.clone();
            if n > 0 {
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(su)
                        .arg(sp)
                        .arg(susp)
                        .arg(&gamma.f)
                        .arg(&*f_onset)
                        .arg(&*f_turb)
                        .arg(s)
                        .arg(&*omega_mag)
                        .arg(&c.f_length)
                        .arg(&c.ca2)
                        .arg(&c.ce2)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }
        }

        fvm_su(gpu, &core.fv, &mut core.a, core.mesh, &core.su, 1.0)?;
        fvm_sp(gpu, &core.fv, &mut core.a, core.mesh, &core.sp, 1.0)?;
        fvm_susp(
            gpu,
            &core.fv,
            &mut core.a,
            core.mesh,
            &core.susp,
            &self.gamma.f,
            1.0,
        )?;

        let sc = self.ctrl.gamma_solver;
        let perf =
            core.solve_equation(gpu, &mut self.gamma, self.ctrl.gamma_relax, &sc, false)?;

        self.bound(gpu, n)?;
        correct_boundary_conditions(gpu, &self.fld, &mut self.gamma, core.mesh)?;
        Ok(perf)
    }

    /// Write `F_3` directly, for Gate 90-R's stamp half.
    ///
    /// The gate's claim is about the two stamps in isolation - each is the
    /// identity at its neutral value, on every bit - and the only way to put
    /// a stamp at its neutral value is to say what the factor is. Test-only,
    /// because a run has no business writing it: it is an output of
    /// [`Self::update_fields`]. (`gamma` needs no seeding here: it is a
    /// transported field the test writes through [`Self::gamma_mut`].)
    #[cfg(test)]
    pub(crate) fn seed_stamp_inputs(&mut self, gpu: &Gpu, f3: &[Scalar]) -> Result<()> {
        gpu.write(&mut self.f3, f3)
    }

    /// Gate 90-R (i)'s other half: the Kato-Launder production stamp OFF.
    /// Test-only, never a case setting (§90.8).
    #[cfg(test)]
    pub(crate) fn set_kato_launder(&mut self, on: bool) {
        self.kato_launder = on;
    }

    /// Launch [`GmKernels::gamma_sources`] in isolation, for the split tests:
    /// they must reach the kernel without the [`RasCore`] a real `solve`
    /// needs, and the capture audit's population is about RUNTIME launches,
    /// so the raw `launch_builder` lives here in the model's own file rather
    /// than in `tests.rs`. Test-only: a run reaches this kernel only through
    /// [`Self::solve`].
    #[cfg(test)]
    pub(crate) fn gamma_sources_for_test(
        &self,
        gpu: &Gpu,
        su: &mut DevBuf<Scalar>,
        sp: &mut DevBuf<Scalar>,
        susp: &mut DevBuf<Scalar>,
        s: &DevBuf<Scalar>,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let nl = n as crate::Label;
        let c = self.coeffs;
        let f = self.kern.gamma_sources.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(su)
                .arg(sp)
                .arg(susp)
                .arg(&self.gamma.f)
                .arg(&self.f_onset)
                .arg(&self.f_turb)
                .arg(s)
                .arg(&self.omega_mag)
                .arg(&c.f_length)
                .arg(&c.ca2)
                .arg(&c.ce2)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
