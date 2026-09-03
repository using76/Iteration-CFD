// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The Langtry-Menter gamma-Re_theta transition model - SPEC-LIT §88.
//!
//! Written from:
//!   Langtry & Menter, "Correlation-Based Transition Modeling for
//!     Unstructured Parallelized Computational Fluid Dynamics Codes",
//!     *AIAA J.* 47 (2009) 2894-2906 - **the** paper. The 2006 pair
//!     withholds the correlations and is not enough to write the model.
//!   NASA / TMBWG, *Turbulence Modeling Resource - Langtry-Menter 4-equation
//!     Transitional SST Model (SST-2003-LM2009)*,
//!     <https://tmbwg.github.io/turbmodels/langtrymenter_4eqn.html> - US
//!     government-authored DOCUMENTATION, not source. **Fetched and read
//!     while writing this module**; every constant below is transcribed from
//!     it, including the three numerical limits the paper's own text is
//!     easier to lose: `lambda_theta` clipped to `[-0.1, 0.1]`, `Tu >= 0.027`
//!     and `Re_theta_eq >= 20`.
//!   Menter, Langtry, Likki, Suzen, Huang & Voelker, *J. Turbomach.* 128
//!     (2006) 413-422 - the model's structure
//!   NASA / TMBWG, *2D T3A Transitional Flat Plate*,
//!     <https://tmbwg.github.io/turbmodels/t3_transition_mainpage.html> -
//!     read; it is where §88.10's inflow state comes from
//!   Patankar, *Numerical Heat Transfer and Fluid Flow* (1980) §4.2
//!   ofgpu `SPEC-LIT.md` §88, and §6.3 for the background model
//! No GPL-licensed source was consulted. OpenFOAM's and SU2's transition
//! implementations were not opened, searched or quoted.
//!
//! # Four equations, two of them somebody else's
//!
//! This module owns `gamma` and `Re_theta~`. `k` and `omega` stay entirely
//! inside [`crate::models::KOmegaSst`], and the coupling runs one way at a
//! time through three stamps:
//!
//! | stamp | where | what |
//! |---|---|---|
//! | [`LangtryMenter::stamp_f1`] | between `sstBlending` and `sstBlendCoeffs` | `F_1 <- max(F_1, F_3)` |
//! | [`LangtryMenter::stamp_k_sources`] | after `sstKSources` | `P_k <- gamma_eff P_k`, `D_k <- min(max(gamma_eff, 0.1), 1) D_k` |
//! | [`LangtryMenter::solve`] | after `correct_nut` | the two transport equations |
//!
//! **`cuda/sst.cu` has a zero-line diff.** A case that names no transition
//! model runs three failed `if let`s and nothing else: not one kernel
//! launch, not one floating-point operation. That is the same construction
//! §57.7 used for the hybrids and it is why "the default is unmoved" is a
//! statement about the diff rather than about a tolerance.
//!
//! # The one loop, and why its trip count is a constant
//!
//! `Re_theta_eq = f(Tu, lambda_theta)` and `lambda_theta` is built from a
//! momentum thickness that is itself `Re_theta_eq nu/U`, so the correlation
//! appears inside its own argument. Langtry & Menter prescribe iterating to
//! convergence. This runs exactly [`LmCoeffs::n_sweeps`] sweeps, every cell,
//! every iteration - a convergence test would make the trip count depend on
//! a floating-point comparison, which costs warp coherence and, far worse
//! here, bitwise reproducibility and CUDA-graph capture. `n_sweeps` is
//! **ours**; §88.4 measures what it is worth.
//!
//! # What is Galilean-invariant here, and what is not
//!
//! `Tu = 100 sqrt(2k/3)/U` and `T = 500 nu/U^2` read an **absolute** velocity
//! magnitude, so this model's answer changes if the frame is translated.
//! That is a property of LM2009, not of this implementation, and it is the
//! defect Menter et al. (2015) fixed with the one-equation `gamma` model.
//! `tests::the_model_is_not_galilean_invariant_and_this_measures_it` measures
//! the size of it rather than leaving it as a remark, and §88.9 records it.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::{GpuScalarField, GpuVectorField};
use crate::field_ops::{advance_time_levels, correct_boundary_conditions, FieldKernels};
use crate::fv::{fvm_sp, fvm_su, fvm_susp};
use crate::io::case::SolverControls;
use crate::io::schemes::DivEntry;
use crate::mesh::GpuMesh;
use crate::solver::SolverPerformance;
use crate::turbulence::{vorticity_mag, FlowState, RasCore, TurbKernels};
use crate::{Label, Scalar, Tensor, Vec3};

// ==========================================================================
//  §88.2  The published correlations, as host functions
//
//  Every one of these is the CPU twin of a device function in
//  `cuda/lmtrans.cu`, written in the same order with the same constants, and
//  `tests::the_host_and_device_correlations_agree` measures the two against
//  each other over a sweep rather than trusting that they were typed alike.
// ==========================================================================

/// `Re_thetac(Re_theta~)` - the CRITICAL momentum-thickness Reynolds number.
///
/// The TMR's expanded form. `Re_thetac < Re_theta~` over the whole fitted
/// range: the correlation is a fit to the distance between "the model may
/// start producing intermittency" and "the experiment records transition".
#[must_use]
pub fn re_thetac(r: Scalar) -> Scalar {
    if r <= 1870.0 {
        -3.96035 + 1.0120656 * r - 868.230e-6 * r * r + 696.506e-9 * r * r * r
            - 174.105e-12 * r * r * r * r
    } else {
        r - (593.11 + 0.482 * (r - 1870.0))
    }
}

/// The same correlation in Langtry & Menter's own NESTED form,
/// `Re_theta~ - f(Re_theta~)`.
///
/// Kept because the two are algebraically identical and numerically are not,
/// and because a reader comparing against the paper rather than against the
/// TMR will find this one. `tests::the_two_forms_of_re_thetac_agree`
/// measures the gap.
#[must_use]
pub fn re_thetac_nested(r: Scalar) -> Scalar {
    if r <= 1870.0 {
        r - (3.96035 - 120.656e-4 * r + 868.230e-6 * r * r - 696.506e-9 * r * r * r
            + 174.105e-12 * r * r * r * r)
    } else {
        r - (593.11 + 0.482 * (r - 1870.0))
    }
}

/// `F_length,1(Re_theta~)` - the transition-length correlation, before the
/// viscous-sublayer blend.
#[must_use]
pub fn f_length1(r: Scalar) -> Scalar {
    if r < 400.0 {
        39.8189 - 119.270e-4 * r - 132.567e-6 * r * r
    } else if r < 596.0 {
        263.404 - 123.939e-2 * r + 194.548e-5 * r * r - 101.695e-8 * r * r * r
    } else if r < 1200.0 {
        0.5 - 3.0e-4 * (r - 596.0)
    } else {
        0.3188
    }
}

/// `F_length = F_length,1 (1 - F_sublayer) + 40 F_sublayer`,
/// `F_sublayer = exp(-(Re_w/200)^2)`, `Re_w = omega d^2/nu`.
#[must_use]
pub fn f_length(r: Scalar, re_w: Scalar) -> Scalar {
    let f_sub = (-(re_w / 200.0).powi(2)).exp();
    f_length1(r) * (1.0 - f_sub) + 40.0 * f_sub
}

/// `Tu = 100 sqrt(2k/3)/U`, floored at the TMR's published `0.027`.
#[must_use]
pub fn turbulence_intensity(k: Scalar, u_mag: Scalar) -> Scalar {
    let tu = 100.0 * (2.0 * k.max(0.0) / 3.0).sqrt() / u_mag.max(1e-30);
    tu.max(0.027)
}

/// `Re_theta_eq(Tu, lambda_theta)` - ONE evaluation, no fixed point.
///
/// The `max(_, 20)` at the end is the TMR's own published limit, not ours.
#[must_use]
pub fn re_theta_eq_raw(tu: Scalar, lambda: Scalar) -> Scalar {
    let f = if lambda <= 0.0 {
        let e = (-(tu / 1.5).powf(1.5)).exp();
        1.0 + (12.986 * lambda + 123.66 * lambda * lambda + 405.689 * lambda * lambda * lambda) * e
    } else {
        1.0 + 0.275 * (1.0 - (-35.0 * lambda).exp()) * (-tu / 0.5).exp()
    };

    let re = if tu <= 1.3 {
        (1173.51 - 589.428 * tu + 0.2196 / (tu * tu)) * f
    } else {
        331.50 * (tu - 0.5658).powf(-0.671) * f
    };

    re.max(20.0)
}

/// The fixed point of [`re_theta_eq_raw`], run for exactly `n_sweeps` sweeps
/// from the zero-pressure-gradient value.
#[must_use]
pub fn re_theta_eq(tu: Scalar, du_ds: Scalar, nu: Scalar, u_mag: Scalar, n_sweeps: usize) -> Scalar {
    let mut re = re_theta_eq_raw(tu, 0.0);
    for _ in 0..n_sweeps {
        let theta = re * nu / u_mag.max(1e-30);
        let lambda = (theta * theta * du_ds / nu.max(1e-30)).clamp(-0.1, 0.1);
        re = re_theta_eq_raw(tu, lambda);
    }
    re
}

/// The free-stream (`lambda_theta = 0`) inlet value of `Re_theta~` for a
/// given free-stream turbulence intensity, in percent.
///
/// This is the TMR's own farfield boundary condition for `Re_theta~`, and it
/// is exposed because writing that number into a `0/ReThetat` file by hand
/// means solving the same correlation on paper, and getting it wrong is
/// silent. §89.2 says where a case reaches it.
#[must_use]
pub fn re_thetat_inlet(tu_percent: Scalar) -> Scalar {
    re_theta_eq_raw(tu_percent.max(0.027), 0.0)
}

/// `F_onset = max(F_onset2 - F_onset3, 0)`.
#[must_use]
pub fn f_onset(re_v: Scalar, re_thetac: Scalar, r_t: Scalar) -> Scalar {
    let fo1 = re_v / (2.193 * re_thetac.max(1e-30));
    let fo2 = fo1.max(fo1.min(1e6).powi(4)).min(2.0);
    let fo3 = (1.0 - (r_t / 2.5).powi(3)).max(0.0);
    (fo2 - fo3).max(0.0)
}

/// `F_turb = exp(-(R_T/4)^4)`.
#[must_use]
pub fn f_turb(r_t: Scalar) -> Scalar {
    (-(r_t / 4.0).min(1e6).powi(4)).exp()
}

/// `F_3 = exp(-(R_y/120)^8)`, `R_y = d sqrt(k)/nu`.
#[must_use]
pub fn f3(r_y: Scalar) -> Scalar {
    (-(r_y / 120.0).min(1e6).powi(8)).exp()
}

/// `2.193`, the ratio of the maximum vorticity Reynolds number across a
/// Blasius profile to that profile's momentum-thickness Reynolds number.
///
/// Named rather than repeated, because it is the constant that makes the
/// strictly local `Re_V` a stand-in for a quantity that is an integral
/// across the layer, and §88.10's gate derives it independently from a
/// Blasius solution rather than accepting it.
pub const BLASIUS_REV_OVER_RETHETA: Scalar = 2.193;

// ==========================================================================
//  §88.9  Coefficients
// ==========================================================================

/// Every constant of the model, and the three that are OURS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LmCoeffs {
    pub ca1: Scalar,
    pub ca2: Scalar,
    pub ce1: Scalar,
    pub ce2: Scalar,
    pub ctt: Scalar,
    pub s1: Scalar,
    /// `sigma_f`: the intermittency equation's diffusivity is
    /// `nu + nu_t/sigma_f`.
    pub sigma_f: Scalar,
    /// `sigma_tt`: `Re_theta~`'s diffusivity is `sigma_tt (nu + nu_t)` -
    /// note that it multiplies the EFFECTIVE viscosity, molecular part
    /// included, which is why this equation goes through
    /// `assemble_transport_affine` and not `assemble_transport`.
    pub sigma_tt: Scalar,

    /// **OURS (§88.4).** How many sweeps the `Re_theta_eq` fixed point runs.
    /// Langtry & Menter say "iterate to convergence"; a data-dependent trip
    /// count is not capturable and not bitwise.
    pub n_sweeps: usize,
    /// **OURS (§88.8).** `gamma` is bounded into `[gamma_min, gamma_max]`
    /// after its solve.
    pub gamma_min: Scalar,
    pub gamma_max: Scalar,
    /// **OURS (§88.8).** `Re_theta~` is floored after its solve, at the same
    /// `20` the TMR puts on `Re_theta_eq`, so that [`re_thetac`] can never be
    /// handed an argument outside the range its polynomial was fitted over.
    pub re_thetat_min: Scalar,
}

impl Default for LmCoeffs {
    fn default() -> Self {
        Self {
            ca1: 2.0,
            ca2: 0.06,
            ce1: 1.0,
            ce2: 50.0,
            ctt: 0.03,
            s1: 2.0,
            sigma_f: 1.0,
            sigma_tt: 2.0,
            n_sweeps: 10,
            gamma_min: 0.0,
            gamma_max: 1.0,
            re_thetat_min: 20.0,
        }
    }
}

impl LmCoeffs {
    /// The checks that stop a coefficient set from being quietly impossible.
    pub fn check(&self) -> Result<()> {
        if self.ce2 <= 1.0 {
            return Err(Error::Config(format!(
                "momentumTransport/RAS/ce2 = {}: F_thetat carries \
                 (ce2 gamma - 1)/(ce2 - 1) (SPEC-LIT (88.11)), which is a \
                 division by zero at ce2 = 1 and changes sign below it. \
                 Langtry & Menter's value is 50",
                self.ce2
            )));
        }
        if self.sigma_f <= 0.0 || self.sigma_tt <= 0.0 {
            return Err(Error::Config(format!(
                "momentumTransport/RAS/sigmaf = {} and sigmaThetat = {}: both \
                 are diffusivity DIVISORS/multipliers and a non-positive one \
                 makes the laplacian anti-diffusive (SPEC-LIT 88.5). Langtry \
                 & Menter's values are 1.0 and 2.0",
                self.sigma_f, self.sigma_tt
            )));
        }
        if self.n_sweeps == 0 || self.n_sweeps > 100 {
            return Err(Error::Config(format!(
                "momentumTransport/RAS/nReThetaSweeps = {}: the Re_theta_eq \
                 fixed point runs a FIXED number of sweeps (SPEC-LIT 88.4), \
                 and zero of them means the pressure-gradient factor \
                 F(lambda_theta) is never evaluated at all. The default is 10 \
                 and 100 is the ceiling",
                self.n_sweeps
            )));
        }
        if self.gamma_min > self.gamma_max {
            return Err(Error::Config(format!(
                "momentumTransport/RAS/gammaMin = {} is above gammaMax = {} \
                 (SPEC-LIT 88.8). The two may be EQUAL, and that is a real \
                 setting rather than a degenerate one: it FREEZES the \
                 intermittency, and gammaMin = gammaMax = 1 is the \
                 fully-turbulent limit Gate 88-R runs the bitwise reduction \
                 to plain kOmegaSST on",
                self.gamma_min, self.gamma_max
            )));
        }
        if self.re_thetat_min < 20.0 {
            return Err(Error::Config(format!(
                "momentumTransport/RAS/ReThetatMin = {}: below 20 the \
                 Re_thetac polynomial (88.3) is evaluated outside the range \
                 Langtry & Menter fitted it over, where it crosses zero and \
                 F_onset1 divides by it (SPEC-LIT 88.8). 20 is the TMR's own \
                 published floor on Re_theta_eq",
                self.re_thetat_min
            )));
        }
        Ok(())
    }

    /// What the run banner prints.
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "ca1 {} ca2 {} ce1 {} ce2 {} cThetat {} s1 {} sigmaf {} \
             sigmaThetat {} | OURS: nReThetaSweeps {} gamma in [{}, {}] \
             ReThetatMin {}",
            self.ca1,
            self.ca2,
            self.ce1,
            self.ce2,
            self.ctt,
            self.s1,
            self.sigma_f,
            self.sigma_tt,
            self.n_sweeps,
            self.gamma_min,
            self.gamma_max,
            self.re_thetat_min
        )
    }
}

/// The two new equations' own `system/` settings.
///
/// A separate struct rather than two more fields on
/// [`crate::io::case::TurbulenceControls`], whose `epsilon_solver` is already
/// documented as "also used for omega - the two never coexist". Three
/// dissipation-like variables DO coexist here, and overloading that slot a
/// third time is exactly the drift §13.4.1's pair tests exist to catch:
/// `solvers/gamma`, `relaxationFactors/equations/gamma` and
/// `divSchemes/div(phi,gamma)` each reach their own equation and are each
/// pinned by a pair test in §89.4.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LmControls {
    pub gamma_solver: SolverControls,
    pub gamma_relax: Scalar,
    pub gamma_conv: DivEntry,
    pub re_thetat_solver: SolverControls,
    pub re_thetat_relax: Scalar,
    pub re_thetat_conv: DivEntry,
}

impl Default for LmControls {
    fn default() -> Self {
        let base = crate::io::case::TurbulenceControls::default();
        Self {
            gamma_solver: base.k_solver,
            gamma_relax: base.k_relax,
            gamma_conv: base.k_conv(),
            re_thetat_solver: base.k_solver,
            re_thetat_relax: base.k_relax,
            re_thetat_conv: base.k_conv(),
        }
    }
}

// ==========================================================================
//  Kernels
// ==========================================================================

/// Every entry point in `cuda/lmtrans.cu`, resolved once.
pub struct LmKernels {
    fields: CudaFunction,
    gamma_sources: CudaFunction,
    re_thetat_sources: CudaFunction,
    stamp_k_sources: CudaFunction,
    stamp_f1: CudaFunction,
    bound_gamma: CudaFunction,
    bound_re_thetat: CudaFunction,
}

impl LmKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::LMTRANS)?;
        Ok(Self {
            fields: k.func("lmFields")?,
            gamma_sources: k.func("lmGammaSources")?,
            re_thetat_sources: k.func("lmReThetatSources")?,
            stamp_k_sources: k.func("lmStampKSources")?,
            stamp_f1: k.func("lmStampF1")?,
            bound_gamma: k.func("lmBoundGamma")?,
            bound_re_thetat: k.func("lmBoundReThetat")?,
        })
    }
}

// ==========================================================================
//  The model
// ==========================================================================

/// The two transported fields, the eight closed forms they stand on, and the
/// three stamps that reach the background model.
///
/// Owns its buffers and allocates nothing in an outer iteration.
pub struct LangtryMenter {
    kern: LmKernels,
    fld: FieldKernels,
    coeffs: LmCoeffs,
    ctrl: LmControls,

    gamma: GpuScalarField,
    re_thetat: GpuScalarField,

    /// `[n_cells]` the wall distance of SPEC-LIT §6.6, copied in at
    /// construction for the same reason [`crate::models::KOmegaSst`] copies
    /// its own: it is computed once at setup and the model outlives the
    /// `WallDistance` that produced it.
    y: DevBuf<Scalar>,

    /// `[n_cells]` the vorticity magnitude `sqrt(2 W_ij W_ij)`. NOT the
    /// strain rate: `P_gamma` reads `S` and `E_gamma` reads `Omega`, and the
    /// two are different numbers wherever the flow rotates.
    omega_mag: DevBuf<Scalar>,

    f_onset: DevBuf<Scalar>,
    f_turb: DevBuf<Scalar>,
    f_length: DevBuf<Scalar>,
    re_thetac: DevBuf<Scalar>,
    f_thetat: DevBuf<Scalar>,
    gamma_eff: DevBuf<Scalar>,
    re_theta_eq: DevBuf<Scalar>,
    f3: DevBuf<Scalar>,
}

impl LangtryMenter {
    /// `y` is the wall distance of SPEC-LIT §6.6 and is copied, not borrowed.
    pub fn new(
        gpu: &Gpu,
        mesh: &GpuMesh,
        coeffs: LmCoeffs,
        ctrl: LmControls,
        y: &DevBuf<Scalar>,
    ) -> Result<Self> {
        coeffs.check()?;
        if y.len() != mesh.n_cells && mesh.n_cells > 0 {
            return Err(Error::Config(format!(
                "LangtryMenter::new: the wall distance has {} entries and the \
                 mesh {} cells (SPEC-LIT §6.6)",
                y.len(),
                mesh.n_cells
            )));
        }
        let nc = mesh.n_cells.max(1);
        let fld = FieldKernels::new(gpu)?;
        let mut y_own: DevBuf<Scalar> = gpu.zeros(nc)?;
        crate::field_ops::copy_field(gpu, &fld, &mut y_own, y, mesh.n_cells)?;

        Ok(Self {
            kern: LmKernels::new(gpu)?,
            fld,
            coeffs,
            ctrl,
            gamma: GpuScalarField::zeros(gpu, mesh, "gamma")?,
            re_thetat: GpuScalarField::zeros(gpu, mesh, "ReThetat")?,
            y: y_own,
            omega_mag: gpu.zeros(nc)?,
            f_onset: gpu.zeros(nc)?,
            f_turb: gpu.zeros(nc)?,
            f_length: gpu.zeros(nc)?,
            re_thetac: gpu.zeros(nc)?,
            f_thetat: gpu.zeros(nc)?,
            gamma_eff: gpu.zeros(nc)?,
            re_theta_eq: gpu.zeros(nc)?,
            f3: gpu.zeros(nc)?,
        })
    }

    #[must_use]
    pub fn coeffs(&self) -> &LmCoeffs {
        &self.coeffs
    }
    #[must_use]
    pub fn controls(&self) -> &LmControls {
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
    pub fn re_thetat(&self) -> &GpuScalarField {
        &self.re_thetat
    }
    pub fn re_thetat_mut(&mut self) -> &mut GpuScalarField {
        &mut self.re_thetat
    }
    /// `gamma_eff = max(gamma, gamma_sep)` - what the `k` equation sees, and
    /// the one diagnostic that says whether the model has switched.
    #[must_use]
    pub fn gamma_eff(&self) -> &DevBuf<Scalar> {
        &self.gamma_eff
    }
    #[must_use]
    pub fn f_onset(&self) -> &DevBuf<Scalar> {
        &self.f_onset
    }
    #[must_use]
    pub fn f_length_field(&self) -> &DevBuf<Scalar> {
        &self.f_length
    }
    #[must_use]
    pub fn re_thetac_field(&self) -> &DevBuf<Scalar> {
        &self.re_thetac
    }
    #[must_use]
    pub fn re_theta_eq_field(&self) -> &DevBuf<Scalar> {
        &self.re_theta_eq
    }
    #[must_use]
    pub fn f3_field(&self) -> &DevBuf<Scalar> {
        &self.f3
    }
    #[must_use]
    pub fn f_thetat_field(&self) -> &DevBuf<Scalar> {
        &self.f_thetat
    }
    #[must_use]
    pub fn vorticity(&self) -> &DevBuf<Scalar> {
        &self.omega_mag
    }
    #[must_use]
    pub fn wall_distance(&self) -> &DevBuf<Scalar> {
        &self.y
    }

    /// The `0/` files a driver has to find for this model, beyond `k` and
    /// `omega`.
    #[must_use]
    pub fn named_fields(&self) -> Vec<(&'static str, &GpuScalarField)> {
        vec![("gamma", &self.gamma), ("ReThetat", &self.re_thetat)]
    }

    pub fn named_fields_mut(&mut self) -> Vec<(&'static str, &mut GpuScalarField)> {
        vec![
            ("gamma", &mut self.gamma),
            ("ReThetat", &mut self.re_thetat),
        ]
    }

    /// `psi^{n-2} <- psi^{n-1} <- psi` on both fields, once per step.
    ///
    /// Called from [`crate::models::KOmegaSst::correct`] beside `k`'s and
    /// `omega`'s, so all four equations see the same time levels.
    pub fn advance_time_levels(&mut self, gpu: &Gpu) -> Result<()> {
        advance_time_levels(gpu, &self.fld, &mut self.gamma)?;
        advance_time_levels(gpu, &self.fld, &mut self.re_thetat)
    }

    /// Bound both fields and evaluate their boundary values, without solving.
    ///
    /// The counterpart of `KOmegaSst::initialise`: a driver that has just
    /// read `0/gamma` and `0/ReThetat` calls this so the first `correct` sees
    /// a field the correlations can be evaluated on.
    pub fn initialise(&mut self, gpu: &Gpu, mesh: &GpuMesh) -> Result<()> {
        let n = mesh.n_cells;
        self.bound(gpu, n)?;
        correct_boundary_conditions(gpu, &self.fld, &mut self.gamma, mesh)?;
        correct_boundary_conditions(gpu, &self.fld, &mut self.re_thetat, mesh)
    }

    fn bound(&mut self, gpu: &Gpu, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let nl = n as Label;
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
        let floor = self.coeffs.re_thetat_min;
        let f = self.kern.bound_re_thetat.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.re_thetat.f)
                .arg(&floor)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// §88.3: the eight closed forms, one launch, from the fields the
    /// previous outer iteration left.
    ///
    /// `s` is the strain-rate magnitude the background model has already
    /// formed; the vorticity magnitude is formed here, because SST does not
    /// need one and a buffer nobody reads is a buffer that goes stale.
    #[allow(clippy::too_many_arguments)]
    pub fn update_fields(
        &mut self,
        gpu: &Gpu,
        turb: &TurbKernels,
        k: &DevBuf<Scalar>,
        omega: &DevBuf<Scalar>,
        s: &DevBuf<Scalar>,
        u: &GpuVectorField,
        grad_u: &DevBuf<Tensor>,
        nu: Scalar,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        vorticity_mag(gpu, turb, &mut self.omega_mag, grad_u, n)?;

        let nl = n as Label;
        let sweeps = self.coeffs.n_sweeps as Label;
        let ce2 = self.coeffs.ce2;
        let s1 = self.coeffs.s1;
        let f = self.kern.fields.clone();

        let Self {
            f_onset,
            f_turb,
            f_length,
            re_thetac,
            f_thetat,
            gamma_eff,
            re_theta_eq,
            f3,
            gamma,
            re_thetat,
            y,
            omega_mag,
            ..
        } = self;

        let uf: &DevBuf<Vec3> = &u.f;
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(f_onset)
                .arg(f_turb)
                .arg(f_length)
                .arg(re_thetac)
                .arg(f_thetat)
                .arg(gamma_eff)
                .arg(re_theta_eq)
                .arg(f3)
                .arg(&gamma.f)
                .arg(&re_thetat.f)
                .arg(k)
                .arg(omega)
                .arg(s)
                .arg(&*omega_mag)
                .arg(&*y)
                .arg(uf)
                .arg(grad_u)
                .arg(&nu)
                .arg(&ce2)
                .arg(&s1)
                .arg(&sweeps)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// §88.6: `F_1 <- max(F_1, F_3)`, stamped between `sstBlending` and
    /// `sstBlendCoeffs`.
    pub fn stamp_f1(&self, gpu: &Gpu, f1: &mut DevBuf<Scalar>, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let nl = n as Label;
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

    /// §88.6: `P_k <- gamma_eff P_k` and
    /// `D_k <- min(max(gamma_eff, 0.1), 1) D_k`, stamped over what
    /// `sstKSources` has just written.
    pub fn stamp_k_sources(
        &self,
        gpu: &Gpu,
        g_lim: &mut DevBuf<Scalar>,
        sp: &mut DevBuf<Scalar>,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let nl = n as Label;
        let f = self.kern.stamp_k_sources.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(g_lim)
                .arg(sp)
                .arg(&self.gamma_eff)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// §88.5: assemble and solve `gamma`, then `Re_theta~`.
    ///
    /// Both run through the background model's own [`RasCore`] - the same
    /// matrix, the same workspace, the same `ddt + div - laplacian` - because
    /// they are the same equation with different sources, which is exactly
    /// what `RasCore` is for.
    pub fn solve(
        &mut self,
        gpu: &Gpu,
        core: &mut RasCore<'_>,
        flow: &FlowState,
        s: &DevBuf<Scalar>,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        let n = core.mesh.n_cells;
        let c = self.coeffs;

        // ---- gamma -------------------------------------------------------
        // Gamma_eff = nu + nu_t/sigma_f, which at sigma_f = 1 is literally
        // nu + nu_t.
        core.assemble_transport(gpu, flow, &self.gamma, self.ctrl.gamma_conv, 1.0 / c.sigma_f)?;

        {
            let Self {
                kern,
                gamma,
                f_onset,
                f_turb,
                f_length,
                omega_mag,
                ..
            } = self;
            let RasCore { su, sp, susp, .. } = core;
            let nl = n as Label;
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
                        .arg(&*f_length)
                        .arg(s)
                        .arg(&*omega_mag)
                        .arg(&c.ca1)
                        .arg(&c.ca2)
                        .arg(&c.ce1)
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
        let g_perf = core.solve_equation(gpu, &mut self.gamma, self.ctrl.gamma_relax, &sc, false)?;

        // ---- Re_theta~ ---------------------------------------------------
        // sigma_tt multiplies the EFFECTIVE viscosity, molecular part
        // included, so this is the affine entry point and not the scaled-nu_t
        // one. At high Reynolds number the difference is negligible; near a
        // wall it is not, and it is silent.
        core.assemble_transport_affine(
            gpu,
            flow,
            &self.re_thetat,
            self.ctrl.re_thetat_conv,
            c.sigma_tt,
            c.sigma_tt,
        )?;

        {
            let Self {
                kern,
                re_theta_eq,
                f_thetat,
                ..
            } = self;
            let RasCore { su, sp, .. } = core;
            let nl = n as Label;
            let nu = flow.nu;
            let f = kern.re_thetat_sources.clone();
            let uf: &DevBuf<Vec3> = &flow.u.f;
            if n > 0 {
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(su)
                        .arg(sp)
                        .arg(&*re_theta_eq)
                        .arg(&*f_thetat)
                        .arg(uf)
                        .arg(&nu)
                        .arg(&c.ctt)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }
        }

        fvm_su(gpu, &core.fv, &mut core.a, core.mesh, &core.su, 1.0)?;
        fvm_sp(gpu, &core.fv, &mut core.a, core.mesh, &core.sp, 1.0)?;

        let sc = self.ctrl.re_thetat_solver;
        let r_perf =
            core.solve_equation(gpu, &mut self.re_thetat, self.ctrl.re_thetat_relax, &sc, false)?;

        self.bound(gpu, n)?;
        correct_boundary_conditions(gpu, &self.fld, &mut self.gamma, core.mesh)?;
        correct_boundary_conditions(gpu, &self.fld, &mut self.re_thetat, core.mesh)?;

        Ok((g_perf, r_perf))
    }

    /// Write `gamma_eff` and `F_3` directly, for Gate 88-R.
    ///
    /// The gate's claim is about the two stamps in isolation - each is the
    /// identity at its neutral value, on every bit - and the only way to put
    /// a stamp at its neutral value is to say what the factor is. Test-only,
    /// because a run has no business writing either: both are outputs of
    /// [`Self::update_fields`].
    #[cfg(test)]
    pub(crate) fn seed_stamp_inputs(
        &mut self,
        gpu: &Gpu,
        gamma_eff: &[Scalar],
        f3: &[Scalar],
    ) -> Result<()> {
        gpu.write(&mut self.gamma_eff, gamma_eff)?;
        gpu.write(&mut self.f3, f3)
    }
}

#[cfg(test)]
mod tests;
