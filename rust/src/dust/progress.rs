// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The progress-variable equation of the dust flame - SPEC-LIT §155
//! (docs/17 DST-04):
//!
//! ```text
//! d(rho c~)/dt + div(rho u c~) = div[rho (kappa + D_t) grad c~] + rho_u U_t |grad c~| + Q + rho W_ign   (155.3)
//! ```
//!
//! [`ProgressEquation`] is (155.3) on the device as a MASS-WEIGHTED
//! transport equation (§155.2): the rho-weighted ddt of §3, the limited
//! convection of §7 on the face mass flux, the Laplacian with
//! `Gamma = rho_f (kappa + D_t)`, all through
//! [`crate::turbulence::RasCore::assemble_transport_mass_weighted`] - the
//! assembly `ScalarTransport` (§86) uses - with the diffusivity `kappa + D_t`
//! in the slot that equation gives nu_t (r_sigma = 1, nu = 0). The
//! propagation term `rho_u U_t |grad c~|` is explicit, `|grad c~|` the cell
//! gradient of §12 at the old time level, taken once per step by
//! [`ProgressEquation::begin_step`] and held over the step's outer
//! iterations. Q enters by the implicit split
//! (154.11) and the ignition source by the same split (155.4), so (1 - c~)
//! is never multiplied on the device. rho_u is the CURRENT unburned density
//! rho_u(p) of (155.10). After each solve every cell with c~ outside [0, 1]
//! by more than the round-off margin [`BOUNDS_TOL`] is counted and
//! reported; nothing is clipped.
//!
//! The host helpers are (155.1), (155.2), (155.10) and the exact planar
//! solution (155.5)-(155.6) the 1-D harness ([`crate::dust::tube1d`]) is
//! held against.
//!
//! **Design (DST-04).** The equation is built on `RasCore` directly rather
//! than on `ScalarTransport`, because the laminar-limit and ignition
//! sources need the implicit coefficient S_P on the diagonal, which
//! `ScalarTransport` has no slot for; the assembly is the same function.
//! The gradient defaults to `cellLimited Gauss linear 1` (Barth-Jespersen,
//! §12.2): with an unlimited gradient the explicit propagation source at
//! the edge of the burned plateau (c~ = 1 beside c~ < 1) lifts that cell
//! above 1 whenever the cell Peclet number `w dx/(kappa + D_t)` exceeds 2,
//! which it does early in a run. The limited gradient is zero at a local
//! maximum and, in 1-D, at most `2 (c_up - c~)/dx` with c_up the
//! burned-side neighbour, so the explicit increment `dt w |grad c~|` is at
//! most `2 Co_f (c_up - c~)` and cannot carry c~ past c_up while
//! Co_f <= 0.5 (155.9). That bound holds for the OLD level's gradient,
//! which is why the gradient is not refreshed by the outer iterations.
//!
//! Written from:
//!   SPEC-LIT §155.1 - (155.1), (155.2); §155.2 - (155.3) and its
//!     discretisation; §155.4 - (155.5), (155.6); §155.7 - (155.10)
//!   SPEC-LIT §154.8 - the implicit split (154.11)
//!   C. Huang, A. N. Lipatnikov & K. Nessvi, J. Loss Prev. Process Ind. 67
//!     (2020) 104237, DOI 10.1016/j.jlp.2020.104237 - eqs. (1)-(2), (8)-(11),
//!     (18)
//!   C. Huang, M. Bloching & A. N. Lipatnikov, J. Loss Prev. Process Ind. 75
//!     (2022) 104707, DOI 10.1016/j.jlp.2021.104707 - eqs. (4), (6)
//!   M. Abramowitz & I. A. Stegun, Handbook of Mathematical Functions (1964)
//!     7.1.26 - the erfc of the reference profile (|error| < 1.5e-7)
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use crate::device::{DevBuf, Gpu};
use crate::dust::device::{count_out_of_bounds, progress_sources, ProgressKernels, SourceInputs};
use crate::dust::fsc::bracket;
use crate::error::{Error, Result};
use crate::field::{GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use crate::field_ops;
use crate::fv::{GradLimit, GradLimiterKind, GradScheme};
use crate::io::case::TurbulenceControls;
use crate::io::schemes::DivEntry;
use crate::mesh::{GpuMesh, HostMesh};
use crate::solver::SolverPerformance;
use crate::turbulence::{FlowState, MassWeighting, RasCore};
use crate::wallfunctions::WallFunctionCoeffs;
use crate::{Scalar, Vec3};

/// The bounds counter's round-off margin (SPEC-LIT §155.2): c~ counts as
/// out of bounds below `-BOUNDS_TOL` or above `1 + BOUNDS_TOL`. The c~
/// solve is iterative, so a cell whose exact answer is 1 comes back up to
/// 17 ulp above it in f64 (measured, DST-04); a margin of none would count
/// the linear solver's round-off, and 64 epsilon of the build counts any
/// excursion of the scheme (the unlimited gradient's was 3e-8) and clips
/// nothing.
pub const BOUNDS_TOL: Scalar = 64.0 * Scalar::EPSILON;

// ==========================================================================
//  Host relations
// ==========================================================================

/// (155.1): the BML density rho = rho_u / (1 + (sigma - 1) c~).
pub fn bml_density(rho_u: f64, sigma: f64, c: f64) -> f64 {
    rho_u / (1.0 + (sigma - 1.0) * c)
}

/// (155.2): the Reynolds mean cbar = rho c~ / rho_b - the kernel
/// `dustReynoldsMean` in the same order.
pub fn reynolds_mean(rho: f64, c: f64, rho_b: f64) -> f64 {
    (rho * c) / rho_b
}

/// (155.10): the current unburned density rho_u0 (p/p0)^{1/gamma_u}.
pub fn unburned_density(rho_u0: f64, p: f64, p0: f64, gamma_u: f64) -> f64 {
    rho_u0 * (p / p0).powf(1.0 / gamma_u)
}

/// (155.10): the current unburned temperature T_u0 (p/p0)^{(gamma_u-1)/gamma_u}.
pub fn unburned_temperature(t_u0: f64, p: f64, p0: f64, gamma_u: f64) -> f64 {
    t_u0 * (p / p0).powf((gamma_u - 1.0) / gamma_u)
}

/// erfc(x) by Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7): the reference
/// profile's function only, never the solver's.
pub fn erfc(x: f64) -> f64 {
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736 + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let tail = poly * (-ax * ax).exp();
    if x >= 0.0 {
        tail
    } else {
        2.0 - tail
    }
}

/// The exact planar solution in frozen turbulence (155.5)-(155.6): the
/// developing D_t (154.2) and U_t (154.3) of one uniform state, from t = 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanarExact {
    /// D_t,inf, m^2/s.
    pub d_inf: f64,
    /// tau_L, s.
    pub tau_l: f64,
    /// U_t,ISP, m/s.
    pub u_t_isp: f64,
}

impl PlanarExact {
    /// `int_0^t D_t = D_inf [t - tau_L (1 - e^{-t/tau_L})]` [derived, §155.4].
    pub fn int_d_t(&self, t: f64) -> f64 {
        self.d_inf * (t + self.tau_l * (-t / self.tau_l).exp_m1())
    }

    /// `int_0^t U_t`, by composite Simpson in s = theta^{1/2} (the
    /// integrand `2 s U_t(s^2)` is smooth at 0, where U_t ~ theta^{1/2}),
    /// the panels doubled until two passes agree to 1e-13 relative.
    pub fn int_u_t(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        let f = |s: f64| 2.0 * s * self.u_t_isp * bracket(s * s / self.tau_l).sqrt();
        let simpson = |n: usize| {
            let b = t.sqrt();
            let h = b / n as f64;
            let mut sum = f(0.0) + f(b);
            for i in 1..n {
                let w = if i % 2 == 1 { 4.0 } else { 2.0 };
                sum += w * f(h * i as f64);
            }
            sum * h / 3.0
        };
        let mut n = 256;
        let mut prev = simpson(n);
        loop {
            n *= 2;
            let next = simpson(n);
            if (next - prev).abs() <= 1e-13 * next.abs() || n >= 1 << 22 {
                return next;
            }
            prev = next;
        }
    }

    /// delta_t(t) = (4 pi int_0^t D_t)^{1/2} (155.5).
    pub fn delta_t(&self, t: f64) -> f64 {
        (4.0 * std::f64::consts::PI * self.int_d_t(t)).sqrt()
    }

    /// cbar(x, t) = 1/2 erfc(xi sqrt(pi)), xi = (x - x_f)/delta_t (155.5),
    /// the flame propagating towards +x from x_f(0).
    pub fn profile(&self, x: f64, x_f0: f64, t: f64) -> f64 {
        let xi = (x - (x_f0 + self.int_u_t(t))) / self.delta_t(t);
        0.5 * erfc(xi * std::f64::consts::PI.sqrt())
    }
}

// ==========================================================================
//  The device equation
// ==========================================================================

/// What one solve of (155.3) reads, all on the device.
pub struct ProgressCoupling<'a> {
    /// The mean density rho: `f` (new level), `f0`, `f00` (old levels).
    pub rho: &'a GpuScalarField,
    /// rho on the faces, the Laplacian's `rho_f`.
    pub rho_face: &'a GpuSurfaceScalarField,
    /// The face MASS flux `rho_f u_f . S_f`, kg/s.
    pub phi_mass: &'a GpuSurfaceScalarField,
    /// The cell velocity (the flow state's; the assembly reads the flux).
    pub u: &'a GpuVectorField,
    /// `kappa + D_t` per cell and on the boundary faces, m^2/s.
    pub diffusivity: &'a GpuScalarField,
    /// U_t per cell, m/s.
    pub u_t: &'a DevBuf<Scalar>,
    /// K of (154.11) per cell, 1/s (0 for TFC or a cell the zero rule hit).
    pub k_rate: &'a DevBuf<Scalar>,
    /// The bracket W of (155.4) per cell, 1/s; `None` with no source.
    pub w_ign: Option<&'a DevBuf<Scalar>>,
    /// rho_u(p) of (155.10), kg/m^3.
    pub rho_u: f64,
}

/// One solve: the linear solver's report and the bounds counter (§155.2).
#[derive(Debug, Clone, Copy)]
pub struct ProgressSolve {
    pub perf: SolverPerformance,
    /// Cells with c~ < -[`BOUNDS_TOL`], c~ > 1 + [`BOUNDS_TOL`] or not a
    /// number after the solve.
    pub out_of_bounds: u64,
}

/// (155.3) for c~ on one mesh, resident on the device.
pub struct ProgressEquation<'m> {
    core: RasCore<'m>,
    c: GpuScalarField,
    conv: DivEntry,
    grad_scheme: GradScheme,
    grad: DevBuf<Vec3>,
    su: DevBuf<Scalar>,
    sp: DevBuf<Scalar>,
    cont_ddt: DevBuf<Scalar>,
    n_out: DevBuf<u64>,
    bounds_tol: Scalar,
    pk: ProgressKernels,
}

impl<'m> ProgressEquation<'m> {
    /// The equation for the field `"c"` with the transient controls `ctrl`:
    /// its `ddt`, `div_scheme`, `bounded_convection`, `k_solver` and
    /// `k_relax` are the equation's. A steady `ctrl` is refused by name:
    /// (155.3) has a time derivative.
    pub fn new(gpu: &Gpu, hm: &HostMesh, m: &'m GpuMesh, ctrl: TurbulenceControls) -> Result<Self> {
        if ctrl.steady {
            return Err(Error::Refused(
                "dust::progress: the progress-variable equation (155.3) is transient; a \
                 steady ddtSchemes entry is refused (SPEC-LIT §155.2)"
                    .into(),
            ));
        }
        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let no_rough = crate::field_setup::NutRoughness::none(hm.n_boundary_faces);
        let core =
            RasCore::new(gpu, hm, m, ctrl, WallFunctionCoeffs::default(), &no_walls, &no_rough)?;
        let n = m.n_cells.max(1);
        Ok(Self {
            conv: core.ctrl.k_conv(),
            core,
            c: GpuScalarField::zeros(gpu, m, "c")?,
            grad_scheme: GradScheme {
                base: crate::fv::GradBase::Gauss,
                limit: GradLimit::Cell(GradLimiterKind::BarthJespersen, 1.0),
            },
            grad: gpu.zeros(n)?,
            su: gpu.zeros(n)?,
            sp: gpu.zeros(n)?,
            cont_ddt: gpu.zeros(n)?,
            n_out: gpu.zeros(1)?,
            bounds_tol: BOUNDS_TOL,
            pk: ProgressKernels::new(gpu)?,
        })
    }

    /// This field's own `div(phi,c)` entry (§11.7).
    pub fn set_convection(&mut self, conv: DivEntry) {
        self.conv = conv;
    }

    /// The gradient `|grad c~|` is taken with (§12); see the module's
    /// design note for the default.
    pub fn set_gradient(&mut self, scheme: GradScheme) {
        self.grad_scheme = scheme;
    }

    /// The bounds counter's margin (§155.2): [`BOUNDS_TOL`] unless the
    /// caller's outer loop converges each step only to a coarser `tol`, in
    /// which case the larger of the two. A margin that is not positive and
    /// finite is refused by name.
    pub fn set_bounds_tolerance(&mut self, tol: f64) -> Result<()> {
        if !(tol.is_finite() && tol > 0.0) {
            return Err(Error::Refused(format!(
                "dust::progress: the bounds margin {tol} must be positive and finite (SPEC-LIT \
                 §155.2)"
            )));
        }
        self.bounds_tol = BOUNDS_TOL.max(tol as Scalar);
        Ok(())
    }

    pub fn field(&self) -> &GpuScalarField {
        &self.c
    }

    pub fn field_mut(&mut self) -> &mut GpuScalarField {
        &mut self.c
    }

    pub fn controls(&self) -> &TurbulenceControls {
        &self.core.ctrl
    }

    /// Evaluates the boundary values and seeds both old levels from the
    /// initial field.
    pub fn initialise(&mut self, gpu: &Gpu) -> Result<()> {
        let m = self.core.mesh;
        field_ops::correct_boundary_conditions(gpu, &self.core.fld, &mut self.c, m)?;
        field_ops::store_old_time(gpu, &self.core.fld, &mut self.c)?;
        field_ops::store_old_time(gpu, &self.core.fld, &mut self.c)?;
        self.old_gradient(gpu)
    }

    /// Starts a time step: the current field becomes the old level, and
    /// `|grad c~|` is taken from it for the whole step (§155.2). Any
    /// number of [`Self::solve`] calls (outer iterations) may follow.
    pub fn begin_step(&mut self, gpu: &Gpu) -> Result<()> {
        field_ops::store_old_time(gpu, &self.core.fld, &mut self.c)?;
        self.old_gradient(gpu)
    }

    /// The gradient of the field as it stands, whose boundary values were
    /// evaluated when it was last set: at [`Self::begin_step`] that is the
    /// old level.
    fn old_gradient(&mut self, gpu: &Gpu) -> Result<()> {
        crate::fv::fvc_grad_scalar_scheme(
            gpu,
            &self.core.fv,
            &mut self.grad,
            &self.c,
            self.core.mesh,
            self.grad_scheme,
        )
    }

    /// One assembly and solve of (155.3) against the old level of the
    /// current step, with the coupling `cp`; returns the solver's report
    /// and the bounds count (read back, so this is not capturable).
    pub fn solve(&mut self, gpu: &Gpu, cp: &ProgressCoupling<'_>) -> Result<ProgressSolve> {
        let m = self.core.mesh;
        let n = m.n_cells;
        if n == 0 {
            return Ok(ProgressSolve { perf: SolverPerformance::default(), out_of_bounds: 0 });
        }
        if !(cp.rho_u.is_finite() && cp.rho_u > 0.0) {
            return Err(Error::Refused(format!(
                "dust::progress: rho_u(p) = {} must be positive and finite (SPEC-LIT §155.7)",
                cp.rho_u
            )));
        }

        // The diffusivity kappa + D_t in the slot nu_t has in the assembly.
        {
            let RasCore { fld, nut, .. } = &mut self.core;
            field_ops::copy_field(gpu, fld, &mut nut.f, &cp.diffusivity.f, n)?;
            field_ops::copy_field(gpu, fld, &mut nut.bf, &cp.diffusivity.bf, m.n_boundary_faces)?;
        }

        // (86.4)'s continuity coefficient of this step's density levels.
        self.core.ddt.rho_continuity(
            gpu,
            &mut self.cont_ddt,
            m,
            &cp.rho.f,
            &cp.rho.f0,
            &cp.rho.f00,
        )?;

        // The S_U / S_P pair, with the old level's |grad c~|.
        progress_sources(
            gpu,
            &self.pk,
            &mut self.su,
            &mut self.sp,
            &SourceInputs {
                grad_c: &self.grad,
                rho: &cp.rho.f,
                u_t: cp.u_t,
                k_rate: cp.k_rate,
                w_ign: cp.w_ign,
                rho_u: cp.rho_u,
            },
            n,
        )?;

        let flow = FlowState { u: cp.u, phi: cp.phi_mass, nu: 0.0 };
        let mass = MassWeighting {
            rho: &cp.rho.f,
            rho0: &cp.rho.f0,
            rho00: &cp.rho.f00,
            rho_face: cp.rho_face,
            cont_ddt: &self.cont_ddt,
        };
        self.core.assemble_transport_mass_weighted(gpu, &flow, &self.c, self.conv, 1.0, &mass)?;
        {
            let RasCore { fv, a, .. } = &mut self.core;
            crate::fv::fvm_su(gpu, fv, a, m, &self.su, 1.0)?;
            crate::fv::fvm_sp(gpu, fv, a, m, &self.sp, 1.0)?;
        }
        let alpha = self.core.ctrl.k_relax;
        let sc = self.core.ctrl.k_solver;
        let perf = self.core.solve_equation_with(gpu, &mut self.c, alpha, &sc, false, false)?;
        field_ops::correct_boundary_conditions(gpu, &self.core.fld, &mut self.c, m)?;

        gpu.write(&mut self.n_out, &[0u64])?;
        count_out_of_bounds(gpu, &self.pk, &mut self.n_out, &self.c.f, self.bounds_tol, n)?;
        let out_of_bounds = gpu.download(&self.n_out)?[0];
        Ok(ProgressSolve { perf, out_of_bounds })
    }
}

#[cfg(test)]
#[path = "tests_progress.rs"]
mod tests;
