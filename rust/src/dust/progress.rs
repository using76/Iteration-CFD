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
//! **The flame-age option (SPEC-LIT §160, DST-20).** With
//! [`crate::dust::age::FlameAge::Local`] on ([`ProgressEquation::enable_age`]),
//! every solve of (155.3) is followed by one solve of the transported age
//! concentration alpha with the SAME coupling - the same rho levels, rho_f,
//! face mass flux, `kappa + D_t`, convection entry, ddt scheme, relaxation
//! factor and linear-solver controls, through the same assembly - with the
//! clock source (160.5) and the implicit inheritance (160.6); the closures
//! read the predicted age a* of (160.4) instead of the global clock. The
//! arithmetic lives in [`crate::dust::age`]. OFF adds nothing to a step:
//! no alpha field exists and the closures are the global-clock kernel
//! `dustFsc` with the same arguments.
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
use crate::dust::age::{self, AgeState};
use crate::dust::device::{
    age_predict, age_sources, count_out_of_bounds, progress_sources, AgeSourceInputs,
    ProgressKernels, SourceInputs,
};
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
    /// The flame-age option (SPEC-LIT §160): `None` until
    /// [`Self::enable_age`]; OFF adds nothing to a step.
    age: Option<AgeState>,
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
            age: None,
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

    /// The `S_U` / `S_P'` pair of (155.3) as [`Self::solve`] last assembled
    /// it, `[n_cells]` each: the net rate the solve leaves is
    /// `S_U - S_P' c~`, the `wdot_c` (152.12)'s heat release burns with
    /// (SPEC-LIT §155.2, (154.11)).
    pub fn source_terms(&self) -> (&DevBuf<Scalar>, &DevBuf<Scalar>) {
        (&self.su, &self.sp)
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
        self.old_gradient(gpu)?;
        // The age's levels move once per step, after this (SPEC-LIT §160.5);
        // `begin_age_step` consumes the flag.
        if let Some(a) = &mut self.age {
            a.pending = true;
        }
        Ok(())
    }

    /// Switches the local flame-age clock (160.1) on: the alpha field
    /// `"flameAge"` with c~'s patch kinds at zero values, the a*/S_alpha
    /// buffers, and the clock's origin t_0 with its floor c_a. Refused by
    /// name: a second call, a c_a that is not positive and finite, a
    /// t_origin that is not finite, and a c~ field carrying any
    /// `mean_rules`, `mapped` or `turbulent` boundary rule - the age field
    /// carries none of them (SPEC-LIT §160.5).
    pub fn enable_age(&mut self, gpu: &Gpu, t_origin: f64, c_a: f64) -> Result<()> {
        const W: &str = "dust::age::enable_age";
        if self.age.is_some() {
            return Err(Error::Refused(format!(
                "{W}: the flame-age clock is already on; one alpha field per equation"
            )));
        }
        if !(c_a.is_finite() && c_a > 0.0) {
            return Err(Error::Refused(format!(
                "{W}: c_a = {c_a} must be positive and finite (SPEC-LIT §160.1)"
            )));
        }
        if !t_origin.is_finite() {
            return Err(Error::Refused(format!(
                "{W}: t_origin = {t_origin} must be finite (SPEC-LIT §160.5)"
            )));
        }
        if !(self.c.mean_rules.is_empty()
            && self.c.mapped.is_empty()
            && self.c.turbulent.is_empty())
        {
            return Err(Error::Refused(format!(
                "{W}: the c~ field carries a mean/mapped/turbulent boundary rule; the age \
                 field carries none (SPEC-LIT §160.5)"
            )));
        }
        let m = self.core.mesh;
        let mut alpha = GpuScalarField::zeros(gpu, m, "flameAge")?;
        // The patch kinds and fractions are c~'s; the values stay zero, so
        // a fixedValue c~ patch gives alpha 0: inflow enters with age 0.
        gpu.write(&mut alpha.bc_kind, &gpu.download(&self.c.bc_kind)?)?;
        gpu.write(&mut alpha.fr, &gpu.download(&self.c.fr)?)?;
        field_ops::correct_boundary_conditions(gpu, &self.core.fld, &mut alpha, m)?;
        let n = m.n_cells.max(1);
        self.age = Some(AgeState {
            alpha,
            closure_age: gpu.zeros(n)?,
            sua: gpu.zeros(n)?,
            spa: gpu.zeros(n)?,
            c_a,
            t_origin,
            t_prev: None,
            delta: 0.0,
            delta_prev: 0.0,
            pending: false,
            last: None,
        });
        Ok(())
    }

    /// Moves the age's levels and forms the step's predicted level and the
    /// closures' age a* of (160.4), once per step after [`Self::begin_step`]
    /// and before the first outer solve. `Ok(())` and nothing with the age
    /// off or nothing pending.
    pub fn begin_age_step(&mut self, gpu: &Gpu, t_new: f64) -> Result<()> {
        let m = self.core.mesh;
        let Some(a) = &mut self.age else { return Ok(()) };
        if !a.pending {
            return Ok(());
        }
        let t_old = a.t_prev.unwrap_or(t_new - f64::from(self.core.ddt.state.dt));
        a.delta_prev = a.delta;
        a.delta = age::clock_increment(t_old, t_new, a.t_origin);
        field_ops::store_old_time(gpu, &self.core.fld, &mut a.alpha)?;
        age_predict(
            gpu,
            &self.pk,
            &mut a.alpha.f,
            &mut a.closure_age,
            &a.alpha.f0,
            &self.c.f0,
            a.delta,
            a.c_a,
            m.n_cells,
        )?;
        field_ops::correct_boundary_conditions(gpu, &self.core.fld, &mut a.alpha, m)?;
        a.t_prev = Some(t_new);
        a.pending = false;
        Ok(())
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

        // The flame-age option (SPEC-LIT §160.5): every solve of (155.3) is
        // followed by one solve of alpha with the SAME coupling - the same
        // rho levels, rho_f, face mass flux, kappa + D_t (the diffusivity
        // is already in place), convection entry, ddt scheme, relaxation
        // factor and linear-solver controls, through the same assembly -
        // with the clock source (160.5) and the implicit inheritance
        // (160.6).
        if let Some(a) = &mut self.age {
            let coeffs = self.core.ddt.state.coeffs(self.core.ddt.scheme)?;
            let (w0, w00) = age::clock_weights(coeffs, a.delta, a.delta_prev);
            age_sources(
                gpu,
                &self.pk,
                &mut a.sua,
                &mut a.spa,
                &AgeSourceInputs {
                    su: &self.su,
                    sp: &self.sp,
                    c: &self.c.f,
                    rho0: &cp.rho.f0,
                    c0: &self.c.f0,
                    rho00: &cp.rho.f00,
                    c00: &self.c.f00,
                    w0,
                    w00,
                    c_a: a.c_a,
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
            self.core
                .assemble_transport_mass_weighted(gpu, &flow, &a.alpha, self.conv, 1.0, &mass)?;
            {
                let RasCore { fv, a: mtrx, .. } = &mut self.core;
                crate::fv::fvm_su(gpu, fv, mtrx, m, &a.sua, 1.0)?;
                crate::fv::fvm_sp(gpu, fv, mtrx, m, &a.spa, 1.0)?;
            }
            let perf = self
                .core
                .solve_equation_with(gpu, &mut a.alpha, alpha, &sc, false, false)?;
            field_ops::correct_boundary_conditions(gpu, &self.core.fld, &mut a.alpha, m)?;
            a.last = Some(perf);
        }
        Ok(ProgressSolve { perf, out_of_bounds })
    }

    /// The flame-age option is on (SPEC-LIT §160.5).
    pub fn age_enabled(&self) -> bool {
        self.age.is_some()
    }

    /// The alpha field, when the age is on.
    pub fn age_field(&self) -> Option<&GpuScalarField> {
        self.age.as_ref().map(|a| &a.alpha)
    }

    /// The alpha field, mutable, when the age is on.
    pub fn age_field_mut(&mut self) -> Option<&mut GpuScalarField> {
        self.age.as_mut().map(|a| &mut a.alpha)
    }

    /// The closures' age a* of (160.4), when the age is on: what
    /// [`crate::dust::device::fsc_aged`] reads instead of the global clock.
    pub fn closure_age(&self) -> Option<&DevBuf<Scalar>> {
        self.age.as_ref().map(|a| &a.closure_age)
    }

    /// The (160.5)-(160.6) source pair as [`Self::solve`] last assembled
    /// it, when the age is on - `(sua, spa)`, the alpha twins of
    /// [`Self::source_terms`].
    pub fn age_source_terms(&self) -> Option<(&DevBuf<Scalar>, &DevBuf<Scalar>)> {
        self.age.as_ref().map(|a| (&a.sua, &a.spa))
    }

    /// The last alpha solve's report, when the age is on.
    pub fn age_last_solve(&self) -> Option<SolverPerformance> {
        self.age.as_ref().and_then(|a| a.last)
    }

    /// Writes c~'s cells and corrects its boundary values - the tube's late
    /// kernel (SPEC-LIT §160.5's `run_aged`) seeds a fresh ignition with it.
    pub fn overwrite_field(&mut self, gpu: &Gpu, c: &[Scalar]) -> Result<()> {
        let m = self.core.mesh;
        gpu.write(&mut self.c.f, c)?;
        field_ops::correct_boundary_conditions(gpu, &self.core.fld, &mut self.c, m)
    }
}

#[cfg(test)]
#[path = "tests_progress.rs"]
mod tests;
