// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Wilcox k-omega - SPEC-LIT §6.2.
//!
//! Written from:
//!   Wilcox, *Turbulence Modeling for CFD*, DCW Industries - the 1988 form,
//!     and §5.4 for the Favre-averaged dilatation term in the `k` equation
//!   Launder & Spalding, *Comput. Methods Appl. Mech. Eng.* 3 (1974) 269-289
//!     - the equilibrium wall treatment, which is shared
//!   Patankar, *Numerical Heat Transfer and Fluid Flow* (1980), §4.2
//!   ofgpu `SPEC-LIT.md` §6, §6.2 and §6.4. The bounding is marked *DESIGN*
//!     there and is ours; so is the wall treatment.
//! No GPL-licensed source was consulted.
//!
//! ```text
//! nu_t = k/omega
//!
//! Dk/Dt      = ∇·((nu + alpha_k·nu_t)∇k)      + G - beta*·k·omega
//! Domega/Dt  = ∇·((nu + alpha_w·nu_t)∇omega)  + gamma·(omega/k)·G - beta·omega²
//! ```
//!
//! # The coefficients, and the `gamma` question SPEC-LIT flags
//!
//! SPEC-LIT §6.2 lists `beta* = 0.09`, `beta = 0.072`, `gamma = 5/9`,
//! `alpha_k = alpha_w = 0.5`, and notes that "`gamma = 5/9` in Wilcox's
//! original; some codes carry 0.52. Verify against the edition in use and
//! record which was chosen."
//!
//! **We use `gamma = 5/9`**, the value SPEC-LIT gives, spelled in the source
//! as `5.0/9.0` so that it is the exact rational and not a transcribed
//! decimal. The reason is that 0.52 is not a k-omega number at all: it is the
//! rounded inner-layer coefficient of the k-omega **SST** model, where
//! `gamma_1 = beta_1/beta* - sigma_w1·kappa²/sqrt(beta*)` evaluates to about
//! 0.5532 with SST's own `beta_1 = 0.075` and `sigma_w1 = 0.5` - a different
//! model with a different `beta`. Carrying it here would silently mix the two
//! closures. SPEC-LIT §6.3 lists `gamma_1 = 5/9` for SST separately; that is
//! SST's business, not this file's. A case dictionary can still override
//! `gamma`, and `KOmegaCoeffs` is the single place the value lives.
//!
//! `beta = 0.072` is likewise SPEC-LIT's number and is what this file
//! defaults to. It is worth recording that Wilcox's 1988 paper carries
//! `beta = 3/40 = 0.075`; 0.072 is the value of the later editions of the
//! book, which is the source SPEC-LIT §6.2 names. The two differ by 4 %, and
//! the decay exponent `beta*/beta` moves from 1.20 to 1.25 with them - see
//! [`KOmegaCoeffs::decay_exponent`], which is measured in the tests.
//!
//! Note that `beta` here and `beta_1` in [`WallFunctionCoeffs`] are different
//! constants that happen to share a letter: `beta_1 = 0.075` is the
//! coefficient of Wilcox's viscous-sublayer limit `omega = 6 nu/(beta_1 y²)`
//! and belongs to the wall function, not to the transport equation.
//!
//! # Order of work in one `correct`
//!
//! Identical to `k_epsilon.rs`, with `omega` in place of `epsilon`: the two
//! models differ only in their sources and in the near-wall relation for the
//! constrained variable.

use crate::device::Gpu;
use crate::error::Result;
use crate::field::GpuScalarField;
use crate::field_ops::{advance_time_levels, correct_boundary_conditions};
use crate::fv::{fvm_sp, fvm_su, fvm_susp};
use crate::mesh::{GpuMesh, HostMesh};
use crate::solver::SolverPerformance;
use crate::turbulence::{
    add_buoyancy_to_k, add_buoyancy_to_omega, bound_k, bound_omega, k_omega_k_sources,
    nut_boundary, nut_k_omega, omega_sources, BuoyancyProduction, FlowState,
    RasCore, TurbulenceControls,
};
use crate::wallfunctions::WallFunctionCoeffs;
use crate::Scalar;

/// The five coefficients of SPEC-LIT §6.2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KOmegaCoeffs {
    /// `beta*`, the coefficient of the `k` destruction term.
    pub beta_star: Scalar,
    /// `beta`, the coefficient of the `omega` destruction term.
    pub beta: Scalar,
    /// `gamma`. See the module header for why this is `5/9` and not `0.52`.
    pub gamma: Scalar,
    /// `alpha_k`, the reciprocal turbulent Prandtl number of `k`.
    pub alpha_k: Scalar,
    /// `alpha_omega`, the same for `omega`.
    pub alpha_omega: Scalar,
}

impl Default for KOmegaCoeffs {
    fn default() -> Self {
        Self {
            beta_star: 0.09,
            beta: 0.072,
            // The exact rational, not a transcribed 0.5556 and not SST's 0.52.
            gamma: 5.0 / 9.0,
            alpha_k: 0.5,
            alpha_omega: 0.5,
        }
    }
}

impl KOmegaCoeffs {
    /// The exponent of the decay law `k ~ t^-n` these coefficients imply for
    /// homogeneous isotropic turbulence.
    ///
    /// With no production and no transport the model is
    /// `dk/dt = -beta* k omega`, `domega/dt = -beta omega²`, whose solution is
    /// `omega = omega_0/(1 + beta omega_0 t)` and
    /// `k = k_0 (1 + beta omega_0 t)^{-beta*/beta}`. So `n = beta*/beta`, and
    /// unlike the k-epsilon case it does not depend on the initial state at
    /// all. `tests::decaying_isotropic_turbulence_follows_beta_star_over_beta`
    /// measures it.
    pub fn decay_exponent(&self) -> Scalar {
        self.beta_star / self.beta
    }
}

/// Wilcox k-omega, resident on the device.
pub struct KOmega<'m> {
    core: RasCore<'m>,
    coeffs: KOmegaCoeffs,
    k: GpuScalarField,
    omega: GpuScalarField,
}

impl<'m> KOmega<'m> {
    /// `wall_faces` carries TWO flags per flattened boundary face, and they
    /// are not the same set: which cells `omega` pins to the near-wall
    /// relation, from `omega`'s own patch types, and which faces `nu_t` gets a
    /// wall value on, from `nut`'s. SPEC-LIT 15.5 - deriving either from the
    /// other is a silent physics substitution in one direction or the other.
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        mesh: &'m GpuMesh,
        coeffs: KOmegaCoeffs,
        ctrl: TurbulenceControls,
        wall: WallFunctionCoeffs,
        wall_faces: &crate::field_setup::WallFaces,
        roughness: &crate::field_setup::NutRoughness,
    ) -> Result<Self> {
        Ok(Self {
            core: RasCore::new(gpu, hm, mesh, ctrl, wall, wall_faces, roughness)?,
            coeffs,
            k: GpuScalarField::zeros(gpu, mesh, "k")?,
            omega: GpuScalarField::zeros(gpu, mesh, "omega")?,
        })
    }

    // ---- accessors --------------------------------------------------------

    pub fn k(&self) -> &GpuScalarField {
        &self.k
    }
    pub fn k_mut(&mut self) -> &mut GpuScalarField {
        &mut self.k
    }
    pub fn omega(&self) -> &GpuScalarField {
        &self.omega
    }
    pub fn omega_mut(&mut self) -> &mut GpuScalarField {
        &mut self.omega
    }
    pub fn nut(&self) -> &GpuScalarField {
        &self.core.nut
    }
    /// `nu_t = 0` everywhere and nothing to put it back - what
    /// `RAS { turbulence off; }` and `simulationType laminar;` ask for.
    /// See [`crate::turbulence::RasCore::freeze_nut`].
    pub fn freeze_nut(&mut self, gpu: &Gpu) -> Result<()> {
        self.core.freeze_nut(gpu)
    }

    pub fn nut_mut(&mut self) -> &mut GpuScalarField {
        &mut self.core.nut
    }
    pub fn coeffs(&self) -> &KOmegaCoeffs {
        &self.coeffs
    }
    pub fn core(&self) -> &RasCore<'m> {
        &self.core
    }
    pub fn core_mut(&mut self) -> &mut RasCore<'m> {
        &mut self.core
    }

    /// Switch the buoyancy production `G_b` on - SPEC-LIT §17.
    ///
    /// See [`crate::models::KEpsilon::set_buoyancy`]. The `omega` equation
    /// takes it by the same route the shear production `G` takes:
    /// `+ (gamma/nu_t) G_b`.
    pub fn set_buoyancy(&mut self, b: BuoyancyProduction) -> Result<()> {
        b.validate()?;
        self.core.buoyancy = Some(b);
        Ok(())
    }

    /// The buoyancy production settings, if any.
    pub fn buoyancy(&self) -> Option<BuoyancyProduction> {
        self.core.buoyancy
    }

    // ---- set-up -----------------------------------------------------------

    /// Bound the initial fields, evaluate their boundaries, and build the
    /// first `nu_t`. Call once, after the initial fields are uploaded.
    pub fn initialise(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        let n = self.core.mesh.n_cells;
        let ctrl = self.core.ctrl;
        let nut_max = self.core.nut_max(flow.nu);

        bound_k(gpu, &self.core.turb, &mut self.k.f, ctrl.k_min, n)?;
        bound_omega(
            gpu,
            &self.core.turb,
            &mut self.omega.f,
            &self.k.f,
            nut_max,
            ctrl.omega_min,
            n,
        )?;

        correct_boundary_conditions(gpu, &self.core.fld, &mut self.k, self.core.mesh)?;
        correct_boundary_conditions(gpu, &self.core.fld, &mut self.omega, self.core.mesh)?;

        self.correct_nut(gpu, flow)?;
        self.core.store_k_prev(gpu, &self.k.f)?;

        Ok(())
    }

    // ---- one outer iteration ---------------------------------------------

    /// Solve `omega`, then `k`, then update `nu_t`.
    ///
    /// Returns `(omega, k)` performance in that order, matching
    /// [`crate::models::KEpsilon::correct`]'s `(epsilon, k)` so that a driver
    /// prints the same two columns whichever model it runs.
    pub fn correct(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        self.correct_buoyant(gpu, flow, None)
    }

    /// [`KOmega::correct`] with the temperature the buoyancy production is
    /// built from - SPEC-LIT §17.
    pub fn correct_buoyant(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        t: Option<&GpuScalarField>,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        let n = self.core.mesh.n_cells;
        let ctrl = self.core.ctrl;
        let wall = self.core.wall;
        let c = self.coeffs;
        let nu = flow.nu;
        let nut_max = self.core.nut_max(nu);

        self.core.store_k_prev(gpu, &self.k.f)?;
        // psi^{n-2} <- psi^{n-1} <- psi, in that order (SPEC-LIT 13.3). One
        // `correct` is one time step for every driver that calls it, so this
        // is the once-per-step rotation BDF2 needs; a driver running several
        // outer correctors per step must lift it out, and the note above says
        // what that costs.
        advance_time_levels(gpu, &self.core.fld, &mut self.k)?;
        advance_time_levels(gpu, &self.core.fld, &mut self.omega)?;
        self.core.ddt.advance(ctrl.delta_t);

        self.core.update_flow_derived(gpu, flow)?;

        // G_b = (nu_t/Pr_t) g.grad(T)/T and its C_3 (SPEC-LIT 17), from the
        // same PREVIOUS nu_t the shear production G uses.
        let buoyant = match t {
            Some(tf) => self.core.update_buoyancy_production(gpu, tf, flow.u)?,
            None => false,
        };

        // Wall functions: nu_t on the wall faces from the current k, then
        // omega and G in the wall-adjacent cells.
        self.core.wd.update_nut(
            gpu,
            &mut self.core.nut.bf,
            &self.k.f,
            flow.u,
            self.core.mesh,
            &wall,
            nu,
            ctrl.k_min,
        )?;
        self.core.wd.update_omega(
            gpu,
            &mut self.omega.f,
            &mut self.core.g,
            &self.k.f,
            flow.u,
            &self.core.nut.bf,
            self.core.mesh,
            &wall,
            nu,
            ctrl.k_min,
        )?;

        // ---- omega -------------------------------------------------------
        self.core
            .assemble_transport(gpu, flow, &self.omega, ctrl.eps_conv(), c.alpha_omega)?;

        omega_sources(
            gpu,
            &self.core.turb,
            &mut self.core.su,
            &mut self.core.sp,
            &self.core.g,
            &self.k.f,
            &self.omega.f,
            c.gamma,
            c.beta,
            ctrl.k_min,
            n,
        )?;

        // + (gamma/nu_t) G_b, split by sign, ACCUMULATED into what
        // `omega_sources` just wrote (SPEC-LIT 17). Unstable branch only
        // unless the case asked for both, matching the k-epsilon model.
        if buoyant {
            let stable = self
                .core
                .buoyancy
                .map(|b| b.epsilon_stable_branch)
                .unwrap_or(false);
            // nu_t below this is treated as laminar: there is no eddy
            // transport of buoyancy there either, so the term is zero rather
            // than a division by one.
            let nut_min = 1e-30 as Scalar;
            let RasCore { turb, su, sp, gb, nut, .. } = &mut self.core;
            add_buoyancy_to_omega(
                gpu,
                turb,
                su,
                sp,
                gb,
                &nut.f,
                &self.omega.f,
                c.gamma,
                nut_min,
                stable,
                n,
            )?;
        }

        fvm_su(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.su, 1.0)?;
        fvm_sp(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.sp, 1.0)?;

        let sc = ctrl.epsilon_solver;
        let w_perf = self
            .core
            .solve_equation(gpu, &mut self.omega, ctrl.eps_relax, &sc, true)?;

        bound_omega(
            gpu,
            &self.core.turb,
            &mut self.omega.f,
            &self.k.f,
            nut_max,
            ctrl.omega_min,
            n,
        )?;
        correct_boundary_conditions(gpu, &self.core.fld, &mut self.omega, self.core.mesh)?;

        // ---- k -----------------------------------------------------------
        self.core.assemble_transport(gpu, flow, &self.k, ctrl.k_conv(), c.alpha_k)?;

        k_omega_k_sources(
            gpu,
            &self.core.turb,
            &mut self.core.sp,
            &mut self.core.susp,
            &self.omega.f,
            &self.core.div_u,
            c.beta_star,
            n,
        )?;

        fvm_su(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.g, 1.0)?;

        // + G_b, both signs (SPEC-LIT 17).
        if buoyant {
            {
                let RasCore { turb, su, sp, gb, .. } = &mut self.core;
                add_buoyancy_to_k(gpu, turb, su, sp, gb, &self.k.f, ctrl.k_min, n)?;
            }
            fvm_su(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.su, 1.0)?;
        }

        fvm_sp(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.sp, 1.0)?;
        fvm_susp(
            gpu,
            &self.core.fv,
            &mut self.core.a,
            self.core.mesh,
            &self.core.susp,
            &self.k.f,
            1.0,
        )?;

        let sc = ctrl.k_solver;
        let k_perf = self
            .core
            .solve_equation(gpu, &mut self.k, ctrl.k_relax, &sc, false)?;

        bound_k(gpu, &self.core.turb, &mut self.k.f, ctrl.k_min, n)?;
        correct_boundary_conditions(gpu, &self.core.fld, &mut self.k, self.core.mesh)?;

        self.correct_nut(gpu, flow)?;

        Ok((w_perf, k_perf))
    }

    /// `nu_t = k/omega`, its boundary values, and the wall-function override.
    pub fn correct_nut(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        let n = self.core.mesh.n_cells;
        let ctrl = self.core.ctrl;
        let wall = self.core.wall;
        let nut_max = self.core.nut_max(flow.nu);

        nut_k_omega(
            gpu,
            &self.core.turb,
            &mut self.core.nut.f,
            &self.k.f,
            &self.omega.f,
            nut_max,
            n,
        )?;

        correct_boundary_conditions(gpu, &self.core.fld, &mut self.core.nut, self.core.mesh)?;
        nut_boundary(gpu, &self.core.turb, &mut self.core.nut, self.core.mesh)?;
        self.core.wd.update_nut(
            gpu,
            &mut self.core.nut.bf,
            &self.k.f,
            flow.u,
            self.core.mesh,
            &wall,
            flow.nu,
            ctrl.k_min,
        )?;

        Ok(())
    }

    /// `max|Δk|/max|k|` since the last call to `correct`.
    pub fn convergence_measure(&mut self, gpu: &Gpu) -> Result<Scalar> {
        self.core.convergence_measure(gpu, &self.k.f)
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{GpuSurfaceScalarField, GpuVectorField};
    use crate::Vec3;

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    fn quiet_box() -> HostMesh {
        let (mut m, points, faces) =
            crate::mesh::topology::tests::box_mesh([4, 4, 4], Vec3::new(0.25, 0.25, 0.25));
        m.compute_geometry(&points, &faces).expect("geometry");
        m.build_cell_face_maps();
        m
    }

    fn decay_controls(dt: Scalar) -> TurbulenceControls {
        let mut ctrl = TurbulenceControls {
            steady: false,
            delta_t: dt,
            k_relax: 1.0,
            eps_relax: 1.0,
            nut_max_coeff: 1e8,
            ..Default::default()
        };
        ctrl.k_solver.tolerance = 1e-14;
        ctrl.k_solver.rel_tol = 0.0;
        ctrl.k_solver.report_residuals = false;
        ctrl.epsilon_solver = ctrl.k_solver;
        ctrl
    }

    /// `gamma = 5/9`, not `0.52`, and the reasoning is in the module header.
    /// Pinned as a test because it is the one coefficient SPEC-LIT §6.2
    /// explicitly asks the implementer to record a decision about.
    #[test]
    fn gamma_is_five_ninths() {
        let c = KOmegaCoeffs::default();
        assert!((c.gamma - 5.0 / 9.0).abs() < 1e-15);
        assert!(
            (c.gamma - 0.52).abs() > 0.03,
            "gamma has drifted towards the SST inner-layer value"
        );
        assert!((c.beta_star - 0.09).abs() < 1e-15);
        assert!((c.beta - 0.072).abs() < 1e-15);
        assert!((c.alpha_k - 0.5).abs() < 1e-15);
        assert!((c.alpha_omega - 0.5).abs() < 1e-15);
    }

    /// `n = beta*/beta`, straight from the coefficients.
    #[test]
    fn the_decay_exponent_is_beta_star_over_beta() {
        let c = KOmegaCoeffs::default();
        assert!((c.decay_exponent() - 0.09 / 0.072).abs() < 1e-13);
        assert!((c.decay_exponent() - 1.25).abs() < 1e-13);
    }

    /// Decaying homogeneous isotropic turbulence for the omega form.
    ///
    /// With `U = 0` the model reduces to `dk/dt = -beta* k omega`,
    /// `domega/dt = -beta omega²`, whose exact solution is
    ///
    /// ```text
    /// omega = omega_0/(1 + beta omega_0 t)
    /// k     = k_0 (1 + beta omega_0 t)^{-beta*/beta}
    /// ```
    ///
    /// Both fields are checked against it, and the exponent is fitted from the
    /// numerical solution rather than assumed. Repeated with a second `beta`
    /// so that the fit is measuring the coefficient.
    #[test]
    fn decaying_isotropic_turbulence_follows_beta_star_over_beta() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let hm = quiet_box();
        let mesh = GpuMesh::upload(&gpu, &hm)?;
        let n_cells = hm.n_cells;

        let u = GpuVectorField::zeros(&gpu, &mesh, "U")?;
        let phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi")?;
        let flow = FlowState::new(&u, &phi, 1e-3);
        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let no_roughness = crate::field_setup::NutRoughness::none(hm.n_boundary_faces);

        for beta in [0.072 as Scalar, 0.09] {
            let coeffs = KOmegaCoeffs { beta, ..Default::default() };
            let n_exp = coeffs.decay_exponent();

            let k0: Scalar = 1.0;
            let w0: Scalar = 2.0;

            let t_end: Scalar = 40.0;
            let steps = 2000;
            let dt = t_end / steps as Scalar;

            let mut model = KOmega::new(
                &gpu,
                &hm,
                &mesh,
                coeffs,
                decay_controls(dt),
                WallFunctionCoeffs::default(),
                &no_walls,
                &no_roughness,
            )?;

            gpu.write(&mut model.k_mut().f, &vec![k0; n_cells])?;
            gpu.write(&mut model.omega_mut().f, &vec![w0; n_cells])?;
            model.initialise(&gpu, &flow)?;

            for _ in 0..steps / 2 {
                model.correct(&gpu, &flow)?;
            }
            gpu.sync()?;
            let k_half = gpu.download(&model.k().f)?[0];
            let t_half = dt * (steps / 2) as Scalar;

            for _ in steps / 2..steps {
                model.correct(&gpu, &flow)?;
            }
            gpu.sync()?;

            let k_num = gpu.download(&model.k().f)?;
            let w_num = gpu.download(&model.omega().f)?;
            let k_end = k_num[0];

            for (i, &v) in k_num.iter().enumerate() {
                assert!(
                    (v - k_end).abs() <= 1e-12 * k_end,
                    "beta {beta}: cell {i} holds k = {v}, cell 0 holds {k_end}"
                );
            }

            // omega first: it is the equation with no coupling at all.
            let w_exact = w0 / (1.0 + beta * w0 * t_end);
            let w_rel = (w_num[0] - w_exact).abs() / w_exact;
            assert!(
                w_rel < 0.01,
                "beta {beta}: omega({t_end}) = {}, analytic {w_exact}, \
                 relative error {w_rel}",
                w_num[0]
            );

            let k_exact = k0 * (1.0 + beta * w0 * t_end).powf(-n_exp);
            let k_rel = (k_end - k_exact).abs() / k_exact;
            assert!(
                k_rel < 0.02,
                "beta {beta}: k({t_end}) = {k_end}, analytic {k_exact}, \
                 relative error {k_rel}"
            );

            // The exponent, fitted against the exact omega history rather
            // than assumed.
            let s_half = 1.0 + beta * w0 * t_half;
            let s_end = 1.0 + beta * w0 * t_end;
            let n_fit = (k_half / k_end).ln() / (s_end / s_half).ln();
            assert!(
                (n_fit - n_exp).abs() < 0.02 * n_exp,
                "beta {beta}: fitted decay exponent {n_fit}, model implies {n_exp}"
            );
        }

        Ok(())
    }

    /// `nu_t = k/omega`, the whole of Wilcox's eddy-viscosity relation.
    #[test]
    fn nut_is_k_over_omega() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let hm = quiet_box();
        let mesh = GpuMesh::upload(&gpu, &hm)?;

        let u = GpuVectorField::zeros(&gpu, &mesh, "U")?;
        let phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi")?;
        let flow = FlowState::new(&u, &phi, 1e-3);
        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let no_roughness = crate::field_setup::NutRoughness::none(hm.n_boundary_faces);

        let mut model = KOmega::new(
            &gpu,
            &hm,
            &mesh,
            KOmegaCoeffs::default(),
            decay_controls(1e-3),
            WallFunctionCoeffs::default(),
            &no_walls,
            &no_roughness,
        )?;

        let k0: Scalar = 0.44;
        let w0: Scalar = 7.5;
        gpu.write(&mut model.k_mut().f, &vec![k0; hm.n_cells])?;
        gpu.write(&mut model.omega_mut().f, &vec![w0; hm.n_cells])?;
        model.initialise(&gpu, &flow)?;
        gpu.sync()?;

        let nut = gpu.download(&model.nut().f)?;
        let want = k0 / w0;
        for (i, &v) in nut.iter().enumerate() {
            assert!(
                (v - want).abs() <= 1e-14 * want,
                "cell {i}: nut {v}, expected {want}"
            );
        }

        Ok(())
    }
}
