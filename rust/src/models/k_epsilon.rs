// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Standard k-epsilon - SPEC-LIT §6.1.
//!
//! Written from:
//!   Launder & Spalding, "The numerical computation of turbulent flows",
//!     *Comput. Methods Appl. Mech. Eng.* 3 (1974) 269-289
//!   Wilcox, *Turbulence Modeling for CFD*, §5.4 - the Favre-averaged
//!     dilatation terms that appear when the flow is not solenoidal
//!   Patankar, *Numerical Heat Transfer and Fluid Flow* (1980), §4.2 - the
//!     linearisation that puts the dissipation on the diagonal
//!   Rodi, *J. Geophys. Res.* 92 (1987) 5305-5328, and Henkes, van der Vlugt
//!     & Hoogendoorn, *Int. J. Heat Mass Transfer* 34 (1991) 377-388 - the
//!     buoyancy production `G_b` and the `C_3` convention
//!   ofgpu `SPEC-LIT.md` §6, §6.1, §6.4 and §17. The bounding of `k` and
//!     `epsilon` is marked *DESIGN* there and is ours; so is the wall
//!     treatment, and so is which branch of `G_b` reaches which equation.
//! No GPL-licensed source was consulted.
//!
//! ```text
//! nu_t = C_mu k²/epsilon
//!
//! Dk/Dt      = ∇·((nu + nu_t/sigma_k)∇k)      + G - epsilon
//! Deps/Dt    = ∇·((nu + nu_t/sigma_eps)∇eps)  + C_1 (eps/k) G - C_2 eps²/k
//!
//! C_mu = 0.09   C_1 = 1.44   C_2 = 1.92   sigma_k = 1.0   sigma_eps = 1.3
//! ```
//!
//! plus the dilatation terms `-(2/3)(∇·u)k` and `-(2/3 C_1 - C_3)(∇·u)eps`,
//! which are identically zero whenever the discrete flux conserves mass and
//! cost one pass each when it does not.
//!
//! # Order of work in one `correct`
//!
//! 1. remember `k` for the convergence measure, and store both old time levels
//! 2. `grad U`, `G` and `∇·u` from the frozen flow
//! 3. wall functions: `nu_t` on the wall faces, then `epsilon` and `G` in the
//!    wall-adjacent cells
//! 4. the `epsilon` equation, with the wall cells constrained
//! 5. the `k` equation
//! 6. `nu_t` from the new `k` and `epsilon`
//!
//! `epsilon` before `k` because the `k` equation's sink is `epsilon/k` and
//! taking it from the equation just solved is one iteration less lag; `nu_t`
//! last because it is the model's output and the momentum equation's input.
//!
//! Nothing in steps 1-6 reads anything back to the host, so a whole call can
//! be captured into a CUDA graph when the linear solver is in its
//! fixed-iteration mode - `src/bin/graph_bench.rs` does exactly that and
//! checks the replayed answer bit for bit.

use crate::device::Gpu;
use crate::error::Result;
use crate::field::GpuScalarField;
use crate::field_ops::{advance_time_levels, correct_boundary_conditions};
use crate::fv::{fvm_sp, fvm_su, fvm_susp};
use crate::mesh::{GpuMesh, HostMesh};
use crate::solver::SolverPerformance;
use crate::turbulence::{
    add_buoyancy_to_epsilon, add_buoyancy_to_k, bound_epsilon, bound_k, epsilon_sources,
    k_sources, nut_boundary, nut_k_epsilon, BuoyancyProduction, FlowState, RasCore,
    TurbulenceControls,
};
use crate::wallfunctions::WallFunctionCoeffs;
use crate::Scalar;

/// The five coefficients of Launder & Spalding (1974), plus `C_3`.
///
/// `C_3` multiplies the dilatation term in the `epsilon` equation and is zero
/// in the incompressible model; it is carried so that a case dictionary can
/// set it without the struct changing shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KEpsilonCoeffs {
    pub cmu: Scalar,
    pub c1: Scalar,
    pub c2: Scalar,
    pub c3: Scalar,
    pub sigmak: Scalar,
    pub sigma_eps: Scalar,
}

impl Default for KEpsilonCoeffs {
    /// The values in Launder & Spalding (1974) and in SPEC-LIT §6.1. They are
    /// facts about a published model, not choices.
    fn default() -> Self {
        Self {
            cmu: 0.09,
            c1: 1.44,
            c2: 1.92,
            c3: 0.0,
            sigmak: 1.0,
            sigma_eps: 1.3,
        }
    }
}

impl KEpsilonCoeffs {
    /// The exponent of the decay law `k ~ t^-n` that these coefficients imply
    /// for homogeneous isotropic turbulence.
    ///
    /// With no production and no transport the model collapses to
    /// `dk/dt = -eps`, `deps/dt = -C_2 eps²/k`; substituting `k = A t^-n`
    /// gives `n(n+1) = C_2 n²`, hence `n = 1/(C_2 - 1)`. This is the sharpest
    /// closed-form statement the standard model makes about anything, which is
    /// why `tests::decaying_isotropic_turbulence_follows_the_model_exponent`
    /// measures it.
    pub fn decay_exponent(&self) -> Scalar {
        1.0 / (self.c2 - 1.0)
    }
}

/// Standard k-epsilon, resident on the device.
pub struct KEpsilon<'m> {
    core: RasCore<'m>,
    coeffs: KEpsilonCoeffs,
    k: GpuScalarField,
    epsilon: GpuScalarField,
}

impl<'m> KEpsilon<'m> {
    /// `wall_faces` carries TWO flags per flattened boundary face, and they
    /// are not the same set: which cells `epsilon` pins to the near-wall
    /// relation, from `epsilon`'s own patch types, and which faces `nu_t` gets
    /// a wall value on, from `nut`'s. SPEC-LIT 15.5 - deriving either from the
    /// other is a silent physics substitution in one direction or the other.
    /// Both come from the FIELDS rather than from the mesh, because a `wall`
    /// patch is entitled to carry a `fixedValue` instead.
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        mesh: &'m GpuMesh,
        coeffs: KEpsilonCoeffs,
        ctrl: TurbulenceControls,
        wall: WallFunctionCoeffs,
        wall_faces: &crate::field_setup::WallFaces,
        roughness: &crate::field_setup::NutRoughness,
    ) -> Result<Self> {
        Ok(Self {
            core: RasCore::new(gpu, hm, mesh, ctrl, wall, wall_faces, roughness)?,
            coeffs,
            k: GpuScalarField::zeros(gpu, mesh, "k")?,
            epsilon: GpuScalarField::zeros(gpu, mesh, "epsilon")?,
        })
    }

    // ---- accessors --------------------------------------------------------

    pub fn k(&self) -> &GpuScalarField {
        &self.k
    }
    pub fn k_mut(&mut self) -> &mut GpuScalarField {
        &mut self.k
    }
    pub fn epsilon(&self) -> &GpuScalarField {
        &self.epsilon
    }
    pub fn epsilon_mut(&mut self) -> &mut GpuScalarField {
        &mut self.epsilon
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
    pub fn coeffs(&self) -> &KEpsilonCoeffs {
        &self.coeffs
    }

    /// `k`, `epsilon` and `nut`, named - the writer seam and the `.mcr`
    /// restart checkpoint's view of this model (SPEC-LIT §30.2's
    /// `CoupledTurbulence::output_fields`). A free function rather than a
    /// trait default because `k`, `epsilon` and `core.nut` are three
    /// disjoint fields of THIS struct; only code with access to them can
    /// destructure a borrow of each at once.
    pub fn named_fields(&self) -> Vec<(&'static str, &GpuScalarField)> {
        vec![("k", &self.k), ("epsilon", &self.epsilon), ("nut", &self.core.nut)]
    }

    /// [`Self::named_fields`], mutable - for `0/` upload and `.mcr` restore.
    pub fn named_fields_mut(&mut self) -> Vec<(&'static str, &mut GpuScalarField)> {
        vec![
            ("k", &mut self.k),
            ("epsilon", &mut self.epsilon),
            ("nut", &mut self.core.nut),
        ]
    }
    pub fn core(&self) -> &RasCore<'m> {
        &self.core
    }
    pub fn core_mut(&mut self) -> &mut RasCore<'m> {
        &mut self.core
    }

    /// Switch the buoyancy production `G_b` on - SPEC-LIT §17.
    ///
    /// Off by default, because a case with no gravity and no temperature has
    /// no such term and would only pay for a gradient it multiplies by zero.
    /// A buoyant driver calls this once at set-up and then passes the
    /// temperature to [`KEpsilon::correct_buoyant`] every iteration.
    ///
    /// Leaving it off on a buoyant run is a leading-order omission, not a
    /// refinement: a 1173 K plume in 293 K air generates most of its
    /// turbulence through exactly this term.
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
    /// first `nu_t`.
    ///
    /// Call once, after the initial `k` and `epsilon` have been uploaded. A
    /// case file is entitled to contain a zero `epsilon` on a patch nobody
    /// thought about, and `nu_t = C_mu k²/0` would then be the first thing the
    /// momentum equation ever saw.
    pub fn initialise(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        let n = self.core.mesh.n_cells;
        let ctrl = self.core.ctrl;
        let nut_max = self.core.nut_max(flow.nu);

        bound_k(gpu, &self.core.turb, &mut self.k.f, ctrl.k_min, n)?;
        bound_epsilon(
            gpu,
            &self.core.turb,
            &mut self.epsilon.f,
            &self.k.f,
            self.coeffs.cmu,
            nut_max,
            ctrl.epsilon_min,
            n,
        )?;

        correct_boundary_conditions(gpu, &self.core.fld, &mut self.k, self.core.mesh)?;
        correct_boundary_conditions(gpu, &self.core.fld, &mut self.epsilon, self.core.mesh)?;

        self.correct_nut(gpu, flow)?;
        self.core.store_k_prev(gpu, &self.k.f)?;

        Ok(())
    }

    // ---- one outer iteration ---------------------------------------------

    /// Solve `epsilon`, then `k`, then update `nu_t`.
    ///
    /// Returns `(epsilon, k)` performance in that order - the order they were
    /// solved in, which is also the order the drivers print them.
    pub fn correct(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        self.correct_buoyant(gpu, flow, None)
    }

    /// [`KEpsilon::correct`] with the temperature the buoyancy production is
    /// built from - SPEC-LIT §17.
    ///
    /// `t` is read, never written, and is ignored unless
    /// [`KEpsilon::set_buoyancy`] has been called with a non-zero gravity.
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

        // 1. the convergence baseline and the old time levels. Both copies
        //    are per OUTER ITERATION, which for `nOuterIterations > 1` makes a
        //    step behave like that many Euler sub-steps; the drivers document
        //    it where it matters.
        self.core.store_k_prev(gpu, &self.k.f)?;
        // psi^{n-2} <- psi^{n-1} <- psi, in that order (SPEC-LIT 13.3). One
        // `correct` is one time step for every driver that calls it, so this
        // is the once-per-step rotation BDF2 needs; a driver running several
        // outer correctors per step must lift it out, and the note above says
        // what that costs.
        advance_time_levels(gpu, &self.core.fld, &mut self.k)?;
        advance_time_levels(gpu, &self.core.fld, &mut self.epsilon)?;
        self.core.ddt.advance(ctrl.delta_t);

        // 2. grad U, G and div u. G uses the PREVIOUS nu_t, which is the
        //    eddy viscosity the momentum equation was solved with.
        self.core.update_flow_derived(gpu, flow)?;

        // 2b. G_b = (nu_t/Pr_t) g.grad(T)/T, and the C_3 that goes with it
        //     (SPEC-LIT 17). Like `G` it uses the PREVIOUS nu_t, which is the
        //     eddy viscosity the momentum equation was solved with, so the
        //     two productions are consistent with each other.
        let buoyant = match t {
            Some(tf) => self.core.update_buoyancy_production(gpu, tf, flow.u)?,
            None => false,
        };

        // 3. wall functions. nu_t on the wall faces first, because the
        //    wall-cell production reads it back.
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
        self.core.wd.update_epsilon(
            gpu,
            &mut self.epsilon.f,
            &mut self.core.g,
            &self.k.f,
            flow.u,
            &self.core.nut.bf,
            self.core.mesh,
            &wall,
            nu,
            ctrl.k_min,
        )?;

        // 4. epsilon.
        self.core
            .assemble_transport(gpu, flow, &self.epsilon, ctrl.eps_conv(), 1.0 / c.sigma_eps)?;

        epsilon_sources(
            gpu,
            &self.core.turb,
            &mut self.core.su,
            &mut self.core.sp,
            &mut self.core.susp,
            &self.core.g,
            &self.k.f,
            &self.epsilon.f,
            &self.core.div_u,
            c.c1,
            c.c2,
            c.c3,
            ctrl.k_min,
            n,
        )?;

        // C_1 (eps/k) C_3 G_b, split by sign, ACCUMULATED into the su/sp the
        // call above just wrote (SPEC-LIT 17). The unstable branch only,
        // unless the case asked for both - see `BuoyancyProduction`.
        if buoyant {
            let stable = self
                .core
                .buoyancy
                .map(|b| b.epsilon_stable_branch)
                .unwrap_or(false);
            let RasCore { turb, su, sp, gb, c3, .. } = &mut self.core;
            add_buoyancy_to_epsilon(
                gpu,
                turb,
                su,
                sp,
                gb,
                c3,
                &self.k.f,
                &self.epsilon.f,
                c.c1,
                ctrl.k_min,
                stable,
                n,
            )?;
        }

        // Su on the right-hand side; Sp and Susp on the left, where Patankar
        // wants them (SPEC-LIT §3.4).
        fvm_su(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.su, 1.0)?;
        fvm_sp(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.sp, 1.0)?;
        fvm_susp(
            gpu,
            &self.core.fv,
            &mut self.core.a,
            self.core.mesh,
            &self.core.susp,
            &self.epsilon.f,
            1.0,
        )?;

        let sc = ctrl.epsilon_solver;
        let eps_perf =
            self.core
                .solve_equation(gpu, &mut self.epsilon, ctrl.eps_relax, &sc, true)?;

        bound_epsilon(
            gpu,
            &self.core.turb,
            &mut self.epsilon.f,
            &self.k.f,
            c.cmu,
            nut_max,
            ctrl.epsilon_min,
            n,
        )?;
        correct_boundary_conditions(gpu, &self.core.fld, &mut self.epsilon, self.core.mesh)?;

        // 5. k. Its source is G itself - no coefficient, no division - so it
        //    goes straight to fvm_su without passing through a kernel.
        self.core
            .assemble_transport(gpu, flow, &self.k, ctrl.k_conv(), 1.0 / c.sigmak)?;

        k_sources(
            gpu,
            &self.core.turb,
            &mut self.core.sp,
            &mut self.core.susp,
            &self.k.f,
            &self.epsilon.f,
            &self.core.div_u,
            ctrl.k_min,
            n,
        )?;

        fvm_su(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.g, 1.0)?;

        // + G_b, BOTH signs (SPEC-LIT 17). The stable branch is a genuine sink
        // of turbulent energy and belongs here whatever the epsilon equation
        // does with it. `add_buoyancy_to_k` WRITES `su` - the shear production
        // G went in on the line above, out of its own array - and accumulates
        // into `sp`, where the dissipation already sits.
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
        // `k` is NOT constrained at the wall: SPEC-LIT §6.4 fixes the
        // dissipation in the wall cell, not the energy, and `k` keeps its
        // zero-gradient wall condition.
        let k_perf = self
            .core
            .solve_equation(gpu, &mut self.k, ctrl.k_relax, &sc, false)?;

        bound_k(gpu, &self.core.turb, &mut self.k.f, ctrl.k_min, n)?;
        correct_boundary_conditions(gpu, &self.core.fld, &mut self.k, self.core.mesh)?;

        // 6. nu_t.
        self.correct_nut(gpu, flow)?;

        Ok((eps_perf, k_perf))
    }

    /// `nu_t = C_mu k²/epsilon`, its boundary values, and the wall-function
    /// override on the wall faces.
    pub fn correct_nut(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        let n = self.core.mesh.n_cells;
        let ctrl = self.core.ctrl;
        let wall = self.core.wall;
        let nut_max = self.core.nut_max(flow.nu);

        nut_k_epsilon(
            gpu,
            &self.core.turb,
            &mut self.core.nut.f,
            &self.k.f,
            &self.epsilon.f,
            self.coeffs.cmu,
            nut_max,
            n,
        )?;

        // Faces with a triple first, then the ones the model owns, then the
        // wall faces - each stage overwriting only what the previous one had
        // no opinion about.
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

    /// `max|Δk|/max|k|` since the last call to `correct`. See
    /// [`RasCore::convergence_measure`]; this is the one host round-trip.
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

    /// A closed box with no flow: `U = 0`, `phi = 0`, no wall functions.
    /// Every cell then evolves by the model's own ODEs and nothing else, and
    /// the mesh is present only because the operators need one.
    fn quiet_box() -> HostMesh {
        let (mut m, points, faces) =
            crate::mesh::topology::tests::box_mesh([4, 4, 4], Vec3::new(0.25, 0.25, 0.25));
        m.compute_geometry(&points, &faces).expect("geometry");
        m.build_cell_face_maps();
        m
    }

    /// Controls for a transient run of pure decay: no relaxation, no
    /// steady-state trick, and a cap so far away that the bounding never
    /// touches the answer.
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

    /// The coefficients' own decay law, which is a statement about the model
    /// and not about this implementation.
    #[test]
    fn the_decay_exponent_follows_from_c2() {
        let c = KEpsilonCoeffs::default();
        assert!((c.decay_exponent() - 1.0 / 0.92).abs() < 1e-13);

        // n(n+1) = C_2 n² is the equation it solves; check it directly rather
        // than checking the algebra against itself.
        let n = c.decay_exponent();
        assert!((n * (n + 1.0) - c.c2 * n * n).abs() < 1e-13);
    }

    /// Decaying homogeneous isotropic turbulence.
    ///
    /// With `U = 0` there is no production and no transport, so the model is
    /// exactly
    ///
    /// ```text
    /// dk/dt = -eps ,   deps/dt = -C_2 eps²/k
    /// ```
    ///
    /// whose solution is `k = k_0 (1 + t/t_0)^{-n}` with `n = 1/(C_2 - 1)` and
    /// a virtual origin `t_0 = n k_0/eps_0`. Both the value and the exponent
    /// are checked, and the whole thing is repeated with a different `C_2` so
    /// that the test is measuring the coefficient rather than a coincidence.
    ///
    /// This is the analytic check SPEC-LIT §10 asks for: no other code is
    /// consulted, and the answer comes from the coefficients themselves.
    #[test]
    fn decaying_isotropic_turbulence_follows_the_model_exponent() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let hm = quiet_box();
        let mesh = GpuMesh::upload(&gpu, &hm)?;
        let n_cells = hm.n_cells;

        let u = GpuVectorField::zeros(&gpu, &mesh, "U")?;
        let phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi")?;
        let nu: Scalar = 1e-3;
        let flow = FlowState::new(&u, &phi, nu);

        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let no_roughness = crate::field_setup::NutRoughness::none(hm.n_boundary_faces);

        for c2 in [1.92 as Scalar, 1.8] {
            let coeffs = KEpsilonCoeffs { c2, ..Default::default() };

            let k0: Scalar = 1.0;
            let eps0: Scalar = 1.0;
            let n_exp = coeffs.decay_exponent();
            let t0 = n_exp * k0 / eps0;

            let t_end: Scalar = 3.0;
            let steps = 1500;
            let dt = t_end / steps as Scalar;

            let mut model = KEpsilon::new(
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
            gpu.write(&mut model.epsilon_mut().f, &vec![eps0; n_cells])?;
            model.initialise(&gpu, &flow)?;

            // Halfway, for the exponent fit.
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
            let eps_num = gpu.download(&model.epsilon().f)?;
            let k_end = k_num[0];

            // Homogeneous: every cell must hold the same number, or the
            // "no transport" premise is false and the rest is meaningless.
            for (i, &v) in k_num.iter().enumerate() {
                assert!(
                    (v - k_end).abs() <= 1e-12 * k_end,
                    "C2 {c2}: cell {i} holds k = {v}, cell 0 holds {k_end}: \
                     the field is not homogeneous"
                );
            }

            // 1. the value
            let k_exact = k0 * (1.0 + t_end / t0).powf(-n_exp);
            let rel = (k_end - k_exact).abs() / k_exact;
            assert!(
                rel < 0.01,
                "C2 {c2}: k({t_end}) = {k_end}, analytic {k_exact}, \
                 relative error {rel}"
            );

            // 2. the exponent, fitted between t_end/2 and t_end against the
            //    virtual origin the initial condition fixes.
            let n_fit = (k_half / k_end).ln() / ((t0 + t_end) / (t0 + t_half)).ln();
            assert!(
                (n_fit - n_exp).abs() < 0.02 * n_exp,
                "C2 {c2}: fitted decay exponent {n_fit}, model implies {n_exp}"
            );

            // 3. epsilon follows from k: eps = n k/(t_0 + t).
            let eps_exact = n_exp * k_exact / (t0 + t_end);
            let eps_rel = (eps_num[0] - eps_exact).abs() / eps_exact;
            assert!(
                eps_rel < 0.02,
                "C2 {c2}: epsilon({t_end}) = {}, analytic {eps_exact}, \
                 relative error {eps_rel}",
                eps_num[0]
            );
        }

        Ok(())
    }

    /// `RAS { turbulence off; }` and `simulationType laminar;` must leave
    /// `nu_t` at zero, in the cells AND on the boundary faces.
    ///
    /// Both settings used to be read and discarded: the model ran regardless,
    /// bit for bit, and the momentum equation saw an eddy viscosity the case
    /// had switched off. The boundary half matters as much as the internal
    /// one - a zero-gradient `nut` face would pick the internal value back up
    /// the next time boundary conditions were evaluated, and on a wall
    /// function `turbNutBoundary` would do it unasked.
    #[test]
    fn turbulence_off_leaves_nut_at_zero_everywhere() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let hm = quiet_box();
        let mesh = GpuMesh::upload(&gpu, &hm)?;

        let u = GpuVectorField::zeros(&gpu, &mesh, "U")?;
        let phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi")?;
        let _flow = FlowState::new(&u, &phi, 1e-3);
        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let no_roughness = crate::field_setup::NutRoughness::none(hm.n_boundary_faces);

        let mut model = KEpsilon::new(
            &gpu,
            &hm,
            &mesh,
            KEpsilonCoeffs::default(),
            decay_controls(1e-3),
            WallFunctionCoeffs::default(),
            &no_walls,
            &no_roughness,
        )?;

        // A k and an epsilon that would give a large nu_t if the model ran.
        gpu.write(&mut model.k_mut().f, &vec![0.37 as Scalar; hm.n_cells])?;
        gpu.write(&mut model.epsilon_mut().f, &vec![2.9 as Scalar; hm.n_cells])?;

        // What a driver does when `select_turbulence_model` says inactive:
        // freeze, and never call `correct`.
        model.freeze_nut(&gpu)?;
        gpu.sync()?;

        for (what, v) in [
            ("cells", gpu.download(&model.nut().f)?),
            ("boundary faces", gpu.download(&model.nut().bf)?),
        ] {
            for (i, &x) in v.iter().enumerate() {
                assert_eq!(x, 0.0, "nu_t is {x} at {what} {i} with turbulence off");
            }
        }

        // And correcting the boundary conditions must not put it back.
        crate::field_ops::correct_boundary_conditions(
            &gpu,
            &crate::field_ops::FieldKernels::new(&gpu)?,
            model.nut_mut(),
            &mesh,
        )?;
        gpu.sync()?;

        for (i, &x) in gpu.download(&model.nut().bf)?.iter().enumerate() {
            assert_eq!(x, 0.0, "nu_t came back as {x} on face {i}");
        }

        Ok(())
    }

    /// `nu_t = C_mu k²/epsilon` in every cell of the decay case, which is the
    /// one place the eddy viscosity is knowable in closed form.
    #[test]
    fn nut_is_the_launder_spalding_quotient() -> Result<()> {
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

        let coeffs = KEpsilonCoeffs::default();
        let mut model = KEpsilon::new(
            &gpu,
            &hm,
            &mesh,
            coeffs,
            decay_controls(1e-3),
            WallFunctionCoeffs::default(),
            &no_walls,
            &no_roughness,
        )?;

        let k0: Scalar = 0.37;
        let e0: Scalar = 2.9;
        gpu.write(&mut model.k_mut().f, &vec![k0; hm.n_cells])?;
        gpu.write(&mut model.epsilon_mut().f, &vec![e0; hm.n_cells])?;
        model.initialise(&gpu, &flow)?;
        gpu.sync()?;

        let nut = gpu.download(&model.nut().f)?;
        let want = coeffs.cmu * k0 * k0 / e0;

        for (i, &v) in nut.iter().enumerate() {
            assert!(
                (v - want).abs() <= 1e-14 * want,
                "cell {i}: nut {v}, expected {want}"
            );
        }

        // The boundary values are zero gradient off the adjacent cell, since
        // there are no wall functions here.
        let nut_b = gpu.download(&model.nut().bf)?;
        for &v in &nut_b {
            assert!((v - want).abs() <= 1e-14 * want, "boundary nut {v}");
        }

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  Wall functions in a whole iteration
    // ----------------------------------------------------------------------

    /// A wall-bounded run: `epsilon` really is pinned in the wall cells to the
    /// value the blended relation gives, the whole field stays finite and
    /// positive, and `k` is NOT pinned.
    ///
    /// The flow is synthetic - a uniform interior velocity with no-slip
    /// boundary values, and no flux at all - because what is under test is the
    /// near-wall treatment and the matrix constraint, not the momentum
    /// equation.
    #[test]
    fn a_wall_bounded_run_pins_epsilon_in_the_wall_cells() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let d = Vec3::new(0.05, 0.20, 0.30);
        let (mut hm, points, faces) = crate::mesh::topology::tests::box_mesh([6, 4, 1], d);
        hm.compute_geometry(&points, &faces)?;
        hm.build_cell_face_maps();

        let mesh = GpuMesh::upload(&gpu, &hm)?;

        // xmin and xmax are the walls: a channel, four cells long.
        let mut wf = vec![false; hm.n_boundary_faces];
        for pi in [0usize, 1] {
            let pp = &hm.patches[pi];
            for i in 0..pp.size {
                wf[pp.start + i] = true;
            }
        }

        // U uniform in the interior, zero on every face: the boundary values
        // of a `zeros` field are already zero, and not correcting them is what
        // makes the wall see a shear.
        let mut u = GpuVectorField::zeros(&gpu, &mesh, "U")?;
        gpu.write(&mut u.f, &vec![Vec3::new(0.0, 1.5, 0.0); hm.n_cells])?;
        let phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi")?;

        let nu: Scalar = 1e-5;
        let flow = FlowState::new(&u, &phi, nu);

        let mut ctrl = TurbulenceControls {
            steady: true,
            k_relax: 0.7,
            eps_relax: 0.7,
            nut_max_coeff: 1e8,
            ..Default::default()
        };
        ctrl.k_solver.tolerance = 1e-13;
        ctrl.k_solver.rel_tol = 0.0;
        ctrl.k_solver.max_iter = 500;
        ctrl.k_solver.report_residuals = false;
        ctrl.epsilon_solver = ctrl.k_solver;

        let wc = WallFunctionCoeffs::default();
        let coeffs = KEpsilonCoeffs::default();

        let mut model =
            KEpsilon::new(
                &gpu,
                &hm,
                &mesh,
                coeffs,
                ctrl,
                wc,
                // This test drives the wall relation itself, so both sets are
                // the same faces; a real case reads them from two files.
                &crate::field_setup::WallFaces {
                    constrained_cells: wf.clone(),
                    nut: wf.clone(),
                },
                &crate::field_setup::NutRoughness::none(hm.n_boundary_faces),
            )?;

        gpu.write(&mut model.k_mut().f, &vec![1e-3 as Scalar; hm.n_cells])?;
        gpu.write(&mut model.epsilon_mut().f, &vec![1e-3 as Scalar; hm.n_cells])?;
        model.initialise(&gpu, &flow)?;

        for _ in 0..30 {
            model.correct(&gpu, &flow)?;
        }
        gpu.sync()?;

        // The k that goes into the next iteration's wall relation.
        let k_in = gpu.download(&model.k().f)?;
        model.correct(&gpu, &flow)?;
        gpu.sync()?;

        let k_out = gpu.download(&model.k().f)?;
        let eps = gpu.download(&model.epsilon().f)?;
        let nut = gpu.download(&model.nut().f)?;

        for c in 0..hm.n_cells {
            assert!(k_out[c].is_finite() && k_out[c] > 0.0, "k[{c}] = {}", k_out[c]);
            assert!(eps[c].is_finite() && eps[c] > 0.0, "epsilon[{c}] = {}", eps[c]);
            assert!(nut[c].is_finite() && nut[c] >= 0.0, "nut[{c}] = {}", nut[c]);
        }

        // Every wall-adjacent cell must hold exactly the area-weighted
        // blended dissipation formed from the k it went in with.
        let mut checked = 0;
        for bf in 0..hm.n_boundary_faces {
            if !wf[bf] {
                continue;
            }
            let c = hm.b_face_cells[bf] as usize;
            let y = hm.b_y[bf];

            // These meshes give one wall face per wall cell, so the area
            // average is the single face's value.
            let want = crate::wallfunctions::epsilon_wall(k_in[c], y, nu, wc.kappa, wc.cmu);
            assert!(
                (eps[c] - want).abs() <= 1e-6 * want,
                "wall cell {c}: epsilon {} , blended relation {want}",
                eps[c]
            );
            checked += 1;
        }
        assert!(checked > 0, "no wall faces were checked");

        // k is not constrained: it must differ from cell to cell, because a
        // pinned k would be the bug this asserts against.
        let k_span = k_out
            .iter()
            .fold((Scalar::MAX, Scalar::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert!(
            k_span.1 > k_span.0 * (1.0 + 1e-9),
            "k is uniform at {}, so it was not solved for", k_span.0
        );

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  Host traffic
    // ----------------------------------------------------------------------

    /// One outer iteration must be capturable as a CUDA graph.
    ///
    /// That is only possible if `correct` performs no host round-trip at all -
    /// no synchronisation, no read-back, no allocation - which is the property
    /// the whole "device-resident scalars" design of `solver.rs` exists to
    /// preserve, and which a stray `download` anywhere in a model would
    /// destroy silently. Capturing one is the only way to find out.
    #[test]
    fn a_fixed_iteration_correct_captures_into_a_cuda_graph() -> Result<()> {
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

        let mut ctrl = decay_controls(1e-3);
        ctrl.k_solver.fixed_iters = true;
        ctrl.k_solver.max_iter = 4;
        ctrl.k_solver.report_residuals = false;
        ctrl.epsilon_solver = ctrl.k_solver;

        let mut model = KEpsilon::new(
            &gpu,
            &hm,
            &mesh,
            KEpsilonCoeffs::default(),
            ctrl,
            WallFunctionCoeffs::default(),
            &no_walls,
            &no_roughness,
        )?;

        gpu.write(&mut model.k_mut().f, &vec![0.05 as Scalar; hm.n_cells])?;
        gpu.write(&mut model.epsilon_mut().f, &vec![0.2 as Scalar; hm.n_cells])?;
        model.initialise(&gpu, &flow)?;

        // Warm up: the first launch of each kernel loads its module, and
        // capturing that would bake nothing useful into the graph.
        for _ in 0..3 {
            model.correct(&gpu, &flow)?;
        }
        gpu.sync()?;

        // The reference history: three more iterations, launched one kernel
        // at a time.
        let mut reference = KEpsilon::new(
            &gpu,
            &hm,
            &mesh,
            KEpsilonCoeffs::default(),
            ctrl,
            WallFunctionCoeffs::default(),
            &no_walls,
            &no_roughness,
        )?;
        gpu.write(&mut reference.k_mut().f, &gpu.download(&model.k().f)?)?;
        gpu.write(
            &mut reference.epsilon_mut().f,
            &gpu.download(&model.epsilon().f)?,
        )?;
        reference.initialise(&gpu, &flow)?;
        for _ in 0..3 {
            reference.correct(&gpu, &flow)?;
        }
        gpu.sync()?;
        let k_ref = gpu.download(&reference.k().f)?;

        // Capture executes nothing, so replaying three times matches.
        let graph = gpu.capture(|_| {
            model.correct(&gpu, &flow)?;
            Ok(())
        })?;

        let Some(mut graph) = graph else {
            panic!("capturing one outer iteration produced an empty graph");
        };
        graph.upload()?;
        for _ in 0..3 {
            graph.launch()?;
        }
        gpu.sync()?;

        let k_graph = gpu.download(&model.k().f)?;

        // Same kernels, same order, same inputs: the answers have to be
        // bit-for-bit equal, which is also the crate's determinism claim.
        for c in 0..hm.n_cells {
            assert_eq!(
                k_ref[c], k_graph[c],
                "cell {c}: per-launch {} , graph replay {}",
                k_ref[c], k_graph[c]
            );
        }

        Ok(())
    }
}
