// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Launder-Sharma low-Reynolds-number k-epsilon - SPEC-LIT §33.
//!
//! Written from:
//!   Launder & Sharma, "Application of the energy-dissipation model of
//!     turbulence to the calculation of flow near a spinning disc", Letters
//!     in Heat and Mass Transfer 1 (1974) 131-138
//!   Patel, Rodi & Scheuerer, "Turbulence models for near-wall and low
//!     Reynolds number flows: a review", AIAA J. 23 (1985) 1308-1319 -
//!     background on which damping functions survive scrutiny
//!   ofgpu `SPEC-LIT.md` §6.1 (the standard model this one extends, and whose
//!     coefficients it reuses unchanged), §15.2 (`nutLowRe`, correct rather
//!     than merely quiet once this model exists), §29.1 (the `lowRe` wall
//!     treatment row) and §33 (this model, and the §32 gate it exists to let
//!     close on a wall-resolving mesh)
//! No GPL-licensed source was consulted.
//!
//! # This is a SEPARATE model, not `KEpsilon` modified
//!
//! `models::k_epsilon::KEpsilon` stays exactly what it is - the high-Re
//! closure every existing case and the §32 gate's wall-function leg already
//! use - and this file is a second, independent type next to it. Both are
//! selectable; `models::registry` dispatches to whichever `RAS { model ...; }`
//! names.
//!
//! # What changes relative to §6.1, and what does not
//!
//! ```text
//! epsilon = epsilon_tilde + D,        D = 2 nu |grad(sqrt(k))|^2
//! nu_t    = C_mu f_mu k^2 / epsilon_tilde
//!
//! Dk/Dt   = div((nu + nu_t/sigma_k) grad k)   + G - epsilon_tilde - D
//! De~/Dt  = div((nu + nu_t/sigma_e) grad e~)  + C_1 (e~/k) G
//!                                             - C_2 f_2 e~^2/k + E
//!
//! Re_t = k^2/(nu epsilon_tilde)
//! f_mu = exp( -3.4 / (1 + Re_t/50)^2 )
//! f_2  = 1 - 0.3 exp(-Re_t^2)
//! E    = 2 nu nu_t |grad(grad U)|^2
//! ```
//!
//! The model solves for `epsilon_tilde` - the ISOTROPIC dissipation, which
//! (unlike `epsilon`) is exactly zero at a solid wall; that substitution is
//! what makes the wall boundary condition homogeneous and the equations
//! integrable through the viscous sublayer. This file (and the `0/epsilon`
//! field a case supplies) still calls it `epsilon`, the same name §6.1 uses -
//! there is no separate `epsilonTilde` file - because that is the quantity a
//! `lowRe` mesh's `0/epsilon` is read as as soon as `RAS { model
//! LaunderSharmaKE; }` says so; [`LaunderSharmaKE::epsilon`]'s own doc repeats
//! this.
//!
//! `C_mu = 0.09, C_1 = 1.44, C_2 = 1.92, sigma_k = 1.0, sigma_eps = 1.3` are
//! carried on [`crate::models::KEpsilonCoeffs`], reused UNCHANGED rather than
//! duplicated into a near-identical struct - SPEC-LIT §33.1 says the model
//! modifies §6.1 with `f_mu`, `f_2`, `D` and `E`, "not with new constants",
//! and `tests::reduces_to_the_standard_model_at_large_re_t` is the test that
//! takes that sentence literally: at `Re_t -> infinity`, `f_mu, f_2 -> 1`,
//! `D, E -> 0`, and every source term above becomes §6.1's own, coefficient
//! for coefficient.
//!
//! SPEC-LIT §33.1 gives no dilatation term for this model (unlike §6.1's
//! Favre-averaged extension - SPEC-LIT §6.1, Wilcox §5.4), so none is added:
//! the `epsilon_tilde` equation's `Sp` is exactly `C_2 f_2 (e~/k)`, the `k`
//! equation's is exactly `epsilon_tilde/k` - [`crate::turbulence::k_sources`]
//! reused unchanged, `epsilon_tilde` standing in for `epsilon` - and neither
//! equation gets a `Susp` contribution.
//!
//! # DESIGN - `grad(grad U)`, the cost this pays
//!
//! The `E` term needs the SECOND spatial derivative of the velocity, which
//! the operator set does not carry directly. SPEC-LIT §33.1 marks the
//! approach *DESIGN*: take the Gauss gradient of the ALREADY-COMPUTED cell
//! velocity gradient (`RasCore::update_flow_derived`'s own `grad_u`), once
//! per outer iteration, and reuse it for both the `k` and `epsilon_tilde`
//! equations. [`crate::turbulence::ls_grad_grad_u_mag_sqr`] is that pass: one
//! more gather over the cell -> face CSR, shaped exactly like
//! `fvc_grad_vector`'s but over 9 field components (all of `grad U`) instead
//! of 3 (`U` itself) - roughly three times the arithmetic of one more vector
//! gradient, paid once per outer iteration, and it is the only extra
//! gradient this model pays for. See that function's own doc, and
//! `cuda/turbulence.cu`'s `turbLsGradGradUMagSqr`, for the boundary
//! extrapolation this needs because `grad U` carries no boundary field of
//! its own.
//!
//! # Wall conditions - SPEC-LIT §33.2
//!
//! Homogeneous Dirichlet on both `k` and `epsilon_tilde` at a solid wall
//! (`k = 0`: no-slip means no velocity fluctuation either; `epsilon_tilde =
//! 0`: the whole point of the tilde substitution), and `nu_t = 0` there,
//! which is `nutLowReWallFunction` (SPEC-LIT §15.2) - CORRECT for this model
//! rather than merely quiet, because `k` and `epsilon_tilde` genuinely go to
//! zero at the wall instead of a wall function pinning them to a log-law
//! value. This is why the model needs no wall-function step in `correct` at
//! all: no `WallData::update_nut`/`update_epsilon` call, no wall-cell matrix
//! constraint - the ordinary Robin-triple boundary fold every transport
//! equation already gets is the whole treatment. [`LaunderSharmaKE::new`]
//! still takes `wall_faces`/`roughness` for API parity with
//! [`crate::models::KEpsilon::new`]; a caller constructing this model should
//! pass [`crate::field_setup::WallFaces::none`] and
//! [`crate::field_setup::NutRoughness::none`], because there is nothing here
//! for either to name.

use crate::device::{DevBuf, Gpu};
use crate::error::Result;
use crate::field::GpuScalarField;
use crate::field_ops::{advance_time_levels, correct_boundary_conditions};
use crate::fv::{fvc_grad_scalar, fvm_sp, fvm_su};
use crate::mesh::{GpuMesh, HostMesh};
use crate::solver::SolverPerformance;
use crate::turbulence::{
    add_buoyancy_to_epsilon, add_buoyancy_to_k, bound_epsilon, bound_k, k_sources,
    ls_d_term, ls_e_term, ls_epsilon_sources, ls_grad_grad_u_mag_sqr, ls_sqrt_positive,
    nut_boundary, nut_launder_sharma, BuoyancyProduction, FlowState, RasCore, TurbulenceControls,
};
use crate::wallfunctions::WallFunctionCoeffs;
use crate::Scalar;
use crate::Vec3;

pub use crate::models::k_epsilon::KEpsilonCoeffs;

// ==========================================================================
//  The damping functions, on the host - SPEC-LIT §33.3's analytic table
// ==========================================================================

/// `f_mu = exp( -3.4 / (1 + Re_t/50)^2 )` - Launder & Sharma (1974).
///
/// The SAME formula `turbNutLaunderSharma` evaluates on the device; kept here
/// as a pure function so SPEC-LIT §33.3's analytic limits - `Re_t -> infinity
/// => f_mu -> 1`, `Re_t = 0 => f_mu = exp(-3.4)`, monotone in between - are
/// checked against the model's own definition rather than against a kernel
/// launch.
#[inline]
pub fn f_mu(re_t: Scalar) -> Scalar {
    let d = 1.0 + re_t / 50.0;
    (-3.4 / (d * d)).exp()
}

/// `f_2 = 1 - 0.3 exp(-Re_t^2)` - Launder & Sharma (1974).
#[inline]
pub fn f2(re_t: Scalar) -> Scalar {
    1.0 - 0.3 * (-re_t * re_t).exp()
}

// ==========================================================================
//  The model
// ==========================================================================

/// Launder-Sharma low-Reynolds-number k-epsilon, resident on the device.
pub struct LaunderSharmaKE<'m> {
    core: RasCore<'m>,
    coeffs: KEpsilonCoeffs,
    k: GpuScalarField,
    /// `epsilon_tilde`, the isotropic dissipation this model actually
    /// transports - see the module doc's "What changes relative to §6.1" for
    /// why the field is still called `epsilon`.
    epsilon: GpuScalarField,

    /// Scratch: `sqrt(max(k, 0))`, interior and boundary, so
    /// [`crate::fv::fvc_grad_scalar`] can be handed a real
    /// [`GpuScalarField`] to gradient. Rebuilt every `correct`.
    sqrt_k: GpuScalarField,
    /// `[n_cells]` `grad(sqrt k)`.
    grad_sqrt_k: DevBuf<Vec3>,
    /// `[n_cells]` `D = 2 nu |grad(sqrt k)|^2`.
    d_term: DevBuf<Scalar>,
    /// `[n_cells]` `|grad(grad U)|^2` - see the module doc's DESIGN note.
    grad_grad_u_mag_sqr: DevBuf<Scalar>,
    /// `[n_cells]` `E = 2 nu nu_t |grad(grad U)|^2`.
    e_term: DevBuf<Scalar>,
}

impl<'m> LaunderSharmaKE<'m> {
    /// See the module doc's "Wall conditions" section for why `wall_faces`
    /// and `roughness` should be the empty ones for this model.
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
        let nc = mesh.n_cells.max(1);
        Ok(Self {
            core: RasCore::new(gpu, hm, mesh, ctrl, wall, wall_faces, roughness)?,
            coeffs,
            k: GpuScalarField::zeros(gpu, mesh, "k")?,
            epsilon: GpuScalarField::zeros(gpu, mesh, "epsilon")?,
            sqrt_k: GpuScalarField::zeros(gpu, mesh, "sqrtK")?,
            grad_sqrt_k: gpu.zeros(nc)?,
            d_term: gpu.zeros(nc)?,
            grad_grad_u_mag_sqr: gpu.zeros(nc)?,
            e_term: gpu.zeros(nc)?,
        })
    }

    // ---- accessors --------------------------------------------------------

    pub fn k(&self) -> &GpuScalarField {
        &self.k
    }
    pub fn k_mut(&mut self) -> &mut GpuScalarField {
        &mut self.k
    }
    /// `epsilon_tilde` - see the module doc for why the field (and this
    /// accessor) is still named `epsilon`.
    pub fn epsilon(&self) -> &GpuScalarField {
        &self.epsilon
    }
    pub fn epsilon_mut(&mut self) -> &mut GpuScalarField {
        &mut self.epsilon
    }
    pub fn nut(&self) -> &GpuScalarField {
        &self.core.nut
    }
    /// `nu_t = 0` everywhere - `RAS { turbulence off; }` and
    /// `simulationType laminar;`. See [`crate::turbulence::RasCore::freeze_nut`].
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
    /// restart checkpoint's view of this model, exactly as
    /// [`crate::models::KEpsilon::named_fields`].
    pub fn named_fields(&self) -> Vec<(&'static str, &GpuScalarField)> {
        vec![("k", &self.k), ("epsilon", &self.epsilon), ("nut", &self.core.nut)]
    }

    /// [`Self::named_fields`], mutable.
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

    /// Switch the buoyancy production `G_b` on - SPEC-LIT §17, exactly as
    /// [`crate::models::KEpsilon::set_buoyancy`].
    pub fn set_buoyancy(&mut self, b: BuoyancyProduction) -> Result<()> {
        b.validate()?;
        self.core.buoyancy = Some(b);
        Ok(())
    }

    pub fn buoyancy(&self) -> Option<BuoyancyProduction> {
        self.core.buoyancy
    }

    // ---- set-up -----------------------------------------------------------

    /// Bound the initial fields, evaluate their boundaries, and build the
    /// first `nu_t`. Call once, after the initial `k` and `epsilon` have been
    /// uploaded.
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

    // ---- one outer iteration -----------------------------------------------

    /// Solve `epsilon_tilde`, then `k`, then update `nu_t`. Returns
    /// `(epsilon, k)` performance, matching
    /// [`crate::models::KEpsilon::correct`]'s order.
    pub fn correct(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        self.correct_buoyant(gpu, flow, None)
    }

    /// [`Self::correct`] with the temperature the buoyancy production is
    /// built from - SPEC-LIT §17, exactly as
    /// [`crate::models::KEpsilon::correct_buoyant`].
    pub fn correct_buoyant(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        t: Option<&GpuScalarField>,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        let n = self.core.mesh.n_cells;
        let nb = self.core.mesh.n_boundary_faces;
        let ctrl = self.core.ctrl;
        let c = self.coeffs;
        let nu = flow.nu;
        let nut_max = self.core.nut_max(nu);

        // 1. the convergence baseline and the old time levels.
        self.core.store_k_prev(gpu, &self.k.f)?;
        advance_time_levels(gpu, &self.core.fld, &mut self.k)?;
        advance_time_levels(gpu, &self.core.fld, &mut self.epsilon)?;
        self.core.ddt.advance(ctrl.delta_t);

        // 2. grad U, G and div u - the PREVIOUS nu_t, same lag §6.1 uses.
        self.core.update_flow_derived(gpu, flow)?;

        // 2b. G_b - SPEC-LIT §17.
        let buoyant = match t {
            Some(tf) => self.core.update_buoyancy_production(gpu, tf, flow.u)?,
            None => false,
        };

        // 3. SPEC-LIT §33.2: no wall-function step. Homogeneous Dirichlet on
        //    `k` and `epsilon_tilde`, `nu_t = 0`, all three from the fields'
        //    own boundary conditions - see the module doc.

        // 3b. D and E - SPEC-LIT §33.1's DESIGN note. Both read the
        //     PREVIOUS `k`/`nu_t`, before this iteration's solves touch them.
        ls_sqrt_positive(gpu, &self.core.turb, &mut self.sqrt_k.f, &self.k.f, n)?;
        ls_sqrt_positive(gpu, &self.core.turb, &mut self.sqrt_k.bf, &self.k.bf, nb)?;
        fvc_grad_scalar(gpu, &self.core.fv, &mut self.grad_sqrt_k, &self.sqrt_k, self.core.mesh)?;
        ls_d_term(gpu, &self.core.turb, &mut self.d_term, &self.grad_sqrt_k, nu, n)?;

        ls_grad_grad_u_mag_sqr(
            gpu,
            &self.core.turb,
            &mut self.grad_grad_u_mag_sqr,
            &self.core.grad_u,
            self.core.mesh,
        )?;
        ls_e_term(
            gpu,
            &self.core.turb,
            &mut self.e_term,
            &self.grad_grad_u_mag_sqr,
            &self.core.nut.f,
            nu,
            n,
        )?;

        // 4. epsilon_tilde.
        self.core
            .assemble_transport(gpu, flow, &self.epsilon, ctrl.eps_conv(), 1.0 / c.sigma_eps)?;

        ls_epsilon_sources(
            gpu,
            &self.core.turb,
            &mut self.core.su,
            &mut self.core.sp,
            &self.core.g,
            &self.k.f,
            &self.epsilon.f,
            &self.e_term,
            nu,
            c.c1,
            c.c2,
            ctrl.k_min,
            n,
        )?;

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

        fvm_su(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.su, 1.0)?;
        fvm_sp(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.sp, 1.0)?;
        // No fvm_susp: SPEC-LIT §33.1 gives no dilatation term for this
        // model - see the module doc.

        let sc = ctrl.epsilon_solver;
        // No wall constraint (`constrain_walls = false`): SPEC-LIT §33.2's
        // homogeneous Dirichlet is an ordinary boundary condition, not a
        // wall-function cell pin.
        let eps_perf = self
            .core
            .solve_equation(gpu, &mut self.epsilon, ctrl.eps_relax, &sc, false)?;

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

        // 5. k.
        self.core
            .assemble_transport(gpu, flow, &self.k, ctrl.k_conv(), 1.0 / c.sigmak)?;

        // Sp = epsilon_tilde/k, exactly SPEC-LIT §6.1's own k-equation sink -
        // `k_sources` reused unchanged, `epsilon_tilde` standing in for
        // `epsilon`. Its `susp` output is discarded (never passed to
        // `fvm_susp`): SPEC-LIT §33.1 has no dilatation term.
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

        if buoyant {
            {
                let RasCore { turb, su, sp, gb, .. } = &mut self.core;
                add_buoyancy_to_k(gpu, turb, su, sp, gb, &self.k.f, ctrl.k_min, n)?;
            }
            fvm_su(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.su, 1.0)?;
        }

        // -D, explicit (SPEC-LIT §33.1: "Dk/Dt = ... - epsilon_tilde - D").
        fvm_su(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.d_term, -1.0)?;

        fvm_sp(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.sp, 1.0)?;

        let sc = ctrl.k_solver;
        let k_perf = self
            .core
            .solve_equation(gpu, &mut self.k, ctrl.k_relax, &sc, false)?;

        bound_k(gpu, &self.core.turb, &mut self.k.f, ctrl.k_min, n)?;
        correct_boundary_conditions(gpu, &self.core.fld, &mut self.k, self.core.mesh)?;

        // 6. nu_t.
        self.correct_nut(gpu, flow)?;

        Ok((eps_perf, k_perf))
    }

    /// `nu_t = C_mu f_mu k^2/epsilon_tilde` and its boundary values.
    ///
    /// No wall-function override: `nutLowReWallFunction`'s own triple/
    /// [`crate::turbulence::nut_boundary`] pin already give `nu_t,w = 0`
    /// (SPEC-LIT §15.2, §33.2) with no model-side call needed.
    pub fn correct_nut(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        let n = self.core.mesh.n_cells;
        let nut_max = self.core.nut_max(flow.nu);

        nut_launder_sharma(
            gpu,
            &self.core.turb,
            &mut self.core.nut.f,
            &self.k.f,
            &self.epsilon.f,
            flow.nu,
            self.coeffs.cmu,
            nut_max,
            n,
        )?;

        correct_boundary_conditions(gpu, &self.core.fld, &mut self.core.nut, self.core.mesh)?;
        nut_boundary(gpu, &self.core.turb, &mut self.core.nut, self.core.mesh)?;

        Ok(())
    }

    /// `max|Δk|/max|k|` since the last call to `correct`.
    pub fn convergence_measure(&mut self, gpu: &Gpu) -> Result<Scalar> {
        self.core.convergence_measure(gpu, &self.k.f)
    }
}

// ==========================================================================
//  The §33.2 mesh check
// ==========================================================================

/// SPEC-LIT §33.2's mesh check, MEASURED rather than assumed: a low-Re model
/// on a wall-function mesh is as wrong as the reverse, and just as silent.
///
/// A pure function of already-downloaded fields, so it carries no GPU
/// dependency of its own - a driver downloads `k` and the Poisson wall
/// distance `y` ([`crate::walldistance::wall_distance`], the same one
/// `kOmegaSST` already needs at set-up) once, and hands both here alongside
/// the owner cell of every wall boundary face.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshResolutionReport {
    /// The largest y+ among the cells immediately off a wall face - what
    /// matters for "does this mesh resolve the sublayer", since any ONE wall
    /// face failing it makes the model wrong there.
    pub max_first_cell_y_plus: Scalar,
    /// How many cells across the mesh sit at y+ < 20. Counted GLOBALLY
    /// rather than per wall-normal column - this crate carries no
    /// wall-normal-column topology to count them the other way - which is a
    /// stated approximation, not a hidden one.
    pub cells_below_y_plus_20: usize,
    pub n_wall_faces: usize,
}

impl MeshResolutionReport {
    /// SPEC-LIT §33.2: "warn when [the first cell is at] y+ > 1 or [there
    /// are] fewer than 10 cells inside y+ < 20". Empty when there is nothing
    /// to warn about, INCLUDING a domain with no walls at all (nothing for a
    /// low-Re treatment to be wrong about there).
    pub fn warnings(&self) -> Vec<String> {
        let mut w = Vec::new();
        if self.n_wall_faces == 0 {
            return w;
        }
        if self.max_first_cell_y_plus > 1.0 {
            w.push(format!(
                "LaunderSharmaKE (SPEC-LIT 33.2): worst first-cell y+ is {:.3} \
                 (> 1) - this mesh does not resolve the viscous sublayer the \
                 model assumes; use wallTreatment standard/spalding, or refine \
                 the wall-normal mesh",
                self.max_first_cell_y_plus
            ));
        }
        if self.cells_below_y_plus_20 < 10 {
            w.push(format!(
                "LaunderSharmaKE (SPEC-LIT 33.2): only {} cells sit at y+ < 20 \
                 (< 10) - too few to resolve the buffer layer the damping \
                 functions f_mu/f_2 act through",
                self.cells_below_y_plus_20
            ));
        }
        w
    }
}

/// Build [`MeshResolutionReport`] from `k`, the wall distance `y` (both
/// indexed by cell) and the owner cell of every wall boundary face.
///
/// `cmu` is [`KEpsilonCoeffs::cmu`] - `y+ = C_mu^{1/4} y sqrt(k)/nu`
/// ([`crate::wallfunctions::y_plus_of`]), the same relation the standard
/// model's own wall functions use, because it is a statement about the mesh
/// and the flow, not about which model reads it.
pub fn mesh_resolution_report(
    k: &[Scalar],
    y: &[Scalar],
    wall_face_owner: &[usize],
    nu: Scalar,
    cmu: Scalar,
) -> MeshResolutionReport {
    let y_plus_at = |c: usize| -> Scalar {
        let kc = k.get(c).copied().unwrap_or(0.0);
        let yc = y.get(c).copied().unwrap_or(0.0);
        crate::wallfunctions::y_plus_of(kc, yc, nu, cmu)
    };

    let max_first_cell_y_plus = wall_face_owner
        .iter()
        .map(|&c| y_plus_at(c))
        .fold(0.0 as Scalar, Scalar::max);

    let cells_below_y_plus_20 = (0..k.len().min(y.len()))
        .filter(|&c| y_plus_at(c) < 20.0)
        .count();

    MeshResolutionReport {
        max_first_cell_y_plus,
        cells_below_y_plus_20,
        n_wall_faces: wall_face_owner.len(),
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::GpuVectorField;
    use crate::Tensor;

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// `nz = 1`, deliberately: `box_mesh`'s `zmin`/`zmax` patches are
    /// `PatchKind::Empty`, and on a SINGLE z-layer both of a cell's z-faces
    /// are that same pair - equal area, opposite normal, cancelling exactly
    /// - so skipping them (as `fvGradScalar`/`turbLsGradGradUMagSqr` do for
    /// every `Empty` face) still leaves a closed cell. `nz > 1` splits a
    /// boundary-layer cell's two z-faces into one `Empty` and one internal,
    /// which does NOT cancel and would make even a UNIFORM field's gradient
    /// spuriously non-zero - exactly the false positive this fixture must
    /// not produce, so this is not merely a smaller mesh but the one shape
    /// that makes "gradient of a uniform/linear field" mean what these tests
    /// need it to.
    fn quiet_box() -> HostMesh {
        let (mut m, points, faces) =
            crate::mesh::topology::tests::box_mesh([4, 4, 1], Vec3::new(0.25, 0.25, 0.25));
        m.compute_geometry(&points, &faces).expect("geometry");
        m.build_cell_face_maps();
        m
    }

    // ----------------------------------------------------------------------
    //  SPEC-LIT §33.3 - the analytic table
    // ----------------------------------------------------------------------

    /// `f_mu, f_2 -> 1` at large `Re_t`, and the model reduces to §6.1
    /// exactly - checked against the STANDARD model's own coefficients,
    /// which is what SPEC-LIT §33.3 asks for: this model reuses
    /// [`KEpsilonCoeffs`] rather than carrying a second copy, so "reduces to
    /// §6.1" is "the same struct, the damping functions at their Re_t ->
    /// infinity limit".
    #[test]
    fn reduces_to_the_standard_model_at_large_re_t() {
        let std = KEpsilonCoeffs::default();

        for re_t in [1e6 as Scalar, 1e9, 1e12] {
            let fmu = f_mu(re_t);
            let f2v = f2(re_t);
            assert!(
                (fmu - 1.0).abs() < 1e-6,
                "Re_t {re_t}: f_mu = {fmu}, expected -> 1"
            );
            assert!(
                (f2v - 1.0).abs() < 1e-9,
                "Re_t {re_t}: f_2 = {f2v}, expected -> 1"
            );
        }

        // The coefficients THEMSELVES are the standard model's, unchanged -
        // SPEC-LIT §33.1: "modifies the model with f_mu, f_2, D and E, not
        // with new constants".
        let ls_default = KEpsilonCoeffs::default();
        assert_eq!(ls_default.cmu, std.cmu);
        assert_eq!(ls_default.c1, std.c1);
        assert_eq!(ls_default.c2, std.c2);
        assert_eq!(ls_default.sigmak, std.sigmak);
        assert_eq!(ls_default.sigma_eps, std.sigma_eps);

        // And the reduced nu_t/epsilon-sink expressions agree with §6.1's own
        // to the same tolerance f_mu/f_2 do, for a representative state.
        let (k, e, g, nu): (Scalar, Scalar, Scalar, Scalar) = (0.5, 0.3, 0.2, 1e-5);
        let re_t = k * k / (nu * e);
        let nut_ls = std.cmu * f_mu(re_t) * k * k / e;
        let nut_std = std.cmu * k * k / e;
        assert!((nut_ls - nut_std).abs() / nut_std < 1e-4, "Re_t {re_t}");

        let sp_ls = std.c2 * f2(re_t) * e / k;
        let sp_std = std.c2 * e / k;
        assert!((sp_ls - sp_std).abs() / sp_std < 1e-8);

        // Su has no f-factor at all in either model.
        let su_ls = std.c1 * (e / k) * g;
        let su_std = std.c1 * (e / k) * g;
        assert_eq!(su_ls, su_std);
    }

    /// `f_mu(Re_t = 0) = exp(-3.4)`, and monotone increasing in between -
    /// SPEC-LIT §33.3.
    #[test]
    fn f_mu_at_zero_is_exp_minus_3_4_and_monotone() {
        let want = (-3.4 as Scalar).exp();
        let got = f_mu(0.0);
        assert!((got - want).abs() < 1e-14, "f_mu(0) = {got}, expected {want}");

        // exp(-3.4) ~ 0.0334: nu_t suppressed by ~30x at the wall, the
        // number SPEC-LIT §33.1 itself quotes.
        assert!((got - 0.0334).abs() < 1e-3, "f_mu(0) = {got}");

        let samples: Vec<Scalar> = (0..=200).map(|i| i as Scalar * 5.0).collect();
        let mut prev = f_mu(samples[0]);
        for &re_t in &samples[1..] {
            let v = f_mu(re_t);
            assert!(
                v >= prev - 1e-15,
                "f_mu not monotone: f_mu({}) = {} < f_mu(prev) = {}",
                re_t,
                v,
                prev
            );
            assert!((0.0..=1.0 + 1e-12).contains(&v), "f_mu({re_t}) = {v} out of [0,1]");
            prev = v;
        }
    }

    /// `f_2` stays within its own bounds and increases toward 1 - the
    /// companion limit to `f_mu`'s.
    #[test]
    fn f2_is_bounded_and_increases_to_one() {
        assert!((f2(0.0) - 0.7).abs() < 1e-12, "f_2(0) = {}", f2(0.0));
        let mut prev = f2(0.0);
        for i in 1..=200 {
            let re_t = i as Scalar * 0.1;
            let v = f2(re_t);
            assert!(v >= prev - 1e-15, "f_2 not monotone at Re_t {re_t}");
            assert!((0.7..=1.0 + 1e-12).contains(&v), "f_2({re_t}) = {v} out of [0.7,1]");
            prev = v;
        }
        assert!((f2(50.0) - 1.0).abs() < 1e-6);
    }

    /// `D = 0` on a uniform `k` field - SPEC-LIT §33.3. A uniform field's
    /// Gauss gradient is zero by construction; this measures the DISCRETE
    /// kernel path (`sqrt`, then `fvc_grad_scalar`, then `turbLsDTerm`)
    /// rather than the algebra alone.
    #[test]
    fn d_term_is_zero_on_a_uniform_k_field() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let hm = quiet_box();
        let mesh = GpuMesh::upload(&gpu, &hm)?;
        let turb = crate::turbulence::TurbKernels::new(&gpu)?;
        let fv = crate::fv::FvKernels::new(&gpu)?;

        let k0: Scalar = 0.42;
        let mut sqrt_k = GpuScalarField::zeros(&gpu, &mesh, "sqrtK")?;
        let mut k = GpuScalarField::zeros(&gpu, &mesh, "k")?;
        gpu.write(&mut k.f, &vec![k0; hm.n_cells])?;
        gpu.write(&mut k.bf, &vec![k0; hm.n_boundary_faces])?;

        ls_sqrt_positive(&gpu, &turb, &mut sqrt_k.f, &k.f, hm.n_cells)?;
        ls_sqrt_positive(&gpu, &turb, &mut sqrt_k.bf, &k.bf, hm.n_boundary_faces)?;

        let mut grad_sqrt_k: DevBuf<Vec3> = gpu.zeros(hm.n_cells.max(1))?;
        fvc_grad_scalar(&gpu, &fv, &mut grad_sqrt_k, &sqrt_k, &mesh)?;

        let mut d: DevBuf<Scalar> = gpu.zeros(hm.n_cells.max(1))?;
        ls_d_term(&gpu, &turb, &mut d, &grad_sqrt_k, 1.5e-5, hm.n_cells)?;
        gpu.sync()?;

        let d_host = gpu.download(&d)?;
        for (c, &v) in d_host.iter().enumerate() {
            assert!(v.abs() < 1e-20, "cell {c}: D = {v}, expected 0 on a uniform k field");
        }

        Ok(())
    }

    /// `E = 0` on a linear velocity field - SPEC-LIT §33.3. `grad U` is then
    /// uniform everywhere (including, exactly, at the boundary-extrapolated
    /// value this kernel's DESIGN note uses), so its own Gauss gradient is
    /// zero and `E` vanishes with it.
    #[test]
    fn e_term_is_zero_on_a_linear_velocity_field() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let hm = quiet_box();
        let mesh = GpuMesh::upload(&gpu, &hm)?;
        let turb = crate::turbulence::TurbKernels::new(&gpu)?;
        let fv = crate::fv::FvKernels::new(&gpu)?;

        // Component (i,j) of grad U is dU_j/dx_i (SPEC-LIT §1 / `Tensor`'s
        // own doc), so `U_j = a.xj x + a.yj y + a.zj z` reconstructs grad U
        // = a exactly - PROVIDED every derivative asked of it is one the
        // mesh can see. `a`'s z-ROW (`zx`, `zy`, `zz` - every d(*)/dz) is
        // zero: `quiet_box` (nz = 1, empty z-patches) cannot represent an
        // actual z-derivative at all, empty or not, and asking it to would
        // be testing the mesh's limitation rather than this kernel. `U_z`
        // itself still varies with x and y (`a.xz`, `a.yz` non-zero, the
        // z-COLUMN), so this is not degenerate in the two directions that
        // matter.
        let a = Tensor {
            xx: 1.3, xy: -0.7, xz: 0.4,
            yx: 0.2, yy: 0.9, yz: -1.1,
            zx: 0.0, zy: 0.0, zz: 0.0,
        };
        let u_of = |p: Vec3| -> Vec3 {
            Vec3::new(
                a.xx * p.x + a.yx * p.y + a.zx * p.z,
                a.xy * p.x + a.yy * p.y + a.zy * p.z,
                a.xz * p.x + a.yz * p.y + a.zz * p.z,
            )
        };

        let mut u = GpuVectorField::zeros(&gpu, &mesh, "U")?;
        let u_cells: Vec<Vec3> = hm.c.iter().map(|&p| u_of(p)).collect();
        let u_bf: Vec<Vec3> = hm.b_cf.iter().map(|&p| u_of(p)).collect();
        gpu.write(&mut u.f, &u_cells)?;
        gpu.write(&mut u.bf, &u_bf)?;

        let mut grad_u: DevBuf<Tensor> = gpu.zeros(hm.n_cells.max(1))?;
        crate::fv::fvc_grad_vector(&gpu, &fv, &mut grad_u, &u, &mesh)?;
        gpu.sync()?;

        // Sanity: the discrete gradient really did reconstruct A everywhere,
        // or the rest of this test would be checking nothing.
        let grad_host = gpu.download(&grad_u)?;
        for (c, &g) in grad_host.iter().enumerate() {
            let scale = 1e-9_f64 as Scalar;
            assert!((g.xx - a.xx).abs() < scale, "cell {c}: grad_u.xx = {}", g.xx);
            assert!((g.xz - a.xz).abs() < scale, "cell {c}: grad_u.xz = {}", g.xz);
            assert!((g.yz - a.yz).abs() < scale, "cell {c}: grad_u.yz = {}", g.yz);
        }

        let mut mag_sqr: DevBuf<Scalar> = gpu.zeros(hm.n_cells.max(1))?;
        ls_grad_grad_u_mag_sqr(&gpu, &turb, &mut mag_sqr, &grad_u, &mesh)?;
        gpu.sync()?;

        let mag_host = gpu.download(&mag_sqr)?;
        for (c, &v) in mag_host.iter().enumerate() {
            assert!(
                v.abs() < 1e-12,
                "cell {c}: |grad(grad U)|^2 = {v}, expected 0 on a linear U"
            );
        }

        // And E itself, with a non-trivial nu_t, so this is not vacuously
        // zero through nu_t alone.
        let mut nut: DevBuf<Scalar> = gpu.zeros(hm.n_cells.max(1))?;
        gpu.write(&mut nut, &vec![3.7 as Scalar; hm.n_cells])?;
        let mut e: DevBuf<Scalar> = gpu.zeros(hm.n_cells.max(1))?;
        ls_e_term(&gpu, &turb, &mut e, &mag_sqr, &nut, 1.5e-5, hm.n_cells)?;
        gpu.sync()?;

        for (c, &v) in gpu.download(&e)?.iter().enumerate() {
            assert!(v.abs() < 1e-12, "cell {c}: E = {v}, expected 0");
        }

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  The §33.2 mesh check
    // ----------------------------------------------------------------------

    #[test]
    fn mesh_report_warns_when_the_first_cell_is_too_coarse() {
        let nu: Scalar = 1.5e-5;
        let cmu: Scalar = 0.09;
        // y+ = Cmu^0.25 * y * sqrt(k) / nu; pick y, k so y+ ~ 30 (wall-function
        // territory, not lowRe).
        let k = vec![0.05 as Scalar; 4];
        let y = vec![2e-3 as Scalar; 4];
        let owners = vec![0usize];

        let report = mesh_resolution_report(&k, &y, &owners, nu, cmu);
        assert!(report.max_first_cell_y_plus > 1.0, "{report:?}");
        let warnings = report.warnings();
        assert!(!warnings.is_empty());
        assert!(warnings.iter().any(|w| w.contains("y+")));
    }

    #[test]
    fn mesh_report_is_clean_on_a_properly_resolved_wall() {
        let nu: Scalar = 1.5e-5;
        let cmu: Scalar = 0.09;
        // A graded column: y+ well under 1 at the wall, growing outward
        // past 20 with at least 10 cells still below it.
        let ys: Vec<Scalar> = (0..40).map(|i| 1e-5 * (1.15 as Scalar).powi(i)).collect();
        let k = vec![0.02 as Scalar; ys.len()];
        let owners = vec![0usize];

        let report = mesh_resolution_report(&k, &ys, &owners, nu, cmu);
        assert!(
            report.max_first_cell_y_plus <= 1.0,
            "first-cell y+ {} should resolve the sublayer",
            report.max_first_cell_y_plus
        );
        assert!(
            report.cells_below_y_plus_20 >= 10,
            "only {} cells below y+ 20",
            report.cells_below_y_plus_20
        );
        assert!(report.warnings().is_empty(), "{:?}", report.warnings());
    }

    #[test]
    fn mesh_report_with_no_walls_warns_about_nothing() {
        let report = mesh_resolution_report(&[], &[], &[], 1.5e-5, 0.09);
        assert_eq!(report.n_wall_faces, 0);
        assert!(report.warnings().is_empty());
    }
}
