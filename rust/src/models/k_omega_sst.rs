// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Menter k-omega SST - SPEC-LIT §6.3.
//!
//! Written from:
//!   Menter, *AIAA J.* 32 (1994) 1598-1605 - the model, F1, F2, the two
//!     coefficient sets, the cross-diffusion term
//!   Menter, Kuntz & Langtry, *Turbulence, Heat and Mass Transfer* 4 (2003)
//!     625-632 - the revision this file implements; see below
//!   Bradshaw, Ferriss & Atwell, *J. Fluid Mech.* 28 (1967) 593-616 - the
//!     `tau = a_1 k` observation the eddy-viscosity limiter encodes
//!   Wilcox, *Turbulence Modeling for CFD*, DCW Industries - the inner model
//!     SST reduces to at `F1 = 1`
//!   Launder & Spalding, *Comput. Methods Appl. Mech. Eng.* 3 (1974) 269-289 -
//!     the outer model it reduces to at `F1 = 0`, and the wall treatment,
//!     which is shared
//!   Tucker, *Applied Mathematical Modelling* 22 (1998) 293-305 - the wall
//!     distance `y` that both blending functions need
//!   Patankar, *Numerical Heat Transfer and Fluid Flow* (1980) §4.2 - the
//!     linearisation the cross-diffusion term is emitted through
//!   ofgpu `SPEC-LIT.md` §6.3, which tabulates every coefficient here, and
//!     §6.6 for the wall distance
//! No GPL-licensed source was consulted.
//!
//! # Which variant, and why it matters
//!
//! **The 2003 revision.** SPEC-LIT §6.3 asks for the choice to be stated
//! because the two published forms are not the same model. They differ in
//! exactly two places and this file takes the later one in both:
//!
//! | | 1994 | 2003, implemented here |
//! |---|---|---|
//! | `nu_t` denominator | vorticity magnitude `Omega` | strain rate `S` |
//! | `k` production | limited against the dissipation with a large factor | `min(G, c_1 beta* k omega)`, `c_1 = 10` |
//!
//! SPEC-LIT's own table writes `nu_t = a_1 k/max(a_1 omega, b_1 F_2 sqrt(S²))`
//! with `S² = 2|symm(grad U)|²` and `min(G, c_1 beta* k omega)` with
//! `c_1 = 10`, so following the specification *is* taking the 2003 form; this
//! note exists so that a reader comparing against the 1994 paper knows the
//! difference is deliberate.
//!
//! # What the model is
//!
//! ```text
//! nu_t = a_1 k / max( a_1 omega , b_1 F_2 sqrt(S²) )
//!
//! Dk/Dt   = ∇·((nu + blend(sigma_k) nu_t)∇k)
//!           + min(G, c_1 beta* k omega) - beta* k omega
//!
//! Dw/Dt   = ∇·((nu + blend(sigma_w) nu_t)∇omega)
//!           + blend(gamma) (G/nu_t) - blend(beta) omega²
//!           + 2 (1 - F_1) sigma_w2 (∇k·∇omega)/omega
//! ```
//!
//! with `blend(phi) = F_1 phi_1 + (1 - F_1) phi_2`.
//!
//! Set 1 is Wilcox k-omega, which behaves in a viscous sublayer and is
//! notoriously sensitive to the free-stream `omega`. Set 2 is the standard
//! k-epsilon transformed into `omega` variables, which is insensitive to the
//! free stream and wrong at a wall. `F_1` is one near a wall and zero away
//! from it, so each model is used where it is right.
//!
//! # Set 2 really is the transformed k-epsilon
//!
//! This is not a slogan; it is an identity between four numbers, and it is
//! worth writing down because it is what makes the blend meaningful rather
//! than a fit. Substituting `epsilon = beta* k omega` into
//!
//! ```text
//! dk/dt = -epsilon ,   d(epsilon)/dt = C_1 (eps/k) G - C_2 eps²/k
//! ```
//!
//! gives, for the homogeneous case,
//!
//! ```text
//! domega/dt = (C_1 - 1)(omega/k) G - beta*(C_2 - 1) omega²
//! ```
//!
//! so `gamma_2 = C_1 - 1` and `beta_2 = beta*(C_2 - 1)`. With Launder &
//! Spalding's `C_1 = 1.44`, `C_2 = 1.92` and `beta* = 0.09` those are
//! `0.44` and `0.0828` - precisely the two numbers SPEC-LIT §6.3 tabulates.
//! [`KOmegaSstCoeffs::k_epsilon_equivalent`] states the identity and
//! `tests::set_two_is_the_transformed_k_epsilon` pins it, and
//! `tests::forcing_f1_to_zero_reproduces_the_transformed_k_epsilon` measures
//! the consequence against an actual k-epsilon run.
//!
//! # Forcing `F_1`
//!
//! [`KOmegaSst::force_f1`] replaces the computed blending function with a
//! constant. It exists for the two tests SPEC-LIT §22 asks for - `F_1 = 1`
//! must reproduce k-omega and `F_1 = 0` the transformed k-epsilon - and those
//! are the real test of the blending, because they are the only ones that can
//! tell a blend of two correct models from a blend of two wrong ones. It is
//! not a modelling switch and no case dictionary reaches it.
//!
//! # Order of work in one `correct`
//!
//! `omega` then `k` then `nu_t`, as in `k_omega.rs`, with the blending
//! functions and the four blended coefficient fields rebuilt at the top from
//! the fields the previous iteration left. The one thing that is genuinely
//! lagged is `F_2` inside `nu_t`: it is formed before the two solves and used
//! after them, which is the same segregated lag `G` already carries.

use crate::device::{DevBuf, Gpu};
use crate::error::{Error, Result};
use crate::field::GpuScalarField;
use crate::field_ops::{advance_time_levels, correct_boundary_conditions, set_field, FieldKernels};
use crate::fv::{fvc_grad_scalar_scheme, fvm_sp, fvm_su, fvm_susp};
use crate::mesh::{GpuMesh, HostMesh};
use crate::solver::SolverPerformance;
use crate::turbulence::{
    add_buoyancy_to_k, add_buoyancy_to_omega_cell, bound_k, bound_omega, nut_boundary,
    strain_rate_mag, BuoyancyProduction, FlowState, RasCore, TurbulenceControls,
};
use crate::wallfunctions::WallFunctionCoeffs;
use crate::{Scalar, Vec3};

mod kernels;
pub use kernels::SstKernels;
use kernels::{
    sst_blend_coeffs, sst_blending, sst_k_sources, sst_nut, sst_omega_sources,
    sst_production_by_nut,
};

// ==========================================================================
//  Coefficients
// ==========================================================================

/// The twelve constants of SPEC-LIT §6.3.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KOmegaSstCoeffs {
    /// Set 1 - the inner, k-omega set.
    pub sigma_k1: Scalar,
    pub sigma_w1: Scalar,
    pub beta_1: Scalar,
    pub gamma_1: Scalar,

    /// Set 2 - the outer, transformed-k-epsilon set. `sigma_w2` is also the
    /// coefficient of the cross-diffusion term itself, and of the third branch
    /// of `arg1`; it is one constant doing three jobs and they must be the
    /// same number.
    pub sigma_k2: Scalar,
    pub sigma_w2: Scalar,
    pub beta_2: Scalar,
    pub gamma_2: Scalar,

    /// Common to both sets.
    pub beta_star: Scalar,
    /// Bradshaw's constant in the eddy-viscosity limiter.
    pub a1: Scalar,
    /// The `F_2` multiplier in the same limiter.
    pub b1: Scalar,
    /// The production limiter's factor on the local dissipation.
    pub c1: Scalar,
}

impl Default for KOmegaSstCoeffs {
    fn default() -> Self {
        Self {
            sigma_k1: 0.85,
            sigma_w1: 0.5,
            beta_1: 0.075,
            // The exact rational, as in `k_omega.rs`: this is Wilcox's 5/9 and
            // not a transcribed 0.5556.
            gamma_1: 5.0 / 9.0,

            sigma_k2: 1.0,
            sigma_w2: 0.856,
            beta_2: 0.0828,
            gamma_2: 0.44,

            beta_star: 0.09,
            a1: 0.31,
            b1: 1.0,
            c1: 10.0,
        }
    }
}

impl KOmegaSstCoeffs {
    /// The `(C_1, C_2)` of the k-epsilon model set 2 is the transform of.
    ///
    /// `gamma_2 = C_1 - 1` and `beta_2 = beta*(C_2 - 1)`, inverted. Returning
    /// the pair rather than asserting the identity means a case that overrides
    /// `beta_2` can be *asked* which k-epsilon it has just implied, which is
    /// more useful than being told it broke a rule.
    pub fn k_epsilon_equivalent(&self) -> (Scalar, Scalar) {
        (self.gamma_2 + 1.0, self.beta_2 / self.beta_star + 1.0)
    }

    /// The decay exponent `beta*/beta` these coefficients give homogeneous
    /// isotropic turbulence at a given `F_1`, the same quantity
    /// [`crate::models::KOmegaCoeffs::decay_exponent`] reports.
    ///
    /// At `F_1 = 1` it is `0.09/0.075 = 1.2`, Wilcox's; at `F_1 = 0` it is
    /// `0.09/0.0828 = 1.087`, which is `1/(C_2 - 1)` with `C_2 = 1.92` - the
    /// k-epsilon answer, as the transform requires.
    pub fn decay_exponent(&self, f1: Scalar) -> Scalar {
        let beta = f1 * self.beta_1 + (1.0 - f1) * self.beta_2;
        self.beta_star / beta
    }

    fn check(&self) -> Result<()> {
        for (name, v) in [
            ("betaStar", self.beta_star),
            ("beta1", self.beta_1),
            ("beta2", self.beta_2),
            ("a1", self.a1),
            ("sigmaW2", self.sigma_w2),
        ] {
            if !(v > 0.0) && v.is_finite() {
                return Err(Error::Config(format!(
                    "kOmegaSST: {name} = {v}; it divides or scales a positive \
                     quantity and must be positive"
                )));
            }
        }
        Ok(())
    }
}

// ==========================================================================
//  The model
// ==========================================================================

/// Menter k-omega SST, resident on the device.
pub struct KOmegaSst<'m> {
    core: RasCore<'m>,
    sst: SstKernels,
    fld: FieldKernels,
    coeffs: KOmegaSstCoeffs,

    k: GpuScalarField,
    omega: GpuScalarField,

    /// `[n_cells]` the wall distance of SPEC-LIT §6.6, copied in at
    /// construction. Owned rather than borrowed because it is computed once at
    /// setup and the model outlives the `WallDistance` that produced it.
    y: DevBuf<Scalar>,

    /// `[n_cells]` the two blending functions, and the four fields they blend
    /// the coefficient sets into.
    f1: DevBuf<Scalar>,
    f2: DevBuf<Scalar>,
    sigma_k: DevBuf<Scalar>,
    sigma_w: DevBuf<Scalar>,
    gamma_b: DevBuf<Scalar>,
    beta_b: DevBuf<Scalar>,

    /// `[n_cells]` `grad k` and `grad omega` - the cross-diffusion term and
    /// `arg1`'s third branch both need them, and they are the reason SST
    /// cannot be assembled from cell values alone.
    grad_k: DevBuf<Vec3>,
    grad_omega: DevBuf<Vec3>,

    /// `[n_cells]` the strain-rate magnitude `sqrt(S²)`, and the production
    /// per unit eddy viscosity `G/nu_t`.
    s: DevBuf<Scalar>,
    p: DevBuf<Scalar>,
    /// `[n_cells]` the limited production `min(G, c_1 beta* k omega)`.
    g_lim: DevBuf<Scalar>,

    f1_override: Option<Scalar>,
}

impl<'m> KOmegaSst<'m> {
    /// `wall_faces` carries the two independent face sets of SPEC-LIT §15.5,
    /// exactly as [`crate::models::KOmega::new`] does: which cells `omega`
    /// pins to the near-wall relation, from `omega`'s own patch types, and
    /// which faces `nu_t` gets a wall value on, from `nut`'s.
    ///
    /// `y` is the wall distance of SPEC-LIT §6.6 -
    /// [`crate::walldistance::wall_distance`] - and is copied, not borrowed.
    /// It must have `n_cells` entries; a model handed a shorter one would read
    /// off the end of it in every blending pass.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        mesh: &'m GpuMesh,
        coeffs: KOmegaSstCoeffs,
        ctrl: TurbulenceControls,
        wall: WallFunctionCoeffs,
        wall_faces: &crate::field_setup::WallFaces,
        y: &DevBuf<Scalar>,
    ) -> Result<Self> {
        coeffs.check()?;

        if y.len() != mesh.n_cells {
            return Err(Error::Config(format!(
                "kOmegaSST: the wall distance has {} entries, the mesh has {} \
                 cells (SPEC-LIT 6.6)",
                y.len(),
                mesh.n_cells
            )));
        }

        let nc = mesh.n_cells.max(1);
        let fld = FieldKernels::new(gpu)?;

        let mut y_own: DevBuf<Scalar> = gpu.zeros(nc)?;
        crate::field_ops::copy_field(gpu, &fld, &mut y_own, y, mesh.n_cells)?;

        Ok(Self {
            // No binary in this crate constructs a `KOmegaSst` from a case
            // file yet (every call site is a unit test), so there is no
            // case-file `Ks`/`Cs` to thread through here - see the matching
            // note in `models::les::Les::new`. `NutRoughness::none` makes
            // every face smooth, exactly SPEC-LIT §29.2's `Ks -> 0` gate.
            core: RasCore::new(
                gpu,
                hm,
                mesh,
                ctrl,
                wall,
                wall_faces,
                &crate::field_setup::NutRoughness::none(hm.n_boundary_faces),
            )?,
            sst: SstKernels::new(gpu)?,
            fld,
            coeffs,

            k: GpuScalarField::zeros(gpu, mesh, "k")?,
            omega: GpuScalarField::zeros(gpu, mesh, "omega")?,

            y: y_own,

            f1: gpu.zeros(nc)?,
            f2: gpu.zeros(nc)?,
            sigma_k: gpu.zeros(nc)?,
            sigma_w: gpu.zeros(nc)?,
            gamma_b: gpu.zeros(nc)?,
            beta_b: gpu.zeros(nc)?,

            grad_k: gpu.zeros(nc)?,
            grad_omega: gpu.zeros(nc)?,

            s: gpu.zeros(nc)?,
            p: gpu.zeros(nc)?,
            g_lim: gpu.zeros(nc)?,

            f1_override: None,
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
    pub fn nut_mut(&mut self) -> &mut GpuScalarField {
        &mut self.core.nut
    }
    /// `F_1`, the blending function: one in the inner layer, zero outside it.
    pub fn f1(&self) -> &DevBuf<Scalar> {
        &self.f1
    }
    /// `F_2`, which confines the eddy-viscosity limiter to a boundary layer.
    pub fn f2(&self) -> &DevBuf<Scalar> {
        &self.f2
    }
    /// The wall distance the model was built with.
    pub fn y(&self) -> &DevBuf<Scalar> {
        &self.y
    }
    pub fn coeffs(&self) -> &KOmegaSstCoeffs {
        &self.coeffs
    }

    /// `k`, `omega` and `nut`, named - see
    /// [`crate::models::KEpsilon::named_fields`] for why this is a method on
    /// the concrete model rather than a trait default.
    pub fn named_fields(&self) -> Vec<(&'static str, &GpuScalarField)> {
        vec![("k", &self.k), ("omega", &self.omega), ("nut", &self.core.nut)]
    }

    /// [`Self::named_fields`], mutable - for `0/` upload and `.mcr` restore.
    pub fn named_fields_mut(&mut self) -> Vec<(&'static str, &mut GpuScalarField)> {
        vec![
            ("k", &mut self.k),
            ("omega", &mut self.omega),
            ("nut", &mut self.core.nut),
        ]
    }
    pub fn core(&self) -> &RasCore<'m> {
        &self.core
    }
    pub fn core_mut(&mut self) -> &mut RasCore<'m> {
        &mut self.core
    }

    /// `nu_t = 0` everywhere and nothing to put it back - what
    /// `RAS { turbulence off; }` and `simulationType laminar;` ask for.
    pub fn freeze_nut(&mut self, gpu: &Gpu) -> Result<()> {
        self.core.freeze_nut(gpu)
    }

    /// Replace the computed `F_1` with a constant, or `None` to compute it.
    ///
    /// A test instrument, not a model switch - see the module header. `F_1`
    /// is still evaluated on every pass so that [`Self::f1`] would report the
    /// real value if the override were removed; only the copy the coefficients
    /// and the cross-diffusion term read is overwritten.
    pub fn force_f1(&mut self, f1: Option<Scalar>) {
        self.f1_override = f1;
    }

    /// Switch the buoyancy production `G_b` on - SPEC-LIT §17 and §30.2.
    ///
    /// See [`crate::models::KEpsilon::set_buoyancy`]. SST's `omega` equation
    /// takes it by the same production route k-omega does,
    /// `+ (gamma/nu_t) G_b`, but with `gamma` the per-cell blend `F1 gamma_1 +
    /// (1 - F1) gamma_2` rather than one constant - SPEC-LIT §6.3's `gamma_b`,
    /// the same field the shear production already reads.
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

    /// Bound the initial fields, evaluate their boundaries, build the blending
    /// functions and the first `nu_t`. Call once, after the initial fields are
    /// uploaded.
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

        self.core.update_flow_derived(gpu, flow)?;
        self.update_blending(gpu, flow)?;
        self.correct_nut(gpu, flow)?;
        self.core.store_k_prev(gpu, &self.k.f)?;

        Ok(())
    }

    /// `S`, `G/nu_t`, `grad k`, `grad omega`, `F_1`, `F_2` and the four
    /// blended coefficient fields.
    ///
    /// Assumes `grad U` is current, which is [`RasCore::update_flow_derived`]'s
    /// job; everything else here is rebuilt from scratch.
    fn update_blending(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        let n = self.core.mesh.n_cells;
        let c = self.coeffs;
        let grad_scheme = self.core.ctrl.grad_scheme;

        strain_rate_mag(gpu, &self.core.turb, &mut self.s, &self.core.grad_u, n)?;
        sst_production_by_nut(gpu, &self.sst, &mut self.p, &self.core.grad_u, n)?;

        fvc_grad_scalar_scheme(
            gpu,
            &self.core.fv,
            &mut self.grad_k,
            &self.k,
            self.core.mesh,
            grad_scheme,
        )?;
        fvc_grad_scalar_scheme(
            gpu,
            &self.core.fv,
            &mut self.grad_omega,
            &self.omega,
            self.core.mesh,
            grad_scheme,
        )?;

        sst_blending(
            gpu,
            &self.sst,
            &mut self.f1,
            &mut self.f2,
            &self.k.f,
            &self.omega.f,
            &self.grad_k,
            &self.grad_omega,
            &self.y,
            flow.nu,
            c.beta_star,
            c.sigma_w2,
            n,
        )?;

        if let Some(v) = self.f1_override {
            set_field(gpu, &self.fld, &mut self.f1, v, n)?;
        }

        sst_blend_coeffs(
            gpu,
            &self.sst,
            &mut self.sigma_k,
            &mut self.sigma_w,
            &mut self.gamma_b,
            &mut self.beta_b,
            &self.f1,
            &c,
            n,
        )?;

        Ok(())
    }

    // ---- one outer iteration ---------------------------------------------

    /// Solve `omega`, then `k`, then update `nu_t`.
    ///
    /// Returns `(omega, k)` performance in that order, matching
    /// [`crate::models::KOmega::correct`] so that a driver prints the same two
    /// columns whichever model it runs.
    pub fn correct(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        self.correct_buoyant(gpu, flow, None)
    }

    /// [`KOmegaSst::correct`] with the temperature the buoyancy production is
    /// built from - SPEC-LIT §17 and §30.2.
    ///
    /// `t` is read, never written, and is ignored unless
    /// [`KOmegaSst::set_buoyancy`] has been called with a non-zero gravity.
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
        // psi^{n-2} <- psi^{n-1} <- psi, once per step (SPEC-LIT 13.3), the
        // same rotation and the same caveat as `k_omega.rs`.
        advance_time_levels(gpu, &self.core.fld, &mut self.k)?;
        advance_time_levels(gpu, &self.core.fld, &mut self.omega)?;
        self.core.ddt.advance(ctrl.delta_t);

        self.core.update_flow_derived(gpu, flow)?;
        self.update_blending(gpu, flow)?;

        // G_b = (nu_t/Pr_t) g.grad(T)/T and its C_3 (SPEC-LIT 17), from the
        // same PREVIOUS nu_t the shear production G uses - identical to
        // k-epsilon's and k-omega's placement.
        let buoyant = match t {
            Some(tf) => self.core.update_buoyancy_production(gpu, tf, flow.u)?,
            None => false,
        };

        // Wall functions: nu_t on the wall faces from the current k, then
        // omega and G in the wall-adjacent cells. Identical to k-omega's - SST
        // changes the transport equations, not the near-wall relation.
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
        self.core.assemble_transport_blended(
            gpu,
            flow,
            &self.omega,
            ctrl.eps_conv(),
            &self.sigma_w,
        )?;

        // `P` rather than the wall-corrected `G/nu_t`: in a wall-adjacent cell
        // `constrain_wall_cells` replaces the whole row with the near-wall
        // relation, so whatever source was assembled there is discarded. Away
        // from a wall the two are the same number.
        sst_omega_sources(
            gpu,
            &self.sst,
            &mut self.core.su,
            &mut self.core.sp,
            &mut self.core.susp,
            &self.p,
            &self.omega.f,
            &self.grad_k,
            &self.grad_omega,
            &self.f1,
            &self.gamma_b,
            &self.beta_b,
            c.sigma_w2,
            n,
        )?;

        // + (gamma_b/nu_t) G_b, split by sign, ACCUMULATED into what
        // `sst_omega_sources` just wrote (SPEC-LIT 17, 30.2). `gamma_b` is the
        // SAME per-cell blend the shear production reads - not the k-omega
        // constant - because SST's omega equation has no single `gamma`
        // either. Unstable branch only unless the case asked for both,
        // matching k-epsilon and k-omega.
        if buoyant {
            let stable = self
                .core
                .buoyancy
                .map(|b| b.epsilon_stable_branch)
                .unwrap_or(false);
            let nut_min = 1e-30 as Scalar;
            let RasCore { turb, su, sp, gb, nut, .. } = &mut self.core;
            add_buoyancy_to_omega_cell(
                gpu,
                turb,
                su,
                sp,
                gb,
                &nut.f,
                &self.omega.f,
                &self.gamma_b,
                nut_min,
                stable,
                n,
            )?;
        }

        fvm_su(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.su, 1.0)?;
        fvm_sp(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.sp, 1.0)?;
        fvm_susp(
            gpu,
            &self.core.fv,
            &mut self.core.a,
            self.core.mesh,
            &self.core.susp,
            &self.omega.f,
            1.0,
        )?;

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
        self.core.assemble_transport_blended(
            gpu,
            flow,
            &self.k,
            ctrl.k_conv(),
            &self.sigma_k,
        )?;

        sst_k_sources(
            gpu,
            &self.sst,
            &mut self.g_lim,
            &mut self.core.sp,
            &mut self.core.susp,
            &self.core.g,
            &self.k.f,
            &self.omega.f,
            &self.core.div_u,
            c.beta_star,
            c.c1,
            n,
        )?;

        fvm_su(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.g_lim, 1.0)?;

        // + G_b, both signs (SPEC-LIT 17) - the same route into `k` every
        // model here uses, unrelated to which equation carries the
        // dissipation.
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

    /// `nu_t = a_1 k/max(a_1 omega, b_1 F_2 S)`, its boundary values, and the
    /// wall-function override.
    pub fn correct_nut(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        let n = self.core.mesh.n_cells;
        let ctrl = self.core.ctrl;
        let wall = self.core.wall;
        let c = self.coeffs;
        let nut_max = self.core.nut_max(flow.nu);

        sst_nut(
            gpu,
            &self.sst,
            &mut self.core.nut.f,
            &self.k.f,
            &self.omega.f,
            &self.f2,
            &self.s,
            c.a1,
            c.b1,
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
        let k = &self.k.f;
        self.core.convergence_measure(gpu, k)
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{GpuSurfaceScalarField, GpuVectorField};
    use crate::models::{KEpsilon, KEpsilonCoeffs, KOmega, KOmegaCoeffs};

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// A closed box, `nz = 1` so that the `empty` patches `box_mesh` puts on
    /// `zmin`/`zmax` describe a genuine two-dimensional case.
    ///
    /// That matters here and it did not for `k_omega.rs`. An `empty` face
    /// contributes nothing to a surface integral, so on a mesh several cells
    /// deep with `empty` ends the Green-Gauss gradient of even a UNIFORM field
    /// comes out at `psi/h` in the through-plane direction - the internal face
    /// on one side has nothing on the other side to cancel it. SST reads
    /// `grad k` and `grad omega` where the two-equation models do not, so it
    /// is the first model in this crate that can tell a closed mesh from an
    /// unclosed one, and a test mesh that is not closed would put a spurious
    /// cross-diffusion term into every cell.
    fn quiet_box() -> HostMesh {
        let (mut m, points, faces) =
            crate::mesh::topology::tests::box_mesh([4, 4, 1], Vec3::new(0.25, 0.25, 0.25));
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

    // ----------------------------------------------------------------------
    //  The coefficients
    // ----------------------------------------------------------------------

    /// The identity the whole blend rests on, and it is checkable with no
    /// mesh, no device and no flow.
    ///
    /// Substituting `epsilon = beta* k omega` into Launder & Spalding's
    /// epsilon equation gives `gamma_2 = C_1 - 1` and
    /// `beta_2 = beta*(C_2 - 1)`. If SPEC-LIT §6.3's tabulated `0.44` and
    /// `0.0828` did NOT satisfy those, "set 2 is the transformed k-epsilon"
    /// would be a claim about a model this code does not contain.
    #[test]
    fn set_two_is_the_transformed_k_epsilon() {
        let c = KOmegaSstCoeffs::default();
        let ke = KEpsilonCoeffs::default();

        assert!(
            (c.gamma_2 - (ke.c1 - 1.0)).abs() < 1e-12,
            "gamma_2 = {} but C_1 - 1 = {}",
            c.gamma_2,
            ke.c1 - 1.0
        );
        assert!(
            (c.beta_2 - c.beta_star * (ke.c2 - 1.0)).abs() < 1e-12,
            "beta_2 = {} but beta*(C_2 - 1) = {}",
            c.beta_2,
            c.beta_star * (ke.c2 - 1.0)
        );

        let (c1, c2) = c.k_epsilon_equivalent();
        assert!((c1 - 1.44).abs() < 1e-12, "{c1}");
        assert!((c2 - 1.92).abs() < 1e-12, "{c2}");

        // And the consequence: the two models decay at the same rate.
        assert!((c.decay_exponent(0.0) - 1.0 / (ke.c2 - 1.0)).abs() < 1e-12);
        // At F1 = 1 it is Wilcox's, with SST's own beta_1 = 0.075.
        assert!((c.decay_exponent(1.0) - 0.09 / 0.075).abs() < 1e-12);
    }

    /// Set 1 is the inner k-omega set, and it is NOT Wilcox's 1988 set - the
    /// two share `gamma` and differ in `beta` and `sigma_k`. Pinned because
    /// mixing them is the easiest possible mistake to make in this file, and
    /// it would be invisible: the model would still converge.
    #[test]
    fn the_two_coefficient_sets_are_the_tabulated_ones() {
        let c = KOmegaSstCoeffs::default();
        assert!((c.sigma_k1 - 0.85).abs() < 1e-15);
        assert!((c.sigma_w1 - 0.5).abs() < 1e-15);
        assert!((c.beta_1 - 0.075).abs() < 1e-15);
        assert!((c.gamma_1 - 5.0 / 9.0).abs() < 1e-15);

        assert!((c.sigma_k2 - 1.0).abs() < 1e-15);
        assert!((c.sigma_w2 - 0.856).abs() < 1e-15);
        assert!((c.beta_2 - 0.0828).abs() < 1e-15);
        assert!((c.gamma_2 - 0.44).abs() < 1e-15);

        assert!((c.beta_star - 0.09).abs() < 1e-15);
        assert!((c.a1 - 0.31).abs() < 1e-15);
        assert!((c.b1 - 1.0).abs() < 1e-15);
        assert!((c.c1 - 10.0).abs() < 1e-15);

        let w = KOmegaCoeffs::default();
        assert!(
            (c.beta_1 - w.beta).abs() > 1e-4,
            "SST's beta_1 has been overwritten with Wilcox's beta"
        );
    }

    // ----------------------------------------------------------------------
    //  The blending functions
    // ----------------------------------------------------------------------

    /// SPEC-LIT §22: "`F1 = 1` at a wall, `-> 0` in the free stream, monotone".
    ///
    /// Driven directly, with no mesh and no model: uniform `k` and `omega`, no
    /// gradients, and `y` swept over six decades. Both surviving branches of
    /// `arg1` - the turbulent length scale `sqrt(k)/(beta* omega y)` and the
    /// viscous `500 nu/(y² omega)` - fall monotonically with `y`, and the
    /// cross-diffusion branch is inactive because `grad k . grad omega` is
    /// zero and `CD_kw+` is at its floor. So `F1` must fall monotonically from
    /// one to zero, and there is nothing else it could be doing.
    #[test]
    fn f1_is_one_at_a_wall_and_zero_in_the_free_stream() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let c = KOmegaSstCoeffs::default();
        let kern = SstKernels::new(&gpu)?;

        let n = 64usize;
        let nu: Scalar = 1e-5;
        let k0: Scalar = 1e-2;
        let w0: Scalar = 10.0;

        // 1e-6 m to 10 m, geometric. Seven decades, because `arg1` falls like
        // `1/y` and `F1 = tanh(arg1^4)` therefore falls like `y^-4`: at one
        // metre it is still 1.5e-4, which is small but is not zero.
        let ys: Vec<Scalar> = (0..n)
            .map(|i| 1e-6 * 10.0_f64.powf(7.0 * i as f64 / (n - 1) as f64) as Scalar)
            .collect();

        let y = gpu.upload(&ys)?;
        let k = gpu.upload(&vec![k0; n])?;
        let w = gpu.upload(&vec![w0; n])?;
        let gk = gpu.upload(&vec![Vec3::ZERO; n])?;
        let gw = gpu.upload(&vec![Vec3::ZERO; n])?;

        let mut f1: DevBuf<Scalar> = gpu.zeros(n)?;
        let mut f2: DevBuf<Scalar> = gpu.zeros(n)?;

        sst_blending(
            &gpu, &kern, &mut f1, &mut f2, &k, &w, &gk, &gw, &y, nu, c.beta_star,
            c.sigma_w2, n,
        )?;
        gpu.sync()?;

        let f1 = gpu.download(&f1)?;
        let f2 = gpu.download(&f2)?;

        assert!(
            f1[0] > 1.0 - 1e-12,
            "F1 at y = {} is {}, not one at the wall",
            ys[0],
            f1[0]
        );
        assert!(
            *f1.last().unwrap_or(&1.0) < 1e-6,
            "F1 at y = {} is {}, not zero in the free stream",
            ys[n - 1],
            f1[n - 1]
        );

        for i in 1..n {
            assert!(
                f1[i] <= f1[i - 1] + 1e-15,
                "F1 rose between y = {} and y = {}: {} -> {}",
                ys[i - 1],
                ys[i],
                f1[i - 1],
                f1[i]
            );
            assert!((0.0..=1.0).contains(&f1[i]), "F1 = {} is out of range", f1[i]);
            assert!(f2[i] <= f2[i - 1] + 1e-15, "F2 rose");
            assert!((0.0..=1.0).contains(&f2[i]), "F2 = {} is out of range", f2[i]);
        }

        // F2's argument is the larger of the two - `2 sqrt(k)/(beta* omega y)`
        // against `sqrt(k)/(beta* omega y)` - and tanh is increasing, so F2
        // can never be below F1 where the turbulent branch is the active one.
        // It is also raised to the second power rather than the fourth, which
        // is what makes it the wider of the two functions.
        assert!(
            f2[n - 1] >= f1[n - 1],
            "F2 {} fell below F1 {} in the free stream",
            f2[n - 1],
            f1[n - 1]
        );

        Ok(())
    }

    /// The third branch of `arg1` is the one that carries the cross-diffusion
    /// information, and with it switched on `F1` must fall - that branch is a
    /// `min`, so it can only ever lower the argument.
    #[test]
    fn cross_diffusion_can_only_lower_f1() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let c = KOmegaSstCoeffs::default();
        let kern = SstKernels::new(&gpu)?;

        let n = 32usize;
        let nu: Scalar = 1e-5;
        let ys: Vec<Scalar> = (0..n)
            .map(|i| 1e-4 * 10.0_f64.powf(4.0 * i as f64 / (n - 1) as f64) as Scalar)
            .collect();

        let y = gpu.upload(&ys)?;
        let k = gpu.upload(&vec![1e-2 as Scalar; n])?;
        let w = gpu.upload(&vec![10.0 as Scalar; n])?;

        let zero = gpu.upload(&vec![Vec3::ZERO; n])?;
        let gk = gpu.upload(&vec![Vec3::new(1.0, 0.0, 0.0); n])?;
        let gw = gpu.upload(&vec![Vec3::new(1e3, 0.0, 0.0); n])?;

        let mut a: DevBuf<Scalar> = gpu.zeros(n)?;
        let mut b: DevBuf<Scalar> = gpu.zeros(n)?;
        let mut f2: DevBuf<Scalar> = gpu.zeros(n)?;

        sst_blending(
            &gpu, &kern, &mut a, &mut f2, &k, &w, &zero, &zero, &y, nu, c.beta_star,
            c.sigma_w2, n,
        )?;
        sst_blending(
            &gpu, &kern, &mut b, &mut f2, &k, &w, &gk, &gw, &y, nu, c.beta_star,
            c.sigma_w2, n,
        )?;
        gpu.sync()?;

        let a = gpu.download(&a)?;
        let b = gpu.download(&b)?;

        let mut lowered_somewhere = false;
        for i in 0..n {
            assert!(
                b[i] <= a[i] + 1e-14,
                "cell {i}: cross-diffusion RAISED F1, {} -> {}",
                a[i],
                b[i]
            );
            if b[i] < a[i] - 1e-6 {
                lowered_somewhere = true;
            }
        }
        assert!(
            lowered_somewhere,
            "the cross-diffusion branch never engaged, so this test measured nothing"
        );

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  The blend, against the two models it blends
    // ----------------------------------------------------------------------

    /// Decaying homogeneous isotropic turbulence, which is the state in which
    /// SST reduces EXACTLY to one of its two limbs: `U = 0` makes `S`, `G` and
    /// both gradients vanish, so the eddy-viscosity limiter is inactive, the
    /// production limiter is inactive, the cross-diffusion term is zero and
    /// the diffusivities multiply a zero laplacian. What is left is
    /// `dk/dt = -beta* k omega`, `domega/dt = -blend(beta) omega²`.
    ///
    /// Returns `k` at `t_end`.
    fn sst_decay(
        gpu: &Gpu,
        hm: &HostMesh,
        mesh: &GpuMesh,
        f1: Scalar,
        k0: Scalar,
        w0: Scalar,
        dt: Scalar,
        steps: usize,
    ) -> Result<(Scalar, Scalar)> {
        let u = GpuVectorField::zeros(gpu, mesh, "U")?;
        let phi = GpuSurfaceScalarField::zeros(gpu, mesh, "phi")?;
        let flow = FlowState::new(&u, &phi, 1e-3);
        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);

        // No wall in this test: a uniform distance is enough, because F1 is
        // forced and nothing else reads y.
        let y = gpu.upload(&vec![crate::walldistance::NO_WALL; mesh.n_cells])?;

        let mut model = KOmegaSst::new(
            gpu,
            hm,
            mesh,
            KOmegaSstCoeffs::default(),
            decay_controls(dt),
            WallFunctionCoeffs::default(),
            &no_walls,
            &y,
        )?;
        model.force_f1(Some(f1));

        gpu.write(&mut model.k_mut().f, &vec![k0; mesh.n_cells])?;
        gpu.write(&mut model.omega_mut().f, &vec![w0; mesh.n_cells])?;
        model.initialise(gpu, &flow)?;

        for _ in 0..steps {
            model.correct(gpu, &flow)?;
        }
        gpu.sync()?;

        let k = gpu.download(&model.k().f)?;
        let w = gpu.download(&model.omega().f)?;
        Ok((k[0], w[0]))
    }

    /// SPEC-LIT §22: "`F1 -> 1` everywhere forced -> reproduces k-omega".
    ///
    /// Against an actual `KOmega` run, not against an analytic curve, and with
    /// `KOmega` given SST's set-1 coefficients - which is the whole content of
    /// the claim. Wilcox's own `beta = 0.072` would NOT reproduce it, and the
    /// second half of the test checks that too, so a pass cannot be an
    /// accident of a loose tolerance.
    #[test]
    fn forcing_f1_to_one_reproduces_k_omega() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let hm = quiet_box();
        let mesh = GpuMesh::upload(&gpu, &hm)?;

        let (k0, w0): (Scalar, Scalar) = (1.0, 2.0);
        let dt: Scalar = 0.02;
        let steps = 1000;

        let (k_sst, w_sst) = sst_decay(&gpu, &hm, &mesh, 1.0, k0, w0, dt, steps)?;

        // k-omega with SST's set 1.
        let sst = KOmegaSstCoeffs::default();
        let coeffs = KOmegaCoeffs {
            beta_star: sst.beta_star,
            beta: sst.beta_1,
            gamma: sst.gamma_1,
            alpha_k: sst.sigma_k1,
            alpha_omega: sst.sigma_w1,
        };

        let u = GpuVectorField::zeros(&gpu, &mesh, "U")?;
        let phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi")?;
        let flow = FlowState::new(&u, &phi, 1e-3);
        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let no_roughness = crate::field_setup::NutRoughness::none(hm.n_boundary_faces);

        let mut kw = KOmega::new(
            &gpu,
            &hm,
            &mesh,
            coeffs,
            decay_controls(dt),
            WallFunctionCoeffs::default(),
            &no_walls,
            &no_roughness,
        )?;
        gpu.write(&mut kw.k_mut().f, &vec![k0; mesh.n_cells])?;
        gpu.write(&mut kw.omega_mut().f, &vec![w0; mesh.n_cells])?;
        kw.initialise(&gpu, &flow)?;
        for _ in 0..steps {
            kw.correct(&gpu, &flow)?;
        }
        gpu.sync()?;

        let k_ref = gpu.download(&kw.k().f)?[0];
        let w_ref = gpu.download(&kw.omega().f)?[0];

        assert!(
            (k_sst - k_ref).abs() <= 1e-11 * k_ref,
            "SST with F1 = 1 gave k = {k_sst}, k-omega with set 1 gave {k_ref}"
        );
        assert!(
            (w_sst - w_ref).abs() <= 1e-11 * w_ref,
            "SST with F1 = 1 gave omega = {w_sst}, k-omega with set 1 gave {w_ref}"
        );

        // And the control: Wilcox's own beta is a different model, and this
        // test would not be measuring anything if it could not tell them apart.
        let k_wilcox = k0
            * (1.0 + KOmegaCoeffs::default().beta * w0 * dt * steps as Scalar)
                .powf(-KOmegaCoeffs::default().decay_exponent());
        assert!(
            (k_sst - k_wilcox).abs() > 1e-3 * k_sst,
            "set 1 and Wilcox's 1988 set are indistinguishable here, so the \
             tolerance above is measuring nothing"
        );

        Ok(())
    }

    /// SPEC-LIT §17 and §30.2: SST's buoyancy production must collapse onto
    /// k-omega's when `F1 = 1`, because `gamma_b` is then exactly `gamma_1`
    /// and the two models are otherwise identical at that limit (see
    /// `forcing_f1_to_one_reproduces_k_omega`). A separate implementation of
    /// the same term that happened to agree at `F1 = 1` by accident would be
    /// a much less useful thing to have proven.
    #[test]
    fn forcing_f1_to_one_reproduces_k_omega_with_buoyancy() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let hm = quiet_box();
        let mesh = GpuMesh::upload(&gpu, &hm)?;
        let n = mesh.n_cells;

        // T varies with j (the y direction) only: cell c -> j = c/4 on this
        // 4x4x1 mesh (SPEC-LIT's i-fastest cell ordering, `box_mesh`'s own
        // `cell(i,j,k) = i + nx*(j + ny*k)`).
        let t_vals: Vec<Scalar> = (0..n).map(|c| 300.0 + 10.0 * ((c / 4) as Scalar)).collect();
        let build_t = |gpu: &Gpu, mesh: &GpuMesh| -> Result<GpuScalarField> {
            let mut t = GpuScalarField::zeros(gpu, mesh, "T")?;
            gpu.write(&mut t.f, &t_vals)?;
            let fld = FieldKernels::new(gpu)?;
            correct_boundary_conditions(gpu, &fld, &mut t, mesh)?;
            Ok(t)
        };

        let buoy = BuoyancyProduction {
            g: Vec3::new(0.0, -9.81, 0.0),
            prt: 0.85,
            c3: crate::turbulence::C3Mode::Constant(0.0),
            epsilon_stable_branch: true,
            t_min: 1.0,
        };

        let (k0, w0): (Scalar, Scalar) = (1.0, 2.0);
        let dt: Scalar = 0.02;
        let steps = 200;

        let (k_sst, w_sst) = {
            let u = GpuVectorField::zeros(&gpu, &mesh, "U")?;
            let phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi")?;
            let flow = FlowState::new(&u, &phi, 1e-3);
            let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
            let y = gpu.upload(&vec![crate::walldistance::NO_WALL; n])?;

            let mut model = KOmegaSst::new(
                &gpu,
                &hm,
                &mesh,
                KOmegaSstCoeffs::default(),
                decay_controls(dt),
                WallFunctionCoeffs::default(),
                &no_walls,
                &y,
            )?;
            model.force_f1(Some(1.0));
            model.set_buoyancy(buoy)?;

            gpu.write(&mut model.k_mut().f, &vec![k0; n])?;
            gpu.write(&mut model.omega_mut().f, &vec![w0; n])?;
            model.initialise(&gpu, &flow)?;

            let t = build_t(&gpu, &mesh)?;
            for _ in 0..steps {
                model.correct_buoyant(&gpu, &flow, Some(&t))?;
            }
            gpu.sync()?;
            (
                gpu.download(&model.k().f)?[0],
                gpu.download(&model.omega().f)?[0],
            )
        };

        let (k_ref, w_ref) = {
            let sst = KOmegaSstCoeffs::default();
            let coeffs = KOmegaCoeffs {
                beta_star: sst.beta_star,
                beta: sst.beta_1,
                gamma: sst.gamma_1,
                alpha_k: sst.sigma_k1,
                alpha_omega: sst.sigma_w1,
            };
            let u = GpuVectorField::zeros(&gpu, &mesh, "U")?;
            let phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi")?;
            let flow = FlowState::new(&u, &phi, 1e-3);
            let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
            let no_roughness = crate::field_setup::NutRoughness::none(hm.n_boundary_faces);

            let mut kw = KOmega::new(
                &gpu,
                &hm,
                &mesh,
                coeffs,
                decay_controls(dt),
                WallFunctionCoeffs::default(),
                &no_walls,
                &no_roughness,
            )?;
            kw.set_buoyancy(buoy)?;
            gpu.write(&mut kw.k_mut().f, &vec![k0; n])?;
            gpu.write(&mut kw.omega_mut().f, &vec![w0; n])?;
            kw.initialise(&gpu, &flow)?;

            let t = build_t(&gpu, &mesh)?;
            for _ in 0..steps {
                kw.correct_buoyant(&gpu, &flow, Some(&t))?;
            }
            gpu.sync()?;
            (gpu.download(&kw.k().f)?[0], gpu.download(&kw.omega().f)?[0])
        };

        assert!(
            (k_sst - k_ref).abs() <= 1e-10 * k_ref.abs().max(1.0),
            "SST F1=1 with buoyancy gave k = {k_sst}, k-omega gave {k_ref}"
        );
        assert!(
            (w_sst - w_ref).abs() <= 1e-10 * w_ref.abs().max(1.0),
            "SST F1=1 with buoyancy gave omega = {w_sst}, k-omega gave {w_ref}"
        );

        // The control: buoyancy must have actually changed something, or the
        // agreement above would be measuring nothing.
        let (k_sst_no_b, _) = sst_decay(&gpu, &hm, &mesh, 1.0, k0, w0, dt, steps)?;
        assert!(
            (k_sst - k_sst_no_b).abs() > 1e-6 * k_sst_no_b.abs().max(1.0),
            "buoyancy made no difference to k; the term is not being applied"
        );

        Ok(())
    }

    /// SPEC-LIT §22: "`F1 -> 0` everywhere forced -> reproduces the
    /// transformed k-epsilon".
    ///
    /// Against an actual `KEpsilon` run. The two are started from the same
    /// physical state - `epsilon_0 = beta* k_0 omega_0` is the definition that
    /// makes the transform a transform - and are then integrated
    /// independently. They solve different equations in different variables,
    /// so they agree to time-discretisation error rather than to round-off,
    /// and the tolerance says so.
    #[test]
    fn forcing_f1_to_zero_reproduces_the_transformed_k_epsilon() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let hm = quiet_box();
        let mesh = GpuMesh::upload(&gpu, &hm)?;

        let sst = KOmegaSstCoeffs::default();
        let (k0, w0): (Scalar, Scalar) = (1.0, 2.0);
        let e0 = sst.beta_star * k0 * w0;

        let dt: Scalar = 0.01;
        let steps = 2000;
        let t_end = dt * steps as Scalar;

        let (k_sst, w_sst) = sst_decay(&gpu, &hm, &mesh, 0.0, k0, w0, dt, steps)?;

        let u = GpuVectorField::zeros(&gpu, &mesh, "U")?;
        let phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi")?;
        let flow = FlowState::new(&u, &phi, 1e-3);
        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let no_roughness = crate::field_setup::NutRoughness::none(hm.n_boundary_faces);

        let mut ke = KEpsilon::new(
            &gpu,
            &hm,
            &mesh,
            KEpsilonCoeffs::default(),
            decay_controls(dt),
            WallFunctionCoeffs::default(),
            &no_walls,
            &no_roughness,
        )?;
        gpu.write(&mut ke.k_mut().f, &vec![k0; mesh.n_cells])?;
        gpu.write(&mut ke.epsilon_mut().f, &vec![e0; mesh.n_cells])?;
        ke.initialise(&gpu, &flow)?;
        for _ in 0..steps {
            ke.correct(&gpu, &flow)?;
        }
        gpu.sync()?;

        let k_ref = gpu.download(&ke.k().f)?[0];
        let e_ref = gpu.download(&ke.epsilon().f)?[0];

        // 0.5 %. The two integrate DIFFERENT equations in different variables
        // - `omega` against `epsilon` - with the same step, so their
        // time-discretisation errors are different sizes and only the
        // continuous limits coincide. At this step each is within 0.2 % of the
        // analytic curve asserted at the bottom of this test, and they are
        // within 0.12 % of each other; halving `dt` halves both.
        assert!(
            (k_sst - k_ref).abs() < 5e-3 * k_ref,
            "SST with F1 = 0 gave k = {k_sst}, k-epsilon gave {k_ref}"
        );

        // And in the epsilon variable, through the transform that defines it.
        let e_sst = sst.beta_star * k_sst * w_sst;
        assert!(
            (e_sst - e_ref).abs() < 1e-2 * e_ref,
            "beta* k omega = {e_sst} from SST, epsilon = {e_ref} from k-epsilon"
        );

        // The analytic decay law both are supposed to be following, so that a
        // shared bug in the two integrators cannot make them agree.
        let n_exp = sst.decay_exponent(0.0);
        let k_exact = k0 * (1.0 + sst.beta_2 * w0 * t_end).powf(-n_exp);
        assert!(
            (k_sst - k_exact).abs() < 0.02 * k_exact,
            "k({t_end}) = {k_sst}, the transformed k-epsilon decay law gives {k_exact}"
        );

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  The eddy viscosity
    // ----------------------------------------------------------------------

    /// `nu_t = a_1 k/max(a_1 omega, b_1 F_2 S)` - both branches, and the point
    /// at which it switches.
    ///
    /// With `S = 0` it must be exactly `k/omega`, which is what makes SST a
    /// k-omega model at all; with `S` large it must be `a_1 k/(b_1 F_2 S)`,
    /// which is Bradshaw's `tau = a_1 k`.
    #[test]
    fn nut_is_shear_limited_only_where_the_shear_is_large() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let c = KOmegaSstCoeffs::default();
        let kern = SstKernels::new(&gpu)?;

        let (k0, w0): (Scalar, Scalar) = (0.44, 7.5);
        let f2v: Scalar = 1.0;

        let n = 3usize;
        let ss: Vec<Scalar> = vec![0.0, 0.5 * c.a1 * w0 / (c.b1 * f2v), 100.0 * w0];

        let k = gpu.upload(&vec![k0; n])?;
        let w = gpu.upload(&vec![w0; n])?;
        let f2 = gpu.upload(&vec![f2v; n])?;
        let s = gpu.upload(&ss)?;
        let mut nut: DevBuf<Scalar> = gpu.zeros(n)?;

        sst_nut(&gpu, &kern, &mut nut, &k, &w, &f2, &s, c.a1, c.b1, 1e30, n)?;
        gpu.sync()?;
        let nut = gpu.download(&nut)?;

        let unlimited = k0 / w0;
        assert!(
            (nut[0] - unlimited).abs() <= 1e-14 * unlimited,
            "at S = 0, nu_t = {} and k/omega = {unlimited}",
            nut[0]
        );
        assert!(
            (nut[1] - unlimited).abs() <= 1e-14 * unlimited,
            "below the switch the limiter must be inert, got {}",
            nut[1]
        );

        let limited = c.a1 * k0 / (c.b1 * f2v * ss[2]);
        assert!(
            (nut[2] - limited).abs() <= 1e-14 * limited,
            "above the switch, nu_t = {} and a1 k/(b1 F2 S) = {limited}",
            nut[2]
        );
        assert!(nut[2] < unlimited, "the limiter did not limit anything");

        Ok(())
    }

    /// A wall distance of the wrong length is a mistake that would otherwise
    /// read off the end of the buffer in every blending pass.
    #[test]
    fn a_mismatched_wall_distance_is_an_error() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let hm = quiet_box();
        let mesh = GpuMesh::upload(&gpu, &hm)?;
        let no_walls = crate::field_setup::WallFaces::none(hm.n_boundary_faces);
        let short: DevBuf<Scalar> = gpu.zeros(hm.n_cells - 1)?;

        assert!(KOmegaSst::new(
            &gpu,
            &hm,
            &mesh,
            KOmegaSstCoeffs::default(),
            decay_controls(1e-3),
            WallFunctionCoeffs::default(),
            &no_walls,
            &short,
        )
        .is_err());

        Ok(())
    }

}
