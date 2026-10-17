// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The CONSERVATIVE rho-weighted Spalart-Allmaras form - SPEC-LIT §127.1.
//!
//! Written from:
//!   NASA / TMBWG, *Turbulence Modeling Resource - The Spalart-Allmaras
//!     Turbulence Model*, <https://tmbwg.github.io/turbmodels/spalart.html> -
//!     US government-authored DOCUMENTATION, not source, **read**. Its
//!     "conservative form" paragraph gives (127.1), constructed by combining
//!     SA with the mass conservation equation; the terms, the constants and
//!     the negative continuation are §56's, and `mu_t = rho nu~ f_v1`.
//!   ofgpu `SPEC-LIT.md` §127.1, and §127.1.1 for the discretisation as
//!     built - including the `f_n` form of the cross term, which is DERIVED
//!     there from the product rule, not quoted.
//!
//! The equation transports `rho nu~` on the mass flux `mdot`, with the
//! per-cell `nu_c = mu/rho` everywhere the kinematic model reads its
//! constant `nu`. At `rho = 1` with `mu = nu` and `mdot = phi` it is the
//! kinematic `correct` BIT FOR BIT: every kernel here computes the kinematic
//! value first and multiplies by `rho` afterwards, never reassociating, and
//! `x*1` is exact - which is what `rho_one_is_bitwise_kinematic` gates.
//!
//! Refused by name (SPEC-LIT §127.1): a `bounded` convection entry, because
//! `div(mdot)` is `-ddt(rho)` in a compressible flow (§123.1/§125.1) and the
//! `bounded` correction would subtract exactly the term (127.1) keeps; an
//! attached DES length scale, whose fixed-point update is defined against
//! the kinematic solve; and wrong buffer lengths. `localEuler` is refused by
//! [`crate::timescheme::Ddt::add_rho`] (§86.7) and that error is let through.

use cudarc::driver::PushKernelArg;

use crate::device::{cfg_for, DevBuf, Gpu};
use crate::error::{Error, Result};
use crate::field::GpuScalarField;
use crate::field_ops::{advance_time_levels, correct_boundary_conditions};
use crate::fv::{
    div_scheme_weights, fvc_grad_scalar_scheme, fvm_div_correction, fvm_div_gauss, fvm_laplacian,
    fvm_laplacian_non_orth_correction, fvm_su, fvm_susp, DivScheme,
};
use crate::mesh::GpuMesh;
use crate::solver::SolverPerformance;
use crate::turbulence::{nut_boundary, FlowState};
use crate::{Label, Scalar, Vec3};

use super::{bound_nu_tilda, expect_len, SaCoeffs, SaKernels, SpalartAllmaras};

/// The two fields the conservative form adds to [`SpalartAllmaras`]: the
/// density, at all three time levels plus the boundary, and the dynamic
/// viscosity `mu`, whose per-cell quotient by `rho` is §127.1's `nu`.
///
/// `flow.nu` is still read - for `nut`'s cap ONLY, exactly as the kinematic
/// model reads it; nothing else in the conservative path sees it.
pub struct SaConservative<'a> {
    /// `rho^n`, `rho^{n-1}`, `rho^{n-2}` and the boundary values - the four
    /// levels `Ddt::add_rho` and the face diffusivity read.
    pub rho: &'a GpuScalarField,
    /// `mu` in the cells and on the boundary faces. `nu_c = mu/rho` per
    /// cell, and per boundary face.
    pub mu: &'a GpuScalarField,
}

// ==========================================================================
//  The launch wrappers - the shape of `sa_nut`, `sa_sources` and
//  `sa_face_diffusivity`, against §127.1's four kernels
// ==========================================================================

/// `nu_t = nu~ f_v1(nu~/nu_c)`, exactly zero where `nu~ < 0`, capped at
/// `nut_max` - SPEC-LIT (56.1)/(56.13) with the per-cell `nu_c = mu/rho`.
#[allow(clippy::too_many_arguments)]
fn sa_nut_rho(
    gpu: &Gpu,
    kern: &SaKernels,
    nut: &mut DevBuf<Scalar>,
    nu_tilda: &DevBuf<Scalar>,
    mu: &DevBuf<Scalar>,
    rho: &DevBuf<Scalar>,
    cv1: Scalar,
    nut_max: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(nut, n, "nut")?;
    expect_len(nu_tilda, n, "nuTilda")?;
    expect_len(mu, n, "mu")?;
    expect_len(rho, n, "rho")?;

    let nl = n as Label;
    let f = kern.nut_rho.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(nut)
            .arg(nu_tilda)
            .arg(mu)
            .arg(rho)
            .arg(&cv1)
            .arg(&nut_max)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// SPEC-LIT (56.15) under a rho weight, plus (127.1)'s cross term: `su =
/// rho*su_kin - G grad rho . grad nu~` with `su_kin` EXACTLY what
/// `sa_sources` forms, `sp = 0`, and `susp = rho*susp_kin`. `d` is the true
/// wall distance; `d_tilde` is §57's hybrid length scale, which the
/// conservative form refuses (the fixed point is defined against the
/// kinematic solve), so a pure RANS run passes `y` for both.
#[allow(clippy::too_many_arguments)]
fn sa_sources_rho(
    gpu: &Gpu,
    kern: &SaKernels,
    su: &mut DevBuf<Scalar>,
    sp: &mut DevBuf<Scalar>,
    susp: &mut DevBuf<Scalar>,
    nu_tilda: &DevBuf<Scalar>,
    grad_nu_tilda: &DevBuf<Vec3>,
    grad_rho: &DevBuf<Vec3>,
    omega: &DevBuf<Scalar>,
    d: &DevBuf<Scalar>,
    d_tilde: &DevBuf<Scalar>,
    mu: &DevBuf<Scalar>,
    rho: &DevBuf<Scalar>,
    c: &SaCoeffs,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(su, n, "su")?;
    expect_len(sp, n, "sp")?;
    expect_len(susp, n, "susp")?;
    expect_len(nu_tilda, n, "nuTilda")?;
    expect_len(grad_nu_tilda, n, "grad nuTilda")?;
    expect_len(grad_rho, n, "grad rho")?;
    expect_len(omega, n, "Omega")?;
    expect_len(d, n, "d")?;
    expect_len(d_tilde, n, "dTilde")?;
    expect_len(mu, n, "mu")?;
    expect_len(rho, n, "rho")?;

    let nl = n as Label;
    let cw1 = c.cw1();
    let ct3_pos = c.ct3_positive();
    let f = kern.sources_rho.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(su)
            .arg(sp)
            .arg(susp)
            .arg(nu_tilda)
            .arg(grad_nu_tilda)
            .arg(grad_rho)
            .arg(omega)
            .arg(d)
            .arg(d_tilde)
            .arg(mu)
            .arg(rho)
            .arg(&c.cb1)
            .arg(&c.cb2)
            .arg(&c.cv1)
            .arg(&c.cv2)
            .arg(&c.cv3)
            .arg(&cw1)
            .arg(&c.cw2)
            .arg(&c.cw3)
            .arg(&ct3_pos)
            .arg(&c.ct3)
            .arg(&c.ct4)
            .arg(&c.sigma)
            .arg(&c.kappa)
            .arg(&c.rlim)
            .arg(&c.cn1)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `Gamma_c |Sf| = (rho (nu_c + nu~ f_n))/sigma` on every face, from the
/// TRANSPORTED field and §127.1's per-cell `nu_c = mu/rho` - the twin of
/// [`super::sa_face_diffusivity`], SPEC-LIT §56.6/§127.1.
#[allow(clippy::too_many_arguments)]
fn sa_face_diffusivity_rho(
    gpu: &Gpu,
    kern: &SaKernels,
    gamma: &mut DevBuf<Scalar>,
    b_gamma: &mut DevBuf<Scalar>,
    nu_tilda: &GpuScalarField,
    mu: &GpuScalarField,
    rho: &GpuScalarField,
    m: &GpuMesh,
    sigma: Scalar,
    cn1: Scalar,
) -> Result<()> {
    expect_len(gamma, m.n_internal_faces, "gamma")?;
    expect_len(b_gamma, m.n_boundary_faces, "b_gamma")?;

    if m.n_internal_faces > 0 {
        let n = m.n_internal_faces;
        let nl = n as Label;
        let f = kern.gamma_internal_rho.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(gamma)
                .arg(&nu_tilda.f)
                .arg(&mu.f)
                .arg(&rho.f)
                .arg(&m.weights)
                .arg(&m.mag_sf)
                .arg(&m.owner)
                .arg(&m.neighbour)
                .arg(&sigma)
                .arg(&cn1)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    if m.n_boundary_faces > 0 {
        let n = m.n_boundary_faces;
        let nl = n as Label;
        let f = kern.gamma_boundary_rho.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(b_gamma)
                .arg(&nu_tilda.bf)
                .arg(&mu.bf)
                .arg(&rho.bf)
                .arg(&m.b_mag_sf)
                .arg(&sigma)
                .arg(&cn1)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    Ok(())
}

// ==========================================================================
//  The model's conservative step
// ==========================================================================

impl<'m> SpalartAllmaras<'m> {
    /// `nu_t = nu~ f_v1(nu~/nu_c)` from the NEW `nu~`, then the boundary
    /// values - the mirror of `correct_nut` with §127.1's per-cell `nu_c`.
    /// `flow.nu` is read for the CAP only; `mu_t = rho nu_t` is the caller's
    /// product, exactly as §127.1 writes it.
    pub fn correct_nut_conservative(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        cons: &SaConservative<'_>,
    ) -> Result<()> {
        let n = self.core.mesh.n_cells;
        let nut_max = self.core.nut_max(flow.nu);

        sa_nut_rho(
            gpu,
            &self.kern,
            &mut self.core.nut.f,
            &self.nu_tilda.f,
            &cons.mu.f,
            &cons.rho.f,
            self.coeffs.cv1,
            nut_max,
            n,
        )?;

        correct_boundary_conditions(gpu, &self.core.fld, &mut self.core.nut, self.core.mesh)?;
        nut_boundary(gpu, &self.core.turb, &mut self.core.nut, self.core.mesh)
    }

    /// One outer step of (127.1): assemble and solve `rho nu~` on the mass
    /// flux, then update `nu_t`.
    ///
    /// `flow.phi` must already BE the mass flux `mdot` - the same contract
    /// `RasCore::assemble_transport_mass_weighted` states, and for the same
    /// reason (§86.2): the flux is one object shared with the equation that
    /// already convects with it. `flow.nu` is read for `nut`'s cap ONLY.
    ///
    /// Returns the same `(dissipation, k)` pair shape `correct` returns - SA
    /// has ONE equation, so the same record comes back twice.
    pub fn correct_conservative(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        cons: &SaConservative<'_>,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        let n = self.core.mesh.n_cells;
        let nbf = self.core.mesh.n_boundary_faces;
        let ctrl = self.core.ctrl;
        let c = self.coeffs;

        // ---- refused by name, BEFORE anything is touched ------------------
        if self.des.is_some() {
            return Err(Error::Config(
                "SpalartAllmaras: the conservative form (SPEC-LIT §127.1) is \
                 refused with an attached DES length scale (`DES { ... }`): \
                 §57's fixed-point update is defined against the kinematic \
                 solve, and d~ would lag a `rho nu~` level it never saw. \
                 Detach DES, or run the kinematic `correct`."
                    .to_string(),
            ));
        }
        if ctrl.eps_conv().bounded {
            return Err(Error::Config(
                "SpalartAllmaras: the conservative form (SPEC-LIT §127.1) \
                 refuses a `bounded` convection entry \
                 (`eps_bounded_convection`): div(mdot) is -ddt(rho) in a \
                 compressible flow (SPEC-LIT §123.1/§125.1), so the spurious \
                 source `bounded` subtracts is exactly the term (127.1) \
                 keeps. Set the entry unbounded."
                    .to_string(),
            ));
        }
        for (b, what) in [
            (&cons.rho.f, "rho"),
            (&cons.rho.f0, "rho0"),
            (&cons.rho.f00, "rho00"),
        ] {
            expect_len(b, n, what)?;
        }
        expect_len(&cons.rho.bf, nbf, "rho boundary")?;
        expect_len(&cons.mu.f, n, "mu")?;
        expect_len(&cons.mu.bf, nbf, "mu boundary")?;

        self.core.store_k_prev(gpu, &self.nu_tilda.f)?;
        advance_time_levels(gpu, &self.core.fld, &mut self.nu_tilda)?;
        self.core.ddt.advance(ctrl.delta_t);

        self.core.update_flow_derived(gpu, flow)?;
        self.update_invariants(gpu)?;

        // `grad nu~` for the `c_b2` term and the scheme corrections, and -
        // new here - `grad rho` for the cross term, both under
        // `ctrl.grad_scheme`.
        fvc_grad_scalar_scheme(
            gpu,
            &self.core.fv,
            &mut self.grad_nu_tilda,
            &self.nu_tilda,
            self.core.mesh,
            ctrl.grad_scheme,
        )?;
        fvc_grad_scalar_scheme(
            gpu,
            &self.core.fv,
            &mut self.grad_rho,
            cons.rho,
            self.core.mesh,
            ctrl.grad_scheme,
        )?;

        // ---- the diffusivity, then the rest of the assembly ---------------
        // The shared core's assembly tail is private to it (turbulence.rs is
        // another lane's file), so this reproduces
        // `RasCore::assemble_after_diffusivity` step by step with the `mass
        // = Some` ddt, into the model's own face buffers. `a` is zeroed
        // first, exactly as every `assemble_transport_*` entry does.
        self.core.a.zero(gpu)?;
        sa_face_diffusivity_rho(
            gpu,
            &self.kern,
            &mut self.gamma_c,
            &mut self.b_gamma_c,
            &self.nu_tilda,
            cons.mu,
            cons.rho,
            self.core.mesh,
            c.sigma,
            c.cn1,
        )?;

        let scheme: DivScheme = ctrl.eps_conv().scheme.into();

        // A limited or deferred-correction scheme reads the transported
        // field's gradient. The kinematic assembly computes it into a
        // private `grad_psi` here; `grad nu~` above is THE SAME kernel on
        // THE SAME unchanged field, so the same buffer is reused and the
        // values are bitwise what the kinematic path would have built.
        let grad_psi: Option<&DevBuf<Vec3>> = if scheme.needs_gradient() {
            Some(&self.grad_nu_tilda)
        } else {
            None
        };

        div_scheme_weights(
            gpu,
            &self.core.fv,
            Some(&mut self.wts_c),
            Some(&mut self.b_wts_c),
            scheme,
            flow.phi,
            &self.nu_tilda,
            grad_psi,
            self.core.mesh,
        )?;

        // `localEuler` needs the local step rebuilt from the flux this
        // iteration produced (SPEC-LIT 13.2); every other scheme ignores
        // this.
        self.core.ddt.update_local_step(gpu, flow.phi, self.core.mesh)?;

        // SPEC-LIT §86.3: `ddt(rho, nu~)`, each old level carrying its OWN
        // density, which is what makes the discrete term conserve `rho nu~`
        // rather than `nu~`. `localEuler` is refused by name here (§86.7)
        // and that error is let through.
        self.core.ddt.add_rho(
            gpu,
            &mut self.core.a,
            self.core.mesh,
            &cons.rho.f,
            &cons.rho.f0,
            &cons.rho.f00,
            &self.nu_tilda.f0,
            &self.nu_tilda.f00,
            1.0,
        )?;

        fvm_div_gauss(
            gpu,
            &self.core.fv,
            &mut self.core.a,
            self.core.mesh,
            flow.phi,
            &self.wts_c,
            &self.b_wts_c,
            &self.nu_tilda,
            1.0,
        )?;

        // A `bounded` entry was refused at the top (it would subtract the
        // -ddt(rho) half the conservative form exists to keep), so §3.1's
        // correction has no branch here.

        // The explicit half of `linearUpwind`/`cubic` (SPEC-LIT §11.1).
        if scheme.correction().is_some() {
            fvm_div_correction(
                gpu,
                &self.core.fv,
                &mut self.core.a,
                self.core.mesh,
                flow.phi,
                &self.grad_nu_tilda,
                scheme,
                1.0,
            )?;
        }

        fvm_laplacian(
            gpu,
            &self.core.fv,
            &mut self.core.a,
            self.core.mesh,
            &self.gamma_c,
            &self.b_gamma_c,
            &self.nu_tilda,
            -1.0,
        )?;

        // The kinematic path recomputes `grad_psi` before this term; it is
        // unchanged, so the same buffer serves again.
        if self.core.ctrl.sn_grad.applies() {
            fvm_laplacian_non_orth_correction(
                gpu,
                &self.core.fv,
                &mut self.core.a,
                self.core.mesh,
                &self.gamma_c,
                &self.b_gamma_c,
                &self.nu_tilda,
                &self.grad_nu_tilda,
                self.core.ctrl.sn_grad,
                -1.0,
            )?;
        }

        // ---- the sources ---------------------------------------------------
        // §56.6/(56.15) under a rho weight, with `d` and `d_tilde` handed in
        // separately; DES was refused at the top, so `d_tilde` is `y` here
        // always - destructured so the borrow checker can see that
        // `su`/`sp`/`susp` and `y`/`omega`/the two gradients are disjoint.
        {
            let Self {
                core,
                kern,
                nu_tilda,
                y,
                omega,
                grad_nu_tilda,
                grad_rho,
                ..
            } = self;
            let crate::turbulence::RasCore { su, sp, susp, .. } = core;
            sa_sources_rho(
                gpu,
                kern,
                su,
                sp,
                susp,
                &nu_tilda.f,
                grad_nu_tilda,
                grad_rho,
                omega,
                y,
                y,
                &cons.mu.f,
                &cons.rho.f,
                &c,
                n,
            )?;
        }

        fvm_su(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.su, 1.0)?;
        fvm_susp(
            gpu,
            &self.core.fv,
            &mut self.core.a,
            self.core.mesh,
            &self.core.susp,
            &self.nu_tilda.f,
            1.0,
        )?;

        // §56.6: `constrain_walls = false`. There is no wall function on
        // `nu~` to pin a near-wall cell to.
        let sc = ctrl.epsilon_solver;
        let perf = self
            .core
            .solve_equation(gpu, &mut self.nu_tilda, ctrl.eps_relax, &sc, false)?;

        if !c.variant.negative() {
            bound_nu_tilda(gpu, &self.kern, &mut self.nu_tilda.f, n)?;
        }
        correct_boundary_conditions(gpu, &self.core.fld, &mut self.nu_tilda, self.core.mesh)?;

        self.correct_nut_conservative(gpu, flow, cons)?;

        Ok((perf, perf))
    }
}
