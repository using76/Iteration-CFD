// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Marangoni convection - the tangential interfacial stress, SPEC-LIT §87.
//!
//! Written from:
//!   J. U. Brackbill, D. B. Kothe, C. Zemach, *J. Comput. Phys.* 100 (1992)
//!     335-354 - the continuum-surface-force regularisation. §20.4 took the
//!     normal half of their force; this module takes the other half.
//!   C. Ma, D. Bothe, *Int. J. Multiphase Flow* 37 (2011) 1045-1058 - the
//!     tangential stress in a VOF code, and what the split costs
//!   N. O. Young, J. S. Goldstein, M. J. Block, *J. Fluid Mech.* 6 (1959)
//!     350-356 - the closed-form terminal velocity of a drop in a temperature
//!     gradient, which is §87.9's Gate 87-D
//!   B. Lafaurie, C. Nardone, R. Scardovelli, S. Zaleski, G. Zanetti,
//!     *J. Comput. Phys.* 113 (1994) 134-147 - the continuum-surface-STRESS
//!     alternative, named in §87.3 and NOT taken
//!   ofgpu `SPEC-LIT.md` §87 (all of it), §20.1 (`n_hat` and its `epsN`),
//!     §20.4 (the normal term this sits beside), §5.1 (why a body force lives
//!     on faces), §13.4 (what happens to a setting this solver does not have)
//! No GPL-licensed source was consulted.
//!
//! # The obstruction, and why it is not a rewrite of `phib`
//!
//! `cuda/vof.cu`'s `vofBodyForceFlux` writes the whole body force as a face
//! **scalar**:
//!
//! ```text
//! phib[f] = -gh snGradRho[f] |Sf| + sigma kappa_f[f] snGradAlpha[f] |Sf|
//! ```
//!
//! A scalar per face carries one degree of freedom and it is spent: the force
//! it represents is along `Sf`. The Marangoni force is by definition
//! **tangential to the interface**, and the interface is not aligned with
//! faces. **No rewrite of `phib` can hold it** - §87.1 makes that a counting
//! argument rather than an opinion.
//!
//! So §87.3 splits the force. The normal half stays exactly where it is, in
//! the balanced-force face representation that makes a static drop hold its
//! Laplace jump; the tangential half becomes a cell-vector `fvm_su` source.
//!
//! # What that costs, and what it does not
//!
//! It does not cost the default answer. A case that configures no Marangoni
//! model launches `vofBodyForceFlux` from the same call site with the same
//! arguments, and `Vof::assemble_component` adds no source at all - the
//! `Option` is `None` and the code is not entered. `cuda/vof.cu` was not
//! edited. That is §87.5's bitwise claim and it is a construction, not a
//! measurement.
//!
//! It does cost one order in the flux. The tangential force never enters
//! `phib`, so it reaches the face flux only through `HbyA` - which is correct
//! and not merely tolerable: a tangential force is not a gradient and **must
//! not** be balanced by a pressure gradient, it is supposed to drive flow.
//! §87.4 works that through.
//!
//! # The trap: a uniform `sigma` does not always interpolate to itself
//!
//! When the model IS on, the normal term needs `sigma_f`, and
//! `sigma_f = w sigma_P + (1-w) sigma_N`. For a uniform field that returns
//! `sigma` exactly when `w` is `1/2` - and often, but **not always**,
//! otherwise. So a `dSigmaDT` of zero routed through the field path would
//! move a case that asked for nothing: the same shape of defect as §39's
//! `cos(pi/2)`, and found the same way, by looking for it because §39 had
//! happened.
//!
//! The measurement refuted the assumption it started from, which was that
//! `w != 1/2` always moves it. Whether it moves depends on `sigma`'s
//! mantissa: over `[0.010, 0.080)` N/m, **524 of 700** values are moved by
//! at least one weight and the worst is moved by a third of them, always by
//! exactly one ulp - but **this crate's own default `0.0728` is moved by
//! none of them**, while `0.0730` beside it is moved by 2 %. The default is
//! safe by luck of its bits, not by design, which is a better reason for the
//! guard than the one this module was written with.
//!
//! [`MarangoniModel::is_active`] is that guard: a zero coefficient **is the
//! model being off**, and `Vof::set_marangoni` stores `None` for it.
//! `a_uniform_sigma_field_does_not_interpolate_to_itself` measures all of the
//! above rather than asserting any of it.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::{GpuScalarField, GpuSurfaceScalarField};
use crate::mesh::GpuMesh;
use crate::types::Vec3;
use crate::{Label, Scalar};

// ==========================================================================
//  §87.2  The closure
// ==========================================================================

/// The linear surface-tension closure, `sigma = sigma0 + (dsigma/dT)(T - T0)`.
///
/// §87.2. Young, Goldstein & Block assume exactly this form, so a case that
/// wants to be held against their closed form must not stray from it.
///
/// The driving field is called `T` throughout because thermocapillary flow is
/// what §87.9 gates. Nothing in the kernel knows that: a solutal Marangoni
/// case supplies a mass fraction instead and reads `dsigma_dt` as
/// `dsigma/dY`. §87.2 says so, and says that the *units* of `dsigma_dt` are
/// then the caller's to keep straight, because this module cannot check them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarangoniModel {
    /// `sigma` at the reference state `T0`, in N/m.
    pub sigma0: Scalar,
    /// `d sigma / dT`, in N/(m K). Negative for almost every clean
    /// liquid-gas pair, which is why a bubble migrates towards the hot side.
    pub dsigma_dt: Scalar,
    /// The reference temperature `T0`, in K.
    pub t0: Scalar,
    /// The floor under `sigma`. §87.2: the linear closure is unbounded below
    /// and a negative surface tension is not a stiff problem, it is an
    /// ill-posed one.
    pub sigma_min: Scalar,
}

impl Default for MarangoniModel {
    /// Water against air at 293.15 K, with the classic linear coefficient.
    ///
    /// `sigma0` is `vof.rs`'s own default (0.0728 N/m) so that turning the
    /// model on does not silently change the capillary scale as well, and
    /// `dsigma_dt = -1.5e-4 N/(m K)` is the round figure for a clean
    /// water-air interface near room temperature. Both are DEFAULTS, not
    /// gates: §87.9 pins its own numbers.
    fn default() -> Self {
        Self { sigma0: 0.0728, dsigma_dt: -1.5e-4, t0: 293.15, sigma_min: 0.0 }
    }
}

impl MarangoniModel {
    /// Whether this model does anything - §87.5.
    ///
    /// A zero `dsigma_dt` makes `sigma` a uniform field, and a uniform field
    /// does not reliably survive the face interpolation: for three quarters
    /// of the plausible surface tensions some weight moves it by one ulp
    /// (`a_uniform_sigma_field_does_not_interpolate_to_itself` measures which,
    /// and finds this crate's own default among the quarter that survive). So
    /// a zero coefficient is the model being OFF, not the model being on with
    /// nothing to do, and `Vof::set_marangoni` stores `None` for it. That is
    /// the whole of §87.5's guard, and it is one comparison.
    pub fn is_active(&self) -> bool {
        self.dsigma_dt != 0.0
    }

    /// §13.4: refuse what cannot be honoured, by name.
    pub fn validate(&self) -> Result<()> {
        if !self.sigma0.is_finite() || self.sigma0 < 0.0 {
            return Err(Error::Config(format!(
                "marangoni: sigma0 = {}; a reference surface tension is finite \
                 and not negative",
                self.sigma0
            )));
        }
        if !self.dsigma_dt.is_finite() {
            return Err(Error::Config(format!(
                "marangoni: dSigmaDT = {}; it must be finite. Zero is allowed \
                 and means the model is off (SPEC-LIT §87.5: a uniform sigma \
                 does not reliably survive the face interpolation, so a zero \
                 coefficient must not be routed through the field path)",
                self.dsigma_dt
            )));
        }
        if !self.t0.is_finite() {
            return Err(Error::Config(format!(
                "marangoni: T0 = {}; the reference temperature must be finite",
                self.t0
            )));
        }
        if !self.sigma_min.is_finite() || self.sigma_min < 0.0 {
            return Err(Error::Config(format!(
                "marangoni: sigmaMin = {}; the floor under sigma is finite and \
                 not negative (SPEC-LIT §87.2)",
                self.sigma_min
            )));
        }
        Ok(())
    }

    /// The closure itself, on the host, so a test can check the device
    /// against something that is not the device.
    pub fn sigma_of(&self, t: Scalar) -> Scalar {
        let s = self.sigma0 + self.dsigma_dt * (t - self.t0);
        if s < self.sigma_min {
            self.sigma_min
        } else {
            s
        }
    }
}

// ==========================================================================
//  The device side
// ==========================================================================

pub struct MarangoniKernels {
    sigma_from_t: CudaFunction,
    sigma_floor: CudaFunction,
    tangential_force: CudaFunction,
    body_force_flux: CudaFunction,
    body_force_flux_boundary: CudaFunction,
    force_integrand: CudaFunction,
    /// `momVecComponent`, borrowed from the momentum module so that one
    /// component of `f_M` can be handed to `fv::fvm_su`, which takes a
    /// scalar.
    vec_component: CudaFunction,
}

impl MarangoniKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::MARANGONI)?;
        let m = KernelSet::new(gpu, crate::kernels::MOMENTUM)?;
        Ok(Self {
            sigma_from_t: k.func("marSigmaFromT")?,
            sigma_floor: k.func("marSigmaFloor")?,
            tangential_force: k.func("marTangentialForce")?,
            body_force_flux: k.func("marBodyForceFlux")?,
            body_force_flux_boundary: k.func("marBodyForceFluxBoundary")?,
            force_integrand: k.func("marForceIntegrand")?,
            vec_component: m.func("momVecComponent")?,
        })
    }
}

/// Everything §87 adds to a [`crate::vof::Vof`], allocated only when a model
/// is configured.
///
/// §87.5: this whole struct is behind an `Option`. A case that names no
/// Marangoni model allocates none of it and launches none of its kernels,
/// which is why the default path is bitwise what it was.
pub struct Marangoni {
    pub kern: MarangoniKernels,
    pub model: MarangoniModel,
    /// The driving field. **Prescribed** - §87 ships the force, not the
    /// two-phase energy equation that would transport it (§87.10 says so in
    /// as many words, and names what that costs). The caller writes `.f` and
    /// `.bf` and this module reads them.
    pub t: GpuScalarField,
    /// `sigma(T)` on cells and on boundary faces. The boundary values are not
    /// a boundary CONDITION - the closure is algebraic, so they are the same
    /// line applied to `T.bf`, and they exist because the Green-Gauss gather
    /// in `fv::fvc_grad_scalar` reads them.
    pub sigma: GpuScalarField,
    /// `sigma` interpolated to faces, for the normal term.
    pub sigma_f: GpuSurfaceScalarField,
    pub grad_sigma: DevBuf<Vec3>,
    /// `f_M = (I - n n) grad(sigma) |grad(alpha)|`, in N/m^3.
    pub f_m: DevBuf<Vec3>,
    /// `1` where the `sigma_min` floor bit this update, `0` where it did not.
    pub clipped: DevBuf<Scalar>,
    /// Scratch: one component of `f_M`, and the per-cell force integrand.
    pub scratch: DevBuf<Scalar>,
}

impl Marangoni {
    pub fn new(gpu: &Gpu, m: &GpuMesh, model: MarangoniModel) -> Result<Self> {
        model.validate()?;
        let n = m.n_cells;
        Ok(Self {
            kern: MarangoniKernels::new(gpu)?,
            model,
            t: GpuScalarField::zeros(gpu, m, "T")?,
            sigma: GpuScalarField::zeros(gpu, m, "sigma")?,
            sigma_f: GpuSurfaceScalarField::zeros(gpu, m, "sigmaf")?,
            grad_sigma: gpu.zeros(n)?,
            f_m: gpu.zeros(n)?,
            clipped: gpu.zeros(n)?,
            scratch: gpu.zeros(n)?,
        })
    }
}

/// `sigma = sigma0 + (dsigma/dT)(T - T0)`, floored, on cells and on boundary
/// faces - §87.2.
pub fn update_sigma(gpu: &Gpu, d: &mut Marangoni, m: &GpuMesh) -> Result<()> {
    let md = d.model;

    // Two launches of ONE kernel: it reads one array and writes one and knows
    // nothing about whether they are cells or boundary faces
    // (cuda/marangoni.cu says so). The boundary pass is not a boundary
    // CONDITION - the closure is algebraic in T, so there is nothing to
    // evaluate, only the same line applied to T's own boundary values.
    if m.n_cells > 0 {
        let n = m.n_cells;
        let nl = n as Label;
        let f = d.kern.sigma_from_t.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut d.sigma.f)
                .arg(&d.t.f)
                .arg(&md.sigma0)
                .arg(&md.dsigma_dt)
                .arg(&md.t0)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }
    if m.n_boundary_faces > 0 {
        let n = m.n_boundary_faces;
        let nl = n as Label;
        let f = d.kern.sigma_from_t.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut d.sigma.bf)
                .arg(&d.t.bf)
                .arg(&md.sigma0)
                .arg(&md.dsigma_dt)
                .arg(&md.t0)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    // The floor, on cells only: `clipped` is a cell diagnostic, and a
    // boundary-face sigma that has gone negative has a negative cell beside
    // it that the count will already have found.
    if m.n_cells > 0 {
        let n = m.n_cells;
        let nl = n as Label;
        let f = d.kern.sigma_floor.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut d.sigma.f)
                .arg(&mut d.clipped)
                .arg(&md.sigma_min)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }
    Ok(())
}

/// `f_M = (grad(sigma) - n_hat (n_hat . grad(sigma))) |grad(alpha)|` - §87.3.
///
/// `grad_sigma` must already hold `fvc_grad_scalar(sigma)`; the caller owns
/// that gather because it owns the `FvKernels`.
pub fn update_tangential_force(
    gpu: &Gpu,
    d: &mut Marangoni,
    grad_alpha: &DevBuf<Vec3>,
    eps_n: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = d.kern.tangential_force.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(&mut d.f_m)
            .arg(&d.grad_sigma)
            .arg(grad_alpha)
            .arg(&eps_n)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// One component of `f_M`, into [`Marangoni::scratch`], ready for
/// `fv::fvm_su`.
pub fn force_component(gpu: &Gpu, d: &mut Marangoni, cmpt: Label, n: usize) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = d.kern.vec_component.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(&mut d.scratch)
            .arg(&d.f_m)
            .arg(&cmpt)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `V_P f_M,P[cmpt]` per cell, into [`Marangoni::scratch`] - §87.6, the
/// integrand Gate 87-C reduces.
pub fn force_integrand(
    gpu: &Gpu,
    d: &mut Marangoni,
    v: &DevBuf<Scalar>,
    cmpt: Label,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = d.kern.force_integrand.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(&mut d.scratch)
            .arg(&d.f_m)
            .arg(v)
            .arg(&cmpt)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// The normal body-force flux with `sigma` a field - §87.4.
///
/// Byte for byte the arithmetic of `vofBodyForceFlux` with `sigma` replaced
/// by `sigmaf[f]`; `the_field_sigma_flux_is_the_scalar_one_when_sigma_is_flat`
/// pins that against the original kernel rather than against this comment.
#[allow(clippy::too_many_arguments)]
pub fn body_force_flux(
    gpu: &Gpu,
    d: &Marangoni,
    phib: &mut GpuSurfaceScalarField,
    sn_grad_rho: &GpuSurfaceScalarField,
    sn_grad_alpha: &GpuSurfaceScalarField,
    kappa_f: &GpuSurfaceScalarField,
    fr: &DevBuf<Scalar>,
    g: Vec3,
    m: &GpuMesh,
) -> Result<()> {
    let (nif, nbf) = (m.n_internal_faces, m.n_boundary_faces);

    if nif > 0 {
        let nl = nif as Label;
        let f = d.kern.body_force_flux.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut phib.f)
                .arg(&sn_grad_rho.f)
                .arg(&sn_grad_alpha.f)
                .arg(&kappa_f.f)
                .arg(&d.sigma_f.f)
                .arg(&m.cf)
                .arg(&g.x)
                .arg(&g.y)
                .arg(&g.z)
                .arg(&nl)
                .launch(cfg_for(nif))?;
        }
    }

    if nbf > 0 {
        let nl = nbf as Label;
        let f = d.kern.body_force_flux_boundary.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut phib.bf)
                .arg(&sn_grad_rho.bf)
                .arg(&sn_grad_alpha.bf)
                .arg(&kappa_f.bf)
                .arg(&d.sigma_f.bf)
                .arg(&m.b_cf)
                .arg(&m.b_kind)
                .arg(fr)
                .arg(&g.x)
                .arg(&g.y)
                .arg(&g.z)
                .arg(&nl)
                .launch(cfg_for(nbf))?;
        }
    }
    Ok(())
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// §87.5's premise, MEASURED — and it came out sharper than assumed.
    ///
    /// The claim being tested is that `sigma_f = w sigma_P + (1-w) sigma_N`
    /// does not return a **uniform** field unchanged, so a `dsigma/dT = 0`
    /// case must not be routed through the field path. Measuring it found
    /// something the assertion would have hidden: **whether it moves depends
    /// on `sigma`'s mantissa, and the crate's own default `sigma = 0.0728` is
    /// one of the values it does not move.**
    ///
    /// So the guard is *not* justified by "the default would shift". It is
    /// justified by this: over the plausible band of surface tensions, a
    /// majority of values ARE moved, by up to a third of the weights a
    /// non-uniform mesh produces, and the default's immunity is a property of
    /// that particular number rather than of anything in the code. A case
    /// naming `sigma 0.0715` or `sigma 0.0730` — both one step either side of
    /// the default in any table — is moved.
    ///
    /// A guard justified by an assertion is a guard nobody will dare delete
    /// and nobody can check. This one is justified by a scan.
    #[test]
    fn a_uniform_sigma_field_does_not_interpolate_to_itself() {
        // Weights a non-uniform mesh actually produces: a low-discrepancy
        // sweep of (0,1) rather than the dyadic k/2^n a uniform box gives.
        // Deterministic, so a bitwise claim stays reproducible.
        let weights: Vec<Scalar> = (1..5000usize)
            .map(|i| (i as Scalar) * 0.618_033_988_749_894_9 % 1.0)
            .filter(|w| *w > 0.0 && *w < 1.0)
            .collect();

        let moved_by = |sigma: Scalar| -> (usize, Scalar) {
            let mut n = 0usize;
            let mut worst: Scalar = 0.0;
            for w in &weights {
                let s = w * sigma + (1.0 - w) * sigma;
                if s != sigma {
                    n += 1;
                    let d = (s - sigma).abs();
                    if d > worst {
                        worst = d;
                    }
                }
            }
            (n, worst)
        };

        // The one weight that IS exact, on any sigma - and a uniform box gives
        // every internal face exactly this one, which is precisely why the
        // defect would never have shown up on the meshes this project gates on.
        let sigma: Scalar = 0.0728;
        assert_eq!(0.5 * sigma + 0.5 * sigma, sigma);

        // The scan: how much of the plausible surface-tension band is
        // vulnerable at all?
        let mut vulnerable = 0usize;
        let mut scanned = 0usize;
        let mut worst_frac: Scalar = 0.0;
        let mut worst_sigma: Scalar = 0.0;
        let mut s: Scalar = 0.010;
        while s < 0.080 {
            scanned += 1;
            let (n, _) = moved_by(s);
            if n > 0 {
                vulnerable += 1;
                let f = n as Scalar / weights.len() as Scalar;
                if f > worst_frac {
                    worst_frac = f;
                    worst_sigma = s;
                }
            }
            s += 0.0001;
        }

        let (n_default, _) = moved_by(sigma);
        let (n_near, worst_near) = moved_by(0.0730);

        println!(
            "  §87.5: over sigma in [0.010, 0.080), {vulnerable} of {scanned} \
             values are moved by SOME interpolation weight ({:.1} %); the worst \
             is sigma = {worst_sigma} at {:.1} % of weights. The crate default \
             sigma = {sigma} is moved by {n_default} of {} - it is one of the \
             safe ones, by luck of its mantissa - while sigma = 0.0730 beside \
             it is moved by {n_near}, always by exactly one ulp ({worst_near:e}).",
            100.0 * vulnerable as Scalar / scanned as Scalar,
            100.0 * worst_frac,
            weights.len(),
        );

        assert!(
            vulnerable * 4 > scanned,
            "only {vulnerable} of {scanned} surface tensions in the plausible \
             band are moved by any interpolation weight. If a uniform sigma \
             really does survive interpolation for nearly every value, the \
             §87.5 guard rests on nothing and this test should say so rather \
             than the guard staying in on faith"
        );
        assert!(
            n_near > 0,
            "sigma = 0.0730, one table entry away from this crate's own \
             default, is not moved by any weight either - re-derive §87.5"
        );
        assert!(
            worst_near > 0.0 && worst_near < 1e-16,
            "a uniform sigma moved by {worst_near}; a single rounding of \
             0.0730 is what §87.5 claims, and this is not it"
        );
    }

    /// §87.5's guard: a zero coefficient is the model being off.
    #[test]
    fn a_zero_coefficient_is_the_model_being_off() {
        let m = MarangoniModel { dsigma_dt: 0.0, ..MarangoniModel::default() };
        assert!(!m.is_active());
        assert!(
            m.validate().is_ok(),
            "a zero coefficient is legal - it is off, not wrong"
        );
        let m = MarangoniModel { dsigma_dt: -1.5e-4, ..MarangoniModel::default() };
        assert!(m.is_active());
    }

    /// The host closure, so the device has something to be checked against.
    #[test]
    fn the_linear_closure_is_the_one_young_goldstein_and_block_assume() {
        let m = MarangoniModel {
            sigma0: 0.05,
            dsigma_dt: -2.0e-4,
            t0: 300.0,
            sigma_min: 0.0,
        };
        assert_eq!(m.sigma_of(300.0), 0.05);
        assert!((m.sigma_of(310.0) - 0.048).abs() < 1e-15);
        assert!((m.sigma_of(290.0) - 0.052).abs() < 1e-15);
        // The floor, and the reason for it: 0.05 - 2e-4 (T - 300) reaches
        // zero at T = 550 and would go negative beyond it.
        assert_eq!(m.sigma_of(600.0), 0.0, "the floor holds sigma at sigmaMin");
    }

    // ------------------------------------------------------------------
    //  Device tests, at the KERNEL level
    //
    //  These need no mesh. Every kernel §87 adds takes flat arrays, so both
    //  the new normal-flux kernel and the one it was copied from can be fed
    //  the same synthetic inputs and their outputs compared BIT FOR BIT -
    //  which is a stronger statement about §87.5 than any mesh-level test,
    //  because it isolates the arithmetic from the geometry.
    // ------------------------------------------------------------------

    /// A machine without a card makes every device test pass vacuously, which
    /// is the convention the rest of the crate follows.
    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// A deterministic, non-uniform, sign-varying filler. Not random: a test
    /// that is bitwise must be reproducible, and a seeded PRNG is one more
    /// thing to have to trust.
    fn spread(i: usize, a: Scalar, b: Scalar) -> Scalar {
        let t = ((i * 2654435761) % 1000) as Scalar / 1000.0;
        a + (b - a) * t
    }

    /// §87.5, guard two, MEASURED: `marBodyForceFlux` is `vofBodyForceFlux`.
    ///
    /// Given a `sigma_f` buffer holding the literal constant rather than an
    /// interpolation of it, the two kernels must agree on **every bit of
    /// every face**. If they do, then the only thing §87 can do to a default
    /// answer is the interpolation - and `MarangoniModel::is_active` keeps
    /// that out. If they do not, §87.5's claim is false and this test says so
    /// before any case does.
    ///
    /// The comparison is on the bit pattern, not on a tolerance. There is no
    /// tolerance at which this test would be interesting.
    #[test]
    fn the_field_sigma_flux_is_the_scalar_one_when_sigma_is_flat() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };
        let n = 4096usize;
        let sigma: Scalar = 0.0728;

        let sn_rho: Vec<Scalar> = (0..n).map(|i| spread(i, -900.0, 900.0)).collect();
        let sn_alp: Vec<Scalar> = (0..n).map(|i| spread(i + 7, -40.0, 40.0)).collect();
        let kappa: Vec<Scalar> = (0..n).map(|i| spread(i + 13, -50.0, 50.0)).collect();
        let cf: Vec<Vec3> = (0..n)
            .map(|i| {
                Vec3::new(spread(i, 0.0, 1.0), spread(i + 3, 0.0, 1.0), spread(i + 5, 0.0, 1.0))
            })
            .collect();
        let sig_f = vec![sigma; n];

        let d_sn_rho = g.upload(&sn_rho)?;
        let d_sn_alp = g.upload(&sn_alp)?;
        let d_kappa = g.upload(&kappa)?;
        let d_cf = g.upload(&cf)?;
        let d_sig_f = g.upload(&sig_f)?;
        let mut old: DevBuf<Scalar> = g.zeros(n)?;
        let mut new: DevBuf<Scalar> = g.zeros(n)?;

        let (gx, gy, gz) = (0.0 as Scalar, 0.0 as Scalar, -9.81 as Scalar);
        let nl = n as Label;

        let v = KernelSet::new(&g, crate::kernels::VOF)?;
        let f = v.func("vofBodyForceFlux")?;
        unsafe {
            g.stream()
                .launch_builder(&f)
                .arg(&mut old)
                .arg(&d_sn_rho)
                .arg(&d_sn_alp)
                .arg(&d_kappa)
                .arg(&d_cf)
                .arg(&gx)
                .arg(&gy)
                .arg(&gz)
                .arg(&sigma)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }

        let k = KernelSet::new(&g, crate::kernels::MARANGONI)?;
        let f = k.func("marBodyForceFlux")?;
        unsafe {
            g.stream()
                .launch_builder(&f)
                .arg(&mut new)
                .arg(&d_sn_rho)
                .arg(&d_sn_alp)
                .arg(&d_kappa)
                .arg(&d_sig_f)
                .arg(&d_cf)
                .arg(&gx)
                .arg(&gy)
                .arg(&gz)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }

        let a = g.download(&old)?;
        let b = g.download(&new)?;
        let mut n_diff = 0usize;
        for i in 0..n {
            if a[i].to_bits() != b[i].to_bits() {
                n_diff += 1;
            }
        }
        assert_eq!(
            n_diff, 0,
            "§87.5: marBodyForceFlux differs from vofBodyForceFlux on {n_diff} \
             of {n} faces given the SAME face sigma. The two are supposed to \
             be the same arithmetic with one argument widened from a scalar \
             to an array; if they are not, the §87.5 bitwise claim is false"
        );
        // ...and the inputs really were varied, so the agreement is not the
        // agreement of two arrays of zeros.
        assert!(
            a.iter().filter(|x| **x != 0.0).count() > n / 2,
            "the fixture produced mostly zeros and proves nothing"
        );
        println!("  §87.5: {n} faces, 0 differing bits between the two flux kernels");
        Ok(())
    }

    /// The boundary twin, and the restated `vofFluxIsPrescribed` predicate.
    ///
    /// `cuda/marangoni.cu` restates the predicate rather than sharing a
    /// header, because §87.5's argument turns on `cuda/vof.cu` not being
    /// edited. A restated predicate can drift, so it is pinned here across
    /// every patch kind and both sides of the `fr >= 1` test.
    #[test]
    fn marangoni_restates_the_prescribed_flux_predicate() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };
        let sigma: Scalar = 0.0728;

        // Every patch kind 0..5, crossed with fr below, at, and above 1.
        let frs: [Scalar; 4] = [0.0, 0.5, 1.0, 1.5];
        let mut kinds: Vec<Label> = Vec::new();
        let mut fr: Vec<Scalar> = Vec::new();
        for k in 0..6 {
            for f in frs {
                kinds.push(k);
                fr.push(f);
            }
        }
        let n = kinds.len();

        let sn_rho: Vec<Scalar> = (0..n).map(|i| spread(i, -900.0, 900.0)).collect();
        let sn_alp: Vec<Scalar> = (0..n).map(|i| spread(i + 7, -40.0, 40.0)).collect();
        let kappa: Vec<Scalar> = (0..n).map(|i| spread(i + 13, -50.0, 50.0)).collect();
        let cf: Vec<Vec3> = (0..n)
            .map(|i| Vec3::new(spread(i, 0.0, 1.0), spread(i + 3, 0.0, 1.0), 0.25))
            .collect();

        let d_sn_rho = g.upload(&sn_rho)?;
        let d_sn_alp = g.upload(&sn_alp)?;
        let d_kappa = g.upload(&kappa)?;
        let d_cf = g.upload(&cf)?;
        let d_sig_f = g.upload(&vec![sigma; n])?;
        let d_kinds = g.upload(&kinds)?;
        let d_fr = g.upload(&fr)?;
        let mut old: DevBuf<Scalar> = g.zeros(n)?;
        let mut new: DevBuf<Scalar> = g.zeros(n)?;

        let (gx, gy, gz) = (0.0 as Scalar, 0.0 as Scalar, -9.81 as Scalar);
        let nl = n as Label;

        let v = KernelSet::new(&g, crate::kernels::VOF)?;
        let f = v.func("vofBodyForceFluxBoundary")?;
        unsafe {
            g.stream()
                .launch_builder(&f)
                .arg(&mut old)
                .arg(&d_sn_rho)
                .arg(&d_sn_alp)
                .arg(&d_kappa)
                .arg(&d_cf)
                .arg(&d_kinds)
                .arg(&d_fr)
                .arg(&gx)
                .arg(&gy)
                .arg(&gz)
                .arg(&sigma)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }

        let k = KernelSet::new(&g, crate::kernels::MARANGONI)?;
        let f = k.func("marBodyForceFluxBoundary")?;
        unsafe {
            g.stream()
                .launch_builder(&f)
                .arg(&mut new)
                .arg(&d_sn_rho)
                .arg(&d_sn_alp)
                .arg(&d_kappa)
                .arg(&d_sig_f)
                .arg(&d_cf)
                .arg(&d_kinds)
                .arg(&d_fr)
                .arg(&gx)
                .arg(&gy)
                .arg(&gz)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }

        let a = g.download(&old)?;
        let b = g.download(&new)?;
        let mut zeroed = 0usize;
        for i in 0..n {
            assert_eq!(
                a[i].to_bits(),
                b[i].to_bits(),
                "kind {} fr {}: vof gave {} and marangoni {}",
                kinds[i],
                fr[i],
                a[i],
                b[i]
            );
            if a[i] == 0.0 {
                zeroed += 1;
            }
        }
        // 2 (empty) and 3 (symmetry) are prescribed at every fr; every other
        // kind is prescribed only at fr >= 1, which is two of the four.
        let expect_zero = 2 * frs.len() + 4 * 2;
        assert_eq!(
            zeroed, expect_zero,
            "the predicate zeroed {zeroed} of {n} faces; empty and symmetry at \
             every fr, plus fr >= 1 elsewhere, is {expect_zero}"
        );
        println!("  §87.5: {n} boundary faces across 6 patch kinds, 0 differing bits");
        Ok(())
    }

    /// Gate 87-B: what the projector must do, and must not.
    ///
    /// Three statements in one launch, all at the kernel level so the
    /// geometry cannot be blamed for any of them:
    ///
    /// * `grad(sigma)` parallel to `n̂` gives **zero** to round-off - there is
    ///   no tangential component to feel;
    /// * `grad(sigma)` orthogonal to `n̂` passes through multiplied by
    ///   `|grad(alpha)|` and by nothing else;
    /// * a pure phase (`grad(alpha) = 0`) gives **exactly** zero, because the
    ///   `epsN` normalisation makes `n̂` a zero vector there and the
    ///   `|grad alpha|` factor is zero as well.
    #[test]
    fn the_projector_annihilates_a_normal_sigma_gradient() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        // A metre-scale mesh: epsN would be 1e-8 there.
        let eps_n: Scalar = 1e-8;
        let mag: Scalar = 40.0; // |grad alpha| across a 2.5 cm band

        // Three cells: normal, tangential, pure phase.
        let n_hat = [
            Vec3::new(0.6, 0.8, 0.0), // unit
            Vec3::new(0.6, 0.8, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
        ];
        let grad_alpha: Vec<Vec3> = n_hat
            .iter()
            .enumerate()
            .map(|(i, v)| {
                if i == 2 {
                    Vec3::new(0.0, 0.0, 0.0)
                } else {
                    Vec3::new(v.x * mag, v.y * mag, v.z * mag)
                }
            })
            .collect();

        let gs_par = Vec3::new(0.6 * 3.0e-4, 0.8 * 3.0e-4, 0.0); // ∥ n̂
        let gs_perp = Vec3::new(-0.8 * 3.0e-4, 0.6 * 3.0e-4, 0.0); // ⟂ n̂
        let grad_sigma = vec![gs_par, gs_perp, gs_perp];

        let d_ga = g.upload(&grad_alpha)?;
        let d_gs = g.upload(&grad_sigma)?;
        let mut d_f: DevBuf<Vec3> = g.zeros(3)?;

        let k = KernelSet::new(&g, crate::kernels::MARANGONI)?;
        let f = k.func("marTangentialForce")?;
        let nl = 3 as Label;
        unsafe {
            g.stream()
                .launch_builder(&f)
                .arg(&mut d_f)
                .arg(&d_gs)
                .arg(&d_ga)
                .arg(&eps_n)
                .arg(&nl)
                .launch(cfg_for(3))?;
        }
        let out = g.download(&d_f)?;

        // (1) The normal gradient is annihilated. Not to zero exactly - the
        // epsN in the normalisation makes n̂ shorter than unity by
        // epsN/|grad alpha| = 2.5e-10 - but to that, and no worse.
        let residue = out[0].mag();
        let full = gs_par.mag() * mag;
        assert!(
            residue / full < 1e-8,
            "a grad(sigma) parallel to n_hat left {residue:e} of {full:e} \
             ({:e} relative); the projector is not annihilating it",
            residue / full
        );

        // (2) The tangential gradient passes through undiminished.
        let expect = gs_perp.mag() * mag;
        let got = out[1].mag();
        assert!(
            ((got - expect) / expect).abs() < 1e-12,
            "a grad(sigma) orthogonal to n_hat came out {got:e} against \
             |grad sigma| |grad alpha| = {expect:e}"
        );
        // ...and in the right DIRECTION, not merely the right size.
        let cosang = (out[1].x * gs_perp.x + out[1].y * gs_perp.y + out[1].z * gs_perp.z)
            / (got * gs_perp.mag());
        assert!(
            (cosang - 1.0).abs() < 1e-12,
            "the tangential force is not along grad_s(sigma): cos = {cosang}"
        );

        // (3) A pure phase is EXACTLY zero, not nearly.
        assert_eq!(out[2].x, 0.0);
        assert_eq!(out[2].y, 0.0);
        assert_eq!(out[2].z, 0.0);

        println!(
            "  Gate 87-B: normal residue {:e} relative, tangential error {:e} \
             relative, pure phase exactly zero",
            residue / full,
            ((got - expect) / expect).abs()
        );
        Ok(())
    }

    #[test]
    fn a_non_finite_coefficient_is_refused_by_name() {
        let m = MarangoniModel { dsigma_dt: Scalar::NAN, ..MarangoniModel::default() };
        let e = m.validate().expect_err("NaN must be refused");
        let s = format!("{e}");
        assert!(s.contains("dSigmaDT"), "the refusal names the entry: {s}");
        assert!(s.contains("§87.5"), "the refusal names the section: {s}");

        let m = MarangoniModel { sigma_min: -1.0, ..MarangoniModel::default() };
        let e = m.validate().expect_err("a negative floor must be refused");
        assert!(format!("{e}").contains("sigmaMin"));
    }
}
