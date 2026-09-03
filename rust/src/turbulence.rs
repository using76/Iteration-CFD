// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The machinery every eddy-viscosity RAS closure shares - SPEC-LIT §6.
//!
//! Written from:
//!   Launder & Spalding, *Comput. Methods Appl. Mech. Eng.* 3 (1974) 269-289
//!   Wilcox, *Turbulence Modeling for CFD*, DCW Industries - the 1988 k-omega
//!     form, and §5.4 for the Favre-averaged dilatation terms
//!   Patankar, *Numerical Heat Transfer and Fluid Flow* (1980), §4.2 and §4.9
//!     - the `S = S_u + S_p psi` linearisation and implicit under-relaxation
//!   Jasak (1996) ch. 3 - the discrete operators this assembles from
//!   ofgpu `SPEC-LIT.md` §6, §6.1, §6.2 and §6.4. Everything marked *DESIGN*
//!     there is ours and is labelled as such where it is implemented.
//!   ofgpu `SPEC-LIT.md` §13 - the time scheme every transport equation here
//!     carries, taken from `ddtSchemes` in full rather than reduced to a
//!     steady/transient boolean
//!   Menter, *AIAA J.* 32 (1994) 1598-1605, and Menter, Kuntz & Langtry,
//!     *Turbulence, Heat and Mass Transfer* 4 (2003) - SPEC-LIT §6.3, whose
//!     blended `sigma` is what [`face_diffusivity_cell`] and
//!     [`RasCore::assemble_transport_blended`] exist for
//!   Tucker, *Applied Mathematical Modelling* 22 (1998) 293-305 - SPEC-LIT
//!     §6.6, the Poisson wall distance whose final algebraic step is
//!     [`wall_distance_from_potential`]
//! No GPL-licensed source was consulted.
//!
//! # What is here
//!
//! A turbulence model is two or three scalar transport equations that differ
//! only in their source terms. Everything else - the effective diffusivity on
//! the faces, `ddt + div - laplacian`, the under-relaxation, the wall
//! constraint, the solve, the bound, the boundary refresh - is identical, and
//! lives in [`RasCore`]. `models/k_epsilon.rs` and `models/k_omega.rs` are
//! then almost entirely the sources of SPEC-LIT §6.1 and §6.2, which is what
//! they should be.
//!
//! The production term itself is [`crate::fv::turbulence_production`], next
//! to the gradient operator that makes its argument.
//!
//! # The one shape everything obeys
//!
//! ```text
//! ddt(psi) + div(phi, psi) - laplacian(Gamma_eff, psi) + Sp·psi = Su
//! ```
//!
//! A physical sink on the right-hand side therefore arrives as a **positive**
//! `Sp`, which is the sign that lands on the diagonal and keeps the matrix
//! diagonally dominant - Patankar's rule (§4.2), and what SPEC-LIT §6.1 means
//! by "implicit sinks on the diagonal, so that both quantities stay positive".
//!
//! # No host round-trip
//!
//! Nothing in [`RasCore::assemble_transport`] or [`RasCore::solve_equation`]
//! reads anything back, so a whole outer iteration is a pure sequence of
//! launches and can be captured into a CUDA graph
//! ([`crate::device::Gpu::capture`]) provided the linear solver is in its
//! fixed-iteration mode. [`RasCore::convergence_measure`] is the one function
//! that does touch the host, and it is deliberately not part of an iteration.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::{GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use crate::field_ops::FieldKernels;
use crate::fv::{
    div_scheme_weights, fvc_div_surface, fvc_grad_scalar_scheme,
    fvm_div_correction,
    fvm_div_bounded_correction, fvm_div_gauss, fvm_laplacian,
    fvm_laplacian_non_orth_correction, FvKernels,
};
use crate::io::schemes::DivEntry;
use crate::ldu::GpuLduMatrix;
use crate::ldu_ops::{add_boundary_contributions, relax, LduKernels};
use crate::timescheme::Ddt;
use crate::mesh::{GpuMesh, HostMesh};
use crate::solver::{
    device_max_mag, SolverControls, SolverKernels, SolverPerformance,
    SolverWorkspace,
};
use crate::wallfunctions::{constrain_wall_cells, WallData, WallFunctionCoeffs};
use crate::{Label, Scalar, Tensor, Vec3};

/// Read from `system/`; defined in [`crate::io::case`] because that is where
/// it is parsed, re-exported here because this is where it is obeyed.
pub use crate::io::case::TurbulenceControls;

// ==========================================================================
//  FlowState
// ==========================================================================

/// The frozen flow a turbulence model is corrected against.
///
/// Three borrows and a number, deliberately: a model must not own the
/// velocity field, because in a coupled run the momentum equation does, and a
/// model that kept its own copy would correct against last iteration's flow
/// without anybody noticing. `Simple::flow_state()` hands one of these out
/// each outer iteration and it lives exactly as long as the call.
///
/// `nu` is the *laminar* kinematic viscosity. The eddy viscosity is the
/// model's own output and is not in here.
pub struct FlowState<'a> {
    pub u: &'a GpuVectorField,
    pub phi: &'a GpuSurfaceScalarField,
    pub nu: Scalar,
}

impl<'a> FlowState<'a> {
    pub fn new(
        u: &'a GpuVectorField,
        phi: &'a GpuSurfaceScalarField,
        nu: Scalar,
    ) -> Self {
        Self { u, phi, nu }
    }
}

// ==========================================================================
//  Kernels
// ==========================================================================

/// Every entry point in `cuda/turbulence.cu`, resolved once.
pub struct TurbKernels {
    gamma_internal: CudaFunction,
    gamma_boundary: CudaFunction,
    gamma_internal_affine: CudaFunction,
    gamma_boundary_affine: CudaFunction,

    nut_k_epsilon: CudaFunction,
    nut_k_omega: CudaFunction,
    nut_boundary: CudaFunction,

    bound_k: CudaFunction,
    bound_epsilon: CudaFunction,
    bound_omega: CudaFunction,

    k_sources: CudaFunction,
    epsilon_sources: CudaFunction,
    k_omega_k_sources: CudaFunction,
    omega_sources: CudaFunction,

    // SPEC-LIT §33 - Launder-Sharma low-Reynolds-number k-epsilon.
    nut_launder_sharma: CudaFunction,
    ls_sqrt_positive: CudaFunction,
    ls_d_term: CudaFunction,
    ls_grad_grad_u_mag_sqr: CudaFunction,
    ls_e_term: CudaFunction,
    ls_epsilon_sources: CudaFunction,

    abs_diff: CudaFunction,
    strain_rate: CudaFunction,
    /// SPEC-LIT §56.2 - the other two invariants of `grad U`. `Omega` is what
    /// Spalart-Allmaras is calibrated on; `F` is what the DES-family
    /// shielding functions read. Neither is `S` and neither is the other.
    vorticity: CudaFunction,
    grad_frobenius: CudaFunction,

    gamma_internal_cell: CudaFunction,
    gamma_boundary_cell: CudaFunction,
    wall_distance: CudaFunction,

    buoyancy_production: CudaFunction,
    add_buoyancy_k: CudaFunction,
    add_buoyancy_epsilon: CudaFunction,
    add_buoyancy_omega: CudaFunction,
    add_buoyancy_omega_cell: CudaFunction,
}

impl TurbKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::TURBULENCE)?;
        Ok(Self {
            gamma_internal: k.func("turbGammaInternal")?,
            gamma_boundary: k.func("turbGammaBoundary")?,
            gamma_internal_affine: k.func("turbGammaInternalAffine")?,
            gamma_boundary_affine: k.func("turbGammaBoundaryAffine")?,

            nut_k_epsilon: k.func("turbNutKEpsilon")?,
            nut_k_omega: k.func("turbNutKOmega")?,
            nut_boundary: k.func("turbNutBoundary")?,

            bound_k: k.func("turbBoundK")?,
            bound_epsilon: k.func("turbBoundEpsilon")?,
            bound_omega: k.func("turbBoundOmega")?,

            k_sources: k.func("turbKSources")?,
            epsilon_sources: k.func("turbEpsilonSources")?,
            k_omega_k_sources: k.func("turbKOmegaKSources")?,
            omega_sources: k.func("turbOmegaSources")?,

            nut_launder_sharma: k.func("turbNutLaunderSharma")?,
            ls_sqrt_positive: k.func("turbLsSqrtPositive")?,
            ls_d_term: k.func("turbLsDTerm")?,
            ls_grad_grad_u_mag_sqr: k.func("turbLsGradGradUMagSqr")?,
            ls_e_term: k.func("turbLsETerm")?,
            ls_epsilon_sources: k.func("turbLsEpsilonSources")?,

            abs_diff: k.func("turbAbsDiff")?,
            strain_rate: k.func("turbStrainRateMag")?,
            vorticity: k.func("turbVorticityMag")?,
            grad_frobenius: k.func("turbGradFrobenius")?,

            gamma_internal_cell: k.func("turbGammaInternalCell")?,
            gamma_boundary_cell: k.func("turbGammaBoundaryCell")?,
            wall_distance: k.func("turbWallDistance")?,

            buoyancy_production: k.func("turbBuoyancyProduction")?,
            add_buoyancy_k: k.func("turbAddBuoyancyToK")?,
            add_buoyancy_epsilon: k.func("turbAddBuoyancyToEpsilon")?,
            add_buoyancy_omega: k.func("turbAddBuoyancyToOmega")?,
            add_buoyancy_omega_cell: k.func("turbAddBuoyancyToOmegaCell")?,
        })
    }
}

fn expect_len<T>(buf: &DevBuf<T>, want: usize, what: &str) -> Result<()> {
    if buf.len() == want {
        Ok(())
    } else {
        Err(Error::Config(format!(
            "turbulence: `{what}` has {} elements, expected {want}",
            buf.len()
        )))
    }
}

// ==========================================================================
//  Launchers
// ==========================================================================

/// `Gamma_eff·|Sf| = (nu + nu_t/sigma)·|Sf|` on every face.
///
/// The product rather than the bare diffusivity, because that is what
/// [`fvm_laplacian`] takes - see its signature in `src/fv.rs`, where the
/// reason is set out.
#[allow(clippy::too_many_arguments)]
pub fn face_diffusivity(
    gpu: &Gpu,
    k: &TurbKernels,
    gamma: &mut DevBuf<Scalar>,
    b_gamma: &mut DevBuf<Scalar>,
    nut: &GpuScalarField,
    m: &GpuMesh,
    nu: Scalar,
    r_sigma: Scalar,
) -> Result<()> {
    expect_len(gamma, m.n_internal_faces, "gamma")?;
    expect_len(b_gamma, m.n_boundary_faces, "b_gamma")?;
    expect_len(&nut.f, m.n_cells, "nut.f")?;
    expect_len(&nut.bf, m.n_boundary_faces, "nut.bf")?;

    if m.n_internal_faces > 0 {
        let n = m.n_internal_faces;
        let nl = n as Label;
        let f = k.gamma_internal.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(gamma)
                .arg(&nut.f)
                .arg(&m.weights)
                .arg(&m.mag_sf)
                .arg(&m.owner)
                .arg(&m.neighbour)
                .arg(&nu)
                .arg(&r_sigma)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    if m.n_boundary_faces > 0 {
        let n = m.n_boundary_faces;
        let nl = n as Label;
        let f = k.gamma_boundary.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(b_gamma)
                .arg(&nut.bf)
                .arg(&m.b_mag_sf)
                .arg(&nu)
                .arg(&r_sigma)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    Ok(())
}

/// `Gamma_eff·|Sf| = (a·nu + b·nu_t)·|Sf|` on every face - SPEC-LIT §41.2.
///
/// [`face_diffusivity`] generalised so that the molecular part carries its own
/// coefficient. RNG k-epsilon (SPEC-LIT §41) needs `alpha (nu + nu_t)`, where
/// the inverse Prandtl number multiplies the EFFECTIVE viscosity rather than
/// the turbulent one; folding `alpha` into `r_sigma` would give
/// `nu + alpha nu_t`, which is wrong in the first cell off a wall and silently
/// so.
///
/// `face_diffusivity(r_sigma)` is exactly this with `a = 1`, and
/// multiplication by an exact `1.0` is exact in IEEE-754, so the two agree bit
/// for bit - `tests::the_affine_diffusivity_reduces_to_the_plain_one_bitwise`
/// measures that rather than assuming it. §40 calls this with `(1, 1/sigma)`
/// and §41 with `(alpha, alpha)`: one kernel, two callers.
#[allow(clippy::too_many_arguments)]
pub fn face_diffusivity_affine(
    gpu: &Gpu,
    k: &TurbKernels,
    gamma: &mut DevBuf<Scalar>,
    b_gamma: &mut DevBuf<Scalar>,
    nut: &GpuScalarField,
    m: &GpuMesh,
    nu: Scalar,
    a: Scalar,
    b: Scalar,
) -> Result<()> {
    expect_len(gamma, m.n_internal_faces, "gamma")?;
    expect_len(b_gamma, m.n_boundary_faces, "b_gamma")?;
    expect_len(&nut.f, m.n_cells, "nut.f")?;
    expect_len(&nut.bf, m.n_boundary_faces, "nut.bf")?;

    if m.n_internal_faces > 0 {
        let n = m.n_internal_faces;
        let nl = n as Label;
        let f = k.gamma_internal_affine.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(gamma)
                .arg(&nut.f)
                .arg(&m.weights)
                .arg(&m.mag_sf)
                .arg(&m.owner)
                .arg(&m.neighbour)
                .arg(&nu)
                .arg(&a)
                .arg(&b)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    if m.n_boundary_faces > 0 {
        let n = m.n_boundary_faces;
        let nl = n as Label;
        let f = k.gamma_boundary_affine.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(b_gamma)
                .arg(&nut.bf)
                .arg(&m.b_mag_sf)
                .arg(&nu)
                .arg(&a)
                .arg(&b)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    Ok(())
}

/// `Gamma_eff·|Sf| = (nu + sigma_P·nu_t)·|Sf|` with `sigma` read **per cell**.
///
/// [`face_diffusivity`] with a field where it takes a constant. SST blends
/// `sigma_k` and `sigma_omega` between two coefficient sets with `F1`
/// (SPEC-LIT §6.3), so its diffusivity multiplier varies from cell to cell and
/// the constant form cannot express it.
///
/// The two must agree exactly when `r_sigma` happens to be uniform - they are
/// the same expression in the same multiplication order - and
/// `tests::blended_diffusivity_matches_the_uniform_one` measures that they do.
#[allow(clippy::too_many_arguments)]
pub fn face_diffusivity_cell(
    gpu: &Gpu,
    k: &TurbKernels,
    gamma: &mut DevBuf<Scalar>,
    b_gamma: &mut DevBuf<Scalar>,
    nut: &GpuScalarField,
    r_sigma: &DevBuf<Scalar>,
    m: &GpuMesh,
    nu: Scalar,
) -> Result<()> {
    expect_len(gamma, m.n_internal_faces, "gamma")?;
    expect_len(b_gamma, m.n_boundary_faces, "b_gamma")?;
    expect_len(&nut.f, m.n_cells, "nut.f")?;
    expect_len(&nut.bf, m.n_boundary_faces, "nut.bf")?;
    expect_len(r_sigma, m.n_cells, "r_sigma")?;

    if m.n_internal_faces > 0 {
        let n = m.n_internal_faces;
        let nl = n as Label;
        let f = k.gamma_internal_cell.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(gamma)
                .arg(&nut.f)
                .arg(r_sigma)
                .arg(&m.weights)
                .arg(&m.mag_sf)
                .arg(&m.owner)
                .arg(&m.neighbour)
                .arg(&nu)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    if m.n_boundary_faces > 0 {
        let n = m.n_boundary_faces;
        let nl = n as Label;
        let f = k.gamma_boundary_cell.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(b_gamma)
                .arg(&nut.bf)
                .arg(r_sigma)
                .arg(&m.b_face_cells)
                .arg(&m.b_mag_sf)
                .arg(&nu)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    Ok(())
}

/// `y = -|grad phi| + sqrt(|grad phi|² + 2 phi)` - the algebraic half of
/// Tucker's (1998) wall distance, SPEC-LIT §6.6.
///
/// The Poisson half is [`crate::walldistance::wall_distance`], which is where
/// the whole procedure and its boundary conditions are documented; this is
/// only the per-cell arithmetic that turns the potential into a length.
pub fn wall_distance_from_potential(
    gpu: &Gpu,
    kern: &TurbKernels,
    y: &mut DevBuf<Scalar>,
    grad_phi: &DevBuf<Vec3>,
    phi: &DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(y, n, "y")?;
    expect_len(grad_phi, n, "grad_phi")?;
    expect_len(phi, n, "phi")?;

    let nl = n as Label;
    let f = kern.wall_distance.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(y)
            .arg(grad_phi)
            .arg(phi)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `nu_t = C_mu k²/epsilon`, capped at `nut_max` (Launder & Spalding 1974;
/// the cap is *DESIGN*, SPEC-LIT §6.1).
pub fn nut_k_epsilon(
    gpu: &Gpu,
    kern: &TurbKernels,
    nut: &mut DevBuf<Scalar>,
    k: &DevBuf<Scalar>,
    epsilon: &DevBuf<Scalar>,
    cmu: Scalar,
    nut_max: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = kern.nut_k_epsilon.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(nut)
            .arg(k)
            .arg(epsilon)
            .arg(&cmu)
            .arg(&nut_max)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `nu_t = k/omega`, capped at `nut_max` (Wilcox 1988).
pub fn nut_k_omega(
    gpu: &Gpu,
    kern: &TurbKernels,
    nut: &mut DevBuf<Scalar>,
    k: &DevBuf<Scalar>,
    omega: &DevBuf<Scalar>,
    nut_max: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = kern.nut_k_omega.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(nut)
            .arg(k)
            .arg(omega)
            .arg(&nut_max)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// Zero-gradient `nu_t` on every boundary face whose value the model owns.
///
/// Wall faces are among them and are overwritten immediately afterwards by
/// [`WallData::update_nut`]; this is what gives every *other* calculated face
/// a defined value instead of whatever the buffer last held.
pub fn nut_boundary(
    gpu: &Gpu,
    kern: &TurbKernels,
    nut: &mut GpuScalarField,
    m: &GpuMesh,
) -> Result<()> {
    let n = m.n_boundary_faces;
    if n == 0 {
        return Ok(());
    }
    expect_len(&nut.bf, n, "nut.bf")?;
    expect_len(&nut.f, m.n_cells, "nut.f")?;

    let nl = n as Label;
    let f = kern.nut_boundary.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(&mut nut.bf)
            .arg(&nut.f)
            .arg(&m.b_face_cells)
            .arg(&nut.bc_kind)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `k <- max(k, k_min)` (*DESIGN*, SPEC-LIT §6.1).
pub fn bound_k(
    gpu: &Gpu,
    kern: &TurbKernels,
    k: &mut DevBuf<Scalar>,
    k_min: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = kern.bound_k.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(k)
            .arg(&k_min)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `epsilon <- max(epsilon, eps_min, C_mu k²/nut_max)` (*DESIGN*, SPEC-LIT
/// §6.1): bound the field that produces `nu_t` rather than clipping `nu_t`,
/// so the two cannot disagree.
#[allow(clippy::too_many_arguments)]
pub fn bound_epsilon(
    gpu: &Gpu,
    kern: &TurbKernels,
    epsilon: &mut DevBuf<Scalar>,
    k: &DevBuf<Scalar>,
    cmu: Scalar,
    nut_max: Scalar,
    eps_min: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = kern.bound_epsilon.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(epsilon)
            .arg(k)
            .arg(&cmu)
            .arg(&nut_max)
            .arg(&eps_min)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `omega <- max(omega, omega_min, k/nut_max)` (*DESIGN*, SPEC-LIT §6.1
/// applied to the omega form).
pub fn bound_omega(
    gpu: &Gpu,
    kern: &TurbKernels,
    omega: &mut DevBuf<Scalar>,
    k: &DevBuf<Scalar>,
    nut_max: Scalar,
    omega_min: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = kern.bound_omega.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(omega)
            .arg(k)
            .arg(&nut_max)
            .arg(&omega_min)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  Buoyancy production - SPEC-LIT §17
// ==========================================================================

/// Which convention the `epsilon` equation's `C_3` follows.
///
/// SPEC-LIT §17 calls this "the one genuinely unsettled constant" and gives
/// two conventions. *DESIGN*: the default is [`C3Mode::Henkes`], as §17 asks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum C3Mode {
    /// A number the case supplied. `0` is the other convention §17 names:
    /// leave `G_b` out of the `epsilon` equation entirely.
    Constant(Scalar),
    /// `C_3 = tanh |u_parallel_to_g / u_normal|` - Henkes, van der Vlugt &
    /// Hoogendoorn (1991). Goes to 1 in a vertical shear layer (a plume) and
    /// to 0 in a horizontal one.
    Henkes,
}

impl Default for C3Mode {
    fn default() -> Self {
        C3Mode::Henkes
    }
}

impl C3Mode {
    /// `(mode, constant)` as the kernel takes them.
    fn as_args(self) -> (Label, Scalar) {
        match self {
            C3Mode::Constant(v) => (0, v),
            C3Mode::Henkes => (1, 0.0),
        }
    }

    pub fn describe(self) -> String {
        match self {
            C3Mode::Constant(v) if v == 0.0 => {
                "C3 = 0 (G_b left out of the epsilon equation)".to_string()
            }
            C3Mode::Constant(v) => format!("C3 = {v} (constant, from the case)"),
            C3Mode::Henkes => "C3 = tanh|u_par/u_norm| (Henkes et al. 1991)".to_string(),
        }
    }
}

/// Everything the buoyancy production term needs that is not a field.
///
/// Written from Rodi, *J. Geophys. Res.* 92 (1987) 5305-5328 and Henkes, van
/// der Vlugt & Hoogendoorn, *Int. J. Heat Mass Transfer* 34 (1991) 377-388,
/// via SPEC-LIT §17.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuoyancyProduction {
    /// Gravity, as `constant/g` carries it.
    pub g: Vec3,
    /// Turbulent Prandtl number for heat. SPEC-LIT §17 gives 0.85.
    pub prt: Scalar,
    pub c3: C3Mode,
    /// Include the STABLE branch (`G_b < 0`) in the `epsilon` equation as well
    /// as in `k`. *DESIGN*: off, which is the combination §17 describes as
    /// usual. `k` always gets both branches.
    pub epsilon_stable_branch: bool,
    /// Absolute-temperature floor for the `1/T` in the term. Not a physical
    /// constant - a guard, so a zero in the temperature field announces itself
    /// as a wrong answer rather than as a NaN.
    pub t_min: Scalar,
}

impl Default for BuoyancyProduction {
    fn default() -> Self {
        Self {
            g: Vec3::new(0.0, 0.0, 0.0),
            prt: 0.85,
            c3: C3Mode::default(),
            epsilon_stable_branch: false,
            t_min: 1.0,
        }
    }
}

impl BuoyancyProduction {
    /// Is there any gravity for the term to work with? A zero `g` makes `G_b`
    /// identically zero, so the whole term can be skipped.
    pub fn is_active(&self) -> bool {
        self.g.mag() > 0.0
    }

    pub fn validate(&self) -> Result<()> {
        if !(self.prt > 0.0) {
            return Err(Error::Config(format!(
                "Prt is {}; the turbulent Prandtl number divides the eddy \
                 viscosity and must be positive (SPEC-LIT §17)",
                self.prt
            )));
        }
        if let C3Mode::Constant(v) = self.c3 {
            if !v.is_finite() {
                return Err(Error::Config(format!("C3 is {v}")));
            }
        }
        Ok(())
    }
}

/// `G_b = (nu_t/Pr_t) g·grad(T)/T` and, alongside it, `C_3` per cell.
///
/// SPEC-LIT §17. `grad_t` must be the gradient of the SAME temperature field
/// `t`, evaluated with its boundary values up to date - the term is a product
/// of the two and a stale gradient is a wrong sign waiting to happen.
///
/// The sign is the whole point: with `g` pointing down and `grad(T)` up -
/// stable stratification - `g·grad(T) < 0` and `G_b < 0`, so buoyancy
/// destroys turbulence. `tests::buoyancy_production_sign` pins it.
#[allow(clippy::too_many_arguments)]
pub fn buoyancy_production(
    gpu: &Gpu,
    kern: &TurbKernels,
    gb: &mut DevBuf<Scalar>,
    c3: &mut DevBuf<Scalar>,
    nut: &DevBuf<Scalar>,
    grad_t: &DevBuf<Vec3>,
    t: &DevBuf<Scalar>,
    u: &DevBuf<Vec3>,
    b: &BuoyancyProduction,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let (mode, c3c) = b.c3.as_args();
    let r_prt = 1.0 / b.prt;
    let (gx, gy, gz) = (b.g.x, b.g.y, b.g.z);
    let t_min = b.t_min;

    let f = kern.buoyancy_production.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(gb)
            .arg(c3)
            .arg(nut)
            .arg(grad_t)
            .arg(t)
            .arg(u)
            .arg(&gx)
            .arg(&gy)
            .arg(&gz)
            .arg(&r_prt)
            .arg(&mode)
            .arg(&c3c)
            .arg(&t_min)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// Add `G_b` to the `k` equation's linearised sources, both signs
/// (SPEC-LIT §17).
///
/// ACCUMULATES: call after [`k_sources`], which writes `sp` outright.
#[allow(clippy::too_many_arguments)]
pub fn add_buoyancy_to_k(
    gpu: &Gpu,
    kern: &TurbKernels,
    su: &mut DevBuf<Scalar>,
    sp: &mut DevBuf<Scalar>,
    gb: &DevBuf<Scalar>,
    k: &DevBuf<Scalar>,
    k_min: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = kern.add_buoyancy_k.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(su)
            .arg(sp)
            .arg(gb)
            .arg(k)
            .arg(&k_min)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// Add `C_1 (eps/k) C_3 G_b` to the `epsilon` equation's sources
/// (SPEC-LIT §17). ACCUMULATES; call after [`epsilon_sources`].
#[allow(clippy::too_many_arguments)]
pub fn add_buoyancy_to_epsilon(
    gpu: &Gpu,
    kern: &TurbKernels,
    su: &mut DevBuf<Scalar>,
    sp: &mut DevBuf<Scalar>,
    gb: &DevBuf<Scalar>,
    c3: &DevBuf<Scalar>,
    k: &DevBuf<Scalar>,
    epsilon: &DevBuf<Scalar>,
    c1: Scalar,
    k_min: Scalar,
    stable_branch: bool,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let sb: Label = if stable_branch { 1 } else { 0 };
    let f = kern.add_buoyancy_epsilon.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(su)
            .arg(sp)
            .arg(gb)
            .arg(c3)
            .arg(k)
            .arg(epsilon)
            .arg(&c1)
            .arg(&k_min)
            .arg(&sb)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// Add `(gamma/nu_t) G_b` to the `omega` equation's sources (SPEC-LIT §17).
/// ACCUMULATES; call after [`omega_sources`].
#[allow(clippy::too_many_arguments)]
pub fn add_buoyancy_to_omega(
    gpu: &Gpu,
    kern: &TurbKernels,
    su: &mut DevBuf<Scalar>,
    sp: &mut DevBuf<Scalar>,
    gb: &DevBuf<Scalar>,
    nut: &DevBuf<Scalar>,
    omega: &DevBuf<Scalar>,
    gamma: Scalar,
    nut_min: Scalar,
    stable_branch: bool,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let sb: Label = if stable_branch { 1 } else { 0 };
    let f = kern.add_buoyancy_omega.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(su)
            .arg(sp)
            .arg(gb)
            .arg(nut)
            .arg(omega)
            .arg(&gamma)
            .arg(&nut_min)
            .arg(&sb)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// [`add_buoyancy_to_omega`] with `gamma` read per cell rather than as one
/// constant - SPEC-LIT §17 and §30.2. SST blends `gamma` between its two
/// coefficient sets with `F1` (SPEC-LIT §6.3), so the production route
/// `(gamma/nu_t) G_b` needs the same blended field the shear production
/// already reads, not the single k-omega `gamma`.
#[allow(clippy::too_many_arguments)]
pub fn add_buoyancy_to_omega_cell(
    gpu: &Gpu,
    kern: &TurbKernels,
    su: &mut DevBuf<Scalar>,
    sp: &mut DevBuf<Scalar>,
    gb: &DevBuf<Scalar>,
    nut: &DevBuf<Scalar>,
    omega: &DevBuf<Scalar>,
    gamma: &DevBuf<Scalar>,
    nut_min: Scalar,
    stable_branch: bool,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let sb: Label = if stable_branch { 1 } else { 0 };
    let f = kern.add_buoyancy_omega_cell.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(su)
            .arg(sp)
            .arg(gb)
            .arg(nut)
            .arg(omega)
            .arg(gamma)
            .arg(&nut_min)
            .arg(&sb)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `Sp = epsilon/k`, `Susp = (2/3)∇·u` - the k equation's sinks
/// (SPEC-LIT §6.1).
#[allow(clippy::too_many_arguments)]
pub fn k_sources(
    gpu: &Gpu,
    kern: &TurbKernels,
    sp: &mut DevBuf<Scalar>,
    susp: &mut DevBuf<Scalar>,
    k: &DevBuf<Scalar>,
    epsilon: &DevBuf<Scalar>,
    div_u: &DevBuf<Scalar>,
    k_min: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = kern.k_sources.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(sp)
            .arg(susp)
            .arg(k)
            .arg(epsilon)
            .arg(div_u)
            .arg(&k_min)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `Su = C_1 (eps/k) G`, `Sp = C_2 eps/k`, `Susp = (2/3 C_1 - C_3)∇·u`
/// (SPEC-LIT §6.1).
#[allow(clippy::too_many_arguments)]
pub fn epsilon_sources(
    gpu: &Gpu,
    kern: &TurbKernels,
    su: &mut DevBuf<Scalar>,
    sp: &mut DevBuf<Scalar>,
    susp: &mut DevBuf<Scalar>,
    g: &DevBuf<Scalar>,
    k: &DevBuf<Scalar>,
    epsilon: &DevBuf<Scalar>,
    div_u: &DevBuf<Scalar>,
    c1: Scalar,
    c2: Scalar,
    c3: Scalar,
    k_min: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = kern.epsilon_sources.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(su)
            .arg(sp)
            .arg(susp)
            .arg(g)
            .arg(k)
            .arg(epsilon)
            .arg(div_u)
            .arg(&c1)
            .arg(&c2)
            .arg(&c3)
            .arg(&k_min)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `Sp = beta*·omega`, `Susp = (2/3)∇·u` - the k equation of the k-omega
/// model (SPEC-LIT §6.2).
#[allow(clippy::too_many_arguments)]
pub fn k_omega_k_sources(
    gpu: &Gpu,
    kern: &TurbKernels,
    sp: &mut DevBuf<Scalar>,
    susp: &mut DevBuf<Scalar>,
    omega: &DevBuf<Scalar>,
    div_u: &DevBuf<Scalar>,
    beta_star: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = kern.k_omega_k_sources.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(sp)
            .arg(susp)
            .arg(omega)
            .arg(div_u)
            .arg(&beta_star)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `Su = gamma (omega/k) G`, `Sp = beta·omega` (SPEC-LIT §6.2).
#[allow(clippy::too_many_arguments)]
pub fn omega_sources(
    gpu: &Gpu,
    kern: &TurbKernels,
    su: &mut DevBuf<Scalar>,
    sp: &mut DevBuf<Scalar>,
    g: &DevBuf<Scalar>,
    k: &DevBuf<Scalar>,
    omega: &DevBuf<Scalar>,
    gamma: Scalar,
    beta: Scalar,
    k_min: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = kern.omega_sources.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(su)
            .arg(sp)
            .arg(g)
            .arg(k)
            .arg(omega)
            .arg(&gamma)
            .arg(&beta)
            .arg(&k_min)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  SPEC-LIT §33 - Launder-Sharma low-Reynolds-number k-epsilon
// ==========================================================================

/// `nu_t = C_mu f_mu k²/epsilon_tilde`, capped at `nut_max` - Launder &
/// Sharma (1974), SPEC-LIT §33.1. `f_mu` is computed inline from
/// `Re_t = k²/(nu epsilon_tilde)`; [`models::launder_sharma::f_mu`] is the
/// same formula on the host, for the analytic limits SPEC-LIT §33.3 asks
/// for.
#[allow(clippy::too_many_arguments)]
pub fn nut_launder_sharma(
    gpu: &Gpu,
    kern: &TurbKernels,
    nut: &mut DevBuf<Scalar>,
    k: &DevBuf<Scalar>,
    epsilon_tilde: &DevBuf<Scalar>,
    nu: Scalar,
    cmu: Scalar,
    nut_max: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = kern.nut_launder_sharma.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(nut)
            .arg(k)
            .arg(epsilon_tilde)
            .arg(&nu)
            .arg(&cmu)
            .arg(&nut_max)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `out[i] = sqrt(max(in[i], 0))`, elementwise - the scratch `sqrt(k)` field
/// SPEC-LIT §33.1's `D` term is gradiented from. Call once for the interior
/// (`n = mesh.n_cells`) and once for the boundary faces
/// (`n = mesh.n_boundary_faces`); the two calls are what fill in a
/// [`crate::field::GpuScalarField`]'s `.f` and `.bf` so
/// [`crate::fv::fvc_grad_scalar`] can be handed it directly.
pub fn ls_sqrt_positive(
    gpu: &Gpu,
    kern: &TurbKernels,
    out: &mut DevBuf<Scalar>,
    input: &DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = kern.ls_sqrt_positive.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(out)
            .arg(input)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `D = 2 nu |grad(sqrt k)|²` (SPEC-LIT §33.1) - the extra sink the
/// `epsilon -> epsilon_tilde` substitution leaves in the `k` equation.
pub fn ls_d_term(
    gpu: &Gpu,
    kern: &TurbKernels,
    d: &mut DevBuf<Scalar>,
    grad_sqrt_k: &DevBuf<Vec3>,
    nu: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = kern.ls_d_term.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(d)
            .arg(grad_sqrt_k)
            .arg(&nu)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `|grad(grad U)|²` (SPEC-LIT §33.1's *DESIGN* note) - the Gauss gradient of
/// the already-computed cell velocity-gradient tensor `grad_u`, specialised
/// to the scalar magnitude the `E` term needs. See `turbLsGradGradUMagSqr`
/// in `cuda/turbulence.cu` for the boundary treatment (`grad U` carries no
/// boundary field of its own, so the boundary contribution extrapolates the
/// owner cell's gradient) and the cost this pays once per outer iteration.
#[allow(clippy::too_many_arguments)]
pub fn ls_grad_grad_u_mag_sqr(
    gpu: &Gpu,
    kern: &TurbKernels,
    out: &mut DevBuf<Scalar>,
    grad_u: &DevBuf<Tensor>,
    m: &GpuMesh,
) -> Result<()> {
    let n = m.n_cells;
    expect_len(out, n, "out")?;
    expect_len(grad_u, n, "grad_u")?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = kern.ls_grad_grad_u_mag_sqr.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(out)
            .arg(grad_u)
            .arg(&m.weights)
            .arg(&m.sf)
            .arg(&m.b_sf)
            .arg(&m.v)
            .arg(&m.owner)
            .arg(&m.neighbour)
            .arg(&m.b_face_cells)
            .arg(&m.b_kind)
            .arg(&m.cf_offset)
            .arg(&m.cf_face)
            .arg(&m.cf_own)
            .arg(&m.bcf_offset)
            .arg(&m.bcf_face)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `E = 2 nu nu_t |grad(grad U)|²` (SPEC-LIT §33.1). `nut` is the PREVIOUS
/// outer iteration's eddy viscosity - the same production-term lag every
/// source in this module uses for `G`.
pub fn ls_e_term(
    gpu: &Gpu,
    kern: &TurbKernels,
    e: &mut DevBuf<Scalar>,
    grad_grad_u_mag_sqr: &DevBuf<Scalar>,
    nut: &DevBuf<Scalar>,
    nu: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = kern.ls_e_term.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(e)
            .arg(grad_grad_u_mag_sqr)
            .arg(nut)
            .arg(&nu)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `Su = C_1 (e~/k) G + E`, `Sp = C_2 f_2 e~/k` - the `epsilon_tilde`
/// equation's sources (SPEC-LIT §33.1). No dilatation `susp` term: §33.1
/// gives none for this model, unlike §6.1's Favre-averaged extension.
#[allow(clippy::too_many_arguments)]
pub fn ls_epsilon_sources(
    gpu: &Gpu,
    kern: &TurbKernels,
    su: &mut DevBuf<Scalar>,
    sp: &mut DevBuf<Scalar>,
    g: &DevBuf<Scalar>,
    k: &DevBuf<Scalar>,
    epsilon_tilde: &DevBuf<Scalar>,
    e_term: &DevBuf<Scalar>,
    nu: Scalar,
    c1: Scalar,
    c2: Scalar,
    k_min: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = kern.ls_epsilon_sources.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(su)
            .arg(sp)
            .arg(g)
            .arg(k)
            .arg(epsilon_tilde)
            .arg(e_term)
            .arg(&nu)
            .arg(&c1)
            .arg(&c2)
            .arg(&k_min)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `out = |a - b|`.
pub fn abs_diff(
    gpu: &Gpu,
    kern: &TurbKernels,
    out: &mut DevBuf<Scalar>,
    a: &DevBuf<Scalar>,
    b: &DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let f = kern.abs_diff.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(out)
            .arg(a)
            .arg(b)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `sqrt(2 |symm(grad U)|²)`, the strain-rate magnitude. Unused by the two
/// models here - both take their production from
/// [`crate::fv::turbulence_production`] - but it is what an SST `nu_t`
/// limiter and every LES delta need (SPEC-LIT §6.3, §6.5).
pub fn strain_rate_mag(
    gpu: &Gpu,
    kern: &TurbKernels,
    out: &mut DevBuf<Scalar>,
    grad_u: &DevBuf<Tensor>,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(out, n, "out")?;
    expect_len(grad_u, n, "grad_u")?;

    let nl = n as Label;
    let f = kern.strain_rate.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(out)
            .arg(grad_u)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `sqrt(2 |skew(grad U)|²)`, the vorticity magnitude - SPEC-LIT §56.2.
///
/// **Not** [`strain_rate_mag`], and the difference is not a factor: in a pure
/// shear the two agree, and in an irrotational strain this is zero while `S`
/// is not. Spalart-Allmaras's production term (SPEC-LIT (56.3)) takes this
/// one, and taking `S` there is a silent error a log-layer test cannot catch.
pub fn vorticity_mag(
    gpu: &Gpu,
    kern: &TurbKernels,
    out: &mut DevBuf<Scalar>,
    grad_u: &DevBuf<Tensor>,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(out, n, "out")?;
    expect_len(grad_u, n, "grad_u")?;

    let nl = n as Label;
    let f = kern.vorticity.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(out)
            .arg(grad_u)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `sqrt(sum_ij (dU_j/dx_i)²)`, the Frobenius norm of the FULL velocity
/// gradient - SPEC-LIT §56.2, and the denominator of SPEC-LIT (57.7)'s `r_d`,
/// `r_dt` and `r_dl`.
///
/// A third invariant, distinct from both [`strain_rate_mag`] and
/// [`vorticity_mag`], satisfying `F² = (S² + Omega²)/2` identically. It is
/// computed from the nine components rather than from that identity so the
/// identity remains something `tests` can check.
pub fn grad_frobenius(
    gpu: &Gpu,
    kern: &TurbKernels,
    out: &mut DevBuf<Scalar>,
    grad_u: &DevBuf<Tensor>,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(out, n, "out")?;
    expect_len(grad_u, n, "grad_u")?;

    let nl = n as Label;
    let f = kern.grad_frobenius.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(out)
            .arg(grad_u)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  RasCore
// ==========================================================================

/// Everything two turbulence models have in common: the kernels, the matrix,
/// the linear-solver workspace, the wall-function tables, `nu_t`, and the
/// scratch each transport equation needs.
///
/// Allocated once. Nothing in an outer iteration allocates, both because a
/// time loop should not and because a CUDA graph cannot capture an
/// allocation at all.
/// **SPEC-LIT §86.3.** What turns `ddt(psi) + div(phi, psi)` into
/// `ddt(rho, psi) + div(phi_m, psi)` on the ONE assembly
/// [`RasCore::assemble_transport`] and its siblings share.
///
/// The `phi` carried in the [`FlowState`] handed alongside this is the MASS
/// flux `phi_m = rho_f phi`, so every convective line of that assembly - the
/// scheme weights, `fvm_div_gauss`, the bounded correction, the deferred
/// correction - is on the mass flux with no branch at all. What this struct
/// supplies is the three things that are NOT a flux: the density time levels
/// the `ddt` carries, the face density the laplacian is weighted by, and
/// (86.4)'s continuity coefficient.
pub struct MassWeighting<'a> {
    /// `rho^n`, `[n_cells]`.
    pub rho: &'a DevBuf<Scalar>,
    /// `rho^{n-1}`, `[n_cells]`.
    pub rho0: &'a DevBuf<Scalar>,
    /// `rho^{n-2}`, `[n_cells]`.
    pub rho00: &'a DevBuf<Scalar>,

    /// `rho_f` on every face, `[n_if]`/`[n_bf]`. The diffusivity
    /// `Gamma_eff |Sf|` is multiplied by it, which turns
    /// `laplacian(D_eff, Y)` into `laplacian(rho D_eff, Y)`.
    pub rho_face: &'a GpuSurfaceScalarField,

    /// `[n_cells]` `a_N rho + a_0 rho^0 + a_00 rho^00` -
    /// [`crate::timescheme::Ddt::rho_continuity`], the ddt half of the
    /// discrete continuity residual. Read ONLY when the case asked for
    /// `bounded`; see (86.4).
    pub cont_ddt: &'a DevBuf<Scalar>,
}

pub struct RasCore<'m> {
    pub mesh: &'m GpuMesh,
    pub ctrl: TurbulenceControls,
    pub wall: WallFunctionCoeffs,

    pub fv: FvKernels,
    pub fld: FieldKernels,
    pub ldu: LduKernels,
    pub sol: SolverKernels,
    pub turb: TurbKernels,

    /// The `ddt` term of every equation this core assembles - SPEC-LIT 13.
    pub ddt: Ddt,

    /// Which cells the wall functions own.
    pub wd: WallData,
    pub a: GpuLduMatrix,
    pub ws: SolverWorkspace,

    /// The eddy viscosity, which is the model's output and the momentum
    /// equation's input.
    pub nut: GpuScalarField,

    /// `[n_cells]` `grad U`, component `(i,j) = dU_j/dx_i`.
    pub grad_u: DevBuf<Tensor>,
    /// `[n_cells]` the production term `G`.
    pub g: DevBuf<Scalar>,
    /// `[n_cells]` `∇·u`, zero for a discretely conservative flux.
    pub div_u: DevBuf<Scalar>,

    /// Buoyancy production, when the case has gravity and a temperature -
    /// SPEC-LIT §17. `None` leaves every equation exactly as it was, which is
    /// what an isothermal or zero-gravity run wants.
    pub buoyancy: Option<BuoyancyProduction>,
    /// `[n_cells]` `G_b`, and the `C_3` that goes with it.
    pub gb: DevBuf<Scalar>,
    pub c3: DevBuf<Scalar>,
    /// `[n_cells]` `grad(T)`, rebuilt from the temperature each outer
    /// iteration.
    grad_t: DevBuf<Vec3>,

    /// `[n_if]` / `[n_bf]` `Gamma_eff·|Sf|`.
    gamma: DevBuf<Scalar>,
    b_gamma: DevBuf<Scalar>,
    /// `[n_if]` / `[n_bf]` convection weights.
    wts: DevBuf<Scalar>,
    b_wts: DevBuf<Scalar>,
    /// `[n_cells]` gradient of the transported scalar - the limited schemes
    /// and the non-orthogonal correction both need it.
    grad_psi: DevBuf<Vec3>,

    /// `[n_cells]` linearised source coefficients, filled by the model's own
    /// source kernel and handed straight to [`crate::fv::fvm_su`],
    /// [`crate::fv::fvm_sp`] and [`crate::fv::fvm_susp`].
    ///
    /// Public because they are the one part of the shared machinery a model
    /// writes rather than reads, and hiding them behind three setters would
    /// only move the same three lines.
    pub su: DevBuf<Scalar>,
    pub sp: DevBuf<Scalar>,
    pub susp: DevBuf<Scalar>,

    /// `[n_cells]` `k` as it was at the top of the last `correct`, and the
    /// scratch the change is formed in.
    k_prev: DevBuf<Scalar>,
    diff: DevBuf<Scalar>,
    /// Two one-element landing pads for the convergence reduction.
    red_a: DevBuf<Scalar>,
    red_b: DevBuf<Scalar>,
}

impl<'m> RasCore<'m> {
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        mesh: &'m GpuMesh,
        ctrl: TurbulenceControls,
        wall: WallFunctionCoeffs,
        wall_faces: &crate::field_setup::WallFaces,
        roughness: &crate::field_setup::NutRoughness,
    ) -> Result<Self> {
        if hm.n_cells != mesh.n_cells || hm.n_boundary_faces != mesh.n_boundary_faces {
            return Err(Error::Config(format!(
                "RasCore::new: the host mesh has ({}, {}) cells/boundary faces \
                 and the device mesh ({}, {})",
                hm.n_cells, hm.n_boundary_faces, mesh.n_cells, mesh.n_boundary_faces
            )));
        }

        // A degenerate mesh still gets one-element buffers: a zero-length
        // device allocation is an error, and every launcher returns early on
        // `n == 0` so nothing reads them.
        let nc = mesh.n_cells.max(1);
        let nif = mesh.n_internal_faces.max(1);
        let nbf = mesh.n_boundary_faces.max(1);

        Ok(Self {
            mesh,
            ctrl,
            wall,

            fv: FvKernels::new(gpu)?,
            fld: FieldKernels::new(gpu)?,
            ldu: LduKernels::new(gpu)?,
            sol: SolverKernels::new(gpu)?,
            turb: TurbKernels::new(gpu)?,

            // The time scheme `ddtSchemes` named, in full. This used to be a
            // single `1/dt` that was zero when steady, which turned
            // `backward` and `localEuler` into first-order Euler in silence
            // (SPEC-LIT 13.3, 13.4).
            // `relaxed = k_relax < 1.0`: this is also the ddt a passive
            // scalar built on `RasCore` (`scalar_transport::Transport`) gets,
            // so `CrankNicolson` becomes reachable there too whenever the
            // case leaves that equation's relaxation factor at 1 (SPEC-LIT
            // §13.1, `timescheme::Ddt::new_with_relax`).
            ddt: Ddt::new_with_relax(
                gpu,
                mesh,
                ctrl.ddt.reconciled(ctrl.steady),
                ctrl.delta_t,
                ctrl.lts,
                ctrl.k_relax < 1.0,
            )?,

            // SPEC-LIT §29.1/§29.2: `roughness` is `nut`'s own per-face
            // `Ks`/`Cs` (or [`crate::field_setup::NutRoughness::none`] where
            // the caller has none - a passive scalar's `RasCore`, or a case
            // with no `nut` file) - see `WallData::build`'s own doc for why
            // this is read from `nut` alone, never derived from another
            // field (SPEC-LIT §15.5).
            wd: WallData::build(gpu, hm, wall_faces, roughness)?,
            a: GpuLduMatrix::new(gpu, mesh)?,
            ws: SolverWorkspace::for_mesh(gpu, mesh)?,

            nut: GpuScalarField::zeros(gpu, mesh, "nut")?,

            grad_u: gpu.zeros(nc)?,
            g: gpu.zeros(nc)?,
            div_u: gpu.zeros(nc)?,

            buoyancy: None,
            gb: gpu.zeros(nc)?,
            c3: gpu.zeros(nc)?,
            grad_t: gpu.zeros(nc)?,

            gamma: gpu.zeros(nif)?,
            b_gamma: gpu.zeros(nbf)?,
            wts: gpu.zeros(nif)?,
            b_wts: gpu.zeros(nbf)?,
            grad_psi: gpu.zeros(nc)?,

            su: gpu.zeros(nc)?,
            sp: gpu.zeros(nc)?,
            susp: gpu.zeros(nc)?,

            k_prev: gpu.zeros(nc)?,
            diff: gpu.zeros(nc)?,
            red_a: gpu.zeros(1)?,
            red_b: gpu.zeros(1)?,
        })
    }

    /// `nu_t = 0` everywhere, boundary faces included, and the Robin triple
    /// rewritten so nothing can put a value back.
    ///
    /// What `simulationType laminar;` and `RAS { turbulence off; }` mean.
    /// Both used to be read and discarded, so the model ran regardless and the
    /// momentum equation saw an eddy viscosity the case had switched off. The
    /// boundary triple is set to `fr = 1, refValue = 0` rather than left
    /// alone, because a zero-gradient `nut` face would otherwise pick the
    /// internal value back up the next time boundary conditions are
    /// evaluated - and on a wall-function patch `turbNutBoundary` would do it
    /// unasked.
    pub fn freeze_nut(&mut self, gpu: &Gpu) -> Result<()> {
        let zeros_c = vec![0.0 as Scalar; self.nut.f.len()];
        let zeros_b = vec![0.0 as Scalar; self.nut.bf.len()];
        let ones_b = vec![1.0 as Scalar; self.nut.fr.len()];

        gpu.write(&mut self.nut.f, &zeros_c)?;
        gpu.write(&mut self.nut.f0, &zeros_c)?;
        gpu.write(&mut self.nut.bf, &zeros_b)?;
        gpu.write(&mut self.nut.fr, &ones_b)?;
        gpu.write(&mut self.nut.ref_value, &zeros_b)?;
        gpu.write(&mut self.nut.ref_grad, &zeros_b)?;
        Ok(())
    }

    /// The eddy-viscosity ceiling of SPEC-LIT §6.1's *DESIGN* note:
    /// `nut_max = nutMaxCoeff · nu`.
    ///
    /// Expressed as a multiple of the molecular viscosity rather than as an
    /// absolute number because that is the only form that means the same
    /// thing in air and in water.
    #[inline]
    pub fn nut_max(&self, nu: Scalar) -> Scalar {
        self.ctrl.nut_max_coeff * nu
    }

    /// `grad U`, `G` and `∇·u` - the three fields both models' sources are
    /// built from.
    ///
    /// `G` uses the `nu_t` of the *previous* outer iteration, which is the
    /// standard segregated lag: the production a cell sees is the one implied
    /// by the eddy viscosity the momentum equation was solved with.
    pub fn update_flow_derived(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        let n = self.mesh.n_cells;

        crate::fv::fvc_grad_vector(gpu, &self.fv, &mut self.grad_u, flow.u, self.mesh)?;
        crate::fv::turbulence_production(gpu, &self.fv, &mut self.g, &self.nut.f, &self.grad_u, n)?;
        fvc_div_surface(gpu, &self.fv, &mut self.div_u, flow.phi, self.mesh)?;

        Ok(())
    }

    /// Assemble `ddt(psi) + div(phi, psi) - laplacian((nu + r_sigma·nu_t), psi)`
    /// into [`Self::a`], which is zeroed first.
    ///
    /// The source terms are the caller's to add afterwards; they are the only
    /// thing that distinguishes one turbulence equation from another.
    ///
    /// `r_sigma` is `1/sigma` rather than `sigma` because that is what the
    /// diffusivity expression multiplies by, and because passing the
    /// reciprocal makes `sigma = 0` - a meaningless setting - impossible to
    /// express rather than a division by zero on the device.
    ///
    /// `conv` is THIS equation's `divSchemes` entry. It is a parameter and not
    /// a field of the controls because `div(phi,k)` and `div(phi,epsilon)` are
    /// two dictionary entries and a case is entitled to differ between them -
    /// the reader this replaced took whichever it found first and used it for
    /// every equation in the run, momentum included.
    /// The controls this core was built with.
    pub fn controls(&self) -> &TurbulenceControls {
        &self.ctrl
    }

    /// `grad(T)`, then `G_b` and `C_3` - SPEC-LIT §17.
    ///
    /// A no-op when [`Self::buoyancy`] is `None` or gravity is zero, and in
    /// that case `gb` keeps whatever it held, which nothing then reads.
    ///
    /// `t` must have had its boundary conditions evaluated since its internal
    /// field last changed: the Green-Gauss gradient reads `t.bf` directly, and
    /// a stale boundary value near a hot inlet is exactly where the sign of
    /// this term is decided.
    pub fn update_buoyancy_production(
        &mut self,
        gpu: &Gpu,
        t: &GpuScalarField,
        u: &GpuVectorField,
    ) -> Result<bool> {
        let Some(b) = self.buoyancy else {
            return Ok(false);
        };
        if !b.is_active() {
            return Ok(false);
        }
        let n = self.mesh.n_cells;
        if n == 0 {
            return Ok(false);
        }

        crate::fv::fvc_grad_scalar(gpu, &self.fv, &mut self.grad_t, t, self.mesh)?;

        let Self { turb, gb, c3, nut, grad_t, .. } = self;
        buoyancy_production(gpu, turb, gb, c3, &nut.f, grad_t, &t.f, &u.f, &b, n)?;
        Ok(true)
    }

    pub fn assemble_transport(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        psi: &GpuScalarField,
        conv: DivEntry,
        r_sigma: Scalar,
    ) -> Result<()> {
        self.a.zero(gpu)?;

        face_diffusivity(
            gpu,
            &self.turb,
            &mut self.gamma,
            &mut self.b_gamma,
            &self.nut,
            self.mesh,
            flow.nu,
            r_sigma,
        )?;

        self.assemble_after_diffusivity(gpu, flow, psi, conv, None)
    }

    /// [`Self::assemble_transport`] for a MASS-WEIGHTED equation -
    /// `SPEC-LIT` §86.3:
    ///
    /// ```text
    /// ddt(rho, psi) + div(phi_m, psi) - laplacian(rho Gamma_eff, psi) = 0
    /// ```
    ///
    /// `flow.phi` must already BE the mass flux `phi_m = rho_f phi`; this
    /// function does not build it, because the whole point of §86.2 is that
    /// the flux is one object shared with the equation that already convects
    /// with it rather than a second one assembled here.
    ///
    /// Everything downstream of the diffusivity is literally the same code
    /// [`Self::assemble_transport`] reaches - the same scheme weights, the
    /// same `fvm_div_gauss`, the same deferred correction - because it is the
    /// same equation with a different flux in it. The two differences are
    /// named in `assemble_after_diffusivity` and are the only places `mass`
    /// is read.
    pub fn assemble_transport_mass_weighted(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        psi: &GpuScalarField,
        conv: DivEntry,
        r_sigma: Scalar,
        mass: &MassWeighting<'_>,
    ) -> Result<()> {
        self.a.zero(gpu)?;

        face_diffusivity(
            gpu,
            &self.turb,
            &mut self.gamma,
            &mut self.b_gamma,
            &self.nut,
            self.mesh,
            flow.nu,
            r_sigma,
        )?;

        // `Gamma_eff |Sf|` -> `rho_f Gamma_eff |Sf|`. The laminar half is
        // then `rho D` and the turbulent half `rho nu_t/Sc_t = mu_t/Sc_t`,
        // which is what a mass fraction in `rho Y` diffuses with.
        crate::field_ops::multiply_field(
            gpu,
            &self.fld,
            &mut self.gamma,
            &mass.rho_face.f,
            self.mesh.n_internal_faces,
        )?;
        crate::field_ops::multiply_field(
            gpu,
            &self.fld,
            &mut self.b_gamma,
            &mass.rho_face.bf,
            self.mesh.n_boundary_faces,
        )?;

        self.assemble_after_diffusivity(gpu, flow, psi, conv, Some(mass))
    }

    /// [`Self::assemble_transport`] with the diffusivity `a·nu + b·nu_t` -
    /// SPEC-LIT §41.2.
    ///
    /// `assemble_transport(r_sigma)` is `assemble_transport_affine(1, r_sigma)`
    /// bit for bit (see [`face_diffusivity_affine`]), so §40 reaches the same
    /// arithmetic through this entry point as §6.1 does through the other, and
    /// §41 reaches `alpha (nu + nu_t)` - which the other cannot express at all.
    pub fn assemble_transport_affine(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        psi: &GpuScalarField,
        conv: DivEntry,
        a: Scalar,
        b: Scalar,
    ) -> Result<()> {
        self.a.zero(gpu)?;

        face_diffusivity_affine(
            gpu,
            &self.turb,
            &mut self.gamma,
            &mut self.b_gamma,
            &self.nut,
            self.mesh,
            flow.nu,
            a,
            b,
        )?;

        self.assemble_after_diffusivity(gpu, flow, psi, conv, None)
    }

    /// [`Self::assemble_transport`] with `sigma` read per cell rather than
    /// passed as a constant - SPEC-LIT §6.3.
    ///
    /// SST's diffusivity multiplier is `F1·sigma_1 + (1 - F1)·sigma_2`, which
    /// is a field. Everything downstream of the diffusivity - the convection
    /// scheme, the time scheme, the bounded correction, the deferred
    /// correction, the laplacian and its non-orthogonal correction - is
    /// literally the same code, because it is the same equation; only the
    /// number multiplying `nu_t` on a face differs.
    pub fn assemble_transport_blended(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        psi: &GpuScalarField,
        conv: DivEntry,
        r_sigma: &DevBuf<Scalar>,
    ) -> Result<()> {
        self.a.zero(gpu)?;

        face_diffusivity_cell(
            gpu,
            &self.turb,
            &mut self.gamma,
            &mut self.b_gamma,
            &self.nut,
            r_sigma,
            self.mesh,
            flow.nu,
        )?;

        self.assemble_after_diffusivity(gpu, flow, psi, conv, None)
    }

    /// [`Self::assemble_transport`] with the face diffusivity supplied by
    /// the CALLER - SPEC-LIT §56.6.
    ///
    /// The three entry points above all build `Gamma_eff` from `nu_t`.
    /// Spalart-Allmaras's is `(nu + nu~ f_n)/sigma`, built from the
    /// TRANSPORTED field, which none of them can express. Rather than adding
    /// a fourth `face_diffusivity_*` here - the coefficient belongs to that
    /// model, not to this shared core - the caller fills the two face buffers
    /// through `fill` and everything downstream is literally the same code,
    /// because it is the same equation.
    ///
    /// `fill` is handed `(gamma, b_gamma, mesh)`: `[n_internal_faces]` and
    /// `[n_boundary_faces]` of `Gamma_eff·|Sf|`, the product
    /// [`crate::fv::fvm_laplacian`] takes.
    pub fn assemble_transport_with_face_diffusivity<F>(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        psi: &GpuScalarField,
        conv: DivEntry,
        fill: F,
    ) -> Result<()>
    where
        F: FnOnce(&mut DevBuf<Scalar>, &mut DevBuf<Scalar>, &GpuMesh) -> Result<()>,
    {
        self.a.zero(gpu)?;
        fill(&mut self.gamma, &mut self.b_gamma, self.mesh)?;
        self.assemble_after_diffusivity(gpu, flow, psi, conv, None)
    }

    /// Everything an eddy-viscosity transport equation does once its face
    /// diffusivity exists. Split out so the constant-`sigma` and blended-
    /// `sigma` entry points cannot drift apart: there is one assembly, and the
    /// only thing the two callers choose is how `Gamma_eff` was formed.
    fn assemble_after_diffusivity(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
        psi: &GpuScalarField,
        conv: DivEntry,
        // SPEC-LIT §86.3. `None` is every line of this function as it stood
        // before §86, and is what all four constant-density entry points
        // above pass: the `match` and the `if let` below are the WHOLE
        // difference, which is what makes the constant-density path bitwise
        // identical by construction rather than by comparison (§86.6).
        mass: Option<&MassWeighting<'_>>,
    ) -> Result<()> {
        let scheme: crate::fv::DivScheme = conv.scheme.into();

        // A limited scheme reads the upwind cell's gradient to form
        // `r = 2(d·grad psi_U)/(psi_N - psi_P) - 1`, and a deferred-correction
        // scheme (SPEC-LIT §11.1) needs the same gradient for its explicit
        // half, so it has to exist before the weights do.
        if scheme.needs_gradient() {
            fvc_grad_scalar_scheme(
                gpu,
                &self.fv,
                &mut self.grad_psi,
                psi,
                self.mesh,
                self.ctrl.grad_scheme,
            )?;
        }

        div_scheme_weights(
            gpu,
            &self.fv,
            Some(&mut self.wts),
            Some(&mut self.b_wts),
            scheme,
            flow.phi,
            psi,
            if scheme.needs_gradient() {
                Some(&self.grad_psi)
            } else {
                None
            },
            self.mesh,
        )?;

        // `localEuler` needs the local step rebuilt from the flux this
        // iteration produced (SPEC-LIT 13.2); every other scheme ignores this.
        self.ddt.update_local_step(gpu, flow.phi, self.mesh)?;

        // SPEC-LIT 13: whichever scheme `ddtSchemes` named. Both old levels
        // are passed even for Euler, which ignores the second - a caller that
        // has only one cannot ask for `backward` and be quietly given Euler.
        match mass {
            None => self.ddt.add(gpu, &mut self.a, self.mesh, &psi.f0, &psi.f00, 1.0)?,
            // SPEC-LIT §86.3: `ddt(rho, psi)`, each old level carrying its
            // OWN density, which is what makes the discrete term conserve
            // `rho psi` rather than `psi`.
            Some(mw) => self.ddt.add_rho(
                gpu,
                &mut self.a,
                self.mesh,
                mw.rho,
                mw.rho0,
                mw.rho00,
                &psi.f0,
                &psi.f00,
                1.0,
            )?,
        }

        fvm_div_gauss(
            gpu,
            &self.fv,
            &mut self.a,
            self.mesh,
            flow.phi,
            &self.wts,
            &self.b_wts,
            psi,
            1.0,
        )?;

        // `bounded Gauss ...`: subtract the spurious source a non-solenoidal
        // flux injects (SPEC-LIT §3.1). Costs one pass and is identically zero
        // when phi conserves mass.
        if conv.bounded {
            fvm_div_bounded_correction(gpu, &self.fv, &mut self.a, self.mesh, flow.phi, 1.0)?;

            // SPEC-LIT (86.4): on a mass-weighted equation the discrete
            // continuity residual has TWO halves, and the line above is only
            // the flux one. Subtracting `psi_P` times the other half as well
            // is what makes `bounded` preserve a uniform field EXACTLY - with
            // `psi = 1` the whole row is then `R_P - R_P = 0` - which is the
            // property `bounded` exists for and which it does NOT have on a
            // variable-density equation without this term. It is also
            // precisely §86.4's second half, "the d(rho)/dt half of the
            // difference above, which `bounded` does not touch".
            if let Some(mw) = mass {
                crate::fv::fvm_sp(gpu, &self.fv, &mut self.a, self.mesh, mw.cont_ddt, -1.0)?;
            }
        }

        // The explicit half of `linearUpwind`/`cubic` (SPEC-LIT §11.1). The
        // gradient is the one the weights pass just built.
        if scheme.correction().is_some() {
            fvm_div_correction(
                gpu,
                &self.fv,
                &mut self.a,
                self.mesh,
                flow.phi,
                &self.grad_psi,
                scheme,
                1.0,
            )?;
        }

        fvm_laplacian(
            gpu,
            &self.fv,
            &mut self.a,
            self.mesh,
            &self.gamma,
            &self.b_gamma,
            psi,
            -1.0,
        )?;

        if self.ctrl.sn_grad.applies() {
            fvc_grad_scalar_scheme(
                gpu,
                &self.fv,
                &mut self.grad_psi,
                psi,
                self.mesh,
                self.ctrl.grad_scheme,
            )?;
            fvm_laplacian_non_orth_correction(
                gpu,
                &self.fv,
                &mut self.a,
                self.mesh,
                &self.gamma,
                &self.b_gamma,
                psi,
                &self.grad_psi,
                self.ctrl.sn_grad,
                -1.0,
            )?;
        }

        Ok(())
    }

    /// Under-relax, apply the wall constraint if the equation has one, fold
    /// the boundary coefficients in, solve, and refresh the boundary values.
    ///
    /// The order is not free. [`relax`] must see the *unfolded* diagonal, so
    /// it comes before [`add_boundary_contributions`]; the wall constraint
    /// must come after the relaxation that would otherwise re-open the row it
    /// closes, and before the fold that would add coefficients back into it.
    pub fn solve_equation(
        &mut self,
        gpu: &Gpu,
        psi: &mut GpuScalarField,
        alpha: Scalar,
        sc: &SolverControls,
        constrain_walls: bool,
    ) -> Result<SolverPerformance> {
        self.solve_equation_with(gpu, psi, alpha, sc, constrain_walls, false)
    }

    /// [`Self::solve_equation`] with the fixed-value constraint of SPEC-LIT
    /// §18 as well.
    ///
    /// `constrain_fixed` runs [`crate::ldu_ops::set_values`] on the flags a
    /// [`crate::sources::SourceSet`] wrote. It happens AFTER the boundary fold
    /// and after relaxation, which is the only order that means anything: a
    /// row eliminated before the fold would have the boundary coefficients
    /// added straight back into it, and a row eliminated before relaxation
    /// would then have its pinned diagonal divided by alpha.
    pub fn solve_equation_with(
        &mut self,
        gpu: &Gpu,
        psi: &mut GpuScalarField,
        alpha: Scalar,
        sc: &SolverControls,
        constrain_walls: bool,
        constrain_fixed: bool,
    ) -> Result<SolverPerformance> {
        relax(gpu, &self.ldu, &mut self.a, self.mesh, &psi.f, alpha)?;

        if constrain_walls {
            constrain_wall_cells(gpu, &self.ldu, &mut self.a, self.mesh, &self.wd)?;
        }

        add_boundary_contributions(gpu, &self.ldu, &mut self.a, self.mesh)?;

        if constrain_fixed {
            crate::ldu_ops::set_values(gpu, &self.ldu, &mut self.a, self.mesh)?;
        }

        // `solvers/<var>/solver` is honoured rather than discarded: PCG when
        // the case asked for it and the matrix is symmetric, PBiCGStab
        // otherwise, and an error where the two cannot be reconciled
        // (SPEC-LIT 8.2, 13.4). A transport equation carrying convection is
        // asymmetric, so `solver PCG;` on one is now refused instead of
        // running BiCGStab and saying nothing.
        crate::solver::solve(gpu, &self.sol, &mut psi.f, &self.a, self.mesh, &mut self.ws, sc)
    }

    /// `k_prev <- k`. Called at the top of every `correct`, so that
    /// [`Self::convergence_measure`] reports the change across whatever span
    /// of iterations the caller chose to check over.
    pub fn store_k_prev(&mut self, gpu: &Gpu, k: &DevBuf<Scalar>) -> Result<()> {
        crate::field_ops::copy_field(gpu, &self.fld, &mut self.k_prev, k, self.mesh.n_cells)
    }

    /// `max|k - k_prev| / max|k|`, the steady-state convergence measure.
    ///
    /// Relative to the largest `k` in the field rather than cell by cell:
    /// a per-cell ratio is dominated by the quietest corner of the domain,
    /// where `k` is at its floor and any change at all is enormous relative to
    /// it, and a run would then never be declared converged.
    ///
    /// **This is the only function in the module that touches the host.** Two
    /// eight-byte reads, once every `convergence_check_every` iterations, and
    /// never from inside a captured graph.
    pub fn convergence_measure(&mut self, gpu: &Gpu, k: &DevBuf<Scalar>) -> Result<Scalar> {
        let n = self.mesh.n_cells;
        if n == 0 {
            return Ok(0.0);
        }

        abs_diff(gpu, &self.turb, &mut self.diff, k, &self.k_prev, n)?;
        device_max_mag(gpu, &self.sol, &mut self.red_a, &self.diff, &mut self.ws.partials, n)?;
        device_max_mag(gpu, &self.sol, &mut self.red_b, k, &mut self.ws.partials, n)?;

        let change = gpu.download(&self.red_a)?;
        let scale = gpu.download(&self.red_b)?;

        let d = change.first().copied().unwrap_or(0.0);
        let s = scale.first().copied().unwrap_or(0.0);

        Ok(if s > 0.0 { d / s } else { d })
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::BcKind;

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// `cuda/turbulence.cu` hard-codes these two numbers to decide which
    /// boundary faces the model owns. If the enum moves, `nu_t` silently
    /// stops being written on half the boundary.
    #[test]
    fn bc_kind_values_match_the_device() {
        assert_eq!(BcKind::Calculated as Label, 4);
        assert_eq!(BcKind::NutkWallFunction as Label, 20);
        // Every wall-function kind must sort at or above the threshold the
        // kernel tests against.
        for k in [
            BcKind::NutkWallFunction,
            BcKind::NutUWallFunction,
            BcKind::NutLowReWallFunction,
            BcKind::EpsilonWallFunction,
            BcKind::OmegaWallFunction,
            BcKind::KqRWallFunction,
            BcKind::KLowReWallFunction,
        ] {
            assert!(k as Label >= 20, "{k:?} sorts below the wall-function range");
        }
    }

    // ----------------------------------------------------------------------
    //  The blended diffusivity - SPEC-LIT §6.3
    // ----------------------------------------------------------------------

    /// [`face_diffusivity_cell`] must be [`face_diffusivity`] when `sigma`
    /// happens to be uniform, bit for bit.
    ///
    /// They are the same expression in the same multiplication order, and the
    /// only difference is where `rSigma` is read from - so anything but exact
    /// equality would mean one of the two had drifted. That matters because
    /// SST at `F_1 = 1` must reproduce k-omega exactly
    /// (`models::k_omega_sst::tests::forcing_f1_to_one_reproduces_k_omega`
    /// measures it to 1e-11 relative), and it cannot if the two diffusivity
    /// kernels disagree in the last bit.
    #[test]
    fn blended_diffusivity_matches_the_uniform_one() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let (mut hm, points, faces) =
            crate::mesh::topology::tests::box_mesh([4, 4, 1], Vec3::new(0.25, 0.3, 0.2));
        hm.compute_geometry(&points, &faces).expect("geometry");
        hm.build_cell_face_maps();
        let m = GpuMesh::upload(&gpu, &hm)?;

        let kern = TurbKernels::new(&gpu)?;
        let mut nut = GpuScalarField::zeros(&gpu, &m, "nut")?;

        // A varying nu_t, so the interpolation weights are actually exercised
        // rather than cancelling.
        let cells: Vec<Scalar> = (0..hm.n_cells)
            .map(|c| 1e-3 + 1e-4 * c as Scalar)
            .collect();
        let bfaces: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|b| 2e-3 + 5e-5 * b as Scalar)
            .collect();
        gpu.write(&mut nut.f, &cells)?;
        gpu.write(&mut nut.bf, &bfaces)?;

        let nu: Scalar = 1.5e-5;
        let sigma: Scalar = 0.856;
        let uniform = gpu.upload(&vec![sigma; hm.n_cells])?;

        let nif = hm.n_internal_faces.max(1);
        let nbf = hm.n_boundary_faces.max(1);

        let mut a_i: DevBuf<Scalar> = gpu.zeros(nif)?;
        let mut a_b: DevBuf<Scalar> = gpu.zeros(nbf)?;
        let mut b_i: DevBuf<Scalar> = gpu.zeros(nif)?;
        let mut b_b: DevBuf<Scalar> = gpu.zeros(nbf)?;

        face_diffusivity(&gpu, &kern, &mut a_i, &mut a_b, &nut, &m, nu, sigma)?;
        face_diffusivity_cell(&gpu, &kern, &mut b_i, &mut b_b, &nut, &uniform, &m, nu)?;
        gpu.sync()?;

        let (ha_i, hb_i) = (gpu.download(&a_i)?, gpu.download(&b_i)?);
        let (ha_b, hb_b) = (gpu.download(&a_b)?, gpu.download(&b_b)?);

        for (f, (x, y)) in ha_i.iter().zip(&hb_i).enumerate() {
            assert_eq!(x, y, "internal face {f}: {x} against {y}");
        }
        for (f, (x, y)) in ha_b.iter().zip(&hb_b).enumerate() {
            assert_eq!(x, y, "boundary face {f}: {x} against {y}");
        }

        // And it must genuinely vary with sigma, or the comparison above is
        // between two constants.
        let doubled = gpu.upload(&vec![2.0 * sigma; hm.n_cells])?;
        face_diffusivity_cell(&gpu, &kern, &mut b_i, &mut b_b, &nut, &doubled, &m, nu)?;
        gpu.sync()?;
        let c_i = gpu.download(&b_i)?;
        assert!(
            c_i.iter().zip(&ha_i).any(|(x, y)| (x - y).abs() > 1e-12 * y.abs()),
            "doubling sigma changed no face diffusivity"
        );

        Ok(())
    }

    /// SPEC-LIT §41.2 and §41.6: `face_diffusivity_affine(1, r)` must be
    /// [`face_diffusivity`] **bit for bit**, or the one kernel §40 and §41
    /// both go through is not the kernel §6.1 has been validated with.
    ///
    /// `a·nu` with `a` an exact `1.0` is exact in IEEE-754, so this is a
    /// statement about the arithmetic ORDER as much as about the value - the
    /// two kernels form `w·gP + (1-w)·gN` in the same order and multiply by
    /// `|Sf|` at the same point, which is the part that would drift if one of
    /// them were rewritten.
    ///
    /// And the second half: `affine(alpha, alpha)` must NOT equal
    /// `face_diffusivity(alpha)`, because `alpha(nu + nu_t)` is not
    /// `nu + alpha nu_t`. That is the whole reason the kernel exists, and a
    /// test that only checked the reduction would pass on a kernel that
    /// ignored `a` entirely.
    #[test]
    fn the_affine_diffusivity_reduces_to_the_plain_one_bitwise() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let (mut hm, points, faces) =
            crate::mesh::topology::tests::box_mesh([4, 4, 1], Vec3::new(0.25, 0.3, 0.2));
        hm.compute_geometry(&points, &faces).expect("geometry");
        hm.build_cell_face_maps();
        let m = GpuMesh::upload(&gpu, &hm)?;

        let kern = TurbKernels::new(&gpu)?;
        let mut nut = GpuScalarField::zeros(&gpu, &m, "nut")?;

        // Awkward magnitudes on purpose: a nu_t that is sometimes far above
        // and sometimes far below nu, so `a*nu` cannot be lost in the sum.
        let cells: Vec<Scalar> = (0..hm.n_cells)
            .map(|c| 1e-7 * (1.0 + 37.0 * c as Scalar).powi(3))
            .collect();
        let bfaces: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|b| 2e-6 + 5e-5 * (b % 13) as Scalar)
            .collect();
        gpu.write(&mut nut.f, &cells)?;
        gpu.write(&mut nut.bf, &bfaces)?;

        let nu: Scalar = 1.5e-5;
        let nif = hm.n_internal_faces.max(1);
        let nbf = hm.n_boundary_faces.max(1);

        let mut a_i: DevBuf<Scalar> = gpu.zeros(nif)?;
        let mut a_b: DevBuf<Scalar> = gpu.zeros(nbf)?;
        let mut b_i: DevBuf<Scalar> = gpu.zeros(nif)?;
        let mut b_b: DevBuf<Scalar> = gpu.zeros(nbf)?;

        for r_sigma in [1.0 as Scalar, 1.0 / 1.2, 1.0 / 1.3, 1.39, 0.769_230_769_230_769_2] {
            face_diffusivity(&gpu, &kern, &mut a_i, &mut a_b, &nut, &m, nu, r_sigma)?;
            face_diffusivity_affine(&gpu, &kern, &mut b_i, &mut b_b, &nut, &m, nu, 1.0, r_sigma)?;
            gpu.sync()?;

            // The bit-identity asserted below is a claim about nvcc's FMA
            // CONTRACTION as much as about arithmetic, so the claim is first
            // shown to be non-vacuous: the two roundings of
            // `nu + r_sigma nu_t` - fused and unfused - must actually DIFFER
            // somewhere in this data, or a kernel that contracted differently
            // from the plain one would pass anyway.
            //
            // The boundary faces are the clean probe: `Gamma * |Sf|` with no
            // interpolation weight, so the rounding is visible undiluted.
            {
                let mut worst_ulp = 0i64;
                for (b, ntb) in bfaces.iter().enumerate() {
                    let sf = hm.b_mag_sf[b];
                    let unfused = (nu + r_sigma * ntb) * sf;
                    let fused = r_sigma.mul_add(*ntb, nu) * sf;
                    let d = (unfused.to_bits() as i64 - fused.to_bits() as i64).abs();
                    worst_ulp = worst_ulp.max(d);
                }
                if r_sigma == 1.0 {
                    assert_eq!(
                        worst_ulp, 0,
                        "at r_sigma = 1 the product is exact, so the two roundings \
                         must agree - if they do not, the probe is wrong"
                    );
                } else {
                    assert!(
                        worst_ulp > 0,
                        "r_sigma {r_sigma}: the fused and unfused roundings agree on \
                         every face, so the bit-identity below would pass on a kernel \
                         that contracted the wrong way. Pick harder numbers."
                    );
                }
                println!(
                    "r_sigma {r_sigma}: the fused and unfused roundings of \
                     nu + r_sigma nu_t differ by up to {worst_ulp} ULP on this data; \
                     the two KERNELS must still agree exactly"
                );
            }

            let (ha_i, hb_i) = (gpu.download(&a_i)?, gpu.download(&b_i)?);
            let (ha_b, hb_b) = (gpu.download(&a_b)?, gpu.download(&b_b)?);
            for (f, (x, y)) in ha_i.iter().zip(&hb_i).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "r_sigma {r_sigma}, internal face {f}: {x} against {y}"
                );
            }
            for (f, (x, y)) in ha_b.iter().zip(&hb_b).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "r_sigma {r_sigma}, boundary face {f}: {x} against {y}"
                );
            }
        }

        // alpha(nu + nu_t) is NOT nu + alpha nu_t, and the difference is
        // exactly (alpha - 1) nu on every face - which is what makes this a
        // measurement of the molecular part rather than of nothing.
        let alpha: Scalar = 1.39;
        face_diffusivity(&gpu, &kern, &mut a_i, &mut a_b, &nut, &m, nu, alpha)?;
        face_diffusivity_affine(&gpu, &kern, &mut b_i, &mut b_b, &nut, &m, nu, alpha, alpha)?;
        gpu.sync()?;
        let (ha_b, hb_b) = (gpu.download(&a_b)?, gpu.download(&b_b)?);
        let want = (alpha - 1.0) * nu;
        for (f, (x, y)) in ha_b.iter().zip(&hb_b).enumerate() {
            // The boundary face is the clean one: Gamma·|Sf| with no
            // interpolation, so the difference divides out exactly.
            let sf = hm.b_mag_sf[f];
            if sf <= 0.0 {
                continue;
            }
            let d = (y - x) / sf;
            assert!(
                (d - want).abs() <= 1e-12 * want,
                "boundary face {f}: affine - plain = {d}, expected (alpha-1)nu = {want}"
            );
        }

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  The production term - SPEC-LIT §6
    // ----------------------------------------------------------------------

    /// SPEC-LIT §6 gives `G/nu_t` two ways and says the second is "the form to
    /// implement, because it avoids building the deviatoric tensor". They are
    /// only interchangeable if they are equal, including for a field with a
    /// non-zero divergence, where the `(2/3)tr²` term is what does the work.
    #[test]
    fn the_reduced_production_equals_the_long_form() {
        let cases = [
            // Solenoidal simple shear.
            Tensor { xx: 0.0, xy: 0.0, xz: 0.0,
                     yx: 3.0, yy: 0.0, yz: 0.0,
                     zx: 0.0, zy: 0.0, zz: 0.0 },
            // Solenoidal, fully three-dimensional.
            Tensor { xx: 1.0, xy: -2.0, xz: 0.5,
                     yx: 0.7, yy: 0.5, yz: -1.3,
                     zx: 2.1, zy: 0.2, zz: -1.5 },
            // Compressible: tr(grad U) = 2.4, so dev() is not the identity.
            Tensor { xx: 1.1, xy: -0.4, xz: 0.9,
                     yx: 0.3, yy: 0.8, yz: 1.7,
                     zx: -2.2, zy: 0.6, zz: 0.5 },
            Tensor::ZERO,
        ];

        for (i, g) in cases.iter().enumerate() {
            // The definition, spelled out: dev(twoSymm(grad U)) : grad U.
            let long = g.two_symm().dev().ddot(*g);
            let reduced = g.g_by_nut();

            let scale = long.abs().max(1.0);
            assert!(
                (reduced - long).abs() <= 1e-13 * scale,
                "case {i}: reduced {reduced}, long {long}"
            );
        }
    }

    /// A uniform strain field whose production is known in closed form.
    ///
    /// For `U = (A y, 0, 0)` the only non-zero gradient component is
    /// `dU_x/dy`, which by SPEC-LIT §1's convention is `(grad U)_{yx} = A`.
    /// Then `symm(grad U)` has `S_xy = S_yx = A/2`, the trace is zero, and
    ///
    /// ```text
    /// G/nu_t = 2 S:S = 2·(2·(A/2)²) = A²
    /// ```
    ///
    /// For the axisymmetric extension `U = (A x, A y, -2 A z)` the trace is
    /// again zero, `S = diag(A, A, -2A)`, and `G/nu_t = 2 S:S = 12 A²`.
    #[test]
    fn a_uniform_strain_gives_the_analytic_production() {
        let a: Scalar = 2.5;

        let shear = Tensor { xx: 0.0, xy: 0.0, xz: 0.0,
                             yx: a,   yy: 0.0, yz: 0.0,
                             zx: 0.0, zy: 0.0, zz: 0.0 };
        assert!((shear.g_by_nut() - a * a).abs() < 1e-13);

        let extension = Tensor { xx: a,   xy: 0.0, xz: 0.0,
                                 yx: 0.0, yy: a,   yz: 0.0,
                                 zx: 0.0, zy: 0.0, zz: -2.0 * a };
        assert!((extension.g_by_nut() - 12.0 * a * a).abs() < 1e-12);

        // Pure rotation does no work on the mean flow and must produce
        // nothing: grad U antisymmetric => twoSymm(grad U) = 0.
        let rotation = Tensor { xx: 0.0, xy: a,   xz: 0.0,
                                yx: -a,  yy: 0.0, yz: 0.0,
                                zx: 0.0, zy: 0.0, zz: 0.0 };
        assert!(rotation.g_by_nut().abs() < 1e-13);
    }

    /// The same three tensors through `fvProduction` on the device. The host
    /// expression is the specification; if the kernel disagrees, every source
    /// term built from `G` is wrong by the same amount.
    #[test]
    fn the_device_production_matches_the_host() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let a: Scalar = 2.5;
        let cases = vec![
            Tensor { xx: 0.0, xy: 0.0, xz: 0.0,
                     yx: a,   yy: 0.0, yz: 0.0,
                     zx: 0.0, zy: 0.0, zz: 0.0 },
            Tensor { xx: a,   xy: 0.0, xz: 0.0,
                     yx: 0.0, yy: a,   yz: 0.0,
                     zx: 0.0, zy: 0.0, zz: -2.0 * a },
            Tensor { xx: 1.1, xy: -0.4, xz: 0.9,
                     yx: 0.3, yy: 0.8,  yz: 1.7,
                     zx: -2.2, zy: 0.6, zz: 0.5 },
        ];
        let n = cases.len();

        let nut_host: Vec<Scalar> = (0..n).map(|i| 0.5 + i as Scalar).collect();

        let fv = FvKernels::new(&gpu)?;
        let grad = gpu.upload(&cases)?;
        let nut = gpu.upload(&nut_host)?;
        let mut out = gpu.zeros::<Scalar>(n)?;

        crate::fv::turbulence_production(&gpu, &fv, &mut out, &nut, &grad, n)?;
        gpu.sync()?;

        let got = gpu.download(&out)?;

        for i in 0..n {
            let want = nut_host[i] * cases[i].g_by_nut();
            assert!(
                (got[i] - want).abs() <= 1e-12 * want.abs().max(1.0),
                "cell {i}: device {} host {want}",
                got[i]
            );
        }

        // And the analytic values, so this is not just self-consistency.
        assert!((got[0] - nut_host[0] * a * a).abs() < 1e-12);
        assert!((got[1] - nut_host[1] * 12.0 * a * a).abs() < 1e-11);

        Ok(())
    }

    /// `turbStrainRateMag` must reproduce [`Tensor::strain_rate_mag_sqr`]'s
    /// square root - the two are used interchangeably by anything that limits
    /// `nu_t` on the strain rate.
    #[test]
    fn the_device_strain_rate_matches_the_host() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let cases = vec![
            Tensor { xx: 0.0, xy: 0.0, xz: 0.0,
                     yx: 3.0, yy: 0.0, yz: 0.0,
                     zx: 0.0, zy: 0.0, zz: 0.0 },
            Tensor { xx: 1.1, xy: -0.4, xz: 0.9,
                     yx: 0.3, yy: 0.8,  yz: 1.7,
                     zx: -2.2, zy: 0.6, zz: 0.5 },
        ];
        let n = cases.len();

        let kern = TurbKernels::new(&gpu)?;
        let grad = gpu.upload(&cases)?;
        let mut out = gpu.zeros::<Scalar>(n)?;

        strain_rate_mag(&gpu, &kern, &mut out, &grad, n)?;
        gpu.sync()?;

        let got = gpu.download(&out)?;
        for i in 0..n {
            let want = cases[i].strain_rate_mag_sqr().sqrt();
            assert!(
                (got[i] - want).abs() <= 1e-12 * want.max(1.0),
                "cell {i}: device {} host {want}",
                got[i]
            );
        }

        // Simple shear of magnitude A: S_xy = A/2, so sqrt(2 S:S) = A.
        assert!((got[0] - 3.0).abs() < 1e-12);

        Ok(())
    }
}
