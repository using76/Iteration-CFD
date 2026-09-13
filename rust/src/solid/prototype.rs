// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The segregated thermo-elastic outer loop, host-side, measured before a
//! kernel is written - the risk-1 experiment of
//! `docs/09-thermal-structural-plan.md` §F.
//!
//! Written from:
//!   I. Demirdžić, S. Muzaferija, *Int. J. Numer. Methods Eng.* 37 (1994)
//!     3751-3766, DOI 10.1002/nme.1620372110 - finite-volume stress analysis
//!     in complex domains: the segregated cell-centred formulation, and the
//!     statement that a boundary face's contribution to the equilibrium sum
//!     IS its prescribed traction
//!   I. Demirdžić, S. Muzaferija, *Comput. Methods Appl. Mech. Eng.* 125
//!     (1995) 235-255, DOI 10.1016/0045-7825(95)00800-G - the coupled
//!     fluid/heat/stress arrangement this plan's Stage 1 is one region of
//!   H. Jasak, H. G. Weller, *Int. J. Numer. Methods Eng.* 48 (2000) 267-287,
//!     DOI 10.1002/(SICI)1097-0207(20000520)48:2<267::AID-NME884>3.0.CO;2-Q -
//!     the `(2 mu + lambda)` implicit split and its convergence behaviour.
//!     (The Crossref date part is 20000520, not the 20000530 that circulates.)
//!   I. Demirdžić, D. Martinović, *Comput. Methods Appl. Mech. Eng.* 109
//!     (1993) 331-349, DOI 10.1016/0045-7825(93)90085-C - the thermal-strain
//!     term in finite-volume form
//!   B. A. Boley, J. H. Weiner, *Theory of Thermal Stresses*, Wiley (1960),
//!     ch. 1 - Duhamel-Neumann, and the free-expansion state this module
//!     gates on
//!   S. P. Timoshenko, J. N. Goodier, *Theory of Elasticity*, 3rd ed.,
//!     McGraw-Hill (1970) ch. 1 - small-strain isotropic elasticity and the
//!     Lamé conversion
//!   U. Küttler, W. A. Wall, *Comput. Mech.* 43 (2008) 61-72, DOI
//!     10.1007/s00466-008-0255-5 - Aitken delta-squared dynamic relaxation of
//!     a partitioned fixed point, §3.2, in the vector form used here
//!   P. Cardiff, I. Demirdžić, *Arch. Comput. Methods Eng.* 28 (2021)
//!     3721-3780, DOI 10.1007/s11831-020-09523-0 - the thirty-year review;
//!     the degradation of a segregated solution as `nu -> 0.5` is a published
//!     observation, and this module is what turns it into a number for THIS
//!     discretisation
//!   P. Cardiff, Ž. Tuković, H. Jasak, A. Ivanković, *Comput. Struct.* 175
//!     (2016) 100-122, DOI 10.1016/j.compstruc.2016.07.004 - block-coupled
//!     finite-volume elasticity. Named as the alternative the measurement
//!     below either does or does not force; NOT implemented, and refused in
//!     the plan's §D.5 because a 3x3 coefficient per face breaks SPEC-LIT
//!     §1's one-entry-per-face storage
//!   H. Jasak, PhD thesis, Imperial College London (1996) §3.4.2 - the
//!     over-relaxed non-orthogonal split, which is SPEC-LIT §2.4
//!   M. R. Hestenes, E. Stiefel, *J. Res. Natl. Bur. Stand.* 49 (1952) 409,
//!     DOI 10.6028/jres.049.044 (a **US Government work, public domain**) -
//!     the conjugate-gradient recurrence of SPEC-LIT §8.2
//!   Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM
//!     (2003), DOI 10.1137/1.9780898718003, §10.2-10.3 - the incomplete
//!     factorisation whose diagonal-only form is what SPEC-LIT §21
//!     multi-colours on the device and what [`dic_factor`] builds here
//!   ofgpu `SPEC-LIT.md` §95 - the section the measurements below are
//!     §95.4 of - and §1, §2.3, §2.4, §3.2, §3.5, §4, §8.2, §8.4, §10, §21,
//!     §46.4
//!
//! OpenFOAM and solids4foam are GPL and were not opened; neither was any
//! other solid-mechanics solver of any licence. The discretisation below is
//! assembled out of this repository's own operators from the four papers
//! named above. **No GPL-licensed source was consulted.**
//!
//! # Why this exists at all
//!
//! `docs/09-thermal-structural-plan.md` §D.3 derives - it does not quote - a
//! contraction factor for the segregated displacement loop:
//!
//! ```text
//! |deferred| / |implicit| = (mu + lambda) / (2 mu + lambda) = 1/(2(1 - nu))
//! ```
//!
//! 0.625 at `nu = 0.2`, 0.714 at 0.3, 0.909 at 0.45, 0.980 at 0.49. The whole
//! of Stage 1 - twelve weeks, a CUDA unit, a gate suite - is sized on that
//! number, and §D.5's refusal threshold is drawn from it. SPEC-LIT §46.4
//! records what this project got for asserting such an estimate instead: the
//! first draft of that section divided by the wrong conductivity and was out
//! by a factor of 180, and the test that measured it is what found out.
//!
//! So the loop is written here, host-side, on a mesh `blockgen` already
//! makes, and the contraction is **measured** - before a kernel exists and
//! before the shape of Stage 1 is fixed.
//!
//! Nothing in this module runs in a solver. It is deliberately the slow,
//! obvious, allocating version, in the scatter-shaped style of
//! [`crate::reference`] and reusing that module's operators, so that what is
//! measured is the *algorithm* and not a new transcription of the mesh.
//!
//! # What it discretises
//!
//! Quasi-static equilibrium of an isotropic linear thermo-elastic solid, in
//! the small-strain Duhamel-Neumann form (Boley & Weiner ch. 1):
//!
//! ```text
//! div(sigma) = 0,   eps = 1/2 (grad u + grad u^T)
//! sigma = 2 mu eps + lambda tr(eps) I - (3 lambda + 2 mu) alpha (T - T_ref) I
//! ```
//!
//! Integrated over a cell and split the way Jasak & Weller (2000) split it,
//! with the thermal term of Demirdžić & Martinović (1993):
//!
//! ```text
//! sum_f (2 mu + lambda) grad(u).Sf                               implicit
//!   + sum_f [ mu grad(u)^T + lambda tr(grad u) I
//!             - (mu + lambda) grad(u) ].Sf                       deferred
//!   - sum_f (3 lambda + 2 mu) alpha (T_f - T_ref) Sf             thermal
//!   = 0
//! ```
//!
//! The implicit term is SPEC-LIT §3.2's Gauss laplacian per displacement
//! component with `gammaMagSf = (2 mu + lambda)|Sf|`, carrying §2.4's
//! over-relaxed non-orthogonal correction; the deferred and thermal terms are
//! surface integrals of quantities SPEC-LIT §3.5's Green-Gauss gradient
//! already produces. Three components, three scalar systems on §1's LDU
//! storage, coupled only through the right-hand side: that is what makes the
//! outer loop a Picard iteration and what makes its contraction a number
//! worth measuring.
//!
//! # Where the boundary goes, and why that matters to the measurement
//!
//! Demirdžić & Muzaferija (1994) make the boundary statement exactly: the
//! contribution of a boundary face to the equilibrium sum **is** the traction
//! prescribed there. So a traction face adds `t |Sf|` and nothing else - no
//! diagonal, no deferred term of its own. What the traction condition then
//! has to supply is the boundary *displacement*, because §3.5's gradient
//! needs a face value, and that value is obtained by inverting the traction:
//!
//! ```text
//! t = (2 mu + lambda) snGrad(u) + Q_b.n,   Q_b the deferred group above
//! refGrad = (t - Q_b.n) / (2 mu + lambda)
//! u_b     = u_P + refGrad / Delta_b                       (SPEC-LIT §4, fr = 0)
//! ```
//!
//! `Q_b` is evaluated from the previous outer iteration's gradient, so the
//! traction boundary is itself part of the fixed point. The plan's risk 1
//! says in as many words that "non-orthogonality and the deferred traction BC
//! both feed it"; this arrangement is what puts the second of those two into
//! the measured number, and [`jittered_block`] is what puts the first in.

use crate::blockgen::{self, BlockSpec, GradedAxis};
use crate::io::polymesh::build_host_mesh;
use crate::mesh::HostMesh;
use crate::reference::{
    add_boundary_contributions, amul, fvc_grad_vector, fvm_laplacian,
    fvm_laplacian_non_orth_correction, is_empty_face, CpuLdu, CpuScalarBc,
};
use crate::{Result, Scalar, Tensor, Vec3};
use crate::fv::SnGradScheme;

// ==========================================================================
//  The material and the boundary statement - moved to the solver, re-exported
// ==========================================================================

// Both moved to the solver in the unit that put the operator on the device;
// re-exported so the tests below read as written.
pub use super::Material;
pub use super::bc::{all_free, fixed_minus_x, free_expansion, CompBc, PatchBcs};

// ==========================================================================
//  Tensor contractions, in SPEC-LIT §1's index convention
// ==========================================================================

/// `(a^T G)_j = sum_i a_i G_ij`.
///
/// With SPEC-LIT §1's convention `G_ij = du_j/dx_i`, this is `grad(u).Sf`
/// when `a = Sf`: the area vector supplies the DERIVATIVE index, which is
/// what makes the implicit term a laplacian per component rather than a
/// gradient of a divergence.
#[inline]
fn dot_left(a: Vec3, g: Tensor) -> Vec3 {
    Vec3::new(
        a.x * g.xx + a.y * g.yx + a.z * g.zx,
        a.x * g.xy + a.y * g.yy + a.z * g.zy,
        a.x * g.xz + a.y * g.yz + a.z * g.zz,
    )
}

/// `(G a)_j = sum_i G_ji a_i`, which is `grad(u)^T.Sf` at `a = Sf`.
#[inline]
fn dot_right(g: Tensor, a: Vec3) -> Vec3 {
    Vec3::new(
        g.xx * a.x + g.xy * a.y + g.xz * a.z,
        g.yx * a.x + g.yy * a.y + g.yz * a.z,
        g.zx * a.x + g.zy * a.y + g.zz * a.z,
    )
}

/// Column `j` of `G`, which is `grad(u_j)` - the vector SPEC-LIT §2.4's
/// correction contracts against for component `j`.
#[inline]
fn grad_component(g: Tensor, j: usize) -> Vec3 {
    match j {
        0 => Vec3::new(g.xx, g.yx, g.zx),
        1 => Vec3::new(g.xy, g.yy, g.zy),
        _ => Vec3::new(g.xz, g.yz, g.zz),
    }
}

/// The deferred group, contracted with an area vector:
///
/// ```text
/// [ mu grad(u)^T + lambda tr(grad u) I - (mu + lambda) grad(u)
///   - (3 lambda + 2 mu) alpha dT I ] . Sf
/// ```
///
/// Everything the implicit `(2 mu + lambda) grad(u).Sf` is not. Adding the
/// two gives `sigma.Sf` exactly, which is what
/// `the_split_reassembles_the_traction` checks rather than assumes.
#[inline]
fn deferred_traction(m: Material, g: Tensor, d_temp: Scalar, sf: Vec3) -> Vec3 {
    let (mu, lam) = (m.mu(), m.lambda());
    dot_right(g, sf) * mu + sf * (lam * g.trace())
        - dot_left(sf, g) * (mu + lam)
        - sf * (m.three_lambda_two_mu() * m.alpha * d_temp)
}

/// The full surface traction `sigma . Sf` from a cell or face gradient - the
/// sum of the implicit and deferred halves, written once so that the two
/// cannot drift apart.
#[inline]
pub fn traction(m: Material, g: Tensor, d_temp: Scalar, sf: Vec3) -> Vec3 {
    dot_left(sf, g) * m.implicit_gamma() + deferred_traction(m, g, d_temp, sf)
}

/// Cauchy stress from a cell gradient (Duhamel-Neumann; Boley & Weiner ch. 1).
#[inline]
pub fn stress(m: Material, g: Tensor, d_temp: Scalar) -> Tensor {
    let (mu, lam) = (m.mu(), m.lambda());
    let mut s = g.two_symm() * mu;
    let d = lam * g.trace() - m.three_lambda_two_mu() * m.alpha * d_temp;
    s.xx += d;
    s.yy += d;
    s.zz += d;
    s
}

// ==========================================================================
//  Meshes
// ==========================================================================

/// A `n x n x n` unit block, every one of the six slots a real patch.
///
/// `blockgen`'s default makes `-z`/`+z` *empty* - the 2-D front and back - and
/// an empty face contributes to no surface integral at all, which for a
/// three-dimensional stress problem would silently delete two of the six
/// tractions. So all six are named here.
pub fn block(n: usize) -> Result<HostMesh> {
    blockgen::build_mesh(&block_spec(n))
}

fn block_spec(n: usize) -> BlockSpec {
    let axis = GradedAxis { lo: 0.0, hi: 1.0, n, expansion: 1.0, two_sided: false };
    BlockSpec {
        x: axis.clone(),
        y: axis.clone(),
        z: axis,
        patch_type: ["patch", "patch", "patch", "patch", "patch", "patch"].map(String::from),
        ..Default::default()
    }
}

/// The same block with its INTERIOR points displaced, so that the faces are
/// neither orthogonal nor unskewed.
///
/// The plan's risk-1 row names non-orthogonality as one of the two things
/// that could make the measured contraction worse than the derived one, and a
/// graded block is not a test of it: grading moves the nodes along the axes
/// and leaves every face normal parallel to `d`. Displacing the interior
/// points by up to `amplitude` times the local spacing does test it - it
/// leaves the boundary planes flat, so the boundary conditions are unchanged,
/// and turns every interior face into one SPEC-LIT §2.4's correction and
/// §2.5's skewness vector have something to say about.
///
/// The displacement is a fixed trigonometric field of the point coordinates,
/// not a random one: an experiment whose mesh is different on every run
/// cannot be re-read.
pub fn jittered_block(n: usize, amplitude: Scalar) -> Result<HostMesh> {
    let mut raw = blockgen::raw_mesh(&block_spec(n))?;
    let h = 1.0 / (n as Scalar);
    let interior = |v: Scalar| v > 1e-12 && v < 1.0 - 1e-12;
    for p in raw.points.iter_mut() {
        if !(interior(p.x) && interior(p.y) && interior(p.z)) {
            continue;
        }
        let (a, b, c) = (7.0 * p.x, 5.0 * p.y, 3.0 * p.z);
        let d = Vec3::new(
            (a + 2.0 * b).sin() * c.cos(),
            (b + 2.0 * c).sin() * a.cos(),
            (c + 2.0 * a).sin() * b.cos(),
        );
        *p += d * (amplitude * h);
    }
    build_host_mesh(&raw)
}

// ==========================================================================
//  The prototype itself
// ==========================================================================

/// What one run of the outer loop cost, and what it contracted at.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OuterReport {
    /// Outer iterations taken to six decades of the fixed-point residual, or
    /// the cap if it never got there.
    pub iterations: usize,
    /// `true` if six decades were reached inside the cap.
    pub converged: bool,
    /// `true` if the residual grew past a million times its first value, or
    /// stopped being a number. The loop stops there rather than at the cap:
    /// a fixed point that has amplified by 1e6 is not going to come back.
    pub diverged: bool,
    /// `||r_k||` for every `k` - the whole history, because a rate read off
    /// a single pair of iterations is not a rate.
    pub norms: Vec<Scalar>,
    /// `||r_k|| / ||r_{k-1}||` for every `k >= 2`.
    pub ratios: Vec<Scalar>,
    /// The geometric mean of the last ten ratios - the observed contraction.
    pub observed_contraction: Scalar,
    /// `1/(2(1 - nu))`, computed from the Lame constants.
    pub predicted_contraction: Scalar,
    /// Total conjugate-gradient iterations over all three components.
    pub linear_iterations: usize,
    /// The Aitken factors that were used, if any.
    pub omegas: Vec<Scalar>,
}

impl OuterReport {
    /// The geometric mean of the last `n` ratios. Reported rather than the
    /// mean of all of them because the first few carry the transient of a
    /// zero initial guess, and it is the ASYMPTOTIC rate the plan's estimate
    /// is a statement about.
    fn geometric_mean_of_last(ratios: &[Scalar], n: usize) -> Scalar {
        let tail = &ratios[ratios.len().saturating_sub(n)..];
        if tail.is_empty() {
            return Scalar::NAN;
        }
        let s: Scalar = tail.iter().map(|r| r.ln()).sum();
        (s / (tail.len() as Scalar)).exp()
    }
}

/// The segregated loop, its mesh, its material and its state.
pub struct Prototype {
    pub mesh: HostMesh,
    pub material: Material,
    /// The uniform `T - T_ref` the whole block is at.
    pub d_temp: Scalar,
    /// SPEC-LIT §12.3's selector for the §2.4 correction. `Corrected` is what
    /// the plan's §D.3 names; `Uncorrected` is here so that what the
    /// correction costs the outer loop can be measured rather than argued.
    pub sn_grad: SnGradScheme,
    /// Tolerance of the inner conjugate-gradient solve, on SPEC-LIT §8.4's
    /// normalised residual.
    pub linear_tolerance: Scalar,
    /// Cap on conjugate-gradient iterations per component per outer pass.
    pub linear_cap: usize,
    /// How many times the traction boundary condition and the gradient are
    /// brought into agreement before each assembly. One is the cheapest
    /// arrangement and lags the boundary by a whole outer iteration; more
    /// closes that sub-loop, which is a local contraction of its own and costs
    /// a Green-Gauss gradient rather than a linear solve. Three is the default
    /// because the sweep measured it at roughly half the outer iterations of
    /// one, and because at `nu = 0.49` one does not converge at all.
    pub boundary_passes: usize,
    /// Which of the two traction conditions of [`Prototype::update_ref_grad`]
    /// to impose. `true` - the normal derivative solved AT the face - is the
    /// one this experiment ended up recommending; `false` is the naive
    /// extrapolation, kept because the difference between them is one of the
    /// numbers the experiment reports.
    pub boundary_gradient_correction: bool,
    /// `kappa`: the factor the implicit coefficient of the split is
    /// multiplied by, with the same amount moved into the deferred term so
    /// that implicit plus deferred is still `sigma.Sf`.
    ///
    /// One is the split Jasak & Weller (2000) write and the only value any
    /// solver runs. It is a field because the *continuum* split is exact for
    /// every `kappa` - the fixed point of the map does not depend on it, only
    /// the iteration matrix does - so `kappa` is a candidate accelerator whose
    /// worth is a thing to measure, not to argue. Read
    /// [`Prototype::new_with_implicit_scale`] for what it does and does not
    /// leave alone.
    pub implicit_scale: Scalar,

    /// Displacement, per cell.
    pub u: Vec<Vec3>,
    /// Displacement on the boundary faces - SPEC-LIT §4's `psi_b`.
    pub ub: Vec<Vec3>,
    /// `grad(u)` per cell, `G_ij = du_j/dx_i` (SPEC-LIT §1, §3.5).
    pub grad: Vec<Tensor>,

    /// One system per component. Assembled ONCE: `mu` and `lambda` are
    /// constant and the mesh does not move, so only the source changes from
    /// one outer iteration to the next. The plan's §D.3 calls that the single
    /// largest lever, and it is free here.
    a: [CpuLdu; 3],
    /// The incomplete-Cholesky diagonals of those three, also built once.
    rd: [Vec<Scalar>; 3],
    /// SPEC-LIT §4's triple per component. `fr` and `ref_value` are fixed by
    /// the configuration; `ref_grad` is rewritten every outer iteration from
    /// the traction condition, which is what makes the boundary part of the
    /// fixed point.
    bc: [CpuScalarBc; 3],
    /// `(2 mu + lambda)|Sf|`, internal and boundary.
    gamma_mag_sf: Vec<Scalar>,
    b_gamma_mag_sf: Vec<Scalar>,
    /// Which components are prescribed by value on which boundary face.
    b_fixed: Vec<[bool; 3]>,
    /// The prescribed traction on each boundary face.
    b_traction: Vec<Vec3>,
    /// `grad(u)` AT each boundary face - the cell gradient with its
    /// normal-derivative row replaced by the face's own `snGrad`, which is the
    /// one part of it the boundary condition knows exactly.
    b_grad: Vec<Tensor>,

    /// Scratch, so the outer loop allocates nothing.
    comp: Vec<Scalar>,
    comp_grad: Vec<Vec3>,
    rhs: Vec<Vec3>,
}

impl Prototype {
    /// Build the three systems and everything about them that does not change.
    pub fn new(mesh: HostMesh, material: Material, d_temp: Scalar, bcs: &PatchBcs) -> Self {
        Self::new_with_implicit_scale(mesh, material, d_temp, bcs, 1.0)
    }

    /// [`Prototype::new`] with the implicit coefficient of the split
    /// multiplied by `implicit_scale` and the same amount moved into the
    /// deferred surface integral, so that implicit plus deferred is still
    /// `sigma.Sf` and the continuous fixed point is unchanged.
    ///
    /// ```text
    /// implicit:  kappa (2 mu + lambda) grad(u).Sf
    /// deferred:  [ mu grad(u)^T + lambda tr(grad u) I - (mu + lambda) grad(u)
    ///              - (3 lambda + 2 mu) alpha dT I ].Sf
    ///            - (kappa - 1)(2 mu + lambda) grad(u).Sf
    /// ```
    ///
    /// What that costs in honesty, said here rather than discovered later:
    /// the two halves are evaluated by DIFFERENT discrete operators - the
    /// implicit one by SPEC-LIT §3.2's compact laplacian with §2.4's
    /// correction, the deferred one by §3.5's Green-Gauss gradient
    /// interpolated to the face - so on an internal face the cancellation is
    /// exact in the continuum and only to discretisation order on the mesh.
    /// A `kappa` other than one therefore moves the converged displacement by
    /// that difference, and the sweep that uses this reports the movement
    /// instead of assuming it away. At a boundary face the cancellation IS
    /// exact: a fixed component's deferred term is evaluated from
    /// [`Prototype::update_boundary_gradient`]'s face gradient, whose normal
    /// row is the same `Delta_b (u_b - u_P)` the implicit term uses, and a
    /// traction component contributes `t |Sf|` and nothing else, with
    /// [`Prototype::update_ref_grad`]'s solved form producing the face's true
    /// normal derivative from `t` whatever `kappa` is.
    pub fn new_with_implicit_scale(
        mesh: HostMesh,
        material: Material,
        d_temp: Scalar,
        bcs: &PatchBcs,
        implicit_scale: Scalar,
    ) -> Self {
        let n_c = mesh.n_cells;
        let n_bf = mesh.n_boundary_faces;
        let gamma = material.implicit_gamma() * implicit_scale;

        let gamma_mag_sf: Vec<Scalar> =
            (0..mesh.n_internal_faces).map(|f| gamma * mesh.mag_sf[f]).collect();
        let b_gamma_mag_sf: Vec<Scalar> =
            (0..n_bf).map(|bf| gamma * mesh.b_mag_sf[bf]).collect();

        let mut b_fixed = vec![[false; 3]; n_bf];
        let mut b_traction = vec![Vec3::ZERO; n_bf];
        let mut bc =
            [CpuScalarBc::new(n_bf), CpuScalarBc::new(n_bf), CpuScalarBc::new(n_bf)];
        for bf in 0..n_bf {
            let patch = mesh.b_patch[bf] as usize;
            for i in 0..3 {
                match bcs[patch][i] {
                    CompBc::Fixed(v) => {
                        b_fixed[bf][i] = true;
                        bc[i].fr[bf] = 1.0;
                        bc[i].ref_value[bf] = v;
                    }
                    CompBc::Traction(t) => {
                        bc[i].fr[bf] = 0.0;
                        match i {
                            0 => b_traction[bf].x = t,
                            1 => b_traction[bf].y = t,
                            _ => b_traction[bf].z = t,
                        }
                    }
                }
            }
        }

        // The operator, once. `sign = -1` so that `A = -laplacian` is
        // positive definite the moment one displacement is fixed, which is
        // what SPEC-LIT §8.2's conjugate gradients require.
        let mut a = [CpuLdu::new(&mesh), CpuLdu::new(&mesh), CpuLdu::new(&mesh)];
        let mut rd: [Vec<Scalar>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for i in 0..3 {
            fvm_laplacian(&mut a[i], &mesh, &gamma_mag_sf, &b_gamma_mag_sf, &bc[i], -1.0);
            // Folds `internal_coeffs` into the diagonal. `boundary_coeffs` is
            // identically zero here, because `ref_value` and `ref_grad` are
            // still zero; the source is rebuilt from nothing every outer pass
            // by `assemble_source`.
            add_boundary_contributions(&mut a[i], &mesh);
            rd[i] = dic_factor(&a[i], &mesh);
        }

        Self {
            u: vec![Vec3::ZERO; n_c],
            ub: vec![Vec3::ZERO; n_bf],
            grad: vec![Tensor::ZERO; n_c],
            material,
            d_temp,
            sn_grad: SnGradScheme::Corrected,
            linear_tolerance: 1.0e-14,
            linear_cap: 5000,
            boundary_passes: 3,
            boundary_gradient_correction: true,
            implicit_scale,
            a,
            rd,
            bc,
            gamma_mag_sf,
            b_gamma_mag_sf,
            b_fixed,
            b_traction,
            b_grad: vec![Tensor::ZERO; n_bf],
            comp: vec![0.0; n_c],
            comp_grad: vec![Vec3::ZERO; n_c],
            rhs: vec![Vec3::ZERO; n_c],
            mesh,
        }
    }

    /// The boundary values and the cell gradient, brought into agreement with
    /// the current `u`.
    ///
    /// There is a circularity to close: SPEC-LIT §3.5's Green-Gauss gradient
    /// needs a boundary value, and the traction condition produces a boundary
    /// value only from a gradient. [`Prototype::boundary_passes`] says how
    /// many times to go round it before the source is assembled - and how
    /// many is the right number is not obvious in advance, which is why it is
    /// a field and not a constant, and why the sweep reports it.
    pub fn correct_boundary(&mut self) {
        for _ in 0..self.boundary_passes.max(1) {
            fvc_grad_vector(&mut self.grad, &self.u, &self.ub, &self.mesh);
            self.update_ref_grad();
            self.evaluate_boundary();
        }
        fvc_grad_vector(&mut self.grad, &self.u, &self.ub, &self.mesh);
        self.update_boundary_gradient();
    }

    /// `grad(u)` at a boundary face.
    ///
    /// A cell-centre gradient extrapolated to the face is wrong in exactly the
    /// direction the boundary condition is right about: its normal-derivative
    /// row is a one-sided guess, while `Delta_b (u_b - u_P)` is the number the
    /// condition itself produced. Replacing that row - and only that row,
    /// leaving the tangential derivatives to the cell gradient that can see
    /// them - is what `boundary_gradient_correction` does.
    fn update_boundary_gradient(&mut self) {
        for bf in 0..self.mesh.n_boundary_faces {
            if is_empty_face(&self.mesh, bf) {
                continue;
            }
            let c = self.mesh.b_face_cells[bf] as usize;
            let g = self.grad[c];
            self.b_grad[bf] = if self.boundary_gradient_correction {
                let n = self.mesh.b_sf[bf].normalised();
                let sn = (self.ub[bf] - self.u[c]) * self.mesh.b_delta_coeffs[bf];
                g + n.outer(sn - dot_left(n, g))
            } else {
                g
            };
        }
    }

    /// The traction condition, inverted for the normal gradient SPEC-LIT §4's
    /// triple carries. Two forms, and which one is used is the difference
    /// between a loop that contracts and one that does not.
    ///
    /// **The naive form** (`boundary_gradient_correction == false`) treats the
    /// face exactly like an interior one, splitting off `(2 mu + lambda)` and
    /// lagging everything else:
    ///
    /// ```text
    /// t |Sf| = (2 mu + lambda) |Sf| refGrad + Q_b.Sf
    /// refGrad = (t |Sf| - Q_b.Sf) / ((2 mu + lambda)|Sf|)
    /// ```
    ///
    /// with `Q_b` from the extrapolated cell gradient. Every component of
    /// `refGrad` is then a lagged quantity, and the measurement below says
    /// what that costs.
    ///
    /// **The solved form** (the default) uses the fact that the normal
    /// derivative is the ONLY thing the face does not already know. Write
    /// `grad(u)|_b = G_t + n (x) refGrad`, where `G_t` is the cell gradient
    /// with its normal-derivative row struck out, and substitute into
    /// `sigma.n = t`:
    ///
    /// ```text
    /// (2 mu + lambda) (refGrad.n) = t.n - mu (G_t.n).n - lambda tr(G_t)
    ///                               + (3 lambda + 2 mu) alpha dT
    /// mu refGrad_t              = t_t - mu (G_t.n)_t
    /// ```
    ///
    /// - which is the continuum statement that a traction boundary sees
    /// `2 mu + lambda` normally and `mu` tangentially (Timoshenko & Goodier
    /// ch. 1; the finite-volume statement is Demirdžić & Muzaferija 1994).
    /// `refGrad` appears on the left of its own condition rather than inside a
    /// lagged `Q_b`, so the boundary contributes no deferred term of its own
    /// at all, and the free-expansion state reproduces `refGrad = alpha dT n`
    /// exactly.
    fn update_ref_grad(&mut self) {
        let (mu, lam) = (self.material.mu(), self.material.lambda());
        let gamma = self.material.implicit_gamma();
        let thermal = self.material.three_lambda_two_mu() * self.material.alpha * self.d_temp;
        for bf in 0..self.mesh.n_boundary_faces {
            if is_empty_face(&self.mesh, bf) {
                continue;
            }
            let c = self.mesh.b_face_cells[bf] as usize;
            let g = self.grad[c];
            let t = self.b_traction[bf];
            let n = self.mesh.b_sf[bf].normalised();

            let r = if self.boundary_gradient_correction {
                let gt = g - n.outer(dot_left(n, g));
                let dr = dot_right(gt, n);
                let (tn, drn) = (t.dot(n), dr.dot(n));
                let normal = (tn - mu * drn - lam * gt.trace() + thermal) / gamma;
                let tangential = (t - n * tn) * (1.0 / mu) - (dr - n * drn);
                n * normal + tangential
            } else {
                let q = deferred_traction(self.material, g, self.d_temp, self.mesh.b_sf[bf]);
                // The UNSCALED coefficient: this branch inverts the original
                // split, whose implicit half is `(2 mu + lambda)` whatever
                // `implicit_scale` is doing to the matrix.
                let gb = self.b_gamma_mag_sf[bf] / self.implicit_scale;
                if gb > 0.0 {
                    (t * self.mesh.b_mag_sf[bf] - q) * (1.0 / gb)
                } else {
                    Vec3::ZERO
                }
            };
            for i in 0..3 {
                if !self.b_fixed[bf][i] {
                    self.bc[i].ref_grad[bf] = r.component(i);
                }
            }
        }
    }

    /// `u_b` from SPEC-LIT §4's one expression, per component.
    fn evaluate_boundary(&mut self) {
        for i in 0..3 {
            for c in 0..self.mesh.n_cells {
                self.comp[c] = self.u[c].component(i);
            }
            let b = self.bc[i].evaluate(&self.mesh, &self.comp);
            for bf in 0..self.mesh.n_boundary_faces {
                match i {
                    0 => self.ub[bf].x = b[bf],
                    1 => self.ub[bf].y = b[bf],
                    _ => self.ub[bf].z = b[bf],
                }
            }
        }
    }
}

#[inline]
fn set_component(v: &mut Vec3, i: usize, s: Scalar) {
    match i {
        0 => v.x = s,
        1 => v.y = s,
        _ => v.z = s,
    }
}

impl Prototype {
    /// The right-hand side of all three systems, rebuilt from the current
    /// gradient and the current boundary values.
    ///
    /// Three pieces, and the sign of each follows from assembling `A =
    /// -laplacian`:
    ///
    /// ```text
    /// laplacian_implicit(u) + laplacian_corr + D = 0
    ///   =>  A u = b_boundary + laplacian_corr + D
    /// ```
    ///
    /// `D` is the deferred surface integral, scattered over the faces the way
    /// [`crate::reference`] scatters everything - `+q` into the owner, `-q`
    /// into the neighbour - where a kernel would gather one row per thread.
    ///
    /// A boundary face enters through whichever of the two statements its
    /// condition makes, and **never through both**:
    ///
    /// * a component fixed by value gets the implicit `snGrad` of SPEC-LIT
    ///   §4's triple plus its own `Q_b.Sf`, which together are `sigma.Sf`;
    /// * a component given a traction gets `t |Sf|` and NOTHING else, because
    ///   Demirdžić & Muzaferija's statement is that the prescribed traction IS
    ///   the face's contribution. `refGrad` exists on such a face only to
    ///   produce `u_b` for §3.5's gradient.
    ///
    /// Getting that wrong is what the first two runs of this module got wrong,
    /// and the experiment is what found it. Putting `(2 mu + lambda)|Sf|
    /// refGrad = t|Sf| - Q_b.Sf` into the equation instead of `t|Sf|` injects
    /// an unbalanced `-Q_b.Sf` of coefficient `(mu + lambda)/h`, and the outer
    /// loop then AMPLIFIES by 22.4 per iteration at `nu = 0.2` where it should
    /// contract by 0.625. Adding `Q_b.Sf` back on top of it cancels that only
    /// if the two are evaluated from the same gradient, which across a
    /// boundary correction they are not: that second arrangement still
    /// amplified, by 15.7.
    pub fn assemble_source(&mut self) {
        let m = &self.mesh;
        // What `implicit_scale` moved OUT of the deferred group and into the
        // implicit one, put back here with the opposite sign. Zero - and so
        // arithmetically absent - at the only scale a solver runs.
        let moved = (self.implicit_scale - 1.0) * self.material.implicit_gamma();
        for c in 0..m.n_cells {
            self.rhs[c] = Vec3::ZERO;
        }
        for f in 0..m.n_internal_faces {
            let p = m.owner[f] as usize;
            let n = m.neighbour[f] as usize;
            let w = m.weights[f];
            let gf = self.grad[p] * w + self.grad[n] * (1.0 - w);
            let mut q = deferred_traction(self.material, gf, self.d_temp, m.sf[f]);
            if moved != 0.0 {
                q -= dot_left(m.sf[f], gf) * moved;
            }
            self.rhs[p] += q;
            self.rhs[n] -= q;
        }
        for bf in 0..m.n_boundary_faces {
            if is_empty_face(m, bf) || !self.b_fixed[bf].iter().any(|f| *f) {
                continue;
            }
            let c = m.b_face_cells[bf] as usize;
            let mut q = deferred_traction(self.material, self.b_grad[bf], self.d_temp, m.b_sf[bf]);
            if moved != 0.0 {
                q -= dot_left(m.b_sf[bf], self.b_grad[bf]) * moved;
            }
            for i in 0..3 {
                if self.b_fixed[bf][i] {
                    let v = self.rhs[c].component(i) + q.component(i);
                    set_component(&mut self.rhs[c], i, v);
                }
            }
        }

        for i in 0..3 {
            for c in 0..m.n_cells {
                self.comp[c] = self.u[c].component(i);
                self.comp_grad[c] = grad_component(self.grad[c], i);
            }
            for v in self.a[i].source.iter_mut() {
                *v = 0.0;
            }
            // `sign = -1` here too, so the correction lands on the source with
            // the sign `A = -laplacian` asks for.
            fvm_laplacian_non_orth_correction(
                &mut self.a[i],
                m,
                &self.gamma_mag_sf,
                &self.b_gamma_mag_sf,
                &self.bc[i],
                &self.comp,
                &self.comp_grad,
                self.sn_grad,
                -1.0,
            );
            for bf in 0..m.n_boundary_faces {
                if is_empty_face(m, bf) {
                    continue;
                }
                let c = m.b_face_cells[bf] as usize;
                self.a[i].source[c] += if self.b_fixed[bf][i] {
                    let delta = m.b_delta_coeffs[bf];
                    self.b_gamma_mag_sf[bf] * self.bc[i].grad_boundary(bf, delta)
                } else {
                    self.b_traction[bf].component(i) * m.b_mag_sf[bf]
                };
            }
            for c in 0..m.n_cells {
                self.a[i].source[c] += self.rhs[c].component(i);
            }
        }
    }

    /// One application of the fixed-point map: correct the boundary, build
    /// the source, solve the three systems. Returns the solution it reached
    /// and the conjugate-gradient iterations it cost.
    ///
    /// The linear solves are warm-started from the current `u`. That is not a
    /// trick to make the experiment look cheap - it is what any outer loop
    /// does - and it changes no fixed point, only how much work the inner
    /// solver has to do to get to it.
    pub fn apply_map(&mut self, out: &mut [Vec3]) -> usize {
        self.correct_boundary();
        self.assemble_source();
        let mut its = 0;
        for i in 0..3 {
            for c in 0..self.mesh.n_cells {
                self.comp[c] = self.u[c].component(i);
            }
            let (n, _) = solve_pcg(
                &mut self.comp,
                &self.a[i],
                &self.mesh,
                &self.rd[i],
                self.linear_tolerance,
                self.linear_cap,
            );
            its += n;
            for c in 0..self.mesh.n_cells {
                set_component(&mut out[c], i, self.comp[c]);
            }
        }
        its
    }

    /// Run the Picard outer loop until the fixed-point residual has dropped
    /// `decades` decades, or `cap` iterations have gone by.
    ///
    /// The quantity measured is `r_k = F(u_{k-1}) - u_{k-1}` - the residual of
    /// the fixed-point map, NOT the increment actually applied. Without
    /// relaxation the two are the same thing; with Aitken they are not, and
    /// comparing a relaxed increment against an unrelaxed one would be
    /// comparing two different numbers and calling the difference an
    /// improvement.
    pub fn run(&mut self, aitken: bool, decades: Scalar, cap: usize) -> OuterReport {
        let n_c = self.mesh.n_cells;
        let mut report = OuterReport {
            predicted_contraction: self.material.predicted_contraction(),
            ..Default::default()
        };
        let mut next = vec![Vec3::ZERO; n_c];
        let mut r = vec![Vec3::ZERO; n_c];
        let mut r_prev = vec![Vec3::ZERO; n_c];
        let mut omega = 1.0 as Scalar;
        let mut first = 0.0 as Scalar;
        let mut prev_norm = 0.0 as Scalar;

        for k in 1..=cap {
            report.linear_iterations += self.apply_map(&mut next);
            for c in 0..n_c {
                r[c] = next[c] - self.u[c];
            }
            let norm = l2(&r);
            report.norms.push(norm);
            report.iterations = k;
            if !norm.is_finite() || (k > 1 && norm > first * 1.0e6) {
                report.diverged = true;
                break;
            }

            if k == 1 {
                first = norm;
                omega = 1.0;
            } else {
                report.ratios.push(if prev_norm > 0.0 { norm / prev_norm } else { 0.0 });
                if aitken {
                    omega = aitken_omega(omega, &r_prev, &r);
                }
                if norm <= first * (10.0 as Scalar).powf(-decades) {
                    report.converged = true;
                    for c in 0..n_c {
                        self.u[c] += r[c] * omega;
                    }
                    break;
                }
            }
            if aitken {
                report.omegas.push(omega);
            }
            for c in 0..n_c {
                self.u[c] += r[c] * omega;
            }
            r_prev.copy_from_slice(&r);
            prev_norm = norm;
        }

        // The boundary values and the gradient the caller will read stress
        // from have to belong to the displacement that was just accepted.
        self.correct_boundary();
        report.observed_contraction =
            OuterReport::geometric_mean_of_last(&report.ratios, 10);
        report
    }

    /// Setup only: the prescribed traction on ONE boundary face, replacing
    /// what the per-patch table put there.
    ///
    /// A patch-uniform table cannot state a load that varies over the patch,
    /// and the end-loaded cantilever's free end carries a parabolic shear
    /// (Timoshenko & Goodier, *Theory of Elasticity*, 3rd ed. 1970, ch. 3)
    /// whose depth integral IS the end load. Which components of the face are
    /// `Traction` and which are `Fixed` is settled by the table
    /// [`Prototype::new`] was given and is not changed here; this writes
    /// values.
    pub fn set_face_traction(&mut self, bf: usize, t: Vec3) {
        self.b_traction[bf] = t;
    }

    /// Setup only: the prescribed displacement on ONE boundary face, for the
    /// components the per-patch table made `Fixed`. The matrix does not
    /// depend on `ref_value` - only on `fr` - so this needs no reassembly.
    pub fn set_face_fixed_value(&mut self, bf: usize, v: Vec3) {
        for i in 0..3 {
            if self.b_fixed[bf][i] {
                self.bc[i].ref_value[bf] = v.component(i);
            }
        }
    }

    /// The same fixed point as [`Prototype::run`], reached by ANDERSON
    /// ACCELERATION of depth `depth` instead of by Aitken relaxation.
    ///
    /// Anderson (*J. ACM* 12 (1965) 547) in the form and notation of H. F.
    /// Walker & P. Ni, *SIAM J. Numer. Anal.* 49 (2011) 1715-1735, DOI
    /// 10.1137/10078356X, Algorithm 2, with `beta = 1`:
    ///
    /// ```text
    /// g_k = F(u_k),  f_k = g_k - u_k
    /// m_k = min(depth, k)
    /// gamma = argmin || f_k - [ df_{k-m_k} ... df_{k-1} ] gamma ||_2
    /// u_{k+1} = g_k - [ dg_{k-m_k} ... dg_{k-1} ] gamma
    /// ```
    ///
    /// where `df_j = f_{j+1} - f_j` and `dg_j = g_{j+1} - g_j`. `depth = 0`
    /// keeps no columns and is bare Picard, which is what
    /// `anderson_of_depth_zero_is_bare_picard` holds it to.
    ///
    /// Aitken is the one-column member of this family restricted to a scalar
    /// step (Kuettler & Wall 2008, DOI 10.1007/s00466-008-0255-5); the same
    /// least-squares combination over an interface displacement is IQN-ILS,
    /// J. Degroote, K.-J. Bathe & J. Vierendeels, *Comput. Struct.* 87 (2009)
    /// 793-801, DOI 10.1016/j.compstruc.2008.11.013 - so what is measured
    /// here is also the accelerator a partitioned fluid-structure coupling
    /// would reach for.
    ///
    /// The least-squares solve is on the HOST, by modified Gram-Schmidt on
    /// the residual differences ([`anderson_gamma`]): `depth <= 10` columns
    /// of a few thousand entries is nothing beside three preconditioned
    /// conjugate-gradient solves, and a normal-equation solve on differences
    /// spanning six decades is the one arrangement that would make the
    /// measurement a measurement of its own round-off.
    pub fn run_anderson(&mut self, depth: usize, decades: Scalar, cap: usize) -> OuterReport {
        let n_c = self.mesh.n_cells;
        let mut report = OuterReport {
            predicted_contraction: self.material.predicted_contraction(),
            ..Default::default()
        };
        let mut g = vec![Vec3::ZERO; n_c];
        let mut f = vec![Vec3::ZERO; n_c];
        let mut g_prev = vec![Vec3::ZERO; n_c];
        let mut f_prev = vec![Vec3::ZERO; n_c];
        let mut d_f: Vec<Vec<Vec3>> = Vec::new();
        let mut d_g: Vec<Vec<Vec3>> = Vec::new();
        let mut first = 0.0 as Scalar;
        let mut prev_norm = 0.0 as Scalar;

        for k in 1..=cap {
            report.linear_iterations += self.apply_map(&mut g);
            for c in 0..n_c {
                f[c] = g[c] - self.u[c];
            }
            let norm = l2(&f);
            report.norms.push(norm);
            report.iterations = k;
            if !norm.is_finite() || (k > 1 && norm > first * 1.0e6) {
                report.diverged = true;
                break;
            }
            if k == 1 {
                first = norm;
            } else {
                report.ratios.push(if prev_norm > 0.0 { norm / prev_norm } else { 0.0 });
                if norm <= first * (10.0 as Scalar).powf(-decades) {
                    report.converged = true;
                    for c in 0..n_c {
                        self.u[c] += f[c];
                    }
                    break;
                }
                let mut df = vec![Vec3::ZERO; n_c];
                let mut dg = vec![Vec3::ZERO; n_c];
                for c in 0..n_c {
                    df[c] = f[c] - f_prev[c];
                    dg[c] = g[c] - g_prev[c];
                }
                d_f.push(df);
                d_g.push(dg);
                while d_f.len() > depth {
                    d_f.remove(0);
                    d_g.remove(0);
                }
            }

            if d_f.is_empty() {
                // `u + (g - u)` and not `g`: the two differ in the last bit,
                // and writing the increment is what makes `depth = 0` bare
                // Picard to the BIT rather than to a tolerance.
                for c in 0..n_c {
                    self.u[c] += f[c];
                }
            } else {
                let gamma = anderson_gamma(&d_f, &f);
                for c in 0..n_c {
                    let mut v = g[c];
                    for (j, col) in d_g.iter().enumerate() {
                        v -= col[c] * gamma[j];
                    }
                    self.u[c] = v;
                }
                // Not a relaxation factor: the one-norm of the least-squares
                // combination, which is what says whether the columns have
                // gone linearly dependent.
                report.omegas.push(gamma.iter().map(|x| x.abs()).sum());
            }
            f_prev.copy_from_slice(&f);
            g_prev.copy_from_slice(&g);
            prev_norm = norm;
        }

        self.correct_boundary();
        report.observed_contraction =
            OuterReport::geometric_mean_of_last(&report.ratios, 10);
        report
    }

    /// Cauchy stress per cell, from the converged gradient.
    pub fn stress_field(&self) -> Vec<Tensor> {
        self.grad
            .iter()
            .map(|g| stress(self.material, *g, self.d_temp))
            .collect()
    }

    /// The largest stress component anywhere, divided by
    /// `(3 lambda + 2 mu) alpha dT` - the plan's Gate 95-B measure.
    pub fn relative_max_stress(&self) -> Scalar {
        let scale =
            (self.material.three_lambda_two_mu() * self.material.alpha * self.d_temp).abs();
        let mut worst = 0.0 as Scalar;
        for s in self.stress_field() {
            for v in [s.xx, s.xy, s.xz, s.yx, s.yy, s.yz, s.zx, s.zy, s.zz] {
                worst = worst.max(v.abs());
            }
        }
        if scale > 0.0 {
            worst / scale
        } else {
            worst
        }
    }

    /// The normalised residual of SPEC-LIT §8.4 for the three assembled
    /// systems at the current `u`, the largest of the three.
    ///
    /// This is the equation the loop is actually solving - implicit laplacian,
    /// non-orthogonal correction, deferred stress, thermal load and boundary
    /// pair, all of them - and not a second discretisation of equilibrium
    /// written beside it. A "check" that sums `sigma_f.Sf` from face-averaged
    /// cell gradients instead measures the gap between the wide and compact
    /// stencils, which is a discretisation error of the scheme and not a
    /// statement about whether the iteration finished.
    ///
    /// Call it after [`Prototype::run`], whose last act is to bring the
    /// boundary values and the gradient back into agreement with the
    /// displacement that was accepted.
    pub fn equilibrium_residual(&mut self) -> Scalar {
        self.assemble_source();
        let mut worst = 0.0 as Scalar;
        for i in 0..3 {
            for c in 0..self.mesh.n_cells {
                self.comp[c] = self.u[c].component(i);
            }
            worst = worst.max(crate::reference::residual(&self.comp, &self.a[i], &self.mesh));
        }
        worst
    }

    /// The assembled system of one displacement component, for the tests that
    /// diff the device operator against this one.
    pub fn system(&self, i: usize) -> &CpuLdu { &self.a[i] }
    /// The boundary triple of one displacement component.
    pub fn boundary_triple(&self, i: usize) -> &CpuScalarBc { &self.bc[i] }
    /// The boundary gradient of the last [`Prototype::correct_boundary`].
    pub fn boundary_gradient(&self) -> &[Tensor] { &self.b_grad }
}

/// The Euclidean norm of a cell vector field.
fn l2(v: &[Vec3]) -> Scalar {
    v.iter().map(|a| a.mag_sqr()).sum::<Scalar>().sqrt()
}

/// Aitken delta-squared dynamic relaxation, Küttler & Wall (2008) §3.2 in its
/// vector form:
///
/// ```text
/// omega_k = -omega_{k-1} (r_{k-1} . (r_k - r_{k-1})) / |r_k - r_{k-1}|^2
/// ```
///
/// For a fixed point that is exactly linear with one contraction `q` this
/// returns `1/(1 - q)` and the next update lands on the answer; the loop
/// below is linear in `u` for a fixed temperature field, so that is very
/// nearly what happens, and the distance between "very nearly" and "exactly"
/// is the spread of the map's spectrum. That is the whole reason the Aitken
/// leg of this experiment is worth running.
///
/// A zero denominator means the two residuals are identical - the loop has
/// stopped moving - and the previous factor is kept rather than a NaN being
/// produced.
/// The inner product of two cell vector fields, in the same sequential order
/// [`l2`] sums in.
fn field_dot(a: &[Vec3], b: &[Vec3]) -> Scalar {
    a.iter().zip(b.iter()).map(|(x, y)| x.dot(*y)).sum()
}

/// `argmin_gamma || rhs - C gamma ||_2` for the few columns of an Anderson
/// mixing, by MODIFIED Gram-Schmidt with back substitution.
///
/// Modified Gram-Schmidt and not the normal equations: the columns are
/// differences of residuals that span the decades the loop is dropping, so
/// `C^T C` carries the square of that condition number and a `gamma` solved
/// from it is arithmetic noise long before the loop has finished. A column
/// that orthogonalises to nothing - which is what a stalled direction looks
/// like - is DROPPED rather than divided by: its `gamma` entry is set to zero
/// and the remaining columns are asked to do the work, which is the standard
/// filtering of Walker & Ni (2011) in its cheapest form.
fn anderson_gamma(cols: &[Vec<Vec3>], rhs: &[Vec3]) -> Vec<Scalar> {
    let m = cols.len();
    let scale = cols.iter().map(|c| l2(c)).fold(0.0 as Scalar, |a, b| a.max(b));
    let mut q: Vec<Vec<Vec3>> = Vec::with_capacity(m);
    let mut r = vec![vec![0.0 as Scalar; m]; m];
    let mut live = vec![true; m];

    for j in 0..m {
        let mut v = cols[j].clone();
        for i in 0..j {
            if !live[i] {
                continue;
            }
            let d = field_dot(&q[i], &v);
            r[i][j] = d;
            for (vc, qc) in v.iter_mut().zip(q[i].iter()) {
                *vc -= *qc * d;
            }
        }
        let nrm = l2(&v);
        if !(nrm > 1.0e-12 * scale) {
            live[j] = false;
            r[j][j] = 1.0;
            q.push(vec![Vec3::ZERO; rhs.len()]);
        } else {
            r[j][j] = nrm;
            let inv = 1.0 / nrm;
            for vc in v.iter_mut() {
                *vc = *vc * inv;
            }
            q.push(v);
        }
    }

    let mut gamma = vec![0.0 as Scalar; m];
    for i in (0..m).rev() {
        if !live[i] {
            continue;
        }
        let mut acc = field_dot(&q[i], rhs);
        for j in (i + 1)..m {
            acc -= r[i][j] * gamma[j];
        }
        gamma[i] = acc / r[i][i];
    }
    gamma
}

fn aitken_omega(prev: Scalar, r_prev: &[Vec3], r: &[Vec3]) -> Scalar {
    let mut num = 0.0 as Scalar;
    let mut den = 0.0 as Scalar;
    for (a, b) in r_prev.iter().zip(r.iter()) {
        let d = *b - *a;
        num += a.dot(d);
        den += d.mag_sqr();
    }
    if den <= 0.0 {
        return prev;
    }
    let w = -prev * num / den;
    if w.is_finite() && w != 0.0 {
        w
    } else {
        prev
    }
}

// ==========================================================================
//  The inner solve - SPEC-LIT §8.2 and §8.4, on the host
// ==========================================================================

/// The diagonal of the incomplete Cholesky factorisation (Saad §10.2-10.3),
/// already inverted.
///
/// ```text
/// rD[P] = diag[P];   rD[N] -= upper[f] lower[f] / rD[P]   for every face f
/// ```
///
/// One pass, and it is correct in one pass only because the mesh is stored in
/// upper-triangular order (`owner[f] < neighbour[f]`, faces ascending by
/// owner): every face whose NEIGHBOUR is `P` has an owner below `P` and has
/// therefore already been visited. `build_host_mesh` validates that ordering,
/// so this is a consequence of the storage rather than an assumption about it.
///
/// This is the serial form of what SPEC-LIT §21 multi-colours on the device.
/// A prototype has no reason to colour anything.
pub fn dic_factor(a: &CpuLdu, m: &HostMesh) -> Vec<Scalar> {
    let mut rd = a.diag.clone();
    for f in 0..m.n_internal_faces {
        let p = m.owner[f] as usize;
        let n = m.neighbour[f] as usize;
        rd[n] -= a.upper[f] * a.lower[f] / rd[p];
    }
    for v in rd.iter_mut() {
        *v = 1.0 / *v;
    }
    rd
}

/// `w = M^-1 r` for the factorisation above: forward substitution over the
/// faces ascending, then backward over them descending.
fn dic_precondition(w: &mut [Scalar], r: &[Scalar], rd: &[Scalar], a: &CpuLdu, m: &HostMesh) {
    for c in 0..m.n_cells {
        w[c] = rd[c] * r[c];
    }
    for f in 0..m.n_internal_faces {
        let p = m.owner[f] as usize;
        let n = m.neighbour[f] as usize;
        w[n] -= rd[n] * a.lower[f] * w[p];
    }
    for f in (0..m.n_internal_faces).rev() {
        let p = m.owner[f] as usize;
        let n = m.neighbour[f] as usize;
        w[p] -= rd[p] * a.upper[f] * w[n];
    }
}

/// Preconditioned conjugate gradients (Hestenes & Stiefel 1952; SPEC-LIT
/// §8.2), stopped on SPEC-LIT §8.4's normalised residual.
///
/// `x` enters as the initial guess and leaves as the solution. Returns the
/// iterations taken and the residual reached.
pub fn solve_pcg(
    x: &mut [Scalar],
    a: &CpuLdu,
    m: &HostMesh,
    rd: &[Scalar],
    tol: Scalar,
    cap: usize,
) -> (usize, Scalar) {
    let n = m.n_cells;
    let mut ap = Vec::new();
    amul(&mut ap, x, a, m);
    let mut r: Vec<Scalar> = (0..n).map(|c| a.source[c] - ap[c]).collect();

    let norm = crate::reference::norm_factor(x, a, m);
    let resid = |r: &[Scalar]| r.iter().map(|v| v.abs()).sum::<Scalar>() / norm;
    let mut res = resid(&r);
    if res <= tol {
        return (0, res);
    }

    let mut p = vec![0.0 as Scalar; n];
    let mut w = vec![0.0 as Scalar; n];
    let mut rho_old = 0.0 as Scalar;

    for it in 1..=cap {
        dic_precondition(&mut w, &r, rd, a, m);
        let rho: Scalar = (0..n).map(|c| w[c] * r[c]).sum();
        if it == 1 {
            p.copy_from_slice(&w);
        } else {
            let beta = rho / rho_old;
            for c in 0..n {
                p[c] = w[c] + beta * p[c];
            }
        }
        amul(&mut ap, &p, a, m);
        let pap: Scalar = (0..n).map(|c| p[c] * ap[c]).sum();
        if pap == 0.0 || !pap.is_finite() {
            return (it, res);
        }
        let alpha = rho / pap;
        for c in 0..n {
            x[c] += alpha * p[c];
            r[c] -= alpha * ap[c];
        }
        rho_old = rho;
        res = resid(&r);
        if res <= tol {
            return (it, res);
        }
    }
    (cap, res)
}

// ==========================================================================
//  Tests
//
//  Against physics and against closed forms, never against a second run of
//  this module (SPEC-LIT §0 rule 4, §10).
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const DT: Scalar = 100.0;

    /// All six faces prescribed, one of them by a non-zero displacement so
    /// that the problem is not the trivial `u = 0`. No traction condition
    /// anywhere, so what this measures is the INTERIOR split and nothing else.
    fn all_fixed() -> PatchBcs {
        let mut b = [[CompBc::Fixed(0.0); 3]; 6];
        b[1] = [CompBc::Fixed(1.0e-3), CompBc::Fixed(5.0e-4), CompBc::Fixed(0.0)];
        b
    }

    /// `(mu + lambda)/(2 mu + lambda)` and `1/(2(1 - nu))` are the same
    /// number. The plan asserts that identity in prose; here it is arithmetic.
    #[test]
    fn the_predicted_contraction_is_one_over_two_one_minus_nu() {
        for nu in [-0.5, 0.0, 0.2, 0.3, 0.45, 0.49, 0.499] {
            let m = Material::steel(nu);
            let closed = 1.0 / (2.0 * (1.0 - nu));
            let from_lame = m.predicted_contraction();
            assert!(
                (from_lame - closed).abs() <= 1e-14 * closed.abs(),
                "nu = {nu}: (mu+lambda)/(2mu+lambda) = {from_lame}, 1/(2(1-nu)) = {closed}"
            );
        }
    }

    /// Implicit plus deferred is the whole traction, for an arbitrary
    /// gradient, temperature and area vector. If this drifts, every number the
    /// module produces is about a different equation than the one the plan's
    /// §D.3 writes down.
    #[test]
    fn the_split_reassembles_the_traction() {
        let m = Material { e: 7.3e10, nu: 0.34, alpha: 2.31e-5 };
        let g = Tensor {
            xx: 1.0e-4, xy: -3.0e-5, xz: 7.0e-6,
            yx: 2.0e-5, yy: -5.0e-4, yz: 1.1e-4,
            zx: -8.0e-6, zy: 4.0e-5, zz: 9.0e-5,
        };
        let sf = Vec3::new(0.31, -0.72, 0.44);

        let s = stress(m, g, 87.0);
        let from_stress = dot_left(sf, s);
        let from_split = traction(m, g, 87.0, sf);
        for i in 0..3 {
            let (a, b) = (from_stress.component(i), from_split.component(i));
            assert!(
                (a - b).abs() <= 1e-12 * a.abs().max(1.0),
                "component {i}: sigma.Sf = {a}, implicit + deferred = {b}"
            );
        }
        // And the deferred half really is the smaller one.
        let implicit = dot_left(sf, g) * m.implicit_gamma();
        assert!(deferred_traction(m, g, 0.0, sf).mag() < implicit.mag());
    }

    /// The jitter has to produce a mesh SPEC-LIT §2.4's correction has
    /// something to do on, or the leg of the experiment that rests on it means
    /// nothing.
    #[test]
    fn the_jittered_mesh_is_actually_non_orthogonal() {
        let m = jittered_block(8, 0.25).unwrap();
        let rep = m.check();
        assert!(
            rep.max_non_orth_deg > 5.0,
            "the jitter produced a mesh of only {} degrees non-orthogonality",
            rep.max_non_orth_deg
        );
        assert!(rep.min_volume > 0.0, "the jitter inverted a cell");
    }

    /// Gate 95-B in prototype form (Boley & Weiner ch. 1): an unrestrained
    /// block at a uniform `dT` expands to `u = alpha dT x` and carries **no**
    /// stress. The field is linear, so the discrete answer is the exact one
    /// and the only thing between the two is the iteration.
    #[test]
    fn free_expansion_carries_no_stress() {
        let mat = Material::steel(0.3);
        let mut p = Prototype::new(block(10).unwrap(), mat, DT, &free_expansion());
        let r = p.run(true, 14.0, 600);
        let rel = p.relative_max_stress();
        println!(
            "free expansion: outer={} conv={} rel|sigma|={rel:e}",
            r.iterations, r.converged
        );
        assert!(
            rel <= 1e-12,
            "free expansion carries stress: max|sigma| / ((3 lambda + 2 mu) alpha dT) = {rel:e}"
        );

        let mut worst = 0.0 as Scalar;
        for c in 0..p.mesh.n_cells {
            let exact = p.mesh.c[c] * (mat.alpha * DT);
            worst = worst.max((p.u[c] - exact).mag() / exact.mag());
        }
        assert!(worst <= 1e-10, "u differs from alpha dT x by {worst:e} relative");
    }

    /// **The interior split does what the plan derives.** With every boundary
    /// prescribed by displacement there is no traction condition anywhere, and
    /// the measured contraction lands on `1/(2(1 - nu))` from just below - the
    /// "just below" being the `cos^2(kh/2)` by which the wide deferred stencil
    /// falls short of the compact implicit one.
    #[test]
    fn the_interior_contraction_tracks_the_prediction() {
        for nu in [0.2, 0.45] {
            let mut p =
                Prototype::new(block(16).unwrap(), Material::steel(nu), DT, &all_fixed());
            let r = p.run(false, 6.0, 600);
            assert!(r.converged, "nu = {nu} did not reach six decades in 600 outer");
            let (o, e) = (r.observed_contraction, r.predicted_contraction);
            assert!(
                o <= e && o >= 0.9 * e,
                "nu = {nu}: observed interior contraction {o:.4}, predicted {e:.4}, \
                 {} outer iterations",
                r.iterations
            );
        }
    }

    /// **The traction-free surface is what the derivation left out.** Same
    /// material, same mesh, same split; one face fixed and five free instead
    /// of six fixed. The contraction is materially worse than `1/(2(1 - nu))`,
    /// and this test exists so that nobody re-derives the estimate from the
    /// interior alone and believes it.
    #[test]
    fn the_traction_boundary_is_what_slows_the_loop() {
        let mat = Material::steel(0.2);
        let interior = {
            let mut p = Prototype::new(block(16).unwrap(), mat, DT, &all_fixed());
            p.run(false, 6.0, 600)
        };
        let mut p = Prototype::new(block(16).unwrap(), mat, DT, &fixed_minus_x());
        let free = p.run(false, 6.0, 600);
        assert!(interior.converged && free.converged);
        assert!(
            free.observed_contraction > interior.observed_contraction + 0.1,
            "traction-free contraction {:.4} vs all-Dirichlet {:.4} - if these have \
             converged on each other the boundary treatment has changed and the \
             plan's §F.1a numbers no longer describe this code",
            free.observed_contraction,
            interior.observed_contraction
        );
        assert!(
            p.equilibrium_residual() <= 1e-4,
            "six decades of the fixed-point residual left an equation residual of {:e}",
            p.equilibrium_residual()
        );
    }

    /// **The headline of the experiment.** At `nu = 0.45` on a body with a
    /// free surface the bare Picard loop of the plan's §D.3 does not converge
    /// at 0.909 - it does not converge at all. Küttler & Wall's factor is what
    /// makes it converge, and at a rate BETTER than the derivation predicts.
    ///
    /// This is the test the refusal threshold of §D.5 should be read off, and
    /// it is written as an assertion so that a later change which quietly
    /// makes bare Picard stable, or quietly makes Aitken stop paying, fails
    /// here and is looked at.
    #[test]
    fn bare_picard_diverges_at_nu_0_45_where_aitken_converges() {
        let mat = Material::steel(0.45);
        let bare = {
            let mut p = Prototype::new(block(16).unwrap(), mat, DT, &fixed_minus_x());
            p.run(false, 6.0, 300)
        };
        assert!(
            bare.diverged || (!bare.converged && bare.norms.last() > bare.norms.first()),
            "bare Picard at nu = 0.45 converged in {} outer iterations - it did not \
             when this was measured, and the plan's §F.1a rests on that",
            bare.iterations
        );

        let mut p = Prototype::new(block(16).unwrap(), mat, DT, &fixed_minus_x());
        let aitken = p.run(true, 6.0, 300);
        assert!(
            aitken.converged,
            "Aitken did not reach six decades at nu = 0.45 in 300 outer iterations"
        );
        assert!(
            aitken.observed_contraction <= aitken.predicted_contraction,
            "Aitken contracted at {:.4}, worse than the predicted {:.4}",
            aitken.observed_contraction,
            aitken.predicted_contraction
        );
        assert!(
            p.equilibrium_residual() <= 1e-4,
            "the Aitken run finished on an equation residual of {:e}",
            p.equilibrium_residual()
        );
    }

    // ======================================================================
    //  The slender cantilever - the region TS-0's sweep never entered
    // ======================================================================

    /// A beam of span `l` along `x`, depth `2c` in `y` and ONE cell through
    /// the thickness in `z`, whose cells are cubes when `n_x/l == n_y/(2c)`.
    ///
    /// Six real patches, `blockgen`'s `-x +x -y +y -z +z` slot order: the
    /// `-z`/`+z` pair is named rather than left empty, because an empty face
    /// contributes to no surface integral and would delete two of the six
    /// tractions of a three-dimensional stress problem - the same reason
    /// [`block`] names all six.
    fn cantilever_mesh(n_x: usize, n_y: usize, l: Scalar, c: Scalar) -> HostMesh {
        let axis = |lo: Scalar, hi: Scalar, n: usize| GradedAxis {
            lo,
            hi,
            n,
            expansion: 1.0,
            two_sided: false,
        };
        blockgen::build_mesh(&BlockSpec {
            x: axis(0.0, l, n_x),
            y: axis(-c, c, n_y),
            z: axis(0.0, 2.0 * c / n_y as Scalar, 1),
            patch_type: ["patch", "patch", "patch", "patch", "patch", "patch"]
                .map(String::from),
            ..Default::default()
        })
        .unwrap()
    }

    /// Timoshenko & Goodier, *Theory of Elasticity*, 3rd ed. (1970) ch. 3:
    /// the cantilever loaded at its free end. Origin at the FREE end on the
    /// neutral axis, `x` toward the built-in end at `x = l`, `y` in
    /// `[-c, c]`, unit thickness, `I = 2c^3/3`, load `P` in `+y` at `x = 0`.
    /// Plane strain through `E' = E/(1 - nu^2)`, `nu' = nu/(1 - nu)`, `G`
    /// unchanged (ch. 2). The rigid-body constant is fixed by "a vertical
    /// element of the cross-section at the built-in end remains vertical".
    ///
    /// Only what the sweep needs: the end load's own traction, the clamp's
    /// own displacement, and the tip deflection the converged runs are
    /// checked against. The gate that measures ORDER is a separate matter and
    /// is not this test's business.
    struct CantileverExact {
        p: Scalar,
        l: Scalar,
        c: Scalar,
        i: Scalar,
        e_p: Scalar,
        nu_p: Scalar,
        g: Scalar,
    }

    impl CantileverExact {
        fn new(e: Scalar, nu: Scalar, l: Scalar, c: Scalar, p: Scalar) -> Self {
            Self {
                p,
                l,
                c,
                i: 2.0 * c * c * c / 3.0,
                e_p: e / (1.0 - nu * nu),
                nu_p: nu / (1.0 - nu),
                g: e / (2.0 * (1.0 + nu)),
            }
        }

        /// `-P (c^2 - y^2)/(2I)`. Its depth integral is `-P`, so the parabolic
        /// shear the free end carries IS the end load and nothing is left to a
        /// Saint-Venant end effect.
        fn sigma_xy(&self, y: Scalar) -> Scalar {
            -self.p * (self.c * self.c - y * y) / (2.0 * self.i)
        }

        fn u(&self, x: Scalar, y: Scalar) -> Scalar {
            -self.p * x * x * y / (2.0 * self.e_p * self.i)
                - self.nu_p * self.p * y * y * y / (6.0 * self.e_p * self.i)
                + self.p * y * y * y / (6.0 * self.i * self.g)
                + self.p * self.l * self.l * y / (2.0 * self.e_p * self.i)
        }

        fn v(&self, x: Scalar, y: Scalar) -> Scalar {
            self.nu_p * self.p * x * y * y / (2.0 * self.e_p * self.i)
                + self.p * x * x * x / (6.0 * self.e_p * self.i)
                - (self.p * self.c * self.c / (2.0 * self.i * self.g)
                    + self.p * self.l * self.l / (2.0 * self.e_p * self.i))
                    * x
                + self.p * self.l * self.l * self.l / (3.0 * self.e_p * self.i)
                + self.p * self.c * self.c * self.l / (2.0 * self.i * self.g)
        }

        /// `v(0, y)`, which does not depend on `y`.
        fn tip(&self) -> Scalar {
            self.p * self.l * self.l * self.l / (3.0 * self.e_p * self.i)
                + self.p * self.c * self.c * self.l / (2.0 * self.i * self.g)
        }
    }

    /// The end-loaded cantilever, built on a [`Prototype`] with an optional
    /// implicit scale: the free end carries the closed form's parabolic
    /// shear face by face, the built-in end carries the closed form's own
    /// displacement face by face, the two faces of the slab are symmetry
    /// planes (`u_z = 0`), and `-y`/`+y` are free. Isothermal - `d_temp = 0`
    /// puts the thermal load at zero on every face, which is the state the
    /// closed form is a solution of.
    fn cantilever(
        n_x: usize,
        n_y: usize,
        l: Scalar,
        c: Scalar,
        nu: Scalar,
        kappa: Scalar,
    ) -> (Prototype, CantileverExact) {
        let e = 200.0e9;
        let exact = CantileverExact::new(e, nu, l, c, 1.0e5);
        let mat = Material { e, nu, alpha: 1.2e-5 };
        let mut bcs = all_free();
        bcs[1] = [CompBc::Fixed(0.0); 3];
        bcs[4][2] = CompBc::Fixed(0.0);
        bcs[5][2] = CompBc::Fixed(0.0);
        let mesh = cantilever_mesh(n_x, n_y, l, c);
        let mut p = Prototype::new_with_implicit_scale(mesh, mat, 0.0, &bcs, kappa);
        for bf in 0..p.mesh.n_boundary_faces {
            let y = p.mesh.b_cf[bf].y;
            match p.mesh.b_patch[bf] as usize {
                0 => p.set_face_traction(bf, Vec3::new(0.0, -exact.sigma_xy(y), 0.0)),
                1 => p.set_face_fixed_value(
                    bf,
                    Vec3::new(exact.u(exact.l, y), exact.v(exact.l, y), 0.0),
                ),
                _ => {}
            }
        }
        (p, exact)
    }

    /// The `|Sf|`-weighted mean of `u_y` over the loaded end, which is the
    /// discrete reading of the closed form's `v(0, y)`.
    fn tip_deflection(p: &Prototype) -> Scalar {
        let (mut w, mut v) = (0.0 as Scalar, 0.0 as Scalar);
        for bf in 0..p.mesh.n_boundary_faces {
            if p.mesh.b_patch[bf] as usize == 0 {
                w += p.mesh.b_mag_sf[bf];
                v += p.mesh.b_mag_sf[bf] * p.ub[bf].y;
            }
        }
        v / w
    }

    /// Keeping no columns at all is bare Picard, exactly - the property that
    /// makes [`Prototype::run_anderson`] a superset of the loop it is being
    /// compared against rather than a second loop with its own habits.
    #[test]
    fn anderson_of_depth_zero_is_bare_picard() {
        let mat = Material::steel(0.3);
        let mut a = Prototype::new(block(8).unwrap(), mat, DT, &fixed_minus_x());
        let picard = a.run(false, 4.0, 200);
        let mut b = Prototype::new(block(8).unwrap(), mat, DT, &fixed_minus_x());
        let anderson = b.run_anderson(0, 4.0, 200);
        assert_eq!(picard.iterations, anderson.iterations);
        assert_eq!(picard.norms, anderson.norms);
        for (x, y) in a.u.iter().zip(b.u.iter()) {
            assert_eq!(*x, *y);
        }
    }

    /// The least-squares solve, against a problem whose answer is known: two
    /// orthogonal columns and a right-hand side that is `2` of the first and
    /// `-3` of the second plus something orthogonal to both.
    #[test]
    fn the_anderson_least_squares_is_a_least_squares() {
        let c0 = vec![Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO];
        let c1 = vec![Vec3::new(0.0, 2.0, 0.0), Vec3::ZERO];
        let rhs = vec![Vec3::new(2.0, -6.0, 7.0), Vec3::new(0.0, 0.0, 5.0)];
        let g = anderson_gamma(&[c0.clone(), c1.clone()], &rhs);
        assert!((g[0] - 2.0).abs() < 1e-14, "{g:?}");
        assert!((g[1] + 3.0).abs() < 1e-14, "{g:?}");
        // A column that repeats another carries no information, and asking a
        // normal-equation solve for it would be a division by zero.
        let g = anderson_gamma(&[c0.clone(), c0], &rhs);
        assert!(g.iter().all(|x| x.is_finite()), "{g:?}");
    }

    /// `implicit_scale` moves the iteration matrix, and moves the CONVERGED
    /// DISPLACEMENT too - by the difference between the two discrete
    /// operators the exact cancellation is written between, which is a
    /// quantity that vanishes under refinement and is not small at any mesh a
    /// person would run.
    ///
    /// This is the test that stops `kappa` being described as free. The
    /// continuum split is exact for every `kappa`; the discrete one is not,
    /// because the implicit half is SPEC-LIT §3.2's compact laplacian and the
    /// half moved out of the deferred group is §3.5's Green-Gauss gradient
    /// interpolated to the face. On the block with one fixed face - whose
    /// continuous stress is singular along every edge where the fixed face
    /// meets a free one - `kappa = 2` was measured to move the peak
    /// displacement by 1.8e-2 of its own size at `h = 1/8` and 1.5e-2 at
    /// `h = 1/16`: a drift that is not going away on any mesh a person would
    /// run. `kappa` buys whatever iteration count it buys by changing the
    /// answer, and that is the sentence it is refused on.
    #[test]
    fn a_larger_implicit_coefficient_moves_the_converged_displacement() {
        let mat = Material::steel(0.3);
        let drift = |n: usize| -> Scalar {
            let mut one = Prototype::new(block(n).unwrap(), mat, DT, &fixed_minus_x());
            let a = one.run(true, 8.0, 400);
            let mut two = Prototype::new_with_implicit_scale(
                block(n).unwrap(),
                mat,
                DT,
                &fixed_minus_x(),
                2.0,
            );
            let b = two.run(true, 8.0, 400);
            assert!(a.converged && b.converged, "{} {}", a.iterations, b.iterations);
            let scale = one.u.iter().map(|v| v.mag()).fold(0.0 as Scalar, |x, y| x.max(y));
            one.u
                .iter()
                .zip(two.u.iter())
                .map(|(x, y)| (*x - *y).mag())
                .fold(0.0 as Scalar, |x, y| x.max(y))
                / scale
        };
        let coarse = drift(8);
        let fine = drift(16);
        assert!(
            coarse > 1.0e-3 && fine > 1.0e-3,
            "kappa = 2 moved the converged displacement by {coarse:.3e} of its own size \
             at h = 1/8 and {fine:.3e} at h = 1/16 - if those have fallen to nothing the \
             drift is no longer what decided against kappa, and the decision written in \
             docs/09-thermal-structural-plan.md F.1b has to be re-read"
        );
    }

    /// **The slender-cantilever sweep**, printed. `#[ignore]`d for the same
    /// reason [`the_risk_one_sweep`] is: minutes of arithmetic whose product
    /// is a table for a document, not a verdict a build should wait on.
    ///
    /// ```text
    /// cargo test --release --lib -- \
    ///     solid::prototype::tests::the_slender_cantilever_sweep --ignored --nocapture
    /// ```
    ///
    /// What it answers: TS-0's sweep measured a COMPACT body - a `20^3` block
    /// with one face fixed - and `docs/09-thermal-structural-plan.md` §F.1a's
    /// numbers are that region's. An end-loaded cantilever is the other
    /// region: bending, where the deformation is carried by the off-diagonal
    /// `du_x/dy` / `du_y/dx` pair that lives ENTIRELY in the deferred half of
    /// the split, while the implicit half is a decoupled laplacian per
    /// component that knows nothing about it. The three candidates are the
    /// baseline, Anderson acceleration at three depths, and a larger implicit
    /// coefficient at three factors, all on the same 10:1 beam.
    #[test]
    #[ignore = "the measurement itself - minutes; run it with --ignored --nocapture"]
    fn the_slender_cantilever_sweep() {
        const CAP: usize = 2000;
        const DECADES: Scalar = 6.0;
        let (c, n_y) = (0.1, 8_usize);
        let cells_per_c = n_y as Scalar / (2.0 * c);

        println!(
            "\n  case                          candidate        outer   observed  predicted  \
             conv   cg iters   tip rel err"
        );
        let row = |case: &str, cand: &str, r: &OuterReport, tip: Scalar| {
            println!(
                "  {case:<28}  {cand:<14}  {:6}  {:9.4}  {:9.4}  {:>5}  {:9}  {:>12}",
                r.iterations,
                r.observed_contraction,
                r.predicted_contraction,
                if r.diverged { "DIVG" } else if r.converged { "true" } else { "false" },
                r.linear_iterations,
                if r.converged { format!("{tip:.3e}") } else { "-".to_string() },
            );
        };

        // Every candidate at every aspect ratio, so that the 1:1 row is a
        // CONTROL for its own candidate: same boundary topology, same load,
        // same material, compact body. An accelerator that does not beat the
        // loop on the case the loop already solves is an accelerator that is
        // not working, and there would be no reading a negative result off
        // it.
        for (l, label) in [(0.2, "1:1"), (0.5, "2.5:1"), (1.0, "5:1"), (2.0, "10:1")] {
            let n_x = (l * cells_per_c).round() as usize;
            for nu in [0.2, 0.3] {
                for aitken in [true, false] {
                    let (mut p, exact) = cantilever(n_x, n_y, l, c, nu, 1.0);
                    let r = p.run(aitken, DECADES, CAP);
                    let tip = (tip_deflection(&p) - exact.tip()).abs() / exact.tip();
                    row(
                        &format!("{label} beam, nu = {nu}"),
                        if aitken { "Picard+Aitken" } else { "Picard" },
                        &r,
                        tip,
                    );
                }
            }
            for depth in [3_usize, 5, 10] {
                let (mut p, exact) = cantilever(n_x, n_y, l, c, 0.3, 1.0);
                let r = p.run_anderson(depth, DECADES, CAP);
                let tip = (tip_deflection(&p) - exact.tip()).abs() / exact.tip();
                row(&format!("{label} beam, nu = 0.3"), &format!("Anderson m={depth}"), &r, tip);
            }
            for kappa in [1.5, 2.0, 4.0] {
                let (mut p, exact) = cantilever(n_x, n_y, l, c, 0.3, kappa);
                let r = p.run(true, DECADES, CAP);
                let tip = (tip_deflection(&p) - exact.tip()).abs() / exact.tip();
                row(&format!("{label} beam, nu = 0.3"), &format!("kappa = {kappa}"), &r, tip);
            }
            println!();
        }

        // Does the edge move with the mesh? A threshold the refusal is set
        // at has to be a statement about the BODY, and one mesh cannot tell
        // a slenderness limit from a resolution limit.
        for (l, label) in [(1.0, "5:1"), (2.0, "10:1")] {
            for ny in [4_usize, 8, 16] {
                let n_x = (l * ny as Scalar / (2.0 * c)).round() as usize;
                let (mut p, exact) = cantilever(n_x, ny, l, c, 0.3, 1.0);
                let r = p.run_anderson(5, DECADES, CAP);
                let tip = (tip_deflection(&p) - exact.tip()).abs() / exact.tip();
                row(
                    &format!("{label} beam, {n_x}x{ny}"),
                    "Anderson m=5",
                    &r,
                    tip,
                );
            }
        }
        println!();

        // The compact body of `docs/09-thermal-structural-plan.md` F.1a, with
        // the same two accelerators on it: the row that says the Anderson
        // written here IS an accelerator, measured where the plan's own sweep
        // already knows the answer (71 outer iterations with Aitken).
        for depth in [0_usize, 3, 5, 10] {
            let mut p = Prototype::new(block(20).unwrap(), Material::steel(0.45), DT, &fixed_minus_x());
            let r = p.run_anderson(depth, DECADES, CAP);
            row("20^3 block, nu = 0.45", &format!("Anderson m={depth}"), &r, Scalar::NAN);
        }
        {
            let mut p = Prototype::new(block(20).unwrap(), Material::steel(0.45), DT, &fixed_minus_x());
            let r = p.run(true, DECADES, CAP);
            row("20^3 block, nu = 0.45", "Picard+Aitken", &r, Scalar::NAN);
        }

        println!(
            "\n  beams: cubic cells of side 2c/{n_y} on a depth of 0.2 and the span the\n  \
             aspect ratio names; E = 200 GPa, nu as the row says, end load P = 1e5,\n  \
             isothermal. Block: the compact body of docs/09-thermal-structural-plan.md\n  \
             F.1a, one face fixed, uniform dT = {DT}. Six decades of the fixed-point\n  \
             residual or {CAP} outer iterations. 'tip rel err' is the |Sf|-weighted mean\n  \
             of u_y over the loaded end against Timoshenko & Goodier's v(0, y), printed\n  \
             only where the loop converged - a number read off a stalled solve is not a\n  \
             number - and on the kappa rows it is also what the moved split did to the\n  \
             answer, the same mesh and the same load being solved."
        );
    }

    /// The whole risk-1 sweep, printed. `#[ignore]`d because it is minutes of
    /// arithmetic and its product is a table for a document, not a verdict a
    /// build should wait on:
    ///
    /// ```text
    /// cargo test --release --lib -- solid::prototype::tests::the_risk_one_sweep \
    ///     --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "the measurement itself - minutes; run it with --ignored --nocapture"]
    fn the_risk_one_sweep() {
        let header = || {
            println!(
                "\n{:>5} {:>10} {:>7} {:>7} {:>7} {:>10} {:>10} {:>6} {:>10} {:>9}",
                "nu", "mesh", "aitken", "passes", "outer", "observed", "predicted",
                "conv", "cg iters", "residual"
            );
        };
        let row = |nu: Scalar, mesh: &str, aitken: bool, passes: usize,
                   r: &OuterReport, eq: Scalar| {
            println!(
                "{:>5} {:>10} {:>7} {:>7} {:>7} {:>10.4} {:>10.4} {:>6} {:>10} {:>9.2e}{}",
                nu, mesh, aitken, passes, r.iterations, r.observed_contraction,
                r.predicted_contraction, r.converged, r.linear_iterations, eq,
                if r.diverged { "  DIVERGED" } else { "" }
            );
        };

        header();
        for &(n, name) in &[(20usize, "20^3"), (40usize, "40^3")] {
            for &nu in &[0.2 as Scalar, 0.3, 0.45, 0.49] {
                for &aitken in &[false, true] {
                    for &passes in &[1usize, 3] {
                        let mut p = Prototype::new(
                            block(n).unwrap(),
                            Material::steel(nu),
                            DT,
                            &fixed_minus_x(),
                        );
                        p.boundary_passes = passes;
                        let r = p.run(aitken, 6.0, 2000);
                        let eq = p.equilibrium_residual();
                        row(nu, name, aitken, passes, &r, eq);
                    }
                }
            }
        }

        println!("\nthe same, on a mesh that is neither orthogonal nor unskewed");
        header();
        for &nu in &[0.2 as Scalar, 0.3, 0.45, 0.49] {
            for &aitken in &[false, true] {
                for &passes in &[1usize, 3] {
                    let mut p = Prototype::new(
                        jittered_block(20, 0.25).unwrap(),
                        Material::steel(nu),
                        DT,
                        &fixed_minus_x(),
                    );
                    p.boundary_passes = passes;
                    let r = p.run(aitken, 6.0, 2000);
                    let eq = p.equilibrium_residual();
                    row(nu, "20^3 jit", aitken, passes, &r, eq);
                }
            }
        }

        println!("\nthe interior alone - every boundary prescribed by displacement");
        header();
        for &nu in &[0.2 as Scalar, 0.3, 0.45, 0.49] {
            let mut p =
                Prototype::new(block(20).unwrap(), Material::steel(nu), DT, &all_fixed());
            let r = p.run(false, 6.0, 2000);
            let eq = p.equilibrium_residual();
            row(nu, "20^3", false, 3, &r, eq);
        }

    }

    /// Gate 95-B's number on two meshes and three Poisson ratios, printed.
    /// Fourteen decades of the fixed-point residual, because the stress is a
    /// gradient of the displacement and a relative `1e-12` on it is a
    /// relative `1e-13` on `u` at ten cells across.
    #[test]
    #[ignore = "the free-expansion table - about a minute; run it with --ignored --nocapture"]
    fn the_free_expansion_table() {
        println!("
free expansion - the stress that should not be there");
        for &nu in &[0.2 as Scalar, 0.3, 0.45] {
            for &n in &[10usize, 20] {
                let mut p =
                    Prototype::new(block(n).unwrap(), Material::steel(nu), DT, &free_expansion());
                let r = p.run(true, 14.0, 1500);
                println!(
                    "  nu={nu} {n}^3: outer={} conv={} max|sigma|/((3lambda+2mu) alpha dT) = {:.3e}",
                    r.iterations,
                    r.converged,
                    p.relative_max_stress()
                );
            }
        }
    }
}
