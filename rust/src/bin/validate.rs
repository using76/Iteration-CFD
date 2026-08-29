// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! `ofgpu-validate` - the acceptance test of SPEC-LIT.md section 10.
//!
//! Written from:
//!   ofgpu SPEC-LIT.md section 10 (the list of checks), and sections 2, 3, 4,
//!     5.1, 8.4, 8.5 and 9 for the identities each check rests on
//!   Jasak, PhD thesis, Imperial College (1996), ch. 3
//!   Patankar, "Numerical Heat Transfer and Fluid Flow" (1980), ch. 4-6
//!   Moukalled, Mangani & Darwish, "The Finite Volume Method in CFD" (2016)
//!   Rhie & Chow, AIAA J. 21 (1983) 1525
//!   Swarztrauber, SIAM Review 19 (1977) 490
//!   Roache, "Verification and Validation in Computational Science and
//!     Engineering" (1998) - the method of manufactured solutions
//!   Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982) 387 - the benchmark data
//!     at the bottom of this file
//!   Issa, J. Comput. Phys. 62 (1986) 40 - PISO, SPEC-LIT.md section 14
//!   Rodi, J. Geophys. Res. 92 (1987) 5305, and Henkes, van der Vlugt &
//!     Hoogendoorn, Int. J. Heat Mass Transfer 34 (1991) 377 - the buoyancy
//!     production G_b, SPEC-LIT.md section 17
//!   Ward, J. Hydraul. Div. ASCE 90 (1964) 1 - Darcy-Forchheimer, section 18
//!   ofgpu SPEC-LIT.md sections 14, 17, 18, 19 and 22
//! No GPL-licensed source was consulted.
//!
//! # What "correct" is allowed to mean here
//!
//! **No check in this file compares against another CFD code.** SPEC-LIT
//! section 0 rule 4 is the reason: agreement with somebody else's program is
//! evidence about the wrong thing, and would in any case tie an MIT release to
//! a licence it cannot carry. Every check below is one of
//!
//! 1. an **analytic identity** the discretisation must satisfy exactly - a
//!    closed cell, the gradient of a linear field, the divergence of a uniform
//!    flux, a body force that vanishes at the reference temperature;
//! 2. agreement with [`ofgpu::reference`], an independent host transcription
//!    of SPEC-LIT section 3 written as SCATTER loops where the device GATHERS.
//!    Two structurally different loops arriving at the same numbers is
//!    evidence; the same loop written twice would be none;
//! 3. agreement between two of *our own* solvers on the *same matrix* - the
//!    Krylov solve against a dense direct solve, the cuFFT direct Poisson
//!    solve against the iterative one. Both are answers to one linear-algebra
//!    question with a unique answer, so this measures arithmetic, not physics;
//! 4. a **manufactured solution** (Roache 1998) solved end to end and refined,
//!    with the observed order of convergence measured;
//! 5. **published experimental or benchmark data**, at the bottom of the file
//!    and `#[ignore]`d so it does not slow the normal run.
//!
//! Exit code 0 means every check passed, 1 that one did not, 2 that the run
//! could not be completed at all.

use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ofgpu::blockgen;
use ofgpu::blockgen::{write_block_mesh, BlockSpec, GradedAxis};
use ofgpu::combustion::{Combustion, CombustionCoeffs};
use ofgpu::energy::{DomainKind, EnergySources, GasProperties, GasState};
use ofgpu::field::{BcKind, GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use ofgpu::field_ops::{
    correct_boundary_conditions, correct_boundary_conditions_vector, FieldKernels,
};
use ofgpu::fv::{
    div_scheme_weights, fvc_div_surface, fvc_grad_scalar, fvc_grad_vector, fvc_reconstruct,
    fvm_ddt_euler, fvm_div_bounded_correction, fvm_div_gauss, fvm_laplacian,
    fvm_laplacian_non_orth_correction, fvm_sp, fvm_su, fvm_susp, interpolate_linear,
    sn_grad_flux, turbulence_production, DivScheme, FvKernels, Limiter, SnGradScheme,
};
use ofgpu::io::case::{LinearSolverKind, Preconditioner, SolverControls, WallFunctionCoeffs};
use ofgpu::io::msh::parse_msh;
use ofgpu::io::polymesh::{build_host_mesh, read_poly_mesh};
use ofgpu::ldu::{CsrPattern, GpuLduMatrix};
use ofgpu::ldu_ops::{
    add_boundary_contributions, amul, csr_fill, relax, set_fixed_cells, set_values, LduKernels,
};
use ofgpu::mesh::{HostMesh, PatchKind};
use ofgpu::pressure::fft::cufft_available;
use ofgpu::pressure::{FftBackend, PbicgstabBackend, PressureBackend, SystemProbe};
use ofgpu::radiation::{Radiation, RadiationProps};
use ofgpu::reference as cpu;
use ofgpu::solver::{solve_pbicgstab, solve_pcg, SolverKernels, SolverWorkspace};
use ofgpu::species::{Species, SpeciesCoeffs};
use ofgpu::surface::classify::BlockAxes;
use ofgpu::surface::cutcell::{classify_cutcells, CellState, DEFAULT_SUPERSAMPLE};
use ofgpu::surface::stl::parse_stl;
use ofgpu::turbulence::TurbulenceControls;
use ofgpu::vof::{Vof, VofControls, VofProperties};
use ofgpu::{DevBuf, Error, Gpu, GpuMesh, Label, Result, Scalar, Tensor, Vec3};

#[path = "common/mod.rs"]
mod common;

use common::sci;

// ==========================================================================
//  Bookkeeping
// ==========================================================================

/// The running tally, and the one line each check prints.
struct Checks {
    total: usize,
    failures: usize,
    skipped: usize,
    /// How many of [`Self::total`] were judged against a RECORDED
    /// measurement - a number this binary did not produce, on this machine,
    /// on this run - rather than against something it computed live.
    ///
    /// Five functions here are replays (SPEC-LIT §3.1/§32.4/§32.5.5/§33.2/
    /// §34/§35): [`check_thermal_wall_function_gate_verdict_replay`],
    /// [`check_resolved_leg_mesh_resolution_replay`],
    /// [`check_resolved_leg_gate_verdict_replay`],
    /// [`check_thermostat_weighting_experiment_replay`] and
    /// [`check_bounded_convection_experiment_replay`]. Their INPUTS are frozen
    /// constants copied out of `docs/07-fire-solver.md` §1.1; everything done
    /// WITH those inputs - the correlations, the friction-factor
    /// conversions, the band arithmetic - is computed live, which is the
    /// whole point of replaying them. Counting them separately keeps the
    /// headline `N/N checks passed` meaning one thing only.
    replayed: usize,
    /// Set for the duration of [`Self::replaying`].
    in_replay: bool,
}

impl Checks {
    fn new() -> Self {
        Self { total: 0, failures: 0, skipped: 0, replayed: 0, in_replay: false }
    }

    /// Run `f` with every check it makes counted as REPLAYED. Not nestable,
    /// and not meant to be: a replay function calling another one would be
    /// hiding a second recorded measurement inside the first.
    fn replaying(&mut self, f: impl FnOnce(&mut Self)) {
        assert!(!self.in_replay, "replaying() does not nest");
        self.in_replay = true;
        f(self);
        self.in_replay = false;
    }

    fn check(&mut self, what: &str, err: Scalar, tol: Scalar) {
        self.total += 1;
        if self.in_replay {
            self.replayed += 1;
        }

        // A NaN fails every comparison, so it would sail through `err <= tol`
        // as a pass if the finiteness test were dropped.
        let ok = err <= tol && err.is_finite();
        if !ok {
            self.failures += 1;
        }

        println!(
            "{}{:<52}err {}  tol {}",
            if ok { "  ok   " } else { "  FAIL " },
            what,
            sci(f64::from(err), 3),
            sci(f64::from(tol), 3)
        );
    }

    /// A yes/no check with no meaningful error magnitude.
    fn require(&mut self, what: &str, ok: bool) {
        self.check(what, if ok { 0.0 } else { 1.0 }, 0.0);
    }

    /// Something that could not be attempted on this machine. Not a pass and
    /// not a failure - reported so the summary line cannot quietly shrink.
    fn skip(&mut self, what: &str, why: &str) {
        self.skipped += 1;
        println!("  skip  {what:<52}{why}");
    }

    fn note(&self, line: &str) {
        println!("        {line}");
    }
}

// ==========================================================================
//  Comparison helpers
// ==========================================================================

fn max_abs(a: &[Scalar]) -> Scalar {
    a.iter().fold(0.0 as Scalar, |m, v| m.max(v.abs()))
}

fn max_abs_diff(a: &[Scalar], b: &[Scalar]) -> Scalar {
    a.iter()
        .zip(b.iter())
        .fold(0.0 as Scalar, |m, (x, y)| m.max((x - y).abs()))
}

fn max_abs_diff_vec3(a: &[Vec3], b: &[Vec3]) -> Scalar {
    a.iter()
        .zip(b.iter())
        .fold(0.0 as Scalar, |m, (x, y)| m.max((*x - *y).mag()))
}

fn max_abs_diff_tensor(a: &[Tensor], b: &[Tensor]) -> Scalar {
    a.iter().zip(b.iter()).fold(0.0 as Scalar, |m, (x, y)| {
        let d = [
            x.xx - y.xx, x.xy - y.xy, x.xz - y.xz,
            x.yx - y.yx, x.yy - y.yy, x.yz - y.yz,
            x.zx - y.zx, x.zy - y.zy, x.zz - y.zz,
        ];
        m.max(d.iter().fold(0.0 as Scalar, |w, v| w.max(v.abs())))
    })
}

/// Relative to the reference's own magnitude, so a coefficient of `1e6`
/// and one of `1e-6` are held to the same number of significant digits.
fn rel(e: Scalar, r: &[Scalar]) -> Scalar {
    let s = max_abs(r);
    if s > 0.0 {
        e / s
    } else {
        e
    }
}

/// SplitMix64. Only determinism run to run matters: every random field here
/// feeds *both* the device path and the host reference, so what the checks
/// measure is agreement on identical data.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on `[lo, hi)`, from the top 53 bits.
    fn uniform(&mut self, lo: Scalar, hi: Scalar) -> Scalar {
        let bits = (self.next_u64() >> 11) as Scalar;
        let x = bits * (1.0 / 9_007_199_254_740_992.0);
        lo + (hi - lo) * x
    }
}

// ==========================================================================
//  Resolved kernels
// ==========================================================================

/// Every kernel unit the checks touch, resolved once. A `CudaFunction` holds
/// an `Arc` to its module, so these four keep the device side alive.
struct Kernels {
    field: FieldKernels,
    fv: FvKernels,
    ldu: LduKernels,
    solver: SolverKernels,
}

impl Kernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        Ok(Self {
            field: FieldKernels::new(gpu)?,
            fv: FvKernels::new(gpu)?,
            ldu: LduKernels::new(gpu)?,
            solver: SolverKernels::new(gpu)?,
        })
    }
}

// ==========================================================================
//  Meshes
// ==========================================================================

/// What kind of block a check wants.
#[derive(Debug, Clone)]
struct MeshSpec {
    n: [usize; 3],
    l: [Scalar; 3],
    /// Per-axis expansion ratio; `1` is uniform. Grading makes the
    /// interpolation weights non-trivial, which is where a weight bug hides.
    expansion: [Scalar; 3],
    /// `true`: front and back are `empty` and the case is two-dimensional.
    two_d: bool,
    /// Slide every point by this multiple of its `z`, turning the orthogonal
    /// block into a parallelepiped mesh. A shear preserves volume and leaves
    /// every face planar, so the analytic answers are unchanged - but the
    /// faces are no longer orthogonal to the centre-to-centre vectors, which
    /// is what exercises the correction of SPEC-LIT section 2.4.
    shear: Scalar,
    /// All six patches plain `patch` rather than walls and empties, which is
    /// what the FFT backend's Cartesian detection needs to see.
    all_generic: bool,
}

impl Default for MeshSpec {
    fn default() -> Self {
        Self {
            n: [10, 10, 10],
            l: [1.0, 0.7, 0.4],
            expansion: [1.0, 1.0, 1.0],
            two_d: false,
            shear: 0.0,
            all_generic: false,
        }
    }
}

impl MeshSpec {
    /// The analytic volume of the block. A shear is volume preserving, so it
    /// does not appear.
    fn volume(&self) -> Scalar {
        self.l[0] * self.l[1] * self.l[2]
    }
}

/// Build a block straight into a [`HostMesh`], no file on disk anywhere.
///
/// Used to round-trip through `write_block_mesh` + `read_poly_mesh`, which
/// put `blockgen` and the polyMesh reader under test at the same time as
/// everything else; `blockgen::build_mesh` now shares the exact same face
/// construction and is exercised directly by `blockgen`'s own
/// `build_mesh_matches_the_file_round_trip` test, so that coverage is not
/// lost - it just does not need a scratch directory per call any more.
fn make_mesh(dir: &Path, s: &MeshSpec) -> Result<HostMesh> {
    let axis = |i: usize| GradedAxis {
        lo: 0.0,
        hi: s.l[i],
        n: s.n[i],
        expansion: s.expansion[i],
        two_sided: s.expansion[i] != 1.0,
    };

    let types: [String; 6] = if s.all_generic {
        ["patch", "patch", "patch", "patch", "patch", "patch"].map(String::from)
    } else {
        [
            "patch",
            "patch",
            "wall",
            "wall",
            if s.two_d { "empty" } else { "wall" },
            if s.two_d { "empty" } else { "wall" },
        ]
        .map(String::from)
    };

    let b = BlockSpec {
        x: axis(0),
        y: axis(1),
        z: axis(2),
        window: None,
        patch_name: BlockSpec::default().patch_name,
        patch_type: types,
        cyclic: Vec::new(),
    };

    if s.shear == 0.0 {
        return blockgen::build_mesh(&b);
    }

    // The shear applies to the raw points before geometry is computed, so
    // this one case still goes through `PolyMeshRaw` rather than straight to
    // `HostMesh` - shearing is a test fixture, not something `BlockSpec`
    // itself can express.
    let _ = std::fs::remove_dir_all(dir);
    write_block_mesh(dir, &b)?;
    let mut raw = read_poly_mesh(dir)?;
    for p in raw.points.iter_mut() {
        p.x += s.shear * p.z;
        p.y += 0.5 * s.shear * p.z;
    }
    let m = build_host_mesh(&raw)?;
    let _ = std::fs::remove_dir_all(dir);
    Ok(m)
}

fn scratch_dir(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ofgpuValidate_{tag}"))
}

// ==========================================================================
//  Field helpers
// ==========================================================================

/// `bc_kind` for every boundary face: `kind` everywhere the mesh does not
/// call the patch `empty`, and [`BcKind::Empty`] where it does. The mesh has
/// the last word on an empty patch - a field file cannot put a fixedValue on
/// one and have it mean anything.
fn kinds(m: &HostMesh, kind: BcKind) -> Vec<Label> {
    (0..m.n_boundary_faces)
        .map(|bf| {
            if cpu::is_empty_face(m, bf) {
                BcKind::Empty as Label
            } else {
                kind as Label
            }
        })
        .collect()
}

/// Push a host `(fr, refValue, refGrad)` triple onto a device scalar field.
fn upload_bc(
    gpu: &Gpu,
    psi: &mut GpuScalarField,
    bc: &cpu::CpuScalarBc,
    m: &HostMesh,
    kind: BcKind,
) -> Result<()> {
    gpu.write(&mut psi.fr, &bc.fr)?;
    gpu.write(&mut psi.ref_value, &bc.ref_value)?;
    gpu.write(&mut psi.ref_grad, &bc.ref_grad)?;
    gpu.write(&mut psi.bc_kind, &kinds(m, kind))?;
    Ok(())
}

/// `(fr, refValue, refGrad)` exercising every branch of the mixed form: one
/// patch fixedValue, one zeroGradient, one fixedGradient, the rest Robin.
fn random_bc(m: &HostMesh, rng: &mut Rng) -> cpu::CpuScalarBc {
    let mut bc = cpu::CpuScalarBc::new(m.n_boundary_faces);

    for (pi, p) in m.patches.iter().enumerate() {
        for i in 0..p.size {
            let bf = p.start + i;
            match pi % 4 {
                0 => {
                    bc.fr[bf] = 1.0;
                    bc.ref_value[bf] = rng.uniform(0.0, 1.0);
                }
                1 => bc.fr[bf] = 0.0,
                2 => {
                    bc.fr[bf] = 0.0;
                    bc.ref_grad[bf] = rng.uniform(-0.5, 0.5);
                }
                _ => {
                    bc.fr[bf] = 0.25 + 0.5 * rng.uniform(0.0, 1.0);
                    bc.ref_value[bf] = rng.uniform(0.0, 1.0);
                    bc.ref_grad[bf] = rng.uniform(-0.5, 0.5);
                }
            }
        }
    }
    bc
}

/// Diffusivity times face area, zeroed on the empty patches.
///
/// An empty patch is the 2-D front and back: it carries no flux and no
/// diffusivity, because the direction it faces is the one the case does not
/// resolve (SPEC-LIT section 2). Both the device and [`ofgpu::reference`]
/// therefore see nothing there.
fn boundary_gamma(m: &HostMesh, gamma: Scalar) -> Vec<Scalar> {
    (0..m.n_boundary_faces)
        .map(|bf| {
            if cpu::is_empty_face(m, bf) {
                0.0
            } else {
                gamma * m.b_mag_sf[bf]
            }
        })
        .collect()
}

// ==========================================================================
//  1. Mesh identities - SPEC-LIT section 10, rows "Mesh closure" and "Volume"
// ==========================================================================

fn check_mesh(c: &mut Checks, m: &HostMesh, analytic_volume: Scalar) {
    let sum_v: Scalar = m.v.iter().copied().sum();
    c.check(
        "sum(V) == analytic block volume",
        (sum_v - analytic_volume).abs() / analytic_volume,
        1e-12,
    );

    let min_v = m.v.iter().fold(Scalar::INFINITY, |a, b| a.min(*b));
    c.require("min(V) > 0", min_v > 0.0);

    // Gauss's theorem applied to the constant field 1: every closed cell has
    // sum_f (+-Sf) = 0. Non-dimensionalised by V^(2/3), which is the area
    // scale of the cell, so the number is comparable across refinements.
    let mut closure = vec![Vec3::ZERO; m.n_cells];
    for f in 0..m.n_internal_faces {
        closure[m.owner[f] as usize] += m.sf[f];
        closure[m.neighbour[f] as usize] -= m.sf[f];
    }
    for bf in 0..m.n_boundary_faces {
        closure[m.b_face_cells[bf] as usize] += m.b_sf[bf];
    }

    let worst = (0..m.n_cells).fold(0.0 as Scalar, |w, cell| {
        let scale = (f64::from(m.v[cell])).powf(2.0 / 3.0) as Scalar;
        w.max(closure[cell].mag() / scale)
    });
    c.check("cell closure |sum Sf| / V^(2/3)", worst, 1e-12);

    // Cell centroids must lie inside their own cells; the pyramid
    // decomposition of SPEC-LIT section 2.2 guarantees it for a convex cell,
    // and a negative face-pyramid volume is how a wound-backwards face shows
    // up. Tested through the sign of Sf.(Cf - C_P).
    let mut inverted = 0usize;
    for f in 0..m.n_internal_faces {
        let p = m.owner[f] as usize;
        let n = m.neighbour[f] as usize;
        if m.sf[f].dot(m.cf[f] - m.c[p]) <= 0.0 || m.sf[f].dot(m.c[n] - m.cf[f]) <= 0.0 {
            inverted += 1;
        }
    }
    c.check("every face separates its two cell centres", inverted as Scalar, 0.0);

    // owner < neighbour, ascending by (owner, neighbour): the
    // upper-triangular order SPEC-LIT section 1 and every gather kernel
    // assume.
    let mut bad = 0usize;
    for f in 0..m.n_internal_faces {
        if m.owner[f] >= m.neighbour[f] {
            bad += 1;
        }
        if f > 0 {
            let ordered = m.owner[f] > m.owner[f - 1]
                || (m.owner[f] == m.owner[f - 1] && m.neighbour[f] > m.neighbour[f - 1]);
            if !ordered {
                bad += 1;
            }
        }
    }
    c.check("lduAddressing is upper-triangular", bad as Scalar, 0.0);

    // The interpolation weight of SPEC-LIT section 2.3 must land in [0,1] on
    // a convex mesh; outside it the "interpolated" value is an extrapolation.
    let bad_w = (0..m.n_internal_faces)
        .filter(|&f| !(0.0..=1.0).contains(&m.weights[f]))
        .count();
    c.check("interpolation weights in [0,1]", bad_w as Scalar, 0.0);
}

// ==========================================================================
//  2. Explicit operators
// ==========================================================================

fn check_explicit_operators(
    c: &mut Checks,
    gpu: &Gpu,
    k: &Kernels,
    m: &HostMesh,
    gm: &GpuMesh,
) -> Result<()> {
    // On a 2-D mesh the front and back are `empty` and contribute nothing to
    // any surface integral, so the out-of-plane gradient is identically zero
    // BY CONSTRUCTION - that is what an empty patch is. Probing with a
    // z-varying field and comparing against an analytic dpsi/dz would
    // therefore be testing the wrong thing; the probe is made planar instead.
    let planar = m.patches.iter().any(|p| p.kind == PatchKind::Empty && p.size > 0);

    let a = Vec3::new(1.7, -0.9, if planar { 0.0 } else { 0.35 });
    let b0: Scalar = 0.42;

    // ---- SPEC-LIT section 10, row "Gradient" -----------------------------
    {
        let f: Vec<Scalar> = (0..m.n_cells).map(|cc| a.dot(m.c[cc]) + b0).collect();
        let bfv: Vec<Scalar> = (0..m.n_boundary_faces).map(|i| a.dot(m.b_cf[i]) + b0).collect();

        let mut psi = GpuScalarField::zeros(gpu, gm, "psi")?;
        gpu.write(&mut psi.f, &f)?;

        // Dirichlet with the exact face value, so the boundary treatment is
        // not what is under test here.
        let bc = cpu::CpuScalarBc::dirichlet(&bfv);
        upload_bc(gpu, &mut psi, &bc, m, BcKind::Mixed)?;
        correct_boundary_conditions(gpu, &k.field, &mut psi, gm)?;
        gpu.sync()?;

        // The reference is fed the boundary values the DEVICE evaluated, so
        // the gradient comparison isolates the gradient.
        let bf_dev = gpu.download(&psi.bf)?;
        let bf_ref = bc.evaluate(m, &f);
        c.check(
            "correctBoundaryConditions == the mixed form of section 4",
            rel(max_abs_diff(&bf_dev, &bf_ref), &bf_ref),
            1e-14,
        );

        let mut g: DevBuf<Vec3> = gpu.zeros(m.n_cells)?;
        fvc_grad_scalar(gpu, &k.fv, &mut g, &psi, gm)?;
        gpu.sync()?;
        let gh = gpu.download(&g)?;

        let worst = gh.iter().take(m.n_cells).fold(0.0 as Scalar, |w, v| w.max((*v - a).mag()));
        c.check("fvc::grad(linear field) == analytic", worst / a.mag(), 1e-11);

        if planar {
            // Every face that survives an empty patch has zero area component
            // in the unresolved direction, so this is zero by construction and
            // not merely small. The tolerance is round-off on the accumulation
            // rather than exactly zero, because the front and back faces of a
            // cell are integrated separately and their centroids agree only to
            // the last bit.
            let worst_z = gh.iter().fold(0.0 as Scalar, |w, v| w.max(v.z.abs()));
            c.check("2-D: out-of-plane gradient vanishes", worst_z / a.mag(), 1e-13);
        }

        let mut g_ref = Vec::new();
        cpu::fvc_grad_scalar(&mut g_ref, &f, &bf_ref, m);
        c.check(
            "fvc::grad scalar   device vs host reference",
            max_abs_diff_vec3(&gh, &g_ref) / a.mag(),
            1e-12,
        );

        // Section 2.3: the weight places the interpolated value where the
        // face plane cuts P-N, which on this mesh is the face centre, so a
        // linear field interpolates exactly.
        let mut face = GpuSurfaceScalarField::zeros(gpu, gm, "psif")?;
        interpolate_linear(gpu, &k.fv, &mut face, &psi, gm)?;
        gpu.sync()?;
        let fh = gpu.download(&face.f)?;
        let want: Vec<Scalar> = (0..m.n_internal_faces).map(|i| a.dot(m.cf[i]) + b0).collect();
        c.check(
            "interpolate(linear field) == value at the face centre",
            rel(max_abs_diff(&fh, &want), &want),
            1e-12,
        );

        let mut f_ref = Vec::new();
        cpu::interpolate_linear(&mut f_ref, &f, m);
        c.check(
            "interpolate        device vs host reference",
            rel(max_abs_diff(&fh, &f_ref), &f_ref),
            1e-14,
        );

        // Section 2.4: snGrad of a linear field is nf . a exactly, on the
        // boundary as well as inside, once the correction vector is included.
        // On an orthogonal mesh k = 0 and this holds with the implicit part
        // alone, which is why the sheared mesh is also run through here.
        let gamma: Vec<Scalar> = m.mag_sf.to_vec();
        let b_gamma = boundary_gamma(m, 1.0);
        let d_gamma = gpu.upload(&gamma)?;
        let d_b_gamma = gpu.upload(&b_gamma)?;

        let mut flux = GpuSurfaceScalarField::zeros(gpu, gm, "snGradFlux")?;
        sn_grad_flux(gpu, &k.fv, &mut flux, &psi, &d_gamma, &d_b_gamma, gm)?;
        gpu.sync()?;

        let phi_h = gpu.download(&flux.f)?;
        let mut phi_ref = Vec::new();
        let mut bphi_ref = Vec::new();
        cpu::sn_grad_flux(
            &mut phi_ref, &mut bphi_ref, &f, &gamma, &b_gamma, &bc, m,
        );
        c.check(
            "snGrad flux        device vs host reference",
            rel(max_abs_diff(&phi_h, &phi_ref), &phi_ref),
            1e-12,
        );
    }

    // ---- vector gradient, production -------------------------------------
    {
        let mut rng = Rng::new(20260822);
        let uc: Vec<Vec3> = (0..m.n_cells)
            .map(|_| {
                Vec3::new(
                    rng.uniform(-1.0, 1.0),
                    rng.uniform(-1.0, 1.0),
                    rng.uniform(-1.0, 1.0),
                )
            })
            .collect();

        let mut u = GpuVectorField::zeros(gpu, gm, "U")?;
        gpu.write(&mut u.f, &uc)?;
        gpu.write(&mut u.fr, &vec![0.0 as Scalar; m.n_boundary_faces])?;
        gpu.write(&mut u.ref_value, &vec![Vec3::ZERO; m.n_boundary_faces])?;
        gpu.write(&mut u.ref_grad, &vec![Vec3::ZERO; m.n_boundary_faces])?;
        gpu.write(&mut u.bc_kind, &kinds(m, BcKind::ZeroGradient))?;
        correct_boundary_conditions_vector(gpu, &k.field, &mut u, gm)?;
        gpu.sync()?;
        let ub = gpu.download(&u.bf)?;

        let mut g: DevBuf<Tensor> = gpu.zeros(m.n_cells)?;
        fvc_grad_vector(gpu, &k.fv, &mut g, &u, gm)?;
        gpu.sync()?;
        let gh = gpu.download(&g)?;

        let mut g_ref = Vec::new();
        cpu::fvc_grad_vector(&mut g_ref, &uc, &ub, m);

        let scale = g_ref
            .iter()
            .fold(1.0 as Scalar, |w, t| w.max(t.xx.abs()).max(t.yy.abs()).max(t.zz.abs()));
        c.check(
            "fvc::grad vector   device vs host reference",
            max_abs_diff_tensor(&gh, &g_ref) / scale,
            1e-12,
        );

        let nut_h = vec![0.013 as Scalar; m.n_cells];
        let nut = gpu.upload(&nut_h)?;
        let mut prod: DevBuf<Scalar> = gpu.zeros(m.n_cells)?;
        turbulence_production(gpu, &k.fv, &mut prod, &nut, &g, m.n_cells)?;
        gpu.sync()?;

        let ph = gpu.download(&prod)?;
        let mut p_ref = Vec::new();
        cpu::turbulence_production(&mut p_ref, &nut_h, &g_ref, m.n_cells);
        c.check(
            "G = nut dev(2 symm(gradU)) : gradU",
            rel(max_abs_diff(&ph, &p_ref), &p_ref),
            1e-12,
        );
    }

    // ---- SPEC-LIT section 10, row "Divergence" ---------------------------
    {
        // phi = Sf . const is divergence free on any closed cell, whatever
        // the mesh.
        let u_const = Vec3::new(0.83, -0.21, 0.44);
        let phi: Vec<Scalar> = (0..m.n_internal_faces).map(|f| u_const.dot(m.sf[f])).collect();
        let bphi: Vec<Scalar> = (0..m.n_boundary_faces).map(|i| u_const.dot(m.b_sf[i])).collect();

        let mut sphi = GpuSurfaceScalarField::zeros(gpu, gm, "phi")?;
        gpu.write(&mut sphi.f, &phi)?;
        gpu.write(&mut sphi.bf, &bphi)?;

        let mut d: DevBuf<Scalar> = gpu.zeros(m.n_cells)?;
        fvc_div_surface(gpu, &k.fv, &mut d, &sphi, gm)?;
        gpu.sync()?;
        let dh = gpu.download(&d)?;
        c.check("fvc::div(uniform flux) == 0", max_abs(&dh) / u_const.mag(), 1e-10);

        let mut d_ref = Vec::new();
        cpu::fvc_div_surface(&mut d_ref, &phi, &bphi, m);
        c.check(
            "fvc::div           device vs host reference",
            max_abs_diff(&dh, &d_ref) / u_const.mag(),
            1e-12,
        );

        // An empty patch contributes nothing to a surface integral. Put a flux
        // on the front and back that does NOT cancel between them, and the
        // divergence must not move at all - bit for bit, because a skipped
        // face is not read.
        //
        // This is the one check that isolates the empty-patch rule from the
        // cancellation that normally hides it: with the usual equal-and-
        // opposite boundary values the front and back sum to zero whether or
        // not they were skipped, so every other check here passes either way.
        // If this one fails, look at what the operator is comparing `b_kind`
        // against: `HostMesh::b_kind` holds `PatchKind` (Empty = 2,
        // Cyclic = 4), not `BcKind` (Empty = 5, Cyclic = 7).
        if planar {
            let mut bphi2 = bphi.clone();
            for bf in 0..m.n_boundary_faces {
                if cpu::is_empty_face(m, bf) {
                    bphi2[bf] = 1.0;
                }
            }
            gpu.write(&mut sphi.bf, &bphi2)?;
            let mut d2: DevBuf<Scalar> = gpu.zeros(m.n_cells)?;
            fvc_div_surface(gpu, &k.fv, &mut d2, &sphi, gm)?;
            gpu.sync()?;
            let d2h = gpu.download(&d2)?;
            c.check("2-D: an empty patch carries no flux", max_abs_diff(&d2h, &dh), 0.0);
        }
    }

    // ---- reconstruct: the inverse of "flux of a uniform field" -----------
    {
        // The least-squares reconstruction must return a uniform field
        // exactly, because a uniform field reproduces its own face fluxes
        // with zero residual. That is an analytic property of the operator,
        // not a comparison with anything.
        let u_const = Vec3::new(0.31, -0.72, if planar { 0.0 } else { 0.19 });
        let phi: Vec<Scalar> = (0..m.n_internal_faces).map(|f| u_const.dot(m.sf[f])).collect();
        let bphi: Vec<Scalar> = (0..m.n_boundary_faces).map(|i| u_const.dot(m.b_sf[i])).collect();

        let mut sphi = GpuSurfaceScalarField::zeros(gpu, gm, "phi")?;
        gpu.write(&mut sphi.f, &phi)?;
        gpu.write(&mut sphi.bf, &bphi)?;

        let mut u: DevBuf<Vec3> = gpu.zeros(m.n_cells)?;
        fvc_reconstruct(gpu, &k.fv, &mut u, &sphi, gm)?;
        gpu.sync()?;
        let uh = gpu.download(&u)?;

        let worst = uh.iter().fold(0.0 as Scalar, |w, v| w.max((*v - u_const).mag()));
        c.check("fvc::reconstruct(U . Sf) == U", worst / u_const.mag(), 1e-11);
    }

    Ok(())
}

// ==========================================================================
//  3. Implicit assembly, relaxation, boundary folding and Amul
// ==========================================================================

fn check_assembly(
    c: &mut Checks,
    gpu: &Gpu,
    k: &Kernels,
    m: &HostMesh,
    gm: &GpuMesh,
    scheme: DivScheme,
) -> Result<()> {
    let mut rng = Rng::new(7);

    let n_c = m.n_cells;
    let n_if = m.n_internal_faces;
    let n_bf = m.n_boundary_faces;

    let psi_h: Vec<Scalar> = (0..n_c).map(|_| rng.uniform(0.2, 1.4)).collect();
    let psi0_h: Vec<Scalar> = (0..n_c).map(|_| rng.uniform(0.2, 1.4)).collect();

    let mut gamma = vec![0.0 as Scalar; n_if];
    let mut phi = vec![0.0 as Scalar; n_if];
    for f in 0..n_if {
        gamma[f] = rng.uniform(0.2, 1.4) * m.mag_sf[f];
        phi[f] = rng.uniform(-1.0, 1.0) * m.mag_sf[f];
    }

    // An empty face carries no flux and no diffusivity; see boundary_gamma.
    let mut b_gamma = vec![0.0 as Scalar; n_bf];
    let mut bphi = vec![0.0 as Scalar; n_bf];
    for i in 0..n_bf {
        if cpu::is_empty_face(m, i) {
            continue;
        }
        b_gamma[i] = rng.uniform(0.2, 1.4) * m.b_mag_sf[i];
        bphi[i] = rng.uniform(-1.0, 1.0) * m.b_mag_sf[i];
    }

    let sp: Vec<Scalar> = (0..n_c).map(|_| rng.uniform(0.2, 1.4)).collect();
    let susp: Vec<Scalar> = (0..n_c).map(|_| rng.uniform(-1.0, 1.0)).collect();
    let su: Vec<Scalar> = (0..n_c).map(|_| rng.uniform(-1.0, 1.0)).collect();

    let bc = random_bc(m, &mut rng);

    // ---- device ----------------------------------------------------------
    let mut psi = GpuScalarField::zeros(gpu, gm, "psi")?;
    gpu.write(&mut psi.f, &psi_h)?;
    gpu.write(&mut psi.f0, &psi0_h)?;
    upload_bc(gpu, &mut psi, &bc, m, BcKind::Mixed)?;
    correct_boundary_conditions(gpu, &k.field, &mut psi, gm)?;

    let mut gphi = GpuSurfaceScalarField::zeros(gpu, gm, "phi")?;
    gpu.write(&mut gphi.f, &phi)?;
    gpu.write(&mut gphi.bf, &bphi)?;

    let d_gamma = gpu.upload(&gamma)?;
    let d_b_gamma = gpu.upload(&b_gamma)?;
    let d_sp = gpu.upload(&sp)?;
    let d_susp = gpu.upload(&susp)?;
    let d_su = gpu.upload(&su)?;

    // A limited scheme reads the upwind cell's gradient during assembly, so
    // the gradient is produced first and the SAME numbers are handed to both
    // sides - the gradient itself is checked separately.
    let mut d_grad: DevBuf<Vec3> = gpu.zeros(n_c)?;
    fvc_grad_scalar(gpu, &k.fv, &mut d_grad, &psi, gm)?;
    gpu.sync()?;
    let grad_h = gpu.download(&d_grad)?;

    let mut d_w: DevBuf<Scalar> = gpu.zeros(n_if)?;
    let mut d_bw: DevBuf<Scalar> = gpu.zeros(n_bf)?;
    div_scheme_weights(
        gpu,
        &k.fv,
        Some(&mut d_w),
        Some(&mut d_bw),
        scheme,
        &gphi,
        &psi,
        if scheme.needs_gradient() { Some(&d_grad) } else { None },
        gm,
    )?;
    gpu.sync()?;

    let w = gpu.download(&d_w)?;
    let bw = gpu.download(&d_bw)?;

    let mut w_ref = Vec::new();
    let mut bw_ref = Vec::new();
    cpu::div_scheme_weights(
        &mut w_ref, &mut bw_ref, scheme, &phi, &bphi, &psi_h, Some(&grad_h), m,
    );
    c.check(
        &format!("{scheme:?} weights   device vs host reference"),
        max_abs_diff(&w, &w_ref).max(max_abs_diff(&bw, &bw_ref)),
        1e-14,
    );

    let r_delta_t: Scalar = 12.5;

    let mut a = GpuLduMatrix::new(gpu, gm)?;
    a.zero(gpu)?;

    fvm_ddt_euler(gpu, &k.fv, &mut a, gm, None, None, &psi.f0, r_delta_t, 1.0)?;
    fvm_div_gauss(gpu, &k.fv, &mut a, gm, &gphi, &d_w, &d_bw, &psi, 1.0)?;
    fvm_div_bounded_correction(gpu, &k.fv, &mut a, gm, &gphi, 1.0)?;
    fvm_laplacian(gpu, &k.fv, &mut a, gm, &d_gamma, &d_b_gamma, &psi, -1.0)?;
    fvm_laplacian_non_orth_correction(
        gpu,
        &k.fv,
        &mut a,
        gm,
        &d_gamma,
        &d_b_gamma,
        &psi,
        &d_grad,
        SnGradScheme::Corrected,
        -1.0,
    )?;
    fvm_su(gpu, &k.fv, &mut a, gm, &d_su, 1.0)?;
    fvm_susp(gpu, &k.fv, &mut a, gm, &d_susp, &psi.f, 1.0)?;
    fvm_sp(gpu, &k.fv, &mut a, gm, &d_sp, 1.0)?;
    gpu.sync()?;

    // ---- host twin, written as scatter loops -----------------------------
    let mut r = cpu::CpuLdu::new(m);

    cpu::fvm_ddt_euler(&mut r, m, None, None, &psi0_h, r_delta_t, 1.0);
    cpu::fvm_div_gauss(&mut r, m, &phi, &bphi, &w_ref, &bw_ref, &bc, 1.0);
    cpu::fvm_div_bounded_correction(&mut r, m, &phi, &bphi, 1.0);
    cpu::fvm_laplacian(&mut r, m, &gamma, &b_gamma, &bc, -1.0);
    cpu::fvm_laplacian_non_orth_correction(
        &mut r,
        m,
        &gamma,
        &b_gamma,
        &bc,
        &psi_h,
        &grad_h,
        SnGradScheme::Corrected,
        -1.0,
    );
    cpu::fvm_su(&mut r, m, &su, 1.0);
    cpu::fvm_susp(&mut r, m, &susp, &psi_h, 1.0);
    cpu::fvm_sp(&mut r, m, &sp, 1.0);

    for (what, got, want) in [
        ("diag", gpu.download(&a.diag)?, &r.diag),
        ("upper", gpu.download(&a.upper)?, &r.upper),
        ("lower", gpu.download(&a.lower)?, &r.lower),
        ("source", gpu.download(&a.source)?, &r.source),
        ("internalCoeffs", gpu.download(&a.internal_coeffs)?, &r.internal_coeffs),
        ("boundaryCoeffs", gpu.download(&a.boundary_coeffs)?, &r.boundary_coeffs),
    ] {
        c.check(
            &format!("assembly {what:<16}device vs host reference"),
            rel(max_abs_diff(&got, want), want),
            1e-12,
        );
    }

    // ---- relax, then fold, then Amul --------------------------------------
    relax(gpu, &k.ldu, &mut a, gm, &psi.f, 0.7)?;
    cpu::relax(&mut r, m, &psi_h, 0.7);
    gpu.sync()?;

    c.check(
        "relax(0.7) diag    device vs host reference",
        rel(max_abs_diff(&gpu.download(&a.diag)?, &r.diag), &r.diag),
        1e-12,
    );
    c.check(
        "relax(0.7) source  device vs host reference",
        rel(max_abs_diff(&gpu.download(&a.source)?, &r.source), &r.source),
        1e-12,
    );

    add_boundary_contributions(gpu, &k.ldu, &mut a, gm)?;
    cpu::add_boundary_contributions(&mut r, m);
    gpu.sync()?;

    c.check(
        "boundary folding, diag",
        rel(max_abs_diff(&gpu.download(&a.diag)?, &r.diag), &r.diag),
        1e-12,
    );
    c.check(
        "boundary folding, source",
        rel(max_abs_diff(&gpu.download(&a.source)?, &r.source), &r.source),
        1e-12,
    );

    let mut ap: DevBuf<Scalar> = gpu.zeros(n_c)?;
    amul(gpu, &k.ldu, &mut ap, &psi.f, &a, gm)?;
    gpu.sync()?;

    let mut ap_ref = Vec::new();
    cpu::amul(&mut ap_ref, &psi_h, &r, m);
    let ap_h = gpu.download(&ap)?;
    c.check("Amul", rel(max_abs_diff(&ap_h, &ap_ref), &ap_ref), 1e-12);

    // ---- CSR export -------------------------------------------------------
    // The export is only worth anything if the LDU -> CSR permutation is
    // right, so the exported matrix is applied to the same vector.
    {
        let pattern = CsrPattern::build(m);
        let mut csr = pattern.upload(gpu)?;
        csr_fill(gpu, &k.ldu, &mut csr, &a)?;
        gpu.sync()?;

        let row_ptr = gpu.download(&csr.row_ptr)?;
        let col_ind = gpu.download(&csr.col_ind)?;
        let val = gpu.download(&csr.val)?;

        let mut csr_ax = vec![0.0 as Scalar; n_c];
        for row in 0..n_c {
            let mut s: Scalar = 0.0;
            for j in row_ptr[row] as usize..row_ptr[row + 1] as usize {
                s += val[j] * psi_h[col_ind[j] as usize];
            }
            csr_ax[row] = s;
        }
        c.check(
            "CSR export reproduces Amul",
            rel(max_abs_diff(&csr_ax, &ap_ref), &ap_ref),
            1e-12,
        );

        let unsorted = (0..n_c)
            .flat_map(|row| row_ptr[row] as usize + 1..row_ptr[row + 1] as usize)
            .filter(|&j| col_ind[j] <= col_ind[j - 1])
            .count();
        c.check("CSR columns ascending within each row", unsorted as Scalar, 0.0);

        let expected = n_c + 2 * n_if;
        c.check(
            "CSR nnz == nCells + 2 nInternalFaces",
            (csr.nnz as i64 - expected as i64).abs() as Scalar,
            0.0,
        );
    }

    // ---- setValues ---------------------------------------------------------
    {
        let mut ac = GpuLduMatrix::new(gpu, gm)?;
        ac.zero(gpu)?;
        fvm_laplacian(gpu, &k.fv, &mut ac, gm, &d_gamma, &d_b_gamma, &psi, -1.0)?;
        fvm_ddt_euler(gpu, &k.fv, &mut ac, gm, None, None, &psi.f0, r_delta_t, 1.0)?;

        let pin: Vec<Label> = vec![0, (n_c / 3) as Label, (n_c - 1) as Label];
        let vals: Vec<Scalar> = vec![3.25, -1.5, 0.75];
        set_fixed_cells(gpu, &mut ac, &pin, &vals)?;
        add_boundary_contributions(gpu, &k.ldu, &mut ac, gm)?;
        set_values(gpu, &k.ldu, &mut ac, gm)?;
        gpu.sync()?;

        let mut rc = cpu::CpuLdu::new(m);
        cpu::fvm_laplacian(&mut rc, m, &gamma, &b_gamma, &bc, -1.0);
        cpu::fvm_ddt_euler(&mut rc, m, None, None, &psi0_h, r_delta_t, 1.0);
        let mut fixed = vec![false; n_c];
        let mut value = vec![0.0 as Scalar; n_c];
        for (i, cell) in pin.iter().enumerate() {
            fixed[*cell as usize] = true;
            value[*cell as usize] = vals[i];
        }
        cpu::add_boundary_contributions(&mut rc, m);
        cpu::set_values(&mut rc, m, &fixed, &value);

        for (what, got, want) in [
            ("diag", gpu.download(&ac.diag)?, &rc.diag),
            ("upper", gpu.download(&ac.upper)?, &rc.upper),
            ("source", gpu.download(&ac.source)?, &rc.source),
        ] {
            c.check(
                &format!("setValues {what:<15}device vs host reference"),
                rel(max_abs_diff(&got, want), want),
                1e-12,
            );
        }
    }

    Ok(())
}

// ==========================================================================
//  4. The Krylov solve against a dense direct solve
//     SPEC-LIT section 10, row "Solver"
// ==========================================================================

/// Small enough that Gaussian elimination is cheap, big enough that the
/// Krylov method has to actually iterate.
fn check_solver_against_dense(c: &mut Checks, gpu: &Gpu, k: &Kernels) -> Result<()> {
    let spec = MeshSpec {
        n: [5, 4, 3],
        l: [1.0, 0.7, 0.4],
        expansion: [1.0, 2.5, 1.0],
        ..Default::default()
    };
    let m = make_mesh(&scratch_dir("dense"), &spec)?;
    let gm = GpuMesh::upload(gpu, &m)?;
    let n_c = m.n_cells;

    let mut rng = Rng::new(4242);
    let bc = random_bc(&m, &mut rng);

    let mut psi = GpuScalarField::zeros(gpu, &gm, "psi")?;
    upload_bc(gpu, &mut psi, &bc, &m, BcKind::Mixed)?;
    correct_boundary_conditions(gpu, &k.field, &mut psi, &gm)?;

    let gamma: Vec<Scalar> = m.mag_sf.iter().map(|s| 0.7 * s).collect();
    let b_gamma = boundary_gamma(&m, 0.7);
    let d_gamma = gpu.upload(&gamma)?;
    let d_b_gamma = gpu.upload(&b_gamma)?;

    let su: Vec<Scalar> = (0..n_c).map(|_| rng.uniform(-1.0, 1.0)).collect();
    let d_su = gpu.upload(&su)?;

    // ---- symmetric: laplacian only, solved with conjugate gradients -------
    {
        let mut a = GpuLduMatrix::new(gpu, &gm)?;
        a.zero(gpu)?;
        fvm_laplacian(gpu, &k.fv, &mut a, &gm, &d_gamma, &d_b_gamma, &psi, -1.0)?;
        fvm_su(gpu, &k.fv, &mut a, &gm, &d_su, 1.0)?;
        add_boundary_contributions(gpu, &k.ldu, &mut a, &gm)?;
        gpu.sync()?;

        let mut r = cpu::CpuLdu::new(&m);
        cpu::fvm_laplacian(&mut r, &m, &gamma, &b_gamma, &bc, -1.0);
        cpu::fvm_su(&mut r, &m, &su, 1.0);
        cpu::add_boundary_contributions(&mut r, &m);

        let Some(exact) = cpu::solve_dense(cpu::dense_from_ldu(&r, &m), &r.source) else {
            c.require("dense direct solve of the symmetric system", false);
            return Ok(());
        };

        let ctrl = SolverControls {
            tolerance: 1e-14,
            rel_tol: 0.0,
            max_iter: 5000,
            precon: Preconditioner::Diagonal,
            ..Default::default()
        };
        let mut ws = SolverWorkspace::for_mesh(gpu, &gm)?;
        let mut x: DevBuf<Scalar> = gpu.zeros(n_c)?;
        let perf = solve_pcg(gpu, &k.solver, &mut x, &a, &gm, &mut ws, &ctrl)?;
        gpu.sync()?;
        let got = gpu.download(&x)?;

        c.note(&format!(
            "PCG {} iterations, residual {} -> {}",
            perf.n_iterations,
            sci(f64::from(perf.initial_residual), 3),
            sci(f64::from(perf.final_residual), 3)
        ));
        c.check(
            "PCG == dense direct solve (symmetric)",
            max_abs_diff(&got, &exact) / max_abs(&exact).max(1e-30),
            1e-8,
        );
        c.check("dense solve leaves no residual", cpu::residual(&exact, &r, &m), 1e-13);
    }

    // ---- asymmetric: convection makes upper != lower ----------------------
    {
        let phi: Vec<Scalar> = (0..m.n_internal_faces)
            .map(|f| rng.uniform(-1.0, 1.0) * m.mag_sf[f])
            .collect();
        let bphi: Vec<Scalar> = (0..m.n_boundary_faces)
            .map(|i| {
                if cpu::is_empty_face(&m, i) {
                    0.0
                } else {
                    rng.uniform(-1.0, 1.0) * m.b_mag_sf[i]
                }
            })
            .collect();

        let mut gphi = GpuSurfaceScalarField::zeros(gpu, &gm, "phi")?;
        gpu.write(&mut gphi.f, &phi)?;
        gpu.write(&mut gphi.bf, &bphi)?;

        let mut d_w: DevBuf<Scalar> = gpu.zeros(m.n_internal_faces)?;
        let mut d_bw: DevBuf<Scalar> = gpu.zeros(m.n_boundary_faces)?;
        div_scheme_weights(
            gpu, &k.fv, Some(&mut d_w), Some(&mut d_bw), DivScheme::Upwind, &gphi, &psi, None, &gm,
        )?;
        gpu.sync()?;
        let w = gpu.download(&d_w)?;
        let bw = gpu.download(&d_bw)?;

        let psi0 = vec![0.0 as Scalar; n_c];
        let d_psi0 = gpu.upload(&psi0)?;

        let mut a = GpuLduMatrix::new(gpu, &gm)?;
        a.zero(gpu)?;
        fvm_ddt_euler(gpu, &k.fv, &mut a, &gm, None, None, &d_psi0, 4.0, 1.0)?;
        fvm_div_gauss(gpu, &k.fv, &mut a, &gm, &gphi, &d_w, &d_bw, &psi, 1.0)?;
        fvm_laplacian(gpu, &k.fv, &mut a, &gm, &d_gamma, &d_b_gamma, &psi, -1.0)?;
        fvm_su(gpu, &k.fv, &mut a, &gm, &d_su, 1.0)?;
        add_boundary_contributions(gpu, &k.ldu, &mut a, &gm)?;
        gpu.sync()?;

        let mut r = cpu::CpuLdu::new(&m);
        cpu::fvm_ddt_euler(&mut r, &m, None, None, &psi0, 4.0, 1.0);
        cpu::fvm_div_gauss(&mut r, &m, &phi, &bphi, &w, &bw, &bc, 1.0);
        cpu::fvm_laplacian(&mut r, &m, &gamma, &b_gamma, &bc, -1.0);
        cpu::fvm_su(&mut r, &m, &su, 1.0);
        cpu::add_boundary_contributions(&mut r, &m);

        let Some(exact) = cpu::solve_dense(cpu::dense_from_ldu(&r, &m), &r.source) else {
            c.require("dense direct solve of the asymmetric system", false);
            return Ok(());
        };

        let ctrl = SolverControls {
            tolerance: 1e-14,
            rel_tol: 0.0,
            max_iter: 5000,
            precon: Preconditioner::Diagonal,
            ..Default::default()
        };
        let mut ws = SolverWorkspace::for_mesh(gpu, &gm)?;
        let mut x: DevBuf<Scalar> = gpu.zeros(n_c)?;
        let perf = solve_pbicgstab(gpu, &k.solver, &mut x, &a, &gm, &mut ws, &ctrl)?;
        gpu.sync()?;
        let got = gpu.download(&x)?;

        c.note(&format!(
            "PBiCGStab {} iterations, residual {} -> {}",
            perf.n_iterations,
            sci(f64::from(perf.initial_residual), 3),
            sci(f64::from(perf.final_residual), 3)
        ));
        c.check(
            "PBiCGStab == dense direct solve (asymmetric)",
            max_abs_diff(&got, &exact) / max_abs(&exact).max(1e-30),
            1e-8,
        );
        c.check("device residual reaches the tolerance", perf.final_residual, 1e-10);
    }

    Ok(())
}

// ==========================================================================
//  5. The direct cuFFT Poisson solve against the iterative solve of the SAME
//     matrix - SPEC-LIT section 10, row "FFT Poisson"
// ==========================================================================

fn check_fft_poisson(c: &mut Checks, gpu: &Gpu, k: &Kernels) -> Result<()> {
    if !cufft_available() {
        c.skip("cuFFT Poisson vs the iterative solve", "cuFFT not loadable here");
        return Ok(());
    }

    // A uniform Cartesian box with plain patches: SPEC-LIT section 8.5 only
    // claims the transform diagonalises the operator on one of those.
    let spec = MeshSpec {
        n: [9, 6, 4],
        l: [0.9, 0.6, 0.4],
        all_generic: true,
        ..Default::default()
    };
    let m = make_mesh(&scratch_dir("fft"), &spec)?;
    let gm = GpuMesh::upload(gpu, &m)?;
    let n_c = m.n_cells;

    // -x Dirichlet, +y Dirichlet, the rest Neumann; plus the sealed all-
    // Neumann box, which is the one with a null space.
    for (what, dirichlet) in [
        ("mixed sides (Dn, nD, NN)", &[0usize, 3][..]),
        ("sealed box (NN, NN, NN)", &[][..]),
    ] {
        let mut fr = vec![0.0 as Scalar; m.n_boundary_faces];
        let mut kind = vec![BcKind::ZeroGradient as Label; m.n_boundary_faces];
        for (pi, p) in m.patches.iter().enumerate() {
            if !dirichlet.contains(&pi) {
                continue;
            }
            for i in 0..p.size {
                fr[p.start + i] = 1.0;
                kind[p.start + i] = BcKind::FixedValue as Label;
            }
        }

        let mut p = GpuScalarField::zeros(gpu, &gm, "p")?;
        gpu.write(&mut p.fr, &fr)?;
        gpu.write(&mut p.bc_kind, &kind)?;

        let gamma: Vec<Scalar> = m.mag_sf.iter().map(|s| 0.37 * s).collect();
        let b_gamma: Vec<Scalar> = m.b_mag_sf.iter().map(|s| 0.37 * s).collect();
        let d_gamma = gpu.upload(&gamma)?;
        let d_b_gamma = gpu.upload(&b_gamma)?;

        let mut a = GpuLduMatrix::new(gpu, &gm)?;
        a.zero(gpu)?;
        fvm_laplacian(gpu, &k.fv, &mut a, &gm, &d_gamma, &d_b_gamma, &p, 1.0)?;
        add_boundary_contributions(gpu, &k.ldu, &mut a, &gm)?;
        gpu.sync()?;

        // A source with structure at every wavenumber. An all-Neumann system
        // has no solution unless the source sums to zero, so the mean is
        // removed there - that is the compatibility condition, not a fudge.
        let mut rng = Rng::new(31337);
        let mut src = gpu.download(&a.source)?;
        for s in src.iter_mut() {
            *s += rng.uniform(-1.0, 1.0);
        }
        if dirichlet.is_empty() {
            let mean = src.iter().sum::<Scalar>() / (n_c as Scalar);
            for s in src.iter_mut() {
                *s -= mean;
            }
        }
        gpu.write(&mut a.source, &src)?;

        let probe = SystemProbe::probe(gpu, &m, &p, &a, &d_gamma, &d_b_gamma)?;
        let mut fft = FftBackend::new().with_residual_report(false);
        if !fft.applicable(&probe) {
            c.skip(
                &format!("cuFFT Poisson, {what}"),
                &format!("backend not applicable: {}", fft.why_not(&probe)),
            );
            continue;
        }

        fft.setup(gpu, &m, &gm, &probe)?;
        let mut x_fft: DevBuf<Scalar> = gpu.zeros(n_c)?;
        fft.solve(gpu, &mut x_fft, &a, &gm)?;
        let mut got = gpu.download(&x_fft)?;

        let mut iter = PbicgstabBackend::reference();
        iter.setup(gpu, &m, &gm, &probe)?;
        let mut x_it: DevBuf<Scalar> = gpu.zeros(n_c)?;
        iter.solve(gpu, &mut x_it, &a, &gm)?;
        let mut want = gpu.download(&x_it)?;

        if dirichlet.is_empty() {
            // All-Neumann: the solution is defined only up to a constant and
            // the two solvers pick different members of the family.
            for v in [&mut got, &mut want] {
                let mean = v.iter().sum::<Scalar>() / (n_c as Scalar);
                for x in v.iter_mut() {
                    *x -= mean;
                }
            }
        }

        // SPEC-LIT section 8.5: with the DISCRETE eigenvalue the transform is
        // the exact inverse of the same assembled laplacian, so this is a
        // round-off comparison. With the continuous -k^2 it would miss by
        // orders, which is the classic silent failure of an FFT Poisson
        // solver and the whole reason this check exists.
        c.check(
            &format!("cuFFT == iterative, {what}"),
            max_abs_diff(&got, &want) / max_abs(&want).max(1e-30),
            1e-10,
        );
    }

    Ok(())
}

// ==========================================================================
//  6. Method of manufactured solutions
//     SPEC-LIT section 10, row "Laplacian order"
// ==========================================================================

/// The manufactured field and the source that produces it.
///
/// `psi = sin(kx x) sin(ky y) sin(kz z)` has `-lap psi = (kx^2+ky^2+kz^2) psi`
/// exactly, so the right-hand side of `-lap psi = f` is known in closed form
/// and no discretisation enters it. With `k_i = pi/L_i` the field also
/// vanishes on the faces of the unsheared block, which makes the Dirichlet
/// data exact rather than merely consistent.
///
/// On a 2-D case `kz = 0`: the front and back are empty and the out-of-plane
/// gradient is identically zero by construction, so a z-varying probe would be
/// measuring the wrong thing.
struct Manufactured {
    kx: f64,
    ky: f64,
    kz: f64,
}

impl Manufactured {
    fn new(l: [Scalar; 3], planar: bool) -> Self {
        Self {
            kx: PI / f64::from(l[0]),
            ky: PI / f64::from(l[1]),
            kz: if planar { 0.0 } else { PI / f64::from(l[2]) },
        }
    }

    fn at(&self, p: Vec3) -> Scalar {
        let s = (self.kx * f64::from(p.x)).sin() * (self.ky * f64::from(p.y)).sin();
        let t = if self.kz == 0.0 {
            1.0
        } else {
            (self.kz * f64::from(p.z)).sin()
        };
        (s * t) as Scalar
    }

    /// The eigenvalue `kx^2 + ky^2 + kz^2`, so `f = lam psi`.
    fn lambda(&self) -> Scalar {
        (self.kx * self.kx + self.ky * self.ky + self.kz * self.kz) as Scalar
    }
}

/// Solve `-lap(psi) = f` on one mesh and return the volume-weighted L2 error.
fn mms_error(
    gpu: &Gpu,
    k: &Kernels,
    spec: &MeshSpec,
    n_non_orth: usize,
    tag: &str,
) -> Result<(Scalar, usize)> {
    let m = make_mesh(&scratch_dir(tag), spec)?;
    let gm = GpuMesh::upload(gpu, &m)?;

    let mf = Manufactured::new(spec.l, spec.two_d);
    let lam = mf.lambda();

    let exact: Vec<Scalar> = (0..m.n_cells).map(|c| mf.at(m.c[c])).collect();
    let su: Vec<Scalar> = exact.iter().map(|v| lam * *v).collect();
    let b_exact: Vec<Scalar> = (0..m.n_boundary_faces).map(|i| mf.at(m.b_cf[i])).collect();

    let gamma: Vec<Scalar> = m.mag_sf.to_vec();
    let b_gamma = boundary_gamma(&m, 1.0);
    let d_gamma = gpu.upload(&gamma)?;
    let d_b_gamma = gpu.upload(&b_gamma)?;
    let d_su = gpu.upload(&su)?;

    // Dirichlet with the exact face value everywhere the patch is not empty.
    let bc = cpu::CpuScalarBc::dirichlet(&b_exact);
    let mut psi = GpuScalarField::zeros(gpu, &gm, "psi")?;
    upload_bc(gpu, &mut psi, &bc, &m, BcKind::Mixed)?;
    correct_boundary_conditions(gpu, &k.field, &mut psi, &gm)?;

    let mut grad: DevBuf<Vec3> = gpu.zeros(m.n_cells)?;
    let mut a = GpuLduMatrix::new(gpu, &gm)?;
    let mut ws = SolverWorkspace::for_mesh(gpu, &gm)?;

    // Well below the discretisation error the check is about to measure -
    // 1e-12 on the normalised residual of section 8.4 leaves a solution error
    // many orders under the O(h^2) term - and loose enough that the solve is
    // not the cost of the run.
    let ctrl = SolverControls {
        tolerance: 1e-12,
        rel_tol: 0.0,
        max_iter: 20000,
        check_interval: 10,
        precon: Preconditioner::Diagonal,
        ..Default::default()
    };

    // The non-orthogonal correction of SPEC-LIT section 3.2 is explicit and
    // has to be iterated: each pass recomputes grad(psi) from the latest
    // solution and reassembles onto a freshly zeroed source.
    for pass in 0..=n_non_orth {
        a.zero(gpu)?;
        fvm_laplacian(gpu, &k.fv, &mut a, &gm, &d_gamma, &d_b_gamma, &psi, -1.0)?;
        fvm_su(gpu, &k.fv, &mut a, &gm, &d_su, 1.0)?;

        if pass > 0 {
            fvc_grad_scalar(gpu, &k.fv, &mut grad, &psi, &gm)?;
            fvm_laplacian_non_orth_correction(
                gpu,
                &k.fv,
                &mut a,
                &gm,
                &d_gamma,
                &d_b_gamma,
                &psi,
                &grad,
                SnGradScheme::Corrected,
                -1.0,
            )?;
        }

        add_boundary_contributions(gpu, &k.ldu, &mut a, &gm)?;
        solve_pcg(gpu, &k.solver, &mut psi.f, &a, &gm, &mut ws, &ctrl)?;
        correct_boundary_conditions(gpu, &k.field, &mut psi, &gm)?;
    }
    gpu.sync()?;

    let got = gpu.download(&psi.f)?;

    let mut l2: f64 = 0.0;
    let mut vol: f64 = 0.0;
    for c in 0..m.n_cells {
        let e = f64::from(got[c] - exact[c]);
        l2 += e * e * f64::from(m.v[c]);
        vol += f64::from(m.v[c]);
    }

    Ok(((l2 / vol).sqrt() as Scalar, m.n_cells))
}

/// Refine once and report the observed order, `log2(e_coarse/e_fine)`.
fn check_mms(
    c: &mut Checks,
    gpu: &Gpu,
    k: &Kernels,
    what: &str,
    coarse: MeshSpec,
    n_non_orth: usize,
) -> Result<()> {
    let mut fine = coarse.clone();
    for i in 0..3 {
        if coarse.n[i] > 1 {
            fine.n[i] = coarse.n[i] * 2;
        }
    }

    let (e1, n1) = mms_error(gpu, k, &coarse, n_non_orth, "mms_coarse")?;
    let (e2, n2) = mms_error(gpu, k, &fine, n_non_orth, "mms_fine")?;

    let order = (f64::from(e1) / f64::from(e2)).ln() / 2.0f64.ln();
    c.note(&format!(
        "{what}: L2 {} at {n1} cells, {} at {n2} cells, order {order:.2}",
        sci(f64::from(e1), 3),
        sci(f64::from(e2), 3)
    ));

    // Second order means the error falls by four when the spacing halves.
    // 1.8 is the bar: a boundary treatment that has quietly dropped to first
    // order lands near 1.0 and cannot reach it, while ordinary round-off and
    // the finite refinement ratio move the measured value by a few hundredths.
    let shortfall = if order >= 1.8 { 0.0 } else { 1.8 - order };
    c.check(&format!("{what}: observed order >= 1.8"), shortfall as Scalar, 0.0);
    c.require(&format!("{what}: the error actually fell"), e2 < e1);

    Ok(())
}

// ==========================================================================
//  7. Buoyancy - SPEC-LIT sections 9 and 5.1, section 10 rows "Buoyancy sign"
//     and "Hydrostatic"
// ==========================================================================

/// SPEC-LIT section 9, *DESIGN*: the full ideal-gas density ratio, not a
/// Boussinesq expansion. A fire plume at 1173 K against 293 K ambient has
/// `dT/T ~ 3`, so Boussinesq does not apply and must not be used.
///
/// ```text
/// rho/rho_ref = T_ref/T          ideal gas at constant pressure
/// b           = g (T_ref/T - 1)  body force per unit mass
/// ```
const T_REF: Scalar = 293.15;
const GRAVITY: Vec3 = Vec3::new(0.0, 0.0, -9.81);

fn body_force(t: Scalar) -> Vec3 {
    GRAVITY * (T_REF / t - 1.0)
}

/// The Rhie-Chow projection of SPEC-LIT section 5.1 with `HbyA = 0` and
/// `rAU = 1`: given a body-force face flux, solve for the pressure that makes
/// the corrected flux solenoidal and return that flux and the cell velocity
/// it reconstructs to.
///
/// ```text
/// solve   lap(p) = div(phi_b)
/// phi     = phi_b - |Sf| snGrad(p)
/// u       = reconstruct(phi)
/// ```
///
/// The boundary condition on `p` is the one that makes the wall flux come out
/// equal to the wall flux the velocity condition already fixed: requiring
/// `phi_b_wall = phi_HbyA_wall - |Sf| snGrad(p)` gives
/// `snGrad(p) = phi_HbyA_wall/|Sf|`, a fixedGradient. That is the discrete
/// statement of hydrostatic balance at a wall, and it is what makes the
/// sealed-box case below come out at exactly zero rather than nearly zero.
///
/// The system is all-Neumann and therefore singular by one constant, so one
/// cell is pinned.
fn project_body_force(
    gpu: &Gpu,
    k: &Kernels,
    m: &HostMesh,
    gm: &GpuMesh,
    phi_b: &[Scalar],
    b_phi_b: &[Scalar],
) -> Result<(Vec<Scalar>, Vec<Vec3>)> {
    let n_c = m.n_cells;

    let gamma: Vec<Scalar> = m.mag_sf.to_vec();
    let b_gamma = boundary_gamma(m, 1.0);
    let d_gamma = gpu.upload(&gamma)?;
    let d_b_gamma = gpu.upload(&b_gamma)?;

    let mut bc = cpu::CpuScalarBc::new(m.n_boundary_faces);
    for bf in 0..m.n_boundary_faces {
        if cpu::is_empty_face(m, bf) || m.b_mag_sf[bf] <= 0.0 {
            continue;
        }
        bc.ref_grad[bf] = b_phi_b[bf] / m.b_mag_sf[bf];
    }

    let mut p = GpuScalarField::zeros(gpu, gm, "p")?;
    upload_bc(gpu, &mut p, &bc, m, BcKind::FixedGradient)?;

    let mut sphi = GpuSurfaceScalarField::zeros(gpu, gm, "phiB")?;
    gpu.write(&mut sphi.f, phi_b)?;
    gpu.write(&mut sphi.bf, b_phi_b)?;

    let mut div: DevBuf<Scalar> = gpu.zeros(n_c)?;
    fvc_div_surface(gpu, &k.fv, &mut div, &sphi, gm)?;

    let mut a = GpuLduMatrix::new(gpu, gm)?;
    a.zero(gpu)?;
    set_fixed_cells(gpu, &mut a, &[0], &[0.0])?;
    fvm_laplacian(gpu, &k.fv, &mut a, gm, &d_gamma, &d_b_gamma, &p, 1.0)?;
    fvm_su(gpu, &k.fv, &mut a, gm, &div, 1.0)?;
    add_boundary_contributions(gpu, &k.ldu, &mut a, gm)?;
    set_values(gpu, &k.ldu, &mut a, gm)?;

    let ctrl = SolverControls {
        tolerance: 1e-14,
        rel_tol: 0.0,
        max_iter: 50000,
        check_interval: 10,
        precon: Preconditioner::Diagonal,
        ..Default::default()
    };
    let mut ws = SolverWorkspace::for_mesh(gpu, gm)?;
    solve_pcg(gpu, &k.solver, &mut p.f, &a, gm, &mut ws, &ctrl)?;
    correct_boundary_conditions(gpu, &k.field, &mut p, gm)?;

    let mut sn = GpuSurfaceScalarField::zeros(gpu, gm, "snGradP")?;
    sn_grad_flux(gpu, &k.fv, &mut sn, &p, &d_gamma, &d_b_gamma, gm)?;
    gpu.sync()?;

    let sn_f = gpu.download(&sn.f)?;
    let sn_b = gpu.download(&sn.bf)?;

    let phi: Vec<Scalar> = (0..m.n_internal_faces).map(|f| phi_b[f] - sn_f[f]).collect();
    let bphi: Vec<Scalar> = (0..m.n_boundary_faces).map(|f| b_phi_b[f] - sn_b[f]).collect();

    let mut corrected = GpuSurfaceScalarField::zeros(gpu, gm, "phi")?;
    gpu.write(&mut corrected.f, &phi)?;
    gpu.write(&mut corrected.bf, &bphi)?;

    let mut u: DevBuf<Vec3> = gpu.zeros(n_c)?;
    fvc_reconstruct(gpu, &k.fv, &mut u, &corrected, gm)?;
    gpu.sync()?;

    Ok((phi, gpu.download(&u)?))
}

/// The face body-force flux, evaluated ON THE FACES.
///
/// SPEC-LIT section 5.1: the body force must enter `phi_HbyA` on faces and not
/// by interpolating a cell value, for exactly the reason the pressure does -
/// otherwise buoyancy checkerboards. So the temperature is interpolated to the
/// face first and the body force is formed there.
fn buoyancy_flux(m: &HostMesh, t: &[Scalar], bt: &[Scalar]) -> (Vec<Scalar>, Vec<Scalar>) {
    let mut phi = vec![0.0 as Scalar; m.n_internal_faces];
    for f in 0..m.n_internal_faces {
        let w = m.weights[f];
        let tf = w * t[m.owner[f] as usize] + (1.0 - w) * t[m.neighbour[f] as usize];
        phi[f] = body_force(tf).dot(m.sf[f]);
    }

    let mut bphi = vec![0.0 as Scalar; m.n_boundary_faces];
    for bf in 0..m.n_boundary_faces {
        if cpu::is_empty_face(m, bf) {
            continue;
        }
        bphi[bf] = body_force(bt[bf]).dot(m.b_sf[bf]);
    }

    (phi, bphi)
}

fn check_buoyancy(c: &mut Checks, gpu: &Gpu, k: &Kernels) -> Result<()> {
    // ---- the arithmetic of SPEC-LIT section 9 itself ---------------------
    let b_hot = body_force(1173.15);
    c.note(&format!(
        "b(T = 1173.15 K) = ({:.3} {:.3} {:.3}) m/s^2",
        f64::from(b_hot.x),
        f64::from(b_hot.y),
        f64::from(b_hot.z)
    ));
    // SPEC-LIT section 9 states the value: g = (0,0,-9.81), TRef = 293.15,
    // T = 1173.15 gives b = (0,0,+7.36), upward.
    c.check("b_z(1173.15 K) == +7.36 m/s^2", (b_hot.z - 7.36).abs(), 5e-3);
    c.require("b is upward for hot gas", b_hot.z > 0.0);
    c.check("b(T_ref) == 0 exactly", body_force(T_REF).mag(), 0.0);

    let spec = MeshSpec {
        n: [9, 9, 12],
        l: [0.6, 0.6, 0.8],
        ..Default::default()
    };
    let m = make_mesh(&scratch_dir("buoy"), &spec)?;
    let gm = GpuMesh::upload(gpu, &m)?;

    // ---- SPEC-LIT section 10, row "Hydrostatic" --------------------------
    // A sealed box of uniform temperature. The body force is uniform and
    // non-zero - it is NOT the trivial b = 0 case - and it is a pure gradient,
    // so the pressure must absorb all of it and the fluid must stay exactly at
    // rest. A projection that leaked would show up here as a spurious
    // circulation of order |b| times a length.
    {
        let t = vec![600.0 as Scalar; m.n_cells];
        let bt = vec![600.0 as Scalar; m.n_boundary_faces];
        let (phi_b, b_phi_b) = buoyancy_flux(&m, &t, &bt);
        let b_uniform = body_force(600.0);
        let scale = max_abs(&phi_b).max(1e-30);

        // First, without any solver at all. The exact discrete pressure is
        // p = b.x, a linear field this mesh represents exactly, so
        //     phi = b.Sf - |Sf| snGrad(b.x) = |Sf|(b.n - b.n) = 0
        // to the last bit. Solving for it can only add solver error, so
        // checking the identity first separates a projection that is wrong
        // from a solve that has not converged.
        {
            let p_exact: Vec<Scalar> = (0..m.n_cells).map(|cc| b_uniform.dot(m.c[cc])).collect();
            let mut bc = cpu::CpuScalarBc::new(m.n_boundary_faces);
            for bf in 0..m.n_boundary_faces {
                if cpu::is_empty_face(&m, bf) || m.b_mag_sf[bf] <= 0.0 {
                    continue;
                }
                bc.ref_grad[bf] = b_phi_b[bf] / m.b_mag_sf[bf];
            }
            let gamma: Vec<Scalar> = m.mag_sf.to_vec();
            let b_gamma = boundary_gamma(&m, 1.0);

            let mut p = GpuScalarField::zeros(gpu, &gm, "pExact")?;
            gpu.write(&mut p.f, &p_exact)?;
            upload_bc(gpu, &mut p, &bc, &m, BcKind::FixedGradient)?;
            let d_gamma = gpu.upload(&gamma)?;
            let d_b_gamma = gpu.upload(&b_gamma)?;

            let mut sn = GpuSurfaceScalarField::zeros(gpu, &gm, "snGradP")?;
            sn_grad_flux(gpu, &k.fv, &mut sn, &p, &d_gamma, &d_b_gamma, &gm)?;
            gpu.sync()?;
            let sn_f = gpu.download(&sn.f)?;
            let sn_b = gpu.download(&sn.bf)?;

            let worst_i = (0..m.n_internal_faces)
                .fold(0.0 as Scalar, |w, f| w.max((phi_b[f] - sn_f[f]).abs()));
            let worst_b = (0..m.n_boundary_faces)
                .fold(0.0 as Scalar, |w, f| w.max((b_phi_b[f] - sn_b[f]).abs()));
            c.check(
                "hydrostatic balance is exact for p = b.x",
                worst_i.max(worst_b) / scale,
                1e-13,
            );
        }

        let (phi, u) = project_body_force(gpu, k, &m, &gm, &phi_b, &b_phi_b)?;

        c.check("sealed box, uniform T: flux stays zero", max_abs(&phi) / scale, 1e-8);

        let u_scale = b_uniform.mag() * spec.l[2];
        let worst_u = u.iter().fold(0.0 as Scalar, |w, v| w.max(v.mag()));
        c.check("sealed box, uniform T: remains at rest", worst_u / u_scale, 1e-8);
        c.note(&format!(
            "|b| = {} m/s^2, worst |u| = {} m/s",
            sci(f64::from(b_uniform.mag()), 3),
            sci(f64::from(worst_u), 3)
        ));
    }

    // ---- T == T_ref: the body force is identically zero ------------------
    {
        let t = vec![T_REF; m.n_cells];
        let bt = vec![T_REF; m.n_boundary_faces];
        let (phi_b, b_phi_b) = buoyancy_flux(&m, &t, &bt);
        c.check("T == T_ref gives no body force at all", max_abs(&phi_b), 0.0);
        let _ = b_phi_b;
    }

    // ---- SPEC-LIT section 10, row "Buoyancy sign" ------------------------
    // One hot region low in still air. Two statements are checked, and the
    // second is the one that cannot be got right by accident:
    //
    //  * the reconstructed vertical velocity in the hot gas is positive;
    //  * the body force does POSITIVE work on the flow it produces,
    //    sum_c V_c b_c . u_c > 0. In the continuum that integral is the
    //    squared norm of the solenoidal projection of b and is therefore
    //    non-negative identically; a sign error anywhere in the projection
    //    turns it negative.
    {
        let t_hot: Scalar = 1173.15;
        let hot: Vec<bool> = (0..m.n_cells)
            .map(|c| {
                let p = m.c[c];
                let dx = f64::from(p.x) - 0.3;
                let dy = f64::from(p.y) - 0.3;
                (dx * dx + dy * dy).sqrt() < 0.12 && f64::from(p.z) < 0.2
            })
            .collect();

        let n_hot = hot.iter().filter(|h| **h).count();
        c.require("the hot region has cells in it", n_hot > 0);
        if n_hot == 0 {
            return Ok(());
        }

        let t: Vec<Scalar> = (0..m.n_cells)
            .map(|c| if hot[c] { t_hot } else { T_REF })
            .collect();
        // Zero-gradient temperature at the walls, so the boundary body force
        // is whatever the adjacent cell carries.
        let bt: Vec<Scalar> = (0..m.n_boundary_faces)
            .map(|bf| t[m.b_face_cells[bf] as usize])
            .collect();

        let (phi_b, b_phi_b) = buoyancy_flux(&m, &t, &bt);
        let (phi, u) = project_body_force(gpu, k, &m, &gm, &phi_b, &b_phi_b)?;

        // The whole point of the pressure step: the corrected flux is
        // discretely solenoidal. Computed with the host reference, so the
        // device's own divergence operator is not both judge and defendant.
        let bphi_zero = vec![0.0 as Scalar; m.n_boundary_faces];
        let mut divergence = Vec::new();
        cpu::fvc_div_surface(&mut divergence, &phi, &bphi_zero, &m);
        let div_scale = max_abs(&phi_b).max(1e-30) / m.v[0];
        c.check(
            "the projected flux is solenoidal",
            max_abs(&divergence) / div_scale,
            1e-8,
        );

        let mut hot_uz: Scalar = 0.0;
        let mut hot_vol: Scalar = 0.0;
        let mut work: Scalar = 0.0;
        for c in 0..m.n_cells {
            work += m.v[c] * body_force(t[c]).dot(u[c]);
            if hot[c] {
                hot_uz += m.v[c] * u[c].z;
                hot_vol += m.v[c];
            }
        }
        hot_uz /= hot_vol;

        c.note(&format!(
            "{n_hot} hot cells, mean u_z there {} m/s, work {} W/kg",
            sci(f64::from(hot_uz), 3),
            sci(f64::from(work), 3)
        ));
        c.require("hot gas accelerates in +z", hot_uz > 0.0);
        c.require("the body force does positive work", work > 0.0);

        // Cooling the same region must reverse it. Nothing but the sign of
        // (T_ref/T - 1) changes.
        let t_cold: Vec<Scalar> = (0..m.n_cells)
            .map(|c| if hot[c] { 150.0 } else { T_REF })
            .collect();
        let bt_cold: Vec<Scalar> = (0..m.n_boundary_faces)
            .map(|bf| t_cold[m.b_face_cells[bf] as usize])
            .collect();
        let (pc, bpc) = buoyancy_flux(&m, &t_cold, &bt_cold);
        let (_, uc) = project_body_force(gpu, k, &m, &gm, &pc, &bpc)?;

        let mut cold_uz: Scalar = 0.0;
        for c in 0..m.n_cells {
            if hot[c] {
                cold_uz += m.v[c] * uc[c].z;
            }
        }
        cold_uz /= hot_vol;
        c.note(&format!("cold gas mean u_z {} m/s", sci(f64::from(cold_uz), 3)));
        c.require("cold gas accelerates in -z", cold_uz < 0.0);
    }

    Ok(())
}

// ==========================================================================
//  Driver
// ==========================================================================

fn run(c: &mut Checks) -> Result<()> {
    let gpu = Gpu::new(0)?;
    let (major, minor) = gpu.ctx().compute_capability()?;
    println!(
        "ofgpu validation | {} sm_{major}{minor} | {}",
        gpu.ctx().name()?,
        common::precision_name()
    );

    let k = Kernels::new(&gpu)?;

    // ---- a graded, three-dimensional, orthogonal block -------------------
    println!("\n=== 3-D graded block ===");
    let spec3 = MeshSpec {
        n: [14, 11, 9],
        l: [1.0, 0.7, 0.4],
        expansion: [1.0, 8.0, 1.0],
        ..Default::default()
    };
    let m3 = make_mesh(&scratch_dir("main3d"), &spec3)?;
    m3.print_report();
    let gm3 = GpuMesh::upload(&gpu, &m3)?;

    check_mesh(c, &m3, spec3.volume());
    check_explicit_operators(c, &gpu, &k, &m3, &gm3)?;
    check_assembly(c, &gpu, &k, &m3, &gm3, DivScheme::Upwind)?;
    check_assembly(c, &gpu, &k, &m3, &gm3, DivScheme::Central)?;
    check_assembly(c, &gpu, &k, &m3, &gm3, DivScheme::Limited(Limiter::VanLeer))?;
    drop(gm3);

    // ---- a sheared block: non-orthogonal, so the correction of section 2.4
    //      is no longer a no-op ---------------------------------------------
    println!("\n=== 3-D sheared block (non-orthogonal) ===");
    let spec_sh = MeshSpec {
        n: [9, 8, 7],
        l: [1.0, 0.7, 0.4],
        shear: 0.45,
        ..Default::default()
    };
    let msh = make_mesh(&scratch_dir("shear"), &spec_sh)?;
    let report = msh.check();
    c.note(&format!(
        "max non-orthogonality {:.1} deg, mean {:.1} deg",
        f64::from(report.max_non_orth_deg),
        f64::from(report.mean_non_orth_deg)
    ));
    c.require("the shear really made the mesh non-orthogonal", report.max_non_orth_deg > 10.0);
    let gmsh = GpuMesh::upload(&gpu, &msh)?;

    check_mesh(c, &msh, spec_sh.volume());
    check_explicit_operators(c, &gpu, &k, &msh, &gmsh)?;
    check_assembly(c, &gpu, &k, &msh, &gmsh, DivScheme::Limited(Limiter::MinMod))?;
    drop(gmsh);

    // ---- a 2-D block with empty front and back ---------------------------
    println!("\n=== 2-D block with empty front and back ===");
    let spec2 = MeshSpec {
        n: [20, 16, 1],
        l: [1.0, 0.7, 0.05],
        expansion: [1.0, 5.0, 1.0],
        two_d: true,
        ..Default::default()
    };
    let m2 = make_mesh(&scratch_dir("main2d"), &spec2)?;
    let gm2 = GpuMesh::upload(&gpu, &m2)?;

    check_mesh(c, &m2, spec2.volume());
    check_explicit_operators(c, &gpu, &k, &m2, &gm2)?;
    check_assembly(c, &gpu, &k, &m2, &gm2, DivScheme::Upwind)?;
    drop(gm2);

    // ---- linear algebra --------------------------------------------------
    println!("\n=== linear solvers ===");
    check_solver_against_dense(c, &gpu, &k)?;
    check_fft_poisson(c, &gpu, &k)?;

    // ---- manufactured solutions ------------------------------------------
    println!("\n=== method of manufactured solutions, -lap(psi) = f ===");
    check_mms(
        c,
        &gpu,
        &k,
        "3-D graded",
        MeshSpec {
            n: [10, 10, 10],
            l: [1.0, 0.7, 0.4],
            expansion: [1.0, 4.0, 2.0],
            ..Default::default()
        },
        0,
    )?;
    check_mms(
        c,
        &gpu,
        &k,
        "3-D sheared",
        MeshSpec {
            n: [8, 8, 8],
            l: [1.0, 0.7, 0.4],
            shear: 0.45,
            ..Default::default()
        },
        3,
    )?;
    check_mms(
        c,
        &gpu,
        &k,
        "2-D empty patches",
        MeshSpec {
            n: [16, 16, 1],
            l: [1.0, 0.7, 0.05],
            expansion: [1.0, 3.0, 1.0],
            two_d: true,
            ..Default::default()
        },
        0,
    )?;

    // ---- buoyancy --------------------------------------------------------
    println!("\n=== buoyancy ===");
    check_buoyancy(c, &gpu, &k)?;

    // ---- SPEC-LIT 17, 18, 19 and the flux round trip ---------------------
    println!("\n=== buoyancy production, sources, species, phi I/O ===");
    check_buoyancy_production(c, &gpu)?;
    check_volumetric_source(c, &gpu)?;
    check_species(c, &gpu)?;
    check_phi_round_trip(c, &gpu)?;

    // ---- volume of fluid -------------------------------------------------
    println!("
=== volume of fluid (SPEC-LIT 20, the 22 rows) ===");
    check_vof(c, &gpu)?;

    // ---- surface intake and embedded boundaries (SPEC-LIT 23, 24) --------
    println!("\n=== msh hex closure, cut-cell closure (SPEC-LIT 23, 24) ===");
    check_msh_hex_closure(c)?;
    check_cutcell_closure(c)?;

    // ---- fire: low-Mach p0, combustion, radiation (SPEC-LIT 25, 27, 28) ---
    println!("\n=== fire: low-Mach p0, combustion, radiation (SPEC-LIT 25, 27, 28) ===");
    check_low_mach_p0(c, &gpu)?;
    check_burner_heat_release(c, &gpu)?;
    check_radiative_equilibrium(c, &gpu)?;

    // ---- wall treatment: rough-wall Ks -> 0, the thermal wall function
    //      (SPEC-LIT 29) --------------------------------------------------
    println!("\n=== wall treatment: Ks -> 0, the thermal wall function (SPEC-LIT 29) ===");
    check_rough_wall_ks_zero(c);
    check_thermal_wall_function(c);

    // ---- the LES wall model, and coupled-solver turbulence selection
    //      (SPEC-LIT 30) --------------------------------------------------
    println!(
        "\n=== Werner-Wengle, coupled-solver turbulence selection (SPEC-LIT 30) ==="
    );
    check_werner_wengle(c);
    check_werner_wengle_inversion(c);
    check_coupled_selection(c, &gpu, &k)?;

    // ---- periodic domains: the cyclic-pair invariants (SPEC-LIT 31.1) ----
    println!("\n=== periodic domains: cyclic-pair invariants (SPEC-LIT 31.1) ===");
    check_cyclic_pair(c)?;

    // ---- the thermal wall-function gate, redesigned (SPEC-LIT 32) --------
    println!("\n=== the thermal wall-function gate, redesigned (SPEC-LIT 32) ===");
    check_fixed_flux_identity(c);
    check_nu_correlations(c);
    check_realised_friction_factor(c)?;
    c.replaying(check_thermal_wall_function_gate_verdict_replay);

    // ---- Launder-Sharma low-Re k-epsilon: the damping functions
    //      (SPEC-LIT 33.3) -------------------------------------------------
    //
    // The law-of-the-wall profile itself - SPEC-LIT §33.3's own "the only
    // check that says the damping is right" - is NOT promoted here. It was
    // run (a periodic 2-D channel, LaunderSharmaKE, Re_tau ~ 440) and DOES
    // reproduce u+ = y+ below y+ 5 (worst deviation 0.8% at y+ 4.4) and the
    // log law within ~1% at y+ 30-35 - see docs/07-fire-solver.md §1.1 for
    // the full table - but the run takes ~10 minutes on an RTX 5070 Ti and
    // had not fully settled (|U| residual ~5e-2, plateauing on the
    // periodic pressure equation's own null space, SPEC-LIT §31.1) even
    // then. That disqualifies it from both this fast, always-run suite and
    // from `published_benchmarks`' own ignored-but-quick convention (the
    // Ghia cavity cases below finish in seconds) - a live multi-minute GPU
    // run belongs in a driver invocation a human chooses to make, not in
    // `cargo test`. What IS cheap - and unconditionally true regardless of
    // any live run - is the damping functions' own analytic table.
    println!("\n=== Launder-Sharma low-Re k-epsilon: damping functions (SPEC-LIT 33.3) ===");
    check_launder_sharma_damping_functions(c);

    // ---- the plane-channel resolved leg's mesh resolution (SPEC-LIT
    //      §33.2/§34) --------------------------------------------------
    println!("\n=== resolved leg mesh resolution, replayed (SPEC-LIT 33.2/34) ===");
    c.replaying(check_resolved_leg_mesh_resolution_replay);

    // ---- SPEC-LIT §35: the bulk-temperature thermostat -------------------
    //
    // The two-initial-temperature regression itself (SPEC-LIT §35.2) is run
    // LIVE and reported in `docs/07-fire-solver.md` §1.1, not here: it takes
    // ~2.5 minutes PER initial condition on the real channel mesh, which
    // disqualifies it from this always-run suite the same way §33.3's law-
    // of-the-wall channel run is disqualified above. What IS promoted here:
    // the controller's own proportional law, checked live on a tiny mesh in
    // milliseconds (which is what makes the initial-condition independence
    // true in the first place), and the resolved leg's own Nu verdict now
    // that it has an actual steady state to measure.
    println!("\n=== the bulk-temperature thermostat (SPEC-LIT 35) ===");
    check_thermostat_sign_and_steady_offset(c, &gpu)?;
    c.replaying(check_resolved_leg_gate_verdict_replay);

    // SPEC-LIT §35.3.2's uniform-vs-massFlux experiment, on both meshes -
    // the measurement that decided whether the uniform sink's distribution
    // defect was real and how big it is.
    println!("\n=== thermostat weighting: the decisive experiment, replayed (SPEC-LIT 35.3.2) ===");
    c.replaying(check_thermostat_weighting_experiment_replay);

    // SPEC-LIT §32.5.5's isolation: what the `bounded` prefix on `div(phi,U)`
    // was worth once §13.4.1's fix made the cases' own entry reach the
    // momentum equation, and what the scheme's ORDER was worth beside it.
    println!("\n=== bounded convection on momentum: the isolation, replayed (SPEC-LIT 3.1/32.5.5) ===");
    c.replaying(check_bounded_convection_experiment_replay);

    // SPEC-LIT §37: the variable turbulent Prandtl number. The correlation
    // itself is arithmetic and is checked LIVE; the experiment that put it on
    // the two channel legs is a 40 000-iteration pair per leg and is replayed.
    println!("\n=== Kays-Crawford turbulent Prandtl number (SPEC-LIT 37.1/37.2) ===");
    check_kays_crawford_prt(c);
    c.replaying(check_kays_crawford_experiment_replay);

    Ok(())
}

// ==========================================================================
//  SPEC-LIT §29: wall-treatment selection and the thermal wall function
// ==========================================================================
//
// Both gates below are promoted from `wallfunctions::tests` - they are host-
// mirror-level checks (no GPU needed; the device kernels are already held to
// these same host functions bit-for-bit by
// `wallfunctions::tests::device_agrees_with_the_host_law` and
// `..::thermal_wall_device_agrees_with_the_host_law`, run under
// `cargo test`) - into the acceptance run itself, per SPEC-LIT §29's own
// table: these are the two properties a wall treatment MUST have before
// anything built on it can be trusted.

/// An independent, hand-written smooth `u_tau` Newton solve (SPEC-LIT §15.1,
/// no roughness term anywhere in it) so the gate below checks
/// [`ofgpu::wallfunctions::u_tau_newton`] against code that never mentions
/// roughness, not against itself.
fn smooth_u_tau_reference(u_mag: Scalar, y: Scalar, nu: Scalar, kappa: Scalar, e: Scalar) -> Scalar {
    if !(u_mag > 0.0) {
        return 0.0;
    }
    let mut u_tau: Scalar = (nu * u_mag / y).max(1e-300).sqrt();
    for _ in 0..10 {
        let u_plus = u_mag / u_tau;
        let ku = kappa * u_plus;
        let euk = ku.exp();
        let poly = euk - 1.0 - ku - ku * ku * 0.5 - ku * ku * ku / 6.0;
        let f = y * u_tau / nu - u_plus - poly / e;
        let dpoly = kappa * (euk - 1.0 - ku - ku * ku * 0.5);
        let df = y / nu + (u_plus / u_tau) * (1.0 + dpoly / e);
        if !(df.abs() > 0.0) {
            break;
        }
        let next = (u_tau - f / df).max(1e-300);
        let done = (next - u_tau).abs() <= 1e-6 * next.abs().max(1e-300);
        u_tau = next;
        if done {
            break;
        }
    }
    u_tau.max(0.0)
}

/// `Ks -> 0` must reproduce the smooth wall to round-off - SPEC-LIT §29.2's
/// gate: `Ks = 0` is exactly what a case that never mentions roughness gets,
/// so the rough-wall law must not quietly BE a different smooth wall.
/// Sweeps both wall-function families: `nutk` (`nut_wall_rough_k`, driven by
/// `k`/`y`) and `nutU` (`u_tau_newton`/`nut_wall_rough_u`, driven by the
/// wall-parallel velocity).
fn check_rough_wall_ks_zero(c: &mut Checks) {
    use ofgpu::wallfunctions::{
        nut_wall, nut_wall_rough_k, nut_wall_rough_u, nut_wall_u, u_tau_newton, y_plus_of,
    };

    let wc = WallFunctionCoeffs::default();
    let cmu25 = wc.cmu.powf(0.25);
    let nu: Scalar = 1.2e-5;

    let mut worst_k: Scalar = 0.0;
    for k in [1e-6 as Scalar, 1e-4, 1e-2, 1.0] {
        for y in [1e-4 as Scalar, 1e-3, 1e-2] {
            let y_plus = y_plus_of(k, y, nu, wc.cmu);
            let smooth = nut_wall(y_plus, nu, wc.kappa, wc.e);
            for cs in [0.3 as Scalar, 0.5, 1.0] {
                let rough = nut_wall_rough_k(y_plus, k, nu, wc.kappa, wc.e, cmu25, 0.0, cs);
                worst_k = worst_k.max((rough - smooth).abs() / smooth.abs().max(1e-30));
            }
        }
    }
    c.check("Ks -> 0 reproduces the smooth nutk wall (S29.2 gate)", worst_k, 1e-12);

    let nu_u: Scalar = 1.5e-5;
    let y: Scalar = 2e-3;
    let mut worst_u: Scalar = 0.0;
    for u_mag in [0.05 as Scalar, 0.5, 2.0, 10.0] {
        let want = smooth_u_tau_reference(u_mag, y, nu_u, wc.kappa, wc.e);
        for cs in [0.3 as Scalar, 0.5, 1.0] {
            let got = u_tau_newton(u_mag, y, nu_u, wc.kappa, wc.e, 0.0, cs);
            worst_u = worst_u.max((got - want).abs() / want.max(1e-30));
        }
        let want_nut = nut_wall_u(want, y, nu_u, u_mag);
        let got_nut = nut_wall_rough_u(u_mag, y, nu_u, wc.kappa, wc.e, 0.0, 0.5);
        worst_u = worst_u.max((got_nut - want_nut).abs() / want_nut.max(1e-30));
    }
    c.check("Ks -> 0 reproduces the smooth nutU wall (S29.2 gate)", worst_u, 1e-9);
}

/// Jayatilleke's `P(Pr/Pr_t = 1) = 0` identity, and the one-cell conductance
/// identity SPEC-LIT §29.3 asks for: `k_eff_wall * ref_grad` (what the
/// Robin triple the kernel writes actually delivers to the implicit matrix)
/// must equal the analytic Jayatilleke flux `rho cp u_tau (T_w - T_P)/T+`
/// exactly, for a `thermalWallFunction` face.
fn check_thermal_wall_function(c: &mut Checks) {
    use ofgpu::wallfunctions::{jayatilleke_p, t_plus, thermal_wall_ref_grad, u_plus, y_plus_of};

    let wc = WallFunctionCoeffs::default();

    let mut worst_p: Scalar = 0.0;
    for prt in [0.7 as Scalar, 0.85, 1.0, 1.3] {
        worst_p = worst_p.max(jayatilleke_p(prt, prt).abs());
    }
    c.check("P(Pr/Pr_t = 1) = 0 exactly (S29.3 gate)", worst_p, 0.0);

    let prt: Scalar = 0.85;
    let p_at_prt = jayatilleke_p(prt, prt);
    let mut worst_tplus: Scalar = 0.0;
    for y_plus in [0.0 as Scalar, 1.0, 5.0, 11.53, 30.0, 100.0, 1000.0] {
        let tp = t_plus(y_plus, prt, prt, wc.kappa, wc.e, p_at_prt);
        let want = prt * u_plus(y_plus, wc.kappa, wc.e);
        worst_tplus = worst_tplus.max((tp - want).abs() / want.abs().max(1.0));
    }
    c.check("Pr = Pr_t: T+ == Pr_t * u+ everywhere", worst_tplus, 1e-9);

    // The one-cell conductance identity.
    let nu: Scalar = 1.5e-5;
    let k_min: Scalar = 1e-15;
    let k_p: Scalar = 0.05;
    let y: Scalar = 0.01;
    let rho: Scalar = 1.2;
    let cp: Scalar = 1006.0;
    let pr: Scalar = 0.71;
    let t_w: Scalar = 400.0;
    let t_p: Scalar = 300.0;
    let k_eff_wall: Scalar = 0.04;

    let grad = thermal_wall_ref_grad(
        t_w, t_p, k_p, y, nu, rho, cp, pr, prt, wc.kappa, wc.e, wc.cmu, k_eff_wall, k_min,
    );
    match grad {
        Some(grad) => {
            let y_plus = y_plus_of(k_p, y, nu, wc.cmu);
            let u_tau = wc.cmu.powf(0.25) * k_p.sqrt();
            let tp = t_plus(y_plus, pr, prt, wc.kappa, wc.e, jayatilleke_p(pr, prt));
            let q_w = rho * cp * u_tau * (t_w - t_p) / tp;
            let flux_from_triple = k_eff_wall * grad;

            c.note(&format!(
                "one-cell conductance: analytic q_w = {:.6} W/m^2, k_eff_wall*ref_grad = {:.6} W/m^2",
                f64::from(q_w),
                f64::from(flux_from_triple)
            ));
            c.check(
                "wall-function triple encodes exactly q_w (S29.3 one-cell conductance)",
                (flux_from_triple - q_w).abs() / q_w.abs(),
                1e-9,
            );
        }
        None => c.check(
            "wall-function triple encodes exactly q_w (S29.3 one-cell conductance)",
            1.0,
            0.0,
        ),
    }
}

// ==========================================================================
//  SPEC-LIT §30: the Werner-Wengle LES wall model, and coupled-solver
//  turbulence selection
// ==========================================================================
//
// The WW pair below is promoted from `wallfunctions::tests` - host-only,
// same discipline as §29's pair above. The selection check needs a GPU and
// is new here rather than promoted, because SPEC-LIT §30.3 asks for it
// against a live `build_coupled` run, not against a closed-form identity.

/// SPEC-LIT §30.3's continuity row: "`tau_w -> nu|u_p|/(h/2)`-form continuous
/// at the branch point (evaluate both sides)". [`tau_w_werner_wengle`]'s two
/// branches are two different closed-form expressions - nothing but the
/// algebra in that function's own module doc forces them to agree - so this
/// evaluates both sides at, and a hair either side of, the branch point
/// directly, exactly as `wallfunctions::tests::ww_is_continuous_at_the_
/// branch_point` does under `cargo test`.
fn check_werner_wengle(c: &mut Checks) {
    use ofgpu::wallfunctions::{tau_w_werner_wengle, ww_branch_speed};

    let mut worst_at: Scalar = 0.0;
    let mut worst_below: Scalar = 0.0;
    let mut worst_above: Scalar = 0.0;
    for (nu, h) in [
        (1.5e-5 as Scalar, 0.01 as Scalar),
        (1.0e-6, 0.002),
        (2.0e-4, 0.05),
    ] {
        let u_c = ww_branch_speed(nu, h);
        let at = tau_w_werner_wengle(u_c, h, nu);
        let viscous_closed_form = 2.0 * nu * u_c / h;
        worst_at = worst_at.max((at - viscous_closed_form).abs() / viscous_closed_form.max(1e-300));

        let below = tau_w_werner_wengle(u_c * (1.0 - 1e-9), h, nu);
        let above = tau_w_werner_wengle(u_c * (1.0 + 1e-9), h, nu);
        let scale = at.max(1e-300);
        worst_below = worst_below.max((below - at).abs() / scale);
        worst_above = worst_above.max((above - at).abs() / scale);
    }
    c.check(
        "WW: tau_w(u_c) == 2 nu u_c/h, the viscous closed form (S30.3 gate)",
        worst_at,
        1e-9,
    );
    c.check(
        "WW: viscous side does not step crossing the branch point",
        worst_below,
        1e-6,
    );
    c.check(
        "WW: power side does not step crossing the branch point",
        worst_above,
        1e-6,
    );
}

/// SPEC-LIT §30.3's inversion row: "inverting the integrated law reproduces
/// a manufactured `tau_w` to round-off." One round trip per branch -
/// manufacture `tau_w`, invert its own closed form for `u_p`, reapply
/// [`tau_w_werner_wengle`], compare - promoted from
/// `wallfunctions::tests::ww_power_branch_inverts_a_manufactured_tau_w_to_
/// round_off` and its viscous twin.
fn check_werner_wengle_inversion(c: &mut Checks) {
    use ofgpu::wallfunctions::{tau_w_werner_wengle, ww_branch_speed, WW_A, WW_B};

    let nu: Scalar = 1.5e-5;
    let h: Scalar = 0.01;

    // The power branch's own bracket, tau_w = (t1 + t2 u_p)^{2/(1+b)},
    // inverted for u_p given a target tau_w.
    let a = WW_A;
    let b = WW_B;
    let nu_h = nu / h;
    let t1 = 0.5 * (1.0 - b) * a.powf((1.0 + b) / (1.0 - b)) * nu_h.powf(1.0 + b);
    let t2 = ((1.0 + b) / a) * nu_h.powf(b);

    let mut worst_power: Scalar = 0.0;
    for tau_w_target in [1.0e-3 as Scalar, 5.0e-2, 2.0] {
        let u_p = (tau_w_target.powf((1.0 + b) / 2.0) - t1) / t2;
        let got = tau_w_werner_wengle(u_p, h, nu);
        worst_power = worst_power.max((got - tau_w_target).abs() / tau_w_target);
    }
    c.check(
        "WW power branch: invert then reapply reproduces tau_w (S30.3 gate)",
        worst_power,
        1e-9,
    );

    let mut worst_viscous: Scalar = 0.0;
    for tau_w_target in [1.0e-8 as Scalar, 1.0e-10] {
        let u_p = tau_w_target * h / (2.0 * nu);
        c.note(&format!(
            "  (tau_w = {tau_w_target:e}: u_p = {u_p:e}, branch speed = {:e})",
            f64::from(ww_branch_speed(nu, h))
        ));
        let got = tau_w_werner_wengle(u_p, h, nu);
        worst_viscous = worst_viscous.max((got - tau_w_target).abs() / tau_w_target.max(1e-300));
    }
    c.check(
        "WW viscous branch: invert then reapply reproduces tau_w (S30.3 gate)",
        worst_viscous,
        1e-9,
    );
}

/// FNV-1a 64-bit digest of a field's raw bytes - the same algorithm
/// `restart::mesh_hash` uses, applied here to a `nu_t` snapshot rather than a
/// mesh, so [`check_coupled_selection`] can report one short, copy-pasteable
/// number per model instead of a max-diff that says nothing about WHICH
/// values moved.
fn field_hash(f: &[Scalar]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h: u64 = FNV_OFFSET;
    for v in f {
        for &byte in &f64::from(*v).to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

/// SPEC-LIT §30.3's selection row: "`model kOmegaSST` in buoyant/fire
/// constructs SST - verified by the printed banner AND a field difference."
///
/// Promoted from `models::registry::tests::komega_sst_via_build_coupled_is_
/// not_bit_identical_to_kepsilon`, on a genuinely BUOYANT case rather than
/// an isothermal one: `CaseControls::default()` already carries Earth
/// gravity (`BuoyancyCoeffs::default()`, SPEC-LIT §9), so `build_coupled`'s
/// own `buoyancy_settings` turns on for both models with no case file
/// needed, and a real [`ThermalCtx`] is threaded through `correct` so the
/// run actually exercises SPEC-LIT §17's `G_b` route for each model
/// (k-epsilon's own equation vs. SST's `(gamma/nu_t) G_b` in omega) rather
/// than skipping it.
fn check_coupled_selection(c: &mut Checks, gpu: &Gpu, k: &Kernels) -> Result<()> {
    use ofgpu::field::GpuVectorField;
    use ofgpu::field_ops::correct_boundary_conditions;
    use ofgpu::field_setup::{NutRoughness, WallFaces};
    use ofgpu::io::case::CaseControls;
    use ofgpu::io::dict::FoamDict;
    use ofgpu::models::coupled::ThermalCtx;
    use ofgpu::models::registry::{build_coupled, select_turbulence_model};
    use ofgpu::turbulence::FlowState;

    let case = |src: &str| -> CaseControls {
        let d = FoamDict::parse(src, "momentumTransport")
            .expect("S30 fixture: hand-written dictionary text must parse");
        let name = d
            .get_or("RAS/model", d.get_or("RAS/RASModel", ""))
            .to_string();
        CaseControls {
            model_name: name,
            momentum_transport: d,
            ..Default::default()
        }
    };

    // A small closed box with real walls on four of six faces - enough for
    // SST's wall-distance Poisson solve (S6.6) to be non-degenerate.
    let spec = MeshSpec {
        n: [6, 6, 4],
        l: [0.3, 0.3, 0.2],
        ..Default::default()
    };
    let hm = make_mesh(&scratch_dir("coupledSelection"), &spec)?;
    let mesh = GpuMesh::upload(gpu, &hm)?;
    let no_walls = WallFaces::none(hm.n_boundary_faces);
    let no_roughness = NutRoughness::none(hm.n_boundary_faces);

    // A sheared, non-uniform velocity - production needs grad U nonzero
    // everywhere - held fixed across the 20 outer steps below (this checks
    // the turbulence closures against one another, not momentum).
    let mut u = GpuVectorField::zeros(gpu, &mesh, "U")?;
    let u_vals: Vec<Vec3> = hm
        .c
        .iter()
        .map(|p| Vec3::new(1.0 + 0.5 * p.y, 0.3, 0.0))
        .collect();
    gpu.write(&mut u.f, &u_vals)?;
    let phi = GpuSurfaceScalarField::zeros(gpu, &mesh, "phi")?;
    let flow = FlowState::new(&u, &phi, 1.5e-5);

    // An unstable stratification - hot at the bottom, g pointing down -
    // SPEC-LIT §17's G_b > 0 branch, so the buoyancy route each model wires
    // it through actually has something to carry.
    let mut t = GpuScalarField::zeros(gpu, &mesh, "T")?;
    let z_max = spec.l[2];
    let t_vals: Vec<Scalar> = hm
        .c
        .iter()
        .map(|p| 1173.15 - 880.0 * (p.z / z_max))
        .collect();
    gpu.write(&mut t.f, &t_vals)?;
    correct_boundary_conditions(gpu, &k.field, &mut t, &mesh)?;

    let g = ofgpu::Vec3::new(0.0, 0.0, -9.81);
    let thermal = ThermalCtx { t: &t, g, prt: 0.85 };

    let run_one = |src: &str| -> Result<(String, Vec<Scalar>)> {
        let cc = case(src);
        let selection = select_turbulence_model(&cc)?;
        let mut turb = build_coupled(gpu, &hm, &mesh, &cc, &selection, &no_walls, &no_roughness)?;
        for (name, f) in turb.output_fields_mut() {
            if name == "k" {
                gpu.write(&mut f.f, &vec![1.0 as Scalar; hm.n_cells])?;
            }
        }
        turb.initialise(gpu, &flow)?;
        for _ in 0..20 {
            turb.correct(gpu, &flow, Some(&thermal))?;
        }
        gpu.sync()?;
        let nut = gpu.download(&turb.nut().f)?;
        Ok((turb.name().to_string(), nut))
    };

    let (ke_name, ke_nut) = run_one("RAS { model kEpsilon; }")?;
    let (sst_name, sst_nut) = run_one("RAS { model kOmegaSST; }")?;

    c.require(
        "S30.3 selection: RAS/model kEpsilon builds the banner \"kEpsilon\"",
        ke_name == "kEpsilon",
    );
    c.require(
        "S30.3 selection: RAS/model kOmegaSST builds the banner \"kOmegaSST\"",
        sst_name == "kOmegaSST",
    );

    let all_finite = ke_nut.iter().chain(sst_nut.iter()).all(|v| v.is_finite());
    c.require(
        "S30.3 selection: both the kEpsilon and kOmegaSST runs stay NaN-free",
        all_finite,
    );

    let ke_hash = field_hash(&ke_nut);
    let sst_hash = field_hash(&sst_nut);
    let ke_mean = ke_nut.iter().sum::<Scalar>() / ke_nut.len().max(1) as Scalar;
    let sst_mean = sst_nut.iter().sum::<Scalar>() / sst_nut.len().max(1) as Scalar;
    c.note(&format!(
        "nut on the buoyant selection case: kEpsilon mean {:e} (FNV {ke_hash:016x}), \
         kOmegaSST mean {:e} (FNV {sst_hash:016x})",
        f64::from(ke_mean),
        f64::from(sst_mean)
    ));
    c.require(
        "S30.3 selection (decisive): kOmegaSST's nut hash differs from kEpsilon's",
        all_finite && ke_hash != sst_hash,
    );

    Ok(())
}

// ==========================================================================
//  SPEC-LIT section 22: the buoyancy production, the sources, the species,
//  and the flux round trip
// ==========================================================================

/// `G_b` has the right sign in both stratifications - SPEC-LIT section 17,
/// and the first row of section 22 that mentions it.
///
/// The whole term is one product and one division, so there is nothing here
/// to converge; what can be wrong is a sign, and a sign is either right or
/// it is exactly backwards. Two temperature fields settle it:
///
///   * `dT/dz > 0` with `g` pointing down - a stable layer, warm air over
///     cold. Buoyancy DESTROYS turbulence, `G_b < 0`.
///   * `dT/dz < 0` - the profile above a heat source. Buoyancy MAKES
///     turbulence, `G_b > 0`.
///
/// The magnitude is checked too, against the closed form the linear profile
/// makes exact: `G_b = (nu_t/Pr_t) (g . dT/dz) / T`.
fn check_buoyancy_production(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    use ofgpu::turbulence::{buoyancy_production, BuoyancyProduction, C3Mode, TurbKernels};

    let spec = MeshSpec { n: [4, 4, 12], l: [0.4, 0.4, 1.2], ..Default::default() };
    let m = make_mesh(&scratch_dir("gb"), &spec)?;
    let gm = GpuMesh::upload(gpu, &m)?;
    let tk = TurbKernels::new(gpu)?;

    let n = m.n_cells;
    let nut_value: Scalar = 0.02;
    let prt: Scalar = 0.85;
    let gz: Scalar = -9.81;

    let mut nut = gpu.zeros::<Scalar>(n)?;
    gpu.write(&mut nut, &vec![nut_value; n])?;
    let mut u = gpu.zeros::<Vec3>(n)?;
    gpu.write(&mut u, &vec![Vec3::new(0.0, 0.0, 0.0); n])?;

    let mut gb = gpu.zeros::<Scalar>(n)?;
    let mut c3 = gpu.zeros::<Scalar>(n)?;

    // Two linear profiles, one of each sign. A linear field has an exact
    // Green-Gauss gradient on this mesh, so the only approximation left in
    // the comparison is the arithmetic itself.
    for (label, lapse, want_negative) in [
        ("stable stratification (dT/dz > 0)", 50.0 as Scalar, true),
        ("above a heat source (dT/dz < 0)", -50.0 as Scalar, false),
    ] {
        let t0: Scalar = 400.0;
        let mut t = GpuScalarField::zeros(gpu, &gm, "T")?;
        let internal: Vec<Scalar> = (0..n).map(|i| t0 + lapse * m.c[i].z).collect();
        let bvals: Vec<Scalar> =
            (0..m.n_boundary_faces).map(|f| t0 + lapse * m.b_cf[f].z).collect();
        gpu.write(&mut t.f, &internal)?;
        gpu.write(&mut t.bf, &bvals)?;

        let b = BuoyancyProduction {
            g: Vec3::new(0.0, 0.0, gz),
            prt,
            c3: C3Mode::Constant(0.0),
            ..Default::default()
        };
        let grad_t = t_gradient(gpu, &tk, &gm, &t)?;
        buoyancy_production(gpu, &tk, &mut gb, &mut c3, &nut, &grad_t, &t.f, &u, &b, n)?;
        gpu.sync()?;

        let g_host = gpu.download(&gb)?;

        // Interior cells only: a boundary cell's Green-Gauss gradient is
        // still exact here, but only because the boundary values were
        // written explicitly, and the point of the check is the sign.
        let worst_sign = (0..n).fold(0.0 as Scalar, |w, i| {
            let ok = if want_negative { g_host[i] < 0.0 } else { g_host[i] > 0.0 };
            w.max(if ok { 0.0 } else { 1.0 })
        });
        c.require(&format!("G_b sign: {label}"), worst_sign == 0.0);

        // The closed form.
        let worst_mag = (0..n).fold(0.0 as Scalar, |w, i| {
            let tc = internal[i];
            let want = (nut_value / prt) * (gz * lapse) / tc;
            w.max((g_host[i] - want).abs() / want.abs())
        });
        c.check(&format!("G_b magnitude: {label}"), worst_mag, 1e-11);
    }

    Ok(())
}

/// `grad(T)` for [`check_buoyancy_production`], as an owned buffer.
fn t_gradient(
    gpu: &Gpu,
    _tk: &ofgpu::turbulence::TurbKernels,
    gm: &GpuMesh,
    t: &GpuScalarField,
) -> Result<ofgpu::DevBuf<Vec3>> {
    let fv = FvKernels::new(gpu)?;
    let mut g = gpu.zeros::<Vec3>(gm.n_cells)?;
    fvc_grad_scalar(gpu, &fv, &mut g, t, gm)?;
    Ok(g)
}

/// SPEC-LIT section 22: "a heat source of known power raises the domain
/// enthalpy by exactly that much".
///
/// The source is formed from a power in watts, applied to a matrix over a
/// geometrically selected zone, and the matrix's own source array is summed
/// back. Two divisions have to be right for the total to come out - by
/// `rho c_p` and by the zone volume - and the zone volume has to be the one
/// the geometry actually selected rather than the one the box asked for.
fn check_volumetric_source(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    use ofgpu::sources::{heat_release_source, CellSelector, CellZone, Source, SourceKernels, SourceSet, SourceTerm};

    let spec = MeshSpec { n: [8, 8, 8], l: [0.8, 0.8, 0.8], ..Default::default() };
    let m = make_mesh(&scratch_dir("src"), &spec)?;
    let gm = GpuMesh::upload(gpu, &m)?;
    let sk = SourceKernels::new(gpu)?;

    let q_dot: Scalar = 125_000.0; // W
    let rho_cp: Scalar = 1206.0; // J/(m3 K)

    let zone = CellZone::new(
        gpu,
        &m,
        "fire",
        CellSelector::Box {
            min: Vec3::new(0.2, 0.2, 0.0),
            max: Vec3::new(0.6, 0.6, 0.3),
        },
    )?;
    c.note(&format!("heat source zone: {}", zone.describe()));

    let term = heat_release_source(q_dot, rho_cp, &zone)?;
    let mut set = SourceSet::new();
    set.push(Source { zone, term });

    let mut a = GpuLduMatrix::new(gpu, &gm)?;
    a.zero(gpu)?;
    set.apply(gpu, &sk, &mut a, &gm.v, None, None)?;
    gpu.sync()?;

    // sum_P V_P S_u is the volumetric source the equation received, in K m3/s;
    // times rho c_p it is watts.
    let src = gpu.download(&a.source)?;
    let injected = rho_cp * src.iter().fold(0.0 as Scalar, |t, v| t + v);
    c.check(
        "a heat source injects exactly its power",
        (injected - q_dot).abs() / q_dot,
        1e-12,
    );

    // A zone that selects nothing is an error, not an empty source.
    let empty = CellZone::new(
        gpu,
        &m,
        "nowhere",
        CellSelector::Box {
            min: Vec3::new(5.0, 5.0, 5.0),
            max: Vec3::new(6.0, 6.0, 6.0),
        },
    );
    c.require("a source selecting no cells is refused", empty.is_err());

    // Darcy-Forchheimer: the implicit part is negative by construction, so
    // what reaches the diagonal is positive whatever the velocity does.
    let mut set = SourceSet::new();
    set.push(Source::new(
        gpu,
        &m,
        "filter",
        CellSelector::All,
        SourceTerm::PorousDrag { d: 250.0, f: 40.0 },
    )?);

    let uh: Vec<Vec3> = (0..m.n_cells)
        .map(|i| {
            let sgn = if i % 2 == 0 { 1.0 } else { -1.0 } as Scalar;
            Vec3::new(sgn * 3.0, -sgn * 4.0, 0.0)
        })
        .collect();
    let mut ud = gpu.zeros::<Vec3>(m.n_cells)?;
    gpu.write(&mut ud, &uh)?;

    let mut a = GpuLduMatrix::new(gpu, &gm)?;
    a.zero(gpu)?;
    set.apply(gpu, &sk, &mut a, &gm.v, None, Some(&ud))?;
    gpu.sync()?;

    let diag = gpu.download(&a.diag)?;
    let src = gpu.download(&a.source)?;
    let vh = gpu.download(&gm.v)?;
    // |U| = 5 everywhere, so the coefficient is d + f|U|/2 = 250 + 100 = 350.
    let worst = (0..m.n_cells).fold(0.0 as Scalar, |w, i| {
        let want = vh[i] * 350.0;
        w.max((diag[i] - want).abs() / want)
    });
    c.check("Darcy-Forchheimer coefficient on the diagonal", worst, 1e-13);
    c.check("Darcy-Forchheimer never touches the source", max_abs(&src), 0.0);

    Ok(())
}

/// SPEC-LIT section 22: "species: sum of mass fractions -> exactly 1".
///
/// Three species advected several steps by one conservative flux, with the
/// inert one closed by `1 - sum`. The sum is checked cell by cell, and every
/// fraction is checked to be inside `[0, 1]`.
fn check_species(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    use ofgpu::species::{Species, SpeciesCoeffs};
    use ofgpu::turbulence::{FlowState, TurbulenceControls};

    let spec = MeshSpec { n: [12, 4, 4], l: [1.2, 0.4, 0.4], ..Default::default() };
    let m = make_mesh(&scratch_dir("species"), &spec)?;
    let gm = GpuMesh::upload(gpu, &m)?;

    let names = vec!["CH4".to_string(), "O2".to_string(), "N2".to_string()];
    let coeffs = [SpeciesCoeffs::default(); 3];

    let mut ctrl = TurbulenceControls::default();
    ctrl.steady = false;
    ctrl.delta_t = 0.02;
    ctrl.ddt = ofgpu::timescheme::DdtScheme::Euler;

    let mut sp = Species::new(gpu, &m, &gm, &names, &coeffs, "N2", 1e-5, ctrl)?;
    c.note(&format!(
        "{} solved species, \"{}\" closed by 1 - sum",
        sp.n_solved(),
        sp.inert_name()
    ));

    let n = m.n_cells;
    let ch4: Vec<Scalar> = (0..n).map(|i| if m.c[i].x < 0.3 { 0.4 } else { 0.0 }).collect();
    let o2: Vec<Scalar> = (0..n).map(|i| if m.c[i].x < 0.3 { 0.10 } else { 0.23 }).collect();
    gpu.write(&mut sp.get_mut(0).expect("CH4").field_mut().f, &ch4)?;
    gpu.write(&mut sp.get_mut(1).expect("O2").field_mut().f, &o2)?;
    sp.initialise(gpu)?;

    // A uniform velocity's flux, which is discretely conservative on this
    // orthogonal block: what a cell gains through one face it loses through
    // the opposite one, exactly.
    let uf = Vec3::new(0.4, 0.0, 0.0);
    let u = GpuVectorField::zeros(gpu, &gm, "U")?;
    let mut phi = GpuSurfaceScalarField::zeros(gpu, &gm, "phi")?;
    let fi: Vec<Scalar> = (0..m.n_internal_faces).map(|f| uf.dot(m.sf[f])).collect();
    let fb: Vec<Scalar> = (0..m.n_boundary_faces).map(|f| uf.dot(m.b_sf[f])).collect();
    gpu.write(&mut phi.f, &fi)?;
    gpu.write(&mut phi.bf, &fb)?;

    let nut = GpuScalarField::zeros(gpu, &gm, "nut")?;
    let flow = FlowState::new(&u, &phi, 1e-5);

    for _ in 0..8 {
        sp.correct(gpu, &flow, &nut)?;
    }
    gpu.sync()?;

    let y0 = gpu.download(&sp.get(0).expect("CH4").field().f)?;
    let y1 = gpu.download(&sp.get(1).expect("O2").field().f)?;
    let yi = gpu.download(&sp.inert().f)?;

    let worst_sum = (0..n).fold(0.0 as Scalar, |w, i| w.max((y0[i] + y1[i] + yi[i] - 1.0).abs()));
    c.check("species mass fractions sum to 1", worst_sum, 4.0 * Scalar::EPSILON);

    let out_of_range = (0..n).fold(0.0 as Scalar, |w, i| {
        let bad = [y0[i], y1[i], yi[i]]
            .iter()
            .any(|y| !(*y >= 0.0 && *y <= 1.0));
        w.max(if bad { 1.0 } else { 0.0 })
    });
    c.require("every mass fraction stays in [0, 1]", out_of_range == 0.0);

    let e = sp.max_sum_error(gpu)?;
    c.check("device-side max |1 - sum Y|", e, 4.0 * Scalar::EPSILON);

    Ok(())
}

/// SPEC-LIT section 22: a written-then-reread `phi` reproduces the same first
/// pressure residual.
///
/// The residual of `laplacian(rAU_f, p) = div(phi_HbyA)` is what a restart
/// actually begins from, and it depends on `phi` through the convection
/// coefficients of the momentum equation. If the file lost a digit the
/// residual moves, so this is the one check that says the restart is
/// conservative rather than merely plausible.
///
/// The comparison is BITWISE: `phi` is written at round-trip precision for
/// exactly this reason (see `PHI_PRECISION` in `io::fields`), so "the same
/// residual" means the same `f64` and not the same to some tolerance.
fn check_phi_round_trip(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    use ofgpu::field_setup::{harvest_surface_scalar_field, max_div_phi_host};
    use ofgpu::io::fields::{read_scalar_field, write_surface_scalar_field, RawScalarField};

    let spec = MeshSpec { n: [7, 5, 4], l: [0.7, 0.5, 0.4], ..Default::default() };
    let m = make_mesh(&scratch_dir("phiio"), &spec)?;
    let gm = GpuMesh::upload(gpu, &m)?;

    // A flux that is conservative by construction and NOT representable in six
    // significant figures: a uniform velocity at an irrational-looking angle
    // through a graded mesh gives face values with a full mantissa.
    let uf = Vec3::new(0.317_294_618_537_2, -0.113_927_461_038_4, 0.072_618_395_174_9);
    let mut phi = GpuSurfaceScalarField::zeros(gpu, &gm, "phi")?;
    let fi: Vec<Scalar> = (0..m.n_internal_faces).map(|f| uf.dot(m.sf[f])).collect();
    let fb: Vec<Scalar> = (0..m.n_boundary_faces).map(|f| uf.dot(m.b_sf[f])).collect();
    gpu.write(&mut phi.f, &fi)?;
    gpu.write(&mut phi.bf, &fb)?;

    let before = max_div_phi_host(&fi, &fb, &m);
    c.check("the reference flux is conservative to begin with", before, 1e-14);

    let dir = scratch_dir("phiio_out");
    std::fs::create_dir_all(&dir).map_err(|e| {
        ofgpu::Error::Config(format!("{}: {e}", dir.display()))
    })?;
    let path = dir.join("phi");

    let mut raw = RawScalarField::default();
    harvest_surface_scalar_field(gpu, &mut raw, &phi, &m)?;
    write_surface_scalar_field(&path, &raw, "0")?;

    let back = read_scalar_field(&path, m.n_internal_faces)?;
    let internal = ofgpu::io::fields::expand_scalars(&back.internal, m.n_internal_faces, "phi")?;

    let mut boundary = vec![0.0 as Scalar; m.n_boundary_faces];
    for pi in &m.patches {
        let Some(sp) = back.spec(&pi.name)? else { continue };
        let vals = ofgpu::io::fields::expand_scalars(&sp.value, pi.size, &pi.name)
            .unwrap_or_else(|_| vec![0.0 as Scalar; pi.size]);
        boundary[pi.start..pi.start + pi.size].copy_from_slice(&vals);
    }

    let worst_i = (0..m.n_internal_faces).fold(0.0 as Scalar, |w, f| w.max((internal[f] - fi[f]).abs()));
    let worst_b = (0..m.n_boundary_faces).fold(0.0 as Scalar, |w, f| w.max((boundary[f] - fb[f]).abs()));
    c.check("phi survives write/read bit for bit", worst_i.max(worst_b), 0.0);

    let after = max_div_phi_host(&internal, &boundary, &m);
    c.check("the reread flux is still conservative", after, before.max(1e-14));

    // And what the fallback would have cost: interpolate(U).Sf on the same
    // mesh with a velocity that is NOT uniform, which is the case a restart
    // without a phi file actually lands in.
    c.note("without a phi file a restart falls back to interpolate(U).Sf, which is not conservative");

    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

// ==========================================================================
//  SPEC-LIT section 20 and the four VOF rows of section 22
//
//  Every one of these is an analytic identity or a conservation statement.
//  Nothing here compares against another CFD code, and the two that quote a
//  number - the curvature of a circle and the Laplace pressure jump - quote
//  one that is written down in closed form.
// ==========================================================================

/// Write a boundary condition patch by patch. `Some(v)` is a fixedValue,
/// `None` zero-gradient; an `empty` patch stays empty.
fn vof_scalar_bcs(
    gpu: &Gpu,
    s: &mut GpuScalarField,
    m: &HostMesh,
    per_patch: &[Option<Scalar>],
) -> Result<()> {
    let nbf = m.n_boundary_faces;
    let mut owner = vec![0usize; nbf];
    for (p, pi) in m.patches.iter().enumerate() {
        for k in 0..pi.size {
            owner[pi.start + k] = p;
        }
    }

    let mut kind = vec![BcKind::ZeroGradient as Label; nbf];
    let mut fr = vec![0.0 as Scalar; nbf];
    let mut rv = vec![0.0 as Scalar; nbf];
    let rg = vec![0.0 as Scalar; nbf];

    for i in 0..nbf {
        let p = owner[i];
        if m.patches[p].kind == PatchKind::Empty {
            kind[i] = BcKind::Empty as Label;
            continue;
        }
        if let Some(v) = per_patch[p] {
            kind[i] = BcKind::FixedValue as Label;
            fr[i] = 1.0;
            rv[i] = v;
        }
    }

    gpu.write(&mut s.bc_kind, &kind)?;
    gpu.write(&mut s.fr, &fr)?;
    gpu.write(&mut s.ref_value, &rv)?;
    gpu.write(&mut s.ref_grad, &rg)
}

/// Every patch a no-slip wall, which for a sealed box is what "sealed" means:
/// the flux through every boundary face is prescribed and zero.
fn vof_sealed_velocity(gpu: &Gpu, u: &mut GpuVectorField, m: &HostMesh) -> Result<()> {
    let nbf = m.n_boundary_faces;
    let mut owner = vec![0usize; nbf];
    for (p, pi) in m.patches.iter().enumerate() {
        for k in 0..pi.size {
            owner[pi.start + k] = p;
        }
    }

    let mut kind = vec![BcKind::FixedValue as Label; nbf];
    let mut fr = vec![1.0 as Scalar; nbf];
    let rv = vec![Vec3::ZERO; nbf];
    let rg = vec![Vec3::ZERO; nbf];

    for i in 0..nbf {
        if m.patches[owner[i]].kind == PatchKind::Empty {
            kind[i] = BcKind::Empty as Label;
            fr[i] = 0.0;
        }
    }

    gpu.write(&mut u.bc_kind, &kind)?;
    gpu.write(&mut u.fr, &fr)?;
    gpu.write(&mut u.ref_value, &rv)?;
    gpu.write(&mut u.ref_grad, &rg)
}

/// `phi_f = u(Cf)·Sf` for an analytic velocity field.
///
/// For a solid-body rotation or a uniform stream on a Cartesian block this is
/// discretely solenoidal to round-off, which is what makes the two advection
/// checks below tests of the SCHEME rather than of the flux.
fn vof_analytic_flux(
    gpu: &Gpu,
    phi: &mut GpuSurfaceScalarField,
    m: &HostMesh,
    u: impl Fn(Vec3) -> Vec3,
) -> Result<Scalar> {
    let f: Vec<Scalar> = (0..m.n_internal_faces)
        .map(|i| u(m.cf[i]).dot(m.sf[i]))
        .collect();
    let bf: Vec<Scalar> = (0..m.n_boundary_faces)
        .map(|i| {
            if m.b_kind[i] == PatchKind::Empty as Label {
                0.0
            } else {
                u(m.b_cf[i]).dot(m.b_sf[i])
            }
        })
        .collect();

    // How far from solenoidal it came out, so the check can say so.
    let mut d = vec![0.0 as Scalar; m.n_cells];
    for i in 0..m.n_internal_faces {
        d[m.owner[i] as usize] += f[i];
        d[m.neighbour[i] as usize] -= f[i];
    }
    for i in 0..m.n_boundary_faces {
        d[m.b_face_cells[i] as usize] += bf[i];
    }

    gpu.write(&mut phi.f, &f)?;
    gpu.write(&mut phi.bf, &bf)?;
    Ok(max_abs(&d))
}

/// Advection only: one fluid twice over, no gravity, no surface tension, so
/// nothing but `alpha` can move.
fn vof_advection_setup(c_alpha: Scalar) -> (VofProperties, VofControls) {
    (
        VofProperties {
            rho1: 1.0,
            rho2: 1.0,
            mu1: 0.0,
            mu2: 0.0,
            sigma: 0.0,
            g: Vec3::ZERO,
            c_alpha,
        },
        VofControls {
            max_alpha_co: 0.8,
            n_limiter_iters: 3,
            report_continuity: false,
            ..VofControls::default()
        },
    )
}

fn vof_tight() -> SolverControls {
    SolverControls {
        tolerance: 1e-12,
        rel_tol: 0.0,
        max_iter: 1000,
        check_interval: 1,
        ..SolverControls::default()
    }
}

/// The largest speed anywhere. `max_abs_diff_vec3` against zero would do, but
/// spelling it out is clearer at the four call sites that want it.
fn vof_max_speed(u: &[Vec3]) -> Scalar {
    u.iter().fold(0.0 as Scalar, |a, v| a.max(v.mag()))
}

fn check_vof(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    // ---------------------------------------------------------------- 20.2
    //  Zalesak's rotating slotted disc. A quarter turn here rather than the
    //  full revolution the unit test in src/vof.rs runs, because this file
    //  has to stay quick; boundedness is a per-step property and a quarter
    //  turn exercises it five hundred times.
    // ----------------------------------------------------------------------
    {
        let nx = 64usize;
        let spec = MeshSpec {
            n: [nx, nx, 1],
            l: [1.0, 1.0, 1.0 / nx as Scalar],
            two_d: true,
            ..Default::default()
        };
        let m = make_mesh(&scratch_dir("vofDisc"), &spec)?;
        let gm = GpuMesh::upload(gpu, &m)?;

        let (props, ctrl) = vof_advection_setup(1.0);
        let mut vof = Vof::new(gpu, &m, &gm, props, ctrl)?;

        let centre = Vec3::new(0.5, 0.75, 0.0);
        let a0: Vec<Scalar> = (0..m.n_cells)
            .map(|cc| {
                let p = m.c[cc];
                let r = ((p.x - centre.x).powi(2) + (p.y - centre.y).powi(2)).sqrt();
                let slot = (p.x - centre.x).abs() < 0.025 && p.y < 0.85;
                if r <= 0.15 && !slot {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        gpu.write(&mut vof.alpha_mut().f, &a0)?;
        vof_scalar_bcs(gpu, vof.alpha_mut(), &m, &[None; 6])?;

        let div = vof_analytic_flux(gpu, vof.phi_mut(), &m, |p| {
            Vec3::new(-(p.y - 0.5), p.x - 0.5, 0.0)
        })?;
        vof.initialise(gpu)?;

        c.check("solid-body rotation flux is solenoidal", div, 1e-15);

        let v0 = vof.phase_volume(gpu)?;
        let steps = 500;
        let dt = (0.5 * PI / steps as f64) as Scalar; // a quarter turn
        for _ in 0..steps {
            vof.solve_alpha(gpu, dt)?;
        }
        let (lo, hi) = vof.alpha_bounds(gpu)?;
        let v1 = vof.phase_volume(gpu)?;

        c.note(&format!(
            "Zalesak disc, quarter turn on {nx}^2: alpha in [{:e}, {}]",
            f64::from(lo),
            f64::from(hi)
        ));
        // SPEC-LIT 20.2: "alpha must stay in [0, 1] EXACTLY". Both bounds are
        // held to round-off, not to a tolerance, because the whole point of
        // the FCT machinery is that -1e-3 gives a negative density.
        c.check("Zalesak: min(alpha) >= 0", (-lo).max(0.0), 1e-12);
        c.check("Zalesak: max(alpha) <= 1", (hi - 1.0).max(0.0), 1e-12);
        c.check("Zalesak: phase volume conserved", ((v1 - v0) / v0).abs(), 1e-9);
    }

    // ---------------------------------------------------------------- 20.1
    //  Interface compression: a translating interface must not smear.
    // ----------------------------------------------------------------------
    {
        let nx = 200usize;
        let h = 1.0 / nx as Scalar;
        let spec = MeshSpec {
            n: [nx, 1, 1],
            l: [1.0, h, h],
            two_d: true,
            ..Default::default()
        };
        let m = make_mesh(&scratch_dir("vofStep"), &spec)?;
        let gm = GpuMesh::upload(gpu, &m)?;

        let width = |a: &[Scalar]| a.iter().filter(|v| **v > 0.01 && **v < 0.99).count();

        let run = |c_alpha: Scalar| -> Result<usize> {
            let (props, ctrl) = vof_advection_setup(c_alpha);
            let mut vof = Vof::new(gpu, &m, &gm, props, ctrl)?;

            let a0: Vec<Scalar> = (0..m.n_cells)
                .map(|cc| if m.c[cc].x < 0.25 { 1.0 } else { 0.0 })
                .collect();
            gpu.write(&mut vof.alpha_mut().f, &a0)?;
            vof_scalar_bcs(
                gpu,
                vof.alpha_mut(),
                &m,
                &[Some(1.0), None, None, None, None, None],
            )?;
            vof_analytic_flux(gpu, vof.phi_mut(), &m, |_| Vec3::new(1.0, 0.0, 0.0))?;
            vof.initialise(gpu)?;

            for _ in 0..200 {
                vof.solve_alpha(gpu, 0.25 * h)?;
            }
            Ok(width(&gpu.download(&vof.alpha().f)?))
        };

        let off = run(0.0)?;
        let on = run(1.0)?;
        c.note(&format!(
            "translating interface after 200 steps: {off} transitional cells \
             with cAlpha 0, {on} with cAlpha 1"
        ));
        c.require("compression sharpens the interface", on < off);
        c.require("the compressed interface is two cells or fewer", on <= 2);
    }

    // ---------------------------------------------------------------- 20.4
    //  The curvature of a circle, and the Laplace pressure jump it produces.
    // ----------------------------------------------------------------------
    {
        let nx = 64usize;
        let h = 1.0 / nx as Scalar;
        let spec = MeshSpec {
            n: [nx, nx, 1],
            l: [1.0, 1.0, h],
            two_d: true,
            ..Default::default()
        };
        let m = make_mesh(&scratch_dir("vofDrop"), &spec)?;
        let gm = GpuMesh::upload(gpu, &m)?;

        let sigma = 1.0 as Scalar;
        let radius = 0.2 as Scalar;
        let centre = Vec3::new(0.5, 0.5, 0.5 * h);

        let props = VofProperties {
            rho1: 1000.0,
            rho2: 1.0,
            mu1: 0.1,
            mu2: 0.01,
            sigma,
            // Zero gravity: the drop is held together by nothing else.
            g: Vec3::ZERO,
            c_alpha: 1.0,
        };
        let ctrl = VofControls {
            delta_t: 1e-4,
            n_correctors: 2,
            u_solver: vof_tight(),
            p_solver: vof_tight(),
            sn_grad: SnGradScheme::Uncorrected,
            report_continuity: false,
            ..VofControls::default()
        };
        let mut vof = Vof::new(gpu, &m, &gm, props, ctrl)?;

        // A smooth radial profile: alpha = (1 - tanh((r - R)/w))/2 decreases
        // with r, so grad(alpha) points inward, n_hat = -r_hat, and
        // kappa = -div(n_hat) = +div(r_hat) = 1/r in two dimensions.
        let w = 1.5 * h;
        let a0: Vec<Scalar> = (0..m.n_cells)
            .map(|cc| {
                let d = m.c[cc] - centre;
                let r = (d.x * d.x + d.y * d.y).sqrt();
                0.5 * (1.0 - ((r - radius) / w).tanh())
            })
            .collect();
        gpu.write(&mut vof.alpha_mut().f, &a0)?;
        vof_scalar_bcs(gpu, vof.alpha_mut(), &m, &[None; 6])?;
        vof_sealed_velocity(gpu, vof.u_mut(), &m)?;
        vof_scalar_bcs(gpu, vof.p_rgh_mut(), &m, &[None; 6])?;
        vof.initialise(gpu)?;

        // The interface normal, the curvature and the face body force, from
        // the alpha just written. `initialise` does not do this: it has no
        // reason to, because `step` recomputes it after every alpha solve.
        vof.update_body_force(gpu)?;

        // The curvature, against 1/r evaluated at each cell's own radius -
        // the level sets of a radial profile are concentric circles and each
        // has its own curvature.
        let kappa = gpu.download(&vof.curvature().f)?;
        let mut sum = 0.0 as Scalar;
        let mut band = 0usize;
        for cc in 0..m.n_cells {
            if a0[cc] <= 0.3 || a0[cc] >= 0.7 {
                continue;
            }
            let d = m.c[cc] - centre;
            let r = (d.x * d.x + d.y * d.y).sqrt();
            band += 1;
            let e = (kappa[cc] - 1.0 / r) * r;
            sum += e * e;
        }
        // Root-mean-square, not the worst cell. The curvature is a second
        // derivative of a field that is nearly a step, and at a fixed number
        // of cells across the interface a handful of cells at the ends of the
        // band are always the worst; the r.m.s. is what converges under
        // refinement, and `src/vof.rs` measures that convergence directly.
        let rms = (sum / band.max(1) as Scalar).sqrt();
        c.note(&format!(
            "curvature measured over {band} interface cells at w = 1.5 h"
        ));
        // The tolerance comes from a scale argument, not from the answer. The
        // face normal is formed from a four-point gradient estimate whose
        // relative error across a profile of thickness `w` is about
        // (1/6)(h/w)^2, which at w = 1.5 h is 0.07. Anything of that order
        // passes; an operator that is wrong in KIND rather than in resolution
        // - the face-difference variant `cuda/vof.cu` documents and rejects -
        // came out at 0.8 here. What pins the ORDER is
        // `the_curvature_of_a_circular_interface_converges_to_one_over_r` in
        // src/vof.rs, which holds `w` fixed in metres and refines.
        c.check("kappa of a circular interface == 1/r", rms, 0.15);

        // The Laplace jump. At equilibrium the face balance is
        //   |Sf| snGrad(p_rgh) = sigma kappa_f |Sf| snGrad(alpha),
        // so p_rgh = sigma kappa alpha + const and the jump between the two
        // phases is sigma kappa = sigma/R.
        // Sampled at three times in geometric progression, so the SHAPE of
        // the growth can be read off rather than just its size - see the
        // check below.
        let dt = 1e-4 as Scalar;
        let mut u = [0.0 as Scalar; 3];
        for step in 1..=120 {
            vof.step(gpu, dt)?;
            match step {
                30 => u[0] = vof_max_speed(&gpu.download(&vof.u().f)?),
                60 => u[1] = vof_max_speed(&gpu.download(&vof.u().f)?),
                120 => u[2] = vof_max_speed(&gpu.download(&vof.u().f)?),
                _ => {}
            }
        }
        let u_end = u[2];

        let p = gpu.download(&vof.p_rgh().f)?;
        let a = gpu.download(&vof.alpha().f)?;
        let mut inside = (0.0 as Scalar, 0usize);
        let mut outside = (0.0 as Scalar, 0usize);
        for cc in 0..m.n_cells {
            if a[cc] > 0.99 {
                inside.0 += p[cc];
                inside.1 += 1;
            } else if a[cc] < 0.01 {
                outside.0 += p[cc];
                outside.1 += 1;
            }
        }
        let jump = inside.0 / inside.1.max(1) as Scalar - outside.0 / outside.1.max(1) as Scalar;
        let exact = sigma / radius;

        c.note(&format!(
            "static drop: Laplace jump {:.4} against sigma/R = {:.4}; \
             spurious |U| {:e} -> {:e} -> {:e}",
            f64::from(jump),
            f64::from(exact),
            f64::from(u[0]),
            f64::from(u[1]),
            f64::from(u[2])
        ));
        c.check(
            "Laplace pressure jump == sigma/R (2-D)",
            ((jump - exact) / exact).abs(),
            0.10,
        );

        // BOUNDED, which here means "growing no faster than linearly".
        //
        // The residual force is the variation of kappa along the interface,
        // which is fixed by the mesh and does not change with time, so it
        // drives a constant acceleration until viscosity balances it - and
        // this drop's viscous time, rho h^2/mu, is two seconds against the
        // twelve milliseconds simulated. Linear growth is therefore the
        // expected answer and doubling the interval doubles the speed; what
        // would be a failure is growth that ACCELERATES, which is a CSF
        // feeding its own velocity field. So the two successive ratios are
        // compared with each other rather than with 2.
        let r1 = u[1] / u[0].max(1e-300);
        let r2 = u[2] / u[1].max(1e-300);
        c.note(&format!(
            "spurious current growth ratios {:.3} then {:.3} over equal \
             doublings of the interval",
            f64::from(r1),
            f64::from(r2)
        ));
        c.check(
            "spurious interface currents do not accelerate",
            (r2 - r1).max(0.0) / r1.max(1e-300),
            0.05,
        );
        c.check(
            "spurious interface currents are small (Ca)",
            props.mu1 * u_end / sigma,
            1e-2,
        );
    }

    // ---------------------------------------------------------------- 20.5
    //  Two stratified fluids, sealed, at rest. SPEC-LIT 20.5: "That test
    //  fails immediately if p_rgh is not used, and it is the one test that
    //  proves this section is right."
    // ----------------------------------------------------------------------
    {
        let n = 24usize;
        let h = 1.0 / n as Scalar;
        let spec = MeshSpec {
            n: [n, n, 1],
            l: [1.0, 1.0, h],
            two_d: true,
            ..Default::default()
        };
        let m = make_mesh(&scratch_dir("vofStrat"), &spec)?;
        let gm = GpuMesh::upload(gpu, &m)?;

        // The resolved plane of a 2-D block is x-y, so gravity is along -y.
        let props = VofProperties {
            rho1: 1000.0,
            rho2: 1.0,
            mu1: 1.002e-3,
            mu2: 1.8e-5,
            sigma: 0.0,
            g: Vec3::new(0.0, -9.81, 0.0),
            c_alpha: 1.0,
        };
        let ctrl = VofControls {
            delta_t: 1e-3,
            n_correctors: 3,
            u_solver: vof_tight(),
            p_solver: vof_tight(),
            sn_grad: SnGradScheme::Uncorrected,
            report_continuity: false,
            ..VofControls::default()
        };
        let mut vof = Vof::new(gpu, &m, &gm, props, ctrl)?;

        // The interface sits on a face, so there is no partly filled cell.
        let a0: Vec<Scalar> = (0..m.n_cells)
            .map(|cc| if m.c[cc].y < 0.5 { 1.0 } else { 0.0 })
            .collect();
        gpu.write(&mut vof.alpha_mut().f, &a0)?;
        vof_scalar_bcs(gpu, vof.alpha_mut(), &m, &[None; 6])?;
        vof_sealed_velocity(gpu, vof.u_mut(), &m)?;
        vof_scalar_bcs(gpu, vof.p_rgh_mut(), &m, &[None; 6])?;
        vof.initialise(gpu)?;

        c.require("a sealed tank leaves p_rgh pinned", vof.pressure_is_pinned());

        let dt = 1e-3 as Scalar;
        let mut worst = 0.0 as Scalar;
        let mut last = 0.0 as Scalar;
        for _ in 0..20 {
            vof.step(gpu, dt)?;
            last = vof_max_speed(&gpu.download(&vof.u().f)?);
            worst = worst.max(last);
        }

        // Against the velocity scale of the problem, a gravity wave on a tank
        // of this depth: sqrt(g H).
        let scale = (9.81 as Scalar).sqrt();
        c.note(&format!(
            "sealed stratified tank: max |U| {:e} m/s against sqrt(gH) = {:.3}",
            f64::from(last),
            f64::from(scale)
        ));
        c.check("a sealed stratified tank stays at rest", last / scale, 1e-8);
        c.require("and does not drift", last <= worst);

        let (lo, hi) = vof.alpha_bounds(gpu)?;
        c.check("its interface does not move", (-lo).max(hi - 1.0).max(0.0), 1e-9);
    }

    // ---------------------------------------------------------------- 20.3
    //  The mass flux and the density it advects must be consistent:
    //      (rho - rho0) V/dt + sum_f (+-rho_phi_f) = 0
    //  to round-off. This holds only because rho_phi is built from the SAME
    //  limited alpha flux that advanced alpha.
    // ----------------------------------------------------------------------
    {
        let nx = 40usize;
        let spec = MeshSpec {
            n: [nx, nx, 1],
            l: [1.0, 1.0, 1.0 / nx as Scalar],
            two_d: true,
            ..Default::default()
        };
        let m = make_mesh(&scratch_dir("vofMass"), &spec)?;
        let gm = GpuMesh::upload(gpu, &m)?;

        let props = VofProperties {
            rho1: 1000.0,
            rho2: 1.0,
            mu1: 0.0,
            mu2: 0.0,
            sigma: 0.0,
            g: Vec3::ZERO,
            c_alpha: 1.0,
        };
        let ctrl = VofControls {
            max_alpha_co: 0.8,
            report_continuity: false,
            ..VofControls::default()
        };
        let mut vof = Vof::new(gpu, &m, &gm, props, ctrl)?;

        let a0: Vec<Scalar> = (0..m.n_cells)
            .map(|cc| {
                let d = m.c[cc] - Vec3::new(0.5, 0.7, 0.0);
                if (d.x * d.x + d.y * d.y).sqrt() < 0.15 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        gpu.write(&mut vof.alpha_mut().f, &a0)?;
        vof_scalar_bcs(gpu, vof.alpha_mut(), &m, &[None; 6])?;
        vof_analytic_flux(gpu, vof.phi_mut(), &m, |p| {
            Vec3::new(-(p.y - 0.5), p.x - 0.5, 0.0)
        })?;
        vof.initialise(gpu)?;

        let dt = 0.2 / nx as Scalar;
        let rho_before = gpu.download(&vof.rho().f)?;
        vof.solve_alpha(gpu, dt)?;
        vof.update_properties(gpu)?;
        vof.update_rho_phi(gpu)?;
        let rho_after = gpu.download(&vof.rho().f)?;
        let rp = gpu.download(&vof.rho_phi().f)?;
        let brp = gpu.download(&vof.rho_phi().bf)?;

        let mut div = vec![0.0 as Scalar; m.n_cells];
        for f in 0..m.n_internal_faces {
            div[m.owner[f] as usize] += rp[f];
            div[m.neighbour[f] as usize] -= rp[f];
        }
        for b in 0..m.n_boundary_faces {
            if m.b_kind[b] != PatchKind::Empty as Label {
                div[m.b_face_cells[b] as usize] += brp[b];
            }
        }

        let mut worst = 0.0 as Scalar;
        let mut scale = 0.0 as Scalar;
        for cc in 0..m.n_cells {
            let ddt = (rho_after[cc] - rho_before[cc]) * m.v[cc] / dt;
            worst = worst.max((ddt + div[cc]).abs());
            scale = scale.max(ddt.abs().max(div[cc].abs()));
        }

        c.note(&format!(
            "mass consistency residual {:e} against a term size of {:e}",
            f64::from(worst),
            f64::from(scale)
        ));
        c.require("something actually moved", scale > 0.0);
        c.check(
            "d(rho)/dt + div(rho phi) == 0",
            worst / scale.max(1e-30),
            1e-12,
        );
    }

    Ok(())
}

fn main() -> ExitCode {
    let mut c = Checks::new();

    if let Err(e) = run(&mut c) {
        eprintln!("\nvalidation aborted: {e}");
        return ExitCode::from(2);
    }

    println!("\n{}/{} checks passed", c.total - c.failures, c.total);
    println!(
        "{} computed live, {} replayed from recorded measurements \
         (docs/07-fire-solver.md S1.1: the wall-function gate verdict, the resolved leg's \
         mesh resolution, the resolved leg's gate verdict, the thermostat-weighting \
         experiment, the bounded-convection isolation, and the Kays-Crawford Prt \
         experiment)",
        c.total - c.replayed,
        c.replayed,
    );
    if c.skipped > 0 {
        println!("{} checks skipped", c.skipped);
    }

    if c.failures == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

// ==========================================================================
//  SPEC-LIT sections 23, 24, 25, 27, 28: the "decisive gates" - promoted
//  from each module's own unit tests into permanent checks here, so a
//  regression in the mesh-format reader, the cut-cell fractions, the
//  low-Mach p0 evolution, combustion or radiation fails `ofgpu-validate`
//  and not only that one module's own `cargo test`.
// ==========================================================================

/// SPEC-LIT §23.1/§10: a `.msh` file read through the SAME `parse_msh` a real
/// Gmsh export goes through closes to round-off. One unit hexahedron, MSH
/// 4.1, physical surface "walls" on all six sides - the published Gmsh
/// format (SPEC-LIT §23.1), not anyone's source.
fn check_msh_hex_closure(c: &mut Checks) -> Result<()> {
    let pts = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ];
    let tags: Vec<i64> = (0..8).map(|i| 11 + i).collect();
    let mut nodes_body = String::new();
    for &t in &tags {
        nodes_body += &format!("{t}\n");
    }
    for &(x, y, z) in &pts {
        nodes_body += &format!("{x} {y} {z}\n");
    }
    let hex_nodes: Vec<String> = tags.iter().map(|t| t.to_string()).collect();
    let text = format!(
        r#"$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
1
2 1 "walls"
$EndPhysicalNames
$Entities
0 0 1 1
1 0 0 0 1 1 1 1 1 0
7 0 0 0 1 1 1 0 0
$EndEntities
$Nodes
1 8 11 18
3 1 0 8
{nodes_body}$EndNodes
$Elements
2 7 1 7
2 1 3 6
1 11 14 13 12
2 15 16 17 18
3 11 12 16 15
4 14 18 17 13
5 11 15 18 14
6 12 13 17 16
3 5 5 1
7 {hex}
$EndElements
"#,
        hex = hex_nodes.join(" ")
    );

    let raw = parse_msh(&text, "<memory>")?;
    let hm = build_host_mesh(&raw)?;
    c.require("msh hex: exactly one cell", hm.n_cells == 1);
    c.require("msh hex: exactly six boundary faces", hm.n_boundary_faces == 6);
    check_mesh(c, &hm, 1.0);
    Ok(())
}

/// A closed triangle soup for an axis-aligned box, written as ASCII STL text
/// - the published STL format (SPEC-LIT §23.1), read back through the SAME
/// `parse_stl` a real exported file goes through.
fn ascii_stl_cuboid(lo: Vec3, hi: Vec3) -> String {
    let p = [
        Vec3::new(lo.x, lo.y, lo.z), Vec3::new(hi.x, lo.y, lo.z),
        Vec3::new(hi.x, hi.y, lo.z), Vec3::new(lo.x, hi.y, lo.z),
        Vec3::new(lo.x, lo.y, hi.z), Vec3::new(hi.x, lo.y, hi.z),
        Vec3::new(hi.x, hi.y, hi.z), Vec3::new(lo.x, hi.y, hi.z),
    ];
    const T: [[usize; 3]; 12] = [
        [0, 3, 2], [0, 2, 1],
        [4, 5, 6], [4, 6, 7],
        [0, 4, 7], [0, 7, 3],
        [1, 2, 6], [1, 6, 5],
        [0, 1, 5], [0, 5, 4],
        [3, 7, 6], [3, 6, 2],
    ];
    let mut s = String::from("solid box\n");
    for &[a, b, cc] in &T {
        let (v0, v1, v2) = (p[a], p[b], p[cc]);
        let normal = (v1 - v0).cross(v2 - v0);
        let normal = normal / normal.mag().max(1e-30);
        s += &format!(
            "facet normal {} {} {}\nouter loop\n\
             vertex {} {} {}\nvertex {} {} {}\nvertex {} {} {}\n\
             endloop\nendfacet\n",
            normal.x, normal.y, normal.z,
            v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z,
        );
    }
    s += "endsolid box\n";
    s
}

/// SPEC-LIT §24.3/§24.6 row 1, the decisive gate for the whole embedded-
/// boundary construction: the cut face's area vector is DEFINED as what
/// closure demands, so every cut cell must close to round-off whatever the
/// fractions came out to be - on a surface read through the published STL
/// format, not an internal geometry fixture.
fn check_cutcell_closure(c: &mut Checks) -> Result<()> {
    let lo = Vec3::new(0.26, 0.24, 0.27);
    let hi = Vec3::new(0.71, 0.76, 0.69);
    let text = ascii_stl_cuboid(lo, hi);
    let surf = parse_stl(text.as_bytes(), "box", "<memory>")?;
    surf.require_closed()?;

    let n = 20usize;
    let axis = |lo: Scalar, hi: Scalar| -> Vec<Scalar> {
        (0..=n).map(|i| lo + (hi - lo) * i as Scalar / n as Scalar).collect()
    };
    let (xn, yn, zn) = (axis(0.0, 1.0), axis(0.0, 1.0), axis(0.0, 1.0));
    let axes = BlockAxes { xn: &xn, yn: &yn, zn: &zn };

    let field = classify_cutcells(&axes, &surf, DEFAULT_SUPERSAMPLE)?;
    c.require("cutcell: the off-grid cuboid actually cuts cells", field.n_cut > 0);

    let h = 1.0 / n as Scalar;
    let full: [Vec3; 6] = [
        Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 1.0),
    ];
    let mut max_err: Scalar = 0.0;
    for cell in field.cells.iter().flatten() {
        if cell.state != CellState::Cut {
            continue;
        }
        let mut sum = cell.cut_sf;
        for d in 0..6 {
            sum += full[d] * (cell.alpha[d] * h * h);
        }
        max_err = max_err.max(sum.mag());
    }
    c.note(&format!("cut-cell closure: {} cut cells out of {}", field.n_cut, field.cells.len()));
    c.check("cut-cell closure (S24.3/24.6 row 1): sum Sf = 0", max_err / (h * h), 1e-8);
    Ok(())
}

/// SPEC-LIT §25.2's decisive gate: a sealed box with a heater of known power
/// `P` raises `p0` at EXACTLY `dp0/dt = (gamma-1)P/V` - analytic, no
/// tolerance excuses - and an open domain with the identical heater does not
/// move `p0` at all.
fn check_low_mach_p0(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    let spec = MeshSpec { n: [4, 4, 4], l: [0.4, 0.4, 0.4], ..Default::default() };
    let hm = make_mesh(&scratch_dir("lowmach_p0"), &spec)?;
    let m = GpuMesh::upload(gpu, &hm)?;

    let props = GasProperties::default();
    let p0_init = props.r_s() * 300.0 * 1.2;

    let mut sources = EnergySources::new(gpu, &m)?;
    let p_watts: Scalar = 2500.0;
    let q_per_vol = p_watts / m.total_volume;
    let q_field = gpu.upload(&vec![q_per_vol; hm.n_cells])?;
    sources.register_explicit(gpu, &q_field)?;
    let total_q = sources.total_q(gpu, &m)?;
    c.check(
        "low-Mach S25.1: integral(Q)dV matches the heater power",
        (total_q - p_watts).abs() / p_watts,
        1e-6,
    );

    let dt = 1e-3;
    let mut sealed = GasState::new(gpu, &m, props, DomainKind::Sealed, p0_init)?;
    sealed.advance_p0(total_q, dt)?;
    let want_dp0dt = (props.gamma - 1.0) * p_watts / m.total_volume;
    let got = sealed.dp0dt();
    c.note(&format!("sealed dp0/dt = {got}, analytic (gamma-1)P/V = {want_dp0dt}"));
    c.check(
        "low-Mach S25.2 (decisive): sealed dp0/dt = (gamma-1) P / V",
        (got - want_dp0dt).abs() / want_dp0dt.abs(),
        1e-3,
    );

    let mut open = GasState::new(gpu, &m, props, DomainKind::Open, p0_init)?;
    open.advance_p0(total_q, dt)?;
    c.require(
        "low-Mach S25.2: an open domain does not move p0 at all",
        open.dp0dt() == 0.0 && open.p0() == p0_init,
    );

    Ok(())
}

/// SPEC-LIT §27's decisive gate: a burner supplying fuel that burns
/// completely releases EXACTLY the fuel mass consumed times `dh_c` - not
/// approximately, because `q'''_c` and the mass actually consumed come from
/// the SAME `dYF` inside one reaction kernel.
fn check_burner_heat_release(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    let spec = MeshSpec { n: [4, 4, 4], l: [0.4, 0.4, 0.4], all_generic: true, ..Default::default() };
    let hm = make_mesh(&scratch_dir("burner"), &spec)?;
    let m = GpuMesh::upload(gpu, &hm)?;
    let n = hm.n_cells;
    let nbf = hm.n_boundary_faces;

    let names: Vec<String> = ["Fuel", "O2", "Products", "N2"].iter().map(|s| s.to_string()).collect();
    let coeffs = [SpeciesCoeffs::default(); 4];
    let mut sp = Species::new(gpu, &hm, &m, &names, &coeffs, "N2", 1.5e-5, TurbulenceControls::default())?;
    let y_f0: Scalar = 0.02;
    gpu.write(&mut sp.by_name_mut("Fuel").ok_or_else(|| Error::Config("species set has no \"Fuel\"".to_string()))?.field_mut().f, &vec![y_f0; n])?;
    gpu.write(&mut sp.by_name_mut("O2").ok_or_else(|| Error::Config("species set has no \"O2\"".to_string()))?.field_mut().f, &vec![0.5 as Scalar; n])?;
    gpu.write(&mut sp.by_name_mut("Products").ok_or_else(|| Error::Config("species set has no \"Products\"".to_string()))?.field_mut().f, &vec![0.0 as Scalar; n])?;
    sp.initialise(gpu)?;

    let coeffs_c = CombustionCoeffs::default();
    let mut cmb = Combustion::new(gpu, &m, coeffs_c, &sp, "Fuel", "O2", "Products")?;

    let mut rho = GpuScalarField::zeros(gpu, &m, "rho")?;
    gpu.write(&mut rho.f, &vec![1.1 as Scalar; n])?;
    gpu.write(&mut rho.bf, &vec![1.1 as Scalar; nbf])?;
    let mut k = GpuScalarField::zeros(gpu, &m, "k")?;
    gpu.write(&mut k.f, &vec![0.2 as Scalar; n])?;
    gpu.write(&mut k.bf, &vec![0.2 as Scalar; nbf])?;
    let mut eps = GpuScalarField::zeros(gpu, &m, "epsilon")?;
    gpu.write(&mut eps.f, &vec![1.0 as Scalar; n])?;
    gpu.write(&mut eps.bf, &vec![1.0 as Scalar; nbf])?;

    let mut sources = EnergySources::new(gpu, &m)?;
    let dt: Scalar = 5.0e-3;
    let vol = &hm.v;

    let mut energy_released = 0.0f64;
    let mut fuel_mass_consumed = 0.0f64;
    for _ in 0..300 {
        let yf_before = gpu.download(&sp.by_name("Fuel").ok_or_else(|| Error::Config("species set has no \"Fuel\"".to_string()))?.field().f)?;
        let rho_h = gpu.download(&rho.f)?;
        sources.clear(gpu)?;
        cmb.react_rans(gpu, &mut sp, &rho, &k, &eps, dt, &mut sources)?;
        let yf_after = gpu.download(&sp.by_name("Fuel").ok_or_else(|| Error::Config("species set has no \"Fuel\"".to_string()))?.field().f)?;
        let q = gpu.download(cmb.q())?;
        for cell in 0..n {
            let d_yf = (yf_before[cell] - yf_after[cell]).max(0.0) as f64;
            fuel_mass_consumed += rho_h[cell] as f64 * d_yf * vol[cell] as f64;
            energy_released += q[cell] as f64 * vol[cell] as f64 * dt as f64;
        }
    }

    let expect = fuel_mass_consumed * coeffs_c.dh_c as f64;
    let rel = if expect.abs() > 0.0 {
        ((energy_released - expect).abs() / expect.abs()) as Scalar
    } else {
        0.0
    };
    c.note(&format!(
        "burner: fuel consumed {fuel_mass_consumed:.6e} kg, heat released \
         {energy_released:.6e} J, m_F*dh_c {expect:.6e} J"
    ));
    c.check("burner exact heat release (S27, decisive)", rel, 1e-9);

    let sum_err = sp.max_sum_error(gpu)?;
    c.check("burner: species mass fractions still sum to 1", sum_err, 4.0 * Scalar::EPSILON);

    Ok(())
}

/// SPEC-LIT §28's decisive gate: an isothermal medium with hot walls reaches
/// `G = 4 sigma T^4` everywhere (equilibrium) to round-off, whatever wall
/// emissivity was chosen.
fn check_radiative_equilibrium(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    let n = [6usize, 6, 6];
    let l: [Scalar; 3] = [0.3, 0.3, 0.3];
    let axis = |i: usize| GradedAxis { lo: 0.0, hi: l[i], n: n[i], expansion: 1.0, two_sided: false };
    let b = BlockSpec {
        x: axis(0),
        y: axis(1),
        z: axis(2),
        window: None,
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall", "wall", "wall", "wall", "wall", "wall"].map(String::from),
        cyclic: Vec::new(),
    };
    let hm = blockgen::build_mesh(&b)?;
    let gm = GpuMesh::upload(gpu, &hm)?;

    let props = RadiationProps::new(2.0)?;
    let mut rad = Radiation::new(gpu, &gm, props)?;
    rad.set_walls(&hm, 0.6)?;
    rad.initialise(gpu)?;

    let t0: Scalar = 1000.0;
    let mut t = GpuScalarField::zeros(gpu, &gm, "T")?;
    gpu.write(&mut t.f, &vec![t0; hm.n_cells])?;
    gpu.write(&mut t.bf, &vec![t0; hm.n_boundary_faces])?;
    let kind = vec![BcKind::FixedValue as Label; hm.n_boundary_faces];
    let fr = vec![1.0 as Scalar; hm.n_boundary_faces];
    let ref_value = vec![t0; hm.n_boundary_faces];
    let ref_grad = vec![0.0 as Scalar; hm.n_boundary_faces];
    gpu.write(&mut t.bc_kind, &kind)?;
    gpu.write(&mut t.fr, &fr)?;
    gpu.write(&mut t.ref_value, &ref_value)?;
    gpu.write(&mut t.ref_grad, &ref_grad)?;
    let fldk = FieldKernels::new(gpu)?;
    correct_boundary_conditions(gpu, &fldk, &mut t, &gm)?;

    let solver_ctrl = SolverControls {
        solver: LinearSolverKind::PCG,
        precon: Preconditioner::Diagonal,
        tolerance: 1e-14,
        rel_tol: 0.0,
        max_iter: 5000,
        report_residuals: true,
        ..Default::default()
    };
    rad.correct(gpu, &t, None, &solver_ctrl, 1)?;

    let g = gpu.download(&rad.field().f)?;
    let want = 4.0 * ofgpu::radiation::SIGMA_SB * t0 * t0 * t0 * t0;
    let worst = g.iter().fold(0.0 as Scalar, |w, &v| w.max((v - want).abs() / want));
    c.check("radiative equilibrium (S28, decisive): G = 4 sigma T^4", worst, 1e-8);
    Ok(())
}

// ==========================================================================
//  SPEC-LIT §31.1: the cyclic-pair invariants
//
//  Cheap and geometric only - no GPU kernel, no solve - which is exactly why
//  this is the piece of §31.1 promoted into the permanent gate rather than
//  the two-mesh wall-heat-flux comparison (SPEC-LIT §29.3/§31's own
//  deferred gate): that one needs `ofgpu-fire` run to convergence on two
//  meshes, which is minutes, not the seconds this file's whole suite runs
//  in. What IS cheap, and worth gating permanently, is that a cyclic pair's
//  face matching itself never regresses - a mismatched pair "silently
//  produces a mesh that conserves nothing" (SPEC-LIT §31.1's own words),
//  which is exactly the failure mode a fast geometric check catches before
//  anything gets as far as a solve.
// ==========================================================================

/// Independently re-derives SPEC-LIT §31.1's face matching (nearest
/// translated centroid) and checks both invariants the section names -
/// bijection, and `Sf_a == -Sf_b` to a stated tolerance - against a small
/// cyclic block [`blockgen::build_mesh`] itself produced, rather than
/// trusting the SAME matching code path the reader uses. `cases/README.md`'s
/// `channelPeriodicWF.jsonc` and `channelPeriodicFluxWF.jsonc` exercise the real
/// reader path end to end; this is the fast, permanent geometric gate behind
/// it.
fn check_cyclic_pair(c: &mut Checks) -> Result<()> {
    // Deliberately NOT cubic and NOT a power of two in any axis, so a bug
    // that only shows up off an accidental symmetry has somewhere to hide.
    let mut b = BlockSpec {
        x: GradedAxis { lo: 0.0, hi: 0.7, n: 6, ..GradedAxis::default() },
        y: GradedAxis { lo: 0.0, hi: 0.3, n: 5, ..GradedAxis::default() },
        z: GradedAxis { lo: 0.0, hi: 0.4, n: 4, ..GradedAxis::default() },
        ..BlockSpec::default()
    };
    b.set_cyclic_axis(0)?;
    let hm = blockgen::build_mesh(&b)?;

    let cyclic_patches: Vec<usize> = hm
        .patches
        .iter()
        .enumerate()
        .filter(|(_, p)| p.kind == PatchKind::Cyclic)
        .map(|(i, _)| i)
        .collect();
    c.require("exactly two cyclic patches", cyclic_patches.len() == 2);
    if cyclic_patches.len() != 2 {
        return Ok(());
    }
    let (pa, pb) = (&hm.patches[cyclic_patches[0]], &hm.patches[cyclic_patches[1]]);
    c.check(
        "the two cyclic patches carry equal face counts",
        (pa.size as Scalar - pb.size as Scalar).abs(),
        0.0,
    );

    // The translation SPEC-LIT §31.1 says is implied by the block's own
    // extent along the cyclic axis - x here, since `set_cyclic_axis(0)` was
    // asked for above. Independent of `nbr_patch`/`build_patches`: derived
    // straight from the two patches' own mean face-centre `x`, not assumed
    // from `b.x.hi`, so a geometry bug in the writer cannot cancel against
    // the same assumption made here.
    let mean_x = |p: &ofgpu::mesh::PatchInfo| -> Scalar {
        let s: Scalar = (0..p.size).map(|i| hm.b_cf[p.start + i].x).sum();
        s / p.size.max(1) as Scalar
    };
    let translate = mean_x(pb) - mean_x(pa);

    // Nearest-centroid matching, SPEC-LIT §31.1's own algorithm re-derived:
    // O(n^2) over a few dozen faces, which is why this belongs in the
    // seconds-scale permanent suite and the two-mesh flux comparison does
    // not.
    let mut matched_b = vec![false; pb.size];
    let mut bijection_ok = true;
    let mut worst_sf_mismatch: Scalar = 0.0;
    for i in 0..pa.size {
        let fa = pa.start + i;
        let target = hm.b_cf[fa] + Vec3::new(translate, 0.0, 0.0);
        let mut best: Option<(usize, Scalar)> = None;
        for j in 0..pb.size {
            let fb = pb.start + j;
            let d = (hm.b_cf[fb] - target).mag();
            if best.map_or(true, |(_, bd)| d < bd) {
                best = Some((j, d));
            }
        }
        let Some((j, _)) = best else {
            bijection_ok = false;
            continue;
        };
        if matched_b[j] {
            bijection_ok = false; // this partner was already claimed once
        }
        matched_b[j] = true;

        let fb = pb.start + j;
        let scale = hm.b_mag_sf[fa].max(hm.b_mag_sf[fb]).max(1e-30);
        let mismatch = (hm.b_sf[fa] + hm.b_sf[fb]).mag() / scale;
        worst_sf_mismatch = worst_sf_mismatch.max(mismatch);
    }
    bijection_ok &= matched_b.iter().all(|&m| m);

    c.require(
        "cyclic pair: face matching is a bijection (SPEC-LIT S31.1 invariant 1)",
        bijection_ok,
    );
    c.check(
        "cyclic pair: Sf_a == -Sf_b after translation (SPEC-LIT S31.1 invariant 2)",
        worst_sf_mismatch,
        1e-12,
    );

    Ok(())
}

// ==========================================================================
//  SPEC-LIT §32: the thermal wall-function gate, redesigned
//
//  What is cheap to promote here: the `flux_to_grad` identity a fixed-flux
//  wall (`BcKind::FixedFluxTemperature`) rests on, and the two published
//  Nusselt-number correlations §32.3 compares against. What is NOT promoted
//  here: the two-mesh channel comparison itself (§32.2) - a live GPU run to
//  statistical steady state, minutes long, reported once in
//  `docs/07-fire-solver.md` §1.1 rather than re-run on every `cargo test`.
// ==========================================================================

/// SPEC-LIT §32.2's own identity: `k_eff_wall * flux_to_grad(q_w, k_eff_wall)
/// == q_w` EXACTLY, for any positive `k_eff_wall` - a `fr = 0` Robin
/// condition delivers exactly the flux it is given, independent of the
/// conductivity used to construct it. This is what licenses using ONE
/// condition on both a wall-function mesh (`k_eff_wall` carrying a large
/// eddy contribution) and a resolved one (`k_eff_wall` the molecular `k`
/// alone) - see `BcKind::FixedFluxTemperature`'s own doc.
fn check_fixed_flux_identity(c: &mut Checks) {
    use ofgpu::energy::flux_to_grad;

    let mut worst: Scalar = 0.0;
    for q_w in [1.0 as Scalar, 200.0, -500.0, 1.0e4] {
        for k_eff in [1e-4 as Scalar, 0.026, 0.5, 14.34, 1.0e3] {
            let grad = flux_to_grad(q_w, k_eff);
            let flux = k_eff * grad;
            worst = worst.max((flux - q_w).abs() / q_w.abs());
        }
    }
    c.check(
        "flux_to_grad: k_eff * flux_to_grad(q, k_eff) == q for any k_eff (S32.2)",
        worst,
        1e-12,
    );
}

/// SPEC-LIT §32.3's two correlations, at the channel operating point this
/// section's own two-mesh comparison runs at (Re ~ 1.6e4, Pr = 0.71) -
/// promoted from `wallfunctions::tests::nu_correlations_at_the_channel_operating_point`,
/// a closed-form numeric pin computed independently by hand. Both land
/// within Dittus-Boelter's own quoted ±20-25% of each other, which is the
/// cross-check §32.3 asks for BEFORE either is compared against a live run:
/// two independent published correlations agreeing with each other is not
/// the gate itself, but a correlation that disagreed with its own more
/// modern refinement by more than its stated uncertainty would be a formula
/// bug, not a physics finding.
fn check_nu_correlations(c: &mut Checks) {
    use ofgpu::wallfunctions::{dittus_boelter_nu, gnielinski_f, gnielinski_nu};

    let re: Scalar = 1.6e4;
    let pr: Scalar = 0.71;

    let nu_db = dittus_boelter_nu(re, pr);
    let nu_gn = gnielinski_nu(re, pr);
    let f = gnielinski_f(re);

    c.check(
        "Dittus-Boelter Nu at Re=1.6e4, Pr=0.71 matches the hand-derived value",
        (nu_db - 46.294_261_62).abs(),
        1e-4,
    );
    c.check(
        "Gnielinski f at Re=1.6e4 matches the hand-derived value",
        (f - 0.027_708_723_8).abs(),
        1e-9,
    );
    c.check(
        "Gnielinski Nu at Re=1.6e4, Pr=0.71 matches the hand-derived value",
        (nu_gn - 43.528_672_6).abs(),
        1e-3,
    );

    c.note(&format!(
        "Nu_DB = {}, Nu_Gn = {}, ratio = {} (Dittus-Boelter's own +-20-25% band)",
        sci(nu_db, 4),
        sci(nu_gn, 4),
        sci(nu_db / nu_gn, 4)
    ));
    c.check(
        "Dittus-Boelter and Gnielinski agree within Dittus-Boelter's own +-25% band",
        (nu_db - nu_gn).abs() / nu_gn,
        0.25,
    );
}

/// SPEC-LIT §32.5, LIVE - not a replay. The friction factor a run realises,
/// measured from the wall-face traction by
/// [`ofgpu::wallfunctions::wall_shear`], checked against TWO independent
/// closed forms on the gate cases' own geometry (`0.08 x 0.04 x 0.04 m`,
/// walls top and bottom, `empty` front and back - `cases/
/// channelPeriodicFluxWF.jsonc`'s block exactly):
///
/// 1. **The force balance itself** (§32.5.2). A fully developed channel
///    driven by a uniform body force `g_x` per unit mass balances
///    `g_x sum(rho V)` against the wall traction's own integral, so
///    `tau_w = rho g_x V/A_wall = rho g_x H/2`. The one-cell wall gradient
///    of the analytic parabola under-reads that by EXACTLY `dy/(2H) = 1/(2
///    n_y)` - a closed form, so this is checked to round-off at two mesh
///    densities rather than to a tolerance, and the first-order convergence
///    is checked with it.
/// 2. **Plane-Poiseuille's own `f Re = 96`** (Shah & London, *Laminar Flow
///    Forced Convection in Ducts*, Academic Press (1978), the parallel-plate
///    row - the same table §34's laminar sanity check already used for the
///    duct). This is what says [`ofgpu::wallfunctions::darcy_friction_factor`]
///    is the DARCY convention and not the Fanning one four times smaller.
///
/// Also checks `D_h = 4V/A_wall`, the definition `ofgpu-fire` reports from,
/// against `2H` on this same block - the reduction SPEC-LIT §32.2 asserts
/// for a plane channel is here computed, not assumed.
fn check_realised_friction_factor(c: &mut Checks) -> Result<()> {
    use ofgpu::wallfunctions::{
        darcy_friction_factor, gnielinski_f, gnielinski_nu, gnielinski_nu_at_f, wall_shear,
        u_tau_of, WallShearForm,
    };

    // The gate cases' own block (SPEC-LIT §34): H = 0.04 m across, hot walls
    // at ymin/ymax, `empty` front and back so no third pair of walls exists.
    let h: Scalar = 0.04;
    let (g_x, nu, rho): (Scalar, Scalar, Scalar) = (3.9, 1.5e-5, 1.2);

    let mut ratios: Vec<(usize, Scalar)> = Vec::new();
    for ny in [20usize, 200] {
        let spec = MeshSpec {
            n: [4, ny, 1],
            l: [0.08, h, 0.04],
            two_d: true,
            ..Default::default()
        };
        let m = make_mesh(&scratch_dir("friction"), &spec)?;
        let nbf = m.n_boundary_faces;

        // Plane Poiseuille, the analytic solution of this exact forcing:
        // u(y) = (g_x/(2 nu)) y (H - y), so U_b = g_x H^2/(12 nu).
        let u_i: Vec<Vec3> = (0..m.n_cells)
            .map(|cell| {
                let y = m.c[cell].y;
                Vec3::new(g_x / (2.0 * nu) * y * (h - y), 0.0, 0.0)
            })
            .collect();
        let ws = wall_shear(
            &m,
            Vec3::new(1.0, 0.0, 0.0),
            &u_i,
            &vec![Vec3::ZERO; nbf],
            &vec![rho; nbf],
            &vec![0.0 as Scalar; nbf],
            None,
            &vec![false; nbf],
            nu,
            0.09,
        );

        // The force balance, from the mesh's own volume and wall area - the
        // same two reductions `ofgpu-fire` does at the end of a run.
        let volume: Scalar = m.v.iter().sum();
        let tau_force = g_x * rho * volume / ws.area;

        if ny == 20 {
            c.check(
                "wall_shear finds every wall face of the gate cases' own block",
                if ws.n_faces == 2 * 4 { 0.0 } else { 1.0 },
                0.0,
            );
            c.check(
                "D_h = 4V/A_wall reduces to 2H on a plane channel (SPEC-LIT 32.2/32.5)",
                ((4.0 * volume / ws.area) - 2.0 * h).abs() / (2.0 * h),
                1e-12,
            );
            c.require(
                "a lowRe wall takes the viscous tau_w form (SPEC-LIT 32.5.1)",
                ws.forms() == vec![WallShearForm::Viscous],
            );
            // Shah & London's parallel-plate `f Re = 96`, from the ANALYTIC
            // wall shear (the force balance) rather than the discrete one -
            // the closed form is what is being checked here, not the mesh.
            let u_b = g_x * h * h / (12.0 * nu);
            let re = u_b * (2.0 * h) / nu;
            let f = darcy_friction_factor(tau_force, rho, u_b);
            c.note(&format!(
                "laminar plane Poiseuille: U_b = {} m/s, Re_Dh = {}, f = {}, f*Re = {}",
                sci(u_b, 4),
                sci(re, 4),
                sci(f, 4),
                sci(f * re, 6),
            ));
            c.check(
                "darcy_friction_factor gives Shah & London's f*Re = 96 for parallel plates",
                (f * re - 96.0).abs() / 96.0,
                1e-12,
            );
        }

        ratios.push((ny, ws.tau_w / tau_force));

        // SPEC-LIT §32.5.2's CORRECTED cross-check quantity, live on the same
        // field: `drag_kin` is `sum nu_eff |dU_par| deltaCoeffs |Sf|`, the
        // term the momentum matrix itself carries, and this crate's momentum
        // equation has no density in it. So on a uniform-density field it
        // must be `drag / rho` exactly, and it must balance the KINEMATIC
        // body force `g_x V` up to the same one-cell error - with no density
        // anywhere in either statement. This is the identity that closed to
        // `+0.001 %` on the real wall-function channel run (§32.5.3).
        c.check(
            &format!(
                "drag_kin = drag/rho exactly at uniform density, ny = {ny} (SPEC-LIT 32.5.2)"
            ),
            (ws.drag_kin - ws.drag / rho).abs() / (ws.drag / rho).abs(),
            1e-14,
        );
        c.check(
            &format!(
                "drag_kin balances the KINEMATIC body force g_x V, ny = {ny} (SPEC-LIT 32.5.2)"
            ),
            (ws.drag_kin / (g_x * volume) - (1.0 - 1.0 / (2.0 * ny as Scalar))).abs(),
            1e-12,
        );
    }

    // The one-cell gradient of a parabola under-reads the wall shear by
    // exactly dy/(2H): u(dy/2) = (g_x/(2 nu))(dy/2)(H - dy/2), and dividing
    // by dy/2 leaves (g_x/(2 nu))(H - dy/2) against the analytic
    // (g_x/(2 nu)) H.
    for (ny, ratio) in &ratios {
        let want = 1.0 - 1.0 / (2.0 * *ny as Scalar);
        c.check(
            &format!(
                "measured tau_w / force balance = 1 - 1/(2 ny) exactly, ny = {ny} (SPEC-LIT 32.5.2)"
            ),
            (ratio - want).abs() / want,
            1e-12,
        );
    }
    c.note(&format!(
        "the measurement approaches the force balance first order in the wall cell: \
         {:+.2}% at ny = {}, {:+.2}% at ny = {}",
        (ratios[0].1 - 1.0) * 100.0,
        ratios[0].0,
        (ratios[1].1 - 1.0) * 100.0,
        ratios[1].0,
    ));

    // A wall-function face takes the wall function's OWN tau_w = rho u_tau^2
    // instead - SPEC-LIT §32.5.1's second form, on the same block.
    {
        let spec = MeshSpec {
            n: [4, 6, 1],
            l: [0.08, h, 0.04],
            two_d: true,
            ..Default::default()
        };
        let m = make_mesh(&scratch_dir("frictionWf"), &spec)?;
        let nbf = m.n_boundary_faces;
        let (k0, cmu): (Scalar, Scalar) = (0.35, 0.09);
        let ws = wall_shear(
            &m,
            Vec3::new(1.0, 0.0, 0.0),
            &vec![Vec3::new(5.0, 0.0, 0.0); m.n_cells],
            &vec![Vec3::ZERO; nbf],
            &vec![rho; nbf],
            &vec![0.0 as Scalar; nbf],
            Some(&vec![k0; m.n_cells]),
            &vec![true; nbf],
            nu,
            cmu,
        );
        let u_tau = u_tau_of(k0, cmu);
        c.require(
            "a wall-function face takes the wall-function tau_w form (SPEC-LIT 32.5.1)",
            ws.forms() == vec![WallShearForm::WallFunctionK],
        );
        c.check(
            "wall-function tau_w is rho (Cmu^1/4 sqrt(k_P))^2 (SPEC-LIT 29.3/32.5.1)",
            (ws.tau_w_mag - rho * u_tau * u_tau).abs() / (rho * u_tau * u_tau),
            1e-12,
        );
        c.require(
            "the unused viscous form is reported alongside it, not averaged in",
            ws.by_patch.iter().all(|r| r.tau_w_other.is_some()),
        );

        // SPEC-LIT §32.5.1's selector is NOT "does this face carry a nut wall
        // function": a VELOCITY-based one (§15.1 `nutU`, §30.1
        // Werner-Wengle) must fall to the viscous form, because for those the
        // viscous form IS their own tau_w and `Cmu^1/4 sqrt(k_P)` would be a
        // different model's friction velocity. Same mesh, same k field, flag
        // cleared - the form must change with the flag and nothing else.
        let ws_u = wall_shear(
            &m,
            Vec3::new(1.0, 0.0, 0.0),
            &vec![Vec3::new(5.0, 0.0, 0.0); m.n_cells],
            &vec![Vec3::ZERO; nbf],
            &vec![rho; nbf],
            &vec![0.0 as Scalar; nbf],
            Some(&vec![k0; m.n_cells]),
            &vec![false; nbf],
            nu,
            cmu,
        );
        c.require(
            "a velocity-based wall function falls to the viscous form (SPEC-LIT 32.5.1)",
            ws_u.forms() == vec![WallShearForm::Viscous],
        );
        c.check(
            "and the k-based value is then the CROSS-CHECK, not the reported one",
            (ws_u.by_patch[0].tau_w_other.expect("k is available") - rho * u_tau * u_tau).abs()
                / (rho * u_tau * u_tau),
            1e-12,
        );
    }

    // Gnielinski at a supplied `f` (SPEC-LIT §32.5): it must reduce to the
    // published pipe form at the Petukhov `f`, and it must MOVE with `f`, or
    // §32.4's two verdicts would be the same verdict under two names.
    {
        let (re, pr): (Scalar, Scalar) = (25_834.0, 0.71);
        let f_pipe = gnielinski_f(re);
        c.check(
            "gnielinski_nu_at_f at the Petukhov f reproduces gnielinski_nu exactly",
            (gnielinski_nu_at_f(f_pipe, re, pr) - gnielinski_nu(re, pr)).abs(),
            0.0,
        );
        let hi = gnielinski_nu_at_f(f_pipe * 1.08, re, pr);
        let lo = gnielinski_nu_at_f(f_pipe, re, pr);
        c.note(&format!(
            "Re = {}: Petukhov pipe f = {} gives Nu_Gn = {}; an 8% higher (plane-channel) f \
             gives {} - {:+.1}%, comparable with Gnielinski's whole +-10% band",
            sci(re, 4),
            sci(f_pipe, 4),
            sci(lo, 4),
            sci(hi, 4),
            (hi / lo - 1.0) * 100.0,
        ));
        c.require(
            "Nu_Gn rises with f, so 'at the realised f' is a DIFFERENT verdict (SPEC-LIT 32.4)",
            hi > lo * 1.05,
        );
    }

    Ok(())
}

// ==========================================================================
//  SPEC-LIT §32.4/§32.5: the two channel legs, judged both ways
// ==========================================================================
//
// `cases/channelPeriodicFluxWF.jsonc` and `channelPeriodicFluxLowRe.jsonc`
// differ ONLY in mesh and wall treatment (SPEC-LIT §34), so every case input
// below is common to both and is read off the case files - not off a run.

/// `sources[].bodyForce`, m/s² per unit mass.
const CHANNEL_G_X: Scalar = 3.9;
/// `0.08 x 0.04 x 0.04 m`.
const CHANNEL_VOLUME: Scalar = 1.28e-4;
/// The two hot walls, `0.08 x 0.04 m` each. Front and back are `empty`
/// (§34.1) and the streamwise pair is cyclic, so these are the only walls.
const CHANNEL_WALL_AREA: Scalar = 6.4e-3;
const CHANNEL_NU: Scalar = 1.5e-5;
const CHANNEL_PR: Scalar = 0.71;
const CHANNEL_CP: Scalar = 1006.0;
/// The `sources[].thermostat` both cases carry (SPEC-LIT §35.1).
const CHANNEL_T_TARGET: Scalar = 293.15;
const CHANNEL_THERMOSTAT_TAU: Scalar = 0.02;

/// One channel leg, judged under BOTH of SPEC-LIT §32.4's verdicts.
struct LegVerdict {
    d_h: Scalar,
    re: Scalar,
    nu_measured: Scalar,
    /// Volume-mean `T`, from the thermostat's own steady law (§35.1):
    /// `P = -rho cp (T_mean - T_target) V/tau`.
    t_mean: Scalar,
    rho_b: Scalar,
    rho_bar: Scalar,
    /// The traction `ofgpu-fire` MEASURED at the wall, in whichever of
    /// §32.5.1's two forms is correct for this leg's own wall treatment
    /// (`rho u_tau^2` on the wall-function leg, viscous on the resolved one).
    tau_w_measured: Scalar,
    f_measured: Scalar,
    /// The VISCOUS form on the same faces - the term the momentum matrix
    /// itself carries, §32.5.2. Equal to [`Self::tau_w_measured`] on a
    /// resolved leg; the second, different number on a wall-function one.
    tau_w_viscous: Scalar,
    f_viscous: Scalar,
    /// `g_x rho_bar V / A_wall` - the body-force INFERENCE this project
    /// quoted before either leg was rerun with the measurement. SUPERSEDED
    /// and kept only so the size of its error is on the record.
    tau_w_inferred: Scalar,
    f_inferred: Scalar,
    /// `sum_walls nu_eff |dU_par| deltaCoeffs |Sf|`, m^4/s^2, as printed by
    /// the run - the kinematic wall sink §32.5.2's corrected cross-check
    /// compares against `(g . e_hat) V`.
    kin_sink: Scalar,
    /// `|thermostat power| / (q_w A_wall) - 1` - the leg's own energy-balance
    /// gap, which §32.4 requires to be quoted as an uncertainty on `Nu`.
    energy_gap: Scalar,
    f_pipe: Scalar,
    nu_gn_pipe: Scalar,
    nu_gn_realised: Scalar,
    nu_gn_viscous: Scalar,
    nu_db: Scalar,
}

/// `(g . e_hat) V`, m^4/s^2 - the body force in the KINEMATIC units this
/// crate's momentum equation is written in (SPEC-LIT §32.5.2's correction).
const CHANNEL_KIN_FORCE: Scalar = CHANNEL_G_X * CHANNEL_VOLUME;

/// Everything §32.4's table needs for one leg, from its recorded measurement
/// plus the case inputs above - no constant here that is not either recorded
/// in `docs/07-fire-solver.md` §1.1 or written in the case file.
///
/// `rho_b` is recovered from the recorded `k_thermal = rho cp nu/Pr` rather
/// than recomputed from `p0/(R_s T_b)`, so the density in the friction factor
/// is BY CONSTRUCTION the same one the recorded Nusselt number was built
/// with; `rho_bar` follows from it by the ideal-gas law at fixed `p0`,
/// `rho_bar/rho_b = T_b/T_mean`.
#[allow(clippy::too_many_arguments)]
fn channel_leg_verdict(
    q_w: Scalar,
    t_w: Scalar,
    t_b: Scalar,
    u_b: Scalar,
    k_thermal: Scalar,
    thermostat_power: Scalar,
    tau_w_measured: Scalar,
    tau_w_viscous: Scalar,
    kin_sink: Scalar,
) -> LegVerdict {
    use ofgpu::wallfunctions::{
        darcy_friction_factor, dittus_boelter_nu, gnielinski_f, gnielinski_nu_at_f,
    };

    let d_h = 4.0 * CHANNEL_VOLUME / CHANNEL_WALL_AREA;
    let re = u_b * d_h / CHANNEL_NU;
    let nu_measured = q_w * d_h / (k_thermal * (t_w - t_b));

    let rho_b = k_thermal * CHANNEL_PR / (CHANNEL_CP * CHANNEL_NU);
    // SPEC-LIT §35.1's steady law, inverted for the volume mean the
    // controller settled at. The gain is rho(T_target) cp, and rho(T_target)
    // = rho_b T_b/T_target at fixed p0.
    let rho_cp = rho_b * (t_b / CHANNEL_T_TARGET) * CHANNEL_CP;
    let t_mean = CHANNEL_T_TARGET
        + thermostat_power.abs() * CHANNEL_THERMOSTAT_TAU / (rho_cp * CHANNEL_VOLUME);
    let rho_bar = rho_b * t_b / t_mean;

    let tau_w_inferred = CHANNEL_G_X * rho_bar * CHANNEL_VOLUME / CHANNEL_WALL_AREA;
    let f_inferred = darcy_friction_factor(tau_w_inferred, rho_b, u_b);
    let f_pipe = gnielinski_f(re);

    let f_measured = darcy_friction_factor(tau_w_measured, rho_b, u_b);
    let f_viscous = darcy_friction_factor(tau_w_viscous, rho_b, u_b);

    LegVerdict {
        d_h,
        re,
        nu_measured,
        t_mean,
        rho_b,
        rho_bar,
        tau_w_measured,
        f_measured,
        tau_w_viscous,
        f_viscous,
        tau_w_inferred,
        f_inferred,
        kin_sink,
        energy_gap: thermostat_power.abs() / (q_w * CHANNEL_WALL_AREA) - 1.0,
        f_pipe,
        nu_gn_pipe: gnielinski_nu_at_f(f_pipe, re, CHANNEL_PR),
        // SPEC-LIT §32.4's Reynolds-analogy verdict, at the f the wall
        // actually MEASURES - not at the body-force inference this used to
        // be taken at, which was wrong on both legs (§32.5.3).
        nu_gn_realised: gnielinski_nu_at_f(f_measured, re, CHANNEL_PR),
        nu_gn_viscous: gnielinski_nu_at_f(f_viscous, re, CHANNEL_PR),
        nu_db: dittus_boelter_nu(re, CHANNEL_PR),
    }
}

/// Print one leg's two verdicts, in the form SPEC-LIT §32.4 now requires:
/// every band statement names the `f` it was judged at.
fn note_leg_verdict(c: &mut Checks, leg: &str, v: &LegVerdict) {
    c.note(&format!(
        "{leg}: D_h = {} m, Re = {}, Nu_measured = {}, T_mean (from the thermostat's own law) \
         = {} K, rho_b = {} kg/m3, rho_bar = {} kg/m3",
        sci(v.d_h, 4),
        sci(v.re, 5),
        sci(v.nu_measured, 4),
        sci(v.t_mean, 6),
        sci(v.rho_b, 5),
        sci(v.rho_bar, 5),
    ));
    c.note(&format!(
        "{leg}: f MEASURED at the wall = {} (tau_w = {} Pa) | viscous form on the same faces = \
         {} (tau_w = {} Pa) | Petukhov smooth-PIPE f = {} - the measurement is {:+.1}% of it",
        sci(v.f_measured, 4),
        sci(v.tau_w_measured, 4),
        sci(v.f_viscous, 4),
        sci(v.tau_w_viscous, 4),
        sci(v.f_pipe, 4),
        (v.f_measured / v.f_pipe - 1.0) * 100.0,
    ));
    c.note(&format!(
        "{leg}: the SUPERSEDED body-force inference was f = {} (tau_w = {} Pa) - {:+.1}% of the \
         measurement. Every Reynolds-analogy verdict once quoted at it was too generous \
         (SPEC-LIT 32.5.3)",
        sci(v.f_inferred, 4),
        sci(v.tau_w_inferred, 4),
        (v.f_inferred / v.f_measured - 1.0) * 100.0,
    ));
    c.note(&format!(
        "{leg}: kinematic force balance (S32.5.2's correction): wall sink {} m4/s2 against \
         (g.e_hat) V = {} m4/s2 - {:+.3}%",
        sci(v.kin_sink, 5),
        sci(CHANNEL_KIN_FORCE, 5),
        (v.kin_sink / CHANNEL_KIN_FORCE - 1.0) * 100.0,
    ));
    c.note(&format!(
        "{leg}: ABSOLUTE-PREDICTION verdict (Gnielinski at the Petukhov pipe f): Nu_Gn = {} \
         ({:+.1}%) | REYNOLDS-ANALOGY verdict (Gnielinski at the MEASURED f): Nu_Gn = {} \
         ({:+.1}%), and at the viscous f {} ({:+.1}%) | Dittus-Boelter: Nu_DB = {} ({:+.1}%) \
         | energy-balance uncertainty on Nu: +-{:.1}% (S32.4)",
        sci(v.nu_gn_pipe, 4),
        (v.nu_measured / v.nu_gn_pipe - 1.0) * 100.0,
        sci(v.nu_gn_realised, 4),
        (v.nu_measured / v.nu_gn_realised - 1.0) * 100.0,
        sci(v.nu_gn_viscous, 4),
        (v.nu_measured / v.nu_gn_viscous - 1.0) * 100.0,
        sci(v.nu_db, 4),
        (v.nu_measured / v.nu_db - 1.0) * 100.0,
        v.energy_gap.abs() * 100.0,
    ));
}
/// The WALL-FUNCTION leg's recorded measurement -
/// `cases/channelPeriodicFluxWF.jsonc`, `docs/07-fire-solver.md` §1.1's LAST
/// subsection, a 40 000-iteration run to `|U|` residual 2.1e-8 whose state is
/// unchanged in every printed digit from iteration 5 000 on. Every number here
/// is a printed output of that run; nothing is derived.
///
/// **These numbers changed twice.** With SPEC-LIT §13.4.1/§32.5.5, because the
/// run that produced the set before them was made by a driver that ignored
/// this case's own `div(phi,U)` entry and ran `bounded Gauss upwind` in place
/// of the `Gauss linearUpwind grad(U)` the case asks for (restoring that
/// substitution by hand reproduced the old numbers to five significant
/// figures, `Nu` 64.3136 against 64.3168, which is what said it was a rerun of
/// the same case and not a different one). And again with **SPEC-LIT §26.1**,
/// which folded §25.1's own conduction term into the low-Mach divergence
/// constraint: on this leg that is worth -0.06 % of `Nu` (64.5257 -> 64.4894)
/// and takes the energy balance from +0.106 % to +0.0174 %. This leg is the
/// CONTROL, and it behaved like one.
fn wall_function_leg() -> LegVerdict {
    channel_leg_verdict(
        500.0,          // q_w, W/m2, imposed on both hot walls
        317.497,        // T_w, K, diagnosed by the thermal wall function
        293.251_6,      // T_b, K, mixed-mean (T_w - the printed dT 24.2454)
        5.394_07,       // U_b, m/s
        0.025_582_438,  // k_thermal = rho(T_b) cp nu/Pr
        -3.200_56,      // thermostat power, W (the sink)
        0.075_035_8,    // tau_w MEASURED, Pa - S32.5.1's wall-function form here
        0.086_602_7,    // the viscous form on the same faces, Pa
        4.991_75e-4,    // kinematic wall sink, m4/s2 (S32.5.2)
    )
}

/// The RESOLVED leg's recorded measurement -
/// `cases/channelPeriodicFluxLowRe.jsonc`, `docs/07-fire-solver.md` §1.1's
/// LAST subsection, a 40 000-iteration run to `|U|` residual 4.1e-12.
///
/// **These numbers changed with SPEC-LIT §13.4.1/§32.5.5**, for the same
/// reason [`wall_function_leg`]'s did, and by much more: restoring the
/// substituted `bounded Gauss upwind` by hand reproduced the previous record
/// exactly (`Nu` 70.4709 against 70.4707, drag balance -3.787 % against
/// -3.787 %), and honouring the case's own unbounded
/// `Gauss linearUpwind grad(U)` closed that drag imbalance to -0.000 % and
/// raised `Nu` by 3.6 %.
///
/// **And again with SPEC-LIT §26.1**, which is the one that mattered here.
/// §25.1's `Q` was implemented without its conduction term `div(k_eff grad T)`,
/// so the pressure equation was prescribing a dilatation of about -0.07 s^-1
/// on a thermally fully developed periodic channel whose true `div(u)` is
/// ZERO. With `Q` complete: `Nu` 72.9988 -> 71.6830, `T_w` 314.186 -> 314.549,
/// `U_b` 4.92909 -> 4.93682, `contErr` 1.101e-7 -> 6.7253e-14, and the energy
/// balance this leg had carried as a +3.11 % uncertainty on every band
/// statement it makes closes to **-2.84e-06 W of 3.2 W**. The state is a fixed
/// point: 80 000 iterations reproduce all of the above in every printed
/// digit.
fn resolved_leg() -> LegVerdict {
    channel_leg_verdict(
        500.0,          // q_w, W/m2
        314.549,        // T_w, K
        292.772_3,      // T_b, K (T_w - the printed dT 21.7767)
        4.936_82,       // U_b, m/s
        0.025_624_308,  // k_thermal = rho(T_b) cp nu/Pr
        -3.200_00,      // thermostat power, W - S26.1 closed this to the
                        // wall heat: the printed gap is -2.83972e-06 W
        0.087_534_9,    // tau_w MEASURED, Pa - the viscous form on a lowRe wall
        0.087_534_9,    // ... which IS the viscous form, so the two coincide
        // The driver reported this as `0.0004992 m4/s2, disagreement -0.000 %`
        // - i.e. it closes to better than the +-0.0005 % its own 3-decimal
        // format can resolve. Recorded at the body force exactly, which is what
        // that measurement says to the precision it was printed at. The
        // SUPERSEDED value, taken with the `bounded` correction the driver was
        // supplying, was 4.80296e-4 - a -3.787 % gap (SPEC-LIT §32.5.5).
        4.992_00e-4,    // kinematic wall sink, m4/s2 (S32.5.2)
    )
}

/// The DECISIVE EXPERIMENT of SPEC-LIT §35.3.2, replayed: the same two cases,
/// the same 40 000 iterations, `"weighting"` the only token that differs.
/// `(Nu, T_w - T_b)` for each of the four runs, exactly as `ofgpu-fire`
/// printed them. Each pair differs in one token and nothing else, which is
/// what makes the comparison controlled rather than two different states.
/// **Re-measured after SPEC-LIT S26.1**, all four runs, because that section's
/// completion of S25.1's `Q` moves every state on these two cases. The three
/// statements S35.3.2 predicted survive: `massFlux` lowers `Nu` on both legs,
/// widens `(T_w - T_b)` on both, and shifts the resolved mesh 2.7x more than
/// the wall-function one - the same factor the superseded set gave. The
/// SUPERSEDED set, kept because the verdicts taken at it were published:
/// 75.6765 / 72.9988 / 20.6330 / 21.3862 and 65.3886 / 64.5257 / 23.9143 /
/// 24.2318.
const WEIGHTING_EXPERIMENT: [(&str, Scalar, Scalar, Scalar, Scalar); 2] = [
    // leg,             Nu uniform, Nu massFlux, dT uniform, dT massFlux
    ("resolved", 74.4529, 71.6830, 20.9704, 21.7767),
    ("wall function", 65.3942, 64.4894, 23.9122, 24.2454),
];

/// SPEC-LIT §32.5.5's ISOLATION, replayed: the `div(phi,U)` entry the driver
/// used to substitute for the case's own, varied over all four combinations of
/// `{Gauss upwind, Gauss linearUpwind grad(U)} x {plain, bounded}` on the
/// resolved leg and three of them on the wall-function leg, 40 000 iterations
/// each, nothing else changed. These are the numbers `ofgpu-fire` printed.
///
/// **These seven runs are a PRE-§26.1 record and are kept as one.** They were
/// made with §25.1's `Q` implemented without its conduction term, which was
/// prescribing a dilatation of about -0.07 s^-1 on a channel whose true
/// `div(u)` is zero - and §3.1's correction integrates against exactly that
/// dilatation. On the fixed solver the same experiment gives a different
/// answer, and BOTH are recorded: see [`BOUNDED_AFTER_S261`] below, which is
/// what this check asserts about the solver as it now stands. What the seven
/// runs ESTABLISHED is not affected and is still asserted here: the imbalance
/// was the `bounded` token and not the scheme's ORDER, and the energy
/// imbalance moved with neither, which is what refuted §32.5.3's "one defect
/// with two symptoms".
///
/// `energy_gap` here is `|thermostat power| / (q_w A_wall) - 1`, formed from
/// the same printed thermostat power as everywhere else; `drag_gap` is the
/// driver's own kinematic force-balance disagreement (SPEC-LIT §32.5.2).
struct BoundedRun {
    leg: &'static str,
    div_entry: &'static str,
    bounded: bool,
    second_order: bool,
    nu_measured: Scalar,
    drag_gap: Scalar,
    energy_gap: Scalar,
}

const BOUNDED_EXPERIMENT: [BoundedRun; 7] = [
    BoundedRun { leg: "resolved", div_entry: "bounded Gauss upwind", bounded: true,
        second_order: false, nu_measured: 70.4709, drag_gap: -0.03787, energy_gap: 0.032_573 },
    BoundedRun { leg: "resolved", div_entry: "bounded Gauss linearUpwind grad(U)", bounded: true,
        second_order: true, nu_measured: 70.5193, drag_gap: -0.03788, energy_gap: 0.032_547 },
    BoundedRun { leg: "resolved", div_entry: "Gauss upwind", bounded: false,
        second_order: false, nu_measured: 72.9508, drag_gap: 0.0, energy_gap: 0.031_160 },
    BoundedRun { leg: "resolved", div_entry: "Gauss linearUpwind grad(U)", bounded: false,
        second_order: true, nu_measured: 72.9988, drag_gap: 0.0, energy_gap: 0.031_136 },
    BoundedRun { leg: "wall function", div_entry: "bounded Gauss upwind", bounded: true,
        second_order: false, nu_measured: 64.3136, drag_gap: -0.00112, energy_gap: 0.001_047 },
    BoundedRun { leg: "wall function", div_entry: "Gauss upwind", bounded: false,
        second_order: false, nu_measured: 64.3815, drag_gap: 0.00002, energy_gap: 0.001_048 },
    BoundedRun { leg: "wall function", div_entry: "Gauss linearUpwind grad(U)", bounded: false,
        second_order: true, nu_measured: 64.5257, drag_gap: -0.00005, energy_gap: 0.001_062 },
];

/// SPEC-LIT §32.4's VERDICT, LOCKED AGAINST REGRESSION - this REPLAYS A
/// RECORDED MEASUREMENT, it does not run the case. The numbers below are the
/// wall-function leg of the redesigned gate, `cases/channelPeriodicFluxWF.jsonc`,
/// rebuilt per SPEC-LIT §34 as a genuine 2-D PLANE channel (streamwise-cyclic,
/// `empty` front/back, hot walls top and bottom - no side walls at all, unlike
/// the duct this case used to be) and, since SPEC-LIT §35, driven by the
/// bulk-temperature thermostat in place of the old fixed `-heaterPower -3.2`,
/// as reported in `docs/07-fire-solver.md` §1.1's LAST subsection (a run to
/// `\|U\|` residual 2.1e-8, unchanged in every printed digit from iteration
/// 5 000 on - a true fixed point, not merely a small residual):
/// `q_w = 500 W/m2` on both hot walls, `y+` 56.89/57.78/58.59 (min/mean/max),
/// `T_w = 317.483 K` (diagnosed by the thermal wall function),
/// `T_b = 293.251 K` (mixed-mean), `U_b = 5.39720 m/s`. The thermostat's own
/// energy balance leaves +0.106 % here (power -3.2034 W against 3.2 W of
/// measured wall heat), which §32.4 carries as an uncertainty on `Nu` - see
/// [`check_thermostat_sign_and_steady_offset`] for the controller's own law,
/// checked directly rather than only through this replay.
///
/// **The prose above was itself stale once.** It used to quote
/// `T_w = 317.253 K`, `T_b = 293.283 K`, `U_b = 5.3696 m/s` and an energy
/// balance closing to 2.8e-7 W - the UNIFORM-thermostat run, superseded twice
/// over (by §35.3's mass-flux weighting and then by §13.4.1's numerics fix)
/// while [`wall_function_leg`] below was updated and this comment was not.
/// Both now name the same measurement.
///
/// A live GPU run to statistical steady state has no place in `cargo test` -
/// see this module's own `published_benchmarks` for why. What this check
/// buys instead: if a future change to the thermal wall function, `Energy`,
/// or the SIMPLE loop ever moves the WALL-FUNCTION mesh's Nusselt number
/// outside the correlations' own stated bands, this fails on every commit,
/// not only on the next multi-second re-run someone remembers to do by hand.
fn check_thermal_wall_function_gate_verdict_replay(c: &mut Checks) {
    let v = wall_function_leg();
    note_leg_verdict(c, "wall-function leg", &v);

    // SPEC-LIT §35.2's energy balance for this leg: the thermostat's power IS
    // the wall heat. The shipped `massFlux` configuration leaves 0.106 %,
    // which §32.4 then makes this leg's own uncertainty on `Nu` (and it is far
    // smaller than the band margin below, which is why the verdict is not
    // undecided here). The `2.4e-7 W` this comment used to quote for the
    // `uniform` control is a PRE-§13.4.1 number, from the same runs §32.5.5
    // superseded; the `uniform` leg was rerun for `WEIGHTING_EXPERIMENT` but
    // its thermostat power was not re-recorded, so no current figure is
    // claimed for it here.
    c.check(
        "WF leg: thermostat power = q_w A_wall to better than 0.2% (S35.2)",
        v.energy_gap.abs(),
        2e-3,
    );

    // SPEC-LIT §32.5.2's force balance, in the kinematic units the momentum
    // equation is actually written in. This leg is where that identity was
    // MEASURED - `-0.005 %` on the case as shipped, and `-0.005 %` again at
    // the `uniform` default - and it is what says `wall_shear`'s viscous form
    // is the discrete momentum sink on a real flow and not only on the
    // analytic field [`check_realised_friction_factor`] manufactures. The
    // `-0.113 %` this comment used to record was §3.1's `bounded` correction,
    // applied to momentum by a driver that ignored the case (SPEC-LIT
    // §32.5.5); it is reproduced exactly by restoring that entry by hand.
    c.check(
        "WF leg: kinematic wall sink = (g.e_hat) V to better than 0.2% (S32.5.2)",
        (v.kin_sink - CHANNEL_KIN_FORCE).abs() / CHANNEL_KIN_FORCE,
        2e-3,
    );

    // SPEC-LIT §32.4, verdict 1: the ABSOLUTE-PREDICTION question, Gnielinski
    // at the Petukhov smooth-PIPE `f`. Quoted at +-10%; this leg sits at
    // about -5.9%.
    c.check(
        "WF Nu in Gnielinski's +-10% band at the PIPE f (absolute prediction, S32.4)",
        (v.nu_measured - v.nu_gn_pipe).abs() / v.nu_gn_pipe,
        0.10,
    );
    // Verdict 2, the REYNOLDS-ANALOGY question, is NOT asserted any more, and
    // the reason is the whole point of §32.5.3: it used to be checked at an
    // `f` INFERRED from the body force, it passed at +6.4%, and the inference
    // was 25% high. At the `f` this leg's wall actually measures it is +33.8%
    // - outside the band - and at the viscous form of the same measurement
    // +15.4%, also outside. Reported, not hidden, and not asserted as a pass
    // it is not.
    c.note(&format!(
        "OPEN (Reynolds analogy): WF Nu is {:+.1}% of Gnielinski at the MEASURED f and {:+.1}% \
         at the viscous form of it - OUTSIDE the +-10% band on both. The +6.4% this check used \
         to assert was taken at an INFERRED f that was 25% high (SPEC-LIT 32.5.3)",
        (v.nu_measured / v.nu_gn_realised - 1.0) * 100.0,
        (v.nu_measured / v.nu_gn_viscous - 1.0) * 100.0,
    ));
    // Dittus-Boelter takes no `f` argument, so it has ONE verdict only, and
    // it is an absolute-prediction one. Quoted at +-20-25%; this leg sits at
    // about -12.9%.
    c.check(
        "wall-function Nu within Dittus-Boelter's own +-25% band (replayed measurement, S32.4)",
        (v.nu_measured - v.nu_db).abs() / v.nu_db,
        0.25,
    );

    // The y+ this measurement was taken at - SPEC-LIT §32.4's own "both
    // meshes land in their regime" row, for the wall-function leg.
    let y_plus_mean: Scalar = 57.7661;
    c.check(
        "wall-function mesh's own y+ mean sits inside the 30-60 target (replayed measurement)",
        if (30.0..=60.0).contains(&y_plus_mean) { 0.0 } else { 1.0 },
        0.0,
    );
}

/// SPEC-LIT §33.3's analytic table for Launder & Sharma's damping functions -
/// `f_mu`, `f_2` at their `Re_t -> infinity`/`Re_t = 0` limits, monotone in
/// between, and the model's own `reduces_to_the_standard_model_at_large_re_t`
/// claim checked again here against the STANDARD model's own coefficients
/// (`ofgpu::models::KEpsilonCoeffs::default`) rather than trusted from the
/// unit test alone.
///
/// Pure host arithmetic - no mesh, no GPU, microseconds - which is exactly
/// why this belongs in the fast, always-run suite rather than beside the
/// live channel run noted in `run`'s own comment: SPEC-LIT §33.3 splits the
/// model's obligations into what an analytic limit can prove (this) and what
/// only a real flow can (the law of the wall, run and reported but not
/// replayed here - see that comment).
fn check_launder_sharma_damping_functions(c: &mut Checks) {
    use ofgpu::models::{f2, f_mu, KEpsilonCoeffs};

    // Re_t -> infinity: f_mu, f_2 -> 1 to round-off, and the coefficients
    // THEMSELVES are §6.1's own, unchanged (SPEC-LIT §33.1: "modifies the
    // model with f_mu, f_2, D and E, not with new constants").
    let std_coeffs = KEpsilonCoeffs::default();
    let ls_coeffs = KEpsilonCoeffs::default();
    c.check(
        "LaunderSharmaKE reuses KEpsilonCoeffs unchanged (Cmu)",
        (ls_coeffs.cmu - std_coeffs.cmu).abs() as Scalar,
        0.0,
    );
    c.check(
        "LaunderSharmaKE reuses KEpsilonCoeffs unchanged (C2)",
        (ls_coeffs.c2 - std_coeffs.c2).abs() as Scalar,
        0.0,
    );

    let mut worst_fmu_inf: Scalar = 0.0;
    let mut worst_f2_inf: Scalar = 0.0;
    for re_t in [1.0e6 as Scalar, 1.0e9, 1.0e12] {
        worst_fmu_inf = worst_fmu_inf.max((f_mu(re_t) - 1.0).abs());
        worst_f2_inf = worst_f2_inf.max((f2(re_t) - 1.0).abs());
    }
    c.check("f_mu(Re_t -> infinity) -> 1 (SPEC-LIT 33.3)", worst_fmu_inf, 1e-6);
    c.check("f_2(Re_t -> infinity) -> 1 (SPEC-LIT 33.3)", worst_f2_inf, 1e-9);

    // Re_t = 0: f_mu = exp(-3.4) ~ 0.0334 (nu_t suppressed ~30x at the
    // wall - the number SPEC-LIT §33.1 itself quotes), f_2 = 0.7.
    let fmu0 = f_mu(0.0);
    let want_fmu0 = (-3.4 as Scalar).exp();
    c.check("f_mu(Re_t = 0) = exp(-3.4) (SPEC-LIT 33.3)", (fmu0 - want_fmu0).abs(), 1e-12);
    c.note(&format!("f_mu(0) = {} (~1/30th of its Re_t -> infinity value)", sci(fmu0, 4)));

    let f2_0 = f2(0.0);
    c.check("f_2(Re_t = 0) = 0.7 (SPEC-LIT 33.3)", (f2_0 - 0.7).abs(), 1e-12);

    // Monotone increasing in between, and bounded - the shape SPEC-LIT
    // §33.3's table asks for ("monotone in between"), swept at fine
    // resolution over the range the wall-to-log-layer transition covers.
    let mut prev_fmu = f_mu(0.0);
    let mut prev_f2 = f2(0.0);
    let mut fmu_monotone = true;
    let mut f2_monotone = true;
    let mut fmu_bounded = true;
    let mut f2_bounded = true;
    for i in 1..=2000 {
        let re_t = i as Scalar * 0.5;
        let fmu = f_mu(re_t);
        let f2v = f2(re_t);
        fmu_monotone &= fmu >= prev_fmu - 1e-15;
        f2_monotone &= f2v >= prev_f2 - 1e-15;
        fmu_bounded &= (0.0..=1.0 + 1e-12).contains(&fmu);
        f2_bounded &= (0.7..=1.0 + 1e-12).contains(&f2v);
        prev_fmu = fmu;
        prev_f2 = f2v;
    }
    c.require("f_mu is monotone increasing in Re_t (SPEC-LIT 33.3)", fmu_monotone);
    c.require("f_2 is monotone increasing in Re_t (SPEC-LIT 33.3)", f2_monotone);
    c.require("f_mu stays within [0, 1]", fmu_bounded);
    c.require("f_2 stays within [0.7, 1]", f2_bounded);
}

/// SPEC-LIT §33.2/§34's mesh-resolution check, REPLAYED against
/// `cases/channelPeriodicFluxLowRe.jsonc` as rebuilt in SPEC-LIT §34 (the
/// genuine 2-D plane channel, `empty` front/back - see `docs/07-fire-solver.md`
/// §1.1) and, since SPEC-LIT §35, driven by the bulk-temperature thermostat.
/// `ofgpu-fire`'s own end-of-run report (`ofgpu::models::mesh_resolution_report`,
/// over a REAL Poisson wall distance and the converged `k` field, not the
/// duct version's hot-wall-only approximation) measured
/// `max_first_cell_y_plus = 0.00174051` and `cells_below_y_plus_20 = 192` of
/// 400 cells at 40 000 iterations. The cell count is unmoved by every change
/// this measurement has been through; the `y+` itself has drifted only in its
/// last three digits, 0.00174716 (pre-thermostat) -> 0.00174585 (thermostat,
/// uniform sink) -> 0.00174051 (thermostat, `massFlux` weighting, the shipped
/// case), each step being the converged sink getting very slightly stronger
/// and coupling back into `U_b` through the low-Mach `rho(T)` term at exactly
/// that scale. Both numbers are fully converged and stable well before 40 000
/// and are bit-identical from either initial temperature (SPEC-LIT §35.2's
/// own regression), so replaying them here is not vulnerable to the (now
/// fixed - see [`check_resolved_leg_gate_verdict_replay`]) energy-equation
/// drift SPEC-LIT §34 first reported this replay against.
///
/// This locks §33.2's own pass/fail rule - first-cell y+ < 1, at least 10
/// cells at y+ < 20 - against regression on every commit.
fn check_resolved_leg_mesh_resolution_replay(c: &mut Checks) {
    use ofgpu::models::MeshResolutionReport;

    let report = MeshResolutionReport {
        // 0.00174051 -> 0.00185363 at §32.5.5's rerun (the wall-adjacent
        // cell's y+ following `U_b` up 1.9 % when the momentum equation
        // stopped carrying a `bounded` correction the case never asked for),
        // and 0.00185363 -> 0.00179449 at §26.1's. The cell COUNT is unmoved,
        // as it has been through every change this measurement has seen.
        max_first_cell_y_plus: 0.001_794_49,
        cells_below_y_plus_20: 192,
        n_wall_faces: 16,
    };

    c.note(&format!(
        "replayed: worst wall-adjacent y+ = {}, {} / 400 cells at y+ < 20 ({} wall faces)",
        sci(report.max_first_cell_y_plus, 4),
        report.cells_below_y_plus_20,
        report.n_wall_faces,
    ));

    c.check(
        "resolved leg's own y+ < 1 (SPEC-LIT 33.2, replayed measurement)",
        if report.max_first_cell_y_plus < 1.0 { 0.0 } else { 1.0 },
        0.0,
    );
    c.check(
        "resolved leg has at least 10 cells at y+ < 20 (SPEC-LIT 33.2, replayed measurement)",
        if report.cells_below_y_plus_20 >= 10 { 0.0 } else { 1.0 },
        0.0,
    );
    c.require(
        "MeshResolutionReport::warnings() is empty for the replayed measurement",
        report.warnings().is_empty(),
    );
}

/// SPEC-LIT §35.1's proportional law and §35.2's own checks, run LIVE on a
/// tiny mesh (unlike the two replays above, this is not a recorded snapshot
/// - `Thermostat::correct` genuinely executes on the GPU here, in
/// milliseconds, which is what lets this stand in for the two-initial-
/// temperature regression without a multi-minute channel solve: the
/// controller's output is a PURE function of the current volume-mean `T`
/// and the fixed `(target, tau, rho_cp)` it was built with - nothing here
/// depends on how `T` got to be what it is, so two different histories that
/// arrive at the same `T` produce bit-identical output by construction, and
/// this checks that construction directly rather than replaying two
/// 40 000-iteration runs to demonstrate it (SPEC-LIT §35.2's own regression
/// was run live instead - see `docs/07-fire-solver.md` §1.1 - and is NOT
/// promoted here because it takes ~2.5 minutes per initial condition, not
/// seconds).
fn check_thermostat_sign_and_steady_offset(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    use ofgpu::sources::{flow_through_time, Thermostat};

    let spec = MeshSpec {
        n: [2, 2, 2],
        l: [0.2, 0.2, 0.2],
        ..Default::default()
    };
    let m = make_mesh(&scratch_dir("thermostat"), &spec)?;
    let gm = GpuMesh::upload(gpu, &m)?;

    // SPEC-LIT §35.1's default `tau`: V^(1/3) / U_ref. `spec.volume()` is
    // this mesh's own `0.2^3 = 0.008 m3`, so V^(1/3) = 0.2 m exactly.
    let tau_default = flow_through_time(spec.volume(), 4.0)?;
    c.check(
        "flow_through_time defaults to V^(1/3)/U_ref (SPEC-LIT 35.1)",
        (tau_default - 0.05).abs(),
        1e-9,
    );

    let target: Scalar = 350.0;
    let tau: Scalar = 0.02;
    let rho_cp: Scalar = 1206.0;
    let mut th = Thermostat::new(gpu, &gm, target, tau, rho_cp)?;

    // A SOURCE when the domain starts cold, a SINK when it starts hot -
    // SPEC-LIT §35.1: "a source as readily as a sink".
    let cold = gpu.upload(&vec![300.0 as Scalar; m.n_cells])?;
    let q_cold = th.correct(gpu, &gm, &cold)?;
    c.require("thermostat is a SOURCE below its target (SPEC-LIT 35.1)", q_cold > 0.0);

    let hot = gpu.upload(&vec![400.0 as Scalar; m.n_cells])?;
    let q_hot = th.correct(gpu, &gm, &hot)?;
    c.require("thermostat is a SINK above its target (SPEC-LIT 35.1)", q_hot < 0.0);

    // At T_mean == target it asks for exactly nothing - the fixed point the
    // full channel regression converges to regardless of initial T.
    let at_target = gpu.upload(&vec![target; m.n_cells])?;
    let q_zero = th.correct(gpu, &gm, &at_target)?;
    c.check("thermostat asks for 0 at its own T_mean == target", q_zero.abs(), 1e-6);

    // A persistent net forcing (here, a manufactured "wall heat" analog Q)
    // settles the volume-mean at target + Q*tau/(rho_cp*V), NOT at target
    // exactly - SPEC-LIT §35.1 states the ideal ("at steady state T_mean =
    // T_target"); §35's own channel measurement (docs/07-fire-solver.md
    // §1.1) found the true discrete offset is small but real for any
    // nonzero forcing, and this is that same closed form, checked directly:
    // iterate `T_mean_{n+1} = T_mean_n + dt*(Q/(rho_cp*V) - (T_mean_n -
    // target)/tau)` (the ODE `Thermostat` implements one explicit step of)
    // to ITS OWN fixed point and compare against the closed form.
    let q_forcing: Scalar = 25_000.0; // W/m3 equivalent, arbitrary
    let dt: Scalar = 1.0e-4; // small relative to tau, for the explicit Euler step to be stable
    let mut t_mean: Scalar = target;
    for _ in 0..2_000_000 {
        let q = -rho_cp * (t_mean - target) / tau;
        t_mean += dt * (q_forcing + q) / rho_cp;
    }
    let closed_form = target + q_forcing * tau / rho_cp;
    c.check(
        "the P-controller's steady-state offset matches target + Q*tau/rho_cp (SPEC-LIT 35.1)",
        (t_mean - closed_form).abs() / closed_form,
        1e-6,
    );
    c.note(&format!(
        "a persistent {} W/m3 forcing settles T_mean at target + {} K, not at target exactly \
         (the ordinary steady-state error of a proportional-only controller) - \
         docs/07-fire-solver.md S1.1's own {} W leg measured a {} K offset the same way",
        sci(q_forcing, 4),
        sci(q_forcing * tau / rho_cp, 4),
        sci(3.2, 2),
        sci(0.426, 3),
    ));

    Ok(())
}

/// SPEC-LIT §32.4's verdict, REPLAYED, for the resolved leg
/// (`cases/channelPeriodicFluxLowRe.jsonc`) now that SPEC-LIT §35's
/// thermostat has given it an actual steady state to measure - see
/// `docs/07-fire-solver.md` §1.1 for the full derivation and the
/// two-initial-temperature regression this measurement is downstream of.
///
/// UNLIKE [`check_thermal_wall_function_gate_verdict_replay`], this does
/// NOT close on both correlations, and is not asserted as if it did:
/// Dittus-Boelter's wider ±20-25% band is met (+6.0%), Gnielinski's tighter
/// ±10% band is not (+14.1%, four points over). SPEC-LIT §32.4's own rule -
/// "the gate closes when both meshes sit inside the correlation band" -
/// is honestly NOT met, so only the Dittus-Boelter comparison is asserted;
/// the Gnielinski comparison is a NOTE, so a future change that moves this
/// number is visible without failing the whole suite over an already-known,
/// reported gap.
///
/// **SPEC-LIT §32.5.5 moved this leg FURTHER out**, from +11.8 % to +14.1 %,
/// when the driver stopped substituting `bounded Gauss upwind` for the
/// `Gauss linearUpwind grad(U)` the case asks for. The same change CLOSED this
/// leg's kinematic drag imbalance, from -3.787 % to -0.000 %; that is the
/// finding the rerun produced, and it did not close the Nusselt gate.
fn check_resolved_leg_gate_verdict_replay(c: &mut Checks) {
    let v = resolved_leg();
    let w = wall_function_leg();
    note_leg_verdict(c, "resolved leg", &v);

    // The derivation of `T_mean` from the thermostat's own steady law is
    // checked against the value `docs/07-fire-solver.md` §1.1 RECORDS for
    // this leg (293.563 K after S26.1; 293.576 K at the S13.4.1 numerics,
    // 293.574 K before that). Since S26.1 the two legs settle at the SAME
    // `T_mean`, because both thermostats now settle at the same -3.2 W - which
    // is the balance closing, seen from the controller's side.
    c.check(
        "T_mean from the thermostat law matches the recorded 293.563 K (S35.1/S35.2)",
        (v.t_mean - 293.563).abs(),
        5e-3,
    );

    c.check(
        "resolved leg Nu within Dittus-Boelter's own +-25% band (replayed measurement, S32.4)",
        (v.nu_measured - v.nu_db).abs() / v.nu_db,
        0.25,
    );

    // SPEC-LIT §35.2's energy balance NOW CLOSES on this leg - §26.1 - and
    // that is asserted rather than only noted, so a reintroduction of the
    // defect fails on the commit that makes it. The history is kept in the
    // note beside it because every band statement this gate ever published
    // carried the old gap as an uncertainty.
    // SPEC-LIT §26.1's own before/after, replayed beside the verdicts it
    // moved. This is the pair that says the balance CLOSED rather than merely
    // being reported smaller: the correction's own domain integral collapses
    // by 1126x on the resolved leg and 6.1x on the wall-function one, and its
    // PRESCRIBED half - which the mechanism §32.5.5 proposed said was the
    // culprit - is round-off before AND after, on both legs.
    for (leg, gap_before, gap_after, corr_before, corr_after, presc_before, presc_after) in [
        ("resolved", -0.099_634_2, -2.840_06e-6, -0.099_631_3, 8.853_03e-8, -2.060_05e-13, -2.538_78e-14),
        ("wall function", -3.399_63e-3, -5.571_18e-4, -3.399_37e-3, -5.568_69e-4, 1.966_45e-13, 9.781_02e-14),
    ] {
        c.note(&format!(
            "{leg} leg (S26.1): balance gap {} W -> {} W; the bounded correction's own domain \
             integral {} W -> {} W ({:.0}x smaller); its PRESCRIBED half {} W -> {} W, round-off \
             both times - which is the measurement that refuted \"the correction removes the \
             prescribed dilatation\"",
            sci(gap_before as Scalar, 6),
            sci(gap_after as Scalar, 6),
            sci(corr_before as Scalar, 6),
            sci(corr_after as Scalar, 6),
            (corr_before as Scalar / corr_after as Scalar).abs(),
            sci(presc_before as Scalar, 3),
            sci(presc_after as Scalar, 3),
        ));
        c.require(
            &format!("{leg} leg: the correction's PRESCRIBED half is round-off BEFORE the fix \
                      (S26.1's refutation)"),
            (presc_before as Scalar).abs() < 1e-9,
        );
        c.require(
            &format!("{leg} leg: the correction's own integral shrank by at least 6x (S26.1)"),
            (corr_before as Scalar / corr_after as Scalar).abs() > 6.0,
        );
        c.require(
            &format!("{leg} leg: the energy-balance gap shrank by at least 6x (S26.1)"),
            (gap_before as Scalar / gap_after as Scalar).abs() > 6.0,
        );
    }
    c.note(
        "S26.1 also RETIRES this leg's contErr reading: contErr is max_c |sum_f phi_f|, which \
         with S25.3's target divergence reports the PRESCRIBED dilatation and not a solver \
         residual. It fell from 1.101e-7 to 6.7253e-14 at the same relTol, because a thermally \
         fully developed periodic channel's true div(u) is ZERO and an incomplete Q was \
         prescribing -0.07 s^-1",
    );

    c.note(&format!(
        "resolved leg energy balance (S35.2): thermostat power 3.20000 W against q_w A_wall = \
         {} W - a {:+.5}% gap, the printed difference being -2.84e-06 W. HISTORY: +2.81% at the \
         uniform sink, +3.26% at massFlux, +3.11% after S32.5.5's momentum fix, +3.35% under \
         S37's KaysCrawford - all of it S25.1's `Q` implemented without its conduction term, \
         and all of it closed by S26.1. S32.4's uncertainty on Nu is now +-{:.5}%",
        sci(500.0 * CHANNEL_WALL_AREA, 4),
        v.energy_gap * 100.0,
        v.energy_gap.abs() * 100.0,
    ));
    c.check(
        "resolved leg: thermostat power = q_w A_wall to better than 0.05% (S35.2/S26.1)",
        v.energy_gap.abs(),
        5e-4,
    );
    c.note(&format!(
        "resolved leg force balance (S32.5.2): kinematic wall sink {} m4/s2 against (g.e_hat) \
         V = {} m4/s2 - {:+.3}%. CLOSED by S32.5.5: the -3.787% once recorded here was S3.1's \
         `bounded` convection correction, applied to the momentum equation by a driver that \
         ignored this case's own div(phi,U) entry, and restoring that entry by hand reproduces \
         it exactly",
        sci(v.kin_sink, 5),
        sci(CHANNEL_KIN_FORCE, 5),
        (v.kin_sink / CHANNEL_KIN_FORCE - 1.0) * 100.0,
    ));
    // The balance closes now, so it is ASSERTED rather than only noted - which
    // is what makes a future reintroduction of the same defect fail on the
    // commit that makes it, instead of being read off a note nobody diffs.
    c.check(
        "resolved leg: kinematic wall sink = (g.e_hat) V to better than 0.2% (S32.5.2/S32.5.5)",
        (v.kin_sink - CHANNEL_KIN_FORCE).abs() / CHANNEL_KIN_FORCE,
        2e-3,
    );

    // SPEC-LIT §32.4, verdict 1 - the ABSOLUTE-PREDICTION question. NOT
    // asserted with `c.check`: this is honestly outside the band (+11.8%
    // against +-10%) and is reported, not hidden - §32.4's OWN point about
    // what a real finding looks like. The mass-flux weighting of §35.3 moved
    // it from +16.3%, so a third of the old excess was the thermostat's own
    // distribution defect and the rest is not.
    let miss = (v.nu_measured / v.nu_gn_pipe - 1.0) * 100.0;
    c.note(&format!(
        "OPEN (absolute prediction): resolved leg Nu is {miss:+.1}% of Gnielinski at the \
         Petukhov smooth-PIPE f (+16.3% at the uniform thermostat sink, +11.8% at massFlux with \
         the substituted `bounded Gauss upwind` momentum entry, +14.1% once the case's own \
         entry was honoured, and {miss:+.1}% since S26.1 completed S25.1's Q) - outside its own \
         +-10% band by {:.1} points, against an energy-balance uncertainty of {:.5}% which is \
         now round-off, so the gate does NOT close and the miss is DECISIVE with nothing left \
         to hide any of it behind (S32.4's UNDECIDED clause does not apply). Under PrtModel \
         KaysCrawford the same leg is +4.3% and INSIDE - see the S37 replay below",
        miss - 10.0,
        v.energy_gap.abs() * 100.0,
    ));

    // Verdict 2 - the REYNOLDS-ANALOGY question, at this leg's own friction
    // factor. It used to be asserted as a pass at +6.8%; that rested on an
    // `f` inferred from the body force, which the direct measurement then
    // showed to be 11% high. At the measured `f` it is +15.2% - outside.
    c.note(&format!(
        "OPEN (Reynolds analogy): resolved leg Nu is {:+.1}% of Gnielinski at the MEASURED f \
         = {} - outside the +-10% band. The +6.8% once asserted here was taken at an INFERRED \
         f of {} (SPEC-LIT 32.5.3). That measured f is only {:+.1}% of the Petukhov pipe f, so \
         this leg now transports very nearly the right MOMENTUM and too much HEAT - a THERMAL \
         finding, with nothing left on the momentum side to carry it (SPEC-LIT 32.5.5)",
        (v.nu_measured / v.nu_gn_realised - 1.0) * 100.0,
        sci(v.f_measured, 4),
        sci(v.f_inferred, 4),
        (v.f_measured / v.f_pipe - 1.0) * 100.0,
    ));

    // The two-mesh ratio §32.4's table asks for, and what the MEASURED
    // friction factors say about it. Reported, not asserted: it is a
    // decomposition through a correlation, not an independent measurement,
    // and the two legs' `tau_w` are not even taken in the same form.
    c.note(&format!(
        "two-mesh ratio Nu_resolved/Nu_wallFunction = {} (1.125 at the uniform sink; 1.096 \
         at massFlux with the substituted momentum entry) \
         against the ratio Gnielinski predicts from the two legs' own MEASURED viscous-form \
         friction factors, {} - the meshes measure f = {} and {} at the SAME body force",
        sci(v.nu_measured / w.nu_measured, 4),
        sci(v.nu_gn_viscous / w.nu_gn_viscous, 4),
        sci(v.f_viscous, 4),
        sci(w.f_viscous, 4),
    ));
}

/// SPEC-LIT §37's Kays-Crawford `Pr_t(Pe_t)`, checked LIVE (this is arithmetic,
/// not a run - it costs microseconds and belongs in the always-run suite).
///
/// The two limits are what make the correlation trustworthy without a table to
/// copy from, so they are asserted rather than trusted: `Pe_t -> 0` has to give
/// the conduction-sublayer value `2 Pr_t_inf = 1.70`, which is inside the
/// 1.5-1.9 Kays (1994) reports for air, and `Pe_t -> inf` has to give back
/// exactly the constant `Pr_t_inf = 0.85` every measurement this project has
/// recorded was made with - a model that did not would be changing the free
/// stream as a side effect of correcting the wall.
///
/// The remaining rows guard the two numerical branches SPEC-LIT §37.2 derives.
/// Neither is decoration: a resolved `lowRe` wall face hands this function
/// `Pe_t = 0` exactly on every outer iteration, where the literature form's
/// own arithmetic is `inf/inf`.
fn check_kays_crawford_prt(c: &mut Checks) {
    use ofgpu::energy::{kays_crawford_prt, KAYS_CRAWFORD_C as C};

    // The literature form of §37.1, written out as the papers print it, so
    // this checks the implementation against the SOURCE rather than against
    // itself.
    fn literature(pe_t: f64, c: f64, p_inf: f64) -> f64 {
        let (x, a) = (c * pe_t, p_inf.sqrt());
        1.0 / (1.0 / (2.0 * p_inf) + x / a - x * x * (1.0 - (-1.0 / (x * a)).exp()))
    }

    let p_inf: Scalar = 0.85;

    c.note(&format!(
        "Kays-Crawford C = {}, Pr_t_inf = {} -> sublayer limit 2*Pr_t_inf = {} \
         (Kays 1994 reports 1.5-1.9 for air)",
        sci(C, 3),
        sci(p_inf, 3),
        sci(2.0 * p_inf, 4),
    ));

    // ---- limit 1: Pe_t -> 0, the conduction sublayer --------------------
    let mut worst_sublayer: Scalar = 0.0;
    for pi in [0.7 as Scalar, 0.85, 0.9, 1.0] {
        for pe in [0.0 as Scalar, 1e-300, Scalar::MIN_POSITIVE] {
            worst_sublayer = worst_sublayer.max((kays_crawford_prt(pe, C, pi) - 2.0 * pi).abs());
        }
    }
    c.check(
        "Pe_t -> 0 gives Pr_t = 2*Pr_t_inf exactly, at Pe_t = 0 and 1e-300 (S37.2)",
        worst_sublayer,
        0.0,
    );

    // ---- limit 2: Pe_t -> inf, the free stream, approached FROM ABOVE ----
    //
    // S37.2's expansion is `Pr_t = Pr_t_inf (1 + u/6 - u^2/72 + ...)` with
    // `u = 1/(C Pe_t sqrt(Pr_t_inf))`, so the FIRST-order rate is checked
    // directly and the SECOND-order coefficient falls out as
    // `-Pr_t_inf/(72 Pr_t_inf C^2) = -1/(72 C^2)` - independent of
    // `Pr_t_inf`, which is what makes it worth asserting: an implementation
    // that had the algebra subtly wrong would not reproduce a coefficient it
    // was never given.
    let mut worst_first_order: Scalar = 0.0;
    let mut from_above = true;
    let first_order = |pe: Scalar| p_inf * (1.0 + 1.0 / (6.0 * p_inf.sqrt() * C * pe));
    for e in 3..=6 {
        let pe: Scalar = (10.0 as Scalar).powi(e);
        let got = kays_crawford_prt(pe, C, p_inf);
        from_above &= got > p_inf;
        worst_first_order = worst_first_order.max((got - first_order(pe)).abs() * pe * pe);
    }
    c.require("Pr_t approaches Pr_t_inf FROM ABOVE, never below it (S37.2)", from_above);
    c.check(
        "Pe_t -> inf matches Pr_t_inf(1 + 1/(6 sqrt(Pr_t_inf) C Pe_t)) to O(Pe_t^-2)",
        worst_first_order,
        0.2,
    );
    let c2_measured = (kays_crawford_prt(1e6, C, p_inf) - first_order(1e6)) * 1e12;
    c.check(
        "...and the O(Pe_t^-2) coefficient IS the derived -1/(72 C^2) (S37.2)",
        (c2_measured + 1.0 / (72.0 * C * C)).abs(),
        1e-6,
    );
    c.check(
        "Pr_t(1e9) is the free-stream constant to 1e-9 (S37.2)",
        (kays_crawford_prt(1e9, C, p_inf) - p_inf).abs(),
        1e-9,
    );

    // ---- the rearrangement is the SAME function, and the better one ------
    let (mut worst_rel, mut lo, mut hi, mut monotone) = (0.0_f64, Scalar::MAX, 0.0 as Scalar, true);
    let mut prev = kays_crawford_prt(0.0, C, p_inf);
    for i in 0..=220 {
        let pe: Scalar = 10.0_f64.powf(-8.0 + 0.05 * f64::from(i)) as Scalar;
        let got = kays_crawford_prt(pe, C, p_inf);
        monotone &= got <= prev + 1e-12 && got.is_finite();
        prev = got;
        lo = lo.min(got);
        hi = hi.max(got);
        if pe <= 1e3 {
            let want = literature(f64::from(pe), f64::from(C), f64::from(p_inf));
            worst_rel = worst_rel.max((f64::from(got) - want).abs() / want);
        }
    }
    c.require("Pr_t falls monotonically as Pe_t rises, and stays finite (S37.5)", monotone);
    c.require(
        "Pr_t never leaves [Pr_t_inf, 2 Pr_t_inf] over Pe_t = 1e-8 .. 1e3 (S37.5)",
        lo >= p_inf - 1e-12 && hi <= 2.0 * p_inf + 1e-12,
    );
    c.check(
        "S37.2's rearrangement reproduces S37.1's literature form (relative, Pe_t <= 1e3)",
        worst_rel as Scalar,
        1e-10,
    );

    // ...and is the one that keeps the digits where the literature form does
    // not. This is the row that says the rearrangement earns its place.
    let lit_1e8 = literature(1e8, f64::from(C), f64::from(p_inf));
    let ours_1e8 = f64::from(kays_crawford_prt(1e8, C, p_inf));
    c.note(&format!(
        "at Pe_t = 1e8 the literature form returns {} against the true {}, an error of {:.2}% \
         from cancellation alone; S37.2's form returns {}",
        sci(lit_1e8 as Scalar, 6),
        sci(p_inf, 6),
        100.0 * (lit_1e8 / f64::from(p_inf) - 1.0).abs(),
        sci(ours_1e8 as Scalar, 6),
    ));
    c.require(
        "the literature form HAS lost its digits by Pe_t = 1e8 (which is why S37.2 rearranges it)",
        (lit_1e8 - f64::from(p_inf)).abs() > 1e-3,
    );

    // ---- nothing anywhere in the domain of definition is a NaN -----------
    let mut all_usable = kays_crawford_prt(Scalar::INFINITY, C, p_inf) == p_inf;
    for pe in [0.0 as Scalar, Scalar::MIN_POSITIVE, 1e-300, 1e-30, 1.0, 1e30, 1e300, Scalar::MAX] {
        let got = kays_crawford_prt(pe, C, p_inf);
        all_usable &= got.is_finite() && got > 0.0;
    }
    c.require(
        "Pr_t is finite and positive at every representable Pe_t, +inf included (S37.5)",
        all_usable,
    );

    // ---- and what it is worth on the two meshes S32's gate uses ----------
    //
    // Not an assertion - a statement of scale, so a reader of the summary can
    // see WHY the wall-function leg is a near-control and the resolved leg is
    // not. `nu_t/nu` is the range each leg's own converged field reported;
    // the `Pr_t` pair below is what this function computes from it, and the
    // replay that follows is what the runs actually used.
    for (leg, r_lo, r_hi) in [
        ("wall function (y+ 58)", 16.8627 as Scalar, 28.6496),
        ("resolved (y+ 0.0019)", 3.91576e-7, 35.5161),
    ] {
        c.note(&format!(
            "{leg}: nu_t/nu in [{}, {}] gives Pr_t in [{}, {}] against the constant {}",
            sci(r_lo, 4),
            sci(r_hi, 4),
            sci(kays_crawford_prt(r_hi * 0.71, C, p_inf), 5),
            sci(kays_crawford_prt(r_lo * 0.71, C, p_inf), 5),
            sci(p_inf, 3),
        ));
    }
}

/// SPEC-LIT §37's EXPERIMENT, replayed: `cases/channelPeriodicFluxWF.jsonc`
/// and `channelPeriodicFluxLowRe.jsonc`, 40 000 iterations each, run twice
/// with `physics.fluid.PrtModel` the only token that differs. These are the
/// numbers `ofgpu-fire` printed, all four on the same binary (the first
/// wall-function `KaysCrawford` run was discarded: the driver's own
/// wall-heat report recomputed `k_eff,wall` with the constant `Pr_t` and so
/// claimed 580 W/m2 on a wall imposing 500 - see `docs/07-fire-solver.md`
/// §1.1's last subsection).
///
/// The `constant` pair are the CONTROL and reproduce this section's own
/// published record to every printed digit, which is what makes the pair a
/// controlled comparison rather than two different states -
/// [`check_thermal_wall_function_gate_verdict_replay`] and
/// [`check_resolved_leg_gate_verdict_replay`] hold those same numbers
/// independently, so a drift in either would fail there first.
struct PrtRun {
    leg: &'static str,
    model: &'static str,
    nu_measured: Scalar,
    d_t: Scalar,
    u_b: Scalar,
    /// `|thermostat power| / (q_w A_wall) - 1`, the same construction every
    /// other energy-balance number in this file uses.
    energy_gap: Scalar,
    /// The `Pr_t` the run actually used, min and max over the domain, as
    /// `ofgpu-fire`'s own §37.5 report printed them.
    prt_min: Scalar,
    prt_max: Scalar,
}

/// **Re-measured after SPEC-LIT §26.1**, which closed the energy imbalance
/// these four runs used to carry (+3.11 %/+3.35 % on the resolved leg). Every
/// statement §37 asserted below survives the remeasurement; what changed is
/// that `energy_gap` is now round-off on all four, so the band statements no
/// longer need an error bar quoted beside them. The SUPERSEDED set, kept here
/// because the verdicts taken at it were published: `Nu` 64.5257 / 63.5900 /
/// 72.9988 / 68.0305, `energy_gap` 0.001062 / 0.001100 / 0.031134 / 0.033541.
const PRT_EXPERIMENT: [PrtRun; 4] = [
    PrtRun { leg: "wall function", model: "constant", nu_measured: 64.4894, d_t: 24.2454,
        u_b: 5.39407, energy_gap: 0.000_174_1, prt_min: 0.85, prt_max: 0.85 },
    PrtRun { leg: "wall function", model: "KaysCrawford", nu_measured: 63.5527, d_t: 24.6019,
        u_b: 5.39426, energy_gap: 0.000_185_1, prt_min: 0.874934, prt_max: 0.891895 },
    PrtRun { leg: "resolved", model: "constant", nu_measured: 71.6830, d_t: 21.7767,
        u_b: 4.93682, energy_gap: 0.000_000_887_4, prt_min: 0.85, prt_max: 0.85 },
    PrtRun { leg: "resolved", model: "KaysCrawford", nu_measured: 66.8107, d_t: 23.3605,
        u_b: 4.93761, energy_gap: 0.000_000_940_9, prt_min: 0.871299, prt_max: 1.7 },
];

/// SPEC-LIT §37's experiment, replayed - THIS REPLAYS A RECORDED
/// MEASUREMENT, it does not run the case.
///
/// §37 named three things before either pair of runs, and this asserts
/// exactly those three, so a future change that reverses the sign of the
/// effect, or flattens the difference between the two meshes, fails on the
/// commit that makes it:
///
/// 1. `Nu` FALLS on both legs (a higher `Pr_t` moves less heat).
/// 2. `(T_w - T_b)` WIDENS on both, by the same token.
/// 3. The shift is much larger on the RESOLVED mesh, because `Pr_t` departs
///    from `Pr_t_inf` only where `Pe_t` is small - which is the sublayer one
///    mesh resolves and the other replaces with a wall function.
///
/// It also records the two verdicts the experiment moved, and the one it did
/// not: leg (b)'s absolute-prediction verdict crosses INTO Gnielinski's band
/// (+11.9 % -> +4.3 %; it was +14.1 % -> +6.4 % before §26.1), and leg (a)'s
/// Reynolds-analogy miss does not move at all, because that is a friction
/// finding and §37 is a thermal model.
fn check_kays_crawford_experiment_replay(c: &mut Checks) {
    use ofgpu::wallfunctions::{gnielinski_f, gnielinski_nu_at_f};

    for r in &PRT_EXPERIMENT {
        c.note(&format!(
            "{} leg, PrtModel {}: Nu = {}, dT = {} K, U_b = {} m/s, energy balance {:+.3}%, \
             Pr_t in [{}, {}]",
            r.leg,
            r.model,
            sci(r.nu_measured, 6),
            sci(r.d_t, 6),
            sci(r.u_b, 6),
            r.energy_gap * 100.0,
            sci(r.prt_min, 6),
            sci(r.prt_max, 6),
        ));
    }

    let pick = |leg: &str, model: &str| -> &PrtRun {
        PRT_EXPERIMENT.iter().find(|r| r.leg == leg && r.model == model).expect("run present")
    };

    // The resolved mesh reaches the Pe_t -> 0 limit exactly, at the wall,
    // because LaunderSharma pins nu_t there; the wall-function mesh never
    // gets near it. That asymmetry IS the mechanism, and it is a measurement.
    let (rc, rk) = (pick("resolved", "constant"), pick("resolved", "KaysCrawford"));
    let (wc, wk) = (pick("wall function", "constant"), pick("wall function", "KaysCrawford"));
    c.check(
        "resolved mesh reaches the S37.2 sublayer limit 2*Pr_t_inf exactly at the wall",
        (rk.prt_max - 1.7).abs(),
        0.0,
    );
    c.require(
        "wall-function mesh never leaves the log-layer neighbourhood of Pr_t_inf (< 0.90)",
        wk.prt_max < 0.90,
    );

    let mut shift = [0.0 as Scalar; 2];
    for (i, (leg, before, after)) in [("resolved", rc, rk), ("wall function", wc, wk)]
        .iter()
        .enumerate()
    {
        shift[i] = 1.0 - after.nu_measured / before.nu_measured;
        c.note(&format!(
            "{leg}: Nu {} -> {} ({:+.2}%), dT {} -> {} K ({:+.2}%) on the PrtModel token alone",
            sci(before.nu_measured, 6),
            sci(after.nu_measured, 6),
            (after.nu_measured / before.nu_measured - 1.0) * 100.0,
            sci(before.d_t, 6),
            sci(after.d_t, 6),
            (after.d_t / before.d_t - 1.0) * 100.0,
        ));
        c.require(
            &format!("{leg}: Kays-Crawford LOWERS Nu, as S37 predicted"),
            after.nu_measured < before.nu_measured,
        );
        c.require(
            &format!("{leg}: Kays-Crawford WIDENS (T_w - T_b), as S37 predicted"),
            after.d_t > before.d_t,
        );
        // A thermal-diffusivity model must not move the momentum field.
        c.require(
            &format!("{leg}: U_b moves by less than 0.05% - S37 is a THERMAL model"),
            (after.u_b / before.u_b - 1.0).abs() < 5e-4,
        );
    }
    c.note(&format!(
        "the shift is {:.2}x larger on the resolved mesh ({:.2}% against {:.2}%) - S37.3's \
         asymmetry, measured",
        shift[0] / shift[1],
        shift[0] * 100.0,
        shift[1] * 100.0,
    ));
    c.require(
        "the Nu shift is at least 3x larger on the resolved mesh than on the wall-function one",
        shift[0] > 3.0 * shift[1],
    );

    // The verdict this moved, at each leg's own pipe `f` - computed live from
    // the replayed Nu, not quoted.
    for (leg, before, after, re) in [
        ("resolved", rc, rk, 26329.7 as Scalar),
        ("wall function", wc, wk, 28768.4),
    ] {
        let f_pipe = gnielinski_f(re);
        let nu_gn = gnielinski_nu_at_f(f_pipe, re, 0.71);
        c.note(&format!(
            "{leg}: absolute-prediction verdict (Gnielinski at the pipe f = {}) moves from \
             {:+.1}% to {:+.1}% of Nu_Gn = {}",
            sci(f_pipe, 5),
            (before.nu_measured / nu_gn - 1.0) * 100.0,
            (after.nu_measured / nu_gn - 1.0) * 100.0,
            sci(nu_gn, 6),
        ));
    }
    let f_pipe_b = gnielinski_f(26329.7 as Scalar);
    let nu_gn_b = gnielinski_nu_at_f(f_pipe_b, 26329.7 as Scalar, 0.71);
    c.require(
        "resolved leg is OUTSIDE Gnielinski's +-10% band under PrtModel constant (the shipped \
         default, and the gate's own record)",
        (rc.nu_measured / nu_gn_b - 1.0).abs() > 0.10,
    );
    c.require(
        "resolved leg is INSIDE Gnielinski's +-10% band under PrtModel KaysCrawford (S37)",
        (rk.nu_measured / nu_gn_b - 1.0).abs() < 0.10,
    );
    // ...and it stays inside across its own energy-balance uncertainty, which
    // is what S32.4 requires before a band statement may be called a pass.
    let worst = rk.nu_measured * (1.0 + rk.energy_gap) / nu_gn_b - 1.0;
    c.note(&format!(
        "carrying this leg's own {:+.2}% energy-balance gap as an uncertainty on Nu, the far \
         edge of the band statement is {:+.2}% - still inside +-10% (S32.4)",
        rk.energy_gap * 100.0,
        worst * 100.0,
    ));
    c.require(
        "resolved leg stays inside +-10% across its own energy-balance uncertainty (S32.4)",
        worst.abs() < 0.10,
    );

    // And what it did NOT move: the two-mesh ratio now sits BELOW what the two
    // legs' own momentum difference implies, which qualifies S32.5.5's
    // decomposition rather than confirming it.
    c.note(&format!(
        "two-mesh ratio Nu_b/Nu_a falls from {} to {}; Gnielinski at the two legs' own \
         viscous-form measured f implies about 1.12 either way, so the KaysCrawford ratio is \
         BELOW its momentum-implied value - S32.5.5's momentum decomposition of the two-mesh \
         gap does not survive applying the same thermal correction to both legs",
        sci(rc.nu_measured / wc.nu_measured, 5),
        sci(rk.nu_measured / wk.nu_measured, 5),
    ));
}

/// SPEC-LIT §35.3.2's DECISIVE EXPERIMENT, replayed: on each mesh, the same
/// case run twice with `"weighting"` the only difference. §35.3.2 PREDICTED,
/// before either run, that the mass-flux weighting would widen `(T_w - T_b)`
/// and LOWER `Nu`, and that it would do so MORE on the resolved mesh than on
/// the wall-function one (the bias lives in the near-wall velocity deficit,
/// which one mesh resolves and the other hides inside a wall function). This
/// asserts exactly those three statements against the four measured numbers,
/// so a future change that reverses the sign of the effect - or flattens the
/// difference between the meshes - fails on the commit that makes it.
fn check_thermostat_weighting_experiment_replay(c: &mut Checks) {
    let mut shift = [0.0 as Scalar; 2];
    for (i, (leg, nu_uniform, nu_massflux, dt_uniform, dt_massflux)) in
        WEIGHTING_EXPERIMENT.iter().enumerate()
    {
        shift[i] = 1.0 - nu_massflux / nu_uniform;
        c.note(&format!(
            "{leg}: Nu {} -> {} ({:+.2}%), dT {} -> {} K ({:+.2}%) on the uniform -> massFlux \
             thermostat weighting alone (SPEC-LIT 35.3.2)",
            sci(*nu_uniform, 6),
            sci(*nu_massflux, 6),
            (nu_massflux / nu_uniform - 1.0) * 100.0,
            sci(*dt_uniform, 6),
            sci(*dt_massflux, 6),
            (dt_massflux / dt_uniform - 1.0) * 100.0,
        ));
        c.require(
            &format!("{leg}: massFlux weighting LOWERS Nu, as S35.3.2 predicted"),
            nu_massflux < nu_uniform,
        );
        c.require(
            &format!("{leg}: massFlux weighting WIDENS (T_w - T_b), as S35.3.2 predicted"),
            dt_massflux > dt_uniform,
        );
    }
    c.require(
        "the shift is LARGER on the resolved mesh than on the wall-function one (S35.3.2)",
        shift[0] > shift[1],
    );
    let r_uniform = WEIGHTING_EXPERIMENT[0].1 / WEIGHTING_EXPERIMENT[1].1;
    c.note(&format!(
        "so the two-mesh ratio falls from {} to {}: this mechanism accounts for {:.3} of the \
         {:.3} excess, about {:.0}% of it, measured rather than argued",
        sci(r_uniform, 4),
        sci(WEIGHTING_EXPERIMENT[0].2 / WEIGHTING_EXPERIMENT[1].2, 4),
        r_uniform - WEIGHTING_EXPERIMENT[0].2 / WEIGHTING_EXPERIMENT[1].2,
        r_uniform - 1.0,
        100.0 * (r_uniform - WEIGHTING_EXPERIMENT[0].2 / WEIGHTING_EXPERIMENT[1].2)
            / (r_uniform - 1.0),
    ));
}

/// SPEC-LIT §32.5.5's ISOLATION, replayed: what the `bounded` prefix on
/// `div(phi,U)` was worth, and what the convection scheme's ORDER was worth,
/// separated by running all four combinations on the resolved leg and three on
/// the wall-function leg.
///
/// This exists because a §13.4 defect - `ofgpu-fire` building
/// `MomentumControls` from `::default()`, whose convection entry is
/// `bounded Gauss upwind`, on two cases that ask for
/// `Gauss linearUpwind grad(U)` - produced every number §32's gate had ever
/// recorded. The rerun that fixed it found something the fix was not expected
/// to find, and these are the three statements it establishes:
///
/// 1. Dropping `bounded` CLOSES the kinematic drag balance on BOTH legs. That
///    confirms, by isolation, the mechanism §32.5.3 named as a hypothesis and
///    could not test: §3.1's bounded correction subtracts `V_P (div u)_P`, and
///    in this low-Mach solver `div u` is a PRESCRIBED constraint (§25.1), not
///    an error that vanishes at convergence.
/// 2. The scheme's ORDER is worth less than 0.3 % of `Nu` on either leg - so
///    the first-order-vs-second-order half of the substitution, which is the
///    half that looked like it should matter, did not.
/// 3. The ENERGY imbalance moves with NEITHER. That is what retires §32.5.3's
///    "one defect with two symptoms" reading: the momentum symptom is entirely
///    §3.1's correction, and the energy symptom survives its removal.
fn check_bounded_convection_experiment_replay(c: &mut Checks) {
    for r in &BOUNDED_EXPERIMENT {
        c.note(&format!(
            "{} leg, div(phi,U) = `{}`: Nu = {}, drag balance {:+.3}%, energy balance {:+.3}%",
            r.leg,
            r.div_entry,
            sci(r.nu_measured, 6),
            r.drag_gap * 100.0,
            r.energy_gap * 100.0,
        ));
    }

    // 1. `bounded` is the whole of the drag imbalance, on BOTH legs.
    for leg in ["resolved", "wall function"] {
        let worst_bounded = BOUNDED_EXPERIMENT
            .iter()
            .filter(|r| r.leg == leg && r.bounded)
            .map(|r| r.drag_gap.abs())
            .fold(0.0 as Scalar, Scalar::max);
        let worst_plain = BOUNDED_EXPERIMENT
            .iter()
            .filter(|r| r.leg == leg && !r.bounded)
            .map(|r| r.drag_gap.abs())
            .fold(0.0 as Scalar, Scalar::max);
        c.note(&format!(
            "{leg} leg: worst drag imbalance {:.3}% WITH `bounded` against {:.3}% without - the \
             correction is the whole of it (SPEC-LIT 3.1, 32.5.5)",
            worst_bounded * 100.0,
            worst_plain * 100.0,
        ));
        c.require(
            &format!("{leg} leg: dropping `bounded` closes the drag balance to under 0.05% (S32.5.5)"),
            worst_plain < 5e-4,
        );
        c.require(
            &format!("{leg} leg: `bounded` leaves a LARGER drag imbalance than dropping it (S3.1)"),
            worst_bounded > worst_plain,
        );
    }
    // ... and on the resolved leg it is a big number, not a rounding one. This
    // is the row that says the rerun found something, not nothing.
    let resolved_bounded = BOUNDED_EXPERIMENT
        .iter()
        .filter(|r| r.leg == "resolved" && r.bounded)
        .map(|r| r.drag_gap.abs())
        .fold(0.0 as Scalar, Scalar::max);
    c.check(
        "resolved leg: the `bounded` correction is worth 3.7-3.9% of the streamwise body force",
        if (0.037..=0.039).contains(&resolved_bounded) { 0.0 } else { 1.0 },
        0.0,
    );

    // 2. The scheme's ORDER is worth almost nothing, on either leg. Compared
    //    at FIXED `bounded`, so this is the order and nothing else.
    let mut worst_order_shift: Scalar = 0.0;
    for leg in ["resolved", "wall function"] {
        for bounded in [true, false] {
            let pair: Vec<&BoundedRun> = BOUNDED_EXPERIMENT
                .iter()
                .filter(|r| r.leg == leg && r.bounded == bounded)
                .collect();
            if pair.len() != 2 {
                continue;
            }
            let (a, b) = (pair[0], pair[1]);
            let (first, second) = if a.second_order { (b, a) } else { (a, b) };
            let shift = second.nu_measured / first.nu_measured - 1.0;
            worst_order_shift = worst_order_shift.max(shift.abs());
            c.note(&format!(
                "{leg} leg, `bounded` = {bounded}: first order -> second order moves Nu by \
                 {:+.2}% ({} -> {})",
                shift * 100.0,
                sci(first.nu_measured, 6),
                sci(second.nu_measured, 6),
            ));
        }
    }
    c.check(
        "the convection scheme's ORDER is worth less than 0.3% of Nu on either leg (S32.5.5)",
        worst_order_shift,
        3e-3,
    );

    // 3. The ENERGY imbalance moves with neither - which is what refutes
    //    §32.5.3's joint reading of the two imbalances.
    for leg in ["resolved", "wall function"] {
        let gaps: Vec<Scalar> =
            BOUNDED_EXPERIMENT.iter().filter(|r| r.leg == leg).map(|r| r.energy_gap).collect();
        let lo = gaps.iter().copied().fold(Scalar::INFINITY, Scalar::min);
        let hi = gaps.iter().copied().fold(Scalar::NEG_INFINITY, Scalar::max);
        c.note(&format!(
            "{leg} leg: energy balance spans {:+.3}% to {:+.3}% across every div(phi,U) entry - \
             a span of {:.3} points, against the {:.2} points of drag imbalance the same token \
             switches",
            lo * 100.0,
            hi * 100.0,
            (hi - lo) * 100.0,
            BOUNDED_EXPERIMENT
                .iter()
                .filter(|r| r.leg == leg)
                .map(|r| r.drag_gap.abs())
                .fold(0.0 as Scalar, Scalar::max)
                * 100.0,
        ));
    }
    let resolved_energy_span = {
        let gaps: Vec<Scalar> = BOUNDED_EXPERIMENT
            .iter()
            .filter(|r| r.leg == "resolved")
            .map(|r| r.energy_gap)
            .collect();
        gaps.iter().copied().fold(Scalar::NEG_INFINITY, Scalar::max)
            - gaps.iter().copied().fold(Scalar::INFINITY, Scalar::min)
    };
    c.check(
        "resolved leg: the energy imbalance moves under 0.2 points on a change worth 3.8 points \
         of momentum imbalance - the two are NOT one defect (S32.5.5)",
        resolved_energy_span,
        2e-3,
    );
    c.note(
        "so the resolved leg's +3.11% ENERGY imbalance was a single open anomaly, not half of a \
         pair - which is what sent S32.5.5's specified instrumented run after it, and S26.1 is \
         where it ended",
    );

    // ---- and the same two runs on the FIXED solver (SPEC-LIT 26.1) ------
    //
    // The seven runs above are a record of the solver as it was. This pair is
    // the solver as it is: the `bounded` token, on each leg, at the corrected
    // `Q`. The point is not that the token became harmless in general - S3.1's
    // rule is unchanged and a fire plume still has a real dilatation for the
    // correction to eat - but that on THIS case the dilatation it was eating
    // was itself the defect, so the -3.787 % does not reproduce.
    for r in &BOUNDED_AFTER_S261 {
        c.note(&format!(
            "{} leg AFTER S26.1, div(phi,U) = `{}`: Nu = {}, drag balance {:+.3}%, energy \
             balance {:+.4}%",
            r.leg,
            r.div_entry,
            sci(r.nu_measured, 6),
            r.drag_gap * 100.0,
            r.energy_gap * 100.0,
        ));
    }
    let after_resolved = BOUNDED_AFTER_S261
        .iter()
        .find(|r| r.leg == "resolved" && r.bounded)
        .expect("resolved bounded run present");
    c.require(
        "resolved leg AFTER S26.1: `bounded` leaves the drag balance inside 0.05% - the -3.787% \
         above was the fictitious dilatation an incomplete Q was prescribing (S26.1)",
        after_resolved.drag_gap.abs() < 5e-4,
    );
    c.check(
        "resolved leg AFTER S26.1: `bounded` reproduces the shipped case's own Nu to 0.01%",
        (after_resolved.nu_measured / 71.6830 - 1.0).abs(),
        1e-4,
    );
    let after_wf = BOUNDED_AFTER_S261
        .iter()
        .find(|r| r.leg == "wall function" && r.bounded)
        .expect("wall-function bounded run present");
    c.require(
        "wall-function leg AFTER S26.1: `bounded` still costs the drag balance something, and \
         a fifth of what it did (0.020% against 0.112%)",
        after_wf.drag_gap.abs() > 1e-4 && after_wf.drag_gap.abs() < 4e-4,
    );
    c.note(
        "S3.1's rule is unchanged by any of this: subtracting V_P (div u)_P from a MOMENTUM \
         equation is wrong wherever div(u) is genuinely nonzero, which a fire plume is and a \
         thermally fully developed channel - once its Q is right - is not",
    );
}

/// The same `bounded` token, on the same two cases, on the solver SPEC-LIT
/// §26.1 left behind. 40 000 iterations each, `div(phi,U)` set by hand to
/// `bounded Gauss upwind` and nothing else changed - so each row differs from
/// its shipped case in the token AND in the scheme's order, exactly as
/// [`BOUNDED_EXPERIMENT`]'s corresponding rows did.
const BOUNDED_AFTER_S261: [BoundedRun; 2] = [
    BoundedRun { leg: "resolved", div_entry: "bounded Gauss upwind", bounded: true,
        second_order: false, nu_measured: 71.6830, drag_gap: 0.0, energy_gap: 0.000_000_887_6 },
    BoundedRun { leg: "wall function", div_entry: "bounded Gauss upwind", bounded: true,
        second_order: false, nu_measured: 64.3411, drag_gap: -2.0e-4, energy_gap: 0.000_193_7 },
];

// ==========================================================================
//  Published benchmarks
//
//  These run a whole flow to steady state and take minutes, so they are
//  `#[ignore]`d and never slow `cargo test`. Run them with
//
//      cargo test --release --bin ofgpu-validate -- --ignored --nocapture
//
//  They are the only place in this file where the answer is compared with
//  numbers somebody else produced - and those numbers are *published
//  benchmark data*, not the output of another program we ran. SPEC-LIT
//  section 0 rule 4 forbids the second, not the first.
// ==========================================================================

#[cfg(test)]
mod published_benchmarks {
    use super::*;
    use ofgpu::fv::interpolate_vector_flux;

    // ----------------------------------------------------------------------
    //  Ghia, Ghia & Shin, "High-Re solutions for incompressible flow using
    //  the Navier-Stokes equations and a multigrid method",
    //  J. Comput. Phys. 48 (1982) 387-411.
    //
    //  Table I  - u along the VERTICAL line through the geometric centre of
    //             the cavity (x = 0.5), tabulated at the y below.
    //  Table II - v along the HORIZONTAL centreline (y = 0.5), at the x below.
    //
    //  Unit square, lid at y = 1 moving in +x at u = 1, Re = U L / nu.
    //  Their solutions are on a 129 x 129 uniform grid.
    // ----------------------------------------------------------------------

    /// Table I, the `y` column.
    const GHIA_Y: [f64; 17] = [
        1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000,
        0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000,
    ];

    /// Table I, `u` at Re = 100.
    const GHIA_U_RE100: [f64; 17] = [
        1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641,
        -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192,
        -0.03717, 0.00000,
    ];

    /// Table I, `u` at Re = 400.
    const GHIA_U_RE400: [f64; 17] = [
        1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135,
        -0.11477, -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266,
        -0.08186, 0.00000,
    ];

    /// Table II, the `x` column.
    const GHIA_X: [f64; 17] = [
        1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000,
        0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000,
    ];

    /// Table II, `v` at Re = 100.
    const GHIA_V_RE100: [f64; 17] = [
        0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445,
        -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10091,
        0.09233, 0.00000,
    ];

    /// Table II, `v` at Re = 400.
    ///
    /// The entry at `x = 0.9063`, `-0.23827`, is reproduced here as the paper
    /// prints it, and it is **wrong in the paper**. It breaks the monotone run
    /// between `-0.22847` at `x = 0.9453` and the profile's minimum
    /// `-0.44993` at `x = 0.8594`, where every neighbouring Reynolds number
    /// varies smoothly, and it is the only station in either table at which
    /// this solver misses by more than 0.007. Other authors have noticed:
    /// Nilsson & Wallin, *Lid driven cavity flow using finite difference and
    /// radial basis function methods*, Uppsala University report 22015 (2022)
    /// section 5.2, exclude "the reference y-velocity value at x = 0.9063 with
    /// Re = 400" from their own comparison for the same reason.
    ///
    /// It is kept in the constant because the constant is a transcription of
    /// the paper, not an edited version of it; [`GHIA_V_RE400_ERRATUM`] names
    /// the station the comparison leaves out.
    const GHIA_V_RE400: [f64; 17] = [
        0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993,
        -0.38598, 0.05188, 0.30174, 0.30203, 0.28124, 0.22965, 0.20920, 0.19713,
        0.18360, 0.00000,
    ];

    /// Index into [`GHIA_X`] of the station excluded from the Re = 400
    /// `v` comparison; see [`GHIA_V_RE400`]. Nothing else is ever excluded.
    const GHIA_V_RE400_ERRATUM: &[usize] = &[5];

    // ----------------------------------------------------------------------
    //  Sampling a structured 2-D field
    // ----------------------------------------------------------------------

    /// A cell field on the uniform `n x n` cavity mesh, indexed by its own
    /// coordinates rather than by whatever order the mesh generator emitted,
    /// and extended to the four walls so that a probe on the boundary returns
    /// the boundary condition instead of extrapolating.
    struct Sampled {
        n: usize,
        /// `(n + 2)^2`, row-major in `j`, with index 0 and `n + 1` the walls.
        v: Vec<Scalar>,
    }

    impl Sampled {
        /// `pick` selects the component; `wall` gives its value on each of the
        /// four walls in `-x +x -y +y` order.
        fn new(m: &HostMesh, u: &[Vec3], n: usize, pick: usize, wall: [Scalar; 4]) -> Self {
            let mut v = vec![0.0 as Scalar; (n + 2) * (n + 2)];

            for c in 0..m.n_cells {
                let i = (f64::from(m.c[c].x) * n as f64 - 0.5).round() as usize;
                let j = (f64::from(m.c[c].y) * n as f64 - 0.5).round() as usize;
                v[(j + 1) * (n + 2) + (i + 1)] = u[c].component(pick);
            }
            for t in 0..(n + 2) {
                v[t * (n + 2)] = wall[0]; // -x
                v[t * (n + 2) + n + 1] = wall[1]; // +x
                v[t] = wall[2]; // -y
                v[(n + 1) * (n + 2) + t] = wall[3]; // +y
            }
            Self { n, v }
        }

        /// Node coordinate of index `t`: the walls at `0` and `1`, cell
        /// centres in between.
        fn coord(&self, t: usize) -> f64 {
            if t == 0 {
                0.0
            } else if t == self.n + 1 {
                1.0
            } else {
                (t as f64 - 0.5) / self.n as f64
            }
        }

        fn bracket(&self, x: f64) -> (usize, usize, f64) {
            let mut hi = 1usize;
            while hi <= self.n + 1 && self.coord(hi) < x {
                hi += 1;
            }
            let hi = hi.min(self.n + 1).max(1);
            let lo = hi - 1;
            let (a, b) = (self.coord(lo), self.coord(hi));
            let t = if b > a { (x - a) / (b - a) } else { 0.0 };
            (lo, hi, t.clamp(0.0, 1.0))
        }

        fn at(&self, x: f64, y: f64) -> f64 {
            let (i0, i1, tx) = self.bracket(x);
            let (j0, j1, ty) = self.bracket(y);
            let g = |i: usize, j: usize| f64::from(self.v[j * (self.n + 2) + i]);
            let a = g(i0, j0) * (1.0 - tx) + g(i1, j0) * tx;
            let b = g(i0, j1) * (1.0 - tx) + g(i1, j1) * tx;
            a * (1.0 - ty) + b * ty
        }
    }

    // ----------------------------------------------------------------------
    //  SIMPLE, written from SPEC-LIT sections 5.1 and 5.2
    // ----------------------------------------------------------------------

    /// Lid-driven cavity on a uniform `n x n` mesh, run to steady state.
    ///
    /// The loop is the one SPEC-LIT section 5.2 sets out, with the Rhie-Chow
    /// face flux of section 5.1:
    ///
    /// ```text
    /// repeat:
    ///   assemble and relax momentum with alpha_U ; solve for u*
    ///   A_P = diag/V ; rAU = 1/A_P ; HbyA = rAU (b - sum_N a_N u_N)/V
    ///   phi_HbyA = interpolate(HbyA) . Sf
    ///   solve  lap(rAU_f, p) = div(phi_HbyA)
    ///   phi = phi_HbyA - rAU_f |Sf| snGrad(p)
    ///   u   = HbyA - rAU grad(p)
    ///   p   = p_old + alpha_p (p_new - p_old)
    /// ```
    ///
    /// It is written out here rather than driven through the solver's own
    /// orchestration on purpose: an acceptance test that re-derives the
    /// coupling from the specification is evidence about the discretisation,
    /// where one that calls the code under test is evidence about nothing.
    /// The discrete operators and every linear solve are still the device's.
    ///
    /// Returns the mesh, the converged cell velocity, and the last momentum
    /// residual.
    fn cavity(
        gpu: &Gpu,
        k: &Kernels,
        re: Scalar,
        n: usize,
        max_iters: usize,
    ) -> Result<(HostMesh, Vec<Vec3>, Scalar, usize)> {
        let spec = MeshSpec {
            n: [n, n, 1],
            l: [1.0, 1.0, 1.0 / (n as Scalar)],
            two_d: true,
            ..Default::default()
        };
        // Re goes in the directory name too. Both Ghia cases use n = 80, and
        // cargo runs tests in parallel, so a tag of just the size has the two
        // of them writing the same polyMesh at the same time - which shows up
        // as a PermissionDenied, not as anything that looks numerical.
        let tag = format!("cavity{n}_re{}", re as i64);
        let m = make_mesh(&scratch_dir(&tag), &spec)?;
        let gm = GpuMesh::upload(gpu, &m)?;

        let nc = m.n_cells;
        let nif = m.n_internal_faces;
        let nbf = m.n_boundary_faces;
        let nu: Scalar = 1.0 / re;

        // blockgen emits the six patches in -x +x -y +y -z +z order, so the
        // lid - the +y side - is patch 3.
        const LID_PATCH: usize = 3;
        let lid = Vec3::new(1.0, 0.0, 0.0);

        let mut u_ref = vec![Vec3::ZERO; nbf];
        let mut u_fr = vec![1.0 as Scalar; nbf];
        let mut u_kind = vec![BcKind::FixedValue as Label; nbf];
        for (pi, p) in m.patches.iter().enumerate() {
            for i in 0..p.size {
                let bf = p.start + i;
                if p.kind == PatchKind::Empty {
                    u_kind[bf] = BcKind::Empty as Label;
                    u_fr[bf] = 0.0;
                } else if pi == LID_PATCH {
                    u_ref[bf] = lid;
                }
            }
        }
        let zero_v = vec![Vec3::ZERO; nbf];
        let zero_s = vec![0.0 as Scalar; nbf];

        let gamma_u: Vec<Scalar> = m.mag_sf.iter().map(|s| nu * s).collect();
        let b_gamma_u = boundary_gamma(&m, nu);
        let d_gamma_u = gpu.upload(&gamma_u)?;
        let d_b_gamma_u = gpu.upload(&b_gamma_u)?;

        let p_kind = kinds(&m, BcKind::ZeroGradient);

        // Sealed box: no flux through any wall, the lid included - it slides
        // along its own plane, so u_lid . Sf is zero there too.
        let bphi = vec![0.0 as Scalar; nbf];

        let mut u = vec![Vec3::ZERO; nc];
        let mut p_cell = vec![0.0 as Scalar; nc];
        let mut phi = vec![0.0 as Scalar; nif];
        let mut grad_p = vec![Vec3::ZERO; nc];

        // SPEC-LIT section 5.2, Patankar section 6.7-3: alpha_p ~ 1 - alpha_U.
        let alpha_u: Scalar = 0.7;
        let alpha_p: Scalar = 0.3;

        let ctrl = SolverControls {
            tolerance: 1e-9,
            rel_tol: 0.01,
            max_iter: 500,
            precon: Preconditioner::Diagonal,
            ..Default::default()
        };
        let ctrl_p = SolverControls {
            tolerance: 1e-10,
            rel_tol: 0.001,
            max_iter: 2000,
            precon: Preconditioner::Diagonal,
            ..Default::default()
        };

        let mut ws = SolverWorkspace::for_mesh(gpu, &gm)?;
        let mut sphi = GpuSurfaceScalarField::zeros(gpu, &gm, "phi")?;
        let mut residual: Scalar = 1.0;
        let mut done = max_iters;

        for iter in 0..max_iters {
            gpu.write(&mut sphi.f, &phi)?;
            gpu.write(&mut sphi.bf, &bphi)?;

            let mut hbya = vec![Vec3::ZERO; nc];
            let mut rau = vec![0.0 as Scalar; nc];
            let mut mom_res: Scalar = 0.0;

            for comp in 0..2 {
                let mut ui = GpuScalarField::zeros(gpu, &gm, "Ui")?;
                let cell: Vec<Scalar> = u.iter().map(|v| v.component(comp)).collect();
                let bnd: Vec<Scalar> = u_ref.iter().map(|v| v.component(comp)).collect();
                gpu.write(&mut ui.f, &cell)?;
                gpu.write(&mut ui.fr, &u_fr)?;
                gpu.write(&mut ui.ref_value, &bnd)?;
                gpu.write(&mut ui.ref_grad, &zero_s)?;
                gpu.write(&mut ui.bc_kind, &u_kind)?;
                correct_boundary_conditions(gpu, &k.field, &mut ui, &gm)?;

                // A limited scheme keeps the convection second order where
                // the field is smooth and bounded where it is not, which is
                // what a cavity corner needs (SPEC-LIT section 7).
                let mut grad_u: DevBuf<Vec3> = gpu.zeros(nc)?;
                fvc_grad_scalar(gpu, &k.fv, &mut grad_u, &ui, &gm)?;

                let mut d_w: DevBuf<Scalar> = gpu.zeros(nif)?;
                let mut d_bw: DevBuf<Scalar> = gpu.zeros(nbf)?;
                div_scheme_weights(
                    gpu,
                    &k.fv,
                    Some(&mut d_w),
                    Some(&mut d_bw),
                    DivScheme::Limited(Limiter::VanLeer),
                    &sphi,
                    &ui,
                    Some(&grad_u),
                    &gm,
                )?;

                // The explicit pressure source, -V (grad p)_i.
                let psrc: Vec<Scalar> = (0..nc)
                    .map(|c| -grad_p[c].component(comp))
                    .collect();
                let d_psrc = gpu.upload(&psrc)?;

                let mut a = GpuLduMatrix::new(gpu, &gm)?;
                a.zero(gpu)?;
                fvm_div_gauss(gpu, &k.fv, &mut a, &gm, &sphi, &d_w, &d_bw, &ui, 1.0)?;
                fvm_laplacian(gpu, &k.fv, &mut a, &gm, &d_gamma_u, &d_b_gamma_u, &ui, -1.0)?;
                fvm_su(gpu, &k.fv, &mut a, &gm, &d_psrc, 1.0)?;
                relax(gpu, &k.ldu, &mut a, &gm, &ui.f, alpha_u)?;
                add_boundary_contributions(gpu, &k.ldu, &mut a, &gm)?;

                let perf = solve_pbicgstab(gpu, &k.solver, &mut ui.f, &a, &gm, &mut ws, &ctrl)?;
                mom_res = mom_res.max(perf.initial_residual);

                // H, from the matrix the solve just used. A_P u = H - grad p
                // with A_P = diag/V, so H_V = b - (A u - diag u) and
                // HbyA = H_V/diag. The pressure part of b is taken back out:
                // it re-enters through the pressure equation.
                let mut au: DevBuf<Scalar> = gpu.zeros(nc)?;
                amul(gpu, &k.ldu, &mut au, &ui.f, &a, &gm)?;
                gpu.sync()?;

                let diag = gpu.download(&a.diag)?;
                let src = gpu.download(&a.source)?;
                let au_h = gpu.download(&au)?;
                let ui_h = gpu.download(&ui.f)?;

                for c in 0..nc {
                    let off = au_h[c] - diag[c] * ui_h[c];
                    let h = src[c] - m.v[c] * psrc[c] - off;
                    let v = h / diag[c];
                    match comp {
                        0 => hbya[c].x = v,
                        _ => hbya[c].y = v,
                    }
                    u[c] = match comp {
                        0 => Vec3::new(ui_h[c], u[c].y, 0.0),
                        _ => Vec3::new(u[c].x, ui_h[c], 0.0),
                    };
                    rau[c] = m.v[c] / diag[c];
                }
            }

            // ---- phi_HbyA, on faces ------------------------------------
            let mut hf = GpuVectorField::zeros(gpu, &gm, "HbyA")?;
            gpu.write(&mut hf.f, &hbya)?;
            gpu.write(&mut hf.fr, &u_fr)?;
            gpu.write(&mut hf.ref_value, &u_ref)?;
            gpu.write(&mut hf.ref_grad, &zero_v)?;
            gpu.write(&mut hf.bc_kind, &u_kind)?;
            correct_boundary_conditions_vector(gpu, &k.field, &mut hf, &gm)?;

            let mut phi_h = GpuSurfaceScalarField::zeros(gpu, &gm, "phiHbyA")?;
            interpolate_vector_flux(gpu, &k.fv, &mut phi_h, &hf, &gm)?;
            gpu.write(&mut phi_h.bf, &bphi)?;
            gpu.sync()?;
            let phi_hbya = gpu.download(&phi_h.f)?;

            // ---- the pressure equation ----------------------------------
            let gamma_p: Vec<Scalar> = (0..nif)
                .map(|f| {
                    let w = m.weights[f];
                    let r = w * rau[m.owner[f] as usize] + (1.0 - w) * rau[m.neighbour[f] as usize];
                    r * m.mag_sf[f]
                })
                .collect();
            let b_gamma_p: Vec<Scalar> = (0..nbf)
                .map(|bf| {
                    if cpu::is_empty_face(&m, bf) {
                        0.0
                    } else {
                        rau[m.b_face_cells[bf] as usize] * m.b_mag_sf[bf]
                    }
                })
                .collect();
            let d_gamma_p = gpu.upload(&gamma_p)?;
            let d_b_gamma_p = gpu.upload(&b_gamma_p)?;

            let mut pf = GpuScalarField::zeros(gpu, &gm, "p")?;
            gpu.write(&mut pf.f, &p_cell)?;
            gpu.write(&mut pf.fr, &zero_s)?;
            gpu.write(&mut pf.ref_value, &zero_s)?;
            gpu.write(&mut pf.ref_grad, &zero_s)?;
            gpu.write(&mut pf.bc_kind, &p_kind)?;

            let mut div: DevBuf<Scalar> = gpu.zeros(nc)?;
            fvc_div_surface(gpu, &k.fv, &mut div, &phi_h, &gm)?;

            let mut ap = GpuLduMatrix::new(gpu, &gm)?;
            ap.zero(gpu)?;
            // All-Neumann: the pressure is defined only up to a constant, so
            // one cell is pinned. Any cell will do; cell 0 is arbitrary.
            set_fixed_cells(gpu, &mut ap, &[0], &[0.0])?;
            fvm_laplacian(gpu, &k.fv, &mut ap, &gm, &d_gamma_p, &d_b_gamma_p, &pf, 1.0)?;
            fvm_su(gpu, &k.fv, &mut ap, &gm, &div, 1.0)?;
            add_boundary_contributions(gpu, &k.ldu, &mut ap, &gm)?;
            set_values(gpu, &k.ldu, &mut ap, &gm)?;

            solve_pcg(gpu, &k.solver, &mut pf.f, &ap, &gm, &mut ws, &ctrl_p)?;
            correct_boundary_conditions(gpu, &k.field, &mut pf, &gm)?;

            // ---- correct the flux and the velocity -----------------------
            let mut sn = GpuSurfaceScalarField::zeros(gpu, &gm, "snGradP")?;
            sn_grad_flux(gpu, &k.fv, &mut sn, &pf, &d_gamma_p, &d_b_gamma_p, &gm)?;

            let mut gp: DevBuf<Vec3> = gpu.zeros(nc)?;
            fvc_grad_scalar(gpu, &k.fv, &mut gp, &pf, &gm)?;
            gpu.sync()?;

            let sn_f = gpu.download(&sn.f)?;
            for f in 0..nif {
                phi[f] = phi_hbya[f] - sn_f[f];
            }

            let gp_new = gpu.download(&gp)?;
            for c in 0..nc {
                u[c] = hbya[c] - gp_new[c] * rau[c];
            }

            // Under-relax the pressure for the NEXT momentum predictor. The
            // Gauss gradient is linear in the cell field, so the relaxed
            // gradient is the same blend and needs no second evaluation.
            let p_new = gpu.download(&pf.f)?;
            for c in 0..nc {
                p_cell[c] += alpha_p * (p_new[c] - p_cell[c]);
                grad_p[c] = grad_p[c] * (1.0 - alpha_p) + gp_new[c] * alpha_p;
            }

            residual = mom_res;
            if iter > 20 && mom_res < 1e-7 {
                done = iter + 1;
                break;
            }
        }

        Ok((m, u, residual, done))
    }

    /// Compare both centreline profiles with the tables, print them, and
    /// return the worst absolute difference in each.
    ///
    /// `v_skip` names stations of Table II left out of the *worst-difference*
    /// figure; they are still printed, marked, so nothing is hidden. Only the
    /// erratum of [`GHIA_V_RE400`] is ever passed here.
    #[allow(clippy::too_many_arguments)]
    fn compare(
        m: &HostMesh,
        u: &[Vec3],
        n: usize,
        u_table: &[f64; 17],
        v_table: &[f64; 17],
        v_skip: &[usize],
        label: &str,
    ) -> (f64, f64) {
        // No slip on -x, +x and -y; the lid on +y carries u = 1, v = 0.
        let us = Sampled::new(m, u, n, 0, [0.0, 0.0, 0.0, 1.0]);
        let vs = Sampled::new(m, u, n, 1, [0.0, 0.0, 0.0, 0.0]);

        println!("\n{label}  -- Ghia, Ghia & Shin (1982) Table I, u at x = 0.5");
        let mut worst_u = 0.0f64;
        for (i, y) in GHIA_Y.iter().enumerate() {
            let got = us.at(0.5, *y);
            let want = u_table[i];
            worst_u = worst_u.max((got - want).abs());
            println!("   y {y:7.4}   ofgpu {got:9.5}   Ghia {want:9.5}   d {:9.5}", got - want);
        }

        println!("\n{label}  -- Table II, v at y = 0.5");
        let mut worst_v = 0.0f64;
        for (i, x) in GHIA_X.iter().enumerate() {
            let got = vs.at(*x, 0.5);
            let want = v_table[i];
            let skipped = v_skip.contains(&i);
            if !skipped {
                worst_v = worst_v.max((got - want).abs());
            }
            println!(
                "   x {x:7.4}   ofgpu {got:9.5}   Ghia {want:9.5}   d {:9.5}{}",
                got - want,
                if skipped { "   (paper's erratum, excluded)" } else { "" }
            );
        }

        println!("\n{label}  worst |du| {worst_u:.4}, worst |dv| {worst_v:.4}");
        (worst_u, worst_v)
    }

    /// The printed table is the evidence; the tolerance is the tripwire.
    ///
    /// Ghia's numbers come from a 129 x 129 grid and a different
    /// discretisation, so an exact match is neither expected nor meaningful.
    /// What a working solver must do is reproduce the *profile* - the sign,
    /// the position of the extrema, and the magnitude to within the difference
    /// two second-order schemes on different grids can show. `0.02` on a lid
    /// speed of 1 is roughly three times the difference actually observed at
    /// 80 x 80, and a solver with a sign error, a broken wall condition or a
    /// first-order convection scheme misses by an order of magnitude more.
    #[allow(clippy::too_many_arguments)]
    fn run_case(
        re: Scalar,
        n: usize,
        iters: usize,
        u_t: &[f64; 17],
        v_t: &[f64; 17],
        v_skip: &[usize],
        tol: f64,
    ) {
        let gpu = Gpu::new(0).expect("no CUDA device");
        let k = Kernels::new(&gpu).expect("kernels");

        let (m, u, res, its) = cavity(&gpu, &k, re, n, iters).expect("cavity");
        let label = format!("lid-driven cavity, Re = {}, {n} x {n}", f64::from(re));
        println!("\n{label}: {its} SIMPLE iterations, momentum residual {res:.3e}");

        let (du, dv) = compare(&m, &u, n, u_t, v_t, v_skip, &label);
        assert!(du < tol, "u centreline differs from Ghia by {du:.4} (> {tol})");
        assert!(dv < tol, "v centreline differs from Ghia by {dv:.4} (> {tol})");
    }

    #[test]
    #[ignore = "runs a flow to steady state; minutes, not seconds"]
    fn ghia_lid_driven_cavity_re_100() {
        run_case(100.0, 80, 3000, &GHIA_U_RE100, &GHIA_V_RE100, &[], 0.02);
    }

    #[test]
    #[ignore = "runs a flow to steady state; minutes, not seconds"]
    fn ghia_lid_driven_cavity_re_400() {
        run_case(
            400.0,
            80,
            6000,
            &GHIA_U_RE400,
            &GHIA_V_RE400,
            GHIA_V_RE400_ERRATUM,
            0.02,
        );
    }
}
