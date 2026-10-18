// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! `ofgpu-bench` - how fast do the two models actually run, and how much of
//! the card do they use?
//!
//! Builds a uniform block of the requested size, imposes a shear flow whose
//! face flux is exactly divergence free by construction, and iterates each
//! model. Reports ms per outer iteration, cell-iterations per second, and the
//! resident device memory - the number that decides how large a mesh fits.
//!
//! ```text
//! ofgpu-bench [nx ny nz] [-iters N] [-fixedIters N] [-model kEpsilon|kOmega|both|simpleCavity]
//! ```
//!
//! `simpleCavity` times the crate's own steady SIMPLE loop on a 3-D
//! lid-driven cavity instead of the frozen-flow turbulence models, for the
//! f32 speed figure of SPEC-LIT §112.6.
//!
//! Carried across from this project's own earlier C++ benchmark driver.
//!
//! Provenance: ORIGINAL - a benchmark harness. It measures this crate's own
//! kernels and reports; there is no external source for it (`PROVENANCE.md`,
//! *GPU plumbing and tooling - original*, `src/bin/*`). No GPL-licensed source
//! was consulted.

use std::process::ExitCode;
use std::time::Instant;

use cudarc::driver::sys::CUdevice_attribute;

use ofgpu::blockgen;
use ofgpu::blockgen::{BlockSpec, GradedAxis};
use ofgpu::field::{BcKind, GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use ofgpu::field_ops::{
    correct_boundary_conditions, correct_boundary_conditions_vector, FieldKernels,
};
use ofgpu::field_setup::max_div_phi;
use ofgpu::io::case::SolverControls;
use ofgpu::mesh::{HostMesh, PatchKind};
use ofgpu::models::{KEpsilon, KEpsilonCoeffs, KOmega, KOmegaCoeffs};
use ofgpu::momentum::{BuoyancyCoeffs, MomentumControls};
use ofgpu::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use ofgpu::simple::{Simple, SimpleControls, SimplePerformance};
use ofgpu::turbulence::{FlowState, TurbulenceControls};
use ofgpu::wallfunctions::WallFunctionCoeffs;
use ofgpu::{Gpu, GpuMesh, Label, Result, Scalar, Vec3};

#[path = "common/mod.rs"]
mod common;

use common::{atoi, precision_name, resident_mib, sci};

// ==========================================================================
//  The memory model, SPEC-LIT 111.3 and 111.4
// ==========================================================================

/// Pool bytes per cell of the mesh and the frozen flow, measured (§111.3).
const MESH_FLOW_BYTES_PER_CELL: f64 = 622.2;
/// Pool bytes per cell of the k-epsilon model, constructor and `init` (§111.3).
const KEPSILON_BYTES_PER_CELL: f64 = 494.3;
/// The whole case, per cell.
const CASE_BYTES_PER_CELL: f64 = MESH_FLOW_BYTES_PER_CELL + KEPSILON_BYTES_PER_CELL;
/// The fixed part, MiB (measured 55, rounded up).
const POOL_INTERCEPT_MIB: f64 = 64.0;
/// The sizes of SPEC-LIT 111.4's table, GiB of pool memory.
#[cfg(test)]
const FIT_TABLE_GIB: [u32; 8] = [2, 4, 6, 8, 10, 12, 14, 16];

/// Cells of the k-epsilon benchmark case that fit in `gib` GiB of pool memory.
#[cfg(test)]
fn fits_in(gib: f64) -> usize {
    let bytes = gib * 1_073_741_824.0 - POOL_INTERCEPT_MIB * 1_048_576.0;
    (bytes / CASE_BYTES_PER_CELL).floor().max(0.0) as usize
}

/// One row of SPEC-LIT 111.4's table, exactly as that section prints it.
#[cfg(test)]
fn fit_table_row(gib: u32) -> String {
    format!("| {gib} GiB | {:.2} M cells |", fits_in(f64::from(gib)) as f64 / 1e6)
}

/// Pool bytes the whole k-epsilon case is predicted to hold at `n_cells`.
fn predicted_case_bytes(n_cells: usize) -> f64 {
    POOL_INTERCEPT_MIB * 1_048_576.0 + CASE_BYTES_PER_CELL * n_cells as f64
}

// ==========================================================================
//  The frozen flow every model is run against
// ==========================================================================

struct Bench {
    hm: HostMesh,
    mesh: GpuMesh,
    u: GpuVectorField,
    phi: GpuSurfaceScalarField,
    wf_faces: ofgpu::field_setup::WallFaces,
    nu: Scalar,
}

fn build_flow(gpu: &Gpu, fk: &FieldKernels, nx: usize, ny: usize, nz: usize) -> Result<Bench> {
    let spec = BlockSpec {
        x: GradedAxis { lo: 0.0, hi: 4.0, n: nx, ..Default::default() },
        y: GradedAxis {
            lo: -1.0,
            hi: 1.0,
            n: ny,
            expansion: 20.0,
            two_sided: true,
        },
        z: GradedAxis { lo: 0.0, hi: 0.5, n: nz, ..Default::default() },
        windows: Vec::new(),
        patch_name: ["inlet", "outlet", "lowerWall", "upperWall", "zMin", "zMax"]
            .map(String::from),
        patch_type: [
            "patch",
            "patch",
            "wall",
            "wall",
            if nz == 1 { "empty" } else { "wall" },
            if nz == 1 { "empty" } else { "wall" },
        ]
        .map(String::from),
        cyclic: Vec::new(),
    };

    let hm = blockgen::build_mesh(&spec)?;
    let mesh = GpuMesh::upload(gpu, &hm)?;

    // A 1/7-power-law channel profile in x only. div(U) is not exactly zero
    // cell by cell for a varying profile, so build phi from the CONSTANT part
    // and let the shear enter only through grad(U) - that keeps the flux
    // discretely conservative while still producing realistic production.
    let uc: Vec<Vec3> = (0..hm.n_cells)
        .map(|c| {
            let yy = (1.0 - 1e-9 as Scalar).min(f64::from(hm.c[c].y).abs() as Scalar);
            let u = (1.0 - f64::from(yy)).powf(1.0 / 7.0) as Scalar;
            Vec3::new(u, 0.0, 0.0)
        })
        .collect();

    let mut u = GpuVectorField::zeros(gpu, &mesh, "U")?;
    gpu.write(&mut u.f, &uc)?;

    let n_bf = hm.n_boundary_faces;
    let mut fr = vec![0.0 as Scalar; n_bf];
    let mut rv = vec![Vec3::ZERO; n_bf];
    let rg = vec![Vec3::ZERO; n_bf];
    let mut kind = vec![BcKind::ZeroGradient as Label; n_bf];

    for p in &hm.patches {
        for i in 0..p.size {
            let bf = p.start + i;

            if p.kind == PatchKind::Empty {
                kind[bf] = BcKind::Empty as Label;
            } else if p.kind == PatchKind::Wall {
                kind[bf] = BcKind::FixedValue as Label; // no slip
                fr[bf] = 1.0;
                rv[bf] = Vec3::ZERO;
            } else if p.name == "inlet" {
                kind[bf] = BcKind::FixedValue as Label;
                fr[bf] = 1.0;
                rv[bf] = uc[hm.b_face_cells[bf] as usize];
            }
        }
    }

    gpu.write(&mut u.fr, &fr)?;
    gpu.write(&mut u.ref_value, &rv)?;
    gpu.write(&mut u.ref_grad, &rg)?;
    gpu.write(&mut u.bc_kind, &kind)?;
    correct_boundary_conditions_vector(gpu, fk, &mut u, &mesh)?;

    // phi from a uniform velocity: exactly divergence free on a closed cell.
    let mut phi = GpuSurfaceScalarField::zeros(gpu, &mesh, "phi")?;
    {
        let u_bulk = Vec3::new(0.9, 0.0, 0.0);

        let phi_i: Vec<Scalar> = (0..hm.n_internal_faces)
            .map(|f| u_bulk.dot(hm.sf[f]))
            .collect();

        let phi_b: Vec<Scalar> = (0..n_bf)
            .map(|i| {
                if hm.b_kind[i] == PatchKind::Empty as Label {
                    0.0
                } else {
                    u_bulk.dot(hm.b_sf[i])
                }
            })
            .collect();

        gpu.write(&mut phi.f, &phi_i)?;
        gpu.write(&mut phi.bf, &phi_b)?;
    }

    // A synthetic benchmark has no case files, so the geometry decides: every
    // wall patch gets a wall function on both the dissipation and nu_t. A real
    // case must read the two sets from the two fields (SPEC-LIT 15.5); this
    // one has neither field to read.
    let mut on = vec![false; n_bf];
    for p in &hm.patches {
        if p.kind != PatchKind::Wall {
            continue;
        }
        for i in 0..p.size {
            on[p.start + i] = true;
        }
    }
    let wf_faces = ofgpu::field_setup::WallFaces {
        constrained_cells: on.clone(),
        nut: on,
    };

    Ok(Bench { hm, mesh, u, phi, wf_faces, nu: 1e-5 })
}

/// A uniform field with the boundary types the benchmark's geometry implies.
///
/// `wall_fn` selects the zeroGradient wall treatment a wall-function case
/// uses. It writes what the default already is, and is kept because the
/// alternative — a wall patch that reaches the `inlet` branch — is what the
/// branch order is there to prevent.
fn init_scalar(
    gpu: &Gpu,
    f: &mut GpuScalarField,
    m: &HostMesh,
    name: &str,
    value: Scalar,
    wall_fn: bool,
) -> Result<()> {
    f.name = name.to_string();

    let internal = vec![value; m.n_cells];
    gpu.write(&mut f.f, &internal)?;
    gpu.write(&mut f.f0, &internal)?;

    let n_bf = m.n_boundary_faces;
    let mut fr = vec![0.0 as Scalar; n_bf];
    let rv = vec![value; n_bf];
    let rg = vec![0.0 as Scalar; n_bf];
    let mut kind = vec![BcKind::ZeroGradient as Label; n_bf];

    for p in &m.patches {
        for i in 0..p.size {
            let bf = p.start + i;

            if p.kind == PatchKind::Empty {
                kind[bf] = BcKind::Empty as Label;
            } else if p.kind == PatchKind::Wall && wall_fn {
                kind[bf] = BcKind::ZeroGradient as Label;
            } else if p.name == "inlet" {
                kind[bf] = BcKind::FixedValue as Label;
                fr[bf] = 1.0;
            }
        }
    }

    gpu.write(&mut f.fr, &fr)?;
    gpu.write(&mut f.ref_value, &rv)?;
    gpu.write(&mut f.ref_grad, &rg)?;
    gpu.write(&mut f.bc_kind, &kind)?;
    gpu.write(&mut f.bf, &rv)?;

    Ok(())
}

// ==========================================================================
//  Timing
// ==========================================================================

/// The two models expose the same shape but share no trait in the library —
/// deliberately, because a `dyn` call has no place in a solver's inner loop.
/// Timing them side by side is the one place where uniformity is worth more
/// than the indirection costs, and one virtual call per *outer iteration*
/// disappears into the noise of the hundred kernel launches it wraps.
trait BenchModel {
    fn init(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()>;
    fn step(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()>;
}

impl BenchModel for KEpsilon<'_> {
    fn init(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        self.initialise(gpu, flow)
    }
    fn step(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        self.correct(gpu, flow).map(|_| ())
    }
}

impl BenchModel for KOmega<'_> {
    fn init(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        self.initialise(gpu, flow)
    }
    fn step(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        self.correct(gpu, flow).map(|_| ())
    }
}

/// Both readings: this process's own pool first (SPEC-LIT 111.2), then the
/// whole card, which also counts every other process on it.
fn report_memory(gpu: &Gpu, tag: &str, n_cells: usize) -> Result<()> {
    let own = gpu.pool_usage()?.used;
    let (used, total) = resident_mib(gpu)?;
    println!(
        "       {tag}: {} MiB in this process's pool, {:.1} B/cell | {used} MiB resident on the whole card of {total} MiB",
        own >> 20,
        own as f64 / n_cells.max(1) as f64
    );
    Ok(())
}

fn run_model(
    gpu: &Gpu,
    name: &str,
    model: &mut dyn BenchModel,
    flow: &FlowState,
    n_iters: usize,
    n_cells: usize,
) -> Result<()> {
    model.init(gpu, flow)?;
    gpu.sync()?;

    report_memory(gpu, name, n_cells)?;

    // Warm-up: the first launch of every kernel pays for module loading.
    for _ in 0..3 {
        model.step(gpu, flow)?;
    }
    gpu.sync()?;

    let t0 = Instant::now();
    for _ in 0..n_iters {
        model.step(gpu, flow)?;
    }
    gpu.sync()?;
    let wall = t0.elapsed().as_secs_f64();

    let n = n_iters.max(1) as f64;

    println!(
        "       {name:<10}{:.3} ms/iter    {:.1} Mcell-iter/s",
        (wall / n) * 1e3,
        (n_cells as f64 * n / wall) / 1e6
    );

    Ok(())
}

/// The k-epsilon model the benchmark times, built and seeded exactly as
/// `run` has always built it. Shared with the memory-model test.
fn kepsilon_for<'a>(gpu: &Gpu, b: &'a Bench, ctrl: TurbulenceControls) -> Result<KEpsilon<'a>> {
    // A synthetic benchmark geometry, not a case file - no `nut` field to
    // read `Ks`/`Cs` from, so every wall face is smooth.
    let no_roughness = ofgpu::field_setup::NutRoughness::none(b.hm.n_boundary_faces);
    let mut model = KEpsilon::new(
        gpu,
        &b.hm,
        &b.mesh,
        KEpsilonCoeffs::default(),
        ctrl,
        WallFunctionCoeffs::default(),
        &b.wf_faces,
        &no_roughness,
    )?;

    init_scalar(gpu, model.k_mut(), &b.hm, "k", 0.01, true)?;
    init_scalar(gpu, model.epsilon_mut(), &b.hm, "epsilon", 0.1, true)?;
    init_scalar(gpu, model.nut_mut(), &b.hm, "nut", 0.0, true)?;

    Ok(model)
}

// ==========================================================================
//  The SIMPLE cavity, F32-18
// ==========================================================================

/// The +y patch of a blockgen box: the six patches come out in
/// -x +x -y +y -z +z order, so the moving lid is the fourth.
const LID_PATCH: usize = 3;

/// What one timed SIMPLE cavity run measured.
struct SimpleCavityTiming {
    n_cells: usize,
    /// `gpu.pool_usage()?.used` after `Simple::new` + `initialise` + the
    /// pressure backend's `setup` and the `n_warm` untimed iterations.
    pool_bytes: u64,
    /// Wall seconds per timed `correct()`, `gpu.sync()` before and after the
    /// timed loop.
    secs_per_iter: f64,
    /// Mean `SolverPerformance::n_iterations` of the three U components over
    /// the timed iterations.
    mean_u_iters: [f64; 3],
    /// The same for p.
    mean_p_iters: f64,
    /// `SimplePerformance::continuity_error` of the last timed iteration.
    continuity_error: Scalar,
}

/// Every boundary face's patch index - how this file says "this face is the
/// lid's" without re-deriving the flattening.
fn patch_of(hm: &HostMesh) -> Vec<usize> {
    let mut v = vec![0usize; hm.n_boundary_faces];
    for (p, pi) in hm.patches.iter().enumerate() {
        for k in 0..pi.size {
            v[pi.start + k] = p;
        }
    }
    v
}

/// Zero-gradient on every wall, `empty` where the mesh says so - the
/// boundary condition both p and T carry on this case.
fn set_zero_gradient_bcs(gpu: &Gpu, f: &mut GpuScalarField, hm: &HostMesh) -> Result<()> {
    let owner = patch_of(hm);
    let nbf = hm.n_boundary_faces;

    let mut kind = vec![BcKind::ZeroGradient as Label; nbf];
    let fr = vec![0.0 as Scalar; nbf];
    let rv = vec![0.0 as Scalar; nbf];
    let rg = vec![0.0 as Scalar; nbf];

    for i in 0..nbf {
        if hm.patches[owner[i]].kind == PatchKind::Empty {
            kind[i] = BcKind::Empty as Label;
        }
    }

    gpu.write(&mut f.bc_kind, &kind)?;
    gpu.write(&mut f.fr, &fr)?;
    gpu.write(&mut f.ref_value, &rv)?;
    gpu.write(&mut f.ref_grad, &rg)
}

/// Time the crate's own steady SIMPLE loop on a lid-driven cavity.
///
/// The case follows the library's Ghia-Ghia-Shin cavity test
/// (`lid_driven_cavity_matches_ghia_ghia_and_shin`, 2-D, central
/// differencing) on the bench's mesh, with `MomentumControls`' default
/// schemes: the unit cube, laminar (`nu = 0.01`, `Re = U L / nu = 100`), the
/// lid (+y) moving at (1, 0, 0) and every other wall stationary. It is
/// modelled on the private test helpers of `src/simple.rs` and that test;
/// they are `#[cfg(test)]`-private to the library, so these are small
/// copies, not shared ones. `n_warm` untimed `correct()` calls come first
/// (the first launch of every kernel pays for module loading), then
/// `n_iters` timed ones.
fn time_simple_cavity(
    gpu: &Gpu,
    nx: usize,
    ny: usize,
    nz: usize,
    n_warm: usize,
    n_iters: usize,
    fixed_iters: i64,
) -> Result<SimpleCavityTiming> {
    let spec = BlockSpec {
        x: GradedAxis { lo: 0.0, hi: 1.0, n: nx, ..Default::default() },
        y: GradedAxis { lo: 0.0, hi: 1.0, n: ny, ..Default::default() },
        z: GradedAxis { lo: 0.0, hi: 1.0, n: nz, ..Default::default() },
        windows: Vec::new(),
        patch_name: ["xMin", "xMax", "yMin", "lid", "zMin", "zMax"].map(String::from),
        patch_type: [
            "wall",
            "wall",
            "wall",
            "wall",
            if nz == 1 { "empty" } else { "wall" },
            if nz == 1 { "empty" } else { "wall" },
        ]
        .map(String::from),
        cyclic: Vec::new(),
    };

    let hm = blockgen::build_mesh(&spec)?;
    let mesh = GpuMesh::upload(gpu, &hm)?;

    // Re = U L / nu = 1 * 1 / 0.01 = 100.
    let nu: Scalar = 0.01;

    let (u_solver, p_solver) = if fixed_iters > 0 {
        // Zero host traffic per solve: the count is the control, not a
        // measurement.
        let both = SolverControls {
            fixed_iters: true,
            max_iter: fixed_iters as Label,
            report_residuals: false,
            ..SolverControls::default()
        };
        (both, both)
    } else {
        (
            SolverControls {
                tolerance: 1e-8,
                rel_tol: 0.1,
                max_iter: 200,
                check_interval: 5,
                ..SolverControls::default()
            },
            SolverControls {
                tolerance: 1e-8,
                rel_tol: 0.05,
                max_iter: 1000,
                check_interval: 5,
                ..SolverControls::default()
            },
        )
    };

    let mut s = Simple::new(
        gpu,
        &hm,
        &mesh,
        SimpleControls {
            momentum: MomentumControls {
                nu,
                u_solver,
                u_relax: 0.7,
                ..MomentumControls::default()
            },
            p_solver,
            p_relax: 0.3,
            ..SimpleControls::default()
        },
        BuoyancyCoeffs { g: Vec3::ZERO, ..BuoyancyCoeffs::default() },
    )?;

    // U: no slip on every stationary wall, the lid moving in +x; an `empty`
    // patch stays `empty`. p: zero-gradient everywhere.
    let owner = patch_of(&hm);
    let nbf = hm.n_boundary_faces;
    let lid = Vec3::new(1.0, 0.0, 0.0);

    let mut kind = vec![BcKind::FixedValue as Label; nbf];
    let mut fr = vec![1.0 as Scalar; nbf];
    let mut rv = vec![Vec3::ZERO; nbf];
    let rg = vec![Vec3::ZERO; nbf];
    for i in 0..nbf {
        if hm.patches[owner[i]].kind == PatchKind::Empty {
            kind[i] = BcKind::Empty as Label;
            fr[i] = 0.0;
        } else if owner[i] == LID_PATCH {
            rv[i] = lid;
        }
    }

    {
        let u = s.u_mut();
        gpu.write(&mut u.bc_kind, &kind)?;
        gpu.write(&mut u.fr, &fr)?;
        gpu.write(&mut u.ref_value, &rv)?;
        gpu.write(&mut u.ref_grad, &rg)?;
    }
    set_zero_gradient_bcs(gpu, s.p_mut(), &hm)?;
    s.initialise(gpu)?;

    // A zero eddy viscosity with calculated boundary values - laminar.
    let mut nut = GpuScalarField::zeros(gpu, &mesh, "nut")?;
    let calculated = vec![BcKind::Calculated as Label; nbf];
    gpu.write(&mut nut.bc_kind, &calculated)?;

    // A uniform temperature with the case's zero-gradient BCs, evaluated
    // once: the momentum equation reads it whether or not g is zero.
    let t_ref = BuoyancyCoeffs::default().t_ref;
    let mut t = GpuScalarField::zeros(gpu, &mesh, "T")?;
    let uniform = vec![t_ref; hm.n_cells];
    gpu.write(&mut t.f, &uniform)?;
    set_zero_gradient_bcs(gpu, &mut t, &hm)?;
    correct_boundary_conditions(gpu, &FieldKernels::new(gpu)?, &mut t, &mesh)?;

    let mut backend = PbicgstabBackend::new(p_solver);
    backend.setup(gpu, &hm, &mesh, &SystemProbe::default())?;

    for _ in 0..n_warm {
        s.correct(gpu, &mut backend, &nut, &t)?;
    }

    // Read after the warm-up, not before it: a solve's workspace is
    // allocated on first use, and the figure is the whole case's footprint.
    gpu.sync()?;
    let pool_bytes = gpu.pool_usage()?.used;

    let mut sum_u = [0.0f64; 3];
    let mut sum_p = 0.0f64;
    let mut last = SimplePerformance::default();

    gpu.sync()?;
    let t0 = Instant::now();
    for _ in 0..n_iters {
        last = s.correct(gpu, &mut backend, &nut, &t)?;
        for (sum, perf) in sum_u.iter_mut().zip(last.u) {
            *sum += perf.n_iterations as f64;
        }
        sum_p += last.p.n_iterations as f64;
    }
    gpu.sync()?;
    let wall = t0.elapsed().as_secs_f64();

    let n = n_iters.max(1) as f64;

    Ok(SimpleCavityTiming {
        n_cells: hm.n_cells,
        pool_bytes,
        secs_per_iter: wall / n,
        mean_u_iters: [sum_u[0] / n, sum_u[1] / n, sum_u[2] / n],
        mean_p_iters: sum_p / n,
        continuity_error: last.continuity_error,
    })
}

// ==========================================================================
//  Driver
// ==========================================================================

struct Options {
    nx: usize,
    ny: usize,
    nz: usize,
    n_iters: usize,
    fixed_iters: i64,
    which: String,
}

fn parse(args: &[String]) -> Options {
    let mut o = Options {
        nx: 400,
        ny: 200,
        nz: 1,
        n_iters: 50,
        fixed_iters: 0,
        which: "both".to_string(),
    };

    let mut pos = 0usize;
    let mut i = 1usize;

    while i < args.len() {
        let has_next = i + 1 < args.len();

        if args[i] == "-iters" && has_next {
            i += 1;
            o.n_iters = atoi(&args[i]).max(0) as usize;
        } else if args[i] == "-fixedIters" && has_next {
            i += 1;
            o.fixed_iters = atoi(&args[i]);
        } else if args[i] == "-model" && has_next {
            i += 1;
            o.which = args[i].clone();
        } else {
            let v = atoi(&args[i]).max(1) as usize;
            match pos {
                0 => o.nx = v,
                1 => o.ny = v,
                2 => o.nz = v,
                _ => {}
            }
            pos += 1;
        }

        i += 1;
    }

    o
}

fn run(o: &Options) -> Result<()> {
    let gpu = Gpu::new(0)?;
    let ctx = gpu.ctx();

    let (major, minor) = ctx.compute_capability()?;
    let sms = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    let bus = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)?;

    // `memoryClockRate` was removed from `cudaDeviceProp` in CUDA 13; the
    // attribute query is the supported way to get it, and is what the driver
    // API offers in any case.
    let mem_clk_khz = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
        .unwrap_or(0);

    let peak_bw = 2.0 * f64::from(mem_clk_khz) * 1e3 * f64::from(bus) / 8.0 / 1e9;

    println!(
        "ofgpu benchmark | {} sm_{major}{minor} | {sms} SMs | {bus}-bit, {peak_bw:.0} GB/s peak | {}",
        ctx.name()?,
        precision_name()
    );

    println!(
        "building {} x {} x {} = {} cells ...",
        o.nx,
        o.ny,
        o.nz,
        o.nx * o.ny * o.nz
    );

    // The SIMPLE cavity does not touch the frozen flow: its case is the
    // library's own lid-driven cavity, built inside `time_simple_cavity`.
    if o.which == "simpleCavity" {
        if o.fixed_iters > 0 {
            println!(
                "       fixed-iteration solver: {} sweeps, zero host transfers",
                o.fixed_iters
            );
        }

        let t = time_simple_cavity(&gpu, o.nx, o.ny, o.nz, 3, o.n_iters, o.fixed_iters)?;

        println!(
            "       simpleCavity: {} MiB in this process's pool, {:.1} B/cell",
            t.pool_bytes >> 20,
            t.pool_bytes as f64 / t.n_cells.max(1) as f64
        );

        let rate = if t.secs_per_iter > 0.0 {
            t.n_cells as f64 / t.secs_per_iter / 1e6
        } else {
            0.0
        };
        println!(
            "       simpleCavity  {:.4} s/iter    {:.1} Mcell-iter/s",
            t.secs_per_iter, rate
        );

        println!(
            "       inner iterations per outer iteration (mean): U {:.1} {:.1} {:.1}, p {:.1}; continuity error {}",
            t.mean_u_iters[0],
            t.mean_u_iters[1],
            t.mean_u_iters[2],
            t.mean_p_iters,
            sci(f64::from(t.continuity_error), 1)
        );

        return Ok(());
    }

    let fk = FieldKernels::new(&gpu)?;
    let b = build_flow(&gpu, &fk, o.nx, o.ny, o.nz)?;

    println!(
        "       {} cells, {} internal faces, {} boundary faces",
        b.hm.n_cells, b.hm.n_internal_faces, b.hm.n_boundary_faces
    );

    // Precision 0, not 6: the peak-bandwidth field above set `setprecision(0)`
    // on the C++ stream and it is sticky, so this number really does print as
    // `7e-18` there rather than `7.000000e-18`.
    println!(
        "       max |sum_f phi| per cell = {}",
        sci(f64::from(max_div_phi(&gpu, &b.phi, &b.hm)?), 0)
    );

    report_memory(&gpu, "mesh only", b.hm.n_cells)?;

    println!(
        "       memory model (SPEC-LIT 111.3): mesh + flow + k-epsilon predicted at {} MiB, {CASE_BYTES_PER_CELL:.1} B/cell",
        (predicted_case_bytes(b.hm.n_cells) / 1_048_576.0).round() as u64
    );

    let mut ctrl = TurbulenceControls {
        steady: true,
        ..Default::default()
    };
    ctrl.k_solver.tolerance = 1e-8;
    ctrl.k_solver.rel_tol = 0.1;
    ctrl.k_solver.max_iter = 200;
    // Nothing here prints a residual, so do not pay for reading one.
    ctrl.k_solver.report_residuals = false;
    ctrl.epsilon_solver = ctrl.k_solver;

    if o.fixed_iters > 0 {
        ctrl.k_solver.fixed_iters = true;
        ctrl.k_solver.max_iter = o.fixed_iters as Label;
        ctrl.epsilon_solver = ctrl.k_solver;
        println!(
            "       fixed-iteration solver: {} sweeps, zero host transfers",
            o.fixed_iters
        );
    }

    let flow = FlowState::new(&b.u, &b.phi, b.nu);
    let wc = WallFunctionCoeffs::default();
    // A synthetic benchmark geometry, not a case file - no `nut` field to
    // read `Ks`/`Cs` from, so every wall face is smooth.
    let no_roughness = ofgpu::field_setup::NutRoughness::none(b.hm.n_boundary_faces);

    if o.which == "kEpsilon" || o.which == "both" {
        let mut model = kepsilon_for(&gpu, &b, ctrl)?;
        run_model(&gpu, "kEpsilon", &mut model, &flow, o.n_iters, b.hm.n_cells)?;
    }

    if o.which == "kOmega" || o.which == "both" {
        let mut model = KOmega::new(
            &gpu,
            &b.hm,
            &b.mesh,
            KOmegaCoeffs::default(),
            ctrl,
            wc,
            &b.wf_faces,
            &no_roughness,
        )?;

        init_scalar(&gpu, model.k_mut(), &b.hm, "k", 0.01, true)?;
        init_scalar(&gpu, model.omega_mut(), &b.hm, "omega", 10.0, true)?;
        init_scalar(&gpu, model.nut_mut(), &b.hm, "nut", 0.0, true)?;

        run_model(&gpu, "kOmega", &mut model, &flow, o.n_iters, b.hm.n_cells)?;
    }

    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let o = parse(&args);

    match run(&o) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("\nbenchmark aborted: {e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// The pool cost of the k-epsilon case is linear in the cell count and
    /// matches the constants of SPEC-LIT 111.3, so the "cells that fit"
    /// table stays an interpolation rather than a guess.
    #[test]
    #[cfg(not(feature = "single"))]
    fn the_kepsilon_case_costs_linear_bytes_per_cell() {
        let Some(gpu) = gpu() else {
            eprintln!("no CUDA device: the memory model was not measured");
            return;
        };
        let (free, _) = gpu.mem_info().unwrap();
        assert!(
            free >= 2 << 30,
            "the memory-model test needs 2 GiB free and the card reports {} MiB free: it refuses rather than measure a card that is paging (SPEC-LIT 111.3)",
            free >> 20
        );
        let fk = FieldKernels::new(&gpu).unwrap();
        let mut rows: Vec<(usize, u64, u64)> = Vec::new();
        for nz in [2usize, 4, 6] {
            gpu.sync().unwrap();
            let base = gpu.pool_usage().unwrap().used;
            let b = build_flow(&gpu, &fk, 400, 250, nz).unwrap();
            gpu.sync().unwrap();
            let mesh = gpu.pool_usage().unwrap().used - base;
            {
                let ctrl = TurbulenceControls { steady: true, ..Default::default() };
                let mut model = kepsilon_for(&gpu, &b, ctrl).unwrap();
                let flow = FlowState::new(&b.u, &b.phi, b.nu);
                BenchModel::init(&mut model, &gpu, &flow).unwrap();
                gpu.sync().unwrap();
                let total = gpu.pool_usage().unwrap().used - base;
                rows.push((b.hm.n_cells, mesh, total - mesh));
            }
        }
        let cells: Vec<usize> = rows.iter().map(|r| r.0).collect();
        let mesh_s01 = (rows[1].1 - rows[0].1) as f64 / (rows[1].0 - rows[0].0) as f64;
        let mesh_s12 = (rows[2].1 - rows[1].1) as f64 / (rows[2].0 - rows[1].0) as f64;
        let model_s01 = (rows[1].2 - rows[0].2) as f64 / (rows[1].0 - rows[0].0) as f64;
        let model_s12 = (rows[2].2 - rows[1].2) as f64 / (rows[2].0 - rows[1].0) as f64;
        println!(
            "memory model: mesh+flow {:.1} / {:.1} B/cell, k-epsilon {:.1} / {:.1} B/cell over {:?} cells",
            mesh_s01, mesh_s12, model_s01, model_s12, cells
        );
        for (s01, s12, name) in
            [(mesh_s01, mesh_s12, "mesh+flow"), (model_s01, model_s12, "k-epsilon")]
        {
            assert!(
                (s01 - s12).abs() <= 0.01 * s12,
                "{name}'s bytes per cell are not linear: {s01:.1} then {s12:.1} B/cell over the three sizes"
            );
        }
        assert!(
            (mesh_s12 - MESH_FLOW_BYTES_PER_CELL).abs() <= 0.02 * MESH_FLOW_BYTES_PER_CELL,
            "the mesh+flow slope {mesh_s12:.1} B/cell is outside 2% of the measured constant {MESH_FLOW_BYTES_PER_CELL:.1} B/cell (SPEC-LIT 111.3)"
        );
        assert!(
            (model_s12 - KEPSILON_BYTES_PER_CELL).abs() <= 0.02 * KEPSILON_BYTES_PER_CELL,
            "the k-epsilon slope {model_s12:.1} B/cell is outside 2% of the measured constant {KEPSILON_BYTES_PER_CELL:.1} B/cell (SPEC-LIT 111.3)"
        );
    }

    /// SPEC-LIT 111.4's table is rebuilt here from the code's constants: the
    /// published table cannot drift from the model, and neither can the
    /// constants of §111.3.
    #[test]
    fn the_fit_table_in_spec_lit_is_the_one_the_model_computes() {
        const SPEC: &str = include_str!("../../SPEC-LIT.md");
        for g in FIT_TABLE_GIB {
            let row = fit_table_row(g);
            assert!(
                SPEC.contains(&row),
                "SPEC-LIT 111.4's table lost the row the model computes: {row}"
            );
        }
        for constant in [
            format!("**{MESH_FLOW_BYTES_PER_CELL:.1} B/cell**"),
            format!("**{KEPSILON_BYTES_PER_CELL:.1} B/cell**"),
            format!("**{CASE_BYTES_PER_CELL:.1} B/cell**"),
            format!("**{POOL_INTERCEPT_MIB:.0} MiB**"),
        ] {
            assert!(
                SPEC.contains(&constant),
                "SPEC-LIT 111.3 lost the constant the code computes: {constant}"
            );
        }
        assert_eq!(fit_table_row(8), "| 8 GiB | 7.63 M cells |");
    }

    /// F32-18: the simpleCavity bench runs end to end and reports finite
    /// figures on both builds, and a fixed-iteration pressure solve reports
    /// exactly the sweeps its control names - the count is the control, not
    /// a measurement (class D).
    #[test]
    fn the_simple_cavity_bench_runs_and_reports_finite_figures() {
        let Some(g) = gpu() else {
            eprintln!("no CUDA device: the simpleCavity bench was not exercised");
            return;
        };

        let t = time_simple_cavity(&g, 12, 12, 12, 1, 4, 0).unwrap();
        assert_eq!(t.n_cells, 1728, "n_cells = {}", t.n_cells);
        assert!(t.pool_bytes > 0, "pool_bytes = {}", t.pool_bytes);
        assert!(
            t.secs_per_iter.is_finite() && t.secs_per_iter > 0.0,
            "secs_per_iter = {}",
            t.secs_per_iter
        );
        assert!(
            t.continuity_error.is_finite(),
            "continuity_error = {}",
            t.continuity_error
        );
        assert!(t.mean_p_iters >= 1.0, "mean_p_iters = {}", t.mean_p_iters);

        let fixed = time_simple_cavity(&g, 12, 12, 12, 1, 3, 7).unwrap();
        assert_eq!(
            fixed.mean_p_iters, 7.0,
            "mean_p_iters = {}; a fixed-iteration solve runs exactly max_iter sweeps and must report them all",
            fixed.mean_p_iters
        );
    }
}
