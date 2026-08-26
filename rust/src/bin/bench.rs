// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! `ofgpu-bench` - how fast do the two models actually run, and how much of
//! the card do they use?
//!
//! Builds a uniform block of the requested size, imposes a shear flow whose
//! face flux is exactly divergence free by construction, and iterates each
//! model. Reports ms per outer iteration, cell-iterations per second, and the
//! resident device memory - the number that decides how large a mesh fits.
//!
//! ```text
//! ofgpu-bench [nx ny nz] [-iters N] [-fixedIters N] [-model kEpsilon|kOmega|both]
//! ```
//!
//! Carried across from this project's own earlier C++ benchmark driver.

use std::process::ExitCode;
use std::time::Instant;

use cudarc::driver::sys::CUdevice_attribute;

use ofgpu::blockgen;
use ofgpu::blockgen::{BlockSpec, GradedAxis};
use ofgpu::field::{BcKind, GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use ofgpu::field_ops::{correct_boundary_conditions_vector, FieldKernels};
use ofgpu::field_setup::max_div_phi;
use ofgpu::mesh::{HostMesh, PatchKind};
use ofgpu::models::{KEpsilon, KEpsilonCoeffs, KOmega, KOmegaCoeffs};
use ofgpu::turbulence::{FlowState, TurbulenceControls};
use ofgpu::wallfunctions::WallFunctionCoeffs;
use ofgpu::{Gpu, GpuMesh, Label, Result, Scalar, Vec3};

#[path = "common/mod.rs"]
mod common;

use common::{atoi, precision_name, resident_mib, sci};

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
        window: None,
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
        cyclic: None,
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

fn report_memory(gpu: &Gpu, tag: &str) -> Result<()> {
    let (used, total) = resident_mib(gpu)?;
    println!("       {tag}: {used} MiB resident of {total} MiB");
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

    report_memory(gpu, name)?;

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

    report_memory(&gpu, "mesh only")?;

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
        let mut model = KEpsilon::new(
            &gpu,
            &b.hm,
            &b.mesh,
            KEpsilonCoeffs::default(),
            ctrl,
            wc,
            &b.wf_faces,
            &no_roughness,
        )?;

        init_scalar(&gpu, model.k_mut(), &b.hm, "k", 0.01, true)?;
        init_scalar(&gpu, model.epsilon_mut(), &b.hm, "epsilon", 0.1, true)?;
        init_scalar(&gpu, model.nut_mut(), &b.hm, "nut", 0.0, true)?;

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
