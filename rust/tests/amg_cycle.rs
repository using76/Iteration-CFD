// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! AMG-04a: the V-cycle, GAMG as a preconditioner and as a solver, and
//! the LU coarsest level, the gate table of SPEC-LIT §141.5.1. The
//! device cycle against its host twin is class R, `1e-12` of `norm(x)`
//! in f64 and `1e-5` under `single` (D-F32-1), because level 0's
//! product is `ldu_ops::amul`; the LU hand values are class A; the
//! device LU solves are bitwise (class D); the iteration gates are f64
//! rows and wait for AMG-06a under `single`.
//!
//! An integration test, so the SAME assertions run in both builds (§118).
//! A machine without a card passes the device rows vacuously, the
//! convention of `tests/amg_coarse.rs`.
//!
//! Provenance: ORIGINAL - the fixtures, the hand-computed solutions and
//! the assertions are this tree's own construction, run against Saad
//! (2003) §13.4.3 Algorithm 13.3, the LAPACK `DGETRF` documentation
//! comment and the open article "Diagonally dominant matrix", via
//! SPEC-LIT §141.5. OpenFOAM's GAMG, AGMG, hypre and ML were not
//! opened. No GPL-licensed source was consulted.

use ofgpu::amg::{
    CoarseMethod, CoarseOperator, CoarseSetup, Gamg, GamgConfig, GpuCoarse, GpuLevelRows,
    HostHierarchy, HostLevel, HostRows, HierarchyConfig, LevelRows, SmootherKind,
    level_amul_host, smooth_level_host,
};
use ofgpu::blockgen::{BlockSpec, GradedAxis, build_mesh};
use ofgpu::distsolve::{DistReduce, DistSystem, DistWorkspace};
use ofgpu::decompose::{Decomposition, PartitionMethod};
use ofgpu::error::Result;
use ofgpu::halo::HaloExchange;
use ofgpu::io::case::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::ldu::{GpuLduMatrix, HostLduMatrix};
use ofgpu::mesh::{GpuMesh, HostMesh};
use ofgpu::solver::{SolverPerformance, SolverWorkspace, solve};
use ofgpu::{Gpu, Scalar};

/// The class-R tolerance: `1e-12` of the field's norm in f64, `1e-5`
/// under `single` (D-F32-1).
const TOL_R: f64 = if cfg!(feature = "single") { 1e-5 } else { 1e-12 };

/// The class-A tolerance: `1e-15` relative in f64, `1e-6` under `single`.
const TOL_A: f64 = if cfg!(feature = "single") { 1e-6 } else { 1e-15 };

/// Every device test needs a card; without one the test passes vacuously.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The build's `Scalar`, from the f64 the hand-computed solutions live in.
#[allow(clippy::useless_conversion)]
fn sc(x: f64) -> Scalar {
    x as Scalar
}

/// A box from three axes, every patch a wall. Cells are numbered
/// `i + nx*(j + ny*k)`.
fn box_mesh(nx: usize, ny: usize, nz: usize) -> HostMesh {
    let axis = |n: usize| GradedAxis {
        lo: 0.0,
        hi: n as Scalar,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    let b = BlockSpec {
        x: axis(nx),
        y: axis(ny),
        z: axis(nz),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall"; 6].map(String::from),
        cyclic: Vec::new(),
    };
    build_mesh(&b).expect("box mesh")
}

/// The test field of `tests/amg_coarse.rs`: `x[c] = 0.1*c + ((7c+3) mod
/// 11)/13`, in f64, cast.
fn test_field(n: usize) -> Vec<Scalar> {
    (0..n)
        .map(|c| sc(0.1 * c as f64 + ((7 * c + 3) % 11) as f64 / 13.0))
        .collect()
}

/// `POI(n)` of (141.5.5): the §2 laplacian negated so `sigma = +1`,
/// `upper = lower = -|S_f| delta_f`, the wall terms folded into the
/// diagonal, `boundary_coeffs` zero, `b = 1`.
fn poi(m: &HostMesh) -> HostLduMatrix {
    let mut a = HostLduMatrix::zeros(m);
    for f in 0..m.n_internal_faces {
        let g = m.mag_sf[f] * m.delta_coeffs[f];
        a.upper[f] = -g;
        a.lower[f] = -g;
        a.diag[m.owner[f] as usize] += g;
        a.diag[m.neighbour[f] as usize] += g;
    }
    for bf in 0..m.n_boundary_faces {
        let c = m.b_face_cells[bf] as usize;
        a.diag[c] += m.b_mag_sf[bf] * m.b_delta_coeffs[bf];
    }
    a.source = vec![sc(1.0); m.n_cells];
    a
}

/// `CD(n)` of (141.5.5): `POI(n)`'s box with the upwind convection of
/// `u = (Pe, 0, 0)`, `Pe = 10`: `F_f = Pe (S_f . x_hat)` from owner to
/// neighbour, `F_b = Pe (S_b . x_hat)` out of the wall,
/// `upper = -(D_f + max(-F_f, 0))`, `lower = -(D_f + max(F_f, 0))`,
/// `diag[o] += D_f + max(F_f, 0)`, `diag[n] += D_f + max(-F_f, 0)`,
/// `diag[c] += D_b + max(F_b, 0)`, `b = 1`.
fn cd(m: &HostMesh, pe: f64) -> HostLduMatrix {
    let mut a = HostLduMatrix::zeros(m);
    for f in 0..m.n_internal_faces {
        let d = m.mag_sf[f] as f64 * m.delta_coeffs[f] as f64;
        let ff = pe * (m.sf[f].x as f64);
        a.upper[f] = sc(-(d + f64::max(-ff, 0.0)));
        a.lower[f] = sc(-(d + f64::max(ff, 0.0)));
        a.diag[m.owner[f] as usize] += sc(d + f64::max(ff, 0.0));
        a.diag[m.neighbour[f] as usize] += sc(d + f64::max(-ff, 0.0));
    }
    for bf in 0..m.n_boundary_faces {
        let c = m.b_face_cells[bf] as usize;
        let d = m.b_mag_sf[bf] as f64 * m.b_delta_coeffs[bf] as f64;
        let fb = pe * (m.b_sf[bf].x as f64);
        a.diag[c] += sc(d + f64::max(fb, 0.0));
    }
    a.source = vec![sc(1.0); m.n_cells];
    a
}

/// The mesh and matrix on the device, as `tests/amg_coarse.rs` uploads
/// them.
fn upload_fine(gpu: &Gpu, m: &HostMesh, a: &HostLduMatrix) -> Result<(GpuMesh, GpuLduMatrix)> {
    let mesh = GpuMesh::upload(gpu, m)?;
    let mut mat = GpuLduMatrix::new(gpu, &mesh)?;
    gpu.write(&mut mat.diag, &a.diag)?;
    gpu.write(&mut mat.upper, &a.upper)?;
    gpu.write(&mut mat.lower, &a.lower)?;
    gpu.write(&mut mat.boundary_coeffs, &a.boundary_coeffs)?;
    gpu.write(&mut mat.source, &a.source)?;
    Ok((mesh, mat))
}

/// Relative L2 distance `||x - y|| / ||y||` in f64 from Scalar values.
fn rel_l2(x: &[Scalar], y: &[Scalar]) -> f64 {
    let (mut sq, mut sy) = (0.0f64, 0.0f64);
    for (a, b) in x.iter().zip(y.iter()) {
        let d = *a as f64 - *b as f64;
        sq += d * d;
        sy += (*b as f64) * (*b as f64);
    }
    sq.sqrt() / sy.sqrt()
}

/// The relative residual `||b - A x|| / ||b||` of the RAW system in f64,
/// through `HostRows::amul` on `x` as f64.
fn residual(rows: &HostRows, b: &[Scalar], x: &[Scalar]) -> f64 {
    let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let ax = rows.amul(&xf);
    let (mut sq, mut nb) = (0.0f64, 0.0f64);
    for (c, &bc) in b.iter().enumerate().take(rows.n_cells) {
        let r = bc as f64 - ax[c];
        sq += r * r;
        nb += (bc as f64) * (bc as f64);
    }
    sq.sqrt() / nb.sqrt()
}

/// Bitwise equality of two Scalar fields, entry by entry.
fn assert_bits(a: &[Scalar], b: &[Scalar], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths {} vs {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}: entry {i} differs: {:?} vs {:?}",
            x.to_bits(),
            y.to_bits()
        );
    }
}

/// The host V-cycle's per-level data: level 0 as a level, the coarsest
/// setup built once.
struct Vc<'a> {
    level0: &'a HostLevel,
    rows0: &'a LevelRows,
    h: &'a HostHierarchy,
    coarse: &'a CoarseSetup,
}

/// One host V(nu1, nu2) cycle on level `l`, in Scalar: pre-smooth nu1,
/// restrict, exact coarse solve (recursive), prolong-add, post-smooth
/// nu2. The host twin of `Gamg::vcycle` in the kernel's order.
fn host_vcycle(
    v: &Vc<'_>,
    l: usize,
    nu1: usize,
    nu2: usize,
    b: &[Scalar],
    x: &mut [Scalar],
) -> Result<()> {
    let n = if l == 0 {
        v.level0.n_cells
    } else {
        v.h.levels[l - 1].n_cells
    };
    if l == v.h.levels.len() {
        let xc = v.coarse.solve_host(b, None)?;
        x[..n].copy_from_slice(&xc);
        return Ok(());
    }
    let (level, rows) = if l == 0 {
        (v.level0, v.rows0)
    } else {
        (&v.h.levels[l - 1], &v.h.rows[l - 1])
    };
    let smoother = &v.h.smoothers[l];
    smooth_level_host(level, rows, smoother, x, b, nu1)?;
    let ax = level_amul_host(level, rows, x)?;
    let r: Vec<Scalar> = (0..n).map(|c| b[c] - ax[c]).collect();
    let rc = v.h.plans[l].restrict_host(&r)?;
    let mut xc = vec![0 as Scalar; rc.len()];
    host_vcycle(v, l + 1, nu1, nu2, &rc, &mut xc)?;
    v.h.plans[l].prolong_add_host(&xc, x)?;
    smooth_level_host(level, rows, smoother, x, b, nu2)?;
    Ok(())
}

/// `POI(16)` with `n_cells_coarsest = 16`, V(2,2) and V(1,3) with
/// l1-Jacobi and V(1,1) with Chebyshev: device against its host twin,
/// class R (SPEC-LIT 141.5.1); and `POI(4)` (64 cells, `L = 0`): the
/// cycle is the exact solve, residual `<= 1e-12`.
#[test]
fn the_device_v_cycle_matches_its_host_twin() {
    let Some(gpu) = gpu() else { return };
    let m = box_mesh(16, 16, 16);
    let a = poi(&m);
    let n = m.n_cells;
    let b = test_field(n);
    let coarse16 = HierarchyConfig {
        n_cells_coarsest: 16,
        ..HierarchyConfig::default()
    };
    let cfgs: [(&str, GamgConfig); 3] = [
        (
            "V(2,2) l1-Jacobi",
            GamgConfig {
                hierarchy: coarse16,
                ..GamgConfig::default()
            },
        ),
        (
            "V(1,3) l1-Jacobi",
            GamgConfig {
                hierarchy: coarse16,
                n_pre: 1,
                n_post: 3,
            },
        ),
        (
            "V(1,1) Chebyshev",
            GamgConfig {
                hierarchy: HierarchyConfig {
                    n_cells_coarsest: 16,
                    smoother: SmootherKind::Chebyshev {
                        degree: 2,
                        fraction: 0.3,
                    },
                    ..HierarchyConfig::default()
                },
                n_pre: 1,
                n_post: 1,
            },
        ),
    ];

    for (name, cfg) in cfgs {
        // The host twin, in Scalar, in the kernel's order.
        let h = HostHierarchy::build(&m, &a, &cfg.hierarchy).expect("hierarchy");
        let level0 = HostLevel {
            n_cells: n,
            owner: m.owner.clone(),
            neighbour: m.neighbour.clone(),
            diag: a.diag.clone(),
            upper: a.upper.clone(),
            lower: a.lower.clone(),
        };
        let rows0 = LevelRows::of_level(&level0).expect("rows0");
        let last = h.levels.last().expect("coarsest level");
        let coarse = CoarseSetup::general(
            &HostRows::of_level(last).expect("coarsest rows"),
            false,
        )
        .expect("coarse setup");
        let v = Vc {
            level0: &level0,
            rows0: &rows0,
            h: &h,
            coarse: &coarse,
        };
        let mut xh = vec![0 as Scalar; n];
        host_vcycle(&v, 0, cfg.n_pre, cfg.n_post, &b, &mut xh)
            .expect("host vcycle");

        // The device cycle, from x = 0.
        let (mesh, mat) = upload_fine(&gpu, &m, &a).expect("upload");
        let mut gamg = Gamg::new(&gpu, &m, &a, &cfg, false).expect("gamg");
        let mut x = gpu.zeros(n).expect("x");
        let bd = gpu.upload(&b).expect("b");
        gamg.vcycle(&gpu, &mesh, &mat, &mut x, &bd).expect("vcycle");
        let xd = gpu.download(&x).expect("download");
        let d = rel_l2(&xd, &xh);
        assert!(d <= TOL_R, "{name}: device vs host rel L2 {d:e} > {TOL_R:e}");

        drop(gamg);
        drop(bd);
        drop(x);
        drop(mat);
        drop(mesh);
    }

    // POI(4): 64 cells, already at n_cells_coarsest, so L = 0 and the
    // cycle is the exact solve on the fine matrix.
    let m = box_mesh(4, 4, 4);
    let a = poi(&m);
    let n = m.n_cells;
    let b = test_field(n);
    let (mesh, mat) = upload_fine(&gpu, &m, &a).expect("upload");
    let mut gamg = Gamg::new(&gpu, &m, &a, &GamgConfig::default(), false).expect("gamg");
    assert_eq!(gamg.host().levels.len(), 0, "POI(4) must have L = 0");
    let mut x = gpu.zeros(n).expect("x");
    let bd = gpu.upload(&b).expect("b");
    gamg.vcycle(&gpu, &mesh, &mat, &mut x, &bd).expect("vcycle");
    let xd = gpu.download(&x).expect("download");
    let level0 = HostLevel {
        n_cells: n,
        owner: m.owner.clone(),
        neighbour: m.neighbour.clone(),
        diag: a.diag.clone(),
        upper: a.upper.clone(),
        lower: a.lower.clone(),
    };
    let rows0 = HostRows::of_level(&level0).expect("rows0");
    let r = residual(&rows0, &b, &xd);
    assert!(r <= TOL_R, "POI(4): relative residual {r:e} > {TOL_R:e}");
}

/// The two-cell asymmetric level `[[d0, u], [l, d1]]`.
fn pair_asym(d0: f64, d1: f64, u: f64, l: f64) -> HostLevel {
    HostLevel {
        n_cells: 2,
        owner: vec![0],
        neighbour: vec![1],
        diag: vec![sc(d0), sc(d1)],
        upper: vec![sc(u)],
        lower: vec![sc(l)],
    }
}

/// `[[4, -1], [-2, 3]]`, `b = (1, 2)`: `l = [4, -1, -0.5, 2.5]` exactly,
/// `x = (0.5, 1)`; its negation: `sigma = -1`, the same factor,
/// `x = -(0.5, 1)`; device bitwise the host twin; `general` on `K4` and
/// `SYM` is `new` bit for bit (SPEC-LIT 141.5.1, classes A and D).
#[test]
fn the_lu_factor_and_solution_of_small_asymmetric_matrices() {
    // The factor and the solution, class A.
    let l1 = pair_asym(4.0, 3.0, -1.0, -2.0);
    let rows = HostRows::of_level(&l1).expect("rows");
    let s = CoarseSetup::general(&rows, false).expect("setup");
    assert_eq!(s.method, CoarseMethod::Lu);
    assert_eq!(s.order, 2);
    assert_eq!(s.sigma, 1.0);
    assert_eq!(s.l, vec![4.0, -1.0, -0.5, 2.5], "the dense LU");
    let x = s.solve_host(&[sc(1.0), sc(2.0)], None).expect("solve");
    let (x0, x1) = (x[0] as f64, x[1] as f64);
    assert!(
        (x0 - 0.5).abs() <= TOL_A * 0.5 && (x1 - 1.0).abs() <= TOL_A,
        "x = ({x0:e}, {x1:e}), want (0.5, 1)"
    );

    // The negated matrix: sigma = -1, the same factor, x = -(0.5, 1).
    let l2 = pair_asym(-4.0, -3.0, 1.0, 2.0);
    let rows2 = HostRows::of_level(&l2).expect("rows");
    let s2 = CoarseSetup::general(&rows2, false).expect("setup");
    assert_eq!(s2.sigma, -1.0);
    assert_eq!(
        s2.l.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        s.l.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "the negated level factors to the same LU"
    );
    let x2 = s2.solve_host(&[sc(1.0), sc(2.0)], None).expect("solve");
    let (x0, x1) = (x2[0] as f64, x2[1] as f64);
    assert!(
        (x0 + 0.5).abs() <= TOL_A * 0.5 && (x1 + 1.0).abs() <= TOL_A,
        "negated x = ({x0:e}, {x1:e}), want -(0.5, 1)"
    );

    // The device solve is bitwise the host twin.
    let Some(gpu) = gpu() else { return };
    let level = l1.upload(&gpu).expect("upload");
    let lrows = LevelRows::of_level(&l1).expect("level rows");
    let grows = GpuLevelRows::new(&gpu, &lrows).expect("device rows");
    let mut gc = GpuCoarse::new(&gpu, &s).expect("device coarse");
    let b = gpu.upload(&[sc(1.0), sc(2.0)]).expect("b");
    let mut x = gpu.zeros(2).expect("x");
    gc.solve(
        &gpu,
        CoarseOperator::Level { level: &level, rows: &grows },
        &b,
        &mut x,
    )
    .expect("device solve");
    let xd = gpu.download(&x).expect("download");
    let host = s.solve_host(&[sc(1.0), sc(2.0)], None).expect("host");
    assert_bits(&xd, &host, "LU device vs host twin");

    // `general` on a symmetric level is `new`, bit for bit: method and
    // every `l` entry. `K4` (the complete graph on 4 cells) and `SYM`
    // (the 6^3 box of tests/amg_coarse.rs).
    let k4 = HostLevel {
        n_cells: 4,
        owner: vec![0, 0, 0, 1, 1, 2],
        neighbour: vec![1, 2, 3, 2, 3, 3],
        diag: vec![sc(2.0); 4],
        upper: vec![sc(1.0); 6],
        lower: vec![sc(1.0); 6],
    };
    let axis = |n: usize| GradedAxis {
        lo: 0.0,
        hi: n as Scalar,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    let b6 = BlockSpec {
        x: axis(6),
        y: axis(6),
        z: axis(6),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall"; 6].map(String::from),
        cyclic: Vec::new(),
    };
    let m6 = build_mesh(&b6).expect("box mesh");
    let mut sym = HostLevel {
        n_cells: m6.n_cells,
        owner: m6.owner.clone(),
        neighbour: m6.neighbour.clone(),
        diag: vec![0 as Scalar; m6.n_cells],
        upper: vec![0 as Scalar; m6.n_internal_faces],
        lower: vec![0 as Scalar; m6.n_internal_faces],
    };
    for (c, d) in sym.diag.iter_mut().enumerate() {
        *d = sc(20.0 + c as f64 / 9.0);
    }
    for f in 0..m6.n_internal_faces {
        let (a, c) = (
            m6.owner[f].min(m6.neighbour[f]) as usize,
            m6.owner[f].max(m6.neighbour[f]) as usize,
        );
        let v = -(0.1 + ((11 * a + 3 * c) % 7) as f64 / 3.0);
        sym.upper[f] = sc(v);
        sym.lower[f] = sc(v);
    }
    for (name, l) in [("K4", k4), ("SYM", sym)] {
        let rows = HostRows::of_level(&l).expect("rows");
        let g = CoarseSetup::general(&rows, false).expect("general");
        let n = CoarseSetup::new(&rows, false).expect("new");
        assert_eq!(g.method, n.method, "{name}: the method must agree");
        assert_eq!(
            g.l.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            n.l.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{name}: every factor entry must agree bit for bit"
        );
    }
}

/// The coarsest level of `CD(8)` under `max_levels = 2`: exactly one
/// coarse level, which still carries x-couplings between its
/// aggregates, so the level is genuinely asymmetric - largest
/// `|A_ij - A_ji| >= 1` over the dense rows, asserted in BOTH builds -
/// and `general` takes `Lu`; the solve's residual is `<= TOL_R` and the
/// device solve is bitwise its host twin, twice (SPEC-LIT 141.5.1,
/// classes R and D).
#[test]
fn the_lu_coarse_solve_on_the_convection_diffusion_coarsest_level() {
    let m = box_mesh(8, 8, 8);
    let a = cd(&m, 10.0);
    let cfg = HierarchyConfig {
        max_levels: 2,
        ..HierarchyConfig::default()
    };
    let h = HostHierarchy::build(&m, &a, &cfg).expect("hierarchy");
    let last = h.levels.last().expect("coarsest level");
    assert_eq!(h.levels.len(), 1, "max_levels = 2 gives exactly one coarse level");
    assert!(
        last.n_cells <= 1024,
        "the coarsest level has {} cells > 1024",
        last.n_cells
    );
    let rows = HostRows::of_level(last).expect("rows");

    // The dense rows, entries summed per column exactly as `CoarseSetup`
    // does, and the largest entrywise asymmetry.
    let n = rows.n_cells;
    let mut dense = vec![0.0f64; n * n];
    for c in 0..n {
        dense[c * n + c] = rows.diag[c];
        for j in rows.offset[c]..rows.offset[c + 1] {
            dense[c * n + rows.col[j]] += rows.val[j];
        }
    }
    let mut max_asym = 0.0f64;
    for i in 1..n {
        for j in 0..i {
            max_asym = max_asym.max((dense[i * n + j] - dense[j * n + i]).abs());
        }
    }
    assert!(
        max_asym >= 1.0,
        "the CD(8) coarsest level ({n} cells) is not genuinely \
         asymmetric: largest |A_ij - A_ji| {max_asym:e} < 1"
    );
    eprintln!(
        "CD(8) coarsest level: {n} cells, largest |A_ij - A_ji| \
         {max_asym:.6e}"
    );
    let s = CoarseSetup::general(&rows, false).expect("setup");
    assert_eq!(s.method, CoarseMethod::Lu, "the CD coarsest level is asymmetric");

    let b = test_field(rows.n_cells);
    let x = s.solve_host(&b, None).expect("host solve");
    let r = residual(&rows, &b, &x);
    assert!(r <= TOL_R, "CD coarsest: relative residual {r:e} > {TOL_R:e}");

    let Some(gpu) = gpu() else { return };
    let level = last.upload(&gpu).expect("upload");
    let lrows = LevelRows::of_level(last).expect("level rows");
    let grows = GpuLevelRows::new(&gpu, &lrows).expect("device rows");
    let mut gc = GpuCoarse::new(&gpu, &s).expect("device coarse");
    let bd = gpu.upload(&b).expect("b");
    let mut xd = gpu.zeros(rows.n_cells).expect("x");
    for run in 1..=2 {
        gc.solve(
            &gpu,
            CoarseOperator::Level { level: &level, rows: &grows },
            &bd,
            &mut xd,
        )
        .expect("device solve");
        let got = gpu.download(&xd).expect("download");
        assert_bits(&got, &x, &format!("LU device vs host twin, run {run}"));
    }
}

/// `[[1, 2], [3, 6]]`: `pivot` at 1 (`Error::Refused`, naming
/// `-permissive`), and under `-permissive` `JacobiFallback`;
/// `[[1, -1], [-2, 2]]`: `singular` (SPEC-LIT 141.5.1, class D).
#[test]
fn an_asymmetric_level_that_cannot_be_factored_is_refused_by_name() {
    // The zero-determinant level with a non-constant null vector: the
    // rows are not near-singular (row 0 sums to 3), so the LU runs and
    // dies on U_11 = 6 - 3*2 = 0.
    let l = pair_asym(1.0, 6.0, 2.0, 3.0);
    let rows = HostRows::of_level(&l).expect("rows");
    let e = CoarseSetup::general(&rows, false)
        .expect_err("a level with a zero pivot must be refused")
        .to_string();
    assert!(e.contains("amg::coarse: "), "{e}");
    assert!(e.contains("pivot"), "{e}");
    assert!(e.contains("-permissive"), "{e}");

    let _guard = ofgpu::io::contract::permissive_test_guard();
    ofgpu::io::contract::set_permissive(true);
    let s = CoarseSetup::general(&rows, true).expect("permissive");
    assert_eq!(s.method, CoarseMethod::JacobiFallback);
    assert!(s.fallback.is_some(), "the substitution carries its smoother");
    ofgpu::io::contract::set_permissive(false);

    // The singular asymmetric level: every row sums to zero, so the
    // projection does not apply and the level is refused before any
    // factorisation, strictly and permissively alike.
    let l = pair_asym(1.0, 2.0, -1.0, -2.0);
    let rows = HostRows::of_level(&l).expect("rows");
    let e = CoarseSetup::general(&rows, false)
        .expect_err("a singular asymmetric level must be refused")
        .to_string();
    assert!(e.contains("amg::coarse: "), "{e}");
    assert!(e.contains("singular"), "{e}");
    let e = CoarseSetup::general(&rows, true)
        .expect_err("the singular refusal is strict")
        .to_string();
    assert!(e.contains("singular"), "{e}");
}

/// One PCG or PBiCGStab or GAMG solve of `A psi = b` from `psi = 0`,
/// its own workspace, dropped with its answer's device buffers.
fn one_solve(
    gpu: &Gpu,
    k: &ofgpu::solver::SolverKernels,
    mesh: &GpuMesh,
    mat: &GpuLduMatrix,
    gamg: Option<Gamg>,
    ctrl: &SolverControls,
) -> Result<(SolverPerformance, Vec<Scalar>)> {
    let n = mesh.n_cells;
    let mut w = SolverWorkspace::for_mesh(gpu, mesh)?;
    if let Some(g) = gamg {
        w.set_gamg(g)?;
    }
    let mut psi = gpu.zeros(n)?;
    let p = solve(gpu, k, &mut psi, mat, mesh, &mut w, ctrl)?;
    let x = gpu.download(&psi)?;
    Ok((p, x))
}

/// `POI(32)`, `POI(64)`, `POI(128)`, tol `1e-8`: DIC-PCG, GAMG-PCG and
/// standalone GAMG all converge; at 128^3 GAMG-PCG takes `<= 1/3` of
/// DIC-PCG's iterations; at 128^3 GAMG-PCG and Jacobi-PBiCGStab agree
/// to `1e-8` of `norm(x)` at tol `1e-8` (the plan's row), and the
/// 1e-12 row is an additional, stricter agreement check on the same
/// pair solved to `1e-12` (SPEC-LIT 141.5.1, classes R and P). The
/// iteration growth is printed as a table (f64 only; the gates wait
/// for AMG-06a under `single`).
#[test]
#[cfg_attr(feature = "single", ignore = "AMG-06a: GAMG under single")]
fn gamg_on_dirichlet_poisson_32_64_128() {
    let Some(gpu) = gpu() else { return };
    let k = ofgpu::solver::SolverKernels::new(&gpu).expect("solver kernels");

    for n in [32usize, 64, 128] {
        let m = box_mesh(n, n, n);
        let a = poi(&m);
        let (mesh, mat) = upload_fine(&gpu, &m, &a).expect("upload");
        let base = SolverControls {
            tolerance: 1e-8,
            rel_tol: 0.0,
            max_iter: 5000,
            check_interval: 1,
            ..SolverControls::default()
        };

        // DIC-PCG.
        let (p_dic, _) = one_solve(
            &gpu,
            &k,
            &mesh,
            &mat,
            None,
            &SolverControls {
                solver: LinearSolverKind::PCG,
                precon: Preconditioner::Dic,
                ..base
            },
        )
        .expect("DIC-PCG");
        assert!(p_dic.converged, "DIC-PCG must converge on POI({n})");

        // GAMG-PCG, default config.
        let gamg = Gamg::new(&gpu, &m, &a, &GamgConfig::default(), false)
            .expect("gamg");
        let sizes = gamg.sizes();
        let (p_gpcg, x_gpcg8) = one_solve(
            &gpu,
            &k,
            &mesh,
            &mat,
            Some(gamg),
            &SolverControls {
                solver: LinearSolverKind::PCG,
                precon: Preconditioner::Gamg,
                ..base
            },
        )
        .expect("GAMG-PCG");
        assert!(p_gpcg.converged, "GAMG-PCG must converge on POI({n})");

        // Standalone GAMG: one iteration is one cycle.
        let gamg = Gamg::new(&gpu, &m, &a, &GamgConfig::default(), false)
            .expect("gamg");
        let (p_gamg, _) = one_solve(
            &gpu,
            &k,
            &mesh,
            &mat,
            Some(gamg),
            &SolverControls {
                solver: LinearSolverKind::Gamg,
                ..base
            },
        )
        .expect("standalone GAMG");
        assert!(p_gamg.converged, "standalone GAMG must converge on POI({n})");

        // Jacobi-PBiCGStab, the agreement reference at 1e-8.
        let (_, x_jac8) = one_solve(
            &gpu,
            &k,
            &mesh,
            &mat,
            None,
            &SolverControls {
                solver: LinearSolverKind::PBiCGStab,
                precon: Preconditioner::Diagonal,
                ..base
            },
        )
        .expect("Jacobi-PBiCGStab");
        let d8 = rel_l2(&x_gpcg8, &x_jac8);

        eprintln!(
            "POI({n}): sizes {} | DIC-PCG {} it | GAMG-PCG {} it | GAMG {} \
             it | GAMG-PCG vs Jacobi-PBiCGStab @1e-8 rel L2 {d8:.3e}",
            sizes
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("->"),
            p_dic.n_iterations,
            p_gpcg.n_iterations,
            p_gamg.n_iterations,
        );

        if n == 128 {
            assert!(
                3 * p_gpcg.n_iterations <= p_dic.n_iterations,
                "at 128^3 GAMG-PCG ({} it) must take at most a third of \
                 DIC-PCG's {} iterations",
                p_gpcg.n_iterations,
                p_dic.n_iterations
            );

            // The plan's agreement row at tol 1e-8 (docs/17's AMG-04a
            // row): GAMG-PCG's answer within 1e-8 rel of
            // Jacobi-PBiCGStab's.
            assert!(
                d8 <= 1e-8,
                "at 128^3 the 1e-8 answers differ by {d8:.3e} > 1e-8"
            );

            // The 1e-12 row is an additional, stricter agreement check:
            // both solved to 1e-12.
            let tight = SolverControls {
                tolerance: 1e-12,
                max_iter: 20000,
                ..base
            };
            let gamg = Gamg::new(&gpu, &m, &a, &GamgConfig::default(), false)
                .expect("gamg");
            let (_, x_gpcg12) = one_solve(
                &gpu,
                &k,
                &mesh,
                &mat,
                Some(gamg),
                &SolverControls {
                    solver: LinearSolverKind::PCG,
                    precon: Preconditioner::Gamg,
                    ..tight
                },
            )
            .expect("GAMG-PCG @1e-12");
            let (_, x_jac12) = one_solve(
                &gpu,
                &k,
                &mesh,
                &mat,
                None,
                &SolverControls {
                    solver: LinearSolverKind::PBiCGStab,
                    precon: Preconditioner::Diagonal,
                    ..tight
                },
            )
            .expect("Jacobi-PBiCGStab @1e-12");
            let d12 = rel_l2(&x_gpcg12, &x_jac12);
            eprintln!(
                "POI(128): GAMG-PCG vs Jacobi-PBiCGStab @1e-12 rel L2 \
                 {d12:.3e}"
            );
            assert!(
                d12 <= 1e-8,
                "the two 1e-12 answers differ by {d12:.3e} > 1e-8"
            );
        }

        drop(mat);
        drop(mesh);
    }
}

/// `CD(64)`, tol `1e-8`: DILU-PBiCGStab and GAMG-PBiCGStab both
/// converge, GAMG's iterations `<= 1/2` of DILU's (SPEC-LIT 141.5.1,
/// class P; f64 only, AMG-06a under `single`).
#[test]
#[cfg_attr(feature = "single", ignore = "AMG-06a: GAMG under single")]
fn gamg_pbicgstab_on_convection_diffusion() {
    let Some(gpu) = gpu() else { return };
    let k = ofgpu::solver::SolverKernels::new(&gpu).expect("solver kernels");
    let m = box_mesh(64, 64, 64);
    let a = cd(&m, 10.0);
    let (mesh, mat) = upload_fine(&gpu, &m, &a).expect("upload");
    let base = SolverControls {
        solver: LinearSolverKind::PBiCGStab,
        tolerance: 1e-8,
        rel_tol: 0.0,
        max_iter: 5000,
        check_interval: 1,
        ..SolverControls::default()
    };

    let (p_dilu, _) = one_solve(
        &gpu,
        &k,
        &mesh,
        &mat,
        None,
        &SolverControls {
            precon: Preconditioner::Dilu,
            ..base
        },
    )
    .expect("DILU-PBiCGStab");
    assert!(p_dilu.converged, "DILU-PBiCGStab must converge on CD(64)");

    let gamg = Gamg::new(&gpu, &m, &a, &GamgConfig::default(), false).expect("gamg");
    let (p_gamg, _) = one_solve(
        &gpu,
        &k,
        &mesh,
        &mat,
        Some(gamg),
        &SolverControls {
            precon: Preconditioner::Gamg,
            ..base
        },
    )
    .expect("GAMG-PBiCGStab");
    assert!(p_gamg.converged, "GAMG-PBiCGStab must converge on CD(64)");

    eprintln!(
        "CD(64): DILU-PBiCGStab {} it | GAMG-PBiCGStab {} it",
        p_dilu.n_iterations, p_gamg.n_iterations
    );
    assert!(
        2 * p_gamg.n_iterations <= p_dilu.n_iterations,
        "GAMG-PBiCGStab ({} it) must take at most half of DILU's {} \
         iterations",
        p_gamg.n_iterations,
        p_dilu.n_iterations
    );
}

/// Every refusal of the 141.5.1 row, by name (SPEC-LIT 141.5.1, class
/// D): `solver GAMG` and `preconditioner GAMG` on a workspace without a
/// hierarchy, PCG with V(1,2), `set_gamg` with another cell count,
/// `nu1 = nu2 = 0`, and the distributed path's `AMG-07b`.
#[test]
fn a_gamg_request_that_cannot_be_served_is_refused_by_name() {
    let _guard = ofgpu::io::contract::permissive_test_guard();
    ofgpu::io::contract::set_permissive(false);
    let Some(gpu) = gpu() else { return };
    let k = ofgpu::solver::SolverKernels::new(&gpu).expect("solver kernels");

    // nu1 = nu2 = 0: refused before anything is built.
    let m8 = box_mesh(8, 8, 8);
    let a8 = poi(&m8);
    let Err(e) = Gamg::new(
        &gpu,
        &m8,
        &a8,
        &GamgConfig {
            n_pre: 0,
            n_post: 0,
            ..GamgConfig::default()
        },
        false,
    ) else {
        panic!("a cycle without sweeps must be refused");
    };
    let e = e.to_string();
    assert!(e.contains("sweeps"), "{e}");

    // `solver GAMG` on a workspace without a hierarchy.
    let (mesh8, mat8) = upload_fine(&gpu, &m8, &a8).expect("upload");
    let mut w = SolverWorkspace::for_mesh(&gpu, &mesh8).expect("ws");
    let mut psi = gpu.zeros(m8.n_cells).expect("psi");
    let e = solve(
        &gpu,
        &k,
        &mut psi,
        &mat8,
        &mesh8,
        &mut w,
        &SolverControls {
            solver: LinearSolverKind::Gamg,
            ..SolverControls::default()
        },
    )
    .expect_err("a GAMG solve without a hierarchy must be refused")
    .to_string();
    assert!(e.contains("GAMG"), "{e}");
    assert!(e.contains("set_gamg"), "{e}");

    // `preconditioner GAMG` on a workspace without a hierarchy.
    let e = ofgpu::solver::effective_preconditioner(Preconditioner::Gamg, &w)
        .expect_err("a GAMG preconditioner without a hierarchy must be refused")
        .to_string();
    assert!(e.contains("GAMG"), "{e}");
    assert!(e.contains("set_gamg"), "{e}");

    // PCG with V(1,2): the cycle is not symmetric.
    let gamg = Gamg::new(
        &gpu,
        &m8,
        &a8,
        &GamgConfig {
            n_pre: 1,
            n_post: 2,
            ..GamgConfig::default()
        },
        false,
    )
    .expect("asymmetric cycle builds");
    w.set_gamg(gamg).expect("attach");
    let e = solve(
        &gpu,
        &k,
        &mut psi,
        &mat8,
        &mesh8,
        &mut w,
        &SolverControls {
            solver: LinearSolverKind::PCG,
            precon: Preconditioner::Gamg,
            ..SolverControls::default()
        },
    )
    .expect_err("PCG on an asymmetric cycle must be refused")
    .to_string();
    assert!(e.contains("nPreSweeps"), "{e}");
    assert!(e.contains("nPostSweeps"), "{e}");
    assert!(e.contains("symmetric"), "{e}");

    // `set_gamg` with another cell count.
    let mut w2 = SolverWorkspace::new(&gpu, 16).expect("small ws");
    let gamg = Gamg::new(&gpu, &m8, &a8, &GamgConfig::default(), false)
        .expect("gamg");
    let e = w2
        .set_gamg(gamg)
        .expect_err("a hierarchy of 512 cells must not attach to 16")
        .to_string();
    assert!(e.contains("cells"), "{e}");

    // The distributed path refuses GAMG by name (AMG-07b).
    let m = box_mesh(4, 4, 3);
    let a = poi(&m);
    let dec = Decomposition::build(&m, 2, &PartitionMethod::Hilbert).expect("cut");
    let mut ex = HaloExchange::new(&gpu, &dec).expect("halo");
    let sys = DistSystem::split(&gpu, &mut ex, &m, &dec, &a).expect("split");
    let dw = DistWorkspace::new(&gpu, &sys, DistReduce::Exact).expect("dws");
    let e = dw
        .effective_preconditioner(Preconditioner::Gamg)
        .expect_err("the distributed path refuses GAMG until AMG-07b")
        .to_string();
    assert!(e.contains("AMG-07b"), "{e}");
    assert!(!dw.partition_invariant(Preconditioner::Gamg));
}

