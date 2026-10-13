// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! AMG-03b: the coarsest-level solve, the gate table of SPEC-LIT
//! §141.4.1. The exact factors and solutions are class A, `1e-15`
//! relative in f64 and `1e-6` under `single`; the residuals are class R,
//! `1e-12` of `norm(b)` in f64 and `1e-5` under `single` (D-F32-1), the
//! zero-mean rows `1e-14` / `1e-5` of `max|x|`; the device rows are
//! bitwise (class D); the refusals name their reasons.
//!
//! An integration test, so the SAME assertions run in both builds (§118).
//! A machine without a card passes the device rows vacuously, the
//! convention of `tests/radiation_dom.rs`.
//!
//! Provenance: ORIGINAL - the fixtures, the hand-computed solutions and
//! the assertions are this tree's own construction, run against the
//! LAPACK 3.9.0 `DPOTRF` documentation and the entrywise recurrence of
//! the open article "Cholesky decomposition", via SPEC-LIT §141.4. No
//! GPL-licensed source was consulted.

use ofgpu::amg::{
    CoarseMethod, CoarseOperator, CoarseSetup, FALLBACK_SWEEPS, GpuCoarse, GpuLevelRows,
    HostHierarchy, HostLevel, HostRows, HierarchyConfig, LevelRows, MAX_DENSE_CELLS,
    level_amul_host, smooth_level_host,
};
use ofgpu::blockgen::{BlockSpec, GradedAxis, build_mesh};
use ofgpu::error::Result;
use ofgpu::ldu::{GpuLduMatrix, HostLduMatrix};
use ofgpu::ldu_ops::LduKernels;
use ofgpu::mesh::{GpuMesh, HostMesh};
use ofgpu::{Gpu, Scalar};

/// The class-R tolerance: `1e-12` of the field's norm in f64, `1e-5`
/// under `single` (D-F32-1).
const TOL_R: f64 = if cfg!(feature = "single") { 1e-5 } else { 1e-12 };

/// The class-A tolerance: `1e-15` relative in f64, `1e-6` under `single`.
const TOL_A: f64 = if cfg!(feature = "single") { 1e-6 } else { 1e-15 };

/// The zero-mean tolerance: `1e-14` of `max|x|` in f64, `1e-5` under
/// `single`.
const TOL_MEAN: f64 = if cfg!(feature = "single") { 1e-5 } else { 1e-14 };

/// Every device test needs a card; without one the test passes vacuously.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The build's `Scalar`, from the f64 the hand-computed solutions live in.
#[allow(clippy::useless_conversion)]
fn sc(x: f64) -> Scalar {
    x as Scalar
}

/// A box from three axes, every patch a wall, the cyclic axes of `cyclic`
/// set. Cells are numbered `i + nx*(j + ny*k)`.
fn mesh_from(x: GradedAxis, y: GradedAxis, z: GradedAxis, cyclic: &[usize]) -> HostMesh {
    let mut b = BlockSpec {
        x,
        y,
        z,
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall"; 6].map(String::from),
        cyclic: Vec::new(),
    };
    for &a in cyclic {
        b.set_cyclic_axis(a).expect("cyclic axis");
    }
    build_mesh(&b).expect("box mesh")
}

/// A uniform box.
fn box_mesh(nx: usize, ny: usize, nz: usize, cyclic: &[usize]) -> HostMesh {
    let axis = |n: usize| GradedAxis {
        lo: 0.0,
        hi: n as Scalar,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    mesh_from(axis(nx), axis(ny), axis(nz), cyclic)
}

/// The graded 6^3 box: axes `lo 0, hi 1, expansion 3.0`.
fn graded_mesh() -> HostMesh {
    let g = GradedAxis {
        lo: 0.0,
        hi: 1.0,
        n: 6,
        expansion: 3.0,
        two_sided: false,
    };
    mesh_from(g.clone(), g.clone(), g, &[])
}

/// The §2 laplacian on `m`: `upper = lower = |S_f| delta_f`, the diagonal
/// minus the same sum at both cells.
fn poisson(m: &HostMesh) -> HostLduMatrix {
    let mut a = HostLduMatrix::zeros(m);
    for f in 0..m.n_internal_faces {
        let g = m.mag_sf[f] * m.delta_coeffs[f];
        a.upper[f] = g;
        a.lower[f] = g;
        a.diag[m.owner[f] as usize] -= g;
        a.diag[m.neighbour[f] as usize] -= g;
    }
    for bf in 0..m.n_boundary_faces {
        a.internal_coeffs[bf] = -(m.b_mag_sf[bf] * m.b_delta_coeffs[bf]);
    }
    a
}

/// `poisson` as a solve receives it: `internal_coeffs` folded into `diag`.
fn poisson_folded(m: &HostMesh) -> HostLduMatrix {
    let mut a = poisson(m);
    for bf in 0..m.n_boundary_faces {
        let c = m.b_face_cells[bf] as usize;
        a.diag[c] += a.internal_coeffs[bf];
    }
    a
}

/// `NEU(m)` of the gate table: the pure-Neumann laplacian - `poisson`
/// WITHOUT the internal_coeffs loop, every wall zero-gradient.
fn neu(m: &HostMesh) -> HostLduMatrix {
    let mut a = HostLduMatrix::zeros(m);
    for f in 0..m.n_internal_faces {
        let g = m.mag_sf[f] * m.delta_coeffs[f];
        a.upper[f] = g;
        a.lower[f] = g;
        a.diag[m.owner[f] as usize] -= g;
        a.diag[m.neighbour[f] as usize] -= g;
    }
    a
}

/// The symmetric 6^3 level of the gate table: `a_ab = a_ba =
/// -(1/10 + ((11a + 3b) mod 7)/3)` on every edge, `a_cc = 20 + c/9`.
fn sym6() -> HostLevel {
    let m = box_mesh(6, 6, 6, &[]);
    let mut l = HostLevel {
        n_cells: m.n_cells,
        owner: m.owner.clone(),
        neighbour: m.neighbour.clone(),
        diag: vec![0 as Scalar; m.n_cells],
        upper: vec![0 as Scalar; m.n_internal_faces],
        lower: vec![0 as Scalar; m.n_internal_faces],
    };
    for (c, d) in l.diag.iter_mut().enumerate() {
        *d = sc(20.0 + c as f64 / 9.0);
    }
    for f in 0..m.n_internal_faces {
        let (a, b) = (
            m.owner[f].min(m.neighbour[f]) as usize,
            m.owner[f].max(m.neighbour[f]) as usize,
        );
        let v = -(0.1 + ((11 * a + 3 * b) % 7) as f64 / 3.0);
        l.upper[f] = sc(v);
        l.lower[f] = sc(v);
    }
    l
}

/// `K4`, the complete graph on 4 cells: faces `(0,1), (0,2), (0,3),
/// (1,2), (1,3), (2,3)`, `a_cc = 2`, every off-diagonal `+1`.
fn k4() -> HostLevel {
    HostLevel {
        n_cells: 4,
        owner: vec![0, 0, 0, 1, 1, 2],
        neighbour: vec![1, 2, 3, 2, 3, 3],
        diag: vec![sc(2.0); 4],
        upper: vec![sc(1.0); 6],
        lower: vec![sc(1.0); 6],
    }
}

/// The two-cell level `[[d0, o], [o, d1]]`.
fn pair(d0: f64, d1: f64, o: f64) -> HostLevel {
    HostLevel {
        n_cells: 2,
        owner: vec![0],
        neighbour: vec![1],
        diag: vec![sc(d0), sc(d1)],
        upper: vec![sc(o)],
        lower: vec![sc(o)],
    }
}

/// The asymmetric two-cell level: `A(0,1) = -1` but `A(1,0) = -2`.
fn asym2() -> HostLevel {
    HostLevel {
        n_cells: 2,
        owner: vec![0],
        neighbour: vec![1],
        diag: vec![sc(4.0); 2],
        upper: vec![sc(-1.0)],
        lower: vec![sc(-2.0)],
    }
}

/// The three-cell path `[[1,-1,0],[-1,2,-1],[0,-1,1]]`.
fn path3() -> HostLevel {
    HostLevel {
        n_cells: 3,
        owner: vec![0, 1],
        neighbour: vec![1, 2],
        diag: vec![sc(1.0), sc(2.0), sc(1.0)],
        upper: vec![sc(-1.0); 2],
        lower: vec![sc(-1.0); 2],
    }
}

/// Two disconnected two-cell Neumann blocks: faces `(0,1)` and `(2,3)`.
fn disc4() -> HostLevel {
    HostLevel {
        n_cells: 4,
        owner: vec![0, 2],
        neighbour: vec![1, 3],
        diag: vec![sc(1.0); 4],
        upper: vec![sc(-1.0); 2],
        lower: vec![sc(-1.0); 2],
    }
}

/// The test field: `x[c] = 0.1*c + ((7c+3) mod 11)/13`, in f64, cast.
fn test_field(n: usize) -> Vec<Scalar> {
    (0..n)
        .map(|c| sc(0.1 * c as f64 + ((7 * c + 3) % 11) as f64 / 13.0))
        .collect()
}

/// `mean(x)` in f64 from the Scalar values.
fn mean_of(x: &[Scalar]) -> f64 {
    x.iter().map(|&v| v as f64).sum::<f64>() / x.len().max(1) as f64
}

/// `max|x|` in f64 from the Scalar values.
fn max_abs(x: &[Scalar]) -> f64 {
    x.iter().map(|&v| (v as f64).abs()).fold(0.0, f64::max)
}

/// Class A: `got` within [`TOL_A`] of the hand-computed `want`.
fn rel_close(got: f64, want: f64, what: &str) {
    assert!(
        (got - want).abs() <= TOL_A * want.abs(),
        "{what}: got {got:e}, want {want:e}"
    );
}

/// The relative residual `||b - A x|| / ||b||` in f64 from the Scalar
/// values (`HostRows::amul` on x as f64). The solve of (141.4.6) answers
/// the RAW system: the factor is of `sigma A`, but the substitutions run
/// on `sigma b`, so `A x = b`.
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

/// The pinned system's relative residual `||b' - A x|| / ||b'||`, with
/// `b' = b - mean(b)` (141.4.3); `A x = b'` as above.
fn pinned_residual(rows: &HostRows, s: &CoarseSetup, b: &[Scalar], x: &[Scalar]) -> f64 {
    let n = s.n_cells;
    let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let ax = rows.amul(&xf);
    let mean = b[..n].iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let (mut sq, mut nb) = (0.0f64, 0.0f64);
    for c in 0..n {
        let bp = b[c] as f64 - mean;
        let r = bp - ax[c];
        sq += r * r;
        nb += bp * bp;
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
            "{what}: entry {i} differs: {:?} vs {:?} ({} vs {})",
            x.to_bits(),
            y.to_bits(),
            x,
            y
        );
    }
}

/// The mesh and matrix on the device, as amg_hierarchy.rs uploads them.
fn upload_fine(gpu: &Gpu, m: &HostMesh, a: &HostLduMatrix) -> Result<(GpuMesh, GpuLduMatrix)> {
    let mesh = GpuMesh::upload(gpu, m)?;
    let mut mat = GpuLduMatrix::new(gpu, &mesh)?;
    gpu.write(&mut mat.diag, &a.diag)?;
    gpu.write(&mut mat.upper, &a.upper)?;
    gpu.write(&mut mat.lower, &a.lower)?;
    gpu.write(&mut mat.boundary_coeffs, &a.boundary_coeffs)?;
    Ok((mesh, mat))
}

/// `[[4, -1], [-1, 3]]`, `K4` and the negated pair: the factors and the
/// solutions of the gate table's first row, class A (SPEC-LIT 141.4.1).
#[test]
fn the_cholesky_factor_and_solution_of_small_matrices() {
    let l11 = 2.75f64.sqrt();

    // [[4, -1], [-1, 3]], b = (1, 2): L = (2; -0.5, sqrt(2.75)),
    // x = (5/11, 9/11).
    let l = pair(4.0, 3.0, -1.0);
    let rows = HostRows::of_level(&l).expect("rows");
    let s = CoarseSetup::new(&rows, false).expect("setup");
    assert_eq!(s.method, CoarseMethod::Cholesky);
    assert_eq!(s.order, 2);
    assert_eq!(s.sigma, 1.0);
    for (k, (got, want)) in s.l.iter().zip([2.0, -0.5, l11]).enumerate() {
        rel_close(*got, want, &format!("L[{k}]"));
    }
    let x = s.solve_host(&[sc(1.0), sc(2.0)], None).expect("solve");
    rel_close(x[0] as f64, 5.0 / 11.0, "x[0]");
    rel_close(x[1] as f64, 9.0 / 11.0, "x[1]");

    // K4, b = 1: x = 0.2 in every cell.
    let rows = HostRows::of_level(&k4()).expect("rows");
    let s = CoarseSetup::new(&rows, false).expect("setup");
    assert_eq!(s.method, CoarseMethod::Cholesky);
    let x = s.solve_host(&[sc(1.0); 4], None).expect("solve");
    for (c, xc) in x.iter().enumerate() {
        rel_close(*xc as f64, 0.2, &format!("K4 x[{c}]"));
    }

    // The negated [[4, -1], [-1, 3]]: sigma = -1, the same L,
    // x = -(5/11, 9/11).
    let rows = HostRows::of_level(&pair(-4.0, -3.0, 1.0)).expect("rows");
    let s = CoarseSetup::new(&rows, false).expect("setup");
    assert_eq!(s.sigma, -1.0);
    for (k, (got, want)) in s.l.iter().zip([2.0, -0.5, l11]).enumerate() {
        rel_close(*got, want, &format!("negated L[{k}]"));
    }
    let x = s.solve_host(&[sc(1.0), sc(2.0)], None).expect("solve");
    rel_close(x[0] as f64, -5.0 / 11.0, "negated x[0]");
    rel_close(x[1] as f64, -9.0 / 11.0, "negated x[1]");
}

/// `SYM`, `-SYM` and the coarsest level of the 32^3 Dirichlet laplacian:
/// relative residual at or below `1e-12` (f64) / `1e-5` (single), class R
/// (SPEC-LIT 141.4.1).
#[test]
fn the_coarse_solve_residual_is_below_tol_on_spd() {
    // SYM.
    let l = sym6();
    let rows = HostRows::of_level(&l).expect("rows");
    let s = CoarseSetup::new(&rows, false).expect("setup");
    assert_eq!(s.method, CoarseMethod::Cholesky);
    let b = test_field(l.n_cells);
    let x = s.solve_host(&b, None).expect("solve");
    let r = residual(&rows, &b, &x);
    assert!(r <= TOL_R, "SYM: relative residual {r:e} > {TOL_R:e}");

    // -SYM: every entry negated, sigma = -1.
    let mut nl = l.clone();
    for d in nl.diag.iter_mut() {
        *d = -*d;
    }
    for v in nl.upper.iter_mut().chain(nl.lower.iter_mut()) {
        *v = -*v;
    }
    let rows = HostRows::of_level(&nl).expect("rows");
    let s = CoarseSetup::new(&rows, false).expect("setup");
    assert_eq!(s.sigma, -1.0);
    let x = s.solve_host(&b, None).expect("solve");
    let r = residual(&rows, &b, &x);
    assert!(r <= TOL_R, "-SYM: relative residual {r:e} > {TOL_R:e}");

    // The coarsest level of the 32^3 laplacian with Dirichlet walls.
    let m = box_mesh(32, 32, 32, &[]);
    let a = poisson_folded(&m);
    let h = HostHierarchy::build(&m, &a, &HierarchyConfig::default()).expect("hierarchy");
    let last = h.levels.last().expect("coarsest level");
    let rows = HostRows::of_level(last).expect("rows");
    let s = CoarseSetup::new(&rows, false).expect("setup");
    let b = test_field(rows.n_cells);
    let x = s.solve_host(&b, None).expect("solve");
    let r = residual(&rows, &b, &x);
    assert!(
        r <= TOL_R,
        "32^3 coarsest: relative residual {r:e} > {TOL_R:e}"
    );
}

/// The pinned level of the gate table's third row: the hand-computed
/// solutions class A, the NEU rows pinned with a residual at or below
/// `TOL_R` of `b'` and zero mean (SPEC-LIT 141.4.1).
#[test]
fn a_pure_neumann_level_is_projected_and_pinned() {
    // [[1, -1], [-1, 1]], b = (1, 0): x = (0.25, -0.25).
    let rows = HostRows::of_level(&pair(1.0, 1.0, -1.0)).expect("rows");
    let s = CoarseSetup::new(&rows, false).expect("setup");
    assert_eq!(s.method, CoarseMethod::PinnedCholesky);
    assert_eq!(s.order, 1);
    let x = s.solve_host(&[sc(1.0), sc(0.0)], None).expect("solve");
    rel_close(x[0] as f64, 0.25, "2-cell x[0]");
    rel_close(x[1] as f64, -0.25, "2-cell x[1]");

    // The 3-cell path, b = (1, 0, 0): x = (5/9, -1/9, -4/9).
    let rows = HostRows::of_level(&path3()).expect("rows");
    let s = CoarseSetup::new(&rows, false).expect("setup");
    assert_eq!(s.method, CoarseMethod::PinnedCholesky);
    assert_eq!(s.order, 2);
    let x = s
        .solve_host(&[sc(1.0), sc(0.0), sc(0.0)], None)
        .expect("solve");
    rel_close(x[0] as f64, 5.0 / 9.0, "path x[0]");
    rel_close(x[1] as f64, -1.0 / 9.0, "path x[1]");
    rel_close(x[2] as f64, -4.0 / 9.0, "path x[2]");

    // NEU on the uniform 8^3 and the graded 6^3: pinned, the projected
    // residual below tol, zero mean.
    for (name, m) in [
        ("uniform 8^3", box_mesh(8, 8, 8, &[])),
        ("graded 6^3", graded_mesh()),
    ] {
        let a = neu(&m);
        let rows = HostRows::of_matrix(&m, &a).expect("rows");
        let s = CoarseSetup::new(&rows, false).expect("setup");
        assert_eq!(s.method, CoarseMethod::PinnedCholesky, "{name}");
        let b = test_field(rows.n_cells);
        let x = s.solve_host(&b, None).expect("solve");
        let r = pinned_residual(&rows, &s, &b, &x);
        assert!(r <= TOL_R, "{name}: relative residual {r:e} > {TOL_R:e}");
        let mx = mean_of(&x).abs();
        assert!(
            mx <= TOL_MEAN * max_abs(&x),
            "{name}: |mean(x)| {mx:e} > {TOL_MEAN:e} max|x|"
        );
    }

    // The coarsest level of NEU on the uniform 16^3.
    let m = box_mesh(16, 16, 16, &[]);
    let a = neu(&m);
    let h = HostHierarchy::build(&m, &a, &HierarchyConfig::default()).expect("hierarchy");
    let last = h.levels.last().expect("coarsest level");
    let rows = HostRows::of_level(last).expect("rows");
    let s = CoarseSetup::new(&rows, false).expect("setup");
    assert_eq!(s.method, CoarseMethod::PinnedCholesky);
    let b = test_field(rows.n_cells);
    let x = s.solve_host(&b, None).expect("solve");
    let r = pinned_residual(&rows, &s, &b, &x);
    assert!(
        r <= TOL_R,
        "16^3 coarsest: relative residual {r:e} > {TOL_R:e}"
    );
    let mx = mean_of(&x).abs();
    assert!(
        mx <= TOL_MEAN * max_abs(&x),
        "16^3 coarsest: |mean(x)| {mx:e} > {TOL_MEAN:e} max|x|"
    );
}

/// The host V-cycle's per-level data: level 0 as a level, the coarsest
/// setup built once.
struct Vc<'a> {
    level0: &'a HostLevel,
    rows0: &'a LevelRows,
    h: &'a HostHierarchy,
    coarse: &'a CoarseSetup,
}

/// One host V(1,1) cycle on level `l`, in Scalar: pre-smooth 1,
/// restrict, exact coarse solve (recursive), prolong-add, post-smooth 1.
/// The pinned coarse solve's correction is asserted zero-mean (141.4.4)
/// every time it runs.
fn vcycle(v: &Vc<'_>, l: usize, b: &[Scalar], x: &mut [Scalar]) -> Result<()> {
    let n = if l == 0 {
        v.level0.n_cells
    } else {
        v.h.levels[l - 1].n_cells
    };
    if l == v.h.levels.len() {
        let xc = v.coarse.solve_host(b, None)?;
        let mean = mean_of(&xc).abs();
        assert!(
            mean <= TOL_MEAN * max_abs(&xc),
            "the coarsest solve's |mean| {mean:e} exceeds {TOL_MEAN:e} \
             max|x| {}",
            max_abs(&xc)
        );
        x[..n].copy_from_slice(&xc);
        return Ok(());
    }
    let (level, rows) = if l == 0 {
        (v.level0, v.rows0)
    } else {
        (&v.h.levels[l - 1], &v.h.rows[l - 1])
    };
    let smoother = &v.h.smoothers[l];
    smooth_level_host(level, rows, smoother, x, b, 1)?;
    let ax = level_amul_host(level, rows, x)?;
    let r: Vec<Scalar> = (0..n).map(|c| b[c] - ax[c]).collect();
    let rc = v.h.plans[l].restrict_host(&r)?;
    let mut xc = vec![0 as Scalar; rc.len()];
    vcycle(v, l + 1, &rc, &mut xc)?;
    v.h.plans[l].prolong_add_host(&xc, x)?;
    smooth_level_host(level, rows, smoother, x, b, 1)?;
    Ok(())
}

/// NEU on the uniform 8^3, b of zero mean, 100 host V(1,1) cycles with
/// l1-Jacobi: the energy never rises beyond `TOL_R |J*|`, the iterates
/// stay within `10 ||x*||_inf`, `||r_100|| <= ||r_0||`, every coarse
/// correction zero-mean (SPEC-LIT 141.4.1, classes P and R).
#[test]
fn the_v_cycle_on_pure_neumann_stays_bounded_over_100_cycles() {
    let m = box_mesh(8, 8, 8, &[]);
    let a = neu(&m);
    let h = HostHierarchy::build(&m, &a, &HierarchyConfig::default()).expect("hierarchy");
    let level0 = HostLevel {
        n_cells: m.n_cells,
        owner: m.owner.clone(),
        neighbour: m.neighbour.clone(),
        diag: a.diag.clone(),
        upper: a.upper.clone(),
        lower: a.lower.clone(),
    };
    let rows0 = LevelRows::of_level(&level0).expect("rows0");
    let hrows0 = HostRows::of_level(&level0).expect("host rows0");
    let coarse0 = CoarseSetup::new(&hrows0, false).expect("level-0 setup");
    let last = h.levels.last().expect("coarsest level");
    let hrows_last = HostRows::of_level(last).expect("coarsest rows");
    let coarse_last = CoarseSetup::new(&hrows_last, false).expect("coarsest setup");
    let v = Vc {
        level0: &level0,
        rows0: &rows0,
        h: &h,
        coarse: &coarse_last,
    };

    // b of zero mean: b_c = sin(0.37 (c+1)) minus its mean, in f64.
    let n = m.n_cells;
    let s_raw: Vec<f64> = (0..n).map(|c| (0.37 * (c + 1) as f64).sin()).collect();
    let mean_s = s_raw.iter().sum::<f64>() / n as f64;
    let b: Vec<Scalar> = s_raw.iter().map(|&s| sc(s - mean_s)).collect();

    // x*, the pinned dense solve of the 512-cell level-0 rows.
    let x_star = coarse0.solve_host(&b, None).expect("x*");
    let x_star_inf = max_abs(&x_star);

    // The energy J = sigma (x.A x / 2 - x.b), in f64.
    let sigma = if hrows0.diag.iter().sum::<f64>() > 0.0 {
        1.0
    } else {
        -1.0
    };
    let energy = |x: &[Scalar]| -> f64 {
        let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let ax = hrows0.amul(&xf);
        let mut xax = 0.0f64;
        let mut xb = 0.0f64;
        for c in 0..n {
            xax += xf[c] * ax[c];
            xb += xf[c] * b[c] as f64;
        }
        sigma * (xax / 2.0 - xb)
    };
    let resid = |x: &[Scalar]| -> f64 {
        let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let ax = hrows0.amul(&xf);
        ax.iter()
            .zip(b.iter())
            .map(|(a, &bc)| {
                let r = bc as f64 - a;
                r * r
            })
            .sum::<f64>()
            .sqrt()
    };
    let j_star = energy(&x_star);

    let mut x = vec![0 as Scalar; n];
    let r0 = resid(&x);
    let mut prev_j = energy(&x);
    for cycle in 0..100 {
        vcycle(&v, 0, &b, &mut x).expect("cycle");
        let j = energy(&x);
        assert!(
            j <= prev_j + TOL_R * j_star.abs(),
            "cycle {cycle}: the energy rose from {prev_j:e} to {j:e} \
             (J* = {j_star:e})"
        );
        prev_j = j;
        let lo = x.iter().map(|&v| v as f64).fold(f64::INFINITY, f64::min);
        let hi = x
            .iter()
            .map(|&v| v as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let spread = hi - lo;
        let mn = mean_of(&x).abs();
        assert!(
            spread <= 10.0 * x_star_inf && mn <= 10.0 * x_star_inf,
            "cycle {cycle}: spread {spread:e}, |mean| {mn:e} exceed \
             10 ||x*||_inf = {:.e}",
            10.0 * x_star_inf
        );
    }
    let r100 = resid(&x);
    assert!(r100 <= r0, "||r_100|| {r100:e} > ||r_0|| {r0:e}");

    // The measured convergence, printed.
    let mean_x = mean_of(&x);
    let diff = x
        .iter()
        .zip(x_star.iter())
        .map(|(&xa, &xs)| ((xa as f64 - mean_x) - xs as f64).abs())
        .fold(0.0, f64::max);
    let factor = (r100 / r0).powf(1.0 / 100.0);
    println!(
        "  V-cycle: ||x_100 - mean - x*||_inf / ||x*||_inf = {:.3e}",
        diff / x_star_inf
    );
    println!("  V-cycle: mean per-cycle residual factor = {factor:.6}");
}

/// The regular, pinned and `-permissive` solves on coarse levels and on
/// level 0, each run twice, bitwise the host twin (SPEC-LIT 141.4.1,
/// class D). The Cholesky solves answer through the one-block kernel on
/// either operator; the `-permissive` fallback runs the smoother, whose
/// level-0 product `lduAmul` contracts into FMAs (`ldu.cu` is not on
/// `build.rs`'s `-fmad=false` list), so its bitwise twin exists only on
/// a coarse level, where `amgAmul` is FMA-free by its own gate.
#[test]
fn the_device_coarse_solve_is_bitwise_its_host_twin() {
    let Some(gpu) = gpu() else { return };

    // ---- regular on a coarse level: SYM through the row map ----
    {
        let l = sym6();
        let hrows = HostRows::of_level(&l).expect("rows");
        let lrows = LevelRows::of_level(&l).expect("lrows");
        let s = CoarseSetup::new(&hrows, false).expect("setup");
        let level = l.upload(&gpu).expect("level");
        let rows = GpuLevelRows::new(&gpu, &lrows).expect("rows");
        let mut g = GpuCoarse::new(&gpu, &s).expect("coarse");
        let b = test_field(l.n_cells);
        let bd = gpu.upload(&b).expect("b");
        let mut xd = gpu.zeros(l.n_cells).expect("x");
        g.solve(
            &gpu,
            CoarseOperator::Level { level: &level, rows: &rows },
            &bd,
            &mut xd,
        )
        .expect("solve");
        let x1 = gpu.download(&xd).expect("x1");
        g.solve(
            &gpu,
            CoarseOperator::Level { level: &level, rows: &rows },
            &bd,
            &mut xd,
        )
        .expect("solve");
        let x2 = gpu.download(&xd).expect("x2");
        let xh = s.solve_host(&b, None).expect("host twin");
        assert_bits(&x1, &x2, "regular coarse: run 1 vs run 2");
        assert_bits(&x1, &xh, "regular coarse: device vs host twin");
    }

    // ---- pinned on a coarse level: the coarsest level of NEU 16^3 ----
    {
        let m = box_mesh(16, 16, 16, &[]);
        let a = neu(&m);
        let h =
            HostHierarchy::build(&m, &a, &HierarchyConfig::default()).expect("hierarchy");
        let l = h.levels.last().expect("coarsest").clone();
        let hrows = HostRows::of_level(&l).expect("rows");
        let lrows = LevelRows::of_level(&l).expect("lrows");
        let s = CoarseSetup::new(&hrows, false).expect("setup");
        assert_eq!(s.method, CoarseMethod::PinnedCholesky);
        let level = l.upload(&gpu).expect("level");
        let rows = GpuLevelRows::new(&gpu, &lrows).expect("rows");
        let mut g = GpuCoarse::new(&gpu, &s).expect("coarse");
        let b = test_field(l.n_cells);
        let bd = gpu.upload(&b).expect("b");
        let mut xd = gpu.zeros(l.n_cells).expect("x");
        g.solve(
            &gpu,
            CoarseOperator::Level { level: &level, rows: &rows },
            &bd,
            &mut xd,
        )
        .expect("solve");
        let x1 = gpu.download(&xd).expect("x1");
        g.solve(
            &gpu,
            CoarseOperator::Level { level: &level, rows: &rows },
            &bd,
            &mut xd,
        )
        .expect("solve");
        let x2 = gpu.download(&xd).expect("x2");
        let xh = s.solve_host(&b, None).expect("host twin");
        assert_bits(&x1, &x2, "pinned coarse: run 1 vs run 2");
        assert_bits(&x1, &xh, "pinned coarse: device vs host twin");
    }

    // ---- the -permissive fallback on a coarse level: disc4 ----
    {
        let l = disc4();
        let hrows = HostRows::of_level(&l).expect("rows");
        let lrows = LevelRows::of_level(&l).expect("lrows");
        let s = CoarseSetup::new(&hrows, true).expect("permissive setup");
        assert_eq!(s.method, CoarseMethod::JacobiFallback);
        let level = l.upload(&gpu).expect("level");
        let rows = GpuLevelRows::new(&gpu, &lrows).expect("rows");
        let mut g = GpuCoarse::new(&gpu, &s).expect("coarse");
        let b = test_field(l.n_cells);
        let bd = gpu.upload(&b).expect("b");
        let mut xd = gpu.zeros(l.n_cells).expect("x");
        g.solve(
            &gpu,
            CoarseOperator::Level { level: &level, rows: &rows },
            &bd,
            &mut xd,
        )
        .expect("solve");
        let x1 = gpu.download(&xd).expect("x1");
        g.solve(
            &gpu,
            CoarseOperator::Level { level: &level, rows: &rows },
            &bd,
            &mut xd,
        )
        .expect("solve");
        let x2 = gpu.download(&xd).expect("x2");
        let xh = s
            .solve_host(&b, Some((&l, &lrows)))
            .expect("host twin");
        assert_bits(&x1, &x2, "fallback coarse: run 1 vs run 2");
        assert_bits(&x1, &xh, "fallback coarse: device vs host twin");
    }

    // ---- regular on level 0: the graded 6^3 Dirichlet laplacian ----
    {
        let m = graded_mesh();
        let a = poisson_folded(&m);
        let hrows = HostRows::of_matrix(&m, &a).expect("rows");
        let s = CoarseSetup::new(&hrows, false).expect("setup");
        let (mesh, mat) = upload_fine(&gpu, &m, &a).expect("upload");
        let k = LduKernels::new(&gpu).expect("kernels");
        let mut g = GpuCoarse::new(&gpu, &s).expect("coarse");
        let b = test_field(m.n_cells);
        let bd = gpu.upload(&b).expect("b");
        let mut xd = gpu.zeros(m.n_cells).expect("x");
        g.solve(
            &gpu,
            CoarseOperator::Fine { k: &k, mesh: &mesh, a: &mat },
            &bd,
            &mut xd,
        )
        .expect("solve");
        let x1 = gpu.download(&xd).expect("x1");
        g.solve(
            &gpu,
            CoarseOperator::Fine { k: &k, mesh: &mesh, a: &mat },
            &bd,
            &mut xd,
        )
        .expect("solve");
        let x2 = gpu.download(&xd).expect("x2");
        let xh = s.solve_host(&b, None).expect("host twin");
        assert_bits(&x1, &x2, "regular fine: run 1 vs run 2");
        assert_bits(&x1, &xh, "regular fine: device vs host twin");
    }

    // ---- pinned on level 0: NEU on the uniform 8^3 ----
    {
        let m = box_mesh(8, 8, 8, &[]);
        let a = neu(&m);
        let hrows = HostRows::of_matrix(&m, &a).expect("rows");
        let s = CoarseSetup::new(&hrows, false).expect("setup");
        assert_eq!(s.method, CoarseMethod::PinnedCholesky);
        let (mesh, mat) = upload_fine(&gpu, &m, &a).expect("upload");
        let k = LduKernels::new(&gpu).expect("kernels");
        let mut g = GpuCoarse::new(&gpu, &s).expect("coarse");
        let b = test_field(m.n_cells);
        let bd = gpu.upload(&b).expect("b");
        let mut xd = gpu.zeros(m.n_cells).expect("x");
        g.solve(
            &gpu,
            CoarseOperator::Fine { k: &k, mesh: &mesh, a: &mat },
            &bd,
            &mut xd,
        )
        .expect("solve");
        let x1 = gpu.download(&xd).expect("x1");
        g.solve(
            &gpu,
            CoarseOperator::Fine { k: &k, mesh: &mesh, a: &mat },
            &bd,
            &mut xd,
        )
        .expect("solve");
        let x2 = gpu.download(&xd).expect("x2");
        let xh = s.solve_host(&b, None).expect("host twin");
        assert_bits(&x1, &x2, "pinned fine: run 1 vs run 2");
        assert_bits(&x1, &xh, "pinned fine: device vs host twin");
    }
}

/// The not-SPD refusals of the gate table: named when strict, the
/// `-permissive` substitution with exactly 50 l1-Jacobi sweeps from zero
/// otherwise, and every strict refusal (SPEC-LIT 141.4.1).
#[test]
fn a_non_spd_coarse_level_is_refused_by_name() {
    // The asymmetric two-cell level: refused, named.
    let rows = HostRows::of_level(&asym2()).expect("rows");
    let msg = CoarseSetup::new(&rows, false).expect_err("asymmetric").to_string();
    assert!(msg.starts_with("amg::coarse: "), "asymmetric: {msg}");
    assert!(msg.contains("not SPD"), "asymmetric: {msg}");
    assert!(msg.contains("symmetric"), "asymmetric: {msg}");
    assert!(msg.contains("-permissive"), "asymmetric: {msg}");

    // The 3-cell [[1,-2,0],[-2,1,-2],[0,-2,1]]: pivot at 1.
    let bad3 = HostLevel {
        n_cells: 3,
        owner: vec![0, 1],
        neighbour: vec![1, 2],
        diag: vec![sc(1.0); 3],
        upper: vec![sc(-2.0); 2],
        lower: vec![sc(-2.0); 2],
    };
    let rows = HostRows::of_level(&bad3).expect("rows");
    let msg = CoarseSetup::new(&rows, false).expect_err("pivot").to_string();
    assert!(msg.starts_with("amg::coarse: "), "pivot: {msg}");
    assert!(msg.contains("not SPD"), "pivot: {msg}");
    assert!(msg.contains("pivot"), "pivot: {msg}");
    assert!(msg.contains('1'), "pivot index: {msg}");

    // Two disconnected 2-cell Neumann blocks: pivot too.
    let rows = HostRows::of_level(&disc4()).expect("rows");
    let msg = CoarseSetup::new(&rows, false).expect_err("disconnected").to_string();
    assert!(msg.contains("not SPD"), "disconnected: {msg}");
    assert!(msg.contains("pivot"), "disconnected: {msg}");

    // Under -permissive the same levels give JacobiFallback, the
    // announcement, and exactly 50 l1-Jacobi sweeps from zero - bitwise.
    for (name, level) in [
        ("asymmetric", asym2()),
        ("pivot", bad3),
        ("disconnected", disc4()),
    ] {
        let rows = HostRows::of_level(&level).expect("rows");
        let s = CoarseSetup::new(&rows, true).expect("permissive setup");
        assert_eq!(s.method, CoarseMethod::JacobiFallback, "{name}");
        let ann = s.announcement.as_deref().expect("announcement");
        assert!(ann.starts_with("amg::coarse: -permissive:"), "{name}: {ann}");
        assert!(ann.contains("50 l1-Jacobi sweeps"), "{name}: {ann}");
        assert!(ann.contains("not an exact solve"), "{name}: {ann}");
        assert!(
            ann.contains(&format!("{}-cell", rows.n_cells)),
            "{name}: the announcement must name the cell count: {ann}"
        );
        let lrows = LevelRows::of_level(&level).expect("lrows");
        let b = test_field(rows.n_cells);
        let x = s
            .solve_host(&b, Some((&level, &lrows)))
            .expect("fallback solve");
        let smoother = s.fallback.as_ref().expect("smoother");
        let mut xm = vec![0 as Scalar; rows.n_cells];
        smooth_level_host(&level, &lrows, smoother, &mut xm, &b, FALLBACK_SWEEPS)
            .expect("sweeps");
        assert_bits(&x, &xm, &format!("{name}: fallback vs 50 manual sweeps"));
    }

    // Every strict refusal. cells: a level too large for a dense factor.
    let big = HostRows {
        n_cells: MAX_DENSE_CELLS + 1,
        diag: vec![1.0; MAX_DENSE_CELLS + 1],
        offset: (0..=MAX_DENSE_CELLS + 1).collect(),
        col: Vec::new(),
        val: Vec::new(),
    };
    let msg = CoarseSetup::new(&big, false).expect_err("cells").to_string();
    assert!(msg.starts_with("amg::coarse: "), "cells: {msg}");
    assert!(msg.contains("cells"), "cells: {msg}");

    // finite: a NaN diagonal, then an infinite off-diagonal.
    let nan = HostRows {
        n_cells: 2,
        diag: vec![f64::NAN, 1.0],
        offset: vec![0, 0, 0],
        col: Vec::new(),
        val: Vec::new(),
    };
    let msg = CoarseSetup::new(&nan, false).expect_err("finite diag").to_string();
    assert!(msg.contains("finite"), "finite diag: {msg}");
    let inf = HostRows {
        n_cells: 2,
        diag: vec![1.0, 1.0],
        offset: vec![0, 1, 2],
        col: vec![1, 0],
        val: vec![f64::INFINITY, -1.0],
    };
    let msg = CoarseSetup::new(&inf, false).expect_err("finite val").to_string();
    assert!(msg.contains("finite"), "finite val: {msg}");

    // diagonal: the diagonal sums to zero.
    let zero = HostRows {
        n_cells: 2,
        diag: vec![1.0, -1.0],
        offset: vec![0, 0, 0],
        col: Vec::new(),
        val: Vec::new(),
    };
    let msg = CoarseSetup::new(&zero, false).expect_err("diagonal").to_string();
    assert!(msg.contains("diagonal"), "diagonal: {msg}");

    // length: b shorter than the level.
    let rows = HostRows::of_level(&pair(4.0, 3.0, -1.0)).expect("rows");
    let s = CoarseSetup::new(&rows, false).expect("setup");
    let msg = s.solve_host(&[sc(1.0)], None).expect_err("length").to_string();
    assert!(msg.contains("length"), "length: {msg}");

    // cells: a fallback operator whose cell count differs.
    let disc_rows = HostRows::of_level(&disc4()).expect("rows");
    let s4 = CoarseSetup::new(&disc_rows, true).expect("permissive setup");
    let level2 = pair(1.0, 1.0, -1.0);
    let lrows2 = LevelRows::of_level(&level2).expect("lrows");
    let msg = s4
        .solve_host(&test_field(4), Some((&level2, &lrows2)))
        .expect_err("operator cells")
        .to_string();
    assert!(msg.contains("cells"), "operator cells: {msg}");
}
