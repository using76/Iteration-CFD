// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! AMG-03a: the hierarchy and the smoothers, the gate table of SPEC-LIT
//! §141.3.1. The device-against-host-twin rows are bitwise in BOTH builds
//! (class D); the energy rows are strict decrease (class P); the damping,
//! bound and Lanczos rows run against their pinned oracles at `1e-12` of
//! `norm(x_0)` in f64 and `1e-5` under `single` (D-F32-1).
//!
//! An integration test, so the SAME assertions run in both builds (§118).
//! A machine without a card passes the device rows vacuously, the
//! convention of `tests/radiation_dom.rs`.
//!
//! Provenance: ORIGINAL - the meshes, the oracles and the assertions are
//! this tree's own construction, run against Baker, Falgout, Kolev & Yang,
//! LLNL-TR-489114 (2011), eqs. 2.2, 4.1-4.3, 6.1-6.2, and Saad (2003)
//! Algorithms 6.15 and 12.1, via SPEC-LIT §141.3. No GPL-licensed source
//! was consulted.

use ofgpu::amg::{
    Aggregation, GpuGalerkin, GpuHierarchy, GpuLevel, GpuLevelRows, GpuSmoother,
    HierarchyConfig, HostHierarchy, HostLevel, HostRows, LANCZOS_STEPS, LevelRows,
    SmootherKind, SmootherSetup, aggregate, lanczos_lambda_max, level_amul_host,
    smooth_level_host, tridiag_max_eig, DEFAULT_OMEGA,
};
use ofgpu::blockgen::{BlockSpec, GradedAxis, build_mesh};
use ofgpu::ldu::HostLduMatrix;
use ofgpu::ldu_ops::LduKernels;
use ofgpu::mesh::{GpuMesh, HostMesh};
use ofgpu::{DevBuf, Gpu, Label, Scalar};

/// The class-R tolerance of SPEC-LIT §141.3: `1e-12` of the field's norm
/// in f64, `1e-5` under `single` (D-F32-1).
const TOL_R: f64 = if cfg!(feature = "single") { 1e-5 } else { 1e-12 };

/// Every device test needs a card; without one the test passes vacuously.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The build's `Scalar`, from the f64 the oracles live in.
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

/// A uniform box, the fixture of `tests/amg_aggregate.rs`.
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

/// `poisson` as a solve receives it: `internal_coeffs` folded into `diag`
/// (`add_boundary_contributions`); the coupled coefficient stays.
fn poisson_folded(m: &HostMesh) -> HostLduMatrix {
    let mut a = poisson(m);
    for bf in 0..m.n_boundary_faces {
        let c = m.b_face_cells[bf] as usize;
        a.diag[c] += a.internal_coeffs[bf];
    }
    a
}

/// The unit-weight matrix of §141.1.1 on `m`: `diag = 100` everywhere,
/// `upper = lower = -1` on every internal face, nothing else - the
/// `matrix(m, true)` of `tests/amg_galerkin.rs` on a box with no
/// coupled face.
fn unit_matrix(m: &HostMesh) -> HostLduMatrix {
    let mut a = HostLduMatrix::zeros(m);
    for d in a.diag.iter_mut() {
        *d = sc(100.0);
    }
    for f in 0..m.n_internal_faces {
        a.upper[f] = sc(-1.0);
        a.lower[f] = sc(-1.0);
    }
    a
}

/// `A(c, d)` of the gate table for an edge `a < b`, symmetric.
fn a_sym(a: usize, b: usize) -> f64 {
    -(0.1 + ((11 * a + 3 * b) % 7) as f64 / 3.0)
}

/// The symmetric 6^3 level of the gate table: `a_ab = a_ba` of `a_sym` on
/// every edge, `a_cc = 20 + c/9`.
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
        l.upper[f] = sc(a_sym(a, b));
        l.lower[f] = sc(a_sym(a, b));
    }
    l
}

/// `K4`, the complete graph on 4 cells: faces `(0,1), (0,2), (0,3), (1,2),
/// (1,3), (2,3)`, `a_cc = 2`, every off-diagonal `+1`.
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

/// `P1D`, `tridiag(-1, 2, -1)` on 64 cells: face `f = (f, f + 1)`.
fn p1d() -> HostLevel {
    HostLevel {
        n_cells: 64,
        owner: (0..63).collect(),
        neighbour: (1..64).collect(),
        diag: vec![sc(2.0); 64],
        upper: vec![sc(-1.0); 63],
        lower: vec![sc(-1.0); 63],
    }
}

/// The 6x5x4 laplacian cyclic on x, as a solve receives it AND with the
/// coupled coefficient set: `boundary_coeffs[bf] = internal_coeffs[bf]`
/// on every coupled face, then the fold. Its rows are symmetric - the
/// fixture asserts it rather than assuming it.
fn poisson_cyclic_x() -> (HostMesh, HostLduMatrix) {
    let m = box_mesh(6, 5, 4, &[0]);
    let mut a = poisson(&m);
    for bf in 0..m.n_boundary_faces {
        if m.b_nbr_cell[bf] >= 0 {
            a.boundary_coeffs[bf] = a.internal_coeffs[bf];
        }
    }
    for bf in 0..m.n_boundary_faces {
        let c = m.b_face_cells[bf] as usize;
        a.diag[c] += a.internal_coeffs[bf];
    }
    let rows = HostRows::of_matrix(&m, &a).expect("rows of cyclic");
    for c in 0..rows.n_cells {
        for j in rows.offset[c]..rows.offset[c + 1] {
            let d = rows.col[j];
            if d == c {
                continue;
            }
            let mut back = None;
            for j2 in rows.offset[d]..rows.offset[d + 1] {
                if rows.col[j2] == c {
                    back = Some(rows.val[j2]);
                }
            }
            let back = back.expect("a symmetric entry behind every entry");
            let scale = rows.val[j].abs().max(back.abs()).max(1.0);
            assert!(
                (rows.val[j] - back).abs() <= 1e-12 * scale,
                "the cyclic fixture's rows are not symmetric: \
                 A({c},{d}) = {} but A({d},{c}) = {back}",
                rows.val[j]
            );
        }
    }
    (m, a)
}

/// `A(c, d)` of (141.18) for any ordered pair.
fn a_of(c: usize, d: usize) -> f64 {
    if c < d {
        -(0.1 + ((11 * c + 3 * d) % 7) as f64 / 3.0)
    } else {
        -(0.1 + ((5 * d + c) % 3) as f64 / 7.0)
    }
}

/// The non-dyadic matrix of (141.18): `upper[f] = A(owner, neighbour)`,
/// computed in f64 then cast, `diag[c] = 20 + c/9`, and
/// `boundary_coeffs[bf] = -A(c, d)` on a coupled face.
fn nondyadic(m: &HostMesh) -> HostLduMatrix {
    let mut a = HostLduMatrix::zeros(m);
    for (c, d) in a.diag.iter_mut().enumerate() {
        *d = sc(20.0 + c as f64 / 9.0);
    }
    for f in 0..m.n_internal_faces {
        let (o, nb) = (m.owner[f] as usize, m.neighbour[f] as usize);
        a.upper[f] = sc(a_of(o, nb));
        a.lower[f] = sc(a_of(nb, o));
    }
    for bf in 0..m.n_boundary_faces {
        let d = m.b_nbr_cell[bf];
        if d < 0 {
            continue;
        }
        a.boundary_coeffs[bf] = sc(-a_of(m.b_face_cells[bf] as usize, d as usize));
    }
    a
}

/// The host coarse level of one mesh/matrix pair.
fn host_level(m: &HostMesh, a: &HostLduMatrix) -> (Aggregation, ofgpu::amg::GalerkinPlan, HostLevel) {
    let agg = aggregate(m, a).expect("aggregate");
    let plan = ofgpu::amg::GalerkinPlan::for_mesh(&agg, m).expect("plan");
    let host = plan
        .coarse_host(&a.diag, &a.upper, &a.lower, &a.boundary_coeffs)
        .expect("coarse host");
    (agg, plan, host)
}

/// The device level 1 of one mesh/matrix pair, its host twin beside it.
fn device_level1(
    gpu: &Gpu,
    m: &HostMesh,
    a: &HostLduMatrix,
) -> (GpuLevel, HostLevel) {
    let (_agg, plan, host) = host_level(m, a);
    let gal = GpuGalerkin::new(gpu, &plan).expect("gpu galerkin");
    let mut level = gal.new_level(gpu).expect("level");
    let d = gpu.upload(&a.diag).expect("diag");
    let u = gpu.upload(&a.upper).expect("upper");
    let l = gpu.upload(&a.lower).expect("lower");
    let b = gpu.upload(&a.boundary_coeffs).expect("boundary coeffs");
    gal.coarse(gpu, &d, &u, &l, Some(&b), &mut level)
        .expect("coarse");
    (level, host)
}

/// The test field: `x[c] = 0.1*c + ((7c+3) mod 11)/13`, in f64, cast.
fn test_field(n: usize) -> Vec<Scalar> {
    (0..n)
        .map(|c| sc(0.1 * c as f64 + ((7 * c + 3) % 11) as f64 / 13.0))
        .collect()
}

/// The source: `b[c] = 1 + (c mod 5)/4`, in f64, cast.
fn src_field(n: usize) -> Vec<Scalar> {
    (0..n)
        .map(|c| sc(1.0 + (c % 5) as f64 / 4.0))
        .collect()
}

/// Every device buffer bitwise equal to its host twin.
fn bits_eq(h: &[Scalar], d: &[Scalar], what: &str) {
    assert_eq!(h.len(), d.len(), "{what}: lengths");
    for (i, (x, y)) in h.iter().zip(d.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: entry {i}");
    }
}

/// The energy `sigma * sum_c x_c (A x)_c` in f64 through the f64 rows.
fn energy(rows: &HostRows, x: &[Scalar]) -> f64 {
    let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let ax = rows.amul(&xf);
    rows_n_sign(rows) * xf.iter().zip(&ax).map(|(&x, &a)| x * a).sum::<f64>()
}

/// The sigma of a rows array, as the smoother setup computes it.
fn rows_n_sign(rows: &HostRows) -> f64 {
    let sum: f64 = rows.diag.iter().sum();
    if sum > 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// The unit-weight matrix of §141.1.1 on the 64^3 box quarters every
/// level exactly, so the sizes are the oracle's - six coarse levels,
/// `ceil(log4(262144 / 64))`. The §2 laplacian is not pinned: its
/// `|S_f| δ_f` differ in the last bits from face to face, which moves
/// the tie-breaks of (141.5)-(141.8) near the walls, so it takes only
/// the `6 ± 1` coarse levels of docs/17 with the limits of (141.3.1) -
/// every level smaller than the one above but at most four times
/// smaller, the last at or under 64 and the one before it over 64.
#[test]
fn the_level_sizes_on_64_cubed_are_ceil_log4_of_n_over_64() {
    let m = box_mesh(64, 64, 64, &[]);
    let h = HostHierarchy::build(&m, &unit_matrix(&m), &HierarchyConfig::default())
        .expect("hierarchy");
    assert_eq!(
        h.sizes(),
        vec![262144, 65536, 16384, 4096, 1024, 256, 64],
        "the unit-weight level sizes"
    );
    assert_eq!(h.n_coarse_levels(), 6, "ceil(log4(n / 64)) coarse levels");
    let h = HostHierarchy::build(&m, &poisson_folded(&m), &HierarchyConfig::default())
        .expect("hierarchy of the laplacian");
    let sizes = h.sizes();
    println!("the §2 laplacian on 64^3: sizes {sizes:?}");
    let nc = h.n_coarse_levels();
    assert!(
        (5..=7).contains(&nc),
        "{nc} coarse levels, not the 6 ± 1 band"
    );
    for l in 0..sizes.len() - 1 {
        assert!(
            sizes[l + 1] < sizes[l],
            "level {} is not smaller than level {l}",
            l + 1
        );
        assert!(
            4 * sizes[l + 1] >= sizes[l],
            "level {} is more than four times smaller than level {l}",
            l + 1
        );
    }
    assert!(
        *sizes.last().expect("a size") <= 64,
        "the last level is over 64"
    );
    assert!(
        sizes[sizes.len() - 2] > 64,
        "the level before the last is at or under 64"
    );
}

/// The unit-weight 32^3 quarters down to 32; `L_max` and `N_c` cut the
/// loop where (141.3.1) says. The per-level books line up: one smoother
/// per level, one plan per coarsening.
#[test]
fn the_level_sizes_on_32_cubed_and_the_limits() {
    let m = box_mesh(32, 32, 32, &[]);
    let a = unit_matrix(&m);
    let h = HostHierarchy::build(&m, &a, &HierarchyConfig::default())
        .expect("hierarchy");
    assert_eq!(
        h.sizes(),
        vec![32768, 8192, 2048, 512, 128, 32],
        "the level sizes on 32 cubed"
    );
    assert_eq!(h.smoothers.len(), h.sizes().len(), "one smoother per level");
    for (l, s) in h.smoothers.iter().enumerate() {
        assert_eq!(s.n_cells, h.sizes()[l], "smoother {l} on its level");
    }
    for (l, p) in h.plans.iter().enumerate() {
        assert_eq!(
            p.n_coarse,
            h.sizes()[l + 1],
            "plan {l} coarsens to level {}",
            l + 1
        );
    }
    for (n_coarsest, want) in [
        (32768usize, vec![32768usize]),
        (8192, vec![32768, 8192]),
        (100000, vec![32768]),
    ] {
        let cfg = HierarchyConfig {
            n_cells_coarsest: n_coarsest,
            ..HierarchyConfig::default()
        };
        let h = HostHierarchy::build(&m, &a, &cfg).expect("hierarchy");
        assert_eq!(h.sizes(), want, "n_cells_coarsest {n_coarsest}");
    }
    for (max_levels, want) in [
        (1usize, vec![32768usize]),
        (3, vec![32768, 8192, 2048]),
    ] {
        let cfg = HierarchyConfig {
            max_levels,
            ..HierarchyConfig::default()
        };
        let h = HostHierarchy::build(&m, &a, &cfg).expect("hierarchy");
        assert_eq!(h.sizes(), want, "max_levels {max_levels}");
    }
}

/// A diagonal matrix cannot coarsen: every aggregation is the identity,
/// `n_coarse >= n`, and the loop drops it and stops - one level.
#[test]
fn a_level_that_does_not_coarsen_ends_the_hierarchy() {
    let m = box_mesh(4, 4, 4, &[]);
    let mut a = HostLduMatrix::zeros(&m);
    for d in a.diag.iter_mut() {
        *d = sc(6.0);
    }
    let cfg = HierarchyConfig {
        n_cells_coarsest: 1,
        ..HierarchyConfig::default()
    };
    let h = HostHierarchy::build(&m, &a, &cfg).expect("hierarchy");
    assert_eq!(h.sizes(), vec![64], "no level was added");
    assert_eq!(h.n_coarse_levels(), 0, "no coarse level");
}

/// Every device level equals its host level bit for bit, on the 32^3
/// laplacian and on the cyclic 6x5x4 with (141.18) - and a second run of
/// the chain changes nothing.
#[test]
fn the_device_hierarchy_is_bitwise_the_host_hierarchy() {
    let Some(gpu) = gpu() else { return };
    let cases: Vec<(HostMesh, HostLduMatrix)> = vec![
        {
            let m = box_mesh(32, 32, 32, &[]);
            let a = poisson_folded(&m);
            (m, a)
        },
        {
            let m = box_mesh(6, 5, 4, &[0, 1, 2]);
            let a = nondyadic(&m);
            (m, a)
        },
    ];
    for (mi, (m, a)) in cases.iter().enumerate() {
        let cfg = HierarchyConfig {
            n_cells_coarsest: 8,
            ..HierarchyConfig::default()
        };
        let h = HostHierarchy::build(m, a, &cfg).expect("host hierarchy");
        let mut g = GpuHierarchy::new(&gpu, &h).expect("gpu hierarchy");
        // `new` only allocates: before the first `galerkin` every level's
        // coefficients are still exactly zero, so this row fails if
        // anything but the device Galerkin chain fills them.
        for l in 0..g.levels.len() {
            let d = g.levels[l].download(&gpu).expect("download fresh");
            for (i, &v) in d.diag.iter().enumerate() {
                assert_eq!(
                    v.to_bits(),
                    (0.0 as Scalar).to_bits(),
                    "case {mi} level {l}: diag[{i}] is not 0 before galerkin"
                );
            }
        }
        let mat = a.upload(&gpu).expect("upload");
        g.galerkin(&gpu, &mat).expect("galerkin");
        let mut first: Vec<(Vec<Scalar>, Vec<Scalar>, Vec<Scalar>)> = Vec::new();
        for l in 0..g.levels.len() {
            let d = g.levels[l].download(&gpu).expect("download");
            let hd = &h.levels[l];
            assert_eq!(d.owner, hd.owner, "case {mi} level {l}: owner");
            assert_eq!(d.neighbour, hd.neighbour, "case {mi} level {l}: neighbour");
            bits_eq(&hd.diag, &d.diag, &format!("case {mi} level {l}: diag"));
            bits_eq(&hd.upper, &d.upper, &format!("case {mi} level {l}: upper"));
            bits_eq(&hd.lower, &d.lower, &format!("case {mi} level {l}: lower"));
            first.push((d.diag, d.upper, d.lower));
        }
        g.galerkin(&gpu, &mat).expect("galerkin again");
        for l in 0..g.levels.len() {
            let d = g.levels[l].download(&gpu).expect("download");
            let (hd, hu, hl) = &first[l];
            bits_eq(
                hd,
                &d.diag,
                &format!("case {mi} level {l}: diag, second run"),
            );
            bits_eq(
                hu,
                &d.upper,
                &format!("case {mi} level {l}: upper, second run"),
            );
            bits_eq(
                hl,
                &d.lower,
                &format!("case {mi} level {l}: lower, second run"),
            );
        }
    }
}

/// The row map of a level is ascending and complete, and the device
/// `amgAmul` is bitwise its host twin, on `SYM` and on the device level 1
/// of the 32^3 laplacian.
#[test]
fn the_level_amul_is_bitwise_its_host_twin() {
    let Some(gpu) = gpu() else { return };
    let m32 = box_mesh(32, 32, 32, &[]);
    let a32 = poisson_folded(&m32);
    let (l_dev, l_host) = device_level1(&gpu, &m32, &a32);
    let sym = sym6();
    let up_sym = sym.upload(&gpu).expect("upload sym6");
    let pairs: [(&HostLevel, &GpuLevel); 2] = [(&sym, &up_sym), (&l_host, &l_dev)];
    for (l_h, l_d) in pairs {
        let n = l_h.n_cells;
        let rows = LevelRows::of_level(l_h).expect("level rows");
        assert_eq!(rows.offset[0], 0, "offset[0] = 0");
        assert_eq!(
            rows.offset[n] as usize,
            2 * l_h.n_faces(),
            "offset[n] = 2 n_faces"
        );
        for c in 0..n {
            let mut last = -1 as Label;
            for j in rows.offset[c] as usize..rows.offset[c + 1] as usize {
                assert!(rows.face[j] > last, "row {c}: faces strictly ascending");
                last = rows.face[j];
                let f = rows.face[j] as usize;
                if rows.flags[j] & 1 != 0 {
                    assert_eq!(l_h.owner[f] as usize, c, "row {c}: owner side");
                } else {
                    assert_eq!(l_h.neighbour[f] as usize, c, "row {c}: neighbour side");
                    assert_eq!(rows.flags[j], 0, "row {c}: neighbour flag 0");
                }
            }
        }
        let grows = GpuLevelRows::new(&gpu, &rows).expect("gpu rows");
        let x = test_field(n);
        let gx = gpu.upload(&x).expect("upload x");
        let mut gy: DevBuf<Scalar> = gpu.zeros(n).expect("zeros");
        grows.amul(&gpu, l_d, &mut gy, &gx).expect("device amul");
        let want = level_amul_host(l_h, &rows, &x).expect("host amul");
        let got = gpu.download(&gy).expect("download");
        bits_eq(&want, &got[..n], "level amul");
    }
}

/// l1-Jacobi is convergent in the energy norm (the report's (6.2), via
/// `M^T + M - A` definite): `sigma x^T A x` strictly decreases every
/// sweep, on the coarse levels through `amgAmul` and on level 0 through
/// `lduAmul`. And on `K4` from the all-ones field one sweep lands exactly
/// on zero, device and host twin alike.
#[test]
fn l1_jacobi_reduces_the_energy_every_sweep() {
    let Some(gpu) = gpu() else { return };
    let m32 = box_mesh(32, 32, 32, &[]);
    let a32 = poisson_folded(&m32);
    let (l_dev1, l_host1) = device_level1(&gpu, &m32, &a32);
    let k4l = k4();
    let p1dl = p1d();
    let syml = sym6();
    let up_k4 = k4l.upload(&gpu).expect("upload k4");
    let up_p1d = p1dl.upload(&gpu).expect("upload p1d");
    let up_sym = syml.upload(&gpu).expect("upload sym6");
    let levels: [(&str, &HostLevel, &GpuLevel); 4] = [
        ("k4", &k4l, &up_k4),
        ("p1d", &p1dl, &up_p1d),
        ("sym6", &syml, &up_sym),
        ("32^3 l1", &l_host1, &l_dev1),
    ];
    for (name, l, gdev) in levels {
        let n = l.n_cells;
        let hrows = HostRows::of_level(l).expect("rows");
        let lrows = LevelRows::of_level(l).expect("level rows");
        let s = SmootherSetup::new(&hrows, SmootherKind::L1Jacobi).expect("setup");
        let grows = GpuLevelRows::new(&gpu, &lrows).expect("gpu rows");
        let mut gs = GpuSmoother::new(&gpu, &s).expect("gpu smoother");
        let x0 = test_field(n);
        let b = vec![0 as Scalar; n];
        let mut gx = gpu.upload(&x0).expect("upload x");
        let gb = gpu.upload(&b).expect("upload b");
        let mut e = vec![energy(&hrows, &x0)];
        for _ in 0..20 {
            gs.smooth_level(&gpu, gdev, &grows, &mut gx, &gb, 1)
                .expect("sweep");
            let x = gpu.download(&gx).expect("download x");
            e.push(energy(&hrows, &x));
        }
        assert!(e[0] > 0.0, "{name}: E_0 = {} is not positive", e[0]);
        for k in 0..20 {
            assert!(
                e[k + 1] < e[k],
                "{name}: E_{} = {} >= E_{k} = {}",
                k + 1,
                e[k + 1],
                e[k]
            );
        }
    }
    for (name, (m, a)) in [
        ("graded", {
            let mm = graded_mesh();
            let aa = poisson_folded(&mm);
            (mm, aa)
        }),
        ("cyclic x", poisson_cyclic_x()),
    ] {
        let n = m.n_cells;
        let hrows = HostRows::of_matrix(&m, &a).expect("rows of level 0");
        let s = SmootherSetup::new(&hrows, SmootherKind::L1Jacobi).expect("setup");
        let mut gs = GpuSmoother::new(&gpu, &s).expect("gpu smoother");
        let mesh = GpuMesh::upload(&gpu, &m).expect("upload mesh");
        let mat = a.upload(&gpu).expect("upload matrix");
        let k = LduKernels::new(&gpu).expect("ldu kernels");
        let x0 = test_field(n);
        let b = vec![0 as Scalar; n];
        let mut gx = gpu.upload(&x0).expect("upload x");
        let gb = gpu.upload(&b).expect("upload b");
        let mut e = vec![energy(&hrows, &x0)];
        for _ in 0..20 {
            gs.smooth_fine(&gpu, &k, &mesh, &mat, &mut gx, &gb, 1)
                .expect("sweep");
            let x = gpu.download(&gx).expect("download x");
            e.push(energy(&hrows, &x));
        }
        assert!(e[0] > 0.0, "{name}: E_0 = {} is not positive", e[0]);
        for k in 0..20 {
            assert!(
                e[k + 1] < e[k],
                "{name}: E_{} = {} >= E_{k} = {}",
                k + 1,
                e[k + 1],
                e[k]
            );
        }
    }
    // K4 from the all-ones field is exactly zero after one sweep, on the
    // device and in the host twin.
    let l = k4l;
    let hrows = HostRows::of_level(&l).expect("rows");
    let lrows = LevelRows::of_level(&l).expect("level rows");
    let s = SmootherSetup::new(&hrows, SmootherKind::L1Jacobi).expect("setup");
    let grows = GpuLevelRows::new(&gpu, &lrows).expect("gpu rows");
    let mut gs = GpuSmoother::new(&gpu, &s).expect("gpu smoother");
    let gdev = l.upload(&gpu).expect("upload k4");
    let b = vec![0 as Scalar; 4];
    let x0 = vec![sc(1.0); 4];
    let mut gx = gpu.upload(&x0).expect("upload x");
    let gb = gpu.upload(&b).expect("upload b");
    gs.smooth_level(&gpu, &gdev, &grows, &mut gx, &gb, 1)
        .expect("one sweep");
    let got = gpu.download(&gx).expect("download x");
    for (c, v) in got.iter().enumerate() {
        assert_eq!(*v, sc(0.0), "device: cell {c} after one l1 sweep");
    }
    let mut xh = x0.clone();
    smooth_level_host(&l, &lrows, &s, &mut xh, &b, 1).expect("host sweep");
    for (c, v) in xh.iter().enumerate() {
        assert_eq!(*v, sc(0.0), "host: cell {c} after one l1 sweep");
    }
}

/// Plain Jacobi (`omega = 1`) diverges on `K4`, whose `D^-1 A` has
/// eigenvalues up to 2.5: the energy GROWS, first sweep and tenth, where
/// the same start under l1-Jacobi strictly decreases it.
#[test]
fn plain_jacobi_grows_the_energy_where_l1_does_not() {
    let l = k4();
    let hrows = HostRows::of_level(&l).expect("rows");
    let lrows = LevelRows::of_level(&l).expect("level rows");
    let s = SmootherSetup::new(
        &hrows,
        SmootherKind::WeightedJacobi { omega: 1.0 },
    )
    .expect("setup");
    let x0 = vec![sc(1.0), sc(0.0), sc(0.0), sc(0.0)];
    let b = vec![0 as Scalar; 4];
    let mut x = x0.clone();
    let mut e = vec![energy(&hrows, &x)];
    for _ in 0..10 {
        smooth_level_host(&l, &lrows, &s, &mut x, &b, 1).expect("sweep");
        e.push(energy(&hrows, &x));
    }
    assert!(e[1] > e[0], "E_1 = {} <= E_0 = {}", e[1], e[0]);
    assert!(e[10] > e[0], "E_10 = {} <= E_0 = {}", e[10], e[0]);
    let s1 = SmootherSetup::new(&hrows, SmootherKind::L1Jacobi).expect("l1 setup");
    let mut x = x0.clone();
    smooth_level_host(&l, &lrows, &s1, &mut x, &b, 1).expect("l1 sweep");
    let e1 = energy(&hrows, &x);
    assert!(e1 < e[0], "l1: E_1 = {} >= E_0 = {}", e1, e[0]);
}

/// Weighted Jacobi with `omega = 2/3` is a linear iteration on `P1D`: the
/// factor on eigenvector `v_k` is `|1 - 2 lambda_k / 3|`, at most `1/3`
/// over the high half `k >= 33`.
#[test]
fn weighted_jacobi_two_thirds_damps_the_high_half_by_a_third() {
    let l = p1d();
    let hrows = HostRows::of_level(&l).expect("rows");
    let lrows = LevelRows::of_level(&l).expect("level rows");
    let s = SmootherSetup::new(
        &hrows,
        SmootherKind::WeightedJacobi {
            omega: DEFAULT_OMEGA,
        },
    )
    .expect("setup");
    let b = vec![0 as Scalar; 64];
    let pi = std::f64::consts::PI;
    let norm = |w: &[Scalar]| -> f64 {
        w.iter().map(|&t| (t as f64) * (t as f64)).sum::<f64>().sqrt()
    };
    for k in 1..=64usize {
        let lam = 1.0 - (k as f64 * pi / 65.0).cos();
        let v: Vec<Scalar> = (0..64usize)
            .map(|i| sc((k as f64 * pi * (i + 1) as f64 / 65.0).sin()))
            .collect();
        let nv = norm(&v);
        let mut x = v.clone();
        smooth_level_host(&l, &lrows, &s, &mut x, &b, 1).expect("sweep");
        let ratio = norm(&x) / nv;
        let want = (1.0 - 2.0 * lam / 3.0).abs();
        assert!(
            (ratio - want).abs() <= TOL_R,
            "k = {k}: ratio {ratio} is not {want} within {TOL_R}"
        );
        if k >= 33 {
            assert!(
                ratio <= 1.0 / 3.0 + TOL_R,
                "k = {k}: ratio {ratio} exceeds 1/3 + {TOL_R}"
            );
        }
    }
}

/// Chebyshev of degree 2 on `P1D` with `a = 1/2`: the interval is exactly
/// `[1, 2]`, the bound `1/T_2(3) = 1/17`, and the factor on `v_k` is
/// `|T_2(3 - 2 lambda_k)|/17`, at most `1/17` over the high half.
#[test]
fn chebyshev_degree_two_meets_its_bound_on_the_high_half() {
    let l = p1d();
    let hrows = HostRows::of_level(&l).expect("rows");
    let lrows = LevelRows::of_level(&l).expect("level rows");
    let s = SmootherSetup::new(
        &hrows,
        SmootherKind::Chebyshev {
            degree: 2,
            fraction: 0.5,
        },
    )
    .expect("setup");
    assert_eq!(s.gershgorin, 2.0, "g of (141.3.7)");
    let cheb = s.chebyshev.as_ref().expect("chebyshev setup");
    assert_eq!(cheb.beta, 2.0, "beta");
    assert_eq!(cheb.alpha, 1.0, "alpha");
    assert_eq!(cheb.lanczos_steps, 10, "lanczos steps");
    assert!(
        (cheb.lambda_ritz - 1.980404662538769).abs() <= 1e-12 * 1.98,
        "lambda_ritz = {}",
        cheb.lambda_ritz
    );
    assert!(
        (cheb.bound - 1.0 / 17.0).abs() <= 1e-15,
        "bound = {}",
        cheb.bound
    );
    let want_c1 = [0.0, 1.0 / 17.0];
    let want_c2 = [2.0 / 3.0, 12.0 / 17.0];
    for k in 0..2 {
        let w1 = want_c1[k] as Scalar;
        assert!(
            ((cheb.c1[k] as f64) - (w1 as f64)).abs()
                <= (Scalar::EPSILON as f64) * (w1 as f64).abs(),
            "c1[{k}] = {}",
            cheb.c1[k]
        );
        let w2 = want_c2[k] as Scalar;
        assert!(
            ((cheb.c2[k] as f64) - (w2 as f64)).abs()
                <= (Scalar::EPSILON as f64) * (w2 as f64).abs(),
            "c2[{k}] = {}",
            cheb.c2[k]
        );
    }
    let b = vec![0 as Scalar; 64];
    let pi = std::f64::consts::PI;
    let norm = |w: &[Scalar]| -> f64 {
        w.iter().map(|&t| (t as f64) * (t as f64)).sum::<f64>().sqrt()
    };
    for k in 1..=64usize {
        let lam = 1.0 - (k as f64 * pi / 65.0).cos();
        let v: Vec<Scalar> = (0..64usize)
            .map(|i| sc((k as f64 * pi * (i + 1) as f64 / 65.0).sin()))
            .collect();
        let nv = norm(&v);
        let mut x = v.clone();
        smooth_level_host(&l, &lrows, &s, &mut x, &b, 1).expect("sweep");
        let ratio = norm(&x) / nv;
        let t = 3.0 - 2.0 * lam;
        let want = (2.0 * t * t - 1.0).abs() / 17.0;
        assert!(
            (ratio - want).abs() <= TOL_R,
            "k = {k}: ratio {ratio} is not {want} within {TOL_R}"
        );
        if k >= 33 {
            assert!(
                ratio <= 1.0 / 17.0 + TOL_R,
                "k = {k}: ratio {ratio} exceeds 1/17 + {TOL_R}"
            );
        }
    }
}

/// Degree-1 Chebyshev IS weighted Jacobi with `omega = 2/3`: `c2_0 =
/// 1/theta = 2/3` on `P1D`'s interval, and one sweep of each from the same
/// start lands within rounding of the same field.
#[test]
fn chebyshev_degree_one_is_weighted_jacobi_two_thirds() {
    let l = p1d();
    let hrows = HostRows::of_level(&l).expect("rows");
    let lrows = LevelRows::of_level(&l).expect("level rows");
    let sa = SmootherSetup::new(
        &hrows,
        SmootherKind::Chebyshev {
            degree: 1,
            fraction: 0.5,
        },
    )
    .expect("chebyshev setup");
    let sb = SmootherSetup::new(
        &hrows,
        SmootherKind::WeightedJacobi {
            omega: DEFAULT_OMEGA,
        },
    )
    .expect("jacobi setup");
    let x0 = test_field(64);
    let b = src_field(64);
    let mut xa = x0.clone();
    let mut xb = x0.clone();
    smooth_level_host(&l, &lrows, &sa, &mut xa, &b, 1).expect("chebyshev sweep");
    smooth_level_host(&l, &lrows, &sb, &mut xb, &b, 1).expect("jacobi sweep");
    let xmax = x0
        .iter()
        .map(|&v| (v as f64).abs())
        .fold(0.0f64, f64::max);
    for c in 0..64usize {
        assert!(
            ((xa[c] as f64) - (xb[c] as f64)).abs() <= TOL_R * xmax,
            "cell {c}: {} vs {} exceeds {} * {xmax}",
            xa[c],
            xb[c],
            TOL_R
        );
    }
}

/// The Lanczos estimate from below: the order-10 `tridiag(-1, 2, -1)`
/// gives `2 - 2 cos(10 pi / 11)`, `P1D` gives the supervisor's oracle at
/// 10 steps, and `K4` breaks down after 2 steps at exactly 2.5.
#[test]
fn lanczos_estimates_lambda_max_from_below() {
    let t10 = tridiag_max_eig(&[2.0; 10], &[-1.0; 9]).expect("tridiag");
    assert!(
        (t10 - 3.918985947228995).abs() <= 1e-14 * 3.918985947228995,
        "theta = {t10}"
    );
    let l = p1d();
    let hrows = HostRows::of_level(&l).expect("rows");
    let (theta, steps) = lanczos_lambda_max(&hrows, LANCZOS_STEPS).expect("lanczos");
    assert!(theta <= 1.9988322268323264, "theta = {theta} exceeds lambda_max");
    assert!(
        (theta - 1.980404662538769).abs() <= 1e-12 * 1.980404662538769,
        "theta = {theta}"
    );
    assert_eq!(steps, 10, "p1d runs all 10 steps");
    let k = k4();
    let krows = HostRows::of_level(&k).expect("rows");
    let (theta4, steps4) = lanczos_lambda_max(&krows, LANCZOS_STEPS).expect("lanczos");
    assert_eq!(steps4, 2, "k4 breaks down after 2 steps");
    assert!((theta4 - 2.5).abs() <= 1e-12, "theta = {theta4}");
}

/// Every smoother kind on `SYM` and on the device level 1 of the 32^3
/// laplacian: 3 sweeps with a source, the device field is bitwise the
/// host twin's, and a second run from the same start gives the same
/// bytes again.
#[test]
fn the_device_smoothers_are_bitwise_their_host_twins() {
    let Some(gpu) = gpu() else { return };
    let kinds = [
        SmootherKind::L1Jacobi,
        SmootherKind::WeightedJacobi {
            omega: DEFAULT_OMEGA,
        },
        SmootherKind::Chebyshev {
            degree: 2,
            fraction: 0.3,
        },
        SmootherKind::Chebyshev {
            degree: 3,
            fraction: 0.3,
        },
    ];
    let mut cases: Vec<(&str, HostLevel, GpuLevel)> = Vec::new();
    let sym = sym6();
    let sym_dev = sym.upload(&gpu).expect("upload sym6");
    cases.push(("sym6", sym, sym_dev));
    let m32 = box_mesh(32, 32, 32, &[]);
    let a32 = poisson_folded(&m32);
    let (l_dev, l_host) = device_level1(&gpu, &m32, &a32);
    cases.push(("32^3 level 1", l_host, l_dev));
    for (name, l_h, l_d) in &cases {
        let n = l_h.n_cells;
        let hrows = HostRows::of_level(l_h).expect("rows");
        let lrows = LevelRows::of_level(l_h).expect("level rows");
        let grows = GpuLevelRows::new(&gpu, &lrows).expect("gpu rows");
        for ki in 0..kinds.len() {
            let kind = kinds[ki];
            let s = SmootherSetup::new(&hrows, kind).expect("setup");
            let mut xh = test_field(n);
            let bh = src_field(n);
            smooth_level_host(l_h, &lrows, &s, &mut xh, &bh, 3)
                .expect("host sweeps");
            let mut gs = GpuSmoother::new(&gpu, &s).expect("gpu smoother");
            let gb = gpu.upload(&src_field(n)).expect("upload b");
            let mut gx = gpu.upload(&test_field(n)).expect("upload x");
            gs.smooth_level(&gpu, l_d, &grows, &mut gx, &gb, 3)
                .expect("device sweeps");
            let gd = gpu.download(&gx).expect("download x");
            bits_eq(&xh, &gd, &format!("{name} kind {ki}: device x"));
            let mut gx = gpu.upload(&test_field(n)).expect("upload x again");
            gs.smooth_level(&gpu, l_d, &grows, &mut gx, &gb, 3)
                .expect("device sweeps again");
            let gd = gpu.download(&gx).expect("download x again");
            bits_eq(&xh, &gd, &format!("{name} kind {ki}: second run"));
        }
    }
}

/// Every refusal of SPEC-LIT 141.3, by name: the diagonal, a non-finite
/// coefficient, the omega interval, the degree range, the fraction
/// interval, a level that is not the setup's, a field that is too short,
/// and a coupled face reaching a halo cell.
#[test]
fn a_bad_smoother_input_is_refused_by_name() {
    let l = p1d();
    let base = HostRows::of_level(&l).expect("rows");
    let check = |e: ofgpu::Error, word: &str| {
        let m = e.to_string();
        assert!(
            m.starts_with("amg::smooth: ") && m.contains(word),
            "refused message is {m:?}, which does not name {word}"
        );
    };
    for v in [0.0f64, -1.0] {
        let mut l2 = l.clone();
        l2.diag[3] = sc(v);
        let r2 = HostRows::of_level(&l2).expect("rows with a zero diagonal");
        check(
            SmootherSetup::new(&r2, SmootherKind::L1Jacobi)
                .expect_err("the diagonal is refused"),
            "diagonal",
        );
    }
    let mut l3 = l.clone();
    l3.upper[0] = Scalar::NAN;
    check(
        HostRows::of_level(&l3).expect_err("the NaN coefficient is refused"),
        "finite",
    );
    for om in [0.0f64, 2.0, f64::NAN] {
        check(
            SmootherSetup::new(
                &base,
                SmootherKind::WeightedJacobi { omega: om },
            )
            .expect_err("the omega is refused"),
            "omega",
        );
    }
    for deg in [0usize, 17] {
        check(
            SmootherSetup::new(
                &base,
                SmootherKind::Chebyshev {
                    degree: deg,
                    fraction: 0.3,
                },
            )
            .expect_err("the degree is refused"),
            "degree",
        );
    }
    for fr in [0.0f64, 1.0] {
        check(
            SmootherSetup::new(
                &base,
                SmootherKind::Chebyshev {
                    degree: 2,
                    fraction: fr,
                },
            )
            .expect_err("the fraction is refused"),
            "fraction",
        );
    }
    let Some(gpu) = gpu() else { return };
    let s = SmootherSetup::new(&base, SmootherKind::L1Jacobi).expect("setup");
    let mut gs = GpuSmoother::new(&gpu, &s).expect("gpu smoother");
    let k = k4();
    let krows = LevelRows::of_level(&k).expect("k4 rows");
    let gk = GpuLevelRows::new(&gpu, &krows).expect("k4 gpu rows");
    let kdev = k.upload(&gpu).expect("upload k4");
    let gb4 = gpu.upload(&vec![0 as Scalar; 4]).expect("upload b");
    let mut gx4 = gpu.upload(&vec![sc(1.0); 4]).expect("upload x");
    check(
        gs.smooth_level(&gpu, &kdev, &gk, &mut gx4, &gb4, 1)
            .expect_err("a 64-cell smoother on 4 cells is refused"),
        "cells",
    );
    let lrows = LevelRows::of_level(&l).expect("p1d rows");
    let gl = GpuLevelRows::new(&gpu, &lrows).expect("p1d gpu rows");
    let ldev = l.upload(&gpu).expect("upload p1d");
    let gb = gpu.upload(&vec![0 as Scalar; 64]).expect("upload b");
    let mut gx3 = gpu.upload(&vec![sc(1.0); 3]).expect("upload short x");
    check(
        gs.smooth_level(&gpu, &ldev, &gl, &mut gx3, &gb, 1)
            .expect_err("a 3-long field on 64 cells is refused"),
        "length",
    );
    let mut m = graded_mesh();
    let a = poisson_folded(&m);
    m.b_nbr_cell[0] = m.n_cells as Label;
    check(
        HostRows::of_matrix(&m, &a).expect_err("the halo cell is refused"),
        "halo",
    );
}

/// The two config refusals of the hierarchy, by name.
#[test]
fn a_bad_hierarchy_config_is_refused_by_name() {
    let m = box_mesh(4, 4, 4, &[]);
    let a = poisson_folded(&m);
    let e = HostHierarchy::build(
        &m,
        &a,
        &HierarchyConfig {
            n_cells_coarsest: 0,
            ..HierarchyConfig::default()
        },
    )
    .expect_err("n_cells_coarsest = 0 is refused");
    let msg = e.to_string();
    assert!(
        msg.starts_with("amg::hierarchy: ") && msg.contains("n_cells_coarsest"),
        "refused message is {msg:?}"
    );
    let e = HostHierarchy::build(
        &m,
        &a,
        &HierarchyConfig {
            max_levels: 0,
            ..HierarchyConfig::default()
        },
    )
    .expect_err("max_levels = 0 is refused");
    let msg = e.to_string();
    assert!(
        msg.starts_with("amg::hierarchy: ") && msg.contains("max_levels"),
        "refused message is {msg:?}"
    );
}
