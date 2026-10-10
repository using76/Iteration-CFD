// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! AMG-01: the oracle table of SPEC-LIT §141.1.1 and the rest of the §141.1
//! gates. Every expected value was pinned by an independent Python oracle
//! before any Rust ran.

use std::collections::{BTreeMap, BTreeSet};

use ofgpu::amg::*;
use ofgpu::blockgen::{BlockSpec, GradedAxis, build_mesh};
use ofgpu::ldu::HostLduMatrix;
use ofgpu::mesh::HostMesh;
use ofgpu::{Label, Scalar};

#[allow(clippy::useless_conversion)]
fn sc(x: f64) -> Scalar {
    x as Scalar
}

/// A uniform box, every patch a wall, the cyclic axes of `cyclic` set. Cells
/// are numbered `i + nx*(j + ny*k)`.
fn box_mesh(nx: usize, ny: usize, nz: usize, cyclic: &[usize]) -> HostMesh {
    let axis = |n: usize| GradedAxis {
        lo: 0.0,
        hi: n as Scalar,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    let mut b = BlockSpec {
        x: axis(nx),
        y: axis(ny),
        z: axis(nz),
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

/// `wu(a, b)` of (141.9).
fn wu(a: usize, b: usize) -> f64 {
    0.125 + ((11 * a + 3 * b) % 7) as f64
}

/// `wl(a, b)` of (141.9).
fn wl(a: usize, b: usize) -> f64 {
    0.125 + ((5 * a + b) % 3) as f64
}

/// The weighted matrix (141.9) on `m`, or the same shape with `wu = wl = 1`:
/// `A(a, b) = -wu(a, b)` and `A(b, a) = -wl(a, b)` for an edge `a < b`,
/// `diag = 100` everywhere, `internal_coeffs = 0.5` on every boundary face,
/// `boundary_coeffs = -A(c, d)` on every coupled face and 0 on the others.
fn matrix(m: &HostMesh, unit: bool) -> HostLduMatrix {
    let mut a = HostLduMatrix::zeros(m);
    for d in a.diag.iter_mut() {
        *d = sc(100.0);
    }
    for f in 0..m.n_internal_faces {
        let o = m.owner[f] as usize;
        let nb = m.neighbour[f] as usize;
        let (lo, hi) = (o.min(nb), o.max(nb));
        let s_lo = if unit { 1.0 } else { wu(lo, hi) };
        let s_hi = if unit { 1.0 } else { wl(lo, hi) };
        // A(lo, hi) = -s_lo, A(hi, lo) = -s_hi; upper[f] holds A(o, nb).
        if o < nb {
            a.upper[f] = sc(-s_lo);
            a.lower[f] = sc(-s_hi);
        } else {
            a.upper[f] = sc(-s_hi);
            a.lower[f] = sc(-s_lo);
        }
    }
    for bf in 0..m.n_boundary_faces {
        a.internal_coeffs[bf] = sc(0.5);
        let d = m.b_nbr_cell[bf];
        if d < 0 {
            continue; // uncoupled: boundary_coeffs stays 0
        }
        let c = m.b_face_cells[bf] as usize;
        let d = d as usize;
        // -A(c, d) is +wu(c, d) when c < d and +wl(d, c) when c > d.
        let v = if c < d {
            if unit { 1.0 } else { wu(c, d) }
        } else if unit {
            1.0
        } else {
            wl(d, c)
        };
        a.boundary_coeffs[bf] = sc(v);
    }
    a
}

/// The six gate rows of SPEC-LIT §141.1.1: `(mesh, weighted matrix)`.
fn all_six() -> Vec<(HostMesh, HostLduMatrix)> {
    vec![
        (box_mesh(7, 5, 3, &[]), false),
        (box_mesh(7, 5, 3, &[]), true),
        (box_mesh(6, 5, 4, &[0, 1, 2]), false),
        (box_mesh(6, 5, 4, &[0, 1, 2]), true),
        (box_mesh(32, 32, 32, &[]), false),
        (box_mesh(32, 32, 32, &[]), true),
    ]
    .into_iter()
    .map(|(m, unit)| {
        let a = matrix(&m, unit);
        (m, a)
    })
    .collect()
}

fn w753() -> (HostMesh, HostLduMatrix) {
    let m = box_mesh(7, 5, 3, &[]);
    let a = matrix(&m, false);
    (m, a)
}

fn u753() -> (HostMesh, HostLduMatrix) {
    let m = box_mesh(7, 5, 3, &[]);
    let a = matrix(&m, true);
    (m, a)
}

fn w654c() -> (HostMesh, HostLduMatrix) {
    let m = box_mesh(6, 5, 4, &[0, 1, 2]);
    let a = matrix(&m, false);
    (m, a)
}

fn u654c() -> (HostMesh, HostLduMatrix) {
    let m = box_mesh(6, 5, 4, &[0, 1, 2]);
    let a = matrix(&m, true);
    (m, a)
}

fn w32() -> (HostMesh, HostLduMatrix) {
    let m = box_mesh(32, 32, 32, &[]);
    let a = matrix(&m, false);
    (m, a)
}

fn u32() -> (HostMesh, HostLduMatrix) {
    let m = box_mesh(32, 32, 32, &[]);
    let a = matrix(&m, true);
    (m, a)
}

/// The §2 laplacian on `m`: `upper = lower = |S_f| delta_f`, the diagonal
/// minus the same sum at both cells - the negative-diagonal sign.
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

/// FNV-1a 64 of `cell_agg`: each entry as 4 little-endian bytes, cell order.
fn fnv64(cell_agg: &[Label]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &c in cell_agg {
        for b in c.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// The aggregate size histogram, e.g. `{4: 25, 3: 1, 2: 1}`.
fn sizes(agg: &Aggregation) -> BTreeMap<usize, usize> {
    let mut t = BTreeMap::new();
    for g in 0..agg.n_coarse {
        *t.entry(agg.size(g)).or_insert(0) += 1;
    }
    t
}

/// One oracle-table row; `None` where SPEC-LIT §141.1.1 does not pin a
/// number. `coupled` counts the boundary faces that are not [`UNCOUPLED`].
#[allow(clippy::too_many_arguments)]
fn check_row(
    agg: &Aggregation,
    p1: usize,
    nc: usize,
    hist: &[(usize, usize)],
    cfaces: Option<usize>,
    int: Option<usize>,
    fold_bf: Option<usize>,
    coupled: usize,
    fnv: Option<u64>,
) {
    assert_eq!(agg.n_pass1, p1, "n_pass1");
    assert_eq!(agg.n_coarse, nc, "n_coarse");
    let h = sizes(agg);
    assert_eq!(h.len(), hist.len(), "histogram {h:?} vs {hist:?}");
    let mut total = 0usize;
    for (s, c) in hist {
        assert_eq!(h.get(s), Some(c), "count of size {s}");
        total += *c;
    }
    assert_eq!(total, nc, "histogram total");
    if let Some(want) = cfaces {
        assert_eq!(agg.n_coarse_faces(), want, "coarse faces");
    }
    if let Some(want) = int {
        assert_eq!(
            agg.face_coarse.iter().filter(|&&f| f == INTERIOR).count(),
            want,
            "INTERIOR internal faces"
        );
    }
    if let Some(want) = fold_bf {
        assert_eq!(
            agg.bface_coarse.iter().filter(|&&f| f == INTERIOR).count(),
            want,
            "folded coupled faces"
        );
    }
    assert_eq!(
        agg.bface_coarse.iter().filter(|&&f| f != UNCOUPLED).count(),
        coupled,
        "coupled boundary faces"
    );
    if let Some(want) = fnv {
        assert_eq!(fnv64(&agg.cell_agg), want, "fnv");
    }
}

/// The members of aggregates `0, 1, 2`.
fn check_members(agg: &Aggregation, wants: &[&[Label]]) {
    for (g, want) in wants.iter().enumerate() {
        let lo = agg.agg_offset[g] as usize;
        let hi = agg.agg_offset[g + 1] as usize;
        assert_eq!(&agg.agg_cells[lo..hi], *want, "aggregate {g}");
    }
}

/// The first five coarse faces.
fn check_faces5(agg: &Aggregation, wants: &[(Label, Label)]) {
    for (i, want) in wants.iter().enumerate() {
        assert_eq!(
            (agg.coarse_owner[i], agg.coarse_neighbour[i]),
            *want,
            "coarse face {i}"
        );
    }
}

/// Every row and member list of the SPEC-LIT §141.1.1 table, exactly.
#[test]
fn the_oracle_table_of_spec_lit_141_1_1() {
    // W 7x5x3 weighted, no couples
    let (m, a) = w753();
    let agg = aggregate(&m, &a).unwrap();
    check_row(
        &agg,
        53,
        27,
        &[(4, 25), (3, 1), (2, 1)],
        Some(84),
        Some(91),
        Some(0),
        0,
        Some(0x5a8dbae8292ab9fa),
    );
    check_members(&agg, &[&[0, 1, 35, 36], &[2, 3, 37, 38], &[4, 5, 39, 40]]);
    check_faces5(&agg, &[(0, 1), (0, 4), (0, 19), (1, 2), (1, 5)]);

    // U 7x5x3, unit weights
    let (m, a) = u753();
    let agg = aggregate(&m, &a).unwrap();
    check_row(
        &agg,
        53,
        27,
        &[(4, 25), (3, 1), (2, 1)],
        Some(88),
        Some(95),
        Some(0),
        0,
        Some(0x3d79f46bb14637ec),
    );
    check_members(&agg, &[&[0, 1, 7, 8], &[2, 3, 9, 10], &[4, 5, 11, 12]]);
    check_faces5(&agg, &[(0, 1), (0, 4), (0, 11), (0, 12), (1, 2)]);

    // W 6x5x4, cyclic on x, y and z
    let (m, a) = w654c();
    let agg = aggregate(&m, &a).unwrap();
    check_row(
        &agg,
        60,
        30,
        &[(4, 30)],
        Some(60),
        Some(90),
        Some(60),
        148,
        Some(0x015f5cd94ef38125),
    );
    check_members(
        &agg,
        &[&[0, 30, 60, 90], &[1, 31, 61, 91], &[2, 32, 62, 92]],
    );
    check_faces5(&agg, &[(0, 1), (0, 5), (0, 6), (0, 24), (1, 2)]);

    // U 6x5x4, cyclic on x, y and z
    let (m, a) = u654c();
    let agg = aggregate(&m, &a).unwrap();
    check_row(
        &agg,
        60,
        30,
        &[(4, 30)],
        Some(93),
        Some(120),
        Some(0),
        148,
        Some(0xeef6efe900a90185),
    );
    check_members(&agg, &[&[0, 1, 6, 7], &[2, 3, 8, 9], &[4, 5, 10, 11]]);
    check_faces5(&agg, &[(0, 1), (0, 2), (0, 3), (0, 6), (0, 9)]);

    // W 32^3: the coarse face and INTERIOR counts are not pinned here
    let (m, a) = w32();
    let agg = aggregate(&m, &a).unwrap();
    check_row(
        &agg,
        16384,
        8192,
        &[(4, 8192)],
        None,
        None,
        Some(0),
        0,
        Some(0x40290cb573e1b725),
    );
    check_members(
        &agg,
        &[&[0, 32, 1024, 1056], &[1, 33, 1025, 1057], &[2, 34, 1026, 1058]],
    );

    // U 32^3: the fnv is not pinned here
    let (m, a) = u32();
    let agg = aggregate(&m, &a).unwrap();
    check_row(
        &agg,
        16384,
        8192,
        &[(4, 8192)],
        Some(23296),
        Some(32768),
        Some(0),
        0,
        None,
    );
    check_faces5(&agg, &[(0, 1), (0, 16), (0, 256), (1, 2), (1, 17)]);
}

/// Every cell in exactly one aggregate of one to four members, ascending
/// inside each aggregate, numbered by (141.8), on all six meshes.
#[test]
fn every_cell_is_in_exactly_one_aggregate_of_one_to_four() {
    for (mi, (m, a)) in all_six().into_iter().enumerate() {
        let agg = aggregate(&m, &a).unwrap();
        let n = agg.n_fine;
        assert_eq!(n, m.n_cells, "mesh {mi}: n_fine");
        assert_eq!(agg.cell_agg.len(), n, "mesh {mi}: cell_agg length");
        for &g in agg.cell_agg.iter() {
            assert!((0..agg.n_coarse as Label).contains(&g), "mesh {mi}: id {g}");
        }
        assert_eq!(agg.agg_offset[0], 0, "mesh {mi}: first offset");
        assert_eq!(
            agg.agg_offset[agg.n_coarse] as usize,
            n,
            "mesh {mi}: last offset"
        );
        let mut seen = vec![false; n];
        for g in 0..agg.n_coarse {
            let s = agg.size(g);
            assert!((1..=4).contains(&s), "mesh {mi}: size {s}");
            let (lo, hi) = (agg.agg_offset[g] as usize, agg.agg_offset[g + 1] as usize);
            let mut prev: Label = -1;
            for k in lo..hi {
                let c = agg.agg_cells[k] as usize;
                assert!(!seen[c], "mesh {mi}: cell {c} twice");
                seen[c] = true;
                assert_eq!(agg.cell_agg[c], g as Label, "mesh {mi}: cell {c}");
                assert!(agg.agg_cells[k] > prev, "mesh {mi}: members ascend");
                prev = agg.agg_cells[k];
            }
        }
        assert!(seen.iter().all(|&s| s), "mesh {mi}: a permutation");
        // (141.8): the first member strictly increases with the aggregate.
        for g in 1..agg.n_coarse {
            let a1 = agg.agg_cells[agg.agg_offset[g] as usize];
            let a0 = agg.agg_cells[agg.agg_offset[g - 1] as usize];
            assert!(a1 > a0, "mesh {mi}: numbering {a0} then {a1}");
        }
    }
}

/// The §2 laplacian of the 32^3 box coarsens by three to five.
#[test]
fn the_poisson_matrix_of_section_2_coarsens_by_three_to_five() {
    let m = box_mesh(32, 32, 32, &[]);
    let a = poisson(&m);
    let agg = aggregate(&m, &a).unwrap();
    let ratio = agg.n_fine as f64 / agg.n_coarse as f64;
    assert!(
        (3.0..=5.0).contains(&ratio),
        "n/n_c = {ratio}, not in [3, 5]"
    );
    for g in 0..agg.n_coarse {
        assert!((1..=4).contains(&agg.size(g)));
    }
}

/// `A` and `-A` aggregate identically, bit for bit (141.3).
#[test]
fn a_and_minus_a_aggregate_identically() {
    for (m, a) in [w753(), w654c()] {
        let mut neg = a.clone();
        for v in neg.diag.iter_mut() {
            *v = -*v;
        }
        for v in neg.upper.iter_mut() {
            *v = -*v;
        }
        for v in neg.lower.iter_mut() {
            *v = -*v;
        }
        for v in neg.boundary_coeffs.iter_mut() {
            *v = -*v;
        }
        let plus = aggregate(&m, &a).unwrap();
        let minus = aggregate(&m, &neg).unwrap();
        assert_eq!(plus, minus);
    }
}

/// A hundred runs on the weighted 32^3 matrix give the same bytes.
#[test]
fn a_hundred_runs_give_the_same_bytes() {
    let (m, a) = w32();
    let first = aggregate(&m, &a).unwrap();
    for _ in 0..99 {
        assert_eq!(aggregate(&m, &a).unwrap(), first);
    }
}

/// The coarse graph is connected on every gate mesh.
#[test]
fn the_coarse_graph_is_connected_when_the_fine_one_is() {
    for (mi, (m, a)) in all_six().into_iter().enumerate() {
        let agg = aggregate(&m, &a).unwrap();
        let nc = agg.n_coarse;
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); nc];
        for i in 0..agg.n_coarse_faces() {
            let o = agg.coarse_owner[i] as usize;
            let nb = agg.coarse_neighbour[i] as usize;
            adj[o].push(nb);
            adj[nb].push(o);
        }
        let mut seen = vec![false; nc];
        let mut stack = vec![0usize];
        seen[0] = true;
        while let Some(x) = stack.pop() {
            for &y in adj[x].iter() {
                if !seen[y] {
                    seen[y] = true;
                    stack.push(y);
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "mesh {mi}: not connected");
    }
}

/// x coefficients 1, y and z coefficients 2^-7: 64 aggregates of 4 cells,
/// each inside one x-line (SPEC-LIT §141.1.1).
#[test]
fn a_strong_x_coupling_gives_x_line_aggregates() {
    let m = box_mesh(16, 4, 4, &[]);
    let mut a = HostLduMatrix::zeros(&m);
    for d in a.diag.iter_mut() {
        *d = sc(100.0);
    }
    for f in 0..m.n_internal_faces {
        let v = if (m.neighbour[f] - m.owner[f]).abs() == 1 {
            1.0
        } else {
            0.0078125
        };
        a.upper[f] = sc(-v);
        a.lower[f] = sc(-v);
    }
    let agg = aggregate(&m, &a).unwrap();
    assert_eq!(agg.n_pass1, 128);
    assert_eq!(agg.n_coarse, 64);
    for g in 0..64 {
        assert_eq!(agg.size(g), 4, "aggregate {g}");
        let (lo, hi) = (agg.agg_offset[g] as usize, agg.agg_offset[g + 1] as usize);
        let first = agg.agg_cells[lo] as usize;
        // One x-line: cells are numbered i + 16*(j + 4*k), so the members
        // share j and k exactly, and their centres agree to within the
        // pyramid estimator's rounding (the corner cell's y sits one ulp
        // below 0.5, its x-neighbours' do not).
        let (j0, k0) = ((first / 16) % 4, first / 64);
        for k in lo..hi {
            let c = agg.agg_cells[k] as usize;
            assert_eq!((c / 16) % 4, j0, "cell {c} j");
            assert_eq!(c / 64, k0, "cell {c} k");
            assert!(
                (m.c[c].y - m.c[first].y).abs() < 1e-12,
                "cell {c} y = {:?} vs {:?}",
                m.c[c].y,
                m.c[first].y
            );
            assert!(
                (m.c[c].z - m.c[first].z).abs() < 1e-12,
                "cell {c} z = {:?} vs {:?}",
                m.c[c].z,
                m.c[first].z
            );
        }
    }
    check_members(&agg, &[&[0, 1, 2, 3], &[4, 5, 6, 7], &[8, 9, 10, 11]]);
    assert_eq!(fnv64(&agg.cell_agg), 0xb65e2f272aae6b25);
}

/// The dense `A` of (141.1)/(141.2) plus the boundary `internal_coeffs`.
fn dense_a(m: &HostMesh, a: &HostLduMatrix) -> Vec<f64> {
    let n = m.n_cells;
    let mut ad = vec![0.0f64; n * n];
    for c in 0..n {
        ad[c * n + c] = a.diag[c] as f64;
    }
    for bf in 0..m.n_boundary_faces {
        let c = m.b_face_cells[bf] as usize;
        ad[c * n + c] += a.internal_coeffs[bf] as f64;
    }
    for f in 0..m.n_internal_faces {
        let o = m.owner[f] as usize;
        let nb = m.neighbour[f] as usize;
        ad[o * n + nb] += a.upper[f] as f64; // (141.1)
        ad[nb * n + o] += a.lower[f] as f64;
    }
    for bf in 0..m.n_boundary_faces {
        let d = m.b_nbr_cell[bf];
        if d < 0 {
            continue;
        }
        let c = m.b_face_cells[bf] as usize;
        ad[c * n + d as usize] += -(a.boundary_coeffs[bf] as f64); // (141.2)
    }
    ad
}

/// The coarse entries the fold table produces, as full vectors.
fn coarse_entries(m: &HostMesh, agg: &Aggregation, a: &HostLduMatrix) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let nc = agg.n_coarse;
    let mut cdiag = vec![0.0f64; nc];
    let mut cupper = vec![0.0f64; agg.n_coarse_faces()];
    let mut clower = vec![0.0f64; agg.n_coarse_faces()];
    for c in 0..m.n_cells {
        cdiag[agg.cell_agg[c] as usize] += a.diag[c] as f64;
    }
    for bf in 0..m.n_boundary_faces {
        let c = agg.cell_agg[m.b_face_cells[bf] as usize] as usize;
        cdiag[c] += a.internal_coeffs[bf] as f64;
    }
    for f in 0..m.n_internal_faces {
        let io = agg.cell_agg[m.owner[f] as usize];
        let inb = agg.cell_agg[m.neighbour[f] as usize];
        if io == inb {
            cdiag[io as usize] += a.upper[f] as f64 + a.lower[f] as f64;
            continue;
        }
        let cf = agg.face_coarse[f] as usize;
        let (u, l) = (a.upper[f] as f64, a.lower[f] as f64);
        if agg.face_flip[f] {
            clower[cf] += u;
            cupper[cf] += l;
        } else {
            cupper[cf] += u;
            clower[cf] += l;
        }
    }
    for bf in 0..m.n_boundary_faces {
        let d = m.b_nbr_cell[bf];
        if d < 0 {
            continue;
        }
        let ic = agg.cell_agg[m.b_face_cells[bf] as usize];
        let idc = agg.cell_agg[d as usize];
        if ic == idc {
            cdiag[ic as usize] += -(a.boundary_coeffs[bf] as f64);
            continue;
        }
        let cf = agg.bface_coarse[bf] as usize;
        let v = -(a.boundary_coeffs[bf] as f64);
        if agg.bface_flip[bf] {
            clower[cf] += v;
        } else {
            cupper[cf] += v;
        }
    }
    (cdiag, cupper, clower)
}

/// The coarse entries placed by the fold table sum to the dense `P^T A P`
/// exactly, on the two 7x5x3 and two 6x5x4 meshes.
#[test]
fn the_coarse_entries_sum_to_the_dense_galerkin_product() {
    for (m, a) in [w753(), u753(), w654c(), u654c()] {
        let agg = aggregate(&m, &a).unwrap();
        let n = m.n_cells;
        let nc = agg.n_coarse;
        let ad = dense_a(&m, &a);
        let mut acd = vec![0.0f64; nc * nc];
        for i in 0..n {
            for j in 0..n {
                let v = ad[i * n + j];
                if v != 0.0 {
                    let (in_, jn) = (agg.cell_agg[i] as usize, agg.cell_agg[j] as usize);
                    acd[in_ * nc + jn] += v;
                }
            }
        }
        let (cdiag, cupper, clower) = coarse_entries(&m, &agg, &a);
        for i in 0..nc {
            assert_eq!(acd[i * nc + i], cdiag[i], "diagonal {i}");
        }
        let mut faces = BTreeSet::new();
        for cf in 0..agg.n_coarse_faces() {
            let o = agg.coarse_owner[cf] as usize;
            let nb = agg.coarse_neighbour[cf] as usize;
            assert_eq!(acd[o * nc + nb], cupper[cf], "upper {cf}");
            assert_eq!(acd[nb * nc + o], clower[cf], "lower {cf}");
            faces.insert((o, nb));
        }
        for i in 0..nc {
            for j in 0..nc {
                // A coarse face fills both its off-diagonal slots.
                if i != j && !faces.contains(&(i, j)) && !faces.contains(&(j, i)) {
                    assert_eq!(acd[i * nc + j], 0.0, "({i}, {j}) must be empty");
                }
            }
        }
    }
}

/// The 1-D level of test 9 as owned arrays:
/// `(owner, neighbour, diag, upper, lower)`.
fn one_d_parts() -> (Vec<Label>, Vec<Label>, Vec<Scalar>, Vec<Scalar>, Vec<Scalar>) {
    (
        (0..7).collect(),
        (1..8).collect(),
        vec![sc(2.0); 8],
        vec![sc(-1.0); 7],
        vec![sc(-1.0); 7],
    )
}

/// A coarse level with no boundary aggregates: two passes on a 1-D line.
#[test]
fn a_one_dimensional_coarse_level_with_no_boundary_aggregates_in_two_passes() {
    let (owner, neighbour, diag, upper, lower) = one_d_parts();
    let l = LevelView {
        n_cells: 8,
        owner: &owner,
        neighbour: &neighbour,
        diag: &diag,
        upper: &upper,
        lower: &lower,
        b_face_cells: &[],
        b_nbr_cell: &[],
        boundary_coeffs: &[],
    };
    let agg = aggregate_level(&l).unwrap();
    assert_eq!(agg.n_pass1, 4);
    assert_eq!(agg.n_coarse, 2);
    assert_eq!(agg.cell_agg, vec![0, 0, 0, 0, 1, 1, 1, 1]);
    assert_eq!(agg.coarse_owner, vec![0]);
    assert_eq!(agg.coarse_neighbour, vec![1]);
    assert_eq!(agg.face_coarse, vec![-1, -1, -1, 0, -1, -1, -1]);
    assert!(agg.face_flip.iter().all(|&f| !f));
}

/// The 1-D level of test 9 plus one boundary face whose neighbour is a halo
/// cell: refused naming AMG-07a (SPEC-LIT §141.1).
#[test]
fn a_halo_couple_is_refused_naming_amg_07a() {
    let (owner, neighbour, diag, upper, lower) = one_d_parts();
    let l = LevelView {
        n_cells: 8,
        owner: &owner,
        neighbour: &neighbour,
        diag: &diag,
        upper: &upper,
        lower: &lower,
        b_face_cells: &[7],
        b_nbr_cell: &[8],
        boundary_coeffs: &[sc(1.0)],
    };
    let msg = aggregate_level(&l).unwrap_err().to_string();
    assert!(msg.contains("AMG-07a"), "message: {msg}");
    assert!(msg.contains("b_nbr_cell[0]"), "message: {msg}");
}

/// A NaN in `upper` is refused by array and index.
#[test]
fn a_non_finite_coefficient_is_refused_by_name() {
    let (owner, neighbour, diag, mut upper, lower) = one_d_parts();
    upper[3] = sc(f64::NAN);
    let l = LevelView {
        n_cells: 8,
        owner: &owner,
        neighbour: &neighbour,
        diag: &diag,
        upper: &upper,
        lower: &lower,
        b_face_cells: &[],
        b_nbr_cell: &[],
        boundary_coeffs: &[],
    };
    let msg = aggregate_level(&l).unwrap_err().to_string();
    assert!(msg.contains("non-finite"), "message: {msg}");
    assert!(msg.contains("upper[3]"), "message: {msg}");
}

/// A zero diagonal sum leaves the sign of (141.3) undefined: refused.
#[test]
fn a_zero_diagonal_sum_is_refused() {
    let (owner, neighbour, mut diag, upper, lower) = one_d_parts();
    for d in diag.iter_mut() {
        *d = 0.0;
    }
    let l = LevelView {
        n_cells: 8,
        owner: &owner,
        neighbour: &neighbour,
        diag: &diag,
        upper: &upper,
        lower: &lower,
        b_face_cells: &[],
        b_nbr_cell: &[],
        boundary_coeffs: &[],
    };
    let msg = aggregate_level(&l).unwrap_err().to_string();
    assert!(msg.contains("diagonal sum"), "message: {msg}");
}

/// An `upper` that disagrees with the face count is refused by name.
#[test]
fn a_length_mismatch_is_refused() {
    let (owner, neighbour, diag, mut upper, lower) = one_d_parts();
    upper.truncate(6);
    let l = LevelView {
        n_cells: 8,
        owner: &owner,
        neighbour: &neighbour,
        diag: &diag,
        upper: &upper,
        lower: &lower,
        b_face_cells: &[],
        b_nbr_cell: &[],
        boundary_coeffs: &[],
    };
    let msg = aggregate_level(&l).unwrap_err().to_string();
    assert!(msg.contains("length"), "message: {msg}");
    assert!(msg.contains("upper"), "message: {msg}");
}
