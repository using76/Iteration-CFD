// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! AMG-02: the Galerkin coarse operator and the transfers, the gate table
//! of SPEC-LIT §141.2.1. The device-against-host-twin rows are bitwise in
//! BOTH builds; the row against the dense `P^T A P` is class R, tolerance
//! `1e-14` of `max |(A_c)_ij|` and `tol32 = 5.36870912e-6` under `single`.
//!
//! An integration test, so the SAME assertions run in both builds (§118).
//! A machine without a card passes the device rows vacuously, the
//! convention of `tests/radiation_dom.rs`.
//!
//! Provenance: ORIGINAL - the meshes, the oracles and the assertions are
//! this tree's own construction, run against Notay, ETNA 37 (2010)
//! 123-146, eqs. (2.1)-(2.3), via SPEC-LIT §141.2. No GPL-licensed source
//! was consulted.

use std::collections::{BTreeSet, HashMap};

use ofgpu::amg::{
    aggregate, aggregate_level, GalerkinPlan, GpuGalerkin, GpuLevel, HostLevel, Aggregation,
    MAX_INDEX, UNCOUPLED,
};
use ofgpu::blockgen::{BlockSpec, GradedAxis, build_mesh};
use ofgpu::ldu::HostLduMatrix;
use ofgpu::mesh::HostMesh;
use ofgpu::{Gpu, Label, Scalar};

/// The R-class tolerance of SPEC-LIT §141.2: `tol32 =
/// max(tol64, min(tol64·2^29, 1e-5))` (D-F32-1), which for `1e-14` is
/// `5.36870912e-6`.
const TOL: f64 = if cfg!(feature = "single") {
    5.36870912e-6
} else {
    1e-14
};

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

/// `wu(a, b)` of (141.9).
fn wu(a: usize, b: usize) -> f64 {
    0.125 + ((11 * a + 3 * b) % 7) as f64
}

/// `wl(a, b)` of (141.9).
fn wl(a: usize, b: usize) -> f64 {
    0.125 + ((5 * a + b) % 3) as f64
}

/// The weighted matrix (141.9) on `m`, or the same shape with `wu = wl =
/// 1`: `A(a, b) = -wu(a, b)` and `A(b, a) = -wl(a, b)` for an edge `a <
/// b`, `diag = 100` everywhere, `internal_coeffs = 0.5` on every boundary
/// face, `boundary_coeffs = -A(c, d)` on every coupled face and 0 on the
/// others.
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
/// `boundary_coeffs[bf] = -A(c, d)` on a coupled face. `internal_coeffs`
/// stays 0 - §141.2 never reads it.
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

/// The four small meshes of §141.1.1 with their weighted matrices.
fn four_small() -> Vec<(HostMesh, HostLduMatrix)> {
    vec![
        (box_mesh(7, 5, 3, &[]), false),
        (box_mesh(7, 5, 3, &[]), true),
        (box_mesh(6, 5, 4, &[0, 1, 2]), false),
        (box_mesh(6, 5, 4, &[0, 1, 2]), true),
    ]
    .into_iter()
    .map(|(m, unit)| {
        let a = matrix(&m, unit);
        (m, a)
    })
    .collect()
}

/// The two 6^3 meshes with (141.18): plain and cyclic on x, y and z.
fn two_six_cubed() -> Vec<(HostMesh, HostLduMatrix)> {
    vec![box_mesh(6, 6, 6, &[]), box_mesh(6, 6, 6, &[0, 1, 2])]
        .into_iter()
        .map(|m| {
            let a = nondyadic(&m);
            (m, a)
        })
        .collect()
}

/// The coupled boundary coefficients a plan reads, empty when it has no
/// boundary source.
fn bc_of(plan: &GalerkinPlan, a: &HostLduMatrix) -> Vec<Scalar> {
    if plan.has_boundary_sources() {
        a.boundary_coeffs.clone()
    } else {
        Vec::new()
    }
}

/// The dense `P^T A P` in f64 from `cell_agg` alone: `ad[i][j]` from
/// (141.10), (141.1) and (141.2) - the diagonal only, NO
/// `internal_coeffs` - then scattered by the aggregates.
fn dense_galerkin(m: &HostMesh, a: &HostLduMatrix, agg: &Aggregation) -> Vec<f64> {
    let n = m.n_cells;
    let nc = agg.n_coarse;
    let g = |c: usize| agg.cell_agg[c] as usize;
    let mut acd = vec![0.0f64; nc * nc];
    for c in 0..n {
        acd[g(c) * nc + g(c)] += a.diag[c] as f64; // (141.10)
    }
    for f in 0..m.n_internal_faces {
        let (o, nb) = (m.owner[f] as usize, m.neighbour[f] as usize);
        acd[g(o) * nc + g(nb)] += a.upper[f] as f64; // (141.1)
        acd[g(nb) * nc + g(o)] += a.lower[f] as f64;
    }
    for bf in 0..m.n_boundary_faces {
        let d = m.b_nbr_cell[bf];
        if d < 0 {
            continue;
        }
        let c = m.b_face_cells[bf] as usize;
        acd[g(c) * nc + g(d as usize)] += -(a.boundary_coeffs[bf] as f64); // (141.2)
    }
    acd
}

/// One slot against the dense entry.
fn one_slot(got: Scalar, want: f64, maxd: f64, exact: bool, what: &str) {
    if exact {
        assert_eq!(got as f64, want, "{what}");
    } else {
        assert!(
            (got as f64 - want).abs() <= TOL * maxd,
            "{what}: got {got}, want {want}"
        );
    }
}

/// The class-R compare of a coarse level against the dense `P^T A P`:
/// every slot to `TOL` of `max |dense|`, and nothing outside the coarse
/// pattern. `exact` rows are `assert_eq!` on the values instead.
fn compare_dense(host: &HostLevel, agg: &Aggregation, dense: &[f64], exact: bool, what: &str) {
    let nc = agg.n_coarse;
    let maxd = dense.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let mut faces = BTreeSet::new();
    for cf in 0..agg.n_coarse_faces() {
        let o = agg.coarse_owner[cf] as usize;
        let nb = agg.coarse_neighbour[cf] as usize;
        faces.insert((o, nb));
        faces.insert((nb, o));
        one_slot(host.upper[cf], dense[o * nc + nb], maxd, exact, what);
        one_slot(host.lower[cf], dense[nb * nc + o], maxd, exact, what);
    }
    for i in 0..nc {
        one_slot(host.diag[i], dense[i * nc + i], maxd, exact, what);
    }
    for i in 0..nc {
        for j in 0..nc {
            if i != j && !faces.contains(&(i, j)) {
                assert_eq!(dense[i * nc + j], 0.0, "{what}: ({i}, {j}) empty");
            }
        }
    }
}

/// Every entry bitwise equal.
fn bits_eq(h: &[Scalar], d: &[Scalar], what: &str) {
    assert_eq!(h.len(), d.len(), "{what}: lengths");
    for (i, (x, y)) in h.iter().zip(d.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: entry {i}");
    }
}

/// The host coarse level of one mesh/matrix pair.
fn host_level(m: &HostMesh, a: &HostLduMatrix) -> (Aggregation, GalerkinPlan, HostLevel) {
    let agg = aggregate(m, a).expect("aggregate");
    let plan = GalerkinPlan::for_mesh(&agg, m).expect("plan");
    let bc = bc_of(&plan, a);
    let host = plan
        .coarse_host(&a.diag, &a.upper, &a.lower, &bc)
        .expect("coarse host");
    (agg, plan, host)
}

/// The device level 1 of one mesh/matrix pair, its host twin beside it.
fn device_level1(
    gpu: &Gpu,
    m: &HostMesh,
    a: &HostLduMatrix,
) -> (GalerkinPlan, GpuGalerkin, GpuLevel, HostLevel) {
    let (_agg, plan, host) = host_level(m, a);
    let gal = GpuGalerkin::new(gpu, &plan).expect("gpu galerkin");
    let mut level = gal.new_level(gpu).expect("level");
    let d = gpu.upload(&a.diag).expect("diag");
    let u = gpu.upload(&a.upper).expect("upper");
    let l = gpu.upload(&a.lower).expect("lower");
    let b = gpu.upload(&bc_of(&plan, a)).expect("boundary coeffs");
    gal.coarse(gpu, &d, &u, &l, Some(&b), &mut level)
        .expect("coarse");
    (plan, gal, level, host)
}

/// The test field: `x[c] = 0.1*c + ((7c+3) mod 11)/13`, in f64, cast.
fn test_field(n: usize) -> Vec<Scalar> {
    (0..n)
        .map(|c| sc(0.1 * c as f64 + ((7 * c + 3) % 11) as f64 / 13.0))
        .collect()
}

/// The CSR is well formed and holds each fine entry exactly once, on the
/// four small meshes of §141.1.1 and the two 6^3 meshes. Pinned totals
/// from the gate table: sources, and how many of them sit in the
/// diagonal, upper and lower slots.
#[test]
fn the_plan_lists_every_fine_entry_exactly_once() {
    // (total; diagonal-slot sources; upper-slot; lower-slot), where the
    // two cyclic rows pin upper + lower together (480) and the two 7x5x3
    // rows pin each of them: 153/153 and 149/149.
    let pinned: [(&str, usize, usize, usize, Option<usize>, Option<usize>); 4] = [
        ("W 7x5x3", 593, 287, 306, Some(153), Some(153)),
        ("U 7x5x3", 593, 295, 298, Some(149), Some(149)),
        ("W 6x5x4 cyclic", 840, 360, 480, None, None),
        ("U 6x5x4 cyclic", 840, 360, 480, None, None),
    ];
    let mut pi = 0usize;
    for (mi, (m, a)) in four_small().into_iter().chain(two_six_cubed()).enumerate() {
        let agg = aggregate(&m, &a).expect("aggregate");
        let plan = GalerkinPlan::for_mesh(&agg, &m).expect("plan");
        let name = if mi < 4 {
            ["W 7x5x3", "U 7x5x3", "W 6x5x4 cyclic", "U 6x5x4 cyclic"][mi]
        } else {
            "6^3"
        };
        // Well formed: starts at 0, non-decreasing, ends at the sources.
        assert_eq!(plan.slot_offset[0], 0, "{name}");
        for s in 0..plan.n_slots() {
            assert!(
                plan.slot_offset[s + 1] >= plan.slot_offset[s],
                "{name}: slot {s} decreases"
            );
        }
        assert_eq!(
            plan.slot_offset[plan.n_slots()] as usize,
            plan.slot_src.len(),
            "{name}: last offset"
        );
        // Each fine entry exactly once; an uncoupled bf never.
        let mut seen: HashMap<(Label, Label), usize> = HashMap::new();
        for &code in plan.slot_src.iter() {
            let (kind, idx) = (code % 4, code / 4);
            *seen.entry((kind, idx)).or_insert(0) += 1;
            match kind {
                0 => assert!(idx >= 0 && (idx as usize) < plan.n_fine, "{name}: {code}"),
                1 | 2 => assert!(
                    idx >= 0 && (idx as usize) < plan.n_fine_faces,
                    "{name}: {code}"
                ),
                _ => {
                    assert!(idx >= 0 && (idx as usize) < plan.n_fine_bfaces, "{name}: {code}");
                    assert_ne!(
                        agg.bface_coarse[idx as usize],
                        UNCOUPLED,
                        "{name}: uncoupled bf {idx} listed"
                    );
                }
            }
        }
        assert!(
            seen.values().all(|&c| c == 1),
            "{name}: an entry listed twice"
        );
        assert_eq!(seen.len(), plan.slot_src.len(), "{name}: distinct codes");
        if mi < 4 {
            let (_want_name, want_total, want_diag, want_uplo, want_up, want_lo) = pinned[pi];
            pi += 1;
            let nd = plan.slot_offset[plan.n_coarse] as usize;
            let nu = (plan.slot_offset[plan.n_coarse + plan.n_coarse_faces]
                - plan.slot_offset[plan.n_coarse])
                as usize;
            let nl = plan.slot_src.len() - nd - nu;
            assert_eq!(plan.slot_src.len(), want_total, "{name}: total sources");
            assert_eq!(nd, want_diag, "{name}: diagonal-slot sources");
            assert_eq!(nu + nl, want_uplo, "{name}: upper+lower sources");
            if let Some(w) = want_up {
                assert_eq!(nu, w, "{name}: upper-slot sources");
            }
            if let Some(w) = want_lo {
                assert_eq!(nl, w, "{name}: lower-slot sources");
            }
        }
    }
}

/// The lists, rebuilt from the order rule §141.2 states - a diagonal slot
/// reads its members' `diag` ascending, then its INTERIOR faces ascending
/// (each upper then lower), then its INTERIOR coupled faces ascending; an
/// off-diagonal slot reads its faces ascending, then its coupled faces -
/// must be the plan's lists, slot by slot.
#[test]
fn the_list_order_is_the_one_spec_lit_states() {
    let m = box_mesh(6, 5, 4, &[0, 1, 2]);
    let a = matrix(&m, false);
    let agg = aggregate(&m, &a).expect("aggregate");
    let plan = GalerkinPlan::for_mesh(&agg, &m).expect("plan");
    let (nc, ncf) = (agg.n_coarse, agg.n_coarse_faces());
    let mut want: Vec<Vec<Label>> = (0..plan.n_slots()).map(|_| Vec::new()).collect();
    for g in 0..nc {
        for k in agg.agg_offset[g]..agg.agg_offset[g + 1] {
            want[g].push(4 * agg.agg_cells[k as usize]);
        }
    }
    for f in 0..m.n_internal_faces {
        let io = agg.cell_agg[m.owner[f] as usize];
        let fc = agg.face_coarse[f];
        if fc == ofgpu::amg::INTERIOR {
            want[io as usize].push(4 * f as Label + 1);
            want[io as usize].push(4 * f as Label + 2);
        } else {
            let (fu, fl) = if agg.face_flip[f] { (2, 1) } else { (1, 2) };
            want[nc + fc as usize].push(4 * f as Label + fu);
            want[nc + ncf + fc as usize].push(4 * f as Label + fl);
        }
    }
    for bf in 0..m.n_boundary_faces {
        let bc = agg.bface_coarse[bf];
        if bc == UNCOUPLED {
            continue;
        }
        if bc == ofgpu::amg::INTERIOR {
            let i = agg.cell_agg[m.b_face_cells[bf] as usize];
            want[i as usize].push(4 * bf as Label + 3);
        } else if agg.bface_flip[bf] {
            want[nc + ncf + bc as usize].push(4 * bf as Label + 3);
        } else {
            want[nc + bc as usize].push(4 * bf as Label + 3);
        }
    }
    let got: Vec<Vec<Label>> = (0..plan.n_slots())
        .map(|s| {
            plan.slot_src[plan.slot_offset[s] as usize..plan.slot_offset[s + 1] as usize].to_vec()
        })
        .collect();
    assert_eq!(want.len(), got.len());
    for s in 0..plan.n_slots() {
        assert_eq!(want[s], got[s], "slot {s}");
    }
}

/// (141.14)-(141.16) against a dense `P^T A P` built from `cell_agg`
/// alone: to TOL of `max |(A_c)_ij|` on the three R meshes (two 6^3 with
/// (141.18), the §2 laplacian folded on the graded 6^3), and exactly on
/// the two weighted meshes of §141.1.1; nothing outside the coarse
/// pattern.
#[test]
fn the_host_coarse_matrix_is_the_dense_galerkin_product() {
    let mut cases: Vec<(HostMesh, HostLduMatrix, bool)> = Vec::new();
    for (m, a) in two_six_cubed() {
        cases.push((m, a, false));
    }
    let g = graded_mesh();
    let gp = poisson_folded(&g);
    cases.push((g, gp, false));
    let m = box_mesh(7, 5, 3, &[]);
    let mw = matrix(&m, false);
    cases.push((m, mw, true));
    let m = box_mesh(6, 5, 4, &[0, 1, 2]);
    let mw = matrix(&m, false);
    cases.push((m, mw, true));
    for (mi, (m, a, exact)) in cases.into_iter().enumerate() {
        let what = if exact { "weighted mesh" } else { "R mesh" };
        let (agg, _plan, host) = host_level(&m, &a);
        let dense = dense_galerkin(&m, &a, &agg);
        compare_dense(&host, &agg, &dense, exact, &format!("{what} {mi}"));
    }
}

/// Coarsening the coarse level again gives `(P1 P2)^T A (P1 P2)` to the
/// same tolerance, 6^3 with (141.18).
#[test]
fn a_second_level_is_the_product_of_both_prolongations() {
    let m = box_mesh(6, 6, 6, &[]);
    let a = nondyadic(&m);
    let (agg1, _plan1, level1) = host_level(&m, &a);
    let agg2 = aggregate_level(&level1.view()).expect("aggregate level 2");
    let plan2 = GalerkinPlan::new(&agg2, &level1.owner, &[]).expect("plan 2");
    let level2 = plan2
        .coarse_host(&level1.diag, &level1.upper, &level1.lower, &[])
        .expect("coarse level 2");
    // The dense `A` at the fine level, scattered by the composed map
    // `agg2.cell_agg[agg1.cell_agg[i]]`: `(P1 P2)^T A (P1 P2)`.
    let n = m.n_cells;
    let nc2 = agg2.n_coarse;
    let comp = |i: usize| agg2.cell_agg[agg1.cell_agg[i] as usize] as usize;
    let mut ad = vec![0.0f64; n * n];
    for c in 0..n {
        ad[c * n + c] = a.diag[c] as f64; // (141.10)
    }
    for f in 0..m.n_internal_faces {
        let (o, nb) = (m.owner[f] as usize, m.neighbour[f] as usize);
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
    let mut acd2 = vec![0.0f64; nc2 * nc2];
    for i in 0..n {
        for j in 0..n {
            if ad[i * n + j] != 0.0 {
                acd2[comp(i) * nc2 + comp(j)] += ad[i * n + j];
            }
        }
    }
    compare_dense(&level2, &agg2, &acd2, false, "level 2");
}

/// `P^T` of the constant 1 is the aggregate sizes, exactly, on host and
/// device, on the six meshes of §141.1.1.
#[test]
fn restricting_the_constant_gives_the_aggregate_sizes_exactly() {
    let mut six: Vec<(HostMesh, HostLduMatrix)> = four_small();
    {
        let m = box_mesh(32, 32, 32, &[]);
        let mw = matrix(&m, false);
        let mu = matrix(&m, true);
        six.push((m.clone(), mw));
        six.push((m, mu));
    }
    let Some(gpu) = gpu() else { return };
    for (mi, (m, a)) in six.into_iter().enumerate() {
        let (agg, plan, _host) = host_level(&m, &a);
        let ones = vec![sc(1.0); agg.n_fine];
        let want: Vec<Scalar> = (0..agg.n_coarse)
            .map(|g| sc((agg.agg_offset[g + 1] - agg.agg_offset[g]) as f64))
            .collect();
        let rc = plan.restrict_host(&ones).expect("restrict host");
        assert_eq!(rc, want, "mesh {mi}: host");
        // Device
        let gal = GpuGalerkin::new(&gpu, &plan).expect("gpu galerkin");
        let rf = gpu.upload(&ones).expect("rf");
        let mut rcg = gpu.zeros(agg.n_coarse.max(1)).expect("rc");
        gal.restrict(&gpu, &rf, &mut rcg).expect("restrict");
        let got = gpu.download(&rcg).expect("download");
        assert_eq!(&got[..agg.n_coarse], want.as_slice(), "mesh {mi}: device");
    }
}

/// `P^T P e_c = size * e_c` and `(P e_c) . r = e_c . (P^T r)` exactly, on
/// small-integer fields.
#[test]
fn the_transfers_are_adjoint_and_restrict_prolong_scales_by_size() {
    let mut cases: Vec<(HostMesh, HostLduMatrix)> = Vec::new();
    let m = box_mesh(7, 5, 3, &[]);
    let mw = matrix(&m, false);
    cases.push((m, mw));
    let m = box_mesh(6, 5, 4, &[0, 1, 2]);
    let mu = matrix(&m, true);
    cases.push((m, mu));
    for (mi, (m, a)) in cases.into_iter().enumerate() {
        let (agg, plan, _host) = host_level(&m, &a);
        let r: Vec<Scalar> = (0..agg.n_fine).map(|i| sc((i % 7) as f64 - 3.0)).collect();
        let rt = plan.restrict_host(&r).expect("restrict r");
        for c in 0..agg.n_coarse {
            let size = agg.agg_offset[c + 1] - agg.agg_offset[c];
            let mut ec = vec![sc(0.0); agg.n_coarse];
            ec[c] = sc(1.0);
            let ef = plan.prolong_host(&ec).expect("prolong e_c");
            // P^T P e_c = size * e_c
            let rc = plan.restrict_host(&ef).expect("restrict P e_c");
            for k in 0..agg.n_coarse {
                let want = if k == c { sc(size as f64) } else { sc(0.0) };
                assert_eq!(rc[k], want, "mesh {mi}: P^T P e_{c} at {k}");
            }
            // (P e_c) . r = e_c . (P^T r) = rt[c]
            let mut dot = 0.0f64;
            for i in 0..agg.n_fine {
                dot += ef[i] as f64 * r[i] as f64;
            }
            assert_eq!(dot as Scalar, rt[c], "mesh {mi}: adjoint at {c}");
        }
    }
}

/// The device coarse matrix is bitwise its host twin, level 1 on the
/// three 6^3 matrices and the §2 laplacian on 32^3, and level 2 from the
/// DEVICE's own level 1.
#[test]
fn the_device_coarse_matrix_is_bitwise_its_host_twin() {
    let Some(gpu) = gpu() else { return };
    let mut cases: Vec<(HostMesh, HostLduMatrix)> = two_six_cubed();
    let g = graded_mesh();
    let gp = poisson_folded(&g);
    cases.push((g, gp));
    {
        let m = box_mesh(32, 32, 32, &[]);
        let mp = poisson_folded(&m);
        cases.push((m, mp));
    }
    for (mi, (m, a)) in cases.into_iter().enumerate() {
        let (_plan, _gal, level1, host1) = device_level1(&gpu, &m, &a);
        let dev1 = level1.download(&gpu).expect("download level 1");
        assert_eq!(dev1, host1, "mesh {mi}: level 1");
        // Level 2, from the device's own level 1.
        let agg2 = aggregate_level(&host1.view()).expect("aggregate level 2");
        let plan2 = GalerkinPlan::new(&agg2, &host1.owner, &[]).expect("plan 2");
        let host2 = plan2
            .coarse_host(&host1.diag, &host1.upper, &host1.lower, &[])
            .expect("coarse host 2");
        let gal2 = GpuGalerkin::new(&gpu, &plan2).expect("gpu galerkin 2");
        let mut level2 = gal2.new_level(&gpu).expect("level 2");
        gal2.coarse_of_level(&gpu, &level1, &mut level2)
            .expect("coarse of level");
        let dev2 = level2.download(&gpu).expect("download level 2");
        assert_eq!(dev2, host2, "mesh {mi}: level 2");
    }
}

/// Restrict, prolong and prolong-add on a non-trivial field are bitwise
/// their host twins, on the same four matrices.
#[test]
fn the_device_transfers_are_bitwise_their_host_twins() {
    let Some(gpu) = gpu() else { return };
    let mut cases: Vec<(HostMesh, HostLduMatrix)> = two_six_cubed();
    let g = graded_mesh();
    let gp = poisson_folded(&g);
    cases.push((g, gp));
    {
        let m = box_mesh(32, 32, 32, &[]);
        let mp = poisson_folded(&m);
        cases.push((m, mp));
    }
    for (mi, (m, a)) in cases.into_iter().enumerate() {
        let (agg, plan, _host) = host_level(&m, &a);
        let gal = GpuGalerkin::new(&gpu, &plan).expect("gpu galerkin");
        let x = test_field(agg.n_fine);
        let xc = test_field(agg.n_coarse);
        // restrict
        let rf = gpu.upload(&x).expect("rf");
        let mut rc = gpu.zeros(agg.n_coarse.max(1)).expect("rc");
        gal.restrict(&gpu, &rf, &mut rc).expect("restrict");
        let got_rc = gpu.download(&rc).expect("rc");
        bits_eq(
            &plan.restrict_host(&x).expect("host rc"),
            &got_rc,
            &format!("mesh {mi}: rc"),
        );
        // prolong
        let ec = gpu.upload(&xc).expect("ec");
        let mut ef = gpu.zeros(agg.n_fine).expect("ef");
        gal.prolong(&gpu, &ec, &mut ef).expect("prolong");
        let got_ef = gpu.download(&ef).expect("ef");
        bits_eq(
            &plan.prolong_host(&xc).expect("host ef"),
            &got_ef,
            &format!("mesh {mi}: ef"),
        );
        // prolong-add, from the same coarse field
        let mut xf = x.clone();
        plan.prolong_add_host(&xc, &mut xf).expect("host xf");
        let mut xfd = gpu.upload(&x).expect("xf");
        gal.prolong_add(&gpu, &ec, &mut xfd).expect("prolong add");
        let got_xf = gpu.download(&xfd).expect("xf");
        bits_eq(&xf, &got_xf, &format!("mesh {mi}: xf"));
    }
}

/// Two plans, two device products, the same bytes; and a second product
/// into the same level changes nothing.
#[test]
fn running_twice_gives_identical_bytes() {
    let Some(gpu) = gpu() else { return };
    let m = box_mesh(6, 6, 6, &[0, 1, 2]);
    let a = nondyadic(&m);
    let agg = aggregate(&m, &a).expect("aggregate");
    let plan1 = GalerkinPlan::for_mesh(&agg, &m).expect("plan 1");
    let plan2 = GalerkinPlan::for_mesh(&agg, &m).expect("plan 2");
    let gal1 = GpuGalerkin::new(&gpu, &plan1).expect("gal 1");
    let gal2 = GpuGalerkin::new(&gpu, &plan2).expect("gal 2");
    let mut level1 = gal1.new_level(&gpu).expect("level a");
    let mut level2 = gal2.new_level(&gpu).expect("level b");
    let d = gpu.upload(&a.diag).expect("diag");
    let u = gpu.upload(&a.upper).expect("upper");
    let l = gpu.upload(&a.lower).expect("lower");
    let b = gpu.upload(&a.boundary_coeffs).expect("bcoeffs");
    gal1.coarse(&gpu, &d, &u, &l, Some(&b), &mut level1)
        .expect("coarse 1");
    gal2.coarse(&gpu, &d, &u, &l, Some(&b), &mut level2)
        .expect("coarse 2");
    let got1 = level1.download(&gpu).expect("download 1");
    let got2 = level2.download(&gpu).expect("download 2");
    assert_eq!(got1, got2, "two plans");
    // A second product into the same level changes nothing.
    gal2.coarse(&gpu, &d, &u, &l, Some(&b), &mut level2)
        .expect("coarse again");
    let again = level2.download(&gpu).expect("download again");
    assert_eq!(got2, again, "second product");
}

/// 2x2x1 unit weights: one aggregate, no coarse face, `diag_c[0] = 4*100
/// - 8 = 392` exactly on host and device, and restrict of ones is `[4]`.
#[test]
fn one_aggregate_has_no_coarse_faces() {
    let m = box_mesh(2, 2, 1, &[]);
    let a = matrix(&m, true);
    let agg = aggregate(&m, &a).expect("aggregate");
    assert_eq!(agg.n_coarse, 1, "one aggregate");
    assert_eq!(agg.n_coarse_faces(), 0, "no coarse face");
    let plan = GalerkinPlan::for_mesh(&agg, &m).expect("plan");
    assert_eq!(plan.n_slots(), 1);
    let host = plan
        .coarse_host(&a.diag, &a.upper, &a.lower, &[])
        .expect("coarse host");
    assert_eq!(host.diag[0], sc(392.0), "host diagonal");
    assert!(host.upper.is_empty() && host.lower.is_empty());
    let ones = vec![sc(1.0); agg.n_fine];
    assert_eq!(plan.restrict_host(&ones).expect("restrict"), vec![sc(4.0)]);
    // Device
    let Some(gpu) = gpu() else { return };
    let gal = GpuGalerkin::new(&gpu, &plan).expect("gpu galerkin");
    let mut level = gal.new_level(&gpu).expect("level");
    let d = gpu.upload(&a.diag).expect("diag");
    let u = gpu.upload(&a.upper).expect("upper");
    let l = gpu.upload(&a.lower).expect("lower");
    let b = gpu.upload(&a.boundary_coeffs).expect("bcoeffs");
    gal.coarse(&gpu, &d, &u, &l, Some(&b), &mut level)
        .expect("coarse");
    let dev = level.download(&gpu).expect("download");
    assert_eq!(dev.diag[0], sc(392.0), "device diagonal");
    let rf = gpu.upload(&ones).expect("rf");
    let mut rc = gpu.zeros(1).expect("rc");
    gal.restrict(&gpu, &rf, &mut rc).expect("restrict");
    assert_eq!(gpu.download(&rc).expect("rc"), vec![sc(4.0)]);
}

/// Every refusal of §141.2 names itself: an addressing length that
/// disagrees with the aggregation, a level too large for (141.17), a
/// short buffer, a plan with boundary sources given no
/// `boundary_coeffs`, and a coarse level whose shape is not the plan's.
#[test]
fn a_mismatched_input_is_refused_by_name() {
    let m = box_mesh(6, 5, 4, &[0, 1, 2]);
    let a = matrix(&m, false);
    let agg = aggregate(&m, &a).expect("aggregate");
    let plan = GalerkinPlan::for_mesh(&agg, &m).expect("plan");
    let n = m.n_cells;
    let nf = m.n_internal_faces;

    // owner length mismatch
    let msg = GalerkinPlan::new(&agg, &m.owner[..nf - 1], &m.b_face_cells)
        .unwrap_err()
        .to_string();
    assert!(msg.contains("length") && msg.contains("owner"), "message: {msg}");
    // b_face_cells length mismatch
    let mut extra = m.b_face_cells.clone();
    extra.push(0);
    let msg = GalerkinPlan::new(&agg, &m.owner, &extra)
        .unwrap_err()
        .to_string();
    assert!(
        msg.contains("length") && msg.contains("b_face_cells"),
        "message: {msg}"
    );
    // A level of 2^29 cells does not fit (141.17), refused through the
    // small checking path - no huge allocation.
    let big = Aggregation {
        n_fine: MAX_INDEX,
        n_pass1: 1,
        n_coarse: 1,
        cell_agg: vec![0],
        agg_offset: vec![0, 1],
        agg_cells: vec![0],
        coarse_owner: vec![],
        coarse_neighbour: vec![],
        face_coarse: vec![],
        face_flip: vec![],
        bface_coarse: vec![],
        bface_flip: vec![],
    };
    let msg = GalerkinPlan::new(&big, &[], &[]).unwrap_err().to_string();
    assert!(msg.contains("2^29"), "message: {msg}");

    // Host short buffers.
    let bc = bc_of(&plan, &a);
    let msg = plan
        .coarse_host(&a.diag[..n - 1], &a.upper, &a.lower, &bc)
        .unwrap_err()
        .to_string();
    assert!(msg.contains("length") && msg.contains("diag"), "message: {msg}");
    let msg = plan
        .coarse_host(&a.diag, &a.upper[..nf - 1], &a.lower, &bc)
        .unwrap_err()
        .to_string();
    assert!(msg.contains("length") && msg.contains("upper"), "message: {msg}");
    let msg = plan
        .coarse_host(&a.diag, &a.upper, &a.lower[..nf - 1], &bc)
        .unwrap_err()
        .to_string();
    assert!(msg.contains("length") && msg.contains("lower"), "message: {msg}");
    // Boundary sources but no boundary_coeffs: the mesh is cyclic, so the
    // plan has them.
    assert!(plan.has_boundary_sources());
    let msg = plan
        .coarse_host(&a.diag, &a.upper, &a.lower, &[])
        .unwrap_err()
        .to_string();
    assert!(msg.contains("boundary_coeffs"), "message: {msg}");

    // The same on the device.
    let Some(gpu) = gpu() else { return };
    let gal = GpuGalerkin::new(&gpu, &plan).expect("gal");
    let mut level = gal.new_level(&gpu).expect("level");
    let d = gpu.upload(&a.diag).expect("diag");
    let u = gpu.upload(&a.upper).expect("upper");
    let l = gpu.upload(&a.lower).expect("lower");
    let b = gpu.upload(&bc).expect("bcoeffs");
    let msg = gal
        .coarse(&gpu, &d, &u, &l, None, &mut level)
        .unwrap_err()
        .to_string();
    assert!(msg.contains("boundary_coeffs"), "message: {msg}");
    let dshort = gpu.upload(&a.diag[..n - 1]).expect("diag short");
    let msg = gal
        .coarse(&gpu, &dshort, &u, &l, Some(&b), &mut level)
        .unwrap_err()
        .to_string();
    assert!(msg.contains("length") && msg.contains("diag"), "message: {msg}");

    // Short transfer buffers: rf, rc, ec, ef, xf.
    let x = test_field(n);
    let nc = plan.n_coarse;
    let rfshort = gpu.upload(&x[..n - 1]).expect("rf short");
    let rfull = gpu.upload(&x).expect("rf");
    let mut rc = gpu.zeros(nc.max(1)).expect("rc");
    let msg = gal
        .restrict(&gpu, &rfshort, &mut rc)
        .unwrap_err()
        .to_string();
    assert!(msg.contains("length") && msg.contains("rf"), "message: {msg}");
    let mut rcshort = gpu.zeros((nc - 1).max(1)).expect("rc short");
    let msg = gal
        .restrict(&gpu, &rfull, &mut rcshort)
        .unwrap_err()
        .to_string();
    assert!(msg.contains("length") && msg.contains("rc"), "message: {msg}");
    let xc = test_field(nc);
    let ecshort = gpu.upload(&xc[..nc - 1]).expect("ec short");
    let ecfull = gpu.upload(&xc).expect("ec");
    let mut ef = gpu.zeros(n).expect("ef");
    let msg = gal
        .prolong(&gpu, &ecshort, &mut ef)
        .unwrap_err()
        .to_string();
    assert!(msg.contains("length") && msg.contains("ec"), "message: {msg}");
    let mut efshort = gpu.zeros(n - 1).expect("ef short");
    let msg = gal
        .prolong(&gpu, &ecfull, &mut efshort)
        .unwrap_err()
        .to_string();
    assert!(msg.contains("length") && msg.contains("ef"), "message: {msg}");
    let mut xf = gpu.upload(&x).expect("xf");
    let msg = gal
        .prolong_add(&gpu, &ecshort, &mut xf)
        .unwrap_err()
        .to_string();
    assert!(msg.contains("length") && msg.contains("ec"), "message: {msg}");
    let mut xshort = gpu.upload(&x[..n - 1]).expect("xf short");
    let msg = gal
        .prolong_add(&gpu, &ecfull, &mut xshort)
        .unwrap_err()
        .to_string();
    assert!(msg.contains("length") && msg.contains("xf"), "message: {msg}");

    // A coarse level whose shape is not the plan's.
    let mut wrong = GpuLevel {
        n_cells: plan.n_coarse + 1,
        n_faces: plan.n_coarse_faces,
        owner: gpu.upload(&[0 as Label; 1]).expect("owner"),
        neighbour: gpu.upload(&[0 as Label; 1]).expect("neighbour"),
        diag: gpu.zeros(1).expect("diag"),
        upper: gpu.zeros(1).expect("upper"),
        lower: gpu.zeros(1).expect("lower"),
    };
    let msg = gal
        .coarse(&gpu, &d, &u, &l, Some(&b), &mut wrong)
        .unwrap_err()
        .to_string();
    assert!(msg.contains("coarse level"), "message: {msg}");

    // A coarse level with the plan's counts but short coefficient
    // buffers: refused by name, one case per buffer, the others full.
    let (n1, nf1) = (plan.n_coarse.max(1), plan.n_coarse_faces.max(1));
    let zeros = |n: usize| gpu.zeros(n).expect("zeros");
    for (name, k) in [
        ("coarse diag", 0),
        ("coarse upper", 1),
        ("coarse lower", 2),
    ] {
        let mut short = GpuLevel {
            n_cells: plan.n_coarse,
            n_faces: plan.n_coarse_faces,
            owner: gpu.upload(&[0 as Label; 1]).expect("owner"),
            neighbour: gpu.upload(&[0 as Label; 1]).expect("neighbour"),
            diag: zeros(if k == 0 { 1 } else { n1 }),
            upper: zeros(if k == 1 { 1 } else { nf1 }),
            lower: zeros(if k == 2 { 1 } else { nf1 }),
        };
        let msg = gal
            .coarse(&gpu, &d, &u, &l, Some(&b), &mut short)
            .unwrap_err()
            .to_string();
        assert!(
            msg.contains("length") && msg.contains(name),
            "message: {msg}"
        );
    }
}

/// The host twin takes the matrix's own `boundary_coeffs` even when no
/// face is coupled - the device path passes the full array - and only a
/// length that is neither 0 nor `n_fine_bfaces` is refused.
#[test]
fn the_host_twin_takes_the_matrix_boundary_array_when_no_face_is_coupled() {
    let m = box_mesh(7, 5, 3, &[]);
    let a = matrix(&m, false);
    let agg = aggregate(&m, &a).expect("aggregate");
    let plan = GalerkinPlan::for_mesh(&agg, &m).expect("plan");
    assert!(!plan.has_boundary_sources(), "no face is coupled");
    assert_eq!(plan.n_fine_bfaces, 142, "the box's boundary faces");
    let full = plan
        .coarse_host(&a.diag, &a.upper, &a.lower, &a.boundary_coeffs)
        .expect("coarse host with the matrix's own array");
    let empty = plan
        .coarse_host(&a.diag, &a.upper, &a.lower, &[])
        .expect("coarse host with no array");
    assert_eq!(full, empty, "the two give the same level");
    let bad = vec![sc(0.0); 5];
    let msg = plan
        .coarse_host(&a.diag, &a.upper, &a.lower, &bad)
        .unwrap_err()
        .to_string();
    assert!(
        msg.contains("length") && msg.contains("boundary_coeffs"),
        "message: {msg}"
    );
}
