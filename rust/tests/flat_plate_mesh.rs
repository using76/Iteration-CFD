// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! VAL-30's gate: SPEC-LIT §144.1's flat-plate mesh through the public API,
//! in both builds. The fixture is `FlatPlateSpec::laminar()` (150 x 80
//! cells, the leading edge 0.25 m from the inlet). The floor is split at the
//! edge into `symmetry` and `plate` with exact face ranges (class D), the
//! edge is a node shared by the last symmetry face and the first plate face
//! (D), the cell-size ratios of the three graded segments hold to 1e-12 on
//! the f64 nodes (A: evaluated in f64, same tolerance in both builds), the
//! mesh points are the nodes rounded once to `Scalar` (D), and the
//! first-cell y+ estimate is (144.1c) at the wall-cell centre.
//!
//! Provenance: ORIGINAL. The grading law is (144.1a)-(144.1b) and the y+
//! estimate Blasius's local skin friction (144.1c), both in SPEC-LIT §144.1.
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{
    build_case, build_flat_plate_mesh, flat_plate_first_cell_yplus, flat_plate_nodes,
    flat_plate_raw_mesh, flat_plate_summary, graded_segment_f64, write_case, CaseKind,
    FlatPlateSpec, FLAT_PLATE_PATCH_NAMES,
};
use ofgpu::mesh::PatchKind;
use ofgpu::Scalar;

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// The x extent `[min, max]` of boundary face `f` (a global face index).
fn face_x_range(raw: &ofgpu::io::polymesh::PolyMeshRaw, f: usize) -> (f64, f64) {
    let xs: Vec<f64> = raw.faces[f].iter().map(|&p| raw.points[p as usize].x as f64).collect();
    (xs.iter().cloned().fold(f64::MAX, f64::min), xs.iter().cloned().fold(f64::MIN, f64::max))
}

#[test]
fn flat_plate_floor_is_split_at_the_leading_edge() {
    let s = FlatPlateSpec::laminar();
    let raw = flat_plate_raw_mesh(&s).expect("plate mesh");
    let n_internal = raw.neighbour.len();
    assert_eq!(n_internal, 149 * 80 + 150 * 79);
    assert_eq!(n_internal, 23_770);

    let names: Vec<&str> = raw.patches.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, FLAT_PLATE_PATCH_NAMES.to_vec());
    let starts: Vec<usize> = raw.patches.iter().map(|p| p.start).collect();
    let sizes: Vec<usize> = raw.patches.iter().map(|p| p.size).collect();
    assert_eq!(sizes, vec![80, 80, 30, 120, 150, 12_000, 12_000]);
    assert_eq!(starts, vec![0, 80, 160, 190, 310, 460, 12_460]);
    let kinds: Vec<PatchKind> = raw.patches.iter().map(|p| p.kind).collect();
    assert_eq!(
        kinds,
        vec![
            PatchKind::Generic,
            PatchKind::Generic,
            PatchKind::Symmetry,
            PatchKind::Wall,
            PatchKind::Generic,
            PatchKind::Empty,
            PatchKind::Empty,
        ]
    );

    // The window's faces are the floor columns i = 0..30 in order, and the
    // rest of the floor is i = 30..150: every face's owner is the wall-row
    // cell i (cell = i + nx j, j = 0).
    for (p, first_i) in [(2usize, 0usize), (3, 30)] {
        let patch = &raw.patches[p];
        for k in 0..patch.size {
            let f = n_internal + patch.start + k;
            assert_eq!(raw.owner[f] as usize, first_i + k, "{} face {k}", patch.name);
            let (lo, hi) = face_x_range(&raw, f);
            if p == 2 {
                assert!(hi <= 0.0, "a symmetry face reaches past the edge: [{lo}, {hi}]");
            } else {
                assert!(lo >= 0.0, "a plate face starts ahead of the edge: [{lo}, {hi}]");
            }
        }
    }

    let mesh = build_flat_plate_mesh(&s).expect("host mesh");
    assert_eq!(mesh.n_cells, 12_000);
    assert_eq!(mesh.patches.len(), 7);
}

#[test]
fn flat_plate_le_lies_on_a_face_boundary() {
    let s = FlatPlateSpec::laminar();
    let [x, _, _] = flat_plate_nodes(&s).expect("nodes");
    assert_eq!(x.len(), 151);
    assert_eq!(x[30].to_bits(), 0.0f64.to_bits(), "the edge node is +0.0 exactly");
    assert_eq!(x[0], -0.25);
    assert_eq!(x[150], 1.0);

    let raw = flat_plate_raw_mesh(&s).expect("plate mesh");
    let n_internal = raw.neighbour.len();
    let last_sym = n_internal + raw.patches[2].start + raw.patches[2].size - 1;
    let first_plate = n_internal + raw.patches[3].start;
    assert_eq!(face_x_range(&raw, last_sym).1, 0.0);
    assert_eq!(face_x_range(&raw, first_plate).0, 0.0);
    // The two faces share the edge's two points.
    let edge = |f: usize| -> Vec<i64> {
        let mut v: Vec<i64> = raw.faces[f]
            .iter()
            .filter(|&&p| raw.points[p as usize].x as f64 == 0.0)
            .map(|&p| p as i64)
            .collect();
        v.sort();
        v
    };
    assert_eq!(edge(last_sym).len(), 2);
    assert_eq!(edge(last_sym), edge(first_plate));
}

#[test]
fn flat_plate_grading_ratios_hold_to_1e_12() {
    let s = FlatPlateSpec::laminar();
    let [x, y, z] = flat_plate_nodes(&s).expect("nodes");
    let sizes = |v: &[f64]| v.windows(2).map(|w| w[1] - w[0]).collect::<Vec<f64>>();

    // The upstream segment, read from the edge outward; the plate; the layer.
    let up: Vec<f64> = sizes(&x[..=30]).into_iter().rev().collect();
    let plate = sizes(&x[30..]);
    let layer = sizes(&y);
    for (name, d, e) in [("upstream", &up, 20.0f64), ("plate", &plate, 20.0), ("y", &layer, 200.0)] {
        let r = e.powf(1.0 / ((d.len() - 1) as f64));
        assert!(rel(d[d.len() - 1] / d[0], e) < 1e-12, "{name}: last/first {}", d[d.len() - 1] / d[0]);
        for (i, w) in d.windows(2).enumerate() {
            assert!(rel(w[1] / w[0], r) < 1e-12, "{name}: ratio {i} = {} vs {r}", w[1] / w[0]);
        }
        assert!(d.iter().all(|&h| h > 0.0), "{name}: a non-positive cell");
    }
    assert_eq!((y[0], y[80]), (0.0, 0.25));
    assert_eq!(z, vec![0.0, 0.01]);

    // (144.1a) itself: E = 1 and n = 1 are uniform, the endpoints exact.
    assert_eq!(graded_segment_f64(0.0, 1.0, 4, 1.0), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    assert_eq!(graded_segment_f64(2.0, 3.0, 1, 7.0), vec![2.0, 3.0]);

    // The mesh points are the nodes rounded once to Scalar (class D).
    let raw = flat_plate_raw_mesh(&s).expect("plate mesh");
    for j in 0..=80usize {
        for i in 0..=150usize {
            let p = raw.points[i + 151 * j];
            assert_eq!(p.x as f64, (x[i] as Scalar) as f64, "x of point ({i}, {j})");
            assert_eq!(p.y as f64, (y[j] as Scalar) as f64, "y of point ({i}, {j})");
        }
    }
}

#[test]
fn the_first_cell_yplus_estimate_is_blasius_at_the_wall_cell_centre() {
    // By hand: Re_x = 2e5, C_f = 0.664/sqrt(2e5) = 1.484753e-3,
    // u_tau = sqrt(C_f/2) = 0.02724655, y+ = 5e-5 u_tau/5e-6.
    let y = flat_plate_first_cell_yplus(1e-4, 1.0, 5e-6, 1.0);
    assert!(rel(y, 0.272_465_514_979_406_2) < 1e-12, "{y}");

    let sum = flat_plate_summary(&FlatPlateSpec::laminar()).expect("summary");
    assert_eq!(sum.n_cells, 12_000);
    assert!(rel(sum.re_l, 2e5) < 1e-15);
    assert!(rel(sum.x_first_face, 6.533_554_633_336_538e-4) < 1e-12, "{}", sum.x_first_face);
    assert!(rel(sum.yplus_first_face, 1.388_347_997_047_032_7) < 1e-10, "{}", sum.yplus_first_face);
    assert!(rel(sum.yplus_trailing_edge, 0.222_696_716_722_249_3) < 1e-10, "{}", sum.yplus_trailing_edge);
    assert!(rel(sum.dx_le_plate / sum.dx_le_upstream, 1.017_101_184_432_189) < 1e-10);
    assert!(rel(sum.dy_wall, 8.146_558_431_918_578e-5) < 1e-12, "{}", sum.dy_wall);
}

#[test]
fn a_flat_plate_that_cannot_be_meshed_is_refused_by_name() {
    let mut s = FlatPlateSpec::laminar();
    s.r_plate = 0.0;
    assert!(flat_plate_nodes(&s).unwrap_err().to_string().contains("r_plate"));
    let mut s = FlatPlateSpec::laminar();
    s.n_upstream = 0;
    assert!(flat_plate_raw_mesh(&s).unwrap_err().to_string().contains("n_upstream"));
    let mut s = FlatPlateSpec::laminar();
    s.height = f64::NAN;
    assert!(s.validate().unwrap_err().to_string().contains("height"));
    assert!(FlatPlateSpec::with_resolution(150, 80, 4).unwrap_err().to_string().contains("nz must be 1"));
    assert!(FlatPlateSpec::with_resolution(1, 80, 1).unwrap_err().to_string().contains("n_plate"));
    let s = FlatPlateSpec::with_resolution(60, 40, 1).expect("resolution");
    assert_eq!((s.n_upstream, s.n_plate, s.n_y), (12, 48, 40));
}

#[test]
fn the_flat_plate_case_writes_a_laminar_case_with_a_symmetry_floor() {
    let dir = std::env::temp_dir().join(format!("ofgpu_flat_plate_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    write_case(&dir, CaseKind::FlatPlate, 20, 10, 1).expect("write the plate case");

    let raw = ofgpu::io::polymesh::read_poly_mesh(&dir).expect("read back");
    let names: Vec<&str> = raw.patches.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, FLAT_PLATE_PATCH_NAMES.to_vec());
    assert_eq!(raw.patches[2].size, 4, "a fifth of nx = 20 ahead of the edge");
    let u = std::fs::read_to_string(dir.join("0").join("U")).expect("0/U");
    assert!(u.contains("symmetryPlane"), "0/U must carry symmetryPlane on the floor");
    assert!(u.contains("noSlip"), "0/U must carry noSlip on the plate");
    let p = std::fs::read_to_string(dir.join("0").join("p")).expect("0/p");
    let top = p.find("top").expect("0/p names the top");
    assert!(p[top..top + 80].contains("fixedValue"), "the top is open at p = 0:\n{}", &p[top..]);
    let mt = std::fs::read_to_string(dir.join("constant").join("momentumTransport")).expect("mt");
    assert!(mt.contains("laminar"), "{mt}");
    let props = std::fs::read_to_string(dir.join("constant").join("physicalProperties")).expect("pp");
    assert!(props.contains("5e-06"), "{props}");

    // The in-memory twin builds the same mesh.
    let (mesh, fields) = build_case(CaseKind::FlatPlate, 20, 10, 1).expect("build_case");
    assert_eq!(mesh.n_cells, 200);
    assert_eq!(fields.vectors[0].internal[0].x as f64, 1.0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// `cases/flatPlateLaminar.jsonc` is the same plate on the JSONC reader's
/// x-uniform mesh: the same patch names and box, the edge on a node, the 30
/// floor columns ahead of it a symmetry plane (class D).
#[test]
fn the_shipped_jsonc_plate_puts_the_edge_on_a_node() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../cases/flatPlateLaminar.jsonc");
    let low = ofgpu::io::case_json::read_case_jsonc(&path).expect("read").lower().expect("lower");
    let raw = ofgpu::blockgen::raw_mesh(&low.block).expect("mesh");
    let names: Vec<&str> = raw.patches.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, FLAT_PLATE_PATCH_NAMES.to_vec());
    let sizes: Vec<usize> = raw.patches.iter().map(|p| p.size).collect();
    assert_eq!(sizes, vec![80, 80, 30, 120, 150, 12_000, 12_000]);
    assert_eq!(raw.patches[2].kind, PatchKind::Symmetry);
    let n_internal = raw.neighbour.len();
    let last_sym = n_internal + raw.patches[2].start + 29;
    let first_plate = n_internal + raw.patches[3].start;
    assert_eq!(face_x_range(&raw, last_sym).1, 0.0);
    assert_eq!(face_x_range(&raw, first_plate).0, 0.0);
}
