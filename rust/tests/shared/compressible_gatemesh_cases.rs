// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.
//
// CMP-26's both-builds gate, written ONCE and included twice - by
// `tests/compressible_gatemesh.rs`, which runs it as an integration test in
// BOTH builds, and by the lib tests of `src/compressible/gatemesh.rs`,
// where the same names sit beside the code. The first ten tests are
// SPEC-LIT §127.5.1's gate table, verbatim: volumes, per-cell closure, the
// 129 stations on the wall, the non-orthogonality bar, the topology
// counts, the `y+` first interval, the one-sided distribution, the parser,
// the spec refusals, and the two builds' identical points. CMP-21 appends
// §127.5.2's six duct tests: the area ratio, the topology counts, the face
// areas, the trapezoid volumes, closure and the axis centres, and the
// duct spec's refusals.
//
// An integration test's body cannot carry `//!` or `#![..]`, and neither
// can this file - it is included, not compiled standalone. Every
// comparison is made in f64 from the downloaded `Scalar` values; the
// R-class tolerances go through `tol_r` (docs/17 §5.1, D-F32-1), written
// once and never as a typed f32 number.
//
// Provenance: ORIGINAL - the meshes are this tree's own construction from
// the answer key `rae2822-coordinates`; the pipeline they exercise is
// §127.5.1's (Thompson, Warsi & Mastin 1985, chs. VI and VIII, cited
// there). No GPL-licensed source was consulted.

use std::sync::OnceLock;

use ofgpu::compressible::gatemesh::{
    airfoil_cmesh, duct_area_ratio, first_cell_for_y_plus, nozzle_duct, one_sided_tanh,
    parse_stations, rae2822_stations, DuctSpec, CMesh, CMeshSpec,
};
use ofgpu::io::polymesh::build_host_mesh;
use ofgpu::{DVec3, HostMesh, PatchKind, Scalar};

/// The build's `Scalar` widened to the f64 the answer keys are computed in:
/// a no-op in the f64 build, the exact widening under `single`.
#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    x as f64
}

/// The R-class tolerance (D-F32-1): `tol64` in the default build; under
/// `single`, `max(tol64, min(tol64 * 2^29, 1e-5))`. Written once, never as
/// a typed f32 number.
fn tol_r(tol64: f64) -> f64 {
    if cfg!(feature = "single") {
        tol64.max((tol64 * 2f64.powi(29)).min(1.0e-5))
    } else {
        tol64
    }
}

/// One mesh, built once: the C-mesh and the host mesh on it.
fn built(spec: &CMeshSpec) -> (CMesh, HostMesh) {
    let st = rae2822_stations().expect("parse the RAE 2822 stations");
    let m = airfoil_cmesh(&st, spec).expect("build the C-mesh");
    let h = build_host_mesh(&m.raw).expect("build the host mesh");
    (m, h)
}

/// The default gate mesh and its host mesh, built once per process.
fn default_mesh() -> &'static (CMesh, HostMesh) {
    static M: OnceLock<(CMesh, HostMesh)> = OnceLock::new();
    M.get_or_init(|| built(&CMeshSpec::default()))
}

/// The coarse gate mesh and its host mesh, built once per process.
fn coarse_mesh() -> &'static (CMesh, HostMesh) {
    static M: OnceLock<(CMesh, HostMesh)> = OnceLock::new();
    M.get_or_init(|| built(&CMeshSpec::coarse()))
}

/// Both gate meshes by name.
fn both() -> [(&'static str, &'static (CMesh, HostMesh)); 2] {
    [("default", default_mesh()), ("coarse", coarse_mesh())]
}

/// The signed area of a closed polygon given once around.
fn shoelace(pts: &[[f64; 2]]) -> f64 {
    let n = pts.len();
    let mut s = 0.0;
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        s += a[0] * b[1] - b[0] * a[1];
    }
    0.5 * s
}

/// Every cell of both meshes has positive volume, and the volumes sum to
/// the span times the outer polygon's area less the aerofoil's, class R.
#[test]
fn no_negative_volume() {
    let span = 1.0; // both gate specs
    for (name, (m, h)) in both() {
        assert_eq!(h.n_cells, m.n_xi * m.n_eta, "{name}: the cell count");
        let mut total = 0.0;
        let mut vmin = f64::INFINITY;
        for &v in &h.v {
            let vv = w64(v);
            total += vv;
            vmin = vmin.min(vv);
        }
        println!("  {name}: min cell volume {vmin:.3e}");
        assert!(vmin > 0.0, "{name}: min cell volume {vmin} is not positive");
        let n = m.n_xi;
        let j = m.n_eta;
        let mut outer: Vec<[f64; 2]> = (0..=j).map(|jj| m.node(0, jj)).collect();
        outer.extend((1..=n).map(|ii| m.node(ii, j)));
        outer.extend((0..j).rev().map(|jj| m.node(n, jj)));
        let a_out = shoelace(&outer).abs();
        let foil: Vec<[f64; 2]> =
            (m.n_wake..=n - m.n_wake - 1).map(|ii| m.node(ii, 0)).collect();
        let a_foil = shoelace(&foil).abs();
        println!("  {name}: outer area {a_out:.6}, aerofoil area {a_foil:.6}");
        let want = span * (a_out - a_foil);
        assert!(
            (total - want).abs() <= tol_r(1.0e-9) * span * a_out,
            "{name}: the volumes sum to {total}, not the span times \
             {a_out:.9} - {a_foil:.9} = {want:.9}"
        );
    }
}

/// Each cell's outward face-area vectors close to round-off, class R.
#[test]
fn closure_round_off() {
    for (name, (m, h)) in both() {
        assert_eq!(h.n_cells, m.n_xi * m.n_eta, "{name}: the cell count");
        let mut acc = vec![[0.0f64; 3]; h.n_cells];
        let mut mag = vec![0.0f64; h.n_cells];
        let mut add = |c: usize, v: [f64; 3], s: f64| {
            acc[c][0] += s * v[0];
            acc[c][1] += s * v[1];
            acc[c][2] += s * v[2];
            mag[c] += s;
        };
        for f in 0..h.n_internal_faces {
            let sf = [w64(h.sf[f].x), w64(h.sf[f].y), w64(h.sf[f].z)];
            add(h.owner[f] as usize, sf, 1.0);
            add(h.neighbour[f] as usize, sf, -1.0);
        }
        for bf in 0..h.n_boundary_faces {
            let sf = [w64(h.b_sf[bf].x), w64(h.b_sf[bf].y), w64(h.b_sf[bf].z)];
            add(h.b_face_cells[bf] as usize, sf, 1.0);
        }
        let mut worst = 0.0f64;
        for c in 0..h.n_cells {
            let sum = (acc[c][0] * acc[c][0]
                + acc[c][1] * acc[c][1]
                + acc[c][2] * acc[c][2])
                .sqrt();
            let bar = tol_r(1.0e-12) * mag[c];
            worst = worst.max(if mag[c] > 0.0 { sum / mag[c] } else { 0.0 });
            assert!(
                sum <= bar,
                "{name}: cell {c}: the outward area vectors sum to {sum:.3e}, \
                 above {bar:.3e} (the faces' magnitudes sum to {})",
                mag[c]
            );
        }
        println!("  {name}: worst closure ratio {worst:.3e}");
    }
}

/// Each of the 129 stations sits on an `airfoil` point of the wall patch,
/// and the wall carries exactly the surface nodes: 256 distinct points for
/// the default mesh, 128 for the coarse one - the walk has one more slot,
/// but the two TE ends of the loop are ONE node through the wake cut.
#[test]
fn surface_points_reproduce_coordinates_1e_9() {
    let stations = rae2822_stations().expect("parse the RAE 2822 stations");
    let loop_pts = stations.loop_points();
    assert_eq!(loop_pts.len(), 129);
    for (name, (m, h)) in both() {
        let want_distinct = m.n_surface;
        let patch = m
            .raw
            .patches
            .iter()
            .find(|p| p.name == "airfoil")
            .expect("the airfoil patch");
        let mut ids: Vec<usize> = Vec::new();
        for bf in patch.start..patch.start + patch.size {
            let face = &m.raw.faces[h.n_internal_faces + bf];
            for &v in face {
                if m.raw.points[v as usize].z == 0.0 {
                    ids.push(v as usize);
                }
            }
        }
        ids.sort_unstable();
        ids.dedup();
        println!("  {name}: {} distinct z = 0 airfoil points", ids.len());
        assert_eq!(ids.len(), want_distinct, "{name}: the distinct points");
        for k in &loop_pts {
            let hit = ids.iter().any(|&id| {
                let p = &m.raw.points[id];
                let dx = p.x - k[0];
                let dy = p.y - k[1];
                (dx * dx + dy * dy).sqrt() <= 1.0e-9
            });
            assert!(
                hit,
                "{name}: the station ({}, {}) is not an airfoil point",
                k[0], k[1]
            );
        }
    }
}

/// The worst internal-face non-orthogonality is below 60 degrees in both
/// builds - the same bar, measured on the widened f64 values.
#[test]
fn max_non_orthogonality_below_60deg() {
    for (name, (_m, h)) in both() {
        let mut worst = 0.0f64;
        for f in 0..h.n_internal_faces {
            let o = h.owner[f] as usize;
            let nb = h.neighbour[f] as usize;
            let dx = w64(h.c[nb].x) - w64(h.c[o].x);
            let dy = w64(h.c[nb].y) - w64(h.c[o].y);
            let sfx = w64(h.sf[f].x);
            let sfy = w64(h.sf[f].y);
            let dot = (sfx * dx + sfy * dy).abs();
            let smag = (sfx * sfx + sfy * sfy).sqrt();
            let dmag = (dx * dx + dy * dy).sqrt();
            let cos = (dot / (smag * dmag)).clamp(-1.0, 1.0);
            worst = worst.max(cos.acos().to_degrees());
        }
        println!("  {name}: max non-orthogonality {worst:.4} deg");
        assert!(
            worst < 60.0,
            "{name}: max non-orthogonality {worst} deg is not below 60"
        );
    }
}

/// The topology counts and the patch table, and the ldu order of the
/// internal faces, and every cell named by exactly 6 faces.
#[test]
fn topology_counts_and_patches() {
    let cases: [(&'static str, &'static (CMesh, HostMesh), usize, usize, usize, usize, usize, [usize; 4]); 2] = [
        ("default", default_mesh(), 352, 96, 256, 68384, 33792, [256, 352, 192, 67584]),
        ("coarse", coarse_mesh(), 176, 32, 128, 11632, 5632, [128, 176, 64, 11264]),
    ];
    for (name, (m, h), n, j, ns, points, cells, sizes) in cases {
        assert_eq!(m.n_xi, n, "{name}: N");
        assert_eq!(m.n_eta, j, "{name}: J");
        assert_eq!(m.n_wake, n / 2 - ns / 2, "{name}: the wake columns");
        assert_eq!(m.n_surface, ns, "{name}: the surface intervals");
        assert_eq!(m.raw.points.len(), points, "{name}: the point count");
        assert_eq!(h.n_points, points, "{name}: the host point count");
        assert_eq!(h.n_cells, cells, "{name}: the cell count");
        assert_eq!(h.n_internal_faces, 2 * cells - n - j + m.n_wake, "{name}: internal faces");
        let names = ["airfoil", "farfield", "outlet", "frontAndBack"];
        let types = ["wall", "patch", "patch", "empty"];
        let kinds = [PatchKind::Wall, PatchKind::Generic, PatchKind::Generic, PatchKind::Empty];
        let mut at = 0usize;
        for (k, p) in h.patches.iter().enumerate() {
            assert_eq!(p.name, names[k], "{name}: patch {k} name");
            assert_eq!(p.type_name, types[k], "{name}: patch {k} type");
            assert_eq!(p.kind, kinds[k], "{name}: patch {k} kind");
            assert_eq!(p.start, at, "{name}: patch {k} start");
            assert_eq!(p.size, sizes[k], "{name}: patch {k} size");
            assert_eq!(p.nbr_patch, None, "{name}: patch {k} couple");
            at += p.size;
        }
        assert_eq!(at, h.n_boundary_faces, "{name}: the boundary face count");
        for f in 0..h.n_internal_faces {
            assert!(h.owner[f] < h.neighbour[f], "{name}: face {f} not upper-triangular");
            if f > 0 {
                assert!(
                    h.owner[f] >= h.owner[f - 1],
                    "{name}: face {f}: owners not non-decreasing"
                );
                if h.owner[f] == h.owner[f - 1] {
                    assert!(
                        h.neighbour[f] > h.neighbour[f - 1],
                        "{name}: face {f}: neighbours not increasing within owner {}",
                        h.owner[f]
                    );
                }
            }
        }
        for c in 0..h.n_cells {
            let nf = (h.cf_offset[c + 1] - h.cf_offset[c]) + (h.bcf_offset[c + 1] - h.bcf_offset[c]);
            assert_eq!(nf, 6, "{name}: cell {c} is named by {nf} faces");
        }
        println!("  {name}: N {n}, J {j}, {} points, {} cells, {} internal faces", points, cells, h.n_internal_faces);
    }
}

/// Every `eta` line of the default mesh starts with the `y+` cell, and the
/// flat-plate estimate behind `first_cell_for_y_plus` holds three keys.
#[test]
fn first_interval_is_the_y_plus_cell() {
    let (m, _h) = default_mesh();
    let spec = CMeshSpec::default();
    let first = spec.first_cell;
    let mut worst = 0.0f64;
    for i in 0..=m.n_xi {
        let d = dist2(m.node(i, 0), m.node(i, 1));
        worst = worst.max((d - first).abs() / first);
    }
    println!("  default: worst first-interval error {worst:.3e}, first cell {first:.9e}");
    assert!(
        worst <= 1.0e-9,
        "the first interval strays from the first cell by {worst:.3e} relative"
    );
    for (y_plus, re_c, want) in [
        (1.0, 6.5e6, 3.833498871772944e-6),
        (30.0, 6.5e6, 1.1500496615318832e-4),
        (1.0, 1.0e5, 2.1823075758644266e-4),
    ] {
        let got = first_cell_for_y_plus(y_plus, re_c).expect("a finite y+ and re_c");
        println!("  first_cell_for_y_plus({y_plus}, {re_c:.1e}) = {got:.15e}");
        assert!(
            (got - want).abs() <= 1.0e-12 * want.abs(),
            "first_cell_for_y_plus({y_plus}, {re_c}): {got} is not {want}"
        );
    }
    for (y_plus, re_c) in [(0.0, 6.5e6), (1.0, 0.0), (f64::NAN, 1.0e6)] {
        assert!(
            first_cell_for_y_plus(y_plus, re_c).is_err(),
            "first_cell_for_y_plus({y_plus}, {re_c}) is not refused"
        );
    }
}

/// The distance between two 2-D points.
fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

/// The one-sided distribution holds its ends exactly, its first interval to
/// round-off, and its monotone rise; out-of-range fractions are refused.
#[test]
fn one_sided_tanh_holds_its_ends_and_first_interval() {
    let s = one_sided_tanh(10, 0.01).expect("f = 0.01 on 10 intervals");
    assert_eq!(s.len(), 11);
    assert_eq!(s[0], 0.0, "s_0 is 0 exactly");
    assert_eq!(s[10], 1.0, "s_10 is 1 exactly");
    println!("  n 10, f 0.01: s_1 = {:.17e}, a-rooted first interval", s[1]);
    assert!((s[1] - 0.01).abs() <= 1.0e-14, "s_1 {} is not 0.01", s[1]);
    for i in 1..s.len() {
        assert!(s[i] > s[i - 1], "s_{i} = {} does not ascend from {}", s[i], s[i - 1]);
    }
    let s = one_sided_tanh(96, 1.9e-7).expect("f = 1.9e-7 on 96 intervals");
    assert_eq!(s.len(), 97);
    println!("  n 96, f 1.9e-7: s_1 = {:.17e}", s[1]);
    assert!(
        (s[1] - 1.9e-7).abs() <= 1.0e-12 * 1.9e-7,
        "s_1 {} is not 1.9e-7",
        s[1]
    );
    for (n, f) in [(10usize, 0.1), (10, 0.2), (10, 0.0), (0, 0.01), (10, f64::NAN)] {
        assert!(
            one_sided_tanh(n, f).is_err(),
            "one_sided_tanh({n}, {f}) is not refused"
        );
    }
}

/// The parser takes the key and refuses the four malformed shapes.
#[test]
fn stations_parse_and_refuse() {
    let st = rae2822_stations().expect("parse the key");
    assert_eq!(st.x.len(), 65, "the key's row count");
    assert_eq!(st.x[1], 0.0006);
    assert_eq!(st.z_lower[1], 0.00317);
    assert_eq!(st.z_upper[1], 0.00323);
    let k = st.loop_points();
    assert_eq!(k.len(), 129);
    assert_eq!(k[0], [1.0, 0.0]);
    assert_eq!(k[1], [0.9994, -0.00001]);
    assert_eq!(k[64], [0.0, 0.0]);
    assert_eq!(k[127], [0.9994, 0.0003]);
    assert_eq!(k[128], [1.0, 0.0]);
    assert!(parse_stations("# c\nx,zl,zu\n0.0,0.0,0.0\n0.5,0.01,0.01\n1.0,0.0,0.0\n").is_err());
    assert!(parse_stations("x_c,z_c_lower,z_c_upper\n0.0,0.0,0.0\n0.5,0.1\n1.0,0.0,0.0\n").is_err());
    assert!(
        parse_stations("x_c,z_c_lower,z_c_upper\n0.0,0.0,0.0\n0.5,0.01,0.01\n0.4,0.02,0.02\n1.0,0.0,0.0\n").is_err()
    );
    assert!(
        parse_stations("x_c,z_c_lower,z_c_upper\n0.0,0.0,0.0\n0.5,0.01,0.01\n0.9,0.0,0.0\n").is_err()
    );
}

/// Out-of-bounds specs are refused, every message starting `gatemesh: `.
#[test]
fn spec_refusals() {
    let st = rae2822_stations().expect("parse the key");
    let refused = |spec: &CMeshSpec, why: &str| {
        let e = airfoil_cmesh(&st, spec).expect_err(why);
        let msg = match &e {
            ofgpu::Error::Mesh(m) => m.clone(),
            other => panic!("{why}: not a mesh error: {other:?}"),
        };
        assert!(
            msg.starts_with("gatemesh: "),
            "{why}: the refusal \"{msg}\" does not name the module"
        );
        println!("  {why}: \"{msg}\"");
    };
    let mut spec = CMeshSpec::coarse();
    spec.refine = 0;
    refused(&spec, "refine 0");
    let mut spec = CMeshSpec::coarse();
    spec.omega = 2.0;
    refused(&spec, "omega 2.0");
    let mut spec = CMeshSpec::coarse();
    spec.first_cell = 1.0;
    refused(&spec, "first_cell 1.0");
    let mut spec = CMeshSpec::coarse();
    spec.max_sweeps = 1;
    refused(&spec, "max_sweeps 1");
}

/// The point bits are the same in both builds: an FNV-1a 64 hash over every
/// raw point's `x`, `y`, `z` in LE byte order, pinned per mesh. The f64 and
/// single builds must print and assert the same two literals.
#[test]
fn points_are_the_same_bits_in_both_builds() {
    let hash = |points: &[DVec3]| {
        let mut h: u64 = 0xcbf29ce484222325;
        for p in points {
            for v in [p.x, p.y, p.z] {
                for b in v.to_bits().to_le_bytes() {
                    h ^= u64::from(b);
                    h = h.wrapping_mul(0x100000001b3);
                }
            }
        }
        h
    };
    let hd = hash(&default_mesh().0.raw.points);
    let hc = hash(&coarse_mesh().0.raw.points);
    println!("  default points hash {hd:016x}, coarse points hash {hc:016x}");
    assert_eq!(hd, 0xb5ca6cb8433c8b1du64, "the default mesh's point bits moved");
    assert_eq!(hc, 0x3f8d774e11d077a1u64, "the coarse mesh's point bits moved");
}

// ==========================================================================
//  CMP-21's duct, SPEC-LIT §127.5.2
// ==========================================================================

/// The default duct and its host mesh, built once per process.
fn duct_mesh() -> &'static (ofgpu::compressible::gatemesh::Duct, HostMesh) {
    static M: OnceLock<(ofgpu::compressible::gatemesh::Duct, HostMesh)> = OnceLock::new();
    M.get_or_init(|| {
        let d = nozzle_duct(&DuctSpec::default()).expect("build the duct");
        let h = build_host_mesh(&d.raw).expect("build the host mesh");
        (d, h)
    })
}

/// The area ratio of (127.22) holds its four key values bitwise: 2 at both
/// ends, 1 at the throat, 1.25 a quarter way in.
#[test]
fn duct_area_ratio_values() {
    assert_eq!(duct_area_ratio(0.0), 2.0);
    assert_eq!(duct_area_ratio(1.0), 2.0);
    assert_eq!(duct_area_ratio(0.5), 1.0);
    assert_eq!(duct_area_ratio(0.25), 1.25);
}

/// The duct's topology: 804 points, 200 cells, 199 internal faces, 1001
/// faces, and the four patches in order inlet/outlet/walls/frontAndBack.
#[test]
fn duct_topology_counts_and_patches() {
    let (d, h) = duct_mesh();
    let n = d.spec.n;
    assert_eq!(d.raw.points.len(), 4 * (n + 1), "the point count");
    assert_eq!(h.n_points, 4 * (n + 1), "the host point count");
    assert_eq!(h.n_cells, n, "the cell count");
    assert_eq!(h.n_internal_faces, n - 1, "the internal face count");
    assert_eq!(d.raw.faces.len(), 1001, "the total face count");
    assert_eq!(h.n_boundary_faces, 2 + 4 * n, "the boundary face count");
    let names = ["inlet", "outlet", "walls", "frontAndBack"];
    let types = ["patch", "patch", "symmetry", "symmetry"];
    let kinds = [PatchKind::Generic, PatchKind::Generic, PatchKind::Symmetry, PatchKind::Symmetry];
    let sizes = [1usize, 1, 2 * n, 2 * n];
    let mut at = 0usize;
    for (k, p) in h.patches.iter().enumerate() {
        assert_eq!(p.name, names[k], "patch {k} name");
        assert_eq!(p.type_name, types[k], "patch {k} type");
        assert_eq!(p.kind, kinds[k], "patch {k} kind");
        assert_eq!(p.start, at, "patch {k} start");
        assert_eq!(p.size, sizes[k], "patch {k} size");
        assert_eq!(p.nbr_patch, None, "patch {k} couple");
        at += p.size;
    }
    assert_eq!(at, h.n_boundary_faces, "the boundary face count");
    for f in 0..h.n_internal_faces {
        assert_eq!(h.owner[f] as usize, f, "face {f} owner");
        assert_eq!(h.neighbour[f] as usize, f + 1, "face {f} neighbour");
    }
    println!("  duct: {} points, {} cells, {} internal faces", h.n_points, h.n_cells, h.n_internal_faces);
}

/// Every internal face's area is `H_j D` along x, the inlet and outlet
/// faces' areas are `2 H_t D`, and the end faces' normals point the way
/// the flow needs: `Sf_x < 0` in, `Sf_x > 0` out.
#[test]
fn duct_face_areas_follow_the_area_ratio() {
    let (d, h) = duct_mesh();
    let spec = &d.spec;
    let tol = tol_r(1.0e-13);
    for j in 1..spec.n {
        let sf = [w64(h.sf[j - 1].x), w64(h.sf[j - 1].y), w64(h.sf[j - 1].z)];
        let mag = (sf[0] * sf[0] + sf[1] * sf[1] + sf[2] * sf[2]).sqrt();
        let want = d.height(j) * spec.depth;
        assert!(
            (mag - want).abs() <= tol * want,
            "internal face {j}: |Sf| {mag} is not H_j D = {want}"
        );
        assert!(sf[0] > 0.0, "internal face {j}: Sf_x {} is not positive", sf[0]);
        assert!(
            sf[1].abs() <= tol * mag && sf[2].abs() <= tol * mag,
            "internal face {j}: Sf_y, Sf_z = ({}, {}) are not round-off",
            sf[1], sf[2]
        );
    }
    let inlet = |sf: [f64; 3]| (sf[0] * sf[0] + sf[1] * sf[1] + sf[2] * sf[2]).sqrt();
    let end = 2.0 * spec.throat_height * spec.depth;
    let fi = [w64(h.b_sf[0].x), w64(h.b_sf[0].y), w64(h.b_sf[0].z)];
    let fo = [w64(h.b_sf[1].x), w64(h.b_sf[1].y), w64(h.b_sf[1].z)];
    assert!((inlet(fi) - end).abs() <= tol * end, "inlet |Sf| {}", inlet(fi));
    assert!(fi[0] < 0.0, "inlet Sf_x {} is not negative", fi[0]);
    assert!((inlet(fo) - end).abs() <= tol * end, "outlet |Sf| {}", inlet(fo));
    assert!(fo[0] > 0.0, "outlet Sf_x {} is not positive", fo[0]);
    println!("  duct: inlet |Sf| {:.6e}, outlet |Sf| {:.6e}", inlet(fi), inlet(fo));
}

/// Every cell is a trapezoid prism: its volume is
/// `(x_{i+1} - x_i) D (H_i + H_{i+1})/2`.
#[test]
fn duct_volumes_are_trapezoids() {
    let (d, h) = duct_mesh();
    let spec = &d.spec;
    let tol = tol_r(1.0e-12);
    let mut worst = 0.0f64;
    for i in 0..spec.n {
        let dx = d.node_x(i + 1) - d.node_x(i);
        let want = dx * spec.depth * 0.5 * (d.height(i) + d.height(i + 1));
        let got = w64(h.v[i]);
        let e = (got - want).abs() / want;
        worst = worst.max(e);
        assert!(e <= tol, "cell {i}: volume {got} is not the trapezoid's {want}");
    }
    println!("  duct: worst trapezoid-volume error {worst:.3e}");
}

/// Every cell closes to round-off, its centre lies on the axis at mid
/// depth, and the centres march strictly with `i`.
#[test]
fn duct_cells_close_and_centres_lie_on_the_axis() {
    let (d, h) = duct_mesh();
    let spec = &d.spec;
    let tol = tol_r(1.0e-13);
    let tol_c = tol_r(1.0e-12) * spec.throat_height;
    let mut acc = vec![[0.0f64; 3]; h.n_cells];
    let mut mag = vec![0.0f64; h.n_cells];
    let mut add = |c: usize, v: [f64; 3], s: f64| {
        acc[c][0] += s * v[0];
        acc[c][1] += s * v[1];
        acc[c][2] += s * v[2];
        mag[c] += s;
    };
    for f in 0..h.n_internal_faces {
        add(h.owner[f] as usize, [w64(h.sf[f].x), w64(h.sf[f].y), w64(h.sf[f].z)], 1.0);
        add(h.neighbour[f] as usize, [w64(h.sf[f].x), w64(h.sf[f].y), w64(h.sf[f].z)], -1.0);
    }
    for bf in 0..h.n_boundary_faces {
        add(h.b_face_cells[bf] as usize, [w64(h.b_sf[bf].x), w64(h.b_sf[bf].y), w64(h.b_sf[bf].z)], 1.0);
    }
    let mut worst = 0.0f64;
    let mut last_x = f64::NEG_INFINITY;
    for c in 0..h.n_cells {
        let sum = (acc[c][0] * acc[c][0] + acc[c][1] * acc[c][1] + acc[c][2] * acc[c][2]).sqrt();
        worst = worst.max(if mag[c] > 0.0 { sum / mag[c] } else { 0.0 });
        assert!(
            sum <= tol * mag[c],
            "cell {c}: the outward area vectors sum to {sum:.3e}, above {:.3e}",
            tol * mag[c]
        );
        let cy = w64(h.c[c].y);
        let cz = w64(h.c[c].z);
        assert!(cy.abs() <= tol_c, "cell {c}: centre y {cy} is not on the axis");
        assert!(
            (cz - 0.5 * spec.depth).abs() <= tol_c,
            "cell {c}: centre z {cz} is not mid depth"
        );
        let cx = w64(h.c[c].x);
        assert!(cx > last_x, "cell {c}: centre x {cx} does not advance from {last_x}");
        last_x = cx;
    }
    println!("  duct: worst closure ratio {worst:.3e}, last centre x {last_x:.6}");
}

/// Out-of-bounds duct specs are refused, every message naming the module
/// and the offending field.
#[test]
fn duct_spec_refusals() {
    let refused = |spec: &DuctSpec, why: &str, field: &str| {
        let e = nozzle_duct(spec).expect_err(why);
        let msg = match &e {
            ofgpu::Error::Mesh(m) => m.clone(),
            other => panic!("{why}: not a mesh error: {other:?}"),
        };
        assert!(
            msg.contains("gatemesh: nozzle_duct"),
            "{why}: the refusal \"{msg}\" does not name the module"
        );
        assert!(msg.contains(field), "{why}: the refusal \"{msg}\" does not name {field}");
        println!("  {why}: \"{msg}\"");
    };
    let mut spec = DuctSpec::default();
    spec.n = 1;
    refused(&spec, "n 1", "n");
    let mut spec = DuctSpec::default();
    spec.length = 0.0;
    refused(&spec, "length 0", "length");
    let mut spec = DuctSpec::default();
    spec.length = -1.0;
    refused(&spec, "length -1", "length");
    let mut spec = DuctSpec::default();
    spec.length = f64::NAN;
    refused(&spec, "length NaN", "length");
    let mut spec = DuctSpec::default();
    spec.throat_height = 0.0;
    refused(&spec, "throat_height 0", "throat_height");
    let mut spec = DuctSpec::default();
    spec.depth = f64::INFINITY;
    refused(&spec, "depth +inf", "depth");
}
