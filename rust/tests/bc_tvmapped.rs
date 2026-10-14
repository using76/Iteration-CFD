// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 120.11 - `timeVaryingMappedFixedValue` through the public API,
//! in BOTH builds: a linear field sampled at 200 scattered points of
//! `constant/boundaryData/<patch>` reproduced on the patch's faces to 1e-12
//! (barycentric interpolation is exact for linear data), the triangulation
//! itself (Delaunay's empty circle, the triangle count, a cocircular grid, a
//! tilted plane, collinear samples, the hull clamp, `nearest`), time
//! interpolation exact at the midpoints with the device bit for bit its host
//! twin, and the refusals by name. Everything the gates measure is f64 in
//! both builds (the face slabs are f64; the triple rounds once). A machine
//! without a card passes the GPU test vacuously, the convention of
//! `tests/f32_boussinesq.rs`.
//!
//! No GPL-licensed source was consulted.

use std::path::{Path, PathBuf};

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::field::tvmapped::{delaunay, weights, MapMethod};
use ofgpu::field_setup::{
    scalar_boundary_timed, scalar_boundary_with, setup_scalar_field_timed,
    setup_vector_field_timed, vector_boundary_timed, BcInputs, F1Setup,
};
use ofgpu::function1::{F1Clock, F1Kernels};
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::types::DVec3;
use ofgpu::{Gpu, GpuMesh, GpuScalarField, GpuVectorField, HostMesh, Scalar, Vec3};

/// A deterministic uniform draw on (0, 1): a 64-bit LCG (Knuth's MMIX
/// constants), the top 53 bits.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

/// A uniform block, the six patches `patch`.
fn box_mesh(n: [usize; 3]) -> HostMesh {
    let axis = |n: usize| GradedAxis {
        lo: 0.0,
        hi: 1.0,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    blockgen::build_mesh(&BlockSpec {
        x: axis(n[0]),
        y: axis(n[1]),
        z: axis(n[2]),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["patch"; 6].map(String::from),
        cyclic: Vec::new(),
    })
    .expect("mesh")
}

/// The WORLD f64 face centres of a patch, as the setup reads them.
fn centres(hm: &HostMesh, patch: &str) -> (usize, Vec<DVec3>) {
    let p = hm.patches.iter().find(|p| p.name == patch).expect("patch");
    let g = hm.geom64();
    (p.start, (0..p.size).map(|i| hm.world64(g.b_cf[p.start + i])).collect())
}

/// A fresh case folder in the temp directory.
fn case_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ofgpu-bc15-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("case dir");
    d
}

fn bd(case: &Path) -> PathBuf {
    case.join("constant").join("boundaryData").join("xMin")
}

fn header(class: &str) -> String {
    format!("FoamFile\n{{\n    format ascii;\n    class {class};\n    object values;\n}}\n")
}

fn write_points(case: &Path, p: &[DVec3]) {
    std::fs::create_dir_all(bd(case)).unwrap();
    let mut s = header("vectorField");
    s += &format!("{}\n(\n", p.len());
    for q in p {
        s += &format!("({:e} {:e} {:e})\n", q.x, q.y, q.z);
    }
    s += ")\n";
    std::fs::write(bd(case).join("points"), s).unwrap();
}

/// A scalar value file with an average line first, as such files carry one.
fn write_scalars(case: &Path, time: &str, field: &str, v: &[f64]) {
    let d = bd(case).join(time);
    std::fs::create_dir_all(&d).unwrap();
    let mut s = header("scalarAverageField");
    s += "// Average\n0\n\n// Data on points\n";
    s += &format!("{}\n(\n", v.len());
    for x in v {
        s += &format!("{x:e}\n");
    }
    s += ")\n";
    std::fs::write(d.join(field), s).unwrap();
}

fn write_vectors(case: &Path, time: &str, field: &str, v: &[[f64; 3]]) {
    let d = bd(case).join(time);
    std::fs::create_dir_all(&d).unwrap();
    let mut s = header("vectorAverageField");
    s += "(0 0 0)\n";
    s += &format!("{}\n(\n", v.len());
    for x in v {
        s += &format!("({:e} {:e} {:e})\n", x[0], x[1], x[2]);
    }
    s += ")\n";
    std::fs::write(d.join(field), s).unwrap();
}

/// 4 corners of xMin plus 196 scattered interior points: 200 samples.
fn samples_200() -> Vec<DVec3> {
    let mut p = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(0.0, 1.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
        DVec3::new(0.0, 1.0, 1.0),
    ];
    let mut r = Lcg(20261014);
    while p.len() < 200 {
        p.push(DVec3::new(0.0, r.next(), r.next()));
    }
    p
}

fn phi(x: DVec3) -> f64 {
    1.5 + 3.0 * x.y - 2.0 * x.z
}

fn uvec(x: DVec3) -> [f64; 3] {
    [1.0 + 0.5 * x.y, 2.0 - x.z, 0.25 * x.y + 0.75 * x.z]
}

fn tv(extra: &[(&str, &str)]) -> PatchFieldSpec {
    let mut s = PatchFieldSpec {
        type_name: "timeVaryingMappedFixedValue".into(),
        ..Default::default()
    };
    for (k, v) in extra {
        s.extra.insert((*k).into(), (*v).into());
    }
    s
}

fn zg() -> PatchFieldSpec {
    PatchFieldSpec {
        type_name: "zeroGradient".into(),
        ..Default::default()
    }
}

fn raw_s(inlet: PatchFieldSpec) -> RawScalarField {
    let mut raw = RawScalarField {
        name: "T".into(),
        ..Default::default()
    };
    for p in ["xMin", "xMax", "yMin", "yMax", "zMin", "zMax"] {
        raw.boundary.insert(p.into(), if p == "xMin" { inlet.clone() } else { zg() });
    }
    raw.internal = vec![0.0];
    raw
}

fn raw_v(inlet: PatchFieldSpec) -> RawVectorField {
    let mut raw = RawVectorField {
        name: "U".into(),
        ..Default::default()
    };
    for p in ["xMin", "xMax", "yMin", "yMax", "zMin", "zMax"] {
        raw.boundary.insert(p.into(), if p == "xMin" { inlet.clone() } else { zg() });
    }
    raw.internal = vec![Vec3::default()];
    raw
}

fn run(case: &Path, t0: f64, t1: f64) -> F1Setup<'_> {
    F1Setup {
        t_start: t0,
        t_end: t1,
        case_dir: Some(case),
    }
}

/// THE GATE (space): a linear scalar and a linear vector sampled at 200
/// scattered points are the exact linear fields on all 30 faces to 1e-12.
#[test]
fn a_linear_field_from_200_scattered_points_is_exact_on_the_faces() {
    let hm = box_mesh([2, 6, 5]);
    let (start, xf) = centres(&hm, "xMin");
    let p = samples_200();
    let case = case_dir("space");
    write_points(&case, &p);
    write_scalars(&case, "0", "T", &p.iter().map(|q| phi(*q)).collect::<Vec<_>>());
    write_vectors(&case, "0", "U", &p.iter().map(|q| uvec(*q)).collect::<Vec<_>>());

    let internal = vec![0.0 as Scalar; hm.n_cells];
    let (b, timed) =
        scalar_boundary_timed(&raw_s(tv(&[])), &hm, &internal, BcInputs::default(), &run(&case, 0.0, 1.0))
            .expect("scalar setup");
    assert!(timed.is_empty());
    assert_eq!(b.tv_rules.len(), 1);
    let v = b.tv_rules[0].values64(0.0);
    let scale = 4.5;
    let mut worst = 0.0f64;
    for (i, x) in xf.iter().enumerate() {
        let e = (v[i] - phi(*x)).abs() / scale;
        worst = worst.max(e);
        assert_eq!(b.ref_value[start + i], v[i] as Scalar, "the triple rounds once");
        assert_eq!(b.fr[start + i], 1.0);
    }
    println!("  [120.11] scalar: worst |phi_f - phi(x_f)|/max|phi| = {worst:.3e} over {} faces", xf.len());
    assert!(worst <= 1e-12, "scalar {worst:e}");

    let internal = vec![Vec3::default(); hm.n_cells];
    let (b, _) =
        vector_boundary_timed(&raw_v(tv(&[])), &hm, &internal, BcInputs::default(), &run(&case, 0.0, 1.0))
            .expect("vector setup");
    let v = b.tv_rules[0].values64(0.0);
    let mut worst = 0.0f64;
    for (i, x) in xf.iter().enumerate() {
        let u = uvec(*x);
        for c in 0..3 {
            worst = worst.max((v[3 * i + c] - u[c]).abs() / 2.0);
        }
        let r = b.ref_value[start + i];
        assert_eq!([r.x, r.y, r.z], [v[3 * i] as Scalar, v[3 * i + 1] as Scalar, v[3 * i + 2] as Scalar]);
    }
    println!("  [120.11] vector: worst component error/max|U| = {worst:.3e}");
    assert!(worst <= 1e-12, "vector {worst:e}");
    let _ = std::fs::remove_dir_all(&case);
}

/// The triangulation (120.11b): Delaunay's empty circle and the count
/// `T = 2N - 2 - h` on the 200 samples; a cocircular 11 x 11 grid on a
/// tilted plane, collinear samples, the hull clamp and `nearest`.
#[test]
fn the_triangulation_is_delaunay_and_exact_on_linear_data() {
    // 200 samples in the (y, z) plane: the scaled coordinates are an affine
    // image, so the empty-circle test on (y, z) is the test on (u, v).
    let p = samples_200();
    let uv: Vec<[f64; 2]> = p.iter().map(|q| [q.y, q.z]).collect();
    let tris = delaunay(&uv).expect("delaunay");
    let mut worst = 0.0f64;
    for t in &tris {
        let (a, b, c) = (uv[t[0]], uv[t[1]], uv[t[2]]);
        for (k, q) in uv.iter().enumerate() {
            if t.contains(&k) {
                continue;
            }
            // The in-circle determinant, positive when q is inside.
            let m = |r: [f64; 2]| [r[0] - q[0], r[1] - q[1], (r[0] - q[0]).powi(2) + (r[1] - q[1]).powi(2)];
            let (x, y, z) = (m(a), m(b), m(c));
            let det = x[0] * (y[1] * z[2] - y[2] * z[1]) - x[1] * (y[0] * z[2] - y[2] * z[0])
                + x[2] * (y[0] * z[1] - y[1] * z[0]);
            worst = worst.max(det);
        }
    }
    // The hull of 4 corners and 196 interior points has h = 4.
    assert_eq!(tris.len(), 2 * 200 - 2 - 4, "the triangle count");
    println!("  [120.11] 200 samples: {} triangles, worst in-circle determinant {worst:.3e}", tris.len());
    assert!(worst <= 1e-12, "a sample inside a circumcircle: {worst:e}");

    // A 30 x 30 grid, every cell's four corners cocircular: still a valid
    // triangulation of all 900 points, 116 of them on the hull.
    let g30: Vec<[f64; 2]> = (0..900).map(|k| [(k / 30) as f64 / 29.0, (k % 30) as f64 / 29.0]).collect();
    let t30 = delaunay(&g30).expect("grid delaunay");
    assert_eq!(t30.len(), 2 * 900 - 2 - 116, "the 30 x 30 grid's triangle count");

    // A tilted plane through a cocircular grid.
    let o = DVec3::new(0.3, -1.2, 2.0);
    let a = DVec3::new(1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0);
    let b = DVec3::new(2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0);
    let lin = |x: DVec3| 0.7 + x.x - 2.0 * x.y + 0.5 * x.z;
    let mut grid = Vec::new();
    for i in 0..11 {
        for j in 0..11 {
            grid.push(o + a * (2.0 * i as f64 / 10.0) + b * (3.0 * j as f64 / 10.0));
        }
    }
    let mut r = Lcg(7);
    let targets: Vec<DVec3> = (0..50).map(|_| o + a * (2.0 * r.next()) + b * (3.0 * r.next())).collect();
    let w = weights(&grid, &targets, MapMethod::Planar, "T", "xMin").expect("grid weights");
    let data: Vec<f64> = grid.iter().map(|q| lin(*q)).collect();
    let v = ofgpu::field::tvmapped::apply_weights(&w, &data, 1);
    let mut worst = 0.0f64;
    for (k, x) in targets.iter().enumerate() {
        assert_eq!(w[k].n, 3, "an interior target takes a triangle");
        worst = worst.max((v[k] - lin(*x)).abs());
    }
    println!("  [120.11] tilted 11x11 grid: worst error {worst:.3e}");
    assert!(worst <= 1e-12 * 4.0, "tilted grid {worst:e}");

    // Collinear samples: 1-D, clamped at the ends.
    let line: Vec<DVec3> = [0.0, 0.1, 0.35, 0.7, 1.0]
        .iter()
        .map(|&s| DVec3::new(1.0 + s, 2.0 * s, -s))
        .collect();
    let at = |s: f64| DVec3::new(1.0 + s, 2.0 * s, -s);
    let w = weights(&line, &[at(0.5), at(-0.2), at(1.3)], MapMethod::Planar, "T", "xMin").unwrap();
    let data: Vec<f64> = line.iter().map(|q| 3.0 * q.x - q.z).collect();
    let v = ofgpu::field::tvmapped::apply_weights(&w, &data, 1);
    assert!((v[0] - (3.0 * 1.5 + 0.5)).abs() <= 1e-14, "collinear interior {}", v[0]);
    assert_eq!(v[1], data[0], "clamped below");
    assert_eq!(v[2], data[4], "clamped above");

    // Outside the hull: the nearest point of the nearest hull edge.
    let sq: Vec<DVec3> = (0..5)
        .flat_map(|i| (0..5).map(move |j| DVec3::new(0.0, i as f64 / 4.0, j as f64 / 4.0)))
        .collect();
    let w = weights(&sq, &[DVec3::new(0.0, 1.5, 0.3)], MapMethod::Planar, "T", "xMin").unwrap();
    let data: Vec<f64> = sq.iter().map(|q| phi(*q)).collect();
    let v = ofgpu::field::tvmapped::apply_weights(&w, &data, 1);
    let e = phi(DVec3::new(0.0, 1.0, 0.3));
    assert!((v[0] - e).abs() <= 1e-14, "the hull clamp {} vs {e}", v[0]);

    // nearest: the closest sample's value, bit for bit.
    let w = weights(&sq, &[DVec3::new(0.01, 0.26, 0.74)], MapMethod::Nearest, "T", "xMin").unwrap();
    assert_eq!((w[0].n, w[0].idx[0], w[0].w[0]), (1, 5 + 3, 1.0));
}

/// THE GATE (time): folders 0, 0.5 and 2 - at t = 0.25 and 1.25 the value is
/// the midpoint of its two folders, at each folder time it is that folder,
/// and on the device each clock step is the host twin bit for bit.
#[test]
fn time_interpolation_is_exact_at_the_midpoints() {
    let hm = box_mesh([2, 6, 5]);
    let (start, _) = centres(&hm, "xMin");
    let p = samples_200();
    let case = case_dir("time");
    write_points(&case, &p);
    let times = ["0", "0.5", "2"];
    for (k, t) in times.iter().enumerate() {
        let s = 1.0 + k as f64;
        write_scalars(&case, t, "T", &p.iter().map(|q| s * phi(*q) + 0.1 * k as f64).collect::<Vec<_>>());
        write_vectors(
            &case,
            t,
            "U",
            &p.iter().map(|q| uvec(*q).map(|u| s * u - 0.3 * k as f64)).collect::<Vec<_>>(),
        );
    }
    let internal = vec![0.0 as Scalar; hm.n_cells];
    let (b, _) =
        scalar_boundary_timed(&raw_s(tv(&[])), &hm, &internal, BcInputs::default(), &run(&case, 0.0, 2.0))
            .unwrap();
    let r = &b.tv_rules[0];
    assert_eq!(r.times, vec![0.0, 0.5, 2.0]);
    let m = r.n * r.n_t;
    for (k, &t) in r.times.iter().enumerate() {
        assert_eq!(r.values64(t), r.values[k * m..(k + 1) * m].to_vec(), "folder {k} at its time");
    }
    let mut worst = 0.0f64;
    for (j, t) in [(0usize, 0.25), (1, 1.25)] {
        let v = r.values64(t);
        for e in 0..m {
            let (a, c) = (r.values[j * m + e], r.values[(j + 1) * m + e]);
            let mid = 0.5 * (a + c);
            worst = worst.max((v[e] - mid).abs() / a.abs().max(c.abs()));
        }
    }
    println!("  [120.11] midpoints: worst |v - (V_j + V_j+1)/2| relative {worst:.3e}");
    assert!(worst <= 2.0 * f64::EPSILON, "midpoint {worst:e}");

    let Some(gpu) = Gpu::new(0).ok() else {
        eprintln!("no GPU: the device half of the gate is skipped");
        return;
    };
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let k = F1Kernels::new(&gpu).unwrap();
    let mut fs = GpuScalarField::zeros(&gpu, &gm, "T").unwrap();
    let ts = setup_scalar_field_timed(&gpu, &mut fs, &raw_s(tv(&[])), &hm, BcInputs::default(), &run(&case, 0.0, 2.0))
        .unwrap();
    let mut fv = GpuVectorField::zeros(&gpu, &gm, "U").unwrap();
    let tvv = setup_vector_field_timed(&gpu, &mut fv, &raw_v(tv(&[])), &hm, BcInputs::default(), &run(&case, 0.0, 2.0))
        .unwrap();
    assert_eq!((ts.len(), tvv.len()), (1, 1));
    let internal = vec![Vec3::default(); hm.n_cells];
    let (bv, _) =
        vector_boundary_timed(&raw_v(tv(&[])), &hm, &internal, BcInputs::default(), &run(&case, 0.0, 2.0))
            .unwrap();
    let rv = &bv.tv_rules[0];
    let mut clock = F1Clock::new(&gpu, 0.0, 0.125).unwrap();
    let mut steps = 0;
    for _ in 0..16 {
        clock.advance(&gpu, &k).unwrap();
        ts.apply_scalar(&gpu, &k, &clock, &mut fs).unwrap();
        tvv.apply_vector(&gpu, &k, &clock, &mut fv).unwrap();
        let (t, _) = clock.read(&gpu).unwrap();
        let ds = gpu.download(&fs.ref_value).unwrap();
        let hs = r.host_values(t);
        for i in 0..r.n {
            assert_eq!(ds[start + i].to_bits(), hs[i].to_bits(), "scalar face {i} at t = {t}");
        }
        let dv = gpu.download(&fv.ref_value).unwrap();
        let hv = rv.host_values(t);
        for i in 0..rv.n {
            let d = dv[start + i];
            assert_eq!(
                [d.x.to_bits(), d.y.to_bits(), d.z.to_bits()],
                [hv[3 * i].to_bits(), hv[3 * i + 1].to_bits(), hv[3 * i + 2].to_bits()],
                "vector face {i} at t = {t}"
            );
        }
        steps += 1;
    }
    let (t, _) = clock.read(&gpu).unwrap();
    println!("  [120.11] device = host twin bitwise over {steps} steps to t = {t}");
    assert_eq!(t, 2.0);
    let _ = std::fs::remove_dir_all(&case);
}

/// Every refusal names the condition and SPEC-LIT 120.11.
#[test]
fn time_varying_mapped_refusals_are_by_name() {
    let hm = box_mesh([2, 3, 3]);
    let p = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(0.0, 1.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
        DVec3::new(0.0, 1.0, 1.0),
    ];
    let case = case_dir("refuse");
    write_points(&case, &p);
    write_scalars(&case, "0", "T", &[1.0, 2.0, 3.0, 4.0]);
    write_scalars(&case, "1", "T", &[1.0, 2.0, 3.0, 4.0]);
    let internal = vec![0.0 as Scalar; hm.n_cells];
    let msg = |r: ofgpu::Result<()>| -> String {
        let e = r.expect_err("must be refused").to_string();
        assert!(e.contains("120.11"), "{e}");
        e
    };
    let timed = |spec: PatchFieldSpec, c: &Path, t0: f64, t1: f64| {
        scalar_boundary_timed(&raw_s(spec), &hm, &internal, BcInputs::default(), &run(c, t0, t1)).map(|_| ())
    };

    let e = msg(scalar_boundary_with(&raw_s(tv(&[])), &hm, &internal, BcInputs::default()).map(|_| ()));
    assert!(e.contains("timeVaryingMappedFixedValue") && e.contains("setup_scalar_field_timed"), "{e}");
    let no_dir = F1Setup { t_start: 0.0, t_end: 1.0, case_dir: None };
    let e = msg(scalar_boundary_timed(&raw_s(tv(&[])), &hm, &internal, BcInputs::default(), &no_dir).map(|_| ()));
    assert!(e.contains("case directory"), "{e}");
    let e = msg(timed(tv(&[("setAverage", "true")]), &case, 0.0, 1.0));
    assert!(e.contains("setAverage"), "{e}");
    let e = msg(timed(tv(&[("offset", "constant 1")]), &case, 0.0, 1.0));
    assert!(e.contains("offset"), "{e}");
    let e = msg(timed(tv(&[("mapMethod", "inverseDistance")]), &case, 0.0, 1.0));
    assert!(e.contains("mapMethod"), "{e}");
    let e = msg(timed(tv(&[]), &case, 0.0, 1.5));
    assert!(e.contains("not inside the database's times"), "{e}");
    let e = msg(timed(tv(&[("fieldTable", "Tother")]), &case, 0.0, 1.0));
    assert!(e.contains("Tother"), "{e}");

    // A value count that is not the point count.
    write_scalars(&case, "1", "T", &[1.0, 2.0, 3.0]);
    let e = msg(timed(tv(&[]), &case, 0.0, 1.0));
    assert!(e.contains("3 values"), "{e}");
    write_scalars(&case, "1", "T", &[1.0, 2.0, 3.0, 4.0]);
    // Two folders naming one time.
    write_scalars(&case, "1.0", "T", &[1.0, 2.0, 3.0, 4.0]);
    let e = msg(timed(tv(&[]), &case, 0.0, 1.0));
    assert!(e.contains("the same time"), "{e}");
    std::fs::remove_dir_all(bd(&case).join("1.0")).unwrap();
    // A repeated sample point.
    let mut dup = p.clone();
    dup[3] = dup[1];
    write_points(&case, &dup);
    let e = msg(timed(tv(&[]), &case, 0.0, 1.0));
    assert!(e.contains("same point"), "{e}");
    // No points file.
    std::fs::remove_file(bd(&case).join("points")).unwrap();
    let e = timed(tv(&[]), &case, 0.0, 1.0).expect_err("no points").to_string();
    assert!(e.contains("points"), "{e}");
    let _ = std::fs::remove_dir_all(&case);
}
