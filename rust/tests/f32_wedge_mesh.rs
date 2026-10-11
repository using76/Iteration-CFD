// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §119.0 - the wedge-mesh gates of BC-02 re-run in BOTH builds
//! through the public API, the same assertions and the same 1e-12
//! tolerances as the lib tests `blockgen::wedge::tests`. Precision class A:
//! the volume, area and normal identities are evaluated in f64 (the points
//! are f64 in both builds, the nodes widened once from `Scalar`), so the
//! gates hold to the same tolerance whichever build runs. Precision class
//! D: the front/back mirror bits and the counts. No
//! `#[cfg_attr(feature = "single", ignore)]` anywhere - the f64 build has
//! no monopoly on any of these identities.

use ofgpu::blockgen::{
    build_wedge_mesh, raw_wedge_mesh, wedge_half_tan, wedge_nodes, GradedAxis, WedgeSpec,
    WEDGE_PATCH_NAMES,
};
use ofgpu::Scalar;

fn spec_of(r: GradedAxis, z: GradedAxis, theta: f64) -> WedgeSpec {
    WedgeSpec { r, z, theta_deg: theta, patch_type: WedgeSpec::default().patch_type }
}

/// Annulus A: r 0.5..1.5 nr 8, z 0..2 nz 5, theta 5 deg.
fn annulus() -> WedgeSpec {
    spec_of(
        GradedAxis { lo: 0.5 as Scalar, hi: 1.5 as Scalar, n: 8, ..GradedAxis::default() },
        GradedAxis { lo: 0.0 as Scalar, hi: 2.0 as Scalar, n: 5, ..GradedAxis::default() },
        5.0,
    )
}

/// Annulus G: A with the radial axis geometrically graded (ratio 4).
fn annulus_graded() -> WedgeSpec {
    let mut s = annulus();
    s.r.expansion = 4.0 as Scalar;
    s
}

/// Pipe P: r 0..1 nr 6, z 0..3 nz 4.
fn pipe(theta: f64) -> WedgeSpec {
    spec_of(
        GradedAxis { lo: 0.0 as Scalar, hi: 1.0 as Scalar, n: 6, ..GradedAxis::default() },
        GradedAxis { lo: 0.0 as Scalar, hi: 3.0 as Scalar, n: 4, ..GradedAxis::default() },
        theta,
    )
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// The volume gate of §119.0, class A: the mesh holds `t (R2^2 - R1^2) L`
/// and every cell its closed form, to 1e-12 relative.
#[test]
fn wedge_annulus_volume_in_both_builds() {
    for theta in [1.0, 5.0, 10.0] {
        let t = wedge_half_tan(theta);
        println!("theta {theta:>5} deg: tan(theta/2)/(theta/2) = {:.7}", t / (theta.to_radians() * 0.5));
        let mut a = annulus();
        a.theta_deg = theta;
        let mut gspec = annulus_graded();
        gspec.theta_deg = theta;
        for spec in [a, gspec] {
            let m = build_wedge_mesh(&spec).unwrap();
            let rpt = m.check();
            let want_total = t * (1.5 * 1.5 - 0.5 * 0.5) * 2.0;
            assert!(
                rel(rpt.total_volume, want_total) <= 1e-12,
                "theta {theta}: total volume {} != {want_total}",
                rpt.total_volume
            );
            let g = m.geom64();
            let (r, z) = wedge_nodes(&spec).unwrap();
            for k in 0..spec.z.n {
                for i in 0..spec.r.n {
                    let c = i + spec.r.n * k;
                    let want = t * (r[i + 1] * r[i + 1] - r[i] * r[i]) * (z[k + 1] - z[k]);
                    assert!(
                        rel(g.v[c], want) <= 1e-12,
                        "theta {theta} cell ({i},{k}): v = {} != {want}",
                        g.v[c]
                    );
                }
            }
            assert!(rpt.max_closure_error <= 1e-12, "closure {}", rpt.max_closure_error);
            assert!(rpt.ldu_ordered);
            assert_eq!(rpt.n_regions, 1);
        }
    }
}

/// The pipe's axis, class A + D: no `inner` patch, no degenerate face, the
/// i = 0 cells 5-face prisms with two axis triangles, the smallest face in
/// the mesh the axis triangle `t r_1^2`.
#[test]
fn wedge_pipe_axis_has_no_degenerate_faces_in_both_builds() {
    for theta in [1.0, 5.0, 10.0] {
        let spec = pipe(theta);
        let nr = spec.r.n;
        let nz = spec.z.n;
        let raw = raw_wedge_mesh(&spec).unwrap();
        let m = build_wedge_mesh(&spec).unwrap();
        let g = m.geom64();

        assert!(!m.patches.iter().any(|p| p.name == "inner"), "no inner patch on a pipe");
        assert_eq!(m.patches.len(), 5);
        assert_eq!(m.n_points, (nz + 1) * (2 * nr + 1));

        for fv in &raw.faces {
            assert!(fv.len() >= 3, "a face with {} points", fv.len());
            let mut s = fv.clone();
            s.sort_unstable();
            s.dedup();
            assert_eq!(s.len(), fv.len(), "repeated point label in a face");
        }

        let n_pts: Vec<usize> = raw.faces.iter().map(|f| f.len()).collect();
        let mut n_faces = vec![0usize; m.n_cells];
        let mut n_tris = vec![0usize; m.n_cells];
        for (fi, o) in raw.owner.iter().enumerate() {
            n_faces[*o as usize] += 1;
            if n_pts[fi] == 3 {
                n_tris[*o as usize] += 1;
            }
        }
        for (fi, n) in raw.neighbour.iter().enumerate() {
            n_faces[*n as usize] += 1;
            if n_pts[fi] == 3 {
                n_tris[*n as usize] += 1;
            }
        }
        for c in 0..m.n_cells {
            if c % nr == 0 {
                assert_eq!(n_faces[c], 5, "axis cell {c}");
                assert_eq!(n_tris[c], 2, "axis cell {c}: two 3-point faces");
            } else {
                assert_eq!(n_faces[c], 6, "cell {c}");
                assert_eq!(n_tris[c], 0, "cell {c}: all faces 4-point");
            }
        }
        let n_tri_total = n_pts.iter().filter(|&&n| n == 3).count();
        assert_eq!(n_tri_total, nz + 1);

        let mut min_sf = f64::INFINITY;
        for x in g.mag_sf.iter().chain(g.b_mag_sf.iter()) {
            assert!(*x >= 1e-14, "face area {x}");
            min_sf = min_sf.min(*x);
        }
        let t = wedge_half_tan(theta);
        let (r, _) = wedge_nodes(&spec).unwrap();
        assert!(
            rel(min_sf, t * r[1] * r[1]) <= 1e-12,
            "smallest face area {min_sf} != axis triangle {}",
            t * r[1] * r[1]
        );
        let want_total = t * 1.0 * 3.0;
        assert!(rel(m.check().total_volume, want_total) <= 1e-12);
        assert!(m.check().max_closure_error <= 1e-12);
    }
}

/// The wedge planes, class A: §119.0's normals on every back/front face,
/// §119.0's areas on both sides and on every radial face, the +r vectors
/// purely along +x.
#[test]
fn wedge_faces_lie_in_the_wedge_planes_in_both_builds() {
    let theta: f64 = 5.0;
    let h = theta.to_radians() * 0.5;
    let (sin_h, cos_h) = h.sin_cos();
    let t = wedge_half_tan(theta);
    for spec in [annulus(), pipe(theta)] {
        let m = build_wedge_mesh(&spec).unwrap();
        let g = m.geom64();
        let (r, z) = wedge_nodes(&spec).unwrap();
        let nr = spec.r.n;
        let nz = spec.z.n;

        let want_names: Vec<&str> = WEDGE_PATCH_NAMES
            .iter()
            .copied()
            .filter(|n| *n != "inner" || spec.r.lo != 0.0 as Scalar)
            .collect();
        let names: Vec<&str> = m.patches.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, want_names);
        for p in &m.patches {
            let want = match p.name.as_str() {
                "inner" | "outer" => "wall",
                "zmin" | "zmax" => "patch",
                "back" | "front" => "wedge",
                other => panic!("unexpected patch {other}"),
            };
            assert_eq!(p.type_name, want, "type of {}", p.name);
        }
        for p in &m.patches {
            let want = match p.name.as_str() {
                "back" => [-sin_h, -cos_h, 0.0],
                "front" => [-sin_h, cos_h, 0.0],
                _ => continue,
            };
            for j in 0..p.size {
                let bf = p.start + j;
                let n = [
                    g.b_sf[bf].x / g.b_mag_sf[bf],
                    g.b_sf[bf].y / g.b_mag_sf[bf],
                    g.b_sf[bf].z / g.b_mag_sf[bf],
                ];
                for (a, b) in n.iter().zip(want) {
                    assert!((a - b).abs() <= 1e-12, "{} face {bf}: {n:?} != {want:?}", p.name);
                }
            }
        }
        let back = m.patches.iter().find(|p| p.name == "back").unwrap();
        let front = m.patches.iter().find(|p| p.name == "front").unwrap();
        for k in 0..nz {
            for i in 0..nr {
                let c = i + nr * k;
                let want = (r[i + 1] - r[i]) * (z[k + 1] - z[k]) / cos_h;
                assert!(rel(g.b_mag_sf[back.start + c], want) <= 1e-12, "back ({i},{k})");
                assert!(rel(g.b_mag_sf[front.start + c], want) <= 1e-12, "front ({i},{k})");
            }
        }
        let mut fi = 0usize;
        for k in 0..nz {
            for i in 0..nr {
                if i + 1 < nr {
                    let want = 2.0 * t * r[i + 1] * (z[k + 1] - z[k]);
                    assert!(rel(g.mag_sf[fi], want) <= 1e-12, "+r face ({i},{k})");
                    assert!(g.sf[fi].y.abs() <= 1e-12 * g.mag_sf[fi], "sf.y ({i},{k})");
                    assert!(g.sf[fi].z.abs() <= 1e-12 * g.mag_sf[fi], "sf.z ({i},{k})");
                    assert!(g.sf[fi].x > 0.0, "sf.x ({i},{k})");
                    fi += 1;
                }
                if k + 1 < nz {
                    fi += 1;
                }
            }
        }
        assert_eq!(fi, m.n_internal_faces);
    }
}

/// Class D: every front point is the bitwise mirror of its back point, in
/// both builds; on a pipe the shared axis point sits on the axis.
#[test]
fn wedge_front_is_the_mirror_of_back_in_both_builds() {
    for spec in [annulus(), pipe(5.0)] {
        let raw = raw_wedge_mesh(&spec).unwrap();
        let nr = spec.r.n;
        let nz = spec.z.n;
        let is_pipe = spec.r.lo == 0.0 as Scalar;
        let i0: usize = if is_pipe { 1 } else { 0 };
        let np = (nr + 1) + (nr + 1 - i0);
        assert_eq!(raw.points.len(), (nz + 1) * np);
        for k in 0..=nz {
            for i in i0..=nr {
                let bp = &raw.points[k * np + i];
                let fp = &raw.points[k * np + (nr + 1) + (i - i0)];
                assert_eq!(fp.x.to_bits(), bp.x.to_bits(), "x of F({i},{k})");
                assert_eq!(fp.z.to_bits(), bp.z.to_bits(), "z of F({i},{k})");
                assert_eq!(fp.y.to_bits(), (-bp.y).to_bits(), "y of F({i},{k})");
            }
        }
        if is_pipe {
            for k in 0..=nz {
                let axis = &raw.points[k * np];
                assert_eq!(axis.x, 0.0, "axis x at k = {k}");
                assert_eq!(axis.y.to_bits(), 0.0f64.to_bits(), "axis y at k = {k} is +0.0, not -0.0");
            }
        }
    }
}
