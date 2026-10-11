// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §119.1 - `PatchKind::Wedge` is its own kind and the mesh reader
//! derives the wedge, through the public API in BOTH builds: the same six
//! derivations and refusals the lib tests `mesh::wedge::tests` pin, at the
//! same 1e-12 and 1e-8 tolerances. Precision class A - the derivation is
//! evaluated in f64 through `geom64` in both builds (the raw mesh's points
//! are f64 in both builds, §118.2), so points are located with 1e-6
//! tolerances, never `==`. No `#[cfg_attr(feature = "single", ignore)]`
//! anywhere.

use ofgpu::blockgen::{GradedAxis, WedgeSpec, build_wedge_mesh, raw_wedge_mesh};
use ofgpu::io::polymesh::build_host_mesh;
use ofgpu::{Error, PatchKind};

fn spec_of(r: GradedAxis, z: GradedAxis, theta: f64) -> WedgeSpec {
    WedgeSpec { r, z, theta_deg: theta, patch_type: WedgeSpec::default().patch_type }
}

/// Annulus A: r 0.5..1.5 nr 8, z 0..2 nz 5, theta 5 deg.
fn annulus() -> WedgeSpec {
    spec_of(
        GradedAxis { lo: 0.5, hi: 1.5, n: 8, ..GradedAxis::default() },
        GradedAxis { lo: 0.0, hi: 2.0, n: 5, ..GradedAxis::default() },
        5.0,
    )
}

/// Annulus A at another angle.
fn annulus_at(theta: f64) -> WedgeSpec {
    let mut s = annulus();
    s.theta_deg = theta;
    s
}

/// Pipe P: r 0..1 nr 6, z 0..3 nz 4.
fn pipe(theta: f64) -> WedgeSpec {
    spec_of(
        GradedAxis { lo: 0.0, hi: 1.0, n: 6, ..GradedAxis::default() },
        GradedAxis { lo: 0.0, hi: 3.0, n: 4, ..GradedAxis::default() },
        theta,
    )
}

#[test]
fn wedge_is_not_symmetry() {
    assert_ne!(PatchKind::from_type("wedge"), PatchKind::Symmetry);
    assert_eq!(PatchKind::from_type("wedge"), PatchKind::Wedge);
    assert_eq!(PatchKind::from_type("wedge").as_str(), "wedge");
}

/// The derivation itself, class A, both builds: axis +z, mid-plane +y, the
/// spec's full angle, the axis point at mid-height on the axis line.
#[test]
fn wedge_axis_angle_and_mid_plane_are_derived() {
    for theta_deg in [1.0, 5.0, 10.0] {
        for (spec, r_out, z_mid, l) in
            [(pipe(theta_deg), 1.0, 1.5, 3.0), (annulus_at(theta_deg), 1.5, 1.0, 2.0)]
        {
            let m = build_wedge_mesh(&spec).unwrap();
            let w = m.wedge.clone().unwrap();
            assert_eq!(m.patches[w.back].name, "back");
            assert_eq!(m.patches[w.front].name, "front");
            assert_eq!(w.n_faces, spec.r.n * spec.z.n);
            let axis_err = (w.axis - ofgpu::DVec3::new(0.0, 0.0, 1.0)).mag();
            assert!(axis_err <= 1e-12, "axis error {axis_err:e}");
            assert!(
                (w.theta - theta_deg.to_radians()).abs() <= 1e-12,
                "theta {} vs {}",
                w.theta,
                theta_deg.to_radians()
            );
            assert!((w.mid_normal - ofgpu::DVec3::new(0.0, 1.0, 0.0)).mag() <= 1e-12);
            assert!(w.axis_point.x.hypot(w.axis_point.y) <= 1e-12 * r_out);
            assert!((w.axis_point.z - z_mid).abs() <= 1e-12 * l);
            println!("theta {theta_deg:>5} deg: axis error {axis_err:e}");
        }
    }
}

/// Both sides typed `symmetry`: the build is fine and there is no wedge.
#[test]
fn a_mesh_without_wedge_patches_has_no_wedge() {
    let mut raw = raw_wedge_mesh(&pipe(5.0)).unwrap();
    for name in ["back", "front"] {
        let p = raw.patches.iter_mut().find(|p| p.name == name).unwrap();
        p.type_name = "symmetry".to_string();
        p.kind = PatchKind::Symmetry;
    }
    let m = build_host_mesh(&raw).unwrap();
    assert!(m.wedge.is_none());
}

/// One wedge patch alone is refused, by name.
#[test]
fn refuses_an_unpaired_wedge() {
    let mut raw = raw_wedge_mesh(&pipe(5.0)).unwrap();
    let p = raw.patches.iter_mut().find(|p| p.name == "front").unwrap();
    p.type_name = "symmetry".to_string();
    p.kind = PatchKind::Symmetry;
    let msg = match build_host_mesh(&raw) {
        Err(Error::Mesh(msg)) => msg,
        other => panic!("expected Error::Mesh, got {other:?}"),
    };
    assert!(msg.contains("'back'"), "{msg}");
    assert!(msg.contains("has no partner"), "{msg}");
}

/// One front-side corner point nudged out of its plane is refused.
#[test]
fn refuses_a_non_planar_wedge() {
    let mut raw = raw_wedge_mesh(&annulus()).unwrap();
    let h = 2.5f64.to_radians();
    let hits: Vec<usize> = raw
        .points
        .iter()
        .enumerate()
        .filter(|(_, p)| p.y > 0.0 && (p.x - 1.0).abs() < 1e-6 && (p.z - 0.8).abs() < 1e-6)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(hits.len(), 1, "{hits:?}");
    let d = ofgpu::DVec3::new(-h.sin(), h.cos(), 0.0);
    let p = &mut raw.points[hits[0]];
    *p = *p + d * 1e-4;
    let msg = match build_host_mesh(&raw) {
        Err(Error::Mesh(msg)) => msg,
        other => panic!("expected Error::Mesh, got {other:?}"),
    };
    assert!(msg.contains("'front'"), "{msg}");
    assert!(msg.contains("is not planar"), "{msg}");
}

/// A front side steeper than the back fails the mirror test face for face.
#[test]
fn refuses_an_asymmetric_wedge() {
    let mut raw = raw_wedge_mesh(&annulus()).unwrap();
    let t3 = 3.0f64.to_radians().tan();
    for p in raw.points.iter_mut() {
        if p.y > 0.0 {
            p.y = p.x * t3;
        }
    }
    let msg = match build_host_mesh(&raw) {
        Err(Error::Mesh(msg)) => msg,
        other => panic!("expected Error::Mesh, got {other:?}"),
    };
    assert!(msg.contains("'back'"), "{msg}");
    assert!(msg.contains("'front'"), "{msg}");
    assert!(msg.contains("not symmetric about the mid-plane"), "{msg}");
}

/// The report line names the pair, the angle and the axis.
#[test]
fn report_line_names_the_pair_axis_and_angle() {
    let m = build_wedge_mesh(&pipe(5.0)).unwrap();
    let w = m.wedge.clone().unwrap();
    let line = w.report_line(&m);
    assert!(line.starts_with("  wedge: "), "{line:?}");
    assert!(line.contains("'back'"), "{line:?}");
    assert!(line.contains("'front'"), "{line:?}");
    assert!(line.contains("theta 5.000000 deg"), "{line:?}");
    assert!(line.contains("axis ("), "{line:?}");
}
