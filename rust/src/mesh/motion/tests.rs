// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
//! SPEC-LIT section 105.9. No GPL-licensed source was consulted.

use super::*;
use crate::ale_flow::{
    piston_rig, set_kinds, stroke_law, stroke_rig, PISTON_C, PISTON_DT, PISTON_N, PISTON_STEPS,
    STROKE_T,
};
use crate::mesh::refined;
use crate::mesh::PatchKind;

/// The piston fixture, built once per use - the gate's box, kinds and rules.
fn piston_box() -> crate::ale_flow::Rig {
    piston_rig().expect("piston rig")
}

/// The stroke fixture, built once per use.
fn stroke_box() -> crate::ale_flow::Rig {
    stroke_rig(stroke_law()).expect("stroke rig")
}

/// The piston's rule table, for the refusal tests to twist.
fn piston_rules() -> Vec<(&'static str, PatchMotion)> {
    vec![
        ("xmin", PatchMotion::Fixed),
        (
            "xmax",
            PatchMotion::Move(Displacement::Linear {
                velocity: Vec3::new(-PISTON_C, 0.0, 0.0),
            }),
        ),
        ("ymin", PatchMotion::Slide),
        ("ymax", PatchMotion::Slide),
        ("zmin", PatchMotion::Slide),
        ("zmax", PatchMotion::Slide),
    ]
}

/// Bitwise equality of two points, all three components.
fn same(a: Vec3, b: Vec3) -> bool {
    a.x.to_bits() == b.x.to_bits()
        && a.y.to_bits() == b.y.to_bits()
        && a.z.to_bits() == b.z.to_bits()
}

/// The point ids of one patch's faces, ascending, from the box's face lists.
fn patch_points(rig: &crate::ale_flow::Rig, name: &str) -> Vec<usize> {
    let p = rig
        .rb
        .mesh
        .patches
        .iter()
        .find(|p| p.name == name)
        .expect("patch present");
    let nif = rig.rb.mesh.n_internal_faces;
    let mut out = Vec::new();
    for k in 0..p.size {
        for &pt in &rig.rb.faces[nif + p.start + k] {
            out.push(pt as usize);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Every cell's volume ratio against rest, on the mesh at `points_at(t)`.
fn min_ratio_at(rig: &crate::ale_flow::Rig, t: Scalar) -> (Scalar, usize) {
    let mut m = rig.rb.mesh.clone();
    let pts = rig.motion.points_at(t);
    m.compute_geometry(&pts, &rig.rb.faces).expect("geometry at t");
    let mut ratio = Scalar::INFINITY;
    for c in 0..m.n_cells {
        ratio = ratio.min(m.v[c] / rig.rb.mesh.v[c]);
    }
    (ratio, m.n_cells)
}

/// M1 - at rest, both rigs reproduce their rest points bit for bit.
#[test]
fn the_smoother_leaves_a_mesh_at_rest_where_it_was() {
    for rig in [piston_box(), stroke_box()] {
        let pts = rig.motion.points_at(0.0);
        assert_eq!(pts.len(), rig.motion.rest().len());
        for (p, r) in pts.iter().zip(rig.motion.rest()) {
            assert!(same(*p, *r), "point moved at rest: {p:?} vs {r:?}");
        }
    }
}

/// M2 - the laws here all move along x, so no point's y or z can move.
#[test]
fn a_displacement_along_x_moves_no_point_in_y_or_z() {
    for rig in [piston_box(), stroke_box()] {
        let pts = rig.motion.points_at(0.13);
        for (p, r) in pts.iter().zip(rig.motion.rest()) {
            assert!(
                p.y.to_bits() == r.y.to_bits() && p.z.to_bits() == r.z.to_bits(),
                "point moved off its plane: {p:?} vs {r:?}"
            );
        }
    }
}

/// M3 - every patch moved by the SAME law is a rigid translation, and the
/// smoother must reproduce it on the interior points.
#[test]
fn a_rigid_translation_is_reproduced() {
    let rb = refined::build([4, 4, 4], Vec3::new(0.25, 0.25, 0.25), &[0u32; 64]).expect("box");
    let law = Displacement::Linear {
        velocity: Vec3::new(0.3, -0.2, 0.1),
    };
    let motion = MeshMotion::new(
        &rb.mesh,
        &rb.points,
        &rb.faces,
        &[
            ("xmin", PatchMotion::Move(law)),
            ("xmax", PatchMotion::Move(law)),
            ("ymin", PatchMotion::Move(law)),
            ("ymax", PatchMotion::Move(law)),
            ("zmin", PatchMotion::Move(law)),
            ("zmax", PatchMotion::Move(law)),
        ],
    )
    .expect("rigid rules");
    let d = law.at(0.5);
    let pts = motion.points_at(0.5);
    let mut worst = 0.0 as Scalar;
    for (p, r) in pts.iter().zip(rb.points.iter()) {
        let want = *r + d;
        let err = (p.x - want.x)
            .abs()
            .max((p.y - want.y).abs())
            .max((p.z - want.z).abs());
        worst = worst.max(err);
    }
    println!("  rigid translation: worst |err| = {worst:.3e} over {} points", pts.len());
    assert!(worst <= 1e-15, "worst |err| {worst:.3e}");
}

/// M4 - the prescribed patches sit exactly where their laws put them, and the
/// fixed patch exactly where it was.
#[test]
fn every_prescribed_point_is_where_its_law_puts_it() {
    let rig = piston_box();
    let t = 0.1;
    let pts = rig.motion.points_at(t);
    let law = Displacement::Linear {
        velocity: Vec3::new(-PISTON_C, 0.0, 0.0),
    };
    for &p in &patch_points(&rig, "xmax") {
        let want = rig.motion.rest()[p] + law.at(t);
        assert!(same(pts[p], want), "piston xmax point {p}: {:?} vs {:?}", pts[p], want);
    }
    for &p in &patch_points(&rig, "xmin") {
        assert!(same(pts[p], rig.motion.rest()[p]), "piston xmin point {p} moved");
    }

    let rig = stroke_box();
    let pts = rig.motion.points_at(STROKE_T);
    let dx = match stroke_law() {
        PatchMotion::Move(l) => l.at(STROKE_T),
        _ => unreachable!("stroke_law is a Move"),
    };
    for &p in &patch_points(&rig, "xmax") {
        let want = rig.motion.rest()[p] + dx;
        assert!(same(pts[p], want), "stroke xmax point {p}: {:?} vs {:?}", pts[p], want);
    }
}

/// M5 - at the gate amplitudes no cell inverts. The two prints are the
/// measured volume ratios SPEC-LIT 105.9 quotes.
#[test]
fn the_smoothed_mesh_keeps_every_cell_positive_at_the_gate_amplitudes() {
    let rig = piston_box();
    let t = PISTON_DT * PISTON_STEPS as Scalar;
    let (ratio, n) = min_ratio_at(&rig, t);
    println!("  piston min V/V0 at t={t}: {ratio:.10e} over {n} cells");
    assert!(ratio > 0.0, "a cell inverted at t={t}");

    let rig = stroke_box();
    let (ratio, n) = min_ratio_at(&rig, STROKE_T);
    println!("  stroke min V/V0 at t={STROKE_T}: {ratio:.10e} over {n} cells");
    assert!(ratio > 0.0, "a cell inverted at t={STROKE_T}");
}

/// M6 - the refusal table, each once, by its substring. (g) through (o) of
/// SPEC-LIT 105.9; (l) - a cyclic patch under a non-`Fixed` rule - is NOT
/// exercised here because no fixture in this file has a cyclic patch; that
/// refusal is covered by reading `MeshMotion::new`, which tests nothing else
/// about the walk before it.
#[test]
fn the_motion_rules_refuse_what_they_cannot_mean() {
    // (g) the smoother's control list
    let e = Idw::new(Vec::new()).expect_err("empty control list");
    let msg = e.to_string();
    assert!(
        msg.starts_with("Idw::new: ") && msg.contains("control point"),
        "(g) empty: {msg}"
    );
    let e = Idw::new(vec![Vec3::ZERO; 3]).expect_err("zero extent");
    let msg = e.to_string();
    assert!(
        msg.starts_with("Idw::new: ") && msg.contains("control point"),
        "(g) zero extent: {msg}"
    );

    let rig = piston_box();
    let rules = piston_rules();

    // (h) a rest list one point short of the mesh's points
    let e = MeshMotion::new(
        &rig.rb.mesh,
        &rig.rb.points[..rig.rb.points.len() - 1],
        &rig.rb.faces,
        &rules,
    )
    .expect_err("short rest list");
    let msg = e.to_string();
    assert!(
        msg.starts_with("MeshMotion::new: ") && msg.contains(" points"),
        "(h) short rest: {msg}"
    );

    // (h2) the same mesh CLAIMING no points at all: the count guard above
    // passes it, and the walk must refuse any face whose point id reaches
    // past the rest list instead of indexing out of it.
    let mut m0 = rig.rb.mesh.clone();
    m0.n_points = 0;
    let e = MeshMotion::new(
        &m0,
        &rig.rb.points[..rig.rb.points.len() - 1],
        &rig.rb.faces,
        &rules,
    )
    .expect_err("a mesh claiming no points");
    let msg = e.to_string();
    assert!(
        msg.starts_with("MeshMotion::new: ") && msg.contains(" points"),
        "(h2) n_points = 0: {msg}"
    );

    // (i) a rule naming no patch of the mesh
    let mut r = rules.clone();
    r[0].0 = "nope";
    let e = MeshMotion::new(&rig.rb.mesh, &rig.rb.points, &rig.rb.faces, &r)
        .expect_err("unknown patch");
    let msg = e.to_string();
    assert!(msg.contains("no patch named"), "(i): {msg}");

    // (j) one patch named twice
    let mut r = rules.clone();
    r[2].0 = "xmin";
    let e = MeshMotion::new(&rig.rb.mesh, &rig.rb.points, &rig.rb.faces, &r)
        .expect_err("xmin twice");
    let msg = e.to_string();
    assert!(msg.contains("is named twice"), "(j): {msg}");

    // (k) a patch left without a rule
    let e = MeshMotion::new(&rig.rb.mesh, &rig.rb.points, &rig.rb.faces, &rules[..5])
        .expect_err("zmax left out");
    let msg = e.to_string();
    assert!(msg.contains("has no motion rule"), "(k): {msg}");

    // (m) two patches prescribing different motions for one shared point
    let mut r = rules.clone();
    r[0].1 = PatchMotion::Move(Displacement::Linear {
        velocity: Vec3::new(-PISTON_C, 0.0, 0.0),
    });
    r[2].1 = PatchMotion::Fixed;
    let e = MeshMotion::new(&rig.rb.mesh, &rig.rb.points, &rig.rb.faces, &r)
        .expect_err("Move meets Fixed");
    let msg = e.to_string();
    assert!(msg.contains("prescribe different motions"), "(m): {msg}");

    // (n) the moving wall asked for on a patch whose rule does not move
    let e = rig
        .motion
        .wall_faces(&rig.rb.mesh, &["xmin"])
        .expect_err("a Fixed patch has no moving wall");
    let msg = e.to_string();
    assert!(
        msg.contains("MeshMotion::wall_faces: patch ") && msg.contains(" does not move"),
        "(n): {msg}"
    );

    // (o) a law with a tangential component on the moving wall
    let mut rb = refined::build(PISTON_N, Vec3::new(0.125, 0.25, 0.25), &[0u32; 72]).expect("box");
    set_kinds(
        &mut rb,
        [
            PatchKind::Generic,
            PatchKind::Wall,
            PatchKind::Symmetry,
            PatchKind::Symmetry,
            PatchKind::Symmetry,
            PatchKind::Symmetry,
        ],
    )
    .expect("kinds");
    let motion = MeshMotion::new(
        &rb.mesh,
        &rb.points,
        &rb.faces,
        &[
            ("xmin", PatchMotion::Fixed),
            (
                "xmax",
                PatchMotion::Move(Displacement::Linear {
                    velocity: Vec3::new(0.0, 0.5, 0.0),
                }),
            ),
            ("ymin", PatchMotion::Slide),
            ("ymax", PatchMotion::Slide),
            ("zmin", PatchMotion::Slide),
            ("zmax", PatchMotion::Slide),
        ],
    )
    .expect("the rules build; only wall_faces refuses");
    let e = motion
        .wall_faces(&rb.mesh, &["xmax"])
        .expect_err("a tangential wall");
    let msg = e.to_string();
    assert!(msg.contains("a tangential wall velocity is not implemented"), "(o): {msg}");
}

/// M7 - the two laws say what they do.
#[test]
fn the_two_laws_are_what_they_say() {
    let lin = Displacement::Linear {
        velocity: Vec3::new(0.3, -0.2, 0.1),
    };
    assert!(same(lin.at(2.0), Vec3::new(0.6, -0.4, 0.2)));

    let period = 2.0 as Scalar;
    let sin = Displacement::Sine {
        amplitude: Vec3::new(0.4, 0.0, 0.0),
        period,
    };
    assert!(same(sin.at(0.0), Vec3::ZERO));
    assert!(same(sin.at(period / 4.0), Vec3::new(0.4, 0.0, 0.0)));
    let half = sin.at(period / 2.0);
    assert!(
        half.x.abs() <= 1e-15 && half.y == 0.0 && half.z == 0.0,
        "half period: {half:?}"
    );
}
