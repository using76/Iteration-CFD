// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
//! SPEC-LIT 105.13 and 105.14. No GPL-licensed source was consulted.

use super::*;
use crate::mesh::refined;

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The eight-by-eight, one-cell-deep box: domain `[0,1]x[0,1]x[0,0.1]`,
/// patches `xmin xmax ymin ymax` generic, `zmin zmax` empty.
fn wobble_box() -> refined::RefinedBox {
    let mut rb = refined::build([8, 8, 1], Vec3::new(0.125, 0.125, 0.1), &[0u32; 64])
        .expect("the fixture box builds");
    for (p, (kind, ty)) in rb.mesh.patches.iter_mut().zip([
        (PatchKind::Generic, "patch"),
        (PatchKind::Generic, "patch"),
        (PatchKind::Generic, "patch"),
        (PatchKind::Generic, "patch"),
        (PatchKind::Empty, "empty"),
        (PatchKind::Empty, "empty"),
    ]) {
        p.kind = kind;
        p.type_name = ty.to_string();
    }
    rb.mesh
        .compute_geometry(&rb.points, &rb.faces)
        .expect("the fixture's geometry");
    rb
}

/// The fixture's law: the whole box, amplitude 0.02, diagonal direction,
/// period 0.2.
fn box_law() -> WobbleLaw {
    let r2 = (2.0f64).sqrt();
    WobbleLaw {
        x0: 0.0,
        x1: 1.0,
        y0: 0.0,
        y1: 1.0,
        amplitude: 0.02,
        direction: Vec3::new((1.0 / r2) as Scalar, (1.0 / r2) as Scalar, 0.0),
        period: 0.2,
    }
}

/// The points of the four side patches, walked the way `ale_flow` walks
/// its boundary: face `n_internal + start + k`, its point list.
fn side_points(rb: &refined::RefinedBox) -> Vec<usize> {
    let mut set: Vec<usize> = Vec::new();
    for p in rb.mesh.patches.iter().take(4) {
        for k in 0..p.size {
            let fi = rb.mesh.n_internal_faces + p.start + k;
            for &pt in &rb.faces[fi] {
                if !set.contains(&(pt as usize)) {
                    set.push(pt as usize);
                }
            }
        }
    }
    set
}

#[test]
fn the_inlet_profile_is_the_papers_parabola() {
    let h = CHANNEL_H;
    assert!(inlet_velocity(0.0, 2.0) <= 1e-15);
    assert!(inlet_velocity(h, 2.0) <= 1e-15);
    assert!((inlet_velocity(h / 2.0, 2.0) - 3.0).abs() <= 1e-12);
    assert_eq!(inlet_velocity(-0.01, 2.0), 0.0);

    // The midpoint-rule mean over 4,000 equal strips of [0, H] is Ubar.
    let n = 4000usize;
    let dy = f64::from(h) / n as f64;
    let mut acc = 0.0f64;
    for i in 0..n {
        acc += f64::from(inlet_velocity(((i as f64 + 0.5) * dy) as Scalar, 2.0));
    }
    let mean = (acc * dy) / f64::from(h);
    assert!((mean - 2.0).abs() <= 1e-6 * 2.0, "midpoint mean {mean}");

    assert_eq!(ramp(0.0), 0.0);
    assert!((ramp(1.0) - 0.5).abs() <= 1e-15);
    assert_eq!(ramp(2.0), 1.0);
    assert_eq!(ramp(5.0), 1.0);
}

#[test]
#[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
fn the_wobble_leaves_every_boundary_point_where_it_is() {
    let rb = wobble_box();
    let law = box_law();
    let w = Wobble::new(&rb.mesh, &rb.points, &rb.faces, law).expect("the box law binds");
    assert!(w.n_moving() > 0);
    let sides = side_points(&rb);
    assert!(!sides.is_empty());
    for k in 0..7 {
        let t = 0.02 + 0.03 * k as f64;
        let pts = w.points_at(t as Scalar);
        for p in 0..rb.points.len() {
            assert_eq!(pts[p].z, rb.points[p].z, "z of point {p} moved at t={t}");
            if sides.contains(&p) {
                assert_eq!(pts[p].x, rb.points[p].x, "x of side point {p} at t={t}");
                assert_eq!(pts[p].y, rb.points[p].y, "y of side point {p} at t={t}");
            }
        }
    }
    // A box that reaches the xmin patch is refused, naming that patch.
    let mut reach = box_law();
    reach.x0 = -0.5;
    match Wobble::new(&rb.mesh, &rb.points, &rb.faces, reach) {
        Ok(_) => panic!("a box reaching xmin must be refused"),
        Err(e) => {
            let text = e.to_string();
            assert!(text.contains("xmin"), "the refusal names the patch: {text}");
        }
    }
}

#[test]
#[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
fn the_wobble_moves_the_interior_by_its_law() {
    let rb = wobble_box();
    let law = box_law();
    let w = Wobble::new(&rb.mesh, &rb.points, &rb.faces, law).expect("binds");
    // The point nearest the box centre.
    let (ci, _) = rb
        .points
        .iter()
        .enumerate()
        .min_by(|(i, a), (j, b)| {
            let da = (a.x - 0.5) * (a.x - 0.5) + (a.y - 0.5) * (a.y - 0.5);
            let db = (b.x - 0.5) * (b.x - 0.5) + (b.y - 0.5) * (b.y - 0.5);
            da.partial_cmp(&db).unwrap().then(i.cmp(j))
        })
        .unwrap();
    let rest = rb.points[ci];
    let at_005 = w.points_at(0.05)[ci];
    let d = at_005 - rest;
    let want = law.displacement(rest, 0.05);
    assert!(
        (d.x - want.x).abs() <= 1e-15 && (d.y - want.y).abs() <= 1e-15 && (d.z - want.z).abs() <= 1e-15,
        "displacement {d:?} against the law {want:?}"
    );
    let at_010 = w.points_at(0.1)[ci];
    let d = at_010 - rest;
    assert!(
        d.x.abs() <= 1e-15 && d.y.abs() <= 1e-15 && d.z.abs() <= 1e-15,
        "at sin(pi) the displacement is round-off: {d:?}"
    );
    // The Turek law's shape: zero on and outside the box, ~1 at its middle.
    let t = WobbleLaw::turek();
    assert_eq!(t.shape(Vec3::new(0.6, 0.2, 0.0)), 0.0);
    assert_eq!(t.shape(Vec3::new(1.5, CHANNEL_H, 0.0)), 0.0);
    assert!(t.shape(Vec3::new(1.5, 0.205, 0.0)) > 0.99);
}

#[test]
#[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
fn a_uniform_pressure_on_a_patch_pushes_with_its_area_vector() {
    let mut rb = wobble_box();
    let kinds = [
        (PatchKind::Generic, "patch"),
        (PatchKind::Wall, "wall"),
        (PatchKind::Generic, "patch"),
        (PatchKind::Generic, "patch"),
        (PatchKind::Empty, "empty"),
        (PatchKind::Empty, "empty"),
    ];
    for (p, (k, ty)) in rb.mesh.patches.iter_mut().zip(kinds) {
        p.kind = k;
        p.type_name = ty.to_string();
    }
    rb.mesh
        .compute_geometry(&rb.points, &rb.faces)
        .expect("recompute");
    let hm = &rb.mesh;
    let xmax = 1usize;
    let nbf = hm.n_boundary_faces;
    let p_bf = vec![2.0 as Scalar; nbf];
    let u = vec![Vec3::ZERO; hm.n_cells];
    let u_bf = vec![Vec3::ZERO; nbf];
    let f = body_forces(hm, &[xmax], 0.1, &p_bf, &u, &u_bf);
    let mut area = Vec3::ZERO;
    for k in 0..hm.patches[xmax].size {
        area += hm.b_sf[hm.patches[xmax].start + k];
    }
    let want = RHO * 2.0 * area / 0.1;
    let rel = |a: Vec3, b: Vec3| {
        ((a.x - b.x).abs() + (a.y - b.y).abs() + (a.z - b.z).abs())
            / (b.x.abs() + b.y.abs() + b.z.abs())
    };
    assert!(rel(f.pressure, want) <= 1e-9, "pressure {:?} against {want:?}", f.pressure);
    assert_eq!(f.viscous, Vec3::ZERO);
    assert!((f.drag - 2.0e3).abs() <= 1e-9 * 2.0e3, "drag {}", f.drag);
    assert!(f.lift.abs() <= 1e-9, "lift {}", f.lift);
}

#[test]
fn the_body_force_is_r4s_two_integrators_and_nothing_else() {
    let mut rb = wobble_box();
    let kinds = [
        (PatchKind::Generic, "patch"),
        (PatchKind::Wall, "wall"),
        (PatchKind::Generic, "patch"),
        (PatchKind::Wall, "wall"),
        (PatchKind::Empty, "empty"),
        (PatchKind::Empty, "empty"),
    ];
    for (p, (k, ty)) in rb.mesh.patches.iter_mut().zip(kinds) {
        p.kind = k;
        p.type_name = ty.to_string();
    }
    rb.mesh
        .compute_geometry(&rb.points, &rb.faces)
        .expect("recompute");
    let hm = &rb.mesh;
    let ymax = 3usize;
    let nbf = hm.n_boundary_faces;
    let u_cells: Vec<Vec3> = hm.c.iter().map(|c| Vec3::new(c.y, 0.0, 0.0)).collect();
    let u_bf = vec![Vec3::ZERO; nbf];
    let p_bf: Vec<Scalar> = hm.b_cf.iter().map(|c| c.x + 2.0 * c.y).collect();
    let f = body_forces(hm, &[ymax], 0.1, &p_bf, &u_cells, &u_bf);

    // The identical call (C1) makes, compared row by row, BITWISE.
    let rho_bf = vec![RHO; nbf];
    let nut_bf = vec![0.0 as Scalar; nbf];
    let kb = vec![false; nbf];
    let pf = pressure_force(hm, &p_bf, &rho_bf);
    let ws = wall_shear(
        hm,
        Vec3::new(1.0, 0.0, 0.0),
        &u_cells,
        &u_bf,
        &rho_bf,
        &nut_bf,
        None,
        &kb,
        NU,
        0.09,
    );
    let prow = pf.by_patch.iter().find(|r| r.patch == ymax).expect("row");
    let wrow = ws.by_patch.iter().find(|r| r.patch == ymax).expect("row");
    let want_p = prow.force / 0.1;
    let want_v = wrow.force / 0.1;
    assert_eq!(f.pressure, want_p, "pressure bitwise");
    assert_eq!(f.viscous, want_v, "viscous bitwise");
}

#[test]
fn the_periodic_reduction_recovers_a_sampled_sine() {
    let n = 2001usize;
    let dt = 1e-3;
    let t: Vec<Scalar> = (0..n).map(|i| i as Scalar * dt).collect();
    let v: Vec<Scalar> = t
        .iter()
        .map(|&ti| {
            (3.0 + 2.0 * (2.0 * std::f64::consts::PI * 4.4 * f64::from(ti) + 0.3).sin()) as Scalar
        })
        .collect();
    let bounds = upward_crossings(&t, &v);
    let p = reduce_periods(&t, &v, &bounds, 4).expect("four full periods");
    assert_eq!(p.periods, 4);
    assert!((p.mean - 3.0).abs() <= 1e-5, "mean {}", p.mean);
    assert!((p.amplitude - 2.0).abs() <= 1e-5 * 2.0, "amplitude {}", p.amplitude);
    assert!(
        (p.frequency - 4.4).abs() <= 1e-5 * 4.4,
        "frequency {}",
        p.frequency
    );
    // A second signal reduced over the FIRST's bounds: two of its periods
    // per interval, so the extremes are still exact.
    let w: Vec<Scalar> = t
        .iter()
        .map(|&ti| (440.0 + 5.0 * (2.0 * std::f64::consts::PI * 8.8 * f64::from(ti)).sin()) as Scalar)
        .collect();
    let q = reduce_periods(&t, &w, &bounds, 4).expect("reduces");
    assert!((q.mean - 440.0).abs() <= 1e-4, "mean {}", q.mean);
    assert!((q.amplitude - 5.0).abs() <= 1e-4 * 5.0, "amplitude {}", q.amplitude);
}

#[test]
fn the_periodic_reduction_refuses_too_few_periods() {
    let n = 501usize;
    let dt = 1e-3;
    let t: Vec<Scalar> = (0..n).map(|i| i as Scalar * dt).collect();
    let v: Vec<Scalar> = t
        .iter()
        .map(|&ti| {
            (3.0 + 2.0 * (2.0 * std::f64::consts::PI * 4.4 * f64::from(ti) + 0.3).sin()) as Scalar
        })
        .collect();
    let bounds = upward_crossings(&t, &v);
    assert_eq!(bounds.len(), 2, "two and a half periods: {bounds:?}");
    match reduce_periods(&t, &v, &bounds, 4) {
        Ok(p) => panic!("four periods from three crossings: {p:?}"),
        Err(e) => assert!(e.to_string().contains("periods"), "names it: {e}"),
    }
    assert!(reduce_periods(&t, &v, &bounds, 0).is_err());
}

#[test]
fn a_missing_mesh_level_is_refused_with_the_command_that_makes_it() {
    let dir = std::env::temp_dir().join(format!("ofgpuTurekHron_absent_{}", std::process::id()));
    match load_rig_from(&dir, 2) {
        Ok(_) => panic!("no mesh exists at {dir:?}"),
        Err(e) => {
            let text = e.to_string();
            assert!(
                text.contains("python tools/mesh/examples/turek_hron.py --level 2"),
                "the refusal names the generating command: {text}"
            );
        }
    }
}

#[test]
#[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
fn a_uniform_flow_stays_uniform_on_a_wobbling_mesh() {
    let Some(g) = gpu() else { return };
    let rb = wobble_box();
    let dt = 0.01;
    let u_inf = Vec3::new(1.0, 0.0, 0.0);
    let gm = GpuMesh::upload(&g, &rb.mesh).expect("upload");
    let ctrl = transient_controls(dt);
    let mut s = Simple::new(&g, &rb.mesh, &gm, ctrl, buoyancy_off()).expect("simple");
    set_bcs(
        &g,
        &mut s,
        &rb.mesh,
        &[
            ("xmin", VelocityBc::Uniform(u_inf), PressureBc::ZeroGradient),
            ("xmax", VelocityBc::ZeroGradient, PressureBc::Fixed(0.0)),
            ("ymin", VelocityBc::Uniform(u_inf), PressureBc::ZeroGradient),
            ("ymax", VelocityBc::Uniform(u_inf), PressureBc::ZeroGradient),
            ("zmin", VelocityBc::Empty, PressureBc::Empty),
            ("zmax", VelocityBc::Empty, PressureBc::Empty),
        ],
    )
    .expect("bcs");
    g.write(&mut s.u_mut().f, &vec![u_inf; rb.mesh.n_cells]).expect("seed U");
    g.write(&mut s.p_mut().f, &vec![0.0 as Scalar; rb.mesh.n_cells]).expect("seed p");
    let phi_f: Vec<Scalar> = rb.mesh.sf.iter().map(|sfi| u_inf.dot(*sfi)).collect();
    g.write(&mut s.phi_mut().f, &phi_f).expect("seed phi");
    let mut phi_b: Vec<Scalar> = rb.mesh.b_sf.iter().map(|sfi| u_inf.dot(*sfi)).collect();
    for p in &rb.mesh.patches {
        if p.kind == PatchKind::Empty {
            for k in 0..p.size {
                phi_b[p.start + k] = 0.0;
            }
        }
    }
    g.write(&mut s.phi_mut().bf, &phi_b).expect("seed phi bf");
    s.initialise(&g).expect("initialise");
    let (nut, t) = laminar_fields(&g, &rb.mesh, &gm).expect("laminar");
    let mut backend = PbicgstabBackend::new(ctrl.p_solver);
    backend
        .setup(&g, &rb.mesh, &gm, &SystemProbe::default())
        .expect("backend");
    let law = box_law();
    let wobble = Wobble::new(&rb.mesh, &rb.points, &rb.faces, law).expect("bind");
    let csr = flatten_faces(&rb.faces);
    let ale = AleMesh::new(&g, &rb.mesh, &gm, &rb.points, &csr).expect("ale");
    s.attach_motion(&g, ale, &[]).expect("attach");

    let (mut max_u_dev, mut max_p_dev, mut max_dv) = (0.0f64, 0.0f64, 0.0f64);
    for k in 0..20 {
        if k > 0 {
            s.begin_time_step(&g, dt).expect("begin");
        }
        let pts = wobble.points_at((k + 1) as Scalar * dt);
        s.motion_mut()
            .ok_or_else(|| "no motion".to_string())
            .expect("motion")
            .set_points(&g, &pts)
            .expect("set_points");
        s.move_mesh(&g).expect("move");
        s.solve_step(&g, &mut backend, &nut, &t).expect("step");
        let u: Vec<Vec3> = g.download(&s.u().f).expect("u");
        for uc in &u {
            let d = *uc - u_inf;
            max_u_dev = max_u_dev.max(f64::from(d.dot(d)));
        }
        let p: Vec<Scalar> = g.download(&s.p().f).expect("p");
        for pc in &p {
            max_p_dev = max_p_dev.max(f64::from(pc.abs()));
        }
        let v: Vec<Scalar> = g.download(&gm.v).expect("v");
        for (vc, v0) in v.iter().zip(rb.mesh.v.iter()) {
            max_dv = max_dv.max(f64::from(vc / v0 - 1.0).abs());
        }
    }
    max_u_dev = max_u_dev.sqrt();
    println!(
        "wobbling uniform flow: max |U - (1,0,0)| {max_u_dev:.3e}, max |p| {max_p_dev:.3e}, \
         max |V/V0 - 1| {max_dv:.3e}"
    );
    assert!(max_u_dev <= 1e-9, "U stayed uniform: {max_u_dev:.3e}");
    assert!(max_p_dev <= 1e-9, "p stayed level: {max_p_dev:.3e}");
    assert!(max_dv > 1e-6, "the mesh did move: {max_dv:.3e}");
}

#[test]
#[ignore = "needs cases/turekHron/mesh (python tools/mesh/examples/turek_hron.py --level 1)"]
fn the_cfd1_rig_converges_on_the_coarse_mesh() {
    let Some(g) = gpu() else { return };
    let rig = load_rig(1).expect("the L1 mesh loads");
    let run = steady(&g, &rig, Case::Cfd1, STEADY_MAX_ITERS[0]).expect("the run completes");
    println!(
        "CFD1 L{}: {} cells, h {:.4e}, {} SIMPLE iterations, stopping rule met {}, \
         residual {:.3e}, drag {:.6}, lift {:.6}, {:.1} s",
        run.level, run.n_cells, run.h, run.iterations, run.converged,
        run.residual, run.forces.drag, run.forces.lift, run.seconds
    );
    assert!(run.converged, "the stopping rule was met");
    assert!(run.forces.drag > 10.0 && run.forces.drag < 20.0, "drag {}", run.forces.drag);
    assert!(run.forces.lift > 0.5 && run.forces.lift < 2.0, "lift {}", run.forces.lift);
}

#[test]
#[ignore = "needs cases/turekHron/mesh (python tools/mesh/examples/turek_hron.py --level 1)"]
fn the_cfd3_pair_runs_on_the_coarse_mesh() {
    let Some(g) = gpu() else { return };
    let rig = load_rig(1).expect("the L1 mesh loads");
    let r = cfd3_traces(&g, &rig, CFD3_DT, 200, 100, WobbleLaw::turek()).expect("the pair runs");
    for tr in [&r.static_run, &r.moving] {
        let last = tr.drag.len() - 1;
        println!(
            "{}: last drag {:.6}, last lift {:.6}, {} samples, {:.1} s ({:.2} ms/step), \
             continuity {:.3e}, min V/V0 {:.6}, boundary disp {:.3e}, upstream disp {:.3e}, \
             max disp {:.3e}",
            if tr.moving { "moving" } else { "static " },
            tr.drag[last], tr.lift[last], tr.drag.len(), tr.seconds,
            tr.seconds * 1.0e3 / tr.drag.len() as f64, tr.worst_continuity,
            tr.min_volume_ratio, tr.max_boundary_displacement,
            tr.max_upstream_displacement, tr.max_displacement
        );
    }
    println!("spin-up: {:.1} s", r.spin_seconds);
    for tr in [&r.static_run, &r.moving] {
        assert_eq!(tr.drag.len(), 100, "the window's samples");
        assert_eq!(tr.lift.len(), 100);
        for (d, l) in tr.drag.iter().zip(tr.lift.iter()) {
            assert!(d.is_finite() && l.is_finite(), "drag {d} lift {l}");
        }
    }
    assert_eq!(r.moving.max_boundary_displacement, 0.0);
    assert_eq!(r.moving.max_upstream_displacement, 0.0);
    assert!(r.moving.max_displacement > 0.0, "the wake wobbled");
    assert!(r.moving.min_volume_ratio > 0.8, "min V/V0 {}", r.moving.min_volume_ratio);
    assert_eq!(r.static_run.max_displacement, 0.0);
}

/// Three synthetic steady runs, finest first, whose drag and lift converge
/// to different limits: the drag study must read the drag and the lift
/// study the lift, whichever case the runs are.
#[test]
fn the_drag_study_reads_drag_and_the_lift_study_reads_lift() {
    let run = |h: Scalar, drag: Scalar, lift: Scalar| SteadyRun {
        case: Case::Cfd1,
        level: 1,
        n_cells: 1,
        h,
        iterations: 1,
        converged: true,
        residual: 0.0,
        forces: Forces { drag, lift, pressure: Vec3::ZERO, viscous: Vec3::ZERO },
        seconds: 0.0,
    };
    // value = limit - c h^2 on h = 1, 2, 4: Richardson recovers the limit.
    let runs = [run(1.0, 14.0 - 0.1, 1.1 - 0.01), run(2.0, 14.0 - 0.4, 1.1 - 0.04), run(4.0, 14.0 - 1.6, 1.1 - 0.16)];
    for case in [Case::Cfd1, Case::Cfd2] {
        let (d, l) = studies_of(case, &runs);
        let (d, l) = (d.expect("drag study"), l.expect("lift study"));
        assert!((d.phi_ext - 14.0).abs() < 1e-9, "{case:?}: drag study phi_ext {}", d.phi_ext);
        assert!((l.phi_ext - 1.1).abs() < 1e-9, "{case:?}: lift study phi_ext {}", l.phi_ext);
    }
}

/// Four synthetic runs, finest first: the three finest follow
/// `limit - c h^2` exactly, the coarsest is far off. The study takes the
/// three finest only, so Richardson recovers both limits exactly.
#[test]
fn the_study_takes_the_three_finest_of_four_levels() {
    let run = |h: Scalar, drag: Scalar, lift: Scalar| SteadyRun {
        case: Case::Cfd1,
        level: 1,
        n_cells: 1,
        h,
        iterations: 1,
        converged: true,
        residual: 0.0,
        forces: Forces { drag, lift, pressure: Vec3::ZERO, viscous: Vec3::ZERO },
        seconds: 0.0,
    };
    let runs = [
        run(1.0, 14.0 - 0.1, 1.1 - 0.01),
        run(2.0, 14.0 - 0.4, 1.1 - 0.04),
        run(4.0, 14.0 - 1.6, 1.1 - 0.16),
        run(8.0, 3.0, -7.0),
    ];
    let (d, l) = studies_of(Case::Cfd2, &runs);
    let (d, l) = (d.expect("drag study"), l.expect("lift study"));
    assert_eq!(d.n_levels, 3, "drag study n_levels {}", d.n_levels);
    assert_eq!(l.n_levels, 3, "lift study n_levels {}", l.n_levels);
    assert!((d.phi_ext - 14.0).abs() < 1e-9, "drag study phi_ext {}", d.phi_ext);
    assert!((l.phi_ext - 1.1).abs() < 1e-9, "lift study phi_ext {}", l.phi_ext);
    assert_eq!(STUDY_LEVELS, 3);
    assert_eq!(LEVEL_DIRS.len(), STEADY_MAX_ITERS.len());
}
