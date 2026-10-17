// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
//! The tests of SPEC-LIT section 161.8. No GPL-licensed source was consulted.

use super::*;
use crate::ale_flow::{buoyancy_off, flow_controls, laminar_fields, seat, set_kinds, Bc};
use crate::mesh::refined;
use crate::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use crate::timescheme::DdtScheme;

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The agreed example of SPEC-LIT 161.1, verbatim.
const AGREED: &str = r#"{
  "mass": 1600.0,
  "inertia": [600.0, 1850.0, 2100.0],
  "centreOfMass": [0.0, 0.0, 0.55],
  "velocity": [15.0, 0.0, 10.0],
  "angularVelocity": [0.0, 0.0, 0.0],
  "gravity": [0.0, 0.0, -9.81],
  "dof": "planarXZ",
  "bodyPatches": ["body", "jetFrontBottom", "jetRearTop", "jetFrontTop", "jetRearBottom"],
  "rho": 1.2,
  "thrusters": [
    {"patch": "jetFrontBottom", "direction": [0.0, 0.0, -1.0], "speed": 120.0, "on": [[0.0, 1.0]]},
    {"patch": "jetRearTop",     "direction": [0.0, 0.0,  1.0], "speed": 120.0, "on": [[0.0, 1.0]]},
    {"patch": "jetFrontTop",    "direction": [0.0, 0.0,  1.0], "speed": 120.0, "on": [[1.0, 2.0]]},
    {"patch": "jetRearBottom",  "direction": [0.0, 0.0, -1.0], "speed": 120.0, "on": [[1.0, 2.0]]}
  ],
  "write": {"every": 10, "file": "rigidBody.csv"}
}"#;

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

#[test]
fn the_agreed_file_parses_and_bad_files_are_refused_by_name() {
    let s = RigidBodySpec::parse(AGREED, "agreed").expect("the agreed file");
    assert_eq!(s.dof, Dof::PlanarXz);
    assert_eq!(s.body_patches.len(), 5);
    assert_eq!(s.thrusters.len(), 4);
    assert!(s.thrusters[0].is_on(0.0) && s.thrusters[0].is_on(0.999) && !s.thrusters[0].is_on(1.0));
    assert!(!s.thrusters[2].is_on(0.5) && s.thrusters[2].is_on(1.0));
    assert_eq!(s.write.every, 10);

    let refuse = |text: String, needle: &str| {
        let e = RigidBodySpec::parse(&text, "bad").expect_err(needle).to_string();
        assert!(e.contains(needle), "{needle}: {e}");
    };
    refuse(AGREED.replace("\"rho\": 1.2,", "\"rho\": 1.2, \"rhoo\": 1.0,"), "rhoo");
    refuse(AGREED.replace("\"velocity\": [15.0, 0.0, 10.0]", "\"velocity\": [15.0, 1.0, 10.0]"), "planarXZ");
    refuse(AGREED.replace("{\"patch\": \"jetRearTop\"", "{\"patch\": \"nozzle\""), "nozzle");
    refuse(AGREED.replace("\"mass\": 1600.0", "\"mass\": -1.0"), "mass");
    refuse(AGREED.replace("[600.0, 1850.0, 2100.0]", "[600.0, 1850.0, 9100.0]"), "triangle");
    refuse(AGREED.replace("\"on\": [[0.0, 1.0]]}", "\"on\": [[1.0, 0.0]]}"), "t0 < t1");
}

/// Free fall: no fluid force, gravity alone - the position is the parabola
/// to round-off at every step (SPEC-LIT 161.8 a).
#[test]
fn free_fall_is_exact() {
    for dof in [Dof::Free, Dof::PlanarXz] {
        let (c0, v0, g) = ([0.0, 0.0, 0.55], [15.0, 0.0, 10.0], [0.0, 0.0, -9.81]);
        let mut b = RigidBody::from_parts(1600.0, [600.0, 1850.0, 2100.0], c0, v0, [0.0; 3], g, dof);
        let dt = 1e-3;
        let mut worst = 0.0f64;
        for n in 1..=2000 {
            b.advance(dt, [0.0; 3], [0.0; 3]);
            let t = n as f64 * dt;
            for i in 0..3 {
                let x = c0[i] + v0[i] * t + 0.5 * g[i] * t * t;
                let v = v0[i] + g[i] * t;
                worst = worst.max((b.state.x[i] - x).abs() / (1.0 + x.abs()));
                worst = worst.max((b.state.v[i] - v).abs() / (1.0 + v.abs()));
            }
        }
        println!("free fall ({}): worst relative error {worst:.3e} over 2 s", dof.name());
        assert!(worst <= 1e-12, "{}: {worst:e}", dof.name());
        assert_eq!(b.state.q, [1.0, 0.0, 0.0, 0.0]);
    }
}

/// A constant moment about y from rest: `w = M t/I`, `theta = M t^2/(2 I)`,
/// past two full turns (SPEC-LIT 161.8 b).
#[test]
fn a_constant_torque_spins_up_as_m_t2_over_2i() {
    for dof in [Dof::Free, Dof::PlanarXz] {
        let inertia = [600.0, 1850.0, 2100.0];
        let m = 6000.0;
        let mut b = RigidBody::from_parts(1600.0, inertia, [0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3], dof);
        let dt = 1e-3;
        let steps = 3000;
        for _ in 0..steps {
            b.advance(dt, [0.0; 3], [0.0, m, 0.0]);
        }
        let t = steps as f64 * dt;
        let w = m * t / inertia[1];
        let theta = m * t * t / (2.0 * inertia[1]);
        let q = b.state.q;
        // q = (cos(theta/2), 0, sin(theta/2), 0), up to the sign of a quaternion.
        let sgn = if q[0] * (0.5 * theta).cos() + q[2] * (0.5 * theta).sin() < 0.0 { -1.0 } else { 1.0 };
        let eq = ((sgn * q[0] - (0.5 * theta).cos()).abs())
            .max((sgn * q[2] - (0.5 * theta).sin()).abs())
            .max(q[1].abs())
            .max(q[3].abs());
        println!(
            "spin-up ({}): theta {:.12} rad (exact {:.12}, {:.1} deg), w {:.12} (exact {:.12}), |q - q_exact| {eq:.3e}",
            dof.name(),
            b.state.rot_y,
            theta,
            theta.to_degrees(),
            b.state.w[1],
            w
        );
        assert!(theta > 4.0 * std::f64::consts::PI, "the test must pass two full turns");
        assert!(rel(b.state.w[1], w) <= 1e-12, "w {} vs {w}", b.state.w[1]);
        assert!(rel(b.state.rot_y, theta) <= 1e-12, "theta {} vs {theta}", b.state.rot_y);
        assert!(eq <= 1e-10, "quaternion off by {eq:e}");
        assert!(b.state.w[0].abs() + b.state.w[2].abs() == 0.0);
    }
}

/// The torque-free symmetric top: `|w|`, the kinetic energy and the world
/// angular momentum are invariants, and `w` precesses about the symmetry
/// axis at `(I3 - I1)/I1 w3` (SPEC-LIT 161.8 c).
#[test]
fn a_torque_free_symmetric_top_keeps_its_energy_and_spin() {
    let inertia = [1.0, 1.0, 2.0];
    let w0 = [1.0, 0.5, 2.0];
    let mut b = RigidBody::from_parts(1.0, inertia, [0.0; 3], [0.0; 3], w0, [0.0; 3], Dof::Free);
    let (ke0, l0, wn0) = (b.kinetic_energy(), b.angular_momentum_world(), norm(w0));
    let dt = 1e-3;
    let steps = 10_000;
    for _ in 0..steps {
        b.advance(dt, [0.0; 3], [0.0; 3]);
    }
    let t = steps as f64 * dt;
    let om = (inertia[2] - inertia[0]) / inertia[0] * w0[2];
    let (c, s) = ((om * t).cos(), (om * t).sin());
    let w_exact = [w0[0] * c - w0[1] * s, w0[0] * s + w0[1] * c, w0[2]];
    let ew = norm(sub(b.state.w, w_exact)) / wn0;
    let eke = rel(b.kinetic_energy(), ke0);
    let ewn = rel(norm(b.state.w), wn0);
    let el = norm(sub(b.angular_momentum_world(), l0)) / norm(l0);
    println!(
        "symmetric top over {t} s: |w| rel {ewn:.3e}, KE rel {eke:.3e}, w vs the precession {ew:.3e}, L_world rel {el:.3e}"
    );
    assert!(ewn <= 1e-10, "|w| drifted {ewn:e}");
    assert!(eke <= 1e-10, "kinetic energy drifted {eke:e}");
    assert!(ew <= 1e-9, "w left the precession by {ew:e}");
    assert!(el <= 1e-6, "the world angular momentum drifted {el:e}");
}

/// A small box of `n^3` cells of side `d`, every patch generic.
fn box_mesh(n: usize, d: f64) -> refined::RefinedBox {
    let mut rb = refined::build([n, n, n], Vec3::new(d as Scalar, d as Scalar, d as Scalar), &vec![0u32; n * n * n])
        .expect("box");
    set_kinds(&mut rb, [PatchKind::Generic; 6]).expect("kinds");
    rb
}

fn geo(p: GeoVec3) -> V3 {
    [f64::from(p.x), f64::from(p.y), f64::from(p.z)]
}

/// The device point map against the host map, through a local origin far
/// from zero (SPEC-LIT 161.8 d).
#[test]
fn the_device_point_map_is_the_host_map() {
    let Some(gpu) = gpu() else { return };
    let rb = box_mesh(3, 0.4);
    let origin = DVec3::new(1000.5, -250.25, 37.0);
    let n = rb.points.len();
    let dev = RigidBodyDevice::new(&gpu, &rb.points, origin, &[], &[], 0, rb.mesh.n_boundary_faces).expect("device");
    let mut b = RigidBody::from_parts(
        2.0,
        [1.0, 2.0, 2.5],
        [1000.9, -249.7, 37.4],
        [1.0, -2.0, 3.0],
        [0.3, -1.2, 0.7],
        [0.0, 0.0, -9.81],
        Dof::Free,
    );
    for _ in 0..7 {
        b.advance(0.1, [0.5, 0.0, 1.0], [0.0, 0.2, -0.1]);
    }
    let mut pts: DevBuf<GeoVec3> = gpu.zeros(n).expect("points");
    dev.write_points(&gpu, &mut pts, &b).expect("write");
    let got = gpu.download(&pts).expect("download");
    let o = [origin.x, origin.y, origin.z];
    let mut worst = 0.0f64;
    for (p, g) in rb.points.iter().zip(&got) {
        let w = b.map_point(dvec(p.to_dvec3()));
        let want = sub(w, o);
        let have = geo(*g);
        for i in 0..3 {
            worst = worst.max((have[i] - want[i]).abs() / want[i].abs().max(1.0));
        }
    }
    println!("device point map vs host: worst relative difference {worst:.3e} over {n} points");
    assert!(worst <= 1e-12, "{worst:e}");
}

/// The force reduction against closed forms on the closed surface of a unit
/// box: a uniform `p` sums to nothing, `p = g.x` to `rho g V`, the
/// hydrostatic term the same, the wall shear of one sliding face, and one
/// face's jet reaction (SPEC-LIT 161.8 e).
#[test]
fn the_force_reduction_meets_its_closed_forms() {
    let Some(gpu) = gpu() else { return };
    let rb = box_mesh(4, 0.25);
    let hm = &rb.mesh;
    let gm = GpuMesh::upload(&gpu, hm).expect("upload");
    let nbf = hm.n_boundary_faces;
    let faces: Vec<Label> = (0..nbf as Label).collect();
    let jets = vec![-1 as Label; nbf];
    let mut dev = RigidBodyDevice::new(&gpu, &rb.points, hm.origin, &faces, &jets, 0, nbf).expect("device");
    let b = RigidBody::from_parts(1.0, [1.0; 3], [0.5, 0.5, 0.5], [0.0; 3], [0.0; 3], [0.0; 3], Dof::Free);
    let (rho, nu) = (1.2, 0.01);
    let g = [0.0, 0.0, -9.81];
    let tol = if cfg!(feature = "single") { 1e-5 } else { 1e-12 };
    let patch = |name: &str| hm.patches.iter().find(|p| p.name == name).expect("patch").clone();
    let zeros_v = vec![Vec3::ZERO; nbf];
    let zeros_s = vec![0.0 as Scalar; nbf];
    let u: DevBuf<Vec3> = gpu.zeros(hm.n_cells).expect("u");
    let run = |dev: &mut RigidBodyDevice, p: &[Scalar], ub: &[Vec3], phi: &[Scalar], gh: V3| -> Wrench {
        let pb = gpu.upload(p).expect("p");
        let ubb = gpu.upload(ub).expect("ub");
        let phib = gpu.upload(phi).expect("phi");
        dev.force_moment(&gpu, &gm, &pb, &u, &ubb, &phib, rho, nu, gh, &b).expect("reduce")
    };

    // (a) uniform p
    let w = run(&mut dev, &vec![3.7 as Scalar; nbf], &zeros_v, &zeros_s, [0.0; 3]);
    let scale_f = rho * 3.7 * 6.0;
    println!("uniform p: F {:?} M {:?}", w.force(), w.moment);
    assert!(norm(w.force()) <= tol * scale_f && norm(w.moment) <= tol * scale_f, "{w:?}");

    // (b) p = g.(x - X), kinematic: F = rho g V, M = 0 about the centroid
    let cf = &hm.b_cf;
    let p_h: Vec<Scalar> = cf
        .iter()
        .map(|c| (g[2] * (f64::from(c.z) - 0.5)) as Scalar)
        .collect();
    let w = run(&mut dev, &p_h, &zeros_v, &zeros_s, [0.0; 3]);
    let want = rho * g[2];
    println!("p = rho g z: F {:?} (want [0, 0, {want}]) M {:?}", w.force(), w.moment);
    assert!(rel(w.force()[2], want) <= tol && w.force()[0].abs() + w.force()[1].abs() <= tol * want.abs());
    assert!(norm(w.moment) <= tol * want.abs(), "{:?}", w.moment);

    // (c) p = 0 and the hydrostatic term restored by g
    let w = run(&mut dev, &zeros_s, &zeros_v, &zeros_s, g);
    println!("hydrostatic term: F {:?} M {:?}", w.force(), w.moment);
    assert!(rel(w.force()[2], want) <= tol && norm(w.moment) <= tol * want.abs());

    // (d) the top face slides at (1, 0, 0) over a fluid at rest
    let top = patch("zmax");
    let mut ub = zeros_v.clone();
    for k in 0..top.size {
        ub[top.start + k] = Vec3::new(1.0, 0.0, 0.0);
    }
    let w = run(&mut dev, &zeros_s, &ub, &zeros_s, [0.0; 3]);
    let fv = -rho * nu * 1.0 * 8.0;
    println!("sliding lid: Fv {:?} (want [{fv}, 0, 0]) M {:?} (want [0, {}, 0])", w.viscous, w.moment, 0.5 * fv);
    assert!(rel(w.viscous[0], fv) <= tol && w.viscous[1].abs() + w.viscous[2].abs() <= tol * fv.abs());
    assert!(rel(w.moment[1], 0.5 * fv) <= tol && w.moment[0].abs() + w.moment[2].abs() <= tol * fv.abs());

    // (e) the bottom face blows (0, 0, 2) into the fluid: phi_rel = 2 * (-0.0625)
    let bot = patch("zmin");
    let mut ub = zeros_v.clone();
    let mut phi = zeros_s.clone();
    for k in 0..bot.size {
        ub[bot.start + k] = Vec3::new(0.0, 0.0, 2.0);
        phi[bot.start + k] = -0.125;
    }
    let w = run(&mut dev, &zeros_s, &ub, &phi, [0.0; 3]);
    let fj = rho * (-0.125) * 2.0 * 16.0;
    println!("jet: Fj {:?} (want [0, 0, {fj}]) Fv {:?} M {:?}", w.jet, w.viscous, w.moment);
    assert!(rel(w.jet[2], fj) <= tol && w.jet[0].abs() + w.jet[1].abs() <= tol * fj.abs());
    assert!(norm(w.viscous) <= tol * fj.abs() && norm(w.moment) <= tol * fj.abs());
}

/// Free-stream preservation: a box whose every face prescribes the same
/// absolute velocity, moved rigidly - translating and turning through about
/// 70 degrees - keeps that velocity to round-off, because the mesh flux
/// closes each cell's volume change (SPEC-LIT 161.8 f).
#[test]
fn a_rigidly_moving_box_preserves_a_free_stream() {
    let Some(gpu) = gpu() else { return };
    let rb = box_mesh(6, 1.0 / 6.0);
    let hm = &rb.mesh;
    let gm = GpuMesh::upload(&gpu, hm).expect("upload");
    let csr = flatten_faces(&rb.faces);
    let ale = AleMesh::new(&gpu, hm, &gm, &rb.points, &csr).expect("ale");
    let dt = 0.05;
    let ctrl = flow_controls(DdtScheme::Euler, dt as Scalar, false);
    let mut s = Simple::new(&gpu, hm, &gm, ctrl, buoyancy_off()).expect("simple");
    let u0 = Vec3::new(0.7, -0.2, 0.4);
    seat(&gpu, &mut s, hm, &gm, &[Bc::Value(u0); 6], &[Bc::ZeroGradient; 6], u0).expect("seat");
    s.attach_motion(&gpu, ale, &[]).expect("attach");
    let mut backend = PbicgstabBackend::new(ctrl.p_solver);
    backend.setup(&gpu, hm, &gm, &SystemProbe::default()).expect("backend");
    let (nut, t) = laminar_fields(&gpu, hm, &gm).expect("fields");
    let dev = RigidBodyDevice::new(&gpu, &rb.points, hm.origin, &[], &[], 0, hm.n_boundary_faces).expect("device");
    let mut body = RigidBody::from_parts(
        1.0,
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
        [0.3, 0.1, -0.2],
        [0.4, 1.0, -0.6],
        [0.0; 3],
        Dof::Free,
    );
    let mut worst = 0.0f64;
    let steps = 20;
    for k in 0..steps {
        if k > 0 {
            s.begin_time_step(&gpu, dt as Scalar).expect("begin");
        }
        body.advance(dt, [0.0; 3], [0.0; 3]);
        let ale = s.motion_mut().expect("attached");
        dev.write_points(&gpu, &mut ale.points, &body).expect("points");
        s.move_mesh(&gpu).expect("move");
        s.solve_step(&gpu, &mut backend, &nut, &t).expect("solve");
        let u = gpu.download(&s.u().f).expect("u");
        for c in u.iter().take(hm.n_cells) {
            worst = worst.max(f64::from((*c - u0).mag()) / f64::from(u0.mag()));
        }
    }
    let turned = norm(body.state.w) * dt * steps as f64;
    let tol = if cfg!(feature = "single") { 1e-4 } else { 1e-9 };
    println!(
        "free stream on a rigidly moving box: {steps} steps, turned {:.1} deg, max |u - u0|/|u0| = {worst:.3e} (tolerance {tol:e})",
        turned.to_degrees()
    );
    assert!(turned > 1.0, "the box must turn through a large angle");
    assert!(worst <= tol, "{worst:e}");
}
