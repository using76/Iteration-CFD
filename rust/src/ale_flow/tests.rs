// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
//! SPEC-LIT section 105.10. No GPL-licensed source was consulted.

use super::*;
use crate::capture::capture_replays_bitwise;
use crate::solver::{LinearSolverKind, Preconditioner};

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The piston's velocity and pressure boundaries and its seed state.
fn seat_piston(
    gpu: &Gpu,
    s: &mut Simple<'_>,
    hm: &HostMesh,
    gm: &GpuMesh,
) -> Result<()> {
    seat(
        gpu,
        s,
        hm,
        gm,
        &[
            Bc::ZeroGradient,
            Bc::Value(Vec3::new(-PISTON_C, 0.0, 0.0)),
            Bc::Symmetry,
            Bc::Symmetry,
            Bc::Symmetry,
            Bc::Symmetry,
        ],
        &[
            Bc::Value(0.0),
            Bc::ZeroGradient,
            Bc::Symmetry,
            Bc::Symmetry,
            Bc::Symmetry,
            Bc::Symmetry,
        ],
        Vec3::new(-PISTON_C, 0.0, 0.0),
    )
}

/// The stroke's velocity and pressure boundaries and its seed state.
fn seat_stroke(
    gpu: &Gpu,
    s: &mut Simple<'_>,
    hm: &HostMesh,
    gm: &GpuMesh,
) -> Result<()> {
    seat(
        gpu,
        s,
        hm,
        gm,
        &[
            Bc::ZeroGradient,
            Bc::ZeroGradient,
            Bc::Symmetry,
            Bc::Symmetry,
            Bc::Symmetry,
            Bc::Symmetry,
        ],
        &[
            Bc::Value(STROKE_P0),
            Bc::Value(0.0),
            Bc::Symmetry,
            Bc::Symmetry,
            Bc::Symmetry,
            Bc::Symmetry,
        ],
        Vec3::new(STROKE_U0, 0.0, 0.0),
    )
}

/// F1 - the OPEN piston keeps its uniform state through the whole SIMPLE
/// loop, euler and backward, in both convective forms.
#[test]
fn the_piston_keeps_the_uniform_state_through_the_simple_loop() {
    let Some(gpu) = gpu() else { return };
    for &scheme in &[DdtScheme::Euler, DdtScheme::Backward] {
        for &bounded in &[false, true] {
            let r = piston(&gpu, scheme, bounded).expect("piston run");
            println!(
                "  piston {scheme:?} bounded={bounded}: steps {} cells {} \
                 worst_u {:.3e} worst_p {:.3e} wall_flux {:.3e} continuity {:.3e} \
                 min V/V0 {:.10e} stroke {:.6}",
                r.steps,
                r.n_cells,
                r.worst_u,
                r.worst_p,
                r.worst_wall_flux,
                r.worst_continuity,
                r.min_volume_ratio,
                r.stroke
            );
            assert!(r.worst_u <= UNIFORM_TOL, "worst_u {:.3e}", r.worst_u);
            assert!(r.worst_p <= UNIFORM_TOL, "worst_p {:.3e}", r.worst_p);
            assert!(
                r.worst_wall_flux <= WALL_FLUX_TOL,
                "wall flux {:.3e}",
                r.worst_wall_flux
            );
            assert!(
                r.worst_continuity <= CONTINUITY_TOL,
                "continuity {:.3e}",
                r.worst_continuity
            );
            assert_eq!(r.steps, PISTON_STEPS);
            assert_eq!(r.n_cells, 72);
            assert!(
                r.min_volume_ratio > 0.5,
                "min V/V0 {:.10e}",
                r.min_volume_ratio
            );
            assert!((r.stroke - 0.1).abs() < 1e-12);
        }
    }
}

/// F2 - the stroking outlet's time order, three step counts through the grid
/// study, and the extrapolation toward the exact value.
#[test]
fn gate_105b_the_stroking_outlet_has_the_scheme_time_order() {
    let Some(gpu) = gpu() else { return };
    for &scheme in &[DdtScheme::Euler, DdtScheme::Backward] {
        let st = stroke_study(&gpu, scheme).expect("stroke study");
        for r in &st.runs {
            println!(
                "  stroke {scheme:?}: steps {} dt {:.10e} value {:.10e} \
                 spread {:.3e} min V/V0 {:.10e}",
                r.steps, r.dt, r.value, r.spread, r.min_volume_ratio
            );
        }
        println!(
            "  stroke {scheme:?}: exact {:.10e} p {:.6e} err_fine {:.3e} err_ext {:.3e}",
            st.exact, st.p, st.err_fine, st.err_ext
        );
        assert!(
            (st.p - expected_order(scheme)).abs() <= ORDER_BAND,
            "observed order p {:.6e}",
            st.p
        );
        assert!(
            st.err_fine <= fine_tolerance(scheme),
            "finest error {:.3e}",
            st.err_fine
        );
        assert!(
            st.err_ext < st.err_fine,
            "err_ext {:.3e} not below err_fine {:.3e}",
            st.err_ext,
            st.err_fine
        );
    }
}

/// F3 - after one moving step the resident relative flux is `phi - phi_mesh`
/// BIT FOR BIT on every face, and the wall value the device wrote is
/// `b_sf (phi_mesh_b / (b_sf . b_sf))` bit for bit, in all three components.
#[test]
fn the_relative_flux_and_the_wall_value_are_what_they_say_bitwise() {
    let Some(gpu) = gpu() else { return };
    let rig = piston_rig().expect("piston rig");
    let hm = &rig.rb.mesh;
    let gm = GpuMesh::upload(&gpu, hm).expect("mesh");
    let ale = AleMesh::new(&gpu, hm, &gm, &rig.rb.points, &rig.csr).expect("ale");
    let ctrl = flow_controls(DdtScheme::Backward, PISTON_DT, false);
    let mut s = Simple::new(&gpu, hm, &gm, ctrl, buoyancy_off()).expect("simple");
    seat_piston(&gpu, &mut s, hm, &gm).expect("seat");
    s.attach_motion(&gpu, ale, &rig.wall_faces).expect("attach");
    let mut backend = PbicgstabBackend::new(ctrl.p_solver);
    backend
        .setup(&gpu, hm, &gm, &SystemProbe::default())
        .expect("backend");
    let (nut, t) = laminar_fields(&gpu, hm, &gm).expect("fields");
    take_moving_step(&gpu, &mut s, &mut backend, &rig.motion, 0, PISTON_DT, &nut, &t)
        .expect("one moving step");

    let phi_f = gpu.download(&s.phi().f).expect("phi");
    let phi_b = gpu.download(&s.phi().bf).expect("phi b");
    let rel_f = gpu.download(&s.convective_flux().f).expect("phi_rel");
    let rel_b = gpu.download(&s.convective_flux().bf).expect("phi_rel b");
    let ale = s.motion().expect("attached");
    let pm_f = gpu.download(&ale.phi_mesh.f).expect("phi_mesh");
    let pm_b = gpu.download(&ale.phi_mesh.bf).expect("phi_mesh b");
    for f in 0..rel_f.len() {
        let want = phi_f[f] - pm_f[f];
        assert_eq!(rel_f[f].to_bits(), want.to_bits(), "internal face {f}");
    }
    for i in 0..rel_b.len() {
        let want = phi_b[i] - pm_b[i];
        assert_eq!(rel_b[i].to_bits(), want.to_bits(), "boundary face {i}");
    }

    let b_sf = gpu.download(&gm.b_sf).expect("b_sf");
    let rv = gpu.download(&s.u().ref_value).expect("ref_value");
    let mut worst_wall = 0.0 as Scalar;
    for &f in &rig.wall_faces {
        let i = f as usize;
        let sfi = b_sf[i];
        let s2 = sfi.dot(sfi);
        let want = sfi * (pm_b[i] / s2);
        let got = rv[i];
        assert_eq!(got.x.to_bits(), want.x.to_bits(), "wall face {i} x");
        assert_eq!(got.y.to_bits(), want.y.to_bits(), "wall face {i} y");
        assert_eq!(got.z.to_bits(), want.z.to_bits(), "wall face {i} z");
        worst_wall = worst_wall.max(((phi_b[i] - pm_b[i]) / pm_b[i]).abs());
    }
    println!(
        "  largest wall-flux imbalance over the piston's faces: {worst_wall:.3e} of the mesh flux"
    );
    assert!(worst_wall <= WALL_FLUX_TOL, "wall imbalance {worst_wall:.3e}");

    // flow_state hands out the relative flux while a motion is attached, and
    // the absolute one when none is.
    assert!(std::ptr::eq(s.flow_state().phi, s.convective_flux()));
    assert!(!std::ptr::eq(s.convective_flux(), s.phi()));
    let s2 = Simple::new(&gpu, hm, &gm, ctrl, buoyancy_off()).expect("second simple");
    assert!(std::ptr::eq(s2.flow_state().phi, s2.phi()));
    assert!(std::ptr::eq(s2.convective_flux(), s2.phi()));
}

/// F4 - a mesh that is attached and NEVER moves gives the static answer: the
/// same ten steps with and without a motion, fields agreeing to 1e-12.
#[test]
fn a_mesh_that_is_attached_and_never_moves_gives_the_static_answer() {
    let Some(gpu) = gpu() else { return };
    let rig = stroke_rig(PatchMotion::Fixed).expect("never-moving rig");
    let hm = &rig.rb.mesh;
    let gm = GpuMesh::upload(&gpu, hm).expect("mesh");
    let scheme = DdtScheme::Backward;
    let dt = 0.01;
    let steps = 10;

    let download_all = |s: &Simple| {
        (
            gpu.download(&s.u().f).expect("U"),
            gpu.download(&s.p().f).expect("p"),
            gpu.download(&s.phi().f).expect("phi"),
            gpu.download(&s.phi().bf).expect("phi b"),
        )
    };

    // Run A: no motion at all.
    let a = {
        let ctrl = flow_controls(scheme, dt, false);
        let mut s = Simple::new(&gpu, hm, &gm, ctrl, buoyancy_off()).expect("run A");
        seat_stroke(&gpu, &mut s, hm, &gm).expect("seat A");
        let mut backend = PbicgstabBackend::new(ctrl.p_solver);
        backend
            .setup(&gpu, hm, &gm, &SystemProbe::default())
            .expect("backend A");
        let (nut, t) = laminar_fields(&gpu, hm, &gm).expect("fields A");
        for k in 0..steps {
            if k > 0 {
                s.begin_time_step(&gpu, dt).expect("begin A");
            }
            s.solve_step(&gpu, &mut backend, &nut, &t).expect("step A");
        }
        download_all(&s)
    };

    // Run B: the same run with a motion attached whose law never moves.
    let b = {
        let ale = AleMesh::new(&gpu, hm, &gm, &rig.rb.points, &rig.csr).expect("ale");
        let ctrl = flow_controls(scheme, dt, false);
        let mut s = Simple::new(&gpu, hm, &gm, ctrl, buoyancy_off()).expect("run B");
        seat_stroke(&gpu, &mut s, hm, &gm).expect("seat B");
        s.attach_motion(&gpu, ale, &[]).expect("attach B");
        let mut backend = PbicgstabBackend::new(ctrl.p_solver);
        backend
            .setup(&gpu, hm, &gm, &SystemProbe::default())
            .expect("backend B");
        let (nut, t) = laminar_fields(&gpu, hm, &gm).expect("fields B");
        for k in 0..steps {
            take_moving_step(&gpu, &mut s, &mut backend, &rig.motion, k, dt, &nut, &t)
                .expect("step B");
        }
        download_all(&s)
    };

    let rel = |x: &[Vec3], y: &[Vec3]| {
        let scale = x.iter().map(|v| v.mag()).fold(0.0 as Scalar, Scalar::max);
        x.iter()
            .zip(y)
            .map(|(a, b)| (a.x - b.x).abs().max((a.y - b.y).abs()).max((a.z - b.z).abs()))
            .fold(0.0 as Scalar, Scalar::max)
            / scale
    };
    let rel_s = |x: &[Scalar], y: &[Scalar]| {
        let scale = x.iter().fold(0.0 as Scalar, |a, v| a.max(v.abs()));
        x.iter()
            .zip(y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0 as Scalar, Scalar::max)
            / scale
    };
    let r_u = rel(&a.0, &b.0);
    let r_p = rel_s(&a.1, &b.1);
    let r_phi = rel_s(&a.2, &b.2).max(rel_s(&a.3, &b.3));
    println!("  static vs never-moving: U {r_u:.3e}, p {r_p:.3e}, phi {r_phi:.3e}");
    assert!(r_u <= 1e-12, "U {r_u:.3e}");
    assert!(r_p <= 1e-12, "p {r_p:.3e}");
    assert!(r_phi <= 1e-12, "phi {r_phi:.3e}");
}

/// F5 - `attach_motion` and `move_mesh` refuse, each by name. (c) - the fan
/// and porous-jump refusal - is NOT exercised here: building
/// `FlowDevices` needs a fan case, and that refusal is covered by reading
/// `attach_motion` and `move_mesh`, which test it before anything else.
#[test]
fn the_motion_refusals_name_what_they_refuse() {
    let Some(gpu) = gpu() else { return };

    // (b) a steady run, `SteadyState`
    {
        let rig = stroke_rig(stroke_law()).expect("rig");
        let hm = &rig.rb.mesh;
        let gm = GpuMesh::upload(&gpu, hm).expect("mesh");
        let ale = AleMesh::new(&gpu, hm, &gm, &rig.rb.points, &rig.csr).expect("ale");
        let mut ctrl = flow_controls(DdtScheme::SteadyState, 0.01, false);
        ctrl.momentum.steady = true;
        let mut s = Simple::new(&gpu, hm, &gm, ctrl, buoyancy_off()).expect("simple");
        let e = s.attach_motion(&gpu, ale, &[]).expect_err("steady refused");
        let msg = e.to_string();
        assert!(
            msg.contains("Simple::attach_motion: a moving mesh needs `Euler` or `backward`"),
            "(b): {msg}"
        );
    }

    // (a) one Simple moves one mesh
    {
        let rig = stroke_rig(stroke_law()).expect("rig");
        let hm = &rig.rb.mesh;
        let gm = GpuMesh::upload(&gpu, hm).expect("mesh");
        let ale = AleMesh::new(&gpu, hm, &gm, &rig.rb.points, &rig.csr).expect("ale");
        let ctrl = flow_controls(DdtScheme::Euler, 0.01, false);
        let mut s = Simple::new(&gpu, hm, &gm, ctrl, buoyancy_off()).expect("simple");
        seat_stroke(&gpu, &mut s, hm, &gm).expect("seat");
        s.attach_motion(&gpu, ale, &[]).expect("first attach");
        let ale2 = AleMesh::new(&gpu, hm, &gm, &rig.rb.points, &rig.csr).expect("ale 2");
        let e = s.attach_motion(&gpu, ale2, &[]).expect_err("second attach");
        let msg = e.to_string();
        assert!(
            msg.contains("Simple::attach_motion: a motion is already attached"),
            "(a): {msg}"
        );
    }

    // (d) an AleMesh built for another mesh
    {
        let piston = piston_rig().expect("piston rig");
        let stroke = stroke_rig(stroke_law()).expect("stroke rig");
        let gm_p = GpuMesh::upload(&gpu, &piston.rb.mesh).expect("piston mesh");
        let ale_p = AleMesh::new(&gpu, &piston.rb.mesh, &gm_p, &piston.rb.points, &piston.csr)
            .expect("piston ale");
        let hm_s = &stroke.rb.mesh;
        let gm_s = GpuMesh::upload(&gpu, hm_s).expect("stroke mesh");
        let ctrl = flow_controls(DdtScheme::Euler, 0.01, false);
        let mut s = Simple::new(&gpu, hm_s, &gm_s, ctrl, buoyancy_off()).expect("simple");
        seat_stroke(&gpu, &mut s, hm_s, &gm_s).expect("seat");
        let e = s.attach_motion(&gpu, ale_p, &[]).expect_err("foreign AleMesh");
        let msg = e.to_string();
        assert!(
            msg.contains("Simple::attach_motion: the AleMesh was built for"),
            "(d): {msg}"
        );
    }

    // (e) a wall face that is not a boundary face
    {
        let rig = stroke_rig(stroke_law()).expect("rig");
        let hm = &rig.rb.mesh;
        let gm = GpuMesh::upload(&gpu, hm).expect("mesh");
        let ale = AleMesh::new(&gpu, hm, &gm, &rig.rb.points, &rig.csr).expect("ale");
        let ctrl = flow_controls(DdtScheme::Euler, 0.01, false);
        let mut s = Simple::new(&gpu, hm, &gm, ctrl, buoyancy_off()).expect("simple");
        seat_stroke(&gpu, &mut s, hm, &gm).expect("seat");
        let e = s
            .attach_motion(&gpu, ale, &[hm.n_boundary_faces as Label])
            .expect_err("face out of range");
        let msg = e.to_string();
        assert!(msg.contains("Simple::attach_motion: wall face"), "(e): {msg}");
    }

    // (f) move_mesh with nothing attached
    {
        let rig = stroke_rig(stroke_law()).expect("rig");
        let hm = &rig.rb.mesh;
        let gm = GpuMesh::upload(&gpu, hm).expect("mesh");
        let ctrl = flow_controls(DdtScheme::Euler, 0.01, false);
        let mut s = Simple::new(&gpu, hm, &gm, ctrl, buoyancy_off()).expect("simple");
        seat_stroke(&gpu, &mut s, hm, &gm).expect("seat");
        let e = s.move_mesh(&gpu).expect_err("no motion attached");
        let msg = e.to_string();
        assert!(
            msg.contains("Simple::move_mesh: no motion is attached"),
            "(f): {msg}"
        );
    }
}

/// F6 - ONE moving step, `move_mesh` plus one outer corrector, captures and
/// replays BITWISE (SPEC-LIT 81.7): the same run through a CUDA graph, with
/// the points moved by device copies and the solvers in fixed-iteration mode,
/// reproduces every buffer the step wrote.
#[test]
fn a_moving_step_replays_bitwise() {
    let Some(g) = gpu() else { return };
    let rig = stroke_rig(stroke_law()).expect("rig");
    let hm = &rig.rb.mesh;
    let n = hm.n_cells;

    // The capture-safe controls of simple.rs's outer-corrector gate: fixed
    // iteration counts, no residual report, no continuity read-back - the
    // three host round-trips a capture refuses.
    let solver = SolverControls {
        solver: LinearSolverKind::PBiCGStab, // never PCG: its symmetry check downloads
        precon: Preconditioner::Diagonal, // never DIC: same read-back (§8.2)
        tolerance: 1e-14,
        rel_tol: 0.0,
        max_iter: 4,
        min_iter: 0,
        check_interval: 1,
        fixed_iters: true, // an adaptive residual test DMAs a flag to the host every
        report_residuals: false, // check_interval sweeps, and the guard refuses that inside a capture
        ..SolverControls::default()
    };
    let mut ctrl = flow_controls(DdtScheme::Euler, 0.01, false);
    ctrl.momentum.u_solver = solver;
    ctrl.p_solver = solver;
    ctrl.report_continuity = false;
    let buoy = buoyancy_off();

    // ONE outer mesh: the reset puts its geometry back at rest in place,
    // through the shared borrow, so every build starts where the last ended.
    let gm = GpuMesh::upload(&g, hm).expect("mesh");

    let build = || -> Result<(Simple, PbicgstabBackend, crate::DevBuf<Vec3>)> {
        let mut reset = AleMesh::new(&g, hm, &gm, &rig.rb.points, &rig.csr)?;
        reset.recompute_in_place(&g, &gm)?;
        drop(reset);
        let ale = AleMesh::new(&g, hm, &gm, &rig.rb.points, &rig.csr)?;
        let mut s = Simple::new(&g, hm, &gm, ctrl, buoy)?;
        seat_stroke(&g, &mut s, hm, &gm)?;
        s.attach_motion(&g, ale, &[])?;
        let mut backend = PbicgstabBackend::new(ctrl.p_solver);
        backend.setup(&g, hm, &gm, &SystemProbe::default())?;
        let spare = g.upload(&rig.motion.points_at(0.125))?;
        Ok((s, backend, spare))
    };

    let (nut, t) = laminar_fields(&g, hm, &gm).expect("fields");

    // The points alternate moved, rest, moved, ... with FIXED device
    // pointers: a device copy moves them, a device copy backs the old level
    // up, and `advance` does the rest - no host write anywhere in the step.
    let iterate = |(s, b, spare): &mut (Simple, PbicgstabBackend, crate::DevBuf<Vec3>)| {
        {
            let ale = s.motion_mut().expect("attached");
            g.stream().memcpy_dtod(&*spare, &mut ale.points)?;
            g.stream().memcpy_dtod(&ale.points_old, &mut *spare)?;
        }
        s.move_mesh(&g)?;
        s.correct_outer(&g, b, &nut, &t, true).map(|_| ())
    };

    // SPEC-LIT 81.5's guard: is there anything for the gate to compare?
    // Three moving steps must move the velocity in more than half the cells,
    // or a graph that recorded no work at all would pass.
    {
        let (s, b, spare) = build().expect("build");
        let mut run = (s, b, spare);
        let before = g.download(&run.0.u().f).expect("U before");
        for _ in 0..3 {
            iterate(&mut run).expect("iterate");
        }
        let after = g.download(&run.0.u().f).expect("U after");
        let moved = before
            .iter()
            .zip(&after)
            .filter(|(a, b)| {
                a.x.to_bits() != b.x.to_bits()
                    || a.y.to_bits() != b.y.to_bits()
                    || a.z.to_bits() != b.z.to_bits()
            })
            .count();
        assert!(
            moved > n / 2,
            "only {moved} of {n} cells moved in three moving steps: a bitwise \
             replay over a field that does not move holds for a graph that \
             launched nothing - SPEC-LIT 81.5"
        );
    }

    let report = capture_replays_bitwise(
        &g,
        "moving step: move_mesh plus outer corrector (SPEC-LIT 105.10)",
        build,
        iterate,
        |(s, _, _): &(Simple, PbicgstabBackend, crate::DevBuf<Vec3>)| {
            let u = g.download(&s.u().f)?;
            let mut flat = Vec::with_capacity(u.len() * 3);
            for c in &u {
                flat.push(c.x);
                flat.push(c.y);
                flat.push(c.z);
            }
            let ale = s.motion().expect("attached");
            Ok(vec![
                ("U", flat),
                crate::capture::buf(&g, "p", &s.p().f)?,
                crate::capture::buf(&g, "phi", &s.phi().f)?,
                crate::capture::buf(&g, "phi_b", &s.phi().bf)?,
                crate::capture::buf(&g, "phi_rel", &s.convective_flux().f)?,
                crate::capture::buf(&g, "phi_rel_b", &s.convective_flux().bf)?,
                crate::capture::buf(&g, "v", &gm.v)?,
                crate::capture::buf(&g, "v0", &ale.v0)?,
                crate::capture::buf(&g, "phi_mesh_b", &ale.phi_mesh.bf)?,
            ])
        },
    )
    .expect("SPEC-LIT 81.7: a moving step must capture and replay bitwise");
    println!("  moving step: {report}");
}
