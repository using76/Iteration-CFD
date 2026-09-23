// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
//! SPEC-LIT section 105. No GPL-licensed source was consulted.

use super::*;
use crate::capture::capture_replays_bitwise;
use crate::mesh::gpugeom::tests::{differences, fixtures, read_back};
use crate::timescheme::{TimeKernels, fvm_ddt, fvm_ddt_rho};

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The box of `gate_105a`, built once per use.
fn gate_box() -> RefinedBox {
    refined::build([6, 5, 4], Vec3::new(0.2, 0.25, 0.3), &[0u32; 120]).expect("gate box")
}

/// A `Vec<Vec3>` as the flat `[x, y, z, ...]` the snapshot buffers hold.
fn flat(v: &[Vec3]) -> Vec<Scalar> {
    let mut out = Vec::with_capacity(v.len() * 3);
    for p in v {
        out.push(p.x);
        out.push(p.y);
        out.push(p.z);
    }
    out
}

/// The host sweep on a fixture's points, ready for `GpuMesh::upload`.
fn swept_host(base: &HostMesh, points: &[Vec3], faces: &[Vec<Label>]) -> HostMesh {
    let mut m = base.clone();
    geometry::compute(&mut m, points, faces).expect("host sweep");
    m
}

/// T1 - the swept volume on the device is its host twin, bit for bit, and a
/// face whose vertices do not move sweeps exactly zero.
#[test]
fn the_swept_volume_kernel_is_its_host_twin_bitwise() {
    let Some(gpu) = gpu() else { return };
    let rb = gate_box();
    let csr = flatten_faces(&rb.faces);
    let rest = swept_host(&rb.mesh, &rb.points, &rb.faces);
    let gm = GpuMesh::upload(&gpu, &rest).expect("gate box mesh");
    let mut ale = AleMesh::new(&gpu, &rest, &gm, &rb.points, &csr).expect("ale");
    let (lo, hi) = point_bounds(&rb.points);
    let n_if = rest.n_internal_faces;
    let n_bf = rest.n_boundary_faces;
    let mut compared = 0usize;

    let mut last: Vec<Vec3> = rb.points.clone();
    for t in [0.125 as Scalar, 0.2] {
        let moved: Vec<Vec3> = rb.points.iter().map(|p| interior_sinusoid(*p, lo, hi, 0.03, 0.5, t)).collect();
        ale.set_points(&gpu, &moved).expect("set_points");
        ale.advance(&gpu, &gm, DdtScheme::Euler.coeffs(0.01, 0.01, 0).expect("euler coeffs"))
            .expect("advance");
        gpu.sync().expect("sync");
        let swept: Vec<Scalar> =
            gpu.download(&ale.swept).expect("swept").into_iter().take(n_if + n_bf).collect();
        let host = host_swept_volumes(&last, &moved, &csr);
        for (f, (d, h)) in swept.iter().zip(host.iter()).enumerate() {
            assert_eq!(
                d.to_bits(),
                h.to_bits(),
                "face {f}: the device swept volume is not the host twin's bits"
            );
        }
        for bf in 0..n_bf {
            assert_eq!(
                swept[n_if + bf].to_bits(),
                (0.0 as Scalar).to_bits(),
                "boundary face {bf} moved: swept volume is not exactly zero"
            );
        }
        compared += swept.len();
        last = moved;
    }

    // The pentagonal fixture: five vertices per face, the one face shape the
    // quad identity of SPEC-LIT 105.3 does not cover - but the TWIN claim,
    // device vs host, must hold on it all the same.
    for (name, base, points, faces) in fixtures() {
        if !name.contains("pentagonal") {
            continue;
        }
        let rest = swept_host(&base, &points, &faces);
        let csr = flatten_faces(&faces);
        let gm = GpuMesh::upload(&gpu, &rest).expect("pentagonal mesh");
        let mut ale = AleMesh::new(&gpu, &rest, &gm, &points, &csr).expect("ale");
        let v_min = rest.v.iter().copied().fold(Scalar::MAX, Scalar::min);
        let amp = 0.1 * v_min.cbrt();
        let (lo, hi) = point_bounds(&points);
        let moved: Vec<Vec3> =
            points.iter().map(|p| interior_sinusoid(*p, lo, hi, amp, 0.5, 0.125)).collect();
        ale.set_points(&gpu, &moved).expect("set_points");
        ale.advance(&gpu, &gm, DdtScheme::Euler.coeffs(0.01, 0.01, 0).expect("euler coeffs"))
            .expect("advance");
        gpu.sync().expect("sync");
        let swept: Vec<Scalar> = gpu.download(&ale.swept).expect("swept").into_iter().take(csr.offset.len() - 1).collect();
        let host = host_swept_volumes(&points, &moved, &csr);
        for (f, (d, h)) in swept.iter().zip(host.iter()).enumerate() {
            assert_eq!(d.to_bits(), h.to_bits(), "{name} face {f}: not the host twin's bits");
        }
        compared += swept.len();
    }
    println!("  compared the swept volume of {compared} faces (gate box twice, pentagonal once)");
}

/// T2 - the contracted build of ale.cu does NOT give the host's bits on at
/// least one face; that is what the -fmad=false flag buys.
#[test]
fn the_contraction_the_ale_unit_turns_off_is_real() {
    let Some(gpu) = gpu() else { return };
    let rb = gate_box();
    let csr = flatten_faces(&rb.faces);
    let rest = swept_host(&rb.mesh, &rb.points, &rb.faces);
    let gm = GpuMesh::upload(&gpu, &rest).expect("gate box mesh");
    let mut ale = AleMesh::new(&gpu, &rest, &gm, &rb.points, &csr).expect("ale");
    let mut fused =
        AleMesh::with_cubin(&gpu, &rest, &gm, &rb.points, &csr, crate::kernels::ALE_FMAD)
            .expect("the contracted twin must load");
    let (lo, hi) = point_bounds(&rb.points);

    let mut last: Vec<Vec3> = rb.points.clone();
    let mut separated = 0usize;
    let mut faces = 0usize;
    for t in [0.125 as Scalar, 0.2] {
        let moved: Vec<Vec3> = rb.points.iter().map(|p| interior_sinusoid(*p, lo, hi, 0.03, 0.5, t)).collect();
        ale.set_points(&gpu, &moved).expect("set_points");
        ale.advance(&gpu, &gm, DdtScheme::Euler.coeffs(0.01, 0.01, 0).expect("euler coeffs"))
            .expect("advance");
        fused.set_points(&gpu, &moved).expect("set_points");
        fused.advance(&gpu, &gm, DdtScheme::Euler.coeffs(0.01, 0.01, 0).expect("euler coeffs"))
            .expect("advance");
        gpu.sync().expect("sync");
        let n_faces = rest.n_internal_faces + rest.n_boundary_faces;
        let unfused: Vec<Scalar> =
            gpu.download(&ale.swept).expect("swept").into_iter().take(n_faces).collect();
        let got: Vec<Scalar> =
            gpu.download(&fused.swept).expect("swept").into_iter().take(n_faces).collect();
        let host = host_swept_volumes(&last, &moved, &csr);
        for (f, ((u, g), h)) in unfused.iter().zip(got.iter()).zip(host.iter()).enumerate() {
            assert_eq!(
                u.to_bits(),
                h.to_bits(),
                "face {f}: the -fmad=false build must match the host bitwise"
            );
            if g.to_bits() != h.to_bits() {
                separated += 1;
            }
        }
        faces += n_faces;
        last = moved;
    }
    println!("  fused differs on {separated} of {faces} faces");
    assert!(
        separated > 0,
        "ale.cu is compiled with -fmad=false to keep the bitwise claim against \
         host_swept_volumes, but on every face here the CONTRACTED build of the \
         same source gives the host's bits too - if this fails, do NOT drop the \
         flag here: report it, and let the supervisor decide"
    );
}

/// T3 - after `recompute_in_place` on moved points, all sixteen arrays of
/// `gm` equal `geometry::compute` on the moved points, bit for bit, on every
/// gpugeom fixture.
#[test]
fn the_recomputed_geometry_is_the_host_sweep_on_the_moved_points() {
    let Some(gpu) = gpu() else { return };
    let mut checked = 0usize;
    for (name, base, points, faces) in fixtures() {
        let rest = swept_host(&base, &points, &faces);
        let gm = GpuMesh::upload(&gpu, &rest).expect("upload");
        let csr = flatten_faces(&faces);
        let mut ale = AleMesh::new(&gpu, &rest, &gm, &points, &csr).expect("ale");

        let v_min = rest.v.iter().copied().fold(Scalar::MAX, Scalar::min);
        let amp = 0.1 * v_min.cbrt();
        let (lo, hi) = point_bounds(&points);
        let moved: Vec<Vec3> =
            points.iter().map(|p| interior_sinusoid(*p, lo, hi, amp, 0.5, 0.125)).collect();

        let host = swept_host(&base, &moved, &faces);
        ale.set_points(&gpu, &moved).expect("set_points");
        ale.recompute_in_place(&gpu, &gm).expect("recompute_in_place");

        let dev = read_back(&gpu, &gm, &rest);
        let bad = differences(&host, &dev);
        assert!(
            bad.is_empty(),
            "SPEC-LIT 105.2 requires the in-place recompute to be BITWISE the \
             host sweep on the moved points, and on '{name}' it is not:\n  {}",
            bad.join("\n  ")
        );
        checked += 1;
    }
    println!("  compared the in-place recompute with the host sweep on {checked} fixtures");
}

/// T4 - Gate 105-A itself: space conservation to round-off over 100 steps,
/// euler and backward, and the uniform state uniform to round-off.
#[test]
fn gate_105a_space_conservation_holds_to_round_off() {
    let Some(gpu) = gpu() else { return };
    for (name, scheme) in [("euler", DdtScheme::Euler), ("backward", DdtScheme::Backward)] {
        let r = gate_105a(&gpu, scheme).expect("the gate run");
        println!(
            "  {name}: steps {} cells {} worst_scl {:.3e} worst_scheme {:.3e} \
             worst_uniform {:.3e} volume_drift {:.3e} boundary_flux_max {} \
             min_volume_ratio {:.6} history_bitwise {} worst at step {} cell {}",
            r.steps, r.n_cells, r.worst_scl, r.worst_scheme, r.worst_uniform,
            r.volume_drift, r.boundary_flux_max, r.min_volume_ratio,
            r.history_bitwise, r.worst_step, r.worst_cell
        );
        assert!(r.worst_scl <= 1e-12, "{name}: worst_scl {}", r.worst_scl);
        assert!(r.worst_scheme <= 1e-12, "{name}: worst_scheme {}", r.worst_scheme);
        assert!(r.worst_uniform <= 1e-12, "{name}: worst_uniform {}", r.worst_uniform);
        assert!(r.volume_drift <= 1e-12, "{name}: volume_drift {}", r.volume_drift);
        assert_eq!(
            r.boundary_flux_max, 0.0,
            "{name}: a boundary face moved, and its phi_mesh is not exactly zero"
        );
        assert!(r.history_bitwise, "{name}: the volume history is not bitwise");
        assert!(r.min_volume_ratio > 0.5, "{name}: min V/V0 {}", r.min_volume_ratio);
        assert_eq!(r.steps, 100);
    }
}

/// The fan of geometry.rs `face_geometry`, host-local: T5 is about the
/// identity, and must not depend on the module under test.
fn fan(verts: &[Label], points: &[Vec3]) -> (Vec3, Vec3) {
    let n = verts.len();
    if n == 0 {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    let mut x_avg = Vec3::ZERO;
    for &v in verts {
        x_avg += points[v as usize];
    }
    x_avg = x_avg / n as Scalar;
    if n < 3 {
        return (Vec3::ZERO, x_avg);
    }
    let mut sf = Vec3::ZERO;
    let mut cf = Vec3::ZERO;
    let mut area: Scalar = 0.0;
    for i in 0..n {
        let a = points[verts[i] as usize];
        let b = points[verts[(i + 1) % n] as usize];
        let t_n = (a - x_avg).cross(b - x_avg);
        let t_c = (x_avg + a + b) / 3.0;
        let t_a = t_n.mag() * 0.5;
        sf += t_n * 0.5;
        cf += t_c * t_a;
        area += t_a;
    }
    if area > 1.0e-150 {
        (sf, cf / area)
    } else {
        (sf, x_avg)
    }
}

/// T5 - the identity of SPEC-LIT 105.3: a warped QUAD has
/// `Sf.(Cf - x_avg) = 0` to round-off, a warped PENTAGON does not.
#[test]
fn a_quadrilateral_keeps_its_centroid_offset_normal_to_sf_and_a_pentagon_does_not() {
    // A fixed 32-bit LCG, as the gpugeom fixture jitter uses, coordinates in
    // [-0.5, 0.5) - the same warping every run and every machine.
    let mut h = 20_260_923u32;
    let mut next = move || {
        h = h.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (h >> 8) as Scalar / (1u32 << 24) as Scalar - 0.5
    };
    let measure = |verts: &[Label], points: &[Vec3]| -> Scalar {
        let (sf, cf) = fan(verts, points);
        let x_avg: Vec3 =
            verts.iter().map(|v| points[*v as usize]).fold(Vec3::ZERO, |a, p| a + p) / verts.len() as Scalar;
        let mut diam: Scalar = 0.0;
        for a in verts {
            for b in verts {
                diam = diam.max((points[*a as usize] - points[*b as usize]).mag());
            }
        }
        sf.dot(cf - x_avg).abs() / (sf.mag() * diam)
    };

    let mut pts: Vec<Vec3> = Vec::new();
    let mut rs: Vec<Scalar> = Vec::new();
    for shape in [4usize, 4, 4, 5] {
        let start = pts.len() as Label;
        for _ in 0..shape {
            pts.push(Vec3::new(next(), next(), next()));
        }
        let verts: Vec<Label> = (start..start + shape as Label).collect();
        let r = measure(&verts, &pts);
        if shape == 4 {
            println!("  quad {}: r = {r:.3e}", rs.len() + 1);
        } else {
            println!("  pentagon: r = {r:.3e}");
        }
        rs.push(r);
    }
    for (k, r) in rs.iter().take(3).enumerate() {
        assert!(
            *r <= 1e-14,
            "quad {}: Sf.(Cf - x_avg) is {r:.3e} of |Sf| diam, and the identity \
             of SPEC-LIT 105.3 says it is zero to round-off",
            k + 1
        );
    }
    assert!(
        rs[3] >= 1e-6,
        "pentagon: r = {:.3e}, but a face of five vertices is exactly the case \
         the quad identity does not cover - the offset should show",
        rs[3]
    );
}

/// T6 - on a mesh that does not move, the ALE ddt equals `timescheme`'s ddt
/// to 1e-14 relative: the same equation, one association apart.
#[test]
fn the_ale_ddt_reduces_to_the_static_ddt_on_a_mesh_that_does_not_move() {
    let Some(gpu) = gpu() else { return };
    let rb = gate_box();
    let csr = flatten_faces(&rb.faces);
    let rest = swept_host(&rb.mesh, &rb.points, &rb.faces);
    let gm = GpuMesh::upload(&gpu, &rest).expect("gate box mesh");
    let ale = AleMesh::new(&gpu, &rest, &gm, &rb.points, &csr).expect("ale");
    let tk = TimeKernels::new(&gpu).expect("the timescheme kernels");

    let n = rest.n_cells;
    let mut seed = 105u64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as Scalar) / (u32::MAX as Scalar)
    };
    let psi0v: Vec<Scalar> = (0..n).map(|_| rnd() - 0.5).collect();
    let psi00v: Vec<Scalar> = (0..n).map(|_| rnd() - 0.5).collect();
    let rhov: Vec<Scalar> = (0..n).map(|_| 0.5 + rnd()).collect();
    let rho0v: Vec<Scalar> = (0..n).map(|_| 0.5 + rnd()).collect();
    let rho00v: Vec<Scalar> = (0..n).map(|_| 0.5 + rnd()).collect();
    let psi0 = gpu.upload(&psi0v).expect("psi0");
    let psi00 = gpu.upload(&psi00v).expect("psi00");
    let rho = gpu.upload(&rhov).expect("rho");
    let rho0 = gpu.upload(&rho0v).expect("rho0");
    let rho00 = gpu.upload(&rho00v).expect("rho00");
    let c = DdtScheme::Backward.coeffs(0.01, 0.01, 1).expect("backward coeffs");

    let rel = |x: &[Scalar], y: &[Scalar]| -> Scalar {
        let nx = x.iter().fold(0.0 as Scalar, |a, b| a.max(b.abs()));
        let d = x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0.0 as Scalar, |a, b| a.max(b));
        if nx == 0.0 { d } else { d / nx }
    };

    let mut a1 = GpuLduMatrix::new(&gpu, &gm).expect("matrix 1");
    let mut a2 = GpuLduMatrix::new(&gpu, &gm).expect("matrix 2");
    a1.zero(&gpu).expect("zero");
    a2.zero(&gpu).expect("zero");
    fvm_ddt(&gpu, &tk, &mut a1, &gm, &psi0, &psi00, c, 1.0).expect("timescheme ddt");
    ale.fvm_ddt(&gpu, &mut a2, &gm, &psi0, &psi00, c, 1.0).expect("ale ddt");
    gpu.sync().expect("sync");
    let d1 = gpu.download(&a1.diag).expect("diag");
    let d2 = gpu.download(&a2.diag).expect("diag");
    let s1 = gpu.download(&a1.source).expect("source");
    let s2 = gpu.download(&a2.source).expect("source");
    let (rd, rs_) = (rel(&d1, &d2), rel(&s1, &s2));
    println!("  psi: diag rel {rd:.3e}, source rel {rs_:.3e}");
    assert!(rd <= 1e-14, "diag: {rd:.3e}");
    assert!(rs_ <= 1e-14, "source: {rs_:.3e}");

    a1.zero(&gpu).expect("zero");
    a2.zero(&gpu).expect("zero");
    fvm_ddt_rho(&gpu, &tk, &mut a1, &gm, &rho, &rho0, &rho00, &psi0, &psi00, c, 1.0)
        .expect("timescheme ddt rho");
    ale.fvm_ddt_rho(&gpu, &mut a2, &gm, &rho, &rho0, &rho00, &psi0, &psi00, c, 1.0)
        .expect("ale ddt rho");
    gpu.sync().expect("sync");
    let d1 = gpu.download(&a1.diag).expect("diag");
    let d2 = gpu.download(&a2.diag).expect("diag");
    let s1 = gpu.download(&a1.source).expect("source");
    let s2 = gpu.download(&a2.source).expect("source");
    let (rd, rs_) = (rel(&d1, &d2), rel(&s1, &s2));
    println!("  rho psi: diag rel {rd:.3e}, source rel {rs_:.3e}");
    assert!(rd <= 1e-14, "rho diag: {rd:.3e}");
    assert!(rs_ <= 1e-14, "rho source: {rs_:.3e}");
}

/// T7 - (C1) works: a `Momentum` holding `&gm` is alive across an `advance`,
/// and `gm.v` changes under it.
#[test]
fn a_solver_that_borrows_the_mesh_still_sees_it_move() {
    let Some(gpu) = gpu() else { return };
    let rb = gate_box();
    let csr = flatten_faces(&rb.faces);
    let rest = swept_host(&rb.mesh, &rb.points, &rb.faces);
    let gm = GpuMesh::upload(&gpu, &rest).expect("gate box mesh");
    let mut ale = AleMesh::new(&gpu, &rest, &gm, &rb.points, &csr).expect("ale");

    let mut ctrl = crate::momentum::MomentumControls {
        nu: 1e-3,
        u_relax: 1.0,
        ..Default::default()
    };
    ctrl.u_solver.fixed_iters = true;
    ctrl.u_solver.max_iter = 4;
    ctrl.u_solver.report_residuals = false;
    let mom = crate::momentum::Momentum::new(&gpu, &gm, ctrl, Default::default())
        .expect("the momentum predictor holds &GpuMesh");

    let before = gpu.download(&gm.v).expect("v before");
    let (lo, hi) = point_bounds(&rb.points);
    let moved: Vec<Vec3> =
        rb.points.iter().map(|p| interior_sinusoid(*p, lo, hi, 0.03, 0.5, 0.125)).collect();
    ale.set_points(&gpu, &moved).expect("set_points");
    ale.advance(&gpu, &gm, DdtScheme::Euler.coeffs(0.01, 0.01, 0).expect("euler coeffs"))
        .expect("advance while the solver borrows the mesh");
    gpu.sync().expect("sync");
    let after = gpu.download(&gm.v).expect("v after");
    assert_ne!(before, after, "the mesh moved and gm.v did not change");

    drop(mom); // the borrow was alive across the advance - that is the test
}

/// T8 - `AleMesh` refuses, by name, what it was not built for.
#[test]
fn the_ale_mesh_refuses_what_it_was_not_built_for() {
    let Some(gpu) = gpu() else { return };
    let rb = gate_box();
    let csr = flatten_faces(&rb.faces);
    let rest = swept_host(&rb.mesh, &rb.points, &rb.faces);
    let gm = GpuMesh::upload(&gpu, &rest).expect("gate box mesh");
    let mut ale = AleMesh::new(&gpu, &rest, &gm, &rb.points, &csr).expect("ale");

    // (a) a second box's GpuMesh against the first box's HostMesh.
    let rb2 = refined::build([3, 3, 3], Vec3::new(0.25, 0.3, 0.2), &[0u32; 27]).expect("second box");
    let rest2 = swept_host(&rb2.mesh, &rb2.points, &rb2.faces);
    let gm2 = GpuMesh::upload(&gpu, &rest2).expect("second mesh");
    let e = AleMesh::new(&gpu, &rest, &gm2, &rb.points, &csr).err().expect("(a)");
    let m = e.to_string();
    assert!(m.contains("AleMesh::new: the device mesh"), "(a): {m}");
    assert!(m.contains("an AleMesh is built for ONE mesh"), "(a): {m}");

    // (b) the wrong number of points.
    let e = AleMesh::new(&gpu, &rest, &gm, &rb.points[..5], &csr).err().expect("(b)");
    let m = e.to_string();
    assert!(m.contains("AleMesh::new: "), "(b): {m}");
    assert!(m.contains(" points for a mesh of "), "(b): {m}");

    // (c) set_points at the wrong length.
    let e = ale.set_points(&gpu, &rb.points[..7]).err().expect("(c)");
    let m = e.to_string();
    assert!(m.contains("AleMesh::set_points: "), "(c): {m}");
    assert!(m.contains(" points, and this AleMesh moves "), "(c): {m}");

    // (d) recompute_in_place against the second mesh.
    let e = ale.recompute_in_place(&gpu, &gm2).err().expect("(d)");
    let m = e.to_string();
    assert!(m.contains("AleMesh::recompute_in_place"), "(d): {m}");
    assert!(m.contains("was built for a mesh of"), "(d): {m}");

    // (e) a steady scheme.
    let e = ale.advance(&gpu, &gm, DdtCoeffs::ZERO).err().expect("(e)");
    let m = e.to_string();
    assert!(
        m.contains("AleMesh::advance: a moving mesh needs a transient time scheme"),
        "(e): {m}"
    );

    // (f) coefficients that do not sum to zero.
    let bad = DdtCoeffs { a_n: 1.0, a_0: -0.5, a_00: -0.5 + 1.0e-3 };
    let e = ale.advance(&gpu, &gm, bad).err().expect("(f)");
    let m = e.to_string();
    assert!(m.contains("AleMesh::advance: the coefficients"), "(f): {m}");
    assert!(m.contains("do not sum to zero"), "(f): {m}");

    // (g) the two ALE ddt wrappers, against a mesh this AleMesh was not built
    // for: its volume history is the first mesh's, so reading it for the
    // second would be the wrong volumes - or past the end of them.
    let c = DdtScheme::Euler.coeffs(0.01, 0.01, 0).expect("euler coeffs");
    let mut a2 = GpuLduMatrix::new(&gpu, &gm2).expect("second matrix");
    let ones2 = gpu.upload(&vec![1.0 as Scalar; rest2.n_cells]).expect("ones");
    let e = ale.fvm_ddt(&gpu, &mut a2, &gm2, &ones2, &ones2, c, 1.0).err().expect("(g) fvm_ddt");
    let m = e.to_string();
    assert!(m.contains("AleMesh::fvm_ddt"), "(g): {m}");
    assert!(m.contains("was built for a mesh of"), "(g): {m}");
    let e = ale
        .fvm_ddt_rho(&gpu, &mut a2, &gm2, &ones2, &ones2, &ones2, &ones2, &ones2, c, 1.0)
        .err()
        .expect("(g) fvm_ddt_rho");
    let m = e.to_string();
    assert!(m.contains("AleMesh::fvm_ddt_rho"), "(g): {m}");
    assert!(m.contains("was built for a mesh of"), "(g): {m}");
}

/// T9 - the gate: one ALE step captured into a CUDA graph and replayed
/// bitwise. The registry row for src/mesh/ale.rs is this test.
#[test]
fn the_ale_step_replays_bitwise() {
    let Some(gpu) = gpu() else { return };
    let report = capture_replays_bitwise(
        &gpu,
        "ALE step (SPEC-LIT 105.2)",
        || {
            let rb = refined::build([4, 3, 3], Vec3::new(0.25, 0.3, 0.2), &[0u32; 36])
                .expect("the gate mesh");
            let rest = swept_host(&rb.mesh, &rb.points, &rb.faces);
            let gm = GpuMesh::upload(&gpu, &rest)?;
            let csr = flatten_faces(&rb.faces);
            let ale = AleMesh::new(&gpu, &rest, &gm, &rb.points, &csr)?;
            let (lo, hi) = point_bounds(&rb.points);
            let b_points: Vec<Vec3> = rb
                .points
                .iter()
                .map(|p| interior_sinusoid(*p, lo, hi, 0.04, 0.5, 0.125))
                .collect();
            let spare = gpu.upload(&b_points)?;
            Ok((gm, ale, spare))
        },
        |(gm, ale, spare): &mut (GpuMesh, AleMesh, DevBuf<Vec3>)| {
            // The points alternate rest, moved, rest, ... with FIXED device
            // pointers, so every replayed step really moves.
            let c = DdtScheme::Backward.coeffs(0.01, 0.01, 1)?;
            gpu.stream().memcpy_dtod(spare, &mut ale.points)?;
            gpu.stream().memcpy_dtod(&ale.points_old, spare)?;
            ale.advance(&gpu, gm, c)
        },
        |(gm, ale, _): &(GpuMesh, AleMesh, DevBuf<Vec3>)| {
            Ok(vec![
                ("v", gpu.download(&gm.v)?),
                ("c", flat(&gpu.download(&gm.c)?)),
                ("sf", flat(&gpu.download(&gm.sf)?)),
                ("mag_sf", gpu.download(&gm.mag_sf)?),
                ("weights", gpu.download(&gm.weights)?),
                ("delta_coeffs", gpu.download(&gm.delta_coeffs)?),
                ("b_sf", flat(&gpu.download(&gm.b_sf)?)),
                ("b_y", gpu.download(&gm.b_y)?),
                ("v0", gpu.download(&ale.v0)?),
                ("v00", gpu.download(&ale.v00)?),
                ("swept", gpu.download(&ale.swept)?),
                ("swept0", gpu.download(&ale.swept0)?),
                ("phi_mesh", gpu.download(&ale.phi_mesh.f)?),
                ("phi_mesh_b", gpu.download(&ale.phi_mesh.bf)?),
                ("points", flat(&gpu.download(&ale.points)?)),
                ("points_old", flat(&gpu.download(&ale.points_old)?)),
            ])
        },
    )
    .expect("SPEC-LIT 81.7: the ALE step must capture and replay bitwise");
    println!("  ale: {report}");
}

/// `SclRun::volume_drift` is the WORST step's drift, as its doc says, not the
/// last step's. A uniform dilation of the whole box - boundary included - that
/// returns to rest after one period has a total-volume change of a few per
/// cent in mid-run and of round-off at the last step; a drift that reported
/// the last step would read ~1e-16 here. The same run is also the one place
/// the swept-volume identity is measured with BOUNDARY faces sweeping, so its
/// SCL is held to the gate's 1e-12 too.
#[test]
fn the_volume_drift_is_the_worst_step_and_not_the_last() {
    let Some(gpu) = gpu() else { return };
    let rb = refined::build([4, 3, 3], Vec3::new(0.25, 0.3, 0.2), &[0u32; 36]).expect("box");
    let pi = std::f64::consts::PI as Scalar;
    let dilate = |p: Vec3, t: Scalar| p * (1.0 + 0.01 * (2.0 * pi * t / 0.5).sin());
    let r = run_scl(&gpu, &rb, &dilate, 0.05, 10, DdtScheme::Euler).expect("the dilation run");
    println!(
        "  dilation: volume_drift {:.3e}, worst_scl {:.3e}, boundary_flux_max {:.3e}",
        r.volume_drift, r.worst_scl, r.boundary_flux_max
    );
    assert!(
        r.volume_drift > 1e-2,
        "volume_drift is {:.3e}, but the box's volume changed by about 3 % in \
         mid-run: the drift reported is the last step's, not the worst",
        r.volume_drift
    );
    assert!(r.boundary_flux_max > 0.0, "the boundary moved, so its phi_mesh cannot be zero");
    assert!(r.worst_scl <= 1e-12, "worst_scl {:.3e} with sweeping boundary faces", r.worst_scl);
}
