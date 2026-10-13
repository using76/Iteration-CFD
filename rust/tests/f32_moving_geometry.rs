// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 118.8 — moving geometry in f64 on the device (F32-11). Under
//! `single` the geometry sweep computes in double from double points and
//! rounds once per stored value, and the moving mesh keeps its points, its
//! volumes, its volume history and its swept volumes in f64 on the device,
//! so the device sweep is the host sweep bit for bit and Gate 105-A's
//! conservation rows are measured in f64 in BOTH builds. An integration test
//! because the library's own test target does not compile under `single`;
//! every test runs in BOTH builds with the same assertions — docs/17 §5.1
//! rule 4: class D on the bitwise equalities and the counts, class A on the
//! swept volume of a displacement f32 cannot see, class R on the one row
//! still assembled from the float copies (the uniform-flow residual of Gate
//! 105-A, tolerance 1e-5 under `single` by D-F32-1's
//! `tol32 = max(tol64, min(tol64*2^29, 1e-5))`).
//!
//! Provenance: ORIGINAL — the meshes are this tree's own generators; no
//! GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::field_setup::{setup_scalar_field, setup_vector_field_with, BcInputs};
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::io::polymesh::{build_host_mesh, build_host_mesh_local, PolyMeshRaw};
use ofgpu::mesh::ale::{gate_105a, host_swept_volumes, AleMesh};
use ofgpu::mesh::gpugeom::{flatten_faces, gpu_compute_geometry, MeshGeomKernels};
use ofgpu::momentum::{BuoyancyCoeffs, MomentumControls};
use ofgpu::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use ofgpu::simple::{Simple, SimpleControls};
use ofgpu::timescheme::DdtScheme;
use ofgpu::types::DVec3;
use ofgpu::{BcKind, Gpu, GpuMesh, GpuScalarField, HostMesh, Label, Result, Scalar, Vec3};

/// The build's `Scalar` widened to the f64 the answers are compared in.
#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    f64::from(x)
}

/// `[lo, hi]` axes at the given expansion (walls), the six patch types in
/// `(xMin, xMax, yMin, yMax, zMin, zMax)` order.
fn box_spec(
    n: [usize; 3],
    lo: [Scalar; 3],
    hi: [Scalar; 3],
    expansion: [Scalar; 3],
    types: [&str; 6],
) -> BlockSpec {
    let axis = |lo: Scalar, hi: Scalar, n: usize, e: Scalar| GradedAxis {
        lo,
        hi,
        n,
        expansion: e,
        two_sided: e != 1.0,
    };
    BlockSpec {
        x: axis(lo[0], hi[0], n[0], expansion[0]),
        y: axis(lo[1], hi[1], n[1], expansion[1]),
        z: axis(lo[2], hi[2], n[2], expansion[2]),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: types.map(String::from),
        cyclic: Vec::new(),
    }
}

/// T1's G: the graded box 14x11x9 on `[0,1]x[0,0.8]x[0,0.6]`, expansions 4,
/// 0.25 and 2 toward the walls.
fn graded_spec() -> BlockSpec {
    box_spec(
        [14, 11, 9],
        [0.0; 3],
        [1.0, 0.8, 0.6],
        [4.0, 0.25, 2.0],
        ["wall"; 6],
    )
}

/// T1's/T4's A: the uniform 8x8x8 box on `[0,1]^3`.
fn uniform_spec() -> BlockSpec {
    box_spec([8, 8, 8], [0.0; 3], [1.0; 3], [1.0; 3], ["wall"; 6])
}

/// T6's mesh: the uniform `N^3` box on `[0,1]^3`, six walls.
fn cost_spec(n: usize) -> BlockSpec {
    box_spec([n, n, n], [0.0; 3], [1.0; 3], [1.0; 3], ["wall"; 6])
}

fn raw_of(spec: &BlockSpec) -> PolyMeshRaw {
    blockgen::raw_mesh(spec).expect("raw mesh")
}

/// "Translated" always means this: every point at +2000 in x and y, the
/// far-from-origin case (SPEC-LIT 118.7).
fn translated(mut raw: PolyMeshRaw) -> PolyMeshRaw {
    for p in raw.points.iter_mut() {
        p.x += 2000.0;
        p.y += 2000.0;
    }
    raw
}

fn scalar_bits(a: &[Scalar], b: &[Scalar]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn vec3_bits(a: &[Vec3], b: &[Vec3]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(p, q)| {
            p.x.to_bits() == q.x.to_bits()
                && p.y.to_bits() == q.y.to_bits()
                && p.z.to_bits() == q.z.to_bits()
        })
}

fn f64_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// All sixteen geometry arrays, bit for bit between two host meshes - the
/// device-built one has been downloaded into `b` by `gpu_compute_geometry`.
fn sixteen_bits(a: &HostMesh, b: &HostMesh) -> bool {
    scalar_bits(&a.v, &b.v)
        && vec3_bits(&a.c, &b.c)
        && vec3_bits(&a.sf, &b.sf)
        && vec3_bits(&a.cf, &b.cf)
        && scalar_bits(&a.mag_sf, &b.mag_sf)
        && scalar_bits(&a.weights, &b.weights)
        && scalar_bits(&a.delta_coeffs, &b.delta_coeffs)
        && vec3_bits(&a.non_orth_corr, &b.non_orth_corr)
        && vec3_bits(&a.skew_corr, &b.skew_corr)
        && vec3_bits(&a.b_sf, &b.b_sf)
        && scalar_bits(&a.b_mag_sf, &b.b_mag_sf)
        && vec3_bits(&a.b_cf, &b.b_cf)
        && scalar_bits(&a.b_delta_coeffs, &b.b_delta_coeffs)
        && vec3_bits(&a.b_non_orth_corr, &b.b_non_orth_corr)
        && scalar_bits(&a.b_y, &b.b_y)
        && scalar_bits(&a.b_weights, &b.b_weights)
}

/// Rebuild `m`'s sixteen arrays ON THE DEVICE from its raw points and faces,
/// exactly as a caller switching sweeps would.
fn device_rebuild(
    raw: &PolyMeshRaw,
    m: &HostMesh,
    gpu: &Gpu,
    k: &MeshGeomKernels,
) -> Result<HostMesh> {
    let mut clone = m.clone();
    gpu_compute_geometry(gpu, k, &mut clone, &raw.points, &raw.faces)?;
    Ok(clone)
}

/// Quantise to a multiple of 2^-20, so that `+ 2000` is exact in f64 (T4).
fn quant20(v: f64) -> f64 {
    (v * 1_048_576.0).round() / 1_048_576.0
}

/// The interior motion of T4 (its steps k = 1..3), T5 and T6: every point
/// strictly inside the points' own box moves by
/// `amp * k * x(1-x) y(1-y) z(1-z)` in each coordinate FROM ITS REST
/// position, the moved coordinate quantised to a multiple of 2^-20. A
/// boundary point returns untouched - `interior_sinusoid`'s rule.
fn moved_points(rest: &[DVec3], k: f64, amp: f64) -> Vec<DVec3> {
    let (mut lo, mut hi) = (rest[0], rest[0]);
    for p in rest {
        lo = DVec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
        hi = DVec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
    }
    rest.iter()
        .map(|p| {
            let inside = p.x > lo.x
                && p.x < hi.x
                && p.y > lo.y
                && p.y < hi.y
                && p.z > lo.z
                && p.z < hi.z;
            if !inside {
                return *p;
            }
            let d = amp * k * p.x * (1.0 - p.x) * p.y * (1.0 - p.y) * p.z * (1.0 - p.z);
            DVec3::new(quant20(p.x + d), quant20(p.y + d), quant20(p.z + d))
        })
        .collect()
}

/// T1 (class D): the device sweep is the host sweep bit for bit, on the
/// graded box, the uniform box and the translated local-origin box.
#[test]
fn the_device_sweep_is_the_host_sweep_bit_for_bit() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let k = MeshGeomKernels::new(&gpu)?;

    let raw_g = raw_of(&graded_spec());
    let g = build_host_mesh(&raw_g)?;
    let gg = device_rebuild(&raw_g, &g, &gpu, &k)?;
    assert!(sixteen_bits(&g, &gg), "G: the device sweep is not the host sweep");

    let raw_a = raw_of(&uniform_spec());
    let a = build_host_mesh(&raw_a)?;
    let aa = device_rebuild(&raw_a, &a, &gpu, &k)?;
    assert!(sixteen_bits(&a, &aa), "A: the device sweep is not the host sweep");

    let raw_b = translated(raw_of(&uniform_spec()));
    let b = build_host_mesh_local(&raw_b)?;
    let bb = device_rebuild(&raw_b, &b, &gpu, &k)?;
    assert!(sixteen_bits(&b, &bb), "B: the device sweep is not the host sweep");

    if cfg!(feature = "single") {
        assert_eq!(b.origin, DVec3::new(2000.0, 2000.0, 0.0), "(118.7a)");
        assert!(
            sixteen_bits(&a, &bb),
            "under single the swept local-origin mesh is the origin mesh bit for bit"
        );
    } else {
        assert_eq!(b.origin, DVec3::ZERO, "the f64 build never shifts");
    }
    Ok(())
}

/// T2 (class A on the conservation rows, class R on the uniform row): Gate
/// 105-A's rows, measured in f64 in BOTH builds. The f64 build's expected
/// values are HEAD's: worst_scl 1.535e-15, scheme 1.531e-15 (euler) /
/// 1.798e-15 (backward), uniform 1.770e-15 / 1.997e-15, drift 2.591e-15.
#[test]
fn gate_105a_scl_rows_hold_in_f64() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let tol64 = 1e-12;
    // D-F32-1: tol32 = max(tol64, min(tol64*2^29, 1e-5)) = 1e-5 here.
    let tol32 = 1e-5;
    for scheme in [DdtScheme::Euler, DdtScheme::Backward] {
        let r = gate_105a(&gpu, scheme)?;
        println!(
            "{scheme:?}: worst_scl {:.3e}, worst_scheme {:.3e}, worst_uniform \
             {:.3e}, drift {:.3e}, history_bitwise {}, boundary_flux_max \
             {:.3e}, min_volume_ratio {:.6}, steps {}, cells {}",
            w64(r.worst_scl),
            w64(r.worst_scheme),
            w64(r.worst_uniform),
            w64(r.volume_drift),
            r.history_bitwise,
            w64(r.boundary_flux_max),
            w64(r.min_volume_ratio),
            r.steps,
            r.n_cells
        );
        assert!(w64(r.worst_scl) <= tol64, "worst_scl {}", w64(r.worst_scl));
        assert!(w64(r.worst_scheme) <= tol64, "worst_scheme {}", w64(r.worst_scheme));
        assert!(w64(r.volume_drift) <= tol64, "drift {}", w64(r.volume_drift));
        assert!(r.history_bitwise, "the history is not bitwise");
        assert_eq!(r.steps, 100, "steps");
        assert_eq!(r.n_cells, 120, "cells");
        assert_eq!(w64(r.boundary_flux_max), 0.0, "a boundary face moved");
        assert!(
            (w64(r.min_volume_ratio) - 0.884630).abs() <= 1e-6,
            "min_volume_ratio {}",
            w64(r.min_volume_ratio)
        );
        let uni = w64(r.worst_uniform);
        if cfg!(feature = "single") {
            assert!(uni <= tol32, "uniform row {uni:.3e} > {tol32} (class R)");
        } else {
            assert!(uni <= tol64, "uniform row {uni:.3e} > {tol64}");
        }
    }
    Ok(())
}

/// T3 (class A): a displacement of 1e-9 beside a coordinate of 0.5 is far
/// below f32 resolution - a float device sweeps zero for it - but the f64
/// device sweeps it exactly: eight cells change by dV* = 1e-9 * 0.25 * 0.25
/// / 4, four faces sweep it, and the space conservation law closes per cell.
#[test]
fn a_displacement_below_f32_resolution_is_swept() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let raw = raw_of(&box_spec(
        [4, 4, 4],
        [0.0; 3],
        [1.0; 3],
        [1.0; 3],
        ["wall"; 6],
    ));
    let rest_m = build_host_mesh(&raw)?;
    let csr = flatten_faces(&raw.faces);
    let gm = GpuMesh::upload(&gpu, &rest_m)?;
    let mut ale = AleMesh::new(&gpu, &rest_m, &gm, &raw.points, &csr)?;
    let v_old = ale.volumes64(&gpu, &gm)?;

    let pi = raw
        .points
        .iter()
        .position(|p| p.x == 0.5 && p.y == 0.5 && p.z == 0.5)
        .expect("the mesh has a point at (0.5, 0.5, 0.5)");
    let mut moved = raw.points.clone();
    moved[pi].x += 1e-9;
    ale.set_points(&gpu, &moved)?;
    ale.advance(&gpu, &gm, DdtScheme::Euler.coeffs(0.01, 0.01, 0)?)?;
    gpu.sync()?;
    let v_new = ale.volumes64(&gpu, &gm)?;
    let sw = ale.swept64(&gpu)?;

    let dv = 1e-9 * 0.25 * 0.25 / 4.0;
    let changed: Vec<usize> =
        (0..v_new.len()).filter(|&c| v_new[c] != v_old[c]).collect();
    assert_eq!(changed.len(), 8, "eight cells share the moved point");
    let pos = changed.iter().filter(|&&c| v_new[c] > v_old[c]).count();
    assert_eq!(pos, 4, "four cells grow, four shrink");
    for &c in &changed {
        let rel = ((v_new[c] - v_old[c]).abs() / dv - 1.0).abs();
        assert!(rel <= 1e-4, "cell {c}: |dV|/dV* off by {rel:.3e}");
    }
    let swept_faces: Vec<usize> = (0..sw.len()).filter(|&f| sw[f].abs() >= 1e-12).collect();
    assert_eq!(swept_faces.len(), 4, "four faces are swept");
    for &f in &swept_faces {
        let rel = (sw[f].abs() / dv - 1.0).abs();
        assert!(rel <= 1e-4, "face {f}: |sw|/dV* off by {rel:.3e}");
    }
    for c in 0..v_new.len() {
        let mut s = 0.0f64;
        for f in 0..rest_m.n_internal_faces {
            let (o, n) = (rest_m.owner[f] as usize, rest_m.neighbour[f] as usize);
            if o == c {
                s += sw[f];
            }
            if n == c {
                s -= sw[f];
            }
        }
        for bf in 0..rest_m.n_boundary_faces {
            if rest_m.b_face_cells[bf] as usize == c {
                s += sw[rest_m.n_internal_faces + bf];
            }
        }
        let res = ((v_new[c] - v_old[c]) - s).abs();
        assert!(
            res <= 1e-4 * dv,
            "cell {c}: the conservation law misses by {res:.3e}"
        );
    }
    Ok(())
}

/// T4 (class D under `single`): a moving mesh about the local origin IS the
/// origin mesh - the points are uploaded shifted to it in f64 (118.8a), so
/// after every step the swept volumes, the volumes, the float copies and the
/// mesh flux are bit for bit A's. In the f64 build nothing shifts; the
/// translated mesh's volumes agree with the origin mesh's to 1e-9 relative
/// per cell and B has no origin. Class R on that tolerance.
#[test]
fn a_moving_mesh_about_a_local_origin_is_the_origin_mesh() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let raw_a = raw_of(&uniform_spec());
    let raw_b = translated(raw_of(&uniform_spec()));
    let a = build_host_mesh(&raw_a)?;
    let b = build_host_mesh_local(&raw_b)?;
    let csr = flatten_faces(&raw_a.faces);
    let gm_a = GpuMesh::upload(&gpu, &a)?;
    let gm_b = GpuMesh::upload(&gpu, &b)?;
    let mut ale_a = AleMesh::new(&gpu, &a, &gm_a, &raw_a.points, &csr)
        .expect("A: the moving state is built");
    let mut ale_b = AleMesh::new(&gpu, &b, &gm_b, &raw_b.points, &csr)
        .expect("B: no refusal in either build");
    let euler = DdtScheme::Euler.coeffs(0.01, 0.01, 0)?;

    for k in 1..=3 {
        let moved_a = moved_points(&raw_a.points, k as f64, 1e-3);
        let moved_b: Vec<DVec3> = moved_a
            .iter()
            .zip(&raw_b.points)
            .zip(&raw_a.points)
            .map(|((&ma, &rb), &ra)| {
                if ma == ra {
                    rb
                } else {
                    DVec3::new(ma.x + 2000.0, ma.y + 2000.0, ma.z)
                }
            })
            .collect();
        ale_a.set_points(&gpu, &moved_a)?;
        ale_b.set_points(&gpu, &moved_b)?;
        ale_a.advance(&gpu, &gm_a, euler)?;
        ale_b.advance(&gpu, &gm_b, euler)?;
        gpu.sync()?;

        if cfg!(feature = "single") {
            assert!(
                f64_bits(&ale_a.swept64(&gpu)?, &ale_b.swept64(&gpu)?),
                "step {k}: swept64 differs"
            );
            assert!(
                f64_bits(&ale_a.volumes64(&gpu, &gm_a)?, &ale_b.volumes64(&gpu, &gm_b)?),
                "step {k}: volumes64 differ"
            );
            let (va, vb) = (
                gpu.download(&gm_a.v)?,
                gpu.download(&gm_b.v)?,
            );
            assert!(scalar_bits(&va, &vb), "step {k}: gm.v differs");
            let (ca, cb) = (
                gpu.download(&gm_a.c)?,
                gpu.download(&gm_b.c)?,
            );
            assert!(vec3_bits(&ca, &cb), "step {k}: gm.c differs");
            let (sfa, sfb) = (gpu.download(&gm_a.sf)?, gpu.download(&gm_b.sf)?);
            assert!(vec3_bits(&sfa, &sfb), "step {k}: gm.sf differs");
            let (ba, bb2) = (gpu.download(&gm_a.b_sf)?, gpu.download(&gm_b.b_sf)?);
            assert!(vec3_bits(&ba, &bb2), "step {k}: gm.b_sf differs");
            let (bca, bcb) = (gpu.download(&gm_a.b_cf)?, gpu.download(&gm_b.b_cf)?);
            assert!(vec3_bits(&bca, &bcb), "step {k}: gm.b_cf differs");
            let (wa, wb) = (gpu.download(&gm_a.weights)?, gpu.download(&gm_b.weights)?);
            assert!(scalar_bits(&wa, &wb), "step {k}: gm.weights differ");
            let (da, db) = (
                gpu.download(&gm_a.delta_coeffs)?,
                gpu.download(&gm_b.delta_coeffs)?,
            );
            assert!(scalar_bits(&da, &db), "step {k}: gm.delta_coeffs differ");
            let (pa, pb) = (
                gpu.download(&ale_a.phi_mesh.f)?,
                gpu.download(&ale_b.phi_mesh.f)?,
            );
            assert!(scalar_bits(&pa, &pb), "step {k}: phi_mesh.f differs");
        } else {
            assert_eq!(b.origin, DVec3::ZERO, "the f64 build never shifts");
            let (va, vb) = (ale_a.volumes64(&gpu, &gm_a)?, ale_b.volumes64(&gpu, &gm_b)?);
            for (c, (x, y)) in va.iter().zip(&vb).enumerate() {
                let rel = ((x - y).abs() / x.abs()).max((x - y).abs() / y.abs());
                assert!(rel <= 1e-9, "step {k} cell {c}: volumes differ by {rel:.3e}");
            }
        }
    }
    Ok(())
}

/// T5 (class D): the swept-volume kernel is its f64 host twin, bit for bit,
/// on the graded box with T4's step-1 motion.
#[test]
fn the_swept_volume_kernel_is_its_f64_host_twin() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let raw = raw_of(&graded_spec());
    let m = build_host_mesh(&raw)?;
    let csr = flatten_faces(&raw.faces);
    let gm = GpuMesh::upload(&gpu, &m)?;
    let mut ale = AleMesh::new(&gpu, &m, &gm, &raw.points, &csr)?;
    let moved = moved_points(&raw.points, 1.0, 1e-3);
    ale.set_points(&gpu, &moved)?;
    ale.advance(&gpu, &gm, DdtScheme::Euler.coeffs(0.01, 0.01, 0)?)?;
    gpu.sync()?;
    let dev = ale.swept64(&gpu)?;
    let host = host_swept_volumes(&raw.points, &moved, &csr);
    assert_eq!(dev.len(), host.len(), "the same faces");
    for (f, (d, h)) in dev.iter().zip(&host).enumerate() {
        assert_eq!(d.to_bits(), h.to_bits(), "face {f}: not the host twin's bits");
    }
    Ok(())
}

/// A raw vector field named U, one uniform internal value, the patch entries
/// (`tests/f32_local_origin.rs`'s helper).
fn raw_u(boundary: Vec<(&str, PatchFieldSpec)>, internal: Vec3) -> RawVectorField {
    let mut raw = RawVectorField { name: "U".into(), ..Default::default() };
    for (p, s) in boundary {
        raw.boundary.insert(p.into(), s);
    }
    raw.internal = vec![internal];
    raw
}

/// A raw scalar field named p, uniform zero inside, the patch entries.
fn raw_p(boundary: Vec<(&str, PatchFieldSpec)>) -> RawScalarField {
    let mut raw = RawScalarField { name: "p".into(), ..Default::default() };
    for (p, s) in boundary {
        raw.boundary.insert(p.into(), s);
    }
    raw.internal = vec![0.0];
    raw
}

/// A patch entry carrying nothing but its type.
fn ps(ty: &str) -> PatchFieldSpec {
    PatchFieldSpec { type_name: ty.into(), ..Default::default() }
}

/// p pinned at zero on one wall: the null space is fixed.
fn p0() -> PatchFieldSpec {
    let mut s = ps("fixedValue");
    s.value = vec![0.0];
    s
}

/// Laminar: zero eddy viscosity with `calculated` boundary values.
fn laminar_nut(gpu: &Gpu, hm: &HostMesh, gm: &GpuMesh) -> Result<GpuScalarField> {
    let mut nut = GpuScalarField::zeros(gpu, gm, "nut")?;
    gpu.write(&mut nut.bc_kind, &vec![BcKind::Calculated as Label; hm.n_boundary_faces])?;
    Ok(nut)
}

/// A uniform T, zeroGradient everywhere.
fn uniform_t(gpu: &Gpu, hm: &HostMesh, gm: &GpuMesh) -> Result<GpuScalarField> {
    let mut t = GpuScalarField::zeros(gpu, gm, "T")?;
    gpu.write(&mut t.bc_kind, &vec![BcKind::ZeroGradient as Label; hm.n_boundary_faces])?;
    Ok(t)
}

/// (a) of T6: the mean `advance` of ten timed steps, two warm-ups before
/// them, each step's `set_points` untimed and synced.
fn timed_advance(gpu: &Gpu, n: usize) -> Result<f64> {
    let raw = raw_of(&cost_spec(n));
    let m = build_host_mesh(&raw)?;
    let csr = flatten_faces(&raw.faces);
    let gm = GpuMesh::upload(gpu, &m)?;
    let mut ale = AleMesh::new(gpu, &m, &gm, &raw.points, &csr)?;
    let euler = DdtScheme::Euler.coeffs(0.01, 0.01, 0)?;
    let mut total = 0.0f64;
    for k in 0..12 {
        let moved = moved_points(&raw.points, (k + 1) as f64, 1.0e-4);
        ale.set_points(gpu, &moved)?;
        gpu.sync()?;
        if k < 2 {
            continue;
        }
        let t0 = std::time::Instant::now();
        ale.advance(gpu, &gm, euler)?;
        gpu.sync()?;
        total += t0.elapsed().as_secs_f64() * 1.0e3;
    }
    Ok(total / 10.0)
}

/// (b) of T6: the mean steady SIMPLE outer iteration of the lid-driven
/// cavity of `tests/f32_local_origin.rs::run_cavity` extended to 3-D
/// (`zMin`/`zMax` noSlip for U, zeroGradient for p), two warm-ups first.
/// Default solver controls, NOT `tight()`.
fn timed_simple(gpu: &Gpu, n: usize) -> Result<f64> {
    let raw = raw_of(&cost_spec(n));
    let hm = build_host_mesh(&raw)?;
    let gm = GpuMesh::upload(gpu, &hm)?;
    let mut lid = ps("fixedValue");
    lid.value_v = vec![Vec3::new(1.0, 0.0, 0.0)];
    let ctrl = SimpleControls {
        momentum: MomentumControls {
            nu: 0.01,
            ..MomentumControls::default()
        },
        ..SimpleControls::default()
    };
    let mut sim = Simple::new(
        gpu,
        &hm,
        &gm,
        ctrl,
        BuoyancyCoeffs { g: Vec3::ZERO, ..BuoyancyCoeffs::default() },
    )?;
    setup_vector_field_with(
        gpu,
        sim.u_mut(),
        &raw_u(
            vec![
                ("yMin", ps("noSlip")),
                ("yMax", lid),
                ("xMin", ps("noSlip")),
                ("xMax", ps("noSlip")),
                ("zMin", ps("noSlip")),
                ("zMax", ps("noSlip")),
            ],
            Vec3::ZERO,
        ),
        &hm,
        BcInputs::default(),
    )?;
    setup_scalar_field(
        gpu,
        sim.p_mut(),
        &raw_p(vec![
            ("yMin", p0()),
            ("yMax", ps("zeroGradient")),
            ("xMin", ps("zeroGradient")),
            ("xMax", ps("zeroGradient")),
            ("zMin", ps("zeroGradient")),
            ("zMax", ps("zeroGradient")),
        ]),
        &hm,
    )?;
    sim.initialise(gpu)?;
    let nut = laminar_nut(gpu, &hm, &gm)?;
    let t = uniform_t(gpu, &hm, &gm)?;
    let mut backend = PbicgstabBackend::new(ctrl.p_solver);
    backend.setup(gpu, &hm, &gm, &SystemProbe::default())?;
    for _ in 0..2 {
        sim.correct(gpu, &mut backend, &nut, &t)?;
    }
    gpu.sync()?;
    let mut total = 0.0f64;
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        sim.correct(gpu, &mut backend, &nut, &t)?;
        gpu.sync()?;
        total += t0.elapsed().as_secs_f64() * 1.0e3;
    }
    Ok(total / 5.0)
}

/// T6: what the f64 geometry costs per step, printed; the 10 % gate holds
/// only at N >= 100, where docs/17's F32-11 claim is made.
#[test]
fn the_f64_geometry_cost_is_printed() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let n: usize = std::env::var("OFGPU_ALE_COST_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let t_geom = { timed_advance(&gpu, n)? };
    let t_simple = timed_simple(&gpu, n)?;
    let ratio = t_geom / t_simple;
    println!(
        "F32-11 cost: N^3 = {} cells, advance {:.3} ms, SIMPLE step {:.3} ms, \
         ratio {:.4}",
        n * n * n,
        t_geom,
        t_simple,
        ratio
    );
    if n >= 100 {
        assert!(
            ratio <= 0.10,
            "the f64 geometry costs {ratio:.3} of one SIMPLE step at {n}^3"
        );
    }
    Ok(())
}
