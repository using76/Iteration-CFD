// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 118.7 — the local origin. A mesh built by
//! `build_host_mesh_local` is stored about a snapped f64 origin `O`, so the
//! kernels' absolute f32 cell-centre differences stay small however far from
//! the coordinate axis the case sits; the f64 build never shifts. An
//! integration test because the library's own test target does not compile
//! under `single`; it runs in BOTH builds with the same assertions —
//! docs/17 §5.1 rule 4: class R on the two cavity gates (1e-5 of the lid
//! velocity in both builds), class D on the bitwise equalities, the counts
//! and the refusal. The host gates run on any machine; the SIMPLE gates pass
//! vacuously without a card, the convention of `tests/f32_boussinesq.rs`.
//!
//! Provenance: ORIGINAL — the meshes are this tree's own generators; no
//! GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::field_setup::{setup_scalar_field, setup_vector_field_with, BcInputs};
use ofgpu::fv::{DivScheme, SnGradScheme};
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::io::polymesh::{build_host_mesh, build_host_mesh_local, PolyMeshRaw};
use ofgpu::io::{usda, writer::cartesian_info};
use ofgpu::mesh::geometry::snapped_origin_axis;
use ofgpu::mesh::gpugeom::{gpu_compute_geometry, MeshGeomKernels};
use ofgpu::momentum::{BuoyancyCoeffs, MomentumControls};
use ofgpu::pressure::cartesian::detect;
use ofgpu::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use ofgpu::simple::{Simple, SimpleControls};
use ofgpu::solver::SolverControls;
use ofgpu::sources::CellSelector;
use ofgpu::types::{to_vec3, DVec3};
use ofgpu::{
    BcKind, Gpu, GpuMesh, GpuScalarField, HostMesh, Label, Result, Scalar, Vec3,
};

/// The build's `Scalar` widened to the f64 the answers are compared in: a
/// no-op in the f64 build, the exact widening under `single`.
#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    f64::from(x)
}

/// `tests/bc_wall_slip.rs`'s `box_spec`: `[lo, hi]` axes, uniform, the given
/// patch types in `(xMin, xMax, yMin, yMax, zMin, zMax)` order.
fn box_spec(n: [usize; 3], lo: [Scalar; 3], hi: [Scalar; 3], types: [&str; 6]) -> BlockSpec {
    let axis = |lo: Scalar, hi: Scalar, n: usize| GradedAxis {
        lo,
        hi,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    BlockSpec {
        x: axis(lo[0], hi[0], n[0]),
        y: axis(lo[1], hi[1], n[1]),
        z: axis(lo[2], hi[2], n[2]),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: types.map(String::from),
        cyclic: Vec::new(),
    }
}

/// The cavity on an `n`x`n`x1 cell mesh: `[0,1]x[0,1]x[0,0.1]`, the
/// lid-driven cavity of `tests/bc_wall_slip.rs`, walls on x and y, `empty`
/// front and back.
fn cavity_spec(n: usize) -> BlockSpec {
    box_spec(
        [n, n, 1],
        [0.0; 3],
        [1.0, 1.0, 0.1],
        ["wall", "wall", "wall", "wall", "empty", "empty"],
    )
}

/// "Translated" always means this: the raw `n`x`n` mesh with every point at
/// +2000 in x and y, the far-from-origin case whose geometry the f32 build
/// used to lose.
fn translated_raw_n(n: usize) -> PolyMeshRaw {
    let mut raw = blockgen::raw_mesh(&cavity_spec(n)).expect("raw");
    for p in raw.points.iter_mut() {
        p.x += 2000.0;
        p.y += 2000.0;
    }
    raw
}

/// The 32x32 translated raw of T1, T2, T5, T6 and T7.
fn translated_raw() -> PolyMeshRaw {
    translated_raw_n(32)
}

/// A = `build_host_mesh` of the untranslated `n`x`n` spec.
fn mesh_a_n(n: usize) -> HostMesh {
    let raw = blockgen::raw_mesh(&cavity_spec(n)).expect("raw");
    build_host_mesh(&raw).expect("A")
}

/// B = `build_host_mesh_local` of the translated `n`x`n` raw.
fn mesh_b_n(n: usize) -> HostMesh {
    build_host_mesh_local(&translated_raw_n(n)).expect("B")
}

/// The 32x32 pair T1, T2, T5, T6 and T7 run on.
fn mesh_a() -> HostMesh {
    mesh_a_n(32)
}

/// The 32x32 local-origin mesh of the same five tests.
fn mesh_b() -> HostMesh {
    mesh_b_n(32)
}

fn scalar_bits(a: &[Scalar], b: &[Scalar]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The `geom64()` view's scalars, which are f64 in BOTH builds (the `Cow`'s
/// element type does not name `Scalar`, so this is its own function).
fn f64_bits(a: &[f64], b: &[f64]) -> bool {
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

fn dvec3_bits(a: &[DVec3], b: &[DVec3]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(p, q)| {
            p.x.to_bits() == q.x.to_bits()
                && p.y.to_bits() == q.y.to_bits()
                && p.z.to_bits() == q.z.to_bits()
        })
}

/// All 16 geometry arrays - stored AND in the `geom64()` view - bit for bit
/// between two meshes.
fn geometry_is_bitwise(a: &HostMesh, b: &HostMesh) -> bool {
    let (ga, gb) = (a.geom64(), b.geom64());
    scalar_bits(&a.v, &b.v)
        && vec3_bits(&a.c, &b.c)
        && vec3_bits(&a.sf, &b.sf)
        && scalar_bits(&a.mag_sf, &b.mag_sf)
        && vec3_bits(&a.cf, &b.cf)
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
        && f64_bits(&ga.v[..], &gb.v[..])
        && dvec3_bits(&ga.c[..], &gb.c[..])
        && dvec3_bits(&ga.sf[..], &gb.sf[..])
        && f64_bits(&ga.mag_sf[..], &gb.mag_sf[..])
        && dvec3_bits(&ga.cf[..], &gb.cf[..])
        && f64_bits(&ga.weights[..], &gb.weights[..])
        && f64_bits(&ga.delta_coeffs[..], &gb.delta_coeffs[..])
        && dvec3_bits(&ga.non_orth_corr[..], &gb.non_orth_corr[..])
        && dvec3_bits(&ga.skew_corr[..], &gb.skew_corr[..])
        && dvec3_bits(&ga.b_sf[..], &gb.b_sf[..])
        && f64_bits(&ga.b_mag_sf[..], &gb.b_mag_sf[..])
        && dvec3_bits(&ga.b_cf[..], &gb.b_cf[..])
        && f64_bits(&ga.b_delta_coeffs[..], &gb.b_delta_coeffs[..])
        && dvec3_bits(&ga.b_non_orth_corr[..], &gb.b_non_orth_corr[..])
        && f64_bits(&ga.b_y[..], &gb.b_y[..])
        && f64_bits(&ga.b_weights[..], &gb.b_weights[..])
}

/// T1: the axis rule is EXACTLY (118.7a) - zero on an axis through 0, on a
/// degenerate axis and on a non-finite one, and the multiple of `q` nearest
/// the box centre otherwise. Class D: the same twelve numbers in both builds.
#[test]
fn the_origin_rule_is_118_7a() {
    let cases: &[([f64; 2], f64)] = &[
        ([0.0, 1.0], 0.0),
        ([-1.0, 0.0], 0.0),
        ([-0.5, 0.5], 0.0),
        ([5.0, 5.0], 0.0),
        ([f64::NAN, 1.0], 0.0),
        ([3.0, 100.0], 0.0),
        ([1.0, 2.0], 1.0),
        ([-2.0, -1.0], -1.0),
        ([2000.0, 2001.0], 2000.0),
        ([-2001.0, -2000.0], -2000.0),
        ([10.0, 11.5], 10.0),
        ([8.0, 16.0], 8.0),
        ([0.3, 0.4], 0.375),
        ([2000.0, 2000.03125], 2000.0),
        ([1.0e6, 1.0e6 + 10.0], 1.0e6),
    ];
    for &([lo, hi], want) in cases {
        assert_eq!(snapped_origin_axis(lo, hi), want, "axis [{lo}, {hi}]");
    }
}

/// T2: the local-origin build of the translated mesh is the origin mesh BIT
/// FOR BIT under `single` - the whole point of the snapped origin - and the
/// f64 build never shifts. Class D.
#[test]
fn a_translated_mesh_is_the_origin_mesh_bit_for_bit() {
    let a = mesh_a();
    let b = mesh_b();
    assert_eq!(a.origin, DVec3::ZERO, "the unshifted mesh has no origin");
    if cfg!(feature = "single") {
        assert_eq!(b.origin, DVec3::new(2000.0, 2000.0, 0.0), "(118.7a)");
        assert!(
            geometry_is_bitwise(&a, &b),
            "the local-origin mesh is the origin mesh bit for bit"
        );
    } else {
        assert_eq!(b.origin, DVec3::ZERO, "the f64 build never shifts");
        let c = build_host_mesh(&translated_raw()).expect("C");
        assert!(
            geometry_is_bitwise(&b, &c),
            "f64: build_host_mesh_local IS build_host_mesh"
        );
    }
    // World = local + origin, and local(world(c)) == c exactly.
    for i in 0..a.n_cells {
        let want = DVec3::new(
            w64(a.c[i].x) + 2000.0,
            w64(a.c[i].y) + 2000.0,
            w64(a.c[i].z),
        );
        let w = b.world(b.c[i]);
        assert!(
            (w.x - want.x).abs() <= 1e-12
                && (w.y - want.y).abs() <= 1e-12
                && (w.z - want.z).abs() <= 1e-12,
            "world(B.c[{i}]) = {w:?}, want {want:?}"
        );
        assert_eq!(b.local(w), b.c[i], "local(world(c)) == c, cell {i}");
    }
}

/// The cavity of `tests/bc_wall_slip.rs`'s `no_flux_through_a_partial_slip_wall`,
/// on the given mesh: Re = 100 (`nu = 0.01`), the lid `yMax` at U = (1, 0, 0),
/// the other three walls `noSlip`, p zeroGradient except pinned at zero on
/// `yMin`. `iters` steady SIMPLE outer iterations, no convergence test; None
/// when any U component went non-finite, which is the caller's verdict.
fn run_cavity(
    hm: &HostMesh,
    gpu: &Gpu,
    iters: usize,
) -> Result<Option<(Vec<Vec3>, f64)>> {
    let gm = GpuMesh::upload(gpu, hm)?;
    let mut lid = ps("fixedValue");
    lid.value_v = vec![Vec3::new(1.0, 0.0, 0.0)];
    let ctrl = couette_controls(0.01);
    let mut sim = Simple::new(
        gpu,
        hm,
        &gm,
        ctrl,
        BuoyancyCoeffs {
            g: Vec3::ZERO,
            ..BuoyancyCoeffs::default()
        },
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
            ],
            Vec3::ZERO,
        ),
        hm,
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
        ]),
        hm,
    )?;
    sim.initialise(gpu)?;
    let nut = laminar_nut(gpu, hm, &gm)?;
    let t = uniform_t(gpu, hm, &gm)?;
    let mut backend = PbicgstabBackend::new(ctrl.p_solver);
    backend.setup(gpu, hm, &gm, &SystemProbe::default())?;

    let mut prev = gpu.download(&sim.u().f)?;
    let mut last_change = 0.0f64;
    for it in 0..iters {
        sim.correct(gpu, &mut backend, &nut, &t)?;
        let now = gpu.download(&sim.u().f)?;
        if now.iter().any(|v| {
            !w64(v.x).is_finite() || !w64(v.y).is_finite() || !w64(v.z).is_finite()
        }) {
            println!("run_cavity: U went non-finite at iteration {it}");
            return Ok(None);
        }
        last_change = prev
            .iter()
            .zip(&now)
            .fold(0.0f64, |m, (a, b)| {
                m.max((w64(b.x) - w64(a.x)).abs())
                    .max((w64(b.y) - w64(a.y)).abs())
                    .max((w64(b.z) - w64(a.z)).abs())
            });
        prev = now;
    }
    Ok(Some((prev, last_change)))
}

/// A raw vector field named U, one uniform internal value, the patch entries
/// (`tests/bc_wall_slip.rs`'s helper).
fn raw_u(boundary: Vec<(&str, PatchFieldSpec)>, internal: Vec3) -> RawVectorField {
    let mut raw = RawVectorField {
        name: "U".into(),
        ..Default::default()
    };
    for (p, s) in boundary {
        raw.boundary.insert(p.into(), s);
    }
    raw.internal = vec![internal];
    raw
}

/// A raw scalar field named p, uniform zero inside, the patch entries.
fn raw_p(boundary: Vec<(&str, PatchFieldSpec)>) -> RawScalarField {
    let mut raw = RawScalarField {
        name: "p".into(),
        ..Default::default()
    };
    for (p, s) in boundary {
        raw.boundary.insert(p.into(), s);
    }
    raw.internal = vec![0.0];
    raw
}

/// A patch entry carrying nothing but its type.
fn ps(ty: &str) -> PatchFieldSpec {
    PatchFieldSpec {
        type_name: ty.into(),
        ..Default::default()
    }
}

/// p pinned at zero on one wall: the null space is fixed (see
/// `tests/bc_wall_slip.rs`).
fn p0() -> PatchFieldSpec {
    let mut s = ps("fixedValue");
    s.value = vec![0.0];
    s
}

/// Steady, the given laminar nu, Uncorrected snGrad on an orthogonal mesh,
/// tight solvers - what the gates measure is the discretisation. The div
/// scheme is `linearUpwind` (`run_cavity`'s only callers are T3 and T4):
/// with the default upwind no device kernel reads a stored position, so the
/// scheme choice decides whether the f32 rounding can reach the field.
fn couette_controls(nu: Scalar) -> SimpleControls {
    SimpleControls {
        momentum: MomentumControls {
            nu,
            u_solver: tight(),
            sn_grad: SnGradScheme::Uncorrected,
            div_scheme: DivScheme::LinearUpwind,
            ..MomentumControls::default()
        },
        p_solver: tight(),
        ..SimpleControls::default()
    }
}

/// Tight enough that what the gates measure is not the linear solver's
/// stopping criterion.
fn tight() -> SolverControls {
    SolverControls {
        tolerance: 1e-14,
        rel_tol: 0.0,
        max_iter: 500,
        check_interval: 1,
        ..SolverControls::default()
    }
}

/// Laminar: zero eddy viscosity with `calculated` boundary values.
fn laminar_nut(gpu: &Gpu, hm: &HostMesh, gm: &GpuMesh) -> Result<GpuScalarField> {
    let mut nut = GpuScalarField::zeros(gpu, gm, "nut")?;
    gpu.write(
        &mut nut.bc_kind,
        &vec![BcKind::Calculated as Label; hm.n_boundary_faces],
    )?;
    Ok(nut)
}

/// A uniform T, zeroGradient everywhere - nothing reads it while the
/// buoyancy vector is zero, but `correct` takes one.
fn uniform_t(gpu: &Gpu, hm: &HostMesh, gm: &GpuMesh) -> Result<GpuScalarField> {
    let mut t = GpuScalarField::zeros(gpu, gm, "T")?;
    gpu.write(
        &mut t.bc_kind,
        &vec![BcKind::ZeroGradient as Label; hm.n_boundary_faces],
    )?;
    Ok(t)
}

/// `max_c |U_x[c] - U_y[c]|` in f64, the two fields' own cells zipped.
fn worst_cell_diff(ua: &[Vec3], ub: &[Vec3]) -> f64 {
    ua.iter().zip(ub).fold(0.0f64, |m, (a, b)| {
        m.max((w64(b.x) - w64(a.x)).abs())
            .max((w64(b.y) - w64(a.y)).abs())
            .max((w64(b.z) - w64(a.z)).abs())
    })
}

/// T3 - the unit's gate: the translated cavity solved about its LOCAL origin
/// lands on the origin cavity's field, within 1e-5 of the lid velocity, in
/// BOTH builds. Class R.
///
/// The div scheme is `linearUpwind`, not the default upwind, because with
/// upwind on an orthogonal mesh no device kernel reads a stored position -
/// the weights, areas and delta coefficients were already computed in f64
/// (SPEC-LIT 118.2) - so the digits `C` lost to the translation never reach
/// the solution. The second-order schemes' kernels DO difference stored
/// positions (`fvWeightsLimited`'s `C[n] - C[o]`, the corrections'
/// `Cf[f] - C[c]`), which is what lets the rounding reach the field the gate
/// measures.
///
/// 30 cells, not the 32 of the other tests: h = 1/32 makes every coordinate
/// 2000 + k/64, a multiple of 2^-6, which f32 holds exactly at 2000 (ulp
/// 2^-13), so run 1's unshifted mesh lost nothing and the gate saw 0.000e0.
/// h = 1/30 is not a dyadic number, so 2000 + x is NOT exact in f32 and the
/// test discriminates.
#[test]
fn f32_translated_cavity_matches() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let a = mesh_a_n(30);
    let b = mesh_b_n(30);
    let (ua, chg) = run_cavity(&a, &gpu, 500)?.expect("U_A stays finite");
    let (ub, _) = run_cavity(&b, &gpu, 500)?.expect("U_B stays finite");
    let worst = worst_cell_diff(&ua, &ub);
    let max_a = ua
        .iter()
        .fold(0.0f64, |m, v| m.max(w64(v.x).abs()).max(w64(v.y).abs()));
    let scheme = DivScheme::LinearUpwind.describe();
    println!(
        "T3 [div = {scheme}]: max_c |U_B - U_A|/U_lid = {worst:.3e} (gate \
         1e-5), max |U_A| = {max_a:.6}, last change of U_A = {chg:.3e}"
    );
    assert!(
        ua.iter().chain(ub.iter()).all(|v| {
            w64(v.x).is_finite() && w64(v.y).is_finite() && w64(v.z).is_finite()
        }),
        "every U component is finite"
    );
    if cfg!(feature = "single") {
        assert_eq!(b.origin, DVec3::new(2000.0, 2000.0, 0.0), "(118.7a)");
        assert!(
            geometry_is_bitwise(&a, &b),
            "the 30x30 local-origin mesh is the origin mesh bit for bit"
        );
    }
    assert!(
        worst <= 1e-5,
        "the local-origin cavity missed the origin cavity: {worst:.3e} > 1e-5"
    );
    Ok(())
}

/// T4 - the state the unit exists to fix: the SAME run on the unshifted
/// translated mesh. Under `single` it must MISS the 1e-5 gate (predicted
/// about 1e-3) - or go non-finite, which is also the miss; in f64 it does
/// not miss, because f64 never loses the geometry.
///
/// The div scheme is `linearUpwind` for T3's reason: upwind reads no stored
/// position on an orthogonal mesh (the geometry was computed in f64, SPEC-LIT
/// 118.2), so under `single` the translation's lost digits would never reach
/// the solution and the miss would hide. linearUpwind's correction
/// differences stored positions (`Cf[f] - C[c]`), which is what surfaces it.
///
/// 30 cells, not 32: h = 1/32 makes every coordinate 2000 + k/64, a
/// multiple of 2^-6 that f32 holds exactly at 2000 (ulp 2^-13), so run 1's
/// unshifted mesh lost nothing and the gate saw 0.000e0; h = 1/30 is not a
/// dyadic number, so the f32 build rounds and the miss shows.
#[test]
fn the_unshifted_translated_cavity_misses_under_single() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let a = mesh_a_n(30);
    let c = build_host_mesh(&translated_raw_n(30))?;
    let (ua, chg) = run_cavity(&a, &gpu, 500)?.expect("U_A stays finite");
    let max_a = ua
        .iter()
        .fold(0.0f64, |m, v| m.max(w64(v.x).abs()).max(w64(v.y).abs()));
    let mut worst = f64::NAN;
    let scheme = DivScheme::LinearUpwind.describe();
    let miss = match run_cavity(&c, &gpu, 500)? {
        Some((uc, _)) => {
            worst = worst_cell_diff(&ua, &uc);
            worst > 1e-5
        }
        None => {
            println!(
                "T4 [div = {scheme}]: U on the unshifted translated mesh \
                 went NON-FINITE - that is the miss"
            );
            true
        }
    };
    if cfg!(feature = "single") {
        println!(
            "T4 [div = {scheme}]: max_c |U_C - U_A|/U_lid = {worst:.3e} \
             (predicted about 1e-3), max |U_A| = {max_a:.6}, last change of \
             U_A = {chg:.3e}"
        );
        assert!(
            miss,
            "the unshifted translated mesh should miss the 1e-5 gate under \
             single, got {worst:.3e}"
        );
    } else {
        println!(
            "T4 (f64) [div = {scheme}]: max_c |U_C - U_A|/U_lid = {worst:.3e}, \
             max |U_A| = {max_a:.6}, last change of U_A = {chg:.3e} - f64 \
             loses nothing to the translation"
        );
        assert!(worst <= 1e-5, "f64: {worst:.3e} > 1e-5");
    }
    Ok(())
}

/// T5: Box and Sphere selectors are stated in WORLD coordinates - the same
/// cells on the origin mesh and on the local-origin mesh. Class D.
#[test]
fn selectors_are_stated_in_world_coordinates() {
    let a = mesh_a();
    let b = mesh_b();
    let box_a = CellSelector::Box {
        min: Vec3::new(0.25, 0.25, -1.0),
        max: Vec3::new(0.75, 0.75, 1.0),
    };
    let box_b = CellSelector::Box {
        min: Vec3::new(2000.25, 2000.25, -1.0),
        max: Vec3::new(2000.75, 2000.75, 1.0),
    };
    let (sa, sb) = (box_a.select(&a), box_b.select(&b));
    assert_eq!(sa.len(), 256, "16x16 of the 32x32 mesh");
    assert_eq!(sa, sb, "the same cells, stated world or local");
    let sph_a = CellSelector::Sphere {
        centre: Vec3::new(0.5, 0.5, 0.05),
        radius: 0.25,
    };
    let sph_b = CellSelector::Sphere {
        centre: Vec3::new(2000.5, 2000.5, 0.05),
        radius: 0.25,
    };
    let (sa, sb) = (sph_a.select(&a), sph_b.select(&b));
    assert!(!sa.is_empty(), "the sphere is not empty on the origin mesh");
    assert_eq!(sa, sb, "the same cells, stated world or local");
}

/// T6: the writers put WORLD positions on disk. Class R on the translation:
/// a stored f32 world coordinate carries `ulp(2000) = 1.22e-4`, so four of
/// those is the tolerance both writers are held to.
#[test]
fn writers_put_world_positions_on_disk() {
    let a = mesh_a();
    let b = mesh_b();
    let tol = 4.883e-4; // 4 ulp(2000)
    let (sa, sb) = (usda::boundary_surface(&a), usda::boundary_surface(&b));
    assert_eq!(
        sa.points.len(),
        sb.points.len(),
        "the same faces survive the degeneracy skip"
    );
    for (k, (pa, pb)) in sa.points.iter().zip(&sb.points).enumerate() {
        assert!(
            (w64(pb.x) - (w64(pa.x) + 2000.0)).abs() <= tol
                && (w64(pb.y) - (w64(pa.y) + 2000.0)).abs() <= tol
                && (w64(pb.z) - w64(pa.z)).abs() <= tol,
            "boundary_surface point {k}: {pb:?} vs {:?} + (2000, 2000, 0)",
            pa
        );
    }
    let ga = cartesian_info(&a, &detect(&a).expect("A is cartesian"));
    let gb = cartesian_info(&b, &detect(&b).expect("B is cartesian"));
    assert!(
        (w64(gb.origin.x) - 2000.0).abs() <= tol
            && (w64(gb.origin.y) - 2000.0).abs() <= tol
            && w64(gb.origin.z).abs() <= tol,
        "cartesian_info(B).origin = {:?}, want (2000, 2000, 0)",
        gb.origin
    );
    assert!(
        w64(ga.origin.x).abs() <= 1e-6
            && w64(ga.origin.y).abs() <= 1e-6
            && w64(ga.origin.z).abs() <= 1e-6,
        "cartesian_info(A).origin = {:?}, want (0, 0, 0)",
        ga.origin
    );
}

/// T7: a path not converted yet REFUSES a local origin by name, and never
/// runs shifted and wrong. The device geometry sweep is F32-11's; in the f64
/// build there is no local origin, so the sweep runs. Class D.
#[test]
fn unconverted_paths_refuse_a_local_origin() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let mut b = mesh_b();
    let raw = translated_raw();
    let points: Vec<Vec3> = raw.points.iter().map(|&p| to_vec3(p)).collect();
    let k = MeshGeomKernels::new(&gpu)?;
    let r = gpu_compute_geometry(&gpu, &k, &mut b, &points, &raw.faces);
    if cfg!(feature = "single") {
        let msg = r.expect_err("the device sweep refuses a local origin").to_string();
        assert!(
            msg.contains("local origin") && msg.contains("F32-11"),
            "the refusal names the path and the unit: {msg}"
        );
    } else {
        r.expect("the f64 build has no local origin and the sweep runs");
    }
    Ok(())
}
