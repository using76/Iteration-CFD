// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 120.5 - the velocity-inlet family through the public API, in
//! BOTH builds: the three fixed-value profiles against their closed forms
//! ((120.5a)-(120.5c)) and the swirl number (120.5d) they deliver, the five
//! pressure-driven kinds against their host twin `field::pressure_inlet_values`
//! ((120.5e)-(120.5i)), the pressure-driven channel at the Poiseuille rate
//! ((120.5j)/(120.5k)), and the refusals by name. The tolerances follow
//! docs/17 §5.1 class R, D-F32-1 - the f64 tolerance in the f64 build,
//! `max(t, min(t*2^29, 1e-5))` under `single` - which the `tol` const fn
//! below encodes; everything is compared in f64. A machine without a card
//! passes vacuously, the convention of `tests/bc_wall_slip.rs`.
//!
//! No GPL-licensed source was consulted.

use std::f64::consts::PI;

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::field::pressure_inlet_values;
use ofgpu::field_ops::{
    correct_boundary_conditions_vector, update_inlet_outlet_vector,
    update_pressure_inlet_velocity, FieldKernels,
};
use ofgpu::field_setup::{setup_scalar_field, setup_vector_field, setup_vector_field_with, BcInputs};
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::momentum::{BuoyancyCoeffs, MomentumControls};
use ofgpu::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use ofgpu::simple::{Simple, SimpleControls};
use ofgpu::solver::SolverControls;
use ofgpu::types::DVec3;
use ofgpu::{
    BcKind, Gpu, GpuMesh, GpuScalarField, GpuSurfaceScalarField, GpuVectorField, HostMesh,
    Result, Scalar, Vec3,
};

/// `t` in the f64 build; under `single` (class R, D-F32-1)
/// `max(t, min(t*2^29, 1e-5))` - i.e. 1e-12 -> 1e-5, 1e-14 -> 5.36870912e-6.
const fn tol(t: f64) -> f64 {
    if cfg!(feature = "single") {
        let scaled = t * 536870912.0;
        let capped = if scaled < 1e-5 { scaled } else { 1e-5 };
        if t > capped { t } else { capped }
    } else {
        t
    }
}

/// The build's `Scalar` widened to the f64 the answer keys are computed in.
#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    x as f64
}

/// A NaN-propagating max: `f64::max` swallows NaN and `f64::maximum` is
/// unstable on this toolchain, so fold by hand - NaN in, NaN out.
fn nmax(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// A checked term for the folds: a non-finite term panics here, naming
/// itself, instead of vanishing through `f64::max`.
fn fin(what: &str, x: f64) -> f64 {
    assert!(x.is_finite(), "{what} is non-finite: {x:e}");
    x
}

/// The f64 unit outward normal of boundary face `f`, normalised from the
/// mesh's own f64 `geom64` `b_sf` (SPEC-LIT 120.5, §118.2).
fn normal64(hm: &HostMesh, f: usize) -> DVec3 {
    let sf = hm.geom64().b_sf[f];
    let a = (sf.x * sf.x + sf.y * sf.y + sf.z * sf.z).sqrt();
    if a > 0.0 {
        DVec3::new(sf.x / a, sf.y / a, sf.z / a)
    } else {
        DVec3::ZERO
    }
}

/// A uniform block, every patch typed by the caller - the `spec` shape of
/// `tests/bc_wall_slip.rs`, sized by the caller.
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

/// The index of a patch by name.
fn patch_of(hm: &HostMesh, name: &str) -> usize {
    hm.patches
        .iter()
        .position(|p| p.name == name)
        .unwrap_or_else(|| panic!("no patch {name}"))
}

/// A raw vector field named U, one uniform internal value, the given patch
/// entries.
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

/// A raw scalar field named p, uniform zero inside, the given patch entries.
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

/// p pinned at the given uniform value on one patch.
fn p_fixed(v: Scalar) -> PatchFieldSpec {
    let mut s = ps("fixedValue");
    s.value = vec![v];
    s
}

/// Tight enough that what the tests measure is not the linear solver's
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
        &vec![BcKind::Calculated as i32; hm.n_boundary_faces],
    )?;
    Ok(nut)
}

/// A uniform T at rest, zeroGradient everywhere - nothing reads it while the
/// buoyancy vector is zero, but `correct` takes one.
fn uniform_t(gpu: &Gpu, hm: &HostMesh, gm: &GpuMesh) -> Result<GpuScalarField> {
    let mut t = GpuScalarField::zeros(gpu, gm, "T")?;
    gpu.write(
        &mut t.bc_kind,
        &vec![BcKind::ZeroGradient as i32; hm.n_boundary_faces],
    )?;
    Ok(t)
}

// ==========================================================================
//  The three fixed-value profiles - (120.5a)-(120.5d)
// ==========================================================================

/// The patch entry of each of the three fixed-value inlets: 0
/// `surfaceNormalFixedValue`, 1 `cylindricalInletVelocity`, 2
/// `swirlFlowRateInletVelocity`.
fn inlet_spec(which: usize) -> PatchFieldSpec {
    let mut s = ps(match which {
        0 => "surfaceNormalFixedValue",
        1 => "cylindricalInletVelocity",
        _ => "swirlFlowRateInletVelocity",
    });
    match which {
        0 => {
            s.extra.insert("refValue".into(), "uniform -2.5".into());
        }
        1 => {
            s.extra.insert("origin".into(), "(0 0.5 0.5)".into());
            s.extra.insert("axis".into(), "(1 0 0)".into());
            s.extra.insert("axialVelocity".into(), "1.5".into());
            s.extra.insert("radialVelocity".into(), "0.25".into());
            s.extra.insert("rpm".into(), "60".into());
        }
        _ => {
            s.extra.insert("origin".into(), "(0 0.5 0.5)".into());
            s.extra.insert("axis".into(), "(1 0 0)".into());
            s.extra.insert("rpm".into(), "120".into());
            s.extra
                .insert("volumetricFlowRate".into(), "0.8".into());
        }
    }
    s
}

/// The closed form of each profile at patch face `i`, in f64 (SPEC-LIT
/// 120.5, (120.5a)-(120.5c)): `n64` the f64 outward normal, `centre` the f64
/// world face centre, `area` the f64 patch area.
fn closed_form(
    which: usize,
    n64: DVec3,
    centre: DVec3,
    area: f64,
) -> DVec3 {
    let a = DVec3::new(1.0, 0.0, 0.0);
    let d = centre - DVec3::new(0.0, 0.5, 0.5);
    match which {
        // (120.5a): U_b = u_n n, u_n signed.
        0 => DVec3::new(-2.5 * n64.x, -2.5 * n64.y, -2.5 * n64.z),
        // (120.5b): u_ax a + u_r e_r + omega (a x r), omega = 2 pi rpm/60.
        1 => {
            let r = d - a * a.dot(d);
            let rho = (r.x * r.x + r.y * r.y + r.z * r.z).sqrt();
            let e_r = if rho > 0.0 { r * (1.0 / rho) } else { DVec3::ZERO };
            let w = a.cross(r);
            let omega = 2.0 * PI * 60.0 / 60.0;
            DVec3::new(
                1.5 * a.x + 0.25 * e_r.x + omega * w.x,
                1.5 * a.y + 0.25 * e_r.y + omega * w.y,
                1.5 * a.z + 0.25 * e_r.z + omega * w.z,
            )
        }
        // (120.5c): -n Q/A + w - n (n . w), w = omega (a x r), Q = 0.8.
        _ => {
            let r = d - a * a.dot(d);
            let w = a.cross(r) * (2.0 * PI * 120.0 / 60.0);
            let nw = n64.x * w.x + n64.y * w.y + n64.z * w.z;
            DVec3::new(
                -n64.x * (0.8 / area) + w.x - n64.x * nw,
                -n64.y * (0.8 / area) + w.y - n64.y * nw,
                -n64.z * (0.8 / area) + w.z - n64.z * nw,
            )
        }
    }
}


/// The 6x6x6 unit box with the inlet on `xMin`, the outlet on `xMax`, and
/// noSlip walls elsewhere - the rig of the three fixed-value gates.
fn inlet_box() -> Result<(HostMesh, GpuMesh)> {
    let hm = blockgen::build_mesh(&box_spec(
        [6, 6, 6],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        ["patch", "patch", "wall", "wall", "wall", "wall"],
    ))?;
    let Some(gpu) = Gpu::new(0).ok() else {
        return Err(ofgpu::Error::Config("no card in inlet_box".into()));
    };
    let gm = GpuMesh::upload(&gpu, &hm)?;
    Ok((hm, gm))
}

#[test]
fn fixed_value_inlets_carry_their_patch_flux() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let (hm, gm) = inlet_box()?;
    let xin = patch_of(&hm, "xMin");
    let at = hm.patches[xin].start;
    let np = hm.patches[xin].size;

    // The f64 patch area the (120.5c) closed form reads.
    let area: f64 = (0..np)
        .map(|i| {
            let s = hm.geom64().b_sf[at + i];
            (s.x * s.x + s.y * s.y + s.z * s.z).sqrt()
        })
        .sum();
    assert!((area - 1.0).abs() < 1e-12, "xMin area {area}");

    // The patch flux each profile must deliver (b): -2.5 and -u_ax through
    // the unit square for (120.5a)/(120.5b), -Q for (120.5c).
    let want_flux = [-2.5, -1.5, -0.8];

    for which in 0..3 {
        let ctrl = SimpleControls {
            momentum: MomentumControls {
                nu: 0.01,
                u_solver: tight(),
                ..MomentumControls::default()
            },
            p_solver: tight(),
            ..SimpleControls::default()
        };
        let mut sim = Simple::new(
            &gpu,
            &hm,
            &gm,
            ctrl,
            BuoyancyCoeffs {
                g: Vec3::ZERO,
                ..BuoyancyCoeffs::default()
            },
        )?;
        setup_vector_field_with(
            &gpu,
            sim.u_mut(),
            &raw_u(
                vec![
                    ("xMin", inlet_spec(which)),
                    ("xMax", ps("zeroGradient")),
                    ("yMin", ps("noSlip")),
                    ("yMax", ps("noSlip")),
                    ("zMin", ps("noSlip")),
                    ("zMax", ps("noSlip")),
                ],
                Vec3::ZERO,
            ),
            &hm,
            BcInputs::default(),
        )?;
        setup_scalar_field(
            &gpu,
            sim.p_mut(),
            &raw_p(vec![
                ("xMin", ps("zeroGradient")),
                ("xMax", p_fixed(0.0)),
                ("yMin", ps("zeroGradient")),
                ("yMax", ps("zeroGradient")),
                ("zMin", ps("zeroGradient")),
                ("zMax", ps("zeroGradient")),
            ]),
            &hm,
        )?;
        sim.initialise(&gpu)?;
        let nut = laminar_nut(&gpu, &hm, &gm)?;
        let t = uniform_t(&gpu, &hm, &gm)?;
        let mut backend = PbicgstabBackend::new(ctrl.p_solver);
        backend.setup(&gpu, &hm, &gm, &SystemProbe::default())?;

        // (a): every face value equals the closed form computed in f64,
        // within tol(1e-14)*max|U|.
        let bf = gpu.download(&sim.u().bf)?;
        let mut max_u = 0.0f64;
        for i in 0..np {
            let g = bf[at + i];
            max_u = nmax(max_u, w64(g.x).abs());
            max_u = nmax(max_u, w64(g.y).abs());
            max_u = nmax(max_u, w64(g.z).abs());
        }
        for i in 0..np {
            let want =
                closed_form(which, normal64(&hm, at + i), hm.geom64().b_cf[at + i], area);
            let g = bf[at + i];
            let dd = nmax(
                nmax((w64(g.x) - want.x).abs(), (w64(g.y) - want.y).abs()),
                (w64(g.z) - want.z).abs(),
            );
            fin("U_b", dd);
            assert!(
                dd <= tol(1e-14) * max_u,
                "case {which} face {i}: worst component {dd:e} (bound {})",
                tol(1e-14) * max_u
            );
        }

        // (b): three outer iterations, and the patch sum of the downloaded
        // flux IS the patch sum of U_b . S_f - the prescribed-flux rule of
        // SPEC-LIT 120.5.
        for _ in 0..3 {
            sim.correct(&gpu, &mut backend, &nut, &t)?;
        }
        let phib = gpu.download(&sim.phi().bf)?;
        let mut sum = 0.0f64;
        for i in 0..np {
            sum += fin("phi_b", w64(phib[at + i]));
        }
        let want = want_flux[which];
        println!("  [120.5] case {which}: inlet flux sum {sum:.12} (want {want})");
        assert!(
            (sum - want).abs() <= tol(1e-12) * want.abs(),
            "case {which}: inlet flux sum {sum:e}, want {want}"
        );

        // (c), swirl only: (120.5d) from the downloaded flux and values
        // equals `omega SUM rho_f^2 |S_f|/(R Q)` - omega = 4 pi, Q = 0.8,
        // rho_f and R from the f64 face centres - and `U_b . a = Q/A` at
        // every face.
        if which == 2 {
            let omega = 2.0 * PI * 120.0 / 60.0;
            let u_bf = gpu.download(&sim.u().bf)?;
            let mut g_theta = 0.0f64;
            let mut g_x = 0.0f64;
            let mut key = 0.0f64;
            let mut r_max = 0.0f64;
            let mut faces = Vec::new();
            for i in 0..np {
                let f = at + i;
                let c = hm.geom64().b_cf[f];
                let (ry, rz) = (c.y - 0.5, c.z - 0.5);
                let rho = (ry * ry + rz * rz).sqrt();
                fin("rho_f", rho);
                r_max = r_max.max(rho);
                let s = hm.geom64().b_sf[f];
                let mag = (s.x * s.x + s.y * s.y + s.z * s.z).sqrt();
                key += omega * rho * rho * mag;
                // e_theta = a x e_r for a = x: (0, -rz, ry)/rho.
                faces.push((rho, -rz / rho, ry / rho));
            }
            for (i, (rho, ety, etz)) in faces.iter().enumerate() {
                let f = at + i;
                let u = u_bf[f];
                let phi = phib[f];
                let u_et = w64(u.y) * ety + w64(u.z) * etz;
                let u_a = w64(u.x);
                g_theta += -w64(phi) * rho * u_et;
                g_x += -w64(phi) * u_a;
                fin("U_b . a", u_a);
                assert!(
                    (u_a - 0.8).abs() <= tol(1e-14),
                    "swirl face {i}: U_b . a = {u_a} (bound {})",
                    tol(1e-14)
                );
            }
            let s_meas = g_theta / (r_max * g_x);
            let s_want = key / (r_max * 0.8);
            println!("  [120.5] swirl number (120.5d): {s_meas:.12} (want {s_want:.12})");
            assert!(
                (s_meas - s_want).abs() <= tol(1e-12) * s_want.abs(),
                "swirl number {s_meas:e}, want {s_want:e}"
            );
        }
    }
    Ok(())
}

// ==========================================================================
//  The five pressure-driven kinds - (120.5e)-(120.5i) against the twin
// ==========================================================================

#[test]
fn pressure_inlet_refresh_is_its_host_twin() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = blockgen::build_mesh(&box_spec(
        [4, 4, 4],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        ["patch", "patch", "patch", "patch", "patch", "patch"],
    ))?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let k = FieldKernels::new(&gpu)?;
    let n = hm.n_boundary_faces;

    // xMin 41, xMax 42, yMin 43, yMax 44 (inletDirection (0 -1 0.2)),
    // zMin 45, zMax zeroGradient; internal U_c = (1 + 0.1 x_c, -0.2 +
    // 0.3 y_c, 0.05 z_c).
    let mut d43 = ps("pressureDirectedInletVelocity");
    d43.extra
        .insert("inletDirection".into(), "uniform (0.3 1 0)".into());
    let mut d44 = ps("pressureDirectedInletOutletVelocity");
    d44.extra
        .insert("inletDirection".into(), "uniform (0 -1 0.2)".into());
    let internal: Vec<Vec3> = (0..hm.n_cells)
        .map(|c| {
            let p = hm.c[c];
            Vec3::new(
                (1.0 + 0.1 * w64(p.x)) as Scalar,
                (-0.2 + 0.3 * w64(p.y)) as Scalar,
                (0.05 * w64(p.z)) as Scalar,
            )
        })
        .collect();
    let mut raw = RawVectorField {
        name: "U".into(),
        internal,
        ..Default::default()
    };
    for (p, s) in [
        ("xMin", ps("pressureInletVelocity")),
        ("xMax", ps("pressureInletUniformVelocity")),
        ("yMin", d43),
        ("yMax", d44),
        ("zMin", ps("pressureInletOutletParSlipVelocity")),
        ("zMax", ps("zeroGradient")),
    ] {
        raw.boundary.insert(p.into(), s);
    }

    let mut u = GpuVectorField::zeros(&gpu, &gm, "U")?;
    setup_vector_field_with(
        &gpu,
        &mut u,
        &raw,
        &hm,
        BcInputs {
            pressure_inlet: true,
            ..BcInputs::default()
        },
    )?;

    // phi_b[f] = |S_f| (0.3 + sin(1.7 f)) - both signs, so the switched
    // kinds really switch.
    let phib = twin_flux(&hm);
    let mut phi = GpuSurfaceScalarField::zeros(&gpu, &gm, "phi")?;
    gpu.write(&mut phi.bf, &phib)?;

    update_inlet_outlet_vector(&gpu, &k, &mut u, &phi)?;
    update_pressure_inlet_velocity(&gpu, &k, &mut u, &phi, &gm)?;
    correct_boundary_conditions_vector(&gpu, &k, &mut u, &gm)?;
    gpu.sync()?;

    let kinds = gpu.download(&u.bc_kind)?;
    let fr = gpu.download(&u.fr)?;
    let rv_dev = gpu.download(&u.ref_value)?;
    let dm = gpu.download(&u.dm)?;
    let bf = gpu.download(&u.bf)?;
    let up: Vec<Vec3> = hm
        .b_face_cells
        .iter()
        .map(|&c| raw.internal[c as usize])
        .collect();

    // The host twin, in the same operation order.
    let mut rv_twin = vec![Vec3::ZERO; n];
    pressure_inlet_values(
        &kinds,
        &hm.b_patch,
        &phib,
        &hm.b_sf,
        &hm.b_mag_sf,
        &dm,
        &up,
        &mut rv_twin,
    );

    let mut max_ub = 0.0f64;
    let mut max_phi = 0.0f64;
    for i in 0..n {
        for v in [rv_dev[i], rv_twin[i]] {
            max_ub = nmax(max_ub, w64(v.x).abs());
            max_ub = nmax(max_ub, w64(v.y).abs());
            max_ub = nmax(max_ub, w64(v.z).abs());
        }
        max_phi = nmax(max_phi, w64(phib[i]).abs());
    }

    // ref_value equals the host twin within tol(1e-14)*max|U_b|.
    for i in 0..n {
        let (a, b) = (rv_dev[i], rv_twin[i]);
        let dd = nmax(
            nmax(
                (w64(a.x) - w64(b.x)).abs(),
                (w64(a.y) - w64(b.y)).abs(),
            ),
            (w64(a.z) - w64(b.z)).abs(),
        );
        fin("ref_value twin", dd);
        assert!(
            dd <= tol(1e-14) * max_ub,
            "face {i} (kind {}): twin gap {dd:e} (bound {})",
            kinds[i],
            tol(1e-14) * max_ub
        );
    }

    // fr exactly 1 on the three always-Dirichlet kinds, exactly phi < 0 on
    // the two switched ones.
    let k41 = BcKind::PressureInletVelocity as i32;
    let k43 = BcKind::PressureDirectedInletVelocity as i32;
    let k45 = BcKind::PressureInletOutletParSlipVelocity as i32;
    for i in 0..n {
        let kd = kinds[i];
        if (k41..=k43).contains(&kd) {
            assert_eq!(w64(fr[i]), 1.0, "fr face {i} kind {kd}");
        } else if (k43 + 1..=k45).contains(&kd) {
            let want = if w64(phib[i]) < 0.0 { 1.0 } else { 0.0 };
            assert_eq!(w64(fr[i]), want, "fr face {i} kind {kd}");
        }
    }

    // On every fr = 1 face U_b . S_f = phi_f - except kind 42, whose faces
    // only carry the patch sum (the gate checks PHI_p per patch instead).
    let k42 = BcKind::PressureInletUniformVelocity as i32;
    let k44 = BcKind::PressureDirectedInletOutletVelocity as i32;
    let mut phi_sum: std::collections::BTreeMap<i32, (f64, f64)> = Default::default();
    for i in 0..n {
        let kd = kinds[i];
        // Faces of every other kind are untouched by the rewrite and by
        // this gate.
        if kd < k41 || kd > k45 {
            continue;
        }
        let is1 = w64(fr[i]) == 1.0;
        let s = hm.b_sf[i];
        let b = bf[i];
        let flux = w64(b.x) * w64(s.x) + w64(b.y) * w64(s.y) + w64(b.z) * w64(s.z);
        fin("U_b . S_f", flux);
        if kd == k42 {
            let e = phi_sum.entry(hm.b_patch[i]).or_insert((0.0, 0.0));
            e.0 += flux;
            e.1 += w64(phib[i]);
            continue;
        }
        // The outflow half of the two switched kinds is zeroGradient by
        // (120.5h)/(120.5i) - fr = 0, no flux prescription to check.
        if (kd == k44 || kd == k45) && w64(phib[i]) >= 0.0 {
            assert_eq!(w64(fr[i]), 0.0, "outflow half face {i}");
            continue;
        }
        assert!(is1, "face {i} kind {kd} should be Dirichlet here");
        assert!(
            (flux - w64(phib[i])).abs() <= tol(1e-12) * max_phi,
            "face {i} kind {kd}: flux {flux:e} vs phi {}",
            w64(phib[i])
        );
        if kd == k41 {
            // U_b x S_f = 0: the value is parallel to the face normal.
            let cx = w64(b.y) * w64(s.z) - w64(b.z) * w64(s.y);
            let cy = w64(b.z) * w64(s.x) - w64(b.x) * w64(s.z);
            let cz = w64(b.x) * w64(s.y) - w64(b.y) * w64(s.x);
            let m = nmax(nmax(cx.abs(), cy.abs()), cz.abs());
            fin("U_b x S_f", m);
            assert!(m <= tol(1e-12) * max_phi, "face {i}: |cross| {m:e}");
        }
    }
    for (p, (got, want)) in &phi_sum {
        println!("  [120.5] patch {p}: sum U_b.S_f {got:.12} vs PHI_p {want:.12}");
        assert!(
            (got - want).abs() <= tol(1e-12) * max_phi,
            "patch {p}: sum {got:e} vs PHI_p {want:e}"
        );
    }

    // bf equals ref_value on every fr = 1 face: the section-4 evaluation
    // collapses to the reference value there.
    for i in 0..n {
        if w64(fr[i]) != 1.0 {
            continue;
        }
        let (b, r) = (bf[i], rv_dev[i]);
        assert_eq!(w64(b.x), w64(r.x), "bf x face {i}");
        assert_eq!(w64(b.y), w64(r.y), "bf y face {i}");
        assert_eq!(w64(b.z), w64(r.z), "bf z face {i}");
    }
    Ok(())
}

/// The flux of the twin gate: `phi_b[f] = |S_f| (0.3 + sin(1.7 f))` for
/// boundary face `f` - both signs, so the two switched kinds really switch.
fn twin_flux(hm: &HostMesh) -> Vec<Scalar> {
    (0..hm.n_boundary_faces)
        .map(|f| {
            (w64(hm.b_mag_sf[f]) * (0.3 + (1.7 * f as f64).sin())) as Scalar
        })
        .collect()
}

// ==========================================================================
//  The pressure-driven channel - (120.5j)/(120.5k)
// ==========================================================================

#[test]
fn a_pressure_driven_channel_reaches_the_poiseuille_rate() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    // 2-D: x in [0,2] 10 cells, y in [0,1] 20 cells, z one cell, empty
    // front and back. H = 1, L = 2, p_in = 24, nu = 1: (120.5j) Q = 1,
    // (120.5k) Q_h = Q (1 + 2/20^2) = 1.005 at n_y = 20.
    let hm = blockgen::build_mesh(&box_spec(
        [10, 20, 1],
        [0.0, 0.0, 0.0],
        [2.0, 1.0, 1.0],
        ["patch", "patch", "wall", "wall", "empty", "empty"],
    ))?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let xin = patch_of(&hm, "xMin");
    let xout = patch_of(&hm, "xMax");
    let atin = hm.patches[xin].start;
    let atout = hm.patches[xout].start;
    let nin = hm.patches[xin].size;
    let nout = hm.patches[xout].size;

    let ctrl = SimpleControls {
        momentum: MomentumControls {
            nu: 1.0,
            u_solver: tight(),
            ..MomentumControls::default()
        },
        p_solver: tight(),
        ..SimpleControls::default()
    };
    let mut sim = Simple::new(
        &gpu,
        &hm,
        &gm,
        ctrl,
        BuoyancyCoeffs {
            g: Vec3::ZERO,
            ..BuoyancyCoeffs::default()
        },
    )?;
    setup_vector_field_with(
        &gpu,
        sim.u_mut(),
        &raw_u(
            vec![
                ("xMin", ps("pressureInletVelocity")),
                ("xMax", ps("zeroGradient")),
                ("yMin", ps("noSlip")),
                ("yMax", ps("noSlip")),
                ("zMin", ps("empty")),
                ("zMax", ps("empty")),
            ],
            Vec3::ZERO,
        ),
        &hm,
        BcInputs {
            pressure_inlet: true,
            ..BcInputs::default()
        },
    )?;
    setup_scalar_field(
        &gpu,
        sim.p_mut(),
        &raw_p(vec![
            ("xMin", p_fixed(24.0)),
            ("xMax", p_fixed(0.0)),
            ("yMin", ps("zeroGradient")),
            ("yMax", ps("zeroGradient")),
            ("zMin", ps("empty")),
            ("zMax", ps("empty")),
        ]),
        &hm,
    )?;
    sim.initialise(&gpu)?;
    let nut = laminar_nut(&gpu, &hm, &gm)?;
    let t = uniform_t(&gpu, &hm, &gm)?;
    let mut backend = PbicgstabBackend::new(ctrl.p_solver);
    backend.setup(&gpu, &hm, &gm, &SystemProbe::default())?;

    // SIMPLE iterations until the change of U between iterations is below
    // 1e-14, at most 5000 of them - run_to_fixed_point's loop.
    let mut prev = gpu.download(&sim.u().f)?;
    let mut iterations = 0usize;
    for it in 0..5000 {
        sim.correct(&gpu, &mut backend, &nut, &t)?;
        iterations = it + 1;
        let now = gpu.download(&sim.u().f)?;
        let nonfinite = now.iter().enumerate().find(|(_, v)| {
            !w64(v.x).is_finite() || !w64(v.y).is_finite() || !w64(v.z).is_finite()
        });
        if let Some((c, v)) = nonfinite {
            return Err(ofgpu::Error::Diverged {
                iteration: it,
                what: format!("U cell {c} went non-finite: {v:?}"),
            });
        }
        let change = prev.iter().zip(&now).fold(0.0f64, |m, (a, b)| {
            m.max(cell_gap(*a, *b))
        });
        prev = now;
        if change < 1e-14 {
            break;
        }
    }

    // The rate: Q = -SUM_inlet phi_b, and the outlet carries it back.
    let phib = gpu.download(&sim.phi().bf)?;
    let q = -(0..nin)
        .map(|i| fin("phi_b", w64(phib[atin + i])))
        .sum::<f64>();
    let q_out = (0..nout)
        .map(|i| fin("phi_b", w64(phib[atout + i])))
        .sum::<f64>();
    let e_rate = (q - 1.0).abs();
    let e_exact = (q / 1.005 - 1.0).abs();
    eprintln!(
        "  [120.5] channel: Q = {q:.9} after {iterations} iterations; \
         |Q - 1| = {e_rate:e}, |Q/1.005 - 1| = {e_exact:e}, \
         outlet sum - Q = {:.e}",
        q_out - q
    );
    assert!(e_rate <= 1e-2, "Q = {q}, want 1 (class P)");
    assert!(e_exact <= tol(1e-6), "Q/1.005 - 1 = {e_exact:e}");
    assert!(
        (q_out - q).abs() <= tol(1e-10),
        "outlet sum {q_out} vs Q {q}"
    );
    // The inlet face: purely normal, and equal to S_f phi_b/(S . S) - the
    // (120.5e) the kernel refreshed it with.
    let ubf = gpu.download(&sim.u().bf)?;
    let mut max_u = 0.0f64;
    for i in 0..nin {
        let b = ubf[atin + i];
        max_u = nmax(max_u, w64(b.x).abs());
        max_u = nmax(max_u, w64(b.y).abs());
        max_u = nmax(max_u, w64(b.z).abs());
    }
    for i in 0..nin {
        let f = atin + i;
        let nn = normal64(&hm, f);
        let b = ubf[f];
        let un = w64(b.x) * nn.x + w64(b.y) * nn.y + w64(b.z) * nn.z;
        let tang = nmax(
            (w64(b.x) - un * nn.x).abs(),
            nmax(
                (w64(b.y) - un * nn.y).abs(),
                (w64(b.z) - un * nn.z).abs(),
            ),
        );
        fin("tangential U_b", tang);
        assert!(
            tang <= tol(1e-12) * max_u,
            "face {i}: tangential {tang:e}"
        );
        let s = hm.b_sf[f];
        let ss = w64(s.x) * w64(s.x) + w64(s.y) * w64(s.y) + w64(s.z) * w64(s.z);
        let want = Vec3::new(
            (w64(s.x) * w64(phib[f]) / ss) as Scalar,
            (w64(s.y) * w64(phib[f]) / ss) as Scalar,
            (w64(s.z) * w64(phib[f]) / ss) as Scalar,
        );
        let gap = cell_gap(b, want);
        fin("U_b vs S phi/S.S", gap);
        assert!(
            gap <= tol(1e-12) * max_u,
            "face {i}: |U_b - S phi/S.S| = {gap:e}"
        );
    }
    Ok(())
}

/// The worst componentwise distance between two cell values, in f64.
fn cell_gap(a: Vec3, b: Vec3) -> f64 {
    nmax(
        nmax((w64(a.x) - w64(b.x)).abs(), (w64(a.y) - w64(b.y)).abs()),
        (w64(a.z) - w64(b.z)).abs(),
    )
}

#[test]
fn pressure_inlets_need_the_opt_in() {
    let Some(gpu) = Gpu::new(0).ok() else {
        return;
    };
    let hm = blockgen::build_mesh(&box_spec(
        [2, 2, 2],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        ["patch", "patch", "patch", "patch", "patch", "patch"],
    ))
    .unwrap();
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    for name in [
        "pressureInletVelocity",
        "pressureInletUniformVelocity",
        "pressureDirectedInletVelocity",
        "pressureDirectedInletOutletVelocity",
        "pressureInletOutletParSlipVelocity",
    ] {
        let mut f = GpuVectorField::zeros(&gpu, &gm, "U").unwrap();
        let err = setup_vector_field(
            &gpu,
            &mut f,
            &raw_u(
                vec![("xMin", ps(name)), ("xMax", ps("zeroGradient"))],
                Vec3::ZERO,
            ),
            &hm,
        )
        .err()
        .unwrap_or_else(|| panic!("{name} built without the opt-in"));
        let msg = err.to_string();
        assert!(msg.contains(name), "{name}: {msg}");
        assert!(msg.contains("BcInputs::pressure_inlet"), "{name}: {msg}");
        assert!(msg.contains("SPEC-LIT 120.5"), "{name}: {msg}");
    }
}

#[test]
fn vector_inlets_on_a_scalar_field_are_refused() {
    let Some(gpu) = Gpu::new(0).ok() else {
        return;
    };
    let hm = blockgen::build_mesh(&box_spec(
        [2, 2, 2],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        ["patch", "patch", "patch", "patch", "patch", "patch"],
    ))
    .unwrap();
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    for name in [
        "surfaceNormalFixedValue",
        "cylindricalInletVelocity",
        "swirlFlowRateInletVelocity",
        "pressureInletVelocity",
        "pressureInletUniformVelocity",
        "pressureDirectedInletVelocity",
        "pressureDirectedInletOutletVelocity",
        "pressureInletOutletParSlipVelocity",
    ] {
        let mut f = GpuScalarField::zeros(&gpu, &gm, "psi").unwrap();
        let mut raw = RawScalarField {
            name: "psi".into(),
            ..Default::default()
        };
        raw.internal = vec![0.0];
        raw.boundary.insert("xMin".into(), ps(name));
        let err = setup_scalar_field(&gpu, &mut f, &raw, &hm)
            .err()
            .unwrap_or_else(|| panic!("{name} built on a scalar field"));
        let msg = err.to_string();
        assert!(
            msg.contains("is a vector condition"),
            "{name}: {msg}"
        );
    }
}

#[test]
fn bad_inlet_inputs_are_refused_by_name() {
    let Some(gpu) = Gpu::new(0).ok() else {
        return;
    };
    let hm = blockgen::build_mesh(&box_spec(
        [2, 2, 2],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        ["patch", "patch", "patch", "patch", "patch", "patch"],
    ))
    .unwrap();
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let refuse = |inlet: PatchFieldSpec| -> String {
        let mut f = GpuVectorField::zeros(&gpu, &gm, "U").unwrap();
        setup_vector_field_with(
            &gpu,
            &mut f,
            &raw_u(
                vec![("xMin", inlet), ("xMax", ps("zeroGradient"))],
                Vec3::ZERO,
            ),
            &hm,
            BcInputs {
                pressure_inlet: true,
                ..BcInputs::default()
            },
        )
        .err()
        .map(|e| e.to_string())
        .unwrap_or_else(|| "the case built".to_string())
    };

    // inletDirection (-1 0 0) on xMin points OUT of the domain.
    let mut d = ps("pressureDirectedInletVelocity");
    d.extra
        .insert("inletDirection".into(), "uniform (-1 0 0)".into());
    let msg = refuse(d);
    assert!(
        msg.contains("into the domain") && msg.contains("inletDirection"),
        "outward inletDirection: {msg}"
    );

    // A time-varying rpm is refused by name - no driver runs the timed path.
    let mut c = ps("cylindricalInletVelocity");
    c.extra.insert("origin".into(), "(0 0.5 0.5)".into());
    c.extra.insert("axis".into(), "(1 0 0)".into());
    c.extra.insert("axialVelocity".into(), "1".into());
    c.extra.insert("radialVelocity".into(), "0".into());
    c.extra
        .insert("rpm".into(), "table ((0 0) (1 60))".into());
    let msg = refuse(c);
    assert!(
        msg.contains("rpm") && msg.contains("SPEC-LIT 120.5"),
        "time-varying rpm: {msg}"
    );

    // A zero axis has no direction to normalise.
    let mut s = ps("swirlFlowRateInletVelocity");
    s.extra.insert("origin".into(), "(0 0.5 0.5)".into());
    s.extra.insert("axis".into(), "(0 0 0)".into());
    s.extra.insert("rpm".into(), "60".into());
    s.extra
        .insert("volumetricFlowRate".into(), "0.8".into());
    let msg = refuse(s);
    assert!(!msg.contains("the case built"), "zero axis built: {msg}");
}
