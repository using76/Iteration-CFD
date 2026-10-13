// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 120.7 - the buoyant pressure family through the public API, in
//! BOTH builds: the four closed forms ((120.7a)-(120.7e)) bit for bit
//! against their host twin, the refusals by name, and three columns at rest
//! - a kinematic one open at top and side through `Simple`, a stratified VOF
//! column open at the top, and a single-fluid column open at the side -
//! each against a control that DOES move. The tolerances follow docs/17 §5.1
//! class R, D-F32-1, which the `tol` const fn below encodes; everything is
//! compared in f64. A machine without a card passes vacuously, the
//! convention of `tests/bc_inlets.rs`.
//!
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::field_setup::{
    scalar_boundary_with, setup_scalar_field_with, setup_vector_field_with, vector_boundary_with,
    BcInputs, PrghInputs,
};
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::momentum::{BuoyancyCoeffs, MomentumControls};
use ofgpu::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use ofgpu::simple::{Simple, SimpleControls};
use ofgpu::vof::{Vof, VofControls, VofProperties};
use ofgpu::{BcKind, Gpu, GpuMesh, HostMesh, Result, Scalar, Vec3, fv};

/// `t` in the f64 build; under `single` (class R, D-F32-1)
/// `max(t, min(t*2^29, 1e-5))` - i.e. 1e-12 -> 1e-5, 1e-10 -> 1e-5.
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

/// A uniform block, every patch typed by the caller - the `spec` shape of
/// `tests/bc_outlets_ii.rs`, sized by the caller.
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

/// A patch entry carrying nothing but its type.
fn ps(ty: &str) -> PatchFieldSpec {
    PatchFieldSpec {
        type_name: ty.into(),
        ..Default::default()
    }
}

/// A patch entry of one of the four buoyant pressure names, with the entries
/// that name is defined by (SPEC-LIT 120.7).
fn prgh_bc(name: &str) -> PatchFieldSpec {
    let mut s = ps(name);
    match name {
        "prghPressure" => {
            s.extra.insert("p".into(), "uniform 1000".into());
        }
        "prghTotalPressure" => {
            s.extra.insert("p0".into(), "uniform 1000".into());
        }
        // (120.7d) reads no entry of its own: `ph_rgh,f` is the caller's.
        "prghTotalHydrostaticPressure" => {}
        _ => {
            for (k, v) in [
                ("rhoRef", "998"),
                ("pRefValue", "100000"),
                ("pRefPoint", "(0.5 0.25 1)"),
            ] {
                s.extra.insert(k.into(), v.into());
            }
        }
    }
    s
}

/// All six patches carrying the one condition, in blockgen's patch order.
fn all_six(spec: PatchFieldSpec) -> Vec<(&'static str, PatchFieldSpec)> {
    ["xMin", "xMax", "yMin", "yMax", "zMin", "zMax"]
        .into_iter()
        .map(|p| (p, spec.clone()))
        .collect()
}

/// `[bounds]` with patch `name`'s entry replaced by `with`.
fn with_patch(
    bounds: Vec<(&'static str, PatchFieldSpec)>,
    name: &str,
    with: PatchFieldSpec,
) -> Vec<(&'static str, PatchFieldSpec)> {
    bounds
        .into_iter()
        .map(|(p, s)| if p == name { (p, with.clone()) } else { (p, s) })
        .collect()
}

/// Tight enough that what the tests measure is not the linear solver's
/// stopping criterion.
fn tight() -> ofgpu::solver::SolverControls {
    ofgpu::solver::SolverControls {
        tolerance: 1e-14,
        rel_tol: 0.0,
        max_iter: 500,
        check_interval: 1,
        ..ofgpu::solver::SolverControls::default()
    }
}

/// Laminar: zero eddy viscosity with `calculated` boundary values.
fn laminar_nut(
    gpu: &Gpu,
    hm: &HostMesh,
    gm: &GpuMesh,
) -> Result<ofgpu::GpuScalarField> {
    let mut nut = ofgpu::GpuScalarField::zeros(gpu, gm, "nut")?;
    gpu.write(
        &mut nut.bc_kind,
        &vec![BcKind::Calculated as i32; hm.n_boundary_faces],
    )?;
    Ok(nut)
}

/// A uniform T at `t` on every cell AND every boundary face, zeroGradient -
/// with `t_ref = 300` the buoyancy vector is zero EXACTLY, which is what
/// makes the kinematic column a test of the pressure conditions alone. The
/// faces are evaluated, not left at zero: a zero face under `t_ref` reads as
/// a 300x density ratio through the `t_min` floor, and would swamp the test.
fn uniform_t(gpu: &Gpu, gm: &GpuMesh, t: Scalar) -> Result<ofgpu::GpuScalarField> {
    let mut f = ofgpu::GpuScalarField::zeros(gpu, gm, "T")?;
    gpu.write(&mut f.f, &vec![t; gm.n_cells])?;
    gpu.write(
        &mut f.bc_kind,
        &vec![BcKind::ZeroGradient as i32; gm.n_boundary_faces],
    )?;
    let k = ofgpu::field_ops::FieldKernels::new(gpu)?;
    ofgpu::field_ops::correct_boundary_conditions(gpu, &k, &mut f, gm)?;
    Ok(f)
}

/// Mesh B: 4 x 3 x 2 cells over [0,2] x [0,1] x [0,1], all six patches
/// `patch` - 52 boundary faces in all.
fn mesh_b(gpu: &Gpu) -> Result<(HostMesh, GpuMesh)> {
    let hm = blockgen::build_mesh(&box_spec(
        [4, 3, 2],
        [0.0, 0.0, 0.0],
        [2.0, 1.0, 1.0],
        ["patch", "patch", "patch", "patch", "patch", "patch"],
    ))?;
    let gm = GpuMesh::upload(gpu, &hm)?;
    Ok((hm, gm))
}

/// A raw scalar field of the given name, one uniform internal value, the
/// given patch entries.
fn raw_scalar(
    name: &str,
    boundary: Vec<(&str, PatchFieldSpec)>,
    internal: Scalar,
) -> RawScalarField {
    let mut raw = RawScalarField {
        name: name.into(),
        ..Default::default()
    };
    for (p, s) in boundary {
        raw.boundary.insert(p.into(), s);
    }
    raw.internal = vec![internal];
    raw
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

/// The face data SPEC-LIT 120.7's closed forms are checked against: one
/// entry per flattened boundary face, built so that every branch of every
/// equation is visited (the inflow test switches on `phi`, the density
/// ramps, the velocity cycles through five magnitudes).
fn prgh_face_data(nb: usize) -> (Vec3, Vec<Scalar>, Vec<Vec3>, Vec<Scalar>, Vec<Scalar>) {
    let g = Vec3::new(0.5, -1.0, -9.81);
    let rho_b: Vec<Scalar> = (0..nb).map(|f| (1.0 + 0.01 * f as f64) as Scalar).collect();
    let u_b: Vec<Vec3> = (0..nb)
        .map(|f| Vec3::new((0.1 * (f % 5) as f64) as Scalar, -0.2, 0.3))
        .collect();
    let phi_b: Vec<Scalar> = (0..nb)
        .map(|f| if f % 2 == 0 { -1.0 } else { 1.0 })
        .collect();
    let ph_rgh_b: Vec<Scalar> = (0..nb).map(|f| (2000.0 + f as f64) as Scalar).collect();
    (g, rho_b, u_b, phi_b, ph_rgh_b)
}

// ==========================================================================
//  (120.7a)-(120.7e) - the closed forms, bit for bit
// ==========================================================================

/// The host twin of SPEC-LIT 120.7's face value, in the operation order the
/// section writes the equations in: `gh` on the STORED centre (120.7a), the
/// dynamic term `(0.5 r) u2`, and (120.7e)'s world-coordinate difference.
/// The entries (`p`/`p0` 1000, `rhoRef` 998, `pRefValue` 100000, `pRefPoint`
/// (0.5 0.25 1)) are the ones `prgh_bc` writes - every one exact in f32, so
/// the widened entries the setup reads are these literals bit for bit.
fn want_prgh(
    name: &str,
    hm: &HostMesh,
    f: usize,
    g: Vec3,
    r: f64,
    u: Vec3,
    ph: f64,
    inflow: bool,
) -> f64 {
    let cf = hm.geom64();
    let x = cf.b_cf[f];
    let xw = hm.world64(x);
    let gh = g.x as f64 * x.x + g.y as f64 * x.y + g.z as f64 * x.z;
    let u2 = w64(u.x) * w64(u.x) + w64(u.y) * w64(u.y) + w64(u.z) * w64(u.z);
    match name {
        "prghPressure" => 1000.0 - r * gh,
        "prghTotalPressure" => {
            if inflow {
                (1000.0 - (0.5 * r) * u2) - r * gh
            } else {
                1000.0 - r * gh
            }
        }
        "prghTotalHydrostaticPressure" => {
            if inflow {
                ph - (0.5 * r) * u2
            } else {
                ph
            }
        }
        _ => {
            let dx = xw.x - 0.5;
            let dy = xw.y - 0.25;
            let dz = xw.z - 1.0;
            let gdr = g.x as f64 * dx + g.y as f64 * dy + g.z as f64 * dz;
            (100000.0 + 998.0 * gdr) - r * gh
        }
    }
}

#[test]
fn prgh_triples_are_the_closed_forms() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let (hm, _gm) = mesh_b(&gpu)?;
    let nb = hm.n_boundary_faces;
    let (g, rho_b, u_b, phi_b, ph_rgh_b) = prgh_face_data(nb);
    let internal = vec![0.0 as Scalar; hm.n_cells];
    for name in [
        "prghPressure",
        "prghTotalPressure",
        "prghTotalHydrostaticPressure",
        "uniformDensityHydrostaticPressure",
    ] {
        let kind = match name {
            "prghPressure" | "uniformDensityHydrostaticPressure" => BcKind::FixedValue,
            _ => BcKind::TotalPressure,
        };
        for (form, rho) in [("absolute", Some(&rho_b)), ("kinematic", None)] {
            let inputs = BcInputs {
                prgh: Some(PrghInputs {
                    g,
                    rho_b: rho.map(|v| v.as_slice()),
                    u_b: Some(&u_b),
                    ph_rgh_b: Some(&ph_rgh_b),
                }),
                phi_b: Some(&phi_b),
                ..BcInputs::default()
            };
            let b = scalar_boundary_with(
                &raw_scalar("p", all_six(prgh_bc(name)), 0.0),
                &hm,
                &internal,
                inputs,
            )?;
            let mut printed = 0usize;
            for f in 0..nb {
                let r = rho.map_or(1.0, |v| w64(v[f]));
                let inflow = w64(phi_b[f]) < 0.0;
                let want =
                    want_prgh(name, &hm, f, g, r, u_b[f], w64(ph_rgh_b[f]), inflow) as Scalar;
                assert_eq!(
                    w64(b.ref_value[f]),
                    w64(want),
                    "{form} {name} face {f}: ref_value"
                );
                assert_eq!(b.fr[f], 1.0, "{form} {name} face {f}: fr");
                assert_eq!(b.ref_grad[f], 0.0, "{form} {name} face {f}: ref_grad");
                assert_eq!(w64(b.value[f]), w64(want), "{form} {name} face {f}: value");
                assert_eq!(b.kind[f], kind as i32, "{form} {name} face {f}: kind");
                if printed < 2 {
                    eprintln!("  [120.7] {form} {name} face {f}: ref_value {:.6}", w64(want));
                    printed += 1;
                }
            }
            let biggest = (0..nb)
                .map(|f| fin("ref_value", (w64(b.ref_value[nb - 1 - f])).abs()))
                .fold(0.0f64, nmax);
            eprintln!("  [120.7] {form} {name}: max |ref_value| {biggest:.6} over {nb} faces");
        }
    }

    // (120.7c) with the flux withdrawn: the inflow test falls to the PRGH
    // inputs' face velocity, `PrghInputs::u_b[f] . n < 0` - on xMin (n = -x)
    // that is `u_b.x > 0`, which the cycling `u_b` turns on and off, and on
    // xMax never. The drivers hand the total forms their velocity HERE, so
    // `BcInputs::u_b` stays None, like the flux. Same formulas, bit for bit.
    let inputs = BcInputs {
        prgh: Some(PrghInputs {
            g,
            rho_b: Some(&rho_b),
            u_b: Some(&u_b),
            ph_rgh_b: Some(&ph_rgh_b),
        }),
        ..BcInputs::default()
    };
    let b = scalar_boundary_with(
        &raw_scalar("p", all_six(prgh_bc("prghTotalPressure")), 0.0),
        &hm,
        &internal,
        inputs,
    )?;
    for pname in ["xMin", "xMax"] {
        let p = &hm.patches[patch_of(&hm, pname)];
        let mut inflows = 0usize;
        for i in 0..p.size {
            let f = p.start + i;
            let inflow = if pname == "xMin" {
                w64(u_b[f].x) > 0.0
            } else {
                false
            };
            if inflow {
                inflows += 1;
            }
            let want = want_prgh(
                "prghTotalPressure",
                &hm,
                f,
                g,
                w64(rho_b[f]),
                u_b[f],
                w64(ph_rgh_b[f]),
                inflow,
            ) as Scalar;
            assert_eq!(
                w64(b.ref_value[f]),
                w64(want),
                "no-phi {pname} face {f}: ref_value"
            );
        }
        eprintln!("  [120.7] no-phi {pname}: {inflows} of {} faces inflow", p.size);
    }

    // (120.7d) under the same withdrawal: the inflow face carries
    // `ph - (1/2) rho_f |U|^2`, the outflow face bare `ph`, and the inflow
    // test reads the same `PrghInputs::u_b` - bit for bit.
    let b = scalar_boundary_with(
        &raw_scalar("p", all_six(prgh_bc("prghTotalHydrostaticPressure")), 0.0),
        &hm,
        &internal,
        inputs,
    )?;
    for pname in ["xMin", "xMax"] {
        let p = &hm.patches[patch_of(&hm, pname)];
        let mut inflows = 0usize;
        for i in 0..p.size {
            let f = p.start + i;
            let inflow = if pname == "xMin" { w64(u_b[f].x) > 0.0 } else { false };
            if inflow {
                inflows += 1;
            }
            let want = want_prgh(
                "prghTotalHydrostaticPressure",
                &hm,
                f,
                g,
                w64(rho_b[f]),
                u_b[f],
                w64(ph_rgh_b[f]),
                inflow,
            ) as Scalar;
            assert_eq!(
                w64(b.ref_value[f]),
                w64(want),
                "no-phi hydrostatic {pname} face {f}: ref_value"
            );
        }
        eprintln!(
            "  [120.7] no-phi hydrostatic {pname}: {inflows} of {} faces inflow",
            p.size
        );
    }

    Ok(())
}

// ==========================================================================
//  Refusals, by name
// ==========================================================================

/// `prgh_bc(name)` with one entry dropped.
fn without_entry(name: &str, key: &str) -> PatchFieldSpec {
    let mut s = prgh_bc(name);
    assert!(s.extra.remove(key).is_some(), "{name} has no {key}");
    s
}

/// `prgh_bc(name)` with one extra entry added.
fn with_entry(name: &str, key: &str, value: &str) -> PatchFieldSpec {
    let mut s = prgh_bc(name);
    s.extra.insert(key.into(), value.into());
    s
}

/// `prghPressure` of the static `p` the caller names - the columns' entries.
fn prgh_p(v: &str) -> PatchFieldSpec {
    let mut s = ps("prghPressure");
    s.extra.insert("p".into(), v.into());
    s
}

/// `uniformDensityHydrostaticPressure` with the entries the columns give it.
fn udh(rho_ref: &str, p_ref: &str, point: &str) -> PatchFieldSpec {
    let mut s = ps("uniformDensityHydrostaticPressure");
    for (k, v) in [("rhoRef", rho_ref), ("pRefValue", p_ref), ("pRefPoint", point)] {
        s.extra.insert(k.into(), v.into());
    }
    s
}

// ==========================================================================
//  A kinematic column, open at the top and the side, at rest
// ==========================================================================

/// SPEC-LIT 120.7's Check column on [`Simple`], in the kinematic form: a
/// 16 x 1 x 16 box, `nu = 0.1`, T = t_ref everywhere so the buoyancy vector
/// is zero EXACTLY, `p` on zMax `prghPressure p uniform 5` and on xMax
/// whatever `side` the caller names. Fifty outer iterations; returns
/// (max |U| over cells and faces, the cell p widened, the f64 cell z).
fn kinematic_column(gpu: &Gpu, g: Vec3, side: (&str, PatchFieldSpec)) -> Result<(f64, Vec<Scalar>, Vec<f64>)> {
    let hm = blockgen::build_mesh(&box_spec(
        [16, 1, 16],
        [0.0, 0.0, 0.0],
        [1.0, 0.0625, 1.0],
        ["wall", "patch", "empty", "empty", "wall", "patch"],
    ))?;
    let gm = GpuMesh::upload(gpu, &hm)?;
    let ctrl = SimpleControls {
        momentum: MomentumControls {
            nu: 0.1,
            u_solver: tight(),
            ..MomentumControls::default()
        },
        p_solver: tight(),
        ..SimpleControls::default()
    };
    let mut sim = Simple::new(
        gpu,
        &hm,
        &gm,
        ctrl,
        BuoyancyCoeffs {
            g,
            t_ref: 300.0,
            ..BuoyancyCoeffs::default()
        },
    )?;
    setup_vector_field_with(
        gpu,
        sim.u_mut(),
        &raw_u(
            vec![
                ("xMin", ps("noSlip")),
                ("xMax", ps("zeroGradient")),
                ("yMin", ps("empty")),
                ("yMax", ps("empty")),
                ("zMin", ps("noSlip")),
                ("zMax", ps("zeroGradient")),
            ],
            Vec3::ZERO,
        ),
        &hm,
        BcInputs::default(),
    )?;
    let t = uniform_t(gpu, &gm, 300.0)?;
    let bounds = vec![
        ("xMin", ps("zeroGradient")),
        side,
        ("yMin", ps("empty")),
        ("yMax", ps("empty")),
        ("zMin", ps("zeroGradient")),
        ("zMax", prgh_p("uniform 5")),
    ];
    // The interior starts on the boundary's own constant: the two open
    // patches both write the uniform 14.81, so an interior of 14.81 is the
    // rest state and nothing kicks it. From an interior of 0 the first
    // momentum predictor sees a 14.81 boundary jump and the transient needs
    // hundreds of iterations to die back (u_relax 0.7, p_relax 0.3) - the
    // gate measures the pair's consistency, not that recovery.
    setup_scalar_field_with(
        gpu,
        sim.p_mut(),
        &raw_scalar("p", bounds, 14.81),
        &hm,
        BcInputs {
            prgh: Some(PrghInputs {
                g,
                rho_b: None,
                u_b: None,
                ph_rgh_b: None,
            }),
            ..BcInputs::default()
        },
    )?;
    sim.initialise(gpu)?;
    let nut = laminar_nut(gpu, &hm, &gm)?;
    let mut backend = PbicgstabBackend::new(ctrl.p_solver);
    backend.setup(gpu, &hm, &gm, &SystemProbe::default())?;
    for _ in 0..50 {
        sim.correct(gpu, &mut backend, &nut, &t)?;
    }
    let uf = gpu.download(&sim.u().f)?;
    let ub = gpu.download(&sim.u().bf)?;
    let mut maxu = 0.0f64;
    for v in uf.iter().chain(ub.iter()) {
        let speed = (w64(v.x) * w64(v.x) + w64(v.y) * w64(v.y) + w64(v.z) * w64(v.z)).sqrt();
        maxu = nmax(maxu, fin("|U|", speed));
    }
    let p = gpu.download(&sim.p().f)?;
    let zs = hm.geom64().c.iter().map(|c| c.z).collect();
    Ok((maxu, p, zs))
}

/// The hydrostatic static pressure of the kinematic column against the
/// solved one, measured: `p_s = p + g . x` cell by cell, `p_exact = 5 +
/// 9.81 (1 - z)`.
fn column_p_error(p: &[Scalar], zs: &[f64]) -> (f64, f64) {
    let mut worst = 0.0f64;
    let mut biggest = 0.0f64;
    for (pc, z) in p.iter().zip(zs) {
        let ps = w64(*pc) + -9.81 * z;
        let want = 5.0 + 9.81 * (1.0 - z);
        biggest = nmax(biggest, fin("|p_exact|", want.abs()));
        worst = nmax(worst, fin("|p - p_exact|", (ps - want).abs()));
    }
    (worst, biggest)
}

#[test]
fn a_kinematic_column_open_at_the_top_and_side_stays_at_rest() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let g = Vec3::new(0.0, 0.0, -9.81);

    // (120.7e) on the side, (120.7b) on the top: both write a UNIFORM
    // p_rgh of 14.81, the fluid stays at rest, and the solved p is the
    // hydrostatic column in kinematic units.
    let (maxu, p, zs) = kinematic_column(&gpu, g, ("xMax", udh("1", "5", "(0 0 1)")))?;
    eprintln!("  [120.7] kinematic column at rest: max |U| {maxu:e}");
    assert!(
        maxu <= tol(1e-10),
        "the column did not stay at rest: max |U| = {maxu:e}"
    );
    let (worst, biggest) = column_p_error(&p, &zs);
    eprintln!(
        "  [120.7] kinematic column: max |p - p_exact| {worst:e} of max |p_exact| {biggest:.6}"
    );
    assert!(
        worst <= tol(1e-12) * biggest,
        "the column is not hydrostatic: max |p - p_exact| = {worst:e}"
    );

    // The control: a uniform STATIC p of 5 through (120.7b) on the vertical
    // patch is not the hydrostatic distribution, and drives a flow.
    let (maxu, _p, _zs) = kinematic_column(&gpu, g, ("xMax", prgh_p("uniform 5")))?;
    eprintln!("  [120.7] kinematic column control: max |U| {maxu:e}");
    fin("control max |U|", maxu);
    assert!(
        maxu >= 1e-3,
        "the control stayed at rest: max |U| = {maxu:e}"
    );

    Ok(())
}

// ==========================================================================
//  VOF columns at rest - the gate
// ==========================================================================

/// The VOF column of the last two tests: `alpha` 1 below the mid-plane (or 1
/// everywhere for `alpha_one`), zeroGradient; U noSlip on walls,
/// zeroGradient on every open patch; `p_rgh` carrying the caller's entries,
/// built through `PrghInputs` with `rho_b` from [`Vof::boundary_density`].
/// Twenty steps of 2.5e-4, in BOTH builds - a quarter of run 1's 1e-3,
/// because under `single` the buoyancy split's f32 round-off floor at the
/// 1000:1 jump scales with `rAU`, so with the time step, and the quarter
/// step is what holds class R (BC-11 run 2). Returns the mesh (for the
/// comparisons), max |U| over cells and faces, the cell p_rgh, the cell
/// rho, and the boundary rho the setup read.
fn rest_column(
    gpu: &Gpu,
    kinds: [&str; 6],
    p_b: Vec<(&str, PatchFieldSpec)>,
    alpha_one: bool,
) -> Result<(HostMesh, f64, Vec<Scalar>, Vec<Scalar>, Vec<Scalar>)> {
    let g = Vec3::new(0.0, 0.0, -9.81);
    let h = 1.0 / 24.0;
    let hm = blockgen::build_mesh(&box_spec(
        [24, 1, 24],
        [0.0, 0.0, 0.0],
        [1.0, h, 1.0],
        kinds,
    ))?;
    let m = GpuMesh::upload(gpu, &hm)?;
    let props = VofProperties {
        rho1: 1000.0,
        rho2: 1.0,
        mu1: 1.002e-3,
        mu2: 1.8e-5,
        sigma: 0.0,
        g,
        c_alpha: 1.0,
    };
    let ctrl = VofControls {
        delta_t: 2.5e-4,
        n_correctors: 3,
        u_solver: tight(),
        p_solver: tight(),
        sn_grad: fv::SnGradScheme::Uncorrected,
        ..VofControls::default()
    };
    let mut vof = Vof::new(gpu, &hm, &m, props, ctrl)?;

    // The interface sits exactly on a face, so the initial alpha is a clean
    // 1/0 split with no partly-filled cell anywhere.
    let a: Vec<Scalar> = hm
        .c
        .iter()
        .map(|c| if alpha_one || c.z < 0.5 { 1.0 } else { 0.0 })
        .collect();
    let mut raw_alpha = raw_scalar(
        "alpha",
        all_six(ps("zeroGradient")),
        0.0,
    );
    raw_alpha.internal = a;
    setup_scalar_field_with(gpu, vof.alpha_mut(), &raw_alpha, &hm, BcInputs::default())?;

    let names = ["xMin", "xMax", "yMin", "yMax", "zMin", "zMax"];
    let u_entries: Vec<(&str, PatchFieldSpec)> = names
        .iter()
        .zip(kinds)
        .map(|(p, t)| {
            let spec = if t == "wall" {
                ps("noSlip")
            } else if t == "empty" {
                ps("empty")
            } else {
                ps("zeroGradient")
            };
            (*p, spec)
        })
        .collect();
    setup_vector_field_with(gpu, vof.u_mut(), &raw_u(u_entries, Vec3::ZERO), &hm, BcInputs::default())?;

    let rho_b = vof.boundary_density(gpu)?;
    // The interior starts on the hydrostatic p_rgh itself - the piecewise
    // constant the static column sits in (5909.905 heavy / 1009.81 light,
    // 10810 single-fluid). From an interior of 0 the first pressure solve
    // lifts the whole column by ~5000 and its rounding floor parks the
    // heavy layer's constant mode a few 1e-8 off for the whole run; the
    // gate measures the condition pair's consistency, not that recovery.
    // Each layer constant is evaluated ONCE in f64 and rounded once, so
    // under `single` the stored layer really is one value and carries no
    // per-cell rounding pattern for the momentum to ring against.
    let heavy = (1000.0 + 9.81 * 0.5 + 1000.0 * 9.81 * 0.5) as Scalar;
    let light = (1000.0 + 9.81 * (1.0 - 0.5) + 1.0 * 9.81 * 0.5) as Scalar;
    let one = (1000.0 + 1000.0 * 9.81 * (1.0 - 0.5) + 1000.0 * 9.81 * 0.5) as Scalar;
    let pr: Vec<Scalar> = hm
        .c
        .iter()
        .map(|c| {
            let z = c.z as f64;
            if alpha_one {
                one
            } else if z < 0.5 {
                heavy
            } else {
                light
            }
        })
        .collect();
    let mut raw_p = raw_scalar("p_rgh", p_b, 0.0);
    raw_p.internal = pr;
    setup_scalar_field_with(
        gpu,
        vof.p_rgh_mut(),
        &raw_p,
        &hm,
        BcInputs {
            prgh: Some(PrghInputs {
                g,
                rho_b: Some(&rho_b),
                u_b: None,
                ph_rgh_b: None,
            }),
            ..BcInputs::default()
        },
    )?;
    vof.initialise(gpu)?;
    assert!(
        !vof.pressure_is_pinned(),
        "the column carries a Dirichlet p_rgh and must not be pinned"
    );
    for _ in 0..20 {
        vof.step(gpu, 2.5e-4)?;
    }

    let uf = gpu.download(&vof.u().f)?;
    let ub = gpu.download(&vof.u().bf)?;
    let mut maxu = 0.0f64;
    for v in uf.iter().chain(ub.iter()) {
        let speed = (w64(v.x) * w64(v.x) + w64(v.y) * w64(v.y) + w64(v.z) * w64(v.z)).sqrt();
        maxu = nmax(maxu, fin("|U|", speed));
    }
    let p_rgh = gpu.download(&vof.p_rgh().f)?;
    let rho = gpu.download(&vof.rho().f)?;
    Ok((hm, maxu, p_rgh, rho, rho_b))
}

/// The p_rgh patches of a column whose only Dirichlet is `top`, on the patch
/// the caller names; everything else zeroGradient or empty.
fn column_p_b(top: (&str, PatchFieldSpec)) -> Vec<(&'static str, PatchFieldSpec)> {
    let mut v = all_six(ps("zeroGradient"));
    v[2] = ("yMin", ps("empty"));
    v[3] = ("yMax", ps("empty"));
    v = with_patch(v, top.0, top.1);
    v
}

/// A plain `fixedValue uniform v` - the control rows' p_rgh level.
fn fixed_value(v: Scalar) -> PatchFieldSpec {
    let mut s = ps("fixedValue");
    s.value = vec![v];
    s
}

/// The stratified column's exact static pressure: `p = p_rgh + rho g.x`
/// against the piecewise hydrostatic column, cell by cell. Returns
/// (max |p - p_exact|, max |p_exact|).
fn stratified_p_error(p_rgh: &[Scalar], rho: &[Scalar], hm: &HostMesh) -> (f64, f64) {
    let c = hm.geom64().c;
    let mut worst = 0.0f64;
    let mut biggest = 0.0f64;
    for i in 0..hm.n_cells {
        let z = c[i].z;
        let gx = -9.81 * z;
        let pc = w64(p_rgh[i]) + w64(rho[i]) * gx;
        let want = if z > 0.5 {
            1000.0 + 9.81 * (1.0 - z)
        } else {
            1000.0 + 9.81 * 0.5 + 1000.0 * 9.81 * (0.5 - z)
        };
        biggest = nmax(biggest, fin("|p_exact|", want.abs()));
        worst = nmax(worst, fin("|p - p_exact|", (pc - want).abs()));
    }
    (worst, biggest)
}

#[test]
fn a_stratified_column_open_at_the_top_is_hydrostatic() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let kinds = ["wall", "wall", "empty", "empty", "wall", "patch"];

    // (120.7b) at the top of the LIGHT phase: rho_f = 1 there, so the face
    // reads p_rgh = 1000 + 9.81, the interior holds it uniform, and the
    // static pressure comes out the exact two-layer hydrostatic column.
    let (hm, maxu, p_rgh, rho, rho_b) =
        rest_column(&gpu, kinds, column_p_b(("zMax", prgh_p("uniform 1000"))), false)?;

    let zmax = &hm.patches[patch_of(&hm, "zMax")];
    for i in 0..zmax.size {
        assert_eq!(w64(rho_b[zmax.start + i]), 1.0, "zMax face {i}: rho_f");
    }
    let zmin = &hm.patches[patch_of(&hm, "zMin")];
    for i in 0..zmin.size {
        assert_eq!(w64(rho_b[zmin.start + i]), 1000.0, "zMin face {i}: rho_f");
    }

    eprintln!("  [120.7] stratified column at rest: max |U| {maxu:e}");
    assert!(
        maxu <= tol(1e-10),
        "the stratified column did not stay at rest: max |U| = {maxu:e}"
    );
    let (worst, biggest) = stratified_p_error(&p_rgh, &rho, &hm);
    eprintln!(
        "  [120.7] stratified column: max |p - p_exact| {worst:e} of max |p_exact| {biggest:.6}"
    );
    assert!(
        worst <= tol(1e-12) * biggest,
        "the stratified column is not hydrostatic: {worst:e} of {biggest:.6}"
    );

    // The control: `fixedValue` of the p_rgh LEVEL 1000 at the top instead
    // of the static pressure 1000 - the level is one `rho g h` short, about
    // 9.81 here, and the column reads it.
    let (hm, _maxu, p_rgh, rho, _rho_b) =
        rest_column(&gpu, kinds, column_p_b(("zMax", fixed_value(1000.0))), false)?;
    let (worst, _biggest) = stratified_p_error(&p_rgh, &rho, &hm);
    eprintln!("  [120.7] stratified column control: max |p - p_exact| {worst:e}");
    assert!(
        worst >= 1.0,
        "the control matched the hydrostatic column after all: {worst:e}"
    );

    Ok(())
}

#[test]
fn a_column_open_at_the_side_stays_at_rest() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let kinds = ["wall", "patch", "empty", "empty", "wall", "wall"];

    // (120.7e) on the VERTICAL patch: with rhoRef = rho it is uniform in
    // p_rgh - `p_ref + rhoRef g.(x_f - x_ref)` and `- rho_f g.x_f` cancel
    // face by face - and the single-fluid column stays at rest.
    let (hm, maxu, p_rgh, rho, _rho_b) =
        rest_column(&gpu, kinds, column_p_b(("xMax", udh("1000", "1000", "(0 0 1)"))), true)?;
    eprintln!("  [120.7] side-open column at rest: max |U| {maxu:e}");
    assert!(
        maxu <= tol(1e-10),
        "the side-open column did not stay at rest: max |U| = {maxu:e}"
    );

    // The static pressure against the hydrostatic column 1000 + rho g (1 - z).
    let c = hm.geom64().c;
    let mut worst = 0.0f64;
    let mut biggest = 0.0f64;
    for i in 0..hm.n_cells {
        let z = c[i].z;
        let gx = -9.81 * z;
        let pc = w64(p_rgh[i]) + w64(rho[i]) * gx;
        let want = 1000.0 + 1000.0 * 9.81 * (1.0 - z);
        biggest = nmax(biggest, fin("|p_exact|", want.abs()));
        worst = nmax(worst, fin("|p - p_exact|", (pc - want).abs()));
    }
    eprintln!(
        "  [120.7] side-open column: max |p - p_exact| {worst:e} of max |p_exact| {biggest:.6}"
    );
    assert!(
        worst <= tol(1e-12) * biggest,
        "the side-open column is not hydrostatic: {worst:e} of {biggest:.6}"
    );

    // The control: a uniform STATIC p of 1000 through (120.7b) on the
    // vertical patch is not the hydrostatic distribution, and drives a flow.
    let (_hm, maxu, _p, _rho, _rb) =
        rest_column(&gpu, kinds, column_p_b(("xMax", prgh_p("uniform 1000"))), true)?;
    eprintln!("  [120.7] side-open column control: max |U| {maxu:e}");
    fin("control max |U|", maxu);
    assert!(
        maxu >= 1e-3,
        "the control stayed at rest: max |U| = {maxu:e}"
    );

    Ok(())
}

#[test]
fn prgh_refusals_are_by_name() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let (hm, _gm) = mesh_b(&gpu)?;
    let nb = hm.n_boundary_faces;
    let internal = vec![0.0 as Scalar; hm.n_cells];
    let (g, rho_b, u_b, phi_b, ph_rgh_b) = prgh_face_data(nb);

    // The full input set, and the two partial ones the refusals are about.
    let full = BcInputs {
        prgh: Some(PrghInputs {
            g,
            rho_b: Some(&rho_b),
            u_b: Some(&u_b),
            ph_rgh_b: Some(&ph_rgh_b),
        }),
        phi_b: Some(&phi_b),
        ..BcInputs::default()
    };
    let no_u = BcInputs {
        prgh: Some(PrghInputs {
            g,
            rho_b: Some(&rho_b),
            u_b: None,
            ph_rgh_b: Some(&ph_rgh_b),
        }),
        phi_b: Some(&phi_b),
        ..BcInputs::default()
    };
    let no_ph = BcInputs {
        prgh: Some(PrghInputs {
            g,
            rho_b: Some(&rho_b),
            u_b: Some(&u_b),
            ph_rgh_b: None,
        }),
        phi_b: Some(&phi_b),
        ..BcInputs::default()
    };

    let check = |what: &str, spec: PatchFieldSpec, inputs: &BcInputs, want: &[&str]| {
        let e = scalar_boundary_with(&raw_scalar("p", all_six(spec), 0.0), &hm, &internal, *inputs)
            .expect_err(what)
            .to_string();
        println!("  [120.7] refused: {e}");
        for w in want {
            assert!(e.contains(w), "{w:?} not in: {e}");
        }
    };

    // prgh None: every one of the four, naming the condition and the input.
    for name in [
        "prghPressure",
        "prghTotalPressure",
        "prghTotalHydrostaticPressure",
        "uniformDensityHydrostaticPressure",
    ] {
        check(
            name,
            prgh_bc(name),
            &BcInputs::default(),
            &["SPEC-LIT 120.7", name, "BcInputs::prgh"],
        );
    }

    // The two total forms, without the fields their dynamic term reads.
    check(
        "prghTotalPressure without U",
        prgh_bc("prghTotalPressure"),
        &no_u,
        &["SPEC-LIT 120.7", "prghTotalPressure", "U"],
    );
    check(
        "prghTotalHydrostaticPressure without ph_rgh",
        prgh_bc("prghTotalHydrostaticPressure"),
        &no_ph,
        &["SPEC-LIT 120.7", "prghTotalHydrostaticPressure", "ph_rgh"],
    );

    // A missing entry, named by key.
    check(
        "prghPressure without p",
        without_entry("prghPressure", "p"),
        &full,
        &["SPEC-LIT 120.7", "prghPressure", "p"],
    );
    check(
        "prghTotalPressure without p0",
        without_entry("prghTotalPressure", "p0"),
        &full,
        &["SPEC-LIT 120.7", "prghTotalPressure", "p0"],
    );
    for key in ["rhoRef", "pRefValue", "pRefPoint"] {
        check(
            key,
            without_entry("uniformDensityHydrostaticPressure", key),
            &full,
            &["SPEC-LIT 120.7", "uniformDensityHydrostaticPressure", key],
        );
    }

    // The density and the hydrostatic field are the solver's own.
    check(
        "a rho entry naming another field",
        with_entry("prghPressure", "rho", "rho1"),
        &full,
        &["SPEC-LIT 120.7", "prghPressure", "rho1"],
    );
    check(
        "a ph_rgh entry naming another field",
        with_entry("prghTotalHydrostaticPressure", "ph_rgh", "ph_rgh0"),
        &full,
        &["SPEC-LIT 120.7", "prghTotalHydrostaticPressure", "ph_rgh0"],
    );

    // ... and the two legal spellings are accepted.
    for (name, key) in [("prghPressure", "rho"), ("prghTotalHydrostaticPressure", "ph_rgh")] {
        let b = scalar_boundary_with(
            &raw_scalar("p", all_six(with_entry(name, key, key)), 0.0),
            &hm,
            &internal,
            full,
        )?;
        fin("ref_value[0]", w64(b.ref_value[0]));
    }

    // The four names on a VECTOR field: refused, though two of them map to
    // `FixedValue` and would ride the static arm unnoticed.
    let uinternal = vec![Vec3::ZERO; hm.n_cells];
    for name in [
        "prghPressure",
        "prghTotalPressure",
        "prghTotalHydrostaticPressure",
        "uniformDensityHydrostaticPressure",
    ] {
        let e = vector_boundary_with(
            &raw_u(all_six(prgh_bc(name)), Vec3::ZERO),
            &hm,
            &uinternal,
            full,
        )
        .expect_err("a vector field cannot carry the family")
        .to_string();
        println!("  [120.7] vector: {e}");
        assert!(e.contains("SPEC-LIT 120.7"), "SPEC-LIT 120.7 not in: {e}");
    }

    Ok(())
}
