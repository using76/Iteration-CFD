// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 120.14 - the VOF family through the public API, in BOTH builds:
//! the four rewrites ((120.14a)-(120.14d)) bit for bit against an oracle
//! written here in the same operation order, with the two identities they
//! are built for (the inlet's liquid flux is the given rate, the outlet's
//! phase-weighted mean is the given velocity); the refusals by name; a
//! channel fed at a variable height through `Vof`, whose transported liquid
//! flux is the rate at every step with `alpha` in `[0, 1]`; and a tank open
//! at a side through `phaseHydrostaticPressure` that stays at rest, against
//! a control that moves. Tolerances follow docs/17 §5.1 class R, D-F32-1
//! (the `tol` const fn); everything is compared in f64. The two `Vof` tests
//! pass vacuously on a machine without a card, the convention of
//! `tests/bc_inlets.rs`.
//!
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::field::vof_family::{self, VofFamilyKind, VofFamilyRule};
use ofgpu::field_setup::{
    scalar_boundary_with, setup_scalar_field_with, setup_vector_field_with, vector_boundary_with,
    BcInputs, PrghInputs,
};
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
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

/// A NaN-propagating max.
fn nmax(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// A uniform block, every patch typed by the caller.
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

fn patch_of(hm: &HostMesh, name: &str) -> (usize, usize) {
    let p = hm
        .patches
        .iter()
        .find(|p| p.name == name)
        .unwrap_or_else(|| panic!("no patch {name}"));
    (p.start, p.size)
}

fn ps(ty: &str) -> PatchFieldSpec {
    PatchFieldSpec {
        type_name: ty.into(),
        ..Default::default()
    }
}

fn with(mut s: PatchFieldSpec, entries: &[(&str, &str)]) -> PatchFieldSpec {
    for (k, v) in entries {
        s.extra.insert((*k).into(), (*v).into());
    }
    s
}

const NAMES: [&str; 6] = ["xMin", "xMax", "yMin", "yMax", "zMin", "zMax"];

fn raw_scalar(name: &str, b: Vec<(&str, PatchFieldSpec)>, internal: Vec<Scalar>) -> RawScalarField {
    let mut raw = RawScalarField {
        name: name.into(),
        ..Default::default()
    };
    for (p, s) in b {
        raw.boundary.insert(p.into(), s);
    }
    raw.internal = internal;
    raw
}

fn raw_u(b: Vec<(&str, PatchFieldSpec)>, internal: Vec<Vec3>) -> RawVectorField {
    let mut raw = RawVectorField {
        name: "U".into(),
        ..Default::default()
    };
    for (p, s) in b {
        raw.boundary.insert(p.into(), s);
    }
    raw.internal = internal;
    raw
}

/// The six patches, `patch` carrying `spec` and the rest `rest`.
fn one_patch(patch: &str, spec: PatchFieldSpec, rest: &str) -> Vec<(&'static str, PatchFieldSpec)> {
    NAMES
        .into_iter()
        .map(|p| (p, if p == patch { spec.clone() } else { ps(rest) }))
        .collect()
}

const ALPHA: &str = "alpha.water";

fn opt_in() -> BcInputs<'static> {
    BcInputs {
        vof_family: Some(ALPHA),
        ..BcInputs::default()
    }
}

/// Host mesh A: 4 x 3 x 2 cells over [0,2] x [0,1] x [0,1], all six patches
/// `patch` - 52 boundary faces.
fn mesh_a() -> Result<HostMesh> {
    blockgen::build_mesh(&box_spec(
        [4, 3, 2],
        [0.0, 0.0, 0.0],
        [2.0, 1.0, 1.0],
        ["patch"; 6],
    ))
}

/// The rule's face data is the mesh's: cells, f64 normals, areas and both
/// centres.
fn check_rule_geometry(rule: &VofFamilyRule, hm: &HostMesh, patch: &str) {
    let (start, size) = patch_of(hm, patch);
    assert_eq!((rule.start, rule.size, rule.patch.as_str()), (start, size, patch));
    let g = hm.geom64();
    for i in 0..size {
        let f = start + i;
        assert_eq!(rule.cells[i], hm.b_face_cells[f] as usize, "{patch} face {i}: cell");
        let s = g.b_sf[f];
        let a = (s.x * s.x + s.y * s.y + s.z * s.z).sqrt();
        let n = rule.normal[i];
        assert_eq!((n.x, n.y, n.z), (s.x / a, s.y / a, s.z / a), "{patch} face {i}: normal");
        assert_eq!(rule.mag_sf[i], g.b_mag_sf[f], "{patch} face {i}: |S_f|");
        assert_eq!(rule.x_local[i], g.b_cf[f], "{patch} face {i}: stored centre");
        assert_eq!(rule.x_world[i], hm.world64(g.b_cf[f]), "{patch} face {i}: world centre");
    }
}

/// `fr*r + (1 - fr)*(psi_P + 0)` - §4's evaluation with `refGrad = 0`.
fn mixed(fr: Scalar, r: Scalar, p: Scalar) -> Scalar {
    fr * r + (1.0 - fr) * p
}

#[test]
fn vof_family_triples_are_the_closed_forms() -> Result<()> {
    let hm = mesh_a()?;
    let nb = hm.n_boundary_faces;
    let nc = hm.n_cells;

    // ---- (120.14a) on alpha.water: a cell field that puts the xMin faces
    // below, inside and above [0.2, 0.7].
    let cells: Vec<Scalar> = (0..nc).map(|c| [0.1, 0.5, 0.9, 0.2, 0.7][c % 5] as Scalar).collect();
    let vh = with(ps("variableHeightFlowRate"), &[("lowerBound", "0.2"), ("upperBound", "0.7")]);
    let raw = raw_scalar(ALPHA, one_patch("xMin", vh, "zeroGradient"), cells.clone());
    let b = scalar_boundary_with(&raw, &hm, &cells, opt_in())?;
    assert_eq!(b.vof_rules.len(), 1, "one rule on xMin");
    let rule = &b.vof_rules[0];
    check_rule_geometry(rule, &hm, "xMin");
    let (lo, hi) = (0.2 as Scalar, 0.7 as Scalar);
    assert_eq!(rule.kind, VofFamilyKind::VariableHeightFlowRate { lower: lo, upper: hi });
    let (mut fr, mut rv, mut rg) = (b.fr.clone(), b.ref_value.clone(), b.ref_grad.clone());
    for i in 0..rule.size {
        let f = rule.start + i;
        assert_eq!(b.kind[f], BcKind::Mixed as i32, "xMin face {i}: the seed kind");
        assert_eq!(b.fr[f], 0.0, "xMin face {i}: the seed is zero-gradient");
    }
    vof_family::variable_height_alpha(rule, &cells, &mut fr, &mut rv, &mut rg);
    let mut branches = [0usize; 3];
    for i in 0..rule.size {
        let f = rule.start + i;
        let a = cells[rule.cells[i]];
        let (want_fr, want_r, k) = if a > hi {
            (1.0, hi, 2)
        } else if a < lo {
            (1.0, lo, 0)
        } else {
            (0.0, 0.0, 1)
        };
        branches[k] += 1;
        assert_eq!((fr[f], rv[f], rg[f]), (want_fr, want_r, 0.0), "(120.14a) xMin face {i}");
        assert_eq!(mixed(fr[f], rv[f], a), a.clamp(lo, hi), "(120.14a) xMin face {i}: the clamp");
    }
    assert!(branches.iter().all(|&k| k > 0), "every branch visited: {branches:?}");
    for f in (0..nb).filter(|f| !(rule.start..rule.start + rule.size).contains(f)) {
        assert_eq!((fr[f], rv[f]), (b.fr[f], b.ref_value[f]), "face {f} off the rule is untouched");
    }

    // ---- (120.14b) and (120.14c) on U: an inlet on xMin, an outlet on xMax.
    let q = 0.3f64;
    let u_mean = 0.25f64;
    let u_cells: Vec<Vec3> = (0..nc)
        .map(|c| {
            let x = c as Scalar;
            Vec3::new(0.1 + 0.01 * x, -0.02 * x, 0.03 - 0.005 * x)
        })
        .collect();
    let mut bu = one_patch(
        "xMin",
        with(ps("variableHeightFlowRateInletVelocity"), &[("flowRate", "constant 0.3"), ("alpha", ALPHA)]),
        "zeroGradient",
    );
    bu[1].1 = with(ps("outletPhaseMeanVelocity"), &[("Umean", "0.25"), ("alpha", ALPHA)]);
    let raw = raw_u(bu, u_cells.clone());
    let b = vector_boundary_with(&raw, &hm, &u_cells, opt_in())?;
    assert_eq!(b.vof_rules.len(), 2, "two rules: xMin and xMax");
    let (inlet, outlet) = (&b.vof_rules[0], &b.vof_rules[1]);
    check_rule_geometry(inlet, &hm, "xMin");
    check_rule_geometry(outlet, &hm, "xMax");
    assert_eq!(inlet.kind, VofFamilyKind::VariableHeightFlowRateInletVelocity { flow_rate: q });
    assert_eq!(outlet.kind, VofFamilyKind::OutletPhaseMeanVelocity { u_mean });
    // A face alpha with every clamp branch: above 1, inside, 0 and below 0.
    let alpha_b: Vec<Scalar> = (0..nb).map(|f| [1.0, 0.5, 0.0, 1.25, -0.125, 0.75][f % 6] as Scalar).collect();
    let a_of = |f: usize| (w64(alpha_b[f])).clamp(0.0, 1.0);
    let (mut fr, mut rv, mut rg) = (b.fr.clone(), b.ref_value.clone(), b.ref_grad.clone());
    vof_family::variable_height_inlet_velocity(inlet, &alpha_b, &mut fr, &mut rv, &mut rg);
    vof_family::outlet_phase_mean_velocity(outlet, &alpha_b, &u_cells, &mut fr, &mut rv, &mut rg);

    // The inlet's oracle, in (120.14b)'s order.
    let mut d = 0.0f64;
    for i in 0..inlet.size {
        let a = a_of(inlet.start + i);
        d += (a * a) * inlet.mag_sf[i];
    }
    let mut flux = 0.0f64;
    for i in 0..inlet.size {
        let f = inlet.start + i;
        let s = (a_of(f) * q) / d;
        let n = inlet.normal[i];
        let want = Vec3::new((-(n.x * s)) as Scalar, (-(n.y * s)) as Scalar, (-(n.z * s)) as Scalar);
        assert_eq!((fr[f], rv[f], rg[f]), (1.0, want, Vec3::ZERO), "(120.14b) xMin face {i}");
        let sf = hm.b_sf[f];
        flux += a_of(f) * (w64(rv[f].x) * w64(sf.x) + w64(rv[f].y) * w64(sf.y) + w64(rv[f].z) * w64(sf.z));
    }
    let err_in = (-flux - q).abs() / q;
    eprintln!("  [120.14] inlet: -sum a_f (U_f . S_f) = {:.17e} for Q = {q} (rel {err_in:e})", -flux);
    assert!(err_in <= tol(1e-12), "(120.14b) the liquid flux is not Q: rel {err_in:e}");

    // The outlet's oracle, in (120.14c)'s order.
    let un = |i: usize| {
        let p = u_cells[outlet.cells[i]];
        let n = outlet.normal[i];
        w64(p.x) * n.x + w64(p.y) * n.y + w64(p.z) * n.z
    };
    let (mut area, mut mean) = (0.0f64, 0.0f64);
    for i in 0..outlet.size {
        let a = a_of(outlet.start + i);
        area += a * outlet.mag_sf[i];
        mean += (a * un(i)) * outlet.mag_sf[i];
    }
    let c = u_mean - mean / area;
    let (mut got_a, mut got_m) = (0.0f64, 0.0f64);
    for i in 0..outlet.size {
        let f = outlet.start + i;
        let p = u_cells[outlet.cells[i]];
        let n = outlet.normal[i];
        let want = Vec3::new(
            (w64(p.x) + c * n.x) as Scalar,
            (w64(p.y) + c * n.y) as Scalar,
            (w64(p.z) + c * n.z) as Scalar,
        );
        assert_eq!((fr[f], rv[f], rg[f]), (1.0, want, Vec3::ZERO), "(120.14c) xMax face {i}");
        let a = a_of(f);
        got_a += a * outlet.mag_sf[i];
        got_m += (a * (w64(rv[f].x) * n.x + w64(rv[f].y) * n.y + w64(rv[f].z) * n.z)) * outlet.mag_sf[i];
    }
    let err_out = (got_m / got_a - u_mean).abs() / u_mean;
    eprintln!("  [120.14] outlet: phase-weighted mean {:.17e} for Umean = {u_mean} (rel {err_out:e})", got_m / got_a);
    assert!(err_out <= tol(1e-12), "(120.14c) the phase mean is not Umean: rel {err_out:e}");

    // No liquid on the patch: the inlet writes 0, the outlet the cell value.
    let dry = vec![0.0 as Scalar; nb];
    let (mut fr, mut rv, mut rg) = (b.fr.clone(), b.ref_value.clone(), b.ref_grad.clone());
    vof_family::variable_height_inlet_velocity(inlet, &dry, &mut fr, &mut rv, &mut rg);
    vof_family::outlet_phase_mean_velocity(outlet, &dry, &u_cells, &mut fr, &mut rv, &mut rg);
    for i in 0..inlet.size {
        assert_eq!(rv[inlet.start + i], Vec3::ZERO, "a dry inlet face {i} carries nothing");
    }
    for i in 0..outlet.size {
        assert_eq!(rv[outlet.start + i], u_cells[outlet.cells[i]], "a dry outlet face {i} is the cell");
    }

    // ---- (120.14d) on p_rgh, on xMax.
    let g = Vec3::new(0.5, -1.0, -9.81);
    let ph = with(
        ps("phaseHydrostaticPressure"),
        &[("rho", "1000"), ("pRefValue", "1004.905"), ("pRefPoint", "(0.5 0.25 0.5)"), ("phaseFraction", ALPHA)],
    );
    let p_cells = vec![0.0 as Scalar; nc];
    let raw = raw_scalar("p_rgh", one_patch("xMax", ph, "zeroGradient"), p_cells.clone());
    let b = scalar_boundary_with(&raw, &hm, &p_cells, opt_in())?;
    assert_eq!(b.vof_rules.len(), 1, "one rule on xMax");
    let rule = &b.vof_rules[0];
    check_rule_geometry(rule, &hm, "xMax");
    let (rho, p_ref) = (1000.0 as Scalar as f64, 1004.905 as Scalar as f64);
    let x_ref = ofgpu::types::DVec3::new(0.5 as Scalar as f64, 0.25 as Scalar as f64, 0.5 as Scalar as f64);
    assert_eq!(rule.kind, VofFamilyKind::PhaseHydrostaticPressure { rho, p_ref, x_ref });
    let rho_b: Vec<Scalar> = (0..nb).map(|f| (1.0 + 999.0 * a_of(f)) as Scalar).collect();
    let (mut fr, mut rv, mut rg) = (b.fr.clone(), b.ref_value.clone(), b.ref_grad.clone());
    vof_family::phase_hydrostatic_pressure(rule, g, &alpha_b, &rho_b, &mut fr, &mut rv, &mut rg);
    let (gx, gy, gz) = (w64(g.x), w64(g.y), w64(g.z));
    for i in 0..rule.size {
        let f = rule.start + i;
        let xw = rule.x_world[i];
        let xl = rule.x_local[i];
        let gdr = gx * (xw.x - x_ref.x) + gy * (xw.y - x_ref.y) + gz * (xw.z - x_ref.z);
        let gh = gx * xl.x + gy * xl.y + gz * xl.z;
        let want = ((p_ref + rho * gdr) - w64(rho_b[f]) * gh) as Scalar;
        assert_eq!((fr[f], rv[f], rg[f]), (a_of(f) as Scalar, want, 0.0), "(120.14d) xMax face {i}");
    }
    Ok(())
}

/// The refusal `name` on `field`, opted in or not, must carry every phrase.
fn refused_scalar(field: &str, spec: PatchFieldSpec, inputs: BcInputs, phrases: &[&str]) -> Result<()> {
    let hm = mesh_a()?;
    let cells = vec![0.0 as Scalar; hm.n_cells];
    let raw = raw_scalar(field, one_patch("xMin", spec, "zeroGradient"), cells.clone());
    let err = scalar_boundary_with(&raw, &hm, &cells, inputs)
        .err()
        .unwrap_or_else(|| panic!("{field}: {phrases:?} must be refused"))
        .to_string();
    for p in phrases.iter().chain(["SPEC-LIT 120.14"].iter()) {
        assert!(err.contains(p), "the refusal must say `{p}`: {err}");
    }
    Ok(())
}

fn refused_vector(spec: PatchFieldSpec, inputs: BcInputs, phrases: &[&str]) -> Result<()> {
    let hm = mesh_a()?;
    let cells = vec![Vec3::ZERO; hm.n_cells];
    let raw = raw_u(one_patch("xMin", spec, "zeroGradient"), cells.clone());
    let err = vector_boundary_with(&raw, &hm, &cells, inputs)
        .err()
        .unwrap_or_else(|| panic!("U: {phrases:?} must be refused"))
        .to_string();
    for p in phrases.iter().chain(["SPEC-LIT 120.14"].iter()) {
        assert!(err.contains(p), "the refusal must say `{p}`: {err}");
    }
    Ok(())
}

#[test]
fn vof_family_refusals_are_by_name() -> Result<()> {
    let vh = || with(ps("variableHeightFlowRate"), &[("lowerBound", "0"), ("upperBound", "1")]);
    let inlet = || with(ps("variableHeightFlowRateInletVelocity"), &[("flowRate", "0.1")]);
    let outlet = || with(ps("outletPhaseMeanVelocity"), &[("Umean", "0.1")]);
    let ph = || {
        with(
            ps("phaseHydrostaticPressure"),
            &[("rho", "1000"), ("pRefValue", "0"), ("pRefPoint", "(0 0 0)")],
        )
    };
    let none = BcInputs::default();

    // No opt-in: every name, by name.
    refused_scalar(ALPHA, vh(), none, &["variableHeightFlowRate", "vof_family"])?;
    refused_scalar("p_rgh", ph(), none, &["phaseHydrostaticPressure", "vof_family"])?;
    refused_vector(inlet(), none, &["variableHeightFlowRateInletVelocity", "vof_family"])?;
    refused_vector(outlet(), none, &["outletPhaseMeanVelocity", "vof_family"])?;

    // The field a name belongs on.
    refused_scalar("p_rgh", vh(), opt_in(), &["belongs on alpha.water"])?;
    refused_scalar(ALPHA, ph(), opt_in(), &["belongs on p_rgh"])?;
    refused_scalar(ALPHA, inlet(), opt_in(), &["is a vector condition"])?;
    refused_vector(vh(), opt_in(), &["is a scalar condition"])?;

    // The entries.
    let only = |name: &str, e: &[(&str, &str)]| with(ps(name), e);
    refused_scalar(ALPHA, only("variableHeightFlowRate", &[("upperBound", "1")]), opt_in(), &["`lowerBound`"])?;
    refused_scalar(ALPHA, only("variableHeightFlowRate", &[("lowerBound", "0")]), opt_in(), &["`upperBound`"])?;
    for (l, u) in [("0.8", "0.2"), ("-0.1", "0.5"), ("0.5", "1.5")] {
        refused_scalar(
            ALPHA,
            only("variableHeightFlowRate", &[("lowerBound", l), ("upperBound", u)]),
            opt_in(),
            &["0 <= lowerBound <= upperBound <= 1"],
        )?;
    }
    refused_vector(ps("variableHeightFlowRateInletVelocity"), opt_in(), &["`flowRate`"])?;
    refused_vector(ps("outletPhaseMeanVelocity"), opt_in(), &["`Umean`"])?;
    refused_vector(
        only("variableHeightFlowRateInletVelocity", &[("flowRate", "table ((0 0.1) (1 0.2))")]),
        opt_in(),
        &["constant `flowRate`"],
    )?;
    refused_vector(
        only("outletPhaseMeanVelocity", &[("Umean", "table ((0 0.1) (1 0.2))")]),
        opt_in(),
        &["constant `Umean`"],
    )?;
    refused_scalar("p_rgh", only("phaseHydrostaticPressure", &[("pRefValue", "0"), ("pRefPoint", "(0 0 0)")]), opt_in(), &["`rho`"])?;
    refused_scalar("p_rgh", only("phaseHydrostaticPressure", &[("rho", "1000"), ("pRefPoint", "(0 0 0)")]), opt_in(), &["`pRefValue`"])?;
    refused_scalar("p_rgh", only("phaseHydrostaticPressure", &[("rho", "1000"), ("pRefValue", "0")]), opt_in(), &["`pRefPoint`"])?;

    // The phase fraction is the solver's own.
    refused_vector(with(inlet(), &[("alpha", "alpha.oil")]), opt_in(), &["`alpha` names `alpha.oil`"])?;
    refused_vector(with(outlet(), &[("alpha", "alpha.oil")]), opt_in(), &["`alpha` names `alpha.oil`"])?;
    refused_scalar("p_rgh", with(ph(), &[("phaseFraction", "alpha.oil")]), opt_in(), &["`phaseFraction` names `alpha.oil`"])?;
    Ok(())
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

fn water_air() -> VofProperties {
    VofProperties {
        rho1: 1000.0,
        rho2: 1.0,
        mu1: 1.002e-3,
        mu2: 1.8e-5,
        sigma: 0.0,
        g: Vec3::new(0.0, 0.0, -9.81),
        c_alpha: 1.0,
    }
}

/// The two-layer hydrostatic `p_rgh` with `p = 1000` at the top `z = 1` and
/// the interface at `z = 0.5`, each layer constant formed once in f64.
fn hydrostatic_p_rgh(hm: &HostMesh) -> Vec<Scalar> {
    let heavy = (1000.0 + 9.81 * 0.5 + 1000.0 * 9.81 * 0.5) as Scalar;
    let light = (1000.0 + 9.81 * (1.0 - 0.5) + 1.0 * 9.81 * 0.5) as Scalar;
    hm.c.iter().map(|c| if w64(c.z) < 0.5 { heavy } else { light }).collect()
}

fn prgh_top() -> PatchFieldSpec {
    with(ps("prghPressure"), &[("p", "uniform 1000")])
}

#[test]
fn a_channel_fed_at_a_variable_height_carries_the_liquid_rate() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    // A 2-D channel 2 x 1, water below z = 0.5: fed on xMin through
    // (120.14b) with alpha (120.14a) at [0.05, 0.95], drained on xMax
    // through (120.14c), slip floor, open top at p = 1000.
    let h = 1.0 / 12.0;
    let hm = blockgen::build_mesh(&box_spec(
        [24, 1, 12],
        [0.0, 0.0, 0.0],
        [2.0, h, 1.0],
        ["patch", "patch", "empty", "empty", "wall", "patch"],
    ))?;
    let m = GpuMesh::upload(&gpu, &hm)?;
    let ctrl = VofControls {
        delta_t: 5e-3,
        n_correctors: 3,
        u_solver: tight(),
        p_solver: tight(),
        sn_grad: fv::SnGradScheme::Uncorrected,
        ..VofControls::default()
    };
    let mut vof = Vof::new(&gpu, &hm, &m, water_air(), ctrl)?;
    let u0 = 0.1f64;
    let q = u0 * 0.5 * w64(h as Scalar);
    let wet = |c: &Vec3| w64(c.z) < 0.5;

    let a: Vec<Scalar> = hm.c.iter().map(|c| if wet(c) { 1.0 } else { 0.0 }).collect();
    let mut ba = one_patch(
        "xMin",
        with(ps("variableHeightFlowRate"), &[("lowerBound", "0.05"), ("upperBound", "0.95")]),
        "zeroGradient",
    );
    ba[2].1 = ps("empty");
    ba[3].1 = ps("empty");
    setup_scalar_field_with(&gpu, vof.alpha_mut(), &raw_scalar(ALPHA, ba, a), &hm, opt_in())?;

    let u: Vec<Vec3> = hm
        .c
        .iter()
        .map(|c| if wet(c) { Vec3::new(u0 as Scalar, 0.0, 0.0) } else { Vec3::ZERO })
        .collect();
    let q_text = format!("{q:.17e}");
    let bu: Vec<(&str, PatchFieldSpec)> = vec![
        ("xMin", with(ps("variableHeightFlowRateInletVelocity"), &[("flowRate", q_text.as_str())])),
        ("xMax", with(ps("outletPhaseMeanVelocity"), &[("Umean", "0.1")])),
        ("yMin", ps("empty")),
        ("yMax", ps("empty")),
        ("zMin", ps("slip")),
        ("zMax", ps("zeroGradient")),
    ];
    setup_vector_field_with(&gpu, vof.u_mut(), &raw_u(bu, u), &hm, opt_in())?;

    let rho_b = vof.boundary_density(&gpu)?;
    let mut bp = one_patch("zMax", prgh_top(), "zeroGradient");
    bp[2].1 = ps("empty");
    bp[3].1 = ps("empty");
    let inputs = BcInputs {
        prgh: Some(PrghInputs {
            g: water_air().g,
            rho_b: Some(&rho_b),
            u_b: None,
            ph_rgh_b: None,
        }),
        ..opt_in()
    };
    setup_scalar_field_with(&gpu, vof.p_rgh_mut(), &raw_scalar("p_rgh", bp, hydrostatic_p_rgh(&hm)), &hm, inputs)?;
    vof.initialise_flux_from_velocity(&gpu)?;
    vof.initialise(&gpu)?;
    assert!(!vof.pressure_is_pinned(), "the open top fixes the level");

    let (xi, ni) = patch_of(&hm, "xMin");
    let (xo, no) = patch_of(&hm, "xMax");
    let g64 = hm.geom64();
    let (mut worst_flux, mut worst_id, mut worst_mean, mut worst_clamp) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for step in 0..20 {
        let perf = vof.step(&gpu, 5e-3)?;
        assert_eq!(perf.n_sub_cycles, 1, "step {step}: no sub-cycling");

        // The liquid the step transported through the inlet: the
        // accumulated limited alpha flux of SPEC-LIT 20.2.
        let ap = gpu.download(&vof.alpha_phi().bf)?;
        let carried: f64 = (xi..xi + ni).map(|f| -w64(ap[f])).sum();
        worst_flux = nmax(worst_flux, (carried - q).abs() / q);

        // The condition's own state for the next step.
        vof.refresh_vof_family(&gpu)?;
        let ab = gpu.download(&vof.alpha().bf)?;
        let ac = gpu.download(&vof.alpha().f)?;
        let rv = gpu.download(&vof.u().ref_value)?;
        let af = |f: usize| w64(ab[f]).clamp(0.0, 1.0);
        let mut id = 0.0f64;
        for f in xi..xi + ni {
            let s = hm.b_sf[f];
            id += af(f) * (w64(rv[f].x) * w64(s.x) + w64(rv[f].y) * w64(s.y) + w64(rv[f].z) * w64(s.z));
            let p = ac[hm.b_face_cells[f] as usize];
            let want = p.clamp(0.05 as Scalar, 0.95 as Scalar);
            worst_clamp = nmax(worst_clamp, (w64(ab[f]) - w64(want)).abs());
        }
        worst_id = nmax(worst_id, (-id - q).abs() / q);
        let (mut area, mut mean) = (0.0f64, 0.0f64);
        for f in xo..xo + no {
            let s = g64.b_sf[f];
            let mag = g64.b_mag_sf[f];
            let un = (w64(rv[f].x) * s.x + w64(rv[f].y) * s.y + w64(rv[f].z) * s.z) / mag;
            area += af(f) * mag;
            mean += af(f) * un * mag;
        }
        worst_mean = nmax(worst_mean, (mean / area - u0).abs() / u0);
        let (l, u) = vof.alpha_bounds(&gpu)?;
        lo = lo.min(w64(l));
        hi = hi.max(w64(u));
    }
    eprintln!(
        "  [120.14] channel, 20 steps: transported liquid rate rel {worst_flux:e}, inlet identity \
         rel {worst_id:e}, outlet phase mean rel {worst_mean:e}, |alpha_b - clamp| {worst_clamp:e}, \
         alpha in [{lo:e}, {hi}]"
    );
    assert!(worst_flux <= tol(1e-12), "the transported liquid rate is not Q: rel {worst_flux:e}");
    assert!(worst_id <= tol(1e-12), "(120.14b) the inlet's liquid flux is not Q: rel {worst_id:e}");
    assert!(worst_mean <= tol(1e-12), "(120.14c) the outlet's phase mean is not Umean: rel {worst_mean:e}");
    assert_eq!(worst_clamp, 0.0, "(120.14a) alpha_b is clamp(alpha_P, 0.05, 0.95) exactly");
    assert!(lo >= -tol(1e-12) && hi <= 1.0 + tol(1e-12), "alpha left [0, 1]: [{lo:e}, {hi}]");
    Ok(())
}

/// A tank of water under air, 1 x 1, interface at z = 0.5, open at the top
/// (p = 1000) and at the side xMax through (120.14d) with `pRefValue`
/// `p_ref` at the interface; returns (max |U|, max |p - p_exact|,
/// max |p_exact|) after 20 steps.
fn side_tank(gpu: &Gpu, p_ref: &str) -> Result<(f64, f64, f64)> {
    let h = 1.0 / 24.0;
    let hm = blockgen::build_mesh(&box_spec(
        [24, 1, 24],
        [0.0, 0.0, 0.0],
        [1.0, h, 1.0],
        ["wall", "patch", "empty", "empty", "wall", "patch"],
    ))?;
    let m = GpuMesh::upload(gpu, &hm)?;
    let ctrl = VofControls {
        delta_t: 2.5e-4,
        n_correctors: 3,
        u_solver: tight(),
        p_solver: tight(),
        sn_grad: fv::SnGradScheme::Uncorrected,
        ..VofControls::default()
    };
    let mut vof = Vof::new(gpu, &hm, &m, water_air(), ctrl)?;
    let a: Vec<Scalar> = hm.c.iter().map(|c| if w64(c.z) < 0.5 { 1.0 } else { 0.0 }).collect();
    let mut ba = one_patch("xMin", ps("zeroGradient"), "zeroGradient");
    ba[2].1 = ps("empty");
    ba[3].1 = ps("empty");
    setup_scalar_field_with(gpu, vof.alpha_mut(), &raw_scalar(ALPHA, ba, a), &hm, opt_in())?;
    let bu: Vec<(&str, PatchFieldSpec)> = vec![
        ("xMin", ps("noSlip")),
        ("xMax", ps("zeroGradient")),
        ("yMin", ps("empty")),
        ("yMax", ps("empty")),
        ("zMin", ps("noSlip")),
        ("zMax", ps("zeroGradient")),
    ];
    setup_vector_field_with(gpu, vof.u_mut(), &raw_u(bu, vec![Vec3::ZERO]), &hm, opt_in())?;
    let rho_b = vof.boundary_density(gpu)?;
    let mut bp = one_patch("zMax", prgh_top(), "zeroGradient");
    bp[1].1 = with(
        ps("phaseHydrostaticPressure"),
        &[("rho", "1000"), ("pRefValue", p_ref), ("pRefPoint", "(0 0 0.5)")],
    );
    bp[2].1 = ps("empty");
    bp[3].1 = ps("empty");
    let inputs = BcInputs {
        prgh: Some(PrghInputs {
            g: water_air().g,
            rho_b: Some(&rho_b),
            u_b: None,
            ph_rgh_b: None,
        }),
        ..opt_in()
    };
    setup_scalar_field_with(gpu, vof.p_rgh_mut(), &raw_scalar("p_rgh", bp, hydrostatic_p_rgh(&hm)), &hm, inputs)?;
    vof.initialise_flux_from_velocity(gpu)?;
    vof.initialise(gpu)?;
    assert!(!vof.pressure_is_pinned(), "the open top fixes the level");
    for _ in 0..20 {
        vof.step(gpu, 2.5e-4)?;
    }
    let uf = gpu.download(&vof.u().f)?;
    let ub = gpu.download(&vof.u().bf)?;
    let mut maxu = 0.0f64;
    for v in uf.iter().chain(ub.iter()) {
        maxu = nmax(maxu, (w64(v.x) * w64(v.x) + w64(v.y) * w64(v.y) + w64(v.z) * w64(v.z)).sqrt());
    }
    let p_rgh = gpu.download(&vof.p_rgh().f)?;
    let rho = gpu.download(&vof.rho().f)?;
    let c = hm.geom64().c;
    let (mut worst, mut biggest) = (0.0f64, 0.0f64);
    for i in 0..hm.n_cells {
        let z = c[i].z;
        let pc = w64(p_rgh[i]) + w64(rho[i]) * (-9.81 * z);
        let want = if z > 0.5 {
            1000.0 + 9.81 * (1.0 - z)
        } else {
            1000.0 + 9.81 * 0.5 + 1000.0 * 9.81 * (0.5 - z)
        };
        biggest = nmax(biggest, want.abs());
        worst = nmax(worst, (pc - want).abs());
    }
    Ok((maxu, worst, biggest))
}

#[test]
fn a_tank_open_at_a_phase_hydrostatic_side_stays_at_rest() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    // (120.14d) with the column's own level at the interface: 1000 + 9.81 x
    // 0.5 of air above it. Wet side faces hold p = p_ref + rho g.(x - x_ref),
    // dry ones are zero-gradient; the tank stays at rest.
    let (maxu, worst, biggest) = side_tank(&gpu, "1004.905")?;
    eprintln!(
        "  [120.14] side tank at rest: max |U| {maxu:e}, max |p - p_exact| {worst:e} of {biggest:.6}"
    );
    assert!(maxu <= tol(1e-10), "the tank did not stay at rest: max |U| = {maxu:e}");
    assert!(worst <= tol(1e-12) * biggest, "the tank is not hydrostatic: {worst:e} of {biggest:.6}");

    // The control: the level 4.905 Pa short at the side, and the water moves.
    let (maxu, worst, _) = side_tank(&gpu, "1000")?;
    eprintln!("  [120.14] side tank control: max |U| {maxu:e}, max |p - p_exact| {worst:e}");
    assert!(maxu >= 1e-4, "the control stayed at rest after all: max |U| = {maxu:e}");
    Ok(())
}
