// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

// The DST-06 case-block gates (SPEC-LIT §157.4), written once and included
// both by src/io/tests_case_dust.rs (as io::case_dust::tests) and by
// tests/case_dust.rs, so both builds run the same eight cases. Host only:
// no card is needed. No GPL-licensed source was consulted.

use ofgpu::dust::courant::CO_F_MAX;
use ofgpu::dust::fsc::{A_FSC, CombustionModel};
use ofgpu::dust::laminar::LaminarSpeed;
use ofgpu::dust::vessel0d::{BurningLaw, UnburnedDensity};
use ofgpu::error::Error;
use ofgpu::io::case_chem::JsonCombustion;
use ofgpu::io::case_dust::{
    DustContext, DustThermo, JsonDust, LoweredDust, LoweredDustSst, decimal_fraction,
    is_dust_model, lower_dust,
};
use ofgpu::io::case_json::{JsonCase, LoweredCase, read_case_jsonc, read_case_jsonc_str};

/// The duct's six boundary names: an inlet at `xmin`, an open outlet at
/// `xmax`, symmetry sides - the heated-duct shape of `src/bin/lowmach.rs`.
const BOUNDS: &str = r#"{ "xmin": "inlet", "xmax": "outlet", "ymin": "sideA",
    "ymax": "sideB", "zmin": "sideC", "zmax": "sideD" }"#;

/// The duct's patch rules at initial temperature `t`: the inlet carries the
/// fresh charge, the outlet is open, the sides are the mirror plane.
fn duct_patches(t: &str) -> String {
    format!(
        r#"[ {{ "match": "inlet", "kind": "inlet", "U": {{ "type": "fixedValue", "value": [1,0,0] }},
                "T": {{ "type": "fixedValue", "value": {t} }} }},
             {{ "match": "outlet", "kind": "open", "U": {{ "type": "zeroGradient" }},
                "p": {{ "type": "fixedValue", "value": 0.0 }} }},
             {{ "match": "side.*", "kind": "symmetry" }} ]"#
    )
}

/// A 16x1x1 cartesian duct at the fresh charge of `patches`/`initial`;
/// `physics_extra` goes inside `physics` and `turbulence` is a top-level
/// block (`""` for none) - each argument's own leading comma included.
#[allow(clippy::too_many_arguments)]
fn duct_case_text(
    physics_extra: &str,
    turbulence: &str,
    boundaries: &str,
    patches: &str,
    initial: &str,
    blocks: &str,
) -> String {
    format!(
        r#"{{
  "name": "dustTest",
  "mesh": {{ "kind": "cartesian", "bounds": {{ "min": [0,0,0], "max": [1.0,0.1,0.1] }},
    "cells": [16,1,1], "boundaries": {boundaries} }},
  "physics": {{ "gravity": [0,0,0], "fluid": {{ "nu": 1.5e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 300.0 }},
    "buoyancy": "densityRatio"{physics_extra} }}{turbulence},
  "patches": {patches},
  "initial": {initial},
  "numerics": {{
    "algorithm": {{ "kind": "SIMPLE" }},
    "ddt": "steadyState",
    "div": {{ "default": "Gauss upwind" }},
    "grad": "Gauss linear",
    "laplacian": {{ "snGrad": "corrected", "nonOrthogonalCorrectors": 0 }},
    "solvers": [] }},
  "run": {{ "endTime": 1.0, "deltaT": 1.0 }}{blocks}
}}
"#
    )
}

/// The duct case at unburned temperature `t` with the `combustion` and
/// `dust` blocks' text (`""` for an absent one).
fn dust_case(t: &str, combustion: &str, dust: &str) -> String {
    duct_case_text(
        "",
        "",
        BOUNDS,
        &duct_patches(t),
        &format!(r#"{{ "U": [0,0,0], "T": {t}, "p": 0.0 }}"#),
        &format!("{combustion}{dust}"),
    )
}

/// The rho-weighted SST case: kOmegaSST with the initial k/omega and their
/// per-patch conditions the reader builds the closure fields from,
/// `gravity` substituted into `physics`, `physics_extra` beside
/// `buoyancy`, and the dust block's text `dust`.
fn sst_case_text(gravity: &str, physics_extra: &str, dust: &str) -> String {
    let turbulence = r#", "turbulence": { "kind": "RAS", "model": "kOmegaSST",
        "wallFunctions": { "kappa": 0.41, "E": 9.8 } }"#;
    let patches = r#"[ { "match": "inlet", "kind": "inlet", "U": { "type": "fixedValue", "value": [1,0,0] },
            "T": { "type": "fixedValue", "value": 273 },
            "k": { "type": "fixedValue", "value": 0.84 },
            "omega": { "type": "fixedValue", "value": 8.0 } },
         { "match": "outlet", "kind": "open", "U": { "type": "zeroGradient" },
            "p": { "type": "fixedValue", "value": 0.0 },
            "k": { "type": "fixedValue", "value": 0.84 },
            "omega": { "type": "fixedValue", "value": 8.0 } },
         { "match": "side.*", "kind": "symmetry" } ]"#;
    let initial = r#"{ "U": [0,0,0], "T": 273, "p": 0.0, "k": 0.84, "omega": 8.0 }"#;
    duct_case_text(
        physics_extra,
        turbulence,
        BOUNDS,
        patches,
        initial,
        &format!("{FSC_MIN}{dust}"),
    )
    .replace(r#""gravity": [0,0,0]"#, &format!(r#""gravity": {gravity}"#))
}

fn lowers(text: &str) -> Result<LoweredCase, Error> {
    read_case_jsonc_str(text, "dustTest").and_then(|c| c.lower())
}

fn dust_of(text: &str) -> Result<LoweredDust, Error> {
    lowers(text)?
        .dust
        .ok_or_else(|| Error::Config("the case lowered to no dust config".to_string()))
}

fn refuses<T>(r: Result<T, Error>, key: &str) {
    match &r {
        Err(Error::Refused(m)) => assert!(m.contains(key), "the refusal does not name {key}: {m}"),
        Err(e) => panic!("{key}: expected a refusal, got {e}"),
        Ok(_) => panic!("{key}: expected a refusal, got Ok"),
    }
}

/// The FSC case of `dust_lowers_formula`, block text as the case states it.
const FSC_CO: &str = r#", "combustion": { "model": "FSC", "A": 0.45, "PrT": 0.28, "tFlacc": 0.15,
    "maxFlameCourant": 0.3,
    "ignition": { "point": [0.01, 0.02, 0.03], "sigmaR": 5e-4, "W0": 1e7, "t0": 2e-3, "sigmaT": 1e-3 } }"#;
const FORMULA_DUST: &str = r#", "dust": { "thermo": "formula", "phi": 0.62, "SL": 0.25 }"#;
/// The TFC case of `dust_lowers_prescribed`, and the minimal dust side.
const TFC_CO: &str = r#", "combustion": { "model": "TFC", "PrT": 0.4 }"#;
const PRESCRIBED_DUST: &str = r#", "dust": { "thermo": "prescribed", "cp": 1000, "Wu": 0.0328,
    "Wb": 0.0297, "Tb": 1500, "SL": 0.12 }"#;
/// The smallest legal FSC pair: no ignition, no A, formula at phi with the
/// stated SL.
const FSC_MIN: &str = r#", "combustion": { "model": "FSC", "PrT": 0.3 }"#;
const FORMULA_MIN: &str = r#", "dust": { "thermo": "formula", "phi": 0.62, "SL": 0.25 }"#;

/// A minimal but lowering species set, and its chemistry - for the refusals
/// that need the case to state them beside the dust model, and for PrT
/// under EDM.
const SPECIES: &str = r#", "species": { "names": ["H2", "O2", "H2O", "N2"], "inert": "N2",
    "initial": { "H2": 0.02, "O2": 0.22, "H2O": 0.0 },
    "patches": [ { "match": "inlet", "Y": { "H2": { "type": "fixedValue", "value": 0.03 },
        "O2": { "type": "fixedValue", "value": 0.23 },
        "H2O": { "type": "zeroGradient" } } } ] }"#;
const CHEMISTRY: &str = r#", "chemistry": { "mechanism": "mech.inp", "thermo": "therm.dat" }"#;

/// §157.4 `io::case_dust::tests::lowers`, arm 1: an FSC case with every
/// combustion key stated lowers to exactly those values, its phi is the
/// exact decimal 0.62, and its charge is the AICC one of §152.11.
#[test]
fn dust_lowers_formula() -> Result<(), Error> {
    let d = dust_of(&dust_case("273", FSC_CO, FORMULA_DUST))?;
    assert_eq!(d.model, CombustionModel::Fsc);
    assert_eq!(d.a, 0.45);
    assert_eq!(d.pr_t, 0.28);
    assert_eq!(d.t_flacc, Some(0.15));
    assert_eq!(d.max_flame_courant, 0.3);
    let ign = d.ignition.expect("the case states an ignition source");
    assert_eq!((ign.point.x, ign.point.y, ign.point.z), (0.01, 0.02, 0.03));
    assert_eq!(ign.sigma_r, 5e-4);
    assert_eq!(ign.w0, 1e7);
    assert_eq!(ign.t0, 2e-3);
    assert_eq!(ign.sigma_t, 1e-3);
    assert_eq!(d.thermo, DustThermo::Formula { phi_num: 62, phi_den: 100 });
    assert_eq!(d.t_u, 273.0);
    assert_eq!(d.s_l, 0.25);
    assert_eq!(d.sst, None);
    assert_eq!(d.c_field.name, "c");
    assert_eq!(d.c_field.internal_uniform, 0.0 as ofgpu::Scalar);
    let a = d.gas()?.aicc(273.0)?;
    assert!(
        (a.p_ratio - 8.600607635).abs() <= 1e-8,
        "P_max/p_0 = {} is not 8.600607635 (SPEC-LIT §152.11)",
        a.p_ratio
    );
    Ok(())
}

/// §157.4's second arm: a TFC case at the prescribed Leeds thermo - A at
/// the FSC default (the Zimont 0.52 is a named non-default input), no
/// clock, no ignition, and sigma 5.0505 from the stated constants.
#[test]
fn dust_lowers_prescribed() -> Result<(), Error> {
    let d = dust_of(&dust_case("328", TFC_CO, PRESCRIBED_DUST))?;
    assert_eq!(d.model, CombustionModel::Tfc);
    assert_eq!(d.a, A_FSC);
    assert_eq!(d.pr_t, 0.4);
    assert_eq!(d.max_flame_courant, CO_F_MAX);
    assert_eq!(d.max_flame_courant, 0.5);
    assert_eq!(d.ignition, None);
    assert_eq!(d.t_flacc, None);
    assert_eq!(
        d.thermo,
        DustThermo::Prescribed { cp: 1000.0, w_u: 0.0328, w_b: 0.0297, t_b: 1500.0 }
    );
    assert_eq!(d.t_u, 328.0);
    assert_eq!(d.s_l, 0.12);
    let b = d.gas()?.isobaric_burned(328.0)?;
    let want = 5.050505050505;
    assert!(
        (b.sigma - want).abs() <= want * 1e-12,
        "sigma = {} is not 5.050505050505 (SPEC-LIT §152.9)",
        b.sigma
    );
    Ok(())
}

/// The defaults: sigmaT omitted is t0 (§151.7 item 8), SL omitted in the
/// formula mode is (152.13)'s S_L(phi) bit for bit, A is A_FSC and the
/// flame-Courant limit CO_F_MAX.
#[test]
fn dust_defaults() -> Result<(), Error> {
    let co = r#", "combustion": { "model": "FSC", "PrT": 0.3,
        "ignition": { "point": [0,0,0], "sigmaR": 5e-4, "W0": 1e7, "t0": 2e-3 } }"#;
    let d = dust_of(&dust_case("273", co, r#", "dust": { "thermo": "formula", "phi": 0.62 }"#))?;
    let ign = d.ignition.expect("the case states an ignition source");
    assert_eq!(ign.sigma_t, ign.t0, "sigmaT omitted is t0");
    assert_eq!(ign.sigma_t, 2e-3);
    let want = LaminarSpeed::huang2022().s_l_phi(0.62)?;
    assert_eq!(d.s_l, want, "the default S_L is (152.13) bit for bit");
    assert!(
        (d.s_l - 0.2411076681537468).abs() <= 0.2411076681537468 * 1e-12,
        "S_L(0.62) = {} is not 0.2411076681537468 (SPEC-LIT (152.13))",
        d.s_l
    );
    assert_eq!(d.a, A_FSC);
    assert_eq!(d.max_flame_courant, CO_F_MAX);
    assert_eq!(d.t_flacc, None);
    Ok(())
}

/// phi is the exact fraction of the decimal the case wrote (§152.11), and a
/// phi that is not a finite non-negative decimal is refused by name.
#[test]
fn dust_phi_is_the_exact_decimal() {
    for (v, want) in [(0.62, (62, 100)), (0.5, (5, 10)), (1.0, (1, 1)), (0.3, (3, 10))] {
        assert_eq!(decimal_fraction(v).unwrap(), want, "{v} is not {want:?}");
    }
    for bad in [-0.1, f64::NAN] {
        let e = decimal_fraction(bad).expect_err(&format!("{bad} lowers"));
        assert!(e.to_string().contains("dust.phi"), "{bad}: {e}");
    }
}

/// The `c` field's defaults: internal 0, the inlet fixedValue [0], the open
/// boundary inletOutlet [0], the constraints their own class, and a wall
/// zeroGradient - so this case's mesh names its `zmin`/`zmax` walls.
#[test]
fn dust_c_field_defaults() {
    let boundaries = r#"{ "xmin": "inlet", "xmax": "outlet", "ymin": "sideA",
        "ymax": "sideB", "zmin": "wallA", "zmax": "wallB" }"#;
    let patches = r#"[ { "match": "inlet", "kind": "inlet", "U": { "type": "fixedValue", "value": [1,0,0] },
            "T": { "type": "fixedValue", "value": 273 } },
         { "match": "outlet", "kind": "open", "U": { "type": "zeroGradient" },
            "p": { "type": "fixedValue", "value": 0.0 } },
         { "match": "wall.*", "kind": "wall" },
         { "match": "side.*", "kind": "symmetry" } ]"#;
    let text = duct_case_text(
        "",
        "",
        boundaries,
        patches,
        r#"{ "U": [0,0,0], "T": 273, "p": 0.0 }"#,
        &format!("{FSC_MIN}{FORMULA_MIN}"),
    );
    let c = &dust_of(&text).expect("the wall case lowers").c_field;
    assert_eq!(c.name, "c");
    assert_eq!(c.dimensions, "[0 0 0 0 0 0 0]");
    assert_eq!(c.internal_uniform, 0.0 as ofgpu::Scalar);
    assert_eq!(c.boundary["inlet"].type_name, "fixedValue");
    assert_eq!(c.boundary["inlet"].value, vec![0.0 as ofgpu::Scalar]);
    assert_eq!(c.boundary["outlet"].type_name, "inletOutlet");
    assert_eq!(c.boundary["outlet"].inlet_value, vec![0.0 as ofgpu::Scalar]);
    assert_eq!(c.boundary["outlet"].value, vec![0.0 as ofgpu::Scalar]);
    for p in ["sideA", "sideB"] {
        assert_eq!(c.boundary[p].type_name, "symmetry", "{p}: the constraint's own class");
    }
    for p in ["wallA", "wallB"] {
        assert_eq!(c.boundary[p].type_name, "zeroGradient", "{p}: the wall's default");
    }
}

/// `dust.sst` lowers onto the SST settings: the ω-production limiter's
/// setting and SST-sust's ambient state (§156.3, §156.4), beside the
/// rho-weighted model with the initial k/omega the reader builds its
/// fields from.
#[test]
fn dust_sst_lowers() {
    let dust = r#", "dust": { "thermo": "formula", "phi": 0.62, "SL": 0.25,
        "sst": { "limitOmegaProduction": true, "sust": { "kAmb": 0.84, "omegaAmb": 8.0 } } }"#;
    let l = lowers(&sst_case_text("[0,0,0]", "", dust)).expect("the SST case lowers");
    assert_eq!(
        l.dust.expect("the dust config").sst,
        Some(LoweredDustSst { limit_omega_production: true, sust: Some((0.84, 8.0)) })
    );
}

/// The block survives a serde round trip unchanged, and a case without the
/// blocks serialises no `"dust"` key and is not refused.
#[test]
fn dust_round_trip_and_absent() {
    let case: JsonCase = read_case_jsonc_str(&dust_case("273", FSC_CO, FORMULA_DUST), "dustTest")
        .expect("the dust case parses");
    let back: JsonCase = serde_json::from_str(&serde_json::to_string(&case).unwrap())
        .expect("the serialised case re-parses");
    assert_eq!(back, case, "the blocks survive a serde round trip");
    let lowered = lowers(&dust_case("273", FSC_CO, FORMULA_DUST)).expect("the dust case lowers");
    assert!(lowered.dust.is_some());
    let e = lowered.refuse_dust().expect_err("every driver but ofgpu-lowmach refuses");
    assert!(e.to_string().contains("dust"), "{e}");
    assert!(e.to_string().contains("ofgpu-lowmach"), "{e}");

    let bare_text = dust_case("273", "", "");
    let bare = lowers(&bare_text).expect("a case without the blocks lowers");
    assert!(bare.dust.is_none());
    bare.refuse_dust().expect("a case without the blocks is not refused");
    let text = serde_json::to_string(&read_case_jsonc_str(&bare_text, "dustTest").unwrap())
        .unwrap();
    assert!(!text.contains("\"dust\""), "an absent dust block is not serialised: {text}");
    // The model predicate the lowering is keyed on.
    let parse = |s: &str| -> JsonCombustion { serde_json::from_str(s).unwrap() };
    assert!(!is_dust_model(None));
    assert!(!is_dust_model(Some(&parse(r#"{ "model": "laminar" }"#))));
    assert!(!is_dust_model(Some(&parse(r#"{ "model": "EDM", "A": 4 }"#))));
    assert!(is_dust_model(Some(&parse(r#"{ "model": "FSC", "PrT": 0.3 }"#))));
    assert!(is_dust_model(Some(&parse(r#"{ "model": "TFC", "PrT": 0.3 }"#))));
}

/// §157.4 `io::case_dust::tests::refusals`: every refusal of the lowering,
/// each naming its key, then the driver seam's own refusal of a dust case.
#[test]
fn dust_refusals() {
    let fsc = |extra: &str| format!(r#", "combustion": {{ "model": "FSC", "PrT": 0.3{extra} }}"#);
    let tfc = |extra: &str| format!(r#", "combustion": {{ "model": "TFC", "PrT": 0.3{extra} }}"#);
    // 1. a dust block without an FSC/TFC model (and without any model).
    refuses(lowers(&dust_case("273", r#", "combustion": { "model": "laminar" }"#, FORMULA_MIN)), "dust");
    refuses(lowers(&dust_case("273", "", FORMULA_MIN)), "dust");
    // 2. the model without a dust block.
    refuses(lowers(&dust_case("273", &fsc(""), "")), "combustion.model");
    refuses(lowers(&dust_case("273", &tfc(""), "")), "combustion.model");
    // 3. the case's species set and mechanism beside the model's own gas.
    refuses(lowers(&dust_case("273", &fsc(""), &format!("{FORMULA_MIN}{SPECIES}"))), "species");
    refuses(lowers(&dust_case("273", &fsc(""), &format!("{FORMULA_MIN}{CHEMISTRY}"))), "chemistry");
    // 4. a key of EDM/EDC/PaSR beside the model.
    refuses(lowers(&dust_case("273", &fsc(r#", "B": 0.5"#), FORMULA_MIN)), "combustion.B");
    refuses(lowers(&dust_case("273", &fsc(r#", "CD1": 0.1"#), FORMULA_MIN)), "combustion.CD1");
    refuses(lowers(&dust_case("273", &tfc(r#", "tauMix": "integral", "tauChem": "sfr""#), FORMULA_MIN)), "combustion.tauMix");
    refuses(lowers(&dust_case("273", &tfc(r#", "Cmix": 0.3"#), FORMULA_MIN)), "combustion.Cmix");
    // 5. Pr_t is required - there is no default.
    refuses(
        lowers(&dust_case("273", r#", "combustion": { "model": "FSC" }"#, FORMULA_MIN)),
        "combustion.PrT",
    );
    // 6. TFC has no clock.
    refuses(lowers(&dust_case("273", &tfc(r#", "tFlacc": 0.15"#), FORMULA_MIN)), "combustion.tFlacc");
    // 7. every stated number is finite and above zero (t0 at or above zero).
    refuses(lowers(&dust_case("273", &fsc(r#", "A": 0"#), FORMULA_MIN)), "combustion.A");
    refuses(lowers(&dust_case("273", &fsc(r#", "PrT": 0"#), FORMULA_MIN)), "combustion.PrT");
    refuses(lowers(&dust_case("273", &fsc(r#", "tFlacc": 0"#), FORMULA_MIN)), "combustion.tFlacc");
    refuses(lowers(&dust_case("273", &fsc(r#", "maxFlameCourant": -1"#), FORMULA_MIN)), "combustion.maxFlameCourant");
    let ign = |extra: &str| {
        fsc(&format!(
            r#", "ignition": {{ "point": [0.01, 0, 0], "sigmaR": 5e-4, "W0": 1e7, "t0": 2e-3{extra} }}"#
        ))
    };
    refuses(lowers(&dust_case("273", &fsc(r#", "ignition": { "point": [0.01, 0, 0], "sigmaR": 0, "W0": 1e7, "t0": 2e-3 }"#), FORMULA_MIN)), "ignition.sigmaR");
    refuses(lowers(&dust_case("273", &fsc(r#", "ignition": { "point": [0.01, 0, 0], "sigmaR": 5e-4, "W0": -1, "t0": 2e-3 }"#), FORMULA_MIN)), "ignition.W0");
    refuses(lowers(&dust_case("273", &fsc(r#", "ignition": { "point": [0.01, 0, 0], "sigmaR": 5e-4, "W0": 1e7, "t0": -2e-3 }"#), FORMULA_MIN)), "ignition.t0");
    refuses(lowers(&dust_case("273", &ign(r#", "sigmaT": 0"#), FORMULA_MIN)), "ignition.sigmaT");
    // 8. the thermo mode's name.
    refuses(
        lowers(&dust_case("273", &fsc(""), r#", "dust": { "thermo": "gulder", "phi": 0.62 }"#)),
        "dust.thermo",
    );
    // 9. the formula mode: phi required, the prescribed keys refused.
    refuses(
        lowers(&dust_case("273", &fsc(""), r#", "dust": { "thermo": "formula", "SL": 0.25 }"#)),
        "dust.phi",
    );
    refuses(
        lowers(&dust_case("273", &fsc(""), r#", "dust": { "thermo": "formula", "phi": 0.62, "cp": 1000 }"#)),
        "dust.cp",
    );
    refuses(
        lowers(&dust_case("273", &fsc(""), r#", "dust": { "thermo": "formula", "phi": 0.62, "Wu": 0.0328 }"#)),
        "dust.Wu",
    );
    refuses(
        lowers(&dust_case("273", &fsc(""), r#", "dust": { "thermo": "formula", "phi": 0.62, "Wb": 0.0297 }"#)),
        "dust.Wb",
    );
    refuses(
        lowers(&dust_case("273", &fsc(""), r#", "dust": { "thermo": "formula", "phi": 0.62, "Tb": 1500 }"#)),
        "dust.Tb",
    );
    // 10-11. the prescribed mode: all five keys required, phi refused, and
    // every stated number finite and above zero.
    for (text, key) in [
        (r#", "dust": { "thermo": "prescribed", "Wu": 0.0328, "Wb": 0.0297, "Tb": 1500, "SL": 0.12 }"#, "dust.cp"),
        (r#", "dust": { "thermo": "prescribed", "cp": 1000, "Wb": 0.0297, "Tb": 1500, "SL": 0.12 }"#, "dust.Wu"),
        (r#", "dust": { "thermo": "prescribed", "cp": 1000, "Wu": 0.0328, "Tb": 1500, "SL": 0.12 }"#, "dust.Wb"),
        (r#", "dust": { "thermo": "prescribed", "cp": 1000, "Wu": 0.0328, "Wb": 0.0297, "SL": 0.12 }"#, "dust.Tb"),
        (r#", "dust": { "thermo": "prescribed", "cp": 1000, "Wu": 0.0328, "Wb": 0.0297, "Tb": 1500 }"#, "dust.SL"),
        (r#", "dust": { "thermo": "prescribed", "cp": 1000, "phi": 0.62, "Wu": 0.0328, "Wb": 0.0297, "Tb": 1500, "SL": 0.12 }"#, "dust.phi"),
        (r#", "dust": { "thermo": "prescribed", "cp": -1, "Wu": 0.0328, "Wb": 0.0297, "Tb": 1500, "SL": 0.12 }"#, "dust.cp"),
        (r#", "dust": { "thermo": "prescribed", "cp": 1000, "Wu": 0.0, "Wb": 0.0297, "Tb": 1500, "SL": 0.12 }"#, "dust.Wu"),
        (r#", "dust": { "thermo": "prescribed", "cp": 1000, "Wu": 0.0328, "Wb": 0.0297, "Tb": 0.0, "SL": 0.12 }"#, "dust.Tb"),
        (r#", "dust": { "thermo": "formula", "phi": 0.62, "SL": 0.0 }"#, "dust.SL"),
    ] {
        refuses(lowers(&dust_case("273", &fsc(""), text)), key);
    }
    // 12. turbulence other than laminar or the rho-weighted kOmegaSST.
    let turb = |kind: &str, model: &str| {
        format!(
            r#", "turbulence": {{ "kind": "{kind}", "model": "{model}", "wallFunctions": {{ "kappa": 0.41, "E": 9.8 }} }}"#
        )
    };
    refuses(lowers(&dust_case("273", &fsc(""), &format!("{FORMULA_MIN}{}", turb("RAS", "kEpsilon")))), "turbulence.model");
    refuses(lowers(&dust_case("273", &fsc(""), &format!("{FORMULA_MIN}{}", turb("LES", "Smagorinsky")))), "turbulence.model");
    refuses(lowers(&dust_case("273", &fsc(""), &format!("{FORMULA_MIN}{}", turb("RAS", "kOmegaSSTLM")))), "turbulence.model");
    // 13. the SST settings without the model.
    refuses(
        lowers(&dust_case(
            "273",
            &fsc(""),
            r#", "dust": { "thermo": "formula", "phi": 0.62, "SL": 0.25, "sst": { "limitOmegaProduction": true } }"#,
        )),
        "dust.sst",
    );
    // 14. the rho-weighted SST carries no buoyancy production; 15. the
    // compressible coupling is DST-10.
    let compressible = r#", "compressible": { "gamma": 1.4, "W": 0.0289647,
        "cp": 1004.692579691832, "viscosity": { "model": "sutherland", "As": 1.458e-6, "Ts": 110.4 },
        "pOperating": 1e5 }"#;
    let sst_dust = r#", "dust": { "thermo": "formula", "phi": 0.62, "SL": 0.25, "sst": { "limitOmegaProduction": true } }"#;
    let grav_case = sst_case_text("[0, 9.81, 0]", "", sst_dust);
    refuses(lowers(&grav_case), "physics.gravity");
    lowers(&sst_case_text("[0,0,0]", "", sst_dust))
        .expect("the rho-weighted SST case with no gravity lowers");
    refuses(lowers(&sst_case_text("[0,0,0]", &compressible, sst_dust)), "physics.compressible");
    // 16. T_u is the case's initial T.
    let pair = format!("{FSC_MIN}{FORMULA_MIN}");
    let cold = duct_case_text(
        "",
        "",
        BOUNDS,
        r#"[ { "match": "inlet", "kind": "inlet", "U": { "type": "fixedValue", "value": [1,0,0] } },
            { "match": "outlet", "kind": "open", "U": { "type": "zeroGradient" }, "p": { "type": "fixedValue", "value": 0.0 } },
            { "match": "side.*", "kind": "symmetry" } ]"#,
        r#"{ "U": [0,0,0], "p": 0.0 }"#,
        &pair,
    );
    refuses(lowers(&cold), "initial.T");
    // PrT under EDM: a key of FSC/TFC refused by the chemistry side's own
    // table, which needs the case's species set to reach.
    let edm = r#", "combustion": { "model": "EDM", "A": 4.0, "PrT": 0.28 }"#;
    refuses(
        lowers(&dust_case("273", edm, &format!("{SPECIES}{CHEMISTRY}"))),
        "combustion.PrT",
    );
    // The driver seam: a lowered dust case is refused, a bare one is not.
    let dust_lowered = lowers(&dust_case("273", &fsc(""), FORMULA_MIN)).expect("lowers");
    let e = dust_lowered.refuse_dust().expect_err("the driver refuses the dust case");
    assert!(e.to_string().contains("dust") && e.to_string().contains("ofgpu-lowmach"), "{e}");
    let bare = lowers(&dust_case("273", "", "")).expect("lowers");
    bare.refuse_dust().expect("the bare case is not refused");
    // refuse_dust_case's text is the driver seam's one message.
    let e = ofgpu::io::case_dust::refuse_dust_case("ofgpu-buoyant");
    assert!(e.to_string().contains("ofgpu-buoyant") && e.to_string().contains("DST-06"), "{e}");
}

/// **SPEC-LIT §157.6.** The three Gate DST-VES case files lower: sphere20L
/// carries the prescribed `U_t` law and the compressed density and lands on
/// the 200-cell REDUCED sector, sphere1m3 the same at its radius, and
/// rembeClosed states neither key. Host only: no card is needed.
#[test]
fn vessel_cases_lower() {
    let case = |name: &str| {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../cases/dust/{name}.jsonc"));
        read_case_jsonc(&path)
            .unwrap_or_else(|e| panic!("{} parses: {e}", path.display()))
            .lower()
            .unwrap_or_else(|e| panic!("{} lowers: {e}", path.display()))
    };
    let law = BurningLaw { s0: 0.5, alpha: 2.0, beta: -0.5 };
    let sphere = case("sphere20L");
    {
        let d = sphere.dust.as_ref().expect("the case states dust");
        assert_eq!(d.model, CombustionModel::Tfc);
        assert_eq!(d.prescribed_ut, Some(law));
        assert_eq!(d.unburned_density, UnburnedDensity::Compressed);
    }
    assert_eq!(sphere.build_host_mesh().expect("mesh").n_cells, 200);
    let big = case("sphere1m3");
    let d = big.dust.as_ref().expect("the case states dust");
    assert_eq!(d.model, CombustionModel::Tfc);
    assert_eq!(d.prescribed_ut, Some(law));
    assert_eq!(d.unburned_density, UnburnedDensity::Compressed);
    let rembe = case("rembeClosed");
    let d = rembe.dust.as_ref().expect("the case states dust");
    assert_eq!(d.model, CombustionModel::Fsc);
    assert_eq!(d.prescribed_ut, None);
    assert_eq!(d.unburned_density, UnburnedDensity::Compressed);
}

/// **SPEC-LIT §157.6.** Every refusal of the two Gate DST-VES keys names
/// its JSON path: `prescribedUt` beside FSC, beside a turbulence model and
/// with a value the law refuses, and `unburnedDensity` misspelled or
/// `"initial"` without the law.
#[test]
fn prescribed_ut_refusals_name_the_setting() {
    // The TFC operator is Gate DST-VES's alone: refused beside model FSC.
    let fsc_ut = r#", "combustion": { "model": "FSC", "PrT": 0.3,
        "prescribedUt": { "S0": 0.5, "alpha": 2.0, "beta": -0.5 } }"#;
    refuses(lowers(&dust_case("300", fsc_ut, FORMULA_MIN)), "combustion.prescribedUt");
    // The law holds U_t itself, so a turbulence model has no job: refused
    // beside kOmegaSST (the FSC_MIN text re-typed TFC with the law).
    let sst_ut = sst_case_text("[0,0,0]", "", &PRESCRIBED_DUST).replace(
        r#""model": "FSC""#,
        r#""model": "TFC", "prescribedUt": { "S0": 0.5, "alpha": 2.0, "beta": -0.5 }"#,
    );
    refuses(lowers(&sst_ut), "combustion.prescribedUt");
    // S0 has to be finite and above zero; the law's own refusal, keyed.
    let s0_zero = r#", "combustion": { "model": "TFC", "PrT": 0.3,
        "prescribedUt": { "S0": 0.0 } }"#;
    refuses(lowers(&dust_case("300", s0_zero, FORMULA_MIN)), "combustion.prescribedUt");
    // alpha/beta not finite cannot ride through JSON, so this one drives
    // `lower_dust` directly.
    let cx_names = vec!["vessel".to_string()];
    let cx = DustContext {
        species: false,
        chemistry: false,
        turbulence_kind: None,
        turbulence_model: None,
        compressible: false,
        gravity: [0.0, 0.0, 0.0],
        initial_t: Some(300.0),
        patch_names: &cx_names,
        rules: &[],
    };
    let dust = JsonDust {
        thermo: "prescribed".to_string(),
        phi: None,
        cp: Some(1000.0),
        w_u: Some(0.029),
        w_b: Some(0.029),
        t_b: Some(2100.0),
        s_l: Some(0.5),
        sst: None,
    };
    let combustion = |u: ofgpu::io::case_chem::JsonPrescribedUt| JsonCombustion {
        model: Some("TFC".to_string()),
        a: None,
        b: None,
        variant: None,
        cd1: None,
        cd2: None,
        gamma_max: None,
        psr_span: None,
        tau_mix: None,
        tau_chem: None,
        c_mix: None,
        pr_t: Some(0.7),
        t_flacc: None,
        max_flame_courant: None,
        ignition: None,
        prescribed_ut: Some(u),
        unburned_density: None,
    };
    for (s0, alpha, beta) in [(0.5, f64::NAN, 0.0), (0.5, 2.0, f64::INFINITY)] {
        let u = ofgpu::io::case_chem::JsonPrescribedUt { s0, alpha, beta };
        refuses(
            lower_dust(Some(&combustion(u)), Some(&dust), &cx),
            "combustion.prescribedUt",
        );
    }
    // The negative control's key: a spelling that is not one of the two
    // modes, and "initial" without the law it controls against.
    let bogus = r#", "combustion": { "model": "TFC", "PrT": 0.3,
        "unburnedDensity": "bogus" }"#;
    refuses(lowers(&dust_case("300", bogus, FORMULA_MIN)), "combustion.unburnedDensity");
    let initial_bare = r#", "combustion": { "model": "TFC", "PrT": 0.3,
        "unburnedDensity": "initial" }"#;
    refuses(
        lowers(&dust_case("300", initial_bare, FORMULA_MIN)),
        "combustion.unburnedDensity",
    );
}
