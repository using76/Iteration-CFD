// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

// The DST-06 case-block gates (SPEC-LIT §157.4) and DST-14's cloud gates
// (SPEC-LIT §159.9), written once and included both by
// src/io/tests_case_dust.rs (as io::case_dust::tests) and by
// tests/case_dust.rs, so both builds run the same cases. Host only: no card
// is needed. No GPL-licensed source was consulted.

use ofgpu::DVec3;
use ofgpu::dust::age::FlameAge;
use ofgpu::dust::char::CharInputs;
use ofgpu::dust::courant::CO_F_MAX;
use ofgpu::dust::fsc::{A_FSC, CombustionModel};
use ofgpu::dust::laminar::LaminarSpeed;
use ofgpu::dust::particle::Drying;
use ofgpu::dust::seeding::{gamma, lower_gamma, mass_weighted_mean, rosin_rammler_bands, upper_gamma};
use ofgpu::dust::vessel0d::{BurningLaw, UnburnedDensity};
use ofgpu::error::Error;
use ofgpu::io::case_chem::JsonCombustion;
use ofgpu::io::case_dust::{
    CHAR_HC_DEFAULT, CloudGas, CloudSize, DustContext, DustThermo, JsonDust, JsonDustCloud,
    LoweredCloud, LoweredDust, LoweredDustSst, decimal_fraction, dust_cloud_of_dict, is_dust_model,
    lower_cloud, lower_dust,
};
use ofgpu::io::case_json::{JsonCase, LoweredCase, read_case_jsonc, read_case_jsonc_str};
use ofgpu::io::case_parcels::parcels_of_dict;
use ofgpu::io::dict::FoamDict;

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
    // 14. the rho-weighted SST carries no buoyancy production; 15. a dust
    // case may state the mixture gas, and both blocks lower (SPEC-LIT
    // §158.6): `ofgpu-compressible` runs the pair.
    let compressible = r#", "compressible": { "gamma": 1.4, "W": 0.0289647,
        "cp": 1004.692579691832, "viscosity": { "model": "sutherland", "As": 1.458e-6, "Ts": 110.4 },
        "pOperating": 1e5 }"#;
    let sst_dust = r#", "dust": { "thermo": "formula", "phi": 0.62, "SL": 0.25, "sst": { "limitOmegaProduction": true } }"#;
    let grav_case = sst_case_text("[0, 9.81, 0]", "", sst_dust);
    refuses(lowers(&grav_case), "physics.gravity");
    lowers(&sst_case_text("[0,0,0]", "", sst_dust))
        .expect("the rho-weighted SST case with no gravity lowers");
    let with_gas = lowers(&sst_case_text("[0,0,0]", &compressible, sst_dust))
        .expect("a dust case with a gas block lowers (SPEC-LIT §158.6)");
    assert!(with_gas.dust.is_some(), "the dust block lowers beside the gas");
    assert!(with_gas.compressible.is_some(), "the gas block lowers beside the dust");
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
        thermo: Some("prescribed".to_string()),
        phi: None,
        cp: Some(1000.0),
        w_u: Some(0.029),
        w_b: Some(0.029),
        t_b: Some(2100.0),
        s_l: Some(0.5),
        sst: None,
        cloud: None,
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
        flame_age: None,
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

// ==========================================================================
//  DST-14: the reacting dust cloud (SPEC-LIT §159.9)
// ==========================================================================

/// The every-key cloud, non-default throughout (SPEC-LIT §159.9's gate
/// `cloud_lowers`): Rosin-Rammler at K = 4, every stage fraction stated,
/// the SFOM rate and the char reaction with an explicit h_c.
const CLOUD: &str = r#", "dust": { "cloud": {
    "concentration": 0.5,
    "rosinRammler": { "d": 5e-5, "n": 2.5 },
    "parcelsPerCell": 4,
    "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 },
    "rhoP": 1500, "cpP": 1800, "emissivity": 0.9,
    "devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 },
    "char": { "C0": 5e-12, "A": 2e-3, "Ea": 7.9e4, "hs": 0.3, "hc": 3e7 }
} }"#;

/// The four-stage composition the seeding tests share.
const STAGES: &str =
    r#""composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 }"#;

/// The SFOM rate those tests' clouds state beside it.
const DEVOL: &str = r#""devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 }"#;

/// The JsonDustCloud a cloud JSON text lowers to.
fn cloud_lowered(json: &str, t: f64) -> Result<LoweredCloud, Error> {
    let jc: JsonDustCloud =
        serde_json::from_str(json).map_err(|e| Error::Config(format!("the cloud JSON: {e}")))?;
    jc.lower(t)
}

/// The 1000-cell mesh of the mass gates: V_i = 1e-6 (1 + 0.37 sin i),
/// centroids on the x axis.
fn mesh_1000() -> (Vec<DVec3>, Vec<f64>) {
    let centroids = (0..1000)
        .map(|i| DVec3::new(i as f64 * 1e-2, 0.0, 0.0))
        .collect();
    let volumes = (0..1000)
        .map(|i| 1e-6 * (1.0 + 0.37 * (i as f64).sin()))
        .collect();
    (centroids, volumes)
}

/// The cloud's `JsonDust` wrapper, for `lower_cloud` on a built cloud.
fn cloud_dust(cloud: &JsonDustCloud) -> JsonDust {
    JsonDust {
        thermo: None,
        phi: None,
        cp: None,
        w_u: None,
        w_b: None,
        t_b: None,
        s_l: None,
        sst: None,
        cloud: Some(cloud.clone()),
    }
}

/// **SPEC-LIT §159.9 `cloud_lowers` (D).** The every-key cloud lowers to
/// exactly those values, `particle_inputs` maps each onto `ParticleInputs`,
/// the defaults are K = 1, inert char and hc = 32.9 MJ/kg, and the
/// dictionary route gives the same `JsonDustCloud` and no `parcels` block.
#[test]
fn cloud_lowers() -> Result<(), Error> {
    let c = lowers(&dust_case("300", "", CLOUD))?
        .cloud
        .expect("the case states a cloud");
    assert_eq!(c.concentration, 0.5);
    assert_eq!(c.size, CloudSize::RosinRammler { d_bar: 5e-5, n: 2.5 });
    assert_eq!(c.parcels_per_cell, 4);
    let sum = 0.1 + 0.6 + 0.25 + 0.05;
    let close = |got: f64, want: f64| (got - want).abs() <= want.abs() * 1e-15;
    assert!(close(c.composition.moisture, 0.1 / sum), "{:?}", c.composition);
    assert!(close(c.composition.volatile, 0.6 / sum), "{:?}", c.composition);
    assert!(close(c.composition.char, 0.25 / sum), "{:?}", c.composition);
    assert!(close(c.composition.ash, 0.05 / sum), "{:?}", c.composition);
    assert_eq!(c.rho_p, 1500.0);
    assert_eq!(c.cp_p, 1800.0);
    assert_eq!(c.emissivity, 0.9);
    let dv = c.devol.as_ref().expect("the case states the SFOM rate");
    assert_eq!((dv.a, dv.e_a, dv.dh), (1.5e5, 7.4e4, 1e5));
    assert_eq!(
        c.char_rate,
        CharInputs { c0: 5e-12, a: 2e-3, e_a: 7.9e4, h_c: 3e7, h_s: 0.3 }
    );
    assert_eq!(c.t_init, 300.0);
    assert_eq!(c.diameters, rosin_rammler_bands(5e-5, 2.5, 4)?);
    let gas = CloudGas {
        mu: 1.8e-5,
        k_g: 0.026,
        cp_g: 1005.0,
        cp_volatile: 1500.0,
        p_ambient: 101325.0,
    };
    let p = c.particle_inputs(&gas)?;
    assert_eq!(p.cp, 1800.0);
    assert_eq!(p.emissivity, 0.9);
    assert_eq!((p.devol_a, p.devol_e_a, p.devol_dh), (1.5e5, 7.4e4, 1e5));
    assert_eq!(p.cp_volatile, 1500.0);
    assert_eq!((p.mu, p.k_g, p.cp_g), (1.8e-5, 0.026, 1005.0));
    assert_eq!(p.char_rate, c.char_rate);
    assert_eq!(p.drying, Drying::water(101325.0)?);

    // The defaults: K 1, inert char, and hc = ISL22's 32.9 MJ/kg when the
    // char reaction states everything but it (SPEC-LIT §159.1).
    let minimal = r#", "dust": { "cloud": { "concentration": 0.3, "diameter": 4e-5,
        "composition": { "moisture": 0, "volatile": 0, "char": 0.9, "ash": 0.1 },
        "rhoP": 1400, "cpP": 1500, "emissivity": 1 } }"#;
    let m = lowers(&dust_case("300", "", minimal))?
        .cloud
        .expect("the minimal cloud lowers");
    assert_eq!(m.size, CloudSize::Monodisperse { d: 4e-5 });
    assert_eq!(m.parcels_per_cell, 1);
    assert_eq!(m.devol, None);
    assert_eq!(m.char_rate, CharInputs::inert());
    assert_eq!(m.diameters, vec![4e-5]);
    let no_hc = r#", "dust": { "cloud": { "concentration": 0.3, "diameter": 4e-5,
        "composition": { "moisture": 0, "volatile": 0, "char": 0.9, "ash": 0.1 },
        "rhoP": 1400, "cpP": 1500, "emissivity": 1,
        "char": { "C0": 1e-12, "A": 1e-3, "Ea": 8e4, "hs": 0.3 } } }"#;
    let h = lowers(&dust_case("300", "", no_hc))?.cloud.expect("lowers");
    assert_eq!(h.char_rate.h_c, CHAR_HC_DEFAULT);
    assert_eq!(CHAR_HC_DEFAULT, 32.9e6);
    // The dictionary route: `dustCloud` through the parcels' dictionary-to-
    // JSON conversion reads as the SAME JsonDustCloud, and a file holding
    // only `dustCloud` has no `parcels` block (SPEC-LIT §159.9).
    let dict_text = "FoamFile { format ascii; class dictionary; object cloudProperties; }
dustCloud { concentration 0.5; rosinRammler { d 5e-5; n 2.5; } parcelsPerCell 4;
    composition { moisture 0.1; volatile 0.6; char 0.25; ash 0.05; }
    rhoP 1500; cpP 1800; emissivity 0.9;
    devolatilisation { A 1.5e5; Ea 7.4e4; dH 1e5; }
    char { C0 5e-12; A 2e-3; Ea 7.9e4; hs 0.3; hc 3e7; } }
";
    let d = FoamDict::parse(dict_text, "cloudProperties")?;
    let from_dict = dust_cloud_of_dict(&d, "cloudProperties")?.expect("dustCloud stated");
    let from_jsonc = read_case_jsonc_str(&dust_case("300", "", CLOUD), "dustTest")?
        .dust
        .and_then(|d| d.cloud)
        .expect("the JSONC case states the cloud");
    assert_eq!(from_dict, from_jsonc);
    assert!(parcels_of_dict(&d, "cloudProperties")?.is_none());
    Ok(())
}

/// **SPEC-LIT §159.9 `cloud_mass_is_exact` (A).** On 1000 cells of unequal
/// volume, monodisperse and Rosin-Rammler at K = 1, 4 and 16: the seeded
/// mass per volume is C to 1e-12 relative, each particle's four stage
/// masses add to rho_p pi d^3/6 to 1e-14, and the handoff rounds once.
#[test]
fn cloud_mass_is_exact() -> Result<(), Error> {
    let (cents, vols) = mesh_1000();
    let total_v: f64 = vols.iter().sum();
    let monodisperse = |k: u32| {
        cloud_lowered(
            &format!(
                r#"{{ "concentration": 0.5, "diameter": 4e-5, "parcelsPerCell": {k}, {STAGES},
                    "rhoP": 1500, "cpP": 1800, "emissivity": 0.9, {DEVOL} }}"#
            ),
            300.0,
        )
    };
    let rosin = |k: u32| {
        cloud_lowered(
            &format!(
                r#"{{ "concentration": 0.5, "rosinRammler": {{ "d": 5e-5, "n": 2.5 }},
                    "parcelsPerCell": {k}, {STAGES},
                    "rhoP": 1500, "cpP": 1800, "emissivity": 0.9, {DEVOL} }}"#
            ),
            300.0,
        )
    };
    for c in [monodisperse(1)?, monodisperse(4)?, rosin(1)?, rosin(4)?, rosin(16)?] {
        let parcels = c.seed(&cents, &vols)?;
        assert_eq!(parcels.len(), 1000 * c.parcels_per_cell as usize);
        let mut seeded = 0.0;
        for p in &parcels {
            let m = c.rho_p * std::f64::consts::PI * p.particle.d * p.particle.d * p.particle.d
                / 6.0;
            let stages = p.particle.m_w + p.particle.m_v + p.particle.m_c + p.particle.m_a;
            assert!(
                (stages - m).abs() <= 1e-14 * m,
                "stage masses {stages} vs particle mass {m}"
            );
            seeded += p.n_p * m;
        }
        let rel = (seeded / total_v - c.concentration).abs() / c.concentration;
        assert!(rel <= 1e-12, "seeded C is off by {rel:e} relative");
        let sp = parcels[0].seed_parcel();
        assert_eq!(sp.diameter, parcels[0].particle.d as ofgpu::Scalar);
        assert_eq!(sp.n_p, parcels[0].n_p as ofgpu::Scalar);
        assert_eq!(sp.velocity, ofgpu::Vec3::ZERO);
        assert_eq!(sp.uid, None);
    }
    Ok(())
}

/// **SPEC-LIT §159.9 `rosin_rammler_moments` (A).** Gamma(1 + 1/n) against
/// its known values, gamma(a, x) against the closed forms on both sides of
/// a + 1, the K = 7 band diameters against the supervisor's oracle, and the
/// seeded set's mass-weighted mean diameter against (159.13) - one printed
/// line per (n, K).
#[test]
fn rosin_rammler_moments() -> Result<(), Error> {
    let rel = |got: f64, want: f64| (got - want).abs() / want.abs();
    for (n, want) in [
        (1.0, 1.0),
        (2.0, std::f64::consts::PI.sqrt() / 2.0),
        (3.0, 0.892979511569249211),
        (4.0, 0.906402477055477078),
        (2.5, 0.887263817503075289),
    ] {
        let got = gamma(1.0 + 1.0 / n);
        assert!(rel(got, want) <= 1e-14, "gamma(1+1/{n}) = {got} vs {want}");
    }
    for a in [2.0, 3.0] {
        let g = gamma(a);
        for x in [0.1, 1.0, 2.9, 3.1, 3.9, 4.1, 10.0, 40.0] {
            let got = if x < a + 1.0 {
                lower_gamma(a, x)
            } else {
                g - upper_gamma(a, x)
            };
            let want = if a == 2.0 {
                1.0 - (-x).exp() * (1.0 + x)
            } else {
                2.0 - (-x).exp() * (x * x + 2.0 * x + 2.0)
            };
            assert!((got - want).abs() <= 1e-14 * g, "gamma({a},{x}) = {got} vs {want}");
        }
    }
    // The supervisor oracle's K = 7 bands, first and last of each.
    let b1 = rosin_rammler_bands(5e-5, 1.0, 7)?;
    assert!(rel(b1[0], 3.7547960518225077e-06) <= 1e-12, "{}", b1[0]);
    assert!(rel(b1[6], 1.4729550745276577e-04) <= 1e-12, "{}", b1[6]);
    let b2 = rosin_rammler_bands(5e-5, 2.0, 7)?;
    assert!(rel(b2[0], 1.2884936209065723e-05) <= 1e-12, "{}", b2[0]);
    assert!(rel(b2[6], 8.4798407702767e-05) <= 1e-12, "{}", b2[6]);
    for bands in [&b1, &b2] {
        for w in bands.windows(2) {
            assert!(w[0] < w[1], "the bands must strictly increase: {bands:?}");
        }
    }
    // (159.13) on the seeded set: the mass-weighted mean diameter is
    // d_bar Gamma(1 + 1/n) for every n at K = 1, 7 and 64.
    // Against the KNOWN Gamma(1 + 1/n), not this file's own `gamma`.
    let (cents, vols) = mesh_1000();
    for (n, g_known) in [
        (1.0, 1.0),
        (2.0, std::f64::consts::PI.sqrt() / 2.0),
        (3.0, 0.892979511569249211),
        (4.0, 0.906402477055477078),
        (2.5, 0.887263817503075289),
    ] {
        let want = 5e-5 * g_known;
        for k in [1u32, 7, 64] {
            let c = cloud_lowered(
                &format!(
                    r#"{{ "concentration": 0.5, "rosinRammler": {{ "d": 5e-5, "n": {n} }},
                        "parcelsPerCell": {k}, {STAGES},
                        "rhoP": 1500, "cpP": 1800, "emissivity": 0.9, {DEVOL} }}"#
                ),
                300.0,
            )?;
            let parcels = c.seed(&cents, &vols)?;
            let masses: Vec<f64> = parcels
                .iter()
                .map(|p| {
                    let d = p.particle.d;
                    p.n_p * c.rho_p * std::f64::consts::PI * d * d * d / 6.0
                })
                .collect();
            let ds: Vec<f64> = parcels.iter().map(|p| p.particle.d).collect();
            let got = mass_weighted_mean(&masses, &ds);
            let err = (got - want).abs() / want;
            println!("n = {n}, K = {k}: relative error {err:.3e}");
            assert!(err <= 1e-10, "n = {n}, K = {k}: mean {got} vs {want} rel {err:e}");
        }
    }
    Ok(())
}

/// **SPEC-LIT §159.9 `cloud_refusals` (D).** Every refusal of the block -
/// value ranges, both/neither size, the presence rules, each Phase A key
/// beside a cloud, `initial.T`, the FSC/TFC seam, and the seeding's own -
/// names its key; an unknown key is named by the deserialiser; and a cloud
/// case is refused through `refuse_parcels`.
#[test]
fn cloud_refusals() {
    let cloud_case = |cloud: &str| {
        lowers(&dust_case("300", "", &format!(r#", "dust": {{ "cloud": {cloud} }}"#)))
    };
    for (cloud, key) in [
        (r#"{ "concentration": 0, "diameter": 4e-5, "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 }, "rhoP": 1500, "cpP": 1800, "emissivity": 0.9, "devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 } }"#, "dust.cloud.concentration"),
        (r#"{ "concentration": 0.5, "diameter": 4e-5, "rosinRammler": { "d": 5e-5, "n": 2.5 }, "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 }, "rhoP": 1500, "cpP": 1800, "emissivity": 0.9, "devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 } }"#, "dust.cloud.diameter"),
        (r#"{ "concentration": 0.5, "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 }, "rhoP": 1500, "cpP": 1800, "emissivity": 0.9, "devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 } }"#, "dust.cloud.diameter"),
        (r#"{ "concentration": 0.5, "rosinRammler": { "d": 0, "n": 2.5 }, "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 }, "rhoP": 1500, "cpP": 1800, "emissivity": 0.9, "devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 } }"#, "dust.cloud.rosinRammler.d"),
        (r#"{ "concentration": 0.5, "rosinRammler": { "d": 5e-5, "n": -1 }, "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 }, "rhoP": 1500, "cpP": 1800, "emissivity": 0.9, "devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 } }"#, "dust.cloud.rosinRammler.n"),
        (r#"{ "concentration": 0.5, "diameter": 4e-5, "parcelsPerCell": 0, "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 }, "rhoP": 1500, "cpP": 1800, "emissivity": 0.9, "devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 } }"#, "dust.cloud.parcelsPerCell"),
        (r#"{ "concentration": 0.5, "diameter": 4e-5, "composition": { "moisture": 1.5, "volatile": 0.6, "char": 0.25, "ash": 0.05 }, "rhoP": 1500, "cpP": 1800, "emissivity": 0.9, "devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 } }"#, "dust.cloud.composition"),
        (r#"{ "concentration": 0.5, "diameter": 4e-5, "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.0 }, "rhoP": 1500, "cpP": 1800, "emissivity": 0.9, "devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 } }"#, "dust.cloud.composition"),
        (r#"{ "concentration": 0.5, "diameter": 4e-5, "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 }, "rhoP": 0, "cpP": 1800, "emissivity": 0.9, "devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 } }"#, "dust.cloud.rhoP"),
        (r#"{ "concentration": 0.5, "diameter": 4e-5, "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 }, "rhoP": 1500, "cpP": -1, "emissivity": 0.9, "devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 } }"#, "dust.cloud.cpP"),
    ] {
        refuses(cloud_case(cloud), key);
    }
    for (cloud, key) in [
        // emissivity outside [0, 1].
        (r#"{ "concentration": 0.5, "diameter": 4e-5, "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 }, "rhoP": 1500, "cpP": 1800, "emissivity": 1.5, "devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 } }"#, "dust.cloud.emissivity"),
        // devolatilisation is required with a volatile and refused without one.
        (r#"{ "concentration": 0.5, "diameter": 4e-5, "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 }, "rhoP": 1500, "cpP": 1800, "emissivity": 0.9 }"#, "dust.cloud.devolatilisation"),
        (r#"{ "concentration": 0.5, "diameter": 4e-5, "composition": { "moisture": 0.1, "volatile": 0, "char": 0.85, "ash": 0.05 }, "rhoP": 1500, "cpP": 1800, "emissivity": 0.9, "devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 } }"#, "dust.cloud.devolatilisation"),
        (r#"{ "concentration": 0.5, "diameter": 4e-5, "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 }, "rhoP": 1500, "cpP": 1800, "emissivity": 0.9, "devolatilisation": { "A": -1, "Ea": 7.4e4, "dH": 1e5 } }"#, "dust.cloud.devolatilisation.A"),
        (r#"{ "concentration": 0.5, "diameter": 4e-5, "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 }, "rhoP": 1500, "cpP": 1800, "emissivity": 0.9, "devolatilisation": { "A": 1.5e5, "Ea": -1, "dH": 1e5 } }"#, "dust.cloud.devolatilisation.Ea"),
        // The char reaction only beside a char, and each of its values.
        (r#"{ "concentration": 0.5, "diameter": 4e-5, "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0, "ash": 0.3 }, "rhoP": 1500, "cpP": 1800, "emissivity": 0.9, "devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 }, "char": { "C0": 5e-12, "A": 2e-3, "Ea": 7.9e4, "hs": 0.3 } }"#, "dust.cloud.char"),
        (r#"{ "concentration": 0.5, "diameter": 4e-5, "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 }, "rhoP": 1500, "cpP": 1800, "emissivity": 0.9, "devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 }, "char": { "C0": -1, "A": 2e-3, "Ea": 7.9e4, "hs": 0.3 } }"#, "dust.cloud.char.C0"),
        (r#"{ "concentration": 0.5, "diameter": 4e-5, "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 }, "rhoP": 1500, "cpP": 1800, "emissivity": 0.9, "devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 }, "char": { "C0": 5e-12, "A": -1, "Ea": 7.9e4, "hs": 0.3 } }"#, "dust.cloud.char.A"),
        (r#"{ "concentration": 0.5, "diameter": 4e-5, "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 }, "rhoP": 1500, "cpP": 1800, "emissivity": 0.9, "devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 }, "char": { "C0": 5e-12, "A": 2e-3, "Ea": -1, "hs": 0.3 } }"#, "dust.cloud.char.Ea"),
        (r#"{ "concentration": 0.5, "diameter": 4e-5, "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 }, "rhoP": 1500, "cpP": 1800, "emissivity": 0.9, "devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 }, "char": { "C0": 5e-12, "A": 2e-3, "Ea": 7.9e4, "hs": 1.5 } }"#, "dust.cloud.char.hs"),
        (r#"{ "concentration": 0.5, "diameter": 4e-5, "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 }, "rhoP": 1500, "cpP": 1800, "emissivity": 0.9, "devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 }, "char": { "C0": 5e-12, "A": 2e-3, "Ea": 7.9e4, "hs": 0.3, "hc": -1 } }"#, "dust.cloud.char.hc"),
    ] {
        refuses(cloud_case(cloud), key);
    }
    // dH and hc not finite cannot ride through JSON; lowered directly off
    // the every-key cloud.
    let every: JsonDustCloud = read_case_jsonc_str(&dust_case("300", "", CLOUD), "dustTest")
        .expect("the every-key case parses")
        .dust
        .and_then(|d| d.cloud)
        .expect("the every-key case states the cloud");
    let mut no_dh = every.clone();
    no_dh.devolatilisation.as_mut().unwrap().dh = f64::NAN;
    refuses(no_dh.lower(300.0), "dust.cloud.devolatilisation.dH");
    let mut no_hc = every.clone();
    no_hc.char_burn.as_mut().unwrap().hc = Some(f64::NAN);
    refuses(no_hc.lower(300.0), "dust.cloud.char.hc");
    // Each Phase A key beside a cloud, whatever the combustion block says.
    let beside = |phase_a: &str| {
        lowers(&dust_case(
            "300",
            "",
            &format!(
                r#", "dust": {{ {phase_a} "cloud": {{ "concentration": 0.5, "diameter": 4e-5,
                    "composition": {{ "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 }},
                    "rhoP": 1500, "cpP": 1800, "emissivity": 0.9 }} }}"#
            ),
        ))
    };
    refuses(beside(r#""thermo": "formula", "phi": 0.62, "#), "dust.thermo");
    refuses(beside(r#""phi": 0.62, "#), "dust.phi");
    refuses(beside(r#""cp": 1000, "#), "dust.cp");
    refuses(beside(r#""Wu": 0.0328, "#), "dust.Wu");
    refuses(beside(r#""Wb": 0.0297, "#), "dust.Wb");
    refuses(beside(r#""Tb": 1500, "#), "dust.Tb");
    refuses(beside(r#""SL": 0.25, "#), "dust.SL");
    refuses(beside(r#""sst": { "limitOmegaProduction": true }, "#), "dust.sst");
    // `initial.T` is where the particles start: absent, refused.
    let cx_names = vec!["vessel".to_string()];
    let cx = DustContext {
        species: false,
        chemistry: false,
        turbulence_kind: None,
        turbulence_model: None,
        compressible: false,
        gravity: [0.0, 0.0, 0.0],
        initial_t: None,
        patch_names: &cx_names,
        rules: &[],
    };
    refuses(lower_cloud(None, Some(&cloud_dust(&every)), &cx), "initial.T");
    // A cloud beside the dust model is refused by its own key: Phase B
    // burns the volatile on the finite-rate path.
    refuses(lowers(&dust_case("300", FSC_MIN, CLOUD)), "dust.cloud");
    refuses(lowers(&dust_case("300", TFC_CO, CLOUD)), "dust.cloud");
    // The seeding's own refusals: unequal lengths and a zero volume.
    let c = cloud_lowered(
        r#"{ "concentration": 0.5, "diameter": 4e-5, "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 }, "rhoP": 1500, "cpP": 1800, "emissivity": 0.9, "devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 } }"#,
        300.0,
    )
    .expect("the cloud lowers");
    refuses(c.seed(&[DVec3::new(1.0, 0.0, 0.0)], &[]), "dust.cloud");
    refuses(
        c.seed(&[DVec3::new(1.0, 0.0, 0.0), DVec3::new(2.0, 0.0, 0.0)], &[1e-6, 0.0]),
        "dust.cloud",
    );
    // An unknown key is named, inside the cloud and inside its composition.
    for text in [
        r#", "dust": { "cloud": { "bogusKey": 1, "concentration": 0.5, "diameter": 4e-5,
            "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 },
            "rhoP": 1500, "cpP": 1800, "emissivity": 0.9 } }"#,
        r#", "dust": { "cloud": { "concentration": 0.5, "diameter": 4e-5,
            "composition": { "bogusKey": 1, "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 },
            "rhoP": 1500, "cpP": 1800, "emissivity": 0.9 } }"#,
    ] {
        let e = read_case_jsonc_str(&dust_case("300", "", text), "dustTest")
            .expect_err("an unknown key is refused")
            .to_string();
        assert!(e.contains("bogusKey"), "{e}");
    }
    // A cloud case lowers, and ofgpu-lowmach is the one driver that runs
    // Phase B (SPEC-LIT §159.10).
    let l = lowers(&dust_case("300", "", CLOUD)).expect("the cloud case lowers");
    let e = l.refuse_parcels().expect_err("the cloud case is refused");
    assert!(e.to_string().contains("dust.cloud"), "{e}");
}

/// **SPEC-LIT 159.12 `cloud_igniter_lowers_and_refuses` (D).** The
/// igniter lowers to its four values exactly; each bad value - radius,
/// power or duration not a finite number above zero, a point component
/// not finite - is refused by name, and an unknown key inside the
/// igniter is named by the deserialiser.
#[test]
fn cloud_igniter_lowers_and_refuses() {
    let base = r#""concentration": 0.5, "diameter": 4e-5,
        "composition": { "moisture": 0.1, "volatile": 0.6, "char": 0.25, "ash": 0.05 },
        "rhoP": 1500, "cpP": 1800, "emissivity": 0.9,
        "devolatilisation": { "A": 1.5e5, "Ea": 7.4e4, "dH": 1e5 }"#;
    let with = |igniter: &str| {
        cloud_lowered(&format!("{{{base}, \"igniter\": {igniter}}}"), 300.0)
    };
    // One case lowers with the four values exactly.
    let l = with(
        r#"{ "point": [0.001, 0.002, 0.003], "radius": 0.0021, "power": 1e9,
            "duration": 4.5e-5 }"#,
    )
    .expect("the cloud with its igniter lowers");
    let ig = l.igniter.as_ref().expect("the igniter lowered");
    assert_eq!(ig.point, [0.001, 0.002, 0.003]);
    assert_eq!(ig.radius, 0.0021);
    assert_eq!(ig.power, 1e9);
    assert_eq!(ig.duration, 4.5e-5);
    // A cloud without an igniter lowers to None.
    assert!(cloud_lowered(&format!("{{{base}}}"), 300.0)
        .expect("the bare cloud lowers")
        .igniter
        .is_none());
    // Each value refusal names its key.
    let refuses_igniter = |r: Result<LoweredCloud, Error>, key: &str| {
        let e = r.expect_err(&format!("the igniter is refused, naming {key}"));
        assert!(e.to_string().contains(key), "the refusal does not name {key}: {e}");
    };
    for (igniter, key) in [
        (
            r#"{ "point": [0, 0, 0], "radius": 0, "power": 1e9, "duration": 4.5e-5 }"#,
            "dust.cloud.igniter.radius",
        ),
        (
            r#"{ "point": [0, 0, 0], "radius": 0.0021, "power": -1, "duration": 4.5e-5 }"#,
            "dust.cloud.igniter.power",
        ),
        (
            r#"{ "point": [0, 0, 0], "radius": 0.0021, "power": 1e9, "duration": 0 }"#,
            "dust.cloud.igniter.duration",
        ),
    ] {
        refuses_igniter(with(igniter), key);
    }
    // A point component that is not finite cannot ride through JSON, so it
    // lowers directly off the parsed cloud.
    let mut nan_point: JsonDustCloud =
        serde_json::from_str(&format!("{{{base}}}")).expect("the bare cloud parses");
    nan_point.igniter = Some(ofgpu::io::case_dust::JsonCloudIgniter {
        point: [f64::NAN, 0.0, 0.0],
        radius: 0.0021,
        power: 1e9,
        duration: 4.5e-5,
    });
    refuses_igniter(nan_point.lower(300.0), "dust.cloud.igniter.point");
    // An unknown key inside the igniter is named by the deserialiser.
    let e = serde_json::from_str::<JsonDustCloud>(&format!(
        "{{{base}, \"igniter\": {{ \"point\": [0, 0, 0], \"radius\": 0.0021, \
         \"power\": 1e9, \"duration\": 4.5e-5, \"energy\": 1 }} }}"
    ))
    .expect_err("the unknown key is refused")
    .to_string();
    assert!(e.contains("energy"), "{e}");
}

// ==========================================================================
//  DST-20: the flame-age clock (SPEC-LIT §160.5)
// ==========================================================================

/// **SPEC-LIT §160.5.** `flameAge` lowers onto the clock of the closures:
/// absent is `Global` (the papers' clock of (154.2)), `"local"` is the
/// transported clock of §160.5, and `"global"` states the default by name.
#[test]
fn flame_age_lowers() {
    let d = dust_of(&dust_case("273", FSC_MIN, FORMULA_MIN)).expect("an FSC case lowers");
    assert_eq!(d.flame_age, FlameAge::Global);
    let local = r#", "combustion": { "model": "FSC", "PrT": 0.3, "flameAge": "local" }"#;
    let d = dust_of(&dust_case("273", local, FORMULA_MIN)).expect("the local case lowers");
    assert_eq!(d.flame_age, FlameAge::Local);
    let global = r#", "combustion": { "model": "FSC", "PrT": 0.3, "flameAge": "global" }"#;
    let d = dust_of(&dust_case("273", global, FORMULA_MIN)).expect("the global case lowers");
    assert_eq!(d.flame_age, FlameAge::Global);
}

/// **SPEC-LIT §160.5.** Every refusal names its key: ANY stated `flameAge`
/// beside TFC (which has no clock), a value that is neither `global` nor
/// `local`, and `flameAge` under a non-dust model (the chemistry side's
/// own table, which needs the case's species set to reach).
#[test]
fn flame_age_refusals() {
    for v in ["local", "global"] {
        let tfc = format!(
            r#", "combustion": {{ "model": "TFC", "PrT": 0.3, "flameAge": "{v}" }}"#
        );
        refuses(lowers(&dust_case("273", &tfc, FORMULA_MIN)), "combustion.flameAge");
    }
    let both = r#", "combustion": { "model": "FSC", "PrT": 0.3, "flameAge": "both" }"#;
    refuses(lowers(&dust_case("273", both, FORMULA_MIN)), "combustion.flameAge");
    let edm = r#", "combustion": { "model": "EDM", "A": 4.0, "flameAge": "local" }"#;
    refuses(
        lowers(&dust_case("273", edm, &format!("{SPECIES}{CHEMISTRY}"))),
        "combustion.flameAge",
    );
}
