// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
//
// Provenance: ORIGINAL - the tests for SPEC-LIT S55.6's case-level pair
// tests and for the reader's own S13.4 refusals. Every pair below is TWO
// DOCUMENTS DIFFERING IN ONE ENTRY, required to lower to different numbers
// and failing by name if they do not.
// No GPL-licensed source was consulted.

use super::*;

/// The smallest complete case this format accepts, as a template. Every pair
/// test below is this document with one entry replaced.
///
/// Deliberately spelled out rather than built through the structs: the point
/// of a §13.4.1 pair test is that two **case documents** differing in one
/// entry produce different output, and building the structs directly would
/// skip exactly the layer that can drop a setting on the floor.
const BASE: &str = r#"{
  "name": "pair",
  "room": {
    "bounds": { "min": [0,0,0], "max": [2.0, 1.0, 1.5] },
    "cells": [8, 4, 6],
    "boundaries": {
      "xMin": "west", "xMax": "east",
      "yMin": "south", "yMax": "north",
      "zMin": "supply", "zMax": "ret"
    }
  },
  "air": { "nu": 1.5e-5, "rho": 1.2, "cp": 1005, "pr": 0.71, "prt": 0.85,
           "tRef": 295.15, "gravity": [0,0,-9.81] },
  "fans": [
    { "patch": "ret", "direction": "outflow",
      "curve": { "type": "quadratic", "dpMax": 8.0, "QMax": 1.0,
                 "rhoCurve": 1.2, "speedCurve": 1.0, "speed": 1.0,
                 "efficiency": 0.62 },
      "ambientPressure": 0.0, "relaxation": 0.5 }
  ],
  "tiles": [
    { "patch": "supply", "K": 300.0, "plenumPressure": 4.0,
      "plenumTemperature": 291.15, "plenumRelativeHumidity": 0.45 }
  ],
  "racks": [
    { "name": "r1", "zone": { "min": [0.8,0.2,0.1], "max": [1.2,0.8,1.0] },
      "power": 2000.0, "flow": 0.15,
      "inletSamples": { "min": [0.5,0.2,0.1], "max": [0.8,0.8,1.0] } }
  ],
  "patches": [
    { "patch": "west", "kind": "adiabaticWall" },
    { "patch": "east", "kind": "adiabaticWall" },
    { "patch": "south", "kind": "adiabaticWall" },
    { "patch": "north", "kind": "adiabaticWall" }
  ],
  "humidity": { "d": 2.5e-5, "scT": 0.7, "barometricPressure": 101325.0,
                "virtualTemperature": true },
  "metrics": { "ashraeClass": "A1", "rciSamples": "thirds",
               "supplyPatch": "supply", "returnPatch": "ret" },
  "run": { "iterations": 20, "reportEvery": 0, "initialTemperature": 295.15 },
  "numerics": { "uRelax": 0.7, "pRelax": 0.3, "tRelax": 0.7,
                "tolerance": 1e-8, "maxIterations": 200 }
}"#;

fn base() -> DcCase {
    DcCase::parse(BASE, "base").expect("the base case must parse")
}

/// The base document with one substring replaced - "two cases identical in
/// every byte but one".
fn variant(from: &str, to: &str) -> DcCase {
    assert!(BASE.contains(from), "the base case does not contain `{from}`");
    let text = BASE.replacen(from, to, 1);
    assert_ne!(text, BASE, "the variant is byte-identical to the base");
    DcCase::parse(&text, "variant").expect("the variant must parse")
}

fn rel(a: Scalar, b: Scalar) -> Scalar {
    let s = a.abs().max(b.abs()).max(1e-300);
    (a - b).abs() / s
}

#[test]
fn the_base_case_lowers() {
    let l = base().lower().expect("lower");
    assert_eq!(l.name, "pair");
    assert_eq!(l.mesh.n_cells, 8 * 4 * 6);
    assert_eq!(l.fans.len(), 1);
    assert_eq!(l.jumps.len(), 1);
    assert_eq!(l.racks.len(), 1);
    assert_eq!(l.class, AshraeClass::A1);
    assert_eq!(l.samples, RciSamples::Thirds);
    // Every conversion the reader performed is reported, not silent.
    assert!(l.notes.iter().any(|n| n.contains("Y_v =")), "{:?}", l.notes);
    assert!(l.notes.iter().any(|n| n.contains("kinematic")), "{:?}", l.notes);
    assert!(l.notes.iter().any(|n| n.contains("IDEAL-gas")), "{:?}", l.notes);
    // The shipped case must lower too - a case in `cases/` that does not is a
    // broken example, and an example is documentation.
    let shipped = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../cases/coldAisle.dc.jsonc");
    let c = DcCase::read(&shipped).expect("cases/coldAisle.dc.jsonc must parse");
    c.lower().expect("cases/coldAisle.dc.jsonc must lower");
}

// ==========================================================================
//  §55.6 - the case-level pair tests
//
//  Each is two documents differing in ONE entry, required to lower to
//  different numbers and failing by name if they do not.
// ==========================================================================

macro_rules! pair {
    ($name:ident, $from:expr, $to:expr, $pick:expr, $what:expr) => {
        #[test]
        fn $name() {
            let a = base().lower().expect("base");
            let b = variant($from, $to).lower().expect("variant");
            let f: &dyn Fn(&LoweredDcCase) -> Scalar = &$pick;
            let (x, y) = (f(&a), f(&b));
            assert!(
                rel(x, y) > 1e-9,
                "SPEC-LIT S13.4.1: two case documents differing only in {} both \
                 lowered to {} - the reader DROPPED the setting",
                $what,
                x
            );
        }
    };
}

pair!(
    pair_test_case_fan_dp_max,
    "\"dpMax\": 8.0",
    "\"dpMax\": 14.0",
    |l: &LoweredDcCase| l.fans[0].curve.dp_max,
    "the fan curve's dpMax"
);
pair!(
    pair_test_case_fan_q_max,
    "\"QMax\": 1.0",
    "\"QMax\": 1.6",
    |l: &LoweredDcCase| l.fans[0].curve.q_max,
    "the fan curve's QMax"
);
pair!(
    pair_test_case_ambient_pressure,
    "\"ambientPressure\": 0.0",
    "\"ambientPressure\": 30.0",
    |l: &LoweredDcCase| l.fans[0].ambient,
    "ambientPressure"
);
pair!(
    pair_test_case_fan_relaxation,
    "\"relaxation\": 0.5",
    "\"relaxation\": 0.9",
    |l: &LoweredDcCase| l.fans[0].relaxation,
    "the fan's relaxation"
);
pair!(
    pair_test_case_rho_curve,
    "\"rhoCurve\": 1.2",
    "\"rhoCurve\": 0.9",
    |l: &LoweredDcCase| l.fans[0].curve.rho_ratio(),
    "rhoCurve, the (S52.13) density correction"
);
pair!(
    pair_test_case_speed,
    "\"speed\": 1.0",
    "\"speed\": 1.4",
    |l: &LoweredDcCase| l.fans[0].curve.speed_ratio(),
    "the shaft speed, the (S52.13) affinity correction"
);
pair!(
    pair_test_case_efficiency,
    "\"efficiency\": 0.62",
    "\"efficiency\": 0.41",
    |l: &LoweredDcCase| l.fans[0].curve.efficiency,
    "efficiency, which divides the reported shaft power"
);
pair!(
    pair_test_case_tile_k,
    "\"K\": 300.0",
    "\"K\": 900.0",
    |l: &LoweredDcCase| match &l.jumps[0] {
        PorousJump::Boundary { coeffs, .. } => coeffs.r_inert,
        _ => unreachable!(),
    },
    "the tile's loss coefficient K"
);
pair!(
    pair_test_case_plenum_pressure,
    "\"plenumPressure\": 4.0",
    "\"plenumPressure\": 9.0",
    |l: &LoweredDcCase| match &l.jumps[0] {
        PorousJump::Boundary { plenum, .. } => *plenum,
        _ => unreachable!(),
    },
    "plenumPressure"
);
pair!(
    pair_test_case_plenum_temperature,
    "\"plenumTemperature\": 291.15",
    "\"plenumTemperature\": 289.15",
    |l: &LoweredDcCase| l.inflow_temperature["supply"],
    "plenumTemperature"
);
pair!(
    pair_test_case_plenum_relative_humidity,
    "\"plenumRelativeHumidity\": 0.45",
    "\"plenumRelativeHumidity\": 0.70",
    |l: &LoweredDcCase| l.inflow_humidity["supply"],
    "plenumRelativeHumidity - the (S54.2)/(S54.4) conversion"
);
pair!(
    pair_test_case_barometric_pressure,
    "\"barometricPressure\": 101325.0",
    "\"barometricPressure\": 84000.0",
    |l: &LoweredDcCase| l.inflow_humidity["supply"],
    "barometricPressure, which (S54.2b) divides the vapour pressure by"
);
pair!(
    pair_test_case_humidity_diffusivity,
    "\"d\": 2.5e-5",
    "\"d\": 5.0e-5",
    |l: &LoweredDcCase| l.humidity.expect("humidity").d as Scalar,
    "the vapour diffusivity D_v"
);
pair!(
    pair_test_case_turbulent_schmidt,
    "\"scT\": 0.7",
    "\"scT\": 1.3",
    |l: &LoweredDcCase| l.humidity.expect("humidity").sc_t as Scalar,
    "the turbulent Schmidt number"
);
pair!(
    pair_test_case_rack_power,
    "\"power\": 2000.0",
    "\"power\": 5000.0",
    |l: &LoweredDcCase| l.racks[0].q_vol,
    "the rack's power"
);
pair!(
    pair_test_case_rack_flow,
    "\"flow\": 0.15",
    "\"flow\": 0.31",
    |l: &LoweredDcCase| l.racks[0].flow,
    "the rack's flow, which is (S55.2)'s dT_equipment denominator"
);
pair!(
    pair_test_case_reference_density,
    "\"rho\": 1.2,",
    "\"rho\": 1.05,",
    |l: &LoweredDcCase| match &l.jumps[0] {
        PorousJump::Boundary { plenum, .. } => *plenum,
        _ => unreachable!(),
    },
    "the reference density, which (S52.2) divides every pressure by"
);

/// `openAreaRatio` must give a different resistance from a directly-named
/// `K`, and (S53.6)'s conversion must be printed.
#[test]
fn pair_test_case_open_area_ratio_converts_and_is_printed() {
    let a = base().lower().expect("base");
    let b = variant("\"K\": 300.0", "\"openAreaRatio\": 0.25")
        .lower()
        .expect("variant");
    let r = |l: &LoweredDcCase| match &l.jumps[0] {
        PorousJump::Boundary { coeffs, .. } => coeffs.r_inert,
        _ => unreachable!(),
    };
    assert!(
        rel(r(&a), r(&b)) > 1e-9,
        "SPEC-LIT S13.4.1: K = 300 and openAreaRatio = 0.25 both gave {}",
        r(&a)
    );
    // (S53.6) at sigma = 0.25 is K = 30.68, and `r_inert = K/2`.
    assert!(rel(r(&b), 30.6782 / 2.0) < 1e-4, "r_inert = {}", r(&b));
    assert!(
        b.notes.iter().any(|n| n.contains("openAreaRatio 0.25 gives K = 30.6782")),
        "the conversion must be PRINTED, or a user cannot check it: {:?}",
        b.notes
    );
    // And the note carries S53.4's recorded contradiction with the design
    // note, so a reader meets it where the conversion happens.
    assert!(b.notes.iter().any(|n| n.contains("contradicted")), "{:?}", b.notes);
}

/// The ASHRAE class and the sample set change the metric, and both are
/// reported.
#[test]
fn pair_test_case_ashrae_class_and_sample_set() {
    let a = base().lower().expect("base");
    let b = variant("\"ashraeClass\": \"A1\"", "\"ashraeClass\": \"A4\"")
        .lower()
        .expect("variant");
    assert_ne!(a.class, b.class);
    assert_ne!(
        a.class.envelope(),
        b.class.envelope(),
        "SPEC-LIT S13.4.1: A1 and A4 must give different ALLOWABLE envelopes"
    );

    let c = variant("\"rciSamples\": \"thirds\"", "\"rciSamples\": \"faces\"")
        .lower()
        .expect("variant");
    assert_ne!(a.samples, c.samples);
    assert_ne!(
        a.racks[0].samples.len(),
        c.racks[0].samples.len(),
        "SPEC-LIT S55.1: `thirds` and `faces` are different sample sets and must \
         report different n; both gave {}",
        a.racks[0].samples.len()
    );
    assert_eq!(a.racks[0].samples.len(), 3, "`thirds` is three points per rack");
}

/// `virtualTemperature` on and off must reach the lowered case, because it is
/// the one setting §54.4 puts on the momentum path.
#[test]
fn pair_test_case_virtual_temperature_switch() {
    let a = base().lower().expect("base");
    let b = variant("\"virtualTemperature\": true", "\"virtualTemperature\": false")
        .lower()
        .expect("variant");
    assert!(a.humidity.expect("h").virtual_temperature);
    assert!(!b.humidity.expect("h").virtual_temperature);
    assert!(
        a.notes.iter().any(|n| n.contains("virtual temperature ON")),
        "the reader must print which way it went: {:?}",
        a.notes
    );
    assert!(b.notes.iter().any(|n| n.contains("virtual temperature OFF")));
}

/// The curve TYPE, on two documents differing in one word plus the entries
/// each type requires.
#[test]
fn pair_test_case_curve_type() {
    let a = base().lower().expect("base");
    let b = variant(
        "\"type\": \"quadratic\", \"dpMax\": 8.0, \"QMax\": 1.0,",
        "\"type\": \"table\", \"points\": [[0.0, 8.0], [0.5, 7.2], [1.0, 0.0]],",
    )
    .lower()
    .expect("variant");
    assert_ne!(a.fans[0].curve.kind, b.fans[0].curve.kind);
    // Same endpoints, different curve between them - which is the whole point
    // of offering a table.
    let (qa, _) = a.fans[0].curve.at(0.5);
    let (qb, _) = b.fans[0].curve.at(0.5);
    assert!(
        rel(qa, qb) > 1e-3,
        "a quadratic and a table through the same endpoints must differ in \
         between; both gave {qa} Pa at Q = 0.5"
    );
}

/// The `direction` word flips `sigma`, which is the sign of everything.
#[test]
fn pair_test_case_direction() {
    let a = base().lower().expect("base");
    let b = variant(
        "\"direction\": \"outflow\",",
        "\"direction\": \"inflow\", \"supplyTemperature\": 291.15,",
    )
    .lower()
    .expect("variant");
    assert_eq!(a.fans[0].direction.sigma(), 1.0);
    assert_eq!(b.fans[0].direction.sigma(), -1.0);
    assert!(
        b.inflow_temperature.contains_key("ret"),
        "an inflow fan carries a supply temperature and an outflow one may not"
    );
}

// ==========================================================================
//  The reader's own §13.4 refusals
// ==========================================================================

fn refused(from: &str, to: &str) -> String {
    let c = variant(from, to);
    match c.lower() {
        Ok(_) => panic!("`{to}` should have been refused"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn an_unnamed_patch_is_refused_listing_what_was_named_and_what_was_not() {
    let text = BASE.replacen("    { \"patch\": \"east\", \"kind\": \"adiabaticWall\" },\n", "", 1);
    let e = DcCase::parse(&text, "v")
        .expect("parses")
        .lower()
        .expect_err("an unnamed patch must be refused")
        .to_string();
    assert!(e.contains("east"), "the message must name the unnamed patch: {e}");
    assert!(e.contains("Named:"), "and list what WAS named: {e}");
    assert!(e.contains("S55.6"), "and say which rule it broke: {e}");
}

#[test]
fn a_patch_claimed_twice_is_refused_naming_both_claimants() {
    let e = refused(
        "{ \"patch\": \"west\", \"kind\": \"adiabaticWall\" }",
        "{ \"patch\": \"supply\", \"kind\": \"adiabaticWall\" }",
    );
    assert!(e.contains("supply"), "{e}");
    assert!(e.contains("whichever ran last would win"), "{e}");
}

#[test]
fn a_supply_temperature_on_an_exhaust_is_refused_by_name() {
    let e = refused(
        "\"ambientPressure\": 0.0, \"relaxation\": 0.5",
        "\"ambientPressure\": 0.0, \"relaxation\": 0.5, \"supplyTemperature\": 290.0",
    );
    assert!(e.contains("exhaust"), "{e}");
    assert!(e.contains("S13.4.1"), "the message must name the rule: {e}");
}

#[test]
fn an_inflow_fan_without_a_supply_temperature_is_refused_by_name() {
    let e = refused("\"direction\": \"outflow\"", "\"direction\": \"inflow\"");
    assert!(e.contains("supplyTemperature"), "{e}");
    assert!(e.contains("no cooling"), "the message must say what would go wrong: {e}");
}

#[test]
fn humidity_without_a_humid_boundary_is_refused_by_name() {
    let text = BASE.replacen(", \"plenumRelativeHumidity\": 0.45", "", 1);
    let e = DcCase::parse(&text, "v")
        .expect("parses")
        .lower()
        .expect_err("humidity with nothing feeding it must be refused")
        .to_string();
    assert!(e.contains("S54.6"), "{e}");
    assert!(e.contains("nothing feeding it"), "{e}");
}

#[test]
fn a_relative_humidity_without_a_humidity_block_is_refused_by_name() {
    let text = BASE.replacen(
        "  \"humidity\": { \"d\": 2.5e-5, \"scT\": 0.7, \"barometricPressure\": 101325.0,\n                \"virtualTemperature\": true },\n",
        "",
        1,
    );
    let e = DcCase::parse(&text, "v")
        .expect("parses")
        .lower()
        .expect_err("rh with nothing transporting it must be refused")
        .to_string();
    assert!(e.contains("no `humidity` block"), "{e}");
    assert!(e.contains("S13.4.1"), "{e}");
}

#[test]
fn two_resistance_parameterisations_on_one_tile_are_refused_by_name() {
    let e = refused("\"K\": 300.0", "\"K\": 300.0, \"openAreaRatio\": 0.25");
    assert!(e.contains("more than one resistance"), "{e}");
    assert!(e.contains("the one it dropped"), "{e}");
}

#[test]
fn a_tile_with_no_resistance_at_all_is_refused_by_name() {
    let e = refused("\"K\": 300.0, ", "");
    assert!(e.contains("no resistance"), "{e}");
    assert!(e.contains("openAreaRatio"), "the alternatives must be named: {e}");
}

#[test]
fn a_partial_darcy_forchheimer_triple_is_refused_by_name() {
    let e = refused("\"K\": 300.0", "\"alpha\": 1e-7, \"C2\": 20.0");
    assert!(e.contains("all three"), "{e}");
    assert!(e.contains("silently default"), "{e}");
}

#[test]
fn table_entries_under_a_quadratic_curve_are_refused_by_name() {
    let e = refused(
        "\"QMax\": 1.0,",
        "\"QMax\": 1.0, \"points\": [[0.0, 8.0], [1.0, 0.0]],",
    );
    assert!(e.contains("nothing would read"), "{e}");
    assert!(e.contains("type \"table\""), "{e}");
}

#[test]
fn a_quadratic_curve_without_its_entries_is_refused_by_name() {
    let e = refused("\"dpMax\": 8.0, \"QMax\": 1.0,", "\"dpMax\": 8.0,");
    assert!(e.contains("QMax"), "{e}");
    let e = refused("\"dpMax\": 8.0, \"QMax\": 1.0,", "\"QMax\": 1.0,");
    assert!(e.contains("dpMax"), "{e}");
}

#[test]
fn an_unknown_patch_kind_or_curve_type_is_refused_by_name() {
    let e = refused("\"kind\": \"adiabaticWall\" }", "\"kind\": \"porousJump\" }");
    assert!(e.contains("available:"), "{e}");
    assert!(e.contains("`tiles` block"), "the right place must be named: {e}");

    let e = refused("\"type\": \"quadratic\"", "\"type\": \"cubic\"");
    assert!(e.contains("constantPressure"), "{e}");
}

#[test]
fn a_wall_without_a_temperature_and_a_temperature_without_a_wall_are_refused() {
    let e = refused(
        "{ \"patch\": \"west\", \"kind\": \"adiabaticWall\" }",
        "{ \"patch\": \"west\", \"kind\": \"wall\" }",
    );
    assert!(e.contains("carries no temperature"), "{e}");
    assert!(e.contains("adiabaticWall"), "{e}");

    let e = refused(
        "{ \"patch\": \"west\", \"kind\": \"adiabaticWall\" }",
        "{ \"patch\": \"west\", \"kind\": \"adiabaticWall\", \"temperature\": 300.0 }",
    );
    assert!(e.contains("nothing would read"), "{e}");
}

#[test]
fn a_rack_whose_zone_or_samples_miss_the_mesh_is_refused_by_name() {
    let e = refused(
        "\"zone\": { \"min\": [0.8,0.2,0.1], \"max\": [1.2,0.8,1.0] }",
        "\"zone\": { \"min\": [80,20,10], \"max\": [82,21,11] }",
    );
    assert!(e.contains("no cell centres"), "{e}");
    assert!(e.contains("S13.4.1"), "{e}");

    let e = refused(
        "\"inletSamples\": { \"min\": [0.5,0.2,0.1], \"max\": [0.8,0.8,1.0] }",
        "\"inletSamples\": { \"min\": [80,20,10], \"max\": [82,21,11] }",
    );
    assert!(e.contains("inletSamples"), "{e}");
}

#[test]
fn one_patch_for_both_supply_and_return_is_refused_by_name() {
    let e = refused("\"returnPatch\": \"ret\"", "\"returnPatch\": \"supply\"");
    assert!(e.contains("exactly zero"), "the message must say why: {e}");
}

#[test]
fn zero_iterations_and_nonsense_numbers_are_refused_by_name() {
    let e = refused("\"iterations\": 20", "\"iterations\": 0");
    assert!(e.contains("initial field as an answer"), "{e}");

    let e = refused("\"uRelax\": 0.7", "\"uRelax\": 1.7");
    assert!(e.contains("outside (0, 1]"), "{e}");

    let e = refused("\"nu\": 1.5e-5", "\"nu\": 0.0");
    assert!(e.contains("air.nu"), "{e}");

    let e = refused("\"relaxation\": 0.5", "\"relaxation\": 0.0");
    assert!(e.contains("S52.14"), "{e}");

    let e = refused("\"plenumRelativeHumidity\": 0.45", "\"plenumRelativeHumidity\": 45");
    assert!(e.contains("FRACTION"), "{e}");
}

#[test]
fn an_unknown_entry_is_refused_rather_than_ignored() {
    // `deny_unknown_fields` everywhere: a case that misspells a key must not
    // have it silently defaulted.
    let text = BASE.replacen("\"relaxation\": 0.5", "\"relaxaton\": 0.5", 1);
    assert!(
        DcCase::parse(&text, "v").is_err(),
        "a misspelled key must be refused, not defaulted"
    );
}

/// §55.5: a rack is a contiguous span, and the alternative is named.
#[test]
fn a_scattered_rack_is_refused_by_name() {
    let e = crate::dcmetrics::refuse_scattered_rack("rack07")
        .unwrap_err()
        .to_string();
    assert!(e.contains("SEGMENTED"), "{e}");
    assert!(e.contains("its own patch"), "{e}");
}

/// §55.1: `thirds` really does sample at 1/6, 1/2 and 5/6 of the box height,
/// and the choice is mesh-independent where `faces` is not.
#[test]
fn the_thirds_sample_set_is_mesh_independent_where_faces_is_not() {
    let coarse = base().lower().expect("base");
    let fine = variant("\"cells\": [8, 4, 6]", "\"cells\": [16, 8, 12]")
        .lower()
        .expect("variant");
    assert_eq!(coarse.racks[0].samples.len(), 3);
    assert_eq!(
        fine.racks[0].samples.len(),
        3,
        "`thirds` must stay three points when the mesh is refined"
    );

    let coarse_f = variant("\"rciSamples\": \"thirds\"", "\"rciSamples\": \"faces\"")
        .lower()
        .expect("v");
    let fine_f = DcCase::parse(
        &BASE
            .replacen("\"rciSamples\": \"thirds\"", "\"rciSamples\": \"faces\"", 1)
            .replacen("\"cells\": [8, 4, 6]", "\"cells\": [16, 8, 12]", 1),
        "v",
    )
    .expect("parses")
    .lower()
    .expect("lowers");
    assert!(
        fine_f.racks[0].samples.len() > coarse_f.racks[0].samples.len(),
        "`faces` IS mesh-dependent - that is why S55.1 makes the sample set a \
         setting and defaults to `thirds`"
    );

    // The three points really are at increasing height.
    let z: Vec<Scalar> = coarse.racks[0]
        .samples
        .iter()
        .map(|c| coarse.mesh.c[*c as usize].z)
        .collect();
    assert!(z[0] < z[1] && z[1] < z[2], "the three thirds must span the height: {z:?}");
}

/// The rack's `q'''` is its power over the zone volume, so the total heat is
/// the power - which is what (S55.4)'s denominator is.
#[test]
fn the_rack_heat_totals_to_its_stated_power() {
    let l = base().lower().expect("base");
    let r = &l.racks[0];
    let v: Scalar = r.cells.iter().map(|c| l.mesh.v[*c as usize]).sum();
    assert!(
        rel(r.q_vol * v, r.power) < 1e-13,
        "q''' x V = {} against the stated {} W",
        r.q_vol * v,
        r.power
    );
    // And the cell list is sorted, which is what makes the gather order fixed
    // and the reduction reproducible (S55.5).
    assert!(r.cells.windows(2).all(|w| w[0] < w[1]), "the zone cells must be sorted");
    assert!(r.samples.windows(2).all(|w| w[0] < w[1]), "and so must the samples");
}

/// SPEC-LIT §53.5 and §52.9: the two things this tranche refuses to build
/// are reachable FROM A CASE, so the refusal fires. A request nobody can
/// express is not a request that was refused.
#[test]
fn baffle_insertion_and_the_capacitance_fft_path_are_refused_from_the_case() {
    crate::io::contract::set_permissive(false);

    let e = refused("\"K\": 300.0", "\"K\": 300.0, \"baffle\": true");
    assert!(e.contains("TOPOLOGY MUTATION"), "{e}");
    assert!(e.contains("mesh-generation time"), "route one must be named: {e}");
    assert!(e.contains("separate region"), "route two must be named: {e}");

    for name in ["fft", "capacitance", "woodbury"] {
        let e = refused(
            "\"uRelax\": 0.7",
            &format!("\"pressureSolver\": \"{name}\", \"uRelax\": 0.7"),
        );
        assert!(e.contains("pbicgstab"), "the fallback must be named for {name}: {e}");
        assert!(e.contains("S52.8"), "the derivation must be named for {name}: {e}");
        assert!(e.contains("NOT implemented"), "and it must say so plainly: {e}");
    }

    // An unknown backend is refused with the menu, not defaulted.
    let e = refused("\"uRelax\": 0.7", "\"pressureSolver\": \"amgx\", \"uRelax\": 0.7");
    assert!(e.contains("available: pbicgstab, pcg"), "{e}");

    // And the two that ARE supported reach the lowered case.
    let a = base().lower().expect("base");
    let b = variant("\"uRelax\": 0.7", "\"pressureSolver\": \"pcg\", \"uRelax\": 0.7")
        .lower()
        .expect("pcg is supported");
    assert_ne!(
        a.solver.solver, b.solver.solver,
        "SPEC-LIT S13.4.1: two cases differing only in pressureSolver must pick \
         different backends"
    );
    assert_ne!(
        a.solver.precon, b.solver.precon,
        "and PCG needs DIC, not DILU - S8.2/S21: conjugate gradients want the \
         symmetric factorisation"
    );

    // A tile that says `baffle: false` is NOT refused - the entry means what
    // it says, and only `true` asks for the thing that is not built.
    variant("\"K\": 300.0", "\"K\": 300.0, \"baffle\": false")
        .lower()
        .expect("baffle: false is the internal-face form, which IS built");
}
