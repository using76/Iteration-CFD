// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
//
// Provenance: ORIGINAL - the tests for the multi-region conduction case
// format (SPEC-LIT §46/§47.4). The §13.4.1 pair tests below run two case
// documents differing in ONE entry and require different output; that is the
// whole point of them. No GPL-licensed source was consulted.

use super::*;

use crate::cht::run_case;
use crate::Gpu;

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// A two-layer slab, written the way a user would write it. Every value that
/// a test varies is threaded through, so the two documents of a pair differ
/// in exactly one place.
#[allow(clippy::too_many_arguments)]
fn slab_case(kappa_a: &str, kappa_b: &str, rc: &str, source: &str, extra: &str) -> String {
    format!(
        r#"{{
  // A two-layer slab: 10 mm of a poor conductor against 20 mm of a good one.
  "name": "twoLayerSlab",
  "regions": [
    {{
      "name": "insulation",
      "mesh": {{
        "bounds": {{ "min": [0.0, 0.0, 0.0], "max": [0.010, 0.02, 0.02] }},
        "cells": [12, 1, 1],
        "boundaries": {{
          "xmin": "hot",  "xmax": "toMetal",
          "ymin": "sideA", "ymax": "sideB",
          "zmin": "frontA", "zmax": "backA"
        }}
      }},
      "material": {{ "rho": 2000.0, "c": 800.0, "kappa": {kappa_a} }},
      {source}
      "patches": [
        {{ "match": "hot",    "T": {{ "type": "fixedValue", "value": 380.0 }} }},
        {{ "match": "sideA",  "T": {{ "type": "zeroGradient" }} }},
        {{ "match": "sideB",  "T": {{ "type": "zeroGradient" }} }},
        {{ "match": "frontA", "T": {{ "type": "zeroGradient" }} }},
        {{ "match": "backA",  "T": {{ "type": "zeroGradient" }} }}
      ]
    }},
    {{
      "name": "metal",
      "mesh": {{
        "bounds": {{ "min": [0.010, 0.0, 0.0], "max": [0.030, 0.02, 0.02] }},
        "cells": [9, 1, 1],
        "boundaries": {{
          "xmin": "toInsulation", "xmax": "cold",
          "ymin": "sideC", "ymax": "sideD",
          "zmin": "frontB", "zmax": "backB"
        }}
      }},
      "material": {{ "rho": 1000.0, "c": 1200.0, "kappa": {kappa_b} }},
      "patches": [
        {{ "match": "cold",   "T": {{ "type": "fixedValue", "value": 300.0 }} }},
        {{ "match": "sideC",  "T": {{ "type": "zeroGradient" }} }},
        {{ "match": "sideD",  "T": {{ "type": "zeroGradient" }} }},
        {{ "match": "frontB", "T": {{ "type": "zeroGradient" }} }},
        {{ "match": "backB",  "T": {{ "type": "zeroGradient" }} }}
      ]
    }}
  ],
  "interfaces": [
    {{
      "regionA": "insulation", "patchA": "toMetal",
      "regionB": "metal",      "patchB": "toInsulation"{rc}
    }}
  ],
  "initial": {{ "T": 340.0 }},
  "run": {{ "steady": true }},
  "numerics": {{
    "solver": "PCG", "preconditioner": "DIC",
    "tolerance": 1e-30, "maxIter": 4000
  }}{extra}
}}"#
    )
}

fn default_slab() -> String {
    slab_case("1.4", "148.0", "", "", "")
}

fn read(text: &str) -> Result<ChtCase> {
    parse_cht_case(text, "test case")
}

// ==========================================================================
//  Reading, and every refusal
// ==========================================================================

#[test]
fn the_example_case_reads_and_lowers() {
    let case = read(&default_slab()).expect("parse");
    let low = case.lower().expect("lower");
    assert_eq!(low.region_names, ["insulation", "metal"]);
    assert_eq!(low.meshes[0].n_cells, 12);
    assert_eq!(low.meshes[1].n_cells, 9);
    assert_eq!(low.interfaces.len(), 1);
    assert_eq!(low.interfaces[0].r_c, 0.0);
    assert!(low.steady);
    // Ten patches carry a rule and two carry the interface.
    assert_eq!(low.patch_bcs.len(), 10);
}

/// The rule the format is built around. A patch that carries no condition is
/// an error listing it, not a silent adiabatic default - that default is
/// exactly how a case comes to say something the solver ignores.
#[test]
fn a_patch_with_no_condition_is_refused_by_name() {
    let text = default_slab().replace(
        r#"{ "match": "backB",  "T": { "type": "zeroGradient" } }"#,
        r#"{ "match": "sideD",  "T": { "type": "zeroGradient" } }"#,
    );
    // `sideD` is now named twice and `backB` not at all.
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("sideD") || msg.contains("backB"), "{msg}");
}

#[test]
fn an_unnamed_patch_is_listed() {
    // Drop one rule, leaving `backB` with no condition at all.
    let text = default_slab().replace(
        r#",
        { "match": "backB",  "T": { "type": "zeroGradient" } }"#,
        "",
    );
    assert_ne!(text, default_slab(), "the rule must actually have been dropped");

    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("carry no condition"), "{msg}");
    assert!(msg.contains("metal:backB"), "the message must NAME the patch: {msg}");
}

/// A patch cannot be both an interface and a `patches` rule - §47.6: they
/// rewrite the same three numbers.
#[test]
fn a_patch_cannot_be_both_an_interface_and_a_rule() {
    let text = default_slab().replace(
        r#"{ "match": "sideA",  "T": { "type": "zeroGradient" } },"#,
        r#"{ "match": "toMetal", "T": { "type": "zeroGradient" } },
        { "match": "sideA",  "T": { "type": "zeroGradient" } },"#,
    );
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("toMetal"), "{msg}");
    assert!(msg.contains("ONE condition"), "{msg}");
}

/// SPEC-LIT §46.4/§13.4: nine components is an error naming the two that are
/// implemented, and it arrives from the case file rather than from a library
/// call.
#[test]
fn a_full_tensor_kappa_in_a_case_is_refused_naming_the_alternatives() {
    crate::io::contract::reset_warnings();
    let text = slab_case("[148,3,0, 3,148,0, 0,0,1.4]", "148.0", "", "", "");
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("kappaSolid <k>"), "{msg}");
    assert!(msg.contains("MPFA") || msg.contains("multipoint"), "{msg}");
    assert!(msg.contains("regions/insulation/material/kappa"), "{msg}");
}

/// The two spellings of the contact resistance are the same number, and a
/// case that writes both has said it twice.
#[test]
fn rc_and_the_layer_lists_cannot_both_be_given() {
    let text = slab_case(
        "1.4",
        "148.0",
        r#", "Rc": 1e-4, "thicknessLayers": [5e-5], "kappaLayers": [3.0]"#,
        "",
        "",
    );
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    assert!(e.to_string().contains("two spellings"), "{e}");

    // And one without the other is incomplete, not half-understood.
    let text = slab_case("1.4", "148.0", r#", "thicknessLayers": [5e-5]"#, "", "");
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    assert!(e.to_string().contains("must both be given"), "{e}");
}

/// The layer form lowers to (S47.11)'s sum.
#[test]
fn the_layer_lists_lower_to_the_series_resistance() {
    let text = slab_case(
        "1.4",
        "148.0",
        r#", "thicknessLayers": [5e-5, 2e-4], "kappaLayers": [3.0, 0.2]"#,
        "",
        "",
    );
    let low = read(&text).expect("parse").lower().expect("lower");
    let want = 5e-5 / 3.0 + 2e-4 / 0.2;
    assert!((low.interfaces[0].r_c - want).abs() < 1e-15, "{}", low.interfaces[0].r_c);
}

/// A fluid region is not something this format can express, and saying so is
/// better than building a solid one and calling it a fluid.
#[test]
fn a_fluid_region_is_refused_naming_what_is_implemented() {
    crate::io::contract::reset_warnings();
    let text = default_slab().replace(
        r#""name": "metal","#,
        r#""name": "metal", "kind": "fluid","#,
    );
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("solid"), "{msg}");
    assert!(msg.contains("SOLID energy equation") || msg.contains("momentum"), "{msg}");
}

/// A mistyped entry is a parse error, not something quietly dropped -
/// `deny_unknown_fields` throughout, and the message names the path.
#[test]
fn a_mistyped_entry_is_a_parse_error_naming_it() {
    let text = default_slab().replace(r#""kappa": 1.4"#, r#""kapa": 1.4"#);
    let e = read(&text).expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("kapa") || msg.contains("unknown field"), "{msg}");
}

#[test]
fn a_steady_case_that_also_names_a_time_is_refused() {
    let text = default_slab().replace(
        r#""run": { "steady": true }"#,
        r#""run": { "steady": true, "endTime": 10.0, "deltaT": 0.1 }"#,
    );
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    assert!(e.to_string().contains("steady"), "{e}");
}

#[test]
fn an_unknown_solver_or_preconditioner_is_refused_by_name() {
    crate::io::contract::reset_warnings();
    let text = default_slab().replace(r#""solver": "PCG""#, r#""solver": "GAMG""#);
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("PCG"), "{msg}");
    assert!(msg.contains("PBiCGStab"), "{msg}");

    let text = default_slab().replace(r#""preconditioner": "DIC""#, r#""preconditioner": "AMG""#);
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    assert!(e.to_string().contains("DILU"), "{e}");
}

#[test]
fn an_interface_naming_a_patch_that_does_not_exist_lists_the_ones_that_do() {
    let text = default_slab().replace(r#""patchA": "toMetal""#, r#""patchA": "toAluminium""#);
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("toAluminium"), "{msg}");
    assert!(msg.contains("toMetal"), "the message must list what IS there: {msg}");
}

// ==========================================================================
//  End to end, and the §13.4.1 pair tests
// ==========================================================================

fn solve(gpu: &Gpu, text: &str) -> crate::cht::ChtSolution {
    let case = read(text).expect("parse");
    let low = case.lower().expect("lower");
    run_case(gpu, &low).expect("run")
}

/// The whole path, against the closed form: a two-layer slab conducts
/// `q = dT/(L1/k1 + Rc + L2/k2)` - SPEC-LIT §47.12 Gate 1, reached from a
/// case document rather than from a rig.
#[test]
fn the_case_file_route_reproduces_gate_1() {
    let Some(gpu) = gpu() else { return };
    let sol = solve(&gpu, &default_slab());

    let area: Scalar = sol
        .mesh
        .pairs
        .iter()
        .map(|p| sol.mesh.host.b_mag_sf[p.bf_a as usize])
        .sum();
    let q_got = -sol.interface.into_a / area;
    let q_exact = (380.0 - 300.0) / (0.010 / 1.4 + 0.020 / 148.0);
    assert!(
        (q_got / q_exact - 1.0).abs() < 1e-12,
        "q = {q_got}, exact {q_exact}"
    );
    assert!(sol.interface.imbalance() < 1e-12);
    assert_eq!(sol.steps, 1, "a steady case is one solve");
}

/// **The §13.4.1 pair test for `Rc`, on two case DOCUMENTS differing in one
/// entry.** They are required to produce different output.
#[test]
fn two_cases_differing_only_in_rc_produce_different_output() {
    let Some(gpu) = gpu() else { return };

    let a = slab_case("1.4", "148.0", "", "", "");
    let b = slab_case("1.4", "148.0", r#", "Rc": 2e-3"#, "", "");
    assert_ne!(a, b, "the two documents must actually differ");

    let sa = solve(&gpu, &a);
    let sb = solve(&gpu, &b);

    let dt = sa
        .t
        .iter()
        .zip(&sb.t)
        .fold(0.0 as Scalar, |m, (x, y)| m.max((x - y).abs()));
    assert!(
        dt > 1.0,
        "a case that says Rc = 2e-3 and one that says nothing differ by only {dt} K - \
         the case said Rc and the solver ignored it (SPEC-LIT 13.4.1)"
    );
    assert!(
        sb.interface.into_a.abs() < sa.interface.into_a.abs(),
        "adding a contact resistance must REDUCE the heat crossing: {} vs {}",
        sb.interface.into_a,
        sa.interface.into_a
    );
}

/// **The §13.4.1 pair test for `kappa`.**
#[test]
fn two_cases_differing_only_in_kappa_produce_different_output() {
    let Some(gpu) = gpu() else { return };
    let sa = solve(&gpu, &slab_case("1.4", "148.0", "", "", ""));
    let sb = solve(&gpu, &slab_case("1.4", "10.0", "", "", ""));
    let dt = sa
        .t
        .iter()
        .zip(&sb.t)
        .fold(0.0 as Scalar, |m, (x, y)| m.max((x - y).abs()));
    assert!(dt > 1.0, "kappa 148 and kappa 10 differ by only {dt} K");
}

/// **The §13.4.1 pair test for an ANISOTROPIC `kappa`.** `[1.4 1.4 1.4]` and
/// `[1.4 1.4 14]` are two different materials.
#[test]
fn two_cases_differing_only_in_the_anisotropy_produce_different_output() {
    let Some(gpu) = gpu() else { return };

    // A 2-D region so the z conductivity is genuinely in the answer: the
    // z faces of region A get a fixed temperature instead of adiabatic.
    let with = |kz: &str| {
        default_slab()
            .replace(r#""cells": [12, 1, 1]"#, r#""cells": [6, 1, 6]"#)
            .replace(r#""cells": [9, 1, 1]"#, r#""cells": [9, 1, 6]"#)
            .replace(r#""kappa": 1.4"#, &format!(r#""kappa": [1.4, 1.4, {kz}]"#))
            .replace(
                r#"{ "match": "frontA", "T": { "type": "zeroGradient" } }"#,
                r#"{ "match": "frontA", "T": { "type": "fixedValue", "value": 300.0 } }"#,
            )
    };
    let sa = solve(&gpu, &with("1.4"));
    let sb = solve(&gpu, &with("14.0"));
    let dt = sa
        .t
        .iter()
        .zip(&sb.t)
        .fold(0.0 as Scalar, |m, (x, y)| m.max((x - y).abs()));
    assert!(
        dt > 1.0,
        "kappa [1.4 1.4 1.4] and [1.4 1.4 14] differ by only {dt} K - the case said \
         an anisotropic conductivity and the solver ignored it (SPEC-LIT 13.4.1)"
    );
}

/// **The §13.4.1 pair test for `source`.** A die that dissipates must be
/// hotter than one that does not.
#[test]
fn two_cases_differing_only_in_the_volumetric_source_produce_different_output() {
    let Some(gpu) = gpu() else { return };
    let sa = solve(&gpu, &slab_case("1.4", "148.0", "", "", ""));
    let sb = solve(&gpu, &slab_case("1.4", "148.0", "", r#""source": 5.0e6,"#, ""));
    let dt = sa
        .t
        .iter()
        .zip(&sb.t)
        .fold(0.0 as Scalar, |m, (x, y)| m.max((x - y).abs()));
    assert!(dt > 1.0, "a 5 MW/m^3 source changed the field by only {dt} K");
    assert!(
        sb.region_mean(0) > sa.region_mean(0),
        "a heated region must be hotter: {} vs {}",
        sb.region_mean(0),
        sa.region_mean(0)
    );
}

/// A fixed heat flux delivers exactly the flux it names - SPEC-LIT §32.2,
/// reached through this format. `q` into a slab whose other face is held
/// gives a linear profile with `dT = q L/k` exactly.
#[test]
fn a_fixed_flux_patch_delivers_exactly_the_flux_it_names() {
    let Some(gpu) = gpu() else { return };
    let q = 4000.0 as Scalar;
    let text = default_slab().replace(
        r#"{ "match": "hot",    "T": { "type": "fixedValue", "value": 380.0 } }"#,
        &format!(r#"{{ "match": "hot", "T": {{ "type": "fixedFluxTemperature", "q": {q} }} }}"#),
    );
    let sol = solve(&gpu, &text);

    // Everything that enters at `hot` leaves through the interface.
    let area: Scalar = sol
        .mesh
        .pairs
        .iter()
        .map(|p| sol.mesh.host.b_mag_sf[p.bf_a as usize])
        .sum();
    let through = -sol.interface.into_a / area;
    assert!(
        (through / q - 1.0).abs() < 1e-11,
        "the interface carries {through} W/m^2 where the patch prescribed {q}"
    );
}

/// A transient case runs the number of steps it asks for, and relaxes toward
/// the steady answer.
#[test]
fn a_transient_case_runs_its_own_steps_and_approaches_the_steady_answer() {
    let Some(gpu) = gpu() else { return };
    let steady = solve(&gpu, &default_slab());

    let transient = |end: &str| {
        default_slab().replace(
            r#""run": { "steady": true }"#,
            &format!(r#""run": {{ "endTime": {end}, "deltaT": 0.05 }}"#),
        )
    };
    let short = solve(&gpu, &transient("0.5"));
    let long = solve(&gpu, &transient("20.0"));
    assert_eq!(short.steps, 10);
    assert_eq!(long.steps, 400);

    let gap = |s: &crate::cht::ChtSolution| {
        s.t.iter()
            .zip(&steady.t)
            .fold(0.0 as Scalar, |m, (x, y)| m.max((x - y).abs()))
    };
    let (g_short, g_long) = (gap(&short), gap(&long));
    assert!(
        g_long < 0.05 * g_short,
        "the long run must be closer to steady: {g_long} K against {g_short} K"
    );
}

/// The contact resistance shows as a temperature JUMP in the reported face
/// values - the thing §47.3 says the cyclic branch could not have carried.
#[test]
fn the_reported_face_values_carry_the_contact_resistance_jump() {
    let Some(gpu) = gpu() else { return };
    let r_c = 2.0e-3 as Scalar;
    let sol = solve(&gpu, &slab_case("1.4", "148.0", r#", "Rc": 2e-3"#, "", ""));

    let area: Scalar = sol
        .mesh
        .pairs
        .iter()
        .map(|p| sol.mesh.host.b_mag_sf[p.bf_a as usize])
        .sum();
    let q = -sol.interface.into_a / area;
    for p in &sol.mesh.pairs {
        let jump = sol.bt[p.bf_a as usize] - sol.bt[p.bf_b as usize];
        assert!(
            (jump - q * r_c).abs() < 1e-9 * 80.0,
            "jump {jump} K against q Rc = {}",
            q * r_c
        );
        assert!(jump > 1.0, "the jump must be visible, not notional: {jump} K");
    }
}

// ==========================================================================
//  The shipped case, against its own closed form
// ==========================================================================

/// **`cases/dieStack.cht.jsonc`, end to end, against hand arithmetic.**
///
/// Four regions, three contact resistances, an anisotropic die dissipating
/// 100 W, and an isothermal cold plate. The whole stack is one-dimensional in
/// `z`, so it has an exact DISCRETE answer:
///
/// ```text
/// T_junction = T_plate
///            + q ( Rc1 + L_sol/k_sol + Rc2 + L_spr/k_spr + Rc3 + L_gre/k_gre )
///            + q (h_die/2)/k_z                        the die's bottom half-cell
///            + (g h_die^2/k_z) (1 + 2 + ... + (n-1))  the source lumped per cell
/// ```
///
/// The last two lines are the finite-volume die, not the analytic one: with a
/// uniform source the two differ at second order, and the point of this gate
/// is that the SOLVER reproduces its own discretisation exactly, not that the
/// discretisation is exact. `q = 1e6 W/m^2`, the whole 100 W over the
/// 10 x 10 mm footprint.
#[test]
fn the_shipped_die_stack_case_matches_its_closed_form() {
    let Some(gpu) = gpu() else { return };

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../cases/dieStack.cht.jsonc");
    let case = super::read_cht_case(&path).expect("read cases/dieStack.cht.jsonc");
    let low = case.lower().expect("lower");
    let sol = run_case(&gpu, &low).expect("run");

    // Every interface carries the whole 100 W: nothing is lost between the
    // die and the plate.
    let flows = sol.interface_flows();
    assert_eq!(flows.len(), 3, "three declared interfaces");
    for (name, into_a, into_b) in &flows {
        assert!(
            (into_a + 100.0).abs() < 1e-8,
            "{name}: {into_a} W leaves the upper side, not -100"
        );
        assert!((into_b - 100.0).abs() < 1e-8, "{name}: {into_b} W");
    }
    assert!(sol.interface.imbalance() < 1e-12);

    // The closed form, built here from the case's own numbers rather than
    // transcribed as a magic constant.
    let q = 1.0e6 as Scalar; // 100 W over 1e-4 m^2
    let r_below = 1.0e-5                     // Rc, die/solder
        + 5.0e-5 / 57.0                      // the solder itself
        + 2.5e-5                             // Rc, solder/spreader
        + 2.0e-3 / 398.0                     // the spreader
        + 4.0e-5 / 1.0                       // thicknessLayers/kappaLayers
        + 9.0e-4 / 3.5; // the grease
    let (l_die, k_z, n_die) = (7.0e-4 as Scalar, 30.0 as Scalar, 4usize);
    let h = l_die / n_die as Scalar;
    let g = q / l_die; // the volumetric source the case names
    let die_rise = q * (h / 2.0) / k_z
        + (g * h * h / k_z) * ((n_die * (n_die - 1) / 2) as Scalar);
    let want = 300.0 + q * r_below + die_rise;

    let (_, t_max) = sol.region_range(0);
    assert!(
        (t_max - want).abs() < 1e-8 * want,
        "junction temperature {t_max} K against the closed form {want} K"
    );

    // And the three contact resistances show as three temperature JUMPS of
    // exactly q Rc - the thing §47.3's face values are for.
    for (i, r_c) in [1.0e-5 as Scalar, 2.5e-5, 4.0e-5].into_iter().enumerate() {
        let range = sol.mesh.interface_ranges[i].1.clone();
        let p = sol.mesh.pairs[range.start];
        let jump = sol.bt[p.bf_a as usize] - sol.bt[p.bf_b as usize];
        assert!(
            (jump - q * r_c).abs() < 1e-7 * (q * r_c),
            "interface {i}: jump {jump} K against q Rc = {}",
            q * r_c
        );
    }
}

/// **The §13.4.1 pair test on the SHIPPED case.** Perturb one contact
/// resistance in `cases/dieStack.cht.jsonc` and the junction temperature must
/// move - by `q dRc`, which is a number the case's own arithmetic predicts.
#[test]
fn perturbing_one_contact_resistance_in_the_shipped_case_moves_the_junction() {
    let Some(gpu) = gpu() else { return };

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../cases/dieStack.cht.jsonc");
    let text = std::fs::read_to_string(&path).expect("read the case");
    assert!(text.contains(r#""Rc": 2.5e-5"#), "the case moved; update this test");
    let bumped = text.replace(r#""Rc": 2.5e-5"#, r#""Rc": 7.5e-5"#);

    let base = run_case(
        &gpu,
        &parse_cht_case(&text, "dieStack").expect("parse").lower().expect("lower"),
    )
    .expect("run");
    let hot = run_case(
        &gpu,
        &parse_cht_case(&bumped, "dieStack+").expect("parse").lower().expect("lower"),
    )
    .expect("run");

    let (_, t0) = base.region_range(0);
    let (_, t1) = hot.region_range(0);
    let rise = t1 - t0;
    let want = 1.0e6 * (7.5e-5 - 2.5e-5); // q dRc
    assert!(
        (rise - want).abs() < 1e-7 * want,
        "adding 5e-5 m^2K/W of contact resistance raised the junction by {rise} K, \
         not the {want} K the case's own arithmetic predicts - if it moved by ZERO \
         the case said Rc and the solver ignored it (SPEC-LIT 13.4.1)"
    );
}

/// The twin of `io::case_json`'s own shipped-case scan, for this format:
/// every `*.cht.jsonc` at the top level of `cases/` reads and lowers cleanly
/// as shipped. A new one that does not fails here rather than in a
/// licensee's fresh clone.
#[test]
fn every_shipped_cht_case_lowers() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../cases");
    let entries = std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));

    let mut checked = 0usize;
    for entry in entries {
        let path = entry.expect("dir entry").path();
        let is_cht = path
            .file_name()
            .and_then(|f| f.to_str())
            .is_some_and(|f| f.ends_with(".cht.jsonc"));
        if !is_cht {
            continue;
        }
        let case = super::read_cht_case(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        case.lower()
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        checked += 1;
    }
    assert!(checked > 0, "no *.cht.jsonc case was found under {}", dir.display());
}
