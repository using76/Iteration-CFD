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
/// SPEC-LIT §60.3: `kind` takes exactly two values, and a third is a §13.4
/// error listing both. (The fluid region itself is implemented - SPEC-LIT
/// §59/§60 - and its own refusals are the block below.)
#[test]
fn an_unknown_region_kind_is_refused_listing_both() {
    crate::io::contract::reset_warnings();
    let text = default_slab().replace(
        r#""name": "metal","#,
        r#""name": "metal", "kind": "porous","#,
    );
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("solid") && msg.contains("fluid"), "{msg}");
}

/// SPEC-LIT §47.4's numbering invariant, refused where the case can say it:
/// the fluid block keeps its own cell and boundary-face indices only if it is
/// region 0.
#[test]
fn a_fluid_region_that_is_not_first_is_refused_naming_it() {
    crate::io::contract::reset_warnings();
    let text = default_slab().replace(
        r#""name": "metal","#,
        r#""name": "metal", "kind": "fluid","#,
    );
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("metal") && msg.contains("FIRST"), "{msg}");
    assert!(msg.contains("47.4"), "{msg}");
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

// ==========================================================================
//  SPEC-LIT §60 - the fluid region: what it says, what is refused, and the
//  §13.4.1 pair tests
//
//  The refusals below are host-only and cost microseconds. The pair tests
//  RUN, because a pair test that did not run would only prove the two
//  documents parse differently.
// ==========================================================================

/// The Kaminski & Prakash configuration as a case document - SPEC-LIT §60.1.
///
/// `n` is the cell count across the whole `1 x 1` enclosure; the wall gets
/// `0.2 n` columns and the air `0.8 n`, which makes every cell square.
/// Everything a pair test varies is a parameter, so the two documents it
/// writes differ in exactly one substring.
#[allow(clippy::too_many_arguments)]
fn kp_case(
    n: usize,
    kappa_solid: &str,
    fluid: &str,
    g: &str,
    t_ref: &str,
    rc: &str,
    iterations: usize,
    residual: &str,
    relax_u: &str,
) -> String {
    let dz = 1.0 / n as f64;
    let n_solid = (0.2 * n as f64).round() as usize;
    format!(
        r#"{{
  "name": "kaminskiPrakash",
  "regions": [
    {{
      "name": "air", "kind": "fluid",
      "mesh": {{
        "bounds": {{ "min": [0.2, 0.0, 0.0], "max": [1.0, 1.0, {dz}] }},
        "cells": [{}, {n}, 1],
        "boundaries": {{
          "xmin": "airToWall", "xmax": "cold",
          "ymin": "airBottom", "ymax": "airTop",
          "zmin": "airFront",  "zmax": "airBack"
        }}
      }},
      "fluid": {fluid},
      "patches": [
        {{ "match": "cold",      "T": {{ "type": "fixedValue", "value": 299.95 }} }},
        {{ "match": "airBottom", "T": {{ "type": "zeroGradient" }} }},
        {{ "match": "airTop",    "T": {{ "type": "zeroGradient" }} }},
        {{ "match": "airFront",  "T": {{ "type": "empty" }} }},
        {{ "match": "airBack",   "T": {{ "type": "empty" }} }}
      ]
    }},
    {{
      "name": "wall", "kind": "solid",
      "mesh": {{
        "bounds": {{ "min": [0.0, 0.0, 0.0], "max": [0.2, 1.0, {dz}] }},
        "cells": [{n_solid}, {n}, 1],
        "boundaries": {{
          "xmin": "hot",        "xmax": "wallToAir",
          "ymin": "wallBottom", "ymax": "wallTop",
          "zmin": "wallFront",  "zmax": "wallBack"
        }}
      }},
      "material": {{ "rho": 1.0, "c": 1.0, "kappa": {kappa_solid} }},
      "patches": [
        {{ "match": "hot",        "T": {{ "type": "fixedValue", "value": 300.05 }} }},
        {{ "match": "wallBottom", "T": {{ "type": "zeroGradient" }} }},
        {{ "match": "wallTop",    "T": {{ "type": "zeroGradient" }} }},
        {{ "match": "wallFront",  "T": {{ "type": "empty" }} }},
        {{ "match": "wallBack",   "T": {{ "type": "empty" }} }}
      ]
    }}
  ],
  "interfaces": [
    {{ "regionA": "air", "patchA": "airToWall",
       "regionB": "wall", "patchB": "wallToAir"{rc} }}
  ],
  "buoyancy": {{ "g": [0.0, {g}, 0.0], "TRef": {t_ref} }},
  "initial": {{ "T": 300.0 }},
  "run": {{ "steady": true, "iterations": {iterations} }},
  "numerics": {{
    "solver": "PBiCGStab", "preconditioner": "DILU",
    "tolerance": 1e-16, "maxIter": 400,
    "flow": {{
      "relaxU": {relax_u}, "relaxP": 0.3, "relaxT": 0.7,
      "divSchemeU": "Gauss linear", "divSchemeT": "Gauss linear",
      "residual": {residual},
      "uTolerance": 1e-14, "pTolerance": 1e-14,
      "uMaxIter": 150, "pMaxIter": 500
    }}
  }}
}}"#,
        n - n_solid
    )
}

/// The pair tests' base document: coarse, and run only far enough that two
/// answers can differ. `Ra = 1e4` with `Kr = 1`.
fn kp_pair_base() -> String {
    kp_case(20, "1.0", r#"{ "rho": 1.0, "cp": 1.0, "kappa": 1.0, "mu": 0.71 }"#,
            "-2.13e7", "300.0", "", 600, "0.0", "0.7")
}

fn run_flow(gpu: &Gpu, text: &str) -> crate::cht::flow::ChtFlowSolution {
    let low = read(text)
        .unwrap_or_else(|e| panic!("parse: {e}"))
        .lower()
        .unwrap_or_else(|e| panic!("lower: {e}"));
    let case = low.flow_case().expect("a fluid case lowers to a FlowCase");
    crate::cht::flow::run_flow_case(gpu, &case).expect("run")
}

/// The §13.4.1 pair test itself, factored: two documents differing in one
/// substring must produce different temperature fields.
fn pair_differs(gpu: &Gpu, from: &str, to: &str, what: &str) {
    let a = kp_pair_base();
    let b = a.replace(from, to);
    assert_ne!(a, b, "the pair test's own substitution '{from}' -> '{to}' changed nothing");
    let sa = run_flow(gpu, &a);
    let sb = run_flow(gpu, &b);
    let gap = sa
        .t
        .iter()
        .zip(&sb.t)
        .fold(0.0 as Scalar, |m, (x, y)| m.max((x - y).abs()));
    assert!(
        gap > 1e-12,
        "changing {what} moved the temperature field by {gap} K. Two cases \
         differing in one entry produced the SAME answer, which means the case \
         said {what} and the solver ignored it (SPEC-LIT 13.4.1)"
    );
}

#[test]
fn the_shipped_kaminski_prakash_case_reads_and_lowers_as_a_flow_case() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../cases/kaminskiPrakash.cht.jsonc");
    let case = super::read_cht_case(&path).expect("read");
    let low = case.lower().expect("lower");
    assert!(low.has_fluid(), "the shipped case has a fluid region");
    assert_eq!(low.kinds(), vec![RegionKind::Fluid, RegionKind::Solid]);
    let flow = low.flow_case().expect("flow case");
    assert_eq!(flow.regions.len(), 2);
    assert!(flow.regions[0].fluid.is_some() && flow.regions[0].solid.is_none());
    assert!(flow.regions[1].solid.is_some() && flow.regions[1].fluid.is_none());
    // Pr = mu cp/kappa, the number a reader checks the case by.
    let pr = flow.regions[0].fluid.as_ref().expect("fluid").pr();
    assert!((pr - 0.71).abs() < 1e-12, "Pr = {pr}");
}

/// SPEC-LIT §60.3, both directions. Every one of these is a setting the case
/// could write and the solver would ignore, or one the solver needs and the
/// case did not write - the §13.4.1 defect on either side.
#[test]
fn the_fluid_only_blocks_are_refused_in_both_directions() {
    crate::io::contract::reset_warnings();

    // Present with no fluid region.
    for (name, extra) in [
        ("buoyancy", r#", "buoyancy": { "g": [0,-9.81,0], "TRef": 300.0 }"#),
        (
            "numerics/flow",
            "", // handled below - it lives INSIDE numerics
        ),
    ] {
        if extra.is_empty() {
            continue;
        }
        let text = slab_case("1.4", "148.0", "", "", extra);
        let e = read(&text).expect("parse").lower().expect_err("must refuse");
        let msg = e.to_string();
        assert!(msg.contains(name), "{msg}");
        assert!(msg.contains("13.4.1"), "{msg}");
    }

    let text = default_slab().replace(
        r#""tolerance": 1e-30, "maxIter": 4000"#,
        r#""tolerance": 1e-30, "maxIter": 4000,
    "flow": { "relaxU": 0.7, "relaxP": 0.3, "relaxT": 0.7 }"#,
    );
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    assert!(e.to_string().contains("numerics/flow"), "{e}");

    let text = default_slab().replace(
        r#""run": { "steady": true }"#,
        r#""run": { "steady": true, "iterations": 100 }"#,
    );
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    assert!(e.to_string().contains("run/iterations"), "{e}");

    // Absent with one.
    for (name, from, to) in [
        (
            "buoyancy",
            r#""buoyancy": { "g": [0.0, -2.13e7, 0.0], "TRef": 300.0 },"#,
            "",
        ),
        (
            "run/iterations",
            r#""run": { "steady": true, "iterations": 600 }"#,
            r#""run": { "steady": true }"#,
        ),
    ] {
        let text = kp_pair_base().replace(from, to);
        assert!(text != kp_pair_base(), "the substitution for {name} matched nothing");
        let e = read(&text).expect("parse").lower().expect_err("must refuse");
        let msg = e.to_string();
        assert!(msg.contains(name), "{msg}");
        assert!(msg.contains("60.2"), "{msg}");
    }
}

/// SPEC-LIT §60.3: a solid region carries `material`, a fluid one `fluid`,
/// and the reader will not choose between them.
#[test]
fn the_wrong_material_block_for_a_kind_is_refused_naming_both() {
    crate::io::contract::reset_warnings();

    // A fluid region with a `material` block as well.
    let text = kp_pair_base().replace(
        r#""fluid": { "rho": 1.0, "cp": 1.0, "kappa": 1.0, "mu": 0.71 },"#,
        r#""fluid": { "rho": 1.0, "cp": 1.0, "kappa": 1.0, "mu": 0.71 },
      "material": { "rho": 1.0, "c": 1.0, "kappa": 1.0 },"#,
    );
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    assert!(e.to_string().contains("FLUID region carries `fluid`"), "{e}");

    // A fluid region with no `fluid` block at all.
    let text = kp_pair_base().replace(
        r#""fluid": { "rho": 1.0, "cp": 1.0, "kappa": 1.0, "mu": 0.71 },"#,
        "",
    );
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    assert!(e.to_string().contains("needs a `fluid` block"), "{e}");

    // A solid region with a `fluid` block.
    let text = kp_pair_base().replace(
        r#""material": { "rho": 1.0, "c": 1.0, "kappa": 1.0 },"#,
        r#""material": { "rho": 1.0, "c": 1.0, "kappa": 1.0 },
      "fluid": { "rho": 1.0, "cp": 1.0, "kappa": 1.0, "mu": 0.71 },"#,
    );
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    assert!(e.to_string().contains("SOLID region carries `material`"), "{e}");
}

/// SPEC-LIT §59.6: a conjugate fluid transient is refused, because nothing in
/// this tree gates its time accuracy.
#[test]
fn a_transient_fluid_case_is_refused_naming_what_is_not_gated() {
    crate::io::contract::reset_warnings();
    let text = kp_pair_base().replace(
        r#""run": { "steady": true, "iterations": 600 }"#,
        r#""run": { "endTime": 1.0, "deltaT": 0.01, "iterations": 600 }"#,
    );
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("steady"), "{msg}");
    assert!(msg.contains("59.6"), "{msg}");
}

/// SPEC-LIT §60.3: a volumetric source on a fluid region would be read and
/// dropped, which is the §13.4.1 defect.
#[test]
fn a_source_on_a_fluid_region_is_refused_rather_than_dropped() {
    crate::io::contract::reset_warnings();
    let text = kp_pair_base().replace(
        r#""fluid": { "rho": 1.0, "cp": 1.0, "kappa": 1.0, "mu": 0.71 },"#,
        r#""fluid": { "rho": 1.0, "cp": 1.0, "kappa": 1.0, "mu": 0.71 },
      "source": 1000.0,"#,
    );
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("source"), "{msg}");
    assert!(msg.contains("13.4.1"), "{msg}");
}

/// SPEC-LIT §60.3: `empty` patches come in opposite pairs.
#[test]
fn a_lone_empty_patch_is_refused() {
    crate::io::contract::reset_warnings();
    let text = kp_pair_base().replace(
        r#"{ "match": "airBack",   "T": { "type": "empty" } }"#,
        r#"{ "match": "airBack",   "T": { "type": "zeroGradient" } }"#,
    );
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    assert!(e.to_string().contains("opposite"), "{e}");
}

/// A fluid case is not `cht::run_case`'s business, and it says so rather than
/// building the fluid as a conducting solid.
#[test]
fn run_case_refuses_a_fluid_case_naming_the_function_that_solves_it() {
    let Some(gpu) = gpu() else { return };
    let low = read(&kp_pair_base()).expect("parse").lower().expect("lower");
    let msg = match crate::cht::run_case(&gpu, &low) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("cht::run_case must refuse a fluid case (SPEC-LIT 60)"),
    };
    assert!(msg.contains("run_flow_case"), "{msg}");
}

// ---- the §13.4.1 pair tests, run ---------------------------------------

/// **The pair test for the whole of SPEC-LIT §59/§60**: if `kind` were
/// ignored, the fluid region would be solved as a conducting solid and the
/// answer would be the pure-conduction one - which is precisely the defect
/// §47.14's refusal existed to prevent, now that the refusal is gone.
///
/// This is the ONE pair in §60.4 whose two documents cannot differ in a single
/// entry, and that is a property of the format rather than a weakness of the
/// test: §60.3 requires `buoyancy`, `numerics.flow` and `run.iterations` with
/// a fluid region and REFUSES all three without one, so a document with
/// `kind: solid` and a `buoyancy` block does not lower at all. The two
/// documents below are therefore the minimal pair that both lower, and the
/// test asserts that the solid twin really is a pure-conduction case
/// (`has_fluid() == false`) before comparing.
#[test]
fn pair_the_region_kind_itself_changes_the_answer() {
    let Some(gpu) = gpu() else { return };
    let fluid = run_flow(&gpu, &kp_pair_base());

    let low = read(&kp_solid_twin(20))
        .expect("parse")
        .lower()
        .expect("lower");
    assert!(!low.has_fluid(), "the twin must be a pure-conduction case");
    let solid = crate::cht::run_case(&gpu, &low).expect("run");

    let gap = fluid
        .t
        .iter()
        .zip(&solid.t)
        .fold(0.0 as Scalar, |m, (x, y)| m.max((x - y).abs()));
    assert!(
        gap > 1e-9,
        "the same geometry solved with `kind: fluid` and with `kind: solid`          gave the same temperature field (worst difference {gap} K). The case          said `fluid` and the solver ignored it - SPEC-LIT 13.4.1"
    );

    // And the direction is the one the physics demands: convection carries
    // MORE heat than conduction alone, so the fluid run's cold wall takes
    // more out.
    let q_fluid = -fluid.patch_heat_flow(0, "cold").expect("cold");
    let h = &solid.mesh.host;
    let mut q_solid = 0.0 as Scalar;
    for bf in solid.mesh.patch_range(0, "cold").expect("cold") {
        let c = h.b_face_cells[bf] as usize;
        // kappa_f = 1 in the twin, so C_b = Delta_b.
        q_solid -= h.b_mag_sf[bf] * h.b_delta_coeffs[bf] * (solid.bt[bf] - solid.t[c]);
    }
    assert!(
        q_fluid > 1.001 * q_solid,
        "convection must carry more than conduction: {q_fluid} W against          {q_solid} W"
    );
}

/// The pure-conduction twin of [`kp_pair_base`]: the same geometry and the
/// same materials with `kind: solid` on region 0 and, necessarily, none of
/// §60.3's fluid-only blocks.
fn kp_solid_twin(n: usize) -> String {
    let dz = 1.0 / n as f64;
    let n_solid = (0.2 * n as f64).round() as usize;
    format!(
        r#"{{
  "name": "kaminskiPrakashConductionTwin",
  "regions": [
    {{
      "name": "air", "kind": "solid",
      "mesh": {{
        "bounds": {{ "min": [0.2, 0.0, 0.0], "max": [1.0, 1.0, {dz}] }},
        "cells": [{}, {n}, 1],
        "boundaries": {{
          "xmin": "airToWall", "xmax": "cold",
          "ymin": "airBottom", "ymax": "airTop",
          "zmin": "airFront",  "zmax": "airBack"
        }}
      }},
      "material": {{ "rho": 1.0, "c": 1.0, "kappa": 1.0 }},
      "patches": [
        {{ "match": "cold",      "T": {{ "type": "fixedValue", "value": 299.95 }} }},
        {{ "match": "airBottom", "T": {{ "type": "zeroGradient" }} }},
        {{ "match": "airTop",    "T": {{ "type": "zeroGradient" }} }},
        {{ "match": "airFront",  "T": {{ "type": "empty" }} }},
        {{ "match": "airBack",   "T": {{ "type": "empty" }} }}
      ]
    }},
    {{
      "name": "wall", "kind": "solid",
      "mesh": {{
        "bounds": {{ "min": [0.0, 0.0, 0.0], "max": [0.2, 1.0, {dz}] }},
        "cells": [{n_solid}, {n}, 1],
        "boundaries": {{
          "xmin": "hot",        "xmax": "wallToAir",
          "ymin": "wallBottom", "ymax": "wallTop",
          "zmin": "wallFront",  "zmax": "wallBack"
        }}
      }},
      "material": {{ "rho": 1.0, "c": 1.0, "kappa": 1.0 }},
      "patches": [
        {{ "match": "hot",        "T": {{ "type": "fixedValue", "value": 300.05 }} }},
        {{ "match": "wallBottom", "T": {{ "type": "zeroGradient" }} }},
        {{ "match": "wallTop",    "T": {{ "type": "zeroGradient" }} }},
        {{ "match": "wallFront",  "T": {{ "type": "empty" }} }},
        {{ "match": "wallBack",   "T": {{ "type": "empty" }} }}
      ]
    }}
  ],
  "interfaces": [
    {{ "regionA": "air", "patchA": "airToWall",
       "regionB": "wall", "patchB": "wallToAir" }}
  ],
  "initial": {{ "T": 300.0 }},
  "run": {{ "steady": true }},
  "numerics": {{
    "solver": "PCG", "preconditioner": "DIC",
    "tolerance": 1e-30, "maxIter": 4000
  }}
}}"#,
        n - n_solid
    )
}

#[test]
fn pair_the_solid_conductivity_changes_the_answer() {
    let Some(gpu) = gpu() else { return };
    // SPEC-LIT §47.12 Gate 5's own parameter: the conductivity ratio.
    pair_differs(
        &gpu,
        r#""material": { "rho": 1.0, "c": 1.0, "kappa": 1.0 }"#,
        r#""material": { "rho": 1.0, "c": 1.0, "kappa": 10.0 }"#,
        "the solid conductivity (the conductivity ratio Kr)",
    );
}

#[test]
fn pair_the_body_force_and_the_reference_temperature_change_the_answer() {
    let Some(gpu) = gpu() else { return };
    pair_differs(&gpu, r#""g": [0.0, -2.13e7, 0.0]"#, r#""g": [0.0, -2.13e6, 0.0]"#, "buoyancy/g");
    pair_differs(&gpu, r#""TRef": 300.0"#, r#""TRef": 600.0"#, "buoyancy/TRef");
}

#[test]
fn pair_every_fluid_property_changes_the_answer() {
    let Some(gpu) = gpu() else { return };
    let base = r#""fluid": { "rho": 1.0, "cp": 1.0, "kappa": 1.0, "mu": 0.71 }"#;
    for (to, what) in [
        (r#""fluid": { "rho": 1.0, "cp": 1.0, "kappa": 1.0, "mu": 1.42 }"#, "fluid/mu"),
        (r#""fluid": { "rho": 1.0, "cp": 1.0, "kappa": 2.0, "mu": 0.71 }"#, "fluid/kappa"),
        (r#""fluid": { "rho": 2.0, "cp": 1.0, "kappa": 1.0, "mu": 0.71 }"#, "fluid/rho"),
        (r#""fluid": { "rho": 1.0, "cp": 2.0, "kappa": 1.0, "mu": 0.71 }"#, "fluid/cp"),
    ] {
        pair_differs(&gpu, base, to, what);
    }
}

#[test]
fn pair_the_contact_resistance_changes_the_answer() {
    let Some(gpu) = gpu() else { return };
    pair_differs(
        &gpu,
        r#""patchB": "wallToAir" }"#,
        r#""patchB": "wallToAir", "Rc": 0.05 }"#,
        "the interface contact resistance Rc",
    );
}

/// A relaxation factor that reached nothing would leave the SEQUENCE of
/// iterates identical, so the two runs are stopped at a fixed count rather
/// than at a residual - the only way this pair can be tested at all.
#[test]
fn pair_the_relaxation_factor_changes_the_iterate() {
    let Some(gpu) = gpu() else { return };
    pair_differs(&gpu, r#""relaxU": 0.7"#, r#""relaxU": 0.4"#, "numerics/flow/relaxU");
}

/// SPEC-LIT §60.4's note on pairs 7 and 8, made a test rather than a claim.
///
/// `rho` and `cp` enter (S59.1) only through `nu = mu/rho`, through
/// `alpha = kappa/(rho cp)` and through the product `cp rho_f` that multiplies
/// the flux. Change all three of `rho`, `cp` and `mu` together so that every
/// one of those is unchanged, and the answer must not move - which is what
/// makes pairs 7 and 8 evidence rather than an accident of scaling: they move
/// the answer because they move a dimensionless group, not because the reader
/// happened to pass the number through.
#[test]
fn changing_rho_cp_and_mu_together_at_fixed_nu_alpha_and_rho_cp_leaves_the_answer() {
    let Some(gpu) = gpu() else { return };
    let a = kp_pair_base();
    // rho 1 -> 2, cp 1 -> 0.5, mu 0.71 -> 1.42: nu = mu/rho stays 0.71,
    // alpha = kappa/(rho cp) stays 1, and cp*rho stays 1.
    let b = a.replace(
        r#""fluid": { "rho": 1.0, "cp": 1.0, "kappa": 1.0, "mu": 0.71 }"#,
        r#""fluid": { "rho": 2.0, "cp": 0.5, "kappa": 1.0, "mu": 1.42 }"#,
    );
    assert_ne!(a, b);
    let sa = run_flow(&gpu, &a);
    let sb = run_flow(&gpu, &b);
    let gap = sa
        .t
        .iter()
        .zip(&sb.t)
        .fold(0.0 as Scalar, |m, (x, y)| m.max((x - y).abs()));
    // Not bitwise: the two runs multiply different constants in different
    // orders. Round-off on a 300 K field over 600 iterations.
    assert!(
        gap < 1e-9,
        "holding nu, alpha and rho*cp fixed while moving rho, cp and mu moved \
         the temperature field by {gap} K. Those three numbers enter the \
         equations ONLY through those three groups (SPEC-LIT S59.1), so \
         something else is reading one of them"
    );
}

// ==========================================================================
//  SPEC-LIT §79 - the openings: what a forced-convection fluid region says,
//  what is refused, and the §13.4.1 pair tests
//
//  §60.2's fluid region was a CLOSED CAVITY and §60.6 recorded what that
//  cost. The rig below is the smallest thing that is not one: a plane channel
//  with an inlet and an outlet, one conducting wall above it, heated from
//  outside. 288 cells, so every pair test RUNS.
// ==========================================================================

/// A heated plane duct: fluid below, one conducting wall above, flow left to
/// right. Every entry a pair test varies is a parameter, so the two documents
/// of a pair differ in exactly one substring.
#[allow(clippy::too_many_arguments)]
fn duct_case(
    inlet_kind: &str,
    inlet_u: &str,
    inlet_t: &str,
    outlet_kind: &str,
    outlet_t: &str,
    buoyancy: &str,
) -> String {
    format!(
        r#"{{
  "name": "heatedDuct",
  "regions": [
    {{
      "name": "water", "kind": "fluid",
      "mesh": {{
        "bounds": {{ "min": [0.0, 0.0, 0.0], "max": [4.0e-3, 4.0e-4, 1.0e-4] }},
        "cells": [24, 8, 1],
        "boundaries": {{
          "xmin": "west",       "xmax": "east",
          "ymin": "floor",      "ymax": "waterToWall",
          "zmin": "waterFront", "zmax": "waterBack"
        }}
      }},
      "fluid": {{ "rho": 1000.0, "cp": 4000.0, "kappa": 0.6, "mu": 1.0e-3 }},
      "patches": [
        {{ "match": "west", "kind": "{inlet_kind}"{inlet_u}, "T": {inlet_t} }},
        {{ "match": "east", "kind": "{outlet_kind}", "T": {outlet_t} }},
        {{ "match": "floor",      "T": {{ "type": "zeroGradient" }} }},
        {{ "match": "waterFront", "T": {{ "type": "empty" }} }},
        {{ "match": "waterBack",  "T": {{ "type": "empty" }} }}
      ]
    }},
    {{
      "name": "lid", "kind": "solid",
      "mesh": {{
        "bounds": {{ "min": [0.0, 4.0e-4, 0.0], "max": [4.0e-3, 6.0e-4, 1.0e-4] }},
        "cells": [24, 4, 1],
        "boundaries": {{
          "xmin": "lidWest",  "xmax": "lidEast",
          "ymin": "wallToWater", "ymax": "heated",
          "zmin": "lidFront", "zmax": "lidBack"
        }}
      }},
      "material": {{ "rho": 2000.0, "c": 700.0, "kappa": 100.0 }},
      "patches": [
        {{ "match": "lidWest",  "T": {{ "type": "zeroGradient" }} }},
        {{ "match": "lidEast",  "T": {{ "type": "zeroGradient" }} }},
        {{ "match": "heated",   "T": {{ "type": "fixedFluxTemperature", "q": 1.0e4 }} }},
        {{ "match": "lidFront", "T": {{ "type": "empty" }} }},
        {{ "match": "lidBack",  "T": {{ "type": "empty" }} }}
      ]
    }}
  ],
  "interfaces": [
    {{ "regionA": "water", "patchA": "waterToWall",
       "regionB": "lid",   "patchB": "wallToWater" }}
  ],{buoyancy}
  "initial": {{ "T": 300.0 }},
  "run": {{ "steady": true, "iterations": 400 }},
  "numerics": {{
    "solver": "PBiCGStab", "preconditioner": "DILU",
    "tolerance": 1e-16, "maxIter": 300,
    "flow": {{
      "relaxU": 0.7, "relaxP": 0.3, "relaxT": 1.0,
      "divSchemeU": "Gauss linear", "divSchemeT": "Gauss linear",
      "residual": 1e-10,
      "uTolerance": 1e-14, "pTolerance": 1e-14,
      "uMaxIter": 150, "pMaxIter": 400
    }}
  }}
}}"#
    )
}

const DUCT_U: &str = r#", "U": [0.05, 0.0, 0.0]"#;
const DUCT_TIN: &str = r#"{ "type": "fixedValue", "value": 300.0 }"#;
const DUCT_TOUT: &str = r#"{ "type": "inletOutlet", "inletValue": 300.0 }"#;
const DUCT_BUOY: &str = "\n  \"buoyancy\": { \"g\": [0.0, -9.81, 0.0], \"TRef\": 300.0 },";

/// The forward-flowing rig every §79 pair test starts from.
fn duct_base() -> String {
    duct_case("inlet", DUCT_U, DUCT_TIN, "outlet", DUCT_TOUT, "")
}

/// The same rig with the inlet velocity REVERSED, so the flow leaves through
/// `west` and comes back in through every face of `east`. SPEC-LIT §79.5:
/// this is the configuration in which `inletValue` is read at all.
fn duct_reversed() -> String {
    duct_case(
        "inlet",
        r#", "U": [-0.05, 0.0, 0.0]"#,
        DUCT_TIN,
        "outlet",
        DUCT_TOUT,
        "",
    )
}

/// The largest temperature difference between two runs of the same rig.
fn field_gap(
    a: &crate::cht::flow::ChtFlowSolution,
    b: &crate::cht::flow::ChtFlowSolution,
) -> Scalar {
    a.t.iter()
        .zip(&b.t)
        .fold(0.0 as Scalar, |m, (x, y)| m.max((x - y).abs()))
}

fn duct_pair_differs(gpu: &Gpu, base: &str, from: &str, to: &str, what: &str) {
    let b = base.replace(from, to);
    assert_ne!(base, b, "the pair test's own substitution '{from}' -> '{to}' changed nothing");
    let gap = field_gap(&run_flow(gpu, base), &run_flow(gpu, &b));
    assert!(
        gap > 1e-12,
        "changing {what} moved the temperature field by {gap} K. Two cases \
         differing in one entry produced the SAME answer, which means the case \
         said {what} and the solver ignored it (SPEC-LIT 13.4.1)"
    );
}

#[test]
fn the_shipped_qu_mudawar_case_reads_and_lowers_as_a_forced_flow_case() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../cases/quMudawar.cht.jsonc");
    let case = super::read_cht_case(&path).expect("read");
    let low = case.lower().expect("lower");

    // Nine boxes, one of them the channel, and it is region 0 (SPEC-LIT
    // §47.4's numbering invariant).
    assert_eq!(low.kinds().len(), 9);
    assert_eq!(low.kinds()[0], RegionKind::Fluid);
    assert!(low.kinds()[1..].iter().all(|k| *k == RegionKind::Solid));
    // Twelve conformal couples: six in y, six in z (SPEC-LIT §79.8).
    assert_eq!(low.interfaces.len(), 12);

    let flow = low.flow_case().expect("a forced case lowers to a FlowCase");
    // SPEC-LIT §79.6: forced convection, so NO body force at all.
    assert!(flow.buoyancy.is_none(), "Qu & Mudawar assumption (6)");
    let o = flow.openings.as_ref().expect("one inlet and one outlet");
    assert_eq!(o.inlet_patch, "inlet");
    assert_eq!(o.outlet_patch, "outlet");

    // Re = rho u d_h/mu at the paper's Table 2 value of 140, from the case's
    // own four numbers and Table 1's channel - the arithmetic the case
    // comment states, checked rather than trusted.
    let f = flow.regions[0].fluid.as_ref().expect("fluid");
    let (w_ch, h_ch) = (57.0e-6 as Scalar, 180.0e-6 as Scalar);
    let d_h = 4.0 * (w_ch * h_ch) / (2.0 * (w_ch + h_ch));
    let re = f.rho * o.inlet_velocity.x * d_h / f.mu;
    assert!((re - 140.0).abs() < 1e-9, "Re = {re}, Qu & Mudawar Table 2 says 140");
    let pr = f.pr();
    assert!((pr - 6.869_449_180_327_87).abs() < 1e-9, "Pr = {pr}");
}

/// SPEC-LIT §79.10, every refusal, in both directions. Each is a setting the
/// document could carry and the solver would ignore, or one the solver needs
/// and no default may be invented for.
#[test]
fn every_opening_refusal_fires_and_names_the_setting() {
    let base = duct_base();
    let zero_grad = r#"{ "type": "zeroGradient" }"#;
    let cases: Vec<(String, &str, &str)> = vec![
        // A velocity nothing reads.
        (
            base.replace(
                r#"{ "match": "floor",      "T": { "type": "zeroGradient" } }"#,
                r#"{ "match": "floor", "U": [1.0, 0.0, 0.0], "T": { "type": "zeroGradient" } }"#,
            ),
            "would ignore",
            "a velocity on a wall patch",
        ),
        // An inlet with no velocity.
        (
            duct_case("inlet", "", DUCT_TIN, "outlet", DUCT_TOUT, ""),
            "an `inlet` needs `U`",
            "an inlet with no velocity",
        ),
        // An inlet through which nothing enters.
        (
            duct_case("inlet", r#", "U": [0.0, 0.0, 0.0]"#, DUCT_TIN, "outlet", DUCT_TOUT, ""),
            "is zero",
            "a zero inlet velocity",
        ),
        // `inletOutlet` where the flux cannot change sign.
        (
            base.replace(
                r#"{ "match": "floor",      "T": { "type": "zeroGradient" } }"#,
                r#"{ "match": "floor", "T": { "type": "inletOutlet", "inletValue": 300.0 } }"#,
            ),
            "SIGN of the face flux",
            "`inletOutlet` on a wall",
        ),
        // An inlet whose entering enthalpy is undetermined.
        (
            duct_case("inlet", DUCT_U, zero_grad, "outlet", DUCT_TOUT, ""),
            "carries `fixedValue`",
            "an inlet with no temperature",
        ),
        // A held temperature at an outlet.
        (
            duct_case("inlet", DUCT_U, DUCT_TIN, "outlet", DUCT_TIN, ""),
            "carries `inletOutlet` or `zeroGradient`",
            "a fixed temperature at an outlet",
        ),
        // An inlet with no outlet.
        (
            duct_case("inlet", DUCT_U, DUCT_TIN, "wall", zero_grad, ""),
            "exactly one of each",
            "an inlet with no outlet",
        ),
        // An outlet with no inlet.
        (
            duct_case("wall", "", DUCT_TIN, "outlet", DUCT_TOUT, ""),
            "exactly one of each",
            "an outlet with no inlet",
        ),
        // A third answer to what a patch is.
        (
            duct_case("inlet", DUCT_U, DUCT_TIN, "farfield", zero_grad, ""),
            "available: wall, inlet, outlet",
            "an unknown patch kind",
        ),
        // An opening on a solid region.
        (
            base.replace(
                r#"{ "match": "lidWest",  "T": { "type": "zeroGradient" } }"#,
                r#"{ "match": "lidWest", "kind": "outlet", "T": { "type": "zeroGradient" } }"#,
            ),
            "is SOLID",
            "an opening on a conducting solid",
        ),
        // An `empty` plane that is also an opening.
        (
            base.replace(
                r#"{ "match": "waterFront", "T": { "type": "empty" } }"#,
                r#"{ "match": "waterFront", "kind": "outlet", "T": { "type": "empty" } }"#,
            ),
            "no surface integral",
            "an `empty` opening",
        ),
    ];
    for (text, needle, what) in cases {
        let err = read(&text)
            .and_then(|c| c.lower())
            .err()
            .unwrap_or_else(|| panic!("{what} was ACCEPTED"))
            .to_string();
        assert!(err.contains(needle), "{what}: the message does not name it - {err}");
    }
}

/// SPEC-LIT §79.6, both directions. `buoyancy` is REQUIRED by a closed cavity,
/// which has nothing else that could drive it, and OPTIONAL once the case
/// names an inlet.
#[test]
fn buoyancy_is_required_by_a_closed_cavity_and_optional_once_there_is_an_inlet() {
    // Forced, no buoyancy: accepted.
    read(&duct_base())
        .expect("parse")
        .lower()
        .expect("a forced case may omit `buoyancy`");

    // Forced, with buoyancy: also accepted - the two are a §13.4.1 pair
    // below, not an either/or.
    read(&duct_case("inlet", DUCT_U, DUCT_TIN, "outlet", DUCT_TOUT, DUCT_BUOY))
        .expect("parse")
        .lower()
        .expect("a forced case may carry `buoyancy`");

    // Closed, no buoyancy: refused, and the message names the way out.
    let closed = duct_case(
        "wall",
        "",
        DUCT_TIN,
        "wall",
        r#"{ "type": "zeroGradient" }"#,
        "",
    );
    let err = read(&closed)
        .expect("parse")
        .lower()
        .expect_err("a closed cavity with no body force has nothing to drive it")
        .to_string();
    assert!(err.contains("CLOSED fluid cavity needs `buoyancy`"), "{err}");
    assert!(err.contains("inlet"), "the refusal names the alternative: {err}");
}

/// SPEC-LIT §79.11 pair 1. The inlet velocity is this case's whole forcing.
#[test]
fn pair_the_inlet_velocity_changes_the_answer() {
    let Some(gpu) = gpu() else { return };
    duct_pair_differs(
        &gpu,
        &duct_base(),
        r#""U": [0.05, 0.0, 0.0]"#,
        r#""U": [0.1, 0.0, 0.0]"#,
        "the inlet velocity",
    );
}

/// SPEC-LIT §79.11 pair 2. The inlet temperature is the enthalpy datum
/// everything downstream is measured from.
#[test]
fn pair_the_inlet_temperature_changes_the_answer() {
    let Some(gpu) = gpu() else { return };
    duct_pair_differs(
        &gpu,
        &duct_base(),
        r#"{ "type": "fixedValue", "value": 300.0 }"#,
        r#"{ "type": "fixedValue", "value": 320.0 }"#,
        "the inlet temperature",
    );
}

/// SPEC-LIT §79.11 pair 3. `buoyancy` on a forced case is not decoration: it
/// switches on §9's body force AND §25's variable density, and §79.6 says the
/// absence of it is a model and not a default.
#[test]
fn pair_buoyancy_on_a_forced_case_changes_the_answer() {
    let Some(gpu) = gpu() else { return };
    let a = run_flow(&gpu, &duct_base());
    let b = run_flow(
        &gpu,
        &duct_case("inlet", DUCT_U, DUCT_TIN, "outlet", DUCT_TOUT, DUCT_BUOY),
    );
    let gap = field_gap(&a, &b);
    assert!(
        gap > 1e-12,
        "adding `buoyancy` to a forced case moved the temperature field by \
         {gap} K - SPEC-LIT 79.6 says it must move both the body force and \
         the density"
    );
}

/// SPEC-LIT §79.11 pair 4, and the honest half of it.
///
/// `inletOutlet`'s `inletValue` is read on exactly the faces where
/// `phi_b < 0`. On a channel that never backflows there are none, so the
/// entry CANNOT move the answer - and this test asserts that equivalent claim
/// (§70.7's precedent) together with the run's own report that no face fired.
/// Reverse the inlet velocity and every outlet face is an inflow face, and
/// then the same entry moves the answer by a lot.
#[test]
fn pair_the_outlet_inlet_value_moves_nothing_until_the_flow_comes_back_in() {
    let Some(gpu) = gpu() else { return };

    // ---- forward: it cannot move the answer, and the run says so ----------
    let a = duct_base();
    let b = a.replace(r#""inletValue": 300.0"#, r#""inletValue": 900.0"#);
    assert_ne!(a, b);
    let sa = run_flow(&gpu, &a);
    let sb = run_flow(&gpu, &b);
    let oa = sa.openings.expect("openings");
    assert!(oa.n_outlet_faces > 0);
    assert_eq!(
        oa.n_backflow, 0,
        "the forward duct backflowed on {} of {} outlet faces, so this half of \
         the pair test is not testing what it says it is",
        oa.n_backflow, oa.n_outlet_faces
    );
    let gap = field_gap(&sa, &sb);
    assert_eq!(
        gap, 0.0,
        "with nothing entering through the outlet, `inletValue` is read by no \
         face at all, so the two answers must agree in every bit and they \
         differ by {gap} K"
    );

    // ---- reversed: every outlet face is an inflow face --------------------
    let c = duct_reversed();
    let d = c.replace(r#""inletValue": 300.0"#, r#""inletValue": 900.0"#);
    let sc = run_flow(&gpu, &c);
    let sd = run_flow(&gpu, &d);
    let oc = sc.openings.expect("openings");
    assert!(
        oc.n_backflow > 0,
        "the reversed duct was supposed to make the flow re-enter through the \
         outlet and did not"
    );
    let gap = field_gap(&sc, &sd);
    assert!(
        gap > 1.0,
        "with the flow coming back in, `inletValue` 300 -> 900 moved the field \
         by only {gap} K"
    );
}

/// SPEC-LIT §79.5's other claim: while the flow is leaving, `inletOutlet` is
/// `zeroGradient` in every bit - not approximately, and not eventually.
#[test]
fn inlet_outlet_is_bitwise_zero_gradient_while_the_flow_leaves() {
    let Some(gpu) = gpu() else { return };
    let a = duct_base();
    let b = duct_case(
        "inlet",
        DUCT_U,
        DUCT_TIN,
        "outlet",
        r#"{ "type": "zeroGradient" }"#,
        "",
    );
    assert_ne!(a, b);
    let sa = run_flow(&gpu, &a);
    let sb = run_flow(&gpu, &b);
    assert_eq!(sa.openings.expect("openings").n_backflow, 0);
    let gap = field_gap(&sa, &sb);
    assert_eq!(gap, 0.0, "the two outflow conditions differ by {gap} K");
    let gap_b = sa
        .bt
        .iter()
        .zip(&sb.bt)
        .fold(0.0 as Scalar, |m, (x, y)| m.max((x - y).abs()));
    assert_eq!(gap_b, 0.0, "the evaluated FACE values differ by {gap_b} K");
}

/// SPEC-LIT §79.7: what goes in comes out, and the bulk temperature rise is
/// `Q/(m cp)` - an identity, not a correlation.
#[test]
fn the_openings_close_the_global_balance_and_the_bulk_rise_is_the_identity() {
    let Some(gpu) = gpu() else { return };
    let sol = run_flow(&gpu, &duct_base());
    let o = sol.openings.expect("openings");

    // Both fluxes signed OUTWARD, so they cancel.
    let imbalance = o.imbalance().abs() / o.outlet_flux.abs();
    assert!(imbalance < 1e-10, "global mass imbalance {imbalance} relative");

    // The heat the case put in, and it is THREE terms rather than the two the
    // obvious reading gives. The heater is one. The inlet is the other: `T`
    // is held at 300 K there while the first cell is warmer, so that face
    // conducts heat back OUT of the domain, and on a duct this short it is
    // 0.3 % of the heater - small, and thirty times the tolerance a balance
    // deserves. It is in the balance because it is in the physics; leaving it
    // out and loosening the tolerance instead would be hiding it.
    let q_heater = sol.patch_heat_flow(1, "heated").expect("heated patch");
    let q_inlet = sol.patch_heat_flow(0, "west").expect("inlet patch");
    assert!(q_inlet < 0.0, "the inlet conducts heat OUT, not in: {q_inlet} W");
    let q_in = q_heater + q_inlet;
    let rel = ((o.enthalpy_rise - q_in) / q_in).abs();
    assert!(
        rel < 1e-5,
        "the flow carried out {} W against the {q_in} W that entered          ({q_heater} through the heater, {q_inlet} through the inlet), {rel}          relative",
        o.enthalpy_rise
    );

    // dT_bulk = Q/(m cp), with m cp = rho cp |phi_outlet|.
    let m_cp = sol.fluid_rho_cp * o.outlet_flux.abs();
    let want = 300.0 + q_in / m_cp;
    let err = (o.outlet_bulk_t - want).abs();
    assert!(
        err < 1e-5,
        "outlet bulk T {} against the identity {want}",
        o.outlet_bulk_t
    );
}

/// SPEC-LIT §79.8. Nine boxes of ONE material, joined by the twelve couples a
/// micro-channel unit cell needs, against the single box they were cut out
/// of. A perfect-contact interface between two cells of the same material on
/// a matched orthogonal mesh IS the internal-face coefficient it replaced, so
/// the cut is not an approximation - and this test says by how much.
#[test]
fn the_nine_box_decomposition_is_the_single_box_it_was_cut_from() {
    let Some(gpu) = gpu() else { return };

    let mono = r#"{
  "name": "monolith",
  "regions": [
    { "name": "block",
      "mesh": {
        "bounds": { "min": [0.0, 0.0, 0.0], "max": [2.0, 3.0, 3.0] },
        "cells": [4, 6, 9],
        "boundaries": { "xmin": "w", "xmax": "e", "ymin": "s", "ymax": "n",
                        "zmin": "bottom", "zmax": "top" } },
      "material": { "rho": 2330.0, "c": 712.0, "kappa": 148.0 },
      "patches": [
        { "match": "w", "T": { "type": "zeroGradient" } },
        { "match": "e", "T": { "type": "zeroGradient" } },
        { "match": "s", "T": { "type": "zeroGradient" } },
        { "match": "n", "T": { "type": "zeroGradient" } },
        { "match": "bottom", "T": { "type": "fixedValue", "value": 300.0 } },
        { "match": "top",    "T": { "type": "fixedFluxTemperature", "q": 5.0e3 } } ] } ],
  "initial": { "T": 300.0 },
  "run": { "steady": true },
  "numerics": { "solver": "PCG", "preconditioner": "DIC",
                "tolerance": 1e-18, "maxIter": 5000 }
}"#;

    let tag = [
        ["botL", "botC", "botR"],
        ["midL", "midC", "midR"],
        ["topL", "topC", "topR"],
    ];
    let mut regions = Vec::new();
    for (j, tj) in tag.iter().enumerate() {
        for (i, n) in tj.iter().enumerate() {
            let mut pats = vec![
                format!(r#"{{ "match": "{n}Xmin", "T": {{ "type": "zeroGradient" }} }}"#),
                format!(r#"{{ "match": "{n}Xmax", "T": {{ "type": "zeroGradient" }} }}"#),
            ];
            if i == 0 {
                pats.push(format!(
                    r#"{{ "match": "{n}Ymin", "T": {{ "type": "zeroGradient" }} }}"#
                ));
            }
            if i == 2 {
                pats.push(format!(
                    r#"{{ "match": "{n}Ymax", "T": {{ "type": "zeroGradient" }} }}"#
                ));
            }
            if j == 0 {
                pats.push(format!(
                    r#"{{ "match": "{n}Zmin", "T": {{ "type": "fixedValue", "value": 300.0 }} }}"#
                ));
            }
            if j == 2 {
                pats.push(format!(
                    r#"{{ "match": "{n}Zmax", "T": {{ "type": "fixedFluxTemperature", "q": 5.0e3 }} }}"#
                ));
            }
            regions.push(format!(
                r#"    {{ "name": "{n}",
      "mesh": {{
        "bounds": {{ "min": [0.0, {}.0, {}.0], "max": [2.0, {}.0, {}.0] }},
        "cells": [4, 2, 3],
        "boundaries": {{ "xmin": "{n}Xmin", "xmax": "{n}Xmax",
                        "ymin": "{n}Ymin", "ymax": "{n}Ymax",
                        "zmin": "{n}Zmin", "zmax": "{n}Zmax" }} }},
      "material": {{ "rho": 2330.0, "c": 712.0, "kappa": 148.0 }},
      "patches": [ {} ] }}"#,
                i,
                j,
                i + 1,
                j + 1,
                pats.join(", ")
            ));
        }
    }
    let mut ifaces = Vec::new();
    for tj in &tag {
        for i in 0..2 {
            let (a, b) = (tj[i], tj[i + 1]);
            ifaces.push(format!(
                r#"    {{ "regionA": "{a}", "patchA": "{a}Ymax", "regionB": "{b}", "patchB": "{b}Ymin" }}"#
            ));
        }
    }
    for j in 0..2 {
        for (a, b) in tag[j].iter().zip(&tag[j + 1]) {
            ifaces.push(format!(
                r#"    {{ "regionA": "{a}", "patchA": "{a}Zmax", "regionB": "{b}", "patchB": "{b}Zmin" }}"#
            ));
        }
    }
    assert_eq!(ifaces.len(), 12);
    let split = format!(
        r#"{{
  "name": "nineBox",
  "regions": [
{}
  ],
  "interfaces": [
{}
  ],
  "initial": {{ "T": 300.0 }},
  "run": {{ "steady": true }},
  "numerics": {{ "solver": "PCG", "preconditioner": "DIC",
                "tolerance": 1e-18, "maxIter": 5000 }}
}}"#,
        regions.join(",\n"),
        ifaces.join(",\n")
    );

    let a = solve(&gpu, mono);
    let b = solve(&gpu, &split);
    assert_eq!(a.mesh.host.n_cells, b.mesh.host.n_cells);
    assert_eq!(b.mesh.interface_ranges.len(), 12);

    // Match cells by centroid: the two meshes number them differently, which
    // is the whole point of the concatenation.
    let key = |p: crate::Vec3| {
        (
            (f64::from(p.x) * 1e9).round() as i64,
            (f64::from(p.y) * 1e9).round() as i64,
            (f64::from(p.z) * 1e9).round() as i64,
        )
    };
    let map: std::collections::HashMap<_, _> = b
        .mesh
        .host
        .c
        .iter()
        .enumerate()
        .map(|(c, p)| (key(*p), b.t[c]))
        .collect();
    let mut worst: Scalar = 0.0;
    let mut span: Scalar = 0.0;
    for (c, p) in a.mesh.host.c.iter().enumerate() {
        let t = *map.get(&key(*p)).unwrap_or_else(|| panic!("cell {c} unmatched"));
        worst = worst.max((t - a.t[c]).abs());
        span = span.max((a.t[c] - 300.0).abs());
    }
    let rel = worst / span;
    assert!(
        rel < 1e-11,
        "the nine-box decomposition differs from the box it was cut from by \
         {worst} K over a {span} K range ({rel} relative). SPEC-LIT 79.8 says \
         a perfect-contact couple between two cells of the SAME material on a \
         matched orthogonal mesh is the internal-face coefficient it replaced"
    );
    println!(
        "SPEC-LIT 79.8: nine boxes against one, {worst} K over {span} K = {rel} relative"
    );
}
