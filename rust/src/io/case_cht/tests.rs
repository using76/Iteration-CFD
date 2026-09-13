// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
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

/// `lower()` keeps one raw polyMesh per region, bitwise what the matching
/// `meshes[r]` was built from: `build_mesh` IS `build_host_mesh(&raw_mesh(b)?)`,
/// so the counts must agree exactly - the point set and face polygons
/// `HostMesh` does not keep travel with the case (SPEC-LIT §49.3's pattern).
#[test]
fn lowering_keeps_the_raw_mesh_of_every_region() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../cases/dieStack.cht.jsonc");
    let case = super::read_cht_case(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let low = case.lower().unwrap_or_else(|e| panic!("{}: {e}", path.display()));

    assert_eq!(low.raw.len(), low.meshes.len(), "one raw mesh per region");
    for (r, (raw, m)) in low.raw.iter().zip(&low.meshes).enumerate() {
        assert_eq!(raw.points.len(), m.n_points, "region {r} point count");
        assert_eq!(
            raw.faces.len(),
            m.n_internal_faces + m.n_boundary_faces,
            "region {r} face count"
        );
        assert_eq!(
            raw.neighbour.len(),
            m.n_internal_faces,
            "region {r} internal-face count"
        );
    }
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

/// SPEC-LIT §60.3: a volumetric source on a fluid region used to be refused,
/// while §18's registry was not wired to this format's fluid side. It is
/// wired now: the source lowers into the `FlowCase`, reaches `EnergySources`
/// by the same registration a solid region's does, and the §13.4.1 pair test
/// applies unchanged - a fluid that dissipates must be hotter than one that
/// does not.
#[test]
fn a_source_on_a_fluid_region_reaches_the_energy_balance() {
    crate::io::contract::reset_warnings();
    let Some(gpu) = gpu() else { return };

    let text = kp_pair_base().replace(
        r#""fluid": { "rho": 1.0, "cp": 1.0, "kappa": 1.0, "mu": 0.71 },"#,
        r#""fluid": { "rho": 1.0, "cp": 1.0, "kappa": 1.0, "mu": 0.71 },
      "source": 1000.0,"#,
    );
    assert_ne!(text, kp_pair_base(), "the substitution must actually have matched");

    // Accepted, and carried all the way into the FlowCase.
    let low = read(&text).expect("parse").lower().expect("a fluid source lowers");
    assert_eq!(low.sources[0], 1000.0, "the fluid region's own source entry");
    let case = low.flow_case().expect("a fluid case lowers to a FlowCase");
    assert_eq!(case.regions[0].source, 1000.0);

    let sa = run_flow(&gpu, &kp_pair_base());
    let sb = run_flow(&gpu, &text);
    let gap = sa
        .t
        .iter()
        .zip(&sb.t)
        .fold(0.0 as Scalar, |m, (x, y)| m.max((x - y).abs()));
    assert!(
        gap > 1e-12,
        "a 1000 W/m^3 source on the fluid changed the field by only {gap} K - the \
         case said source and the solver ignored it (SPEC-LIT 13.4.1)"
    );
    assert!(
        sb.region_mean(0) > sa.region_mean(0),
        "a heater on the fluid region must make it HOTTER, not merely different: \
         {} K against {} K",
        sb.region_mean(0),
        sa.region_mean(0)
    );
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

// ==========================================================================
//  SPEC-LIT §96 - the stress fixtures, shared with the driver unit
// ==========================================================================

/// One steel bar, clamped at x = 0, held at 300 K there and 400 K at x = 0.1
/// (a linear T), TRef 300: every argument is a &str substituted ONCE, so two
/// documents of a pair differ in one substring.
#[allow(clippy::too_many_arguments)]
fn stress_block(
    mech_material: &str,
    ymin_u: &str,
    traction_x: &str,
    clamp_ux: &str,
    tol: &str,
    max_outer: &str,
    mech_extra: &str,
    mode: &str,
    case_extra: &str,
) -> String {
    format!(
        r#"{{
  // A bar heating from 300 K at the clamp to 400 K at the tip; TRef 300.
  "name": "bar",
  "regions": [
    {{
      "name": "bar",
      "kind": "solid",
      "mesh": {{
        "bounds": {{ "min": [0.0, 0.0, 0.0], "max": [0.1, 0.02, 0.02] }},
        "cells": [10, 4, 4],
        "boundaries": {{
          "xmin": "clamp", "xmax": "hot",
          "ymin": "ymin", "ymax": "ymax",
          "zmin": "zmin", "zmax": "zmax"
        }}
      }},
      "material": {{ "rho": 7800.0, "c": 460.0, "kappa": 45.0 }},
      "patches": [
        {{ "match": "clamp", "T": {{ "type": "fixedValue", "value": 300.0 }} }},
        {{ "match": "hot",   "T": {{ "type": "fixedValue", "value": 400.0 }} }},
        {{ "match": "ymin",  "T": {{ "type": "zeroGradient" }} }},
        {{ "match": "ymax",  "T": {{ "type": "zeroGradient" }} }},
        {{ "match": "zmin",  "T": {{ "type": "zeroGradient" }} }},
        {{ "match": "zmax",  "T": {{ "type": "zeroGradient" }} }}
      ],
      "mechanics": {{
        {mech_material},
        "patches": [
          {{ "match": "clamp", "u": {{ "type": "fixedDisplacement", "value": [{clamp_ux}, 0.0, 0.0] }} }},
          {{ "match": "hot",   "u": {{ "type": "traction", "value": [{traction_x}, 0.0, 0.0] }} }},
          {{ "match": "ymin",  "u": {ymin_u} }},
          {{ "match": "ymax",  "u": {{ "type": "free" }} }},
          {{ "match": "zmin",  "u": {{ "type": "free" }} }},
          {{ "match": "zmax",  "u": {{ "type": "free" }} }}
        ],
        "solver": {{ "tolerance": {tol}, "maxOuter": {max_outer} }}{mech_extra}
      }}
    }}
  ],
  "initial": {{ "T": 350.0 }},
  "run": {{ "steady": true, "mode": "{mode}" }},
  "numerics": {{
    "solver": "PCG", "preconditioner": "DIC",
    "tolerance": 1e-12, "maxIter": 4000
  }}{case_extra}
}}"#
    )
}

/// The `material` entry of a stress document - the four numbers a pair test
/// turns, substituted ONCE.
fn steel(alpha: &str, e: &str, nu: &str, t_ref: &str) -> String {
    format!(r#""material": {{ "E": {e}, "nu": {nu}, "alpha": {alpha}, "TRef": {t_ref} }}"#)
}

/// The bonded two-zone `materials` list - copper from x = 0 to 0.05, steel
/// from 0.05 to 0.1, tiling the bar exactly (docs/10's R4).
fn two_zones() -> String {
    r#""materials": [
          { "name": "copper", "bounds": { "min": [0.0, 0.0, 0.0], "max": [0.05, 0.02, 0.02] },
            "material": { "E": 120e9, "nu": 0.3, "alpha": 17e-6, "TRef": 300.0 } },
          { "name": "steel", "bounds": { "min": [0.05, 0.0, 0.0], "max": [0.1, 0.02, 0.02] },
            "material": { "E": 200e9, "nu": 0.3, "alpha": 12e-6, "TRef": 300.0 } }
        ]"#
    .to_string()
}

/// `{ "type": "free" }`, spelled once so a pair test can swap it.
const FREE: &str = r#"{ "type": "free" }"#;

fn default_stress() -> String {
    stress_block(
        &steel("1.2e-5", "200e9", "0.3", "300.0"),
        FREE,
        "0.0",
        "0.0",
        "1e-8",
        "500",
        "",
        "stress",
        "",
    )
}

fn bond_block(bond: &str) -> String {
    stress_block(
        &two_zones(),
        FREE,
        "0.0",
        "0.0",
        "1e-8",
        "500",
        &format!(r#", "bond": "{bond}""#),
        "stress",
        "",
    )
}

/// Appends an `output` block before the document's closing brace - the
/// fixture's `{case_extra}` slot, reached by replace so the helper works on
/// any of the builders above.
fn with_output(doc: &str, output: &str) -> String {
    let text = doc.replace("  }\n}", &format!("  }},\n  \"output\": {output}\n}}"));
    assert_ne!(text, doc, "the output slot must exist");
    text
}

// ==========================================================================
//  SPEC-LIT §96 - the mechanics block reads and lowers
// ==========================================================================

/// The block reads, lowers, and every unknown key is a parse error naming
/// its JSON path (the `deny_unknown_fields` rule every format here runs
/// under).
#[test]
fn a_mechanics_block_reads_and_lowers() {
    let case = read(&default_stress()).expect("parse");
    let low = case.lower().expect("lower");
    assert_eq!(low.mechanics.len(), 1);
    let m = low.mechanics[0].as_ref().expect("bar carries mechanics");
    assert!(low.stress);
    assert_eq!(m.zones.len(), 1);
    assert_eq!(m.zones[0].name, "bar");
    assert_eq!(m.zones[0].material.e, 200.0e9);
    assert_eq!(m.zones[0].material.nu, 0.3);
    assert_eq!(m.zones[0].t_ref, 300.0);
    assert_eq!(m.zone_of_cell, vec![0; low.meshes[0].n_cells]);
    assert_eq!(m.patch_bcs.len(), 6);
    assert_eq!(m.solver.tolerance, 1e-8);
    assert_eq!(m.solver.max_outer, 500);

    // A mistyped key is a parse error naming the key, not a silent drop.
    let typo = default_stress().replace(r#""maxOuter": 500"#, r#""maxOutre": 500"#);
    assert_ne!(typo, default_stress(), "the substitution must change the text");
    let e = read(&typo).expect_err("must refuse");
    assert!(e.to_string().contains("maxOutre"), "{e}");
}

/// The thermal format is bitwise what it was: no `mechanics`, no `mode`, no
/// `output`, and the lowered case carries the None/false/None defaults.
#[test]
fn a_thermal_case_lowers_to_no_mechanics() {
    let low = read(&default_slab()).expect("parse").lower().expect("lower");
    assert_eq!(low.mechanics, vec![None, None]);
    assert!(!low.stress);
    assert!(low.output.is_none());
}

/// An `output` block lowers to the resolved plan: `vtu`, once, nothing else
/// named.
#[test]
fn an_output_block_lowers_to_a_vtu_plan() {
    let text = with_output(&default_stress(), r#"{ "exact": { "format": "vtu" } }"#);
    let low = read(&text).expect("parse").lower().expect("lower");
    let plan = low.output.expect("the block lowers to a plan");
    assert_eq!(
        plan.exact.expect("exact").formats,
        vec![crate::io::output_plan::OutputFormat::Vtu]
    );
    assert!(plan.vis.is_none());
    assert!(plan.restart.is_none());
}

/// The zones' materials went through `Material::validate` (rows 1-2 above)
/// and `bond` lowers to the S8 enum: `linear` spelled in the JSON, `series`
/// the default when the word is absent.
#[test]
fn bond_lowers_to_s8s_enum() {
    let low = read(&bond_block("linear"))
        .expect("parse")
        .lower()
        .expect("lower");
    let m = low.mechanics[0].as_ref().expect("mechanics");
    assert_eq!(m.bond, crate::solid::BondTreatment::Linear);
    assert_eq!(m.zones.len(), 2);
    assert_eq!(m.zones[0].name, "copper");
    assert_eq!(m.zones[0].material.e, 120.0e9);
    assert_eq!(m.zone_of_cell[0], 0);
    let series = read(&bond_block("series"))
        .expect("parse")
        .lower()
        .expect("lower");
    assert_eq!(
        series.mechanics[0].as_ref().expect("mechanics").bond,
        crate::solid::BondTreatment::Series
    );
    let absent = read(&default_stress()).expect("parse").lower().expect("lower");
    assert_eq!(
        absent.mechanics[0].as_ref().expect("mechanics").bond,
        crate::solid::BondTreatment::Series
    );
}

// ==========================================================================
//  SPEC-LIT §96.3 - the refusal list, rows 1-19
// ==========================================================================

/// Row 1: a bad elastic constant is refused naming the number, all three
/// ways - `E <= 0`, `nu` outside (-1, 0.5), and a negative `alpha`.
#[test]
fn a_bad_elastic_constant_is_refused_naming_the_number() {
    let cases = [
        (r#""E": 200e9"#, r#""E": -1.0"#, "-1"),
        (r#""nu": 0.3"#, r#""nu": 0.6"#, "0.6"),
        (r#""alpha": 1.2e-5"#, r#""alpha": -1e-5"#, "-1e-5"),
    ];
    for (from, to, number) in cases {
        let text = default_stress().replace(from, to);
        assert_ne!(text, default_stress(), "{from} -> {to} changed nothing");
        let e = read(&text).expect("parse").lower().expect_err("must refuse");
        let msg = e.to_string();
        assert!(msg.contains(number), "{msg}");
        assert!(msg.contains("mechanics/material"), "{msg}");
    }
}

/// Row 2: `nu` above the measured edge is S6's own refusal, which names the
/// block-coupled route.
#[test]
fn a_near_incompressible_solid_is_refused_naming_the_route() {
    let text = default_stress().replace(r#""nu": 0.3"#, r#""nu": 0.46"#);
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("0.46"), "{msg}");
    assert!(msg.contains("block-coupled"), "{msg}");
}

/// Row 3: `alpha > 0` without `TRef` - no reference, no thermal strain.
#[test]
fn alpha_without_tref_is_refused() {
    let text = default_stress().replace(r#", "TRef": 300.0"#, "");
    assert_ne!(text, default_stress(), "TRef must actually be dropped");
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("TRef"), "{msg}");
    assert!(msg.contains("no TRef, no strain"), "{msg}");
}

/// Row 4: `TRef` with `alpha == 0` - a reference nothing reads.
#[test]
fn tref_without_alpha_is_refused() {
    let text = default_stress().replace(r#""alpha": 1.2e-5"#, r#""alpha": 0.0"#);
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("nothing reads"), "{msg}");
    assert!(msg.contains("TRef"), "{msg}");
}

/// Row 5: `rho` in an ELASTIC material - the static solve reads no density.
#[test]
fn rho_in_mechanics_is_refused_naming_the_static_solve() {
    let text = default_stress().replace(
        r#""TRef": 300.0 }"#,
        r#""TRef": 300.0, "rho": 7800.0 }"#,
    );
    assert_ne!(text, default_stress());
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("static solve"), "{msg}");
    assert!(msg.contains("106b"), "{msg}");
    assert!(msg.contains("rho"), "{msg}");
}

/// Row 6: a `ddtScheme` on displacement is §95's inertia refusal, naming
/// Newmark and `rho_infinity`.
#[test]
fn a_ddt_scheme_on_displacement_is_refused_naming_newmark() {
    let text = default_stress().replace(
        r#""maxOuter": 500 }"#,
        r#""maxOuter": 500, "ddtScheme": "Newmark" }"#,
    );
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("Newmark"), "{msg}");
    assert!(msg.contains("rho_infinity"), "{msg}");
    assert!(msg.contains("ddtScheme"), "{msg}");
}

/// Row 7: a static relaxation factor, any value - the outer loop is Aitken
/// delta-squared and its first omega is 1.
#[test]
fn relaxation_in_mechanics_is_refused_naming_aitken() {
    let text = default_stress().replace(
        r#""maxOuter": 500 }"#,
        r#""maxOuter": 500, "relaxation": 0.7 }"#,
    );
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("Aitken"), "{msg}");
    assert!(msg.contains("relaxation"), "{msg}");
}

/// Row 8: `mechanics` on a fluid region.
#[test]
fn mechanics_on_a_fluid_region_is_refused() {
    let text = duct_base().replace(
        r#""fluid": { "rho": 1000.0, "cp": 4000.0, "kappa": 0.6, "mu": 1.0e-3 },"#,
        r#""fluid": { "rho": 1000.0, "cp": 4000.0, "kappa": 0.6, "mu": 1.0e-3 },
      "mechanics": {
        "material": { "E": 2e9, "nu": 0.3, "alpha": 1e-4, "TRef": 300.0 },
        "patches": [
          { "match": "west", "u": { "type": "fixedDisplacement", "value": [0.0, 0.0, 0.0] } },
          { "match": "east", "u": { "type": "free" } },
          { "match": "floor", "u": { "type": "free" } },
          { "match": "waterToWall", "u": { "type": "free" } }
        ]
      },"#,
    );
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("water"), "{msg}");
    assert!(msg.contains("mechanics"), "{msg}");
}

/// Row 9: exactly one spelling of the material - both, neither, and the
/// empty `materials` list that counts as neither.
#[test]
fn one_material_spelling_exactly() {
    let steel_mat = steel("1.2e-5", "200e9", "0.3", "300.0");
    let cases = [
        ("both", default_stress().replace(
            &steel_mat,
            r#""materials": [], "material": { "E": 200e9, "nu": 0.3, "alpha": 1.2e-5, "TRef": 300.0 }"#,
        )),
        ("neither", default_stress().replace(&steel_mat, r#""bond": "series""#)),
        ("empty list", default_stress().replace(&steel_mat, r#""materials": []"#)),
    ];
    for (what, text) in cases {
        let e = read(&text).expect("parse").lower().expect_err("must refuse");
        let msg = e.to_string();
        assert!(
            msg.to_lowercase().contains("exactly one") || msg.contains("neither"),
            "{what}: {msg}"
        );
        assert!(msg.contains("mechanics"), "{what}: {msg}");
    }
}

/// Row 10: a cell in no zone - the count and the first uncovered centroid.
#[test]
fn a_cell_in_no_zone_is_refused() {
    let text = bond_block("series").replace(
        r#""min": [0.05, 0.0, 0.0]"#,
        r#""min": [0.06, 0.0, 0.0]"#,
    );
    assert_ne!(text, bond_block("series"));
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("16 cells are in no zone"), "{msg}");
    assert!(msg.contains("centroid"), "{msg}");
}

/// Row 11: a cell in two zones - both zone names and the centroid.
#[test]
fn a_cell_in_two_zones_is_refused() {
    let text = bond_block("series").replace(
        r#""min": [0.05, 0.0, 0.0]"#,
        r#""min": [0.04, 0.0, 0.0]"#,
    );
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("copper") && msg.contains("steel"), "{msg}");
    assert!(msg.contains("exactly one zone"), "{msg}");
}

/// Row 12: `mechanics.patches` - a non-empty patch unnamed, named twice, a
/// name that is not in `mesh.boundaries`, and an `empty` patch named.
#[test]
fn every_mechanical_patch_is_named_exactly_once() {
    // (a) zmax's rule dropped: the patch carries no mechanical condition.
    let a = default_stress().replace(
        r#",
          { "match": "zmax",  "u": { "type": "free" } }"#,
        "",
    );
    assert_ne!(a, default_stress(), "the rule must actually be dropped");
    let e = read(&a).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("zmax"), "{msg}");
    assert!(msg.contains("no mechanical condition"), "{msg}");

    // (b) named twice: clamp's rule duplicated over zmin.
    let b = default_stress().replace(
        r#"{ "match": "zmin",  "u": { "type": "free" } }"#,
        r#"{ "match": "clamp", "u": { "type": "free" } }"#,
    );
    let e = read(&b).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("twice"), "{msg}");

    // (c) a name that is not in mesh.boundaries.
    let c = default_stress().replace(
        r#"{ "match": "zmin",  "u": { "type": "free" } }"#,
        r#"{ "match": "bottom", "u": { "type": "free" } }"#,
    );
    let e = read(&c).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("no patch 'bottom'"), "{msg}");

    // (d) an `empty` patch named: the z pair becomes `empty` on a one-cell
    // axis (blockgen refuses anything else), zmax's mechanical rule is
    // dropped, and `zmin` is named.
    let d = default_stress()
        .replace(r#""cells": [10, 4, 4]"#, r#""cells": [10, 4, 1]"#)
        .replace(
            r#"{ "match": "zmin",  "T": { "type": "zeroGradient" } },
        { "match": "zmax",  "T": { "type": "zeroGradient" } }"#,
            r#"{ "match": "zmin",  "T": { "type": "empty" } },
        { "match": "zmax",  "T": { "type": "empty" } }"#,
        )
        .replace(
            r#",
          { "match": "zmax",  "u": { "type": "free" } }"#,
            "",
        );
    assert_ne!(d, default_stress());
    let e = read(&d).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("contributes to no surface integral"), "{msg}");
    assert!(msg.contains("zmin"), "{msg}");
}

/// Row 13: `bond` needs two materials and a known treatment.
#[test]
fn bond_needs_two_materials_and_a_known_treatment() {
    let single = stress_block(
        &steel("1.2e-5", "200e9", "0.3", "300.0"),
        FREE,
        "0.0",
        "0.0",
        "1e-8",
        "500",
        r#", "bond": "series""#,
        "stress",
        "",
    );
    let e = read(&single).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("no bond face"), "{msg}");

    let e = read(&bond_block("glue"))
        .expect("parse")
        .lower()
        .expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("glue"), "{msg}");
    assert!(msg.contains("series") && msg.contains("linear"), "{msg}");
}

/// Row 14, both directions: `"mode": "stress"` with no `mechanics` anywhere,
/// and `mechanics` present with the default `"thermal"`.
#[test]
fn mode_and_mechanics_are_refused_in_both_directions() {
    let no_mech = default_slab().replace(
        r#""run": { "steady": true }"#,
        r#""run": { "steady": true, "mode": "stress" }"#,
    );
    let e = read(&no_mech).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("run/mode"), "{msg}");
    assert!(msg.contains("mechanics"), "{msg}");

    let thermal = default_stress().replace(r#""mode": "stress""#, r#""mode": "thermal""#);
    let e = read(&thermal).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("mechanics"), "{msg}");
    assert!(msg.contains("stress"), "{msg}");
}

/// Row 15: `"mode": "stress"` with a fluid region - the fluid side of a
/// thermo-elastic run is docs/10's address 106 (WF-B).
#[test]
fn stress_with_a_fluid_region_is_refused() {
    let text = duct_base()
        .replace(
            r#""material": { "rho": 2000.0, "c": 700.0, "kappa": 100.0 },"#,
            r#""material": { "rho": 2000.0, "c": 700.0, "kappa": 100.0 },
      "mechanics": {
        "material": { "E": 200e9, "nu": 0.3, "alpha": 1.2e-5, "TRef": 300.0 },
        "patches": [
          { "match": "lidWest", "u": { "type": "fixedDisplacement", "value": [0.0, 0.0, 0.0] } },
          { "match": "lidEast", "u": { "type": "free" } },
          { "match": "wallToWater", "u": { "type": "free" } },
          { "match": "heated", "u": { "type": "free" } }
        ]
      },"#,
        )
        .replace(
            r#""run": { "steady": true, "iterations": 400 },"#,
            r#""run": { "steady": true, "iterations": 400, "mode": "stress" },"#,
        );
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("run/mode"), "{msg}");
    assert!(msg.contains("106"), "{msg}");
}

/// Row 16: `"mode": "stress"` on a transient case - the §93 verdict is a
/// steady residual.
#[test]
fn stress_on_a_transient_case_is_refused() {
    let text = default_stress().replace(
        r#""run": { "steady": true, "mode": "stress" }"#,
        r#""run": { "steady": false, "endTime": 1.0, "deltaT": 0.1, "mode": "stress" }"#,
    );
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("steady"), "{msg}");
    assert!(msg.contains("103"), "{msg}");
}

/// Row 18: `output` on a case with a fluid region - the flow path's VTU is a
/// follow-up, not in this unit.
#[test]
fn output_on_a_fluid_case_is_refused() {
    let text = with_output(&duct_base(), r#"{ "exact": { "format": "vtu" } }"#);
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("output"), "{msg}");
    assert!(msg.contains("follow-up"), "{msg}");
}

/// Row 19: a mode that is neither spelling, refused listing both.
#[test]
fn an_unknown_run_mode_is_refused_listing_both() {
    let text = default_stress().replace(r#""mode": "stress""#, r#""mode": "creep""#);
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("creep"), "{msg}");
    assert!(msg.contains("thermal"), "{msg}");
    assert!(msg.contains("stress"), "{msg}");
}

/// Row 17: the `output` block on this format accepts exactly
/// `exact.format: "vtu"`, once - four sub-cases, the fourth in both
/// steady and transient dress.
#[test]
fn the_output_block_on_cht_accepts_exactly_vtu_once() {
    // (1) `output.visualisation` - a multi-region mesh is not one Cartesian
    // lattice, so there is no voxel grid to sample onto.
    let vis = with_output(&default_stress(), r#"{ "visualisation": { "format": "vdb" } }"#);
    let e = read(&vis).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("visualisation"), "{msg}");
    assert!(msg.contains("Cartesian"), "{msg}");

    // (2) `output.restart` - the driver writes no checkpoint of any kind.
    let restart = with_output(&default_stress(), r#"{ "restart": { "keep": 2 } }"#);
    let e = read(&restart).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("ofgpu-cht"), "{msg}");
    assert!(msg.contains("checkpoint"), "{msg}");

    // (3) `exact.format` naming openfoam - one polyMesh per region is
    // docs/10's address 97 layout.
    let foam = with_output(&default_stress(), r#"{ "exact": { "format": "openfoam" } }"#);
    let e = read(&foam).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("polyMesh per region"), "{msg}");

    // (4a) a positive interval on a steady run - §44.4's refusal.
    let interval = with_output(
        &default_stress(),
        r#"{ "exact": { "format": "vtu", "interval": 2.0 } }"#,
    );
    let e = read(&interval).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("interval"), "{msg}");
    assert!(msg.contains("steady"), "{msg}");

    // (4b) ... on a transient THERMAL one (a transient STRESS run is row 16):
    // the driver's run_case returns one state.
    let transient = default_slab().replace(
        r#""run": { "steady": true }"#,
        r#""run": { "steady": false, "endTime": 2.0, "deltaT": 0.5 }"#,
    );
    let text = with_output(&transient, r#"{ "exact": { "format": "vtu", "interval": 2.0 } }"#);
    let e = read(&text).expect("parse").lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("one state"), "{msg}");
}

// ==========================================================================
//  SPEC-LIT §96 - the generated schema, the case_json trio carried over
// ==========================================================================

/// The generated schema is a real artifact: written once by this test, then
/// only ever compared against, never hand-edited.
#[test]
fn cht_schema_writes_to_docs_schema_directory() {
    let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/schema");
    std::fs::create_dir_all(&out_dir).expect("create docs/schema");
    let out_path = out_dir.join("cht-1.json");
    std::fs::write(&out_path, emit_cht_schema()).expect("write docs/schema/cht-1.json");
    assert!(out_path.exists());
}

/// The shipped schema IS the generated one. Line endings are normalised
/// because git may check the file out with CRLF; nothing else is.
#[test]
fn the_shipped_cht_schema_is_the_generated_one() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/schema/cht-1.json");
    let shipped = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let norm = |s: &str| s.replace("\r\n", "\n").trim_end().to_string();
    assert_eq!(
        norm(&shipped),
        norm(&emit_cht_schema()),
        "docs/schema/cht-1.json has drifted from emit_cht_schema(); regenerate it"
    );
}

/// The mechanics block is DOCUMENTED by the schema, word for word: the serde
/// renames are the JSON keys a user types, and `ChtRun::mode`'s doc comment
/// is the only place the word `stress` reaches it.
#[test]
fn cht_schema_documents_the_mechanics_block() {
    let text = emit_cht_schema();
    for word in [
        "mechanics",
        "fixedDisplacement",
        "traction",
        "symmetry",
        "free",
        "stress",
        "TRef",
        "bond",
        "materials",
        "maxOuter",
    ] {
        assert!(text.contains(word), "the schema must document '{word}'");
    }
}

// ==========================================================================
//  SPEC-LIT 96 - the driver unit: the bridge, and Gate 96-A's pairs
// ==========================================================================

use crate::solid::case::{self, RegionStress};
use crate::cht::ChtSolution;

/// The whole `mode: stress` path of SPEC-LIT 96.2 - parse, lower, run the
/// thermal solve, demand its verdict, solve the stress - so a pair test
/// turns one knob of the case and reads the answer.
fn run_stress_doc(gpu: &Gpu, text: &str) -> (LoweredChtCase, ChtSolution, Vec<RegionStress>) {
    let case = read(text).expect("parse");
    let low = case.lower().expect("lower");
    let sol = run_case(gpu, &low).expect("run");
    case::thermal_converged(&low, &sol).expect("thermal converged");
    let stress = case::run_stress(gpu, &low, &sol).expect("stress");
    (low, sol, stress)
}

/// `max_c |u_a - u_b|`, the biggest cell-to-cell move between two runs.
fn du(a: &[Vec3], b: &[Vec3]) -> Scalar {
    a.iter().zip(b).fold(0.0 as Scalar, |m, (x, y)| m.max((*x - *y).mag()))
}

fn umax(a: &[Vec3]) -> Scalar {
    a.iter().map(|x| x.mag()).fold(0.0 as Scalar, Scalar::max)
}

fn vm_max(s: &[RegionStress]) -> Scalar {
    s[0].von_mises.iter().copied().fold(0.0 as Scalar, Scalar::max)
}

/// SPEC-LIT 96.2: `stress` mode refuses to go on unless the thermal solve's
/// own per-region verdict says every region met the criterion - here, a
/// `numerics.tolerance` no f64 solve can reach, so the refusal names the
/// region and the number the case asked for.
#[test]
fn stress_on_an_unconverged_thermal_solve_is_refused_naming_the_region() {
    let Some(gpu) = gpu() else { return };
    let a = default_stress().replace(r#""tolerance": 1e-12"#, r#""tolerance": 1e-30"#);
    assert_ne!(a, default_stress(), "the substitution must change the document");
    let case = read(&a).expect("parse");
    let low = case.lower().expect("lower");
    let sol = run_case(&gpu, &low).expect("run");
    let e = case::thermal_converged(&low, &sol).expect_err("must refuse");
    let m = e.to_string();
    assert!(m.contains("bar"), "{m}");
    assert!(m.contains("1e-30"), "{m}");
}

/// The banner (SPEC-LIT 13.4.2) names every zone with its constants, the
/// predicted contraction, the coupling parameter, and the measured `nu`
/// edge by number; a region without `mechanics` says so in one line.
#[test]
fn the_stress_banner_names_every_zone() {
    let case = read(&bond_block("series")).expect("parse");
    let low = case.lower().expect("lower");
    let banner = case::banner_lines(&low).join("\n");
    for what in ["copper", "steel", "predicted", "delta", "0.45"] {
        assert!(banner.contains(what), "banner is missing '{what}':\n{banner}");
    }
}

// ==========================================================================
//  Gate 96-A - the ten pair tests (SPEC-LIT 13.4.1): two runs differing in
//  exactly one setting of the case file must write DIFFERENT output.
// ==========================================================================

#[test]
fn pair_alpha_changes_the_answer() {
    let Some(gpu) = gpu() else { return };
    let a = default_stress();
    let b = a.replace(
        &steel("1.2e-5", "200e9", "0.3", "300.0"),
        &steel("2.4e-5", "200e9", "0.3", "300.0"),
    );
    assert_ne!(a, b, "the two documents must actually differ");
    let (_, _, sa) = run_stress_doc(&gpu, &a);
    let (_, _, sb) = run_stress_doc(&gpu, &b);
    let (d, u) = (du(&sa[0].u, &sb[0].u), umax(&sa[0].u));
    println!("pair alpha: du = {d:.6e}, umax(a) = {u:.6e}");
    assert!(d > 0.5 * u, "du = {d} against umax(a) = {u}: the case said alpha \
        and the solver ignored it (SPEC-LIT 13.4.1)");
}

/// With a thermal load and displacement/zero-traction BCs only, `u` is
/// independent of `E` - the displacement equation is homogeneous in `E` -
/// so this pair asserts on the STRESS, which scales with `E`.
#[test]
fn pair_e_changes_the_answer() {
    let Some(gpu) = gpu() else { return };
    let a = default_stress();
    let b = a.replace(
        &steel("1.2e-5", "200e9", "0.3", "300.0"),
        &steel("1.2e-5", "100e9", "0.3", "300.0"),
    );
    assert_ne!(a, b, "the two documents must actually differ");
    let (_, _, sa) = run_stress_doc(&gpu, &a);
    let (_, _, sb) = run_stress_doc(&gpu, &b);
    let (va, vb) = (vm_max(&sa), vm_max(&sb));
    println!("pair E: vm_max(a) = {va:.6e}, vm_max(b) = {vb:.6e}");
    assert!((va - vb).abs() > 0.3 * va, "|{va} - {vb}|: the case said E and \
        the solver ignored it (SPEC-LIT 13.4.1)");
}

#[test]
fn pair_nu_changes_the_answer() {
    let Some(gpu) = gpu() else { return };
    let a = default_stress();
    let b = a.replace(
        &steel("1.2e-5", "200e9", "0.3", "300.0"),
        &steel("1.2e-5", "200e9", "0.2", "300.0"),
    );
    assert_ne!(a, b, "the two documents must actually differ");
    let (_, _, sa) = run_stress_doc(&gpu, &a);
    let (_, _, sb) = run_stress_doc(&gpu, &b);
    let (d, u) = (du(&sa[0].u, &sb[0].u), umax(&sa[0].u));
    println!("pair nu: du = {d:.6e}, umax(a) = {u:.6e}");
    assert!(d > 1e-3 * u, "du = {d} against umax(a) = {u}: the case said nu \
        and the solver ignored it (SPEC-LIT 13.4.1)");
}

#[test]
fn pair_tref_changes_the_answer() {
    let Some(gpu) = gpu() else { return };
    let a = default_stress();
    let b = a.replace(
        &steel("1.2e-5", "200e9", "0.3", "300.0"),
        &steel("1.2e-5", "200e9", "0.3", "350.0"),
    );
    assert_ne!(a, b, "the two documents must actually differ");
    let (_, _, sa) = run_stress_doc(&gpu, &a);
    let (_, _, sb) = run_stress_doc(&gpu, &b);
    let (d, u) = (du(&sa[0].u, &sb[0].u), umax(&sa[0].u));
    println!("pair TRef: du = {d:.6e}, umax(a) = {u:.6e}");
    assert!(d > 0.2 * u, "du = {d} against umax(a) = {u}: the case said TRef \
        and the solver ignored it (SPEC-LIT 13.4.1)");
}

#[test]
fn pair_traction_changes_the_answer() {
    let Some(gpu) = gpu() else { return };
    let a = default_stress();
    let b = a.replace(
        r#""type": "traction", "value": [0.0,"#,
        r#""type": "traction", "value": [1e6,"#,
    );
    assert_ne!(a, b, "the two documents must actually differ");
    let (_, _, sa) = run_stress_doc(&gpu, &a);
    let (_, _, sb) = run_stress_doc(&gpu, &b);
    let (d, u) = (du(&sa[0].u, &sb[0].u), umax(&sa[0].u));
    println!("pair traction: du = {d:.6e}, umax(a) = {u:.6e}");
    assert!(d > 1e-4 * u, "du = {d} against umax(a) = {u}: the case said \
        traction and the solver ignored it (SPEC-LIT 13.4.1)");
}

#[test]
fn pair_fixed_displacement_changes_the_answer() {
    let Some(gpu) = gpu() else { return };
    let a = default_stress();
    let b = a.replace(
        r#""type": "fixedDisplacement", "value": [0.0,"#,
        r#""type": "fixedDisplacement", "value": [1e-4,"#,
    );
    assert_ne!(a, b, "the two documents must actually differ");
    let (_, _, sa) = run_stress_doc(&gpu, &a);
    let (_, _, sb) = run_stress_doc(&gpu, &b);
    let (d, u) = (du(&sa[0].u, &sb[0].u), umax(&sa[0].u));
    println!("pair fixedDisplacement: du = {d:.6e}, umax(a) = {u:.6e}");
    assert!(d > 0.1 * u, "du = {d} against umax(a) = {u}: the case said \
        fixedDisplacement and the solver ignored it (SPEC-LIT 13.4.1)");
}

#[test]
fn pair_symmetry_versus_free_changes_the_answer() {
    let Some(gpu) = gpu() else { return };
    let a = default_stress();
    let b = a.replace(FREE, r#"{ "type": "symmetry" }"#);
    assert_ne!(a, b, "the two documents must actually differ");
    let (_, _, sa) = run_stress_doc(&gpu, &a);
    let (_, _, sb) = run_stress_doc(&gpu, &b);
    let (d, u) = (du(&sa[0].u, &sb[0].u), umax(&sa[0].u));
    println!("pair symmetry/free: du = {d:.6e}, umax(a) = {u:.6e}");
    assert!(d > 1e-2 * u, "du = {d} against umax(a) = {u}: the case said \
        symmetry and the solver ignored it (SPEC-LIT 13.4.1)");
}

#[test]
fn pair_tolerance_changes_the_answer() {
    let Some(gpu) = gpu() else { return };
    let a = default_stress();
    let b = a.replace(r#""tolerance": 1e-8"#, r#""tolerance": 1e-2"#);
    assert_ne!(a, b, "the two documents must actually differ");
    let (_, _, sa) = run_stress_doc(&gpu, &a);
    let (_, _, sb) = run_stress_doc(&gpu, &b);
    let (na, nb) = (sa[0].report.iterations, sb[0].report.iterations);
    println!("pair tolerance: iterations(a) = {na}, iterations(b) = {nb}");
    assert!(na != nb, "the case said mechanics/solver/tolerance and the solver \
        ignored it (SPEC-LIT 13.4.1)");
    assert!(sa[0].report.converged && sb[0].report.converged);
}

#[test]
fn pair_max_outer_changes_the_answer() {
    let Some(gpu) = gpu() else { return };
    let a = default_stress();
    let b = a.replace(r#""maxOuter": 500"#, r#""maxOuter": 2"#);
    assert_ne!(a, b, "the two documents must actually differ");
    let (_, _, sa) = run_stress_doc(&gpu, &a);
    assert!(sa[0].report.converged, "a (maxOuter 500) must converge");
    // `b` by hand: `run_stress` is Err naming the region and the knob
    // (SPEC-LIT 96.3 row 21).
    let case = read(&b).expect("parse");
    let low = case.lower().expect("lower");
    let sol = run_case(&gpu, &low).expect("run");
    case::thermal_converged(&low, &sol).expect("thermal converged");
    let e = case::run_stress(&gpu, &low, &sol).expect_err("maxOuter 2 must refuse");
    let m = e.to_string();
    println!("pair maxOuter: refusal = {m}");
    assert!(m.contains("maxOuter"), "{m}");
    assert!(m.contains("bar"), "{m}");
}

#[test]
fn pair_bond_treatment_changes_the_answer() {
    let Some(gpu) = gpu() else { return };
    let a = bond_block("series");
    let b = bond_block("linear");
    assert_ne!(a, b, "the two documents must actually differ");
    let (_, _, sa) = run_stress_doc(&gpu, &a);
    let (_, _, sb) = run_stress_doc(&gpu, &b);
    let (d, u) = (du(&sa[0].u, &sb[0].u), umax(&sa[0].u));
    println!("pair bond: du = {d:.6e}, umax(a) = {u:.6e}");
    assert!(d > 1e-6 * u, "du = {d} against umax(a) = {u}: the case said bond \
        and the solver ignored it (SPEC-LIT 13.4.1)");
}

/// The shipped bimetal strip, end to end: the conduction solve meets its
/// criterion, the outer loop converges, and the tip deflection matches
/// Timoshenko's 1925 closed form (SPEC-LIT (S95.19) carries the same
/// constant) to 10 % with the right sign. `kappa` is rebuilt here from the
/// case's own constants, never transcribed.
#[test]
fn the_bimetal_strip_case_reproduces_timoshenko() {
    let Some(gpu) = gpu() else { return };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../cases/bimetalStrip.cht.jsonc");
    let case = super::read_cht_case(&path).expect("read cases/bimetalStrip.cht.jsonc");
    let low = case.lower().expect("lower");
    let sol = run_case(&gpu, &low).expect("run");
    case::thermal_converged(&low, &sol).expect("thermal converged");
    let stress = case::run_stress(&gpu, &low, &sol).expect("stress");
    assert_eq!(stress.len(), 1, "one mechanical region");
    let s = &stress[0];
    assert!(s.report.converged);

    let m = &low.meshes[0];
    // The cells of the last x column, their mean centroid x, and the mean
    // u_y across them - the tip.
    let last: Vec<usize> =
        (0..m.n_cells).filter(|&c| m.c[c].x > 0.06 - 0.06 / 96.0).collect();
    assert!(!last.is_empty());
    let x_c = last.iter().map(|&c| m.c[c].x).sum::<Scalar>() / last.len() as Scalar;
    let d_meas = last.iter().map(|&c| s.u[c].y).sum::<Scalar>() / last.len() as Scalar;

    // Timoshenko 1925 (SPEC-LIT (S95.19)): m = a1/a2, n = E1/E2.
    let (a1, a2) = (0.005 as Scalar, 0.005 as Scalar);
    let h = a1 + a2;
    let (mm, n) = (a1 / a2, 200.0e9 / 100.0e9);
    let (alpha1, alpha2, d_t) = (1.2e-5 as Scalar, 2.0e-5 as Scalar, 10.0 as Scalar);
    let kappa = 6.0 * (alpha2 - alpha1) * d_t * (1.0 + mm) * (1.0 + mm)
        / (h * (3.0 * (1.0 + mm) * (1.0 + mm) + (1.0 + mm * n) * (mm * mm + 1.0 / (mm * n))));
    let d_exp = -kappa * x_c * x_c / 2.0;
    let err = ((d_meas - d_exp) / d_exp).abs();
    println!(
        "bimetal: kappa = {kappa:.7} /m, x_c = {x_c:.6} m, expected d_y = {d_exp:.4e} m, \
         measured d_y = {d_meas:.4e} m, error {:.3} %, outer iterations {}",
        100.0 * err,
        s.report.iterations
    );
    assert!(err <= 0.10, "tip deflection {d_meas} against closed form {d_exp}: {err} %");
    assert!(d_meas < 0.0, "brass on top bends the tip DOWN (-y), got {d_meas}");
}

/// The written VTU is S2's own point format, read back by S2's own python
/// reader: 1536 cells, `sigma` 9 and symmetric, `u` on points, `T` in both
/// blocks.
#[test]
fn the_written_region_vtu_is_read_back() {
    let Some(gpu) = gpu() else { return };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../cases/bimetalStrip.cht.jsonc");
    let case = super::read_cht_case(&path).expect("read");
    let low = case.lower().expect("lower");
    let sol = run_case(&gpu, &low).expect("run");
    let stress = case::run_stress(&gpu, &low, &sol).expect("stress");
    let dir = std::env::temp_dir().join("ofgpu_s9_vtk");
    let paths = case::write_region_vtu(&dir, &low, &sol, &stress).expect("write");
    assert_eq!(paths.len(), 1, "one region, one VTU");

    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tools/vtu_read.py");
    let out = std::process::Command::new("python")
        .arg(&script)
        .arg(&paths[0])
        .arg("--cells").arg("1536")
        .arg("--cell").arg("T:1")
        .arg("--cell").arg("sigma:9")
        .arg("--cell").arg("vonMises:1")
        .arg("--cell").arg("magU:1")
        .arg("--cell").arg("sigmaPrincipal:3")
        .arg("--cell").arg("u:3")
        .arg("--point").arg("u:3")
        .arg("--point").arg("T:1")
        .arg("--symmetric").arg("sigma")
        .output();
    let out = match out {
        Ok(o) => o,
        Err(e) => {
            eprintln!("python not spawnable ({e}); the python-reader check passes vacuously");
            return;
        }
    };
    assert!(
        out.status.success(),
        "tools/vtu_read.py rejected the region VTU: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The shipped die stack in `mode: stress`: three regions carry
/// `mechanics`, each is solved on its OWN mesh with its own slice of the
/// conjugate field, the grease is skipped, and every run converges
/// (SPEC-LIT 96.2). Prints the summary the driver prints.
#[test]
fn the_shipped_die_stack_case_runs_in_stress_mode() {
    let Some(gpu) = gpu() else { return };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../cases/dieStack.cht.jsonc");
    let case = super::read_cht_case(&path).expect("read cases/dieStack.cht.jsonc");
    let low = case.lower().expect("lower");
    let sol = run_case(&gpu, &low).expect("run");
    case::thermal_converged(&low, &sol).expect("thermal converged");
    let stress = case::run_stress(&gpu, &low, &sol).expect("stress");

    let names: Vec<&str> = stress.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["die", "solder", "spreader"], "one entry per mechanical region, in order");
    for s in &stress {
        assert!(s.report.converged, "region '{}' did not converge", s.name);
        assert!(s.max_u > 0.0, "region '{}' did not move", s.name);
    }
    for line in case::summary_lines(&low, &stress) {
        println!("{line}");
    }
}

// ==========================================================================
//  SPEC-LIT §97 - the imported region
// ==========================================================================

use crate::io::polymesh::{write_poly_mesh_raw, PolyMeshRaw};

/// §97 test rig: lower `text` as a block case, write every region's mesh
/// through `write_poly_mesh_raw` under `<base>/<region>/polyMesh`, and hand
/// back the SAME document with every region's mesh swapped for a `polyMesh`
/// reference - the starting point every refusal below mutates. `case_dir`
/// for the lowering is `base` itself.
fn imported_case(base: &std::path::Path, text: &str) -> ChtCase {
    let case = read(text).expect("parse");
    let low = case.lower().expect("lower as blocks");
    for (i, name) in low.region_names.iter().enumerate() {
        write_poly_mesh_raw(&base.join(name).join("polyMesh"), &low.raw[i])
            .expect("write region polyMesh");
    }
    let mut imported = case.clone();
    for r in &mut imported.regions {
        r.mesh = ChtRegionMesh::PolyMesh(ChtPolyMeshRef {
            poly_mesh: format!("{}/polyMesh", r.name),
        });
    }
    imported
}

/// A private per-test scratch root, cleared before and after.
fn s97_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("ofgpu_s97_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create s97 scratch dir");
    dir
}

/// Requirement 1: a `polyMesh` region lowers from disk, a `bounds`/`cells`
/// region lowers exactly as before, and the two forms mix in one case. The
/// imported region reaches `build_host_mesh` with the SAME raw the block
/// path built, so the meshes agree bitwise.
#[test]
fn a_region_may_come_from_a_poly_mesh_on_disk() {
    let base = s97_dir("mixed");
    let blocks = read(&default_slab()).expect("parse");
    let low_a = blocks.lower().expect("block lower");
    let mut case = imported_case(&base, &default_slab());
    // Leave region 0 a block; import region 1 only.
    case.regions[0].mesh = blocks.regions[0].mesh.clone();
    let low_b = case.lower_in(Some(&base)).expect("lower from disk");

    assert_eq!(low_b.meshes[0].n_cells, 12, "block region unmoved");
    assert_eq!(low_b.meshes[1].n_cells, 9, "imported region same cell count");
    for r in 0..2 {
        assert_eq!(low_b.raw[r].points, low_a.raw[r].points);
        assert_eq!(low_b.raw[r].faces, low_a.raw[r].faces);
        assert_eq!(low_b.raw[r].owner, low_a.raw[r].owner);
        assert_eq!(low_b.raw[r].neighbour, low_a.raw[r].neighbour);
        let names = |m: &crate::mesh::HostMesh| {
            m.patches.iter().map(|p| p.name.clone()).collect::<Vec<_>>()
        };
        assert_eq!(names(&low_b.meshes[r]), names(&low_a.meshes[r]));
    }
    let _ = std::fs::remove_dir_all(&base);
}

/// Requirement 2: a mistyped key is a parse error under EITHER form, and a
/// document carrying both forms matches neither variant of the untagged
/// enum - `polyMesh` and `cells` in one `mesh` is refused, not merged.
#[test]
fn a_mesh_block_with_a_mistyped_key_is_refused_in_both_forms() {
    // Block form, key mistyped.
    let text = default_slab().replace(r#""cells": [12, 1, 1]"#, r#""celss": [12, 1, 1]"#);
    let e = read(&text).expect_err("block typo must refuse");
    let msg = e.to_string();
    assert!(
        msg.contains("celss") || msg.contains("variant") || msg.contains("unknown field"),
        "{msg}"
    );

    // Imported form, key mistyped - the metal region's mesh block rewritten.
    let text = default_slab().replace(
        r#""cells": [9, 1, 1],"#,
        r#""polyMeshh": "metal/polyMesh","#,
    );
    let e = read(&text).expect_err("polyMesh typo must refuse");
    let msg = e.to_string();
    assert!(
        msg.contains("polyMeshh") || msg.contains("variant"),
        "{msg}"
    );

    // Both forms in one `mesh` - neither variant matches, so it is a parse
    // error and not a merge.
    let text = default_slab().replace(
        r#""cells": [9, 1, 1],"#,
        r#""cells": [9, 1, 1], "polyMesh": "metal/polyMesh","#,
    );
    let e = read(&text).expect_err("both forms must refuse");
    assert!(!e.to_string().is_empty());
}

/// Requirement 3: a `patches` rule naming a patch the IMPORTED mesh does not
/// have is refused listing the mesh's OWN patch names - the boundary file's,
/// not a block's six.
#[test]
fn a_rule_naming_a_patch_the_imported_mesh_does_not_have_is_refused() {
    let base = s97_dir("ghost_rule");
    let mut case = imported_case(&base, &default_slab());
    case.regions[0].patches.push(ChtPatchRule {
        match_: "ghost".to_string(),
        kind: "wall".to_string(),
        u: None,
        t: ChtScalarBc::ZeroGradient,
    });
    let e = case.lower_in(Some(&base)).expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("ghost"), "{msg}");
    assert!(msg.contains("toMetal"), "the message must list what IS there: {msg}");
    let _ = std::fs::remove_dir_all(&base);
}

/// Requirement 4: a patch of the imported mesh that no rule and no interface
/// names is refused by `region:patch` - the module doc's rule, now reached
/// from disk.
#[test]
fn an_imported_patch_with_no_condition_is_refused() {
    let base = s97_dir("unnamed");
    let mut case = imported_case(&base, &default_slab());
    // Drop the `hot` rule; nothing else names it.
    case.regions[0].patches.remove(0);
    let e = case.lower_in(Some(&base)).expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("insulation:hot"), "{msg}");
    assert!(msg.contains("no condition"), "{msg}");
    let _ = std::fs::remove_dir_all(&base);
}

/// Requirement 8: `lower()` carries no case directory, and a document with a
/// `polyMesh` region is refused by name rather than resolved against
/// nothing.
#[test]
fn lower_without_a_directory_refuses_a_poly_mesh_region() {
    let base = s97_dir("no_dir");
    let case = imported_case(&base, &default_slab());
    let e = case.lower().expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("lower_in"), "{msg}");
    let _ = std::fs::remove_dir_all(&base);
}

/// Requirement 5, three sub-cases in one test: a `polyMesh` path that is
/// ABSOLUTE, that does not EXIST, or that resolves OUTSIDE the case
/// directory is refused by name, under
/// `regions/<name>/mesh/polyMesh` (SPEC-LIT §97.2).
#[test]
fn a_mesh_path_outside_the_case_directory_is_refused() {
    let base = s97_dir("paths");
    let case_dir = base.join("case");
    std::fs::create_dir_all(&case_dir).expect("create the case dir");
    let blocks = read(&default_slab()).expect("parse");
    let low = blocks.lower().expect("lower as blocks");

    // A real mesh OUTSIDE the case directory, for the third sub-case.
    let outside = base.join("outside");
    write_poly_mesh_raw(&outside.join("die").join("polyMesh"), &low.raw[0])
        .expect("write the outside mesh");

    let attempt = |path: String| {
        let mut case = blocks.clone();
        case.regions[0].mesh =
            ChtRegionMesh::PolyMesh(ChtPolyMeshRef { poly_mesh: path });
        case.lower_in(Some(&case_dir))
    };

    // 1. absolute.
    let abs = outside.join("die").join("polyMesh");
    assert!(abs.is_absolute(), "the fixture itself must be absolute");
    let e = attempt(abs.display().to_string()).expect_err("absolute must refuse");
    let msg = e.to_string();
    assert!(msg.contains("mesh/polyMesh") && msg.contains("absolute"), "{msg}");

    // 2. missing.
    let e = attempt("no_such_mesh/polyMesh".to_string()).expect_err("missing must refuse");
    let msg = e.to_string();
    assert!(msg.contains("does not exist"), "{msg}");
    assert!(
        msg.contains("no_such_mesh"),
        "the joined path is printed: {msg}"
    );

    // 3. outside.
    let e = attempt("../outside/die/polyMesh".to_string()).expect_err("outside must refuse");
    let msg = e.to_string();
    assert!(msg.contains("outside the case"), "{msg}");
    let _ = std::fs::remove_dir_all(&base);
}

/// A two-hex MSH 4.1 text, in ONE volume entity or TWO - the same mesh
/// `io::msh`'s own fixture builds, written out here because that fixture is
/// private to its module. Two volumes, one mesh: the shared x = 1 face is
/// still a single internal face, and nothing in the file but the volume
/// count says the cells belong to two regions.
fn hex_pair_msh(volumes: usize) -> String {
    let (entities, elements) = if volumes == 2 {
        (
            "0 0 0 2\n7 0 0 0 1 1 1 0 0\n8 1 0 0 2 1 1 0 0\n",
            "2 2 1 2\n3 7 5 1\n1 1 2 3 4 5 6 7 8\n3 8 5 1\n2 2 9 10 3 6 11 12 7\n",
        )
    } else {
        (
            "0 0 0 1\n7 0 0 0 2 1 1 0 0\n",
            "1 2 1 2\n3 5 5 2\n1 1 2 3 4 5 6 7 8\n2 2 9 10 3 6 11 12 7\n",
        )
    };
    format!(
        "$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
1
2 1 \"walls\"
$EndPhysicalNames
$Entities
{entities}$EndEntities
$Nodes
1 12 1 12
3 1 0 12
1
2
3
4
5
6
7
8
9
10
11
12
0.0 0.0 0.0
1.0 0.0 0.0
1.0 1.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
1.0 1.0 1.0
0.0 1.0 1.0
2.0 0.0 0.0
2.0 1.0 0.0
2.0 0.0 1.0
2.0 1.0 1.0
$EndNodes
$Elements
{elements}$EndElements
"
    )
}

/// Requirement 6: a `.msh` with SEVERAL volume entities is refused naming
/// `tools/mesh/regions_from_msh.py` and `ofgpu-regions`, while a ONE-volume
/// `.msh` lowers as a region.
#[test]
fn a_msh_with_several_volumes_is_refused_naming_the_layout_route() {
    let base = s97_dir("msh_volumes");
    std::fs::write(base.join("one.msh"), hex_pair_msh(1)).expect("write one.msh");
    std::fs::write(base.join("two.msh"), hex_pair_msh(2)).expect("write two.msh");
    // The two-hex file names no physical SURFACE, so every boundary face
    // lands in `defaultFaces` - the only patch a rule has to name here.
    let doc = |msh: &str| {
        format!(
            r#"{{
  "name": "mshRegion",
  "regions": [ {{ "name": "pair",
      "mesh": {{ "polyMesh": "{msh}" }},
      "material": {{ "rho": 1000.0, "c": 800.0, "kappa": 1.0 }},
      "patches": [ {{ "match": "defaultFaces", "T": {{ "type": "zeroGradient" }} }} ] }} ],
  "initial": {{ "T": 300.0 }},
  "run": {{ "steady": true }}
}}"#
        )
    };

    // One volume: the file lowers, and its two cells arrive as one region.
    let low = read(&doc("one.msh"))
        .expect("parse")
        .lower_in(Some(&base))
        .expect("a one-volume .msh lowers");
    assert_eq!(low.region_names, ["pair"]);
    assert_eq!(low.meshes[0].n_cells, 2);

    // Two volumes: refused, naming the layout route.
    let e = read(&doc("two.msh"))
        .expect("parse")
        .lower_in(Some(&base))
        .expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("2 volume entities"), "{msg}");
    assert!(msg.contains("regions_from_msh.py"), "{msg}");
    assert!(msg.contains("ofgpu-regions"), "{msg}");
    let _ = std::fs::remove_dir_all(&base);
}

/// A unit-cube `PolyMeshRaw` with ONE patch covering all six faces - the
/// smallest mesh an imported-region refusal can be exercised on. The patch
/// type is the caller's, because what is under test is exactly the mesh's
/// own patch TYPE.
fn cube_raw(name: &str, type_name: &str, kind: crate::mesh::PatchKind) -> PolyMeshRaw {
    let v = |x: f64, y: f64, z: f64| crate::Vec3::new(x, y, z);
    PolyMeshRaw {
        points: vec![
            v(0.0, 0.0, 0.0),
            v(1.0, 0.0, 0.0),
            v(1.0, 1.0, 0.0),
            v(0.0, 1.0, 0.0),
            v(0.0, 0.0, 1.0),
            v(1.0, 0.0, 1.0),
            v(1.0, 1.0, 1.0),
            v(0.0, 1.0, 1.0),
        ],
        faces: vec![
            vec![3, 2, 1, 0],
            vec![4, 5, 6, 7],
            vec![0, 1, 5, 4],
            vec![1, 2, 6, 5],
            vec![2, 3, 7, 6],
            vec![3, 0, 4, 7],
        ],
        owner: vec![0; 6],
        neighbour: vec![],
        patches: vec![crate::mesh::PatchInfo {
            name: name.to_string(),
            type_name: type_name.to_string(),
            kind,
            start: 0,
            size: 6,
            nbr_patch: None,
        }],
    }
}

/// Requirement 7, both directions: an `empty` patch of an imported mesh must
/// carry the `empty` rule, and an `empty` rule must land on a patch the mesh
/// itself types `empty` - only the mesh can make one.
#[test]
fn an_imported_empty_patch_must_carry_the_empty_rule_and_vice_versa() {
    let base = s97_dir("empty_agreement");

    // Mesh says `empty`, the case says zeroGradient.
    write_poly_mesh_raw(&base.join("thin").join("polyMesh"), &cube_raw("front", "empty", crate::mesh::PatchKind::Empty))
        .expect("write the empty-patched mesh");
    let text = format!(
        r#"{{
  "name": "importedThin",
  "regions": [ {{ "name": "solid",
      "mesh": {{ "polyMesh": "thin/polyMesh" }},
      "material": {{ "rho": 1000.0, "c": 800.0, "kappa": 1.0 }},
      "patches": [ {{ "match": "front", "T": {{ "type": "zeroGradient" }} }} ] }} ],
  "initial": {{ "T": 300.0 }},
  "run": {{ "steady": true }}
}}"#
    );
    let e = read(&text).expect("parse").lower_in(Some(&base)).expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("front") && msg.contains("empty"), "{msg}");

    // The case says `empty`, the mesh says `patch`.
    write_poly_mesh_raw(&base.join("thick").join("polyMesh"), &cube_raw("shell", "patch", crate::mesh::PatchKind::Generic))
        .expect("write the generic-patched mesh");
    let text = text
        .replace("thin/polyMesh", "thick/polyMesh")
        .replace(
            r#"{ "match": "front", "T": { "type": "zeroGradient" } }"#,
            r#"{ "match": "shell", "T": { "type": "empty" } }"#,
        );
    let e = read(&text).expect("parse").lower_in(Some(&base)).expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("shell") && msg.contains("empty"), "{msg}");
    let _ = std::fs::remove_dir_all(&base);
}

/// Requirement 7: a `cyclic` (or `processor`) patch of an imported region is
/// refused by name - a periodic conducting region is not gated.
#[test]
fn a_cyclic_patch_in_an_imported_region_is_refused() {
    let base = s97_dir("cyclic");
    // The declared couple must satisfy the READER first (a cyclic patch
    // carries its neighbourPatch), so the refusal below is the imported
    // region's own and not the boundary file's.
    let mut raw = cube_raw("periodic", "cyclic", crate::mesh::PatchKind::Cyclic);
    raw.patches[0].nbr_patch = Some(0);
    write_poly_mesh_raw(&base.join("ring").join("polyMesh"), &raw)
        .expect("write the cyclic-patched mesh");
    let text = r#"
{ "name": "importedRing",
  "regions": [ { "name": "solid",
      "mesh": { "polyMesh": "ring/polyMesh" },
      "material": { "rho": 1000.0, "c": 800.0, "kappa": 1.0 },
      "patches": [ { "match": "periodic", "T": { "type": "zeroGradient" } } ] } ],
  "initial": { "T": 300.0 },
  "run": { "steady": true }
}"#;
    let e = read(text).expect("parse").lower_in(Some(&base)).expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("periodic") && msg.contains("cyclic"), "{msg}");
    assert!(msg.contains("31"), "the unexercised section is named: {msg}");
    let _ = std::fs::remove_dir_all(&base);
}

/// Requirement 7: an `inlet`/`outlet` rule on an imported `wall`-typed patch
/// is refused - an opening is `patch`, not `wall` (SPEC-LIT §79.2), the same
/// distinction the block build writes.
#[test]
fn an_opening_on_an_imported_wall_typed_patch_is_refused() {
    let base = s97_dir("wall_opening");
    write_poly_mesh_raw(
        &base.join("water").join("polyMesh"),
        &cube_raw("port", "wall", crate::mesh::PatchKind::Wall),
    )
    .expect("write the wall-patched mesh");
    let text = r#"
{ "name": "importedWater",
  "regions": [ { "name": "water", "kind": "fluid",
      "mesh": { "polyMesh": "water/polyMesh" },
      "fluid": { "rho": 998.0, "cp": 4182.0, "kappa": 0.6, "mu": 1.0e-3 },
      "patches": [ { "match": "port", "kind": "inlet", "U": [0.1, 0.0, 0.0],
                     "T": { "type": "fixedValue", "value": 300.0 } } ] } ],
  "initial": { "T": 300.0 },
  "run": { "steady": true }
}"#;
    let e = read(text).expect("parse").lower_in(Some(&base)).expect_err("must refuse");
    let msg = e.to_string();
    assert!(msg.contains("port"), "{msg}");
    assert!(msg.contains("wall"), "{msg}");
    assert!(msg.contains("79.2"), "{msg}");
    let _ = std::fs::remove_dir_all(&base);
}

/// **Gate 97-A, the lib twin.** The shipped die stack, every region written
/// out through `write_poly_mesh_raw` and read back as a `polyMesh` region,
/// must reproduce the block run BIT FOR BIT: `t`, `bt`, `steps`, `residual`
/// and the pair fluxes, `==` with no tolerance. Both lowerings reach
/// `build_host_mesh` from the same five numbers, so any difference is a mesh
/// path and not a physics.
#[test]
fn the_shipped_case_round_trips_through_poly_mesh_bit_for_bit() {
    let Some(gpu) = gpu() else { return };
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../cases/dieStack.cht.jsonc");
    let case = super::read_cht_case(&path).expect("read cases/dieStack.cht.jsonc");
    let low_a = case.lower().expect("lower the block case");
    let sol_a = run_case(&gpu, &low_a).expect("run the block case");

    let dir = std::env::temp_dir().join(format!("ofgpu_s97_diestack_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    for (i, name) in low_a.region_names.iter().enumerate() {
        write_poly_mesh_raw(&dir.join(name).join("polyMesh"), &low_a.raw[i])
            .expect("write the region polyMesh");
    }
    let mut case_b = case.clone();
    for r in &mut case_b.regions {
        r.mesh = ChtRegionMesh::PolyMesh(ChtPolyMeshRef {
            poly_mesh: format!("{}/polyMesh", r.name),
        });
    }
    let low_b = case_b.lower_in(Some(&dir)).expect("lower the imported case");
    for (i, name) in low_a.region_names.iter().enumerate() {
        assert_eq!(
            low_b.meshes[i].n_cells,
            low_a.meshes[i].n_cells,
            "region '{name}' reads back with a different cell count"
        );
        let names = |m: &crate::mesh::HostMesh| {
            m.patches.iter().map(|p| p.name.clone()).collect::<Vec<_>>()
        };
        assert_eq!(names(&low_b.meshes[i]), names(&low_a.meshes[i]));
    }
    let sol_b = run_case(&gpu, &low_b).expect("run the imported case");

    let same = |what: &str, a: &[Scalar], b: &[Scalar]| {
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x, y, "{what}: first differing value at [{i}]");
        }
        assert_eq!(a.len(), b.len(), "{what}: different lengths");
    };
    same("cell temperature t", &sol_a.t, &sol_b.t);
    same("boundary temperature bt", &sol_a.bt, &sol_b.bt);
    same("pair flux (a side)", &sol_a.pair_flux.0, &sol_b.pair_flux.0);
    same("pair flux (b side)", &sol_a.pair_flux.1, &sol_b.pair_flux.1);
    assert_eq!(sol_a.steps, sol_b.steps, "step count");
    assert_eq!(sol_a.residual, sol_b.residual, "final residual");

    let cells: Vec<usize> = low_a.meshes.iter().map(|m| m.n_cells).collect();
    let (_, t_junction) = sol_a.region_range(0);
    println!("S97 Gate 97-A: regions {:?} cells {:?}", low_a.region_names, cells);
    println!("S97 Gate 97-A: junction temperature {t_junction:.4} K, reproduced identically");
    println!(
        "S97 Gate 97-A: {} cell values, {} boundary values, {} flux pairs, steps {}, \
         residual {:.3e} - all identical",
        sol_a.t.len(),
        sol_a.bt.len(),
        sol_a.pair_flux.0.len(),
        sol_a.steps,
        sol_a.residual
    );
    let _ = std::fs::remove_dir_all(&dir);
}
