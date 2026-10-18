// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate CHR-16's replay: Sandia Flame D, the piloted CH4/air jet flame,
//! EDC (1989) + DRM19 with k-epsilon at C1 = 1.6, REDUCED on the true
//! wedge (SPEC-LIT §134.8; the band §137.4, the answer key §137.5).**
//!
//! `cases/sandiaFlameD.jsonc` runs on `ofgpu-lowmach` - a coarse wedge,
//! EDC with the user-supplied `mech/drm19.dat` + `mech/thermo12.dat`, and
//! the case-stated C1 = 1.6 - and the gate is REPLAYED from the recorded
//! run, the pattern of `flame1d.rs`: a run changes DATA, not code. `RUN`
//! below holds the supervisor's run as recorded (SPEC-LIT §134.8 (g)).
//!
//! The REDUCED rows are held against the plan's REDUCED gate: 300
//! iterations, finite fields, and the inventory element and mass budgets
//! of the last step within 1e-6 ((134.19), (134.21)). The recorded run
//! stopped on the reactor's negativity guard, so the gate MISSES and the
//! miss is registered (§69). The DEFERRED comparison - centreline T_max
//! within 10 % of TNF's 1945 K and its x/d within ±5 of 45, §137.4's
//! DESIGN band - is NAMED and NOT RUN (`DEFERRED_PRICE`): never replaced
//! by a weaker check under the same name (docs/17 §5.3 rule 3).
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §134.8 (this gate as built and the recorded
//!     runs), §137.4 (the DESIGN band) and §137.5 (the TNF answer key)
//!   the TNF documentation, *Sandia/TUD Piloted CH4/Air Flames - Data
//!     Release 2.0*, documentation Release 2.1 (15 June 2007),
//!     https://tnfworkshop.org/wp-content/uploads/2019/02/SandiaPilotDoc21.pdf,
//!     sha256 4e748acd113f3d72fe814be6e5b1e27154577b5040951d59a015e9d287209f72
//!     - the burner geometry, the three streams and the mixture-fraction
//!     constants the inlet test below closes on (pages 2, 3, 4, 5, 7)
//! No GPL-licensed source was consulted.

use ofgpu::{Result, Scalar};

use crate::{Checks, GateReport, How, Uncertainty, Verdict};

/// The gate's scope and registry name.
pub(crate) const GATE: &str =
    "SPEC-LIT 134.8 Gate CHR-16 Sandia Flame D, EDC + DRM19, k-epsilon C1 1.6, REDUCED";

/// The case, relative to the repository root - the tests' input.
#[allow(dead_code)]
pub(crate) const CASE: &str = "cases/sandiaFlameD.jsonc";
/// The steps the REDUCED run was asked for.
pub(crate) const STEPS: usize = 300;
/// The case's time step, s - the recipe's input, not a row's.
#[allow(dead_code)]
pub(crate) const DT: f64 = 1.0e-5;
/// The inventory budgets' bar - the plan's REDUCED gate.
pub(crate) const BUDGET_TOL: f64 = 1.0e-6;
/// TNF's centreline T_max, K - the DEFERRED band's anchor (SPEC-LIT §137.5).
#[allow(dead_code)]
pub(crate) const T_MAX_TNF: f64 = 1945.0;
/// Its x/d - the stations are 5 apart, so §137.4's ±5 is one station.
#[allow(dead_code)]
pub(crate) const XD_T_MAX_TNF: f64 = 45.0;
/// The DESIGN band on T_max, a fraction of `T_MAX_TNF` (SPEC-LIT §137.4).
#[allow(dead_code)]
pub(crate) const BAND_T: f64 = 0.10;
/// The DESIGN band on the position, in x/d (SPEC-LIT §137.4).
#[allow(dead_code)]
pub(crate) const BAND_XD: f64 = 5.0;
/// The driver's normal-end wording (`lowmach.rs`'s `run_end_line`).
const ENDED_OK: &str = "run ended: budget";

/// Why the DEFERRED band row is NOT RUN, verbatim in the row.
pub(crate) const DEFERRED_PRICE: &str = "the REDUCED run stops at step 41 on the reactor's \
     negativity guard, and a statistically steady flame needs about 0.2 s of flow (2e4 steps at \
     dt 1e-5) at the measured 1.5 s per step, about 8 h of card time, over this unit's 3 h \
     (SPEC-LIT 134.8 (g))";

/// What the recorded run left behind - the replay slot's type.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FlameDRun {
    /// The commit and tree the run was made at.
    pub(crate) built_at: &'static str,
    /// The steps the run was asked for.
    pub(crate) steps_asked: usize,
    /// The steps it completed before the refusal.
    pub(crate) steps_reached: usize,
    /// The driver's `run ended:` line, verbatim.
    pub(crate) ended: &'static str,
    /// The steps of the budget run (the same case to a shorter end time).
    pub(crate) budget_steps: usize,
    /// The budget run's `run ended:` line, verbatim.
    pub(crate) budget_ended: &'static str,
    /// The inventory mass budget of its last step, |R|/inflow by (134.19).
    pub(crate) mass_rel: f64,
    /// The inventory element budgets by (134.21), O H C N.
    pub(crate) elements: &'static [(&'static str, f64)],
    /// The same two as the species correction read them.
    pub(crate) as_read_mass_rel: f64,
    pub(crate) as_read_elements: &'static [(&'static str, f64)],
    /// The step's EDC line: split steps, capped and unconverged cell-steps.
    pub(crate) edc: &'static str,
    /// Wall time of the stopped run, s.
    pub(crate) wall_s: f64,
}

// RECORDS-BEGIN (the supervisor's runs, SPEC-LIT 134.8 (g))
pub(crate) static RUN: Option<FlameDRun> = Some(FlameDRun {
    built_at: "1111b2a with CHR-16's working tree",
    steps_asked: 300, steps_reached: 40,
    ended: "run ended: refused | chemistry: EDC refused 1 cell(s); the first is cell 244 with status 8 (BELOW_NEGATIVE_ATOL) (SPEC-LIT §134.8) | exit code 3",
    budget_steps: 40, budget_ended: "run ended: budget | endTime 0.0004 s reached in 40 steps | exit code 0",
    mass_rel: 3.785e-2, elements: &[("O", 1.175e-1), ("H", 2.194e-1), ("C", 2.354e-1), ("N", 1.298e-2)],
    as_read_mass_rel: 1.389e-1, as_read_elements: &[("O", 3.799e-14), ("H", 5.499e-14), ("C", 9.686e-15), ("N", 1.802e-1)],
    edc: "EDC (SPEC-LIT §134.8): 40 split steps, 384000 capped and 737 unconverged cell-steps",
    wall_s: 60.3,
});
// RECORDS-END

/// The replay (SPEC-LIT §134.8): the REDUCED rows replay the recorded run
/// and register the gate's miss; the TNF band row is named and NOT RUN.
pub(crate) fn check_flame_d(c: &mut Checks) -> Result<()> {
    let Some(r) = RUN.as_ref() else {
        c.skip(
            &format!("Gate CHR-16 REDUCED: the run reached {STEPS} steps and ended on its budget"),
            "NOT RECORDED: the recorded run of SPEC-LIT §134.8 has not been made",
        );
        c.skip(
            "Gate CHR-16 REDUCED: the inventory mass and element budgets of the last step \
             ((134.19), (134.21))",
            "NOT RECORDED: the recorded run of SPEC-LIT §134.8 has not been made",
        );
        c.note(
            "the recipe: ofgpu-lowmach <copy>/sandiaFlameD.jsonc -endTime 3e-3 -deltaT 1e-5 \
             -p0 100615.725 -check 10, on a copy of cases/sandiaFlameD.jsonc whose mech/ holds \
             the user-supplied drm19.dat and thermo12.dat (SPEC-LIT §134.8)",
        );
        c.note("the DEFERRED band is named and NOT RUN whatever the REDUCED rows say");
        return Ok(());
    };
    let mut failed = false;
    let normal_end = r.ended.starts_with(ENDED_OK);
    let e = if r.steps_reached == r.steps_asked && normal_end { 0.0 } else { 1.0 };
    let label =
        format!("Gate CHR-16 REDUCED: the run reached {STEPS} steps and ended on its budget");
    c.replaying(|c| c.check_d(&label, e as Scalar, 0.0));
    if e != 0.0 {
        failed = true;
        c.note(&format!("the run stopped at step {}: {}", r.steps_reached, r.ended));
    }
    let e = if r.budget_ended.starts_with(ENDED_OK) { 0.0 } else { 1.0 };
    c.replaying(|c| {
        c.check_d(
            &format!(
                "every field finite at the last good step (the {}-step budget run ended on its \
                 budget)",
                r.budget_steps
            ),
            e,
            0.0,
        )
    });
    failed |= e != 0.0;
    c.replaying(|c| {
        c.check_a(
            "inventory mass budget of the last step, |residual|/inflow (134.19)",
            r.mass_rel as Scalar,
            BUDGET_TOL as Scalar,
        )
    });
    failed |= !(r.mass_rel <= BUDGET_TOL && r.mass_rel.is_finite());
    for (name, v) in r.elements {
        c.replaying(|c| {
            c.check_a(
                &format!("inventory element {name} budget (134.21)"),
                *v as Scalar,
                BUDGET_TOL as Scalar,
            )
        });
        failed |= !(*v <= BUDGET_TOL && v.is_finite());
    }
    c.note(&format!(
        "as read (both density levels the species correction read): mass {:.3e}, {}",
        r.as_read_mass_rel,
        r.as_read_elements
            .iter()
            .map(|(n, v)| format!("{n} {v:.3e}"))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    c.note(&format!("built at {} (the run's commit and tree)", r.built_at));
    c.note(r.edc);
    c.note(&format!("wall time {} s for {} steps", r.wall_s, r.steps_reached));
    if failed {
        let worst = r
            .elements
            .iter()
            .copied()
            .chain(std::iter::once(("mass", r.mass_rel)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).expect("the budgets are finite"))
            .expect("the chain is not empty");
        let refusal = r.ended.split(" | ").nth(1).unwrap_or(r.ended);
        c.report(GateReport {
            verdict: Verdict::Misses,
            how: How::Replayed,
            gate: GATE,
            against: "the plan's REDUCED gate: 300 iterations, finite fields, element and mass \
                      budgets within 1e-6",
            headline: format!(
                "{} of {STEPS} steps: {refusal}; the worst inventory budget is {} at {:.3e} \
                 against 1e-6",
                r.steps_reached, worst.0, worst.1
            ),
            detail: vec!["the causes and the input search are SPEC-LIT 134.8 (g)".to_string()],
            uncertainty: Some(Uncertainty::SingleMesh("one coarse level, REDUCED")),
        });
    }
    c.skip(
        &format!(
            "Gate CHR-16 DEFERRED: centreline T_max within {:.0} % of TNF's {} K and its x/d \
             within +-{:.0} of {} (the proposed DESIGN band, SPEC-LIT 137.4)",
            BAND_T * 100.0,
            T_MAX_TNF,
            BAND_XD,
            XD_T_MAX_TNF
        ),
        &format!("NOT RUN: {DEFERRED_PRICE}"),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ofgpu::io::case_json::{ScalarBc, read_case_jsonc_str};

    const CASE_TEXT: &str = include_str!("../../../../cases/sandiaFlameD.jsonc");
    const KEY: &str = include_str!("../../../../reference/tnf-flameD/centreline_Yfav.csv");

    /// The recorded run as it stands (SPEC-LIT §134.8, recorded
    /// 2026-10-18): the REDUCED run stopped on the reactor's negativity
    /// guard at step 41 of 300, so the gate's rows are MISSED. The test
    /// asserts the record's own shape and prints the miss, so a record
    /// that changes is seen here first.
    #[test]
    fn flame_d_record_is_consistent() {
        let r = RUN.as_ref().expect("the run is recorded");
        println!("  [134.8] MISSED: stopped at step {} of {}: {}", r.steps_reached, r.steps_asked, r.ended);
        assert!(
            r.steps_reached < r.steps_asked,
            "the record reached its {STEPS} steps; update this test"
        );
        assert_eq!(
            r.budget_steps, r.steps_reached,
            "the budget run mirrors the stopped run's steps"
        );
        assert!(r.ended.contains("BELOW_NEGATIVE_ATOL"), "the cause is named: {}", r.ended);
        let names: Vec<&str> = r.elements.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, ["O", "H", "C", "N"], "the four elements");
        assert!(r.mass_rel.is_finite(), "mass_rel");
        assert!(r.as_read_mass_rel.is_finite(), "as_read_mass_rel");
        for (label, list) in
            [("inventory", r.elements), ("as read", r.as_read_elements)]
        {
            for (n, v) in list {
                assert!(v.is_finite(), "{label} {n}");
            }
        }
        assert!(r.wall_s.is_finite(), "wall_s");
    }

    // (137.4)-(137.5)'s constants: the documentation's atomic weights and
    // the two streams' H and C element fractions (SandiaPilotDoc21.pdf
    // p. 7; SPEC-LIT §137.5).
    const W_H: f64 = 1.008;
    const W_C: f64 = 12.011;
    const W_O: f64 = 15.999;
    #[allow(dead_code)]
    const W_N2: f64 = 28.0134;
    const Y1H: f64 = 0.0393;
    const Y1C: f64 = 0.1170;
    const Y2H: f64 = 0.0007;
    const Y2C: f64 = 0.0;

    /// (n_H, n_C, n_O) of one species of the case's inlet map, by its
    /// formula.
    fn atoms(name: &str) -> (f64, f64, f64) {
        match name {
            "H2" => (2.0, 0.0, 0.0),
            "H" => (1.0, 0.0, 0.0),
            "O" => (0.0, 0.0, 1.0),
            "O2" => (0.0, 0.0, 2.0),
            "OH" => (1.0, 0.0, 1.0),
            "H2O" => (2.0, 0.0, 1.0),
            "HO2" => (1.0, 0.0, 2.0),
            "CH2" | "CH2(S)" => (2.0, 1.0, 0.0),
            "CH3" => (3.0, 1.0, 0.0),
            "CH4" => (4.0, 1.0, 0.0),
            "CO" => (0.0, 1.0, 1.0),
            "CO2" => (0.0, 1.0, 2.0),
            "HCO" => (1.0, 1.0, 1.0),
            "CH2O" => (2.0, 1.0, 1.0),
            "CH3O" => (3.0, 1.0, 1.0),
            "C2H4" => (4.0, 2.0, 0.0),
            "C2H5" => (5.0, 2.0, 0.0),
            "C2H6" => (6.0, 2.0, 0.0),
            "AR" => (0.0, 0.0, 0.0),
            other => panic!("an inlet species this test does not know: {other}"),
        }
    }

    /// `W_k` from `W_H`, `W_C` and `W_O` (§137.5's DESIGN rule); argon has
    /// none of the three, so its weight is stated.
    fn molar_mass(name: &str) -> f64 {
        if name == "AR" {
            return 39.948;
        }
        let (h, c, o) = atoms(name);
        h * W_H + c * W_C + o * W_O
    }

    /// Function1 `table`'s linear interpolation at `s` (SPEC-LIT §120.1).
    fn table_at(profile: &[[f64; 2]], s: f64) -> f64 {
        let first = profile.first().expect("a first point");
        let last = profile.last().expect("a last point");
        assert!(s >= first[0] && s <= last[0], "s = {s} outside the table");
        for w in profile.windows(2) {
            if s <= w[1][0] {
                let f = (s - w[0][0]) / (w[1][0] - w[0][0]);
                return w[0][1] + f * (w[1][1] - w[0][1]);
            }
        }
        last[1]
    }

    /// (137.4): the C/H mixture fraction of element fractions `(y_h, y_c)`
    /// against the jet's and the coflow's constants.
    fn mixture_fraction(y_h: f64, y_c: f64) -> f64 {
        let num = 0.5 * (y_h - Y2H) / W_H + 2.0 * (y_c - Y2C) / W_C;
        let den = 0.5 * (Y1H - Y2H) / W_H + 2.0 * (Y1C - Y2C) / W_C;
        num / den
    }

    /// T2: the three streams' inlet compositions close mass and mixture
    /// fraction. Each stream is read off the case's own `fixedProfile`
    /// tables at a radius inside it - 1 mm jet, 6 mm pilot, 20 mm coflow -
    /// the solved sum gives the N2 remainder, and (137.4)-(137.5) rebuild
    /// the documentation's F (class A on the doc's own numbers).
    #[test]
    fn flame_d_inlets_close_mass_and_mixture_fraction() {
        let case = read_case_jsonc_str(CASE_TEXT, "cases/sandiaFlameD.jsonc").expect("parse");
        let species = case.species.as_ref().expect("the species block");
        let y = &species.patches[0].y;
        assert_eq!(species.inert.as_deref(), Some("N2"), "the inert");
        let at = |name: &str, r: f64| -> f64 {
            match y.get(name).unwrap_or_else(|| panic!("{name} has an inlet condition")) {
                ScalarBc::FixedProfile { profile, .. } => table_at(profile, r),
                ScalarBc::FixedValue { value } => *value,
                other => panic!("{name}: {other:?} is not an inlet condition"),
            }
        };
        let names: Vec<String> = y.keys().cloned().collect();
        assert_eq!(names.len(), 20, "the solved species have inlet conditions");
        for (r, want_n2) in [(0.001, 0.6471331552), (0.006, 0.7342292), (0.02, 0.758)] {
            let sum: f64 = names.iter().map(|n| at(n, r)).sum();
            let n2 = 1.0 - sum;
            assert!(
                (n2 - want_n2).abs() <= 1e-9,
                "the N2 remainder at r = {r}: {n2} against {want_n2}"
            );
            let (mut y_h, mut y_c) = (0.0, 0.0);
            for n in &names {
                let (nh, nc, _) = atoms(n);
                let yk = at(n, r);
                let w = molar_mass(n);
                y_h += nh * W_H * yk / w;
                y_c += nc * W_C * yk / w;
            }
            let f = mixture_fraction(y_h, y_c);
            println!("  [134.8] r = {r}: N2 {n2:.10}, Y_H {y_h:.6}, Y_C {y_c:.6}, F {f:.5}");
            if r == 0.001 {
                assert!((f - 1.0).abs() <= 1e-3, "the jet's F is {f}, against 1");
            } else if r == 0.006 {
                assert!((f - 0.27).abs() <= 0.005, "the pilot's F is {f}, against the doc's 0.27");
            } else {
                assert!(f.abs() <= 1e-5, "the coflow's F is {f}, against 0");
            }
        }
    }

    /// T3: the answer key's centreline anchor the band is set against
    /// (SPEC-LIT §137.5): the largest T in `centreline_Yfav.csv` is 1945 K
    /// at x/d = 45. The columns are found by name, as the key's own tests
    /// do.
    #[test]
    fn flame_d_tnf_anchor() {
        let header = KEY.lines().find(|l| l.starts_with("x_d,")).expect("the header line");
        let cols: Vec<&str> = header.split(',').collect();
        let ix = cols.iter().position(|c| *c == "x_d").expect("the x_d column");
        let it = cols.iter().position(|c| *c == "t").expect("the t column");
        let rows = KEY.split_once("x_d,").expect("the header").1.lines().skip(1);
        let mut best: Option<(f64, f64)> = None;
        for line in rows {
            if line.trim().is_empty() {
                continue;
            }
            let v: Vec<&str> = line.split(',').collect();
            let x: f64 = v[ix].parse().expect("an x_d");
            let t: f64 = v[it].parse().expect("a t");
            if best.is_none() || t > best.expect("a best").0 {
                best = Some((t, x));
            }
        }
        let (t, x) = best.expect("rows");
        println!("  [137.5] the centreline's largest T is {t} K at x/d = {x}");
        assert_eq!(t, T_MAX_TNF, "the centreline T_max");
        assert_eq!(x, XD_T_MAX_TNF, "its x/d");
    }
}
