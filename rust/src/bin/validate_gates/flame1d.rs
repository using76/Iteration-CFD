// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate CHR-13's replay: the 1-D freely propagating CH4-air flame, BFER at
//! phi = 1.0, S_L by the 1500 K isotherm on three meshes (SPEC-LIT
//! §133.12; docs/17 §12 row CHR-13).**
//!
//! `cases/flame1d_ch4_bfer.jsonc` runs on `ofgpu-lowmach` at dx = 100, 50
//! and 25 µm (150, 300 and 600 cells over 15 mm) - §94.1 refuses a
//! two-level study, so the plan's two meshes are the finest two of three
//! levels - and the gate is REPLAYED from those recorded runs, the pattern
//! of `plume.rs`: a run changes DATA, not code. `RUNS` below holds the
//! supervisor's runs as `record_flame1d` printed them (SPEC-LIT 133.12 (j)).
//!
//! What each record carries: `reached`, the last written time; `ended`, the
//! driver's `run ended:` line verbatim; `fronts`, the `(t, x_f)` of every
//! written time whose `T` profile crosses the 1500 K isotherm, `x_f` the
//! first crossing from the inlet, linear between cell centres. The row's
//! formula is `S_L = U_in - dx_f/dt`, the slope by least squares over the
//! fronts from 3 ms (label (133.25)).
//!
//! No comparison against a published value is made here (a level that
//! stopped before its end time registers the gate's miss through §69's
//! registry, nothing else does): the ±10 % row is against Konnov et al.
//! 2018's recommended
//! phi = 1.0 flame speed, that paper is CLOSED and no read copy exists
//! (§133.6), so the row is NAMED and NOT RUN - never replaced by a weaker
//! check under the same name (docs/17 §5.3 rule 3).
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §133.12 (this gate as built), §133.6 (Konnov
//!     closed), §94 (the mesh study) and §133.11 (the driver)
//!   B. Franzelli, E. Riber, L. Y. M. Gicquel and T. Poinsot, *Combust.
//!     Flame* 159 (2012) 621-637, DOI 10.1016/j.combustflame.2011.08.004 -
//!     read from the authors' manuscript on CERFACS's site (URL and sha256
//!     in §133.12); only eqs. (6)-(7) and Tables 3-4 are used, by the
//!     mechanism file and the `bfer_f1`/`bfer_f2` tests below
//!   B. J. McBride, S. Gordon and M. A. Reno, NASA TM-4513 (1993), Table II
//!     - a US-Government work, public domain, tracked in
//!     `reference/mcbride-gordon-reno1993/table_II.csv`
//! No GPL-licensed source was consulted.

use ofgpu::vv::{self, Level};
use ofgpu::{Result, Scalar};

use crate::{Checks, GateReport, How, Uncertainty, Verdict};

/// The gate's scope and registry name.
pub(crate) const GATE: &str =
    "SPEC-LIT 133.12 Gate CHR-13 1-D freely propagating CH4-air flame, BFER phi 1.0";

/// The gate's isotherm, K - the tests' locator input; the replayed rows
/// read fronts already measured against it.
#[allow(dead_code)]
pub(crate) const T_ISO: f64 = 1500.0;
/// The case's inlet U, m/s.
pub(crate) const U_IN: f64 = 1.0;
/// The domain length, m - the tests' mesh input, as the case's.
#[allow(dead_code)]
pub(crate) const LENGTH: f64 = 0.015;
/// The three levels, finest last in the case's order.
pub(crate) const DX_UM: [f64; 3] = [100.0, 50.0, 25.0];
/// The cells of `DX_UM` over `LENGTH` - the three case copies' inputs.
#[allow(dead_code)]
pub(crate) const CELLS: [usize; 3] = [150, 300, 600];
/// The case's time step, s - a record's `dt` comes from here via the
/// recorder, so the replayed rows never read this constant.
#[allow(dead_code)]
pub(crate) const DT: f64 = 1.0e-6;
/// The run's end time, s.
pub(crate) const END_TIME: f64 = 6.0e-3;
/// The write interval of the recipe, s - the recipe's input, not a row's.
#[allow(dead_code)]
pub(crate) const WRITE_INTERVAL: f64 = 2.0e-4;
/// The slope's fit starts here, s (a level's first fronts are start-up).
pub(crate) const FIT_FROM: f64 = 3.0e-3;
/// The case, relative to the repository root - the tests' input.
#[allow(dead_code)]
pub(crate) const CASE: &str = "cases/flame1d_ch4_bfer.jsonc";
/// The commit the recorded runs were made at; "" until recorded - the
/// recorder prints the literal, the supervisor pastes it.
#[allow(dead_code)]
pub(crate) const BUILT_AT: &str = "3a53be6 with CHR-13's working tree";
/// The driver's normal-end wording (`lowmach.rs`'s `run_end_line`).
const ENDED_OK: &str = "run ended: budget";
/// What the registered verdict is held against (§69).
const AGAINST: &str = "the plan's S_L = U_in - dx_f/dt within 10 % of Konnov et al. 2018 (closed, so that \
     comparison is NOT RUN); a level that stops before its end time gives no S_L at all";

/// Franzelli et al. (2012) eq. (6) with Table 4's j = 1 constants: the CH4
/// reaction's phi correction, `f1(phi)`, evaluated in f64 - the
/// mechanism's A1' folded it in once, so the replayed rows never call this.
#[allow(dead_code)]
pub(crate) fn bfer_f1(phi: f64) -> f64 {
    const PHI0: f64 = 1.1;
    const SIGMA0: f64 = 0.09;
    const B1: f64 = 0.37;
    const PHI1: f64 = 1.13;
    const SIGMA1: f64 = 0.03;
    const C1: f64 = 6.7;
    const PHI2: f64 = 1.6;
    const SIGMA2: f64 = 0.22;
    2.0 / ((1.0 + ((PHI0 - phi) / SIGMA0).tanh())
        + B1 * (1.0 + ((phi - PHI1) / SIGMA1).tanh())
        + C1 * (1.0 + ((phi - PHI2) / SIGMA2).tanh()))
}

/// Franzelli et al. (2012) eq. (7) with Table 4's j = 2 constants: the CO
/// reaction's phi correction, `f2(phi)`, evaluated in f64 - the
/// mechanism's A2' folded it in once, so the replayed rows never call this.
#[allow(dead_code)]
pub(crate) fn bfer_f2(phi: f64) -> f64 {
    const PHI0: f64 = 0.95;
    const SIGMA0: f64 = 0.08;
    const B2: f64 = 2.5e-5;
    const PHI1: f64 = 1.3;
    const SIGMA1: f64 = 0.04;
    const C2: f64 = 0.0087;
    const PHI2: f64 = 1.2;
    const SIGMA2: f64 = 0.04;
    const PHI3: f64 = 1.2;
    const SIGMA3: f64 = 0.05;
    0.5 * (1.0 + ((PHI0 - phi) / SIGMA0).tanh())
        + (B2 / 2.0) * (1.0 + ((phi - PHI1) / SIGMA1).tanh())
        + (C2 / 2.0)
            * (1.0 + ((phi - PHI2) / SIGMA2).tanh())
            * (1.0 + ((PHI3 - phi) / SIGMA3).tanh())
}

/// The flame front's position at one written time: the FIRST crossing of
/// `t_iso` from the inlet (index 0 upward), `T_i < t_iso <= T_(i+1)`, linear
/// between the cell centres `x_i = (i + 0.5) dx`; `None` when no cell pair
/// crosses.
#[allow(dead_code)] // the recorder runs it over the recorded T profiles; the replayed rows read its fronts
pub(crate) fn front_position(t: &[f64], dx: f64, t_iso: f64) -> Option<f64> {
    for i in 0..t.len().saturating_sub(1) {
        if t[i] < t_iso && t_iso <= t[i + 1] {
            let frac = (t_iso - t[i]) / (t[i + 1] - t[i]);
            return Some((i as f64 + 0.5 + frac) * dx);
        }
    }
    None
}

/// The ordinary least squares slope of `x_f` against `t` over the pairs with
/// `t >= from`; `None` with fewer than 4 pairs.
pub(crate) fn fit_slope(fronts: &[(f64, f64)], from: f64) -> Option<f64> {
    let pts: Vec<(f64, f64)> = fronts.iter().copied().filter(|(t, _)| *t >= from).collect();
    if pts.len() < 4 {
        return None;
    }
    let n = pts.len() as f64;
    let (st, sx) = pts.iter().fold((0.0, 0.0), |(a, b), (t, x)| (a + t, b + x));
    let (mt, mx) = (st / n, sx / n);
    let num = pts.iter().fold(0.0, |a, (t, x)| a + (t - mt) * (x - mx));
    let den = pts.iter().fold(0.0, |a, (t, _)| a + (t - mt) * (t - mt));
    Some(num / den)
}

/// The row's formula: `S_L = U_in - dx_f/dt`.
pub(crate) fn flame_speed(u_in: f64, slope: f64) -> f64 {
    u_in - slope
}

// ---------------------------------------------------------------------------
//  The records (SPEC-LIT 133.12's runs)
// ---------------------------------------------------------------------------

/// What one recorded run leaves behind - the replay slot's type.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FlameRun {
    /// The level's dx, µm.
    pub(crate) dx_um: f64,
    /// The level's cells along the tube (`n_cells = CELLS of the copy`).
    pub(crate) n_cells: usize,
    /// The time step, s.
    pub(crate) dt: f64,
    /// The end time the run was given, s - the recorder prints it so the
    /// record says what the run was given; the row reads `reached`.
    #[allow(dead_code)]
    pub(crate) end_time: f64,
    /// The last written time, s.
    pub(crate) reached: f64,
    /// The driver's `run ended:` line, verbatim.
    pub(crate) ended: &'static str,
    /// The `(t, x_f)` of every written time that has a crossing.
    pub(crate) fronts: &'static [(f64, f64)],
}

// RECORDS-BEGIN (printed by `record_flame1d`)
pub(crate) static RUNS: Option<[FlameRun; 3]> = Some([
    FlameRun {
        dx_um: 100.0, n_cells: 150, dt: 1e-6, end_time: 0.006,
        reached: 0.0006,
        ended: "run ended: refused | chemistry: the reactor refused 1 cell; the first is cell 13 with status 8 (BELOW_NEGATIVE_ATOL) (SPEC-LIT §133.11) | exit code 3",
        fronts: &[(0.0002, 0.0005220665606264528), (0.0004, 0.0007213247508305648), (0.0006, 0.0008163299217936211)],
    },
    FlameRun {
        dx_um: 50.0, n_cells: 300, dt: 1e-6, end_time: 0.006,
        reached: 0.0006,
        ended: "run ended: refused | chemistry: the reactor refused 1 cell; the first is cell 25 with status 8 (BELOW_NEGATIVE_ATOL) (SPEC-LIT §133.11) | exit code 3",
        fronts: &[(0.0002, 0.0004829209392183088), (0.0004, 0.0006999983001291901), (0.0006, 0.0008291755757834931)],
    },
    FlameRun {
        dx_um: 25.0, n_cells: 600, dt: 1e-6, end_time: 0.006,
        reached: 0.0006,
        ended: "run ended: refused | chemistry: the reactor refused 1 cell; the first is cell 48 with status 8 (BELOW_NEGATIVE_ATOL) (SPEC-LIT §133.11) | exit code 3",
        fronts: &[(0.0002, 0.00046800107629753644), (0.0004, 0.0006861913565981048), (0.0006, 0.0008221112987578992)],
    },
]);
// RECORDS-END

/// The replay (SPEC-LIT 133.12): R1's Konnov row is named and NOT RUN, the
/// per-level rows replay the recorded runs, and the §94 study runs on the
/// three S_L when all three are measurable. No comparison against a
/// published value is made; a level that stopped registers the gate's miss.
pub(crate) fn check_flame1d(c: &mut Checks) -> Result<()> {
    c.skip(
        "Gate CHR-13: S_L within ±10 % of Konnov et al. 2018's recommended phi = 1.0 value \
         (SPEC-LIT 133.12)",
        "NOT RUN: Konnov et al. 2018 (DOI 10.1016/j.pecs.2018.05.003) is closed and no read copy \
         exists (SPEC-LIT 133.6)",
    );
    let Some(runs) = RUNS.as_ref() else {
        for &dx in DX_UM.iter() {
            c.skip(
                &format!("Gate CHR-13: the {dx} µm flame run recorded"),
                "NOT RECORDED: the recorded runs of SPEC-LIT 133.12 have not been made",
            );
        }
        c.note(
            "the recipe: ofgpu-lowmach <copy>/flame1d_ch4_bfer.jsonc -endTime 6e-3 -deltaT 1e-6 \
             -check 500 -writeInterval 2e-4, on three copies of cases/flame1d_ch4_bfer.jsonc \
             with ONLY cells edited to [150,1,1], [300,1,1], [600,1,1] (SPEC-LIT 133.12)",
        );
        c.note("no comparison against a published value is made, so this gate registers no verdict");
        return Ok(());
    };
    let mut speeds: Vec<(f64, f64)> = Vec::new();
    let mut stopped: Vec<String> = Vec::new();
    for r in runs.iter() {
        let normal_end = r.ended.starts_with(ENDED_OK);
        let e = if r.reached >= END_TIME - r.dt / 2.0 && normal_end { 0.0 } else { 1.0 };
        let label = format!(
            "Gate CHR-13: the {} µm flame run reached its end time 6 ms ({} cells, dt 1e-6 s, \
             replayed from the recorded run)",
            r.dx_um, r.n_cells
        );
        c.replaying(|c| c.check_d(&label, e as Scalar, 0.0));
        if e != 0.0 {
            c.note(&format!("{} µm stopped at t = {} s: {}", r.dx_um, r.reached, r.ended));
            stopped.push(format!("{} µm at {} s", r.dx_um, r.reached));
            continue;
        }
        match fit_slope(r.fronts, FIT_FROM) {
            Some(slope) => {
                let pairs = r.fronts.iter().filter(|(t, _)| *t >= FIT_FROM).count();
                let sl = flame_speed(U_IN, slope);
                c.note(&format!(
                    "{} µm: S_L = {sl:.6} m/s (slope {slope:.6} m/s over {pairs} fronts from 3 ms)",
                    r.dx_um
                ));
                speeds.push((r.dx_um, sl));
            }
            None => {
                let label = format!(
                    "Gate CHR-13: the {} µm flame's S_L measurable (>= 4 fronts after 3 ms)",
                    r.dx_um
                );
                c.replaying(|c| c.check_d(&label, 1.0, 0.0));
            }
        }
    }
    if speeds.len() == 3 {
        speeds.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("dx_um is finite"));
        let levels: Vec<Level> = speeds
            .iter()
            .map(|(dx, sl)| Level { h: (dx * 1.0e-6) as Scalar, value: *sl as Scalar })
            .collect();
        match vv::grid_study(&levels) {
            Ok(study) => c.note(&format!("the §94 study of the three S_L: {}", study.one_line())),
            Err(e) => c.note(&format!("the §94 study refused the three S_L: {e}")),
        }
    } else {
        c.note(&format!("§94 study NOT RUN: {} of 3 levels", speeds.len()));
    }
    if stopped.is_empty() {
        c.note("no comparison against a published value is made, so this gate registers no verdict");
        return Ok(());
    }
    // SPEC-LIT 133.12 (j): a level that stopped is the gate's miss (the
    // user's 2026-10-14 rule), registered so the verdict word is the
    // registry's (§69.2).
    c.report(GateReport {
        verdict: Verdict::Misses,
        how: How::Replayed,
        gate: GATE,
        against: AGAINST,
        headline: format!(
            "{} of 3 levels stopped before 6 ms on the reactor's refusal ({}), so no S_L was measured",
            stopped.len(),
            stopped.join(", ")
        ),
        detail: vec!["the cause and the input probes are SPEC-LIT 133.12 (j)".to_string()],
        uncertainty: Some(Uncertainty::SingleMesh(
            "three levels were recorded and none reached its end time, so there is no 94 study",
        )),
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ofgpu::chem::mechanism::{CAL_J, Mechanism};
    use ofgpu::chem::thermo::{ElementTable, P_ATM, R_GAS, ThermoSet};
    use std::path::{Path, PathBuf};

    const MECH_TEXT: &str = include_str!("../../../../cases/mech/bfer2_phi1.ck");
    const THERMO_TEXT: &str = include_str!("../../../../cases/mech/tm4513_thermo.dat");
    const TABLE_II: &str = include_str!("../../../../reference/mcbride-gordon-reno1993/table_II.csv");

    /// The mechanism file's species, in its own order.
    const MECH_SPECIES: [&str; 6] = ["CH4", "O2", "CO", "H2O", "CO2", "N2"];

    /// The ten TM-4513 species: the table's code, the name.
    const CODES: [(&str, &str); 10] = [
        ("1", "N2"),
        ("2", "O2"),
        ("5", "CH4"),
        ("6", "CO2"),
        ("7", "H2O"),
        ("8", "CO"),
        ("9", "H2"),
        ("10", "OH"),
        ("11", "O"),
        ("12", "H"),
    ];

    /// One CSV row's seven coefficients, keyed by (code, t_lo).
    fn csv_coeffs() -> Vec<(String, f64, [f64; 7])> {
        let mut rows = Vec::new();
        for line in TABLE_II.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("species") {
                continue;
            }
            let c: Vec<&str> = line.split(',').collect();
            let mut a = [0.0f64; 7];
            for (j, v) in a.iter_mut().enumerate() {
                *v = c[4 + j].parse().expect("a coefficient");
            }
            rows.push((c[0].to_string(), c[1].parse().expect("t_lo"), a));
        }
        rows
    }

    /// T1: the phi = 1 corrections, to 1e-14 relative.
    #[test]
    fn bfer_corrections_at_phi_one() {
        let f1 = bfer_f1(1.0);
        let f2 = bfer_f2(1.0);
        assert!((f1 / 1.0743172004768002 - 1.0).abs() < 1e-14, "f1(1) = {f1}");
        assert!((f2 / 0.22270092849097087 - 1.0).abs() < 1e-14, "f2(1) = {f2}");
    }

    /// T2: the mechanism file through the crate's own readers, against
    /// Franzelli's Tables 3-4 and the TM-4513 Table II records (class D).
    #[test]
    fn bfer_mechanism_matches_franzelli_tables() {
        let set = ThermoSet::read_chemkin(THERMO_TEXT, "tm4513_thermo.dat", P_ATM, &ElementTable::standard())
            .expect("the thermo file reads");
        let mech = Mechanism::read_chemkin(MECH_TEXT, "bfer2_phi1.ck", Some(&set)).expect("the mechanism reads");
        assert_eq!(
            mech.species().iter().map(String::as_str).collect::<Vec<_>>(),
            MECH_SPECIES,
            "the mechanism's species"
        );
        assert_eq!(mech.reactions().len(), 2, "two reactions");
        let rx1 = &mech.reactions()[0];
        let rx2 = &mech.reactions()[1];
        assert!(!rx1.reversible, "reaction 1 is irreversible");
        assert!(rx2.reversible, "reaction 2 is the CO-CO2 equilibrium");
        // The reader stores A in SI, converting the file's cm3-mol-s number
        // with the FORD-corrected molecularity (130.2); theta = E/R_c.
        let m1 = rx1.molecularity();
        let a1 = 4.9e9 * bfer_f1(1.0) * 1.0e-6_f64.powf(m1 - 1.0);
        assert!((rx1.rate.a / a1 - 1.0).abs() < 1e-9, "A1: {} against {a1}", rx1.rate.a);
        assert_eq!(rx1.rate.beta, 0.0, "beta1");
        assert_eq!(rx1.rate.theta, CAL_J * 35500.0 / R_GAS, "Ea1 = 35500 cal/mol");
        let orders = rx1.forward_orders();
        let (i_ch4, i_o2) = (mech.species_index("CH4").expect("CH4"), mech.species_index("O2").expect("O2"));
        assert_eq!(orders, vec![(i_ch4, 0.5), (i_o2, 0.65)], "reaction 1's forward orders");
        let m2 = rx2.molecularity();
        let a2 = 2.0e8 * bfer_f2(1.0) * 1.0e-6_f64.powf(m2 - 1.0);
        assert!((rx2.rate.a / a2 - 1.0).abs() < 1e-9, "A2: {} against {a2}", rx2.rate.a);
        assert_eq!(rx2.rate.beta, 0.8, "beta2 (the paper's Table 3, not the cti's 0.7)");
        assert_eq!(rx2.rate.theta, CAL_J * 12000.0 / R_GAS, "Ea2 = 12000 cal/mol");
        // The mechanism's thermo carries exactly the six species' TM-4513
        // records, every coefficient equal to the CSV's number; the file's
        // own set holds all ten.
        let rows = csv_coeffs();
        let check = |name: &str, upper: [f64; 7], lower: [f64; 7]| {
            let code = CODES.iter().find(|(_, n)| *n == name).expect("a fixture species").0;
            let pick = |t_lo: f64| -> [f64; 7] {
                rows.iter().find(|(c, t, _)| c == code && *t == t_lo).expect("row").2
            };
            assert_eq!(upper, pick(1000.0), "{name} upper set");
            assert_eq!(lower, pick(200.0), "{name} lower set");
        };
        for sp in set.species() {
            check(&sp.name, sp.upper, sp.lower);
        }
        let mth = mech.thermo();
        assert_eq!(mth.species().len(), 6, "the mechanism's thermo: one record per species");
        for sp in mth.species() {
            check(&sp.name, sp.upper, sp.lower);
        }
    }

    /// The solved members' fresh and burnt mass fractions, computed from the
    /// solver's own molar masses (SPEC-LIT 128.2) as the case's values were.
    /// Fresh: CH4 + 2 (O2 + 3.76 N2) by moles. Burnt: its complete-combustion
    /// products, CO2 + 2 H2O + 7.52 N2.
    fn compositions(
        set: &ThermoSet,
    ) -> Vec<(&'static str, f64, f64)> {
        let w = |name: &str| set.species()[set.index_of(name).expect("a fixture species")].molar_mass;
        let (w_ch4, w_o2, w_n2, w_co2, w_h2o) = (w("CH4"), w("O2"), w("N2"), w("CO2"), w("H2O"));
        let m_fresh = w_ch4 + 2.0 * w_o2 + 7.52 * w_n2;
        let m_burnt = w_co2 + 2.0 * w_h2o + 7.52 * w_n2;
        let fresh = |n: f64, wn: f64| n * wn / m_fresh;
        let burnt = |n: f64, wn: f64| n * wn / m_burnt;
        vec![
            ("CH4", fresh(1.0, w_ch4), burnt(0.0, w_ch4)),
            ("O2", fresh(2.0, w_o2), burnt(0.0, w_o2)),
            ("CO", 0.0, 0.0),
            ("H2O", 0.0, burnt(2.0, w_h2o)),
            ("CO2", 0.0, burnt(1.0, w_co2)),
        ]
    }

    /// A written value against the composition, relative 1e-9 (exact zero
    /// against exact zero). Class R: under `single` the lowered fraction is
    /// the case's 10 digits rounded to f32, so the bar is D-F32-1's
    /// `tol32 = max(tol64, min(tol64 2^29, 1e-5))` of `tol64` = 1e-9.
    fn assert_fraction(what: &str, written: f64, expected: f64) {
        let tol64: f64 = 1e-9;
        let tol: f64 = if cfg!(feature = "single") { tol64.max((tol64 * 2f64.powi(29)).min(1e-5)) } else { tol64 };
        if expected == 0.0 {
            assert_eq!(written, 0.0, "{what}: {written} against 0");
        } else {
            assert!(
                (written / expected - 1.0).abs() < tol,
                "{what}: {written} against {expected}"
            );
        }
    }

    /// T3: the case file lowers to the record inputs of SPEC-LIT 133.12 -
    /// the mesh, the inlet U, the run, the two compositions, phi = 1 and the
    /// transport rule's nu.
    #[test]
    fn flame1d_case_carries_the_record_inputs() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(CASE);
        let l = ofgpu::io::case_json::read_case_jsonc(&path).expect("parse").lower().expect("lower");
        // Exact (class D) in both builds: the lowered scalar is the case's
        // decimal text rounded once to `Scalar`, so it equals `want as Scalar`.
        let eq = |what: &str, written: f64, want: f64| {
            assert_eq!(written, f64::from(want as Scalar), "{what}");
        };
        assert_eq!(l.block.x.n, CELLS[1], "the 50 µm level's cells along the tube");
        assert_eq!((l.block.y.n, l.block.z.n), (1, 1), "one cell across");
        eq("the domain length", f64::from(l.block.x.hi - l.block.x.lo), LENGTH);
        let inlet_u = &l.u_field.boundary["inlet"];
        assert_eq!(inlet_u.type_name, "fixedValue", "the inlet U's condition");
        eq("the inlet U", f64::from(inlet_u.value_v[0].x), U_IN);
        eq("deltaT", f64::from(l.run.delta_t), DT);
        eq("endTime", f64::from(l.run.end_time), END_TIME);
        eq("the case's nu, verbatim", f64::from(l.nu), 5.688e-5);

        let set = ThermoSet::read_chemkin(THERMO_TEXT, "tm4513_thermo.dat", P_ATM, &ElementTable::standard())
            .expect("the thermo file reads");
        let comp = compositions(&set);
        let ch = l.chem.as_ref().expect("the case states chemistry");
        for (name, fresh, burnt) in &comp {
            let f = ch
                .species
                .fields
                .iter()
                .find(|f| f.name == *name)
                .unwrap_or_else(|| panic!("{name} is a solved field"));
            assert_fraction(&format!("species.initial.{name}"), f64::from(f.internal_uniform), *burnt);
            let bc = f.boundary.get("inlet").expect("the inlet patch rule");
            assert_eq!(bc.type_name, "fixedValue", "{name}'s inlet condition");
            assert_fraction(&format!("inlet Y {name}"), f64::from(bc.value[0]), *fresh);
        }
        // The fresh mixture's C:O atom ratio is phi = 1: n_C = Y_CH4/W_CH4
        // (one C atom per CH4), n_O = 2 Y_O2/W_O2, stoichiometric C/O = 1/4.
        let (_, y_ch4, _) = comp[0];
        let (_, y_o2, _) = comp[1];
        let w_ch4 = set.species()[set.index_of("CH4").expect("CH4")].molar_mass;
        let w_o2 = set.species()[set.index_of("O2").expect("O2")].molar_mass;
        let phi = 4.0 * (y_ch4 / w_ch4) / (2.0 * y_o2 / w_o2);
        assert!((phi - 1.0).abs() <= 1e-12, "the fresh mixture's phi is {phi}");

        // The transport rule (133.12): nu = Pr k / (rho cp) of the burnt
        // products at 1500 K and 1 atm, with the crate's NASA-7 cp.
        let w = |name: &str| set.species()[set.index_of(name).expect("a fixture species")].molar_mass;
        let (w_co2, w_h2o, w_n2) = (w("CO2"), w("H2O"), w("N2"));
        let m_b = w_co2 + 2.0 * w_h2o + 7.52 * w_n2;
        let idx = |name: &str| set.index_of(name).expect("a fixture species");
        let mut yb = vec![0.0f64; set.species().len()];
        yb[idx("CO2")] = w_co2 / m_b;
        yb[idx("H2O")] = 2.0 * w_h2o / m_b;
        yb[idx("N2")] = 7.52 * w_n2 / m_b;
        let w_b = set.molar_mass_mix(&yb).expect("W_b");
        let cp_b = set.cp_mass(1500.0, &yb).expect("cp_b");
        let rho = P_ATM * w_b / (R_GAS * 1500.0);
        let nu = 0.7 * 0.026 / (rho * cp_b);
        assert!(
            (f64::from(l.nu) / nu - 1.0).abs() < 2e-4,
            "nu: the case's {} against the rule's {nu}",
            f64::from(l.nu)
        );
    }

    /// T4: the front locator takes the FIRST crossing from the inlet, linear
    /// between cell centres.
    #[test]
    fn front_position_interpolates_linearly() {
        let dx = 1.0e-4;
        let t: Vec<f64> = (0..20)
            .map(|i| (2200.0f64).min(300.0 + 190.0 * (i as f64 - 5.0).max(0.0)))
            .collect();
        let x = front_position(&t, dx, 1500.0).expect("a crossing");
        let want = (11.5 + 60.0 / 190.0) * dx;
        assert!((x - want).abs() <= 1e-15, "{x} against {want}");
        let flat = vec![300.0; 20];
        assert_eq!(front_position(&flat, dx, 1500.0), None, "no crossing in an all-300 profile");
        let mut twice = vec![300.0; 20];
        twice[3] = 1600.0;
        twice[4] = 1600.0;
        twice[5] = 300.0;
        twice[6] = 1600.0;
        let x2 = front_position(&twice, dx, 1500.0).expect("two crossings");
        let first = (2.5 + 1200.0 / 1300.0) * dx;
        let second = (5.5 + 1200.0 / 1300.0) * dx;
        assert!(
            (x2 - first).abs() <= 1e-15,
            "the first crossing from the inlet is {x2}, not {second}"
        );
    }

    /// T5: the slope of an exact line, the row's formula, and the four-pair
    /// floor.
    #[test]
    fn fit_slope_recovers_a_line() {
        let fronts: Vec<(f64, f64)> =
            (0..30).map(|j| (2.0e-4 * j as f64, 3.0e-3 + 0.25 * 2.0e-4 * j as f64)).collect();
        let slope = fit_slope(&fronts, 3.0e-3).expect("a slope");
        assert!((slope - 0.25).abs() <= 1e-12, "{slope}");
        assert_eq!(flame_speed(1.0, 0.25), 0.75, "S_L = U_in - dx_f/dt");
        assert_eq!(fit_slope(&fronts[..3], 0.0), None, "three pairs are fewer than four");
    }

    /// T6: three synthetic levels of S(h) = 0.4 + 0.1 (h/1e-4)^2 feed the
    /// §94 study: p = 2 and phi_ext = 0.4.
    #[test]
    fn three_levels_feed_the_94_study() {
        let level = |h: f64| Level { h: h as Scalar, value: (0.4 + 0.1 * (h / 1.0e-4).powi(2)) as Scalar };
        let levels = vec![level(2.5e-5), level(5.0e-5), level(1.0e-4)];
        let study = vv::grid_study(&levels).expect("the study runs");
        let p = study.p.expect("an observed order");
        // Class R: the levels are `Scalar`, so under `single` the bars are
        // D-F32-1's `tol32 = max(tol64, min(tol64 2^29, 1e-5))`.
        let tol = |tol64: f64| if cfg!(feature = "single") { tol64.max((tol64 * 2f64.powi(29)).min(1e-5)) } else { tol64 };
        assert!((f64::from(p) - 2.0).abs() <= tol(1e-6), "p = {}", f64::from(p));
        assert!((f64::from(study.phi_ext) - 0.4).abs() <= tol(1e-9), "phi_ext = {}", f64::from(study.phi_ext));
    }

    /// The recorded runs as they stand (SPEC-LIT 133.12, recorded
    /// 2026-10-18): every level stopped on the reactor's negativity guard
    /// before its end time, so each level's row is MISSED. The test asserts
    /// the miss and its named cause, and prints it, so a record that
    /// changes is seen here first.
    #[test]
    fn flame1d_records_state_their_miss() {
        let runs = RUNS.as_ref().expect("the runs are recorded");
        for (r, dx) in runs.iter().zip(DX_UM) {
            assert_eq!(r.dx_um, dx, "the record's level order");
            println!("  [133.12] MISSED: {dx} um stopped at t = {} s of {END_TIME} s: {}", r.reached, r.ended);
            assert!(r.reached < END_TIME, "{dx} um: the record reached its end; update this test");
            assert!(r.ended.contains("BELOW_NEGATIVE_ATOL"), "{dx} um: the cause is named: {}", r.ended);
            assert!(fit_slope(r.fronts, FIT_FROM).is_none(), "{dx} um: no front after 3 ms");
        }
    }

    /// One written time directory's `T` profile (OpenFOAM ASCII:
    /// `internalField nonuniform List<scalar> N ( ... )`).
    fn read_t(path: &Path) -> Vec<f64> {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let body = text
            .split("internalField")
            .nth(1)
            .unwrap_or_else(|| panic!("{}: no internalField", path.display()));
        let list = body
            .split("List<scalar>")
            .nth(1)
            .unwrap_or_else(|| panic!("{}: not a scalar list", path.display()));
        let open = list.find('(').expect("the list's (");
        let close = list.rfind(')').expect("the list's )");
        list[open + 1..close]
            .split_whitespace()
            .map(|v| v.parse::<f64>().expect("a T value"))
            .collect()
    }

    /// The recorder (SPEC-LIT 133.12's split): prints the `RUNS` literal and
    /// `BUILT_AT` from the runs directory `CHR13_RUNS` - `flame_100`,
    /// `flame_50`, `flame_25`, each a run copy holding the case, `mech/`, the
    /// driver's written `flame1d_ch4_bfer_jsonc/` and `run.log`. Not part of
    /// any verification; it must not panic on a run that stopped early.
    #[test]
    #[ignore = "reads a recorded runs directory (env CHR13_RUNS); prints the record literals"]
    fn record_flame1d() {
        let dir = PathBuf::from(std::env::var("CHR13_RUNS").expect("CHR13_RUNS"));
        let levels = [("flame_100", 150), ("flame_50", 300), ("flame_25", 600)];
        let mut runs = Vec::new();
        for (k, (stem, n_cells)) in levels.into_iter().enumerate() {
            let root = dir.join(stem).join("flame1d_ch4_bfer_jsonc");
            let mut times: Vec<(f64, PathBuf)> = std::fs::read_dir(&root)
                .unwrap_or_else(|e| panic!("{}: {e}", root.display()))
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter_map(|p| {
                    let t = p.file_name()?.to_str()?.parse::<f64>().ok()?;
                    (t > 0.0).then_some((t, p))
                })
                .collect();
            times.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("times are finite"));
            let dx = LENGTH / n_cells as f64;
            let mut fronts = Vec::new();
            for (t, p) in &times {
                if let Some(x) = front_position(&read_t(&p.join("T")), dx, T_ISO) {
                    fronts.push((*t, x));
                }
            }
            let reached = times.last().map(|(t, _)| *t).unwrap_or(0.0);
            let log = std::fs::read_to_string(dir.join(stem).join("run.log")).expect("run.log");
            let ended = log
                .lines()
                .find(|l| l.starts_with("run ended:"))
                .unwrap_or("")
                .to_string();
            println!("  [133.12] {stem}: reached {reached} s, {} fronts, ended: {ended}", fronts.len());
            runs.push(FlameRun {
                dx_um: DX_UM[k],
                n_cells,
                dt: DT,
                end_time: END_TIME,
                reached,
                ended: Box::leak(ended.into_boxed_str()),
                fronts: Box::leak(fronts.into_boxed_slice()),
            });
        }
        let at = std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();
        println!("// RECORDS-BEGIN (the supervisor pastes this over the None)");
        println!("pub(crate) const BUILT_AT: &str = {at:?};");
        println!("pub(crate) static RUNS: Option<[FlameRun; 3]> = Some([");
        for r in &runs {
            println!("    FlameRun {{");
            println!("        dx_um: {:?}, n_cells: {}, dt: {:?}, end_time: {:?},", r.dx_um, r.n_cells, r.dt, r.end_time);
            println!("        reached: {:?},", r.reached);
            println!("        ended: {:?},", r.ended);
            println!("        fronts: &{:?},", r.fronts);
            println!("    }},");
        }
        println!("]);");
        println!("// RECORDS-END");
    }
}
