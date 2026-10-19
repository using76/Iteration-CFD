// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Gate DST-B-LAM (SPEC-LIT §159.11): the Phase B dust cloud's 1-D laminar
//! flame, run live SIX times through the sibling `ofgpu-lowmach` - the gas
//! flame of the cloud's own volatile mixture, then the cloud itself at
//! d = 1 um ((i), the fine-particle limit) and d = 5, 10, 20 and 40 um
//! ((ii), the size order). The speed is (159.23): the front's position the
//! first upward crossing of the 1000 K isotherm counted from the closed end,
//! linear between cell centres, `S_L` the least-squares slope over the
//! writes of [20, 40] ms, negated - the front runs toward the closed end.
//! The gas variant is (159.24): the case with its `dust` block removed and
//! the cloud's mass put in the gas as VOL at the VOL-to-air mass ratio the
//! converted cloud leaves. (iii), the measured speed against Dahoe et al.
//! 2002 via (152.13), is NAMED and NOT RUN - no open corn-starch
//! devolatilisation kinetics (§151.8).
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §159.11 (this gate as built), §133.12 ((133.25),
//!     the front-speed formula with the inflow speed 0), §152.1 (the
//!     mixture's stoichiometry) and §151.8 (the kinetics (iii) would need)
//!
//! The specification was informed by a review of FSCDustFoam (Huang & Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use ofgpu::chem::thermo::R_GAS;
use ofgpu::error::IoContext;
use ofgpu::io::case_json::read_case_jsonc;
use ofgpu::io::fields::read_scalar_field;
use ofgpu::{Error, Result, Scalar};

use crate::validate_gates::flame1d::{fit_slope, front_position};
use crate::{Checks, GateReport, How, Uncertainty, Verdict};

/// The gate's name: `validate.rs` opens the scope with this constant and
/// [`check_dst_b_laminar`] reports under it, so the two cannot drift apart.
pub(crate) const GATE: &str = "SPEC-LIT 159.11 Gate DST-B-LAM Phase B laminar dust flame: the fine-particle limit and the size order";

/// The case, relative to the repository root - the tests' input.
pub(crate) const CASE: &str = "cases/dust/flame1dCloud.jsonc";
/// The isotherm of (159.23), K - the tests' locator input.
pub(crate) const T_ISO: f64 = 1000.0;
/// The slope's fit starts here, s - the window's open end.
pub(crate) const FIT_FROM: f64 = 0.020;
/// The write interval the spawn asks for, s - 0.5 ms, 41 window fronts.
pub(crate) const WRITE: f64 = 5.0e-4;
/// The ambient pressure the tube starts at, Pa - rho_air's and `-p0`'s.
pub(crate) const P0: f64 = 101_325.0;
/// (i)'s band on |S_L,B(1 um)/S_L,gas - 1|.
pub(crate) const BAND_I: f64 = 0.02;
/// (ii)'s diameters, um - (i)'s 1 um run is the series' first member.
pub(crate) const SIZES_UM: [f64; 4] = [5.0, 10.0, 20.0, 40.0];
/// The tube's cells along x and its length, m - the case's own mesh.
pub(crate) const CELLS: usize = 120;
pub(crate) const LENGTH: f64 = 0.006;

/// The driver's normal-end wording (`lowmach.rs`'s `run_end_line`): a run
/// only "completes" when the exit is a success AND the log ends this way.
const ENDED_OK: &str = "run ended: budget";

/// What the gate's rows are held against: its own bands of 159.11 - (i)'s
/// 2 % on the fine-particle limit and (ii)'s non-increasing size order. No
/// published measurement is compared here; (iii) names one and is NOT RUN.
const AGAINST: &str = "(i) |S_L,B(1 um)/S_L,gas - 1| within 2 % and (ii) S_L(d_{j+1}) <= \
     S_L(d_j) along d = 5, 10, 20, 40 um - the gate's own bands; no published measurement";

/// (159.24)'s air density from the case's mechanism records' molar masses
/// (kg/mol): `p_0 / (R0 T_0 (y_O2/W_O2 + (1 - y_O2)/W_N2))` - the case's
/// O2/N2 molar mix at its own `initial.T`.
pub(crate) fn air_density(w_o2: f64, w_n2: f64, y_o2: f64, p0: f64, t0: f64) -> f64 {
    p0 / (R_GAS * t0 * (y_o2 / w_o2 + (1.0 - y_o2) / w_n2))
}

/// (159.24)'s gas variant of `v`: the copy without its `dust` key whose
/// `species.initial` carries the cloud's mass as VOL at the VOL-to-air mass
/// ratio the converted cloud leaves - `Y_VOL = C/(rho_air + C)` and
/// `Y_O2 = y_O2 rho_air/(rho_air + C)`, N2 the rest. Every other key is the
/// shipped case's, unchanged.
pub(crate) fn gas_variant(v: &serde_json::Value, rho_air: f64) -> Result<serde_json::Value> {
    let conc = v["dust"]["cloud"]["concentration"].as_f64().ok_or_else(|| {
        Error::Config(format!("{CASE} has no dust.cloud.concentration"))
    })?;
    let y_o2 = v["species"]["initial"]["O2"].as_f64().ok_or_else(|| {
        Error::Config(format!("{CASE} has no species.initial.O2"))
    })?;
    let mut g = v.clone();
    g.as_object_mut()
        .ok_or_else(|| Error::Config(format!("{CASE} is not a JSON object")))?
        .remove("dust");
    let denom = rho_air + conc;
    g["species"]["initial"]["VOL"] = serde_json::json!(conc / denom);
    g["species"]["initial"]["O2"] = serde_json::json!(y_o2 * rho_air / denom);
    Ok(g)
}

/// A copy of `v` whose cloud's `diameter` is `d` (m) - one member of (ii)'s
/// size series; everything else is the shipped case's.
pub(crate) fn with_diameter(v: &serde_json::Value, d: f64) -> serde_json::Value {
    let mut w = v.clone();
    w["dust"]["cloud"]["diameter"] = serde_json::json!(d);
    w
}

/// (ii)'s measure between the series' consecutive speeds: the relative rise
/// `(s[j+1] - s[j])/s[j]`, floored at 0 - a fall IS the row's pass, 0.
pub(crate) fn rises(s: &[f64]) -> Vec<f64> {
    s.windows(2).map(|w| ((w[1] - w[0]) / w[0]).max(0.0)).collect()
}

/// The output root's sample directories whose names parse as times,
/// ascending - the schedule `-writeInterval` asked for.
fn samples(root: &Path) -> Vec<(f64, PathBuf)> {
    let mut s: Vec<(f64, PathBuf)> = std::fs::read_dir(root)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter_map(|e| {
                    let t = e.file_name().to_str()?.parse::<f64>().ok()?;
                    (t > 0.0).then_some((t, e.path()))
                })
                .collect()
        })
        .unwrap_or_default();
    s.sort_by(|a, b| a.0.total_cmp(&b.0));
    s
}

/// The log's last `n` lines, for a run that failed.
fn log_tail(log: &Path, n: usize) -> Vec<String> {
    let text = std::fs::read_to_string(log).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    let skip = lines.len().saturating_sub(n);
    lines[skip..].iter().map(|l| format!("  | {l}")).collect()
}

/// One run's outcome for the gate's rows: its S_L - `None` with fewer than
/// 4 fronts in the window, the row already failed then - and the run's
/// wall seconds. (The two class D rows per run are `run_case`'s.)
struct RunOutcome {
    s_l: Option<f64>,
    seconds: f64,
}

/// One run: `case` written into `<scratch>/<tag>.json` with the chemistry
/// paths rewritten to the repository's own mechanism files (absolute - the
/// scratch dir is not the case dir), the sibling driver spawned on it with
/// the case's own end time and step, then every written time's `T` read
/// back and (159.23)'s front tracked through it. The two class D rows are
/// emitted here; a failed run fails its rows, notes its log's tail, and
/// the gate goes on.
fn run_case(
    c: &mut Checks,
    driver: &Path,
    scratch: &Path,
    tag: &str,
    case: &serde_json::Value,
    missed: &mut Vec<String>,
) -> Result<RunOutcome> {
    let what = format!("Gate DST-B-LAM {tag}");
    let end_time = case["run"]["endTime"]
        .as_f64()
        .ok_or_else(|| Error::Config(format!("{tag}.json has no run.endTime")))?;
    let delta_t = case["run"]["deltaT"]
        .as_f64()
        .ok_or_else(|| Error::Config(format!("{tag}.json has no run.deltaT")))?;
    let mech = Path::new(env!("CARGO_MANIFEST_DIR")).join("../cases/mech/volatile1.ck");
    let thermo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../cases/mech/tm4513_thermo.dat");
    let mut v = case.clone();
    v["chemistry"]["mechanism"] = serde_json::json!(mech);
    v["chemistry"]["thermo"] = serde_json::json!(thermo);
    let file = scratch.join(format!("{tag}.json"));
    let text = serde_json::to_string_pretty(&v)
        .map_err(|e| Error::Config(format!("the run case does not serialise: {e}")))?;
    std::fs::write(&file, text).path(&file)?;
    let log = scratch.join("run.log");
    let out = std::fs::File::create(&log).path(&log)?;
    let err = out.try_clone().path(&log)?;
    let clock = Instant::now();
    let status = Command::new(driver)
        .arg(&file)
        .arg("-p0")
        .arg(format!("{P0}"))
        .arg("-endTime")
        .arg(format!("{end_time}"))
        .arg("-deltaT")
        .arg(format!("{delta_t}"))
        .arg("-check")
        .arg("100")
        .arg("-writeInterval")
        .arg(format!("{WRITE}"))
        .stdout(out)
        .stderr(err)
        .status();
    let seconds = clock.elapsed().as_secs_f64();
    let completed = match status {
        Err(e) => {
            let why = format!("{what}: the run completes: the driver could not be spawned: {e}");
            c.require(&format!("{what}: the run completes"), false);
            missed.push(format!("MISSED: {why}"));
            false
        }
        Ok(s) if !s.success() => {
            c.require(&format!("{what}: the run completes"), false);
            for line in log_tail(&log, 15) {
                c.note(&line);
            }
            missed.push(format!(
                "MISSED: {what}: the run completes: the run failed (exit {s}); the bar is exit 0 \
                 and a \"{}\" line",
                ENDED_OK
            ));
            false
        }
        Ok(_) => {
            let ended = std::fs::read_to_string(&log).unwrap_or_default().contains(ENDED_OK);
            c.require(&format!("{what}: the run completes"), ended);
            if !ended {
                for line in log_tail(&log, 15) {
                    c.note(&line);
                }
                missed.push(format!(
                    "MISSED: {what}: the run completes: exit 0 but the log holds no \"{}\" line",
                    ENDED_OK
                ));
            }
            ended
        }
    };
    // (159.23): every written time's `T` read back as f64 (host in BOTH
    // builds), the first 1000 K crossing from the closed end tracked, the
    // speed the least-squares slope over [FIT_FROM, 40 ms], negated.
    let root = crate::common::json_case_output_dir(&file);
    let mut fronts: Vec<(f64, f64)> = Vec::new();
    if completed {
        for (t, p) in samples(&root) {
            // A time directory without `T` holds no front (nothing else is
            // written without the fields); it is skipped, not fatal.
            if !p.join("T").is_file() {
                continue;
            }
            let field = read_scalar_field(&p.join("T"), CELLS)?;
            let profile: Vec<f64> = field.internal.iter().map(|&v| v as f64).collect();
            if let Some(x) = front_position(&profile, LENGTH / CELLS as f64, T_ISO) {
                fronts.push((t, x));
            }
        }
    }
    let fronts_window = fronts.iter().filter(|(t, _)| *t >= FIT_FROM).count();
    let s_l = fit_slope(&fronts, FIT_FROM).map(|slope| -slope);
    let what4 = format!("{what}: >= 4 fronts in [20, 40] ms");
    c.require(&what4, fronts_window >= 4);
    if fronts_window < 4 {
        missed.push(format!("MISSED: {what4}: {fronts_window} fronts, the bar is 4"));
    }
    c.note(&format!(
        "{what}: wall {seconds:.1} s, {fronts_window} fronts in [20, 40] ms, S_L = {} m/s",
        s_l.map_or("none".to_string(), |s| format!("{s:.6}"))
    ));
    Ok(RunOutcome { s_l, seconds })
}

/// Gate DST-B-LAM. Opened by `validate.rs`'s section under [`GATE`].
pub(crate) fn check_dst_b_laminar(c: &mut Checks) -> Result<()> {
    let clock_all = Instant::now();
    // The shipped case, as a Value for the variant builds and as a lowered
    // case for its mechanism records: rho_air comes from the case's own
    // W_O2 and W_N2 at its own initial T (159.24).
    let shipped = Path::new(env!("CARGO_MANIFEST_DIR")).join("../").join(CASE);
    let case = read_case_jsonc(&shipped)?;
    let v = serde_json::to_value(&case)
        .map_err(|e| Error::Config(format!("{CASE} does not serialise: {e}")))?;
    let lowered = case.lower()?;
    let chem = lowered
        .chem
        .as_ref()
        .ok_or_else(|| Error::Config(format!("{CASE} lowers with no chemistry")))?;
    let mech = chem.load(shipped.parent().expect("the case's directory"))?;
    let y_o2 = v["species"]["initial"]["O2"]
        .as_f64()
        .ok_or_else(|| Error::Config(format!("{CASE} has no species.initial.O2")))?;
    let t0 = v["initial"]["T"]
        .as_f64()
        .ok_or_else(|| Error::Config(format!("{CASE} has no initial.T")))?;
    let conc = v["dust"]["cloud"]["concentration"]
        .as_f64()
        .ok_or_else(|| Error::Config(format!("{CASE} has no dust.cloud.concentration")))?;
    let w_o2 = mech
        .thermo()
        .get("O2")
        .ok_or_else(|| Error::Config(format!("{CASE}'s mechanism has no O2 record")))?
        .molar_mass;
    let w_n2 = mech
        .thermo()
        .get("N2")
        .ok_or_else(|| Error::Config(format!("{CASE}'s mechanism has no N2 record")))?
        .molar_mass;
    let rho_air = air_density(w_o2, w_n2, y_o2, P0, t0);
    let gas = gas_variant(&v, rho_air)?;
    c.note(&format!(
        "Gate DST-B-LAM: C = {conc} kg/m3, rho_air = {rho_air:.6} kg/m3 (W_O2 = {:.4}, \
         W_N2 = {:.4} g/mol, T0 = {t0} K, p0 = {P0} Pa), the gas variant's Y_VOL = {:.7}, \
         Y_O2 = {:.7}",
        w_o2 * 1.0e3,
        w_n2 * 1.0e3,
        gas["species"]["initial"]["VOL"].as_f64().unwrap_or(f64::NAN),
        gas["species"]["initial"]["O2"].as_f64().unwrap_or(f64::NAN),
    ));

    // The driver is the sibling executable of THIS binary.
    let driver = std::env::current_exe()
        .path("current_exe")?
        .with_file_name(format!("ofgpu-lowmach{}", std::env::consts::EXE_SUFFIX));
    if !driver.exists() {
        c.skip(
            "Gate DST-B-LAM: the cloud's 1-D laminar flame against the gas flame of its volatile, \
             runs (i) and (ii)",
            "ofgpu-lowmach is not built",
        );
        return Ok(());
    }

    // The six runs, in order: the gas flame, the shipped cloud (i), then
    // (ii)'s four coarser clouds. One scratch dir per run, removed at the end.
    let runs: [(&str, serde_json::Value); 6] = [
        ("gas", gas),
        ("1 um", v.clone()),
        ("5 um", with_diameter(&v, SIZES_UM[0] * 1.0e-6)),
        ("10 um", with_diameter(&v, SIZES_UM[1] * 1.0e-6)),
        ("20 um", with_diameter(&v, SIZES_UM[2] * 1.0e-6)),
        ("40 um", with_diameter(&v, SIZES_UM[3] * 1.0e-6)),
    ];
    let mut scratch: Vec<PathBuf> = Vec::new();
    let mut missed: Vec<String> = Vec::new();
    let mut out: Vec<RunOutcome> = Vec::with_capacity(6);
    for (i, (tag, case)) in runs.iter().enumerate() {
        let dir =
            std::env::temp_dir().join(format!("ofgpu_dst_b_lam_{}_{}", std::process::id(), i));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).path(&dir)?;
        scratch.push(dir.clone());
        out.push(run_case(c, &driver, &dir, tag, case, &mut missed)?);
    }

    // (i) the fine-particle limit: the 1 um cloud's speed at the gas
    // flame's, within 2 %.
    let what_i = "Gate DST-B-LAM (i): |S_L(1 um)/S_L(gas) - 1|";
    match (out[0].s_l, out[1].s_l) {
        (Some(sg), Some(s1)) if sg > 0.0 => {
            let err = (s1 / sg - 1.0).abs();
            c.note(&format!(
                "Gate DST-B-LAM (i): S_L(gas) = {sg:.6} m/s, S_L(1 um) = {s1:.6} m/s, \
                 ratio {:.6}, the bar {BAND_I}",
                s1 / sg
            ));
            c.check_p(what_i, err as Scalar, BAND_I as Scalar);
            if !(err <= BAND_I) {
                missed.push(format!("MISSED: {what_i}: {err:.6}, the bar is {BAND_I}"));
            }
        }
        _ => {
            c.require(what_i, false);
            missed.push(format!(
                "MISSED: {what_i}: a speed is missing (gas {}, 1 um {}), the bar is {BAND_I}",
                out[0].s_l.map_or("none".to_string(), |s| format!("{s:.6}")),
                out[1].s_l.map_or("none".to_string(), |s| format!("{s:.6}")),
            ));
        }
    }

    // (ii) the size order: S_L(d_{j+1}) <= S_L(d_j) along the series, err
    // the relative rise (a fall IS the pass, 0).
    let sizes = ["5 um", "10 um", "20 um", "40 um"];
    let series: Vec<Option<f64>> = (2..6).map(|i| out[i].s_l).collect();
    if let Some(s) = series
        .iter()
        .copied()
        .collect::<Option<Vec<f64>>>()
        .map(|v| rises(&v))
    {
        for (j, rise) in s.iter().enumerate() {
            let what = format!(
                "Gate DST-B-LAM (ii): S_L({}) <= S_L({}), relative rise",
                sizes[j + 1], sizes[j]
            );
            c.note(&format!(
                "Gate DST-B-LAM (ii): S_L({}) = {:.6} against S_L({}) = {:.6}, rise {rise:.6}",
                sizes[j + 1],
                series[j + 1].unwrap_or(f64::NAN),
                sizes[j],
                series[j].unwrap_or(f64::NAN),
            ));
            c.check_d(&what, *rise as Scalar, 0.0);
            if !(*rise <= 0.0) {
                missed.push(format!("MISSED: {what}: {rise:.6}, the bar is 0"));
            }
        }
    } else {
        let missing = sizes
            .iter()
            .zip(series.iter())
            .filter(|(_, s)| s.is_none())
            .map(|(n, _)| *n)
            .collect::<Vec<_>>()
            .join(", ");
        let what = "Gate DST-B-LAM (ii): S_L(d_{j+1}) <= S_L(d_j) along the size series";
        c.require(what, false);
        missed.push(format!(
            "MISSED: {what}: no speed for {missing} (each row's bar is a rise of 0)"
        ));
    }

    // (iii) the measured speed, NAMED and NOT RUN.
    c.skip(
        "Gate DST-B-LAM (iii): S_L against Dahoe et al. 2002 via (152.13), 0.241 m/s at phi 0.62",
        "NAMED and NOT RUN: no open corn-starch devolatilisation kinetics (SPEC-LIT 151.8)",
    );

    c.note(&format!(
        "six runs, {:.1} s wall total ({})",
        clock_all.elapsed().as_secs_f64(),
        out.iter()
            .map(|o| format!("{:.1}", o.seconds))
            .collect::<Vec<_>>()
            .join(" + ")
    ));
    // Every failed row's MISSED note, carried by the registry - §69.2 lets
    // the word through no other way.
    if !missed.is_empty() {
        c.report(GateReport {
            verdict: Verdict::Misses,
            how: How::Live,
            gate: GATE,
            against: AGAINST,
            headline: format!("{} of the gate's rows leave the bands of 159.11", missed.len()),
            detail: missed,
            uncertainty: Some(Uncertainty::SingleMesh(
                "one 120-cell tube per run, as the section runs it; no refinement sequence in \
                 this run",
            )),
        });
    }
    for d in &scratch {
        let _ = std::fs::remove_dir_all(d);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ofgpu::io::case_json::read_case_jsonc_str;

    /// The shipped case lowers to exactly the inputs the gate runs on: the
    /// 120-cell, 6 mm tube at 40 ms and dt 5e-5 with VOL's Le 2, and the
    /// all-volatile 1 um cloud at C = 0.15418 with A = 1.5e7, Ea 7.4e4,
    /// dH 0. Host only, no GPU.
    #[test]
    fn the_cloud_case_lowers_to_the_gate_inputs() {
        let shipped = Path::new(env!("CARGO_MANIFEST_DIR")).join("../").join(CASE);
        let case = read_case_jsonc(&shipped).expect("the case reads");
        let v = serde_json::to_value(&case).expect("serialises");
        assert_eq!(v["mesh"]["cells"][0], serde_json::json!(CELLS));
        assert_eq!(v["mesh"]["bounds"]["max"][0], serde_json::json!(LENGTH));
        assert_eq!(v["run"]["endTime"], serde_json::json!(0.04));
        assert_eq!(v["run"]["deltaT"], serde_json::json!(5e-5));
        assert_eq!(v["species"]["transport"]["Le"]["VOL"], serde_json::json!(2.0));
        let lowered = case.lower().expect("lowers");
        let cloud = lowered.cloud.as_ref().expect("the cloud");
        assert!((cloud.concentration - 0.15418).abs() <= 1e-12, "{}", cloud.concentration);
        assert_eq!(cloud.diameters, vec![1.0e-6]);
        let devol = cloud.devol.as_ref().expect("the devol");
        assert_eq!((devol.a, devol.e_a, devol.dh), (1.5e7, 7.4e4, 0.0));
        assert_eq!(
            (cloud.composition.moisture, cloud.composition.volatile, cloud.composition.char,
             cloud.composition.ash),
            (0.0, 1.0, 0.0, 0.0)
        );
    }

    /// The gas variant (159.24) carries the cloud as VOL at the same
    /// VOL-to-air mass ratio, hits the hand values at W_O2 31.998 and
    /// W_N2 28.014 g/mol, and serialises and lowers without its cloud.
    #[test]
    fn the_gas_variant_carries_the_cloud_as_vol() {
        let shipped = Path::new(env!("CARGO_MANIFEST_DIR")).join("../").join(CASE);
        let case = read_case_jsonc(&shipped).expect("the case reads");
        let v = serde_json::to_value(&case).expect("serialises");
        let lowered = case.lower().expect("lowers");
        let chem = lowered.chem.as_ref().expect("the chem");
        let mech = chem.load(shipped.parent().expect("the case's dir")).expect("loads");
        let w_o2 = mech.thermo().get("O2").expect("an O2 record").molar_mass;
        let w_n2 = mech.thermo().get("N2").expect("an N2 record").molar_mass;
        let y_o2 = v["species"]["initial"]["O2"].as_f64().expect("O2");
        let t0 = v["initial"]["T"].as_f64().expect("T");
        let conc = v["dust"]["cloud"]["concentration"].as_f64().expect("C");
        let rho_air = air_density(w_o2, w_n2, y_o2, P0, t0);
        let gas = gas_variant(&v, rho_air).expect("the variant");
        assert!(gas.get("dust").is_none(), "the variant has no dust");
        let y_vol = gas["species"]["initial"]["VOL"].as_f64().expect("VOL");
        let y_o2g = gas["species"]["initial"]["O2"].as_f64().expect("O2");
        // The variant's mass ratio is the cloud's own: Y_VOL/Y_O2 = C/(0.233 rho_air).
        let ratio_want = conc / (0.233 * rho_air);
        assert!(
            (y_vol / y_o2g / ratio_want - 1.0).abs() <= 1e-12,
            "{}/{} against {}",
            y_vol, y_o2g, ratio_want
        );
        // The hand values (W_O2 31.998, W_N2 28.014 g/mol).
        assert!((y_vol / 0.116_2601 - 1.0).abs() <= 1e-4, "Y_VOL {y_vol}");
        assert!((y_o2g / 0.205_9114 - 1.0).abs() <= 1e-4, "Y_O2 {y_o2g}");
        let text = serde_json::to_string(&gas).expect("serialises");
        let gcase = read_case_jsonc_str(&text, "the gas variant").expect("reads");
        let gl = gcase.lower().expect("lowers");
        assert!(gl.cloud.is_none(), "the variant lowers with no cloud");
        assert!(gl.chem.is_some(), "the variant lowers with its chemistry");
    }

    /// (159.23) on an exact ramp: a 1700 K ramp whose 1000 K point sits at
    /// x_c(t) = 5 mm - 0.05 t over the case's own mesh gives S_L = 0.05 m/s
    /// to round-off - the interpolation and the least squares are exact on
    /// a front that is linear in x and moves uniformly.
    #[test]
    fn the_front_speed_of_a_ramp_is_exact() {
        let dx = LENGTH / CELLS as f64;
        let mut fronts: Vec<(f64, f64)> = Vec::new();
        for k in 0..=80 {
            let t = 5.0e-4 * k as f64; // 0, 0.5 ms, ..., 40 ms - the writes
            let x_c = 5.0e-3 - 0.05 * t;
            let profile: Vec<f64> = (0..CELLS)
                .map(|i| {
                    let x = (i as f64 + 0.5) * dx;
                    (1000.0 + 1700.0 * (x - x_c) / 1.0e-3).clamp(300.0, 2000.0)
                })
                .collect();
            if let Some(x) = front_position(&profile, dx, T_ISO) {
                fronts.push((t, x));
            }
        }
        assert_eq!(fronts.len(), 81, "every write crosses");
        let s_l = -fit_slope(&fronts, FIT_FROM).expect("a slope");
        assert!((s_l / 0.05 - 1.0).abs() <= 1e-9, "S_L {s_l}");
    }

    /// (ii)'s rise counts a rise and not a fall: a fall IS the pass, 0,
    /// and a rise comes out as its own relative value.
    #[test]
    fn the_size_rows_count_a_rise() {
        assert_eq!(rises(&[0.05, 0.049, 0.049, 0.045]), vec![0.0, 0.0, 0.0]);
        let got = rises(&[0.05, 0.051, 0.04, 0.041]);
        let want = [0.02, 0.0, 0.025];
        assert_eq!(got.len(), 3);
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() <= 1e-12, "{g} against {w}");
        }
    }
}
