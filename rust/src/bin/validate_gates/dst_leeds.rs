// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Gate DST-LEEDS (SPEC-LIT §157.5): the Leeds fan-stirred cornflour bomb,
//! run live FOUR times through the sibling `ofgpu-lowmach` executable on the
//! equal-volume 1-D sector of `cases/dust/leedsBomb.jsonc` (§157.3), once at
//! each of the key's four rms velocities, and compared against Bradley, Chen
//! & Swithenbank's (1989) measured flame speeds as Huang et al. (2020) plot
//! them in their Fig. 9 (`reference/huang2020-leeds/st_b_exp.csv`, read
//! through the answer-key reader with its sha256 printed).
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §157.5 (this gate), §157.3 (the sector), §151.5
//!     (the case's inputs), (155.8) (S_t,b, the c‾ = 0.1 surface), (155.9)
//!     (the flame-Courant step), §69 (the verdict registry)
//!   D. Bradley, Z. Chen and J. R. Swithenbank, *Proc. Combust. Inst.* 22
//!     (1989) 1767-1775, DOI 10.1016/S0082-0784(89)80190-0 (not read) - the
//!     measured S_t,b of the Leeds bomb's four series
//!   C. Huang, A. N. Lipatnikov and K. Nessvi, *J. Loss Prev. Process Ind.*
//!     67 (2020) 104237, DOI 10.1016/j.jlp.2020.104237 - Fig. 9 (the
//!     digitised markers of the key) and Table 4 (the four series' k, omega)
//!
//! The specification was informed by a review of FSCDustFoam (Huang & Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use ofgpu::dust::gatemesh::{cell_radii, check_sector};
use ofgpu::error::IoContext;
use ofgpu::io::case_json::read_case_jsonc;
use ofgpu::io::fields::read_scalar_field;
use ofgpu::{Error, Result, Scalar};

use crate::{key, Checks, GateReport, How, Uncertainty, Verdict};

/// The gate's name: `validate.rs` opens the scope with this constant and
/// [`check_dst_leeds`] reports under it, so the two cannot drift apart.
pub(crate) const GATE: &str = "SPEC-LIT 157.5 Gate DST-LEEDS Leeds fan-stirred cornflour bomb";

/// The published measurement the gate is held against.
const AGAINST: &str = "Bradley, Chen & Swithenbank (1989) via Huang et al. (2020) Fig. 9, \
                       key huang2020-leeds-st-b";

/// The answer key's manifest id (SPEC-LIT §151.6, key J.1).
// answer-key: huang2020-leeds-st-b
const KEY_ID: &str = "huang2020-leeds-st-b";

/// The four series of the key (H20 Table 4), in order: the rms velocity u',
/// its k_0 and omega_0 (= eps_0/(0.09 k_0), §156.4), and the step its run
/// takes - the predicted flame-Courant number (155.9) stays under 0.5 at
/// every one and the ignition's first step stays in the low-Mach regime
/// (§157.8). The shipped case's own row is u' = 1.62 m/s.
const U_PRIME: [f64; 4] = [0.80, 1.62, 2.45, 3.31];
const K0: [f64; 4] = [0.96, 3.94, 9.00, 16.43];
const OMEGA0: [f64; 4] = [544.33, 1101.33, 1667.70, 2252.74];
const DT: [f64; 4] = [1.9e-5, 1.1e-5, 8.0e-6, 6.5e-6];

/// p_0 of §157.5: the sealed run's `-p0`.
const P0: f64 = 110_000.0;
/// The runs' end and write schedule: 60 samples, 0.5 ms apart.
const END_TIME: f64 = 0.03;
const WRITE_INTERVAL: f64 = 5.0e-4;
/// The gated radii, mm - the key's `gated = 1` rows; 20 mm is a note only
/// (H20 §3.3: ignition-affected, `gated = 0` in the key).
const R_STAR_MM: [f64; 3] = [25.0, 30.0, 35.0];
const R_NOTE_MM: f64 = 20.0;
/// The bands of §157.5: ± 20 % on S_t,b, ± 15 % on its slope, 1 % on the
/// pressure rise at R_f = 35 mm; the key's own slope of the 30 mm rows; the
/// c‾ surface R_f is measured at; the sector-mesh bars (D).
const S_BAND: f64 = 0.20;
const SLOPE_BAND: f64 = 0.15;
const DP_BAND: f64 = 0.01;
const SLOPE_KEY: f64 = 0.678;
const CBAR_SURF: f64 = 0.1;
const MESH_TOL: f64 = 1.0e-12;

/// One u' run read back off disk: the sample times ascending, `R_f` of
/// (155.8) at each, and the uniform `p0` at each. Empty when the run failed.
struct Run {
    t: Vec<f64>,
    rf: Vec<Option<f64>>,
    p0: Vec<f64>,
    seconds: f64,
}

impl Run {
    fn empty() -> Self {
        Run { t: Vec::new(), rf: Vec::new(), p0: Vec::new(), seconds: 0.0 }
    }
}

/// (155.8) as built: `R_f`, where `c‾` falls through [`CBAR_SURF`], scanned
/// from the burned side outward - the first `i` with `c‾[i] >= 0.1` and
/// `c‾[i+1] < 0.1`, linear between the two cell centres. `None` when the
/// field is unburned at the origin or the fall never comes.
fn flame_radius(r: &[f64], cbar: &[f64]) -> Option<f64> {
    match cbar.first() {
        Some(&c0) if c0 >= CBAR_SURF => {}
        _ => return None,
    }
    for i in 0..r.len().saturating_sub(1) {
        if cbar[i] >= CBAR_SURF && cbar[i + 1] < CBAR_SURF {
            let f = (cbar[i] - CBAR_SURF) / (cbar[i] - cbar[i + 1]);
            return Some(r[i] + f * (r[i + 1] - r[i]));
        }
    }
    None
}

/// `S_t,b` at `r_star`: the first bracket `rf[j] <= r_star < rf[j+1]` with
/// both ends present, the difference quotient across it - the central
/// difference of `R_f(t)` at the bracket's midpoint. `None` when no sample
/// brackets `r_star`.
fn speed_at(t: &[f64], rf: &[Option<f64>], r_star: f64) -> Option<f64> {
    for j in 0..t.len().saturating_sub(1) {
        if let (Some(a), Some(b)) = (rf[j], rf[j + 1]) {
            if a <= r_star && r_star < b {
                return Some((b - a) / (t[j + 1] - t[j]));
            }
        }
    }
    None
}

/// The first time `R_f` crosses `r` upward, linear between the bracketing
/// samples; `None` when a bracket sample is missing or `r` is never reached.
fn first_crossing(t: &[f64], rf: &[Option<f64>], r: f64) -> Option<f64> {
    for j in 0..t.len().saturating_sub(1) {
        if let (Some(a), Some(b)) = (rf[j], rf[j + 1]) {
            if a < r && r <= b {
                return Some(t[j] + (r - a) / (b - a) * (t[j + 1] - t[j]));
            }
        }
    }
    None
}

/// `y(x)`, linear between the samples; the first sample's value at or before
/// `x`; `None` when `x` lies outside them.
fn interp(t: &[f64], y: &[f64], x: f64) -> Option<f64> {
    let j = t.iter().position(|&tt| tt >= x)?;
    if j == 0 {
        return Some(y[0]);
    }
    let f = (x - t[j - 1]) / (t[j] - t[j - 1]);
    Some(y[j - 1] + f * (y[j] - y[j - 1]))
}

/// The least-squares slope `dS/du'` of the paired samples.
fn slope(u: &[f64], s: &[f64]) -> f64 {
    let n = u.len() as f64;
    let (mu, ms) = (u.iter().sum::<f64>() / n, s.iter().sum::<f64>() / n);
    let sxy: f64 = u.iter().zip(s).map(|(&a, &b)| (a - mu) * (b - ms)).sum();
    let sxx: f64 = u.iter().map(|&a| (a - mu) * (a - mu)).sum();
    sxy / sxx
}

/// The written history of one run: every sub-directory of the output root
/// whose name parses as a positive time, ascending, with `R_f` and the
/// uniform `p0` at each (`n` cells, [`Run`]).
fn read_run(root: &Path, radii: &[f64]) -> Result<Run> {
    let mut samples: Vec<(f64, PathBuf)> = std::fs::read_dir(root)
        .path(root)?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let t = e.file_name().to_str()?.parse::<f64>().ok()?;
            (t > 0.0).then_some((t, e.path()))
        })
        .collect();
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    let n = radii.len();
    let mut run = Run { t: Vec::with_capacity(samples.len()), ..Run::empty() };
    for (t, p) in samples {
        // (155.8) is host f64 in BOTH builds, so the two agree bit for bit.
        let cbar: Vec<f64> =
            read_scalar_field(&p.join("cbar"), n)?.internal.iter().map(|&v| v as f64).collect();
        let p0 = read_scalar_field(&p.join("p0"), n)?.internal;
        run.t.push(t);
        run.rf.push(flame_radius(radii, &cbar));
        run.p0.push(p0.iter().map(|&v| v as f64).sum::<f64>() / p0.len() as f64);
    }
    Ok(run)
}

/// One series' case, re-raised from the shipped one with the series' k_0 and
/// omega_0 set everywhere they appear - the initial field, the vessel patch's
/// wall values, the SST-sust ambient - and the step. Written into `dir`.
fn write_series_case(dir: &Path, i: usize) -> Result<PathBuf> {
    let shipped = Path::new(env!("CARGO_MANIFEST_DIR")).join("../cases/dust/leedsBomb.jsonc");
    let case = read_case_jsonc(&shipped)?;
    let mut v = serde_json::to_value(&case)
        .map_err(|e| Error::Config(format!("leedsBomb.jsonc does not serialise: {e}")))?;
    v["initial"]["k"] = serde_json::json!(K0[i]);
    v["initial"]["omega"] = serde_json::json!(OMEGA0[i]);
    v["dust"]["sst"]["sust"]["kAmb"] = serde_json::json!(K0[i]);
    v["dust"]["sst"]["sust"]["omegaAmb"] = serde_json::json!(OMEGA0[i]);
    v["run"]["deltaT"] = serde_json::json!(DT[i]);
    let mut vessel = false;
    if let Some(patches) = v["patches"].as_array_mut() {
        for p in patches.iter_mut() {
            if p["match"] == "vessel" {
                if let Some(k) = p.get_mut("k") {
                    k["value"] = serde_json::json!(K0[i]);
                }
                if let Some(w) = p.get_mut("omega") {
                    w["value"] = serde_json::json!(OMEGA0[i]);
                }
                vessel = true;
            }
        }
    }
    if !vessel {
        return Err(Error::Config(
            "leedsBomb.jsonc has no patch rule matching \"vessel\"".to_string(),
        ));
    }
    let path = dir.join("leedsBomb.json");
    let text = serde_json::to_string_pretty(&v)
        .map_err(|e| Error::Config(format!("the series case does not serialise: {e}")))?;
    std::fs::write(&path, text).path(&path)?;
    Ok(path)
}

/// The log's last `n` lines, for a run that failed.
fn log_tail(log: &Path, n: usize) -> Vec<String> {
    let text = std::fs::read_to_string(log).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    let skip = lines.len().saturating_sub(n);
    lines[skip..].iter().map(|l| format!("  | {l}")).collect()
}

/// One u' run: the case written, the sibling driver spawned on it to
/// t = 30 ms, the history read back. A failed or missing run is an EMPTY
/// [`Run`], its row failed, its log's last 15 lines printed, and the gate
/// goes on with the next u'.
fn run_one(
    c: &mut Checks,
    driver: &Path,
    dir: &Path,
    radii: &[f64],
    i: usize,
    missed: &mut Vec<String>,
) -> Result<Run> {
    let up = U_PRIME[i];
    let what = format!("Gate DST-LEEDS u' = {up:.2} m/s");
    let case_file = write_series_case(dir, i)?;
    let log = dir.join("run.log");
    let out = std::fs::File::create(&log).path(&log)?;
    let err = out.try_clone().path(&log)?;
    let clock = Instant::now();
    let status = Command::new(driver)
        .arg(&case_file)
        .arg("-sealed")
        .arg("-p0")
        .arg(format!("{P0}"))
        .arg("-endTime")
        .arg(format!("{END_TIME}"))
        .arg("-deltaT")
        .arg(format!("{}", DT[i]))
        .arg("-writeInterval")
        .arg(format!("{WRITE_INTERVAL}"))
        .arg("-check")
        .arg("500")
        .stdout(out)
        .stderr(err)
        .status();
    let seconds = clock.elapsed().as_secs_f64();
    let status = match status {
        Ok(s) => s,
        Err(e) => {
            c.require(&format!("{what}: the run completes"), false);
            c.note(&format!("{what}: the driver could not be spawned: {e}"));
            missed.push(format!("{what}: the driver could not be spawned: {e}"));
            return Ok(Run::empty());
        }
    };
    c.note(&format!("{what}: deltaT = {} s, wall {:.1} s, exit {}", DT[i], seconds, status));
    if !status.success() {
        c.require(&format!("{what}: the run completes"), false);
        for line in log_tail(&log, 15) {
            c.note(&line);
        }
        missed.push(format!("{what}: the run failed (exit {status}); see run.log"));
        return Ok(Run::empty());
    }
    c.require(&format!("{what}: the run completes"), true);
    let mut run = read_run(&crate::common::json_case_output_dir(&case_file), radii)?;
    run.seconds = seconds;
    Ok(run)
}

/// Gate DST-LEEDS. Opened by `validate.rs`'s section under [`GATE`].
pub(crate) fn check_dst_leeds(c: &mut Checks) -> Result<()> {
    let kf = match key::load(KEY_ID) {
        Ok(kf) => kf,
        Err(why) => {
            let why = why.to_string();
            c.note(&why);
            c.skip(
                "Gate DST-LEEDS: S_t,b against Bradley, Chen & Swithenbank's Leeds bomb",
                "the answer key is absent",
            );
            c.report(GateReport {
                verdict: Verdict::Open,
                how: How::Live,
                gate: GATE,
                against: AGAINST,
                headline: "the answer key is absent from reference/, so no comparison was run"
                    .to_string(),
                detail: vec![why],
                uncertainty: Some(Uncertainty::SingleMesh(
                    "no mesh was run: the answer key was absent",
                )),
            });
            return Ok(());
        }
    };
    c.note(&kf.digest_line());

    // The key's rows: S_t,b at (u', R_f), mm.
    let (u_k, r_k, s_k) = (kf.column("u_prime")?, kf.column("R_f_mm")?, kf.column("S_tb")?);
    let key_at = |up: f64, r_mm: f64| -> Option<f64> {
        u_k.iter().zip(&r_k).zip(&s_k).find_map(|((&uu, &rr), &ss)| {
            ((uu - up).abs() < 1e-9 && (rr - r_mm).abs() < 1e-9).then_some(ss)
        })
    };

    // The driver is the sibling executable of THIS binary.
    let driver = std::env::current_exe()
        .path("current_exe")?
        .with_file_name(format!("ofgpu-lowmach{}", std::env::consts::EXE_SUFFIX));
    if !driver.exists() {
        let why = format!("the sibling driver {} does not exist", driver.display());
        c.note(&why);
        c.skip(
            "Gate DST-LEEDS: S_t,b against Bradley, Chen & Swithenbank's Leeds bomb",
            "ofgpu-lowmach is not built",
        );
        c.report(GateReport {
            verdict: Verdict::Open,
            how: How::Live,
            gate: GATE,
            against: AGAINST,
            headline: format!("{}: build it before this gate", driver.display()),
            detail: vec![why],
            uncertainty: Some(Uncertainty::SingleMesh("no mesh was run: the driver was absent")),
        });
        return Ok(());
    }

    // The sector mesh rows of §157.3 (D), on the shipped case's own spec.
    let shipped = Path::new(env!("CARGO_MANIFEST_DIR")).join("../cases/dust/leedsBomb.jsonc");
    let lowered = read_case_jsonc(&shipped)?.lower()?;
    let spec = lowered.sector.clone().ok_or_else(|| {
        Error::Config("cases/dust/leedsBomb.jsonc has no mesh.sector block (SPEC-LIT 157.3)".into())
    })?;
    let radii = cell_radii(&spec)?;
    let mesh = check_sector(&spec)?;
    c.note(&format!(
        "the sector: {} cells, R = {} mm, angle = {} deg; closure {:.2e}, area/r2 {:.2e}",
        mesh.cells,
        spec.radius * 1e3,
        spec.angle_deg,
        mesh.closure_max,
        mesh.area_r2_max
    ));
    c.check(
        "Gate DST-LEEDS: the sector's worst relative cell closure",
        mesh.closure_max as Scalar,
        MESH_TOL as Scalar,
    );
    c.check(
        "Gate DST-LEEDS: the worst |A/(4 a^2 x^2) - 1| of the radial squares",
        mesh.area_r2_max as Scalar,
        MESH_TOL as Scalar,
    );

    // The four runs, one after another.
    let mut missed: Vec<String> = Vec::new();
    let mut runs: Vec<Run> = Vec::new();
    let mut scratch: Vec<PathBuf> = Vec::new();
    let clock_all = Instant::now();
    for i in 0..4 {
        let dir = std::env::temp_dir()
            .join(format!("ofgpu_dst_leeds_{}_{}", std::process::id(), i));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).path(&dir)?;
        scratch.push(dir.clone());
        runs.push(run_one(c, &driver, &dir, &radii, i, &mut missed)?);
    }
    c.note(&format!(
        "four runs, {:.1} s wall total ({})",
        clock_all.elapsed().as_secs_f64(),
        runs.iter().map(|r| format!("{:.1}", r.seconds)).collect::<Vec<_>>().join(" + ")
    ));

    // The comparison: S_t,b of (155.8) at R* = 20 mm (a note, gated = 0 in
    // the key) and at the three gated radii, ± 20 %.
    c.note("   u'    R*    S_model    S_key  |S/S_key - 1|");
    for (i, run) in runs.iter().enumerate() {
        let up = U_PRIME[i];
        for &r_mm in std::iter::once(&R_NOTE_MM).chain(R_STAR_MM.iter()) {
            let s_model = speed_at(&run.t, &run.rf, r_mm * 1e-3);
            let s_key = key_at(up, r_mm);
            let cell = s_model.map_or("not reached".to_string(), |s| format!("{s:.4}"));
            let ratio = match (s_model, s_key) {
                (Some(s), Some(k)) => (s / k - 1.0).abs(),
                _ => f64::INFINITY,
            };
            c.note(&format!(
                "{:5.2} {:5.1} {:>9} {:7.3} {:13.3}{}",
                up,
                r_mm,
                cell,
                s_key.unwrap_or(f64::NAN),
                ratio,
                if r_mm == R_NOTE_MM { "  (not gated)" } else { "" }
            ));
            if r_mm == R_NOTE_MM {
                continue;
            }
            let what =
                format!("Gate DST-LEEDS u' = {up:.2} m/s R_f = {r_mm:.0} mm: |S_t,b/S_key - 1|");
            c.check(&what, ratio as Scalar, S_BAND as Scalar);
            if !(ratio <= S_BAND) {
                missed.push(format!(
                    "{what}: S_model {cell} against the key's {:.3}",
                    s_key.unwrap_or(f64::NAN)
                ));
            }
        }
    }

    // The slope at R_f = 30 mm over the four u', ± 15 % of the key's 0.678.
    let s30: Vec<Option<f64>> =
        runs.iter().map(|r| speed_at(&r.t, &r.rf, 30.0 * 1e-3)).collect();
    if s30.iter().all(Option::is_some) {
        let s: Vec<f64> = s30.iter().map(|s| s.unwrap()).collect();
        let sl = slope(&U_PRIME, &s);
        c.note(&format!(
            "S_t,b at 30 mm: {}; least-squares slope {:.4} against the key's {SLOPE_KEY}",
            s.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(", "),
            sl
        ));
        let what = "Gate DST-LEEDS: the least-squares dS_t,b/du' at 30 mm";
        c.check(what, (sl / SLOPE_KEY - 1.0).abs() as Scalar, SLOPE_BAND as Scalar);
        if !((sl / SLOPE_KEY - 1.0).abs() <= SLOPE_BAND) {
            missed.push(format!("{what}: {sl:.4} against {SLOPE_KEY}"));
        }
    } else {
        c.require(
            "Gate DST-LEEDS: the slope at 30 mm is computed (every u' reaches R_f = 30 mm)",
            false,
        );
        missed.push("the slope at 30 mm: a series never reached R_f = 30 mm".to_string());
    }

    // The pressure rise at R_f = 35 mm, per u', below 1 %.
    for (i, run) in runs.iter().enumerate() {
        let up = U_PRIME[i];
        let what = format!("Gate DST-LEEDS u' = {up:.2} m/s: (p0 - p_0)/p_0 at R_f = 35 mm");
        let dp = first_crossing(&run.t, &run.rf, 35.0 * 1e-3)
            .and_then(|ts| interp(&run.t, &run.p0, ts))
            .map_or(f64::INFINITY, |p| (p - P0) / P0);
        c.check(&what, dp as Scalar, DP_BAND as Scalar);
        if !(dp <= DP_BAND) {
            let at = if dp.is_finite() { format!("{dp:.4}") } else { "never reached".to_string() };
            missed.push(format!("{what}: {at}"));
        }
    }

    if !missed.is_empty() {
        c.report(GateReport {
            verdict: Verdict::Misses,
            how: How::Live,
            gate: GATE,
            against: AGAINST,
            headline: format!(
                "{} of the gate's rows leave the bands of §157.5",
                missed.len()
            ),
            detail: missed,
            uncertainty: Some(Uncertainty::SingleMesh(
                "one sector mesh, H20's centre resolution; no refinement sequence in this run",
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

    /// (155.8): the crossing is taken from the BURNED side outward - the
    /// first fall through 0.1 - and there is no radius when the field is
    /// unburned at the origin or never falls.
    #[test]
    fn dst_leeds_flame_radius_takes_the_burned_side_crossing() {
        let r = [0.5, 1.5, 2.5, 3.5];
        let got = flame_radius(&r, &[1.0, 0.6, 0.05, 0.0]).unwrap();
        assert!((got - (1.5 + 0.5 / 0.55)).abs() < 1e-12, "{got}");
        // The burned cell is i = 0 here: the first fall, not the first rise.
        let got = flame_radius(&r, &[1.0, 0.05, 0.5, 0.0]).unwrap();
        assert!((got - (0.5 + 0.9 / 0.95)).abs() < 1e-12, "{got}");
        // Unburned at the origin, and never falling.
        assert_eq!(flame_radius(&r, &[0.05, 0.0, 0.0, 0.0]), None);
        assert_eq!(flame_radius(&r, &[1.0, 1.0, 1.0, 1.0]), None);
    }

    /// `S_t,b` is the difference quotient across the FIRST bracketing pair
    /// of samples, and there is none when a sample of the bracket is missing
    /// or `r_star` lies beyond the history.
    #[test]
    fn dst_leeds_speed_is_the_bracketing_difference() {
        let t: Vec<f64> = (0..10).map(|j| 0.5e-3 * j as f64).collect();
        let rf: Vec<Option<f64>> = t.iter().map(|&ti| Some(1.0e-3 + 2.0 * ti)).collect();
        let s = speed_at(&t, &rf, 4.0e-3).unwrap();
        assert!((s - 2.0).abs() <= 1e-12, "{s}");
        let mut hole = rf.clone();
        hole[3] = None;
        assert_eq!(speed_at(&t, &hole, 4.0e-3), None);
        assert_eq!(speed_at(&t, &rf, 2.0e-2), None);
    }

    /// Each series' case, written from the shipped one through serde, reads
    /// back and lowers with that series' k_0, omega_0 and step everywhere the
    /// gate sets them - the path every live run starts from, with no GPU.
    #[test]
    fn dst_leeds_series_cases_lower_with_their_turbulence() {
        let dir = std::env::temp_dir().join(format!("ofgpu_dst_leeds_test_{}", std::process::id()));
        for i in 0..4 {
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let path = write_series_case(&dir, i).expect("the series case is written");
            let case = read_case_jsonc(&path).expect("the series case reads back");
            let v = serde_json::to_value(&case).unwrap();
            assert_eq!(v["initial"]["k"].as_f64(), Some(K0[i]));
            assert_eq!(v["initial"]["omega"].as_f64(), Some(OMEGA0[i]));
            assert_eq!(v["run"]["deltaT"].as_f64(), Some(DT[i]));
            let vessel = v["patches"].as_array().unwrap().iter().find(|p| p["match"] == "vessel").unwrap();
            assert_eq!(vessel["k"]["value"].as_f64(), Some(K0[i]));
            assert_eq!(vessel["omega"]["value"].as_f64(), Some(OMEGA0[i]));
            let l = case.lower().expect("the series case lowers");
            let sst = l.dust.as_ref().and_then(|d| d.sst.as_ref()).expect("dust.sst");
            assert_eq!(sst.sust, Some((K0[i], OMEGA0[i])));
            assert_eq!(l.sector.as_ref().map(|s| s.angle_deg), Some(90.0));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The key's own least-squares slope of the R_f = 30 mm rows is 0.678
    /// (SPEC-LIT §151.6) - the number the gate's slope check is held to.
    #[test]
    fn dst_leeds_key_slope_at_30_mm_is_0_678() {
        let kf = key::load(KEY_ID).expect("the Leeds key loads");
        let u = kf.column("u_prime").unwrap();
        let r = kf.column("R_f_mm").unwrap();
        let s = kf.column("S_tb").unwrap();
        let mut su = Vec::new();
        let mut ss = Vec::new();
        for j in 0..u.len() {
            if (r[j] - 30.0).abs() < 1e-9 {
                su.push(u[j]);
                ss.push(s[j]);
            }
        }
        assert_eq!(su.len(), 4, "the four R_f = 30 mm rows");
        let sl = slope(&su, &ss);
        assert!((sl - 0.678).abs() <= 5e-4, "{sl}");
    }
}
