// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Gate DST-VES (SPEC-LIT §157.6): the closed vessels, run live FOUR times
//! through the sibling `ofgpu-lowmach` executable - the 20 L and 1 m3
//! spheres of (i) on the 1-D sector of `cases/dust/{sphere20L, sphere1m3}.jsonc`
//! at the prescribed `U_t = S_u(p)`, held against
//! `ofgpu::dust::vessel0d`'s host integration of the thin-flame pressure
//! equation (157.4) (the time to (p0 + pe)/2, (dp/dt)_max, the AICC end
//! pressure, the cube-root law (157.5), the brush of (155.6)); the
//! constant-rho_u negative control that must MISS (dp/dt)_max by >= 10 %
//! (INSIGHTS (g)); and the REMBE closed stage of (ii) on the equal-volume
//! sphere of `cases/dust/rembeClosed.jsonc`, its overpressure against
//! Huang, Bloching & Lipatnikov (2022) Fig. 10 (`reference/huang2022-rembe/
//! p_closed_exp.csv`, read through the answer-key reader with its sha256
//! printed) - a REPRODUCTION row: H22 guessed and adjusted u' and L on
//! this curve (H22, section 3.6).
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §157.6 (this gate), §157.2 ((157.2)-(157.5), the
//!     thin-flame relations `dust::vessel0d` integrates), (155.6) (the
//!     brush), (155.10) (the isentropic compression), §157.9 ((157.10) the
//!     reference's quadrature, (157.11) its anchor), §69 (the verdict
//!     registry)
//!   C. Huang, M. Bloching and A. N. Lipatnikov, *J. Loss Prev. Process
//!     Ind.* 75 (2022) 104707, DOI 10.1016/j.jlp.2021.104707 (CC BY 4.0)
//!     - Fig. 10, the digitised overpressure of the key
//!
//! The specification was informed by a review of FSCDustFoam (Huang & Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use ofgpu::dust::gatemesh::cell_radii;
use ofgpu::dust::vessel0d::{self, BurningLaw, UnburnedDensity, Vessel};
use ofgpu::error::IoContext;
use ofgpu::io::case_json::read_case_jsonc;
use ofgpu::io::fields::read_scalar_field;
use ofgpu::{Error, Result, Scalar};

use crate::{key, Checks, GateReport, How, Uncertainty, Verdict};

/// The gate's name: `validate.rs` opens the scope with this constant and
/// [`check_dst_vessel`] reports under it, so the two cannot drift apart.
pub(crate) const GATE: &str = "SPEC-LIT 157.6 Gate DST-VES closed vessels: thin-flame spheres and the REMBE closed stage";

/// The published measurement the REMBE rows are held against.
const AGAINST: &str = "Huang, Bloching & Lipatnikov (2022) Fig. 10, key huang2022-rembe-p-closed";

/// The answer key's manifest id (SPEC-LIT §151.6, key J.2).
// answer-key: huang2022-rembe-p-closed
const KEY_ID: &str = "huang2022-rembe-p-closed";

/// The four runs, in order: (case, `-p0`, `-writeInterval`, control). The
/// spheres sit at the ambient; REMBE carries key J.2's initial overpressure
/// of 0.06 bar (§151.7 item 9). The control is sphere20L re-raised with
/// `combustion.unburnedDensity "initial"` - it must MISS (dp/dt)_max.
const RUNS: [(&str, f64, f64, bool); 4] = [
    ("sphere20L", 101_325.0, 0.01, false),
    ("sphere1m3", 101_325.0, 0.03684, false),
    ("sphere20L", 101_325.0, 0.01, true),
    ("rembeClosed", 107_325.0, 0.01, false),
];

/// The ambient pressure the spheres start at and the overpressures come off.
const P_AMBIENT: f64 = 101_325.0;
/// The bands of §157.6: 1 % on t_half, 2 % on (dp/dt)_max, 1e-3 on p_end
/// against the AICC p_e, the brush at 5 cells of R/1000, the cube-root law
/// at 2 %, the control's miss at 10 %, ± 0.009 bar on the REMBE
/// overpressures and ± 0.01 s on the 0.1 bar crossing at 0.10 s.
const T_HALF_BAND: f64 = 0.01;
const DPDT_BAND: f64 = 0.02;
const P_END_BAND: f64 = 1.0e-3;
const BRUSH_BAND: f64 = 5.0e-3;
const KST_BAND: f64 = 0.02;
const CONTROL_MISS: f64 = 0.10;
const DP_BAND: f64 = 0.009;
const T01_BAND: f64 = 0.01;
const T01_TARGET: f64 = 0.10;
/// The anchor: the first history point whose burned fraction (157.2) has
/// reached this.
const X_ANCHOR: f64 = 1.0e-3;

/// The history of one run, parsed from run.log's `iter` lines (the
/// `print_report` layout of `src/bin/lowmach.rs`): the step N after `iter`,
/// the number after `p0 ` and the number after `dp0/dt `, the step's time
/// (N + 1) deltaT - the loop adds dt to the clock BEFORE each step.
#[derive(Debug, Clone, Default)]
struct History {
    t: Vec<f64>,
    p0: Vec<f64>,
    dpdt: Vec<f64>,
    seconds: f64,
}

impl History {
    fn is_empty(&self) -> bool {
        self.p0.is_empty()
    }

    /// The largest logged dp0/dt - the run's own (dp/dt)_max.
    fn dpdt_max(&self) -> Option<f64> {
        self.dpdt.iter().copied().reduce(f64::max)
    }
}

/// Every `iter` line of `text` as one [`History`] point; every other line
/// is ignored. `p0` and `dp0/dt` are read as the tokens AFTER those names,
/// so a layout change upstream shows up as an absent point, never as a
/// wrong number.
fn parse_history(text: &str, delta_t: f64) -> History {
    let mut h = History::default();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("iter") else { continue };
        let mut toks = rest.split_whitespace();
        let Some(n) = toks.next().and_then(|s| s.parse::<f64>().ok()) else {
            continue;
        };
        let (mut p0, mut dp, mut prev) = (None, None, "");
        for tok in toks {
            if prev == "p0" {
                p0 = tok.parse::<f64>().ok();
            }
            if prev == "dp0/dt" {
                dp = tok.parse::<f64>().ok();
            }
            prev = tok;
        }
        if let (Some(p), Some(d)) = (p0, dp) {
            h.t.push((n + 1.0) * delta_t);
            h.p0.push(p);
            h.dpdt.push(d);
        }
    }
    h
}

/// `y`'s first upward crossing of `target`, linear between the bracketing
/// samples; `None` when it never crosses.
fn cross_up(t: &[f64], y: &[f64], target: f64) -> Option<f64> {
    for j in 1..t.len() {
        if y[j - 1] < target && target <= y[j] {
            let f = (target - y[j - 1]) / (y[j] - y[j - 1]);
            return Some(t[j - 1] + f * (t[j] - t[j - 1]));
        }
    }
    None
}

/// `y(x)`, linear between the samples; the first sample's value at or
/// before `x`; `None` when `x` lies outside them.
fn interp(t: &[f64], y: &[f64], x: f64) -> Option<f64> {
    let j = t.iter().position(|&tt| tt >= x)?;
    if j == 0 {
        return Some(y[0]);
    }
    let f = (x - t[j - 1]) / (t[j] - t[j - 1]);
    Some(y[j - 1] + f * (y[j] - y[j - 1]))
}

/// A row's relative deviation `|run/reference - 1|`: every band of §157.6
/// is two-sided, so a run faster or higher than its reference misses a band
/// exactly as a slower or lower one does.
fn rel_dev(run: f64, reference: f64) -> f64 {
    (run / reference - 1.0).abs()
}

/// (155.6): the brush thickness `1 / max|grad cbar|`, the finite-difference
/// gradient over the cell radii; `None` when the field is flat.
fn brush_delta_t(r: &[f64], cbar: &[f64]) -> Option<f64> {
    let mut g = 0.0_f64;
    for i in 0..r.len().saturating_sub(1) {
        g = g.max((cbar[i + 1] - cbar[i]).abs() / (r[i + 1] - r[i]));
    }
    (g > 0.0).then_some(1.0 / g)
}

/// The host trace's anchor: the first history index `k >= 1` whose burned
/// fraction (157.2) has reached [`X_ANCHOR`] - the state the run's own p0
/// hands the (157.4) integration as its start.
fn anchor(v: &Vessel, h: &History) -> Option<usize> {
    (1..h.p0.len()).find(|&k| v.burned_fraction(h.p0[k]) >= X_ANCHOR)
}

/// The host reference of one sphere case: the vessel of (157.2)-(157.5)
/// from the case's own thermo, the prescribed `U_t` law and the sector's
/// cell radii - all of it host f64 in BOTH builds, so the two agree.
struct SphereHost {
    v: Vessel,
    law: BurningLaw,
    radii: Vec<f64>,
}

impl SphereHost {
    /// Built from the shipped case's own lowering - the SAME lowering the
    /// live run starts from, so the host and the run read one `S_u(p)`.
    fn build(name: &str) -> Result<SphereHost> {
        const W: &str = "Gate DST-VES: the shipped case";
        let shipped =
            Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../cases/dust/{name}.jsonc"));
        let lowered = read_case_jsonc(&shipped)?.lower()?;
        let d = lowered
            .dust
            .as_ref()
            .ok_or_else(|| Error::Config(format!("{W} {name}.jsonc has no dust block")))?;
        let spec = lowered.sector.clone().ok_or_else(|| {
            Error::Config(format!("{W} {name}.jsonc has no mesh.sector block"))
        })?;
        let law = d.prescribed_ut.ok_or_else(|| {
            Error::Config(format!("{W} {name}.jsonc has no combustion.prescribedUt"))
        })?;
        Ok(SphereHost {
            v: Vessel::from_gas(&d.gas()?, d.t_u, P_AMBIENT, spec.radius)?,
            law,
            radii: cell_radii(&spec)?,
        })
    }
}

/// The host reference of one sphere run: (157.4) integrated from the run's
/// own anchor at deltaT/10, in `mode`, its metrics. `None` when the anchor
/// or the trace is unavailable - the caller has already row-failed the run.
fn host_reference(
    host: &SphereHost,
    h: &History,
    delta_t: f64,
    mode: UnburnedDensity,
) -> Option<vessel0d::ThinFlameMetrics> {
    let k = anchor(&host.v, h)?;
    let tr = vessel0d::integrate(&host.v, &host.law, mode, h.t[k], h.p0[k], delta_t / 10.0).ok()?;
    vessel0d::metrics(&host.v, &host.law, mode, &tr).ok()
}

/// One run's case, re-raised from the shipped one through serde - the
/// control with `combustion.unburnedDensity` set to `"initial"` - and
/// written into `dir`. Returns the file and the case's own end time and
/// step, which the spawn passes straight back to the driver.
fn write_run_case(dir: &Path, name: &str, control: bool) -> Result<(PathBuf, f64, f64)> {
    let shipped =
        Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../cases/dust/{name}.jsonc"));
    let case = read_case_jsonc(&shipped)?;
    let mut v = serde_json::to_value(&case)
        .map_err(|e| Error::Config(format!("{name}.jsonc does not serialise: {e}")))?;
    if control {
        v["combustion"]["unburnedDensity"] = serde_json::json!("initial");
    }
    let end_time = v["run"]["endTime"]
        .as_f64()
        .ok_or_else(|| Error::Config(format!("{name}.jsonc has no run.endTime")))?;
    let delta_t = v["run"]["deltaT"]
        .as_f64()
        .ok_or_else(|| Error::Config(format!("{name}.jsonc has no run.deltaT")))?;
    let path = dir.join(format!("{name}.json"));
    let text = serde_json::to_string_pretty(&v)
        .map_err(|e| Error::Config(format!("the run case does not serialise: {e}")))?;
    std::fs::write(&path, text).path(&path)?;
    Ok((path, end_time, delta_t))
}

/// The log's last `n` lines, for a run that failed.
fn log_tail(log: &Path, n: usize) -> Vec<String> {
    let text = std::fs::read_to_string(log).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    let skip = lines.len().saturating_sub(n);
    lines[skip..].iter().map(|l| format!("  | {l}")).collect()
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

/// `cbar` of the sample whose time is nearest `t`, as f64 (host in BOTH
/// builds), with that sample's time; `None` when nothing was written.
fn cbar_at(root: &Path, t: f64, n: usize) -> Result<Option<(f64, Vec<f64>)>> {
    let (ts, p) = match samples(root)
        .into_iter()
        .min_by(|a, b| (a.0 - t).abs().total_cmp(&(b.0 - t).abs()))
    {
        Some(x) => x,
        None => return Ok(None),
    };
    let f = read_scalar_field(&p.join("cbar"), n)?;
    Ok(Some((ts, f.internal.iter().map(|&v| v as f64).collect())))
}

/// One run: the case written, the sibling driver spawned on it to the
/// case's own end time, the history parsed off run.log's `iter` lines.
/// A failed or unparseable run is an EMPTY history, its row failed, the
/// log's last 15 lines printed, and the gate goes on with the next run.
fn run_one(
    c: &mut Checks,
    driver: &Path,
    dir: &Path,
    i: usize,
    missed: &mut Vec<String>,
) -> Result<(PathBuf, f64, f64, History)> {
    let (name, p0, write, control) = RUNS[i];
    let what =
        format!("Gate DST-VES run {i} ({name}{})", if control { ", control" } else { "" });
    let (case_file, end_time, delta_t) = write_run_case(dir, name, control)?;
    let log = dir.join("run.log");
    let out = std::fs::File::create(&log).path(&log)?;
    let err = out.try_clone().path(&log)?;
    let clock = Instant::now();
    let status = Command::new(driver)
        .arg(&case_file)
        .arg("-sealed")
        .arg("-p0")
        .arg(format!("{p0}"))
        .arg("-endTime")
        .arg(format!("{end_time}"))
        .arg("-deltaT")
        .arg(format!("{delta_t}"))
        .arg("-writeInterval")
        .arg(format!("{write}"))
        .arg("-check")
        .arg("1")
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
            return Ok((case_file, end_time, delta_t, History { seconds, ..History::default() }));
        }
    };
    c.note(&format!(
        "{what}: endTime = {end_time} s, deltaT = {delta_t} s, wall {seconds:.1} s, exit {status}"
    ));
    let mut h = History { seconds, ..History::default() };
    if !status.success() {
        c.require(&format!("{what}: the run completes"), false);
        for line in log_tail(&log, 15) {
            c.note(&line);
        }
        missed.push(format!("{what}: the run failed (exit {status}); see run.log"));
        return Ok((case_file, end_time, delta_t, h));
    }
    let text = std::fs::read_to_string(&log).unwrap_or_default();
    h = History { seconds, ..parse_history(&text, delta_t) };
    if h.is_empty() {
        c.require(&format!("{what}: the run completes"), false);
        c.note(&format!("{what}: the log carries no iter lines; see run.log"));
        missed.push(format!("{what}: the log carries no iter lines; see run.log"));
    } else {
        c.require(&format!("{what}: the run completes"), true);
    }
    Ok((case_file, end_time, delta_t, h))
}

/// The (i) rows of one sphere run: the host trace of (157.4) from the
/// run's own anchor, then t_half, (dp/dt)_max, p_end and the brush of
/// (155.6) at the half-time sample. Returns the run's largest logged
/// dp0/dt, for the cube-root-law row; `Ok(None)` when there is nothing to
/// compare (the run's own rows are already failed then).
fn sphere_rows(
    c: &mut Checks,
    tag: &str,
    host: &SphereHost,
    h: &History,
    root: &Path,
    delta_t: f64,
    missed: &mut Vec<String>,
) -> Result<Option<f64>> {
    let k = match anchor(&host.v, h) {
        Some(k) => k,
        None => {
            let what = format!("Gate DST-VES {tag}: p0 reaches x_b = 1e-3");
            c.require(&what, false);
            missed.push(format!("{what}: the burned fraction never gets there in the history"));
            return Ok(None);
        }
    };
    let (t_s, p_s) = (h.t[k], h.p0[k]);
    let m = match vessel0d::integrate(
        &host.v,
        &host.law,
        UnburnedDensity::Compressed,
        t_s,
        p_s,
        delta_t / 10.0,
    )
    .and_then(|tr| vessel0d::metrics(&host.v, &host.law, UnburnedDensity::Compressed, &tr).map(|m| (tr, m)))
    {
        Ok((_, m)) => m,
        Err(e) => {
            let what = format!("Gate DST-VES {tag}: the host trace integrates");
            c.require(&what, false);
            missed.push(format!("{what}: refused: {e}"));
            return Ok(None);
        }
    };
    c.note(&format!(
        "Gate DST-VES {tag}: anchor t_s = {:.6} s, p_s = {:.1} Pa (x_b = {:.4}); host t_half \
         = {:.6} s, (dp/dt)_max = {:.3e} Pa/s, AICC p_e = {:.1} Pa",
        t_s, p_s, host.v.burned_fraction(p_s), m.t_half, m.dpdt_max, host.v.pe
    ));
    let dpdt_run = match h.dpdt_max() {
        Some(d) => d,
        None => return Ok(None),
    };
    let what_t = format!("Gate DST-VES {tag}: |t_half run/host - 1|");
    match cross_up(&h.t, &h.p0, (host.v.p0 + host.v.pe) / 2.0) {
        Some(t_half) => {
            let r = rel_dev(t_half, m.t_half);
            c.note(&format!(
                "Gate DST-VES {tag}: t_half run {t_half:.6} s against the host's {:.6} s",
                m.t_half
            ));
            c.check(&what_t, r as Scalar, T_HALF_BAND as Scalar);
            if !(r <= T_HALF_BAND) {
                missed.push(format!(
                    "{what_t}: run {t_half:.6} s against the host's {:.6} s",
                    m.t_half
                ));
            }
            // The brush (155.6) at the sample nearest the half time, at most
            // 5 cells of R/1000.
            let what_b = format!("Gate DST-VES {tag}: brush delta_t/R");
            let brushed = cbar_at(root, t_half, host.radii.len())?
                .and_then(|(ts, cb)| brush_delta_t(&host.radii, &cb).map(|d| (ts, d)));
            match brushed {
                Some((ts, d)) => {
                    let cells = d / (host.v.radius * 1.0e-3);
                    c.note(&format!(
                        "Gate DST-VES {tag}: brush delta_t = {:.5} m = {:.1} cells of R/1000 at \
                         the t = {ts:.4} s sample",
                        d, cells
                    ));
                    c.check(&what_b, (d / host.v.radius) as Scalar, BRUSH_BAND as Scalar);
                    if !(d / host.v.radius <= BRUSH_BAND) {
                        missed.push(format!("{what_b}: {:.1} cells of R/1000", cells));
                    }
                }
                None => {
                    c.require(&format!("Gate DST-VES {tag}: cbar varies at the half time"), false);
                    missed.push(format!("{what_b}: no varying cbar at the half-time sample"));
                }
            }
        }
        None => {
            c.require(&what_t, false);
            missed.push(format!(
                "{what_t}: p0 never reaches (p0 + pe)/2 = {:.1} Pa",
                (host.v.p0 + host.v.pe) / 2.0
            ));
        }
    }
    let what_d = format!("Gate DST-VES {tag}: |(dp/dt)_max run/host - 1|");
    let r_d = rel_dev(dpdt_run, m.dpdt_max);
    c.check(&what_d, r_d as Scalar, DPDT_BAND as Scalar);
    if !(r_d <= DPDT_BAND) {
        missed.push(format!(
            "{what_d}: run {dpdt_run:.3e} against the host's {:.3e} Pa/s",
            m.dpdt_max
        ));
    }
    let p_end = *h.p0.last().unwrap_or(&f64::NAN);
    let what_p = format!("Gate DST-VES {tag}: |p_end/p_e - 1|");
    let r_p = rel_dev(p_end, host.v.pe);
    c.check(&what_p, r_p as Scalar, P_END_BAND as Scalar);
    if !(r_p <= P_END_BAND) {
        missed.push(format!(
            "{what_p}: p_end {p_end:.1} Pa against the AICC p_e {:.1} Pa",
            host.v.pe
        ));
    }
    c.note(&format!(
        "Gate DST-VES {tag}: K_st run = {:.2} bar m/s (closed form {:.2}), (dp/dt)_max run \
         {:.3e} against host {:.3e} Pa/s, p_end {p_end:.1} Pa",
        host.v.k_st(dpdt_run) / 1.0e5,
        host.v.k_st_closed(&host.law) / 1.0e5,
        dpdt_run,
        m.dpdt_max
    ));
    Ok(Some(dpdt_run))
}

/// Gate DST-VES. Opened by `validate.rs`'s section under [`GATE`].
pub(crate) fn check_dst_vessel(c: &mut Checks) -> Result<()> {
    let kf = match key::load(KEY_ID) {
        Ok(kf) => kf,
        Err(why) => {
            let why = why.to_string();
            c.note(&why);
            c.skip(
                "Gate DST-VES: the closed vessels against the thin-flame law and Huang et al. 2022",
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

    // The driver is the sibling executable of THIS binary.
    let driver = std::env::current_exe()
        .path("current_exe")?
        .with_file_name(format!("ofgpu-lowmach{}", std::env::consts::EXE_SUFFIX));
    if !driver.exists() {
        let why = format!("the sibling driver {} does not exist", driver.display());
        c.note(&why);
        c.skip(
            "Gate DST-VES: the closed vessels against the thin-flame law and Huang et al. 2022",
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
    // The host references, from the shipped cases' own lowering.
    let host20 = SphereHost::build("sphere20L")?;
    let host1m3 = SphereHost::build("sphere1m3")?;
    for (tag, host) in [("sphere20L", &host20), ("sphere1m3", &host1m3)] {
        c.note(&format!(
            "Gate DST-VES {tag}: R = {:.4} m, V = {:.4} m3, p0 = {:.1} Pa, AICC p_e = {:.1} Pa \
             (pe/p0 = {:.9}), K_st closed-form = {:.2} bar m/s",
            host.v.radius,
            host.v.volume(),
            host.v.p0,
            host.v.pe,
            host.v.pe / host.v.p0,
            host.v.k_st_closed(&host.law) / 1.0e5
        ));
    }

    // The key's gated rows: t_s, dp_bar with gated = 1 (t = 0.05..0.10 s).
    let (t_k, dp_k, g_k) = (kf.column("t_s")?, kf.column("dp_bar")?, kf.column("gated")?);

    // The four runs, one after another.
    let mut missed: Vec<String> = Vec::new();
    let mut scratch: Vec<PathBuf> = Vec::new();
    let mut roots: Vec<PathBuf> = Vec::new();
    let mut ends: Vec<f64> = Vec::new();
    let mut dts: Vec<f64> = Vec::new();
    let mut hist: Vec<History> = Vec::new();
    let clock_all = Instant::now();
    for i in 0..RUNS.len() {
        let dir =
            std::env::temp_dir().join(format!("ofgpu_dst_vessel_{}_{}", std::process::id(), i));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).path(&dir)?;
        scratch.push(dir.clone());
        let (case_file, end_time, delta_t, h) = run_one(c, &driver, &dir, i, &mut missed)?;
        roots.push(crate::common::json_case_output_dir(&case_file));
        ends.push(end_time);
        dts.push(delta_t);
        hist.push(h);
    }
    c.note(&format!(
        "four runs, {:.1} s wall total ({})",
        clock_all.elapsed().as_secs_f64(),
        hist.iter().map(|h| format!("{:.1}", h.seconds)).collect::<Vec<_>>().join(" + ")
    ));
    // The two spheres' rows against their host traces.
    let dp0 = sphere_rows(c, "sphere20L", &host20, &hist[0], &roots[0], dts[0], &mut missed)?;
    let dp1 = sphere_rows(c, "sphere1m3", &host1m3, &hist[1], &roots[1], dts[1], &mut missed)?;

    // The cube-root law (157.5): K_st of the two runs' own (dp/dt)_max,
    // equal within 2 % for a thin flame.
    let what = "Gate DST-VES: |K_st(20 L)/K_st(1 m3) - 1|";
    let kst = match (dp0, dp1) {
        (Some(a), Some(b)) => {
            let (k20, k1) = (host20.v.k_st(a), host1m3.v.k_st(b));
            c.note(&format!(
                "Gate DST-VES cube-root law: K_st(20 L) = {:.2}, K_st(1 m3) = {:.2} bar m/s \
                 (run (dp/dt)_max x V^(1/3))",
                k20 / 1.0e5,
                k1 / 1.0e5
            ));
            Some(rel_dev(k20, k1))
        }
        _ => None,
    };
    match kst {
        Some(r) => {
            c.check(what, r as Scalar, KST_BAND as Scalar);
            if !(r <= KST_BAND) {
                missed.push(format!("{what}: {r:.4}"));
            }
        }
        None => {
            c.require(what, false);
            missed.push(format!("{what}: a sphere run carried no history"));
        }
    }

    // The negative control (INSIGHTS (g)): a constant rho_u must MISS the
    // host reference by >= 10 % - a control that passes is a failed gate.
    let what = "Gate DST-VES: the negative control misses (dp/dt)_max by >= 10 %";
    let d_ref = host_reference(&host20, &hist[0], dts[0], UnburnedDensity::Compressed)
        .map(|m| m.dpdt_max);
    match hist[2].dpdt_max().zip(d_ref) {
        Some((d2, d0)) => {
            let miss = 1.0 - d2 / d0;
            let host_ratio = host_reference(&host20, &hist[2], dts[2], UnburnedDensity::Initial)
                .map(|m| m.dpdt_max / d0);
            c.note(&format!(
                "Gate DST-VES control: (dp/dt)_max {d2:.3e} against the reference {d0:.3e} Pa/s, \
                 miss {:.1} %; the host's own Initial mode at the same anchor: {}",
                100.0 * miss,
                host_ratio.map_or("not available".to_string(), |r| format!(
                    "ratio {:.4} (miss {:.1} %)",
                    r,
                    100.0 * (1.0 - r)
                ))
            ));
            c.require(what, miss >= CONTROL_MISS);
            if !(miss >= CONTROL_MISS) {
                missed.push(format!("{what}: miss {miss:.4}"));
            }
        }
        None => {
            c.require(what, false);
            missed.push(format!("{what}: the control or the reference carried no history"));
        }
    }
    // (ii) The REMBE closed stage (run 3), overpressure at the key's gated
    // rows - a REPRODUCTION row: H22 guessed and adjusted u' and L on this
    // curve (H22, section 3.6).
    c.note("REPRODUCTION row: H22 guessed and adjusted u' and L on this curve (H22 3.6)");
    if hist[3].is_empty() {
        let what = "Gate DST-VES REMBE: the run completes with a history";
        c.require(what, false);
        missed.push(format!("{what}: no iter lines"));
    } else {
        c.note("     t   dp_run  dp_key  |dp_run - dp_key|");
        for j in 0..t_k.len() {
            if g_k[j] < 0.5 {
                continue;
            }
            let dp_run =
                interp(&hist[3].t, &hist[3].p0, t_k[j]).map_or(f64::NAN, |p| (p - P_AMBIENT) / 1.0e5);
            let dev = (dp_run - dp_k[j]).abs();
            c.note(&format!("{:6.2} {:8.4} {:7.4} {:18.5}", t_k[j], dp_run, dp_k[j], dev));
            let what = format!("Gate DST-VES REMBE t = {:.2} s: |dp_run - dp_key|, bar", t_k[j]);
            c.check(&what, dev as Scalar, DP_BAND as Scalar);
            if !(dev <= DP_BAND) {
                missed.push(format!("{what}: run {dp_run:.4} against the key's {:.4}", dp_k[j]));
            }
        }
        // The 0.1 bar overpressure's time, within ± 0.01 s of 0.10 s.
        let what = "Gate DST-VES REMBE: |t(0.1 bar overpressure) - 0.10 s|, s";
        match cross_up(&hist[3].t, &hist[3].p0, P_AMBIENT + 1.0e4) {
            Some(t01) => {
                c.note(&format!("Gate DST-VES REMBE: 0.1 bar overpressure at t = {t01:.4} s"));
                c.check(what, (t01 - T01_TARGET).abs() as Scalar, T01_BAND as Scalar);
                if !((t01 - T01_TARGET).abs() <= T01_BAND) {
                    missed.push(format!("{what}: t = {t01:.4} s"));
                }
            }
            None => {
                let why = format!(
                    "Gate DST-VES REMBE: 0.1 bar overpressure not reached by t = {:.2} s",
                    ends[3]
                );
                c.require(&why, false);
                missed.push(why);
            }
        }
    }
    if !missed.is_empty() {
        c.report(GateReport {
            verdict: Verdict::Misses,
            how: How::Live,
            gate: GATE,
            against: AGAINST,
            headline: format!(
                "{} of the gate's rows leave the bands of §157.6",
                missed.len()
            ),
            detail: missed,
            uncertainty: Some(Uncertainty::SingleMesh(
                "one REDUCED sector per vessel (200 cells for the spheres, R/200; 182 for REMBE)",
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

    /// The parser reads the exact `print_report` layout of
    /// `src/bin/lowmach.rs`: the step after `iter`, the numbers after
    /// `p0 ` and `dp0/dt `, the time (N + 1) deltaT - and every line that
    /// is not an `iter` line is ignored.
    #[test]
    fn history_parses_the_iter_lines() {
        let line = |n: usize, p0: &str, dp: &str| {
            format!(
                "iter {n:6}  |U| res 1.2e-3  |p| res 3.4e-9  contErr 5e-7  \
                 T [300.1, 2100] K  rho [1.2, 1.25] kg/m3  p0 {p0} Pa  dp0/dt {dp} Pa/s  \
                 M max 0.02 (cell 7) mean 0.001"
            )
        };
        let text = format!(
            "{}\nTime = 0.0005 s\n\n{}\n",
            line(4, "102179.3", "4.4e7"),
            line(9, "103000", "5e7")
        );
        let h = parse_history(&text, 1.0e-4);
        assert_eq!(h.t.len(), 2);
        assert_eq!(h.t[0], 5.0 * 1.0e-4);
        assert_eq!(h.t[1], 10.0 * 1.0e-4);
        assert_eq!(h.p0, vec![102179.3, 103000.0]);
        assert_eq!(h.dpdt, vec![4.4e7, 5.0e7]);
        assert_eq!(h.dpdt_max(), Some(5.0e7));
        assert!(parse_history("no iter lines here\n", 1.0e-4).is_empty());
    }

    /// Every band of §157.6 is two-sided: a run 10 % FASTER (or higher)
    /// than its reference misses a 1 % or 2 % band as a run 10 % slower
    /// does. A signed `run/ref - 1` would pass any fast run.
    #[test]
    fn a_fast_run_misses_as_a_slow_one_does() {
        assert!((rel_dev(0.9, 1.0) - 0.1).abs() <= 1e-15);
        assert!((rel_dev(1.1, 1.0) - 0.1).abs() <= 1e-15);
        assert!(!(rel_dev(0.9, 1.0) <= T_HALF_BAND));
        assert!(!(rel_dev(0.95, 1.0) <= KST_BAND));
    }

    /// The crossing and the brush on synthetic data: p = 1 + t sampled at
    /// t = 0, 0.1, .., 1 crosses 1.55 at 0.55, and a two-cell brush on unit
    /// cells is 2 - both exact.
    #[test]
    fn crossing_and_brush_on_synthetic_data() {
        let t: Vec<f64> = (0..=10).map(|j| 0.1 * j as f64).collect();
        let p: Vec<f64> = t.iter().map(|&x| 1.0 + x).collect();
        let got = cross_up(&t, &p, 1.55).unwrap();
        assert!((got - 0.55).abs() <= 1e-12, "{got}");
        assert_eq!(cross_up(&t, &p, 2.5), None);
        let r: Vec<f64> = (0..5).map(|j| j as f64).collect();
        let d = brush_delta_t(&r, &[1.0, 1.0, 0.5, 0.0, 0.0]).unwrap();
        assert!((d - 2.0).abs() <= 1e-12, "{d}");
        assert_eq!(brush_delta_t(&r, &[1.0; 5]), None);
    }

    /// The three shipped cases lower to exactly the inputs the gate runs
    /// on: the spheres' vessels at the ambient p_0 with the AICC ratio
    /// 9.41167414637292 and radii 0.1684 / 0.6204 m, REMBE's equal-volume
    /// sphere of R = 1.4 m with no prescribed law (it runs the FSC
    /// closures). Host only, no GPU.
    #[test]
    fn the_vessel_cases_lower_to_the_gate_inputs() {
        for (name, r) in [("sphere20L", 0.1684), ("sphere1m3", 0.6204)] {
            let h = SphereHost::build(name).expect(name);
            assert!((h.v.radius - r).abs() <= 1e-12, "{}: {}", name, h.v.radius);
            let ratio = h.v.pe / h.v.p0;
            assert!((ratio / 9.41167414637292 - 1.0).abs() <= 1e-10, "{name}: {ratio}");
        }
        let shipped =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../cases/dust/rembeClosed.jsonc");
        let l = read_case_jsonc(&shipped).expect("rembeClosed.jsonc reads").lower().expect("lowers");
        let d = l.dust.as_ref().expect("the dust block");
        assert!((l.sector.as_ref().expect("the sector").radius - 1.4).abs() <= 1e-12);
        assert!(d.prescribed_ut.is_none(), "REMBE runs the FSC closures");
    }
}
