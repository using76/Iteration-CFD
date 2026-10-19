// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Gate DST-REMBE (SPEC-LIT §158.7): the vented REMBE explosion of
//! Huang, Bloching & Lipatnikov (2022) Fig. 11, key J.3. The vessel's
//! dimensions are NOT published (H22; SPEC-LIT §151.7 item 9), so the gate's
//! four comparisons - the peak overpressure (0.60 bar, within 10 %), its
//! time (0.31 s, within 0.015 s), the stage-(iii) rise rate (4.55 bar/s over
//! 0.20 to 0.28 s, within 15 %) and the rupture time (0.10 s, within
//! 0.01 s) - are NAMED and NOT RUN, four `skip` rows, and nothing here is
//! ever compared with that figure.
//!
//! What the gate does run is a REDUCED SMOKE of the case side on the
//! labelled PLACEHOLDER vessel of `cases/dust/rembeVented.geometry.json`
//! (SPEC-LIT §158.10): the siblings `ofgpu-automesher` (castellation only)
//! and `ofgpu-compressible` are spawned on the shipped
//! `cases/dust/rembeVented.jsonc`, the mesh is read back (its cell count, the
//! membrane faces, the sealed-region walk of (158.10j) of SPEC-LIT 158.10),
//! the run is a handful of steps with the vent's pStat replaced by the one
//! input [`SMOKE_P_STAT`], and the rows are: every written field finite, the
//! membrane opened exactly once, a pressure peak after the opening by
//! (158.10i) of SPEC-LIT 158.10, and the mass budget (158.10h) of SPEC-LIT
//! 158.10 closed to 1e-10 relative (class A; the expected single-build miss
//! is asserted and printed, never loosened).
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §158.7 (this gate), §158.10 ((158.10a)-(158.10j),
//!     the placeholder, the driver beside a vent, the probe series and the
//!     mass budget), §155.3 (the ignition and the flame-Courant rule the
//!     smoke's step must keep), §69 (the verdict registry)
//!   C. Huang, M. Bloching and A. N. Lipatnikov, *J. Loss Prev. Process Ind.*
//!     75 (2022) 104707, DOI 10.1016/j.jlp.2021.104707 (CC BY 4.0), Fig. 11
//!     (key J.3), whose bands are NOT RUN
//!
//! The specification was informed by a review of FSCDustFoam (Huang & Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use ofgpu::dust::gatemesh::{
    automesher_config, membrane_plane, read_vessel_geometry, seal_reach, vessel_stl,
    VesselGeometry,
};
use ofgpu::dust::vent::{select_faces, VentSelect};
use ofgpu::error::IoContext;
use ofgpu::io::case_json::read_case_jsonc;
use ofgpu::io::fields::{read_scalar_field, read_vector_field};
use ofgpu::io::polymesh::{build_host_mesh, read_poly_mesh};
use ofgpu::mesh::HostMesh;
use ofgpu::{Error, Result, Scalar};

use crate::{key, Checks, GateReport, How, Uncertainty, Verdict};

/// The gate's name: `validate.rs` opens the scope with this constant and
/// [`check_dst_rembe`] reports under it, so the two cannot drift apart.
pub(crate) const GATE: &str =
    "SPEC-LIT 158.7 Gate DST-REMBE: the vented REMBE explosion, REDUCED PLACEHOLDER smoke on the castellated vessel, key J.3 NOT RUN";

/// The published measurement the (NOT RUN) comparisons are held against.
const AGAINST: &str =
    "Huang, Bloching & Lipatnikov (2022) Fig. 11, key huang2022-rembe-p-vented";

/// The answer key's manifest id (SPEC-LIT §151.6, key J.3).
// answer-key: huang2022-rembe-p-vented
const KEY_ID: &str = "huang2022-rembe-p-vented";

/// The smoke's step count `-iters`: an input chosen by measurement, not a
/// tolerance (the precedent of D9 of SPEC-LIT 158.10).
///
/// The measurement (2026-10-20, a 10-step run of this section in each build,
/// `vent.pStat` 1e30 so the membrane stays closed): the driver's closing line
/// gave 1.29 s for 10 steps in the f64 build and 1.32 s in the single build,
/// i.e. s_f64 = 0.129 and s_single = 0.132 s per step (the warm-up of the
/// first steps included). The rule: K = 10 floor(420 / (10 max_b s_b)), at
/// most endTime/deltaT = 3500, so K = 10 floor(318.2) = 3180 steps = 0.318 s
/// of the case's 0.35 s. The long run is much quicker per step than those
/// ten: 3180 closed steps took 31.95 s in f64 (0.0100 s per step), so the
/// smoke is about a minute with the mesh.
const SMOKE_ITERS: u64 = 3180;

/// The smoke's `vent.pStat` token value, substituted for the case's own
/// `1e4` ([`P_STAT_TOKEN`]; the substitution is the identity at this value):
/// an input chosen by measurement, not a tolerance (D12 of SPEC-LIT 158.10).
///
/// The measurement (2026-10-20, f64, the closed-history run: K = 3180 steps
/// with `vent.pStat` 1e30, so the membrane never opens and, the outside box
/// being still air, dp is about P1): P1 = 8.911834e-4 Pa at step 1,
/// 1.796159e4 Pa at step 530 (K/6), 3.216173e4 Pa at step 1060 (K/3),
/// 4.840663e4 Pa at step 1590 (K/2) and 1.426455e5 Pa at step 3180 (K). The
/// rule: when max over k <= K/3 of P1_k is at least 1e4 Pa the value is
/// H22's `1e4` (0.1 bar) and nothing is substituted; otherwise it is P1 at
/// K/3 rounded down to two significant digits. The maximum over k <= 1060 is
/// 3.216173e4 Pa >= 1e4 Pa, so H22's value stands. No input was changed after
/// this measurement, whatever the peak row or the budget row shows.
const SMOKE_P_STAT: &str = "1e4";

/// The W4 supervisor's measured cell count of the castellated placeholder
/// mesh (SPEC-LIT 158.10 D1).
const MESH_CELLS: usize = 29_662;

/// The vent's area A_v of `rembeVented.geometry.json`, m2 (the membrane's
/// staircase area is noted against it).
const VENT_AREA: f64 = 0.5;

/// The one token of the shipped case the smoke replaces. It occurs exactly
/// once in the file, and [`smoke_case_text`] refuses a text where it does
/// not.
const P_STAT_TOKEN: &str = "\"pStat\": 1e4";

/// Why the four comparisons of key J.3 are not run.
const NOT_RUN: &str =
    "NOT RUN: H22 publishes no vessel geometry (SPEC-LIT 151.7 item 9); a placeholder is never compared";

/// The rows of steps 5 to 11, in order: the class of each is D except the
/// last, which is A. A failed stage fails every later row by these labels.
const ROWS: [&str; 9] = [
    "Gate DST-REMBE smoke: the automesher builds the placeholder and its polyMesh reads",
    "Gate DST-REMBE smoke: |cells - 29662| of the castellated placeholder",
    "Gate DST-REMBE smoke seal ((158.10j) of SPEC-LIT 158.10): cavity cells not reached from the seed",
    "Gate DST-REMBE smoke seal ((158.10j) of SPEC-LIT 158.10): cells reached outside the shell envelope",
    "Gate DST-REMBE smoke: the run completes",
    "Gate DST-REMBE smoke: missing fields + non-finite values in the final state",
    "Gate DST-REMBE smoke: |membrane opened lines - 1|",
    "Gate DST-REMBE smoke ((158.10i) of SPEC-LIT 158.10): a P1 pressure peak after the opening",
    "Gate DST-REMBE smoke mass budget ((158.10h) of SPEC-LIT 158.10): |residual| against 1e-10",
];

/// The A-class bar of the mass budget, relative. NEVER widened and NEVER
/// switched by build.
const BUDGET_BAR: f64 = 1.0e-10;

/// `<CARGO_MANIFEST_DIR>/../cases/dust/<file>`, the shipped case files.
fn shipped(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../cases/dust").join(file)
}

/// The shipped case text with the [`P_STAT_TOKEN`] replaced by
/// `"pStat": <p_stat>`. Refused by name unless the token occurs exactly
/// once.
fn smoke_case_text(text: &str, p_stat: &str) -> Result<String> {
    let n = text.matches(P_STAT_TOKEN).count();
    if n != 1 {
        return Err(Error::Config(format!(
            "Gate DST-REMBE: the case text carries the token {P_STAT_TOKEN} {n} time(s), not \
             exactly once - the smoke cannot say which vent.pStat to replace (SPEC-LIT 158.10)"
        )));
    }
    Ok(text.replace(P_STAT_TOKEN, &format!("\"pStat\": {p_stat}")))
}

/// Write the smoke's three files into `dir` (which exists): `vessel.stl` =
/// [`vessel_stl`], `cfg.json` = [`automesher_config`] with the case
/// directory `rembeVented.mesh`, and `rembeVented.jsonc` = the shipped case
/// with `vent.pStat` replaced ([`smoke_case_text`]). Returns the copied case
/// path and the geometry read from `rembeVented.geometry.json`.
fn write_smoke_dir(dir: &Path, p_stat: &str) -> Result<(PathBuf, VesselGeometry)> {
    let g = read_vessel_geometry(&shipped("rembeVented.geometry.json"))?;
    let case_src = shipped("rembeVented.jsonc");
    let text = std::fs::read_to_string(&case_src).path(&case_src)?;
    let case_text = smoke_case_text(&text, p_stat)?;
    let stl = dir.join("vessel.stl");
    std::fs::write(&stl, vessel_stl(&g)?).path(&stl)?;
    let cfg = dir.join("cfg.json");
    let cfg_text = serde_json::to_string_pretty(&automesher_config(
        &g,
        "vessel.stl",
        "rembeVented.mesh",
    ))
    .map_err(|e| Error::Config(format!("the automesher config does not serialise: {e}")))?;
    std::fs::write(&cfg, cfg_text).path(&cfg)?;
    let case = dir.join("rembeVented.jsonc");
    std::fs::write(&case, case_text).path(&case)?;
    Ok((case, g))
}

/// What the smoke's `run.log` says: how many `vent: opened after step <n>`
/// lines it carries, the step of the first one, and the `residual` of the
/// last `mass budget:` line ((158.10h) of SPEC-LIT 158.10).
#[derive(Debug, Clone, PartialEq)]
struct SmokeLog {
    opened_lines: usize,
    opened_step: Option<u64>,
    residual: Option<f64>,
}

/// [`SmokeLog`] of a driver log. The tokens are read AFTER their names, so a
/// layout change upstream shows up as an absent value, never as a wrong one.
fn parse_smoke_log(text: &str) -> SmokeLog {
    let mut s = SmokeLog { opened_lines: 0, opened_step: None, residual: None };
    for line in text.lines() {
        let line = line.trim_start();
        if let Some(rest) = line.strip_prefix("vent: opened after step ") {
            s.opened_lines += 1;
            if s.opened_step.is_none() {
                s.opened_step =
                    rest.split_whitespace().next().and_then(|t| t.parse::<u64>().ok());
            }
        } else if line.starts_with("mass budget:") {
            s.residual = line
                .split("residual ")
                .nth(1)
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|t| t.parse::<f64>().ok());
        }
    }
    s
}

/// The driver's closing line `ofgpu-compressible: <N> steps in <X> s` as
/// `(N, X)`; `None` when the log has none.
fn driver_line(text: &str) -> Option<(u64, f64)> {
    const HEAD: &str = "ofgpu-compressible: ";
    let line = text.lines().rev().find(|l| l.starts_with(HEAD) && l.contains(" steps in "))?;
    let mut t = line[HEAD.len()..].split_whitespace();
    let n = t.next()?.parse::<u64>().ok()?;
    if t.next()? != "steps" || t.next()? != "in" {
        return None;
    }
    Some((n, t.next()?.parse::<f64>().ok()?))
}

/// `probes.csv` as its header names and its rows. A row whose column count
/// is not the header's, or whose cell is not a number, is refused.
fn parse_probes_csv(text: &str) -> Result<(Vec<String>, Vec<Vec<f64>>)> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| Error::Config("probes.csv is empty: it has no header".to_string()))?;
    let names: Vec<String> = header.split(',').map(|s| s.trim().to_string()).collect();
    let mut rows = Vec::new();
    for (i, line) in lines.enumerate() {
        let cells: Vec<&str> = line.split(',').collect();
        if cells.len() != names.len() {
            return Err(Error::Config(format!(
                "probes.csv row {}: {} column(s) against the header's {}",
                i + 1,
                cells.len(),
                names.len()
            )));
        }
        let mut row = Vec::with_capacity(cells.len());
        for cell in cells {
            row.push(cell.trim().parse::<f64>().map_err(|e| {
                Error::Config(format!("probes.csv row {}: `{}` is not a number: {e}", i + 1, cell))
            })?);
        }
        rows.push(row);
    }
    Ok((names, rows))
}

/// k* of (158.10i) of SPEC-LIT 158.10: the first index of the series'
/// maximum (a tie keeps the first). `None` for an empty series or one with a
/// non-finite value.
fn first_argmax(p1: &[f64]) -> Option<usize> {
    if p1.is_empty() || p1.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mut k = 0;
    for (j, &v) in p1.iter().enumerate() {
        if v > p1[k] {
            k = j;
        }
    }
    Some(k)
}

/// (158.10i) of SPEC-LIT 158.10: a pressure peak exists when
/// `n_o <= k* < K` and `P1_K < P1_k*`, with `P1_k` the series (k = 0 at t_0
/// to K after the last step), `n_o` the step of the opening and `k*` the
/// first index of the maximum. `Some(k*)` then, otherwise `None` - which
/// includes an empty series, `n_o > K` and a non-finite sample.
fn pressure_peak(p1: &[f64], opened_step: u64) -> Option<usize> {
    let k_star = first_argmax(p1)?;
    let last = p1.len() - 1;
    let n_o = usize::try_from(opened_step).ok()?;
    (n_o <= last && n_o <= k_star && k_star < last && p1[last] < p1[k_star]).then_some(k_star)
}

/// The cell counts of (158.10j) of SPEC-LIT 158.10.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SealSets {
    /// |C|: the cells whose centre is in the cavity.
    cavity: usize,
    /// |S|: the cells `seal_reach` reaches from `min C` with the membrane's
    /// faces closed.
    reached: usize,
    /// |C \ S|: a cavity cell the walk did not reach. An EMPTY cavity counts
    /// as one, so the row cannot pass on no cells at all.
    cavity_unreached: usize,
    /// |S \ E|: a reached cell outside the shell's envelope E.
    reached_outside: usize,
}

/// (158.10j) of SPEC-LIT 158.10: C = cells with `in_cavity(x_i)`, seed =
/// `min C`, `S = seal_reach(hm, membrane, seed)` and E the shell's outer
/// N-gon envelope on the vessel side of the membrane plane:
/// `x0 - t < x.x < plane` and `|(y - axisY, z - axisZ)| < (R + t) cos(pi/N)`.
/// The vessel is sealed except through the membrane when `C ⊆ S` and
/// `S ⊆ E`. `x_i` is the f64 world centre `hm.origin + hm.geom64().c[i]`.
fn seal_sets(hm: &HostMesh, g: &VesselGeometry, plane: f64, membrane: &[usize]) -> SealSets {
    let d = g.dims();
    let geom = hm.geom64();
    let n_gon = g.vessel.segments as f64;
    let envelope = (d.r + d.t) * (std::f64::consts::PI / n_gon).cos();
    let centre = |i: usize| hm.origin + geom.c[i];
    let in_envelope = |i: usize| {
        let x = centre(i);
        let (dy, dz) = (x.y - d.axis_y, x.z - d.axis_z);
        d.x0 - d.t < x.x && x.x < plane && (dy * dy + dz * dz).sqrt() < envelope
    };
    let cavity: Vec<usize> = (0..hm.n_cells).filter(|&i| d.in_cavity(centre(i))).collect();
    let Some(&seed) = cavity.first() else {
        return SealSets { cavity: 0, reached: 0, cavity_unreached: 1, reached_outside: 0 };
    };
    let reached = seal_reach(hm, membrane, seed);
    let in_s: std::collections::HashSet<usize> = reached.iter().copied().collect();
    SealSets {
        cavity: cavity.len(),
        reached: reached.len(),
        cavity_unreached: cavity.iter().filter(|i| !in_s.contains(i)).count(),
        reached_outside: reached.iter().filter(|&&i| !in_envelope(i)).count(),
    }
}

/// The largest numeric time directory above 0 under `root`, the run's final
/// state; `None` when there is none.
fn final_time_dir(root: &Path) -> Option<PathBuf> {
    std::fs::read_dir(root)
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let t = e.file_name().to_str()?.parse::<f64>().ok()?;
            (t > 0.0).then_some((t, e.path()))
        })
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, p)| p)
}

/// The fields the final state must carry, and the three it carries when the
/// turbulence model writes them.
const REQUIRED: [&str; 9] = ["U", "p", "T", "rho", "Mach", "c", "Ut", "Cof", "HRR"];
const OPTIONAL: [&str; 3] = ["k", "omega", "nut"];

/// The step-8 count: every missing required field, every field that exists
/// but does not read at `n` cells, and every non-finite value of those that
/// read. `U` is a vector; the rest are scalars. The names of the defects go
/// into `why`.
fn field_defects(dir: &Path, n: usize, why: &mut Vec<String>) -> usize {
    let mut bad = 0;
    for name in REQUIRED.iter().chain(OPTIONAL.iter()) {
        let path = dir.join(name);
        if !path.exists() {
            if REQUIRED.contains(name) {
                bad += 1;
                why.push(format!("{name}: missing"));
            }
            continue;
        }
        let nonfinite = if *name == "U" {
            read_vector_field(&path, n).map(|f| {
                f.internal
                    .iter()
                    .flat_map(|v| [f64::from(v.x), f64::from(v.y), f64::from(v.z)])
                    .filter(|x| !x.is_finite())
                    .count()
            })
        } else {
            read_scalar_field(&path, n)
                .map(|f| f.internal.iter().filter(|&&x| !f64::from(x).is_finite()).count())
        };
        match nonfinite {
            Ok(0) => {}
            Ok(k) => {
                bad += k;
                why.push(format!("{name}: {k} non-finite value(s)"));
            }
            Err(e) => {
                bad += 1;
                why.push(format!("{name}: does not read at {n} cells: {e}"));
            }
        }
    }
    bad
}

/// The log's last `n` lines, for a run that failed.
fn log_tail(log: &Path, n: usize) -> Vec<String> {
    let text = std::fs::read_to_string(log).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    let skip = lines.len().saturating_sub(n);
    lines[skip..].iter().map(|l| format!("  | {l}")).collect()
}

/// Fail rows `first..` of [`ROWS`], each by its own label and class (D, the
/// last A): a stage that did not complete fails every later row, none is
/// skipped.
fn fail_from(c: &mut Checks, first: usize) {
    for (i, what) in ROWS.iter().enumerate().skip(first) {
        if i == ROWS.len() - 1 {
            c.check_a(what, Scalar::NAN, BUDGET_BAR as Scalar);
        } else {
            c.check_d(what, 1.0, 0.0);
        }
    }
}

/// A class D row `err == 0`. A row that misses is folded into `held` and
/// recorded in `missed` as `MISSED <row>: <value>; cause: <cause>` - the line
/// the gate's registry report prints (the verdict words are the registry's
/// alone, SPEC-LIT §69.2).
fn d_row(
    c: &mut Checks,
    held: &mut bool,
    missed: &mut Vec<String>,
    what: &str,
    err: usize,
    cause: &str,
) {
    c.check_d(what, err as Scalar, 0.0);
    if err != 0 {
        *held = false;
        missed.push(format!("MISSED {what}: value {err}; cause: {cause}"));
    }
}

/// Spawn `exe args` in `dir` with stdout and stderr to `log`, timed; the
/// exit status, or the spawn error text.
fn spawn_logged(
    exe: &Path,
    args: &[String],
    dir: &Path,
    log: &Path,
) -> Result<(std::result::Result<std::process::ExitStatus, String>, f64)> {
    let out = std::fs::File::create(log).path(log)?;
    let err = out.try_clone().path(log)?;
    let clock = Instant::now();
    let status = Command::new(exe)
        .args(args)
        .current_dir(dir)
        .stdout(out)
        .stderr(err)
        .status()
        .map_err(|e| e.to_string());
    Ok((status, clock.elapsed().as_secs_f64()))
}

/// The f64 build's twin of a single-build sibling: `<target>/release/<name>`
/// for `<target>/<dir>/release/<name>` (cargo's `--target-dir target/single`
/// layout); `None` when it is not there.
fn f64_sibling(exe: &Path) -> Option<PathBuf> {
    let twin = exe.parent()?.parent()?.parent()?.join("release").join(exe.file_name()?);
    (twin != exe && twin.exists()).then_some(twin)
}

/// The first `error:` line of a driver log, or a note that it has none.
fn refusal_line(log: &Path) -> String {
    std::fs::read_to_string(log)
        .unwrap_or_default()
        .lines()
        .find_map(|l| l.trim().strip_prefix("error:").map(|r| r.trim().to_string()))
        .unwrap_or_else(|| "no error line in the log".to_string())
}

/// The sibling executable `name` of THIS binary.
fn sibling(name: &str) -> Result<PathBuf> {
    Ok(std::env::current_exe()
        .path("current_exe")?
        .with_file_name(format!("{name}{}", std::env::consts::EXE_SUFFIX)))
}

/// Steps 4 to 11 of [`check_dst_rembe`] in `dir` (which exists): the mesh,
/// the seal, the run and the rows. `held` goes false and `missed` gains a
/// line for every row that misses; a stage that cannot complete fails every
/// later row by its label ([`fail_from`]) and returns.
fn smoke(
    c: &mut Checks,
    mesher: &Path,
    driver: &Path,
    dir: &Path,
    held: &mut bool,
    missed: &mut Vec<String>,
) -> Result<()> {
    // 4. The three files and the castellated mesh.
    let (case_path, g) = write_smoke_dir(dir, SMOKE_P_STAT)?;
    let mesh_log = dir.join("mesh.log");
    let (status, mesh_s) = spawn_logged(
        mesher,
        &["cfg.json".into(), "-stopAfter".into(), "castellate".into()],
        dir,
        &mesh_log,
    )?;
    let mesh_dir = dir.join("rembeVented.mesh");
    let mut mesh_exit = match &status {
        Ok(s) => {
            c.note(&format!("automesher: exit {s}, wall {mesh_s:.1} s (mesh.log)"));
            s.success()
        }
        Err(e) => {
            c.note(&format!("the automesher could not be spawned: {e}"));
            false
        }
    };
    // Under single the sibling automesher refuses this mesh at SPEC-LIT 92.3
    // G2 (closure summed from f32 area vectors, the known A28 of
    // docs/17-review-later). The row records that refusal; the rest of the
    // smoke then runs on the mesh the f64 automesher writes - a polyMesh is
    // host f64 text, and Gate 116-P's single solve does the same on its
    // f64-built mesh.
    let mut single_refusal: Option<String> = None;
    if !mesh_exit && cfg!(feature = "single") {
        if let Some(alt) = f64_sibling(mesher) {
            single_refusal = Some(refusal_line(&mesh_log));
            let _ = std::fs::rename(&mesh_log, dir.join("mesh_single.log"));
            let _ = std::fs::remove_dir_all(&mesh_dir);
            let (st2, s2) = spawn_logged(
                &alt,
                &["cfg.json".into(), "-stopAfter".into(), "castellate".into()],
                dir,
                &mesh_log,
            )?;
            c.note(&format!(
                "the single-build automesher refused the placeholder (mesh_single.log); the f64 \
                 automesher {} writes it instead: {:?}, wall {s2:.1} s",
                alt.display(),
                st2
            ));
            mesh_exit = matches!(&st2, Ok(s) if s.success());
        }
    }

    // 5. The mesh rows.
    let raw = if mesh_exit {
        read_poly_mesh(&mesh_dir)
    } else {
        Err(Error::Mesh("the automesher did not exit 0".into()))
    };
    let raw = match raw {
        Ok(r) => r,
        Err(e) => {
            c.require(ROWS[0], false);
            c.note(&format!("the placeholder mesh is unavailable: {e}"));
            for line in log_tail(&mesh_log, 15) {
                c.note(&line);
            }
            fail_from(c, 1);
            *held = false;
            missed.push(format!("MISSED {}: {e}; cause: see mesh.log", ROWS[0]));
            return Ok(());
        }
    };
    c.require(ROWS[0], single_refusal.is_none());
    if let Some(refusal) = &single_refusal {
        *held = false;
        missed.push(format!(
            "MISSED {}: the single-build automesher refuses the placeholder: {refusal}; cause: \
             SPEC-LIT 92.3 sums f32 area vectors under single (docs/17-review-later A28); the \
             rows below run on the f64 automesher's mesh",
            ROWS[0]
        ));
    }
    let hm = build_host_mesh(&raw)?;
    d_row(
        c,
        held,
        missed,
        ROWS[1],
        hm.n_cells.abs_diff(MESH_CELLS),
        "the automesher's castellated placeholder is not the measured mesh (SPEC-LIT 158.10 D1)",
    );
    c.note(&format!(
        "the castellated placeholder has {} cells, {} faces and {} patches (the measured count is {MESH_CELLS})",
        hm.n_cells,
        raw.faces.len(),
        raw.patches.len()
    ));
    let case = read_case_jsonc(&case_path)?;
    let vent = match case
        .vent
        .as_ref()
        .ok_or_else(|| Error::Config("the case has no vent block".into()))
        .and_then(|v| v.lower())
    {
        Ok(v) => v,
        Err(e) => {
            c.note(&format!("the vent block does not lower: {e}"));
            fail_from(c, 2);
            *held = false;
            missed.push(format!("MISSED {}: {e}; cause: the vent block of the case", ROWS[2]));
            return Ok(());
        }
    };
    let faces = match select_faces(&raw, &vent) {
        Ok(f) => f,
        Err(e) => {
            c.note(&format!("the vent selects no membrane: {e}"));
            fail_from(c, 2);
            *held = false;
            missed.push(format!("MISSED {}: {e}; cause: the vent selects no membrane", ROWS[2]));
            return Ok(());
        }
    };
    let radius = match &vent.select {
        VentSelect::Disc { radius, .. } => *radius,
        VentSelect::Faces(_) => f64::NAN,
    };
    let area: f64 = faces.iter().map(|f| f.area).sum();
    c.note(&format!(
        "the membrane M: {} face(s), total area {area:.6} m2 against A_v = {VENT_AREA} m2 \
         (the staircase hole), vent radius {radius} m",
        faces.len()
    ));

    // 6. The seal rows, (158.10j) of SPEC-LIT 158.10.
    let membrane: Vec<usize> = faces.iter().map(|f| f.face).collect();
    let plane = membrane_plane(&g)?;
    let seal = seal_sets(&hm, &g, plane, &membrane);
    d_row(
        c,
        held,
        missed,
        ROWS[2],
        seal.cavity_unreached,
        "a cavity cell is not connected to the seed by internal faces outside the membrane ((158.10j) of SPEC-LIT 158.10)",
    );
    d_row(
        c,
        held,
        missed,
        ROWS[3],
        seal.reached_outside,
        "the walk leaks outside the shell's envelope: a gap in the castellated wall ((158.10j) of SPEC-LIT 158.10)",
    );
    c.note(&format!(
        "the sealed-region walk: |C| = {} cavity cells, |S| = {} reached cells, membrane plane x = {plane}",
        seal.cavity, seal.reached
    ));

    // 7. The run, timed.
    let run_log = dir.join("run.log");
    let (status, run_s) = spawn_logged(
        driver,
        &[
            "rembeVented.jsonc".into(),
            "-iters".into(),
            SMOKE_ITERS.to_string(),
            "-reportEvery".into(),
            "50".into(),
        ],
        dir,
        &run_log,
    )?;
    c.note(&format!(
        "the run: {SMOKE_ITERS} steps of deltaT = {} s, vent.pStat {SMOKE_P_STAT} Pa, wall {run_s:.1} s",
        case.run.delta_t
    ));
    let run_ok = match &status {
        Ok(s) => {
            c.note(&format!("the driver exits {s}"));
            s.success()
        }
        Err(e) => {
            c.note(&format!("the driver could not be spawned: {e}"));
            false
        }
    };
    c.require(ROWS[4], run_ok);
    let text = std::fs::read_to_string(&run_log).unwrap_or_default();
    if let Some((n, secs)) = driver_line(&text) {
        c.note(&format!(
            "the driver's own line: {n} steps in {secs:.2} s = {:.3} s per step",
            secs / n.max(1) as f64
        ));
    }
    if !run_ok {
        for line in log_tail(&run_log, 15) {
            c.note(&line);
        }
        fail_from(c, 5);
        *held = false;
        missed.push(format!("MISSED {}: the driver failed; cause: see run.log", ROWS[4]));
        return Ok(());
    }

    // 8. Finite.
    let out_root = crate::common::json_case_output_dir(&case_path);
    let mut why = Vec::new();
    let defects = match final_time_dir(&out_root) {
        Some(t) => {
            c.note(&format!("the final state: {}", t.display()));
            field_defects(&t, hm.n_cells, &mut why)
        }
        None => {
            why.push(format!("no final time directory under {}", out_root.display()));
            1
        }
    };
    d_row(c, held, missed, ROWS[5], defects, &format!("the final state holds: {}", why.join("; ")));
    for w in &why {
        c.note(&format!("a defect of the final state: {w}"));
    }

    // 9. The membrane opened exactly once.
    let log = parse_smoke_log(&text);
    d_row(
        c,
        held,
        missed,
        ROWS[6],
        log.opened_lines.abs_diff(1),
        &format!(
            "the membrane opens once (it is irreversible): {} opening line(s) at pStat {SMOKE_P_STAT} Pa",
            log.opened_lines
        ),
    );
    c.note(&format!(
        "vent opening: {} line(s), after step {:?}",
        log.opened_lines, log.opened_step
    ));

    // 10. The pressure peak, (158.10i) of SPEC-LIT 158.10.
    let probes = std::fs::read_to_string(out_root.join("probes.csv"))
        .map_err(|e| Error::Config(format!("probes.csv does not read: {e}")))
        .and_then(|t| parse_probes_csv(&t));
    let (p1, p1_why): (Vec<f64>, String) = match &probes {
        Ok((names, rows)) => match names.iter().position(|n| n == "P1") {
            Some(col) => (rows.iter().map(|r| r[col]).collect(), String::new()),
            None => (Vec::new(), format!("probes.csv has no P1 column: {names:?}")),
        },
        Err(e) => (Vec::new(), e.to_string()),
    };
    let peak = log.opened_step.and_then(|n_o| pressure_peak(&p1, n_o));
    c.require(ROWS[7], peak.is_some());
    let k_star = first_argmax(&p1);
    let at = |k: Option<usize>| {
        k.and_then(|k| p1.get(k)).map_or("n/a".to_string(), |v| format!("{v:.6e}"))
    };
    let last_k = p1.len().checked_sub(1);
    c.note(&format!(
        "P1: n_o = {:?}, k* = {:?}, P1_k* = {} Pa, P1_K = {} Pa, K = {}",
        log.opened_step,
        k_star,
        at(k_star),
        at(last_k),
        p1.len().saturating_sub(1)
    ));
    if peak.is_none() {
        let cause = if !p1_why.is_empty() {
            p1_why
        } else if log.opened_step.is_none() {
            "the membrane never opened in the run, so (158.10i) of SPEC-LIT 158.10 has no n_o"
                .to_string()
        } else {
            "P1 has no maximum between the opening and the last step with a fall after it".to_string()
        };
        *held = false;
        missed.push(format!(
            "MISSED {}: n_o = {:?}, k* = {:?}, P1_k* = {}, P1_K = {}; cause: {cause}",
            ROWS[7],
            log.opened_step,
            k_star,
            at(k_star),
            at(last_k)
        ));
    }

    // 11. The mass budget, (158.10h) of SPEC-LIT 158.10.
    match log.residual {
        Some(r) => {
            c.check_a(ROWS[8], r.abs() as Scalar, BUDGET_BAR as Scalar);
            if !(r.abs() <= BUDGET_BAR) {
                *held = false;
                let cause = if cfg!(feature = "single") {
                    "f32 rho and mdot rounded at every step (SPEC-LIT 158.10 part 2: 1.04e-7 on the tube)"
                } else {
                    "not expected in f64: read run.log"
                };
                missed.push(format!(
                    "MISSED Gate DST-REMBE smoke mass budget (A): residual {r:.6e} against \
                     {BUDGET_BAR:e} relative; cause: {cause}"
                ));
            }
        }
        None => {
            c.check_a(ROWS[8], Scalar::NAN, BUDGET_BAR as Scalar);
            *held = false;
            missed.push(format!(
                "MISSED {}: no value; cause: the run log has no mass budget line",
                ROWS[8]
            ));
        }
    }
    Ok(())
}

/// Gate DST-REMBE. Opened by `validate.rs`'s section under [`GATE`].
pub(crate) fn check_dst_rembe(c: &mut Checks) -> Result<()> {
    // 1. The key. The smoke runs either way; nothing is compared with it.
    match key::load(KEY_ID) {
        Ok(kf) => c.note(&kf.digest_line()),
        Err(why) => c.note(&format!("the answer key is absent: {why}; the smoke runs without it")),
    }

    // 2. The four comparisons of key J.3 are NAMED and NOT RUN.
    for what in [
        "Gate DST-REMBE key J.3 (huang2022-rembe-p-vented, SPEC-LIT 158.7): peak overpressure within +-10 % of 0.60 bar",
        "Gate DST-REMBE key J.3 (huang2022-rembe-p-vented, SPEC-LIT 158.7): time of the peak within +-0.015 s of 0.31 s",
        "Gate DST-REMBE key J.3 (huang2022-rembe-p-vented, SPEC-LIT 158.7): stage-(iii) rise rate 0.20 to 0.28 s within +-15 % of 4.55 bar/s",
        "Gate DST-REMBE key J.3 (huang2022-rembe-p-vented, SPEC-LIT 158.7): rupture within +-0.01 s of 0.10 s",
    ] {
        c.skip(what, NOT_RUN);
    }
    c.note("R-run prerequisite 1: H22's 0.06 bar initial overpressure needs a non-uniform initial p, which DST-10's driver refuses for a dust case (SPEC-LIT 158.9)");
    c.note("R-run prerequisite 2: H22's half-by-symmetry box cannot come from the automesher's stage-0 margin rule (SPEC-LIT 158.10)");
    c.note("R-run prerequisite 3: an outside boundary that is not a wall: a dust case refuses SPEC-LIT 126.9's outlet that p fixedValue lowers to (SPEC-LIT 158.6), so this smoke's box is sealed");

    // 3. The drivers are the siblings of THIS binary.
    let mesher = sibling("ofgpu-automesher")?;
    let driver = sibling("ofgpu-compressible")?;
    if let Some(absent) = [&mesher, &driver].into_iter().find(|p| !p.exists()) {
        let name = absent.file_name().and_then(|n| n.to_str()).unwrap_or("a sibling driver");
        let why = format!("the sibling driver {} does not exist", absent.display());
        c.note(&why);
        c.skip("Gate DST-REMBE: the REDUCED PLACEHOLDER smoke", &format!("{name} is not built"));
        c.report(GateReport {
            verdict: Verdict::Open,
            how: How::Live,
            gate: GATE,
            against: AGAINST,
            headline: format!("{}: build it before this gate", absent.display()),
            detail: vec![why],
            uncertainty: Some(Uncertainty::SingleMesh("no mesh was run: a driver was absent")),
        });
        return Ok(());
    }

    // 4.-11. The scratch directory and the smoke in it.
    let dir = std::env::temp_dir().join(format!("ofgpu_dst_rembe_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).path(&dir)?;
    let mut held = true;
    let mut missed: Vec<String> = Vec::new();
    let r = smoke(c, &mesher, &driver, &dir, &mut held, &mut missed);
    if !missed.is_empty() {
        c.report(GateReport {
            verdict: Verdict::Misses,
            how: How::Live,
            gate: GATE,
            against: AGAINST,
            headline: format!(
                "{} row(s) of the REDUCED PLACEHOLDER smoke leave their bars (never compared with H22)",
                missed.len()
            ),
            detail: missed,
            uncertainty: Some(Uncertainty::SingleMesh(
                "one castellated placeholder mesh of 29,662 cells, a placeholder never compared with data",
            )),
        });
    }

    // 12. The scratch directory goes only when every row of steps 5-11 held.
    if held && r.is_ok() {
        let _ = std::fs::remove_dir_all(&dir);
    } else {
        c.note(&format!("the scratch directory is kept: {}", dir.display()));
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use ofgpu::DVec3;

    /// The shipped case lowers to exactly the inputs the gate runs on: the
    /// polyMesh route, the vent on the membrane plane x = 5.425 of (158.10d)
    /// of SPEC-LIT 158.10 with the probes of the geometry file bit for bit,
    /// the ignition at the cavity centre of (158.10c) of SPEC-LIT 158.10, the
    /// FSC / formula / kOmegaSST blocks, and the pStat token the smoke
    /// replaces occurring once. Host only, no GPU, no spawn.
    #[test]
    fn the_rembe_vented_case_lowers_to_the_gate_inputs() {
        let g = read_vessel_geometry(&shipped("rembeVented.geometry.json")).expect("the geometry");
        let d = g.dims();
        let path = shipped("rembeVented.jsonc");
        let text = std::fs::read_to_string(&path).expect("the case reads");
        let case = read_case_jsonc(&path).expect("the case parses");
        assert_eq!(case.mesh.kind, ofgpu::io::case_json::MeshKind::Polymesh);
        assert_eq!(case.mesh.path.as_deref(), Some("rembeVented.mesh"));
        assert_eq!(text.matches(P_STAT_TOKEN).count(), 1, "the pStat token occurs once");

        let vent = case.vent.as_ref().expect("the vent block").lower().expect("the vent lowers");
        assert_eq!(vent.normal, DVec3::new(1.0, 0.0, 0.0));
        let ofgpu::dust::vent::VentSelect::Disc { centre, radius } = vent.select.clone() else {
            panic!("the vent is a disc");
        };
        let plane = membrane_plane(&g).expect("the membrane plane");
        assert!((centre.x - plane).abs() <= 1.0e-12, "centre.x {} against the plane {plane}", centre.x);
        assert_eq!((centre.y, centre.z), (d.axis_y, d.axis_z));
        assert_eq!(radius, 0.45);
        let h_f = 0.775 / 8.0;
        assert!(d.r_vent + h_f / 2.0 < radius && radius < d.r, "r_v {} R {}", d.r_vent, d.r);
        assert_eq!(vent.p_stat, 1.0e4);
        assert_eq!(vent.probes.len(), g.probes.len());
        for (p, q) in vent.probes.iter().zip(&g.probes) {
            assert_eq!(p.name, q.name);
            assert_eq!(
                [p.point.x.to_bits(), p.point.y.to_bits(), p.point.z.to_bits()],
                [q.point[0].to_bits(), q.point[1].to_bits(), q.point[2].to_bits()],
                "probe {} moved",
                q.name
            );
        }

        let comb = case.combustion.as_ref().expect("the combustion block");
        assert_eq!(comb.model.as_deref(), Some("FSC"));
        assert_eq!((comb.a, comb.pr_t, comb.t_flacc), (Some(0.4), Some(0.28), Some(0.15)));
        let ig = comb.ignition.as_ref().expect("the ignition");
        let want = [(d.x0 + d.x1) / 2.0, d.axis_y, d.axis_z];
        for (got, want) in ig.point.iter().zip(want) {
            assert!((got - want).abs() <= 1.0e-12, "ignition {:?} against {want:?}", ig.point);
        }
        assert!((ig.point[0] - 3.3832529481393823).abs() <= 1.0e-12);
        let dust = case.dust.as_ref().expect("the dust block");
        assert_eq!(dust.thermo.as_deref(), Some("formula"));
        assert_eq!(dust.phi, Some(0.62));
        assert_eq!(case.turbulence.as_ref().expect("turbulence").model, "kOmegaSST");
        assert_eq!(case.initial.p, 0.0);
        assert_eq!(case.initial.t, Some(273.0));
        assert_eq!(case.initial.k, Some(0.8438));
        assert_eq!(case.initial.omega, Some(102.06));
        assert_eq!(case.run.delta_t, 1.0e-4);
        assert_eq!(case.run.end_time, 0.35);

        // write_smoke_dir with another pStat changes that one value only.
        let dir = std::env::temp_dir().join(format!("ofgpu_dst_rembe_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("the test scratch directory");
        let (copied, _) = write_smoke_dir(&dir, "2.5e3").expect("write_smoke_dir");
        let other = read_case_jsonc(&copied).expect("the copy parses");
        let v2 = other.vent.as_ref().expect("the copy's vent").lower().expect("the copy's vent lowers");
        assert_eq!(v2.p_stat, 2500.0);
        assert_eq!((v2.normal, v2.select.clone(), v2.wall, v2.probes.clone()), (vent.normal, vent.select.clone(), vent.wall, vent.probes.clone()));
        for f in ["vessel.stl", "cfg.json"] {
            assert!(dir.join(f).exists(), "{f} is written");
        }
        let _ = std::fs::remove_dir_all(&dir);

        // ...and refuses a text with the token absent or twice.
        assert!(smoke_case_text(&text, "1e3").is_ok());
        let absent = text.replace(P_STAT_TOKEN, "\"pStat\": 7");
        let e = smoke_case_text(&absent, "1e3").unwrap_err().to_string();
        assert!(e.contains("pStat") && e.contains("0 time"), "{e}");
        let twice = format!("{text}\n{P_STAT_TOKEN}\n");
        let e = smoke_case_text(&twice, "1e3").unwrap_err().to_string();
        assert!(e.contains("pStat") && e.contains("2 time"), "{e}");
    }

    /// The parsers read the exact driver layouts: the opening line, the
    /// `mass budget:` line of `src/bin/compressible.rs`, the closing line
    /// and the `probes.csv` header and rows.
    #[test]
    fn the_smoke_log_parses() {
        let text = "vent: 120 membrane face(s), area 0.5 m^2, pStat 1e4 Pa, wall noSlip; closed\n\
            probes: t 1.0e-4 P1=1 P2=2\n\
            vent: opened after step 37 (t = 0.0037 s): dp = 1e4 Pa >= pStat 1e4 Pa (p_in 1 Pa, p_out 0 Pa); the 120 faces are interior faces from step 38 on, SPEC-LIT 158.1\n\
            mass budget: M0 1.0e3 kg, M 9.9e2 kg, boundary outflow 1.0e1 kg, residual 1.234560e-15 ((158.10h) of SPEC-LIT 158.10)\n\
            ofgpu-compressible: 50 steps in 12.50 s\n";
        assert_eq!(
            parse_smoke_log(text),
            SmokeLog { opened_lines: 1, opened_step: Some(37), residual: Some(1.23456e-15) }
        );
        assert_eq!(driver_line(text), Some((50, 12.5)));
        assert_eq!(
            parse_smoke_log("nothing of interest\n"),
            SmokeLog { opened_lines: 0, opened_step: None, residual: None }
        );
        assert_eq!(driver_line("nothing of interest\n"), None);
        let twice = format!("{text}vent: opened after step 38 (t = 0.0038 s)\n");
        assert_eq!(parse_smoke_log(&twice).opened_lines, 2);

        let (names, rows) =
            parse_probes_csv("t,P1,P2\n0.000000000000e0,0.0,0.0\n1.0e-4,5.0e1,4.0e1\n").expect("parses");
        assert_eq!(names, vec!["t", "P1", "P2"]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1], vec![1.0e-4, 50.0, 40.0]);
        assert!(parse_probes_csv("t,P1,P2\n0.0,1.0\n").is_err(), "a short row is refused");
        assert!(parse_probes_csv("t,P1,P2\n0.0,1.0,2.0,3.0\n").is_err(), "a long row is refused");
        assert!(parse_probes_csv("").is_err(), "no header is refused");
    }

    /// (158.10i) of SPEC-LIT 158.10: the maximum must come at or after the
    /// opening, strictly before the last sample, with a fall after it; a tie
    /// keeps the first; a non-finite sample voids the series.
    #[test]
    fn a_peak_needs_a_fall_after_the_opening() {
        assert_eq!(pressure_peak(&[0.0, 1.0, 2.0, 3.0, 2.5, 2.0], 3), Some(3));
        assert_eq!(pressure_peak(&[0.0, 1.0, 2.0, 3.0], 1), None, "monotone");
        assert_eq!(pressure_peak(&[0.0, 3.0, 2.0, 1.0], 2), None, "the maximum precedes the opening");
        assert_eq!(pressure_peak(&[0.0, 1.0, 0.5, 4.0], 1), None, "the maximum is the last sample");
        assert_eq!(pressure_peak(&[0.0, 2.0, 2.0, 1.0], 1), Some(1), "a tie keeps the first");
        assert_eq!(pressure_peak(&[0.0, f64::NAN, 2.0, 1.0], 1), None);
        assert_eq!(pressure_peak(&[0.0, 1.0, 2.0, f64::NAN], 1), None);
        assert_eq!(pressure_peak(&[], 0), None);
        assert_eq!(pressure_peak(&[0.0, 1.0, 0.5], 9), None, "the opening is after the last step");
    }
}
