// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The `-json` writer of `ofgpu-validate`: one document per run, carrying the
//! gate registry of SPEC-LIT §69, the tally, the machine, the commit and the
//! wall time. The per-check rows are a later unit's; this document is the
//! header they will hang from.
//!
//! Written from: `AIP ontology/DOMAIN-MODEL.md` (the `Gate`, `GateVerdict` and
//! `Check` object types and their primary keys) and
//! `docs/12-ontology-and-ai-native-chat.md` §C, stream R1. The civil-date
//! arithmetic of [`iso8601_utc`] is Howard Hinnant's `civil_from_days`,
//! "chrono-Compatible Low-Level Date Algorithms"
//! (howardhinnant.github.io/date_algorithms.html), released by its author into
//! the public domain. No GPL-licensed source was consulted.
//!
//! **The verdict word is never spelled here.** [`Verdict::word`] is the only
//! place in this binary either word exists (SPEC-LIT §69.2), and this module's
//! `Serialize` impl calls it, so the document and the screen cannot disagree.

use serde::ser::{SerializeMap, SerializeStruct};

/// The machine a run ran on. `hostname` and `device`/`computeCapability` are
/// `null` when the environment does not name them - never a fiction.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Machine {
    hostname: Option<String>,
    os: &'static str,
    arch: &'static str,
    device: Option<String>,
    compute_capability: Option<String>,
    precision: &'static str,
}

/// The tally `Checks` already keeps, read straight off its four counters.
/// Nothing here counts anything a second time.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Totals {
    total: usize,
    failures: usize,
    skipped: usize,
    replayed: usize,
}

/// `ofgpu::vv::Triplet` on the wire: the same eleven fields, widened to `f64`
/// whatever `Scalar` the build chose - what `vv`'s own `w()` does - and the
/// behaviour spelled as its fixed word.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TripletJson {
    r21: f64,
    r32: f64,
    eps21: f64,
    eps32: f64,
    ratio: f64,
    behaviour: &'static str,
    p: Option<f64>,
    phi_ext: Option<f64>,
    e_a: f64,
    gci_fine: Option<f64>,
    iterations: usize,
}

/// `ofgpu::vv::GridStudy` on the wire: all twelve fields, unrounded. A
/// non-finite number reaches the document as `null` - `serde_json`'s own
/// answer - so an importer reads it as unknown, never as zero.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct StudyJson {
    n_levels: usize,
    behaviour: &'static str,
    estimator: &'static str,
    weighted: bool,
    p: Option<f64>,
    phi_ext: f64,
    eps_fine: f64,
    sigma: f64,
    delta_phi: f64,
    fs: f64,
    u_fine: f64,
    triplet: TripletJson,
}

/// The fixed wire word of a [`ofgpu::vv::Behaviour`]. The crate's own
/// `words` is private to `ofgpu` and spells two of them with a space; the
/// document spells all four with an underscore.
pub(crate) fn behaviour_word(b: ofgpu::vv::Behaviour) -> &'static str {
    match b {
        ofgpu::vv::Behaviour::Monotone => "monotone",
        ofgpu::vv::Behaviour::MonotoneDiverging => "monotone_diverging",
        ofgpu::vv::Behaviour::Oscillatory => "oscillatory",
        ofgpu::vv::Behaviour::OscillatoryDiverging => "oscillatory_diverging",
    }
}

/// The fixed wire word of an [`ofgpu::vv::Estimator`], private to `ofgpu`
/// for the same reason.
pub(crate) fn estimator_word(e: ofgpu::vv::Estimator) -> &'static str {
    match e {
        ofgpu::vv::Estimator::PowerSeries => "power_series",
        ofgpu::vv::Estimator::First => "first",
        ofgpu::vv::Estimator::Second => "second",
        ofgpu::vv::Estimator::FirstSecond => "first_second",
    }
}

/// `vv`'s own widening, mirrored here: `Scalar` to the wire's `f64`,
/// whatever the build chose. Under the default build the conversion is an
/// identity and clippy calls it useless - it is not one under
/// `--features single`, where it is `f32`'s promotion.
#[allow(clippy::useless_conversion)]
fn w(x: ofgpu::Scalar) -> f64 {
    f64::from(x)
}

/// The mirror of one [`ofgpu::vv::GridStudy`]: the same numbers, widened,
/// nothing rounded and nothing mapped onto a friendlier zero.
pub(crate) fn study_json(s: &ofgpu::vv::GridStudy) -> StudyJson {
    StudyJson {
        n_levels: s.n_levels,
        behaviour: behaviour_word(s.behaviour),
        estimator: estimator_word(s.estimator),
        weighted: s.weighted,
        p: s.p.map(w),
        phi_ext: w(s.phi_ext),
        eps_fine: w(s.eps_fine),
        sigma: w(s.sigma),
        delta_phi: w(s.delta_phi),
        fs: w(s.fs),
        u_fine: w(s.u_fine),
        triplet: triplet_json(&s.triplet),
    }
}

fn triplet_json(t: &ofgpu::vv::Triplet) -> TripletJson {
    TripletJson {
        r21: w(t.r21),
        r32: w(t.r32),
        eps21: w(t.eps21),
        eps32: w(t.eps32),
        ratio: w(t.ratio),
        behaviour: behaviour_word(t.behaviour),
        p: t.p.map(w),
        phi_ext: t.phi_ext.map(w),
        e_a: w(t.e_a),
        gci_fine: t.gci_fine.map(w),
        iterations: t.iterations,
    }
}

/// The document itself: fifteen keys, always all of them, in this order.
/// Absence is spelled `null`, never an omitted key.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RunDoc<'a> {
    schema: &'static str,
    run_id: &'a str,
    started_at: String,
    ended_at: String,
    wall_seconds: f64,
    exit_code: u8,
    aborted: Option<String>,
    argv: &'a [String],
    cwd: Option<String>,
    git_sha: Option<String>,
    git_dirty: Option<bool>,
    machine: Machine,
    totals: Totals,
    gates: &'a [super::GateReport],
    rows: &'a [Row],
}

// The four crate-root types cannot carry `#[derive(serde::Serialize)]`: a
// derive on the verdict enum would emit the VARIANT name, and the wire must
// carry the same word `Verdict::word` shouts at the screen (SPEC-LIT §69.2).
// So all four impls are hand-written here and `validate.rs` imports no serde
// at all - the file and the screen take their word from the same one line.

impl serde::Serialize for super::Verdict {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.word())
    }
}

impl serde::Serialize for super::How {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(match self {
            super::How::Live => "Live",
            super::How::Replayed => "Replayed",
        })
    }
}

impl serde::Serialize for super::Uncertainty {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(2))?;
        match self {
            super::Uncertainty::SingleMesh(reason) => {
                map.serialize_entry("kind", "single_mesh")?;
                map.serialize_entry("reason", reason)?;
            }
            super::Uncertainty::Study(study) => {
                map.serialize_entry("kind", "study")?;
                map.serialize_entry("study", &study_json(study))?;
            }
        }
        map.end()
    }
}

impl serde::Serialize for super::GateReport {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut st = s.serialize_struct("GateReport", 7)?;
        st.serialize_field("gate", &self.gate)?;
        st.serialize_field("verdict", &self.verdict)?;
        st.serialize_field("how", &self.how)?;
        st.serialize_field("against", &self.against)?;
        st.serialize_field("headline", &self.headline)?;
        st.serialize_field("detail", &self.detail)?;
        st.serialize_field("uncertainty", &self.uncertainty)?;
        st.end()
    }
}

/// Milliseconds since the Unix epoch, signed - a clock behind 1970 goes
/// negative instead of wearing a huge positive lie.
pub(crate) fn epoch_ms(t: std::time::SystemTime) -> i64 {
    match t.duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => d.as_millis() as i64,
        Err(e) => -(e.duration().as_millis() as i64),
    }
}

/// The civil date of a Unix day number, proleptic Gregorian. Transcribed
/// from Howard Hinnant's `civil_from_days` ("chrono-Compatible Low-Level
/// Date Algorithms", howardhinnant.github.io/date_algorithms.html), which
/// its author released into the public domain. Exact for every day Unix time
/// can name; Unix time ignores leap seconds and so does this.
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// ISO-8601 UTC with milliseconds - `2026-09-15T01:12:03.114Z` - the one
/// time encoding this document uses.
pub(crate) fn iso8601_utc(epoch_ms: i64) -> String {
    let days = epoch_ms.div_euclid(86_400_000);
    let ms = epoch_ms.rem_euclid(86_400_000);
    let (y, m, d) = civil_from_days(days);
    let hh = ms / 3_600_000;
    let mm = (ms % 3_600_000) / 60_000;
    let ss = (ms % 60_000) / 1000;
    let mmm = ms % 1000;
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}.{mmm:03}Z")
}

/// The compact stamp of the default `runId` - `20260915T011203Z` - the same
/// instant as `startedAt` with the milliseconds dropped.
pub(crate) fn compact_stamp(epoch_ms: i64) -> String {
    let days = epoch_ms.div_euclid(86_400_000);
    let ms = epoch_ms.rem_euclid(86_400_000);
    let (y, m, d) = civil_from_days(days);
    let hh = ms / 3_600_000;
    let mm = (ms % 3_600_000) / 60_000;
    let ss = (ms % 60_000) / 1000;
    format!("{y:04}{m:02}{d:02}T{hh:02}{mm:02}{ss:02}Z")
}

/// The `-json <path>` flag: the path of the document this run will write, or
/// `None` when the run was asked for the screen only. Given twice, the last
/// wins; given without a value, the error is `next_arg`'s named one.
pub(crate) fn json_path(argv: &[String]) -> ofgpu::Result<Option<std::path::PathBuf>> {
    let mut out = None;
    let mut i = 0;
    while i < argv.len() {
        if argv[i] == "-json" {
            out = Some(std::path::PathBuf::from(super::common::next_arg(argv, &mut i)?));
        } else {
            i += 1;
        }
    }
    Ok(out)
}

/// The `-run-id <id>` flag, else `v_<compact startedAt>`: the ontology makes
/// `gate + runId` a verdict's key, so the document always carries one.
pub(crate) fn run_id(argv: &[String], started_ms: i64) -> ofgpu::Result<String> {
    let mut i = 0;
    while i < argv.len() {
        if argv[i] == "-run-id" {
            return super::common::next_arg(argv, &mut i);
        }
        i += 1;
    }
    Ok(format!("v_{}", compact_stamp(started_ms)))
}

/// The commit this run ran from, probed NOW, in the directory the process
/// carries. Any failure - `git` absent, non-zero status, non-UTF-8 output,
/// an unborn branch - yields `(None, None)`: a record that cannot name a
/// commit names nothing rather than a guess.
pub(crate) fn git_probe(dir: &std::path::Path) -> (Option<String>, Option<bool>) {
    let ask = |args: &[&str]| -> Option<String> {
        let out = std::process::Command::new("git")
            .current_dir(dir)
            .args(args)
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        String::from_utf8(out.stdout).ok()
    };
    let sha = ask(&["rev-parse", "HEAD"]).map(|s| s.trim().to_string());
    let dirty = ask(&["status", "--porcelain"]).map(|s| !s.trim().is_empty());
    match (sha, dirty) {
        (Some(sha), Some(dirty)) if !sha.is_empty() => (Some(sha), Some(dirty)),
        _ => (None, None),
    }
}

/// The machine block, from the environment and from the device `run` named
/// in its banner.
pub(crate) fn machine(device: Option<&(String, String)>) -> Machine {
    Machine {
        hostname: std::env::var("COMPUTERNAME")
            .or_else(|_| std::env::var("HOSTNAME"))
            .ok(),
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        device: device.map(|(name, _)| name.clone()),
        compute_capability: device.map(|(_, cc)| cc.clone()),
        precision: super::common::precision_name(),
    }
}

/// Whether a row is a comparison or a check that was not attempted. It is
/// the one key that tells a NaN error apart from a skip, both of which
/// reach the importer as `"err": null`.
#[derive(serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum RowKind {
    Check,
    Skip,
}

/// One row of the per-check table: what was compared, against what, whether
/// it held, and the gate whose scope it was taken under. Ten keys, always
/// all ten, absence spelled `null` - the column set an importer keyed on
/// `runId + seq` can rely on (`AIP ontology/DOMAIN-MODEL.md`'s check table).
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Row {
    seq: usize,
    kind: RowKind,
    what: String,
    err: Option<f64>,
    tol: Option<f64>,
    ok: Option<bool>,
    replayed: Option<bool>,
    why: Option<String>,
    /// `"nan"`, `"inf"`, `"-inf"`, or `None` when the error was finite.
    ///
    /// `serde_json` writes a non-finite `f64` as `null`, so a NaN error and a
    /// skip both reach the importer with `"err": null`. `kind` tells them
    /// apart; this says which non-finite the number was.
    non_finite: Option<&'static str>,
    /// The gate whose scope was open when this row was taken - its name, which
    /// is its primary key in `AIP ontology/DOMAIN-MODEL.md`'s check table. `None` is not
    /// a defect: it is the honest statement that this check is not under a gate
    /// the registry names.
    gate: Option<&'static str>,
}

/// `None` when `x` is finite; otherwise the word for which non-finite it is.
/// Called by [`check_row`] from the `f64` it already has, so `validate.rs`
/// never learns what the wire format calls a NaN.
pub(crate) fn non_finite_word(x: f64) -> Option<&'static str> {
    if x.is_finite() {
        None
    } else if x.is_nan() {
        Some("nan")
    } else if x > 0.0 {
        Some("inf")
    } else {
        Some("-inf")
    }
}

/// One comparison as a row. `err` is `None` when the number was not finite -
/// `serde_json` refuses a non-finite `f64` outright, and `non_finite` is
/// where the fact goes instead, so `validate.rs` never learns what the wire
/// format calls a NaN.
pub(crate) fn check_row(
    seq: usize,
    what: &str,
    err: f64,
    tol: f64,
    ok: bool,
    replayed: bool,
    gate: Option<&'static str>,
) -> Row {
    Row {
        seq,
        kind: RowKind::Check,
        what: what.to_string(),
        err: if err.is_finite() { Some(err) } else { None },
        tol: if tol.is_finite() { Some(tol) } else { None },
        ok: Some(ok),
        replayed: Some(replayed),
        why: None,
        non_finite: non_finite_word(err),
        gate,
    }
}

/// A check that was not attempted, as a row. It has no error, so `err`,
/// `tol`, `ok` and `replayed` are all `null`, and `why` carries the reason.
pub(crate) fn skip_row(seq: usize, what: &str, why: &str, gate: Option<&'static str>) -> Row {
    Row {
        seq,
        kind: RowKind::Skip,
        what: what.to_string(),
        err: None,
        tol: None,
        ok: None,
        replayed: None,
        why: Some(why.to_string()),
        non_finite: None,
        gate,
    }
}

/// The document of one run: the tally read off `Checks`, every gate it
/// registered, the machine, the commit, the clock. The commit is probed here,
/// once, because this is the one moment the process knows it is recording.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_document<'a>(
    c: &'a super::Checks,
    run_id: &'a str,
    argv: &'a [String],
    started_ms: i64,
    ended_ms: i64,
    wall_seconds: f64,
    exit_code: u8,
    aborted: Option<String>,
) -> RunDoc<'a> {
    let (git_sha, git_dirty) = git_probe(std::path::Path::new("."));
    RunDoc {
        schema: "ofgpu-validate/1",
        run_id,
        started_at: iso8601_utc(started_ms),
        ended_at: iso8601_utc(ended_ms),
        wall_seconds,
        exit_code,
        aborted,
        argv,
        cwd: std::env::current_dir().ok().map(|p| p.display().to_string()),
        git_sha,
        git_dirty,
        machine: machine(c.device.as_ref()),
        totals: Totals {
            total: c.total,
            failures: c.failures,
            skipped: c.skipped,
            replayed: c.replayed,
        },
        gates: c.gates.as_slice(),
        rows: c.rows.as_slice(),
    }
}

/// Compact JSON plus one trailing newline - the shape `run.json` uses, and
/// the shape an importer expects. The parent directory is made if absent;
/// every failure names the path it died on.
pub(crate) fn write_document(path: &std::path::Path, doc: &RunDoc) -> ofgpu::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ofgpu::Error::Config(format!("cannot create {}: {e}", parent.display())))?;
        }
    }
    let mut text = serde_json::to_string(doc)
        .map_err(|e| ofgpu::Error::Config(format!("cannot serialise the run document: {e}")))?;
    text.push('\n');
    std::fs::write(path, text)
        .map_err(|e| ofgpu::Error::Config(format!("cannot write {}: {e}", path.display())))?;
    Ok(())
}

// Inside this module `super` is `json`, NOT the crate root - so everything
// the fixtures build out of `validate.rs` is named `crate::X`, and
// `ofgpu::vv`'s numerics are written out in full. Nothing here touches a
// GPU, a network or a port.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Checks, GateReport, How, Scalar, Uncertainty, Verdict, ONE_MESH_AS_RUN};
    use serde_json::json;

    fn report(verdict: Verdict, uncertainty: Option<Uncertainty>) -> GateReport {
        GateReport {
            verdict,
            how: How::Live,
            gate: "SPEC-LIT §32.4 verdict 2 (Reynolds analogy), wall-function leg",
            against: "Dittus-Boelter, Univ. Calif. Publ. Eng. 2 (1930) 443",
            headline: "Nu is 31 % low at Re = 5e4".to_string(),
            detail: vec![
                "first detail line".to_string(),
                "second detail line".to_string(),
            ],
            uncertainty,
        }
    }

    /// The one sequence §94.3's machinery is exact on: `phi = 1 + 0.1 h^2`
    /// over `h = 1, 2, 4`, whose observed order is 2.
    fn study() -> ofgpu::vv::GridStudy {
        ofgpu::vv::grid_study(&[
            ofgpu::vv::Level { h: 1.0, value: 1.1 },
            ofgpu::vv::Level { h: 2.0, value: 1.4 },
            ofgpu::vv::Level { h: 4.0, value: 2.6 },
        ])
        .unwrap()
    }

    /// The document's verdict word is the one [`Verdict::word`] shouts; the
    /// derive's variant name never reaches the wire.
    #[test]
    fn a_verdict_word_in_the_document_comes_from_verdict_word() {
        let text = serde_json::to_string(&report(Verdict::Misses, None)).unwrap();
        assert!(
            text.contains(&format!("\"verdict\":\"{}\"", Verdict::Misses.word())),
            "the document does not carry Verdict::word's own word: {text}"
        );
        assert!(!text.contains("\"Misses\""), "a variant name reached the wire: {text}");
    }

    /// The claim `only_the_registry_spells_a_verdict_word` makes about
    /// `validate.rs`, extended to this file: no non-comment line of the
    /// writer carries a verdict word, because test 1's word came from
    /// `Verdict::word` and nowhere else.
    #[test]
    fn the_json_writer_spells_no_verdict_word() {
        let source = include_str!("mod.rs");
        let hits: Vec<(usize, &str)> = source
            .lines()
            .enumerate()
            .filter(|(_, l)| !l.trim_start().starts_with("//"))
            .filter(|(_, l)| crate::has_verdict_word(l))
            .map(|(i, l)| (i + 1, l.trim()))
            .collect();
        assert!(
            hits.is_empty(),
            "the JSON writer spells a verdict word on {} line(s):\n  {}",
            hits.len(),
            hits.iter()
                .map(|(n, l)| format!("{n}: {l}"))
                .collect::<Vec<_>>()
                .join("\n  ")
        );
    }

    #[test]
    fn a_gate_report_carries_all_seven_fields() {
        let v = serde_json::to_value(&report(
            Verdict::Open,
            Some(Uncertainty::SingleMesh(ONE_MESH_AS_RUN)),
        ))
        .unwrap();
        let obj = v.as_object().unwrap();
        assert_eq!(obj.len(), 7);
        assert_eq!(
            obj["gate"],
            "SPEC-LIT §32.4 verdict 2 (Reynolds analogy), wall-function leg"
        );
        assert_eq!(obj["against"], "Dittus-Boelter, Univ. Calif. Publ. Eng. 2 (1930) 443");
        assert_eq!(obj["headline"], "Nu is 31 % low at Re = 5e4");
        assert_eq!(obj["detail"].as_array().unwrap().len(), 2);
        assert_eq!(obj["how"], "Live");
        assert_eq!(obj["uncertainty"]["kind"], "single_mesh");
        assert_eq!(obj["uncertainty"]["reason"], ONE_MESH_AS_RUN);
    }

    /// Absence is `null`, never an omitted key.
    #[test]
    fn an_undeclared_mesh_study_is_null_not_absent() {
        let v = serde_json::to_value(&report(Verdict::Open, None)).unwrap();
        let obj = v.as_object().unwrap();
        assert!(obj.contains_key("uncertainty"), "the key must exist: {obj:?}");
        assert_eq!(obj["uncertainty"], serde_json::Value::Null);
    }

    /// Every one of the study's twelve fields and the triplet's eleven
    /// survives, unrounded: `p` by a 1e-6 tolerance (it is a fixed-point
    /// result), every other number by exact `f64` equality.
    #[test]
    fn a_grid_study_uncertainty_carries_every_number() {
        let s = study();
        let u_fine = f64::from(s.u_fine);
        let v = serde_json::to_value(&Uncertainty::Study(s)).unwrap();
        assert_eq!(v["kind"], "study");
        assert_eq!(v["study"]["nLevels"], 3);
        assert_eq!(v["study"]["behaviour"], "monotone");
        let p = v["study"]["p"].as_f64().unwrap();
        assert!((p - 2.0).abs() < 1e-6, "p = {p}, not the known 2 of 1 + 0.1 h^2");
        assert_eq!(v["study"]["uFine"], u_fine);
        assert_eq!(v["study"]["triplet"].as_object().unwrap().len(), 11);
    }

    /// A study's arithmetic can go non-finite; `serde_json` answers `null`
    /// and the document still parses. Pinned so no one ever "helps" by
    /// mapping it to a zero an importer would believe.
    #[test]
    fn a_non_finite_study_number_is_null_not_a_zero() {
        let mut s = study();
        s.sigma = Scalar::NAN;
        s.triplet.gci_fine = Some(Scalar::INFINITY);
        let text = serde_json::to_string(&Uncertainty::Study(s)).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["study"]["sigma"], serde_json::Value::Null);
        assert_eq!(v["study"]["triplet"]["gciFine"], serde_json::Value::Null);
        assert_ne!(v["study"]["sigma"], json!(0.0));
        assert_ne!(v["study"]["triplet"]["gciFine"], json!(0.0));
    }

    /// `totals` is `Checks`' own tally, read off the four counters - the
    /// writer does no counting of its own.
    #[test]
    fn the_totals_are_the_tally_and_nothing_else() {
        let mut c = Checks::new();
        c.check("the ok check", 0.0, 1.0);
        c.replaying(|c| c.check("the replayed check", 0.0, 1.0));
        c.check("the failing check", 1.0, 0.0);
        c.skip("the skipped check", "not attempted on this machine");
        assert_eq!((c.total, c.failures, c.skipped, c.replayed), (3, 1, 1, 1));
        let argv = vec!["ofgpu-validate".to_string()];
        let doc = build_document(&c, "v_test", &argv, 0, 1, 0.001, 0, None);
        let v = serde_json::to_value(&doc).unwrap();
        assert_eq!(
            v["totals"],
            json!({"total": 3, "failures": 1, "skipped": 1, "replayed": 1})
        );
    }

    /// The fifteen keys of the contract, and only those; `machine`'s six
    /// keys are what prove the `rename_all = "camelCase"` attribute is on
    /// the derived structs.
    #[test]
    fn the_document_names_every_field_of_the_contract() {
        let c = Checks::new();
        let argv = vec![
            "ofgpu-validate".to_string(),
            "-json".to_string(),
            "gates.json".to_string(),
        ];
        let doc = build_document(&c, "v_test", &argv, 0, 1, 0.001, 0, None);
        let v = serde_json::to_value(&doc).unwrap();
        let obj = v.as_object().unwrap();
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "aborted", "argv", "cwd", "endedAt", "exitCode", "gates", "gitDirty",
                "gitSha", "machine", "rows", "runId", "schema", "startedAt", "totals",
                "wallSeconds",
            ]
        );
        assert_eq!(obj.len(), 15);
        assert_eq!(obj["schema"], "ofgpu-validate/1");
        assert_eq!(obj["aborted"], serde_json::Value::Null);
        let machine = obj["machine"].as_object().unwrap();
        assert_eq!(machine.len(), 6);
        assert!(machine.contains_key("computeCapability"));
    }

    /// Five known instants, exactly - the 2000-02-29 row is the leap day a
    /// hand-rolled civil-date routine gets wrong.
    #[test]
    fn iso8601_utc_matches_known_epochs() {
        assert_eq!(iso8601_utc(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(iso8601_utc(951_782_400_000), "2000-02-29T00:00:00.000Z");
        assert_eq!(iso8601_utc(1_000_000_000_000), "2001-09-09T01:46:40.000Z");
        assert_eq!(iso8601_utc(1_234_567_890_123), "2009-02-13T23:31:30.123Z");
        assert_eq!(iso8601_utc(1_757_900_000_123), "2025-09-15T01:33:20.123Z");
    }

    #[test]
    fn the_default_run_id_is_the_stamp() {
        let plain = ["ofgpu-validate".to_string()];
        assert_eq!(
            run_id(&plain, 1_757_900_000_123).unwrap(),
            "v_20250915T013320Z"
        );
        let named = [
            "ofgpu-validate".to_string(),
            "-run-id".to_string(),
            "r_42".to_string(),
        ];
        assert_eq!(run_id(&named, 0).unwrap(), "r_42");
    }

    #[test]
    fn the_json_flag_takes_the_next_argument() {
        let given = ["x".to_string(), "-json".to_string(), "out/g.json".to_string()];
        assert_eq!(
            json_path(&given).unwrap(),
            Some(std::path::PathBuf::from("out/g.json"))
        );
        let absent = ["x".to_string()];
        assert_eq!(json_path(&absent).unwrap(), None);
        let bare = ["x".to_string(), "-json".to_string()];
        let err = json_path(&bare).unwrap_err();
        assert!(err.to_string().contains("-json"), "the error names the flag: {err}");
    }

    /// Shape only: the value is the machine's, and asserting one would be
    /// inventing it.
    #[test]
    fn a_git_sha_is_forty_hex_digits_or_absent() {
        let (sha, dirty) = git_probe(std::path::Path::new("."));
        if let Some(sha) = &sha {
            assert_eq!(sha.len(), 40, "the full 40 hex, never a short form: {sha}");
            assert!(sha.bytes().all(|b| b.is_ascii_hexdigit()), "not hex: {sha}");
        }
        assert_eq!(dirty.is_some(), sha.is_some());
    }

    #[test]
    fn the_document_is_written_compact_and_parses() {
        let dir = std::env::temp_dir().join(format!("ofgpu-json-test-{}", std::process::id()));
        let path = dir.join("nested").join("run.json");
        let c = Checks::new();
        let argv = vec!["ofgpu-validate".to_string()];
        let doc = build_document(&c, "v_test", &argv, 0, 1, 0.001, 0, None);
        write_document(&path, &doc).unwrap();
        assert!(path.exists());
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.matches('\n').count(), 1, "compact, plus the trailing newline");
        assert!(text.ends_with('\n'));
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["schema"], "ofgpu-validate/1");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A check taken with no gate scope open carries `gate: null` - the
    /// honest statement that it is not under a gate the registry names.
    #[test]
    fn a_row_outside_a_gate_has_a_null_parent() {
        let mut c = Checks::new();
        c.check("x", 0.0, 1.0);
        assert_eq!(c.rows[0].gate, None);
        let v = serde_json::to_value(&c.rows[0]).unwrap();
        assert_eq!(v["gate"], serde_json::Value::Null);
    }

    /// A check taken inside a gate's scope names that gate, and the scope
    /// ends where `leave_gate` says it ends.
    #[test]
    fn a_row_inside_a_gate_carries_its_name() {
        let mut c = Checks::new();
        c.enter_gate("S99.1 Gate A");
        c.check("x", 0.0, 1.0);
        c.leave_gate();
        c.check("y", 0.0, 1.0);
        assert_eq!(c.rows[0].gate, Some("S99.1 Gate A"));
        assert_eq!(c.rows[1].gate, None);
        assert_eq!((c.rows[0].seq, c.rows[1].seq), (1, 2));
    }

    /// One gate, never a stack: entering a second gate REPLACES the first.
    /// This is the shape `check_resolved_leg_gate_verdict_replay` relies on.
    #[test]
    fn entering_a_second_gate_replaces_the_first() {
        let mut c = Checks::new();
        c.enter_gate("S99.1 Gate A");
        c.check("x", 0.0, 1.0);
        c.enter_gate("S99.2 Gate B");
        c.check("y", 0.0, 1.0);
        c.leave_gate();
        c.check("z", 0.0, 1.0);
        assert_eq!(c.rows[0].gate, Some("S99.1 Gate A"));
        assert_eq!(c.rows[1].gate, Some("S99.2 Gate B"));
        assert_eq!(c.rows[2].gate, None);
    }

    /// A skip is a row in the SAME sequence as the checks, and it carries
    /// the gate it was skipped under.
    #[test]
    fn a_skip_row_carries_its_gate_too() {
        let mut c = Checks::new();
        c.enter_gate("S99.1 Gate A");
        c.check("x", 0.0, 1.0);
        c.skip("cuFFT Poisson", "no cuFFT on this device");
        c.check("y", 0.0, 1.0);
        c.leave_gate();
        let v = serde_json::to_value(&c.rows[1]).unwrap();
        assert_eq!(v["kind"], "skip");
        assert_eq!(v["err"], serde_json::Value::Null);
        assert_eq!(v["tol"], serde_json::Value::Null);
        assert_eq!(v["ok"], serde_json::Value::Null);
        assert_eq!(v["replayed"], serde_json::Value::Null);
        assert_eq!(v["why"], "no cuFFT on this device");
        assert_eq!(v["gate"], "S99.1 Gate A");
        assert_eq!((c.rows[0].seq, c.rows[1].seq, c.rows[2].seq), (1, 2, 3));
        assert_eq!((c.skipped, c.total), (1, 2));
    }

    /// A non-finite error serialises `err: null` AND says which non-finite
    /// it was; the check itself still fails exactly as it always did.
    #[test]
    fn a_non_finite_error_says_which_one_it_was() {
        let mut c = Checks::new();
        c.check("x", Scalar::NAN, 1.0);
        let v = serde_json::to_value(&c.rows[0]).unwrap();
        assert_eq!(v["err"], serde_json::Value::Null);
        assert_eq!(v["nonFinite"], "nan");
        assert_eq!(v["ok"], false);
        assert_eq!(c.failures, 1);
        assert_eq!(non_finite_word(f64::INFINITY), Some("inf"));
        assert_eq!(non_finite_word(f64::NEG_INFINITY), Some("-inf"));
        assert_eq!(non_finite_word(0.0), None);
    }

    /// The column set an importer keyed on `runId + seq` can rely on: the
    /// same ten keys on every row, none optional.
    #[test]
    fn every_row_has_the_same_ten_keys() {
        let mut c = Checks::new();
        c.enter_gate("S99.1 Gate A");
        c.check("gated", 0.0, 1.0);
        c.leave_gate();
        c.check("ungated", 0.0, 1.0);
        c.skip("cuFFT Poisson", "no cuFFT on this device");
        let want = [
            "err", "gate", "kind", "nonFinite", "ok", "replayed", "seq", "tol", "what", "why",
        ];
        for row in &c.rows {
            let v = serde_json::to_value(row).unwrap();
            let obj = v.as_object().unwrap();
            let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
            keys.sort_unstable();
            assert_eq!(keys, want, "{keys:?}");
            assert_eq!(obj.len(), 10);
        }
    }

    /// The document R1 writes carries the gate on its rows - built in memory
    /// here, with no GPU, no process and no file.
    #[test]
    fn the_document_rows_carry_their_gate() {
        let mut c = Checks::new();
        c.enter_gate("S99.1 Gate A");
        c.check("gated", 0.0, 1.0);
        c.skip("cuFFT Poisson", "no cuFFT on this device");
        c.leave_gate();
        c.check("ungated", 0.0, 1.0);
        let argv = vec!["ofgpu-validate".to_string()];
        let doc = build_document(&c, "v_test", &argv, 0, 1, 0.001, 0, None);
        let v = serde_json::to_value(&doc).unwrap();
        assert_eq!(v["rows"][0]["gate"], "S99.1 Gate A");
        assert_eq!(v["rows"][1]["gate"], "S99.1 Gate A");
        assert_eq!(v["rows"][2]["gate"], serde_json::Value::Null);
        assert_eq!(v["rows"].as_array().unwrap().len(), 3);
    }
}
