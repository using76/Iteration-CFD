// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Gate DST-B-VES (SPEC-LIT §159.12): the Phase B dust cloud in the closed
//! 20 L sphere, REPLAYED from the supervisor's two recorded runs - the gate
//! itself runs nothing. Every row is recomputed from the shipped cases and
//! the record of the build it reports on: the tube states of (159.28) on
//! the gas-and-cloud isentrope of (159.27), each tube's `S_u` by (159.23)
//! over its recorded fronts, the sphere's last logged `p0` against `p_v` of
//! (159.21) - the cloud's own AICC - and the time the history crosses
//! `(p0 + p_e)/2` against `t_half,ref` of (159.30), (157.4)'s thin-flame
//! rate integrated from the run's own (157.11) anchor with `S_u(p)`
//! log-log between the tube speeds. K_st against the measured 200 bar m/s
//! +- 15 % is NAMED and NOT RUN.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §159.12 (this gate as built), §157.2
//!     ((157.2)-(157.5), the thin-flame relations and (157.5)'s K_st),
//!     §157.9 ((157.11), the run's own anchor), §159.10 ((159.21), the host
//!     AICC of the cloud) and §159.11 ((159.23), the tube front's speed)
//!
//! The specification was informed by a review of FSCDustFoam (Huang & Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use std::path::Path;
use std::time::Instant;

use ofgpu::chem::thermo::{ElementTable, R_GAS, SpeciesThermo, ThermoSet};
use ofgpu::dust::cloud::{cloud_aicc, CloudCharge, CloudThermo, StageDatums};
use ofgpu::dust::particle::Drying;
use ofgpu::dust::vessel0d::Vessel;
use ofgpu::error::IoContext;
use ofgpu::io::case_json::read_case_jsonc;
use ofgpu::{Error, Result, Scalar};

use crate::{Checks, GateReport, How, Uncertainty, Verdict};

/// The gate's name: `validate.rs` opens the scope with this constant and
/// [`check_dst_b_vessel`] reports under it, so the two cannot drift apart.
pub(crate) const GATE: &str = "SPEC-LIT 159.12 Gate DST-B-VES Phase B cloud in a closed 20 L sphere: the AICC end pressure and the thin-flame half time";

/// What the rows are held against: this tree's own host references.
const AGAINST: &str = "the cloud's host AICC (159.21) and the thin-flame half time (159.30) with S_u(p) of the 1-D Phase B tube";

/// The sphere case, relative to the repository root - the host state's,
/// the AICC's and the anchor's input.
pub(crate) const CASE: &str = "cases/dust/sphere20LCloud.jsonc";
/// The burning-velocity tube, relative to the repository root - the FNV
/// rows' file, and (159.28)'s fresh state and mesh length.
pub(crate) const TUBE: &str = "cases/dust/tube20LCloud.jsonc";
/// The f64 build's record: the supervisor's recorded sphere run and its
/// three tube runs. The replay reads it, never repeats them.
pub(crate) const RECORD_F64: &str = "cases/dust/records/sphere20LCloud.f64.json";
/// The `single` build's record, the same runs recorded with f32 storage.
pub(crate) const RECORD_SINGLE: &str = "cases/dust/records/sphere20LCloud.single.json";
/// (P) the band on |p_end/p_v - 1| - DST-17's 0.5 % (SPEC-LIT 159.6).
pub(crate) const P_END_BAND: f64 = 0.005;
/// (P) the band on |t_half,run/t_half,ref - 1| - DST-17's 5 %.
pub(crate) const T_HALF_BAND: f64 = 0.05;
/// (157.11)'s anchor: the first logged k >= 1 with x_b >= 1e-3.
pub(crate) const X_ANCHOR: f64 = 1.0e-3;
/// The front isotherm of (159.23), K - the fronts the record already holds.
pub(crate) const T_ISO: f64 = 1000.0;
/// The speed window, fractions of the tube's length (SPEC-LIT 159.12).
pub(crate) const WINDOW: (f64, f64) = (0.2, 0.7);
/// The tube states, p/p0 - one recorded run each.
pub(crate) const P_RATIOS: [f64; 3] = [1.0, 3.0, 6.0];
/// The band on the record's (159.28) tube states: recomputed here, they
/// must agree to a re-run of the same formula in f64.
pub(crate) const REL_STATE: f64 = 1.0e-9;

/// The driver's normal-end wording (`lowmach.rs`'s `run_end_line`): a run
/// only "completes" when the exit was a success AND the log ended this way.
const ENDED_OK: &str = "run ended: budget";
/// The Simpson intervals of each piece of (159.30)'s integral.
const SIMPSON_N: usize = 4096;
/// The five gas members, `species.names`' order, N2 last - its mass
/// fraction is what the others leave (SPEC-LIT 159.12).
const GAS_MEMBERS: [&str; 5] = ["VOL", "O2", "H2O", "CO2", "N2"];

/// The sphere rows' names, one source for the live path and a dead record's.
const ROW_SPHERE_ENDS: &str = "sphere: the run completes";
const ROW_SPHERE_P_END: &str = "sphere: |p_end/p_v - 1|";
const ROW_SPHERE_T_HALF: &str = "sphere: |t_half,run/t_half,ref - 1|";

/// The rows' names, one source for the live path and a dead record's.
fn row_build(want_build: &str) -> String {
    format!("the record's build is {want_build}")
}

fn row_sphere_case(sphere_hex: u64) -> String {
    format!("the record was made from the shipped sphere case (FNV-1a 0x{sphere_hex:x})")
}

fn row_tube_case(r: f64, tube_hex: u64) -> String {
    format!("tube p/p0 = {r}: the record was made from the shipped tube case (FNV-1a 0x{tube_hex:x})")
}

fn row_tube_state(r: f64) -> String {
    format!("tube p/p0 = {r}: T_u and C are (159.28)'s")
}

fn row_tube_ends(r: f64) -> String {
    format!("tube p/p0 = {r}: the run completes")
}

fn row_tube_fronts(r: f64) -> String {
    format!("tube p/p0 = {r}: >= 4 fronts in [0.2 L, 0.7 L]")
}

/// FNV-1a 64 of the bytes with every "\r\n" replaced by "\n" - the hash
/// the records pin the shipped cases with (offset 0xcbf29ce484222325,
/// prime 0x100000001b3, lower-case hex when spelled).
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut h = OFFSET;
    let mut i = 0;
    while i < bytes.len() {
        let b = if bytes[i] == b'\r' && bytes.get(i + 1) == Some(&b'\n') {
            i += 1;
            b'\n'
        } else {
            bytes[i]
        };
        h ^= u64::from(b);
        h = h.wrapping_mul(PRIME);
        i += 1;
    }
    h
}

/// A "0x…" word as a u64, blind to the digits' case.
fn hex64(s: &str) -> Option<u64> {
    u64::from_str_radix(s.strip_prefix("0x").unwrap_or(s), 16).ok()
}

/// The record's tube entry at `p/p0 = r`, if it holds one.
fn tube_of<'a>(rec: &'a serde_json::Value, r: f64) -> Option<&'a serde_json::Value> {
    rec["tubes"]
        .as_array()?
        .iter()
        .find(|t| t["p_ratio"].as_f64().is_some_and(|p| (p - r).abs() <= 1.0e-9))
}

/// The entry's fronts as (t, x_f) pairs; an entry without fronts holds none.
fn fronts_of(entry: &serde_json::Value) -> Vec<(f64, f64)> {
    entry["fronts"]
        .as_array()
        .map(|fs| {
            fs.iter()
                .filter_map(|f| Some((f.get(0)?.as_f64()?, f.get(1)?.as_f64()?)))
                .collect()
        })
        .unwrap_or_default()
}

/// (159.27)'s Y_k from the case's `species.initial`, N2 what the others
/// leave: VOL 0, O2 0.233, H2O 0, CO2 0 and N2 0.767 here.
fn mass_fractions(v: &serde_json::Value) -> [f64; 5] {
    let ini = &v["species"]["initial"];
    let mut y = [0.0_f64; 5];
    let mut sum = 0.0;
    for (k, name) in GAS_MEMBERS.iter().enumerate() {
        if *name == "N2" {
            continue;
        }
        y[k] = ini.get(*name).and_then(|x| x.as_f64()).unwrap_or(0.0);
        sum += y[k];
    }
    y[4] = 1.0 - sum;
    y
}

/// (159.27)'s opening gas state from the sphere case and its mechanism
/// records: the partial densities `Y_k rho_g0` for [`cloud_aicc`], then
/// `rho_g0 = p0/(R0 T0 sum Y_k/W_k)`, `cp_g = sum Y_k cp_k(T0)/W_k` and
/// `cv_g = cp_g - R0 sum Y_k/W_k`.
fn initial_gas(
    v: &serde_json::Value,
    set: &ThermoSet,
    t0: f64,
    p0: f64,
) -> Result<(Vec<(SpeciesThermo, f64)>, f64, f64, f64)> {
    let y = mass_fractions(v);
    let recs: Vec<&SpeciesThermo> = GAS_MEMBERS
        .iter()
        .map(|n| {
            set.get(n)
                .ok_or_else(|| Error::Config(format!("{CASE}'s mechanism has no {n} record")))
        })
        .collect::<Result<Vec<_>>>()?;
    let sum_y_w: f64 = y.iter().zip(&recs).map(|(yk, r)| yk / r.molar_mass).sum();
    let mut cp_g = 0.0;
    for (yk, r) in y.iter().zip(&recs) {
        cp_g += yk * r.cp(t0)? / r.molar_mass;
    }
    let rho_g0 = p0 / (R_GAS * t0 * sum_y_w);
    let cv_g = cp_g - R_GAS * sum_y_w;
    let gas = recs
        .iter()
        .zip(y.iter())
        .map(|(r, yk)| ((*r).clone(), yk * rho_g0))
        .collect();
    Ok((gas, rho_g0, cp_g, cv_g))
}

/// (159.27): the gas-and-cloud isentrope's exponent at the opening state,
/// `(rho_g0 cp_g + C cP)/(rho_g0 cv_g + C cP)` - the cloud's `cP` stiffens
/// the gas's own ratio down to about 1.313 here.
pub(crate) fn gamma_u(rho_g0: f64, cp_g: f64, cv_g: f64, c: f64, cp_p: f64) -> f64 {
    (rho_g0 * cp_g + c * cp_p) / (rho_g0 * cv_g + c * cp_p)
}

/// (159.28): the compressed fresh state at `p/p0 = ratio` - `(T_u, C)` with
/// `T_u = T0 (p/p0)^((gamma_u - 1)/gamma_u)` and `C = C0 (p/p0)^(1/gamma_u)`,
/// the cloud's mass per volume of the compressed unburned gas.
pub(crate) fn compressed(t0: f64, c0: f64, gamma_u: f64, ratio: f64) -> (f64, f64) {
    (
        t0 * ratio.powf((gamma_u - 1.0) / gamma_u),
        c0 * ratio.powf(1.0 / gamma_u),
    )
}

/// `S_u(p)` of (159.29): log-log linear in `p/p0` between the tube speeds,
/// the end segment's exponent beyond the end nodes. The nodes are the
/// recorded tubes' speeds.
#[derive(Debug, Clone)]
pub(crate) struct SuTable {
    /// The nodes' pressure ratios, increasing.
    pub ratio: Vec<f64>,
    /// m/s, each node's speed.
    pub s: Vec<f64>,
}

impl SuTable {
    /// The table, refused by name (`Error::Refused`, SPEC-LIT 159.12):
    /// fewer than two nodes, non-increasing ratios, or a speed that is not
    /// a finite positive burning velocity.
    pub(crate) fn new(ratio: Vec<f64>, s: Vec<f64>) -> Result<SuTable> {
        const W: &str = "SPEC-LIT 159.12's S_u table of the 1-D tube flames";
        if ratio.len() < 2 || s.len() != ratio.len() {
            return Err(Error::Refused(format!(
                "{W}: {} ratio nodes against {} speeds - the log-log law needs \
                 two or more of each",
                ratio.len(),
                s.len()
            )));
        }
        for w in ratio.windows(2) {
            if !(w[1] > w[0]) {
                return Err(Error::Refused(format!(
                    "{W}: the ratios do not increase: {} then {}",
                    w[0], w[1]
                )));
            }
        }
        for &v in &s {
            if !v.is_finite() || v <= 0.0 {
                return Err(Error::Refused(format!(
                    "{W}: a speed is {v}, not a finite positive burning velocity"
                )));
            }
        }
        Ok(SuTable { ratio, s })
    }

    /// `S_u` at the pressure ratio `p/p0`: log-log linear between the
    /// nodes, the end segment's exponent outside them.
    pub(crate) fn s_u(&self, ratio: f64) -> f64 {
        let mut i = 0;
        for k in 0..self.ratio.len() - 1 {
            if ratio >= self.ratio[k] {
                i = k;
            }
        }
        let (r0, s0) = (self.ratio[i], self.s[i]);
        let (r1, s1) = (self.ratio[i + 1], self.s[i + 1]);
        let e = (s1 / s0).ln() / (r1 / r0).ln();
        s0 * (ratio / r0).powf(e)
    }
}

/// (157.4)'s thin-flame `dp/dt` with [`SuTable`]'s `S_u`: `3 (p_e - p0)/R
/// bracket^(2/3) (p/p0)^(1/gamma_u) S_u(p)`, the vessel's own (157.3)
/// bracket.
fn rate_of(v: &Vessel, table: &SuTable, p: f64) -> f64 {
    3.0 * (v.pe - v.p0) / v.radius
        * v.bracket(p).powf(2.0 / 3.0)
        * (p / v.p0).powf(1.0 / v.gamma_u)
        * table.s_u(p / v.p0)
}

/// Composite Simpson at [`SIMPSON_N`] intervals of `1/rate` over one piece.
fn simpson_time(v: &Vessel, table: &SuTable, a: f64, b: f64) -> f64 {
    let f = |p: f64| 1.0 / rate_of(v, table, p);
    let h = (b - a) / SIMPSON_N as f64;
    let mut s = f(a) + f(b);
    for i in 1..SIMPSON_N {
        s += if i % 2 == 1 { 4.0 } else { 2.0 } * f(a + i as f64 * h);
    }
    s * h / 3.0
}

/// (159.30): `t_half,ref = t_s + integral_{p_s}^{p_half} dp/rate(p)` with
/// `p_half = (p0 + p_e)/2`, the integral split at the table's nodes that
/// fall inside it, composite Simpson at 4096 intervals on each piece.
pub(crate) fn t_half_ref(v: &Vessel, table: &SuTable, t_s: f64, p_s: f64) -> f64 {
    let p_half = 0.5 * (v.p0 + v.pe);
    let mut t = t_s;
    let mut lo = p_s;
    for &node in &table.ratio {
        let cut = node * v.p0;
        if cut > lo && cut < p_half {
            t += simpson_time(v, table, lo, cut);
            lo = cut;
        }
    }
    t + simpson_time(v, table, lo, p_half)
}

/// (159.23) over the recorded fronts: the least-squares slope of `x_f` on
/// `t` for the fronts inside `[0.2 L, 0.7 L]`, negated - the front runs
/// toward the closed end. `None` below four fronts.
pub(crate) fn tube_speed(fronts: &[(f64, f64)], length: f64) -> Option<f64> {
    let pts: Vec<(f64, f64)> = fronts
        .iter()
        .copied()
        .filter(|(_, x)| *x >= WINDOW.0 * length && *x <= WINDOW.1 * length)
        .collect();
    if pts.len() < 4 {
        return None;
    }
    let n = pts.len() as f64;
    let (st, sx) = pts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
    let (mt, mx) = (st / n, sx / n);
    let mut num = 0.0;
    let mut den = 0.0;
    for (t, x) in &pts {
        num += (t - mt) * (x - mx);
        den += (t - mt) * (t - mt);
    }
    if den == 0.0 {
        return None;
    }
    Some(-(num / den))
}

/// The first upward crossing of `target`, linear between the samples: the
/// crossing's time, or `None` when `y` never rises through it.
pub(crate) fn cross_up(t: &[f64], y: &[f64], target: f64) -> Option<f64> {
    for i in 0..t.len().saturating_sub(1) {
        if y[i] < target && y[i + 1] >= target {
            let f = (target - y[i]) / (y[i + 1] - y[i]);
            return Some(t[i] + f * (t[i + 1] - t[i]));
        }
    }
    None
}

/// A record that does not read - or holds nothing this gate knows - fails
/// every later row by name, each with its MISSED note: a row that fails
/// stays failed.
fn fail_the_replay(
    c: &mut Checks,
    want_build: &str,
    sphere_hex: u64,
    tube_hex: u64,
    cause: &str,
    mut missed: Vec<String>,
) {
    let mut rows = vec![row_build(want_build), row_sphere_case(sphere_hex)];
    for r in P_RATIOS {
        rows.push(row_tube_case(r, tube_hex));
        rows.push(row_tube_state(r));
        rows.push(row_tube_ends(r));
        rows.push(row_tube_fronts(r));
    }
    rows.push(ROW_SPHERE_ENDS.to_string());
    rows.push(ROW_SPHERE_P_END.to_string());
    rows.push(ROW_SPHERE_T_HALF.to_string());
    for what in &rows {
        c.require(what, false);
        missed.push(format!("MISSED: {what}: {cause}"));
    }
    report_misses(c, missed);
}

/// Every failed row's MISSED line, carried by the registry - §69.2 lets
/// the word through no other way (a plain note of it is an unregistered
/// verdict). A replay: the runs are the records'.
fn report_misses(c: &mut Checks, missed: Vec<String>) {
    if missed.is_empty() {
        return;
    }
    c.report(GateReport {
        verdict: Verdict::Misses,
        how: How::Replayed,
        gate: GATE,
        against: AGAINST,
        headline: format!("{} of the gate's rows leave the bands of 159.12", missed.len()),
        detail: missed,
        uncertainty: Some(Uncertainty::SingleMesh(
            "one 801-cell sector and one 120-cell tube per pressure, as recorded; no \
             refinement sequence",
        )),
    });
}

/// Gate DST-B-VES. Opened by `validate.rs`'s section under [`GATE`].
pub(crate) fn check_dst_b_vessel(c: &mut Checks) -> Result<()> {
    let clock = Instant::now();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let want_build = if cfg!(feature = "single") { "single" } else { "f64" };
    let record_rel = if cfg!(feature = "single") { RECORD_SINGLE } else { RECORD_F64 };
    let record_path = root.join(record_rel);

    // The shipped cases' hashes, independent of the record's own health:
    // the FNV rows state them whatever the record turns out to hold.
    let sphere_path = root.join(CASE);
    let tube_path = root.join(TUBE);
    let sphere_hex = fnv1a64(&std::fs::read(&sphere_path).path(&sphere_path)?);
    let tube_hex = fnv1a64(&std::fs::read(&tube_path).path(&tube_path)?);

    // Row 1: the record reads. Everything else hangs off it, so a record
    // that does not read fails every later row by name and stops.
    let rec: serde_json::Value = match std::fs::read_to_string(&record_path)
        .map_err(|e| e.to_string())
    {
        Ok(text) => match serde_json::from_str(&text).map_err(|e| e.to_string()) {
            Ok(v) => v,
            Err(e) => {
                let what = format!("the record {record_rel} reads");
                c.require(&what, false);
                let first = vec![format!("MISSED: {what}: it does not parse: {e}")];
                fail_the_replay(c, want_build, sphere_hex, tube_hex, &format!("the record does not parse: {e}"), first);
                return Ok(());
            }
        },
        Err(e) => {
            let what = format!("the record {record_rel} reads");
            c.require(&what, false);
            let first = vec![format!("MISSED: {what}: {e}")];
            fail_the_replay(c, want_build, sphere_hex, tube_hex, &format!("the record does not read: {e}"), first);
            return Ok(());
        }
    };
    c.require(&format!("the record {record_rel} reads"), true);
    let mut missed: Vec<String> = Vec::new();
    if !(rec.get("sphere").map_or(false, |s| s.is_object())
        && rec.get("tubes").map_or(false, |t| t.is_array())
        && rec.get("build").map_or(false, |b| b.is_string()))
    {
        fail_the_replay(
            c,
            want_build,
            sphere_hex,
            tube_hex,
            "the record holds no sphere/tubes/build fields as DST-17 records them",
            Vec::new(),
        );
        return Ok(());
    }

    // The record's build, then its two case pins.
    let got_build = rec["build"].as_str().unwrap_or("");
    let build_ok = got_build == want_build;
    c.require(&row_build(want_build), build_ok);
    if !build_ok {
        missed.push(format!(
            "MISSED: {}: the record says \"{got_build}\"",
            row_build(want_build)
        ));
    }
    let got_hex = |s: &serde_json::Value| s.as_str().and_then(hex64);
    let sphere_ok = got_hex(&rec["sphere"]["case_fnv1a"]) == Some(sphere_hex);
    c.require(&row_sphere_case(sphere_hex), sphere_ok);
    if !sphere_ok {
        missed.push(format!(
            "MISSED: {}: the record says {}",
            row_sphere_case(sphere_hex),
            rec["sphere"]["case_fnv1a"].as_str().unwrap_or("nothing")
        ));
    }

    // The host state, all from the sphere case, its mechanism and the
    // record's p0: the gas-and-cloud isentrope, the cloud's own AICC and
    // the equal-volume sphere the thin-flame rate integrates over.
    let sphere_case = read_case_jsonc(&sphere_path)?;
    let v = serde_json::to_value(&sphere_case)
        .map_err(|e| Error::Config(format!("{CASE} does not serialise: {e}")))?;
    let lowered = sphere_case.lower()?;
    let chem = lowered
        .chem
        .as_ref()
        .ok_or_else(|| Error::Config(format!("{CASE} lowers with no chemistry")))?;
    let mechanism = chem.load(sphere_path.parent().expect("the case's directory"))?;
    let set = mechanism.thermo();
    let t0 = v["initial"]["T"]
        .as_f64()
        .ok_or_else(|| Error::Config(format!("{CASE} has no initial.T")))?;
    let c0 = v["dust"]["cloud"]["concentration"]
        .as_f64()
        .ok_or_else(|| Error::Config(format!("{CASE} has no dust.cloud.concentration")))?;
    let cp_p = v["dust"]["cloud"]["cpP"]
        .as_f64()
        .ok_or_else(|| Error::Config(format!("{CASE} has no dust.cloud.cpP")))?;
    let dh = v["dust"]["cloud"]["devolatilisation"]["dH"]
        .as_f64()
        .ok_or_else(|| Error::Config(format!("{CASE} has no dust.cloud.devolatilisation.dH")))?;
    let fractions = [
        v["dust"]["cloud"]["composition"]["moisture"].as_f64().unwrap_or(0.0),
        v["dust"]["cloud"]["composition"]["volatile"].as_f64().unwrap_or(0.0),
        v["dust"]["cloud"]["composition"]["char"].as_f64().unwrap_or(0.0),
        v["dust"]["cloud"]["composition"]["ash"].as_f64().unwrap_or(0.0),
    ];
    let r_s = v["mesh"]["bounds"]["max"][0]
        .as_f64()
        .ok_or_else(|| Error::Config(format!("{CASE} has no mesh.bounds.max[0]")))?;
    let p0 = rec["sphere"]["p0"]
        .as_f64()
        .ok_or_else(|| Error::Config(format!("{record_rel} has no sphere.p0")))?;
    let (gas, rho_g0, cp_g, cv_g) = initial_gas(&v, set, t0, p0)?;
    let g = gamma_u(rho_g0, cp_g, cv_g, c0, cp_p);
    let th = CloudThermo::from_set(set, &ElementTable::standard())?;
    let datums = StageDatums::new(&th, &Drying::water(p0)?, dh)?;
    let charge = CloudCharge { concentration: c0, fractions, cp: cp_p, datums };
    let p_v = cloud_aicc(&gas, t0, &charge, &th)?.p_v;
    let p_half = 0.5 * (p0 + p_v);
    let vessel = Vessel::new(r_s, p0, p_v, g)?;
    c.note(&format!(
        "Gate DST-B-VES host: C = {c0} kg/m3, rho_g0 = {rho_g0:.6} kg/m3, gamma_u = {g:.8}, \
         T0 = {t0} K, p0 = {p0} Pa, p_v = {p_v:.3} Pa (AICC), p_half = {p_half:.3} Pa, R = {r_s} m"
    ));

    // The tube rows, one recorded run per p/p0: the state of (159.28), the
    // completion, and the fronts of (159.23) whose least squares is S_u.
    let tube_case = read_case_jsonc(&tube_path)?;
    let tv = serde_json::to_value(&tube_case)
        .map_err(|e| Error::Config(format!("{TUBE} does not serialise: {e}")))?;
    let t0_tube = tv["initial"]["T"]
        .as_f64()
        .ok_or_else(|| Error::Config(format!("{TUBE} has no initial.T")))?;
    let c0_tube = tv["dust"]["cloud"]["concentration"]
        .as_f64()
        .ok_or_else(|| Error::Config(format!("{TUBE} has no dust.cloud.concentration")))?;
    let l_tube = tv["mesh"]["bounds"]["max"][0]
        .as_f64()
        .ok_or_else(|| Error::Config(format!("{TUBE} has no mesh.bounds.max[0]")))?;
    let mut speeds: [Option<f64>; 3] = [None; 3];
    for (j, r) in P_RATIOS.iter().enumerate() {
        let entry = tube_of(&rec, *r);
        let Some(e) = entry else {
            for what in [
                row_tube_case(*r, tube_hex),
                row_tube_state(*r),
                row_tube_ends(*r),
                row_tube_fronts(*r),
            ] {
                c.require(&what, false);
                missed.push(format!("MISSED: {what}: the record holds no p/p0 = {r} tube"));
            }
            continue;
        };
        let what_case = row_tube_case(*r, tube_hex);
        let hash_ok = got_hex(&e["case_fnv1a"]) == Some(tube_hex);
        c.require(&what_case, hash_ok);
        if !hash_ok {
            missed.push(format!(
                "MISSED: {what_case}: the record says {}",
                e["case_fnv1a"].as_str().unwrap_or("nothing")
            ));
        }
        // (159.28): the state the tube run was made at.
        let (t_u_own, c_own) = compressed(t0_tube, c0_tube, g, *r);
        let what_state = row_tube_state(*r);
        match (e["T_u"].as_f64(), e["C"].as_f64()) {
            (Some(rt), Some(rc)) => {
                let err = ((rt / t_u_own - 1.0).abs()).max((rc / c_own - 1.0).abs());
                c.check_d(&what_state, err as Scalar, REL_STATE as Scalar);
                if !(err <= REL_STATE) {
                    missed.push(format!(
                        "MISSED: {what_state}: T_u = {rt} against {t_u_own:.6}, \
                         C = {rc} against {c_own:.6}, the bar {REL_STATE}"
                    ));
                }
            }
            _ => {
                c.require(&what_state, false);
                missed.push(format!("MISSED: {what_state}: the record holds no T_u or no C"));
            }
        }
        let ends_ok = e["end"].as_str().is_some_and(|s| s.contains(ENDED_OK));
        let what_ends = row_tube_ends(*r);
        c.require(&what_ends, ends_ok);
        if !ends_ok {
            missed.push(format!(
                "MISSED: {what_ends}: the end line is \"{}\"",
                e["end"].as_str().unwrap_or("none")
            ));
        }
        let fronts = fronts_of(e);
        let in_window = fronts
            .iter()
            .filter(|(_, x)| *x >= WINDOW.0 * l_tube && *x <= WINDOW.1 * l_tube)
            .count();
        let what_fronts = row_tube_fronts(*r);
        c.require(&what_fronts, in_window >= 4);
        speeds[j] = tube_speed(&fronts, l_tube);
        match speeds[j] {
            Some(s) => c.note(&format!(
                "tube p/p0 = {r}: S_u = {s:.6} m/s over {in_window} fronts of the \
                 {T_ISO:.0} K isotherm in [{:.5}, {:.5}] m, wall {:.1} s",
                WINDOW.0 * l_tube,
                WINDOW.1 * l_tube,
                e["wall_s"].as_f64().unwrap_or(f64::NAN)
            )),
            None => missed.push(format!(
                "MISSED: {what_fronts}: {in_window} fronts in [0.2 L, 0.7 L], the bar is 4 - no S_u"
            )),
        }
    }

    // The sphere rows: the completion, then (P) the end pressure and (P)
    // the half time - each only where its input exists.
    let ends_ok = rec["sphere"]["end"].as_str().is_some_and(|s| s.contains(ENDED_OK));
    c.require(ROW_SPHERE_ENDS, ends_ok);
    if !ends_ok {
        missed.push(format!(
            "MISSED: {ROW_SPHERE_ENDS}: the end line is \"{}\"",
            rec["sphere"]["end"].as_str().unwrap_or("none")
        ));
    }
    let hist = rec["sphere"]["history"].as_array().cloned().unwrap_or_default();
    let ts: Vec<f64> = hist.iter().filter_map(|h| h.get(0).and_then(|x| x.as_f64())).collect();
    let ps: Vec<f64> = hist.iter().filter_map(|h| h.get(1).and_then(|x| x.as_f64())).collect();
    let dpdts: Vec<f64> = hist.iter().filter_map(|h| h.get(2).and_then(|x| x.as_f64())).collect();
    let what_p = ROW_SPHERE_P_END;
    if ends_ok {
        match ps.last().copied() {
            Some(p_end) => {
                let err = (p_end / p_v - 1.0).abs();
                c.check_p(what_p, err as Scalar, P_END_BAND as Scalar);
                c.note(&format!(
                    "sphere: p_end = {p_end:.3} Pa against p_v = {p_v:.3} Pa, \
                     |p_end/p_v - 1| = {err:.6}, the bar {P_END_BAND}"
                ));
                if !(err <= P_END_BAND) {
                    missed.push(format!("MISSED: {what_p}: {err:.6}, the bar is {P_END_BAND}"));
                }
            }
            None => {
                c.require(what_p, false);
                missed.push(format!(
                    "MISSED: {what_p}: the completed run's record holds no history"
                ));
            }
        }
    } else {
        c.require(what_p, false);
        missed.push(format!(
            "MISSED: {what_p}: the run did not complete; its end line is \"{}\"",
            rec["sphere"]["end"].as_str().unwrap_or("none")
        ));
    }

    // (P) the half time: t_half,run the history's crossing of p_half,
    // t_half,ref (159.30) from the run's own (157.11) anchor - only where
    // all three tube speeds, the crossing and the anchor exist.
    let table = match speeds.iter().copied().collect::<Option<Vec<f64>>>() {
        Some(ss) => SuTable::new(P_RATIOS.to_vec(), ss)
            .map(Some)
            .unwrap_or_else(|e| {
                missed.push(format!("MISSED: the S_u table of (159.29) is refused: {e}"));
                None
            }),
        None => None,
    };
    let t_cross = cross_up(&ts, &ps, p_half);
    let anchor = ps
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, &p)| (p - p0) / (p_v - p0) >= X_ANCHOR)
        .map(|(k, &p)| (ts[k], p));
    match (&table, t_cross, anchor) {
        (Some(tab), Some(t_run), Some((t_s, p_s))) => {
            let t_ref = t_half_ref(&vessel, tab, t_s, p_s);
            let err = (t_run / t_ref - 1.0).abs();
            c.check_p(ROW_SPHERE_T_HALF, err as Scalar, T_HALF_BAND as Scalar);
            c.note(&format!(
                "sphere: t_half,run = {t_run:.6} s (the history's crossing of p_half), \
                 t_half,ref = {t_ref:.6} s (from t_s = {t_s:.6} s at p_s = {p_s:.3} Pa), \
                 the bar {T_HALF_BAND}"
            ));
            if !(err <= T_HALF_BAND) {
                missed.push(format!(
                    "MISSED: {ROW_SPHERE_T_HALF}: {err:.6}, the bar is {T_HALF_BAND}"
                ));
            }
        }
        _ => {
            c.require(ROW_SPHERE_T_HALF, false);
            let mut why = format!("MISSED: {ROW_SPHERE_T_HALF}:");
            let no_speed: Vec<String> = P_RATIOS
                .iter()
                .zip(speeds.iter())
                .filter(|(_, s)| s.is_none())
                .map(|(r, _)| format!("p/p0 = {r}"))
                .collect();
            if !no_speed.is_empty() {
                why.push_str(&format!(" no S_u from the tubes {}", no_speed.join(", ")));
            }
            if t_cross.is_none() {
                if !no_speed.is_empty() {
                    why.push(';');
                }
                why.push_str(" the history does not cross p_half");
            }
            if anchor.is_none() {
                why.push_str("; no history point k >= 1 reaches x_b = 1e-3");
            }
            missed.push(why);
        }
    }

    // K_st: the run's own (dp/dt)_max through the cube-root law, the
    // (157.5) value when the S_u table exists, and the measured number
    // that stays NOT RUN.
    let last_t = ts.last().copied().unwrap_or(f64::NAN);
    let last_p = ps.last().copied().unwrap_or(f64::NAN);
    let dpdt_max = dpdts.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if ends_ok {
        let k_run = vessel.k_st(dpdt_max) / 1.0e5;
        c.note(&format!(
            "sphere: the last history point t = {last_t:.4} s at p0 = {last_p:.3} Pa; \
             (dp/dt)_max = {dpdt_max:.1} Pa/s, K_st,run = {k_run:.3} bar m/s"
        ));
    } else {
        // A run that stopped before it burned has no (dp/dt)_max to put
        // through the cube-root law: K_st,run is not reported.
        c.note(&format!(
            "sphere: the last history point t = {last_t:.4} s at p0 = {last_p:.3} Pa; \
             the run stopped before it burned, so K_st,run is not reported"
        ));
    }
    if let Some(tab) = &table {
        let s_end = tab.s_u(p_v / p0);
        let k_law = 3.0 * (4.0 * std::f64::consts::PI / 3.0).cbrt() * (p_v - p0)
            * (p_v / p0).powf(1.0 / g)
            * s_end
            / 1.0e5;
        c.note(&format!(
            "sphere: K_st of (157.5) = {k_law:.3} bar m/s with S_u(p_e/p0) = {s_end:.6} m/s"
        ));
    }
    c.skip(
        "Gate DST-B-VES: K_st against the measured 200 bar m/s +- 15 % (H22 section 2)",
        "NAMED and NOT RUN: the standard test's decaying-turbulence initial state is not openly published (SPEC-LIT 159.12)",
    );
    report_misses(c, missed);
    c.note(&format!(
        "Gate DST-B-VES: a replay of the supervisor's records, no run of its own - \
         {:.3} s wall",
        clock.elapsed().as_secs_f64()
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ofgpu::dust::vessel0d::{integrate, metrics, BurningLaw, UnburnedDensity};

    /// The tests' expected mesh numbers: the sphere's equal-volume radius
    /// and the tube's length, each the case's own `mesh.bounds.max[0]`.
    const R_SPHERE: f64 = 0.1684;
    const L_TUBE: f64 = 0.0252;

    /// Both shipped cases lower to exactly the inputs the records were made
    /// from: the 800-cell sector of R = 0.1684 m with the all-volatile
    /// 1 um cloud at C = 0.15418, A = 1.5e7 and the apex igniter, and the
    /// 120-cell, 25.2 mm tube with the same cloud, no igniter, Le_VOL 2.
    #[test]
    fn the_cases_lower_to_the_gate_inputs() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let sphere = read_case_jsonc(&root.join(CASE)).expect("the sphere case reads");
        let v = serde_json::to_value(&sphere).expect("serialises");
        assert_eq!(v["mesh"]["bounds"]["max"][0], serde_json::json!(R_SPHERE));
        assert_eq!(v["mesh"]["cells"][0], serde_json::json!(800));
        assert_eq!(v["mesh"]["sector"]["fineRadius"], serde_json::json!(0.16819));
        assert_eq!(v["mesh"]["sector"]["growth"], serde_json::json!(1.0));
        let lowered = sphere.lower().expect("lowers");
        let cloud = lowered.cloud.as_ref().expect("the cloud");
        assert!((cloud.concentration - 0.15418).abs() <= 1.0e-12, "{}", cloud.concentration);
        assert_eq!(cloud.diameters, vec![1.0e-6]);
        assert_eq!(
            (cloud.composition.moisture, cloud.composition.volatile,
             cloud.composition.char, cloud.composition.ash),
            (0.0, 1.0, 0.0, 0.0)
        );
        let devol = cloud.devol.as_ref().expect("the devol");
        assert_eq!(devol.a, 1.5e7);
        assert_eq!(v["dust"]["cloud"]["igniter"]["point"], serde_json::json!([0.0, 0.0, 0.0]));
        assert_eq!(v["dust"]["cloud"]["igniter"]["radius"], serde_json::json!(0.0021));
        assert_eq!(v["dust"]["cloud"]["igniter"]["power"], serde_json::json!(1e8));
        assert_eq!(v["dust"]["cloud"]["igniter"]["duration"], serde_json::json!(0.01));
        let tube = read_case_jsonc(&root.join(TUBE)).expect("the tube case reads");
        let tv = serde_json::to_value(&tube).expect("serialises");
        assert_eq!(tv["mesh"]["cells"][0], serde_json::json!(120));
        assert_eq!(tv["mesh"]["bounds"]["max"][0], serde_json::json!(L_TUBE));
        let tl = tube.lower().expect("lowers");
        let tcloud = tl.cloud.as_ref().expect("the cloud");
        assert!((tcloud.concentration - 0.15418).abs() <= 1.0e-12);
        assert_eq!(
            (tcloud.composition.moisture, tcloud.composition.volatile,
             tcloud.composition.char, tcloud.composition.ash),
            (0.0, 1.0, 0.0, 0.0)
        );
        assert!(
            tv["dust"]["cloud"].get("igniter").is_none(),
            "the tube lights from its 2000 K window, not an igniter"
        );
        assert_eq!(tv["species"]["transport"]["Le"]["VOL"], serde_json::json!(2.0));
    }

    /// (159.27) on the sphere case's own records: the gate's pipeline and
    /// the same number written out by hand from `cp(300)` and the molar
    /// masses agree to 1e-12 relative, the value is 1.31305 to 1e-4, and
    /// `compressed` is the (159.28) isentrope to round-off.
    #[test]
    fn gamma_u_is_the_isentrope_of_gas_and_cloud() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let path = root.join(CASE);
        let case = read_case_jsonc(&path).expect("the sphere case reads");
        let v = serde_json::to_value(&case).expect("serialises");
        let lowered = case.lower().expect("lowers");
        let chem = lowered.chem.as_ref().expect("the chemistry");
        let mech = chem.load(path.parent().expect("the case's dir")).expect("loads");
        let set = mech.thermo();
        let t0 = v["initial"]["T"].as_f64().expect("T");
        let p0 = 101_325.0;
        let c0 = v["dust"]["cloud"]["concentration"].as_f64().expect("C");
        let cpp = v["dust"]["cloud"]["cpP"].as_f64().expect("cpP");
        let (gas, rho_g0, cp_g, cv_g) = initial_gas(&v, set, t0, p0).expect("the opening gas");
        // The partial densities ride the same rho_g0 the isentrope reads.
        let mut rho_sum = 0.0;
        for (rec, rho) in &gas {
            rho_sum += rho / rec.molar_mass;
        }
        assert!((rho_sum / (p0 / (R_GAS * t0)) - 1.0).abs() <= 1.0e-12, "p0 = R0 T0 sum rho/W");
        // The hand formula: the case's own Y (VOL 0, O2 0.233, H2O 0,
        // CO2 0, N2 the rest) against cp(300) and the molar masses.
        let rec = |n: &str| set.get(n).expect("a thermo record");
        let y_o2 = 0.233_f64;
        let y_n2 = 1.0 - y_o2;
        let w_o2 = rec("O2").molar_mass;
        let w_n2 = rec("N2").molar_mass;
        let cp_o2 = rec("O2").cp(300.0).expect("cp");
        let cp_n2 = rec("N2").cp(300.0).expect("cp");
        let sum_y_w = y_o2 / w_o2 + y_n2 / w_n2;
        let rho_h = p0 / (R_GAS * 300.0 * sum_y_w);
        let cp_h = y_o2 * cp_o2 / w_o2 + y_n2 * cp_n2 / w_n2;
        let cv_h = cp_h - R_GAS * sum_y_w;
        let g_hand = (rho_h * cp_h + c0 * cpp) / (rho_h * cv_h + c0 * cpp);
        let g = gamma_u(rho_g0, cp_g, cv_g, c0, cpp);
        assert!((g / g_hand - 1.0).abs() <= 1.0e-12, "{g} against {g_hand}");
        assert!((g / 1.31305 - 1.0).abs() <= 1.0e-4, "gamma_u = {g}");
        let (t3, c3) = compressed(300.0, 0.15418, 1.4, 3.0);
        assert!((t3 / (300.0 * 3.0_f64.powf(0.4 / 1.4)) - 1.0).abs() <= 1.0e-14);
        assert!((c3 / (0.15418 * 3.0_f64.powf(1.0 / 1.4)) - 1.0).abs() <= 1.0e-14);
    }

    /// (159.29): the log-log law is exact on a power law between the nodes
    /// and extrapolates on the end segments' exponent, and `new` refuses
    /// a one-node table, non-increasing ratios and every speed that is
    /// not a finite positive burning velocity, naming SPEC-LIT 159.12.
    #[test]
    fn the_table_is_a_power_law_between_nodes() {
        let table = SuTable::new(
            vec![1.0, 3.0, 6.0],
            vec![0.5, 0.5 * 3.0_f64.powf(0.3), 0.5 * 6.0_f64.powf(0.3)],
        )
        .expect("the table");
        for x in [2.0_f64, 4.5, 10.0] {
            let want = 0.5 * x.powf(0.3);
            assert!((table.s_u(x) / want - 1.0).abs() <= 1.0e-12, "s_u({x})");
        }
        assert_eq!(table.s_u(3.0), table.s[1], "a node replays itself");
        let refused = |ratio: Vec<f64>, s: Vec<f64>, why: &str| {
            let e = SuTable::new(ratio, s).expect_err(why);
            assert!(
                format!("{e}").contains("159.12"),
                "the refusal names SPEC-LIT 159.12: {e}"
            );
        };
        refused(vec![], vec![], "no nodes");
        refused(vec![1.0], vec![0.5], "one node");
        refused(vec![3.0, 1.0], vec![0.5, 0.6], "falling ratios");
        refused(vec![1.0, 1.0], vec![0.5, 0.6], "flat ratios");
        refused(vec![1.0, 3.0], vec![0.5, 0.0], "a zero speed");
        refused(vec![1.0, 3.0], vec![0.5, -1.0], "a negative speed");
        refused(vec![1.0, 3.0], vec![0.5, f64::NAN], "a NaN speed");
        refused(vec![1.0, 3.0], vec![0.5, f64::INFINITY], "an infinite speed");
    }

    /// (159.30) is vessel0d's own integral: from the (157.11) anchor of
    /// x_b = 1e-3 at t_s = 0.01, the vessel0d RK4 trace's `t_half` and
    /// the gate's composite-Simpson `t_half,ref` agree to 1e-8 relative.
    #[test]
    fn the_reference_half_time_is_vessel0d_s() {
        let v = Vessel::new(0.1684, 101_325.0, 9.41167414637 * 101_325.0, 1.40194569106215)
            .expect("the vessel");
        let table = SuTable::new(
            vec![1.0, 3.0, 6.0],
            vec![0.5, 0.5 * 3.0_f64.powf(0.3), 0.5 * 6.0_f64.powf(0.3)],
        )
        .expect("the table");
        let law = BurningLaw { s0: 0.5, alpha: 0.0, beta: 0.3 };
        let p_s = 101_325.0 + X_ANCHOR * (v.pe - 101_325.0);
        let t_s = 0.01;
        let tr = integrate(&v, &law, UnburnedDensity::Compressed, t_s, p_s, 1.0e-6)
            .expect("the trace");
        let m = metrics(&v, &law, UnburnedDensity::Compressed, &tr).expect("the metrics");
        let t_ref = t_half_ref(&v, &table, t_s, p_s);
        assert!(
            (m.t_half / t_ref - 1.0).abs() <= 1.0e-8,
            "vessel0d t_half {} against the gate's {t_ref}",
            m.t_half
        );
    }

    /// A recorded history replays its own half time: the vessel0d trace's
    /// (t, p) as a history, `cross_up` at p_half lands on `t_half,ref`
    /// within 1e-6 relative - the linear interpolation costs nothing at a
    /// recorded step.
    #[test]
    fn a_trace_replays_its_own_half_time() {
        let v = Vessel::new(0.1684, 101_325.0, 9.41167414637 * 101_325.0, 1.40194569106215)
            .expect("the vessel");
        let table = SuTable::new(
            vec![1.0, 3.0, 6.0],
            vec![0.5, 0.5 * 3.0_f64.powf(0.3), 0.5 * 6.0_f64.powf(0.3)],
        )
        .expect("the table");
        let law = BurningLaw { s0: 0.5, alpha: 0.0, beta: 0.3 };
        let p_s = 101_325.0 + X_ANCHOR * (v.pe - 101_325.0);
        let t_s = 0.01;
        let tr = integrate(&v, &law, UnburnedDensity::Compressed, t_s, p_s, 1.0e-6)
            .expect("the trace");
        let t_ref = t_half_ref(&v, &table, t_s, p_s);
        let t_run = cross_up(&tr.t, &tr.p, 0.5 * (v.p0 + v.pe)).expect("a crossing");
        assert!(
            (t_run / t_ref - 1.0).abs() <= 1.0e-6,
            "cross_up t_half {t_run} against {t_ref}"
        );
        assert!(cross_up(&tr.t, &tr.p, 2.0 * v.pe).is_none(), "no crossing, no time");
    }

    /// (159.23) on an exact ramp: a front at x_f = 0.024 - 0.07 t over the
    /// tube's own length gives S_u = 0.07 m/s to round-off, and below four
    /// window fronts there is no speed.
    #[test]
    fn the_tube_speed_of_a_ramp_is_exact() {
        let fronts: Vec<(f64, f64)> = (0..=60)
            .map(|k| {
                let t = 0.005 * k as f64; // 0, 5 ms, ..., 300 ms
                (t, 0.024 - 0.07 * t)
            })
            .collect();
        let s = tube_speed(&fronts, L_TUBE).expect("a speed");
        assert!((s / 0.07 - 1.0).abs() <= 1.0e-12, "S_u {s}");
        assert!(tube_speed(&fronts[..3], L_TUBE).is_none(), "3 fronts, no speed");
    }

    /// Both shipped records replay: the build fields, the case pins
    /// against the shipped cases' own FNV-1a, three tubes at p/p0 = 1, 3,
    /// 6 and a sphere history whose t increases. Each record's sphere end
    /// line and tube speeds are printed - they are what the gate's rows
    /// decide from.
    #[test]
    fn the_records_replay() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let sphere_bytes = std::fs::read(root.join(CASE)).expect("the sphere case reads");
        let tube_bytes = std::fs::read(root.join(TUBE)).expect("the tube case reads");
        let sphere_hash = format!("0x{:x}", fnv1a64(&sphere_bytes));
        let tube_hash = format!("0x{:x}", fnv1a64(&tube_bytes));
        for (rel, build) in [(RECORD_F64, "f64"), (RECORD_SINGLE, "single")] {
            let text = std::fs::read_to_string(root.join(rel)).expect("the record reads");
            let rec: serde_json::Value = serde_json::from_str(&text).expect("the record parses");
            assert_eq!(rec["build"].as_str().expect("a build"), build, "{rel}");
            assert_eq!(
                rec["sphere"]["case_fnv1a"].as_str().expect("a hash"),
                sphere_hash,
                "{rel}: the sphere case's pin"
            );
            let hist = rec["sphere"]["history"].as_array().expect("a history");
            assert!(!hist.is_empty(), "{rel}: the sphere history is not empty");
            let mut last = -1.0;
            for h in hist {
                let t = h[0].as_f64().expect("a time");
                assert!(t > last, "{rel}: the history's t increases");
                last = t;
            }
            println!("  {rel} sphere end: {}", rec["sphere"]["end"].as_str().expect("an end"));
            let tubes = rec["tubes"].as_array().expect("tubes");
            assert_eq!(tubes.len(), 3, "{rel}: three tubes");
            for r in P_RATIOS {
                let e = tubes
                    .iter()
                    .find(|t| t["p_ratio"].as_f64().is_some_and(|p| (p - r).abs() <= 1.0e-9))
                    .unwrap_or_else(|| panic!("{rel}: no p/p0 = {r} tube"));
                assert_eq!(
                    e["case_fnv1a"].as_str().expect("a hash"),
                    tube_hash,
                    "{rel}: p/p0 = {r}'s pin"
                );
                let fronts: Vec<(f64, f64)> = fronts_of(e);
                let s = tube_speed(&fronts, 0.0252);
                println!(
                    "  {rel} tube p/p0 = {r}: {} fronts, S_u = {}",
                    fronts.len(),
                    s.map_or("none".to_string(), |s| format!("{s:.6}"))
                );
            }
        }
    }
}
