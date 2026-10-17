// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate 131-P: the fully developed turbulent pipe with kOmegaSST against
//! TG0's four bands (SPEC-LIT §143.5, docs/17 VAL-23).**
//!
//! TG0's periodic pipe (§114.1) runs as `cases/pipeTurb576.jsonc` and
//! `cases/pipeTurb2358.jsonc` on `ofgpu-lowmach`, on the true wedge of §119
//! (BC-04/05) with the `k` and `omega` solver tolerance 1e-12 of §114.5's O1,
//! at three radial levels. The bands are TG0's, unchanged: `f` within 5 % of
//! Prandtl's smooth-pipe law at the input `Re_tau`, `f` within 5 % of Blasius
//! at the measured `Re_D`, `u+` within 1.0 of `2.5 ln y+ + 5.5` for
//! `30 <= y+ <= 0.2 Re_tau`, and the core defect within 10 % of 4.07. The
//! references are Nikuradse's measured smooth pipes as Schlichting reduces
//! them, so a result outside its band is reported MISSES; within `u_val` of
//! the band edge it is OPEN (undecided), §32.4 and §110.1.
//!
//! The runs are long, so `ofgpu-validate` does not make them: each is an
//! `ofgpu-lowmach` run of a level copy of a case file, and the ignored test
//! `record_pipe` of this binary reads its written fields and prints a
//! [`PipeRecord`] literal (§110.1's split of recipe and verdict). The metrics
//! are eq 143.13 and eq 143.17, evaluated by `vv::rans1d` on the device's own
//! radial cells, so the device and the 1-D oracle of §143.1 are compared like
//! for like; that comparison is a row of its own. Every reduction is f64
//! whatever `Scalar` is (§118.2).
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §143.5 (this gate), §143.1-§143.2 (the oracle and
//!     the band metrics), §114 (TG0 and its O1), §119 (the wedge), §94 (the
//!     mesh study), §110.1 (the replay shape) and §32.4 (the band rule)
//!   Schlichting, NACA TM 1218 (1949), for the four band references, as
//!     §143.2 cites them
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{graded_nodes, GradedAxis};
use ofgpu::vv::{self, rans1d, GridStudy, Level};
use ofgpu::{Result, Scalar};

use super::t3a::band_110;
use crate::{Band, Checks, GateReport, How, Uncertainty, Verdict};

// ---------------------------------------------------------------------------
//  The rig
// ---------------------------------------------------------------------------

/// TG0's pipe radius, viscosity and periodic length (§114.1, §143.1).
pub(crate) const R: f64 = rans1d::TG0_R;
pub(crate) const NU: f64 = rans1d::TG0_NU;
pub(crate) const LENGTH: f64 = 0.1;
pub(crate) const N_AXIAL: usize = 4;
/// The two input `Re_tau` of TG0.
pub(crate) const RE_TAU: [f64; 2] = [576.69, rans1d::TG0_RE_TAU_MAX];
/// The three radial levels, finest first (§143.2's levels: `n = 40 2^l`).
pub(crate) const LEVELS: [u32; 3] = [2, 1, 0];
/// TG0's bars (§114.1): Prandtl and Blasius 5 %, the log law 1.0 in `u+`,
/// the defect 10 % of Schlichting's 4.07.
pub(crate) const BAR_F: f64 = 0.05;
pub(crate) const BAR_LOG: f64 = 1.0;
pub(crate) const BAR_DEFECT: f64 = 0.10;
pub(crate) const DEFECT_REF: f64 = 4.07;
/// The device-against-oracle row: `U_b` and `f` within this relative
/// difference of `rans1d` on the same cells (§143.5).
pub(crate) const ORACLE_TOL: f64 = 1e-4;

fn w(x: Scalar) -> f64 {
    f64::from(x)
}

/// Radial cells of level `l`.
pub(crate) fn n_radial(level: u32) -> usize {
    40usize << level
}

/// The wall cell of level `l`, `(R / 2358) 2^-l` (§143.2).
pub(crate) fn h1(level: u32) -> f64 {
    R / rans1d::TG0_RE_TAU_MAX * 2.0f64.powi(-(level as i32))
}

/// `mesh.grading.x.expansion` of level `l`: the last (wall) cell over the
/// first (axis) cell, `q^-(n-1)` with `q` from eq 143.2.
pub(crate) fn expansion(level: u32) -> Result<f64> {
    let n = n_radial(level);
    let q = rans1d::solve_q(h1(level), R, n)?;
    Ok(q.powi(-((n - 1) as i32)))
}

/// `u_tau,in = Re_tau nu / R` and the drive `g = 2 u_tau,in^2 / R`, eq 143.5.
pub(crate) fn u_tau_in(re_tau: f64) -> f64 {
    re_tau * NU / R
}

pub(crate) fn body_force(re_tau: f64) -> f64 {
    2.0 * u_tau_in(re_tau).powi(2) / R
}

/// The case's start state: `U_b` from Prandtl's law, and TG0's `k` and
/// `omega` from `case_writer.py`'s `turb_refs` at `I = 1 %`, `l = 3 mm`.
pub(crate) fn initial_state(re_tau: f64) -> (f64, f64, f64) {
    let u_b = u_tau_in(re_tau) * (8.0 / rans1d::f_prandtl(re_tau)).sqrt();
    let k = 1.5 * (0.01 * u_b).powi(2);
    (u_b, k, k.sqrt() / (0.09f64.powf(0.25) * 0.003))
}

/// The oracle's pipe on the device's radial cells: `blockgen`'s graded nodes
/// in `Scalar`, widened once, the axis entry exactly 0. With `theta_deg = 0`
/// (143.1)'s `rho` is the radius itself, which is §119.0's tangent wedge:
/// its radial faces are the planes `x = r_i` and its wall is at `x = R`.
pub(crate) fn pipe_mesh(n: usize, expansion: f64) -> rans1d::PipeMesh {
    let axis = GradedAxis { lo: 0.0, hi: R as Scalar, n, expansion: expansion as Scalar, two_sided: false };
    let mut faces: Vec<f64> = graded_nodes(&axis).into_iter().map(w).collect();
    faces[0] = 0.0;
    let mut volumes = vec![0.0f64; n];
    let mut centres = vec![0.0f64; n];
    for j in 0..n {
        let (a, b) = (faces[j], faces[j + 1]);
        volumes[j] = (b * b - a * a) / 2.0;
        centres[j] = 2.0 * (b * b * b - a * a * a) / (3.0 * (b * b - a * a));
    }
    rans1d::PipeMesh { r: R, theta_deg: 0.0, faces, centres, volumes }
}

/// One recorded `ofgpu-lowmach` run: one `Re_tau` on one level.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PipeRecord {
    pub(crate) re_tau_in: f64,
    pub(crate) level: u32,
    pub(crate) n_radial: usize,
    /// The case file's `mesh.grading.x.expansion`, as the run read it.
    pub(crate) expansion: f64,
    pub(crate) iterations: usize,
    /// eq 143.13 on the station-averaged profile.
    pub(crate) u_b: f64,
    pub(crate) u_tau: f64,
    pub(crate) u_axis: f64,
    pub(crate) f: f64,
    pub(crate) defect: f64,
    pub(crate) nut_axis: f64,
    /// eq 143.17: `Re_D`, and the log-law deviation over its cells.
    pub(crate) re_d: f64,
    pub(crate) loglaw_dev: f64,
    pub(crate) loglaw_cells: usize,
    /// The largest change of a cell's `U_z` between the last two writes,
    /// 1000 iterations apart, over `U_axis`.
    pub(crate) change: f64,
    /// The largest difference of a cell's `U_z` from its radial station's
    /// mean over the four axial stations, over `U_axis` (x-invariance).
    pub(crate) spread: f64,
    pub(crate) seconds: f64,
    /// `"f64"` or `"single"`, and the commit the binary was built at.
    pub(crate) build: &'static str,
    pub(crate) commit: &'static str,
}

/// The four band errors of one record, eq 143.17: `f` against Prandtl at the
/// input `Re_tau`, `f` against Blasius at the measured `Re_D`, the log-law
/// deviation (absolute, in `u+`), and the defect against 4.07.
pub(crate) fn band_errors(r: &PipeRecord) -> [f64; 4] {
    [
        r.f / rans1d::f_prandtl(r.re_tau_in) - 1.0,
        r.f / rans1d::f_blasius(r.re_d) - 1.0,
        r.loglaw_dev,
        r.defect / DEFECT_REF - 1.0,
    ]
}

/// The oracle (§143.1, kOmegaSST, Tucker's distance) on the record's cells:
/// `(U_b, f)`.
pub(crate) fn oracle(r: &PipeRecord) -> Result<(f64, f64)> {
    let mesh = pipe_mesh(r.n_radial, r.expansion);
    let case = rans1d::PipeCase::tg0(r.re_tau_in);
    let model = rans1d::Model::KOmegaSst;
    let sol = rans1d::solve(&mesh, &case, model, &rans1d::Controls::for_model(model))?;
    let m = rans1d::metrics(&mesh, &case, &sol);
    Ok((m.u_b, m.f))
}

// ---------------------------------------------------------------------------
//  The records (SPEC-LIT 143.5's runs), finest level first
// ---------------------------------------------------------------------------

pub(crate) const RECORDS: [[Option<&PipeRecord>; 3]; 2] = [
    [Some(&F64_576_L2), Some(&F64_576_L1), Some(&F64_576_L0)],
    [Some(&F64_2358_L2), Some(&F64_2358_L1), Some(&F64_2358_L0)],
];

const F64_576_L2: PipeRecord = PipeRecord {
    re_tau_in: 576.69,
    level: 2,
    n_radial: 160,
    expansion: 0.00293619304606138,
    iterations: 80000,
    u_b: 6.00578113747351,
    u_tau: 0.34970461677669623,
    u_axis: 7.269483183508223,
    f: 0.02712399874018929,
    defect: 3.6136270023613952,
    nut_axis: 0.12402123940987148,
    re_d: 19807.985281904716,
    loglaw_dev: 1.1768658813286663,
    loglaw_cells: 35,
    change: 1.376e-6,
    spread: 0.000e0,
    seconds: 1716.8,
    build: "f64",
    commit: "f9bcb4a",
};

const F64_576_L1: PipeRecord = PipeRecord {
    re_tau_in: 576.69,
    level: 1,
    n_radial: 80,
    expansion: 0.00297925479726894,
    iterations: 24000,
    u_b: 6.027164696991512,
    u_tau: 0.3497044369562466,
    u_axis: 7.291003362451531,
    f: 0.02693184795488639,
    defect: 3.6140195316370693,
    nut_axis: 0.12380054533141853,
    re_d: 19878.511533613164,
    loglaw_dev: 1.072036681716991,
    loglaw_cells: 17,
    change: 0.000e0,
    spread: 0.000e0,
    seconds: 358.3,
    build: "f64",
    commit: "f9bcb4a",
};

const F64_576_L0: PipeRecord = PipeRecord {
    re_tau_in: 576.69,
    level: 0,
    n_radial: 40,
    expansion: 0.00306719135840698,
    iterations: 12000,
    u_b: 6.079580788316029,
    u_tau: 0.34970489425335344,
    u_axis: 7.344056761735615,
    f: 0.026469524521919818,
    defect: 3.615837222179399,
    nut_axis: 0.12314154221931382,
    re_d: 20051.387824261314,
    loglaw_dev: 0.9931327679478272,
    loglaw_cells: 9,
    change: 0.000e0,
    spread: 0.000e0,
    seconds: 114.2,
    build: "f64",
    commit: "f9bcb4a",
};

const F64_2358_L2: PipeRecord = PipeRecord {
    re_tau_in: 2358.0,
    level: 2,
    n_radial: 160,
    expansion: 0.00293619304606138,
    iterations: 80000,
    u_b: 30.0052451108398,
    u_tau: 1.4298942282880676,
    u_axis: 34.75466290703211,
    f: 0.018167846656385767,
    defect: 3.32151686623599,
    nut_axis: 0.13070812952631014,
    re_d: 98961.89020725529,
    loglaw_dev: 1.230196349773676,
    loglaw_cells: 70,
    change: 2.877e-6,
    spread: 0.000e0,
    seconds: 1558.3,
    build: "f64",
    commit: "f9bcb4a",
};

const F64_2358_L1: PipeRecord = PipeRecord {
    re_tau_in: 2358.0,
    level: 1,
    n_radial: 80,
    expansion: 0.00297925479726894,
    iterations: 32000,
    u_b: 30.287037565841718,
    u_tau: 1.4298908679756834,
    u_axis: 35.03751202297345,
    f: 0.017831266038356764,
    defect: 3.3222636520904896,
    nut_axis: 0.13048380416901362,
    re_d: 99891.28484776292,
    loglaw_dev: 1.0488574102938966,
    loglaw_cells: 35,
    change: 2.854e-7,
    spread: 0.000e0,
    seconds: 393.7,
    build: "f64",
    commit: "f9bcb4a",
};

const F64_2358_L0: PipeRecord = PipeRecord {
    re_tau_in: 2358.0,
    level: 0,
    n_radial: 40,
    expansion: 0.00306719135840698,
    iterations: 16000,
    u_b: 30.818479733040643,
    u_tau: 1.4298911439191933,
    u_axis: 35.570602326216516,
    f: 0.017221600765797252,
    defect: 3.3234156413828573,
    nut_axis: 0.12982163068106284,
    re_d: 101644.06244406545,
    loglaw_dev: 0.6302666159695054,
    loglaw_cells: 17,
    change: 0.000e0,
    spread: 1.406e-7,
    seconds: 131.7,
    build: "f64",
    commit: "f9bcb4a",
};

/// The single build's runs: the finest level only, which is what the class-R
/// row reads.
pub(crate) const RECORDS_SINGLE: [[Option<&PipeRecord>; 3]; 2] =
    [[Some(&SINGLE_576_L2), None, None], [Some(&SINGLE_2358_L2), None, None]];

const SINGLE_576_L2: PipeRecord = PipeRecord {
    re_tau_in: 576.69,
    level: 2,
    n_radial: 160,
    expansion: 0.0029361930210143328,
    iterations: 80000,
    u_b: 6.00579156261169,
    u_tau: 0.34959917659158607,
    u_axis: 7.269463343786775,
    f: 0.027107550667167787,
    defect: 3.6146303131925004,
    nut_axis: 0.12406207605849465,
    re_d: 19808.01996076833,
    loglaw_dev: 1.1720526568597123,
    loglaw_cells: 35,
    change: 1.377e-6,
    spread: 1.341e-5,
    seconds: 1241.2,
    build: "single",
    commit: "f9bcb4a",
};

const SINGLE_2358_L2: PipeRecord = PipeRecord {
    re_tau_in: 2358.0,
    level: 2,
    n_radial: 160,
    expansion: 0.0029361930210143328,
    iterations: 80000,
    u_b: 30.00404278231704,
    u_tau: 1.4293700812510166,
    u_axis: 34.75345400297017,
    f: 0.018155984772233482,
    defect: 3.322730259259619,
    nut_axis: 0.13075550061518837,
    re_d: 98957.92621837769,
    loglaw_dev: 1.2251361615797922,
    loglaw_cells: 70,
    change: 2.964e-6,
    spread: 7.203e-6,
    seconds: 1622.3,
    build: "single",
    commit: "f9bcb4a",
};

// ---------------------------------------------------------------------------
//  The replay
// ---------------------------------------------------------------------------

const GATES: [[&str; 4]; 2] = [
    [
        "SPEC-LIT 143.5 Gate 131-P turbulent pipe, kOmegaSST, Re_tau 576.69, f against Prandtl",
        "SPEC-LIT 143.5 Gate 131-P turbulent pipe, kOmegaSST, Re_tau 576.69, f against Blasius",
        "SPEC-LIT 143.5 Gate 131-P turbulent pipe, kOmegaSST, Re_tau 576.69, the log law",
        "SPEC-LIT 143.5 Gate 131-P turbulent pipe, kOmegaSST, Re_tau 576.69, the core defect",
    ],
    [
        "SPEC-LIT 143.5 Gate 131-P turbulent pipe, kOmegaSST, Re_tau 2358, f against Prandtl",
        "SPEC-LIT 143.5 Gate 131-P turbulent pipe, kOmegaSST, Re_tau 2358, f against Blasius",
        "SPEC-LIT 143.5 Gate 131-P turbulent pipe, kOmegaSST, Re_tau 2358, the log law",
        "SPEC-LIT 143.5 Gate 131-P turbulent pipe, kOmegaSST, Re_tau 2358, the core defect",
    ],
];
const AGAINST: [&str; 4] = [
    "Prandtl's smooth-pipe law at the input Re_tau (Nikuradse's measured pipes, Schlichting NACA TM 1218 eq 16.26)",
    "Blasius at the measured Re_D (Schlichting NACA TM 1218 eq 16.4)",
    "the log law u+ = 2.5 ln y+ + 5.5 over 30 <= y+ <= 0.2 Re_tau (Schlichting NACA TM 1218 eq 16.14)",
    "Nikuradse's core defect (U_axis - U_b)/u_tau = 4.07 (Schlichting NACA TM 1218 eq 16.22)",
];
const WHAT: [&str; 4] = ["f against Prandtl", "f against Blasius", "log-law deviation", "defect against 4.07"];
const BARS: [f64; 4] = [BAR_F, BAR_F, BAR_LOG, BAR_DEFECT];

/// The printed size of a band error: per cent, except the log law (`u+`).
fn shown(row: usize, e: f64) -> String {
    if row == 2 {
        format!("{e:.3} in u+")
    } else {
        format!("{:+.2} %", 100.0 * e)
    }
}

/// §94's study of one band error over the three levels, `h = R / n`.
fn study_of(recs: &[Option<&PipeRecord>; 3], row: usize) -> std::result::Result<GridStudy, String> {
    if !recs.iter().all(Option::is_some) {
        return Err("fewer than three levels are recorded".to_string());
    }
    let levels: Vec<Level> = recs
        .iter()
        .flatten()
        .map(|r| Level { h: (R / r.n_radial as f64) as Scalar, value: band_errors(r)[row] as Scalar })
        .collect();
    vv::grid_study(&levels).map_err(|e| e.to_string())
}

/// The section: [`check_pipe_on`] on the recorded runs.
pub(crate) fn check_pipe(c: &mut Checks) -> Result<()> {
    check_pipe_on(c, RECORDS, RECORDS_SINGLE)
}

pub(crate) fn check_pipe_on(
    c: &mut Checks,
    records: [[Option<&PipeRecord>; 3]; 2],
    single: [[Option<&PipeRecord>; 3]; 2],
) -> Result<()> {
    c.note(&format!(
        "143.5: TG0's pipe on the 119.0 wedge, R {R} m, L {LENGTH} m, {N_AXIAL} axial cells, 5 deg, nu {NU:e} \
         m^2/s; levels n = 160 / 80 / 40, wall cell (R/2358) 2^-l; drive g = {:.9} / {:.9} m/s^2",
        body_force(RE_TAU[0]),
        body_force(RE_TAU[1])
    ));
    for re_tau in RE_TAU {
        let (u_b, k, omega) = initial_state(re_tau);
        c.note(&format!(
            "143.5: Re_tau {re_tau} starts from U_z {u_b:.8} m/s (Prandtl's U_b), k {k:.9e} m^2/s^2, omega \
             {omega:.9e} 1/s (TG0's turb_refs)"
        ));
    }
    for (ri, re_tau) in RE_TAU.into_iter().enumerate() {
        let recs = records[ri];
        let Some(fine) = recs[0] else {
            for row in 0..4 {
                c.enter_gate(GATES[ri][row]);
                c.report(GateReport {
                    verdict: Verdict::Open,
                    how: How::Replayed,
                    gate: GATES[ri][row],
                    against: AGAINST[row],
                    headline: format!(
                        "no kOmegaSST run at Re_tau {re_tau} is recorded on the finest level - NOT RUN; \
                         SPEC-LIT 143.5's recipe makes it"
                    ),
                    detail: vec![],
                    uncertainty: Some(Uncertainty::SingleMesh("no run recorded (SPEC-LIT 143.5)")),
                });
            }
            c.leave_gate();
            continue;
        };
        c.enter_gate(GATES[ri][0]);
        for r in recs.iter().chain(single[ri].iter()).flatten() {
            let e = band_errors(r);
            c.note(&format!(
                "143.5 Re_tau {} level {} ({} x {N_AXIAL}): {} iterations, change {:.2e}, spread {:.2e}, \
                 u_tau/u_tau,in - 1 = {:+.2e}; U_b {:.6} U_axis {:.6} f {:.7} defect {:.4} Re_D {:.0} \
                 nu_t,axis/(u_tau R) {:.4}; \
                 Prandtl {}, Blasius {}, log law {} over {} cells, defect {}; {:.0} s, {} build at {}",
                r.re_tau_in,
                r.level,
                r.n_radial,
                r.iterations,
                r.change,
                r.spread,
                r.u_tau / u_tau_in(r.re_tau_in) - 1.0,
                r.u_b,
                r.u_axis,
                r.f,
                r.defect,
                r.re_d,
                r.nut_axis,
                shown(0, e[0]),
                shown(1, e[1]),
                shown(2, e[2]),
                r.loglaw_cells,
                shown(3, e[3]),
                r.seconds,
                r.build,
                r.commit
            ));
        }
        // Each record is the level its slot says, on that level's cells. The
        // expansion is the case file's 15 digits as the run read them, in
        // `Scalar` (a single run's is f32-rounded), so it is held to 1e-6.
        let levels_ok = recs.iter().chain(single[ri].iter()).enumerate().all(|(j, r)| {
            r.map_or(true, |r| {
                r.level == LEVELS[j % 3]
                    && r.n_radial == n_radial(r.level)
                    && expansion(r.level).is_ok_and(|e| (r.expansion / e - 1.0).abs() <= 1e-6)
                    && r.re_tau_in == re_tau
            })
        });
        let label = format!("143.5 Re_tau {re_tau}: every record is its slot's level and Re_tau");
        c.replaying(|c| c.require(&label, levels_ok));
        // The device against the oracle on the same cells (class P): what
        // separates them is how far the run is from its fixed point.
        for r in recs.iter().flatten() {
            let (ub, f) = oracle(r)?;
            let d = ((r.u_b / ub - 1.0).abs()).max((r.f / f - 1.0).abs());
            let label = format!(
                "143.5 Re_tau {} level {}: device U_b {:.6} and f {:.7} against the 143.1 oracle's {:.6} and \
                 {:.7} on the same cells, relative (bar {ORACLE_TOL:e})",
                r.re_tau_in, r.level, r.u_b, r.f, ub, f
            );
            if d > ORACLE_TOL {
                c.note(&format!("143.5 MISSED: {label}: {d:.3e}"));
            }
            c.replaying(|c| c.check(&label, d as Scalar, ORACLE_TOL as Scalar));
        }
        c.leave_gate();
        let e_fine = band_errors(fine);
        let mut bands = Vec::new();
        for row in 0..4 {
            let gate = GATES[ri][row];
            c.enter_gate(gate);
            let study = study_of(&recs, row);
            match &study {
                Ok(st) => c.note(&format!("143.5 Re_tau {re_tau} {}, study: {}", WHAT[row], st.one_line())),
                Err(why) => c.note(&format!("143.5 Re_tau {re_tau} {}, study refused: {why}", WHAT[row])),
            }
            let u_val = study.as_ref().map_or(0.0, |st| w(st.u_fine).abs());
            let e = e_fine[row];
            let b = band_110(e.abs(), BARS[row], u_val);
            bands.push((b, u_val));
            let label = format!(
                "143.5 Re_tau {re_tau}: |E| + u_val of the {}, finest level ({})",
                WHAT[row],
                if row == 2 { "1.0 in u+".to_string() } else { format!("{:.0} %", 100.0 * BARS[row]) }
            );
            if b == Band::Inside {
                c.replaying(|c| c.check(&label, (e.abs() + u_val) as Scalar, BARS[row] as Scalar));
                continue;
            }
            let headline = format!(
                "kOmegaSST at Re_tau {re_tau}, {}: E = {} on the finest level (n = {}), bar {}, u_val {}",
                WHAT[row],
                shown(row, e),
                fine.n_radial,
                if row == 2 { "1.0 in u+".to_string() } else { format!("{:.0} %", 100.0 * BARS[row]) },
                if row == 2 { format!("{u_val:.3}") } else { format!("{:.2} %", 100.0 * u_val) },
            );
            c.report(GateReport {
                verdict: if b == Band::Undecided { Verdict::Open } else { Verdict::Misses },
                how: How::Replayed,
                gate,
                against: AGAINST[row],
                headline: if b == Band::Undecided {
                    format!("undecided, within u_val of the band edge (SPEC-LIT 32.4, 110.1): {headline}")
                } else {
                    headline
                },
                detail: vec![],
                uncertainty: Some(match study {
                    Ok(st) => Uncertainty::Study(st),
                    Err(_) => Uncertainty::SingleMesh("the 94 study refused the three levels; the finest alone"),
                }),
            });
        }
        // Class R: the single build's finest record lands in the same band.
        if let Some(s) = single[ri][0] {
            let es = band_errors(s);
            let same = (0..4).all(|row| band_110(es[row].abs(), BARS[row], bands[row].1) == bands[row].0);
            c.replaying(|c| {
                c.require(&format!("143.5 Re_tau {re_tau} single build: each band lands in the f64 build's"), same)
            });
        } else {
            c.note(&format!("143.5 Re_tau {re_tau}: no single-build run is recorded on the finest level"));
        }
        c.leave_gate();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
//  Tests and the recorder
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use ofgpu::io::case_json::read_case_jsonc;
    use ofgpu::io::fields::{read_scalar_field, read_vector_field};

    use super::*;

    fn case_path(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("cases").join(name)
    }

    /// The rig's numbers are the contract's (SPEC-LIT 143.5), and the two
    /// shipped cases state them: level 1's cells and expansion, the drive,
    /// and the start state.
    #[test]
    fn the_rig_and_the_shipped_cases_are_tg0s() {
        let close = |a: f64, b: f64, tol: f64| (a / b - 1.0).abs() <= tol;
        assert!(close(body_force(RE_TAU[0]), 9.78347667, 1e-8), "{}", body_force(RE_TAU[0]));
        assert!(close(body_force(RE_TAU[1]), 163.567108, 1e-8), "{}", body_force(RE_TAU[1]));
        for (l, e) in [(0, 0.00306719135840698), (1, 0.00297925479726894), (2, 0.00293619304606138)] {
            assert!(close(expansion(l).expect("q"), e, 1e-12), "level {l}: {}", expansion(l).unwrap());
        }
        let (ub, k, om) = initial_state(RE_TAU[0]);
        assert!(close(ub, 6.15925098, 1e-8) && close(k, 0.00569045589, 1e-8) && close(om, 45.9083462, 1e-8));
        let (ub, k, om) = initial_state(RE_TAU[1]);
        assert!(close(ub, 30.1313166, 1e-8) && close(k, 0.136184436, 1e-8) && close(om, 224.585574, 1e-8));
        for (name, drive) in [("pipeTurb576.jsonc", "9.78347667"), ("pipeTurb2358.jsonc", "163.567108")] {
            let text = std::fs::read_to_string(case_path(name)).expect("case text");
            assert!(text.contains(drive), "{name} states the drive {drive}");
            let lowered = read_case_jsonc(&case_path(name)).expect("read").lower().expect("lower");
            let x = &lowered.block.x;
            assert_eq!(x.n, n_radial(1), "{name}");
            assert!(close(w(x.expansion), expansion(1).unwrap(), 1e-6), "{name}: {}", x.expansion);
            assert!(close(w(lowered.nu), NU, 1e-6), "{name}");
            let hm = lowered.build_host_mesh().expect("mesh");
            assert_eq!(hm.n_cells, n_radial(1) * N_AXIAL, "{name}");
        }
    }

    /// The oracle on the tangent wedge's cells is §143.1's pipe of radius R:
    /// (143.6) makes the measured u_tau the input one to round-off.
    #[test]
    fn the_oracle_on_the_device_cells_balances_the_drive() {
        let mesh = pipe_mesh(n_radial(0), expansion(0).unwrap());
        assert_eq!(mesh.faces[0], 0.0);
        assert!((mesh.rho_wall() / R - 1.0).abs() < 1e-6);
        let case = rans1d::PipeCase::tg0(RE_TAU[0]);
        let model = rans1d::Model::KOmegaSst;
        let sol = rans1d::solve(&mesh, &case, model, &rans1d::Controls::for_model(model)).expect("solve");
        let m = rans1d::metrics(&mesh, &case, &sol);
        assert!((m.u_tau / u_tau_in(RE_TAU[0]) - 1.0).abs() < 1e-6, "{}", m.u_tau);
        for l in LEVELS {
            for re_tau in RE_TAU {
                let mesh = pipe_mesh(n_radial(l), expansion(l).unwrap());
                let case = rans1d::PipeCase::tg0(re_tau);
                let sol = rans1d::solve(&mesh, &case, model, &rans1d::Controls::for_model(model)).expect("solve");
                let m = rans1d::metrics(&mesh, &case, &sol);
                let b = rans1d::tg0_bands(&mesh, &case, &sol, &m);
                println!(
                    "oracle Re_tau {re_tau} level {l}: U_b {:.7} f {:.8} defect {:.5} Prandtl {:+.3} % Blasius \
                     {:+.3} % log {:.4} defect {:+.2} % sweeps {}",
                    m.u_b,
                    m.f,
                    m.defect,
                    100.0 * b.f_vs_prandtl,
                    100.0 * b.f_vs_blasius,
                    b.loglaw_dev,
                    100.0 * b.defect_vs_4_07,
                    sol.iterations
                );
            }
        }
    }

    fn synthetic(re_tau: f64, level: u32, df: f64, dlog: f64, ddef: f64) -> PipeRecord {
        let n = n_radial(level);
        let ex = expansion(level).unwrap();
        let mesh = pipe_mesh(n, ex);
        let case = rans1d::PipeCase::tg0(re_tau);
        let model = rans1d::Model::KOmegaSst;
        let sol = rans1d::solve(&mesh, &case, model, &rans1d::Controls::for_model(model)).expect("solve");
        let m = rans1d::metrics(&mesh, &case, &sol);
        // A record whose Blasius, log-law and defect errors are set, each
        // converging at second order to its stated value (u_val about 1e-4
        // of it); f and its Prandtl error are the oracle's own.
        let s = 1.0 + 0.002 * 4.0f64.powi(2 - level as i32);
        let (df, dlog, ddef) = (df * s, dlog * s, ddef * s);
        let f = m.f;
        let re_d = (0.3164 / (f / (1.0 + df))).powi(4);
        PipeRecord {
            re_tau_in: re_tau,
            level,
            n_radial: n,
            expansion: ex,
            iterations: 8000,
            u_b: m.u_b,
            u_tau: m.u_tau,
            u_axis: m.u_axis,
            f,
            defect: DEFECT_REF * (1.0 + ddef),
            nut_axis: m.nut_axis_over_utau_r,
            re_d,
            loglaw_dev: dlog,
            loglaw_cells: 10,
            change: 0.0,
            spread: 0.0,
            seconds: 1.0,
            build: "f64",
            commit: "synthetic",
        }
    }

    #[test]
    fn the_replay_holds_inside_the_bands_and_misses_outside_them() {
        // Re_tau 576.69: Blasius +1 %, log 0.5, defect -5 % (inside); f's
        // Prandtl error is the oracle's own, outside or inside as it falls.
        let a: Vec<PipeRecord> = LEVELS.iter().map(|&l| synthetic(RE_TAU[0], l, 0.01, 0.5, -0.05)).collect();
        // Re_tau 2358: log 1.5 and defect -20 % (outside both).
        let b: Vec<PipeRecord> = LEVELS.iter().map(|&l| synthetic(RE_TAU[1], l, 0.01, 1.5, -0.20)).collect();
        let records = [[Some(&a[0]), Some(&a[1]), Some(&a[2])], [Some(&b[0]), Some(&b[1]), Some(&b[2])]];
        let single = [[Some(&a[0]), None, None], [None, None, None]];
        let mut c = Checks::new();
        check_pipe_on(&mut c, records, single).expect("replay");
        assert_eq!(c.failures, 0, "the oracle rows hold on the oracle's own values");
        let got: Vec<&str> = c.gates.iter().map(|g| g.gate).collect();
        assert!(got.contains(&GATES[1][2]) && got.contains(&GATES[1][3]), "{got:?}");
        assert!(!got.contains(&GATES[0][1]) && !got.contains(&GATES[0][2]) && !got.contains(&GATES[0][3]));
        assert!(!got.contains(&GATES[1][1]));
        for g in &c.gates {
            if g.gate == GATES[1][2] || g.gate == GATES[1][3] {
                assert_eq!(g.verdict, Verdict::Misses, "{}", g.gate);
            }
        }
        // Six oracle rows, the bands inside (Re_tau 576.69: 3 or 4, 2358: 2
        // or 3), and one single-build row.
        assert!(c.total >= 6 + 5 + 1, "{}", c.total);
    }

    #[test]
    fn every_gate_is_open_and_not_run_when_no_run_is_recorded() {
        let mut c = Checks::new();
        check_pipe_on(&mut c, [[None; 3]; 2], [[None; 3]; 2]).expect("replay");
        assert_eq!((c.total, c.failures, c.gates.len()), (0, 0, 8));
        assert!(c.gates.iter().all(|g| g.verdict == Verdict::Open && g.headline.contains("NOT RUN")));
    }

    /// The recorded runs of SPEC-LIT 143.5, replayed: the verdicts it records.
    #[test]
    fn the_recorded_runs_replay() {
        let mut c = Checks::new();
        check_pipe(&mut c).expect("replay");
        for g in &c.gates {
            println!("  {} {}: {}", g.verdict.word(), g.gate, g.headline);
        }
        println!("rows {} failures {}", c.total, c.failures);
    }

    /// The written fields of one `ofgpu-lowmach` run of a pipe case:
    /// `OFGPU_PIPE_CASE` names the case file (its output is `<stem>_jsonc`
    /// next to it), `OFGPU_PIPE_META` is `re_tau,level,seconds,build,commit`.
    /// Prints a [`PipeRecord`] literal (SPEC-LIT 143.5's recipe).
    #[test]
    #[ignore = "reads a recorded ofgpu-lowmach run of SPEC-LIT 143.5"]
    fn record_pipe() {
        let case = PathBuf::from(std::env::var("OFGPU_PIPE_CASE").expect("OFGPU_PIPE_CASE"));
        let meta = std::env::var("OFGPU_PIPE_META").expect("OFGPU_PIPE_META");
        let m: Vec<&str> = meta.split(',').collect();
        let (re_tau, level): (f64, u32) = (m[0].parse().unwrap(), m[1].parse().unwrap());
        let lowered = read_case_jsonc(&case).expect("read").lower().expect("lower");
        let hm = lowered.build_host_mesh().expect("mesh");
        let n = lowered.block.x.n;
        assert_eq!(n, n_radial(level));
        let ex = w(lowered.block.x.expansion);
        let mesh = pipe_mesh(n, ex);
        // The device numbers cell (i, k) as i + n k (§119.0); its centroid
        // is the oracle's centre i.
        let g = hm.geom64();
        assert_eq!(hm.n_cells, n * N_AXIAL);
        for c in 0..hm.n_cells {
            let d = (g.c[c].x - mesh.centres[c % n]).abs();
            assert!(d <= 1e-12 * R, "cell {c}: x {} against {}", g.c[c].x, mesh.centres[c % n]);
        }
        let stem = case.file_stem().and_then(|s| s.to_str()).expect("stem");
        let out = case.with_file_name(format!("{stem}_jsonc"));
        let mut steps: Vec<usize> = std::fs::read_dir(&out)
            .expect("output dir")
            .filter_map(|e| e.ok()?.file_name().to_str()?.parse().ok())
            .collect();
        steps.sort_unstable();
        let (last, prev) = (steps[steps.len() - 1], steps[steps.len() - 2]);
        let read_u = |step: usize| -> Vec<f64> {
            read_vector_field(&out.join(step.to_string()).join("U"), hm.n_cells)
                .expect("U")
                .internal
                .iter()
                .map(|v| w(v.z))
                .collect()
        };
        let (u, u_prev) = (read_u(last), read_u(prev));
        let nut: Vec<f64> = read_scalar_field(&out.join(last.to_string()).join("nut"), hm.n_cells)
            .expect("nut")
            .internal
            .iter()
            .map(|&x| w(x))
            .collect();
        let station_mean = |f: &[f64]| -> Vec<f64> {
            (0..n).map(|i| (0..N_AXIAL).map(|k| f[i + n * k]).sum::<f64>() / N_AXIAL as f64).collect()
        };
        let (prof, nut_prof) = (station_mean(&u), station_mean(&nut));
        let sol = rans1d::Solution {
            model: rans1d::Model::KOmegaSst,
            u: prof.clone(),
            nut: nut_prof,
            k: vec![],
            omega: vec![],
            eps_tilde: vec![],
            nu_tilde: vec![],
            iterations: last,
            last_change: 0.0,
        };
        let pc = rans1d::PipeCase::tg0(re_tau);
        let met = rans1d::metrics(&mesh, &pc, &sol);
        let bands = rans1d::tg0_bands(&mesh, &pc, &sol, &met);
        let change = u.iter().zip(&u_prev).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / met.u_axis;
        let spread = (0..hm.n_cells).map(|c| (u[c] - prof[c % n]).abs()).fold(0.0, f64::max) / met.u_axis;
        println!(
            "RECORD PipeRecord {{ re_tau_in: {re_tau:?}, level: {level}, n_radial: {n}, expansion: {ex:?}, \
             iterations: {last}, u_b: {:?}, u_tau: {:?}, u_axis: {:?}, f: {:?}, defect: {:?}, nut_axis: {:?}, \
             re_d: {:?}, loglaw_dev: {:?}, loglaw_cells: {}, change: {change:.3e}, spread: {spread:.3e}, \
             seconds: {}, build: {:?}, commit: {:?} }}",
            met.u_b,
            met.u_tau,
            met.u_axis,
            met.f,
            met.defect,
            met.nut_axis_over_utau_r,
            bands.re_d,
            bands.loglaw_dev,
            bands.loglaw_cells,
            m[2],
            m[3],
            m[4]
        );
    }
}
