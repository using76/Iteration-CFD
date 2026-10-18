// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate CMP-OBL (SPEC-LIT §127.6, §127.6.6): the oblique shock on a 2-D
//! compression ramp at `M_1 = 2`, deflection 10 degrees, live.**
//!
//! The 90 x 60 ramp of §127.5.3 (`compression_ramp`'s default: 30 columns
//! of approach, 60 on the ramp, 60 cells of height, `x_c = 0.5` m,
//! `l_r = H = 1` m, `D = 1/60` m, slip walls) carries air (`gamma = 1.4`,
//! inviscid, `p_1 = p_op = 1e5` Pa, `T_1 = 300` K) at `M_1 = 2` along `x`,
//! marched with SIMPLE (five outer iterations per pseudo step) and the
//! local time step of §125.4 at LTS CFL 1 until no cell's `M_i` moves by more than `1e-5` across a chunk of 200
//! iterations (capped at 10000). The key is §127.2's (127.4), NACA 1135's
//! theta-beta-M weak branch, evaluated live by `compressible::exact`
//! (CMP-06) in `f64` in BOTH builds: `beta = 39.314` degrees,
//! `p_2/p_1 = 1.7066`, `M_2 = 1.6405`. Three rows, class P (the same bar
//! in both builds, docs/17 section 5.1.4): the post-shock `p_2/p_1` and
//! `M_2` within 1 % of the key, averaged over (127.27)'s region between
//! the wall and the shock, and the shock angle within 0.5 degrees, fitted
//! as the least-squares line of the pressure-jump locus column by column.
//! A march that stops on a solver bound (§122.3) has no field: its rows
//! read NaN and the gate's one verdict says why.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §127.6 (the gate table), §127.6.6 (this gate as
//!     built), §127.2's (127.4) via CMP-06's `compressible::exact`, not
//!     read directly - the key is CMP-06's, §127.5.3 (the ramp), §122
//!     (the gas), §125.4 (the local time step), §126.10 (the
//!     characteristic far field) and §127.6.5 (the march this copies)
//! No GPL-licensed source was consulted.

use std::collections::BTreeMap;
use std::time::Instant;

use ofgpu::compressible::bc as cbc;
use ofgpu::compressible::exact::oblique_shock_weak;
use ofgpu::compressible::gatemesh::{compression_ramp, RampSpec};
use ofgpu::compressible::pressure::Retarded;
use ofgpu::compressible::thermo::{PerfectGas, Viscosity};
use ofgpu::compressible::{Compressible, CompressibleControls, CompressibleState};
use ofgpu::field::{GpuScalarField, GpuVectorField};
use ofgpu::field_setup::{setup_scalar_field, setup_vector_field};
use ofgpu::fv::{DivScheme, Limiter};
use ofgpu::io::case::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::io::polymesh::build_host_mesh;
use ofgpu::timescheme::DdtScheme;
use ofgpu::{Error, Gpu, GpuMesh, Result, Scalar, Vec3};

use super::nozzle::{mach_field, max_change, P_OP};
use crate::{Checks, GateReport, How, Uncertainty, Verdict};

/// The gate's scope and registry name.
pub(crate) const GATE: &str = "SPEC-LIT 127.6 Gate CMP-OBL oblique shock";

const AGAINST: &str = "the oblique-shock key (127.4), NACA 1135's theta-beta-M weak \
                       branch evaluated live by compressible::exact (CMP-06)";

/// What the verdict says about the mesh, §94.3.
const ONE_RAMP: &str = "one 90 x 60 ramp, as the row states";

/// The free stream: Mach 2, `T_1 = 300` K, deflection 10 degrees.
pub(crate) const M_1: f64 = 2.0;
const T_1: f64 = 300.0;
const DEFLECTION_DEG: f64 = 10.0;
/// The march: the chunk the field stop is tested on, the stop itself - no
/// cell's `M_i` moves by more than this across a chunk - and the
/// iteration budget (§127.6.5's march, half its cap).
pub(crate) const LTS_CFL: f64 = 1.0;
/// The outer iterations per pseudo step (decision TAKEN, inputs only,
/// §127.6.6): with §127.6.5's one outer iteration the march stops on a
/// `T <= 0` bound at a ramp-wall cell within 23 steps; three hold, and
/// five leave the per-chunk cell Mach change about four times smaller and
/// the outer residuals about twenty times smaller than three do.
pub(crate) const N_OUTER: usize = 5;
pub(crate) const MAX_ITERS: usize = 10000;
pub(crate) const CHUNK: usize = 200;
pub(crate) const FIELD_STOP: f64 = 1.0e-5;
/// The rows' bars, the plan's, fixed before measurement (class P).
const P_BAR: f64 = 0.01;
const BETA_BAR: f64 = 0.5;
const M_BAR: f64 = 0.01;

fn w(x: Scalar) -> f64 {
    f64::from(x)
}

fn s(x: f64) -> Scalar {
    x as Scalar
}

/// The gate's key, §127.2's (127.4) evaluated live by
/// `compressible::exact` (CMP-06): every value in `f64` whatever `Scalar`
/// is, an answer key never being a `Scalar`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ObliqueKey {
    pub gamma: f64,
    pub m1: f64,
    pub deflection_deg: f64,
    /// The shock angle to the free stream, degrees (NACA's `theta`).
    pub beta_deg: f64,
    /// The static pressure ratio `p_2/p_1`, NACA (128).
    pub p2_p1: f64,
    /// The Mach number behind the shock, NACA (132).
    pub m2: f64,
}

impl ObliqueKey {
    /// The weak-branch key at `m1` and `deflection_deg`; a deflection
    /// above the maximum is refused by `exact` as detached.
    pub(crate) fn new(gamma: f64, m1: f64, deflection_deg: f64) -> Result<Self> {
        let sh = oblique_shock_weak(gamma, m1, deflection_deg.to_radians())?;
        Ok(Self {
            gamma,
            m1,
            deflection_deg,
            beta_deg: sh.shock_angle.to_degrees(),
            p2_p1: sh.p2_p1,
            m2: sh.m2,
        })
    }
}

/// The first `k` with `p[k] < p_mid <= p[k+1]`; returns `(k, t)` with
/// `t = (p_mid - p[k]) / (p[k+1] - p[k])`, or None when no pair crosses.
/// The shock locus reads each column from the TOP cell down, so the jump
/// is an upward crossing.
pub(crate) fn locate_jump(p: &[f64], p_mid: f64) -> Option<(usize, f64)> {
    for k in 1..p.len() {
        if p[k - 1] < p_mid && p_mid <= p[k] {
            let t = (p_mid - p[k - 1]) / (p[k] - p[k - 1]);
            return Some((k - 1, t));
        }
    }
    None
}

/// Least squares `y = a + b x` over the points; returns `(a, b)`, or None
/// for fewer than 2 points or a zero spread in `x`.
pub(crate) fn fit_line(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    if x.len() < 2 || x.len() != y.len() {
        return None;
    }
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut cov, mut var) = (0.0f64, 0.0f64);
    for k in 0..x.len() {
        let dx = x[k] - mx;
        cov += dx * (y[k] - my);
        var += dx * dx;
    }
    if var == 0.0 {
        return None;
    }
    let b = cov / var;
    Some((my - b * mx, b))
}

/// (127.27)'s region test for a cell centre at `x' = x_rel`, `y`: the
/// point sits between 0.4 and 0.9 of the ramp from the corner, and its
/// angle from the corner, `atan2(y, x_rel)` in degrees, between five
/// degrees off the wall and five degrees inside the shock - so neither
/// the smeared shock nor the corner nor the outlet enters it.
pub(crate) fn in_region(
    x_rel: f64,
    y: f64,
    ramp_length: f64,
    deflection_deg: f64,
    beta_deg: f64,
) -> bool {
    if !(0.4 * ramp_length..=0.9 * ramp_length).contains(&x_rel) {
        return false;
    }
    let phi = f64::atan2(y, x_rel).to_degrees();
    deflection_deg + 5.0 <= phi && phi <= beta_deg - 5.0
}

fn entry(t: &str) -> PatchFieldSpec {
    PatchFieldSpec { type_name: t.into(), ..PatchFieldSpec::default() }
}

/// A `characteristicFarfield` entry stating its free-stream scalar.
fn ff_s(v: f64) -> PatchFieldSpec {
    PatchFieldSpec {
        type_name: "characteristicFarfield".into(),
        value: vec![s(v)],
        ..PatchFieldSpec::default()
    }
}

/// A `characteristicFarfield` entry stating its free-stream `x` velocity.
fn ff_v(v: f64) -> PatchFieldSpec {
    PatchFieldSpec {
        type_name: "characteristicFarfield".into(),
        value_v: vec![Vec3::new(s(v), s(0.0), s(0.0))],
        ..PatchFieldSpec::default()
    }
}

/// `characteristicFarfield` on the two flow-through patches and
/// `symmetry` on the three slip patches, on one field. The far field
/// takes the free stream on supersonic inflow and the interior on
/// supersonic outflow (§126.10).
fn field_bcs(
    mut b: BTreeMap<String, PatchFieldSpec>,
    ff: PatchFieldSpec,
) -> BTreeMap<String, PatchFieldSpec> {
    b.insert("inlet".into(), ff.clone());
    b.insert("outlet".into(), ff);
    for side in ["lowerWall", "top", "frontAndBack"] {
        b.insert(side.into(), entry("symmetry"));
    }
    b
}

/// One march's field and bookkeeping, after it stopped.
struct Run {
    /// The cell centres' `x` and `y` [m].
    cx: Vec<f64>,
    cy: Vec<f64>,
    /// The cell velocities [m/s], the Mach numbers `|u|/sqrt(gamma r_s T)`
    /// and the ABSOLUTE static pressures [Pa], from the downloaded fields.
    u: Vec<Vec3>,
    mach: Vec<f64>,
    p_abs: Vec<f64>,
    /// The largest Mach number of the field.
    m_max: f64,
    iterations: usize,
    /// The field stop was met.
    converged: bool,
    /// `(iterations, max |change|)` after each chunk.
    history: Vec<(usize, f64)>,
    last_line: String,
    seconds: f64,
}

/// One march of the gate (§127.6.6's setup): the default ramp, the
/// uniform free-stream start, the chunked march to the field stop. A
/// solver bound (`Error::Refused`, §122.3) is the gate's miss; the error
/// carries the iterations reached.
fn run_once(gpu: &Gpu, gas: PerfectGas, r_s: f64) -> Result<Run> {
    let t_start = Instant::now();
    let gamma = gas.gamma;
    let ramp = compression_ramp(&RampSpec::default())?;
    let hm = build_host_mesh(&ramp.raw)?;
    let gm = GpuMesh::upload(gpu, &hm)?;
    let n = hm.n_cells;
    let u_1 = 2.0 * (gamma * r_s * T_1).sqrt();

    // The uniform free stream in every cell: `U_1`, gauge `p = 0` (so
    // `p_1 = P_OP` absolute), `T_1`.
    let mut ru = RawVectorField { name: "U".into(), ..RawVectorField::default() };
    ru.internal = vec![Vec3::new(s(u_1), s(0.0), s(0.0)); n];
    ru.boundary = field_bcs(ru.boundary, ff_v(u_1));
    let mut rp = RawScalarField { name: "p".into(), ..RawScalarField::default() };
    rp.internal = vec![s(0.0); n];
    rp.boundary = field_bcs(rp.boundary, ff_s(0.0));
    let mut rt = RawScalarField { name: "T".into(), ..RawScalarField::default() };
    rt.internal = vec![s(T_1); n];
    rt.boundary = field_bcs(rt.boundary, ff_s(T_1));

    // §127.6.5's controls (CMP-21's march), but N_OUTER outer iterations.
    let mut ctrl = CompressibleControls { gas, ..CompressibleControls::default() };
    let lin = |tolerance: Scalar, rel_tol: Scalar| SolverControls {
        solver: LinearSolverKind::PBiCGStab,
        precon: Preconditioner::Dilu,
        tolerance,
        rel_tol,
        max_iter: 1000,
        ..Default::default()
    };
    ctrl.momentum.u_solver = lin(1.0e-9, 0.01);
    ctrl.pressure.p_solver = lin(1.0e-10, 0.001);
    ctrl.energy.t_solver = lin(1.0e-9, 0.01);
    ctrl.lts.cfl = s(LTS_CFL);
    ctrl.momentum.ddt = DdtScheme::LocalEuler;
    ctrl.energy.ddt = DdtScheme::LocalEuler;
    ctrl.momentum.u_relax = 0.7;
    ctrl.energy.t_relax = 0.7;
    ctrl.p_relax = 0.3;
    ctrl.n_outer = N_OUTER;
    ctrl.residual_control = None;
    ctrl.momentum.div_scheme = DivScheme::Limited(Limiter::VanLeer);
    ctrl.energy.div_scheme = DivScheme::Limited(Limiter::VanLeer);
    ctrl.pressure.retarded = Retarded::Limited(Limiter::VanLeer);

    let low = cbc::lower(&ru, &rp, &rt, &hm, &ctrl.gas)?;
    assert_eq!(low.farfields.len(), 2, "inlet and outlet are far-field patches");
    let mut uf = GpuVectorField::zeros(gpu, &gm, "U")?;
    let mut pf = GpuScalarField::zeros(gpu, &gm, "p")?;
    let mut tf = GpuScalarField::zeros(gpu, &gm, "T")?;
    setup_vector_field(gpu, &mut uf, &low.u, &hm)?;
    setup_scalar_field(gpu, &mut pf, &low.p, &hm)?;
    setup_scalar_field(gpu, &mut tf, &low.t, &hm)?;
    let mut st = CompressibleState::new(gpu, &gm, uf, pf, tf)?;
    let mut solver = Compressible::new(gpu, &gm, ctrl)?;
    solver.set_boundary_rules(gpu, low.rules)?;
    solver.set_outlet_rules(gpu, low.outlets)?;
    // Without the far-field rules the far-field faces stay fixed
    // placeholders (§126.10).
    solver.set_farfield_rules(gpu, low.farfields)?;
    solver.initialise(gpu, &mut st)?;

    // The stop is the field's own: chunks of CHUNK iterations until no
    // cell's `M_i` moves by more than FIELD_STOP across a chunk.
    let machs = |st: &CompressibleState| -> Result<Vec<f64>> {
        Ok(mach_field(&gpu.download(&st.u.f)?, &gpu.download(&st.t.f)?, gamma, r_s))
    };
    let mut prev = machs(&st)?;
    let mut iterations = 0usize;
    let mut change = f64::INFINITY;
    let mut history = Vec::new();
    let mut last_line = "no iteration".to_string();
    while iterations < MAX_ITERS && !(change <= FIELD_STOP) {
        let k = CHUNK.min(MAX_ITERS - iterations);
        // A bound (T, p_abs or rho <= 0, §122.3) stops `run` with
        // Error::Refused: the gate's miss, carrying the iterations.
        let rep = match solver.run(gpu, &mut st, k) {
            Ok(rep) => rep,
            Err(Error::Refused(why)) => {
                return Err(Error::Refused(format!("after {iterations} iterations: {why}")));
            }
            Err(e) => return Err(e),
        };
        iterations += rep.steps;
        if let Some(o) = rep.last.as_ref().and_then(|l| l.outers.last()) {
            last_line = o.line();
        }
        let next = machs(&st)?;
        change = max_change(&prev, &next);
        history.push((iterations, change));
        prev = next;
    }
    let u: Vec<Vec3> = gpu.download(&st.u.f)?;
    let mach = machs(&st)?;
    let p_abs: Vec<f64> = gpu.download(&st.p.f)?.iter().map(|v| w(*v) + P_OP).collect();
    let m_max = mach.iter().copied().fold(0.0f64, |m, v| {
        if v.is_nan() || m.is_nan() { f64::NAN } else { m.max(v) }
    });
    Ok(Run {
        cx: hm.c.iter().map(|c| w(c.x)).collect(),
        cy: hm.c.iter().map(|c| w(c.y)).collect(),
        u,
        m_max,
        mach,
        p_abs,
        iterations,
        converged: change <= FIELD_STOP,
        history,
        last_line,
        seconds: t_start.elapsed().as_secs_f64(),
    })
}

/// The rows' short names, for the per-row notes.
const ROW_NAMES: [&str; 3] = ["post-shock p2/p1", "shock angle", "post-shock M2"];

/// The rows' labels and bars: each label starts `Gate CMP-OBL: `, states
/// its bar in words against the key, and ends `, 90 x 60 cells (SPEC-LIT
/// 127.6)`.
fn row_labels(key: &ObliqueKey) -> [(String, f64); 3] {
    [
        (
            format!(
                "Gate CMP-OBL: post-shock p2/p1 within 1 % of the key's {:.4}, \
                 90 x 60 cells (SPEC-LIT 127.6)",
                key.p2_p1
            ),
            P_BAR,
        ),
        (
            format!(
                "Gate CMP-OBL: shock angle within 0.5 degrees of the key's {:.2} \
                 degrees, 90 x 60 cells (SPEC-LIT 127.6)",
                key.beta_deg
            ),
            BETA_BAR,
        ),
        (
            format!(
                "Gate CMP-OBL: post-shock M2 within 1 % of the key's {:.4}, \
                 90 x 60 cells (SPEC-LIT 127.6)",
                key.m2
            ),
            M_BAR,
        ),
    ]
}

/// The measured diagnostics of one field: (127.27)'s region and its
/// stats, the shock locus and its fit, and the printed-only flow angle
/// and wall pressure.
struct Diag {
    region_count: usize,
    p2_p1: f64,
    m2: f64,
    p_min: f64,
    p_max: f64,
    m_min: f64,
    m_max_region: f64,
    crossings: usize,
    a: f64,
    b: f64,
    beta_fit: f64,
    flow_angle: f64,
    wall_p: f64,
}

impl Diag {
    /// Everything the rows and the notes read, all in `f64` from the
    /// downloaded field.
    fn compute(key: &ObliqueKey, r: &Run) -> Diag {
        let spec = RampSpec::default();
        let (x_c, l_r) = (spec.x_corner, spec.ramp_length);
        let nx = spec.n_up + spec.n_ramp;
        let ny = spec.ny;
        let (mut p_sum, mut m_sum, mut ang_sum) = (0.0f64, 0.0f64, 0.0f64);
        let (mut p_min, mut p_max) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut m_min, mut m_max_region) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut region_count = 0usize;
        for c in 0..r.cx.len() {
            let xr = r.cx[c] - x_c;
            if !in_region(xr, r.cy[c], l_r, key.deflection_deg, key.beta_deg) {
                continue;
            }
            let p = r.p_abs[c] / P_OP;
            region_count += 1;
            p_sum += p;
            m_sum += r.mach[c];
            ang_sum += f64::atan2(w(r.u[c].y), w(r.u[c].x)).to_degrees();
            p_min = p_min.min(p);
            p_max = p_max.max(p);
            m_min = m_min.min(r.mach[c]);
            m_max_region = m_max_region.max(r.mach[c]);
        }
        let n_r = region_count as f64;
        let mut d = Self::locus(key, r, nx, ny);
        d.region_count = region_count;
        d.p2_p1 = p_sum / n_r;
        d.m2 = m_sum / n_r;
        d.p_min = p_min;
        d.p_max = p_max;
        d.m_min = m_min;
        d.m_max_region = m_max_region;
        d.flow_angle = ang_sum / n_r;
        d.wall_p = Self::wall_pressure(&spec, r);
        d
    }

    /// The mean `p/p_1` of the lowerWall cells (`j = 0`) whose centre `x'`
    /// is in `[0.4, 0.9] l_r` (printed only).
    fn wall_pressure(spec: &RampSpec, r: &Run) -> f64 {
        let nx = spec.n_up + spec.n_ramp;
        let (mut sum, mut count) = (0.0f64, 0.0f64);
        for i in 0..nx {
            let xr = r.cx[i] - spec.x_corner;
            if (0.4 * spec.ramp_length..=0.9 * spec.ramp_length).contains(&xr) {
                sum += r.p_abs[i] / P_OP;
                count += 1.0;
            }
        }
        sum / count
    }

    /// The shock locus (§127.6.6): in each cell column whose bottom
    /// cell's centre `x'` is in `[0.2, 0.9] l_r`, the column's `p/p_1`
    /// read from the TOP cell down; its first upward crossing of
    /// `p_mid = (1 + p_2/p_1)/2` gives `(x_s, y_s)` by linear
    /// interpolation between the two cell centres. The fit
    /// `y_s = a + b (x_s - x_c)` over every crossing column gives the
    /// measured shock angle `atan(b)`.
    fn locus(key: &ObliqueKey, r: &Run, nx: usize, ny: usize) -> Diag {
        let spec = RampSpec::default();
        let p_mid = 0.5 * (1.0 + key.p2_p1);
        let mut xs: Vec<f64> = Vec::new();
        let mut ys: Vec<f64> = Vec::new();
        for i in 0..nx {
            let xr = r.cx[i] - spec.x_corner;
            if !(0.2 * spec.ramp_length..=0.9 * spec.ramp_length).contains(&xr) {
                continue;
            }
            let col: Vec<f64> =
                (0..ny).rev().map(|j| r.p_abs[j * nx + i] / P_OP).collect();
            let Some((k, t)) = locate_jump(&col, p_mid) else {
                continue;
            };
            // col[k] is cell (i, ny-1-k), col[k+1] the cell below it.
            let j0 = ny - 1 - k;
            let j1 = j0 - 1;
            let (xa, ya) = (r.cx[j0 * nx + i], r.cy[j0 * nx + i]);
            let (xb, yb) = (r.cx[j1 * nx + i], r.cy[j1 * nx + i]);
            xs.push(xa + t * (xb - xa));
            ys.push(ya + t * (yb - ya));
        }
        let crossings = xs.len();
        let rel: Vec<f64> = xs.iter().map(|x| x - spec.x_corner).collect();
        match fit_line(&rel, &ys) {
            Some((a, b)) => Diag {
                crossings,
                a,
                b,
                beta_fit: b.atan().to_degrees(),
                region_count: 0,
                p2_p1: f64::NAN,
                m2: f64::NAN,
                p_min: f64::NAN,
                p_max: f64::NAN,
                m_min: f64::NAN,
                m_max_region: f64::NAN,
                flow_angle: f64::NAN,
                wall_p: f64::NAN,
            },
            None => Diag {
                crossings,
                a: f64::NAN,
                b: f64::NAN,
                beta_fit: f64::NAN,
                region_count: 0,
                p2_p1: f64::NAN,
                m2: f64::NAN,
                p_min: f64::NAN,
                p_max: f64::NAN,
                m_min: f64::NAN,
                m_max_region: f64::NAN,
                flow_angle: f64::NAN,
                wall_p: f64::NAN,
            },
        }
    }
}

/// The march's notes: the state and per-chunk history, then the region's
/// stats, the locus's fit, and the printed-only flow angle and wall
/// pressure.
fn note_run(c: &mut Checks, key: &ObliqueKey, r: &Run, d: &Diag) {
    c.note(&format!(
        "Gate CMP-OBL: {} iterations, field stop {}, M_max {:.4}, {:.1} s; last: {}",
        r.iterations,
        if r.converged { "met" } else { "NOT met" },
        r.m_max,
        r.seconds,
        r.last_line
    ));
    let hist: Vec<String> = r.history.iter().map(|(i, ch)| format!("{i}: {ch:.2e}")).collect();
    c.note(&format!("Gate CMP-OBL: max |change| per chunk: {}", hist.join(", ")));
    c.note(&format!(
        "Gate CMP-OBL: the region (127.27) holds {} cells; p/p_1 in [{:.6}, {:.6}], \
         M in [{:.6}, {:.6}]",
        d.region_count, d.p_min, d.p_max, d.m_min, d.m_max_region
    ));
    c.note(&format!(
        "Gate CMP-OBL: the locus has {} crossings of p_mid {:.6}; the fit is \
         y = {:.6} + {:.6} (x - x_c), beta {:.6} degrees",
        d.crossings,
        0.5 * (1.0 + key.p2_p1),
        d.a,
        d.b,
        d.beta_fit
    ));
    c.note(&format!(
        "Gate CMP-OBL: printed, never checked: the mean flow angle over the region is \
         {:.4} degrees against the wall's 10; the lowerWall cells' mean p/p_1 over \
         [0.4, 0.9] l_r is {:.6}",
        d.flow_angle, d.wall_p
    ));
    if !r.converged {
        let last = r.history.last().map_or(f64::NAN, |h| h.1);
        c.note(&format!(
            "Gate CMP-OBL: the solve did not converge: the field stop {FIELD_STOP:.0e} is \
             NOT met in {} iterations (the last chunk moved a cell's M by {last:.3e}), so \
             the rows read a field that still moves (SPEC-LIT 127.6.6)",
            r.iterations
        ));
    }
}

/// The rows of a field, in the section's order: every measure in `f64`.
fn measure(key: &ObliqueKey, d: &Diag) -> Vec<(String, f64, f64)> {
    let labels = row_labels(key);
    let err_beta =
        if d.crossings >= 10 { (d.beta_fit - key.beta_deg).abs() } else { f64::NAN };
    vec![
        (labels[0].0.clone(), (d.p2_p1 / key.p2_p1 - 1.0).abs(), labels[0].1),
        (labels[1].0.clone(), err_beta, labels[1].1),
        (labels[2].0.clone(), (d.m2 / key.m2 - 1.0).abs(), labels[2].1),
    ]
}

/// **Gate CMP-OBL** (SPEC-LIT §127.6, §127.6.6): the M 2 free stream over
/// the 10-degree ramp, marched to its fixed point and held against the
/// theta-beta-M key.
pub(crate) fn check_oblique(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    let gas = PerfectGas {
        viscosity: Viscosity::Constant { mu: 0.0 },
        ..PerfectGas::air(P_OP)
    };
    let (gamma, r_s) = (gas.gamma, gas.r_s());
    let key = ObliqueKey::new(gamma, M_1, DEFLECTION_DEG)?;
    c.note(&format!(
        "Gate CMP-OBL: the key (127.4) at M_1 {M_1}, theta {DEFLECTION_DEG} degrees, \
         gamma {gamma}: beta {:.6} degrees, p_2/p_1 {:.6}, M_2 {:.6}",
        key.beta_deg, key.p2_p1, key.m2
    ));
    let mut misses: Vec<String> = Vec::new();
    let rows = match run_once(gpu, gas, r_s) {
        Ok(r) => {
            let d = Diag::compute(&key, &r);
            note_run(c, &key, &r, &d);
            measure(&key, &d)
        }
        Err(Error::Refused(why)) => {
            // The verdict word stays in the registry's report: §69.2 fails
            // any note that prints one, so the notes state the miss
            // without the word (CMP-21's shape).
            c.note(&format!(
                "Gate CMP-OBL: the march at LTS CFL {LTS_CFL}, {N_OUTER} outer \
                 iterations, stopped on a bound, \
                 so there is no field to compare: {why}"
            ));
            for (k, (label, bar)) in row_labels(&key).into_iter().enumerate() {
                c.check_p(&label, Scalar::NAN, s(bar));
                c.note(&format!(
                    "Gate CMP-OBL {}: no field: NaN against the bar {bar}",
                    ROW_NAMES[k]
                ));
                misses.push(format!("{label}: no field, the march stopped on a bound"));
            }
            Vec::new()
        }
        Err(e) => return Err(e),
    };
    for (k, (label, err, bar)) in rows.iter().enumerate() {
        c.check_p(label, s(*err), s(*bar));
        // `!(err <= bar)`, so that a NaN measure counts as the miss it is.
        if !(*err <= *bar) {
            c.note(&format!(
                "Gate CMP-OBL {}: {err:.6e} against the bar {bar}",
                ROW_NAMES[k]
            ));
            misses.push(format!("{label}: {err:.4e} against {bar}"));
        }
    }
    if misses.is_empty() {
        return Ok(());
    }
    c.report(GateReport {
        verdict: Verdict::Misses,
        how: How::Live,
        gate: GATE,
        against: AGAINST,
        headline: misses.join("; "),
        detail: vec![format!(
            "one march at LTS CFL {LTS_CFL} with {N_OUTER} outer iterations, chunks of {CHUNK} iterations to a cell \
             Mach change of {FIELD_STOP:.1e}, {MAX_ITERS} iterations: every measured \
             value is in the rows and notes above"
        )],
        uncertainty: Some(Uncertainty::SingleMesh(ONE_RAMP)),
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close_abs(got: f64, want: f64, tol: f64, what: &str) {
        assert!((got - want).abs() <= tol, "{what}: {got} is not {want}");
    }

    fn close_rel(got: f64, want: f64, tol: f64, what: &str) {
        assert!((got - want).abs() <= tol * want.abs(), "{what}: {got} is not {want}");
    }

    /// The key, from CMP-06's `oblique_shock_weak`: the theta-beta-M
    /// weak root at M 2, 10 degrees, and the detached refusal at 30.
    #[test]
    fn the_oblique_key() {
        let key = ObliqueKey::new(1.4, 2.0, 10.0).expect("the attached shock");
        assert_eq!(key.gamma, 1.4);
        assert_eq!(key.m1, 2.0);
        assert_eq!(key.deflection_deg, 10.0);
        close_abs(key.beta_deg, 39.31393184481887, 1.0e-9, "beta_deg");
        close_rel(key.p2_p1, 1.7065786040000334, 1.0e-12, "p2_p1");
        close_rel(key.m2, 1.6405222290010812, 1.0e-12, "m2");
        assert!(
            ObliqueKey::new(1.4, 2.0, 30.0).is_err(),
            "a 30-degree turn at M 2 is a detached shock"
        );
    }

    /// The locator takes the first upward crossing, interpolates linearly,
    /// and returns None when nothing crosses or there is no pair.
    #[test]
    fn the_jump_locator() {
        let (k, t) = locate_jump(&[1.0, 1.0, 1.2, 1.7, 1.7], 1.35).expect("the crossing");
        assert_eq!(k, 2, "the crossing is between the third and fourth entries");
        close_abs(t, 0.3, 1.0e-14, "t");
        let (k, t) = locate_jump(&[1.7, 1.0, 1.0, 1.7], 1.35).expect("the crossing");
        assert_eq!(k, 2, "the first crossing does not rise");
        close_abs(t, 0.5, 1.0e-14, "t");
        let (k, t) = locate_jump(&[1.0, 1.35], 1.35).expect("the crossing");
        assert_eq!(k, 0);
        close_abs(t, 1.0, 1.0e-14, "t");
        assert_eq!(locate_jump(&[1.0, 1.1, 1.2], 1.35), None, "nothing reaches p_mid");
        assert_eq!(locate_jump(&[], 1.35), None, "an empty column");
        assert_eq!(locate_jump(&[2.0], 1.35), None, "a single entry");
    }

    /// The line fit is the least-squares intercept and slope, and is
    /// refused for a single point or a zero spread in `x`.
    #[test]
    fn the_line_fit() {
        let (a, b) = fit_line(&[0.0, 1.0, 2.0], &[1.0, 3.0, 5.0]).expect("the fit");
        close_abs(a, 1.0, 1.0e-14, "a");
        close_abs(b, 2.0, 1.0e-14, "b");
        let (a, b) =
            fit_line(&[0.0, 1.0, 2.0, 3.0], &[0.0, 1.1, 1.9, 3.0]).expect("the fit");
        close_abs(a, 0.03, 1.0e-13, "a");
        close_abs(b, 0.98, 1.0e-13, "b");
        assert_eq!(fit_line(&[1.0], &[1.0]), None, "one point is no line");
        assert_eq!(fit_line(&[2.0, 2.0, 2.0], &[0.0, 1.0, 2.0]), None, "no spread");
    }

    /// The region selector: five degrees off the wall, five inside the
    /// shock, between 0.4 and 0.9 of the ramp.
    #[test]
    fn the_region_selector() {
        let (l_r, theta, beta) = (1.0, 10.0, 39.31393184481887);
        let tan25 = f64::tan(25.0f64.to_radians());
        assert!(
            in_region(0.6, 0.6 * tan25, l_r, theta, beta),
            "25 degrees off the wall, mid-ramp, is inside"
        );
        assert!(
            !in_region(0.6, 0.6 * f64::tan(12.0f64.to_radians()), l_r, theta, beta),
            "12 degrees is nearer the wall than the 5-degree floor"
        );
        assert!(
            !in_region(0.6, 0.6 * f64::tan(36.0f64.to_radians()), l_r, theta, beta),
            "36 degrees is inside the shock's band"
        );
        assert!(
            !in_region(0.3, 0.3 * tan25, l_r, theta, beta),
            "ahead of 0.4 l_r is the undisturbed approach"
        );
        assert!(
            !in_region(0.95, 0.95 * tan25, l_r, theta, beta),
            "past 0.9 l_r is the outlet's neighbourhood"
        );
    }
}
