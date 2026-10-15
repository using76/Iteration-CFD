// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate CMP-NOZ (SPEC-LIT §127.6, §127.6.5): `ofgpu::compressible` in the
//! quasi-1-D nozzle, against §127.2's isentropic and shocked keys, live.**
//!
//! The 200-cell duct of §127.5.2 (`A_e/A_t = 2`, `L = 1` m, `H_t = D =
//! 0.05` m, slip walls) carries air (`gamma = 1.4`, inviscid, `p_op = 1e5`
//! Pa, `T_0 = 300` K) through three regimes, each from rest at a linear
//! gauge `p` to the regime's back pressure, marched with SIMPLE and the
//! local time step of §125.4 at LTS CFL 1 until no cell's
//! `M_i = |u_i|/sqrt(gamma R_s T_i)` moves by more than `1e-5` across a
//! chunk of 200 iterations (capped at 20000): **subsonic** at
//! `p_b/p_t1 = 0.95`, wholly subsonic; **supersonic started** at
//! `p_b/p_t1 = 0.05`, below the design exit, so the whole diverging side is
//! supersonic; **shocked** at the back pressure of a normal shock standing
//! at `A_s/A_t = 1.5`. Seven rows, class P (the same bar in both builds,
//! docs/17 section 5.1.4): `M(x)` within 0.5 % of the key in the two
//! isentropic regimes, the exit static `p` within 0.5 % in all three, and
//! the shocked regime's shock position and `p_t2/p_t1` within 1 %. The keys
//! (127.23)-(127.25) are §127.2's relations evaluated live by
//! `compressible::exact` (CMP-06), computed in `f64` in BOTH builds. A
//! regime that stops on a solver bound (§122.3) has no field: its rows
//! read NaN and the gate's one verdict says why.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §127.2 ((127.2)-(127.3), the shocked nozzle and
//!     the oracle values), §127.5.2 (the duct), §122 (the gas), §125.4 (the
//!     local time step), §126.4 (the outlets' supersonic switch), §127.6
//!     (the gate table) and §127.6.5 (this gate as built)
//!   NACA 1135 eqs. (43), (44), (80), (93), (99), via CMP-06's
//!     `compressible::exact`; not read directly - the key is CMP-06's
//! No GPL-licensed source was consulted.

use std::collections::BTreeMap;
use std::time::Instant;

use ofgpu::compressible::bc as cbc;
use ofgpu::compressible::exact::{
    area_ratio, mach_from_area_ratio, normal_shock, pressure_ratio, shocked_nozzle_back_pressure,
    Branch,
};
use ofgpu::compressible::gatemesh::{duct_area_ratio, nozzle_duct, DuctSpec};
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

use crate::{Checks, GateReport, How, Uncertainty, Verdict};

/// The gate's scope and registry name.
pub(crate) const GATE: &str = "SPEC-LIT 127.6 Gate CMP-NOZ quasi-1-D nozzle";

const AGAINST: &str = "the quasi-1-D keys (127.23)-(127.25), SPEC-LIT §127.2's isentropic \
                       and shocked nozzle relations evaluated live by compressible::exact (CMP-06)";

/// What the verdict says about the mesh, §94.3.
const ONE_DUCT: &str = "one 200-cell duct, as the row states";

/// The rest state's gauge datum, which is also the inlet total pressure:
/// the inlet's `p0` is `0` gauge, so `p_t1 = P_OP` absolute.
pub(crate) const P_OP: f64 = 1.0e5;
/// The inlet total temperature [K].
pub(crate) const T0: f64 = 300.0;
/// The row's mesh: the duct generator's default, 200 cells on `L = 1` m.
pub(crate) const N_CELLS: usize = 200;
/// The `CFL` of the local time step (125.2). Decision TAKEN (inputs
/// only, §127.6.5): CFL 5 and 2 stop the subsonic regime on the
/// `p_abs <= 0` bound in f64; CFL 1 reaches its fixed point.
pub(crate) const LTS_CFL: f64 = 1.0;
/// The iteration budget of a regime, the chunk the field stop is
/// tested on, and the stop: no cell's `M_i` moves by more than
/// `FIELD_STOP` across a chunk, a 500th of the 0.5 % bar.
pub(crate) const MAX_ITERS: usize = 20000;
pub(crate) const CHUNK: usize = 200;
pub(crate) const FIELD_STOP: f64 = 1.0e-5;
/// The shocked regime's shock area over the throat area, and the duct's
/// exit area over the throat area, (127.22): `a(0) = a(1) = 2`.
pub(crate) const AS_AT: f64 = 1.5;
pub(crate) const AE_AT: f64 = 2.0;
/// The rows' bars, the plan's, fixed before measurement.
pub(crate) const M_BAR: f64 = 0.005;
pub(crate) const P_BAR: f64 = 0.005;
pub(crate) const SHOCK_BAR: f64 = 0.01;
pub(crate) const PT_BAR: f64 = 0.01;
/// The back pressures of the two isentropic regimes, over `p_t1`: 0.95 is
/// above the choked limit 0.9371625024322055 and 0.05 below the design
/// exit 0.09393264573284488 (§127.2's oracle for `A_e/A_* = 2`).
pub(crate) const PB_SUBSONIC: f64 = 0.95;
pub(crate) const PB_SUPERSONIC: f64 = 0.05;

fn w(x: Scalar) -> f64 {
    f64::from(x)
}

fn s(x: f64) -> Scalar {
    x as Scalar
}

/// The three regimes of the row, §127.2's back-pressure intervals for
/// `A_e/A_t = 2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Regime {
    Subsonic,
    Supersonic,
    Shocked,
}

impl Regime {
    /// The regime's name in the rows and notes.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Regime::Subsonic => "subsonic",
            Regime::Supersonic => "supersonic",
            Regime::Shocked => "shocked",
        }
    }
}

/// The regime's key, §127.6.5: every value in `f64` whatever `Scalar` is
/// (an answer key is a constant of a gate, never a `Scalar`), against
/// §127.2's relations evaluated live by `compressible::exact` (CMP-06).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NozzleKey {
    pub regime: Regime,
    pub gamma: f64,
    /// The back pressure over the inlet total pressure `p_t1`: 0.95
    /// (subsonic), 0.05 (supersonic), `shocked_nozzle_back_pressure` at
    /// `A_s/A_t = 1.5` (shocked).
    pub pb_pt1: f64,
    /// `A_*/A_t`: the throat area the isentropic key is written about,
    /// `a(1)/(A/A_*)(M_e)` when the throat is unchoked (subsonic) and 1
    /// when it is (`A_* = A_t`, supersonic and shocked).
    pub astar_over_at: f64,
    /// The shock's station `xi_s = 1/2 + sqrt(0.125)`, where
    /// `a(xi_s) = A_s/A_t = 1.5` on the diverging side; `None` in the two
    /// isentropic regimes.
    pub xi_s: Option<f64>,
    /// The total-pressure ratio across the shock, (127.25); 1 when there
    /// is no shock.
    pub pt2_pt1: f64,
}

impl NozzleKey {
    /// The key of `regime` for `gamma` and the gate's duct `A_e/A_t = 2`.
    pub(crate) fn new(gamma: f64, regime: Regime) -> Result<Self> {
        let key = match regime {
            Regime::Subsonic => {
                // (127.23): the exit Mach of the back pressure, NACA (44)
                // inverted, then the sonic area A_* (below the throat's)
                // the subsonic duct's isentropic relation is written about.
                let m_e =
                    ((2.0 / (gamma - 1.0)) * ((1.0 / PB_SUBSONIC).powf((gamma - 1.0) / gamma)
                        - 1.0))
                    .sqrt();
                let astar_over_at = duct_area_ratio(1.0) / area_ratio(gamma, m_e);
                Self {
                    regime,
                    gamma,
                    pb_pt1: PB_SUBSONIC,
                    astar_over_at,
                    xi_s: None,
                    pt2_pt1: 1.0,
                }
            }
            Regime::Supersonic => Self {
                regime,
                gamma,
                pb_pt1: PB_SUPERSONIC,
                astar_over_at: 1.0,
                xi_s: None,
                pt2_pt1: 1.0,
            },
            Regime::Shocked => {
                // (127.25): the shock at A_s/A_t = 1.5 fixes the back
                // pressure (the forward map, §127.2), the subsonic side
                // runs about A_*2 = A_*1 p_t1/p_t2 (derived).
                let pb_pt1 = shocked_nozzle_back_pressure(gamma, AS_AT, AE_AT)?;
                let m1 = mach_from_area_ratio(gamma, AS_AT, Branch::Supersonic)?;
                let pt2_pt1 = normal_shock(gamma, m1)?.pt2_pt1;
                Self {
                    regime,
                    gamma,
                    pb_pt1,
                    astar_over_at: 1.0,
                    xi_s: Some(0.5 + 0.125f64.sqrt()),
                    pt2_pt1,
                }
            }
        };
        Ok(key)
    }

    /// The key's Mach number at the cell station `xi` in [0, 1]: the
    /// isentropic regimes read the branch the regime runs on; the shocked
    /// key is supersonic to the shock and reads `A/A_*2 = a(xi) p_t2/p_t1`
    /// behind it ((127.25), derived from §127.2's `A_*2 = A_*1 p_t1/p_t2`).
    pub(crate) fn mach(&self, xi: f64) -> Result<f64> {
        match self.regime {
            Regime::Subsonic => mach_from_area_ratio(
                self.gamma,
                duct_area_ratio(xi) / self.astar_over_at,
                Branch::Subsonic,
            ),
            Regime::Supersonic => {
                let branch = if xi <= 0.5 { Branch::Subsonic } else { Branch::Supersonic };
                mach_from_area_ratio(self.gamma, duct_area_ratio(xi), branch)
            }
            Regime::Shocked => {
                // Ahead of the shock the throat is A_*1 = A_t, the
                // supersonic-started branch; behind it the throat is
                // A_*2 = A_*1 p_t1/p_t2, so A/A_*2 = a(xi) p_t2/p_t1
                // ((127.25), derived from §127.2).
                let xi_s = self.xi_s.expect("the shock station");
                if xi < xi_s {
                    let branch = if xi <= 0.5 { Branch::Subsonic } else { Branch::Supersonic };
                    mach_from_area_ratio(self.gamma, duct_area_ratio(xi), branch)
                } else {
                    mach_from_area_ratio(
                        self.gamma,
                        duct_area_ratio(xi) * self.pt2_pt1,
                        Branch::Subsonic,
                    )
                }
            }
        }
    }

    /// The key's static pressure over `p_t1` at `xi`: `p_t2/p_t1` times
    /// `(p/p_t)(M(xi))` ((127.24)-(127.25); `p_t2/p_t1 = 1` isentropic).
    pub(crate) fn p_over_pt1(&self, xi: f64) -> Result<f64> {
        let m = self.mach(xi)?;
        Ok(self.pt2_pt1 * pressure_ratio(self.gamma, m))
    }
}

/// `p_t/p = (1 + (gamma-1)/2 M^2)^(gamma/(gamma-1))`, NACA (44) inverted
/// (the reciprocal of CMP-06's `pressure_ratio`).
pub(crate) fn total_over_static(gamma: f64, m: f64) -> f64 {
    (1.0 + 0.5 * (gamma - 1.0) * m * m).powf(gamma / (gamma - 1.0))
}

/// The first upward crossing of `p_mid` by `p` between consecutive entries
/// whose xi are both > `after`, linearly interpolated in xi; None if there
/// is none. The shocked row reads the shock's station with it, so a
/// crossing at or below `after` (the throat side) never counts.
pub(crate) fn locate_shock(xi: &[f64], p: &[f64], p_mid: f64, after: f64) -> Option<f64> {
    for i in 1..xi.len().min(p.len()) {
        if xi[i - 1] > after && xi[i] > after && p[i - 1] <= p_mid && p[i] > p_mid {
            let f = (p_mid - p[i - 1]) / (p[i] - p[i - 1]);
            return Some(xi[i - 1] + f * (xi[i] - xi[i - 1]));
        }
    }
    None
}

/// The largest change between two Mach fields, cell by cell; a NaN
/// anywhere makes it NaN, so a broken run never meets the field stop.
pub(crate) fn max_change(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, |m: f64, v| if v.is_nan() || m.is_nan() { f64::NAN } else { m.max(v) })
}

/// The cell Mach numbers of a downloaded state, `M_i = |u_i| / c_i` with
/// `c_i = sqrt(gamma R_s T_i)`, all in `f64`.
pub(crate) fn mach_field(u: &[Vec3], t: &[Scalar], gamma: f64, r_s: f64) -> Vec<f64> {
    u.iter()
        .zip(t)
        .map(|(v, ti)| w(v.mag()) / (gamma * r_s * w(*ti)).sqrt())
        .collect()
}

fn entry(t: &str) -> PatchFieldSpec {
    PatchFieldSpec { type_name: t.into(), ..PatchFieldSpec::default() }
}

/// The inlet `inlet`, the outlet `outlet`, and `symmetry` on the two
/// slip-wall patches `walls` and `frontAndBack`.
fn bcs(
    mut b: BTreeMap<String, PatchFieldSpec>,
    inlet: PatchFieldSpec,
    outlet: PatchFieldSpec,
) -> BTreeMap<String, PatchFieldSpec> {
    b.insert("inlet".into(), inlet);
    b.insert("outlet".into(), outlet);
    for side in ["walls", "frontAndBack"] {
        b.insert(side.into(), entry("symmetry"));
    }
    b
}

/// One regime's fields and bookkeeping, after its march.
pub(crate) struct Run {
    pub key: NozzleKey,
    /// The cell stations `xi_i = x_i/L`, the cell Mach numbers and the
    /// ABSOLUTE static pressures [Pa], from the downloaded fields.
    pub xi: Vec<f64>,
    pub mach: Vec<f64>,
    pub p_abs: Vec<f64>,
    /// The outlet FACE `p` over `p_t1 (p/p_t)(M_e)` (printed only).
    pub outlet_face_ratio: f64,
    /// The first cell's `p_t/p_t1` (printed only).
    pub p_t_first: f64,
    pub iterations: usize,
    /// The field stop was met.
    pub converged: bool,
    /// `(iterations, max |change|)` after each chunk.
    pub history: Vec<(usize, f64)>,
    pub last_line: String,
    pub seconds: f64,
}

/// One march of one regime (§127.6.5's setup): the duct,
/// the rest start, the chunked march to the field stop. A solver bound
/// (`Error::Refused`, §122.3) is the caller's, as `Err`.
fn run_once(gpu: &Gpu, gamma: f64, r_s: f64, key: &NozzleKey) -> Result<Run> {
    let t_start = Instant::now();
    let gas = PerfectGas {
        viscosity: Viscosity::Constant { mu: 0.0 },
        ..PerfectGas::air(P_OP)
    };
    let duct = nozzle_duct(&DuctSpec::default())?;
    let hm = build_host_mesh(&duct.raw)?;
    let gm = GpuMesh::upload(gpu, &hm)?;
    let n = hm.n_cells;
    let pb_gauge = s((key.pb_pt1 - 1.0) * P_OP);

    // At rest, NOT from the key: `u = 0`, `T = T_0`, and a gauge `p` that
    // falls linearly from the inlet's total pressure to the back pressure,
    // so the start carries no cell's answer.
    let mut rp = RawScalarField { name: "p".into(), ..RawScalarField::default() };
    rp.internal = (0..n)
        .map(|i| s((w(hm.c[i].x) / duct.spec.length) * (key.pb_pt1 - 1.0) * P_OP))
        .collect();
    let out_p = PatchFieldSpec {
        type_name: "fixedValue".into(),
        value: vec![pb_gauge],
        ..PatchFieldSpec::default()
    };
    rp.boundary = bcs(
        rp.boundary,
        with(entry("totalPressure"), &[("p0", "uniform 0"), ("gamma", "1.4")]),
        out_p,
    );
    let mut rt = RawScalarField { name: "T".into(), ..RawScalarField::default() };
    rt.internal = vec![s(T0); n];
    rt.boundary = bcs(
        rt.boundary,
        with(entry("totalTemperature"), &[("T0", "uniform 300"), ("gamma", "1.4")]),
        entry("zeroGradient"),
    );
    let mut ru = RawVectorField { name: "U".into(), ..RawVectorField::default() };
    ru.internal = vec![Vec3::ZERO; n];
    ru.boundary =
        bcs(ru.boundary, entry("zeroGradient"), entry("zeroGradient"));

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
    ctrl.n_outer = 1;
    ctrl.residual_control = None;
    ctrl.momentum.div_scheme = DivScheme::Limited(Limiter::VanLeer);
    ctrl.energy.div_scheme = DivScheme::Limited(Limiter::VanLeer);
    ctrl.pressure.retarded = Retarded::Limited(Limiter::VanLeer);

    let low = cbc::lower(&ru, &rp, &rt, &hm, &ctrl.gas)?;
    let mut uf = GpuVectorField::zeros(gpu, &gm, "U")?;
    let mut pf = GpuScalarField::zeros(gpu, &gm, "p")?;
    let mut tf = GpuScalarField::zeros(gpu, &gm, "T")?;
    setup_vector_field(gpu, &mut uf, &low.u, &hm)?;
    setup_scalar_field(gpu, &mut pf, &low.p, &hm)?;
    setup_scalar_field(gpu, &mut tf, &low.t, &hm)?;
    let mut st = CompressibleState::new(gpu, &gm, uf, pf, tf)?;
    let mut solver = Compressible::new(gpu, &gm, ctrl)?;
    solver.set_boundary_rules(gpu, low.rules)?;
    // §126.4's supersonic switch on the fixed-p outlet: without it a
    // supersonic exit cannot work.
    solver.set_outlet_rules(gpu, low.outlets)?;
    solver.initialise(gpu, &mut st)?;

    // The stop is the field's own: chunks of CHUNK iterations until no
    // cell's `M_i` moves by more than FIELD_STOP across a chunk.
    // `residualControl` is off, as CMP-COU's is.
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
        // A bound (T, p_abs or rho <= 0) stops `run` with Error::Refused (§122.3).
        let rep = solver.run(gpu, &mut st, k)?;
        iterations += rep.steps;
        if let Some(o) = rep.last.as_ref().and_then(|l| l.outers.last()) {
            last_line = o.line();
        }
        let next = machs(&st)?;
        change = max_change(&prev, &next);
        history.push((iterations, change));
        prev = next;
    }
    let mach = machs(&st)?;
    let p_abs: Vec<f64> = gpu.download(&st.p.f)?.iter().map(|v| w(*v) + P_OP).collect();
    let outlet = hm.patches.iter().find(|p| p.name == "outlet").expect("the outlet patch");
    let p_bf = gpu.download(&st.p.bf)?;
    let face = w(p_bf[outlet.start]);
    let outlet_face_ratio = (face + P_OP) / (P_OP * key.p_over_pt1(1.0)?);
    let p_t_first = (p_abs[0] * total_over_static(gamma, mach[0])) / P_OP;
    Ok(Run {
        key: *key,
        xi: (0..n).map(|i| w(hm.c[i].x) / duct.spec.length).collect(),
        mach,
        p_abs,
        outlet_face_ratio,
        p_t_first,
        iterations,
        converged: change <= FIELD_STOP,
        history,
        last_line,
        seconds: t_start.elapsed().as_secs_f64(),
    })
}

/// A patch entry with extra keys (`p0`, `T0`, `gamma`), as the case reader
/// keeps them.
fn with(mut e: PatchFieldSpec, kv: &[(&str, &str)]) -> PatchFieldSpec {
    for (k, v) in kv {
        e.extra.insert((*k).into(), (*v).into());
    }
    e
}

/// The row labels: each starts `Gate CMP-NOZ <regime>: `, states its bar in
/// words, and ends `, 200 cells (SPEC-LIT 127.6)`.
fn mach_label(regime: Regime) -> String {
    format!(
        "Gate CMP-NOZ {}: M(x) within 0.5 % of the key's M(xi), all {N_CELLS} cells (SPEC-LIT 127.6)",
        regime.name()
    )
}

fn exit_label(regime: Regime) -> String {
    format!(
        "Gate CMP-NOZ {}: exit p within 0.5 % of the key at the last cell centre (SPEC-LIT 127.6)",
        regime.name()
    )
}

/// The `M(x)` row's measure: `max_i |M_i / M(xi_i) - 1|` over all the
/// cells. A NaN anywhere makes it NaN.
pub(crate) fn mach_error(m: &[f64], key: &NozzleKey, xi: &[f64]) -> Result<f64> {
    let mut err = 0.0f64;
    for (mi, &xi_i) in m.iter().zip(xi) {
        let e = (mi / key.mach(xi_i)? - 1.0).abs();
        err = if e.is_nan() || err.is_nan() { f64::NAN } else { err.max(e) };
    }
    Ok(err)
}

/// The exit `p` row's measure at the LAST cell's centre, absolute over the
/// key's at that centre: `|(p_last + p_op)/(p_t1 (p/p_t1)(xi_last)) - 1|`.
pub(crate) fn exit_p_error(p_abs_last: f64, key: &NozzleKey, xi_last: f64) -> Result<f64> {
    Ok((p_abs_last / (P_OP * key.p_over_pt1(xi_last)?) - 1.0).abs())
}

/// **Gate CMP-NOZ** (SPEC-LIT §127.6, §127.6.5): the three regimes in
/// order, each marched from rest and held against its key.
pub(crate) fn check_nozzle(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    let gas = PerfectGas::air(P_OP);
    let (gamma, r_s) = (gas.gamma, gas.r_s());
    let mut misses: Vec<String> = Vec::new();
    for regime in [Regime::Subsonic, Regime::Supersonic, Regime::Shocked] {
        let key = NozzleKey::new(gamma, regime)?;
        let name = regime.name();
        c.note(&format!(
            "Gate CMP-NOZ {name}: back pressure p_b/p_t1 {:.10}, p_t1 {P_OP} Pa, T_0 {T0} K, \
             A_e/A_t {AE_AT}, gamma {gamma}, LTS CFL {LTS_CFL}",
            key.pb_pt1
        ));

        // A bound (T, p_abs or rho <= 0, §122.3) stops the march with
        // Error::Refused: the regime has no field, and its rows read NaN.
        let r = match run_once(gpu, gamma, r_s, &key) {
            Ok(r) => r,
            Err(Error::Refused(why)) => {
                c.note(&format!(
                    "Gate CMP-NOZ {name}: the march at LTS CFL {LTS_CFL} stopped on a bound, \
                     so this regime has no field to compare: {why}"
                ));
                for label in row_labels(regime) {
                    c.check_p(&label, Scalar::NAN, s(row_bar(regime, &label)));
                    misses.push(format!("{label}: no field, the march stopped on a bound"));
                }
                continue;
            }
            Err(e) => return Err(e),
        };
        note_run(c, regime, &r)?;
        let rows = measure(regime, &r, gamma)?;
        for (label, err) in &rows {
            c.check_p(label, s(*err), s(row_bar(regime, label)));
            // `!(err <= bar)`, so that a NaN measure counts as the miss it is.
            if !(*err <= row_bar(regime, label)) {
                misses.push(format!("{label}: {err:.4e} against {}", row_bar(regime, label)));
            }
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
            "the three regimes in order at LTS CFL {LTS_CFL}, \
             chunks of {CHUNK} iterations to a cell Mach change of {FIELD_STOP:.1e}, \
             {MAX_ITERS} iterations a regime: every measured value is in the rows and notes above"
        )],
        uncertainty: Some(Uncertainty::SingleMesh(ONE_DUCT)),
    });
    Ok(())
}

/// The regime's row labels, in the section's order.
fn row_labels(regime: Regime) -> Vec<String> {
    match regime {
        Regime::Subsonic | Regime::Supersonic => vec![mach_label(regime), exit_label(regime)],
        Regime::Shocked => vec![
            "Gate CMP-NOZ shocked: shock position within 1 % of the length, 200 cells (SPEC-LIT 127.6)"
                .to_string(),
            "Gate CMP-NOZ shocked: p_t2/p_t1 within 1 % of (127.25)'s, the last cell (SPEC-LIT 127.6)"
                .to_string(),
            exit_label(regime),
        ],
    }
}

/// The bar a label carries, by the row it names.
fn row_bar(regime: Regime, label: &str) -> f64 {
    if label == mach_label(regime) {
        M_BAR
    } else if label == exit_label(regime) {
        P_BAR
    } else if label.contains("shock position") {
        SHOCK_BAR
    } else {
        PT_BAR
    }
}

/// The largest of a fold that keeps a NaN: the rows' L-infinity reads NaN
/// when any cell is NaN, like `max_change` does.
fn nan_max(iter: impl Iterator<Item = f64>) -> f64 {
    iter.fold(0.0, |m, v| if v.is_nan() || m.is_nan() { f64::NAN } else { m.max(v) })
}

/// The run's notes: the march's state and per-chunk history, then the
/// printed-only checks - the outlet FACE p over the key's exit value, the
/// first cell's `p_t/p_t1`, and for the shocked regime the `M(x)`
/// L-infinity over the cells away from the shock.
fn note_run(c: &mut Checks, regime: Regime, r: &Run) -> Result<()> {
    let name = regime.name();
    let m_max = nan_max(r.mach.iter().copied());
    c.note(&format!(
        "Gate CMP-NOZ {name}: {} iterations, field stop {}, M_max {:.4}, {:.1} s; last: {}",
        r.iterations,
        if r.converged { "met" } else { "NOT met" },
        m_max,
        r.seconds,
        r.last_line
    ));
    let hist: Vec<String> = r.history.iter().map(|(i, d)| format!("{i}: {d:.2e}")).collect();
    c.note(&format!(
        "Gate CMP-NOZ {name}: max |change| per chunk: {}",
        hist.join(", ")
    ));
    c.note(&format!(
        "Gate CMP-NOZ {name}: printed, never checked: the outlet face p / (p_t1 (p/p_t)(M_e)) \
         is {:.6}; the first cell's p_t/p_t1 is {:.6}",
        r.outlet_face_ratio, r.p_t_first
    ));
    if let Some(xi_s) = r.key.xi_s {
        let away = nan_max(r.xi.iter().zip(&r.mach).filter(|(xi, _)| (**xi - xi_s).abs() > 0.03)
            .map(|(xi, m)| (m / r.key.mach(*xi).unwrap_or(f64::NAN) - 1.0).abs()));
        c.note(&format!(
            "Gate CMP-NOZ shocked: printed, never checked: M(x) L-infinity {away:.3e} over the \
             cells with |xi - xi_s| > 0.03"
        ));
    }
    Ok(())
}

/// The regime's rows as `(label, error)` pairs, in the section's order,
/// every measure computed in `f64` from the downloaded fields.
fn measure(regime: Regime, r: &Run, gamma: f64) -> Result<Vec<(String, f64)>> {
    let labels = row_labels(regime);
    let n_last = r.xi.len() - 1;
    let rows = match regime {
        Regime::Subsonic | Regime::Supersonic => vec![
            (labels[0].clone(), mach_error(&r.mach, &r.key, &r.xi)?),
            (labels[1].clone(), exit_p_error(r.p_abs[n_last], &r.key, r.xi[n_last])?),
        ],
        Regime::Shocked => {
            // The shock's measured station: the p/p_t1 profile's first
            // upward crossing, at or after the throat, of the mean of the
            // key's face values `p_1/p_t1` and `p_2/p_t1` ((127.25)).
            let m1 = mach_from_area_ratio(gamma, AS_AT, Branch::Supersonic)?;
            let p1 = pressure_ratio(gamma, m1);
            let p2 = p1 * normal_shock(gamma, m1)?.p2_p1;
            let p_t1: Vec<f64> = r.p_abs.iter().map(|p| p / P_OP).collect();
            let xi_num =
                locate_shock(&r.xi, &p_t1, 0.5 * (p1 + p2), 0.5).unwrap_or(f64::NAN);
            let pt_last = r.p_abs[n_last] * total_over_static(gamma, r.mach[n_last]);
            vec![
                (labels[0].clone(), (xi_num - r.key.xi_s.expect("the shock station")).abs()),
                (labels[1].clone(), (pt_last / P_OP / r.key.pt2_pt1 - 1.0).abs()),
                (labels[2].clone(), exit_p_error(r.p_abs[n_last], &r.key, r.xi[n_last])?),
            ]
        }
    };
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The key values, from a 40-digit mpmath oracle: the subsonic regime
    /// at `p_b/p_t1 = 0.95`, all in `f64` whatever `Scalar` is.
    #[test]
    fn the_subsonic_key() {
        let key = NozzleKey::new(1.4, Regime::Subsonic).expect("the subsonic key");
        let close = |got: f64, want: f64, tol: f64, what: &str| {
            assert!(
                (got - want).abs() <= tol * want.abs(),
                "{what}: {got} is not {want}"
            );
        };
        close(key.mach(1.0).expect("M_e"), 0.27169046111338039, 1e-12, "M_e");
        close(key.mach(0.5).expect("M at the throat"), 0.67603052349157071, 1e-12, "M(1/2)");
        close(key.mach(0.25).expect("M"), 0.47486927485876826, 1e-12, "M(1/4)");
        close(key.p_over_pt1(1.0).expect("p"), 0.95, 1e-13, "p_e/p_t1");
        close(key.astar_over_at, 0.89857446310320031, 1e-12, "A_*/A_t");
        assert_eq!(key.xi_s, None);
        assert_eq!(key.pt2_pt1, 1.0);
    }

    /// The supersonic started regime at `p_b/p_t1 = 0.05`: sonic at the
    /// throat EXACTLY, supersonic down the diverging side to the design
    /// exit pressure.
    #[test]
    fn the_supersonic_key() {
        let key = NozzleKey::new(1.4, Regime::Supersonic).expect("the supersonic key");
        let close = |got: f64, want: f64, tol: f64, what: &str| {
            assert!(
                (got - want).abs() <= tol * want.abs(),
                "{what}: {got} is not {want}"
            );
        };
        assert_eq!(key.mach(0.5).expect("the throat"), 1.0, "the throat is sonic exactly");
        close(key.mach(0.0).expect("M"), 0.30590383418910821, 1e-12, "M(0)");
        close(key.mach(0.75).expect("M"), 1.5997084401485519, 1e-12, "M(3/4)");
        close(key.mach(1.0).expect("M_e"), 2.1971981216521865, 1e-12, "M_e");
        close(key.p_over_pt1(1.0).expect("p"), 0.093932645732844884, 1e-12, "p_e/p_t1");
        assert_eq!(key.astar_over_at, 1.0);
    }

    /// The shocked regime at `A_s/A_t = 1.5`: its back pressure is the
    /// forward map's, `xi_s` puts `a(xi_s) = 1.5` on the diverging side,
    /// and the subsonic side behind the shock runs about `A_*2`.
    #[test]
    fn the_shocked_key() {
        let key = NozzleKey::new(1.4, Regime::Shocked).expect("the shocked key");
        let close = |got: f64, want: f64, tol: f64, what: &str| {
            assert!(
                (got - want).abs() <= tol * want.abs(),
                "{what}: {got} is not {want}"
            );
        };
        close(key.pb_pt1, 0.70445197792232862, 1e-12, "p_b/p_t1");
        close(key.xi_s.expect("xi_s"), 0.85355339059327376, 1e-15, "xi_s");
        close(key.pt2_pt1, 0.78835942907796554, 1e-12, "p_t2/p_t1");
        let xi_s = key.xi_s.expect("xi_s");
        let m_up = key.mach(xi_s - 1.0e-9).expect("M ahead of the shock");
        assert!(
            (m_up - 1.8541235267373254).abs() <= 1e-6,
            "M(xi_s - 1e-9) {m_up} is not M_1 1.8541235267373254"
        );
        close(key.mach(xi_s).expect("M"), 0.60484304654839300, 1e-10, "M just behind the shock");
        close(key.mach(0.9).expect("M"), 0.52613022109693165, 1e-12, "M(0.9)");
        close(key.p_over_pt1(0.9).expect("p"), 0.65285737593055416, 1e-12, "p(0.9)/p_t1");
        close(key.mach(1.0).expect("M_e"), 0.40419695203975395, 1e-12, "M_e");
        close(
            key.p_over_pt1(1.0).expect("p"),
            key.pb_pt1,
            1e-12,
            "p_e/p_t1 = p_b/p_t1",
        );
    }

    /// `total_over_static` is NACA (44) inverted: the reciprocal of CMP-06's
    /// `pressure_ratio`, to round-off, at M = 0 exactly 1.
    #[test]
    fn total_over_static_is_the_isentropic_ratio() {
        close_to(1.0e-14, total_over_static(1.4, 2.0), 7.8244490668672641, "p_t/p at M 2");
        assert_eq!(total_over_static(1.4, 0.0), 1.0);
        assert!((total_over_static(1.4, 0.5) * pressure_ratio(1.4, 0.5) - 1.0).abs() <= 1e-14);
    }

    fn close_to(tol: f64, got: f64, want: f64, what: &str) {
        assert!((got - want).abs() <= tol * want.abs(), "{what}: {got} is not {want}");
    }

    /// The locator takes the first upward crossing whose BOTH entries sit
    /// after `after`, interpolates linearly, and returns None when no pair
    /// crosses or when the only crossing is before `after`.
    #[test]
    fn the_shock_locator() {
        let xi = [0.40, 0.80, 0.81, 0.82, 0.83];
        let p = [0.10, 0.16, 0.16, 0.62, 0.62];
        let hit = locate_shock(&xi, &p, 0.39, 0.5).expect("the crossing");
        assert!((hit - 0.815).abs() <= 1e-14, "xi_s {hit} is not 0.815");
        assert_eq!(locate_shock(&xi, &p, 0.39, 0.9), None, "nothing after 0.9");
        // The crossing at xi = 0.15 is ignored: its entries are not both
        // after 0.5, and no later pair crosses.
        let xi = [0.1, 0.2, 0.6, 0.7];
        let p = [0.1, 0.9, 0.2, 0.3];
        assert_eq!(locate_shock(&xi, &p, 0.5, 0.5), None);
    }
}
