// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate CMP-RING (SPEC-LIT §127.6, §127.6.7): Ringleb's transonic smooth
//! flow on three meshes, the observed order of the L2 entropy error,
//! live.**
//!
//! §127.5.4's Ringleb domain - walls on the streamlines `k = 1.0` and
//! `k = 0.6`, the ends the iso-speed lines `q = 0.5`, the nodes ON the
//! answer key (127.28) - is run at three levels, `h`, `h/2`, `h/4`
//! ((24, 6), (48, 12) and (96, 24) cells), carrying air (`gamma = 1.4`,
//! inviscid) from rest at a linear gauge `p` to the outlet's (127.8)
//! pressure, marched with SIMPLE and the local time step of §125.4 at LTS
//! CFL 1 until no cell's `M_i` moves by more than `1e-5` across a chunk of
//! 200 iterations (capped at 10000). The key is the exact field itself,
//! evaluated live by `compressible::exact` (CMP-07) in `f64` in BOTH
//! builds: isentropic, so §127.3's `p/rho^gamma * gamma - 1`, written
//! (127.29) in the downloaded `p` and `T`, is `e = 0` everywhere on it,
//! and `E` by (94.1) is the error. One row, class P in the f64 build
//! (docs/17 section 5.1.4): the observed order of `E` across the
//! three meshes `>= 1.5` (SPEC-LIT 127.6), computed in `f64` ONLY - under
//! `single` the gate measures, reports its error floor and checks
//! nothing. A march that stops on a solver bound (§122.3) leaves that
//! mesh's `E` NaN and the gate's one verdict says why.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §127.3 (the Ringleb field, (127.8)-(127.11a) and
//!     its oracle values, via CMP-07's `compressible::exact`, not read
//!     directly - the key is CMP-07's), §127.5.4 (the domain), §127.6
//!     (the gate table) and §127.6.7 (this gate as built), §94 (the mesh
//!     study), §122 (the gas), §125.4 (the local time step) and §126.4
//!     (the outlets' supersonic switch)
//!   F. Ringleb (1940), via the HiOCFD page and AGARD-AR-211, both cited
//!     by docs/17 - not read; the key is CMP-07's
//! No GPL-licensed source was consulted.

use std::collections::BTreeMap;
use std::time::Instant;

use ofgpu::compressible::bc as cbc;
use ofgpu::compressible::exact::{ringleb_state, RINGLEB_GAMMA};
use ofgpu::compressible::gatemesh::{ringleb_domain, RinglebSpec};
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
use ofgpu::vv::{self, GridStudy, Level};
use ofgpu::{Error, Gpu, GpuMesh, Result, Scalar, Vec3};

use super::nozzle::{mach_field, max_change, P_OP};
use super::sod::fitted_order;
use crate::{Checks, GateReport, How, Uncertainty, Verdict};

/// The gate's scope and registry name.
pub(crate) const GATE: &str = "SPEC-LIT 127.6 Gate CMP-RING Ringleb flow";

const AGAINST: &str = "the isentropic Ringleb field of (127.8)-(127.11a), SPEC-LIT \
                       section 127.3's smooth answer key evaluated live by \
                       compressible::exact (CMP-07)";

/// The three gate meshes `(nt, nk)`, finest first: each level doubles both
/// counts, so (94.1)'s `h` halves and the levels are `h`, `h/2`, `h/4`.
pub(crate) const MESHES: [(usize, usize); 3] = [(96, 24), (48, 12), (24, 6)];

/// The stagnation temperature [K], the inlet's `totalTemperature`.
const T0: f64 = 300.0;
/// The march: the chunk the field stop is tested on, the stop itself - no
/// cell's `M_i` moves by more than this across a chunk - and the
/// iteration budget per mesh (§127.6.5's, CMP-NOZ's march). Decision
/// TAKEN (inputs only, §127.6.7): the budget is CMP-OBL's 10000, because
/// the field stop is not met in 20000 either (the supervisor's run: a
/// cell-M change of 5.2e-3 to 7.1e-3 per chunk at 20000 on all three
/// meshes) and 20000 put the gate past ten minutes on the shared GPU.
pub(crate) const LTS_CFL: f64 = 1.0;
pub(crate) const N_OUTER: usize = 1;
pub(crate) const MAX_ITERS: usize = 10000;
pub(crate) const CHUNK: usize = 200;
pub(crate) const FIELD_STOP: f64 = 1.0e-5;
/// The row's bar, the plan's, fixed before measurement (class P, f64 only).
pub(crate) const ORDER_MIN: f64 = 1.5;

/// The scheme decisions (TAKEN by the supervisor, inputs only): momentum
/// runs the plan's `linearUpwind` with NO limiter; energy takes `vanLeer`,
/// because the energy equation refuses a deferred-correction scheme by name
/// (its face value is not a weight, and `K_f` must use the face weights of
/// `T_f` so the face sum is a flux of `h0`, SPEC-LIT §125.1), so the tree
/// wins over the plan's wording, and its one unlimited second-order choice,
/// `Gauss linear`, stops every mesh on a `T <= 0` bound within 811 steps in
/// both builds (measured, §127.6.7); and the retarded density takes
/// `vanLeer`, its only options being first-order upwind or a limiter.
const SCHEMES: (DivScheme, DivScheme, Retarded) = (
    DivScheme::LinearUpwind,
    DivScheme::Limited(Limiter::VanLeer),
    Retarded::Limited(Limiter::VanLeer),
);

fn w(x: Scalar) -> f64 {
    f64::from(x)
}

fn s(x: f64) -> Scalar {
    x as Scalar
}

/// (127.29), derived: §127.3's entropy error `p/rho^gamma * gamma - 1`
/// with `rho = p/(R_s T)` and `p`, `T` read over their stagnation values
/// `p0`, `T0` - the exact field isentropic, so `e = 0` on the key and a
/// cell's `e` is its spurious entropy.
pub(crate) fn entropy_error(p_over_p0: f64, t_over_t0: f64, gamma: f64) -> f64 {
    p_over_p0.powf(1.0 - gamma) * t_over_t0.powf(gamma) - 1.0
}

/// (94.1)'s weighted L2, `sqrt(sum V_i e_i^2 / sum V_i)`. NaN when `e` is
/// empty or the weights do not sum positive.
pub(crate) fn weighted_l2(e: &[f64], v: &[f64]) -> f64 {
    if e.is_empty() {
        return f64::NAN;
    }
    let (mut se, mut sv) = (0.0f64, 0.0f64);
    for (ei, vi) in e.iter().zip(v) {
        se += vi * ei * ei;
        sv += vi;
    }
    if sv <= 0.0 {
        return f64::NAN;
    }
    (se / sv).sqrt()
}

/// (127.8)'s static over stagnation pressure at speed `q`:
/// `(p/p0)(q) = (1 - (gamma-1)/2 q^2)^(gamma/(gamma-1))` - NACA (44) with
/// `M = q/a0` at `a0` the stagnation sound speed (derived; `a0` = 1 in
/// §127.3's scaling).
pub(crate) fn static_over_stagnation(gamma: f64, q: f64) -> f64 {
    (1.0 - 0.5 * (gamma - 1.0) * q * q).powf(gamma / (gamma - 1.0))
}

/// The observed order between two mesh levels, the finer `a` and the
/// coarser `b`: `ln(E_b/E_a)/ln(h_b/h_a)` (§94.2's quotient on a pair).
pub(crate) fn pair_order(h_fine: f64, e_fine: f64, h_coarse: f64, e_coarse: f64) -> f64 {
    (e_coarse / e_fine).ln() / (h_coarse / h_fine).ln()
}

fn entry(t: &str) -> PatchFieldSpec {
    PatchFieldSpec { type_name: t.into(), ..PatchFieldSpec::default() }
}

/// A patch entry with extra keys (`p0`, `T0`, `gamma`), as the case reader
/// keeps them.
fn with(mut e: PatchFieldSpec, kv: &[(&str, &str)]) -> PatchFieldSpec {
    for (k, v) in kv {
        e.extra.insert((*k).into(), (*v).into());
    }
    e
}

/// The inlet `inlet`, the outlet `outlet`, and `symmetry` on the three
/// slip patches `innerWall`, `outerWall` and `frontAndBack`.
fn bcs(
    mut b: BTreeMap<String, PatchFieldSpec>,
    inlet: PatchFieldSpec,
    outlet: PatchFieldSpec,
) -> BTreeMap<String, PatchFieldSpec> {
    b.insert("inlet".into(), inlet);
    b.insert("outlet".into(), outlet);
    for side in ["innerWall", "outerWall", "frontAndBack"] {
        b.insert(side.into(), entry("symmetry"));
    }
    b
}

/// One mesh's fields and bookkeeping, after its march stopped.
struct Run {
    nt: usize,
    nk: usize,
    /// The cell volumes [m^3] and centres [m], widened once.
    v: Vec<f64>,
    cx: Vec<f64>,
    cy: Vec<f64>,
    /// The cell Mach numbers and the ABSOLUTE static pressures [Pa] and
    /// static T [K], from the downloaded fields.
    mach: Vec<f64>,
    p_abs: Vec<f64>,
    t: Vec<f64>,
    /// (94.1)'s `h = sqrt(sum V / (D n_cells))` [m].
    h: f64,
    iterations: usize,
    /// The field stop was met.
    converged: bool,
    /// `(iterations, max |change|)` after each chunk.
    history: Vec<(usize, f64)>,
    last_line: String,
    seconds: f64,
}

/// One march of one mesh (§127.6.7's setup): the domain, the rest start,
/// the chunked march to the field stop. A solver bound (`Error::Refused`,
/// §122.3) is the caller's, as `Err`, carrying the iterations reached.
fn run_once(gpu: &Gpu, gas: PerfectGas, r_s: f64, nt: usize, nk: usize) -> Result<Run> {
    let t_start = Instant::now();
    let gamma = gas.gamma;
    let spec = RinglebSpec { nt, nk, ..RinglebSpec::default() };
    let ring = ringleb_domain(&spec)?;
    let hm = build_host_mesh(&ring.raw)?;
    let gm = GpuMesh::upload(gpu, &hm)?;
    let n = hm.n_cells;
    // The outlet is an iso-q line of the key, so it is isobaric: (127.8)'s
    // static p there, gauge (the datum P_OP is the stagnation pressure).
    let p_out_gauge = (static_over_stagnation(gamma, ring.spec.q_b) - 1.0) * P_OP;

    // At rest, NOT from the key: `u = 0`, `T = T0`, and the nozzle's
    // linear gauge `p` fall along the streamwise index to the outlet's
    // value, so the start carries no cell's answer.
    let mut rp = RawScalarField { name: "p".into(), ..RawScalarField::default() };
    rp.internal = (0..n)
        .map(|c| s((((c % nt) as f64) + 0.5) / (nt as f64) * p_out_gauge))
        .collect();
    let out_p = PatchFieldSpec {
        type_name: "fixedValue".into(),
        value: vec![s(p_out_gauge)],
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
    ru.boundary = bcs(ru.boundary, entry("zeroGradient"), entry("zeroGradient"));

    // §127.6.5's controls (CMP-21's march), but the gate's own schemes.
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
    ctrl.momentum.div_scheme = SCHEMES.0;
    ctrl.energy.div_scheme = SCHEMES.1;
    ctrl.pressure.retarded = SCHEMES.2;

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
    solver.set_outlet_rules(gpu, low.outlets)?;
    solver.initialise(gpu, &mut st)?;

    // The stop is the field's own: chunks of CHUNK iterations until no
    // cell's `M_i` moves by more than FIELD_STOP across a chunk.
    // `residualControl` is off, as CMP-NOZ's is.
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
        // Error::Refused: the mesh's miss, carrying the iterations.
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
    let mach = machs(&st)?;
    let p_abs: Vec<f64> = gpu.download(&st.p.f)?.iter().map(|v| w(*v) + P_OP).collect();
    let t: Vec<f64> = gpu.download(&st.t.f)?.iter().map(|v| w(*v)).collect();
    let v: Vec<f64> = hm.v.iter().map(|x| w(*x)).collect();
    let h = (v.iter().sum::<f64>() / (ring.spec.depth * n as f64)).sqrt();
    Ok(Run {
        nt,
        nk,
        v,
        cx: hm.c.iter().map(|c| w(c.x)).collect(),
        cy: hm.c.iter().map(|c| w(c.y)).collect(),
        mach,
        p_abs,
        t,
        h,
        iterations,
        converged: change <= FIELD_STOP,
        history,
        last_line,
        seconds: t_start.elapsed().as_secs_f64(),
    })
}

/// A run's entropy errors, (127.29) cell by cell, in `f64` from the
/// downloaded fields: the key is `e = 0` everywhere.
fn entropy_errors(r: &Run, gamma: f64) -> Vec<f64> {
    (0..r.v.len())
        .map(|c| entropy_error(r.p_abs[c] / P_OP, r.t[c] / T0, gamma))
        .collect()
}

/// The printed-only pressure L2 over the cells:
/// `sqrt(mean (p_i/P0 - gamma p_key(x_c, y_c))^2)`, `None` when the key
/// refuses a centre.
fn printed_p_l2(r: &Run, gamma: f64) -> Option<f64> {
    let mut sum = 0.0f64;
    for c in 0..r.v.len() {
        let st = ringleb_state(r.cx[c], r.cy[c]).ok()?;
        let d = r.p_abs[c] / P_OP - gamma * st.p;
        sum += d * d;
    }
    Some((sum / r.v.len() as f64).sqrt())
}

/// The f32 error floor, measured: (127.29) on the key's own `p`, `T` at
/// the finest gate mesh's cell centres after a round trip through `f32` -
/// the smallest `E` fields stored as `Scalar = f32` can carry there,
/// whatever the scheme and whatever the march did. NaN when the key
/// refuses a centre or the mesh refuses the spec.
fn f32_floor(gamma: f64) -> f64 {
    let spec = RinglebSpec { nt: MESHES[0].0, nk: MESHES[0].1, ..RinglebSpec::default() };
    let ring = match ringleb_domain(&spec) {
        Ok(r) => r,
        Err(_) => return f64::NAN,
    };
    let hm = match build_host_mesh(&ring.raw) {
        Ok(h) => h,
        Err(_) => return f64::NAN,
    };
    let v: Vec<f64> = hm.v.iter().map(|x| w(*x)).collect();
    let mut e = Vec::with_capacity(v.len());
    for cell in &hm.c {
        let st = match ringleb_state(w(cell.x), w(cell.y)) {
            Ok(st) => st,
            Err(_) => return f64::NAN,
        };
        // The key's static values, narrowed the way a `single` field
        // stores them: the GAUGE `p - P0 = (gamma p_key - 1) P0` (the key's
        // `p0 = 1/gamma`; the solver's `p` is the gauge about `P_OP = P0`)
        // and `T/T0 = (p/p0)^((gamma-1)/gamma)` (derived from (127.8)'s
        // isentropic law).
        let p32 = ((gamma * st.p - 1.0) * P_OP) as f32 as f64 + P_OP;
        let t32 = (T0 * (gamma * st.p).powf((gamma - 1.0) / gamma)) as f32 as f64;
        e.push(entropy_error(p32 / P_OP, t32 / T0, gamma));
    }
    weighted_l2(&e, &v)
}

/// **Gate CMP-RING** (SPEC-LIT §127.6, §127.6.7): the three meshes, each
/// marched from rest and held against the key through (127.29)'s `E`.
pub(crate) fn check_ringleb(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    let gamma = RINGLEB_GAMMA;
    let gas = PerfectGas {
        viscosity: Viscosity::Constant { mu: 0.0 },
        ..PerfectGas::air(P_OP)
    };
    let r_s = gas.r_s();
    let p_out_gauge = (static_over_stagnation(gamma, 0.5) - 1.0) * P_OP;
    c.note(&format!(
        "Gate CMP-RING: Ringleb's domain, walls on the streamlines k = 1.0 (inner) and k = 0.6 \
         (outer), the ends the iso-q lines q = 0.5; gamma {gamma}, p0 {P_OP} Pa (the gauge \
         datum), T0 {T0} K; the outlet's gauge p {p_out_gauge:.6} Pa, (127.8) at q_b = 0.5; \
         linearUpwind on momentum with no limiter, vanLeer on energy (the energy equation \
         refuses linearUpwind by name), vanLeer on the retarded density; LTS CFL {LTS_CFL}, \
         {N_OUTER} outer iteration, chunks of {CHUNK} iterations to a cell Mach change of \
         {FIELD_STOP:.1e}, {MAX_ITERS} iterations a mesh"
    ));

    let mut runs: Vec<Run> = Vec::new();
    let mut bounds: Vec<(usize, usize)> = Vec::new();
    for (nt, nk) in MESHES {
        let t0 = Instant::now();
        match run_once(gpu, gas, r_s, nt, nk) {
            Ok(r) => runs.push(r),
            Err(Error::Refused(why)) => {
                c.note(&format!(
                    "Gate CMP-RING {nt} x {nk}: the march at LTS CFL {LTS_CFL} stopped on a \
                     bound after {:.1} s, so this mesh has no field to measure: {why}",
                    t0.elapsed().as_secs_f64()
                ));
                bounds.push((nt, nk));
            }
            Err(e) => return Err(e),
        }
    }

    // Per mesh: the class D row (both builds), the notes, then E.
    let mut es: Vec<f64> = Vec::new();
    let mut hs: Vec<f64> = Vec::new();
    for r in &runs {
        let e = entropy_errors(r, gamma);
        let e_inf = e
            .iter()
            .fold(0.0f64, |m, v| if v.is_nan() || m.is_nan() { f64::NAN } else { m.max(v.abs()) });
        let big_e = weighted_l2(&e, &r.v);
        let finite = e.iter().all(|v| v.is_finite()) && big_e.is_finite();
        c.require(
            &format!(
                "Gate CMP-RING: the {} x {} run stops on no bound (SPEC-LIT 127.6)",
                r.nt, r.nk
            ),
            finite,
        );
        let m_max = r
            .mach
            .iter()
            .copied()
            .fold(0.0f64, |m, v| if v.is_nan() || m.is_nan() { f64::NAN } else { m.max(v) });
        let last = r.history.last().map_or(f64::NAN, |h| h.1);
        c.note(&format!(
            "Gate CMP-RING {} x {}: {} cells, h {:.6e}, {} iterations, field stop {}, last change \
             {last:.3e}, M_max {m_max:.6} against the key's 1.1180339887498948 on the inner wall, \
             {:.1} s; last: {}",
            r.nt,
            r.nk,
            r.v.len(),
            r.h,
            r.iterations,
            if r.converged { "met" } else { "NOT met" },
            r.seconds,
            r.last_line
        ));
        let p_l2 = printed_p_l2(r, gamma);
        c.note(&format!(
            "Gate CMP-RING {} x {}: E {:.6e} (94.1's weighted L2 of (127.29)), its L-infinity \
             {e_inf:.6e}; printed, never checked: the L2 of p/p0 - gamma p_key over the cells is {}",
            r.nt,
            r.nk,
            big_e,
            p_l2.map_or("n/a (the key refuses a centre)".to_string(), |v| format!("{v:.6e}"))
        ));
        if !r.converged {
            c.note(&format!(
                "Gate CMP-RING {} x {}: the solve did not converge: the field stop {FIELD_STOP:.0e} \
                 is NOT met in {} iterations, so E reads a field that still moves (SPEC-LIT 127.6.7)",
                r.nt,
                r.nk,
                r.iterations
            ));
        }
        es.push(big_e);
        hs.push(r.h);
    }
    for (nt, nk) in &bounds {
        c.require(
            &format!("Gate CMP-RING: the {nt} x {nk} run stops on no bound (SPEC-LIT 127.6)"),
            false,
        );
    }

    // The levels finest first, the pair orders, the fitted order and the
    // 94 study - all measured, whatever the build checks.
    let levels: Vec<Level> =
        hs.iter().zip(&es).map(|(h, e)| Level { h: s(*h), value: s(*e) }).collect();
    let study: std::result::Result<GridStudy, String> =
        vv::grid_study(&levels).map_err(|e| e.to_string());
    let (pair_0, pair_1, p_fit) = if runs.len() == 3 {
        (
            // MESHES is finest first: levels 1 and 2 are h/2 and h, levels
            // 0 and 1 are h/4 and h/2.
            pair_order(hs[1], es[1], hs[2], es[2]),
            pair_order(hs[0], es[0], hs[1], es[1]),
            fitted_order(&hs, &es),
        )
    } else {
        (f64::NAN, f64::NAN, f64::NAN)
    };
    let mesh_names: Vec<String> =
        runs.iter().map(|r| format!("{} x {}", r.nt, r.nk)).collect();
    let es_line = if es.is_empty() {
        "no field ran".to_string()
    } else {
        es.iter().map(|v| format!("{v:.4e}")).collect::<Vec<_>>().join(" / ")
    };
    c.note(&format!(
        "Gate CMP-RING: E = {es_line} on {} (finest first); pair orders {pair_0:.3} (h/2 vs h), \
         {pair_1:.3} (h/4 vs h/2); least-squares order {p_fit:.3}",
        if mesh_names.is_empty() { "no mesh".to_string() } else { mesh_names.join(" / ") }
    ));
    match &study {
        Ok(st) => c.note(&format!("Gate CMP-RING: 94 study of the L2 entropy levels: {}", st.one_line())),
        Err(e) => c.note(&format!("Gate CMP-RING: 94 study of the L2 entropy levels refused: {e}")),
    }

    // The row: f64 checks the order; f32 measures and reports its floor.
    let mut misses: Vec<String> = Vec::new();
    for (nt, nk) in &bounds {
        misses.push(format!("Gate CMP-RING {nt} x {nk}: no field, the march stopped on a bound"));
    }
    let order_label = "Gate CMP-RING: observed order of the L2 entropy error >= 1.5 across h, \
                       h/2, h/4 (SPEC-LIT 127.6)";
    if !cfg!(feature = "single") {
        // A NaN must FAIL, so the guard is written out: (1.5 - NaN).max(0.0)
        // is 0, which would read as a pass.
        let err = if p_fit.is_finite() { (ORDER_MIN - p_fit).max(0.0) } else { f64::NAN };
        c.check_p(order_label, s(err), 0.0);
        // `!(err <= 0.0)`, so that a NaN measure counts as the miss it is.
        if !(err <= 0.0) {
            misses.push(format!("{order_label}: {err:.4e} against 0.0"));
        }
    } else {
        let e_min = f32_floor(gamma);
        let e_at = |i: usize| -> String {
            es.get(i).map_or("n/a".to_string(), |v| {
                if v.is_finite() { format!("{v:.4e}") } else { "NaN".to_string() }
            })
        };
        c.note(&format!(
            "Gate CMP-RING f32: measured, not checked (SPEC-LIT 127.6: the order is f64 only): \
             E = {} / {} / {}, fitted order {p_fit:.3}, the f32 error floor is E_min = {e_min:.3e} \
             ((127.29) on the key's own p, T at the finest mesh's centres after a round trip \
             through f32 - the smallest E an f32 field can carry there)",
            e_at(0),
            e_at(1),
            e_at(2)
        ));
    }
    if misses.is_empty() {
        return Ok(());
    }
    let iterations: Vec<String> = runs.iter().map(|r| r.iterations.to_string()).collect();
    let uncertainty = match &study {
        Ok(st) => Uncertainty::Study(st.clone()),
        Err(_) => Uncertainty::SingleMesh("the 94 study refused the three L2 entropy levels"),
    };
    let iters_line = if iterations.is_empty() {
        "none ran".to_string()
    } else {
        iterations.join(" / ")
    };
    c.report(GateReport {
        verdict: Verdict::Misses,
        how: How::Live,
        gate: GATE,
        against: AGAINST,
        headline: format!(
            "the L2 entropy error reads E = {es_line} (finest first, {}); pair orders \
             {pair_0:.3} (h/2 vs h), {pair_1:.3} (h/4 vs h/2), fitted order {p_fit:.3} against \
             the bar {ORDER_MIN}; iterations {iters_line}",
            if mesh_names.is_empty() {
                "no mesh ran".to_string()
            } else {
                mesh_names.join(" / ")
            },
        ),
        detail: vec![format!(
            "three meshes of SPEC-LIT 127.5.4 at LTS CFL {LTS_CFL}, {N_OUTER} outer iteration, \
             chunks of {CHUNK} iterations to a cell Mach change of {FIELD_STOP:.1e}, \
             {MAX_ITERS} iterations a mesh: every measured value is in the rows and notes above"
        )],
        uncertainty: Some(uncertainty),
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close_abs(got: f64, want: f64, tol: f64, what: &str) {
        assert!((got - want).abs() <= tol, "{what}: {got} is not {want}");
    }

    /// (127.29) is 0 on the key: at `p = p0`, `T = T0` exactly, and at the
    /// sonic pair `p/p0 = c^3.5`, `T/T0 = c` of (127.8) to round-off; and
    /// the hand value at (0.9, 0.95).
    #[test]
    fn the_entropy_error() {
        assert!((entropy_error(1.0, 1.0, 1.4)).abs() <= 1.0e-15);
        let c2: f64 = 1.0 - 0.2 * 0.36;
        assert!((entropy_error(c2.powf(3.5), c2, 1.4)).abs() <= 1.0e-14);
        close_abs(entropy_error(0.9, 0.95, 1.4), -0.029230677523138016, 1.0e-14, "e(0.9, 0.95)");
    }

    /// (94.1)'s weighted L2, its hand value, and its two NaN refusals.
    #[test]
    fn the_weighted_l2() {
        close_abs(
            weighted_l2(&[0.1, -0.2, 0.3], &[1.0, 1.0, 2.0]),
            0.23979157616563598,
            1.0e-15,
            "the weighted L2",
        );
        assert!(weighted_l2(&[], &[]).is_nan(), "an empty slice is NaN");
        assert!(weighted_l2(&[0.1], &[0.0]).is_nan(), "zero weights are NaN");
    }

    /// (127.8)'s static over stagnation at the outlet's speed `q_b = 0.5`:
    /// the outlet's gauge pressure.
    #[test]
    fn the_outlet_pressure() {
        close_abs(static_over_stagnation(1.4, 0.5), 0.8356657801380584, 1.0e-15, "p/p0(0.5)");
        close_abs(
            (static_over_stagnation(1.4, 0.5) - 1.0) * P_OP,
            -16433.421986194164,
            1.0e-9,
            "the outlet's gauge p",
        );
    }

    /// The pair order of a clean second-order law is 2.
    #[test]
    fn the_pair_order() {
        close_abs(pair_order(0.1, 2.5e-3, 0.2, 1.0e-2), 2.0, 1.0e-14, "the pair order");
    }
}

