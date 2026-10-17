// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate CMP-SOD (SPEC-LIT §127.6, §127.6.1): `ofgpu::compressible` on
//! Sod's shock tube, against the exact Riemann solution, live.**
//!
//! Sod's problem, `(rho, u, p) = (1, 0, 1 | 0.125, 0, 0.1)` with
//! `gamma = 1.4` and the diaphragm at `x_0 = 0.5` of a unit tube, is run to
//! `t = 0.2` (non-dimensional) on 200, 400 and 800 cells with PISO at an
//! acoustic CFL of 0.5, and held against the exact solution of (127.5) to
//! (127.7b), evaluated live by `compressible::exact::riemann` (CMP-07). The
//! rows, all class P (the same bar in both builds, docs/17 section 5.1.4): on 800
//! cells the two density plateaus, the star velocity and the star pressure
//! within 1 % and the shock position within 0.5 % of the length; across the
//! three meshes an observed L1(`rho`) order `>= 0.6`. A miss fails its row
//! and registers the verdict (§69), with the measured values; the same three
//! meshes are then run with PIMPLE (five outer iterations, `p` unrelaxed)
//! and printed beside it, never checked, because that leg isolates the
//! cause (§127.6.1): the row's own setup is PISO and stays PISO.
//!
//! The run is dimensional air, scaled: `p_L = 1e5` Pa, `rho_L = 1` kg/m^3,
//! `T = p/(rho R_s)`, the velocity unit `sqrt(p_L/rho_L)` and the time unit
//! `L/sqrt(p_L/rho_L)`, so the non-dimensional problem is Sod's exactly and
//! the comparison is made in it. Every reduction is f64 whatever `Scalar`
//! is: the fields are widened once on download.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §127.2 (the exact Riemann problem, (127.5)-(127.7b),
//!     and its oracle values), §127.6 (the gate table) and §127.6.1 (this
//!     gate as built), §125.3 (the outer loop), §94 (the mesh study)
//!   G. A. Sod, *J. Comput. Phys.* 27 (1978) 1-31,
//!     doi:10.1016/0021-9991(78)90023-2 - the problem's two states
//!   E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
//!     3rd ed., Springer (2009) - cited by docs/17 for the key, which §127.2
//!     derives from NACA 1135 instead (not read)
//! No GPL-licensed source was consulted.

use std::collections::BTreeMap;
use std::time::Instant;

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::compressible::bc as cbc;
use ofgpu::compressible::exact::riemann::{self, Primitive, RiemannSolution, Wave};
use ofgpu::compressible::pressure::Retarded;
use ofgpu::compressible::thermo::{PerfectGas, Viscosity};
use ofgpu::compressible::{Compressible, CompressibleControls, CompressibleState};
use ofgpu::field::{GpuScalarField, GpuVectorField};
use ofgpu::field_setup::{setup_scalar_field, setup_vector_field};
use ofgpu::fv::{DivScheme, Limiter};
use ofgpu::io::case::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::mesh::HostMesh;
use ofgpu::timescheme::DdtScheme;
use ofgpu::vv::{self, GridStudy, Level};
use ofgpu::{Error, Gpu, GpuMesh, Result, Scalar, Vec3};

use crate::{Checks, GateReport, How, Uncertainty, Verdict};

/// The gate's scope and registry name.
pub(crate) const GATE: &str = "SPEC-LIT 127.6 Gate CMP-SOD Sod shock tube";

const AGAINST: &str = "the exact Riemann solution of Sod's (1978) shock tube, (127.5)-(127.7b), \
                       evaluated live (CMP-07)";

/// The three meshes, finest first.
pub(crate) const MESHES: [usize; 3] = [800, 400, 200];
/// Sod's two states, non-dimensional, and the diaphragm.
pub(crate) const LEFT: Primitive = Primitive { rho: 1.0, u: 0.0, p: 1.0 };
pub(crate) const RIGHT: Primitive = Primitive { rho: 0.125, u: 0.0, p: 0.1 };
const GAMMA: f64 = 1.4;
const X0: f64 = 0.5;
/// The end time, non-dimensional.
pub(crate) const T_END: f64 = 0.2;
/// The acoustic CFL of the row.
const CFL: f64 = 0.5;
/// The dimensional scales: `p_L` [Pa] and `rho_L` [kg/m^3]. The gauge
/// datum is `p_L`, so the gauge `p` lies in `[-0.9 p_L, 0]`.
const P_REF: f64 = 1.0e5;
const RHO_REF: f64 = 1.0;

/// The row's bars, the plan's, fixed before measurement.
pub(crate) const PLATEAU_TOL: f64 = 0.01;
pub(crate) const SHOCK_TOL: f64 = 0.005;
pub(crate) const ORDER_MIN: f64 = 0.6;

fn w(x: Scalar) -> f64 {
    f64::from(x)
}

fn s(x: f64) -> Scalar {
    x as Scalar
}

/// The tube: `n` cells on `[0, 1]` m, one cell across a `0.01` m square,
/// the four sides `symmetry`.
fn tube(n: usize) -> Result<HostMesh> {
    let axis = |hi: Scalar, n: usize| GradedAxis { lo: 0.0, hi, n, expansion: 1.0, two_sided: false };
    let spec = BlockSpec {
        x: axis(1.0, n),
        y: axis(0.01, 1),
        z: axis(0.01, 1),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["patch", "patch", "symmetry", "symmetry", "symmetry", "symmetry"].map(String::from),
        cyclic: Vec::new(),
    };
    blockgen::build_mesh(&spec)
}

fn entry(t: &str) -> PatchFieldSpec {
    PatchFieldSpec { type_name: t.into(), ..PatchFieldSpec::default() }
}

/// The two ends `ends`, the four sides `symmetry`.
fn with_sides(mut b: BTreeMap<String, PatchFieldSpec>, ends: PatchFieldSpec) -> BTreeMap<String, PatchFieldSpec> {
    b.insert("xMin".into(), ends.clone());
    b.insert("xMax".into(), ends);
    for side in ["yMin", "yMax", "zMin", "zMax"] {
        b.insert(side.into(), entry("symmetry"));
    }
    b
}

/// The largest `|u| + c` of the exact solution: the four constant states
/// and the fan's ends bound it (inside a left fan `u + c` rises from head
/// to tail).
pub(crate) fn max_signal_speed(sol: &RiemannSolution) -> f64 {
    let c = |rho: f64, p: f64| (GAMMA * p / rho).sqrt();
    [
        LEFT.u.abs() + c(LEFT.rho, LEFT.p),
        sol.u_star.abs() + c(sol.rho_star_left, sol.p_star),
        sol.u_star.abs() + c(sol.rho_star_right, sol.p_star),
        RIGHT.u.abs() + c(RIGHT.rho, RIGHT.p),
    ]
    .into_iter()
    .fold(0.0, f64::max)
}

/// The steps of an `n`-cell run: the fewest whose `dt` keeps the acoustic
/// CFL at or below [`CFL`], with `dt = T_END/steps` so the run ends at
/// `T_END` exactly (non-dimensional).
pub(crate) fn steps_for(n: usize, s_max: f64) -> usize {
    let dx = 1.0 / n as f64;
    (T_END / (CFL * dx / s_max)).ceil() as usize
}

/// The exact solution's wave positions at `T_END`: the left fan's tail,
/// the contact and the right shock (Sod has exactly that wave pair).
pub(crate) fn waves(sol: &RiemannSolution) -> Result<(f64, f64, f64)> {
    let tail = match sol.left_wave {
        Wave::Rarefaction { tail, .. } => tail,
        Wave::Shock { .. } => return Err(Error::Config("Gate CMP-SOD: Sod's left wave should be a fan, the key gives a shock".into())),
    };
    let shock = match sol.right_wave {
        Wave::Shock { speed } => speed,
        Wave::Rarefaction { .. } => return Err(Error::Config("Gate CMP-SOD: Sod's right wave should be a shock, the key gives a fan".into())),
    };
    Ok((X0 + tail * T_END, X0 + sol.u_star * T_END, X0 + shock * T_END))
}

/// The pressure-velocity coupling of a run: the row's PISO, or the PIMPLE
/// leg that shows the cause of a PISO miss (§127.6.1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Coupling {
    /// `nOuterCorrectors`.
    pub n_outer: usize,
    /// `nCorrectors`.
    pub n_corr: usize,
    /// `alpha_p` on the non-final outer iterations (§125.3 step 6).
    pub p_relax: Scalar,
}

/// The row's PISO: one outer iteration, two correctors (§125.3).
pub(crate) const PISO: Coupling = Coupling { n_outer: 1, n_corr: 2, p_relax: 0.3 };
/// The cause leg: five outer iterations with `p` unrelaxed, so each step's
/// energy equation sees that step's `dp/dt` and `K` (§127.6.1).
pub(crate) const PIMPLE: Coupling = Coupling { n_outer: 5, n_corr: 2, p_relax: 1.0 };

/// One run's non-dimensional cell values, and its cost.
pub(crate) struct Run {
    pub n: usize,
    pub x: Vec<f64>,
    pub rho: Vec<f64>,
    pub u: Vec<f64>,
    pub p: Vec<f64>,
    pub steps: usize,
    pub seconds: f64,
}

/// Sod's problem on `n` cells, run to `T_END` (§127.6.1's setup).
fn run(gpu: &Gpu, n: usize, s_max: f64, coupling: Coupling) -> Result<Run> {
    let t0 = Instant::now();
    let gas = PerfectGas { viscosity: Viscosity::Constant { mu: 0.0 }, ..PerfectGas::air(P_REF) };
    let rs = gas.r_s();
    let u_ref = (P_REF / RHO_REF).sqrt();
    let hm = tube(n)?;
    let gm = GpuMesh::upload(gpu, &hm)?;
    let state = |i: usize| if w(hm.c[i].x) < X0 { LEFT } else { RIGHT };

    // The gauge p about p_op = p_L, and T from the gas law, (122.1).
    let mut rp = RawScalarField { name: "p".into(), ..RawScalarField::default() };
    rp.internal = (0..hm.n_cells).map(|i| s(state(i).p * P_REF - P_REF)).collect();
    rp.boundary = with_sides(rp.boundary, entry("zeroGradient"));
    let mut rt = RawScalarField { name: "T".into(), ..RawScalarField::default() };
    rt.internal = (0..hm.n_cells).map(|i| s(state(i).p * P_REF / (state(i).rho * RHO_REF * rs))).collect();
    rt.boundary = with_sides(rt.boundary, entry("zeroGradient"));
    let mut ru = RawVectorField { name: "U".into(), ..RawVectorField::default() };
    ru.internal = vec![Vec3::ZERO; hm.n_cells];
    let wall = PatchFieldSpec { type_name: "fixedValue".into(), value_v: vec![Vec3::ZERO], ..PatchFieldSpec::default() };
    ru.boundary = with_sides(ru.boundary, wall);

    let mut ctrl = CompressibleControls { gas, ..CompressibleControls::default() };
    let lin = SolverControls {
        solver: LinearSolverKind::PBiCGStab,
        precon: Preconditioner::Dilu,
        tolerance: 1.0e-12,
        rel_tol: 0.0,
        max_iter: 200,
        ..Default::default()
    };
    ctrl.momentum.u_solver = lin;
    ctrl.pressure.p_solver = lin;
    ctrl.energy.t_solver = lin;
    let steps = steps_for(n, s_max);
    let dt = T_END / u_ref / steps as f64;
    for (ddt, d) in [(&mut ctrl.momentum.ddt, &mut ctrl.momentum.delta_t), (&mut ctrl.energy.ddt, &mut ctrl.energy.delta_t)] {
        *ddt = DdtScheme::Euler;
        *d = s(dt);
    }
    ctrl.momentum.div_scheme = DivScheme::Limited(Limiter::VanLeer);
    ctrl.energy.div_scheme = DivScheme::Limited(Limiter::VanLeer);
    ctrl.pressure.retarded = Retarded::Limited(Limiter::VanLeer);
    ctrl.momentum.u_relax = 1.0;
    ctrl.energy.t_relax = 1.0;
    ctrl.n_outer = coupling.n_outer;
    ctrl.n_corr = coupling.n_corr;
    ctrl.p_relax = coupling.p_relax;

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
    solver.initialise(gpu, &mut st)?;
    // A bound (T, p_abs or rho <= 0) stops `step` with Error::Refused (§122.3).
    solver.run(gpu, &mut st, steps)?;

    let rho: Vec<f64> = gpu.download(&st.rho.f)?.iter().map(|v| w(*v) / RHO_REF).collect();
    let u: Vec<f64> = gpu.download(&st.u.f)?.iter().map(|v| w(v.x) / u_ref).collect();
    let p: Vec<f64> = gpu.download(&st.p.f)?.iter().map(|v| (w(*v) + P_REF) / P_REF).collect();
    let x: Vec<f64> = (0..hm.n_cells).map(|i| w(hm.c[i].x)).collect();
    Ok(Run { n, x, rho, u, p, steps, seconds: t0.elapsed().as_secs_f64() })
}

/// `max |v_i - key|/|key|` over the cells whose centre lies in the middle
/// half of each window `[a, b]`, and the number of those cells. A
/// non-finite cell makes the result NaN.
pub(crate) fn plateau_error(x: &[f64], v: &[f64], key: f64, windows: &[(f64, f64)]) -> (f64, usize) {
    let mut worst = 0.0f64;
    let mut cells = 0usize;
    for &(a, b) in windows {
        let (lo, hi) = (a + 0.25 * (b - a), b - 0.25 * (b - a));
        for (xi, vi) in x.iter().zip(v) {
            if *xi > lo && *xi < hi {
                cells += 1;
                let e = (vi - key).abs() / key.abs();
                worst = if e.is_nan() || worst.is_nan() { f64::NAN } else { worst.max(e) };
            }
        }
    }
    (worst, cells)
}

/// The numerical shock: scanning from the right end, the first pair of
/// cells across which `rho` rises through `level`, located by linear
/// interpolation between the two centres. `None` when no pair does.
pub(crate) fn shock_position(x: &[f64], rho: &[f64], level: f64) -> Option<f64> {
    for i in (0..x.len().saturating_sub(1)).rev() {
        if rho[i] >= level && rho[i + 1] < level {
            let f = (rho[i] - level) / (rho[i] - rho[i + 1]);
            return Some(x[i] + f * (x[i + 1] - x[i]));
        }
    }
    None
}

/// `L1(rho) = sum_i |rho_i - rho(x_i, T_END)| dx` on the unit tube.
pub(crate) fn l1_rho(r: &Run, sol: &RiemannSolution) -> Result<f64> {
    let dx = 1.0 / r.n as f64;
    let mut e = 0.0;
    for (xi, ri) in r.x.iter().zip(&r.rho) {
        e += (ri - sol.sample_at(*xi, T_END, X0)?.rho).abs() * dx;
    }
    Ok(e)
}

/// The total energy `sum_i (p_i/(gamma - 1) + rho_i u_i^2/2) dx` of a run
/// over its initial value, minus one. The tube is closed, adiabatic and
/// inviscid with walls at rest, so the exact value is 0.
pub(crate) fn energy_drift(r: &Run) -> f64 {
    let dx = 1.0 / r.n as f64;
    let e = |rho: f64, u: f64, p: f64| (p / (GAMMA - 1.0) + 0.5 * rho * u * u) * dx;
    let now: f64 = (0..r.n).map(|i| e(r.rho[i], r.u[i], r.p[i])).sum();
    let start: f64 = r.x.iter().map(|x| if *x < X0 { e(LEFT.rho, LEFT.u, LEFT.p) } else { e(RIGHT.rho, RIGHT.u, RIGHT.p) }).sum();
    now / start - 1.0
}

/// The least-squares slope of `ln E` on `ln h`: the observed order across
/// the levels.
pub(crate) fn fitted_order(h: &[f64], e: &[f64]) -> f64 {
    let n = h.len() as f64;
    let lx: Vec<f64> = h.iter().map(|v| v.ln()).collect();
    let ly: Vec<f64> = e.iter().map(|v| v.ln()).collect();
    let (mx, my) = (lx.iter().sum::<f64>() / n, ly.iter().sum::<f64>() / n);
    let sxy: f64 = lx.iter().zip(&ly).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = lx.iter().map(|a| (a - mx) * (a - mx)).sum();
    sxy / sxx
}

/// What one leg measured: on the finest mesh the four plateaus (name,
/// error, cells) and the shock, over the three meshes `L1(rho)`, its order
/// and the total-energy drift.
pub(crate) struct Measures {
    pub plateaus: [(&'static str, f64, usize); 4],
    pub shock_x: Option<f64>,
    pub shock_err: f64,
    pub h: [f64; 3],
    pub l1: [f64; 3],
    pub order: f64,
    pub drift: [f64; 3],
    pub finite: [bool; 3],
}

impl Measures {
    /// The worst plateau (name, error); NaN is the worst.
    pub fn worst_plateau(&self) -> (&'static str, f64) {
        let mut out = ("", 0.0f64);
        for (name, e, _) in self.plateaus {
            if !(e <= out.1) {
                out = (name, e);
            }
        }
        out
    }

    /// Every bar of the row met.
    pub fn passes(&self) -> bool {
        self.worst_plateau().1 <= PLATEAU_TOL && self.shock_err <= SHOCK_TOL && self.order >= ORDER_MIN
    }

    /// The pair order between levels `a` (finer) and `b`.
    pub fn pair_order(&self, a: usize, b: usize) -> f64 {
        (self.l1[b] / self.l1[a]).ln() / (self.h[b] / self.h[a]).ln()
    }
}

/// The shock row in words: the offset in % of the length, or that no
/// shock was found inside the tube.
fn shock_words(m: &Measures) -> String {
    if m.shock_x.is_some() {
        format!("{:.3} % of the length off", 100.0 * m.shock_err)
    } else {
        "not found inside the tube".to_string()
    }
}

/// Run one leg on the three meshes and measure it.
fn leg(c: &mut Checks, gpu: &Gpu, sol: &RiemannSolution, label: &str, coupling: Coupling) -> Result<Measures> {
    let s_max = max_signal_speed(sol);
    let (x_tail, x_contact, x_shock) = waves(sol)?;
    let mut runs = Vec::new();
    for n in MESHES {
        let r = run(gpu, n, s_max, coupling)?;
        c.note(&format!(
            "Gate CMP-SOD {label}, {n} cells: {} steps of dt = {:.4e} (acoustic CFL {:.4}), {:.1} s",
            r.steps,
            T_END / r.steps as f64,
            s_max * T_END / r.steps as f64 * n as f64,
            r.seconds
        ));
        runs.push(r);
    }
    let fine = &runs[0];
    let both = [(x_tail, x_contact), (x_contact, x_shock)];
    let pl = |name: &'static str, v: &[f64], key: f64, win: &[(f64, f64)]| {
        let (e, cells) = plateau_error(&fine.x, v, key, win);
        (name, e, cells)
    };
    let plateaus = [
        pl("rho*_L", &fine.rho, sol.rho_star_left, &both[..1]),
        pl("rho*_R", &fine.rho, sol.rho_star_right, &both[1..]),
        pl("u*", &fine.u, sol.u_star, &both),
        pl("p*", &fine.p, sol.p_star, &both),
    ];
    let shock_x = shock_position(&fine.x, &fine.rho, 0.5 * (sol.rho_star_right + RIGHT.rho));
    let shock_err = shock_x.map_or(f64::INFINITY, |v| (v - x_shock).abs());
    let mut l1 = [0.0; 3];
    let mut h = [0.0; 3];
    let mut drift = [0.0; 3];
    let mut finite = [false; 3];
    for (k, r) in runs.iter().enumerate() {
        h[k] = 1.0 / r.n as f64;
        l1[k] = l1_rho(r, sol)?;
        drift[k] = energy_drift(r);
        finite[k] = r.rho.iter().chain(&r.u).chain(&r.p).all(|v| v.is_finite());
    }
    let order = fitted_order(&h, &l1);
    let m = Measures { plateaus, shock_x, shock_err, h, l1, order, drift, finite };
    for (name, e, cells) in m.plateaus {
        c.note(&format!("Gate CMP-SOD {label}, 800 cells: plateau {name} worst relative error {e:.4e} over {cells} cells"));
    }
    c.note(&format!(
        "Gate CMP-SOD {label}, 800 cells: shock (rho through (rho*_R + rho_R)/2) at x {} against {x_shock:.6}",
        m.shock_x.map_or("none".to_string(), |v| format!("{v:.6}"))
    ));
    c.note(&format!(
        "Gate CMP-SOD {label}: L1(rho) {:.4e} / {:.4e} / {:.4e} on 800 / 400 / 200 cells; pair orders {:.3} (400/800), \
         {:.3} (200/400); least-squares order {:.3}",
        m.l1[0],
        m.l1[1],
        m.l1[2],
        m.pair_order(0, 1),
        m.pair_order(1, 2),
        m.order
    ));
    c.note(&format!(
        "Gate CMP-SOD {label}: total energy drift at t = 0.2 (exact 0) {:+.3e} / {:+.3e} / {:+.3e} on 800 / 400 / 200 cells",
        m.drift[0], m.drift[1], m.drift[2]
    ));
    Ok(m)
}

/// **Gate CMP-SOD** (SPEC-LIT §127.6, §127.6.1): the row's PISO leg is
/// checked; the PIMPLE leg is printed beside it as the cause of a miss.
pub(crate) fn check_sod(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    let sol = riemann::solve(GAMMA, LEFT, RIGHT)?;
    let (x_tail, x_contact, x_shock) = waves(&sol)?;
    c.note(&format!(
        "Gate CMP-SOD key (127.5)-(127.7b): p* {:.6}, u* {:.6}, rho*_L {:.6}, rho*_R {:.6}; at t = 0.2 the fan tail \
         at x {x_tail:.6}, the contact at {x_contact:.6}, the shock at {x_shock:.6}; largest |u|+c {:.6}",
        sol.p_star,
        sol.u_star,
        sol.rho_star_left,
        sol.rho_star_right,
        max_signal_speed(&sol)
    ));

    // ---- the row: PISO ---------------------------------------------------------
    let piso = leg(c, gpu, &sol, "PISO (nOuterCorrectors 1, nCorrectors 2)", PISO)?;
    for (k, n) in MESHES.iter().enumerate() {
        c.require(&format!("Gate CMP-SOD: the {n}-cell PISO run is finite at t = 0.2, no bound counted"), piso.finite[k]);
    }
    for (name, e, _) in piso.plateaus {
        c.check_p(&format!("Gate CMP-SOD: plateau {name} within 1 %, 800 cells (SPEC-LIT 127.6)"), s(e), s(PLATEAU_TOL));
    }
    c.check_p("Gate CMP-SOD: shock position within 0.5 % of the length, 800 cells (SPEC-LIT 127.6)", s(piso.shock_err), s(SHOCK_TOL));
    c.check_p(
        "Gate CMP-SOD: observed L1(rho) order >= 0.6 across 200/400/800 cells (SPEC-LIT 127.6)",
        s((ORDER_MIN - piso.order).max(0.0)),
        0.0,
    );
    let levels: Vec<Level> = piso.h.iter().zip(&piso.l1).map(|(hv, ev)| Level { h: s(*hv), value: s(*ev) }).collect();
    let study: std::result::Result<GridStudy, String> = vv::grid_study(&levels).map_err(|e| e.to_string());
    match &study {
        Ok(st) => c.note(&format!("Gate CMP-SOD PISO: 94 study of L1(rho): {}", st.one_line())),
        Err(e) => c.note(&format!("Gate CMP-SOD PISO: 94 study of L1(rho) refused: {e}")),
    }
    if piso.passes() {
        return Ok(());
    }

    // ---- the cause leg: PIMPLE, printed only -------------------------------------
    let pimple = leg(c, gpu, &sol, "PIMPLE (nOuterCorrectors 5, alpha_p 1), printed only", PIMPLE)?;
    let worst = piso.worst_plateau();
    let pw = pimple.worst_plateau();
    let uncertainty = match study {
        Ok(st) => Uncertainty::Study(st),
        Err(_) => Uncertainty::SingleMesh("the 94 study refused the three L1(rho) levels; the 800-cell rows alone"),
    };
    c.report(GateReport {
        verdict: Verdict::Misses,
        how: How::Live,
        gate: GATE,
        against: AGAINST,
        headline: format!(
            "PISO, 800 cells: worst plateau {} {:.2} % (bar 1 %), shock {} (bar 0.5 % of the length); \
             L1(rho) order {:.3} (bar >= 0.6)",
            worst.0,
            100.0 * worst.1,
            shock_words(&piso),
            piso.order
        ),
        detail: vec![
            format!(
                "PISO total energy drift {:+.2} % on 800 cells: the energy equation (125.1) is solved before the \
                 correctors, so in a one-outer step its dp/dt reads p - p^o = 0 and its K the predicted u (SPEC-LIT §127.6.1)",
                100.0 * piso.drift[0]
            ),
            format!(
                "the same meshes with PIMPLE (5 outer iterations, alpha_p 1; inputs only): worst plateau {} {:.2} %, \
                 shock {}, order {:.3}, energy drift {:+.2e}",
                pw.0,
                100.0 * pw.1,
                shock_words(&pimple),
                pimple.order,
                pimple.drift[0]
            ),
        ],
        uncertainty: Some(uncertainty),
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The step count and the CFL bound come from the key: the largest
    /// `|u| + c` is `u* + c*_R` (§127.6.1), and 800 cells take 702 steps.
    #[test]
    fn the_step_comes_from_the_key() {
        let sol = riemann::solve(GAMMA, LEFT, RIGHT).expect("Sod solves");
        let s_max = max_signal_speed(&sol);
        assert!((s_max - 2.191566).abs() < 5e-7, "largest |u|+c {s_max}");
        assert_eq!([steps_for(800, s_max), steps_for(400, s_max), steps_for(200, s_max)], [702, 351, 176]);
        let (tail, contact, shock) = waves(&sol).expect("a fan and a shock");
        assert!((contact - 0.685491).abs() < 5e-7 && (shock - 0.850431).abs() < 5e-7, "{contact} {shock}");
        assert!((tail - (0.5 - 0.07027281256118327 * 0.2)).abs() < 1e-12, "{tail}");
    }

    /// The measures on hand-made profiles: a step located between two
    /// centres, a power law's order, a plateau window's middle half, and the
    /// energy of the initial state.
    #[test]
    fn the_measures_read_hand_made_profiles() {
        let x = [0.1, 0.2, 0.3, 0.4];
        let rho = [1.0, 1.0, 0.5, 0.0];
        let xs = shock_position(&x, &rho, 0.75).expect("a crossing");
        assert!((xs - 0.25).abs() < 1e-15, "{xs}");
        assert_eq!(shock_position(&x, &[0.0, 0.0, 0.0, 0.0], 0.75), None);
        let h: [f64; 3] = [0.00125, 0.0025, 0.005];
        let e: Vec<f64> = h.iter().map(|v| 3.0 * v.powf(0.7)).collect();
        assert!((fitted_order(&h, &e) - 0.7).abs() < 1e-12);
        // Window [0, 0.8]: its middle half (0.2, 0.6) holds only 0.3 and 0.4.
        let (err, cells) = plateau_error(&x, &[9.0, 9.0, 1.01, 0.98], 1.0, &[(0.0, 0.8)]);
        assert_eq!(cells, 2);
        assert!((err - 0.02).abs() < 1e-12, "{err}");
        let (nan, _) = plateau_error(&x, &[1.0, 1.0, f64::NAN, 1.0], 1.0, &[(0.0, 0.8)]);
        assert!(nan.is_nan());
        let r = Run {
            n: 4,
            x: vec![0.125, 0.375, 0.625, 0.875],
            rho: vec![1.0, 1.0, 0.125, 0.125],
            u: vec![0.0; 4],
            p: vec![1.0, 1.0, 0.1, 0.1],
            steps: 0,
            seconds: 0.0,
        };
        assert!(energy_drift(&r).abs() < 1e-15);
    }
}
