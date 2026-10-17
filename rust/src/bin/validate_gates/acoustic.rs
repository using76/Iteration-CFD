// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate CMP-AC (SPEC-LIT §127.6, §127.6.2): the speed of a small
//! acoustic pulse in `ofgpu::compressible`, against linear acoustics, live.**
//!
//! A right-running Gaussian pulse, `p' = A exp(-((x - x0)/W)^2)` with
//! `A = 1e-3 p`, `W = 10` cells and the simple-wave `u' = p'/(rho_0 c_0)`
//! and isentropic `T'`, crosses a closed 400-cell tube of air at rest at
//! `T_0 = 300` K. Its peak is located at two times by a parabola through
//! the largest cell and its two neighbours, and the speed between them is
//! held against (122.4)'s `c_0 = sqrt(gamma R_s T_0)`: within 1 %, class P
//! (the same bar in both builds, docs/17 section 5.1.4). The row's coupling is
//! the solver's default (`nOuterCorrectors 1`, `nCorrectors 2`); when it
//! misses, the same tube is run with PIMPLE (five outer iterations, `p`
//! unrelaxed) and printed beside it, never checked (§127.6.2).
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §122.4 (the speed of sound, (122.4)), §125.3 (the
//!     outer loop), §126.9 (Gate CMP-NR's pulse and its measured speed),
//!     §127.6 (the gate table) and §127.6.2 (this gate as built)
//!   linear acoustics: a small isentropic disturbance of a gas at rest
//!     travels at `c_0`, and a simple wave carries `u' = p'/(rho_0 c_0)`
//!     ((127.20), derived in §127.6.2 from the linearised Euler equations)
//! No GPL-licensed source was consulted.

use std::collections::BTreeMap;
use std::time::Instant;

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::compressible::bc as cbc;
use ofgpu::compressible::thermo::{PerfectGas, Viscosity};
use ofgpu::compressible::{Compressible, CompressibleControls, CompressibleState};
use ofgpu::field::{GpuScalarField, GpuVectorField};
use ofgpu::field_setup::{setup_scalar_field, setup_vector_field};
use ofgpu::io::case::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::mesh::HostMesh;
use ofgpu::timescheme::DdtScheme;
use ofgpu::{Gpu, GpuMesh, Result, Scalar, Vec3};

use crate::{Checks, GateReport, How, Uncertainty, Verdict};

/// The gate's scope and registry name.
pub(crate) const GATE: &str = "SPEC-LIT 127.6 Gate CMP-AC acoustic pulse speed";

const AGAINST: &str = "linear acoustics: c_0 = sqrt(gamma R_s T_0), (122.4)";

/// The row's mesh: 400 cells on a 1 m tube.
pub(crate) const CELLS: usize = 400;
/// The pulse: amplitude over `p_op`, e-folding half-width in cells, centre [m].
const AMP: f64 = 1.0e-3;
const WIDTH_CELLS: f64 = 10.0;
const X0: f64 = 0.25;
/// The rest state.
const P_OP: f64 = 1.0e5;
const T0: f64 = 300.0;
/// The acoustic CFL on `c_0`, and the two sample steps: the peak travels
/// 0.1 m by the first and 0.5 m by the second, 100 cells short of the wall.
pub(crate) const CFL: f64 = 0.2;
pub(crate) const STEP_A: usize = 200;
pub(crate) const STEP_B: usize = 1000;
/// The row's bar, the plan's, fixed before measurement.
pub(crate) const SPEED_TOL: f64 = 0.01;

fn w(x: Scalar) -> f64 {
    f64::from(x)
}

fn s(x: f64) -> Scalar {
    x as Scalar
}

/// The tube: `n` cells on `[0, 1]` m, one cell across a `1/n` m square,
/// the four sides `symmetry`.
fn tube(n: usize) -> Result<HostMesh> {
    let side = s(1.0 / n as f64);
    let axis = |hi: Scalar, n: usize| GradedAxis { lo: 0.0, hi, n, expansion: 1.0, two_sided: false };
    let spec = BlockSpec {
        x: axis(1.0, n),
        y: axis(side, 1),
        z: axis(side, 1),
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

/// The peak of a sampled profile: a parabola through the largest sample
/// and its two neighbours, on a uniform spacing `dx` with sample `i` at
/// `x[i]`. `None` when the largest sample is at an end or the three are
/// not a maximum.
pub(crate) fn peak_position(x: &[f64], v: &[f64]) -> Option<f64> {
    if v.len() < 3 {
        return None;
    }
    let k = (0..v.len()).max_by(|a, b| v[*a].total_cmp(&v[*b]))?;
    if k == 0 || k == v.len() - 1 {
        return None;
    }
    let (a, b, c) = (v[k - 1], v[k], v[k + 1]);
    let curv = a - 2.0 * b + c;
    if !(curv < 0.0) {
        return None;
    }
    let dx = x[k + 1] - x[k];
    Some(x[k] + 0.5 * (a - c) / curv * dx)
}

/// The pressure-velocity coupling of a run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Coupling {
    pub n_outer: usize,
    pub n_corr: usize,
    pub p_relax: Scalar,
}

/// The row's coupling: the solver's default (§125.3).
pub(crate) fn row_coupling() -> Coupling {
    let d = CompressibleControls::default();
    Coupling { n_outer: d.n_outer, n_corr: d.n_corr, p_relax: d.p_relax }
}

/// The cause leg: five outer iterations with `p` unrelaxed (§127.6.1).
pub(crate) const PIMPLE: Coupling = Coupling { n_outer: 5, n_corr: 2, p_relax: 1.0 };

/// What one run measured.
pub(crate) struct Run {
    /// The peak at [`STEP_A`] and [`STEP_B`] [m].
    pub x_a: Option<f64>,
    pub x_b: Option<f64>,
    /// `(x_b - x_a)/((STEP_B - STEP_A) dt)` over `c_0`.
    pub ratio: f64,
    /// The largest `|p'|` left of `x0` at [`STEP_B`], over `A`: what the
    /// simple-wave start sent the wrong way.
    pub left: f64,
    /// The peak `p'` at [`STEP_B`] over `A`.
    pub height: f64,
    pub c0: f64,
    pub dt: f64,
    pub seconds: f64,
}

/// The pulse on `n` cells, `STEP_B` steps (§127.6.2's setup).
fn run(gpu: &Gpu, n: usize, coupling: Coupling) -> Result<Run> {
    let t_start = Instant::now();
    let gas = PerfectGas { viscosity: Viscosity::Constant { mu: 0.0 }, ..PerfectGas::air(P_OP) };
    let rs = gas.r_s();
    let c0 = (gas.gamma * rs * T0).sqrt();
    let rho0 = P_OP / (rs * T0);
    let dx = 1.0 / n as f64;
    let hm = tube(n)?;
    let gm = GpuMesh::upload(gpu, &hm)?;
    let amp = AMP * P_OP;
    let width = WIDTH_CELLS * dx;
    let dp = |i: usize| amp * (-((w(hm.c[i].x) - X0) / width).powi(2)).exp();
    let ge = (gas.gamma - 1.0) / gas.gamma;

    let mut rp = RawScalarField { name: "p".into(), ..RawScalarField::default() };
    rp.internal = (0..hm.n_cells).map(|i| s(dp(i))).collect();
    rp.boundary = with_sides(rp.boundary, entry("zeroGradient"));
    let mut rt = RawScalarField { name: "T".into(), ..RawScalarField::default() };
    rt.internal = (0..hm.n_cells).map(|i| s(T0 * (1.0 + dp(i) / P_OP).powf(ge))).collect();
    rt.boundary = with_sides(rt.boundary, entry("zeroGradient"));
    let mut ru = RawVectorField { name: "U".into(), ..RawVectorField::default() };
    ru.internal = (0..hm.n_cells).map(|i| Vec3::new(s(dp(i) / (rho0 * c0)), 0.0, 0.0)).collect();
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
    let dt = CFL * dx / c0;
    for (ddt, d) in [(&mut ctrl.momentum.ddt, &mut ctrl.momentum.delta_t), (&mut ctrl.energy.ddt, &mut ctrl.energy.delta_t)] {
        *ddt = DdtScheme::Euler;
        *d = s(dt);
    }
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

    let x: Vec<f64> = (0..hm.n_cells).map(|i| w(hm.c[i].x)).collect();
    // A bound (T, p_abs or rho <= 0) stops `run` with Error::Refused (§122.3).
    solver.run(gpu, &mut st, STEP_A)?;
    let pa: Vec<f64> = gpu.download(&st.p.f)?.iter().map(|v| w(*v)).collect();
    solver.run(gpu, &mut st, STEP_B - STEP_A)?;
    let pb: Vec<f64> = gpu.download(&st.p.f)?.iter().map(|v| w(*v)).collect();

    let x_a = peak_position(&x, &pa);
    let x_b = peak_position(&x, &pb);
    let ratio = match (x_a, x_b) {
        (Some(a), Some(b)) => (b - a) / ((STEP_B - STEP_A) as f64 * dt) / c0,
        _ => f64::NAN,
    };
    let left = x.iter().zip(&pb).filter(|(xi, _)| **xi < X0).map(|(_, v)| v.abs()).fold(0.0, f64::max) / amp;
    let height = pb.iter().copied().fold(f64::NEG_INFINITY, f64::max) / amp;
    Ok(Run { x_a, x_b, ratio, left, height, c0, dt, seconds: t_start.elapsed().as_secs_f64() })
}

fn describe(c: &mut Checks, label: &str, r: &Run) {
    let at = |v: Option<f64>| v.map_or("none".to_string(), |v| format!("{v:.6}"));
    c.note(&format!(
        "Gate CMP-AC {label}, {CELLS} cells: {STEP_B} steps of dt = {:.4e} s (acoustic CFL {CFL}), c_0 = {:.4} m/s; \
         peak at x {} m (step {STEP_A}) and {} m (step {STEP_B}); speed {:.5} c_0; peak height {:.4} A; \
         largest |p'| left of x0 {:.3e} A; {:.1} s",
        r.dt,
        r.c0,
        at(r.x_a),
        at(r.x_b),
        r.ratio,
        r.height,
        r.left,
        r.seconds
    ));
}

/// **Gate CMP-AC** (SPEC-LIT §127.6, §127.6.2).
pub(crate) fn check_acoustic(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    let coupling = row_coupling();
    let label = format!("(nOuterCorrectors {}, nCorrectors {})", coupling.n_outer, coupling.n_corr);
    let r = run(gpu, CELLS, coupling)?;
    describe(c, &label, &r);
    c.require("Gate CMP-AC: the pulse's peak is found at both sample steps", r.x_a.is_some() && r.x_b.is_some());
    let err = (r.ratio - 1.0).abs();
    c.check_p("Gate CMP-AC: pulse speed within 1 % of c_0 = sqrt(gamma R_s T_0), 400 cells (SPEC-LIT 127.6)", s(err), s(SPEED_TOL));
    if err <= SPEED_TOL {
        return Ok(());
    }
    let p = run(gpu, CELLS, PIMPLE)?;
    describe(c, "PIMPLE (nOuterCorrectors 5, alpha_p 1), printed only", &p);
    c.report(GateReport {
        verdict: Verdict::Misses,
        how: How::Live,
        gate: GATE,
        against: AGAINST,
        headline: format!(
            "the solver's default coupling {label}, {CELLS} cells: the pulse travels at {:.4} c_0 (bar: within 1 %)",
            r.ratio
        ),
        detail: vec![format!(
            "the same tube with PIMPLE (5 outer iterations, alpha_p 1; inputs only): {:.4} c_0 (SPEC-LIT §127.6.2)",
            p.ratio
        )],
        uncertainty: Some(Uncertainty::SingleMesh("one 400-cell tube, as the row states")),
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parabola finds the vertex of a sampled parabola exactly, and
    /// refuses a maximum at an end.
    #[test]
    fn the_peak_is_the_parabola_vertex() {
        let x: Vec<f64> = (0..8).map(|i| 0.5 * i as f64).collect();
        let v: Vec<f64> = x.iter().map(|xi| 2.0 - (xi - 1.3).powi(2)).collect();
        let xp = peak_position(&x, &v).expect("an interior maximum");
        assert!((xp - 1.3).abs() < 1e-12, "{xp}");
        let rising: Vec<f64> = x.clone();
        assert_eq!(peak_position(&x, &rising), None);
    }

    /// The sample steps keep the pulse clear of both walls: the peak
    /// travels `STEP_B * CFL = 200` cells (0.5 m) by step 1000, and eight
    /// e-folding half-widths either side of it (`exp(-64)`) stay inside the
    /// tube, at the start and at step 1000.
    #[test]
    fn the_pulse_stays_clear_of_the_walls() {
        let travel_b = STEP_B as f64 * CFL / CELLS as f64;
        let front = X0 + travel_b + 8.0 * WIDTH_CELLS / CELLS as f64;
        let back = X0 - 8.0 * WIDTH_CELLS / CELLS as f64;
        assert!(front < 1.0 && back > 0.0, "front {front}, back {back}");
        assert!((travel_b - 0.5).abs() < 1e-12, "{travel_b}");
    }
}
