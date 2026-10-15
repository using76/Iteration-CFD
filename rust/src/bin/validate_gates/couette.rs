// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate CMP-COU (SPEC-LIT §127.6, §127.6.4): compressible Couette with
//! viscous heating, against (127.12), live.**
//!
//! A channel `NY = 40` cells tall on `h = 1e-3` m and `NX = 4` cells long,
//! cyclic in x, one cell of `h/NY` across in z: the wall at `y = 0`
//! stationary and adiabatic, the wall at `y = h` moving at `U = 200` m/s
//! and held at `T_top = 300` K, air at `p_op = 1e5` Pa with constant
//! `mu = rho_0 U h/10`. Started at rest and marched with SIMPLE and the
//! local time step of §125.4 at CFL 20 until no cell's
//! `(T - T_top)/(T_w - T_top)` or `u/U` moves by more than `2.5e-5` across
//! a chunk of 200 iterations. Two rows, class P (the same bar in both
//! builds, docs/17 section 5.1.4): the adiabatic-wall rise `T_w - T_top`
//! within 0.5 % of `Pr U^2/(2 c_p)`, and the `T` profile L-infinity within
//! 0.5 % of the rise, both against (127.12) evaluated live by
//! `compressible::exact` (CMP-07). A miss fails its row and registers the
//! verdict (§69) with the measured values.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §127.4 ((127.12) and the oracle rise), §122 (the
//!     gas), §123-§125 (the equations and the outer loop), §127.6 (the gate
//!     table) and §127.6.4 (this gate as built)
//!   F. M. White, *Viscous Fluid Flow* 3-3, cited by docs/17 for the same
//!     result, *not read*
//! No GPL-licensed source was consulted.

use std::collections::BTreeMap;
use std::time::Instant;

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::compressible::bc as cbc;
use ofgpu::compressible::exact::{Couette, couette_adiabatic_wall_rise};
use ofgpu::compressible::thermo::{PerfectGas, Viscosity};
use ofgpu::compressible::{Compressible, CompressibleControls, CompressibleState};
use ofgpu::field::{GpuScalarField, GpuVectorField};
use ofgpu::field_setup::{setup_scalar_field, setup_vector_field};
use ofgpu::io::case::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::mesh::HostMesh;
use ofgpu::timescheme::DdtScheme;
use ofgpu::{Error, Gpu, GpuMesh, Result, Scalar, Vec3};

use crate::{Checks, GateReport, How, Uncertainty, Verdict};

/// The gate's scope and registry name.
pub(crate) const GATE: &str =
    "SPEC-LIT 127.6 Gate CMP-COU compressible Couette with viscous heating";

const AGAINST: &str =
    "(127.12): T_w - T_top = Pr U^2/(2 c_p) and the quadratic T(y), evaluated by compressible::exact (CMP-07)";

/// What the verdict says about the mesh, §94.3.
const ONE_CHANNEL: &str = "one 40-cell channel, as the row states";

/// The row's inputs: the channel height, the top-wall speed and
/// temperature, and the Reynolds number the constant `mu` is set from.
pub(crate) const H: f64 = 1.0e-3;
pub(crate) const U_TOP: f64 = 200.0;
pub(crate) const T_TOP: f64 = 300.0;
pub(crate) const RE: f64 = 10.0;
/// The rest state's gauge datum.
const P_OP: f64 = 1.0e5;
/// The row's mesh: 40 cells across the height, 4 cells of the same size
/// along the cyclic x, one cell of `h/NY` across in z.
pub(crate) const NY: usize = 40;
pub(crate) const NX: usize = 4;
/// The `CFL` of the local time step (125.2), as CMP-LM's.
pub(crate) const LTS_CFL: f64 = 20.0;
const MARCH: &str = "SIMPLE with the local time step (125.2) at CFL 20";
/// The iteration budget of the row, the chunk the field stop is tested on,
/// and the stop: no cell's `(T - T_top)/rise` or `u/U` moves by more than
/// `FIELD_STOP` across a chunk, a two-hundredth of the bar: the change
/// contracts by about 0.84 per chunk, so the march's remaining distance,
/// about `FIELD_STOP r/(1 - r)`, stays below the discrete fixed point's own
/// `1/(4 NY^2)` offset (§127.6.4).
pub(crate) const MAX_ITERS: usize = 16000;
pub(crate) const CHUNK: usize = 200;
pub(crate) const FIELD_STOP: f64 = 2.5e-5;
/// The row's bar, the plan's, fixed before measurement.
pub(crate) const BAR: f64 = 0.005;

fn w(x: Scalar) -> f64 {
    f64::from(x)
}

fn s(x: f64) -> Scalar {
    x as Scalar
}

/// The largest change between two fields, cell by cell; a NaN anywhere
/// makes it NaN, so a broken run never meets the field stop.
pub(crate) fn max_change(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, |m: f64, v| if v.is_nan() || m.is_nan() { f64::NAN } else { m.max(v) })
}

/// The row's profile measure: the largest `|T_i - T(y_i)|` of (127.12)
/// over the cells, over the exact rise `T_w - T_top`. A NaN anywhere makes
/// it NaN.
pub(crate) fn profile_error(y: &[f64], t: &[f64], key: &Couette) -> f64 {
    let rise = key.wall_temperature() - key.t_top;
    y.iter()
        .zip(t)
        .map(|(y_i, t_i)| (t_i - key.temperature(*y_i)).abs() / rise)
        .fold(0.0, |m: f64, v| if v.is_nan() || m.is_nan() { f64::NAN } else { m.max(v) })
}

/// The row's first bar, as the section names it.
fn row_1_label() -> String {
    format!(
        "Gate CMP-COU: adiabatic-wall T_w - T_top within 0.5 % of Pr U^2/(2 c_p) (127.12), \
         U {U_TOP} m/s, {NY} cells (SPEC-LIT 127.6)"
    )
}

/// The row's second bar, as the section names it.
fn row_2_label() -> String {
    format!(
        "Gate CMP-COU: T profile L-infinity within 0.5 % of the rise (127.12), \
         {NY} cells (SPEC-LIT 127.6)"
    )
}

/// The channel: `NY` cells of `h/NY` on `[0, h]`, `NX` cells of the same
/// size along x as a cyclic pair, one cell of `h/NY` across in z with both
/// z sides `symmetry`, the two y sides `wall`.
fn couette_mesh() -> Result<HostMesh> {
    let cell = H / NY as f64;
    let axis = |lo: f64, hi: f64, n: usize| GradedAxis {
        lo: s(lo),
        hi: s(hi),
        n,
        expansion: 1.0,
        two_sided: false,
    };
    let mut spec = BlockSpec {
        x: axis(0.0, NX as f64 * cell, NX),
        y: axis(0.0, H, NY),
        z: axis(0.0, cell, 1),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall", "wall", "wall", "wall", "symmetry", "symmetry"].map(String::from),
        cyclic: Vec::new(),
    };
    spec.set_cyclic_axis(0)?;
    blockgen::build_mesh(&spec)
}

fn entry(t: &str) -> PatchFieldSpec {
    PatchFieldSpec { type_name: t.into(), ..PatchFieldSpec::default() }
}

/// The six patches: `ends` on the cyclic x pair, `wall_lo` on the
/// stationary adiabatic wall, `wall_hi` on the moving isothermal wall, and
/// `symmetry` on both z sides.
fn bcs(
    mut b: BTreeMap<String, PatchFieldSpec>,
    ends: PatchFieldSpec,
    wall_lo: PatchFieldSpec,
    wall_hi: PatchFieldSpec,
) -> BTreeMap<String, PatchFieldSpec> {
    b.insert("xMin".into(), ends.clone());
    b.insert("xMax".into(), ends);
    b.insert("yMin".into(), wall_lo);
    b.insert("yMax".into(), wall_hi);
    for side in ["zMin", "zMax"] {
        b.insert(side.into(), entry("symmetry"));
    }
    b
}

/// One run: the mesh, the key it is held against, the cell `T` and
/// velocity, the adiabatic wall's face `T`, and what the loop reported.
pub(crate) struct Run {
    pub mesh: HostMesh,
    pub key: Couette,
    pub t: Vec<f64>,
    pub u: Vec<Vec3>,
    /// The area-mean of `T`'s boundary face values on yMin, minus `T_top`.
    pub rise_num: f64,
    pub iterations: usize,
    /// The field stop was met.
    pub converged: bool,
    /// `(iterations, max |change|)` after each chunk.
    pub history: Vec<(usize, f64)>,
    pub last_line: String,
    pub mach_max: f64,
    pub seconds: f64,
}

/// The Couette channel of §127.6.4, at most `max_iters` iterations.
fn run(gpu: &Gpu, max_iters: usize) -> Result<Run> {
    let t_start = Instant::now();
    let base = PerfectGas::air(P_OP);
    let (pr, cp) = (base.pr, base.cp());
    let rho0 = P_OP / (base.r_s() * T_TOP);
    let mu = rho0 * U_TOP * H / RE;
    let key = Couette::new(H, U_TOP, T_TOP, mu, mu * cp / pr)?;
    let rise = couette_adiabatic_wall_rise(pr, U_TOP, cp)?;
    let gas = PerfectGas { viscosity: Viscosity::Constant { mu }, ..base };
    let hm = couette_mesh()?;
    let gm = GpuMesh::upload(gpu, &hm)?;

    // At rest: `u = 0`, `p = 0` (gauge), `T = T_top` everywhere - not from
    // the key, so the profile the run settles to is its own.
    let mut rp = RawScalarField { name: "p".into(), ..RawScalarField::default() };
    rp.internal = vec![0.0; hm.n_cells];
    rp.boundary = bcs(rp.boundary, entry("cyclic"), entry("zeroGradient"), entry("zeroGradient"));
    let mut rt = RawScalarField { name: "T".into(), ..RawScalarField::default() };
    rt.internal = vec![s(T_TOP); hm.n_cells];
    let cold = PatchFieldSpec {
        type_name: "fixedValue".into(),
        value: vec![s(T_TOP)],
        ..PatchFieldSpec::default()
    };
    rt.boundary = bcs(rt.boundary, entry("cyclic"), entry("zeroGradient"), cold);
    let mut ru = RawVectorField { name: "U".into(), ..RawVectorField::default() };
    ru.internal = vec![Vec3::ZERO; hm.n_cells];
    let fixed = |v: Vec3| {
        PatchFieldSpec { type_name: "fixedValue".into(), value_v: vec![v], ..PatchFieldSpec::default() }
    };
    ru.boundary =
        bcs(ru.boundary, entry("cyclic"), fixed(Vec3::ZERO), fixed(Vec3::new(s(U_TOP), 0.0, 0.0)));

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

    // The stop is the field's own: chunks of CHUNK iterations until no
    // cell's `(T - T_top)/rise` or `u/U` moves by more than FIELD_STOP
    // across a chunk. `residualControl` is off, as CMP-LM's is.
    let stop_of = |u: &[Vec3], t: &[Scalar]| -> Vec<f64> {
        u.iter()
            .zip(t)
            .flat_map(|(v, ti)| [w(v.x) / U_TOP, w(v.y) / U_TOP, (w(*ti) - T_TOP) / rise])
            .collect()
    };
    let mut prev = stop_of(&gpu.download(&st.u.f)?, &gpu.download(&st.t.f)?);
    let mut iterations = 0usize;
    let mut change = f64::INFINITY;
    let mut history = Vec::new();
    let mut last_line = "no iteration".to_string();
    let mut mach_max = f64::NAN;
    while iterations < max_iters && !(change <= FIELD_STOP) {
        let k = CHUNK.min(max_iters - iterations);
        // A bound (T, p_abs or rho <= 0) stops `run` with Error::Refused (§122.3).
        let rep = solver.run(gpu, &mut st, k)?;
        iterations += rep.steps;
        if let Some(o) = rep.last.as_ref().and_then(|l| l.outers.last()) {
            last_line = o.line();
            mach_max = o.mach_max;
        }
        let next = stop_of(&gpu.download(&st.u.f)?, &gpu.download(&st.t.f)?);
        change = max_change(&prev, &next);
        history.push((iterations, change));
        prev = next;
    }
    let t: Vec<f64> = gpu.download(&st.t.f)?.iter().map(|v| w(*v)).collect();
    let u = gpu.download(&st.u.f)?;
    // The adiabatic wall's face values are `T`'s boundary field on the
    // yMin patch, `bf[p.start .. p.start + p.size]`.
    let y_min = hm.patches.iter().find(|p| p.name == "yMin").expect("the yMin patch");
    let t_bf = gpu.download(&st.t.bf)?;
    let (mut area, mut t_face) = (0.0f64, 0.0f64);
    for k in 0..y_min.size {
        let a = w(hm.b_mag_sf[y_min.start + k]);
        area += a;
        t_face += a * w(t_bf[y_min.start + k]);
    }
    let rise_num = t_face / area - T_TOP;
    Ok(Run {
        mesh: hm,
        key,
        t,
        u,
        rise_num,
        iterations,
        converged: change <= FIELD_STOP,
        history,
        last_line,
        mach_max,
        seconds: t_start.elapsed().as_secs_f64(),
    })
}

/// **Gate CMP-COU** (SPEC-LIT §127.6, §127.6.4).
pub(crate) fn check_couette(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    let gas = PerfectGas::air(P_OP);
    let rise_exact = couette_adiabatic_wall_rise(gas.pr, U_TOP, gas.cp())?;
    let r = match run(gpu, MAX_ITERS) {
        Ok(r) => r,
        // A solver stop (a bound, §122.3) is the row's miss, measured as
        // such; any other error ends the section as before.
        Err(e @ Error::Refused(_)) => {
            let why = e.to_string();
            c.note(&format!("Gate CMP-COU: the {MARCH} run stopped: {why}"));
            c.check_p(&row_1_label(), Scalar::NAN, s(BAR));
            c.check_p(&row_2_label(), Scalar::NAN, s(BAR));
            c.report(GateReport {
                verdict: Verdict::Misses,
                how: How::Live,
                gate: GATE,
                against: AGAINST,
                headline: format!(
                    "the {MARCH} run stopped before a steady state: no field to compare"
                ),
                detail: vec![why],
                uncertainty: Some(Uncertainty::SingleMesh(ONE_CHANNEL)),
            });
            return Ok(());
        }
        Err(e) => return Err(e),
    };
    let mu = r.key.mu;
    let c0 = (gas.gamma * gas.r_s() * T_TOP).sqrt();
    c.note(&format!(
        "Gate CMP-COU: {NX} x {NY} x 1, h {H} m, U {U_TOP} m/s (U/c_0 {:.4}), Re {RE}, \
         mu {mu:.4e} Pa s, T_top {T_TOP} K: {MARCH}, {} iterations, field stop \
         (max |change| over {CHUNK} iterations <= {FIELD_STOP:.1e}) {}, M_max {:.4}, {:.1} s; \
         last: {}",
        U_TOP / c0,
        r.iterations,
        if r.converged { "met" } else { "NOT met" },
        r.mach_max,
        r.seconds,
        r.last_line
    ));
    let hist: Vec<String> = r.history.iter().map(|(i, d)| format!("{i}: {d:.2e}")).collect();
    c.note(&format!("Gate CMP-COU: max |change| per chunk: {}", hist.join(", ")));
    // `f64::max` folds drop a NaN, so the field's own finiteness is its
    // row: `require` the `T` field before the profile reads it.
    let finite = r.t.iter().all(|v| v.is_finite());
    c.require(&format!("Gate CMP-COU: the {NY}-cell temperature field is finite"), finite);
    let rise_err = (r.rise_num / rise_exact - 1.0).abs();
    c.note(&format!(
        "Gate CMP-COU: T_w - T_top {:+.6} K measured, {rise_exact:.6} K exact (127.12): \
         relative error {rise_err:.3e} (bar {BAR})",
        r.rise_num
    ));
    let y: Vec<f64> =
        r.mesh.c.iter().map(|p| w(p.y)).collect();
    let err = profile_error(&y, &r.t, &r.key);
    let du = r
        .mesh
        .c
        .iter()
        .zip(&r.u)
        .map(|(p, v)| (w(v.x) - r.key.velocity(w(p.y))).abs() / U_TOP)
        .fold(0.0, f64::max);
    c.note(&format!(
        "Gate CMP-COU: T profile L-infinity {err:.3e} of the rise (bar {BAR}); \
         u L-infinity {du:.3e} of U against (127.12)'s u = U y/h"
    ));
    c.check_p(&row_1_label(), s(rise_err), s(BAR));
    c.check_p(&row_2_label(), s(err), s(BAR));
    if finite && rise_err <= BAR && err <= BAR {
        return Ok(());
    }
    c.report(GateReport {
        verdict: Verdict::Misses,
        how: How::Live,
        gate: GATE,
        against: AGAINST,
        headline: format!(
            "T_w - T_top {:+.6} K against {rise_exact:.6} K ({rise_err:.3e}), \
             T profile L-infinity {err:.3e} of the rise, both against 0.005, {NY} cells",
            r.rise_num
        ),
        detail: vec![format!(
            "{}, {} iterations, field stop {}; {}",
            MARCH,
            r.iterations,
            if r.converged { "met" } else { "not met" },
            r.last_line
        )],
        uncertainty: Some(Uncertainty::SingleMesh(ONE_CHANNEL)),
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gas() -> PerfectGas {
        PerfectGas::air(P_OP)
    }

    fn mu_of(g: &PerfectGas) -> f64 {
        let rho0 = P_OP / (g.r_s() * T_TOP);
        rho0 * U_TOP * H / RE
    }

    /// The rise is the oracle value of §127.4, from the gas's own `Pr` and
    /// `c_p` and from the key's `Couette` alike.
    #[test]
    fn the_rise_is_the_oracle_value() {
        let g = gas();
        let rise = couette_adiabatic_wall_rise(g.pr, U_TOP, g.cp()).expect("the rise");
        assert!((rise / 14.133676596233594 - 1.0).abs() <= 1e-12, "{rise}");
        let mu = mu_of(&g);
        let key = Couette::new(H, U_TOP, T_TOP, mu, mu * g.cp() / g.pr).expect("the key");
        assert!((key.wall_temperature() - T_TOP - rise).abs() / rise <= 1e-12);
    }

    /// The constant viscosity is the row's Reynolds number, and the top
    /// speed is the Mach `U/c_0 = 0.576` the run is at.
    #[test]
    fn the_viscosity_gives_the_reynolds_number() {
        let g = gas();
        let c0 = (g.gamma * g.r_s() * T_TOP).sqrt();
        let rho0 = P_OP / (g.r_s() * T_TOP);
        let mu = mu_of(&g);
        assert!((rho0 * U_TOP * H / mu - RE).abs() <= 1e-12);
        assert!((U_TOP / c0 - 0.576).abs() <= 1e-3, "U/c_0 = {}", U_TOP / c0);
    }

    /// The profile measure is normalised by the exact rise: the key's own
    /// profile is 0, one cell raised by a hundredth of the rise is 0.01,
    /// and a NaN anywhere is NaN.
    #[test]
    fn the_profile_error_is_normalised_by_the_rise() {
        let g = gas();
        let key = Couette::new(H, U_TOP, T_TOP, mu_of(&g), mu_of(&g) * g.cp() / g.pr)
            .expect("the key");
        let y: Vec<f64> = (0..NY).map(|i| (i as f64 + 0.5) * H / NY as f64).collect();
        let t: Vec<f64> = y.iter().map(|y_i| key.temperature(*y_i)).collect();
        assert!(profile_error(&y, &t, &key) <= 1e-14);
        let mut hot = t.clone();
        hot[3] += 0.01 * (key.wall_temperature() - T_TOP);
        assert!((profile_error(&y, &hot, &key) - 0.01).abs() <= 1e-12);
        hot[7] = f64::NAN;
        assert!(profile_error(&y, &hot, &key).is_nan());
    }

    /// The field stop reads the largest change across the two fields and
    /// turns a NaN anywhere into NaN, so a broken run never meets it.
    #[test]
    fn the_field_stop_keeps_nan() {
        assert_eq!(max_change(&[0.0, 1.0], &[0.25, 1.0]), 0.25);
        let nan = [0.0, f64::NAN];
        assert!(max_change(&[0.0, 1.0], &nan).is_nan());
        assert!(!(max_change(&[0.0, 1.0], &nan) <= FIELD_STOP));
    }
}
