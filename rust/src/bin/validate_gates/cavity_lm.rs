// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate CMP-LM (SPEC-LIT §127.6, §127.6.3): `ofgpu::compressible` at lid
//! Mach 0.05 on the lid-driven cavity at Re 100, against Ghia, Ghia & Shin
//! (1982), live.**
//!
//! The unit square of air at `T_0 = 300` K and `p_op = 1e5` Pa, the lid at
//! `U = 0.05 c_0`, the walls at `T_0`, constant `mu = rho_0 U L / 100`, run
//! with SIMPLE and the local time step of §125.4 until the field stops
//! moving. `u/U` along
//! `x = 0.5` and `v/U` along `y = 0.5` are held against Tables I and II
//! (`reference/ghia1982`, the key Gate 94-D reads) with Gate 94-D's own
//! tolerance and sampler, row for row: worst `|u/U - Ghia| <= 0.02` and
//! worst `|v/U - Ghia| <= 0.02` on the 80 x 80 mesh. Class P (the same bar
//! in both builds, docs/17 section 5.1.4). A miss fails its row and
//! registers the verdict (§69) with the measured values.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §94.4 (Gate 94-D, the tolerance and the sampler),
//!     §122 (the gas), §123-§125 (the equations and the outer loop), §124.3
//!     (the incompressible limit), §127.6 (the gate table) and §127.6.3
//!     (this gate as built)
//!   U. Ghia, K. N. Ghia & C. T. Shin, *J. Comput. Phys.* 48 (1982) 387-411,
//!     doi:10.1016/0021-9991(82)90058-4 - Tables I and II, read through the
//!     answer-key reader
//! No GPL-licensed source was consulted.

use std::collections::BTreeMap;
use std::time::Instant;

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::compressible::bc as cbc;
use ofgpu::compressible::thermo::{PerfectGas, Viscosity};
use ofgpu::compressible::{Compressible, CompressibleControls, CompressibleState};
use ofgpu::field::{GpuScalarField, GpuVectorField};
use ofgpu::field_setup::{setup_scalar_field, setup_vector_field};
use ofgpu::fv::{DivScheme, Limiter};
use ofgpu::io::case::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::mesh::HostMesh;
use ofgpu::timescheme::DdtScheme;
use ofgpu::{Error, Gpu, GpuMesh, Result, Scalar, Vec3};

use crate::published_benchmarks::compare;
use crate::{key, Checks, GateReport, How, Uncertainty, Verdict};

/// The gate's scope and registry name.
pub(crate) const GATE: &str = "SPEC-LIT 127.6 Gate CMP-LM lid-driven cavity at lid Mach 0.05";

const AGAINST: &str = "Ghia, Ghia & Shin (1982) Tables I and II, Re 100, 129 x 129 grid, centreline profiles";

/// The row's mesh: Gate 94-D's finest.
pub(crate) const CELLS: usize = 80;
/// The Reynolds number and the lid Mach number of the row.
pub(crate) const RE: f64 = 100.0;
pub(crate) const MACH: f64 = 0.05;
/// Gate 94-D's tolerance on a lid speed of 1 (validate.rs, check_ghia_cavity_on).
pub(crate) const GHIA_TOL: f64 = 0.02;
/// The rest state.
const P_OP: f64 = 1.0e5;
const T0: f64 = 300.0;
/// The `CFL` of the local time step (125.2), acoustic: about 1 on the lid
/// speed at lid Mach 0.05.
pub(crate) const LTS_CFL: f64 = 20.0;
const MARCH: &str = "SIMPLE with the local time step (125.2) at CFL 20";
/// The iteration budget of the row, the chunk the field stop is tested on,
/// and the stop: no cell's `u/U` or `v/U` moves by more than `FIELD_STOP`
/// across a chunk, a twentieth of the bar. Measured in both builds, the
/// change falls by a factor 0.4-0.7 per chunk and meets the stop at 2200
/// iterations (§127.6.3).
pub(crate) const MAX_ITERS: usize = 4000;
pub(crate) const CHUNK: usize = 200;
pub(crate) const FIELD_STOP: f64 = 1.0e-3;

/// The largest change of `u/U` or `v/U` in any cell between two fields; a
/// NaN anywhere makes it NaN.
pub(crate) fn max_change(a: &[Vec3], b: &[Vec3]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(p, q)| [(w(p.x) - w(q.x)).abs(), (w(p.y) - w(q.y)).abs()])
        .fold(0.0, |m: f64, v| if v.is_nan() || m.is_nan() { f64::NAN } else { m.max(v) })
}

fn w(x: Scalar) -> f64 {
    f64::from(x)
}

fn s(x: f64) -> Scalar {
    x as Scalar
}

/// The cavity: `n x n` cells on the unit square, one cell of `1/n` m
/// across in z with both z sides `symmetry`.
fn cavity_mesh(n: usize) -> Result<HostMesh> {
    let axis = |hi: Scalar, n: usize| GradedAxis { lo: 0.0, hi, n, expansion: 1.0, two_sided: false };
    let spec = BlockSpec {
        x: axis(1.0, n),
        y: axis(1.0, n),
        z: axis(s(1.0 / n as f64), 1),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall", "wall", "wall", "wall", "symmetry", "symmetry"].map(String::from),
        cyclic: Vec::new(),
    };
    blockgen::build_mesh(&spec)
}

fn entry(t: &str) -> PatchFieldSpec {
    PatchFieldSpec { type_name: t.into(), ..PatchFieldSpec::default() }
}

/// The four walls `walls` (the lid `lid` when given), z sides `symmetry`.
fn bcs(
    mut b: BTreeMap<String, PatchFieldSpec>,
    walls: PatchFieldSpec,
    lid: Option<PatchFieldSpec>,
) -> BTreeMap<String, PatchFieldSpec> {
    for side in ["xMin", "xMax", "yMin"] {
        b.insert(side.into(), walls.clone());
    }
    b.insert("yMax".into(), lid.unwrap_or(walls));
    for side in ["zMin", "zMax"] {
        b.insert(side.into(), entry("symmetry"));
    }
    b
}

/// One run: the mesh, `u/U` per cell, and what the loop reported.
pub(crate) struct Run {
    pub mesh: HostMesh,
    pub u: Vec<Vec3>,
    pub iterations: usize,
    /// The field stop was met.
    pub converged: bool,
    /// `(iterations, max |change of u/U|)` after each chunk.
    pub history: Vec<(usize, f64)>,
    pub last_line: String,
    pub mach_max: f64,
    pub lid: f64,
    pub seconds: f64,
}

/// The cavity on `n x n` cells, at most `max_iters` iterations
/// (§127.6.3's setup).
fn run(gpu: &Gpu, n: usize, max_iters: usize) -> Result<Run> {
    let t_start = Instant::now();
    let base = PerfectGas::air(P_OP);
    let rs = base.r_s();
    let c0 = (base.gamma * rs * T0).sqrt();
    let rho0 = P_OP / (rs * T0);
    let lid = MACH * c0;
    let mu = rho0 * lid * 1.0 / RE;
    let gas = PerfectGas { viscosity: Viscosity::Constant { mu }, ..base };
    let hm = cavity_mesh(n)?;
    let gm = GpuMesh::upload(gpu, &hm)?;

    let mut rp = RawScalarField { name: "p".into(), ..RawScalarField::default() };
    rp.internal = vec![0.0; hm.n_cells];
    rp.boundary = bcs(rp.boundary, entry("zeroGradient"), None);
    let mut rt = RawScalarField { name: "T".into(), ..RawScalarField::default() };
    rt.internal = vec![s(T0); hm.n_cells];
    let wall_t = PatchFieldSpec { type_name: "fixedValue".into(), value: vec![s(T0)], ..PatchFieldSpec::default() };
    rt.boundary = bcs(rt.boundary, wall_t, None);
    let mut ru = RawVectorField { name: "U".into(), ..RawVectorField::default() };
    ru.internal = vec![Vec3::ZERO; hm.n_cells];
    let fixed = |v: Vec3| PatchFieldSpec { type_name: "fixedValue".into(), value_v: vec![v], ..PatchFieldSpec::default() };
    ru.boundary = bcs(ru.boundary, fixed(Vec3::ZERO), Some(fixed(Vec3::new(s(lid), 0.0, 0.0))));

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
    // SIMPLE with the local time step (125.2), not plain steady SIMPLE: the
    // closed box's pressure level is set by the mass in it, which only the
    // ddt(rho) term carries (§127.6.3).
    ctrl.lts.cfl = s(LTS_CFL);
    ctrl.momentum.ddt = DdtScheme::LocalEuler;
    ctrl.energy.ddt = DdtScheme::LocalEuler;
    ctrl.momentum.div_scheme = DivScheme::Limited(Limiter::VanLeer);
    ctrl.energy.div_scheme = DivScheme::Limited(Limiter::VanLeer);
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
    // The stop is the field's own: chunks of CHUNK iterations until no cell's
    // `u/U` or `v/U` moves by more than FIELD_STOP across a chunk.
    // `residualControl` cannot stop this run, because the one-cell z
    // direction leaves `Uz`'s normalised residual at O(0.1) (§127.6.3).
    let to_u = |d: &[Vec3]| -> Vec<Vec3> {
        d.iter().map(|v| Vec3::new(s(w(v.x) / lid), s(w(v.y) / lid), s(w(v.z) / lid))).collect()
    };
    let mut u = to_u(&gpu.download(&st.u.f)?);
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
        let next = to_u(&gpu.download(&st.u.f)?);
        change = max_change(&u, &next);
        history.push((iterations, change));
        u = next;
    }
    Ok(Run {
        mesh: hm,
        u,
        iterations,
        converged: change <= FIELD_STOP,
        history,
        last_line,
        mach_max,
        lid,
        seconds: t_start.elapsed().as_secs_f64(),
    })
}

/// **Gate CMP-LM** (SPEC-LIT §127.6, §127.6.3).
pub(crate) fn check_cavity_lm(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    // answer-key: ghia1982-table-I
    // answer-key: ghia1982-table-II
    let key = key::ghia_1982()?;
    for kf in &key.keys {
        c.note(&kf.digest_line());
    }
    let r = match run(gpu, CELLS, MAX_ITERS) {
        Ok(r) => r,
        // A solver stop (a bound, §122.3) is the row's miss, measured as
        // such; any other error ends the section as before.
        Err(e @ Error::Refused(_)) => {
            let why = e.to_string();
            c.note(&format!("Gate CMP-LM: the {} run stopped: {why}", MARCH));
            for what in ["u/U", "v/U"] {
                c.check_p(&format!("Gate CMP-LM Re 100, lid M 0.05: worst |{what} - Ghia| at {CELLS} x {CELLS} (Gate 94-D's 0.02, SPEC-LIT 127.6)"), Scalar::NAN, s(GHIA_TOL));
            }
            c.report(GateReport {
                verdict: Verdict::Misses,
                how: How::Live,
                gate: GATE,
                against: AGAINST,
                headline: format!("the {} run stopped before a steady state: no profile to compare", MARCH),
                detail: vec![why],
                uncertainty: Some(Uncertainty::SingleMesh("one 80 x 80 mesh, Gate 94-D's finest, as the row states")),
            });
            return Ok(());
        }
        Err(e) => return Err(e),
    };
    c.note(&format!(
        "Gate CMP-LM: {CELLS} x {CELLS}, lid {:.4} m/s (M {MACH}), Re {RE}, {}: {} iterations, field stop (max |change of u/U| over {CHUNK} iterations <= {FIELD_STOP:.0e}) {}, \
         M_max {:.4}, {:.1} s; last: {}",
        r.lid,
        MARCH,
        r.iterations,
        if r.converged { "met" } else { "NOT met" },
        r.mach_max,
        r.seconds,
        r.last_line
    ));
    let hist: Vec<String> = r.history.iter().map(|(i, d)| format!("{i}: {d:.2e}")).collect();
    c.note(&format!("Gate CMP-LM: max |change of u/U| per chunk: {}", hist.join(", ")));
    // `compare` folds with `f64::max`, which drops a NaN: the field's own
    // finiteness is its row.
    let finite = r.u.iter().all(|v| w(v.x).is_finite() && w(v.y).is_finite());
    c.require(&format!("Gate CMP-LM: the {CELLS} x {CELLS} velocity field is finite"), finite);
    let label = format!("Gate CMP-LM, compressible at lid M {MACH}, Re {RE}, {CELLS} x {CELLS}");
    let (du, dv) = compare(&r.mesh, &r.u, CELLS, &key, RE as u32, &label);
    c.note(&format!("Gate CMP-LM: worst |u/U - Ghia| {du:.4}, worst |v/U - Ghia| {dv:.4} (bar {GHIA_TOL} each)"));
    c.check_p(
        &format!("Gate CMP-LM Re 100, lid M 0.05: worst |u/U - Ghia| over Table I at {CELLS} x {CELLS} (Gate 94-D's 0.02, SPEC-LIT 127.6)"),
        s(du),
        s(GHIA_TOL),
    );
    c.check_p(
        &format!("Gate CMP-LM Re 100, lid M 0.05: worst |v/U - Ghia| over Table II at {CELLS} x {CELLS} (Gate 94-D's 0.02, SPEC-LIT 127.6)"),
        s(dv),
        s(GHIA_TOL),
    );
    let pass = finite && du <= GHIA_TOL && dv <= GHIA_TOL;
    if pass {
        return Ok(());
    }
    c.report(GateReport {
        verdict: Verdict::Misses,
        how: How::Live,
        gate: GATE,
        against: AGAINST,
        headline: format!("worst |du| {du:.4}, worst |dv| {dv:.4} at {CELLS} x {CELLS} against 0.02"),
        detail: vec![format!(
            "{}, {} iterations, field stop {}; {}",
            MARCH,
            r.iterations,
            if r.converged { "met" } else { "not met" },
            r.last_line
        )],
        uncertainty: Some(Uncertainty::SingleMesh("one 80 x 80 mesh, Gate 94-D's finest, as the row states")),
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lid speed and the viscosity give the row's Mach and Reynolds
    /// numbers: `U = 0.05 c_0` with `c_0 = sqrt(1.4 R_s 300)`.
    #[test]
    fn the_lid_gives_mach_and_reynolds() {
        let gas = PerfectGas::air(P_OP);
        let c0 = (gas.gamma * gas.r_s() * T0).sqrt();
        assert!((c0 - 347.2).abs() < 0.1, "{c0}");
        let rho0 = P_OP / (gas.r_s() * T0);
        let lid = MACH * c0;
        let mu = rho0 * lid / RE;
        assert!((rho0 * lid * 1.0 / mu - 100.0).abs() < 1e-9);
        assert!((lid / c0 - 0.05).abs() < 1e-15);
    }

    /// The field stop reads the larger of the two in-plane changes, ignores
    /// `w`, and turns a NaN anywhere into NaN, so a broken run never meets it.
    #[test]
    fn the_field_stop_reads_u_and_v_and_keeps_nan() {
        let a = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.5, 0.25, 0.0)];
        let b = [Vec3::new(0.125, 0.0, 9.0), Vec3::new(0.5, -0.25, 0.0)];
        assert_eq!(max_change(&a, &b), 0.5);
        let c = [Vec3::new(Scalar::NAN, 0.0, 0.0), Vec3::new(0.5, 0.25, 0.0)];
        assert!(max_change(&a, &c).is_nan());
        assert!(!(max_change(&a, &c) <= FIELD_STOP));
    }
}
