// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.
//
// CMP-13's both-builds gate, written ONCE and included twice - by
// `tests/compressible_lts.rs`, which runs it as an integration test in BOTH
// builds, and by `src/compressible/lts.rs`, whose lib tests these are. The
// tests are SPEC-LIT §125.7's gate table for the local time step of a steady
// run: (125.2) against its host twin, a uniform step against the Euler step,
// the fixed point against the Courant number, and the refusals.
//
// An integration test's body cannot carry `//!` or `#![..]`, and neither can
// this file - it is included, not compiled standalone. Every comparison is
// made in f64 from the downloaded `Scalar` values; the R-class tolerances go
// through `tol_r` (docs/17 §5.1, D-F32-1), written once and never as a typed
// f32 number.
//
// Provenance: ORIGINAL - the meshes, the fields and the answer keys below are
// this tree's own construction; the step they exercise is §125.4's (125.2).
// No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::compressible::lts::{host_delta_t, LtsControls};
use ofgpu::compressible::thermo::PerfectGas;
use ofgpu::compressible::{Compressible, CompressibleControls, CompressibleState};
use ofgpu::solver::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::timescheme::DdtScheme;
use ofgpu::{
    BcKind, Error, Gpu, GpuMesh, GpuScalarField, GpuVectorField, HostMesh, Label, Result, Scalar, Vec3,
};

/// The process's device, if the box has one: every GPU test returns early
/// rather than failing.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The build's `Scalar` widened to the f64 the answer keys are computed in.
#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    x as f64
}

/// The R-class tolerance (D-F32-1): `tol64` in the default build; under
/// `single`, `max(tol64, min(tol64 * 2^29, 1e-5))`.
fn tol_r(tol64: f64) -> f64 {
    if cfg!(feature = "single") {
        tol64.max((tol64 * 2f64.powi(29)).min(1.0e-5))
    } else {
        tol64
    }
}

/// A box on `[0, hi]`, every patch a plain `patch`, graded per axis.
fn box_mesh(n: [usize; 3], hi: [Scalar; 3], expansion: [Scalar; 3]) -> Result<HostMesh> {
    let axis = |hi: Scalar, n: usize, ex: Scalar| GradedAxis {
        lo: 0.0,
        hi,
        n,
        expansion: ex,
        two_sided: false,
    };
    blockgen::build_mesh(&BlockSpec {
        x: axis(hi[0], n[0], expansion[0]),
        y: axis(hi[1], n[1], expansion[1]),
        z: axis(hi[2], n[2], expansion[2]),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["patch"; 6].map(String::from),
        cyclic: Vec::new(),
    })
}

/// The 5x4x3 box on [0,1]x[0,0.8]x[0,0.6], graded 2:1 in y.
fn graded_box() -> Result<HostMesh> {
    box_mesh([5, 4, 3], [1.0, 0.8, 0.6], [1.0, 2.0, 1.0])
}

/// The outward direction of a boundary face as `(axis, sign)`.
fn face_axis(sf: Vec3) -> (usize, f64) {
    let (ax, ay, az) = (w64(sf.x).abs(), w64(sf.y).abs(), w64(sf.z).abs());
    let s = |v: Scalar| if v < 0.0 { -1.0 } else { 1.0 };
    if ax >= ay && ax >= az {
        (0, s(sf.x))
    } else if ay >= az {
        (1, s(sf.y))
    } else {
        (2, s(sf.z))
    }
}

/// A scalar field: cells from `cell(i, centre)`, a boundary face fixed at
/// `Some(v)` or zeroGradient at `None`.
fn scalar(
    g: &Gpu,
    gm: &GpuMesh,
    hm: &HostMesh,
    name: &str,
    cell: impl Fn(usize, Vec3) -> f64,
    face: impl Fn(Vec3) -> Option<f64>,
) -> Result<GpuScalarField> {
    let nbf = hm.n_boundary_faces;
    let mut f = GpuScalarField::zeros(g, gm, name)?;
    let v: Vec<Scalar> = (0..hm.n_cells).map(|i| cell(i, hm.c[i]) as Scalar).collect();
    g.write(&mut f.f, &v)?;
    let (mut kind, mut fr, mut rv) = (vec![0 as Label; nbf], vec![0.0 as Scalar; nbf], vec![0.0 as Scalar; nbf]);
    for b in 0..nbf {
        match face(hm.b_sf[b]) {
            Some(x) => {
                kind[b] = BcKind::FixedValue as Label;
                fr[b] = 1.0;
                rv[b] = x as Scalar;
            }
            None => kind[b] = BcKind::ZeroGradient as Label,
        }
    }
    g.write(&mut f.bc_kind, &kind)?;
    g.write(&mut f.fr, &fr)?;
    g.write(&mut f.ref_value, &rv)?;
    g.write(&mut f.ref_grad, &vec![0.0 as Scalar; nbf])?;
    Ok(f)
}

/// A vector field: cells from `cell(centre)`, a boundary face fixed at
/// `Some(v)` or zeroGradient at `None`.
fn vector(
    g: &Gpu,
    gm: &GpuMesh,
    hm: &HostMesh,
    cell: impl Fn(Vec3) -> Vec3,
    face: impl Fn(Vec3, Vec3) -> Option<Vec3>,
) -> Result<GpuVectorField> {
    let nbf = hm.n_boundary_faces;
    let mut u = GpuVectorField::zeros(g, gm, "U")?;
    let v: Vec<Vec3> = hm.c.iter().map(|c| cell(*c)).collect();
    g.write(&mut u.f, &v)?;
    let (mut kind, mut fr, mut rv) = (vec![0 as Label; nbf], vec![0.0 as Scalar; nbf], vec![Vec3::ZERO; nbf]);
    for b in 0..nbf {
        match face(hm.b_sf[b], hm.b_cf[b]) {
            Some(x) => {
                kind[b] = BcKind::FixedValue as Label;
                fr[b] = 1.0;
                rv[b] = x;
            }
            None => kind[b] = BcKind::ZeroGradient as Label,
        }
    }
    g.write(&mut u.bc_kind, &kind)?;
    g.write(&mut u.fr, &fr)?;
    g.write(&mut u.ref_value, &rv)?;
    g.write(&mut u.ref_grad, &vec![Vec3::ZERO; nbf])?;
    Ok(u)
}

/// The channel: inlet -x with a sheared fixed `u` and `T = 300`, no-slip
/// walls on y and z, outlet +x with `p = 0` fixed and `u`, `T`
/// zeroGradient. The cells start at the inlet speed and at `300 + 5 x`.
fn channel(g: &Gpu, gm: &GpuMesh, hm: &HostMesh) -> Result<CompressibleState> {
    let u_in = |c: Vec3| Vec3::new((20.0 + 10.0 * w64(c.y)) as Scalar, 0.0, 0.0);
    let u = vector(g, gm, hm, u_in, |sf, cf| match face_axis(sf) {
        (0, s) if s < 0.0 => Some(u_in(cf)),
        (0, _) => None,
        _ => Some(Vec3::ZERO),
    })?;
    let p = scalar(g, gm, hm, "p", |_, _| 0.0, |sf| match face_axis(sf) {
        (0, s) if s > 0.0 => Some(0.0),
        _ => None,
    })?;
    let t = scalar(g, gm, hm, "T", |_, c| 300.0 + 5.0 * w64(c.x), |sf| match face_axis(sf) {
        (0, s) if s < 0.0 => Some(300.0),
        _ => None,
    })?;
    CompressibleState::new(g, gm, u, p, t)
}

/// The closed box: no-slip walls, `p` and `T` zeroGradient, `T = 300`,
/// gauge `p` from `p_at(cell)`.
fn closed_box(g: &Gpu, gm: &GpuMesh, hm: &HostMesh, p_at: impl Fn(usize) -> f64) -> Result<CompressibleState> {
    let u = vector(g, gm, hm, |_| Vec3::ZERO, |_, _| Some(Vec3::ZERO))?;
    let p = scalar(g, gm, hm, "p", |i, _| p_at(i), |_| None)?;
    let t = scalar(g, gm, hm, "T", |_, _| 300.0, |_| None)?;
    CompressibleState::new(g, gm, u, p, t)
}

/// The linear solvers every test uses: PBiCGStab with DILU, tight.
fn solvers(ctrl: &mut CompressibleControls) {
    let s = SolverControls {
        solver: LinearSolverKind::PBiCGStab,
        precon: Preconditioner::Dilu,
        tolerance: 1.0e-12,
        rel_tol: 0.0,
        max_iter: 500,
        ..Default::default()
    };
    ctrl.momentum.u_solver = s;
    ctrl.pressure.p_solver = s;
    ctrl.energy.t_solver = s;
}

/// Controls for `ddt` on both equations at `dt`, `n_outer` outer
/// iterations and `n_corr` correctors, with the relaxation given.
fn controls(ddt: DdtScheme, dt: Scalar, n_outer: usize, relax: [Scalar; 3]) -> CompressibleControls {
    let mut c = CompressibleControls::default();
    c.momentum.ddt = ddt;
    c.momentum.delta_t = dt;
    c.energy.ddt = ddt;
    c.energy.delta_t = dt;
    c.momentum.u_relax = relax[0];
    c.energy.t_relax = relax[1];
    c.p_relax = relax[2];
    c.n_outer = n_outer;
    c.n_corr = 2;
    solvers(&mut c);
    c
}

/// The cell values of `u`, `p`, `T` and `rho`, widened to f64.
struct Fields {
    u: Vec<[f64; 3]>,
    p: Vec<f64>,
    t: Vec<f64>,
    rho: Vec<f64>,
}

fn fields(g: &Gpu, s: &CompressibleState) -> Result<Fields> {
    Ok(Fields {
        u: g.download(&s.u.f)?.iter().map(|u| [w64(u.x), w64(u.y), w64(u.z)]).collect(),
        p: g.download(&s.p.f)?.iter().map(|x| w64(*x)).collect(),
        t: g.download(&s.t.f)?.iter().map(|x| w64(*x)).collect(),
        rho: g.download(&s.rho.f)?.iter().map(|x| w64(*x)).collect(),
    })
}

/// `max_i |a_i - b_i|` over scalars.
fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// `max_i |a_i - b_i|` over vectors, in the Euclidean norm.
fn max_diff_v(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2)).sqrt())
        .fold(0.0, f64::max)
}

fn max_abs(a: &[f64]) -> f64 {
    a.iter().map(|x| x.abs()).fold(0.0, f64::max)
}

fn max_abs_v(a: &[[f64; 3]]) -> f64 {
    a.iter().map(|x| (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt()).fold(0.0, f64::max)
}

// --------------------------------------------------------------------------
//  Gate 1 - SPEC-LIT 125.7, class R
// --------------------------------------------------------------------------

/// (125.2) against its host twin: after one pseudo step of a steady
/// `localEuler` run, the momentum equation's `r_P` equals
/// `1 / host_delta_t(cfl, V_P, u_P, sqrt(grs T_P))` in f64 on the device's
/// own inputs, cell by cell, for the state the step started from; the
/// energy equation's copy is the same bits. Control: the graded cells and
/// the sheared, heated channel make `r` vary by more than 1.2 across the box
/// (the 2:1 grading alone moves `V^(1/3)` by `2^(1/3) = 1.26`).
#[test]
fn dt_field_matches_host_formula() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = graded_box()?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let n = hm.n_cells;
    let cfl: Scalar = 0.7;
    let mut ctrl = controls(DdtScheme::LocalEuler, 1.0, 1, [0.7, 0.9, 0.3]);
    ctrl.lts = LtsControls { cfl };
    let grs = w64(ctrl.gas.constants().grs);

    let mut s = channel(&g, &gm, &hm)?;
    let mut cs = Compressible::new(&g, &gm, ctrl)?;
    cs.initialise(&g, &mut s)?;
    let u0: Vec<Vec3> = g.download(&s.u.f)?;
    let t0: Vec<Scalar> = g.download(&s.t.f)?;
    cs.step(&g, &mut s)?;

    let r = g.download(cs.local_step().expect("localEuler attaches a local step"))?;
    let re = g.download(&cs.energy.local_step().expect("and the energy equation's").r_delta_t)?;
    let (mut worst, mut r_min, mut r_max) = (0.0f64, f64::INFINITY, 0.0f64);
    for i in 0..n {
        let c = (grs * w64(t0[i])).sqrt();
        let u = [w64(u0[i].x), w64(u0[i].y), w64(u0[i].z)];
        let want = 1.0 / host_delta_t(w64(cfl), w64(hm.v[i]), u, c);
        let got = w64(r[i]);
        worst = worst.max(((got - want) / want).abs());
        r_min = r_min.min(got);
        r_max = r_max.max(got);
        assert_eq!(r[i].to_bits(), re[i].to_bits(), "cell {i}: the energy equation's step differs");
    }
    assert!(worst <= tol_r(1.0e-12), "r against (125.2): worst relative {worst:e}");
    assert!(r_max / r_min > 1.2, "control: r varies only {r_min:e}..{r_max:e}, so the test proves little");
    println!("(125.2): worst relative {worst:.3e} (bar {:.1e}); r in [{r_min:.4e}, {r_max:.4e}] 1/s", tol_r(1.0e-12));
    Ok(())
}

// --------------------------------------------------------------------------
//  Gate 2 - SPEC-LIT 125.7, class R
// --------------------------------------------------------------------------

/// A step field equal to `1/dt` everywhere is the Euler step: on a uniform
/// 4^3 closed box at rest and 300 K, with one cell 2e4 Pa above the rest
/// and gravity on, `r_P = c/(cfl h)` is one value, and one pseudo step of
/// `localEuler` (two outer iterations, relaxation 1) equals one Euler step
/// at `dt = cfl h / c` within the quiescent box's bars. That holds every
/// local term at once: (123.3)'s, (124.1)'s and (125.1)'s time terms and the
/// `dp/dt`, `d(rho K)/dt` and gravity-work source. Controls: the step moves
/// `u` by more than 100 bars, and Euler at `2 dt` misses by more than 100,
/// in either build.
#[test]
fn uniform_step_equals_euler() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = box_mesh([4, 4, 4], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let gas = PerfectGas::air(1.0e5);
    let c0 = (gas.gamma * gas.r_s() * 300.0).sqrt();
    let rho0 = 1.0e5 / (gas.r_s() * 300.0);
    let hot = 21;
    let cfl: Scalar = 4.0;
    let dt = host_delta_t(w64(cfl), 0.25f64.powi(3), [0.0; 3], c0);
    let gravity = Vec3::new(0.0, -9.81, 0.0);

    let run = |ddt: DdtScheme, dt: f64| -> Result<(Fields, f64)> {
        let mut ctrl = controls(ddt, dt as Scalar, 2, [1.0, 1.0, 1.0]);
        ctrl.lts = LtsControls { cfl };
        ctrl.momentum.g = gravity;
        ctrl.energy.g = gravity;
        let mut s = closed_box(&g, &gm, &hm, |i| if i == hot { 2.0e4 } else { 0.0 })?;
        let mut cs = Compressible::new(&g, &gm, ctrl)?;
        cs.initialise(&g, &mut s)?;
        let rep = cs.step(&g, &mut s)?;
        let mi = rep.outers.last().map(|o| o.mass_imbalance).unwrap_or(f64::NAN);
        Ok((fields(&g, &s)?, mi))
    };
    let (lts, mi_l) = run(DdtScheme::LocalEuler, 1.0)?;
    let (eul, mi_e) = run(DdtScheme::Euler, dt)?;
    let (eul2, _) = run(DdtScheme::Euler, 2.0 * dt)?;

    let bar_u = tol_r(1.0e-12) * c0;
    let du = max_diff_v(&lts.u, &eul.u);
    let dp = max_diff(&lts.p, &eul.p);
    let dtt = max_diff(&lts.t, &eul.t);
    let dr = max_diff(&lts.rho, &eul.rho);
    assert!(du <= bar_u, "|u_lts - u_euler| = {du:e}, bar {bar_u:e}");
    assert!(dp <= tol_r(1.0e-12) * 1.0e5, "|p_lts - p_euler| = {dp:e}");
    assert!(dtt <= tol_r(1.0e-12) * 300.0, "|T_lts - T_euler| = {dtt:e}");
    assert!(dr <= tol_r(1.0e-12) * rho0, "|rho_lts - rho_euler| = {dr:e}");

    let moved = max_abs_v(&eul.u);
    assert!(moved > 1.0e2 * bar_u, "control: the bump moved |u| only to {moved:e}");
    let off = max_diff_v(&lts.u, &eul2.u);
    assert!(off > 1.0e2 * bar_u, "control: Euler at 2 dt is within {off:e} of the local step");
    println!(
        "uniform step = Euler at dt {dt:.4e} s: |du| {du:.3e} (bar {bar_u:.3e}), |dp| {dp:.3e}, \
         |dT| {dtt:.3e}, |drho| {dr:.3e}; |u| moved {moved:.3e} = {:.2e} bars; Euler 2dt off by \
         {:.2e} bars; mass imbalance {mi_l:.3e} (local) {mi_e:.3e} (Euler)",
        moved / bar_u,
        off / bar_u
    );
    Ok(())
}

// --------------------------------------------------------------------------
//  Gate 3 - SPEC-LIT 125.4, class P
// --------------------------------------------------------------------------

/// §125.4 argues that at a converged fixed point every ddt difference is
/// zero, so the fixed point does not depend on `CFL`, and asks for two CFLs
/// within `1e-8` in f64. **That row is MISSED, and this test records the
/// miss with its cause** (SPEC-LIT §125.7). The steady channel runs 1500
/// pseudo steps at `CFL` 2, 20 and 1e6, and steady SIMPLE at `alpha_U` 0.9
/// and 0.7, each to its initial-residual floor (in f64 `u` 1e-9, `p` 1e-8,
/// `h_0` 1e-8; the measured floors are about 3e-11, 1e-9 and 5e-9). Each
/// field's difference is taken against that field's largest magnitude.
///
/// The cause: `rAU = V/a_P` (123.5) carries the time term's diagonal
/// `V r_P rho_P`, and the Rhie-Chow flux of (123.6)-(123.7) carries `rAU`
/// at the fixed point, where the ddt differences themselves vanish. So the
/// fixed point moves with `CFL` as steady SIMPLE's moves with `alpha_U`
/// (whose relaxed diagonal enters `rAU` the same way). Asserted in f64: the
/// two CFLs differ by more than the bar (the recorded miss); the existing
/// steady SIMPLE differs between its two `alpha_U` by more than the bar
/// too; the distance to steady SIMPLE at the same `alpha_U` falls as `CFL`
/// grows; and at `CFL` 1e6 `u` and `T` are steady SIMPLE's within the bar.
/// Under `single` the rows are REPORTED (docs/17 section 5.1, rule 4): f32
/// resolves `T` near 300 K to 6e-8 of it, above the bar. Control: after five pseudo
/// steps the two CFLs are far apart, so the step mattered on the way.
#[test]
fn steady_fixed_point_independent_of_cfl() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = graded_box()?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let f64_build = !cfg!(feature = "single");
    let steps = 1500;
    let bar = 1.0e-8;

    // The cell fields and the last outer iteration's initial residuals:
    // the largest velocity component's, the first corrector's, h_0's.
    let run = |ddt: DdtScheme, cfl: Scalar, u_relax: Scalar, steps: usize| -> Result<(Fields, [f64; 3])> {
        let mut ctrl = controls(ddt, 1.0, 1, [u_relax, 0.95, 0.5]);
        ctrl.lts = LtsControls { cfl };
        let mut s = channel(&g, &gm, &hm)?;
        let mut cs = Compressible::new(&g, &gm, ctrl)?;
        cs.initialise(&g, &mut s)?;
        let rep = cs.run(&g, &mut s, steps)?;
        let last = rep.last.as_ref().and_then(|r| r.outers.last()).expect("an outer iteration");
        let ru = last.u.iter().map(|u| w64(u.initial_residual)).fold(0.0, f64::max);
        let rp = last.p.first().map(|p| w64(p.initial_residual)).unwrap_or(f64::NAN);
        Ok((fields(&g, &s)?, [ru, rp, w64(last.h0_residual)]))
    };
    // (u, p, T) differences of b from a, each over a's largest magnitude.
    let rel = |a: &Fields, b: &Fields| -> [f64; 3] {
        [
            max_diff_v(&a.u, &b.u) / max_abs_v(&a.u),
            max_diff(&a.p, &b.p) / max_abs(&a.p),
            max_diff(&a.t, &b.t) / max_abs(&a.t),
        ]
    };

    let legs = [
        ("cfl 2", DdtScheme::LocalEuler, 2.0, 0.9),
        ("cfl 20", DdtScheme::LocalEuler, 20.0, 0.9),
        ("cfl 1e6", DdtScheme::LocalEuler, 1.0e6, 0.9),
        ("SIMPLE alpha_U 0.9", DdtScheme::SteadyState, 1.0, 0.9),
        ("SIMPLE alpha_U 0.7", DdtScheme::SteadyState, 1.0, 0.7),
    ];
    let mut out = Vec::new();
    for (name, ddt, cfl, ur) in legs {
        let (f, r) = run(ddt, cfl, ur, steps)?;
        println!("{name}: initial residuals after {steps} steps u {:.2e} p {:.2e} h0 {:.2e}", r[0], r[1], r[2]);
        if f64_build {
            assert!(
                r[0] <= 1.0e-9 && r[1] <= 1.0e-8 && r[2] <= 1.0e-8,
                "{name}: not at its floor after {steps} steps: u {:e}, p {:e}, h0 {:e}",
                r[0],
                r[1],
                r[2]
            );
        }
        out.push(f);
    }
    let (l2, l20, l1e6, s9, s7) = (&out[0], &out[1], &out[2], &out[3], &out[4]);
    let gate = rel(l2, l20);
    let simple = rel(s9, s7);
    let (to2, to20, to1e6) = (rel(s9, l2), rel(s9, l20), rel(s9, l1e6));
    let worst = gate.iter().cloned().fold(0.0, f64::max);
    println!(
        "the row, cfl 2 vs cfl 20 (u, p, T of max): {:.3e} {:.3e} {:.3e}, bar {bar:.0e}: {} \
         (max |p| {:.4e} Pa)",
        gate[0],
        gate[1],
        gate[2],
        if worst <= bar { "met" } else { "MISSED" },
        max_abs(&l2.p)
    );
    println!("cause: SIMPLE alpha_U 0.9 vs 0.7: {:.3e} {:.3e} {:.3e}", simple[0], simple[1], simple[2]);
    println!(
        "cause: from SIMPLE alpha_U 0.9, u at cfl 2 {:.3e}, 20 {:.3e}, 1e6 {:.3e}; T at 1e6 {:.3e}; p at 1e6 {:.3e}",
        to2[0], to20[0], to1e6[0], to1e6[2], to1e6[1]
    );
    if f64_build {
        assert!(worst > bar, "the recorded miss: cfl 2 vs 20 is now within {worst:e}; update SPEC-LIT 125.7");
        assert!(simple[0] > bar, "the cause: steady SIMPLE moves with alpha_U by {:e} only", simple[0]);
        assert!(to2[0] > to20[0] && to20[0] > to1e6[0], "the cause: the distance to SIMPLE must fall with cfl");
        assert!(to1e6[0] <= bar && to1e6[2] <= bar, "cfl 1e6 is not steady SIMPLE: u {:e}, T {:e}", to1e6[0], to1e6[2]);
    } else {
        println!("f32 rows (docs/17 section 5.1, rule 4): reported, not asserted");
    }

    // Control: five pseudo steps in, the Courant number shows.
    let (a5, _) = run(DdtScheme::LocalEuler, 2.0, 0.9, 5)?;
    let (b5, _) = run(DdtScheme::LocalEuler, 20.0, 0.9, 5)?;
    let e5 = rel(&a5, &b5)[0];
    assert!(e5 > 1.0e3 * bar, "control: after 5 pseudo steps the two runs differ only {e5:e}");
    println!("control: after 5 pseudo steps u differs by {e5:.3e} of max");
    Ok(())
}

// --------------------------------------------------------------------------
//  Gate 4 - SPEC-LIT 125.7, class D
// --------------------------------------------------------------------------

/// `cfl` must be finite and above 0, refused by name with its value; one
/// equation on `localEuler` and the other not is the existing `ddtSchemes
/// differ` refusal; a lone mass-weighted equation still refuses
/// `localEuler`.
#[test]
fn lts_refusals_name_the_setting() -> Result<()> {
    let Some(g) = gpu() else { return Ok(()) };
    let hm = graded_box()?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let base = controls(DdtScheme::LocalEuler, 1.0, 1, [0.7, 0.9, 0.3]);
    for cfl in [0.0, -1.0, Scalar::NAN, Scalar::INFINITY] {
        let ctrl = CompressibleControls { lts: LtsControls { cfl }, ..base };
        match Compressible::new(&g, &gm, ctrl) {
            Err(Error::Refused(msg)) => assert!(msg.contains("cfl"), "cfl {cfl}: {msg}"),
            Err(e) => panic!("cfl {cfl}: refused by the wrong kind: {e}"),
            Ok(_) => panic!("cfl {cfl} was accepted"),
        }
    }
    let mut one = base;
    one.energy.ddt = DdtScheme::SteadyState;
    match Compressible::new(&g, &gm, one) {
        Err(Error::Refused(msg)) => assert!(msg.contains("ddtSchemes differ"), "{msg}"),
        Err(e) => panic!("one localEuler: refused by the wrong kind: {e}"),
        Ok(_) => panic!("localEuler on momentum alone was accepted"),
    }
    match ofgpu::compressible::momentum::RhoMomentum::new(&g, &gm, base.momentum) {
        Err(Error::Refused(msg)) => assert!(msg.contains("localEuler"), "{msg}"),
        Err(e) => panic!("lone momentum: refused by the wrong kind: {e}"),
        Ok(_) => panic!("a lone momentum equation accepted localEuler"),
    }
    // Control: the base controls build.
    Compressible::new(&g, &gm, base)?;
    Ok(())
}
