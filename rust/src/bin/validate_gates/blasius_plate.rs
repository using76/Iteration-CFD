// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate 132-L (SPEC-LIT §144.2): the flat-plate harness reproduces Blasius,
//! live.**
//!
//! VAL-30's plate (§144.1) is solved laminar and steady on three meshes, and
//! its wall shear is held against the Blasius similarity solution, which is
//! an EXACT solution of the boundary-layer equations: H. Blasius, *Z. Math.
//! Phys.* 56 (1908) 1-37. The reference is not a table copied from anywhere:
//! `f''' + f f''/2 = 0` is integrated here by RK4 with a secant shot on
//! `f''(0)`, as §88.7 does, and its `f''(0)` is itself a checked row.
//!
//! The rows: every plate face with `2e4 <= Re_x <= Re_L` within 3 % of
//! `C_f = 2 f''(0) Re_x^(-1/2)` on the finest mesh; `u/U` within 0.02 of
//! `f'(eta)` up the column at `Re_x = 1e5`; and §94's study of `C_f` at
//! `Re_x = 1e5` over the three meshes with the (94.10) metric. A miss fails
//! its row and registers the verdict with that study (§69, §94.3).
//!
//! Every reduction is f64 whatever `Scalar` is: the field is widened once on
//! download and the geometry is `geom64()` (§118.2).
//!
//! No GPL-licensed source was consulted.

use std::time::Instant;

use ofgpu::blockgen::{build_flat_plate_mesh, FlatPlateSpec};
use ofgpu::field::{BcKind, GpuScalarField, GpuVectorField};
use ofgpu::fv::{DivScheme, SnGradScheme};
use ofgpu::io::case::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::mesh::HostMesh;
use ofgpu::momentum::{BuoyancyCoeffs, MomentumControls};
use ofgpu::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use ofgpu::simple::{nan_propagating_max, Simple, SimpleControls};
use ofgpu::vv::{self, GridStudy, Level};
use ofgpu::{Error, Gpu, GpuMesh, Label, Result, Scalar, Vec3};

use crate::{Checks, GateReport, How, Uncertainty, Verdict};

/// The gate's scope and registry name.
pub(super) const GATE: &str = "SPEC-LIT 144.2 Gate 132-L Blasius flat plate";

/// The three meshes, finest first: `(n_upstream, n_plate, n_y)` on
/// `FlatPlateSpec::laminar()`'s box, end-cell ratios and free stream. Each
/// coarser level halves every count.
const LEVELS: [(usize, usize, usize); 3] = [(32, 128, 96), (16, 64, 48), (8, 32, 24)];

/// The `Re_x` band whose every plate face is held to (144.2a).
const RE_X_LO: f64 = 2e4;
/// The station of the profile row and of the §94 study.
const RE_X_STATION: f64 = 1e5;
/// The profile row reads the column up to this `eta` (`f'(8) = 1 - 1e-5`):
/// the layer, not the free stream above it.
const ETA_PROFILE_MAX: f64 = 8.0;
/// (144.2a)'s and (144.2b)'s bars, the plan's.
const CF_TOL: f64 = 0.03;
const PROFILE_TOL: f64 = 0.02;

/// The stopping rule: every `CHECK_EVERY` SIMPLE iterations, the largest
/// relative change of `C_f` over the band's faces since the last check.
const CHECK_EVERY: usize = 50;
const MIN_ITERS: usize = 200;
const STOP_TOL: f64 = 1e-6;
const MAX_ITERS: usize = 20000;

fn w(x: Scalar) -> f64 {
    f64::from(x)
}

fn s(x: f64) -> Scalar {
    x as Scalar
}

// ---------------------------------------------------------------------------
//  The reference: Blasius, integrated live
// ---------------------------------------------------------------------------

/// `f'` and `f''` on `eta_i = i h`, `i = 0..=n`, from the shot `f''(0)`.
pub(super) struct Blasius {
    pub fpp0: f64,
    h: f64,
    fp: Vec<f64>,
    fpp: Vec<f64>,
}

/// §88.7: `f''' + f f''/2 = 0` as the first-order system `y = (f, f', f'')`.
fn blasius_rhs(y: [f64; 3]) -> [f64; 3] {
    [y[1], y[2], -0.5 * y[0] * y[2]]
}

/// RK4 from `eta = 0` with `f''(0) = fpp0`; the whole trajectory.
fn blasius_march(fpp0: f64, h: f64, n: usize) -> Vec<[f64; 3]> {
    let mut y = [0.0, 0.0, fpp0];
    let mut out = Vec::with_capacity(n + 1);
    out.push(y);
    for _ in 0..n {
        let k1 = blasius_rhs(y);
        let k2 = blasius_rhs([y[0] + 0.5 * h * k1[0], y[1] + 0.5 * h * k1[1], y[2] + 0.5 * h * k1[2]]);
        let k3 = blasius_rhs([y[0] + 0.5 * h * k2[0], y[1] + 0.5 * h * k2[1], y[2] + 0.5 * h * k2[2]]);
        let k4 = blasius_rhs([y[0] + h * k3[0], y[1] + h * k3[1], y[2] + h * k3[2]]);
        for j in 0..3 {
            y[j] += h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
        }
        out.push(y);
    }
    out
}

impl Blasius {
    /// The secant shot on `f''(0)` driving `f'(eta_max)` to 1, at
    /// `eta_max = 10`, `h = 1e-3`.
    pub fn solve() -> Self {
        const ETA_MAX: f64 = 10.0;
        const N: usize = 10_000;
        let h = ETA_MAX / N as f64;
        let miss = |a: f64| blasius_march(a, h, N)[N][1] - 1.0;
        let (mut a, mut b) = (0.3f64, 0.4f64);
        let (mut fa, mut fb) = (miss(a), miss(b));
        for _ in 0..60 {
            if fb.abs() < 1e-14 || (fb - fa).abs() < f64::MIN_POSITIVE {
                break;
            }
            let c = b - fb * (b - a) / (fb - fa);
            a = b;
            fa = fb;
            b = c;
            fb = miss(b);
        }
        let traj = blasius_march(b, h, N);
        Self {
            fpp0: b,
            h,
            fp: traj.iter().map(|y| y[1]).collect(),
            fpp: traj.iter().map(|y| y[2]).collect(),
        }
    }

    /// `f'(eta)`: the cubic Hermite interpolant of the RK4 nodes, with `f''`
    /// as its slope; `1` beyond the shot's `eta_max`, where the shot put it.
    pub fn f_prime(&self, eta: f64) -> f64 {
        let n = self.fp.len() - 1;
        if eta <= 0.0 {
            return 0.0;
        }
        let t = eta / self.h;
        if t >= n as f64 {
            return 1.0;
        }
        let i = t.floor() as usize;
        let u = t - i as f64;
        let (y0, y1) = (self.fp[i], self.fp[i + 1]);
        let (m0, m1) = (self.fpp[i] * self.h, self.fpp[i + 1] * self.h);
        let (u2, u3) = (u * u, u * u * u);
        (2.0 * u3 - 3.0 * u2 + 1.0) * y0
            + (u3 - 2.0 * u2 + u) * m0
            + (-2.0 * u3 + 3.0 * u2) * y1
            + (u3 - u2) * m1
    }

    /// (144.2a): `C_f = 2 f''(0) Re_x^(-1/2)`.
    pub fn cf(&self, re_x: f64) -> f64 {
        2.0 * self.fpp0 / re_x.sqrt()
    }
}

// ---------------------------------------------------------------------------
//  One mesh, solved
// ---------------------------------------------------------------------------

/// One plate face: its centre `x`, `Re_x` there, and its `C_f`.
#[derive(Clone, Copy, Debug)]
struct WallFace {
    x: f64,
    re_x: f64,
    cf: f64,
}

struct PlateRun {
    cells: (usize, usize, usize),
    n_cells: usize,
    /// (94.1) in 2-D: `sqrt(A/N)`, `A` the x-y area.
    h: f64,
    iterations: usize,
    converged: bool,
    last_change: f64,
    residual: f64,
    seconds: f64,
    faces: Vec<WallFace>,
    /// `C_f` at `Re_x = 1e5`, interpolated (144.2c).
    cf_station: f64,
    /// The station column: its `x`, and `(eta, u/U)` per cell.
    station_x: f64,
    column: Vec<(f64, f64)>,
}

/// The spec of one level: `laminar()` with its counts replaced.
fn level_spec(n: (usize, usize, usize)) -> FlatPlateSpec {
    let mut spec = FlatPlateSpec::laminar();
    spec.n_upstream = n.0;
    spec.n_plate = n.1;
    spec.n_y = n.2;
    spec
}

/// The patch index of `name`, or a refusal naming it.
fn patch_index(hm: &HostMesh, name: &str) -> Result<usize> {
    hm.patches
        .iter()
        .position(|p| p.name == name)
        .ok_or_else(|| Error::Config(format!("Gate 132-L: the plate mesh has no patch {name}")))
}

/// What each patch carries: `(U kind, U value, p kind)`. The free stream is
/// the inlet; the outlet and the top are open at `p = 0` so the free stream
/// carries no pressure gradient and the displacement leaves through the top
/// (`cases/flatPlateLaminar.jsonc`'s conditions).
fn patch_rule(name: &str, u_inf: f64) -> Option<(BcKind, Vec3, BcKind)> {
    let zero = Vec3::ZERO;
    Some(match name {
        "inlet" => (BcKind::FixedValue, Vec3::new(s(u_inf), 0.0, 0.0), BcKind::ZeroGradient),
        "outlet" | "top" => (BcKind::ZeroGradient, zero, BcKind::FixedValue),
        "symmetry" => (BcKind::Symmetry, zero, BcKind::Symmetry),
        "plate" => (BcKind::FixedValue, zero, BcKind::ZeroGradient),
        "back" | "front" => (BcKind::Empty, zero, BcKind::Empty),
        _ => return None,
    })
}

fn set_bcs(gpu: &Gpu, sim: &mut Simple<'_>, hm: &HostMesh, u_inf: f64) -> Result<()> {
    let nbf = hm.n_boundary_faces;
    let mut u_kind = vec![BcKind::ZeroGradient as Label; nbf];
    let mut u_fr = vec![0.0 as Scalar; nbf];
    let mut u_rv = vec![Vec3::ZERO; nbf];
    let mut p_kind = vec![BcKind::ZeroGradient as Label; nbf];
    let mut p_fr = vec![0.0 as Scalar; nbf];
    for p in &hm.patches {
        let Some((uk, uv, pk)) = patch_rule(&p.name, u_inf) else {
            return Err(Error::Config(format!(
                "Gate 132-L: patch {} has no boundary rule",
                p.name
            )));
        };
        for i in p.start..p.start + p.size {
            u_kind[i] = uk as Label;
            p_kind[i] = pk as Label;
            if uk == BcKind::FixedValue {
                u_fr[i] = 1.0;
                u_rv[i] = uv;
            }
            if pk == BcKind::FixedValue {
                p_fr[i] = 1.0;
            }
        }
    }
    let u: &mut GpuVectorField = sim.u_mut();
    gpu.write(&mut u.bc_kind, &u_kind)?;
    gpu.write(&mut u.fr, &u_fr)?;
    gpu.write(&mut u.ref_value, &u_rv)?;
    gpu.write(&mut u.ref_grad, &vec![Vec3::ZERO; nbf])?;
    let p: &mut GpuScalarField = sim.p_mut();
    gpu.write(&mut p.bc_kind, &p_kind)?;
    gpu.write(&mut p.fr, &p_fr)?;
    gpu.write(&mut p.ref_value, &vec![0.0 as Scalar; nbf])?;
    gpu.write(&mut p.ref_grad, &vec![0.0 as Scalar; nbf])
}

/// Steady SIMPLE, `alpha_U = 0.7` and `alpha_p = 0.3` (§5.2), linearUpwind
/// on `div(phi,U)` as `cases/flatPlateLaminar.jsonc` names it.
fn controls(nu: f64) -> SimpleControls {
    SimpleControls {
        momentum: MomentumControls {
            nu: s(nu),
            u_solver: SolverControls {
                solver: LinearSolverKind::PBiCGStab,
                tolerance: 1e-12,
                rel_tol: 0.1,
                max_iter: 200,
                precon: Preconditioner::Dilu,
                check_interval: 5,
                ..SolverControls::default()
            },
            u_relax: 0.7,
            div_scheme: DivScheme::LinearUpwind,
            bounded_convection: true,
            sn_grad: SnGradScheme::Corrected,
            variable_viscosity_stress: false,
            ..MomentumControls::default()
        },
        p_solver: SolverControls {
            solver: LinearSolverKind::PCG,
            tolerance: 1e-12,
            rel_tol: 0.05,
            max_iter: 2000,
            precon: Preconditioner::Dic,
            check_interval: 5,
            ..SolverControls::default()
        },
        p_relax: 0.3,
        n_non_orth_correctors: 0,
        n_correctors: 1,
        n_outer_correctors: 1,
        momentum_predictor: true,
        report_continuity: false,
    }
}

/// The plate faces' `C_f` from the cell velocities, in f64:
/// `tau_w = nu u_P,x / d`, `d = (c_P - c_f) . n_hat`, `C_f = 2 tau_w / U^2`
/// (144.2d) - the wall cell's one-sided gradient, the one the momentum
/// equation's own wall flux uses.
fn wall_faces(hm: &HostMesh, plate: usize, u: &[Vec3], u_inf: f64, nu: f64) -> Vec<WallFace> {
    let g = hm.geom64();
    let p = &hm.patches[plate];
    (p.start..p.start + p.size)
        .map(|i| {
            let cell = hm.b_face_cells[i] as usize;
            let (cf_, cp, sf) = (g.b_cf[i], g.c[cell], g.b_sf[i]);
            let mag = (sf.x * sf.x + sf.y * sf.y + sf.z * sf.z).sqrt();
            let d = ((cp.x - cf_.x) * sf.x + (cp.y - cf_.y) * sf.y + (cp.z - cf_.z) * sf.z).abs() / mag;
            let tau = nu * w(u[cell].x) / d;
            WallFace { x: cf_.x, re_x: u_inf * cf_.x / nu, cf: 2.0 * tau / (u_inf * u_inf) }
        })
        .collect()
}

/// (144.2c): `C_f` at `Re_x = re`, from the two faces bracketing it, by
/// linear interpolation of `C_f sqrt(Re_x)` - the quantity Blasius makes
/// constant - in `x`.
fn cf_at(faces: &[WallFace], re: f64) -> Result<f64> {
    for pair in faces.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if a.re_x <= re && re <= b.re_x {
            let t = (re - a.re_x) / (b.re_x - a.re_x);
            let ga = a.cf * a.re_x.sqrt();
            let gb = b.cf * b.re_x.sqrt();
            return Ok((ga + t * (gb - ga)) / re.sqrt());
        }
    }
    Err(Error::Config(format!("Gate 132-L: no two plate faces bracket Re_x = {re}")))
}

/// The largest relative `C_f` change between two snapshots over the band;
/// a NaN anywhere is the result (`f64::max` would drop it).
fn cf_change(a: &[WallFace], b: &[WallFace], re_l: f64) -> f64 {
    a.iter()
        .zip(b)
        .filter(|(f, _)| f.re_x >= RE_X_LO && f.re_x <= re_l)
        .map(|(f, g)| ((g.cf - f.cf) / f.cf).abs())
        .fold(0.0f64, |m, d| if m.is_nan() || d.is_nan() { f64::NAN } else { m.max(d) })
}

fn run_level(gpu: &Gpu, n: (usize, usize, usize)) -> Result<PlateRun> {
    let started = Instant::now();
    let spec = level_spec(n);
    let (u_inf, nu) = (spec.u_inf, spec.nu);
    let re_l = u_inf * spec.length / nu;
    let hm = build_flat_plate_mesh(&spec)?;
    let plate = patch_index(&hm, "plate")?;
    let gm = GpuMesh::upload(gpu, &hm)?;
    let ctrl = controls(nu);
    let mut sim = Simple::new(
        gpu,
        &hm,
        &gm,
        ctrl,
        BuoyancyCoeffs { g: Vec3::ZERO, ..BuoyancyCoeffs::default() },
    )?;
    set_bcs(gpu, &mut sim, &hm, u_inf)?;

    // Seed: the free stream in every cell, its flux through every face
    // whose velocity condition lets flow through (the inlet, outlet and
    // top; the wall, the symmetry plane and the empties carry none), and
    // a level pressure.
    let free = Vec3::new(s(u_inf), 0.0, 0.0);
    gpu.write(&mut sim.u_mut().f, &vec![free; hm.n_cells])?;
    gpu.write(&mut sim.p_mut().f, &vec![0.0 as Scalar; hm.n_cells])?;
    let phi_f: Vec<Scalar> = hm.sf.iter().map(|sf| free.dot(*sf)).collect();
    gpu.write(&mut sim.phi_mut().f, &phi_f)?;
    let mut phi_b = vec![0.0 as Scalar; hm.n_boundary_faces];
    for p in &hm.patches {
        if matches!(p.name.as_str(), "inlet" | "outlet" | "top") {
            for i in p.start..p.start + p.size {
                phi_b[i] = free.dot(hm.b_sf[i]);
            }
        }
    }
    gpu.write(&mut sim.phi_mut().bf, &phi_b)?;
    sim.initialise(gpu)?;

    // Laminar: zero eddy viscosity with `calculated` boundary values, and
    // a uniform T nothing reads while the buoyancy vector is zero.
    let mut nut = GpuScalarField::zeros(gpu, &gm, "nut")?;
    gpu.write(&mut nut.bc_kind, &vec![BcKind::Calculated as Label; hm.n_boundary_faces])?;
    let mut t = GpuScalarField::zeros(gpu, &gm, "T")?;
    gpu.write(&mut t.bc_kind, &vec![BcKind::ZeroGradient as Label; hm.n_boundary_faces])?;
    let mut backend = PbicgstabBackend::new(ctrl.p_solver);
    backend.setup(gpu, &hm, &gm, &SystemProbe::default())?;

    let mut last = wall_faces(&hm, plate, &gpu.download(&sim.u().f)?, u_inf, nu);
    let (mut iterations, mut converged) = (MAX_ITERS, false);
    let (mut last_change, mut residual) = (f64::INFINITY, f64::INFINITY);
    for it in 1..=MAX_ITERS {
        let perf = sim.solve_step(gpu, &mut backend, &nut, &t)?;
        let res = nan_propagating_max(
            nan_propagating_max(perf.last.u[0].initial_residual, perf.last.u[1].initial_residual),
            perf.last.p.initial_residual,
        );
        residual = w(res);
        if !residual.is_finite() {
            return Err(Error::Diverged {
                iteration: it,
                what: format!("Gate 132-L plate {n:?}: initial residual {residual}"),
            });
        }
        if it % CHECK_EVERY == 0 {
            let now = wall_faces(&hm, plate, &gpu.download(&sim.u().f)?, u_inf, nu);
            last_change = cf_change(&last, &now, re_l);
            last = now;
            if it >= MIN_ITERS && last_change <= STOP_TOL {
                iterations = it;
                converged = true;
                break;
            }
        }
    }
    let u = gpu.download(&sim.u().f)?;
    let faces = wall_faces(&hm, plate, &u, u_inf, nu);
    let cf_station = cf_at(&faces, RE_X_STATION)?;

    // The station column: the cells above the plate face nearest Re_x = 1e5.
    let g = hm.geom64();
    let pp = &hm.patches[plate];
    let x_station = RE_X_STATION * nu / u_inf;
    let near = (pp.start..pp.start + pp.size)
        .min_by(|&a, &b| {
            (g.b_cf[a].x - x_station).abs().total_cmp(&(g.b_cf[b].x - x_station).abs())
        })
        .ok_or_else(|| Error::Config("Gate 132-L: the plate patch is empty".to_string()))?;
    let station_x = g.c[hm.b_face_cells[near] as usize].x;
    let mut column: Vec<(f64, f64)> = (0..hm.n_cells)
        .filter(|&k| (g.c[k].x - station_x).abs() <= 1e-9 * spec.length)
        .map(|k| (g.c[k].y * (u_inf / (nu * station_x)).sqrt(), w(u[k].x) / u_inf))
        .collect();
    column.sort_by(|a, b| a.0.total_cmp(&b.0));

    let area: f64 = g.v.iter().sum::<f64>() / spec.depth;
    let h = (area / hm.n_cells as f64).sqrt();
    Ok(PlateRun {
        cells: n,
        n_cells: hm.n_cells,
        h,
        iterations,
        converged,
        last_change,
        residual,
        seconds: started.elapsed().as_secs_f64(),
        faces,
        cf_station,
        station_x,
        column,
    })
}

// ---------------------------------------------------------------------------
//  The gate
// ---------------------------------------------------------------------------

/// Gate 132-L, live: the reference row, three solves, the finest mesh's two
/// rows and §94's study at `Re_x = 1e5`.
pub(super) fn check_blasius_plate(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    let b = Blasius::solve();
    c.note(&format!(
        "Gate 132-L: Blasius shot f''(0) = {:.8} (eta_max 10, RK4 h 1e-3), so C_f sqrt(Re_x) = {:.6}",
        b.fpp0,
        2.0 * b.fpp0
    ));
    c.check(
        "Gate 132-L: |f''(0) - 0.332057|, the reference itself (SPEC-LIT 88.7)",
        s((b.fpp0 - 0.332057).abs()),
        s(1e-6),
    );

    let mut runs = Vec::with_capacity(LEVELS.len());
    for n in LEVELS {
        let r = run_level(gpu, n)?;
        c.note(&format!(
            "Gate 132-L mesh {}+{} x {}: {} cells, h {:.4e}, {} SIMPLE iterations, stopping rule met {} \
             (C_f change {:.2e} per {CHECK_EVERY}), initial residual {:.2e}, C_f(Re_x 1e5) {:.6e}, {:.1} s",
            r.cells.0, r.cells.1, r.cells.2, r.n_cells, r.h, r.iterations, r.converged,
            r.last_change, r.residual, r.cf_station, r.seconds
        ));
        c.require(
            &format!(
                "Gate 132-L: mesh {}+{} x {} meets the C_f stopping rule ({STOP_TOL:.0e} per {CHECK_EVERY} iterations) within {MAX_ITERS}",
                r.cells.0, r.cells.1, r.cells.2
            ),
            r.converged,
        );
        runs.push(r);
    }

    // ---- (144.2a): every band face of the finest mesh -------------------
    let fine = &runs[0];
    let spec = level_spec(fine.cells);
    let re_l = spec.u_inf * spec.length / spec.nu;
    let band: Vec<&WallFace> =
        fine.faces.iter().filter(|f| f.re_x >= RE_X_LO && f.re_x <= re_l).collect();
    let worst_cf = band
        .iter()
        .map(|f| (f, (f.cf / b.cf(f.re_x) - 1.0).abs()))
        .fold(None::<(&WallFace, f64)>, |m, (f, e)| match m {
            Some((_, me)) if me.is_nan() || me >= e => m,
            _ => Some((*f, e)),
        });
    let (cf_err, cf_where) = match worst_cf {
        Some((f, e)) => (e, format!("x {:.4} m, Re_x {:.4e}, C_f {:.5e} vs {:.5e}", f.x, f.re_x, f.cf, b.cf(f.re_x))),
        None => (f64::INFINITY, "no plate face in the band".to_string()),
    };
    for re in [2e4, 5e4, 1e5, 2e5 * 0.999] {
        if let Ok(v) = cf_at(&fine.faces, re) {
            c.note(&format!(
                "Gate 132-L finest: Re_x {re:.3e}  C_f {v:.5e}  Blasius {:.5e}  ratio {:.4}",
                b.cf(re),
                v / b.cf(re)
            ));
        }
    }
    c.note(&format!("Gate 132-L finest: {} plate faces in 2e4 <= Re_x <= {re_l:.0e}; worst at {cf_where}", band.len()));
    c.check(
        "Gate 132-L: max |C_f/C_f,Blasius - 1|, finest mesh, 2e4 <= Re_x <= Re_L (144.2a)",
        s(cf_err),
        s(CF_TOL),
    );

    // ---- (144.2b): the profile at Re_x = 1e5 -----------------------------
    let in_layer: Vec<&(f64, f64)> =
        fine.column.iter().filter(|(eta, _)| *eta <= ETA_PROFILE_MAX).collect();
    let worst_u = in_layer
        .iter()
        .map(|&&(eta, u)| (eta, u, (u - b.f_prime(eta)).abs()))
        .fold((0.0, 0.0, if in_layer.is_empty() { f64::INFINITY } else { 0.0 }), |m, x| {
            if !m.2.is_nan() && (x.2.is_nan() || x.2 > m.2) { x } else { m }
        });
    c.note(&format!(
        "Gate 132-L finest: column at x {:.4} m (Re_x {:.4e}), {} cells with eta <= {ETA_PROFILE_MAX}; \
         worst at eta {:.3}: u/U {:.5} vs f' {:.5}",
        fine.station_x,
        fine.station_x * spec.u_inf / spec.nu,
        in_layer.len(),
        worst_u.0,
        worst_u.1,
        b.f_prime(worst_u.0)
    ));
    c.check(
        "Gate 132-L: max |u/U - f'(eta)|, finest mesh, eta <= 8 at Re_x = 1e5 (144.2b)",
        s(worst_u.2),
        s(PROFILE_TOL),
    );

    // ---- (144.2c): the §94 study of C_f at Re_x = 1e5 ---------------------
    let levels: Vec<Level> = runs.iter().map(|r| Level { h: s(r.h), value: s(r.cf_station) }).collect();
    let d = b.cf(RE_X_STATION);
    let study: std::result::Result<GridStudy, String> =
        vv::grid_study(&levels).map_err(|e| e.to_string());
    match &study {
        Ok(st) => {
            c.note(&format!("Gate 132-L C_f(Re_x 1e5) study: {}", st.one_line()));
            let v = vv::validation(s(fine.cf_station), s(d), st.u_fine, 0.0, 0.0);
            c.note(&format!(
                "Gate 132-L {} (u_input 0: the inputs are exact; u_D 0: Blasius is exact)",
                v.one_line("C_f at Re_x = 1e5")
            ));
        }
        Err(e) => c.note(&format!("Gate 132-L C_f(Re_x 1e5) study refused: {e}")),
    }
    c.require("Gate 132-L: the 94 study reads the three levels of C_f at Re_x = 1e5 (144.2c)", study.is_ok());

    let missed = !(cf_err <= CF_TOL) || !(worst_u.2 <= PROFILE_TOL);
    if missed {
        let uncertainty = match study {
            Ok(st) => Uncertainty::Study(st),
            Err(_) => Uncertainty::SingleMesh("the 94 study refused the three levels; the finest mesh alone"),
        };
        c.report(GateReport {
            verdict: Verdict::Misses,
            how: How::Live,
            gate: GATE,
            against: "the Blasius (1908) similarity solution, integrated live (SPEC-LIT 88.7)",
            headline: format!(
                "C_f within {:.2} % of Blasius at worst (bar 3 %), u/U within {:.4} of f'(eta) (bar 0.02)",
                100.0 * cf_err,
                worst_u.2
            ),
            detail: vec![format!("worst C_f face: {cf_where}")],
            uncertainty: Some(uncertainty),
        });
    }
    Ok(())
}
