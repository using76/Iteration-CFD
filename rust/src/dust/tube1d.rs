// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The 1-D flame tube that verifies the progress-variable equation before
//! any driver exists - SPEC-LIT §155.8 (docs/17 DST-04). A test harness:
//! no driver calls it.
//!
//! The tube runs along x on a blockgen line of cubes (`dx = dy = dz`, so
//! the flame-Courant length V^{1/3} of (155.9) is dx), `empty` across.
//! The burned end x_lo is open and the fresh end x_hi is closed with the
//! fresh gas at rest there: the flame propagates towards +x into it. The
//! density is the BML relation (155.1) of the current iterate, and the
//! face mass flux comes from discrete continuity as a host cumulative sum
//! from the fresh end,
//!
//! ```text
//! m_{j-1/2} = m_{j+1/2} + V_j (rho_j^{n+1} - rho_j^n) / (dt A) ,   m_{N+1/2} = 0
//! ```
//!
//! (§155.8's sum, written from the fresh end), so the c~ equation sees a
//! flux that conserves the BML mass exactly under its Euler ddt. Each step
//! iterates c~ -> rho -> flux -> c~ until c~ changes by less than the
//! outer tolerance. The FSC closures come from the DST-03 kernel
//! ([`crate::dust::device::fsc`]) with frozen uniform turbulence, the
//! temperature from the published closure T = T_u (1 + (sigma - 1) c~)
//! (153.1), and the molecular diffusivity from the law the case states.
//!
//! Written from:
//!   SPEC-LIT §155.8 - the harness: the BML density and the cumulative sum
//!   SPEC-LIT §155.1-§155.6, §153.1, §154 - the relations it composes
//!   SPEC-LIT §160.5 - [`run_aged`], the same tube with the flame-age clock
//!     on: late kernels seed fresh ignitions, one alpha solve follows every
//!     c~ solve with the same coupling, the closures read the predicted age
//!     a* of (160.4), and each step leaves one row of the alpha budget
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use crate::blockgen::{self, BlockSpec, GradedAxis};
use crate::device::{DevBuf, Gpu};
use crate::dust::age;
use crate::dust::device::{
    fsc as launch_fsc, fsc_aged as launch_fsc_aged, flame_courant, ignition as launch_ignition,
    reynolds_mean, FscFields, FscInputsDev, FscKernels, ProgressKernels,
};
use crate::dust::fsc::{development_time, FscConstants};
use crate::dust::ignition::IgnitionSource;
use crate::dust::progress::{bml_density, ProgressCoupling, ProgressEquation};
use crate::error::{Error, Result};
use crate::field::{GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use crate::field_ops::{self, FieldKernels};
use crate::io::case::TurbulenceControls;
use crate::mesh::{GpuMesh, HostMesh};
use crate::{Scalar, Vec3};

/// The molecular diffusivity kappa of (155.3) along the tube.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KappaLaw {
    /// kappa = 0: the exact planar solution (155.5) has no kappa.
    Zero,
    /// rho kappa = rho_u kappa_u (n = 1 of §153.2): kappa = kappa_u rho_u / rho.
    ConstantRhoKappa { kappa_u: f64 },
}

/// Frozen uniform turbulence for the FSC kernel: k, omega and nu_t.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrozenTurbulence {
    pub k: f64,
    pub omega: f64,
    pub nu_t: f64,
}

/// One tube.
#[derive(Debug, Clone)]
pub struct TubeSpec {
    pub n_cells: usize,
    /// The open burned end, m.
    pub x_lo: f64,
    /// The closed fresh end, m.
    pub x_hi: f64,
    /// sigma = rho_u / rho_b of (155.1).
    pub sigma: f64,
    /// rho_u, kg/m^3 (the pressure is constant, so rho_u(p) = rho_u0).
    pub rho_u: f64,
    /// T_u of the published closure, K.
    pub t_u: f64,
    pub kappa: KappaLaw,
    pub fsc: FscConstants,
    /// `None`: D_t = U_t = 0 (k = omega = nu_t = 0 in the kernel).
    pub turbulence: Option<FrozenTurbulence>,
    /// `false` zeroes K: the planar solution (155.5) has Q = 0.
    pub with_q: bool,
    pub ignition: Option<IgnitionSource>,
    /// The origin of t_fd (§154.2): t_0 of the ignition source, or 0.
    pub t_fd_origin: f64,
    /// The fixed step, s.
    pub dt: f64,
    /// The outer iterations per step, at most.
    pub max_outer: usize,
    /// The outer loop stops when max |c~ change| falls below this.
    pub outer_tol: f64,
    /// The c~ equation's controls (ddt, convection, solver).
    pub ctrl: TurbulenceControls,
}

/// The tube's state at the end of a run.
#[derive(Debug, Clone)]
pub struct TubeRun {
    pub t: f64,
    pub steps: usize,
    /// Cell centres in world coordinates, ascending, m.
    pub x: Vec<f64>,
    /// c~ and cbar at the centres.
    pub c: Vec<f64>,
    pub cbar: Vec<f64>,
    /// Cells counted outside [0, 1] (by more than the margin) in each
    /// step's result, its last outer iterate, summed over the run (§155.2).
    pub out_of_bounds: u64,
    /// The largest distance of c~ outside [0, 1] over the steps' results.
    pub max_excursion: f64,
    /// The largest measured Co_f of (155.9) over the run.
    pub max_co_f: f64,
    /// The largest number of outer iterations a step used, and the total.
    pub max_outer_used: usize,
    pub outer_total: usize,
}

#[allow(clippy::unnecessary_cast, clippy::useless_conversion)]
fn wide(v: Scalar) -> f64 {
    v as f64
}

/// A fresh ignition seeded mid-run (SPEC-LIT §160.5): at the top of the
/// first step whose time has reached `t_ig`, every cell's c~ becomes
/// `max(c~, profile(x))` and the cells it burns start to age from that
/// step.
#[derive(Debug, Clone, Copy)]
pub struct LateKernel {
    /// The kernel's ignition time, s.
    pub t_ig: f64,
    /// The seeded c~ profile, x in m.
    pub profile: fn(f64) -> f64,
}

/// The age option's settings for a run: the floor c_a and the late kernels,
/// applied in the order they are listed.
#[derive(Debug, Clone)]
pub struct AgeOptions {
    /// The floor c_a of (160.1).
    pub c_a: f64,
    /// Fresh ignitions seeded mid-run.
    pub late: Vec<LateKernel>,
}

/// What [`run_aged`]'s sampler sees after every step: x ascending, c~ and
/// the local age a = alpha/max(c~, c_a) in the same order.
pub struct AgeSample<'a> {
    pub t: f64,
    pub x: &'a [f64],
    pub c: &'a [f64],
    pub age: &'a [f64],
}

/// One step's row of the alpha budget (SPEC-LIT §160.5's
/// `age_budget_closes`), the Euler form:
/// `storage + outflow = clock + inherit`.
#[derive(Debug, Clone, Copy)]
pub struct AgeBudgetRow {
    /// The step's end time, s.
    pub t: f64,
    /// sum V (rho alpha - rho0 alpha0)/dt.
    pub storage: f64,
    /// sum over boundary faces of phi_m.bf alpha.bf.
    pub outflow: f64,
    /// sum V sua.
    pub clock: f64,
    /// -sum V spa alpha.
    pub inherit: f64,
}

impl AgeBudgetRow {
    /// `storage + outflow - clock - inherit`: zero for the exact solution.
    pub fn residual(&self) -> f64 {
        self.storage + self.outflow - self.clock - self.inherit
    }

    /// The sum of the four terms' magnitudes, the row's scale.
    pub fn scale(&self) -> f64 {
        self.storage.abs() + self.outflow.abs() + self.clock.abs() + self.inherit.abs()
    }
}

/// A run with the age on: the plain run, the final alpha and local age
/// ascending in x, and each step's budget row.
#[derive(Debug, Clone)]
pub struct AgedRun {
    pub run: TubeRun,
    pub alpha: Vec<f64>,
    pub age: Vec<f64>,
    pub budget: Vec<AgeBudgetRow>,
}

/// The tube's line of cubes, `empty` across.
pub fn tube_mesh(spec: &TubeSpec) -> Result<HostMesh> {
    if spec.n_cells < 2 || !(spec.x_hi > spec.x_lo) {
        return Err(Error::Refused(format!(
            "dust::tube1d: a tube needs at least 2 cells and x_hi > x_lo (got {} cells on \
             [{}, {}])",
            spec.n_cells, spec.x_lo, spec.x_hi
        )));
    }
    let dx = (spec.x_hi - spec.x_lo) / spec.n_cells as f64;
    let axis = |lo: f64, hi: f64, n: usize| GradedAxis {
        lo: lo as Scalar,
        hi: hi as Scalar,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    let bs = BlockSpec {
        x: axis(spec.x_lo, spec.x_hi, spec.n_cells),
        y: axis(0.0, dx, 1),
        z: axis(0.0, dx, 1),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["patch", "patch", "empty", "empty", "empty", "empty"].map(String::from),
        cyclic: Vec::new(),
    };
    blockgen::build_mesh(&bs)
}

/// Runs the tube from `c0(x)` to `t_end`; `sample(t, x, cbar)` is called
/// after every step.
pub fn run(
    gpu: &Gpu,
    spec: &TubeSpec,
    c0: &dyn Fn(f64) -> f64,
    t_end: f64,
    sample: &mut dyn FnMut(f64, &[f64], &[f64]),
) -> Result<TubeRun> {
    run_tube(gpu, spec, c0, t_end, None, sample, &mut |_: &AgeSample<'_>| {})
        .map(|(run, _, _, _)| run)
}

/// Runs the tube with the flame-age clock on (SPEC-LIT §160.5): the clock's
/// origin is the spec's `t_fd_origin` with the floor `age.c_a`, every c~
/// solve is followed by one alpha solve, the closures read the predicted
/// age a* of (160.4) instead of the global clock, and after every step
/// `sample` is called with x ascending, c~ and the local age in the same
/// order. A late kernel of `age` whose time has come seeds
/// `c~ := max(c~, profile(x))` at the top of its step.
pub fn run_aged(
    gpu: &Gpu,
    spec: &TubeSpec,
    age: &AgeOptions,
    c0: &dyn Fn(f64) -> f64,
    t_end: f64,
    sample: &mut dyn FnMut(&AgeSample<'_>),
) -> Result<AgedRun> {
    let (run, budget, alpha, age_f) = run_tube(
        gpu,
        spec,
        c0,
        t_end,
        Some(age),
        &mut |_: f64, _: &[f64], _: &[f64]| {},
        sample,
    )?;
    Ok(AgedRun { run, alpha, age: age_f, budget })
}

/// The closures at this outer iterate's temperature: `dustFsc` at the
/// global clock, or `dustFscAged` at the per-cell age a* with the age on.
#[allow(clippy::too_many_arguments)]
fn launch_closures(
    gpu: &Gpu,
    fk: &FscKernels,
    c: &FscConstants,
    out: FscFields<'_>,
    n_zero: &mut DevBuf<u64>,
    inp: FscInputsDev<'_>,
    t_fd: f64,
    t_new: f64,
    age: Option<&DevBuf<Scalar>>,
    n: usize,
) -> Result<()> {
    match age {
        Some(a) => launch_fsc_aged(gpu, fk, c, out, n_zero, inp, a, n),
        None => launch_fsc(gpu, fk, c, out, n_zero, inp, t_fd, t_new, n),
    }
}

/// The one body `run` and `run_aged` share: the step loop, with the age
/// option's additions when `age` is `Some` (SPEC-LIT §160.5). Returns the
/// plain run and, with the age on, the steps' budget rows and the final
/// alpha and local age ascending in x (all empty with the age off).
#[allow(clippy::too_many_arguments)]
fn run_tube(
    gpu: &Gpu,
    spec: &TubeSpec,
    c0: &dyn Fn(f64) -> f64,
    t_end: f64,
    age_opt: Option<&AgeOptions>,
    plain_sample: &mut dyn FnMut(f64, &[f64], &[f64]),
    age_sample: &mut dyn FnMut(&AgeSample<'_>),
) -> Result<(TubeRun, Vec<AgeBudgetRow>, Vec<f64>, Vec<f64>)> {
    if !(spec.dt.is_finite() && spec.dt > 0.0) || spec.max_outer == 0 {
        return Err(Error::Refused(format!(
            "dust::tube1d: dt = {} must be positive and max_outer = {} at least 1",
            spec.dt, spec.max_outer
        )));
    }
    // The step as the device holds it: the c~ matrix's ddt and the host
    // cumulative sum use the same rounded value.
    let dt = wide(spec.dt as Scalar);
    let mut ctrl = spec.ctrl;
    ctrl.steady = false;
    ctrl.delta_t = dt as Scalar;
    let hm = tube_mesh(spec)?;
    let gm = GpuMesh::upload(gpu, &hm)?;
    let n = hm.n_cells;
    let (nif, nbf) = (hm.n_internal_faces, hm.n_boundary_faces);

    // Cells in ascending x, and each face's place on the line: slot s lies
    // between the cells of rank s - 1 and s, slot 0 at x_lo, slot n at x_hi.
    let xs: Vec<f64> = hm.c.iter().map(|&p| hm.world(p).x).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| xs[a].total_cmp(&xs[b]));
    let mut rank = vec![0usize; n];
    for (r, &i) in order.iter().enumerate() {
        rank[i] = r;
    }
    let area = wide(hm.mag_sf[0]);
    let face_slot: Vec<usize> = (0..nif)
        .map(|f| rank[hm.owner[f] as usize].max(rank[hm.neighbour[f] as usize]))
        .collect();
    // Boundary faces: +1 the fresh end, -1 the burned end, 0 empty.
    let b_end: Vec<i8> = (0..nbf)
        .map(|i| {
            let s = hm.b_sf[i];
            if wide(s.x).abs() < 0.5 * wide(hm.b_mag_sf[i]) {
                0
            } else if wide(s.x) > 0.0 {
                1
            } else {
                -1
            }
        })
        .collect();
    let vol: Vec<f64> = hm.v.iter().map(|&v| wide(v)).collect();

    let fld = FieldKernels::new(gpu)?;
    let fk = FscKernels::new(gpu)?;
    let pk = ProgressKernels::new(gpu)?;
    let mut prog = ProgressEquation::new(gpu, &hm, &gm, ctrl)?;
    // The bounds margin is the step's own convergence (§155.2).
    prog.set_bounds_tolerance(spec.outer_tol)?;
    let mut rho = GpuScalarField::zeros(gpu, &gm, "rho")?;
    let mut rho_face = GpuSurfaceScalarField::zeros(gpu, &gm, "rhof")?;
    let mut phi_m = GpuSurfaceScalarField::zeros(gpu, &gm, "phiM")?;
    let mut u = GpuVectorField::zeros(gpu, &gm, "U")?;
    let mut gamma = GpuScalarField::zeros(gpu, &gm, "kappaDt")?;
    let mut d_t: DevBuf<Scalar> = gpu.zeros(n)?;
    let mut u_t: DevBuf<Scalar> = gpu.zeros(n)?;
    let mut u_t_isp: DevBuf<Scalar> = gpu.zeros(n)?;
    let mut tau_l: DevBuf<Scalar> = gpu.zeros(n)?;
    let mut k_rate: DevBuf<Scalar> = gpu.zeros(n)?;
    let mut n_zero: DevBuf<u64> = gpu.zeros(1)?;
    let mut t_cell: DevBuf<Scalar> = gpu.zeros(n)?;
    let (tk, tw, tn) = match spec.turbulence {
        Some(f) => (f.k, f.omega, f.nu_t),
        None => (0.0, 0.0, 0.0),
    };
    let k_buf = gpu.upload(&vec![tk as Scalar; n])?;
    let w_buf = gpu.upload(&vec![tw as Scalar; n])?;
    let nut_buf = gpu.upload(&vec![tn as Scalar; n])?;
    let zeros = vec![0.0 as Scalar; n];
    let mut w_ign: DevBuf<Scalar> = gpu.zeros(n)?;
    let mut co: DevBuf<Scalar> = gpu.zeros(n)?;
    let mut cbar_d: DevBuf<Scalar> = gpu.zeros(n)?;
    let p_local = spec.ignition.map(|s| s.local_point(&hm));
    if let Some(s) = &spec.ignition {
        s.validate()?;
    }

    let to_s = |v: &[f64]| v.iter().map(|&x| x as Scalar).collect::<Vec<Scalar>>();
    let c_init: Vec<f64> = xs.iter().map(|&x| c0(x)).collect();
    gpu.write(&mut prog.field_mut().f, &to_s(&c_init))?;
    prog.initialise(gpu)?;
    // The age option's clock: the origin is the spec's t_fd_origin (D-20a),
    // the floor the caller's c_a.
    if let Some(ag) = age_opt {
        prog.enable_age(gpu, spec.t_fd_origin, ag.c_a)?;
    }
    let mut applied: Vec<bool> = vec![false; age_opt.map(|a| a.late.len()).unwrap_or(0)];
    let mut budget: Vec<AgeBudgetRow> = Vec::new();
    let mut c_sorted = vec![0.0f64; n];
    let mut age_sorted = vec![0.0f64; n];
    let mut c_now: Vec<f64> = gpu.download(&prog.field().f)?.into_iter().map(wide).collect();
    let bml = |c: &[f64]| -> Vec<f64> {
        c.iter().map(|&ci| bml_density(spec.rho_u, spec.sigma, ci)).collect()
    };
    let mut rho_n = bml(&c_now);
    let mut rho_nm1 = rho_n.clone();
    let mut c_prev = c_now.clone();
    let rho_b = spec.rho_u / spec.sigma;

    let mut t = 0.0f64;
    let mut steps = 0usize;
    let mut out_of_bounds = 0u64;
    let mut max_co_f = 0.0f64;
    let mut max_excursion = 0.0f64;
    let mut max_outer_used = 0usize;
    let mut outer_total = 0usize;
    let mut cbar_h = vec![0.0f64; n];
    let x_sorted: Vec<f64> = order.iter().map(|&j| xs[j]).collect();
    while t < t_end * (1.0 - 1e-12) {
        let t_new = t + dt;
        // A late kernel whose time has come seeds c~ (SPEC-LIT §160.5): the
        // cells it burn start to age from this step. The old levels move
        // with it, so the step's alpha predict reads the seeded c~.
        if let Some(ag) = age_opt {
            for (li, lk) in ag.late.iter().enumerate() {
                if applied[li] || !(t >= lk.t_ig * (1.0 - 1e-12)) {
                    continue;
                }
                applied[li] = true;
                let seeded: Vec<Scalar> = xs
                    .iter()
                    .zip(&c_now)
                    .map(|(&x, &ci)| ci.max((lk.profile)(x)) as Scalar)
                    .collect();
                prog.overwrite_field(gpu, &seeded)?;
                c_now = seeded.iter().map(|&v| wide(v)).collect();
                rho_n = bml(&c_now);
                rho_nm1 = rho_n.clone();
                c_prev = c_now.clone();
            }
        }
        prog.begin_step(gpu)?;
        if age_opt.is_some() {
            prog.begin_age_step(gpu, t_new)?;
        }
        gpu.write(&mut rho.f0, &to_s(&rho_n))?;
        gpu.write(&mut rho.f00, &to_s(&rho_nm1))?;
        if let (Some(src), Some(p)) = (&spec.ignition, p_local) {
            launch_ignition(gpu, &pk, &mut w_ign, &gm.c, src, p, t_new, n)?;
        }
        let t_fd = development_time(t_new, spec.t_fd_origin);
        // The first iterate's density comes from the linear extrapolation
        // 2 c^n - c^{n-1}, so the outer loop starts O(dt^2) from its fixed
        // point; the c~ field itself starts from c^n.
        let mut c_it: Vec<f64> = c_now.iter().zip(&c_prev).map(|(a, b)| 2.0 * a - b).collect();
        let mut used = 0usize;
        let mut step_oob = 0u64;
        for _ in 0..spec.max_outer {
            used += 1;
            let rho_it = bml(&c_it);
            gpu.write(&mut rho.f, &to_s(&rho_it))?;
            // The face density and the cumulative mass flux.
            let rf_i: Vec<f64> = (0..nif)
                .map(|f| {
                    let (p, q) = (hm.owner[f] as usize, hm.neighbour[f] as usize);
                    let w = wide(hm.weights[f]);
                    w * rho_it[p] + (1.0 - w) * rho_it[q]
                })
                .collect();
            let rf_b: Vec<f64> = (0..nbf).map(|i| rho_it[hm.b_face_cells[i] as usize]).collect();
            // m[s]: the +x mass flux per unit area at slot s (m[n] = 0 at the
            // closed fresh end x_hi).
            let mut m = vec![0.0f64; n + 1];
            for s in (0..n).rev() {
                let j = order[s];
                m[s] = m[s + 1] + vol[j] * (rho_it[j] - rho_n[j]) / (dt * area);
            }
            let phi_i: Vec<f64> = (0..nif).map(|f| m[face_slot[f]] * wide(hm.sf[f].x)).collect();
            let phi_b: Vec<f64> = (0..nbf)
                .map(|i| match b_end[i] {
                    -1 => m[0] * wide(hm.b_sf[i].x),
                    _ => 0.0,
                })
                .collect();
            gpu.write(&mut rho_face.f, &to_s(&rf_i))?;
            gpu.write(&mut rho_face.bf, &to_s(&rf_b))?;
            gpu.write(&mut phi_m.f, &to_s(&phi_i))?;
            gpu.write(&mut phi_m.bf, &to_s(&phi_b))?;
            // The cell velocity: the mean of its two face velocities.
            let uc: Vec<Vec3> = (0..n)
                .map(|j| {
                    let s = rank[j];
                    let rl = if s == 0 { rho_it[j] } else { 0.5 * (rho_it[j] + rho_it[order[s - 1]]) };
                    let rr =
                        if s + 1 == n { rho_it[j] } else { 0.5 * (rho_it[j] + rho_it[order[s + 1]]) };
                    let (ul, ur) = (m[s] / rl, m[s + 1] / rr);
                    Vec3::new((0.5 * (ul + ur)) as Scalar, 0.0, 0.0)
                })
                .collect();
            gpu.write(&mut u.f, &uc)?;
            // The closures at this iterate's temperature (153.1).
            let t_h: Vec<f64> =
                c_it.iter().map(|&ci| spec.t_u * (1.0 + (spec.sigma - 1.0) * ci)).collect();
            gpu.write(&mut t_cell, &to_s(&t_h))?;
            gpu.write(&mut n_zero, &[0u64])?;
            launch_closures(
                gpu,
                &fk,
                &spec.fsc,
                FscFields {
                    d_t: &mut d_t,
                    u_t: &mut u_t,
                    u_t_isp: &mut u_t_isp,
                    tau_l: &mut tau_l,
                    k_rate: &mut k_rate,
                },
                &mut n_zero,
                FscInputsDev { t: &t_cell, k: &k_buf, omega: &w_buf, nu_t: &nut_buf },
                t_fd,
                t_new,
                prog.closure_age(),
                n,
            )?;
            if !spec.with_q {
                gpu.write(&mut k_rate, &zeros)?;
            }
            let d_h: Vec<f64> = gpu.download(&d_t)?.into_iter().map(wide).collect();
            let g_h: Vec<f64> = (0..n)
                .map(|j| {
                    let kappa = match spec.kappa {
                        KappaLaw::Zero => 0.0,
                        KappaLaw::ConstantRhoKappa { kappa_u } => kappa_u * spec.rho_u / rho_it[j],
                    };
                    kappa + d_h[j]
                })
                .collect();
            gpu.write(&mut gamma.f, &to_s(&g_h))?;
            field_ops::correct_boundary_conditions(gpu, &fld, &mut gamma, &gm)?;

            let s = prog.solve(
                gpu,
                &ProgressCoupling {
                    rho: &rho,
                    rho_face: &rho_face,
                    phi_mass: &phi_m,
                    u: &u,
                    diffusivity: &gamma,
                    u_t: &u_t,
                    k_rate: &k_rate,
                    w_ign: spec.ignition.as_ref().map(|_| &w_ign),
                    rho_u: spec.rho_u,
                },
            )?;
            step_oob = s.out_of_bounds;
            let c_new: Vec<f64> = gpu.download(&prog.field().f)?.into_iter().map(wide).collect();
            let change = c_new
                .iter()
                .zip(&c_it)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            c_it = c_new;
            if change < spec.outer_tol {
                break;
            }
        }
        max_outer_used = max_outer_used.max(used);
        outer_total += used;
        // The bounds of the step's result, its last outer iterate (§155.2):
        // the earlier iterates are not states of the solution.
        out_of_bounds += step_oob;
        for &v in &c_it {
            max_excursion = max_excursion.max(-v).max(v - 1.0);
        }

        // The measured flame-Courant number of this step (155.9), at the
        // density the last solve used.
        flame_courant(gpu, &pk, &mut co, &u.f, &rho.f, &u_t, &gm.v, spec.rho_u, dt, n)?;
        let co_h = gpu.download(&co)?;
        max_co_f = co_h.iter().fold(max_co_f, |a, &v| a.max(wide(v)));

        // The alpha budget row of the step (SPEC-LIT §160.5's
        // `age_budget_closes`), BEFORE rho.f is overwritten with the step's
        // own density: the Euler form, cells summed in f64 from downloads.
        if let (Some(_), Some(al)) = (age_opt, prog.age_field()) {
            let (sua, spa) = prog.age_source_terms().expect("the age sources");
            let (rho_f, rho_0) = (gpu.download(&rho.f)?, gpu.download(&rho.f0)?);
            let (al_f, al_0) = (gpu.download(&al.f)?, gpu.download(&al.f0)?);
            let (al_bf, phi_b) = (gpu.download(&al.bf)?, gpu.download(&phi_m.bf)?);
            let (sua_h, spa_h) = (gpu.download(sua)?, gpu.download(spa)?);
            let (mut storage, mut clock, mut inherit) = (0.0f64, 0.0f64, 0.0f64);
            for j in 0..n {
                storage += vol[j]
                    * ((wide(rho_f[j]) * wide(al_f[j]) - wide(rho_0[j]) * wide(al_0[j])) / dt);
                clock += vol[j] * wide(sua_h[j]);
                inherit -= vol[j] * wide(spa_h[j]) * wide(al_f[j]);
            }
            let mut outflow = 0.0f64;
            for i in 0..nbf {
                outflow += wide(phi_b[i]) * wide(al_bf[i]);
            }
            budget.push(AgeBudgetRow { t: t_new, storage, outflow, clock, inherit });
        }

        c_prev = std::mem::replace(&mut c_now, c_it);
        rho_nm1 = std::mem::replace(&mut rho_n, bml(&c_now));
        t = t_new;
        steps += 1;

        // The Reynolds mean (155.2) on the device at the step's density.
        gpu.write(&mut rho.f, &to_s(&rho_n))?;
        reynolds_mean(gpu, &pk, &mut cbar_d, &rho.f, &prog.field().f, rho_b, n)?;
        let cb = gpu.download(&cbar_d)?;
        for (s, &j) in order.iter().enumerate() {
            cbar_h[s] = wide(cb[j]);
        }
        match age_opt {
            Some(ag) => {
                // x ascending, c~ and the local age a = alpha/max(c~, c_a)
                // in the same order.
                let ch = gpu.download(&prog.field().f)?;
                let ah = gpu.download(&prog.age_field().expect("the age field").f)?;
                for (s, &j) in order.iter().enumerate() {
                    let cj = wide(ch[j]);
                    c_sorted[s] = cj;
                    age_sorted[s] = age::local_age(wide(ah[j]), cj, ag.c_a);
                }
                age_sample(&AgeSample { t, x: &x_sorted, c: &c_sorted, age: &age_sorted });
            }
            None => plain_sample(t, &x_sorted, &cbar_h),
        }
    }

    // The final alpha and local age, ascending in x.
    let (alpha_f, age_f) = match age_opt {
        Some(ag) => {
            let ch = gpu.download(&prog.field().f)?;
            let ah = gpu.download(&prog.age_field().expect("the age field").f)?;
            let (mut al, mut agv) = (vec![0.0f64; n], vec![0.0f64; n]);
            for (s, &j) in order.iter().enumerate() {
                let (aj, cj) = (wide(ah[j]), wide(ch[j]));
                al[s] = aj;
                agv[s] = age::local_age(aj, cj, ag.c_a);
            }
            (al, agv)
        }
        None => (Vec::new(), Vec::new()),
    };

    Ok((
        TubeRun {
            t,
            steps,
            x: x_sorted,
            c: order.iter().map(|&j| c_now[j]).collect(),
            cbar: cbar_h,
            out_of_bounds,
            max_excursion,
            max_co_f,
            max_outer_used,
            outer_total,
        },
        budget,
        alpha_f,
        age_f,
    ))
}
