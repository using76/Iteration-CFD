// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate 132-Z: the NASA TMR zero-pressure-gradient plate with SA and SST,
//! against Karman-Schoenherr and Coles (SPEC-LIT §144.5, docs/17 VAL-34).**
//!
//! The TMR's 2DZP validation case runs on VAL-31's flat-plate harness (§144.2:
//! the crate's SIMPLE on §144.1's plate) at the TMR's `Re = 5e6` per unit
//! length, the plate from `x = 0` to `2`, with Spalart-Allmaras (§56) and
//! k-omega SST. Two quantities are held against the TMR's two theory files:
//! the wall `C_f(Re_theta)` against the Karman-Schoenherr correlation over
//! `4000 <= Re_theta <= 13000` (3 %), and `u+(y+)` at `Re_theta = 1e4`
//! against Coles' profile over `30 <= y+ <= 300`, to 1.0 in `u+`. Both references are
//! correlations, not measurements, so a result outside its bar is reported
//! OPEN, never MISSES. The TMR's drag coefficient is not used (§56.11).
//!
//! The runs are long, so they are not made by `ofgpu-validate`: each is an
//! ignored test of this binary (`record_zpg_*`) that prints a [`ZpgRecord`]
//! literal, and the records below are what the validate section replays
//! (§110.1's split of recipe and verdict). Every reduction is f64 whatever
//! `Scalar` is: fields are widened on download and the geometry is
//! `geom64()` (§118.2).
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §144.5 (this gate), §144.1-§144.3 (the harness),
//!     §56 (SA) and §6.3 (SST), §94 (the mesh study), §69 (the gate registry),
//!     §110.1 (the replay shape) and §32.4 (the band rule)
//!   the NASA/TMBWG Turbulence Modeling Resource's 2DZP validation page
//!     (tmbwg.github.io/turbmodels/flatplate_val.html, read 2026-10-15) and
//!     its SA and SST result pages, for the case, the definitions and the
//!     free-stream turbulence; the two theory files are read from
//!     `reference/tmr-zpg/` at run time and no number of them is in this file
//! No GPL-licensed source was consulted.

use ofgpu::vv::{self, GridStudy, Level};
use ofgpu::{Result, Scalar};

use super::t3a::band_110;
use crate::{Band, Checks, GateReport, How, Uncertainty, Verdict};

// ---------------------------------------------------------------------------
//  The rig
// ---------------------------------------------------------------------------

/// The TMR's Reynolds number per unit length; the harness runs it at
/// `U = 1 m/s`, so `nu = 2e-7 m^2/s`.
pub(crate) const RE_PER_M: f64 = 5.0e6;
pub(crate) const U_INF: f64 = 1.0;
pub(crate) const NU: f64 = U_INF / RE_PER_M;

/// The TMR's free-stream turbulence, (144.5a): SA's `nu~/nu = 3`; SST's
/// `Tu = 0.039 %` and `mu_t/mu = 0.009`.
pub(crate) const SA_NU_TILDA_RATIO: f64 = 3.0;
pub(crate) const SST_TU_PCT: f64 = 0.039;
pub(crate) const SST_NUT_RATIO: f64 = 0.009;

/// The plan's bars: `C_f` within 3 % of K-S, `u+` within 1.0 of Coles.
pub(crate) const CF_BAND: f64 = 0.03;
pub(crate) const UP_BAND: f64 = 1.0;

/// The `Re_theta` stations the record carries: 4000 to 13000 in steps of
/// 500 (19), the plan's range end to end.
pub(crate) const N_RT: usize = 19;
pub(crate) const RT_LO: f64 = 4000.0;
pub(crate) const RT_STEP: f64 = 500.0;
/// The profile station, and the `y+` points it is read at (30 to 300).
pub(crate) const RT_PROFILE: f64 = 1.0e4;
pub(crate) const Y_PLUS: [f64; 11] = [30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 150.0, 200.0, 250.0, 300.0];
/// The study quantities: `C_f` at `Re_theta = 1e4` and `u+` at `y+ = 100`.
const K_CF_STUDY: usize = 12;
const K_UP_STUDY: usize = 5;

fn w(x: Scalar) -> f64 {
    f64::from(x)
}

/// `Re_theta` of station `k`.
pub(crate) fn re_theta_at(k: usize) -> f64 {
    RT_LO + RT_STEP * k as f64
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ZpgModel {
    Sa,
    Sst,
}

impl ZpgModel {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Sa => "SpalartAllmaras",
            Self::Sst => "kOmegaSST",
        }
    }
}

/// (144.5a): SST's inflow `(k, omega)` from the TMR's `Tu` and `mu_t/mu`.
pub(crate) fn sst_inflow(u: f64, nu: f64) -> (f64, f64) {
    let k = 1.5 * (SST_TU_PCT / 100.0 * u).powi(2);
    (k, k / (SST_NUT_RATIO * nu))
}

/// One recorded run: one model on one level.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ZpgRecord {
    pub(crate) model: ZpgModel,
    pub(crate) cells: (usize, usize, usize),
    pub(crate) n_cells: usize,
    /// (94.1) in 2-D: `sqrt(A/N)`, `A` the x-y area.
    pub(crate) h: f64,
    pub(crate) iterations: usize,
    pub(crate) converged: bool,
    pub(crate) last_change: f64,
    /// `C_f` at `Re_theta = 4000 + 500 k` (144.5c); NaN where no column pair
    /// brackets it.
    pub(crate) cf: [f64; N_RT],
    /// `Re_theta` of the last plate column.
    pub(crate) re_theta_end: f64,
    /// The profile station (144.5d): its `x`, its `C_f`, and `u+` at
    /// [`Y_PLUS`].
    pub(crate) x_profile: f64,
    pub(crate) cf_profile: f64,
    pub(crate) up: [f64; 11],
    /// The largest wall-cell `y+` over the columns with `Re_x >= 1e5`.
    pub(crate) y_plus_wall: f64,
    pub(crate) seconds: f64,
    /// `"f64"` or `"single"`, and the commit the run was made at.
    pub(crate) build: &'static str,
    pub(crate) commit: &'static str,
}

/// The recorder: one run of one model on one level, on VAL-31's harness.
/// Test code, because the runs are made by this binary's ignored tests
/// (`record_zpg_*`), never by `ofgpu-validate` itself (SPEC-LIT 110.1).
#[cfg(test)]
pub(crate) mod runner {
    use std::time::Instant;

    use ofgpu::blockgen::{build_flat_plate_mesh, FlatPlateSpec};
    use ofgpu::field::{BcKind, GpuScalarField};
    use ofgpu::field_setup::{setup_scalar_field, NutRoughness, WallFaces};
    use ofgpu::io::case::{TurbulenceControls, WallFunctionCoeffs};
    use ofgpu::mesh::HostMesh;
    use ofgpu::models::{KOmegaSst, KOmegaSstCoeffs, SaCoeffs, SpalartAllmaras};
    use ofgpu::momentum::BuoyancyCoeffs;
    use ofgpu::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
    use ofgpu::simple::{nan_propagating_max, Simple};
    use ofgpu::solver::SolverPerformance;
    use ofgpu::turbulence::FlowState;
    use ofgpu::{Error, Gpu, GpuMesh, Label, Result, Scalar, Vec3};

    use super::super::blasius_plate::{controls, seed_free_stream, set_bcs, wall_faces, WallFace};
    use super::super::t3a::runner::{patch_index, raw, transported};
    use super::{
        re_theta_at, sst_inflow, w, ZpgModel, ZpgRecord, NU, N_RT, RT_PROFILE, SA_NU_TILDA_RATIO, U_INF, Y_PLUS,
    };

    /// The three meshes, finest first: `(n_upstream, n_plate, n_y)`, the
    /// cell counts of the TMR's 273x193, 137x97 and 69x49 grids.
    pub(crate) const LEVELS: [(usize, usize, usize); 3] = [(48, 224, 192), (24, 112, 96), (12, 56, 48)];

    /// The band the stopping rule reads: the plate from `Re_x = 1e5`.
    const RE_X_LO: f64 = 1e5;
    const CHECK_EVERY: usize = 100;
    const MIN_ITERS: usize = 1000;
    const STOP_TOL: f64 = 1e-6;

    /// The plate of one level: the TMR's box (`x` from -1/3 to 2, `y` to 1),
    /// end-cell ratios 10 / 10 / 54200 (the wall cell `2.0e-6` at `n_y = 96`,
    /// the TMR 137x97 grid's).
    pub(crate) fn zpg_spec(n: (usize, usize, usize)) -> FlatPlateSpec {
        let mut spec = FlatPlateSpec::laminar();
        spec.x_upstream = 1.0 / 3.0;
        spec.length = 2.0;
        spec.height = 1.0;
        spec.n_upstream = n.0;
        spec.n_plate = n.1;
        spec.n_y = n.2;
        spec.r_upstream = 10.0;
        spec.r_plate = 10.0;
        spec.r_y = 54200.0;
        spec.u_inf = U_INF;
        spec.nu = NU;
        spec
    }

    /// One plate column: the wall face's `x` and `C_f`, the cells above it
    /// as `(y, u_x)` from the wall up, and its `Re_theta` (144.5b).
    #[derive(Clone, Debug)]
    pub(crate) struct Column {
        pub(crate) x: f64,
        pub(crate) cf: f64,
        pub(crate) rows: Vec<(f64, f64)>,
        pub(crate) re_theta: f64,
    }

    /// (144.5b): `theta` of one column, `(y, u)` from the wall up. `U_e` is
    /// the column's largest `u`, `delta_995` the least `y` at which
    /// `u >= 0.995 U_e` (linear between points), and `theta` integrates
    /// `r (1 - r)`, `r = u/U_e` piecewise linear on the wall point `(0, 0)`
    /// and the cell centres, exactly, to `y = 2 delta_995`. Returns
    /// `(theta, U_e)`; NaN when the column never reaches `0.995 U_e` or ends
    /// below `2 delta_995`.
    pub(crate) fn theta_of(rows: &[(f64, f64)]) -> (f64, f64) {
        let ue = rows.iter().map(|r| r.1).fold(f64::NEG_INFINITY, f64::max);
        if !(ue > 0.0) {
            return (f64::NAN, ue);
        }
        let pts: Vec<(f64, f64)> = std::iter::once((0.0, 0.0)).chain(rows.iter().map(|&(y, u)| (y, u / ue))).collect();
        let mut delta = f64::NAN;
        for p in pts.windows(2) {
            if p[1].1 >= 0.995 {
                let t = (0.995 - p[0].1) / (p[1].1 - p[0].1);
                delta = p[0].0 + t * (p[1].0 - p[0].0);
                break;
            }
        }
        let cut = 2.0 * delta;
        if !cut.is_finite() || cut > pts[pts.len() - 1].0 {
            return (f64::NAN, ue);
        }
        // The exact integral of r(1 - r) over a segment where r is linear.
        let seg = |h: f64, a: f64, b: f64| h * (0.5 * (a + b) - (a * a + a * b + b * b) / 3.0);
        let mut theta = 0.0;
        for p in pts.windows(2) {
            let ((y0, r0), (y1, r1)) = (p[0], p[1]);
            if y1 <= cut {
                theta += seg(y1 - y0, r0, r1);
            } else {
                let rc = r0 + (cut - y0) / (y1 - y0) * (r1 - r0);
                theta += seg(cut - y0, r0, rc);
                break;
            }
        }
        (theta, ue)
    }

    /// The plate's columns, edge to trailing edge: each cell behind the
    /// edge joins the wall face whose `x` is nearest its centre's.
    pub(crate) fn columns(hm: &HostMesh, faces: &[WallFace], u: &[Vec3]) -> Result<Vec<Column>> {
        let mut faces: Vec<WallFace> = faces.to_vec();
        faces.sort_by(|a, b| a.x.total_cmp(&b.x));
        let xs: Vec<f64> = faces.iter().map(|f| f.x).collect();
        let mut rows: Vec<Vec<(f64, f64)>> = vec![Vec::new(); xs.len()];
        let g = hm.geom64();
        for c in 0..hm.n_cells {
            let x = g.c[c].x;
            if x <= 0.0 || xs.is_empty() {
                continue;
            }
            let i = xs.partition_point(|&a| a < x);
            let j = if i == 0 {
                0
            } else if i == xs.len() || (x - xs[i - 1]) <= (xs[i] - x) {
                i - 1
            } else {
                i
            };
            rows[j].push((g.c[c].y, w(u[c].x)));
        }
        let n_y = rows.first().map_or(0, Vec::len);
        let mut out = Vec::with_capacity(xs.len());
        for (f, mut r) in faces.into_iter().zip(rows) {
            if r.len() != n_y || n_y == 0 {
                return Err(Error::Config(format!(
                    "Gate 132-Z (SPEC-LIT 144.5): the column at x = {} holds {} cells, the first {n_y}",
                    f.x,
                    r.len()
                )));
            }
            r.sort_by(|a, b| a.0.total_cmp(&b.0));
            let (theta, ue) = theta_of(&r);
            out.push(Column { x: f.x, cf: f.cf, rows: r, re_theta: ue * theta / NU });
        }
        Ok(out)
    }

    /// The first pair of columns from the edge whose `Re_theta` brackets
    /// `rt`, and the fraction `t` between them.
    fn bracket(cols: &[Column], rt: f64) -> Option<(usize, f64)> {
        cols.windows(2).position(|p| p[0].re_theta <= rt && rt <= p[1].re_theta).map(|i| {
            let (a, b) = (cols[i].re_theta, cols[i + 1].re_theta);
            (i, if b > a { (rt - a) / (b - a) } else { 0.0 })
        })
    }

    /// (144.5c): `C_f` at `Re_theta = rt`, linear in `Re_theta`.
    pub(crate) fn cf_at_re_theta(cols: &[Column], rt: f64) -> f64 {
        bracket(cols, rt).map_or(f64::NAN, |(i, t)| cols[i].cf + t * (cols[i + 1].cf - cols[i].cf))
    }

    /// (144.5d): the profile at `Re_theta = rt`: `(x, C_f, u+ at Y_PLUS)`.
    pub(crate) fn profile_at(cols: &[Column], rt: f64) -> (f64, f64, [f64; 11]) {
        let mut up = [f64::NAN; 11];
        let Some((i, t)) = bracket(cols, rt) else {
            return (f64::NAN, f64::NAN, up);
        };
        let (a, b) = (&cols[i], &cols[i + 1]);
        let x = a.x + t * (b.x - a.x);
        let cf = a.cf + t * (b.cf - a.cf);
        let vstar = U_INF * (0.5 * cf).sqrt();
        let plus: Vec<(f64, f64)> = a
            .rows
            .iter()
            .zip(&b.rows)
            .map(|(&(y, ua), &(_, ub))| (y * vstar / NU, (ua + t * (ub - ua)) / vstar))
            .collect();
        for (k, &yp) in Y_PLUS.iter().enumerate() {
            if let Some(p) = plus.windows(2).find(|p| p[0].0 <= yp && yp <= p[1].0) {
                let s = (yp / p[0].0).ln() / (p[1].0 / p[0].0).ln();
                up[k] = p[0].1 + s * (p[1].1 - p[0].1);
            }
        }
        (x, cf, up)
    }

    fn cf_change(a: &[WallFace], b: &[WallFace]) -> f64 {
        a.iter()
            .zip(b)
            .filter(|(f, _)| f.re_x >= RE_X_LO)
            .map(|(f, g)| ((g.cf - f.cf) / f.cf).abs())
            .fold(0.0f64, |m, d| if m.is_nan() || d.is_nan() { f64::NAN } else { m.max(d) })
    }

    /// The two models behind one interface: what the loop needs of each.
    enum Turb<'m> {
        Sa(SpalartAllmaras<'m>),
        Sst(KOmegaSst<'m>),
    }

    impl Turb<'_> {
        fn nut(&self) -> &GpuScalarField {
            match self {
                Self::Sa(m) => m.nut(),
                Self::Sst(m) => m.nut(),
            }
        }
        fn correct(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<(SolverPerformance, SolverPerformance)> {
            match self {
                Self::Sa(m) => m.correct(gpu, flow),
                Self::Sst(m) => m.correct(gpu, flow),
            }
        }
    }

    /// The record of one state of the solve.
    fn measure(hm: &HostMesh, plate: usize, u: &[Vec3]) -> Result<(Vec<Column>, [f64; N_RT], f64, f64, [f64; 11], f64)> {
        let faces = wall_faces(hm, plate, u, U_INF, NU);
        let cols = columns(hm, &faces, u)?;
        let mut cf = [f64::NAN; N_RT];
        for (k, v) in cf.iter_mut().enumerate() {
            *v = cf_at_re_theta(&cols, re_theta_at(k));
        }
        let (x, cfp, up) = profile_at(&cols, RT_PROFILE);
        let yw = cols
            .iter()
            .filter(|c| U_INF * c.x / NU >= RE_X_LO)
            .map(|c| c.rows[0].0 * U_INF * (0.5 * c.cf).sqrt() / NU)
            .fold(0.0f64, f64::max);
        Ok((cols, cf, x, cfp, up, yw))
    }

    /// Run one model on one level, steady, until the stopping rule or
    /// `max_iters`. `progress` prints a line per check.
    pub(crate) fn run_zpg(
        gpu: &Gpu,
        model: ZpgModel,
        n: (usize, usize, usize),
        max_iters: usize,
        progress: bool,
    ) -> Result<ZpgRecord> {
        let started = Instant::now();
        let spec = zpg_spec(n);
        let hm = build_flat_plate_mesh(&spec)?;
        let plate = patch_index(&hm, "plate")?;
        let gm = GpuMesh::upload(gpu, &hm)?;
        let ctrl = controls(NU);
        let mut sim = Simple::new(gpu, &hm, &gm, ctrl, BuoyancyCoeffs { g: Vec3::ZERO, ..BuoyancyCoeffs::default() })?;
        set_bcs(gpu, &mut sim, &hm, U_INF)?;
        seed_free_stream(gpu, &mut sim, &hm, U_INF)?;
        sim.initialise(gpu)?;

        let raw_nut = raw(
            "nut",
            0.0,
            &[
                ("inlet", "calculated", Some(0.0)),
                ("outlet", "calculated", Some(0.0)),
                ("top", "calculated", Some(0.0)),
                ("symmetry", "symmetryPlane", None),
                ("plate", "nutLowReWallFunction", Some(0.0)),
                ("back", "empty", None),
                ("front", "empty", None),
            ],
        );
        let rough = NutRoughness::none(hm.n_boundary_faces);
        let wd = ofgpu::walldistance::wall_distance(gpu, &hm, &gm, &ctrl.p_solver, 0)?;
        // The turbulence fields keep the default bounded upwind (144.5).
        let tctrl = TurbulenceControls::default();
        let mut m = match model {
            ZpgModel::Sa => {
                let raw_nt = transported("nuTilda", SA_NU_TILDA_RATIO * NU, ("fixedValue", Some(0.0)));
                let wall_set = WallFaces::from_case(&raw_nt, Some(&raw_nut), &hm)?;
                let mut sa = SpalartAllmaras::new(
                    gpu,
                    &hm,
                    &gm,
                    SaCoeffs::default(),
                    tctrl,
                    WallFunctionCoeffs::default(),
                    &wall_set,
                    &rough,
                    &wd.y.f,
                )?;
                setup_scalar_field(gpu, sa.nu_tilda_mut(), &raw_nt, &hm)?;
                setup_scalar_field(gpu, sa.nut_mut(), &raw_nut, &hm)?;
                sa.initialise(gpu, &sim.flow_state())?;
                Turb::Sa(sa)
            }
            ZpgModel::Sst => {
                let (k0, omega0) = sst_inflow(U_INF, NU);
                let raw_k = transported("k", k0, ("fixedValue", Some(0.0)));
                let raw_omega = transported("omega", omega0, ("omegaWallFunction", Some(omega0)));
                let wall_set = WallFaces::from_case(&raw_omega, Some(&raw_nut), &hm)?;
                let mut sst = KOmegaSst::new(
                    gpu,
                    &hm,
                    &gm,
                    KOmegaSstCoeffs::default(),
                    tctrl,
                    WallFunctionCoeffs::default(),
                    &wall_set,
                    &wd.y.f,
                    &rough,
                )?;
                setup_scalar_field(gpu, sst.k_mut(), &raw_k, &hm)?;
                setup_scalar_field(gpu, sst.omega_mut(), &raw_omega, &hm)?;
                setup_scalar_field(gpu, sst.nut_mut(), &raw_nut, &hm)?;
                sst.initialise(gpu, &sim.flow_state())?;
                Turb::Sst(sst)
            }
        };

        let mut t = GpuScalarField::zeros(gpu, &gm, "T")?;
        gpu.write(&mut t.bc_kind, &vec![BcKind::ZeroGradient as Label; hm.n_boundary_faces])?;
        let mut backend = PbicgstabBackend::new(ctrl.p_solver);
        backend.setup(gpu, &hm, &gm, &SystemProbe::default())?;

        let mut last = wall_faces(&hm, plate, &gpu.download(&sim.u().f)?, U_INF, NU);
        let (mut iterations, mut converged, mut last_change) = (max_iters, false, f64::INFINITY);
        for it in 1..=max_iters {
            let perf = sim.solve_step(gpu, &mut backend, m.nut(), &t)?;
            let (p1, p2) = m.correct(gpu, &sim.flow_state())?;
            let res = [
                perf.last.u[0].initial_residual,
                perf.last.u[1].initial_residual,
                perf.last.p.initial_residual,
                p1.initial_residual,
                p2.initial_residual,
            ]
            .into_iter()
            .fold(0.0 as Scalar, nan_propagating_max);
            if !w(res).is_finite() {
                return Err(Error::Diverged {
                    iteration: it,
                    what: format!("ZPG {} {n:?}: initial residual {}", model.name(), w(res)),
                });
            }
            if it % CHECK_EVERY == 0 {
                let u = gpu.download(&sim.u().f)?;
                let now = wall_faces(&hm, plate, &u, U_INF, NU);
                last_change = cf_change(&last, &now);
                last = now;
                if progress {
                    let (cols, cf, _, _, _, yw) = measure(&hm, plate, &u)?;
                    println!(
                        "ZPG {} {n:?} iter {it}/{max_iters} ({:.1} %): max res {:.3e}, C_f change {:.3e}, \
                         Re_theta at the end {:.1}, C_f(Re_theta 1e4) {:.5e}, wall y+ {:.3}, {:.1} s",
                        model.name(),
                        100.0 * it as f64 / max_iters as f64,
                        w(res),
                        last_change,
                        cols.last().map_or(f64::NAN, |c| c.re_theta),
                        cf[12],
                        yw,
                        started.elapsed().as_secs_f64()
                    );
                }
                if it >= MIN_ITERS && last_change <= STOP_TOL {
                    iterations = it;
                    converged = true;
                    break;
                }
            }
        }

        let u = gpu.download(&sim.u().f)?;
        let (cols, cf, x_profile, cf_profile, up, y_plus_wall) = measure(&hm, plate, &u)?;
        let g = hm.geom64();
        let area: f64 = g.v.iter().sum::<f64>() / spec.depth;
        Ok(ZpgRecord {
            model,
            cells: n,
            n_cells: hm.n_cells,
            h: (area / hm.n_cells as f64).sqrt(),
            iterations,
            converged,
            last_change,
            cf,
            re_theta_end: cols.last().map_or(f64::NAN, |c| c.re_theta),
            x_profile,
            cf_profile,
            up,
            y_plus_wall,
            seconds: started.elapsed().as_secs_f64(),
            build: if cfg!(feature = "single") { "single" } else { "f64" },
            commit: "",
        })
    }

    /// The record as the Rust literal the slots below hold.
    pub(crate) fn record_literal(r: &ZpgRecord) -> String {
        let list = |v: &[f64]| v.iter().map(|x| format!("{x:.6e}")).collect::<Vec<_>>().join(", ");
        format!(
            "ZpgRecord {{ model: ZpgModel::{:?}, cells: {:?}, n_cells: {}, h: {:.6e}, iterations: {}, \
             converged: {}, last_change: {:.3e}, cf: [{}], re_theta_end: {:.1}, x_profile: {:.6e}, \
             cf_profile: {:.6e}, up: [{}], y_plus_wall: {:.4}, seconds: {:.1}, build: \"{}\", commit: \"\" }}",
            r.model,
            r.cells,
            r.n_cells,
            r.h,
            r.iterations,
            r.converged,
            r.last_change,
            list(&r.cf),
            r.re_theta_end,
            r.x_profile,
            r.cf_profile,
            list(&r.up),
            r.y_plus_wall,
            r.seconds,
            r.build
        )
    }
}

// ---------------------------------------------------------------------------
//  The records (SPEC-LIT 144.5's table)
// ---------------------------------------------------------------------------

/// The replay slots: `[model][level]`, SA first, finest level first. `None`
/// is a run not recorded. `RECORDS` is the f64 build's, the gate's;
/// `RECORDS_SINGLE` is the same six runs under `--features single`.
pub(crate) const RECORDS: [[Option<&ZpgRecord>; 3]; 2] = [
    [Some(&F64_SA_FINE), Some(&F64_SA_MEDIUM), Some(&F64_SA_COARSE)],
    [Some(&F64_SST_FINE), Some(&F64_SST_MEDIUM), Some(&F64_SST_COARSE)],
];
pub(crate) const RECORDS_SINGLE: [[Option<&ZpgRecord>; 3]; 2] = [
    [Some(&SINGLE_SA_FINE), Some(&SINGLE_SA_MEDIUM), Some(&SINGLE_SA_COARSE)],
    [Some(&SINGLE_SST_FINE), Some(&SINGLE_SST_MEDIUM), Some(&SINGLE_SST_COARSE)],
];

// The f64 records, as `record_zpg_*` printed them.
const F64_SST_FINE: ZpgRecord = ZpgRecord {
    model: ZpgModel::Sst, cells: (48, 224, 192), n_cells: 52224, h: 6.684260e-3, iterations: 10000, converged: false, last_change: 4.780e-6,
    cf: [3.008464e-3, 2.944330e-3, 2.888731e-3, 2.839866e-3, 2.796196e-3, 2.757108e-3, 2.721444e-3, 2.689081e-3, 2.659320e-3, 2.631801e-3, 2.606260e-3, 2.582412e-3, 2.560106e-3, 2.539145e-3, 2.519381e-3, 2.500692e-3, 2.482998e-3, 2.466312e-3, 2.450458e-3],
    re_theta_end: 14191.3, x_profile: 1.323780e0, cf_profile: 2.560106e-3,
    up: [1.279524e1, 1.376291e1, 1.447653e1, 1.503971e1, 1.589861e1, 1.654387e1, 1.706002e1, 1.768021e1, 1.846487e1, 1.906488e1, 1.955158e1],
    y_plus_wall: 0.1273, seconds: 667.3, build: "f64", commit: ""
};
const F64_SA_FINE: ZpgRecord = ZpgRecord {
    model: ZpgModel::Sa, cells: (48, 224, 192), n_cells: 52224, h: 6.684260e-3, iterations: 10000, converged: false, last_change: 1.403e-5,
    cf: [3.003888e-3, 2.980101e-3, 2.944445e-3, 2.917263e-3, 2.820865e-3, 2.826003e-3, 2.783552e-3, 2.736747e-3, 2.688343e-3, 2.714265e-3, 2.718861e-3, 2.655988e-3, 2.571550e-3, 2.541322e-3, 2.552830e-3, 2.540064e-3, 2.556930e-3, 2.539390e-3, 2.503805e-3],
    re_theta_end: 14390.4, x_profile: 1.302617e0, cf_profile: 2.571550e-3,
    up: [1.349571e1, 1.424406e1, 1.479879e1, 1.524482e1, 1.594480e1, 1.648916e1, 1.693663e1, 1.748928e1, 1.821328e1, 1.878756e1, 1.926803e1],
    y_plus_wall: 0.1243, seconds: 1422.2, build: "f64", commit: ""
};
const F64_SST_COARSE: ZpgRecord = ZpgRecord {
    model: ZpgModel::Sst, cells: (12, 56, 48), n_cells: 3264, h: 2.673704e-2, iterations: 1200, converged: true, last_change: 3.214e-7,
    cf: [2.902964e-3, 2.843724e-3, 2.790832e-3, 2.744970e-3, 2.703911e-3, 2.665964e-3, 2.633129e-3, 2.602343e-3, 2.573232e-3, 2.547596e-3, 2.523667e-3, 2.500651e-3, 2.478712e-3, 2.459228e-3, 2.440757e-3, 2.422937e-3, 2.405540e-3, 2.388865e-3, 2.374336e-3],
    re_theta_end: 13993.5, x_profile: 1.350755e0, cf_profile: 2.478712e-3,
    up: [1.314432e1, 1.411015e1, 1.481847e1, 1.537722e1, 1.622573e1, 1.686485e1, 1.737782e1, 1.799575e1, 1.878226e1, 1.938915e1, 1.988481e1],
    y_plus_wall: 0.4572, seconds: 56.1, build: "f64", commit: ""
};
const F64_SST_MEDIUM: ZpgRecord = ZpgRecord {
    model: ZpgModel::Sst, cells: (24, 112, 96), n_cells: 13056, h: 1.336852e-2, iterations: 2800, converged: true, last_change: 5.163e-7,
    cf: [2.975510e-3, 2.913486e-3, 2.859543e-3, 2.811906e-3, 2.769533e-3, 2.731648e-3, 2.696888e-3, 2.664838e-3, 2.635826e-3, 2.608399e-3, 2.583500e-3, 2.559702e-3, 2.537978e-3, 2.516944e-3, 2.497750e-3, 2.479087e-3, 2.461690e-3, 2.445211e-3, 2.428984e-3],
    re_theta_end: 14115.1, x_profile: 1.332867e0, cf_profile: 2.537978e-3,
    up: [1.285649e1, 1.380958e1, 1.450959e1, 1.506236e1, 1.590543e1, 1.654050e1, 1.705019e1, 1.766470e1, 1.844696e1, 1.904982e1, 1.954203e1],
    y_plus_wall: 0.2457, seconds: 261.0, build: "f64", commit: ""
};
const F64_SA_MEDIUM: ZpgRecord = ZpgRecord {
    model: ZpgModel::Sa, cells: (24, 112, 96), n_cells: 13056, h: 1.336852e-2, iterations: 3600, converged: true, last_change: 6.768e-7,
    cf: [3.049418e-3, 2.990087e-3, 2.962883e-3, 2.891785e-3, 2.845641e-3, 2.812937e-3, 2.766008e-3, 2.738507e-3, 2.714317e-3, 2.682889e-3, 2.661907e-3, 2.640533e-3, 2.620851e-3, 2.593972e-3, 2.568437e-3, 2.551262e-3, 2.533997e-3, 2.522663e-3, 2.511418e-3],
    re_theta_end: 14432.5, x_profile: 1.300066e0, cf_profile: 2.620851e-3,
    up: [1.333359e1, 1.406952e1, 1.461702e1, 1.505866e1, 1.575315e1, 1.629400e1, 1.673887e1, 1.728893e1, 1.800997e1, 1.858161e1, 1.905946e1],
    y_plus_wall: 0.2410, seconds: 364.8, build: "f64", commit: ""
};
const F64_SA_COARSE: ZpgRecord = ZpgRecord {
    model: ZpgModel::Sa, cells: (12, 56, 48), n_cells: 3264, h: 2.673704e-2, iterations: 1200, converged: true, last_change: 1.078e-7,
    cf: [3.086438e-3, 3.009912e-3, 2.972078e-3, 2.917544e-3, 2.848522e-3, 2.803405e-3, 2.776033e-3, 2.752425e-3, 2.724833e-3, 2.693820e-3, 2.669817e-3, 2.639634e-3, 2.616594e-3, 2.587203e-3, 2.567321e-3, 2.550158e-3, 2.538874e-3, 2.527831e-3, 2.513482e-3],
    re_theta_end: 14609.8, x_profile: 1.280222e0, cf_profile: 2.616594e-3,
    up: [1.329646e1, 1.403579e1, 1.458616e1, 1.502932e1, 1.572510e1, 1.626678e1, 1.671196e1, 1.726319e1, 1.798713e1, 1.856119e1, 1.904131e1],
    y_plus_wall: 0.4588, seconds: 97.8, build: "f64", commit: ""
};

// The single-build records, as `record_zpg_*` printed them.
const SINGLE_SST_COARSE: ZpgRecord = ZpgRecord {
    model: ZpgModel::Sst, cells: (12, 56, 48), n_cells: 3264, h: 2.673704e-2, iterations: 3700, converged: true, last_change: 8.150e-7,
    cf: [2.902810e-3, 2.843599e-3, 2.790715e-3, 2.744866e-3, 2.703819e-3, 2.665880e-3, 2.633051e-3, 2.602273e-3, 2.573171e-3, 2.547545e-3, 2.523614e-3, 2.500600e-3, 2.478670e-3, 2.459185e-3, 2.440705e-3, 2.422876e-3, 2.405472e-3, 2.388798e-3, 2.374256e-3],
    re_theta_end: 13992.7, x_profile: 1.350857e0, cf_profile: 2.478670e-3,
    up: [1.314343e1, 1.410901e1, 1.481715e1, 1.537577e1, 1.622411e1, 1.686316e1, 1.737610e1, 1.799407e1, 1.878074e1, 1.938784e1, 1.988371e1],
    y_plus_wall: 0.4571, seconds: 33.9, build: "single", commit: ""
};
const SINGLE_SST_MEDIUM: ZpgRecord = ZpgRecord {
    model: ZpgModel::Sst, cells: (24, 112, 96), n_cells: 13056, h: 1.336852e-2, iterations: 14100, converged: true, last_change: 9.750e-7,
    cf: [2.975758e-3, 2.913822e-3, 2.859969e-3, 2.812411e-3, 2.770112e-3, 2.732298e-3, 2.697620e-3, 2.665659e-3, 2.636715e-3, 2.609350e-3, 2.584499e-3, 2.560742e-3, 2.539055e-3, 2.518057e-3, 2.498880e-3, 2.480273e-3, 2.462862e-3, 2.446414e-3, 2.430160e-3],
    re_theta_end: 14119.8, x_profile: 1.332610e0, cf_profile: 2.539055e-3,
    up: [1.284103e1, 1.379009e1, 1.448706e1, 1.503747e1, 1.587721e1, 1.651017e1, 1.701851e1, 1.763195e1, 1.841396e1, 1.901756e1, 1.951102e1],
    y_plus_wall: 0.2457, seconds: 114.9, build: "single", commit: ""
};
const SINGLE_SST_FINE: ZpgRecord = ZpgRecord {
    model: ZpgModel::Sst, cells: (48, 224, 192), n_cells: 52224, h: 6.684260e-3, iterations: 10000, converged: false, last_change: 4.915e-6,
    cf: [3.014997e-3, 2.951646e-3, 2.896560e-3, 2.848053e-3, 2.804681e-3, 2.765880e-3, 2.730449e-3, 2.698203e-3, 2.668509e-3, 2.641005e-3, 2.615429e-3, 2.591561e-3, 2.569160e-3, 2.548077e-3, 2.528188e-3, 2.509395e-3, 2.491664e-3, 2.474882e-3, 2.458745e-3],
    re_theta_end: 14222.3, x_profile: 1.320609e0, cf_profile: 2.569160e-3,
    up: [1.268959e1, 1.363051e1, 1.432391e1, 1.487136e1, 1.570787e1, 1.633857e1, 1.684517e1, 1.745709e1, 1.823754e1, 1.883996e1, 1.933259e1],
    y_plus_wall: 0.1272, seconds: 127.1, build: "single", commit: ""
};
const SINGLE_SA_COARSE: ZpgRecord = ZpgRecord {
    model: ZpgModel::Sa, cells: (12, 56, 48), n_cells: 3264, h: 2.673704e-2, iterations: 1300, converged: true, last_change: 9.682e-7,
    cf: [3.069863e-3, 3.013737e-3, 2.960575e-3, 2.913922e-3, 2.865500e-3, 2.823673e-3, 2.787857e-3, 2.744178e-3, 2.709674e-3, 2.699708e-3, 2.682180e-3, 2.647106e-3, 2.615749e-3, 2.601487e-3, 2.586351e-3, 2.567646e-3, 2.552427e-3, 2.531809e-3, 2.507650e-3],
    re_theta_end: 14626.3, x_profile: 1.277746e0, cf_profile: 2.615749e-3,
    up: [1.329020e1, 1.402673e1, 1.457582e1, 1.501849e1, 1.571467e1, 1.625729e1, 1.670348e1, 1.725602e1, 1.798145e1, 1.855656e1, 1.903751e1],
    y_plus_wall: 0.4589, seconds: 14.2, build: "single", commit: ""
};
const SINGLE_SA_MEDIUM: ZpgRecord = ZpgRecord {
    model: ZpgModel::Sa, cells: (24, 112, 96), n_cells: 13056, h: 1.336852e-2, iterations: 3700, converged: true, last_change: 9.800e-7,
    cf: [3.058472e-3, 2.993112e-3, 2.954577e-3, 2.906428e-3, 2.853956e-3, 2.819895e-3, 2.780907e-3, 2.740448e-3, 2.711427e-3, 2.688177e-3, 2.659685e-3, 2.635748e-3, 2.611886e-3, 2.593759e-3, 2.579485e-3, 2.560359e-3, 2.537940e-3, 2.518557e-3, 2.500795e-3],
    re_theta_end: 14437.1, x_profile: 1.299524e0, cf_profile: 2.611886e-3,
    up: [1.336896e1, 1.410603e1, 1.465394e1, 1.509580e1, 1.579051e1, 1.633153e1, 1.677654e1, 1.732681e1, 1.804820e1, 1.862015e1, 1.909823e1],
    y_plus_wall: 0.2415, seconds: 41.2, build: "single", commit: ""
};
const SINGLE_SA_FINE: ZpgRecord = ZpgRecord {
    model: ZpgModel::Sa, cells: (48, 224, 192), n_cells: 52224, h: 6.684260e-3, iterations: 10000, converged: false, last_change: 1.610e-5,
    cf: [3.124311e-3, 2.997451e-3, 3.088366e-3, 2.898184e-3, 2.868054e-3, 2.883914e-3, 2.821347e-3, 2.588102e-3, 2.717010e-3, 2.716931e-3, 2.616072e-3, 2.615558e-3, 2.662394e-3, 2.606526e-3, 2.634486e-3, 2.642917e-3, 2.544452e-3, 2.504931e-3, 2.459279e-3],
    re_theta_end: 14428.5, x_profile: 1.302027e0, cf_profile: 2.662394e-3,
    up: [1.322628e1, 1.394885e1, 1.448788e1, 1.492342e1, 1.561056e1, 1.614848e1, 1.659297e1, 1.714384e1, 1.786618e1, 1.843754e1, 1.891378e1],
    y_plus_wall: 0.1236, seconds: 121.8, build: "single", commit: ""
};

// ---------------------------------------------------------------------------
//  The replay
// ---------------------------------------------------------------------------

const GATE_CF: [&str; 2] = [
    "SPEC-LIT 144.5 Gate 132-Z ZPG plate, SpalartAllmaras, C_f against Karman-Schoenherr",
    "SPEC-LIT 144.5 Gate 132-Z ZPG plate, kOmegaSST, C_f against Karman-Schoenherr",
];
const GATE_UP: [&str; 2] = [
    "SPEC-LIT 144.5 Gate 132-Z ZPG plate, SpalartAllmaras, u+ against Coles at Re_theta 1e4",
    "SPEC-LIT 144.5 Gate 132-Z ZPG plate, kOmegaSST, u+ against Coles at Re_theta 1e4",
];
const AGAINST_KS: &str = "the Karman-Schoenherr C_f(Re_theta) as the NASA TMR tabulates it (a correlation, \
                          SPEC-LIT 144.5)";
const AGAINST_COLES: &str = "Coles' u+(y+) at Re_theta = 1e4 as the NASA TMR tabulates it (a theory profile, \
                             SPEC-LIT 144.5)";

/// A tabulated curve read linearly in its first column; NaN outside it.
pub(crate) fn curve_at(c: &[(f64, f64)], x: f64) -> f64 {
    c.windows(2)
        .find(|p| p[0].0 <= x && x <= p[1].0)
        .map_or(f64::NAN, |p| p[0].1 + (x - p[0].0) / (p[1].0 - p[0].0) * (p[1].1 - p[0].1))
}

/// The worst of a set of `(at, error)` by `|error|`; a NaN error wins.
fn worst(v: impl Iterator<Item = (f64, f64)>) -> (f64, f64) {
    v.fold((f64::NAN, 0.0), |m, x| {
        if m.1.is_nan() {
            m
        } else if !(x.1.abs() <= m.1.abs()) {
            x
        } else {
            m
        }
    })
}

/// The two rows of one record: the worst `C_f/C_f,KS - 1` and its
/// `Re_theta`, and the worst `u+ - u+_Coles` and its `y+`.
pub(crate) fn rows_of(r: &ZpgRecord, ks: &[(f64, f64)], coles: &[(f64, f64)]) -> ((f64, f64), (f64, f64)) {
    let cf = worst((0..N_RT).map(|k| (re_theta_at(k), r.cf[k] / curve_at(ks, re_theta_at(k)) - 1.0)));
    let up = worst(Y_PLUS.iter().zip(&r.up).map(|(&y, &u)| (y, u - curve_at(coles, y))));
    (cf, up)
}

/// §94's study of one quantity over a model's three levels.
fn study_of(recs: &[Option<&ZpgRecord>; 3], f: impl Fn(&ZpgRecord) -> f64) -> std::result::Result<GridStudy, String> {
    if !recs.iter().all(Option::is_some) {
        return Err("fewer than three levels are recorded".to_string());
    }
    let levels: Vec<Level> = recs.iter().flatten().map(|r| Level { h: r.h as Scalar, value: f(r) as Scalar }).collect();
    vv::grid_study(&levels).map_err(|e| e.to_string())
}

/// Report every gate not closed, for `why`.
fn all_open(c: &mut Checks, why: &str) {
    for mi in 0..2 {
        for (gate, against) in [(GATE_CF[mi], AGAINST_KS), (GATE_UP[mi], AGAINST_COLES)] {
            c.enter_gate(gate);
            c.report(GateReport {
                verdict: Verdict::Open,
                how: How::Replayed,
                gate,
                against,
                headline: format!("Gate 132-Z is not closed: {why}"),
                detail: vec![],
                uncertainty: Some(Uncertainty::SingleMesh("no comparison was made (SPEC-LIT 144.5)")),
            });
        }
    }
    c.leave_gate();
}

/// The section: the two keys (their digest lines or the missing lines,
/// §110.1), then [`check_zpg_on`] on the recorded runs.
pub(crate) fn check_zpg(c: &mut Checks) -> Result<()> {
    let ks = match crate::key::tmr_ks_cf() {
        Ok(k) => {
            c.note(&k.digest_line);
            Some(k.points)
        }
        Err(e) => {
            c.note(&crate::answer_key_missing_line(crate::key::TMR_KS_ID, "tmr-zpg/cf_K-S.dat", &e));
            None
        }
    };
    let coles = match crate::key::tmr_coles_uplus() {
        Ok(k) => {
            c.note(&k.digest_line);
            Some(k.points)
        }
        Err(e) => {
            c.note(&crate::answer_key_missing_line(crate::key::TMR_COLES_ID, "tmr-zpg/u+y+.dat", &e));
            None
        }
    };
    check_zpg_on(c, ks.as_deref().zip(coles.as_deref()), RECORDS, RECORDS_SINGLE)
}

pub(crate) fn check_zpg_on(
    c: &mut Checks,
    keys: Option<(&[(f64, f64)], &[(f64, f64)])>,
    records: [[Option<&ZpgRecord>; 3]; 2],
    single: [[Option<&ZpgRecord>; 3]; 2],
) -> Result<()> {
    let Some((ks, coles)) = keys else {
        all_open(c, "an answer key (tmr-ks-cf or tmr-coles-uplus) is missing; reference/PROVENANCE.md names both");
        return Ok(());
    };
    // The keys' own rows (§110.1): each covers the range its gate reads.
    let covers = (0..N_RT).all(|k| curve_at(ks, re_theta_at(k)).is_finite())
        && Y_PLUS.iter().all(|&y| curve_at(coles, y).is_finite());
    c.require("144.5: the K-S key covers Re_theta 4000..13000 and the Coles key y+ 30..300", covers);
    if !covers {
        all_open(c, "a key does not cover the range its gate reads");
        return Ok(());
    }
    c.note(&format!(
        "144.5: K-S {} points, C_f {:.6e} at Re_theta 4000, {:.6e} at 1e4, {:.6e} at 13000; Coles {} points, \
         u+ {:.4} at y+ 30, {:.4} at 100, {:.4} at 300",
        ks.len(),
        curve_at(ks, 4000.0),
        curve_at(ks, 1e4),
        curve_at(ks, 13000.0),
        coles.len(),
        curve_at(coles, 30.0),
        curve_at(coles, 100.0),
        curve_at(coles, 300.0)
    ));

    let (k0, omega0) = sst_inflow(U_INF, NU);
    c.note(&format!(
        "144.5: the rig at Re/m {RE_PER_M:e}: U {U_INF} m/s, nu {NU:e} m^2/s, the plate from x = 0 to 2 m; inflow \
         nu~ = {SA_NU_TILDA_RATIO} nu (SA), k {k0:.6e} m^2/s^2 and omega {omega0:.6e} 1/s (SST: Tu {SST_TU_PCT} %, \
         mu_t/mu {SST_NUT_RATIO}); the profile station Re_theta {RT_PROFILE:e}"
    ));

    for (mi, model) in [ZpgModel::Sa, ZpgModel::Sst].into_iter().enumerate() {
        let recs = records[mi];
        let Some(fine) = recs[0] else {
            for (gate, against) in [(GATE_CF[mi], AGAINST_KS), (GATE_UP[mi], AGAINST_COLES)] {
                c.enter_gate(gate);
                c.report(GateReport {
                    verdict: Verdict::Open,
                    how: How::Replayed,
                    gate,
                    against,
                    headline: format!(
                        "no {} run recorded on the finest mesh - NOT RUN; the record_zpg_* tests of SPEC-LIT 144.5 \
                         make it",
                        model.name()
                    ),
                    detail: vec![],
                    uncertainty: Some(Uncertainty::SingleMesh("no run recorded (SPEC-LIT 144.5)")),
                });
            }
            c.leave_gate();
            continue;
        };
        for r in recs.iter().chain(single[mi].iter()).flatten() {
            let ((rt, ecf), (yp, eup)) = rows_of(r, ks, coles);
            c.note(&format!(
                "144.5 {} {}+{} x {}: {} cells, h {:.4e}, {} iterations, stopping rule met {} (C_f change \
                 {:.2e}), Re_theta {:.0} at the trailing edge, wall y+ <= {:.3}; worst C_f {:+.2} % of K-S at \
                 Re_theta {:.0}; profile at x {:.4} (C_f {:.4e}), worst u+ {:+.3} of Coles at y+ {:.0}; {:.0} s, \
                 {} build at {}",
                r.model.name(),
                r.cells.0,
                r.cells.1,
                r.cells.2,
                r.n_cells,
                r.h,
                r.iterations,
                r.converged,
                r.last_change,
                r.re_theta_end,
                r.y_plus_wall,
                100.0 * ecf,
                rt,
                r.x_profile,
                r.cf_profile,
                eup,
                yp,
                r.seconds,
                r.build,
                r.commit
            ));
        }
        let (cf_row, up_row) = rows_of(fine, ks, coles);
        let judged = [
            (
                GATE_CF[mi],
                AGAINST_KS,
                "worst C_f against K-S over Re_theta 4000..13000",
                cf_row,
                CF_BAND,
                study_of(&recs, |r| r.cf[K_CF_STUDY]),
                curve_at(ks, re_theta_at(K_CF_STUDY)),
                "Re_theta",
                100.0,
                "%",
                "3 %",
            ),
            (
                GATE_UP[mi],
                AGAINST_COLES,
                "worst u+ against Coles over y+ 30..300 at Re_theta 1e4",
                up_row,
                UP_BAND,
                study_of(&recs, |r| r.up[K_UP_STUDY]),
                1.0,
                "y+",
                1.0,
                "",
                "1.0 in u+",
            ),
        ];
        let mut bands = Vec::new();
        for (gate, against, what, (at, e), band, study, scale, at_name, unit, sym, bar_text) in judged {
            c.enter_gate(gate);
            match &study {
                Ok(st) => c.note(&format!("144.5 {} {what}, study: {}", model.name(), st.one_line())),
                Err(why) => c.note(&format!("144.5 {} {what}, study refused: {why}", model.name())),
            }
            let u_val = study.as_ref().map_or(0.0, |st| (w(st.u_fine) / scale).abs());
            let b = band_110(e.abs(), band, u_val);
            bands.push((b, band, u_val));
            let label = format!("144.5 {}: |E| + u_val of the {what}, finest mesh (bar {bar_text})", model.name());
            if b == Band::Inside {
                c.replaying(|c| c.check(&label, (e.abs() + u_val) as Scalar, band as Scalar));
                continue;
            }
            let uncertainty = match study {
                Ok(st) => Uncertainty::Study(st),
                Err(_) => Uncertainty::SingleMesh("the 94 study refused the three levels; the finest mesh alone"),
            };
            let headline = if e.is_finite() {
                format!(
                    "{} {what}: E = {:+.3}{sym} at {at_name} {:.0} on the finest mesh, bar {bar_text}, u_val {:.3}{sym}",
                    model.name(),
                    unit * e,
                    at,
                    unit * u_val
                )
            } else {
                format!(
                    "{} {what}: not measured on the finest mesh (a station lies outside the solved range)",
                    model.name()
                )
            };
            c.report(GateReport {
                verdict: Verdict::Open,
                how: How::Replayed,
                gate,
                against,
                headline: if b == Band::Undecided {
                    format!("undecided, within u_val of the bar (SPEC-LIT 32.4, 110.1): {headline}")
                } else {
                    format!("outside the bar, against a correlation (SPEC-LIT 144.5): {headline}")
                },
                detail: vec![],
                uncertainty: Some(uncertainty),
            });
        }
        // Class R: the single build's finest record lands in the same band.
        if let Some(s) = single[mi][0] {
            let (scf, sup) = rows_of(s, ks, coles);
            let same = [scf.1, sup.1]
                .iter()
                .zip(&bands)
                .all(|(e, &(b, band, u_val))| band_110(e.abs(), band, u_val) == b);
            c.replaying(|c| {
                c.require(&format!("144.5 {} single build: each row lands in the f64 build's band", model.name()), same)
            });
        } else {
            c.note(&format!("144.5 {}: no single-build run is recorded on the finest mesh", model.name()));
        }
        c.leave_gate();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
//  Tests and the recorders
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::runner::{cf_at_re_theta, profile_at, record_literal, run_zpg, theta_of, Column, LEVELS};
    use super::*;

    fn keys() -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let ks = crate::key::tmr_ks_cf().expect("reference/tmr-zpg/cf_K-S.dat is tracked");
        let coles = crate::key::tmr_coles_uplus().expect("reference/tmr-zpg/u+y+.dat is tracked");
        (ks.points, coles.points)
    }

    #[test]
    fn the_tmr_files_are_the_k_s_relation_and_a_coles_profile() {
        let (ks, coles) = keys();
        // The page's relation (144.5's K-S), against every tabulated row.
        let worst = ks
            .iter()
            .map(|&(rt, cf)| {
                let l = rt.log10();
                (cf * (17.08 * l * l + 25.11 * l + 6.012) - 1.0).abs()
            })
            .fold(0.0f64, f64::max);
        assert!(worst <= 5e-6, "K-S file against the TMR page's relation: {worst:e}");
        assert!(ks.first().unwrap().0 <= RT_LO && ks.last().unwrap().0 >= re_theta_at(N_RT - 1));
        // Coles: the sublayer u+ = y+, and u+ rising through the log layer.
        for &(yp, up) in coles.iter().filter(|p| p.0 <= 1.2) {
            assert!((up - yp).abs() <= 1e-3, "y+ {yp}: u+ {up}");
        }
        assert!(coles.windows(2).all(|p| p[1].1 > p[0].1), "u+ increases with y+");
        assert!(coles.first().unwrap().0 <= Y_PLUS[0] && coles.last().unwrap().0 >= Y_PLUS[10]);
        println!(
            "K-S {} points, worst {worst:e} of the relation; Coles {} points, u+(100) {:.4}",
            ks.len(),
            coles.len(),
            curve_at(&coles, 100.0)
        );
    }

    #[test]
    fn theta_of_a_seventh_power_profile_is_seven_seventy_seconds_of_delta() {
        let delta = 0.01;
        let rows: Vec<(f64, f64)> = (1..=600)
            .map(|j| {
                let y = 1e-7 * 1.025f64.powi(j);
                (y, if y < delta { (y / delta).powf(1.0 / 7.0) } else { 1.0 })
            })
            .filter(|r| r.0 <= 0.05)
            .collect();
        let (theta, ue) = theta_of(&rows);
        assert_eq!(ue, 1.0);
        let exact = 7.0 / 72.0 * delta;
        assert!((theta / exact - 1.0).abs() <= 2e-3, "theta {theta:e} against {exact:e}");
        // A column that stops below 2 delta_995 (= 1.93 delta here) has no
        // theta, and neither has one whose u is not positive anywhere.
        let short: Vec<(f64, f64)> = rows.iter().copied().filter(|r| r.0 < 1.9 * delta).collect();
        assert!(theta_of(&short).0.is_nan());
        assert!(theta_of(&[(1e-3, 0.0), (2e-3, 0.0)]).0.is_nan());
    }

    fn synthetic_columns() -> Vec<Column> {
        // Re_theta = 1e3 x (column index), C_f linear in it, u = y (sublayer).
        (0..20)
            .map(|i| Column {
                x: 0.1 * i as f64,
                cf: 4e-3 - 1e-7 * (1e3 * i as f64),
                rows: (1..400).map(|j| (1e-6 * j as f64, 0.0)).collect(),
                re_theta: 1e3 * i as f64,
            })
            .collect()
    }

    #[test]
    fn re_theta_stations_and_the_profile_station_interpolate_exactly() {
        let mut cols = synthetic_columns();
        assert!((cf_at_re_theta(&cols, 4500.0) - (4e-3 - 4.5e-4)).abs() <= 1e-15);
        assert!(cf_at_re_theta(&cols, 25000.0).is_nan());
        // A linear u in y: u+ = y+ exactly wherever it is read.
        for c in &mut cols {
            let vstar = U_INF * (0.5 * c.cf).sqrt();
            for r in &mut c.rows {
                r.1 = r.0 * vstar * vstar / NU;
            }
        }
        let (x, cf, up) = profile_at(&cols, 1e4);
        assert!((x - 1.0).abs() <= 1e-12 && (cf - 3e-3).abs() <= 1e-15, "{x} {cf}");
        // The two columns' u+ = y+ rows differ in v*; linear in x between
        // them is linear in u, so u+ at the station is within the two v*s.
        for (k, &yp) in Y_PLUS.iter().enumerate() {
            if yp * NU / (U_INF * (0.5 * cf).sqrt()) < 3.9e-4 {
                assert!((up[k] / yp - 1.0).abs() <= 1e-3, "y+ {yp}: u+ {}", up[k]);
            }
        }
    }

    fn synthetic(model: ZpgModel, level: usize, dcf: f64, dup: f64, ks: &[(f64, f64)], coles: &[(f64, f64)]) -> ZpgRecord {
        let mut cf = [0.0; N_RT];
        for (k, v) in cf.iter_mut().enumerate() {
            *v = curve_at(ks, re_theta_at(k)) * (1.0 + dcf);
        }
        let mut up = [0.0; 11];
        for (k, v) in up.iter_mut().enumerate() {
            *v = curve_at(coles, Y_PLUS[k]) + dup;
        }
        ZpgRecord {
            model,
            cells: LEVELS[level],
            n_cells: 1000 >> level,
            h: 0.01 * (1 << level) as f64,
            iterations: 1000,
            converged: true,
            last_change: 1e-7,
            cf,
            re_theta_end: 14000.0,
            x_profile: 1.3,
            cf_profile: curve_at(ks, 1e4) * (1.0 + dcf),
            up,
            y_plus_wall: 0.3,
            seconds: 1.0,
            build: "f64",
            commit: "",
        }
    }

    #[test]
    fn the_replay_passes_inside_the_bars_and_reports_open_outside_them() {
        let (ks, coles) = keys();
        let sa: Vec<ZpgRecord> = (0..3).map(|l| synthetic(ZpgModel::Sa, l, 0.01, 0.5, &ks, &coles)).collect();
        let sst: Vec<ZpgRecord> = (0..3).map(|l| synthetic(ZpgModel::Sst, l, 0.05, 1.5, &ks, &coles)).collect();
        let records = [[Some(&sa[0]), Some(&sa[1]), Some(&sa[2])], [Some(&sst[0]), Some(&sst[1]), Some(&sst[2])]];
        let single = [[Some(&sa[0]), None, None], [Some(&sst[0]), None, None]];
        let mut c = Checks::new();
        check_zpg_on(&mut c, Some((&ks, &coles)), records, single).expect("replay");
        assert_eq!((c.failures, c.total, c.replayed), (0, 5, 4), "the key row, SA's two rows, two single rows");
        let got: Vec<(&str, Verdict)> = c.gates.iter().map(|g| (g.gate, g.verdict)).collect();
        assert_eq!(got, vec![(GATE_CF[1], Verdict::Open), (GATE_UP[1], Verdict::Open)], "SST's two rows are OPEN");
        assert!(c.gates.iter().all(|g| g.headline.contains("outside the bar")), "{:?}", c.gates[0].headline);
    }

    #[test]
    fn every_gate_is_open_when_a_key_is_missing_or_no_run_is_recorded() {
        let mut c = Checks::new();
        check_zpg_on(&mut c, None, RECORDS, RECORDS_SINGLE).expect("replay");
        assert_eq!((c.total, c.gates.len()), (0, 4));
        assert!(c.gates.iter().all(|g| g.verdict == Verdict::Open));
        let (ks, coles) = keys();
        let mut c = Checks::new();
        check_zpg_on(&mut c, Some((&ks, &coles)), [[None; 3]; 2], [[None; 3]; 2]).expect("replay");
        assert_eq!((c.total, c.failures, c.gates.len()), (1, 0, 4), "the key row holds, four gates open");
        assert!(c.gates.iter().all(|g| g.verdict == Verdict::Open && g.headline.contains("NOT RUN")));
    }

    /// The recorded runs of 144.5 against the tracked keys: both models'
    /// `C_f` rows are OPEN (outside 3 % of K-S near `Re_theta = 4000`) and
    /// both `u+` rows hold (SPEC-LIT 144.5 records them).
    #[test]
    fn the_recorded_runs_against_the_tmr_keys() {
        let mut c = Checks::new();
        check_zpg(&mut c).expect("replay");
        let got: Vec<(&str, Verdict)> = c.gates.iter().map(|g| (g.gate, g.verdict)).collect();
        assert_eq!(c.failures, 0);
        assert_eq!(got, vec![(GATE_CF[0], Verdict::Open), (GATE_CF[1], Verdict::Open)]);
        for g in &c.gates {
            println!("  MISSED (OPEN, a correlation) {}: {}", g.gate, g.headline);
        }
    }

    #[test]
    fn the_harness_runs_a_short_coarse_sa_solve() {
        let Ok(gpu) = ofgpu::Gpu::new(0) else { return };
        let r = run_zpg(&gpu, ZpgModel::Sa, LEVELS[2], 300, false).expect("run");
        println!("{}", record_literal(&r));
        assert_eq!(r.iterations, 300);
        assert!(r.re_theta_end.is_finite() && r.re_theta_end > 100.0, "{}", r.re_theta_end);
        assert!(r.y_plus_wall > 0.0 && r.y_plus_wall < 3.0, "{}", r.y_plus_wall);
    }

    fn record(model: ZpgModel, level: usize, max_iters: usize) {
        let gpu = ofgpu::Gpu::new(0).expect("a GPU");
        match run_zpg(&gpu, model, LEVELS[level], max_iters, true) {
            Ok(r) => println!("RECORD {}", record_literal(&r)),
            Err(e) => println!("NOT RECORDED {} {:?}: {e}", model.name(), LEVELS[level]),
        }
    }

    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.5 (up to an hour of GPU time)"]
    fn record_zpg_sa_fine() {
        record(ZpgModel::Sa, 0, 10000);
    }

    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.5"]
    fn record_zpg_sa_medium() {
        record(ZpgModel::Sa, 1, 20000);
    }

    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.5"]
    fn record_zpg_sa_coarse() {
        record(ZpgModel::Sa, 2, 20000);
    }

    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.5 (up to an hour of GPU time)"]
    fn record_zpg_sst_fine() {
        record(ZpgModel::Sst, 0, 10000);
    }

    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.5"]
    fn record_zpg_sst_medium() {
        record(ZpgModel::Sst, 1, 20000);
    }

    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.5"]
    fn record_zpg_sst_coarse() {
        record(ZpgModel::Sst, 2, 20000);
    }

    /// The price of a level: `ZPG_PRICE=<sa|sst>,<level>,<iters>` runs that
    /// many iterations with progress lines (SPEC-LIT 144.5's pricing).
    #[test]
    #[ignore = "pricing for SPEC-LIT 144.5"]
    fn price_zpg() {
        let spec = std::env::var("ZPG_PRICE").unwrap_or_else(|_| "sa,2,200".to_string());
        let f: Vec<&str> = spec.split(',').collect();
        let model = if f[0] == "sst" { ZpgModel::Sst } else { ZpgModel::Sa };
        let level: usize = f[1].parse().expect("level");
        let iters: usize = f[2].parse().expect("iters");
        record(model, level, iters);
    }
}

