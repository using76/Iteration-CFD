// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **The T3A plate on the flat-plate harness, and leg 1 measured in 2-D
//! (SPEC-LIT §144.3, docs/17 VAL-32).**
//!
//! §88.10's leg 1 is the 1-D closed form of SST's free-stream decay: from the
//! NASA/TMBWG T3A inflow it gives `Tu = 3.3530 %` at the leading edge against
//! the TMR's stated `3.300 %`. This file runs the same rig in 2-D on VAL-31's
//! harness (§144.2: the crate's SIMPLE on §144.1's plate) with `kOmegaSSTLM`
//! (§88) and `kOmegaSSTGamma` (§90), and reads the solver's own `Tu` at the
//! leading edge.
//!
//! The runs are long, so they are not made by `ofgpu-validate`: each is an
//! ignored test of this binary (`record_t3a_*`) that prints a [`T3aRecord`]
//! literal, and the records below are what the validate section replays
//! (§110.1's split of recipe and verdict). Leg 4 (§144.4, docs/17 VAL-33)
//! reads the `C_f` the same records carry against Roach & Brierley's
//! measured `C_f(Re_x)` (the ERCOFTAC T3A key, §144.6) and the laminar part
//! against Blasius.
//!
//! Every reduction is f64 whatever `Scalar` is: fields are widened on download
//! and the geometry is `geom64()` (§118.2).
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §88.10 (the rig and the closed form), §88/§89 and
//!     §90/§91 (the two models and their boundary conditions), §144.1-§144.3,
//!     §94 (the mesh study), §69 (the gate registry), §110.1 (the replay
//!     shape), §32.4 (the band rule), §144.4 (leg 4) and §144.6 (the key)
//!   the NASA/TMBWG T3A page's inflow, as §88.10 quotes it
//!   P. E. Roach and D. H. Brierley's T3A measurement, read from the fetched
//!     ERCOFTAC key at run time; no number of it is in this file
//! No GPL-licensed source was consulted.

use ofgpu::models::KOmegaSstCoeffs;
use ofgpu::vv::{self, GridStudy, Level};
use ofgpu::{Result, Scalar};

use crate::{Band, Checks, GateReport, How, Uncertainty, Verdict};

// ---------------------------------------------------------------------------
//  The rig
// ---------------------------------------------------------------------------

/// The NASA/TMBWG 2D T3A inflow, as §88.10 quotes it: `U`, the unit
/// Reynolds number, the inflow `Tu` (%), `mu_t/mu` at the inflow, the
/// leading edge's distance from the inflow, and the stated `Tu` there (%).
pub(crate) const TMR_U: f64 = 69.44;
pub(crate) const TMR_RE_PER_M: f64 = 2.0e5;
pub(crate) const TU_IN: f64 = 5.855;
pub(crate) const NUT_RATIO: f64 = 11.90;
pub(crate) const X_LE: f64 = 0.250;
pub(crate) const TU_LE_TMR: f64 = 3.300;
/// §88.10's closed-form value, to the digits it prints.
pub(crate) const TU_LE_CLOSED: f64 = 3.3530;
/// The plan's bar on both comparisons (docs/17 VAL-32).
pub(crate) const TU_TOL: f64 = 0.005;

/// The harness runs the rig at `U = 1 m/s`, `nu = 5e-6 m^2/s`: the same
/// `Re/m = 2e5` (`FlatPlateSpec::laminar()`'s free stream). (144.3a) says
/// why every `Tu(x)` is unchanged.
pub(crate) const U_INF: f64 = 1.0;
pub(crate) const NU: f64 = 5.0e-6;

/// `C_f` is recorded at `Re_x = RE_GRID_STEP (i + 1)`, `i < N_CF`, for leg 4.
pub(crate) const N_CF: usize = 23;
pub(crate) const RE_GRID_STEP: f64 = 2.5e4;

fn w(x: Scalar) -> f64 {
    f64::from(x)
}

/// The two transition models of the rig.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum T3aModel {
    Lm,
    Gamma,
}

impl T3aModel {
    pub(crate) fn name(self) -> &'static str {
        match self {
            T3aModel::Lm => "kOmegaSSTLM",
            T3aModel::Gamma => "kOmegaSSTGamma",
        }
    }
}

/// `(k_0, omega_0)` of the T3A inflow at free stream `u` and viscosity `nu`:
/// `k_0 = 3/2 (Tu_0 U)^2`, `omega_0 = k_0 / (r nu)` (§88.10).
pub(crate) fn inflow(u: f64, nu: f64) -> (f64, f64) {
    let k0 = 1.5 * (TU_IN / 100.0 * u) * (TU_IN / 100.0 * u);
    (k0, k0 / (NUT_RATIO * nu))
}

/// §88.10's closed form, `Tu(x)` in %, at free stream `u` and viscosity `nu`,
/// with the crate's own SST constants.
pub(crate) fn tu_closed(u: f64, nu: f64, x: f64) -> f64 {
    let c = KOmegaSstCoeffs::default();
    let (beta2, beta_star) = (w(c.beta_2), w(c.beta_star));
    let (_, omega0) = inflow(u, nu);
    let tau = 1.0 + beta2 * omega0 * x / u;
    TU_IN * tau.powf(-beta_star / (2.0 * beta2))
}

// ---------------------------------------------------------------------------
//  One run
// ---------------------------------------------------------------------------

/// What one recorded run leaves behind - the replay slot's type.
#[derive(Clone, Copy, Debug)]
pub(crate) struct T3aRecord {
    pub(crate) model: T3aModel,
    pub(crate) cells: (usize, usize, usize),
    pub(crate) n_cells: usize,
    /// (94.1) in 2-D: `sqrt(A/N)`, `A` the x-y area.
    pub(crate) h: f64,
    pub(crate) iterations: usize,
    pub(crate) converged: bool,
    pub(crate) last_change: f64,
    /// `Tu` at the leading edge (%), (144.3b), on the row nearest `H/2`.
    pub(crate) tu_le: f64,
    /// The least and greatest `Tu` at the edge over the rows `0.1 H..0.9 H`.
    pub(crate) tu_le_min: f64,
    pub(crate) tu_le_max: f64,
    /// `C_f` at `Re_x = 2.5e4 (i + 1)`, linear in `x` between the two
    /// faces bracketing it.
    pub(crate) cf: [f64; N_CF],
    /// The plate face of least `C_f` in the band: its `Re_x` and `C_f`.
    pub(crate) cf_min_re_x: f64,
    pub(crate) cf_min: f64,
    pub(crate) seconds: f64,
    /// `"f64"` or `"single"`, and the commit the binary was built from.
    pub(crate) build: &'static str,
    pub(crate) commit: &'static str,
}

/// The recorder: one run of one model on one level, on VAL-31's harness.
/// Test code, because the runs are made by this binary's ignored tests
/// (`record_t3a_*`), never by `ofgpu-validate` itself (SPEC-LIT 110.1).
#[cfg(test)]
pub(crate) mod runner {
    use std::collections::BTreeMap;
    use std::time::Instant;

    use ofgpu::blockgen::{build_flat_plate_mesh, FlatPlateSpec};
    use ofgpu::field::{BcKind, GpuScalarField};
    use ofgpu::field_setup::{setup_scalar_field, NutRoughness, WallFaces};
    use ofgpu::fv::DivScheme;
    use ofgpu::io::case::{TurbulenceControls, WallFunctionCoeffs};
    use ofgpu::io::fields::{PatchFieldSpec, RawScalarField};
    use ofgpu::mesh::HostMesh;
    use ofgpu::models::transition::re_thetat_inlet;
    use ofgpu::models::{
        GammaCoeffs, GammaControls, KOmegaSst, KOmegaSstCoeffs, LangtryMenter, LmCoeffs, LmControls,
        MenterGamma,
    };
    use ofgpu::momentum::BuoyancyCoeffs;
    use ofgpu::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
    use ofgpu::simple::{nan_propagating_max, Simple};
    use ofgpu::{Error, Gpu, GpuMesh, Label, Result, Scalar, Vec3};

    use super::super::blasius_plate::{controls, seed_free_stream, set_bcs, wall_faces, WallFace};
    use super::{inflow, w, T3aModel, T3aRecord, N_CF, NU, RE_GRID_STEP, TU_IN, U_INF, X_LE};

    /// The three meshes, finest first: `(n_upstream, n_plate, n_y)`; each
    /// coarser level halves every count.
    pub(crate) const LEVELS: [(usize, usize, usize); 3] = [(48, 256, 128), (24, 128, 64), (12, 64, 32)];

    /// The band the stopping rule reads (§144.2's lower end).
    const RE_X_LO: f64 = 2e4;

    /// The stopping rule: every `CHECK_EVERY` iterations, after at least
    /// `MIN_ITERS`, the largest relative `C_f` change over the band.
    const CHECK_EVERY: usize = 100;
    const MIN_ITERS: usize = 1000;
    const STOP_TOL: f64 = 1e-6;

    /// The rig as inputs: the plate's leading edge and box, the end-cell
    /// ratio in `y`, the viscosity, and the inflow state. `RigInputs::t3a()`
    /// is §144.3's; `t3_fitted.rs` builds the T3B and T3A- rigs and the
    /// refitted T3A one from it (SPEC-LIT §144.7).
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct RigInputs {
        pub(crate) x_le: f64,
        pub(crate) length: f64,
        pub(crate) height: f64,
        pub(crate) r_y: f64,
        pub(crate) nu: f64,
        /// %, for `re_thetat_inlet` and the record.
        pub(crate) tu_in: f64,
        pub(crate) k0: f64,
        pub(crate) omega0: f64,
    }

    impl RigInputs {
        /// §144.3's rig: `x_le` 0.25 m, `L` 3 m, `H` 0.5 m, `r_y` 1000,
        /// `nu = 5e-6 m^2/s` at `U = 1 m/s`, and the NASA/TMBWG inflow state.
        pub(crate) fn t3a() -> Self {
            let (k0, omega0) = inflow(U_INF, NU);
            Self {
                x_le: X_LE,
                length: 3.0,
                height: 0.5,
                r_y: 1000.0,
                nu: NU,
                tu_in: TU_IN,
                k0,
                omega0,
            }
        }
    }

    /// The plate of one level on `rig`: `x_up` the rig's leading edge, `L`,
    /// `H` and `r_y` the rig's, end-cell ratios 3 / 10 / 1000, `u_inf` 1 m/s,
    /// `nu` the rig's, `n` the level's counts.
    pub(crate) fn rig_spec(rig: &RigInputs, n: (usize, usize, usize)) -> FlatPlateSpec {
        let mut spec = FlatPlateSpec::laminar();
        spec.x_upstream = rig.x_le;
        spec.length = rig.length;
        spec.height = rig.height;
        spec.n_upstream = n.0;
        spec.n_plate = n.1;
        spec.n_y = n.2;
        spec.r_upstream = 3.0;
        spec.r_plate = 10.0;
        spec.r_y = rig.r_y;
        spec.u_inf = U_INF;
        spec.nu = rig.nu;
        spec
    }

    /// The plate of one level on §144.3's rig: x_up 0.25 m (the rig's),
    /// L 3 m (`Re_L = 6e5`), H 0.5 m, end-cell ratios 3 / 10 / 1000.
    pub(crate) fn t3a_spec(n: (usize, usize, usize)) -> FlatPlateSpec {
        rig_spec(&RigInputs::t3a(), n)
    }

    /// One field's boundary rules, as a driver would read them from `0/`.
    pub(crate) fn raw(name: &str, internal: f64, rules: &[(&str, &str, Option<f64>)]) -> RawScalarField {
        let mut boundary = BTreeMap::new();
        for &(patch, type_name, value) in rules {
            let spec = PatchFieldSpec {
                type_name: type_name.to_string(),
                value: value.map(|v| vec![v as Scalar]).unwrap_or_default(),
                ..PatchFieldSpec::default()
            };
            boundary.insert(patch.to_string(), spec);
        }
        RawScalarField {
            name: name.to_string(),
            dimensions: "[0 0 0 0 0 0 0]".to_string(),
            internal: vec![internal as Scalar],
            boundary,
            boundary_patterns: Vec::new(),
        }
    }

    /// The rules of a transported field: `fixedValue v` at the inlet, `wall` at
    /// the plate, zero-gradient at the outlet and top, and the constraints.
    pub(crate) fn transported(name: &str, v: f64, wall: (&'static str, Option<f64>)) -> RawScalarField {
        raw(
            name,
            v,
            &[
                ("inlet", "fixedValue", Some(v)),
                ("outlet", "zeroGradient", None),
                ("top", "zeroGradient", None),
                ("symmetry", "symmetryPlane", None),
                ("plate", wall.0, wall.1),
                ("back", "empty", None),
                ("front", "empty", None),
            ],
        )
    }

    pub(crate) fn patch_index(hm: &HostMesh, name: &str) -> Result<usize> {
        hm.patches
            .iter()
            .position(|p| p.name == name)
            .ok_or_else(|| Error::Config(format!("T3A plate (SPEC-LIT 144.3): the mesh has no patch {name}")))
    }

    /// (144.3b): `Tu` at `x = 0`, from `k` interpolated linearly in `x` between
    /// the two cell columns straddling the edge, row by row. Returns the row
    /// nearest `H/2` and the least and greatest over `0.1 H..0.9 H`.
    pub(crate) fn tu_at_le(hm: &HostMesh, k: &[f64], height: f64) -> Result<(f64, f64, f64)> {
        let g = hm.geom64();
        let n = hm.n_cells;
        let xa = (0..n).map(|c| g.c[c].x).filter(|&x| x < 0.0).fold(f64::NEG_INFINITY, f64::max);
        let xb = (0..n).map(|c| g.c[c].x).filter(|&x| x > 0.0).fold(f64::INFINITY, f64::min);
        let column = |x0: f64| -> Vec<(f64, f64)> {
            let mut v: Vec<(f64, f64)> = (0..n)
                .filter(|&c| (g.c[c].x - x0).abs() <= 1e-9)
                .map(|c| (g.c[c].y, k[c]))
                .collect();
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
            v
        };
        let (a, b) = (column(xa), column(xb));
        if a.is_empty() || a.len() != b.len() {
            return Err(Error::Config(format!(
                "T3A plate (SPEC-LIT 144.3): the columns at x = {xa} and {xb} hold {} and {} cells",
                a.len(),
                b.len()
            )));
        }
        let t = -xa / (xb - xa);
        let rows: Vec<(f64, f64)> = a
            .iter()
            .zip(&b)
            .map(|(&(y, ka), &(_, kb))| {
                let k0 = ka + t * (kb - ka);
                (y, 100.0 * (2.0 * k0 / 3.0).sqrt() / U_INF)
            })
            .collect();
        let mid = rows
            .iter()
            .min_by(|p, q| (p.0 - 0.5 * height).abs().total_cmp(&(q.0 - 0.5 * height).abs()))
            .map(|r| r.1)
            .unwrap_or(f64::NAN);
        let band = rows.iter().filter(|r| r.0 >= 0.1 * height && r.0 <= 0.9 * height).map(|r| r.1);
        let lo = band.clone().fold(f64::INFINITY, f64::min);
        let hi = band.fold(f64::NEG_INFINITY, f64::max);
        Ok((mid, lo, hi))
    }

    /// `C_f` at `Re_x = re`, linear in `x` between the two faces bracketing it.
    pub(crate) fn cf_interp(faces: &[WallFace], re: f64) -> f64 {
        for p in faces.windows(2) {
            if p[0].re_x <= re && re <= p[1].re_x {
                let t = (re - p[0].re_x) / (p[1].re_x - p[0].re_x);
                return p[0].cf + t * (p[1].cf - p[0].cf);
            }
        }
        f64::NAN
    }

    fn cf_change(a: &[WallFace], b: &[WallFace], re_l: f64) -> f64 {
        a.iter()
            .zip(b)
            .filter(|(f, _)| f.re_x >= RE_X_LO && f.re_x <= re_l)
            .map(|(f, g)| ((g.cf - f.cf) / f.cf).abs())
            .fold(0.0f64, |m, d| if m.is_nan() || d.is_nan() { f64::NAN } else { m.max(d) })
    }

    /// What one run leaves behind, before the record's own grid: everything
    /// [`run_t3a`] and `t3_fitted`'s `run_fitted` read at the end. `faces` is
    /// the whole plate, in file order, for the caller's `C_f` grid.
    pub(crate) struct RigRun {
        pub(crate) n_cells: usize,
        pub(crate) h: f64,
        pub(crate) iterations: usize,
        pub(crate) converged: bool,
        pub(crate) last_change: f64,
        pub(crate) tu_le: f64,
        pub(crate) tu_le_min: f64,
        pub(crate) tu_le_max: f64,
        pub(crate) faces: Vec<WallFace>,
        pub(crate) cf_min_re_x: f64,
        pub(crate) cf_min: f64,
        pub(crate) seconds: f64,
    }

    /// Run one model on one level of `rig`, steady, until the stopping rule or
    /// `max_iters`. `progress` prints a line per check, led by `label`.
    /// §144.3's own run is [`run_t3a`] (`rig = RigInputs::t3a()`).
    pub(crate) fn run_rig(
        gpu: &Gpu,
        rig: &RigInputs,
        label: &str,
        model: T3aModel,
        n: (usize, usize, usize),
        max_iters: usize,
        progress: bool,
    ) -> Result<RigRun> {
        let started = Instant::now();
        let spec = rig_spec(rig, n);
        let re_l = spec.u_inf * spec.length / spec.nu;
        let hm = build_flat_plate_mesh(&spec)?;
        let plate = patch_index(&hm, "plate")?;
        let gm = GpuMesh::upload(gpu, &hm)?;
        let ctrl = controls(rig.nu);
        let mut sim = Simple::new(gpu, &hm, &gm, ctrl, BuoyancyCoeffs { g: Vec3::ZERO, ..BuoyancyCoeffs::default() })?;
        set_bcs(gpu, &mut sim, &hm, U_INF)?;
        seed_free_stream(gpu, &mut sim, &hm, U_INF)?;
        sim.initialise(gpu)?;

        // The inflow state and the six fields' rules (§89.1, §91.1, (143.8)).
        let (k0, omega0) = (rig.k0, rig.omega0);
        let re_t0 = w(re_thetat_inlet(rig.tu_in as Scalar));
        let raw_k = transported("k", k0, ("fixedValue", Some(0.0)));
        let raw_omega = transported("omega", omega0, ("omegaWallFunction", Some(omega0)));
        let raw_gamma = transported("gamma", 1.0, ("zeroGradient", None));
        let raw_ret = transported("ReThetat", re_t0, ("zeroGradient", None));
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
        let wall_set = WallFaces::from_case(&raw_omega, Some(&raw_nut), &hm)?;
        let rough = NutRoughness::none(hm.n_boundary_faces);
        let wd = ofgpu::walldistance::wall_distance(gpu, &hm, &gm, &ctrl.p_solver, 0)?;

        // linearUpwind on k and omega: the free-stream decay is what leg 1
        // reads, and upwind's first-order error on it is of the order of the
        // bar (SPEC-LIT 144.3). The two transition fields keep the bounded
        // upwind default.
        let tctrl = TurbulenceControls {
            div_scheme: DivScheme::LinearUpwind,
            eps_div_scheme: DivScheme::LinearUpwind,
            ..TurbulenceControls::default()
        };
        let mut m = KOmegaSst::new(
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
        match model {
            T3aModel::Lm => {
                let mut lm = LangtryMenter::new(gpu, &gm, LmCoeffs::default(), LmControls::default(), &wd.y.f)?;
                setup_scalar_field(gpu, lm.gamma_mut(), &raw_gamma, &hm)?;
                setup_scalar_field(gpu, lm.re_thetat_mut(), &raw_ret, &hm)?;
                lm.initialise(gpu, &gm)?;
                m.set_transition(Some(lm))?;
            }
            T3aModel::Gamma => {
                let mut g = MenterGamma::new(
                    gpu,
                    &gm,
                    GammaCoeffs::default(),
                    GammaControls::default(),
                    &wd.y.f,
                    &wd.grad_y,
                )?;
                setup_scalar_field(gpu, g.gamma_mut(), &raw_gamma, &hm)?;
                g.initialise(gpu, &gm)?;
                m.set_gamma_transition(Some(g))?;
            }
        }
        setup_scalar_field(gpu, m.k_mut(), &raw_k, &hm)?;
        setup_scalar_field(gpu, m.omega_mut(), &raw_omega, &hm)?;
        setup_scalar_field(gpu, m.nut_mut(), &raw_nut, &hm)?;
        m.initialise(gpu, &sim.flow_state())?;

        let mut t = GpuScalarField::zeros(gpu, &gm, "T")?;
        gpu.write(&mut t.bc_kind, &vec![BcKind::ZeroGradient as Label; hm.n_boundary_faces])?;
        let mut backend = PbicgstabBackend::new(ctrl.p_solver);
        backend.setup(gpu, &hm, &gm, &SystemProbe::default())?;

        let widen = |v: Vec<Scalar>| -> Vec<f64> { v.into_iter().map(w).collect() };
        let mut last = wall_faces(&hm, plate, &gpu.download(&sim.u().f)?, U_INF, rig.nu);
        let (mut iterations, mut converged, mut last_change) = (max_iters, false, f64::INFINITY);
        for it in 1..=max_iters {
            let perf = sim.solve_step(gpu, &mut backend, m.nut(), &t)?;
            let (pw, pk) = m.correct(gpu, &sim.flow_state())?;
            let res = [
                perf.last.u[0].initial_residual,
                perf.last.u[1].initial_residual,
                perf.last.p.initial_residual,
                pw.initial_residual,
                pk.initial_residual,
            ]
            .into_iter()
            .fold(0.0 as Scalar, nan_propagating_max);
            if !w(res).is_finite() {
                return Err(Error::Diverged {
                    iteration: it,
                    what: format!("{label} {} {n:?}: initial residual {}", model.name(), w(res)),
                });
            }
            if it % CHECK_EVERY == 0 {
                let now = wall_faces(&hm, plate, &gpu.download(&sim.u().f)?, U_INF, rig.nu);
                last_change = cf_change(&last, &now, re_l);
                last = now;
                if progress {
                    let k = widen(gpu.download(&m.k().f)?);
                    let (tu, _, _) = tu_at_le(&hm, &k, spec.height)?;
                    println!(
                        "{label} {} {n:?} iter {it}/{max_iters} ({:.1} %): max res {:.3e}, C_f change {:.3e}, Tu_LE {:.5} %, {:.1} s",
                        model.name(),
                        100.0 * it as f64 / max_iters as f64,
                        w(res),
                        last_change,
                        tu,
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
        let faces = wall_faces(&hm, plate, &u, U_INF, rig.nu);
        let k = widen(gpu.download(&m.k().f)?);
        let (tu_le, tu_le_min, tu_le_max) = tu_at_le(&hm, &k, spec.height)?;
        let (cf_min_re_x, cf_min) = faces
            .iter()
            .filter(|f| f.re_x >= RE_X_LO && f.re_x <= re_l)
            .map(|f| (f.re_x, f.cf))
            .fold((f64::NAN, f64::INFINITY), |m, x| if x.1 < m.1 { x } else { m });
        let g = hm.geom64();
        let area: f64 = g.v.iter().sum::<f64>() / spec.depth;
        Ok(RigRun {
            n_cells: hm.n_cells,
            h: (area / hm.n_cells as f64).sqrt(),
            iterations,
            converged,
            last_change,
            tu_le,
            tu_le_min,
            tu_le_max,
            faces,
            cf_min_re_x,
            cf_min,
            seconds: started.elapsed().as_secs_f64(),
        })
    }

    /// One run of one model on one level, on §144.3's rig and inflow: the
    /// recorded run, and the shape the replay slots below hold.
    pub(crate) fn run_t3a(
        gpu: &Gpu,
        model: T3aModel,
        n: (usize, usize, usize),
        max_iters: usize,
        progress: bool,
    ) -> Result<T3aRecord> {
        let run = run_rig(gpu, &RigInputs::t3a(), "T3A", model, n, max_iters, progress)?;
        let mut cf = [f64::NAN; N_CF];
        for (i, v) in cf.iter_mut().enumerate() {
            *v = cf_interp(&run.faces, RE_GRID_STEP * (i + 1) as f64);
        }
        Ok(T3aRecord {
            model,
            cells: n,
            n_cells: run.n_cells,
            h: run.h,
            iterations: run.iterations,
            converged: run.converged,
            last_change: run.last_change,
            tu_le: run.tu_le,
            tu_le_min: run.tu_le_min,
            tu_le_max: run.tu_le_max,
            cf,
            cf_min_re_x: run.cf_min_re_x,
            cf_min: run.cf_min,
            seconds: run.seconds,
            build: if cfg!(feature = "single") { "single" } else { "f64" },
            commit: "",
        })
    }

    /// The record as the Rust literal the slots below hold.
    pub(crate) fn record_literal(r: &T3aRecord) -> String {
        let cf: Vec<String> = r.cf.iter().map(|v| format!("{v:.6e}")).collect();
        format!(
            "T3aRecord {{ model: T3aModel::{:?}, cells: {:?}, n_cells: {}, h: {:.6e}, iterations: {}, \
             converged: {}, last_change: {:.3e}, tu_le: {:.6}, tu_le_min: {:.6}, tu_le_max: {:.6}, \
             cf: [{}], cf_min_re_x: {:.6e}, cf_min: {:.6e}, seconds: {:.1}, build: \"{}\", commit: \"\" }}",
            r.model,
            r.cells,
            r.n_cells,
            r.h,
            r.iterations,
            r.converged,
            r.last_change,
            r.tu_le,
            r.tu_le_min,
            r.tu_le_max,
            cf.join(", "),
            r.cf_min_re_x,
            r.cf_min,
            r.seconds,
            r.build
        )
    }
}

// ---------------------------------------------------------------------------
//  The records (SPEC-LIT 144.3's table)
// ---------------------------------------------------------------------------

/// The replay slots: `[model][level]`, LM first, finest level first. `None`
/// is a run not recorded yet. `RECORDS` is the f64 build's, the gate's;
/// `RECORDS_SINGLE` is the same six runs under `--features single`.
pub(crate) const RECORDS: [[Option<&T3aRecord>; 3]; 2] = [
    [Some(&F64_LM_FINE), Some(&F64_LM_MEDIUM), Some(&F64_LM_COARSE)],
    [Some(&F64_GAMMA_FINE), Some(&F64_GAMMA_MEDIUM), Some(&F64_GAMMA_COARSE)],
];
pub(crate) const RECORDS_SINGLE: [[Option<&T3aRecord>; 3]; 2] = [
    [Some(&SINGLE_LM_FINE), Some(&SINGLE_LM_MEDIUM), Some(&SINGLE_LM_COARSE)],
    [Some(&SINGLE_GAMMA_FINE), Some(&SINGLE_GAMMA_MEDIUM), Some(&SINGLE_GAMMA_COARSE)],
];

// The f64 records, as `record_t3a_*` printed them (2026-10-15, the one
// RTX 5070 Ti, shared with two other lanes' work).
const F64_LM_COARSE: T3aRecord = T3aRecord {
    model: T3aModel::Lm, cells: (12, 64, 32), n_cells: 2432, h: 2.584907e-2, iterations: 1000, converged: true, last_change: 2.401e-7, tu_le: 3.349395, tu_le_min: 3.348417, tu_le_max: 3.349395,
    cf: [4.656335e-3, 3.418138e-3, 2.877587e-3, 2.857785e-3, 3.268060e-3, 3.803297e-3, 4.212684e-3, 4.497459e-3, 4.532426e-3, 4.486091e-3, 4.424474e-3, 4.360361e-3, 4.297270e-3, 4.236845e-3, 4.180080e-3, 4.127558e-3, 4.078665e-3, 4.032577e-3, 3.989461e-3, 3.949017e-3, 3.911240e-3, 3.876506e-3, 3.840193e-3],
    cf_min_re_x: 8.712690e4, cf_min: 2.797500e-3, seconds: 34.2, build: "f64", commit: "f1f8567 + VAL-32",
};
const F64_LM_MEDIUM: T3aRecord = T3aRecord {
    model: T3aModel::Lm, cells: (24, 128, 64), n_cells: 9728, h: 1.292453e-2, iterations: 3100, converged: true, last_change: 4.700e-7, tu_le: 3.352550, tu_le_min: 3.351417, tu_le_max: 3.352550,
    cf: [4.648781e-3, 3.385899e-3, 2.828602e-3, 2.814000e-3, 3.224201e-3, 3.653769e-3, 4.159792e-3, 4.525451e-3, 4.583395e-3, 4.553316e-3, 4.498816e-3, 4.437649e-3, 4.375737e-3, 4.315676e-3, 4.258686e-3, 4.205397e-3, 4.155899e-3, 4.109785e-3, 4.066747e-3, 4.026627e-3, 3.988881e-3, 3.953134e-3, 3.920379e-3],
    cf_min_re_x: 8.648130e4, cf_min: 2.743978e-3, seconds: 164.0, build: "f64", commit: "f1f8567 + VAL-32",
};
const F64_LM_FINE: T3aRecord = T3aRecord {
    model: T3aModel::Lm, cells: (48, 256, 128), n_cells: 38912, h: 6.462267e-3, iterations: 40000, converged: false, last_change: 9.369e-6, tu_le: 3.353320, tu_le_min: 3.352173, tu_le_max: 3.353320,
    cf: [4.652370e-3, 3.379588e-3, 2.819550e-3, 2.776216e-3, 3.088216e-3, 3.572794e-3, 4.082624e-3, 4.518958e-3, 4.651140e-3, 4.654843e-3, 4.607816e-3, 4.543896e-3, 4.475097e-3, 4.394006e-3, 4.331546e-3, 4.276135e-3, 4.224645e-3, 4.176814e-3, 4.132349e-3, 4.090874e-3, 4.052031e-3, 4.015434e-3, 3.980130e-3],
    cf_min_re_x: 8.893237e4, cf_min: 2.742582e-3, seconds: 1666.7, build: "f64", commit: "f1f8567 + VAL-32",
};
const F64_GAMMA_COARSE: T3aRecord = T3aRecord {
    model: T3aModel::Gamma, cells: (12, 64, 32), n_cells: 2432, h: 2.584907e-2, iterations: 1000, converged: true, last_change: 1.567e-10, tu_le: 3.349395, tu_le_min: 3.348380, tu_le_max: 3.349395,
    cf: [4.655791e-3, 3.417522e-3, 2.845437e-3, 2.550788e-3, 2.503780e-3, 2.988902e-3, 3.580043e-3, 4.002986e-3, 4.319389e-3, 4.386975e-3, 4.366386e-3, 4.321310e-3, 4.268684e-3, 4.214633e-3, 4.161154e-3, 4.109984e-3, 4.061910e-3, 4.017179e-3, 3.974964e-3, 3.935221e-3, 3.898192e-3, 3.864160e-3, 3.828018e-3],
    cf_min_re_x: 1.174328e5, cf_min: 2.475237e-3, seconds: 42.5, build: "f64", commit: "f1f8567 + VAL-32",
};
const F64_GAMMA_MEDIUM: T3aRecord = T3aRecord {
    model: T3aModel::Gamma, cells: (24, 128, 64), n_cells: 9728, h: 1.292453e-2, iterations: 2000, converged: true, last_change: 4.438e-7, tu_le: 3.352549, tu_le_min: 3.351360, tu_le_max: 3.352549,
    cf: [4.645045e-3, 3.382816e-3, 2.797856e-3, 2.450043e-3, 2.239451e-3, 2.184459e-3, 2.671635e-3, 3.461959e-3, 3.977944e-3, 4.330513e-3, 4.447818e-3, 4.445500e-3, 4.408726e-3, 4.360308e-3, 4.307460e-3, 4.254470e-3, 4.203709e-3, 4.155625e-3, 4.110192e-3, 4.067520e-3, 4.027240e-3, 3.989126e-3, 3.954147e-3],
    cf_min_re_x: 1.450687e5, cf_min: 2.176664e-3, seconds: 123.0, build: "f64", commit: "f1f8567 + VAL-32",
};
const F64_GAMMA_FINE: T3aRecord = T3aRecord {
    model: T3aModel::Gamma, cells: (48, 256, 128), n_cells: 38912, h: 6.462267e-3, iterations: 40000, converged: false, last_change: 1.879e-5, tu_le: 3.353318, tu_le_min: 3.352114, tu_le_max: 3.353319,
    cf: [4.645718e-3, 3.373389e-3, 2.785441e-3, 2.431156e-3, 2.204555e-3, 2.086304e-3, 2.185990e-3, 3.142984e-3, 3.747466e-3, 4.233863e-3, 4.451269e-3, 4.498360e-3, 4.476641e-3, 4.433693e-3, 4.382412e-3, 4.328909e-3, 4.276619e-3, 4.226511e-3, 4.178976e-3, 4.134214e-3, 4.092047e-3, 4.052190e-3, 4.014008e-3],
    cf_min_re_x: 1.563341e5, cf_min: 2.078658e-3, seconds: 1494.6, build: "f64", commit: "f1f8567 + VAL-32",
};

// The single-build records: the same six runs under `--features single`.
const SINGLE_LM_COARSE: T3aRecord = T3aRecord {
    model: T3aModel::Lm, cells: (12, 64, 32), n_cells: 2432, h: 2.584907e-2, iterations: 1000, converged: true, last_change: 6.946e-7, tu_le: 3.349395, tu_le_min: 3.348417, tu_le_max: 3.349395,
    cf: [4.656337e-3, 3.418140e-3, 2.877590e-3, 2.857791e-3, 3.268059e-3, 3.803295e-3, 4.212679e-3, 4.497455e-3, 4.532425e-3, 4.486088e-3, 4.424473e-3, 4.360361e-3, 4.297265e-3, 4.236840e-3, 4.180078e-3, 4.127558e-3, 4.078665e-3, 4.032575e-3, 3.989459e-3, 3.949016e-3, 3.911237e-3, 3.876504e-3, 3.840190e-3],
    cf_min_re_x: 8.712690e4, cf_min: 2.797504e-3, seconds: 27.9, build: "single", commit: "f1f8567 + VAL-32",
};
const SINGLE_LM_MEDIUM: T3aRecord = T3aRecord {
    model: T3aModel::Lm, cells: (24, 128, 64), n_cells: 9728, h: 1.292453e-2, iterations: 4600, converged: true, last_change: 9.846e-7, tu_le: 3.352550, tu_le_min: 3.351417, tu_le_max: 3.352550,
    cf: [4.648602e-3, 3.386063e-3, 2.828845e-3, 2.822072e-3, 3.226463e-3, 3.651908e-3, 4.159143e-3, 4.525328e-3, 4.583235e-3, 4.553145e-3, 4.498704e-3, 4.437592e-3, 4.375712e-3, 4.315640e-3, 4.258644e-3, 4.205361e-3, 4.155857e-3, 4.109732e-3, 4.066717e-3, 4.026584e-3, 3.988833e-3, 3.953084e-3, 3.920337e-3],
    cf_min_re_x: 8.648130e4, cf_min: 2.747267e-3, seconds: 137.0, build: "single", commit: "f1f8567 + VAL-32",
};
const SINGLE_LM_FINE: T3aRecord = T3aRecord {
    model: T3aModel::Lm, cells: (48, 256, 128), n_cells: 38912, h: 6.462267e-3, iterations: 40000, converged: false, last_change: 1.484e-5, tu_le: 3.353319, tu_le_min: 3.352173, tu_le_max: 3.353321,
    cf: [4.651858e-3, 3.379505e-3, 2.815591e-3, 2.767848e-3, 3.106336e-3, 3.588937e-3, 4.104870e-3, 4.531502e-3, 4.654332e-3, 4.656985e-3, 4.608100e-3, 4.543766e-3, 4.474548e-3, 4.393454e-3, 4.330982e-3, 4.275605e-3, 4.224139e-3, 4.176339e-3, 4.131886e-3, 4.090429e-3, 4.051624e-3, 4.015047e-3, 3.979746e-3],
    cf_min_re_x: 8.893237e4, cf_min: 2.731311e-3, seconds: 1011.4, build: "single", commit: "f1f8567 + VAL-32",
};
const SINGLE_GAMMA_COARSE: T3aRecord = T3aRecord {
    model: T3aModel::Gamma, cells: (12, 64, 32), n_cells: 2432, h: 2.584907e-2, iterations: 1400, converged: true, last_change: 9.711e-7, tu_le: 3.349395, tu_le_min: 3.348380, tu_le_max: 3.349395,
    cf: [4.655792e-3, 3.417525e-3, 2.845441e-3, 2.550788e-3, 2.503777e-3, 2.988903e-3, 3.580040e-3, 4.002987e-3, 4.319393e-3, 4.386976e-3, 4.366387e-3, 4.321314e-3, 4.268683e-3, 4.214631e-3, 4.161154e-3, 4.109986e-3, 4.061912e-3, 4.017179e-3, 3.974964e-3, 3.935222e-3, 3.898193e-3, 3.864161e-3, 3.828019e-3],
    cf_min_re_x: 1.174328e5, cf_min: 2.475233e-3, seconds: 32.3, build: "single", commit: "f1f8567 + VAL-32",
};
const SINGLE_GAMMA_MEDIUM: T3aRecord = T3aRecord {
    model: T3aModel::Gamma, cells: (24, 128, 64), n_cells: 9728, h: 1.292453e-2, iterations: 9400, converged: true, last_change: 9.934e-7, tu_le: 3.352549, tu_le_min: 3.351360, tu_le_max: 3.352549,
    cf: [4.644829e-3, 3.382893e-3, 2.797696e-3, 2.449432e-3, 2.237998e-3, 2.164118e-3, 2.579302e-3, 3.417798e-3, 3.944663e-3, 4.307853e-3, 4.441314e-3, 4.443745e-3, 4.408773e-3, 4.361182e-3, 4.308771e-3, 4.255941e-3, 4.205235e-3, 4.157142e-3, 4.111684e-3, 4.068966e-3, 4.028627e-3, 3.990450e-3, 3.955418e-3],
    cf_min_re_x: 1.450687e5, cf_min: 2.161866e-3, seconds: 217.5, build: "single", commit: "f1f8567 + VAL-32",
};
const SINGLE_GAMMA_FINE: T3aRecord = T3aRecord {
    model: T3aModel::Gamma, cells: (48, 256, 128), n_cells: 38912, h: 6.462267e-3, iterations: 40000, converged: false, last_change: 3.819e-5, tu_le: 3.353319, tu_le_min: 3.352115, tu_le_max: 3.353319,
    cf: [4.645273e-3, 3.373460e-3, 2.785338e-3, 2.431150e-3, 2.204440e-3, 2.086706e-3, 2.189464e-3, 3.145749e-3, 3.747542e-3, 4.233935e-3, 4.451201e-3, 4.498439e-3, 4.476663e-3, 4.433674e-3, 4.382396e-3, 4.328844e-3, 4.276553e-3, 4.226495e-3, 4.178929e-3, 4.134184e-3, 4.091982e-3, 4.052125e-3, 4.013926e-3],
    cf_min_re_x: 1.563341e5, cf_min: 2.079342e-3, seconds: 848.5, build: "single", commit: "f1f8567 + VAL-32",
};

// ---------------------------------------------------------------------------
//  Leg 1 in 2-D
// ---------------------------------------------------------------------------

/// The four gates' registry names: per model, against the closed form and
/// against the TMR's stated value.
const GATE_CLOSED: [&str; 2] = [
    "SPEC-LIT 144.3 T3A leg 1 in 2-D, kOmegaSSTLM, against the 1-D closed form",
    "SPEC-LIT 144.3 T3A leg 1 in 2-D, kOmegaSSTGamma, against the 1-D closed form",
];
const GATE_TMR: [&str; 2] = [
    "SPEC-LIT 144.3 T3A leg 1 in 2-D, kOmegaSSTLM, against the TMR's 3.300 %",
    "SPEC-LIT 144.3 T3A leg 1 in 2-D, kOmegaSSTGamma, against the TMR's 3.300 %",
];

/// The section: the closed form's own rows, then each model's records.
pub(crate) fn check_t3a_leg1(c: &mut Checks) -> Result<()> {
    check_t3a_leg1_on(c, RECORDS, RECORDS_SINGLE)
}

pub(crate) fn check_t3a_leg1_on(
    c: &mut Checks,
    records: [[Option<&T3aRecord>; 3]; 2],
    single: [[Option<&T3aRecord>; 3]; 2],
) -> Result<()> {
    let tu_tmr = tu_closed(TMR_U, TMR_U / TMR_RE_PER_M, X_LE);
    let tu_run = tu_closed(U_INF, NU, X_LE);
    c.note(&format!(
        "144.3: the closed form (88.10) gives Tu = {tu_tmr:.6} % at the leading edge from the TMR inflow, \
         {tu_run:.6} % from the harness's scaled inflow (U = 1 m/s, nu = 5e-6 m^2/s, Re/m 2e5)"
    ));
    c.check(
        "144.3: the closed form at the leading edge is 88.10's 3.3530 %",
        ((tu_tmr - TU_LE_CLOSED).abs()) as Scalar,
        5e-5,
    );
    c.check(
        "144.3: the scaled inflow keeps the closed form's Tu (144.3a)",
        ((tu_run / tu_tmr - 1.0).abs()) as Scalar,
        1e-12,
    );

    for (mi, model) in [T3aModel::Lm, T3aModel::Gamma].into_iter().enumerate() {
        let recs = records[mi];
        let Some(fine) = recs[0] else {
            for gate in [GATE_CLOSED[mi], GATE_TMR[mi]] {
                c.enter_gate(gate);
                c.report(GateReport {
                    verdict: Verdict::Open,
                    how: How::Replayed,
                    gate,
                    against: "SPEC-LIT 144.3's recorded harness run",
                    headline: format!(
                        "no {} run recorded yet - run the record_t3a_* tests of SPEC-LIT 144.3 and record them",
                        model.name()
                    ),
                    detail: vec![],
                    uncertainty: Some(Uncertainty::SingleMesh(
                        "no run recorded yet; the three-mesh study is taken when the runs are (SPEC-LIT 144.3)",
                    )),
                });
            }
            c.leave_gate();
            continue;
        };
        for r in recs.iter().chain(single[mi].iter()).flatten() {
            c.note(&format!(
                "144.3 {} {}+{} x {}: {} cells, h {:.4e}, {} iterations, stopping rule met {} (C_f change {:.2e}), \
                 Tu_LE {:.5} % (rows 0.1H..0.9H {:.5}..{:.5} %), least C_f {:.4e} at Re_x {:.4e}, \
                 C_f {:.4e} at Re_x 1e5 and {:.4e} at 5e5, {:.0} s, {} build at {}",
                r.model.name(),
                r.cells.0,
                r.cells.1,
                r.cells.2,
                r.n_cells,
                r.h,
                r.iterations,
                r.converged,
                r.last_change,
                r.tu_le,
                r.tu_le_min,
                r.tu_le_max,
                r.cf_min,
                r.cf_min_re_x,
                r.cf[3],
                r.cf[19],
                r.seconds,
                r.build,
                r.commit
            ));
        }
        let study: std::result::Result<GridStudy, String> = if recs.iter().all(Option::is_some) {
            let levels: Vec<Level> =
                recs.iter().flatten().map(|r| Level { h: r.h as Scalar, value: r.tu_le as Scalar }).collect();
            vv::grid_study(&levels).map_err(|e| e.to_string())
        } else {
            Err("fewer than three levels are recorded".to_string())
        };
        match &study {
            Ok(st) => c.note(&format!("144.3 {} Tu_LE study: {}", model.name(), st.one_line())),
            Err(e) => c.note(&format!("144.3 {} Tu_LE study refused: {e}", model.name())),
        }
        let uncertainty = || match &study {
            Ok(st) => Uncertainty::Study(st.clone()),
            Err(_) => Uncertainty::SingleMesh("fewer than three levels recorded; the finest mesh alone"),
        };

        let e_closed = (fine.tu_le / TU_LE_CLOSED - 1.0).abs();
        let e_tmr = (fine.tu_le / TU_LE_TMR - 1.0).abs();
        // Class R: the same bar under single, as 144.2's rows 3 and 4.
        let e_single = single[mi][0].map(|r| (r.tu_le / TU_LE_CLOSED - 1.0).abs());
        c.enter_gate(GATE_CLOSED[mi]);
        c.replaying(|c| {
            c.check(
                &format!("144.3 {}: |Tu_LE/3.3530 - 1|, finest mesh (0.5 %)", model.name()),
                e_closed as Scalar,
                TU_TOL as Scalar,
            );
            if let Some(e) = e_single {
                c.check(
                    &format!("144.3 {} single build: |Tu_LE/3.3530 - 1|, finest mesh (0.5 %)", model.name()),
                    e as Scalar,
                    TU_TOL as Scalar,
                );
            }
        });
        if e_single.is_none() {
            c.note(&format!("144.3 {}: no single-build run is recorded", model.name()));
        }
        if !(e_closed <= TU_TOL) || e_single.is_some_and(|e| !(e <= TU_TOL)) {
            c.report(GateReport {
                verdict: Verdict::Misses,
                how: How::Replayed,
                gate: GATE_CLOSED[mi],
                against: "SPEC-LIT 88.10's closed form of SST's free-stream decay, 3.3530 % at the leading edge",
                headline: format!(
                    "Tu at the leading edge {:.5} % on the finest mesh, {:.3} % from 3.3530 % (bar 0.5 %)",
                    fine.tu_le,
                    100.0 * e_closed
                ),
                detail: vec![],
                uncertainty: Some(uncertainty()),
            });
        }
        c.enter_gate(GATE_TMR[mi]);
        c.note(&format!(
            "144.3 {}: Tu_LE {:.5} % against the TMR's stated 3.300 %: {:.3} % (bar 0.5 %); the closed form \
             itself is {:.3} % from it",
            model.name(),
            fine.tu_le,
            100.0 * e_tmr,
            100.0 * (tu_tmr / TU_LE_TMR - 1.0).abs()
        ));
        if !(e_tmr <= TU_TOL) {
            c.report(GateReport {
                verdict: Verdict::Open,
                how: How::Replayed,
                gate: GATE_TMR[mi],
                against: "the NASA/TMBWG T3A page's stated leading-edge Tu, 3.300 % (SPEC-LIT 88.10)",
                headline: format!(
                    "Tu at the leading edge {:.5} % on the finest mesh, {:.3} % from 3.300 % (bar 0.5 %); \
                     the 1-D closed form of the same model is {:.3} % from it",
                    fine.tu_le,
                    100.0 * e_tmr,
                    100.0 * (tu_tmr / TU_LE_TMR - 1.0).abs()
                ),
                detail: vec![],
                uncertainty: Some(uncertainty()),
            });
        }
        c.leave_gate();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
//  Leg 4: C_f(Re_x) against Roach & Brierley (SPEC-LIT 144.4)
// ---------------------------------------------------------------------------

/// docs/17 VAL-33's bands, fixed in its brief: onset within 15 % in `Re_x`,
/// turbulent `C_f` within 10 % (with `u_D`), the laminar part within 10 % of
/// Blasius.
pub(crate) const ONSET_BAND: f64 = 0.15;
pub(crate) const TURB_BAND: f64 = 0.10;
pub(crate) const LAM_BAND: f64 = 0.10;
/// `u_D` of a turbulent `C_f`: the case page's 2 % agreement of its three
/// methods in turbulent layers (§144.6).
pub(crate) const U_D_TURB: f64 = 0.02;
/// The laminar part starts where §144.2's band does.
pub(crate) const RE_X_LAM_LO: f64 = 2.0e4;
/// Blasius' `C_f sqrt(Re_x)` (§88.7).
pub(crate) const BLASIUS_CF: f64 = 0.664;

/// The three leg-4 gates of each model (Gate 88-T for LM, 90-T for gamma).
const GATE_ONSET: [&str; 2] = [
    "SPEC-LIT 144.4 Gate 88-T leg 4, kOmegaSSTLM, transition onset Re_x against Roach & Brierley",
    "SPEC-LIT 144.4 Gate 90-T leg 4, kOmegaSSTGamma, transition onset Re_x against Roach & Brierley",
];
const GATE_TURB: [&str; 2] = [
    "SPEC-LIT 144.4 Gate 88-T leg 4, kOmegaSSTLM, turbulent C_f against Roach & Brierley",
    "SPEC-LIT 144.4 Gate 90-T leg 4, kOmegaSSTGamma, turbulent C_f against Roach & Brierley",
];
const GATE_LAM: [&str; 2] = [
    "SPEC-LIT 144.4 Gate 88-T leg 4, kOmegaSSTLM, laminar C_f against Blasius",
    "SPEC-LIT 144.4 Gate 90-T leg 4, kOmegaSSTGamma, laminar C_f against Blasius",
];
const AGAINST_KEY: &str = "Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A key (SPEC-LIT 144.6)";
const AGAINST_BLASIUS: &str =
    "Blasius' 0.664/sqrt(Re_x) (SPEC-LIT 88.7) over the part of the plate the key measures laminar";

/// What leg 4 reads from the key's stations `(Re_x, C_f)`, file order.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Leg4Key {
    /// (144.4a): the vertex of the parabola through the least-`C_f` station
    /// and its two neighbours.
    pub(crate) onset_re_x: f64,
    /// The least-`C_f` station (first on a tie) and the greatest-`C_f`
    /// station downstream of it.
    pub(crate) least: usize,
    pub(crate) peak: usize,
    /// The turbulent stations: `peak` to the last.
    pub(crate) turbulent: Vec<(f64, f64)>,
    /// The laminar part ends at the station before `least`.
    pub(crate) laminar_hi: f64,
}

/// The key's reading for leg 4, refused when the least `C_f` is not at an
/// interior station (there would be no rise to call transition).
pub(crate) fn read_key(st: &[(f64, f64)]) -> std::result::Result<Leg4Key, String> {
    let least = (0..st.len())
        .reduce(|m, i| if st[i].1 < st[m].1 { i } else { m })
        .ok_or_else(|| "the key has no station".to_string())?;
    if least == 0 || least + 1 >= st.len() {
        return Err(format!(
            "the least C_f is at station {} of {}, not an interior one, so the key shows no transition",
            least + 1,
            st.len()
        ));
    }
    let peak = (least + 1..st.len()).reduce(|m, i| if st[i].1 > st[m].1 { i } else { m }).unwrap_or(least + 1);
    let ((x0, y0), (x1, y1), (x2, y2)) = (st[least - 1], st[least], st[least + 1]);
    Ok(Leg4Key {
        onset_re_x: vertex((x0, y0), (x1, y1), (x2, y2)),
        least,
        peak,
        turbulent: st[peak..].to_vec(),
        laminar_hi: st[least - 1].0,
    })
}

/// (144.4a): the vertex of the parabola through three `(Re_x, C_f)` points
/// - the key's least-`C_f` station's, and (144.7e)'s grid-minimum reading of
/// a record whose least face is its trailing edge (t3_fitted.rs).
pub(crate) fn vertex((x0, y0): (f64, f64), (x1, y1): (f64, f64), (x2, y2): (f64, f64)) -> f64 {
    let num = (x1 - x0) * (x1 - x0) * (y1 - y2) - (x1 - x2) * (x1 - x2) * (y1 - y0);
    let den = (x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0);
    x1 - 0.5 * num / den
}

/// (144.4b): a record's `C_f` at `re_x`, interpolated as a power law between
/// the two recorded points that bracket it. NaN outside the recorded span.
pub(crate) fn cf_at(r: &T3aRecord, re_x: f64) -> f64 {
    for k in 0..N_CF - 1 {
        let (g0, g1) = (RE_GRID_STEP * (k + 1) as f64, RE_GRID_STEP * (k + 2) as f64);
        if re_x >= g0 && re_x <= g1 {
            let t = (re_x / g0).ln() / (g1 / g0).ln();
            return (r.cf[k].ln() + t * (r.cf[k + 1].ln() - r.cf[k].ln())).exp();
        }
    }
    f64::NAN
}

/// (144.4c) at the record's own point `k`: `C_f sqrt(Re_x)/0.664 - 1`.
fn blasius_excess(r: &T3aRecord, k: usize) -> f64 {
    let g = RE_GRID_STEP * (k + 1) as f64;
    r.cf[k] * g.sqrt() / BLASIUS_CF - 1.0
}

/// The record's points in the laminar part: from 2e4 to the key's
/// `laminar_hi`, and upstream of the record's own least `C_f`.
fn laminar_points(r: &T3aRecord, k: &Leg4Key) -> Vec<usize> {
    (0..N_CF)
        .filter(|&i| {
            let g = RE_GRID_STEP * (i + 1) as f64;
            g >= RE_X_LAM_LO && g <= k.laminar_hi && g < r.cf_min_re_x
        })
        .collect()
}

/// The worst `(where, signed error)` of a list, by magnitude; a NaN wins, so
/// it reaches the check as a failure. An empty list is NaN. §144.7's leg 3
/// reads it too (SPEC-LIT 144.7, t3_fitted.rs).
pub(crate) fn worst(v: impl Iterator<Item = (f64, f64)>) -> (f64, f64) {
    v.fold((f64::NAN, f64::NAN), |m, x| {
        if m.1.is_nan() && !m.0.is_nan() {
            m
        } else if m.0.is_nan() || x.1.is_nan() || x.1.abs() > m.1.abs() {
            x
        } else {
            m
        }
    })
}

/// One record's three leg-4 numbers: each a `(Re_x, signed relative error)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Leg4Rows {
    /// The record's least-`C_f` face against (144.4a).
    pub(crate) onset: (f64, f64),
    /// The worst turbulent station, against its measured `C_f`.
    pub(crate) turbulent: (f64, f64),
    /// The worst laminar point, (144.4c).
    pub(crate) laminar: (f64, f64),
}

pub(crate) fn leg4_rows(r: &T3aRecord, k: &Leg4Key) -> Leg4Rows {
    Leg4Rows {
        onset: (r.cf_min_re_x, r.cf_min_re_x / k.onset_re_x - 1.0),
        turbulent: worst(k.turbulent.iter().map(|&(re, cf)| (re, cf_at(r, re) / cf - 1.0))),
        laminar: worst(laminar_points(r, k).into_iter().map(|i| (RE_GRID_STEP * (i + 1) as f64, blasius_excess(r, i)))),
    }
}

/// §94's study of one quantity over a model's three levels.
fn study_of(
    recs: &[Option<&T3aRecord>; 3],
    f: impl Fn(&T3aRecord) -> f64,
) -> std::result::Result<GridStudy, String> {
    if !recs.iter().all(Option::is_some) {
        return Err("fewer than three levels are recorded".to_string());
    }
    let levels: Vec<Level> = recs.iter().flatten().map(|r| Level { h: r.h as Scalar, value: f(r) as Scalar }).collect();
    vv::grid_study(&levels).map_err(|e| e.to_string())
}

/// §110.1's band rule as its text states it (§32.4): a result within its
/// `u_val` of the band edge, on either side, is undecided - not a pass and
/// not a miss. Inside is `|E| + u_val <= band`, outside `|E| - u_val > band`.
/// (`crate::band_of`, which 110-A/B/C use, states the same rule since
/// docs/17 VAL-51; before it made only the outside half undecided.) A NaN is
/// outside.
pub(crate) fn band_110(e_abs: f64, band: f64, u_val: f64) -> Band {
    if e_abs + u_val <= band {
        Band::Inside
    } else if e_abs - u_val <= band {
        Band::Undecided
    } else {
        Band::Outside
    }
}

/// Report every leg-4 gate not closed, for `why`.
fn leg4_all_open(c: &mut Checks, why: &str) {
    for mi in 0..2 {
        for gate in [GATE_ONSET[mi], GATE_TURB[mi], GATE_LAM[mi]] {
            c.enter_gate(gate);
            c.report(GateReport {
                verdict: Verdict::Open,
                how: How::Replayed,
                gate,
                against: if gate == GATE_LAM[mi] { AGAINST_BLASIUS } else { AGAINST_KEY },
                headline: format!("leg 4 is not closed: {why}"),
                detail: vec![],
                uncertainty: Some(Uncertainty::SingleMesh("no comparison was made (SPEC-LIT 144.4)")),
            });
        }
    }
    c.leave_gate();
}

/// The section: the key (its digest line or the missing line, §110.1), then
/// [`check_t3a_leg4_on`] on the recorded runs.
pub(crate) fn check_t3a_leg4(c: &mut Checks) -> Result<()> {
    let key = match crate::key::ercoftac_t3a() {
        Ok(k) => {
            c.note(&k.digest_line);
            Some(k.stations.iter().map(|s| (s.re_x, s.cf)).collect::<Vec<_>>())
        }
        Err(e) => {
            c.note(&crate::answer_key_missing_line(crate::key::T3A_ID, "ercoftac_t3a/t3ay.dat", &e));
            None
        }
    };
    check_t3a_leg4_on(c, key.as_deref(), RECORDS, RECORDS_SINGLE)
}

pub(crate) fn check_t3a_leg4_on(
    c: &mut Checks,
    key: Option<&[(f64, f64)]>,
    records: [[Option<&T3aRecord>; 3]; 2],
    single: [[Option<&T3aRecord>; 3]; 2],
) -> Result<()> {
    let Some(stations) = key else {
        leg4_all_open(c, "answer key ercoftac-t3a missing (SPEC-LIT 144.6 says how to place it)");
        return Ok(());
    };
    // The key's own row (§110.1): leg 4's reading needs a transition in it.
    let k = read_key(stations);
    c.require("144.4: the key's least C_f lies at an interior station (144.4a has a vertex)", k.is_ok());
    let k = match k {
        Ok(k) => k,
        Err(why) => {
            c.note(&format!("144.4: {why}"));
            leg4_all_open(c, "the key shows no transition to hold a run against");
            return Ok(());
        }
    };
    c.note(&format!(
        "144.4: {} stations; measured onset (144.4a) Re_x {:.5e} about the least-C_f station {}; turbulent \
         stations {}..{} (from Re_x {:.4e}, the greatest C_f past the onset); laminar part Re_x {:.0e}..{:.4e}",
        stations.len(),
        k.onset_re_x,
        k.least + 1,
        k.peak + 1,
        stations.len(),
        k.turbulent[0].0,
        RE_X_LAM_LO,
        k.laminar_hi
    ));

    for (mi, model) in [T3aModel::Lm, T3aModel::Gamma].into_iter().enumerate() {
        let recs = records[mi];
        let Some(fine) = recs[0] else {
            for gate in [GATE_ONSET[mi], GATE_TURB[mi], GATE_LAM[mi]] {
                c.enter_gate(gate);
                c.report(GateReport {
                    verdict: Verdict::Open,
                    how: How::Replayed,
                    gate,
                    against: if gate == GATE_LAM[mi] { AGAINST_BLASIUS } else { AGAINST_KEY },
                    headline: format!(
                        "no {} run recorded yet - run the record_t3a_* tests of SPEC-LIT 144.3 and record them",
                        model.name()
                    ),
                    detail: vec![],
                    uncertainty: Some(Uncertainty::SingleMesh(
                        "no run recorded yet; the three-mesh study is taken when the runs are (SPEC-LIT 144.3)",
                    )),
                });
            }
            c.leave_gate();
            continue;
        };
        for r in recs.iter().chain(single[mi].iter()).flatten() {
            let v = leg4_rows(r, &k);
            c.note(&format!(
                "144.4 {} {}+{} x {} ({}): onset Re_x {:.4e} ({:+.2} %), worst turbulent C_f {:+.2} % at Re_x \
                 {:.4e}, worst laminar C_f {:+.2} % of Blasius at Re_x {:.3e}",
                r.model.name(),
                r.cells.0,
                r.cells.1,
                r.cells.2,
                r.build,
                v.onset.0,
                100.0 * v.onset.1,
                100.0 * v.turbulent.1,
                v.turbulent.0,
                100.0 * v.laminar.1,
                v.laminar.0
            ));
        }
        let rows = leg4_rows(fine, &k);
        let turb_cf = k.turbulent.iter().find(|s| s.0 == rows.turbulent.0).map_or(f64::NAN, |s| s.1);
        let lam_k = ((rows.laminar.0 / RE_GRID_STEP).round() as usize).saturating_sub(1).min(N_CF - 1);
        // The row's number, its band, u_D, its study (of the same quantity
        // on the three levels) and the scale that makes u_num relative.
        let judged = [
            (
                GATE_ONSET[mi],
                AGAINST_KEY,
                "onset Re_x (least C_f) against (144.4a)",
                rows.onset,
                ONSET_BAND,
                0.0,
                study_of(&recs, |r| r.cf_min_re_x),
                k.onset_re_x,
            ),
            (
                GATE_TURB[mi],
                AGAINST_KEY,
                "worst turbulent C_f against the station's (144.4b)",
                rows.turbulent,
                TURB_BAND,
                U_D_TURB,
                study_of(&recs, |r| cf_at(r, rows.turbulent.0)),
                turb_cf,
            ),
            (
                GATE_LAM[mi],
                AGAINST_BLASIUS,
                "worst laminar C_f against Blasius (144.4c)",
                rows.laminar,
                LAM_BAND,
                0.0,
                study_of(&recs, |r| blasius_excess(r, lam_k)),
                1.0,
            ),
        ];
        let mut bands = Vec::new();
        for (gate, against, what, (at, e), band, u_d, study, scale) in judged {
            c.enter_gate(gate);
            match &study {
                Ok(st) => c.note(&format!("144.4 {} {what}, study: {}", model.name(), st.one_line())),
                Err(why) => c.note(&format!("144.4 {} {what}, study refused: {why}", model.name())),
            }
            let u_num = study.as_ref().map_or(0.0, |st| w(st.u_fine) / scale);
            let u_val = (u_num * u_num + u_d * u_d).sqrt();
            let label =
                format!("144.4 {}: |E| + u_val of the {what}, finest mesh ({:.0} %)", model.name(), 100.0 * band);
            let b = band_110(e.abs(), band, u_val);
            bands.push((b, band, u_val));
            if !e.is_finite() || b == Band::Inside {
                c.replaying(|c| c.check(&label, (e.abs() + u_val) as Scalar, band as Scalar));
                continue;
            }
            let uncertainty = match study {
                Ok(st) => Uncertainty::Study(st),
                Err(_) => Uncertainty::SingleMesh("the 94 study refused the three levels; the finest mesh alone"),
            };
            let headline = format!(
                "{} {what}: E = {:+.2} % at Re_x {:.4e} on the finest mesh, band {:.0} %, u_val {:.2} % \
                 (u_num {:.2} %, u_D {:.0} %)",
                model.name(),
                100.0 * e,
                at,
                100.0 * band,
                100.0 * u_val,
                100.0 * u_num,
                100.0 * u_d
            );
            c.report(GateReport {
                verdict: if b == Band::Undecided { Verdict::Open } else { Verdict::Misses },
                how: How::Replayed,
                gate,
                against,
                headline: if b == Band::Undecided {
                    format!("undecided, within u_val of the band edge (SPEC-LIT 32.4, 110.1): {headline}")
                } else {
                    headline
                },
                detail: vec![],
                uncertainty: Some(uncertainty),
            });
        }
        // Class R: the single build's finest record lands in the same band.
        if let Some(s) = single[mi][0] {
            let v = leg4_rows(s, &k);
            let same = [v.onset.1, v.turbulent.1, v.laminar.1]
                .iter()
                .zip(&bands)
                .all(|(e, &(b, band, u_val))| band_110(e.abs(), band, u_val) == b);
            c.replaying(|c| {
                c.require(
                    &format!("144.4 {} single build: each leg-4 row lands in the f64 build's band", model.name()),
                    same,
                )
            });
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
    use super::runner::*;
    use super::*;
    use ofgpu::Gpu;

    #[test]
    fn the_closed_form_gives_88_10s_leading_edge_tu_from_both_inflows() {
        let tmr = tu_closed(TMR_U, TMR_U / TMR_RE_PER_M, X_LE);
        assert!((tmr - TU_LE_CLOSED).abs() <= 5e-5, "{tmr}");
        let run = tu_closed(U_INF, NU, X_LE);
        assert!((run / tmr - 1.0).abs() <= 1e-12, "{run} vs {tmr}");
        let (k0, omega0) = inflow(U_INF, NU);
        assert!((k0 / 5.14215375e-3 - 1.0).abs() <= 1e-14, "{k0}");
        assert!((omega0 - 86.4227521).abs() <= 1e-6, "{omega0}");
    }

    fn synthetic(model: T3aModel, level: usize, tu: f64) -> T3aRecord {
        let n = LEVELS[level];
        T3aRecord {
            model,
            cells: n,
            n_cells: (n.0 + n.1) * n.2,
            h: 0.01 * (1 << level) as f64,
            iterations: 1000,
            converged: true,
            last_change: 1e-7,
            tu_le: tu,
            tu_le_min: tu,
            tu_le_max: tu,
            cf: [3e-3; N_CF],
            cf_min_re_x: 1e5,
            cf_min: 2e-3,
            seconds: 1.0,
            build: "f64",
            commit: "",
        }
    }

    #[test]
    fn leg1_replay_holds_the_closed_form_and_reports_the_tmr_row_on_synthetic_records() {
        let lm = [0, 1, 2].map(|l| synthetic(T3aModel::Lm, l, TU_LE_CLOSED * (1.0 + 1e-4 * l as f64)));
        let ga = [0, 1, 2].map(|l| synthetic(T3aModel::Gamma, l, TU_LE_CLOSED * (1.0 + 1e-4 * l as f64)));
        let recs = [[Some(&lm[0]), Some(&lm[1]), Some(&lm[2])], [Some(&ga[0]), Some(&ga[1]), Some(&ga[2])]];
        let mut c = Checks::new();
        check_t3a_leg1_on(&mut c, recs, [[None; 3]; 2]).expect("replay");
        assert_eq!(c.failures, 0);
        assert_eq!(c.total, 4, "two closed-form rows and one row per model");
        assert_eq!(c.replayed, 2);
        let names: Vec<&str> = c.gates.iter().map(|g| g.gate).collect();
        assert_eq!(names, vec![GATE_TMR[0], GATE_TMR[1]], "3.3530 is 1.6 % from 3.300");
        assert!(c.gates.iter().all(|g| g.verdict == Verdict::Open));
    }

    #[test]
    fn leg1_reports_a_closed_form_miss_and_each_unrecorded_model() {
        let lm = [0, 1, 2].map(|l| synthetic(T3aModel::Lm, l, 3.300));
        let recs = [[Some(&lm[0]), Some(&lm[1]), Some(&lm[2])], [None, None, None]];
        let mut c = Checks::new();
        check_t3a_leg1_on(&mut c, recs, [[None; 3]; 2]).expect("replay");
        assert_eq!(c.failures, 1, "3.300 is 1.6 % from the closed form");
        let names: Vec<&str> = c.gates.iter().map(|g| g.gate).collect();
        assert_eq!(names, vec![GATE_CLOSED[0], GATE_CLOSED[1], GATE_TMR[1]]);
        assert_eq!(c.gates[0].verdict, Verdict::Misses);
    }

    #[test]
    fn the_recorded_runs_hold_the_closed_form_and_leave_the_tmr_rows_open() {
        let mut c = Checks::new();
        check_t3a_leg1(&mut c).expect("replay");
        assert_eq!(c.failures, 0);
        let names: Vec<&str> = c.gates.iter().map(|g| g.gate).collect();
        assert_eq!(names, vec![GATE_TMR[0], GATE_TMR[1]]);
        assert_eq!((c.total, c.replayed), (6, 4), "two closed-form rows, then f64 and single per model");
        for (set, build) in [(RECORDS, "f64"), (RECORDS_SINGLE, "single")] {
            for (mi, model) in set.iter().enumerate() {
                let r: Vec<&T3aRecord> = model.iter().flatten().copied().collect();
                assert_eq!(r.len(), 3);
                assert!(r.iter().all(|r| r.build == build && r.cf.iter().all(|v| v.is_finite() && *v > 0.0)));
                assert!(r.iter().all(|r| r.model == [T3aModel::Lm, T3aModel::Gamma][mi]));
                assert!(r.windows(2).all(|p| p[0].h < p[1].h), "finest first");
            }
        }
    }

    // ---- leg 4 (SPEC-LIT 144.4); every key here is invented --------------

    #[test]
    fn leg4_reads_a_synthetic_key_exactly() {
        let st = [(1e4, 5e-3), (3e4, 3e-3), (5e4, 2e-3), (7e4, 4e-3), (9e4, 6e-3), (11e4, 5e-3)];
        let k = read_key(&st).expect("an interior least C_f");
        // (144.4a) through (3e4, 3e-3), (5e4, 2e-3), (7e4, 4e-3): 140000/3.
        assert!((k.onset_re_x / (140000.0 / 3.0) - 1.0).abs() <= 1e-12, "{}", k.onset_re_x);
        assert_eq!((k.least, k.peak, k.laminar_hi), (2, 4, 3e4));
        assert_eq!(k.turbulent, vec![(9e4, 6e-3), (11e4, 5e-3)]);
        let falling = [(1e4, 5e-3), (2e4, 4e-3), (3e4, 3e-3)];
        assert!(read_key(&falling).unwrap_err().contains("station 3 of 3"));
        let rising = [(1e4, 1e-3), (2e4, 4e-3), (3e4, 5e-3)];
        assert!(read_key(&rising).unwrap_err().contains("station 1 of 3"));
        assert!(read_key(&[]).is_err());
    }

    #[test]
    fn band_110_is_undecided_within_u_val_of_either_side_of_the_edge() {
        assert_eq!(band_110(0.07, 0.10, 0.02), Band::Inside);
        assert_eq!(band_110(0.09, 0.10, 0.02), Band::Undecided);
        assert_eq!(band_110(0.11, 0.10, 0.02), Band::Undecided);
        assert_eq!(band_110(0.13, 0.10, 0.02), Band::Outside);
        assert_eq!(band_110(f64::NAN, 0.10, 0.02), Band::Outside);
    }

    /// An invented key: laminar to 1.1e5, least C_f at 1.5e5, the greatest
    /// past it at 3e5. Its turbulent C_f is `TURB_A Re_x^-0.2`.
    const TURB_A: f64 = 0.06;
    fn synthetic_key() -> Vec<(f64, f64)> {
        let mut st = vec![(3e4, 4e-3), (6e4, 2.8e-3), (1.1e5, 2.2e-3), (1.5e5, 2.0e-3), (2e5, 3e-3)];
        st.extend([3e5, 4e5, 5e5].map(|re: f64| (re, TURB_A * re.powf(-0.2))));
        st
    }

    /// A record whose onset is `1.39e5 (1 + onset)` (the key's is 1.39e5), whose laminar points sit
    /// `lam` above Blasius and whose turbulent C_f is `(1 + turb)` times the
    /// key's; level `l` scales all of it by `1 + 1e-4 4^l` (second order).
    fn synthetic_leg4(model: T3aModel, level: usize, onset: f64, turb: f64, lam: f64) -> T3aRecord {
        let f = 1.0 + 1e-4 * 4f64.powi(level as i32);
        let mut r = synthetic(model, level, TU_LE_CLOSED);
        for (i, v) in r.cf.iter_mut().enumerate() {
            let g = RE_GRID_STEP * (i + 1) as f64;
            *v = f * if g < 1.5e5 { (1.0 + lam) * BLASIUS_CF / g.sqrt() } else { (1.0 + turb) * TURB_A * g.powf(-0.2) };
        }
        r.cf_min_re_x = 1.39e5 * (1.0 + onset) * f;
        r
    }

    #[test]
    fn leg4_replay_passes_misses_and_reports_on_synthetic_records() {
        let key = synthetic_key();
        let k = read_key(&key).expect("key");
        // (144.4a) through (1.1e5, 2.2e-3), (1.5e5, 2.0e-3), (2e5, 3e-3).
        assert!((k.onset_re_x / 1.39e5 - 1.0).abs() <= 1e-12, "{}", k.onset_re_x);
        let lm = [0, 1, 2].map(|l| synthetic_leg4(T3aModel::Lm, l, 0.05, 0.05, 0.05));
        let ga = [0, 1, 2].map(|l| synthetic_leg4(T3aModel::Gamma, l, 0.05, 0.05, 0.05));
        let recs = [[Some(&lm[0]), Some(&lm[1]), Some(&lm[2])], [Some(&ga[0]), Some(&ga[1]), Some(&ga[2])]];
        let mut c = Checks::new();
        check_t3a_leg4_on(&mut c, Some(&key), recs, [[None; 3]; 2]).expect("replay");
        assert_eq!((c.failures, c.total, c.replayed), (0, 7, 6), "the key row, then three rows per model");
        assert!(c.gates.is_empty());
        let v = leg4_rows(&lm[0], &k);
        assert!((v.turbulent.1 - (1.05 * 1.0001 - 1.0)).abs() <= 1e-12, "{v:?}");
        assert!((v.laminar.1 - (1.05 * 1.0001 - 1.0)).abs() <= 1e-12, "{v:?}");
        assert_eq!(laminar_points(&lm[0], &k), vec![0, 1, 2, 3], "2.5e4..1e5: from 2e4 to 1.1e5");

        // LM misses its onset and its laminar part; gamma's turbulent C_f is
        // 11 % off, inside 10 % + u_val, so undecided.
        let lm = [0, 1, 2].map(|l| synthetic_leg4(T3aModel::Lm, l, 0.30, 0.05, 0.20));
        let ga = [0, 1, 2].map(|l| synthetic_leg4(T3aModel::Gamma, l, 0.05, 0.11, 0.05));
        let recs = [[Some(&lm[0]), Some(&lm[1]), Some(&lm[2])], [Some(&ga[0]), Some(&ga[1]), Some(&ga[2])]];
        let single = [[Some(&lm[0]), None, None], [Some(&ga[0]), None, None]];
        let mut c = Checks::new();
        check_t3a_leg4_on(&mut c, Some(&key), recs, single).expect("replay");
        assert_eq!((c.failures, c.total, c.replayed), (0, 6, 5), "key row, three in-band rows, two single rows");
        let got: Vec<(&str, Verdict)> = c.gates.iter().map(|g| (g.gate, g.verdict)).collect();
        assert_eq!(
            got,
            vec![
                (GATE_ONSET[0], Verdict::Misses),
                (GATE_LAM[0], Verdict::Misses),
                (GATE_TURB[1], Verdict::Open),
            ]
        );
    }

    #[test]
    fn leg4_reports_every_gate_open_when_the_key_is_missing_or_shows_no_transition() {
        let mut c = Checks::new();
        check_t3a_leg4_on(&mut c, None, RECORDS, RECORDS_SINGLE).expect("replay");
        assert_eq!((c.total, c.gates.len()), (0, 6));
        assert!(c.gates.iter().all(|g| g.verdict == Verdict::Open));
        let mut c = Checks::new();
        check_t3a_leg4_on(&mut c, Some(&[(1e4, 5e-3), (2e4, 4e-3)]), RECORDS, RECORDS_SINGLE).expect("replay");
        assert_eq!((c.total, c.failures, c.gates.len()), (1, 1, 6), "the key row fails, six gates open");
    }

    /// The recorded runs of 144.3 against the tree's key. With the key
    /// placed, LM's onset and both laminar parts are MISSED, LM's turbulent
    /// C_f and gamma's onset are undecided, and gamma's turbulent C_f holds
    /// (SPEC-LIT 144.4 records them); without it, the six gates are open.
    #[test]
    fn the_recorded_runs_against_the_t3a_key() {
        let mut c = Checks::new();
        check_t3a_leg4(&mut c).expect("replay");
        let got: Vec<(&str, Verdict)> = c.gates.iter().map(|g| (g.gate, g.verdict)).collect();
        if c.total == 0 {
            assert_eq!(got.len(), 6, "key absent: six gates open");
            println!("  [answer-key] ercoftac-t3a absent: leg 4's six gates are open by name");
            return;
        }
        assert_eq!(c.failures, 0);
        assert_eq!((c.total, c.replayed), (4, 3), "key row, gamma's turbulent C_f, two single rows");
        assert_eq!(
            got,
            vec![
                (GATE_ONSET[0], Verdict::Misses),
                (GATE_TURB[0], Verdict::Open),
                (GATE_LAM[0], Verdict::Misses),
                (GATE_ONSET[1], Verdict::Open),
                (GATE_LAM[1], Verdict::Misses),
            ]
        );
        for g in &c.gates {
            let word = if g.verdict == Verdict::Misses { "MISSED" } else { "UNDECIDED" };
            println!("  {word} {}: {}", g.gate, g.headline);
        }
    }

    #[test]
    fn the_harness_runs_a_short_coarse_lm_solve() {
        let Ok(gpu) = Gpu::new(0) else { return };
        let r = run_t3a(&gpu, T3aModel::Lm, LEVELS[2], 200, false).expect("run");
        assert_eq!(r.iterations, 200);
        assert!(r.tu_le > 3.0 && r.tu_le < TU_IN, "{}", r.tu_le);
        assert!(r.cf.iter().all(|v| v.is_finite() && *v > 0.0), "{:?}", r.cf);
    }

    fn record(model: T3aModel, level: usize, max_iters: usize) {
        let gpu = Gpu::new(0).expect("a GPU");
        let r = run_t3a(&gpu, model, LEVELS[level], max_iters, true).expect("run");
        println!("RECORD {}", record_literal(&r));
    }

    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.3 (minutes of GPU time)"]
    fn record_t3a_lm_fine() {
        record(T3aModel::Lm, 0, 40000);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.3 (minutes of GPU time)"]
    fn record_t3a_lm_medium() {
        record(T3aModel::Lm, 1, 40000);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.3 (minutes of GPU time)"]
    fn record_t3a_lm_coarse() {
        record(T3aModel::Lm, 2, 40000);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.3 (minutes of GPU time)"]
    fn record_t3a_gamma_fine() {
        record(T3aModel::Gamma, 0, 40000);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.3 (minutes of GPU time)"]
    fn record_t3a_gamma_medium() {
        record(T3aModel::Gamma, 1, 40000);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.3 (minutes of GPU time)"]
    fn record_t3a_gamma_coarse() {
        record(T3aModel::Gamma, 2, 40000);
    }

    /// The price of a level: `T3A_PRICE=<lm|gamma>,<level>,<iters>` runs that
    /// many iterations with progress lines (SPEC-LIT 144.3's pricing).
    #[test]
    #[ignore = "pricing for SPEC-LIT 144.3"]
    fn price_t3a() {
        let spec = std::env::var("T3A_PRICE").unwrap_or_else(|_| "lm,2,200".to_string());
        let f: Vec<&str> = spec.split(',').collect();
        let model = if f[0] == "gamma" { T3aModel::Gamma } else { T3aModel::Lm };
        let level: usize = f[1].parse().expect("level");
        let iters: usize = f[2].parse().expect("iters");
        record(model, level, iters);
    }
}
