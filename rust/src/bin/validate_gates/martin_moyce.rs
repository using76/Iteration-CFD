// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Gate 145-A (SPEC-LIT §145.2): the dam break of §22, run live through the
//! `ofgpu::vof` library as `ofgpu-vof -surge` runs it, against Martin &
//! Moyce's Table 2 (`reference/martin_moyce_1952/surge_front.csv`, §145.1).
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §145.1 (the key, (145.1)-(145.4)) and §145.2 (this
//!     gate, (145.5)-(145.7)), §20 (the method), §22 (the benchmark row),
//!     §94 (the three-mesh study, (94.1)-(94.10)), §69 (the verdict registry)
//!   J. C. Martin and W. J. Moyce, *Phil. Trans. R. Soc. Lond. A* 244 (1952)
//!     312-324, doi:10.1098/rsta.1952.0006 - Table 2, read through the
//!     answer-key reader with its sha256 printed
//!   A. Ritter, *Z. Ver. Dtsch. Ing.* 36 (1892) 947 - the front speed
//!     2 (g h0)^(1/2) of the frictionless dam break, the analytic bound
//! No GPL-licensed source was consulted.
//!
//! What the gate does, in order: three meshes (`h = a/12, a/18, a/27`) of
//! one 2-D tank; on each the front is sampled every step on the bottom row
//! (the sampler of `bin/vof.rs`), held under the Ritter bound (145.7), and
//! the water volume is held constant; on the finest the front is moved onto
//! the key's time origin (145.5) and compared at every keyed `T` in the
//! window against each printed mean, within the band (145.6); and `Z` at
//! `T = 2` goes through §94's study. A comparison that misses fails its row
//! and registers the verdict with its study (the Gate 105-C pattern).

use std::time::Instant;

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::field::{BcKind, GpuVectorField};
use ofgpu::fv::SnGradScheme;
use ofgpu::io::case::{Preconditioner, SolverControls};
use ofgpu::mesh::{HostMesh, PatchKind};
use ofgpu::vof::{Vof, VofControls, VofProperties};
use ofgpu::vv::{self, Level};
use ofgpu::{Gpu, GpuMesh, Label, Result, Scalar, Vec3};

use crate::{key, vof_scalar_bcs, Checks, GateReport, How, Uncertainty, Verdict};

/// The gate's name: `validate.rs` opens the scope with this constant and
/// [`check_martin_moyce`] reports under it, so the two cannot drift apart.
pub(crate) const GATE: &str = "SPEC-LIT 145.2 Gate 145-A Martin & Moyce surge front";

const AGAINST: &str = "Martin & Moyce (1952) Table 2, n^2 = 2, the printed means of the \
                       1 1/8 in. and 2 1/4 in. series";

/// The answer key's manifest id (VAL-03).
const KEY_ID: &str = "martin-moyce-1952";

/// The column width: 2 1/4 in., the scale of the nine-record series.
const A: f64 = 0.05715;
/// `n^2 = 2`: the column is `2a` high.
const N2: f64 = 2.0;
/// The tank, in column widths: 9a long, 3a high; the column stands against
/// the `x = 0` wall, the plane of symmetry.
const TANK: [f64; 2] = [9.0, 3.0];
/// Cells per column width, finest first (r = 1.5).
const LEVELS: [usize; 3] = [27, 18, 12];
const GMAG: f64 = 9.81;
/// The time step as a fraction of `h / (2 (g h0)^(1/2))`.
const COURANT: f64 = 0.25;
/// The run's end, in the key's `T` (145.1).
const T_END: f64 = 6.0;

/// The key's time origin (§145.1): every record reads `T = 1.19` at `Z = 1.44`.
const Z_ORIGIN: f64 = 1.44;
const T_ORIGIN: f64 = 1.19;
/// (145.4): the key's datum uncertainty, in `T`.
const U_D_T: f64 = 0.17;
/// The band's relative floor in `Z` (145.6).
const Z_FRACTION: f64 = 0.1;
/// The window's last station: two column widths short of the far wall.
const Z_WINDOW_MAX: f64 = 7.0;
/// The study's quantity is `Z` at this `T` (shifted, (145.5)).
const T_STUDY: f64 = 2.0;

/// One printed-mean series of the key, in file order (`Z` and `T` both
/// strictly increase - the transcription guard holds that).
#[derive(Debug, Clone)]
pub(crate) struct Series {
    pub(crate) a_in: f64,
    pub(crate) z: Vec<f64>,
    pub(crate) t: Vec<f64>,
}

/// The secant `dZ/dT` of a series through station `k`'s neighbours, one-sided
/// at either end of the series (145.6).
pub(crate) fn slope(s: &Series, k: usize) -> f64 {
    let lo = k.saturating_sub(1);
    let hi = (k + 1).min(s.z.len() - 1);
    (s.z[hi] - s.z[lo]) / (s.t[hi] - s.t[lo])
}

/// (145.6): `b_k = max(u_D,T |dZ/dT|_k, 0.1 Z_k)`.
pub(crate) fn band(s: &Series, k: usize) -> f64 {
    (U_D_T * slope(s, k).abs()).max(Z_FRACTION * s.z[k])
}

/// The first `T` at which a history of `(T, Z)` samples, ascending in `T`,
/// reaches `z`, linear between the two samples that bracket the crossing.
pub(crate) fn first_time_at(hist: &[(f64, f64)], z: f64) -> Option<f64> {
    let i = hist.iter().position(|&(_, zz)| zz >= z)?;
    if i == 0 {
        return Some(hist[0].0);
    }
    let (t0, z0) = hist[i - 1];
    let (t1, z1) = hist[i];
    Some(t0 + (z - z0) / (z1 - z0) * (t1 - t0))
}

/// `Z` at time `t`, linear between samples; `None` outside the history.
pub(crate) fn z_at(hist: &[(f64, f64)], t: f64) -> Option<f64> {
    let first = hist.first()?;
    if t < first.0 {
        return None;
    }
    let i = hist.iter().position(|&(tt, _)| tt >= t)?;
    if i == 0 {
        return Some(first.1);
    }
    let (t0, z0) = hist[i - 1];
    let (t1, z1) = hist[i];
    Some(z0 + (t - t0) / (t1 - t0) * (z1 - z0))
}

/// (145.5): the shift that puts this history's `Z = 1.44` at `T = 1.19`.
pub(crate) fn origin_shift(hist: &[(f64, f64)]) -> Option<f64> {
    first_time_at(hist, Z_ORIGIN).map(|t| T_ORIGIN - t)
}

/// One mesh level's run.
struct LevelRun {
    n: usize,
    n_cells: usize,
    h: Scalar,
    steps: usize,
    /// `(T_S, Z)` after every step, `T_S` by (145.1)-(145.2), unshifted.
    hist: Vec<(f64, f64)>,
    /// `max(0, z - bound)` over the steps, in column widths (145.7).
    ritter_excess: f64,
    /// The closest approach `min(bound - z)`, in column widths.
    ritter_margin: f64,
    v0: f64,
    v1: f64,
    alpha_lo: f64,
    alpha_hi: f64,
    seconds: f64,
}

/// The tank's no-slip walls and its open top: `U = 0` on every wall,
/// zero-gradient on the generic top, empty front and back.
fn open_top_velocity(gpu: &Gpu, u: &mut GpuVectorField, m: &HostMesh) -> Result<()> {
    let nbf = m.n_boundary_faces;
    let mut kind = vec![BcKind::ZeroGradient as Label; nbf];
    let mut fr = vec![0.0 as Scalar; nbf];
    for pi in &m.patches {
        for f in pi.start..pi.start + pi.size {
            match pi.kind {
                PatchKind::Empty => kind[f] = BcKind::Empty as Label,
                PatchKind::Wall => {
                    kind[f] = BcKind::FixedValue as Label;
                    fr[f] = 1.0;
                }
                _ => {}
            }
        }
    }
    gpu.write(&mut u.bc_kind, &kind)?;
    gpu.write(&mut u.fr, &fr)?;
    gpu.write(&mut u.ref_value, &vec![Vec3::ZERO; nbf])?;
    gpu.write(&mut u.ref_grad, &vec![Vec3::ZERO; nbf])
}

fn run_level(gpu: &Gpu, n: usize) -> Result<LevelRun> {
    let a = A as Scalar;
    let h = a / n as Scalar;
    let axis = |len: f64, cells: usize| GradedAxis {
        lo: 0.0,
        hi: (len * A) as Scalar,
        n: cells,
        expansion: 1.0,
        two_sided: false,
    };
    let (nx, ny) = (TANK[0] as usize * n, TANK[1] as usize * n);
    let m = blockgen::build_mesh(&BlockSpec {
        x: axis(TANK[0], nx),
        y: axis(TANK[1], ny),
        z: GradedAxis { lo: 0.0, hi: h, n: 1, expansion: 1.0, two_sided: false },
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        // The column's wall, the far wall, the floor, the open top, and the
        // two empty faces of a 2-D case.
        patch_type: ["wall", "wall", "wall", "patch", "empty", "empty"].map(String::from),
        cyclic: Vec::new(),
    })?;
    let gm = GpuMesh::upload(gpu, &m)?;

    let props = VofProperties {
        rho1: 998.2,
        rho2: 1.2,
        mu1: 1.002e-3,
        mu2: 1.8e-5,
        sigma: 0.0728,
        // The resolved plane of the block is x-y, so down is -y.
        g: Vec3::new(0.0, -(GMAG as Scalar), 0.0),
        c_alpha: 1.0,
    };
    let h0 = N2 * A;
    let u_ritter = 2.0 * (GMAG * h0).sqrt();
    // The step the solver takes, and the same step widened for the clock, so
    // the host time is the solver's in either build.
    let dt_s = (COURANT * f64::from(h) / u_ritter) as Scalar;
    let dt = f64::from(dt_s);
    let ctrl = VofControls {
        delta_t: dt_s,
        n_correctors: 3,
        u_solver: SolverControls {
            tolerance: 1e-8,
            rel_tol: 0.0,
            max_iter: 200,
            ..SolverControls::default()
        },
        p_solver: SolverControls {
            tolerance: 1e-9,
            rel_tol: 1e-3,
            max_iter: 2000,
            precon: Preconditioner::Dic,
            ..SolverControls::default()
        },
        sn_grad: SnGradScheme::Uncorrected,
        report_continuity: false,
        ..VofControls::default()
    };
    let mut vof = Vof::new(gpu, &m, &gm, props, ctrl)?;

    // The column, in world coordinates (SPEC-LIT 118.7; the block has no
    // local origin, so this is the stored position in either build).
    let a0: Vec<Scalar> = (0..m.n_cells)
        .map(|c| {
            let p = m.world(m.c[c]);
            if f64::from(p.x) < A && f64::from(p.y) < h0 { 1.0 } else { 0.0 }
        })
        .collect();
    gpu.write(&mut vof.alpha_mut().f, &a0)?;
    let top: Vec<Option<Scalar>> = m
        .patches
        .iter()
        .map(|p| if p.kind == PatchKind::Generic { Some(0.0) } else { None })
        .collect();
    vof_scalar_bcs(gpu, vof.alpha_mut(), &m, &vec![None; m.patches.len()])?;
    open_top_velocity(gpu, vof.u_mut(), &m)?;
    vof_scalar_bcs(gpu, vof.p_rgh_mut(), &m, &top)?;
    vof.initialise(gpu)?;

    // The bottom row, ascending in x - `bin/vof.rs`'s sampler on a uniform
    // block: the front is the last cell with alpha >= 1/2, linear to the
    // first one below, measured from the column's wall.
    let mut floor: Vec<(f64, usize)> = (0..m.n_cells)
        .filter_map(|c| {
            let p = m.world(m.c[c]);
            (f64::from(p.y) < f64::from(h)).then_some((f64::from(p.x), c))
        })
        .collect();
    floor.sort_by(|p, q| p.0.total_cmp(&q.0));
    let dx = f64::from(h);
    let wall = floor[0].0 - 0.5 * dx;
    let front = |alpha: &[Scalar]| -> f64 {
        let Some(i) = floor.iter().rposition(|&(_, c)| alpha[c] >= 0.5) else {
            return 0.0;
        };
        if i + 1 >= floor.len() {
            return floor[i].0 - wall;
        }
        let a0 = f64::from(alpha[floor[i].1]);
        let a1 = f64::from(alpha[floor[i + 1].1]);
        let f = if (a0 - a1).abs() > 1e-30 { ((a0 - 0.5) / (a0 - a1)).clamp(0.0, 1.0) } else { 0.0 };
        floor[i].0 + f * dx - wall
    };

    // (145.1)-(145.2): the key's T = n t (g/a)^(1/2), n = 2^(1/2).
    let t_scale = N2.sqrt() * (GMAG / A).sqrt();
    let steps = (T_END / t_scale / dt).ceil() as usize;
    let v0 = f64::from(vof.phase_volume(gpu)?);
    let clock = Instant::now();
    let mut hist = Vec::with_capacity(steps);
    let (mut excess, mut margin) = (0.0f64, f64::INFINITY);
    let (mut lo, mut hi) = (0.0f64, 0.0f64);
    for step in 1..=steps {
        vof.step(gpu, dt_s)?;
        let t = step as f64 * dt;
        let z = front(&gpu.download(&vof.alpha().f)?);
        // (145.7): Ritter's front from the column's edge, plus the one cell
        // the sampler resolves.
        let bound = A + u_ritter * t + dx;
        excess = excess.max((z - bound).max(0.0) / A);
        margin = margin.min((bound - z) / A);
        let (l, u) = vof.alpha_bounds(gpu)?;
        lo = lo.max(-f64::from(l));
        hi = hi.max(f64::from(u) - 1.0);
        hist.push((t * t_scale, z / A));
    }
    let v1 = f64::from(vof.phase_volume(gpu)?);
    Ok(LevelRun {
        n,
        n_cells: m.n_cells,
        h: vv::h_of((TANK[0] * TANK[1] * A * A) as Scalar, m.n_cells, 2)?,
        steps,
        hist,
        ritter_excess: excess,
        ritter_margin: margin,
        v0,
        v1,
        alpha_lo: lo,
        alpha_hi: hi,
        seconds: clock.elapsed().as_secs_f64(),
    })
}

/// The two printed-mean series (`record = 0`), in file order.
fn mean_series(k: &key::KeyFile) -> Result<Vec<Series>> {
    let (a_in, record, z, t) =
        (k.column("a_in")?, k.column("record")?, k.column("Z")?, k.column("T")?);
    let mut out: Vec<Series> = Vec::new();
    for i in 0..k.rows.len() {
        if record[i] != 0.0 {
            continue;
        }
        match out.iter_mut().find(|s| s.a_in == a_in[i]) {
            Some(s) => {
                s.z.push(z[i]);
                s.t.push(t[i]);
            }
            None => out.push(Series { a_in: a_in[i], z: vec![z[i]], t: vec![t[i]] }),
        }
    }
    Ok(out)
}

/// Gate 145-A. Opened by `validate.rs`'s section under [`GATE`].
pub(crate) fn check_martin_moyce(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    let kf = match key::load(KEY_ID) {
        Ok(kf) => kf,
        Err(why) => {
            let why = why.to_string();
            c.note(&why);
            c.skip("Gate 145-A: Z(T) against Martin & Moyce's printed means", "the answer key is absent");
            c.report(GateReport {
                verdict: Verdict::Open,
                how: How::Live,
                gate: GATE,
                against: AGAINST,
                headline: "the answer key is absent from reference/, so no comparison was run".to_string(),
                detail: vec![why],
                uncertainty: Some(Uncertainty::SingleMesh("no mesh was run: the answer key was absent")),
            });
            return Ok(());
        }
    };
    c.note(&kf.digest_line());
    let series = mean_series(&kf)?;

    let mut runs = Vec::with_capacity(LEVELS.len());
    for &n in &LEVELS {
        runs.push(run_level(gpu, n)?);
    }
    // Class R (docs/17 5.1.4): tol32 = max(tol64, min(tol64 2^29, 1e-5)).
    let tol_v = if cfg!(feature = "single") { 1.0e-5 } else { 1.0e-12 };
    for r in &runs {
        c.note(&format!(
            "a/{}: {} cells, h {:.4e} m, {} steps to T = {T_END}, {:.1} s; front at T = {T_END}: Z {:.3}; \
             Ritter margin {:.4} a; alpha excursions {:.2e} below 0, {:.2e} above 1",
            r.n, r.n_cells, f64::from(r.h), r.steps, r.seconds,
            r.hist.last().map_or(0.0, |s| s.1), r.ritter_margin, r.alpha_lo, r.alpha_hi
        ));
        c.check(
            &format!("Gate 145-A a/{}: the front stays under Ritter's bound (145.7)", r.n),
            r.ritter_excess as Scalar,
            0.0,
        );
        c.check(
            &format!("Gate 145-A a/{}: the water volume is conserved", r.n),
            ((r.v1 - r.v0) / r.v0).abs() as Scalar,
            tol_v as Scalar,
        );
    }

    // The comparison, on the finest mesh.
    let fine = &runs[0];
    let shift = origin_shift(&fine.hist);
    c.note(&format!(
        "(145.5) on a/{}: the front reaches Z = {Z_ORIGIN} at T_S = {}, so dT = {}",
        fine.n,
        shift.map_or("never".to_string(), |s| format!("{:.4}", T_ORIGIN - s)),
        shift.map_or("n/a".to_string(), |s| format!("{s:+.4}")),
    ));
    let mut missed: Vec<String> = Vec::new();
    for s in &series {
        let mut worst = 0.0f64;
        let mut at = 0usize;
        c.note(&format!("{} in. series:      T       Z_key    Z_S      band    |dZ|/band", s.a_in));
        for k in 0..s.z.len() {
            if s.z[k] <= Z_ORIGIN || s.z[k] > Z_WINDOW_MAX {
                continue;
            }
            let b = band(s, k);
            let got = shift.and_then(|d| z_at(&fine.hist, s.t[k] - d));
            let ratio = got.map_or(f64::INFINITY, |g| (g - s.z[k]).abs() / b);
            if !(ratio <= worst) {
                worst = ratio;
                at = k;
            }
            c.note(&format!(
                "                {:7.2} {:8.2} {:8} {:8.4} {:9.3}",
                s.t[k],
                s.z[k],
                got.map_or("-".to_string(), |g| format!("{g:.3}")),
                b,
                ratio
            ));
        }
        let what = format!(
            "Gate 145-A a/{}: Z(T) against the {} in. printed mean, worst |dZ|/band (145.6)",
            fine.n, s.a_in
        );
        c.check(&what, worst as Scalar, 1.0);
        if !(worst <= 1.0) {
            missed.push(format!(
                "the {} in. mean: worst |dZ|/band {:.3} at T = {:.2} (Z_key {:.2})",
                s.a_in, worst, s.t[at], s.z[at]
            ));
        }
    }

    // §94: Z at T = 2 (shifted) over the three meshes, finest first.
    let levels: Option<Vec<Level>> = runs
        .iter()
        .map(|r| {
            let d = origin_shift(&r.hist)?;
            Some(Level { h: r.h, value: z_at(&r.hist, T_STUDY - d)? as Scalar })
        })
        .collect();
    let study = levels.as_deref().map(vv::grid_study);
    match (&levels, &study) {
        (Some(l), Some(Ok(st))) => {
            let v: Vec<String> = l.iter().map(|x| format!("{:.4}", f64::from(x.value))).collect();
            c.note(&format!("Z(T = {T_STUDY}) on a/27, a/18, a/12: {}; {}", v.join(", "), st.one_line()));
            // The datum: the 2 1/4 in. mean, linear between its stations
            // either side of T = 2, and u_D = 0.17 |dZ/dT| there.
            if let Some(s) = series.iter().find(|s| s.a_in == 2.25) {
                if let Some(k) = s.t.iter().position(|&t| t >= T_STUDY).filter(|&k| k > 0) {
                    let sl = (s.z[k] - s.z[k - 1]) / (s.t[k] - s.t[k - 1]);
                    let d = s.z[k - 1] + (T_STUDY - s.t[k - 1]) * sl;
                    let val = vv::validation(l[0].value, d as Scalar, st.u_fine, 0.0, (U_D_T * sl) as Scalar);
                    c.note(&val.one_line(&format!("Z at T = {T_STUDY} against the 2 1/4 in. mean")));
                }
            }
        }
        (_, Some(Err(e))) => c.note(&format!("the study was refused by name: {e}")),
        _ => c.note(&format!("the study has no value: a mesh's front never reached Z = {Z_ORIGIN} or T = {T_STUDY}")),
    }
    c.require(
        "Gate 145-A: the three-mesh study of Z at T = 2 is computed (SPEC-LIT 94)",
        matches!(study, Some(Ok(_))),
    );

    if !missed.is_empty() {
        c.report(GateReport {
            verdict: Verdict::Misses,
            how: How::Live,
            gate: GATE,
            against: AGAINST,
            headline: format!(
                "on a/{} the surge front leaves the band (145.6) of {} of the 2 series",
                fine.n,
                missed.len()
            ),
            detail: missed,
            uncertainty: Some(match study {
                Some(Ok(st)) => Uncertainty::Study(st),
                _ => Uncertainty::SingleMesh("the three-mesh study could not be computed on this run"),
            }),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn series_225() -> Series {
        // The 2 1/4 in. printed means around the stations the tests use.
        Series {
            a_in: 2.25,
            z: vec![1.44, 1.67, 1.89, 2.78, 3.00, 3.22, 5.00, 5.89, 7.0],
            t: vec![1.19, 1.43, 1.63, 2.32, 2.51, 2.65, 4.02, 4.44, 5.09],
        }
    }

    #[test]
    fn martin_moyce_band_is_the_larger_of_u_d_and_a_tenth_of_z() {
        let s = series_225();
        // Z = 3.00: neighbours (2.78, 2.32) and (3.22, 2.65).
        assert!((slope(&s, 4) - 0.44 / 0.33).abs() < 1e-12);
        assert!((band(&s, 4) - 0.3).abs() < 1e-12);
        // Z = 1.67: neighbours (1.44, 1.19) and (1.89, 1.63); u_D,Z wins.
        assert!((slope(&s, 1) - 0.45 / 0.44).abs() < 1e-12);
        assert!((band(&s, 1) - 0.17 * 0.45 / 0.44).abs() < 1e-12);
        // Z = 5.89: a tenth of Z wins.
        assert!((band(&s, 7) - 0.589).abs() < 1e-12);
        // The series' end is one-sided: (5.89, 4.44) to (7.0, 5.09).
        assert!((slope(&s, 8) - 1.11 / 0.65).abs() < 1e-12);
    }

    #[test]
    fn martin_moyce_origin_shift_and_interpolation() {
        let hist = [(0.0, 1.0), (1.0, 1.5), (2.0, 2.5)];
        assert!((first_time_at(&hist, 1.44).unwrap() - 0.88).abs() < 1e-12);
        let d = origin_shift(&hist).unwrap();
        assert!((d - 0.31).abs() < 1e-12);
        assert!((z_at(&hist, 2.0 - d).unwrap() - 2.19).abs() < 1e-12);
        // Outside the history there is no value, and a front that never
        // reaches the origin station has no shift.
        assert_eq!(z_at(&hist, 2.5), None);
        assert_eq!(z_at(&hist, -0.1), None);
        assert_eq!(origin_shift(&[(0.0, 1.0), (1.0, 1.2)]), None);
    }
}
