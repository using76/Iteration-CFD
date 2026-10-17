// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gates 88-T and 90-T leg 3: T3B and T3A- on the flat-plate harness, each
//! with a per-case inflow fitted to ERCOFTAC's measured free-stream `Tu(x)`
//! decay (SPEC-LIT §144.7, docs/17 VAL-35).**
//!
//! §88.10's leg 3 ran the T3 series on one rig with only the leading-edge
//! `Tu` changed and measured a 51.9x onset spread against the ~10x the series
//! shows: on a single rig the free-stream decay is as strong a lever on onset
//! as `Tu` itself. Here each case's inflow `omega` is fitted to that case's
//! own measured decay (144.7a), the fit is held to the measurement within
//! 3 % as a stated input check, and the inlet state (144.7c) reproduces the
//! fitted decay from `x_up` ahead of the edge. The runs are long, so they are
//! not made by `ofgpu-validate`: each is an ignored test of this binary
//! (`record_t3fit_*`) that prints a [`T3FitRecord`] literal, and the records
//! below are what the validate section replays (§110.1's split of recipe and
//! verdict), as §144.3's are.
//!
//! Every reduction is f64 whatever `Scalar` is, as §144.3's are.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §144.7 (the fit, the inflow and the rows), §144.3
//!     and §144.4 (the harness and leg 4's definitions, on the log grid
//!     here), §144.6 (the T3A key), §88.10 and §90.10 (the single-rig
//!     construction this retires), §94 (the mesh study), §110.1 (the replay
//!     shape), §32.4 (the band rule), §118.14 (the row classes)
//!   P. E. Roach and D. H. Brierley's T3B and T3A- measurements, read from
//!     the fetched ERCOFTAC keys at run time; no number of them is in this
//!     file
//! No GPL-licensed source was consulted.

use ofgpu::models::KOmegaSstCoeffs;
use ofgpu::vv::{self, GridStudy, Level};
use ofgpu::{Result, Scalar};

use crate::{Band, Checks, GateReport, How, Uncertainty, Verdict};

use super::t3a;

// ---------------------------------------------------------------------------
//  The fit of the measured free-stream decay (144.7a)
// ---------------------------------------------------------------------------

/// §144.7's decay exponent `n = beta_star / (2 beta_2)`, from the crate's own
/// SST constants, as `t3a::tu_closed` reads them: 0.543478... in f64.
pub(crate) fn decay_exponent() -> f64 {
    let c = KOmegaSstCoeffs::default();
    f64::from(c.beta_star) / (2.0 * f64::from(c.beta_2))
}

/// (144.7a)'s fit of one case's measured free-stream decay: `Tu_fit(Re_x) =
/// Tu_LE (1 + b* Re_x)^-n`, with `b*` the Chebyshev point of the stations in
/// ln Tu and `worst` the largest relative miss the fitted decay leaves at a
/// station.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DecayFit {
    pub(crate) b: f64,
    pub(crate) tu_le: f64,
    pub(crate) worst: f64,
}

/// The half-range of `e_i(b) = ln Tu_i + n ln(1 + b Re_x,i)` over the
/// stations: `h(b) = (max_i e_i - min_i e_i)/2`, what (144.7a) minimises.
fn h_of(st: &[(f64, f64)], n: f64, b: f64) -> f64 {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(re, tu) in st {
        let e = tu.ln() + n * (1.0 + b * re).ln();
        lo = lo.min(e);
        hi = hi.max(e);
    }
    (hi - lo) / 2.0
}

/// The mid-range of the same `e_i(b)`; its exponential is (144.7a)'s Tu_LE.
fn m_of(st: &[(f64, f64)], n: f64, b: f64) -> f64 {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(re, tu) in st {
        let e = tu.ln() + n * (1.0 + b * re).ln();
        lo = lo.min(e);
        hi = hi.max(e);
    }
    (hi + lo) / 2.0
}

/// (144.7a) exactly: `b* = argmin_{b > 0} h(b)`, `Tu_LE = exp(m(b*))`, on the
/// stations `(Re_x, Tu %)` in file order. The search is the stated
/// deterministic one: a 4001-point scan of `ln b` over `[ln 1e-9, ln 1e-2]`,
/// then 200 golden-section steps inside the least-`h` point's two neighbours,
/// and `b*` at the interval's midpoint. Refuses fewer than three stations,
/// or any non-finite or non-positive station value.
pub(crate) fn fit_decay(st: &[(f64, f64)]) -> std::result::Result<DecayFit, String> {
    if st.len() < 3 {
        return Err(format!(
            "the fit of the measured Tu(x) needs three stations at least, got {}",
            st.len()
        ));
    }
    for &(re, tu) in st {
        if !re.is_finite() || re <= 0.0 {
            return Err(format!("the station Re_x = {re} is not a positive finite number"));
        }
        if !tu.is_finite() || tu <= 0.0 {
            return Err(format!("the station Tu = {tu} % is not a positive finite number"));
        }
    }
    let n = decay_exponent();
    let (l9, l2) = (1e-9f64.ln(), 1e-2f64.ln());
    let step = (l2 - l9) / 4000.0;
    let mut j_best = 0usize;
    let mut h_best = h_of(st, n, l9.exp());
    for j in 1..=4000usize {
        let h_j = h_of(st, n, (l9 + j as f64 * step).exp());
        if h_j < h_best {
            h_best = h_j;
            j_best = j;
        }
    }
    let mut lo = l9 + j_best.saturating_sub(1) as f64 * step;
    let mut hi = l9 + (j_best + 1).min(4000) as f64 * step;
    let phi = (5.0f64.sqrt() - 1.0) / 2.0;
    for _ in 0..200 {
        let a = hi - phi * (hi - lo);
        let c = lo + phi * (hi - lo);
        if h_of(st, n, a.exp()) < h_of(st, n, c.exp()) {
            hi = c;
        } else {
            lo = a;
        }
    }
    let b = ((lo + hi) / 2.0).exp();
    let tu_le = m_of(st, n, b).exp();
    let f = DecayFit { b, tu_le, worst: 0.0 };
    let worst = st
        .iter()
        .map(|&(re, tu)| (tu_fit(&f, re) / tu - 1.0).abs())
        .fold(0.0f64, f64::max);
    Ok(DecayFit { b, tu_le, worst })
}

/// (144.7a)'s fitted decay at `Re_x`, in %.
pub(crate) fn tu_fit(f: &DecayFit, re_x: f64) -> f64 {
    f.tu_le * (1.0 + f.b * re_x).powf(-decay_exponent())
}

/// (144.7c)'s inflow state: what the harness inlet carries, `x_up` ahead of
/// the edge, so that §88.10's closed form from it reproduces (144.7a)'s
/// fitted decay at every `x >= x_up`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Inflow {
    /// %, as `t3a::RigInputs::tu_in` takes it.
    pub(crate) tu_in: f64,
    /// `m^2/s^2`.
    pub(crate) k0: f64,
    /// `1/s`.
    pub(crate) omega0: f64,
    /// `s = b* Re_up`: the fitted decay's fraction consumed crossing `x_up`.
    pub(crate) s: f64,
}

/// (144.7c) at free stream `u` and viscosity `nu`, `re_up = u x_up/nu`:
/// `Tu_in = Tu_LE (1 - s)^-n`, `k_in = 3/2 (Tu_in U/100)^2`, `omega_in =
/// b* U^2/(beta_2 nu (1 - s))`. Refuses `s >= 1`: past the pole the closed
/// form has no inflow state to hand a solver.
pub(crate) fn inflow_from_fit(
    f: &DecayFit,
    re_up: f64,
    nu: f64,
    u: f64,
) -> std::result::Result<Inflow, String> {
    let n = decay_exponent();
    let beta2 = f64::from(KOmegaSstCoeffs::default().beta_2);
    let s = f.b * re_up;
    if !s.is_finite() || s >= 1.0 {
        return Err(format!("the inlet sits at or past the fitted decay's pole: s = {s} >= 1"));
    }
    let tu_in = f.tu_le * (1.0 - s).powf(-n);
    let k_in = 1.5 * (tu_in * u / 100.0) * (tu_in * u / 100.0);
    let omega_in = f.b * u * u / (beta2 * nu * (1.0 - s));
    Ok(Inflow {
        tu_in,
        k0: k_in,
        omega0: omega_in,
        s,
    })
}

// ---------------------------------------------------------------------------
//  The three cases (144.7's rig table)
// ---------------------------------------------------------------------------

/// The T3 series' three cases, in every array's index order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum T3Case {
    /// ERCOFTAC T3B, the high-turbulence plate (key `ercoftac-t3b`).
    T3b,
    /// ERCOFTAC T3A-, the low-turbulence plate (key `ercoftac-t3am`).
    T3am,
    /// The T3A rig refitted from its own key (`ercoftac-t3a`), so the
    /// ordering compares one inflow method; §144.3's TMR-inflow records are
    /// its fallback until a run is recorded.
    T3aFit,
}

impl T3Case {
    pub(crate) fn name(self) -> &'static str {
        match self {
            T3Case::T3b => "T3B",
            T3Case::T3am => "T3A-",
            T3Case::T3aFit => "T3A",
        }
    }
}

pub(crate) const CASES: [T3Case; 3] = [T3Case::T3b, T3Case::T3am, T3Case::T3aFit];

/// Every rig runs the harness free stream: `U = 1 m/s` (§144.3's).
pub(crate) const U: f64 = t3a::U_INF;
/// The leading edge's distance from the inlet, the same on the three rigs.
pub(crate) const X_UP: f64 = 0.25;
/// The plate's length, `L = 3 m`, as §144.3's.
pub(crate) const LENGTH: f64 = 3.0;

/// The rig's viscosity: T3B `3.0e-6 m^2/s` (`Re_L = 1e6`), T3A- `1.25e-6`
/// (`Re_L = 2.4e6`), T3A §144.3's `5.0e-6` (`Re_L = 6e5`).
pub(crate) fn rig_nu(case: T3Case) -> f64 {
    match case {
        T3Case::T3b => 3.0e-6,
        T3Case::T3am => 1.25e-6,
        T3Case::T3aFit => t3a::NU,
    }
}

/// The rig's box height: `0.25 m` for T3B and T3A- (wall cells half
/// §144.3's), `0.5 m` for the refitted T3A.
pub(crate) fn rig_height(case: T3Case) -> f64 {
    match case {
        T3Case::T3b | T3Case::T3am => 0.25,
        T3Case::T3aFit => 0.5,
    }
}

/// The key a case is held against: its manifest id and its file, for the
/// digest and the missing-key lines (§110.1).
pub(crate) fn key_file(case: T3Case) -> (&'static str, &'static str) {
    match case {
        T3Case::T3b => (crate::key::T3B_ID, "ercoftac_t3a/t3by.dat"),
        T3Case::T3am => (crate::key::T3AM_ID, "ercoftac_t3a/t3amy.dat"),
        T3Case::T3aFit => (crate::key::T3A_ID, "ercoftac_t3a/t3ay.dat"),
    }
}

/// The case's key through the manifest: [`crate::key::ercoftac_t3b`],
/// [`crate::key::ercoftac_t3am`] or [`crate::key::ercoftac_t3a`].
pub(crate) fn load_key(case: T3Case) -> Result<crate::key::T3a> {
    match case {
        T3Case::T3b => crate::key::ercoftac_t3b(),
        T3Case::T3am => crate::key::ercoftac_t3am(),
        T3Case::T3aFit => crate::key::ercoftac_t3a(),
    }
}

// ---------------------------------------------------------------------------
//  The record grid
// ---------------------------------------------------------------------------

/// `C_f` is recorded at `Re_x = re_log(i)`: `N_LOG` points log-spaced over
/// the three cases' span, which a power law interpolates exactly (144.4b).
pub(crate) const N_LOG: usize = 49;
pub(crate) const RE_LOG_LO: f64 = 1.0e4;
pub(crate) const RE_LOG_HI: f64 = 2.4e6;

/// The grid's `i`-th `Re_x`.
pub(crate) fn re_log(i: usize) -> f64 {
    RE_LOG_LO * (RE_LOG_HI / RE_LOG_LO).powf(i as f64 / (N_LOG - 1) as f64)
}

/// A record's `C_f` at `re_x`: the power law through the two grid points
/// bracketing it, (144.4b) on the log grid. NaN outside the grid, and NaN
/// when a bracketing value is NaN - which is how a station the run never
/// reached fails its row.
pub(crate) fn cf_at_log(r: &T3FitRecord, re_x: f64) -> f64 {
    for k in 0..N_LOG - 1 {
        let (g0, g1) = (re_log(k), re_log(k + 1));
        if re_x >= g0 && re_x <= g1 {
            let t = (re_x / g0).ln() / (g1 / g0).ln();
            return (r.cf[k].ln() + t * (r.cf[k + 1].ln() - r.cf[k].ln())).exp();
        }
    }
    f64::NAN
}

/// What one recorded run of one case leaves behind - the replay slot's type.
/// `tu_in` and `omega_in` are the inflow the run used, so the replay can hold
/// every record to its own key's fitted inflow.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // RECORDED fields: a run fills the whole record; the row reads what it reads
pub(crate) struct T3FitRecord {
    pub(crate) case: T3Case,
    pub(crate) model: t3a::T3aModel,
    pub(crate) cells: (usize, usize, usize),
    pub(crate) n_cells: usize,
    /// (94.1) in 2-D: `sqrt(A/N)`, `A` the x-y area.
    pub(crate) h: f64,
    pub(crate) iterations: usize,
    pub(crate) converged: bool,
    pub(crate) last_change: f64,
    /// The inflow the run used: (144.7c) from its own key's fit.
    pub(crate) tu_in: f64,
    pub(crate) omega_in: f64,
    /// `Tu` at the leading edge (%), (144.3b), on the row nearest `H/2`.
    pub(crate) tu_le: f64,
    pub(crate) tu_le_min: f64,
    pub(crate) tu_le_max: f64,
    /// `C_f` at `re_log(i)`, linear in `x` between the two faces
    /// bracketing it; NaN off the plate.
    pub(crate) cf: [f64; N_LOG],
    /// The plate face of least `C_f` in the band: its `Re_x` and `C_f`.
    pub(crate) cf_min_re_x: f64,
    pub(crate) cf_min: f64,
    pub(crate) seconds: f64,
    /// `"f64"` or `"single"`, and the commit the binary was built from.
    pub(crate) build: &'static str,
    pub(crate) commit: &'static str,
}

/// The replay slots: `[case][model][level]`, cases in [`CASES`] order, LM
/// first, finest level first. `None` is a run not recorded yet: run 1 of
/// docs/17 VAL-35 ships the recipe only, and the supervisor's runs fill the
/// slots. `RECORDS` is the f64 build's, the gates'; `RECORDS_SINGLE` is the
/// same eighteen runs under `--features single`.
pub(crate) const RECORDS: [[[Option<&T3FitRecord>; 3]; 2]; 3] = [
    [
        [Some(&F64_T3B_LM_FINE), Some(&F64_T3B_LM_MEDIUM), Some(&F64_T3B_LM_COARSE)],
        [None, Some(&F64_T3B_GAMMA_MEDIUM), Some(&F64_T3B_GAMMA_COARSE)],
    ],
    [
        [Some(&F64_T3AM_LM_FINE), Some(&F64_T3AM_LM_MEDIUM), Some(&F64_T3AM_LM_COARSE)],
        [None, Some(&F64_T3AM_GAMMA_MEDIUM), None],
    ],
    [[None; 3], [None; 3]],
];
pub(crate) const RECORDS_SINGLE: [[[Option<&T3FitRecord>; 3]; 2]; 3] = [[[None; 3]; 2]; 3];

// The f64 records, SPEC-LIT 144.7's table: one RTX 5070 Ti, 2026-10-17,
// printed by the record_t3fit_* tests (the literals as printed).
#[rustfmt::skip]
const F64_T3B_LM_FINE: T3FitRecord = T3FitRecord { case: T3Case::T3b, model: t3a::T3aModel::Lm, cells: (48, 256, 128), n_cells: 38912, h: 4.569513e-3, iterations: 12000, converged: false, last_change: 5.184532e-4, tu_in: 7.986061e0, omega_in: 3.133420e1, tu_le: 6.087064, tu_le_min: 6.086198, tu_le_max: 6.087295, cf: [8.190284e-3, 7.826950e-3, 7.483804e-3, 7.157168e-3, 6.853559e-3, 6.568565e-3, 6.307236e-3, 6.067108e-3, 5.853301e-3, 5.669891e-3, 5.517476e-3, 5.395095e-3, 5.298171e-3, 5.226181e-3, 5.206843e-3, 5.224119e-3, 5.284577e-3, 5.373852e-3, 5.490667e-3, 5.599187e-3, 5.667960e-3, 5.659105e-3, 5.603295e-3, 5.545603e-3, 5.459322e-3, 5.358164e-3, 5.239879e-3, 5.130969e-3, 5.024592e-3, 4.917546e-3, 4.810925e-3, 4.704648e-3, 4.598695e-3, 4.486071e-3, 4.380912e-3, 4.284953e-3, 4.190239e-3, 4.097130e-3, 4.005389e-3, 3.905009e-3, 3.820215e-3, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN], cf_min_re_x: 9.950110e5, cf_min: 3.790129e-3, seconds: 2665.2, build: "f64", commit: "" };
#[rustfmt::skip]
const F64_T3B_LM_MEDIUM: T3FitRecord = T3FitRecord { case: T3Case::T3b, model: t3a::T3aModel::Lm, cells: (24, 128, 64), n_cells: 9728, h: 9.139026e-3, iterations: 8000, converged: false, last_change: 1.066074e-5, tu_in: 7.986061e0, omega_in: 3.133420e1, tu_le: 6.086961, tu_le_min: 6.086085, tu_le_max: 6.087202, cf: [8.060876e-3, 7.703610e-3, 7.384649e-3, 7.067005e-3, 6.769679e-3, 6.499830e-3, 6.243020e-3, 6.003416e-3, 5.784499e-3, 5.592732e-3, 5.432913e-3, 5.299393e-3, 5.197065e-3, 5.118106e-3, 5.059268e-3, 5.070499e-3, 5.149772e-3, 5.215680e-3, 5.339822e-3, 5.460639e-3, 5.491734e-3, 5.558554e-3, 5.538275e-3, 5.467741e-3, 5.379842e-3, 5.283411e-3, 5.182566e-3, 5.079636e-3, 4.975613e-3, 4.868797e-3, 4.761473e-3, 4.648115e-3, 4.514651e-3, 4.417354e-3, 4.322199e-3, 4.228825e-3, 4.137368e-3, 4.047850e-3, 3.960454e-3, 3.874402e-3, 3.791378e-3, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN], cf_min_re_x: 9.900381e5, cf_min: 3.760870e-3, seconds: 1186.8, build: "f64", commit: "" };
#[rustfmt::skip]
const F64_T3B_LM_COARSE: T3FitRecord = T3FitRecord { case: T3Case::T3b, model: t3a::T3aModel::Lm, cells: (12, 64, 32), n_cells: 2432, h: 1.827805e-2, iterations: 2800, converged: true, last_change: 2.645663e-7, tu_in: 7.986061e0, omega_in: 3.133420e1, tu_le: 6.086654, tu_le_min: 6.085816, tu_le_max: 6.086893, cf: [8.031564e-3, 7.703437e-3, 7.417191e-3, 7.096323e-3, 6.845585e-3, 6.614594e-3, 6.389503e-3, 6.208389e-3, 6.039818e-3, 5.902496e-3, 5.800062e-3, 5.716950e-3, 5.657716e-3, 5.619492e-3, 5.591710e-3, 5.568693e-3, 5.547607e-3, 5.523763e-3, 5.493579e-3, 5.455014e-3, 5.405853e-3, 5.346589e-3, 5.278189e-3, 5.202248e-3, 5.120621e-3, 5.034944e-3, 4.946522e-3, 4.856351e-3, 4.764344e-3, 4.670549e-3, 4.575330e-3, 4.480404e-3, 4.386769e-3, 4.294050e-3, 4.203012e-3, 4.113443e-3, 4.026200e-3, 3.940496e-3, 3.856738e-3, 3.774853e-3, 3.689013e-3, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN], cf_min_re_x: 9.801408e5, cf_min: 3.668529e-3, seconds: 286.2, build: "f64", commit: "" };
#[rustfmt::skip]
const F64_T3B_GAMMA_MEDIUM: T3FitRecord = T3FitRecord { case: T3Case::T3b, model: t3a::T3aModel::Gamma, cells: (24, 128, 64), n_cells: 9728, h: 9.139026e-3, iterations: 8000, converged: false, last_change: 1.712461e-5, tu_in: 7.986061e0, omega_in: 3.133420e1, tu_le: 6.086964, tu_le_min: 6.085657, tu_le_max: 6.087205, cf: [8.063146e-3, 7.705557e-3, 7.384163e-3, 7.062567e-3, 6.758773e-3, 6.481095e-3, 6.216204e-3, 5.967173e-3, 5.733678e-3, 5.516425e-3, 5.316437e-3, 5.130433e-3, 4.967452e-3, 4.824728e-3, 4.705123e-3, 4.611063e-3, 4.545918e-3, 4.526248e-3, 4.569610e-3, 4.647642e-3, 4.757952e-3, 4.936444e-3, 5.083116e-3, 5.202667e-3, 5.220128e-3, 5.173459e-3, 5.099004e-3, 5.011839e-3, 4.918498e-3, 4.821656e-3, 4.722188e-3, 4.622210e-3, 4.522066e-3, 4.422807e-3, 4.325178e-3, 4.229392e-3, 4.135562e-3, 4.043439e-3, 3.952621e-3, 3.859908e-3, 3.762800e-3, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN], cf_min_re_x: 9.900381e5, cf_min: 3.732732e-3, seconds: 731.2, build: "f64", commit: "" };
#[rustfmt::skip]
const F64_T3B_GAMMA_COARSE: T3FitRecord = T3FitRecord { case: T3Case::T3b, model: t3a::T3aModel::Gamma, cells: (12, 64, 32), n_cells: 2432, h: 1.827805e-2, iterations: 2700, converged: true, last_change: 3.776349e-7, tu_in: 7.986061e0, omega_in: 3.133420e1, tu_le: 6.086656, tu_le_min: 6.085509, tu_le_max: 6.086898, cf: [8.048816e-3, 7.714191e-3, 7.418665e-3, 7.087395e-3, 6.815855e-3, 6.557345e-3, 6.298007e-3, 6.071349e-3, 5.849973e-3, 5.650832e-3, 5.481540e-3, 5.327864e-3, 5.197911e-3, 5.095638e-3, 5.019418e-3, 4.964608e-3, 4.932246e-3, 4.921856e-3, 4.931682e-3, 4.961718e-3, 5.011013e-3, 5.050148e-3, 5.142834e-3, 5.195584e-3, 5.166172e-3, 5.096821e-3, 5.011652e-3, 4.918031e-3, 4.821519e-3, 4.723781e-3, 4.624360e-3, 4.525042e-3, 4.426645e-3, 4.329644e-3, 4.233669e-3, 4.138495e-3, 4.027214e-3, 3.909992e-3, 3.827493e-3, 3.746782e-3, 3.661917e-3, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN], cf_min_re_x: 9.801408e5, cf_min: 3.641607e-3, seconds: 259.6, build: "f64", commit: "" };
#[rustfmt::skip]
const F64_T3AM_LM_FINE: T3FitRecord = T3FitRecord { case: T3Case::T3am, model: t3a::T3aModel::Lm, cells: (48, 256, 128), n_cells: 38912, h: 4.569513e-3, iterations: 12000, converged: false, last_change: 4.816424e-6, tu_in: 1.063466e0, omega_in: 1.594405e1, tu_le: 0.910754, tu_le_min: 0.910707, tu_le_max: 0.910766, cf: [6.568710e-3, 6.188098e-3, 5.867981e-3, 5.546557e-3, 5.243932e-3, 4.968969e-3, 4.702451e-3, 4.449025e-3, 4.209011e-3, 3.981259e-3, 3.764148e-3, 3.561220e-3, 3.367084e-3, 3.184609e-3, 3.011091e-3, 2.846749e-3, 2.690871e-3, 2.543987e-3, 2.404520e-3, 2.272706e-3, 2.148033e-3, 2.029858e-3, 1.918226e-3, 1.812640e-3, 1.712802e-3, 1.618340e-3, 1.528996e-3, 1.444533e-3, 1.364708e-3, 1.289314e-3, 1.218052e-3, 1.150808e-3, 1.087387e-3, 1.027635e-3, 9.715187e-4, 9.206678e-4, 8.770315e-4, 8.428164e-4, 8.195226e-4, 8.078413e-4, 7.964145e-4, 7.795390e-4, 7.700175e-4, 7.904531e-4, 9.822232e-4, 2.384666e-3, 3.554285e-3, 3.474622e-3, f64::NAN], cf_min_re_x: 1.210603e6, cf_min: 7.700106e-4, seconds: 1240.2, build: "f64", commit: "" };
#[rustfmt::skip]
const F64_T3AM_LM_MEDIUM: T3FitRecord = T3FitRecord { case: T3Case::T3am, model: t3a::T3aModel::Lm, cells: (24, 128, 64), n_cells: 9728, h: 9.139026e-3, iterations: 4000, converged: true, last_change: 2.480563e-7, tu_in: 1.063466e0, omega_in: 1.594405e1, tu_le: 0.910755, tu_le_min: 0.910709, tu_le_max: 0.910768, cf: [6.890453e-3, 6.334961e-3, 5.828947e-3, 5.538269e-3, 5.212434e-3, 4.891697e-3, 4.650949e-3, 4.381408e-3, 4.166359e-3, 3.935421e-3, 3.730141e-3, 3.531257e-3, 3.339859e-3, 3.159067e-3, 2.988799e-3, 2.828157e-3, 2.676080e-3, 2.531398e-3, 2.392919e-3, 2.263467e-3, 2.139407e-3, 2.022722e-3, 1.912180e-3, 1.807304e-3, 1.708270e-3, 1.614574e-3, 1.525678e-3, 1.441580e-3, 1.362216e-3, 1.287155e-3, 1.216314e-3, 1.149246e-3, 1.086054e-3, 1.026465e-3, 9.704777e-4, 9.195195e-4, 8.752903e-4, 8.401188e-4, 8.151143e-4, 8.039529e-4, 7.966832e-4, 7.850254e-4, 7.779219e-4, 8.028874e-4, 1.120381e-3, 2.740493e-3, 3.475068e-3, 3.385326e-3, f64::NAN], cf_min_re_x: 1.215083e6, cf_min: 7.778465e-4, seconds: 662.6, build: "f64", commit: "" };
#[rustfmt::skip]
const F64_T3AM_LM_COARSE: T3FitRecord = T3FitRecord { case: T3Case::T3am, model: t3a::T3aModel::Lm, cells: (12, 64, 32), n_cells: 2432, h: 1.827805e-2, iterations: 1200, converged: true, last_change: 2.727737e-7, tu_in: 1.063466e0, omega_in: 1.594405e1, tu_le: 0.910762, tu_le_min: 0.910717, tu_le_max: 0.910775, cf: [9.986536e-3, 8.834033e-3, 7.542131e-3, 6.093968e-3, 5.518098e-3, 5.222139e-3, 4.890382e-3, 4.518498e-3, 4.135112e-3, 3.932437e-3, 3.705247e-3, 3.454023e-3, 3.291248e-3, 3.108784e-3, 2.947396e-3, 2.789256e-3, 2.644601e-3, 2.498281e-3, 2.366596e-3, 2.240179e-3, 2.118981e-3, 2.004154e-3, 1.895610e-3, 1.792981e-3, 1.695776e-3, 1.603399e-3, 1.515247e-3, 1.433252e-3, 1.354675e-3, 1.280822e-3, 1.210338e-3, 1.144214e-3, 1.081684e-3, 1.022592e-3, 9.669105e-4, 9.154108e-4, 8.699300e-4, 8.331675e-4, 8.101973e-4, 8.014117e-4, 7.920707e-4, 7.871438e-4, 8.063872e-4, 9.403215e-4, 2.176715e-3, 3.346514e-3, 3.376865e-3, 3.223189e-3, f64::NAN], cf_min_re_x: 1.046459e6, cf_min: 7.868235e-4, seconds: 129.8, build: "f64", commit: "" };
#[rustfmt::skip]
const F64_T3AM_GAMMA_MEDIUM: T3FitRecord = T3FitRecord { case: T3Case::T3am, model: t3a::T3aModel::Gamma, cells: (24, 128, 64), n_cells: 9728, h: 9.139026e-3, iterations: 1600, converged: true, last_change: 7.682736e-7, tu_in: 1.063466e0, omega_in: 1.594405e1, tu_le: 0.910754, tu_le_min: 0.910667, tu_le_max: 0.910768, cf: [6.886618e-3, 6.331137e-3, 5.825153e-3, 5.534547e-3, 5.208792e-3, 4.888162e-3, 4.647578e-3, 4.378222e-3, 4.163350e-3, 3.932603e-3, 3.727509e-3, 3.528817e-3, 3.337616e-3, 3.157020e-3, 2.986943e-3, 2.826482e-3, 2.674574e-3, 2.530049e-3, 2.391717e-3, 2.262399e-3, 2.138463e-3, 2.021889e-3, 1.911445e-3, 1.806657e-3, 1.707700e-3, 1.614072e-3, 1.525237e-3, 1.441193e-3, 1.361870e-3, 1.286837e-3, 1.216006e-3, 1.148917e-3, 1.085644e-3, 1.025826e-3, 9.693410e-4, 9.160085e-4, 8.656326e-4, 8.182035e-4, 7.735603e-4, 7.320270e-4, 6.943345e-4, 6.646925e-4, 6.320105e-4, 3.577226e-3, 3.637881e-3, 3.542915e-3, 3.432491e-3, 3.326914e-3, f64::NAN], cf_min_re_x: 1.215083e6, cf_min: 6.240123e-4, seconds: 258.2, build: "f64", commit: "" };

// ---------------------------------------------------------------------------
//  A record's three rows: 144.4's definitions on the log grid
// ---------------------------------------------------------------------------

/// One record's three leg-3 numbers, `t3a::leg4_rows`'s on the log grid,
/// plus the grid index of the worst laminar point (its study reads it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Leg3Rows {
    /// The record's least-`C_f` face against (144.4a), or (144.7e)'s grid
    /// reading of it.
    pub(crate) onset: (f64, f64),
    /// How that onset was read, (144.7e).
    pub(crate) onset_kind: OnsetKind,
    /// The worst turbulent station, against its measured `C_f`.
    pub(crate) turbulent: (f64, f64),
    /// The worst laminar point, (144.4c).
    pub(crate) laminar: (f64, f64),
    /// That point's grid index; `None` when the set is empty, which leaves
    /// the row NaN and fails it.
    pub(crate) lam_i: Option<usize>,
}

/// (144.4c) at the record's own grid point `i`.
fn lam_excess(r: &T3FitRecord, i: usize) -> f64 {
    r.cf[i] * re_log(i).sqrt() / t3a::BLASIUS_CF - 1.0
}

/// How a record's onset was read, (144.7e).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OnsetKind {
    /// The least-`C_f` face, §144.4's own reading.
    LeastFace,
    /// The first grid minimum, the least face being the trailing edge.
    GridMinimum,
    /// `C_f` falls over the whole band: onset upstream of `2e4`, read as
    /// that bound.
    Upstream,
}

/// The record's onset (144.7e):
///
/// ```text
/// onset_rec = cf_min_re_x                    if cf_min_re_x <= Re_log(last finite)
///           = vertex (144.4a) through the first local minimum of the
///             record's grid C_f in [2e4, Re_L]          otherwise, if one exists
///           = 2e4 (an upper bound)                     otherwise
/// ```
///
/// §144.4's least face is read unchanged while it lies inside the recorded
/// grid (the T3A- records). On T3B's plate the turbulent `C_f` decays below
/// its transition minimum, so the least face is the trailing edge and the
/// onset is the vertex through the first local minimum of the grid `C_f`
/// instead; when the grid `C_f` falls over the whole band, transition lies
/// upstream of `2e4` and the onset reads as that bound. A record with no
/// finite grid point at all keeps §144.4's least face, which is NaN-safe.
pub(crate) fn record_onset(r: &T3FitRecord) -> (f64, OnsetKind) {
    let Some(last) = (0..N_LOG).rev().find(|&i| r.cf[i].is_finite()) else {
        return (r.cf_min_re_x, OnsetKind::LeastFace);
    };
    if r.cf_min_re_x <= re_log(last) {
        return (r.cf_min_re_x, OnsetKind::LeastFace);
    }
    for i in 1..last {
        if re_log(i) >= t3a::RE_X_LAM_LO
            && r.cf[i - 1].is_finite()
            && r.cf[i - 1] > r.cf[i]
            && r.cf[i] <= r.cf[i + 1]
        {
            let v = t3a::vertex(
                (re_log(i - 1), r.cf[i - 1]),
                (re_log(i), r.cf[i]),
                (re_log(i + 1), r.cf[i + 1]),
            );
            return (v, OnsetKind::GridMinimum);
        }
    }
    (t3a::RE_X_LAM_LO, OnsetKind::Upstream)
}

/// (144.7e)'s phrase for how the onset was read, as the per-record note and
/// the onset row's headline name it; empty for the least face, §144.4's own
/// reading.
fn onset_kind_note(kind: OnsetKind) -> &'static str {
    match kind {
        OnsetKind::LeastFace => "",
        OnsetKind::GridMinimum => {
            "; onset read at the first grid minimum (144.7e), the least face being the trailing edge"
        }
        OnsetKind::Upstream => {
            "; C_f falls over the whole band: onset upstream of 2e4, read as that bound (144.7e)"
        }
    }
}

/// The record's grid points in the laminar part: from §144.2's `2e4` to the
/// key's `laminar_hi`, and upstream of the record's own onset (144.7e).
fn laminar_points(onset: f64, k: &t3a::Leg4Key) -> Vec<usize> {
    (0..N_LOG)
        .filter(|&i| {
            let g = re_log(i);
            g >= t3a::RE_X_LAM_LO && g <= k.laminar_hi && g < onset
        })
        .collect()
}

pub(crate) fn leg3_rows(r: &T3FitRecord, k: &t3a::Leg4Key) -> Leg3Rows {
    let (onset_re, onset_kind) = record_onset(r);
    let lam = laminar_points(onset_re, k);
    let (lam_re, lam_e) = t3a::worst(lam.iter().copied().map(|i| (re_log(i), lam_excess(r, i))));
    Leg3Rows {
        onset: (onset_re, onset_re / k.onset_re_x - 1.0),
        onset_kind,
        turbulent: t3a::worst(
            k.turbulent
                .iter()
                .map(|&(re, cf)| (re, cf_at_log(r, re) / cf - 1.0)),
        ),
        laminar: (lam_re, lam_e),
        lam_i: (0..N_LOG).find(|&i| re_log(i) == lam_re),
    }
}

/// The larger of two magnitudes, NaN when either is: a stale-input row must
/// fail on a NaN, which `f64::max` would drop.
fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// §94's study of one quantity over one (case, model)'s three levels.
fn study_of(
    recs: &[Option<&T3FitRecord>; 3],
    f: impl Fn(&T3FitRecord) -> f64,
) -> std::result::Result<GridStudy, String> {
    if !recs.iter().all(Option::is_some) {
        return Err("fewer than three levels are recorded".to_string());
    }
    let levels: Vec<Level> =
        recs.iter().flatten().map(|r| Level { h: r.h as Scalar, value: f(r) as Scalar }).collect();
    vv::grid_study(&levels).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
//  The gates
// ---------------------------------------------------------------------------

/// The fit's input check, one per case.
pub(crate) const GATE_FIT: [&str; 3] = [
    "SPEC-LIT 144.7 the T3B inflow fit against ERCOFTAC's measured free-stream Tu(x)",
    "SPEC-LIT 144.7 the T3A- inflow fit against ERCOFTAC's measured free-stream Tu(x)",
    "SPEC-LIT 144.7 the T3A inflow fit against ERCOFTAC's measured free-stream Tu(x)",
];

/// The three §144.4 rows, per case and model (Gate 88-T for LM, 90-T for
/// gamma).
pub(crate) const GATE_ONSET: [[&str; 2]; 3] = [
    [
        "SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3B onset Re_x against Roach & Brierley",
        "SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3B onset Re_x against Roach & Brierley",
    ],
    [
        "SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3A- onset Re_x against Roach & Brierley",
        "SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3A- onset Re_x against Roach & Brierley",
    ],
    [
        "SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3A onset Re_x against Roach & Brierley",
        "SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3A onset Re_x against Roach & Brierley",
    ],
];
pub(crate) const GATE_TURB: [[&str; 2]; 3] = [
    [
        "SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3B turbulent C_f against Roach & Brierley",
        "SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3B turbulent C_f against Roach & Brierley",
    ],
    [
        "SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3A- turbulent C_f against Roach & Brierley",
        "SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3A- turbulent C_f against Roach & Brierley",
    ],
    [
        "SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3A turbulent C_f against Roach & Brierley",
        "SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3A turbulent C_f against Roach & Brierley",
    ],
];
pub(crate) const GATE_LAM: [[&str; 2]; 3] = [
    [
        "SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3B laminar C_f against Blasius",
        "SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3B laminar C_f against Blasius",
    ],
    [
        "SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3A- laminar C_f against Blasius",
        "SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3A- laminar C_f against Blasius",
    ],
    [
        "SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, T3A laminar C_f against Blasius",
        "SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, T3A laminar C_f against Blasius",
    ],
];
/// The onset ordering and the spread, per model.
pub(crate) const GATE_ORDER: [&str; 2] = [
    "SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, onset ordering T3B < T3A < T3A- against Roach & Brierley",
    "SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, onset ordering T3B < T3A < T3A- against Roach & Brierley",
];
pub(crate) const GATE_SPREAD: [&str; 2] = [
    "SPEC-LIT 144.7 Gate 88-T leg 3, kOmegaSSTLM, onset spread Re_x(T3A-)/Re_x(T3B) against Roach & Brierley",
    "SPEC-LIT 144.7 Gate 90-T leg 3, kOmegaSSTGamma, onset spread Re_x(T3A-)/Re_x(T3B) against Roach & Brierley",
];

/// What the fit gate is held against, and the leg-3 rows.
const AGAINST_FIT: &str = "the ERCOFTAC key's measured free-stream Tu(x), its TUFS column";
const AGAINST_KEY: [&str; 3] = [
    "Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3B key (ercoftac-t3b, SPEC-LIT 144.7)",
    "Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A- key (ercoftac-t3am, SPEC-LIT 144.7)",
    "Roach & Brierley's measured C_f(Re_x), the ERCOFTAC T3A key (ercoftac-t3a, SPEC-LIT 144.6)",
];
const AGAINST_BLASIUS: &str =
    "Blasius' 0.664/sqrt(Re_x) (SPEC-LIT 88.7) over the part of the plate the key measures laminar";
const AGAINST_ORDER: &str =
    "Roach & Brierley's T3 series, the keys' own (144.4a) onsets and their ratio";

/// An open verdict, with the reason in its headline. The caller has opened
/// `gate`'s scope.
fn open_gate(
    c: &mut Checks,
    gate: &'static str,
    how: How,
    against: &'static str,
    headline: String,
) {
    c.report(GateReport {
        verdict: Verdict::Open,
        how,
        gate,
        against,
        headline,
        detail: vec![],
        uncertainty: Some(Uncertainty::SingleMesh("no comparison was made (SPEC-LIT 144.7)")),
    });
}

/// The section: the three keys (a digest line or a missing line each,
/// §110.1), then [`check_t3_fitted_on`] on the recorded runs.
pub(crate) fn check_t3_fitted(c: &mut Checks) -> Result<()> {
    let mut keys: [Option<Vec<crate::key::T3aStation>>; 3] = [None, None, None];
    for (ci, case) in CASES.into_iter().enumerate() {
        match load_key(case) {
            Ok(k) => {
                c.note(&k.digest_line);
                keys[ci] = Some(k.stations);
            }
            Err(e) => {
                let (id, file) = key_file(case);
                c.note(&crate::answer_key_missing_line(id, file, &e));
            }
        }
    }
    check_t3_fitted_on(
        c,
        [keys[0].as_deref(), keys[1].as_deref(), keys[2].as_deref()],
        RECORDS,
        RECORDS_SINGLE,
        t3a::RECORDS,
    )
}

/// The replay: (A) each case's fit (144.7a) and its input check, with the
/// inflow (144.7c) noted; (B) the three §144.4 rows per case and model on
/// the log grid, every recorded record held to its key's fitted inflow, and
/// the class-R row; (C) the ordering and the spread (144.7d) per model.
/// `keys` are by case index; `t3a_tmr` is `t3a::RECORDS`, the ordering's
/// fallback for T3A.
pub(crate) fn check_t3_fitted_on(
    c: &mut Checks,
    keys: [Option<&[crate::key::T3aStation]>; 3],
    records: [[[Option<&T3FitRecord>; 3]; 2]; 3],
    single: [[[Option<&T3FitRecord>; 3]; 2]; 3],
    t3a_tmr: [[Option<&t3a::T3aRecord>; 3]; 2],
) -> Result<()> {
    let mut key_on: [Option<f64>; 3] = [None, None, None];
    for (ci, case) in CASES.into_iter().enumerate() {
        // ---- A: the fit (144.7a), its input check, and (144.7c) -----------
        let mut fit_in: Option<(DecayFit, Inflow)> = None;
        c.enter_gate(GATE_FIT[ci]);
        'fit: {
            let Some(st) = keys[ci] else {
                let (id, _) = key_file(case);
                open_gate(c, GATE_FIT[ci], How::Live, AGAINST_FIT, format!("answer key {id} missing"));
                break 'fit;
            };
            let tu: Vec<(f64, f64)> = st.iter().map(|s| (s.re_x, s.tu_pct)).collect();
            let fit = match fit_decay(&tu) {
                Ok(f) => f,
                Err(why) => {
                    let headline = format!("{}: the fit refused the key: {why}", case.name());
                    open_gate(c, GATE_FIT[ci], How::Live, AGAINST_FIT, headline);
                    break 'fit;
                }
            };
            let nu = rig_nu(case);
            let re_up = X_UP * U / nu;
            let inf = match inflow_from_fit(&fit, re_up, nu, U) {
                Ok(inf) => inf,
                Err(why) => {
                    let headline = format!("{}: {why}", case.name());
                    open_gate(c, GATE_FIT[ci], How::Live, AGAINST_FIT, headline);
                    break 'fit;
                }
            };
            c.note(&format!(
                "144.7 {}: the fit of the key's measured Tu(x): b* = {:.6e}, Tu_LE = {:.6} %, the worst \
                 station misses the fitted decay by {:.3} % (band 3 %); rig nu = {:.3e} m^2/s, H = {:.2} m, \
                 Re_up = {:.4e}, Re_L = {:.4e}; the inflow (144.7c): s = {:.4}, Tu_in = {:.4} %, \
                 k_in = {:.6e} m^2/s^2, omega_in = {:.6e} 1/s",
                case.name(), fit.b, fit.tu_le, 100.0 * fit.worst, nu, rig_height(case), re_up,
                LENGTH * U / nu, inf.s, inf.tu_in, inf.k0, inf.omega0
            ));
            if fit.worst <= 0.03 {
                c.check_p(
                    &format!(
                        "144.7 {}: the fitted decay's worst miss of the measured Tu(x), an input check (3 %)",
                        case.name()
                    ),
                    fit.worst as Scalar,
                    0.03,
                );
            } else {
                let (where_re, _) = st.iter().fold((f64::NAN, 0.0f64), |m, s| {
                    let e = (tu_fit(&fit, s.re_x) / s.tu_pct - 1.0).abs();
                    if e > m.1 { (s.re_x, e) } else { m }
                });
                c.report(GateReport {
                    verdict: Verdict::Misses,
                    how: How::Live,
                    gate: GATE_FIT[ci],
                    against: AGAINST_FIT,
                    headline: format!(
                        "fitted decay misses the measured Tu(x) by {:.2} % at Re_x {:.4e} (band 3 %); \
                         the run uses this inflow anyway (an input check, SPEC-LIT 144.7)",
                        100.0 * fit.worst,
                        where_re
                    ),
                    detail: vec![],
                    uncertainty: Some(Uncertainty::SingleMesh(
                        "an input check on the key's own stations; no mesh",
                    )),
                });
            }
            fit_in = Some((fit, inf));
        }
        c.leave_gate();
        // ---- the key's leg-3 reading, once --------------------------------
        let mut k_opt: Option<t3a::Leg4Key> = None;
        if let Some(st) = keys[ci] {
            let cf: Vec<(f64, f64)> = st.iter().map(|s| (s.re_x, s.cf)).collect();
            let read = t3a::read_key(&cf);
            c.require(
                &format!(
                    "144.7 {}: the key's least C_f lies at an interior station (144.4a has a vertex)",
                    case.name()
                ),
                read.is_ok(),
            );
            match read {
                Ok(k) => {
                    key_on[ci] = Some(k.onset_re_x);
                    k_opt = Some(k);
                }
                Err(why) => c.note(&format!("144.7 {}: {why}", case.name())),
            }
        }
        // ---- B: the three rows per model ----------------------------------
        for mi in 0..2 {
            let model = [t3a::T3aModel::Lm, t3a::T3aModel::Gamma][mi];
            let (_, inf) = match fit_in {
                Some(x) => x,
                None => {
                    let (id, _) = key_file(case);
                    for gate in [GATE_ONSET[ci][mi], GATE_TURB[ci][mi], GATE_LAM[ci][mi]] {
                        c.enter_gate(gate);
                        let headline = format!("answer key {id} missing");
                        open_gate(c, gate, How::Replayed, AGAINST_KEY[ci], headline);
                        c.leave_gate();
                    }
                    continue;
                }
            };
            let Some(k) = k_opt.as_ref() else {
                for gate in [GATE_ONSET[ci][mi], GATE_TURB[ci][mi], GATE_LAM[ci][mi]] {
                    c.enter_gate(gate);
                    let headline = format!("the key shows no transition to hold a run against");
                    open_gate(c, gate, How::Replayed, AGAINST_KEY[ci], headline);
                    c.leave_gate();
                }
                continue;
            };
            let Some(fine) = records[ci][mi][0] else {
                for gate in [GATE_ONSET[ci][mi], GATE_TURB[ci][mi], GATE_LAM[ci][mi]] {
                    c.enter_gate(gate);
                    let headline = format!(
                        "no {} {} run recorded yet - run the record_t3fit_* tests of SPEC-LIT 144.7",
                        model.name(),
                        case.name()
                    );
                    open_gate(c, gate, How::Replayed, AGAINST_KEY[ci], headline);
                    c.leave_gate();
                }
                continue;
            };
            for r in records[ci][mi].iter().chain(single[ci][mi].iter()).flatten() {
                let v = leg3_rows(r, &k);
                c.note(&format!(
                    "144.7 {} {} {}+{} x {} ({}): onset Re_x {:.4e} ({:+.2} %), worst turbulent C_f \
                     {:+.2} % at Re_x {:.4e}, worst laminar C_f {:+.2} % of Blasius at Re_x {:.3e}{}",
                    case.name(), r.model.name(), r.cells.0, r.cells.1, r.cells.2, r.build,
                    v.onset.0, 100.0 * v.onset.1, 100.0 * v.turbulent.1, v.turbulent.0,
                    100.0 * v.laminar.1, v.laminar.0,
                    onset_kind_note(v.onset_kind)
                ));
                c.check_d(
                    &format!(
                        "144.7 {} {} {}+{} x {} {}: the record ran this key's fitted inflow",
                        case.name(), r.model.name(), r.cells.0, r.cells.1, r.cells.2, r.build
                    ),
                    nan_max((r.tu_in / inf.tu_in - 1.0).abs(), (r.omega_in / inf.omega0 - 1.0).abs()) as Scalar,
                    1e-4,
                );
            }
            let recs = &records[ci][mi];
            let rows = leg3_rows(fine, &k);
            let turb_cf =
                k.turbulent.iter().find(|s| s.0 == rows.turbulent.0).map_or(f64::NAN, |s| s.1);
            let onset_what = match rows.onset_kind {
                OnsetKind::LeastFace => "onset Re_x (least C_f) against (144.4a)".to_string(),
                OnsetKind::GridMinimum => {
                    "onset Re_x (the first grid minimum (144.7e), the least face being the trailing \
                     edge) against (144.4a)"
                        .to_string()
                }
                OnsetKind::Upstream => {
                    "onset Re_x (C_f falls over the whole band: read as the 2e4 bound (144.7e)) \
                     against (144.4a)"
                        .to_string()
                }
            };
            let judged = [
                (
                    GATE_ONSET[ci][mi],
                    AGAINST_KEY[ci],
                    onset_what,
                    rows.onset,
                    t3a::ONSET_BAND,
                    0.0,
                    study_of(recs, |r| record_onset(r).0),
                    k.onset_re_x,
                ),
                (
                    GATE_TURB[ci][mi],
                    AGAINST_KEY[ci],
                    "worst turbulent C_f against the station's (144.4b)".to_string(),
                    rows.turbulent,
                    t3a::TURB_BAND,
                    t3a::U_D_TURB,
                    study_of(recs, |r| cf_at_log(r, rows.turbulent.0)),
                    turb_cf,
                ),
                (
                    GATE_LAM[ci][mi],
                    AGAINST_BLASIUS,
                    "worst laminar C_f against Blasius (144.4c)".to_string(),
                    rows.laminar,
                    t3a::LAM_BAND,
                    0.0,
                    study_of(recs, move |r| rows.lam_i.map_or(f64::NAN, |i| lam_excess(r, i))),
                    1.0,
                ),
            ];
            let mut bands = Vec::new();
            for (gate, against, what, (at, e), band, u_d, study, scale) in judged {
                c.enter_gate(gate);
                match &study {
                    Ok(st) => c.note(&format!(
                        "144.7 {} {}: {what}, study: {}",
                        case.name(),
                        model.name(),
                        st.one_line()
                    )),
                    Err(why) => c.note(&format!(
                        "144.7 {} {}: {what}, study refused: {why}",
                        case.name(),
                        model.name()
                    )),
                }
                let u_num = study.as_ref().map_or(0.0, |st| f64::from(st.u_fine) / scale);
                let u_val = (u_num * u_num + u_d * u_d).sqrt();
                let label = format!(
                    "144.7 {} {}: |E| + u_val of the {what}, finest mesh ({:.0} %)",
                    case.name(),
                    model.name(),
                    100.0 * band
                );
                let b = t3a::band_110(e.abs(), band, u_val);
                bands.push((b, band, u_val));
                if e.is_finite() && b == Band::Inside {
                    c.replaying(|c| c.check_p(&label, (e.abs() + u_val) as Scalar, band as Scalar));
                    c.leave_gate();
                    continue;
                }
                let uncertainty = match study {
                    Ok(st) => Uncertainty::Study(st),
                    Err(_) => {
                        Uncertainty::SingleMesh("the 94 study refused the three levels; the finest mesh alone")
                    }
                };
                let headline = format!(
                    "{} {what}: E = {:+.2} % at Re_x {:.4e} on the finest mesh, band {:.0} %, \
                     u_val {:.2} % (u_num {:.2} %, u_D {:.0} %)",
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
                c.leave_gate();
            }
            if let Some(s) = single[ci][mi][0] {
                let v = leg3_rows(s, &k);
                let same = [v.onset.1, v.turbulent.1, v.laminar.1]
                    .iter()
                    .zip(&bands)
                    .all(|(e, &(b, band, u_val))| t3a::band_110(e.abs(), band, u_val) == b);
                c.replaying(|c| {
                    c.require(
                        &format!(
                            "144.7 {} {}: the single build's three rows land in the f64 build's bands",
                            case.name(),
                            model.name()
                        ),
                        same,
                    )
                });
            }
        }
    }
    // ---- C: the ordering and the spread, per model ------------------------
    if let (Some(b), Some(a), Some(am)) = (key_on[0], key_on[2], key_on[1]) {
        c.require(
            "144.7: the keys' own onset ordering T3B < T3A < T3A- (144.4a)",
            b < a && a < am,
        );
    }
    for (mi, model) in [t3a::T3aModel::Lm, t3a::T3aModel::Gamma].into_iter().enumerate() {
        let o_b = records[0][mi][0].map(|r| record_onset(r).0);
        let o_am = records[1][mi][0].map(|r| record_onset(r).0);
        let (o_a, o_a_from) = match records[2][mi][0] {
            Some(r) => (Some(record_onset(r).0), "the refitted T3A run (144.7)"),
            None => (
                t3a_tmr[mi][0].map(|r| r.cf_min_re_x),
                "the TMR-inflow T3A record of SPEC-LIT 144.3, the fallback",
            ),
        };
        c.enter_gate(GATE_ORDER[mi]);
        match (o_b, o_a, o_am) {
            (Some(b), Some(a), Some(am)) => {
                c.note(&format!(
                    "144.7 {}: onsets O(T3B) = {:.4e}, O(T3A) = {:.4e} from {}, O(T3A-) = {:.4e}",
                    model.name(),
                    b,
                    a,
                    o_a_from,
                    am
                ));
                if b < a && a < am {
                    c.check_d(
                        &format!("144.7 {}: the onset ordering T3B < T3A < T3A- holds", model.name()),
                        0.0,
                        0.0,
                    );
                } else {
                    c.report(GateReport {
                        verdict: Verdict::Misses,
                        how: How::Replayed,
                        gate: GATE_ORDER[mi],
                        against: AGAINST_ORDER,
                        headline: format!(
                            "the onsets read O(T3B) = {:.4e}, O(T3A) = {:.4e}, O(T3A-) = {:.4e} - the \
                             ordering T3B < T3A < T3A- is broken",
                            b, a, am
                        ),
                        detail: vec![],
                        uncertainty: Some(Uncertainty::SingleMesh(
                            "the finest f64 record of each case; each onset's three-level study is \
                             in its own onset gate (SPEC-LIT 144.7)",
                        )),
                    });
                }
            }
            _ => {
                let mut missing = Vec::new();
                if o_b.is_none() {
                    missing.push("a T3B fine f64 run");
                }
                if o_a.is_none() {
                    missing.push("a T3A fine f64 run (or the 144.3 fallback)");
                }
                if o_am.is_none() {
                    missing.push("a T3A- fine f64 run");
                }
                let headline = format!("missing {}", missing.join(", "));
                open_gate(c, GATE_ORDER[mi], How::Replayed, AGAINST_ORDER, headline);
            }
        }
        c.leave_gate();
        c.enter_gate(GATE_SPREAD[mi]);
        match (o_b, o_am, key_on[0], key_on[1]) {
            (Some(b), Some(am), Some(kb), Some(kam)) => {
                let s_sim = am / b;
                let s_key = kam / kb;
                let e = s_sim / s_key - 1.0;
                let st_b = study_of(&records[0][mi], |r| record_onset(r).0);
                let st_am = study_of(&records[1][mi], |r| record_onset(r).0);
                let u_num = ((st_b.as_ref().map_or(0.0, |s| f64::from(s.u_fine)) / b).powi(2)
                    + (st_am.as_ref().map_or(0.0, |s| f64::from(s.u_fine)) / am).powi(2))
                .sqrt();
                c.note(&format!(
                    "144.7 {}: the fitted-inflow spread S_sim = O(T3A-)/O(T3B) = {:.4e} against the \
                     keys' S_key = Re_on(T3A-)/Re_on(T3B) = {:.4e}; 88.10 leg 3's single-rig \
                     construction read 51.9x",
                    model.name(),
                    s_sim,
                    s_key
                ));
                let band = t3a::ONSET_BAND;
                let label = format!(
                    "144.7 {}: |E| + u_val of the onset spread (144.7d), finest mesh ({:.0} %)",
                    model.name(),
                    100.0 * band
                );
                let bnd = t3a::band_110(e.abs(), band, u_num);
                if e.is_finite() && bnd == Band::Inside {
                    c.replaying(|c| c.check_p(&label, (e.abs() + u_num) as Scalar, band as Scalar));
                } else {
                    let uncertainty = match (st_b, st_am) {
                        (Ok(_), Ok(s_am)) => Uncertainty::Study(s_am),
                        _ => Uncertainty::SingleMesh(
                            "one of the two three-level studies of cf_min_re_x is refused; the finest meshes alone",
                        ),
                    };
                    let headline = format!(
                        "the onset spread (144.7d): S_sim = {:.4e} against S_key = {:.4e}, E = {:+.2} %, \
                         band {:.0} %, u_val {:.2} % (u_num from the two onsets' studies)",
                        s_sim,
                        s_key,
                        100.0 * e,
                        100.0 * band,
                        100.0 * u_num
                    );
                    c.report(GateReport {
                        verdict: if bnd == Band::Undecided { Verdict::Open } else { Verdict::Misses },
                        how: How::Replayed,
                        gate: GATE_SPREAD[mi],
                        against: AGAINST_ORDER,
                        headline: if bnd == Band::Undecided {
                            format!(
                                "undecided, within u_val of the band edge (SPEC-LIT 32.4, 110.1): {headline}"
                            )
                        } else {
                            headline
                        },
                        detail: vec![],
                        uncertainty: Some(uncertainty),
                    });
                }
            }
            _ => {
                let mut missing = Vec::new();
                if o_b.is_none() {
                    missing.push("a T3B fine f64 run");
                }
                if o_am.is_none() {
                    missing.push("a T3A- fine f64 run");
                }
                if key_on[0].is_none() {
                    missing.push("the T3B key");
                }
                if key_on[1].is_none() {
                    missing.push("the T3A- key");
                }
                let headline = format!("missing {}", missing.join(", "));
                open_gate(c, GATE_SPREAD[mi], How::Replayed, AGAINST_ORDER, headline);
            }
        }
        c.leave_gate();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
//  The recorder side
// ---------------------------------------------------------------------------

/// The rig from a fit, the runs, and the record literals. Test code, because
/// the runs are made by this binary's ignored tests (`record_t3fit_*`), never
/// by `ofgpu-validate` itself (SPEC-LIT 110.1).
#[cfg(test)]
pub(crate) mod runner {
    use ofgpu::{Error, Gpu, Result};

    use super::super::t3a::runner::{cf_interp, run_rig, RigInputs};
    use super::{
        fit_decay, inflow_from_fit, key_file, load_key, rig_height, rig_nu, DecayFit, T3Case,
        T3FitRecord, LENGTH, N_LOG, U, X_UP,
    };

    /// The caps the recorders run to, finest level first. §144.3 ran to
    /// 40 000; these were priced against the shared card before any run and
    /// fit docs/17 VAL-35's three hours of card time (SPEC-LIT 144.7).
    pub(crate) const MAX_ITERS: [usize; 3] = [12000, 8000, 10000];

    /// The end-cell ratio in `y`, §144.3's, the same on the three rigs.
    const R_Y: f64 = 1000.0;

    /// The case's rig: §144.3's plate geometry (`x_up` 0.25 m, `L` 3 m, `r_y`
    /// 1000) at the case's `nu` and `H`, with (144.7c)'s inflow from the
    /// case's fit. `t3a::runner::rig_spec` turns it into the mesh's spec.
    pub(crate) fn rig_inputs(
        case: T3Case,
        fit: &DecayFit,
    ) -> std::result::Result<RigInputs, String> {
        let nu = rig_nu(case);
        let f = inflow_from_fit(fit, X_UP * U / nu, nu, U)?;
        Ok(RigInputs {
            x_le: X_UP,
            length: LENGTH,
            height: rig_height(case),
            r_y: R_Y,
            nu,
            tu_in: f.tu_in,
            k0: f.k0,
            omega0: f.omega0,
        })
    }

    /// One recorded run of one case: its key's fit (144.7a), the rig from
    /// (144.7c), and §144.3's run under the case's label. Every error names
    /// the case and SPEC-LIT 144.7.
    pub(crate) fn run_fitted(
        gpu: &Gpu,
        case: T3Case,
        model: super::t3a::T3aModel,
        n: (usize, usize, usize),
        max_iters: usize,
        progress: bool,
    ) -> Result<T3FitRecord> {
        let (id, file) = key_file(case);
        let key = load_key(case).map_err(|e| {
            Error::Config(format!(
                "SPEC-LIT 144.7 {}: answer key {id} ({file}) could not be read: {e}",
                case.name()
            ))
        })?;
        let tu: Vec<(f64, f64)> = key.stations.iter().map(|s| (s.re_x, s.tu_pct)).collect();
        let fit = fit_decay(&tu)
            .map_err(|e| Error::Config(format!("SPEC-LIT 144.7 {}: {e}", case.name())))?;
        let rig = rig_inputs(case, &fit)
            .map_err(|e| Error::Config(format!("SPEC-LIT 144.7 {}: {e}", case.name())))?;
        let run = run_rig(gpu, &rig, case.name(), model, n, max_iters, progress)?;
        let mut cf = [f64::NAN; N_LOG];
        for (i, v) in cf.iter_mut().enumerate() {
            *v = cf_interp(&run.faces, super::re_log(i));
        }
        Ok(T3FitRecord {
            case,
            model,
            cells: n,
            n_cells: run.n_cells,
            h: run.h,
            iterations: run.iterations,
            converged: run.converged,
            last_change: run.last_change,
            tu_in: rig.tu_in,
            omega_in: rig.omega0,
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

    /// The record as the Rust literal the slots above hold. NaN prints as
    /// `f64::NAN` so the literal compiles; the numbers keep `{:.6e}`.
    pub(crate) fn record_literal(r: &T3FitRecord) -> String {
        let sci = |v: f64| -> String {
            if v.is_nan() { "f64::NAN".to_string() } else { format!("{v:.6e}") }
        };
        let f6 = |v: f64| -> String {
            if v.is_nan() { "f64::NAN".to_string() } else { format!("{v:.6}") }
        };
        let cf: Vec<String> = r.cf.iter().map(|&v| sci(v)).collect();
        format!(
            "T3FitRecord {{ case: T3Case::{:?}, model: t3a::T3aModel::{:?}, cells: {:?}, n_cells: {}, \
             h: {}, iterations: {}, converged: {}, last_change: {}, tu_in: {}, omega_in: {}, \
             tu_le: {}, tu_le_min: {}, tu_le_max: {}, cf: [{}], cf_min_re_x: {}, cf_min: {}, \
             seconds: {:.1}, build: \"{}\", commit: \"\" }}",
            r.case,
            r.model,
            r.cells,
            r.n_cells,
            sci(r.h),
            r.iterations,
            r.converged,
            sci(r.last_change),
            sci(r.tu_in),
            sci(r.omega_in),
            f6(r.tu_le),
            f6(r.tu_le_min),
            f6(r.tu_le_max),
            cf.join(", "),
            sci(r.cf_min_re_x),
            sci(r.cf_min),
            r.seconds,
            r.build
        )
    }
}

// ---------------------------------------------------------------------------
//  The tests: invented keys and records, then the tree's own keys
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::runner::MAX_ITERS;
    use super::*;
    use crate::key::T3aStation;
    use ofgpu::Gpu;

    const MODELS: [t3a::T3aModel; 2] = [t3a::T3aModel::Lm, t3a::T3aModel::Gamma];

    /// An invented station: `run` names itself synthetic, the rest of the
    /// columns are the ones this gate reads (`re_x`, `cf`, `tu_pct`) and
    /// zeros elsewhere.
    fn st(re_x: f64, cf: f64, tu_pct: f64) -> T3aStation {
        T3aStation {
            run: "synthetic".to_string(),
            x_mm: re_x / 1000.0,
            re_x,
            u0: U,
            cf,
            h: 0.0,
            re_theta: 0.0,
            delta_mm: 0.0,
            tu_pct,
        }
    }

    /// The turbulent `C_f` law the invented keys and records share, so a
    /// record's turbulent stations meet the key's exactly and the log grid's
    /// power-law interpolation is exact on them.
    const TURB_A: f64 = 0.06;

    /// An invented key: the least `C_f` at `onset` with symmetric neighbours
    /// (so (144.4a)'s vertex is `onset` exactly), the greatest at `peak`,
    /// `turb` the turbulent stations past it, and `Tu` on the exact law
    /// `tu_le (1 + b Re_x)^-n` at every station, which the fit holds with no
    /// miss. The `b`s keep (144.7c)'s `s` at 0.25 / 0.4 / 0.2 on the rigs.
    fn syn_key(onset: f64, peak: f64, turb: &[f64], b: f64, tu_le: f64) -> Vec<T3aStation> {
        let n = decay_exponent();
        let tu = |re: f64| tu_le * (1.0 + b * re).powf(-n);
        let mut cf = vec![
            (onset * 0.6, 2.2e-3),
            (onset, 2.0e-3),
            (onset * 1.4, 2.2e-3),
            (peak, TURB_A * peak.powf(-0.2)),
        ];
        cf.extend(turb.iter().map(|&re| (re, TURB_A * re.powf(-0.2))));
        cf.into_iter().map(|(re, v)| st(re, v, tu(re))).collect()
    }

    /// The invented keys' `(onset, peak, turbulent Re_x, b, Tu_LE)` per
    /// case, in [`CASES`] order: onsets 5.6e4, 1.4e6, 1.5e5 - the keys' own
    /// ordering T3B < T3A < T3A- - and every turbulent station inside the
    /// grid.
    const SYN: [(f64, f64, &[f64], f64, f64); 3] = [
        (5.6e4, 1.2e5, &[1.44e5, 1.728e5], 3.0e-6, 4.0),
        (1.4e6, 2.0e6, &[2.4e6], 2.0e-6, 1.2),
        (1.5e5, 3.0e5, &[3.6e5, 4.32e5], 4.0e-6, 3.0),
    ];

    /// The three invented keys, in [`CASES`] order.
    fn syn_keys() -> [Vec<T3aStation>; 3] {
        std::array::from_fn(|ci| {
            let (onset, peak, turb, b, tu_le) = SYN[ci];
            syn_key(onset, peak, turb, b, tu_le)
        })
    }

    /// Each invented key's fit and (144.7c) inflow on its own rig.
    fn syn_fits(keys: &[Vec<T3aStation>; 3]) -> [(DecayFit, Inflow); 3] {
        std::array::from_fn(|ci| {
            let tu: Vec<(f64, f64)> = keys[ci].iter().map(|s| (s.re_x, s.tu_pct)).collect();
            let fit = fit_decay(&tu).expect("the invented key's Tu lies on its law");
            let nu = rig_nu(CASES[ci]);
            let inf =
                inflow_from_fit(&fit, X_UP * U / nu, nu, U).expect("s < 1 by construction");
            (fit, inf)
        })
    }

    /// A record shell: every field dummy, `cf` all NaN.
    fn blank(case: T3Case, model: t3a::T3aModel) -> T3FitRecord {
        let cells = t3a::runner::LEVELS[0];
        T3FitRecord {
            case,
            model,
            cells,
            n_cells: (cells.0 + cells.1) * cells.2,
            h: 0.01,
            iterations: 1000,
            converged: true,
            last_change: 1e-7,
            tu_in: 1.0,
            omega_in: 1.0,
            tu_le: 1.0,
            tu_le_min: 1.0,
            tu_le_max: 1.0,
            cf: [f64::NAN; N_LOG],
            cf_min_re_x: 1e5,
            cf_min: 2e-3,
            seconds: 1.0,
            build: "f64",
            commit: "",
        }
    }

    /// An invented record of `case`: laminar `1.02` Blasius on the grid up to
    /// its own least-`C_f` face and `TURB_A Re_x^-0.2` past it, that face at
    /// `onset (1 + 0.005 level) (1 + 1e-4 4^level)` - so its onset, turbulent
    /// and laminar rows hold with second-order grid convergence - and the
    /// case's fitted inflow, so the stale-inflow rows hold.
    fn syn_run(
        case: T3Case,
        model: t3a::T3aModel,
        level: usize,
        onset: f64,
        fit: &DecayFit,
        inf: &Inflow,
    ) -> T3FitRecord {
        let f = (1.0 + 0.005 * level as f64) * (1.0 + 1e-4 * 4f64.powi(level as i32));
        let on_r = onset * f;
        let mut r = blank(case, model);
        r.cells = t3a::runner::LEVELS[level];
        r.h = 0.01 * (1 << level) as f64;
        r.tu_in = inf.tu_in;
        r.omega_in = inf.omega0;
        r.tu_le = fit.tu_le;
        r.tu_le_min = fit.tu_le;
        r.tu_le_max = fit.tu_le;
        r.cf_min_re_x = on_r;
        for (i, v) in r.cf.iter_mut().enumerate() {
            let g = re_log(i);
            *v = f * if g < on_r {
                1.02 * t3a::BLASIUS_CF / g.sqrt()
            } else {
                TURB_A * g.powf(-0.2)
            };
        }
        r
    }

    /// The full invented slot set: every case, model and level, each case's
    /// records at its key's onset - or, for T3A-, at `am_onset`.
    fn syn_recs(fits: &[(DecayFit, Inflow); 3], am_onset: f64) -> Owned {
        std::array::from_fn(|ci| {
            let onset = if ci == 1 { am_onset } else { SYN[ci].0 };
            std::array::from_fn(|mi| {
                std::array::from_fn(|l| {
                    syn_run(CASES[ci], MODELS[mi], l, onset, &fits[ci].0, &fits[ci].1)
                })
            })
        })
    }

    /// `[Some; 3]` views of owned slots and keys, what the replay reads.
    type Owned = [[[T3FitRecord; 3]; 2]; 3];
    fn view(r: &Owned) -> [[[Option<&T3FitRecord>; 3]; 2]; 3] {
        std::array::from_fn(|ci| std::array::from_fn(|mi| std::array::from_fn(|l| Some(&r[ci][mi][l]))))
    }
    fn key_view(keys: &[Vec<T3aStation>; 3]) -> [Option<&[T3aStation]>; 3] {
        std::array::from_fn(|ci| Some(keys[ci].as_slice()))
    }

    /// Stations exactly on one decay law, the fit's cleanest input.
    fn exact_law() -> [(f64, f64); 5] {
        let n = decay_exponent();
        [1e4, 3e4, 1e5, 3e5, 1e6].map(|re: f64| (re, 5.0 * (1.0 + 3e-6 * re).powf(-n)))
    }

    #[test]
    fn the_fit_recovers_an_exact_decay() {
        let f = fit_decay(&exact_law()).expect("five stations on one law");
        assert!((f.b / 3e-6 - 1.0).abs() <= 1e-9, "b* = {}", f.b);
        assert!((f.tu_le / 5.0 - 1.0).abs() <= 1e-12, "Tu_LE = {}", f.tu_le);
        assert!(f.worst <= 1e-12, "worst = {}", f.worst);
        assert!(fit_decay(&exact_law()[..2]).is_err(), "fewer than three stations");
        let dead = [(1e4, 0.0), (3e4, 1.0), (1e5, 2.0)];
        assert!(fit_decay(&dead).is_err(), "a zero Tu");
        let nan = [(1e4, 1.0), (3e4, f64::NAN), (1e5, 2.0)];
        assert!(fit_decay(&nan).is_err(), "a NaN Tu");
    }

    #[test]
    fn the_inflow_reproduces_the_fit_downstream() {
        let f = fit_decay(&exact_law()).expect("fit");
        let nu = 3.0e-6;
        let inf = inflow_from_fit(&f, X_UP / nu, nu, U).expect("s = 0.25");
        assert!((inf.s / 0.25 - 1.0).abs() <= 1e-12, "s = {}", inf.s);
        let beta2 = f64::from(KOmegaSstCoeffs::default().beta_2);
        let n = decay_exponent();
        for r in [0.0, 1e4, 1e5, 1e6] {
            // 88.10's closed form from (Tu_in, omega_in) at x = x_up + R nu/U.
            let tau = 1.0 + beta2 * inf.omega0 * (X_UP + r * nu / U) / U;
            let tu = inf.tu_in * tau.powf(-n);
            assert!(
                (tu / tu_fit(&f, r) - 1.0).abs() <= 1e-12,
                "Re_x {r}: closed form {tu} against {}",
                tu_fit(&f, r)
            );
        }
        assert!(inflow_from_fit(&f, 1e6, nu, U).is_err(), "s = 3 >= 1");
        let k_fit = 1.5 * (inf.tu_in * U / 100.0) * (inf.tu_in * U / 100.0);
        assert!((inf.k0 / k_fit - 1.0).abs() <= 1e-12, "k_in = {}", inf.k0);
    }

    /// The three keys' fits against the supervisor's f64 oracle (docs/17
    /// VAL-35): `(b*, Tu_LE, worst, Tu_in, k_in, omega_in)` per case. `b*`
    /// and `omega_in` carry 1e-7 in f64, the rest 1e-9: the golden section
    /// walks a V-shaped minimum, whose last digits are its own noise floor.
    /// A case whose key is absent prints and is skipped.
    #[test]
    fn the_fits_of_the_three_keys_match_the_supervisor_oracle() {
        const ORACLE: [(f64, f64, f64, f64, f64, f64); 3] = [
            (4.721176040e-6, 6.086008929, 1.527599e-2, 7.986061, 9.566574591e-3, 31.33420158),
            (1.240719999e-6, 0.910763252, 3.633843e-2, 1.063466, 1.696439929e-4, 15.94405423),
            (1.281351728e-5, 3.442107740, 2.688323e-2, 6.003547, 5.406387201e-3, 86.13539070),
        ];
        let single = cfg!(feature = "single");
        let tol = if single { 1e-5 } else { 1e-9 };
        let tol_b = if single { 1e-5 } else { 1e-7 };
        let abs = if single { 1e-4 } else { 1e-6 };
        let mut seen = [false; 3];
        for (ci, case) in CASES.into_iter().enumerate() {
            let Ok(key) = load_key(case) else {
                let (id, file) = key_file(case);
                println!("  [answer-key] {id} ({file}) absent: the {} fit is not checked", case.name());
                continue;
            };
            seen[ci] = true;
            let tu: Vec<(f64, f64)> = key.stations.iter().map(|s| (s.re_x, s.tu_pct)).collect();
            let f = fit_decay(&tu).expect("the ERCOFTAC summary's Tu(x) fits");
            let nu = rig_nu(case);
            let inf = inflow_from_fit(&f, X_UP * U / nu, nu, U).expect("each case's s < 1");
            let (b, tu_le, worst, tu_in, k_in, omega_in) = ORACLE[ci];
            assert!((f.b / b - 1.0).abs() <= tol_b, "{}: b* = {}", case.name(), f.b);
            assert!((f.tu_le / tu_le - 1.0).abs() <= tol, "{}: Tu_LE = {}", case.name(), f.tu_le);
            assert!((f.worst - worst).abs() <= abs, "{}: worst = {}", case.name(), f.worst);
            assert!((inf.tu_in - tu_in).abs() <= abs, "{}: Tu_in = {}", case.name(), inf.tu_in);
            assert!((inf.k0 / k_in - 1.0).abs() <= tol, "{}: k_in = {}", case.name(), inf.k0);
            assert!(
                (inf.omega0 / omega_in - 1.0).abs() <= tol_b,
                "{}: omega_in = {}",
                case.name(),
                inf.omega0
            );
        }
        assert!(seen[0] && seen[1], "the T3B and T3A- keys carry the oracle's ordering");
        let on = |case| {
            let key = load_key(case).expect("seen above");
            let cf: Vec<(f64, f64)> = key.stations.iter().map(|s| (s.re_x, s.cf)).collect();
            t3a::read_key(&cf).expect("each key shows its transition")
        };
        let (kb, kam) = (on(T3Case::T3b), on(T3Case::T3am));
        assert!((kb.onset_re_x / 5.642944e4 - 1.0).abs() <= 1e-6, "{}", kb.onset_re_x);
        assert_eq!((kb.least, kb.peak), (3, 5), "T3B's least and greatest C_f stations");
        assert!((kb.laminar_hi / 4.31e4 - 1.0).abs() <= 1e-6, "{}", kb.laminar_hi);
        assert!((kam.onset_re_x / 1.443366e6 - 1.0).abs() <= 1e-6, "{}", kam.onset_re_x);
        assert_eq!((kam.least, kam.peak), (10, 15), "T3A-'s least and greatest C_f stations");
        assert!((kam.laminar_hi / 1.306e6 - 1.0).abs() <= 1e-6, "{}", kam.laminar_hi);
        assert!(
            (kam.onset_re_x / kb.onset_re_x / 25.578250 - 1.0).abs() <= 1e-6,
            "S_key = {}",
            kam.onset_re_x / kb.onset_re_x
        );
    }

    #[test]
    fn the_log_grid_and_its_interpolation() {
        assert!((re_log(0) / 1e4 - 1.0).abs() <= 1e-12, "{}", re_log(0));
        assert!((re_log(N_LOG - 1) / 2.4e6 - 1.0).abs() <= 1e-12, "{}", re_log(N_LOG - 1));
        let mut r = blank(T3Case::T3b, t3a::T3aModel::Lm);
        for (i, v) in r.cf.iter_mut().enumerate() {
            *v = TURB_A * re_log(i).powf(-0.2);
        }
        let at = cf_at_log(&r, 3.33e5);
        assert!(
            (at / (TURB_A * 3.33e5f64.powf(-0.2)) - 1.0).abs() <= 1e-12,
            "C_f at 3.33e5 = {at}"
        );
        assert!(cf_at_log(&r, 5e3).is_nan(), "below the grid");
        assert!(cf_at_log(&r, 3e6).is_nan(), "above the grid");
    }

    /// (144.7e)'s three readings, on synthetic records, exact.
    #[test]
    fn the_record_onset_reads_the_least_face_the_grid_minimum_or_the_bound() {
        // A V at re_log(10), the least face past the last finite point: the
        // onset is the vertex (144.4a) through the V's three grid points,
        // hand-computed here so the test does not restate the code's own
        // helper.
        let mut r = blank(T3Case::T3b, t3a::T3aModel::Lm);
        r.cf_min_re_x = 9.950110e5;
        r.cf[9] = 3.0e-3;
        r.cf[10] = 2.0e-3;
        r.cf[11] = 2.5e-3;
        let (x0, y0) = (re_log(9), r.cf[9]);
        let (x1, y1) = (re_log(10), r.cf[10]);
        let (x2, y2) = (re_log(11), r.cf[11]);
        let num = (x1 - x0) * (x1 - x0) * (y1 - y2) - (x1 - x2) * (x1 - x2) * (y1 - y0);
        let den = (x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0);
        let want = x1 - 0.5 * num / den;
        let (got, kind) = record_onset(&r);
        assert_eq!(kind, OnsetKind::GridMinimum, "the V at re_log(10)");
        assert!((got / want - 1.0).abs() <= 1e-12, "vertex {got} against {want}");

        // Monotone-falling C_f over the whole band: the onset reads as the
        // 2e4 bound.
        let mut r = blank(T3Case::T3b, t3a::T3aModel::Lm);
        r.cf_min_re_x = 9.801408e5;
        for (i, v) in r.cf.iter_mut().enumerate().take(21) {
            *v = 5.0e-3 - i as f64 * 1.0e-4;
        }
        assert_eq!(
            record_onset(&r),
            (t3a::RE_X_LAM_LO, OnsetKind::Upstream),
            "C_f falls over the whole band"
        );

        // The least face inside the recorded grid: §144.4's reading,
        // unchanged.
        let mut r = blank(T3Case::T3am, t3a::T3aModel::Lm);
        r.cf_min_re_x = 3.0e4;
        for (i, v) in r.cf.iter_mut().enumerate().take(31) {
            *v = 2.0e-3 + i as f64 * 1.0e-5;
        }
        assert_eq!(record_onset(&r), (3.0e4, OnsetKind::LeastFace), "the least face interior");
    }

    /// The supervisor's oracle on the recorded runs (docs/17 VAL-35): the
    /// (onset, kind) each recorded f64 run must read, 1e-6 relative.
    #[test]
    fn the_recorded_onsets_read_the_supervisor_oracle() {
        const ORACLE: [(&T3FitRecord, f64, OnsetKind); 9] = [
            (&F64_T3AM_LM_FINE, 1.210603e6, OnsetKind::LeastFace),
            (&F64_T3AM_LM_MEDIUM, 1.215083e6, OnsetKind::LeastFace),
            (&F64_T3AM_LM_COARSE, 1.046459e6, OnsetKind::LeastFace),
            (&F64_T3AM_GAMMA_MEDIUM, 1.215083e6, OnsetKind::LeastFace),
            (&F64_T3B_LM_FINE, 4.993813e4, OnsetKind::GridMinimum),
            (&F64_T3B_LM_MEDIUM, 5.162461e4, OnsetKind::GridMinimum),
            (&F64_T3B_LM_COARSE, 2.0e4, OnsetKind::Upstream),
            (&F64_T3B_GAMMA_MEDIUM, 6.858995e4, OnsetKind::GridMinimum),
            (&F64_T3B_GAMMA_COARSE, 7.022652e4, OnsetKind::GridMinimum),
        ];
        for (r, want, kind) in ORACLE {
            let (got, got_kind) = record_onset(r);
            assert_eq!(got_kind, kind, "{} {} {}", r.case.name(), r.model.name(), r.build);
            assert!(
                (got / want - 1.0).abs() <= 1e-6,
                "{} {} {}: onset {} against {}",
                r.case.name(),
                r.model.name(),
                r.build,
                got,
                want
            );
        }
    }

    #[test]
    fn replay_on_synthetic_records_holds_and_misses_by_gate_name() {
        let keys = syn_keys();
        let fits = syn_fits(&keys);
        let kv = key_view(&keys);
        let single_none: [[[Option<&T3FitRecord>; 3]; 2]; 3] = [[[None; 3]; 2]; 3];
        let tmr_none: [[Option<&t3a::T3aRecord>; 3]; 2] = [[None; 3]; 2];

        // (a) every invented record holds its key: no failure, no verdict.
        let recs = syn_recs(&fits, SYN[1].0);
        let mut c = Checks::new();
        check_t3_fitted_on(&mut c, kv, view(&recs), single_none, tmr_none).expect("replay");
        assert_eq!(c.failures, 0, "{:?}", c.gates.iter().map(|g| g.headline.clone()).collect::<Vec<_>>());
        assert!(
            c.gates.is_empty(),
            "{:?}",
            c.gates.iter().map(|g| (g.gate, g.verdict)).collect::<Vec<_>>()
        );

        // (b) T3A-'s onsets at 3e6, past the grid's end: (144.7e) reads each
        // record's onset at the 2e4 bound (its grid C_f falls over the whole
        // band), so the onset and turbulent rows and both spreads miss, the
        // laminar rows have no points upstream of that bound and miss, and
        // both orderings break.
        let recs = syn_recs(&fits, 3.0e6);
        let mut c = Checks::new();
        check_t3_fitted_on(&mut c, kv, view(&recs), single_none, tmr_none).expect("replay");
        assert_eq!(c.failures, 0);
        let got: Vec<(&str, Verdict)> = c.gates.iter().map(|g| (g.gate, g.verdict)).collect();
        assert_eq!(
            got,
            vec![
                (GATE_ONSET[1][0], Verdict::Misses),
                (GATE_TURB[1][0], Verdict::Misses),
                (GATE_LAM[1][0], Verdict::Misses),
                (GATE_ONSET[1][1], Verdict::Misses),
                (GATE_TURB[1][1], Verdict::Misses),
                (GATE_LAM[1][1], Verdict::Misses),
                (GATE_ORDER[0], Verdict::Misses),
                (GATE_SPREAD[0], Verdict::Misses),
                (GATE_ORDER[1], Verdict::Misses),
                (GATE_SPREAD[1], Verdict::Misses),
            ]
        );

        // (c) T3A-'s onsets at 1e5, below T3A's: the two ordering gates miss
        // as well, and the keys' own ordering require still holds.
        let recs = syn_recs(&fits, 1.0e5);
        let mut c = Checks::new();
        check_t3_fitted_on(&mut c, kv, view(&recs), single_none, tmr_none).expect("replay");
        assert_eq!(c.failures, 0);
        // (94.3) on this replay too: every verdict this section reaches the
        // summary with declares its mesh study, the ordering's included.
        let unnamed: Vec<&str> =
            c.gates.iter().filter(|g| g.uncertainty.is_none()).map(|g| g.gate).collect();
        assert!(unnamed.is_empty(), "no mesh study: {unnamed:?}");
        let got: Vec<(&str, Verdict)> = c.gates.iter().map(|g| (g.gate, g.verdict)).collect();
        assert_eq!(
            got,
            vec![
                (GATE_ONSET[1][0], Verdict::Misses),
                (GATE_ONSET[1][1], Verdict::Misses),
                (GATE_ORDER[0], Verdict::Misses),
                (GATE_SPREAD[0], Verdict::Misses),
                (GATE_ORDER[1], Verdict::Misses),
                (GATE_SPREAD[1], Verdict::Misses),
            ]
        );
    }

    #[test]
    fn every_gate_is_open_when_keys_or_records_are_missing() {
        let none: [[[Option<&T3FitRecord>; 3]; 2]; 3] = [[[None; 3]; 2]; 3];
        let tmr_none: [[Option<&t3a::T3aRecord>; 3]; 2] = [[None; 3]; 2];
        let mut c = Checks::new();
        check_t3_fitted_on(&mut c, [None; 3], none, none, tmr_none).expect("replay");
        assert_eq!((c.total, c.failures), (0, 0));
        assert_eq!(c.gates.len(), 25, "three fit gates, eighteen leg gates, four ordering ones");
        assert!(c.gates.iter().all(|g| g.verdict == Verdict::Open));

        let keys = syn_keys();
        let kv = key_view(&keys);
        let mut c = Checks::new();
        check_t3_fitted_on(&mut c, kv, none, none, tmr_none).expect("replay");
        assert_eq!(c.failures, 0, "{:?}", c.gates.iter().map(|g| (g.gate, g.verdict)).collect::<Vec<_>>());
        assert_eq!(
            c.gates.len(),
            22,
            "the fit gates are judged; the eighteen leg gates and four ordering ones are open"
        );
        assert!(c.gates.iter().all(|g| g.verdict == Verdict::Open));
        assert_eq!(c.total, 7, "three fit rows, three interior-station rows, one ordering row");
    }

    #[test]
    fn the_stale_inflow_row_fails_a_record_from_another_fit() {
        let keys = syn_keys();
        let fits = syn_fits(&keys);
        let kv = key_view(&keys);
        let single_none: [[[Option<&T3FitRecord>; 3]; 2]; 3] = [[[None; 3]; 2]; 3];
        let tmr_none: [[Option<&t3a::T3aRecord>; 3]; 2] = [[None; 3]; 2];
        let mut recs = syn_recs(&fits, SYN[1].0);
        recs[0][0][0].omega_in *= 1.01;
        let mut c = Checks::new();
        check_t3_fitted_on(&mut c, kv, view(&recs), single_none, tmr_none).expect("replay");
        assert_eq!(c.failures, 1, "only the stale-inflow row fails");
        assert!(c.gates.is_empty(), "the stale row is a check, not a verdict");
    }

    /// How a registered gate landed on the recorded runs: `Open` or `Misses`
    /// is its verdict in the registry; `Check` is a gate whose rows landed
    /// inside their bands, which registers no verdict - it is held to its
    /// absence from the registry, its rows being check rows (`c.failures`
    /// carries them).
    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Measured {
        Open,
        Misses,
        Check,
    }

    /// The tree's own replay: the recorded runs of 144.7 against the three
    /// keys, held to the verdicts measured on the supervisor's runs (docs/17
    /// VAL-35 run 2, SPEC-LIT 144.7): the T3B fit and the T3A fit inside
    /// their 3 % band and T3A-'s 3.63 % past it; T3B LM's onset undecided
    /// under the coarse record's 2e4 bound and its turbulent row inside,
    /// its laminar row +58.30 % past the 10 % band; T3A- LM's onset
    /// undecided, its turbulent +132.12 % and laminar +27.55 % past theirs;
    /// the LM ordering holding at the (144.7e) onsets and the LM spread
    /// undecided under the T3B study; every gate with no fine run recorded
    /// open by name. Without the keys the twenty-five gates are open.
    #[test]
    fn the_recorded_runs_against_the_fitted_keys() {
        // (gate, the measured verdict, and for a MISS the E and band as the
        // section's own headline carries them)
        const MEASURED: [(&str, Measured, &str, &str); 25] = [
            (GATE_FIT[0], Measured::Check, "", ""),
            (GATE_FIT[1], Measured::Misses, "by 3.63 %", "band 3 %"),
            (GATE_FIT[2], Measured::Check, "", ""),
            (GATE_ONSET[0][0], Measured::Open, "", ""),
            (GATE_TURB[0][0], Measured::Check, "", ""),
            (GATE_LAM[0][0], Measured::Misses, "E = +58.30 %", "band 10 %"),
            (GATE_ONSET[0][1], Measured::Open, "", ""),
            (GATE_TURB[0][1], Measured::Open, "", ""),
            (GATE_LAM[0][1], Measured::Open, "", ""),
            (GATE_ONSET[1][0], Measured::Open, "", ""),
            (GATE_TURB[1][0], Measured::Misses, "E = +132.12 %", "band 10 %"),
            (GATE_LAM[1][0], Measured::Misses, "E = +27.55 %", "band 10 %"),
            (GATE_ONSET[1][1], Measured::Open, "", ""),
            (GATE_TURB[1][1], Measured::Open, "", ""),
            (GATE_LAM[1][1], Measured::Open, "", ""),
            (GATE_ONSET[2][0], Measured::Open, "", ""),
            (GATE_TURB[2][0], Measured::Open, "", ""),
            (GATE_LAM[2][0], Measured::Open, "", ""),
            (GATE_ONSET[2][1], Measured::Open, "", ""),
            (GATE_TURB[2][1], Measured::Open, "", ""),
            (GATE_LAM[2][1], Measured::Open, "", ""),
            (GATE_ORDER[0], Measured::Check, "", ""),
            (GATE_SPREAD[0], Measured::Open, "", ""),
            (GATE_ORDER[1], Measured::Open, "", ""),
            (GATE_SPREAD[1], Measured::Open, "", ""),
        ];
        let mut c = Checks::new();
        check_t3_fitted(&mut c).expect("replay");
        if c.total == 0 {
            assert_eq!(c.gates.len(), 25);
            println!("  [answer-key] the ercoftac keys are absent: all 25 gates open by name");
            return;
        }
        assert_eq!(c.failures, 0, "{:?}", c.gates.iter().map(|g| (g.gate, g.headline.clone())).collect::<Vec<_>>());
        let open = c.gates.iter().filter(|g| g.verdict == Verdict::Open).count();
        let missed = c.gates.iter().filter(|g| g.verdict == Verdict::Misses).count();
        assert_eq!(
            (open, missed, c.gates.len()),
            (17, 4, 21),
            "the four gates inside their bands register no verdict"
        );
        for (gate, want, e, band) in MEASURED {
            let got = c.gates.iter().find(|g| g.gate == gate);
            match want {
                Measured::Check => assert!(
                    got.is_none(),
                    "{gate} lands inside its bands and registers no verdict: {:?}",
                    got.map(|g| (g.verdict, g.headline.clone()))
                ),
                Measured::Open => {
                    assert_eq!(got.map(|g| g.verdict), Some(Verdict::Open), "{gate}")
                }
                Measured::Misses => {
                    let g = got.unwrap_or_else(|| panic!("{gate} misses and must report"));
                    assert_eq!(g.verdict, Verdict::Misses, "{gate}");
                    assert!(
                        g.headline.contains(e) && g.headline.contains(band),
                        "{gate}: {}",
                        g.headline
                    );
                    println!("MISSED: {gate}: {e} ({band})");
                }
            }
        }
    }

    /// The refitted T3A rig is §144.3's in everything but its inflow, so the
    /// mesh it builds is `t3a::runner::t3a_spec`'s bit for bit.
    #[test]
    fn the_fitted_t3a_rig_is_144_3s_but_for_its_inflow() {
        use super::runner::rig_inputs;
        use super::super::t3a::runner::{rig_spec, t3a_spec, RigInputs};
        let n = t3a::runner::LEVELS[0];
        let fit = DecayFit { b: 1.281351728e-5, tu_le: 3.442107740, worst: 0.0 };
        let rig = rig_inputs(T3Case::T3aFit, &fit).expect("T3A's s = 0.64");
        let tmr = RigInputs::t3a();
        assert_eq!((rig.x_le, rig.length, rig.height, rig.r_y, rig.nu),
            (tmr.x_le, tmr.length, tmr.height, tmr.r_y, tmr.nu));
        assert_ne!(rig.tu_in, tmr.tu_in, "the inflow is the one thing the refit changes");
        let (a, t) = (rig_spec(&rig, n), t3a_spec(n));
        assert_eq!(
            (a.x_upstream, a.length, a.height, a.n_upstream, a.n_plate, a.n_y),
            (t.x_upstream, t.length, t.height, t.n_upstream, t.n_plate, t.n_y)
        );
        assert_eq!((a.r_upstream, a.r_plate, a.r_y), (t.r_upstream, t.r_plate, t.r_y));
        assert_eq!((a.u_inf, a.nu), (t.u_inf, t.nu));
    }

    // ---- the recorded runs and their price (the supervisor runs these) ----

    /// One recorded run of one case and model on one level, printed as the
    /// record literal its slot above holds.
    fn record(case: T3Case, model: t3a::T3aModel, level: usize) {
        let gpu = Gpu::new(0).expect("a GPU");
        let r = super::runner::run_fitted(&gpu, case, model, t3a::runner::LEVELS[level],
            MAX_ITERS[level], true).expect("run");
        println!("RECORD {}", super::runner::record_literal(&r));
    }

    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.7 (minutes of GPU time)"]
    fn record_t3fit_t3b_lm_fine() {
        record(T3Case::T3b, t3a::T3aModel::Lm, 0);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.7 (minutes of GPU time)"]
    fn record_t3fit_t3b_lm_medium() {
        record(T3Case::T3b, t3a::T3aModel::Lm, 1);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.7 (minutes of GPU time)"]
    fn record_t3fit_t3b_lm_coarse() {
        record(T3Case::T3b, t3a::T3aModel::Lm, 2);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.7 (minutes of GPU time)"]
    fn record_t3fit_t3b_gamma_fine() {
        record(T3Case::T3b, t3a::T3aModel::Gamma, 0);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.7 (minutes of GPU time)"]
    fn record_t3fit_t3b_gamma_medium() {
        record(T3Case::T3b, t3a::T3aModel::Gamma, 1);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.7 (minutes of GPU time)"]
    fn record_t3fit_t3b_gamma_coarse() {
        record(T3Case::T3b, t3a::T3aModel::Gamma, 2);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.7 (minutes of GPU time)"]
    fn record_t3fit_t3am_lm_fine() {
        record(T3Case::T3am, t3a::T3aModel::Lm, 0);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.7 (minutes of GPU time)"]
    fn record_t3fit_t3am_lm_medium() {
        record(T3Case::T3am, t3a::T3aModel::Lm, 1);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.7 (minutes of GPU time)"]
    fn record_t3fit_t3am_lm_coarse() {
        record(T3Case::T3am, t3a::T3aModel::Lm, 2);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.7 (minutes of GPU time)"]
    fn record_t3fit_t3am_gamma_fine() {
        record(T3Case::T3am, t3a::T3aModel::Gamma, 0);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.7 (minutes of GPU time)"]
    fn record_t3fit_t3am_gamma_medium() {
        record(T3Case::T3am, t3a::T3aModel::Gamma, 1);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.7 (minutes of GPU time)"]
    fn record_t3fit_t3am_gamma_coarse() {
        record(T3Case::T3am, t3a::T3aModel::Gamma, 2);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.7 (minutes of GPU time)"]
    fn record_t3fit_t3a_lm_fine() {
        record(T3Case::T3aFit, t3a::T3aModel::Lm, 0);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.7 (minutes of GPU time)"]
    fn record_t3fit_t3a_lm_medium() {
        record(T3Case::T3aFit, t3a::T3aModel::Lm, 1);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.7 (minutes of GPU time)"]
    fn record_t3fit_t3a_lm_coarse() {
        record(T3Case::T3aFit, t3a::T3aModel::Lm, 2);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.7 (minutes of GPU time)"]
    fn record_t3fit_t3a_gamma_fine() {
        record(T3Case::T3aFit, t3a::T3aModel::Gamma, 0);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.7 (minutes of GPU time)"]
    fn record_t3fit_t3a_gamma_medium() {
        record(T3Case::T3aFit, t3a::T3aModel::Gamma, 1);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 144.7 (minutes of GPU time)"]
    fn record_t3fit_t3a_gamma_coarse() {
        record(T3Case::T3aFit, t3a::T3aModel::Gamma, 2);
    }

    /// The price of a level: `T3F_PRICE=<t3b|t3am|t3a>,<level>,<iters>` runs
    /// that case's LM on that level for that many iterations, no progress
    /// lines, and prints one PRICE line.
    #[test]
    #[ignore = "pricing for SPEC-LIT 144.7"]
    fn price_t3_fitted() {
        let spec = std::env::var("T3F_PRICE").unwrap_or_else(|_| "t3b,2,200".to_string());
        let f: Vec<&str> = spec.split(',').collect();
        let case = match f[0] {
            "t3am" => T3Case::T3am,
            "t3a" => T3Case::T3aFit,
            _ => T3Case::T3b,
        };
        let level: usize = f[1].parse().expect("level");
        let iters: usize = f[2].parse().expect("iters");
        let gpu = Gpu::new(0).expect("a GPU");
        let started = std::time::Instant::now();
        let r = super::runner::run_fitted(&gpu, case, t3a::T3aModel::Lm,
            t3a::runner::LEVELS[level], iters, false).expect("run");
        println!("PRICE {} {:?} {:.3} s/200 iter ({} iterations)", case.name(), level,
            started.elapsed().as_secs_f64() * 200.0 / iters as f64, r.iterations);
    }
}
