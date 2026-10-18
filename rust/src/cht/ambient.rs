// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! A face that exchanges heat with something not meshed - §98. The
//! convective and the grey-radiative external condition as SPEC-LIT §4's
//! Robin triple (S98.1)-(S98.3), the datasheet's `h_r` and `h_total`
//! (S98.4), the linearisation residual (S98.5), the host half of a Newton
//! pass (§98.3), Churchill & Chu's vertical-plate correlation (S98.6), and
//! the two closed forms Gates 98-A and 98-B are held against, (S98.7) and
//! (S98.8). Host arithmetic and one host round trip; nothing here launches
//! a kernel, and nothing here runs inside a captured region.
//!
//! Written from:
//!   F. P. Incropera, D. P. DeWitt, *Fundamentals of Heat and Mass
//!     Transfer*, Wiley - ch. 3 (the straight fin with an adiabatic tip,
//!     which SPEC-LIT §98.6 derives again) and ch. 9 (the restatement of
//!     Churchill & Chu's correlation that `CHURCHILL_CHU` is transcribed
//!     from)
//!   S. W. Churchill, H. H. S. Chu, *Int. J. Heat Mass Transfer* 18 (1975)
//!     1323-1329, DOI 10.1016/0017-9310(75)90243-4 - PAYWALLED and NOT
//!     read; its coefficients come from the restatement above
//!   ofgpu `SPEC-LIT.md` §4, §47.2, §98
//!
//! SPEC-LIT 120.8 extends the same triple onto any fluid wall the energy
//! equation solves: [`ThermalFace`] is one boundary face whose `(fr,
//! refValue, refGrad)` `crate::energy::Energy` rewrites every outer
//! iteration from the face's CURRENT conductance - (120.8b)-(120.8f)'s
//! outer wall with its layers and radiation, (120.8h)'s Danckwerts inlet -
//! and [`LumpedMass`] / [`ThermalWalls`] carry the lumped
//! `lumpedMassTemperature` patch and the whole set. [`lump_coefficients`]
//! is (120.8g)'s area-shared film coefficient and reference temperature.
//!
//! OpenFOAM's `externalWallHeatFluxTemperature` is GPL and was not opened.
//! No GPL-licensed source was consulted.

use crate::device::Gpu;
use crate::error::{Error, Result};
use crate::field::GpuScalarField;
use crate::radiation::SIGMA_SB;
use crate::Scalar;

/// What one external face loses, in the case's own units (K, W/(m^2 K)):
/// `h = 0` on a face that only radiates, `emissivity = 0` on one that only
/// convects (§98.1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExternalLoss {
    pub h: Scalar,
    pub t_inf: Scalar,
    pub emissivity: Scalar,
    pub t_env: Scalar,
}

impl ExternalLoss {
    /// Does the face radiate - i.e. does its triple move with `T_b`?
    pub fn radiates(&self) -> bool {
        self.emissivity > 0.0
    }

    /// The true outward loss at `t_b`, W/m^2:
    /// `h (T_b - TInf) + eps sigma (T_b^4 - TEnv^4)`.
    pub fn q_out(&self, t_b: Scalar) -> Scalar {
        self.h * (t_b - self.t_inf)
            + self.emissivity * SIGMA_SB * (t_b.powi(4) - self.t_env.powi(4))
    }

    /// (S98.2)-(S98.3): `(H, T_ref)` with `q_out(T_b) ~ H (T_b - T_ref)`,
    /// the tangent about `t_star`. A face that only convects returns
    /// `(h, TInf)` as they are - taken, not computed - so (S98.3) is
    /// (S98.1) in every bit there.
    pub fn linearised(&self, t_star: Scalar) -> (Scalar, Scalar) {
        if !self.radiates() {
            return (self.h, self.t_inf);
        }
        let es = self.emissivity * SIGMA_SB;
        let b = 4.0 * es * t_star.powi(3);
        let a = es * (t_star.powi(4) - self.t_env.powi(4)) - b * t_star;
        let hh = self.h + b;
        if hh > 0.0 {
            (hh, (self.h * self.t_inf - a) / hh)
        } else {
            (0.0, 0.0)
        }
    }

    /// (S98.1)/(S98.3): the triple `(fr, refValue, refGrad)` on a face of
    /// cell-to-face conductance `c_b` (S47.4), linearised about `t_star`.
    /// Exactly adiabatic when either conductance is not positive - the
    /// convention of (S47.8).
    pub fn triple(&self, c_b: Scalar, t_star: Scalar) -> (Scalar, Scalar, Scalar) {
        let (hh, t_ref) = self.linearised(t_star);
        if hh > 0.0 && c_b > 0.0 {
            (hh / (hh + c_b), t_ref, 0.0)
        } else {
            (0.0, 0.0, 0.0)
        }
    }

    /// (S98.4): the datasheet's secant radiative coefficient at `t_b`,
    /// `eps sigma (T_b^2 + TEnv^2)(T_b + TEnv)`. Reported, never solved
    /// with: freezing it is a secant iteration (§98.2).
    pub fn h_r(&self, t_b: Scalar) -> Scalar {
        self.emissivity
            * SIGMA_SB
            * (t_b * t_b + self.t_env * self.t_env)
            * (t_b + self.t_env)
    }

    /// (S98.4): `h + h_r(t_b)`, what a datasheet quotes.
    pub fn h_total(&self, t_b: Scalar) -> Scalar {
        self.h + self.h_r(t_b)
    }
}

/// One boundary face carrying an [`ExternalLoss`]: its index in the
/// concatenated boundary, its cell-to-face conductance (S47.4), and the
/// temperature its triple is currently linearised about.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExternalFace {
    pub bf: usize,
    pub c_b: Scalar,
    pub loss: ExternalLoss,
    pub t_star: Scalar,
}

impl ExternalFace {
    /// (S98.3) on this face, about its current `t_star`.
    pub fn triple(&self) -> (Scalar, Scalar, Scalar) {
        self.loss.triple(self.c_b, self.t_star)
    }
}

/// SPEC-LIT 120.8: one boundary face whose `T` triple `crate::energy::Energy`
/// rewrites every outer iteration. `q_in` is positive INTO the domain (the
/// sign of SPEC-LIT 32.2's `q`); `r_layers` is (120.8a)'s series resistance
/// `sum_i t_i/kappa_i`; in flux/power modes `h` is 0 and `q_in` carries the
/// prescribed heat, in the coefficient mode it is the other way round, and on
/// a Danckwerts face `advective` is set and `t_inf` is the inlet value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalFace {
    /// Flattened boundary-face index.
    pub bf: usize,
    /// Outer film coefficient, W/(m^2 K); a lump's `m cp a_n / A`; 0 in the
    /// flux and power modes.
    pub h: f64,
    /// `Ta`, K; a lump's `T_ref` of (120.8g) (rewritten every step); a
    /// Danckwerts face's `T_in`.
    pub t_inf: f64,
    /// Grey emissivity to a surround at `Ta`; 0 = no radiation.
    pub emissivity: f64,
    /// (120.8a): the wall layers' series resistance, m^2 K/W.
    pub r_layers: f64,
    /// W/m^2 into the domain (the flux and power modes), else 0.
    pub q_in: f64,
    /// `totalFlowRateAdvectiveDiffusive`: `H` comes from (120.8h), off
    /// `phi_conv`, not off `h`.
    pub advective: bool,
}

impl ThermalFace {
    /// (120.8b)-(120.8f) and (120.8h): `Some((fr, refValue, refGrad))`, or
    /// `None` = leave the face untouched (the guard: `k_eff` or `C_b` not
    /// positive, a radiating face's `T_o*` not positive, or a Danckwerts
    /// face with no area). `c` is the face conductance `k_eff Delta_b`,
    /// `t_b`/`t_p` the face and owner-cell temperatures, `phi_conv` the
    /// face's `cp rho_f phi` (positive out) and `mag_sf` its area. The
    /// operation order here is the kernel's; both round once per stored
    /// value (SPEC-LIT 118.2). Arithmetic in f64 - the widened build's own
    /// precision, and the kernel widens every input to `ofacc`.
    pub fn triple(
        &self,
        k_eff: f64,
        delta: f64,
        t_b: f64,
        t_p: f64,
        phi_conv: f64,
        mag_sf: f64,
    ) -> Option<(f64, f64, f64)> {
        if !(k_eff > 0.0) {
            return None;
        }
        let c = k_eff * delta;
        if !(c > 0.0) {
            return None;
        }
        let sigma = SIGMA_SB as f64;
        let (mut hh, t_ref);
        if self.advective {
            // (120.8h): the advected heat of the incoming stream. `phi_conv`
            // is positive out, so an INFLOW face has `phi_conv < 0` and
            // carries `H = -phi_conv/|S_f|`; an outflow face is
            // zero-gradient (`hh = 0` -> `fr = 0` below).
            if !(mag_sf > 0.0) {
                return None;
            }
            hh = if phi_conv < 0.0 { -phi_conv / mag_sf } else { 0.0 };
            t_ref = self.t_inf;
        } else {
            // (120.8b): the heat the last solve conducted from the cell to
            // the face - which a converged face also passes through the
            // layers, so (120.8c) is the outer temperature that iterate
            // implies.
            let q_c = c * (t_p - t_b);
            let t_o = t_b - self.r_layers * q_c;
            if self.emissivity > 0.0 {
                if !(t_o > 0.0) {
                    return None;
                }
                // (120.8d): (S98.2)'s tangent of the quartic, taken about
                // the OUTER surface's temperature.
                let es = self.emissivity * sigma;
                let b = 4.0 * es * t_o * t_o * t_o;
                let a = es * (t_o * t_o * t_o * t_o - self.t_inf * self.t_inf
                    * self.t_inf * self.t_inf)
                    - b * t_o;
                let ho = self.h + b;
                t_ref = (self.h * self.t_inf - a) / ho;
                hh = ho;
            } else {
                // Taken, not computed - (S98.1) in every bit.
                hh = self.h;
                t_ref = self.t_inf;
            }
            // (120.8e): the layers in series with the outer film.
            hh = hh / (1.0 + self.r_layers * hh);
        }
        // (120.8f), the identity `C_b (T_P - T_b) = H (T_b - T_ref) - q_in`
        // at every iterate.
        let fr = hh / (hh + c);
        Some((fr, t_ref, self.q_in / k_eff))
    }
}

/// SPEC-LIT 120.8 (120.8g): one `lumpedMassTemperature` patch - a mass
/// `mass` of specific heat `Cp` behind the patch at one temperature. Each of
/// its `faces` is a flattened boundary-face index that is ALSO the `bf` of
/// one entry of `ThermalWalls::faces`, where the per-step film the
/// coefficients give is written; `Energy::set_thermal_walls` refuses a lump
/// face that is not.
#[derive(Debug, Clone, PartialEq)]
pub struct LumpedMass {
    pub patch: String,
    /// The lump's faces, as flattened boundary-face indices, in face order.
    pub faces: Vec<usize>,
    /// Each face's area, f64, in the same order - (120.8g)'s area mean.
    pub area: Vec<f64>,
    /// `mass * Cp`, J/K.
    pub mass_cp: f64,
    /// The file's `value`, area-meaned once: `T_m^0`.
    pub t0: f64,
}

/// SPEC-LIT 120.8: the thermal faces and lumps `field_setup::thermal_walls`
/// reads from `T`'s field and `Energy::set_thermal_walls` takes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ThermalWalls {
    pub faces: Vec<ThermalFace>,
    pub lumps: Vec<LumpedMass>,
}

/// (120.8g)'s area-shared pair `(H, T_ref)` for one lump this step:
/// `H = mass_cp a_n / A` and `T_ref = -(a_0 T_m^n + a_00 T_m^{n-1})/a_n`,
/// from the energy equation's OWN ddt row, shared out over the patch by
/// area. `a_n` is never zero on a transient step - the caller refuses that
/// by name.
pub fn lump_coefficients(
    mass_cp: f64,
    area: f64,
    a_n: f64,
    a_0: f64,
    a_00: f64,
    t_n: f64,
    t_nm1: f64,
) -> (f64, f64) {
    (mass_cp * a_n / area, -(a_0 * t_n + a_00 * t_nm1) / a_n)
}

/// §98.3's criterion: a pass is the last when `max_f |T_b - T*|` is at most
/// this times `max_f T_b`. The f32 build's former `1e-4` arm is gone: an f32
/// iterate settles to an exact fixed point, change 0 (SPEC-LIT 118.3).
pub const NEWTON_RTOL: Scalar = 1.0e-10;
/// §98.3: the most Newton passes one step may take before it is refused.
pub const NEWTON_MAX_PASSES: usize = 50;

/// What one [`relinearise`] measured before it moved the faces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Relinearised {
    /// `max_f |T_b - T*|`, K - the pass's Newton correction.
    pub change: Scalar,
    /// `max_f T_b`, K - what [`NEWTON_RTOL`] is relative to.
    pub scale: Scalar,
    /// (S98.5) of the triples the face values were solved with.
    pub residual: Scalar,
}

/// (S98.5) over `faces`, from face values `bt` that were solved with the
/// triples the faces carry NOW (their current `t_star`). Zero when no face
/// loses anything.
pub fn linearisation_residual(bt: &[Scalar], faces: &[ExternalFace]) -> Scalar {
    let (mut num, mut den) = (0.0 as Scalar, 0.0 as Scalar);
    for f in faces {
        let tb = bt[f.bf];
        let q = f.loss.q_out(tb);
        let (hh, t_ref) = f.loss.linearised(f.t_star);
        num = num.max((q - hh * (tb - t_ref)).abs());
        den = den.max(q.abs());
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

/// §98.3: the host half of one Newton pass. Reads the face values back,
/// measures (S98.5) of the triples they were solved with, moves every
/// face's `t_star` to its `T_b` and rewrites `(fr, refValue)` there;
/// `refGrad` stays the zero it was written with. Three downloads and two
/// writes BETWEEN two solves - never inside a captured region, where
/// `Gpu::download` refuses by name.
pub fn relinearise(
    gpu: &Gpu,
    t: &mut GpuScalarField,
    faces: &mut [ExternalFace],
) -> Result<Relinearised> {
    if faces.is_empty() {
        return Ok(Relinearised { change: 0.0, scale: 0.0, residual: 0.0 });
    }
    let bt = gpu.download(&t.bf)?;
    let residual = linearisation_residual(&bt, faces);
    let mut fr = gpu.download(&t.fr)?;
    let mut rv = gpu.download(&t.ref_value)?;
    let (mut change, mut scale) = (0.0 as Scalar, 0.0 as Scalar);
    for f in faces.iter_mut() {
        let tb = bt[f.bf];
        if !(tb > 0.0) || !tb.is_finite() {
            return Err(Error::Config(format!(
                "boundary face {}: a radiating face reached T_b = {tb} K, and the \
                 quartic of (S98.2) needs an absolute, finite temperature (SPEC-LIT 98.3)",
                f.bf
            )));
        }
        change = change.max((tb - f.t_star).abs());
        scale = scale.max(tb);
        f.t_star = tb;
        let (a, b, _) = f.triple();
        fr[f.bf] = a;
        rv[f.bf] = b;
    }
    gpu.write(&mut t.fr, &fr)?;
    gpu.write(&mut t.ref_value, &rv)?;
    Ok(Relinearised { change, scale, residual })
}

/// §98.5 row 10: the refusal of a step whose radiating faces did not meet
/// [`NEWTON_RTOL`] in [`NEWTON_MAX_PASSES`] passes, naming the last
/// corrections.
pub fn newton_refused(case: &str, passes: &[Scalar]) -> Error {
    let from = passes.len().saturating_sub(4);
    let tail: Vec<String> = passes[from..].iter().map(|d| format!("{d:.3e}")).collect();
    Error::Config(format!(
        "{case}: the radiating faces did not meet max|T_b - T*| <= {:e} max T_b in {} \
         Newton passes (SPEC-LIT 98.3); the last corrections were [{}] K. A correction \
         that stops shrinking has reached the linear solver's own floor - tighten \
         `numerics.tolerance`",
        NEWTON_RTOL,
        passes.len(),
        tail.join(", ")
    ))
}

// answer-key: churchill-chu1975
/// (S98.6)'s five coefficients `[0.825, 0.387, 0.492, 9/16, 8/27]`, as the
/// restatement in Incropera & DeWitt gives them (§98.4) - trusted to that
/// transcription and gated against nothing.
pub const CHURCHILL_CHU: [Scalar; 5] = [0.825, 0.387, 0.492, 9.0 / 16.0, 8.0 / 27.0];
/// The Rayleigh range Churchill & Chu state for (S98.6).
pub const CHURCHILL_CHU_RA: (Scalar, Scalar) = (1.0e-1, 1.0e12);

/// (S98.6): `Nu_L` of a vertical plate. `setting` is the JSON path of the
/// `h` entry, which every refusal names.
pub fn churchill_chu_nu(setting: &str, ra: Scalar, pr: Scalar) -> Result<Scalar> {
    let (lo, hi) = CHURCHILL_CHU_RA;
    if !(ra >= lo && ra <= hi) {
        return Err(Error::Config(format!(
            "{setting}/Ra = {ra:e} lies outside the range Churchill & Chu state for \
             (S98.6), [1e-1, 1e12]; a correlation is not extrapolated (SPEC-LIT 98.4)"
        )));
    }
    if !(pr > 0.0) || !pr.is_finite() {
        return Err(Error::Config(format!(
            "{setting}/Pr = {pr}: the Prandtl number has to be finite and positive \
             (SPEC-LIT 98.4)"
        )));
    }
    let [c0, c1, c2, p1, p2] = CHURCHILL_CHU;
    let psi = (1.0 + (c2 / pr).powf(p1)).powf(p2);
    let s = c0 + c1 * ra.powf(1.0 / 6.0) / psi;
    Ok(s * s)
}

/// (S98.6): `h = Nu_L kappa / L`, W/(m^2 K), evaluated once at lowering
/// from the Rayleigh number the case states (§98.4).
pub fn churchill_chu_h(
    setting: &str,
    ra: Scalar,
    pr: Scalar,
    kappa: Scalar,
    l: Scalar,
) -> Result<Scalar> {
    for (what, v) in [("kappa", kappa), ("L", l)] {
        if !(v > 0.0) || !v.is_finite() {
            return Err(Error::Config(format!(
                "{setting}/{what} = {v}: it has to be finite and positive (SPEC-LIT 98.4)"
            )));
        }
    }
    Ok(churchill_chu_nu(setting, ra, pr)? * kappa / l)
}

/// (S98.7): the base heat flow of a straight fin with an adiabatic tip, W:
/// `sqrt(h P k A) theta_b tanh(m L)`, `m^2 = h P/(k A)` - Gate 98-A's
/// closed form (§98.6).
pub fn fin_heat_flow(
    h: Scalar,
    perimeter: Scalar,
    k: Scalar,
    area: Scalar,
    length: Scalar,
    theta_b: Scalar,
) -> Scalar {
    let m = (h * perimeter / (k * area)).sqrt();
    (h * perimeter * k * area).sqrt() * theta_b * (m * length).tanh()
}

/// What [`radiating_slab_root`] found.
#[derive(Debug, Clone, PartialEq)]
pub struct SlabRoot {
    /// The root `T_r` of (S98.8), K.
    pub root: Scalar,
    /// `|T_{k+1} - T_k|` of every Newton step taken, K.
    pub corrections: Vec<Scalar>,
    /// (S98.8)'s asymptotic constant `C = f''/(2 f')` at the root, 1/K.
    pub newton_constant: Scalar,
}

/// (S98.8): Newton on `f(T) = eps sigma (T^4 - TEnv^4) - (k/d)(T_i - T)`
/// from `T_i`, to `1e-14` relative - Gate 98-B's closed form (§98.6).
pub fn radiating_slab_root(
    k_over_d: Scalar,
    t_i: Scalar,
    emissivity: Scalar,
    t_env: Scalar,
) -> SlabRoot {
    let es = emissivity * SIGMA_SB;
    let f = |t: Scalar| es * (t.powi(4) - t_env.powi(4)) - k_over_d * (t_i - t);
    let fp = |t: Scalar| 4.0 * es * t.powi(3) + k_over_d;
    let mut t = t_i;
    let mut corrections = Vec::new();
    for _ in 0..60 {
        let next = t - f(t) / fp(t);
        let d = (next - t).abs();
        corrections.push(d);
        t = next;
        if d <= 1.0e-14 * t.abs() {
            break;
        }
    }
    let newton_constant = 12.0 * es * t * t / (2.0 * fp(t));
    SlabRoot { root: t, corrections, newton_constant }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_convective_triple_is_the_series_conductance() {
        let loss = ExternalLoss { h: 25.0, t_inf: 300.0, emissivity: 0.0, t_env: 0.0 };
        let c_b = 296.0;
        let (fr, rv, rg) = loss.triple(c_b, 123.0);
        assert_eq!(rv, 300.0);
        assert_eq!(rg, 0.0);
        assert!((fr - 25.0 / 321.0).abs() < 1e-15);
        let t_p = 380.0;
        let t_b = fr * rv + (1.0 - fr) * t_p;
        assert!(
            (c_b * (t_p - t_b) / (loss.h * (t_b - loss.t_inf)) - 1.0).abs() < 1e-13,
            "the conducted and the convected flux of (S98.1) are the same number: \
             {} vs {}",
            c_b * (t_p - t_b),
            loss.h * (t_b - loss.t_inf)
        );
        assert_eq!(loss.triple(0.0, 300.0), (0.0, 0.0, 0.0));
    }

    #[test]
    fn the_radiative_triple_is_the_tangent_and_its_residual_is_second_order() {
        let loss = ExternalLoss { h: 10.0, t_inf: 300.0, emissivity: 0.8, t_env: 290.0 };
        let t_star = 420.0;
        let (hh, t_ref) = loss.linearised(t_star);
        assert!(
            ((hh * (t_star - t_ref) / loss.q_out(t_star) - 1.0).abs() as f64)
                < crate::precision::roundoff_tol(1e-12),
            "the tangent touches the quartic at T*: {} vs {}",
            hh * (t_star - t_ref),
            loss.q_out(t_star)
        );
        let slope = loss.h + 4.0 * loss.emissivity * SIGMA_SB * t_star.powi(3);
        assert!(((hh / slope - 1.0).abs() as f64)
            < crate::precision::roundoff_tol(1e-14));
        let err = |d: Scalar| loss.q_out(t_star + d) - hh * (t_star + d - t_ref);
        assert!(err(1.0) > 0.0, "the quartic lies above its tangent");
        assert!(((err(2.0) / err(1.0)) - 4.0).abs() < 0.05);
    }

    #[test]
    fn the_secant_h_r_reproduces_the_quartic_and_h_total_is_the_sum() {
        let loss = ExternalLoss { h: 10.0, t_inf: 300.0, emissivity: 0.9, t_env: 300.0 };
        let ts: [Scalar; 3] = [301.0, 350.0, 600.0];
        for t_b in ts {
            let q_r = loss.emissivity * SIGMA_SB * (t_b.powi(4) - loss.t_env.powi(4));
            assert!(
                ((loss.h_r(t_b) * (t_b - loss.t_env) / q_r - 1.0).abs() as f64)
                    < crate::precision::roundoff_tol(1e-12),
                "h_r (T_b - TEnv) = q_r at {t_b} K"
            );
            assert!(
                ((loss.h_total(t_b) * (t_b - loss.t_inf) / loss.q_out(t_b) - 1.0).abs() as f64)
                    < crate::precision::roundoff_tol(1e-12),
                "h_total (T_b - TInf) = q_out at {t_b} K"
            );
        }
    }

    #[test]
    fn churchill_chu_is_refused_outside_its_stated_range_and_names_it() {
        for ra in [1.0e-2, 2.0e12] {
            let msg = churchill_chu_h("regions/r/patches/p/T/h", ra, 0.71, 0.026, 0.2)
                .unwrap_err()
                .to_string();
            assert!(msg.contains("regions/r/patches/p/T/h/Ra"), "{msg}");
            assert!(msg.contains("[1e-1, 1e12]"), "{msg}");
            assert!(msg.contains("SPEC-LIT 98.4"), "{msg}");
        }
        assert!(churchill_chu_h("x", 1.0e9, 0.71, 0.026, 0.2).is_ok());
        for (pr, kappa, l, what) in [
            (0.0, 0.026, 0.2, "Pr"),
            (0.71, 0.0, 0.2, "kappa"),
            (0.71, 0.026, -1.0, "L"),
        ] {
            let msg = churchill_chu_h("x", 1.0e6, pr, kappa, l).unwrap_err().to_string();
            assert!(msg.contains(&format!("x/{what}")), "{msg}");
        }
    }

    #[test]
    fn churchill_chu_rises_with_rayleigh_and_reaches_its_large_pr_limit() {
        let nu = |ra: Scalar| churchill_chu_nu("x", ra, 0.71).unwrap();
        assert!(nu(1e4) < nu(1e6) && nu(1e6) < nu(1e8) && nu(1e8) < nu(1e10));
        let big = churchill_chu_nu("x", 1.0e9, 1.0e12).unwrap();
        let limit = (0.825 + 0.387 * (1.0e9 as Scalar).powf(1.0 / 6.0)).powi(2);
        assert!(
            (big / limit - 1.0).abs() < 1e-4,
            "at Pr -> inf psi -> 1: {big:e} vs {limit:e}"
        );
    }

    #[test]
    fn the_radiating_slab_root_converges_quadratically_to_the_quartic() {
        let s = radiating_slab_root(50.0, 500.0, 0.8, 300.0);
        assert!(s.root > 300.0 && s.root < 500.0);
        let es = 0.8 * SIGMA_SB;
        let f = es * (s.root.powi(4) - (300.0 as Scalar).powi(4)) - 50.0 * (500.0 - s.root);
        assert!(
            (f.abs() as f64)
                < crate::precision::roundoff_tol(1e-9) * 50.0 * (500.0 - s.root) as f64,
            "the root solves the quartic: f = {f:e}"
        );
        let d = &s.corrections;
        let mut count = 0;
        for k in 0..d.len() - 1 {
            if d[k + 1] > 1e-12 * s.root {
                count += 1;
                assert!(
                    d[k + 1] / (d[k] * d[k]) <= 2.0 * s.newton_constant,
                    "pass {k}: d[k+1]/d[k]^2 = {} exceeds 2C = {}",
                    d[k + 1] / (d[k] * d[k]),
                    2.0 * s.newton_constant
                );
            }
        }
        assert!(count >= 2, "{d:?}");
        assert!(d.len() <= 8, "{d:?}");
    }

    #[test]
    fn the_fin_closed_form_reaches_its_two_limits() {
        let (h, p, k, a, th) = (25.0, 0.02, 148.0, 1.0e-5, 100.0);
        let short = fin_heat_flow(h, p, k, a, 1.0e-5, th);
        assert!(
            (short / (h * p * 1.0e-5 * th) - 1.0).abs() < 1e-6,
            "a short fin is its whole surface at the base temperature: {short:e}"
        );
        let long = fin_heat_flow(h, p, k, a, 10.0, th);
        assert!(
            (long / ((h * p * k * a).sqrt() * th) - 1.0).abs() < 1e-12,
            "a long fin sheds sqrt(h P k A) theta_b: {long:e}"
        );
    }
}
