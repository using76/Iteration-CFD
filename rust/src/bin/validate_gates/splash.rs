// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Gate 78-D's measured leg (SPEC-LIT §146.2): each splash criterion of
//! §78.4, as `ofgpu::parcels` implements it, predicts the threshold speed of
//! every row of Riboux & Gordillo's key
//! (`reference/riboux_gordillo_2014/critical_speed.csv`, §146.1), and the
//! prediction is held against the measured `V*` within the band (146.5),
//! per liquid by the rule (146.6).
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §146.1 (the key, (146.1)-(146.4)) and §146.2 (this
//!     leg, (146.5)-(146.6)), §78.3-§78.4 (the groups and the two criteria,
//!     (78.4a)-(78.4c)), §78.10 (Gate 78-D), §69 (the verdict registry)
//!   G. Riboux and J. M. Gordillo, *Phys. Rev. Lett.* 113 (2014) 024507,
//!     doi:10.1103/PhysRevLett.113.024507 - read as arXiv:1401.6943v3,
//!     Table I and Fig. 4(c), through the answer-key reader with its sha256
//!     printed
//!   C. Mundo, M. Sommerfeld and C. Tropea, *Int. J. Multiphase Flow* 21
//!     (1995) 151, and Bai & Gosman, SAE 950283 (1995) - the two criteria,
//!     as §78.4 states them; neither is re-read here
//! No GPL-licensed source was consulted.
//!
//! Host only: no mesh, no device. The band is a physics band (precision
//! class P, docs/17 section 5.1.4), the same in both builds. The key and the
//! comparison are `f64`; the criterion is the crate's own
//! `WallImpactControls::boundary_speed`, in `Scalar`, at the shipped
//! defaults `K_crit = 57.7` and `A = 2630` - nothing is tuned to the key.

use ofgpu::parcels::{SplashCriterion, SurfaceTension, WallImpactControls, WallRegime};
use ofgpu::{Result, Scalar};

use crate::{key, Checks, GateReport, How, Uncertainty, Verdict};

/// The registry names, one per criterion: each criterion carries its own
/// verdict (docs/17's row: "per criterion"), and the 4.78 comparison keeps
/// its own name `78-D`.
pub(crate) const GATE_MUNDO: &str = "SPEC-LIT 146.2 Gate 78-D measured leg, mundo";
pub(crate) const GATE_BAI_GOSMAN: &str = "SPEC-LIT 146.2 Gate 78-D measured leg, baiGosman";

const AGAINST: &str = "Riboux & Gordillo, Phys. Rev. Lett. 113 (2014) 024507 (arXiv:1401.6943v3): \
                       the splash threshold speeds of Fig. 4(c), eight liquids on smooth dry \
                       glass in air, band +-20 % per row";

/// The answer key's manifest id (VAL-05).
const KEY_ID: &str = "riboux-gordillo-2014";

/// (146.5): the band on `e = (V_c - V*) / V*`, stated before the run.
pub(crate) const BAND: f64 = 0.20;
/// (146.6): the liquids that must be inside, of the key's eight.
pub(crate) const LIQUIDS_NEEDED: usize = 6;
/// The droplet temperature handed to the map; `SurfaceTension::Constant`
/// does not read it.
const T_DROPLET: Scalar = 293.15;

const NO_MESH: &str = "no mesh: the criterion is the closed-form inverse (78.4c), evaluated on \
                       the host for each row of the key";

/// One row of the key, in SI units, with `d = 2R` (146.2).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Row {
    pub liquid: usize,
    pub rho: f64,
    pub sigma: f64,
    pub mu: f64,
    /// The viscosity that reproduces the printed `Oh`, `Oh (rho sigma R)^(1/2)`
    /// (§146.1 check (b)); used only for the sensitivity note.
    pub mu_oh: f64,
    pub r: f64,
    pub v_star: f64,
}

pub(crate) fn rows(kf: &key::KeyFile) -> Result<Vec<Row>> {
    let (liq, rho, sig, mu, r, oh, v) = (
        kf.column("liquid")?,
        kf.column("rho")?,
        kf.column("sigma")?,
        kf.column("mu")?,
        kf.column("R")?,
        kf.column("Oh_e3")?,
        kf.column("V_star")?,
    );
    Ok((0..kf.rows.len())
        .map(|i| {
            let (s, rr) = (sig[i] * 1e-3, r[i] * 1e-3);
            Row {
                liquid: liq[i] as usize,
                rho: rho[i],
                sigma: s,
                mu: mu[i] * 1e-3,
                mu_oh: oh[i] * 1e-3 * (rho[i] * s * rr).sqrt(),
                r: rr,
                v_star: v[i],
            }
        })
        .collect())
}

/// (78.4c) through the crate's own map: the lowest normal speed at which a
/// droplet of diameter `d` splashes under `crit`, at the shipped defaults.
pub(crate) fn predicted(crit: SplashCriterion, rho: f64, sigma: f64, mu: f64, d: f64) -> f64 {
    let m = WallImpactControls {
        splash: crit,
        tension: SurfaceTension::Constant,
        sigma: sigma as Scalar,
        mu_liquid: mu as Scalar,
        ..WallImpactControls::default()
    };
    m.boundary_speed(rho as Scalar, d as Scalar, T_DROPLET, WallRegime::Splash)
        .map_or(f64::NAN, f64::from)
}

/// (146.5): a row is inside when `|e| <= BAND`; a NaN is never inside.
pub(crate) fn inside(e: f64) -> bool {
    e.abs() <= BAND
}

/// One criterion's comparison: `e` per row, and the liquids (146.6) puts
/// inside - a liquid is inside only when every one of its rows is.
pub(crate) struct Leg {
    pub e: Vec<f64>,
    pub rows_inside: usize,
    pub liquids: usize,
    pub liquids_inside: Vec<usize>,
}

pub(crate) fn leg(rows: &[Row], crit: SplashCriterion, use_mu_oh: bool) -> Leg {
    let e: Vec<f64> = rows
        .iter()
        .map(|w| {
            let mu = if use_mu_oh { w.mu_oh } else { w.mu };
            (predicted(crit, w.rho, w.sigma, mu, 2.0 * w.r) - w.v_star) / w.v_star
        })
        .collect();
    let mut ids: Vec<usize> = rows.iter().map(|w| w.liquid).collect();
    ids.dedup();
    let liquids_inside = ids
        .iter()
        .copied()
        .filter(|&l| rows.iter().zip(&e).filter(|(w, _)| w.liquid == l).all(|(_, &x)| inside(x)))
        .collect();
    Leg { rows_inside: e.iter().filter(|&&x| inside(x)).count(), e, liquids: ids.len(), liquids_inside }
}

/// Gate 78-D's measured leg. Called by `validate.rs`'s section with no gate
/// open; it opens each criterion's scope itself, so each verdict is
/// registered inside its own (S69.4).
pub(crate) fn check_splash_thresholds(c: &mut Checks) -> Result<()> {
    let legs = [(SplashCriterion::Mundo, GATE_MUNDO), (SplashCriterion::BaiGosman, GATE_BAI_GOSMAN)];
    let kf = match key::load(KEY_ID) {
        Ok(kf) => kf,
        Err(why) => {
            let why = why.to_string();
            c.note(&why);
            for (_, gate) in legs {
                c.enter_gate(gate);
                c.skip("Gate 78-D measured leg: V_c against Riboux & Gordillo's V*", "the answer key is absent");
                c.report(GateReport {
                    verdict: Verdict::Open,
                    how: How::Live,
                    gate,
                    against: AGAINST,
                    headline: "the answer key is absent from reference/, so no comparison was run".to_string(),
                    detail: vec![why.clone()],
                    uncertainty: Some(Uncertainty::SingleMesh(NO_MESH)),
                });
                c.leave_gate();
            }
            return Ok(());
        }
    };
    c.note(&kf.digest_line());
    let rows = rows(&kf)?;

    for (crit, gate) in legs {
        c.enter_gate(gate);
        let l = leg(&rows, crit, false);
        let alt = leg(&rows, crit, true);
        c.note(&format!(
            "{}: liquid   R mm    V* m/s   V_c m/s      e     (146.5) band +-{:.0} %",
            crit.name(),
            100.0 * BAND
        ));
        for (w, &e) in rows.iter().zip(&l.e) {
            c.note(&format!(
                "           {}    {:5.2}   {:6.3}   {:6.3}   {:+7.3}  {}",
                w.liquid,
                1e3 * w.r,
                w.v_star,
                w.v_star * (1.0 + e),
                e,
                if inside(e) { "inside" } else { "outside" }
            ));
        }
        let shift = l
            .e
            .iter()
            .zip(&alt.e)
            .fold(0.0f64, |m, (a, b)| m.max(((1.0 + b) / (1.0 + a) - 1.0).abs()));
        c.note(&format!(
            "{}: with mu_Oh in place of the printed mu, V_c moves by at most {:.1} % and {} of {} \
             liquids are inside (against {} with the printed mu)",
            crit.name(),
            100.0 * shift,
            alt.liquids_inside.len(),
            alt.liquids,
            l.liquids_inside.len()
        ));
        c.require(
            &format!("78-D leg ({}): 17 rows, 8 liquids, V_c > 0", crit.name()),
            rows.len() == 17 && l.liquids == 8 && l.e.iter().all(|e| e.is_finite() && *e > -1.0),
        );
        let (lo, hi) = l.e.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &e| (a.min(e), b.max(e)));
        if l.liquids_inside.len() >= LIQUIDS_NEEDED {
            c.check_d(
                &format!("78-D leg ({}): liquids outside the band (146.6)", crit.name()),
                (l.liquids - l.liquids_inside.len()) as Scalar,
                (l.liquids - LIQUIDS_NEEDED) as Scalar,
            );
        } else {
            c.report(GateReport {
                verdict: Verdict::Misses,
                how: How::Live,
                gate,
                against: AGAINST,
                headline: format!(
                    "{} of {} liquids inside the band (146.6), {} needed; {} of {} rows inside \
                     (146.5), e from {lo:+.3} to {hi:+.3}",
                    l.liquids_inside.len(),
                    l.liquids,
                    LIQUIDS_NEEDED,
                    l.rows_inside,
                    rows.len()
                ),
                detail: vec![
                    format!(
                        "liquids inside: {:?} (1 to 8 = acetone, water, methanol, ethanol, \
                         decamethyltetrasiloxane, dodecamethylpentasiloxane, PDMS, 10 cP \
                         silicone oil)",
                        l.liquids_inside
                    ),
                    "the criterion is the shipped default (K_crit = 57.7, A = 2630), with each \
                     row's printed rho, sigma, mu and d = 2R; nothing is tuned to the key, and \
                     neither criterion of SPEC-LIT 78.4 contains a gas property, while the \
                     paper's own model does (its Table II)"
                        .to_string(),
                ],
                uncertainty: Some(Uncertainty::SingleMesh(NO_MESH)),
            });
        }
        c.leave_gate();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key_rows() -> Vec<Row> {
        rows(&key::load(KEY_ID).expect("load the answer key")).expect("the key's columns")
    }

    #[test]
    fn splash_band_is_twenty_percent_and_refuses_nan() {
        assert!(inside(0.2) && inside(-0.2) && inside(0.0));
        assert!(!inside(0.2001) && !inside(-0.2001) && !inside(f64::NAN));
    }

    #[test]
    fn splash_threshold_speeds_against_riboux_gordillo() {
        let rows = key_rows();
        let at = |liquid: usize, r_mm: f64| {
            rows.iter().find(|w| w.liquid == liquid && (1e3 * w.r - r_mm).abs() < 1e-9).expect("the row")
        };
        let v = |crit, w: &Row| predicted(crit, w.rho, w.sigma, w.mu, 2.0 * w.r);
        // (78.4c) with d = 2R (146.2), from the scratch oracle of VAL-70.
        let eth = at(4, 1.04);
        assert!((v(SplashCriterion::Mundo, eth) - 1.0508).abs() < 1e-3);
        assert!((v(SplashCriterion::BaiGosman, eth) - 2.3347).abs() < 1e-3);
        let water = at(2, 1.96);
        assert!((v(SplashCriterion::Mundo, water) - 0.9794).abs() < 1e-3);
        assert!((v(SplashCriterion::BaiGosman, water) - 2.2230).abs() < 1e-3);

        let m = leg(&rows, SplashCriterion::Mundo, false);
        assert_eq!((m.rows_inside, m.liquids, m.liquids_inside.len()), (0, 8, 0));
        let b = leg(&rows, SplashCriterion::BaiGosman, false);
        assert_eq!((b.rows_inside, b.liquids), (3, 8));
        assert_eq!(b.liquids_inside, vec![4]);
        // The verdict does not hinge on the printed viscosity's rounding.
        assert_eq!(leg(&rows, SplashCriterion::Mundo, true).liquids_inside.len(), 0);
        assert_eq!(leg(&rows, SplashCriterion::BaiGosman, true).liquids_inside, vec![4]);
    }
}
