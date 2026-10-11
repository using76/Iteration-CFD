// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Finite-rate kinetics: the rate of progress, third bodies, fall-off, and
//! the reverse rate from equilibrium (SPEC-LIT §130), after SAND89-8009
//! Chapter II eqs. (48)-(70). Host only, and f64 in every build, so that
//! CHR-07a's device kernel can mirror the design choices of §130.7 one for
//! one: the reverse term is one exponential of a sum of logs (130.16),
//! never `exp(ln k_f) / exp(ln K_c)`. `omega` is production, CHEMKIN's sign
//! convention (§130.1). No GPL-licensed source was consulted.

use super::mechanism::{Arrhenius, Blending, Collider, Mechanism, Reaction, ThirdBody};
use super::thermo::R_GAS;
use crate::Error;

/// The forward and reverse terms of one reaction's rate of progress,
/// `q_f` and `q_r` of (130.14) and (130.16)/(130.17), in mol/(m^3 s);
/// [`Progress::net`] is (130.2)'s `q_i = q_f - q_r`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Progress {
    pub forward: f64,
    pub reverse: f64,
}

impl Progress {
    /// (130.2)'s `q_i`, the net rate of progress.
    pub fn net(&self) -> f64 {
        self.forward - self.reverse
    }
}

/// (130.4), SAND89-8009 eq. (52): `k = A T^beta exp(-theta/T)`, with `a`,
/// `beta` and `theta` already SI (SPEC-LIT §130.2).
pub fn arrhenius(r: &Arrhenius, t: f64) -> f64 {
    r.a * t.powf(r.beta) * (-r.theta / t).exp()
}

/// The log of (130.4)'s magnitude, `ln|A| + beta ln T - theta/T`, the core
/// of (130.15); the reverse term exponentiates one such log sum.
pub fn ln_arrhenius(r: &Arrhenius, t: f64) -> f64 {
    r.a.abs().ln() + r.beta * t.ln() - r.theta / t
}

/// (130.11): `F_cent = (1 - a) exp(-T/T***) + a exp(-T/T*) [+ exp(-T**/T)]`.
/// Without `T**` the last term is absent (SAND89-8009 p. 23, §130.4).
pub fn troe_f_cent(a: f64, t3: f64, t1: f64, t2: Option<f64>, t: f64) -> f64 {
    let mut f_cent = (1.0 - a) * (-t / t3).exp() + a * (-t / t1).exp();
    if let Some(t2) = t2 {
        f_cent += (-t2 / t).exp();
    }
    f_cent
}

/// (130.9)-(130.10) with `d = 0.14`; `pr > 0`, because `Pr = 0` never
/// reaches a blending factor (§130.4).
pub fn troe_f(f_cent: f64, pr: f64) -> f64 {
    let log_pr = pr.log10();
    let log_fc = f_cent.log10();
    let c = -0.4 - 0.67 * log_fc;
    let n = 0.75 - 1.27 * log_fc;
    let x = log_pr + c;
    10f64.powf(log_fc / (1.0 + (x / (n - 0.14 * x)).powi(2)))
}

/// (130.12), SAND89-8009 eqs. (69)-(70): `F_SRI = d [a exp(-b/T) +
/// exp(-T/c)]^X T^e` with `X = 1/(1 + log10(Pr)^2)`; `pr > 0` (§130.4).
pub fn sri_f(a: f64, b: f64, c: f64, d: f64, e: f64, t: f64, pr: f64) -> f64 {
    let x = 1.0 / (1.0 + pr.log10().powi(2));
    d * (a * (-b / t).exp() + (-t / c).exp()).powf(x) * t.powf(e)
}

/// (130.8)'s blending factor `F` (§130.4): Lindemann's `F = 1`, else Troe's
/// (130.9)-(130.11) or SRI's (130.12) at the given `T` and `Pr`.
pub fn blending_f(b: &Blending, t: f64, pr: f64) -> f64 {
    match b {
        Blending::Lindemann => 1.0,
        Blending::Troe { a, t3, t1, t2 } => troe_f(troe_f_cent(*a, *t3, *t1, *t2, t), pr),
        Blending::Sri { a, b, c, d, e } => sri_f(*a, *b, *c, *d, *e, t, pr),
    }
}

/// (130.3): `[M]_i = sum_k alpha_ki [X_k]`, with `alpha = 1` for every
/// species not listed among the reaction's efficiencies (§130.1).
pub fn third_body_concentration(tb: &ThirdBody, conc: &[f64]) -> f64 {
    conc.iter()
        .enumerate()
        .map(|(k, c)| {
            let alpha = tb
                .efficiencies
                .iter()
                .find(|(j, _)| *j == k)
                .map_or(1.0, |(_, a)| *a);
            alpha * c
        })
        .sum()
}

/// The `[M]` of (130.8) for a fall-off reaction's collider (§130.4): a
/// `(+M)` mixture carries the reaction's efficiencies (130.3), while a
/// `(+SPECIES)` collider takes that species' concentration alone
/// (SAND89-8009 eq. (63) and footnote).
pub fn collider_concentration(c: &Collider, conc: &[f64]) -> f64 {
    match c {
        Collider::Mixture(tb) => third_body_concentration(tb, conc),
        Collider::Species(j) => conc[*j],
    }
}

/// The §130.7 input refusals, each `Error::Config` message ending
/// "(SPEC-LIT §130.7)": `T` not finite or not above 0, a concentration
/// vector whose length is not the species count (both lengths named), or a
/// negative or non-finite `c_k` (the species and the value named). A stiff
/// integrator clips its undershoot before it calls; the host twin does not.
fn checked(t: f64, conc: &[f64], names: &[String]) -> crate::Result<()> {
    if !t.is_finite() || t <= 0.0 {
        return Err(Error::Config(format!(
            "the kinetics of SPEC-LIT §130 need a temperature that is finite and above \
             zero, given T = {t} (SPEC-LIT §130.7)"
        )));
    }
    if conc.len() != names.len() {
        return Err(Error::Config(format!(
            "the kinetics of SPEC-LIT §130 were given {} concentration value(s) for {} \
             species (SPEC-LIT §130.7)",
            conc.len(),
            names.len()
        )));
    }
    for (k, c) in conc.iter().enumerate() {
        if !c.is_finite() || *c < 0.0 {
            return Err(Error::Config(format!(
                "the kinetics of SPEC-LIT §130 were given the concentration {c} for \
                 species {} (index {k}); it must be finite and not negative (SPEC-LIT §130.7)",
                names[k]
            )));
        }
    }
    Ok(())
}

/// (130.6), in log form (§130.3):
/// `ln K_ci = - sum_k nu_ki G_k(T)/(R T) + ( sum_k nu_ki ) ln(p_ref/(R T))`, with `nu` the rational stoichiometry
/// as f64, `sum nu` excluding M, and `p_ref` the thermo set's own. Only the
/// species the reaction touches get their `G/RT` evaluated, so a `T`
/// outside a range returns that record's own error, unchanged.
pub fn ln_equilibrium_constant(mech: &Mechanism, i: usize, t: f64) -> crate::Result<f64> {
    let rx = mech.reactions().get(i).ok_or_else(|| {
        Error::Config(format!(
            "ln_equilibrium_constant was given reaction index {i} of {} ({} reactions)",
            mech.source(),
            mech.reactions().len()
        ))
    })?;
    let tset = mech.thermo();
    let mut acc = 0.0;
    let mut nu_sum = 0.0;
    for (k, nu) in &rx.products {
        let n = nu.to_f64();
        acc -= n * tset.species()[*k].g_rt(t)?;
        nu_sum += n;
    }
    for (k, nu) in &rx.reactants {
        let n = nu.to_f64();
        acc += n * tset.species()[*k].g_rt(t)?;
        nu_sum -= n;
    }
    Ok(acc + nu_sum * (tset.p_ref() / (R_GAS * t)).ln())
}

/// (130.6) from the `G/RT` values `rates_of_progress` computed once for the
/// species a reversible reaction without `REV` touches (§130.7).
fn ln_kc_precomputed(rx: &Reaction, t: f64, p_ref: f64, g: &[f64]) -> f64 {
    let mut acc = 0.0;
    let mut nu_sum = 0.0;
    for (k, nu) in &rx.products {
        let n = nu.to_f64();
        acc -= n * g[*k];
        nu_sum += n;
    }
    for (k, nu) in &rx.reactants {
        let n = nu.to_f64();
        acc += n * g[*k];
        nu_sum -= n;
    }
    acc + nu_sum * (p_ref / (R_GAS * t)).ln()
}

/// One reaction's `q_f` and `q_r` (§130.7): the forward term (130.14)
/// directly, `k_f` from (130.4) times `Pr/(1+Pr) F` for fall-off and each
/// factor with `powf`; the reverse term one exponential of the log sum
/// (130.16), or (130.17) with `REV`. `ln c_k = -inf` at `c_k = 0` makes the
/// term exactly 0, and a factor whose order is 0 is skipped on both sides,
/// so `0 * ln 0` is never formed.
fn progress(r: &Reaction, t: f64, conc: &[f64], p_ref: f64, g: &[f64]) -> Progress {
    let (k_f, ln_k_f, c_i) = if let Some(fo) = &r.falloff {
        // Fall-off (130.8): C_i = 1, [M] already sits in Pr (§130.4). With
        // Pr = 0 (no collider present) q_f = q_r = 0 exactly, and log10 Pr
        // is not evaluated.
        let m = collider_concentration(&fo.collider, conc);
        let pr = arrhenius(&fo.low, t) * m / arrhenius(&r.rate, t);
        if pr == 0.0 {
            return Progress {
                forward: 0.0,
                reverse: 0.0,
            };
        }
        // An infinite Pr (k_inf = 0) takes the high-pressure limit Pr/(1+Pr) = 1.
        let p_fac = if pr.is_finite() { pr / (1.0 + pr) } else { 1.0 };
        let f = blending_f(&fo.blending, t, pr);
        let k_f = arrhenius(&r.rate, t) * p_fac * f;
        let ln_k_f = ln_arrhenius(&r.rate, t) + p_fac.ln() + f.ln();
        (k_f, ln_k_f, 1.0)
    } else {
        let c_i = r
            .third_body
            .as_ref()
            .map_or(1.0, |tb| third_body_concentration(tb, conc));
        (arrhenius(&r.rate, t), ln_arrhenius(&r.rate, t), c_i)
    };
    let mut q_f = c_i * k_f;
    for (k, o) in r.forward_orders() {
        if o == 0.0 {
            continue;
        }
        q_f *= conc[k].powf(o);
    }
    let q_r = if !r.reversible {
        0.0
    } else {
        let mut ln_c = 0.0;
        for (k, o) in r.reverse_orders() {
            if o == 0.0 {
                continue;
            }
            ln_c += o * conc[k].ln();
        }
        match &r.reverse {
            // (130.17): the explicit reverse Arrhenius, one exponential.
            Some(rev) => c_i * rev.a.signum() * (ln_arrhenius(rev, t) + ln_c).exp(),
            // (130.16): one exponential of the whole log sum; never
            // exp(ln k_f) / exp(ln K_c) (§130.3).
            None => {
                c_i * r.rate.a.signum() * (ln_k_f - ln_kc_precomputed(r, t, p_ref, g) + ln_c).exp()
            }
        }
    };
    Progress {
        forward: q_f,
        reverse: q_r,
    }
}

/// (130.2) for every reaction, in mechanism order: `q_i = C_i [ k_f
/// prod [X]^o' - k_r prod [X]^o'' ]`, one [`Progress`] per reaction
/// (§130.1). `G/RT` is computed once per call, and only when some reaction
/// is reversible without `REV` - that is when a `T` outside a species'
/// thermo range is refused by the thermo record's own message (§130.7).
pub fn rates_of_progress(mech: &Mechanism, t: f64, conc: &[f64]) -> crate::Result<Vec<Progress>> {
    checked(t, conc, mech.species())?;
    let tset = mech.thermo();
    let p_ref = tset.p_ref();
    let n = mech.species().len();
    let mut g = vec![0.0f64; n];
    if mech
        .reactions()
        .iter()
        .any(|r| r.reversible && r.reverse.is_none())
    {
        let mut needed = vec![false; n];
        for r in mech.reactions() {
            if r.reversible && r.reverse.is_none() {
                for (k, _) in r.products.iter().chain(r.reactants.iter()) {
                    needed[*k] = true;
                }
            }
        }
        for (k, need) in needed.iter().enumerate() {
            if *need {
                g[k] = tset.species()[k].g_rt(t)?;
            }
        }
    }
    Ok(mech
        .reactions()
        .iter()
        .map(|r| progress(r, t, conc, p_ref, &g))
        .collect())
}

/// (130.1), the molar production rate of every species in mol/(m^3 s),
/// accumulated reaction by reaction in mechanism order: `-nu' q` to each
/// reactant and `+nu'' q` to each product, with `q = q_f - q_r` and `nu`
/// the exact rational of §129.4 converted once to f64 (§130.7). CHEMKIN's
/// sign convention: production, positive (§130.1).
pub fn production_rates(mech: &Mechanism, t: f64, conc: &[f64]) -> crate::Result<Vec<f64>> {
    let q = rates_of_progress(mech, t, conc)?;
    let mut omega = vec![0.0f64; mech.species().len()];
    for (r, q_i) in mech.reactions().iter().zip(&q) {
        let net = q_i.net();
        for (k, nu) in &r.reactants {
            omega[*k] -= nu.to_f64() * net;
        }
        for (k, nu) in &r.products {
            omega[*k] += nu.to_f64() * net;
        }
    }
    Ok(omega)
}

/// (130.1) from mass fractions: `conc = mech.thermo().concentrations(p, t,
/// y)?` (128.9), then [`production_rates`] (§130.7).
pub fn production_rates_tpy(
    mech: &Mechanism,
    t: f64,
    p: f64,
    y: &[f64],
) -> crate::Result<Vec<f64>> {
    let conc = mech.thermo().concentrations(p, t, y)?;
    production_rates(mech, t, &conc)
}
