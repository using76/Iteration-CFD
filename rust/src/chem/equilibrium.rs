// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Gas-phase chemical equilibrium by Gibbs minimisation (SPEC-LIT §131):
//! the element-potential iteration after **Gordon & McBride**, NASA RP-1311
//! (1994), eqs. (2.7)-(2.29) and §§3.1-3.3. The reduced Newton system
//! (131.3)-(131.6) is solved for `pi_i = -lambda_i/(RT)`, `dln n` and, for
//! an assigned enthalpy, `dln T`; RP-1311's trace-species rule
//! (`SIZE = 18.420681`, §3.2) is phase 1, and phase 2 repeats the same
//! iteration over every active species. Host only, and f64 in every build:
//! no `Scalar`, no device types, so the module behaves identically in the
//! default build and in `--features single`.
//! No GPL-licensed source was consulted.

use crate::error::Error;

use super::thermo::{ThermoSet, R_GAS};

/// `SIZE` of §131.3: a species whose `ln(n_j/n) < -SIZE` is trace in
/// phase 1.
const SIZE: f64 = 18.420681;
/// `-ln 1e-4` of (131.8): a growing trace species is held at `n_j/n = 1e-4`.
const TRACE_CAP: f64 = 9.2103404;

/// Iteration controls (SPEC-LIT §131.3). Default: `max_iterations` 200,
/// `t_start` 3800.0 K.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EqOptions {
    /// The most Newton iterations the two phases may take together; a run
    /// that needs more is refused and no state is returned (§131.5).
    pub max_iterations: usize,
    /// The starting temperature of an assigned-enthalpy run (RP-1311 §3.1),
    /// capped at the active species' common `T_hi` (§131.3 D3).
    pub t_start: f64,
}

impl Default for EqOptions {
    fn default() -> Self {
        EqOptions {
            max_iterations: 200,
            t_start: 3800.0,
        }
    }
}

/// A converged equilibrium state, per kg of mixture (SPEC-LIT §131.1).
#[derive(Clone, Debug, PartialEq)]
pub struct Equilibrium {
    /// K.
    pub t: f64,
    /// Pa.
    pub p: f64,
    /// Upper-case element symbols, in first-appearance order over
    /// `set.species()`, each record's own element order.
    pub elements: Vec<String>,
    /// Assigned element moles b_i°, mol/kg, in `elements` order.
    pub b0: Vec<f64>,
    /// n_j, mol/kg, in `set.species()` order; exactly 0.0 for a species
    /// with an absent element (§131.2 D1).
    pub n: Vec<f64>,
    /// The iterated unknown `n` of (131.4), mol/kg.
    pub n_total: f64,
    /// `pi_i = -lambda_i/(RT)`, in `elements` order; exactly 0.0 for an
    /// absent element.
    pub pi: Vec<f64>,
    /// Newton iterations taken, both phases together.
    pub iterations: usize,
}

impl Equilibrium {
    /// X_j = n_j / sum_j n_j, in `set.species()` order.
    pub fn mole_fractions(&self) -> Vec<f64> {
        let s: f64 = self.n.iter().sum();
        self.n.iter().map(|nj| nj / s).collect()
    }

    /// Y_j = n_j W_j, in `set.species()` order; the sum is 1 kg.
    pub fn mass_fractions(&self, set: &ThermoSet) -> Vec<f64> {
        self.n
            .iter()
            .zip(set.species())
            .map(|(nj, s)| nj * s.molar_mass)
            .collect()
    }

    /// c_k = p X_k / (R_GAS T), mol/m^3, in `set.species()` order.
    pub fn concentrations(&self) -> Vec<f64> {
        let s: f64 = self.n.iter().sum();
        self.n
            .iter()
            .map(|nj| self.p * (nj / s) / (R_GAS * self.t))
            .collect()
    }
}

/// The assigned element moles, `b_i° = sum_j a_ij y_j / W_j / sum_j y_j`
/// mol/kg with `y` normalised by its sum, and the elements in
/// first-appearance order over `set.species()`, each record's own order.
/// `y` is checked by [`ThermoSet::molar_mass_mix`] first, and its errors
/// pass through unchanged (SPEC-LIT §131.5).
pub fn element_moles(set: &ThermoSet, y: &[f64]) -> crate::Result<(Vec<String>, Vec<f64>)> {
    set.molar_mass_mix(y)?;
    let sp = set.species();
    let mut elements: Vec<String> = Vec::new();
    for s in sp {
        for (sym, _) in &s.elements {
            if !elements.iter().any(|e| e == sym) {
                elements.push(sym.clone());
            }
        }
    }
    let sum_y: f64 = y.iter().sum();
    let mut b0 = vec![0.0f64; elements.len()];
    for (j, s) in sp.iter().enumerate() {
        let yj = y[j] / sum_y;
        for (sym, cnt) in &s.elements {
            let i = elements
                .iter()
                .position(|e| e == sym)
                .expect("the element was collected above");
            b0[i] += *cnt as f64 * yj / s.molar_mass;
        }
    }
    Ok((elements, b0))
}

/// The §131.5 option refusals shared by the two entry points.
fn checked_options(opts: &EqOptions) -> crate::Result<()> {
    if opts.max_iterations == 0 {
        return Err(Error::Config(
            "the equilibrium needs at least one Newton iteration, given \
             max_iterations = 0 (SPEC-LIT §131.5)"
                .to_string(),
        ));
    }
    if !opts.t_start.is_finite() || opts.t_start <= 0.0 {
        return Err(Error::Config(format!(
            "the equilibrium starting temperature t_start = {} K is not a \
             finite temperature above zero (SPEC-LIT §131.5)",
            opts.t_start
        )));
    }
    Ok(())
}

/// Assigned (T, p): (131.3)+(131.4) with dln T = 0, RP-1311's "tp" problem
/// (SPEC-LIT §131.2). The elements come from `y0`; (131.2)'s pressure term
/// is `ln(p / set.p_ref())`.
pub fn equilibrate_tp(
    set: &ThermoSet,
    t: f64,
    p: f64,
    y0: &[f64],
    opts: &EqOptions,
) -> crate::Result<Equilibrium> {
    if !t.is_finite() || t <= 0.0 {
        return Err(Error::Config(format!(
            "the equilibrium temperature T = {t} K is not a finite temperature \
             above zero (SPEC-LIT §131.5)"
        )));
    }
    if !p.is_finite() || p <= 0.0 {
        return Err(Error::Config(format!(
            "the equilibrium pressure p = {p} Pa is not a finite pressure above \
             zero (SPEC-LIT §131.5)"
        )));
    }
    checked_options(opts)?;
    Solve::new(set, false, 0.0, p, y0)?.run(t, opts)
}

/// Assigned (h, p): (131.3)+(131.4)+(131.5) (SPEC-LIT §131.2), with `h0`
/// the reactants' specific enthalpy in J/kg, formed by the caller (e.g.
/// `set.h_mass(298.15, y0)`, RP-1311 eq. (2.13a)). The starting temperature
/// is `min(t_start, T_hi)` over the active species (§131.3 D3).
pub fn equilibrate_hp(
    set: &ThermoSet,
    h0: f64,
    p: f64,
    y0: &[f64],
    opts: &EqOptions,
) -> crate::Result<Equilibrium> {
    if !p.is_finite() || p <= 0.0 {
        return Err(Error::Config(format!(
            "the equilibrium pressure p = {p} Pa is not a finite pressure above \
             zero (SPEC-LIT §131.5)"
        )));
    }
    if !h0.is_finite() {
        return Err(Error::Config(format!(
            "the assigned enthalpy h0 = {h0} J/kg is not finite (SPEC-LIT §131.5)"
        )));
    }
    checked_options(opts)?;
    let solve = Solve::new(set, true, h0, p, y0)?;
    let t0 = opts.t_start.min(solve.t_hi);
    solve.run(t0, opts)
}

/// The per-problem data of one equilibrium run: the element table, the
/// active species (§131.2 D1), and the last iteration's diagnostics.
struct Solve<'a> {
    set: &'a ThermoSet,
    /// Element symbols, first appearance over `set.species()`.
    elements: Vec<String>,
    /// b_i°, mol/kg, in `elements` order.
    b0: Vec<f64>,
    /// a_ij, the count of element i in species j.
    a: Vec<Vec<f64>>,
    /// The active species' indices into `set.species()`.
    act: Vec<usize>,
    /// Each active element's index in `elements`.
    el_act: Vec<usize>,
    /// True for the assigned-enthalpy problem, whose system carries (131.5).
    hp: bool,
    /// J/kg, the assigned specific enthalpy (hp only).
    h0: f64,
    /// Pa, the assigned pressure.
    p: f64,
    /// `ln(p / p_ref)` of (131.2).
    ln_pr: f64,
    /// The active species' common temperature range (§131.3 D3).
    t_lo: f64,
    t_hi: f64,
    /// The last solve's pi, in active-element order.
    last_pi: Vec<f64>,
    /// The last iteration's element residual and max |dln n_j|, named by
    /// the §131.5 refusal.
    last_res: f64,
    last_dln: f64,
}

impl<'a> Solve<'a> {
    fn new(set: &'a ThermoSet, hp: bool, h0: f64, p: f64, y0: &[f64]) -> crate::Result<Solve<'a>> {
        let (elements, b0) = element_moles(set, y0)?;
        let sp = set.species();
        let index = |sym: &str| {
            elements
                .iter()
                .position(|e| e == sym)
                .expect("element known")
        };
        let mut a = vec![vec![0.0f64; sp.len()]; elements.len()];
        for (j, s) in sp.iter().enumerate() {
            for (sym, cnt) in &s.elements {
                a[index(sym)][j] = *cnt as f64;
            }
        }
        // D1: an element with b_i° == 0 is inactive, and so is every species
        // containing one; the rest are active.
        let active_el: Vec<bool> = b0.iter().map(|b| *b > 0.0).collect();
        let act: Vec<usize> = (0..sp.len())
            .filter(|&j| sp[j].elements.iter().all(|(sym, _)| active_el[index(sym)]))
            .collect();
        let el_act: Vec<usize> = (0..elements.len()).filter(|&i| active_el[i]).collect();
        let t_lo = act
            .iter()
            .map(|&j| sp[j].t_low)
            .fold(f64::NEG_INFINITY, f64::max);
        let t_hi = act
            .iter()
            .map(|&j| sp[j].t_high)
            .fold(f64::INFINITY, f64::min);
        Ok(Solve {
            set,
            elements,
            b0,
            a,
            act,
            el_act,
            hp,
            h0,
            p,
            ln_pr: (p / set.p_ref()).ln(),
            t_lo,
            t_hi,
            last_pi: Vec::new(),
            last_res: 0.0,
            last_dln: 0.0,
        })
    }
}

/// The iteration state of §131.3: `ln n_j` per active species, the iterated
/// `ln n`, and the temperature, whose log carries the hp update.
struct State {
    lnnj: Vec<f64>,
    lnn: f64,
    t: f64,
    lnt: f64,
}

impl Solve<'_> {
    /// The two phases of §131.3 from the start state: phase 1 with
    /// RP-1311's trace rule, then phase 2 over every active species. The
    /// budget is shared; an exhausted one is a §131.5 refusal.
    fn run(mut self, t0: f64, opts: &EqOptions) -> crate::Result<Equilibrium> {
        let ns = self.act.len();
        // D2: the start is n = 100 mol/kg, n_j = 100/NS.
        let mut st = State {
            lnnj: vec![(100.0 / ns as f64).ln(); ns],
            lnn: 100.0f64.ln(),
            t: t0,
            lnt: t0.ln(),
        };
        let mut iterations = 0usize;
        for phase in [1, 2] {
            loop {
                if iterations >= opts.max_iterations {
                    return Err(Error::Config(format!(
                        "the {} equilibrium did not converge in {} iterations \
                         (last element residual {:.3e}, max |dln n_j| {:.3e}) \
                         (SPEC-LIT §131.5)",
                        if self.hp { "hp" } else { "tp" },
                        opts.max_iterations,
                        self.last_res,
                        self.last_dln
                    )));
                }
                iterations += 1;
                if self.step(&mut st, phase)? {
                    break;
                }
            }
        }
        // The full vectors: an inactive species' n_j and an absent element's
        // pi are exactly 0.0 (D1).
        let mut n = vec![0.0f64; self.set.species().len()];
        for (k, &j) in self.act.iter().enumerate() {
            n[j] = st.lnnj[k].exp();
        }
        let mut pi = vec![0.0f64; self.elements.len()];
        for (k, &i) in self.el_act.iter().enumerate() {
            pi[i] = self.last_pi[k];
        }
        Ok(Equilibrium {
            t: st.t,
            p: self.p,
            elements: self.elements,
            b0: self.b0,
            n,
            n_total: st.lnn.exp(),
            pi,
            iterations,
        })
    }

    /// One Newton iteration at `phase` (1 with the trace rule, 2 over every
    /// active species): evaluate at the current T, build (131.3)-(131.5),
    /// solve, test §131.3's convergence on the corrections, damp by
    /// (131.7)-(131.8) and apply. True when the phase converged (the step
    /// then went with lambda = 1).
    fn step(&mut self, st: &mut State, phase: usize) -> crate::Result<bool> {
        let t = st.t;
        let ns = self.act.len();
        let ke = self.el_act.len();
        let sp = self.set.species();
        // g_j, h_rt and cp_r at the current T; a T outside a record's range
        // is that record's own refusal (SPEC-LIT §131.5).
        let mut g = vec![0.0f64; ns];
        let mut h = vec![0.0f64; ns];
        let mut cp = vec![0.0f64; ns];
        for (k, &j) in self.act.iter().enumerate() {
            g[k] = sp[j].g_rt(t)?;
            h[k] = sp[j].h_rt(t)?;
            cp[k] = sp[j].cp_r(t)?;
        }
        let nj: Vec<f64> = st.lnnj.iter().map(|v| v.exp()).collect();
        let n = st.lnn.exp();
        // mu_j/(RT) of (131.2); ln(n_j/n) stays a difference of the state's
        // logs, so a trace species never rounds through an exp/log pair.
        let mu: Vec<f64> = (0..ns)
            .map(|k| g[k] + (st.lnnj[k] - st.lnn) + self.ln_pr)
            .collect();
        // Phase 1's sums and lambda1 run over the non-trace species,
        // phase 2's over every active one.
        let sums: Vec<bool> = (0..ns)
            .map(|k| phase == 2 || st.lnnj[k] - st.lnn >= -SIZE)
            .collect();

        // The sums of (131.3)-(131.5) over the phase's species.
        let mut bi = vec![0.0f64; ke];
        let mut hki = vec![0.0f64; ke];
        let (mut sum_n, mut sum_mu, mut sum_h) = (0.0, 0.0, 0.0);
        let (mut sum_cp, mut sum_h2, mut sum_hmu) = (0.0, 0.0, 0.0);
        for (k, &j) in self.act.iter().enumerate() {
            if !sums[k] {
                continue;
            }
            let (nk, nk_h, nk_mu) = (nj[k], nj[k] * h[k], nj[k] * mu[k]);
            sum_n += nk;
            sum_mu += nk_mu;
            sum_h += nk_h;
            sum_cp += nk * cp[k];
            sum_h2 += nk_h * h[k];
            sum_hmu += nk_h * mu[k];
            for (i, &ki) in self.el_act.iter().enumerate() {
                let aij = self.a[ki][j];
                if aij != 0.0 {
                    bi[i] += aij * nk;
                    hki[i] += aij * nk_h;
                }
            }
        }
        // The matrix of (131.3)-(131.5): one row per active element, then
        // (131.4), then (131.5) when the enthalpy is assigned.
        let d = ke + 1 + usize::from(self.hp);
        let mut mat = vec![vec![0.0f64; d]; d];
        let mut rhs = vec![0.0f64; d];
        let mut brs = vec![0.0f64; ke];
        for (r, &kr) in self.el_act.iter().enumerate() {
            let ar = &self.a[kr];
            let (mut br, mut gr, mut hr) = (0.0, 0.0, 0.0);
            for (k, &j) in self.act.iter().enumerate() {
                if !sums[k] {
                    continue;
                }
                let ark = ar[j];
                if ark == 0.0 {
                    continue;
                }
                let w = ark * nj[k];
                br += w;
                gr += w * mu[k];
                hr += w * h[k];
                for (i, &ki) in self.el_act.iter().enumerate() {
                    let aik = self.a[ki][j];
                    if aik != 0.0 {
                        mat[r][i] += w * aik;
                    }
                }
            }
            brs[r] = br;
            mat[r][ke] = br;
            if self.hp {
                mat[r][ke + 1] = hr;
            }
            rhs[r] = self.b0[kr] - br + gr;
        }

        for (i, v) in bi.iter().enumerate() {
            mat[ke][i] = *v;
        }
        mat[ke][ke] = sum_n - n;
        rhs[ke] = n - sum_n + sum_mu;
        if self.hp {
            mat[ke][ke + 1] = sum_h;
            for (i, v) in hki.iter().enumerate() {
                mat[ke + 1][i] = *v;
            }
            mat[ke + 1][ke] = sum_h;
            // (131.5)'s diagonal and right side, with
            // (h_0 - h)/(RT) = h0/(R_GAS T) - sum_j n_j h_rt_j.
            mat[ke + 1][ke + 1] = sum_cp + sum_h2;
            rhs[ke + 1] = self.h0 / (R_GAS * t) - sum_h + sum_hmu;
        }
        let x = solve_linear(&mut mat, &mut rhs)?;
        let dln_n = x[ke];
        let dln_t = if self.hp { x[ke + 1] } else { 0.0 };
        self.last_pi = x[..ke].to_vec();
        // (131.6) for EVERY active species, the trace ones too.
        let dlnj: Vec<f64> = (0..ns)
            .map(|k| {
                let j = self.act[k];
                let mut s = dln_n + h[k] * dln_t - mu[k];
                for (i, &ki) in self.el_act.iter().enumerate() {
                    let aij = self.a[ki][j];
                    if aij != 0.0 {
                        s += aij * x[i];
                    }
                }
                s
            })
            .collect();
        // §131.3's convergence test, on the corrections before they are
        // applied: the element residual, and the phase's |dln| set.
        let bmax = self.b0.iter().cloned().fold(0.0f64, f64::max);
        let res = (0..ke)
            .map(|r| (self.b0[self.el_act[r]] - brs[r]).abs())
            .fold(0.0f64, f64::max);
        let dln_max = (0..ns)
            .filter(|&k| sums[k])
            .map(|k| dlnj[k].abs())
            .fold(0.0f64, f64::max);
        self.last_res = res;
        self.last_dln = dln_max;
        let converged = res <= 1e-14 * bmax
            && dln_max <= 1e-12
            && dln_n.abs() <= 1e-12
            && (!self.hp || dln_t.abs() <= 1e-12);

        // (131.7)-(131.8): lambda = min(1, lambda1, lambda2), lambda2 only
        // in phase 1. lambda1 = +inf at a zero max.
        let lambda = if converged {
            1.0
        } else {
            let mut m1 = 5.0 * dln_t.abs().max(dln_n.abs());
            for k in 0..ns {
                if sums[k] {
                    m1 = m1.max(dlnj[k].abs());
                }
            }
            let l1 = if m1 > 0.0 { 2.0 / m1 } else { f64::INFINITY };
            let mut l2 = f64::INFINITY;
            if phase == 1 {
                for k in 0..ns {
                    if !sums[k] && dlnj[k] >= 0.0 {
                        let v = ((-(st.lnnj[k] - st.lnn) - TRACE_CAP) / (dlnj[k] - dln_n)).abs();
                        l2 = l2.min(v);
                    }
                }
            }
            1.0_f64.min(l1).min(l2)
        };
        for k in 0..ns {
            st.lnnj[k] += lambda * dlnj[k];
        }
        st.lnn += lambda * dln_n;
        if self.hp {
            st.lnt += lambda * dln_t;
            // D3: T is clamped into the active species' common range.
            st.t = st.lnt.exp().clamp(self.t_lo, self.t_hi);
            st.lnt = st.t.ln();
        }
        Ok(converged)
    }
}

/// Gaussian elimination with partial pivoting on the augmented system; a
/// zero or non-finite pivot is a singular system (SPEC-LIT §131.5).
fn solve_linear(a: &mut [Vec<f64>], b: &mut [f64]) -> crate::Result<Vec<f64>> {
    let n = b.len();
    for c in 0..n {
        let mut p = c;
        for r in c + 1..n {
            if a[r][c].abs() > a[p][c].abs() {
                p = r;
            }
        }
        if !a[p][c].is_finite() || a[p][c] == 0.0 {
            return Err(Error::Config(format!(
                "the equilibrium correction system is singular: the pivot of \
                 column {c} is {} (SPEC-LIT §131.5)",
                a[p][c]
            )));
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..n {
            let f = a[r][c] / a[c][c];
            if f != 0.0 {
                for cc in c..n {
                    a[r][cc] -= f * a[c][cc];
                }
                b[r] -= f * b[c];
            }
        }
    }
    let mut x = vec![0.0f64; n];
    for r in (0..n).rev() {
        let mut s = b[r];
        for c in r + 1..n {
            s -= a[r][c] * x[c];
        }
        x[r] = s / a[r][r];
    }
    Ok(x)
}
