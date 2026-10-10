// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The precision constants and stop rules of SPEC-LIT §118.4, the
//! mixed-precision iterative refinement that carries a `single` build's
//! Krylov solution and residual in double while the unchanged float method
//! computes only a scaled correction.
//!
//! ORIGINAL - the host-side arithmetic of the outer loop's targets and stop
//! rules, exactly equations (118.4.4), (118.4.6) and (118.4.8) of §118.4.
//! Pure host logic, no GPU, compiled in BOTH builds: in the f64 build
//! nothing calls it, and the constants are unused there by design.
//!
//! No GPL-licensed source was consulted.

/// The floor of the inner tolerance `tau_k` (118.4.6): a correction solve
/// never works harder than a `1e-4` relative reduction of its own residual,
/// so a step whose outer target is already close costs one cheap solve.
pub const IR_INNER_REL: f64 = 1e-4;

/// The stagnation limit of (118.4.8): when the residual GROWS past half of
/// the previous step's, the float correction no longer contracts (`k·u32`
/// is near 1) and refinement cannot help.
pub const IR_STAGNATION: f64 = 0.5;

/// The outer-step budget of (118.4.8): twelve refinement steps reach the
/// working-precision limit of any system worth refining (Carson & Higham
/// 2018: a 1e-14 target takes about four on a well-conditioned system).
pub const IR_MAX_OUTER: usize = 12;

/// The outer target `T` of (118.4.4): `max(tolerance·norm, rel_tol·r0)`, the
/// second term only when `rel_tol > 0` - a zero `rel_tol` must not smuggle a
/// relative term in through a multiplication by zero that a negative or NaN
/// `r0` could resurrect.
pub fn ir_target(tolerance: f64, rel_tol: f64, norm: f64, r0: f64) -> f64 {
    let mut t = tolerance * norm;
    if rel_tol > 0.0 {
        t = t.max(rel_tol * r0);
    }
    t
}

/// The inner tolerance `tau_k` of (118.4.6): `max(IR_INNER_REL, target/r_k)`
/// when `r_k > 0`, else the floor alone. `target/r_k` is exactly the
/// reduction the outer test still needs, scaled off the unit-norm correction
/// right-hand side (118.4.5).
pub fn ir_inner_tolerance(target: f64, r_k: f64) -> f64 {
    if r_k > 0.0 {
        IR_INNER_REL.max(target / r_k)
    } else {
        IR_INNER_REL
    }
}

/// The outer loop's verdict on one step of (118.4.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrVerdict {
    /// `R_k` met the target: the double solution is the answer.
    Converged,
    /// Take another correction step.
    Continue,
    /// The inner sweeps have spent `maxIter`.
    BudgetSpent,
    /// The float correction no longer contracts.
    Stagnated,
    /// [`IR_MAX_OUTER`] steps and still no target.
    OuterCapReached,
}

/// The stop rules of (118.4.8), checked in this order: `r_k == 0` is
/// Converged; `r_k <= target` with at least `min_iter` inner sweeps is
/// Converged; `inner_sweeps >= max_iter` is BudgetSpent;
/// `r_k > IR_STAGNATION * r_prev` is Stagnated; `outer_steps >=
/// IR_MAX_OUTER` is OuterCapReached; otherwise Continue.
pub fn ir_verdict(
    r_k: f64,
    r_prev: Option<f64>,
    target: f64,
    inner_sweeps: usize,
    min_iter: usize,
    max_iter: usize,
    outer_steps: usize,
) -> IrVerdict {
    if r_k == 0.0 {
        return IrVerdict::Converged;
    }
    if r_k <= target && inner_sweeps >= min_iter {
        return IrVerdict::Converged;
    }
    if inner_sweeps >= max_iter {
        return IrVerdict::BudgetSpent;
    }
    if let Some(p) = r_prev {
        if r_k > IR_STAGNATION * p {
            return IrVerdict::Stagnated;
        }
    }
    if outer_steps >= IR_MAX_OUTER {
        return IrVerdict::OuterCapReached;
    }
    IrVerdict::Continue
}

#[cfg(test)]
mod tests {
    use super::*;

    // The expectations are exact: every literal below is the EXACT value the
    // f64 arithmetic produces, not a tolerance.
    #[test]
    fn the_outer_target_is_the_relative_or_absolute_term() {
        assert_eq!(ir_target(1e-8, 0.0, 2.0, 5.0), 2e-8);
        assert_eq!(ir_target(1e-8, 0.1, 2.0, 5.0), 0.5);
        assert_eq!(ir_target(0.0, 0.0, 2.0, 5.0), 0.0);
    }

    #[test]
    fn the_inner_tolerance_never_dips_below_its_floor() {
        assert_eq!(ir_inner_tolerance(1e-10, 1.0), 1e-4);
        assert_eq!(ir_inner_tolerance(0.05, 1.0), 0.05);
        assert_eq!(ir_inner_tolerance(0.5, 0.0), 1e-4);
    }

    #[test]
    fn the_stop_rules_fire_in_their_stated_order() {
        use IrVerdict::*;
        assert_eq!(ir_verdict(0.0, None, 0.0, 0, 5, 100, 0), Converged);
        assert_eq!(ir_verdict(1e-9, None, 1e-8, 0, 0, 100, 0), Converged);
        assert_eq!(ir_verdict(1e-9, None, 1e-8, 0, 1, 100, 0), Continue);
        assert_eq!(ir_verdict(1.0, Some(1.0), 0.5, 100, 0, 100, 3), BudgetSpent);
        assert_eq!(ir_verdict(0.6, Some(1.0), 0.1, 10, 0, 100, 1), Stagnated);
        assert_eq!(ir_verdict(0.5, Some(1.0), 0.1, 10, 0, 100, 1), Continue);
        assert_eq!(ir_verdict(1e-3, Some(1.0), 1e-9, 10, 0, 100, 12), OuterCapReached);
    }
}
