// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The precision constants and stop rules of SPEC-LIT §118.4, the
//! mixed-precision iterative refinement that carries a `single` build's
//! Krylov solution and residual in double while the unchanged float method
//! computes only a scaled correction; and the row classes of SPEC-LIT
//! §118.14 with D-F32-1's round-off tolerance (118.14.1).
//!
//! ORIGINAL - the host-side arithmetic of the outer loop's targets and stop
//! rules, exactly equations (118.4.4), (118.4.6) and (118.4.8) of §118.4,
//! and (118.14.1). Pure host logic, no GPU, compiled in BOTH builds: in the
//! f64 build nothing calls the refinement rules, and `roundoff_tol` is the
//! identity there by design.
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

/// `eps32 / eps64 = 2^-23 / 2^-52 = 2^29`: one f64 unit in the last place,
/// counted in f32 units, is this many times smaller (SPEC-LIT §118.14).
pub const ROUNDOFF_ULP_RATIO: f64 = 536_870_912.0;

/// The cap of (118.14.1): `1e-5`, about 84 f32 units in the last place at 1
/// (`1e-5 / 2^-23 = 83.9`).
pub const ROUNDOFF_CAP: f64 = 1e-5;

/// The f32 tolerance of a round-off-class (R) row, D-F32-1 (docs/17 section 11
/// item 1, and section 5.1 rule 4), exactly (118.14.1):
/// `tol32 = max(tol64, min(tol64 · 2^29, 1e-5))` - the same number of units
/// in the last place as the f64 tolerance, capped at 84 f32 ulps, and never
/// tighter than the f64 tolerance. It is computed in both builds so the rule
/// itself is testable in either; [`roundoff_tol`] is what a row uses.
pub fn tol32(tol64: f64) -> f64 {
    tol64.max((tol64 * ROUNDOFF_ULP_RATIO).min(ROUNDOFF_CAP))
}

/// The tolerance an R row is judged at in THIS build: [`tol32`] under
/// `single`, and `tol64` itself in the f64 build, so an f64 row is the same
/// bits it was before it had a class.
pub fn roundoff_tol(tol64: f64) -> f64 {
    if cfg!(feature = "single") {
        tol32(tol64)
    } else {
        tol64
    }
}

/// The precision class of a gate row or test tolerance (docs/17 section 5.1 rule 4,
/// SPEC-LIT §118.14). Only R moves between the builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowClass {
    /// Physics: a discretisation, model or published band, the same in f32.
    P,
    /// An accumulated identity evaluated in f64, the same tolerance in f32.
    A,
    /// A round-off twin, held to [`roundoff_tol`] of its f64 tolerance.
    R,
    /// Discrete or bitwise: a count, an ordering, a yes/no, an exact zero.
    D,
}

impl RowClass {
    /// The four classes in the order a tally prints them.
    pub const ALL: [RowClass; 4] = [RowClass::P, RowClass::A, RowClass::R, RowClass::D];

    /// The class's letter.
    pub fn letter(self) -> char {
        match self {
            RowClass::P => 'P',
            RowClass::A => 'A',
            RowClass::R => 'R',
            RowClass::D => 'D',
        }
    }

    /// The tolerance a row of this class with f64 tolerance `tol64` is
    /// judged at in this build: [`roundoff_tol`] for R, `tol64` otherwise.
    pub fn tol(self, tol64: f64) -> f64 {
        match self {
            RowClass::R => roundoff_tol(tol64),
            RowClass::P | RowClass::A | RowClass::D => tol64,
        }
    }
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

    #[test]
    fn tol32_is_the_same_ulps_capped_at_84_and_never_tighter() {
        assert_eq!(ROUNDOFF_ULP_RATIO, 2f64.powi(29));
        assert_eq!(tol32(1e-12), 1e-5);
        assert_eq!(tol32(1e-14), 1e-14 * 536_870_912.0);
        assert_eq!(tol32(0.0), 0.0);
        assert_eq!(tol32(1e-5), 1e-5);
        assert_eq!(tol32(1e-3), 1e-3);
        // Only R moves; in the f64 build nothing does.
        assert_eq!(RowClass::P.tol(1e-12), 1e-12);
        assert_eq!(RowClass::D.tol(0.0), 0.0);
        if !cfg!(feature = "single") {
            assert_eq!(roundoff_tol(1e-12), 1e-12);
            assert_eq!(RowClass::R.tol(1e-14), 1e-14);
        }
    }
}
