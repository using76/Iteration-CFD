// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Solid mechanics - the thermo-elastic solid.
//!
//! Written from: nothing yet. This file is **ORIGINAL** - it is the module
//! root and carries no numerics of its own; every child declares its own
//! provenance in its own header. No GPL-licensed source was consulted.
//!
//! # What is here today, and what is not
//!
//! Three children. [`bc`] and [`displacement`] **are** part of the solver:
//! they carry the per-component boundary statement and the displacement
//! operator itself, one application of the segregated thermo-elastic
//! fixed-point map on the device. [`prototype`] is still not part of any
//! solver - it is the **risk-1 experiment** of
//! `docs/09-thermal-structural-plan.md` §F: the same outer loop written
//! host-side, before a kernel existed, so that the contraction the plan
//! *derives* -
//! `|deferred|/|implicit| = (mu + lambda)/(2 mu + lambda) = 1/(2(1 - nu))` -
//! is **measured** rather than asserted. SPEC-LIT §46.4 is the precedent that
//! made that the order of work: an estimate written into a specification,
//! corrected afterwards by the test that measured it.
//!
//! [`displacement`] applies the prototype's map ONCE per call - three
//! boundary sub-passes, the deferred stress gather, the thermal load, one
//! scalar system per displacement component solved by the crate's own
//! [`crate::solver::solve`] - and is proven against the prototype by
//! diffing every stage of the application. The outer loop that drives the
//! map to convergence, the relaxation that makes it converge quickly and the
//! stress field it eventually produces are later units' work; nothing here
//! iterates the map on its own.
//!
//! The displacement equation itself has **no SPEC-LIT section yet**. The plan
//! reserves the ninety-fifth for it, and nothing in this module may cite that
//! number until the section exists - §80's audit fails the build on a
//! citation whose address is vacant, and the whole point of §0 rule 6 is that
//! an address is either occupied or it is not. What the prototype cites are the sections it
//! genuinely reuses: §1's LDU storage, §2.4's over-relaxed non-orthogonal
//! correction, §3.2's Gauss laplacian, §3.5's Green-Gauss gradient, §4's one
//! mixed boundary triple, §8.2's conjugate gradients and §8.4's residual
//! normalisation.

use crate::{Error, Result, Scalar};

pub mod bc;
pub mod displacement;
pub mod prototype;
#[cfg(test)]
mod tests;

/// Isotropic linear thermo-elasticity, in the two engineering constants a
/// case would state plus the expansion coefficient (Timoshenko & Goodier
/// ch. 1; Boley & Weiner ch. 1). `T_ref` is not here because the prototype
/// carries `T - T_ref` directly.
///
/// The Lamé conversion is the one the plan's §D.1 writes. It moved here from
/// [`prototype`] with the unit that put the operator on the device: the
/// kernels of [`displacement`] take `mu`, `lambda` and `(3 lambda + 2 mu) alpha`
/// from it, and the case-format unit's case-facing material lowers INTO it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// Young's modulus `E` [Pa].
    pub e: Scalar,
    /// Poisson's ratio `nu`, in `(-1, 0.5)`.
    pub nu: Scalar,
    /// Linear thermal expansion coefficient `alpha` [1/K].
    pub alpha: Scalar,
}

impl Material {
    /// Steel, near enough for an experiment whose answer is a ratio: `E = 200
    /// GPa`, `nu` as given, `alpha = 1.2e-5 / K`.
    pub fn steel(nu: Scalar) -> Self {
        Self { e: 200.0e9, nu, alpha: 1.2e-5 }
    }

    /// `mu = E / (2(1 + nu))`.
    #[inline]
    pub fn mu(self) -> Scalar {
        self.e / (2.0 * (1.0 + self.nu))
    }

    /// `lambda = E nu / ((1 + nu)(1 - 2 nu))`.
    #[inline]
    pub fn lambda(self) -> Scalar {
        self.e * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
    }

    /// `2 mu + lambda`, the implicit coefficient of the split.
    #[inline]
    pub fn implicit_gamma(self) -> Scalar {
        2.0 * self.mu() + self.lambda()
    }

    /// `3 lambda + 2 mu = E / (1 - 2 nu)`, the bulk factor the thermal term
    /// carries.
    #[inline]
    pub fn three_lambda_two_mu(self) -> Scalar {
        3.0 * self.lambda() + 2.0 * self.mu()
    }

    /// The plan's derived contraction, `(mu + lambda)/(2 mu + lambda)`, which
    /// reduces to `1/(2(1 - nu))`. Computed from the Lamé constants and NOT
    /// from the closed form, so that the identity is something the tests can
    /// check rather than something this file assumes.
    #[inline]
    pub fn predicted_contraction(self) -> Scalar {
        (self.mu() + self.lambda()) / self.implicit_gamma()
    }
}

impl Material {
    /// Refuses by name: `E <= 0` or not finite; `nu` outside the open interval
    /// (-1, 0.5) (Timoshenko & Goodier ch. 1: the Lamé constants change sign
    /// or blow up); `alpha` not finite. The unit that adds the outer loop
    /// extends this in place with the measured 0.45 threshold.
    pub fn validate(&self) -> Result<()> {
        if !self.e.is_finite() || self.e <= 0.0 {
            return Err(Error::Config(format!(
                "solid: E = {} is not a positive finite Young's modulus",
                self.e
            )));
        }
        if !(self.nu > -1.0 && self.nu < 0.5) {
            return Err(Error::Config(format!(
                "solid: nu = {} is outside the open interval (-1, 0.5) in which \
                 the Lamé constants are finite and mu > 0",
                self.nu
            )));
        }
        if !self.alpha.is_finite() {
            return Err(Error::Config(format!(
                "solid: alpha = {} is not finite",
                self.alpha
            )));
        }
        Ok(())
    }
}
