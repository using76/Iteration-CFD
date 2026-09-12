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
//! map to convergence, with the Aitken relaxation that makes it converge
//! quickly, is [`outer`]'s work; [`displacement`] never iterates the map
//! on its own.
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

use crate::mesh::HostMesh;
use crate::{Error, Result, Scalar, Vec3};

/// The largest Poisson ratio the segregated loop is accepted at, read off
/// the TS-0 sweep (`docs/09-thermal-structural-plan.md` F.1a: 71 outer
/// iterations at 0.45 on a 20^3 block, 615 at 0.49, and no convergence in
/// 2000 at 0.49 on a jittered block) and not off the algebra. Inclusive:
/// `nu = 0.45` is accepted, anything above is refused.
pub const NU_MAX: Scalar = 0.45;

/// The largest `max|u| / min_c V_c^(1/3)` a coupled case may report before
/// the driver has to refuse it: past a tenth of its own smallest cell the
/// small-displacement premise the space conservation law rests on is gone.
pub const MOTION_RATIO_MAX: Scalar = 0.1;

pub mod bc;
pub mod displacement;
pub mod outer;
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
    /// or blow up); `nu` above the measured edge 0.45; `alpha` not finite.
    pub fn validate(&self) -> Result<()> {
        if !self.e.is_finite() || self.e <= 0.0 {
            return Err(Error::Config(format!(
                "solid: E = {} is not positive and finite - which a Young's \
                 modulus has to be",
                self.e
            )));
        }
        if !(self.nu > -1.0 && self.nu < 0.5) {
            return Err(Error::Config(format!(
                "solid: nu = {} is outside the open interval (-1, 0.5) in which \
                 the Lamé constants are finite and mu > 0 (Timoshenko & Goodier \
                 ch. 1)",
                self.nu
            )));
        }
        if self.nu > NU_MAX {
            return Err(Error::Config(format!(
                "solid: nu = {} is above the measured edge 0.45: the segregated \
                 displacement loop stalls as the solid approaches \
                 incompressibility - docs/09-thermal-structural-plan.md F.1a \
                 measured 71 outer iterations at nu = 0.45 on a 20^3 block, 615 \
                 at nu = 0.49, and no convergence in 2000 at nu = 0.49 on a \
                 jittered block, where a near-incompressible solid entered here \
                 would stall at a residual read as converged. The route is a \
                 block-coupled solve (Cardiff, Tuković, Jasak & Ivanković 2016, \
                 DOI 10.1016/j.compstruc.2016.07.004), which is not built",
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

// ==========================================================================
//  What is not built - refused by name, with its route
// ==========================================================================

/// Everything a solid case can ask for that this solver has not built, each
/// with the route that WOULD take it. Refused the way SPEC-LIT §46.4
/// refuses a full conductivity tensor: measured and named, not approximated
/// by something that will not do.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NotBuilt {
    /// Large rotation, Green-Lagrange strain.
    FiniteStrain,
    /// Yielding; the route is a J2 return map on the same assembly.
    Plasticity,
    /// A search, an active set, a non-symmetric system.
    Contact,
    /// Crack growth.
    Fracture,
    /// A ddt scheme on displacement (Newmark, HHT-alpha, generalised-alpha).
    Inertia,
    /// An orthotropic stiffness; when it arrives, aligned meshes only.
    Orthotropic,
    /// The coupling back into the energy equation; the variant carries the
    /// parameter that says how large the omission is.
    TwoWayCoupling {
        /// `(3 lambda + 2 mu)^2 alpha^2 T_0 / ((lambda + 2 mu) rho c)`.
        delta: Scalar,
    },
    /// The 3x3 block-coupled solve a near-incompressible solid needs.
    BlockCoupled,
}

/// The refusal, as an [`Error::Config`] naming the feature, the setting that
/// asked for it and the route that would take it. One sentence each, in the
/// measured-and-refused voice: the gap is not approximated by something that
/// will not do, it is named and declined.
pub fn refuse(what: NotBuilt, setting: &str) -> Error {
    let head = format!("solid {setting}");
    match what {
        NotBuilt::FiniteStrain => Error::Config(format!(
            "{head}: finite strain is not built - small strain is a stated limit \
             of this segregated solve, and large rotation belongs to the FSI \
             section of docs/10-fsi-solid-mesh-plan.md (Cardiff, Karač & \
             Ivanković 2014), which is not written yet"
        )),
        NotBuilt::Plasticity => Error::Config(format!(
            "{head}: plasticity is not built - the route is a J2 return map on \
             the same assembly (Demirdžić & Martinović 1993, DOI \
             10.1016/0045-7825(93)90085-C), the first thing after release 1"
        )),
        NotBuilt::Contact => Error::Config(format!(
            "{head}: contact is not built - it needs a search, an active set and \
             a non-symmetric, non-LDU system, and none of the three is here"
        )),
        NotBuilt::Fracture => {
            Error::Config(format!("{head}: fracture is not built"))
        }
        NotBuilt::Inertia => Error::Config(format!(
            "{head}: inertia is not built - a ddt scheme on displacement is what \
             is missing: Newmark (1959, DOI 10.1061/JMCEA3.0000098), HHT-alpha \
             (Hilber, Hughes & Taylor 1977, DOI 10.1002/eqe.4290050306) and \
             generalised-alpha (Chung & Hulbert 1993, DOI 10.1115/1.2900803) \
             are the schemes not built, and controlled algorithmic damping \
             (rho_infinity) is what BDF2 would not give"
        )),
        NotBuilt::Orthotropic => Error::Config(format!(
            "{head}: an orthotropic stiffness is not built; when it is, it is \
             accepted only on a mesh aligned with its axes, alignment measured \
             per face and refused the way SPEC-LIT §46.4 measures the \
             conductivity's alignment"
        )),
        NotBuilt::TwoWayCoupling { delta } => Error::Config(format!(
            "{head}: two-way coupling is not built - the -T_0 (3 lambda + 2 mu) \
             alpha d(tr eps)/dt term is not carried into the energy equation; \
             delta = {delta:.3e} is what that omits"
        )),
        NotBuilt::BlockCoupled => Error::Config(format!(
            "{head}: a block-coupled solve is not built - Cardiff, Tuković, Jasak \
             & Ivanković, *Comput. Struct.* 175 (2016) 100-122, DOI \
             10.1016/j.compstruc.2016.07.004 is the route a near-incompressible \
             case has to take, and a 3x3 coefficient per face breaks the \
             one-entry-per-face LDU storage of SPEC-LIT §1"
        )),
    }
}

/// The two-way coupling parameter (Boley & Weiner ch. 1-2: the
/// `-(3 lambda + 2 mu) alpha T_0 d(tr eps)/dt` term the one-way energy
/// equation omits):
///
/// ```text
/// delta = (3 lambda + 2 mu)^2 alpha^2 T_0 / ((lambda + 2 mu) rho_s c_s)
/// ```
///
/// About one percent for steel at room temperature - the number that says
/// the one-way coupling is an approximation with a KNOWN size, and the
/// number the [`NotBuilt::TwoWayCoupling`] refusal prints.
pub fn two_way_coupling_delta(m: &Material, rho: Scalar, c: Scalar, t0: Scalar) -> Scalar {
    let bulk = m.three_lambda_two_mu();
    bulk * bulk * m.alpha * m.alpha * t0 / ((m.lambda() + 2.0 * m.mu()) * rho * c)
}

/// `max|u| / h_min`, with `h_min = min_c V_c^(1/3)`: how far the solid has
/// moved in units of its own smallest cell. Computed here, on the outer
/// loop's report; DECIDED by the driver, because a solid-only case
/// legitimately deflects several cell widths and must not be refused.
pub fn motion_ratio_of(u: &[Vec3], v: &[Scalar]) -> Scalar {
    let h_min = v.iter().map(|vc| vc.cbrt()).fold(Scalar::INFINITY, |a, b| a.min(b));
    let max_u = u.iter().map(|a| a.mag()).fold(0.0 as Scalar, |a, b| a.max(b));
    max_u / h_min
}

/// [`motion_ratio_of`] on a host mesh's cell volumes.
pub fn mesh_motion_ratio(u: &[Vec3], m: &HostMesh) -> Scalar {
    motion_ratio_of(u, &m.v)
}

/// Refuses a displacement that has moved past a tenth of the mesh's own
/// smallest cell: an ALE step obeying the space conservation law needs a
/// mesh-motion solver, and this solver has not built one.
pub fn refuse_displacement_reaching_the_fluid(ratio: Scalar) -> Result<()> {
    if ratio > MOTION_RATIO_MAX {
        return Err(Error::Config(format!(
            "solid: motion_ratio = {ratio:.3e} is above the threshold 0.1 - the \
             displacement has reached the fluid mesh. The space conservation law \
             (Demirdžić & Perić 1988) that an ALE step obeys needs a mesh-motion \
             solver, which is not built; docs/10-fsi-solid-mesh-plan.md D, \
             address 105 is where it is planned"
        )));
    }
    Ok(())
}
