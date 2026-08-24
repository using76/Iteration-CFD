// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The discrete finite-volume operators - the heart of the solver.
//!
//! Written from:
//!   H. Jasak, "Error Analysis and Estimation for the Finite Volume Method
//!     with Applications to Fluid Flows", PhD thesis, Imperial College (1996),
//!     ch. 3
//!   F. Moukalled, L. Mangani, M. Darwish, "The Finite Volume Method in
//!     Computational Fluid Dynamics", Springer (2016), ch. 8, 11, 12, §15.4
//!   S. V. Patankar, "Numerical Heat Transfer and Fluid Flow", Hemisphere
//!     (1980), ch. 4-6
//!   J. H. Ferziger, M. Perić, "Computational Methods for Fluid Dynamics",
//!     §6.3.2
//!   P. K. Sweby, SIAM J. Numer. Anal. 21 (1984) 995
//!   B. van Leer, J. Comput. Phys. 23 (1977) 276; ibid. 32 (1979) 101
//!   G. D. van Albada, B. van Leer, W. W. Roberts, Astron. Astrophys. 108
//!     (1982) 76
//!   P. L. Roe, Ann. Rev. Fluid Mech. 18 (1986) 337
//!   M. Darwish, F. Moukalled, Int. J. Heat Mass Transfer 46 (2003) 599
//!   K. C. Khosla, S. G. Rubin, Computers & Fluids 2 (1974) 207 - deferred
//!     correction
//!   R. F. Warming, R. M. Beam, AIAA J. 14 (1976) 1241 - `linearUpwind`
//!   B. P. Leonard, CMAME 19 (1979) 59 - QUICK
//!   H. Jasak, H. G. Weller, A. D. Gosman, Int. J. Numer. Methods Fluids 31
//!     (1999) 431 - the Gamma NVD scheme
//!   W. H. Press et al., "Numerical Recipes", §3.3 - Hermite cubic
//!   T. J. Barth, D. C. Jespersen, AIAA 89-0366 (1989) - the cell-limited
//!     gradient
//!   V. Venkatakrishnan, AIAA 93-0880 (1993) - its differentiable variant
//! and from `SPEC-LIT.md` §2, §3, §4, §6, §7, §11 and §12, which cite all of
//! the above. No GPL-licensed source was consulted.
//!
//! # What an operator does to a matrix
//!
//! Every implicit operator here **adds into** an existing [`GpuLduMatrix`]
//! with a `sign` factor, so an equation is written the way it is written on
//! paper. For
//!
//! ```text
//! ddt(psi) + div(phi, psi) - laplacian(gamma, psi) == su
//! ```
//!
//! the call sequence is `fvm_ddt_euler(.., 1.0)`, `fvm_div_gauss(.., 1.0)`,
//! `fvm_laplacian(.., -1.0)`, `fvm_su(.., 1.0)`, and what comes out is the
//! `A` and the `b` of `A·psi = b`.
//!
//! # The boundary contract
//!
//! Boundary faces do not go straight into `diag`/`source`; they land in
//! `internal_coeffs`/`boundary_coeffs`, and the convention - which
//! `ldu_ops::add_boundary_contributions` and `ldu_ops::amul` must both honour
//! - is
//!
//! ```text
//! uncoupled face:  diag[P]   += internal_coeffs[bf]
//!                  source[P] += boundary_coeffs[bf]
//! coupled face:    diag[P]   += internal_coeffs[bf]
//!                  Apsi[P]   -= boundary_coeffs[bf] * psi[nbr]     (in amul)
//! ```
//!
//! It falls straight out of the algebra. Row `P` of the operator gains
//! `T(psi_b)`, the mixed form of §4 splits `psi_b` into `vic·psi_P + vbc`, the
//! part multiplying `psi_P` is implicit and stays on the left, and the known
//! part crosses to the right with a sign change. That sign change is the only
//! reason `boundary_coeffs` carries the minus it does.
//!
//! # The one subtlety
//!
//! Both `div` and `laplacian` want *"subtract my own column sum from the
//! diagonal"*. If the per-cell pass got that sum by reading `upper`/`lower`
//! back, it would also read whatever a **previously applied operator** had
//! already added there, and subtract it a second time. Every diagonal pass in
//! this module therefore recomputes its own face coefficient from its own
//! inputs - `w·phi` for convection, `gammaMagSf·deltaCoeffs` for diffusion -
//! and never reads the matrix it is writing. It costs one multiply per face
//! and it makes the operators completely order-independent, which
//! `tests::operators_do_not_couple_through_the_diagonal` pins down.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::{GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use crate::ldu::GpuLduMatrix;
use crate::mesh::GpuMesh;
use crate::{Label, Scalar, Tensor, Vec3};

// ==========================================================================
//  Convection schemes and the TVD limiter family (SPEC-LIT §7)
// ==========================================================================

/// Device discriminants for the weight kernels. Kept next to the `Limiter`
/// they encode so the two cannot drift; mirrored by the `OFLIM_*` defines in
/// `cuda/fv.cu`.
mod code {
    use crate::Label;
    pub const CENTRAL: Label = 0;
    pub const UPWIND: Label = 1;
    pub const MINMOD: Label = 2;
    pub const VAN_LEER: Label = 3;
    pub const VAN_ALBADA: Label = 4;
    pub const SUPERBEE: Label = 5;
    pub const MUSCL: Label = 6;
    pub const SWEBY: Label = 7;
    /// SPEC-LIT §11.3, clipped into the TVD region.
    pub const QUICK: Label = 8;
    /// SPEC-LIT §11.3, `Psi = (3 + r)/4` as written.
    pub const QUICK_UNLIMITED: Label = 9;
    /// SPEC-LIT §11.6, Gamma NVD.
    pub const GAMMA: Label = 10;
    /// SPEC-LIT §11.5, a constant central/upwind blend.
    pub const BLENDED: Label = 11;
}

/// Device discriminants for the deferred correction of SPEC-LIT §11.1;
/// mirrored by the `OFDIVCORR_*` defines in `cuda/fv.cu`.
mod corr_code {
    use crate::Label;
    pub const NONE: Label = 0;
    pub const LINEAR_UPWIND: Label = 1;
    pub const CUBIC: Label = 2;
}

/// Device discriminants for the gradient limiter of SPEC-LIT §12.2; mirrored
/// by the `OFGRADLIM_*` and `OFGRADMODE_*` defines in `cuda/fv.cu`.
mod grad_code {
    use crate::Label;
    pub const BARTH_JESPERSEN: Label = 0;
    pub const VENKATAKRISHNAN: Label = 1;
    pub const MODE_CELL: Label = 0;
    pub const MODE_FACE: Label = 1;
}

/// A TVD flux limiter, `Psi(r)`.
///
/// SPEC-LIT §7 tabulates all six with their sources. Every one satisfies
/// `Psi(r) = 0` for `r <= 0`, which is what makes the scheme total-variation
/// diminishing, and `Psi(1) = 1`, which is what makes it second order on
/// smooth data. [`Limiter::psi`] is the host mirror of `limiterPsi` in
/// `cuda/fv.cu`; `tests::device_limiter_agrees_with_the_host` pins them
/// together.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Limiter {
    /// `max(0, min(1, r))` - Roe (1986). The most diffusive of the family.
    MinMod,
    /// `(r + |r|)/(1 + |r|)` - van Leer, *JCP* 23 (1977).
    VanLeer,
    /// `(r² + r)/(r² + 1)` - van Albada et al., *A&A* 108 (1982).
    VanAlbada,
    /// `max(0, max(min(2r,1), min(r,2)))` - Roe (1986). The least diffusive
    /// limiter that is still TVD, and the one most likely to steepen a smooth
    /// profile into a staircase.
    Superbee,
    /// `max(0, min(2r, (r+1)/2, 2))` - van Leer (1979).
    Muscl,
    /// `max(0, max(min(βr,1), min(r,β)))` with `1 <= β <= 2` - Sweby (1984).
    /// `β = 1` is minmod and `β = 2` is Superbee, so this one parameter sweeps
    /// the whole width of the TVD region.
    Sweby(Scalar),
}

impl Limiter {
    fn code(self) -> Label {
        match self {
            Self::MinMod => code::MINMOD,
            Self::VanLeer => code::VAN_LEER,
            Self::VanAlbada => code::VAN_ALBADA,
            Self::Superbee => code::SUPERBEE,
            Self::Muscl => code::MUSCL,
            Self::Sweby(_) => code::SWEBY,
        }
    }

    /// Sweby's `β`, clamped to the range the TVD proof covers. Outside
    /// `1 <= β <= 2` the scheme leaves the TVD region - below 1 it is no
    /// longer second order, above 2 it is no longer bounded - so the value is
    /// clamped rather than trusted.
    fn beta(self) -> Scalar {
        match self {
            Self::Sweby(b) => b.clamp(1.0, 2.0),
            _ => 0.0,
        }
    }

    /// `Psi(r)`, the host mirror of the device function.
    ///
    /// Two things here are not literal transcriptions of the table and are
    /// deliberate:
    ///
    /// * `r <= 0` returns zero for **all six**. Five of them give that from
    ///   their formula alone; van Albada does not - `(r²+r)/(r²+1)` turns
    ///   positive again below `r = -1` - and SPEC-LIT §7 states the property
    ///   as a requirement of the family, so it is enforced rather than
    ///   assumed.
    /// * `r` is clamped to a large finite value first. Every limiter has a
    ///   finite limit as `r -> inf`, but van Leer and van Albada reach theirs
    ///   as `inf/inf`, which is a NaN, and a NaN weight silently destroys a
    ///   whole matrix row.
    pub fn psi(self, r: Scalar) -> Scalar {
        // NaN included, which is why this is not written `r <= 0.0`: a NaN
        // ratio must give a zero limiter, not propagate.
        if r.is_nan() || r <= 0.0 {
            return 0.0;
        }

        const RMAX: Scalar = 1e12;
        let r = if r > RMAX { RMAX } else { r };

        match self {
            Self::MinMod => r.min(1.0),
            Self::VanLeer => 2.0 * r / (1.0 + r),
            Self::VanAlbada => (r * r + r) / (r * r + 1.0),
            Self::Superbee => (2.0 * r).min(1.0).max(r.min(2.0)),
            Self::Muscl => (2.0 * r).min(0.5 * (r + 1.0)).min(2.0),
            Self::Sweby(_) => {
                let b = self.beta();
                (b * r).min(1.0).max(r.min(b))
            }
        }
    }
}

/// How the convection operator builds its face value.
///
/// This is the scheme as the *operator* sees it;
/// [`crate::io::case::DivScheme`] is the scheme as an `fvSchemes` entry spells
/// it, and converts into this one. `div_scheme_weights` takes
/// `impl Into<DivScheme>` so a caller may pass either without a cast.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DivScheme {
    /// `Gauss linear`: the mesh's own interpolation weight. Second order,
    /// unbounded.
    Central,
    /// `Gauss upwind`: `w = 1` for `phi >= 0`, else `0`. First order, bounded.
    Upwind,
    /// `Gauss linearUpwind` - SPEC-LIT §11.2, Warming & Beam (1976).
    ///
    /// The **implicit** weights are the upwind ones; the second-order part is
    /// the explicit gradient correction `(C_f - C_U)·grad(psi)_U` that
    /// [`fvm_div_correction`] adds to the source. Assembling the weights
    /// alone gives plain first-order upwind, which is exactly the bug this
    /// variant used to ship: **a caller that selects this scheme must call
    /// [`fvm_div_correction`] too**, and [`DivScheme::correction`] is what
    /// tells it so.
    LinearUpwind,
    /// `cubic` - SPEC-LIT §11.4, Hermite interpolation through the two cell
    /// values and the two cell gradients. Implicit base central, correction
    /// `[d·grad_P - d·grad_N]/8`. Fourth order on a uniform mesh and freely
    /// unbounded: a verification scheme, not a production one.
    Cubic,
    /// `QUICK` - SPEC-LIT §11.3, Leonard (1979), in the TVD form
    /// `Psi(r) = max(0, min((3+r)/4, 2r, 2))`. *DESIGN*: a bare `QUICK` in
    /// fvSchemes selects this, the **limited** form, because an unbounded
    /// scheme reached by a name that does not say "unbounded" is a trap.
    Quick,
    /// `QUICKV`/`quickUnlimited` - the same with `Psi(r) = (3 + r)/4` as
    /// written, clipping nothing. Unbounded by construction.
    QuickUnlimited,
    /// `Gamma <beta_m>` - SPEC-LIT §11.6, Jasak, Weller & Gosman (1999). An
    /// NVD scheme that switches smoothly between upwind and central. The
    /// coefficient is clamped to `[0.1, 0.5]`, the range the paper recommends
    /// and warns against leaving.
    Gamma(Scalar),
    /// `blended <gamma>` - SPEC-LIT §11.5.
    /// `psi_f = (1-gamma)·psi_upwind + gamma·psi_central`. Wholly implicit:
    /// the blend is a face weight, so this one needs no correction pass and no
    /// gradient.
    Blended(Scalar),
    /// `linearUpwindBlended <gamma>` - SPEC-LIT §11.5's second blend,
    /// `psi_f = (1-gamma)·psi_linearUpwind + gamma·psi_central`. Both ends are
    /// second order, so the parameter trades dispersion against dissipation
    /// rather than accuracy against stability, which is why it is the usual
    /// LES choice. Implicit part is the blended weight; explicit part is
    /// `(1-gamma)` times the linearUpwind correction.
    LinearUpwindBlended(Scalar),
    /// One of the TVD limiters of SPEC-LIT §7. Needs the upwind cell gradient
    /// during assembly.
    Limited(Limiter),
}

/// The explicit half of a deferred-correction scheme (SPEC-LIT §11.1).
///
/// Returned by [`DivScheme::correction`] so a caller can tell, without a
/// match of its own, whether assembling the weights was the whole job.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DivCorrection {
    /// The scheme is wholly implicit; the weights are the whole assembly.
    None,
    /// `coef · (C_f - C_U)·grad(psi)_U` (SPEC-LIT §11.2). `coef` is 1 for
    /// plain `linearUpwind` and `1 - gamma` for the blend of §11.5.
    LinearUpwind(Scalar),
    /// `[d·grad_P - d·grad_N]/8` (SPEC-LIT §11.4).
    Cubic,
}

impl DivCorrection {
    fn code(self) -> Label {
        match self {
            Self::None => corr_code::NONE,
            Self::LinearUpwind(_) => corr_code::LINEAR_UPWIND,
            Self::Cubic => corr_code::CUBIC,
        }
    }

    fn coef(self) -> Scalar {
        match self {
            Self::LinearUpwind(c) => c,
            Self::Cubic => 1.0,
            Self::None => 0.0,
        }
    }

    /// True when [`fvm_div_correction`] has anything to do.
    pub fn is_some(self) -> bool {
        !matches!(self, Self::None)
    }
}

impl DivScheme {
    fn code(self) -> Label {
        match self {
            // `cubic` is deferred onto a CENTRAL base, `linearUpwind` onto an
            // upwind one - SPEC-LIT §11.1 - so the implicit code is the base,
            // never the scheme's own name.
            Self::Central | Self::Cubic => code::CENTRAL,
            Self::Upwind | Self::LinearUpwind => code::UPWIND,
            Self::Quick => code::QUICK,
            Self::QuickUnlimited => code::QUICK_UNLIMITED,
            Self::Gamma(_) => code::GAMMA,
            Self::Blended(_) | Self::LinearUpwindBlended(_) => code::BLENDED,
            Self::Limited(l) => l.code(),
        }
    }

    fn beta(self) -> Scalar {
        match self {
            Self::Limited(l) => l.beta(),
            // Jasak et al. (1999) recommend 0.1 to 0.5 and warn against going
            // outside it, so the value is clamped rather than trusted - the
            // same treatment Sweby's beta gets.
            Self::Gamma(b) => b.clamp(0.1, 0.5),
            Self::Blended(g) | Self::LinearUpwindBlended(g) => g.clamp(0.0, 1.0),
            _ => 0.0,
        }
    }

    /// The explicit correction this scheme needs on top of its weights
    /// (SPEC-LIT §11.1).
    pub fn correction(self) -> DivCorrection {
        match self {
            Self::LinearUpwind => DivCorrection::LinearUpwind(1.0),
            Self::LinearUpwindBlended(g) => {
                DivCorrection::LinearUpwind(1.0 - g.clamp(0.0, 1.0))
            }
            Self::Cubic => DivCorrection::Cubic,
            _ => DivCorrection::None,
        }
    }

    /// True when the **face weight** is a function of `r`, and so cannot be
    /// formed without the upwind cell gradient.
    ///
    /// Distinct from [`Self::needs_gradient`] on purpose. `linearUpwind` and
    /// `cubic` need a gradient for their *correction* but their weights are
    /// plain upwind and plain central; routing them through the limited weight
    /// kernel would evaluate `Psi` for a code that has no `Psi` and silently
    /// produce central weights.
    fn weight_needs_gradient(self) -> bool {
        matches!(
            self,
            Self::Limited(_) | Self::Quick | Self::QuickUnlimited | Self::Gamma(_)
        )
    }

    /// The scheme, spelled the way an `fvSchemes` entry spells it.
    ///
    /// For the start-up banner: SPEC-LIT §13.4 stops a setting being
    /// substituted in silence, and printing what is actually in force is the
    /// other half of that - a reader of the log must not have to infer the
    /// scheme from the case files, because the case files are exactly what may
    /// have been overridden.
    pub fn describe(self) -> String {
        match self {
            Self::Central => "Gauss linear".to_string(),
            Self::Upwind => "Gauss upwind".to_string(),
            Self::LinearUpwind => "Gauss linearUpwind".to_string(),
            Self::Cubic => "Gauss cubic".to_string(),
            Self::Quick => "Gauss QUICK (limited)".to_string(),
            Self::QuickUnlimited => "Gauss QUICKUnlimited".to_string(),
            Self::Gamma(b) => format!("Gauss Gamma {}", b.clamp(0.1, 0.5)),
            Self::Blended(g) => format!("Gauss blended {}", g.clamp(0.0, 1.0)),
            Self::LinearUpwindBlended(g) => {
                format!("Gauss linearUpwindBlended {}", g.clamp(0.0, 1.0))
            }
            Self::Limited(Limiter::VanLeer) => "Gauss vanLeer".to_string(),
            Self::Limited(Limiter::VanAlbada) => "Gauss vanAlbada".to_string(),
            Self::Limited(Limiter::MinMod) => "Gauss Minmod".to_string(),
            Self::Limited(Limiter::Superbee) => "Gauss SuperBee".to_string(),
            Self::Limited(Limiter::Muscl) => "Gauss MUSCL".to_string(),
            Self::Limited(Limiter::Sweby(b)) => {
                format!("Gauss limitedLinear {}", b.clamp(1.0, 2.0))
            }
        }
    }

    /// True when assembling this scheme needs `grad(psi)` at all - for the
    /// weights, for the correction, or for both. The whole reason
    /// [`div_scheme_weights`] carries an optional gradient: a scheme that
    /// cannot get one is an error, not a quiet demotion to upwind.
    pub fn needs_gradient(self) -> bool {
        self.weight_needs_gradient() || self.correction().is_some()
    }
}

// `crate::io::case::DivScheme` IS this type, re-exported. It used to be a
// separate four-variant enum with a lossy `From` between them, which is how
// five of the six limiters of §7 came to be unreachable from a case file: the
// conversion had nowhere to put them. The blanket `From<T> for T` is what
// `div_scheme_weights`' `impl Into<DivScheme>` now resolves through.

// ==========================================================================
//  §12  Gradient and surface-normal-gradient schemes
// ==========================================================================

/// How `grad(psi)` is evaluated before any limiter is applied
/// (SPEC-LIT §3.5, §12.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GradBase {
    /// `Gauss linear`: `(1/V_P) Σ_f (±Sf)·psi_f`.
    #[default]
    Gauss,
    /// `leastSquares`: the inverse-distance-weighted fit of SPEC-LIT §3.5.
    /// Exact for a linear field on any mesh, where Green-Gauss is not.
    LeastSquares,
}

/// `Phi(y)` in the cell-limited gradient of SPEC-LIT §12.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradLimiterKind {
    /// `min(1, y)` - Barth & Jespersen, AIAA 89-0366 (1989). Sharper, and can
    /// stall a steady solve by chattering between iterations.
    BarthJespersen,
    /// `(y² + 2y)/(y² + y + 2)` - Venkatakrishnan, AIAA 93-0880 (1993).
    /// Differentiable, which is what lets a steady solve converge.
    Venkatakrishnan,
}

impl GradLimiterKind {
    fn code(self) -> Label {
        match self {
            Self::BarthJespersen => grad_code::BARTH_JESPERSEN,
            Self::Venkatakrishnan => grad_code::VENKATAKRISHNAN,
        }
    }
}

/// Whether and how the gradient is scaled back so it cannot extrapolate a new
/// extremum (SPEC-LIT §12.2).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum GradLimit {
    /// `Gauss linear` / `leastSquares` with no wrapper.
    #[default]
    None,
    /// `cellLimited <base> <coeff>`: bounds taken over the cell and all its
    /// neighbours.
    Cell(GradLimiterKind, Scalar),
    /// `faceLimited <base> <coeff>`: bounds taken face by face over just the
    /// two cells of each face. *DERIVED* - SPEC-LIT §12.2 gives only the
    /// cell-limited algorithm; this is the same statement made per face, and
    /// is therefore never weaker than the cell-limited one.
    Face(GradLimiterKind, Scalar),
}

/// A complete `gradSchemes` entry: a base plus an optional limiter.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GradScheme {
    pub base: GradBase,
    pub limit: GradLimit,
}

impl GradScheme {
    /// `Gauss linear`, the scheme every operator in this crate used before
    /// `gradSchemes` was read at all.
    pub const GAUSS: Self = Self {
        base: GradBase::Gauss,
        limit: GradLimit::None,
    };

    /// `(kind, mode, coeff)` for the limiter kernel, or `None` when the
    /// gradient is used unlimited.
    ///
    /// A `coeff` of zero disables the limiter outright - SPEC-LIT §12.2 - so
    /// it is answered here as "no pass at all" rather than as a pass that
    /// multiplies by one.
    fn limiter_args(self) -> Option<(Label, Label, Scalar)> {
        match self.limit {
            GradLimit::None => None,
            GradLimit::Cell(k, c) if c > 0.0 => {
                Some((k.code(), grad_code::MODE_CELL, c))
            }
            GradLimit::Face(k, c) if c > 0.0 => {
                Some((k.code(), grad_code::MODE_FACE, c))
            }
            _ => None,
        }
    }
}

impl GradScheme {
    /// The scheme, spelled the way a `gradSchemes` entry spells it.
    pub fn describe(self) -> String {
        let base = match self.base {
            GradBase::Gauss => "Gauss linear",
            GradBase::LeastSquares => "leastSquares",
        };
        let name = |k: GradLimiterKind| match k {
            GradLimiterKind::BarthJespersen => "BarthJespersen",
            GradLimiterKind::Venkatakrishnan => "Venkatakrishnan",
        };
        match self.limit {
            GradLimit::None => base.to_string(),
            GradLimit::Cell(k, c) => format!("cellLimited<{}> {base} {c}", name(k)),
            GradLimit::Face(k, c) => format!("faceLimited<{}> {base} {c}", name(k)),
        }
    }
}

/// A `snGradSchemes` entry - how much of the non-orthogonal correction of
/// SPEC-LIT §2.4 is actually applied.
///
/// This is **not** `nNonOrthogonalCorrectors`. That number says how many extra
/// times the correction is recomputed from a fresher solution; this says
/// whether it is applied at all and how far it is allowed to go. Conflating
/// the two - which this solver used to do - means writing
/// `nNonOrthogonalCorrectors 0`, entirely normal on an orthogonal mesh,
/// silently switches the correction off everywhere.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SnGradScheme {
    /// `uncorrected`: the orthogonal part only. Exact on an orthogonal mesh
    /// and first order on anything else.
    Uncorrected,
    /// `corrected`: the full over-relaxed correction, unlimited.
    #[default]
    Corrected,
    /// `limited <alpha>`: the correction capped at `alpha` times the
    /// orthogonal part - SPEC-LIT §12.3, our own expression. `alpha = 0` is
    /// `uncorrected` bit for bit; large `alpha` tends to `corrected`.
    Limited(Scalar),
}

impl SnGradScheme {
    /// The `alpha` the kernels take: negative means "no limit".
    pub fn alpha(self) -> Scalar {
        match self {
            Self::Uncorrected => 0.0,
            Self::Corrected => -1.0,
            Self::Limited(a) => a.max(0.0),
        }
    }

    /// The scheme, spelled the way an `snGradSchemes` entry spells it.
    pub fn describe(self) -> String {
        match self {
            Self::Uncorrected => "uncorrected".to_string(),
            Self::Corrected => "corrected".to_string(),
            Self::Limited(a) => format!("limited {a}"),
        }
    }

    /// False only for `uncorrected`, where the correction pass is a
    /// guaranteed no-op and the caller may skip it - and skipping it gives
    /// exactly the same bits as running it with `alpha = 0`.
    pub fn applies(self) -> bool {
        !matches!(self, Self::Uncorrected)
    }
}

// ==========================================================================
//  Kernels
// ==========================================================================

/// Every entry point in `cuda/fv.cu`, resolved once.
pub struct FvKernels {
    ddt_euler: CudaFunction,
    ddt_euler_rho: CudaFunction,
    ddt_bdf2: CudaFunction,
    ddt_bdf2_rho: CudaFunction,

    weights_unlimited: CudaFunction,
    weights_limited: CudaFunction,
    weights_boundary: CudaFunction,
    weights_boundary_limited: CudaFunction,

    div_faces: CudaFunction,
    div_diag: CudaFunction,
    div_boundary: CudaFunction,
    div_bounded_diag: CudaFunction,
    div_correction: CudaFunction,
    div_scheme_face_value: CudaFunction,

    lap_faces: CudaFunction,
    lap_diag: CudaFunction,
    lap_boundary: CudaFunction,
    lap_non_orth: CudaFunction,

    sp: CudaFunction,
    susp: CudaFunction,
    su: CudaFunction,

    grad_scalar: CudaFunction,
    grad_vector: CudaFunction,
    grad_scalar_ls: CudaFunction,
    grad_vector_ls: CudaFunction,
    grad_limit_scalar: CudaFunction,
    grad_limit_vector: CudaFunction,
    div_surface: CudaFunction,
    interpolate_linear: CudaFunction,
    interpolate_boundary: CudaFunction,
    flux_internal: CudaFunction,
    flux_boundary: CudaFunction,
    sn_grad_internal: CudaFunction,
    sn_grad_boundary: CudaFunction,
    sn_grad_corr_internal: CudaFunction,
    sn_grad_corr_boundary: CudaFunction,
    reconstruct: CudaFunction,
    production: CudaFunction,
}

impl FvKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::FV)?;
        Ok(Self {
            ddt_euler: k.func("fvDdtEuler")?,
            ddt_euler_rho: k.func("fvDdtEulerRho")?,
            ddt_bdf2: k.func("fvDdtBdf2")?,
            ddt_bdf2_rho: k.func("fvDdtBdf2Rho")?,

            weights_unlimited: k.func("fvWeightsUnlimited")?,
            weights_limited: k.func("fvWeightsLimited")?,
            weights_boundary: k.func("fvWeightsBoundary")?,
            weights_boundary_limited: k.func("fvWeightsBoundaryLimited")?,

            div_faces: k.func("fvDivFaces")?,
            div_diag: k.func("fvDivDiag")?,
            div_boundary: k.func("fvDivBoundary")?,
            div_bounded_diag: k.func("fvDivBoundedDiag")?,
            div_correction: k.func("fvDivCorrection")?,
            div_scheme_face_value: k.func("fvDivSchemeFaceValue")?,

            lap_faces: k.func("fvLapFaces")?,
            lap_diag: k.func("fvLapDiag")?,
            lap_boundary: k.func("fvLapBoundary")?,
            lap_non_orth: k.func("fvLapNonOrth")?,

            sp: k.func("fvSp")?,
            susp: k.func("fvSusp")?,
            su: k.func("fvSu")?,

            grad_scalar: k.func("fvGradScalar")?,
            grad_vector: k.func("fvGradVector")?,
            grad_scalar_ls: k.func("fvGradScalarLeastSquares")?,
            grad_vector_ls: k.func("fvGradVectorLeastSquares")?,
            grad_limit_scalar: k.func("fvGradLimitScalar")?,
            grad_limit_vector: k.func("fvGradLimitVector")?,
            div_surface: k.func("fvDivSurface")?,
            interpolate_linear: k.func("fvInterpolateLinear")?,
            interpolate_boundary: k.func("fvInterpolateBoundary")?,
            flux_internal: k.func("fvFluxInternal")?,
            flux_boundary: k.func("fvFluxBoundary")?,
            sn_grad_internal: k.func("fvSnGradFluxInternal")?,
            sn_grad_boundary: k.func("fvSnGradFluxBoundary")?,
            sn_grad_corr_internal: k.func("fvSnGradCorrInternal")?,
            sn_grad_corr_boundary: k.func("fvSnGradCorrBoundary")?,
            reconstruct: k.func("fvReconstruct")?,
            production: k.func("fvProduction")?,
        })
    }
}

/// A buffer handed to an operator has to be the size the operator will index,
/// because a short one is read out of bounds by the kernel and there is no
/// bounds check on a device pointer. Checked on the host, where it is free.
fn expect_len<T>(buf: &DevBuf<T>, want: usize, what: &str) -> Result<()> {
    if buf.len() == want {
        Ok(())
    } else {
        Err(Error::Config(format!(
            "fv: `{what}` has {} elements, expected {want}",
            buf.len()
        )))
    }
}

fn check_matrix(a: &GpuLduMatrix, m: &GpuMesh) -> Result<()> {
    if a.n_cells != m.n_cells
        || a.n_internal_faces != m.n_internal_faces
        || a.n_boundary_faces != m.n_boundary_faces
    {
        return Err(Error::Config(format!(
            "fv: matrix is sized ({}, {}, {}) but the mesh is ({}, {}, {})",
            a.n_cells,
            a.n_internal_faces,
            a.n_boundary_faces,
            m.n_cells,
            m.n_internal_faces,
            m.n_boundary_faces
        )));
    }
    Ok(())
}

// ==========================================================================
//  §3.3  ddt
// ==========================================================================

/// Euler implicit (backward difference, first order) - Patankar §4.2.
///
/// ```text
/// diag[P]   += sign · rho_P  · V_P · rDeltaT
/// source[P] += sign · rho0_P · V_P · rDeltaT · psi0_P
/// ```
///
/// `rho`/`rho0` are the new- and old-time densities and are `None` together
/// for an incompressible equation, where the term is `d(psi)/dt` rather than
/// `d(rho psi)/dt`. Supplying one without the other is rejected: it would
/// silently discretise neither form.
///
/// `r_delta_t == 0` writes nothing at all. That is how a steady-state run
/// drops the time derivative (`ddtSchemes { default steadyState; }`) without a
/// branch at every call site.
#[allow(clippy::too_many_arguments)]
pub fn fvm_ddt_euler(
    gpu: &Gpu,
    k: &FvKernels,
    a: &mut GpuLduMatrix,
    m: &GpuMesh,
    rho: Option<&DevBuf<Scalar>>,
    rho0: Option<&DevBuf<Scalar>>,
    psi0: &DevBuf<Scalar>,
    r_delta_t: Scalar,
    sign: Scalar,
) -> Result<()> {
    check_matrix(a, m)?;
    expect_len(psi0, m.n_cells, "psi0")?;

    let n = m.n_cells;
    if n == 0 || r_delta_t == 0.0 {
        return Ok(());
    }
    let nl = n as Label;

    match (rho, rho0) {
        (None, None) => {
            let f = k.ddt_euler.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut a.diag)
                    .arg(&mut a.source)
                    .arg(&m.v)
                    .arg(psi0)
                    .arg(&r_delta_t)
                    .arg(&sign)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }
        (Some(r), Some(r0)) => {
            expect_len(r, n, "rho")?;
            expect_len(r0, n, "rho0")?;
            let f = k.ddt_euler_rho.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut a.diag)
                    .arg(&mut a.source)
                    .arg(&m.v)
                    .arg(r)
                    .arg(r0)
                    .arg(psi0)
                    .arg(&r_delta_t)
                    .arg(&sign)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }
        _ => {
            return Err(Error::Config(
                "fv: fvm_ddt_euler was given one of rho/rho0 but not the \
                 other; a compressible ddt needs both time levels"
                    .to_string(),
            ))
        }
    }

    Ok(())
}

/// Second-order backward differencing (BDF2), constant `Δt` - Ferziger &
/// Perić §6.3.2.
///
/// ```text
/// diag[P]   += sign · 3/2 · rho_P · V_P · rDeltaT
/// source[P] += sign · V_P · rDeltaT · (2·rho0·psi0 - 1/2·rho00·psi00)
/// ```
///
/// The first step of a run has no `psi^{n-2}` and must be taken with
/// [`fvm_ddt_euler`]; the caller knows the step number and this function does
/// not, so the choice is left there rather than guessed here.
#[allow(clippy::too_many_arguments)]
pub fn fvm_ddt_bdf2(
    gpu: &Gpu,
    k: &FvKernels,
    a: &mut GpuLduMatrix,
    m: &GpuMesh,
    rho: Option<&DevBuf<Scalar>>,
    rho0: Option<&DevBuf<Scalar>>,
    rho00: Option<&DevBuf<Scalar>>,
    psi0: &DevBuf<Scalar>,
    psi00: &DevBuf<Scalar>,
    r_delta_t: Scalar,
    sign: Scalar,
) -> Result<()> {
    check_matrix(a, m)?;
    expect_len(psi0, m.n_cells, "psi0")?;
    expect_len(psi00, m.n_cells, "psi00")?;

    let n = m.n_cells;
    if n == 0 || r_delta_t == 0.0 {
        return Ok(());
    }
    let nl = n as Label;

    match (rho, rho0, rho00) {
        (None, None, None) => {
            let f = k.ddt_bdf2.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut a.diag)
                    .arg(&mut a.source)
                    .arg(&m.v)
                    .arg(psi0)
                    .arg(psi00)
                    .arg(&r_delta_t)
                    .arg(&sign)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }
        (Some(r), Some(r0), Some(r00)) => {
            expect_len(r, n, "rho")?;
            expect_len(r0, n, "rho0")?;
            expect_len(r00, n, "rho00")?;
            let f = k.ddt_bdf2_rho.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut a.diag)
                    .arg(&mut a.source)
                    .arg(&m.v)
                    .arg(r)
                    .arg(r0)
                    .arg(r00)
                    .arg(psi0)
                    .arg(psi00)
                    .arg(&r_delta_t)
                    .arg(&sign)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }
        _ => {
            return Err(Error::Config(
                "fv: fvm_ddt_bdf2 needs all three of rho/rho0/rho00 or none \
                 of them"
                    .to_string(),
            ))
        }
    }

    Ok(())
}

// ==========================================================================
//  §7  Convection weights
// ==========================================================================

/// Fill the face weights a convection assembly needs.
///
/// `w` is `[n_internal_faces]` and `bw` is `[n_boundary_faces]`; either may be
/// `None` when the caller does not want it. Both go straight on to
/// [`fvm_div_gauss`].
///
/// `grad_psi` is the **cell gradient of the transported field**, `[n_cells]`.
/// A limited scheme reads the *upwind* cell's entry to form
/// `r = 2(d·grad psi_U)/(psi_N - psi_P) - 1`, so it cannot be assembled
/// without one. Passing `None` with a limited scheme is an error rather than a
/// quiet fall-back to upwind: silently turning a second-order run into a
/// first-order one is the kind of failure that shows up as "the results look
/// smeared" three weeks later. Produce the gradient with
/// [`fvc_grad_scalar`] into a scratch buffer that lives as long as the
/// equation.
///
/// `scheme` accepts either [`DivScheme`] or [`crate::io::case::DivScheme`].
///
/// # Boundary weights
///
/// A weight is only meaningful on a **coupled** face, where two cells really
/// are being interpolated between. Everywhere else the face value comes from
/// the field's `(fr, refValue, refGrad)` triple and `bw` is never read, so `1`
/// is written there.
#[allow(clippy::too_many_arguments)]
pub fn div_scheme_weights(
    gpu: &Gpu,
    k: &FvKernels,
    w: Option<&mut DevBuf<Scalar>>,
    bw: Option<&mut DevBuf<Scalar>>,
    scheme: impl Into<DivScheme>,
    phi: &GpuSurfaceScalarField,
    psi: &GpuScalarField,
    grad_psi: Option<&DevBuf<Vec3>>,
    m: &GpuMesh,
) -> Result<()> {
    let scheme: DivScheme = scheme.into();

    expect_len(&phi.f, m.n_internal_faces, "phi.f")?;
    expect_len(&phi.bf, m.n_boundary_faces, "phi.bf")?;
    expect_len(&psi.f, m.n_cells, "psi.f")?;

    // Only a scheme whose WEIGHT is a function of r takes the limited kernel.
    // `linearUpwind` and `cubic` need a gradient too, but for their explicit
    // correction; their weights are plain upwind and plain central, and
    // sending them through the limited kernel would evaluate a `Psi` that does
    // not exist for their code.
    let grad = match (scheme.weight_needs_gradient(), grad_psi) {
        (false, _) => None,
        (true, Some(g)) => {
            expect_len(g, m.n_cells, "grad_psi")?;
            Some(g)
        }
        (true, None) => {
            return Err(Error::Config(format!(
                "fv: {scheme:?} forms its face weight from the upwind cell \
                 gradient, but div_scheme_weights was given grad_psi = None. \
                 Compute it with fvc_grad_scalar into a scratch buffer; \
                 falling back to upwind here would silently drop the scheme \
                 to first order."
            )))
        }
    };

    let cd = scheme.code();
    let beta = scheme.beta();

    // ---- internal faces --------------------------------------------------
    if let Some(w) = w {
        expect_len(w, m.n_internal_faces, "w")?;
        let n = m.n_internal_faces;
        if n > 0 {
            let nl = n as Label;
            match grad {
                None => {
                    let f = k.weights_unlimited.clone();
                    unsafe {
                        gpu.stream()
                            .launch_builder(&f)
                            .arg(w)
                            .arg(&phi.f)
                            .arg(&m.weights)
                            .arg(&cd)
                            .arg(&beta)
                            .arg(&nl)
                            .launch(cfg_for(n))?;
                    }
                }
                Some(g) => {
                    let f = k.weights_limited.clone();
                    unsafe {
                        gpu.stream()
                            .launch_builder(&f)
                            .arg(w)
                            .arg(&phi.f)
                            .arg(&m.weights)
                            .arg(&psi.f)
                            .arg(g)
                            .arg(&m.c)
                            .arg(&m.owner)
                            .arg(&m.neighbour)
                            .arg(&cd)
                            .arg(&beta)
                            .arg(&nl)
                            .launch(cfg_for(n))?;
                    }
                }
            }
        }
    }

    // ---- boundary faces --------------------------------------------------
    if let Some(bw) = bw {
        expect_len(bw, m.n_boundary_faces, "bw")?;
        let n = m.n_boundary_faces;
        if n > 0 {
            let nl = n as Label;
            match grad {
                None => {
                    let f = k.weights_boundary.clone();
                    unsafe {
                        gpu.stream()
                            .launch_builder(&f)
                            .arg(bw)
                            .arg(&phi.bf)
                            .arg(&m.b_weights)
                            .arg(&m.b_kind)
                            .arg(&cd)
                            .arg(&beta)
                            .arg(&nl)
                            .launch(cfg_for(n))?;
                    }
                }
                Some(g) => {
                    let f = k.weights_boundary_limited.clone();
                    unsafe {
                        gpu.stream()
                            .launch_builder(&f)
                            .arg(bw)
                            .arg(&phi.bf)
                            .arg(&m.b_weights)
                            .arg(&psi.f)
                            .arg(g)
                            .arg(&m.b_sf)
                            .arg(&m.b_mag_sf)
                            .arg(&m.b_delta_coeffs)
                            .arg(&m.b_face_cells)
                            .arg(&m.b_nbr_cell)
                            .arg(&m.b_kind)
                            .arg(&cd)
                            .arg(&beta)
                            .arg(&nl)
                            .launch(cfg_for(n))?;
                    }
                }
            }
        }
    }

    Ok(())
}

// ==========================================================================
//  §3.1  Convection
// ==========================================================================

/// Gauss convection, `sign · div(phi, psi)`.
///
/// ```text
/// lower[f] += sign · (-w_f·phi_f)
/// upper[f] += sign · ( (1-w_f)·phi_f )
/// diag[P]  += sign · w_f·phi_f              for a face P owns
/// diag[P]  += sign · (-(1-w_f)·phi_f)       for a face P neighbours
/// ```
///
/// `w`/`bw` come from [`div_scheme_weights`] and select the scheme; this
/// function is the same for every one of them, which is exactly why the
/// limiter lives in the weights and not here.
///
/// The diagonal pass recomputes `w·phi` rather than reading `upper`/`lower`
/// back - see the module note.
#[allow(clippy::too_many_arguments)]
pub fn fvm_div_gauss(
    gpu: &Gpu,
    k: &FvKernels,
    a: &mut GpuLduMatrix,
    m: &GpuMesh,
    phi: &GpuSurfaceScalarField,
    w: &DevBuf<Scalar>,
    bw: &DevBuf<Scalar>,
    psi: &GpuScalarField,
    sign: Scalar,
) -> Result<()> {
    check_matrix(a, m)?;
    expect_len(&phi.f, m.n_internal_faces, "phi.f")?;
    expect_len(&phi.bf, m.n_boundary_faces, "phi.bf")?;
    expect_len(w, m.n_internal_faces, "w")?;
    expect_len(bw, m.n_boundary_faces, "bw")?;
    expect_len(&psi.fr, m.n_boundary_faces, "psi.fr")?;

    if m.n_internal_faces > 0 {
        let n = m.n_internal_faces;
        let nl = n as Label;
        let f = k.div_faces.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut a.upper)
                .arg(&mut a.lower)
                .arg(&phi.f)
                .arg(w)
                .arg(&sign)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    if m.n_cells > 0 {
        let n = m.n_cells;
        let nl = n as Label;
        let f = k.div_diag.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut a.diag)
                .arg(&phi.f)
                .arg(w)
                .arg(&m.cf_offset)
                .arg(&m.cf_face)
                .arg(&m.cf_own)
                .arg(&sign)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    if m.n_boundary_faces > 0 {
        let n = m.n_boundary_faces;
        let nl = n as Label;
        let f = k.div_boundary.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut a.internal_coeffs)
                .arg(&mut a.boundary_coeffs)
                .arg(&phi.bf)
                .arg(bw)
                .arg(&psi.fr)
                .arg(&psi.ref_value)
                .arg(&psi.ref_grad)
                .arg(&m.b_delta_coeffs)
                .arg(&m.b_kind)
                .arg(&sign)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    Ok(())
}

/// The explicit half of a deferred-correction convection scheme
/// (SPEC-LIT §11.1, Khosla & Rubin 1974).
///
/// ```text
/// source[P] -= sign · phi_f · corr_f          for a face P owns
/// source[N] += sign · phi_f · corr_f
/// corr_f = psi_f,scheme - psi_f,implicit
/// ```
///
/// Apply it with the **same `sign`** as the [`fvm_div_gauss`] whose weights it
/// corrects, and after the source has been zeroed for this assembly.
///
/// `grad_psi` is the cell gradient of the transported field, `[n_cells]`, the
/// same one [`div_scheme_weights`] takes. `scheme` decides what `corr_f` is:
///
/// * [`DivScheme::LinearUpwind`] - `(C_f - C_U)·grad(psi)_U`, §11.2;
/// * [`DivScheme::LinearUpwindBlended`] - the same, scaled by `1 - gamma`,
///   §11.5;
/// * [`DivScheme::Cubic`] - `[d·grad_P - d·grad_N]/8`, §11.4;
/// * everything else - nothing, and the launch is skipped entirely.
///
/// Skipping it for a scheme that wants it does not fail, it just silently
/// assembles the implicit base: plain upwind for `linearUpwind`, plain central
/// for `cubic`. That is precisely the bug this function exists to fix, so
/// check [`DivScheme::correction`] rather than assuming.
pub fn fvm_div_correction(
    gpu: &Gpu,
    k: &FvKernels,
    a: &mut GpuLduMatrix,
    m: &GpuMesh,
    phi: &GpuSurfaceScalarField,
    grad_psi: &DevBuf<Vec3>,
    scheme: impl Into<DivScheme>,
    sign: Scalar,
) -> Result<()> {
    let scheme: DivScheme = scheme.into();
    let corr = scheme.correction();
    if !corr.is_some() {
        return Ok(());
    }

    check_matrix(a, m)?;
    expect_len(&phi.f, m.n_internal_faces, "phi.f")?;
    expect_len(&phi.bf, m.n_boundary_faces, "phi.bf")?;
    expect_len(grad_psi, m.n_cells, "grad_psi")?;

    let n = m.n_cells;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let cd = corr.code();
    let coef = corr.coef();

    let f = k.div_correction.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(&mut a.source)
            .arg(&phi.f)
            .arg(&phi.bf)
            .arg(grad_psi)
            .arg(&m.c)
            .arg(&m.cf)
            .arg(&m.b_sf)
            .arg(&m.b_mag_sf)
            .arg(&m.b_delta_coeffs)
            .arg(&m.b_weights)
            .arg(&m.owner)
            .arg(&m.neighbour)
            .arg(&m.b_face_cells)
            .arg(&m.b_nbr_cell)
            .arg(&m.b_kind)
            .arg(&m.cf_offset)
            .arg(&m.cf_face)
            .arg(&m.cf_own)
            .arg(&m.bcf_offset)
            .arg(&m.bcf_face)
            .arg(&cd)
            .arg(&coef)
            .arg(&sign)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }

    Ok(())
}

/// The face value a convection scheme forms, on the internal faces.
///
/// ```text
/// psi_f = w·psi_P + (1-w)·psi_N + corr_f
/// ```
///
/// The whole scheme, implicit weights and deferred correction together. The
/// assembly does not go through here - [`fvm_div_gauss`] and
/// [`fvm_div_correction`] put the same arithmetic into the matrix and the
/// source directly - but the **order of accuracy** of a scheme is a statement
/// about this number, so a convergence test that measures it against the exact
/// value at the face centre is measuring the scheme and not the midpoint
/// quadrature rule the Gauss divergence is built on. It is also what a caller
/// wanting the convected face value itself would ask for.
///
/// `w` is what [`div_scheme_weights`] wrote for the same `scheme`.
#[allow(clippy::too_many_arguments)]
pub fn div_scheme_face_value(
    gpu: &Gpu,
    k: &FvKernels,
    out: &mut DevBuf<Scalar>,
    scheme: impl Into<DivScheme>,
    phi: &GpuSurfaceScalarField,
    psi: &GpuScalarField,
    w: &DevBuf<Scalar>,
    grad_psi: &DevBuf<Vec3>,
    m: &GpuMesh,
) -> Result<()> {
    let scheme: DivScheme = scheme.into();
    let corr = scheme.correction();

    expect_len(out, m.n_internal_faces, "out")?;
    expect_len(w, m.n_internal_faces, "w")?;
    expect_len(&phi.f, m.n_internal_faces, "phi.f")?;
    expect_len(&psi.f, m.n_cells, "psi.f")?;
    expect_len(grad_psi, m.n_cells, "grad_psi")?;

    let n = m.n_internal_faces;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let cd = corr.code();
    let coef = corr.coef();

    let f = k.div_scheme_face_value.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(out)
            .arg(&psi.f)
            .arg(w)
            .arg(&phi.f)
            .arg(grad_psi)
            .arg(&m.c)
            .arg(&m.cf)
            .arg(&m.owner)
            .arg(&m.neighbour)
            .arg(&cd)
            .arg(&coef)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }

    Ok(())
}

/// The bounded-convection correction, `- sign · Sp(div(phi), psi)`
/// (Moukalled et al. §15.4, SPEC-LIT §3.1).
///
/// ```text
/// diag[P] -= sign · Σ_f (±phi_f)
/// ```
///
/// Part-way through a pressure-velocity iteration the discrete flux is not
/// solenoidal, and the convection operator then injects a spurious source
/// proportional to `psi·(Σ_f phi_f)`. Subtracting `V_P·(∇·u)_P` from the
/// diagonal removes exactly that, and leaves the diagonal equal to minus the
/// row's off-diagonal sum - the discrete statement that a uniform field is
/// convected without change. When `phi` *is* conservative the correction is
/// identically zero and costs only the pass.
///
/// Apply it with the same `sign` as the [`fvm_div_gauss`] it corrects.
pub fn fvm_div_bounded_correction(
    gpu: &Gpu,
    k: &FvKernels,
    a: &mut GpuLduMatrix,
    m: &GpuMesh,
    phi: &GpuSurfaceScalarField,
    sign: Scalar,
) -> Result<()> {
    check_matrix(a, m)?;
    expect_len(&phi.f, m.n_internal_faces, "phi.f")?;
    expect_len(&phi.bf, m.n_boundary_faces, "phi.bf")?;

    let n = m.n_cells;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let f = k.div_bounded_diag.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(&mut a.diag)
            .arg(&phi.f)
            .arg(&phi.bf)
            .arg(&m.b_kind)
            .arg(&m.cf_offset)
            .arg(&m.cf_face)
            .arg(&m.cf_own)
            .arg(&m.bcf_offset)
            .arg(&m.bcf_face)
            .arg(&sign)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }

    Ok(())
}

// ==========================================================================
//  §3.2  Diffusion
// ==========================================================================

/// Gauss laplacian, `sign · laplacian(gamma, psi)` - the implicit, orthogonal
/// part.
///
/// ```text
/// upper[f] = lower[f] += sign · gammaMagSf_f · Delta_f
/// diag[P]            -= sign · gammaMagSf_f · Delta_f     for every face of P
/// ```
///
/// `gamma_mag_sf` is `gamma_f · |Sf_f|` **already multiplied together**,
/// `[n_internal_faces]`, and `b_gamma_mag_sf` the same on the boundary. Taking
/// the product rather than a bare `gamma` removes any doubt at a call site
/// about which of the two was meant, and lets a caller with `gamma = 1` pass
/// the mesh's own `mag_sf` with no scratch field at all.
///
/// The non-orthogonal part of `snGrad` is **not** included; it is explicit and
/// iterated, and belongs to [`fvm_laplacian_non_orth_correction`]. On an
/// orthogonal mesh that correction is identically zero.
///
/// The diagonal pass recomputes `gammaMagSf·Delta` rather than reading
/// `upper`/`lower` back - see the module note.
#[allow(clippy::too_many_arguments)]
pub fn fvm_laplacian(
    gpu: &Gpu,
    k: &FvKernels,
    a: &mut GpuLduMatrix,
    m: &GpuMesh,
    gamma_mag_sf: &DevBuf<Scalar>,
    b_gamma_mag_sf: &DevBuf<Scalar>,
    psi: &GpuScalarField,
    sign: Scalar,
) -> Result<()> {
    check_matrix(a, m)?;
    expect_len(gamma_mag_sf, m.n_internal_faces, "gamma_mag_sf")?;
    expect_len(b_gamma_mag_sf, m.n_boundary_faces, "b_gamma_mag_sf")?;
    expect_len(&psi.fr, m.n_boundary_faces, "psi.fr")?;

    if m.n_internal_faces > 0 {
        let n = m.n_internal_faces;
        let nl = n as Label;
        let f = k.lap_faces.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut a.upper)
                .arg(&mut a.lower)
                .arg(gamma_mag_sf)
                .arg(&m.delta_coeffs)
                .arg(&sign)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    if m.n_cells > 0 {
        let n = m.n_cells;
        let nl = n as Label;
        let f = k.lap_diag.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut a.diag)
                .arg(gamma_mag_sf)
                .arg(&m.delta_coeffs)
                .arg(&m.cf_offset)
                .arg(&m.cf_face)
                .arg(&m.cf_own)
                .arg(&sign)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    if m.n_boundary_faces > 0 {
        let n = m.n_boundary_faces;
        let nl = n as Label;
        let f = k.lap_boundary.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut a.internal_coeffs)
                .arg(&mut a.boundary_coeffs)
                .arg(b_gamma_mag_sf)
                .arg(&m.b_delta_coeffs)
                .arg(&psi.fr)
                .arg(&psi.ref_value)
                .arg(&psi.ref_grad)
                .arg(&m.b_kind)
                .arg(&sign)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    Ok(())
}

/// The explicit non-orthogonal correction of the laplacian (Jasak §3.4.3).
///
/// ```text
/// source[P] -= sign · [ Σ_f (±1)·gammaMagSf_f·( k_f · (grad psi)_f )
///                     + Σ_b  fr_b ·bGammaMagSf_b·( k_b · (grad psi)_P ) ]
/// ```
///
/// with `k` the over-relaxed correction vector of SPEC-LIT §2.4 and
/// `(grad psi)_f` the linear interpolation of the two cell gradients. Deferred
/// to the source and iterated: with `nNonOrthogonalCorrectors` extra passes,
/// `grad_psi` is recomputed from the latest solution and this is applied again
/// to a freshly-zeroed source.
///
/// On an orthogonal mesh `k = 0` everywhere and the whole pass is
/// arithmetically a no-op, so a caller may skip it on the strength of the mesh
/// report.
///
/// # Why the boundary term is here, and why it carries `fr`
///
/// It is tempting to leave the boundary out on the grounds that the mixed form
/// of §4 "already gives `snGrad`". It does not: for the Dirichlet part of the
/// condition, `snGrad_b = fr·Delta_b·(refValue - psi_P)` is an *estimate* of
/// the normal gradient obtained by differencing across `d_b`, and on a
/// non-orthogonal mesh that estimate is wrong by exactly `k_b·grad psi`. Omit
/// it and the solve is first order however well the interior is corrected -
/// `tests::the_non_orthogonal_correction_restores_second_order` measures the
/// difference on a sheared mesh. The `(1 - fr)` part of the condition is a
/// *prescribed* normal gradient, which is already the normal gradient and has
/// nothing to correct, hence the `fr` factor.
///
/// `k_b` is rebuilt in the kernel from `Sf_b`, `Cf_b`, `C_P` and `Delta_b`,
/// because the mesh carries a correction vector for internal faces only -
/// for an UNCOUPLED patch, where the condition is imposed directly on the
/// face and `d = Cf - C_P` really is the separation the condition sees.
///
/// A cyclic couple is different: it is one internal face folded in half, so
/// it gets the identical over-relaxed correction an internal face gets,
/// through `m.b_non_orth_corr` (`SPEC-LIT` §2.4, built by
/// `mesh/geometry.rs::compute` from the `d` that spans the periodic image,
/// not from `Cf - C_P`) and a gradient interpolated between the two coupled
/// cells via `m.b_nbr_cell`/`m.b_weights` exactly as an internal face's is.
/// Before this these faces were skipped outright, so a cyclic boundary on a
/// sheared mesh silently lost its non-orthogonal correction.
///
/// # The `sn_grad` argument
///
/// SPEC-LIT §12.3. [`SnGradScheme::Corrected`] applies the correction in full,
/// which is what this function always used to do. [`SnGradScheme::Limited`]
/// caps it at `alpha` times the orthogonal part, and
/// [`SnGradScheme::Uncorrected`] returns without launching anything - which
/// gives bit-identical results to `Limited(0.0)`, since the kernel's scale
/// factor is then exactly zero and `source -= 0.0` changes no bits.
///
/// This is a *different question* from `nNonOrthogonalCorrectors`, which says
/// how many times the caller re-runs this pass against a fresher solution.
#[allow(clippy::too_many_arguments)]
pub fn fvm_laplacian_non_orth_correction(
    gpu: &Gpu,
    k: &FvKernels,
    a: &mut GpuLduMatrix,
    m: &GpuMesh,
    gamma_mag_sf: &DevBuf<Scalar>,
    b_gamma_mag_sf: &DevBuf<Scalar>,
    psi: &GpuScalarField,
    grad_psi: &DevBuf<Vec3>,
    sn_grad: SnGradScheme,
    sign: Scalar,
) -> Result<()> {
    check_matrix(a, m)?;
    expect_len(gamma_mag_sf, m.n_internal_faces, "gamma_mag_sf")?;
    expect_len(b_gamma_mag_sf, m.n_boundary_faces, "b_gamma_mag_sf")?;
    expect_len(&psi.f, m.n_cells, "psi.f")?;
    expect_len(&psi.fr, m.n_boundary_faces, "psi.fr")?;
    expect_len(grad_psi, m.n_cells, "grad_psi")?;

    if !sn_grad.applies() {
        return Ok(());
    }

    let n = m.n_cells;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let alpha = sn_grad.alpha();

    let f = k.lap_non_orth.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(&mut a.source)
            .arg(gamma_mag_sf)
            .arg(&m.non_orth_corr)
            .arg(&m.weights)
            .arg(&m.delta_coeffs)
            .arg(&psi.f)
            .arg(grad_psi)
            .arg(b_gamma_mag_sf)
            .arg(&m.b_sf)
            .arg(&m.b_mag_sf)
            .arg(&m.b_cf)
            .arg(&m.b_delta_coeffs)
            .arg(&m.b_non_orth_corr)
            .arg(&m.b_nbr_cell)
            .arg(&m.b_weights)
            .arg(&psi.fr)
            .arg(&psi.ref_value)
            .arg(&psi.ref_grad)
            .arg(&m.c)
            .arg(&m.owner)
            .arg(&m.neighbour)
            .arg(&m.b_kind)
            .arg(&m.cf_offset)
            .arg(&m.cf_face)
            .arg(&m.cf_own)
            .arg(&m.bcf_offset)
            .arg(&m.bcf_face)
            .arg(&alpha)
            .arg(&sign)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }

    Ok(())
}

// ==========================================================================
//  §3.4  Source terms
// ==========================================================================

/// An implicit source whose sign the caller has already settled:
/// `diag[P] += sign · V_P · sp_P`.
///
/// Patankar §4.2 requires the linearised coefficient `S_p` to be negative for
/// the matrix to stay diagonally dominant. Written this way, a *sink* of
/// magnitude `sp > 0` is `fvm_sp(.., sp, +1)` on the left-hand side of
/// `A psi = b`, which is the stabilising sign. Use [`fvm_susp`] when the sign
/// is not known in advance.
pub fn fvm_sp(
    gpu: &Gpu,
    k: &FvKernels,
    a: &mut GpuLduMatrix,
    m: &GpuMesh,
    sp: &DevBuf<Scalar>,
    sign: Scalar,
) -> Result<()> {
    check_matrix(a, m)?;
    expect_len(sp, m.n_cells, "sp")?;

    let n = m.n_cells;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let f = k.sp.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(&mut a.diag)
            .arg(&m.v)
            .arg(sp)
            .arg(&sign)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }

    Ok(())
}

/// Patankar's linearisation for a source of unknown sign (§4.2, SPEC-LIT
/// §3.4):
///
/// ```text
/// diag[P]   += sign · V_P · max(S, 0)
/// source[P] -= sign · V_P · min(S, 0) · psi_P
/// ```
///
/// Whichever part stabilises the matrix goes on the diagonal and the rest goes
/// to the right-hand side at the current `psi`. The two branches agree exactly
/// when `S >= 0`, and when `S < 0` the explicit branch is precisely what a
/// fully implicit treatment would have moved across anyway - so this is a
/// stability choice, not an approximation.
///
/// `psi` is the **current cell values** of the field being solved for, i.e.
/// `field.f`.
pub fn fvm_susp(
    gpu: &Gpu,
    k: &FvKernels,
    a: &mut GpuLduMatrix,
    m: &GpuMesh,
    susp: &DevBuf<Scalar>,
    psi: &DevBuf<Scalar>,
    sign: Scalar,
) -> Result<()> {
    check_matrix(a, m)?;
    expect_len(susp, m.n_cells, "susp")?;
    expect_len(psi, m.n_cells, "psi")?;

    let n = m.n_cells;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let f = k.susp.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(&mut a.diag)
            .arg(&mut a.source)
            .arg(&m.v)
            .arg(susp)
            .arg(psi)
            .arg(&sign)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }

    Ok(())
}

/// A wholly explicit source: `source[P] += sign · V_P · su_P`.
pub fn fvm_su(
    gpu: &Gpu,
    k: &FvKernels,
    a: &mut GpuLduMatrix,
    m: &GpuMesh,
    su: &DevBuf<Scalar>,
    sign: Scalar,
) -> Result<()> {
    check_matrix(a, m)?;
    expect_len(su, m.n_cells, "su")?;

    let n = m.n_cells;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let f = k.su.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(&mut a.source)
            .arg(&m.v)
            .arg(su)
            .arg(&sign)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }

    Ok(())
}

// ==========================================================================
//  §3.5  Explicit operators
// ==========================================================================

/// Green-Gauss gradient of a cell scalar field (Jasak §3.3):
/// `(grad psi)_P = (1/V_P) Σ_f (±Sf) psi_f`.
///
/// Reads the field's **evaluated** boundary values `psi.bf`, so
/// `field_ops::correct_boundary_conditions` must have run since the internal
/// field last changed. Empty faces contribute nothing.
pub fn fvc_grad_scalar(
    gpu: &Gpu,
    k: &FvKernels,
    out: &mut DevBuf<Vec3>,
    psi: &GpuScalarField,
    m: &GpuMesh,
) -> Result<()> {
    fvc_grad_scalar_scheme(gpu, k, out, psi, m, GradScheme::GAUSS)
}

/// The gradient of a cell scalar field under a chosen `gradSchemes` entry
/// (SPEC-LIT §3.5, §12.1, §12.2).
///
/// [`fvc_grad_scalar`] is this with [`GradScheme::GAUSS`], which is what every
/// operator in this crate used before `gradSchemes` was read at all.
///
/// Reads the field's **evaluated** boundary values `psi.bf`, so
/// `field_ops::correct_boundary_conditions` must have run since the internal
/// field last changed.
pub fn fvc_grad_scalar_scheme(
    gpu: &Gpu,
    k: &FvKernels,
    out: &mut DevBuf<Vec3>,
    psi: &GpuScalarField,
    m: &GpuMesh,
    scheme: GradScheme,
) -> Result<()> {
    expect_len(out, m.n_cells, "out")?;
    expect_len(&psi.f, m.n_cells, "psi.f")?;
    expect_len(&psi.bf, m.n_boundary_faces, "psi.bf")?;

    let n = m.n_cells;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    match scheme.base {
        GradBase::Gauss => {
            let f = k.grad_scalar.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut *out)
                    .arg(&psi.f)
                    .arg(&psi.bf)
                    .arg(&m.weights)
                    .arg(&m.sf)
                    .arg(&m.b_sf)
                    .arg(&m.v)
                    .arg(&m.owner)
                    .arg(&m.neighbour)
                    .arg(&m.b_kind)
                    .arg(&m.cf_offset)
                    .arg(&m.cf_face)
                    .arg(&m.cf_own)
                    .arg(&m.bcf_offset)
                    .arg(&m.bcf_face)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }
        GradBase::LeastSquares => {
            let f = k.grad_scalar_ls.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut *out)
                    .arg(&psi.f)
                    .arg(&psi.bf)
                    .arg(&m.c)
                    .arg(&m.b_sf)
                    .arg(&m.b_mag_sf)
                    .arg(&m.b_cf)
                    .arg(&m.b_delta_coeffs)
                    .arg(&m.owner)
                    .arg(&m.neighbour)
                    .arg(&m.b_nbr_cell)
                    .arg(&m.b_kind)
                    .arg(&m.cf_offset)
                    .arg(&m.cf_face)
                    .arg(&m.cf_own)
                    .arg(&m.bcf_offset)
                    .arg(&m.bcf_face)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }
    }

    if let Some((kind, mode, coeff)) = scheme.limiter_args() {
        let f = k.grad_limit_scalar.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut *out)
                .arg(&psi.f)
                .arg(&psi.bf)
                .arg(&m.c)
                .arg(&m.cf)
                .arg(&m.b_cf)
                .arg(&m.owner)
                .arg(&m.neighbour)
                .arg(&m.b_nbr_cell)
                .arg(&m.b_kind)
                .arg(&m.cf_offset)
                .arg(&m.cf_face)
                .arg(&m.cf_own)
                .arg(&m.bcf_offset)
                .arg(&m.bcf_face)
                .arg(&kind)
                .arg(&mode)
                .arg(&coeff)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    Ok(())
}

/// The gradient of a cell vector field under a chosen `gradSchemes` entry.
/// Component `(i,j)` is `dU_j/dx_i`, as everywhere else in this crate.
pub fn fvc_grad_vector_scheme(
    gpu: &Gpu,
    k: &FvKernels,
    out: &mut DevBuf<Tensor>,
    u: &GpuVectorField,
    m: &GpuMesh,
    scheme: GradScheme,
) -> Result<()> {
    expect_len(out, m.n_cells, "out")?;
    expect_len(&u.f, m.n_cells, "u.f")?;
    expect_len(&u.bf, m.n_boundary_faces, "u.bf")?;

    let n = m.n_cells;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    match scheme.base {
        GradBase::Gauss => {
            let f = k.grad_vector.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut *out)
                    .arg(&u.f)
                    .arg(&u.bf)
                    .arg(&m.weights)
                    .arg(&m.sf)
                    .arg(&m.b_sf)
                    .arg(&m.v)
                    .arg(&m.owner)
                    .arg(&m.neighbour)
                    .arg(&m.b_kind)
                    .arg(&m.cf_offset)
                    .arg(&m.cf_face)
                    .arg(&m.cf_own)
                    .arg(&m.bcf_offset)
                    .arg(&m.bcf_face)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }
        GradBase::LeastSquares => {
            let f = k.grad_vector_ls.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut *out)
                    .arg(&u.f)
                    .arg(&u.bf)
                    .arg(&m.c)
                    .arg(&m.b_sf)
                    .arg(&m.b_mag_sf)
                    .arg(&m.b_cf)
                    .arg(&m.b_delta_coeffs)
                    .arg(&m.owner)
                    .arg(&m.neighbour)
                    .arg(&m.b_nbr_cell)
                    .arg(&m.b_kind)
                    .arg(&m.cf_offset)
                    .arg(&m.cf_face)
                    .arg(&m.cf_own)
                    .arg(&m.bcf_offset)
                    .arg(&m.bcf_face)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }
    }

    if let Some((kind, mode, coeff)) = scheme.limiter_args() {
        let f = k.grad_limit_vector.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut *out)
                .arg(&u.f)
                .arg(&u.bf)
                .arg(&m.c)
                .arg(&m.cf)
                .arg(&m.b_cf)
                .arg(&m.owner)
                .arg(&m.neighbour)
                .arg(&m.b_nbr_cell)
                .arg(&m.b_kind)
                .arg(&m.cf_offset)
                .arg(&m.cf_face)
                .arg(&m.cf_own)
                .arg(&m.bcf_offset)
                .arg(&m.bcf_face)
                .arg(&kind)
                .arg(&mode)
                .arg(&coeff)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    Ok(())
}

/// Green-Gauss gradient of a cell vector field:
/// `(grad U)_P = (1/V_P) Σ_f (±Sf) ⊗ U_f`.
///
/// Component `(i,j)` is `dU_j/dx_i`, because the area vector supplies the
/// first index (SPEC-LIT §1). That is the convention [`crate::Tensor`] is
/// documented and tested against.
pub fn fvc_grad_vector(
    gpu: &Gpu,
    k: &FvKernels,
    out: &mut DevBuf<Tensor>,
    u: &GpuVectorField,
    m: &GpuMesh,
) -> Result<()> {
    fvc_grad_vector_scheme(gpu, k, out, u, m, GradScheme::GAUSS)
}

/// Divergence of a face flux: `(div phi)_P = (1/V_P) Σ_f (±phi_f)`.
///
/// Volumetric, i.e. divided by the cell volume, so that feeding the result to
/// [`fvm_sp`] reproduces `Σ_f (±phi_f)` exactly after the `V` multiply - which
/// is what makes the bounded-convection correction expressible two ways and
/// identical both times.
pub fn fvc_div_surface(
    gpu: &Gpu,
    k: &FvKernels,
    out: &mut DevBuf<Scalar>,
    phi: &GpuSurfaceScalarField,
    m: &GpuMesh,
) -> Result<()> {
    expect_len(out, m.n_cells, "out")?;
    expect_len(&phi.f, m.n_internal_faces, "phi.f")?;
    expect_len(&phi.bf, m.n_boundary_faces, "phi.bf")?;

    let n = m.n_cells;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let f = k.div_surface.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(out)
            .arg(&phi.f)
            .arg(&phi.bf)
            .arg(&m.v)
            .arg(&m.b_kind)
            .arg(&m.cf_offset)
            .arg(&m.cf_face)
            .arg(&m.cf_own)
            .arg(&m.bcf_offset)
            .arg(&m.bcf_face)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }

    Ok(())
}

/// Linear interpolation of a cell scalar onto the faces, `psi_f = w·psi_P +
/// (1-w)·psi_N`, with the boundary faces taking the field's evaluated value.
pub fn interpolate_linear(
    gpu: &Gpu,
    k: &FvKernels,
    out: &mut GpuSurfaceScalarField,
    psi: &GpuScalarField,
    m: &GpuMesh,
) -> Result<()> {
    expect_len(&out.f, m.n_internal_faces, "out.f")?;
    expect_len(&out.bf, m.n_boundary_faces, "out.bf")?;
    expect_len(&psi.f, m.n_cells, "psi.f")?;
    expect_len(&psi.bf, m.n_boundary_faces, "psi.bf")?;

    if m.n_internal_faces > 0 {
        let n = m.n_internal_faces;
        let nl = n as Label;
        let f = k.interpolate_linear.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut out.f)
                .arg(&psi.f)
                .arg(&m.weights)
                .arg(&m.owner)
                .arg(&m.neighbour)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    if m.n_boundary_faces > 0 {
        let n = m.n_boundary_faces;
        let nl = n as Label;
        let f = k.interpolate_boundary.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut out.bf)
                .arg(&psi.bf)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    Ok(())
}

/// `phi_f = interpolate(U)_f · Sf`.
///
/// **Not** a conservative flux: nothing constrains an interpolated cell
/// velocity to satisfy the discrete continuity equation. See
/// [`crate::potential_flow`] for the flux that does. This exists because a
/// SIMPLE loop needs somewhere to start and because the Rhie-Chow correction
/// is applied *to* this flux rather than replacing it.
pub fn interpolate_vector_flux(
    gpu: &Gpu,
    k: &FvKernels,
    out: &mut GpuSurfaceScalarField,
    u: &GpuVectorField,
    m: &GpuMesh,
) -> Result<()> {
    expect_len(&out.f, m.n_internal_faces, "out.f")?;
    expect_len(&out.bf, m.n_boundary_faces, "out.bf")?;
    expect_len(&u.f, m.n_cells, "u.f")?;
    expect_len(&u.bf, m.n_boundary_faces, "u.bf")?;

    if m.n_internal_faces > 0 {
        let n = m.n_internal_faces;
        let nl = n as Label;
        let f = k.flux_internal.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut out.f)
                .arg(&u.f)
                .arg(&m.weights)
                .arg(&m.sf)
                .arg(&m.owner)
                .arg(&m.neighbour)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    if m.n_boundary_faces > 0 {
        let n = m.n_boundary_faces;
        let nl = n as Label;
        let f = k.flux_boundary.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut out.bf)
                .arg(&u.bf)
                .arg(&m.b_sf)
                .arg(&m.b_kind)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    Ok(())
}

/// The diffusive face flux `gamma_f·|Sf|·snGrad(psi)_f`, written into `phi` on
/// internal and boundary faces alike.
///
/// Deliberately written against the same coefficients in the same
/// multiplication order as [`fvm_laplacian`], and the boundary half is rebuilt
/// from `(fr, refValue, refGrad)` rather than from an evaluated face value. A
/// flux read off after solving `laplacian(gamma, psi) = 0` therefore satisfies
/// the discrete conservation statement the matrix enforced, to the last bit,
/// whether or not the field's faces have been corrected since.
pub fn sn_grad_flux(
    gpu: &Gpu,
    k: &FvKernels,
    phi: &mut GpuSurfaceScalarField,
    psi: &GpuScalarField,
    gamma_mag_sf: &DevBuf<Scalar>,
    b_gamma_mag_sf: &DevBuf<Scalar>,
    m: &GpuMesh,
) -> Result<()> {
    expect_len(&phi.f, m.n_internal_faces, "phi.f")?;
    expect_len(&phi.bf, m.n_boundary_faces, "phi.bf")?;
    expect_len(gamma_mag_sf, m.n_internal_faces, "gamma_mag_sf")?;
    expect_len(b_gamma_mag_sf, m.n_boundary_faces, "b_gamma_mag_sf")?;
    expect_len(&psi.f, m.n_cells, "psi.f")?;

    if m.n_internal_faces > 0 {
        let n = m.n_internal_faces;
        let nl = n as Label;
        let f = k.sn_grad_internal.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut phi.f)
                .arg(&psi.f)
                .arg(gamma_mag_sf)
                .arg(&m.delta_coeffs)
                .arg(&m.owner)
                .arg(&m.neighbour)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    if m.n_boundary_faces > 0 {
        let n = m.n_boundary_faces;
        let nl = n as Label;
        let f = k.sn_grad_boundary.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut phi.bf)
                .arg(&psi.f)
                .arg(b_gamma_mag_sf)
                .arg(&m.b_delta_coeffs)
                .arg(&psi.fr)
                .arg(&psi.ref_value)
                .arg(&psi.ref_grad)
                .arg(&m.b_face_cells)
                .arg(&m.b_nbr_cell)
                .arg(&m.b_kind)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    Ok(())
}

/// The non-orthogonal correction of the diffusive flux, **added** to an
/// existing `phi`.
///
/// [`sn_grad_flux`] reproduces [`fvm_laplacian`] exactly; this reproduces
/// [`fvm_laplacian_non_orth_correction`] exactly. Apply both and the flux is
/// the full operator, so a cell's fluxes sum to the residual of the row the
/// matrix solved - which is what keeps
/// `phi = phi_HbyA - rAUf·|Sf|·snGrad(p)` conservative on a non-orthogonal
/// mesh. On an orthogonal mesh it adds nothing and can be skipped.
///
/// Separate from [`sn_grad_flux`] rather than an option on it, because the
/// uncorrected pair is a complete and self-consistent operator on its own -
/// that is exactly what [`crate::potential_flow`] uses - and because the
/// correction needs a gradient that a single-pass caller has no reason to
/// compute.
#[allow(clippy::too_many_arguments)]
pub fn sn_grad_flux_correction(
    gpu: &Gpu,
    k: &FvKernels,
    phi: &mut GpuSurfaceScalarField,
    psi: &GpuScalarField,
    gamma_mag_sf: &DevBuf<Scalar>,
    b_gamma_mag_sf: &DevBuf<Scalar>,
    grad_psi: &DevBuf<Vec3>,
    sn_grad: SnGradScheme,
    m: &GpuMesh,
) -> Result<()> {
    expect_len(&phi.f, m.n_internal_faces, "phi.f")?;
    expect_len(&phi.bf, m.n_boundary_faces, "phi.bf")?;
    expect_len(gamma_mag_sf, m.n_internal_faces, "gamma_mag_sf")?;
    expect_len(b_gamma_mag_sf, m.n_boundary_faces, "b_gamma_mag_sf")?;
    expect_len(&psi.f, m.n_cells, "psi.f")?;
    expect_len(&psi.fr, m.n_boundary_faces, "psi.fr")?;
    expect_len(grad_psi, m.n_cells, "grad_psi")?;

    // The same short-circuit as fvm_laplacian_non_orth_correction, and for the
    // same reason: the flux and the matrix must agree face for face, so the
    // two functions have to take the same decision from the same scheme.
    if !sn_grad.applies() {
        return Ok(());
    }
    let alpha = sn_grad.alpha();

    if m.n_internal_faces > 0 {
        let n = m.n_internal_faces;
        let nl = n as Label;
        let f = k.sn_grad_corr_internal.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut phi.f)
                .arg(gamma_mag_sf)
                .arg(&m.non_orth_corr)
                .arg(&m.weights)
                .arg(&m.delta_coeffs)
                .arg(&psi.f)
                .arg(grad_psi)
                .arg(&m.owner)
                .arg(&m.neighbour)
                .arg(&alpha)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    if m.n_boundary_faces > 0 {
        let n = m.n_boundary_faces;
        let nl = n as Label;
        let f = k.sn_grad_corr_boundary.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut phi.bf)
                .arg(b_gamma_mag_sf)
                .arg(&m.b_sf)
                .arg(&m.b_mag_sf)
                .arg(&m.b_cf)
                .arg(&m.b_delta_coeffs)
                .arg(&m.b_non_orth_corr)
                .arg(&m.b_nbr_cell)
                .arg(&m.b_weights)
                .arg(&psi.fr)
                .arg(&psi.ref_value)
                .arg(&psi.ref_grad)
                .arg(&psi.f)
                .arg(&m.c)
                .arg(grad_psi)
                .arg(&m.b_face_cells)
                .arg(&m.b_kind)
                .arg(&alpha)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    Ok(())
}

/// Reconstruct a cell vector from a face flux.
///
/// *DERIVED* - SPEC-LIT gives no formula for this one. We want the cell vector
/// whose own face fluxes best reproduce the given ones; weighting each face
/// residual by `1/|Sf|` so that a large face does not dominate a small one
/// purely by area,
///
/// ```text
/// minimise  Σ_f (1/|Sf|)·(U·Sf - phi_f)²
///    =>     [ Σ_f (Sf ⊗ Sf)/|Sf| ] · U = Σ_f (Sf/|Sf|)·phi_f
/// ```
///
/// The owner/neighbour sign cancels on both sides, since flipping `Sf` flips
/// `phi_f` with it. Empty faces are deliberately **included**: on a 2-D mesh
/// they are the only faces with an area component in the unresolved direction
/// and without them the 3×3 system is singular; their flux is zero, so what
/// they contribute is exactly the constraint `U·n = 0` there.
pub fn fvc_reconstruct(
    gpu: &Gpu,
    k: &FvKernels,
    out: &mut DevBuf<Vec3>,
    phi: &GpuSurfaceScalarField,
    m: &GpuMesh,
) -> Result<()> {
    expect_len(out, m.n_cells, "out")?;
    expect_len(&phi.f, m.n_internal_faces, "phi.f")?;
    expect_len(&phi.bf, m.n_boundary_faces, "phi.bf")?;

    let n = m.n_cells;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;

    let f = k.reconstruct.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(out)
            .arg(&phi.f)
            .arg(&phi.bf)
            .arg(&m.sf)
            .arg(&m.mag_sf)
            .arg(&m.b_sf)
            .arg(&m.b_mag_sf)
            .arg(&m.cf_offset)
            .arg(&m.cf_face)
            .arg(&m.cf_own)
            .arg(&m.bcf_offset)
            .arg(&m.bcf_face)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }

    Ok(())
}

/// The turbulence production term of SPEC-LIT §6:
///
/// ```text
/// G = nu_t · ( dev(2 symm(grad U)) : grad U )
///   = nu_t · ( (grad U + grad Uᵀ) : grad U  -  (2/3)·tr(grad U)² )
/// ```
///
/// The second form is the one implemented, because it avoids building the
/// deviatoric tensor; it follows from `dev(A):B = A:B - tr(A)tr(B)/3` and
/// `tr(2 symm(grad U)) = 2 tr(grad U)`. Evaluated in the same term order as
/// [`Tensor::g_by_nut`], so the host and the device agree.
///
/// `n_cells` is taken explicitly rather than from a mesh, because every caller
/// already has both buffers and nothing else about the mesh is needed.
pub fn turbulence_production(
    gpu: &Gpu,
    k: &FvKernels,
    out: &mut DevBuf<Scalar>,
    nut: &DevBuf<Scalar>,
    grad_u: &DevBuf<Tensor>,
    n_cells: usize,
) -> Result<()> {
    expect_len(out, n_cells, "out")?;
    expect_len(nut, n_cells, "nut")?;
    expect_len(grad_u, n_cells, "grad_u")?;

    if n_cells == 0 {
        return Ok(());
    }
    let nl = n_cells as Label;

    let f = k.production.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(out)
            .arg(nut)
            .arg(grad_u)
            .arg(&nl)
            .launch(cfg_for(n_cells))?;
    }

    Ok(())
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::BcKind;
    use crate::mesh::{HostMesh, PatchKind};

    /// Every device test needs a card. Returning `None` makes the test pass
    /// vacuously on a machine without one, which is the convention the rest of
    /// the crate follows.
    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    // ----------------------------------------------------------------------
    //  A box of hexahedra, optionally sheared, with its geometry in closed
    //  form.
    //
    //  The topology comes from `mesh::topology::tests::box_mesh`, but the
    //  geometry is built here rather than by calling `compute_geometry`. That
    //  is deliberate: these tests are about the OPERATORS, and an exact,
    //  hand-checkable geometry means a failure can only be the operator's
    //  fault. It also keeps `Sf`, `V`, `weights` and `deltaCoeffs` exact to
    //  the last bit, which is what lets the linear-field gradient test assert
    //  1e-13 rather than 1e-9.
    //
    //  `shear = (a, b)` applies the volume-preserving affine map
    //
    //      X = x + a*z ,   Y = y + b*z ,   Z = z
    //
    //  which turns every cell into a parallelepiped and makes the mesh
    //  non-orthogonal by atan(a) on the x faces and atan(b) on the y faces.
    //  Under a constant affine map M the exact geometry is
    //
    //      C  = M.c                                  centroids are affine
    //      V  = det(M)*dx*dy*dz
    //      Sf = det(M)*M^-T.Sf_cartesian             Nanson's formula
    //
    //  and `weights`, `deltaCoeffs` and `non_orth_corr` then follow from
    //  SPEC-LIT S2.3 and S2.4 applied to those. `shear = (0,0)` reduces to the
    //  Cartesian case exactly - `w = 1/2`, `Delta = 1/dx`, `k = 0` to the last
    //  bit - so one builder serves both.
    // ----------------------------------------------------------------------
    fn boxed(n: [usize; 3], d: Vec3, all_generic: bool, shear: (Scalar, Scalar)) -> HostMesh {
        let (mut m, _pts, _faces) = crate::mesh::topology::tests::box_mesh(n, d);

        if all_generic {
            // box_mesh calls the z patches `empty`, which is right for a 2-D
            // case and wrong for a test that wants a fully 3-D operator.
            for p in m.patches.iter_mut() {
                p.kind = PatchKind::Generic;
                p.type_name = "patch".to_string();
            }
        }

        m.build_cell_face_maps();

        let (a, b) = shear;
        let (nx, ny) = (n[0], n[1]);
        let nc = m.n_cells;

        let map = |x: Vec3| Vec3::new(x.x + a * x.z, x.y + b * x.z, x.z);

        let cart_centre = |c: usize| -> Vec3 {
            let i = c % nx;
            let j = (c / nx) % ny;
            let k = c / (nx * ny);
            Vec3::new(
                (i as Scalar + 0.5) * d.x,
                (j as Scalar + 0.5) * d.y,
                (k as Scalar + 0.5) * d.z,
            )
        };

        let area = [d.y * d.z, d.z * d.x, d.x * d.y];
        let spacing = [d.x, d.y, d.z];

        // det(M) = 1 and M^-T = [[1,0,0],[0,1,0],[-a,-b,1]], so the Piola
        // transform of an axis-aligned area vector gains one component.
        let sf_of = |axis: usize, sgn: Scalar| -> Vec3 {
            let ar = area[axis];
            let v = match axis {
                0 => Vec3::new(ar, 0.0, -a * ar),
                1 => Vec3::new(0.0, ar, -b * ar),
                _ => Vec3::new(0.0, 0.0, ar),
            };
            v * sgn
        };

        // The offset from a cell centre to one of its face centres, in the
        // Cartesian frame, before the map is applied.
        let half_step = |axis: usize, sgn: Scalar| -> Vec3 {
            let h = sgn * 0.5 * spacing[axis];
            match axis {
                0 => Vec3::new(h, 0.0, 0.0),
                1 => Vec3::new(0.0, h, 0.0),
                _ => Vec3::new(0.0, 0.0, h),
            }
        };

        m.v = vec![d.x * d.y * d.z; nc];
        m.c = (0..nc).map(|c| map(cart_centre(c))).collect();

        // ---- internal faces ------------------------------------------------
        let n_if = m.n_internal_faces;
        m.sf = Vec::with_capacity(n_if);
        m.mag_sf = Vec::with_capacity(n_if);
        m.cf = Vec::with_capacity(n_if);
        m.weights = Vec::with_capacity(n_if);
        m.delta_coeffs = Vec::with_capacity(n_if);
        m.non_orth_corr = Vec::with_capacity(n_if);

        for f in 0..n_if {
            let o = m.owner[f] as usize;
            let nb = m.neighbour[f] as usize;

            // The Cartesian index stride between the two cells names the axis.
            let stride = nb - o;
            let axis = if stride == 1 {
                0
            } else if stride == nx {
                1
            } else {
                2
            };

            let sf = sf_of(axis, 1.0);
            let cf = map(cart_centre(o) + half_step(axis, 1.0));

            let (cp, cn) = (m.c[o], m.c[nb]);

            // SPEC-LIT S2.3: the weight that places the interpolated value
            // where the face plane cuts the line P-N.
            let dp = sf.dot(cf - cp).abs();
            let dn = sf.dot(cn - cf).abs();
            let w = dn / (dp + dn);

            // SPEC-LIT S2.4: over-relaxed decomposition.
            let delta = cn - cp;
            let nf = sf.normalised();
            let dc = 1.0 / nf.dot(delta).max(0.05 * delta.mag());

            m.sf.push(sf);
            m.mag_sf.push(sf.mag());
            m.cf.push(cf);
            m.weights.push(w);
            m.delta_coeffs.push(dc);
            m.non_orth_corr.push(nf - delta * dc);
        }

        // ---- boundary faces: patch order xmin xmax ymin ymax zmin zmax -----
        let n_bf = m.n_boundary_faces;
        m.b_sf = vec![Vec3::ZERO; n_bf];
        m.b_mag_sf = vec![0.0; n_bf];
        m.b_cf = vec![Vec3::ZERO; n_bf];
        m.b_delta_coeffs = vec![0.0; n_bf];
        m.b_y = vec![0.0; n_bf];
        m.b_nbr_cell = vec![-1; n_bf];
        m.b_weights = vec![1.0; n_bf];
        m.b_kind = vec![0; n_bf];
        m.b_patch = vec![0; n_bf];

        for (p, patch) in m.patches.iter().enumerate() {
            let axis = p / 2;
            let outward: Scalar = if p % 2 == 0 { -1.0 } else { 1.0 };
            let sf = sf_of(axis, outward);
            let nf = sf.normalised();

            for i in 0..patch.size {
                let bf = patch.start + i;
                let cell = m.b_face_cells[bf] as usize;

                let cf = map(cart_centre(cell) + half_step(axis, outward));
                let delta = cf - m.c[cell];

                m.b_sf[bf] = sf;
                m.b_mag_sf[bf] = sf.mag();
                m.b_cf[bf] = cf;
                m.b_delta_coeffs[bf] = 1.0 / nf.dot(delta).max(0.05 * delta.mag());
                m.b_y[bf] = nf.dot(delta);
                m.b_kind[bf] = patch.kind as Label;
                m.b_patch[bf] = p as Label;
            }
        }

        m
    }

    fn uniform_box(n: [usize; 3], d: Vec3, all_generic: bool) -> HostMesh {
        boxed(n, d, all_generic, (0.0, 0.0))
    }

    /// A mesh that does not close is not worth solving on (SPEC-LIT S10), and
    /// a hand-built one has to be checked before anything is concluded from
    /// it.
    fn assert_closes(m: &HostMesh) {
        for c in 0..m.n_cells {
            let mut s = Vec3::ZERO;
            for j in m.cf_offset[c] as usize..m.cf_offset[c + 1] as usize {
                let f = m.cf_face[j] as usize;
                s += if m.cf_own[j] != 0 { m.sf[f] } else { -m.sf[f] };
            }
            for j in m.bcf_offset[c] as usize..m.bcf_offset[c + 1] as usize {
                s += m.b_sf[m.bcf_face[j] as usize];
            }
            let scale = m.v[c].powf(2.0 / 3.0);
            assert!(
                s.mag() / scale < 1e-12,
                "cell {c} does not close: |sum Sf| = {}",
                s.mag()
            );
        }
    }

    /// `A·psi` in the LDU convention `ldu.rs` documents.
    fn amul(m: &HostMesh, diag: &[Scalar], up: &[Scalar], lo: &[Scalar], x: &[Scalar]) -> Vec<Scalar> {
        let mut y: Vec<Scalar> = (0..m.n_cells).map(|c| diag[c] * x[c]).collect();
        for f in 0..m.n_internal_faces {
            let o = m.owner[f] as usize;
            let n = m.neighbour[f] as usize;
            y[o] += up[f] * x[n];
            y[n] += lo[f] * x[o];
        }
        y
    }

    /// Conjugate gradients on the host, for the symmetric systems the
    /// laplacian produces. Only a test needs this - the real solvers live in
    /// `solver.rs` - but it lets the MMS check stand entirely on this module.
    fn cg(m: &HostMesh, diag: &[Scalar], up: &[Scalar], lo: &[Scalar], b: &[Scalar]) -> Vec<Scalar> {
        let n = m.n_cells;
        let mut x = vec![0.0 as Scalar; n];
        let mut r = b.to_vec();
        let mut z: Vec<Scalar> = (0..n).map(|c| r[c] / diag[c]).collect();
        let mut p = z.clone();
        let mut rz: Scalar = (0..n).map(|c| r[c] * z[c]).sum();

        let b_norm: Scalar = b.iter().map(|v| v * v).sum::<Scalar>().sqrt().max(1e-300);

        for _ in 0..20_000 {
            let ap = amul(m, diag, up, lo, &p);
            let pap: Scalar = (0..n).map(|c| p[c] * ap[c]).sum();
            if pap == 0.0 {
                break;
            }
            let alpha = rz / pap;
            for c in 0..n {
                x[c] += alpha * p[c];
                r[c] -= alpha * ap[c];
            }
            let rn: Scalar = r.iter().map(|v| v * v).sum::<Scalar>().sqrt();
            if rn / b_norm < 1e-13 {
                break;
            }
            for c in 0..n {
                z[c] = r[c] / diag[c];
            }
            let rz_new: Scalar = (0..n).map(|c| r[c] * z[c]).sum();
            let beta = rz_new / rz;
            rz = rz_new;
            for c in 0..n {
                p[c] = z[c] + beta * p[c];
            }
        }

        x
    }

    /// Fold the boundary coefficients into the matrix, in the convention this
    /// module documents. `ldu_ops::add_boundary_contributions` does the same
    /// thing on the device; doing it here by hand means the MMS test verifies
    /// the convention rather than assuming it.
    fn fold_boundary(
        m: &HostMesh,
        diag: &mut [Scalar],
        source: &mut [Scalar],
        ic: &[Scalar],
        bc: &[Scalar],
    ) {
        for bf in 0..m.n_boundary_faces {
            let c = m.b_face_cells[bf] as usize;
            diag[c] += ic[bf];
            source[c] += bc[bf];
        }
    }

    fn max_abs(v: &[Scalar]) -> Scalar {
        v.iter().fold(0.0 as Scalar, |a, b| a.max(b.abs()))
    }

    // ----------------------------------------------------------------------
    //  §7  Limiters - pure host, so they are checked on every machine
    // ----------------------------------------------------------------------

    fn all_limiters() -> Vec<Limiter> {
        vec![
            Limiter::MinMod,
            Limiter::VanLeer,
            Limiter::VanAlbada,
            Limiter::Superbee,
            Limiter::Muscl,
            Limiter::Sweby(1.0),
            Limiter::Sweby(1.5),
            Limiter::Sweby(2.0),
        ]
    }

    /// SPEC-LIT §7: "All satisfy `Psi(r) = 0` for `r <= 0` and `Psi(1) = 1`,
    /// which is what makes the scheme TVD and second-order on smooth data
    /// respectively."
    #[test]
    fn every_limiter_vanishes_for_non_positive_r() {
        for l in all_limiters() {
            for r in [-1e9, -100.0, -2.0, -1.0, -0.5, -1e-12, 0.0] {
                assert_eq!(l.psi(r), 0.0, "{l:?} at r = {r}");
            }
        }
    }

    #[test]
    fn every_limiter_is_second_order_at_r_equals_one() {
        for l in all_limiters() {
            assert!(
                (l.psi(1.0) - 1.0).abs() < 1e-15,
                "{l:?} gives Psi(1) = {}",
                l.psi(1.0)
            );
        }
    }

    /// Sweby's TVD region: `0 <= Psi(r) <= min(2r, 2)`.
    #[test]
    fn every_limiter_lies_in_the_tvd_region() {
        for l in all_limiters() {
            // A fine sweep plus the corners where the piecewise definitions
            // change, which a uniform sweep can step straight over.
            let mut rs: Vec<Scalar> = (0..4001).map(|i| i as Scalar * 0.0025).collect();
            rs.extend([0.5, 1.0, 2.0, 1e3, 1e6, 1e12, 1e300]);

            for r in rs {
                let p = l.psi(r);
                assert!(p.is_finite(), "{l:?} at r = {r} gave {p}");
                assert!(p >= 0.0, "{l:?} at r = {r} gave {p} < 0");
                let ub = (2.0 * r).min(2.0);
                assert!(
                    p <= ub + 1e-12,
                    "{l:?} at r = {r} gave {p}, above the TVD ceiling {ub}"
                );
            }
        }
    }

    /// Sweby-φ sweeps the region: `β = 1` is minmod and `β = 2` is Superbee.
    #[test]
    fn sweby_spans_minmod_to_superbee() {
        for i in 0..200 {
            let r = i as Scalar * 0.05;
            assert!((Limiter::Sweby(1.0).psi(r) - Limiter::MinMod.psi(r)).abs() < 1e-15);
            assert!((Limiter::Sweby(2.0).psi(r) - Limiter::Superbee.psi(r)).abs() < 1e-15);
        }
    }

    /// β outside `[1, 2]` leaves the TVD region, so it is clamped rather than
    /// trusted.
    #[test]
    fn sweby_beta_is_clamped_to_the_tvd_range() {
        for r in [0.25, 0.5, 1.0, 3.0] {
            assert_eq!(Limiter::Sweby(0.1).psi(r), Limiter::Sweby(1.0).psi(r));
            assert_eq!(Limiter::Sweby(9.0).psi(r), Limiter::Sweby(2.0).psi(r));
        }
    }

    /// `crate::io::case::DivScheme` IS this enum - the two used to be separate
    /// types joined by a lossy `From`, and the lossy half is where five of the
    /// six limiters of SPEC-LIT S7 went missing. Nothing may reintroduce a
    /// conversion between them.
    #[test]
    fn an_fvschemes_entry_is_an_operator_scheme() {
        use crate::io::case::DivScheme as Cs;

        assert_eq!(DivScheme::from(Cs::Central), DivScheme::Central);
        assert_eq!(DivScheme::from(Cs::Upwind), DivScheme::Upwind);
        assert_eq!(DivScheme::from(Cs::LinearUpwind), DivScheme::LinearUpwind);
        assert_eq!(
            DivScheme::from(Cs::Limited(Limiter::VanLeer)),
            DivScheme::Limited(Limiter::VanLeer)
        );

        // `limitedLinear 1` is Sweby-phi with beta = 1, which is minmod - the
        // most strongly bounded member of the S7 family.
        for i in 0..40 {
            let r = i as Scalar * 0.1;
            assert_eq!(Limiter::Sweby(1.0).psi(r), Limiter::MinMod.psi(r));
        }
    }

    /// Every scheme's implicit base and explicit correction, from SPEC-LIT
    /// S11.1: getting these two wrong is exactly how `linearUpwind` came to
    /// assemble as plain upwind.
    #[test]
    fn each_scheme_declares_the_right_implicit_base_and_correction() {
        use code::*;

        // linearUpwind and cubic differ from their bases ONLY by a correction.
        assert_eq!(DivScheme::LinearUpwind.code(), UPWIND);
        assert_eq!(DivScheme::Cubic.code(), CENTRAL);
        assert_eq!(
            DivScheme::LinearUpwind.correction(),
            DivCorrection::LinearUpwind(1.0)
        );
        assert_eq!(DivScheme::Cubic.correction(), DivCorrection::Cubic);

        // ... and the wholly implicit ones have none at all.
        for s in [
            DivScheme::Central,
            DivScheme::Upwind,
            DivScheme::Quick,
            DivScheme::QuickUnlimited,
            DivScheme::Gamma(0.2),
            DivScheme::Blended(0.5),
            DivScheme::Limited(Limiter::VanLeer),
        ] {
            assert_eq!(s.correction(), DivCorrection::None, "{s:?}");
        }

        // The S11.5 blend of central with SECOND-ORDER upwind carries
        // (1 - gamma) of the linearUpwind correction, so gamma = 1 is pure
        // central and gamma = 0 is pure linearUpwind.
        assert_eq!(
            DivScheme::LinearUpwindBlended(0.0).correction(),
            DivCorrection::LinearUpwind(1.0)
        );
        assert_eq!(
            DivScheme::LinearUpwindBlended(1.0).correction(),
            DivCorrection::LinearUpwind(0.0)
        );

        // Only an r-dependent WEIGHT forces the limited kernel. Routing
        // linearUpwind or cubic through it would evaluate a Psi that does not
        // exist for their code and silently give central weights.
        assert!(!DivScheme::LinearUpwind.weight_needs_gradient());
        assert!(!DivScheme::Cubic.weight_needs_gradient());
        assert!(!DivScheme::Blended(0.5).weight_needs_gradient());
        assert!(DivScheme::Quick.weight_needs_gradient());
        assert!(DivScheme::Gamma(0.2).weight_needs_gradient());
        assert!(DivScheme::Limited(Limiter::MinMod).weight_needs_gradient());

        // But all of them still need a gradient from SOMEWHERE.
        assert!(DivScheme::LinearUpwind.needs_gradient());
        assert!(DivScheme::Cubic.needs_gradient());
        assert!(!DivScheme::Upwind.needs_gradient());
        assert!(!DivScheme::Central.needs_gradient());
        assert!(!DivScheme::Blended(0.5).needs_gradient());
    }

    /// SPEC-LIT S12.3: `alpha = 0` is `uncorrected`, `corrected` is unlimited.
    #[test]
    fn the_sn_grad_scheme_maps_onto_its_alpha() {
        assert_eq!(SnGradScheme::Uncorrected.alpha(), 0.0);
        assert_eq!(SnGradScheme::Limited(0.0).alpha(), 0.0);
        assert!(SnGradScheme::Corrected.alpha() < 0.0);
        assert_eq!(SnGradScheme::Limited(0.75).alpha(), 0.75);
        assert!(!SnGradScheme::Uncorrected.applies());
        assert!(SnGradScheme::Corrected.applies());
        assert!(SnGradScheme::Limited(0.0).applies());
    }

    #[test]
    fn a_limited_scheme_says_it_needs_a_gradient() {
        assert!(!DivScheme::Central.needs_gradient());
        assert!(!DivScheme::Upwind.needs_gradient());
        // `linearUpwind` needs one too, for its CORRECTION rather than for
        // its weight - which is exactly what this used to assert the opposite
        // of, and exactly why the scheme assembled as plain upwind.
        assert!(DivScheme::LinearUpwind.needs_gradient());
        for l in all_limiters() {
            assert!(DivScheme::Limited(l).needs_gradient());
        }
    }

    // ----------------------------------------------------------------------
    //  Device fixtures
    // ----------------------------------------------------------------------

    struct Fixture {
        gpu: Gpu,
        k: FvKernels,
        hm: HostMesh,
        m: GpuMesh,
    }

    fn fixture(n: [usize; 3], d: Vec3, all_generic: bool) -> Option<Fixture> {
        let hm = uniform_box(n, d, all_generic);
        let gpu = gpu()?;
        let k = FvKernels::new(&gpu).ok()?;
        let m = GpuMesh::upload(&gpu, &hm).ok()?;
        Some(Fixture { gpu, k, hm, m })
    }

    /// A scalar field with a prescribed internal field and exact Dirichlet
    /// faces, so a test of an operator is not also a test of a boundary
    /// condition.
    fn dirichlet_field(
        gpu: &Gpu,
        m: &GpuMesh,
        hm: &HostMesh,
        f: &[Scalar],
        bf: &[Scalar],
    ) -> Result<GpuScalarField> {
        let mut psi = GpuScalarField::zeros(gpu, m, "psi")?;
        gpu.write(&mut psi.f, f)?;
        gpu.write(&mut psi.bf, bf)?;
        gpu.write(&mut psi.fr, &vec![1.0 as Scalar; hm.n_boundary_faces])?;
        gpu.write(&mut psi.ref_value, bf)?;
        gpu.write(&mut psi.ref_grad, &vec![0.0 as Scalar; hm.n_boundary_faces])?;
        gpu.write(
            &mut psi.bc_kind,
            &vec![BcKind::FixedValue as Label; hm.n_boundary_faces],
        )?;
        Ok(psi)
    }

    // ----------------------------------------------------------------------
    //  §3.5  Explicit operators
    // ----------------------------------------------------------------------

    /// SPEC-LIT §10: "Gauss gradient of a linear field -> exact."
    #[test]
    fn gradient_of_a_linear_field_is_exact() -> Result<()> {
        let Some(fx) = fixture([5, 4, 3], Vec3::new(0.3, 0.7, 0.2), true) else {
            return Ok(());
        };
        let (gpu, k, hm, m) = (&fx.gpu, &fx.k, &fx.hm, &fx.m);

        let a = Vec3::new(1.7, -0.9, 0.35);
        let b0: Scalar = 0.42;

        let f: Vec<Scalar> = (0..hm.n_cells).map(|c| a.dot(hm.c[c]) + b0).collect();
        let bf: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|i| a.dot(hm.b_cf[i]) + b0)
            .collect();

        let psi = dirichlet_field(gpu, m, hm, &f, &bf)?;

        let mut g: DevBuf<Vec3> = gpu.zeros(hm.n_cells)?;
        fvc_grad_scalar(gpu, k, &mut g, &psi, m)?;
        gpu.sync()?;

        let got = gpu.download(&g)?;
        let worst = got.iter().fold(0.0 as Scalar, |w, v| w.max((*v - a).mag()));

        assert!(
            worst / a.mag() < 1e-13,
            "grad(linear) is off by {worst} (relative {})",
            worst / a.mag()
        );
        Ok(())
    }

    /// The same for the tensor gradient, whose index convention is the thing
    /// most easily got backwards: `(i,j)` must be `dU_j/dx_i`.
    #[test]
    fn vector_gradient_of_a_linear_field_is_exact_and_correctly_indexed() -> Result<()> {
        let Some(fx) = fixture([4, 4, 3], Vec3::new(0.25, 0.4, 0.6), true) else {
            return Ok(());
        };
        let (gpu, k, hm, m) = (&fx.gpu, &fx.k, &fx.hm, &fx.m);

        // U = G . x with G(i,j) = dU_j/dx_i
        let gt = Tensor {
            xx: 0.3, xy: -1.1, xz: 0.7,
            yx: 2.0, yy: 0.5, yz: -0.2,
            zx: -0.4, zy: 0.9, zz: -0.8,
        };
        let eval = |x: Vec3| -> Vec3 {
            Vec3::new(
                gt.xx * x.x + gt.yx * x.y + gt.zx * x.z,
                gt.xy * x.x + gt.yy * x.y + gt.zy * x.z,
                gt.xz * x.x + gt.yz * x.y + gt.zz * x.z,
            )
        };

        let uc: Vec<Vec3> = (0..hm.n_cells).map(|c| eval(hm.c[c])).collect();
        let ub: Vec<Vec3> = (0..hm.n_boundary_faces).map(|i| eval(hm.b_cf[i])).collect();

        let mut u = GpuVectorField::zeros(gpu, m, "U")?;
        gpu.write(&mut u.f, &uc)?;
        gpu.write(&mut u.bf, &ub)?;

        let mut g: DevBuf<Tensor> = gpu.zeros(hm.n_cells)?;
        fvc_grad_vector(gpu, k, &mut g, &u, m)?;
        gpu.sync()?;

        let got = gpu.download(&g)?;
        let mut worst: Scalar = 0.0;
        for t in &got {
            let d = *t - gt;
            for c in [d.xx, d.xy, d.xz, d.yx, d.yy, d.yz, d.zx, d.zy, d.zz] {
                worst = worst.max(c.abs());
            }
        }
        assert!(worst < 1e-12, "grad(U) is off by {worst}");

        // And the production term built on it, against the host expression.
        let nut_h = vec![0.013 as Scalar; hm.n_cells];
        let nut = gpu.upload(&nut_h)?;
        let mut prod: DevBuf<Scalar> = gpu.zeros(hm.n_cells)?;
        turbulence_production(gpu, k, &mut prod, &nut, &g, hm.n_cells)?;
        gpu.sync()?;

        let got_prod = gpu.download(&prod)?;
        for (c, p) in got_prod.iter().enumerate() {
            let want = nut_h[c] * got[c].g_by_nut();
            assert!(
                (p - want).abs() <= 1e-12 * want.abs().max(1.0),
                "G at cell {c}: {p} vs {want}"
            );
        }
        Ok(())
    }

    /// SPEC-LIT §10: "`div(u)` of a uniform field -> zero." A uniform velocity
    /// gives `phi_f = u·Sf`, and a closed cell sums those to zero identically.
    #[test]
    fn divergence_of_a_uniform_flux_is_zero() -> Result<()> {
        let Some(fx) = fixture([4, 5, 3], Vec3::new(0.3, 0.2, 0.5), true) else {
            return Ok(());
        };
        let (gpu, k, hm, m) = (&fx.gpu, &fx.k, &fx.hm, &fx.m);

        let uc = Vec3::new(0.83, -0.21, 0.44);

        let phi_h: Vec<Scalar> = (0..hm.n_internal_faces).map(|f| uc.dot(hm.sf[f])).collect();
        let bphi_h: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|i| uc.dot(hm.b_sf[i]))
            .collect();

        let mut phi = GpuSurfaceScalarField::zeros(gpu, m, "phi")?;
        gpu.write(&mut phi.f, &phi_h)?;
        gpu.write(&mut phi.bf, &bphi_h)?;

        let mut d: DevBuf<Scalar> = gpu.zeros(hm.n_cells)?;
        fvc_div_surface(gpu, k, &mut d, &phi, m)?;
        gpu.sync()?;

        let got = gpu.download(&d)?;
        assert!(
            max_abs(&got) / uc.mag() < 1e-12,
            "div(uniform) = {}",
            max_abs(&got)
        );

        // ... and reconstructing a velocity from that flux gives the velocity
        // back, which is the least-squares fit being an exact fit when the
        // data really do come from a single vector.
        let mut ur: DevBuf<Vec3> = gpu.zeros(hm.n_cells)?;
        fvc_reconstruct(gpu, k, &mut ur, &phi, m)?;
        gpu.sync()?;

        let got_u = gpu.download(&ur)?;
        let worst = got_u.iter().fold(0.0 as Scalar, |w, v| w.max((*v - uc).mag()));
        assert!(worst / uc.mag() < 1e-12, "reconstruct is off by {worst}");
        Ok(())
    }

    // ----------------------------------------------------------------------
    //  THE SUBTLETY: operators must not couple through the diagonal
    // ----------------------------------------------------------------------

    /// Assemble `div` and `laplacian` into one matrix in both orders, and each
    /// into its own matrix, and check that all three agree bit for bit.
    ///
    /// This is the test for the one mistake this module is structured to
    /// avoid. If either diagonal pass read `upper`/`lower` back instead of
    /// recomputing its own coefficient, the second operator applied would
    /// subtract the first operator's contribution a second time and the three
    /// answers would differ.
    #[test]
    fn operators_do_not_couple_through_the_diagonal() -> Result<()> {
        let Some(fx) = fixture([4, 3, 3], Vec3::new(0.3, 0.4, 0.25), true) else {
            return Ok(());
        };
        let (gpu, k, hm, m) = (&fx.gpu, &fx.k, &fx.hm, &fx.m);

        let mut seed = 12345u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as Scalar) / (u32::MAX as Scalar) - 0.5
        };

        let f: Vec<Scalar> = (0..hm.n_cells).map(|_| rnd()).collect();
        let bf: Vec<Scalar> = (0..hm.n_boundary_faces).map(|_| rnd()).collect();
        let psi = dirichlet_field(gpu, m, hm, &f, &bf)?;

        let phi_h: Vec<Scalar> = (0..hm.n_internal_faces).map(|_| rnd()).collect();
        let bphi_h: Vec<Scalar> = (0..hm.n_boundary_faces).map(|_| rnd()).collect();
        let mut phi = GpuSurfaceScalarField::zeros(gpu, m, "phi")?;
        gpu.write(&mut phi.f, &phi_h)?;
        gpu.write(&mut phi.bf, &bphi_h)?;

        let gamma: Vec<Scalar> = (0..hm.n_internal_faces)
            .map(|i| (0.5 + 0.3 * rnd()) * hm.mag_sf[i])
            .collect();
        let b_gamma: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|i| (0.5 + 0.3 * rnd()) * hm.b_mag_sf[i])
            .collect();
        let d_gamma = gpu.upload(&gamma)?;
        let d_b_gamma = gpu.upload(&b_gamma)?;

        let mut w: DevBuf<Scalar> = gpu.zeros(hm.n_internal_faces)?;
        let mut bw: DevBuf<Scalar> = gpu.zeros(hm.n_boundary_faces)?;
        div_scheme_weights(
            gpu,
            k,
            Some(&mut w),
            Some(&mut bw),
            DivScheme::Upwind,
            &phi,
            &psi,
            None,
            m,
        )?;

        let snapshot = |a: &GpuLduMatrix| -> Result<Vec<Vec<Scalar>>> {
            Ok(vec![
                gpu.download(&a.diag)?,
                gpu.download(&a.upper)?,
                gpu.download(&a.lower)?,
                gpu.download(&a.source)?,
                gpu.download(&a.internal_coeffs)?,
                gpu.download(&a.boundary_coeffs)?,
            ])
        };

        // div then laplacian
        let mut a1 = GpuLduMatrix::new(gpu, m)?;
        a1.zero(gpu)?;
        fvm_div_gauss(gpu, k, &mut a1, m, &phi, &w, &bw, &psi, 1.0)?;
        fvm_laplacian(gpu, k, &mut a1, m, &d_gamma, &d_b_gamma, &psi, -1.0)?;
        gpu.sync()?;
        let s1 = snapshot(&a1)?;

        // laplacian then div
        let mut a2 = GpuLduMatrix::new(gpu, m)?;
        a2.zero(gpu)?;
        fvm_laplacian(gpu, k, &mut a2, m, &d_gamma, &d_b_gamma, &psi, -1.0)?;
        fvm_div_gauss(gpu, k, &mut a2, m, &phi, &w, &bw, &psi, 1.0)?;
        gpu.sync()?;
        let s2 = snapshot(&a2)?;

        // each on its own, summed on the host
        let mut a3 = GpuLduMatrix::new(gpu, m)?;
        a3.zero(gpu)?;
        fvm_div_gauss(gpu, k, &mut a3, m, &phi, &w, &bw, &psi, 1.0)?;
        gpu.sync()?;
        let sd = snapshot(&a3)?;

        a3.zero(gpu)?;
        fvm_laplacian(gpu, k, &mut a3, m, &d_gamma, &d_b_gamma, &psi, -1.0)?;
        gpu.sync()?;
        let sl = snapshot(&a3)?;

        // Not bit equality, and the reason is worth stating: nvcc contracts
        // `coefficient*x + accumulator` into a single FMA, so `(0 + D) + L`
        // and `(0 + L) + D` round differently in the last place even though
        // real addition is commutative. That is a property of the
        // ACCUMULATION, not of what is accumulated - a genuine
        // double-subtraction of another operator's coefficients would show up
        // here at O(1) relative, four orders of magnitude above this bound.
        const TOL: Scalar = 1e-14;

        let names = [
            "diag",
            "upper",
            "lower",
            "source",
            "internal_coeffs",
            "boundary_coeffs",
        ];

        for (i, name) in names.iter().enumerate() {
            for j in 0..s1[i].len() {
                let (a, b) = (s1[i][j], s2[i][j]);
                assert!(
                    (a - b).abs() <= TOL * a.abs().max(b.abs()).max(1.0),
                    "{name}[{j}] depends on the order the operators were \
                     applied: {a} vs {b}"
                );

                let sum = sd[i][j] + sl[i][j];
                assert!(
                    (a - sum).abs() <= TOL * sum.abs().max(1.0),
                    "{name}[{j}]: combined {a} != separate sum {sum}"
                );
            }
        }

        Ok(())
    }

    /// SPEC-LIT §3.1: after the bounded correction the diagonal is exactly
    /// minus the row's off-diagonal sum, which is the discrete statement that
    /// a uniform field is convected without change even when the flux is not
    /// solenoidal.
    #[test]
    fn the_bounded_correction_makes_a_uniform_field_stationary() -> Result<()> {
        let Some(fx) = fixture([4, 3, 3], Vec3::new(0.3, 0.4, 0.25), true) else {
            return Ok(());
        };
        let (gpu, k, hm, m) = (&fx.gpu, &fx.k, &fx.hm, &fx.m);

        let mut seed = 99u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as Scalar) / (u32::MAX as Scalar) - 0.5
        };

        // A deliberately NON-solenoidal flux - the state a SIMPLE loop is in
        // part-way through an iteration.
        let phi_h: Vec<Scalar> = (0..hm.n_internal_faces).map(|_| rnd()).collect();
        let bphi_h: Vec<Scalar> = (0..hm.n_boundary_faces).map(|_| rnd()).collect();
        let mut phi = GpuSurfaceScalarField::zeros(gpu, m, "phi")?;
        gpu.write(&mut phi.f, &phi_h)?;
        gpu.write(&mut phi.bf, &bphi_h)?;

        let ones = vec![1.0 as Scalar; hm.n_cells];
        let bones = vec![1.0 as Scalar; hm.n_boundary_faces];
        let psi = dirichlet_field(gpu, m, hm, &ones, &bones)?;

        let mut w: DevBuf<Scalar> = gpu.zeros(hm.n_internal_faces)?;
        let mut bw: DevBuf<Scalar> = gpu.zeros(hm.n_boundary_faces)?;
        div_scheme_weights(
            gpu, k, Some(&mut w), Some(&mut bw), DivScheme::Upwind, &phi, &psi, None, m,
        )?;

        let mut a = GpuLduMatrix::new(gpu, m)?;
        a.zero(gpu)?;
        fvm_div_gauss(gpu, k, &mut a, m, &phi, &w, &bw, &psi, 1.0)?;
        fvm_div_bounded_correction(gpu, k, &mut a, m, &phi, 1.0)?;
        gpu.sync()?;

        let mut diag = gpu.download(&a.diag)?;
        let up = gpu.download(&a.upper)?;
        let lo = gpu.download(&a.lower)?;
        let mut src = vec![0.0 as Scalar; hm.n_cells];
        let ic = gpu.download(&a.internal_coeffs)?;
        let bc = gpu.download(&a.boundary_coeffs)?;
        fold_boundary(hm, &mut diag, &mut src, &ic, &bc);

        // A psi = b with psi == 1 everywhere, including on the Dirichlet
        // faces, must leave zero residual.
        let ax = amul(hm, &diag, &up, &lo, &ones);
        let res: Vec<Scalar> = (0..hm.n_cells).map(|c| ax[c] - src[c]).collect();

        let scale = max_abs(&phi_h).max(max_abs(&bphi_h));
        assert!(
            max_abs(&res) <= 1e-12 * scale,
            "bounded convection leaves a residual of {} on a uniform field",
            max_abs(&res)
        );
        Ok(())
    }

    // ----------------------------------------------------------------------
    //  §11, §12, §22  Scheme selection: the schemes must differ, and each must
    //  converge at the order its paper claims.
    // ----------------------------------------------------------------------

    /// A uniform flow and a smooth field, with the exact face values to
    /// compare a scheme's own against.
    ///
    /// `psi = sin(2 pi x/Lx) cos(2 pi y/Ly) + 0.3 z` - smooth, non-polynomial,
    /// and varying in every direction, so no scheme can be right by accident.
    struct Convected {
        psi: GpuScalarField,
        phi: GpuSurfaceScalarField,
        grad: DevBuf<Vec3>,
        exact_face: Vec<Scalar>,
    }

    fn convected(fx: &Fixture, l: Vec3) -> Result<Convected> {
        let (gpu, k, hm, m) = (&fx.gpu, &fx.k, &fx.hm, &fx.m);
        let tau = 2.0 * std::f64::consts::PI as Scalar;

        let exact = |p: Vec3| -> Scalar {
            (tau * p.x / l.x).sin() * (tau * p.y / l.y).cos() + 0.3 * p.z
        };

        let f: Vec<Scalar> = (0..hm.n_cells).map(|c| exact(hm.c[c])).collect();
        let bf: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|i| exact(hm.b_cf[i]))
            .collect();
        let psi = dirichlet_field(gpu, m, hm, &f, &bf)?;

        // A constant velocity, so the flux is discretely solenoidal and no
        // component of the error comes from the flux itself.
        let u = Vec3::new(1.0, 0.6, 0.3);
        let phi_h: Vec<Scalar> = (0..hm.n_internal_faces).map(|i| u.dot(hm.sf[i])).collect();
        let bphi_h: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|i| u.dot(hm.b_sf[i]))
            .collect();

        let mut phi = GpuSurfaceScalarField::zeros(gpu, m, "phi")?;
        gpu.write(&mut phi.f, &phi_h)?;
        gpu.write(&mut phi.bf, &bphi_h)?;

        let mut grad: DevBuf<Vec3> = gpu.zeros(hm.n_cells)?;
        fvc_grad_scalar(gpu, k, &mut grad, &psi, m)?;
        gpu.sync()?;

        let exact_face: Vec<Scalar> = (0..hm.n_internal_faces)
            .map(|i| exact(hm.cf[i]))
            .collect();

        Ok(Convected {
            psi,
            phi,
            grad,
            exact_face,
        })
    }

    /// The r.m.s. error of a scheme's face value against the exact value at
    /// the face centre.
    ///
    /// This measures the SCHEME. Measuring the assembled divergence instead
    /// would cap every scheme at second order whatever its face value, because
    /// `sum_f psi_f Sf` evaluates each face integral by the midpoint rule and
    /// that rule is second order on its own - so a fourth-order face value
    /// would show up as second order and the test would prove nothing about
    /// `cubic`.
    fn face_value_error(fx: &Fixture, scheme: DivScheme) -> Result<Scalar> {
        let (gpu, k, hm, m) = (&fx.gpu, &fx.k, &fx.hm, &fx.m);
        let l = Vec3::new(1.0, 0.8, 0.5);
        let cv = convected(fx, l)?;

        let mut w: DevBuf<Scalar> = gpu.zeros(hm.n_internal_faces)?;
        div_scheme_weights(
            gpu,
            k,
            Some(&mut w),
            None,
            scheme,
            &cv.phi,
            &cv.psi,
            Some(&cv.grad),
            m,
        )?;

        let mut psif: DevBuf<Scalar> = gpu.zeros(hm.n_internal_faces)?;
        div_scheme_face_value(gpu, k, &mut psif, scheme, &cv.phi, &cv.psi, &w, &cv.grad, m)?;
        gpu.sync()?;

        let got = gpu.download(&psif)?;
        let n = hm.n_internal_faces.max(1) as Scalar;
        let ss: Scalar = (0..hm.n_internal_faces)
            .map(|i| {
                let e = got[i] - cv.exact_face[i];
                e * e
            })
            .sum();
        Ok((ss / n).sqrt())
    }

    fn face_value_order(scheme: DivScheme) -> Result<Option<Scalar>> {
        let dims = |n: usize| Vec3::new(1.0 / n as Scalar, 0.8 / n as Scalar, 0.5 / n as Scalar);
        let Some(f1) = fixture([8, 8, 8], dims(8), true) else {
            return Ok(None);
        };
        let Some(f2) = fixture([16, 16, 16], dims(16), true) else {
            return Ok(None);
        };
        let e1 = face_value_error(&f1, scheme)?;
        let e2 = face_value_error(&f2, scheme)?;
        Ok(Some(observed_order(e1, e2)))
    }

    /// SPEC-LIT §22: "linearUpwind order | MMS, smooth solution, refine | 2nd
    /// order".
    ///
    /// The test that would have caught the original bug outright: with the
    /// deferred correction of §11.2 missing, `linearUpwind` IS upwind and this
    /// comes out at 1.
    #[test]
    fn linear_upwind_is_second_order() -> Result<()> {
        let Some(order) = face_value_order(DivScheme::LinearUpwind)? else {
            return Ok(());
        };
        println!("  linearUpwind face value: observed order {order:.3}");
        assert!(
            order >= 1.8,
            "linearUpwind converged at order {order:.3}; below 1.8 means the \
             explicit gradient correction of SPEC-LIT §11.2 is not being applied \
             and the scheme is plain upwind"
        );
        Ok(())
    }

    /// And upwind really is first order, so the comparison above means
    /// something.
    #[test]
    fn upwind_is_first_order() -> Result<()> {
        let Some(order) = face_value_order(DivScheme::Upwind)? else {
            return Ok(());
        };
        println!("  upwind face value: observed order {order:.3}");
        assert!(
            (0.7..1.4).contains(&order),
            "upwind converged at order {order:.3}, expected about 1"
        );
        Ok(())
    }

    /// SPEC-LIT §22: "cubic order | MMS on a uniform mesh | better than 2nd".
    #[test]
    fn cubic_is_better_than_second_order() -> Result<()> {
        let Some(order) = face_value_order(DivScheme::Cubic)? else {
            return Ok(());
        };
        println!("  cubic face value: observed order {order:.3}");
        assert!(
            order >= 2.5,
            "cubic converged at order {order:.3}; SPEC-LIT §11.4 makes it \
             fourth order on a uniform mesh and §22 asks only for better than \
             second"
        );
        Ok(())
    }

    /// QUICK fits a quadratic, so on smooth data - where the TVD clip is
    /// inactive - it is third order (Leonard 1979).
    #[test]
    fn quick_is_at_least_second_order() -> Result<()> {
        let Some(order) = face_value_order(DivScheme::Quick)? else {
            return Ok(());
        };
        println!("  QUICK face value: observed order {order:.3}");
        assert!(
            order >= 1.8,
            "QUICK converged at order {order:.3}, expected at least 2"
        );
        Ok(())
    }

    /// The blend of §11.5 has to land between its two ends, and reach each of
    /// them exactly.
    #[test]
    fn the_blend_reaches_both_of_its_ends() -> Result<()> {
        let Some(fx) = fixture([8, 8, 8], Vec3::new(0.125, 0.1, 0.0625), true) else {
            return Ok(());
        };
        let e_up = face_value_error(&fx, DivScheme::Upwind)?;
        let e_ce = face_value_error(&fx, DivScheme::Central)?;
        let e_0 = face_value_error(&fx, DivScheme::Blended(0.0))?;
        let e_1 = face_value_error(&fx, DivScheme::Blended(1.0))?;
        let e_h = face_value_error(&fx, DivScheme::Blended(0.5))?;

        assert_eq!(e_0, e_up, "blended 0 must be exactly upwind");
        assert_eq!(e_1, e_ce, "blended 1 must be exactly central");
        assert!(
            e_h < e_up && e_h > e_ce,
            "blended 0.5 error {e_h:e} is outside its two ends {e_ce:e}..{e_up:e}"
        );
        Ok(())
    }

    /// SPEC-LIT §22: "QUICK limited | a step profile | no new extremum".
    ///
    /// Every limited scheme must keep the face value between the two cell
    /// values it interpolates - that is what boundedness means face by face -
    /// and the unbounded ones must not, or the test is measuring nothing.
    #[test]
    fn a_limited_scheme_creates_no_new_extremum_at_a_step() -> Result<()> {
        let n = 24;
        let hm = uniform_box([n, 1, 1], Vec3::new(1.0 / n as Scalar, 0.1, 0.1), true);
        let Some(gpu) = gpu() else { return Ok(()) };
        let k = FvKernels::new(&gpu)?;
        let m = GpuMesh::upload(&gpu, &hm)?;

        // A step half way along x, then a gentle ramp. The step is what a
        // limiter has to survive - r goes to zero on one side of it and to
        // infinity on the other - and the ramp after it is what makes an
        // UNBOUNDED scheme misbehave: at the first ramp face the jump the face
        // spans is tiny while the upwind cell's gradient is still the step's,
        // so r is large and an unclipped Psi extrapolates past the neighbour.
        // A bare step alone would not separate the two, because with equal
        // weights every Psi in [0, 2] lands inside the two cell values.
        let step = |x: Scalar| -> Scalar {
            if x < 0.5 {
                1.0
            } else {
                0.05 * (x - 0.5)
            }
        };
        let f: Vec<Scalar> = (0..hm.n_cells).map(|c| step(hm.c[c].x)).collect();
        let bf: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|i| step(hm.b_cf[i].x))
            .collect();
        let psi = dirichlet_field(&gpu, &m, &hm, &f, &bf)?;

        let u = Vec3::new(1.0, 0.0, 0.0);
        let phi_h: Vec<Scalar> = (0..hm.n_internal_faces).map(|i| u.dot(hm.sf[i])).collect();
        let bphi_h: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|i| u.dot(hm.b_sf[i]))
            .collect();
        let mut phi = GpuSurfaceScalarField::zeros(&gpu, &m, "phi")?;
        gpu.write(&mut phi.f, &phi_h)?;
        gpu.write(&mut phi.bf, &bphi_h)?;

        let mut grad: DevBuf<Vec3> = gpu.zeros(hm.n_cells)?;
        fvc_grad_scalar(&gpu, &k, &mut grad, &psi, &m)?;

        // The largest amount by which a face value leaves the range of the two
        // cells it sits between.
        let overshoot = |scheme: DivScheme| -> Result<Scalar> {
            let mut w: DevBuf<Scalar> = gpu.zeros(hm.n_internal_faces)?;
            div_scheme_weights(
                &gpu,
                &k,
                Some(&mut w),
                None,
                scheme,
                &phi,
                &psi,
                Some(&grad),
                &m,
            )?;
            let mut psif: DevBuf<Scalar> = gpu.zeros(hm.n_internal_faces)?;
            div_scheme_face_value(&gpu, &k, &mut psif, scheme, &phi, &psi, &w, &grad, &m)?;
            gpu.sync()?;

            let got = gpu.download(&psif)?;
            let mut worst: Scalar = 0.0;
            for i in 0..hm.n_internal_faces {
                let o = hm.owner[i] as usize;
                let nb = hm.neighbour[i] as usize;
                let lo = f[o].min(f[nb]);
                let hi = f[o].max(f[nb]);
                worst = worst.max((got[i] - hi).max(lo - got[i]).max(0.0));
            }
            Ok(worst)
        };

        for scheme in [
            DivScheme::Upwind,
            DivScheme::Quick,
            DivScheme::Limited(Limiter::VanLeer),
            DivScheme::Limited(Limiter::Superbee),
            DivScheme::Limited(Limiter::MinMod),
            DivScheme::Gamma(0.2),
        ] {
            let o = overshoot(scheme)?;
            assert!(
                o <= 1e-13,
                "{scheme:?} overshoots the step by {o:e}; a bounded scheme must \
                 not create a new extremum (SPEC-LIT §22)"
            );
        }

        // The unbounded ones must actually overshoot, or the assertion above
        // is satisfied by a scheme that does nothing.
        for unbounded_scheme in [DivScheme::QuickUnlimited, DivScheme::Cubic] {
        let unbounded = overshoot(unbounded_scheme)?;
        assert!(
            unbounded > 1e-9,
            "{unbounded_scheme:?} did not overshoot at all ({unbounded:e}); \
             it is unbounded by construction, so either it is not being \
             applied or the profile no longer tests anything"
        );
        }
        Ok(())
    }

    /// The property that would have caught the original bug: two different
    /// entries must not assemble the same matrix.
    #[test]
    fn different_schemes_do_not_assemble_identically() -> Result<()> {
        let Some(fx) = fixture([8, 6, 4], Vec3::new(0.12, 0.15, 0.2), true) else {
            return Ok(());
        };
        let (gpu, k, hm, m) = (&fx.gpu, &fx.k, &fx.hm, &fx.m);
        let cv = convected(&fx, Vec3::new(1.0, 0.9, 0.8))?;

        let assemble = |scheme: DivScheme| -> Result<(Vec<Scalar>, Vec<Scalar>, Vec<Scalar>)> {
            let mut w: DevBuf<Scalar> = gpu.zeros(hm.n_internal_faces)?;
            let mut bw: DevBuf<Scalar> = gpu.zeros(hm.n_boundary_faces)?;
            div_scheme_weights(
                gpu,
                k,
                Some(&mut w),
                Some(&mut bw),
                scheme,
                &cv.phi,
                &cv.psi,
                Some(&cv.grad),
                m,
            )?;

            let mut a = GpuLduMatrix::new(gpu, m)?;
            a.zero(gpu)?;
            fvm_div_gauss(gpu, k, &mut a, m, &cv.phi, &w, &bw, &cv.psi, 1.0)?;
            fvm_div_correction(gpu, k, &mut a, m, &cv.phi, &cv.grad, scheme, 1.0)?;
            gpu.sync()?;

            Ok((
                gpu.download(&a.diag)?,
                gpu.download(&a.upper)?,
                gpu.download(&a.source)?,
            ))
        };

        let schemes = [
            DivScheme::Upwind,
            DivScheme::Central,
            DivScheme::LinearUpwind,
            DivScheme::Cubic,
            DivScheme::Quick,
            DivScheme::Gamma(0.2),
            DivScheme::Blended(0.5),
            DivScheme::Limited(Limiter::VanLeer),
            DivScheme::Limited(Limiter::Superbee),
            // Sweby's beta sweeps the width of the TVD region, so beta = 1
            // is minmod and beta = 2 IS Superbee (SPEC-LIT S7) - which is why
            // 2 is not in this list: it would have to collide with Superbee,
            // and that collision is the parameterisation working.
            DivScheme::Limited(Limiter::Sweby(1.0)),
            DivScheme::Limited(Limiter::Sweby(1.5)),
        ];

        let mut built = Vec::new();
        for s in schemes {
            built.push((s, assemble(s)?));
        }

        for i in 0..built.len() {
            for j in (i + 1)..built.len() {
                let (si, ai) = &built[i];
                let (sj, aj) = &built[j];
                assert!(
                    ai != aj,
                    "{si:?} and {sj:?} assemble bit-identical matrices; one of \
                     them is not being honoured"
                );
            }
        }
        Ok(())
    }

    /// SPEC-LIT §22: "limited snGrad | `alpha = 0` reproduces `uncorrected` |
    /// exact".
    #[test]
    fn sn_grad_limited_zero_reproduces_uncorrected_exactly() -> Result<()> {
        let hm = boxed([6, 5, 4], Vec3::new(0.2, 0.25, 0.3), true, (0.4, 0.25));
        let Some(gpu) = gpu() else { return Ok(()) };
        let k = FvKernels::new(&gpu)?;
        let m = GpuMesh::upload(&gpu, &hm)?;

        let a0 = Vec3::new(1.3, -0.7, 0.9);
        let f: Vec<Scalar> = (0..hm.n_cells).map(|c| a0.dot(hm.c[c])).collect();
        let bf: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|i| a0.dot(hm.b_cf[i]))
            .collect();
        let psi = dirichlet_field(&gpu, &m, &hm, &f, &bf)?;

        let d_gamma = gpu.upload(&hm.mag_sf)?;
        let d_b_gamma = gpu.upload(&hm.b_mag_sf)?;
        let mut grad: DevBuf<Vec3> = gpu.zeros(hm.n_cells)?;
        fvc_grad_scalar(&gpu, &k, &mut grad, &psi, &m)?;

        let source_for = |sn: SnGradScheme| -> Result<Vec<Scalar>> {
            let mut a = GpuLduMatrix::new(&gpu, &m)?;
            a.zero(&gpu)?;
            fvm_laplacian(&gpu, &k, &mut a, &m, &d_gamma, &d_b_gamma, &psi, -1.0)?;
            fvm_laplacian_non_orth_correction(
                &gpu, &k, &mut a, &m, &d_gamma, &d_b_gamma, &psi, &grad, sn, -1.0,
            )?;
            gpu.sync()?;
            gpu.download(&a.source)
        };

        let unc = source_for(SnGradScheme::Uncorrected)?;
        let lim0 = source_for(SnGradScheme::Limited(0.0))?;
        let cor = source_for(SnGradScheme::Corrected)?;

        assert_eq!(
            unc, lim0,
            "`limited 0` must reproduce `uncorrected` bit for bit (SPEC-LIT §12.3)"
        );
        assert_ne!(
            unc, cor,
            "`corrected` produced the same source as `uncorrected` on a sheared \
             mesh; the correction is not being applied at all"
        );

        // And the cap really caps: a small alpha must sit between the two.
        let lim_small = source_for(SnGradScheme::Limited(0.25))?;
        assert_ne!(lim_small, unc);
        assert_ne!(lim_small, cor);
        Ok(())
    }

    // ----------------------------------------------------------------------
    //  §12.1, §12.2  Gradient schemes
    // ----------------------------------------------------------------------

    /// Least squares is exact for a linear field on any mesh - that is the
    /// whole reason SPEC-LIT §3.5 offers it beside Green-Gauss.
    #[test]
    fn least_squares_gradient_of_a_linear_field_is_exact() -> Result<()> {
        let hm = boxed([5, 4, 3], Vec3::new(0.3, 0.7, 0.2), true, (0.5, 0.3));
        let Some(gpu) = gpu() else { return Ok(()) };
        let k = FvKernels::new(&gpu)?;
        let m = GpuMesh::upload(&gpu, &hm)?;

        let a0 = Vec3::new(1.7, -0.9, 0.35);
        let b0: Scalar = 0.42;
        let f: Vec<Scalar> = (0..hm.n_cells).map(|c| a0.dot(hm.c[c]) + b0).collect();
        let bf: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|i| a0.dot(hm.b_cf[i]) + b0)
            .collect();
        let psi = dirichlet_field(&gpu, &m, &hm, &f, &bf)?;

        let mut g: DevBuf<Vec3> = gpu.zeros(hm.n_cells)?;
        fvc_grad_scalar_scheme(
            &gpu,
            &k,
            &mut g,
            &psi,
            &m,
            GradScheme {
                base: GradBase::LeastSquares,
                limit: GradLimit::None,
            },
        )?;
        gpu.sync()?;

        let got = gpu.download(&g)?;
        for c in 0..hm.n_cells {
            assert!(
                (got[c] - a0).mag() < 1e-11,
                "cell {c}: leastSquares gave {:?}, exact {:?}",
                got[c],
                a0
            );
        }
        Ok(())
    }

    /// SPEC-LIT §22: "cell-limited gradient | a step profile | no extrapolated
    /// overshoot".
    #[test]
    fn a_cell_limited_gradient_cannot_extrapolate_a_new_extremum() -> Result<()> {
        let n = 16;
        let hm = uniform_box([n, 4, 1], Vec3::new(1.0 / n as Scalar, 0.1, 0.1), true);
        let Some(gpu) = gpu() else { return Ok(()) };
        let k = FvKernels::new(&gpu)?;
        let m = GpuMesh::upload(&gpu, &hm)?;

        let step = |x: Scalar| -> Scalar {
            if x < 0.5 {
                2.0
            } else {
                0.0
            }
        };
        let f: Vec<Scalar> = (0..hm.n_cells).map(|c| step(hm.c[c].x)).collect();
        let bf: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|i| step(hm.b_cf[i].x))
            .collect();
        let psi = dirichlet_field(&gpu, &m, &hm, &f, &bf)?;

        // psi_min/psi_max over each cell and its face neighbours, which is the
        // range SPEC-LIT §12.2 says an extrapolated face value must stay in.
        let mut lo = f.clone();
        let mut hi = f.clone();
        for i in 0..hm.n_internal_faces {
            let o = hm.owner[i] as usize;
            let nb = hm.neighbour[i] as usize;
            lo[o] = lo[o].min(f[nb]);
            hi[o] = hi[o].max(f[nb]);
            lo[nb] = lo[nb].min(f[o]);
            hi[nb] = hi[nb].max(f[o]);
        }
        for b in 0..hm.n_boundary_faces {
            if hm.patches[hm.b_patch[b] as usize].kind == PatchKind::Empty {
                continue;
            }
            let c = hm.b_face_cells[b] as usize;
            lo[c] = lo[c].min(bf[b]);
            hi[c] = hi[c].max(bf[b]);
        }

        let worst_overshoot = |scheme: GradScheme| -> Result<Scalar> {
            let mut g: DevBuf<Vec3> = gpu.zeros(hm.n_cells)?;
            fvc_grad_scalar_scheme(&gpu, &k, &mut g, &psi, &m, scheme)?;
            gpu.sync()?;
            let grad = gpu.download(&g)?;

            let mut worst: Scalar = 0.0;
            for c in 0..hm.n_cells {
                let mut check = |cf: Vec3| {
                    let v = f[c] + (cf - hm.c[c]).dot(grad[c]);
                    worst = worst.max((v - hi[c]).max(lo[c] - v).max(0.0));
                };
                for j in hm.cf_offset[c] as usize..hm.cf_offset[c + 1] as usize {
                    check(hm.cf[hm.cf_face[j] as usize]);
                }
                for j in hm.bcf_offset[c] as usize..hm.bcf_offset[c + 1] as usize {
                    let b = hm.bcf_face[j] as usize;
                    if hm.patches[hm.b_patch[b] as usize].kind == PatchKind::Empty {
                        continue;
                    }
                    check(hm.b_cf[b]);
                }
            }
            Ok(worst)
        };

        for kind in [
            GradLimiterKind::BarthJespersen,
            GradLimiterKind::Venkatakrishnan,
        ] {
            for base in [GradBase::Gauss, GradBase::LeastSquares] {
                let o = worst_overshoot(GradScheme {
                    base,
                    limit: GradLimit::Cell(kind, 1.0),
                })?;
                assert!(
                    o <= 1e-12,
                    "cellLimited<{kind:?}> on {base:?} extrapolated {o:e} outside \
                     the neighbour range (SPEC-LIT §12.2)"
                );

                let o = worst_overshoot(GradScheme {
                    base,
                    limit: GradLimit::Face(kind, 1.0),
                })?;
                assert!(
                    o <= 1e-12,
                    "faceLimited<{kind:?}> on {base:?} extrapolated {o:e} outside \
                     the neighbour range"
                );
            }
        }

        // The unlimited gradient must overshoot, or the limiter is being
        // credited with something the step never had.
        let unlimited = worst_overshoot(GradScheme::GAUSS)?;
        assert!(
            unlimited > 1e-6,
            "the unlimited gradient did not overshoot at all ({unlimited:e}); the \
             limiter test proves nothing"
        );

        // coeff 0 disables the limiter - SPEC-LIT §12.2 - so it must reproduce
        // the unlimited gradient exactly.
        let off = worst_overshoot(GradScheme {
            base: GradBase::Gauss,
            limit: GradLimit::Cell(GradLimiterKind::BarthJespersen, 0.0),
        })?;
        assert_eq!(off, unlimited, "cellLimited ... 0 must disable the limiter");
        Ok(())
    }

    /// A limited gradient must leave a linear field alone: there is no new
    /// extremum to prevent, so the limiter must be exactly 1 everywhere.
    #[test]
    fn a_cell_limited_gradient_leaves_a_linear_field_alone() -> Result<()> {
        let Some(fx) = fixture([6, 5, 4], Vec3::new(0.2, 0.25, 0.3), true) else {
            return Ok(());
        };
        let (gpu, k, hm, m) = (&fx.gpu, &fx.k, &fx.hm, &fx.m);

        let a0 = Vec3::new(1.1, -0.4, 0.8);
        let f: Vec<Scalar> = (0..hm.n_cells).map(|c| a0.dot(hm.c[c])).collect();
        let bf: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|i| a0.dot(hm.b_cf[i]))
            .collect();
        let psi = dirichlet_field(gpu, m, hm, &f, &bf)?;

        let mut g: DevBuf<Vec3> = gpu.zeros(hm.n_cells)?;
        fvc_grad_scalar_scheme(
            gpu,
            k,
            &mut g,
            &psi,
            m,
            GradScheme {
                base: GradBase::Gauss,
                limit: GradLimit::Cell(GradLimiterKind::BarthJespersen, 1.0),
            },
        )?;
        gpu.sync()?;

        let got = gpu.download(&g)?;
        for c in 0..hm.n_cells {
            assert!(
                (got[c] - a0).mag() < 1e-11,
                "cell {c}: the limiter scaled a linear field's gradient to {:?}",
                got[c]
            );
        }
        Ok(())
    }

    // ----------------------------------------------------------------------
    //  §10  Manufactured solution: second-order convergence of the laplacian
    // ----------------------------------------------------------------------

    /// `-laplacian(psi) = lambda·psi_exact` with
    /// `psi = sin(pi x/Lx) sin(pi y/Ly) sin(pi z/Lz)`, homogeneous Dirichlet on
    /// all six patches. Assemble, fold the boundary in with this module's own
    /// convention, solve with a host CG, and measure the volume-weighted L2
    /// error against the exact solution on two meshes.
    ///
    /// Because the boundary folding is done here rather than by a device
    /// helper, a sign error in `internal_coeffs`/`boundary_coeffs` shows up as
    /// a failure to converge at all rather than as agreement with an equally
    /// wrong twin.
    fn poisson_error(fx: &Fixture, n_non_orth: usize) -> Result<Scalar> {
        let (gpu, k, hm, m) = (&fx.gpu, &fx.k, &fx.hm, &fx.m);

        let (lx, ly, lz) = (1.0 as Scalar, 0.7 as Scalar, 0.4 as Scalar);
        let pi = std::f64::consts::PI as Scalar;

        let exact = |p: Vec3| -> Scalar {
            (pi * p.x / lx).sin() * (pi * p.y / ly).sin() * (pi * p.z / lz).sin()
        };
        let lam = pi * pi * (1.0 / (lx * lx) + 1.0 / (ly * ly) + 1.0 / (lz * lz));

        let ex: Vec<Scalar> = (0..hm.n_cells).map(|c| exact(hm.c[c])).collect();
        let ex_b: Vec<Scalar> = (0..hm.n_boundary_faces).map(|i| exact(hm.b_cf[i])).collect();
        let su_h: Vec<Scalar> = ex.iter().map(|v| lam * *v).collect();

        // gamma = 1, so gammaMagSf is the mesh's own magSf.
        let d_gamma = gpu.upload(&hm.mag_sf)?;
        let d_b_gamma = gpu.upload(&hm.b_mag_sf)?;
        let d_su = gpu.upload(&su_h)?;

        let zeros_c = vec![0.0 as Scalar; hm.n_cells];
        let mut psi = dirichlet_field(gpu, m, hm, &zeros_c, &ex_b)?;

        let mut grad: DevBuf<Vec3> = gpu.zeros(hm.n_cells)?;
        let mut a = GpuLduMatrix::new(gpu, m)?;
        let mut got = zeros_c.clone();

        // One pass, plus `n_non_orth` more in which the explicit correction is
        // rebuilt from the latest solution - Jasak S3.4.3. On an orthogonal
        // mesh `k = 0` and the extra passes change nothing at all.
        for pass in 0..=n_non_orth {
            a.zero(gpu)?;
            fvm_laplacian(gpu, k, &mut a, m, &d_gamma, &d_b_gamma, &psi, -1.0)?;
            if pass > 0 {
                fvm_laplacian_non_orth_correction(
                    gpu,
                    k,
                    &mut a,
                    m,
                    &d_gamma,
                    &d_b_gamma,
                    &psi,
                    &grad,
                    SnGradScheme::Corrected,
                    -1.0,
                )?;
            }
            fvm_su(gpu, k, &mut a, m, &d_su, 1.0)?;
            gpu.sync()?;

            let mut diag = gpu.download(&a.diag)?;
            let up = gpu.download(&a.upper)?;
            let lo = gpu.download(&a.lower)?;
            let mut src = gpu.download(&a.source)?;
            let ic = gpu.download(&a.internal_coeffs)?;
            let bc = gpu.download(&a.boundary_coeffs)?;
            fold_boundary(hm, &mut diag, &mut src, &ic, &bc);

            got = cg(hm, &diag, &up, &lo, &src);

            gpu.write(&mut psi.f, &got)?;
            fvc_grad_scalar(gpu, k, &mut grad, &psi, m)?;
            gpu.sync()?;
        }

        let mut l2: Scalar = 0.0;
        let mut vol: Scalar = 0.0;
        for c in 0..hm.n_cells {
            let e = got[c] - ex[c];
            l2 += e * e * hm.v[c];
            vol += hm.v[c];
        }
        Ok((l2 / vol).sqrt())
    }

    fn observed_order(e1: Scalar, e2: Scalar) -> Scalar {
        (e1 / e2).ln() / (2.0 as Scalar).ln()
    }

    #[test]
    fn laplacian_converges_at_second_order() -> Result<()> {
        let dims = |n: usize| Vec3::new(1.0 / n as Scalar, 0.7 / n as Scalar, 0.4 / n as Scalar);

        let Some(f1) = fixture([8, 8, 8], dims(8), true) else {
            return Ok(());
        };
        let Some(f2) = fixture([16, 16, 16], dims(16), true) else {
            return Ok(());
        };
        assert_closes(&f1.hm);

        let e1 = poisson_error(&f1, 0)?;
        let e2 = poisson_error(&f2, 0)?;
        let order = observed_order(e1, e2);

        println!("  orthogonal MMS: L2 {e1:.4e} -> {e2:.4e}, observed order {order:.3}");
        assert!(
            order >= 1.8,
            "observed order of accuracy {order:.3} (L2 {e1:e} -> {e2:e}); below \
             1.8 means the discretisation or the boundary treatment is wrong"
        );
        Ok(())
    }

    /// The same manufactured solution on a sheared mesh, where the correction
    /// vector `k` of SPEC-LIT S2.4 is not zero and the deferred non-orthogonal
    /// correction is the only thing keeping the scheme second order.
    ///
    /// Run twice: once with the correction switched off, to show that it is
    /// doing something, and once with it iterated.
    #[test]
    fn the_non_orthogonal_correction_restores_second_order() -> Result<()> {
        let dims = |n: usize| Vec3::new(1.0 / n as Scalar, 0.7 / n as Scalar, 0.4 / n as Scalar);
        let shear = (0.35, 0.2); // 19.3 and 11.3 degrees of non-orthogonality

        let Some(gpu0) = gpu() else { return Ok(()) };
        drop(gpu0);

        let mut errs_off = Vec::new();
        let mut errs_on = Vec::new();

        for n in [8usize, 16usize] {
            let hm = boxed([n, n, n], dims(n), true, shear);
            assert_closes(&hm);

            let gpu = match gpu() {
                Some(g) => g,
                None => return Ok(()),
            };
            let k = FvKernels::new(&gpu)?;
            let m = GpuMesh::upload(&gpu, &hm)?;
            let fx = Fixture { gpu, k, hm, m };

            errs_off.push(poisson_error(&fx, 0)?);
            errs_on.push(poisson_error(&fx, 12)?);
        }

        let order_off = observed_order(errs_off[0], errs_off[1]);
        let order_on = observed_order(errs_on[0], errs_on[1]);

        println!(
            "  sheared MMS: uncorrected L2 {:.4e} -> {:.4e} (order {order_off:.3}); \
             corrected L2 {:.4e} -> {:.4e} (order {order_on:.3})",
            errs_off[0], errs_off[1], errs_on[0], errs_on[1]
        );

        assert!(
            order_on >= 1.8,
            "with the non-orthogonal correction the observed order is only \
             {order_on:.3}"
        );
        assert!(
            errs_on[1] < errs_off[1],
            "the non-orthogonal correction made the finer mesh worse: \
             {:.4e} vs {:.4e}",
            errs_on[1],
            errs_off[1]
        );
        Ok(())
    }

    /// The same MMS again, but on a mesh `poisson_error`'s hand-rolled `boxed`
    /// geometry cannot build at all: a SHEARED mesh with a CYCLIC pair of
    /// patches, i.e. a periodic channel. This is what exercises the fix to
    /// `fvLapNonOrth`/`fvSnGradCorrBoundary`'s cyclic branches - a mesh needs
    /// both a real coupled boundary (`mesh::geometry::compute`, not `boxed`'s
    /// exact-but-uncoupled-only geometry) and non-orthogonality on exactly
    /// that boundary for the correction this module ships to have anything to
    /// do.
    ///
    /// `psi(x, y) = sin(2*pi*x/Lx)*sin(pi*y/Ly)` is exactly periodic in `x` -
    /// value AND normal derivative match across the `x = 0`/`x = Lx` couple -
    /// and exactly zero on the `y` walls, so both boundaries get their exact
    /// condition with no approximation of their own. `-laplacian(psi) =
    /// lambda*psi` with `lambda = (2*pi/Lx)^2 + (pi/Ly)^2`, the same
    /// `fvm_su`-as-manufactured-source pattern as `poisson_error`.
    ///
    /// Solved with the crate's own device solver and
    /// `ldu_ops::add_boundary_contributions` - which is what actually turns a
    /// cyclic face's `internalCoeffs`/`boundaryCoeffs` into a coupled matrix
    /// entry - rather than `poisson_error`'s host `cg`/`fold_boundary`, which
    /// know only the uncoupled boundary fold.
    fn periodic_channel(n: usize, lx: Scalar, ly: Scalar, shear: Scalar) -> Result<HostMesh> {
        let d = Vec3::new(lx / n as Scalar, ly / n as Scalar, 1.0);
        let (mut m, mut points, faces) = crate::mesh::topology::tests::box_mesh([n, n, 1], d);

        // x-min / x-max, Generic by box_mesh's default, become one cyclic
        // pair - a periodic streamwise direction, as SPEC-LIT S2.4's cyclic
        // extension is meant for.
        m.patches[0].kind = PatchKind::Cyclic;
        m.patches[0].type_name = "cyclic".to_string();
        m.patches[0].nbr_patch = Some(1);
        m.patches[1].kind = PatchKind::Cyclic;
        m.patches[1].type_name = "cyclic".to_string();
        m.patches[1].nbr_patch = Some(0);

        // x += s*y: planar, volume-preserving, and (SPEC-LIT S2.4) it tilts
        // every x-normal face - the internal ones AND the cyclic couple -
        // while leaving `d` axis-aligned, exactly as
        // `mesh::geometry::tests::a_sheared_cyclic_couple_...` establishes.
        for q in points.iter_mut() {
            q.x += shear * q.y;
        }

        m.build_cell_face_maps();
        m.compute_geometry(&points, &faces)?;
        Ok(m)
    }

    fn periodic_channel_error(
        hm: &HostMesh,
        lx: Scalar,
        ly: Scalar,
        n_non_orth: usize,
    ) -> Result<Scalar> {
        use crate::ldu_ops::{self, LduKernels};
        use crate::solver::{self, SolverControls, SolverKernels, SolverWorkspace};

        let Some(gpu) = gpu() else {
            return Ok(0.0);
        };
        let k = FvKernels::new(&gpu)?;
        let lduk = LduKernels::new(&gpu)?;
        let solk = SolverKernels::new(&gpu)?;
        let m = GpuMesh::upload(&gpu, hm)?;

        let pi = std::f64::consts::PI as Scalar;
        let (kx, ky) = (2.0 * pi / lx, pi / ly);
        let lambda = kx * kx + ky * ky;
        let exact = |p: Vec3| (kx * p.x).sin() * (ky * p.y).sin();

        let ex: Vec<Scalar> = (0..hm.n_cells).map(|c| exact(hm.c[c])).collect();
        // The y walls' exact value; the x (cyclic) patches ignore `bf`/`fr`
        // entirely; `dirichlet_field` writing them anyway is harmless.
        let ex_b: Vec<Scalar> = (0..hm.n_boundary_faces).map(|i| exact(hm.b_cf[i])).collect();
        let su_h: Vec<Scalar> = ex.iter().map(|v| lambda * *v).collect();

        let d_gamma = gpu.upload(&hm.mag_sf)?;
        let d_b_gamma = gpu.upload(&hm.b_mag_sf)?;
        let d_su = gpu.upload(&su_h)?;

        let zeros_c = vec![0.0 as Scalar; hm.n_cells];
        let mut psi = dirichlet_field(&gpu, &m, hm, &zeros_c, &ex_b)?;

        let mut grad: DevBuf<Vec3> = gpu.zeros(hm.n_cells.max(1))?;
        let mut a = GpuLduMatrix::new(&gpu, &m)?;
        let mut ws = SolverWorkspace::for_mesh(&gpu, &m)?;
        let ctrl = SolverControls {
            tolerance: 1e-14,
            rel_tol: 0.0,
            max_iter: 8000,
            ..SolverControls::default()
        };

        for pass in 0..=n_non_orth {
            a.zero(&gpu)?;
            fvm_laplacian(&gpu, &k, &mut a, &m, &d_gamma, &d_b_gamma, &psi, -1.0)?;
            if pass > 0 {
                fvm_laplacian_non_orth_correction(
                    &gpu,
                    &k,
                    &mut a,
                    &m,
                    &d_gamma,
                    &d_b_gamma,
                    &psi,
                    &grad,
                    SnGradScheme::Corrected,
                    -1.0,
                )?;
            }
            fvm_su(&gpu, &k, &mut a, &m, &d_su, 1.0)?;
            ldu_ops::add_boundary_contributions(&gpu, &lduk, &mut a, &m)?;
            gpu.sync()?;

            let perf =
                solver::solve_pbicgstab(&gpu, &solk, &mut psi.f, &a, &m, &mut ws, &ctrl)?;
            assert!(
                perf.converged,
                "periodic-channel solve stagnated at {:e}",
                perf.final_residual
            );

            fvc_grad_scalar(&gpu, &k, &mut grad, &psi, &m)?;
            gpu.sync()?;
        }

        let got = gpu.download(&psi.f)?;
        let mut l2: Scalar = 0.0;
        let mut vol: Scalar = 0.0;
        for c in 0..hm.n_cells {
            let e = got[c] - ex[c];
            l2 += e * e * hm.v[c];
            vol += hm.v[c];
        }
        Ok((l2 / vol).sqrt())
    }

    #[test]
    fn the_cyclic_non_orthogonal_correction_improves_a_sheared_periodic_channel() -> Result<()> {
        let Some(gpu0) = gpu() else { return Ok(()) };
        drop(gpu0);

        let (lx, ly) = (1.0 as Scalar, 0.7 as Scalar);
        let shear: Scalar = 0.3; // atan(0.3) = 16.7 degrees at the cyclic couple

        let mut errs_off = Vec::new();
        let mut errs_on = Vec::new();

        for n in [8usize, 16usize] {
            let hm = periodic_channel(n, lx, ly, shear)?;
            assert_closes(&hm);

            errs_off.push(periodic_channel_error(&hm, lx, ly, 0)?);
            errs_on.push(periodic_channel_error(&hm, lx, ly, 12)?);
        }

        let order_off = observed_order(errs_off[0], errs_off[1]);
        let order_on = observed_order(errs_on[0], errs_on[1]);

        println!(
            "  sheared periodic channel: uncorrected L2 {:.4e} -> {:.4e} (order \
             {order_off:.3}); corrected L2 {:.4e} -> {:.4e} (order {order_on:.3})",
            errs_off[0], errs_off[1], errs_on[0], errs_on[1]
        );

        assert!(
            order_on >= 1.8,
            "with the cyclic non-orthogonal correction the observed order on \
             the periodic channel is only {order_on:.3} (L2 {:.4e} -> {:.4e})",
            errs_on[0],
            errs_on[1]
        );
        assert!(
            errs_on[1] < errs_off[1],
            "the cyclic non-orthogonal correction made the finer periodic \
             mesh worse: {:.4e} vs {:.4e}",
            errs_on[1],
            errs_off[1]
        );
        Ok(())
    }

    /// The correction kernel's arithmetic, term by term, against the same two
    /// sums written out on the host - including the boundary half and its `fr`
    /// factor, which is the part that is easy to get wrong and impossible to
    /// see in a converged answer.
    #[test]
    fn the_non_orthogonal_correction_matches_the_written_sum() -> Result<()> {
        let hm = boxed([4, 3, 3], Vec3::new(0.3, 0.4, 0.25), true, (0.35, 0.2));
        assert_closes(&hm);

        // A sheared mesh really does have a non-zero correction vector, so
        // this is not a test of zero == zero.
        assert!(
            hm.non_orth_corr.iter().any(|k| k.mag() > 0.1),
            "the sheared mesh came out orthogonal"
        );

        let Some(gpu) = gpu() else { return Ok(()) };
        let k = FvKernels::new(&gpu)?;
        let m = GpuMesh::upload(&gpu, &hm)?;

        // A spread of valueFractions, so the fr weighting of the boundary term
        // is exercised rather than sitting at 1.
        let fr_h: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|i| ((i % 5) as Scalar) * 0.25)
            .collect();
        let mut psi = GpuScalarField::zeros(&gpu, &m, "psi")?;
        gpu.write(&mut psi.fr, &fr_h)?;

        let grad_h: Vec<Vec3> = (0..hm.n_cells)
            .map(|c| {
                let p = hm.c[c];
                Vec3::new(p.x - 0.3, 2.0 * p.y, 1.0 - p.z)
            })
            .collect();
        let grad = gpu.upload(&grad_h)?;

        let gamma_h: Vec<Scalar> = (0..hm.n_internal_faces)
            .map(|f| (0.5 + 0.25 * (f as Scalar).sin()) * hm.mag_sf[f])
            .collect();
        let b_gamma_h: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|i| (0.5 + 0.25 * (i as Scalar).cos()) * hm.b_mag_sf[i])
            .collect();
        let gamma = gpu.upload(&gamma_h)?;
        let b_gamma = gpu.upload(&b_gamma_h)?;

        let sign: Scalar = -1.0;

        let mut a = GpuLduMatrix::new(&gpu, &m)?;
        a.zero(&gpu)?;
        fvm_laplacian_non_orth_correction(
            &gpu,
            &k,
            &mut a,
            &m,
            &gamma,
            &b_gamma,
            &psi,
            &grad,
            SnGradScheme::Corrected,
            sign,
        )?;
        gpu.sync()?;

        let got = gpu.download(&a.source)?;

        for c in 0..hm.n_cells {
            let mut acc: Scalar = 0.0;

            for j in hm.cf_offset[c] as usize..hm.cf_offset[c + 1] as usize {
                let f = hm.cf_face[j] as usize;
                let o = hm.owner[f] as usize;
                let nb = hm.neighbour[f] as usize;
                let w = hm.weights[f];
                let gf = grad_h[o] * w + grad_h[nb] * (1.0 - w);
                let t = gamma_h[f] * hm.non_orth_corr[f].dot(gf);
                acc += if hm.cf_own[j] != 0 { t } else { -t };
            }

            for j in hm.bcf_offset[c] as usize..hm.bcf_offset[c + 1] as usize {
                let b = hm.bcf_face[j] as usize;
                // k_b = nf - d_b * Delta_b, rebuilt exactly as the kernel does
                let nf = hm.b_sf[b].normalised();
                let db = hm.b_cf[b] - hm.c[c];
                let kb = nf - db * hm.b_delta_coeffs[b];
                acc += fr_h[b] * b_gamma_h[b] * kb.dot(grad_h[c]);
            }

            let want = -sign * acc;
            assert!(
                (got[c] - want).abs() <= 1e-12 * want.abs().max(1.0),
                "cell {c}: {} vs {want}",
                got[c]
            );
        }
        Ok(())
    }

    /// The flux and the matrix must be the same operator.
    ///
    /// `sn_grad_flux` + `sn_grad_flux_correction` computes
    /// `phi_f = gamma_f |Sf| snGrad(psi)_f`, and the matrix computes the same
    /// thing implicitly. Summing the fluxes over a cell must therefore give
    /// exactly what the matrix row gives:
    ///
    /// ```text
    /// Σ_f (±phi_f)  ==  (A·psi)_P - source_P        for sign = +1
    /// ```
    ///
    /// This is the property SIMPLE leans on when it corrects `phi` with the
    /// pressure it just solved for: get it wrong and continuity is violated by
    /// whatever the discrepancy is, forever. It is checked on a SHEARED mesh
    /// so that both halves of the non-orthogonal correction are in play.
    #[test]
    fn the_flux_and_the_matrix_are_the_same_operator() -> Result<()> {
        let hm = boxed([4, 4, 3], Vec3::new(0.3, 0.4, 0.25), true, (0.35, 0.2));
        assert_closes(&hm);

        let Some(gpu) = gpu() else { return Ok(()) };
        let k = FvKernels::new(&gpu)?;
        let m = GpuMesh::upload(&gpu, &hm)?;

        let f_h: Vec<Scalar> = (0..hm.n_cells)
            .map(|c| {
                let p = hm.c[c];
                (2.0 * p.x).sin() + 0.5 * p.y * p.y - p.z
            })
            .collect();
        let bf_h: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|i| {
                let p = hm.b_cf[i];
                (2.0 * p.x).sin() + 0.5 * p.y * p.y - p.z
            })
            .collect();

        // A mix of Dirichlet and Neumann faces, so both halves of the mixed
        // form contribute.
        let mut psi = GpuScalarField::zeros(&gpu, &m, "psi")?;
        let fr_h: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|i| if i % 3 == 0 { 0.0 } else { 1.0 })
            .collect();
        let rg_h: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|i| 0.3 * ((i % 7) as Scalar - 3.0))
            .collect();
        gpu.write(&mut psi.f, &f_h)?;
        gpu.write(&mut psi.bf, &bf_h)?;
        gpu.write(&mut psi.fr, &fr_h)?;
        gpu.write(&mut psi.ref_value, &bf_h)?;
        gpu.write(&mut psi.ref_grad, &rg_h)?;

        let gamma_h: Vec<Scalar> = (0..hm.n_internal_faces)
            .map(|f| (0.7 + 0.2 * (f as Scalar).sin()) * hm.mag_sf[f])
            .collect();
        let b_gamma_h: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|i| (0.7 + 0.2 * (i as Scalar).cos()) * hm.b_mag_sf[i])
            .collect();
        let gamma = gpu.upload(&gamma_h)?;
        let b_gamma = gpu.upload(&b_gamma_h)?;

        let mut grad: DevBuf<Vec3> = gpu.zeros(hm.n_cells)?;
        fvc_grad_scalar(&gpu, &k, &mut grad, &psi, &m)?;

        let mut a = GpuLduMatrix::new(&gpu, &m)?;
        a.zero(&gpu)?;
        fvm_laplacian(&gpu, &k, &mut a, &m, &gamma, &b_gamma, &psi, 1.0)?;
        fvm_laplacian_non_orth_correction(
            &gpu,
            &k,
            &mut a,
            &m,
            &gamma,
            &b_gamma,
            &psi,
            &grad,
            SnGradScheme::Corrected,
            1.0,
        )?;

        let mut phi = GpuSurfaceScalarField::zeros(&gpu, &m, "phi")?;
        sn_grad_flux(&gpu, &k, &mut phi, &psi, &gamma, &b_gamma, &m)?;
        sn_grad_flux_correction(
            &gpu,
            &k,
            &mut phi,
            &psi,
            &gamma,
            &b_gamma,
            &grad,
            SnGradScheme::Corrected,
            &m,
        )?;
        gpu.sync()?;

        let mut diag = gpu.download(&a.diag)?;
        let up = gpu.download(&a.upper)?;
        let lo = gpu.download(&a.lower)?;
        let mut src = gpu.download(&a.source)?;
        let ic = gpu.download(&a.internal_coeffs)?;
        let bc = gpu.download(&a.boundary_coeffs)?;
        fold_boundary(&hm, &mut diag, &mut src, &ic, &bc);

        let ax = amul(&hm, &diag, &up, &lo, &f_h);

        let iphi = gpu.download(&phi.f)?;
        let bphi = gpu.download(&phi.bf)?;

        let mut worst: Scalar = 0.0;
        let mut scale: Scalar = 1.0;
        for c in 0..hm.n_cells {
            let mut sum: Scalar = 0.0;
            for j in hm.cf_offset[c] as usize..hm.cf_offset[c + 1] as usize {
                let f = hm.cf_face[j] as usize;
                sum += if hm.cf_own[j] != 0 { iphi[f] } else { -iphi[f] };
            }
            for j in hm.bcf_offset[c] as usize..hm.bcf_offset[c + 1] as usize {
                sum += bphi[hm.bcf_face[j] as usize];
            }
            let from_matrix = ax[c] - src[c];
            worst = worst.max((sum - from_matrix).abs());
            scale = scale.max(from_matrix.abs());
        }

        assert!(
            worst <= 1e-11 * scale,
            "the flux and the matrix disagree by {worst} (scale {scale})"
        );
        Ok(())
    }

    // ----------------------------------------------------------------------
    //  §7  the device limiter against the host mirror
    // ----------------------------------------------------------------------

    /// Run every limited scheme through `div_scheme_weights` on a real mesh
    /// and recompute the same weights on the host from
    /// `w = w_up + Psi(r)(w_c - w_up)`.
    ///
    /// This is what stops `Limiter::psi` and `limiterPsi` in `cuda/fv.cu` from
    /// drifting apart: the host mirror is used by the pure-host TVD tests
    /// above, and this ties the device to it.
    #[test]
    fn device_limiter_agrees_with_the_host() -> Result<()> {
        let Some(fx) = fixture([5, 4, 4], Vec3::new(0.3, 0.4, 0.25), true) else {
            return Ok(());
        };
        let (gpu, k, hm, m) = (&fx.gpu, &fx.k, &fx.hm, &fx.m);

        // A field with real structure - a smooth part and a step - so that r
        // lands on both sides of every kink in every limiter.
        let f: Vec<Scalar> = (0..hm.n_cells)
            .map(|c| {
                let p = hm.c[c];
                let smooth = (3.0 * p.x).sin() * (2.0 * p.y).cos();
                let step: Scalar = if p.x > 0.5 { 1.0 } else { 0.0 };
                smooth + step
            })
            .collect();
        let bf: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|i| {
                let p = hm.b_cf[i];
                (3.0 * p.x).sin() * (2.0 * p.y).cos() + if p.x > 0.5 { 1.0 } else { 0.0 }
            })
            .collect();

        let psi = dirichlet_field(gpu, m, hm, &f, &bf)?;

        // A flux that changes sign, so both upwind directions are exercised.
        let uc = Vec3::new(0.8, -0.3, 0.2);
        let phi_h: Vec<Scalar> = (0..hm.n_internal_faces)
            .map(|i| {
                let s = if hm.cf[i].y > 0.5 { -1.0 } else { 1.0 };
                s * uc.dot(hm.sf[i])
            })
            .collect();
        let mut phi = GpuSurfaceScalarField::zeros(gpu, m, "phi")?;
        gpu.write(&mut phi.f, &phi_h)?;

        let mut grad: DevBuf<Vec3> = gpu.zeros(hm.n_cells)?;
        fvc_grad_scalar(gpu, k, &mut grad, &psi, m)?;
        gpu.sync()?;
        let grad_h = gpu.download(&grad)?;

        for l in all_limiters() {
            let mut w: DevBuf<Scalar> = gpu.zeros(hm.n_internal_faces)?;
            div_scheme_weights(
                gpu,
                k,
                Some(&mut w),
                None,
                DivScheme::Limited(l),
                &phi,
                &psi,
                Some(&grad),
                m,
            )?;
            gpu.sync()?;
            let got = gpu.download(&w)?;

            for i in 0..hm.n_internal_faces {
                let o = hm.owner[i] as usize;
                let nn = hm.neighbour[i] as usize;
                let d = hm.c[nn] - hm.c[o];
                let den = f[nn] - f[o];
                let w_up: Scalar = if phi_h[i] >= 0.0 { 1.0 } else { 0.0 };
                let wc = hm.weights[i];

                let want = if den == 0.0 {
                    wc
                } else {
                    let gu = if phi_h[i] >= 0.0 { grad_h[o] } else { grad_h[nn] };
                    let r = 2.0 * d.dot(gu) / den - 1.0;
                    w_up + l.psi(r) * (wc - w_up)
                };

                assert!(
                    (got[i] - want).abs() < 1e-13,
                    "{l:?} face {i}: device {} vs host {want}",
                    got[i]
                );
            }
        }
        Ok(())
    }

    /// A limited scheme with no gradient is an error, not a quiet fall-back to
    /// upwind. This is the whole reason the argument exists.
    #[test]
    fn a_limited_scheme_without_a_gradient_is_refused() -> Result<()> {
        let Some(fx) = fixture([3, 3, 3], Vec3::new(0.3, 0.3, 0.3), true) else {
            return Ok(());
        };
        let (gpu, k, hm, m) = (&fx.gpu, &fx.k, &fx.hm, &fx.m);

        let zeros_c = vec![0.0 as Scalar; hm.n_cells];
        let zeros_b = vec![0.0 as Scalar; hm.n_boundary_faces];
        let psi = dirichlet_field(gpu, m, hm, &zeros_c, &zeros_b)?;
        let phi = GpuSurfaceScalarField::zeros(gpu, m, "phi")?;

        let mut w: DevBuf<Scalar> = gpu.zeros(hm.n_internal_faces)?;
        let e = div_scheme_weights(
            gpu,
            k,
            Some(&mut w),
            None,
            DivScheme::Limited(Limiter::VanLeer),
            &phi,
            &psi,
            None,
            m,
        );
        assert!(e.is_err(), "a limited scheme with no gradient was accepted");
        Ok(())
    }

    /// Central and upwind weights are what they say they are.
    #[test]
    fn unlimited_weights_are_central_and_upwind() -> Result<()> {
        let Some(fx) = fixture([4, 3, 3], Vec3::new(0.3, 0.4, 0.25), true) else {
            return Ok(());
        };
        let (gpu, k, hm, m) = (&fx.gpu, &fx.k, &fx.hm, &fx.m);

        let zeros_c = vec![0.0 as Scalar; hm.n_cells];
        let zeros_b = vec![0.0 as Scalar; hm.n_boundary_faces];
        let psi = dirichlet_field(gpu, m, hm, &zeros_c, &zeros_b)?;

        let phi_h: Vec<Scalar> = (0..hm.n_internal_faces)
            .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        let mut phi = GpuSurfaceScalarField::zeros(gpu, m, "phi")?;
        gpu.write(&mut phi.f, &phi_h)?;

        let mut w: DevBuf<Scalar> = gpu.zeros(hm.n_internal_faces)?;

        div_scheme_weights(gpu, k, Some(&mut w), None, DivScheme::Central, &phi, &psi, None, m)?;
        gpu.sync()?;
        let got = gpu.download(&w)?;
        for (i, g) in got.iter().enumerate() {
            assert_eq!(*g, hm.weights[i]);
        }

        div_scheme_weights(gpu, k, Some(&mut w), None, DivScheme::Upwind, &phi, &psi, None, m)?;
        gpu.sync()?;
        let got = gpu.download(&w)?;
        for (i, g) in got.iter().enumerate() {
            assert_eq!(*g, if phi_h[i] >= 0.0 { 1.0 } else { 0.0 });
        }

        // linearUpwind's IMPLICIT weights are the upwind ones.
        div_scheme_weights(
            gpu, k, Some(&mut w), None, DivScheme::LinearUpwind, &phi, &psi, None, m,
        )?;
        gpu.sync()?;
        let got2 = gpu.download(&w)?;
        assert_eq!(got, got2);
        Ok(())
    }

    // ----------------------------------------------------------------------
    //  §3.3 / §3.4  ddt and sources
    // ----------------------------------------------------------------------

    #[test]
    fn ddt_and_sources_are_what_the_specification_says() -> Result<()> {
        let Some(fx) = fixture([3, 3, 2], Vec3::new(0.5, 0.25, 0.4), true) else {
            return Ok(());
        };
        let (gpu, k, hm, m) = (&fx.gpu, &fx.k, &fx.hm, &fx.m);

        let n = hm.n_cells;
        let psi0: Vec<Scalar> = (0..n).map(|c| 1.0 + 0.1 * c as Scalar).collect();
        let psi00: Vec<Scalar> = (0..n).map(|c| 2.0 - 0.05 * c as Scalar).collect();
        let psi: Vec<Scalar> = (0..n).map(|c| 0.3 + 0.02 * c as Scalar).collect();

        let d_psi0 = gpu.upload(&psi0)?;
        let d_psi00 = gpu.upload(&psi00)?;
        let d_psi = gpu.upload(&psi)?;

        let r_dt: Scalar = 12.5;

        // ---- Euler ----
        let mut a = GpuLduMatrix::new(gpu, m)?;
        a.zero(gpu)?;
        fvm_ddt_euler(gpu, k, &mut a, m, None, None, &d_psi0, r_dt, 1.0)?;
        gpu.sync()?;
        let diag = gpu.download(&a.diag)?;
        let src = gpu.download(&a.source)?;
        for c in 0..n {
            let want_d = hm.v[c] * r_dt;
            assert!((diag[c] - want_d).abs() < 1e-13 * want_d);
            assert!((src[c] - want_d * psi0[c]).abs() < 1e-13 * want_d * psi0[c]);
        }

        // ---- BDF2 ----
        a.zero(gpu)?;
        fvm_ddt_bdf2(gpu, k, &mut a, m, None, None, None, &d_psi0, &d_psi00, r_dt, 1.0)?;
        gpu.sync()?;
        let diag = gpu.download(&a.diag)?;
        let src = gpu.download(&a.source)?;
        for c in 0..n {
            let base = hm.v[c] * r_dt;
            let want_d = 1.5 * base;
            let want_s = base * (2.0 * psi0[c] - 0.5 * psi00[c]);
            assert!((diag[c] - want_d).abs() < 1e-13 * want_d);
            assert!((src[c] - want_s).abs() < 1e-13 * want_s.abs());
        }

        // ---- steady state writes nothing ----
        a.zero(gpu)?;
        fvm_ddt_euler(gpu, k, &mut a, m, None, None, &d_psi0, 0.0, 1.0)?;
        gpu.sync()?;
        assert_eq!(max_abs(&gpu.download(&a.diag)?), 0.0);
        assert_eq!(max_abs(&gpu.download(&a.source)?), 0.0);

        // ---- Su / Sp / SuSp ----
        let s: Vec<Scalar> = (0..n)
            .map(|c| if c % 2 == 0 { 0.7 } else { -0.4 } + 0.01 * c as Scalar)
            .collect();
        let d_s = gpu.upload(&s)?;

        a.zero(gpu)?;
        fvm_su(gpu, k, &mut a, m, &d_s, 1.0)?;
        gpu.sync()?;
        let src = gpu.download(&a.source)?;
        for c in 0..n {
            assert!((src[c] - hm.v[c] * s[c]).abs() < 1e-14);
        }

        a.zero(gpu)?;
        fvm_sp(gpu, k, &mut a, m, &d_s, 1.0)?;
        gpu.sync()?;
        let diag = gpu.download(&a.diag)?;
        for c in 0..n {
            assert!((diag[c] - hm.v[c] * s[c]).abs() < 1e-14);
        }

        a.zero(gpu)?;
        fvm_susp(gpu, k, &mut a, m, &d_s, &d_psi, 1.0)?;
        gpu.sync()?;
        let diag = gpu.download(&a.diag)?;
        let src = gpu.download(&a.source)?;
        for c in 0..n {
            let v = hm.v[c];
            assert!((diag[c] - v * s[c].max(0.0)).abs() < 1e-14);
            assert!((src[c] + v * s[c].min(0.0) * psi[c]).abs() < 1e-14);
        }

        // Patankar's split is exact: SuSp and a fully implicit Sp give the
        // same equation at the current psi.
        for c in 0..n {
            let implicit = hm.v[c] * s[c] * psi[c];
            let split = diag[c] * psi[c] - src[c];
            assert!((implicit - split).abs() < 1e-13 * implicit.abs().max(1.0));
        }

        // Passing one density but not the other is rejected rather than
        // silently discretising the wrong equation.
        let rho = gpu.upload(&vec![1.0 as Scalar; n])?;
        assert!(fvm_ddt_euler(gpu, k, &mut a, m, Some(&rho), None, &d_psi0, r_dt, 1.0).is_err());

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  sn_grad_flux against the matrix it must agree with
    // ----------------------------------------------------------------------

    /// A linear field has zero laplacian, so the diffusive flux it implies is
    /// discretely conservative: every cell's faces sum to zero. That is the
    /// property `potential_flow` relies on, stated as a test.
    #[test]
    fn sn_grad_flux_of_a_linear_field_is_conservative() -> Result<()> {
        let Some(fx) = fixture([4, 3, 3], Vec3::new(0.3, 0.4, 0.25), true) else {
            return Ok(());
        };
        let (gpu, k, hm, m) = (&fx.gpu, &fx.k, &fx.hm, &fx.m);

        let a = Vec3::new(1.3, -0.6, 0.9);
        let f: Vec<Scalar> = (0..hm.n_cells).map(|c| a.dot(hm.c[c])).collect();
        let bf: Vec<Scalar> = (0..hm.n_boundary_faces).map(|i| a.dot(hm.b_cf[i])).collect();
        let psi = dirichlet_field(gpu, m, hm, &f, &bf)?;

        let d_gamma = gpu.upload(&hm.mag_sf)?;
        let d_b_gamma = gpu.upload(&hm.b_mag_sf)?;

        let mut phi = GpuSurfaceScalarField::zeros(gpu, m, "phi")?;
        sn_grad_flux(gpu, k, &mut phi, &psi, &d_gamma, &d_b_gamma, m)?;
        gpu.sync()?;

        let iphi = gpu.download(&phi.f)?;
        let bphi = gpu.download(&phi.bf)?;

        let mut worst: Scalar = 0.0;
        for c in 0..hm.n_cells {
            let mut sum: Scalar = 0.0;
            for j in hm.cf_offset[c] as usize..hm.cf_offset[c + 1] as usize {
                let ff = hm.cf_face[j] as usize;
                sum += if hm.cf_own[j] != 0 { iphi[ff] } else { -iphi[ff] };
            }
            for j in hm.bcf_offset[c] as usize..hm.bcf_offset[c + 1] as usize {
                sum += bphi[hm.bcf_face[j] as usize];
            }
            worst = worst.max(sum.abs());
        }

        let scale = max_abs(&iphi).max(max_abs(&bphi));
        assert!(
            worst <= 1e-12 * scale,
            "snGrad flux of a linear field leaves {worst} of imbalance"
        );
        Ok(())
    }

    /// A zero mesh must not launch a zero-block grid, which is an illegal
    /// configuration rather than a no-op.
    #[test]
    fn an_empty_mesh_launches_nothing() -> Result<()> {
        let Some(gpu) = gpu() else { return Ok(()) };
        let k = FvKernels::new(&gpu)?;

        let hm = HostMesh {
            n_cells: 0,
            n_internal_faces: 0,
            n_boundary_faces: 0,
            cf_offset: vec![0],
            bcf_offset: vec![0],
            ..Default::default()
        };
        let m = GpuMesh::upload(&gpu, &hm)?;

        let mut a = GpuLduMatrix::new(&gpu, &m)?;
        a.zero(&gpu)?;

        let empty: DevBuf<Scalar> = gpu.zeros(0)?;
        let psi = GpuScalarField::zeros(&gpu, &m, "psi")?;
        let phi = GpuSurfaceScalarField::zeros(&gpu, &m, "phi")?;

        fvm_ddt_euler(&gpu, &k, &mut a, &m, None, None, &empty, 1.0, 1.0)?;
        fvm_laplacian(&gpu, &k, &mut a, &m, &empty, &empty, &psi, 1.0)?;
        fvm_div_gauss(&gpu, &k, &mut a, &m, &phi, &empty, &empty, &psi, 1.0)?;
        fvm_div_bounded_correction(&gpu, &k, &mut a, &m, &phi, 1.0)?;
        fvm_su(&gpu, &k, &mut a, &m, &empty, 1.0)?;
        fvm_sp(&gpu, &k, &mut a, &m, &empty, 1.0)?;
        fvm_susp(&gpu, &k, &mut a, &m, &empty, &empty, 1.0)?;
        gpu.sync()?;
        Ok(())
    }
}
