// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Spalart-Allmaras, and the negative continuation - SPEC-LIT §56.
//!
//! Written from:
//!   Spalart & Allmaras, *AIAA Paper* 92-0439 (1992); *La Recherche
//!     Aerospatiale* 1 (1994) 5-21 - the original
//!   Allmaras, Johnson & Spalart, "Modifications and Clarifications for the
//!     Implementation of the Spalart-Allmaras Turbulence Model",
//!     **ICCFD7-1902** (2012),
//!     <https://www.iccfd.org/iccfd7/assets/pdf/papers/ICCFD7-1902_paper.pdf>
//!     - a freely distributed conference paper, read in full. **This is the
//!     implementation reference**: §3.1 gives the `S~` positivity fix and §3.2
//!     gives the negative model.
//!   NASA / TMBWG, *Turbulence Modeling Resource - The Spalart-Allmaras
//!     Turbulence Model*, <https://tmbwg.github.io/turbmodels/spalart.html> -
//!     US government-authored DOCUMENTATION, not source. Read. It states
//!     SA-noft2 and SA-neg to the printed digit, it is where the `r = 10`
//!     rule for the `Omega == 0` corner comes from, and it publishes
//!     `nu_t/nu = 0.210438` and `1.294234` at the two ends of the recommended
//!     far-field range - the two numbers [`fv1`] is gated against.
//!   Rumsey & Spalart, *AIAA J.* 47 (2009) 982-993 - why the free-stream
//!     `nu~/nu` matters
//!   Patankar, *Numerical Heat Transfer and Fluid Flow* (1980) §4.2
//!   ofgpu `SPEC-LIT.md` §56, and §57 for the one length scale DES replaces
//! No GPL-licensed source was consulted. OpenFOAM and SU2 were not opened,
//! searched or quoted.
//!
//! # Which variant, and why the default is `noft2`
//!
//! `SA-noft2` (`c_t3 = 0`) is the default because it is what the TMR treats
//! as the baseline for verification, and because the trip terms `f_t1` that
//! `f_t2` accompanies need a trip location the case format has no way to
//! express. All four combinations - with and without `f_t2`, with and without
//! the negative continuation - are reachable by name through
//! `RAS { variant ...; }`, and an unknown name is refused with the menu
//! rather than being read as the default (SPEC-LIT §56.8).
//!
//! # The one constant that is derived and not read
//!
//! `c_w1 = c_b1/kappa^2 + (1 + c_b2)/sigma` is **exactly** the condition that
//! makes the log layer an exact solution of the model (SPEC-LIT §56.4), so a
//! case that could set it independently could ask for a model with no log
//! layer. [`SaCoeffs::cw1`] computes it and `RAS { Cw1 ...; }` is refused by
//! name.
//!
//! # Where the published C1 claim does not hold, said here rather than found
//!
//! Allmaras et al. list "the PDE functions are C1 continuous at `nu~ = 0`"
//! among the design goals of the negative model. It is true term by term when
//! `c_t3 = 1.2` on BOTH sides, and **false for the production term under
//! SA-noft2**: the positive branch's slope at `nu~ = 0` is `c_b1 Omega` while
//! the negative branch's is `-0.2 c_b1 Omega`, a jump of `1.2 c_b1 Omega`.
//! That is what combining the TMR's two named variants produces, and it is
//! pinned by a test rather than tolerated - see
//! `tests::the_production_slope_jump_at_zero_is_exactly_1_2_cb1_omega`.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::GpuScalarField;
use crate::field_ops::{advance_time_levels, correct_boundary_conditions};
use crate::fv::{fvc_grad_scalar_scheme, fvm_su, fvm_susp};
use crate::mesh::{GpuMesh, HostMesh};
use crate::models::des::DesLengthScale;
use crate::solver::SolverPerformance;
use crate::turbulence::{
    nut_boundary, vorticity_mag, FlowState, RasCore, TurbulenceControls,
};
use crate::wallfunctions::WallFunctionCoeffs;
use crate::{Label, Scalar, Vec3};

// ==========================================================================
//  The closed forms, on the host - SPEC-LIT §56.1 to §56.5
//
//  Every one of these is transcribed independently of `cuda/sa.cu`'s device
//  inline of the same name, and `tests::host_and_device_agree` compares them
//  pointwise. A copy-paste divergence in either then shows up as a
//  measurement rather than as nothing.
// ==========================================================================

/// `f_v1 = chi^3/(chi^3 + c_v1^3)` - SPEC-LIT (56.1).
///
/// `chi f_v1(chi)` is the free-stream `nu_t/nu` the TMR publishes:
/// `0.210438` at `chi = 3` and `1.294234` at `chi = 5`, both to six figures.
#[must_use]
pub fn fv1(chi: Scalar, cv1: Scalar) -> Scalar {
    let c3 = chi * chi * chi;
    c3 / (c3 + cv1 * cv1 * cv1)
}

/// `f_v2 = 1 - chi/(1 + chi f_v1)` - SPEC-LIT (56.3).
///
/// Negative over a range of `chi`, which is the entire reason (56.9) exists.
#[must_use]
pub fn fv2(chi: Scalar, cv1: Scalar) -> Scalar {
    1.0 - chi / (1.0 + chi * fv1(chi, cv1))
}

/// The `S~` positivity fix, Allmaras et al. (11)-(13) - SPEC-LIT (56.9).
///
/// Identical to `Omega + Sbar` wherever `Sbar >= -c_v2 Omega`; asymptotes to
/// `(1 - c_v3) Omega = 0.1 Omega` as `Sbar/Omega -> -inf`, so `S~` is
/// strictly positive wherever `Omega` is. C0 and C1 at the join, which the
/// two constants `0.7`/`0.9` are exactly what arrange - see
/// `tests::the_stilde_fix_is_c1_at_the_join`.
#[must_use]
pub fn stilde(omega: Scalar, sbar: Scalar, cv2: Scalar, cv3: Scalar) -> Scalar {
    if sbar >= -cv2 * omega {
        omega + sbar
    } else {
        let num = cv2 * cv2 * omega + cv3 * sbar;
        let den = (cv3 - 2.0 * cv2) * omega - sbar;
        omega + omega * num / den
    }
}

/// `f_w = g[(1 + c_w3^6)/(g^6 + c_w3^6)]^(1/6)` with
/// `g = r + c_w2(r^6 - r)` - SPEC-LIT (56.4).
///
/// `f_w(1) = 1` exactly - the log-layer value - and `f_w` is bounded above by
/// `(1 + c_w3^6)^(1/6) = 65^(1/6) = 2.0051747` as `r -> inf`.
#[must_use]
pub fn fw(r: Scalar, cw2: Scalar, cw3: Scalar) -> Scalar {
    let g = r + cw2 * (r.powi(6) - r);
    let c6 = cw3.powi(6);
    g * ((1.0 + c6) / (g.powi(6) + c6)).powf(1.0 / 6.0)
}

/// `f_n = (c_n1 + chi^3)/(c_n1 - chi^3)` for `chi < 0`, exactly `1`
/// otherwise - SPEC-LIT (56.12).
#[must_use]
pub fn fn_(chi: Scalar, cn1: Scalar) -> Scalar {
    if chi >= 0.0 {
        return 1.0;
    }
    let c3 = chi * chi * chi;
    (cn1 + c3) / (cn1 - c3)
}

/// `N(x) = x^4 + x^3 - c_n1 x + c_n1`, the numerator of
/// `(nu + nu~ f_n)/nu` at `x = -chi > 0` - SPEC-LIT §56.5.
///
/// The negative model's diffusivity is positive for every `nu~ < 0` exactly
/// when `N > 0` on `x > 0`.
#[must_use]
pub fn neg_diffusivity_numerator(x: Scalar, cn1: Scalar) -> Scalar {
    x * x * x * x + x * x * x - cn1 * x + cn1
}

/// The `x` at which `N` and `N'` vanish together, `(1 + sqrt(10))/3` -
/// SPEC-LIT (56.14).
#[must_use]
pub fn cn1_bound_x() -> Scalar {
    (1.0 + (10.0 as Scalar).sqrt()) / 3.0
}

/// The largest `c_n1` for which `nu + nu~ f_n` stays positive everywhere:
/// `4 x*^3 + 3 x*^2 = 16.4577569` - SPEC-LIT (56.14).
///
/// The design note says the diffusivity "first goes negative at
/// `c_n1 ~ 16.46`". This is that number, derived rather than quoted, and
/// `tests::the_cn1_bound_is_where_the_derivation_says` checks both halves.
#[must_use]
pub fn cn1_bound() -> Scalar {
    let x = cn1_bound_x();
    4.0 * x * x * x + 3.0 * x * x
}

/// `f_w`'s supremum, `(1 + c_w3^6)^(1/6)` - `65^(1/6) = 2.0051747` at
/// `c_w3 = 2`.
#[must_use]
pub fn fw_supremum(cw3: Scalar) -> Scalar {
    (1.0 + cw3.powi(6)).powf(1.0 / 6.0)
}

// ==========================================================================
//  The variant, and the coefficients
// ==========================================================================

/// Which of the TMR's four named variants the case asked for - SPEC-LIT
/// §56.8.
///
/// *DESIGN.* The `variant` dictionary entry is ours; the four values are the
/// TMR's own nomenclature and each of its spellings is accepted as an alias.
/// Nothing is substituted: an unknown name is refused with the menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaVariant {
    /// `SA-noft2` - `f_t2` absent (`c_t3 = 0`), `nu~` bounded below at zero.
    /// The default, and what the TMR verifies against.
    Noft2,
    /// `SA-noft2-neg` - `f_t2` absent, and the negative continuation of
    /// SPEC-LIT (56.11) active instead of the bound.
    Noft2Neg,
    /// `SA` - the full model with `f_t2 = c_t3 exp(-c_t4 chi^2)`, `nu~`
    /// bounded below at zero.
    Ft2,
    /// `SA-neg` - the full model with the negative continuation.
    Ft2Neg,
}

impl SaVariant {
    /// The name a case writes.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Noft2 => "noft2",
            Self::Noft2Neg => "noft2-neg",
            Self::Ft2 => "ft2",
            Self::Ft2Neg => "ft2-neg",
        }
    }

    /// True when the negative continuation runs and `nu~` is therefore NOT
    /// bounded below - SPEC-LIT §56.5.
    #[must_use]
    pub fn negative(self) -> bool {
        matches!(self, Self::Noft2Neg | Self::Ft2Neg)
    }

    /// True when the POSITIVE branch carries `f_t2`.
    #[must_use]
    pub fn ft2(self) -> bool {
        matches!(self, Self::Ft2 | Self::Ft2Neg)
    }

    /// Every spelling that selects a variant, including the TMR's own.
    pub const NAMES: &'static [(&'static str, SaVariant)] = &[
        ("noft2", SaVariant::Noft2),
        ("SA-noft2", SaVariant::Noft2),
        ("noft2-neg", SaVariant::Noft2Neg),
        ("SA-noft2-neg", SaVariant::Noft2Neg),
        ("ft2", SaVariant::Ft2),
        ("SA", SaVariant::Ft2),
        ("standard", SaVariant::Ft2),
        ("ft2-neg", SaVariant::Ft2Neg),
        ("SA-neg", SaVariant::Ft2Neg),
        ("neg", SaVariant::Ft2Neg),
    ];

    /// The menu a rejected name is shown.
    #[must_use]
    pub fn menu() -> Vec<&'static str> {
        vec!["noft2", "noft2-neg", "ft2", "ft2-neg"]
    }

    /// Parse, or `None` - the caller turns that into the §13.4 refusal so the
    /// message names the dictionary path.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Self::NAMES
            .iter()
            .find(|(n, _)| *n == s)
            .map(|(_, v)| *v)
    }
}

/// SPEC-LIT §56.1's constants, and §56.8's dictionary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SaCoeffs {
    pub variant: SaVariant,
    pub cb1: Scalar,
    pub cb2: Scalar,
    pub cv1: Scalar,
    pub cv2: Scalar,
    pub cv3: Scalar,
    pub cw2: Scalar,
    pub cw3: Scalar,
    /// The NEGATIVE branch's `c_t3`, always. It must exceed 1 for
    /// `P_n >= 0` (SPEC-LIT §56.5), so it is `1.2` even when the positive
    /// branch is `noft2` and carries `c_t3 = 0`. **This is the one place the
    /// two branches must not share a constant.**
    pub ct3: Scalar,
    pub ct4: Scalar,
    pub cn1: Scalar,
    pub sigma: Scalar,
    pub kappa: Scalar,
    pub rlim: Scalar,
}

impl Default for SaCoeffs {
    fn default() -> Self {
        Self {
            variant: SaVariant::Noft2,
            cb1: 0.1355,
            cb2: 0.622,
            cv1: 7.1,
            cv2: 0.7,
            cv3: 0.9,
            cw2: 0.3,
            cw3: 2.0,
            ct3: 1.2,
            ct4: 0.5,
            cn1: 16.0,
            sigma: 2.0 / 3.0,
            kappa: 0.41,
            rlim: 10.0,
        }
    }
}

impl SaCoeffs {
    /// `c_w1 = c_b1/kappa^2 + (1 + c_b2)/sigma` - SPEC-LIT (56.6).
    ///
    /// DERIVED, never read from a case: (56.6) IS the log layer (§56.4).
    #[must_use]
    pub fn cw1(&self) -> Scalar {
        self.cb1 / (self.kappa * self.kappa) + (1.0 + self.cb2) / self.sigma
    }

    /// The POSITIVE branch's `c_t3` - zero under a `noft2` variant.
    #[must_use]
    pub fn ct3_positive(&self) -> Scalar {
        if self.variant.ft2() {
            self.ct3
        } else {
            0.0
        }
    }

    pub fn check(&self) -> Result<()> {
        let bad = |what: &str, v: Scalar| {
            Err(Error::Config(format!(
                "SpalartAllmaras: `{what}` = {v} is not usable; SPEC-LIT §56.1"
            )))
        };
        if !(self.sigma > 0.0) {
            return bad("sigmaNut", self.sigma);
        }
        if !(self.kappa > 0.0) {
            return bad("kappa", self.kappa);
        }
        if !(self.cv1 > 0.0) {
            return bad("Cv1", self.cv1);
        }
        if !(self.rlim > 0.0) {
            return bad("rlim", self.rlim);
        }
        // SPEC-LIT (56.14): above 16.4577569 the negative branch's diffusivity
        // `nu + nu~ f_n` goes negative for some `nu~ < 0`, which is a
        // laplacian with the wrong sign, not a stiff one.
        if self.variant.negative() && self.cn1 >= cn1_bound() {
            return Err(Error::Config(format!(
                "SpalartAllmaras: `Cn1` = {} is at or above {:.6}, where the \
                 negative branch's diffusivity `nu + nu~ f_n` first goes \
                 negative (SPEC-LIT (56.14): the bound is 16.4577569, 4x^3 + 3x^2 at \
                 x = (1 + sqrt(10))/3). Allmaras et al. use 16",
                self.cn1,
                cn1_bound()
            )));
        }
        // SPEC-LIT §56.5: `P_n = c_b1 (1 - c_t3) Omega nu~` is non-negative
        // for `nu~ < 0` only when `c_t3 > 1`. A case that set it lower would
        // get a negative production driving `nu~` further negative, which is
        // the exact failure the negative model exists to prevent.
        if self.variant.negative() && self.ct3 <= 1.0 {
            return Err(Error::Config(format!(
                "SpalartAllmaras: `Ct3` = {} is not greater than 1, so the \
                 negative branch's production `c_b1 (1 - Ct3) Omega nu~` is \
                 NEGATIVE for nu~ < 0 and drives nu~ further from zero \
                 (SPEC-LIT §56.5). Allmaras et al. use 1.2 in the negative \
                 branch even when the positive branch is noft2",
                self.ct3
            )));
        }
        Ok(())
    }
}

// ==========================================================================
//  Kernels
// ==========================================================================

/// Every entry point in `cuda/sa.cu`, resolved once.
pub struct SaKernels {
    nut: CudaFunction,
    sources: CudaFunction,
    gamma_internal: CudaFunction,
    gamma_boundary: CudaFunction,
    bound: CudaFunction,
    log_layer_terms: CudaFunction,
}

impl SaKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::SA)?;
        Ok(Self {
            nut: k.func("saNut")?,
            sources: k.func("saSources")?,
            gamma_internal: k.func("saGammaInternal")?,
            gamma_boundary: k.func("saGammaBoundary")?,
            bound: k.func("saBoundNuTilda")?,
            log_layer_terms: k.func("saLogLayerTerms")?,
        })
    }
}

fn expect_len<T>(buf: &DevBuf<T>, want: usize, what: &str) -> Result<()> {
    if buf.len() == want {
        Ok(())
    } else {
        Err(Error::Config(format!(
            "spalart_allmaras: `{what}` has {} elements, expected {want}",
            buf.len()
        )))
    }
}

/// `nu_t = nu~ f_v1`, exactly zero where `nu~ < 0`, capped at `nut_max` -
/// SPEC-LIT (56.1)/(56.13).
pub fn sa_nut(
    gpu: &Gpu,
    kern: &SaKernels,
    nut: &mut DevBuf<Scalar>,
    nu_tilda: &DevBuf<Scalar>,
    nu: Scalar,
    cv1: Scalar,
    nut_max: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(nut, n, "nut")?;
    expect_len(nu_tilda, n, "nuTilda")?;

    let nl = n as Label;
    let f = kern.nut.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(nut)
            .arg(nu_tilda)
            .arg(&nu)
            .arg(&cv1)
            .arg(&nut_max)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// SPEC-LIT (56.15): `su` explicit, `sp` zero, and production AND destruction
/// in ONE `susp` whose sign Patankar's rule splits.
///
/// `d` is the true wall distance; `d_tilde` is §57's hybrid length scale and
/// is read by the destruction term alone. A pure RANS run passes the same
/// buffer for both, which is why plain SA is the hybrid with the substitution
/// not made rather than a second code path.
#[allow(clippy::too_many_arguments)]
pub fn sa_sources(
    gpu: &Gpu,
    kern: &SaKernels,
    su: &mut DevBuf<Scalar>,
    sp: &mut DevBuf<Scalar>,
    susp: &mut DevBuf<Scalar>,
    nu_tilda: &DevBuf<Scalar>,
    grad_nu_tilda: &DevBuf<Vec3>,
    omega: &DevBuf<Scalar>,
    d: &DevBuf<Scalar>,
    d_tilde: &DevBuf<Scalar>,
    nu: Scalar,
    c: &SaCoeffs,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(su, n, "su")?;
    expect_len(sp, n, "sp")?;
    expect_len(susp, n, "susp")?;
    expect_len(nu_tilda, n, "nuTilda")?;
    expect_len(grad_nu_tilda, n, "grad nuTilda")?;
    expect_len(omega, n, "Omega")?;
    expect_len(d, n, "d")?;
    expect_len(d_tilde, n, "dTilde")?;

    let nl = n as Label;
    let cw1 = c.cw1();
    let ct3_pos = c.ct3_positive();
    let f = kern.sources.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(su)
            .arg(sp)
            .arg(susp)
            .arg(nu_tilda)
            .arg(grad_nu_tilda)
            .arg(omega)
            .arg(d)
            .arg(d_tilde)
            .arg(&nu)
            .arg(&c.cb1)
            .arg(&c.cb2)
            .arg(&c.cv1)
            .arg(&c.cv2)
            .arg(&c.cv3)
            .arg(&cw1)
            .arg(&c.cw2)
            .arg(&c.cw3)
            .arg(&ct3_pos)
            .arg(&c.ct3)
            .arg(&c.ct4)
            .arg(&c.sigma)
            .arg(&c.kappa)
            .arg(&c.rlim)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `Gamma_eff |Sf| = ((nu + nu~ f_n)/sigma)|Sf|` on every face - SPEC-LIT
/// §56.6.
///
/// The one place this model departs from `turbulence::face_diffusivity`: the
/// coefficient is built from the TRANSPORTED field, not from `nu_t`.
#[allow(clippy::too_many_arguments)]
pub fn sa_face_diffusivity(
    gpu: &Gpu,
    kern: &SaKernels,
    gamma: &mut DevBuf<Scalar>,
    b_gamma: &mut DevBuf<Scalar>,
    nu_tilda: &GpuScalarField,
    m: &GpuMesh,
    nu: Scalar,
    sigma: Scalar,
    cn1: Scalar,
) -> Result<()> {
    expect_len(gamma, m.n_internal_faces, "gamma")?;
    expect_len(b_gamma, m.n_boundary_faces, "b_gamma")?;

    if m.n_internal_faces > 0 {
        let n = m.n_internal_faces;
        let nl = n as Label;
        let f = kern.gamma_internal.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(gamma)
                .arg(&nu_tilda.f)
                .arg(&m.weights)
                .arg(&m.mag_sf)
                .arg(&m.owner)
                .arg(&m.neighbour)
                .arg(&nu)
                .arg(&sigma)
                .arg(&cn1)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    if m.n_boundary_faces > 0 {
        let n = m.n_boundary_faces;
        let nl = n as Label;
        let f = kern.gamma_boundary.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(b_gamma)
                .arg(&nu_tilda.bf)
                .arg(&m.b_mag_sf)
                .arg(&nu)
                .arg(&sigma)
                .arg(&cn1)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
    }

    Ok(())
}

/// `nu~ <- max(nu~, 0)` - *DESIGN*, SPEC-LIT §56.5, and launched ONLY by the
/// variants without the negative continuation.
pub fn bound_nu_tilda(
    gpu: &Gpu,
    kern: &SaKernels,
    nu_tilda: &mut DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(nu_tilda, n, "nuTilda")?;
    let nl = n as Label;
    let f = kern.bound.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(nu_tilda)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// The production and destruction terms of (56.2), separately, as fields -
/// the diagnostic SPEC-LIT §56.4's log-layer identity is measured through.
///
/// Separately, because a gate that reports one number cannot say which term
/// moved, and the whole value of the identity is that it does.
#[allow(clippy::too_many_arguments)]
pub fn sa_log_layer_terms(
    gpu: &Gpu,
    kern: &SaKernels,
    prod: &mut DevBuf<Scalar>,
    dest: &mut DevBuf<Scalar>,
    nu_tilda: &DevBuf<Scalar>,
    omega: &DevBuf<Scalar>,
    d: &DevBuf<Scalar>,
    nu: Scalar,
    c: &SaCoeffs,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(prod, n, "prod")?;
    expect_len(dest, n, "dest")?;

    let nl = n as Label;
    let cw1 = c.cw1();
    let ct3_pos = c.ct3_positive();
    let f = kern.log_layer_terms.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(prod)
            .arg(dest)
            .arg(nu_tilda)
            .arg(omega)
            .arg(d)
            .arg(&nu)
            .arg(&c.cb1)
            .arg(&c.cv1)
            .arg(&c.cv2)
            .arg(&c.cv3)
            .arg(&cw1)
            .arg(&c.cw2)
            .arg(&c.cw3)
            .arg(&ct3_pos)
            .arg(&c.ct4)
            .arg(&c.kappa)
            .arg(&c.rlim)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  The model
// ==========================================================================

/// Spalart-Allmaras, resident on the device - SPEC-LIT §56.
///
/// `des` is §57: `Some` makes this a DES97/DDES/IDDES hybrid, and `None`
/// leaves the destruction term reading the wall distance itself. There is no
/// second code path either way - `d_tilde` is an argument to one kernel.
pub struct SpalartAllmaras<'m> {
    core: RasCore<'m>,
    kern: SaKernels,
    coeffs: SaCoeffs,

    nu_tilda: GpuScalarField,

    /// `[n_cells]` the wall distance of SPEC-LIT §6.6, copied in at
    /// construction. Owned rather than borrowed for the reason
    /// [`crate::models::KOmegaSst`] owns its own: it is computed once at
    /// setup and the model outlives the `WallDistance` that produced it.
    y: DevBuf<Scalar>,

    /// `[n_cells]` the vorticity magnitude of (56.3), the Frobenius norm of
    /// the full velocity gradient that §57's `r_d` reads, and `grad nu~` for
    /// the `c_b2` term.
    omega: DevBuf<Scalar>,
    grad_frob: DevBuf<Scalar>,
    grad_nu_tilda: DevBuf<Vec3>,

    des: Option<DesLengthScale>,
}

impl<'m> SpalartAllmaras<'m> {
    /// `y` is the wall distance of SPEC-LIT §6.6 and is copied, not borrowed.
    ///
    /// `wall_faces` carries §15.5's two independent face sets. SA pins no
    /// near-wall cell to a wall relation - `nu~ = 0` is an exact Dirichlet
    /// condition (§56.7) - so `constrained_cells` is not read here; `nut`'s
    /// own set still is, because `nu_t`'s wall value is `nut`'s business
    /// whichever model computed the interior.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        mesh: &'m GpuMesh,
        coeffs: SaCoeffs,
        ctrl: TurbulenceControls,
        wall: WallFunctionCoeffs,
        wall_faces: &crate::field_setup::WallFaces,
        roughness: &crate::field_setup::NutRoughness,
        y: &DevBuf<Scalar>,
    ) -> Result<Self> {
        coeffs.check()?;
        if y.len() != mesh.n_cells && mesh.n_cells > 0 {
            return Err(Error::Config(format!(
                "SpalartAllmaras::new: the wall distance has {} entries and \
                 the mesh {} cells (SPEC-LIT §6.6)",
                y.len(),
                mesh.n_cells
            )));
        }
        let nc = mesh.n_cells.max(1);
        let fld = crate::field_ops::FieldKernels::new(gpu)?;
        let mut y_own: DevBuf<Scalar> = gpu.zeros(nc)?;
        crate::field_ops::copy_field(gpu, &fld, &mut y_own, y, mesh.n_cells)?;
        Ok(Self {
            core: RasCore::new(gpu, hm, mesh, ctrl, wall, wall_faces, roughness)?,
            kern: SaKernels::new(gpu)?,
            coeffs,
            nu_tilda: GpuScalarField::zeros(gpu, mesh, "nuTilda")?,
            y: y_own,
            omega: gpu.zeros(nc)?,
            grad_frob: gpu.zeros(nc)?,
            grad_nu_tilda: gpu.zeros(nc)?,
            des: None,
        })
    }

    /// Attach §57's hybrid length scale. `None` (the default) leaves this a
    /// pure RANS model with the destruction term reading `d` itself.
    pub fn set_des(&mut self, des: Option<DesLengthScale>) {
        self.des = des;
    }

    #[must_use]
    pub fn des(&self) -> Option<&DesLengthScale> {
        self.des.as_ref()
    }

    pub fn nu_tilda(&self) -> &GpuScalarField {
        &self.nu_tilda
    }
    pub fn nu_tilda_mut(&mut self) -> &mut GpuScalarField {
        &mut self.nu_tilda
    }
    pub fn nut(&self) -> &GpuScalarField {
        &self.core.nut
    }
    pub fn nut_mut(&mut self) -> &mut GpuScalarField {
        &mut self.core.nut
    }
    pub fn coeffs(&self) -> &SaCoeffs {
        &self.coeffs
    }
    pub fn wall_distance(&self) -> &DevBuf<Scalar> {
        &self.y
    }
    /// The vorticity magnitude of (56.3), for the gates of §56.10 - it is
    /// what the production term reads and there is no way to check it from
    /// `nu_t` alone.
    pub fn omega(&self) -> &DevBuf<Scalar> {
        &self.omega
    }
    pub fn grad_frobenius(&self) -> &DevBuf<Scalar> {
        &self.grad_frob
    }
    pub fn core(&self) -> &RasCore<'m> {
        &self.core
    }
    pub fn core_mut(&mut self) -> &mut RasCore<'m> {
        &mut self.core
    }
    pub fn freeze_nut(&mut self, gpu: &Gpu) -> Result<()> {
        self.core.freeze_nut(gpu)
    }

    pub fn named_fields(&self) -> Vec<(&'static str, &GpuScalarField)> {
        vec![("nuTilda", &self.nu_tilda), ("nut", &self.core.nut)]
    }
    pub fn named_fields_mut(&mut self) -> Vec<(&'static str, &mut GpuScalarField)> {
        vec![
            ("nuTilda", &mut self.nu_tilda),
            ("nut", &mut self.core.nut),
        ]
    }

    /// The length scale the destruction term divides by: `d` for a RANS run,
    /// §57's `d_tilde` for a hybrid. **One buffer, chosen once.**
    fn destruction_length(&self) -> &DevBuf<Scalar> {
        match &self.des {
            Some(d) => d.length(),
            None => &self.y,
        }
    }

    /// Bound (or not), evaluate the boundaries, build `grad U`-derived
    /// quantities and the first `nu_t`.
    pub fn initialise(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        let n = self.core.mesh.n_cells;

        if !self.coeffs.variant.negative() {
            bound_nu_tilda(gpu, &self.kern, &mut self.nu_tilda.f, n)?;
        }
        correct_boundary_conditions(gpu, &self.core.fld, &mut self.nu_tilda, self.core.mesh)?;

        self.core.update_flow_derived(gpu, flow)?;
        self.update_invariants(gpu)?;
        self.correct_nut(gpu, flow)?;
        self.core.store_k_prev(gpu, &self.nu_tilda.f)?;
        Ok(())
    }

    /// `Omega` and `F` from the current `grad U` - SPEC-LIT §56.2.
    fn update_invariants(&mut self, gpu: &Gpu) -> Result<()> {
        let n = self.core.mesh.n_cells;
        vorticity_mag(gpu, &self.core.turb, &mut self.omega, &self.core.grad_u, n)?;
        crate::turbulence::grad_frobenius(
            gpu,
            &self.core.turb,
            &mut self.grad_frob,
            &self.core.grad_u,
            n,
        )
    }

    /// One outer step: assemble and solve `nu~`, then update `nu_t`.
    ///
    /// Returns the same `(dissipation, k)` pair shape every other model here
    /// returns, so a driver that prints two residual columns keeps printing
    /// two. SA has ONE equation, so the same performance record is returned
    /// twice rather than a second solve being invented - `correct` says so
    /// and `CoupledSpalartAllmaras` repeats it.
    pub fn correct(
        &mut self,
        gpu: &Gpu,
        flow: &FlowState,
    ) -> Result<(SolverPerformance, SolverPerformance)> {
        let n = self.core.mesh.n_cells;
        let ctrl = self.core.ctrl;
        let c = self.coeffs;
        let nu = flow.nu;

        self.core.store_k_prev(gpu, &self.nu_tilda.f)?;
        advance_time_levels(gpu, &self.core.fld, &mut self.nu_tilda)?;
        self.core.ddt.advance(ctrl.delta_t);

        self.core.update_flow_derived(gpu, flow)?;
        self.update_invariants(gpu)?;

        // §57: the hybrid length scale, from the PREVIOUS iteration's `nu_t`.
        // The lag is deliberate and is SPEC-LIT §57.9's named fixed point.
        if let Some(des) = &mut self.des {
            des.update_sa(gpu, &self.core.nut.f, &self.grad_frob, &self.y, nu, n)?;
        }

        // `grad nu~` for the non-conservative `c_b2 (grad nu~)^2` term.
        fvc_grad_scalar_scheme(
            gpu,
            &self.core.fv,
            &mut self.grad_nu_tilda,
            &self.nu_tilda,
            self.core.mesh,
            ctrl.grad_scheme,
        )?;

        // ---- the diffusivity, then the rest of the assembly ---------------
        // §56.6: `(nu + nu~ f_n)/sigma` is built from the TRANSPORTED field,
        // which is the one thing `RasCore`'s three constant/blended entry
        // points cannot express, so the face buffers are filled here and the
        // shared tail runs unchanged.
        {
            let Self { core, kern, nu_tilda, .. } = self;
            let psi: &GpuScalarField = nu_tilda;
            core.assemble_transport_with_face_diffusivity(
                gpu,
                flow,
                psi,
                ctrl.eps_conv(),
                |gamma, b_gamma, mesh| {
                    sa_face_diffusivity(
                        gpu, kern, gamma, b_gamma, psi, mesh, nu, c.sigma, c.cn1,
                    )
                },
            )?;
        }

        // §56.6/(56.15): one `su` and one `susp`, with `d` and `d_tilde`
        // handed in separately. Destructured rather than reached through two
        // `&self` methods so the borrow checker can see that `su`/`sp`/`susp`
        // and `y`/`omega`/`des` are disjoint fields.
        {
            let Self {
                core,
                kern,
                coeffs,
                nu_tilda,
                y,
                omega,
                grad_nu_tilda,
                des,
                ..
            } = self;
            let dtil: &DevBuf<Scalar> = match des.as_ref() {
                Some(d) => d.length(),
                None => y,
            };
            let RasCore { su, sp, susp, .. } = core;
            sa_sources(
                gpu,
                kern,
                su,
                sp,
                susp,
                &nu_tilda.f,
                grad_nu_tilda,
                omega,
                y,
                dtil,
                nu,
                coeffs,
                n,
            )?;
        }

        fvm_su(gpu, &self.core.fv, &mut self.core.a, self.core.mesh, &self.core.su, 1.0)?;
        fvm_susp(
            gpu,
            &self.core.fv,
            &mut self.core.a,
            self.core.mesh,
            &self.core.susp,
            &self.nu_tilda.f,
            1.0,
        )?;

        // §56.6: `constrain_walls = false`. There is no wall function on
        // `nu~` to pin a near-wall cell to.
        let sc = ctrl.epsilon_solver;
        let perf = self
            .core
            .solve_equation(gpu, &mut self.nu_tilda, ctrl.eps_relax, &sc, false)?;

        if !c.variant.negative() {
            bound_nu_tilda(gpu, &self.kern, &mut self.nu_tilda.f, n)?;
        }
        correct_boundary_conditions(gpu, &self.core.fld, &mut self.nu_tilda, self.core.mesh)?;

        self.correct_nut(gpu, flow)?;

        Ok((perf, perf))
    }

    /// `nu_t = nu~ f_v1` from the NEW `nu~`, then the boundary values.
    pub fn correct_nut(&mut self, gpu: &Gpu, flow: &FlowState) -> Result<()> {
        let n = self.core.mesh.n_cells;
        let ctrl = self.core.ctrl;
        let wall = self.core.wall;
        let nut_max = self.core.nut_max(flow.nu);

        sa_nut(
            gpu,
            &self.kern,
            &mut self.core.nut.f,
            &self.nu_tilda.f,
            flow.nu,
            self.coeffs.cv1,
            nut_max,
            n,
        )?;

        correct_boundary_conditions(gpu, &self.core.fld, &mut self.core.nut, self.core.mesh)?;
        nut_boundary(gpu, &self.core.turb, &mut self.core.nut, self.core.mesh)?;

        // §56.7: `nu_t`'s wall value is still `nut`'s own business. SA gives
        // the wall function `k` nowhere to come from, so the near-wall value
        // is whatever `nut`'s patch type says - `nutLowReWallFunction`
        // (zero) on a resolved mesh, and the existing `NutkWallFunction`
        // triple where a case asked for one, evaluated from the `k` a
        // one-equation model does not have. That is why `update_nut` is NOT
        // called here: there is no `k` to hand it. A case whose `nut` names a
        // `k`-based wall function under SA is refused in `registry`.
        let _ = (wall, ctrl);
        Ok(())
    }

    /// `max|Δnu~|/max|nu~|` since the last call to `correct`.
    pub fn convergence_measure(&mut self, gpu: &Gpu) -> Result<Scalar> {
        let Self { core, nu_tilda, .. } = self;
        core.convergence_measure(gpu, &nu_tilda.f)
    }

    /// The destruction length actually in use, for the gates of §57.11.
    pub fn destruction_length_buffer(&self) -> &DevBuf<Scalar> {
        self.destruction_length()
    }
}

#[cfg(test)]
pub(crate) mod tests;
