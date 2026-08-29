// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The contact angle in the volume-of-fluid interface - SPEC-LIT §39.
//!
//! Written from:
//!   T. Young, *Phil. Trans. R. Soc.* 95 (1805) 65-87 - the equilibrium angle
//!   C. Huh, L. E. Scriven, *J. Colloid Interface Sci.* 35 (1971) 85-101 -
//!     the moving contact-line singularity, and why every model needs a
//!     microscopic cut-off
//!   O. V. Voinov, *Fluid Dyn.* 11 (1976) 714-721
//!   R. G. Cox, *J. Fluid Mech.* 168 (1986) 169-194 - the asymptotic matching
//!     that gives `theta^3 = theta_m^3 + 9 Ca ln(L/L_m)`
//!   R. L. Hoffman, *J. Colloid Interface Sci.* 50 (1975) 228-241 - the
//!     master curve both correlations here are fitted to
//!   T.-S. Jiang, S.-G. Oh, J. C. Slattery, *J. Colloid Interface Sci.* 69
//!     (1979) 74-77 - the explicit fit used here
//!   S. Afkhami, S. Zaleski, M. Bussmann, *J. Comput. Phys.* 228 (2009)
//!     5370-5389 - the mesh-dependent angle, named and NOT implemented here
//!   Y. Sui, H. Ding, P. D. M. Spelt, *Annu. Rev. Fluid Mech.* 46 (2014)
//!     97-119 - the review
//!   E. W. Washburn, *Phys. Rev.* 17 (1921) 273-283 - capillary rise, and
//!     Jurin's height, which is §39.7's Gate 1
//!   ofgpu `SPEC-LIT.md` §39 (all of it), §20.4 (the curvature gather this
//!     feeds), §4 (the Robin triple), §13.4 (what happens to a setting this
//!     solver does not have)
//! No GPL-licensed source was consulted.
//!
//! # The whole model, in one line
//!
//! `cuda/vof.cu`'s `vofFaceUnitNormalBoundary` writes `n_hat·Sf = 0` on every
//! non-cyclic boundary face, and its own comment says that is a modelling
//! statement: the interface meets the wall at ninety degrees. §39.2 derives
//! the replacement,
//!
//! ```text
//! bNHatf[i] = |Sf[i]| cos(theta_i)          (was: 0)
//! ```
//!
//! and that is the entire coupling into the curvature gather. Everything else
//! in this module exists to decide what `theta_i` is.
//!
//! # The `cos(pi/2)` trap
//!
//! `cos(pi/2)` is `6.123233995736766e-17` in double precision, not zero.
//! Writing `|Sf| cos(theta)` unconditionally would move every recorded VOF
//! measurement by that much times `|Sf|`, silently, for a case that asked for
//! nothing at all. So there are two guards, and both are tested:
//!
//! * the kernel takes an `enabled` flag and writes a LITERAL `0` when no
//!   contact-angle model is configured - not `|Sf| cos(pi/2)`;
//! * [`cos_deg`] maps ninety degrees to exactly `0.0` on the host, so a case
//!   that DOES configure `theta0 90` is also bit-for-bit the old behaviour.
//!
//! `the_cosine_of_ninety_degrees_is_not_zero` measures the premise rather
//! than asserting it, because a guard justified by an assertion is a guard
//! nobody will dare delete and nobody can check.
//!
//! # What is claimed
//!
//! §39.8: the geometry, the bit-identical default, the `alpha` fixed-gradient
//! triple, the closed-form behaviour of both correlations and of hysteresis,
//! and Jurin's height with the sign checked at both ends. NOT claimed: that a
//! live capillary-rise or drop-impact run reproduces a published
//! `theta_d(t)`. The mesh-dependent correction of Afkhami, Zaleski & Bussmann
//! is deliberately not implemented until there is a gate that would show it
//! doing what it is for.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::io::contract::unsupported;
use crate::io::fields::PatchFieldSpec;
use crate::{Label, Scalar};

// ==========================================================================
//  The correlation
// ==========================================================================

/// How `theta` depends on the contact-line speed - SPEC-LIT §39.4.
///
/// The discriminants are the `OFCA_*` codes in `cuda/vof.cu`, pinned by
/// [`tests::correlation_codes_match_the_device`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContactAngleCorrelation {
    /// `theta = theta_ref`, whatever the contact line is doing. With
    /// hysteresis off (`thetaA = thetaR = theta0`) this is Young's angle and
    /// nothing else.
    #[default]
    Static = 0,
    /// Jiang, Oh & Slattery's explicit fit to Hoffman's master curve:
    /// `(cos theta_e - cos theta_d)/(cos theta_e + 1) = tanh(4.96 Ca^0.702)`.
    /// Explicit in `theta_d`, so no inverse and no iteration.
    JiangOhSlattery = 1,
    /// Cox-Voinov: `theta_d^3 = theta_ref^3 + 9 Ca ln(L/L_m)`, with the
    /// logarithm supplied by the case as `lnLRatio`.
    CoxVoinov = 2,
}

impl ContactAngleCorrelation {
    pub fn name(self) -> &'static str {
        match self {
            Self::Static => "static",
            Self::JiangOhSlattery => "JiangOhSlattery",
            Self::CoxVoinov => "CoxVoinov",
        }
    }

    pub const NAMES: [&'static str; 3] = ["static", "JiangOhSlattery", "CoxVoinov"];

    /// SPEC-LIT §13.4. Kistler's fit to the same data is deliberately NOT
    /// here: its four constants come from a book chapter this project has not
    /// read, and SPEC-LIT §0 forbids pinning numbers on a recollection. An
    /// unrecognised spelling is an error naming the three that exist.
    pub fn parse(setting: &str, value: &str) -> Result<Self> {
        match value {
            "static" | "Static" | "constant" => Ok(Self::Static),
            "JiangOhSlattery" | "jiangOhSlattery" | "Jiang" => Ok(Self::JiangOhSlattery),
            "CoxVoinov" | "coxVoinov" | "Cox" => Ok(Self::CoxVoinov),
            other => unsupported(
                setting,
                other,
                &Self::NAMES,
                "static (the equilibrium angle, whatever the contact line is doing)",
                Self::Static,
            ),
        }
    }
}

// ==========================================================================
//  cos(theta), and the trap
// ==========================================================================

/// `cos(theta)` for `theta` in DEGREES, with ninety degrees mapped to
/// EXACTLY `0.0` - SPEC-LIT §39.2.
///
/// `cos(pi/2)` in IEEE-754 double is `6.123233995736766e-17`. That number
/// times `|Sf|` is what a case asking for `theta0 90` would add to every wall
/// face's `n_hat·Sf`, where the pre-§39 code wrote a literal zero. It is far
/// too small to see in a plot and far too large to be bitwise nothing, which
/// is the worst size for a difference to be: every recorded VOF measurement
/// would move and no test would say why.
///
/// The other three quarter-turns are exact too, for the same reason and at no
/// extra cost: `0 -> 1`, `180 -> -1`.
pub fn cos_deg(theta_deg: Scalar) -> Scalar {
    if theta_deg == 90.0 {
        0.0
    } else if theta_deg == 0.0 {
        1.0
    } else if theta_deg == 180.0 {
        -1.0
    } else {
        (theta_deg * (std::f64::consts::PI as Scalar) / 180.0).cos()
    }
}

/// The inverse, in degrees, for reporting. `acos` is clipped so a `cos` that
/// has drifted a bit outside `[-1, 1]` reports `0` or `180` rather than NaN.
pub fn acos_deg(c: Scalar) -> Scalar {
    c.clamp(-1.0, 1.0).acos() * 180.0 / (std::f64::consts::PI as Scalar)
}

// ==========================================================================
//  The dynamic angle - SPEC-LIT §39.4
// ==========================================================================

/// Jiang, Oh & Slattery's constants. *Not* case settings: they are the two
/// numbers that DEFINE the correlation, and a case that wants different ones
/// wants a different correlation. `J. Colloid Interface Sci.` **69** (1979)
/// 74-77.
pub const JIANG_A: Scalar = 4.96;
pub const JIANG_B: Scalar = 0.702;

/// `cos(theta_d)` from the equilibrium/advancing/receding angle and the
/// contact-line capillary number - SPEC-LIT §39.4, the host twin of
/// `vofDynamicCosTheta` in `cuda/vof.cu`.
///
/// `ca > 0` is ADVANCING. Hysteresis picks the reference angle first, then
/// the correlation is evaluated at it, so the two compose rather than
/// compete:
///
/// ```text
/// cos theta_ref = cos_a  if ca > 0 ,  cos_r  if ca < 0 ,  cos_e  if ca = 0
/// ```
///
/// **`ca = 0` returns `cos theta_ref` EXACTLY**, for every correlation. That
/// is what makes the dynamic model reduce to the static one at zero
/// contact-line speed to the last bit, and it is checked
/// ([`tests::every_correlation_reduces_to_the_static_angle_at_zero_speed`]).
///
/// `ln_l_ratio` is `ln(L/L_m)` and is read only by
/// [`ContactAngleCorrelation::CoxVoinov`].
pub fn cos_theta_dynamic(
    correlation: ContactAngleCorrelation,
    cos_e: Scalar,
    cos_a: Scalar,
    cos_r: Scalar,
    ca: Scalar,
    ln_l_ratio: Scalar,
) -> Scalar {
    // Hysteresis first - SPEC-LIT §39.4. The branch is on the SIGN of Ca and
    // not on a pinning band: a band would be a third number with no source in
    // the literature to fix it, and a case that wants "no motion over a
    // range" gets it exactly by writing thetaA = thetaR.
    let cos_ref = if ca > 0.0 {
        cos_a
    } else if ca < 0.0 {
        cos_r
    } else {
        cos_e
    };

    if !ca.is_finite() || ca == 0.0 {
        return cos_ref;
    }

    match correlation {
        ContactAngleCorrelation::Static => cos_ref,

        // (cos theta_ref - cos theta_d)/(cos theta_ref + 1) = tanh(A |Ca|^B),
        // solved for cos theta_d and signed by the direction of motion.
        //
        // Ca -> inf gives cos theta_d -> -1, i.e. theta_d -> 180 deg: the
        // displaced phase is completely dewetted. Ca -> 0 gives cos theta_ref
        // back, because tanh(0) is exactly 0.
        ContactAngleCorrelation::JiangOhSlattery => {
            let t = (JIANG_A * ca.abs().powf(JIANG_B)).tanh();
            let d = if ca > 0.0 { -t } else { t };
            (cos_ref + d * (1.0 + cos_ref)).clamp(-1.0, 1.0)
        }

        // theta_d^3 = theta_ref^3 + 9 Ca ln(L/L_m), in RADIANS, clipped into
        // (0, pi). The cube root is taken of a quantity that can go negative
        // for a strongly receding line, which is exactly where Cox's
        // small-angle asymptotics stop meaning anything - so the clip is not
        // cosmetic, it is the statement that the model has left its range.
        ContactAngleCorrelation::CoxVoinov => {
            let th = cos_ref.clamp(-1.0, 1.0).acos();
            let cubed = th * th * th + 9.0 * ca * ln_l_ratio;
            let d = if cubed <= 0.0 {
                0.0
            } else {
                cubed.cbrt().min(std::f64::consts::PI as Scalar)
            };
            d.cos()
        }
    }
}

/// Jurin's height, `h = 2 sigma cos(theta)/(rho g R)` - SPEC-LIT §39.7 Gate 1.
///
/// The equilibrium rise of a wetting liquid in a vertical capillary of radius
/// `R`, and the cleanest closed form there is for checking that `cos theta`
/// enters with the right SIGN: `theta > 90` gives a negative height, i.e.
/// depression, and `theta = 90` gives exactly zero.
///
/// Washburn, *Phys. Rev.* 17 (1921) 273-283.
pub fn jurin_height(
    sigma: Scalar,
    theta_deg: Scalar,
    rho: Scalar,
    g: Scalar,
    radius: Scalar,
) -> Scalar {
    2.0 * sigma * cos_deg(theta_deg) / (rho * g * radius)
}

/// The Lucas-Washburn viscous-regime rise, `h(t) = sqrt(sigma R cos(theta)
/// t/(2 mu))` - the same statement in time, and zero at ninety degrees for
/// the same reason.
///
/// Returns `0` where the angle is non-wetting: the closed form has no real
/// root there, and the physics is that the liquid does not rise at all.
pub fn washburn_height(
    sigma: Scalar,
    theta_deg: Scalar,
    radius: Scalar,
    mu: Scalar,
    t: Scalar,
) -> Scalar {
    let x = sigma * radius * cos_deg(theta_deg) * t / (2.0 * mu);
    if x <= 0.0 {
        0.0
    } else {
        x.sqrt()
    }
}

// ==========================================================================
//  The per-patch specification
// ==========================================================================

/// What one wall patch's `alpha` entry says about the contact angle -
/// SPEC-LIT §39.3, §39.4.
///
/// ```text
/// walls
/// {
///     type            constantAlphaContactAngle;
///     theta0          45;                 // degrees, through the LIQUID
///     value           uniform 0;
/// }
///
/// walls
/// {
///     type            dynamicAlphaContactAngle;
///     theta0          45;
///     correlation     JiangOhSlattery;    // or CoxVoinov
///     thetaA          60;                 // optional, default theta0
///     thetaR          30;                 // optional, default theta0
///     lnLRatio        9.2;                // CoxVoinov only: ln(L/L_m)
///     value           uniform 0;
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactAnglePatch {
    /// Young's angle, in degrees, measured THROUGH the liquid (phase 1).
    pub theta_e: Scalar,
    /// Advancing and receding angles, in degrees. Equal to `theta_e` unless
    /// the case asked for hysteresis.
    pub theta_a: Scalar,
    pub theta_r: Scalar,
    pub correlation: ContactAngleCorrelation,
    /// `ln(L/L_m)` - Cox-Voinov only.
    pub ln_l_ratio: Scalar,
}

impl Default for ContactAnglePatch {
    /// Ninety degrees, static: the no-wall-adhesion angle the pre-§39 solver
    /// applies to every wall, so the default of this struct IS the old
    /// behaviour.
    fn default() -> Self {
        Self {
            theta_e: 90.0,
            theta_a: 90.0,
            theta_r: 90.0,
            correlation: ContactAngleCorrelation::Static,
            ln_l_ratio: 0.0,
        }
    }
}

impl ContactAnglePatch {
    /// Read one `boundaryField` entry, under §13.4's contract.
    ///
    /// `dynamic` selects whether the extra entries are read at all: a
    /// `constantAlphaContactAngle` patch carrying a `thetaA` is a user who
    /// believes they configured hysteresis, so it is refused BY NAME rather
    /// than dropped.
    pub fn from_spec(spec: &PatchFieldSpec, patch: &str, dynamic: bool) -> Result<Self> {
        let type_name = if dynamic {
            "dynamicAlphaContactAngle"
        } else {
            "constantAlphaContactAngle"
        };

        let theta_e = spec.required_number("theta0", patch, type_name)?;
        let mut c = Self { theta_e, theta_a: theta_e, theta_r: theta_e, ..Self::default() };

        if !dynamic {
            for k in ["thetaA", "thetaR", "correlation", "lnLRatio"] {
                if spec.extra.contains_key(k) {
                    return unsupported(
                        &format!("alpha: boundaryField/{patch}/{k}"),
                        k,
                        &["theta0"],
                        "nothing - constantAlphaContactAngle reads theta0 alone; \
                         name dynamicAlphaContactAngle to use it",
                        (),
                    )
                    .map(|()| c);
                }
            }
            c.validate(patch)?;
            return Ok(c);
        }

        c.correlation = match spec.extra.get("correlation") {
            Some(raw) => {
                let tok = raw.split_whitespace().next().unwrap_or("");
                ContactAngleCorrelation::parse(
                    &format!("alpha: boundaryField/{patch}/correlation"),
                    tok,
                )?
            }
            None => {
                return Err(Error::Field {
                    field: patch.to_string(),
                    msg: format!(
                        "dynamicAlphaContactAngle needs a `correlation` entry \
                         naming one of: {} (SPEC-LIT 39.4)",
                        ContactAngleCorrelation::NAMES.join(", ")
                    ),
                })
            }
        };

        if let Some(v) = spec.number("thetaA", patch)? {
            c.theta_a = v;
        }
        if let Some(v) = spec.number("thetaR", patch)? {
            c.theta_r = v;
        }

        if c.correlation == ContactAngleCorrelation::CoxVoinov {
            c.ln_l_ratio = spec.required_number("lnLRatio", patch, "CoxVoinov")?;
        } else if spec.extra.contains_key("lnLRatio") {
            return unsupported(
                &format!("alpha: boundaryField/{patch}/lnLRatio"),
                "lnLRatio",
                &["correlation CoxVoinov"],
                "nothing - only CoxVoinov reads ln(L/L_m)",
                (),
            )
            .map(|()| c);
        }

        c.validate(patch)?;
        Ok(c)
    }

    fn validate(&self, patch: &str) -> Result<()> {
        let bad = |what: &str, v: Scalar, why: &str| Error::Field {
            field: patch.to_string(),
            msg: format!("contact angle: {what} = {v} - {why} (SPEC-LIT 39.6)"),
        };
        for (what, v) in [
            ("theta0", self.theta_e),
            ("thetaA", self.theta_a),
            ("thetaR", self.theta_r),
        ] {
            if !v.is_finite() || !(0.0..=180.0).contains(&v) {
                return Err(bad(what, v, "an angle is in [0, 180] degrees"));
            }
        }
        if self.theta_a < self.theta_r {
            return Err(bad(
                "thetaA",
                self.theta_a,
                "the advancing angle cannot be below the receding one",
            ));
        }
        if self.correlation == ContactAngleCorrelation::CoxVoinov && !(self.ln_l_ratio > 0.0) {
            return Err(bad(
                "lnLRatio",
                self.ln_l_ratio,
                "Cox-Voinov needs ln(L/L_m) > 0, i.e. a macroscopic length above \
                 the microscopic cut-off",
            ));
        }
        Ok(())
    }

    /// A one-line summary for the run banner.
    pub fn describe(&self, patch: &str) -> String {
        if self.correlation == ContactAngleCorrelation::Static
            && self.theta_a == self.theta_e
            && self.theta_r == self.theta_e
        {
            return format!("{patch}: theta {} deg (static)", self.theta_e);
        }
        format!(
            "{patch}: theta {} deg, advancing {} receding {}, {}{}",
            self.theta_e,
            self.theta_a,
            self.theta_r,
            self.correlation.name(),
            if self.correlation == ContactAngleCorrelation::CoxVoinov {
                format!(" ln(L/Lm) {}", self.ln_l_ratio)
            } else {
                String::new()
            }
        )
    }
}

/// The per-boundary-face arrays the kernels read - SPEC-LIT §39.
///
/// Struct of arrays rather than array of structs: every one of these is read
/// by one thread per face with a unit stride, and the `owns` flag is what
/// makes a face the model does NOT own keep the pre-§39 behaviour exactly.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ContactAngleFaces {
    /// `1` where a contact-angle condition owns the face.
    pub owns: Vec<Label>,
    pub cos_e: Vec<Scalar>,
    pub cos_a: Vec<Scalar>,
    pub cos_r: Vec<Scalar>,
    pub correlation: Vec<Label>,
    pub ln_l_ratio: Vec<Scalar>,
}

impl ContactAngleFaces {
    /// Every face un-owned, i.e. exactly the pre-§39 solver.
    pub fn none(nbf: usize) -> Self {
        Self {
            owns: vec![0; nbf],
            cos_e: vec![0.0; nbf],
            cos_a: vec![0.0; nbf],
            cos_r: vec![0.0; nbf],
            correlation: vec![0; nbf],
            ln_l_ratio: vec![0.0; nbf],
        }
    }

    /// Give `[start, start + n)` this patch's angles.
    pub fn set_patch(&mut self, start: usize, n: usize, p: &ContactAnglePatch) {
        for i in start..(start + n).min(self.owns.len()) {
            self.owns[i] = 1;
            self.cos_e[i] = cos_deg(p.theta_e);
            self.cos_a[i] = cos_deg(p.theta_a);
            self.cos_r[i] = cos_deg(p.theta_r);
            self.correlation[i] = p.correlation as Label;
            self.ln_l_ratio[i] = p.ln_l_ratio;
        }
    }

    /// True when no face is owned, in which case the whole model is switched
    /// off and `vofFaceUnitNormalBoundary` keeps writing its literal zero.
    pub fn is_empty(&self) -> bool {
        self.owns.iter().all(|o| *o == 0)
    }

    pub fn n_owned(&self) -> usize {
        self.owns.iter().filter(|o| **o != 0).count()
    }

    /// Build the per-face arrays from the `alpha` field as it sits on disk -
    /// SPEC-LIT §39.3.
    ///
    /// The angle is a property of the WALL, and the wall is named in the
    /// field file's `boundaryField`, so that is where the entry belongs and
    /// this is the only reader of it. Returns the per-face arrays and one
    /// banner line per owning patch.
    ///
    /// A patch whose `type` is not one of the two contact-angle spellings is
    /// left un-owned, which is the pre-§39 behaviour to the bit. `type` is
    /// read verbatim here rather than through [`crate::field::BcKind`]
    /// because the two spellings map to ONE `BcKind` and this reader is
    /// exactly the thing that has to tell them apart.
    pub fn from_alpha_field(
        raw: &crate::io::fields::RawScalarField,
        m: &crate::mesh::HostMesh,
    ) -> Result<(Self, Vec<String>)> {
        let mut f = Self::none(m.n_boundary_faces);
        let mut banner = Vec::new();

        for p in &m.patches {
            let Some(key) = crate::io::fields::governing_key(
                &raw.boundary,
                &raw.boundary_patterns,
                &p.name,
            )?
            else {
                continue;
            };
            let Some(spec) = raw.boundary.get(&key) else { continue };

            let dynamic = match spec.type_name.as_str() {
                "constantAlphaContactAngle" => false,
                "dynamicAlphaContactAngle" => true,
                _ => continue,
            };

            let patch = ContactAnglePatch::from_spec(spec, &p.name, dynamic)?;
            f.set_patch(p.start, p.size, &patch);
            banner.push(patch.describe(&p.name));
        }

        Ok((f, banner))
    }
}

// ==========================================================================
//  Kernels
// ==========================================================================

/// The two device kernels §39 adds to `cuda/vof.cu`.
pub struct ContactAngleKernels {
    cos_theta: CudaFunction,
    alpha_grad: CudaFunction,
}

impl ContactAngleKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let v = KernelSet::new(gpu, crate::kernels::VOF)?;
        Ok(Self {
            cos_theta: v.func("vofContactAngleCos")?,
            alpha_grad: v.func("vofAlphaContactAngleGrad")?,
        })
    }
}

/// The per-face device state, uploaded once.
pub struct ContactAngleDevice {
    pub kern: ContactAngleKernels,
    pub owns: DevBuf<Label>,
    pub cos_e: DevBuf<Scalar>,
    pub cos_a: DevBuf<Scalar>,
    pub cos_r: DevBuf<Scalar>,
    pub correlation: DevBuf<Label>,
    pub ln_l_ratio: DevBuf<Scalar>,
    /// `cos(theta)` as of the last update, and `1` where the model actually
    /// applied (an interface is present at the face).
    pub cos_theta: DevBuf<Scalar>,
    pub applies: DevBuf<Label>,
    /// For the banner, and for a test that wants to know the model is on.
    pub n_owned: usize,
}

impl ContactAngleDevice {
    pub fn upload(gpu: &Gpu, f: &ContactAngleFaces) -> Result<Self> {
        let n = f.owns.len().max(1);
        let pad = |v: &[Scalar]| -> Vec<Scalar> {
            let mut w = v.to_vec();
            w.resize(n, 0.0);
            w
        };
        let padl = |v: &[Label]| -> Vec<Label> {
            let mut w = v.to_vec();
            w.resize(n, 0);
            w
        };
        Ok(Self {
            kern: ContactAngleKernels::new(gpu)?,
            owns: gpu.upload(&padl(&f.owns))?,
            cos_e: gpu.upload(&pad(&f.cos_e))?,
            cos_a: gpu.upload(&pad(&f.cos_a))?,
            cos_r: gpu.upload(&pad(&f.cos_r))?,
            correlation: gpu.upload(&padl(&f.correlation))?,
            ln_l_ratio: gpu.upload(&pad(&f.ln_l_ratio))?,
            cos_theta: gpu.zeros(n)?,
            applies: gpu.zeros(n)?,
            n_owned: f.n_owned(),
        })
    }
}

/// *DESIGN*, SPEC-LIT §39.4: a wall face carries an interface only where
/// `eps < alpha_b < 1 - eps`. A dry or fully wet face has no interface to
/// orient, and there the pre-§39 `bNHatf = 0` is not a fallback - it is the
/// right answer.
pub const ALPHA_INTERFACE_EPS: Scalar = 1e-3;

/// `cos(theta)` per boundary face, and the flag saying where the model
/// applies - SPEC-LIT §39.4, §39.5.
///
/// Must run AFTER `grad(alpha)` and BEFORE the curvature gather: §20.4's own
/// discipline, "a stale normal is a wrong curvature", extended one step.
#[allow(clippy::too_many_arguments)]
pub fn update_cos_theta(
    gpu: &Gpu,
    d: &ContactAngleDevice,
    alpha_b: &DevBuf<Scalar>,
    grad_alpha: &DevBuf<crate::Vec3>,
    u: &DevBuf<crate::Vec3>,
    bu: &DevBuf<crate::Vec3>,
    mu_liquid: Scalar,
    sigma: Scalar,
    m: &crate::mesh::GpuMesh,
) -> Result<()> {
    let nbf = m.n_boundary_faces;
    if nbf == 0 {
        return Ok(());
    }
    let nl = nbf as Label;
    let eps = ALPHA_INTERFACE_EPS;
    let f = d.kern.cos_theta.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(&d.cos_theta)
            .arg(&d.applies)
            .arg(&d.owns)
            .arg(&d.cos_e)
            .arg(&d.cos_a)
            .arg(&d.cos_r)
            .arg(&d.correlation)
            .arg(&d.ln_l_ratio)
            .arg(alpha_b)
            .arg(grad_alpha)
            .arg(u)
            .arg(bu)
            .arg(&m.b_sf)
            .arg(&m.b_mag_sf)
            .arg(&m.b_face_cells)
            .arg(&mu_liquid)
            .arg(&sigma)
            .arg(&eps)
            .arg(&nl)
            .launch(cfg_for(nbf))?;
    }
    Ok(())
}

/// `refGrad = |grad(alpha)_P| cos(theta)` on the faces the model owns -
/// SPEC-LIT §39.3.
///
/// A plain fixed-gradient condition in §4's triple, rewritten every outer
/// iteration exactly as §32.2's fixed wall heat flux rewrites its own. Faces
/// the model owns but where no interface is present get `refGrad = 0`, i.e.
/// zero-gradient, i.e. the pre-§39 condition.
pub fn update_alpha_ref_grad(
    gpu: &Gpu,
    d: &ContactAngleDevice,
    ref_grad: &mut DevBuf<Scalar>,
    grad_alpha: &DevBuf<crate::Vec3>,
    m: &crate::mesh::GpuMesh,
) -> Result<()> {
    let nbf = m.n_boundary_faces;
    if nbf == 0 {
        return Ok(());
    }
    let nl = nbf as Label;
    let f = d.kern.alpha_grad.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(ref_grad)
            .arg(&d.cos_theta)
            .arg(&d.applies)
            .arg(&d.owns)
            .arg(grad_alpha)
            .arg(&m.b_face_cells)
            .arg(&nl)
            .launch(cfg_for(nbf))?;
    }
    Ok(())
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// **The premise of the whole `enabled` flag, MEASURED.**
    ///
    /// If `cos(pi/2)` were bitwise zero, `bNHatf = |Sf| cos(theta)` could be
    /// written unconditionally and every recorded VOF result would be safe.
    /// It is not, and this test is what says so - so that the guard in
    /// `vofFaceUnitNormalBoundary` and the special case in [`cos_deg`] are
    /// justified by a number rather than by a comment.
    #[test]
    fn the_cosine_of_ninety_degrees_is_not_zero() {
        let raw = ((90.0 as Scalar) * (std::f64::consts::PI as Scalar) / 180.0).cos();
        assert_ne!(
            raw, 0.0,
            "cos(pi/2) came out exactly zero, which would make SPEC-LIT 39.2's \
             enabled flag unnecessary - check the platform before deleting it"
        );
        assert!(raw.abs() < 1e-15, "cos(pi/2) = {raw}, which is not small either");

        // And the host mapping fixes it.
        assert_eq!(cos_deg(90.0), 0.0);
        assert_eq!(cos_deg(90.0).to_bits(), (0.0 as Scalar).to_bits());
    }

    /// SPEC-LIT §39.2's three hand-derived values, plus the two that decide
    /// the sign of the whole model.
    #[test]
    fn the_geometry_is_the_hand_derived_cosine() {
        assert_eq!(cos_deg(0.0), 1.0);
        assert_eq!(cos_deg(180.0), -1.0);
        assert_eq!(cos_deg(90.0), 0.0);

        for (deg, want) in [(45.0 as Scalar, 0.5 as Scalar), (135.0, -0.5)] {
            let got = cos_deg(deg);
            let want = if want > 0.0 { (2.0 as Scalar).sqrt() / 2.0 } else { -(2.0 as Scalar).sqrt() / 2.0 };
            assert!((got - want).abs() < 1e-12, "cos({deg}) = {got}, not {want}");
        }

        // Wetting is positive, non-wetting negative - the sign that decides
        // whether a drop beads or spreads.
        assert!(cos_deg(30.0) > 0.0);
        assert!(cos_deg(150.0) < 0.0);

        // And the round trip.
        for deg in [0.0 as Scalar, 30.0, 45.0, 90.0, 120.0, 180.0] {
            assert!((acos_deg(cos_deg(deg)) - deg).abs() < 1e-10);
        }
    }

    /// SPEC-LIT §39.6. Every correlation returns the static angle EXACTLY at
    /// zero contact-line speed - not to a tolerance, bitwise - so that a
    /// dynamic case with a stationary line is the static case.
    #[test]
    fn every_correlation_reduces_to_the_static_angle_at_zero_speed() {
        for deg in [0.0 as Scalar, 30.0, 90.0, 135.0, 180.0] {
            let c = cos_deg(deg);
            for corr in [
                ContactAngleCorrelation::Static,
                ContactAngleCorrelation::JiangOhSlattery,
                ContactAngleCorrelation::CoxVoinov,
            ] {
                let got = cos_theta_dynamic(corr, c, c, c, 0.0, 9.2);
                assert_eq!(
                    got.to_bits(),
                    c.to_bits(),
                    "{} at Ca = 0 and theta = {deg} gave {got}, not {c}",
                    corr.name()
                );
            }
        }
    }

    /// And hysteresis with all three angles equal is the static model, at
    /// every speed, for the `Static` correlation.
    #[test]
    fn hysteresis_with_one_angle_is_the_static_model() {
        let c = cos_deg(45.0);
        for ca in [-10.0 as Scalar, -1e-6, 0.0, 1e-6, 10.0] {
            let got = cos_theta_dynamic(ContactAngleCorrelation::Static, c, c, c, ca, 0.0);
            assert_eq!(got.to_bits(), c.to_bits(), "Ca = {ca} moved the static angle");
        }
    }

    /// Hysteresis, when it IS configured, picks the branch the sign of `Ca`
    /// names - SPEC-LIT §39.4.
    #[test]
    fn hysteresis_picks_the_advancing_and_receding_branches() {
        let (e, a, r) = (cos_deg(45.0), cos_deg(60.0), cos_deg(30.0));
        let s = ContactAngleCorrelation::Static;
        assert_eq!(cos_theta_dynamic(s, e, a, r, 1e-9, 0.0), a);
        assert_eq!(cos_theta_dynamic(s, e, a, r, -1e-9, 0.0), r);
        assert_eq!(cos_theta_dynamic(s, e, a, r, 0.0, 0.0), e);
    }

    /// SPEC-LIT §39.6: `theta_d` rises with `Ca` when advancing, falls when
    /// receding, and never leaves `[0, 180]`.
    #[test]
    fn the_dynamic_angle_is_monotone_and_bounded() {
        for corr in [
            ContactAngleCorrelation::JiangOhSlattery,
            ContactAngleCorrelation::CoxVoinov,
        ] {
            let c = cos_deg(45.0);
            let mut prev = acos_deg(cos_theta_dynamic(corr, c, c, c, 0.0, 9.2));
            let mut ca: Scalar = 1e-6;
            while ca < 10.0 {
                let th = acos_deg(cos_theta_dynamic(corr, c, c, c, ca, 9.2));
                assert!(
                    th >= prev - 1e-12,
                    "{}: theta fell from {prev} to {th} as Ca rose to {ca}",
                    corr.name()
                );
                assert!((0.0..=180.0).contains(&th), "{}: theta = {th}", corr.name());
                prev = th;
                ca *= 2.0;
            }

            // Receding.
            let mut prev = acos_deg(cos_theta_dynamic(corr, c, c, c, 0.0, 9.2));
            let mut ca: Scalar = -1e-6;
            while ca > -10.0 {
                let th = acos_deg(cos_theta_dynamic(corr, c, c, c, ca, 9.2));
                assert!(
                    th <= prev + 1e-12,
                    "{}: theta rose from {prev} to {th} as Ca fell to {ca}",
                    corr.name()
                );
                assert!((0.0..=180.0).contains(&th), "{}: theta = {th}", corr.name());
                prev = th;
                ca *= 2.0;
            }
        }
    }

    /// Jiang, Oh & Slattery's stated limits: `Ca -> inf` gives complete
    /// dewetting of the displaced phase, `theta_d -> 180 deg`.
    #[test]
    fn jiang_reaches_complete_dewetting_at_large_capillary_number() {
        let c = cos_deg(45.0);
        let j = ContactAngleCorrelation::JiangOhSlattery;
        let th = acos_deg(cos_theta_dynamic(j, c, c, c, 100.0, 0.0));
        assert!((th - 180.0).abs() < 1e-6, "Ca = 100 gave {th} deg, not 180");
        // And the correlation as PUBLISHED, rearranged back:
        // (cos_e - cos_d)/(cos_e + 1) = tanh(4.96 Ca^0.702).
        for ca in [1e-4 as Scalar, 1e-3, 1e-2, 1e-1] {
            let cd = cos_theta_dynamic(j, c, c, c, ca, 0.0);
            let lhs = (c - cd) / (c + 1.0);
            let rhs = (JIANG_A * ca.powf(JIANG_B)).tanh();
            assert!(
                (lhs - rhs).abs() < 1e-12,
                "at Ca = {ca} the rearrangement gives {lhs}, the correlation {rhs}"
            );
        }
    }

    /// Cox-Voinov, as PUBLISHED: `theta_d^3 - theta_e^3 = 9 Ca ln(L/L_m)`,
    /// wherever the clip is not biting.
    #[test]
    fn cox_voinov_is_the_published_cubic() {
        let deg: Scalar = 45.0;
        let th_e = deg * (std::f64::consts::PI as Scalar) / 180.0;
        let c = cos_deg(deg);
        let ln_ratio: Scalar = 9.2;
        for ca in [-1e-3 as Scalar, -1e-4, 1e-4, 1e-3, 1e-2] {
            let cd = cos_theta_dynamic(ContactAngleCorrelation::CoxVoinov, c, c, c, ca, ln_ratio);
            let th_d = cd.acos();
            let lhs = th_d * th_d * th_d - th_e * th_e * th_e;
            let rhs = 9.0 * ca * ln_ratio;
            assert!(
                (lhs - rhs).abs() < 1e-10,
                "at Ca = {ca}: {lhs} against 9 Ca ln(L/Lm) = {rhs}"
            );
        }
    }

    /// No NaN anywhere it can be called.
    #[test]
    fn the_dynamic_angle_is_finite_everywhere() {
        let c = cos_deg(45.0);
        for corr in [
            ContactAngleCorrelation::Static,
            ContactAngleCorrelation::JiangOhSlattery,
            ContactAngleCorrelation::CoxVoinov,
        ] {
            for ca in [
                0.0 as Scalar,
                Scalar::MIN_POSITIVE,
                -Scalar::MIN_POSITIVE,
                1e-300,
                -1e-300,
                1e300,
                -1e300,
                Scalar::INFINITY,
                Scalar::NEG_INFINITY,
                Scalar::NAN,
            ] {
                let v = cos_theta_dynamic(corr, c, c, c, ca, 9.2);
                assert!(
                    v.is_finite() && (-1.0..=1.0).contains(&v),
                    "{} at Ca = {ca} gave {v}",
                    corr.name()
                );
            }
        }
    }

    /// SPEC-LIT §39.7 Gate 1. Jurin's height, with the sign checked at both
    /// ends: a non-wetting liquid is DEPRESSED, and ninety degrees does not
    /// move at all.
    #[test]
    fn jurin_height_has_the_right_sign_at_both_ends() {
        // Water against air at 20 C in a 0.5 mm capillary.
        let (sigma, rho, g, r): (Scalar, Scalar, Scalar, Scalar) =
            (0.0728, 998.2, 9.81, 5e-4);

        let mut prev = Scalar::INFINITY;
        for deg in [0.0 as Scalar, 30.0, 60.0, 90.0, 120.0, 150.0] {
            let h = jurin_height(sigma, deg, rho, g, r);
            assert!(h < prev, "the rise did not fall as theta went to {deg}");
            prev = h;
            match deg {
                d if d < 90.0 => assert!(h > 0.0, "theta = {deg} must RISE, got {h}"),
                d if d > 90.0 => assert!(h < 0.0, "theta = {deg} must be DEPRESSED, got {h}"),
                _ => assert_eq!(h, 0.0, "theta = 90 must give exactly zero rise"),
            }
        }

        // The classic number: water fully wetting a 0.5 mm-radius tube rises
        // 2 sigma/(rho g R) = about 30 mm.
        let h0 = jurin_height(sigma, 0.0, rho, g, r);
        let want = 2.0 * sigma / (rho * g * r);
        assert!((h0 - want).abs() <= 1e-14 * want);
        assert!(
            (h0 - 0.0297).abs() < 5e-4,
            "Jurin's height for water in a 0.5 mm tube is about 29.7 mm, got {} m",
            h0
        );

        // Mercury, non-wetting: depressed.
        assert!(jurin_height(0.487, 140.0, 13534.0, 9.81, 5e-4) < 0.0);
    }

    /// Lucas-Washburn: the same statement in time, including the zero at
    /// ninety degrees and the no-rise-at-all above it.
    #[test]
    fn the_washburn_rise_is_zero_at_ninety_degrees_and_above() {
        let (sigma, r, mu): (Scalar, Scalar, Scalar) = (0.0728, 5e-4, 1.002e-3);
        assert_eq!(washburn_height(sigma, 90.0, r, mu, 1.0), 0.0);
        assert_eq!(washburn_height(sigma, 120.0, r, mu, 1.0), 0.0);
        assert!(washburn_height(sigma, 0.0, r, mu, 1.0) > 0.0);
        // h ~ sqrt(t): four times the time is twice the height.
        let h1 = washburn_height(sigma, 30.0, r, mu, 1.0);
        let h4 = washburn_height(sigma, 30.0, r, mu, 4.0);
        assert!((h4 - 2.0 * h1).abs() <= 1e-12 * h1);
    }

    /// SPEC-LIT §39.6's §13.4 row.
    #[test]
    fn an_unrecognised_correlation_is_a_13_4_error_naming_the_alternatives() {
        assert_eq!(
            ContactAngleCorrelation::parse("x", "JiangOhSlattery").unwrap(),
            ContactAngleCorrelation::JiangOhSlattery
        );
        let e = ContactAngleCorrelation::parse(
            "alpha: boundaryField/walls/correlation",
            "HoffmanKistler",
        )
        .expect_err("Kistler is deliberately not implemented");
        let msg = e.to_string();
        for want in ContactAngleCorrelation::NAMES {
            assert!(msg.contains(want), "the message does not name {want}: {msg}");
        }
    }

    fn spec(type_name: &str, pairs: &[(&str, &str)]) -> PatchFieldSpec {
        let mut s = PatchFieldSpec { type_name: type_name.to_string(), ..Default::default() };
        for (k, v) in pairs {
            s.extra.insert((*k).to_string(), (*v).to_string());
        }
        s
    }

    #[test]
    fn a_static_patch_reads_theta0_and_nothing_else() {
        let p = ContactAnglePatch::from_spec(
            &spec("constantAlphaContactAngle", &[("theta0", "45")]),
            "walls",
            false,
        )
        .expect("reads");
        assert_eq!(p.theta_e, 45.0);
        assert_eq!(p.theta_a, 45.0);
        assert_eq!(p.theta_r, 45.0);
        assert_eq!(p.correlation, ContactAngleCorrelation::Static);
        assert!(p.describe("walls").contains("45"));
    }

    /// A missing `theta0` is refused BY NAME - the condition is defined by it.
    #[test]
    fn a_missing_theta0_is_refused_by_name() {
        let e = ContactAnglePatch::from_spec(
            &spec("constantAlphaContactAngle", &[]),
            "walls",
            false,
        )
        .expect_err("theta0 is not optional");
        assert!(e.to_string().contains("theta0"), "{e}");
    }

    /// SPEC-LIT §13.4: a `thetaA` on a STATIC patch is a user who believes
    /// they configured hysteresis, so it is refused rather than dropped.
    #[test]
    fn hysteresis_on_a_static_patch_is_refused() {
        crate::io::contract::reset_warnings();
        let e = ContactAnglePatch::from_spec(
            &spec("constantAlphaContactAngle", &[("theta0", "45"), ("thetaA", "60")]),
            "walls",
            false,
        )
        .expect_err("constantAlphaContactAngle does not read thetaA");
        let msg = e.to_string();
        assert!(msg.contains("thetaA"), "{msg}");
        assert!(msg.contains("dynamicAlphaContactAngle"), "the way out: {msg}");
    }

    #[test]
    fn a_dynamic_patch_reads_its_correlation_and_hysteresis() {
        let p = ContactAnglePatch::from_spec(
            &spec(
                "dynamicAlphaContactAngle",
                &[
                    ("theta0", "45"),
                    ("correlation", "JiangOhSlattery"),
                    ("thetaA", "60"),
                    ("thetaR", "30"),
                ],
            ),
            "walls",
            true,
        )
        .expect("reads");
        assert_eq!(p.correlation, ContactAngleCorrelation::JiangOhSlattery);
        assert_eq!(p.theta_a, 60.0);
        assert_eq!(p.theta_r, 30.0);
        let d = p.describe("walls");
        assert!(d.contains("JiangOhSlattery") && d.contains("60") && d.contains("30"), "{d}");
    }

    /// A dynamic patch with no `correlation` is refused: there is no default
    /// correlation, because picking one would be picking the physics.
    #[test]
    fn a_dynamic_patch_without_a_correlation_is_refused() {
        let e = ContactAnglePatch::from_spec(
            &spec("dynamicAlphaContactAngle", &[("theta0", "45")]),
            "walls",
            true,
        )
        .expect_err("a dynamic patch needs a correlation");
        let msg = e.to_string();
        assert!(msg.contains("correlation"), "{msg}");
        for want in ContactAngleCorrelation::NAMES {
            assert!(msg.contains(want), "{msg}");
        }
    }

    /// Cox-Voinov needs `ln(L/L_m)` and refuses it from anyone else.
    #[test]
    fn cox_voinov_needs_its_length_ratio_and_no_one_else_may_have_one() {
        crate::io::contract::reset_warnings();
        let e = ContactAnglePatch::from_spec(
            &spec(
                "dynamicAlphaContactAngle",
                &[("theta0", "45"), ("correlation", "CoxVoinov")],
            ),
            "walls",
            true,
        )
        .expect_err("CoxVoinov needs lnLRatio");
        assert!(e.to_string().contains("lnLRatio"), "{e}");

        let ok = ContactAnglePatch::from_spec(
            &spec(
                "dynamicAlphaContactAngle",
                &[("theta0", "45"), ("correlation", "CoxVoinov"), ("lnLRatio", "9.2")],
            ),
            "walls",
            true,
        )
        .expect("reads");
        assert_eq!(ok.ln_l_ratio, 9.2);

        let e = ContactAnglePatch::from_spec(
            &spec(
                "dynamicAlphaContactAngle",
                &[
                    ("theta0", "45"),
                    ("correlation", "JiangOhSlattery"),
                    ("lnLRatio", "9.2"),
                ],
            ),
            "walls",
            true,
        )
        .expect_err("only CoxVoinov reads lnLRatio");
        assert!(e.to_string().contains("lnLRatio"), "{e}");
    }

    /// SPEC-LIT §39.6: an angle outside `[0, 180]`, and an advancing angle
    /// below the receding one, are both errors at READ time.
    #[test]
    fn the_angle_ranges_are_checked_at_read_time() {
        for bad in ["-5", "181", "nan"] {
            let r = ContactAnglePatch::from_spec(
                &spec("constantAlphaContactAngle", &[("theta0", bad)]),
                "walls",
                false,
            );
            assert!(r.is_err(), "theta0 = {bad} was accepted");
        }
        let e = ContactAnglePatch::from_spec(
            &spec(
                "dynamicAlphaContactAngle",
                &[
                    ("theta0", "45"),
                    ("correlation", "static"),
                    ("thetaA", "30"),
                    ("thetaR", "60"),
                ],
            ),
            "walls",
            true,
        )
        .expect_err("thetaA below thetaR is not a hysteresis loop");
        assert!(e.to_string().contains("thetaA"), "{e}");
    }

    /// The per-face expansion, and the "no faces owned" state that IS the
    /// pre-§39 solver.
    #[test]
    fn the_face_arrays_start_empty_and_fill_by_patch() {
        let mut f = ContactAngleFaces::none(10);
        assert!(f.is_empty());
        assert_eq!(f.n_owned(), 0);

        let p = ContactAnglePatch { theta_e: 45.0, theta_a: 45.0, theta_r: 45.0, ..Default::default() };
        f.set_patch(2, 3, &p);
        assert!(!f.is_empty());
        assert_eq!(f.n_owned(), 3);
        assert_eq!(f.owns, vec![0, 0, 1, 1, 1, 0, 0, 0, 0, 0]);
        assert_eq!(f.cos_e[2], cos_deg(45.0));
        assert_eq!(f.cos_e[0], 0.0);

        // Ninety degrees writes an EXACT zero, so a case that names it is
        // bit-for-bit the case that names nothing.
        let mut g = ContactAngleFaces::none(4);
        g.set_patch(0, 4, &ContactAnglePatch::default());
        for c in &g.cos_e {
            assert_eq!(c.to_bits(), (0.0 as Scalar).to_bits());
        }
    }

    // ----------------------------------------------------------------------
    //  The device - SPEC-LIT §39.6
    // ----------------------------------------------------------------------

    use crate::blockgen::{build_mesh, BlockSpec, GradedAxis};
    use crate::mesh::GpuMesh;
    use crate::Vec3;

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// `cuda/vof.cu` hard-codes these three codes. If the enum moves, the
    /// kernel silently evaluates a different correlation - the same failure
    /// `bc_kind_values_match_the_device` exists to stop for `BcKind`.
    #[test]
    fn correlation_codes_match_the_device() {
        assert_eq!(ContactAngleCorrelation::Static as Label, 0);
        assert_eq!(ContactAngleCorrelation::JiangOhSlattery as Label, 1);
        assert_eq!(ContactAngleCorrelation::CoxVoinov as Label, 2);
    }

    /// A 4 x 8 x 1 box, walls on `-y`/`+y`.
    fn block() -> (crate::mesh::HostMesh, BlockSpec) {
        let spec = BlockSpec {
            x: GradedAxis { lo: 0.0, hi: 0.08, n: 4, ..GradedAxis::default() },
            y: GradedAxis { lo: 0.0, hi: 0.04, n: 8, ..GradedAxis::default() },
            z: GradedAxis { lo: 0.0, hi: 0.04, n: 1, ..GradedAxis::default() },
            ..BlockSpec::default()
        };
        let hm = build_mesh(&spec).expect("build the block");
        (hm, spec)
    }

    /// SPEC-LIT §39.2, on the device: `bNHatf = |Sf| cos(theta)` on the faces
    /// the model owns, and a LITERAL zero everywhere else - including on
    /// every face when the model is off.
    ///
    /// This is the kernel the `cos(pi/2)` trap lives in, so both branches are
    /// exercised on the same mesh in the same test.
    #[test]
    fn the_boundary_normal_is_the_magnitude_times_the_cosine() {
        let Some(g) = gpu() else { return };
        let (hm, _) = block();
        let m = GpuMesh::upload(&g, &hm).expect("upload");
        let nbf = hm.n_boundary_faces;
        let nl = nbf as Label;

        let k = KernelSet::new(&g, crate::kernels::VOF).expect("the vof module");
        let f = k.func("vofFaceUnitNormalBoundary").expect("vofFaceUnitNormalBoundary");

        // grad(alpha) is read only by the CYCLIC branch, which this block has
        // none of; fill it with something that would be obvious if it leaked.
        let grad = g.upload(&vec![Vec3::new(7.0, -3.0, 11.0); hm.n_cells]).expect("upload");
        let eps: Scalar = 1e-6;

        for deg in [0.0 as Scalar, 45.0, 90.0, 135.0, 180.0] {
            let cosv = cos_deg(deg);
            let d_cos = g.upload(&vec![cosv; nbf]).expect("upload");
            let d_applies = g.upload(&vec![1 as Label; nbf]).expect("upload");
            let mut out = g.zeros::<Scalar>(nbf).expect("alloc");

            for enabled in [0 as Label, 1] {
                unsafe {
                    g.stream()
                        .launch_builder(&f)
                        .arg(&mut out)
                        .arg(&grad)
                        .arg(&m.b_weights)
                        .arg(&m.b_sf)
                        .arg(&m.b_face_cells)
                        .arg(&m.b_nbr_cell)
                        .arg(&m.b_kind)
                        .arg(&d_cos)
                        .arg(&d_applies)
                        .arg(&m.b_mag_sf)
                        .arg(&enabled)
                        .arg(&eps)
                        .arg(&nl)
                        .launch(cfg_for(nbf))
                        .expect("launch");
                }
                g.sync().expect("sync");
                let got = g.download(&out).expect("download");

                for i in 0..nbf {
                    let want = if enabled == 0 { 0.0 } else { hm.b_mag_sf[i] * cosv };
                    assert_eq!(
                        got[i].to_bits(),
                        want.to_bits(),
                        "theta = {deg}, enabled = {enabled}, face {i}: {} against {want}",
                        got[i]
                    );
                }
            }

            // The trap, on the device: ninety degrees must give the SAME BITS
            // as the model switched off, not something 1e-17 |Sf| away.
            if deg == 90.0 {
                let mut a = g.zeros::<Scalar>(nbf).expect("alloc");
                let mut b = g.zeros::<Scalar>(nbf).expect("alloc");
                let off: Label = 0;
                let on: Label = 1;
                unsafe {
                    g.stream()
                        .launch_builder(&f)
                        .arg(&mut a)
                        .arg(&grad)
                        .arg(&m.b_weights)
                        .arg(&m.b_sf)
                        .arg(&m.b_face_cells)
                        .arg(&m.b_nbr_cell)
                        .arg(&m.b_kind)
                        .arg(&d_cos)
                        .arg(&d_applies)
                        .arg(&m.b_mag_sf)
                        .arg(&off)
                        .arg(&eps)
                        .arg(&nl)
                        .launch(cfg_for(nbf))
                        .expect("launch");
                    g.stream()
                        .launch_builder(&f)
                        .arg(&mut b)
                        .arg(&grad)
                        .arg(&m.b_weights)
                        .arg(&m.b_sf)
                        .arg(&m.b_face_cells)
                        .arg(&m.b_nbr_cell)
                        .arg(&m.b_kind)
                        .arg(&d_cos)
                        .arg(&d_applies)
                        .arg(&m.b_mag_sf)
                        .arg(&on)
                        .arg(&eps)
                        .arg(&nl)
                        .launch(cfg_for(nbf))
                        .expect("launch");
                }
                g.sync().expect("sync");
                let (ga, gb) = (
                    g.download(&a).expect("download"),
                    g.download(&b).expect("download"),
                );
                for i in 0..nbf {
                    assert_eq!(
                        ga[i].to_bits(),
                        gb[i].to_bits(),
                        "face {i}: theta = 90 gave {} with the model on and {} with \
                         it off - SPEC-LIT 39.2's cos(pi/2) trap",
                        gb[i],
                        ga[i]
                    );
                }
            }
        }
    }

    /// Build the device state for a block whose `-y`/`+y` walls carry a
    /// contact angle, with a prescribed `grad(alpha)` and velocity, and
    /// return `cos(theta)` per face.
    #[allow(clippy::too_many_arguments)]
    fn run_cos_theta(
        g: &Gpu,
        hm: &crate::mesh::HostMesh,
        m: &GpuMesh,
        patch: &ContactAnglePatch,
        alpha_b: Scalar,
        grad_alpha: Vec3,
        u: Vec3,
        mu: Scalar,
        sigma: Scalar,
    ) -> Vec<Scalar> {
        let nbf = hm.n_boundary_faces;
        let mut faces = ContactAngleFaces::none(nbf);
        for p in &hm.patches {
            if p.kind == crate::mesh::PatchKind::Wall {
                faces.set_patch(p.start, p.size, patch);
            }
        }
        let d = ContactAngleDevice::upload(g, &faces).expect("upload");

        let d_alpha_b = g.upload(&vec![alpha_b; nbf]).expect("upload");
        let d_grad = g.upload(&vec![grad_alpha; hm.n_cells]).expect("upload");
        let d_u = g.upload(&vec![u; hm.n_cells]).expect("upload");
        let d_bu = g.upload(&vec![Vec3::ZERO; nbf]).expect("upload");

        update_cos_theta(g, &d, &d_alpha_b, &d_grad, &d_u, &d_bu, mu, sigma, m)
            .expect("launch");
        g.sync().expect("sync");
        g.download(&d.cos_theta).expect("download")
    }

    /// SPEC-LIT §39.4 on the device, static branch: with the interface normal
    /// perpendicular to the wall there is no wall-parallel component of
    /// `grad(alpha)`, `Ca` is zero, and every owned face gets exactly
    /// `cos(theta0)`.
    #[test]
    fn the_static_angle_reaches_every_owned_wall_face() {
        let Some(g) = gpu() else { return };
        let (hm, _) = block();
        let m = GpuMesh::upload(&g, &hm).expect("upload");

        let p = ContactAnglePatch { theta_e: 45.0, theta_a: 45.0, theta_r: 45.0, ..Default::default() };
        let got = run_cos_theta(
            &g,
            &hm,
            &m,
            &p,
            0.5,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            1.002e-3,
            0.0728,
        );

        let mut walls = 0usize;
        for (i, pinfo) in hm.b_patch.iter().enumerate() {
            let is_wall = hm.patches[*pinfo as usize].kind == crate::mesh::PatchKind::Wall;
            if is_wall {
                walls += 1;
                assert_eq!(
                    got[i].to_bits(),
                    cos_deg(45.0).to_bits(),
                    "wall face {i} got {} not cos(45)",
                    got[i]
                );
            } else {
                assert_eq!(got[i], 0.0, "face {i} is not a wall and must be untouched");
            }
        }
        assert!(walls > 0, "the block has no wall faces");
    }

    /// A dry or fully wet wall face has no interface to orient and keeps the
    /// pre-§39 zero - SPEC-LIT §39.4's detection predicate.
    #[test]
    fn a_dry_or_fully_wet_face_is_left_alone() {
        let Some(g) = gpu() else { return };
        let (hm, _) = block();
        let m = GpuMesh::upload(&g, &hm).expect("upload");
        let p = ContactAnglePatch { theta_e: 45.0, theta_a: 45.0, theta_r: 45.0, ..Default::default() };

        for alpha_b in [0.0 as Scalar, 1.0, ALPHA_INTERFACE_EPS / 2.0] {
            let got = run_cos_theta(
                &g,
                &hm,
                &m,
                &p,
                alpha_b,
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::ZERO,
                1.002e-3,
                0.0728,
            );
            for (i, v) in got.iter().enumerate() {
                assert_eq!(*v, 0.0, "alpha_b = {alpha_b}, face {i} got {v}");
            }
        }
    }

    /// SPEC-LIT §39.6's device twin, dynamic branch. `grad(alpha)` along `x`
    /// is wall-PARALLEL on a `y` wall, so `t_hat = x_hat` and the
    /// contact-line speed is the derived `-1/2 (U_P + U_b)·t_hat`. The device
    /// must reproduce [`cos_theta_dynamic`] at exactly that `Ca`.
    ///
    /// The sign matters more than the magnitude: a wall-adjacent velocity
    /// pointing AWAY from the liquid is an ADVANCING line, so `theta` must
    /// RISE. Getting that backwards makes a spreading drop bead up, which
    /// still looks like a drop.
    #[test]
    fn the_device_agrees_with_the_host_contact_angle() {
        let Some(g) = gpu() else { return };
        let (hm, _) = block();
        let m = GpuMesh::upload(&g, &hm).expect("upload");

        let (mu, sigma): (Scalar, Scalar) = (1.002e-3, 0.0728);

        for corr in [
            ContactAngleCorrelation::Static,
            ContactAngleCorrelation::JiangOhSlattery,
            ContactAngleCorrelation::CoxVoinov,
        ] {
            let p = ContactAnglePatch {
                theta_e: 45.0,
                theta_a: 60.0,
                theta_r: 30.0,
                correlation: corr,
                ln_l_ratio: 9.2,
            };
            for ux in [-10.0 as Scalar, -1.0, -1e-3, 0.0, 1e-3, 1.0, 10.0] {
                let got = run_cos_theta(
                    &g,
                    &hm,
                    &m,
                    &p,
                    0.5,
                    // grad(alpha) along +x: t_hat = +x_hat on a y wall.
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(ux, 0.0, 0.0),
                    mu,
                    sigma,
                );

                // U_b is zero, so U_cl = -1/2 U_P . t_hat = -ux/2.
                let ca = mu * (-0.5 * ux) / sigma;
                let want = cos_theta_dynamic(
                    corr,
                    cos_deg(p.theta_e),
                    cos_deg(p.theta_a),
                    cos_deg(p.theta_r),
                    ca,
                    p.ln_l_ratio,
                );

                for (i, pinfo) in hm.b_patch.iter().enumerate() {
                    if hm.patches[*pinfo as usize].kind != crate::mesh::PatchKind::Wall {
                        continue;
                    }
                    let rel = (got[i] - want).abs() / want.abs().max(1e-30);
                    assert!(
                        rel < 1e-12,
                        "{} at U = {ux} (Ca = {ca}): device {} host {want}",
                        corr.name(),
                        got[i]
                    );
                }
            }
        }

        // The SIGN, stated on its own so it cannot be lost in the sweep: a
        // wall-adjacent velocity pointing away from the liquid (-x, since
        // grad(alpha) is +x) is ADVANCING, and theta must rise above 45.
        let p = ContactAnglePatch {
            theta_e: 45.0,
            theta_a: 45.0,
            theta_r: 45.0,
            correlation: ContactAngleCorrelation::JiangOhSlattery,
            ln_l_ratio: 0.0,
        };
        let adv = run_cos_theta(
            &g, &hm, &m, &p, 0.5,
            Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), mu, sigma,
        );
        let rec = run_cos_theta(
            &g, &hm, &m, &p, 0.5,
            Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), mu, sigma,
        );
        for (i, pinfo) in hm.b_patch.iter().enumerate() {
            if hm.patches[*pinfo as usize].kind != crate::mesh::PatchKind::Wall {
                continue;
            }
            assert!(
                acos_deg(adv[i]) > 45.0,
                "advancing must RAISE theta; face {i} gave {} deg",
                acos_deg(adv[i])
            );
            assert!(
                acos_deg(rec[i]) < 45.0,
                "receding must LOWER theta; face {i} gave {} deg",
                acos_deg(rec[i])
            );
        }
    }

    /// SPEC-LIT §39.3 on the device: the `alpha` triple's `refGrad` becomes
    /// `|grad(alpha)_P| cos(theta)` on an owned face with an interface, zero
    /// on an owned face without one, and is LEFT ALONE on a face the model
    /// does not own.
    #[test]
    fn the_alpha_ref_grad_is_the_tilted_wall_gradient() {
        let Some(g) = gpu() else { return };
        let (hm, _) = block();
        let m = GpuMesh::upload(&g, &hm).expect("upload");
        let nbf = hm.n_boundary_faces;

        let p = ContactAnglePatch { theta_e: 45.0, theta_a: 45.0, theta_r: 45.0, ..Default::default() };
        let mut faces = ContactAngleFaces::none(nbf);
        for pi in &hm.patches {
            if pi.kind == crate::mesh::PatchKind::Wall {
                faces.set_patch(pi.start, pi.size, &p);
            }
        }
        let d = ContactAngleDevice::upload(&g, &faces).expect("upload");

        let grad_v = Vec3::new(0.0, 3.0, 4.0); // |grad| = 5 exactly
        let d_grad = g.upload(&vec![grad_v; hm.n_cells]).expect("upload");
        let d_alpha_b = g.upload(&vec![0.5 as Scalar; nbf]).expect("upload");
        let d_u = g.upload(&vec![Vec3::ZERO; hm.n_cells]).expect("upload");
        let d_bu = g.upload(&vec![Vec3::ZERO; nbf]).expect("upload");

        update_cos_theta(&g, &d, &d_alpha_b, &d_grad, &d_u, &d_bu, 1.002e-3, 0.0728, &m)
            .expect("launch");

        // A sentinel the kernel must not overwrite on an unowned face.
        let sentinel: Scalar = -12345.0;
        let mut ref_grad = g.upload(&vec![sentinel; nbf]).expect("upload");
        update_alpha_ref_grad(&g, &d, &mut ref_grad, &d_grad, &m).expect("launch");
        g.sync().expect("sync");
        let got = g.download(&ref_grad).expect("download");

        let want = 5.0 * cos_deg(45.0);
        for (i, pinfo) in hm.b_patch.iter().enumerate() {
            if hm.patches[*pinfo as usize].kind == crate::mesh::PatchKind::Wall {
                assert!(
                    (got[i] - want).abs() <= 1e-12 * want,
                    "wall face {i}: refGrad {} against |grad| cos(theta) = {want}",
                    got[i]
                );
            } else {
                assert_eq!(got[i], sentinel, "face {i} is not owned and was overwritten");
            }
        }

        // theta = 90 gives refGrad EXACTLY zero, i.e. zero-gradient, i.e. the
        // condition a wall carried before S39 - the other half of the trap.
        let p90 = ContactAnglePatch::default();
        let mut faces90 = ContactAngleFaces::none(nbf);
        for pi in &hm.patches {
            if pi.kind == crate::mesh::PatchKind::Wall {
                faces90.set_patch(pi.start, pi.size, &p90);
            }
        }
        let d90 = ContactAngleDevice::upload(&g, &faces90).expect("upload");
        update_cos_theta(&g, &d90, &d_alpha_b, &d_grad, &d_u, &d_bu, 1.002e-3, 0.0728, &m)
            .expect("launch");
        let mut rg90 = g.upload(&vec![sentinel; nbf]).expect("upload");
        update_alpha_ref_grad(&g, &d90, &mut rg90, &d_grad, &m).expect("launch");
        g.sync().expect("sync");
        let got90 = g.download(&rg90).expect("download");
        for (i, pinfo) in hm.b_patch.iter().enumerate() {
            if hm.patches[*pinfo as usize].kind == crate::mesh::PatchKind::Wall {
                assert_eq!(
                    got90[i].to_bits(),
                    (0.0 as Scalar).to_bits(),
                    "wall face {i}: theta = 90 gave refGrad {} and not a literal zero",
                    got90[i]
                );
            }
        }
    }
}
