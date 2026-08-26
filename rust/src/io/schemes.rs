// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! `system/fvSchemes`, read per equation.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §3.1, §3.5, §7, §11 (convection beyond the TVD
//!     family), §12 (gradient and surface-normal-gradient schemes) and §13.4
//!     (the rule that an unimplemented setting is an error, not a
//!     substitution), and the papers those sections cite:
//!   P. K. Sweby, SIAM J. Numer. Anal. 21 (1984) 995
//!   B. van Leer, J. Comput. Phys. 23 (1977) 276; ibid. 32 (1979) 101
//!   G. D. van Albada, B. van Leer, W. W. Roberts, Astron. Astrophys. 108
//!     (1982) 76
//!   P. L. Roe, Ann. Rev. Fluid Mech. 18 (1986) 337
//!   R. F. Warming, R. M. Beam, AIAA J. 14 (1976) 1241
//!   B. P. Leonard, Comput. Methods Appl. Mech. Eng. 19 (1979) 59
//!   H. Jasak, H. G. Weller, A. D. Gosman, Int. J. Numer. Methods Fluids 31
//!     (1999) 431
//!   T. J. Barth, D. C. Jespersen, AIAA 89-0366 (1989)
//!   V. Venkatakrishnan, AIAA 93-0880 (1993)
//!   H. Jasak, PhD thesis, Imperial College (1996), §3.3, §3.4
//! The dictionary syntax itself is read off the data files, which are a file
//! format and not a work. No GPL-licensed source was consulted.
//!
//! # Why this module exists
//!
//! The reader it replaces looked at `div(phi,k)`, `div(phi,epsilon)`,
//! `div(phi,omega)` and `default` in that order, took the first hit, and used
//! it for **every** equation in the case - including momentum, whose own
//! `div(phi,U)` entry was never read at all. A case saying
//!
//! ```text
//! div(phi,U)  Gauss linearUpwind grad(U);
//! div(phi,k)  bounded Gauss upwind;
//! ```
//!
//! ran its momentum equation on first-order upwind and said nothing. Every
//! lookup here is by the equation's own key, with `default` as the *only*
//! fallback, and a name this solver does not implement is an error under the
//! rule of SPEC-LIT §13.4 rather than a quiet demotion.
//!
//! # The shape of an entry
//!
//! ```text
//! divSchemes          [bounded] Gauss <scheme> [coeff]
//! gradSchemes         [cellLimited|faceLimited[<limiter>]] <base> [coeff]
//! laplacianSchemes    Gauss <interpolation> <snGradScheme>
//! snGradSchemes       uncorrected | corrected | limited <alpha> | orthogonal
//! interpolationSchemes <scheme>
//! ```
//!
//! `bounded` is a property of the **entry**, not of the case: one equation may
//! be bounded and another not, and the reader this replaces made it global.

use std::collections::BTreeMap;

use crate::error::Result;
use crate::fv::{
    DivScheme, GradBase, GradLimit, GradLimiterKind, GradScheme, Limiter, SnGradScheme,
};
use crate::io::contract::{unreadable, unsupported};
use crate::io::dict::FoamDict;
use crate::Scalar;

// ==========================================================================
//  One divSchemes entry
// ==========================================================================

/// A `divSchemes` entry: the interpolation, and whether it was wrapped in
/// `bounded`.
///
/// The two travel together because they are one line of the dictionary.
/// Splitting them - which the previous reader did - is how `div(phi,U)`'s
/// scheme came to be taken from one entry and its boundedness from another.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DivEntry {
    pub scheme: DivScheme,

    /// `bounded Gauss ...`: subtract `V_P·(∇·u)_P` from the diagonal, which
    /// cancels the spurious source a non-solenoidal flux injects
    /// (Moukalled et al. §15.4, SPEC-LIT §3.1). Costs one pass and is
    /// identically zero when `phi` conserves mass.
    pub bounded: bool,
}

impl DivEntry {
    /// The fallback `-permissive` substitutes, and the value a driver with no
    /// `fvSchemes` at all runs with: first-order upwind, bounded.
    ///
    /// Bounded even with no `bounded` prefix anywhere, because a case that
    /// says nothing about its schemes is exactly the case whose flux nothing
    /// has made conservative yet.
    pub const UPWIND: Self = Self {
        scheme: DivScheme::Upwind,
        bounded: true,
    };
}

impl Default for DivEntry {
    fn default() -> Self {
        Self::UPWIND
    }
}

// ==========================================================================
//  The menus, printed in every diagnostic
// ==========================================================================

/// Every `divSchemes` name this solver implements. Printed verbatim in the
/// error a rejected entry raises, because "not supported" without the menu
/// sends the user to the source.
pub const DIV_AVAILABLE: &[&str] = &[
    "Gauss linear",
    "Gauss upwind",
    "Gauss linearUpwind [grad]",
    "Gauss cubic",
    "Gauss QUICK",
    "Gauss QUICKUnlimited",
    "Gauss Gamma <0.1..0.5>",
    "Gauss blended <0..1>",
    "Gauss linearUpwindBlended <0..1>",
    "Gauss limitedLinear <1..2>",
    "Gauss vanLeer",
    "Gauss vanAlbada",
    "Gauss Minmod",
    "Gauss SuperBee",
    "Gauss MUSCL",
];

pub const GRAD_AVAILABLE: &[&str] = &[
    "Gauss linear",
    "leastSquares",
    "cellLimited[<BarthJespersen|Venkatakrishnan>] <base> <coeff>",
    "faceLimited[<BarthJespersen|Venkatakrishnan>] <base> <coeff>",
];

pub const SNGRAD_AVAILABLE: &[&str] = &["corrected", "uncorrected", "orthogonal", "limited <alpha>"];

pub const INTERP_AVAILABLE: &[&str] = &["linear"];

/// Names that are real schemes in the literature, and that this solver
/// recognises as such, but does not implement. Listed so the diagnostic can
/// say "recognised, not implemented" rather than "not recognised" - SPEC-LIT
/// §13.4 makes them different sentences on purpose, because one is a typo and
/// the other is a missing feature.
const DIV_KNOWN_UNIMPLEMENTED: &[&str] = &[
    "SFCD",
    "filteredLinear",
    "filteredLinear2",
    "filteredLinear3",
    "localBlended",
    "midPoint",
    "linearFit",
    "quadraticFit",
    "limitWith",
    "limitedCubic",
    "SuperBeeV",
    "vanLeerV",
    "limitedLinearV",
    "GammaV",
    "linearUpwindV",
    "interfaceCompression",
    "downwind",
    "skewCorrected",
    "clippedLinear",
];

const GRAD_KNOWN_UNIMPLEMENTED: &[&str] = &[
    "pointCellsLeastSquares",
    "edgeCellsLeastSquares",
    "fourth",
    "iterativeGauss",
    "extendedLeastSquares",
];

const SNGRAD_KNOWN_UNIMPLEMENTED: &[&str] =
    &["faceCorrected", "quadraticFit", "linearFit", "skewCorrected"];

// ==========================================================================
//  The reader
// ==========================================================================

/// `system/fvSchemes`, looked up one equation at a time.
///
/// Lookups are **lazy** on purpose: an entry is parsed, and therefore
/// validated, only when an equation actually asks for it. A case listing
/// `div(phi,h)` for an enthalpy equation this build does not solve is not
/// wrong, and refusing to start over it would be.
#[derive(Debug, Clone, Default)]
pub struct FvSchemes {
    d: FoamDict,

    /// Whether the file named a `divSchemes` / `gradSchemes` / ... dictionary
    /// at all. A case with no `fvSchemes` is not making a claim about its
    /// schemes and gets the documented defaults; a case that *has* the
    /// dictionary and still has no usable entry for an equation it is solving
    /// is making a claim that cannot be honoured, and that is an error.
    has: BTreeMap<String, bool>,
}

impl FvSchemes {
    pub fn from_dict(d: FoamDict) -> Self {
        let mut has = BTreeMap::new();
        for group in [
            "divSchemes",
            "gradSchemes",
            "laplacianSchemes",
            "snGradSchemes",
            "interpolationSchemes",
        ] {
            let present = d
                .entries
                .keys()
                .any(|k| k.starts_with(group) && k[group.len()..].starts_with('/'));
            has.insert(group.to_string(), present);
        }
        Self { d, has }
    }

    /// The dictionary itself, for a reader that wants something this module
    /// does not model - `ddtSchemes`, say, which belongs to §13.
    pub fn dict(&self) -> &FoamDict {
        &self.d
    }

    fn group_present(&self, group: &str) -> bool {
        self.has.get(group).copied().unwrap_or(false)
    }

    /// The raw value for `<group>/<key>`, falling back to `<group>/default`.
    ///
    /// `none` - the near-universal `default none;` idiom - is treated as *no
    /// entry*, because that is what it says: this dictionary declines to give
    /// a scheme for anything it has not named explicitly.
    fn lookup(&self, group: &str, key: &str) -> Option<String> {
        let usable = |v: Option<&str>| -> Option<String> {
            let v = v?.trim();
            if v.is_empty() || v == "none" {
                None
            } else {
                Some(v.to_string())
            }
        };

        usable(self.d.get(&format!("{group}/{key}")))
            .or_else(|| usable(self.d.get(&format!("{group}/default"))))
    }

    /// The three-way outcome of a lookup, before any parsing.
    fn resolve(&self, group: &str, key: &str, fallback_name: &str) -> Result<Option<String>> {
        if let Some(v) = self.lookup(group, key) {
            return Ok(Some(v));
        }

        if !self.group_present(group) {
            // No such dictionary at all: the case says nothing, so the
            // documented default stands and nothing is being substituted for.
            return Ok(None);
        }

        // The dictionary exists and has neither this key nor a usable
        // `default`. SPEC-LIT §13.4: a missing setting the solver needs is an
        // error naming the setting, not a guess.
        unsupported(
            &format!("{group}/{key}"),
            "<missing>",
            &[],
            fallback_name,
            (),
        )?;
        Ok(None)
    }

    // ---------------------------------------------------------------- div

    /// The `divSchemes` entry for one equation, e.g. `div(phi,U)`.
    pub fn div(&self, key: &str) -> Result<DivEntry> {
        let setting = format!("divSchemes/{key}");
        match self.resolve("divSchemes", key, "Gauss upwind")? {
            None => Ok(DivEntry::UPWIND),
            Some(raw) => parse_div(&setting, &raw),
        }
    }

    // --------------------------------------------------------------- grad

    /// The `gradSchemes` entry for one field, e.g. `grad(U)`.
    pub fn grad(&self, key: &str) -> Result<GradScheme> {
        let setting = format!("gradSchemes/{key}");
        match self.resolve("gradSchemes", key, "Gauss linear")? {
            None => Ok(GradScheme::GAUSS),
            Some(raw) => parse_grad(&setting, &raw),
        }
    }

    // ------------------------------------------------------------- snGrad

    /// The `snGradSchemes` entry for one field.
    pub fn sn_grad(&self, key: &str) -> Result<SnGradScheme> {
        let setting = format!("snGradSchemes/{key}");
        match self.resolve("snGradSchemes", key, "corrected")? {
            None => Ok(SnGradScheme::Corrected),
            Some(raw) => parse_sn_grad(&setting, &raw),
        }
    }

    /// The `laplacianSchemes` entry for one equation, e.g.
    /// `laplacian(nuEff,U)`.
    ///
    /// The entry is `Gauss <interpolation> <snGradScheme>`. The interpolation
    /// names how the diffusivity reaches the face; only `linear` is
    /// implemented, and anything else is rejected rather than ignored. What
    /// comes back is the surface-normal-gradient half, which is the half that
    /// changes the discretisation.
    pub fn laplacian(&self, key: &str) -> Result<SnGradScheme> {
        let setting = format!("laplacianSchemes/{key}");
        match self.resolve("laplacianSchemes", key, "Gauss linear corrected")? {
            None => Ok(SnGradScheme::Corrected),
            Some(raw) => parse_laplacian(&setting, &raw),
        }
    }

    /// The `interpolationSchemes` entry. Only `linear` is implemented; this
    /// exists so that a case asking for something else is told so instead of
    /// having the entry silently dropped, which is what used to happen.
    pub fn interpolation(&self, key: &str) -> Result<()> {
        let setting = format!("interpolationSchemes/{key}");
        match self.resolve("interpolationSchemes", key, "linear")? {
            None => Ok(()),
            Some(raw) => parse_interpolation(&setting, &raw),
        }
    }

    /// Validate the whole file for the equations a driver is going to solve.
    ///
    /// Every lookup here is one a driver will make later; making them up front
    /// means a case with a typo in `div(phi,epsilon)` fails in the first
    /// second rather than after the mesh, the fields and the GPU context are
    /// up. `keys` are the div keys the driver needs.
    pub fn validate(&self, div_keys: &[&str]) -> Result<()> {
        for k in div_keys {
            self.div(k)?;
        }
        self.grad("default")?;
        self.sn_grad("default")?;
        self.laplacian("default")?;
        self.interpolation("default")?;
        Ok(())
    }
}

// ==========================================================================
//  Parsing
// ==========================================================================

fn words(raw: &str) -> Vec<&str> {
    raw.split_whitespace().filter(|s| !s.is_empty()).collect()
}

/// The trailing coefficient of an entry, if the last word is a number.
fn trailing_coeff(w: &[&str]) -> Option<Scalar> {
    w.last().and_then(|s| s.parse::<Scalar>().ok())
}

/// `cellLimited<Venkatakrishnan>` -> `("cellLimited", Some("Venkatakrishnan"))`
fn split_angle(w: &str) -> (&str, Option<&str>) {
    match (w.find('<'), w.strip_suffix('>')) {
        (Some(i), Some(_)) => (&w[..i], Some(&w[i + 1..w.len() - 1])),
        _ => (w, None),
    }
}

/// SPEC-LIT §13.4 makes "recognised, not implemented" and "not recognised"
/// different sentences on purpose: one is a missing feature and the other is a
/// typo, and the user does different things about them.
fn reject_div<T: Copy>(setting: &str, raw: &str, name: &str, fallback: T) -> Result<T> {
    let value = if DIV_KNOWN_UNIMPLEMENTED.iter().any(|k| *k == name) {
        format!("{raw} (recognised, not implemented)")
    } else {
        format!("{raw} (not a recognised scheme name)")
    };
    unsupported(setting, &value, DIV_AVAILABLE, "Gauss upwind", fallback)
}

/// Parse one `divSchemes` value, e.g. `bounded Gauss limitedLinear 1`.
///
/// `limitedLinear <c>` maps onto Sweby-φ with `β = c`, clamped to the
/// `1 ≤ β ≤ 2` the TVD proof covers. SPEC-LIT §7 has no separate entry for
/// `limitedLinear`; Sweby-φ is its parameterised limiter and `β = 1` is
/// minmod, the most strongly bounded member of the family, so the usual
/// `limitedLinear 1` keeps meaning "fully limited". *DESIGN*, and the reason
/// the coefficient is no longer thrown away: the previous reader mapped every
/// `limitedLinear` to `β = 1` whatever the case wrote.
pub fn parse_div(setting: &str, raw: &str) -> Result<DivEntry> {
    let mut w = words(raw);

    let mut bounded = false;
    if w.first() == Some(&"bounded") {
        bounded = true;
        w.remove(0);
    }

    // `Gauss` is the only integration this solver has: every operator in
    // `fv.rs` is a Gauss-theorem face sum. A case naming another one is
    // asking for a discretisation that does not exist here.
    match w.first() {
        Some(&"Gauss") => {
            w.remove(0);
        }
        Some(other) => {
            let other = (*other).to_string();
            return reject_div(setting, raw, &other, DivEntry::UPWIND);
        }
        None => {
            return unsupported(setting, raw, DIV_AVAILABLE, "Gauss upwind", DivEntry::UPWIND)
        }
    }

    let Some(name) = w.first().copied() else {
        return unsupported(setting, raw, DIV_AVAILABLE, "Gauss upwind", DivEntry::UPWIND);
    };

    let coeff = trailing_coeff(&w);

    // A coefficient the scheme requires and the case did not write.
    let need = |what: &str| -> Result<Scalar> {
        match coeff {
            Some(c) => Ok(c),
            None => unreadable(
                setting,
                raw,
                &format!("{name} <{what}>: the coefficient is missing"),
                0.0,
            ),
        }
    };

    let scheme = match name {
        "linear" => DivScheme::Central,
        "upwind" => DivScheme::Upwind,

        // The trailing `grad(U)` names the gradient scheme the correction
        // should use. ofgpu uses the field's own `gradSchemes` entry for it,
        // which is the same gradient every other operator on that field sees;
        // a per-scheme override is not modelled, and the word is accepted and
        // ignored rather than rejected because ignoring it changes only which
        // *equally valid* gradient the correction is built from.
        "linearUpwind" => DivScheme::LinearUpwind,

        "cubic" => DivScheme::Cubic,

        // SPEC-LIT §11.3 DESIGN: a bare QUICK is the LIMITED form.
        "QUICK" | "quick" => DivScheme::Quick,
        "QUICKUnlimited" | "quickUnlimited" => DivScheme::QuickUnlimited,

        "Gamma" | "gamma" => DivScheme::Gamma(need("beta_m 0.1..0.5")?),

        "blended" => DivScheme::Blended(need("gamma 0..1")?),

        // SPEC-LIT §11.5 DESIGN: 0.75 when the case does not say. It is a
        // tuning constant, not a canonical value, and the doc comment on
        // `DivScheme::LinearUpwindBlended` says so.
        "linearUpwindBlended" => DivScheme::LinearUpwindBlended(coeff.unwrap_or(0.75)),

        "limitedLinear" | "Sweby" | "sweby" => {
            DivScheme::Limited(Limiter::Sweby(need("beta 1..2")?))
        }

        "vanLeer" | "vanleer" => DivScheme::Limited(Limiter::VanLeer),
        "vanAlbada" | "vanalbada" => DivScheme::Limited(Limiter::VanAlbada),
        "Minmod" | "minmod" | "MinMod" => DivScheme::Limited(Limiter::MinMod),
        "SuperBee" | "superBee" | "superbee" | "Superbee" => {
            DivScheme::Limited(Limiter::Superbee)
        }
        "MUSCL" | "Muscl" | "muscl" => DivScheme::Limited(Limiter::Muscl),

        other => return reject_div(setting, raw, other, DivEntry::UPWIND),
    };

    Ok(DivEntry { scheme, bounded })
}

/// Parse one `gradSchemes` value.
pub fn parse_grad(setting: &str, raw: &str) -> Result<GradScheme> {
    let w = words(raw);
    let Some(head) = w.first().copied() else {
        return unsupported(setting, raw, GRAD_AVAILABLE, "Gauss linear", GradScheme::GAUSS);
    };

    let (wrapper, angle) = split_angle(head);

    let limited = matches!(wrapper, "cellLimited" | "faceLimited");
    let inner = if limited { &w[1..] } else { &w[..] };

    let base = match inner.first().copied() {
        Some("Gauss") => match inner.get(1).copied() {
            // `Gauss linear` is the only Gauss gradient there is: the
            // gradient's face value is an interpolation, and `linear` is the
            // only interpolation this solver implements (§12.1).
            Some("linear") | None => GradBase::Gauss,
            Some(other) => {
                let other = other.to_string();
                return reject_grad(setting, raw, &other);
            }
        },
        Some("leastSquares") => GradBase::LeastSquares,
        Some(other) => {
            let other = other.to_string();
            return reject_grad(setting, raw, &other);
        }
        None => return reject_grad(setting, raw, ""),
    };

    if !limited {
        return Ok(GradScheme {
            base,
            limit: GradLimit::None,
        });
    }

    // *DESIGN*: an unqualified `cellLimited` is Barth-Jespersen, the algorithm
    // SPEC-LIT §12.2 states first and the one the name comes from.
    // Venkatakrishnan's smooth variant has to be asked for.
    let kind = match angle {
        None | Some("BarthJespersen") | Some("barthJespersen") => {
            GradLimiterKind::BarthJespersen
        }
        Some("Venkatakrishnan") | Some("venkatakrishnan") => GradLimiterKind::Venkatakrishnan,
        Some(other) => {
            let other = other.to_string();
            return reject_grad(setting, raw, &other);
        }
    };

    let Some(coeff) = trailing_coeff(&w) else {
        return unreadable(
            setting,
            raw,
            &format!("{wrapper} <base> <coeff>: the coefficient is missing"),
            GradScheme::GAUSS,
        );
    };

    let limit = if wrapper == "cellLimited" {
        GradLimit::Cell(kind, coeff)
    } else {
        GradLimit::Face(kind, coeff)
    };

    Ok(GradScheme { base, limit })
}

fn reject_grad(setting: &str, raw: &str, name: &str) -> Result<GradScheme> {
    let value = if GRAD_KNOWN_UNIMPLEMENTED.iter().any(|k| *k == name) {
        format!("{raw} (recognised, not implemented)")
    } else {
        format!("{raw} (not a recognised gradient scheme)")
    };
    unsupported(setting, &value, GRAD_AVAILABLE, "Gauss linear", GradScheme::GAUSS)
}

/// Parse one `snGradSchemes` value (SPEC-LIT §12.3).
pub fn parse_sn_grad(setting: &str, raw: &str) -> Result<SnGradScheme> {
    let w = words(raw);
    let Some(head) = w.first().copied() else {
        return unsupported(
            setting,
            raw,
            SNGRAD_AVAILABLE,
            "corrected",
            SnGradScheme::Corrected,
        );
    };

    match head {
        "corrected" => Ok(SnGradScheme::Corrected),

        // `orthogonal` asserts the mesh is orthogonal, so the correction is
        // zero by assumption - the same discretisation `uncorrected` gives.
        "uncorrected" | "orthogonal" => Ok(SnGradScheme::Uncorrected),

        // `limited <a>` and `limited corrected <a>` are the same entry; the
        // coefficient is the last word either way.
        "limited" => match trailing_coeff(&w) {
            Some(a) => Ok(SnGradScheme::Limited(a)),
            None => unreadable(
                setting,
                raw,
                "limited <alpha>: the coefficient is missing",
                SnGradScheme::Corrected,
            ),
        },

        other => {
            let value = if SNGRAD_KNOWN_UNIMPLEMENTED.iter().any(|k| *k == other) {
                format!("{raw} (recognised, not implemented)")
            } else {
                format!("{raw} (not a recognised snGrad scheme)")
            };
            unsupported(
                setting,
                &value,
                SNGRAD_AVAILABLE,
                "corrected",
                SnGradScheme::Corrected,
            )
        }
    }
}

/// Parse one `laplacianSchemes` value, `Gauss <interpolation> <snGrad>`.
pub fn parse_laplacian(setting: &str, raw: &str) -> Result<SnGradScheme> {
    let w = words(raw);

    if w.first() != Some(&"Gauss") {
        return unsupported(
            setting,
            raw,
            &["Gauss linear <snGradScheme>"],
            "Gauss linear corrected",
            SnGradScheme::Corrected,
        );
    }

    // The diffusivity's face interpolation. `fvm_laplacian` takes
    // `gamma_f·|Sf|` already multiplied together and every caller forms
    // `gamma_f` by linear interpolation, so this is the only value the
    // assembly can honour.
    match w.get(1).copied() {
        Some("linear") => {}
        Some(other) => {
            let other = other.to_string();
            return unsupported(
                setting,
                &other,
                INTERP_AVAILABLE,
                "Gauss linear corrected",
                SnGradScheme::Corrected,
            );
        }
        None => {
            return unsupported(
                setting,
                raw,
                &["Gauss linear <snGradScheme>"],
                "Gauss linear corrected",
                SnGradScheme::Corrected,
            )
        }
    }

    if w.len() < 3 {
        return unsupported(
            setting,
            raw,
            &["Gauss linear <snGradScheme>"],
            "Gauss linear corrected",
            SnGradScheme::Corrected,
        );
    }

    parse_sn_grad(setting, &w[2..].join(" "))
}

/// Parse one `interpolationSchemes` value.
pub fn parse_interpolation(setting: &str, raw: &str) -> Result<()> {
    match words(raw).first().copied() {
        Some("linear") => Ok(()),
        _ => unsupported(setting, raw, INTERP_AVAILABLE, "linear", ()),
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::contract::{reset_warnings, set_permissive};

    /// `-permissive` and the warn-once set are process-wide, so this takes
    /// the CRATE-WIDE guard - a mutex private to this module excludes only
    /// this module's tests, and the flag is shared with every other one.
    /// (A private SERIAL here was exactly how
    /// `case_json::tests::jsonc_wall_row_contradiction_is_resolved_under_permissive`
    /// flaked: it held the crate guard, this held its own, and the two ran
    /// concurrently with one setting the flag the other had just cleared.)
    fn strict() -> std::sync::MutexGuard<'static, ()> {
        let g = crate::io::contract::permissive_test_guard();
        set_permissive(false);
        reset_warnings();
        g
    }

    const PLUME_B: &str = r#"
        ddtSchemes { default steadyState; }
        gradSchemes { default Gauss linear; }
        divSchemes
        {
            default          none;
            div(phi,U)       Gauss linearUpwind grad(U);
            div(phi,k)       bounded Gauss upwind;
            div(phi,epsilon) bounded Gauss upwind;
        }
        laplacianSchemes { default Gauss linear corrected; }
        interpolationSchemes { default linear; }
        snGradSchemes { default corrected; }
    "#;

    fn plume_b() -> FvSchemes {
        FvSchemes::from_dict(FoamDict::parse(PLUME_B, "fvSchemes").unwrap())
    }

    /// The bug this module was written for: momentum was discretised by
    /// whatever the turbulence entry said.
    #[test]
    fn momentum_reads_its_own_entry() {
        let _g = strict();
        let s = plume_b();

        assert_eq!(s.div("div(phi,U)").unwrap().scheme, DivScheme::LinearUpwind);
        assert!(!s.div("div(phi,U)").unwrap().bounded);

        assert_eq!(s.div("div(phi,k)").unwrap().scheme, DivScheme::Upwind);
        assert!(s.div("div(phi,k)").unwrap().bounded);
    }

    #[test]
    fn a_missing_key_with_default_none_is_an_error() {
        let _g = strict();
        let e = plume_b().div("div(phi,T)").unwrap_err().to_string();
        assert!(e.contains("div(phi,T)"), "{e}");
        assert!(e.contains("-permissive"), "{e}");
    }

    #[test]
    fn permissive_downgrades_and_says_what_it_substituted() {
        // The crate-wide guard alone - this used to take a second, private
        // mutex as well, which is a lock-ordering hazard as well as no extra
        // protection.
        let _guard = crate::io::contract::permissive_test_guard();
        reset_warnings();
        set_permissive(true);
        let e = plume_b().div("div(phi,T)").unwrap();
        assert_eq!(e, DivEntry::UPWIND);
        set_permissive(false);
    }

    #[test]
    fn every_limiter_of_section_7_is_reachable_by_name() {
        let _g = strict();
        let cases: &[(&str, DivScheme)] = &[
            ("Gauss vanLeer", DivScheme::Limited(Limiter::VanLeer)),
            ("Gauss vanAlbada", DivScheme::Limited(Limiter::VanAlbada)),
            ("Gauss Minmod", DivScheme::Limited(Limiter::MinMod)),
            ("Gauss SuperBee", DivScheme::Limited(Limiter::Superbee)),
            ("Gauss MUSCL", DivScheme::Limited(Limiter::Muscl)),
            (
                "Gauss limitedLinear 1",
                DivScheme::Limited(Limiter::Sweby(1.0)),
            ),
            (
                "bounded Gauss limitedLinear 1.5",
                DivScheme::Limited(Limiter::Sweby(1.5)),
            ),
        ];
        for (raw, want) in cases {
            let got = parse_div("divSchemes/div(phi,U)", raw).unwrap();
            assert_eq!(got.scheme, *want, "{raw}");
        }
    }

    /// The coefficient used to be thrown away; two different `limitedLinear`
    /// entries produced the same limiter.
    #[test]
    fn the_limiter_coefficient_survives() {
        let _g = strict();
        let a = parse_div("d", "Gauss limitedLinear 1").unwrap().scheme;
        let b = parse_div("d", "Gauss limitedLinear 2").unwrap().scheme;
        assert_ne!(a, b);
    }

    #[test]
    fn section_11_schemes_parse_with_their_coefficients() {
        let _g = strict();
        assert_eq!(
            parse_div("d", "Gauss cubic").unwrap().scheme,
            DivScheme::Cubic
        );
        assert_eq!(
            parse_div("d", "Gauss QUICK").unwrap().scheme,
            DivScheme::Quick
        );
        assert_eq!(
            parse_div("d", "Gauss QUICKUnlimited").unwrap().scheme,
            DivScheme::QuickUnlimited
        );
        assert_eq!(
            parse_div("d", "Gauss Gamma 0.2").unwrap().scheme,
            DivScheme::Gamma(0.2)
        );
        assert_eq!(
            parse_div("d", "Gauss blended 0.9").unwrap().scheme,
            DivScheme::Blended(0.9)
        );
        assert_eq!(
            parse_div("d", "Gauss linearUpwindBlended 0.5").unwrap().scheme,
            DivScheme::LinearUpwindBlended(0.5)
        );
        // SPEC-LIT §11.5 DESIGN default.
        assert_eq!(
            parse_div("d", "Gauss linearUpwindBlended").unwrap().scheme,
            DivScheme::LinearUpwindBlended(0.75)
        );
    }

    #[test]
    fn a_scheme_missing_its_required_coefficient_is_an_error() {
        let _g = strict();
        assert!(parse_div("d", "Gauss Gamma").is_err());
        assert!(parse_div("d", "Gauss blended").is_err());
        assert!(parse_div("d", "Gauss limitedLinear").is_err());
    }

    #[test]
    fn an_unimplemented_scheme_is_an_error_naming_it_and_the_menu() {
        let _g = strict();
        let e = parse_div("divSchemes/div(phi,U)", "Gauss SFCD")
            .unwrap_err()
            .to_string();
        assert!(e.contains("Gauss SFCD"), "{e}");
        assert!(e.contains("Gauss vanLeer"), "{e}");

        // Not a scheme at all.
        assert!(parse_div("d", "Gauss banana").is_err());
        // Not Gauss integration.
        assert!(parse_div("d", "leastSquares linear").is_err());
    }

    #[test]
    fn grad_schemes_parse() {
        let _g = strict();
        assert_eq!(parse_grad("g", "Gauss linear").unwrap(), GradScheme::GAUSS);
        assert_eq!(
            parse_grad("g", "leastSquares").unwrap(),
            GradScheme {
                base: GradBase::LeastSquares,
                limit: GradLimit::None
            }
        );
        assert_eq!(
            parse_grad("g", "cellLimited Gauss linear 1").unwrap(),
            GradScheme {
                base: GradBase::Gauss,
                limit: GradLimit::Cell(GradLimiterKind::BarthJespersen, 1.0)
            }
        );
        assert_eq!(
            parse_grad("g", "cellLimited<Venkatakrishnan> leastSquares 0.5").unwrap(),
            GradScheme {
                base: GradBase::LeastSquares,
                limit: GradLimit::Cell(GradLimiterKind::Venkatakrishnan, 0.5)
            }
        );
        assert_eq!(
            parse_grad("g", "faceLimited Gauss linear 1").unwrap(),
            GradScheme {
                base: GradBase::Gauss,
                limit: GradLimit::Face(GradLimiterKind::BarthJespersen, 1.0)
            }
        );
        assert!(parse_grad("g", "pointCellsLeastSquares").is_err());
        assert!(parse_grad("g", "cellLimited Gauss linear").is_err());
    }

    /// The entry that used to be parsed into the dictionary and never read.
    #[test]
    fn sn_grad_schemes_parse() {
        let _g = strict();
        assert_eq!(
            parse_sn_grad("s", "uncorrected").unwrap(),
            SnGradScheme::Uncorrected
        );
        assert_eq!(
            parse_sn_grad("s", "orthogonal").unwrap(),
            SnGradScheme::Uncorrected
        );
        assert_eq!(
            parse_sn_grad("s", "corrected").unwrap(),
            SnGradScheme::Corrected
        );
        assert_eq!(
            parse_sn_grad("s", "limited 0.33").unwrap(),
            SnGradScheme::Limited(0.33)
        );
        assert_eq!(
            parse_sn_grad("s", "limited corrected 0.5").unwrap(),
            SnGradScheme::Limited(0.5)
        );
        assert!(parse_sn_grad("s", "limited").is_err());
        assert!(parse_sn_grad("s", "faceCorrected").is_err());
    }

    /// SPEC-LIT §12.3: `limited 0` is `uncorrected`, and the alpha the kernel
    /// sees has to make that exact.
    #[test]
    fn limited_zero_is_uncorrected() {
        let _g = strict();
        let a = parse_sn_grad("s", "limited 0").unwrap();
        assert_eq!(a.alpha(), SnGradScheme::Uncorrected.alpha());
        assert_eq!(a.alpha(), 0.0);
    }

    #[test]
    fn laplacian_yields_its_sn_grad_half() {
        let _g = strict();
        assert_eq!(
            parse_laplacian("l", "Gauss linear corrected").unwrap(),
            SnGradScheme::Corrected
        );
        assert_eq!(
            parse_laplacian("l", "Gauss linear limited 0.5").unwrap(),
            SnGradScheme::Limited(0.5)
        );
        assert_eq!(
            parse_laplacian("l", "Gauss linear uncorrected").unwrap(),
            SnGradScheme::Uncorrected
        );
        assert!(parse_laplacian("l", "Gauss harmonic corrected").is_err());
    }

    #[test]
    fn interpolation_is_read_rather_than_discarded() {
        let _g = strict();
        assert!(parse_interpolation("i", "linear").is_ok());
        assert!(parse_interpolation("i", "midPoint").is_err());
    }

    /// A case with no fvSchemes at all still runs: it is making no claim.
    #[test]
    fn an_absent_dictionary_keeps_the_documented_defaults() {
        let _g = strict();
        let s = FvSchemes::default();
        assert_eq!(s.div("div(phi,U)").unwrap(), DivEntry::UPWIND);
        assert_eq!(s.grad("grad(U)").unwrap(), GradScheme::GAUSS);
        assert_eq!(s.sn_grad("default").unwrap(), SnGradScheme::Corrected);
    }

    /// Two different entries must not produce the same scheme - the property
    /// that would have caught the original bug.
    #[test]
    fn different_entries_give_different_schemes() {
        let _g = strict();
        let src = r#"
            divSchemes
            {
                div(phi,U) Gauss vanLeer;
                div(phi,k) Gauss upwind;
                div(phi,T) Gauss linear;
            }
        "#;
        let s = FvSchemes::from_dict(FoamDict::parse(src, "fvSchemes").unwrap());
        let u = s.div("div(phi,U)").unwrap().scheme;
        let k = s.div("div(phi,k)").unwrap().scheme;
        let t = s.div("div(phi,T)").unwrap().scheme;
        assert_ne!(u, k);
        assert_ne!(u, t);
        assert_ne!(k, t);
    }
}
