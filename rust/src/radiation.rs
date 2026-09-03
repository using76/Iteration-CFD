// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Which radiation model, and the one constant every model shares
//! (SPEC-LIT `SPEC-LIT.md` sections 13.4, 49 and 50).
//!
//! Radiation splits at one question: does the medium between the surfaces
//! take part? If it absorbs, emits or scatters, the radiative transfer
//! equation carries a volumetric source and the answer is a FIELD on the
//! cells. If it does not, there is no field to solve for at all: the surfaces
//! exchange directly across a transparent gap and the unknown is one
//! radiosity per boundary face - that is S49/S50's surface-to-surface model,
//! [`crate::s2s`], and it is the model this engine carries.
//!
//! Two things are here:
//!
//! * [`SIGMA_SB`], the Stefan-Boltzmann constant. Every radiative flux in
//!   this crate is proportional to it and none of them care which model
//!   produced the temperature it multiplies.
//! * [`RadiationModel`], S13.4's name gate, and [`RadiationConfig`], what a
//!   case file resolves to. The gate recognises `viewFactor` - and `s2s`, the
//!   native spelling of the same model - and nothing else. Any other value is
//!   an unrecognised setting, refused with the recognised set beside it
//!   rather than substituted for silently, which is what S13.4 asks for.
//!
//! What is NOT here is any solver for a PARTICIPATING medium, nor any of the
//! entries a medium is described by. This reader takes `emissivity`,
//! `agglomerate` and `quadrature` out of `constant/radiationProperties` and
//! nothing else; `absorptionCoefficient` describes a medium, and a
//! `viewFactor` case that sets a non-zero one is refused by
//! [`crate::s2s::S2sConfig::from_dict`] (§50.9) rather than having it read
//! and then ignored.
//!
//! Provenance: ORIGINAL. The Stefan-Boltzmann constant is CODATA 2018's
//! exact value, fixed by the SI redefinition of the kelvin; the selector and
//! its refusals are this project's own S13.4 contract.
//! No GPL-licensed source was consulted.

use std::path::Path;

use crate::error::{Error, Result};
use crate::io::contract;
use crate::io::dict::FoamDict;
use crate::Scalar;

/// Stefan-Boltzmann constant, W/(m2 K4) - CODATA 2018 exact value (fixed by
/// the SI redefinition of the kelvin).
pub const SIGMA_SB: Scalar = 5.670_374_419e-8;

// ==========================================================================
//  S13.4  Which radiation model
// ==========================================================================

/// Every radiation model this crate names, and what happens when a case asks
/// for something else.
///
/// [`RadiationModel::ViewFactor`] (SPEC-LIT S49/S50) is grey diffuse
/// surface-to-surface exchange, [`crate::s2s`]. It is the only value this
/// gate recognises; anything else is refused by name with the recognised set
/// beside it, which is the S13.4 contract - a refusal, never a silent
/// substitution.
///
/// An enum with one arm rather than a bare marker, because what
/// [`Self::from_name`] answers is "which model did this case select", and the
/// answer is a member of a set whose size is not a promise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadiationModel {
    /// Grey diffuse surface-to-surface exchange through a NON-participating
    /// medium - SPEC-LIT S49/S50, [`crate::s2s`]. The right model for an
    /// enclosure with nothing in it to absorb, emit or scatter, and the wrong
    /// one for anything else.
    ViewFactor,
}

impl RadiationModel {
    /// Parse a `radiationModel` entry, per SPEC-LIT S13.4: recognised and
    /// implemented -> use it; anything else -> an error naming the setting
    /// and what IS recognised.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            // SPEC-LIT S49/S50. `viewFactor` is OpenFOAM's spelling of the
            // selector and `s2s` is the native one; both name one model.
            "viewFactor" | "s2s" => Ok(Self::ViewFactor),
            other => contract::unsupported(
                "radiationModel",
                other,
                &["viewFactor"],
                "viewFactor",
                Self::ViewFactor,
            ),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::ViewFactor => "viewFactor",
        }
    }
}

// ==========================================================================
//  S13.4  Which radiation model, resolved to its properties
// ==========================================================================

/// What `constant/radiationProperties` resolves to - exactly one model,
/// [`crate::s2s`]'s grey diffuse surface-to-surface exchange (SPEC-LIT
/// S49/S50).
///
/// An enum with one arm rather than a bare [`crate::s2s::S2sConfig`], because
/// what [`Self::from_case`] returns is "the radiation model this case
/// selected". The arm carries no absorption coefficient deliberately: a
/// surface-to-surface enclosure has no participating medium, and
/// `S2sConfig::from_dict` refuses a case that sets one rather than reading it
/// and dropping it (§50.9).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RadiationConfig {
    S2s(crate::s2s::S2sConfig),
}

impl RadiationConfig {
    pub fn model(&self) -> RadiationModel {
        match self {
            Self::S2s(_) => RadiationModel::ViewFactor,
        }
    }

    /// Read `radiationModel` from `constant/radiationProperties` and then the
    /// entries that describe the enclosure - SPEC-LIT S13.4's own gate
    /// ([`RadiationModel::from_name`]) and then
    /// [`crate::s2s::S2sConfig::from_dict`].
    ///
    /// Neither a missing file nor a missing `radiationModel` entry has a
    /// default. S13.4 selects the model BY NAME, and with one recognised
    /// value it would be easy to guess - but a reader that guessed would be
    /// answering a question the case never asked, which is the substitution
    /// this section exists to refuse.
    pub fn from_case(case_dir: &Path) -> Result<Self> {
        let p = case_dir.join("constant").join("radiationProperties");
        if !p.exists() {
            return Err(Error::Config(format!(
                "{} does not exist; SPEC-LIT S13.4 selects a radiation model \
                 by name in that file and there is no default for it",
                p.display()
            )));
        }
        let d = FoamDict::read(&p)?;
        if !d.has("radiationModel") {
            return Err(Error::Config(format!(
                "{}: no `radiationModel` entry. SPEC-LIT S13.4 selects the \
                 model by name and there is no default for it; recognised: \
                 viewFactor",
                p.display()
            )));
        }
        let model = RadiationModel::from_name(d.get_or("radiationModel", ""))?;
        match model {
            RadiationModel::ViewFactor => Ok(Self::S2s(crate::s2s::S2sConfig::from_dict(&d)?)),
        }
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn case_with(body: &str, tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ofgpu_radiation_sel_{tag}"));
        let c = dir.join("constant");
        std::fs::create_dir_all(&c).expect("mkdir");
        std::fs::write(c.join("radiationProperties"), body).expect("write");
        dir
    }

    /// SPEC-LIT S13.4: the model this engine carries is recognised under both
    /// of its spellings, and the two spellings name ONE model.
    #[test]
    fn every_model_name_is_recognised() {
        assert_eq!(
            RadiationModel::from_name("viewFactor").unwrap(),
            RadiationModel::ViewFactor
        );
        // OpenFOAM's spelling and the native one name ONE model.
        assert_eq!(RadiationModel::from_name("s2s").unwrap(), RadiationModel::ViewFactor);
        assert_eq!(RadiationModel::ViewFactor.name(), "viewFactor");
    }

    /// SPEC-LIT S13.4: an unrecognised value is refused by name, with the
    /// recognised set beside it. That is the whole of what the gate owes a
    /// case it cannot run - nothing is substituted.
    #[test]
    fn an_unknown_model_is_refused() {
        let _guard = crate::io::contract::permissive_test_guard();
        contract::set_permissive(false);
        let e = RadiationModel::from_name("banana").unwrap_err().to_string();
        assert!(e.contains("banana"), "the refusal must name what was asked for: {e}");
        assert!(e.contains("viewFactor"), "the refusal must name what IS here: {e}");
    }

    /// SPEC-LIT S13.4 selects the model BY NAME, so a `radiationProperties`
    /// that exists but names none is refused, and so is a missing file.
    #[test]
    fn a_case_that_names_no_model_is_refused() {
        let dir = case_with("emissivity 0.83;\n", "nomodel");
        let got = RadiationConfig::from_case(&dir).err().map(|e| e.to_string());
        let missing =
            RadiationConfig::from_case(&dir.join("nowhere")).err().map(|e| e.to_string());
        let _ = std::fs::remove_dir_all(&dir);

        let got = got.expect("a case naming no model must not resolve");
        assert!(got.contains("radiationModel"), "{got}");
        assert!(got.contains("no default"), "{got}");
        let missing = missing.expect("a missing file must not resolve");
        assert!(missing.contains("no default"), "{missing}");
    }
}
