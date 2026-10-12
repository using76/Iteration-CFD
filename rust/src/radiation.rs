// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
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
//!   native spelling of the same model - and also the participating models
//!   `P1` and `fvDOM` (SPEC-LIT §135.6). Any other value is an unrecognised
//!   setting, refused with the recognised set beside it rather than
//!   substituted for silently, which is what S13.4 asks for.
//!
//! What is NOT here is any solver; the models live in their own modules -
//! the participating medium's P1 solve is [`p1`] (SPEC-LIT §135.7), and the
//! entries a medium is described by are [`props`]. [`RadiationConfig`]
//! resolves `viewFactor` and nothing else: given `P1` or `fvDOM` it refuses
//! the case naming [`props::ParticipatingConfig`], the reader of a
//! participating medium (SPEC-LIT §137.2). A `viewFactor` case that
//! describes a medium anyway - a non-zero `absorptionCoefficient`, or an
//! `absorptionEmissionModel`/`scatterModel` other than `none` - is refused
//! (§50.9, §137.2) rather than having it read and then ignored.
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

/// What a participating medium is made of - the entries a `P1` or `fvDOM`
/// case describes itself by (SPEC-LIT §137.2's table), parsed with its
/// refusals and read by nothing that solves until a model takes it.
pub mod props;

/// The P1 solve (SPEC-LIT §135.3-§135.7, §137.1): the participating medium
/// is now SOLVED - one scalar field `G`, Marshak walls, `q_r`, and the
/// linearised source registered on [`crate::energy::EnergySources`]. A
/// uniform grey medium; fvDOM (SPEC-LIT §136) is still read here and solved
/// by nothing.
pub mod p1;

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
/// surface-to-surface exchange, [`crate::s2s`]. [`RadiationModel::P1`] and
/// [`RadiationModel::FvDom`] (SPEC-LIT §135-§136) are the participating
/// media: recognised here, read by [`props::ParticipatingConfig`], and
/// solved by no driver until §137.3's units land. Anything else is refused
/// by name with the recognised set beside it, which is the S13.4 contract -
/// a refusal, never a silent substitution.
///
/// An enum rather than a bare marker, because what
/// [`Self::from_name`] answers is "which model did this case select", and the
/// answer is a member of a set whose size is not a promise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadiationModel {
    /// The P1 moment approximation to the transfer equation (SPEC-LIT
    /// §135.3): a participating medium whose intensity is carried by one
    /// scalar field `G`. Recognised here; solved by CHR-18.
    P1,
    /// The discrete-ordinates model (SPEC-LIT §136): a participating medium
    /// carried by a set of directions. Recognised here; solved by CHR-19 up.
    FvDom,
    /// Grey diffuse surface-to-surface exchange through a NON-participating
    /// medium - SPEC-LIT S49/S50, [`crate::s2s`]. The right model for an
    /// enclosure with nothing in it to absorb, emit or scatter, and the wrong
    /// one for anything else.
    ViewFactor,
}

impl RadiationModel {
    /// Every name [`Self::from_name`] recognises - the set a refusal prints
    /// beside what was asked for (SPEC-LIT §135.6).
    pub const RECOGNISED: [&'static str; 3] = ["P1", "fvDOM", "viewFactor"];

    /// Parse a `radiationModel` entry, per SPEC-LIT S13.4: recognised ->
    /// the model, and the reader that takes it answers for the rest;
    /// anything else -> an error naming the setting and what IS recognised.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            // SPEC-LIT §135.6: the two participating models, exact and
            // case-sensitive.
            "P1" => Ok(Self::P1),
            "fvDOM" => Ok(Self::FvDom),
            // SPEC-LIT S49/S50. `viewFactor` is OpenFOAM's spelling of the
            // selector and `s2s` is the native one; both name one model.
            "viewFactor" | "s2s" => Ok(Self::ViewFactor),
            other => contract::unsupported(
                "radiationModel",
                other,
                &Self::RECOGNISED,
                "viewFactor",
                Self::ViewFactor,
            ),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::P1 => "P1",
            Self::FvDom => "fvDOM",
            Self::ViewFactor => "viewFactor",
        }
    }

    /// Does this model's medium absorb, emit or scatter? `true` for P1 and
    /// fvDOM (SPEC-LIT §135-§136), `false` for surface-to-surface.
    pub fn is_participating(self) -> bool {
        !matches!(self, Self::ViewFactor)
    }
}

// ==========================================================================
//  S13.4  Which radiation model, resolved to its properties
// ==========================================================================

/// What `constant/radiationProperties` resolves to for the enclosure reader -
/// [`crate::s2s`]'s grey diffuse surface-to-surface exchange (SPEC-LIT
/// S49/S50).
///
/// An enum with one arm rather than a bare [`crate::s2s::S2sConfig`], because
/// what [`Self::from_case`] returns is "the radiation model this case
/// selected". The arm carries no absorption coefficient deliberately: a
/// surface-to-surface enclosure has no participating medium, and
/// `S2sConfig::from_dict` refuses a case that sets one rather than reading it
/// and dropping it (§50.9).
///
/// A case naming `P1` or `fvDOM` does NOT resolve here: [`Self::from_case`]
/// refuses it naming [`props::ParticipatingConfig`], which is what reads a
/// participating medium (SPEC-LIT §137.2). No driver solves one until
/// §137.3's units land.
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
    /// This is the enclosure reader (§98.7 calls it): it resolves
    /// `viewFactor` and nothing else. Given `P1` or `fvDOM` it refuses the
    /// case naming [`props::ParticipatingConfig`] (SPEC-LIT §137.2), because
    /// reading half of a participating medium and dropping the rest would be
    /// the substitution this project refuses in every other dictionary.
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
                 model by name and there is no default for it; recognised: {}",
                p.display(),
                RadiationModel::RECOGNISED.join(", ")
            )));
        }
        let model = RadiationModel::from_name(d.get_or("radiationModel", ""))?;
        match model {
            RadiationModel::P1 | RadiationModel::FvDom => Err(Error::Config(format!(
                "{}: radiationModel {} is a participating medium, which this \
                 reader does not resolve; a participating medium is read by \
                 radiation::props::ParticipatingConfig (SPEC-LIT §135.6, \
                 §137.2), and no driver solves one yet",
                p.display(),
                model.name()
            ))),
            RadiationModel::ViewFactor => {
                Self::refuse_medium_entries(&d)?;
                Ok(Self::S2s(crate::s2s::S2sConfig::from_dict(&d)?))
            }
        }
    }

    /// Under `viewFactor` only `none` is accepted for the two entries that
    /// describe a medium (SPEC-LIT §137.2): the enclosure has none, so
    /// anything else would be read and then ignored (§50.9). `-permissive`
    /// is allowed to substitute `none` and continue, which is why this goes
    /// through [`contract::unsupported_note`] rather than a plain refusal.
    fn refuse_medium_entries(d: &FoamDict) -> Result<()> {
        for entry in ["absorptionEmissionModel", "scatterModel"] {
            let Some(raw) = d.get(entry) else { continue };
            let v = raw.trim();
            if v != "none" {
                contract::unsupported_note(
                    &format!("radiationProperties/{entry}"),
                    v,
                    &["none"],
                    "radiationModel viewFactor is a NON-PARTICIPATING medium: \
                     the entry describes a medium, and under viewFactor it \
                     would be read and then ignored (SPEC-LIT §50.9, §137.2)",
                    "none",
                    (),
                )?;
            }
        }
        Ok(())
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

    /// SPEC-LIT §135.6: every name the gate recognises parses, `name()`
    /// round-trips, and the two spellings of the enclosure name ONE model.
    #[test]
    fn every_model_name_is_recognised() {
        assert_eq!(
            RadiationModel::from_name("viewFactor").unwrap(),
            RadiationModel::ViewFactor
        );
        // OpenFOAM's spelling and the native one name ONE model.
        assert_eq!(RadiationModel::from_name("s2s").unwrap(), RadiationModel::ViewFactor);
        assert_eq!(RadiationModel::ViewFactor.name(), "viewFactor");
        // The participating models, recognised here and read by
        // `props::ParticipatingConfig` (§137.2).
        assert_eq!(RadiationModel::from_name("P1").unwrap(), RadiationModel::P1);
        assert_eq!(RadiationModel::from_name("fvDOM").unwrap(), RadiationModel::FvDom);
        for (m, n) in [
            (RadiationModel::P1, "P1"),
            (RadiationModel::FvDom, "fvDOM"),
            (RadiationModel::ViewFactor, "viewFactor"),
        ] {
            assert_eq!(RadiationModel::from_name(n).unwrap(), m);
            assert_eq!(m.name(), n);
        }
        assert!(RadiationModel::P1.is_participating());
        assert!(RadiationModel::FvDom.is_participating());
        assert!(!RadiationModel::ViewFactor.is_participating());
        // Each absorption and scatter name parses and round-trips.
        for n in props::AbsorptionKind::RECOGNISED {
            let k = props::AbsorptionKind::from_name(n).unwrap();
            assert_eq!(k.name(), n);
            assert_eq!(props::AbsorptionKind::from_name(k.name()).unwrap(), k);
        }
        for n in props::ScatterKind::RECOGNISED {
            let k = props::ScatterKind::from_name(n).unwrap();
            assert_eq!(k.name(), n);
            assert_eq!(props::ScatterKind::from_name(k.name()).unwrap(), k);
        }
        strict_near_misses_are_refused();
    }

    /// The strict half of `every_model_name_is_recognised`: a near-miss is
    /// refused BY NAME, with the recognised set beside it (S13.4).
    fn strict_near_misses_are_refused() {
        let _g = contract::permissive_test_guard();
        contract::set_permissive(false);
        for (asked, gate) in [
            ("banana", "radiationModel"),
            ("p1", "radiationModel"),
            ("wsggg", "absorptionEmissionModel"),
            ("anisotropic", "scatterModel"),
        ] {
            let e = match gate {
                "radiationModel" => RadiationModel::from_name(asked).unwrap_err().to_string(),
                "absorptionEmissionModel" => {
                    props::AbsorptionKind::from_name(asked).unwrap_err().to_string()
                }
                _ => props::ScatterKind::from_name(asked).unwrap_err().to_string(),
            };
            assert!(e.contains(asked), "{gate}: must name what was asked: {e}");
            let set = match gate {
                "radiationModel" => &RadiationModel::RECOGNISED[..],
                "absorptionEmissionModel" => &props::AbsorptionKind::RECOGNISED[..],
                _ => &props::ScatterKind::RECOGNISED[..],
            };
            for n in set {
                assert!(e.contains(n), "{gate}: must name `{n}`: {e}");
            }
        }
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

    /// SPEC-LIT §135.6's `viewfactor_still_refuses_absorption`: §50.9 is
    /// unchanged, the two entries that describe a medium are refused BEFORE
    /// the enclosure is read, and a participating case is refused naming its
    /// reader.
    #[test]
    fn viewfactor_still_refuses_absorption() {
        let _g = contract::permissive_test_guard();
        contract::set_permissive(false);

        // a. A non-zero `absorptionCoefficient` is still §50.9's refusal.
        let dir = case_with(
            "radiationModel viewFactor;\nemissivity 0.8;\nabsorptionCoefficient 0.2;\n",
            "vf_abs",
        );
        let e = RadiationConfig::from_case(&dir).err().map(|e| e.to_string());
        let _ = std::fs::remove_dir_all(&dir);
        let e = e.expect("a non-zero kappa under viewFactor must be refused");
        assert!(e.contains("NON-PARTICIPATING"), "{e}");

        // b. An `absorptionEmissionModel` other than `none` describes a
        //    medium, and there is no medium to describe.
        let dir = case_with(
            "radiationModel viewFactor;\nemissivity 0.8;\nabsorptionEmissionModel constant;\n",
            "vf_aem",
        );
        let e = RadiationConfig::from_case(&dir).err().map(|e| e.to_string());
        let _ = std::fs::remove_dir_all(&dir);
        let e = e.expect("absorptionEmissionModel constant under viewFactor");
        assert!(e.contains("absorptionEmissionModel"), "{e}");

        // c. So is a `scatterModel` other than `none`.
        let dir = case_with(
            "radiationModel viewFactor;\nemissivity 0.8;\nscatterModel isotropic;\n",
            "vf_sct",
        );
        let e = RadiationConfig::from_case(&dir).err().map(|e| e.to_string());
        let _ = std::fs::remove_dir_all(&dir);
        let e = e.expect("scatterModel isotropic under viewFactor");
        assert!(e.contains("scatterModel"), "{e}");

        // d. Explicit `none`s are accepted, and the enclosure still reads.
        let dir = case_with(
            "radiationModel viewFactor;\nemissivity 0.8;\nabsorptionCoefficient 0;\n\
             absorptionEmissionModel none;\nscatterModel none;\n",
            "vf_none",
        );
        let cfg = RadiationConfig::from_case(&dir);
        let _ = std::fs::remove_dir_all(&dir);
        match cfg.expect("explicit `none`s must read") {
            RadiationConfig::S2s(c) => assert_eq!(c.emissivity, 0.8 as Scalar),
        }

        // e. A P1 case does not resolve here: this reader names the reader a
        //    participating medium belongs to, and that reader reads the same
        //    file without complaint.
        let dir = case_with("radiationModel P1;\nabsorptionCoefficient 0.5;\n", "vf_p1");
        let e = RadiationConfig::from_case(&dir).err().map(|e| e.to_string());
        let pc = props::ParticipatingConfig::from_case(&dir);
        let _ = std::fs::remove_dir_all(&dir);
        let e = e.expect("a P1 case must not resolve in the enclosure reader");
        assert!(e.contains("P1"), "{e}");
        assert!(e.contains("participating"), "{e}");
        assert!(e.contains("ParticipatingConfig"), "{e}");
        let pc = pc.expect("the same file is a valid participating medium");
        assert_eq!(pc.model, RadiationModel::P1);
        assert_eq!(pc.absorption, props::Absorption::Constant { kappa: 0.5 });
        assert_eq!(pc.scatter, props::Scatter::None);
    }
}
