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
//! cells - the P1 approximation and discrete ordinates are the two models
//! that name, and **this engine carries neither**. If it does not, there is
//! no field to solve for at all: the surfaces exchange directly across a
//! transparent gap and the unknown is one radiosity per boundary face - that
//! is S49/S50's surface-to-surface model, [`crate::s2s`], and it is what is
//! here.
//!
//! Three things are common to both, and they are what is here:
//!
//! * [`SIGMA_SB`], the Stefan-Boltzmann constant. Every radiative flux in
//!   this crate is proportional to it and none of them care which model
//!   produced the temperature it multiplies.
//! * [`RadiationModel`], S13.4's name gate. It RECOGNISES all three names a
//!   case may write, whether or not this build carries the solver behind
//!   each, because "recognised, and here is what it needs" is a better error
//!   than "unknown setting" - which is the whole of what S13.4 asks for. Two
//!   of the three are recognised here and resolved nowhere, and that is not
//!   an oversight: a case written for a participating medium has to be told
//!   what it asked for.
//! * [`RadiationConfig`], what a case file resolves to for the models that
//!   need no participating medium.
//!
//! What is NOT here is any solver, any kernel, and any of the entries a
//! medium is described by. This reader takes `emissivity`, `agglomerate` and
//! `quadrature` out of `constant/radiationProperties` and nothing else;
//! `absorptionCoefficient`, `chiR`, `spectralModel` and `openBoundary`
//! describe a medium, are read by no one here, and a case that sets them has
//! selected a model this engine does not carry. It is told so by name, and
//! nothing is substituted.
//!
//! Provenance: ORIGINAL. The Stefan-Boltzmann constant is CODATA 2018's
//! exact value, fixed by the SI redefinition of the kelvin; the selector,
//! its three names and its refusals are this project's own S13.4 contract.
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

/// Every radiation model this crate can name, and what happens when a case
/// asks for one it cannot run.
///
/// [`RadiationModel::P1`] and [`RadiationModel::FvDom`] are the two
/// participating-medium models, and no solver in this engine implements
/// either; [`RadiationModel::ViewFactor`] (SPEC-LIT S49/S50) is the
/// surface-to-surface one, [`crate::s2s`]. Asking for anything else is the
/// S13.4 contract - an error naming these three as what is RECOGNISED, not a
/// silent substitution.
///
/// Recognising a name is not the same as being able to resolve it: which
/// reader owns which model is [`RadiationConfig::from_case`]'s question, and
/// a name recognised here and refused there is exactly the error S13.4 wants
/// in place of "unknown setting".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadiationModel {
    /// The P1 differential approximation. Recognised, and carried by no
    /// solver in this engine.
    P1,
    /// Finite-volume discrete ordinates. Recognised, and carried by no solver
    /// in this engine.
    FvDom,
    /// Grey diffuse surface-to-surface exchange through a NON-participating
    /// medium - SPEC-LIT S49/S50, [`crate::s2s`]. The right model for an
    /// enclosure with nothing in it to absorb, emit or scatter, which is
    /// exactly where P1 and fvDOM are wrong.
    ViewFactor,
}

impl RadiationModel {
    /// Parse a `radiationModel` entry, per SPEC-LIT S13.4: recognised and
    /// implemented -> use it; recognised, not implemented -> an error naming
    /// the alternative; anything else -> an error naming the setting.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "P1" => Ok(Self::P1),
            "fvDOM" => Ok(Self::FvDom),
            // SPEC-LIT S49/S50. `viewFactor` is OpenFOAM's spelling of the
            // selector and `s2s` is the native one; both name one model.
            "viewFactor" | "s2s" => Ok(Self::ViewFactor),
            other => contract::unsupported(
                "radiationModel",
                other,
                &["P1", "fvDOM", "viewFactor"],
                "P1",
                Self::P1,
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
}

// ==========================================================================
//  S13.4  Which radiation model, resolved to its properties
// ==========================================================================

/// What `constant/radiationProperties` resolves to for a model that needs no
/// PARTICIPATING medium - exactly one, [`crate::s2s`]'s grey diffuse
/// surface-to-surface exchange (SPEC-LIT S49/S50).
///
/// An enum with one arm rather than a bare [`crate::s2s::S2sConfig`], because
/// what [`Self::from_case`] returns is "the radiation model this case
/// selected", and [`RadiationModel`] has three names to select between. The
/// arm carries no absorption coefficient deliberately: a surface-to-surface
/// enclosure has no participating medium, and `S2sConfig::from_dict` refuses
/// a case that sets one rather than reading it and dropping it (§50.9).
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

    /// Read `radiationModel` from `constant/radiationProperties` and, for a
    /// model with no medium, the entries that describe the enclosure -
    /// SPEC-LIT S13.4's own gate ([`RadiationModel::from_name`]) and then
    /// [`crate::s2s::S2sConfig::from_dict`].
    ///
    /// The two PARTICIPATING models are refused here, and the refusal is an
    /// [`Error::Config`] rather than an `io::contract` note. A note can be
    /// waived by `-permissive`, and waiving this one would hand a case that
    /// asked for a radiating GAS an enclosure of bare surfaces instead. That
    /// is the S13.4 substitution this project exists to refuse, and it is
    /// not a preference a command-line flag gets to overrule.
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
        let model = RadiationModel::from_name(d.get_or("radiationModel", "P1"))?;
        match model {
            RadiationModel::ViewFactor => Ok(Self::S2s(crate::s2s::S2sConfig::from_dict(&d)?)),
            RadiationModel::P1 | RadiationModel::FvDom => Err(Error::Config(format!(
                "{}: radiationModel {} is a PARTICIPATING-medium model - it \
                 solves a radiation field on the CELLS out of \
                 `absorptionCoefficient`, `chiR` and a spectral model, and \
                 this reader reads none of those. What it reads is the model \
                 that needs no medium at all: SPEC-LIT S49/S50's \
                 `viewFactor`, surfaces exchanging across a transparent gap. \
                 The name is recognised (SPEC-LIT S13.4) and no solver in \
                 this engine resolves it. Nothing is substituted - an \
                 enclosure of surfaces is not a gas, and being handed one for \
                 the other is the defect S13.4 exists to stop",
                p.display(),
                model.name()
            ))),
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

    /// SPEC-LIT S13.4: all three names are RECOGNISED here, which is what
    /// lets a refusal say what the case asked for instead of "unknown
    /// setting". Whether the solver behind a name is in this build is a
    /// separate question, answered by whichever reader owns it.
    #[test]
    fn every_model_name_is_recognised() {
        assert_eq!(RadiationModel::from_name("P1").unwrap(), RadiationModel::P1);
        assert_eq!(RadiationModel::from_name("fvDOM").unwrap(), RadiationModel::FvDom);
        assert_eq!(
            RadiationModel::from_name("viewFactor").unwrap(),
            RadiationModel::ViewFactor
        );
        // OpenFOAM's spelling and the native one name ONE model.
        assert_eq!(RadiationModel::from_name("s2s").unwrap(), RadiationModel::ViewFactor);
    }

    #[test]
    fn an_unknown_model_is_refused() {
        let _guard = crate::io::contract::permissive_test_guard();
        contract::set_permissive(false);
        let e = RadiationModel::from_name("banana").unwrap_err();
        assert!(e.to_string().contains("banana"));
    }

    /// SPEC-LIT S13.4, and the reason the refusal is an `Error::Config`:
    /// this runs with `-permissive` ON and is still refused. A note would
    /// have been waived here, and a case that asked for a radiating gas
    /// would have been handed an enclosure of bare surfaces.
    #[test]
    fn a_participating_model_is_recognised_and_refused_here() {
        let _guard = crate::io::contract::permissive_test_guard();
        contract::set_permissive(true);
        let mut got = Vec::new();
        for (name, tag) in [("P1", "p1"), ("fvDOM", "fvdom")] {
            let dir = case_with(
                &format!("radiationModel {name};\nabsorptionCoefficient 0.1;\n"),
                tag,
            );
            got.push((name, RadiationConfig::from_case(&dir).err().map(|e| e.to_string())));
            let _ = std::fs::remove_dir_all(&dir);
        }
        // The flag is process-wide: restore it before any assertion can
        // panic and leave a later strict-mode test observing it.
        contract::set_permissive(false);
        for (name, m) in got {
            let m = m.unwrap_or_else(|| panic!("{name} was resolved by the wrong reader"));
            assert!(m.contains(name), "the refusal must name what was asked for: {m}");
            assert!(m.contains("PARTICIPATING"), "{m}");
            assert!(m.contains("viewFactor"), "the refusal must name what IS here: {m}");
        }
    }
}
