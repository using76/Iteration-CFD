// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! What a participating medium is made of - SPEC-LIT §137.2's DESIGN table
//! for `constant/radiationProperties` under `radiationModel P1` or `fvDOM`:
//! the absorption/emission model, its coefficient data, and scattering.
//!
//! [`super::RadiationConfig`] resolves `viewFactor` and refuses these
//! models; this is the reader a P1 or fvDOM case is meant to reach. It
//! parses and refuses, and solves nothing: the RTE solvers are CHR-18 (P1)
//! and CHR-19 up (fvDOM), and until §137.3's units land a participating case
//! has no driver - which is why the enclosure reader refuses by name instead
//! of reading half of it.
//!
//! Every number is read as f64 in BOTH builds, straight from
//! [`crate::io::dict::last_number`] and never through [`crate::Scalar`] -
//! §137.2's rule, so `wsggPressure 0.1` matches the §136.4 table exactly
//! under `--features single` too.
//!
//! Provenance: ORIGINAL. The ten [`WSGG_PRESSURES_ATM`] values are Bordbar
//! et al. 2021's file set exactly as SPEC-LIT §136.4 lists it; no
//! coefficient data is carried (those files' own licence is non-commercial,
//! no distribution, so a case points at a user-supplied path). The entry
//! spellings and every refusal are this project's own §137.2 contract.
//! No GPL-licensed source was consulted.

use std::path::{Path, PathBuf};

use super::RadiationModel;
use crate::error::{Error, Result};
use crate::io::contract;
use crate::io::dict::{last_number, FoamDict};

/// Bordbar et al. 2021's ten discrete total pressures, in atm - the ONLY
/// values `wsggPressure` may take (SPEC-LIT §136.4: the model exists at ten
/// pressures and is never interpolated in pressure).
pub const WSGG_PRESSURES_ATM: [f64; 10] =
    [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0];

// ==========================================================================
//  The two name gates
// ==========================================================================

/// Which absorption/emission model a case names (SPEC-LIT §137.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbsorptionKind {
    /// A grey constant `absorptionCoefficient`.
    Constant,
    /// Bordbar et al. 2021's pressure-dependent WSGG (SPEC-LIT §136.4).
    Wsgg,
    /// `wsggGrey`: the same set used as ONE grey mean (§136.4).
    WsggGrey,
}

impl AbsorptionKind {
    /// Every name this gate recognises (SPEC-LIT §137.2).
    pub const RECOGNISED: [&'static str; 3] = ["constant", "wsgg", "wsggGrey"];

    /// Parse an `absorptionEmissionModel` entry, per S13.4: recognised ->
    /// the model; anything else -> an error naming the setting and what IS
    /// recognised.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "constant" => Ok(Self::Constant),
            "wsgg" => Ok(Self::Wsgg),
            "wsggGrey" => Ok(Self::WsggGrey),
            other => contract::unsupported(
                "radiationProperties/absorptionEmissionModel",
                other,
                &Self::RECOGNISED,
                "constant",
                Self::Constant,
            ),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::Wsgg => "wsgg",
            Self::WsggGrey => "wsggGrey",
        }
    }
}

/// Which scattering model a case names - §135.1's two built phase functions
/// and their absence (SPEC-LIT §137.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterKind {
    /// No scattering.
    None,
    /// `Φ = 1`.
    Isotropic,
    /// `Φ = 1 + A_1 Ω'·Ω`, non-negative only for `|A_1| <= 1`.
    LinearAnisotropic,
}

impl ScatterKind {
    /// Every name this gate recognises (SPEC-LIT §137.2).
    pub const RECOGNISED: [&'static str; 3] = ["none", "isotropic", "linearAnisotropic"];

    /// Parse a `scatterModel` entry, per S13.4, as
    /// [`AbsorptionKind::from_name`] does.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "none" => Ok(Self::None),
            "isotropic" => Ok(Self::Isotropic),
            "linearAnisotropic" => Ok(Self::LinearAnisotropic),
            other => contract::unsupported(
                "radiationProperties/scatterModel",
                other,
                &Self::RECOGNISED,
                "none",
                Self::None,
            ),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Isotropic => "isotropic",
            Self::LinearAnisotropic => "linearAnisotropic",
        }
    }
}

// ==========================================================================
//  The values the entries read to
// ==========================================================================

/// The absorption/emission a case selected, with its data (SPEC-LIT §137.2).
#[derive(Debug, Clone, PartialEq)]
pub enum Absorption {
    /// A constant grey `κ`, 1/m.
    Constant {
        /// The absorption coefficient, 1/m, non-negative and finite.
        kappa: f64,
    },
    /// The pressure-dependent WSGG set (SPEC-LIT §136.4).
    Wsgg {
        /// The user-supplied coefficient file, relative to the case
        /// directory as parsed; [`ParticipatingConfig::from_case`] resolves
        /// it and requires it to exist.
        file: PathBuf,
        /// The total pressure the file was fitted at, in atm - one of
        /// [`WSGG_PRESSURES_ATM`], matched exactly.
        pressure_atm: f64,
    },
    /// The same set used as ONE grey mean (SPEC-LIT §136.4).
    WsggGrey {
        /// As [`Absorption::Wsgg::file`].
        file: PathBuf,
        /// As [`Absorption::Wsgg::pressure_atm`].
        pressure_atm: f64,
    },
}

impl Absorption {
    pub fn kind(&self) -> AbsorptionKind {
        match self {
            Self::Constant { .. } => AbsorptionKind::Constant,
            Self::Wsgg { .. } => AbsorptionKind::Wsgg,
            Self::WsggGrey { .. } => AbsorptionKind::WsggGrey,
        }
    }
}

/// The scattering a case selected, with its phase function (SPEC-LIT §137.2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scatter {
    /// No scattering.
    None,
    /// `Φ = 1`.
    Isotropic {
        /// `σ_s`, 1/m, non-negative and finite.
        sigma: f64,
    },
    /// `Φ = 1 + A_1 Ω'·Ω` (SPEC-LIT §135.1).
    LinearAnisotropic {
        /// `σ_s`, 1/m.
        sigma: f64,
        /// `A_1`, with `|A_1| <= 1` - beyond that the phase function would
        /// be negative for some pair of directions, which is refused.
        a1: f64,
    },
}

impl Scatter {
    pub fn kind(&self) -> ScatterKind {
        match self {
            Self::None => ScatterKind::None,
            Self::Isotropic { .. } => ScatterKind::Isotropic,
            Self::LinearAnisotropic { .. } => ScatterKind::LinearAnisotropic,
        }
    }

    /// `σ_s`, 1/m; `0` where there is no scattering.
    pub fn sigma(&self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Isotropic { sigma } | Self::LinearAnisotropic { sigma, .. } => *sigma,
        }
    }

    /// `A_1` of the phase function; `0` unless linear-anisotropic.
    pub fn a1(&self) -> f64 {
        match self {
            Self::LinearAnisotropic { a1, .. } => *a1,
            _ => 0.0,
        }
    }
}

// ==========================================================================
//  The reader
// ==========================================================================

/// What `constant/radiationProperties` resolves to for a PARTICIPATING
/// medium - SPEC-LIT §137.2's table, read.
#[derive(Debug, Clone, PartialEq)]
pub struct ParticipatingConfig {
    /// The model the case named. `P1` or `fvDOM` - a non-participating
    /// model is refused, because this is the reader of a participating
    /// medium.
    pub model: RadiationModel,
    pub absorption: Absorption,
    pub scatter: Scatter,
}

impl ParticipatingConfig {
    /// Read §137.2's entries out of a `radiationProperties` dictionary for
    /// the model `model`. Every refusal names the entry it is about, and
    /// every number is f64 in both builds (§137.2's rule).
    #[allow(clippy::too_many_lines)]
    pub fn from_dict(d: &FoamDict, model: RadiationModel) -> Result<Self> {
        if !model.is_participating() {
            return Err(Error::Config(format!(
                "radiationModel {} is not a participating medium; this reader \
                 is for P1 and fvDOM, and a viewFactor case is resolved by \
                 RadiationConfig::from_case (SPEC-LIT §137.2)",
                model.name()
            )));
        }
        let absorption = Self::absorption_from_dict(d)?;
        let scatter = Self::scatter_from_dict(d)?;
        // §137.2: under P1, a constant medium with κ + σ_s = 0 makes
        // (135.7)'s Γ infinite. fvDOM accepts such a medium: the
        // discrete-ordinates row of a transparent medium is finite.
        if model == RadiationModel::P1 {
            if let Absorption::Constant { kappa } = absorption {
                if kappa + scatter.sigma() == 0.0 {
                    return Err(Error::Config(format!(
                        "radiationModel P1 with a constant medium at \
                         κ + σ_s = {} is refused: (135.7)'s Γ = 1/(3β - A1 σ_s) \
                         is then infinite (SPEC-LIT §135.3); radiationModel \
                         fvDOM accepts such a medium",
                        kappa + scatter.sigma()
                    )));
                }
            }
        }
        Ok(Self { model, absorption, scatter })
    }

    /// §137.2's absorption table: `constant` needs a `κ`;
    /// `wsgg`/`wsggGrey` need their file and pressure and refuse a `κ`.
    fn absorption_from_dict(d: &FoamDict) -> Result<Absorption> {
        let kind = AbsorptionKind::from_name(d.get_or("absorptionEmissionModel", "constant"))?;
        Ok(match kind {
            AbsorptionKind::Constant => {
                // The wsgg entries belong to wsgg/wsggGrey; with `constant`
                // they would be read and then ignored (SPEC-LIT §137.2).
                for entry in ["wsggFile", "wsggPressure"] {
                    if d.has(entry) {
                        return Err(Error::Config(format!(
                            "radiationProperties/{entry} is present with \
                             absorptionEmissionModel constant, where it would \
                             be read and then ignored (SPEC-LIT §137.2)"
                        )));
                    }
                }
                let Some(kappa) = entry_number(d, "absorptionCoefficient")? else {
                    return Err(Error::Config(
                        "radiationProperties: absorptionEmissionModel constant \
                         needs an `absorptionCoefficient` entry - κ has no \
                         honest default (SPEC-LIT §137.2)"
                            .into(),
                    ));
                };
                if !kappa.is_finite() || kappa < 0.0 {
                    return Err(Error::Config(format!(
                        "radiationProperties/absorptionCoefficient: {kappa} is \
                         refused - κ must be non-negative and finite (SPEC-LIT \
                         §137.2)"
                    )));
                }
                Absorption::Constant { kappa }
            }
            AbsorptionKind::Wsgg | AbsorptionKind::WsggGrey => {
                if d.has("absorptionCoefficient") {
                    return Err(Error::Config(
                        "radiationProperties/absorptionCoefficient is present \
                         with absorptionEmissionModel wsgg/wsggGrey, where it \
                         would be read and then ignored - the coefficients come \
                         from the file (SPEC-LIT §137.2)"
                            .into(),
                    ));
                }
                let file = entry_file(d)?;
                let pressure_atm = wsgg_pressure(d)?;
                if kind == AbsorptionKind::Wsgg {
                    Absorption::Wsgg { file, pressure_atm }
                } else {
                    Absorption::WsggGrey { file, pressure_atm }
                }
            }
        })
    }

    /// §137.2's scatter table, entry by entry: `σ_s` is required with
    /// `isotropic` and `linearAnisotropic`, `A_1` only with the latter, and
    /// an entry the case did not select would be read and then ignored.
    fn scatter_from_dict(d: &FoamDict) -> Result<Scatter> {
        let kind = ScatterKind::from_name(d.get_or("scatterModel", "none"))?;
        Ok(match kind {
            ScatterKind::None => {
                if d.has("scatterCoefficient") {
                    return Err(Error::Config(
                        "radiationProperties/scatterCoefficient is present with \
                         scatterModel none, where it would be read and then \
                         ignored (SPEC-LIT §137.2)"
                            .into(),
                    ));
                }
                if d.has("A1") {
                    return Err(Error::Config(
                        "radiationProperties/A1 is present with scatterModel \
                         none - A1 belongs to the linear-anisotropic phase \
                         function (SPEC-LIT §137.2)"
                            .into(),
                    ));
                }
                Scatter::None
            }
            ScatterKind::Isotropic => {
                let sigma = required_number(d, "scatterCoefficient", "scatterModel isotropic")?;
                refuse_negative(sigma, "scatterCoefficient", "σ_s")?;
                if let Some(a1) = entry_number(d, "A1")? {
                    return Err(Error::Config(format!(
                        "radiationProperties/A1: {a1} is present with \
                         scatterModel isotropic, where Φ = 1 and it would be \
                         read and then ignored (SPEC-LIT §137.2)"
                    )));
                }
                Scatter::Isotropic { sigma }
            }
            ScatterKind::LinearAnisotropic => {
                let sigma =
                    required_number(d, "scatterCoefficient", "scatterModel linearAnisotropic")?;
                refuse_negative(sigma, "scatterCoefficient", "σ_s")?;
                let a1 = required_number(d, "A1", "scatterModel linearAnisotropic")?;
                if !a1.is_finite() || a1.abs() > 1.0 {
                    return Err(Error::Config(format!(
                        "radiationProperties/A1: {a1} is refused - |A1| > 1 \
                         makes the phase function Φ = 1 + A1 Ω'·Ω negative for \
                         some pair of directions (SPEC-LIT §135.1)"
                    )));
                }
                Scatter::LinearAnisotropic { sigma, a1 }
            }
        })
    }

    /// Read `constant/radiationProperties` of the case directory for a
    /// participating medium: [`RadiationModel::from_name`], then
    /// [`Self::from_dict`], then the `wsggFile` resolved against the case
    /// directory and required to exist (SPEC-LIT §136.4: the coefficient
    /// data is not vendored and is read from the user's own path).
    pub fn from_case(case_dir: &Path) -> Result<Self> {
        let p = case_dir.join("constant").join("radiationProperties");
        if !p.exists() {
            return Err(Error::Config(format!(
                "{} does not exist; SPEC-LIT §137.2 selects a radiation model \
                 by name in that file and there is no default for it",
                p.display()
            )));
        }
        let d = FoamDict::read(&p)?;
        if !d.has("radiationModel") {
            return Err(Error::Config(format!(
                "{}: no `radiationModel` entry. SPEC-LIT §137.2 selects the \
                 model by name and there is no default for it; recognised: {}",
                p.display(),
                RadiationModel::RECOGNISED.join(", ")
            )));
        }
        let model = RadiationModel::from_name(d.get_or("radiationModel", ""))?;
        let mut cfg = Self::from_dict(&d, model)?;
        let rel = match &cfg.absorption {
            Absorption::Wsgg { file, .. } | Absorption::WsggGrey { file, .. } => file.clone(),
            Absorption::Constant { .. } => return Ok(cfg),
        };
        let full = case_dir.join(&rel);
        if !full.exists() {
            return Err(Error::Config(format!(
                "radiationProperties/wsggFile: '{}' does not exist (case \
                 directory '{}') - the Bordbar coefficient data is not vendored \
                 and is read from the user's own path (SPEC-LIT §136.4)",
                rel.display(),
                case_dir.display()
            )));
        }
        if let Absorption::Wsgg { file, .. } | Absorption::WsggGrey { file, .. } =
            &mut cfg.absorption
        {
            *file = full;
        }
        Ok(cfg)
    }
}

// ==========================================================================
//  Entry readers
// ==========================================================================

/// A numeric entry's value: the last whitespace-separated token that parses
/// as a number, as f64 in BOTH builds (SPEC-LIT §137.2). `None` = the entry
/// is absent; an entry with no number in it is refused, naming the entry.
fn entry_number(d: &FoamDict, entry: &str) -> Result<Option<f64>> {
    match d.get(entry) {
        None => Ok(None),
        Some(raw) => match last_number(raw) {
            Some(v) => Ok(Some(v)),
            None => Err(Error::Config(format!(
                "radiationProperties/{entry}: \"{raw}\" holds no number, and \
                 the entry is read as one (SPEC-LIT §137.2)"
            ))),
        },
    }
}

/// [`entry_number`] where the entry is required - the refusal says what
/// needs it.
fn required_number(d: &FoamDict, entry: &str, needs: &str) -> Result<f64> {
    entry_number(d, entry)?.ok_or_else(|| {
        Error::Config(format!(
            "radiationProperties: {needs} needs a `{entry}` entry - there is \
             no honest default (SPEC-LIT §137.2)"
        ))
    })
}

/// A coefficient that must be non-negative and finite (SPEC-LIT §137.2).
fn refuse_negative(v: f64, entry: &str, greek: &str) -> Result<()> {
    if !v.is_finite() || v < 0.0 {
        return Err(Error::Config(format!(
            "radiationProperties/{entry}: {v} is refused - {greek} must be \
             non-negative and finite (SPEC-LIT §137.2)"
        )));
    }
    Ok(())
}

/// The `wsggFile` entry: the trimmed raw value with surrounding `"` stripped,
/// empty refused (SPEC-LIT §137.2).
fn entry_file(d: &FoamDict) -> Result<PathBuf> {
    let stripped = d.get_or("wsggFile", "").trim().trim_matches('"');
    if stripped.is_empty() {
        return Err(Error::Config(
            "radiationProperties: absorptionEmissionModel wsgg/wsggGrey needs a \
             `wsggFile` entry naming the user's own coefficient file - there is \
             no default, and the data is not vendored (SPEC-LIT §136.4)"
                .into(),
        ));
    }
    Ok(PathBuf::from(stripped))
}

/// The `wsggPressure` entry: one of [`WSGG_PRESSURES_ATM`], matched exactly;
/// any other value is refused naming the ten (SPEC-LIT §136.4 - no
/// interpolation in pressure).
fn wsgg_pressure(d: &FoamDict) -> Result<f64> {
    let v = required_number(d, "wsggPressure", "absorptionEmissionModel wsgg/wsggGrey")?;
    if !WSGG_PRESSURES_ATM.contains(&v) {
        let ten: Vec<String> = WSGG_PRESSURES_ATM.iter().map(|p| format!("{p}")).collect();
        return Err(Error::Config(format!(
            "radiationProperties/wsggPressure: {v} atm is refused - the model \
             exists at ten discrete total pressures and is never interpolated \
             in pressure (SPEC-LIT §136.4): {}",
            ten.join(", ")
        )));
    }
    Ok(v)
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(src: &str) -> FoamDict {
        FoamDict::parse(src, "radiationProperties").expect("parse")
    }

    /// A case directory carrying one `radiationProperties`, distinct by tag,
    /// removed by the caller.
    fn case_with(body: &str, tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ofgpu_radiation_props_{tag}"));
        let c = dir.join("constant");
        std::fs::create_dir_all(&c).expect("mkdir");
        std::fs::write(c.join("radiationProperties"), body).expect("write");
        dir
    }

    fn msg(r: Result<ParticipatingConfig>) -> String {
        r.err().expect("this case must be refused").to_string()
    }

    /// SPEC-LIT §137.2's absorption table, entry by entry: what each model
    /// parses to, and every refusal naming the entry it is about.
    #[test]
    fn absorption_model_refusals() {
        // `constant` with a κ, nothing else: the smallest participating
        // case.
        let c = ParticipatingConfig::from_dict(
            &parsed("absorptionCoefficient 0.5;\n"),
            RadiationModel::P1,
        )
        .expect("a plain constant medium reads");
        assert_eq!(c.model, RadiationModel::P1);
        assert_eq!(c.absorption, Absorption::Constant { kappa: 0.5 });
        assert_eq!(c.scatter, Scatter::None);
        assert_eq!(c.absorption.kind(), AbsorptionKind::Constant);
        assert_eq!(c.scatter.kind(), ScatterKind::None);
        assert_eq!(c.absorption.kind().name(), "constant");

        // κ is required with `constant`, negative and unreadable refused,
        // each naming the entry.
        let e = msg(ParticipatingConfig::from_dict(&parsed(""), RadiationModel::FvDom));
        assert!(e.contains("absorptionCoefficient"), "{e}");
        let e = msg(ParticipatingConfig::from_dict(
            &parsed("absorptionCoefficient -0.1;\n"),
            RadiationModel::P1,
        ));
        assert!(e.contains("absorptionCoefficient"), "{e}");
        let e = msg(ParticipatingConfig::from_dict(
            &parsed("absorptionCoefficient abc;\n"),
            RadiationModel::P1,
        ));
        assert!(e.contains("absorptionCoefficient"), "{e}");

        // wsgg: the file with its quotes stripped, and the pressure read as
        // f64 (`1` matches the table's 1.0 exactly).
        let c = ParticipatingConfig::from_dict(
            &parsed(
                "absorptionEmissionModel wsgg;\nwsggFile \"bordbar.dat\";\nwsggPressure 1;\n",
            ),
            RadiationModel::P1,
        )
        .expect("wsgg reads");
        assert_eq!(
            c.absorption,
            Absorption::Wsgg { file: PathBuf::from("bordbar.dat"), pressure_atm: 1.0 }
        );
        assert_eq!(c.absorption.kind(), AbsorptionKind::Wsgg);

        // wsggGrey under fvDOM; the pressure must come back as the f64 0.1,
        // never read through Scalar.
        let c = ParticipatingConfig::from_dict(
            &parsed(
                "absorptionEmissionModel wsggGrey;\nwsggFile x.dat;\nwsggPressure 0.1;\n",
            ),
            RadiationModel::FvDom,
        )
        .expect("wsggGrey reads");
        match c.absorption {
            Absorption::WsggGrey { file, pressure_atm } => {
                assert_eq!(file, PathBuf::from("x.dat"));
                assert_eq!(pressure_atm, 0.1_f64);
            }
            other => panic!("expected wsggGrey, got {other:?}"),
        }

        // A pressure off the table is refused naming the ten.
        let e = msg(ParticipatingConfig::from_dict(
            &parsed("absorptionEmissionModel wsgg;\nwsggFile x.dat;\nwsggPressure 3;\n"),
            RadiationModel::P1,
        ));
        assert!(e.contains("wsggPressure"), "{e}");
        assert!(
            e.contains("0.1, 0.5, 1, 2, 5, 10, 20, 30, 50, 80"),
            "the refusal must name the ten: {e}"
        );

        // wsgg without its file; with its file but no pressure.
        let e = msg(ParticipatingConfig::from_dict(
            &parsed("absorptionEmissionModel wsgg;\nwsggPressure 1;\n"),
            RadiationModel::P1,
        ));
        assert!(e.contains("wsggFile"), "{e}");
        let e = msg(ParticipatingConfig::from_dict(
            &parsed("absorptionEmissionModel wsgg;\nwsggFile x.dat;\n"),
            RadiationModel::P1,
        ));
        assert!(e.contains("wsggPressure"), "{e}");

        // Each entry the case did not select would be read and then ignored:
        // κ with wsgg, wsggPressure with `constant`.
        let e = msg(ParticipatingConfig::from_dict(
            &parsed(
                "absorptionEmissionModel wsgg;\nwsggFile x.dat;\nwsggPressure 1;\n\
                 absorptionCoefficient 0.5;\n",
            ),
            RadiationModel::P1,
        ));
        assert!(e.contains("absorptionCoefficient"), "{e}");
        let e = msg(ParticipatingConfig::from_dict(
            &parsed("absorptionCoefficient 0.5;\nwsggPressure 1;\n"),
            RadiationModel::P1,
        ));
        assert!(e.contains("wsggPressure"), "{e}");

        // A transparent medium: refused under P1 ((135.7)'s Γ is then
        // infinite), fine under fvDOM.
        let e = msg(ParticipatingConfig::from_dict(
            &parsed("absorptionCoefficient 0;\n"),
            RadiationModel::P1,
        ));
        assert!(e.contains("P1"), "{e}");
        assert!(e.contains("135.7"), "{e}");
        let c = ParticipatingConfig::from_dict(
            &parsed("absorptionCoefficient 0;\n"),
            RadiationModel::FvDom,
        )
        .expect("fvDOM accepts a transparent medium");
        assert_eq!(c.absorption, Absorption::Constant { kappa: 0.0 });

        // ... unless scattering makes β > 0.
        let c = ParticipatingConfig::from_dict(
            &parsed(
                "absorptionCoefficient 0;\nscatterModel isotropic;\nscatterCoefficient 0.2;\n",
            ),
            RadiationModel::P1,
        )
        .expect("κ + σ_s = 0.2 is not transparent");
        assert_eq!(c.scatter.sigma(), 0.2);

        // A non-participating model has no business in this reader.
        let e = msg(ParticipatingConfig::from_dict(
            &parsed("absorptionCoefficient 0.5;\n"),
            RadiationModel::ViewFactor,
        ));
        assert!(e.contains("RadiationConfig"), "{e}");

        // from_case on real files: a valid case, the missing file, a file
        // naming no model, and a `wsggFile` that is not on disk.
        let dir = case_with("radiationModel P1;\nabsorptionCoefficient 0.5;\n", "props_ok");
        let c = ParticipatingConfig::from_case(&dir).expect("a valid P1 case reads");
        assert_eq!(c.model, RadiationModel::P1);
        let e = msg(ParticipatingConfig::from_case(&dir.join("nowhere")));
        assert!(e.contains("does not exist"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);

        let dir = case_with("absorptionCoefficient 0.5;\n", "props_nomodel");
        let e = msg(ParticipatingConfig::from_case(&dir));
        assert!(e.contains("radiationModel"), "{e}");
        assert!(e.contains("no default"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);

        let dir = case_with(
            "radiationModel fvDOM;\nabsorptionEmissionModel wsgg;\n\
             wsggFile gone.dat;\nwsggPressure 1;\n",
            "props_wsgg",
        );
        let e = msg(ParticipatingConfig::from_case(&dir));
        assert!(e.contains("wsggFile"), "{e}");
        assert!(e.contains("does not exist"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// SPEC-LIT §135.6's `scatter_refuses_negative_phase`: `|A_1| > 1` would
    /// make Φ negative for some pair of directions, and the rest of §137.2's
    /// scatter table holds entry by entry.
    #[test]
    fn scatter_refuses_negative_phase() {
        // Every src below is prefixed with a κ that keeps the medium
        // participating under P1.
        const K: &str = "absorptionCoefficient 0.5;\n";
        let p1 = |src: &str| {
            ParticipatingConfig::from_dict(&parsed(&format!("{K}{src}")), RadiationModel::P1)
        };

        // |A1| > 1 refused, at both ends, naming A1 and the reason.
        let e = msg(p1("scatterModel linearAnisotropic;\nscatterCoefficient 0.3;\nA1 1.5;\n"));
        assert!(e.contains("A1"), "{e}");
        assert!(e.contains("negative"), "{e}");
        let e = msg(p1("scatterModel linearAnisotropic;\nscatterCoefficient 0.3;\nA1 -1.01;\n"));
        assert!(e.contains("A1"), "{e}");

        // |A1| <= 1 reads, at both ends of the range.
        let c = p1("scatterModel linearAnisotropic;\nscatterCoefficient 0.3;\nA1 1;\n")
            .expect("A1 = 1 is the limit");
        assert_eq!(c.scatter, Scatter::LinearAnisotropic { sigma: 0.3, a1: 1.0 });
        assert!(p1("scatterModel linearAnisotropic;\nscatterCoefficient 0.3;\nA1 -1;\n").is_ok());
        let c = p1("scatterModel linearAnisotropic;\nscatterCoefficient 0.3;\nA1 0.5;\n")
            .expect("A1 = 0.5 reads");
        assert_eq!(c.scatter.a1(), 0.5);
        assert_eq!(c.scatter.sigma(), 0.3);

        // A1 is required with linearAnisotropic...
        let e = msg(p1("scatterModel linearAnisotropic;\nscatterCoefficient 0.3;\n"));
        assert!(e.contains("A1"), "{e}");

        // ...σ_s is required with isotropic...
        let e = msg(p1("scatterModel isotropic;\n"));
        assert!(e.contains("scatterCoefficient"), "{e}");

        // ...and A1 is refused with `isotropic`.
        let e = msg(p1("scatterModel isotropic;\nscatterCoefficient 0.3;\nA1 0.5;\n"));
        assert!(e.contains("A1"), "{e}");

        // Negative σ_s refused; σ_s = 0.3 under isotropic reads, with a1()
        // still 0.
        let e = msg(p1("scatterModel isotropic;\nscatterCoefficient -0.3;\n"));
        assert!(e.contains("scatterCoefficient"), "{e}");
        let c = p1("scatterModel isotropic;\nscatterCoefficient 0.3;\n")
            .expect("isotropic reads");
        assert_eq!(c.scatter, Scatter::Isotropic { sigma: 0.3 });
        assert_eq!(c.scatter.a1(), 0.0);

        // σ_s with no scatterModel would be read and then ignored.
        let e = msg(p1("scatterCoefficient 0.3;\n"));
        assert!(e.contains("scatterCoefficient"), "{e}");

        // An unrecognised scatter name is refused with the set beside it,
        // in strict mode.
        let _g = contract::permissive_test_guard();
        contract::set_permissive(false);
        let e = ScatterKind::from_name("mie")
            .err()
            .expect("mie is not a scatter model")
            .to_string();
        assert!(e.contains("mie"), "{e}");
        for n in ScatterKind::RECOGNISED {
            assert!(e.contains(n), "must name `{n}`: {e}");
        }
    }
}
