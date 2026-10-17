// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The `radiation` case block - JSONC `radiation` and
//! `constant/radiationProperties` - lowered onto one
//! [`ParticipatingConfig`] (SPEC-LIT §137.2) and, against the mesh, onto
//! the per-patch conditions `ofgpu-lowmach`'s P1 and fvDOM solves read
//! (SPEC-LIT §137.8).
//!
//! One lowering for both routes: the JSONC block is written out as the
//! entries of §137.2's dictionary table, each number in Rust's shortest
//! round-trip form, and read by `ParticipatingConfig::from_dict`, so both
//! routes share every refusal of §137.2. The dictionary route's own extras -
//! `solverFreq` (which is `every`), `quadrature`, `scatterTol` and
//! `maxScatterIterations` - are read here, and the walls come from the
//! radiation field file, `0/G` under P1 and `0/IDefault` under fvDOM, when
//! the case has one. [`LoweredRadiation::p1_walls`] and
//! [`LoweredRadiation::dom_patches`] apply §137.8's patch table over the
//! mesh's `PatchKind`s.
//!
//! Written from:
//!   ofgpu SPEC-LIT.md §137.2 - the `radiation` table and its refusals
//!   ofgpu SPEC-LIT.md §137.8 - the as-built rules of this file
//!
//! No GPL-licensed source was consulted.

use std::collections::BTreeMap;
use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::io::dict::FoamDict;
use crate::io::fields::{read_scalar_field, PatchFieldSpec};
use crate::mesh::{HostMesh, PatchKind};
use crate::radiation::dom::{DomCondition, DomPatch};
use crate::radiation::dom_quad::Quadrature;
use crate::radiation::p1::{MarshakPatch, WallTemperature};
use crate::radiation::props::{self, ParticipatingConfig};
use crate::radiation::RadiationModel;

/// The dictionary route's file inside a case directory.
pub const FILE: &str = "constant/radiationProperties";

/// The `radiation` block. An absent block is bitwise the current run; the
/// JSONC route has no `none` - an absent block IS no radiation (SPEC-LIT
/// §137.8).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct JsonRadiation {
    /// `P1` | `fvDOM` | `s2s` | `viewFactor`; the last two name CHR-24's
    /// enclosure model and are refused here by name.
    pub model: String,
    /// The absorption/emission the medium carries; `constant` by default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub absorption: Option<JsonRadAbsorption>,
    /// The scattering; `none` by default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scatter: Option<JsonRadScatter>,
    /// fvDOM only: `S2` | `S4` | `S6` | `S8`; `S4` by default. Refused under
    /// P1, where it would be read and then ignored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quadrature: Option<String>,
    /// fvDOM only: the sweep loop's tolerance on `change` (§136.7).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scatter_tol: Option<f64>,
    /// fvDOM only: the sweep cap (§136.2's `maxScatterIterations`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_scatter_iterations: Option<i64>,
    /// Solve when the outer-iteration count `i mod every = 0`; 1 by default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub every: Option<i64>,
    /// Radiating walls by patch name; a wall not named is black at its own
    /// `T` (SPEC-LIT §137.8).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub walls: BTreeMap<String, JsonRadWall>,
    /// Open boundaries by patch name, each at its stated `Tinf`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub open: BTreeMap<String, JsonRadOpen>,
}

/// `radiation.absorption` - SPEC-LIT §137.2's absorption table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct JsonRadAbsorption {
    /// `constant` (default) | `wsgg` | `wsggGrey`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The grey absorption coefficient, 1/m; required with `constant`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kappa: Option<f64>,
    /// Path to a Bordbar 2021 `.dat`; required with `wsgg`/`wsggGrey`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// atm, one of Bordbar 2021's ten; required with `wsgg`/`wsggGrey`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pressure: Option<f64>,
}

/// `radiation.scatter` - SPEC-LIT §137.2's scatter table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct JsonRadScatter {
    /// `none` (default) | `isotropic` | `linearAnisotropic`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The scattering coefficient, 1/m; required with the two scattering
    /// models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sigma: Option<f64>,
    /// The linear-anisotropic phase function's `A_1`; required with
    /// `linearAnisotropic`, `|A1| <= 1`.
    #[serde(rename = "A1", default, skip_serializing_if = "Option::is_none")]
    pub a1: Option<f64>,
}

/// One entry of `radiation.walls`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct JsonRadWall {
    /// The grey diffuse emissivity, in `[0, 1]`; `0` is refused later, by
    /// `P1::new`/`DomRows::new`, which need `(0, 1]` (SPEC-LIT §137.8).
    pub emissivity: f64,
    /// The stated wall temperature, K; absent, the wall radiates at its own
    /// `T` patch value.
    #[serde(rename = "T", default, skip_serializing_if = "Option::is_none")]
    pub t: Option<f64>,
}

/// One entry of `radiation.open`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct JsonRadOpen {
    /// The ambient temperature the boundary is black at, K.
    #[serde(rename = "Tinf")]
    pub t_inf: f64,
}

/// One lowered radiating wall, as the case stated it.
#[derive(Debug, Clone, PartialEq)]
pub struct RadWall {
    pub patch: String,
    pub emissivity: f64,
    /// The stated `T`; `None` is the wall's own `T` patch value.
    pub t: Option<f64>,
}

/// One lowered open boundary, as the case stated it.
#[derive(Debug, Clone, PartialEq)]
pub struct RadOpen {
    pub patch: String,
    pub t_inf: f64,
}

/// One patch's condition read from the radiation field file (`0/G` under P1,
/// `0/IDefault` under fvDOM), resolved to its numbers - SPEC-LIT §137.8's
/// four names. The wall/ordinary split of a `greyDiffusiveRadiation` entry
/// is applied against the mesh, in [`LoweredRadiation::p1_walls`] /
/// [`LoweredRadiation::dom_patches`], which hold the `PatchKind`s.
#[derive(Debug, Clone, PartialEq)]
pub enum RadPatchType {
    /// A grey diffuse wall condition: `MarshakRadiation` (P1, `t: None`),
    /// `MarshakRadiationFixedTemperature` (P1, `t: Some(Trad)`) or
    /// `greyDiffusiveRadiation` (fvDOM, either).
    Wall(RadWall),
    /// An open boundary at a stated temperature.
    Open(RadOpen),
    /// `symmetry`/`symmetryPlane`; the name is the patch's.
    Symmetry(String),
    /// `empty`; the name is the patch's.
    Empty(String),
}

/// What [`JsonRadiation::lower`] and [`read_radiation_dir`] produce: the
/// config both routes share, the fvDOM solve's settings, the stated walls
/// and open boundaries, and - dictionary route only - the radiation field
/// file's per-patch types.
#[derive(Debug, Clone, PartialEq)]
pub struct LoweredRadiation {
    pub config: ParticipatingConfig,
    /// fvDOM only; `None` -> `S4` at build time (SPEC-LIT §137.8).
    pub quadrature: Option<String>,
    /// fvDOM only; `None` -> `dom::DEFAULT_SWEEP_TOL`.
    pub scatter_tol: Option<f64>,
    /// fvDOM only; `None` -> `dom::DEFAULT_MAX_SWEEPS`.
    pub max_sweeps: Option<usize>,
    /// Solve when `i mod every = 0`; `>= 1`, 1 by default.
    pub every: usize,
    /// Only the patches the case stated.
    pub walls: Vec<RadWall>,
    /// Only the patches the case stated.
    pub open: Vec<RadOpen>,
    /// Dictionary route only: per-patch types read from `0/G` or
    /// `0/IDefault` (patch -> resolved condition).
    pub field_file: Option<Vec<RadPatchType>>,
}

/// The refusal every driver but `ofgpu-lowmach` gives a case that states
/// `radiation` (SPEC-LIT §137.8). `source` names where the case stated it.
#[must_use]
pub fn refuse_radiation_case(source: &str) -> Error {
    Error::Refused(format!(
        "{source}: this case states radiation. ofgpu-lowmach is the one driver that \
         runs it (CHR-23); every other driver refuses the case rather than run it \
         without the source its energy equation is asking for (SPEC-LIT §137.8)"
    ))
}

fn keyed(key: &str, e: Error) -> Error {
    match e {
        Error::Config(m) => Error::Config(format!("radiation.{key}: {m}")),
        Error::Refused(m) => Error::Refused(format!("radiation.{key}: {m}")),
        other => other,
    }
}

fn refused(key: &str, msg: &str) -> Error {
    Error::Config(format!("radiation.{key}: {msg} (SPEC-LIT §137.2)"))
}

/// A temperature a case states: K, positive and finite (SPEC-LIT §137.2).
fn checked_t(key: &str, t: f64) -> Result<f64> {
    if !t.is_finite() || t <= 0.0 {
        return Err(refused(key, &format!("{t} is refused - a radiation temperature must be positive and finite")));
    }
    Ok(t)
}

/// An emissivity a case states: `[0, 1]` at the lowering; `0` is refused
/// later, by `P1::new`/`DomRows::new` (SPEC-LIT §137.8).
fn checked_emissivity(key: &str, e: f64) -> Result<f64> {
    if !e.is_finite() || !(0.0..=1.0).contains(&e) {
        return Err(refused(key, &format!("{e} is refused - an emissivity must lie in [0, 1]")));
    }
    Ok(e)
}

impl JsonRadiation {
    /// The one lowering of both routes (SPEC-LIT §137.2, §137.8): the block
    /// is written out as §137.2's dictionary entries and read by
    /// `ParticipatingConfig::from_dict`, so both routes share every refusal.
    pub fn lower(&self) -> Result<LoweredRadiation> {
        let model = match RadiationModel::from_name(&self.model) {
            Ok(RadiationModel::ViewFactor) => {
                return Err(refused(
                    "model",
                    &format!(
                        "{} is CHR-24's surface-to-surface enclosure in a \
                         non-participating medium, not built in ofgpu-lowmach \
                         (SPEC-LIT §137.8)",
                        self.model
                    ),
                ));
            }
            Ok(m) => m,
            Err(e) => return Err(keyed("model", e)),
        };
        let d = self.dict_text()?;
        let config = ParticipatingConfig::from_dict(&d, model)?;

        // The fvDOM-only settings: under P1 they would be read and then
        // ignored, which is the §13.4.1 defect every reader here refuses.
        let fvdom = model == RadiationModel::FvDom;
        for (key, set) in [
            ("quadrature", self.quadrature.is_some()),
            ("scatterTol", self.scatter_tol.is_some()),
            ("maxScatterIterations", self.max_scatter_iterations.is_some()),
        ] {
            if set && !fvdom {
                return Err(refused(
                    key,
                    &format!(
                        "is read by fvDOM only, and model is {} - it would be \
                         read and then ignored",
                        model.name()
                    ),
                ));
            }
        }
        let quadrature = match &self.quadrature {
            None => None,
            Some(q) => {
                Quadrature::from_name(q).map_err(|e| keyed("quadrature", e))?;
                Some(q.clone())
            }
        };
        let scatter_tol = match self.scatter_tol {
            None => None,
            Some(v) if !v.is_finite() || v <= 0.0 => {
                return Err(refused(
                    "scatterTol",
                    &format!("{v} is refused - the sweep tolerance must be positive and finite"),
                ));
            }
            Some(v) => Some(v),
        };
        let max_sweeps = match self.max_scatter_iterations {
            None => None,
            Some(v) if v < 1 => {
                return Err(refused("maxScatterIterations", &format!("{v} is refused - the sweep cap must be at least 1")));
            }
            Some(v) => Some(v as usize),
        };
        let every = match self.every {
            None => 1,
            Some(v) if v < 1 => {
                return Err(refused("every", &format!("{v} is refused - the solve schedule must be at least every 1st iteration")));
            }
            Some(v) => v as usize,
        };
        let mut walls = Vec::with_capacity(self.walls.len());
        for (name, w) in &self.walls {
            checked_emissivity(&format!("walls.{name}.emissivity"), w.emissivity)?;
            if let Some(t) = w.t {
                checked_t(&format!("walls.{name}.T"), t)?;
            }
            walls.push(RadWall { patch: name.clone(), emissivity: w.emissivity, t: w.t });
        }
        let mut open = Vec::with_capacity(self.open.len());
        for (name, o) in &self.open {
            checked_t(&format!("open.{name}.Tinf"), o.t_inf)?;
            open.push(RadOpen { patch: name.clone(), t_inf: o.t_inf });
        }
        Ok(LoweredRadiation {
            config,
            quadrature,
            scatter_tol,
            max_sweeps,
            every,
            walls,
            open,
            field_file: None,
        })
    }

    /// §137.2's dictionary entries, written from the block's keys - the one
    /// lowering both routes share (SPEC-LIT §137.8).
    fn dict_text(&self) -> Result<FoamDict> {
        let mut t = format!("radiationModel {};\n", self.model);
        if let Some(a) = &self.absorption {
            t.push_str(&format!(
                "absorptionEmissionModel {};\n",
                a.model.clone().unwrap_or_else(|| "constant".to_string())
            ));
            if let Some(k) = a.kappa {
                t.push_str(&format!("absorptionCoefficient {k};\n"));
            }
            if let Some(f) = &a.file {
                t.push_str(&format!("wsggFile \"{f}\";\n"));
            }
            if let Some(p) = a.pressure {
                t.push_str(&format!("wsggPressure {p};\n"));
            }
        }
        if let Some(s) = &self.scatter {
            t.push_str(&format!(
                "scatterModel {};\n",
                s.model.clone().unwrap_or_else(|| "none".to_string())
            ));
            if let Some(v) = s.sigma {
                t.push_str(&format!("scatterCoefficient {v};\n"));
            }
            if let Some(v) = s.a1 {
                t.push_str(&format!("A1 {v};\n"));
            }
        }
        FoamDict::parse(&t, "radiation (JSONC lowering)")
    }
}

/// The dictionary route: `constant/radiationProperties` when the case has
/// one, `None` when it is absent or says `radiationModel none` (SPEC-LIT
/// §137.8). The walls come from the radiation field file, `0/G` under P1 and
/// `0/IDefault` under fvDOM, when it exists.
pub fn read_radiation_dir(case_dir: &Path) -> Result<Option<LoweredRadiation>> {
    let p = case_dir.join(FILE);
    if !p.exists() {
        return Ok(None);
    }
    let d = FoamDict::read(&p)?;
    let name = d.get_or("radiationModel", "").trim();
    if name == "none" {
        return Ok(None);
    }
    if !d.has("radiationModel") {
        return Err(Error::Config(format!(
            "{FILE}: no `radiationModel` entry - SPEC-LIT §137.2 selects the \
             model by name and there is no default; recognised: {} (`none` is \
             the no-radiation case)",
            RadiationModel::RECOGNISED.join(", ")
        )));
    }
    let model = match RadiationModel::from_name(name) {
        Ok(RadiationModel::ViewFactor) => {
            return Err(Error::Config(format!(
                "{FILE}/radiationModel {name}: s2s/viewFactor is CHR-24's \
                 enclosure model, not built in ofgpu-lowmach (SPEC-LIT §137.8)"
            )));
        }
        Ok(m) => m,
        Err(e) => return Err(e),
    };
    let mut config = ParticipatingConfig::from_dict(&d, model)?;
    // `wsggFile` is relative to the case directory and required to exist,
    // exactly as `ParticipatingConfig::from_case` resolves it.
    if let props::Absorption::Wsgg { file, .. } | props::Absorption::WsggGrey { file, .. } =
        &mut config.absorption
    {
        let full = case_dir.join(&*file);
        if !full.exists() {
            return Err(Error::Config(format!(
                "{FILE}/wsggFile: '{}' does not exist (case directory '{}') - \
                 the Bordbar coefficient data is not vendored and is read from \
                 the user's own path (SPEC-LIT §136.4)",
                file.display(),
                case_dir.display()
            )));
        }
        *file = full;
    }
    let fvdom = model == RadiationModel::FvDom;
    // The entries beyond §137.2's table: `solverFreq` (which is `every`),
    // `quadrature`, `scatterTol` and `maxScatterIterations` (SPEC-LIT §137.8).
    let every = match crate::io::dict::last_number(d.get("solverFreq").unwrap_or("1")) {
        None => return Err(Error::Config(format!(
            "{FILE}/solverFreq: \"{}\" holds no number, and the entry is read as one (SPEC-LIT §137.2)",
            d.get("solverFreq").unwrap_or("")
        ))),
        Some(v) if v < 1.0 || v.fract() != 0.0 => {
            return Err(Error::Config(format!(
                "{FILE}/solverFreq: {v} is refused - the solve schedule is a whole number of iterations, at least 1 (SPEC-LIT §137.2)"
            )));
        }
        Some(v) => v as usize,
    };
    let quadrature = match word(&d, "quadrature")? {
        None => None,
        Some(q) => {
            if !fvdom {
                return Err(Error::Config(format!(
                    "{FILE}/quadrature: it is read by fvDOM only, and radiationModel \
                     {} - it would be read and then ignored (SPEC-LIT §137.2, §13.4.1)",
                    model.name()
                )));
            }
            Quadrature::from_name(&q).map_err(|e| keyed("quadrature", e))?;
            Some(q)
        }
    };
    let read_num = |entry: &str| -> Result<Option<f64>> {
        match d.get(entry) {
            None => Ok(None),
            Some(raw) => match crate::io::dict::last_number(raw) {
                Some(v) => Ok(Some(v)),
                None => Err(Error::Config(format!(
                    "{FILE}/{entry}: \"{raw}\" holds no number, and the entry is read as one (SPEC-LIT §137.2)"
                ))),
            },
        }
    };
    let scatter_tol = match read_num("scatterTol")? {
        None => None,
        Some(v) => {
            if !fvdom {
                return Err(Error::Config(format!(
                    "{FILE}/scatterTol: it is read by fvDOM only, and radiationModel \
                     {} - it would be read and then ignored (SPEC-LIT §137.2, §13.4.1)",
                    model.name()
                )));
            }
            if !v.is_finite() || v <= 0.0 {
                return Err(Error::Config(format!(
                    "{FILE}/scatterTol: {v} is refused - the sweep tolerance must be positive and finite (SPEC-LIT §137.2)"
                )));
            }
            Some(v)
        }
    };
    let max_sweeps = match read_num("maxScatterIterations")? {
        None => None,
        Some(v) => {
            if !fvdom {
                return Err(Error::Config(format!(
                    "{FILE}/maxScatterIterations: it is read by fvDOM only, and \
                     radiationModel {} - it would be read and then ignored \
                     (SPEC-LIT §137.2, §13.4.1)",
                    model.name()
                )));
            }
            if v < 1.0 || v.fract() != 0.0 {
                return Err(Error::Config(format!(
                    "{FILE}/maxScatterIterations: {v} is refused - the sweep cap is a whole number, at least 1 (SPEC-LIT §137.2)"
                )));
            }
            Some(v as usize)
        }
    };
    let field_file = read_field_file(case_dir, model)?;
    Ok(Some(LoweredRadiation {
        config,
        quadrature,
        scatter_tol,
        max_sweeps,
        every,
        walls: Vec::new(),
        open: Vec::new(),
        field_file,
    }))
}

/// The one whitespace-separated word an entry holds - `quadrature S4;`.
fn word(d: &FoamDict, entry: &str) -> Result<Option<String>> {
    match d.get(entry) {
        None => Ok(None),
        Some(raw) => match raw.split_whitespace().next() {
            Some(w) => Ok(Some(w.trim_matches(';').to_string())),
            None => Err(Error::Config(format!(
                "{FILE}/{entry}: \"{raw}\" holds no name, and the entry is read as one (SPEC-LIT §137.2)"
            ))),
        },
    }
}

/// The radiation field file of `model` - `0/G` under P1, `0/IDefault` under
/// fvDOM - when the case has one (SPEC-LIT §137.8). Only its `boundaryField`
/// is read: four type names, resolved to their numbers.
fn read_field_file(case_dir: &Path, model: RadiationModel) -> Result<Option<Vec<RadPatchType>>> {
    let rel = match model {
        RadiationModel::P1 => "0/G",
        _ => "0/IDefault",
    };
    let p = case_dir.join(rel);
    if !p.exists() {
        return Ok(None);
    }
    let raw = read_scalar_field(&p, 0)?;
    let mut out = Vec::new();
    for (name, spec) in &raw.boundary {
        out.push(patch_type_from_entry(model, name, spec)?);
    }
    Ok(Some(out))
}

/// One `boundaryField` entry of the radiation field file, resolved to a
/// [`RadPatchType`] (SPEC-LIT §137.8's four names). The model mismatch - a P1
/// name in `0/IDefault` or the reverse - is refused here, where the file's
/// model is known.
fn patch_type_from_entry(model: RadiationModel, name: &str, spec: &PatchFieldSpec) -> Result<RadPatchType> {
    const WHERE: &str = "radiation field file";
    if name.starts_with('"') {
        return Err(Error::Config(format!(
            "{WHERE}: the quoted pattern {name} is not a patch name - the \
             radiation file's boundaryField names every patch exactly \
             (SPEC-LIT §137.8)"
        )));
    }
    let refuses_extra = |allowed: &[&str]| -> Result<()> {
        for k in spec.extra.keys() {
            if !allowed.contains(&k.as_str()) {
                return Err(Error::Config(format!(
                    "{WHERE}: {name}/{k} is not an entry of {} - recognised: {} \
                     (SPEC-LIT §137.8)",
                    spec.type_name,
                    allowed.join(", ")
                )));
            }
        }
        Ok(())
    };
    let emissivity = |allowed: &[&str]| -> Result<f64> {
        refuses_extra(allowed)?;
        match spec.number("emissivity", name)? {
            None => Ok(1.0),
            Some(e) => checked_emissivity(&format!("{name}.emissivity"), f64::from(e)),
        }
    };
    match spec.type_name.as_str() {
        "symmetry" | "symmetryPlane" => {
            refuses_extra(&[])?;
            Ok(RadPatchType::Symmetry(name.to_string()))
        }
        "empty" => {
            refuses_extra(&[])?;
            Ok(RadPatchType::Empty(name.to_string()))
        }
        "MarshakRadiation" if model == RadiationModel::P1 => {
            let e = emissivity(&["emissivity"])?;
            Ok(RadPatchType::Wall(RadWall { patch: name.to_string(), emissivity: e, t: None }))
        }
        "MarshakRadiationFixedTemperature" if model == RadiationModel::P1 => {
            let e = emissivity(&["emissivity", "Trad"])?;
            let t = spec
                .number("Trad", name)?
                .map(f64::from)
                .map(|t| checked_t(&format!("{name}.Trad"), t))
                .transpose()?
                .ok_or_else(|| {
                    Error::Config(format!(
                        "{WHERE}: {name}/Trad is required - a \
                         MarshakRadiationFixedTemperature wall states its own \
                         temperature (SPEC-LIT §137.8)"
                    ))
                })?;
            Ok(RadPatchType::Wall(RadWall { patch: name.to_string(), emissivity: e, t: Some(t) }))
        }
        "greyDiffusiveRadiation" if model == RadiationModel::FvDom => {
            let e = emissivity(&["emissivity", "T"])?;
            let t = match spec.number("T", name)? {
                None => None,
                Some(t) => Some(checked_t(&format!("{name}.T"), f64::from(t))?),
            };
            Ok(RadPatchType::Wall(RadWall { patch: name.to_string(), emissivity: e, t }))
        }
        "wideBandDiffusiveRadiation" => Err(Error::Config(format!(
            "{WHERE}: {name} is greyDiffusiveRadiation, and \
             wideBandDiffusiveRadiation names a banded medium, which is named \
             and not built (SPEC-LIT §136.4, §137.8)"
        ))),
        "MarshakRadiation" | "MarshakRadiationFixedTemperature" | "greyDiffusiveRadiation" => Err(Error::Config(format!(
            "{WHERE}: {name} is {}, a {} condition, in the {} file - a model \
             mismatch (SPEC-LIT §137.8)",
            spec.type_name,
            if model == RadiationModel::P1 { "fvDOM" } else { "P1" },
            match model {
                RadiationModel::P1 => "P1 (0/G)",
                _ => "fvDOM (0/IDefault)",
            }
        ))),
        other => Err(Error::Config(format!(
            "{WHERE}: {name} carries the type {other:?} - the four names this \
             reader resolves are MarshakRadiation, \
             MarshakRadiationFixedTemperature (P1), greyDiffusiveRadiation \
             (fvDOM), and symmetry/symmetryPlane/empty (SPEC-LIT §137.8)"
        ))),
    }
}

/// One patch's resolved condition, before the P1/fvDOM split (SPEC-LIT
/// §137.8's patch table).
enum Resolved {
    /// A grey diffuse wall: emissivity, and `T` stated or from the field.
    Wall { emissivity: f64, t: Option<f64> },
    /// A boundary black at a stated temperature - an `open` entry, or a
    /// fixed-temperature condition on an ordinary patch.
    Black { t: f64 },
    /// A specular symmetry plane.
    Symmetry,
    /// Nothing: an `empty` patch, or one with no faces.
    Nothing,
}

fn no_patch(name: &str, key: &str) -> Error {
    Error::Config(format!(
        "radiation.{key}: no patch named \"{name}\" in the mesh (SPEC-LIT §137.8)"
    ))
}

impl LoweredRadiation {
    /// §137.8's patch table over `hm.patches`, shared by
    /// [`Self::p1_walls`] and [`Self::dom_patches`]: one [`Resolved`] per
    /// patch, with every refusal of the table and of the stated names. The
    /// field file's types govern where it exists; the stated `walls`/`open`
    /// where it does not.
    fn resolve(&self, hm: &HostMesh, model: RadiationModel) -> Result<Vec<(String, Resolved)>> {
        let walls: BTreeMap<&str, &RadWall> =
            self.walls.iter().map(|w| (w.patch.as_str(), w)).collect();
        let open: BTreeMap<&str, &RadOpen> =
            self.open.iter().map(|o| (o.patch.as_str(), o)).collect();
        let file: Option<BTreeMap<&str, &RadPatchType>> = self.field_file.as_ref().map(|v| {
            v.iter()
                .map(|t| match t {
                    RadPatchType::Wall(w) => (w.patch.as_str(), t),
                    RadPatchType::Open(o) => (o.patch.as_str(), t),
                    RadPatchType::Symmetry(n) | RadPatchType::Empty(n) => (n.as_str(), t),
                })
                .collect()
        });
        for w in &self.walls {
            if !hm.patches.iter().any(|p| p.name == w.patch) {
                return Err(no_patch(&w.patch, "walls"));
            }
            if open.contains_key(w.patch.as_str()) {
                return Err(Error::Config(format!(
                    "radiation: the patch \"{}\" is named in walls and in open - \
                     one patch carries one radiative condition (SPEC-LIT §137.8)",
                    w.patch
                )));
            }
        }
        for o in &self.open {
            if !hm.patches.iter().any(|p| p.name == o.patch) {
                return Err(no_patch(&o.patch, "open"));
            }
        }
        if let Some(f) = &file {
            for name in f.keys() {
                if !hm.patches.iter().any(|p| p.name == *name) {
                    return Err(no_patch(name, "field file"));
                }
            }
        }
        let open_on_wall = |name: &str| {
            Error::Config(format!(
                "radiation.open: \"{name}\" is a wall - an open boundary is an \
                 ordinary patch (SPEC-LIT §137.8)"
            ))
        };
        let walls_on_other = |name: &str| {
            Error::Config(format!(
                "radiation.walls: \"{name}\" is not a wall patch (SPEC-LIT §137.8)"
            ))
        };
        let refused_open = |name: &str, model: RadiationModel| {
            Error::Config(format!(
                "radiation: the ordinary patch \"{name}\" carries no radiative \
                 condition - an open boundary needs an entry in `open` under {} \
                 (SPEC-LIT §137.8)",
                model.name()
            ))
        };
        let needs_t = |name: &str, model: RadiationModel| {
            Error::Config(format!(
                "radiation field file: \"{name}\" is an ordinary patch, and its \
                 condition states no temperature - an open boundary needs one \
                 under {} (SPEC-LIT §137.8)",
                model.name()
            ))
        };
        let mut out = Vec::with_capacity(hm.patches.len());
        for p in &hm.patches {
            if p.size == 0 {
                if walls.contains_key(p.name.as_str()) {
                    return Err(walls_on_other(&p.name));
                }
                if open.contains_key(p.name.as_str()) {
                    return Err(open_on_wall(&p.name));
                }
                out.push((p.name.clone(), Resolved::Nothing));
                continue;
            }
            match p.kind {
                PatchKind::Wall => {
                    if let Some(o) = open.get(p.name.as_str()) {
                        return Err(open_on_wall(&o.patch));
                    }
                    if let Some(w) = walls.get(p.name.as_str()) {
                        out.push((p.name.clone(), Resolved::Wall { emissivity: w.emissivity, t: w.t }));
                    } else if let Some(t) = file.as_ref().and_then(|f| f.get(p.name.as_str())) {
                        out.push((p.name.clone(), file_on_patch(t)));
                    } else {
                        // Every wall not named is black at its own `T`.
                        out.push((p.name.clone(), Resolved::Wall { emissivity: 1.0, t: None }));
                    }
                }
                // A wedge face is an ordinary boundary to the rows, exactly
                // as DomRows::new and P1 treat it.
                PatchKind::Generic | PatchKind::Wedge => {
                    if let Some(w) = walls.get(p.name.as_str()) {
                        return Err(walls_on_other(&w.patch));
                    }
                    if let Some(o) = open.get(p.name.as_str()) {
                        out.push((p.name.clone(), Resolved::Black { t: o.t_inf }));
                    } else if let Some(t) = file.as_ref().and_then(|f| f.get(p.name.as_str())) {
                        match t {
                            RadPatchType::Wall(rw) => match rw.t {
                                None => return Err(needs_t(&p.name, model)),
                                Some(tv) => {
                                    if rw.emissivity != 1.0 {
                                        return Err(Error::Config(format!(
                                            "radiation field file: \"{}\" is an \
                                             ordinary patch and is black - its \
                                             emissivity must be 1, not {} \
                                             (SPEC-LIT §137.8)",
                                            p.name, rw.emissivity
                                        )));
                                    }
                                    out.push((p.name.clone(), Resolved::Black { t: tv }));
                                }
                            },
                            RadPatchType::Open(o) => out.push((p.name.clone(), Resolved::Black { t: o.t_inf })),
                            RadPatchType::Symmetry(_) => out.push((p.name.clone(), Resolved::Symmetry)),
                            RadPatchType::Empty(_) => out.push((p.name.clone(), Resolved::Nothing)),
                        }
                    } else {
                        return Err(refused_open(&p.name, model));
                    }
                }
                PatchKind::Symmetry => {
                    if walls.contains_key(p.name.as_str()) {
                        return Err(walls_on_other(p.name.as_str()));
                    }
                    if open.contains_key(p.name.as_str()) {
                        return Err(open_on_wall(p.name.as_str()));
                    }
                    out.push((p.name.clone(), Resolved::Symmetry));
                }
                PatchKind::Cyclic | PatchKind::Processor | PatchKind::Interface => {
                    return Err(Error::Config(format!(
                        "radiation: the patch \"{}\" is coupled (cyclic, \
                         processor or conjugate interface) and carries no \
                         radiative condition in v1 (SPEC-LIT §136.6, §137.8)",
                        p.name
                    )));
                }
                PatchKind::Empty => {
                    if walls.contains_key(p.name.as_str()) {
                        return Err(walls_on_other(&p.name));
                    }
                    if open.contains_key(p.name.as_str()) {
                        return Err(open_on_wall(&p.name));
                    }
                    out.push((p.name.clone(), Resolved::Nothing));
                }
            }
        }
        Ok(out)
    }

    /// The P1 solve's Marshak walls (SPEC-LIT §137.8's patch table, P1
    /// column). `G` carries nothing on a symmetry patch - it is
    /// zero-gradient there - and nothing on an `empty` one.
    pub fn p1_walls(&self, hm: &HostMesh) -> Result<Vec<MarshakPatch>> {
        Ok(self
            .resolve(hm, RadiationModel::P1)?
            .into_iter()
            .filter_map(|(patch, r)| match r {
                Resolved::Wall { emissivity, t } => Some(MarshakPatch {
                    patch,
                    emissivity,
                    temperature: t.map_or(WallTemperature::FromField, WallTemperature::Fixed),
                }),
                Resolved::Black { t } => Some(MarshakPatch {
                    patch,
                    emissivity: 1.0,
                    temperature: WallTemperature::Fixed(t),
                }),
                Resolved::Symmetry | Resolved::Nothing => None,
            })
            .collect())
    }

    /// The fvDOM rows' per-patch conditions (SPEC-LIT §137.8's patch table,
    /// fvDOM column).
    pub fn dom_patches(&self, hm: &HostMesh) -> Result<Vec<DomPatch>> {
        Ok(self
            .resolve(hm, RadiationModel::FvDom)?
            .into_iter()
            .filter_map(|(patch, r)| match r {
                Resolved::Wall { emissivity, t } => Some(DomPatch {
                    patch,
                    condition: DomCondition::GreyDiffuse {
                        emissivity,
                        temperature: t.map_or(WallTemperature::FromField, WallTemperature::Fixed),
                    },
                }),
                Resolved::Black { t } => Some(DomPatch { patch, condition: DomCondition::Open { t_inf: t } }),
                Resolved::Symmetry => Some(DomPatch { patch, condition: DomCondition::Symmetry }),
                Resolved::Nothing => None,
            })
            .collect())
    }
}

/// A field-file entry on a WALL patch: the type is what the file says.
fn file_on_patch(t: &RadPatchType) -> Resolved {
    match t {
        RadPatchType::Wall(rw) => Resolved::Wall { emissivity: rw.emissivity, t: rw.t },
        RadPatchType::Open(o) => Resolved::Black { t: o.t_inf },
        RadPatchType::Symmetry(_) => Resolved::Symmetry,
        RadPatchType::Empty(_) => Resolved::Nothing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockgen::{self, BlockSpec, GradedAxis};
    use std::path::PathBuf;

    /// The cube [0,1]^3, every side a wall named `xMin`/`xMax`/`yMin`/...
    fn cube(names: [&str; 6]) -> HostMesh {
        cube_of(names, ["wall"; 6])
    }

    /// The cube with per-side OpenFOAM boundary types.
    fn cube_of(names: [&str; 6], types: [&str; 6]) -> HostMesh {
        let axis = |lo: f64, hi: f64, n: usize| GradedAxis { lo, hi, n, expansion: 1.0, two_sided: false };
        let spec = BlockSpec {
            x: axis(0.0, 1.0, 3),
            y: axis(0.0, 1.0, 3),
            z: axis(0.0, 1.0, 3),
            patch_name: names.map(String::from),
            patch_type: types.map(String::from),
            windows: Vec::new(),
            cyclic: Vec::new(),
        };
        blockgen::build_mesh(&spec).expect("the cube builds")
    }

    fn parse(text: &str) -> Result<JsonRadiation> {
        serde_json::from_str(text).map_err(|e| Error::Config(format!("{e}")))
    }

    fn base(model: &str) -> String {
        format!(r#"{{ "model": "{model}", "absorption": {{ "kappa": 1.0 }} }}"#)
    }

    fn lower_text(text: &str) -> Result<LoweredRadiation> {
        parse(text)?.lower()
    }

    fn wants(text: &str, key: &str, want: &[&str]) {
        let e = lower_text(text).err().unwrap_or_else(|| panic!("{key}: expected a refusal"));
        let m = format!("{e}");
        for w in want {
            assert!(m.contains(w), "{key}: {w:?} not in {m}");
        }
    }

    #[test]
    fn case_radiation_refusals() {
        // The model, named with the recognised set (SPEC-LIT 137.2).
        wants(r#"{ "model": "P_one" }"#, "model", &["radiation.model", "P1", "fvDOM", "viewFactor"]);
        // s2s / viewFactor: CHR-24's, refused here by name (SPEC-LIT 137.8).
        for m in ["s2s", "viewFactor"] {
            wants(&base(m), "model", &[m, "CHR-24", "137.8"]);
        }
        // The absorption table.
        wants(
            r#"{ "model": "P1", "absorption": { "model": "greyMean" } }"#,
            "absorption.model",
            &["absorptionEmissionModel"],
        );
        wants(r#"{ "model": "P1", "absorption": {} }"#, "absorption.kappa", &["absorptionCoefficient"]);
        wants(
            r#"{ "model": "P1", "absorption": { "kappa": -1.0 } }"#,
            "absorption.kappa",
            &["absorptionCoefficient", "-1"],
        );
        wants(
            r#"{ "model": "P1", "absorption": { "model": "wsgg", "pressure": 1.0 } }"#,
            "absorption.file",
            &["wsggFile"],
        );
        wants(
            r#"{ "model": "P1", "absorption": { "model": "wsggGrey", "file": "b.dat" } }"#,
            "absorption.pressure",
            &["wsggPressure"],
        );
        wants(
            r#"{ "model": "fvDOM", "absorption": { "model": "wsgg", "file": "b.dat", "pressure": 0.13 } }"#,
            "absorption.pressure",
            &["wsggPressure", "0.13", "ten"],
        );
        wants(
            r#"{ "model": "fvDOM", "absorption": { "model": "wsgg", "kappa": 1.0, "file": "b.dat", "pressure": 1.0 } }"#,
            "absorption.kappa",
            &["absorptionCoefficient", "read and then ignored"],
        );
        // The scatter table.
        wants(
            r#"{ "model": "fvDOM", "absorption": { "kappa": 1.0 }, "scatter": { "model": "henyey" } }"#,
            "scatter.model",
            &["scatterModel"],
        );
        wants(
            r#"{ "model": "fvDOM", "absorption": { "kappa": 1.0 }, "scatter": { "sigma": 0.1 } }"#,
            "scatter.sigma",
            &["scatterCoefficient", "read and then ignored"],
        );
        wants(
            r#"{ "model": "fvDOM", "absorption": { "kappa": 1.0 }, "scatter": { "A1": 0.1 } }"#,
            "scatter.A1",
            &["A1", "scatterModel none", "137.2"],
        );
        wants(
            r#"{ "model": "fvDOM", "absorption": { "kappa": 1.0 }, "scatter": { "model": "isotropic" } }"#,
            "scatter.sigma",
            &["scatterCoefficient"],
        );
        wants(
            r#"{ "model": "fvDOM", "absorption": { "kappa": 1.0 }, "scatter": { "model": "linearAnisotropic", "sigma": 0.1 } }"#,
            "scatter.A1",
            &["A1"],
        );
        wants(
            r#"{ "model": "fvDOM", "absorption": { "kappa": 1.0 }, "scatter": { "model": "linearAnisotropic", "sigma": 0.1, "A1": 1.5 } }"#,
            "scatter.A1",
            &["|A1|", "1.5"],
        );
        wants(
            r#"{ "model": "fvDOM", "absorption": { "kappa": 1.0 }, "scatter": { "model": "isotropic", "sigma": 0.1, "A1": 0.2 } }"#,
            "scatter.A1",
            &["isotropic"],
        );
        // P1 refuses a transparent constant medium: (135.7)'s Gamma.
        wants(r#"{ "model": "P1", "absorption": { "kappa": 0.0 } }"#, "absorption.kappa", &["P1", "infinite"]);
        // The fvDOM-only settings, refused under P1 (read and ignored).
        for (key, extra) in [
            ("quadrature", r#""quadrature": "S4""#),
            ("scatterTol", r#""scatterTol": 1e-8"#),
            ("maxScatterIterations", r#""maxScatterIterations": 200"#),
        ] {
            let text = format!(
                r#"{{ "model": "P1", "absorption": {{ "kappa": 1.0 }}, {extra} }}"#
            );
            wants(&text, key, &[key, "fvDOM only", "read and then ignored"]);
        }
        wants(
            r#"{ "model": "fvDOM", "absorption": { "kappa": 1.0 }, "quadrature": "S12" }"#,
            "quadrature",
            &["S12", "negative weights"],
        );
        wants(
            r#"{ "model": "fvDOM", "absorption": { "kappa": 1.0 }, "scatterTol": 0.0 }"#,
            "scatterTol",
            &["positive and finite"],
        );
        wants(
            r#"{ "model": "fvDOM", "absorption": { "kappa": 1.0 }, "scatterTol": -1e-9 }"#,
            "scatterTol",
            &["positive and finite"],
        );
        wants(
            r#"{ "model": "fvDOM", "absorption": { "kappa": 1.0 }, "maxScatterIterations": 0 }"#,
            "maxScatterIterations",
            &["at least 1"],
        );
        wants(
            r#"{ "model": "P1", "absorption": { "kappa": 1.0 }, "every": 0 }"#,
            "every",
            &["at least every 1st"],
        );
        // The walls and open entries.
        wants(
            r#"{ "model": "P1", "absorption": { "kappa": 1.0 }, "walls": { "hot": { "emissivity": 1.2 } } }"#,
            "walls.hot.emissivity",
            &["[0, 1]"],
        );
        wants(
            r#"{ "model": "P1", "absorption": { "kappa": 1.0 }, "walls": { "hot": { "emissivity": -0.1 } } }"#,
            "walls.hot.emissivity",
            &["[0, 1]"],
        );
        wants(
            r#"{ "model": "P1", "absorption": { "kappa": 1.0 }, "walls": { "hot": { "emissivity": 0.5, "T": 0.0 } } }"#,
            "walls.hot.T",
            &["positive and finite"],
        );
        wants(
            r#"{ "model": "fvDOM", "absorption": { "kappa": 1.0 }, "open": { "amb": { "Tinf": -3.0 } } }"#,
            "open.amb.Tinf",
            &["positive and finite"],
        );
        // A key the block does not define is refused by the reader.
        wants(
            r#"{ "model": "P1", "absorption": { "kappa": 1.0 }, "qRad": 1.0 }"#,
            "unknown key",
            &["qRad"],
        );
    }

    /// The lowering with every key set, read back field by field; then the
    /// serde round trip through the shortest JSON the struct serialises to.
    #[test]
    fn every_key_lowers_and_round_trips() {
        let text = r#"{
            "model": "fvDOM",
            "absorption": { "model": "constant", "kappa": 0.37 },
            "scatter": { "model": "linearAnisotropic", "sigma": 0.11, "A1": 0.3 },
            "quadrature": "S6",
            "scatterTol": 1e-9,
            "maxScatterIterations": 40,
            "every": 2,
            "walls": { "hot": { "emissivity": 0.8, "T": 450.0 }, "cold": { "emissivity": 1.0 } },
            "open": { "amb": { "Tinf": 300.0 } }
        }"#;
        let j: JsonRadiation = serde_json::from_str(text).expect("parse");
        let l = j.lower().expect("lowers");
        assert_eq!(l.config.model, RadiationModel::FvDom);
        match l.config.absorption {
            props::Absorption::Constant { kappa } => assert_eq!(kappa, 0.37),
            other => panic!("{other:?}"),
        }
        assert_eq!(l.config.scatter.sigma(), 0.11);
        assert_eq!(l.config.scatter.a1(), 0.3);
        assert_eq!(l.quadrature.as_deref(), Some("S6"));
        assert_eq!(l.scatter_tol, Some(1e-9));
        assert_eq!(l.max_sweeps, Some(40));
        assert_eq!(l.every, 2);
        assert_eq!(l.walls.len(), 2);
        assert_eq!(l.walls[0].patch, "cold");
        assert_eq!(l.walls[0].emissivity, 1.0);
        assert_eq!(l.walls[0].t, None);
        assert_eq!(l.walls[1].patch, "hot");
        assert_eq!(l.walls[1].emissivity, 0.8);
        assert_eq!(l.walls[1].t, Some(450.0));
        assert_eq!(l.open.len(), 1);
        assert_eq!(l.open[0].t_inf, 300.0);
        assert!(l.field_file.is_none());

        // Round trip: serialise, parse again, compare.
        let s = serde_json::to_string(&j).expect("serialise");
        let back: JsonRadiation = serde_json::from_str(&s).expect("re-parse");
        assert_eq!(back, j);
    }

    const CUBE_WALLS: [&str; 6] = ["xMin", "xMax", "yMin", "yMax", "zMin", "zMax"];

    fn fvdom() -> String {
        r#"{ "model": "fvDOM", "absorption": { "kappa": 1.0 }, "quadrature": "S4" }"#.to_string()
    }

    /// §137.8's patch table, JSONC route: stated walls and open entries
    /// resolve per patch kind, and every mis-stated name is refused by name.
    #[test]
    fn patch_table_resolves_and_refuses_by_name() {
        let hm = cube(CUBE_WALLS);
        // Every wall named: xmin stated, the rest default black.
        let l = lower_text(&format!(
            r#"{{ "model": "P1", "absorption": {{ "kappa": 1.0 }}, "walls": {{ "xMin": {{ "emissivity": 0.8, "T": 450.0 }} }} }}"#
        ))
        .expect("lowers");
        let w = l.p1_walls(&hm).expect("the walls resolve");
        assert_eq!(w.len(), 6, "every wall patch carries a Marshak condition");
        assert_eq!(w[0].patch, "xMin");
        assert_eq!(w[0].emissivity, 0.8);
        assert_eq!(w[0].temperature, WallTemperature::Fixed(450.0));
        assert_eq!(w[1].temperature, WallTemperature::FromField);
        assert_eq!(w[1].emissivity, 1.0);

        // fvDOM: the same stated block, GreyDiffuse.
        let l = lower_text(&format!(
            r#"{{ "model": "fvDOM", "absorption": {{ "kappa": 1.0 }}, "walls": {{ "xMin": {{ "emissivity": 0.5 }} }} }}"#
        ))
        .expect("lowers");
        let p = l.dom_patches(&hm).expect("the patches resolve");
        assert!(matches!(
            &p[0].condition,
            DomCondition::GreyDiffuse { emissivity: e, temperature: WallTemperature::FromField } if *e == 0.5
        ));

        // An inlet patch must be named in `open`; a wall entry on it is
        // refused, and so is an open entry on a wall.
        let mut types = ["wall"; 6];
        types[1] = "patch";
        let hm = cube_of(CUBE_WALLS, types);
        let l = lower_text(&fvdom()).expect("lowers");
        let e = l.dom_patches(&hm).err().expect("an unnamed ordinary patch is refused");
        let m = format!("{e}");
        assert!(m.contains("xMax") && m.contains("open") && m.contains("137.8"), "{m}");
        let l = lower_text(&format!(
            r#"{{ "model": "fvDOM", "absorption": {{ "kappa": 1.0 }}, "walls": {{ "xMax": {{ "emissivity": 1.0 }} }} }}"#
        ))
        .expect("lowers");
        let m = format!("{}", l.p1_walls(&hm).err().expect("a walls entry on an ordinary patch is refused"));
        assert!(m.contains("xMax") && m.contains("not a wall"), "{m}");
        let l = lower_text(&format!(
            r#"{{ "model": "fvDOM", "absorption": {{ "kappa": 1.0 }}, "open": {{ "xMin": {{ "Tinf": 300.0 }} }} }}"#
        ))
        .expect("lowers");
        let m = format!("{}", l.dom_patches(&hm).err().expect("an open entry on a wall is refused"));
        assert!(m.contains("xMin") && m.contains("wall"), "{m}");
        let l = lower_text(&format!(
            r#"{{ "model": "fvDOM", "absorption": {{ "kappa": 1.0 }}, "walls": {{ "nope": {{ "emissivity": 1.0 }} }} }}"#
        ))
        .expect("lowers");
        let m = format!("{}", l.p1_walls(&hm).err().expect("a name that is no patch is refused"));
        assert!(m.contains("nope"), "{m}");
        let l = lower_text(&format!(
            r#"{{ "model": "fvDOM", "absorption": {{ "kappa": 1.0 }}, "walls": {{ "xMin": {{ "emissivity": 1.0 }} }}, "open": {{ "xMin": {{ "Tinf": 300.0 }} }} }}"#
        ))
        .expect("lowers");
        let m = format!("{}", l.p1_walls(&hm).err().expect("a patch named in both is refused"));
        assert!(m.contains("xMin") && m.contains("open"), "{m}");

        // An inlet named in `open` is black at its Tinf, P1 and fvDOM.
        let l = lower_text(&format!(
            r#"{{ "model": "P1", "absorption": {{ "kappa": 1.0 }}, "open": {{ "xMax": {{ "Tinf": 300.0 }} }} }}"#
        ))
        .expect("lowers");
        let w = l.p1_walls(&hm).expect("resolves");
        let open = w.iter().find(|p| p.patch == "xMax").expect("xMax");
        assert_eq!(open.emissivity, 1.0);
        assert_eq!(open.temperature, WallTemperature::Fixed(300.0));

        // A symmetry patch carries nothing (P1) and a Symmetry condition
        // (fvDOM); a stated name on it is refused.
        let mut types = ["wall"; 6];
        types[4] = "symmetry";
        let hm = cube_of(CUBE_WALLS, types);
        let l = lower_text(&fvdom()).expect("lowers");
        let p = l.dom_patches(&hm).expect("resolves");
        assert_eq!(p.len(), 6, "the symmetry patch carries a Symmetry condition");
        assert!(matches!(p.iter().find(|x| x.patch == "zMin"), Some(d) if d.condition == DomCondition::Symmetry));
        let l = lower_text(&format!(
            r#"{{ "model": "fvDOM", "absorption": {{ "kappa": 1.0 }}, "walls": {{ "zMin": {{ "emissivity": 1.0 }} }} }}"#
        ))
        .expect("lowers");
        let m = format!("{}", l.dom_patches(&hm).err().expect("a walls entry on a symmetry patch is refused"));
        assert!(m.contains("zMin") && m.contains("not a wall"), "{m}");
    }

    /// A scratch case directory holding `constant/`.
    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("case_rad_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("constant")).expect("the constant directory is created");
        d
    }

    /// Write `constant/radiationProperties` with `body` as its entries.
    fn write_dict(dir: &Path, body: &str) {
        std::fs::write(
            dir.join(FILE),
            format!(
                "FoamFile {{ version 2.0; format ascii; class dictionary; \
                 object radiationProperties; }}\n{body}\n"
            ),
        )
        .expect("the dictionary is written");
    }

    /// Write a radiation field file (`rel` is `0/G` or `0/IDefault`) whose
    /// `boundaryField` holds `entries`.
    fn write_field(dir: &Path, rel: &str, object: &str, entries: &str) {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).expect("the 0 directory is created");
        std::fs::write(
            p,
            format!(
                "FoamFile {{ version 2.0; format ascii; class volScalarField; \
                 object {object}; }}\ndimensions [1 0 -3 0 0 0 0];\n\
                 internalField uniform 0;\nboundaryField\n{{\n{entries}}}\n"
            ),
        )
        .expect("the field file is written");
    }

    fn refuses_where(dir: &Path, want: &[&str]) {
        let e = read_radiation_dir(dir).err().expect("expected a refusal");
        let m = format!("{e}");
        for w in want {
            assert!(m.contains(w), "{w:?} not in {m}");
        }
    }

    #[test]
    fn dictionary_route_lowers_and_refuses() {
        let dir = scratch("dict");
        // No file: no radiation.
        assert!(read_radiation_dir(&dir).expect("absent").is_none());
        write_dict(&dir, "radiationModel none;");
        assert!(read_radiation_dir(&dir).expect("none").is_none());
        write_dict(&dir, "absorptionEmissionModel constant;");
        refuses_where(&dir, &["radiationModel", "no default"]);
        write_dict(&dir, "radiationModel viewFactor;");
        refuses_where(&dir, &["viewFactor", "CHR-24"]);
        write_dict(&dir, "radiationModel P1; absorptionCoefficient 1; solverFreq 0;");
        refuses_where(&dir, &["solverFreq"]);
        write_dict(&dir, "radiationModel P1; absorptionCoefficient 1; solverFreq 2.5;");
        refuses_where(&dir, &["solverFreq", "whole number"]);
        write_dict(&dir, "radiationModel P1; absorptionCoefficient 1; quadrature S4;");
        refuses_where(&dir, &["quadrature", "fvDOM only"]);
        write_dict(&dir, "radiationModel fvDOM; absorptionCoefficient 1; quadrature S4; scatterTol -1;");
        refuses_where(&dir, &["scatterTol"]);
        write_dict(&dir, "radiationModel fvDOM; absorptionCoefficient 1; quadrature S4; maxScatterIterations 0;");
        refuses_where(&dir, &["maxScatterIterations"]);
        write_dict(&dir, "radiationModel fvDOM; absorptionCoefficient 1; quadrature S5;");
        refuses_where(&dir, &["quadrature", "S5"]);
        write_dict(
            &dir,
            "radiationModel P1; absorptionEmissionModel wsgg; wsggFile absent.txt; wsggPressure 1;",
        );
        refuses_where(&dir, &["wsggFile", "absent.txt"]);
        // The happy dictionary: model, absorption, schedule, quadrature.
        write_dict(
            &dir,
            "radiationModel fvDOM; absorptionCoefficient 1; scatterModel isotropic; scatterCoefficient 0.1; \
             quadrature S4; solverFreq 3; scatterTol 1e-6; maxScatterIterations 7;",
        );
        let l = read_radiation_dir(&dir).expect("lowers").expect("some");
        assert_eq!(l.quadrature.as_deref(), Some("S4"));
        assert_eq!(l.every, 3);
        assert_eq!(l.scatter_tol, Some(1e-6));
        assert_eq!(l.max_sweeps, Some(7));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn field_file_lowers_and_refuses() {
        let dir = scratch("field");
        let hm = cube(CUBE_WALLS);
        write_dict(&dir, "radiationModel P1; absorptionCoefficient 1;");
        // Happy path: every wall named, two stated, the rest default.
        write_field(
            &dir,
            "0/G",
            "G",
            "    xMin { type MarshakRadiation; emissivity 0.8; }\n    \
             xMax { type MarshakRadiationFixedTemperature; Trad 400; }\n    \
             yMin { type MarshakRadiation; }\n    yMax { type MarshakRadiation; }\n    \
             zMin { type MarshakRadiation; }\n    zMax { type MarshakRadiation; }\n",
        );
        let l = read_radiation_dir(&dir).expect("lowers").expect("some");
        let ff = l.field_file.as_ref().expect("the field file was read");
        assert_eq!(ff.len(), 6);
        let w = l.p1_walls(&hm).expect("the walls resolve");
        assert_eq!(w.len(), 6);
        let xmin = w.iter().find(|p| p.patch == "xMin").expect("xMin");
        assert_eq!(xmin.emissivity, 0.8);
        assert_eq!(xmin.temperature, WallTemperature::FromField);
        let xmax = w.iter().find(|p| p.patch == "xMax").expect("xMax");
        assert_eq!(xmax.temperature, WallTemperature::Fixed(400.0));

        let refuses_file = |entries: &str, want: &[&str]| {
            write_field(&dir, "0/G", "G", entries);
            let e = match read_radiation_dir(&dir) {
                Err(e) => e,
                Ok(Some(l)) => l.p1_walls(&hm).err().expect("expected a refusal"),
                Ok(None) => panic!("no radiation"),
            };
            let m = format!("{e}");
            for w in want {
                assert!(m.contains(w), "{w:?} not in {m}");
            }
        };
        let wall = |t: &str| format!("    xMin {{ type {t}; }}\n");
        refuses_file(&wall("magicWalls"), &["xMin", "magicWalls"]);
        refuses_file(&wall("greyDiffusiveRadiation"), &["xMin", "P1", "greyDiffusiveRadiation"]);
        refuses_file(&wall("wideBandDiffusiveRadiation"), &["wideBandDiffusiveRadiation"]);
        refuses_file(
            "    xMin { type MarshakRadiation; Trad 400; }\n",
            &["xMin/Trad", "MarshakRadiation"],
        );
        refuses_file(
            "    xMin { type MarshakRadiationFixedTemperature; }\n",
            &["xMin", "Trad"],
        );
        refuses_file("    xMin { type MarshakRadiation; emissivity 1.5; }\n", &["emissivity"]);
        refuses_file("    nope { type MarshakRadiation; }\n", &["nope", "field file"]);
        refuses_file("    \".*\" { type MarshakRadiation; }\n", &["quoted pattern"]);
        // Under fvDOM the P1 names are the mismatch, in 0/IDefault.
        write_dict(&dir, "radiationModel fvDOM; absorptionCoefficient 1; quadrature S4;");
        write_field(&dir, "0/IDefault", "IDefault", "    xMin { type MarshakRadiation; }\n");
        refuses_where(&dir, &["MarshakRadiation", "fvDOM"]);
        write_field(
            &dir,
            "0/IDefault",
            "IDefault",
            "    xMin { type greyDiffusiveRadiation; emissivity 0.9; }\n    \
             xMax { type greyDiffusiveRadiation; }\n    yMin { type symmetry; }\n    \
             yMax { type greyDiffusiveRadiation; }\n    \
             zMin { type greyDiffusiveRadiation; }\n    zMax { type greyDiffusiveRadiation; }\n",
        );
        let l = read_radiation_dir(&dir).expect("lowers").expect("some");
        let p = l.dom_patches(&hm).expect("the patches resolve");
        assert!(matches!(
            p.iter().find(|d| d.patch == "yMin"),
            Some(d) if d.condition == DomCondition::Symmetry
        ));
        let xmax = p.iter().find(|d| d.patch == "xMax").expect("xMax");
        assert!(matches!(&xmax.condition, DomCondition::GreyDiffuse { temperature: WallTemperature::FromField, .. }));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
