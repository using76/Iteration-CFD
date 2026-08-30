// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The data-centre room case - SPEC-LIT §52, §53, §54 and §55, the case
//! format side.
//!
//! Provenance: ORIGINAL. This is meteor-cfd's own case format, in the JSONC
//! style `crate::io::case_json` established (`docs/05-io-redesign.md`); the
//! physics it names is SPEC-LIT §52's fan curve, §53's porous jump, §54's
//! humidity and §55's metrics, and the numbers a user types are the ones the
//! literature cited in those sections defines. Nothing was transcribed from
//! another code's case format. No GPL-licensed source was consulted.
//!
//! # The rule that shapes it, the same one §47's format uses
//!
//! **Every patch of the room must be named exactly once**, by a `patches`
//! rule, a `fans` entry or a `tiles` entry. Not defaulted, not inferred. An
//! unnamed patch is an error listing which were named and which were not,
//! because "an adiabatic wall unless you say otherwise" is precisely how a
//! case comes to say something the solver ignores (§13.4.1).
//!
//! # What a rack is here, and what it is not
//!
//! A rack is a **heat-release cell zone with a stated flow**, plus a box in
//! which the rack-inlet temperature is sampled. It is not a flow-through
//! device with its own inlet and outlet patches: a block mesh has no patch
//! inside the room, and pretending otherwise would put a boundary condition
//! where there is no boundary. §55.2's `dT_equipment` is therefore the
//! DERIVED one, `Q_IT/(mdot cp)`, and the report says so - which is the
//! distinction that section exists to keep visible.

use std::collections::BTreeMap;
use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::blockgen::{self, BlockSpec, GradedAxis};
use crate::dcmetrics::{AshraeClass, FaceSpan, RciSamples};
use crate::error::{Error, Result};
use crate::fan::{
    CurveKind, FanCurve, FanDirection, FanPatch, PorousJump, PorousJumpCoeffs,
};
use crate::io::case::{LinearSolverKind, Preconditioner, SolverControls};
use crate::io::case_json::{JsonBounds, JsonGrading, JsonGradingAxis};
use crate::mesh::HostMesh;
use crate::psychro;
use crate::{Label, Scalar, Vec3};

#[cfg(test)]
mod tests;

// ==========================================================================
//  1. The document
// ==========================================================================

/// One data-centre room.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DcCase {
    #[serde(rename = "$schema", default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<String>,
    pub name: String,
    pub room: DcRoom,
    #[serde(default)]
    pub air: DcAir,
    /// CRAC/CRAH blowers and exhausts - SPEC-LIT §52.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fans: Vec<DcFan>,
    /// Perforated floor tiles, grilles and filters - SPEC-LIT §53.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tiles: Vec<DcTile>,
    /// Server racks as heat-release zones - see the module doc.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub racks: Vec<DcRack>,
    /// One rule per remaining patch. Every patch must appear here or in a
    /// `fans`/`tiles` entry.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub patches: Vec<DcPatchRule>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub humidity: Option<DcHumidity>,
    pub metrics: DcMetricsSpec,
    pub run: DcRun,
    #[serde(default)]
    pub numerics: DcNumerics,
}

/// The room: an axis-aligned block, its six patches named by the case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DcRoom {
    pub bounds: JsonBounds,
    pub cells: [u32; 3],
    /// The six face names, `-x +x -y +y -z +z`.
    pub boundaries: DcBoundaries,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grading: Option<JsonGrading>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct DcBoundaries {
    pub x_min: String,
    pub x_max: String,
    pub y_min: String,
    pub y_max: String,
    pub z_min: String,
    pub z_max: String,
}

impl DcBoundaries {
    fn names(&self) -> [&str; 6] {
        [
            &self.x_min,
            &self.x_max,
            &self.y_min,
            &self.y_max,
            &self.z_min,
            &self.z_max,
        ]
    }
}

/// The air's properties. Every one of them reaches a kernel; none is
/// decoration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct DcAir {
    /// Kinematic viscosity, m^2/s.
    pub nu: f64,
    /// Reference density, kg/m^3. Divides the manufacturer's curve into this
    /// solver's kinematic pressure (S52.2), so it is not optional.
    pub rho: f64,
    /// Specific heat, J/kg/K - (S55.4)'s `c_p`.
    pub cp: f64,
    /// Molecular and turbulent Prandtl numbers.
    pub pr: f64,
    pub prt: f64,
    /// The buoyancy reference temperature, K.
    pub t_ref: f64,
    /// Gravity, m/s^2.
    pub gravity: [f64; 3],
}

impl Default for DcAir {
    fn default() -> Self {
        Self {
            nu: 1.5e-5,
            rho: 1.2,
            cp: 1005.0,
            pr: 0.71,
            prt: 0.85,
            t_ref: 295.15,
            gravity: [0.0, 0.0, -9.81],
        }
    }
}

/// One fan patch - SPEC-LIT §52.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct DcFan {
    pub patch: String,
    /// `"inflow"` (a supply blower) or `"outflow"` (an exhaust) - (S52.3)'s
    /// `sigma`. There is no default: which way a machine blows is not
    /// something to guess.
    pub direction: String,
    pub curve: DcCurve,
    /// Ambient **static** pressure on the far side, Pa gauge. Converted to
    /// the kinematic pressure this solver carries by dividing by `air.rho`,
    /// and the conversion is printed.
    #[serde(default)]
    pub ambient_pressure: f64,
    /// (S52.14)'s under-relaxation of the operating point.
    #[serde(default = "half")]
    pub relaxation: f64,
    /// The supply air temperature at an `inflow` fan, K. Required there,
    /// refused on an `outflow` one - a temperature on an exhaust is a
    /// setting the case can say and the solver must ignore.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supply_temperature: Option<f64>,
    /// Supply relative humidity, `0..1`. Requires a `humidity` block; naming
    /// it without one is refused.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supply_relative_humidity: Option<f64>,
}

fn half() -> f64 {
    0.5
}

/// The pressure-flow characteristic - SPEC-LIT §52.1 and (S52.13).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct DcCurve {
    /// `"constantPressure"`, `"quadratic"` or `"table"`.
    #[serde(rename = "type")]
    pub kind: String,
    /// `dp_max` (quadratic) or the constant rise, Pa.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dp_max: Option<f64>,
    /// Free delivery, m^3/s.
    #[serde(rename = "QMax", default, skip_serializing_if = "Option::is_none")]
    pub q_max: Option<f64>,
    /// `[[Q, dp], ...]` manufacturer points.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub points: Vec<[f64; 2]>,
    /// AMCA 210's stated measurement conditions, and the ones being run.
    #[serde(default = "one_two")]
    pub rho_curve: f64,
    #[serde(default = "one")]
    pub speed_curve: f64,
    #[serde(default = "one")]
    pub speed: f64,
    #[serde(default = "one")]
    pub efficiency: f64,
}

fn one() -> f64 {
    1.0
}
fn one_two() -> f64 {
    1.2
}

/// One perforated tile / grille / filter - SPEC-LIT §53.
///
/// Exactly one of `K`, `openAreaRatio` or the `(alpha, C2, thickness)` triple
/// must be given. Naming two is refused: the solver would have to pick, and
/// the one it dropped is a setting the case said and the solver ignored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct DcTile {
    pub patch: String,
    /// The loss coefficient on the approach velocity - (S53.3).
    #[serde(rename = "K", default, skip_serializing_if = "Option::is_none")]
    pub k: Option<f64>,
    /// The tile's open-area fraction, `(0, 1]` - (S53.6) converts it, and
    /// the derived `K` is printed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_area_ratio: Option<f64>,
    /// Permeability, m^2 - (S53.1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f64>,
    /// Inertial resistance factor, 1/m.
    #[serde(rename = "C2", default, skip_serializing_if = "Option::is_none")]
    pub c2: Option<f64>,
    /// Sheet thickness, m - a PARAMETER, not a mesh dimension.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thickness: Option<f64>,
    /// Ask for the jump to be an internal **baffle** - two coincident
    /// boundary faces the solver would have to create. SPEC-LIT §53.5:
    /// refused by name, listing the two routes that exist. The entry is here
    /// so the refusal can FIRE: a request nobody can express is not a
    /// request that was refused, it is one that was never heard.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baffle: Option<bool>,
    /// The plenum's static pressure behind the tile, Pa gauge.
    pub plenum_pressure: f64,
    /// The plenum air temperature, K.
    pub plenum_temperature: f64,
    /// The plenum's relative humidity, `0..1`. Requires a `humidity` block;
    /// naming it without one is refused, exactly as it is on a fan - a tile
    /// fed from a plenum carries the plenum's humidity, and (S54.2)/(S54.4)
    /// convert it at setup and PRINT what they converted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plenum_relative_humidity: Option<f64>,
}

/// One rack - a heat-release zone with a stated flow, plus the box its
/// inlet temperature is sampled in. See the module doc for what this is not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct DcRack {
    pub name: String,
    /// The cells the heat is released in.
    pub zone: JsonBounds,
    /// IT power, W.
    pub power: f64,
    /// Volumetric flow through the rack, m^3/s - (S55.2)'s `mdot_IT/rho`.
    pub flow: f64,
    /// Where the rack-inlet temperature is sampled for RCI (S55.1).
    pub inlet_samples: JsonBounds,
}

/// SPEC-LIT §54: humidity transport and psychrometrics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct DcHumidity {
    /// Molecular diffusivity of water vapour in air, m^2/s.
    #[serde(default = "d_v")]
    pub d: f64,
    /// Turbulent Schmidt number.
    #[serde(default = "sct")]
    pub sc_t: f64,
    /// Total barometric pressure, Pa - (S54.2b)'s `p_atm`, which is NOT this
    /// solver's kinematic gauge pressure.
    #[serde(default = "p_atm")]
    pub barometric_pressure: f64,
    /// Whether (S54.7)'s virtual temperature feeds the buoyancy. On by
    /// default, because moist air really is lighter and the correction is
    /// exact; off is a legitimate comparison and is a §13.4.1 setting.
    #[serde(default = "yes")]
    pub virtual_temperature: bool,
}

fn d_v() -> f64 {
    2.5e-5
}
fn sct() -> f64 {
    0.7
}
fn p_atm() -> f64 {
    101325.0
}
fn yes() -> bool {
    true
}

impl Default for DcHumidity {
    fn default() -> Self {
        Self {
            d: d_v(),
            sc_t: sct(),
            barometric_pressure: p_atm(),
            virtual_temperature: true,
        }
    }
}

/// SPEC-LIT §55: what the report measures, and over what.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct DcMetricsSpec {
    /// `"A1"`..`"A4"` - ASHRAE TC 9.9's allowable envelope (S55.1).
    #[serde(default = "a1")]
    pub ashrae_class: String,
    /// `"thirds"` (Herrlin's convention, mesh-independent) or `"faces"`.
    #[serde(default = "thirds")]
    pub rci_samples: String,
    /// The patch the supply temperature is measured on.
    pub supply_patch: String,
    /// The patch the return temperature is measured on.
    pub return_patch: String,
    /// A supply-temperature sweep for §55.4's free-cooling ceiling: the
    /// highest supply temperature at which `RCI_HI` stays at 100 %.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supply_temperature_sweep: Option<[f64; 3]>,
}

fn a1() -> String {
    "A1".to_string()
}
fn thirds() -> String {
    "thirds".to_string()
}

/// One patch rule, for every patch not owned by a fan or a tile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct DcPatchRule {
    pub patch: String,
    /// `"wall"`, `"adiabaticWall"`, `"empty"`, `"symmetry"` or
    /// `"fixedPressure"`.
    pub kind: String,
    /// Wall temperature, K - only on `"wall"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Static pressure, Pa gauge - only on `"fixedPressure"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pressure: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct DcRun {
    /// Outer SIMPLE iterations.
    pub iterations: u32,
    /// How often the metric report is printed. `0` prints only at the end.
    #[serde(default)]
    pub report_every: u32,
    /// Initial room temperature, K.
    pub initial_temperature: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct DcNumerics {
    /// Which pressure backend to use. `"pbicgstab"` (the default) and
    /// `"pcg"` are supported; `"fft"`, `"capacitance"` and `"woodbury"` are
    /// SPEC-LIT §52.9's refusal - the direct Poisson path a fan patch
    /// disables, and the rank-1 correction that would put it back and is
    /// **not implemented**. The entry exists so that refusal can fire.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pressure_solver: Option<String>,
    pub u_relax: f64,
    pub p_relax: f64,
    pub t_relax: f64,
    pub tolerance: f64,
    pub max_iterations: u32,
}

impl Default for DcNumerics {
    fn default() -> Self {
        Self {
            pressure_solver: None,
            u_relax: 0.7,
            p_relax: 0.3,
            t_relax: 0.7,
            tolerance: 1e-8,
            max_iterations: 500,
        }
    }
}

// ==========================================================================
//  2. What the reader produces
// ==========================================================================

/// One rack, resolved against the mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct LoweredRack {
    pub name: String,
    /// The cells the heat is released in, sorted.
    pub cells: Vec<Label>,
    /// `q'''` = power / zone volume, W/m^3.
    pub q_vol: Scalar,
    pub power: Scalar,
    pub flow: Scalar,
    /// RCI sample cells, sorted - `thirds` or `faces` per §55.1.
    pub samples: Vec<Label>,
}

/// Everything the driver needs, resolved against the mesh.
#[derive(Debug)]
pub struct LoweredDcCase {
    pub name: String,
    pub mesh: HostMesh,
    pub air: DcAir,
    /// Kinematic ambient pressures, m^2/s^2 - already divided by `air.rho`.
    pub fans: Vec<FanPatch>,
    pub jumps: Vec<PorousJump>,
    pub racks: Vec<LoweredRack>,
    pub humidity: Option<DcHumidity>,
    pub class: AshraeClass,
    pub samples: RciSamples,
    pub supply_span: FaceSpan,
    pub return_span: FaceSpan,
    /// Per-patch pressure BC kind and value, and per-patch temperature.
    pub patch_pressure: BTreeMap<String, Option<Scalar>>,
    pub patch_temperature: BTreeMap<String, Option<Scalar>>,
    /// The `Y_v` an `inflow` fan or a tile injects, from (S54.2)/(S54.4).
    pub inflow_humidity: BTreeMap<String, Scalar>,
    pub inflow_temperature: BTreeMap<String, Scalar>,
    pub run: DcRun,
    pub numerics: DcNumerics,
    /// The lines the reader wants printed: every conversion it performed.
    pub notes: Vec<String>,
    pub solver: SolverControls,
}

// ==========================================================================
//  3. Reading
// ==========================================================================

impl DcCase {
    /// Read a JSONC document, with comments and trailing commas - the same
    /// reader every other case format in this crate goes through, so the
    /// syntax a user has learned once is the syntax everywhere.
    pub fn read(path: &Path) -> Result<Self> {
        crate::io::case_json::parse_jsonc_file(path)
    }

    pub fn parse(text: &str, what: &str) -> Result<Self> {
        crate::io::case_json::parse_jsonc_str(text, what)
    }

    /// Build the mesh and resolve every entry against it - SPEC-LIT §13.4's
    /// contract, applied to every setting this format carries.
    #[allow(clippy::too_many_lines)]
    pub fn lower(&self) -> Result<LoweredDcCase> {
        let mut notes = Vec::new();
        self.air.validate()?;

        // ---- the mesh -----------------------------------------------------
        let b = &self.room.bounds;
        let g = self.room.grading.clone().unwrap_or_default();
        let axis = |i: usize| -> GradedAxis {
            // `JsonGradingAxis` has no `Default` - the whole point of
            // `case_json`'s design there is that a case grading an axis must
            // say which way. An axis the case did not mention is UNGRADED,
            // which is `expansion = 1`, and that is a statement about this
            // format rather than a default that silently picks a setup.
            let ga = match i {
                0 => g.x.clone(),
                1 => g.y.clone(),
                _ => g.z.clone(),
            }
            .unwrap_or(JsonGradingAxis { expansion: 1.0, two_sided: false });
            GradedAxis {
                lo: b.min[i],
                hi: b.max[i],
                n: self.room.cells[i] as usize,
                expansion: ga.expansion,
                two_sided: ga.two_sided,
            }
        };
        let names = self.room.boundaries.names();
        let mut spec = BlockSpec {
            x: axis(0),
            y: axis(1),
            z: axis(2),
            patch_name: [
                names[0].to_string(),
                names[1].to_string(),
                names[2].to_string(),
                names[3].to_string(),
                names[4].to_string(),
                names[5].to_string(),
            ],
            patch_type: ["patch".to_string(), "patch".to_string(), "patch".to_string(),
                         "patch".to_string(), "patch".to_string(), "patch".to_string()],
            ..BlockSpec::default()
        };
        // A one-cell axis is a 2-D case and its two slots are `empty`; that is
        // the mesh's own constraint (S34.1), not a boundary condition, so it
        // is applied here rather than asked of the case.
        for (i, n) in self.room.cells.iter().enumerate() {
            if *n == 1 {
                spec.patch_type[2 * i] = "empty".to_string();
                spec.patch_type[2 * i + 1] = "empty".to_string();
                notes.push(format!(
                    "axis {} is one cell across, so patches \"{}\" and \"{}\" are the \
                     2-D `empty` constraint of SPEC-LIT S34.1",
                    ["x", "y", "z"][i],
                    names[2 * i],
                    names[2 * i + 1]
                ));
            }
        }
        let mesh = blockgen::build_mesh(&spec)?;

        // ---- every patch named exactly once -------------------------------
        let mut owner: BTreeMap<String, String> = BTreeMap::new();
        let mut claim = |patch: &str, by: &str| -> Result<()> {
            if !names.contains(&patch) {
                return Err(Error::Config(format!(
                    "{by} names patch \"{patch}\", which is not one of the room's six: \
                     {}",
                    names.join(", ")
                )));
            }
            if let Some(prev) = owner.get(patch) {
                return Err(Error::Config(format!(
                    "patch \"{patch}\" is claimed by both {prev} and {by}. Each patch \
                     carries ONE condition; whichever ran last would win, silently"
                )));
            }
            owner.insert(patch.to_string(), by.to_string());
            Ok(())
        };
        for f in &self.fans {
            claim(&f.patch, &format!("fan \"{}\"", f.patch))?;
        }
        for t in &self.tiles {
            claim(&t.patch, &format!("tile \"{}\"", t.patch))?;
        }
        for r in &self.patches {
            claim(&r.patch, &format!("patch rule \"{}\"", r.patch))?;
        }
        let unnamed: Vec<&str> =
            names.iter().copied().filter(|n| !owner.contains_key(*n)).collect();
        if !unnamed.is_empty() {
            return Err(Error::Config(format!(
                "{} of the room's patches are not named by any fan, tile or patch \
                 rule: {}. Named: {}. SPEC-LIT S55.6: \"an adiabatic wall unless you \
                 say otherwise\" is exactly how a case comes to say something the \
                 solver ignores, so every patch must be named EXACTLY ONCE",
                unnamed.len(),
                unnamed.join(", "),
                owner.keys().cloned().collect::<Vec<_>>().join(", ")
            )));
        }

        // ---- the fans -----------------------------------------------------
        let rho = self.air.rho as Scalar;
        let mut fans = Vec::new();
        let mut inflow_humidity: BTreeMap<String, Scalar> = BTreeMap::new();
        let mut inflow_temperature: BTreeMap<String, Scalar> = BTreeMap::new();
        for f in &self.fans {
            let dir = FanDirection::from_name(&f.direction, &f.patch)?;
            let curve = f.curve.lower(&f.patch, rho)?;
            curve.validate(&f.patch)?;

            match (dir, f.supply_temperature) {
                (FanDirection::Inflow, None) => {
                    return Err(Error::Config(format!(
                        "fan \"{}\" is an INFLOW (a supply blower) and carries no \
                         supplyTemperature. The air it pushes in has to be at some \
                         temperature, and defaulting it to the room's would make the \
                         CRAC do no cooling while still reporting a flow rate",
                        f.patch
                    )))
                }
                (FanDirection::Outflow, Some(t)) => {
                    return Err(Error::Config(format!(
                        "fan \"{}\" is an OUTFLOW (an exhaust) and carries \
                         supplyTemperature {t}. An exhaust takes whatever temperature \
                         the room gives it; the entry would be read and ignored, which \
                         is the SPEC-LIT S13.4.1 defect. Remove it, or set \
                         direction to \"inflow\"",
                        f.patch
                    )))
                }
                (FanDirection::Inflow, Some(t)) => {
                    if !(t > 0.0) {
                        return Err(Error::Config(format!(
                            "fan \"{}\": supplyTemperature {t} K is not an absolute \
                             temperature",
                            f.patch
                        )));
                    }
                    inflow_temperature.insert(f.patch.clone(), t as Scalar);
                }
                (FanDirection::Outflow, None) => {}
            }

            if let Some(rh) = f.supply_relative_humidity {
                let Some(h) = self.humidity else {
                    return Err(Error::Config(format!(
                        "fan \"{}\" carries supplyRelativeHumidity {rh} but the case \
                         has no `humidity` block, so nothing transports water vapour \
                         and the entry would be read and ignored (SPEC-LIT S13.4.1)",
                        f.patch
                    )));
                };
                if !(0.0..=1.0).contains(&rh) {
                    return Err(Error::Config(format!(
                        "fan \"{}\": supplyRelativeHumidity {rh} is outside [0, 1] - \
                         it is a FRACTION, not a percentage",
                        f.patch
                    )));
                }
                let t = f.supply_temperature.unwrap_or(self.run.initial_temperature);
                let yv = psychro::yv_from_t_rh_p(
                    t as Scalar,
                    rh as Scalar,
                    h.barometric_pressure as Scalar,
                );
                notes.push(format!(
                    "fan \"{}\": rh {rh} at {t} K and {} Pa is Y_v = {yv:.7} kg/kg \
                     (W = {:.7} kg/kg dry air) - SPEC-LIT (S54.2)/(S54.4)",
                    f.patch,
                    h.barometric_pressure,
                    psychro::w_from_yv(yv)
                ));
                inflow_humidity.insert(f.patch.clone(), yv);
            }

            if !(f.relaxation > 0.0 && f.relaxation <= 1.0) {
                return Err(Error::Config(format!(
                    "fan \"{}\": relaxation {} is outside (0, 1] - it is SPEC-LIT \
                     (S52.14)'s under-relaxation of the OPERATING POINT",
                    f.patch, f.relaxation
                )));
            }
            let ambient = f.ambient_pressure as Scalar / rho;
            notes.push(format!(
                "fan \"{}\": ambientPressure {} Pa is {ambient:.6} m^2/s^2 kinematic \
                 (divided by rho = {rho})",
                f.patch, f.ambient_pressure
            ));
            fans.push(FanPatch {
                patch: f.patch.clone(),
                curve,
                direction: dir,
                ambient,
                relaxation: f.relaxation as Scalar,
            });
        }

        // ---- the tiles ----------------------------------------------------
        let mut jumps = Vec::new();
        for t in &self.tiles {
            if t.baffle == Some(true) {
                // SPEC-LIT §53.5. Never succeeds; under `-permissive` it
                // warns and substitutes the internal-face form, which is
                // printed.
                crate::fan::refuse_baffle_insertion(&format!("tiles/{}/baffle", t.patch))?;
            }
            let (coeffs, how) = t.lower(self.air.nu as Scalar)?;
            notes.push(format!("tile \"{}\": {how}", t.patch));
            if !(t.plenum_temperature > 0.0) {
                return Err(Error::Config(format!(
                    "tile \"{}\": plenumTemperature {} K is not an absolute \
                     temperature",
                    t.patch, t.plenum_temperature
                )));
            }
            inflow_temperature.insert(t.patch.clone(), t.plenum_temperature as Scalar);
            if let Some(rh) = t.plenum_relative_humidity {
                let Some(h) = self.humidity else {
                    return Err(Error::Config(format!(
                        "tile \"{}\" carries plenumRelativeHumidity {rh} but the case \
                         has no `humidity` block, so nothing transports water vapour \
                         and the entry would be read and ignored (SPEC-LIT S13.4.1)",
                        t.patch
                    )));
                };
                if !(0.0..=1.0).contains(&rh) {
                    return Err(Error::Config(format!(
                        "tile \"{}\": plenumRelativeHumidity {rh} is outside [0, 1] - \
                         it is a FRACTION, not a percentage",
                        t.patch
                    )));
                }
                let yv = psychro::yv_from_t_rh_p(
                    t.plenum_temperature as Scalar,
                    rh as Scalar,
                    h.barometric_pressure as Scalar,
                );
                notes.push(format!(
                    "tile \"{}\": rh {rh} at {} K and {} Pa is Y_v = {yv:.7} kg/kg \
                     (W = {:.7} kg/kg dry air) - SPEC-LIT (S54.2)/(S54.4)",
                    t.patch,
                    t.plenum_temperature,
                    h.barometric_pressure,
                    psychro::w_from_yv(yv)
                ));
                inflow_humidity.insert(t.patch.clone(), yv);
            }
            jumps.push(PorousJump::Boundary {
                patch: t.patch.clone(),
                coeffs,
                plenum: t.plenum_pressure as Scalar / rho,
            });
        }

        // ---- the racks ----------------------------------------------------
        let class = AshraeClass::from_name(&self.metrics.ashrae_class)?;
        let samples = RciSamples::from_name(&self.metrics.rci_samples)?;
        let mut racks = Vec::new();
        for r in &self.racks {
            racks.push(r.lower(&mesh, samples)?);
        }
        if racks.is_empty() && !self.metrics.rci_samples.is_empty() {
            notes.push(
                "no racks: RCI has no sample points and is reported as 100 % over \
                 n = 0, which is what an empty room deserves rather than a division \
                 by zero"
                    .to_string(),
            );
        }

        // ---- the metric patches -------------------------------------------
        let supply_span = FaceSpan::of_patch(&mesh, &self.metrics.supply_patch, "supply")?;
        let return_span = FaceSpan::of_patch(&mesh, &self.metrics.return_patch, "return")?;
        if self.metrics.supply_patch == self.metrics.return_patch {
            return Err(Error::Config(format!(
                "metrics: supplyPatch and returnPatch are both \"{}\". (S55.2)'s RTI \
                 is (T_return - T_supply)/dT_equipment, which would then be exactly \
                 zero whatever the room did",
                self.metrics.supply_patch
            )));
        }

        // ---- the remaining patch rules ------------------------------------
        let mut patch_pressure = BTreeMap::new();
        let mut patch_temperature = BTreeMap::new();
        for r in &self.patches {
            match r.kind.as_str() {
                "wall" => {
                    let Some(t) = r.temperature else {
                        return Err(Error::Config(format!(
                            "patch \"{}\" is a `wall` and carries no temperature. \
                             Available: adiabaticWall (a wall that exchanges no heat), \
                             or give it one",
                            r.patch
                        )));
                    };
                    patch_temperature.insert(r.patch.clone(), Some(t as Scalar));
                    patch_pressure.insert(r.patch.clone(), None);
                }
                "adiabaticWall" | "empty" | "symmetry" => {
                    patch_temperature.insert(r.patch.clone(), None);
                    patch_pressure.insert(r.patch.clone(), None);
                }
                "fixedPressure" => {
                    let Some(p) = r.pressure else {
                        return Err(Error::Config(format!(
                            "patch \"{}\" is a `fixedPressure` and carries no pressure",
                            r.patch
                        )));
                    };
                    patch_pressure.insert(r.patch.clone(), Some(p as Scalar / rho));
                    patch_temperature.insert(r.patch.clone(), None);
                }
                other => {
                    return Err(Error::Config(format!(
                        "patch \"{}\": kind \"{other}\" is not supported by ofgpu; \
                         available: wall (with a temperature), adiabaticWall, empty, \
                         symmetry, fixedPressure. A fan or a tile is declared in the \
                         `fans` or `tiles` block instead, because it carries a curve \
                         or a resistance that a patch rule has nowhere to put",
                        r.patch
                    )))
                }
            }
            if r.temperature.is_some() && r.kind != "wall" {
                return Err(Error::Config(format!(
                    "patch \"{}\" is a `{}` and carries a temperature, which nothing \
                     would read (SPEC-LIT S13.4.1)",
                    r.patch, r.kind
                )));
            }
            if r.pressure.is_some() && r.kind != "fixedPressure" {
                return Err(Error::Config(format!(
                    "patch \"{}\" is a `{}` and carries a pressure, which nothing \
                     would read (SPEC-LIT S13.4.1)",
                    r.patch, r.kind
                )));
            }
        }

        // ---- humidity ------------------------------------------------------
        if let Some(h) = self.humidity {
            h.validate()?;
            if inflow_humidity.is_empty() {
                return Err(Error::Config(
                    "the case has a `humidity` block but no fan carries a \
                     supplyRelativeHumidity and no tile a plenumRelativeHumidity, so \
                     Y_v would be transported from an initial field with nothing \
                     feeding it. SPEC-LIT S54.6: a humid boundary is what makes the \
                     transport mean anything"
                        .to_string(),
                ));
            }
            notes.push(format!(
                "humidity: D_v = {} m^2/s, Sc_t = {}, p_atm = {} Pa, virtual \
                 temperature {} (SPEC-LIT S54)",
                h.d,
                h.sc_t,
                h.barometric_pressure,
                if h.virtual_temperature { "ON" } else { "OFF" }
            ));
            let (i, r, bias) = psychro::enhancement_bias(298.15, h.barometric_pressure as Scalar, 1.0044);
            notes.push(format!(
                "the psychrometrics here are the IDEAL-gas relations. W_s(25 C) comes \
                 out {i:.7} where the ASHRAE table (which carries the real-gas \
                 enhancement factor) gives {r:.7} - a {:.2} % low bias, documented in \
                 SPEC-LIT S54.3, not corrected",
                100.0 * bias
            ));
        }

        if self.run.iterations == 0 {
            return Err(Error::Config(
                "run.iterations is zero - the case would build a mesh and report the \
                 initial field as an answer"
                    .to_string(),
            ));
        }
        if !(self.run.initial_temperature > 0.0) {
            return Err(Error::Config(format!(
                "run.initialTemperature {} K is not an absolute temperature",
                self.run.initial_temperature
            )));
        }
        self.numerics.validate()?;

        let kind = self.numerics.solver_kind()?;
        let solver = SolverControls {
            solver: kind,
            precon: if kind == LinearSolverKind::PCG {
                Preconditioner::Dic
            } else {
                Preconditioner::Dilu
            },
            tolerance: self.numerics.tolerance as Scalar,
            rel_tol: 0.0,
            max_iter: self.numerics.max_iterations as _,
            ..SolverControls::default()
        };

        Ok(LoweredDcCase {
            name: self.name.clone(),
            mesh,
            air: self.air,
            fans,
            jumps,
            racks,
            humidity: self.humidity,
            class,
            samples,
            supply_span,
            return_span,
            patch_pressure,
            patch_temperature,
            inflow_humidity,
            inflow_temperature,
            run: self.run.clone(),
            numerics: self.numerics.clone(),
            notes,
            solver,
        })
    }
}

impl DcAir {
    fn validate(&self) -> Result<()> {
        for (n, v) in [
            ("nu", self.nu),
            ("rho", self.rho),
            ("cp", self.cp),
            ("Pr", self.pr),
            ("Prt", self.prt),
            ("TRef", self.t_ref),
        ] {
            if !(v > 0.0) {
                return Err(Error::Config(format!(
                    "air.{n} = {v} must be positive"
                )));
            }
        }
        Ok(())
    }
}

impl DcNumerics {
    /// The linear solver this case asked for, or PBiCGStab.
    fn solver_kind(&self) -> Result<LinearSolverKind> {
        match self.pressure_solver.as_deref() {
            None | Some("pbicgstab") => Ok(LinearSolverKind::PBiCGStab),
            Some("pcg") => Ok(LinearSolverKind::PCG),
            // SPEC-LIT §52.9: the direct Poisson path and the rank-1
            // correction that would keep it alive under a fan patch.
            Some("fft" | "capacitance" | "woodbury") => {
                crate::fan::refuse_capacitance_fft("numerics/pressureSolver")?;
                Ok(LinearSolverKind::PBiCGStab)
            }
            Some(other) => Err(Error::Config(format!(
                "numerics.pressureSolver: \"{other}\" is not supported by ofgpu; \
                 available: pbicgstab, pcg. SPEC-LIT S52.8: a fan patch makes a face \
                 neither uniformly Dirichlet nor uniformly Neumann and a jump makes \
                 the coefficient non-constant, so the cuFFT direct path is not \
                 available on a room that has either - S52.9 names the Woodbury \
                 correction that would put it back and says it is not implemented"
            ))),
        }
    }

    fn validate(&self) -> Result<()> {
        for (n, v) in [
            ("uRelax", self.u_relax),
            ("pRelax", self.p_relax),
            ("tRelax", self.t_relax),
        ] {
            if !(v > 0.0 && v <= 1.0) {
                return Err(Error::Config(format!(
                    "numerics.{n} = {v} is outside (0, 1]"
                )));
            }
        }
        if !(self.tolerance > 0.0) {
            return Err(Error::Config(format!(
                "numerics.tolerance = {} must be positive",
                self.tolerance
            )));
        }
        Ok(())
    }
}

impl DcHumidity {
    fn validate(&self) -> Result<()> {
        if !(self.d >= 0.0) {
            return Err(Error::Config(format!(
                "humidity.D = {} is not a diffusivity",
                self.d
            )));
        }
        if !(self.sc_t > 0.0) {
            return Err(Error::Config(format!(
                "humidity.Sct = {} must be positive (SPEC-LIT S19)",
                self.sc_t
            )));
        }
        if !(self.barometric_pressure > 0.0) {
            return Err(Error::Config(format!(
                "humidity.barometricPressure = {} Pa must be positive - it is the \
                 TOTAL pressure of (S54.2b), not this solver's kinematic gauge \
                 pressure",
                self.barometric_pressure
            )));
        }
        Ok(())
    }
}

impl DcCurve {
    /// `rho` is the air the machine is actually working in - `air.rho`.
    /// It is passed in rather than left for the driver to patch afterwards,
    /// because a `rho_ratio()` that is always 1 on the lowered case is a
    /// setting the document can say and the reader drops, which is the
    /// S13.4.1 defect this format exists to prevent. A pair test on
    /// `rhoCurve` is what found it.
    fn lower(&self, patch: &str, rho: Scalar) -> Result<FanCurve> {
        let kind = match self.kind.as_str() {
            "constantPressure" | "constant" => CurveKind::Constant,
            "quadratic" => CurveKind::Quadratic,
            "table" => CurveKind::Table,
            other => {
                return Err(Error::Config(format!(
                    "fan \"{patch}\": curve type \"{other}\" is not supported by \
                     ofgpu; available: constantPressure (a flat curve, which is \
                     fixedValue on p - SPEC-LIT S52.4), quadratic \
                     (dp = dpMax[1 - (Q/QMax)^2]), table (a monotone Hermite through \
                     manufacturer points)"
                )))
            }
        };

        // Every entry that does not belong to the chosen kind is REFUSED, not
        // ignored: a case that writes `points` under `quadratic` has said
        // something the solver would drop (SPEC-LIT S13.4.1).
        match kind {
            CurveKind::Table => {
                if self.dp_max.is_some() || self.q_max.is_some() {
                    return Err(Error::Config(format!(
                        "fan \"{patch}\": a `table` curve carries dpMax/QMax, which a \
                         table has no use for - the points ARE the curve. Remove them, \
                         or use type \"quadratic\""
                    )));
                }
            }
            _ => {
                if !self.points.is_empty() {
                    return Err(Error::Config(format!(
                        "fan \"{patch}\": a `{}` curve carries {} table points, which \
                         nothing would read. Use type \"table\"",
                        self.kind,
                        self.points.len()
                    )));
                }
            }
        }

        let c = FanCurve {
            kind,
            dp_max: self.dp_max.unwrap_or(0.0) as Scalar,
            q_max: self.q_max.unwrap_or(1.0) as Scalar,
            points: self
                .points
                .iter()
                .map(|p| (p[0] as Scalar, p[1] as Scalar))
                .collect(),
            rho_curve: self.rho_curve as Scalar,
            rho,
            n_curve: self.speed_curve as Scalar,
            n_speed: self.speed as Scalar,
            efficiency: self.efficiency as Scalar,
        };
        if kind != CurveKind::Table && self.dp_max.is_none() {
            return Err(Error::Config(format!(
                "fan \"{patch}\": a `{}` curve needs dpMax",
                self.kind
            )));
        }
        if kind == CurveKind::Quadratic && self.q_max.is_none() {
            return Err(Error::Config(format!(
                "fan \"{patch}\": a `quadratic` curve needs QMax (free delivery)"
            )));
        }
        Ok(c)
    }
}

impl DcTile {
    /// `(coefficients, the sentence describing how they were arrived at)`.
    fn lower(&self, nu: Scalar) -> Result<(PorousJumpCoeffs, String)> {
        let named = [
            self.k.is_some(),
            self.open_area_ratio.is_some(),
            self.alpha.is_some() || self.c2.is_some() || self.thickness.is_some(),
        ];
        let n = named.iter().filter(|b| **b).count();
        if n == 0 {
            return Err(Error::Config(format!(
                "tile \"{}\" has no resistance. Available: K (the loss coefficient on \
                 the approach velocity - what a tile datasheet gives), openAreaRatio \
                 (converted by SPEC-LIT (S53.6), and the derived K is printed), or the \
                 triple (alpha, C2, thickness) of (S53.1)",
                self.patch
            )));
        }
        if n > 1 {
            return Err(Error::Config(format!(
                "tile \"{}\" gives more than one resistance parameterisation. The \
                 solver would have to pick, and the one it dropped would be a setting \
                 the case said and the solver ignored (SPEC-LIT S13.4.1). Give exactly \
                 one of: K, openAreaRatio, or (alpha, C2, thickness)",
                self.patch
            )));
        }

        if let Some(k) = self.k {
            return Ok((
                PorousJumpCoeffs::from_loss_coefficient(k as Scalar)?,
                format!("loss coefficient K = {k} on the approach velocity (S53.3)"),
            ));
        }
        if let Some(s) = self.open_area_ratio {
            let k = PorousJumpCoeffs::loss_coefficient_of_open_area(s as Scalar)?;
            return Ok((
                PorousJumpCoeffs::from_loss_coefficient(k)?,
                format!(
                    "openAreaRatio {s} gives K = {k:.4} through (S53.6). The design \
                     note SPEC-LIT S53 was written from quotes K ~ 30 at sigma = 0.25 \
                     (reproduced) and K ~ 4 at sigma = 0.56 (contradicted - the formula \
                     gives 2.94 there); S53.4 records that"
                ),
            ));
        }

        let (Some(a), Some(c2), Some(t)) = (self.alpha, self.c2, self.thickness) else {
            return Err(Error::Config(format!(
                "tile \"{}\": the Darcy-Forchheimer form needs all three of alpha, C2 \
                 and thickness (SPEC-LIT (S53.1)); a partial triple would silently \
                 default the missing one",
                self.patch
            )));
        };
        Ok((
            PorousJumpCoeffs::from_darcy_forchheimer(a as Scalar, c2 as Scalar, t as Scalar, nu)?,
            format!(
                "Darcy-Forchheimer alpha = {a} m^2, C2 = {c2} 1/m, thickness = {t} m \
                 at nu = {nu} m^2/s (S53.1)"
            ),
        ))
    }
}

impl DcRack {
    fn lower(&self, mesh: &HostMesh, samples: RciSamples) -> Result<LoweredRack> {
        let inside = |c: Vec3, b: &JsonBounds| -> bool {
            c.x >= b.min[0] as Scalar
                && c.x <= b.max[0] as Scalar
                && c.y >= b.min[1] as Scalar
                && c.y <= b.max[1] as Scalar
                && c.z >= b.min[2] as Scalar
                && c.z <= b.max[2] as Scalar
        };
        // Sorted by construction (the scan is in cell order), which is what
        // makes the gather order fixed and the reduction reproducible.
        let cells: Vec<Label> = (0..mesh.n_cells)
            .filter(|i| inside(mesh.c[*i], &self.zone))
            .map(|i| i as Label)
            .collect();
        if cells.is_empty() {
            return Err(Error::Config(format!(
                "rack \"{}\": its zone contains no cell centres. A rack that releases \
                 its heat nowhere would still be counted in the IT total, which is the \
                 SPEC-LIT S13.4.1 defect",
                self.name
            )));
        }
        if !(self.power > 0.0) {
            return Err(Error::Config(format!(
                "rack \"{}\": power {} W must be positive",
                self.name, self.power
            )));
        }
        if !(self.flow > 0.0) {
            return Err(Error::Config(format!(
                "rack \"{}\": flow {} m^3/s must be positive - it is the denominator \
                 of (S55.2)'s dT_equipment = Q_IT/(mdot cp)",
                self.name, self.flow
            )));
        }
        let vol: Scalar = cells.iter().map(|c| mesh.v[*c as usize]).sum();

        let mut in_box: Vec<Label> = (0..mesh.n_cells)
            .filter(|i| inside(mesh.c[*i], &self.inlet_samples))
            .map(|i| i as Label)
            .collect();
        if in_box.is_empty() {
            return Err(Error::Config(format!(
                "rack \"{}\": its inletSamples box contains no cell centres, so RCI \
                 would be measured over nothing",
                self.name
            )));
        }
        // `thirds` is Herrlin's own convention: three points per rack, at 1/6,
        // 1/2 and 5/6 of the sample box's HEIGHT (the gravity axis), so the
        // index does not move when the mesh is refined (S55.1).
        let sample_cells = match samples {
            RciSamples::Faces => in_box.clone(),
            RciSamples::Thirds => {
                in_box.sort_by(|a, b| {
                    mesh.c[*a as usize]
                        .z
                        .partial_cmp(&mesh.c[*b as usize].z)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.cmp(b))
                });
                let n = in_box.len();
                let mut out: Vec<Label> =
                    [1usize, 3, 5].iter().map(|k| in_box[(n * k) / 6]).collect();
                out.sort_unstable();
                out.dedup();
                out
            }
        };

        Ok(LoweredRack {
            name: self.name.clone(),
            cells,
            q_vol: self.power as Scalar / vol,
            power: self.power as Scalar,
            flow: self.flow as Scalar,
            samples: sample_cells,
        })
    }
}
