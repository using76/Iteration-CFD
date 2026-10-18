// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The `parcels` case block - JSONC `parcels` and `constant/cloudProperties`
//! - lowered onto the parcel library's own settings (docs/17 CHR-25,
//! SPEC-LIT §137.2 and §137.6).
//!
//! There is one key per [`ParcelControls`], [`EvaporationControls`],
//! [`LiquidProperties`], [`WallImpactControls`], [`CouplingControls`] and
//! [`Injector`] field, so that every library field is reachable from a case.
//! Both routes fill the same [`JsonParcels`]: the dictionary route turns
//! `constant/cloudProperties` into the JSON object the JSONC route reads and
//! hands it to the same deserialiser, so the two routes cannot drift, and
//! [`JsonParcels::lower`] is the one lowering.
//!
//! Every model name goes through the library's own `from_name`, so a refused
//! name (`film`, `splash`) is refused with the library's §78 reason. The
//! table's capitalised spellings (`SchillerNaumann`, `Weber`) are accepted
//! by lowering their first letter.
//!
//! Written from:
//!   ofgpu SPEC-LIT.md §137.2 - the `parcels` table and its refusals
//!   ofgpu SPEC-LIT.md §137.6 - the as-built rules of this file
//!   ofgpu SPEC-LIT.md §66, §68, §76, §77, §78 - the settings themselves,
//!     through `src/parcels.rs` and `src/parcels/*.rs`
//!
//! Provenance: ORIGINAL. No external source. No GPL-licensed source was
//! consulted.

use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::io::dict::{last_number, FoamDict};
use crate::parcels::couple::{CouplingControls, CouplingMode, MassCoupling};
use crate::parcels::evaporation::{
    EvaporationControls, LiquidProperties, MassTransfer, SaturationCurve,
};
use crate::parcels::impact::{SplashCriterion, SurfaceTension, WallImpactControls};
use crate::parcels::{DragModel, Injector, ParcelControls, ParcelPhysics, WallAction};
use crate::{Scalar, Vec3};

/// The dictionary route's file inside a case directory.
pub const FILE: &str = "constant/cloudProperties";

/// The `parcels` block. Every key but `capacity` and `injectors` is optional,
/// and an absent key is the library's `Default` (SPEC-LIT §137.2), except
/// where §137.6 says otherwise.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct JsonParcels {
    /// Slots in the parcel pool (`ParcelControls::capacity`).
    pub capacity: usize,
    /// `none` | `stokes` | `schillerNaumann` (default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drag: Option<String>,
    /// `inert` (default) | `heating` | `evaporating`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physics: Option<String>,
    /// `remove` (default) | `rebound` | `stick` | `weber`; `film`, `spread`
    /// and `splash` are refused with their SPEC-LIT §78 reasons.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall: Option<String>,
    /// Normal restitution, `[0, 1]`; read by `rebound` and `weber` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restitution: Option<f64>,
    /// Tangential loss, `[0, 1]`; read by `rebound` and `weber` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tangential_loss: Option<f64>,
    /// m/s2. Absent: the case's `physics.gravity` on the JSONC route.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gravity: Option<[f64; 3]>,
    /// kg/m3.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rho_liquid: Option<f64>,
    /// Pa s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mu_gas: Option<f64>,
    /// J/(kg K); read by `heating` and `evaporating` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub c_liquid: Option<f64>,
    /// W/(m K); read by `heating` and `evaporating` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub k_gas: Option<f64>,
    /// J/(kg K); read by `heating` and `evaporating` only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cp_gas: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub added_mass: Option<bool>,
    /// The motion sub-step's fraction of a cell (66.7).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfl: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_substeps: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_walk: Option<u32>,
    /// Blocks in the persistent grid; absent derives it from `capacity`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persistent_blocks: Option<u32>,
    /// SPEC-LIT §76; only with `physics evaporating`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaporation: Option<JsonEvaporation>,
    /// SPEC-LIT §78; only with `wall weber`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub impact: Option<JsonImpact>,
    /// SPEC-LIT §68.6 and §77; absent is no coupling at all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coupling: Option<JsonCoupling>,
    /// One entry per nozzle, at least one (SPEC-LIT §66.8).
    pub injectors: Vec<JsonInjector>,
}

/// `parcels.evaporation` - `EvaporationControls`, field by field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct JsonEvaporation {
    /// `clausiusClapeyron` | `hylandWexler` (default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saturation: Option<String>,
    /// `ranzMarshall` | `spalding` | `abramzonSirignano` (default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transfer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub liquid: Option<JsonLiquid>,
    /// Carrier-gas molar mass, kg/mol.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub w_carrier: Option<f64>,
    /// Thermodynamic pressure, Pa.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p_ambient: Option<f64>,
    /// The largest fraction of `d^2` one sub-step may remove (76.10).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfl: Option<f64>,
}

/// `parcels.evaporation.liquid` - a preset (`water`, the default, or
/// `benzene`) and any of its nine properties overridden.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct JsonLiquid {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset: Option<String>,
    /// kg/mol.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub w_vapour: Option<f64>,
    /// K.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t_boil: Option<f64>,
    /// Pa.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p_boil: Option<f64>,
    /// J/kg.
    #[serde(rename = "hVBoil", default, skip_serializing_if = "Option::is_none")]
    pub h_v_boil: Option<f64>,
    /// K.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t_crit: Option<f64>,
    /// m2/s at `tRefD` and 1 atm.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub d_vapour: Option<f64>,
    /// K.
    #[serde(rename = "tRefD", default, skip_serializing_if = "Option::is_none")]
    pub t_ref_d: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub d_exponent: Option<f64>,
    /// J/(kg K).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cp_vapour: Option<f64>,
}

/// `parcels.impact` - `WallImpactControls`, field by field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct JsonImpact {
    /// `mundo` (default) | `baiGosman`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub splash: Option<String>,
    /// `constant` (default) | `iapwsR1-76`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tension: Option<String>,
    /// N/m.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sigma: Option<f64>,
    /// Pa s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mu_liquid: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub we_stick: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub we_spread: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub k_crit: Option<f64>,
    #[serde(rename = "splashA", default, skip_serializing_if = "Option::is_none")]
    pub splash_a: Option<f64>,
}

/// `parcels.coupling` - `CouplingControls`, and the species `vapour` names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct JsonCoupling {
    /// `off` (default) | `explicit` | `semiImplicit`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub momentum: Option<String>,
    /// `off` (default) | `explicit` | `semiImplicit`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy: Option<String>,
    /// `none` (default) | `evaporation`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mass: Option<String>,
    /// The species that receives the vapour; required with `mass
    /// evaporation`, and one of `species.names`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vapour: Option<String>,
}

/// One `parcels.injectors[]` entry - an `Injector`. `position` and
/// `massFlow` are required; every other field is `Injector::default()`'s.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct JsonInjector {
    /// m.
    pub position: [f64; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axis: Option<[f64; 3]>,
    /// rad.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cone_half_angle: Option<f64>,
    /// m.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standoff: Option<f64>,
    /// m/s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    /// m.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diameter: Option<f64>,
    /// K.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// kg/s.
    pub mass_flow: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parcels_per_event: Option<u32>,
    /// s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval: Option<f64>,
}

/// What [`JsonParcels::lower`] produces: the library's three settings, and
/// the species the vapour goes to.
#[derive(Debug, Clone, PartialEq)]
pub struct LoweredParcels {
    pub controls: ParcelControls,
    pub injectors: Vec<Injector>,
    pub coupling: CouplingControls,
    /// `coupling.vapour`; `Some` exactly when `coupling.mass` is on.
    pub vapour: Option<String>,
}

/// The refusal every driver gives a case that states parcels (SPEC-LIT
/// §137.2, §137.6). `source` names where the case stated them.
#[must_use]
pub fn refuse_parcel_case(source: &str) -> Error {
    Error::Refused(format!(
        "{source}: this case states parcels. ofgpu-lowmach is the one driver that \
         carries parcels (docs/17 section 10.2), and it drives them from a case only from \
         CHR-26 on; until then every driver refuses a parcel case rather than run it \
         without its parcels (SPEC-LIT §137.2, §137.6)"
    ))
}

fn keyed(key: &str, e: Error) -> Error {
    match e {
        Error::Refused(m) => Error::Refused(format!("parcels.{key}: {m}")),
        Error::Config(m) => Error::Config(format!("parcels.{key}: {m}")),
        other => other,
    }
}

fn refused(key: &str, msg: &str) -> Error {
    Error::Refused(format!("parcels.{key}: {msg} (SPEC-LIT §137.2, §137.6)"))
}

/// A model name through the library's `from_name`, the first letter lowered
/// so the table's capitalised spelling is the library's camelCase one.
fn pick<T>(key: &str, v: &Option<String>, from: fn(&str) -> Result<T>, d: T) -> Result<T> {
    let Some(s) = v else { return Ok(d) };
    let mut c = s.chars();
    let name = match c.next() {
        Some(f) => f.to_lowercase().collect::<String>() + c.as_str(),
        None => String::new(),
    };
    from(&name).map_err(|e| keyed(key, e))
}

fn sc(v: Option<f64>, d: Scalar) -> Scalar {
    v.map_or(d, |x| x as Scalar)
}

fn v3(v: Option<[f64; 3]>, d: Vec3) -> Vec3 {
    v.map_or(d, |g| Vec3::new(g[0] as Scalar, g[1] as Scalar, g[2] as Scalar))
}

fn liquid_preset(name: Option<&str>) -> Result<LiquidProperties> {
    match name {
        None | Some("water") => Ok(LiquidProperties::water()),
        Some("benzene") => Ok(LiquidProperties::benzene()),
        Some(other) => crate::io::contract::unsupported(
            "parcels.evaporation.liquid.preset",
            other,
            &["water", "benzene"],
            "water",
            LiquidProperties::water(),
        ),
    }
}

impl JsonParcels {
    /// The one lowering of both routes (SPEC-LIT §137.2). `species` is the
    /// case's `species.names` (`None` when the case has no `species` block);
    /// `gravity` is the case's gravity, used when the block states none
    /// (`None` keeps the library's).
    pub fn lower(
        &self,
        species: Option<&[String]>,
        gravity: Option<[f64; 3]>,
    ) -> Result<LoweredParcels> {
        let d = ParcelControls::default();
        let drag = pick("drag", &self.drag, DragModel::from_name, d.drag)?;
        let physics = pick("physics", &self.physics, ParcelPhysics::from_name, d.physics)?;
        let wall = pick("wall", &self.wall, WallAction::from_name, d.wall)?;

        // §13.4.1: a setting the selected model would not read is refused,
        // because it would be read and then ignored.
        if !matches!(wall, WallAction::Rebound | WallAction::Weber) {
            for (key, set) in [
                ("restitution", self.restitution.is_some()),
                ("tangentialLoss", self.tangential_loss.is_some()),
            ] {
                if set {
                    return Err(refused(
                        key,
                        &format!("is read by wall rebound or weber only, and wall is {}", wall.name()),
                    ));
                }
            }
        }
        if !physics.is_thermal() {
            for (key, set) in [
                ("cLiquid", self.c_liquid.is_some()),
                ("kGas", self.k_gas.is_some()),
                ("cpGas", self.cp_gas.is_some()),
            ] {
                if set {
                    return Err(refused(
                        key,
                        "is read by physics heating or evaporating only, and physics is inert",
                    ));
                }
            }
        }
        if self.evaporation.is_some() && physics != ParcelPhysics::Evaporating {
            return Err(refused(
                "evaporation",
                &format!("is read by physics evaporating only, and physics is {}", physics.name()),
            ));
        }
        if self.impact.is_some() && !wall.reads_impact_model() {
            return Err(refused(
                "impact",
                &format!("is read by wall weber only, and wall is {}", wall.name()),
            ));
        }

        let evaporation = match &self.evaporation {
            None => d.evaporation,
            Some(e) => lower_evaporation(e, d.evaporation)?,
        };
        let impact = match &self.impact {
            None => d.impact,
            Some(i) => lower_impact(i, d.impact)?,
        };

        let controls = ParcelControls {
            capacity: self.capacity,
            drag,
            physics,
            wall,
            restitution: sc(self.restitution, d.restitution),
            tangential_loss: sc(self.tangential_loss, d.tangential_loss),
            gravity: v3(self.gravity.or(gravity), d.gravity),
            rho_liquid: sc(self.rho_liquid, d.rho_liquid),
            mu_gas: sc(self.mu_gas, d.mu_gas),
            c_liquid: sc(self.c_liquid, d.c_liquid),
            k_gas: sc(self.k_gas, d.k_gas),
            cp_gas: sc(self.cp_gas, d.cp_gas),
            added_mass: self.added_mass.unwrap_or(d.added_mass),
            cfl: sc(self.cfl, d.cfl),
            max_substeps: self.max_substeps.unwrap_or(d.max_substeps),
            max_walk: self.max_walk.unwrap_or(d.max_walk),
            evaporation,
            impact,
            persistent_blocks: self.persistent_blocks.or(d.persistent_blocks),
        };
        controls.validate()?;

        let (coupling, vapour) = self.lower_coupling(species)?;

        if self.injectors.is_empty() {
            return Err(refused(
                "injectors",
                "is empty; a parcels block with no injector emits nothing",
            ));
        }
        let di = Injector::default();
        let injectors = self
            .injectors
            .iter()
            .map(|j| Injector {
                position: v3(Some(j.position), di.position),
                axis: v3(j.axis, di.axis),
                cone_half_angle: sc(j.cone_half_angle, di.cone_half_angle),
                standoff: sc(j.standoff, di.standoff),
                speed: sc(j.speed, di.speed),
                diameter: sc(j.diameter, di.diameter),
                temperature: sc(j.temperature, di.temperature),
                mass_flow: j.mass_flow as Scalar,
                parcels_per_event: j.parcels_per_event.unwrap_or(di.parcels_per_event),
                interval: sc(j.interval, di.interval),
                // SPEC-LIT 147.1: the case blocks of §137.6 do not reach
                // the size distribution yet; it is a library field, and an
                // injector read from a case is Monodisperse.
                size: di.size,
            })
            .collect();

        Ok(LoweredParcels { controls, injectors, coupling, vapour })
    }

    fn lower_coupling(
        &self,
        species: Option<&[String]>,
    ) -> Result<(CouplingControls, Option<String>)> {
        let off = CouplingControls {
            momentum: CouplingMode::Off,
            energy: CouplingMode::Off,
            mass: MassCoupling::None,
        };
        let Some(c) = &self.coupling else { return Ok((off, None)) };
        let coupling = CouplingControls {
            momentum: pick("coupling.momentum", &c.momentum, CouplingMode::from_name, off.momentum)?,
            energy: pick("coupling.energy", &c.energy, CouplingMode::from_name, off.energy)?,
            mass: pick("coupling.mass", &c.mass, MassCoupling::from_name, off.mass)?,
        };
        if !coupling.mass.is_on() {
            if c.vapour.is_some() {
                return Err(refused(
                    "coupling.vapour",
                    "is read with coupling.mass evaporation only, and mass is none",
                ));
            }
            return Ok((coupling, None));
        }
        let Some(names) = species else {
            return Err(refused(
                "coupling.mass",
                &format!(
                    "is {}, which needs a species block to put the vapour in, and this \
                     case has none",
                    coupling.mass.name()
                ),
            ));
        };
        let Some(v) = &c.vapour else {
            return Err(refused(
                "coupling.vapour",
                "is missing; coupling.mass evaporation names the species that receives \
                 the vapour",
            ));
        };
        if !names.iter().any(|n| n == v) {
            return Err(refused(
                "coupling.vapour",
                &format!("\"{v}\" is not in species.names [{}]", names.join(", ")),
            ));
        }
        Ok((coupling, Some(v.clone())))
    }
}

fn lower_evaporation(e: &JsonEvaporation, d: EvaporationControls) -> Result<EvaporationControls> {
    let (l, base) = match &e.liquid {
        None => (None, d.liquid),
        Some(l) => (Some(l), liquid_preset(l.preset.as_deref())?),
    };
    let f = |get: fn(&JsonLiquid) -> Option<f64>, dv: Scalar| sc(l.and_then(get), dv);
    let liquid = LiquidProperties {
        w_vapour: f(|l| l.w_vapour, base.w_vapour),
        t_boil: f(|l| l.t_boil, base.t_boil),
        p_boil: f(|l| l.p_boil, base.p_boil),
        h_v_boil: f(|l| l.h_v_boil, base.h_v_boil),
        t_crit: f(|l| l.t_crit, base.t_crit),
        d_vapour: f(|l| l.d_vapour, base.d_vapour),
        t_ref_d: f(|l| l.t_ref_d, base.t_ref_d),
        d_exponent: f(|l| l.d_exponent, base.d_exponent),
        cp_vapour: f(|l| l.cp_vapour, base.cp_vapour),
    };
    Ok(EvaporationControls {
        saturation: pick("evaporation.saturation", &e.saturation, SaturationCurve::from_name, d.saturation)?,
        transfer: pick("evaporation.transfer", &e.transfer, MassTransfer::from_name, d.transfer)?,
        liquid,
        w_carrier: sc(e.w_carrier, d.w_carrier),
        p_ambient: sc(e.p_ambient, d.p_ambient),
        cfl: sc(e.cfl, d.cfl),
    })
}

fn lower_impact(i: &JsonImpact, d: WallImpactControls) -> Result<WallImpactControls> {
    Ok(WallImpactControls {
        splash: pick("impact.splash", &i.splash, SplashCriterion::from_name, d.splash)?,
        tension: pick("impact.tension", &i.tension, SurfaceTension::from_name, d.tension)?,
        sigma: sc(i.sigma, d.sigma),
        mu_liquid: sc(i.mu_liquid, d.mu_liquid),
        we_stick: sc(i.we_stick, d.we_stick),
        we_spread: sc(i.we_spread, d.we_spread),
        k_crit: sc(i.k_crit, d.k_crit),
        splash_a: sc(i.splash_a, d.splash_a),
    })
}

// ==========================================================================
//  The dictionary route: constant/cloudProperties
// ==========================================================================

/// One dictionary entry as the JSON value the JSONC route would hold:
/// `(a b c)` is an array of numbers, `addedMass` a switch, a number (after
/// any `[dimensions]`) a number, and anything else a word.
fn entry_value(what: &str, key: &str, leaf: &str, raw: &str) -> Result<serde_json::Value> {
    use serde_json::Value;
    let t = raw.trim().trim_matches('"');
    let bad = |m: &str| Error::Config(format!("{what}: {key} \"{t}\" {m} (SPEC-LIT §137.6)"));
    if let Some(inner) = t.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
        let mut out = Vec::new();
        for tok in inner.split_whitespace() {
            let x: f64 = tok.parse().map_err(|_| bad("is not a list of numbers"))?;
            out.push(serde_json::json!(x));
        }
        return Ok(Value::Array(out));
    }
    if leaf == "addedMass" {
        return match t.to_ascii_lowercase().as_str() {
            "yes" | "true" | "on" => Ok(Value::Bool(true)),
            "no" | "false" | "off" => Ok(Value::Bool(false)),
            _ => Err(bad("is not a switch (yes/no, true/false, on/off)")),
        };
    }
    if let Ok(i) = t.parse::<i64>() {
        return Ok(serde_json::json!(i));
    }
    if let Ok(x) = t.parse::<f64>() {
        return if x.is_finite() { Ok(serde_json::json!(x)) } else { Err(bad("is not finite")) };
    }
    if t.starts_with('[') {
        return match last_number(t) {
            Some(x) if x.is_finite() => Ok(serde_json::json!(x)),
            _ => Err(bad("has no finite number after its dimensions")),
        };
    }
    Ok(Value::String(t.to_string()))
}

/// One dictionary as the JSON object tree the JSONC route reads: every
/// entry keyed by its flattened path, `FoamFile` excluded, an `(a b c)`
/// list a number array, a switch a bool, a number a number, anything else a
/// word. Shared by the `parcels` block's own lowering and the cloud's
/// `dustCloud` route (`case_dust::dust_cloud_of_dict`), so the two cannot
/// drift (SPEC-LIT §159.9). The `injectors` and `dustCloud`
/// sub-dictionaries are left as the plain objects they arrive as.
pub(crate) fn dict_tree(
    d: &FoamDict,
    what: &str,
) -> Result<serde_json::Map<String, serde_json::Value>> {
    use serde_json::{Map, Value};
    let mut root = Map::new();
    for (key, raw) in &d.entries {
        if key == "FoamFile" || key.starts_with("FoamFile/") {
            continue;
        }
        let parts: Vec<&str> = key.split('/').filter(|s| !s.is_empty()).collect();
        let Some((leaf, dirs)) = parts.split_last() else { continue };
        let mut node = &mut root;
        for dir in dirs {
            let slot = node.entry((*dir).to_string()).or_insert_with(|| Value::Object(Map::new()));
            node = match slot {
                Value::Object(m) => m,
                _ => {
                    return Err(Error::Config(format!(
                        "{what}: {dir} is both an entry and a sub-dictionary (SPEC-LIT §137.6)"
                    )))
                }
            };
        }
        if key.ends_with('/') {
            // An empty sub-dictionary's marker.
            node.entry((*leaf).to_string()).or_insert_with(|| Value::Object(Map::new()));
            continue;
        }
        node.insert((*leaf).to_string(), entry_value(what, key, leaf, raw)?);
    }
    Ok(root)
}

impl JsonParcels {
    pub fn read(path: &Path) -> Result<Self> {
        let d = FoamDict::read(path)?;
        Self::from_dict(&d, &path.display().to_string())
    }

    pub fn parse(src: &str, what: &str) -> Result<Self> {
        let d = FoamDict::parse(src, what)?;
        Self::from_dict(&d, what)
    }

    /// The dictionary as the JSONC block, through the same deserialiser:
    /// an unknown key is refused by the same `deny_unknown_fields`. The
    /// `injectors` sub-dictionaries are taken in their name order. A
    /// `dustCloud` sub-dictionary is NOT part of this block - it is the
    /// cloud's route (SPEC-LIT §159.9, `case_dust::dust_cloud_of_dict`) -
    /// so it is removed before the deserialiser, whose
    /// `deny_unknown_fields` would refuse it here.
    pub fn from_dict(d: &FoamDict, what: &str) -> Result<Self> {
        use serde_json::Value;
        let mut root = dict_tree(d, what)?;
        root.remove("dustCloud");
        if let Some(inj) = root.remove("injectors") {
            let list = match inj {
                Value::Object(m) => m.into_iter().map(|(_, v)| v).collect(),
                _ => {
                    return Err(Error::Config(format!(
                        "{what}: injectors must be a dictionary of named injector \
                         sub-dictionaries (SPEC-LIT §137.6)"
                    )))
                }
            };
            root.insert("injectors".to_string(), Value::Array(list));
        }
        serde_json::from_value(Value::Object(root))
            .map_err(|e| Error::Refused(format!("{what}: {e} (SPEC-LIT §137.2, §137.6)")))
    }
}

/// The `parcels` block on the dictionary route, ignoring the cloud's
/// `dustCloud` sub-dictionary (SPEC-LIT §159.9): `None` when the file
/// states no `parcels` keys at all beside `dustCloud` - a file that states
/// only `dustCloud` has no `parcels` block - else the same lowering
/// `JsonParcels::from_dict` gives.
pub fn parcels_of_dict(d: &FoamDict, what: &str) -> Result<Option<JsonParcels>> {
    let mut root = dict_tree(d, what)?;
    root.remove("dustCloud");
    if root.is_empty() {
        return Ok(None);
    }
    JsonParcels::from_dict(d, what).map(Some)
}

/// The dictionary route: `constant/cloudProperties` when the case has one.
/// A file holding only the cloud's `dustCloud` is `Ok(None)` here -
/// `case_dust::read_dust_cloud` reads that one.
pub fn read_cloud_properties(case_dir: &Path) -> Result<Option<JsonParcels>> {
    let p = case_dir.join(FILE);
    if !p.exists() {
        return Ok(None);
    }
    let d = FoamDict::read(&p)?;
    parcels_of_dict(&d, &p.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::contract::{permissive_test_guard, set_permissive};

    /// Every key of the block set to a value that is not its default.
    const EVERY: &str = r#"{
        "capacity": 4096, "drag": "Stokes", "physics": "evaporating", "wall": "Weber",
        "restitution": 0.6, "tangentialLoss": 0.25, "gravity": [0.0, -9.81, 0.0],
        "rhoLiquid": 998.2, "muGas": 1.85e-5, "cLiquid": 4180.0, "kGas": 0.0262,
        "cpGas": 1007.0, "addedMass": true, "cfl": 0.5, "maxSubsteps": 32,
        "maxWalk": 24, "persistentBlocks": 8,
        "evaporation": {
            "saturation": "clausiusClapeyron", "transfer": "spalding",
            "liquid": { "preset": "water", "wVapour": 0.018, "tBoil": 373.0,
                "pBoil": 101300.0, "hVBoil": 2.25e6, "tCrit": 647.0, "dVapour": 2.5e-5,
                "tRefD": 300.0, "dExponent": 2.0, "cpVapour": 1900.0 },
            "wCarrier": 0.029, "pAmbient": 100000.0, "cfl": 0.2
        },
        "impact": { "splash": "baiGosman", "tension": "iapwsR1-76", "sigma": 0.07,
            "muLiquid": 1.1e-3, "weStick": 3.0, "weSpread": 25.0, "kCrit": 60.0,
            "splashA": 2700.0 },
        "coupling": { "momentum": "SemiImplicit", "energy": "explicit",
            "mass": "evaporation", "vapour": "H2O" },
        "injectors": [
            { "position": [0.1, 0.2, 0.3], "axis": [0.0, 0.0, 1.0], "coneHalfAngle": 0.3,
              "standoff": 0.002, "speed": 12.0, "diameter": 5e-5, "temperature": 300.0,
              "massFlow": 1e-3, "parcelsPerEvent": 4, "interval": 0.01 },
            { "position": [0.5, 0.5, 0.5], "massFlow": 2e-3 }
        ]
    }"#;

    /// The same block on the dictionary route.
    const EVERY_DICT: &str = "FoamFile { format ascii; class dictionary; object cloudProperties; }
capacity 4096; drag Stokes; physics evaporating; wall Weber;
restitution 0.6; tangentialLoss 0.25; gravity (0 -9.81 0);
rhoLiquid [1 -3 0 0 0 0 0] 998.2; muGas 1.85e-5; cLiquid 4180; kGas 0.0262;
cpGas 1007; addedMass yes; cfl 0.5; maxSubsteps 32; maxWalk 24; persistentBlocks 8;
evaporation
{
    saturation clausiusClapeyron; transfer spalding;
    liquid { preset water; wVapour 0.018; tBoil 373; pBoil 101300; hVBoil 2.25e6;
             tCrit 647; dVapour 2.5e-5; tRefD 300; dExponent 2; cpVapour 1900; }
    wCarrier 0.029; pAmbient 1e5; cfl 0.2;
}
impact { splash baiGosman; tension iapwsR1-76; sigma 0.07; muLiquid 1.1e-3;
         weStick 3; weSpread 25; kCrit 60; splashA 2700; }
coupling { momentum SemiImplicit; energy explicit; mass evaporation; vapour H2O; }
injectors
{
    nozzle1 { position (0.1 0.2 0.3); axis (0 0 1); coneHalfAngle 0.3; standoff 0.002;
              speed 12; diameter 5e-5; temperature 300; massFlow 1e-3;
              parcelsPerEvent 4; interval 0.01; }
    nozzle2 { position (0.5 0.5 0.5); massFlow 2e-3; }
}
";

    fn names() -> Vec<String> {
        ["N2", "O2", "H2O"].iter().map(|s| s.to_string()).collect()
    }

    fn s(x: f64) -> Scalar {
        x as Scalar
    }

    fn v(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3::new(s(x), s(y), s(z))
    }

    fn expected() -> LoweredParcels {
        let di = Injector::default();
        LoweredParcels {
            controls: ParcelControls {
                capacity: 4096,
                drag: DragModel::Stokes,
                physics: ParcelPhysics::Evaporating,
                wall: WallAction::Weber,
                restitution: s(0.6),
                tangential_loss: s(0.25),
                gravity: v(0.0, -9.81, 0.0),
                rho_liquid: s(998.2),
                mu_gas: s(1.85e-5),
                c_liquid: s(4180.0),
                k_gas: s(0.0262),
                cp_gas: s(1007.0),
                added_mass: true,
                cfl: s(0.5),
                max_substeps: 32,
                max_walk: 24,
                evaporation: EvaporationControls {
                    saturation: SaturationCurve::ClausiusClapeyron,
                    transfer: MassTransfer::Spalding,
                    liquid: LiquidProperties {
                        w_vapour: s(0.018),
                        t_boil: s(373.0),
                        p_boil: s(101300.0),
                        h_v_boil: s(2.25e6),
                        t_crit: s(647.0),
                        d_vapour: s(2.5e-5),
                        t_ref_d: s(300.0),
                        d_exponent: s(2.0),
                        cp_vapour: s(1900.0),
                    },
                    w_carrier: s(0.029),
                    p_ambient: s(1e5),
                    cfl: s(0.2),
                },
                impact: WallImpactControls {
                    splash: SplashCriterion::BaiGosman,
                    tension: SurfaceTension::IapwsR176,
                    sigma: s(0.07),
                    mu_liquid: s(1.1e-3),
                    we_stick: s(3.0),
                    we_spread: s(25.0),
                    k_crit: s(60.0),
                    splash_a: s(2700.0),
                },
                persistent_blocks: Some(8),
            },
            injectors: vec![
                Injector {
                    position: v(0.1, 0.2, 0.3),
                    axis: v(0.0, 0.0, 1.0),
                    cone_half_angle: s(0.3),
                    standoff: s(0.002),
                    speed: s(12.0),
                    diameter: s(5e-5),
                    temperature: s(300.0),
                    mass_flow: s(1e-3),
                    parcels_per_event: 4,
                    interval: s(0.01),
                    size: crate::parcels::SizeDistribution::Monodisperse,
                },
                Injector { position: v(0.5, 0.5, 0.5), mass_flow: s(2e-3), ..di },
            ],
            coupling: CouplingControls {
                momentum: CouplingMode::SemiImplicit,
                energy: CouplingMode::Explicit,
                mass: MassCoupling::Evaporation,
            },
            vapour: Some("H2O".to_string()),
        }
    }

    /// SPEC-LIT §137.3 `case_parcels_lowers_every_field`: one case sets every
    /// `ParcelControls`, `CouplingControls` and `Injector` field to a value
    /// that is not its default, and each is read back; both routes give the
    /// same lowering, and the block survives a serde round trip.
    #[test]
    fn case_parcels_lowers_every_field() {
        let _g = permissive_test_guard();
        set_permissive(false);
        let sp = names();
        let json: JsonParcels = serde_json::from_str(EVERY).expect("the JSONC block parses");
        let got = json.lower(Some(&sp), None).expect("the JSONC block lowers");
        let want = expected();
        assert_eq!(got, want);

        // Every field differs from the default it would otherwise take.
        let (c, d) = (&got.controls, ParcelControls::default());
        assert_ne!(c.capacity, d.capacity);
        assert_ne!(c.drag, d.drag);
        assert_ne!(c.physics, d.physics);
        assert_ne!(c.wall, d.wall);
        assert_ne!(c.restitution, d.restitution);
        assert_ne!(c.tangential_loss, d.tangential_loss);
        assert_ne!(c.gravity, d.gravity);
        assert_ne!(c.rho_liquid, d.rho_liquid);
        assert_ne!(c.mu_gas, d.mu_gas);
        assert_ne!(c.c_liquid, d.c_liquid);
        assert_ne!(c.k_gas, d.k_gas);
        assert_ne!(c.cp_gas, d.cp_gas);
        assert_ne!(c.added_mass, d.added_mass);
        assert_ne!(c.cfl, d.cfl);
        assert_ne!(c.max_substeps, d.max_substeps);
        assert_ne!(c.max_walk, d.max_walk);
        assert_ne!(c.persistent_blocks, d.persistent_blocks);
        let (e, de) = (&c.evaporation, d.evaporation);
        assert_ne!(e.saturation, de.saturation);
        assert_ne!(e.transfer, de.transfer);
        assert_ne!(e.w_carrier, de.w_carrier);
        assert_ne!(e.p_ambient, de.p_ambient);
        assert_ne!(e.cfl, de.cfl);
        let (l, dl) = (&e.liquid, de.liquid);
        assert_ne!(l.w_vapour, dl.w_vapour);
        assert_ne!(l.t_boil, dl.t_boil);
        assert_ne!(l.p_boil, dl.p_boil);
        assert_ne!(l.h_v_boil, dl.h_v_boil);
        assert_ne!(l.t_crit, dl.t_crit);
        assert_ne!(l.d_vapour, dl.d_vapour);
        assert_ne!(l.t_ref_d, dl.t_ref_d);
        assert_ne!(l.d_exponent, dl.d_exponent);
        assert_ne!(l.cp_vapour, dl.cp_vapour);
        let (i, di) = (&c.impact, d.impact);
        assert_ne!(i.splash, di.splash);
        assert_ne!(i.tension, di.tension);
        assert_ne!(i.sigma, di.sigma);
        assert_ne!(i.mu_liquid, di.mu_liquid);
        assert_ne!(i.we_stick, di.we_stick);
        assert_ne!(i.we_spread, di.we_spread);
        assert_ne!(i.k_crit, di.k_crit);
        assert_ne!(i.splash_a, di.splash_a);
        let (k, dk) = (&got.coupling, CouplingControls::default());
        assert_ne!(k.momentum, dk.momentum);
        assert_ne!(k.energy, dk.energy);
        assert_ne!(k.mass, dk.mass);
        let (j, dj) = (&got.injectors[0], Injector::default());
        assert_ne!(j.position, dj.position);
        assert_ne!(j.axis, dj.axis);
        assert_ne!(j.cone_half_angle, dj.cone_half_angle);
        assert_ne!(j.standoff, dj.standoff);
        assert_ne!(j.speed, dj.speed);
        assert_ne!(j.diameter, dj.diameter);
        assert_ne!(j.temperature, dj.temperature);
        assert_ne!(j.mass_flow, dj.mass_flow);
        assert_ne!(j.parcels_per_event, dj.parcels_per_event);
        assert_ne!(j.interval, dj.interval);

        // The dictionary route is the same lowering.
        let dict = JsonParcels::parse(EVERY_DICT, "cloudProperties").expect("the dictionary parses");
        assert_eq!(dict.lower(Some(&sp), None).expect("the dictionary lowers"), want);

        // Serde round trip.
        let text = serde_json::to_string(&json).expect("serialises");
        let back: JsonParcels = serde_json::from_str(&text).expect("reads back");
        assert_eq!(back, json);

        // Absent keys are the library's defaults; gravity falls back to the
        // case's; absent coupling is no coupling at all.
        let min: JsonParcels =
            serde_json::from_str(r#"{ "capacity": 16, "injectors": [ { "position": [0,0,0], "massFlow": 0.0 } ] }"#)
                .expect("the minimal block parses");
        let l = min.lower(None, Some([1.0, 2.0, 3.0])).expect("the minimal block lowers");
        let mut want_min = ParcelControls { capacity: 16, ..ParcelControls::default() };
        want_min.gravity = v(1.0, 2.0, 3.0);
        assert_eq!(l.controls, want_min);
        assert_eq!(l.injectors, vec![Injector::default()]);
        assert_eq!(
            l.coupling,
            CouplingControls { momentum: CouplingMode::Off, energy: CouplingMode::Off, mass: MassCoupling::None }
        );
        assert_eq!(l.vapour, None);
        assert_eq!(min.lower(None, None).expect("lowers").controls.gravity, ParcelControls::default().gravity);
    }

    fn err_of(block: &str, species: Option<&[String]>) -> String {
        let p: JsonParcels = match serde_json::from_str(block) {
            Ok(p) => p,
            Err(e) => return e.to_string(),
        };
        match p.lower(species, None) {
            Ok(_) => panic!("{block} lowered; it should be refused"),
            Err(e) => e.to_string(),
        }
    }

    /// SPEC-LIT §137.3 `case_parcels_refusals`: every refusal of §137.2 and
    /// §137.6, each naming the key it refuses.
    #[test]
    fn case_parcels_refusals() {
        let _g = permissive_test_guard();
        set_permissive(false);
        let sp = names();
        let inj = r#""injectors": [ { "position": [0,0,0], "massFlow": 1e-3 } ]"#;
        let with = |body: &str| format!("{{ \"capacity\": 64, {body}, {inj} }}");

        // film and splash keep their §78 reasons.
        let e = err_of(&with(r#""wall": "film""#), None);
        assert!(e.contains("parcels.wall") && e.contains("film") && e.contains("78.11"), "{e}");
        let e = err_of(&with(r#""wall": "splash""#), None);
        assert!(e.contains("parcels.wall") && e.contains("splash") && e.contains("78.7"), "{e}");
        let e = err_of(&with(r#""wall": "spread""#), None);
        assert!(e.contains("parcels.wall") && e.contains("78.11"), "{e}");

        // Unknown names, through each library menu.
        for (body, key) in [
            (r#""drag": "putnam""#, "parcels.drag"),
            (r#""physics": "boiling""#, "parcels.physics"),
            (r#""coupling": { "momentum": "twoWay" }"#, "parcels.coupling.momentum"),
            (r#""coupling": { "energy": "twoWay" }"#, "parcels.coupling.energy"),
            (r#""coupling": { "mass": "condensation" }"#, "parcels.coupling.mass"),
            (r#""physics": "evaporating", "evaporation": { "saturation": "antoine" }"#, "parcels.evaporation.saturation"),
            (r#""physics": "evaporating", "evaporation": { "transfer": "abramzonSirignanoFilm" }"#, "parcels.evaporation.transfer"),
            (r#""physics": "evaporating", "evaporation": { "liquid": { "preset": "ethanol" } }"#, "parcels.evaporation.liquid.preset"),
            (r#""wall": "weber", "impact": { "splash": "stanton" }"#, "parcels.impact.splash"),
            (r#""wall": "weber", "impact": { "tension": "linear" }"#, "parcels.impact.tension"),
        ] {
            let e = err_of(&with(body), None);
            assert!(e.contains(key), "{body}: {e}");
        }

        // Unknown keys, at every level.
        for body in [
            r#""bogus": 1"#,
            r#""physics": "evaporating", "evaporation": { "bogus": 1 }"#,
            r#""physics": "evaporating", "evaporation": { "liquid": { "bogus": 1 } }"#,
            r#""wall": "weber", "impact": { "bogus": 1 }"#,
            r#""coupling": { "bogus": "x" }"#,
        ] {
            let e = err_of(&with(body), None);
            assert!(e.contains("bogus"), "{body}: {e}");
        }
        let e = err_of(r#"{ "capacity": 4, "injectors": [ { "position": [0,0,0], "massFlow": 1, "bogus": 2 } ] }"#, None);
        assert!(e.contains("bogus"), "{e}");
        let e = err_of(r#"{ "capacity": 4, "injectors": [ { "position": [0,0,0] } ] }"#, None);
        assert!(e.contains("massFlow"), "{e}");

        // Settings the selected model would not read.
        for (body, key) in [
            (r#""restitution": 0.5"#, "parcels.restitution"),
            (r#""wall": "stick", "tangentialLoss": 0.5"#, "parcels.tangentialLoss"),
            (r#""cLiquid": 4000"#, "parcels.cLiquid"),
            (r#""kGas": 0.03"#, "parcels.kGas"),
            (r#""cpGas": 1000"#, "parcels.cpGas"),
            (r#""physics": "heating", "evaporation": {}"#, "parcels.evaporation"),
            (r#""wall": "rebound", "impact": {}"#, "parcels.impact"),
        ] {
            let e = err_of(&with(body), None);
            assert!(e.contains(key), "{body}: {e}");
        }

        // The vapour and the species block.
        let mass = r#""physics": "evaporating", "coupling": { "energy": "explicit", "mass": "evaporation""#;
        let e = err_of(&with(&format!("{mass}, \"vapour\": \"H2O\" }}")), None);
        assert!(e.contains("parcels.coupling.mass") && e.contains("species block"), "{e}");
        let e = err_of(&with(&format!("{mass} }}")), Some(&sp));
        assert!(e.contains("parcels.coupling.vapour") && e.contains("missing"), "{e}");
        let e = err_of(&with(&format!("{mass}, \"vapour\": \"C7H16\" }}")), Some(&sp));
        assert!(e.contains("parcels.coupling.vapour") && e.contains("C7H16"), "{e}");
        let e = err_of(&with(r#""coupling": { "vapour": "H2O" }"#), Some(&sp));
        assert!(e.contains("parcels.coupling.vapour") && e.contains("mass is none"), "{e}");

        // No injector, and the library's own range checks.
        let e = err_of(r#"{ "capacity": 4, "injectors": [] }"#, None);
        assert!(e.contains("parcels.injectors"), "{e}");
        let e = err_of(&with(r#""capacity": 0"#).replacen("\"capacity\": 64, ", "", 1), None);
        assert!(e.contains("capacity"), "{e}");
        let e = err_of(&with(r#""rhoLiquid": -1"#), None);
        assert!(e.contains("rhoLiquid"), "{e}");

        // The dictionary route refuses an unknown key and a bad switch by name.
        let e = JsonParcels::parse("capacity 4; bogus 1; injectors { a { position (0 0 0); massFlow 0; } }", "cloudProperties")
            .unwrap_err()
            .to_string();
        assert!(e.contains("bogus"), "{e}");
        let e = JsonParcels::parse("capacity 4; addedMass maybe; injectors { a { position (0 0 0); massFlow 0; } }", "cloudProperties")
            .unwrap_err()
            .to_string();
        assert!(e.contains("addedMass"), "{e}");

        // A parcel case is refused by every driver until CHR-26, naming the one
        // driver that carries parcels.
        let e = refuse_parcel_case("parcels").to_string();
        assert!(e.contains("ofgpu-lowmach") && e.contains("CHR-26"), "{e}");

        // Through a whole JSONC case: the block lowers (its gravity is the
        // case's), the driver seam refuses it, a mass coupling is refused for
        // want of a species block, and a case without the block passes.
        let case = |parcels: &str| {
            format!(
                r#"{{
                "name": "parcelCase",
                "mesh": {{ "kind": "cartesian", "bounds": {{ "min": [0,0,0], "max": [1,1,1] }},
                    "cells": [2, 2, 2],
                    "boundaries": {{ "xmin": "w", "xmax": "w", "ymin": "w", "ymax": "w", "zmin": "w", "zmax": "w" }} }},
                "physics": {{ "gravity": [0, -9.81, 0],
                    "fluid": {{ "nu": 1e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 293.15 }},
                    "buoyancy": "densityRatio" }},
                "patches": [ {{ "match": ".*", "kind": "wall" }} ],
                "initial": {{ "U": [0,0,0], "p": 0.0 }},
                "numerics": {{ "algorithm": {{ "kind": "SIMPLE" }}, "ddt": "steadyState",
                    "div": {{ "default": "Gauss upwind" }}, "grad": "Gauss linear",
                    "laplacian": {{ "snGrad": "corrected", "nonOrthogonalCorrectors": 0 }},
                    "solvers": [] }},
                "run": {{ "endTime": 1.0, "deltaT": 1.0 }}{parcels}
            }}"#
            )
        };
        let lower = |t: String| crate::io::case_json::read_case_jsonc_str(&t, "parcelCase").and_then(|c| c.lower());
        let l = lower(case(&format!(", \"parcels\": {{ \"capacity\": 8, {inj} }}"))).expect("the parcel case lowers");
        assert_eq!(l.parcels.as_ref().expect("parcels lower").controls.gravity, v(0.0, -9.81, 0.0));
        let e = l.refuse_parcels().unwrap_err().to_string();
        assert!(e.contains("ofgpu-lowmach"), "{e}");
        assert!(lower(case("")).expect("the plain case lowers").refuse_parcels().is_ok());
        let e = match lower(case(&format!(
            ", \"parcels\": {{ \"capacity\": 8, {mass}, \"vapour\": \"H2O\" }}, {inj} }}"
        ))) {
            Ok(_) => panic!("a mass coupling without a species block lowered"),
            Err(e) => e.to_string(),
        };
        assert!(e.contains("parcels.coupling.mass") && e.contains("species block"), "{e}");

        // The dictionary route's driver seam: read_case_controls refuses a
        // case directory with constant/cloudProperties, naming the file.
        let dir = std::env::temp_dir().join(format!("ofgpu_chr25_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("constant/polyMesh")).expect("mkdir");
        std::fs::write(dir.join("constant/polyMesh/owner"), "").expect("owner");
        std::fs::write(dir.join(FILE), "capacity 4;").expect("cloudProperties");
        let e = match crate::io::case::read_case_controls(&dir) {
            Ok(_) => panic!("a case with constant/cloudProperties was read"),
            Err(e) => e.to_string(),
        };
        let _ = std::fs::remove_dir_all(&dir);
        assert!(e.contains(FILE) && e.contains("ofgpu-lowmach"), "{e}");
    }
}
