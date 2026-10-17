// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The `dust` case block and the FSC/TFC side of `combustion` - the DST-06
//! host-only lowering of SPEC-LIT §157.4.
//!
//! A dust case is a JSONC case stating `combustion.model` `FSC` or `TFC`
//! (§154) with a `dust` block naming the equivalent-gas thermo mode
//! (§152.9). [`lower_dust`] is the one lowering, called by
//! `JsonCase::lower` BEFORE the chemistry blocks: when it returns `Some`,
//! the case's gas is the two pseudo-species of §152.4 and `lower_blocks`
//! is not called at all. Every refusal is §13.4-shaped - a key, a sentence,
//! `(SPEC-LIT §157.4)`.
//!
//! What a case says here is only the HOST configuration: the model and its
//! constants (§151.5), the ignition source (§155.3), the flame-Courant
//! limit (155.9), the thermo mode and `T_u` (the case's `initial.T`), the
//! laminar burning velocity (152.13), the ρ-weighted SST settings (§156.3,
//! §156.4) and the `c` field's boundary conditions. The kernels, the
//! driver and the coupling are DST-06's later parts, and every driver but
//! the two that run the model refuses the case by name
//! ([`refuse_dust_case`]): `ofgpu-lowmach` the sealed one (§157.4),
//! `ofgpu-compressible` the mixture-gas one (§158.6).
//!
//! Written from:
//!   ofgpu SPEC-LIT.md §157.4 - the case blocks and their §13.4 refusals
//!   ofgpu SPEC-LIT.md §152.9, (152.13) - the two thermo modes and S_L(phi)
//!   ofgpu SPEC-LIT.md §151.5, §154 - the closures' stated inputs
//!   ofgpu SPEC-LIT.md §155.3, 155.9 - the ignition source and Co_f,max
//!   ofgpu SPEC-LIT.md §156.3, §156.4, §156.7 - the SST settings
//!
//! The specification was informed by a review of FSCDustFoam (Huang &
//! Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::dust::char::CharInputs;
use crate::dust::courant::CO_F_MAX;
use crate::dust::fsc::{CombustionModel, A_FSC};
use crate::dust::ignition::IgnitionSource;
use crate::dust::laminar::LaminarSpeed;
use crate::dust::particle::{Drying, ParticleInputs, ParticleState};
use crate::dust::seeding::rosin_rammler_bands;
use crate::dust::thermo::{
    EquivalentGas, FormulaInputs, N2_PER_O2_CENTI, PrescribedInputs, StarchFuel, tm4513_gas_set,
};
use crate::dust::vessel0d::{BurningLaw, UnburnedDensity};
use crate::error::{Error, Result};
use crate::io::case_chem::JsonCombustion;
use crate::io::case_json::{
    LoweredScalarField, PatchPresetKind, TurbulenceKind, patch_class, resolve_patch_rule,
    JsonPatchRule,
};
use crate::io::dict::FoamDict;
use crate::io::fields::PatchFieldSpec;
use crate::{DVec3, Scalar, Vec3};

const TAG: &str = "(SPEC-LIT §157.4)";
/// The cloud's section, which this file's cloud refusals name (§159.9).
const CLOUD_TAG: &str = "(SPEC-LIT §159.9)";
/// The progress variable's field name and its dimensionless units.
const C_FIELD: &str = "c";
const C_DIMS: &str = "[0 0 0 0 0 0 0]";
/// The exact fraction's widest denominator: 10^17 (SPEC-LIT §152.11 holds
/// phi as an exact integer fraction; a longer decimal is refused, never
/// rounded).
const MAX_FRAC_DIGITS: usize = 17;

// ==========================================================================
//  The JSON block
// ==========================================================================

/// The `dust` block (SPEC-LIT §157.4). `thermo` names one of §152.9's two
/// modes; the keys each mode reads are refused on the other. Required in a
/// Phase A case; a case stating a `cloud` states none of the Phase A keys,
/// `thermo` included (§159.9).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonDust {
    /// `formula` (the pseudo-species of (152.1)-(152.8) at phi) |
    /// `prescribed` (one constant c_p, §152.9). Required in a Phase A case
    /// (refused by name when absent there) and refused beside a `cloud`;
    /// optional in the schema only so a cloud can stand without it
    /// (SPEC-LIT §159.9).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermo: Option<String>,
    /// The equivalence ratio (152.2); REQUIRED in the formula mode, refused
    /// in the prescribed one. Held as the exact fraction of the decimal the
    /// case wrote.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phi: Option<f64>,
    /// J/(kg K), the one c_p both pseudo-species carry; prescribed only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cp: Option<f64>,
    /// kg/mol, the unburned molar mass (Leeds 0.0328); prescribed only.
    #[serde(rename = "Wu", default, skip_serializing_if = "Option::is_none")]
    pub w_u: Option<f64>,
    /// kg/mol, the burned molar mass (Leeds 0.0297); prescribed only.
    #[serde(rename = "Wb", default, skip_serializing_if = "Option::is_none")]
    pub w_b: Option<f64>,
    /// K, the measured burned temperature (Leeds 1500); prescribed only.
    #[serde(rename = "Tb", default, skip_serializing_if = "Option::is_none")]
    pub t_b: Option<f64>,
    /// m/s, the laminar burning velocity. REQUIRED in the prescribed mode;
    /// in the formula mode omitted is (152.13)'s S_L(phi).
    #[serde(rename = "SL", default, skip_serializing_if = "Option::is_none")]
    pub s_l: Option<f64>,
    /// The ρ-weighted SST's settings (§156.3, §156.4); refused without
    /// turbulence.model kOmegaSST.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sst: Option<JsonDustSst>,
    /// The reacting dust cloud (SPEC-LIT §159.9, DST-14); `None` for every
    /// Phase A case. Beside `combustion.model` FSC or TFC it is refused:
    /// Phase B burns the volatile on the finite-rate path, and Phase A's
    /// equivalent gas would count the same fuel twice.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cloud: Option<JsonDustCloud>,
}

/// `dust.sst` (SPEC-LIT §156.3, §156.4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonDustSst {
    /// (156.8)'s limiter on the ω production; the default is false, the
    /// tree's kinematic behaviour (§156.3).
    #[serde(rename = "limitOmegaProduction", default)]
    pub limit_omega_production: bool,
    /// SST-sust's ambient state (156.9); absent is off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sust: Option<JsonDustSust>,
}

/// `dust.sst.sust`, the (k_amb, omega_amb) of (156.9).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonDustSust {
    /// m²/s², the ambient turbulent kinetic energy.
    #[serde(rename = "kAmb")]
    pub k_amb: f64,
    /// 1/s, the ambient specific dissipation.
    #[serde(rename = "omegaAmb")]
    pub omega_amb: f64,
}

// ==========================================================================
//  The lowered config
// ==========================================================================

/// What [`JsonCase::lower`] hands [`lower_dust`] besides the two blocks:
/// the case's own words about its species, turbulence, physics and
/// boundaries, against which the dust model's refusals are read.
pub struct DustContext<'a> {
    /// Whether the case states a `species` block.
    pub species: bool,
    /// Whether the case states a `chemistry` block.
    pub chemistry: bool,
    /// `turbulence.kind`, `None` without a turbulence block.
    pub turbulence_kind: Option<TurbulenceKind>,
    /// `turbulence.model`, `None` without a turbulence block.
    pub turbulence_model: Option<&'a str>,
    /// Whether the case states `physics.compressible`.
    pub compressible: bool,
    /// `physics.gravity`.
    pub gravity: [f64; 3],
    /// `initial.T`, the case's unburned temperature T_u.
    pub initial_t: Option<f64>,
    /// Every boundary patch name actually present in the mesh.
    pub patch_names: &'a [String],
    /// The case's ordered `patches[]` rules.
    pub rules: &'a [JsonPatchRule],
}

/// `dust.thermo`, lowered (SPEC-LIT §152.9). phi is the exact fraction of
/// the decimal the case wrote (§152.11).
#[derive(Debug, Clone, PartialEq)]
pub enum DustThermo {
    Formula {
        phi_num: i64,
        phi_den: i64,
    },
    Prescribed {
        cp: f64,
        w_u: f64,
        w_b: f64,
        t_b: f64,
    },
}

/// `dust.sst`, lowered (SPEC-LIT §156.3, §156.4). `sust` is
/// (k_amb, omega_amb).
#[derive(Debug, Clone, PartialEq)]
pub struct LoweredDustSst {
    pub limit_omega_production: bool,
    pub sust: Option<(f64, f64)>,
}

/// The dust case, lowered - everything DST-06's driver and kernels read,
/// and nothing more (SPEC-LIT §157.4).
#[derive(Debug, Clone, PartialEq)]
pub struct LoweredDust {
    /// `FSC` or its TFC limit (§154.6).
    pub model: CombustionModel,
    /// A of (154.4); `A_FSC` = 0.4 unless the case names another - for TFC
    /// too, whose Zimont 0.52 is a named non-default input (§151.5).
    pub a: f64,
    /// Pr_t, required: each case its own (§151.5).
    pub pr_t: f64,
    /// The extended FSC's t_flacc (154.7); `None` for TFC, which has no
    /// clock (§154.6), and for an FSC case that named none.
    pub t_flacc: Option<f64>,
    /// The flame-Courant limit Co_f,max of (155.9); `CO_F_MAX` = 0.5 unless
    /// the case named another.
    pub max_flame_courant: f64,
    /// The ignition source of (155.4); `None` when the case named none.
    pub ignition: Option<IgnitionSource>,
    /// Gate DST-VES's prescribed `U_t` law (SPEC-LIT §157.6); `None` runs
    /// the §154 closures. TFC only, and only on a laminar case.
    pub prescribed_ut: Option<BurningLaw>,
    /// Which density the propagation reads (SPEC-LIT §157.6): the
    /// compressed (155.10) gas, or the initial one - INSIGHTS (g)'s
    /// negative control, `unburnedDensity "initial"`.
    pub unburned_density: UnburnedDensity,
    /// The equivalent-gas thermo mode (§152.9).
    pub thermo: DustThermo,
    /// The unburned temperature T_u, the case's `initial.T`: the fresh
    /// charge is the initial state.
    pub t_u: f64,
    /// The laminar burning velocity, m/s: the stated `SL`, else (152.13)'s
    /// S_L(phi) in the formula mode.
    pub s_l: f64,
    /// The ρ-weighted SST's settings; `None` without a `dust.sst` block.
    pub sst: Option<LoweredDustSst>,
    /// The progress variable's initial field: internal 0, one condition per
    /// patch, built on the same patch names and rules as every other
    /// field's.
    pub c_field: LoweredScalarField,
}

impl LoweredDust {
    /// The equivalent gas at the lowered thermo: formula builds the two
    /// pseudo-species of (152.1)-(152.8) at phi, prescribed states one c_p
    /// and both molar masses with the measured T_b (§152.9). Called once by
    /// [`lower_dust`], so a bad charge is refused at lowering keyed `dust`.
    pub fn gas(&self) -> Result<EquivalentGas> {
        match &self.thermo {
            DustThermo::Formula { phi_num, phi_den } => {
                let inp = FormulaInputs {
                    fuel: StarchFuel::huang2022_table1()?,
                    n2_per_o2_centi: N2_PER_O2_CENTI,
                    phi_num: *phi_num,
                    phi_den: *phi_den,
                };
                EquivalentGas::formula(&inp, &tm4513_gas_set()?)
            }
            DustThermo::Prescribed { cp, w_u, w_b, t_b } => EquivalentGas::prescribed(
                &PrescribedInputs { cp: *cp, w_u: *w_u, w_b: *w_b, t_u: self.t_u, t_b: *t_b },
            ),
        }
        .map_err(|e| keyed("dust", e))
    }
}

// ==========================================================================
//  Refusals
// ==========================================================================

/// The one house refusal: a key, a sentence, the section.
fn refused(key: &str, msg: &str) -> Error {
    Error::Refused(format!("{key}: {msg} {TAG}"))
}

/// The cloud's own refusal, named to its section (SPEC-LIT §159.9).
fn cloud_refused(key: &str, msg: &str) -> Error {
    Error::Refused(format!("{key}: {msg} {CLOUD_TAG}"))
}

/// The library error a case's charge or fuel raised, carried under the key
/// that asked for it.
fn keyed(key: &str, e: Error) -> Error {
    match e {
        Error::Refused(m) => Error::Refused(format!("{key}: {m}")),
        Error::Config(m) => Error::Config(format!("{key}: {m}")),
        other => other,
    }
}

/// A finite number above zero, or the refusal naming `key`.
fn positive(key: &str, v: f64) -> Result<f64> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(refused(key, &format!("{v} is not a finite number above zero")))
    }
}

/// A driver's refusal of a case that states the dust model when it does not
/// run it (SPEC-LIT §157.4): `ofgpu-lowmach` is the one driver DST-06 wires,
/// and part 3 of that unit is the wiring. `source` names where.
#[must_use]
pub fn refuse_dust_case(source: &str) -> Error {
    Error::Refused(format!(
        "{source}: this case states the dust model (combustion.model FSC or TFC with a dust \
         block), which this driver does not run; ofgpu-lowmach runs it (DST-06, SPEC-LIT §157.4), \
         and ofgpu-compressible with a gas block (DST-10, SPEC-LIT §158.6)"
    ))
}

/// Whether `combustion.model` names the dust model's closure, `FSC` or
/// `TFC` (SPEC-LIT §154).
pub fn is_dust_model(c: Option<&JsonCombustion>) -> bool {
    matches!(c.and_then(|c| c.model.as_deref()), Some("FSC") | Some("TFC"))
}

/// phi as the exact fraction of the decimal the case wrote (SPEC-LIT
/// §152.11): `0.62` -> `(62, 100)`, `0.5` -> `(5, 10)`, `1.0` -> `(1, 1)`,
/// `0.3` -> `(3, 10)`. `{}` Display is f64's shortest round-trip decimal
/// and never an exponent, so its digits ARE the written decimal. Refused:
/// not finite, negative, or a denominator above 10^17 - refused, never
/// rounded.
pub fn decimal_fraction(v: f64) -> Result<(i64, i64)> {
    if !v.is_finite() || v < 0.0 {
        return Err(refused(
            "dust.phi",
            &format!("{v} is not a finite equivalence ratio at or above zero"),
        ));
    }
    let text = format!("{v}");
    let (int, frac) = match text.split_once('.') {
        Some((i, f)) => (i, f),
        None => (text.as_str(), ""),
    };
    if frac.len() > MAX_FRAC_DIGITS {
        return Err(refused(
            "dust.phi",
            &format!(
                "{v} has {} fractional digits; phi is held as the exact fraction of the \
                 decimal, and a denominator above 1e{MAX_FRAC_DIGITS} is refused, not rounded",
                frac.len()
            ),
        ));
    }
    let num: i64 = format!("{int}{frac}")
        .parse()
        .map_err(|_| refused("dust.phi", &format!("{v} does not fit the exact fraction phi is held as")))?;
    Ok((num, 10i64.pow(frac.len() as u32)))
}

// ==========================================================================
//  The one lowering
// ==========================================================================

/// The one lowering of the dust case (SPEC-LIT §157.4). `Ok(None)` when the
/// case states neither a `dust` block nor `combustion.model` FSC/TFC - then
/// nothing else about the case changes. `Some` means the case IS a dust
/// one: the caller skips the chemistry blocks entirely, the case's gas is
/// the two pseudo-species (§152.4), and every driver but `ofgpu-lowmach`
/// refuses it by name.
pub fn lower_dust(
    combustion: Option<&JsonCombustion>,
    dust: Option<&JsonDust>,
    cx: &DustContext<'_>,
) -> Result<Option<LoweredDust>> {
    // The entry refusals: a dust block without the model, and the model
    // without a dust block, each by its own key. A `cloud` changes both
    // arms (SPEC-LIT §159.9): beside FSC/TFC it is refused - Phase B burns
    // the volatile on the finite-rate path (§159.4), and Phase A's
    // equivalent gas would count the same fuel twice - and without the
    // model the block lowers to nothing here, `lower_cloud` owning it.
    let dust = match (dust, is_dust_model(combustion)) {
        (None, false) => return Ok(None),
        (Some(d), false) if d.cloud.is_some() => return Ok(None),
        (Some(_), false) => {
            return Err(refused(
                "dust",
                "stated without combustion.model FSC or TFC; the block belongs to the dust \
                 model's closures, and laminar/EDM/EDC/PaSR read none of it",
            ))
        }
        (None, true) => {
            return Err(refused(
                "combustion.model",
                "FSC or TFC without a dust block; the closures' equivalent gas, ignition and \
                 turbulence settings are the dust block's to state",
            ))
        }
        (Some(d), true) if d.cloud.is_some() => {
            return Err(cloud_refused(
                "dust.cloud",
                "stated beside combustion.model FSC or TFC; Phase B burns the volatile on the \
                 finite-rate path (SPEC-LIT §159.4), and Phase A's equivalent gas would count \
                 the same fuel twice",
            ))
        }
        (Some(d), true) => d,
    };
    let c = combustion.expect("is_dust_model is Some, so the combustion block is stated");
    let model = match c.model.as_deref().expect("is_dust_model read it") {
        "FSC" => CombustionModel::Fsc,
        _ => CombustionModel::Tfc,
    };
    // The dust model's gas is its own two pseudo-species: the case's species
    // set and mechanism would be read and ignored beside it.
    for (stated, key) in [(cx.species, "species"), (cx.chemistry, "chemistry")] {
        if stated {
            return Err(refused(
                key,
                "stated with the dust model; the dust model's gas is its own two \
                 pseudo-species (SPEC-LIT §152.4), not the case's species",
            ));
        }
    }
    // A key of EDM/EDC/PaSR beside an FSC/TFC model would be read and
    // ignored: §13.4.1's rule, the dust side's own table.
    for (key, set) in [
        ("B", c.b.is_some()),
        ("variant", c.variant.is_some()),
        ("CD1", c.cd1.is_some()),
        ("CD2", c.cd2.is_some()),
        ("gammaMax", c.gamma_max.is_some()),
        ("psrSpan", c.psr_span.is_some()),
        ("tauMix", c.tau_mix.is_some()),
        ("tauChem", c.tau_chem.is_some()),
        ("Cmix", c.c_mix.is_some()),
    ] {
        if set {
            return Err(refused(
                &format!("combustion.{key}"),
                &format!("a setting of EDM/EDC/PaSR, and combustion.model is {}", c.model.as_deref().unwrap_or("FSC")),
            ));
        }
    }
    // Pr_t is REQUIRED: each case its own (§151.5) is a stated input, not a
    // default one. t_flacc is the extended FSC's clock; TFC has none (§154.6).
    let Some(pr_t) = c.pr_t else {
        return Err(refused(
            "combustion.PrT",
            "absent; Pr_t is each case its own (Leeds 0.4, REMBE 0.28, SPEC-LIT §151.5), \
             so there is no default",
        ));
    };
    if model == CombustionModel::Tfc && c.t_flacc.is_some() {
        return Err(refused(
            "combustion.tFlacc",
            "stated with combustion.model TFC; TFC has no clock (SPEC-LIT §154.6) - the \
             extended FSC's acceleration is the FSC model's option alone",
        ));
    }
    let a = c.a.map(|v| positive("combustion.A", v)).transpose()?.unwrap_or(A_FSC);
    let pr_t = positive("combustion.PrT", pr_t)?;
    let t_flacc = c.t_flacc.map(|v| positive("combustion.tFlacc", v)).transpose()?;
    let max_flame_courant = c
        .max_flame_courant
        .map(|v| positive("combustion.maxFlameCourant", v))
        .transpose()?
        .unwrap_or(CO_F_MAX);
    let ignition = match &c.ignition {
        None => None,
        Some(g) => Some(ignition_of(g)?),
    };
    // ---- thermo: the two modes of §152.9, each refused on the other's keys.
    // Optional in the schema only so a `cloud` can stand without it (§159.9);
    // a Phase A case states it, and its absence is refused by name.
    let thermo_name = dust.thermo.as_deref();
    let thermo = match thermo_name {
        None => {
            return Err(refused(
                "dust.thermo",
                "absent; the mode names one of the equivalent-gas thermos of SPEC-LIT §152.9, \
                 and neither has a default",
            ))
        }
        Some("formula") => {
            for (key, set) in [
                ("cp", dust.cp.is_some()),
                ("Wu", dust.w_u.is_some()),
                ("Wb", dust.w_b.is_some()),
                ("Tb", dust.t_b.is_some()),
            ] {
                if set {
                    return Err(refused(
                        &format!("dust.{key}"),
                        "stated in the formula mode; the prescribed thermo's constant-c_p \
                         inputs, and the formula mode builds the gas from (152.1)-(152.8)",
                    ));
                }
            }
            let Some(phi) = dust.phi else {
                return Err(refused(
                    "dust.phi",
                    "absent; the formula mode builds the equivalent gas at the case's phi",
                ));
            };
            let (phi_num, phi_den) = decimal_fraction(phi)?;
            DustThermo::Formula { phi_num, phi_den }
        }
        Some("prescribed") => {
            if dust.phi.is_some() {
                return Err(refused(
                    "dust.phi",
                    "stated in the prescribed mode; phi is the formula mode's charge, and the \
                     prescribed pseudo-species are the case's stated constants",
                ));
            }
            let cp = required(dust.cp, "dust.cp", "the prescribed mode states one c_p for both pseudo-species")?;
            let w_u = required(dust.w_u, "dust.Wu", "the prescribed mode states the unburned molar mass")?;
            let w_b = required(dust.w_b, "dust.Wb", "the prescribed mode states the burned molar mass")?;
            let t_b = required(dust.t_b, "dust.Tb", "the prescribed mode states the measured burned temperature")?;
            DustThermo::Prescribed {
                cp: positive("dust.cp", cp)?,
                w_u: positive("dust.Wu", w_u)?,
                w_b: positive("dust.Wb", w_b)?,
                t_b: positive("dust.Tb", t_b)?,
            }
        }
        Some(other) => {
            return Err(refused(
                "dust.thermo",
                &format!("\"{other}\": the recognised modes are \"formula, prescribed\" (SPEC-LIT §152.9)"),
            ))
        }
    };
    // ---- S_L: the stated one, else (152.13) at phi in the formula mode ----
    let s_l = match dust.s_l {
        Some(v) => positive("dust.SL", v)?,
        None => match &thermo {
            DustThermo::Formula { phi_num, phi_den } => {
                LaminarSpeed::huang2022().s_l_phi(*phi_num as f64 / *phi_den as f64)?
            }
            DustThermo::Prescribed { .. } => {
                return Err(refused(
                    "dust.SL",
                    "absent; the prescribed mode has no phi for (152.13), so the case states S_L",
                ))
            }
        },
    };
    // ---- turbulence: laminar, or the rho-weighted SST of §156 -------------
    let sst_model = cx.turbulence_model == Some("kOmegaSST");
    match (cx.turbulence_kind, cx.turbulence_model) {
        (None, None) | (_, Some("laminar")) | (Some(TurbulenceKind::Ras), Some("kOmegaSST")) => {}
        _ => {
            return Err(refused(
                "turbulence.model",
                "the dust model's turbulence is laminar or the rho-weighted kOmegaSST of \
                 SPEC-LIT §156",
            ))
        }
    }
    let sst = match &dust.sst {
        None => None,
        Some(s) => {
            if !sst_model {
                return Err(refused(
                    "dust.sst",
                    "stated without turbulence.model kOmegaSST; the settings belong to the \
                     rho-weighted SST of SPEC-LIT §156",
                ));
            }
            Some(LoweredDustSst {
                limit_omega_production: s.limit_omega_production,
                sust: s.sust.as_ref().map(|s| (s.k_amb, s.omega_amb)),
            })
        }
    };
    if sst_model && cx.gravity != [0.0, 0.0, 0.0] {
        return Err(refused(
            "physics.gravity",
            "not zero with the rho-weighted kOmegaSST; the form carries no buoyancy \
             production (SPEC-LIT §156.7)",
        ));
    }
    // ---- Gate DST-VES's prescribed U_t and its negative control (§157.6) --
    let prescribed_ut = match &c.prescribed_ut {
        None => None,
        Some(u) => {
            if model == CombustionModel::Fsc {
                return Err(refused(
                    "combustion.prescribedUt",
                    "stated with combustion.model FSC; the prescribed U_t is the TFC \
                     operator of Gate DST-VES alone (SPEC-LIT §157.6) - FSC forms U_t \
                     from its own closures",
                ));
            }
            if sst_model {
                return Err(refused(
                    "combustion.prescribedUt",
                    "stated with turbulence.model kOmegaSST; the case's law holds U_t \
                     itself, so the turbulent diffusivity has to be negligible - run \
                     it laminar",
                ));
            }
            let law = BurningLaw { s0: u.s0, alpha: u.alpha, beta: u.beta };
            law.validate()
                .map_err(|e| keyed("combustion.prescribedUt", e))?;
            Some(law)
        }
    };
    let unburned_density = match c.unburned_density.as_deref() {
        None => UnburnedDensity::Compressed,
        Some("compressed") => UnburnedDensity::Compressed,
        Some("initial") => {
            if c.prescribed_ut.is_none() {
                return Err(refused(
                    "combustion.unburnedDensity",
                    "\"initial\" without combustion.prescribedUt; holding rho_u at its \
                     initial value is the negative control of Gate DST-VES (INSIGHTS (g), \
                     SPEC-LIT §157.6), and the control exists to MISS (dp/dt)_max only \
                     against the prescribed law's compressed run",
                ));
            }
            UnburnedDensity::Initial
        }
        Some(other) => {
            return Err(refused(
                "combustion.unburnedDensity",
                &format!(
                    "\"{other}\": the recognised modes are \"compressed, initial\" \
                     (SPEC-LIT §157.6)"
                ),
            ))
        }
    };
    // SPEC-LIT §158.6: a dust case MAY state `physics.compressible` - the
    // mixture gas law's own air (§122.6) - and `ofgpu-compressible` is the
    // driver that runs the pair; the lowered dust block is the same either
    // way. Every other driver refuses the case whole (`refuse_dust_case`),
    // and `ofgpu-lowmach` refuses the gas beside the model by name.
    let Some(t_u) = cx.initial_t else {
        return Err(refused(
            "initial.T",
            "absent; T_u is the case's initial T - the fresh charge is the initial state",
        ));
    };
    // ---- the c field: internal 0, one condition per patch -----------------
    let mut boundary = BTreeMap::new();
    for pname in cx.patch_names {
        let kind = resolve_patch_rule(cx.rules, pname)?.kind;
        let s = match kind {
            PatchPresetKind::Wall => spec("zeroGradient"),
            PatchPresetKind::Inlet => {
                let mut s = spec("fixedValue");
                s.value = vec![0.0 as Scalar];
                s
            }
            PatchPresetKind::Open => {
                let mut s = spec("inletOutlet");
                s.inlet_value = vec![0.0 as Scalar];
                s.value = vec![0.0 as Scalar];
                s
            }
            PatchPresetKind::Empty | PatchPresetKind::Symmetry | PatchPresetKind::Wedge => {
                spec(patch_class(kind))
            }
            PatchPresetKind::Farfield | PatchPresetKind::SupersonicInlet => {
                return Err(refused(
                    "dust",
                    &format!(
                        "patch \"{pname}\" is a compressible preset; the dust model runs on \
                         fixedValue/zeroGradient/inletOutlet patches only, and the §126 rules \
                         are not carried by the mixture gas law (SPEC-LIT §158.6)"
                    ),
                ))
            }
        };
        boundary.insert(pname.clone(), s);
    }
    let lowered = LoweredDust {
        model,
        a,
        pr_t,
        t_flacc,
        max_flame_courant,
        ignition,
        prescribed_ut,
        unburned_density,
        thermo,
        t_u,
        s_l,
        sst,
        c_field: LoweredScalarField {
            name: C_FIELD.to_string(),
            dimensions: C_DIMS.to_string(),
            internal_uniform: 0.0 as Scalar,
            boundary,
        },
    };
    // The charge is built once HERE, so a phi the composition refuses, or a
    // prescribed set that does not close, is refused at lowering keyed
    // `dust` - before any driver or gate reads the case again.
    lowered.gas()?;
    Ok(Some(lowered))
}

/// `combustion.ignition`, lowered onto the library's source: each value
/// checked by key first, then the library's own `validate` (its message is
/// kept), and sigmaT omitted is t0 (§151.7 item 8).
fn ignition_of(g: &crate::io::case_chem::JsonIgnition) -> Result<IgnitionSource> {
    for (i, v) in g.point.iter().enumerate() {
        if !v.is_finite() {
            return Err(refused(
                "ignition.point",
                &format!("component {i} ({v}) is not a finite coordinate"),
            ));
        }
    }
    let sigma_r = positive("ignition.sigmaR", g.sigma_r)?;
    let w0 = positive("ignition.W0", g.w0)?;
    if !(g.t0.is_finite() && g.t0 >= 0.0) {
        return Err(refused(
            "ignition.t0",
            &format!("{} is not a finite time at or above zero", g.t0),
        ));
    }
    let sigma_t = match g.sigma_t {
        Some(v) => positive("ignition.sigmaT", v)?,
        None => g.t0,
    };
    let src = IgnitionSource {
        point: DVec3::new(g.point[0], g.point[1], g.point[2]),
        sigma_r,
        w0,
        t0: g.t0,
        sigma_t,
    };
    src.validate()?;
    Ok(src)
}

/// A stated value or the refusal naming `key` with why the mode needs it.
fn required(v: Option<f64>, key: &str, why: &str) -> Result<f64> {
    v.ok_or_else(|| refused(key, &format!("absent; {why}")))
}

/// One boundary-condition placeholder of the named type, values to follow.
fn spec(type_name: &str) -> PatchFieldSpec {
    PatchFieldSpec { type_name: type_name.to_string(), ..PatchFieldSpec::default() }
}

// ==========================================================================
//  The cloud (SPEC-LIT §159.9, DST-14)
// ==========================================================================

/// `dust.cloud` (SPEC-LIT §159.9): the uniform reacting dust cloud DST-14
/// states and lowers, DST-15 runs. The size is exactly one of `diameter`
/// and `rosinRammler`. The GAS side of [`ParticleInputs`] is not the
/// cloud's - [`LoweredCloud::particle_inputs`] takes it from the caller,
/// which in DST-15 is the gas.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonDustCloud {
    /// kg/m3, the cloud's mass per gas volume; finite and above zero.
    pub concentration: f64,
    /// m, the one diameter of a monodisperse cloud; exactly one of this
    /// and `rosinRammler` states the size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diameter: Option<f64>,
    /// The Rosin-Rammler law of (147.1): `d` is d_bar, its size parameter
    /// X; `n` is the spread q.
    #[serde(rename = "rosinRammler", default, skip_serializing_if = "Option::is_none")]
    pub rosin_rammler: Option<JsonRosinRammler>,
    /// Parcels per cell, the K of (159.12); absent is 1.
    #[serde(rename = "parcelsPerCell", default, skip_serializing_if = "Option::is_none")]
    pub parcels_per_cell: Option<u32>,
    /// The four stage fractions; each in [0, 1], the sum within 1e-9 of 1.
    pub composition: JsonComposition,
    /// kg/m3, the particle density.
    #[serde(rename = "rhoP")]
    pub rho_p: f64,
    /// J/(kg K), the one c_p of (159.1) for the whole particle.
    #[serde(rename = "cpP")]
    pub cp_p: f64,
    /// epsilon_0 of (159.1), in [0, 1].
    pub emissivity: f64,
    /// SFOM (159.4)'s rate; required with volatile > 0, refused with
    /// volatile == 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub devolatilisation: Option<JsonDevolatilisation>,
    /// The char reaction (159.5); only with char > 0, absent is inert.
    #[serde(rename = "char", default, skip_serializing_if = "Option::is_none")]
    pub char_burn: Option<JsonCloudChar>,
}

/// `dust.cloud.rosinRammler` - (147.1)'s two constants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonRosinRammler {
    /// m, the size parameter X of (147.1).
    pub d: f64,
    /// The spread exponent q of (147.1).
    pub n: f64,
}

/// `dust.cloud.composition` - the four stage fractions of one particle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonComposition {
    /// The moisture's share.
    pub moisture: f64,
    /// The volatile's share.
    pub volatile: f64,
    /// The char's share (`char` is a legal field name).
    pub char: f64,
    /// The ash's share.
    pub ash: f64,
}

/// `dust.cloud.devolatilisation` - SFOM (159.4)'s rate and its heat.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonDevolatilisation {
    /// 1/s.
    #[serde(rename = "A")]
    pub a: f64,
    /// J/mol.
    #[serde(rename = "Ea")]
    pub e_a: f64,
    /// J/kg; positive is endothermic.
    #[serde(rename = "dH")]
    pub dh: f64,
}

/// `dust.cloud.char` - the char reaction (159.5)'s inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonCloudChar {
    /// kg/(m s Pa K^0.75), C_0 of R_diff.
    #[serde(rename = "C0")]
    pub c0: f64,
    /// kg/(m2 s Pa), A of R_kin.
    #[serde(rename = "A")]
    pub a: f64,
    /// J/mol, E_a of R_kin.
    #[serde(rename = "Ea")]
    pub e_a: f64,
    /// The retention factor h_s, in [0, 1] (ISL22: 0.3).
    pub hs: f64,
    /// J/kg, the heat per kg of carbon burnt; absent is ISL22's 32.9 MJ/kg
    /// (§159.1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hc: Option<f64>,
}

/// The cloud's stated size: one diameter, or the Rosin-Rammler law.
#[derive(Debug, Clone, PartialEq)]
pub enum CloudSize {
    /// The monodisperse cloud: every parcel the same diameter.
    Monodisperse { d: f64 },
    /// (147.1)'s law: the size parameter X and its spread q.
    RosinRammler { d_bar: f64, n: f64 },
}

/// The four stage fractions, normalised by their own sum, so the four stage
/// masses of a particle add to its mass to round-off (§159.9).
#[derive(Debug, Clone, PartialEq)]
pub struct CloudComposition {
    pub moisture: f64,
    pub volatile: f64,
    pub char: f64,
    pub ash: f64,
}

/// SFOM (159.4)'s rate, lowered.
#[derive(Debug, Clone, PartialEq)]
pub struct CloudDevol {
    /// 1/s.
    pub a: f64,
    /// J/mol.
    pub e_a: f64,
    /// J/kg.
    pub dh: f64,
}

/// `hc` absent is ISL22's 32.9 MJ/kg (SPEC-LIT §159.1).
pub const CHAR_HC_DEFAULT: f64 = 32.9e6;

/// The cloud, lowered - everything DST-15's seeding and the reacting
/// particle kernel read, and nothing more (SPEC-LIT §159.9). f64
/// throughout, both builds.
#[derive(Debug, Clone, PartialEq)]
pub struct LoweredCloud {
    /// kg/m3, the stated C.
    pub concentration: f64,
    /// The stated size.
    pub size: CloudSize,
    /// The stated K, absent 1.
    pub parcels_per_cell: u32,
    /// The stated fractions over their own sum.
    pub composition: CloudComposition,
    /// kg/m3.
    pub rho_p: f64,
    /// J/(kg K).
    pub cp_p: f64,
    /// In [0, 1].
    pub emissivity: f64,
    /// SFOM's rate; `None` with volatile == 0.
    pub devol: Option<CloudDevol>,
    /// The char reaction; [`CharInputs::inert`] when `char` was absent.
    pub char_rate: CharInputs,
    /// The case's `initial.T`: the particles start at rest at it.
    pub t_init: f64,
    /// K entries of (159.12) - band j's mass-weighted mean diameter - or
    /// the one diameter repeated K times for a monodisperse cloud.
    pub diameters: Vec<f64>,
}

/// The gas-side inputs [`ParticleInputs`] needs and the cloud does not own
/// (§159.9): the caller states them, which in DST-15 is the gas.
#[derive(Debug, Clone, PartialEq)]
pub struct CloudGas {
    /// Pa s.
    pub mu: f64,
    /// W/(m K).
    pub k_g: f64,
    /// J/(kg K).
    pub cp_g: f64,
    /// J/(kg K), the volatile's.
    pub cp_volatile: f64,
    /// Pa, the drying pressure.
    pub p_ambient: f64,
}

/// One seeded parcel of the cloud, in `LoweredCloud::seed`'s cell-major
/// band-minor order (SPEC-LIT §159.9).
#[derive(Debug, Clone, PartialEq)]
pub struct CloudParcel {
    /// The cell the parcel sits in.
    pub cell: usize,
    /// The cell's centroid.
    pub position: DVec3,
    /// n_p of (159.14): the physical particles this parcel stands for.
    pub n_p: f64,
    /// The particle's four stage masses, its temperature and its diameter.
    pub particle: ParticleState,
}

/// A finite number above zero, or the cloud's refusal naming `key`.
fn cloud_positive(key: &str, v: f64) -> Result<f64> {
    if v.is_finite() && v > 0.0 {
        Ok(v)
    } else {
        Err(cloud_refused(key, &format!("{v} is not a finite number above zero")))
    }
}

/// A finite number at or above zero, or the cloud's refusal naming `key`.
fn cloud_at_least_zero(key: &str, v: f64) -> Result<f64> {
    if v.is_finite() && v >= 0.0 {
        Ok(v)
    } else {
        Err(cloud_refused(key, &format!("{v} is not a finite number at or above zero")))
    }
}

/// A finite number, or the cloud's refusal naming `key`.
fn cloud_finite(key: &str, v: f64) -> Result<f64> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(cloud_refused(key, &format!("{v} is not finite")))
    }
}

/// A finite fraction in [0, 1], or the cloud's refusal naming `key`.
fn cloud_unit_interval(key: &str, v: f64) -> Result<f64> {
    if v.is_finite() && v >= 0.0 && v <= 1.0 {
        Ok(v)
    } else {
        Err(cloud_refused(key, &format!("{v} is not a finite fraction in [0, 1]")))
    }
}

impl JsonDustCloud {
    /// The one lowering of the cloud block (SPEC-LIT §159.9): every value
    /// refusal names its key, the composition is normalised by its own sum,
    /// and the K band diameters of (159.12) are built here.
    pub fn lower(&self, t_init: f64) -> Result<LoweredCloud> {
        let concentration = cloud_positive("dust.cloud.concentration", self.concentration)?;
        let size = match (&self.diameter, &self.rosin_rammler) {
            (Some(d), None) => {
                CloudSize::Monodisperse { d: cloud_positive("dust.cloud.diameter", *d)? }
            }
            (None, Some(rr)) => CloudSize::RosinRammler {
                d_bar: cloud_positive("dust.cloud.rosinRammler.d", rr.d)?,
                n: cloud_positive("dust.cloud.rosinRammler.n", rr.n)?,
            },
            (Some(_), Some(_)) => {
                return Err(cloud_refused(
                    "dust.cloud.diameter",
                    "stated beside rosinRammler; exactly one of the two states the size",
                ))
            }
            (None, None) => {
                return Err(cloud_refused(
                    "dust.cloud.diameter",
                    "absent, and so is rosinRammler; exactly one of the two states the size",
                ))
            }
        };
        let parcels_per_cell = match self.parcels_per_cell {
            None => 1,
            Some(0) => {
                return Err(cloud_refused(
                    "dust.cloud.parcelsPerCell",
                    "0; a cell seeded with no parcels carries none of the cloud's mass",
                ))
            }
            Some(k) => k,
        };
        let c = &self.composition;
        for (name, v) in [("moisture", c.moisture), ("volatile", c.volatile), ("char", c.char), ("ash", c.ash)] {
            if !(v.is_finite() && v >= 0.0 && v <= 1.0) {
                return Err(cloud_refused(
                    "dust.cloud.composition",
                    &format!("{name} = {v} is not a finite fraction in [0, 1]"),
                ));
            }
        }
        let sum = c.moisture + c.volatile + c.char + c.ash;
        if !((sum - 1.0).abs() <= 1.0e-9) {
            return Err(cloud_refused(
                "dust.cloud.composition",
                &format!("the four fractions sum to {sum}, not within 1e-9 of 1"),
            ));
        }
        let composition = CloudComposition {
            moisture: c.moisture / sum,
            volatile: c.volatile / sum,
            char: c.char / sum,
            ash: c.ash / sum,
        };
        let rho_p = cloud_positive("dust.cloud.rhoP", self.rho_p)?;
        let cp_p = cloud_positive("dust.cloud.cpP", self.cp_p)?;
        cloud_unit_interval("dust.cloud.emissivity", self.emissivity)?;
        let devol = match &self.devolatilisation {
            None => {
                if c.volatile > 0.0 {
                    return Err(cloud_refused(
                        "dust.cloud.devolatilisation",
                        "absent with a volatile fraction above zero; SFOM (159.4) is the rate \
                         the volatile leaves the particle at",
                    ));
                }
                None
            }
            Some(dv) => {
                if c.volatile == 0.0 {
                    return Err(cloud_refused(
                        "dust.cloud.devolatilisation",
                        "stated with a volatile fraction of 0; there is no volatile to \
                         devolatilise",
                    ));
                }
                Some(CloudDevol {
                    a: cloud_at_least_zero("dust.cloud.devolatilisation.A", dv.a)?,
                    e_a: cloud_at_least_zero("dust.cloud.devolatilisation.Ea", dv.e_a)?,
                    dh: cloud_finite("dust.cloud.devolatilisation.dH", dv.dh)?,
                })
            }
        };
        let char_rate = match &self.char_burn {
            None => CharInputs::inert(),
            Some(ch) => {
                if c.char == 0.0 {
                    return Err(cloud_refused(
                        "dust.cloud.char",
                        "stated with a char fraction of 0; there is no char to burn",
                    ));
                }
                CharInputs {
                    c0: cloud_at_least_zero("dust.cloud.char.C0", ch.c0)?,
                    a: cloud_at_least_zero("dust.cloud.char.A", ch.a)?,
                    e_a: cloud_at_least_zero("dust.cloud.char.Ea", ch.e_a)?,
                    h_s: cloud_unit_interval("dust.cloud.char.hs", ch.hs)?,
                    h_c: match ch.hc {
                        None => CHAR_HC_DEFAULT,
                        Some(hc) => cloud_at_least_zero("dust.cloud.char.hc", hc)?,
                    },
                }
            }
        };
        let diameters = match &size {
            CloudSize::Monodisperse { d } => vec![*d; parcels_per_cell as usize],
            CloudSize::RosinRammler { d_bar, n } => {
                rosin_rammler_bands(*d_bar, *n, parcels_per_cell)?
            }
        };
        Ok(LoweredCloud {
            concentration,
            size,
            parcels_per_cell,
            composition,
            rho_p,
            cp_p,
            emissivity: self.emissivity,
            devol,
            char_rate,
            t_init,
            diameters,
        })
    }
}

/// The cloud's one lowering beside a whole case (SPEC-LIT §159.9), called
/// by `JsonCase::lower` right after `lower_dust` with the same
/// `DustContext`. `Ok(None)` without `dust.cloud` - then nothing about the
/// case changes. With one, each Phase A key stated beside it is refused
/// (the keys belong to the equivalent-gas lowering, and a cloud case
/// states none of them), `initial.T` is required - the particles start at
/// rest at it - and the block lowers.
pub fn lower_cloud(
    combustion: Option<&JsonCombustion>,
    dust: Option<&JsonDust>,
    cx: &DustContext<'_>,
) -> Result<Option<LoweredCloud>> {
    let Some(d) = dust else { return Ok(None) };
    let Some(cloud) = &d.cloud else { return Ok(None) };
    if is_dust_model(combustion) {
        // `lower_dust` refuses this first on the JSONC route; carried here
        // too, so a direct call cannot lower a cloud beside the model.
        return Err(cloud_refused(
            "dust.cloud",
            "stated beside combustion.model FSC or TFC; Phase B burns the volatile on the \
             finite-rate path (SPEC-LIT §159.4), and Phase A's equivalent gas would count \
             the same fuel twice",
        ));
    }
    for (key, set) in [
        ("dust.thermo", d.thermo.is_some()),
        ("dust.phi", d.phi.is_some()),
        ("dust.cp", d.cp.is_some()),
        ("dust.Wu", d.w_u.is_some()),
        ("dust.Wb", d.w_b.is_some()),
        ("dust.Tb", d.t_b.is_some()),
        ("dust.SL", d.s_l.is_some()),
        ("dust.sst", d.sst.is_some()),
    ] {
        if set {
            return Err(cloud_refused(
                key,
                "stated beside dust.cloud; the key belongs to Phase A's equivalent-gas \
                 lowering, and a cloud case states none of it",
            ));
        }
    }
    let Some(t_init) = cx.initial_t else {
        return Err(cloud_refused(
            "initial.T",
            "absent; the cloud's particles start at rest at the case's initial T",
        ));
    };
    cloud.lower(t_init).map(Some)
}

impl LoweredCloud {
    /// [`ParticleInputs`] for the reacting particle kernel of §159: the
    /// cloud states the PARTICLE side (c_p, emissivity, the SFOM rate, the
    /// char reaction) and `gas` states the gas side (mu, k_g, c_p,g, the
    /// volatile's c_p and the drying pressure), which in DST-15 is the gas
    /// (§159.9). `devol` absent is the all-zero rate.
    pub fn particle_inputs(&self, gas: &CloudGas) -> Result<ParticleInputs> {
        let zero = CloudDevol { a: 0.0, e_a: 0.0, dh: 0.0 };
        let dv = self.devol.as_ref().unwrap_or(&zero);
        Ok(ParticleInputs {
            cp: self.cp_p,
            emissivity: self.emissivity,
            devol_a: dv.a,
            devol_e_a: dv.e_a,
            devol_dh: dv.dh,
            cp_volatile: gas.cp_volatile,
            drying: Drying::water(gas.p_ambient)?,
            mu: gas.mu,
            k_g: gas.k_g,
            cp_g: gas.cp_g,
            char_rate: self.char_rate,
        })
    }

    /// The seeding (SPEC-LIT §159.9): each cell c of volume V_c receives
    /// `parcels_per_cell` parcels at its centroid, one per band, in
    /// cell-major band-minor order. Band j's parcel carries the band's
    /// mass-weighted mean diameter d_j of (159.12), the mass
    /// `m = rho_p pi d_j^3 / 6`, the count `n_p = (C V_c / K) / m` of
    /// (159.14), the composition's four stage shares of m, and the case's
    /// `initial.T`; so the seeded mass per volume is C.
    pub fn seed(&self, centroids: &[DVec3], volumes: &[f64]) -> Result<Vec<CloudParcel>> {
        if centroids.len() != volumes.len() {
            return Err(cloud_refused(
                "dust.cloud",
                &format!(
                    "the seeding was handed {} centroids and {} volumes; one volume per cell",
                    centroids.len(),
                    volumes.len()
                ),
            ));
        }
        if centroids.is_empty() {
            return Err(cloud_refused(
                "dust.cloud",
                "the mesh has no cells; there is no cloud to seed",
            ));
        }
        for (c, v) in volumes.iter().enumerate() {
            if !(v.is_finite() && *v > 0.0) {
                return Err(cloud_refused(
                    "dust.cloud",
                    &format!("cell {c} has volume {v}, not a finite volume above zero"),
                ));
            }
        }
        let k = self.diameters.len();
        let comp = &self.composition;
        let mut out = Vec::with_capacity(centroids.len() * k);
        for (c, (x, v)) in centroids.iter().zip(volumes.iter()).enumerate() {
            for d in &self.diameters {
                let m = self.rho_p * PI * d * d * d / 6.0;
                let n_p = (self.concentration * v / k as f64) / m;
                out.push(CloudParcel {
                    cell: c,
                    position: *x,
                    n_p,
                    particle: ParticleState {
                        m_w: comp.moisture * m,
                        m_v: comp.volatile * m,
                        m_c: comp.char * m,
                        m_a: comp.ash * m,
                        t: self.t_init,
                        d: *d,
                    },
                });
            }
        }
        Ok(out)
    }
}

impl CloudParcel {
    /// The handoff to the parcel pool (§66): [`SeedParcel`]'s position,
    /// diameter, temperature and count, rounded to `Scalar` ONCE here -
    /// everything upstream is f64 in both builds (§159.9). At rest, and no
    /// identity: the pool derives one from the slot.
    pub fn seed_parcel(&self) -> crate::parcels::SeedParcel {
        crate::parcels::SeedParcel {
            position: Vec3::new(
                self.position.x as Scalar,
                self.position.y as Scalar,
                self.position.z as Scalar,
            ),
            velocity: Vec3::ZERO,
            diameter: self.particle.d as Scalar,
            temperature: self.particle.t as Scalar,
            n_p: self.n_p as Scalar,
            uid: None,
        }
    }
}

// ==========================================================================
//  The cloud on the dictionary route (SPEC-LIT §159.9)
// ==========================================================================

/// The `dustCloud` sub-dictionary of `constant/cloudProperties` (SPEC-LIT
/// §159.9), through the SAME dictionary-to-JSON conversion
/// `JsonParcels::from_dict` uses and then the same `deny_unknown_fields`
/// deserialiser, so the two routes cannot drift. `None` when the
/// dictionary states no `dustCloud`; a serde error is refused naming
/// `dust.cloud` and the key the deserialiser complained about.
pub fn dust_cloud_of_dict(d: &FoamDict, what: &str) -> Result<Option<JsonDustCloud>> {
    let tree = crate::io::case_parcels::dict_tree(d, what)?;
    let Some(v) = tree.get("dustCloud") else { return Ok(None) };
    serde_json::from_value(v.clone())
        .map(Some)
        .map_err(|e| Error::Refused(format!("{what}: dust.cloud: {e} {CLOUD_TAG}")))
}

/// The dictionary route of the cloud: `constant/cloudProperties` in the
/// case directory, `None` when the case has no such file.
pub fn read_dust_cloud(case_dir: &std::path::Path) -> Result<Option<JsonDustCloud>> {
    let p = case_dir.join(crate::io::case_parcels::FILE);
    if !p.exists() {
        return Ok(None);
    }
    dust_cloud_of_dict(&crate::io::dict::FoamDict::read(&p)?, &p.display().to_string())
}

#[cfg(test)]
#[path = "tests_case_dust.rs"]
mod tests;
