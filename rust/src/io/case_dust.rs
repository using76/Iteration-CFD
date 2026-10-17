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

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::dust::courant::CO_F_MAX;
use crate::dust::fsc::{CombustionModel, A_FSC};
use crate::dust::ignition::IgnitionSource;
use crate::dust::laminar::LaminarSpeed;
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
use crate::io::fields::PatchFieldSpec;
use crate::{DVec3, Scalar};

const TAG: &str = "(SPEC-LIT §157.4)";
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
/// modes; the keys each mode reads are refused on the other.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonDust {
    /// `formula` (the pseudo-species of (152.1)-(152.8) at phi) |
    /// `prescribed` (one constant c_p, §152.9). Required, no default.
    pub thermo: String,
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
    // without a dust block, each by its own key.
    let dust = match (dust, is_dust_model(combustion)) {
        (None, false) => return Ok(None),
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
    // ---- thermo: the two modes of §152.9, each refused on the other's keys
    let thermo = match dust.thermo.as_str() {
        "formula" => {
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
        "prescribed" => {
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
        other => {
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

#[cfg(test)]
#[path = "tests_case_dust.rs"]
mod tests;
