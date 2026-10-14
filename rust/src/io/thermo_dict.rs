// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! `constant/thermophysicalProperties`, the subset that states SPEC-LIT
//! §122.1's perfect gas (docs/17 CMP-14, SPEC-LIT §122.6).
//!
//! The subset is one `thermoType` - `hePsiThermo` or `heRhoThermo`,
//! `pureMixture`, `sutherland` or `const` transport, `hConst` thermo,
//! `perfectGas`, `specie`, and `sensibleEnthalpy` or
//! `sensibleInternalEnergy` - with its `mixture` coefficients and a top-level
//! `pOperating`. Every key outside it is refused by name, with the keys that
//! are read beside it (§13.4); every model outside it is refused with the
//! one this reader has. A dictionary states `molWeight` (g/mol) and `Cp`,
//! not `gamma`: [`ThermoDict::gas`] turns them into `W` and `gamma` by
//! (122.14), so [`PerfectGas`] stays "the gas is `(gamma, W)`" (§122.1).
//!
//! Written from:
//!   ofgpu SPEC-LIT.md §122.1 and §122.2 - the gas, its defaults
//!     (`Pr = 0.71`, `Pr_t = 0.85`) and `p_op` with no default
//!   ofgpu SPEC-LIT.md §122.6 - this subset, (122.14), and why `Hf` must
//!     be 0 and `Pr` is read under both transport models
//!   OpenFOAM User Guide v12, "Thermophysical models",
//!     <https://doc.cfd.direct/openfoam/user-guide-v12/thermophysical>
//!     (read 2026-10-14) - DOCUMENTATION, for the keyword NAMES and the
//!     unit of `molWeight` (g/mol) only
//!
//! Provenance: LITERATURE. No OpenFOAM source file was opened. No
//! GPL-licensed source was consulted.

use std::path::Path;

use crate::compressible::thermo::{PerfectGas, Viscosity, R_UNIVERSAL};
use crate::error::{Error, Result};
use crate::io::contract::unsupported;
use crate::io::dict::{last_number, FoamDict};

/// The file inside a case directory.
pub const FILE: &str = "constant/thermophysicalProperties";

/// `thermoType/type`: both name a perfect gas whose density follows from
/// `p` and `T`; which of them a case names does not change the gas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThermoType {
    HePsiThermo,
    HeRhoThermo,
}

/// `thermoType/energy`, kept for the driver (CMP-15): with constant `c_p`
/// both describe the same state, and which one a solver advances is the
/// solver's choice, not the gas's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnergyForm {
    SensibleEnthalpy,
    SensibleInternalEnergy,
}

/// The dictionary as written: every number in the unit the file states it
/// in, so [`ThermoDict::to_text`] writes back exactly what was read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThermoDict {
    pub thermo_type: ThermoType,
    pub energy: EnergyForm,
    /// `mixture/specie/molWeight`, g/mol as the User Guide states it.
    pub mol_weight: f64,
    /// `mixture/thermodynamics/Cp`, J/(kg K).
    pub cp: f64,
    /// `mixture/transport`: `sutherland` (`As`, `Ts`) or `const` (`mu`).
    pub viscosity: Viscosity,
    /// `mixture/transport/Pr`; `None` is §122.1's default, 0.71.
    pub pr: Option<f64>,
    /// `pOperating`, Pa: the gauge datum `p_op` of §122.2, no default.
    pub p_operating: f64,
}

/// §122.1's laminar Prandtl number when the dictionary states none.
pub const PR_DEFAULT: f64 = 0.71;

/// §122.1's turbulent Prandtl number (no key in this subset).
pub const PR_T_DEFAULT: f64 = 0.85;

fn refused(what: &str, msg: String) -> Error {
    Error::Refused(format!("{what}: {msg}"))
}

/// Every key directly under `prefix` must be one of `known`; anything else
/// is a §13.4 refusal naming the key and the keys that are read
/// (`-permissive` ignores it, and says so).
fn only_known_keys(d: &FoamDict, what: &str, prefix: &str, known: &[&str]) -> Result<()> {
    for key in d.sub_keys(prefix) {
        if known.contains(&key.as_str()) {
            continue;
        }
        let setting = if prefix.is_empty() {
            format!("{what}: key")
        } else {
            format!("{what}: key in {prefix}")
        };
        unsupported(&setting, &key, known, "ignoring the key", ())?;
    }
    Ok(())
}

/// A word entry that must be present.
fn word<'a>(d: &'a FoamDict, what: &str, key: &str) -> Result<&'a str> {
    d.get(key)
        .map(str::trim)
        .ok_or_else(|| Error::Config(format!("{what}: {key} is missing")))
}

/// A word entry that must be one of `menu`.
fn choice(d: &FoamDict, what: &str, key: &str, menu: &[&str]) -> Result<usize> {
    let v = word(d, what, key)?;
    menu.iter().position(|m| *m == v).ok_or_else(|| {
        refused(
            what,
            format!(
                "{key} \"{v}\" is not supported by ofgpu; available: {} (SPEC-LIT §122.6)",
                menu.join(", ")
            ),
        )
    })
}

/// A number entry, read as f64 in both builds (a dimensioned entry's last
/// number, as `FoamDict::scalar` reads one); absent is `None`, present and
/// unreadable is an error naming the key.
fn number(d: &FoamDict, what: &str, key: &str) -> Result<Option<f64>> {
    match d.get(key) {
        None => Ok(None),
        Some(raw) => last_number(raw).map(Some).ok_or_else(|| {
            Error::Config(format!("{what}: {key} \"{}\" is not a number", raw.trim()))
        }),
    }
}

fn required(d: &FoamDict, what: &str, key: &str) -> Result<f64> {
    number(d, what, key)?.ok_or_else(|| Error::Config(format!("{what}: {key} is missing")))
}

impl ThermoDict {
    /// `<case>/constant/thermophysicalProperties`.
    pub fn read(path: &Path) -> Result<Self> {
        let d = FoamDict::read(path)?;
        Self::from_dict(&d, &path.display().to_string())
    }

    /// The dictionary text, `what` naming it in messages.
    pub fn parse(src: &str, what: &str) -> Result<Self> {
        let d = FoamDict::parse(src, what)?;
        Self::from_dict(&d, what)
    }

    /// The subset, refused by name outside it (§122.6).
    pub fn from_dict(d: &FoamDict, what: &str) -> Result<Self> {
        only_known_keys(d, what, "", &["FoamFile", "thermoType", "mixture", "pOperating"])?;
        only_known_keys(
            d,
            what,
            "thermoType",
            &["type", "mixture", "transport", "thermo", "equationOfState", "specie", "energy"],
        )?;
        let thermo_type = match choice(d, what, "thermoType/type", &["hePsiThermo", "heRhoThermo"])? {
            0 => ThermoType::HePsiThermo,
            _ => ThermoType::HeRhoThermo,
        };
        choice(d, what, "thermoType/mixture", &["pureMixture"])?;
        let sutherland = choice(d, what, "thermoType/transport", &["sutherland", "const"])? == 0;
        choice(d, what, "thermoType/thermo", &["hConst"])?;
        choice(d, what, "thermoType/equationOfState", &["perfectGas"])?;
        choice(d, what, "thermoType/specie", &["specie"])?;
        let energy = match choice(
            d,
            what,
            "thermoType/energy",
            &["sensibleEnthalpy", "sensibleInternalEnergy"],
        )? {
            0 => EnergyForm::SensibleEnthalpy,
            _ => EnergyForm::SensibleInternalEnergy,
        };

        only_known_keys(d, what, "mixture", &["specie", "thermodynamics", "transport"])?;
        only_known_keys(d, what, "mixture/specie", &["molWeight", "nMoles"])?;
        let mol_weight = required(d, what, "mixture/specie/molWeight")?;
        if let Some(n) = number(d, what, "mixture/specie/nMoles")? {
            if n != 1.0 {
                return Err(refused(
                    what,
                    format!(
                        "mixture/specie/nMoles {n} is not supported by ofgpu; a pure \
                         perfect gas has nMoles 1 (SPEC-LIT §122.6)"
                    ),
                ));
            }
        }
        only_known_keys(d, what, "mixture/thermodynamics", &["Cp", "Hf"])?;
        let cp = required(d, what, "mixture/thermodynamics/Cp")?;
        if let Some(hf) = number(d, what, "mixture/thermodynamics/Hf")? {
            if hf != 0.0 {
                return Err(refused(
                    what,
                    format!(
                        "mixture/thermodynamics/Hf {hf} is not supported by ofgpu; the \
                         enthalpy datum is h(0 K) = 0 by (122.5), so Hf must be 0 \
                         (SPEC-LIT §122.6)"
                    ),
                ));
            }
        }
        let viscosity = if sutherland {
            only_known_keys(d, what, "mixture/transport", &["As", "Ts", "Pr"])?;
            Viscosity::Sutherland {
                a_s: required(d, what, "mixture/transport/As")?,
                t_s: required(d, what, "mixture/transport/Ts")?,
            }
        } else {
            only_known_keys(d, what, "mixture/transport", &["mu", "Pr"])?;
            Viscosity::Constant { mu: required(d, what, "mixture/transport/mu")? }
        };
        let pr = number(d, what, "mixture/transport/Pr")?;
        let p_operating = number(d, what, "pOperating")?.ok_or_else(|| {
            Error::Config(format!(
                "{what}: pOperating is missing; the gauge datum p_op has no default \
                 (SPEC-LIT §122.2)"
            ))
        })?;
        Ok(Self { thermo_type, energy, mol_weight, cp, viscosity, pr, p_operating })
    }

    /// The gas this dictionary states, by (122.14):
    /// `W = molWeight / 1000` and `gamma = c_p / (c_p - R_s)` with
    /// `R_s = R / W` (122.2). `Pr` defaults to 0.71 and `Pr_t` is 0.85
    /// (§122.1). [`PerfectGas::validate`] runs on the result, and `c_p` at or
    /// below `R_s` (no `gamma` above 1) is refused by name first.
    pub fn gas(&self) -> Result<PerfectGas> {
        let w = self.mol_weight / 1000.0;
        let r_s = R_UNIVERSAL / w;
        if !(self.cp.is_finite() && self.cp > r_s) {
            return Err(refused(
                "thermophysicalProperties",
                format!(
                    "mixture/thermodynamics/Cp {} must exceed R_s = R/W = {r_s} J/(kg K), \
                     or no gamma above 1 exists (SPEC-LIT §122.6)",
                    self.cp
                ),
            ));
        }
        let g = PerfectGas {
            gamma: self.cp / (self.cp - r_s),
            w,
            pr: self.pr.unwrap_or(PR_DEFAULT),
            pr_t: PR_T_DEFAULT,
            viscosity: self.viscosity,
            p_op: self.p_operating,
        };
        g.validate()?;
        Ok(g)
    }

    /// The subset written back as a dictionary that [`Self::parse`] reads to
    /// the same value bit for bit (every number in Rust's shortest
    /// round-trip form).
    pub fn to_text(&self) -> String {
        let ty = match self.thermo_type {
            ThermoType::HePsiThermo => "hePsiThermo",
            ThermoType::HeRhoThermo => "heRhoThermo",
        };
        let energy = match self.energy {
            EnergyForm::SensibleEnthalpy => "sensibleEnthalpy",
            EnergyForm::SensibleInternalEnergy => "sensibleInternalEnergy",
        };
        let (transport, coeffs) = match self.viscosity {
            Viscosity::Sutherland { a_s, t_s } => {
                ("sutherland", format!("        As              {a_s:?};\n        Ts              {t_s:?};\n"))
            }
            Viscosity::Constant { mu } => ("const", format!("        mu              {mu:?};\n")),
        };
        let pr = self
            .pr
            .map(|v| format!("        Pr              {v:?};\n"))
            .unwrap_or_default();
        format!(
            "FoamFile\n{{\n    format      ascii;\n    class       dictionary;\n    \
             object      thermophysicalProperties;\n}}\n\n\
             thermoType\n{{\n    type            {ty};\n    mixture         pureMixture;\n    \
             transport       {transport};\n    thermo          hConst;\n    \
             equationOfState perfectGas;\n    specie          specie;\n    \
             energy          {energy};\n}}\n\n\
             mixture\n{{\n    specie\n    {{\n        molWeight       {mw:?};\n    }}\n    \
             thermodynamics\n    {{\n        Cp              {cp:?};\n        Hf              0;\n    }}\n    \
             transport\n    {{\n{coeffs}{pr}    }}\n}}\n\n\
             pOperating      {pop:?};\n",
            mw = self.mol_weight,
            cp = self.cp,
            pop = self.p_operating,
        )
    }
}

/// `<case_dir>/constant/thermophysicalProperties`, read and lowered to the
/// gas in one call.
pub fn read_thermophysical_properties(case_dir: &Path) -> Result<PerfectGas> {
    ThermoDict::read(&case_dir.join(FILE))?.gas()
}

#[cfg(test)]
mod tests {
    use super::*;

    const AIR: &str = "FoamFile { format ascii; class dictionary; object thermophysicalProperties; }
thermoType
{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       sutherland;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleInternalEnergy;
}
mixture
{
    specie { molWeight 28.9647; }
    thermodynamics { Cp 1004.692579691832; Hf 0; }
    transport { As 1.458e-06; Ts 110.4; }
}
pOperating [1 -1 -2 0 0 0 0] 1e5;
";

    #[test]
    fn thermophysical_subset_round_trips() {
        let _g = crate::io::contract::permissive_test_guard();
        let d1 = ThermoDict::parse(AIR, "air").unwrap();
        assert_eq!(d1.thermo_type, ThermoType::HePsiThermo);
        assert_eq!(d1.energy, EnergyForm::SensibleInternalEnergy);
        assert_eq!(d1.viscosity, Viscosity::Sutherland { a_s: 1.458e-6, t_s: 110.4 });
        assert_eq!(d1.pr, None);
        let d2 = ThermoDict::parse(&d1.to_text(), "air, written").unwrap();
        assert_eq!(d1, d2);
        assert_eq!(d1.mol_weight.to_bits(), d2.mol_weight.to_bits());
        assert_eq!(d1.cp.to_bits(), d2.cp.to_bits());
        assert_eq!(d1.p_operating.to_bits(), d2.p_operating.to_bits());

        // (122.14): W = 28.9647 / 1000 and gamma = c_p / (c_p - R_s); the
        // oracle gives W = 0.0289647, R_s = 287.0550227690948 and
        // gamma = 1.4000000000000001.
        let g = d1.gas().unwrap();
        assert_eq!(g.w, 0.0289647);
        assert_eq!(g.r_s(), 287.0550227690948);
        assert!((g.gamma - 1.4).abs() <= 1e-15 * 1.4, "gamma {}", g.gamma);
        assert_eq!(g.pr, 0.71);
        assert_eq!(g.pr_t, 0.85);
        assert_eq!(g.p_op, 1e5);
        assert_eq!(g.viscosity, PerfectGas::air(1e5).viscosity);

        // The const transport, with its Pr, and heRhoThermo / enthalpy.
        let c = AIR
            .replace("hePsiThermo", "heRhoThermo")
            .replace("sensibleInternalEnergy", "sensibleEnthalpy")
            .replace("transport       sutherland", "transport       const")
            .replace("As 1.458e-06; Ts 110.4;", "mu 1.8e-05; Pr 0.7;");
        let c1 = ThermoDict::parse(&c, "const").unwrap();
        assert_eq!(c1.thermo_type, ThermoType::HeRhoThermo);
        assert_eq!(c1.energy, EnergyForm::SensibleEnthalpy);
        assert_eq!(c1.viscosity, Viscosity::Constant { mu: 1.8e-5 });
        assert_eq!(c1.pr, Some(0.7));
        assert_eq!(ThermoDict::parse(&c1.to_text(), "const, written").unwrap(), c1);
        assert_eq!(c1.gas().unwrap().pr, 0.7);
    }

    #[test]
    fn unknown_thermo_key_is_refused_by_name() {
        let _g = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);
        let err = |text: String| -> String {
            ThermoDict::parse(&text, "case").and_then(|d| d.gas()).unwrap_err().to_string()
        };
        let cases: [(String, &str); 8] = [
            (format!("{AIR}dpdt yes;\n"), "\"dpdt\""),
            (AIR.replace("Ts 110.4;", "Ts 110.4; kappa 0.02;"), "\"kappa\""),
            (AIR.replace("thermo          hConst", "thermo          janaf"), "hConst"),
            (
                AIR.replace("equationOfState perfectGas", "equationOfState incompressiblePerfectGas"),
                "equationOfState",
            ),
            (AIR.replace("pOperating [1 -1 -2 0 0 0 0] 1e5;", ""), "pOperating"),
            (AIR.replace("Hf 0;", "Hf 1;"), "Hf"),
            (AIR.replace("molWeight 28.9647;", ""), "molWeight"),
            (AIR.replace("Cp 1004.692579691832;", "Cp 200;"), "Cp"),
        ];
        for (text, key) in cases {
            let msg = err(text);
            assert!(msg.contains(key), "expected the message to name {key}: {msg}");
        }
    }
}
