// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The equivalent-gas thermochemistry of the corn-starch dust: the starch
//! fuel record from its Table-1 polynomial, the exact per-mole-fuel
//! composition at phi, the two NASA-7 pseudo-species U and B built as exact
//! linear combinations of their components, the prescribed constant-c_p
//! mode, and the isobaric, AICC and heat-of-reaction states. Host only and
//! `f64` everywhere, so it is identical in both builds.
//!
//! Written from:
//!   SPEC-LIT §152 - the reaction and phi of §152.1, the starch polynomial
//!     and its derived a5 of §152.2, the heat of reaction (152.6) of §152.3,
//!     the pseudo-species combination (152.7)-(152.8) of §152.4, the state
//!     (152.9) and the isobaric burned state (152.10) of §152.5, the AICC
//!     state (152.11) of §152.6, the heat release (152.12) of §152.7 and the
//!     two modes of §152.9
//!   C. Huang, M. Bloching & A. N. Lipatnikov, J. Loss Prev. Process Ind. 75
//!     (2022) 104707, DOI 10.1016/j.jlp.2021.104707 - eqs. 1-2, the starch
//!     polynomial c_p,cs(T) and its integral, and Table 1
//!   C. Huang, A. N. Lipatnikov & K. Nessvi, J. Loss Prev. Process Ind. 67
//!     (2020) 104237, DOI 10.1016/j.jlp.2020.104237 - Table 1, the Leeds
//!     prescribed mode's W_u, W_b, T_u and T_b
//!   NASA TM-4513 Table II through §128 - the N2, O2, CO2 and H2O records
//! No GPL-licensed source was consulted.

use crate::chem::thermo::{P_BAR, R_GAS, SpeciesThermo, ThermoSet};
use crate::error::Error;

/// C, g/mol: the IUPAC conventional atomic weight the formula molar mass is
/// built from (SPEC-LIT §152.1). The TM-4513 element table is NOT used here:
/// the pseudo-species' component sums must reproduce the formula's W.
pub const IUPAC_C: f64 = 12.011;
/// H, g/mol (SPEC-LIT §152.1).
pub const IUPAC_H: f64 = 1.008;
/// O, g/mol (SPEC-LIT §152.1).
pub const IUPAC_O: f64 = 15.999;
/// K, the standard temperature the formation enthalpies and the derived a5
/// live at (SPEC-LIT §152.2).
pub const T_STD: f64 = 298.15;
/// Air as O2 : N2 = 1 : 3.76, in hundredths (SPEC-LIT §152.1).
pub const N2_PER_O2_CENTI: i64 = 376;

/// The corn starch fuel, as the integer atom counts of its formula
/// `C6H7.88O4.98` held to hundredths and the two coefficients of its linear
/// c_p polynomial with the heat of formation (SPEC-LIT §152.1-§152.2).
#[derive(Clone, Debug, PartialEq)]
pub struct StarchFuel {
    /// Carbon atoms x 100 of the formula (600 for C6).
    pub c_centi: i64,
    /// Hydrogen atoms x 100 (788 for H7.88).
    pub h_centi: i64,
    /// Oxygen atoms x 100 (498 for O4.98).
    pub o_centi: i64,
    /// a0 of (152.4), dimensionless (-3.2726).
    pub a0: f64,
    /// a1 of (152.4), 1/K (0.10056). a2 = a3 = a4 = 0 is the contract.
    pub a1: f64,
    /// The standard heat of formation, J/mol (-792.6e3). The datum the
    /// derived a5 is built from; the printed a5 is never read.
    pub hf: f64,
}

impl StarchFuel {
    /// The fuel read from a Table-1 CSV by header name: the columns n_C,
    /// n_H, n_O, Hf_cs (kJ/mol) and a0..a4. `#` lines are skipped. The atom
    /// counts are parsed from their decimal text to hundredths and refused
    /// above two decimals or at or below zero; a2, a3 or a4 not exactly zero
    /// is refused, because the linear polynomial IS the contract
    /// (SPEC-LIT §152.2). The W_cs, a5, R_spec, R0 and later columns are
    /// not used.
    pub fn from_table1_csv(text: &str) -> crate::Result<StarchFuel> {
        const W: &str = "the corn starch table-1 CSV";
        let mut header: Option<&str> = None;
        let mut rows: Vec<&str> = Vec::new();
        for l in text.lines() {
            let l = l.trim();
            if l.is_empty() || l.starts_with('#') {
                continue;
            }
            if header.is_none() {
                header = Some(l);
            } else {
                rows.push(l);
            }
        }
        let header = header.ok_or_else(|| {
            Error::Config(format!("{W} has no header line naming its columns (SPEC-LIT §152.2)"))
        })?;
        let names: Vec<&str> = header.split(',').map(|s| s.trim()).collect();
        let col = |name: &str| -> crate::Result<usize> {
            names.iter().position(|n| *n == name).ok_or_else(|| {
                Error::Config(format!("{W} has no column named {name} (SPEC-LIT §152.2)"))
            })
        };
        let ic = col("n_C")?;
        let ih = col("n_H")?;
        let io = col("n_O")?;
        let ihf = col("Hf_cs")?;
        let ia = [col("a0")?, col("a1")?, col("a2")?, col("a3")?, col("a4")?];
        if rows.len() != 1 {
            return Err(Error::Config(format!(
                "{W} holds {} data row(s) after its header; exactly one fuel row is \
                 expected (SPEC-LIT §152.2)",
                rows.len()
            )));
        }
        let c: Vec<&str> = rows[0].split(',').map(|s| s.trim()).collect();
        let field = |i: usize, name: &str| -> crate::Result<&str> {
            c.get(i).copied().ok_or_else(|| {
                Error::Config(format!(
                    "{W} row has no field {i} for column {name} (SPEC-LIT §152.2)"
                ))
            })
        };
        let c_centi = Self::centi(field(ic, "n_C")?, "n_C")?;
        let h_centi = Self::centi(field(ih, "n_H")?, "n_H")?;
        let o_centi = Self::centi(field(io, "n_O")?, "n_O")?;
        let hf_text = field(ihf, "Hf_cs")?;
        let hf_kj: f64 = hf_text.parse().map_err(|_| {
            Error::Config(format!(
                "{W} column Hf_cs reads {hf_text:?}, which is not a number (SPEC-LIT §152.2)"
            ))
        })?;
        let mut a = [0.0f64; 5];
        for (k, &i) in ia.iter().enumerate() {
            let name = format!("a{k}");
            let text = field(i, &name)?;
            a[k] = text.parse().map_err(|_| {
                Error::Config(format!(
                    "{W} column a{k} reads {text:?}, which is not a number (SPEC-LIT §152.2)"
                ))
            })?;
        }
        for (k, v) in a.iter().enumerate().skip(2) {
            if *v != 0.0 {
                return Err(Error::Config(format!(
                    "{W} column a{k} reads {v}, which is not zero; the starch polynomial \
                     is linear in T with a2 = a3 = a4 = 0, and the linear form is the \
                     contract (SPEC-LIT §152.2)"
                )));
            }
        }
        Ok(StarchFuel { c_centi, h_centi, o_centi, a0: a[0], a1: a[1], hf: hf_kj * 1000.0 })
    }

    /// One atom count from its decimal text to hundredths: "7.88" is 788.
    /// Refused: more than two decimal places, a non-integer part, or a
    /// value at or below zero (SPEC-LIT §152.1).
    fn centi(text: &str, name: &str) -> crate::Result<i64> {
        let bad = |why: String| {
            Error::Config(format!(
                "the corn starch table-1 CSV column {name} reads {text:?}, which {why} \
                 (SPEC-LIT §152.1)"
            ))
        };
        let negative = text.starts_with('-');
        let body = text.strip_prefix(['+', '-']).unwrap_or(text);
        let (int_part, frac_part) = match body.split_once('.') {
            Some((i, f)) => (i, f),
            None => (body, ""),
        };
        if frac_part.len() > 2 {
            return Err(bad("has more than two decimal places; the atom counts are \
                            read to hundredths"
                .to_string()));
        }
        let int_v: i64 = if int_part.is_empty() {
            0
        } else {
            int_part.parse().map_err(|_| bad("is not a whole atom count".to_string()))?
        };
        let mut v = int_v * 100;
        if !frac_part.is_empty() {
            let mut f: i64 =
                frac_part.parse().map_err(|_| bad("is not a decimal atom count".to_string()))?;
            if frac_part.len() == 1 {
                f *= 10;
            }
            v += f;
        }
        if negative {
            v = -v;
        }
        if v <= 0 {
            return Err(bad("is not an atom count above zero".to_string()));
        }
        Ok(v)
    }

    /// The fuel of H22 Table 1 as the repository tracks it
    /// (SPEC-LIT §152.1-§152.2).
    pub fn huang2022_table1() -> crate::Result<StarchFuel> {
        Self::from_table1_csv(include_str!(
            "../../../reference/huang2022-rembe/corn_starch_table1.csv"
        ))
    }

    /// kg/mol, from the formula with the IUPAC conventional weights:
    /// (c*12.011 + h*1.008 + o*15.999)/1000 = 0.15968406 for the starch
    /// (SPEC-LIT §152.1). The printed 0.16 enters nothing.
    pub fn molar_mass(&self) -> f64 {
        let c = self.c_centi as f64 / 100.0;
        let h = self.h_centi as f64 / 100.0;
        let o = self.o_centi as f64 / 100.0;
        (c * IUPAC_C + h * IUPAC_H + o * IUPAC_O) / 1000.0
    }

    /// The constant of (152.5) re-derived from the heat of formation, K:
    /// a5 = H_f,cs/R0 - (a1 T^2/2 + a0 T) at T = 298.15 K, in f64 from the
    /// inputs (SPEC-LIT §152.2). The printed -99808 is never used.
    pub fn a5(&self) -> f64 {
        self.hf / R_GAS - (self.a1 * T_STD * T_STD / 2.0 + self.a0 * T_STD)
    }

    /// 100*(n_C + n_H/4 - n_O/2), the stoichiometric oxygen per mole of
    /// fuel in hundredths, exactly: 548 for the starch. Refused: a result
    /// that is not a whole number of hundredths, or one at or below zero
    /// (SPEC-LIT §152.1).
    pub fn o2_stoich_centi(&self) -> crate::Result<i64> {
        let num = 4 * self.c_centi + self.h_centi - 2 * self.o_centi;
        if num <= 0 || num % 4 != 0 {
            return Err(Error::Config(format!(
                "the starch formula n_C = {}, n_H = {}, n_O = {} gives the stoichiometric \
                 oxygen 100(n_C + n_H/4 - n_O/2) = {num}/4 in hundredths, which is not a \
                 whole count above zero (SPEC-LIT §152.1)",
                self.c_centi, self.h_centi, self.o_centi
            )));
        }
        Ok(num / 4)
    }

    /// The starch as an equivalent-gas "vapour" record in the NASA-7 form:
    /// b1 = a0, b2 = a1, b3 = b4 = b5 = 0, b6 = the derived a5, b7 = 0 (no
    /// equilibrium or entropy uses the starch), one range on both intervals,
    /// elements empty because the formula is fractional (SPEC-LIT §152.4).
    pub fn species(&self) -> SpeciesThermo {
        let a5 = self.a5();
        let coeffs = [self.a0, self.a1, 0.0, 0.0, 0.0, a5, 0.0];
        SpeciesThermo {
            name: "STARCH".to_string(),
            elements: Vec::new(),
            molar_mass: self.molar_mass(),
            t_low: 200.0,
            t_mid: 1000.0,
            t_high: 6000.0,
            upper: coeffs,
            lower: coeffs,
            line: 0,
        }
    }
}

// ==========================================================================
//  Exact rational mole numbers
// ==========================================================================

/// An exact rational with an i128 numerator and a positive i128 denominator,
/// always reduced. The mole numbers of (152.1) at phi are formed in this
/// arithmetic from the integer inputs and converted to f64 once
/// (SPEC-LIT §152.4).
#[derive(Clone, Copy, Debug)]
struct Rat {
    num: i128,
    den: i128,
}

fn gcd(a: u128, b: u128) -> u128 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

impl Rat {
    /// Reduced, denominator positive. Err when the denominator is zero or a
    /// numerator or denominator overflows i128.
    fn new(num: i128, den: i128) -> crate::Result<Rat> {
        if den == 0 {
            return Err(Error::Config(format!(
                "a mole number was formed as {num}/0, which is not a rational \
                 (SPEC-LIT §152.4)"
            )));
        }
        let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };
        let g = gcd(num.unsigned_abs(), den as u128) as i128;
        let (num, den) = (num.checked_div(g).ck()?, den.checked_div(g).ck()?);
        Ok(Rat { num, den })
    }

    /// A whole number.
    fn whole(v: i64) -> Rat {
        Rat { num: v as i128, den: 1 }
    }

    fn mul(self, o: Rat) -> crate::Result<Rat> {
        Rat::new(self.num.checked_mul(o.num).ck()?, self.den.checked_mul(o.den).ck()?)
    }

    fn add(self, o: Rat) -> crate::Result<Rat> {
        let hi = self.num.checked_mul(o.den).ck()?;
        let lo = o.num.checked_mul(self.den).ck()?;
        Rat::new(hi.checked_add(lo).ck()?, self.den.checked_mul(o.den).ck()?)
    }

    fn sub(self, o: Rat) -> crate::Result<Rat> {
        self.add(Rat { num: -o.num, den: o.den })
    }

    /// The value as f64, converted once: num as f64 / den as f64
    /// (SPEC-LIT §152.4).
    fn value(self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

/// The i128-overflow refusal of the exact mole arithmetic.
fn oflow() -> Error {
    Error::Config(
        "the exact mole arithmetic of the equivalent gas overflowed i128 \
         (SPEC-LIT §152.4)"
            .to_string(),
    )
}

/// A `?`-friendly view of `checked_mul`: the overflow carries the full
/// refusal message.
trait Checked {
    fn ck(self) -> crate::Result<i128>;
}

impl Checked for Option<i128> {
    fn ck(self) -> crate::Result<i128> {
        self.ok_or_else(oflow)
    }
}

// ==========================================================================
//  The composition at phi
// ==========================================================================

/// The mole numbers per mole of fuel, exact (SPEC-LIT §152.1): the
/// charge's O2 and N2 and the products' CO2, H2O and excess O2. N2 of B
/// equals N2 of U.
#[derive(Clone, Copy, Debug)]
struct Moles {
    o2_u: Rat,
    n2_u: Rat,
    co2_b: Rat,
    h2o_b: Rat,
    o2_b: Rat,
}

/// The formula mode's inputs: the starch fuel, the air ratio in hundredths
/// and phi as an exact fraction (SPEC-LIT §152.1-§152.2).
#[derive(Clone, Debug)]
pub struct FormulaInputs {
    /// The starch fuel record.
    pub fuel: StarchFuel,
    /// N2 per O2 in hundredths (376: air is O2 : N2 = 1 : 3.76).
    pub n2_per_o2_centi: i64,
    /// The numerator of phi.
    pub phi_num: i64,
    /// The denominator of phi.
    pub phi_den: i64,
}

impl FormulaInputs {
    /// The REMBE charge of H22: the Table-1 fuel, air 1 : 3.76 and
    /// phi = 62/100, the corrected equivalence ratio of the nominal
    /// 750 g/m3 vessel charge (SPEC-LIT §152.1-§152.2).
    pub fn huang2022_rembe() -> crate::Result<FormulaInputs> {
        Ok(FormulaInputs {
            fuel: StarchFuel::huang2022_table1()?,
            n2_per_o2_centi: N2_PER_O2_CENTI,
            phi_num: 62,
            phi_den: 100,
        })
    }

    /// phi as f64.
    pub fn phi(&self) -> f64 {
        self.phi_num as f64 / self.phi_den as f64
    }

    /// The exact composition at phi, validated: phi_den > 0 and
    /// 0 < phi <= 1, since only lean or stoichiometric charges burn
    /// (SPEC-LIT §152.2). The phi <= 1 half is the exact integer comparison
    /// phi_num <= phi_den, not an f64 one: phi = (1e18 + 1)/1e18 rounds to
    /// exactly 1.0 in f64 and is still richer than stoichiometric.
    fn composition(&self) -> crate::Result<Moles> {
        if self.phi_den <= 0 {
            return Err(Error::Config(format!(
                "phi_den = {} is not a positive denominator; phi is the exact fraction \
                 phi_num/phi_den (SPEC-LIT §152.2)",
                self.phi_den
            )));
        }
        if self.phi_num <= 0 {
            return Err(Error::Config(format!(
                "phi = {}/{} is not a charge above zero; only lean or stoichiometric \
                 charges burn, 0 < phi <= 1 (SPEC-LIT §152.2)",
                self.phi_num, self.phi_den
            )));
        }
        if self.phi_num > self.phi_den {
            return Err(Error::Config(format!(
                "phi = {}/{} is richer than stoichiometric; only lean or \
                 stoichiometric charges burn, 0 < phi <= 1 (SPEC-LIT §152.2)",
                self.phi_num, self.phi_den
            )));
        }
        if self.n2_per_o2_centi <= 0 {
            return Err(Error::Config(format!(
                "n2_per_o2_centi = {} is not an air ratio above zero; air is \
                 O2 : N2 = 1 : 3.76 (SPEC-LIT §152.1)",
                self.n2_per_o2_centi
            )));
        }
        let o2_st = Rat::new(self.fuel.o2_stoich_centi()? as i128, 100)?;
        let o2_u = o2_st.mul(Rat::new(self.phi_den as i128, self.phi_num as i128)?)?;
        let n2_u = o2_u.mul(Rat::new(self.n2_per_o2_centi as i128, 100)?)?;
        let co2_b = Rat::new(self.fuel.c_centi as i128, 100)?;
        let h2o_b = Rat::new(self.fuel.h_centi as i128, 200)?;
        let o2_b = o2_u.sub(o2_st)?;
        Ok(Moles { o2_u, n2_u, co2_b, h2o_b, o2_b })
    }

    /// The element balance of the reaction at phi: C, H, O, N, the atoms of
    /// U minus the atoms of B per mole of fuel, each a reduced exact rational
    /// with a positive denominator. All four are zero for the complete
    /// reaction (SPEC-LIT §152.1). Refused: phi_den <= 0, phi <= 0,
    /// phi > 1.
    pub fn element_residual_exact(&self) -> crate::Result<[(i128, i128); 4]> {
        let m = self.composition()?;
        let two = Rat::whole(2);
        let n_c = Rat::new(self.fuel.c_centi as i128, 100)?;
        let n_h = Rat::new(self.fuel.h_centi as i128, 100)?;
        let n_o = Rat::new(self.fuel.o_centi as i128, 100)?;
        // C: the fuel's carbon against the CO2 of B.
        let c = n_c.sub(m.co2_b)?;
        // H: the fuel's hydrogen against the H2O of B.
        let h = n_h.sub(m.h2o_b.mul(two)?)?;
        // O: the fuel's oxygen and the O2 of U against CO2, H2O and the
        // excess O2 of B.
        let u_o = n_o.add(m.o2_u.mul(two)?)?;
        let b_o = m.co2_b.mul(two)?.add(m.h2o_b)?.add(m.o2_b.mul(two)?)?;
        let o = u_o.sub(b_o)?;
        // N: twice the N2 of U against twice the N2 of B.
        let n = m.n2_u.mul(two)?.sub(m.n2_u.mul(two)?)?;
        Ok([(c.num, c.den), (h.num, h.den), (o.num, o.den), (n.num, n.den)])
    }
}

/// One TM-4513 species entry: its table code, its name and its element
/// composition pairs.
type GasCode = (&'static str, &'static str, &'static [(&'static str, u32)]);

/// The four gas species of the reaction, read from the tracked TM-4513
/// Table II transcription: N2, O2, CO2 and H2O, codes 1, 2, 6 and 7, with
/// elements N2, O2, C1O2 and H2O1. The 1000-6000 K row of a species is its
/// `upper` interval, the 200-1000 K row its `lower`; t 200/1000/6000 K,
/// molar mass mol_weight/1000, standard pressure 1 bar, line 0.
pub fn tm4513_gas_set() -> crate::Result<ThermoSet> {
    const SOURCE: &str = "reference/mcbride-gordon-reno1993/table_II.csv";
    const TEXT: &str = include_str!("../../../reference/mcbride-gordon-reno1993/table_II.csv");
    const CODES: [GasCode; 4] = [
        ("1", "N2", &[("N", 2)]),
        ("2", "O2", &[("O", 2)]),
        ("6", "CO2", &[("C", 1), ("O", 2)]),
        ("7", "H2O", &[("H", 2), ("O", 1)]),
    ];
    let mut rows: Vec<(String, f64, f64, [f64; 7])> = Vec::new();
    for (no, line) in TEXT.lines().enumerate() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') || l.starts_with("species") {
            continue;
        }
        let bad = |msg: String| {
            Error::Config(format!("{SOURCE} line {}: {msg} (SPEC-LIT §152.4)", no + 1))
        };
        let c: Vec<&str> = l.split(',').map(|s| s.trim()).collect();
        if c.len() < 11 {
            return Err(bad(format!("{} columns, 11 expected", c.len())));
        }
        let num = |i: usize| -> crate::Result<f64> {
            c[i].parse::<f64>().map_err(|_| bad(format!("{:?} is not a number", c[i])))
        };
        rows.push((
            c[0].to_string(),
            num(1)?,
            num(3)?,
            [num(4)?, num(5)?, num(6)?, num(7)?, num(8)?, num(9)?, num(10)?],
        ));
    }
    let pick = |code: &str, t_lo: f64| -> crate::Result<(f64, [f64; 7])> {
        rows
            .iter()
            .find(|(k, t, _, _)| k == code && *t == t_lo)
            .map(|r| (r.2, r.3))
            .ok_or_else(|| {
                Error::Config(format!(
                    "{SOURCE} has no row for species code {code} starting at t = {t_lo} K \
                     (SPEC-LIT §152.4)"
                ))
            })
    };
    let mut v = Vec::new();
    for (code, name, els) in CODES {
        let (mol_weight, upper) = pick(code, 1000.0)?;
        let (_, lower) = pick(code, 200.0)?;
        v.push(SpeciesThermo {
            name: name.to_string(),
            elements: els.iter().map(|(s, n)| (s.to_string(), *n)).collect(),
            molar_mass: mol_weight / 1000.0,
            t_low: 200.0,
            t_mid: 1000.0,
            t_high: 6000.0,
            upper,
            lower,
            line: 0,
        });
    }
    ThermoSet::from_species(v, P_BAR, "TM-4513 Table II")
}

/// The thermo mode of an [`EquivalentGas`] (SPEC-LIT §152.9): formula (the
/// REMBE charge built from the reaction and the species records) or
/// prescribed (the Leeds constant-c_p form).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThermoMode {
    /// U and B built from (152.1)-(152.8) at the case's phi.
    Formula,
    /// One constant c_p for both pseudo-species, with the stated W and the
    /// measured burned temperature.
    Prescribed,
}

/// The prescribed mode's inputs, SI (SPEC-LIT §152.9).
#[derive(Clone, Debug)]
pub struct PrescribedInputs {
    /// J/(kg K), the one c_p both pseudo-species carry.
    pub cp: f64,
    /// kg/mol, the unburned molar mass (0.0328 for Leeds).
    pub w_u: f64,
    /// kg/mol, the burned molar mass (0.0297 for Leeds).
    pub w_b: f64,
    /// K, the unburned temperature (328 for Leeds).
    pub t_u: f64,
    /// K, the measured burned temperature (1500 for Leeds).
    pub t_b: f64,
}

impl PrescribedInputs {
    /// The Leeds case of H20 Table 1 at the case's stated c_p (SPEC-LIT
    /// §152.9): W_u = 32.8 g/mol, W_b = 29.7 g/mol, T_u = 328 K and the
    /// measured T_b = 1500 K.
    pub fn huang2020_leeds(cp: f64) -> PrescribedInputs {
        PrescribedInputs { cp, w_u: 0.0328, w_b: 0.0297, t_u: 328.0, t_b: 1500.0 }
    }
}

/// The isobaric adiabatic burned state (SPEC-LIT §152.5).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BurnedState {
    /// K, the burned temperature that makes h_B(T_b) = h_U(T_u).
    pub t_b: f64,
    /// rho_u/rho_b = (T_b W_U)/(T_u W_B).
    pub sigma: f64,
}

/// The adiabatic isochoric complete-combustion state (SPEC-LIT §152.6).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AiccState {
    /// K, the constant-volume burned temperature u_B(T_v) = u_U(T_0).
    pub t_v: f64,
    /// P_max/p_0 = (T_v W_U)/(T_0 W_B), the dust counted as gas moles.
    pub p_ratio: f64,
}

/// The two pseudo-species of the equivalent premixed gas and everything the
/// thermo of the stream reads from them (SPEC-LIT §152.4). Both are exact
/// NASA-7 linear combinations of their components, so [`crate::chem::thermo`]
/// carries them unchanged on host and device.
#[derive(Clone, Debug)]
pub struct EquivalentGas {
    /// Formula or prescribed.
    pub mode: ThermoMode,
    /// The unburned pseudo-species record, named DUST_U.
    pub unburned: SpeciesThermo,
    /// The burned pseudo-species record, named DUST_B.
    pub burned: SpeciesThermo,
    /// (record, Y_k) of U; empty in the prescribed mode.
    pub components_u: Vec<(SpeciesThermo, f64)>,
    /// (record, Y_k) of B; empty in the prescribed mode.
    pub components_b: Vec<(SpeciesThermo, f64)>,
    /// Moles per mole of fuel, parallel to `components_u`; empty if prescribed.
    pub moles_u: Vec<f64>,
    /// Moles per mole of fuel, parallel to `components_b`; empty if prescribed.
    pub moles_b: Vec<f64>,
}

impl EquivalentGas {
    /// The formula mode at phi (SPEC-LIT §152.4): U from STARCH, O2 and N2,
    /// B from CO2, H2O, the excess O2 (omitted when its moles are exactly
    /// zero, that is at phi = 1) and N2, each pseudo-species the exact NASA-7
    /// combination of its components. Refused: a species the set does not
    /// carry, or a component whose two intervals do not join at 1000 K.
    pub fn formula(inp: &FormulaInputs, gas: &ThermoSet) -> crate::Result<EquivalentGas> {
        let m = inp.composition()?;
        let starch = inp.fuel.species();
        let need = |name: &str| -> crate::Result<SpeciesThermo> {
            gas.get(name).cloned().ok_or_else(|| {
                Error::Config(format!(
                    "the thermo set read from {} has no record named {name}; the \
                     equivalent gas of (152.8) needs it (SPEC-LIT §152.4)",
                    gas.source()
                ))
            })
        };
        let o2 = need("O2")?;
        let n2 = need("N2")?;
        let co2 = need("CO2")?;
        let h2o = need("H2O")?;
        for r in [&starch, &o2, &n2, &co2, &h2o] {
            if r.t_mid != 1000.0 {
                return Err(Error::Config(format!(
                    "species {} has t_mid = {} K; every component of the equivalent gas \
                     joins its two intervals at 1000 K, which is what makes the \
                     pseudo-species an exact NASA-7 combination (SPEC-LIT §152.4)",
                    r.name, r.t_mid
                )));
            }
        }
        let u_pairs: Vec<(&SpeciesThermo, Rat)> =
            vec![(&starch, Rat::whole(1)), (&o2, m.o2_u), (&n2, m.n2_u)];
        let mut b_pairs: Vec<(&SpeciesThermo, Rat)> = vec![(&co2, m.co2_b), (&h2o, m.h2o_b)];
        if m.o2_b.num != 0 {
            b_pairs.push((&o2, m.o2_b));
        }
        b_pairs.push((&n2, m.n2_u));
        let (unburned, components_u, moles_u) = build_pseudo("DUST_U", &u_pairs)?;
        let (burned, components_b, moles_b) = build_pseudo("DUST_B", &b_pairs)?;
        Ok(EquivalentGas {
            mode: ThermoMode::Formula,
            unburned,
            burned,
            components_u,
            components_b,
            moles_u,
            moles_b,
        })
    }

    /// The prescribed mode (SPEC-LIT §152.9): one constant c_p for both
    /// pseudo-species, h_U^f = 0 and h_B^f = c_p (T_u - T_b), so per kg
    /// h_X(T) = h_X^f + c_p (T - 298.15 K) and the isobaric burned state is
    /// exactly T_b. As NASA-7 per mole: b1 = c_p W_X/R0, b6 =
    /// (h_X^f - c_p 298.15) W_X/R0, one set on both intervals. Refused: a
    /// non-finite or non-positive input, or t_b not above t_u.
    pub fn prescribed(p: &PrescribedInputs) -> crate::Result<EquivalentGas> {
        for (nm, v) in
            [("cp", p.cp), ("w_u", p.w_u), ("w_b", p.w_b), ("t_u", p.t_u), ("t_b", p.t_b)]
        {
            if !v.is_finite() || v <= 0.0 {
                return Err(Error::Config(format!(
                    "the prescribed thermo input {nm} = {v} is not a finite positive \
                     value (SPEC-LIT §152.9)"
                )));
            }
        }
        if p.t_b <= p.t_u {
            return Err(Error::Config(format!(
                "the prescribed thermo input t_b = {} K is not above t_u = {} K; the \
                 burned state is the hotter one (SPEC-LIT §152.9)",
                p.t_b, p.t_u
            )));
        }
        let record = |name: &str, w: f64, hf: f64| -> SpeciesThermo {
            let b1 = p.cp * w / R_GAS;
            let b6 = (hf - p.cp * T_STD) * w / R_GAS;
            let coeffs = [b1, 0.0, 0.0, 0.0, 0.0, b6, 0.0];
            SpeciesThermo {
                name: name.to_string(),
                elements: Vec::new(),
                molar_mass: w,
                t_low: 200.0,
                t_mid: 1000.0,
                t_high: 6000.0,
                upper: coeffs,
                lower: coeffs,
                line: 0,
            }
        };
        Ok(EquivalentGas {
            mode: ThermoMode::Prescribed,
            unburned: record("DUST_U", p.w_u, 0.0),
            burned: record("DUST_B", p.w_b, p.cp * (p.t_u - p.t_b)),
            components_u: Vec::new(),
            components_b: Vec::new(),
            moles_u: Vec::new(),
            moles_b: Vec::new(),
        })
    }

    /// kg/mol, the unburned pseudo-species' molar mass (SPEC-LIT §152.4).
    pub fn w_u(&self) -> f64 {
        self.unburned.molar_mass
    }

    /// kg/mol, the burned pseudo-species' molar mass (SPEC-LIT §152.4).
    pub fn w_b(&self) -> f64 {
        self.burned.molar_mass
    }

    /// Refuses a progress variable that is not a finite number in [0, 1]
    /// (SPEC-LIT §152.4).
    fn check_c(c: f64) -> crate::Result<()> {
        if !c.is_finite() || !(0.0..=1.0).contains(&c) {
            return Err(Error::Config(format!(
                "the progress variable c = {c} is not a finite number in [0, 1] \
                 (SPEC-LIT §152.4)"
            )));
        }
        Ok(())
    }

    /// 1/W(c) = (1 - c)/W_U + c/W_B, kg/mol (SPEC-LIT §152.4).
    pub fn w(&self, c: f64) -> crate::Result<f64> {
        Self::check_c(c)?;
        Ok(1.0 / ((1.0 - c) / self.w_u() + c / self.w_b()))
    }

    /// J/(kg K): the record's c_p per mole over its W (SPEC-LIT §152.4).
    pub fn cp_u(&self, t: f64) -> crate::Result<f64> {
        Ok(self.unburned.cp(t)? / self.w_u())
    }

    /// J/(kg K): the burned record's c_p per kg (SPEC-LIT §152.4).
    pub fn cp_b(&self, t: f64) -> crate::Result<f64> {
        Ok(self.burned.cp(t)? / self.w_b())
    }

    /// J/kg, formation included: the record's h per mole over its W
    /// (SPEC-LIT §152.4).
    pub fn h_u(&self, t: f64) -> crate::Result<f64> {
        Ok(self.unburned.h(t)? / self.w_u())
    }

    /// J/kg, formation included: the burned record's h per kg
    /// (SPEC-LIT §152.4).
    pub fn h_b(&self, t: f64) -> crate::Result<f64> {
        Ok(self.burned.h(t)? / self.w_b())
    }

    /// J/(kg K) at p_ref, no mixing term: the record's s per kg
    /// (SPEC-LIT §152.4).
    pub fn s_u(&self, t: f64) -> crate::Result<f64> {
        Ok(self.unburned.s(t)? / self.w_u())
    }

    /// J/(kg K) at p_ref, no mixing term: the burned record's s per kg
    /// (SPEC-LIT §152.4).
    pub fn s_b(&self, t: f64) -> crate::Result<f64> {
        Ok(self.burned.s(t)? / self.w_b())
    }

    /// J/(kg K), the mixture's c_p at the progress variable
    /// (SPEC-LIT §152.4).
    pub fn cp(&self, t: f64, c: f64) -> crate::Result<f64> {
        Self::check_c(c)?;
        Ok((1.0 - c) * self.cp_u(t)? + c * self.cp_b(t)?)
    }

    /// J/kg, the mixture's enthalpy at the progress variable, with c exactly
    /// the mass fraction of B (SPEC-LIT §152.4).
    pub fn h(&self, t: f64, c: f64) -> crate::Result<f64> {
        Self::check_c(c)?;
        Ok((1.0 - c) * self.h_u(t)? + c * self.h_b(t)?)
    }

    /// kg/m3, (152.9): p W(c)/(R0 T) (SPEC-LIT §152.5).
    pub fn rho(&self, p: f64, t: f64, c: f64) -> crate::Result<f64> {
        if !p.is_finite() || p <= 0.0 {
            return Err(Error::Config(format!(
                "the pressure p = {p} Pa is not a finite positive value (SPEC-LIT §152.5)"
            )));
        }
        if !t.is_finite() {
            return Err(Error::Config(format!(
                "the temperature t = {t} K is not finite (SPEC-LIT §152.5)"
            )));
        }
        Ok(p * self.w(c)? / (R_GAS * t))
    }

    /// J/kg, the heat-release factor of (152.12): h_U(t) - h_B(t)
    /// (SPEC-LIT §152.7).
    pub fn heat_release(&self, t: f64) -> crate::Result<f64> {
        Ok(self.h_u(t)? - self.h_b(t)?)
    }

    /// J per mole of fuel, (152.6) at 298.15 K from the formation
    /// enthalpies of the products against the reactants, negative for an
    /// exothermic reaction (SPEC-LIT §152.3). Err in the prescribed mode,
    /// which states no formation enthalpies.
    pub fn heat_of_reaction(&self) -> crate::Result<f64> {
        if self.mode == ThermoMode::Prescribed {
            return Err(Error::Config(
                "the heat of reaction (152.6) needs the formation enthalpies of the \
                 components, which the prescribed mode does not state (SPEC-LIT §152.3, \
                 §152.9)"
                    .to_string(),
            ));
        }
        let sum = |comps: &[(SpeciesThermo, f64)], moles: &[f64]| -> crate::Result<f64> {
            let mut s = 0.0;
            for (k, (r, _)) in comps.iter().enumerate() {
                s += moles[k] * r.h(T_STD)?;
            }
            Ok(s)
        };
        let products = sum(&self.components_b, &self.moles_b)?;
        let reactants = sum(&self.components_u, &self.moles_u)?;
        Ok(products - reactants)
    }

    /// The isobaric adiabatic burned state (152.10): the T_b on
    /// [T_u, t_high of the burned pseudo-species] that makes h_B(T_b) =
    /// h_U(T_u), bisected to full f64 precision, and the density ratio
    /// sigma = (T_b W_U)/(T_u W_B) (SPEC-LIT §152.5).
    pub fn isobaric_burned(&self, t_u: f64) -> crate::Result<BurnedState> {
        let h_u = self.h_u(t_u)?;
        let f = |t: f64| -> crate::Result<f64> {
            let h_b = self.h_b(t)?;
            Ok(h_b - h_u)
        };
        let t_b = bisect(f, t_u, self.burned.t_high, "(152.10)")?;
        let sigma = t_b * self.w_u() / (t_u * self.w_b());
        Ok(BurnedState { t_b, sigma })
    }

    /// The AICC state (152.11): the T_v on [T_0, t_high] that makes
    /// u_B(T_v) = u_U(T_0), with u_X(T) = h_X(T) - R0 T/W_X per kg and the
    /// dust counted as gas moles, and P_max/p_0 = (T_v W_U)/(T_0 W_B)
    /// (SPEC-LIT §152.6).
    pub fn aicc(&self, t0: f64) -> crate::Result<AiccState> {
        let w_u = self.w_u();
        let w_b = self.w_b();
        let u_u0 = self.h_u(t0)? - R_GAS * t0 / w_u;
        let f = |t: f64| -> crate::Result<f64> {
            let u_b = self.h_b(t)? - R_GAS * t / w_b;
            Ok(u_b - u_u0)
        };
        let t_v = bisect(f, t0, self.burned.t_high, "(152.11)")?;
        let p_ratio = t_v * w_u / (t0 * w_b);
        Ok(AiccState { t_v, p_ratio })
    }

    /// The two pseudo-species as a [`ThermoSet`] at 1 bar, so the CHEMRAD
    /// machinery reads the gas unchanged (SPEC-LIT §152.4).
    pub fn thermo_set(&self) -> crate::Result<ThermoSet> {
        ThermoSet::from_species(
            vec![self.unburned.clone(), self.burned.clone()],
            P_BAR,
            "dust equivalent gas",
        )
    }
}

/// What [`build_pseudo`] yields: the record, its (record, Y_k) pairs and the
/// f64 moles.
type PseudoBuilt = (SpeciesThermo, Vec<(SpeciesThermo, f64)>, Vec<f64>);

/// One pseudo-species from its (component, exact moles) pairs (SPEC-LIT
/// §152.4): W = sum n_k W_k / sum n_k, Y_k = n_k W_k / sum n_k W_k, and the
/// NASA-7 coefficients b_j = W sum_k Y_k b_j^k / W_k on both intervals, each
/// mole number converted to f64 once. The record's range is the intersection
/// its components allow. Returns the record, the (record, Y_k) pairs and the
/// f64 moles.
fn build_pseudo(
    name: &str,
    pairs: &[(&SpeciesThermo, Rat)],
) -> crate::Result<PseudoBuilt> {
    let moles: Vec<f64> = pairs.iter().map(|(_, n)| n.value()).collect();
    let mut m_tot = 0.0;
    let mut n_tot = 0.0;
    for (k, (r, _)) in pairs.iter().enumerate() {
        m_tot += moles[k] * r.molar_mass;
        n_tot += moles[k];
    }
    if n_tot <= 0.0 || n_tot.is_nan() || !m_tot.is_finite() || m_tot <= 0.0 {
        return Err(Error::Config(format!(
            "the pseudo-species {name} is built from mole numbers with no positive total \
             mass or moles (SPEC-LIT §152.4)"
        )));
    }
    let w = m_tot / n_tot;
    let y: Vec<f64> = pairs
        .iter()
        .enumerate()
        .map(|(k, (r, _))| moles[k] * r.molar_mass / m_tot)
        .collect();
    let mut upper = [0.0f64; 7];
    let mut lower = [0.0f64; 7];
    for (k, (r, _)) in pairs.iter().enumerate() {
        for j in 0..7 {
            upper[j] += y[k] * r.upper[j] / r.molar_mass;
            lower[j] += y[k] * r.lower[j] / r.molar_mass;
        }
    }
    for v in upper.iter_mut() {
        *v *= w;
    }
    for v in lower.iter_mut() {
        *v *= w;
    }
    let t_low = pairs.iter().map(|(r, _)| r.t_low).fold(f64::NAN, f64::max);
    let t_high = pairs.iter().map(|(r, _)| r.t_high).fold(f64::NAN, f64::min);
    let record = SpeciesThermo {
        name: name.to_string(),
        elements: Vec::new(),
        molar_mass: w,
        t_low,
        t_mid: 1000.0,
        t_high,
        upper,
        lower,
        line: 0,
    };
    let comps: Vec<(SpeciesThermo, f64)> =
        pairs.iter().map(|(r, _)| (*r).clone()).zip(y).collect();
    Ok((record, comps, moles))
}

/// A bracketed root to full f64 precision: at most 300 halvings of [lo, hi],
/// stopping when the midpoint equals an end. Refused, with both end values,
/// when the function does not change sign across the bracket.
fn bisect(
    mut f: impl FnMut(f64) -> crate::Result<f64>,
    mut lo: f64,
    mut hi: f64,
    what: &str,
) -> crate::Result<f64> {
    let mut f_lo = f(lo)?;
    if f_lo == 0.0 {
        return Ok(lo);
    }
    let f_hi = f(hi)?;
    if f_hi == 0.0 {
        return Ok(hi);
    }
    if (f_lo > 0.0) == (f_hi > 0.0) {
        return Err(Error::Config(format!(
            "the root of {what} is not bracketed: the function is {f_lo} at {lo} K and \
             {f_hi} at {hi} K (SPEC-LIT §152.5)"
        )));
    }
    for _ in 0..300 {
        let mid = 0.5 * (lo + hi);
        if mid == lo || mid == hi {
            return Ok(mid);
        }
        let f_mid = f(mid)?;
        if f_mid == 0.0 {
            return Ok(mid);
        }
        if (f_mid > 0.0) == (f_lo > 0.0) {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

#[cfg(test)]
#[path = "tests_thermo.rs"]
mod tests;
