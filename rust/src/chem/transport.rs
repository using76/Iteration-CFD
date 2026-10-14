// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Mixture-averaged transport (SPEC-LIT §133.4), host side: the
//! pure-species viscosity and conductivity of NASA TM-4513's eq. (10)
//! fits (route (a), read from `reference/mcbride-gordon-reno1993/
//! table_IV.csv`), the Wilke mixture viscosity (133.8) with (133.10), the
//! mixture conductivity (133.9) with (133.10) and (133.11), and the
//! Lewis-number route - `lambda = mu cp / Pr`, `D_m = lambda /
//! (rho cp Le_m)` - the one v1 route that needs no diffusion data. Route
//! (b), the Chapman-Enskog binary diffusion of (133.7) with the Neufeld,
//! Janzen & Aziz collision integrals, is NAMED and NOT BUILT: every entry
//! point refuses by name ([`TransportSet::binary_diffusion`],
//! [`TransportSet::mixture_diffusivities`]). The device twins of
//! [`TransportSet::mixture_ty`] and the Lewis route live in
//! [`crate::chem::device`]. Everything here is f64 in every build.
//!
//! No GPL-licensed source was consulted.

use crate::error::{Error, Result};

/// Pa s per micropoise: table_IV.csv's viscosity fits are micropoise.
pub const MICROPOISE: f64 = 1.0e-7;

/// W/(m K) per microwatt/(cm K): table_IV.csv's conductivity fits are
/// microwatt/(cm K).
pub const MICROWATT_PER_CM_K: f64 = 1.0e-4;

/// The lowest accepted temperature, K (SPEC-LIT 133.7 T1, the user's
/// decision of 2026-10-14). Between this floor and the first fit's `t_lo`
/// (TM-4513 publishes from 300 K) the first fit is evaluated outside its
/// published range, an extrapolation; below it a cell is refused. The
/// device twin is `CHEM_TRANSPORT_FLOOR_K` in `cuda/chem.cu`.
pub const TRANSPORT_FLOOR_K: f64 = 200.0;

/// One TM-4513 eq. (10) fit over one temperature interval: `ln q = a ln T
/// + b/T + c/T^2 + d`, with `q` the viscosity in micropoise or the
/// conductivity in microwatt/(cm K).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransportFit {
    pub t_lo: f64,
    pub t_hi: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

impl TransportFit {
    /// Eq. (10) itself: the natural log of the property in its table
    /// unit. No range check - the callers pick the interval first.
    pub fn ln_value(&self, t: f64) -> f64 {
        self.a * t.ln() + self.b / t + self.c / (t * t) + self.d
    }
}

/// The transport record of one species: its TM-4513 eq. (10) fits, the
/// viscosity intervals and the conductivity intervals, each contiguous.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeciesTransport {
    pub name: String,
    pub viscosity: Vec<TransportFit>,
    pub conductivity: Vec<TransportFit>,
}

impl SpeciesTransport {
    /// A record, checked: a name; one to three fits on each property, every
    /// number finite, every interval `0 < t_lo < t_hi`, neighbouring
    /// intervals contiguous. The message names the species and the property.
    pub fn new(
        name: &str,
        viscosity: Vec<TransportFit>,
        conductivity: Vec<TransportFit>,
    ) -> Result<Self> {
        if name.is_empty() {
            return Err(Error::Config(
                "a species transport record needs a name".to_string(),
            ));
        }
        check_fits(name, &viscosity, "viscosity")?;
        check_fits(name, &conductivity, "conductivity")?;
        Ok(Self {
            name: name.to_string(),
            viscosity,
            conductivity,
        })
    }

    /// The species' viscosity at T (eq. (10), Pa s), on the FIRST fit with
    /// `t_lo <= T <= t_hi`, or on the first fit, extrapolated, for
    /// [`TRANSPORT_FLOOR_K`] `<= T <` its `t_lo`.
    pub fn viscosity_at(&self, t: f64) -> Result<f64> {
        self.value_at(t, &self.viscosity, "viscosity", MICROPOISE)
    }

    /// The species' conductivity at T (eq. (10), W/(m K)), on the FIRST fit
    /// with `t_lo <= T <= t_hi`, or on the first fit, extrapolated, for
    /// [`TRANSPORT_FLOOR_K`] `<= T <` its `t_lo`.
    pub fn conductivity_at(&self, t: f64) -> Result<f64> {
        self.value_at(t, &self.conductivity, "conductivity", MICROWATT_PER_CM_K)
    }

    /// One property at T: the interval choice and the two refusals, then
    /// eq. (10) and the unit conversion.
    fn value_at(
        &self,
        t: f64,
        fits: &[TransportFit],
        property: &str,
        scale: f64,
    ) -> Result<f64> {
        if !t.is_finite() || t <= 0.0 {
            return Err(Error::Config(format!(
                "species {}: the {} is asked for T = {t} K, which is not a finite \
                 temperature above zero",
                self.name, property
            )));
        }
        let pick = fits
            .iter()
            .find(|f| f.t_lo <= t && t <= f.t_hi)
            .or_else(|| (TRANSPORT_FLOOR_K <= t && t < fits[0].t_lo).then_some(&fits[0]));
        let Some(f) = pick else {
            return Err(Error::Config(format!(
                "species {}: T = {t} K is outside the {} fits' accepted range [{:e}, {:e}] K \
                 of TM-4513 eq. (10) (published from {:e} K, extrapolated down to the \
                 {:e} K floor)",
                self.name,
                property,
                TRANSPORT_FLOOR_K.min(fits[0].t_lo),
                fits[fits.len() - 1].t_hi,
                fits[0].t_lo,
                TRANSPORT_FLOOR_K
            )));
        };
        Ok(f.ln_value(t).exp() * scale)
    }
}

/// The checks [`SpeciesTransport::new`] runs on one property's fit list;
/// every refusal names the species and the property.
fn check_fits(name: &str, fits: &[TransportFit], property: &str) -> Result<()> {
    if fits.is_empty() || fits.len() > 3 {
        return Err(Error::Config(format!(
            "species {name}: the {property} record holds {} fit(s), and TM-4513's eq. (10) \
             records are one to three temperature intervals",
            fits.len()
        )));
    }
    for f in fits {
        if [f.t_lo, f.t_hi, f.a, f.b, f.c, f.d]
            .iter()
            .any(|v| !v.is_finite())
        {
            return Err(Error::Config(format!(
                "species {name}: a {property} fit carries a number that is not finite \
                 (t_lo = {:e}, t_hi = {:e}, a = {:e}, b = {:e}, c = {:e}, d = {:e})",
                f.t_lo, f.t_hi, f.a, f.b, f.c, f.d
            )));
        }
        if f.t_lo <= 0.0 || f.t_hi <= f.t_lo {
            return Err(Error::Config(format!(
                "species {name}: a {property} fit covers [{:e}, {:e}] K; an eq. (10) \
                 interval needs t_lo above zero and t_hi above t_lo",
                f.t_lo, f.t_hi
            )));
        }
    }
    for pair in fits.windows(2) {
        if pair[0].t_hi != pair[1].t_lo {
            return Err(Error::Config(format!(
                "species {name}: the {property} fits are not contiguous - one interval \
                 ends at {:e} K and the next begins at {:e} K",
                pair[0].t_hi, pair[1].t_lo
            )));
        }
    }
    Ok(())
}

/// Reads `table_IV.csv`'s format: `#` comment lines, the header row
/// `species,name,property,t_lo,t_hi,a,b,c,d`, then four rows per species -
/// the `V` (viscosity) rows and the `C` (conductivity) rows on each
/// interval - the numbers in the record's own E15.8 form. Records come
/// back in first-appearance order, each one's fits in file order, through
/// [`SpeciesTransport::new`]. Err ([`Error::Parse`] naming the 1-based
/// line): a row without 9 fields, an unparsable number, a property that is
/// not `V` or `C`, a missing header row.
pub fn read_table_iv_csv(text: &str, source: &str) -> Result<Vec<SpeciesTransport>> {
    const HEADER: &str = "species,name,property,t_lo,t_hi,a,b,c,d";
    let mut rows: Vec<(String, Vec<TransportFit>, Vec<TransportFit>)> = Vec::new();
    let mut header = false;
    for (no, raw) in text.lines().enumerate() {
        let line_no = no + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if !header {
            if line != HEADER {
                return Err(Error::Parse {
                    path: source.to_string(),
                    msg: format!("line {line_no}: expected the header row `{HEADER}`"),
                });
            }
            header = true;
            continue;
        }
        let c: Vec<&str> = line.split(',').collect();
        if c.len() != 9 {
            return Err(Error::Parse {
                path: source.to_string(),
                msg: format!(
                    "line {line_no}: a table_IV data row has 9 comma-separated fields, \
                     this one has {}",
                    c.len()
                ),
            });
        }
        let num = |s: &str| -> Result<f64> {
            s.parse::<f64>().map_err(|_| Error::Parse {
                path: source.to_string(),
                msg: format!("line {line_no}: `{s}` is not a number"),
            })
        };
        let fit = TransportFit {
            t_lo: num(c[3])?,
            t_hi: num(c[4])?,
            a: num(c[5])?,
            b: num(c[6])?,
            c: num(c[7])?,
            d: num(c[8])?,
        };
        push_row(&mut rows, c[1], c[2], fit, line_no, source)?;
    }
    if !header {
        return Err(Error::Parse {
            path: source.to_string(),
            msg: "line 1: the file holds no `species,name,property,t_lo,t_hi,a,b,c,d` \
                  header row"
                .to_string(),
        });
    }
    rows.into_iter()
        .map(|(name, viscosity, conductivity)| {
            SpeciesTransport::new(&name, viscosity, conductivity)
        })
        .collect()
}

/// One data row onto its record, created on the species' first appearance;
/// the property must be `V` or `C`.
fn push_row(
    rows: &mut Vec<(String, Vec<TransportFit>, Vec<TransportFit>)>,
    name: &str,
    property: &str,
    fit: TransportFit,
    line_no: usize,
    source: &str,
) -> Result<()> {
    match property {
        "V" => match rows.iter_mut().find(|(n, ..)| n.as_str() == name) {
            Some((_, v, _)) => v.push(fit),
            None => rows.push((name.to_string(), vec![fit], Vec::new())),
        },
        "C" => match rows.iter_mut().find(|(n, ..)| n.as_str() == name) {
            Some((_, _, w)) => w.push(fit),
            None => rows.push((name.to_string(), Vec::new(), vec![fit])),
        },
        other => {
            return Err(Error::Parse {
                path: source.to_string(),
                msg: format!(
                    "line {line_no}: the property is `{other}`, and table_IV's rows are \
                     V (viscosity) or C (conductivity)"
                ),
            });
        }
    }
    Ok(())
}

/// (133.10), the Wilke viscosity's pair coefficient: `phi_ij = (1/4) [1 +
/// (mu_i/mu_j)^(1/2) (W_j/W_i)^(1/4)]^2 (2 W_j/(W_i+W_j))^(1/2)`, computed
/// exactly in this order.
pub fn wilke_phi(mu_i: f64, mu_j: f64, w_i: f64, w_j: f64) -> f64 {
    let r = (mu_i / mu_j).sqrt() * (w_j / w_i).powf(0.25);
    let a = 1.0 + r;
    0.25 * a * a * (2.0 * w_j / (w_i + w_j)).sqrt()
}

/// (133.11): `psi_ij = phi_ij [1 + 2.41 (W_i-W_j)(W_i-0.142 W_j)/(W_i+W_j)^2]`.
/// (133.9)'s psi is [`wilke_phi`] on the *viscosities* times this
/// correction, as the equation reads it.
pub fn conductivity_psi(phi: f64, w_i: f64, w_j: f64) -> f64 {
    let m = (w_i - w_j) * (w_i - 0.142 * w_j) / ((w_i + w_j) * (w_i + w_j));
    phi * (1.0 + 2.41 * m)
}

/// The mixture transport of one state: [`TransportSet::mixture_ty`]'s output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MixtureTransport {
    pub viscosity: f64,
    pub conductivity: f64,
}

/// The transport records of a whole thermo set: one record per species, in
/// thermo order, with the species' molar masses (kg/mol) beside them. The
/// host side of SPEC-LIT §133.4's route (a); the device flattening is
/// [`crate::chem::device::flatten_transport`].
#[derive(Clone, Debug)]
pub struct TransportSet {
    species: Vec<SpeciesTransport>,
    molar_mass: Vec<f64>,
    source: String,
}

impl TransportSet {
    /// One record per thermo species, matched by exact name, in thermo
    /// order, with W from [`crate::chem::SpeciesThermo::molar_mass`]. Err
    /// ([`Error::Config`]): a thermo species with no record, or two
    /// records with one name; the message names the species.
    pub fn for_thermo(
        thermo: &crate::chem::ThermoSet,
        records: &[SpeciesTransport],
    ) -> Result<Self> {
        let mut seen = std::collections::HashSet::new();
        for r in records {
            if !seen.insert(r.name.as_str()) {
                return Err(Error::Config(format!(
                    "the transport records hold two records named {}; a species has one \
                     record (SPEC-LIT §133.4)",
                    r.name
                )));
            }
        }
        let mut species = Vec::with_capacity(thermo.species().len());
        let mut molar_mass = Vec::with_capacity(thermo.species().len());
        for sp in thermo.species() {
            let Some(r) = records.iter().find(|r| r.name == sp.name) else {
                return Err(Error::Config(format!(
                    "the thermo set read from {} carries species {}, and the transport \
                     records hold no record for it (SPEC-LIT §133.4)",
                    thermo.source(),
                    sp.name
                )));
            };
            species.push(r.clone());
            molar_mass.push(sp.molar_mass);
        }
        Ok(Self {
            species,
            molar_mass,
            source: thermo.source().to_string(),
        })
    }

    /// The records, in thermo order.
    pub fn species(&self) -> &[SpeciesTransport] {
        &self.species
    }

    /// The species' molar masses, kg/mol, in thermo order.
    pub fn molar_masses(&self) -> &[f64] {
        &self.molar_mass
    }

    /// The mixture's mass fractions against the set, checked like
    /// [`crate::chem::ThermoSet::check_y`]: one value per species, every
    /// one finite and at or above zero, and a positive sum of Y/W.
    fn check_y(&self, y: &[f64]) -> Result<f64> {
        if y.len() != self.species.len() {
            return Err(Error::Config(format!(
                "the mixture is given {} mass fraction value(s) but the transport set \
                 read from {} carries {} species",
                y.len(),
                self.source,
                self.species.len()
            )));
        }
        for (k, sp) in self.species.iter().enumerate() {
            let v = y[k];
            if !v.is_finite() || v < 0.0 {
                return Err(Error::Config(format!(
                    "species {}: the mass fraction {} is not a finite number at or \
                     above zero",
                    sp.name, v
                )));
            }
        }
        let mut sum = 0.0;
        for k in 0..self.species.len() {
            sum += y[k] / self.molar_mass[k];
        }
        if sum.is_nan() || sum <= 0.0 {
            return Err(Error::Config(
                "all mass fractions are zero; a mixture needs a positive sum of Y/W"
                    .to_string(),
            ));
        }
        Ok(sum)
    }

    /// The mole fractions, exactly as [`crate::chem::ThermoSet::mole_fractions`]
    /// computes them: `s = Σ y_k / W_k`, `w = 1/s`, `x_k = y_k w / W_k`.
    pub fn mole_fractions(&self, y: &[f64]) -> Result<Vec<f64>> {
        let sum = self.check_y(y)?;
        let w = 1.0 / sum;
        Ok((0..self.species.len())
            .map(|k| y[k] * w / self.molar_mass[k])
            .collect())
    }

    /// (133.8) and (133.9) at T: the Wilke mixture viscosity and the
    /// mixture conductivity of the present species (`y_k != 0`) from the
    /// records' eq. (10) fits, whose range every present species' T must
    /// sit inside; a `y_k == 0` species is not evaluated, like
    /// [`crate::chem::ThermoSet::cp_mass`]. The sums run exactly as
    /// §133.4 states them: outer `i` in index order, `den = x_i`, then
    /// `j` in index order with `j != i` and `y_j != 0`, `den += x_j
    /// phi_ij` - the conductivity's psi (133.11) on the same viscosity
    /// phi (133.10) - then `acc += x_i * prop_i / den`.
    pub fn mixture_ty(&self, t: f64, y: &[f64]) -> Result<MixtureTransport> {
        if !t.is_finite() || t <= 0.0 {
            return Err(Error::Config(format!(
                "mixture_ty needs a temperature that is finite and above zero, given \
                 T = {t} K"
            )));
        }
        let sum = self.check_y(y)?;
        let w = 1.0 / sum;
        let k = self.species.len();
        let mut x = vec![0.0; k];
        for i in 0..k {
            x[i] = y[i] * w / self.molar_mass[i];
        }
        let mut mu = vec![0.0; k];
        let mut lam = vec![0.0; k];
        for i in 0..k {
            if y[i] != 0.0 {
                mu[i] = self.species[i].viscosity_at(t)?;
                lam[i] = self.species[i].conductivity_at(t)?;
            }
        }
        let mut viscosity = 0.0;
        for i in 0..k {
            if y[i] == 0.0 {
                continue;
            }
            let mut den = x[i];
            for j in 0..k {
                if j == i || y[j] == 0.0 {
                    continue;
                }
                den += x[j] * wilke_phi(mu[i], mu[j], self.molar_mass[i], self.molar_mass[j]);
            }
            viscosity += x[i] * mu[i] / den;
        }
        let mut conductivity = 0.0;
        for i in 0..k {
            if y[i] == 0.0 {
                continue;
            }
            let mut den = x[i];
            for j in 0..k {
                if j == i || y[j] == 0.0 {
                    continue;
                }
                let phi = wilke_phi(mu[i], mu[j], self.molar_mass[i], self.molar_mass[j]);
                den += x[j]
                    * conductivity_psi(phi, self.molar_mass[i], self.molar_mass[j]);
            }
            conductivity += x[i] * lam[i] / den;
        }
        Ok(MixtureTransport {
            viscosity,
            conductivity,
        })
    }

    /// Route (b), (133.7): NAMED and NOT BUILT. The Neufeld, Janzen & Aziz
    /// (1972) collision-integral fit was not read, so no `D_kj` exists to
    /// return; the Lewis route is the available alternative. Always
    /// [`Error::Refused`].
    pub fn binary_diffusion(&self, i: usize, j: usize, t: f64, p: f64) -> Result<f64> {
        let _ = (i, j, t, p);
        Err(refused_route_b("the binary diffusion D_kj"))
    }

    /// (133.12): NAMED and NOT BUILT with route (b), which its `D_mj`
    /// waits on. Always [`Error::Refused`].
    pub fn mixture_diffusivities(&self, t: f64, p: f64, y: &[f64]) -> Result<Vec<f64>> {
        let _ = (t, p, y);
        Err(refused_route_b("the mixture-averaged diffusivity D_m of (133.12)"))
    }
}

/// The refusal both route-(b) entry points return: the coefficients were
/// not read, so nothing is built from them (SPEC-LIT 133.4).
fn refused_route_b(what: &str) -> Error {
    Error::Refused(format!(
        "{what} is route (b) and is NAMED and NOT BUILT: the Chapman-Enskog collision \
         integrals of Neufeld, Janzen & Aziz (1972) were not read, so no D_kj of (133.7) \
         exists to build it from (SPEC-LIT 133.4). The Lewis route (`Lewis`) is the \
         available alternative: lambda = mu cp / Pr, D_m = lambda / (rho cp Le_m)."
    ))
}

/// The stated constants of the Lewis-number route (SPEC-LIT §133.4): the
/// mixture's Prandtl number and one Lewis number per species. `pr` and
/// every `le` must be finite and above zero.
#[derive(Clone, Debug, PartialEq)]
pub struct Lewis {
    pub pr: f64,
    pub le: Vec<f64>,
}

impl Lewis {
    /// A set of Lewis-route constants, checked; the message names the
    /// offending species index.
    pub fn new(pr: f64, le: Vec<f64>) -> Result<Self> {
        if !pr.is_finite() || pr <= 0.0 {
            return Err(Error::Config(format!(
                "the Lewis route needs a Prandtl number that is finite and above zero, \
                 given Pr = {pr}"
            )));
        }
        if le.is_empty() {
            return Err(Error::Config(
                "the Lewis route needs one Lewis number per species; the list is empty"
                    .to_string(),
            ));
        }
        for (k, v) in le.iter().enumerate() {
            if !v.is_finite() || *v <= 0.0 {
                return Err(Error::Config(format!(
                    "the Lewis route needs species {k}'s Lewis number to be finite and \
                     above zero, given Le = {v}"
                )));
            }
        }
        Ok(Self { pr, le })
    }
}

/// The Lewis route's conductivity: `lambda = mu cp / Pr`, exactly
/// `(mu * cp) / pr`.
pub fn prandtl_conductivity(mu: f64, cp: f64, pr: f64) -> f64 {
    (mu * cp) / pr
}

/// The Lewis route's mixture diffusivity: `D_m = lambda / (rho cp Le)`,
/// exactly `lambda / ((rho * cp) * le)`.
pub fn lewis_diffusivity(lambda: f64, rho: f64, cp: f64, le: f64) -> f64 {
    lambda / ((rho * cp) * le)
}
