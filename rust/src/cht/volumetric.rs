// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! A volumetric heat source that is not one number per region - §100.11
//! and §100.12. A case writes `q'''` as a number, a curve in `T` or a table
//! in `t`, over a region or over a box of it. This module holds what the
//! case reader lowered, puts it on the cells of the thermal mesh once, and
//! evaluates it per cell at a temperature field and a time: a number and a
//! table in `t` into the explicit part, a curve in `T` split Patankar's way
//! into `S_C + S_P T` with `S_P <= 0` (S100.12). Host arithmetic; nothing
//! here launches a kernel, and the drivers upload what it returns between
//! two corrections, never inside a captured region.
//!
//! Written from:
//!   S. V. Patankar, *Numerical Heat Transfer and Fluid Flow*, Hemisphere
//!     (1980), §4.2 - the linearisation `S = S_C + S_P T_P` and the rule
//!     `S_P <= 0`
//!   ofgpu `SPEC-LIT.md` §3.4, §18, §100.11, §100.12, §100.13 - the last
//!     the viscous dissipation function a Newtonian fluid's strain gives
//!
//! No GPL-licensed source was consulted.

use super::ThermalMesh;
use crate::error::{Error, Result};
use crate::properties::Property;
use crate::{Scalar, Tensor};

/// (S100.11): `q'''(t)`, W/m^3, piecewise linear in time on knots
/// `t_0 < t_1 < ... < t_n`, `t_0 >= 0`, with (S100.2)'s segment rule. A
/// time outside `[t_0, t_n]` is refused, never extrapolated.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeTable {
    pub t: Vec<Scalar>,
    pub v: Vec<Scalar>,
}

impl TimeTable {
    /// The knots, validated under `setting` - §100.11 row 3.
    pub fn new(setting: &str, knots: &[(Scalar, Scalar)]) -> Result<Self> {
        if knots.len() < 2 {
            return Err(Error::Config(format!(
                "{setting}: a table in t needs at least two knots, it has {} (SPEC-LIT 100.11)",
                knots.len()
            )));
        }
        for (i, &(t, v)) in knots.iter().enumerate() {
            if !(t.is_finite() && v.is_finite()) {
                return Err(Error::Config(format!(
                    "{setting}: knot {i} = [{t}, {v}] is not finite (SPEC-LIT 100.11)"
                )));
            }
            if i == 0 && t < 0.0 {
                return Err(Error::Config(format!(
                    "{setting}: the first time is {t} s; a table in t starts at or after t = 0 \
                     (SPEC-LIT 100.11)"
                )));
            }
            if i > 0 && !(t > knots[i - 1].0) {
                return Err(Error::Config(format!(
                    "{setting}: the times do not increase strictly at knot {i} ({} then {t} s) \
                     (SPEC-LIT 100.11)",
                    knots[i - 1].0
                )));
            }
        }
        Ok(Self {
            t: knots.iter().map(|k| k.0).collect(),
            v: knots.iter().map(|k| k.1).collect(),
        })
    }

    /// The source at `time`, s - refused outside the table (§100.11 row 7).
    pub fn value(&self, setting: &str, time: Scalar) -> Result<Scalar> {
        let (lo, hi) = (self.t[0], self.t[self.t.len() - 1]);
        if !(time >= lo && time <= hi) {
            return Err(Error::Config(format!(
                "{setting}: t = {time} s lies outside the table's [{lo}, {hi}] s; a table in t \
                 is not extrapolated (SPEC-LIT 100.11)"
            )));
        }
        let k = self.t.partition_point(|&x| x <= time);
        if k >= self.t.len() {
            return Ok(self.v[self.t.len() - 1]);
        }
        let i = k.saturating_sub(1);
        Ok(self.v[i] + (time - self.t[i]) / (self.t[k] - self.t[i]) * (self.v[k] - self.v[i]))
    }
}

/// What one source says `q'''` is - §100.11.
#[derive(Debug, Clone, PartialEq)]
pub enum SourceLaw {
    /// W/m^3.
    Number(Scalar),
    /// `q'''(T)`, split per cell by (S100.12).
    Temperature(Property),
    /// `q'''(t)`, (S100.11).
    Time(TimeTable),
}

impl SourceLaw {
    /// One phrase for the banner.
    pub fn describe(&self) -> String {
        match self {
            SourceLaw::Number(q) => format!("{q:e} W/m^3"),
            SourceLaw::Temperature(p) => format!("q'''(T), {}", p.describe()),
            SourceLaw::Time(tt) => format!(
                "q'''(t), a table of {} knots on [{}, {}] s",
                tt.t.len(),
                tt.t[0],
                tt.t[tt.t.len() - 1]
            ),
        }
    }
}

/// One lowered source that is not a region's number: a region's curve or
/// table, or one box - §100.11.
#[derive(Debug, Clone, PartialEq)]
pub struct LoweredSource {
    /// The JSON path every refusal names: `regions/<name>/source` or
    /// `regions/<name>/sourceBoxes/<i>/source`.
    pub path: String,
    /// The region, in case order.
    pub region: usize,
    /// The region's own cells the source covers, ascending; `None` is the
    /// whole region.
    pub cells: Option<Vec<usize>>,
    pub law: SourceLaw,
}

/// Every volumetric source of a case on the thermal mesh's cells - §100.11.
#[derive(Debug, Clone)]
pub struct CellSources {
    /// Each region's number and every box whose law is a number, per thermal
    /// cell, W/m^3.
    fixed: Vec<Scalar>,
    /// Every term that is not a number: its path, its law, its thermal cells.
    varying: Vec<(String, SourceLaw, Vec<usize>)>,
}

impl CellSources {
    /// `region_sources` is one number per region (`LoweredChtCase::sources`),
    /// `terms` the rest (`LoweredChtCase::volumetric`). A region's number is
    /// ASSIGNED to its cells and a number box ADDED, so a case with no box
    /// has exactly the array the drivers wrote before §100.11.
    pub fn build(tm: &ThermalMesh, region_sources: &[Scalar], terms: &[LoweredSource]) -> Result<Self> {
        let mut fixed = vec![0.0 as Scalar; tm.host.n_cells];
        for (block, s) in tm.regions.iter().zip(region_sources) {
            for c in block.cells() {
                fixed[c] = *s;
            }
        }
        let mut varying = Vec::new();
        for term in terms {
            let block = tm.regions.get(term.region).ok_or_else(|| {
                Error::Config(format!(
                    "{}: region {} is not a region of the thermal mesh (SPEC-LIT 100.11)",
                    term.path, term.region
                ))
            })?;
            let cells: Vec<usize> = match &term.cells {
                None => block.cells().collect(),
                Some(local) => {
                    let mut v = Vec::with_capacity(local.len());
                    for &i in local {
                        if i >= block.n_cells {
                            return Err(Error::Config(format!(
                                "{}: cell {i} is not a cell of its region, which has {} \
                                 (SPEC-LIT 100.11)",
                                term.path, block.n_cells
                            )));
                        }
                        v.push(block.cell_offset + i);
                    }
                    v
                }
            };
            match &term.law {
                SourceLaw::Number(q) => {
                    for &c in &cells {
                        fixed[c] += *q;
                    }
                }
                law => varying.push((term.path.clone(), law.clone(), cells)),
            }
        }
        Ok(Self { fixed, varying })
    }

    /// The numbers, per thermal cell, W/m^3.
    pub fn fixed(&self) -> &[Scalar] {
        &self.fixed
    }

    /// `true` when some source is a curve in `T`.
    pub fn in_temperature(&self) -> bool {
        self.varying.iter().any(|(_, l, _)| matches!(l, SourceLaw::Temperature(_)))
    }

    /// `true` when some source is a table in `t`.
    pub fn in_time(&self) -> bool {
        self.varying.iter().any(|(_, l, _)| matches!(l, SourceLaw::Time(_)))
    }

    /// (S100.12): the varying part alone, `(S_C, S_P)` per thermal cell,
    /// W/m^3 and W/(m^3 K) - a table in `t` at `time`, a curve in `T` split
    /// about `t_cells`. Refused by name (§100.11 rows 6-7): a curve evaluated
    /// outside its range, a positive `S_P`, a time outside its table, and a
    /// table in `t` on a run with no time.
    pub fn varying(&self, t_cells: Option<&[Scalar]>, time: Option<Scalar>) -> Result<(Vec<Scalar>, Vec<Scalar>)> {
        let n = self.fixed.len();
        let mut su = vec![0.0 as Scalar; n];
        let mut sp = vec![0.0 as Scalar; n];
        for (path, law, cells) in &self.varying {
            match law {
                SourceLaw::Number(_) => {}
                SourceLaw::Time(tt) => {
                    let Some(time) = time else {
                        return Err(Error::Config(format!(
                            "{path}: a table in t on a run with no time - a steady solve and \
                             the conjugate path have none (SPEC-LIT 100.11)"
                        )));
                    };
                    let q = tt.value(path, time)?;
                    for &c in cells {
                        su[c] += q;
                    }
                }
                SourceLaw::Temperature(p) => {
                    let Some(t) = t_cells else {
                        return Err(Error::Config(format!(
                            "{path}: a curve in T evaluated with no temperature field \
                             (SPEC-LIT 100.12)"
                        )));
                    };
                    for &c in cells {
                        let ts = t[c];
                        let s_p = p.slope(path, ts)?;
                        if s_p > 0.0 {
                            return Err(Error::Config(format!(
                                "{path}: at T = {ts} K the curve rises, S_P = dq/dT = {s_p:e} \
                                 W/(m^3 K) > 0; Patankar's split needs S_P <= 0, and a rising \
                                 source is refused, not lagged (SPEC-LIT 100.12)"
                            )));
                        }
                        su[c] += p.value(path, ts)? - s_p * ts;
                        sp[c] += s_p;
                    }
                }
            }
        }
        Ok((su, sp))
    }

    /// The whole explicit part - the numbers plus [`Self::varying`]'s `S_C` -
    /// and the implicit part (S100.12).
    pub fn evaluate(&self, t_cells: Option<&[Scalar]>, time: Option<Scalar>) -> Result<(Vec<Scalar>, Vec<Scalar>)> {
        let (mut su, sp) = self.varying(t_cells, time)?;
        for (a, f) in su.iter_mut().zip(&self.fixed) {
            *a += *f;
        }
        Ok((su, sp))
    }
}

/// (S100.12): `SUM_c (S_C + S_P T_c) V_c`, W - the power sources written as
/// `(su, sp)` deliver at the temperatures `t`.
pub fn delivered_power(su: &[Scalar], sp: &[Scalar], t: &[Scalar], v: &[Scalar]) -> Scalar {
    su.iter().zip(sp).zip(t).zip(v).map(|(((a, b), t), v)| (a + b * t) * v).sum()
}

/// (S100.13): `Phi = 2 mu |S - (1/3)(div u) I|^2` per cell, W/m^3, from the
/// velocity gradient and `mu` of each cell. `2 mu S:S - (2/3) mu (div u)^2`
/// written as a sum of squares, so it is non-negative by construction.
pub fn viscous_dissipation(grad: &[Tensor], mu: &[Scalar]) -> Vec<Scalar> {
    grad.iter()
        .zip(mu)
        .map(|(g, m)| {
            let (sxy, sxz, syz) = (0.5 * (g.xy + g.yx), 0.5 * (g.xz + g.zx), 0.5 * (g.yz + g.zy));
            let third = (g.xx + g.yy + g.zz) / 3.0;
            let (dx, dy, dz) = (g.xx - third, g.yy - third, g.zz - third);
            let s = dx * dx + dy * dy + dz * dz + 2.0 * (sxy * sxy + sxz * sxz + syz * syz);
            2.0 * *m * s
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SPEC-LIT §100.11: every knot to the bit, the midpoint the mean.
    #[test]
    fn a_time_table_returns_its_knots_and_interpolates_between_them() {
        let tt = TimeTable::new("s", &[(0.0, 0.0), (2.0, 4.0), (3.0, 1.0)]).unwrap();
        assert_eq!(tt.value("s", 0.0).unwrap(), 0.0);
        assert_eq!(tt.value("s", 1.0).unwrap(), 2.0);
        assert_eq!(tt.value("s", 2.0).unwrap(), 4.0);
        assert_eq!(tt.value("s", 2.5).unwrap(), 2.5);
        assert_eq!(tt.value("s", 3.0).unwrap(), 1.0);
    }

    /// SPEC-LIT §100.11 rows 3 and 7.
    #[test]
    fn a_time_table_is_refused_outside_its_range_and_for_bad_knots() {
        let tt = TimeTable::new("regions/die/source", &[(0.0, 1.0), (0.5, 1.0)]).unwrap();
        let e = tt.value("regions/die/source", 0.75).unwrap_err().to_string();
        println!("{e}");
        for w in ["regions/die/source", "t = 0.75 s", "[0, 0.5] s", "SPEC-LIT 100.11"] {
            assert!(e.contains(w), "{w:?} not in: {e}");
        }
        let cases: [(Vec<(Scalar, Scalar)>, &str); 4] = [
            (vec![(0.0, 1.0)], "two knots"),
            (vec![(0.0, 1.0), (0.0, 2.0)], "increase strictly"),
            (vec![(-1.0, 1.0), (1.0, 2.0)], "t = 0"),
            (vec![(0.0, 1.0), (1.0, Scalar::NAN)], "not finite"),
        ];
        for (knots, w) in cases {
            let e = TimeTable::new("x", &knots).unwrap_err().to_string();
            assert!(e.contains(w), "{w:?} not in: {e}");
        }
    }

    fn tensor(v: [Scalar; 9]) -> Tensor {
        Tensor { xx: v[0], xy: v[1], xz: v[2], yx: v[3], yy: v[4], yz: v[5], zx: v[6], zy: v[7], zz: v[8] }
    }

    /// SPEC-LIT §100.13: a simple shear dissipates `mu gamma^2`; a pure
    /// dilatation and a pure rotation dissipate nothing, to the bit.
    #[test]
    fn dissipation_is_mu_gamma_squared_in_shear_and_zero_in_dilatation_and_rotation() {
        let shear = tensor([0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let dilate = tensor([2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]);
        let rotate = tensor([0.0, 5.0, 0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let phi = viscous_dissipation(&[shear, dilate, rotate], &[2.0, 2.0, 2.0]);
        assert_eq!(phi, vec![18.0, 0.0, 0.0], "mu gamma^2 = 2 * 3^2, then zero twice");
    }

    /// SPEC-LIT §100.13: non-negative on any gradient.
    #[test]
    fn dissipation_is_non_negative_on_random_gradients() {
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 11) as Scalar / (1u64 << 53) as Scalar * 2.0 - 1.0
        };
        let grads: Vec<Tensor> = (0..4096)
            .map(|_| tensor([next(), next(), next(), next(), next(), next(), next(), next(), next()]))
            .collect();
        let phi = viscous_dissipation(&grads, &vec![1.0e-3; 4096]);
        assert!(phi.iter().all(|p| *p >= 0.0), "a negative Phi: {:?}", phi.iter().find(|p| **p < 0.0));
    }
}
