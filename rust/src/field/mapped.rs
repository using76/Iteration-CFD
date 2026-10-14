// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 120.10 - the recycled inflow: `mapped`/`mappedFixedValue`,
//! which copy the field from an offset plane inside the domain onto the
//! patch (optionally rescaled to a prescribed area mean), and
//! `turbulentInlet`, a reference value plus a seeded, deterministic
//! fluctuation with a set RMS and memory.
//!
//! Both are a §4 FIXED VALUE (`fr = 1`, `refGrad = 0`) whose `refValue` a
//! refresh kernel rewrites - `fldMapped` once per outer iteration
//! ([`crate::field_ops::update_mapped_scalar`]/`_vector`) and
//! `fldTurbulentInlet` once per time step
//! ([`crate::field_ops::advance_turbulent_inlet_scalar`]/`_vector`). The
//! functions here are their host twins, in the SAME operation order: the
//! (120.10c) sums run per face in face order in `Acc` (SPEC-LIT 118.1), the
//! (120.10g) update in f64, and each face value rounds once into `Scalar`.
//! The device contracts FMAs, so the two agree to round-off, not bitwise,
//! except where nothing is computed: without `setAverage` the face value IS
//! the cell value, bit for bit, and (120.10e)'s `eps` is integer arithmetic
//! and one product in both.
//!
//! The recycling idea - sample a plane downstream and re-introduce it at the
//! inlet - is Lund, Wu & Squires (1998), doi:10.1006/jcph.1998.5882; the
//! rescaling rule (120.10c) is DESIGN, ours. (120.10g) is Gillespie (1996)'s
//! exact Ornstein-Uhlenbeck update, doi:10.1103/PhysRevE.54.2084; (120.10e)'s
//! mixer is the SplitMix64 finaliser of Steele, Lea & Flood (2014),
//! doi:10.1145/2660193.2660195.
//!
//! No GPL-licensed source was consulted.

use crate::device::{DevBuf, Gpu};
use crate::error::Result;
use crate::types::DVec3;
use crate::{Acc, Label, Scalar, Vec3};

/// One `mapped`/`mappedFixedValue` patch (SPEC-LIT 120.10), built at setup.
///
/// `cells[i]` is the cell holding `x_f + offset` for the patch's face `i`
/// ((120.10a)). A scalar rule keeps its `average` in `.x`.
#[derive(Clone, Debug, PartialEq)]
pub struct MappedRule {
    /// This patch's faces in the flattened boundary.
    pub start: usize,
    pub size: usize,
    /// The sampling offset `d` of (120.10a), f64.
    pub offset: DVec3,
    /// `[size]` the sampled cell of each face.
    pub cells: Vec<Label>,
    /// (120.10c) on or off.
    pub set_average: bool,
    /// (120.10c)'s `a`; zero when `set_average` is off.
    pub average: Vec3,
}

/// One `turbulentInlet` patch (SPEC-LIT 120.10), built at setup.
#[derive(Clone, Debug, PartialEq)]
pub struct TurbulentInletRule {
    pub start: usize,
    pub size: usize,
    /// 1 on a scalar field, 3 on a vector field.
    pub ncomp: usize,
    /// `[size * ncomp]` `psi_ref` per face, component-fastest.
    pub reference: Vec<Scalar>,
    /// `fluctuationScale` `s` of (120.10f); a scalar rule uses `.x`.
    pub scale: Vec3,
    /// `alpha`, `0 < alpha <= 1`; `mu = 1 - alpha` in (120.10g).
    pub alpha: Scalar,
    /// The key's seed of (120.10e).
    pub seed: u64,
}

/// `sqrt(3)`, the factor that gives (120.10e)'s `eps` unit variance.
pub const SQRT_3: f64 = 1.7320508075688772;

/// (120.10a)-(120.10c): write the patch's `refValue` from the field's cell
/// values `psi` (`[n_cells * ncomp]`, component-fastest) into `ref_value`
/// (`[n_bf * ncomp]`). Only the faces `rule.start..rule.start + rule.size`
/// are written. Without `set_average` each face value is its sampled cell's
/// value, copied. With it, the sums `A` and `M` run per face in face order
/// in `Acc`, and each face value is `s psi* + (a - s m)` rounded once.
pub fn mapped_values(
    rule: &MappedRule,
    ncomp: usize,
    psi: &[Scalar],
    b_mag_sf: &[Scalar],
    ref_value: &mut [Scalar],
) {
    let avg = [rule.average.x as Acc, rule.average.y as Acc, rule.average.z as Acc];
    let mut area: Acc = 0.0;
    let mut sum = [0.0 as Acc; 3];
    if rule.set_average {
        for i in 0..rule.size {
            let s = b_mag_sf[rule.start + i] as Acc;
            let c = rule.cells[i] as usize;
            area += s;
            for k in 0..ncomp {
                sum[k] += s * psi[c * ncomp + k] as Acc;
            }
        }
    }
    if !rule.set_average || area <= 0.0 {
        // (120.10b): the cell value, unchanged.
        for i in 0..rule.size {
            let c = rule.cells[i] as usize;
            for k in 0..ncomp {
                ref_value[(rule.start + i) * ncomp + k] = psi[c * ncomp + k];
            }
        }
        return;
    }
    // (120.10c): the mean `m`, the factor `s` and the shift `a - s m`.
    let mut mbar = [0.0 as Acc; 3];
    let mut m2: Acc = 0.0;
    let mut a2: Acc = 0.0;
    for k in 0..ncomp {
        mbar[k] = sum[k] / area;
        m2 += mbar[k] * mbar[k];
        a2 += avg[k] * avg[k];
    }
    let (mm, am) = (m2.sqrt(), a2.sqrt());
    let fac: Acc = if am > 0.0 && mm >= 0.5 * am { am / mm } else { 1.0 };
    let mut shift = [0.0 as Acc; 3];
    for k in 0..ncomp {
        shift[k] = avg[k] - fac * mbar[k];
    }
    for i in 0..rule.size {
        let c = rule.cells[i] as usize;
        for k in 0..ncomp {
            let v = psi[c * ncomp + k] as Acc;
            ref_value[(rule.start + i) * ncomp + k] = (fac * v + shift[k]) as Scalar;
        }
    }
}

/// (120.10e): the unit-variance draw for step `n`, global boundary face `g`,
/// component `c` - the SplitMix64 finaliser of the key, its top 53 bits as
/// `u` in `[0, 1)`, and `sqrt(3) (2u - 1)`, uniform on `[-sqrt 3, sqrt 3)`.
#[must_use]
pub fn turbulent_eps(seed: u64, n: u64, g: u64, c: u64) -> f64 {
    let mut z = seed ^ (n << 32) ^ (3 * g + c);
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let u = (z >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0);
    SQRT_3 * (2.0 * u - 1.0)
}

/// (120.10f): `|psi_ref,f|` of face `i`, in f64 - the vector magnitude, or
/// the absolute value on a scalar field.
fn reference_magnitude(rule: &TurbulentInletRule, i: usize) -> f64 {
    let mut r2 = 0.0f64;
    for k in 0..rule.ncomp {
        let r = rule.reference[i * rule.ncomp + k] as f64;
        r2 += r * r;
    }
    r2.sqrt()
}

/// (120.10f)'s `s_c`, in f64.
fn scale_of(rule: &TurbulentInletRule, k: usize) -> f64 {
    match k {
        0 => rule.scale.x as f64,
        1 => rule.scale.y as f64,
        _ => rule.scale.z as f64,
    }
}

/// (120.10g)'s `x^0 = sigma eps^0` and (120.10h): the seed face values
/// `psi_ref + x^0`, written into `ref_value` (`[n_bf * ncomp]`).
pub fn turbulent_seed(rule: &TurbulentInletRule, ref_value: &mut [Scalar]) {
    for i in 0..rule.size {
        let g = (rule.start + i) as u64;
        let mag = reference_magnitude(rule, i);
        for k in 0..rule.ncomp {
            let r = rule.reference[i * rule.ncomp + k] as f64;
            let sigma = scale_of(rule, k) * mag;
            let x = sigma * turbulent_eps(rule.seed, 0, g, k as u64);
            ref_value[(rule.start + i) * rule.ncomp + k] = (r + x) as Scalar;
        }
    }
}

/// (120.10g)-(120.10h) for step `n >= 1`: `x^(n-1) = refValue - psi_ref`,
/// `x^n = mu x^(n-1) + sqrt(1 - mu^2) sigma eps^n`, the face value
/// `psi_ref + x^n` rounded once - the host twin of one `fldTurbulentInlet`
/// launch.
pub fn turbulent_advance(rule: &TurbulentInletRule, n: u64, ref_value: &mut [Scalar]) {
    let mu = 1.0 - rule.alpha as f64;
    let q = (1.0 - mu * mu).sqrt();
    for i in 0..rule.size {
        let g = (rule.start + i) as u64;
        let mag = reference_magnitude(rule, i);
        for k in 0..rule.ncomp {
            let j = (rule.start + i) * rule.ncomp + k;
            let r = rule.reference[i * rule.ncomp + k] as f64;
            let sigma = scale_of(rule, k) * mag;
            let x_old = ref_value[j] as f64 - r;
            let x = mu * x_old + q * sigma * turbulent_eps(rule.seed, n, g, k as u64);
            ref_value[j] = (r + x) as Scalar;
        }
    }
}

/// A [`MappedRule`] with its sampled cells on the device.
pub struct GpuMapped {
    pub rule: MappedRule,
    /// `[size]` [`MappedRule::cells`].
    pub cells: DevBuf<Label>,
}

/// A [`TurbulentInletRule`] with its reference and its step counter on the
/// device. `step[0]` is the last step drawn: 0 after the seed, and
/// `fldTurbulentInlet` draws `n = step[0] + 1` and stores it, so the
/// counter lives where a captured graph can advance it (SPEC-LIT 81).
pub struct GpuTurbulentInlet {
    pub rule: TurbulentInletRule,
    /// `[size * ncomp]` [`TurbulentInletRule::reference`].
    pub reference: DevBuf<Scalar>,
    /// `[1]` the step counter.
    pub step: DevBuf<Label>,
}

/// Upload the setup's mapped rules; an empty patch carries no buffer worth
/// launching on, but keeps its rule so the record is whole.
pub fn upload_mapped(gpu: &Gpu, rules: Vec<MappedRule>) -> Result<Vec<GpuMapped>> {
    let mut out = Vec::with_capacity(rules.len());
    for rule in rules {
        let cells = if rule.cells.is_empty() {
            gpu.upload(&[0 as Label])?
        } else {
            gpu.upload(&rule.cells)?
        };
        out.push(GpuMapped { rule, cells });
    }
    Ok(out)
}

/// Upload the setup's turbulentInlet rules with their counters at 0.
pub fn upload_turbulent(
    gpu: &Gpu,
    rules: Vec<TurbulentInletRule>,
) -> Result<Vec<GpuTurbulentInlet>> {
    let mut out = Vec::with_capacity(rules.len());
    for rule in rules {
        let reference = if rule.reference.is_empty() {
            gpu.upload(&[0.0 as Scalar])?
        } else {
            gpu.upload(&rule.reference)?
        };
        let step = gpu.upload(&[0 as Label])?;
        out.push(GpuTurbulentInlet { rule, reference, step });
    }
    Ok(out)
}
