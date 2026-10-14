// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 120.14 - the VOF family: `variableHeightFlowRate`,
//! `variableHeightFlowRateInletVelocity`, `outletPhaseMeanVelocity` and
//! `phaseHydrostaticPressure`. The four names are taken from the user guides
//! of SPEC-LIT 120 by name only; every equation here is ours (*DESIGN*),
//! built on the phase fraction of §20.1, the mixture density of §20.3 and the
//! `p_rgh` decomposition of §20.5 and (120.7a).
//!
//! A [`VofFamilyRule`] is one patch's slice of the four names: the kind and
//! its constant entries, the patch's face range in the flattened boundary,
//! and the f64 face data the setup read from `geom64` (SPEC-LIT 118.2). The
//! four functions below rewrite the §4 triple of the patch's faces from the
//! CURRENT phase fraction, on the host, in the operation order SPEC-LIT
//! 120.14 writes: every input widened once to f64, the patch sums in face
//! order, every result rounded once into `Scalar`. `Vof::refresh_vof_family`
//! calls them once per time step, before the boundary values are
//! evaluated; nothing else does.
//!
//! No GPL-licensed source was consulted.

use crate::types::DVec3;
use crate::{Scalar, Vec3};

/// Which of the four names one patch carries, with its constant entries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VofFamilyKind {
    /// (120.14a) on the phase fraction: `lowerBound`, `upperBound`, each
    /// rounded once into `Scalar` at setup, `0 <= lower <= upper <= 1`.
    VariableHeightFlowRate { lower: Scalar, upper: Scalar },
    /// (120.14b) on the velocity: `flowRate`, the LIQUID volumetric rate in
    /// m^3/s, into the domain.
    VariableHeightFlowRateInletVelocity { flow_rate: f64 },
    /// (120.14c) on the velocity: `Umean`, the phase-weighted mean outward
    /// normal velocity in m/s.
    OutletPhaseMeanVelocity { u_mean: f64 },
    /// (120.14d) on `p_rgh`: the phase's density `rho`, `pRefValue` and
    /// `pRefPoint` (a world position).
    PhaseHydrostaticPressure { rho: f64, p_ref: f64, x_ref: DVec3 },
}

/// One patch's VOF-family rule (SPEC-LIT 120.14), built by `field_setup` and
/// held by the field (`GpuScalarField::vof_rules`, `GpuVectorField::vof_rules`).
///
/// The per-face arrays are indexed by the face's position in the patch,
/// `0..size`; the boundary arrays the functions write are indexed by
/// flattened boundary face, `start + i`.
#[derive(Clone, Debug, PartialEq)]
pub struct VofFamilyRule {
    pub kind: VofFamilyKind,
    pub patch: String,
    /// This patch's faces in the flattened boundary.
    pub start: usize,
    pub size: usize,
    /// `[size]` the cell behind each face (`b_face_cells`).
    pub cells: Vec<usize>,
    /// `[size]` unit outward normals, normalised in f64 from `geom64`.
    pub normal: Vec<DVec3>,
    /// `[size]` face areas `|S_f|`, f64 from `geom64`.
    pub mag_sf: Vec<f64>,
    /// `[size]` face centres in WORLD coordinates, for (120.14d)'s
    /// `x_f - x_ref`.
    pub x_world: Vec<DVec3>,
    /// `[size]` face centres as STORED (local under `single`, SPEC-LIT
    /// 118.7), for (120.14d)'s `g . (x_f - O)` - the solver's own body force
    /// reads the stored centres, as (120.7a) records.
    pub x_local: Vec<DVec3>,
}

/// `a_f = clamp(alpha_b, 0, 1)` widened to f64 - the face phase fraction
/// (120.14b)-(120.14d) weight by. A NaN stays NaN.
#[inline]
pub fn face_fraction(alpha_b: Scalar) -> f64 {
    let a = alpha_b as f64;
    if a < 0.0 {
        0.0
    } else if a > 1.0 {
        1.0
    } else {
        a
    }
}

/// (120.14a): `alpha_P > upper` fixes the face at `upper`, `alpha_P < lower`
/// at `lower`, and anything between is zero-gradient - so the evaluated face
/// value is `clamp(alpha_P, lower, upper)` exactly (`fr` is exactly 0 or 1).
/// `alpha` is the cell field; only the rule's faces are written.
pub fn variable_height_alpha(
    rule: &VofFamilyRule,
    alpha: &[Scalar],
    fr: &mut [Scalar],
    ref_value: &mut [Scalar],
    ref_grad: &mut [Scalar],
) {
    let VofFamilyKind::VariableHeightFlowRate { lower, upper } = rule.kind else {
        return;
    };
    for i in 0..rule.size {
        let f = rule.start + i;
        let a = alpha[rule.cells[i]];
        let (w, r) = if a > upper {
            (1.0, upper)
        } else if a < lower {
            (1.0, lower)
        } else {
            (0.0, 0.0)
        };
        fr[f] = w;
        ref_value[f] = r;
        ref_grad[f] = 0.0;
    }
}

/// (120.14b): `D = sum_f a_f^2 |S_f|` in face order, then per face
/// `s_f = (a_f Q)/D` and `U_f = -(n_f s_f)`, componentwise, rounded once;
/// `fr = 1`. A patch with `D = 0` (no liquid on any face) writes `U_f = 0`.
/// The liquid flux it carries is `-sum_f a_f (U_f . S_f) = Q`.
pub fn variable_height_inlet_velocity(
    rule: &VofFamilyRule,
    alpha_b: &[Scalar],
    fr: &mut [Scalar],
    ref_value: &mut [Vec3],
    ref_grad: &mut [Vec3],
) {
    let VofFamilyKind::VariableHeightFlowRateInletVelocity { flow_rate: q } = rule.kind else {
        return;
    };
    let mut d = 0.0f64;
    for i in 0..rule.size {
        let a = face_fraction(alpha_b[rule.start + i]);
        d += (a * a) * rule.mag_sf[i];
    }
    for i in 0..rule.size {
        let f = rule.start + i;
        let a = face_fraction(alpha_b[f]);
        let s = if d > 0.0 { (a * q) / d } else { 0.0 };
        let n = rule.normal[i];
        fr[f] = 1.0;
        ref_value[f] = Vec3::new(
            (-(n.x * s)) as Scalar,
            (-(n.y * s)) as Scalar,
            (-(n.z * s)) as Scalar,
        );
        ref_grad[f] = Vec3::ZERO;
    }
}

/// (120.14c): with `un_f = U_P . n_f = U_x n_x + U_y n_y + U_z n_z`,
/// `A = sum_f a_f |S_f|` and `M = sum_f (a_f un_f) |S_f|` in face order,
/// `c = Umean - M/A`, and per face `U_f = U_P + c n_f`, componentwise,
/// rounded once; `fr = 1`. A patch with `A = 0` writes `U_f = U_P`
/// (`c = 0`). The phase-weighted mean outward velocity is then
/// `sum_f a_f (U_f . n_f) |S_f| / A = Umean`. `u` is the cell field.
pub fn outlet_phase_mean_velocity(
    rule: &VofFamilyRule,
    alpha_b: &[Scalar],
    u: &[Vec3],
    fr: &mut [Scalar],
    ref_value: &mut [Vec3],
    ref_grad: &mut [Vec3],
) {
    let VofFamilyKind::OutletPhaseMeanVelocity { u_mean } = rule.kind else {
        return;
    };
    let un = |i: usize| -> f64 {
        let p = u[rule.cells[i]];
        let n = rule.normal[i];
        p.x as f64 * n.x + p.y as f64 * n.y + p.z as f64 * n.z
    };
    let mut area = 0.0f64;
    let mut mean = 0.0f64;
    for i in 0..rule.size {
        let a = face_fraction(alpha_b[rule.start + i]);
        area += a * rule.mag_sf[i];
        mean += (a * un(i)) * rule.mag_sf[i];
    }
    let c = if area > 0.0 { u_mean - mean / area } else { 0.0 };
    for i in 0..rule.size {
        let f = rule.start + i;
        let p = u[rule.cells[i]];
        let n = rule.normal[i];
        fr[f] = 1.0;
        ref_value[f] = Vec3::new(
            (p.x as f64 + c * n.x) as Scalar,
            (p.y as f64 + c * n.y) as Scalar,
            (p.z as f64 + c * n.z) as Scalar,
        );
        ref_grad[f] = Vec3::ZERO;
    }
}

/// (120.14d): `fr = a_f` and
/// `r_f = (p_ref + rho g.(x_f - x_ref)) - rho_f g.(x_f - O)`, with
/// `g . d = g_x d_x + g_y d_y + g_z d_z`, `x_f - x_ref` in world coordinates
/// and `x_f - O` the stored centre, `rho_f` the face's mixture density;
/// `refGrad = 0`. In the phase (`a_f = 1`, `rho_f = rho`) the face holds the
/// hydrostatic `p = p_ref + rho g.(x_f - x_ref)` written in `p_rgh`; out of
/// it (`a_f = 0`) the face is zero-gradient.
pub fn phase_hydrostatic_pressure(
    rule: &VofFamilyRule,
    g: Vec3,
    alpha_b: &[Scalar],
    rho_b: &[Scalar],
    fr: &mut [Scalar],
    ref_value: &mut [Scalar],
    ref_grad: &mut [Scalar],
) {
    let VofFamilyKind::PhaseHydrostaticPressure { rho, p_ref, x_ref } = rule.kind else {
        return;
    };
    let (gx, gy, gz) = (g.x as f64, g.y as f64, g.z as f64);
    for i in 0..rule.size {
        let f = rule.start + i;
        let a = face_fraction(alpha_b[f]);
        let xw = rule.x_world[i];
        let gdr = gx * (xw.x - x_ref.x) + gy * (xw.y - x_ref.y) + gz * (xw.z - x_ref.z);
        let xl = rule.x_local[i];
        let gh = gx * xl.x + gy * xl.y + gz * xl.z;
        let r = (p_ref + rho * gdr) - rho_b[f] as f64 * gh;
        fr[f] = a as Scalar;
        ref_value[f] = r as Scalar;
        ref_grad[f] = 0.0;
    }
}
