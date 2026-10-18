// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The simple compressible thruster of SPEC-LIT 161.10 and the
//! momentum-equivalent plume the low-Mach solver carries for it, 161.11.
//!
//! Written from:
//!   ofgpu SPEC-LIT.md 161.10 (the nozzle: one-dimensional isentropic flow of
//!     a perfect gas from a regulated chamber, ideally expanded or at a given
//!     area ratio, and the throttle)
//!   ofgpu SPEC-LIT.md 161.11 (the notional-nozzle plume on the meshed exit
//!     patch, its Mach-area refusal and its turbulence inflow)
//!   J. D. Anderson, *Modern Compressible Flow*, 3rd ed. (McGraw-Hill, 2003),
//!     ch. 3 and 5 - the isentropic relations T0/T, p0/p, the area-Mach
//!     relation and the thrust of a nozzle F = mdot U_e + (p_e - p_a) A_e
//!   G. P. Sutton, O. Biblarz, *Rocket Propulsion Elements*, 8th ed. (Wiley,
//!     2010), ch. 3 - the ideally expanded nozzle and the pressure thrust
//!   B. E. Launder, D. B. Spalding, "The numerical computation of turbulent
//!     flows", Comput. Methods Appl. Mech. Eng. 3 (1974) 269-289 - the inflow
//!     k = 3/2 (I U)^2 and the length-scale relations epsilon = C_mu^{3/4}
//!     k^{3/2}/L, omega = k^{1/2}/(C_mu^{1/4} L)
//! No GPL-licensed source was consulted.
//!
//! Provenance: ORIGINAL.
//!
//! Host only, f64 in both builds. Nothing here launches a kernel.

use std::collections::HashMap;

use crate::error::{Error, Result};

use super::{add, cross, norm, scale, sub, V3};

/// `C_mu` of the inflow length-scale relations (Launder & Spalding 1974).
pub const CMU: f64 = 0.09;

/// The default `gamma` and gas constant of a nozzle: air.
pub const AIR_GAMMA: f64 = 1.4;
pub const AIR_R: f64 = 287.05;

/// The full-valve design point of one nozzle - SPEC-LIT 161.10.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NozzleDesign {
    pub gamma: f64,
    /// J/(kg K).
    pub r: f64,
    /// The regulated chamber's stagnation pressure (Pa) and temperature (K).
    pub p0: f64,
    pub t0: f64,
    /// The exit area (m^2) and the ambient pressure (Pa).
    pub exit_area: f64,
    pub p_amb: f64,
    /// `A_e/A*`.
    pub area_ratio: f64,
    /// The exit state.
    pub mach_exit: f64,
    pub t_exit: f64,
    pub p_exit: f64,
    pub rho_exit: f64,
    pub u_exit: f64,
    /// The full-valve mass flow (kg/s) and thrust (N).
    pub mdot_max: f64,
    pub f_max: f64,
}

/// The ideally expanded exit Mach number, `M_e = sqrt(2/(g-1) ((p0/pa)^((g-1)/g) - 1))`.
pub fn ideally_expanded_mach(p0_over_pa: f64, gamma: f64) -> f64 {
    let e = (gamma - 1.0) / gamma;
    (2.0 / (gamma - 1.0) * (p0_over_pa.powf(e) - 1.0)).max(0.0).sqrt()
}

/// The area-Mach relation `A/A* = (1/M) [2/(g+1) (1 + (g-1)/2 M^2)]^((g+1)/(2(g-1)))`.
pub fn area_ratio_of_mach(m: f64, gamma: f64) -> f64 {
    let e = (gamma + 1.0) / (2.0 * (gamma - 1.0));
    (2.0 / (gamma + 1.0) * (1.0 + 0.5 * (gamma - 1.0) * m * m)).powf(e) / m
}

/// The supersonic root of the area-Mach relation for `A/A* >= 1`: Newton from
/// a start above the root, kept inside a bisection bracket on `[1, M_hi]`.
pub fn supersonic_mach_of_area_ratio(ar: f64, gamma: f64) -> Result<f64> {
    if !(ar.is_finite() && ar >= 1.0) {
        return Err(Error::Config(format!(
            "nozzle areaRatio {ar}: A_e/A* must be at least 1 (SPEC-LIT 161.10)"
        )));
    }
    if ar == 1.0 {
        return Ok(1.0);
    }
    // Bracket: f(M) = A/A*(M) - ar is negative at M = 1 and grows without bound.
    let f = |m: f64| area_ratio_of_mach(m, gamma) - ar;
    let (mut lo, mut hi) = (1.0, 2.0);
    while f(hi) < 0.0 {
        hi *= 2.0;
        if hi > 1e6 {
            return Err(Error::Config(format!("nozzle areaRatio {ar}: no supersonic root below Mach 1e6")));
        }
    }
    let mut m = hi;
    for _ in 0..200 {
        let fm = f(m);
        if fm > 0.0 {
            hi = m;
        } else {
            lo = m;
        }
        // d(A/A*)/dM = (A/A*) (M^2 - 1) / (M (1 + (g-1)/2 M^2))
        let a = area_ratio_of_mach(m, gamma);
        let d = a * (m * m - 1.0) / (m * (1.0 + 0.5 * (gamma - 1.0) * m * m));
        let mut next = if d > 0.0 { m - fm / d } else { 0.5 * (lo + hi) };
        if !(next > lo && next < hi) {
            next = 0.5 * (lo + hi);
        }
        if (next - m).abs() <= 1e-15 * m {
            return Ok(next);
        }
        m = next;
    }
    Ok(m)
}

/// The nozzle of SPEC-LIT 161.10 at full valve. `area_ratio` `None` is the
/// ideally expanded nozzle (`p_e = p_a`).
#[allow(clippy::too_many_arguments)]
pub fn nozzle(
    name: &str,
    p0: f64,
    t0: f64,
    exit_area: f64,
    gamma: f64,
    r: f64,
    p_amb: f64,
    area_ratio: Option<f64>,
) -> Result<NozzleDesign> {
    let bad = |what: String| -> Result<NozzleDesign> {
        Err(Error::Config(format!("-rigidBody: thruster {name}: nozzle {what} (SPEC-LIT 161.10)")))
    };
    for (k, v) in [("p0", p0), ("T0", t0), ("exitArea", exit_area), ("R", r), ("pAmbient", p_amb)] {
        if !(v.is_finite() && v > 0.0) {
            return bad(format!("{k} is {v}; it must be positive"));
        }
    }
    if !(gamma.is_finite() && gamma > 1.0) {
        return bad(format!("gamma is {gamma}; a perfect gas has gamma > 1"));
    }
    let k = 0.5 * (gamma - 1.0);
    let (m_e, p_e, ar) = match area_ratio {
        None => {
            if p0 <= p_amb {
                return bad(format!("p0 {p0} Pa is not above pAmbient {p_amb} Pa; nothing flows out"));
            }
            let m = ideally_expanded_mach(p0 / p_amb, gamma);
            (m, p_amb, area_ratio_of_mach(m, gamma))
        }
        Some(ar) => {
            let m = supersonic_mach_of_area_ratio(ar, gamma)
                .map_err(|e| Error::Config(format!("-rigidBody: thruster {name}: {e}")))?;
            let p = p0 * (1.0 + k * m * m).powf(-gamma / (gamma - 1.0));
            (m, p, ar)
        }
    };
    let t_e = t0 / (1.0 + k * m_e * m_e);
    let u_e = m_e * (gamma * r * t_e).sqrt();
    let rho_e = p_e / (r * t_e);
    let mdot = rho_e * u_e * exit_area;
    let f = mdot * u_e + (p_e - p_amb) * exit_area;
    if !(f > 0.0 && f.is_finite()) {
        return bad(format!(
            "thrust {f} N at full valve (exit Mach {m_e}, exit pressure {p_e} Pa); an over-expanded nozzle \
             this far is not the simple model's"
        ));
    }
    Ok(NozzleDesign {
        gamma,
        r,
        p0,
        t0,
        exit_area,
        p_amb,
        area_ratio: ar,
        mach_exit: m_e,
        t_exit: t_e,
        p_exit: p_e,
        rho_exit: rho_e,
        u_exit: u_e,
        mdot_max: mdot,
        f_max: f,
    })
}

/// The momentum-equivalent plume of SPEC-LIT 161.11 on one exit patch.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PlumeState {
    /// m/s, relative to the body, along the jet's direction.
    pub u_eq: f64,
    /// K.
    pub t_j: f64,
    /// kg/m^3.
    pub rho_j: f64,
    /// `M_eq = sqrt(F/(gamma p_a A_patch))`.
    pub mach_eq: f64,
}

/// `c_p = gamma R/(gamma - 1)`.
pub fn cp(gamma: f64, r: f64) -> f64 {
    gamma * r / (gamma - 1.0)
}

/// The plume carrying the thrust `f` (N) through `a_patch` (m^2) at ambient
/// pressure with the nozzle's total enthalpy - SPEC-LIT 161.11:
///
/// ```text
/// rho_j U_eq^2 A_patch = F,  T_j = T0 - U_eq^2/(2 c_p),  rho_j = p_a/(R T_j)
/// => U_eq^2 = (F/A) R T0 / (p_a + (F/A) R/(2 c_p))
/// ```
pub fn equivalent_plume(f: f64, a_patch: f64, t0: f64, gamma: f64, r: f64, p_amb: f64) -> PlumeState {
    if !(f > 0.0) || !(a_patch > 0.0) {
        return PlumeState { u_eq: 0.0, t_j: t0, rho_j: p_amb / (r * t0), mach_eq: 0.0 };
    }
    let q = f / a_patch;
    let c = cp(gamma, r);
    let x = q * r * t0 / (p_amb + q * r / (2.0 * c));
    let t_j = t0 - x / (2.0 * c);
    PlumeState { u_eq: x.sqrt(), t_j, rho_j: p_amb / (r * t_j), mach_eq: (f / (gamma * p_amb * a_patch)).sqrt() }
}

/// The smallest exit patch whose equivalent plume stays at or below
/// `max_mach` at full valve: `A >= F_max/(gamma p_a maxMach^2)`.
pub fn min_patch_area(f_max: f64, gamma: f64, p_amb: f64, max_mach: f64) -> f64 {
    f_max / (gamma * p_amb * max_mach * max_mach)
}

/// The Mach-area check of SPEC-LIT 161.11 for the thruster `name`: `Ok(None)`
/// when its exit patch carries the full-valve plume at or below `max_mach`;
/// otherwise the refusal, by name - or, under `-permissive`, the same
/// sentence as `Ok(Some(warning))`.
#[allow(clippy::too_many_arguments)]
pub fn check_patch_area(
    name: &str,
    f_max: f64,
    a_patch: f64,
    t0: f64,
    gamma: f64,
    r: f64,
    p_amb: f64,
    max_mach: f64,
    permissive: bool,
) -> Result<Option<String>> {
    let a_min = min_patch_area(f_max, gamma, p_amb, max_mach);
    if a_patch >= a_min {
        return Ok(None);
    }
    let full = equivalent_plume(f_max, a_patch, t0, gamma, r, p_amb);
    let msg = format!(
        "-rigidBody: thruster {name}: its exit patch is {a_patch:.6e} m2, below F_max/(gamma pa maxMach^2) = \
         {a_min:.6e} m2 - the equivalent plume of {f_max:.6e} N would run at Mach {:.4} > plume.maxMach {max_mach} \
         (SPEC-LIT 161.11)",
        full.mach_eq
    );
    if permissive {
        Ok(Some(msg))
    } else {
        Err(Error::Config(msg))
    }
}

/// The turbulence a jet patch carries in - SPEC-LIT 161.11.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct JetTurbulence {
    pub k: f64,
    pub omega: f64,
    pub epsilon: f64,
    pub nu_tilda: f64,
}

/// `k = 3/2 (I U)^2`, `omega = k^{1/2}/(C_mu^{1/4} L)`, `epsilon = C_mu^{3/4}
/// k^{3/2}/L`, `nuTilda = (3/2)^{1/2} I U L`, with `U = max(u, u_floor)` so
/// that a shut valve still writes finite, positive values.
pub fn jet_turbulence(u: f64, u_floor: f64, intensity: f64, length: f64) -> JetTurbulence {
    let ue = u.max(u_floor);
    let k = 1.5 * (intensity * ue).powi(2);
    JetTurbulence {
        k,
        omega: k.sqrt() / (CMU.powf(0.25) * length),
        epsilon: CMU.powf(0.75) * k.powf(1.5) / length,
        nu_tilda: 1.5f64.sqrt() * intensity * ue * length,
    }
}

/// The geometry of one patch, from the rest points, in f64.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PatchGeometry {
    /// m^2, the sum of the face areas.
    pub area: f64,
    /// The area-weighted centroid (rest frame).
    pub centroid: V3,
    /// The length of the patch's outer edges (those one face of the patch owns).
    pub perimeter: f64,
    /// `4 A / P`.
    pub hydraulic_diameter: f64,
    /// The sum of the face area vectors.
    pub area_vector: V3,
}

/// The patch geometry of `faces` (each a point loop) on `points`: every face
/// fanned about its vertex average, as SPEC-LIT 2.1 fans it.
pub fn patch_geometry(points: &[V3], faces: &[&[usize]]) -> PatchGeometry {
    let mut area = 0.0;
    let mut moment = [0.0; 3];
    let mut sv = [0.0; 3];
    let mut edges: HashMap<(usize, usize), u32> = HashMap::new();
    for f in faces {
        let n = f.len();
        if n < 3 {
            continue;
        }
        let mut avg = [0.0; 3];
        for &p in f.iter() {
            avg = add(avg, points[p]);
        }
        avg = scale(avg, 1.0 / n as f64);
        for i in 0..n {
            let a = points[f[i]];
            let c = points[f[(i + 1) % n]];
            let s = scale(cross(sub(a, avg), sub(c, avg)), 0.5);
            let ar = norm(s);
            area += ar;
            sv = add(sv, s);
            let cen = scale(add(add(a, c), avg), 1.0 / 3.0);
            moment = add(moment, scale(cen, ar));
            let key = if f[i] < f[(i + 1) % n] { (f[i], f[(i + 1) % n]) } else { (f[(i + 1) % n], f[i]) };
            *edges.entry(key).or_insert(0) += 1;
        }
    }
    let perimeter: f64 = edges
        .iter()
        .filter(|(_, &c)| c == 1)
        .map(|(&(a, b), _)| norm(sub(points[a], points[b])))
        .sum();
    let centroid = if area > 0.0 { scale(moment, 1.0 / area) } else { [0.0; 3] };
    PatchGeometry {
        area,
        centroid,
        perimeter,
        hydraulic_diameter: if perimeter > 0.0 { 4.0 * area / perimeter } else { 0.0 },
        area_vector: sv,
    }
}
