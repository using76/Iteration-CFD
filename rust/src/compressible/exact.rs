// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The analytic answer keys of the compressible gates: the isentropic
//! ratios, the area-Mach relation with both roots, the normal shock and its
//! inverse from the total-pressure ratio, the oblique shock (weak branch and
//! maximum deflection), and the shocked quasi-1-D nozzle with its regime
//! limits. SPEC-LIT §127.2 is the contract: its equations (127.2) to
//! (127.4) are evaluated here, and its derived nozzle relations are
//! inverted here.
//!
//! Written from:
//!   SPEC-LIT §127.2 - the equations, the weak-branch rule, the shocked
//!     nozzle and its forward map, the regime limits, and the oracle values
//!     the tests quote
//!   Ames Research Staff, *Equations, Tables, and Charts for Compressible
//!     Flow*, NACA Report 1135 (1953), NTRS 19930091059, a US-Government
//!     work - eqs. 43 to 45 (the isentropic ratios), 80 (the area ratio),
//!     93, 96 and 99 (the normal shock), 128, 132 and 139b (the oblique
//!     shock), read from the page images by the CMP-05 supervisor
//! The roots are found by bisection on a bracket where the function is
//! monotone, and the maximum deflection by golden-section search. No
//! compressible-flow code (OpenFOAM, SU2, or any table generator) was
//! opened. No GPL-licensed source was consulted.
//!
//! Precision (SPEC-LIT §127.6): `f64` everywhere, in both builds. An answer
//! key is a constant of a gate and never a `Scalar`, and the module has no
//! device type, so it behaves identically in the default build and in
//! `--features single`. Its only crate import is [`Error`].
//!
//! Angles are in RADIANS. NACA 1135 writes the shock angle `theta` and the
//! deflection `delta`; this API names them `shock_angle` and `deflection`,
//! and the code below keeps NACA's `theta` for the shock angle inside its
//! own equations. Every refusal is an `Err(Error::Refused(..))` whose
//! message starts `compressible::exact::<function name>: `.

use crate::error::Error;

/// The top of every supersonic bracket. A supersonic root that lies above it
/// is refused by name rather than returned.
pub const MACH_MAX: f64 = 50.0;

/// The most halvings (or golden-section steps) any search takes. Every
/// search stops earlier, when its bracket stops shrinking in `f64`.
const MAX_ITER: usize = 200;

/// Which of the two roots of the area-Mach relation to return.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Branch {
    /// The root with `M <= 1`.
    Subsonic,
    /// The root with `M >= 1`.
    Supersonic,
}

/// What a normal shock does to a flow of Mach `m1`: [`normal_shock`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalShock {
    /// The Mach number behind the shock, NACA 1135 eq. 96.
    pub m2: f64,
    /// The static pressure ratio `p_2/p_1`, NACA 1135 eq. 93.
    pub p2_p1: f64,
    /// The total pressure ratio `p_t2/p_t1`, NACA (99).
    pub pt2_pt1: f64,
}

/// An oblique shock: [`oblique_from_shock_angle`] and [`oblique_shock_weak`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObliqueShock {
    /// The shock angle to the incoming flow, radians (NACA's `theta`).
    pub shock_angle: f64,
    /// The flow deflection, radians (NACA's `delta`).
    pub deflection: f64,
    /// The static pressure ratio `p_2/p_1`, NACA (128).
    pub p2_p1: f64,
    /// The Mach number behind the shock, NACA (132).
    pub m2: f64,
}

/// The three back pressures, as `p_b/p_t1`, that bound the regimes of a
/// nozzle `A_e/A_*`: [`nozzle_limits`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NozzleLimits {
    /// The shock at the throat: above this the flow downstream of the throat
    /// is subsonic and no shock stands in the nozzle.
    pub pb_choked: f64,
    /// The shock at the exit plane: below this the shock stands outside the
    /// exit and none stands inside.
    pub pb_shock_at_exit: f64,
    /// The shock-free design exit, `(p/p_t)` at the supersonic root of
    /// `A_e/A_*`.
    pub pb_design: f64,
}

/// A nozzle with a normal shock inside it: [`shocked_nozzle`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShockedNozzle {
    /// The exit Mach number, subsonic.
    pub m_e: f64,
    /// The total pressure ratio across the shock, `p_t2/p_t1`.
    pub pt2_pt1: f64,
    /// The Mach number just upstream of the shock, supersonic.
    pub m1: f64,
    /// The area at which the shock stands, `A_s/A_*1`, in units of the
    /// throat area.
    pub as_astar: f64,
}

// ==========================================================================
//  Refusals and root finding
// ==========================================================================

/// A refusal in the one house format.
fn refuse(who: &str, msg: String) -> Error {
    Error::Refused(format!("compressible::exact::{who}: {msg}"))
}

/// Every `Result` function checks `gamma` first.
fn check_gamma(who: &str, gamma: f64) -> Result<(), Error> {
    if gamma.is_finite() && gamma > 1.0 {
        Ok(())
    } else {
        Err(refuse(
            who,
            format!("gamma must be finite and above 1, got {gamma}"),
        ))
    }
}

/// Every named input must be a finite number.
fn check_finite(who: &str, inputs: &[(&str, f64)]) -> Result<(), Error> {
    for &(name, v) in inputs {
        if !v.is_finite() {
            return Err(refuse(who, format!("{name} is not finite, got {v}")));
        }
    }
    Ok(())
}

/// `A_e/A_*` of a nozzle is at least 1.
fn check_exit_area(who: &str, ae_astar: f64) -> Result<(), Error> {
    if ae_astar < 1.0 {
        return Err(refuse(
            who,
            format!("A_e/A_* = {ae_astar} is below 1, and no nozzle exit is smaller than its throat"),
        ));
    }
    Ok(())
}

/// Bisection. `root_above(mid)` says the root lies above `mid`, which is how
/// a function that is monotone on `[lo, hi]` is read, increasing or
/// decreasing. It stops when the midpoint equals an end, that is when the
/// bracket stops shrinking in `f64`, or after [`MAX_ITER`] halvings.
fn bisect(mut lo: f64, mut hi: f64, root_above: impl Fn(f64) -> bool) -> f64 {
    for _ in 0..MAX_ITER {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if root_above(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Golden-section search for the maximiser of a function that is unimodal on
/// `[a, b]`, to the bracket's `f64` floor: it stops when the two interior
/// points can no longer be told apart from each other and from the ends, or
/// after [`MAX_ITER`] steps.
fn golden_max(mut a: f64, mut b: f64, f: impl Fn(f64) -> f64) -> f64 {
    const INV_PHI: f64 = 0.618_033_988_749_894_9;
    for _ in 0..MAX_ITER {
        let c = b - INV_PHI * (b - a);
        let d = a + INV_PHI * (b - a);
        if !(a < c && c < d && d < b) {
            break;
        }
        if f(c) < f(d) {
            a = c;
        } else {
            b = d;
        }
    }
    0.5 * (a + b)
}

// ==========================================================================
//  Isentropic flow
// ==========================================================================

/// `1 + (gamma-1)/2 M^2`, the factor of NACA 1135 eqs. 43 to 45.
fn stagnation_factor(gamma: f64, mach: f64) -> f64 {
    1.0 + 0.5 * (gamma - 1.0) * mach * mach
}

/// `T/T_t = (1 + (gamma-1)/2 M^2)^-1`, NACA 1135 eq. 43, (127.2). Checks
/// nothing: NaN in, NaN out.
pub fn temperature_ratio(gamma: f64, mach: f64) -> f64 {
    1.0 / stagnation_factor(gamma, mach)
}

/// `p/p_t = (1 + (gamma-1)/2 M^2)^(-gamma/(gamma-1))`, NACA 1135 eq. 44,
/// (127.2). Checks nothing: NaN in, NaN out.
pub fn pressure_ratio(gamma: f64, mach: f64) -> f64 {
    stagnation_factor(gamma, mach).powf(-gamma / (gamma - 1.0))
}

/// `rho/rho_t = (1 + (gamma-1)/2 M^2)^(-1/(gamma-1))`, NACA 1135 eq. 45,
/// (127.2). Checks nothing: NaN in, NaN out.
pub fn density_ratio(gamma: f64, mach: f64) -> f64 {
    stagnation_factor(gamma, mach).powf(-1.0 / (gamma - 1.0))
}

/// `A/A_*`, the reciprocal of NACA 1135 eq. 80 (`A_*/A`), (127.2):
/// `A/A_* = [ (2 + (gamma-1) M^2)/(gamma+1) ]^((gamma+1)/(2(gamma-1))) / M`.
/// It is `+inf` at `M = 0`, 1 at `M = 1`, and rises on both sides. Checks
/// nothing: NaN in, NaN out.
pub fn area_ratio(gamma: f64, mach: f64) -> f64 {
    let e = (gamma + 1.0) / (2.0 * (gamma - 1.0));
    ((2.0 + (gamma - 1.0) * mach * mach) / (gamma + 1.0)).powf(e) / mach
}

/// The Mach number at which `A/A_*` equals `a`, on the chosen branch.
/// `a` is finite and at least 1, and `gamma` is valid: the callers checked.
/// `who` names the public function a refusal is reported under.
fn mach_from_area_inner(who: &str, gamma: f64, a: f64, branch: Branch) -> Result<f64, Error> {
    if a == 1.0 {
        return Ok(1.0);
    }
    match branch {
        // A/A_* falls from +inf at M = 0 to 1 at M = 1.
        Branch::Subsonic => Ok(bisect(0.0, 1.0, |m| area_ratio(gamma, m) > a)),
        // A/A_* rises from 1 at M = 1.
        Branch::Supersonic => {
            let top = area_ratio(gamma, MACH_MAX);
            if a > top {
                return Err(refuse(
                    who,
                    format!(
                        "the supersonic root of A/A_* = {a} lies above MACH_MAX = {MACH_MAX}, \
                         where A/A_* is {top}"
                    ),
                ));
            }
            Ok(bisect(1.0, MACH_MAX, |m| area_ratio(gamma, m) < a))
        }
    }
}

/// The Mach number at which `A/A_*` equals `area_ratio`: the inverse of
/// [`area_ratio`], (127.2), by bisection on `[0, 1]` (subsonic) or
/// `[1, MACH_MAX]` (supersonic). `area_ratio == 1` gives exactly `1.0` on
/// both branches.
///
/// Refuses (`Error::Refused`): `gamma` not finite or `<= 1`; an input not
/// finite; `area_ratio` below 1; and, for [`Branch::Supersonic`], a root
/// above [`MACH_MAX`].
pub fn mach_from_area_ratio(gamma: f64, area_ratio: f64, branch: Branch) -> Result<f64, Error> {
    const WHO: &str = "mach_from_area_ratio";
    check_gamma(WHO, gamma)?;
    check_finite(WHO, &[("area_ratio", area_ratio)])?;
    if area_ratio < 1.0 {
        return Err(refuse(
            WHO,
            format!("A/A_* = {area_ratio} is below 1, and no flow has an area under its sonic area"),
        ));
    }
    mach_from_area_inner(WHO, gamma, area_ratio, branch)
}

// ==========================================================================
//  The normal shock
// ==========================================================================

/// `p_t2/p_t1` across a normal shock at `m1`, NACA (99), (127.3).
fn normal_total_pressure_ratio(gamma: f64, m1: f64) -> f64 {
    let m1s = m1 * m1;
    ((gamma + 1.0) * m1s / ((gamma - 1.0) * m1s + 2.0)).powf(gamma / (gamma - 1.0))
        * ((gamma + 1.0) / (2.0 * gamma * m1s - (gamma - 1.0))).powf(1.0 / (gamma - 1.0))
}

/// The normal shock at a flow of Mach `m1`: `p_2/p_1` (NACA 1135 eq. 93),
/// `M_2` (eq. 96) and `p_t2/p_t1` (NACA (99)), (127.3). `m1 = 1` is the
/// identity, exactly.
///
/// Refuses (`Error::Refused`): `gamma` not finite or `<= 1`; `m1` not
/// finite; `m1 < 1`, which would be an expansion shock.
pub fn normal_shock(gamma: f64, m1: f64) -> Result<NormalShock, Error> {
    const WHO: &str = "normal_shock";
    check_gamma(WHO, gamma)?;
    check_finite(WHO, &[("m1", m1)])?;
    if m1 < 1.0 {
        return Err(refuse(
            WHO,
            format!("m1 = {m1} is subsonic, and a shock there would be an expansion shock"),
        ));
    }
    if m1 == 1.0 {
        return Ok(NormalShock {
            m2: 1.0,
            p2_p1: 1.0,
            pt2_pt1: 1.0,
        });
    }
    let m1s = m1 * m1;
    Ok(NormalShock {
        m2: (((gamma - 1.0) * m1s + 2.0) / (2.0 * gamma * m1s - (gamma - 1.0))).sqrt(),
        p2_p1: (2.0 * gamma * m1s - (gamma - 1.0)) / (gamma + 1.0),
        pt2_pt1: normal_total_pressure_ratio(gamma, m1),
    })
}

/// The supersonic root of NACA (99) at `r`. `r` is in `(0, 1]` and `gamma`
/// is valid: the callers checked. `who` names the function a refusal is
/// reported under.
fn mach_from_total_pressure_ratio(who: &str, gamma: f64, r: f64) -> Result<f64, Error> {
    if r == 1.0 {
        return Ok(1.0);
    }
    let floor = normal_total_pressure_ratio(gamma, MACH_MAX);
    if r < floor {
        return Err(refuse(
            who,
            format!(
                "p_t2/p_t1 = {r} needs an upstream Mach above MACH_MAX = {MACH_MAX}, \
                 where it is {floor}"
            ),
        ));
    }
    // NACA (99) falls from 1 at M_1 = 1.
    Ok(bisect(1.0, MACH_MAX, |m| {
        normal_total_pressure_ratio(gamma, m) > r
    }))
}

/// The upstream Mach number of a normal shock whose total-pressure ratio is
/// `pt2_pt1`: the inverse of NACA (99), (127.3), by bisection on
/// `[1, MACH_MAX]`. `pt2_pt1 = 1` gives exactly `1.0`.
///
/// Refuses (`Error::Refused`): `gamma` not finite or `<= 1`; `pt2_pt1` not
/// finite; `pt2_pt1` outside `(0, 1]`; a root above [`MACH_MAX`].
pub fn normal_shock_mach_from_pt_ratio(gamma: f64, pt2_pt1: f64) -> Result<f64, Error> {
    const WHO: &str = "normal_shock_mach_from_pt_ratio";
    check_gamma(WHO, gamma)?;
    check_finite(WHO, &[("pt2_pt1", pt2_pt1)])?;
    if !(pt2_pt1 > 0.0 && pt2_pt1 <= 1.0) {
        return Err(refuse(
            WHO,
            format!("p_t2/p_t1 = {pt2_pt1} is outside (0, 1], the range of a normal shock"),
        ));
    }
    mach_from_total_pressure_ratio(WHO, gamma, pt2_pt1)
}

// ==========================================================================
//  The oblique shock
// ==========================================================================

/// `tan delta` of NACA (139b), (127.4), at shock angle `theta`.
fn tan_deflection(gamma: f64, m1: f64, theta: f64) -> f64 {
    let m1s = m1 * m1;
    let (sin2, cos2) = (2.0 * theta).sin_cos();
    let cot = theta.cos() / theta.sin();
    (m1s * sin2 - 2.0 * cot) / (2.0 + m1s * (gamma + cos2))
}

/// The state behind an oblique shock at shock angle `theta`: NACA (139b),
/// (128) and (132), (127.4). The caller checked the angle's range.
fn oblique_state(gamma: f64, m1: f64, theta: f64) -> ObliqueShock {
    let m1s = m1 * m1;
    let sin_sq = theta.sin() * theta.sin();
    let x = m1s * sin_sq;
    let p2_p1 = (2.0 * gamma * m1s * sin_sq - (gamma - 1.0)) / (gamma + 1.0);
    let num = (gamma + 1.0) * (gamma + 1.0) * m1s * m1s * sin_sq
        - 4.0 * (x - 1.0) * (gamma * x + 1.0);
    let den = (2.0 * gamma * x - (gamma - 1.0)) * ((gamma - 1.0) * x + 2.0);
    ObliqueShock {
        shock_angle: theta,
        deflection: tan_deflection(gamma, m1, theta).atan(),
        p2_p1,
        m2: (num / den).sqrt(),
    }
}

/// The shock angle of maximum deflection: the maximiser of NACA (139b)'s
/// `tan delta` over `(asin(1/m1), pi/2)`, where it is unimodal, by
/// golden-section search. `m1` is above 1: the callers checked.
fn max_deflection_angle(gamma: f64, m1: f64) -> f64 {
    golden_max((1.0 / m1).asin(), std::f64::consts::FRAC_PI_2, |t| {
        tan_deflection(gamma, m1, t)
    })
}

/// The oblique shock at a given shock angle: the deflection (NACA (139b),
/// as `atan` of `tan delta`), `p_2/p_1` (NACA (128)) and `M_2` (NACA (132)),
/// (127.4). Angles are in radians. At `shock_angle = pi/2` it is the normal
/// shock of [`normal_shock`], with a deflection of zero to round-off.
///
/// Refuses (`Error::Refused`): `gamma` not finite or `<= 1`; an input not
/// finite; `m1 <= 1`, which needs a supersonic flow; `shock_angle` outside
/// `[asin(1/m1), pi/2]`.
pub fn oblique_from_shock_angle(
    gamma: f64,
    m1: f64,
    shock_angle: f64,
) -> Result<ObliqueShock, Error> {
    const WHO: &str = "oblique_from_shock_angle";
    check_gamma(WHO, gamma)?;
    check_finite(WHO, &[("m1", m1), ("shock_angle", shock_angle)])?;
    if m1 <= 1.0 {
        return Err(refuse(
            WHO,
            format!("m1 = {m1} is not supersonic, and an oblique shock needs M_1 above 1"),
        ));
    }
    let mach_angle = (1.0 / m1).asin();
    if !(mach_angle..=std::f64::consts::FRAC_PI_2).contains(&shock_angle) {
        return Err(refuse(
            WHO,
            format!(
                "shock angle {shock_angle} rad is outside [{mach_angle}, {}], \
                 the Mach angle and a right angle",
                std::f64::consts::FRAC_PI_2
            ),
        ));
    }
    Ok(oblique_state(gamma, m1, shock_angle))
}

/// The maximum deflection an attached oblique shock turns a flow of Mach
/// `m1` through, as `(shock_angle, deflection)` in radians: the maximum of
/// NACA (139b) over the shock angle, found by golden-section search. The
/// maximum is flat in the angle, so the angle is good to about the square
/// root of the `f64` unit round-off, and the deflection to round-off.
///
/// Refuses (`Error::Refused`): `gamma` not finite or `<= 1`; `m1` not
/// finite; `m1 <= 1`, which needs a supersonic flow.
pub fn max_deflection(gamma: f64, m1: f64) -> Result<(f64, f64), Error> {
    const WHO: &str = "max_deflection";
    check_gamma(WHO, gamma)?;
    check_finite(WHO, &[("m1", m1)])?;
    if m1 <= 1.0 {
        return Err(refuse(
            WHO,
            format!("m1 = {m1} is not supersonic, and an oblique shock needs M_1 above 1"),
        ));
    }
    let theta = max_deflection_angle(gamma, m1);
    Ok((theta, tan_deflection(gamma, m1, theta).atan()))
}

/// The weak-branch oblique shock for a given deflection, (127.4): the root
/// of NACA (139b) between the Mach angle `asin(1/m1)` and the angle of
/// maximum deflection, by bisection on `tan delta`, then NACA (128) and
/// (132). Angles are in radians; the returned `deflection` is the input.
/// `deflection = 0` gives the Mach angle and a pressure ratio of 1.
///
/// Refuses (`Error::Refused`): `gamma` not finite or `<= 1`; an input not
/// finite; `m1 <= 1`, which needs a supersonic flow; `deflection < 0`;
/// `deflection` above the maximum [`max_deflection`] returns, where the
/// shock detaches.
pub fn oblique_shock_weak(gamma: f64, m1: f64, deflection: f64) -> Result<ObliqueShock, Error> {
    const WHO: &str = "oblique_shock_weak";
    check_gamma(WHO, gamma)?;
    check_finite(WHO, &[("m1", m1), ("deflection", deflection)])?;
    if m1 <= 1.0 {
        return Err(refuse(
            WHO,
            format!("m1 = {m1} is not supersonic, and an oblique shock needs M_1 above 1"),
        ));
    }
    if deflection < 0.0 {
        return Err(refuse(
            WHO,
            format!("deflection {deflection} rad is negative, and a negative turn is the mirror image of a positive one"),
        ));
    }
    let theta_max = max_deflection_angle(gamma, m1);
    let delta_max = tan_deflection(gamma, m1, theta_max).atan();
    if deflection > delta_max {
        return Err(refuse(
            WHO,
            format!(
                "deflection {deflection} rad exceeds the maximum {delta_max} rad at M_1 = {m1}: \
                 the shock is detached"
            ),
        ));
    }
    let target = deflection.tan();
    // tan delta rises from 0 at the Mach angle to its maximum at theta_max.
    let theta = bisect((1.0 / m1).asin(), theta_max, |t| {
        tan_deflection(gamma, m1, t) < target
    });
    let mut state = oblique_state(gamma, m1, theta);
    state.deflection = deflection;
    Ok(state)
}

// ==========================================================================
//  The shocked nozzle
// ==========================================================================

/// The forward map of SPEC-LIT §127.2: `p_b/p_t1` of a nozzle `A_e/A_*1`
/// with a normal shock standing at area `A_s/A_*1`. `as_astar` is in
/// `[1, ae_astar]` and `gamma` is valid: the callers checked.
fn back_pressure_inner(who: &str, gamma: f64, as_astar: f64, ae_astar: f64) -> Result<f64, Error> {
    let m1 = mach_from_area_inner(who, gamma, as_astar, Branch::Supersonic)?;
    let r = if m1 == 1.0 {
        1.0
    } else {
        normal_total_pressure_ratio(gamma, m1)
    };
    // A_e/A_*2 = (A_e/A_*1) r is at least 1 by the algebra; the clamp only
    // absorbs round-off at a shock sitting on the throat.
    let m_e = mach_from_area_inner(who, gamma, (ae_astar * r).max(1.0), Branch::Subsonic)?;
    Ok(pressure_ratio(gamma, m_e) * r)
}

/// The two ends of the interval of `p_b/p_t1` in which a normal shock stands
/// inside a nozzle `A_e/A_*1`, and the design exit. The shock at the throat
/// and the shock at the exit are the forward map at `A_s/A_*1 = 1` and
/// `A_s/A_*1 = A_e/A_*1`, so the limits and the inverse in [`shocked_nozzle`]
/// agree to the last bit at both ends.
fn nozzle_limits_inner(who: &str, gamma: f64, ae_astar: f64) -> Result<NozzleLimits, Error> {
    let m_sup = mach_from_area_inner(who, gamma, ae_astar, Branch::Supersonic)?;
    Ok(NozzleLimits {
        pb_choked: back_pressure_inner(who, gamma, 1.0, ae_astar)?,
        pb_shock_at_exit: back_pressure_inner(who, gamma, ae_astar, ae_astar)?,
        pb_design: pressure_ratio(gamma, m_sup),
    })
}

/// The regimes of a nozzle `A_e/A_*1`, SPEC-LIT §127.2, as `p_b/p_t1`. With
/// `M_sub` and `M_sup` the two roots of `A/A_* = A_e/A_*1`, a normal shock
/// stands inside the nozzle exactly when `p_b/p_t1` lies in
/// `[pb_shock_at_exit, pb_choked]`, where `pb_choked = (p/p_t)(M_sub)` puts
/// the shock at the throat and `pb_shock_at_exit = (p/p_t)(M_sup)(p_2/p_1)(M_sup)`
/// puts it at the exit. `pb_design = (p/p_t)(M_sup)` is the shock-free design
/// exit.
///
/// Refuses (`Error::Refused`): `gamma` not finite or `<= 1`; `ae_astar` not
/// finite; `ae_astar < 1`; a supersonic root above [`MACH_MAX`].
pub fn nozzle_limits(gamma: f64, ae_astar: f64) -> Result<NozzleLimits, Error> {
    const WHO: &str = "nozzle_limits";
    check_gamma(WHO, gamma)?;
    check_finite(WHO, &[("ae_astar", ae_astar)])?;
    check_exit_area(WHO, ae_astar)?;
    nozzle_limits_inner(WHO, gamma, ae_astar)
}

/// The shocked nozzle's inverse, SPEC-LIT §127.2: where the normal shock
/// stands, given `p_b/p_t1` and `A_e/A_*1`. The exit Mach `M_e` is the
/// subsonic root of `(p/p_t)(M_e)(A/A_*)(M_e) = (p_b/p_t1)(A_e/A_*1)`, found
/// by bisection on `[0, 1]`; then `p_t2/p_t1 = (p_b/p_t1)/(p/p_t)(M_e)`,
/// clamped to at most 1 so that a back pressure exactly at `pb_choked` gives
/// `m1 = 1` and `as_astar = 1`; `M_1` is the supersonic root of NACA (99) at
/// that ratio; and `A_s/A_*1 = (A/A_*)(M_1)`.
///
/// Refuses (`Error::Refused`): `gamma` not finite or `<= 1`; an input not
/// finite; `ae_astar < 1`; `pb_pt1` above `pb_choked` or below
/// `pb_shock_at_exit`, both by `no normal shock inside the nozzle`, with the
/// back pressure and the limits in the message.
pub fn shocked_nozzle(gamma: f64, pb_pt1: f64, ae_astar: f64) -> Result<ShockedNozzle, Error> {
    const WHO: &str = "shocked_nozzle";
    check_gamma(WHO, gamma)?;
    check_finite(WHO, &[("pb_pt1", pb_pt1), ("ae_astar", ae_astar)])?;
    check_exit_area(WHO, ae_astar)?;
    let lim = nozzle_limits_inner(WHO, gamma, ae_astar)?;
    if pb_pt1 > lim.pb_choked {
        return Err(refuse(
            WHO,
            format!(
                "no normal shock inside the nozzle: p_b/p_t1 = {pb_pt1} is above the choked \
                 limit {} (A_e/A_* = {ae_astar}), so the flow behind the throat is subsonic; \
                 a shock stands inside for p_b/p_t1 in [{}, {}]",
                lim.pb_choked, lim.pb_shock_at_exit, lim.pb_choked
            ),
        ));
    }
    if pb_pt1 < lim.pb_shock_at_exit {
        return Err(refuse(
            WHO,
            format!(
                "no normal shock inside the nozzle: p_b/p_t1 = {pb_pt1} is below the \
                 shock-at-exit limit {} (A_e/A_* = {ae_astar}), so the shock stands outside \
                 the exit; a shock stands inside for p_b/p_t1 in [{}, {}]",
                lim.pb_shock_at_exit, lim.pb_shock_at_exit, lim.pb_choked
            ),
        ));
    }
    let target = pb_pt1 * ae_astar;
    // (p/p_t)(A/A_*) falls monotonically with M, from +inf at M = 0.
    let m_e = bisect(0.0, 1.0, |m| {
        pressure_ratio(gamma, m) * area_ratio(gamma, m) > target
    });
    let pt2_pt1 = (pb_pt1 / pressure_ratio(gamma, m_e)).min(1.0);
    let m1 = mach_from_total_pressure_ratio(WHO, gamma, pt2_pt1)?;
    let as_astar = if m1 == 1.0 { 1.0 } else { area_ratio(gamma, m1) };
    Ok(ShockedNozzle {
        m_e,
        pt2_pt1,
        m1,
        as_astar,
    })
}

/// The shocked nozzle's forward map, SPEC-LIT §127.2: `p_b/p_t1` of a
/// nozzle `A_e/A_*1` with a normal shock at area `A_s/A_*1`. `M_1` is the
/// supersonic root of `(A/A_*)(M_1) = A_s/A_*1`; `r = p_t2/p_t1` is NACA (99)
/// at `M_1`; `M_e` is the subsonic root of `(A/A_*)(M_e) = (A_e/A_*1) r`; and
/// `p_b/p_t1 = (p/p_t)(M_e) r`. It is what [`shocked_nozzle`] inverts.
///
/// Refuses (`Error::Refused`): `gamma` not finite or `<= 1`; an input not
/// finite; `ae_astar < 1`; `as_astar` outside `[1, ae_astar]`; a root above
/// [`MACH_MAX`].
pub fn shocked_nozzle_back_pressure(
    gamma: f64,
    as_astar: f64,
    ae_astar: f64,
) -> Result<f64, Error> {
    const WHO: &str = "shocked_nozzle_back_pressure";
    check_gamma(WHO, gamma)?;
    check_finite(WHO, &[("as_astar", as_astar), ("ae_astar", ae_astar)])?;
    check_exit_area(WHO, ae_astar)?;
    if !(1.0..=ae_astar).contains(&as_astar) {
        return Err(refuse(
            WHO,
            format!(
                "A_s/A_* = {as_astar} is outside [1, {ae_astar}], the throat area and the exit \
                 area of the nozzle"
            ),
        ));
    }
    back_pressure_inner(WHO, gamma, as_astar, ae_astar)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    /// The relative difference `|got/want - 1|`.
    fn rel(got: f64, want: f64) -> f64 {
        (got / want - 1.0).abs()
    }

    fn close_rel(what: &str, got: f64, want: f64, tol: f64) {
        assert!(
            rel(got, want) <= tol,
            "{what}: got {got:e}, want {want:e}, rel {:e} > {tol:e}",
            rel(got, want)
        );
    }

    fn close_abs(what: &str, got: f64, want: f64, tol: f64) {
        assert!(
            (got - want).abs() <= tol,
            "{what}: got {got:e}, want {want:e}, abs {:e} > {tol:e}",
            (got - want).abs()
        );
    }

    /// The result is `Err(Error::Refused(msg))`, `msg` starts with
    /// `compressible::exact::` and contains every needle; panics with the
    /// message (or the value) otherwise.
    fn refused<T: std::fmt::Debug>(res: Result<T, Error>, needles: &[&str]) {
        match res {
            Err(Error::Refused(msg)) => {
                assert!(
                    msg.starts_with("compressible::exact::"),
                    "refusal does not start with compressible::exact:: : {msg}"
                );
                for n in needles {
                    assert!(msg.contains(n), "refusal lacks {n:?}: {msg}");
                }
            }
            other => panic!("expected a refusal containing {needles:?}, got {other:?}"),
        }
    }

    #[test]
    fn isentropic_ratios_match_naca_43_to_45_and_80() {
        let g = 1.4;
        let tol = 4e-15;
        // M 2: the plan's bar A/A_* = 1.6875.
        close_rel("T/Tt M2", temperature_ratio(g, 2.0), 0.5555555555555556, tol);
        close_rel("p/pt M2", pressure_ratio(g, 2.0), 0.12780452546295094, tol);
        close_rel("rho/rhot M2", density_ratio(g, 2.0), 0.2300481458333117, tol);
        close_rel("A/A* M2", area_ratio(g, 2.0), 1.6875, tol);
        // M 0.5.
        close_rel("T/Tt M0.5", temperature_ratio(g, 0.5), 0.9523809523809523, tol);
        close_rel("p/pt M0.5", pressure_ratio(g, 0.5), 0.8430191754225532, tol);
        close_rel("rho/rhot M0.5", density_ratio(g, 0.5), 0.8851701341936809, tol);
        close_rel("A/A* M0.5", area_ratio(g, 0.5), 1.33984375, tol);
        // M 1.
        close_rel("T/Tt M1", temperature_ratio(g, 1.0), 0.8333333333333334, tol);
        close_rel("A/A* M1", area_ratio(g, 1.0), 1.0, tol);
        // M 0: the stagnation state, exactly.
        assert_eq!(temperature_ratio(g, 0.0), 1.0);
        assert_eq!(pressure_ratio(g, 0.0), 1.0);
        assert_eq!(density_ratio(g, 0.0), 1.0);
        assert_eq!(area_ratio(g, 0.0), f64::INFINITY);
        // gamma 5/3.
        close_rel(
            "T/Tt g5/3 M2",
            temperature_ratio(5.0 / 3.0, 2.0),
            0.42857142857142855,
            tol,
        );
    }

    #[test]
    fn area_mach_inverse_gives_both_roots() {
        let g = 1.4;
        let root = |a: f64, b: Branch| mach_from_area_ratio(g, a, b).unwrap();
        close_abs("1.6875 sub", root(1.6875, Branch::Subsonic), 0.3722444862027501, 1e-12);
        close_abs("1.6875 sup", root(1.6875, Branch::Supersonic), 2.0, 1e-12);
        close_abs("2 sub", root(2.0, Branch::Subsonic), 0.3059038341891082, 1e-12);
        close_abs("2 sup", root(2.0, Branch::Supersonic), 2.1971981216521865, 1e-12);
        close_abs("10 sub", root(10.0, Branch::Subsonic), 0.05798720292287384, 1e-12);
        close_abs("10 sup", root(10.0, Branch::Supersonic), 3.922551820933724, 1e-12);
        // The sonic throat, exactly, on both branches.
        assert_eq!(root(1.0, Branch::Subsonic), 1.0);
        assert_eq!(root(1.0, Branch::Supersonic), 1.0);
        // The round trip.
        for a in [1.001, 1.5, 3.0, 25.0] {
            for b in [Branch::Subsonic, Branch::Supersonic] {
                let m = root(a, b);
                close_rel(
                    &format!("round trip a={a} {b:?}"),
                    area_ratio(g, m),
                    a,
                    1e-13,
                );
            }
        }
        // The roots lie on their own sides of the throat.
        assert!(root(1.5, Branch::Subsonic) < 1.0);
        assert!(root(1.5, Branch::Supersonic) > 1.0);
    }

    #[test]
    fn area_mach_inverse_refuses_by_name() {
        let g = 1.4;
        refused(mach_from_area_ratio(g, 0.99, Branch::Subsonic), &["below 1"]);
        refused(mach_from_area_ratio(g, 0.99, Branch::Supersonic), &["below 1"]);
        refused(mach_from_area_ratio(g, f64::NAN, Branch::Subsonic), &["not finite"]);
        refused(
            mach_from_area_ratio(g, f64::INFINITY, Branch::Supersonic),
            &["not finite"],
        );
        // The ratio of M 60 is beyond the supersonic bracket, but its subsonic
        // root is an ordinary one.
        let a60 = area_ratio(g, 60.0);
        refused(mach_from_area_ratio(g, a60, Branch::Supersonic), &["MACH_MAX"]);
        let sub = mach_from_area_ratio(g, a60, Branch::Subsonic).unwrap();
        assert!(sub > 0.0 && sub < 1.0, "subsonic root {sub}");
        close_rel("subsonic root of the M 60 ratio", area_ratio(g, sub), a60, 1e-13);
    }

    #[test]
    fn normal_shock_matches_naca_93_96_99() {
        let tol = 4e-15;
        let s2 = normal_shock(1.4, 2.0).unwrap();
        close_rel("p2/p1 M2", s2.p2_p1, 4.5, tol);
        close_rel("m2 M2", s2.m2, 0.5773502691896258, tol);
        close_rel("pt2/pt1 M2", s2.pt2_pt1, 0.7208738614847454, tol);
        // The plan's printed digits.
        assert!((s2.m2 - 0.57735).abs() < 5e-6, "m2 {}", s2.m2);
        assert!((s2.pt2_pt1 - 0.72087).abs() < 5e-6, "pt2/pt1 {}", s2.pt2_pt1);
        let s3 = normal_shock(1.4, 3.0).unwrap();
        close_rel("p2/p1 M3", s3.p2_p1, 10.333333333333334, tol);
        close_rel("m2 M3", s3.m2, 0.4751909633114915, tol);
        close_rel("pt2/pt1 M3", s3.pt2_pt1, 0.32834388819073695, tol);
        let sg = normal_shock(5.0 / 3.0, 2.0).unwrap();
        close_rel("p2/p1 g5/3", sg.p2_p1, 4.75, tol);
        close_rel("m2 g5/3", sg.m2, 0.6069769786668839, tol);
        close_rel("pt2/pt1 g5/3", sg.pt2_pt1, 0.7629822631945855, tol);
    }

    #[test]
    fn normal_shock_is_the_identity_at_mach_1_and_refuses_an_expansion() {
        let s = normal_shock(1.4, 1.0).unwrap();
        assert_eq!(s.m2, 1.0);
        assert_eq!(s.p2_p1, 1.0);
        assert_eq!(s.pt2_pt1, 1.0);
        refused(normal_shock(1.4, 0.9), &["expansion"]);
        refused(normal_shock(1.4, f64::INFINITY), &["not finite"]);
        refused(normal_shock(1.4, f64::NAN), &["not finite"]);
    }

    #[test]
    fn normal_shock_inverse_from_the_total_pressure_ratio() {
        let g = 1.4;
        let pt = normal_shock(g, 2.0).unwrap().pt2_pt1;
        close_abs(
            "inverse of M2",
            normal_shock_mach_from_pt_ratio(g, pt).unwrap(),
            2.0,
            1e-12,
        );
        close_abs(
            "inverse of 0.5",
            normal_shock_mach_from_pt_ratio(g, 0.5).unwrap(),
            2.4975410565602847,
            1e-12,
        );
        assert_eq!(normal_shock_mach_from_pt_ratio(g, 1.0).unwrap(), 1.0);
        refused(normal_shock_mach_from_pt_ratio(g, 0.0), &["(0, 1]"]);
        refused(normal_shock_mach_from_pt_ratio(g, 1.2), &["(0, 1]"]);
        refused(normal_shock_mach_from_pt_ratio(g, -0.1), &["(0, 1]"]);
        refused(normal_shock_mach_from_pt_ratio(g, f64::NAN), &["not finite"]);
        // A ratio below the one at MACH_MAX is refused by name.
        let floor = normal_shock(g, MACH_MAX).unwrap().pt2_pt1;
        refused(
            normal_shock_mach_from_pt_ratio(g, 0.5 * floor),
            &["MACH_MAX"],
        );
    }

    #[test]
    fn oblique_shock_mach_2_deflection_10_weak_branch() {
        let g = 1.4;
        let s = oblique_shock_weak(g, 2.0, 10f64.to_radians()).unwrap();
        let beta = s.shock_angle.to_degrees();
        println!(
            "CMP06 oblique beta_deg={beta} p2p1={} m2={}",
            s.p2_p1, s.m2
        );
        close_abs("beta M2 d10", beta, 39.31393184481887, 1e-9);
        assert!((beta - 39.31).abs() <= 0.01, "the plan's bar: beta {beta}");
        close_rel("p2/p1 M2 d10", s.p2_p1, 1.7065786040000334, 1e-12);
        assert!((s.p2_p1 - 1.7066).abs() <= 1e-4, "the plan's bar: {}", s.p2_p1);
        close_rel("m2 M2 d10", s.m2, 1.6405222290010812, 1e-12);
        assert_eq!(s.deflection, 10f64.to_radians());
        let s3 = oblique_shock_weak(g, 3.0, 20f64.to_radians()).unwrap();
        close_abs("beta M3 d20", s3.shock_angle.to_degrees(), 37.76363414837577, 1e-9);
        close_rel("p2/p1 M3 d20", s3.p2_p1, 3.77125746308266, 1e-12);
        close_rel("m2 M3 d20", s3.m2, 1.994131665564559, 1e-12);
    }

    #[test]
    fn oblique_at_ninety_degrees_is_the_normal_shock() {
        for m1 in [2.0, 3.0] {
            let o = oblique_from_shock_angle(1.4, m1, FRAC_PI_2).unwrap();
            let n = normal_shock(1.4, m1).unwrap();
            close_rel(&format!("p2/p1 M{m1}"), o.p2_p1, n.p2_p1, 4e-15);
            close_rel(&format!("m2 M{m1}"), o.m2, n.m2, 4e-15);
            assert!(o.deflection.abs() <= 1e-15, "deflection {}", o.deflection);
            assert_eq!(o.shock_angle, FRAC_PI_2);
        }
    }

    #[test]
    fn max_deflection_at_mach_2() {
        let (theta, delta) = max_deflection(1.4, 2.0).unwrap();
        close_abs("angle of max deflection", theta.to_degrees(), 64.6689798305795, 1e-5);
        close_abs("max deflection", delta.to_degrees(), 22.973531760937938, 1e-9);
        // The maximum is attained: the weak root at it is Ok, near the angle.
        let s = oblique_shock_weak(1.4, 2.0, delta).unwrap();
        close_abs(
            "weak root at the maximum",
            s.shock_angle.to_degrees(),
            theta.to_degrees(),
            1e-4,
        );
    }

    #[test]
    fn oblique_shock_refuses_by_name() {
        let g = 1.4;
        refused(oblique_shock_weak(g, 2.0, 23f64.to_radians()), &["detached"]);
        refused(oblique_shock_weak(g, 1.0, 0.1), &["supersonic"]);
        refused(oblique_from_shock_angle(g, 1.0, 1.0), &["supersonic"]);
        refused(max_deflection(g, 1.0), &["supersonic"]);
        refused(oblique_shock_weak(g, 2.0, -1f64.to_radians()), &["deflection"]);
        refused(
            oblique_from_shock_angle(g, 2.0, 20f64.to_radians()),
            &["shock angle"],
        );
        refused(
            oblique_from_shock_angle(g, 2.0, FRAC_PI_2 + 0.1),
            &["shock angle"],
        );
        refused(oblique_shock_weak(g, 2.0, f64::NAN), &["not finite"]);
        // No turn at all: the Mach wave, with no pressure rise.
        let s = oblique_shock_weak(g, 2.0, 0.0).unwrap();
        close_abs("Mach angle", s.shock_angle.to_degrees(), 30.0, 1e-9);
        assert!((s.p2_p1 - 1.0).abs() <= 1e-12, "p2/p1 {}", s.p2_p1);
    }

    #[test]
    fn shocked_nozzle_inverse_round_trips_to_1e_10() {
        let g = 1.4;
        let mut worst = 0.0f64;
        let mut cases: Vec<(f64, f64)> = [1.05, 1.2, 1.5, 1.8, 1.95, 2.0]
            .iter()
            .map(|&a| (a, 2.0))
            .collect();
        cases.push((2.5, 3.0));
        for (a_s, a_e) in cases {
            let pb = shocked_nozzle_back_pressure(g, a_s, a_e).unwrap();
            let back = shocked_nozzle(g, pb, a_e).unwrap();
            let err = (back.as_astar - a_s).abs();
            worst = worst.max(err);
            assert!(
                err <= 1e-10,
                "round trip as={a_s} ae={a_e}: pb {pb} gives as {} (err {err:e})",
                back.as_astar
            );
        }
        println!("CMP06 nozzle max_roundtrip_err={worst:e}");
        // The oracle values: the shock at 1.5 in a nozzle of 2.
        let pb = shocked_nozzle_back_pressure(g, 1.5, 2.0).unwrap();
        close_rel("pb as1.5 ae2", pb, 0.7044519779223286, 1e-12);
        let s = shocked_nozzle(g, pb, 2.0).unwrap();
        close_rel("m1", s.m1, 1.8541235267373254, 1e-10);
        close_rel("m_e", s.m_e, 0.40419695203975395, 1e-10);
        close_rel("pt2/pt1", s.pt2_pt1, 0.7883594290779655, 1e-10);
        close_rel(
            "pb as1.05 ae2",
            shocked_nozzle_back_pressure(g, 1.05, 2.0).unwrap(),
            0.9219469785399678,
            1e-12,
        );
        close_rel(
            "pb as2 ae2",
            shocked_nozzle_back_pressure(g, 2.0, 2.0).unwrap(),
            0.5134007279957163,
            1e-12,
        );
        close_rel(
            "pb as2.5 ae3",
            shocked_nozzle_back_pressure(g, 2.5, 3.0).unwrap(),
            0.4658980070459114,
            1e-12,
        );
    }

    #[test]
    fn nozzle_limits_and_regime_refusals() {
        let g = 1.4;
        let lim = nozzle_limits(g, 2.0).unwrap();
        close_rel("pb_choked", lim.pb_choked, 0.9371625024322055, 1e-12);
        close_rel("pb_shock_at_exit", lim.pb_shock_at_exit, 0.5134007279957163, 1e-12);
        close_rel("pb_design", lim.pb_design, 0.09393264573284488, 1e-12);
        // The two ends of the interval are inside it.
        let choked = shocked_nozzle(g, lim.pb_choked, 2.0).unwrap();
        close_abs("as at pb_choked", choked.as_astar, 1.0, 1e-6);
        let exit = shocked_nozzle(g, lim.pb_shock_at_exit, 2.0).unwrap();
        close_abs("as at pb_shock_at_exit", exit.as_astar, 2.0, 1e-9);
        // Outside it the shocked nozzle refuses, by name and by side.
        refused(
            shocked_nozzle(g, 0.95, 2.0),
            &["no normal shock inside the nozzle", "above"],
        );
        refused(
            shocked_nozzle(g, 0.3, 2.0),
            &["no normal shock inside the nozzle", "below"],
        );
        refused(nozzle_limits(g, 0.9), &["A_e/A_*"]);
        refused(shocked_nozzle(g, 0.7, 0.9), &["A_e/A_*"]);
        refused(shocked_nozzle_back_pressure(g, 1.5, 0.9), &["A_e/A_*"]);
        refused(shocked_nozzle_back_pressure(g, 2.5, 2.0), &["A_s/A_*"]);
        refused(shocked_nozzle_back_pressure(g, 0.9, 2.0), &["A_s/A_*"]);
        refused(nozzle_limits(g, f64::NAN), &["not finite"]);
        refused(shocked_nozzle(g, f64::NAN, 2.0), &["not finite"]);
    }

    #[test]
    fn every_entry_refuses_a_bad_gamma() {
        for g in [1.0, 0.5, f64::NAN, f64::INFINITY] {
            refused(mach_from_area_ratio(g, 2.0, Branch::Subsonic), &["gamma"]);
            refused(normal_shock(g, 2.0), &["gamma"]);
            refused(normal_shock_mach_from_pt_ratio(g, 0.5), &["gamma"]);
            refused(oblique_from_shock_angle(g, 2.0, 0.7), &["gamma"]);
            refused(max_deflection(g, 2.0), &["gamma"]);
            refused(oblique_shock_weak(g, 2.0, 0.1), &["gamma"]);
            refused(nozzle_limits(g, 2.0), &["gamma"]);
            refused(shocked_nozzle(g, 0.7, 2.0), &["gamma"]);
            refused(shocked_nozzle_back_pressure(g, 1.5, 2.0), &["gamma"]);
        }
    }
}
