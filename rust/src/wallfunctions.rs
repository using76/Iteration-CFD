// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Equilibrium wall functions - SPEC-LIT §6.4.
//!
//! Written from:
//!   Launder & Spalding, *Comput. Methods Appl. Mech. Eng.* 3 (1974) 269-289
//!     - the equilibrium near-wall relations for `nu_t`, `epsilon` and `G`
//!   Spalding, *J. Appl. Mech.* 28 (1961) 455 - one law of the wall across
//!     the sublayer, the buffer layer and the log layer, instead of two
//!     branches and a switch
//!   Kader, *Int. J. Heat Mass Transfer* 24 (1981) 1541-1544 - the
//!     exponential blending function that realises that single law explicitly
//!   Menter & Esch, 16th Brazilian Congress of Mechanical Engineering (2001)
//!     - root-sum-square blending of a viscous and a logarithmic branch
//!   Popovac & Hanjalic, *Flow Turbul. Combust.* 78 (2007) 177-202 - compound
//!     wall treatment; named by SPEC-LIT §6.4 as a precedent for blending
//!   Wilcox, *Turbulence Modeling for CFD* - `omega = 6 nu/(beta_1 y^2)`
//!   Cebeci & Bradshaw, *Momentum Transfer in Boundary Layers*, Hemisphere
//!     (1977) - the rough-wall downshift `dB(Ks+, Cs)` ([`roughness_db`])
//!   ofgpu `SPEC-LIT.md` §6.4. The two items marked *DESIGN* there - the
//!     blending, and the treatment of the wall-adjacent CELL - are ours.
//!   ofgpu `SPEC-LIT.md` §15.1 - the §15.1 Newton solve for `u_tau`
//!     ([`u_tau_newton`]), the `nutU` family's own reason to exist
//!   ofgpu `SPEC-LIT.md` §15.2 - `nutLowRe` is `nu_t = 0`, and §15.5 - each
//!     field's OWN patch type decides what happens to it at the wall
//!   ofgpu `SPEC-LIT.md` §15.3/§29.2 - the downshift itself, and `E_eff`, the
//!     single substitution every relation containing `ln(E y+)` shifts
//!     through; `Ks+` iterates with `u_tau` in the Newton above
//! No GPL-licensed source was consulted.
//!
//! # The two design decisions, in one paragraph each
//!
//! **Blending.** The log law and its viscous limit disagree by a large factor
//! at `y+_lam`, so switching between them makes a first cell that sits near
//! `y+_lam` oscillate from one outer iteration to the next. Nothing here
//! switches. `nu_t` at the wall comes from a single blended `u+`,
//! `u+ = y+ e^Gamma + ln(E y+)/kappa · e^{1/Gamma}` with Kader's
//! `Gamma = -0.01 (y+)^4/(1 + 5 y+)`; `epsilon` and `omega` are the
//! root-sum-square of their two branches; and `G` carries the log-branch
//! weight `e^{1/Gamma}` so that it vanishes in the sublayer, where there is no
//! turbulent stress left to do work. `cuda/wallfunctions.cu` derives all
//! three at length and states which parts are ours.
//!
//! **The wall-adjacent cell.** The relations prescribe values at the first
//! *cell*, so its matrix row is fixed and decoupled ([`constrain_wall_cells`]).
//! A cell with several wall faces averages them weighted by face area, which
//! is the weighting the relations' own flux interpretation implies.
//!
//! # Why the host mirrors exist
//!
//! Every device expression in `cuda/wallfunctions.cu` has a `Scalar -> Scalar`
//! twin here ([`u_plus`], [`nut_wall`], [`epsilon_wall`], [`omega_wall`],
//! [`production_wall`]). They are not duplicated for convenience: they are
//! what lets the continuity of the blend - the entire point of the design - be
//! tested on a machine with no GPU, and
//! `tests::device_agrees_with_the_host_law` pins the two together where there
//! is one.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::GpuVectorField;
use crate::ldu::GpuLduMatrix;
use crate::ldu_ops::{set_values, LduKernels};
use crate::mesh::{GpuMesh, HostMesh};
use crate::{Label, Scalar};

/// Read from `constant/momentumTransport`; defined in [`crate::io::case`]
/// because that is where it is parsed, re-exported here because this is where
/// its contents are obeyed.
pub use crate::io::case::WallFunctionCoeffs;

// ==========================================================================
//  y+_lam
// ==========================================================================

/// The `y+` at which the linear and logarithmic branches of the law of the
/// wall meet: the root of `y+ = ln(E y+)/kappa` (SPEC-LIT §6.4).
///
/// Solved by fixed-point iteration on `y <- ln(E y)/kappa`, never hard-coded.
/// The map is a contraction near the root - its derivative is `1/(kappa y)`,
/// about `0.21` at `y ~ 11.5` for the standard constants - so plain iteration
/// converges geometrically from any sensible start, and 11 is one.
///
/// The value is a *diagnostic* in this implementation rather than a switch:
/// nothing branches on it, because nothing here has two branches to choose
/// between. It is reported, and it is what the continuity test is centred on.
///
/// `max(E y, 1)` inside the logarithm keeps the iterate non-negative for an
/// absurd `E`; with any physical `E > 1` it never binds.
pub fn compute_y_plus_lam(kappa: Scalar, e: Scalar) -> Scalar {
    // A non-physical pair has no root to find. Returning the standard value
    // rather than a NaN keeps a mis-typed dictionary from poisoning a field.
    if !(kappa > 0.0) || !(e > 1.0) {
        return 11.53;
    }

    let mut y: Scalar = 11.0;

    for _ in 0..200 {
        let next = ((e * y).max(1.0)).ln() / kappa;
        let done = (next - y).abs() <= 1e-14 * next.abs().max(1.0);
        y = next.max(1e-6);
        if done {
            break;
        }
    }

    y
}

// ==========================================================================
//  The blended law of the wall - host mirrors of cuda/wallfunctions.cu
// ==========================================================================

/// Kader's blending exponent, `Gamma = -a (y+)^4/(1 + b y+)` with `a = 0.01`,
/// `b = 5`. Strictly negative for `y+ > 0`, tends to `0-` at the wall and to
/// `-inf` in the log layer, which is all the blend depends on.
#[inline]
pub fn blend_gamma(y_plus: Scalar) -> Scalar {
    let y2 = y_plus * y_plus;
    -0.01 * y2 * y2 / (1.0 + 5.0 * y_plus)
}

/// `e^{1/Gamma}`: the weight of the logarithmic branch. Zero at the wall, one
/// far from it. Guarded exactly as the kernel is, so the two agree bit for
/// bit at `y+ = 0`.
#[inline]
pub fn log_weight(gamma: Scalar) -> Scalar {
    if gamma < -1e-30 {
        (1.0 / gamma).exp()
    } else {
        0.0
    }
}

/// `u+` from `y+`, continuous from the wall to the log layer.
///
/// Reduces to `y+` as `y+ -> 0` and to `ln(E y+)/kappa` as `y+ -> inf`,
/// exactly, and dips below both where they cross - which is what a measured
/// buffer-layer profile does.
#[inline]
pub fn u_plus(y_plus: Scalar, kappa: Scalar, e: Scalar) -> Scalar {
    let gamma = blend_gamma(y_plus);
    let u_log = (e * y_plus).max(1.0).ln() / kappa;
    y_plus * gamma.exp() + u_log * log_weight(gamma)
}

/// `nu_t` at a wall face, from the blended law: `nu (y+/u+ - 1)`, floored at
/// zero.
#[inline]
pub fn nut_wall(y_plus: Scalar, nu: Scalar, kappa: Scalar, e: Scalar) -> Scalar {
    if !(y_plus > 0.0) {
        return 0.0;
    }
    let up = u_plus(y_plus, kappa, e);
    if !(up > 0.0) {
        return 0.0;
    }
    (nu * (y_plus / up - 1.0)).max(0.0)
}

/// `y+ = C_mu^{1/4} y sqrt(k) / nu` (SPEC-LIT §6.4).
#[inline]
pub fn y_plus_of(k: Scalar, y: Scalar, nu: Scalar, cmu: Scalar) -> Scalar {
    cmu.powf(0.25) * y * k.max(0.0).sqrt() / nu
}

/// `epsilon` in the wall-adjacent cell: the root-sum-square blend of
/// `C_mu^{3/4} k^{3/2}/(kappa y)` and `2 k nu / y^2`.
#[inline]
pub fn epsilon_wall(k: Scalar, y: Scalar, nu: Scalar, kappa: Scalar, cmu: Scalar) -> Scalar {
    let kc = k.max(0.0);
    let e_log = cmu.powf(0.75) * kc * kc.sqrt() / (kappa * y);
    let e_vis = 2.0 * kc * nu / (y * y);
    (e_log * e_log + e_vis * e_vis).sqrt()
}

/// `omega` in the wall-adjacent cell: the root-sum-square blend of
/// `sqrt(k)/(C_mu^{1/4} kappa y)` and Wilcox's `6 nu/(beta_1 y^2)`.
#[inline]
pub fn omega_wall(
    k: Scalar,
    y: Scalar,
    nu: Scalar,
    kappa: Scalar,
    cmu: Scalar,
    beta1: Scalar,
) -> Scalar {
    let w_log = k.max(0.0).sqrt() / (cmu.powf(0.25) * kappa * y);
    let w_vis = 6.0 * nu / (beta1 * y * y);
    (w_log * w_log + w_vis * w_vis).sqrt()
}

/// The blended production in the wall-adjacent cell:
///
/// ```text
/// G = e^{1/Gamma} · (nu_t,w + nu) · |du/dy|_w · C_mu^{1/4} sqrt(k)/(kappa y)
/// ```
///
/// The bracket is SPEC-LIT §6.4's log-layer relation; the leading weight is
/// ours, and takes `G` smoothly to zero in the viscous sublayer, where the
/// substitution of the log-layer mean shear that the relation rests on is not
/// valid and the physical production is zero.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn production_wall(
    y_plus: Scalar,
    nut_w: Scalar,
    nu: Scalar,
    mag_grad_u_w: Scalar,
    k: Scalar,
    y: Scalar,
    kappa: Scalar,
    cmu: Scalar,
) -> Scalar {
    let shear_log = cmu.powf(0.25) * k.max(0.0).sqrt() / (kappa * y);
    log_weight(blend_gamma(y_plus)) * (nut_w + nu) * mag_grad_u_w * shear_log
}

// ==========================================================================
//  Rough walls - SPEC-LIT §15.3, completed by §29.2
// ==========================================================================
//
// Written from:
//   Cebeci & Bradshaw, *Momentum Transfer in Boundary Layers*, Hemisphere
//     (1977) - the downshift `dB(Ks+, Cs)` below, all three regimes
//   ofgpu `SPEC-LIT.md` §15.3 (the downshift itself) and §29.2 (`E_eff`, and
//     that `Ks+` iterates with `u_tau` in the §15.1 Newton)
// No GPL-licensed source was consulted.
//
// Every relation of §6.4 above that contains `ln(E y+)` - which, in this
// implementation, is exactly [`u_plus`]/[`nut_wall`] and nothing else -
// shifts by replacing `E` with `E_eff = E exp(-kappa dB)`. `epsilon_wall` and
// `omega_wall` have no `E` in them to begin with (SPEC-LIT §6.4's log-layer
// relations for them are pure functions of `k`), so roughness reaches them
// only indirectly, through the wall production `G`, which already reads back
// whatever `nut_wall`/`nut_wall_u` wrote to the boundary value.

/// `Ks+ = Cs Ks u_tau / nu` (SPEC-LIT §15.3). `u_tau <= 0` or `Ks <= 0` give
/// `Ks+ = 0`, i.e. hydraulically smooth.
#[inline]
pub fn ks_plus_of(ks: Scalar, cs: Scalar, u_tau: Scalar, nu: Scalar) -> Scalar {
    cs * ks.max(0.0) * u_tau.max(0.0) / nu
}

/// The Cebeci & Bradshaw (1977) roughness downshift `dB(Ks+, Cs)` - SPEC-LIT
/// §15.3, all three regimes in one function, with the transitional band
/// sine-blended into the other two exactly as the specification writes it.
///
/// Continuous at both seams, and for reasons that belong to the published
/// constants rather than to this implementation: at `Ks+ = 2.25`,
/// `ln(2.25) = 0.81093...` is already so close to the constant `0.811` that
/// the sine factor is of order `1e-4` there, and at `Ks+ = 90`,
/// `(90 - 2.25)/87.75 = 1` exactly, so the transitional branch's log argument
/// matches the fully-rough branch's `1 + Cs·90` while
/// `0.4258 (ln 90 - 0.811) = pi/2` exactly, so the sine factor is 1. Neither
/// seam is rounded to make it land exactly on zero here - the constants are
/// kept at the literature's own precision, and `tests` measures how small the
/// resulting step actually is rather than asserting it is nothing.
#[inline]
pub fn roughness_db(ks_plus: Scalar, cs: Scalar, kappa: Scalar) -> Scalar {
    if !(ks_plus > 2.25) {
        0.0
    } else if ks_plus < 90.0 {
        let arg = (ks_plus - 2.25) / 87.75 + cs * ks_plus;
        let sine = (0.4258 * (ks_plus.ln() - 0.811)).sin();
        arg.max(1e-300).ln() * sine / kappa
    } else {
        (1.0 + cs * ks_plus).ln() / kappa
    }
}

/// `E_eff = E exp(-kappa dB)` (SPEC-LIT §29.2) - the one substitution every
/// relation containing `ln(E y+)` shifts through. `dB = 0` gives `E_eff = E`
/// exactly, which is what makes `Ks -> 0` reproduce the smooth wall to
/// round-off (the §22 gate).
#[inline]
pub fn e_eff(e: Scalar, kappa: Scalar, db: Scalar) -> Scalar {
    e * (-kappa * db).exp()
}

/// `u_tau` implied directly by `k`: `Cmu^{1/4} sqrt(k)`, the same friction
/// velocity `y+ = Cmu^{1/4} y sqrt(k)/nu` is built from.
///
/// `nutk`'s `Ks+` uses this rather than iterating: `y+` there already comes
/// from `k` alone, so this `u_tau` is already known and no Newton solve is
/// needed - unlike `nutU`, where `u_tau` is the Newton's own unknown and
/// `Ks+` must iterate alongside it ([`u_tau_newton`], SPEC-LIT §29.2).
#[inline]
pub fn u_tau_from_k(k: Scalar, cmu25: Scalar) -> Scalar {
    cmu25 * k.max(0.0).sqrt()
}

/// `nu_t` at a wall face from the `nutk` family, with the roughness downshift
/// folded into `E` (SPEC-LIT §15.3/§29.2). `ks = 0` reproduces [`nut_wall`]
/// exactly.
#[inline]
pub fn nut_wall_rough_k(
    y_plus: Scalar,
    k: Scalar,
    nu: Scalar,
    kappa: Scalar,
    e: Scalar,
    cmu25: Scalar,
    ks: Scalar,
    cs: Scalar,
) -> Scalar {
    let u_tau = u_tau_from_k(k, cmu25);
    let ks_plus = ks_plus_of(ks, cs, u_tau, nu);
    let db = roughness_db(ks_plus, cs, kappa);
    nut_wall(y_plus, nu, kappa, e_eff(e, kappa, db))
}

/// Spalding's law (SPEC-LIT §15.1) inverted for `u_tau`, extended by §29.2
/// with the roughness downshift: `Ks+ = Cs Ks u_tau/nu` is recomputed from
/// the CURRENT `u_tau` iterate every step, exactly like `u+` itself, so the
/// `u_tau` this returns and its own `Ks+`/`dB` are mutually consistent at
/// convergence. `ks = 0` keeps `E_eff = E` on every iteration regardless of
/// `u_tau`, so this collapses to the plain smooth Newton bit for bit - the
/// §22 gate.
///
/// ```text
/// F(u_tau) = y u_tau/nu - u+ - (1/E_eff)[e^{kappa u+} - 1 - kappa u+
///                                         - (kappa u+)^2/2 - (kappa u+)^3/6]
/// u+ = |U_parallel| / u_tau
/// ```
///
/// `dF/du_tau` is taken with `E_eff` frozen at the value the current
/// iterate's `Ks+` gives - the derivative SPEC-LIT leaves to calculus, not a
/// literature formula - which is exactly what "iterates with u_tau" means:
/// `Ks+`/`dB`/`E_eff` are refreshed once per outer step rather than
/// differentiated through.
///
/// *DESIGN* (SPEC-LIT §15.1): the viscous guess `u_tau = sqrt(nu |U|/y)`, 10
/// iterations, relative tolerance `1e-6`, clamped to `u_tau >= 0`.
/// `|U_parallel| = 0` returns `0` with no iteration - there is nothing to
/// solve for.
pub fn u_tau_newton(
    u_mag: Scalar,
    y: Scalar,
    nu: Scalar,
    kappa: Scalar,
    e: Scalar,
    ks: Scalar,
    cs: Scalar,
) -> Scalar {
    if !(u_mag > 0.0) || !(y > 0.0) {
        return 0.0;
    }

    let mut u_tau: Scalar = (nu * u_mag / y).max(1e-300).sqrt();

    for _ in 0..10 {
        let ks_plus = ks_plus_of(ks, cs, u_tau, nu);
        let db = roughness_db(ks_plus, cs, kappa);
        let e_eff = e_eff(e, kappa, db);

        let u_plus = u_mag / u_tau;
        let ku = kappa * u_plus;
        let euk = ku.exp();
        let poly = euk - 1.0 - ku - ku * ku * 0.5 - ku * ku * ku / 6.0;
        let f = y * u_tau / nu - u_plus - poly / e_eff;

        let dpoly = kappa * (euk - 1.0 - ku - ku * ku * 0.5);
        let df = y / nu + (u_plus / u_tau) * (1.0 + dpoly / e_eff);

        if !(df.abs() > 0.0) {
            break;
        }

        let next = (u_tau - f / df).max(1e-300);
        let done = (next - u_tau).abs() <= 1e-6 * next.abs().max(1e-300);
        u_tau = next;
        if done {
            break;
        }
    }

    u_tau.max(0.0)
}

/// `nu_t,w = max(0, u_tau^2 y / |U_parallel| - nu)` (SPEC-LIT §15.1), from a
/// `u_tau` already solved for - by [`u_tau_newton`], smooth or rough.
#[inline]
pub fn nut_wall_u(u_tau: Scalar, y: Scalar, nu: Scalar, u_mag: Scalar) -> Scalar {
    if !(u_mag > 0.0) {
        return 0.0;
    }
    (u_tau * u_tau * y / u_mag - nu).max(0.0)
}

/// `nu_t` at a wall face from the `nutU` family: [`u_tau_newton`] then
/// [`nut_wall_u`]. `ks = 0` reproduces the plain §15.1 `nutU` exactly.
#[inline]
pub fn nut_wall_rough_u(
    u_mag: Scalar,
    y: Scalar,
    nu: Scalar,
    kappa: Scalar,
    e: Scalar,
    ks: Scalar,
    cs: Scalar,
) -> Scalar {
    let u_tau = u_tau_newton(u_mag, y, nu, kappa, e, ks, cs);
    nut_wall_u(u_tau, y, nu, u_mag)
}

// ==========================================================================
//  The Werner-Wengle LES wall model - SPEC-LIT §30.1
// ==========================================================================
//
// Written from:
//   Werner & Wengle, "Large-eddy simulation of turbulent flow over and
//     around a cube in a plate channel", 8th Symp. Turb. Shear Flows (1991)
//   ofgpu `SPEC-LIT.md` §30.1
// No GPL-licensed source was consulted.
//
// An LES resolves the outer eddies and models only the sublayer the mesh
// cannot afford. Werner-Wengle replaces the RAS log law with an
// analytically-invertible power law, integrated over the first cell so the
// wall shear comes directly from the CELL-AVERAGED velocity an LES actually
// carries - no Newton iteration, unlike [`u_tau_newton`] above:
//
// ```text
// u+ = y+                      y+ <= 11.81
// u+ = A (y+)^B                y+  > 11.81,   A = 8.3,  B = 1/7
// ```
//
// Integrated across the first cell of height `h` and inverted for `tau_w`,
// with `|u_p|` the wall-parallel cell-average speed:
//
// ```text
// viscous:  |u_p| <= nu/(2h) A^{2/(1-B)}     ->  tau_w = 2 nu |u_p| / h
// power:    otherwise ->
//   tau_w = [ (1-B)/2 A^{(1+B)/(1-B)} (nu/h)^{1+B}
//             + (1+B)/A (nu/h)^B |u_p| ]^{2/(1+B)}
// ```
//
// **Continuity at the branch point.** Substituting `u_p = u_c` (the branch
// speed) into the power form: writing `nu_h = nu/h`, both terms of the
// bracket share the exponent `(1+B)/(1-B)` on `A` (the power branch's own
// `(1+B)/(1-B)` and the viscous exponent `2/(1-B) - 1` are the same number),
// so the bracket collapses to `A^{(1+B)/(1-B)} nu_h^{1+B} [(1-B)/2 + (1+B)/2]
// = A^{(1+B)/(1-B)} nu_h^{1+B}`, and raising that to `2/(1+B)` gives
// `A^{2/(1-B)} nu_h^2` - exactly the viscous branch's own value at `u_c`,
// `2 nu_h u_c = 2 nu_h (nu_h/2) A^{2/(1-B)} = nu_h^2 A^{2/(1-B)}`. The two
// sides of the branch are the same function evaluated at the same point, not
// two different functions that happen to agree - `tests::the_ww_power_branch_
// reduces_to_the_viscous_one_at_the_branch_point` is that algebra run as a
// numerical check.
pub const WW_A: Scalar = 8.3;
pub const WW_B: Scalar = 1.0 / 7.0;

/// The wall-parallel cell-average speed at which the two branches meet:
/// `nu/(2h) A^{2/(1-B)}` (SPEC-LIT §30.1).
#[inline]
pub fn ww_branch_speed(nu: Scalar, h: Scalar) -> Scalar {
    if !(h > 0.0) {
        return 0.0;
    }
    (nu / (2.0 * h)) * WW_A.powf(2.0 / (1.0 - WW_B))
}

/// `tau_w` from the integrated-and-inverted Werner-Wengle power law -
/// SPEC-LIT §30.1. Continuous at the branch point by construction (see the
/// module section above); [`tests`] evaluates both sides rather than relying
/// on the branch itself to hide a discontinuity.
#[inline]
pub fn tau_w_werner_wengle(u_p: Scalar, h: Scalar, nu: Scalar) -> Scalar {
    let u_p = u_p.max(0.0);
    if !(h > 0.0) || !(nu > 0.0) {
        return 0.0;
    }

    if u_p <= ww_branch_speed(nu, h) {
        return 2.0 * nu * u_p / h;
    }

    let a = WW_A;
    let b = WW_B;
    let nu_h = nu / h;
    let t1 = 0.5 * (1.0 - b) * a.powf((1.0 + b) / (1.0 - b)) * nu_h.powf(1.0 + b);
    let t2 = ((1.0 + b) / a) * nu_h.powf(b) * u_p;
    (t1 + t2).powf(2.0 / (1.0 + b))
}

/// `nu_t,w = tau_w h/|u_p| - nu`, clamped at zero - SPEC-LIT §30.1: chosen so
/// the wall face's diffusive flux `(nu + nu_t,w)|u_p|/h` reproduces `tau_w`
/// exactly, whichever branch produced it.
#[inline]
pub fn nut_wall_werner_wengle(tau_w: Scalar, h: Scalar, u_p: Scalar, nu: Scalar) -> Scalar {
    if !(u_p > 0.0) || !(h > 0.0) {
        return 0.0;
    }
    (tau_w * h / u_p - nu).max(0.0)
}

/// `u_tau = sqrt(tau_w)` - the substitution SPEC-LIT §30.1 wires into the
/// thermal wall function under LES ([`thermal_wall_ref_grad_from_u_tau`]),
/// in place of the RAS `Cmu^{1/4} sqrt(k)` [`u_tau_of`] computes.
#[inline]
pub fn u_tau_werner_wengle(tau_w: Scalar) -> Scalar {
    tau_w.max(0.0).sqrt()
}

// ==========================================================================
//  Kernels
// ==========================================================================

/// Every entry point in `cuda/wallfunctions.cu`, resolved once.
struct WallKernels {
    nut_wall: CudaFunction,
    y_plus: CudaFunction,
    epsilon_cell: CudaFunction,
    omega_cell: CudaFunction,
    mark_fixed: CudaFunction,
}

impl WallKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::WALLFUNCTIONS)?;
        Ok(Self {
            nut_wall: k.func("wfNutWall")?,
            y_plus: k.func("wfYPlus")?,
            epsilon_cell: k.func("wfEpsilonWallCell")?,
            omega_cell: k.func("wfOmegaWallCell")?,
            mark_fixed: k.func("wfMarkFixed")?,
        })
    }
}

// ==========================================================================
//  WallData
// ==========================================================================

/// Which cells the wall functions own, and the wall faces of each.
///
/// Built once at setup from the host mesh and a per-boundary-face flag saying
/// whether that face carries a wall function. The flag comes from the *field*,
/// not from the mesh: a `wall` patch whose `epsilon` entry says `fixedValue`
/// is not a wall-function patch, and a case is entitled to say so.
///
/// The layout is a CSR over wall cells, mirroring the mesh's own cell -> face
/// map and for the same reason (SPEC-LIT and the crate's gather rule): every
/// kernel that averages over a cell's wall faces is one thread per cell
/// walking its own entries, so the average is deterministic however the
/// blocks are scheduled.
pub struct WallData {
    /// Distinct cells whose `epsilon`/`omega` is pinned to the near-wall
    /// relation.
    pub n_wall_cells: usize,
    /// Boundary faces belonging to those cells.
    pub n_wall_faces: usize,

    /// `[n_wall_cells]` cell indices, ascending.
    pub wall_cells: DevBuf<Label>,
    /// `[n_wall_cells + 1]` CSR offsets into [`Self::wf_face`].
    pub wf_offset: DevBuf<Label>,
    /// `[n_wall_faces]` boundary-face indices, grouped by cell.
    pub wf_face: DevBuf<Label>,

    /// Faces that get a wall value for `nu_t` - from `nut`'s OWN patch types,
    /// which are not the same set as [`Self::wf_face`].
    ///
    /// SPEC-LIT §15.5. Sharing one list between the two was the bug: a case
    /// with `nut = nutLowReWallFunction` (or `fixedValue 0`) and
    /// `epsilon = epsilonWallFunction` - the standard resolved-sublayer setup
    /// - had a wall function written onto `nu_t` that it had explicitly
    /// refused, and no diagnostic said so.
    pub n_nut_faces: usize,
    /// `[n_nut_faces]` boundary-face indices, ascending.
    pub nut_face: DevBuf<Label>,
    /// `[n_nut_faces]` `1` where the face is in the `nutU` family (SPEC-LIT
    /// §15.1) - velocity-based `y+`/`u_tau` - rather than `nutk`'s `k`-based
    /// one, aligned with [`Self::nut_face`].
    pub nut_u_based: DevBuf<Label>,
    /// `[n_nut_faces]` sand-grain height, aligned with [`Self::nut_face`].
    /// Zero on every smooth face - SPEC-LIT §15.3/§29.2.
    pub nut_ks: DevBuf<Scalar>,
    /// `[n_nut_faces]` the roughness constant, aligned with
    /// [`Self::nut_face`]. Meaningful only where `nut_ks > 0`.
    pub nut_cs: DevBuf<Scalar>,

    /// `[n_wall_cells]` the value `epsilon` (or `omega`) is pinned to.
    ///
    /// Written by [`Self::update_epsilon`] / [`Self::update_omega`] and read
    /// by [`constrain_wall_cells`]. Public because the validation binary
    /// writes it directly to check that `setValues` does what it claims.
    pub wall_cell_value: DevBuf<Scalar>,

    /// `[n_wall_faces]` scratch for [`Self::update_y_plus`].
    pub y_plus: DevBuf<Scalar>,

    k: WallKernels,
}

impl WallData {
    /// Invert the two face sets into the wall-cell CSR and the `nu_t` list.
    ///
    /// Both slices are indexed by *flattened boundary face*, the same indexing
    /// `HostMesh::b_face_cells` uses, and each must have exactly
    /// `n_boundary_faces` entries - a shorter slice would silently drop the
    /// last patch.
    ///
    /// `faces.constrained_cells` comes from `epsilon`/`omega`'s patch types
    /// and `faces.nut` from `nut`'s, and they are deliberately independent -
    /// SPEC-LIT §15.5. `roughness` is `nut`'s own `Ks`/`Cs`/family data
    /// (SPEC-LIT §15.3/§29.2); a case with none passes
    /// [`crate::field_setup::NutRoughness::none`].
    pub fn build(
        gpu: &Gpu,
        m: &HostMesh,
        faces: &crate::field_setup::WallFaces,
        roughness: &crate::field_setup::NutRoughness,
    ) -> Result<Self> {
        let is_wall_function: &[bool] = &faces.constrained_cells;

        for (what, v) in [
            ("constrained-cell", &faces.constrained_cells),
            ("nut wall", &faces.nut),
        ] {
            if v.len() != m.n_boundary_faces {
                return Err(Error::Config(format!(
                    "WallData::build: the {what} flag has {} entries, the \
                     mesh has {} boundary faces",
                    v.len(),
                    m.n_boundary_faces
                )));
            }
        }

        for (what, n) in [
            ("nut u-based", roughness.u_based.len()),
            ("nut Ks", roughness.ks.len()),
            ("nut Cs", roughness.cs.len()),
        ] {
            if n != m.n_boundary_faces {
                return Err(Error::Config(format!(
                    "WallData::build: the {what} flag has {n} entries, the \
                     mesh has {} boundary faces",
                    m.n_boundary_faces
                )));
            }
        }

        // Faces first, grouped by cell. Ascending face index within a cell and
        // ascending cell index overall, so the gather order is fixed and the
        // area average is bitwise reproducible from run to run.
        let mut per_cell: Vec<Vec<Label>> = vec![Vec::new(); m.n_cells];
        let mut n_faces = 0usize;

        for (bf, &on) in is_wall_function.iter().enumerate() {
            if !on {
                continue;
            }
            let c = m.b_face_cells[bf];
            if c < 0 || c as usize >= m.n_cells {
                return Err(Error::Config(format!(
                    "WallData::build: boundary face {bf} names cell {c}, which \
                     is outside [0, {})",
                    m.n_cells
                )));
            }
            per_cell[c as usize].push(bf as Label);
            n_faces += 1;
        }

        let mut wall_cells: Vec<Label> = Vec::new();
        let mut wf_offset: Vec<Label> = vec![0];
        let mut wf_face: Vec<Label> = Vec::with_capacity(n_faces);

        for (c, faces) in per_cell.iter().enumerate() {
            if faces.is_empty() {
                continue;
            }
            wall_cells.push(c as Label);
            wf_face.extend_from_slice(faces);
            wf_offset.push(wf_face.len() as Label);
        }

        let n_cells_w = wall_cells.len();

        // The nu_t list is a flat set of faces, not a CSR: `wfNutWall` is one
        // thread per FACE and never averages over a cell.
        let nut_faces: Vec<Label> = faces
            .nut
            .iter()
            .enumerate()
            .filter(|(_, on)| **on)
            .map(|(bf, _)| bf as Label)
            .collect();
        let n_nut = nut_faces.len();

        // Gathered from `roughness`'s per-boundary-face vectors onto the SAME
        // indexing as `nut_faces`, so `nut_face[i]`/`nut_ks[i]`/`nut_cs[i]`
        // describe the one face throughout - SPEC-LIT §15.3/§29.2.
        let nut_u_based: Vec<Label> = nut_faces
            .iter()
            .map(|&bf| roughness.u_based[bf as usize] as Label)
            .collect();
        let nut_ks: Vec<Scalar> = nut_faces
            .iter()
            .map(|&bf| roughness.ks[bf as usize])
            .collect();
        let nut_cs: Vec<Scalar> = nut_faces
            .iter()
            .map(|&bf| roughness.cs[bf as usize])
            .collect();

        // A zero-length device allocation is an error rather than an empty
        // buffer, so a case with no wall functions still gets one element -
        // which no kernel ever reads, because every launcher returns early on
        // `n_wall_cells == 0`.
        let pad = |v: Vec<Label>| if v.is_empty() { vec![0 as Label] } else { v };
        let pad_s = |v: Vec<Scalar>| if v.is_empty() { vec![0.0 as Scalar] } else { v };

        Ok(Self {
            n_wall_cells: n_cells_w,
            n_wall_faces: n_faces,
            wall_cells: gpu.upload(&pad(wall_cells))?,
            wf_offset: gpu.upload(&wf_offset)?,
            wf_face: gpu.upload(&pad(wf_face))?,

            n_nut_faces: n_nut,
            nut_face: gpu.upload(&pad(nut_faces))?,
            nut_u_based: gpu.upload(&pad(nut_u_based))?,
            nut_ks: gpu.upload(&pad_s(nut_ks))?,
            nut_cs: gpu.upload(&pad_s(nut_cs))?,

            wall_cell_value: gpu.zeros(n_cells_w.max(1))?,
            y_plus: gpu.zeros(n_faces.max(1))?,
            k: WallKernels::new(gpu)?,
        })
    }

    /// `nu_t` on every face whose OWN patch type asked for a wall function,
    /// from the blended law of the wall.
    ///
    /// Writes into `nut_bf`, i.e. the *evaluated boundary values* of the `nut`
    /// field, at those faces only. Everything else in `nut_bf` is left alone,
    /// so the caller's earlier zero-gradient fill stands - and a
    /// `nutLowReWallFunction` face, which is not in this list, keeps the zero
    /// `turbNutBoundary` gave it (SPEC-LIT §15.2).
    ///
    /// Must run before [`Self::update_epsilon`] / [`Self::update_omega`],
    /// which read the value back to form `G`.
    ///
    /// `u` is only read on the faces [`Self::nut_u_based`] marks (the `nutU`
    /// family, SPEC-LIT §15.1) - a case with none of those still passes it,
    /// because a face's own family is a per-face device flag, not something
    /// the host can skip ahead of time.
    pub fn update_nut(
        &self,
        gpu: &Gpu,
        nut_bf: &mut DevBuf<Scalar>,
        k: &DevBuf<Scalar>,
        u: &GpuVectorField,
        m: &GpuMesh,
        wc: &WallFunctionCoeffs,
        nu: Scalar,
        k_min: Scalar,
    ) -> Result<()> {
        let n = self.n_nut_faces;
        if n == 0 {
            return Ok(());
        }
        self.check(nut_bf.len(), k.len(), m)?;
        expect_count(u.f.len(), m.n_cells, "U")?;
        expect_count(u.bf.len(), m.n_boundary_faces, "U boundary values")?;

        let cmu25 = wc.cmu.powf(0.25);
        let nl = n as Label;
        let f = self.k.nut_wall.clone();

        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(nut_bf)
                .arg(k)
                .arg(&u.f)
                .arg(&u.bf)
                .arg(&m.b_face_cells)
                .arg(&m.b_y)
                .arg(&m.b_sf)
                .arg(&m.b_mag_sf)
                .arg(&self.nut_face)
                .arg(&self.nut_u_based)
                .arg(&self.nut_ks)
                .arg(&self.nut_cs)
                .arg(&nu)
                .arg(&wc.kappa)
                .arg(&wc.e)
                .arg(&cmu25)
                .arg(&k_min)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// `y+` on every wall-function face, into [`Self::y_plus`].
    ///
    /// Diagnostic only - no model reads it. A user deciding whether the mesh
    /// is fit for a wall function does.
    pub fn update_y_plus(
        &mut self,
        gpu: &Gpu,
        k: &DevBuf<Scalar>,
        m: &GpuMesh,
        wc: &WallFunctionCoeffs,
        nu: Scalar,
        k_min: Scalar,
    ) -> Result<()> {
        let n = self.n_wall_faces;
        if n == 0 {
            return Ok(());
        }

        let cmu25 = wc.cmu.powf(0.25);
        let nl = n as Label;
        let f = self.k.y_plus.clone();

        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.y_plus)
                .arg(k)
                .arg(&m.b_face_cells)
                .arg(&m.b_y)
                .arg(&self.wf_face)
                .arg(&nu)
                .arg(&cmu25)
                .arg(&k_min)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// `epsilon` and `G` in every wall-adjacent cell, area-averaged over that
    /// cell's wall faces.
    ///
    /// Overwrites `epsilon` and `g` in those cells and records the same
    /// `epsilon` in [`Self::wall_cell_value`] for [`constrain_wall_cells`].
    #[allow(clippy::too_many_arguments)]
    pub fn update_epsilon(
        &mut self,
        gpu: &Gpu,
        epsilon: &mut DevBuf<Scalar>,
        g: &mut DevBuf<Scalar>,
        k: &DevBuf<Scalar>,
        u: &GpuVectorField,
        nut_bf: &DevBuf<Scalar>,
        m: &GpuMesh,
        wc: &WallFunctionCoeffs,
        nu: Scalar,
        k_min: Scalar,
    ) -> Result<()> {
        let n = self.n_wall_cells;
        if n == 0 {
            return Ok(());
        }
        self.check(nut_bf.len(), k.len(), m)?;
        expect_count(epsilon.len(), m.n_cells, "epsilon")?;
        expect_count(g.len(), m.n_cells, "G")?;

        let cmu25 = wc.cmu.powf(0.25);
        let cmu75 = wc.cmu.powf(0.75);
        let nl = n as Label;
        let f = self.k.epsilon_cell.clone();

        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(epsilon)
                .arg(g)
                .arg(&mut self.wall_cell_value)
                .arg(k)
                .arg(&u.f)
                .arg(&u.bf)
                .arg(nut_bf)
                .arg(&m.b_sf)
                .arg(&m.b_mag_sf)
                .arg(&m.b_y)
                .arg(&self.wall_cells)
                .arg(&self.wf_offset)
                .arg(&self.wf_face)
                .arg(&nu)
                .arg(&wc.kappa)
                .arg(&cmu25)
                .arg(&cmu75)
                .arg(&k_min)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// `omega` and `G` in every wall-adjacent cell. See
    /// [`Self::update_epsilon`]; only the near-wall relation differs.
    #[allow(clippy::too_many_arguments)]
    pub fn update_omega(
        &mut self,
        gpu: &Gpu,
        omega: &mut DevBuf<Scalar>,
        g: &mut DevBuf<Scalar>,
        k: &DevBuf<Scalar>,
        u: &GpuVectorField,
        nut_bf: &DevBuf<Scalar>,
        m: &GpuMesh,
        wc: &WallFunctionCoeffs,
        nu: Scalar,
        k_min: Scalar,
    ) -> Result<()> {
        let n = self.n_wall_cells;
        if n == 0 {
            return Ok(());
        }
        self.check(nut_bf.len(), k.len(), m)?;
        expect_count(omega.len(), m.n_cells, "omega")?;
        expect_count(g.len(), m.n_cells, "G")?;

        let cmu25 = wc.cmu.powf(0.25);
        let nl = n as Label;
        let f = self.k.omega_cell.clone();

        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(omega)
                .arg(g)
                .arg(&mut self.wall_cell_value)
                .arg(k)
                .arg(&u.f)
                .arg(&u.bf)
                .arg(nut_bf)
                .arg(&m.b_sf)
                .arg(&m.b_mag_sf)
                .arg(&m.b_y)
                .arg(&self.wall_cells)
                .arg(&self.wf_offset)
                .arg(&self.wf_face)
                .arg(&nu)
                .arg(&wc.kappa)
                .arg(&cmu25)
                .arg(&wc.beta1)
                .arg(&k_min)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    fn check(&self, n_bf: usize, n_cells: usize, m: &GpuMesh) -> Result<()> {
        expect_count(n_bf, m.n_boundary_faces, "nut boundary values")?;
        expect_count(n_cells, m.n_cells, "k")
    }
}

fn expect_count(got: usize, want: usize, what: &str) -> Result<()> {
    if got == want {
        Ok(())
    } else {
        Err(Error::Config(format!(
            "wallfunctions: `{what}` has {got} elements, expected {want}"
        )))
    }
}

// ==========================================================================
//  WernerWengleData - the LES wall model, SPEC-LIT §30.1
// ==========================================================================

struct WernerWengleKernels {
    update: CudaFunction,
}

impl WernerWengleKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::WALLFUNCTIONS)?;
        Ok(Self {
            update: k.func("wfWernerWengle")?,
        })
    }
}

/// Which boundary faces the Werner-Wengle LES wall model owns, from `nut`'s
/// own patch type (`wernerWengleWallFunction` - SPEC-LIT §15.5's rule,
/// extended to the LES wall model exactly as [`ThermalWallData`] extends it
/// to temperature).
///
/// A flat per-face list, not a CSR: like [`WallData::update_nut`], this
/// writes one face's `nu_t,w` at a time and never averages over a cell -
/// there is no wall-adjacent-CELL constraint here, because an LES has no
/// `epsilon`/`omega` cell for one to pin.
pub struct WernerWengleData {
    pub n_faces: usize,
    /// `[n_faces]` boundary-face indices, ascending.
    pub face: DevBuf<Label>,
    /// `[n_boundary_faces]` `tau_w` from the last [`Self::update_nut`] call -
    /// indexed by BOUNDARY FACE, not by [`Self::face`], so it chains
    /// straight into [`ThermalWallData::update_from_tau_w`] with no
    /// re-gather in between (the thermal wall function's `u_tau =
    /// sqrt(tau_w)` substitution, [`u_tau_werner_wengle`], taken on the
    /// device inside that call). Zero on every face this model does not own
    /// - the callee's own `u_tau > 0` guard is what keeps that from being
    /// read as a real (and wrong) friction velocity there.
    pub tau_w: DevBuf<Scalar>,
    k: WernerWengleKernels,
}

impl WernerWengleData {
    /// `faces[bf]` is `nut`'s own patch type test
    /// ([`crate::field_setup::les_nut_wall_faces`]) - one entry per boundary
    /// face, in the same flattened order [`HostMesh::b_face_cells`] uses;
    /// `faces.len()` is therefore `n_boundary_faces`, which is what
    /// [`Self::tau_w`] is sized from.
    pub fn build(gpu: &Gpu, faces: &[bool]) -> Result<Self> {
        let list: Vec<Label> = faces
            .iter()
            .enumerate()
            .filter(|(_, on)| **on)
            .map(|(bf, _)| bf as Label)
            .collect();
        let n = list.len();
        // Same convention as `WallData`/`ThermalWallData`: a zero-length
        // device buffer is an error, so a case with no LES wall faces still
        // gets one element, which `update_nut` never reads because it
        // returns early on `n_faces == 0`.
        let padded = if list.is_empty() { vec![0 as Label] } else { list };

        Ok(Self {
            n_faces: n,
            face: gpu.upload(&padded)?,
            tau_w: gpu.zeros(faces.len().max(1))?,
            k: WernerWengleKernels::new(gpu)?,
        })
    }

    /// `nu_t,w` on every face this owns, from the wall-parallel CELL-AVERAGE
    /// speed - SPEC-LIT §30.1. Writes into `nut_bf` at those faces only,
    /// exactly as [`WallData::update_nut`] does for the RAS families, and
    /// records `tau_w` for [`Self::tau_w`]/the thermal wall substitution.
    pub fn update_nut(
        &mut self,
        gpu: &Gpu,
        nut_bf: &mut DevBuf<Scalar>,
        u: &GpuVectorField,
        m: &GpuMesh,
        nu: Scalar,
    ) -> Result<()> {
        let n = self.n_faces;
        if n == 0 {
            return Ok(());
        }
        expect_count(nut_bf.len(), m.n_boundary_faces, "nut boundary values")?;
        expect_count(self.tau_w.len(), m.n_boundary_faces, "tau_w")?;
        expect_count(u.f.len(), m.n_cells, "U")?;
        expect_count(u.bf.len(), m.n_boundary_faces, "U boundary values")?;

        let nl = n as Label;
        let f = self.k.update.clone();

        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(nut_bf)
                .arg(&mut self.tau_w)
                .arg(&u.f)
                .arg(&u.bf)
                .arg(&m.b_face_cells)
                .arg(&m.b_y)
                .arg(&m.b_sf)
                .arg(&m.b_mag_sf)
                .arg(&self.face)
                .arg(&nu)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }
}

// ==========================================================================
//  The matrix constraint
// ==========================================================================

/// Pin `epsilon` (or `omega`) in every wall-adjacent cell to the value the
/// wall function computed, and decouple those rows.
///
/// *DESIGN* (SPEC-LIT §6.4): the near-wall relations give the value at the
/// first cell, not a flux at the face, so the transport equation must not be
/// allowed to have an opinion there. The row becomes `diag·psi = diag·value`
/// and the corresponding column is eliminated into the neighbours' sources by
/// [`crate::ldu_ops::set_values`], which keeps a symmetric matrix symmetric.
///
/// Call it **after** [`crate::ldu_ops::relax`] and **before**
/// [`crate::ldu_ops::add_boundary_contributions`]: relaxation would otherwise
/// re-open the row it just closed, and the boundary fold would add
/// coefficients back into a row `set_values` had already cleared.
///
/// No-op when the case has no wall-function faces, which is the common case
/// for a free-shear flow and costs one branch on the host.
pub fn constrain_wall_cells(
    gpu: &Gpu,
    k: &LduKernels,
    a: &mut GpuLduMatrix,
    m: &GpuMesh,
    wd: &WallData,
) -> Result<()> {
    let n = wd.n_wall_cells;
    if n == 0 {
        return Ok(());
    }
    if a.n_cells != m.n_cells {
        return Err(Error::Config(format!(
            "constrain_wall_cells: the matrix has {} rows, the mesh {} cells",
            a.n_cells, m.n_cells
        )));
    }

    let nl = n as Label;
    let f = wd.k.mark_fixed.clone();

    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(&mut a.is_fixed)
            .arg(&mut a.fixed_value)
            .arg(&wd.wall_cells)
            .arg(&wd.wall_cell_value)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }

    set_values(gpu, k, a, m)
}

// ==========================================================================
//  The Jayatilleke thermal wall function - SPEC-LIT §29.3
// ==========================================================================
//
// Written from:
//   Jayatilleke, Prog. Heat Mass Transfer 1 (1969) 193-330 - the sublayer
//     resistance correction `P(Pr/Pr_t)` to the thermal log law
//   ofgpu `SPEC-LIT.md` §29.3, which states the thermal law and names the
//     blending as "the same §6.4 *DESIGN* blending as every other wall
//     quantity" - the choice of WHICH of §6.4's two blends (Kader's
//     exponential weight, or epsilon/omega's root-sum-square) is ours: the
//     exponential one, because it is the one [`u_plus`] itself already uses,
//     and §29.3 asks for `T+ = Pr_t(u+ + P)` in the log branch - a relation
//     stated in terms of `u+`, not of `eps_log`/`eps_vis`. Reusing `u_plus`'s
//     OWN blend makes the `Pr = Pr_t` identity below exact rather than
//     approximate.
// No GPL-licensed source was consulted.
//
// # The law, and where it comes from
//
// ```text
// P(r)  = 9.24 [ r^{3/4} - 1 ] [ 1 + 0.28 exp(-0.007 r) ],   r = Pr/Pr_t
// T+    = Pr_t (u_log + P)                      log branch
// T+    = Pr y+                                 viscous branch
// T+    = t_vis·e^Gamma + t_log·e^{1/Gamma}      the blend, Gamma = blend_gamma(y+)
// ```
//
// exactly [`u_plus`]'s own blend with `y+` standing in for itself and
// `u_log` replaced by `t_log = Pr_t(u_log + P)`. At `Pr = Pr_t`, `P(1) = 0`
// identically (the first bracket vanishes whatever the second one is), so
// `t_vis = Pr_t·y+ `, `t_log = Pr_t·u_log`, and the blend factors:
//
// ```text
// T+ = Pr_t·y+·e^Gamma + Pr_t·u_log·e^{1/Gamma} = Pr_t·(y+ e^Gamma + u_log e^{1/Gamma}) = Pr_t·u+
// ```
//
// which is `tests::t_plus_reduces_to_prt_times_u_plus_when_pr_equals_prt`
// below, and SPEC-LIT §29.3's own consistency check.
//
// # The Robin triple this rewrites, and which of the two SPEC-LIT forms
//
// SPEC-LIT §29.3 gives both a fixed-T and a fixed-q form. **This module
// ships fixed-T** ([`crate::field::BcKind::ThermalWallFunction`]; `T_w` is
// the field file's `value` entry, exactly like every other wall-function kind
// in `src/field.rs` reads theirs). The triple it writes is the
// `fr = 0` (fixedGradient) degenerate case with
//
// ```text
// ref_grad = q_w / k_eff_wall,     q_w = rho·cp·u_tau·(T_w - T_P)/T+
// ```
//
// which is [`thermal_wall_ref_grad`] below - literally
// `crate::energy::flux_to_grad(q_w, k_eff_wall)` with `q_w` now DERIVED by
// the wall function instead of a case constant. `k_eff_wall` is whatever
// `src/energy.rs`'s own `update_k_eff` already computed at that face
// (molecular `k` plus the momentum wall function's `nu_t,w`/Pr_t) - choosing
// `ref_grad` this way makes the total flux come out to `q_w` WHATEVER
// `k_eff_wall` is, so the two models' resistances are not double-counted.
//
// **The fixed-q form falls out of the same function.** Given a case-supplied
// `q_w` instead of deriving one from `T_w`, `ref_grad = q_w/k_eff_wall` is
// the identical expression - only the source of `q_w` differs - and the wall
// temperature SPEC-LIT §29.3 says to diagnose,
// `T_w = T_P + q_w·T+/(rho·cp·u_tau)`, is `t_p + q_w*t_plus/(rho*cp*u_tau)`,
// a rearrangement of exactly the formula [`thermal_wall_ref_grad`] inverts.
//
// SPEC-LIT §32.2 is what wires the fixed-q form into a case: NOT a second
// wall-function `BcKind` here, but `crate::field::BcKind::FixedFluxTemperature`
// in `src/field.rs`/`src/energy.rs` - `ref_grad = q/k_eff_wall` directly
// (`crate::energy::flux_to_grad`, refreshed every outer iteration against the
// CURRENT `k_eff_wall`), because a `fr = 0` Robin condition delivers exactly
// the flux it is given whatever `k_eff_wall` is - the ratio cancels exactly
// against the same `k_eff_wall` the matrix assembly multiplies by, so no
// Jayatilleke machinery is needed to get the FLUX right, on a wall-function
// mesh or a resolved one. Diagnosing the wall TEMPERATURE that flux produces
// still needs the formula above - that is a postprocessing read of
// [`jayatilleke_p`]/[`t_plus`]/[`u_tau_of`], done in `src/bin/fire.rs`'s own
// report, not a second device kernel.
//
// # Why `fr = 0` rather than a genuine Robin fraction
//
// A `fr` between 0 and 1 that made the row's IMPLICIT slope
// (`-fr·k_eff_wall·delta`, what `fvLapBoundary` puts in the matrix) match the
// physical conductance `h = rho cp u_tau/T+` would need `k_eff_wall` divided
// back out of `fr` itself, recomputed every outer iteration alongside `y+`
// and `T_P` - exactly the same lag every OTHER coefficient in this crate
// already runs at (`nu_t,w`, `G`, `epsilon_wall`...). `fr = 0` is the
// simplest member of that family: the flux is exact at whatever `T_P` the
// matrix was assembled with, and outer iteration converges it exactly as it
// converges every other lagged source term here - `Su`/`Sp` included.
//
// # Parallelism
//
// One thread per wall face, exactly [`WallData::update_nut`]'s shape: no
// wall-adjacent-CELL constraint here (§15.4's `k`-at-the-wall distinction has
// no thermal analogue - `T` is not pinned, only its Robin triple is
// rewritten), so there is no CSR and no scatter.

/// Jayatilleke's sublayer-resistance correction, `P(Pr/Pr_t)` - SPEC-LIT
/// §29.3. `P(1) = 0` exactly: the first bracket is `1^{0.75} - 1 = 0`
/// regardless of the second.
#[inline]
pub fn jayatilleke_p(pr: Scalar, prt: Scalar) -> Scalar {
    let r = pr / prt;
    9.24 * (r.powf(0.75) - 1.0) * (1.0 + 0.28 * (-0.007 * r).exp())
}

/// `T+` from `y+`, continuous from the wall to the log layer - SPEC-LIT
/// §29.3, blended by [`u_plus`]'s own `blend_gamma`/`log_weight`. `p` is
/// [`jayatilleke_p`], precomputed by the caller once per case rather than
/// once per face - the same convention `nut_wall`'s callers already follow
/// for `cmu.powf(0.25)`, and the reason the device twin in
/// `cuda/wallfunctions.cu` never calls `pow`.
#[inline]
pub fn t_plus(y_plus: Scalar, pr: Scalar, prt: Scalar, kappa: Scalar, e: Scalar, p: Scalar) -> Scalar {
    let gamma = blend_gamma(y_plus);
    let u_log = (e * y_plus).max(1.0).ln() / kappa;
    let t_vis = pr * y_plus;
    let t_log = prt * (u_log + p);
    t_vis * gamma.exp() + t_log * log_weight(gamma)
}

/// `u_tau = C_mu^{1/4} sqrt(k_P)` - SPEC-LIT §29.3.
#[inline]
pub fn u_tau_of(k: Scalar, cmu: Scalar) -> Scalar {
    cmu.powf(0.25) * k.max(0.0).sqrt()
}

// ==========================================================================
//  §32.3  The independent Nusselt-number references
// ==========================================================================
//
// Two published turbulent-pipe-flow correlations, applied to a parallel-plate
// channel through the hydraulic diameter (standard practice, carrying its own
// error - see each function's own doc) - SPEC-LIT §32.3's gate for the
// redesigned thermal-wall comparison of §32.2: `Nu = q_w D_h/(k(T_w - T_b))`,
// measured from a run, is compared against these, NOT against another run of
// this solver (§32.3's own point, and the same shape §10/§22 already use).

/// Dittus & Boelter, *Univ. Calif. Publ. Eng.* 2 (1930) 443 (reprinted in
/// *Int. Commun. Heat Mass Transfer* 12 (1985) 3) - `Nu = 0.023 Re^0.8 Pr^n`,
/// `n = 0.4` for HEATING (wall hotter than the bulk, SPEC-LIT §32.2's own
/// case), valid for `0.6 < Pr < 160`, `Re > 1e4`. Conventionally quoted at
/// ±20-25% - that uncertainty is part of a verdict against this function's
/// output, not a detail to round away.
#[inline]
pub fn dittus_boelter_nu(re: Scalar, pr: Scalar) -> Scalar {
    0.023 * re.powf(0.8) * pr.powf(0.4)
}

/// Gnielinski, *Int. Chem. Eng.* 16 (1976) 359 - the more accurate modern
/// form, covering the transitional range `2300 < Re < 5e6`,
/// `0.5 < Pr < 2000`. Quoted at ±10%.
///
/// ```text
/// f  = (0.79 ln Re - 1.64)^-2                  Petukhov friction factor
/// Nu = (f/8)(Re - 1000) Pr / (1 + 12.7 sqrt(f/8) (Pr^(2/3) - 1))
/// ```
#[inline]
pub fn gnielinski_f(re: Scalar) -> Scalar {
    let d = 0.79 * re.ln() - 1.64;
    1.0 / (d * d)
}

#[inline]
pub fn gnielinski_nu(re: Scalar, pr: Scalar) -> Scalar {
    let f = gnielinski_f(re);
    let f8 = f / 8.0;
    (f8 * (re - 1000.0) * pr) / (1.0 + 12.7 * f8.sqrt() * (pr.powf(2.0 / 3.0) - 1.0))
}

/// The `ref_grad` that rewrites a fixed-T wall's Robin triple to encode the
/// Jayatilleke-corrected conductance - see the module section above.
/// `k_p`/`y`/`nu`/`cmu` feed [`y_plus_of`] exactly as every other wall
/// quantity's does; `k_min` floors `k_p` the same way [`nut_wall`]'s callers
/// floor it before the square root.
///
/// `None` where the correction has nothing to divide by: no standoff
/// (`y <= 0`), a non-positive `k_eff_wall`, or `T+ <= 0` (only at `y+ = 0`
/// itself, since `T+` is otherwise strictly positive for `Pr, Pr_t > 0`).
/// The caller leaves the face's existing triple alone in that case - the same
/// "degenerate to fixedValue until the kernel can run" convention
/// `src/field_setup.rs` already uses for every other wall-function kind.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn thermal_wall_ref_grad(
    t_w: Scalar,
    t_p: Scalar,
    k_p: Scalar,
    y: Scalar,
    nu: Scalar,
    rho: Scalar,
    cp: Scalar,
    pr: Scalar,
    prt: Scalar,
    kappa: Scalar,
    e: Scalar,
    cmu: Scalar,
    k_eff_wall: Scalar,
    k_min: Scalar,
) -> Option<Scalar> {
    if !(y > 0.0) || !(k_eff_wall > 0.0) {
        return None;
    }
    let kc = k_p.max(k_min);
    let y_plus = y_plus_of(kc, y, nu, cmu);
    let tp = t_plus(y_plus, pr, prt, kappa, e, jayatilleke_p(pr, prt));
    if !(tp > 0.0) {
        return None;
    }
    let u_tau = u_tau_of(kc, cmu);
    let q_w = rho * cp * u_tau * (t_w - t_p) / tp;
    Some(q_w / k_eff_wall)
}

/// [`thermal_wall_ref_grad`]'s twin for a wall model that computes `u_tau`
/// DIRECTLY rather than through `k` - SPEC-LIT §30.1: LES's Werner-Wengle
/// substitutes `u_tau = sqrt(tau_w)` ([`u_tau_werner_wengle`]) for the RAS
/// `Cmu^{1/4} sqrt(k)`, and an LES case has no `k` for
/// [`thermal_wall_ref_grad`] to build `y+` from in the first place.
///
/// `y+ = u_tau y/nu` here - SPEC-LIT §15.1's own definition of `y+`, rather
/// than [`y_plus_of`]'s `k`-based one. `None` under the same conditions as
/// [`thermal_wall_ref_grad`] (no standoff, no `k_eff_wall`, `T+ <= 0`), plus
/// `u_tau <= 0` - a face this wall model does not own, or one where the flow
/// has locally separated and the wall shear is momentarily zero.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn thermal_wall_ref_grad_from_u_tau(
    t_w: Scalar,
    t_p: Scalar,
    u_tau: Scalar,
    y: Scalar,
    nu: Scalar,
    rho: Scalar,
    cp: Scalar,
    pr: Scalar,
    prt: Scalar,
    kappa: Scalar,
    e: Scalar,
    k_eff_wall: Scalar,
) -> Option<Scalar> {
    if !(y > 0.0) || !(k_eff_wall > 0.0) || !(u_tau > 0.0) {
        return None;
    }
    let y_plus = u_tau * y / nu;
    let tp = t_plus(y_plus, pr, prt, kappa, e, jayatilleke_p(pr, prt));
    if !(tp > 0.0) {
        return None;
    }
    let q_w = rho * cp * u_tau * (t_w - t_p) / tp;
    Some(q_w / k_eff_wall)
}

/// Which boundary faces the Jayatilleke thermal wall function owns, from
/// `T`'s own patch types (SPEC-LIT §15.5's rule, extended to a fifth field -
/// see [`crate::field::BcKind::is_thermal_wall_function`]).
///
/// A flat per-face list, not a CSR: like [`WallData::update_nut`] this
/// rewrites one face's triple at a time and never averages over a cell.
pub struct ThermalWallData {
    pub n_faces: usize,
    /// `[n_faces]` boundary-face indices, ascending.
    pub face: DevBuf<Label>,
    k: ThermalWallKernels,
}

struct ThermalWallKernels {
    update: CudaFunction,
    update_tau_w: CudaFunction,
}

impl ThermalWallKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::WALLFUNCTIONS)?;
        Ok(Self {
            update: k.func("wfThermalWall")?,
            update_tau_w: k.func("wfThermalWallTauW")?,
        })
    }
}

impl ThermalWallData {
    /// `faces[bf]` is whatever [`crate::field_setup::faces_where`] with
    /// [`crate::field::BcKind::is_thermal_wall_function`] computed from `T`'s
    /// own field file - one entry per boundary face, in the same flattened
    /// order `HostMesh::b_face_cells` uses.
    pub fn build(gpu: &Gpu, faces: &[bool]) -> Result<Self> {
        let list: Vec<Label> = faces
            .iter()
            .enumerate()
            .filter(|(_, on)| **on)
            .map(|(bf, _)| bf as Label)
            .collect();
        let n = list.len();
        // Same convention as `WallData`: a zero-length device buffer is an
        // error, so a case with no thermal wall function still gets one
        // element, which `update` never reads because it returns early on
        // `n_faces == 0`.
        let padded = if list.is_empty() { vec![0 as Label] } else { list };

        Ok(Self {
            n_faces: n,
            face: gpu.upload(&padded)?,
            k: ThermalWallKernels::new(gpu)?,
        })
    }

    /// Rewrite `T`'s Robin triple (`fr = 0`, `ref_grad` from
    /// [`thermal_wall_ref_grad`]) on every face this owns. `T_w` is read from
    /// `ref_value` and never written - it is the field file's `value` entry,
    /// seeded once by `src/field_setup.rs` - so it survives being read again
    /// next outer iteration. `t_internal` is `T`'s cell field (`T_P`); `k` is
    /// the turbulence kinetic energy's cell field; `rho` is the cell density;
    /// `k_eff_wall` is the effective conductivity `src/energy.rs`'s
    /// `update_k_eff` already computed at every boundary face - call this
    /// AFTER that, and before the equation is assembled.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &self,
        gpu: &Gpu,
        fr: &mut DevBuf<Scalar>,
        ref_grad: &mut DevBuf<Scalar>,
        ref_value: &DevBuf<Scalar>,
        t_internal: &DevBuf<Scalar>,
        k: &DevBuf<Scalar>,
        rho: &DevBuf<Scalar>,
        k_eff_wall: &DevBuf<Scalar>,
        m: &GpuMesh,
        wc: &WallFunctionCoeffs,
        nu: Scalar,
        cp: Scalar,
        pr: Scalar,
        prt: Scalar,
        k_min: Scalar,
    ) -> Result<()> {
        let n = self.n_faces;
        if n == 0 {
            return Ok(());
        }
        expect_count(fr.len(), m.n_boundary_faces, "T fr")?;
        expect_count(ref_grad.len(), m.n_boundary_faces, "T ref_grad")?;
        expect_count(ref_value.len(), m.n_boundary_faces, "T ref_value")?;
        expect_count(t_internal.len(), m.n_cells, "T")?;
        expect_count(k.len(), m.n_cells, "k")?;
        expect_count(rho.len(), m.n_cells, "rho")?;
        expect_count(k_eff_wall.len(), m.n_boundary_faces, "k_eff wall")?;

        let cmu25 = wc.cmu.powf(0.25);
        let jay_p = jayatilleke_p(pr, prt);
        let nl = n as Label;
        let f = self.k.update.clone();

        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(fr)
                .arg(ref_grad)
                .arg(ref_value)
                .arg(t_internal)
                .arg(k)
                .arg(rho)
                .arg(k_eff_wall)
                .arg(&m.b_face_cells)
                .arg(&m.b_y)
                .arg(&self.face)
                .arg(&nu)
                .arg(&cp)
                .arg(&pr)
                .arg(&prt)
                .arg(&jay_p)
                .arg(&wc.kappa)
                .arg(&wc.e)
                .arg(&cmu25)
                .arg(&k_min)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// [`Self::update`]'s twin for a wall model that supplies `tau_w`
    /// directly rather than `k` - SPEC-LIT §30.1: LES's Werner-Wengle.
    /// `tau_w`, not `u_tau`, is what this takes: [`WernerWengleData::tau_w`]
    /// is already indexed by boundary face, so it is passed straight
    /// through with no re-gather, and the `u_tau = sqrt(tau_w)` substitution
    /// happens once, on the device, inside the kernel this launches - rather
    /// than needing a separate elementwise-sqrt pass over every boundary
    /// face first. There is no `k` argument: an LES case carries none for
    /// this to read.
    #[allow(clippy::too_many_arguments)]
    pub fn update_from_tau_w(
        &self,
        gpu: &Gpu,
        fr: &mut DevBuf<Scalar>,
        ref_grad: &mut DevBuf<Scalar>,
        ref_value: &DevBuf<Scalar>,
        t_internal: &DevBuf<Scalar>,
        tau_w: &DevBuf<Scalar>,
        rho: &DevBuf<Scalar>,
        k_eff_wall: &DevBuf<Scalar>,
        m: &GpuMesh,
        wc: &WallFunctionCoeffs,
        nu: Scalar,
        cp: Scalar,
        pr: Scalar,
        prt: Scalar,
    ) -> Result<()> {
        let n = self.n_faces;
        if n == 0 {
            return Ok(());
        }
        expect_count(fr.len(), m.n_boundary_faces, "T fr")?;
        expect_count(ref_grad.len(), m.n_boundary_faces, "T ref_grad")?;
        expect_count(ref_value.len(), m.n_boundary_faces, "T ref_value")?;
        expect_count(t_internal.len(), m.n_cells, "T")?;
        expect_count(tau_w.len(), m.n_boundary_faces, "tau_w")?;
        expect_count(rho.len(), m.n_cells, "rho")?;
        expect_count(k_eff_wall.len(), m.n_boundary_faces, "k_eff wall")?;

        let jay_p = jayatilleke_p(pr, prt);
        let nl = n as Label;
        let f = self.k.update_tau_w.clone();

        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(fr)
                .arg(ref_grad)
                .arg(ref_value)
                .arg(t_internal)
                .arg(tau_w)
                .arg(rho)
                .arg(k_eff_wall)
                .arg(&m.b_face_cells)
                .arg(&m.b_y)
                .arg(&self.face)
                .arg(&nu)
                .arg(&cp)
                .arg(&pr)
                .arg(&prt)
                .arg(&jay_p)
                .arg(&wc.kappa)
                .arg(&wc.e)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    /// A test that does not care about SPEC-LIT 15.5's distinction puts the
    /// same faces in both sets. A real case must not: `nut`'s patch types and
    /// `epsilon`'s are read separately, from their own files.
    fn same_for_both(flags: &[bool]) -> crate::field_setup::WallFaces {
        crate::field_setup::WallFaces {
            constrained_cells: flags.to_vec(),
            nut: flags.to_vec(),
        }
    }

    use super::*;

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    const KAPPA: Scalar = 0.41;
    const E: Scalar = 9.8;
    const CMU: Scalar = 0.09;

    // ----------------------------------------------------------------------
    //  y+_lam
    // ----------------------------------------------------------------------

    /// The whole point of solving for it rather than writing 11.53 down: the
    /// number that comes out has to satisfy the equation that defines it.
    #[test]
    fn y_plus_lam_satisfies_its_own_fixed_point() {
        let y = compute_y_plus_lam(KAPPA, E);
        let residual = y - (E * y).ln() / KAPPA;

        assert!(
            residual.abs() < 1e-12,
            "y+_lam = {y} leaves residual {residual}"
        );
        assert!(
            (11.0..12.0).contains(&y),
            "y+_lam = {y} is nowhere near the documented 11.53"
        );
    }

    /// A different pair of constants must give a different root, and that
    /// root must satisfy its own equation too - otherwise the iteration is
    /// converging to something structural rather than to the answer.
    #[test]
    fn y_plus_lam_tracks_kappa_and_e() {
        for (kappa, e) in [(0.41, 9.8), (0.4187, 9.0), (0.38, 12.0), (0.41, 5.5)] {
            let y = compute_y_plus_lam(kappa, e);
            let residual = y - (e * y).ln() / kappa;
            assert!(
                residual.abs() < 1e-12,
                "kappa {kappa}, E {e}: y+_lam = {y}, residual {residual}"
            );
        }

        assert!(compute_y_plus_lam(0.41, 9.8) != compute_y_plus_lam(0.41, 5.5));
    }

    // ----------------------------------------------------------------------
    //  Continuity of the blend - SPEC-LIT 6.4, *DESIGN*
    // ----------------------------------------------------------------------

    /// The switched relations of SPEC-LIT 6.4, for comparison only. This is
    /// what the specification writes down before the blending paragraph, and
    /// what this implementation deliberately does NOT do.
    fn switched_eps_log(k: Scalar, y: Scalar, kappa: Scalar, cmu: Scalar) -> Scalar {
        cmu.powf(0.75) * k * k.sqrt() / (kappa * y)
    }

    fn switched_eps_vis(k: Scalar, y: Scalar, nu: Scalar) -> Scalar {
        2.0 * k * nu / (y * y)
    }

    fn switched_omega_log(k: Scalar, y: Scalar, kappa: Scalar, cmu: Scalar) -> Scalar {
        k.sqrt() / (cmu.powf(0.25) * kappa * y)
    }

    fn switched_omega_vis(y: Scalar, nu: Scalar, beta1: Scalar) -> Scalar {
        6.0 * nu / (beta1 * y * y)
    }

    /// The wall distance at which `y+` takes a given value.
    fn y_at(y_plus: Scalar, k: Scalar, nu: Scalar, cmu: Scalar) -> Scalar {
        y_plus * nu / (cmu.powf(0.25) * k.max(0.0).sqrt())
    }

    /// **The test the whole design exists for.**
    ///
    /// At `y+_lam` the two branches of the dissipation relation disagree by a
    /// large factor - the specification's own observation, and the reason a
    /// first cell sitting there limit-cycles between them. Measure that jump,
    /// then measure what the blend does at the same place.
    ///
    /// `nu_t,w` is deliberately not the quantity this is centred on: the log
    /// branch `nu(y+ kappa/ln(E y+) - 1)` is identically zero at `y+_lam`,
    /// because that is what `y+_lam` means, so even the switched form of
    /// `nu_t` is continuous there - it merely has a kink. `epsilon`, `omega`
    /// and `G` are the ones that jump. All four are checked below.
    #[test]
    fn the_switched_dissipation_jumps_at_y_plus_lam_and_the_blend_does_not() {
        let nu: Scalar = 1e-5;
        let k: Scalar = 0.05;
        let beta1: Scalar = 0.075;
        let y_lam = compute_y_plus_lam(KAPPA, E);
        let y_star = y_at(y_lam, k, nu, CMU);

        // 1. the switched form really is discontinuous there.
        let e_log = switched_eps_log(k, y_star, KAPPA, CMU);
        let e_vis = switched_eps_vis(k, y_star, nu);
        let eps_jump = (e_log / e_vis).max(e_vis / e_log);
        assert!(
            eps_jump > 3.0,
            "the two epsilon branches differ by only a factor {eps_jump} at \
             y+_lam; either y+_lam is wrong or there is nothing to blend"
        );

        let w_log = switched_omega_log(k, y_star, KAPPA, CMU);
        let w_vis = switched_omega_vis(y_star, nu, beta1);
        assert!(
            (w_log / w_vis - 1.0).abs() > 0.1,
            "the two omega branches agree to within 10% at y+_lam"
        );

        // 2. crossing y+_lam, the blend moves by essentially nothing. Sample
        //    y+ from y+_lam - 1 to y+_lam + 1 in steps of 1e-3 and take the
        //    largest relative step between neighbours.
        let n = 2001;
        let lo = y_lam - 1.0;
        let hi = y_lam + 1.0;
        let h = (hi - lo) / (n - 1) as Scalar;

        let mut worst_e: Scalar = 0.0;
        let mut worst_w: Scalar = 0.0;
        let mut worst_nut: Scalar = 0.0;

        let mut prev_e = epsilon_wall(k, y_at(lo, k, nu, CMU), nu, KAPPA, CMU);
        let mut prev_w = omega_wall(k, y_at(lo, k, nu, CMU), nu, KAPPA, CMU, beta1);
        let mut prev_nut = nut_wall(lo, nu, KAPPA, E);
        let nut_scale = nut_wall(hi, nu, KAPPA, E);
        assert!(nut_scale > 0.0);

        for i in 1..n {
            let y_plus = lo + h * i as Scalar;
            let y = y_at(y_plus, k, nu, CMU);

            let e = epsilon_wall(k, y, nu, KAPPA, CMU);
            let w = omega_wall(k, y, nu, KAPPA, CMU, beta1);
            let nut = nut_wall(y_plus, nu, KAPPA, E);

            worst_e = worst_e.max((e - prev_e).abs() / e.max(prev_e));
            worst_w = worst_w.max((w - prev_w).abs() / w.max(prev_w));
            worst_nut = worst_nut.max((nut - prev_nut).abs() / nut_scale);

            prev_e = e;
            prev_w = w;
            prev_nut = nut;
        }

        // A step of 1e-3 in y+ moves nothing by as much as 1% of itself. What
        // this excludes is a jump; the blends are smooth, so the bound is
        // generous by orders of magnitude.
        assert!(worst_e < 0.01, "epsilon blend steps by {worst_e} across y+_lam");
        assert!(worst_w < 0.01, "omega blend steps by {worst_w} across y+_lam");
        assert!(worst_nut < 0.01, "nu_t,w steps by {worst_nut} across y+_lam");

        // And the switched epsilon, sampled the same way, does jump - by
        // roughly the branch ratio, in a single step.
        let switched = |y_plus: Scalar| -> Scalar {
            let y = y_at(y_plus, k, nu, CMU);
            if y_plus > y_lam {
                switched_eps_log(k, y, KAPPA, CMU)
            } else {
                switched_eps_vis(k, y, nu)
            }
        };
        let mut worst_switched: Scalar = 0.0;
        let mut prev = switched(lo);
        for i in 1..n {
            let v = switched(lo + h * i as Scalar);
            worst_switched = worst_switched.max((v - prev).abs() / v.max(prev));
            prev = v;
        }
        assert!(
            worst_switched > 0.5,
            "the switched epsilon stepped by only {worst_switched}; the \
             comparison is not measuring the discontinuity it claims to"
        );
    }

    /// The production relation is discontinuous under switching too, and for
    /// the same reason: above `y+_lam` it is the log-layer expression, below
    /// it there is no turbulent stress and it should be zero. The blend
    /// crosses without a step.
    #[test]
    fn the_production_blend_crosses_y_plus_lam_without_a_step() {
        let nu: Scalar = 1e-5;
        let k: Scalar = 0.05;
        let shear: Scalar = 250.0;
        let y_lam = compute_y_plus_lam(KAPPA, E);

        let n = 2001;
        let lo = y_lam - 1.0;
        let hi = y_lam + 1.0;
        let h = (hi - lo) / (n - 1) as Scalar;

        let g_of = |y_plus: Scalar| -> Scalar {
            let y = y_at(y_plus, k, nu, CMU);
            let nut = nut_wall(y_plus, nu, KAPPA, E);
            production_wall(y_plus, nut, nu, shear, k, y, KAPPA, CMU)
        };

        let scale = g_of(hi);
        assert!(scale > 0.0);

        let mut worst: Scalar = 0.0;
        let mut prev = g_of(lo);
        for i in 1..n {
            let v = g_of(lo + h * i as Scalar);
            worst = worst.max((v - prev).abs() / scale);
            prev = v;
        }
        assert!(worst < 0.01, "production steps by {worst} of its own size");
    }

    /// Continuity is worth nothing if the blend has stopped agreeing with the
    /// branches it blends. Far below `y+_lam` it must be the viscous law, far
    /// above it the log law.
    #[test]
    fn the_blend_recovers_both_branches_in_their_own_limits() {
        let nu: Scalar = 1e-5;

        // Deep sublayer: u+ -> y+, so nu_t,w -> 0. The departure is
        // Gamma(y+) ~ -0.01 (y+)^4, i.e. 1e-10 at y+ = 0.01 and 2e-4 at
        // y+ = 0.5: the blend leaves the linear law smoothly rather than at a
        // point, which is exactly the property being bought.
        for (y_plus, tol) in [
            (0.0 as Scalar, 1e-15 as Scalar),
            (0.01, 1e-9),
            (0.1, 1e-6),
            (0.5, 3e-4),
        ] {
            let up = u_plus(y_plus, KAPPA, E);
            assert!(
                (up - y_plus).abs() <= tol * y_plus.max(1e-3),
                "y+ = {y_plus}: u+ = {up}, expected the linear law to within {tol}"
            );

            let nut = nut_wall(y_plus, nu, KAPPA, E);
            assert!(
                nut <= 2.0 * tol.max(1e-15) * nu,
                "y+ = {y_plus}: nu_t,w = {nut} is not negligible against nu = {nu}"
            );
        }

        // Log layer: u+ -> ln(E y+)/kappa. The remaining departure is
        // 1 - exp(1/Gamma) ~ |1/Gamma| ~ 100/(y+)^3, so the tolerance is
        // written in terms of Gamma rather than as a constant that would only
        // hold at one y+.
        for y_plus in [300.0 as Scalar, 1000.0, 1e4] {
            let tol = 3.0 * (1.0 / blend_gamma(y_plus)).abs();

            let up = u_plus(y_plus, KAPPA, E);
            let want = (E * y_plus).ln() / KAPPA;
            assert!(
                (up - want).abs() <= tol * want,
                "y+ = {y_plus}: u+ = {up}, log law gives {want} (tolerance {tol})"
            );

            let nut = nut_wall(y_plus, nu, KAPPA, E);
            let want_nut = nu * (y_plus * KAPPA / (E * y_plus).ln() - 1.0);
            assert!(
                (nut - want_nut).abs() <= 10.0 * tol * want_nut,
                "y+ = {y_plus}: nu_t,w = {nut}, log branch gives {want_nut}"
            );
        }

        // ... and it really does converge: the departure at y+ = 1e4 is
        // smaller than at y+ = 300 by the cube of the ratio, near enough.
        let d = |y_plus: Scalar| {
            (u_plus(y_plus, KAPPA, E) - (E * y_plus).ln() / KAPPA).abs()
                / ((E * y_plus).ln() / KAPPA)
        };
        assert!(d(1e4) < 1e-4 * d(300.0));
    }

    /// `epsilon` and `omega` are blended by root-sum-square, so they are
    /// smooth everywhere and never below either branch. Both properties are
    /// load-bearing: smoothness stops the limit cycle, and the lower bound is
    /// what makes the blend the *stable* side of the two.
    #[test]
    fn epsilon_and_omega_blends_bound_their_branches_and_stay_smooth() {
        let nu: Scalar = 1e-5;
        let k: Scalar = 0.1;
        let beta1: Scalar = 0.075;

        // Geometric sweep of y over four decades, 4000 samples, so each step
        // is a factor 1.0023 in y. The viscous branch goes as 1/y^2, so the
        // largest relative step the sweep itself can produce is about
        // 2 x 0.0023 = 0.5%; anything much above that is a step in the
        // FUNCTION rather than in the sampling.
        let n = 4000;
        let y_lo: Scalar = 1e-6;
        let ratio = (1e4 as Scalar).powf(1.0 / (n - 1) as Scalar);

        let mut prev_e: Scalar = 0.0;
        let mut prev_w: Scalar = 0.0;
        let mut worst_e_rel: Scalar = 0.0;
        let mut worst_w_rel: Scalar = 0.0;

        for i in 0..n {
            let y = y_lo * ratio.powi(i as i32);

            let e = epsilon_wall(k, y, nu, KAPPA, CMU);
            let w = omega_wall(k, y, nu, KAPPA, CMU, beta1);

            // Never below either branch: the blend errs towards MORE
            // dissipation, which is the stable direction for a sink.
            assert!(e >= switched_eps_log(k, y, KAPPA, CMU) * (1.0 - 1e-12));
            assert!(e >= switched_eps_vis(k, y, nu) * (1.0 - 1e-12));
            assert!(w >= switched_omega_log(k, y, KAPPA, CMU) * (1.0 - 1e-12));
            assert!(w >= switched_omega_vis(y, nu, beta1) * (1.0 - 1e-12));

            if i > 0 {
                worst_e_rel = worst_e_rel.max((e - prev_e).abs() / e.max(prev_e));
                worst_w_rel = worst_w_rel.max((w - prev_w).abs() / w.max(prev_w));
            }
            prev_e = e;
            prev_w = w;
        }

        assert!(worst_e_rel < 0.01, "epsilon blend step {worst_e_rel}");
        assert!(worst_w_rel < 0.01, "omega blend step {worst_w_rel}");
    }

    /// Production must vanish in the sublayer and reproduce SPEC-LIT §6.4's
    /// log-layer relation above it. Without the weight it would tend to
    /// `nu·|du/dy|_w·C_mu^{1/4}sqrt(k)/(kappa y)`, which is not zero.
    #[test]
    fn production_vanishes_in_the_sublayer_and_recovers_the_log_relation() {
        let nu: Scalar = 1e-5;
        let k: Scalar = 0.01;
        let y: Scalar = 1e-4;
        let shear: Scalar = 100.0;

        let unweighted = |y_plus: Scalar| {
            let nut = nut_wall(y_plus, nu, KAPPA, E);
            (nut + nu) * shear * CMU.powf(0.25) * k.sqrt() / (KAPPA * y)
        };

        let deep = production_wall(0.2, nut_wall(0.2, nu, KAPPA, E), nu, shear, k, y, KAPPA, CMU);
        assert!(
            deep < 1e-8 * unweighted(0.2),
            "production {deep} in the sublayer is not negligible against \
             {} ",
            unweighted(0.2)
        );

        let far = production_wall(
            2000.0,
            nut_wall(2000.0, nu, KAPPA, E),
            nu,
            shear,
            k,
            y,
            KAPPA,
            CMU,
        );
        let want = unweighted(2000.0);
        assert!(
            (far - want).abs() < 1e-6 * want,
            "production {far} does not recover the log relation {want}"
        );
    }

    // ----------------------------------------------------------------------
    //  Device
    // ----------------------------------------------------------------------

    /// The host mirrors above are the specification the kernels are tested
    /// against; if they drift apart, every continuity guarantee this module
    /// makes is about code that does not run.
    #[test]
    fn device_agrees_with_the_host_law() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        // A one-cell-thick slab: 4 cells in a row, the xmin patch a wall.
        let (mut m, points, faces) =
            crate::mesh::topology::tests::box_mesh([4, 1, 1], crate::Vec3::new(0.25, 1.0, 1.0));
        m.compute_geometry(&points, &faces)?;
        m.build_cell_face_maps();

        let gm = GpuMesh::upload(&gpu, &m)?;

        // Only the xmin patch (the first patch) carries a wall function.
        let mut flags = vec![false; m.n_boundary_faces];
        let p = &m.patches[0];
        for i in 0..p.size {
            flags[p.start + i] = true;
        }

        let wd = WallData::build(
            &gpu,
            &m,
            &same_for_both(&flags),
            &crate::field_setup::NutRoughness::none(m.n_boundary_faces),
        )?;
        assert_eq!(wd.n_wall_faces, p.size);
        assert_eq!(wd.n_wall_cells, p.size);

        let nu: Scalar = 1e-5;
        let k_min: Scalar = 1e-15;
        let wc = WallFunctionCoeffs::default();

        // k chosen so y+ straddles y+_lam: y = 0.125, C_mu^0.25 = 0.5477.
        let k_host = vec![2.0e-6 as Scalar; m.n_cells];
        let k_dev = gpu.upload(&k_host)?;
        let mut nut_bf = gpu.zeros::<Scalar>(m.n_boundary_faces)?;

        // Every face here is the k-based (`nutk`) family - see
        // `same_for_both` above - so `U` is never read; it still has to be
        // the right shape.
        let u = GpuVectorField::zeros(&gpu, &gm, "U")?;

        wd.update_nut(&gpu, &mut nut_bf, &k_dev, &u, &gm, &wc, nu, k_min)?;
        gpu.sync()?;

        let got = gpu.download(&nut_bf)?;
        let face_ids = gpu.download(&wd.wf_face)?;

        for &bf in face_ids.iter().take(wd.n_wall_faces) {
            let bf = bf as usize;
            let y = m.b_y[bf];
            let y_plus = y_plus_of(k_host[0], y, nu, wc.cmu);
            let want = nut_wall(y_plus, nu, wc.kappa, wc.e);

            assert!(
                (got[bf] - want).abs() <= 1e-12 * want.abs().max(nu),
                "face {bf}: y+ {y_plus}, device {} host {want}",
                got[bf]
            );
        }

        Ok(())
    }

    /// *DESIGN 2*: a cell with more than one wall face averages them weighted
    /// by face area.
    ///
    /// A 2x2x1 block with the `xmin` and `ymin` patches both carrying wall
    /// functions gives one corner cell with two wall faces of DIFFERENT area
    /// and different standoff, so a plain mean and an area-weighted mean give
    /// different numbers and the test can tell them apart. The expectation is
    /// computed from the host mirrors, face by face.
    #[test]
    fn a_cell_with_two_wall_faces_takes_the_area_weighted_average() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        // Deliberately anisotropic: dx != dy, so area weighting matters.
        let d = crate::Vec3::new(0.20, 0.05, 0.30);
        let (mut m, points, faces) = crate::mesh::topology::tests::box_mesh([2, 2, 1], d);
        m.compute_geometry(&points, &faces)?;
        m.build_cell_face_maps();

        let gm = GpuMesh::upload(&gpu, &m)?;

        // xmin (patch 0) and ymin (patch 2) are the walls.
        let mut flags = vec![false; m.n_boundary_faces];
        for p in [0usize, 2] {
            let p = &m.patches[p];
            for i in 0..p.size {
                flags[p.start + i] = true;
            }
        }

        let mut wd = WallData::build(
            &gpu,
            &m,
            &same_for_both(&flags),
            &crate::field_setup::NutRoughness::none(m.n_boundary_faces),
        )?;
        assert_eq!(wd.n_wall_faces, 4);
        assert_eq!(wd.n_wall_cells, 3, "the corner cell must not be counted twice");

        let nu: Scalar = 2e-5;
        let k_min: Scalar = 1e-15;
        let wc = WallFunctionCoeffs::default();

        let kc: Scalar = 3.0e-3;
        let k_dev = gpu.upload(&vec![kc; m.n_cells])?;

        // A velocity with a component along each wall, so the two faces see
        // different tangential shear rates as well as different areas.
        let uc = crate::Vec3::new(1.0, 2.0, 3.0);
        let mut u = GpuVectorField::zeros(&gpu, &gm, "U")?;
        gpu.write(&mut u.f, &vec![uc; m.n_cells])?;
        // u.bf stays zero: a no-slip wall.

        let mut nut_bf = gpu.zeros::<Scalar>(m.n_boundary_faces)?;
        let mut eps = gpu.zeros::<Scalar>(m.n_cells)?;
        let mut g = gpu.zeros::<Scalar>(m.n_cells)?;

        wd.update_nut(&gpu, &mut nut_bf, &k_dev, &u, &gm, &wc, nu, k_min)?;
        wd.update_epsilon(
            &gpu, &mut eps, &mut g, &k_dev, &u, &nut_bf, &gm, &wc, nu, k_min,
        )?;
        gpu.sync()?;

        let cells = gpu.download(&wd.wall_cells)?;
        let offset = gpu.download(&wd.wf_offset)?;
        let face = gpu.download(&wd.wf_face)?;
        let eps_dev = gpu.download(&eps)?;
        let g_dev = gpu.download(&g)?;
        let pinned = gpu.download(&wd.wall_cell_value)?;

        let mut saw_a_corner = false;

        for i in 0..wd.n_wall_cells {
            let c = cells[i] as usize;
            let lo = offset[i] as usize;
            let hi = offset[i + 1] as usize;
            if hi - lo == 2 {
                saw_a_corner = true;
            }

            let mut sum_a: Scalar = 0.0;
            let mut sum_e: Scalar = 0.0;
            let mut sum_g: Scalar = 0.0;

            for j in lo..hi {
                let bf = face[j] as usize;
                let y = m.b_y[bf];
                let a = m.b_mag_sf[bf];

                // The tangential wall shear rate, spelled out on the host.
                let n = m.b_sf[bf] / m.b_mag_sf[bf];
                let t = uc - n * uc.dot(n);
                let shear = t.mag() / y;

                let y_plus = y_plus_of(kc, y, nu, wc.cmu);
                let nutw = nut_wall(y_plus, nu, wc.kappa, wc.e);

                sum_a += a;
                sum_e += a * epsilon_wall(kc, y, nu, wc.kappa, wc.cmu);
                sum_g += a * production_wall(y_plus, nutw, nu, shear, kc, y, wc.kappa, wc.cmu);
            }

            let want_e = sum_e / sum_a;
            let want_g = sum_g / sum_a;

            assert!(
                (eps_dev[c] - want_e).abs() <= 1e-11 * want_e,
                "cell {c} ({} wall faces): epsilon {} , area-weighted {want_e}",
                hi - lo,
                eps_dev[c]
            );
            assert!(
                (g_dev[c] - want_g).abs() <= 1e-11 * want_g.abs().max(1e-30),
                "cell {c}: G {} , area-weighted {want_g}",
                g_dev[c]
            );
            assert!(
                (pinned[i] - want_e).abs() <= 1e-11 * want_e,
                "cell {c}: the constraint value {} is not the field value",
                pinned[i]
            );

            // And an area-weighted mean is not a plain mean, on this mesh.
            if hi - lo == 2 {
                let plain = (0.5 as Scalar)
                    * ((0..2)
                        .map(|q| {
                            let bf = face[lo + q] as usize;
                            epsilon_wall(kc, m.b_y[bf], nu, wc.kappa, wc.cmu)
                        })
                        .sum::<Scalar>());
                assert!(
                    (plain - want_e).abs() > 0.05 * want_e,
                    "the two weightings agree to within 5%, so this mesh does \
                     not distinguish them"
                );
            }
        }

        assert!(saw_a_corner, "no cell in this mesh had two wall faces");

        // Cells with no wall face are untouched.
        let corner_free: Vec<usize> = (0..m.n_cells)
            .filter(|c| !cells[..wd.n_wall_cells].contains(&(*c as Label)))
            .collect();
        for c in corner_free {
            assert_eq!(eps_dev[c], 0.0, "cell {c} has no wall face but was written");
            assert_eq!(g_dev[c], 0.0);
        }

        Ok(())
    }

    /// The matrix constraint: after [`constrain_wall_cells`], every wall
    /// row reads `diag*psi = diag*value` and is completely decoupled from its
    /// neighbours, whichever side of a face the constrained cell is on.
    #[test]
    fn constrain_wall_cells_pins_and_decouples_every_wall_row() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let d = crate::Vec3::new(0.25, 0.25, 0.25);
        let (mut m, points, faces) = crate::mesh::topology::tests::box_mesh([4, 3, 2], d);
        m.compute_geometry(&points, &faces)?;
        m.build_cell_face_maps();

        let gm = GpuMesh::upload(&gpu, &m)?;
        let ldu = LduKernels::new(&gpu)?;

        // xmin and xmax, so constrained cells appear as both owner and
        // neighbour of an internal face.
        let mut flags = vec![false; m.n_boundary_faces];
        for p in [0usize, 1] {
            let p = &m.patches[p];
            for i in 0..p.size {
                flags[p.start + i] = true;
            }
        }

        let mut wd = WallData::build(
            &gpu,
            &m,
            &same_for_both(&flags),
            &crate::field_setup::NutRoughness::none(m.n_boundary_faces),
        )?;
        assert!(wd.n_wall_cells > 0);

        let value: Scalar = 4.25;
        gpu.write(&mut wd.wall_cell_value, &vec![value; wd.n_wall_cells])?;

        let mut a = GpuLduMatrix::new(&gpu, &gm)?;
        a.zero(&gpu)?;
        gpu.write(&mut a.diag, &vec![3.0 as Scalar; m.n_cells])?;
        gpu.write(&mut a.upper, &vec![-0.5 as Scalar; m.n_internal_faces])?;
        gpu.write(&mut a.lower, &vec![-0.25 as Scalar; m.n_internal_faces])?;

        constrain_wall_cells(&gpu, &ldu, &mut a, &gm, &wd)?;
        gpu.sync()?;

        let upper = gpu.download(&a.upper)?;
        let lower = gpu.download(&a.lower)?;
        let diag = gpu.download(&a.diag)?;
        let src = gpu.download(&a.source)?;
        let cells = gpu.download(&wd.wall_cells)?;

        let mut fixed = vec![false; m.n_cells];
        for &c in cells.iter().take(wd.n_wall_cells) {
            fixed[c as usize] = true;
        }

        for f in 0..m.n_internal_faces {
            let o = m.owner[f] as usize;
            let nb = m.neighbour[f] as usize;
            if fixed[o] || fixed[nb] {
                assert_eq!(upper[f], 0.0, "face {f} still couples a pinned row");
                assert_eq!(lower[f], 0.0, "face {f} still couples a pinned row");
            }
        }

        for &c in cells.iter().take(wd.n_wall_cells) {
            let c = c as usize;
            assert!(
                (src[c] - value * diag[c]).abs() <= 1e-13 * (value * diag[c]).abs(),
                "cell {c}: source {} , diag*value {}",
                src[c],
                value * diag[c]
            );
        }

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  Rough walls - SPEC-LIT §15.3, completed by §29.2
    // ----------------------------------------------------------------------

    /// `roughness_db`'s own doc comment explains why: at `Ks+ = 2.25` the
    /// sine factor is of order `1e-4` rather than exactly zero, and at
    /// `Ks+ = 90` the two branches' log arguments and sine factors coincide
    /// exactly, in the formula's own terms. Both are properties of the
    /// published constants, so this measures the actual size of each step
    /// rather than asserting there is none.
    #[test]
    fn roughness_db_is_continuous_at_both_seams() {
        let cs: Scalar = 0.5;
        let eps: Scalar = 1e-6;

        let smooth_side = roughness_db(2.25 - eps, cs, KAPPA);
        let rough_side_at_225 = roughness_db(2.25 + eps, cs, KAPPA);
        assert_eq!(smooth_side, 0.0, "Ks+ <= 2.25 must be exactly hydraulically smooth");
        assert!(
            rough_side_at_225.abs() < 1e-3,
            "dB jumps to {rough_side_at_225} immediately above Ks+ = 2.25"
        );

        let trans_side = roughness_db(90.0 - eps, cs, KAPPA);
        let fully_rough_side = roughness_db(90.0 + eps, cs, KAPPA);
        let jump = (trans_side - fully_rough_side).abs();
        assert!(
            jump < 1e-4,
            "dB jumps by {jump} across Ks+ = 90 (transitional side {trans_side}, \
             fully-rough side {fully_rough_side})"
        );
    }

    /// SPEC-LIT §15.3's fully-rough branch, `dB = ln(1 + Cs Ks+)/kappa`, is
    /// exactly what [`roughness_db`] returns for `Ks+ >= 90` - not an
    /// asymptote it merely approaches.
    #[test]
    fn roughness_db_matches_the_fully_rough_analytic_limit() {
        for (ks_plus, cs) in [
            (90.0 as Scalar, 0.5 as Scalar),
            (150.0, 0.6),
            (1000.0, 1.0),
            (5000.0, 0.3),
        ] {
            let got = roughness_db(ks_plus, cs, KAPPA);
            let want = (1.0 + cs * ks_plus).ln() / KAPPA;
            assert!(
                (got - want).abs() <= 1e-13 * want.abs().max(1e-30),
                "Ks+={ks_plus} Cs={cs}: dB {got}, analytic {want}"
            );
        }
    }

    /// `E_eff` composes with [`u_plus`] to give exactly SPEC-LIT §29.2's
    /// `u+ = ln(E y+)/kappa - dB` far above `y+_lam`, where the log branch
    /// carries essentially all the weight.
    #[test]
    fn the_rough_log_law_matches_ln_e_yplus_over_kappa_minus_db() {
        let ks: Scalar = 8e-3;
        let cs: Scalar = 0.5;
        let u_tau: Scalar = 0.4;
        let nu: Scalar = 1.5e-5;

        let ks_plus = ks_plus_of(ks, cs, u_tau, nu);
        assert!(ks_plus > 90.0, "test wants the fully-rough regime, got Ks+ = {ks_plus}");

        let db = roughness_db(ks_plus, cs, KAPPA);
        let eeff = e_eff(E, KAPPA, db);

        let y_plus: Scalar = 5000.0;
        let up = u_plus(y_plus, KAPPA, eeff);
        let want = (E * y_plus).ln() / KAPPA - db;
        assert!(
            (up - want).abs() < 1e-6 * want.abs(),
            "u+ = {up}, ln(E y+)/kappa - dB = {want}"
        );
    }

    /// `Ks -> 0` must reproduce the smooth `nutk` wall to round-off, on every
    /// `k`/`y`/`Cs` this sweeps - the §22 gate, at the host-mirror level.
    #[test]
    fn ks_zero_reproduces_the_smooth_nutk_wall_everywhere() {
        let nu: Scalar = 1.2e-5;
        let cmu25 = CMU.powf(0.25);

        for k in [1e-6 as Scalar, 1e-4, 1e-2, 1.0] {
            for y in [1e-4 as Scalar, 1e-3, 1e-2] {
                let y_plus = y_plus_of(k, y, nu, CMU);
                let smooth = nut_wall(y_plus, nu, KAPPA, E);

                for cs in [0.3 as Scalar, 0.5, 1.0] {
                    let rough = nut_wall_rough_k(y_plus, k, nu, KAPPA, E, cmu25, 0.0, cs);
                    assert!(
                        (rough - smooth).abs() <= 1e-13 * smooth.abs().max(1e-30),
                        "k={k} y={y} cs={cs}: rough(Ks=0) {rough}, smooth {smooth}"
                    );
                }
            }
        }
    }

    /// An independent, hand-written smooth Newton (SPEC-LIT §15.1, no
    /// roughness term anywhere in it), so the `Ks -> 0` gate below checks
    /// [`u_tau_newton`] against code that never mentions roughness, not
    /// against itself.
    fn smooth_u_tau_newton_reference(
        u_mag: Scalar,
        y: Scalar,
        nu: Scalar,
        kappa: Scalar,
        e: Scalar,
    ) -> Scalar {
        if !(u_mag > 0.0) {
            return 0.0;
        }
        let mut u_tau: Scalar = (nu * u_mag / y).max(1e-300).sqrt();
        for _ in 0..10 {
            let u_plus = u_mag / u_tau;
            let ku = kappa * u_plus;
            let euk = ku.exp();
            let poly = euk - 1.0 - ku - ku * ku * 0.5 - ku * ku * ku / 6.0;
            let f = y * u_tau / nu - u_plus - poly / e;
            let dpoly = kappa * (euk - 1.0 - ku - ku * ku * 0.5);
            let df = y / nu + (u_plus / u_tau) * (1.0 + dpoly / e);
            if !(df.abs() > 0.0) {
                break;
            }
            let next = (u_tau - f / df).max(1e-300);
            let done = (next - u_tau).abs() <= 1e-6 * next.abs().max(1e-300);
            u_tau = next;
            if done {
                break;
            }
        }
        u_tau.max(0.0)
    }

    /// `Ks -> 0` must reproduce the smooth `nutU` wall to round-off - the §22
    /// gate, checked against [`smooth_u_tau_newton_reference`] rather than
    /// against [`u_tau_newton`] agreeing with itself.
    #[test]
    fn ks_zero_reproduces_the_smooth_nutu_wall_everywhere() {
        let nu: Scalar = 1.5e-5;
        let y: Scalar = 2e-3;

        for u_mag in [0.05 as Scalar, 0.5, 2.0, 10.0] {
            let want = smooth_u_tau_newton_reference(u_mag, y, nu, KAPPA, E);

            for cs in [0.3 as Scalar, 0.5, 1.0] {
                let got = u_tau_newton(u_mag, y, nu, KAPPA, E, 0.0, cs);
                assert!(
                    (got - want).abs() <= 1e-10 * want.max(1e-30),
                    "u_mag={u_mag} cs={cs}: rough(Ks=0) u_tau {got}, independent smooth {want}"
                );
            }

            let want_nut = nut_wall_u(want, y, nu, u_mag);
            let got_nut = nut_wall_rough_u(u_mag, y, nu, KAPPA, E, 0.0, 0.5);
            assert!(
                (got_nut - want_nut).abs() <= 1e-10 * want_nut.max(1e-30),
                "u_mag={u_mag}: rough(Ks=0) nu_t,w {got_nut}, independent smooth {want_nut}"
            );
        }
    }

    /// One boundary-face flag per patch face, `on` on `p`'s faces and `off`
    /// everywhere else - the setup every device rough-wall test below shares.
    fn wall_flags(m: &HostMesh, p: &crate::mesh::PatchInfo) -> Vec<bool> {
        let mut flags = vec![false; m.n_boundary_faces];
        for i in 0..p.size {
            flags[p.start + i] = true;
        }
        flags
    }

    /// The device `nutk`-family kernel against [`nut_wall_rough_k`], across a
    /// sweep of `Ks` from exactly zero (the §22 gate) through hydraulically
    /// smooth, transitional and fully rough - on a real mesh, not a single
    /// face.
    #[test]
    fn device_rough_nutk_agrees_with_the_host_law() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        // 6 cells, one x-slice each: every cell touches the xmin patch, so
        // there are 6 distinct wall faces/cells to spread the sweep over.
        let d = crate::Vec3::new(0.2, 0.1, 1.0);
        let (mut m, points, faces) = crate::mesh::topology::tests::box_mesh([1, 6, 1], d);
        m.compute_geometry(&points, &faces)?;
        m.build_cell_face_maps();
        let gm = GpuMesh::upload(&gpu, &m)?;

        let p = m.patches[0].clone();
        assert_eq!(p.size, 6);
        let flags = wall_flags(&m, &p);

        // Ks = 0 (the gate), then hydraulically smooth, transitional and
        // fully rough - see the module comment on `nut_wall_rough_k` for how
        // Ks+ relates to Ks at fixed k.
        let cs_val: Scalar = 0.5;
        let ks_vals: [Scalar; 6] = [0.0, 2e-4, 8e-4, 3e-3, 0.02, 0.06];

        let mut ks_v = vec![0.0 as Scalar; m.n_boundary_faces];
        let mut cs_v = vec![0.5 as Scalar; m.n_boundary_faces];
        for i in 0..p.size {
            ks_v[p.start + i] = ks_vals[i];
            cs_v[p.start + i] = cs_val;
        }
        let roughness = crate::field_setup::NutRoughness {
            u_based: vec![false; m.n_boundary_faces],
            ks: ks_v,
            cs: cs_v,
        };

        let wd = WallData::build(&gpu, &m, &same_for_both(&flags), &roughness)?;
        assert_eq!(wd.n_nut_faces, p.size);

        let nu: Scalar = 1e-5;
        let k_min: Scalar = 1e-15;
        let wc = WallFunctionCoeffs::default();
        let k_val: Scalar = 0.01;
        let k_dev = gpu.upload(&vec![k_val; m.n_cells])?;
        let u = GpuVectorField::zeros(&gpu, &gm, "U")?;
        let mut nut_bf = gpu.zeros::<Scalar>(m.n_boundary_faces)?;

        wd.update_nut(&gpu, &mut nut_bf, &k_dev, &u, &gm, &wc, nu, k_min)?;
        gpu.sync()?;

        let got = gpu.download(&nut_bf)?;
        let cmu25 = wc.cmu.powf(0.25);

        for i in 0..p.size {
            let bf = p.start + i;
            let y = m.b_y[bf];
            let y_plus = y_plus_of(k_val, y, nu, wc.cmu);
            let want =
                nut_wall_rough_k(y_plus, k_val, nu, wc.kappa, wc.e, cmu25, ks_vals[i], cs_val);

            assert!(
                (got[bf] - want).abs() <= 1e-11 * want.abs().max(nu),
                "face {bf} (Ks={}): device {}, host {want}",
                ks_vals[i],
                got[bf]
            );

            if ks_vals[i] == 0.0 {
                let smooth = nut_wall(y_plus, nu, wc.kappa, wc.e);
                assert!(
                    (got[bf] - smooth).abs() <= 1e-12 * smooth.abs().max(nu),
                    "face {bf}: Ks=0 gave {} on the device, smooth wall gives {smooth}",
                    got[bf]
                );
            }
        }

        Ok(())
    }

    /// The device `nutU`-family kernel (the Newton solve, SPEC-LIT §15.1)
    /// against [`nut_wall_rough_u`], across the same `Ks` sweep, with a real
    /// wall-parallel velocity driving the Newton on the device exactly as it
    /// does on the host.
    #[test]
    fn device_rough_nutu_agrees_with_the_host_law() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let d = crate::Vec3::new(0.2, 0.1, 1.0);
        let (mut m, points, faces) = crate::mesh::topology::tests::box_mesh([1, 6, 1], d);
        m.compute_geometry(&points, &faces)?;
        m.build_cell_face_maps();
        let gm = GpuMesh::upload(&gpu, &m)?;

        let p = m.patches[0].clone();
        assert_eq!(p.size, 6);
        let flags = wall_flags(&m, &p);

        let cs_val: Scalar = 0.5;
        let ks_vals: [Scalar; 6] = [0.0, 2e-4, 8e-4, 3e-3, 0.02, 0.06];

        let mut u_based = vec![false; m.n_boundary_faces];
        let mut ks_v = vec![0.0 as Scalar; m.n_boundary_faces];
        let mut cs_v = vec![0.5 as Scalar; m.n_boundary_faces];
        for i in 0..p.size {
            u_based[p.start + i] = true;
            ks_v[p.start + i] = ks_vals[i];
            cs_v[p.start + i] = cs_val;
        }
        let roughness = crate::field_setup::NutRoughness {
            u_based,
            ks: ks_v,
            cs: cs_v,
        };

        let wd = WallData::build(&gpu, &m, &same_for_both(&flags), &roughness)?;
        assert_eq!(wd.n_nut_faces, p.size);

        let nu: Scalar = 1e-5;
        let k_min: Scalar = 1e-15;
        let wc = WallFunctionCoeffs::default();
        // Not read by the u-based branch, but still has to be the right
        // shape - `update_nut` checks `k` unconditionally.
        let k_dev = gpu.zeros::<Scalar>(m.n_cells)?;

        // A different tangential velocity per cell, so the Newton solves a
        // different u_tau at every face. No-slip: `u.bf` stays zero.
        let mut u = GpuVectorField::zeros(&gpu, &gm, "U")?;
        let uc: Vec<crate::Vec3> = (0..m.n_cells)
            .map(|c| crate::Vec3::new(0.3, 0.5 + 0.15 * c as Scalar, 0.2))
            .collect();
        gpu.write(&mut u.f, &uc)?;

        let mut nut_bf = gpu.zeros::<Scalar>(m.n_boundary_faces)?;

        wd.update_nut(&gpu, &mut nut_bf, &k_dev, &u, &gm, &wc, nu, k_min)?;
        gpu.sync()?;

        let got = gpu.download(&nut_bf)?;

        for i in 0..p.size {
            let bf = p.start + i;
            let c = m.b_face_cells[bf] as usize;
            let y = m.b_y[bf];

            let n = m.b_sf[bf] / m.b_mag_sf[bf];
            let t = uc[c] - n * uc[c].dot(n);
            let u_mag = t.mag();

            let want =
                nut_wall_rough_u(u_mag, y, nu, wc.kappa, wc.e, ks_vals[i], cs_val);

            assert!(
                (got[bf] - want).abs() <= 1e-9 * want.abs().max(nu),
                "face {bf} (Ks={}, |U|={u_mag}): device {}, host {want}",
                ks_vals[i],
                got[bf]
            );

            if ks_vals[i] == 0.0 {
                let u_tau = u_tau_newton(u_mag, y, nu, wc.kappa, wc.e, 0.0, cs_val);
                let smooth = nut_wall_u(u_tau, y, nu, u_mag);
                assert!(
                    (got[bf] - smooth).abs() <= 1e-9 * smooth.abs().max(nu),
                    "face {bf}: Ks=0 gave {} on the device, smooth nutU gives {smooth}",
                    got[bf]
                );
            }
        }

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  The Jayatilleke thermal wall function - SPEC-LIT §29.3
    // ----------------------------------------------------------------------

    /// `P(1) = 0` exactly, whatever the second bracket is.
    #[test]
    fn jayatilleke_p_is_exactly_zero_at_pr_equals_prt() {
        for prt in [0.7, 0.85, 1.0, 1.3] {
            assert_eq!(jayatilleke_p(prt, prt), 0.0, "Pr = Pr_t = {prt}");
        }
    }

    /// SPEC-LIT §32.3: [`dittus_boelter_nu`]/[`gnielinski_nu`] against a hand
    /// re-derivation written a DIFFERENT way (`exp(n ln x)` for the power
    /// laws rather than `powf`, the friction factor inlined rather than
    /// through [`gnielinski_f`]) - the "independent host mirror" category
    /// this file's own validate.rs promotion note asks for: two structurally
    /// different expressions landing on the same number is evidence a typo'd
    /// exponent or a transposed term would not survive.
    #[test]
    fn nu_correlations_match_an_independently_written_derivation() {
        for (re, pr) in [(1.0e4, 0.71), (1.6e4, 0.71), (1.0e5, 7.0), (5.0e5, 0.6)] {
            let db = dittus_boelter_nu(re, pr);
            let db_mirror = 0.023 * (0.8 * re.ln()).exp() * (0.4 * pr.ln()).exp();
            assert!(
                (db - db_mirror).abs() <= 1e-9 * db_mirror,
                "Re {re} Pr {pr}: dittus_boelter_nu {db}, mirror {db_mirror}"
            );

            let f = gnielinski_f(re);
            let f_mirror = {
                let d = 0.79 * re.ln() - 1.64;
                (d * d).recip()
            };
            assert!(
                (f - f_mirror).abs() <= 1e-12 * f_mirror,
                "Re {re}: gnielinski_f {f}, mirror {f_mirror}"
            );

            let gn = gnielinski_nu(re, pr);
            let sqrt_f8 = (f / 8.0).sqrt();
            let gn_mirror =
                (f / 8.0) * (re - 1000.0) * pr / (1.0 + 12.7 * sqrt_f8 * ((2.0 / 3.0 * pr.ln()).exp() - 1.0));
            assert!(
                (gn - gn_mirror).abs() <= 1e-9 * gn_mirror.abs().max(1.0),
                "Re {re} Pr {pr}: gnielinski_nu {gn}, mirror {gn_mirror}"
            );
        }
    }

    /// A closed-form numeric pin at the two channel cases' own operating
    /// point (SPEC-LIT §32.2: Re ~ 1.6e4, Pr = 0.71) - computed independently
    /// by hand to 10 significant figures and asserted tightly, so a future
    /// change to either formula is caught here rather than only downstream in
    /// the two-mesh comparison.
    #[test]
    fn nu_correlations_at_the_channel_operating_point() {
        let re: Scalar = 1.6e4;
        let pr: Scalar = 0.71;
        assert!(
            (dittus_boelter_nu(re, pr) - 46.294_261_62).abs() < 1e-4,
            "Nu_DB = {}",
            dittus_boelter_nu(re, pr)
        );
        assert!(
            (gnielinski_f(re) - 0.027_708_723_8).abs() < 1e-9,
            "f = {}",
            gnielinski_f(re)
        );
        assert!(
            (gnielinski_nu(re, pr) - 43.528_672_6).abs() < 1e-3,
            "Nu_Gn = {}",
            gnielinski_nu(re, pr)
        );
    }

    /// SPEC-LIT §29.3's own consistency check: at `Pr = Pr_t`, `T+` reduces
    /// to `Pr_t · u+` because `t_vis = Pr_t·y+ `, `t_log = Pr_t·u_log` and
    /// both share `u_plus`'s own blend weights exactly.
    #[test]
    fn t_plus_reduces_to_prt_times_u_plus_when_pr_equals_prt() {
        let prt: Scalar = 0.85;
        let p = jayatilleke_p(prt, prt);
        assert_eq!(p, 0.0);

        for y_plus in [0.0, 1.0, 5.0, 11.53, 30.0, 100.0, 1000.0] {
            let tp = t_plus(y_plus, prt, prt, KAPPA, E, p);
            let want = prt * u_plus(y_plus, KAPPA, E);
            assert!(
                (tp - want).abs() <= 1e-9 * want.abs().max(1.0),
                "y+ {y_plus}: T+ {tp}, Pr_t*u+ {want}"
            );
        }
    }

    /// The blend has no switch to jump at - `t_plus` is built from the SAME
    /// `blend_gamma`/`log_weight` [`u_plus`] uses, which is smooth by
    /// construction. This pins that down concretely at the "thermal
    /// crossover" - the `y+` where the two RAW branches (`Pr y+` and
    /// `Pr_t(u_log + P)`) cross, found by bisection for a `Pr != Pr_t` case -
    /// by evaluating `t_plus` from both sides and checking it moves smoothly
    /// through the crossing rather than jumping.
    #[test]
    fn t_plus_is_continuous_across_the_thermal_crossover() {
        let pr: Scalar = 0.71;
        let prt: Scalar = 0.85;
        let p = jayatilleke_p(pr, prt);

        let raw_gap = |y_plus: Scalar| -> Scalar {
            let u_log = (E * y_plus).max(1.0).ln() / KAPPA;
            pr * y_plus - prt * (u_log + p)
        };

        // Bisect for the crossing: `raw_gap` is negative near the wall
        // (`Pr y+` is small) and positive far from it (the log branch grows
        // like `ln y+` while the viscous one grows LINEARLY... but Pr < Pr_t
        // here keeps the viscous branch below the log one until quite far
        // out - bracket generously and let bisection find it).
        let mut lo: Scalar = 1.0;
        let mut hi: Scalar = 2000.0;
        assert!(raw_gap(lo) < 0.0 && raw_gap(hi) > 0.0, "bad bracket");
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if raw_gap(mid) < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let crossover = 0.5 * (lo + hi);
        assert!(raw_gap(crossover).abs() < 1e-6, "bisection did not converge");

        let eps: Scalar = 1e-4;
        let left = t_plus(crossover - eps, pr, prt, KAPPA, E, p);
        let mid = t_plus(crossover, pr, prt, KAPPA, E, p);
        let right = t_plus(crossover + eps, pr, prt, KAPPA, E, p);

        // A smooth function moves by O(eps) across a step of size eps; a
        // switched one would jump by O(1) (the two raw branches differ by a
        // large factor away from `y+_lam`-scale y+, and this crossover is
        // nowhere near either endpoint of the blend).
        assert!(
            (left - mid).abs() < 1e-2 && (mid - right).abs() < 1e-2,
            "T+ jumps at the thermal crossover y+={crossover}: {left}, {mid}, {right}"
        );
    }

    /// The `ref_grad` [`thermal_wall_ref_grad`] returns encodes EXACTLY the
    /// analytic Jayatilleke flux, by construction - the "one-cell energy
    /// balance" SPEC-LIT §29.3 asks for: `k_eff_wall * ref_grad` must equal
    /// `rho cp u_tau (T_w - T_P)/T+`.
    #[test]
    fn thermal_wall_ref_grad_encodes_the_analytic_flux() {
        let wc = WallFunctionCoeffs::default();
        let nu: Scalar = 1.5e-5;
        let k_min: Scalar = 1e-15;

        let k_p: Scalar = 0.05; // an arbitrary wall-adjacent k
        let y: Scalar = 0.01;
        let rho: Scalar = 1.2;
        let cp: Scalar = 1006.0;
        let pr: Scalar = 0.71;
        let prt: Scalar = 0.85;
        let k_eff_wall: Scalar = 0.04; // molecular-plus-eddy conductivity
        let t_w: Scalar = 400.0;
        let t_p: Scalar = 300.0;

        let grad = thermal_wall_ref_grad(
            t_w, t_p, k_p, y, nu, rho, cp, pr, prt, wc.kappa, wc.e, wc.cmu, k_eff_wall, k_min,
        )
        .expect("a valid standoff and k_eff must produce a ref_grad");

        // Recompute q_w independently, from the same law but written out
        // longhand rather than through the function under test.
        let y_plus = y_plus_of(k_p, y, nu, wc.cmu);
        let u_tau = wc.cmu.powf(0.25) * k_p.sqrt();
        let tp = t_plus(y_plus, pr, prt, wc.kappa, wc.e, jayatilleke_p(pr, prt));
        let q_w = rho * cp * u_tau * (t_w - t_p) / tp;

        let flux_from_triple = k_eff_wall * grad;
        assert!(
            (flux_from_triple - q_w).abs() <= 1e-9 * q_w.abs(),
            "triple encodes flux {flux_from_triple}, analytic q_w is {q_w}"
        );

        // And the sign is physical: a hotter wall than cell (T_w > T_P)
        // drives heat INTO the domain, a positive outward... this is the
        // wall's own convention (SPEC-LIT §29.3's q_w), so q_w > 0 here.
        assert!(q_w > 0.0, "T_w > T_P must give a positive q_w, got {q_w}");

        // No standoff, or no conductivity to divide by: `None`, not a NaN or
        // an infinity a caller could carry into the matrix unnoticed.
        assert!(thermal_wall_ref_grad(
            t_w, t_p, k_p, 0.0, nu, rho, cp, pr, prt, wc.kappa, wc.e, wc.cmu, k_eff_wall, k_min
        )
        .is_none());
        assert!(thermal_wall_ref_grad(
            t_w, t_p, k_p, y, nu, rho, cp, pr, prt, wc.kappa, wc.e, wc.cmu, 0.0, k_min
        )
        .is_none());
    }

    /// The device kernel `wfThermalWall` must reproduce
    /// [`thermal_wall_ref_grad`] bit-for-bit up to floating-point tolerance -
    /// the same discipline `device_agrees_with_the_host_law` holds `nut_wall`
    /// to.
    #[test]
    fn thermal_wall_device_agrees_with_the_host_law() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        // A one-cell-thick slab: 4 cells in a row, the xmin patch a wall.
        let (mut m, points, faces) =
            crate::mesh::topology::tests::box_mesh([4, 1, 1], crate::Vec3::new(0.25, 1.0, 1.0));
        m.compute_geometry(&points, &faces)?;
        m.build_cell_face_maps();
        let gm = GpuMesh::upload(&gpu, &m)?;

        let mut flags = vec![false; m.n_boundary_faces];
        let p = &m.patches[0];
        for i in 0..p.size {
            flags[p.start + i] = true;
        }

        let twd = ThermalWallData::build(&gpu, &flags)?;
        assert_eq!(twd.n_faces, p.size);

        let wc = WallFunctionCoeffs::default();
        let nu: Scalar = 1.5e-5;
        let cp: Scalar = 1006.0;
        let pr: Scalar = 0.71;
        let prt: Scalar = 0.85;
        let k_min: Scalar = 1e-15;

        // k chosen so y+ straddles y+_lam, same choice
        // `device_agrees_with_the_host_law` makes for nu_t.
        let k_host = vec![2.0e-6 as Scalar; m.n_cells];
        let k_dev = gpu.upload(&k_host)?;
        let rho_host = vec![1.2 as Scalar; m.n_cells];
        let rho_dev = gpu.upload(&rho_host)?;
        // A non-trivial k_eff per face, standing in for `energy::update_k_eff`.
        let k_eff_host: Vec<Scalar> = (0..m.n_boundary_faces).map(|i| 0.03 + 0.001 * i as Scalar).collect();
        let k_eff_dev = gpu.upload(&k_eff_host)?;

        let t_w: Scalar = 400.0;
        let t_host = vec![300.0 as Scalar; m.n_cells];
        let t_dev = gpu.upload(&t_host)?;

        let mut fr = gpu.upload(&vec![1.0 as Scalar; m.n_boundary_faces])?;
        let mut ref_grad = gpu.zeros::<Scalar>(m.n_boundary_faces)?;
        let ref_value = gpu.upload(&vec![t_w; m.n_boundary_faces])?;

        twd.update(
            &gpu,
            &mut fr,
            &mut ref_grad,
            &ref_value,
            &t_dev,
            &k_dev,
            &rho_dev,
            &k_eff_dev,
            &gm,
            &wc,
            nu,
            cp,
            pr,
            prt,
            k_min,
        )?;
        gpu.sync()?;

        let got_fr = gpu.download(&fr)?;
        let got_grad = gpu.download(&ref_grad)?;
        let face_ids = gpu.download(&twd.face)?;

        for &bf in face_ids.iter().take(twd.n_faces) {
            let bf = bf as usize;
            let y = m.b_y[bf];
            let c = m.b_face_cells[bf] as usize;

            let want = thermal_wall_ref_grad(
                t_w,
                t_host[c],
                k_host[c],
                y,
                nu,
                rho_host[c],
                cp,
                pr,
                prt,
                wc.kappa,
                wc.e,
                wc.cmu,
                k_eff_host[bf],
                k_min,
            )
            .expect("every face in this test has a positive standoff and k_eff");

            assert_eq!(got_fr[bf], 0.0, "face {bf}: fr must be rewritten to 0");
            assert!(
                (got_grad[bf] - want).abs() <= 1e-9 * want.abs().max(1e-6),
                "face {bf}: device ref_grad {}, host {want}",
                got_grad[bf]
            );
        }

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  Werner-Wengle - SPEC-LIT §30.1
    // ----------------------------------------------------------------------

    /// **The test SPEC-LIT §30.1's continuity requirement exists for.** The
    /// power branch and the viscous branch are two different closed-form
    /// expressions; nothing in [`tau_w_werner_wengle`] forces them to agree
    /// except the algebra the module section above works out. Evaluate both
    /// sides of the branch point directly.
    #[test]
    fn ww_is_continuous_at_the_branch_point() {
        for (nu, h) in [
            (1.5e-5 as Scalar, 0.01 as Scalar),
            (1.0e-6, 0.002),
            (2.0e-4, 0.05),
        ] {
            let u_c = ww_branch_speed(nu, h);
            assert!(u_c > 0.0, "nu {nu} h {h}: branch speed is not positive");

            // At the point itself: the viscous closed form, since `<=` picks
            // that branch there.
            let at = tau_w_werner_wengle(u_c, h, nu);
            let viscous_closed_form = 2.0 * nu * u_c / h;
            assert!(
                (at - viscous_closed_form).abs() < 1e-9 * viscous_closed_form,
                "nu {nu} h {h}: tau_w(u_c) = {at}, 2 nu u_c/h = {viscous_closed_form}"
            );

            // A hair below and a hair above, evaluated from EACH SIDE's own
            // formula (a tiny perturbation of u_p is guaranteed to fall on
            // one side or the other of `<=`).
            let below = tau_w_werner_wengle(u_c * (1.0 - 1e-9), h, nu);
            let above = tau_w_werner_wengle(u_c * (1.0 + 1e-9), h, nu);
            let scale = at.max(1e-300);
            assert!(
                (below - at).abs() < 1e-6 * scale,
                "nu {nu} h {h}: viscous side steps by {} crossing the branch point",
                (below - at).abs() / scale
            );
            assert!(
                (above - at).abs() < 1e-6 * scale,
                "nu {nu} h {h}: power side steps by {} crossing the branch point",
                (above - at).abs() / scale
            );
        }
    }

    /// Inverting the integrated power law reproduces a manufactured `tau_w`
    /// to round-off - SPEC-LIT §30.3's own wording for this gate.
    #[test]
    fn ww_power_branch_inverts_a_manufactured_tau_w_to_round_off() {
        let nu: Scalar = 1.5e-5;
        let h: Scalar = 0.01;
        let a = WW_A;
        let b = WW_B;
        let nu_h = nu / h;

        // The power branch's own bracket, `tau_w = (t1 + t2 u_p)^{2/(1+b)}`,
        // inverted for `u_p` given a target `tau_w`.
        let t1 = 0.5 * (1.0 - b) * a.powf((1.0 + b) / (1.0 - b)) * nu_h.powf(1.0 + b);
        let t2 = ((1.0 + b) / a) * nu_h.powf(b);

        for tau_w_target in [1.0e-3 as Scalar, 5.0e-2, 2.0] {
            let u_p = (tau_w_target.powf((1.0 + b) / 2.0) - t1) / t2;
            assert!(
                u_p > ww_branch_speed(nu, h),
                "test setup: tau_w {tau_w_target} landed in the viscous branch, not the power one"
            );

            let got = tau_w_werner_wengle(u_p, h, nu);
            assert!(
                (got - tau_w_target).abs() < 1e-9 * tau_w_target,
                "tau_w = {tau_w_target}: inverting and reapplying the power law gave {got}"
            );
        }
    }

    /// The viscous branch's own round trip - linear, so the inversion is
    /// exact algebra rather than a root-find, but the same discipline
    /// applies: manufacture `tau_w`, invert, reapply, compare.
    #[test]
    fn ww_viscous_branch_inverts_a_manufactured_tau_w_to_round_off() {
        let nu: Scalar = 1.5e-5;
        let h: Scalar = 0.01;

        for tau_w_target in [1.0e-8 as Scalar, 1.0e-10] {
            let u_p = tau_w_target * h / (2.0 * nu);
            assert!(
                u_p <= ww_branch_speed(nu, h),
                "test setup: tau_w {tau_w_target} landed in the power branch, not the viscous one"
            );

            let got = tau_w_werner_wengle(u_p, h, nu);
            assert!(
                (got - tau_w_target).abs() < 1e-9 * tau_w_target.max(1e-300),
                "tau_w = {tau_w_target}: inverting and reapplying the viscous law gave {got}"
            );
        }
    }

    /// `nu_t,w` is well defined only where there IS a wall-parallel speed to
    /// divide by, and non-negative everywhere else it is defined.
    #[test]
    fn ww_nut_wall_is_zero_with_no_wall_parallel_speed_and_never_negative() {
        assert_eq!(nut_wall_werner_wengle(1.0, 0.01, 0.0, 1.5e-5), 0.0);
        assert_eq!(nut_wall_werner_wengle(1.0, 0.0, 0.5, 1.5e-5), 0.0);

        for u_p in [1e-4 as Scalar, 1e-2, 1.0, 10.0] {
            let nu: Scalar = 1.5e-5;
            let h: Scalar = 0.01;
            let tau_w = tau_w_werner_wengle(u_p, h, nu);
            let nut = nut_wall_werner_wengle(tau_w, h, u_p, nu);
            assert!(nut >= 0.0, "u_p {u_p}: nu_t,w = {nut}");
        }
    }

    #[test]
    fn u_tau_werner_wengle_is_the_square_root_of_tau_w() {
        assert_eq!(u_tau_werner_wengle(4.0), 2.0);
        assert_eq!(u_tau_werner_wengle(0.0), 0.0);
        assert_eq!(u_tau_werner_wengle(-1.0), 0.0);
    }

    /// The device kernel `wfWernerWengle` must reproduce
    /// [`tau_w_werner_wengle`]/[`nut_wall_werner_wengle`] - the same
    /// discipline `device_agrees_with_the_host_law` holds the RAS `nutk`
    /// kernel to.
    #[test]
    fn device_ww_agrees_with_the_host_law() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        // A one-cell-thick slab: 4 cells in a row, the xmin patch a wall.
        let (mut m, points, faces) =
            crate::mesh::topology::tests::box_mesh([4, 1, 1], crate::Vec3::new(0.25, 1.0, 1.0));
        m.compute_geometry(&points, &faces)?;
        m.build_cell_face_maps();
        let gm = GpuMesh::upload(&gpu, &m)?;

        let mut flags = vec![false; m.n_boundary_faces];
        let p = &m.patches[0];
        for i in 0..p.size {
            flags[p.start + i] = true;
        }

        let mut wwd = WernerWengleData::build(&gpu, &flags)?;
        assert_eq!(wwd.n_faces, p.size);

        let nu: Scalar = 1.5e-5;
        let mut u = GpuVectorField::zeros(&gpu, &gm, "U")?;
        // Tangential to the xmin wall (its normal is x): a pure-y velocity
        // has no normal component to project out, so wfMagUParallel returns
        // it exactly. The wall's own boundary value is left at the zero
        // `GpuVectorField::zeros` already gives it - no-slip.
        let u_mag: Scalar = 0.7;
        let u_host = vec![crate::Vec3::new(0.0, u_mag, 0.0); m.n_cells];
        gpu.write(&mut u.f, &u_host)?;

        let mut nut_bf = gpu.zeros::<Scalar>(m.n_boundary_faces)?;
        wwd.update_nut(&gpu, &mut nut_bf, &u, &gm, nu)?;
        gpu.sync()?;

        let got_nut = gpu.download(&nut_bf)?;
        let got_tau = gpu.download(&wwd.tau_w)?;
        let face_ids = gpu.download(&wwd.face)?;

        for &bf in face_ids.iter().take(wwd.n_faces) {
            let bf = bf as usize;
            let h = m.b_y[bf];
            let want_tau = tau_w_werner_wengle(u_mag, h, nu);
            let want_nut = nut_wall_werner_wengle(want_tau, h, u_mag, nu);

            assert!(
                (got_tau[bf] - want_tau).abs() <= 1e-9 * want_tau.max(1e-30),
                "face {bf}: tau_w device {} host {want_tau}",
                got_tau[bf]
            );
            assert!(
                (got_nut[bf] - want_nut).abs() <= 1e-9 * want_nut.max(nu),
                "face {bf}: nu_t,w device {} host {want_nut}",
                got_nut[bf]
            );
        }

        Ok(())
    }

    /// The `u_tau = sqrt(tau_w)` substitution, wired end to end: build a
    /// Werner-Wengle wall, run its kernel, feed the `tau_w` it wrote straight
    /// into [`ThermalWallData::update_from_tau_w`], and check the rewritten
    /// `ref_grad` against the pure host [`thermal_wall_ref_grad_from_u_tau`]
    /// evaluated at `u_tau = sqrt(tau_w)`.
    #[test]
    fn device_thermal_wall_u_tau_agrees_with_the_host_law() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let (mut m, points, faces) =
            crate::mesh::topology::tests::box_mesh([4, 1, 1], crate::Vec3::new(0.25, 1.0, 1.0));
        m.compute_geometry(&points, &faces)?;
        m.build_cell_face_maps();
        let gm = GpuMesh::upload(&gpu, &m)?;

        let mut flags = vec![false; m.n_boundary_faces];
        let p = &m.patches[0];
        for i in 0..p.size {
            flags[p.start + i] = true;
        }

        let nu: Scalar = 1.5e-5;
        let mut wwd = WernerWengleData::build(&gpu, &flags)?;
        let mut u = GpuVectorField::zeros(&gpu, &gm, "U")?;
        let u_mag: Scalar = 0.7;
        gpu.write(&mut u.f, &vec![crate::Vec3::new(0.0, u_mag, 0.0); m.n_cells])?;
        let mut nut_bf = gpu.zeros::<Scalar>(m.n_boundary_faces)?;
        wwd.update_nut(&gpu, &mut nut_bf, &u, &gm, nu)?;
        gpu.sync()?;

        // Same `nut`-owned faces feed the thermal wall function - realistic
        // for a case that has both.
        let twd = ThermalWallData::build(&gpu, &flags)?;

        let wc = WallFunctionCoeffs::default();
        let cp: Scalar = 1006.0;
        let pr: Scalar = 0.71;
        let prt: Scalar = 0.85;

        let rho_host = vec![1.2 as Scalar; m.n_cells];
        let rho_dev = gpu.upload(&rho_host)?;
        let k_eff_host: Vec<Scalar> =
            (0..m.n_boundary_faces).map(|i| 0.03 + 0.001 * i as Scalar).collect();
        let k_eff_dev = gpu.upload(&k_eff_host)?;

        let t_w: Scalar = 400.0;
        let t_host = vec![300.0 as Scalar; m.n_cells];
        let t_dev = gpu.upload(&t_host)?;

        let mut fr = gpu.upload(&vec![1.0 as Scalar; m.n_boundary_faces])?;
        let mut ref_grad = gpu.zeros::<Scalar>(m.n_boundary_faces)?;
        let ref_value = gpu.upload(&vec![t_w; m.n_boundary_faces])?;

        twd.update_from_tau_w(
            &gpu,
            &mut fr,
            &mut ref_grad,
            &ref_value,
            &t_dev,
            &wwd.tau_w,
            &rho_dev,
            &k_eff_dev,
            &gm,
            &wc,
            nu,
            cp,
            pr,
            prt,
        )?;
        gpu.sync()?;

        let got_fr = gpu.download(&fr)?;
        let got_grad = gpu.download(&ref_grad)?;
        let got_tau = gpu.download(&wwd.tau_w)?;
        let face_ids = gpu.download(&twd.face)?;

        for &bf in face_ids.iter().take(twd.n_faces) {
            let bf = bf as usize;
            let y = m.b_y[bf];
            let c = m.b_face_cells[bf] as usize;
            let u_tau = u_tau_werner_wengle(got_tau[bf]);

            let want = thermal_wall_ref_grad_from_u_tau(
                t_w,
                t_host[c],
                u_tau,
                y,
                nu,
                rho_host[c],
                cp,
                pr,
                prt,
                wc.kappa,
                wc.e,
                k_eff_host[bf],
            )
            .expect("every face in this test has a positive standoff, u_tau and k_eff");

            assert_eq!(got_fr[bf], 0.0, "face {bf}: fr must be rewritten to 0");
            assert!(
                (got_grad[bf] - want).abs() <= 1e-9 * want.abs().max(1e-6),
                "face {bf}: device ref_grad {}, host {want}",
                got_grad[bf]
            );
        }

        Ok(())
    }
}
