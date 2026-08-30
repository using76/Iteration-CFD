// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Fan curves and porous jumps - SPEC-LIT §52 and §53.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §52.2 - the patch-integral coupling as a symmetric
//!     rank-1 downdate `A = diag(D) - kappa d d^T`
//!   ofgpu `SPEC-LIT.md` §52.3 - the row-sum-preserving lumping,
//!     `fr = 1/(1 + S SIGMA_D)`, `refValue = c + S Phi`, and why those are
//!     not the design note's `1/(1 + S A rAU_f Delta_f)` and
//!     `c + (S A/a_f) phi_HbyA,f`
//!   ofgpu `SPEC-LIT.md` §52.5 - the monotone Hermite curve, the affinity
//!     corrections, and the refusals
//!   ofgpu `SPEC-LIT.md` §52.7 - the determinism argument: every reduction is
//!     a gather plus the existing `solver::device_sum`
//!   ofgpu `SPEC-LIT.md` §53.2/§53.3 - the porous jump as a per-face division
//!     of three arrays by `(1 + R D_f)`
//!   F. N. Fritsch, R. E. Carlson, *SIAM J. Numer. Anal.* 17 (1980) 238-246 -
//!     the monotone slope limiter
//!   W. H. Press et al., *Numerical Recipes* 3rd ed. §3.3 - the Hermite basis
//!   AMCA 210 / ASHRAE 51 - what a manufacturer's curve is measured at, and
//!     hence why (S52.13) has a density and a speed correction
//!   FDS 6 (NIST, US Government public domain; `reference/fds/LICENSE.md`
//!     read verbatim) - the DISCIPLINE that the density scaling is applied at
//!     every evaluation, and the WARNING that its tabulated branch resolves
//!     the operating point by a bisection with a data-dependent trip count.
//!     Its `Verification/HVAC/fan_test.fds` and `qfan_test.fds` case decks
//!     and their published CSVs are the external cross-check of §52.12
//!     Gate 52-B - data, not source.
//! No GPL-licensed source was consulted. OpenFOAM's `fanPressure`, `fan` and
//! `porousBafflePressure` were not opened.
//!
//! # The one thing worth knowing before reading the code
//!
//! A fan curve looks like it destroys the pressure matrix and does not.
//! Linearising the pressure-flow characteristic about the current operating
//! point turns the patch-integral coupling into a provably **symmetric**
//! rank-1 downdate, whose row-sum-preserving lumping is exactly one Robin
//! triple. Symmetry survives, the M-matrix property survives (the diagonal
//! gains `fr D_f >= 0`, so the solve gets *easier*), no new storage is
//! needed, and a flat curve degenerates to `fixedValue` **bitwise**.
//!
//! That last property is the regression test for the whole module:
//! [`FanCurve::flat`] must reproduce the existing `fixedValue` answer
//! exactly, not nearly.
//!
//! # The one correction to the design note that a user can trip over
//!
//! The note says the velocity side of a fan patch needs
//! `pressureInletOutletVelocity`, "exactly right - the flux sets the normal
//! component on inflow". **In this solver it is not.** `field_setup` seeds
//! kind 12's `refValue` from the interior velocity once, nothing refreshes it
//! from the flux, and `momFluxIsPrescribed` treats any `fr >= 1` face as a
//! prescribed velocity - so an inflow face is pinned at whatever it was
//! seeded with (zero, on a room starting from rest) and the fan's pressure
//! moves no air at all. Use a plain **`zeroGradient`** on `U`: `fr = 0` makes
//! `momFluxIsPrescribed` false and the PRESSURE equation owns the flux, which
//! is the entire point of putting a fan on `p`. SPEC-LIT §52.10 records the
//! measurement.
//!
//! # And the one thing worth knowing about the jump
//!
//! It is smaller still. `rAU_f`, `rAU_f|Sf|` and `phi_HbyA,f` are all divided
//! by the same `(1 + R D_f)` on the listed faces, `fvm_laplacian` is called
//! unmodified, `upper[f]` and `lower[f]` get the same reduced coefficient so
//! symmetry is identical, and `R = 0` is `x/1.0 == x` - bitwise inert.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::io::contract::unsupported_note;
use crate::mesh::{GpuMesh, HostMesh};
use crate::solver::{self, SolverKernels};
use crate::{Label, Scalar};

#[cfg(test)]
mod tests;

// ==========================================================================
//  1. The curve - SPEC-LIT §52.5
// ==========================================================================

/// The largest table a curve may carry.
///
/// A fixed bound is what lets `fanCurveAt`'s table scan be a
/// **fixed-trip-count** loop over a compile-time maximum, which is what CUDA
/// Graph capture needs (§52.7). Mirrored by `OFGPU_FAN_MAX_POINTS` in
/// `cuda/fan.cu` and pinned by [`tests::curve_kind_values_match_the_device`].
pub const MAX_CURVE_POINTS: usize = 64;

/// Which of §52.1's four parameterisations a patch carries.
///
/// Stored as an `i32` so the kernel can switch on it; the discriminants are
/// pinned to `cuda/fan.cu`'s `OFGPU_FAN_*` macros by a test.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurveKind {
    /// `dp = dp_0`. `S = 0` everywhere, so this is `fixedValue` (§52.4).
    Constant = 0,
    /// `dp = dp_max [1 - (Q/Q_max)^2]` - FDS `FAN_TYPE 2`, and the curve
    /// §52.12's closed-form gate is written against.
    Quadratic = 1,
    /// A monotone Hermite cubic through manufacturer points.
    Table = 2,
}

/// One fan's pressure-flow characteristic, plus the corrections AMCA 210
/// says a measured curve carries.
///
/// Every field is validated by [`FanCurve::validate`], which is where §52.5's
/// refusal table lives. A curve that reaches the device has already been
/// refused if it was going to be.
#[derive(Debug, Clone, PartialEq)]
pub struct FanCurve {
    pub kind: CurveKind,
    /// `dp_max` for [`CurveKind::Quadratic`]; the constant rise for
    /// [`CurveKind::Constant`]. Pa.
    pub dp_max: Scalar,
    /// Free delivery, m^3/s. Only read by [`CurveKind::Quadratic`].
    pub q_max: Scalar,
    /// `(Q_i, dp_i)` manufacturer points, m^3/s and Pa. Only read by
    /// [`CurveKind::Table`].
    pub points: Vec<(Scalar, Scalar)>,
    /// The density the curve was measured at, kg/m^3 (AMCA 210).
    pub rho_curve: Scalar,
    /// The air density the fan is actually working in, kg/m^3.
    pub rho: Scalar,
    /// The shaft speed the curve was measured at, and the one being run.
    /// Any consistent unit; only the ratio is used.
    pub n_curve: Scalar,
    pub n_speed: Scalar,
    /// Total efficiency, `(0, 1]` - divides the shaft power of (S55.5).
    pub efficiency: Scalar,
}

impl Default for FanCurve {
    /// A dead fan: zero rise, no correction. Chosen so that a `FanCurve`
    /// built and never filled in cannot invent a flow - it is exactly
    /// `fixedValue p = p_a`.
    fn default() -> Self {
        Self {
            kind: CurveKind::Constant,
            dp_max: 0.0,
            q_max: 1.0,
            points: Vec::new(),
            rho_curve: 1.2,
            rho: 1.2,
            n_curve: 1.0,
            n_speed: 1.0,
            efficiency: 1.0,
        }
    }
}

impl FanCurve {
    /// A flat curve at `dp` Pa - §52.4's `S = 0` endpoint, which is
    /// `fixedValue` bitwise.
    pub fn flat(dp: Scalar) -> Self {
        Self { kind: CurveKind::Constant, dp_max: dp, ..Self::default() }
    }

    /// `dp = dp_max [1 - (Q/Q_max)^2]`.
    pub fn quadratic(dp_max: Scalar, q_max: Scalar) -> Self {
        Self { kind: CurveKind::Quadratic, dp_max, q_max, ..Self::default() }
    }

    /// A tabulated curve. Refused unless `Q` is strictly increasing and `dp`
    /// is non-increasing - see [`Self::validate`].
    pub fn table(points: Vec<(Scalar, Scalar)>) -> Self {
        Self { kind: CurveKind::Table, points, ..Self::default() }
    }

    /// `rho/rho_curve` of (S52.13).
    #[inline]
    pub fn rho_ratio(&self) -> Scalar {
        self.rho / self.rho_curve
    }

    /// `N/N_curve` of (S52.13).
    #[inline]
    pub fn speed_ratio(&self) -> Scalar {
        self.n_speed / self.n_curve
    }

    /// SPEC-LIT §52.5's refusal table, in one place.
    ///
    /// `name` is only for the diagnostic; a case with eight fans needs to be
    /// told which one is wrong.
    pub fn validate(&self, name: &str) -> Result<()> {
        let bad = |msg: String| -> Result<()> { Err(Error::Config(msg)) };

        if !(self.rho_curve > 0.0) || !(self.rho > 0.0) {
            return bad(format!(
                "fan \"{name}\": rhoCurve = {} and rho = {} must both be positive - \
                 they are the density correction of SPEC-LIT (S52.13), \
                 dp(rho) = dp_curve rho/rho_curve",
                self.rho_curve, self.rho
            ));
        }
        if !(self.n_curve > 0.0) || !(self.n_speed > 0.0) {
            return bad(format!(
                "fan \"{name}\": speedCurve = {} and speed = {} must both be positive - \
                 they are the affinity correction of SPEC-LIT (S52.13), \
                 dp(N) = dp_curve (N/N_curve)^2",
                self.n_curve, self.n_speed
            ));
        }
        if !(self.efficiency > 0.0) || self.efficiency > 1.0 {
            return bad(format!(
                "fan \"{name}\": efficiency = {} is outside (0, 1] - it divides the \
                 shaft power W_fan = Q dp/eta of SPEC-LIT (S55.5), so a zero or a \
                 value above one would report a power that is not a power",
                self.efficiency
            ));
        }

        match self.kind {
            CurveKind::Constant => {
                if !self.dp_max.is_finite() {
                    return bad(format!(
                        "fan \"{name}\": constantPressure dp = {} is not a number",
                        self.dp_max
                    ));
                }
            }
            CurveKind::Quadratic => {
                if !(self.dp_max > 0.0) || !(self.q_max > 0.0) {
                    return bad(format!(
                        "fan \"{name}\": a quadratic curve needs dpMax > 0 and \
                         QMax > 0 (SPEC-LIT S52.1: dp = dpMax[1 - Q|Q|/QMax^2]); \
                         got dpMax = {}, QMax = {}",
                        self.dp_max, self.q_max
                    ));
                }
            }
            CurveKind::Table => {
                let p = &self.points;
                if p.len() < 2 {
                    return bad(format!(
                        "fan \"{name}\": a tabulated curve needs at least two points, \
                         got {}; available: constantPressure (a flat curve, which is \
                         fixedValue on p), quadratic",
                        p.len()
                    ));
                }
                if p.len() > MAX_CURVE_POINTS {
                    return bad(format!(
                        "fan \"{name}\": {} curve points, but at most {MAX_CURVE_POINTS} \
                         are supported - the bound is what makes the device-side table \
                         scan a fixed-trip-count loop and therefore CUDA-Graph \
                         capturable (SPEC-LIT S52.7)",
                        p.len()
                    ));
                }
                for (i, w) in p.windows(2).enumerate() {
                    if !(w[1].0 > w[0].0) {
                        return bad(format!(
                            "fan \"{name}\": curve point {} has Q = {} and point {} has \
                             Q = {}; the flow rates must be STRICTLY increasing",
                            i, w[0].0, i + 1, w[1].0
                        ));
                    }
                    if w[1].1 > w[0].1 {
                        return bad(format!(
                            "fan \"{name}\": curve rises from dp = {} at Q = {} to \
                             dp = {} at Q = {}. That is a STALL branch: a machine \
                             sitting on it is unstable, and a solver that picked one \
                             of the two intersections would report a fixed point the \
                             machine does not have (SPEC-LIT S52.5/S52.6). \
                             Available: a monotonically non-increasing table, \
                             quadratic, constantPressure",
                            w[0].1, w[0].0, w[1].1, w[1].0
                        ));
                    }
                }
                for (q, dp) in p {
                    if !q.is_finite() || !dp.is_finite() {
                        return bad(format!(
                            "fan \"{name}\": curve point ({q}, {dp}) is not a pair of \
                             numbers"
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// The monotone Hermite node slopes of §52.5, on the host.
    ///
    /// Three-point (Fritsch-Butland-style) interior slopes, then the
    /// **Fritsch & Carlson (1980)** limiter: zero the slope at a local
    /// extremum, and rescale a pair whose `(alpha, beta) = (m_k/d_k,
    /// m_{k+1}/d_k)` leaves the circle of radius 3. Without the limiter a
    /// plain Catmull-Rom spline through four monotone points overshoots and
    /// produces `S < 0` between breakpoints - a stall branch the case did
    /// not ask for.
    pub fn hermite_slopes(&self) -> Vec<Scalar> {
        let p = &self.points;
        let n = p.len();
        if n < 2 {
            return vec![0.0; n];
        }
        let d: Vec<Scalar> =
            (0..n - 1).map(|k| (p[k + 1].1 - p[k].1) / (p[k + 1].0 - p[k].0)).collect();

        let mut m = vec![0.0 as Scalar; n];
        m[0] = d[0];
        m[n - 1] = d[n - 2];
        for k in 1..n - 1 {
            m[k] = if d[k - 1] * d[k] <= 0.0 {
                // A local extremum: the only slope that cannot overshoot.
                0.0
            } else {
                0.5 * (d[k - 1] + d[k])
            };
        }

        // Fritsch & Carlson: keep (alpha, beta) inside the radius-3 circle.
        for k in 0..n - 1 {
            if d[k] == 0.0 {
                m[k] = 0.0;
                m[k + 1] = 0.0;
                continue;
            }
            let a = m[k] / d[k];
            let b = m[k + 1] / d[k];
            let s = a * a + b * b;
            if s > 9.0 {
                let t = 3.0 / s.sqrt();
                m[k] = t * a * d[k];
                m[k + 1] = t * b * d[k];
            }
        }
        m
    }

    /// `(dp, S)` at one flow rate, on the host - the exact mirror of
    /// `fanCurveAt` in `cuda/fan.cu`, and what
    /// [`tests::the_device_curve_mirrors_the_host`] pins the two together on.
    ///
    /// `dp` is Pa; `S = -d(dp)/dQ` is Pa per m^3/s. Both carry (S52.13)'s
    /// density and speed corrections.
    pub fn at(&self, q: Scalar) -> (Scalar, Scalar) {
        let sr = self.speed_ratio();
        let qc = q / sr;
        let (v, s) = match self.kind {
            CurveKind::Constant => (self.dp_max, 0.0),
            // `Q|Q|`, not `Q^2` - see `fanCurveAt` in `cuda/fan.cu` for why
            // the textbook form's evenness is a positive feedback loop on the
            // reverse branch. Identical for `Q >= 0`.
            CurveKind::Quadratic => (
                self.dp_max * (1.0 - qc * qc.abs() / (self.q_max * self.q_max)),
                -2.0 * self.dp_max * qc.abs() / (self.q_max * self.q_max),
            ),
            CurveKind::Table => self.table_at(qc),
        };
        let f = self.rho_ratio() * sr * sr;
        (v * f, -s * f / sr)
    }

    fn table_at(&self, qc: Scalar) -> (Scalar, Scalar) {
        let p = &self.points;
        let n = p.len();
        if n == 0 {
            return (0.0, 0.0);
        }
        if n == 1 {
            return (p[0].1, 0.0);
        }
        let m = self.hermite_slopes();
        let (q0, qn) = (p[0].0, p[n - 1].0);

        if qc <= q0 || qc >= qn {
            // One expression for both tails - see `cuda/fan.cu`.
            let e = if qc <= q0 { 0 } else { n - 1 };
            let me = m[e];
            let d = qc - p[e].0;
            let qref = (qn - q0).abs().max(1e-30);
            let k = (me.abs() / qref).max(p[0].1.abs() / (qref * qref)).max(1e-300);
            return (p[e].1 + me * d - k * d * d.abs(), me - 2.0 * k * d.abs());
        }

        for k in 0..n - 1 {
            let (a, b) = (p[k].0, p[k + 1].0);
            if qc >= a && qc < b {
                let h = b - a;
                let t = (qc - a) / h;
                let (t2, t3) = (t * t, t * t * t);
                let (y0, y1) = (p[k].1, p[k + 1].1);
                let (m0, m1) = (m[k], m[k + 1]);
                let val = (2.0 * t3 - 3.0 * t2 + 1.0) * y0
                    + (t3 - 2.0 * t2 + t) * h * m0
                    + (-2.0 * t3 + 3.0 * t2) * y1
                    + (t3 - t2) * h * m1;
                let der = (6.0 * t2 - 6.0 * t) / h * y0
                    + (3.0 * t2 - 4.0 * t + 1.0) / h * h * m0
                    + (-6.0 * t2 + 6.0 * t) / h * y1
                    + (3.0 * t2 - 2.0 * t) / h * h * m1;
                return (val, der);
            }
        }
        (p[n - 1].1, m[n - 1])
    }

    /// The flow at which the curve first delivers zero pressure - free
    /// delivery, where an unloaded machine sits.
    ///
    /// This is the operating point the FIRST update linearises about when
    /// there is no flux yet (§52.6). Shut-off would be the obvious choice and
    /// is the wrong one: it is where the pressure is MAXIMAL, and on a
    /// quadratic curve it is also where `S = 0`, so the patch starts life as
    /// a `fixedValue` at the full shut-off pressure - the stiffest and most
    /// violent linearisation the curve has. Measured on
    /// `cases/coldAisle.dc.jsonc`, that start put `Q = 135 m^3/s` through a
    /// 35 m^3 room on the second iteration and the outer loop never
    /// recovered. Free delivery starts at `dp = 0`, and the iteration walks
    /// DOWN to the operating point with every intermediate state modest.
    pub fn free_delivery(&self) -> Scalar {
        match self.kind {
            // A flat curve has no free delivery, and `S = 0` makes the seed
            // irrelevant: the triple is the same `fixedValue` at any `Q*`.
            CurveKind::Constant => 0.0,
            CurveKind::Quadratic => self.q_max * self.speed_ratio(),
            CurveKind::Table => {
                let p = &self.points;
                match p.iter().position(|(_, dp)| *dp <= 0.0) {
                    // The first point at or below zero, linearly interpolated
                    // from the one before it. Linear and not the Hermite: this
                    // is a starting guess, and an exact zero crossing of an
                    // interpolant is not what makes it a good one.
                    Some(0) => p[0].0 * self.speed_ratio(),
                    Some(i) => {
                        let (q0, d0) = p[i - 1];
                        let (q1, d1) = p[i];
                        (q0 + (q1 - q0) * d0 / (d0 - d1)) * self.speed_ratio()
                    }
                    // A table that never reaches zero: its last point is as
                    // far out as the case was willing to describe.
                    None => p.last().map_or(0.0, |(q, _)| *q) * self.speed_ratio(),
                }
            }
        }
    }

    /// (S55.5): the shaft power at one operating point, W.
    ///
    /// `q` is the flow **through the device**, `Q_dev = sigma Q`. A negative
    /// `Q_dev` (a machine being driven backwards) gives a negative power,
    /// which is reported rather than hidden: it says the fan is not doing
    /// the work.
    pub fn shaft_power(&self, q_dev: Scalar) -> Scalar {
        let (dp, _) = self.at(q_dev);
        q_dev * dp / self.efficiency
    }
}

/// Which way the device pushes across the patch - `sigma` of (S52.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FanDirection {
    /// An exhaust: the fan discharges out across the patch. `sigma = +1`.
    Outflow,
    /// A supply blower: the fan pushes in across the patch. `sigma = -1`.
    Inflow,
}

impl FanDirection {
    #[inline]
    pub fn sigma(self) -> Scalar {
        match self {
            Self::Outflow => 1.0,
            Self::Inflow => -1.0,
        }
    }

    /// SPEC-LIT §13.4: a direction this solver does not have is an error
    /// naming the two it does.
    pub fn from_name(name: &str, patch: &str) -> Result<Self> {
        match name {
            "outflow" | "exhaust" | "discharge" => Ok(Self::Outflow),
            "inflow" | "supply" | "intake" => Ok(Self::Inflow),
            other => Err(Error::Config(format!(
                "fan patch \"{patch}\": direction \"{other}\" is not supported by \
                 ofgpu; available: outflow (the fan discharges out across the patch), \
                 inflow (a supply blower pushing in). SPEC-LIT (S52.3): the direction \
                 is the sign sigma in p_b = p_a - sigma F(sigma Q)"
            ))),
        }
    }
}

/// One fan patch, as a case describes it.
#[derive(Debug, Clone, PartialEq)]
pub struct FanPatch {
    /// The patch this condition owns. Must name a patch of the mesh.
    pub patch: String,
    pub curve: FanCurve,
    pub direction: FanDirection,
    /// Ambient **kinematic** pressure on the far side, m^2/s^2. `p_a` of
    /// (S52.3).
    pub ambient: Scalar,
    /// (S52.14)'s under-relaxation of the operating point.
    pub relaxation: Scalar,
}

impl FanPatch {
    pub fn new(patch: impl Into<String>, curve: FanCurve, direction: FanDirection) -> Self {
        Self { patch: patch.into(), curve, direction, ambient: 0.0, relaxation: 0.5 }
    }

    fn validate(&self) -> Result<()> {
        self.curve.validate(&self.patch)?;
        if !(self.relaxation > 0.0) || self.relaxation > 1.0 {
            return Err(Error::Config(format!(
                "fan patch \"{}\": fanRelaxation = {} is outside (0, 1] - it is \
                 SPEC-LIT (S52.14)'s under-relaxation of the OPERATING POINT, \
                 Q* <- Q* + alpha (Q - Q*), and a value outside that range is not a \
                 relaxation",
                self.patch, self.relaxation
            )));
        }
        if !self.ambient.is_finite() {
            return Err(Error::Config(format!(
                "fan patch \"{}\": ambientPressure = {} is not a number (it is \
                 KINEMATIC, m^2/s^2 = Pa/rho_ref - SPEC-LIT S1)",
                self.patch, self.ambient
            )));
        }
        Ok(())
    }
}

// ==========================================================================
//  2. The porous jump - SPEC-LIT §53
// ==========================================================================

/// A resistive sheet's two coefficients, in the form (S53.2) uses.
///
/// Both are non-negative by construction, which is what makes `R >= 0` and
/// therefore what makes §53.2 unconditionally stable with no sign branch -
/// the same argument §18 makes for the volumetric drag.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PorousJumpCoeffs {
    /// `nu t_m/alpha`, m/s - the Darcy (viscous) half.
    pub r_visc: Scalar,
    /// `C2 t_m/2 = K/2`, dimensionless - the Forchheimer (inertial) half.
    pub r_inert: Scalar,
}

impl Default for PorousJumpCoeffs {
    /// No resistance. Bitwise inert (§53.2): `x/(1 + 0*D) == x`.
    fn default() -> Self {
        Self { r_visc: 0.0, r_inert: 0.0 }
    }
}

impl PorousJumpCoeffs {
    /// The Fluent-style `(alpha, C2, t_m)` parameterisation of (S53.1),
    /// with the kinematic viscosity the momentum equation carries.
    pub fn from_darcy_forchheimer(
        alpha: Scalar,
        c2: Scalar,
        thickness: Scalar,
        nu: Scalar,
    ) -> Result<Self> {
        if !(alpha > 0.0) {
            return Err(Error::Config(format!(
                "porousJump: alpha (permeability, m^2) = {alpha} must be positive; \
                 for a purely inertial sheet write the loss coefficient K instead \
                 (SPEC-LIT S53.1)"
            )));
        }
        if !(c2 >= 0.0) || !(thickness >= 0.0) || !(nu >= 0.0) {
            return Err(Error::Config(format!(
                "porousJump: C2 = {c2}, thickness = {thickness} and nu = {nu} must \
                 all be non-negative - SPEC-LIT (S53.2)'s R >= 0 is what makes the \
                 jump unconditionally stable"
            )));
        }
        Ok(Self { r_visc: nu * thickness / alpha, r_inert: 0.5 * c2 * thickness })
    }

    /// The tile-datasheet parameterisation: one loss coefficient on the
    /// **approach** velocity, `dp = K (1/2) rho u_n |u_n|` (S53.3).
    pub fn from_loss_coefficient(k: Scalar) -> Result<Self> {
        if !(k >= 0.0) {
            return Err(Error::Config(format!(
                "porousJump: the loss coefficient K = {k} must be non-negative \
                 (SPEC-LIT S53.3: dp = K (1/2) rho u_n |u_n|)"
            )));
        }
        Ok(Self { r_visc: 0.0, r_inert: 0.5 * k })
    }

    /// The loss coefficient of a thin perforated plate of open-area ratio
    /// `sigma`, (S53.6).
    ///
    /// **The design note this was written from quotes two values and (S53.6)
    /// contradicts one of them.** It says `K ~= 30` at `sigma = 0.25`, which
    /// is reproduced (`30.68`), and `K ~= 4` at `sigma = 0.56`, where
    /// (S53.6) gives `2.94` - `4.37` is its value at `sigma = 0.50`. The
    /// formula is gated on its two limits instead of on either quoted
    /// number, and the derived `K` is printed so the conversion is never
    /// invisible.
    pub fn loss_coefficient_of_open_area(sigma: Scalar) -> Result<Scalar> {
        if !(sigma > 0.0) || sigma > 1.0 {
            return Err(Error::Config(format!(
                "porousJump: openAreaRatio = {sigma} is outside (0, 1]. It is the \
                 fraction of the tile that is HOLE; a 25 %-open tile is 0.25 \
                 (SPEC-LIT S53.4). Available: write the loss coefficient K directly"
            )));
        }
        let one_m = 1.0 - sigma;
        let a = 0.707 * one_m.powf(0.375) + one_m;
        Ok(a * a / (sigma * sigma))
    }

    /// The same, as coefficients.
    pub fn from_open_area_ratio(sigma: Scalar) -> Result<Self> {
        Self::from_loss_coefficient(Self::loss_coefficient_of_open_area(sigma)?)
    }

    /// (S53.2)'s `R` at one face, on the host - the mirror of `fanJumpR`.
    #[inline]
    pub fn resistance(&self, phi: Scalar, area: Scalar) -> Scalar {
        (self.r_visc + self.r_inert * phi.abs() / area) / area
    }
}

/// A porous jump on **internal** faces (§53.2) or on a **boundary** patch
/// against a prescribed plenum pressure (§53.3).
#[derive(Debug, Clone, PartialEq)]
pub enum PorousJump {
    /// Every internal face in the list. Selected on the host at setup and
    /// sorted, so the gather order is fixed.
    Internal { faces: Vec<Label>, coeffs: PorousJumpCoeffs },
    /// A named boundary patch, with the plenum behind it at a prescribed
    /// **kinematic** pressure.
    Boundary { patch: String, coeffs: PorousJumpCoeffs, plenum: Scalar },
}

/// SPEC-LIT §53.5: splitting an existing internal face into a baffle pair is
/// a topology mutation and is refused by name.
///
/// Call this where a case asks for one. It never succeeds; the `Result` shape
/// is so the caller can `?` it into the same error channel every other §13.4
/// refusal uses, and so `-permissive` sees a documented substitution rather
/// than a panic.
pub fn refuse_baffle_insertion(what: &str) -> Result<PorousJump> {
    unsupported_note(
        what,
        "baffle",
        &["porousJump on an internal face", "porousJump on a boundary patch"],
        "splitting an existing internal face into a coincident pair of boundary \
         faces is a TOPOLOGY MUTATION, which this solver does not do (SPEC-LIT \
         S53.5). The two routes that exist are: emit the cyclic pair at \
         mesh-generation time and let io/polymesh.rs pair it, or model the plenum \
         as a separate region and use the boundary form. A jump on an ordinary \
         internal face needs no baffle at all - what it cannot do is make the \
         SCALARS jump, which is the only thing a baffle adds",
        "porousJump on the internal face, which gets the pressure-flux relation \
         right and leaves T and Y_v continuous",
        PorousJump::Internal { faces: Vec::new(), coeffs: PorousJumpCoeffs::default() },
    )
}

/// SPEC-LIT §52.9: the Woodbury / capacitance-matrix path that would keep the
/// cuFFT direct Poisson backend alive under a fan patch. Refused by name.
pub fn refuse_capacitance_fft(what: &str) -> Result<()> {
    unsupported_note(
        what,
        "capacitance",
        &["pbicgstab", "pcg", "amgx"],
        "SPEC-LIT (S52.8) shows the fan patch is a SYMMETRIC RANK-1 downdate of an \
         operator that would otherwise be separable, so the Sherman-Morrison / \
         capacitance-matrix correction of Buzbee et al. (1971) would keep the FFT \
         path at the price of one extra FFT solve per outer iteration. It is NOT \
         implemented (S52.9): it needs the fan patch to be a whole side of the box, \
         its denominator 1 - kappa d^T L^-1 d approaches zero exactly where the fan \
         is stiffest, and a corrected direct solve is a new claim that the \
         backend-agreement check would have to be re-gated against",
        "pbicgstab - the iterative backend the selector falls back to, which is \
         printed",
        (),
    )
}

// ==========================================================================
//  3. Kernels
// ==========================================================================

struct FanKernels {
    gather_patch: CudaFunction,
    gather_flux_weighted: CudaFunction,
    gather_inflow: CudaFunction,
    operating_point: CudaFunction,
    stamp: CudaFunction,
    store3: CudaFunction,
    jump_internal: CudaFunction,
    jump_boundary: CudaFunction,
    rci: CudaFunction,
    zone_heat: CudaFunction,
}

impl FanKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::FAN)?;
        Ok(Self {
            gather_patch: k.func("fanGatherPatch")?,
            gather_flux_weighted: k.func("fanGatherFluxWeighted")?,
            gather_inflow: k.func("fanGatherInflow")?,
            operating_point: k.func("fanOperatingPoint")?,
            stamp: k.func("fanStampTriple")?,
            store3: k.func("fanStoreScalar3")?,
            jump_internal: k.func("fanJumpInternal")?,
            jump_boundary: k.func("fanJumpBoundary")?,
            rci: k.func("dcRciExcess")?,
            zone_heat: k.func("dcZoneHeat")?,
        })
    }
}

// ==========================================================================
//  4. The device-resident set of flow devices
// ==========================================================================

/// What one fan patch reported on the last update - §52.6's printed
/// operating point.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FanState {
    /// `Q`, the raw patch flow this iteration, m^3/s (outward positive).
    pub q: Scalar,
    /// `Q*`, the under-relaxed operating point the curve was evaluated at.
    pub q_star: Scalar,
    /// `dp_fan(sigma Q*)`, Pa.
    pub dp: Scalar,
    /// `S = -dF/dQ_dev`, **kinematic** (m^2/s^2 per m^3/s).
    pub s: Scalar,
    /// The value fraction the triple was stamped with, `1/(1 + S SIGMA_D)`.
    pub fr: Scalar,
    /// The reference value the triple was stamped with, `c + S Phi`.
    pub ref_value: Scalar,
}

/// Every fan patch and porous jump of one mesh, on the device.
///
/// Attached to a [`crate::simple::Simple`] through
/// [`crate::simple::Simple::set_flow_devices`]; a solver that never attaches
/// one runs a code path that is bit-for-bit what it was before this module
/// existed, which is how §52's "the default is unmoved" is proved from the
/// diff rather than argued.
pub struct FlowDevices {
    k: FanKernels,
    solk: SolverKernels,

    fans: Vec<FanPatch>,
    /// `(start, size)` into the flattened boundary arrays, per fan.
    spans: Vec<(usize, usize)>,

    // ---- per-fan device inputs -------------------------------------------
    kind: DevBuf<Label>,
    dp_max: DevBuf<Scalar>,
    q_max: DevBuf<Scalar>,
    p_amb: DevBuf<Scalar>,
    sigma: DevBuf<Scalar>,
    rho_ratio: DevBuf<Scalar>,
    speed_ratio: DevBuf<Scalar>,
    alpha: DevBuf<Scalar>,
    /// Per fan, the `Q*` the first update linearises about when no flux
    /// exists yet - `sigma` times free delivery. See
    /// [`FanCurve::free_delivery`].
    q_seed: DevBuf<Scalar>,
    tq: DevBuf<Scalar>,
    tdp: DevBuf<Scalar>,
    tm: DevBuf<Scalar>,
    n_points: DevBuf<Label>,

    // ---- reduction scratch -----------------------------------------------
    gq: DevBuf<Scalar>,
    gph: DevBuf<Scalar>,
    gsd: DevBuf<Scalar>,
    partials: DevBuf<Scalar>,
    sum_q: DevBuf<Scalar>,
    sum_ph: DevBuf<Scalar>,
    sum_sd: DevBuf<Scalar>,
    red_q: DevBuf<Scalar>,
    red_ph: DevBuf<Scalar>,
    red_sd: DevBuf<Scalar>,
    /// `[6*n_fans]`, the operating-point block `fanOperatingPoint` writes.
    out: DevBuf<Scalar>,

    // ---- jumps ------------------------------------------------------------
    int_faces: DevBuf<Label>,
    int_rvisc: DevBuf<Scalar>,
    int_rinert: DevBuf<Scalar>,
    n_int: usize,
    bnd_faces: DevBuf<Label>,
    bnd_rvisc: DevBuf<Scalar>,
    bnd_rinert: DevBuf<Scalar>,
    bnd_plenum: DevBuf<Scalar>,
    n_bnd: usize,

    rho_ref: Scalar,
    first: bool,
    /// Set when any jump is present, for §53.6's printed caveat.
    jump_patches: Vec<String>,
}

impl FlowDevices {
    /// Build from a validated list of fan patches and porous jumps.
    ///
    /// `rho_ref` is the reference density the kinematic pressure is divided
    /// by (§1). Every fan patch must name a patch of `hm`, and every jump
    /// face must be an internal face - both are errors naming the offender,
    /// not silent drops.
    pub fn new(
        gpu: &Gpu,
        hm: &HostMesh,
        fans: Vec<FanPatch>,
        jumps: &[PorousJump],
        rho_ref: Scalar,
    ) -> Result<Self> {
        if !(rho_ref > 0.0) {
            return Err(Error::Config(format!(
                "FlowDevices: rho_ref = {rho_ref} must be positive - it converts the \
                 manufacturer's curve to the kinematic pressure this solver carries, \
                 F = dp/rho_ref (SPEC-LIT S52.1)"
            )));
        }
        for f in &fans {
            f.validate()?;
        }

        // ---- resolve the fan patches --------------------------------------
        let mut spans = Vec::with_capacity(fans.len());
        for f in &fans {
            let Some(p) = hm.patches.iter().find(|p| p.name == f.patch) else {
                return Err(Error::Config(format!(
                    "fan patch \"{}\" is not a patch of this mesh; the mesh has: {}",
                    f.patch,
                    hm.patches.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(", ")
                )));
            };
            if p.size == 0 {
                return Err(Error::Config(format!(
                    "fan patch \"{}\" has no faces - a fan on an empty patch would \
                     move no air and report a flow rate of zero, which is a setting \
                     the case can say and the solver silently ignores (SPEC-LIT \
                     S13.4.1)",
                    f.patch
                )));
            }
            spans.push((p.start, p.size));
        }
        // Two fans on one patch would each stamp the other's triple, and the
        // last one launched would win - silently.
        for i in 0..fans.len() {
            for j in i + 1..fans.len() {
                if fans[i].patch == fans[j].patch {
                    return Err(Error::Config(format!(
                        "two fan patches both name \"{}\"; each would stamp the \
                         other's Robin triple and whichever launched last would win, \
                         silently",
                        fans[i].patch
                    )));
                }
            }
        }

        let nf = fans.len();
        let n_max = spans.iter().map(|s| s.1).max().unwrap_or(0).max(1);

        let mut kind = Vec::with_capacity(nf);
        let mut dp_max = Vec::with_capacity(nf);
        let mut q_max = Vec::with_capacity(nf);
        let mut p_amb = Vec::with_capacity(nf);
        let mut sigma = Vec::with_capacity(nf);
        let mut rho_ratio = Vec::with_capacity(nf);
        let mut speed_ratio = Vec::with_capacity(nf);
        let mut alpha = Vec::with_capacity(nf);
        let mut q_seed = Vec::with_capacity(nf);
        let mut n_points = Vec::with_capacity(nf);
        let mut tq = vec![0.0 as Scalar; nf.max(1) * MAX_CURVE_POINTS];
        let mut tdp = tq.clone();
        let mut tm = tq.clone();

        for (j, f) in fans.iter().enumerate() {
            kind.push(f.curve.kind as Label);
            dp_max.push(f.curve.dp_max);
            q_max.push(if f.curve.q_max > 0.0 { f.curve.q_max } else { 1.0 });
            p_amb.push(f.ambient);
            sigma.push(f.direction.sigma());
            rho_ratio.push(f.curve.rho_ratio());
            speed_ratio.push(f.curve.speed_ratio());
            alpha.push(f.relaxation);
            q_seed.push(f.direction.sigma() * f.curve.free_delivery());
            n_points.push(f.curve.points.len().max(1) as Label);

            let slopes = f.curve.hermite_slopes();
            for (i, (q, dp)) in f.curve.points.iter().enumerate() {
                tq[j * MAX_CURVE_POINTS + i] = *q;
                tdp[j * MAX_CURVE_POINTS + i] = *dp;
                tm[j * MAX_CURVE_POINTS + i] = slopes[i];
            }
        }

        // ---- the jumps ----------------------------------------------------
        let mut int_faces: Vec<Label> = Vec::new();
        let mut int_rvisc: Vec<Scalar> = Vec::new();
        let mut int_rinert: Vec<Scalar> = Vec::new();
        let mut bnd_faces: Vec<Label> = Vec::new();
        let mut bnd_rvisc: Vec<Scalar> = Vec::new();
        let mut bnd_rinert: Vec<Scalar> = Vec::new();
        let mut bnd_plenum: Vec<Scalar> = Vec::new();
        let mut jump_patches: Vec<String> = Vec::new();

        for jmp in jumps {
            match jmp {
                PorousJump::Internal { faces, coeffs } => {
                    for &f in faces {
                        if f < 0 || (f as usize) >= hm.n_internal_faces {
                            return Err(Error::Config(format!(
                                "porousJump: face {f} is not an internal face of this \
                                 mesh (it has {} of them). SPEC-LIT S53.5: a jump on \
                                 a BOUNDARY face is the boundary form and takes a \
                                 plenum pressure",
                                hm.n_internal_faces
                            )));
                        }
                        int_faces.push(f);
                        int_rvisc.push(coeffs.r_visc);
                        int_rinert.push(coeffs.r_inert);
                    }
                    if !faces.is_empty() {
                        jump_patches.push(format!("{} internal faces", faces.len()));
                    }
                }
                PorousJump::Boundary { patch, coeffs, plenum } => {
                    let Some(p) = hm.patches.iter().find(|p| &p.name == patch) else {
                        return Err(Error::Config(format!(
                            "porousJump patch \"{patch}\" is not a patch of this mesh; \
                             the mesh has: {}",
                            hm.patches
                                .iter()
                                .map(|p| p.name.as_str())
                                .collect::<Vec<_>>()
                                .join(", ")
                        )));
                    };
                    if p.size == 0 {
                        return Err(Error::Config(format!(
                            "porousJump patch \"{patch}\" has no faces"
                        )));
                    }
                    for bf in p.start..p.start + p.size {
                        bnd_faces.push(bf as Label);
                        bnd_rvisc.push(coeffs.r_visc);
                        bnd_rinert.push(coeffs.r_inert);
                        bnd_plenum.push(*plenum);
                    }
                    jump_patches.push(patch.clone());
                }
            }
        }

        // A fixed gather order is what makes the reduction reproducible; the
        // sort is the same argument S2 makes for `build_cell_face_maps`.
        let mut order: Vec<usize> = (0..int_faces.len()).collect();
        order.sort_by_key(|&i| int_faces[i]);
        let int_faces: Vec<Label> = order.iter().map(|&i| int_faces[i]).collect();
        let int_rvisc: Vec<Scalar> = order.iter().map(|&i| int_rvisc[i]).collect();
        let int_rinert: Vec<Scalar> = order.iter().map(|&i| int_rinert[i]).collect();
        for w in int_faces.windows(2) {
            if w[0] == w[1] {
                return Err(Error::Config(format!(
                    "porousJump: internal face {} is named by two jumps. The two \
                     resistances would not add - whichever kernel ran last would \
                     divide the already-divided coefficient again",
                    w[0]
                )));
            }
        }

        let n_int = int_faces.len();
        let n_bnd = bnd_faces.len();
        let nparts = solver::reduce_partitions(n_max);

        Ok(Self {
            k: FanKernels::new(gpu)?,
            solk: SolverKernels::new(gpu)?,
            fans,
            spans,
            kind: gpu.upload(&pad(kind, 0))?,
            dp_max: gpu.upload(&pad(dp_max, 0.0))?,
            q_max: gpu.upload(&pad(q_max, 1.0))?,
            p_amb: gpu.upload(&pad(p_amb, 0.0))?,
            sigma: gpu.upload(&pad(sigma, 1.0))?,
            rho_ratio: gpu.upload(&pad(rho_ratio, 1.0))?,
            speed_ratio: gpu.upload(&pad(speed_ratio, 1.0))?,
            alpha: gpu.upload(&pad(alpha, 1.0))?,
            q_seed: gpu.upload(&pad(q_seed, 0.0))?,
            tq: gpu.upload(&tq)?,
            tdp: gpu.upload(&tdp)?,
            tm: gpu.upload(&tm)?,
            n_points: gpu.upload(&pad(n_points, 1))?,
            gq: gpu.zeros(n_max)?,
            gph: gpu.zeros(n_max)?,
            gsd: gpu.zeros(n_max)?,
            partials: gpu.zeros(nparts.max(1))?,
            sum_q: gpu.zeros(nf.max(1))?,
            sum_ph: gpu.zeros(nf.max(1))?,
            sum_sd: gpu.zeros(nf.max(1))?,
            red_q: gpu.zeros(1)?,
            red_ph: gpu.zeros(1)?,
            red_sd: gpu.zeros(1)?,
            out: gpu.zeros(6 * nf.max(1))?,
            int_faces: gpu.upload(&pad(int_faces, 0))?,
            int_rvisc: gpu.upload(&pad(int_rvisc, 0.0))?,
            int_rinert: gpu.upload(&pad(int_rinert, 0.0))?,
            n_int,
            bnd_faces: gpu.upload(&pad(bnd_faces, 0))?,
            bnd_rvisc: gpu.upload(&pad(bnd_rvisc, 0.0))?,
            bnd_rinert: gpu.upload(&pad(bnd_rinert, 0.0))?,
            bnd_plenum: gpu.upload(&pad(bnd_plenum, 0.0))?,
            n_bnd,
            rho_ref,
            first: true,
            jump_patches,
        })
    }

    pub fn n_fans(&self) -> usize {
        self.fans.len()
    }

    pub fn n_internal_jump_faces(&self) -> usize {
        self.n_int
    }

    pub fn n_boundary_jump_faces(&self) -> usize {
        self.n_bnd
    }

    pub fn fans(&self) -> &[FanPatch] {
        &self.fans
    }

    /// True where §53.6's near-tile velocity caveat applies.
    pub fn has_jump(&self) -> bool {
        self.n_int > 0 || self.n_bnd > 0
    }

    /// SPEC-LIT §53.6: what a pressure-jump tile gets wrong, in the words a
    /// report has to carry. Returned rather than printed so a driver can put
    /// it where its own output goes - and so a test can read it.
    pub fn jump_caveat(&self) -> Option<String> {
        if !self.has_jump() {
            return None;
        }
        Some(format!(
            "porous jump on {}: a pressure-jump model gets the tile FLOW RATE right \
             and the near-tile VELOCITY FIELD wrong. Abdelmaksoud et al. (2010, \
             ITherm) measured the vena contracta and the off-tile jet that this model \
             cannot produce; Arghode & Joshi (2013, IEEE T-CPMT) show a \
             momentum-source or prescribed-velocity model is needed when the JET \
             matters. Do not read a cold-aisle velocity field off these faces \
             (SPEC-LIT S53.6).",
            self.jump_patches.join(", ")
        ))
    }

    /// The operating point of every fan, downloaded.
    ///
    /// A host readback, so a CUDA-Graph-captured run must not call it inside
    /// the captured region - the same rule `report_continuity` already obeys
    /// (§52.7). Everything the *solve* needs stays on the device.
    pub fn states(&self, gpu: &Gpu) -> Result<Vec<FanState>> {
        if self.fans.is_empty() {
            return Ok(Vec::new());
        }
        let raw = gpu.download(&self.out)?;
        Ok((0..self.fans.len())
            .map(|j| FanState {
                fr: raw[6 * j],
                ref_value: raw[6 * j + 1],
                q_star: raw[6 * j + 2],
                s: raw[6 * j + 3],
                dp: raw[6 * j + 4],
                q: raw[6 * j + 5],
            })
            .collect())
    }

    /// (S55.5): the shaft power of every fan at its converged operating
    /// point, W, and their sum.
    pub fn shaft_power(&self, gpu: &Gpu) -> Result<(Vec<Scalar>, Scalar)> {
        let st = self.states(gpu)?;
        let per: Vec<Scalar> = self
            .fans
            .iter()
            .zip(&st)
            .map(|(f, s)| f.curve.shaft_power(f.direction.sigma() * s.q_star))
            .collect();
        let total = per.iter().sum();
        Ok((per, total))
    }

    /// One outer iteration's device update: the three reductions, the
    /// operating point, the triple, and the jump coefficient division.
    ///
    /// Call **after** `rhie_chow` (which writes `rAU_f`, `rAU_f|Sf|` and
    /// `phi_HbyA`) and **before** the pressure assembly.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        phi: &crate::field::GpuSurfaceScalarField,
        phi_hbya: &mut crate::field::GpuSurfaceScalarField,
        rauf: &mut crate::field::GpuSurfaceScalarField,
        rauf_mag_sf: &mut crate::field::GpuSurfaceScalarField,
        p: &mut crate::field::GpuScalarField,
    ) -> Result<()> {
        // ---- S53.2: the jumps go FIRST -----------------------------------
        //
        // The fan's SIGMA_D must be the conductance the matrix will actually
        // be assembled with. If a fan patch also carried a jump and the two
        // ran the other way round, the triple would be built from a
        // coefficient the assembly then changed underneath it.
        self.apply_jumps(gpu, m, phi, phi_hbya, rauf, rauf_mag_sf, p)?;

        if self.fans.is_empty() {
            return Ok(());
        }

        // ---- S52.7: three gathers, three device_sums, per patch ----------
        for (j, &(start, size)) in self.spans.iter().enumerate() {
            let (s, n) = (start as Label, size as Label);
            let f = self.k.gather_patch.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.gq)
                    .arg(&mut self.gph)
                    .arg(&mut self.gsd)
                    .arg(&phi.bf)
                    .arg(&phi_hbya.bf)
                    .arg(&rauf_mag_sf.bf)
                    .arg(&m.b_delta_coeffs)
                    .arg(&s)
                    .arg(&n)
                    .launch(cfg_for(size))?;
            }

            // `device_sum` reduces the FIRST `size` entries, which is exactly
            // what the gather wrote - so no offset entry point is needed and
            // NO NEW REDUCTION IS WRITTEN (S52.7). It answers into element 0
            // of a scratch scalar; `fanStoreScalar3` then moves the three
            // numbers into this patch's slot. That copy is one thread and no
            // arithmetic - it is not a reduction stage.
            solver::device_sum(gpu, &self.solk, &mut self.red_q, &self.gq, &mut self.partials, size)?;
            solver::device_sum(gpu, &self.solk, &mut self.red_ph, &self.gph, &mut self.partials, size)?;
            solver::device_sum(gpu, &self.solk, &mut self.red_sd, &self.gsd, &mut self.partials, size)?;

            let slot = j as Label;
            let f = self.k.store3.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.sum_q)
                    .arg(&mut self.sum_ph)
                    .arg(&mut self.sum_sd)
                    .arg(&self.red_q)
                    .arg(&self.red_ph)
                    .arg(&self.red_sd)
                    .arg(&slot)
                    .launch(cfg_for(1))?;
            }
        }

        // ---- S52.3: the operating point and the triple --------------------
        let nf = self.fans.len();
        let nfl = nf as Label;
        let first = Label::from(self.first);
        let rho_ref = self.rho_ref;
        {
            let f = self.k.operating_point.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.out)
                    .arg(&self.sum_q)
                    .arg(&self.sum_ph)
                    .arg(&self.sum_sd)
                    .arg(&self.kind)
                    .arg(&self.dp_max)
                    .arg(&self.q_max)
                    .arg(&self.p_amb)
                    .arg(&self.sigma)
                    .arg(&self.rho_ratio)
                    .arg(&self.speed_ratio)
                    .arg(&self.alpha)
                    .arg(&self.q_seed)
                    .arg(&self.tq)
                    .arg(&self.tdp)
                    .arg(&self.tm)
                    .arg(&self.n_points)
                    .arg(&rho_ref)
                    .arg(&first)
                    .arg(&nfl)
                    .launch(cfg_for(nf))?;
            }
        }
        self.first = false;

        for (j, &(start, size)) in self.spans.iter().enumerate() {
            let (pj, s, n) = (j as Label, start as Label, size as Label);
            let f = self.k.stamp.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut p.fr)
                    .arg(&mut p.ref_value)
                    .arg(&mut p.ref_grad)
                    .arg(&self.out)
                    .arg(&pj)
                    .arg(&s)
                    .arg(&n)
                    .launch(cfg_for(size))?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_jumps(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        phi: &crate::field::GpuSurfaceScalarField,
        phi_hbya: &mut crate::field::GpuSurfaceScalarField,
        rauf: &mut crate::field::GpuSurfaceScalarField,
        rauf_mag_sf: &mut crate::field::GpuSurfaceScalarField,
        p: &mut crate::field::GpuScalarField,
    ) -> Result<()> {
        if self.n_int > 0 {
            let n = self.n_int as Label;
            let f = self.k.jump_internal.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut rauf_mag_sf.f)
                    .arg(&mut rauf.f)
                    .arg(&mut phi_hbya.f)
                    .arg(&phi.f)
                    .arg(&m.mag_sf)
                    .arg(&m.delta_coeffs)
                    .arg(&self.int_faces)
                    .arg(&self.int_rvisc)
                    .arg(&self.int_rinert)
                    .arg(&n)
                    .launch(cfg_for(self.n_int))?;
            }
        }
        if self.n_bnd > 0 {
            let n = self.n_bnd as Label;
            let f = self.k.jump_boundary.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut rauf_mag_sf.bf)
                    .arg(&mut rauf.bf)
                    .arg(&mut phi_hbya.bf)
                    .arg(&mut p.fr)
                    .arg(&mut p.ref_value)
                    .arg(&mut p.ref_grad)
                    .arg(&phi.bf)
                    .arg(&m.b_mag_sf)
                    .arg(&m.b_delta_coeffs)
                    .arg(&self.bnd_faces)
                    .arg(&self.bnd_rvisc)
                    .arg(&self.bnd_rinert)
                    .arg(&self.bnd_plenum)
                    .arg(&n)
                    .launch(cfg_for(self.n_bnd))?;
            }
        }
        Ok(())
    }

    /// The kernels, for the metric module - `crate::dcmetrics` reduces
    /// through the same gathers rather than writing its own.
    pub(crate) fn metric_kernels(&self) -> (&CudaFunction, &CudaFunction, &CudaFunction, &CudaFunction) {
        (&self.k.gather_flux_weighted, &self.k.gather_inflow, &self.k.rci, &self.k.zone_heat)
    }
}

/// Grow a per-fan vector to at least one element, so `gpu.upload` is never
/// handed an empty slice (a zero-length allocation is not a valid device
/// buffer).
fn pad<T: Clone>(mut v: Vec<T>, fill: T) -> Vec<T> {
    if v.is_empty() {
        v.push(fill);
    }
    v
}

// ==========================================================================
//  5. The host mirror of §52.3 - what the tests measure against
// ==========================================================================

/// The lumped Robin triple of (S52.10)/(S52.11), on the host.
///
/// `q_star` is the operating point the curve is linearised about; `phi_sum`
/// is `Phi`; `sigma_d` is `SIGMA_D`. Returns `(fr, ref_value, s_kin)`.
pub fn lumped_triple(
    curve: &FanCurve,
    direction: FanDirection,
    ambient: Scalar,
    rho_ref: Scalar,
    q_star: Scalar,
    phi_sum: Scalar,
    sigma_d: Scalar,
) -> (Scalar, Scalar, Scalar) {
    let sg = direction.sigma();
    let (dp, s) = curve.at(sg * q_star);
    let f = dp / rho_ref;
    let s_kin = (s / rho_ref).max(0.0);
    let c = ambient - sg * f - s_kin * q_star;
    let beta = s_kin * sigma_d;
    (1.0 / (1.0 + beta), c + s_kin * phi_sum, s_kin)
}

/// The **exact** rank-1 operator of (S52.8), built densely.
///
/// `A = diag(D) - kappa d d^T`, `kappa = S/(1 + S SIGMA_D)`. Only ever built
/// on the host and only by the tests: §52.7 explains why it is not put
/// inside `amul`.
///
/// **The association is load-bearing.** `kappa*(d_i d_j)` is bitwise
/// symmetric, because IEEE-754 multiplication is commutative: `d_i*d_j` and
/// `d_j*d_i` are the same number, to the bit. `(kappa*d_i)*d_j` is **not** -
/// it rounds twice, in a different order for each half, and the two entries
/// come out one ulp apart. Measured on `D = (0.3, 1.7, 0.55, 2.2, 0.9)` at
/// `S = 0.7`: `-0.023309788092835522` against `-0.02330978809283552`.
/// SPEC-LIT §52.2's "identical numbers, not merely equal ones" is a claim
/// about the *mathematics*, and it survives into f64 only under this
/// association. Nothing in the shipped solver builds this operator - the
/// lumped triple of §52.3 gets its symmetry from `fvm_laplacian` writing
/// `upper[f] == lower[f]`, a different and unconditional mechanism - but a
/// reference implementation that quietly lost it would make the gate check
/// the wrong thing.
pub fn exact_rank1(d: &[Scalar], s: Scalar) -> Vec<Vec<Scalar>> {
    let sd: Scalar = d.iter().sum();
    let kappa = s / (1.0 + s * sd);
    let n = d.len();
    let mut a = vec![vec![0.0 as Scalar; n]; n];
    for (i, row) in a.iter_mut().enumerate() {
        for (j, e) in row.iter_mut().enumerate() {
            *e = if i == j { d[i] } else { 0.0 } - kappa * (d[i] * d[j]);
        }
    }
    a
}

/// (S52.15): the closed-form intersection of a quadratic fan and a quadratic
/// system, `dp_sys = K Q^2`.
///
/// Written out rather than stored as a constant, so §52.12 Gate 52-A fails on
/// a transcription error instead of agreeing with one.
pub fn quadratic_operating_point(dp_max: Scalar, q_max: Scalar, k_sys: Scalar) -> Scalar {
    q_max / (1.0 + k_sys * q_max * q_max / dp_max).sqrt()
}
