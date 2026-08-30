// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Psychrometrics and moist-air buoyancy - SPEC-LIT §54.
//!
//! Written from:
//!   ASHRAE, *Handbook—Fundamentals* Ch. 1 "Psychrometrics" (2021) - the
//!     equation numbering of (S54.2)-(S54.5) is this chapter's, and its
//!     Table 2 is the external comparison of §54.8 Gate 54-B
//!   R. W. Hyland, A. Wexler, *ASHRAE Transactions* 89(2A) (1983) 500-519 -
//!     the thirteen `C1`-`C13` coefficients of (S54.3)
//!   D. P. Gatley, S. Herrmann, H.-J. Kretzschmar, *HVAC&R Research* 14(5)
//!     (2008) 655-662, DOI 10.1080/10789669.2008.10391032 - `M_a = 28.966
//!     g/mol` and hence `eps = 0.621945`
//!   S. Herrmann, H.-J. Kretzschmar, D. P. Gatley, *HVAC&R Research* 15(5)
//!     (2009) 961-986, DOI 10.1080/10789669.2009.10390874 (RP-1485) - the
//!     real-gas enhancement factor that makes these ideal relations 0.44 %
//!     low in `W_s` at 25 C. NAMED AND NOT IMPLEMENTED (§54.3)
//!   EnergyPlus `src/EnergyPlus/Psychrometrics.hh` - BSD-3-clause style
//!     (UIUC / UC Regents / DOE; `LICENSE.txt` fetched and read). Taken from
//!     it: the NAMING CONVENTION HVAC engineers already read
//!     (`PsyWFnTdbRhPb` -> [`w_from_t_rh_p`], and so on), and the NEGATIVE
//!     lesson that its `PsyPsatFnTemp` cache and its 1651-entry `Tsat(p)`
//!     spline exist because those functions are hot on a CPU. On a GPU the
//!     polynomial is cheaper than the lookup it would replace, so this
//!     module COMPUTES AND DOES NOT CACHE. Its wet-bulb function is an
//!     iterative solve, which is the confirmation §54.5 needed.
//!   CoolProp - MIT (`LICENSE` fetched and read). `HumidAirProp` is the
//!     RP-1485 implementation to check against if the 0.44 % bias ever needs
//!     removing; nothing was ported from it - it is a host-side
//!     equation-of-state library, not a kernel.
//!   FDS 6 `Source/func.f90` - US public domain. Its
//!     `WATER_VAPOR_MASS_FRACTION`/`RELATIVE_HUMIDITY` use a
//!     Clausius-Clapeyron integral rather than the ASHRAE polynomial;
//!     DELIBERATELY NOT USED, because a data-centre customer checks the
//!     number against a psychrometric chart.
//!   ofgpu `SPEC-LIT.md` §19 (species transport, which carries `Y_v`
//!     verbatim), §9 (buoyancy, whose kernel this module leaves untouched),
//!     §13.4
//! No GPL-licensed source was consulted.
//!
//! # The two things this module is careful about
//!
//! **1. The ideal-gas bias is printed, not hidden.** (S54.2)-(S54.5) are the
//! ideal relations. `W_s(25 C)` comes out `0.0200811`; the ASHRAE table says
//! `0.020169`, because the table carries the real-gas enhancement factor
//! `f_e ~= 1.0044`. That is a **0.44 % low bias**, it is inside any
//! data-centre measurement uncertainty, and [`enhancement_bias`] reports it
//! so a report can print it. Quietly widening a tolerance around a known bias
//! is the failure mode the two-sided gate of §54.8 exists to prevent.
//!
//! **2. The buoyancy default is unmoved BY CONSTRUCTION.**
//! `momentum::Momentum::update_buoyancy` already takes the temperature field
//! as an argument, so the virtual temperature is computed into a *separate*
//! field and handed to that same, unmodified function. `src/momentum.rs` is
//! not touched at all - the diff is the proof - and at `Y_v == 0` the kernel
//! computes `T*(1.0 + c*0.0) = T*1.0 = T`, bitwise.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::GpuScalarField;
use crate::io::contract::unsupported_note;
use crate::mesh::GpuMesh;
use crate::solver::{self, SolverKernels};
use crate::{Label, Scalar};

#[cfg(test)]
mod tests;

// ==========================================================================
//  1. Constants - SPEC-LIT (S54.3)
// ==========================================================================

/// `M_w/M_a = 18.015268/28.966` - Gatley, Herrmann & Kretzschmar (2008).
pub const EPS: Scalar = 0.621945;

/// `1/eps - 1 = 0.607858`, the coefficient of (S54.7)'s virtual temperature.
pub const VIRTUAL_COEFF: Scalar = 1.0 / EPS - 1.0;

/// Standard atmosphere, Pa.
pub const P_ATM: Scalar = 101325.0;

/// Hyland & Wexler (1983) over **ice**, `T < 273.15 K`.
const C_ICE: [Scalar; 7] = [
    -5.6745359e3,
    6.3925247,
    -9.677843e-3,
    6.2215701e-7,
    2.0747825e-9,
    -9.484024e-13,
    4.1635019,
];

/// Hyland & Wexler (1983) over **liquid water**, `T >= 273.15 K`.
const C_LIQ: [Scalar; 6] =
    [-5.8002206e3, 1.3914993, -4.8640239e-2, 4.1764768e-5, -1.4452093e-8, 6.5459673];

// ==========================================================================
//  2. The host relations - SPEC-LIT §54.2
//
//  Named in EnergyPlus's convention - "property, from these arguments" -
//  because HVAC engineers already read it. `PsyWFnTdbRhPb` is
//  `w_from_t_rh_p`; the mapping is one-to-one and deliberate.
// ==========================================================================

/// Saturation vapour pressure, Pa, from absolute temperature - (S54.3).
///
/// The two branches meet at 273.15 K; the ice branch is used strictly below
/// it, per Hyland & Wexler's own split.
pub fn p_ws(t: Scalar) -> Scalar {
    let l = t.ln();
    if t < 273.15 {
        let c = &C_ICE;
        (c[0] / t
            + c[1]
            + c[2] * t
            + c[3] * t * t
            + c[4] * t * t * t
            + c[5] * t * t * t * t
            + c[6] * l)
            .exp()
    } else {
        let c = &C_LIQ;
        (c[0] / t + c[1] + c[2] * t + c[3] * t * t + c[4] * t * t * t + c[5] * l).exp()
    }
}

/// Humidity ratio from water-vapour mass fraction - (S54.2a).
///
/// `W` is kg vapour per kg **dry** air; `Y_v` is kg vapour per kg **moist**
/// air. Confusing the two is the commonest psychrometric error there is,
/// which is why both names appear in every signature here.
#[inline]
pub fn w_from_yv(yv: Scalar) -> Scalar {
    yv / (1.0 - yv)
}

/// The inverse of [`w_from_yv`] - (S54.2a).
#[inline]
pub fn yv_from_w(w: Scalar) -> Scalar {
    w / (1.0 + w)
}

/// Vapour partial pressure, Pa, from the humidity ratio - (S54.2b).
#[inline]
pub fn p_w_from_w_p(w: Scalar, p_atm: Scalar) -> Scalar {
    p_atm * w / (EPS + w)
}

/// Saturation humidity ratio - (S54.4).
pub fn w_s(t: Scalar, p_atm: Scalar) -> Scalar {
    let pws = p_ws(t);
    EPS * pws / (p_atm - pws)
}

/// Relative humidity, `0..1`, from temperature and humidity ratio.
pub fn rh_from_t_w_p(t: Scalar, w: Scalar, p_atm: Scalar) -> Scalar {
    p_w_from_w_p(w, p_atm) / p_ws(t)
}

/// Humidity ratio from temperature and relative humidity - EnergyPlus's
/// `PsyWFnTdbRhPb`, and the conversion an inlet that says `rh` goes through.
pub fn w_from_t_rh_p(t: Scalar, rh: Scalar, p_atm: Scalar) -> Scalar {
    let pw = rh * p_ws(t);
    EPS * pw / (p_atm - pw)
}

/// The water-vapour mass fraction an inlet at `(T, rh)` carries - §54.6.
pub fn yv_from_t_rh_p(t: Scalar, rh: Scalar, p_atm: Scalar) -> Scalar {
    yv_from_w(w_from_t_rh_p(t, rh, p_atm))
}

/// Moist-air specific enthalpy, kJ per kg **dry** air - (S54.4).
pub fn h_from_t_w(t: Scalar, w: Scalar) -> Scalar {
    let c = t - 273.15;
    1.006 * c + w * (2501.0 + 1.86 * c)
}

/// Moist-air specific volume, m^3 per kg **dry** air - (S54.4).
pub fn v_from_t_w_p(t: Scalar, w: Scalar, p_atm: Scalar) -> Scalar {
    0.287042 * t * (1.0 + 1.607858 * w) / (p_atm / 1000.0)
}

/// Dew-point temperature, **degrees Celsius**, from the vapour pressure -
/// (S54.5).
///
/// ASHRAE's two-branch correlation, valid to 93 C. Above that it is
/// extrapolation and this returns the high branch anyway; a data centre never
/// gets there, and a silent second branch would be worse than a documented
/// extrapolation.
pub fn t_d_from_pw(p_w: Scalar) -> Scalar {
    let kpa = p_w / 1000.0;
    if kpa <= 0.0 {
        return Scalar::NEG_INFINITY;
    }
    let a = kpa.ln();
    let hi = 6.54 + 14.526 * a + 0.7389 * a * a + 0.09486 * a * a * a + 0.4569 * kpa.powf(0.1984);
    if hi >= 0.0 {
        hi
    } else {
        6.09 + 12.608 * a + 0.4959 * a * a
    }
}

/// (S54.7): the virtual temperature, K.
///
/// Exact, not a linearisation: `M_a/M_mix = 1 + (1/eps - 1) Y_v` identically,
/// so `T_v/T_v,ref == rho_ref/rho` exactly. At `Y_v == 0` this is `T*1.0`,
/// which is `T` **bitwise**.
#[inline]
pub fn virtual_temperature(t: Scalar, yv: Scalar) -> Scalar {
    t * (1.0 + VIRTUAL_COEFF * yv)
}

/// The mixture molar mass, kg/mol, from the water-vapour mass fraction.
///
/// Only used by the tests, which check (S54.7) against `rho = p M_mix/(R T)`
/// computed from this rather than against the identity it is derived from -
/// a gate that assumes its own algebra is not a gate.
pub fn molar_mass(yv: Scalar) -> Scalar {
    const M_A: Scalar = 28.966e-3;
    const M_W: Scalar = 18.015268e-3;
    1.0 / (yv / M_W + (1.0 - yv) / M_A)
}

/// Wet-bulb temperature, degrees Celsius, by Newton on (S54.6).
///
/// **On the host, in the report, and nowhere else** (§54.5): the iteration
/// count is data-dependent, which is warp divergence and - worse - makes a
/// kernel's trip count depend on its input, so it is not CUDA-Graph
/// capturable. A host function may have a convergence test and a named error
/// where a captured kernel may not, and this one does.
pub fn t_wb(t: Scalar, w: Scalar, p_atm: Scalar) -> Result<Scalar> {
    let tc = t - 273.15;
    let pw = p_w_from_w_p(w, p_atm);
    // §54.5's own initial guess, within about 0.3 K over the data-centre
    // range - accurate enough to start a Newton from, not accurate enough to
    // BE the answer, which is exactly why the fixed-3-step kernel is refused.
    let td = t_d_from_pw(pw);
    let mut x = td + (tc - td) / 3.0;

    let f = |ts: Scalar| -> Scalar {
        let wss = w_s(ts + 273.15, p_atm);
        ((2501.0 - 2.326 * ts) * wss - 1.006 * (tc - ts)) / (2501.0 + 1.86 * tc - 4.186 * ts) - w
    };

    for _ in 0..60 {
        let f0 = f(x);
        if f0.abs() < 1e-13 {
            return Ok(x);
        }
        let h = 1e-6 * (1.0 + x.abs());
        let d = (f(x + h) - f(x - h)) / (2.0 * h);
        if !(d.abs() > 0.0) {
            break;
        }
        let step = f0 / d;
        x -= step;
        if step.abs() < 1e-12 {
            return Ok(x);
        }
        if !x.is_finite() {
            break;
        }
    }
    Err(Error::Config(format!(
        "wet-bulb temperature did not converge for T = {t} K, W = {w} kg/kg, \
         p = {p_atm} Pa (SPEC-LIT S54.5). This is a HOST function precisely so \
         that a non-convergence is an error rather than a wrong number: the \
         fixed-step device form the design note proposed could not have said this"
    )))
}

/// SPEC-LIT §54.3: how far below the ASHRAE table the ideal relations sit, at
/// one state.
///
/// Returns `(W_s ideal, W_s with the enhancement factor, relative bias)`. The
/// enhancement factor is **not** implemented; `f_e` is the published value at
/// the state asked for and this function exists so a report can print the
/// gap rather than pretend it is not there.
pub fn enhancement_bias(t: Scalar, p_atm: Scalar, f_e: Scalar) -> (Scalar, Scalar, Scalar) {
    let pws = p_ws(t);
    let ideal = EPS * pws / (p_atm - pws);
    let real = EPS * f_e * pws / (p_atm - f_e * pws);
    (ideal, real, (real - ideal) / real)
}

/// SPEC-LIT §54.5: field-level condensation is a different model. Refused by
/// name.
pub fn refuse_condensation(what: &str) -> Result<()> {
    unsupported_note(
        what,
        "condensation",
        &["supersaturation reporting", "a coil-surface saturated boundary condition"],
        "field-level condensation (fog) is a saturation-constrained source with its \
         own inner iteration - a different model, not a switch on this one, and not \
         needed for the market SPEC-LIT S54 was written for. What this solver does \
         instead is REPORT supersaturation, with the cell count and the worst \
         excess, rather than silently clipping Y_v to Y_v,sat or silently condensing \
         the difference (S54.5)",
        "supersaturation reporting - Y_v is transported unclipped and the excess is \
         printed",
        (),
    )
}

/// SPEC-LIT §54.5: wet bulb as an in-loop field. Refused by name.
pub fn refuse_wet_bulb_field(what: &str) -> Result<()> {
    unsupported_note(
        what,
        "wetBulb",
        &["the report's host-side wet bulb", "dewPoint", "relativeHumidity"],
        "(S54.6) is a scalar root-find per cell. Its ITERATION COUNT depends on the \
         data, which makes a kernel's trip count data-dependent and therefore not \
         CUDA-Graph capturable (S54.5). The fixed-3-step Newton that would be \
         capturable is accurate to about 0.3 K, and 0.3 K is not accurate enough for \
         a number a customer reads off a psychrometric chart. Wet bulb is a \
         REPORTING quantity - nothing in the physics needs it - so it is computed on \
         the host, from downloaded fields, where a convergence failure can be an \
         error instead of a wrong number",
        "the report's host-side wet bulb (crate::psychro::t_wb)",
        (),
    )
}

// ==========================================================================
//  3. The device side
// ==========================================================================

struct PsychroKernels {
    state: CudaFunction,
    tv: CudaFunction,
    tv_boundary: CudaFunction,
    supersat: CudaFunction,
}

impl PsychroKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::FAN)?;
        Ok(Self {
            state: k.func("psyState")?,
            tv: k.func("psyVirtualTemperature")?,
            tv_boundary: k.func("psyVirtualTemperatureBoundary")?,
            supersat: k.func("psySupersaturation")?,
        })
    }
}

/// What a supersaturation sweep found - §54.5's report, not a clip.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Supersaturation {
    /// How many cells hold more vapour than saturation allows.
    pub cells: usize,
    /// The worst `Y_v - Y_v,sat`.
    pub worst: Scalar,
}

/// The psychrometric state of one mesh, on the device.
///
/// Holds `W`, `rh`, `h`, `v` and the virtual temperature. Nothing here is on
/// the flow solver's critical path: the psychrometric fields are diagnostics
/// and the virtual temperature is one elementwise kernel whose output is
/// handed to an **unmodified** `momentum::update_buoyancy`.
pub struct Psychrometrics {
    k: PsychroKernels,
    solk: SolverKernels,
    n: usize,
    nbf: usize,
    p_atm: Scalar,

    pub w: DevBuf<Scalar>,
    pub rh: DevBuf<Scalar>,
    pub h: DevBuf<Scalar>,
    pub v: DevBuf<Scalar>,
    excess: DevBuf<Scalar>,
    partials: DevBuf<Scalar>,
    red: DevBuf<Scalar>,

    /// The virtual temperature, as a full field so it can be passed straight
    /// to `momentum::update_buoyancy` in place of `T`.
    tv: GpuScalarField,
}

impl Psychrometrics {
    pub fn new(gpu: &Gpu, m: &GpuMesh, p_atm: Scalar) -> Result<Self> {
        if !(p_atm > 0.0) {
            return Err(Error::Config(format!(
                "Psychrometrics: p_atm = {p_atm} Pa must be positive - it is the \
                 TOTAL barometric pressure of (S54.2b), not this solver's kinematic \
                 gauge pressure"
            )));
        }
        let n = m.n_cells.max(1);
        Ok(Self {
            k: PsychroKernels::new(gpu)?,
            solk: SolverKernels::new(gpu)?,
            n: m.n_cells,
            nbf: m.n_boundary_faces,
            p_atm,
            w: gpu.zeros(n)?,
            rh: gpu.zeros(n)?,
            h: gpu.zeros(n)?,
            v: gpu.zeros(n)?,
            excess: gpu.zeros(n)?,
            partials: gpu.zeros(solver::reduce_partitions(n).max(1))?,
            red: gpu.zeros(1)?,
            tv: GpuScalarField::zeros(gpu, m, "Tv")?,
        })
    }

    pub fn p_atm(&self) -> Scalar {
        self.p_atm
    }

    /// The virtual-temperature field. Hand this to
    /// `momentum::update_buoyancy` in place of `T`; see §54.4 for why that
    /// leaves `src/momentum.rs` untouched.
    pub fn virtual_temperature_field(&self) -> &GpuScalarField {
        &self.tv
    }

    /// Recompute `T_v` from `(T, Y_v)`, cells and boundary faces.
    ///
    /// The boundary half matters: the buoyancy flux is evaluated on faces
    /// from the interpolated face temperature, and a `T_v` whose boundary
    /// values were still `T` would put a spurious density step on every wall.
    pub fn update_virtual_temperature(
        &mut self,
        gpu: &Gpu,
        t: &GpuScalarField,
        yv: &GpuScalarField,
    ) -> Result<()> {
        if self.n > 0 {
            let n = self.n as Label;
            let f = self.k.tv.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.tv.f)
                    .arg(&t.f)
                    .arg(&yv.f)
                    .arg(&n)
                    .launch(cfg_for(self.n))?;
            }
        }
        if self.nbf > 0 {
            let n = self.nbf as Label;
            let f = self.k.tv_boundary.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.tv.bf)
                    .arg(&t.bf)
                    .arg(&yv.bf)
                    .arg(&n)
                    .launch(cfg_for(self.nbf))?;
            }
        }
        Ok(())
    }

    /// Recompute `W`, `rh`, `h`, `v` from `(T, Y_v, p_atm)` - the diagnostic
    /// fields a report reads.
    pub fn update(&mut self, gpu: &Gpu, t: &GpuScalarField, yv: &GpuScalarField) -> Result<()> {
        if self.n == 0 {
            return Ok(());
        }
        let n = self.n as Label;
        let p = self.p_atm;
        let f = self.k.state.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.w)
                .arg(&mut self.rh)
                .arg(&mut self.h)
                .arg(&mut self.v)
                .arg(&t.f)
                .arg(&yv.f)
                .arg(&p)
                .arg(&n)
                .launch(cfg_for(self.n))?;
        }
        Ok(())
    }

    /// §54.5: how much of the field is supersaturated, **reported and not
    /// clipped**.
    pub fn supersaturation(
        &mut self,
        gpu: &Gpu,
        t: &GpuScalarField,
        yv: &GpuScalarField,
    ) -> Result<Supersaturation> {
        if self.n == 0 {
            return Ok(Supersaturation::default());
        }
        let n = self.n as Label;
        let p = self.p_atm;
        let f = self.k.supersat.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.excess)
                .arg(&t.f)
                .arg(&yv.f)
                .arg(&p)
                .arg(&n)
                .launch(cfg_for(self.n))?;
        }
        solver::device_max_mag(gpu, &self.solk, &mut self.red, &self.excess, &mut self.partials, self.n)?;
        let worst = gpu.download(&self.red)?[0];
        let e = gpu.download(&self.excess)?;
        let cells = e.iter().take(self.n).filter(|x| **x > 0.0).count();
        Ok(Supersaturation { cells, worst })
    }

    /// §54.4's fixed-molar-mass caveat, as the sentence a report has to
    /// carry, or `None` where it does not apply.
    ///
    /// `GasProperties::w` stays a scalar; the virtual temperature makes that
    /// exact for buoyancy and approximate for the low-Mach divergence
    /// constraint. Under `Y_v = 0.05` the error on `rho` is under 3 %; above
    /// it, the case is told.
    pub fn molar_mass_caveat(max_yv: Scalar) -> Option<String> {
        if max_yv <= 0.05 {
            return None;
        }
        Some(format!(
            "humidity reaches Y_v = {max_yv:.4}. SPEC-LIT S54.4: the mixture molar \
             mass is carried as the SCALAR GasProperties::w plus a virtual \
             temperature. That is EXACT for buoyancy and approximate for the \
             low-Mach divergence constraint and the p0 ODE, where the error on rho \
             is about {:.1} %. Below Y_v = 0.05 that is under 3 % and inside every \
             other modelling assumption here; above it, the assumption is worth \
             knowing about.",
            100.0 * VIRTUAL_COEFF * max_yv
        ))
    }
}
