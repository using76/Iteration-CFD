// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The perfect-gas state of the compressible solver, its host twins and its
//! device launchers - SPEC-LIT §122.1, §122.2 and §122.5 (docs/17 CMP-08).
//! The gas is thermally and calorically perfect. The pressure unknown `p`
//! is a GAUGE pressure about the host constant `p_op` (§122.2), which
//! reaches a kernel only inside the one sum `p + p_op` of (122.10). A case
//! states `gamma` and `W`; `c_p` follows from them by (122.5), and a stated
//! `c_p` that disagrees by more than 1e-6 relative is refused by name
//! (`physics.compressible.cp`), because an inconsistent `c_p` would make
//! the energy equation and the isentropic boundary relations disagree
//! about the same total state (§122.1).
//!
//! Written from:
//!   ofgpu SPEC-LIT.md §122.1 - the state, Sutherland's law with its
//!     constants for air, the `c_p` consistency rule
//!   ofgpu SPEC-LIT.md §122.2 - the gauge pressure, and why it is the f32
//!     design
//!   ofgpu SPEC-LIT.md §122.5 - the operation order of (122.9)-(122.13),
//!     which fixes the device/host gate at four units in the last place
//!   U.S. Standard Atmosphere 1976, NOAA-S/T 76-1562 (NASA-TM-X-74335,
//!     NTRS 19770009539), a US-Government work - eq. (51), Sutherland's
//!     law as `mu = A_s T^(3/2) / (T + T_s)` with `A_s = 1.458e-6`
//!     kg/(m s K^(1/2)) and `T_s = 110.4` K for air
//!   Ames Research Staff, *Equations, Tables, and Charts for Compressible
//!     Flow*, NACA Report 1135 (1953), NTRS 19930091059, a US-Government
//!     work - eq. (34), `p/rho^gamma = constant`, from which SPEC-LIT
//!     §122.1 derives the sound speed (122.4), `c = sqrt(gamma R_s T)`
//!
//! Provenance: LITERATURE. §122.5 fixes the order of every operation in
//! (122.9)-(122.13); the host twins below and the kernels of
//! `cuda/compressible.cu` make it operation for operation, and the gate
//! between them is four units in the last place, not bitwise, because the
//! device is the reference and the host is a check (§67.11) and the
//! contraction may fuse the one sum of squares in (122.12). No
//! GPL-licensed source was consulted.

#![allow(clippy::excessive_precision)]

use crate::chem::thermo::R_GAS;
use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::dust::thermo::EquivalentGas;
use crate::error::{Error, Result};
use crate::types::Vec3;
use crate::{Label, Scalar};
use cudarc::driver::{CudaFunction, PushKernelArg};

/// A refusal in the one house format.
fn refuse(who: &str, msg: String) -> Error {
    Error::Refused(format!("compressible::thermo::{who}: {msg}"))
}

/// The universal gas constant, (122.2), J/(mol K): CODATA 2018's value,
/// which SPEC-LIT §122.1 fixes for both builds.
pub const R_UNIVERSAL: f64 = 8.314462618;

/// The viscosity law of the gas, §122.1: a constant, or Sutherland's law
/// in the form the U.S. Standard Atmosphere 1976 writes it (eq. (51)),
/// `mu(T) = a_s T^(3/2) / (T + t_s)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Viscosity {
    /// `mu` at every temperature; `mu = 0` is the inviscid gas and is
    /// allowed.
    Constant { mu: f64 },
    /// Sutherland's law with its two constants, (122.6).
    Sutherland { a_s: f64, t_s: f64 },
}

/// The perfect gas of §122.1: thermally and calorically perfect, stated by
/// its `gamma` and `W` - `c_p` follows from them by (122.5) and is never an
/// input. `p_op` is the gauge datum of §122.2, a host `f64` constant in
/// both builds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerfectGas {
    /// The ratio of specific heats, (122.4)'s `gamma`.
    pub gamma: f64,
    /// The molar mass, kg/mol; `R_s = R / W` by (122.2).
    pub w: f64,
    /// The laminar Prandtl number, defaulting to 0.71 for air (§122.1).
    pub pr: f64,
    /// The turbulent Prandtl number, defaulting to 0.85 (§122.1).
    pub pr_t: f64,
    /// The viscosity law, (122.6) or the constant.
    pub viscosity: Viscosity,
    /// The gauge datum `p_op`, Pa, absolute on the host (§122.2). A case
    /// that omits it is refused by name (CMP-14), not defaulted.
    pub p_op: f64,
}

impl PerfectGas {
    /// Air at the gauge datum `p_op`: `gamma = 1.4`, `W = 0.0289647` kg/mol,
    /// `Pr = 0.71`, `Pr_t = 0.85`, Sutherland's constants for air
    /// `a_s = 1.458e-6`, `t_s = 110.4` (§122.1). From these,
    /// `R_s = 287.0550227690948` and `c_p = 1004.692579691832` J/(kg K)
    /// by (122.2) and (122.5).
    pub fn air(p_op: f64) -> Self {
        Self {
            gamma: 1.4,
            w: 0.0289647,
            pr: 0.71,
            pr_t: 0.85,
            viscosity: Viscosity::Sutherland { a_s: 1.458e-6, t_s: 110.4 },
            p_op,
        }
    }

    /// Every field must give the gas law a state: anything else is refused
    /// by name, with its value (§13.4).
    pub fn validate(&self) -> Result<()> {
        if !self.gamma.is_finite() || self.gamma <= 1.0 {
            return Err(refuse(
                "validate",
                format!("gamma must be finite and above 1, got {}", self.gamma),
            ));
        }
        if !self.w.is_finite() || self.w <= 0.0 {
            return Err(refuse(
                "validate",
                format!("w must be finite and above 0, got {}", self.w),
            ));
        }
        if !self.pr.is_finite() || self.pr <= 0.0 {
            return Err(refuse(
                "validate",
                format!("pr must be finite and above 0, got {}", self.pr),
            ));
        }
        if !self.pr_t.is_finite() || self.pr_t <= 0.0 {
            return Err(refuse(
                "validate",
                format!("pr_t must be finite and above 0, got {}", self.pr_t),
            ));
        }
        if !self.p_op.is_finite() || self.p_op < 0.0 {
            return Err(refuse(
                "validate",
                format!("p_op must be finite and not below 0, got {}", self.p_op),
            ));
        }
        match self.viscosity {
            Viscosity::Constant { mu } => {
                if !mu.is_finite() || mu < 0.0 {
                    return Err(refuse(
                        "validate",
                        format!(
                            "mu must be finite and not below 0 (mu = 0 is the \
                             inviscid gas), got {mu}"
                        ),
                    ));
                }
            }
            Viscosity::Sutherland { a_s, t_s } => {
                if !a_s.is_finite() || a_s <= 0.0 {
                    return Err(refuse(
                        "validate",
                        format!("a_s must be finite and above 0, got {a_s}"),
                    ));
                }
                if !t_s.is_finite() || t_s < 0.0 {
                    return Err(refuse(
                        "validate",
                        format!("t_s must be finite and not below 0, got {t_s}"),
                    ));
                }
            }
        }
        Ok(())
    }

    /// The specific gas constant, (122.2): `R_s = R / W`, in f64 in both
    /// builds - it feeds (122.9)'s `rs` and `grs`, and the round-trip gate
    /// quotes it to `1e-15`.
    pub fn r_s(&self) -> f64 {
        R_UNIVERSAL / self.w
    }

    /// The specific heat at constant pressure, (122.5):
    /// `gamma * r_s / (gamma - 1)`, formed in that order, in f64 in both
    /// builds (§122.5: the constants are formed once in f64 and rounded
    /// once).
    pub fn cp(&self) -> f64 {
        self.gamma * self.r_s() / (self.gamma - 1.0)
    }

    /// The specific heat at constant volume, (122.5): `c_v = c_p - R_s`.
    pub fn cv(&self) -> f64 {
        self.cp() - self.r_s()
    }

    /// §122.1's consistency rule: a stated `c_p` must equal (122.5) within
    /// `1e-6` relative, or the case is refused by name
    /// (`physics.compressible.cp`) with the stated and the derived value
    /// beside each other.
    pub fn check_stated_cp(&self, stated: f64) -> Result<()> {
        let cp = self.cp();
        if (stated - cp).abs() <= 1e-6 * cp {
            return Ok(());
        }
        Err(refuse(
            "check_stated_cp",
            format!(
                "the stated c_p {stated} disagrees with the c_p (122.5) \
                 derives from gamma and W, {cp}, by more than 1e-6 relative \
                 (physics.compressible.cp)"
            ),
        ))
    }

    /// The numbers of (122.9), each formed once in f64 and rounded once to
    /// the working precision. The kernels receive exactly these, and the
    /// host twins below evaluate on the same rounded numbers - that is what
    /// makes the gate a count of ulps rather than a hope. When the
    /// viscosity is constant, `a_s = 0`, `t_s = 0` and `mu0 = fl(mu)`;
    /// when it is Sutherland, `mu0 = 0`.
    pub fn constants(&self) -> GasConstants {
        let rs = self.r_s();
        let cpr = self.cp() / self.pr;
        let (sutherland, a_s, t_s, mu0) = match self.viscosity {
            Viscosity::Constant { mu } => (false, 0.0, 0.0, mu),
            Viscosity::Sutherland { a_s, t_s } => (true, a_s, t_s, 0.0),
        };
        GasConstants {
            rs: rs as Scalar,
            pop: self.p_op as Scalar,
            grs: (self.gamma * rs) as Scalar,
            cpr: cpr as Scalar,
            sutherland,
            a_s: a_s as Scalar,
            t_s: t_s as Scalar,
            mu0: mu0 as Scalar,
        }
    }
}

/// The rounded constants of (122.9): everything a kernel sees, and
/// everything the host twins evaluate on. No kernel ever receives `gamma`,
/// `W` or `Pr` (§122.5).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GasConstants {
    /// `rs = fl(R_s)`, (122.2).
    pub rs: Scalar,
    /// `pop = fl(p_op)`, the gauge datum of §122.2.
    pub pop: Scalar,
    /// `grs = fl(gamma R_s)`, the sound speed's constant in (122.12).
    pub grs: Scalar,
    /// `cpr = fl(c_p / Pr)`, (122.7)'s constant in (122.11).
    pub cpr: Scalar,
    /// Which viscosity branch (122.11) takes: Sutherland when true.
    pub sutherland: bool,
    /// `as = fl(A_s)`, 0 under a constant viscosity.
    pub a_s: Scalar,
    /// `ts = fl(T_s)`, 0 under a constant viscosity.
    pub t_s: Scalar,
    /// `mu0 = fl(mu)` under a constant viscosity, 0 under Sutherland.
    pub mu0: Scalar,
}

impl GasConstants {
    /// (122.10): `psi = 1 / (rs T)`. `psi` is `d rho/d p` at constant `T`
    /// exactly, which §124 uses.
    pub fn psi(&self, t: Scalar) -> Scalar {
        1.0 as Scalar / (self.rs * t)
    }

    /// (122.10): `rho = psi (p + pop)` - the gauge pressure meets `pop` in
    /// one sum in the working precision, then one product. That sum is
    /// what §122.2's f32 oracle measured.
    pub fn rho(&self, p: Scalar, t: Scalar) -> Scalar {
        self.psi(t) * (p + self.pop)
    }

    /// (122.13), the host-only inverse: `p = rho / psi - pop`.
    pub fn p_from_rho(&self, rho: Scalar, t: Scalar) -> Scalar {
        rho / self.psi(t) - self.pop
    }

    /// (122.11): Sutherland's `mu = (as (T sqrt(T))) / (T + ts)` - one
    /// correctly rounded square root and one product, where `pow` would
    /// differ in its last bit between the host's library and the device's
    /// (§122.5) - or the constant `mu0`.
    pub fn mu(&self, t: Scalar) -> Scalar {
        if !self.sutherland {
            return self.mu0;
        }
        (self.a_s * (t * t.sqrt())) / (t + self.t_s)
    }

    /// (122.11): `kappa = mu cpr`.
    pub fn kappa(&self, t: Scalar) -> Scalar {
        self.mu(t) * self.cpr
    }

    /// (122.12): `c = sqrt(grs T)`.
    pub fn sound_speed(&self, t: Scalar) -> Scalar {
        (self.grs * t).sqrt()
    }

    /// (122.12): `M = sqrt(u_x u_x + u_y u_y + u_z u_z) / c` - the sum of
    /// squares left to right, one square root, one division. At rest this
    /// is exactly 0.
    pub fn mach(&self, u: Vec3, t: Scalar) -> Scalar {
        (u.x * u.x + u.y * u.y + u.z * u.z).sqrt() / self.sound_speed(t)
    }
}

/// The three kernels of `cuda/compressible.cu` (§122.5), loaded from the
/// `COMPRESSIBLE` module `build.rs` compiles.
pub struct ThermoKernels {
    gas_state: CudaFunction,
    transport: CudaFunction,
    sound_mach: CudaFunction,
}

impl ThermoKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::COMPRESSIBLE)?;
        Ok(Self {
            gas_state: k.func("cmpGasState")?,
            transport: k.func("cmpTransport")?,
            sound_mach: k.func("cmpSoundMach")?,
        })
    }
}

/// A launcher's one precondition: the buffer holds at least `n` elements.
/// A shorter buffer would have the kernel write past its end.
fn check_len(who: &str, name: &str, len: usize, n: usize) -> Result<()> {
    if len < n {
        return Err(refuse(
            who,
            format!("buffer {name} holds {len} elements, fewer than n = {n}"),
        ));
    }
    Ok(())
}

/// (122.10) elementwise: `psi[i] = psi(t[i])` and
/// `rho[i] = psi(t[i]) (p[i] + pop)`. A launcher only launches: no sync,
/// no download, no allocation - so a captured region can hold it.
#[allow(clippy::too_many_arguments)]
pub fn gas_state(
    gpu: &Gpu,
    k: &ThermoKernels,
    c: &GasConstants,
    psi: &mut DevBuf<Scalar>,
    rho: &mut DevBuf<Scalar>,
    p: &DevBuf<Scalar>,
    t: &DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    check_len("gas_state", "psi", psi.len(), n)?;
    check_len("gas_state", "rho", rho.len(), n)?;
    check_len("gas_state", "p", p.len(), n)?;
    check_len("gas_state", "t", t.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.gas_state)
            .arg(&mut *psi)
            .arg(&mut *rho)
            .arg(p)
            .arg(t)
            .arg(&nl)
            .arg(&c.rs)
            .arg(&c.pop)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// (122.11) elementwise: `mu[i]` by Sutherland's law or the constant, and
/// `kappa[i] = mu cpr`. The `sutherland` half of `c` picks the kernel's
/// `model` argument.
#[allow(clippy::too_many_arguments)]
pub fn transport(
    gpu: &Gpu,
    k: &ThermoKernels,
    c: &GasConstants,
    mu: &mut DevBuf<Scalar>,
    kappa: &mut DevBuf<Scalar>,
    t: &DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    check_len("transport", "mu", mu.len(), n)?;
    check_len("transport", "kappa", kappa.len(), n)?;
    check_len("transport", "t", t.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let model: Label = if c.sutherland { 1 } else { 0 };
    unsafe {
        gpu.stream()
            .launch_builder(&k.transport)
            .arg(&mut *mu)
            .arg(&mut *kappa)
            .arg(t)
            .arg(&nl)
            .arg(&model)
            .arg(&c.a_s)
            .arg(&c.t_s)
            .arg(&c.mu0)
            .arg(&c.cpr)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// (122.12) elementwise: the sound speed and the Mach number. At rest the
/// device's `M` is exactly 0.
#[allow(clippy::too_many_arguments)]
pub fn sound_mach(
    gpu: &Gpu,
    k: &ThermoKernels,
    c: &GasConstants,
    cs: &mut DevBuf<Scalar>,
    mach: &mut DevBuf<Scalar>,
    u: &DevBuf<Vec3>,
    t: &DevBuf<Scalar>,
    n: usize,
) -> Result<()> {
    check_len("sound_mach", "c", cs.len(), n)?;
    check_len("sound_mach", "mach", mach.len(), n)?;
    check_len("sound_mach", "u", u.len(), n)?;
    check_len("sound_mach", "t", t.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.sound_mach)
            .arg(&mut *cs)
            .arg(&mut *mach)
            .arg(u)
            .arg(t)
            .arg(&nl)
            .arg(&c.grs)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  The dust model's hConst pair and its mixture kernels (SPEC-LIT §158.4)
// ==========================================================================

/// The `hConst` pair of the two pseudo-species as the compressible energy
/// equation and the mixture gas law read them (SPEC-LIT §158.4): one
/// constant `c_p` per pseudo-species, so `h_X(T) = a_X + cp_X T` per kg on
/// the 0 K datum, `R_mix(c~)` from (152.8), and Δh_c the formation-enthalpy
/// difference of U and B. Every number is host f64; a launcher rounds once.
pub struct DustHConst {
    /// J/(kg K), the unburned pseudo-species' constant `c_p`.
    pub cp_u: f64,
    /// J/(kg K), the burned pseudo-species' constant `c_p`.
    pub cp_b: f64,
    /// kg/mol, the unburned pseudo-species' molar mass.
    pub w_u: f64,
    /// kg/mol, the burned pseudo-species' molar mass.
    pub w_b: f64,
    /// J/kg, the unburned formation enthalpy `h_U(T_u) - cp_u T_u`.
    pub a_u: f64,
    /// J/kg, the burned formation enthalpy `h_B(T_u) - cp_b T_u`.
    pub a_b: f64,
}

impl DustHConst {
    /// From the equivalent gas at the unburned temperature `t_u` (SPEC-LIT
    /// §158.4): `cp_b` the mean of `h_B` over [`T_u`, `T_b`], which keeps
    /// the NASA-7 isobaric `T_b` exactly (`a_b + cp_b T_b = h_B(T_b) =
    /// h_U(T_u)`), and the formation enthalpies `a_X = h_X(T_u) - cp_X T_u`.
    /// A `c_p` or `c_v` that is not finite and positive is refused by name
    /// (§158.4).
    pub fn from_gas(gas: &EquivalentGas, t_u: f64) -> Result<Self> {
        let who = "compressible::thermo::DustHConst::from_gas";
        let t_b = gas.isobaric_burned(t_u)?.t_b;
        let cp_u = gas.cp_u(t_u)?;
        let (h_b_hi, h_b_lo) = (gas.h_b(t_b)?, gas.h_b(t_u)?);
        let cp_b = (h_b_hi - h_b_lo) / (t_b - t_u);
        let a_u = gas.h_u(t_u)? - cp_u * t_u;
        let a_b = h_b_lo - cp_b * t_u;
        let out = Self { cp_u, cp_b, w_u: gas.w_u(), w_b: gas.w_b(), a_u, a_b };
        for (name, v) in [
            ("cp_u", out.cp_u),
            ("cp_b", out.cp_b),
            ("cv_u", out.cp_u - R_GAS / out.w_u),
            ("cv_b", out.cp_b - R_GAS / out.w_b),
        ] {
            if !v.is_finite() || v <= 0.0 {
                return Err(Error::Config(format!(
                    "{who}: {name} = {v} is not a finite positive value (SPEC-LIT §158.4)"
                )));
            }
        }
        Ok(out)
    }

    /// (152.8): `c_p(T, c~) = (1 - c~) c_p,U + c~ c_p,B` - the constant per
    /// pseudo-species, so the temperature drops out.
    pub fn cp(&self, c: f64) -> f64 {
        self.cp_u + c * (self.cp_b - self.cp_u)
    }

    /// J/(kg K), the unburned gas constant `R0/W_U`.
    pub fn r_u(&self) -> f64 {
        R_GAS / self.w_u
    }

    /// J/(kg K), the burned gas constant `R0/W_B`.
    pub fn r_b(&self) -> f64 {
        R_GAS / self.w_b
    }

    /// (152.8): `r(c~) = r_u + c~ (r_b - r_u)`, `R0/W(c~)` per kg.
    pub fn r(&self, c: f64) -> f64 {
        self.r_u() + c * (self.r_b() - self.r_u())
    }

    /// (158.4)'s `q~ = Δh_c wdot_c`: the formation-enthalpy difference of
    /// U and B, J/kg, positive for a burn.
    pub fn delta_h_c(&self) -> f64 {
        self.a_u - self.a_b
    }

    /// K, the isobaric burned temperature of the hConst pair,
    /// `T_b = (a_u - a_b + cp_u T_u)/cp_b` - the NASA-7 (152.10) value.
    pub fn isobaric_tb(&self, t_u: f64) -> f64 {
        (self.a_u - self.a_b + self.cp_u * t_u) / self.cp_b
    }

    /// (152.11) on the hConst pair: `t_v` from `u_B(T_v) = u_U(T_0)` with
    /// `u_X = a_X + cv_X T`, and `P_max/p_0 = T_v W_U/(T_0 W_B)` - the
    /// compressible AICC identity, NOT the NASA-7 evaluation.
    pub fn aicc(&self, t0: f64) -> (f64, f64) {
        let cv_u = self.cp_u - R_GAS / self.w_u;
        let cv_b = self.cp_b - R_GAS / self.w_b;
        let t_v = (self.a_u - self.a_b + cv_u * t0) / cv_b;
        let p_ratio = t_v * self.w_u / (t0 * self.w_b);
        (t_v, p_ratio)
    }

    /// The unburned `gamma` of the hConst pair, (155.10)'s exponent.
    pub fn gamma_u(&self) -> f64 {
        let cv_u = self.cp_u - R_GAS / self.w_u;
        self.cp_u / cv_u
    }
}

/// The three mixture kernels of `cuda/compressible.cu` (§158.4), loaded
/// from the `COMPRESSIBLE` module `build.rs` compiles.
pub struct MixKernels {
    gas_state_mix: CudaFunction,
    scale_cp: CudaFunction,
    rho_u_ratio: CudaFunction,
}

impl MixKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::COMPRESSIBLE)?;
        Ok(Self {
            gas_state_mix: k.func("cmpGasStateMix")?,
            scale_cp: k.func("cmpScaleCp")?,
            rho_u_ratio: k.func("cmpDustRhoURatio")?,
        })
    }
}

/// (152.8)+(152.9) elementwise: `rs = rs_u + c drs`, `psi = 1/(rs T)` and
/// `rho = psi (p + pop)` - the mixture gas law about the gauge datum, the
/// constants formed in f64 and rounded once here. A launcher only launches.
#[allow(clippy::too_many_arguments)]
pub fn gas_state_mix(
    gpu: &Gpu,
    k: &MixKernels,
    hc: &DustHConst,
    psi: &mut DevBuf<Scalar>,
    rho: &mut DevBuf<Scalar>,
    p: &DevBuf<Scalar>,
    t: &DevBuf<Scalar>,
    c: &DevBuf<Scalar>,
    n: usize,
    pop: f64,
) -> Result<()> {
    check_len("gas_state_mix", "psi", psi.len(), n)?;
    check_len("gas_state_mix", "rho", rho.len(), n)?;
    check_len("gas_state_mix", "p", p.len(), n)?;
    check_len("gas_state_mix", "t", t.len(), n)?;
    check_len("gas_state_mix", "c", c.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let (rs_u, drs) = (hc.r_u() as Scalar, (hc.r_b() - hc.r_u()) as Scalar);
    let pop = pop as Scalar;
    unsafe {
        gpu.stream()
            .launch_builder(&k.gas_state_mix)
            .arg(&mut *psi)
            .arg(&mut *rho)
            .arg(p)
            .arg(t)
            .arg(c)
            .arg(&nl)
            .arg(&rs_u)
            .arg(&drs)
            .arg(&pop)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// (152.8) elementwise: `out = a (cp_u + c dcp)` - `a` the carried field
/// (rho for the energy weights, 1 for a bare `c_p`), the constants formed
/// in f64 and rounded once here.
#[allow(clippy::too_many_arguments)]
pub fn scale_cp(
    gpu: &Gpu,
    k: &MixKernels,
    out: &mut DevBuf<Scalar>,
    a: &DevBuf<Scalar>,
    c: &DevBuf<Scalar>,
    n: usize,
    cp_u: f64,
    dcp: f64,
) -> Result<()> {
    check_len("scale_cp", "out", out.len(), n)?;
    check_len("scale_cp", "a", a.len(), n)?;
    check_len("scale_cp", "c", c.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let (cp_u, dcp) = (cp_u as Scalar, dcp as Scalar);
    unsafe {
        gpu.stream()
            .launch_builder(&k.scale_cp)
            .arg(&mut *out)
            .arg(a)
            .arg(c)
            .arg(&nl)
            .arg(&cp_u)
            .arg(&dcp)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// (155.10) elementwise: `ut_out = ut ((p + pop)/p_init)^(1/gamma_u)` - the
/// propagation speed carried at the LOCAL pressure, the power in double on
/// the device and the constants in f64 on the host, rounded once at the
/// write.
#[allow(clippy::too_many_arguments)]
pub fn rho_u_ratio(
    gpu: &Gpu,
    k: &MixKernels,
    ut_out: &mut DevBuf<Scalar>,
    ut: &DevBuf<Scalar>,
    p: &DevBuf<Scalar>,
    n: usize,
    pop: f64,
    p_init: f64,
    gamma_u: f64,
) -> Result<()> {
    check_len("rho_u_ratio", "ut_out", ut_out.len(), n)?;
    check_len("rho_u_ratio", "ut", ut.len(), n)?;
    check_len("rho_u_ratio", "p", p.len(), n)?;
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    let inv_gamma = 1.0 / gamma_u;
    unsafe {
        gpu.stream()
            .launch_builder(&k.rho_u_ratio)
            .arg(&mut *ut_out)
            .arg(ut)
            .arg(p)
            .arg(&nl)
            .arg(&pop)
            .arg(&p_init)
            .arg(&inv_gamma)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The process's device, if the box has one: every GPU test returns
    /// early rather than failing.
    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// SplitMix64, seed 20261013; `u()` is uniform on [0, 1) as
    /// `(x >> 11) 2^-53`.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn u(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        }
    }

    /// The states of §122.5's gate table: 4096 random states, each drawn in
    /// the order `T = 200 + 1800 u`, `p = -5e4 + 5.5e5 u`, then one draw
    /// per velocity component on [-800, 800], all formed in f64 and cast
    /// once to `Scalar`; then state 4096, the rest state
    /// `T = 300`, `p = 0`, `U = 0`. So `n = 4097`.
    fn fixture() -> (Vec<Scalar>, Vec<Scalar>, Vec<Vec3>) {
        let mut r = SplitMix64(20261013);
        let mut t = Vec::with_capacity(4097);
        let mut p = Vec::with_capacity(4097);
        let mut u = Vec::with_capacity(4097);
        for _ in 0..4096 {
            let tf = 200.0 + 1800.0 * r.u();
            let pf = -5.0e4 + 5.5e5 * r.u();
            let ux = -800.0 + 1600.0 * r.u();
            let uy = -800.0 + 1600.0 * r.u();
            let uz = -800.0 + 1600.0 * r.u();
            t.push(tf as Scalar);
            p.push(pf as Scalar);
            u.push(Vec3::new(ux as Scalar, uy as Scalar, uz as Scalar));
        }
        t.push(300.0 as Scalar);
        p.push(0.0 as Scalar);
        u.push(Vec3::ZERO);
        (t, p, u)
    }

    /// The two gases of the gate table: Sutherland air and the same gas at
    /// a constant viscosity, both about `p_op = 101325` Pa.
    fn gases() -> Vec<(&'static str, PerfectGas)> {
        let suth = PerfectGas::air(101325.0);
        let konst = PerfectGas { viscosity: Viscosity::Constant { mu: 1.8e-5 }, ..suth };
        vec![("sutherland air", suth), ("constant 1.8e-5", konst)]
    }

    /// §122.5's ulp distance: the count of representable values between two
    /// finite numbers of one sign, which is the difference of their bit
    /// patterns read as integers. A different sign or a non-finite value
    /// fails here, naming the field and the state.
    fn ulp_distance(a: Scalar, b: Scalar, field: &str, i: usize) -> u64 {
        assert!(
            a.is_finite() && b.is_finite(),
            "{field} at state {i}: non-finite, device {a:e} host {b:e}"
        );
        assert!(
            (a == 0.0 && b == 0.0) || a.is_sign_negative() == b.is_sign_negative(),
            "{field} at state {i}: the signs differ, device {a:e} host {b:e}"
        );
        a.to_bits().abs_diff(b.to_bits()) as u64
    }

    /// The distance from `x` to the next representable number above it -
    /// the one-ulp bar for the round trip through (122.10) and (122.13).
    fn ulp(x: Scalar) -> Scalar {
        Scalar::from_bits(x.to_bits() + 1) - x
    }

    /// The R-class relative tolerance (D-F32-1): `1e-14` in the default
    /// build; under `single`, `max(1e-14, min(1e-14 * 2^29, 1e-5))`.
    fn r_tol() -> f64 {
        if cfg!(feature = "single") {
            (1.0e-14 * (1u64 << 29) as f64).min(1.0e-5).max(1.0e-14)
        } else {
            1.0e-14
        }
    }

    /// `|got - want| / |want|`, in f64 in both builds - the R rows compare
    /// in f64 (§122.5).
    fn rel(got: Scalar, want: f64) -> f64 {
        (f64::from(got) - want).abs() / want.abs()
    }

    /// The six fields the device twin returns, in launcher order.
    const FIELDS: [&str; 6] = ["psi", "rho", "mu", "kappa", "c", "mach"];

    /// §122.5's first gate: the three kernels against the host twins, both
    /// gases, all 4097 states, each field within four ulp - and `M`
    /// exactly 0 at the rest state.
    #[test]
    fn gpu_matches_host_to_4ulp() {
        let Some(g) = gpu() else { return };
        let k = ThermoKernels::new(&g).expect("kernels");
        let (t, p, u) = fixture();
        let n = t.len();
        let dt = g.upload(&t).expect("upload t");
        let dp = g.upload(&p).expect("upload p");
        let du = g.upload(&u).expect("upload u");
        let mut dpsi = g.zeros::<Scalar>(n).expect("alloc psi");
        let mut drho = g.zeros::<Scalar>(n).expect("alloc rho");
        let mut dmu = g.zeros::<Scalar>(n).expect("alloc mu");
        let mut dkappa = g.zeros::<Scalar>(n).expect("alloc kappa");
        let mut dc = g.zeros::<Scalar>(n).expect("alloc c");
        let mut dmach = g.zeros::<Scalar>(n).expect("alloc mach");
        for (name, gas) in gases() {
            let c = gas.constants();
            gas_state(&g, &k, &c, &mut dpsi, &mut drho, &dp, &dt, n).expect("gas_state");
            transport(&g, &k, &c, &mut dmu, &mut dkappa, &dt, n).expect("transport");
            sound_mach(&g, &k, &c, &mut dc, &mut dmach, &du, &dt, n).expect("sound_mach");
            g.sync().expect("sync");
            let psi = g.download(&dpsi).expect("download psi");
            let rho = g.download(&drho).expect("download rho");
            let mu = g.download(&dmu).expect("download mu");
            let kappa = g.download(&dkappa).expect("download kappa");
            let cs = g.download(&dc).expect("download c");
            let mach = g.download(&dmach).expect("download mach");
            let mut worst = [0u64; 6];
            for i in 0..n {
                let want = [
                    c.psi(t[i]),
                    c.rho(p[i], t[i]),
                    c.mu(t[i]),
                    c.kappa(t[i]),
                    c.sound_speed(t[i]),
                    c.mach(u[i], t[i]),
                ];
                let got = [psi[i], rho[i], mu[i], kappa[i], cs[i], mach[i]];
                for (f, _) in want.iter().enumerate() {
                    let d = ulp_distance(got[f], want[f], FIELDS[f], i);
                    worst[f] = worst[f].max(d);
                    assert!(
                        d <= 4,
                        "{}: at state {i} the device gave {:e} and the host \
                         {:e} - {d} ulp",
                        FIELDS[f],
                        got[f],
                        want[f]
                    );
                }
            }
            assert!(
                mach[n - 1] == 0.0,
                "at rest M is exactly 0, got {:e}",
                mach[n - 1]
            );
            println!(
                "  {name}: worst ulp  psi {}  rho {}  mu {}  kappa {}  c {}  \
                 mach {}",
                worst[0], worst[1], worst[2], worst[3], worst[4], worst[5]
            );
        }
    }

    /// §122.5's second gate: Sutherland air lands on the four digits White
    /// and the Standard Atmosphere print at 273.15 K, on the oracle's three
    /// values to the R-class tolerance, and the constant gas returns its
    /// number at every temperature.
    #[test]
    fn sutherland_air_at_273_15_is_1_716e_5() {
        let c = PerfectGas::air(101325.0).constants();
        let m273 = c.mu(273.15);
        assert!(
            (f64::from(m273) - 1.716e-5).abs() < 5e-9,
            "mu(273.15) = {m273:e}, the four digits White and the Standard \
             Atmosphere print"
        );
        assert!(
            rel(m273, 1.716079266245527e-05) <= r_tol(),
            "mu(273.15) = {m273:e}, wanted 1.716079266245527e-05"
        );
        let m300 = c.mu(300.0);
        assert!(
            rel(m300, 1.8460015185931457e-05) <= r_tol(),
            "mu(300) = {m300:e}, wanted 1.8460015185931457e-05"
        );
        let k300 = c.kappa(300.0);
        assert!(
            rel(k300, 0.026122028560991366) <= r_tol(),
            "kappa(300) = {k300:e}, wanted 0.026122028560991366"
        );
        let base = PerfectGas::air(101325.0);
        let konst = PerfectGas { viscosity: Viscosity::Constant { mu: 1.8e-5 }, ..base };
        let kc = konst.constants();
        let want: Scalar = 1.8e-5;
        for t in [200.0, 300.0, 2000.0] {
            assert!(
                kc.mu(t).to_bits() == want.to_bits(),
                "the constant gas at {t} K gave {:e}, wanted 1.8e-5",
                kc.mu(t)
            );
            assert!(
                kc.kappa(t).to_bits() == (kc.mu0 * kc.cpr).to_bits(),
                "kappa({t}) is mu0 cpr bit for bit, got {:e}",
                kc.kappa(t)
            );
        }
    }

    /// §122.5's third gate: `R_s` and `c_p` on the oracle to `1e-15`
    /// relative, `rho` and the sound speed on theirs to the R-class
    /// tolerance, the round trip `p -> rho -> p` within four ulp over all
    /// 4097 states, the stated-`c_p` refusals of §122.1, and `validate`'s
    /// refusals by name.
    #[test]
    fn ideal_gas_round_trip() {
        let gas = PerfectGas::air(101325.0);
        assert!(
            (gas.r_s() - 287.0550227690948).abs() / 287.0550227690948 <= 1e-15,
            "r_s = {}, wanted 287.0550227690948",
            gas.r_s()
        );
        assert!(
            (gas.cp() - 1004.692579691832).abs() / 1004.692579691832 <= 1e-15,
            "c_p = {}, wanted 1004.692579691832",
            gas.cp()
        );
        assert_eq!(gas.cv(), gas.cp() - gas.r_s(), "c_v is c_p - R_s exactly");
        let c = gas.constants();
        assert!(
            rel(c.rho(0.0, 288.15), 1.2249908312368587) <= r_tol(),
            "rho(0, 288.15) = {:e}, wanted 1.2249908312368587",
            c.rho(0.0, 288.15)
        );
        assert!(
            rel(c.sound_speed(288.15), 340.29526405061904) <= r_tol(),
            "c(288.15) = {:e}, wanted 340.29526405061904",
            c.sound_speed(288.15)
        );
        assert!(
            rel(c.sound_speed(300.0), 347.2219888817812) <= r_tol(),
            "c(300) = {:e}, wanted 347.2219888817812",
            c.sound_speed(300.0)
        );
        // The round trip over the 4097 states: (122.13) inverts (122.10)
        // to within four ulp of p_abs.
        let (t, p, _) = fixture();
        for (i, (&ti, &pi)) in t.iter().zip(p.iter()).enumerate() {
            let a = pi + c.pop;
            let back = c.p_from_rho(c.rho(pi, ti), ti);
            assert!(
                (back - pi).abs() <= 4.0 * ulp(a),
                "state {i}: p -> rho -> p gave {back:e}, wanted {pi:e}"
            );
        }
        // The stated-c_p rule of §122.1.
        assert!(gas.check_stated_cp(gas.cp()).is_ok());
        assert!(gas.check_stated_cp(gas.cp() * (1.0 + 5e-7)).is_ok());
        let Err(e) = gas.check_stated_cp(1006.0) else {
            panic!("c_p = 1006 is 0.13 percent off (122.5); it must be refused")
        };
        let m = format!("{e}");
        for what in ["physics.compressible.cp", "1006", "1004.69"] {
            assert!(m.contains(what), "'{what}' is not in the refusal: {m}");
        }
        // validate: air is fine, and each bad field is refused by name.
        assert!(gas.validate().is_ok());
        let mut g1 = gas;
        g1.gamma = 1.0;
        assert!(format!("{}", g1.validate().unwrap_err()).contains("gamma"));
        let mut g2 = gas;
        g2.w = 0.0;
        assert!(format!("{}", g2.validate().unwrap_err()).contains("w "));
        let mut g3 = gas;
        g3.p_op = -1.0;
        assert!(format!("{}", g3.validate().unwrap_err()).contains("p_op"));
        let mut g4 = gas;
        g4.viscosity = Viscosity::Constant { mu: -1.0 };
        assert!(format!("{}", g4.validate().unwrap_err()).contains("mu"));
        let g5 = PerfectGas { viscosity: Viscosity::Constant { mu: 0.0 }, ..gas };
        assert!(g5.validate().is_ok(), "mu = 0 is the inviscid gas");
    }

    /// §122.5's fourth gate, §122.2's point: a 1 Pa gauge perturbation on
    /// 1e5 Pa survives in the working precision, on the device and on the
    /// host twin alike - `rho(1) > rho(0)` and the perturbation back
    /// through `psi` within 1 percent. Under `single` the error at 300 K
    /// is asserted ABOVE 1e-3 (§122.2's oracle: 4.209e-3), so the row is
    /// known to measure f32 arithmetic and not a vacuous pass.
    #[test]
    fn f32_gauge_pressure_resolves_1pa() {
        let Some(g) = gpu() else { return };
        let k = ThermoKernels::new(&g).expect("kernels");
        let gas = PerfectGas::air(1.0e5);
        let c = gas.constants();
        let p = [0.0 as Scalar, 1.0 as Scalar];
        let dp = g.upload(&p).expect("upload p");
        let mut dpsi = g.zeros::<Scalar>(2).expect("alloc psi");
        let mut drho = g.zeros::<Scalar>(2).expect("alloc rho");
        for tv in [300.0, 288.15, 273.15] {
            let t = [tv as Scalar, tv as Scalar];
            let dt = g.upload(&t).expect("upload t");
            gas_state(&g, &k, &c, &mut dpsi, &mut drho, &dp, &dt, 2).expect("gas_state");
            g.sync().expect("sync");
            let r = g.download(&drho).expect("download rho");
            let h0 = c.rho(0.0, tv as Scalar);
            let h1 = c.rho(1.0, tv as Scalar);
            assert!(r[1] > r[0], "device: rho(1) = {:e}, rho(0) = {:e}", r[1], r[0]);
            assert!(h1 > h0, "host: rho(1) = {h1:e}, rho(0) = {h0:e}");
            let psi1 = f64::from(c.psi(tv as Scalar));
            let err_dev = ((f64::from(r[1]) - f64::from(r[0])) / psi1 - 1.0).abs();
            let err_host = ((f64::from(h1) - f64::from(h0)) / psi1 - 1.0).abs();
            assert!(
                err_dev <= 0.01,
                "device at {tv} K: the 1 Pa perturbation came back as \
                 {err_dev:.3e} of psi"
            );
            assert!(
                err_host <= 0.01,
                "host at {tv} K: the 1 Pa perturbation came back as \
                 {err_host:.3e} of psi"
            );
            println!("  {tv} K: device {err_dev:.4e}  host {err_host:.4e}");
            #[cfg(feature = "single")]
            {
                if tv == 300.0 {
                    assert!(
                        err_dev > 1e-3,
                        "the f32 control: the error at 300 K measured \
                         {err_dev:.3e}, and §122.2's oracle gives 4.209e-3 - \
                         above 1e-3"
                    );
                }
            }
        }
    }

    /// §81.7's row for `src/compressible/thermo.rs`: the three kernels
    /// capture into one graph and replay bitwise, on the precedent of
    /// `the_property_evaluation_replays_bitwise`. One iteration runs
    /// `gas_state`, `transport`, `sound_mach`, then feeds `mach` - scaled
    /// to at most a few millikelvin - into the next iteration's
    /// temperatures, so a replay that skipped a kernel cannot pass.
    #[test]
    fn the_compressible_thermo_update_replays_bitwise() {
        let Some(g) = gpu() else { return };
        let (t0, p0, u0) = fixture();
        let n = 256;
        let gas = PerfectGas::air(101325.0);
        let c = gas.constants();
        let k = ThermoKernels::new(&g).expect("kernels");
        let fk = crate::field_ops::FieldKernels::new(&g).expect("field kernels");
        let report = crate::capture::capture_replays_bitwise(
            &g,
            "the compressible gas state (SPEC-LIT §122.5)",
            || {
                Ok((
                    g.upload(&t0[..n].to_vec())?,
                    g.upload(&p0[..n].to_vec())?,
                    g.upload(&u0[..n].to_vec())?,
                    g.zeros::<Scalar>(n)?,
                    g.zeros::<Scalar>(n)?,
                    g.zeros::<Scalar>(n)?,
                    g.zeros::<Scalar>(n)?,
                    g.zeros::<Scalar>(n)?,
                    g.zeros::<Scalar>(n)?,
                ))
            },
            |(t, p, u, psi, rho, mu, kappa, cs, mach)| {
                gas_state(&g, &k, &c, psi, rho, p, t, n)?;
                transport(&g, &k, &c, mu, kappa, t, n)?;
                sound_mach(&g, &k, &c, cs, mach, u, t, n)?;
                // The result, scaled to at most a few millikelvin, feeds
                // the next iteration's temperatures, so a replay that
                // skipped a kernel cannot pass.
                crate::field_ops::scale_field(&g, &fk, mach, 1.0e-3, n)?;
                crate::field_ops::add_field(&g, &fk, t, mach, n)
            },
            |(t, _p, _u, psi, rho, mu, kappa, cs, mach)| {
                Ok(vec![
                    ("t", g.download(t)?),
                    ("psi", g.download(psi)?),
                    ("rho", g.download(rho)?),
                    ("mu", g.download(mu)?),
                    ("kappa", g.download(kappa)?),
                    ("c", g.download(cs)?),
                    ("mach", g.download(mach)?),
                ])
            },
        )
        .expect("SPEC-LIT 81.7: the compressible gas state must capture and \
                 replay bitwise");
        println!("  compressible thermo: {report}");
    }
}
