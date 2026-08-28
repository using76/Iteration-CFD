// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The low-Mach variable-density formulation and the energy equation -
//! `ofgpu SPEC-LIT.md` sections 25 and 26.
//!
//! Written from:
//!   R. G. Rehm, H. R. Baum, *J. Res. Natl. Bur. Stand.* 83 (1978) 297-308 -
//!     the pressure split `p = p0(t) + p~(x,t)`, the divergence constraint
//!     (S25.1) and the `p0` evolution equation (S25.2)
//!   J. Majda, J. A. Sethian, *Combust. Sci. Technol.* 42 (1985) 185 -
//!     background on the low-Mach filtering of acoustics this rests on
//!   the FDS Technical Reference Guide (McGrattan et al., NIST Special
//!     Publication 1018, public domain) - `reference/fds` was read and
//!     adapted for the SHAPE of the divergence constraint and the sealed/open
//!     `p0` bookkeeping; acknowledged here as SPEC-LIT S0 requires. No FDS
//!     *code* was copied - this module's assembly is built entirely out of
//!     ofgpu's own, already-tested `fv`/`timescheme` operators (S3, S13)
//!   S. V. Patankar, *Numerical Heat Transfer and Fluid Flow* (1980), S4.2 -
//!     the `S = S_u + S_p psi`, `S_p <= 0` linearisation the source registry
//!     hands to [`crate::fv::fvm_su`] / [`crate::fv::fvm_sp`]
//!   ofgpu `SPEC-LIT.md` S3 (the operators this assembly is built from), S9
//!     (the density-ratio buoyancy this module's gas state feeds), S13
//!     (`ddtSchemes`, S13.4's unsupported-setting contract) and S18 (the
//!     volumetric source registry pattern [`EnergySources`] specialises)
//!   Jayatilleke, *Prog. Heat Mass Transfer* 1 (1969) 193-330 - the sublayer
//!     resistance correction to the thermal log law, SPEC-LIT S29.3
//!     ([`Self::set_thermal_wall`]; the law itself and its device kernel live
//!     in `crate::wallfunctions`/`cuda/wallfunctions.cu`, this module only
//!     owns the wall-BC wiring - which faces, and where in [`Energy::correct`]
//!     the triple gets rewritten)
//! No GPL-licensed source was consulted.
//!
//! # What lives here, and what does not
//!
//! * [`GasProperties`] / [`GasState`] - S25: the ideal-gas density
//!   `rho = p0/(R_s T)` and the thermodynamic pressure `p0(t)`, sealed or
//!   open.
//! * [`EnergySources`] - S18/S26: the registry combustion (S27) and
//!   radiation (S28) push their volumetric heat sources into. This module
//!   sums what is registered; it does not know how either number was
//!   computed, which is what keeps S27/S28 out of this file entirely.
//! * [`Energy`] - S26: the temperature equation itself, `rho cp`-weighted,
//!   plus [`Energy::target_divergence`] - S25.1's `(div u)_target`, the ONE
//!   number the pressure equation needs from this module (SPEC-LIT S25.3:
//!   "SIMPLE/PISO change in ONE place").
//!
//! # Three *DESIGN* choices this v1 makes, stated up front
//!
//! **1. Momentum's own `ddt`/convection stay density-UNWEIGHTED.**
//! SPEC-LIT S25.3 gives the full compressible momentum equation
//! `rho Du/Dt = -grad p~ + (rho - rho_inf)g + ...` but then says the
//! SIMPLE/PISO loop changes "in ONE place": the pressure equation's source.
//! Taken literally, that means [`crate::momentum::Momentum`] itself needs no
//! `rho` weighting at all - only the pressure equation gains the target
//! divergence, exactly the seam `SPEC-LIT` names. That works out to be exact,
//! not approximate, in the one place it matters:
//!
//! ```text
//! rho/rho_inf = T_inf/T           (ideal gas, same p0, same W, at ANY p0(t))
//! ```
//!
//! so `(rho - rho_inf)g / rho_inf = g(T_inf/T - 1)` identically - not just as
//! `dT/T -> 0` - which is exactly [`crate::momentum::BuoyancyCoeffs`],
//! already implemented and tested. `crate::momentum::Momentum` is reused
//! completely unmodified; see `tests::buoyancy_matches_the_density_ratio_at_any_deltat`
//! below, which is `SPEC-LIT` S25.3's "show it in a test" line. The velocity
//! field's own inertia (`rho Du/Dt` proper) is therefore a leading-order
//! "anelastic" approximation in v1 - the physics S25.1's divergence
//! constraint on the PRESSURE equation captures (thermal expansion driving a
//! nonzero `div u`) is exactly what distinguishes this from the existing
//! Boussinesq-style buoyant solver (`ofgpu-buoyant`), and is not touched by
//! this simplification.
//!
//! **2. `Q` for both the S25.2 `p0` ODE and the S25.1 target-divergence field
//! is exactly what [`EnergySources`] has accumulated** - combustion's
//! `q'''_c` and radiation's `-div(q_r)`, once those modules exist and
//! register. `S25.1`'s own `Q` also names `div(k_eff grad T)`, the
//! CONDUCTION term, which this module does not fold in: doing so needs an
//! explicit divergence-of-diffusive-flux operator this crate does not
//! otherwise have. What is lost by leaving it out: nothing, for the decisive
//! S25.2 gate, because a sealed box with adiabatic walls has
//! `integral(div(k_eff grad T)) dV = 0` exactly (divergence theorem - the
//! conduction term only ever redistributes heat, and a closed, insulated
//! boundary redistributes none of it out), which is exactly the box the gate
//! tests. A case with an actual imposed wall heat flux would be missing that
//! contribution to `p0`'s ramp - flagged here rather than silently wrong, per
//! `SPEC-LIT` S13.4's own rule applied to a gap in this module rather than in
//! a case setting.
//!
//! **3. The T-equation wall function - SPEC-LIT S29.3, the Jayatilleke
//! correction - is now implemented.** `crate::field::BcKind::ThermalWallFunction`
//! is the patch type that asks for it; [`Self::set_thermal_wall`] wires the
//! faces `T`'s own patch types name (SPEC-LIT S15.5's rule, extended to a
//! fifth field) onto a [`crate::wallfunctions::ThermalWallData`], and
//! [`Self::correct`] refreshes its Robin triple every outer iteration, right
//! after [`Self::update_k_eff`] and before [`Self::assemble`] - the log-law
//! sublayer resistance this crate's high-Re wall-function meshes were
//! missing (S26's original note, quoted for history: "the convective wall
//! function for temperature (Jayatilleke-type) is deferred"). A plain
//! fixed-T or fixed-flux wall is still exactly the generic S4 Robin triple
//! every other scalar in this crate uses, with [`flux_to_grad`] doing the one
//! conversion a fixed-flux wall needs (`g_ref = q_w/k_eff`); a case that never
//! calls [`Self::set_thermal_wall`] behaves exactly as before.

use std::path::Path;

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::{GpuScalarField, GpuSurfaceScalarField};
use crate::field_ops::{self, FieldKernels};
use crate::fv::{self, DivScheme, FvKernels, GradScheme, SnGradScheme};
use crate::io::case::SolverControls;
use crate::io::contract;
use crate::io::dict::FoamDict;
use crate::io::schemes::DivEntry;
use crate::ldu::GpuLduMatrix;
use crate::ldu_ops::{self, LduKernels};
use crate::mesh::GpuMesh;
use crate::solver::{self, SolverKernels, SolverPerformance, SolverWorkspace};
use crate::timescheme::{Ddt, DdtScheme, TimeKernels};
use crate::wallfunctions::{ThermalWallData, WallFunctionCoeffs};
use crate::{Label, Scalar, Vec3};

// ==========================================================================
//  §37  The turbulent Prandtl number: constant, or Kays-Crawford
// ==========================================================================

/// Which closure supplies `Pr_t` in S26's `k_eff = k + rho cp nu_t/Pr_t` -
/// SPEC-LIT S37.
///
/// The DEFAULT is [`Self::Constant`], deliberately: every measurement this
/// project has recorded through `ofgpu-fire` was made with a single case-wide
/// `Pr_t`, and a default that changed would move all of them at once. A case
/// opts in by naming `KaysCrawford`; anything else is a S13.4 error naming
/// both spellings (see [`Self::parse`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrtModel {
    /// `Pr_t = Pr_t_inf` everywhere - [`GasProperties::pr_t`] as written.
    #[default]
    Constant,
    /// Kays-Crawford: `Pr_t` a function of the local turbulent Peclet number,
    /// rising to `2 Pr_t_inf` through the conduction sublayer - SPEC-LIT S37,
    /// [`kays_crawford_prt`].
    KaysCrawford,
}

impl PrtModel {
    /// The spelling a case file uses, and what [`Self::parse`] prints.
    pub fn name(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::KaysCrawford => "KaysCrawford",
        }
    }

    /// Every spelling a case may name, for a S13.4 menu.
    pub const NAMES: [&'static str; 2] = ["constant", "KaysCrawford"];

    /// SPEC-LIT S13.4: a recognised spelling selects the model; anything else
    /// is an error that NAMES the alternatives (`-permissive` substitutes
    /// `constant`, the default, and says so).
    ///
    /// `setting` is the dictionary entry as the user wrote it, so the message
    /// points at `physics.fluid.PrtModel` for a JSONC case and at
    /// `thermophysicalProperties/PrtModel` for an OpenFOAM one.
    pub fn parse(setting: &str, value: &str) -> Result<Self> {
        match value {
            "constant" | "Constant" => Ok(Self::Constant),
            "KaysCrawford" | "kaysCrawford" => Ok(Self::KaysCrawford),
            other => contract::unsupported(
                setting,
                other,
                &Self::NAMES,
                "constant (a single case-wide Pr_t)",
                Self::Constant,
            ),
        }
    }
}

/// Kays-Crawford's `C` - SPEC-LIT S37.1. Kays, *ASME J. Heat Transfer* 116
/// (1994) 284-295, and Kays & Crawford, *Convective Heat and Mass Transfer*,
/// 4th ed., ch. 13. Not a case setting: it is one of the two numbers that
/// define the correlation, and a case that wants a different one wants a
/// different correlation.
pub const KAYS_CRAWFORD_C: Scalar = 0.3;

/// `Pr_t(Pe_t)` - SPEC-LIT S37.1, evaluated in the rearranged form S37.2
/// derives, which is the SAME function and is the one that survives floating
/// point:
///
/// ```text
/// Pe_t = (nu_t/nu) Pr                       turbulent Peclet number
/// u    = 1/(C Pe_t sqrt(Pr_t_inf))
/// h(u) = (exp(-u) + u - 1)/u^2
/// Pr_t = Pr_t_inf / (1/2 + h(u))
/// ```
///
/// Both limits are one line in this form and are asserted as tests below:
/// `h(0) = 1/2` gives `Pr_t -> Pr_t_inf` as `Pe_t -> inf` (the free stream),
/// and `h(inf) = 0` gives `Pr_t -> 2 Pr_t_inf` as `Pe_t -> 0` (the conduction
/// sublayer).
///
/// Two branches, one at each end, both derived in S37.2 rather than tuned:
///
/// * `Pe_t` at or below the point where `2 C Pe_t sqrt(Pr_t_inf)` - the whole
///   correction to the limit - falls under [`Scalar::EPSILON`], including
///   `Pe_t = 0` exactly, returns `2 Pr_t_inf`. Without it `u` is `+inf`,
///   `u*u` is `+inf`, and `h` evaluates `inf/inf = NaN` at the one input a
///   resolved mesh's own wall face hands it.
/// * `u` small (`Pe_t` large) evaluates `h` by its Taylor series, because
///   `exp(-u) + u - 1` is a difference of numbers near 1 whose true value is
///   `u^2/2`: at `u = 1e-3` the direct form has already lost ten digits.
#[inline]
pub fn kays_crawford_prt(pe_t: Scalar, c: Scalar, pr_t_inf: Scalar) -> Scalar {
    let a = pr_t_inf.sqrt();
    let x = c * pe_t;

    // The Pe_t -> 0 branch. Written as a NOT of the positive test so a NaN
    // `pe_t` takes it too rather than propagating.
    if !(2.0 * x * a > Scalar::EPSILON) {
        return 2.0 * pr_t_inf;
    }

    let u = 1.0 / (x * a);
    let h = if u < 1e-2 {
        // h(u) = sum_{k>=0} (-u)^k/(k+2)! - the series S37.2 derives.
        // Truncated after u^4/720, whose first dropped term is u^5/5040
        // < 4e-14 of h at the switch-over point.
        0.5 - u / 6.0 + u * u / 24.0 - u * u * u / 120.0 + u * u * u * u / 720.0
    } else {
        ((-u).exp() + u - 1.0) / (u * u)
    };

    pr_t_inf / (0.5 + h)
}

// ==========================================================================
//  §25.2  Gas properties and the ideal-gas state
// ==========================================================================

/// The fixed gas properties SPEC-LIT S25 states for v1: constant molar mass
/// (air), constant `cp`, constant molecular conductivity. A temperature- or
/// composition-dependent `cp(T)` is, per SPEC-LIT S26, "a coefficient
/// change, not a structure change" - it would replace [`Self::cp`] with a
/// function without touching how [`Energy`] assembles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GasProperties {
    /// Universal gas constant, J/(mol K).
    pub r_universal: Scalar,
    /// Molar mass, kg/mol. `0.0289647` is dry air (CODATA/ICAO standard
    /// atmosphere).
    pub w: Scalar,
    /// Specific heat at constant pressure, J/(kg K).
    pub cp: Scalar,
    /// `cp/cv`, used only in the S25.1/S25.2 divergence constraint and `p0`
    /// ODE.
    pub gamma: Scalar,
    /// Molecular thermal conductivity, W/(m K).
    pub k: Scalar,
    /// Turbulent Prandtl number - S26's `k_eff = k + rho cp nu_t/Pr_t`.
    pub pr_t: Scalar,
    /// Molecular Prandtl number, `nu cp rho / k` - SPEC-LIT S29.3's
    /// Jayatilleke thermal wall function needs it apart from `Pr_t`; nothing
    /// else in this module does, because `k_eff`'s molecular half is already
    /// `k` directly. 0.71 is air at ambient conditions, the same default
    /// [`crate::scalar_transport::ScalarTransportCoeffs::pr`] carries.
    pub pr: Scalar,
    /// Which closure supplies `Pr_t` - SPEC-LIT S37. [`PrtModel::Constant`]
    /// (this struct's own [`Self::pr_t`] everywhere) unless a case names
    /// otherwise; [`PrtModel::KaysCrawford`] reads [`Self::pr_t`] as
    /// `Pr_t_inf`, the free-stream asymptote, and varies `Pr_t` between that
    /// and `2 Pr_t_inf` with the local turbulent Peclet number.
    pub pr_t_model: PrtModel,
}

impl Default for GasProperties {
    /// Air at approximately 300 K: `R_s = R/W approx 287.05 J/(kg K)`,
    /// `cp = 1006 J/(kg K)`, `gamma = 1.4`, `k = 0.026 W/(m K)`
    /// (Incropera, table A.4), `Pr_t = 0.85` (Kays 1994 - the same value
    /// [`crate::scalar_transport::ScalarTransportCoeffs`] defaults to, which
    /// is what makes the two comparable in the Boussinesq-consistency test).
    fn default() -> Self {
        Self {
            r_universal: 8.314_462_618,
            w: 0.0289647,
            cp: 1006.0,
            gamma: 1.4,
            k: 0.026,
            pr_t: 0.85,
            pr: 0.71,
            pr_t_model: PrtModel::Constant,
        }
    }
}

impl GasProperties {
    /// `R_s = R/W`, the specific gas constant SPEC-LIT S25 puts in
    /// `rho = p0/(R_s T)`.
    pub fn r_s(&self) -> Scalar {
        self.r_universal / self.w
    }

    pub fn validate(&self) -> Result<()> {
        let positive = [
            ("gasProperties/R", self.r_universal),
            ("gasProperties/W", self.w),
            ("gasProperties/Cp", self.cp),
            ("gasProperties/k", self.k),
            ("gasProperties/Prt", self.pr_t),
            ("gasProperties/Pr", self.pr),
        ];
        for (name, v) in positive {
            if !(v > 0.0) || !v.is_finite() {
                return Err(Error::Config(format!(
                    "{name} is {v}; it has to be finite and positive"
                )));
            }
        }
        if !(self.gamma > 1.0) || !self.gamma.is_finite() {
            return Err(Error::Config(format!(
                "gasProperties/gamma is {}; cp/cv > 1 for any real gas, and \
                 the S25.2 p0 ODE divides by it",
                self.gamma
            )));
        }
        Ok(())
    }

    /// Read `R`, `W` (or `M`), `Cp`, `gamma` and `k` from
    /// `constant/thermophysicalProperties` where present, leaving
    /// [`Default::default`]'s air values in place for anything the file does
    /// not name. A missing file is not an error - same convention as
    /// [`crate::momentum::BuoyancyCoeffs::from_case`].
    pub fn from_case(case_dir: &Path) -> Result<Self> {
        let mut c = Self::default();
        let p = case_dir.join("constant").join("thermophysicalProperties");
        if !p.exists() {
            return Ok(c);
        }
        let d = FoamDict::read(&p)?;
        c.r_universal = d.scalar("R", c.r_universal);
        c.w = d.scalar("W", d.scalar("M", c.w));
        c.cp = d.scalar("Cp", c.cp);
        c.gamma = d.scalar("gamma", c.gamma);
        c.k = d.scalar("k", c.k);
        c.pr_t = d.scalar("Prt", c.pr_t);
        c.pr = d.scalar("Pr", c.pr);
        // SPEC-LIT S37.4, the OpenFOAM route's own spelling. Absent keeps
        // `constant`, which is what every case written before S37 existed
        // means; a spelling this solver does not know is a S13.4 error
        // naming both, not a silent fall-back to the default.
        if let Some(w) = d.get("PrtModel") {
            c.pr_t_model = PrtModel::parse("thermophysicalProperties/PrtModel", w)?;
        }
        Ok(c)
    }
}

/// Whether the pressure `p0(t)` is free to rise (a closed compartment) or
/// pinned to the ambient value (anything with an opening to atmosphere) -
/// SPEC-LIT S25.2's two cases, stated verbatim: "sealed: Phi_b = 0" and
/// "open domain: p0 = const, dp0/dt = 0". A sealed domain that genuinely
/// vents (`Phi_b != 0`) is out of v1's scope; see [`GasState::advance_p0`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainKind {
    Sealed,
    Open,
}

/// `g_ref = q_w / k_eff` - SPEC-LIT S26's translation from an imposed wall
/// heat flux to the S4 Robin triple's `ref_grad`, for a case or test setting
/// up a fixed-flux boundary on `T`. `k_eff` should be the value AT THAT FACE;
/// with a turbulent `nu_t` at the wall this is exact only at the iteration it
/// is evaluated at, same as any other lagged effective diffusivity in this
/// crate.
pub fn flux_to_grad(q_w: Scalar, k_eff: Scalar) -> Scalar {
    q_w / k_eff
}

// ==========================================================================
//  Kernels - cuda/energy.cu
// ==========================================================================

struct EnergyKernels {
    accumulate: CudaFunction,
    k_eff: CudaFunction,
    /// SPEC-LIT S37.3 - the `k_eff` pass with a LOCAL `Pr_t`. Loaded
    /// unconditionally beside [`Self::k_eff`] (one `cuModuleGetFunction`
    /// against a module already resident) rather than lazily behind the
    /// model selection, so a case that switches models mid-life cannot
    /// discover a missing symbol at the first `correct()`.
    k_eff_kays_crawford: CudaFunction,
    target_divergence: CudaFunction,
    fixed_flux: CudaFunction,
}

impl EnergyKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::ENERGY)?;
        Ok(Self {
            accumulate: k.func("energyAccumulate")?,
            k_eff: k.func("energyKEff")?,
            k_eff_kays_crawford: k.func("energyKEffKaysCrawford")?,
            target_divergence: k.func("energyTargetDivergence")?,
            fixed_flux: k.func("energyFixedFluxTemperature")?,
        })
    }

    fn accumulate(&self, gpu: &Gpu, dst: &mut DevBuf<Scalar>, src: &DevBuf<Scalar>, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        if dst.len() < n || src.len() < n {
            return Err(Error::Config(format!(
                "energy: accumulate wants {n} elements, dst has {} and src has {}",
                dst.len(),
                src.len()
            )));
        }
        let nl = n as Label;
        unsafe {
            gpu.stream()
                .launch_builder(&self.accumulate)
                .arg(&mut *dst)
                .arg(src)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn k_eff(
        &self,
        gpu: &Gpu,
        dst: &mut DevBuf<Scalar>,
        rho_f: &DevBuf<Scalar>,
        nut_f: &DevBuf<Scalar>,
        k_mol: Scalar,
        cp_over_prt: Scalar,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let nl = n as Label;
        unsafe {
            gpu.stream()
                .launch_builder(&self.k_eff)
                .arg(&mut *dst)
                .arg(rho_f)
                .arg(nut_f)
                .arg(&k_mol)
                .arg(&cp_over_prt)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// SPEC-LIT S37.3's `k_eff` pass: the same face loop as [`Self::k_eff`],
    /// with `Pr_t` evaluated per face from the local turbulent Peclet number
    /// instead of divided into `cp` once on the host.
    #[allow(clippy::too_many_arguments)]
    fn k_eff_kays_crawford(
        &self,
        gpu: &Gpu,
        dst: &mut DevBuf<Scalar>,
        rho_f: &DevBuf<Scalar>,
        nut_f: &DevBuf<Scalar>,
        k_mol: Scalar,
        cp: Scalar,
        nu: Scalar,
        pr: Scalar,
        pr_t_inf: Scalar,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let nl = n as Label;
        let c = KAYS_CRAWFORD_C;
        let eps = Scalar::EPSILON;
        unsafe {
            gpu.stream()
                .launch_builder(&self.k_eff_kays_crawford)
                .arg(&mut *dst)
                .arg(rho_f)
                .arg(nut_f)
                .arg(&k_mol)
                .arg(&cp)
                .arg(&nu)
                .arg(&pr)
                .arg(&pr_t_inf)
                .arg(&c)
                .arg(&eps)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn target_divergence(
        &self,
        gpu: &Gpu,
        dst: &mut DevBuf<Scalar>,
        q: &DevBuf<Scalar>,
        rho: &DevBuf<Scalar>,
        t: &DevBuf<Scalar>,
        cp: Scalar,
        inv_gamma_p0: Scalar,
        dp0dt: Scalar,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let nl = n as Label;
        unsafe {
            gpu.stream()
                .launch_builder(&self.target_divergence)
                .arg(&mut *dst)
                .arg(q)
                .arg(rho)
                .arg(t)
                .arg(&cp)
                .arg(&inv_gamma_p0)
                .arg(&dp0dt)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// SPEC-LIT §32.2's fixed-flux rewrite, on the `n` faces `face` names.
    fn fixed_flux(
        &self,
        gpu: &Gpu,
        fr: &mut DevBuf<Scalar>,
        ref_grad: &mut DevBuf<Scalar>,
        ref_value: &DevBuf<Scalar>,
        k_eff_wall: &DevBuf<Scalar>,
        face: &DevBuf<Label>,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let nl = n as Label;
        unsafe {
            gpu.stream()
                .launch_builder(&self.fixed_flux)
                .arg(fr)
                .arg(ref_grad)
                .arg(ref_value)
                .arg(k_eff_wall)
                .arg(face)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }
}

// ==========================================================================
//  §25  Gas state - rho(p0, T) and the p0(t) ODE
// ==========================================================================

/// `rho = p0/(R_s T)` on the device, and the thermodynamic pressure `p0(t)`
/// on the host - SPEC-LIT S25. `rho` carries its own old time levels so that
/// [`Energy`]'s `rho*cp`-weighted `ddt` (S26) has `rho^{n-1}`, `rho^{n-2}`
/// available whichever time scheme is running, the same way every other
/// field in this crate does (S13.3).
pub struct GasState<'m> {
    m: &'m GpuMesh,
    props: GasProperties,
    domain: DomainKind,

    p0: Scalar,
    /// `p0^{n-1}`, `p0^{n-2}` - rotated by [`Self::advance_time_levels`],
    /// once per time step, in step with every field's `f00 <- f0 <- f`.
    p0_0: Scalar,
    p0_00: Scalar,
    dp0dt: Scalar,

    rho: GpuScalarField,
    fldk: FieldKernels,
}

impl<'m> GasState<'m> {
    pub fn new(
        gpu: &Gpu,
        m: &'m GpuMesh,
        props: GasProperties,
        domain: DomainKind,
        p0_init: Scalar,
    ) -> Result<Self> {
        props.validate()?;
        if !(p0_init > 0.0) || !p0_init.is_finite() {
            return Err(Error::Config(format!(
                "p0 initial value is {p0_init}; the ideal gas law needs a \
                 positive absolute pressure"
            )));
        }

        Ok(Self {
            m,
            props,
            domain,
            p0: p0_init,
            p0_0: p0_init,
            p0_00: p0_init,
            dp0dt: 0.0,
            rho: GpuScalarField::zeros(gpu, m, "rho")?,
            fldk: FieldKernels::new(gpu)?,
        })
    }

    pub fn props(&self) -> &GasProperties {
        &self.props
    }

    pub fn domain(&self) -> DomainKind {
        self.domain
    }

    pub fn p0(&self) -> Scalar {
        self.p0
    }

    pub fn dp0dt(&self) -> Scalar {
        self.dp0dt
    }

    /// Restore `dp0dt` directly - SPEC-LIT §31.2's restart requirement of
    /// substance for `ofgpu-fire`: [`Energy::update_target_divergence`]
    /// reads `dp0dt` at a ONE-ITERATION LAG (the value [`Self::advance_p0`]
    /// left behind at the end of the previous unit of work), exactly the
    /// segregated lag every other coupling coefficient in that driver
    /// already runs at. A `GasState` rebuilt fresh from a checkpoint's `p0`
    /// alone starts with `dp0dt = 0`, which is correct on a cold start and
    /// wrong on a restart of a sealed (§25.2) case with an ongoing heat
    /// release - the FIRST pressure solve after resuming would then
    /// assemble the low-Mach target divergence missing the
    /// `-dp0dt/(gamma p0)` term the continuous run's own next step carried.
    /// Does not touch `p0` itself or either old time level.
    pub fn set_dp0dt(&mut self, v: Scalar) {
        self.dp0dt = v;
    }

    pub fn rho(&self) -> &GpuScalarField {
        &self.rho
    }

    /// `rho(T)` at the CURRENT `p0`, on the host - exactly the formula
    /// [`Self::update_density`] evaluates on the device, exposed so a test or
    /// a diagnostic can compute it without a round trip. This is what
    /// `tests::buoyancy_matches_the_density_ratio_at_any_deltat` checks
    /// against [`crate::momentum::BuoyancyCoeffs`].
    pub fn rho_at(&self, t: Scalar) -> Scalar {
        self.p0 / (self.props.r_s() * t)
    }

    /// Refresh `rho` (and its two old time levels) from `T` (and ITS two old
    /// time levels) at the `p0` each level held - SPEC-LIT S25: "rho lives on
    /// the device; update each outer iteration."
    ///
    /// Uses only [`field_ops::set_field`] and [`field_ops::divide_field`],
    /// which is why there is no `energyGasDensity` kernel in `cuda/energy.cu`
    /// - two existing, already-tested launches per level cost less than a new
    /// one to save them.
    pub fn update_density(&mut self, gpu: &Gpu, t: &GpuScalarField) -> Result<()> {
        let n = self.m.n_cells;
        let nbf = self.m.n_boundary_faces;
        let rs = self.props.r_s();

        field_ops::set_field(gpu, &self.fldk, &mut self.rho.f, self.p0 / rs, n)?;
        field_ops::divide_field(gpu, &self.fldk, &mut self.rho.f, &t.f, n)?;

        field_ops::set_field(gpu, &self.fldk, &mut self.rho.bf, self.p0 / rs, nbf)?;
        field_ops::divide_field(gpu, &self.fldk, &mut self.rho.bf, &t.bf, nbf)?;

        field_ops::set_field(gpu, &self.fldk, &mut self.rho.f0, self.p0_0 / rs, n)?;
        field_ops::divide_field(gpu, &self.fldk, &mut self.rho.f0, &t.f0, n)?;

        field_ops::set_field(gpu, &self.fldk, &mut self.rho.f00, self.p0_00 / rs, n)?;
        field_ops::divide_field(gpu, &self.fldk, &mut self.rho.f00, &t.f00, n)?;

        Ok(())
    }

    /// SPEC-LIT S25.2: advance `p0` by one explicit-Euler step of
    ///
    /// ```text
    /// dp0/dt = (gamma/V_dom) * ((gamma-1)/gamma * integral(Q) dV)     sealed
    /// dp0/dt = 0                                                       open
    /// ```
    ///
    /// `total_q` is `integral(Q) dV` over the whole domain, in watts -
    /// [`EnergySources::total_q`] computes exactly that from whatever
    /// combustion and radiation registered this iteration (see the module
    /// doc's second *DESIGN* note for what `Q` does and does not include
    /// here). `Phi_b` is taken to be exactly zero for `Sealed`, per
    /// SPEC-LIT S25.2's own parenthetical - a vented sealed compartment is
    /// out of v1's scope.
    ///
    /// Returns the new `p0`. A non-positive result is refused rather than
    /// returned: it means the ODE has been driven somewhere the ideal gas
    /// law no longer makes sense, almost always a `dt` far larger than the
    /// heat input can be integrated explicitly at.
    pub fn advance_p0(&mut self, total_q: Scalar, dt: Scalar) -> Result<Scalar> {
        if !(dt >= 0.0) || !dt.is_finite() {
            return Err(Error::Config(format!(
                "GasState::advance_p0: dt is {dt}"
            )));
        }
        if !total_q.is_finite() {
            return Err(Error::Config(format!(
                "GasState::advance_p0: total_q (integral of Q over the \
                 domain) is {total_q}"
            )));
        }

        match self.domain {
            DomainKind::Open => {
                self.dp0dt = 0.0;
            }
            DomainKind::Sealed => {
                let gamma = self.props.gamma;
                let v_dom = self.m.total_volume;
                if !(v_dom > 0.0) {
                    return Err(Error::Config(
                        "GasState::advance_p0: the mesh's total volume is \
                         zero or negative"
                            .to_string(),
                    ));
                }
                self.dp0dt = (gamma / v_dom) * ((gamma - 1.0) / gamma * total_q);
                self.p0 += self.dp0dt * dt;
                if !(self.p0 > 0.0) || !self.p0.is_finite() {
                    return Err(Error::Config(format!(
                        "GasState::advance_p0: p0 advanced to {}, which is \
                         not a usable absolute pressure - dt is probably too \
                         large for an explicit step of this heat input",
                        self.p0
                    )));
                }
            }
        }
        Ok(self.p0)
    }

    /// Rotate `p0^{n-2} <- p0^{n-1} <- p0`, in that order - SPEC-LIT S13.3,
    /// applied to the one scalar this module's `ddt` needs it for. Call ONCE
    /// per time step, next to [`field_ops::advance_time_levels`] and
    /// [`crate::timescheme::Ddt::advance`] - the same event, same rule about
    /// not calling it once per outer corrector (SPEC-LIT S13.3).
    pub fn advance_time_levels(&mut self) {
        self.p0_00 = self.p0_0;
        self.p0_0 = self.p0;
    }

    /// Seed both old levels of `p0` at the current value - the scalar
    /// equivalent of [`field_ops::seed_old_time`], for the start of a run.
    pub fn seed_time_levels(&mut self) {
        self.p0_0 = self.p0;
        self.p0_00 = self.p0;
    }
}

// ==========================================================================
//  §18 / §26  The source registry
// ==========================================================================

/// The volumetric sources on the energy equation - SPEC-LIT S18 specialised
/// to S26's units (W/m3 explicit, W/(m3 K) implicit).
///
/// Combustion (S27) and radiation (S28) push their contribution in every
/// outer iteration; this struct sums them and knows nothing else about
/// either - which is the whole point (the module doc's hook that keeps S27
/// and S28 out of this file).
pub struct EnergySources {
    /// `[n_cells]` W/m3, explicit - `q'''_c`, `-div(q_r)`, ...
    q: DevBuf<Scalar>,
    /// `[n_cells]` W/(m3 K), implicit sink, `<= 0` per cell (Patankar S4.2).
    /// The caller guarantees the sign; nothing here can check it without a
    /// device-side reduction on every registration, which would turn a
    /// cheap accumulate into a synchronising one.
    sp: DevBuf<Scalar>,

    fldk: FieldKernels,
    ek: EnergyKernels,
    solk: SolverKernels,
    dot_out: DevBuf<Scalar>,
    /// Scratch for the two-stage reduction - SPEC-LIT `MAX_REDUCE_BLOCKS`'s
    /// value in `src/solver.rs`, duplicated here because that constant is
    /// private to its module and this is the only other reduction in the
    /// crate that does not already own a [`crate::solver::SolverWorkspace`].
    partials: DevBuf<Scalar>,

    n: usize,
}

const REDUCE_PARTIALS: usize = 1024;

impl EnergySources {
    pub fn new(gpu: &Gpu, m: &GpuMesh) -> Result<Self> {
        let n = m.n_cells;
        let one = |k: usize| k.max(1);
        Ok(Self {
            q: gpu.zeros(one(n))?,
            sp: gpu.zeros(one(n))?,
            fldk: FieldKernels::new(gpu)?,
            ek: EnergyKernels::new(gpu)?,
            solk: SolverKernels::new(gpu)?,
            dot_out: gpu.zeros(1)?,
            partials: gpu.zeros(REDUCE_PARTIALS)?,
            n,
        })
    }

    /// Zero both accumulators. Call once at the top of every outer iteration,
    /// before combustion and radiation register - otherwise a source from two
    /// iterations ago is still being added in.
    pub fn clear(&mut self, gpu: &Gpu) -> Result<()> {
        field_ops::set_field(gpu, &self.fldk, &mut self.q, 0.0, self.n)?;
        field_ops::set_field(gpu, &self.fldk, &mut self.sp, 0.0, self.n)
    }

    fn check_len(&self, contribution: &DevBuf<Scalar>, who: &str) -> Result<()> {
        if contribution.len() < self.n {
            return Err(Error::Config(format!(
                "EnergySources::{who}: contribution has {} elements, the \
                 mesh has {} cells",
                contribution.len(),
                self.n
            )));
        }
        Ok(())
    }

    /// `q += contribution`, W/m3 - a combustion heat-release rate, a
    /// radiative source, a heater's `Q_dot/V`, or anything else that is a
    /// volumetric power density with a definite sign.
    pub fn register_explicit(&mut self, gpu: &Gpu, contribution: &DevBuf<Scalar>) -> Result<()> {
        self.check_len(contribution, "register_explicit")?;
        self.ek.accumulate(gpu, &mut self.q, contribution, self.n)
    }

    /// `sp += contribution`, W/(m3 K). The caller's `contribution` must be
    /// `<= 0` everywhere (Patankar S4.2 / SPEC-LIT S3.4, S18) - a positive
    /// entry would make the energy matrix's diagonal less dominant instead of
    /// more, which [`crate::fv::fvm_sp`] documents and does not itself guard
    /// against either.
    pub fn register_implicit_sink(&mut self, gpu: &Gpu, contribution: &DevBuf<Scalar>) -> Result<()> {
        self.check_len(contribution, "register_implicit_sink")?;
        self.ek.accumulate(gpu, &mut self.sp, contribution, self.n)
    }

    pub fn q(&self) -> &DevBuf<Scalar> {
        &self.q
    }

    pub fn sp(&self) -> &DevBuf<Scalar> {
        &self.sp
    }

    /// `integral(Q) dV` over the whole domain, in watts - SPEC-LIT S25.2's
    /// `integral(Q) dV` and exactly what [`GasState::advance_p0`] wants for
    /// `total_q`. One reduction, one 8-byte transfer.
    pub fn total_q(&mut self, gpu: &Gpu, m: &GpuMesh) -> Result<Scalar> {
        if self.n == 0 {
            return Ok(0.0);
        }
        solver::device_dot(gpu, &self.solk, &mut self.dot_out, &self.q, &m.v, &mut self.partials, self.n)?;
        Ok(gpu.download(&self.dot_out)?[0])
    }
}

// ==========================================================================
//  §26  Controls
// ==========================================================================

/// Everything the energy equation reads out of a case - the S26 analogue of
/// [`crate::momentum::MomentumControls`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyControls {
    pub t_solver: SolverControls,
    pub t_relax: Scalar,

    /// This field's own `divSchemes` entry, `div(phi,T)`. Unlike every other
    /// equation in this crate, `bounded` is not read from it: SPEC-LIT S26
    /// makes the bounded correction PHYSICS for a variable-density flow, not
    /// an optional stabiliser, so [`Energy`] applies it unconditionally
    /// whatever this entry says (see the module doc).
    pub div_scheme: DivEntry,

    pub grad_scheme: GradScheme,
    pub sn_grad: SnGradScheme,
    pub n_non_orth_correctors: usize,

    /// `ddtSchemes` for `T`. Reconciled against what a `rho*cp`-weighted ddt
    /// can actually do in [`Energy::new`] - see its doc.
    pub ddt: DdtScheme,
    pub steady: bool,
    pub delta_t: Scalar,
}

impl Default for EnergyControls {
    fn default() -> Self {
        Self {
            t_solver: SolverControls::default(),
            t_relax: 0.7,
            div_scheme: DivEntry::default(),
            grad_scheme: GradScheme::GAUSS,
            sn_grad: SnGradScheme::Corrected,
            n_non_orth_correctors: 0,
            ddt: DdtScheme::SteadyState,
            steady: true,
            delta_t: 1.0,
        }
    }
}

impl EnergyControls {
    pub fn r_delta_t(&self) -> Scalar {
        if self.steady {
            0.0
        } else {
            1.0 / self.delta_t
        }
    }

    fn validate(&self) -> Result<()> {
        if !(self.t_relax > 0.0 && self.t_relax <= 1.0) {
            return Err(Error::Config(format!(
                "relaxationFactors/equations/T is {}; implicit \
                 under-relaxation needs 0 < alpha <= 1 (SPEC-LIT S5.2)",
                self.t_relax
            )));
        }
        if !self.steady && !(self.delta_t > 0.0) {
            return Err(Error::Config(format!(
                "deltaT is {} but the energy equation is transient",
                self.delta_t
            )));
        }
        Ok(())
    }
}

/// SPEC-LIT S13.4 applied to the one restriction this module's `rho*cp`
/// ddt has that a plain scalar's does not: there is no rho-weighted
/// local-step kernel (`localEuler`, S13.2) and no rho-weighted theta method
/// (`CrankNicolson`, S13.1) yet. `steadyState`, `Euler` and `backward` (at
/// any step ratio, via [`crate::timescheme::fvm_ddt_rho`]) are unaffected.
fn reconcile_ddt(scheme: DdtScheme) -> Result<DdtScheme> {
    match scheme {
        DdtScheme::SteadyState | DdtScheme::Euler | DdtScheme::Backward => Ok(scheme),
        DdtScheme::LocalEuler => contract::unsupported_note(
            "ddtSchemes/T",
            "localEuler",
            &["steadyState", "Euler", "backward"],
            "the energy equation is rho*cp-weighted (SPEC-LIT S26) and \
             crate::timescheme::fvm_ddt_local has no rho-weighted \
             counterpart; localEuler is available on U, k and epsilon/omega \
             but not yet on T",
            "Euler",
            DdtScheme::Euler,
        ),
        DdtScheme::CrankNicolson(_) => contract::unsupported_note(
            "ddtSchemes/T",
            "CrankNicolson",
            &["steadyState", "Euler", "backward"],
            "the theta method (SPEC-LIT S13.1) has to scale the spatial \
             operator before the boundary fold and be relaxed afterwards \
             (crate::timescheme::apply_theta); this module does not wire \
             that up for a rho*cp-weighted equation yet",
            "Euler",
            DdtScheme::Euler,
        ),
    }
}

// ==========================================================================
//  §26  The energy equation
// ==========================================================================

/// The temperature equation, `rho cp`-weighted, resident on the device -
/// SPEC-LIT S26:
///
/// ```text
/// rho cp [ ddt(T) + div(phi_m, T) - T*div(u) ] = laplacian(k_eff, T)
///                                                 + q'''_c - div(q_r) + dp0/dt
/// k_eff = k + rho cp nu_t / Pr_t
/// phi_m = cp * rho_f * phi                (the mass flux, scaled by cp so
///                                           ddt and convection carry the
///                                           SAME weight, S26)
/// ```
///
/// Also exposes [`Energy::target_divergence`], S25.1's `(div u)_target`,
/// which is the one number [`crate::simple::Simple`]'s pressure equation
/// needs from this module.
pub struct Energy<'m> {
    m: &'m GpuMesh,
    ctrl: EnergyControls,
    props: GasProperties,

    fvk: FvKernels,
    fldk: FieldKernels,
    lduk: LduKernels,
    solk: SolverKernels,
    tsk: TimeKernels,
    ek: EnergyKernels,

    ddt: Ddt,

    t: GpuScalarField,
    sources: EnergySources,

    a: GpuLduMatrix,
    ws: SolverWorkspace,

    /// `rho*cp` at the three time levels `ddt` needs - refreshed from
    /// [`GasState::rho`] at the top of every [`Energy::correct`].
    rho_cp: DevBuf<Scalar>,
    rho_cp0: DevBuf<Scalar>,
    rho_cp00: DevBuf<Scalar>,

    rho_face: GpuSurfaceScalarField,
    nut_face: GpuSurfaceScalarField,
    k_eff_face: GpuSurfaceScalarField,
    /// `k_eff_f * |Sf|` - the laplacian's coefficient.
    k_eff_mag_sf: GpuSurfaceScalarField,

    /// `cp * rho_f * phi` - the mass flux this equation convects with.
    phi_conv: GpuSurfaceScalarField,

    w: DevBuf<Scalar>,
    bw: DevBuf<Scalar>,
    grad_t: DevBuf<Vec3>,

    dp0dt_su: DevBuf<Scalar>,
    target_div: DevBuf<Scalar>,

    /// `kappa`/`E`/`C_mu` for [`Self::twd`] - SPEC-LIT S15.6: the same
    /// constants a case's momentum wall function uses, not a second set.
    /// [`WallFunctionCoeffs::default`] until [`Self::set_thermal_wall`] says
    /// otherwise.
    wall: WallFunctionCoeffs,
    /// The Jayatilleke thermal wall function's faces - SPEC-LIT S29.3.
    /// `None` until [`Self::set_thermal_wall`] is called, which is exactly
    /// what a case with no `thermalWallFunction` patch wants: every
    /// [`ThermalWallData`] launcher is skipped by [`Self::update_thermal_wall`]
    /// rather than run over zero faces.
    twd: Option<ThermalWallData>,

    /// SPEC-LIT §32.2's fixed wall heat flux faces - `T`'s own
    /// `fixedFluxTemperature` patch type, set by [`Self::set_fixed_flux_walls`].
    /// `None` until then, same "nothing to do" convention as [`Self::twd`].
    ffq_faces: Option<DevBuf<Label>>,
    ffq_n: usize,
}

impl<'m> Energy<'m> {
    pub fn new(gpu: &Gpu, m: &'m GpuMesh, ctrl: EnergyControls, props: GasProperties) -> Result<Self> {
        ctrl.validate()?;
        props.validate()?;

        let n = m.n_cells;
        let nif = m.n_internal_faces;
        let nbf = m.n_boundary_faces;
        let one = |k: usize| k.max(1);

        let scheme = reconcile_ddt(ctrl.ddt.reconciled(ctrl.steady))?;

        Ok(Self {
            m,
            ctrl,
            props,

            fvk: FvKernels::new(gpu)?,
            fldk: FieldKernels::new(gpu)?,
            lduk: LduKernels::new(gpu)?,
            solk: SolverKernels::new(gpu)?,
            tsk: TimeKernels::new(gpu)?,
            ek: EnergyKernels::new(gpu)?,

            ddt: Ddt::new(gpu, m, scheme, ctrl.delta_t, crate::timescheme::LtsControls::default())?,

            t: GpuScalarField::zeros(gpu, m, "T")?,
            sources: EnergySources::new(gpu, m)?,

            a: GpuLduMatrix::new(gpu, m)?,
            ws: SolverWorkspace::for_mesh(gpu, m)?,

            rho_cp: gpu.zeros(one(n))?,
            rho_cp0: gpu.zeros(one(n))?,
            rho_cp00: gpu.zeros(one(n))?,

            rho_face: GpuSurfaceScalarField::zeros(gpu, m, "rhoTf")?,
            nut_face: GpuSurfaceScalarField::zeros(gpu, m, "nutTf")?,
            k_eff_face: GpuSurfaceScalarField::zeros(gpu, m, "kEfff")?,
            k_eff_mag_sf: GpuSurfaceScalarField::zeros(gpu, m, "kEffMagSf")?,

            phi_conv: GpuSurfaceScalarField::zeros(gpu, m, "phiEnergy")?,

            w: gpu.zeros(one(nif))?,
            bw: gpu.zeros(one(nbf))?,
            grad_t: gpu.zeros(one(n))?,

            dp0dt_su: gpu.zeros(one(n))?,
            target_div: gpu.zeros(one(n))?,

            wall: WallFunctionCoeffs::default(),
            twd: None,
            ffq_faces: None,
            ffq_n: 0,
        })
    }

    // ---- accessors ---------------------------------------------------

    pub fn controls(&self) -> &EnergyControls {
        &self.ctrl
    }

    pub fn props(&self) -> &GasProperties {
        &self.props
    }

    pub fn field(&self) -> &GpuScalarField {
        &self.t
    }

    pub fn field_mut(&mut self) -> &mut GpuScalarField {
        &mut self.t
    }

    pub fn sources_mut(&mut self) -> &mut EnergySources {
        &mut self.sources
    }

    /// [`Self::field`] and [`Self::sources_mut`] at once - the split borrow
    /// neither can give alone through `&mut Energy`. Exists for a caller
    /// (combustion/radiation, SPEC-LIT S27/S28) that needs to read `T` WHILE
    /// registering onto `sources`, e.g. radiation's Marshak wall stamp reads
    /// `T`'s boundary field in the same call that registers its energy
    /// coupling.
    pub fn field_and_sources_mut(&mut self) -> (&GpuScalarField, &mut EnergySources) {
        (&self.t, &mut self.sources)
    }

    pub fn sources(&self) -> &EnergySources {
        &self.sources
    }

    pub fn matrix(&self) -> &GpuLduMatrix {
        &self.a
    }

    /// SPEC-LIT S25.1's `(div u)_target`, as of the last
    /// [`Energy::update_target_divergence`] - what
    /// [`crate::simple::Simple`]'s pressure equation subtracts.
    pub fn target_divergence(&self) -> &DevBuf<Scalar> {
        &self.target_div
    }

    /// Use this field's own `divSchemes` entry - SPEC-LIT S11.7.
    pub fn set_convection(&mut self, conv: DivEntry) {
        self.ctrl.div_scheme = conv;
    }

    /// Wire the Jayatilleke thermal wall function (SPEC-LIT S29.3) onto
    /// whichever faces `T`'s OWN patch type asked for -
    /// `crate::field::BcKind::ThermalWallFunction` - the S15.5 rule this
    /// crate already applies to `nut`/`epsilon`/`omega` extended to `T`:
    /// `faces[bf]` must come from `T`'s own field file
    /// (`crate::field_setup::faces_where` with
    /// `crate::field::BcKind::is_thermal_wall_function`), never derived from
    /// another field's.
    ///
    /// `wall` is the case's `kappa`/`E`/`C_mu` - SPEC-LIT S15.6, the SAME
    /// triple the momentum wall function reads, not a second copy a case
    /// could override independently by accident.
    ///
    /// Call once, after [`Self::new`] and before the first [`Self::correct`].
    /// A case with no `thermalWallFunction` patch on `T` never needs to call
    /// this at all - [`Self::correct`] skips the update entirely while
    /// [`Self::twd`] is `None`, and every field this module already set up
    /// (a plain fixed-T or fixed-flux wall, S26's original behaviour) is
    /// untouched.
    pub fn set_thermal_wall(&mut self, gpu: &Gpu, wall: WallFunctionCoeffs, faces: &[bool]) -> Result<()> {
        if faces.len() != self.m.n_boundary_faces {
            return Err(Error::Config(format!(
                "Energy::set_thermal_wall: {} face flags, the mesh has {} \
                 boundary faces",
                faces.len(),
                self.m.n_boundary_faces
            )));
        }
        self.wall = wall;
        self.twd = Some(ThermalWallData::build(gpu, faces)?);
        Ok(())
    }

    /// Wire SPEC-LIT §32.2's fixed wall heat flux onto whichever faces `T`'s
    /// OWN patch type named `fixedFluxTemperature` -
    /// `crate::field::BcKind::is_fixed_flux_temperature`, the same S15.5
    /// discipline [`Self::set_thermal_wall`] follows for
    /// `ThermalWallFunction`. Unlike that condition, this one needs no
    /// `WallFunctionCoeffs` at all - see `BcKind::FixedFluxTemperature`'s own
    /// doc for why `flux_to_grad` needs nothing but the current `k_eff_wall`.
    ///
    /// Call once, after [`Self::new`] and before the first [`Self::correct`].
    /// A case with no `fixedFluxTemperature` patch never needs to call this -
    /// [`Self::correct`] skips the update entirely while this list is empty.
    pub fn set_fixed_flux_walls(&mut self, gpu: &Gpu, faces: &[bool]) -> Result<()> {
        if faces.len() != self.m.n_boundary_faces {
            return Err(Error::Config(format!(
                "Energy::set_fixed_flux_walls: {} face flags, the mesh has {} \
                 boundary faces",
                faces.len(),
                self.m.n_boundary_faces
            )));
        }
        let list: Vec<Label> = faces
            .iter()
            .enumerate()
            .filter(|(_, on)| **on)
            .map(|(bf, _)| bf as Label)
            .collect();
        self.ffq_n = list.len();
        // Same zero-length convention as `ThermalWallData::build`: a padded
        // one-element buffer that `update_fixed_flux` never reads, because it
        // returns early on `ffq_n == 0`.
        let padded = if list.is_empty() { vec![0 as Label] } else { list };
        self.ffq_faces = Some(gpu.upload(&padded)?);
        Ok(())
    }

    /// Advance the ddt scheme's own time-step bookkeeping - call ONCE per
    /// time step, alongside [`GasState::advance_time_levels`] and
    /// [`field_ops::advance_time_levels`] on `T` itself.
    pub fn advance_time_step(&mut self, next_dt: Scalar) {
        self.ddt.advance(next_dt);
    }

    /// Evaluate the boundary faces and seed both old time levels from the
    /// initial field - call once, before the first [`GasState::update_density`].
    pub fn initialise(&mut self, gpu: &Gpu) -> Result<()> {
        field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut self.t, self.m)?;
        field_ops::seed_old_time(gpu, &self.fldk, &mut self.t)
    }

    // ---- assembly pieces -----------------------------------------------

    fn refresh_rho_cp(&mut self, gpu: &Gpu, gas: &GasState) -> Result<()> {
        let n = self.m.n_cells;
        let cp = self.props.cp;

        field_ops::copy_field(gpu, &self.fldk, &mut self.rho_cp, &gas.rho().f, n)?;
        field_ops::scale_field(gpu, &self.fldk, &mut self.rho_cp, cp, n)?;

        field_ops::copy_field(gpu, &self.fldk, &mut self.rho_cp0, &gas.rho().f0, n)?;
        field_ops::scale_field(gpu, &self.fldk, &mut self.rho_cp0, cp, n)?;

        field_ops::copy_field(gpu, &self.fldk, &mut self.rho_cp00, &gas.rho().f00, n)?;
        field_ops::scale_field(gpu, &self.fldk, &mut self.rho_cp00, cp, n)
    }

    /// `k_eff` on every face (S26), from the CURRENT `gas.rho()` interpolated
    /// to faces by linear interpolation (S25.3's DESIGN note: "rho_f by
    /// linear interpolation of cell rho" - the same convention this crate
    /// already uses for every other face diffusivity).
    fn update_k_eff(
        &mut self,
        gpu: &Gpu,
        nut: &GpuScalarField,
        gas: &GasState,
        nu: Scalar,
    ) -> Result<()> {
        let m = self.m;
        fv::interpolate_linear(gpu, &self.fvk, &mut self.rho_face, gas.rho(), m)?;
        fv::interpolate_linear(gpu, &self.fvk, &mut self.nut_face, nut, m)?;

        // SPEC-LIT S37.3. The two branches differ in nothing but where `Pr_t`
        // comes from: one number divided into `cp` once on the host, or one
        // evaluation of the correlation per face on the device. `Constant` is
        // the default and is bit-for-bit the pass this module has always run.
        match self.props.pr_t_model {
            PrtModel::Constant => {
                let cp_over_prt = self.props.cp / self.props.pr_t;
                self.ek.k_eff(
                    gpu,
                    &mut self.k_eff_face.f,
                    &self.rho_face.f,
                    &self.nut_face.f,
                    self.props.k,
                    cp_over_prt,
                    m.n_internal_faces,
                )?;
                self.ek.k_eff(
                    gpu,
                    &mut self.k_eff_face.bf,
                    &self.rho_face.bf,
                    &self.nut_face.bf,
                    self.props.k,
                    cp_over_prt,
                    m.n_boundary_faces,
                )?;
            }
            PrtModel::KaysCrawford => {
                self.ek.k_eff_kays_crawford(
                    gpu,
                    &mut self.k_eff_face.f,
                    &self.rho_face.f,
                    &self.nut_face.f,
                    self.props.k,
                    self.props.cp,
                    nu,
                    self.props.pr,
                    self.props.pr_t,
                    m.n_internal_faces,
                )?;
                self.ek.k_eff_kays_crawford(
                    gpu,
                    &mut self.k_eff_face.bf,
                    &self.rho_face.bf,
                    &self.nut_face.bf,
                    self.props.k,
                    self.props.cp,
                    nu,
                    self.props.pr,
                    self.props.pr_t,
                    m.n_boundary_faces,
                )?;
            }
        }

        field_ops::copy_field(gpu, &self.fldk, &mut self.k_eff_mag_sf.f, &self.k_eff_face.f, m.n_internal_faces)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.k_eff_mag_sf.f, &m.mag_sf, m.n_internal_faces)?;

        field_ops::copy_field(gpu, &self.fldk, &mut self.k_eff_mag_sf.bf, &self.k_eff_face.bf, m.n_boundary_faces)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.k_eff_mag_sf.bf, &m.b_mag_sf, m.n_boundary_faces)
    }

    /// SPEC-LIT S29.3: rewrite `T`'s Robin triple on every
    /// `thermalWallFunction` face - a no-op while [`Self::twd`] is `None`
    /// (`set_thermal_wall` was never called). Reads `self.k_eff_face.bf`, so
    /// it MUST run after [`Self::update_k_eff`] and before
    /// [`Self::assemble`]; `k` is the turbulence kinetic energy's cell field
    /// and `rho` its density, both at the SAME lag `nut` itself carries into
    /// this equation - the standard segregated coupling every other
    /// coefficient here already runs at.
    ///
    /// `k_min` is [`crate::io::case::TurbulenceControls::k_min`]'s own
    /// default (`1e-15`) rather than a case override reaching in here: unlike
    /// `kappa`/`E`/`C_mu` (SPEC-LIT S15.6), it is a floor against `sqrt(0)`,
    /// not a physical constant the model and the wall treatment could
    /// disagree about.
    fn update_thermal_wall(&mut self, gpu: &Gpu, k: &DevBuf<Scalar>, rho: &DevBuf<Scalar>, nu: Scalar) -> Result<()> {
        const K_MIN: Scalar = 1e-15;

        let Some(twd) = &self.twd else {
            return Ok(());
        };

        twd.update(
            gpu,
            &mut self.t.fr,
            &mut self.t.ref_grad,
            &self.t.ref_value,
            &self.t.f,
            k,
            rho,
            &self.k_eff_face.bf,
            self.m,
            &self.wall,
            nu,
            self.props.cp,
            self.props.pr,
            self.props.pr_t,
            K_MIN,
        )
    }

    /// SPEC-LIT §32.2: rewrite `T`'s Robin triple on every
    /// `fixedFluxTemperature` face to `ref_grad = q/k_eff_wall` with the
    /// CURRENT `k_eff_wall` - a no-op while [`Self::ffq_faces`] is `None`
    /// (`set_fixed_flux_walls` was never called). Reads `self.k_eff_face.bf`,
    /// so it MUST run after [`Self::update_k_eff`] and before
    /// [`Self::assemble`], exactly like [`Self::update_thermal_wall`].
    fn update_fixed_flux(&mut self, gpu: &Gpu) -> Result<()> {
        let Some(face) = &self.ffq_faces else {
            return Ok(());
        };
        self.ek.fixed_flux(
            gpu,
            &mut self.t.fr,
            &mut self.t.ref_grad,
            &self.t.ref_value,
            &self.k_eff_face.bf,
            face,
            self.ffq_n,
        )
    }

    /// `phi_conv = cp * rho_f * phi` - S26's mass flux, reusing the
    /// `rho_face` [`Self::update_k_eff`] just built (call that one first).
    fn update_conv_flux(&mut self, gpu: &Gpu, phi: &GpuSurfaceScalarField) -> Result<()> {
        let m = self.m;
        let cp = self.props.cp;

        field_ops::copy_field(gpu, &self.fldk, &mut self.phi_conv.f, &phi.f, m.n_internal_faces)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.phi_conv.f, &self.rho_face.f, m.n_internal_faces)?;
        field_ops::scale_field(gpu, &self.fldk, &mut self.phi_conv.f, cp, m.n_internal_faces)?;

        field_ops::copy_field(gpu, &self.fldk, &mut self.phi_conv.bf, &phi.bf, m.n_boundary_faces)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.phi_conv.bf, &self.rho_face.bf, m.n_boundary_faces)?;
        field_ops::scale_field(gpu, &self.fldk, &mut self.phi_conv.bf, cp, m.n_boundary_faces)
    }

    fn add_ddt(&mut self, gpu: &Gpu) -> Result<()> {
        if !self.ddt.is_active() {
            return Ok(());
        }
        match self.ddt.scheme {
            DdtScheme::SteadyState => Ok(()),
            DdtScheme::Euler | DdtScheme::Backward => {
                let c = self.ddt.state.coeffs(self.ddt.scheme)?;
                crate::timescheme::fvm_ddt_rho(
                    gpu,
                    &self.tsk,
                    &mut self.a,
                    self.m,
                    &self.rho_cp,
                    &self.rho_cp0,
                    &self.rho_cp00,
                    &self.t.f0,
                    &self.t.f00,
                    c,
                    1.0,
                )
            }
            other => Err(Error::Config(format!(
                "energy: ddt scheme {other:?} reached assembly unreconciled \
                 - Energy::new is supposed to reject this before it gets here"
            ))),
        }
    }

    /// Assemble the matrix once (one non-orthogonal pass). SPEC-LIT S26's
    /// terms, in the order they are added:
    ///
    /// 1. `ddt(rho cp, T)` - rho*cp-weighted, whichever of Euler/backward
    ///    `ddtSchemes` named ([`Self::add_ddt`]).
    /// 2. `div(phi_m, T)` - Gauss convection on the mass flux.
    /// 3. the bounded correction, UNCONDITIONALLY (S26: "with a nonzero
    ///    target divergence it is PHYSICS, not stabilisation").
    /// 4. the scheme's own deferred correction, if it has one.
    /// 5. `laplacian(k_eff, T)`, plus its non-orthogonal correction if the
    ///    case asked for one.
    /// 6. the S18 registry (`fvm_su`/`fvm_sp`) and `dp0/dt`, both explicit.
    fn assemble(&mut self, gpu: &Gpu, gas: &GasState) -> Result<()> {
        self.a.zero(gpu)?;
        let m = self.m;
        let scheme: DivScheme = self.ctrl.div_scheme.scheme.into();

        if scheme.needs_gradient() {
            fv::fvc_grad_scalar_scheme(gpu, &self.fvk, &mut self.grad_t, &self.t, m, self.ctrl.grad_scheme)?;
        }
        fv::div_scheme_weights(
            gpu,
            &self.fvk,
            Some(&mut self.w),
            Some(&mut self.bw),
            scheme,
            &self.phi_conv,
            &self.t,
            if scheme.needs_gradient() { Some(&self.grad_t) } else { None },
            m,
        )?;

        self.add_ddt(gpu)?;

        fv::fvm_div_gauss(gpu, &self.fvk, &mut self.a, m, &self.phi_conv, &self.w, &self.bw, &self.t, 1.0)?;
        fv::fvm_div_bounded_correction(gpu, &self.fvk, &mut self.a, m, &self.phi_conv, 1.0)?;

        if scheme.correction().is_some() {
            fv::fvm_div_correction(gpu, &self.fvk, &mut self.a, m, &self.phi_conv, &self.grad_t, scheme, 1.0)?;
        }

        fv::fvm_laplacian(gpu, &self.fvk, &mut self.a, m, &self.k_eff_mag_sf.f, &self.k_eff_mag_sf.bf, &self.t, -1.0)?;

        if self.ctrl.sn_grad.applies() {
            fv::fvc_grad_scalar_scheme(gpu, &self.fvk, &mut self.grad_t, &self.t, m, self.ctrl.grad_scheme)?;
            fv::fvm_laplacian_non_orth_correction(
                gpu,
                &self.fvk,
                &mut self.a,
                m,
                &self.k_eff_mag_sf.f,
                &self.k_eff_mag_sf.bf,
                &self.t,
                &self.grad_t,
                self.ctrl.sn_grad,
                -1.0,
            )?;
        }

        fv::fvm_su(gpu, &self.fvk, &mut self.a, m, self.sources.q(), 1.0)?;
        // sp is stored as the true, non-positive S_p (SPEC-LIT S3.4/S18); a
        // sink strengthens the diagonal, which is `sign = -1` against
        // `fvm_sp`'s own "sign*sp >= 0 stabilises" convention.
        fv::fvm_sp(gpu, &self.fvk, &mut self.a, m, self.sources.sp(), -1.0)?;

        field_ops::set_field(gpu, &self.fldk, &mut self.dp0dt_su, gas.dp0dt(), m.n_cells)?;
        fv::fvm_su(gpu, &self.fvk, &mut self.a, m, &self.dp0dt_su, 1.0)
    }

    /// SPEC-LIT S25.1: `(div u)_target = Q/(rho cp T) - dp0/dt/(gamma p0)`,
    /// from [`EnergySources::q`] and `gas`'s current `p0`/`dp0dt`. Call before
    /// [`crate::simple::Simple`]'s pressure equation this outer iteration;
    /// `T` here is the CURRENT (pre-solve) field, matching the same lag
    /// `nu_t` and every other coupling coefficient in this crate already
    /// runs at.
    pub fn update_target_divergence(&mut self, gpu: &Gpu, gas: &GasState) -> Result<()> {
        let n = self.m.n_cells;
        if n == 0 {
            return Ok(());
        }
        let inv_gamma_p0 = 1.0 / (self.props.gamma * gas.p0());
        self.ek.target_divergence(
            gpu,
            &mut self.target_div,
            self.sources.q(),
            &gas.rho().f,
            &self.t.f,
            self.props.cp,
            inv_gamma_p0,
            gas.dp0dt(),
            n,
        )
    }

    /// One implicit step, or one outer iteration if the run is steady.
    ///
    /// `phi` is the CURRENT volumetric flux (from
    /// [`crate::momentum::Momentum::phi_hbya`]/the corrected `phi`, not a
    /// mass flux - this function builds the mass flux itself from `gas`).
    /// `nut` is the eddy viscosity the momentum/turbulence equations solved
    /// with this iteration, the same segregated lag every other equation in
    /// this crate reads it with. `k` (the turbulence kinetic energy's cell
    /// field) is read ONLY by [`Self::update_thermal_wall`] - SPEC-LIT S29.3
    /// - and only where [`Self::set_thermal_wall`] has been called; a case
    /// with no thermal wall function may pass any field of the right length
    /// (it is never dereferenced). `nu`, the molecular kinematic viscosity,
    /// is read by that same wall function AND - since SPEC-LIT S37 - by
    /// [`Self::update_k_eff`] whenever [`GasProperties::pr_t_model`] is
    /// [`PrtModel::KaysCrawford`], which forms the turbulent Peclet number
    /// `Pe_t = (nu_t/nu) Pr` from it. Under [`PrtModel::Constant`] it reaches
    /// nothing but the wall function, exactly as before.
    pub fn correct(
        &mut self,
        gpu: &Gpu,
        phi: &GpuSurfaceScalarField,
        nut: &GpuScalarField,
        k: &DevBuf<Scalar>,
        nu: Scalar,
        gas: &GasState,
    ) -> Result<SolverPerformance> {
        let m = self.m;
        let n = m.n_cells;
        if n == 0 {
            return Ok(SolverPerformance::default());
        }

        field_ops::store_old_time(gpu, &self.fldk, &mut self.t)?;

        self.refresh_rho_cp(gpu, gas)?;
        self.update_k_eff(gpu, nut, gas, nu)?;
        self.update_thermal_wall(gpu, k, &gas.rho().f, nu)?;
        self.update_fixed_flux(gpu)?;
        self.update_conv_flux(gpu, phi)?;

        let alpha = self.ctrl.t_relax;
        let sc = self.ctrl.t_solver;
        let mut perf = SolverPerformance::default();

        for _pass in 0..=self.ctrl.n_non_orth_correctors {
            self.assemble(gpu, gas)?;

            ldu_ops::relax(gpu, &self.lduk, &mut self.a, m, &self.t.f, alpha)?;
            ldu_ops::add_boundary_contributions(gpu, &self.lduk, &mut self.a, m)?;

            perf = solver::solve(gpu, &self.solk, &mut self.t.f, &self.a, m, &mut self.ws, &sc)?;

            field_ops::correct_boundary_conditions(gpu, &self.fldk, &mut self.t, m)?;
        }

        Ok(perf)
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::BcKind;
    use crate::wallfunctions::thermal_wall_ref_grad;
    use crate::mesh::{HostMesh, PatchKind};
    use crate::momentum::BuoyancyCoeffs;

    fn gpu() -> Option<crate::Gpu> {
        crate::Gpu::new(0).ok()
    }

    // ----------------------------------------------------------------------
    //  GasProperties / GasState
    // ----------------------------------------------------------------------

    // ----------------------------------------------------------------------
    //  §37  Kays-Crawford's variable turbulent Prandtl number
    // ----------------------------------------------------------------------

    /// The literature form of SPEC-LIT S37.1, written out exactly as the
    /// papers print it, so the rearrangement [`kays_crawford_prt`] evaluates
    /// can be CHECKED against it rather than merely asserted to equal it.
    /// Not used by the solver - it is the thing being checked, and it is the
    /// form that loses precision (see the test that measures where).
    fn kays_crawford_prt_literature(pe_t: f64, c: f64, pr_t_inf: f64) -> f64 {
        let x = c * pe_t;
        let a = pr_t_inf.sqrt();
        1.0 / (1.0 / (2.0 * pr_t_inf) + x / a - x * x * (1.0 - (-1.0 / (x * a)).exp()))
    }

    /// SPEC-LIT S37.2's first limit, DERIVED there and asserted here:
    /// `Pe_t -> 0` gives `Pr_t -> 2 Pr_t_inf`, the conduction-sublayer value.
    /// Exact, not approximate, because the small-`Pe_t` branch returns the
    /// limit itself.
    #[test]
    fn kays_crawford_at_zero_peclet_is_exactly_twice_the_free_stream_value() {
        for pr_t_inf in [0.7 as Scalar, 0.85, 0.9, 1.0] {
            let got = kays_crawford_prt(0.0, KAYS_CRAWFORD_C, pr_t_inf);
            assert_eq!(got, 2.0 * pr_t_inf, "Pe_t = 0, Pr_t_inf = {pr_t_inf}");
        }
        // The number this project's own air cases land on at a wall where a
        // low-Re model has pinned nu_t to zero.
        assert_eq!(kays_crawford_prt(0.0, KAYS_CRAWFORD_C, 0.85), 1.7);
    }

    /// The branch exists because `Pe_t` can be positive and still send the
    /// formula's inner argument to infinity. SPEC-LIT S37.2's own worked
    /// case: at `Pe_t = 1e-300` the literature bracket's third term is
    /// `0 * (1 - exp(-huge))` and the rearranged form's `u*u` overflows -
    /// both have to come out at the limit rather than at NaN.
    #[test]
    fn kays_crawford_at_1e_300_is_still_the_sublayer_limit() {
        let got = kays_crawford_prt(1e-300, KAYS_CRAWFORD_C, 0.85);
        assert!(got.is_finite(), "Pe_t = 1e-300 gave {got}");
        assert_eq!(got, 1.7);
    }

    /// SPEC-LIT S37.2's second limit: `Pe_t -> inf` gives `Pr_t -> Pr_t_inf`,
    /// the free-stream value, approached FROM ABOVE at the rate S37.2
    /// derives - `Pr_t = Pr_t_inf (1 + 1/(6 sqrt(Pr_t_inf) C Pe_t)) + O(Pe_t^-2)`.
    #[test]
    fn kays_crawford_at_large_peclet_approaches_the_free_stream_value_from_above() {
        let (c, p_inf) = (KAYS_CRAWFORD_C, 0.85 as Scalar);
        for pe_t in [1e3 as Scalar, 1e4, 1e5, 1e6] {
            let got = kays_crawford_prt(pe_t, c, p_inf);
            assert!(got > p_inf, "Pe_t = {pe_t}: {got} is not above {p_inf}");
            let want = p_inf * (1.0 + 1.0 / (6.0 * p_inf.sqrt() * c * pe_t));
            // The next term is O(Pe_t^-2), so the first-order estimate has to
            // agree to better than that.
            assert!(
                (got - want).abs() < 10.0 / (pe_t * pe_t),
                "Pe_t = {pe_t}: {got} against the asymptote {want}"
            );
        }
        assert!((kays_crawford_prt(1e9, c, p_inf) - p_inf).abs() < 1e-9);
    }

    /// The rearrangement is the SAME function, not an approximation to it -
    /// SPEC-LIT S37.2. Checked against the literature form everywhere the
    /// literature form is still trustworthy.
    #[test]
    fn the_rearranged_form_reproduces_the_literature_form() {
        let (c, p_inf) = (0.3_f64, 0.85_f64);
        let mut worst: f64 = 0.0;
        // 1e-4 .. 1e3 in Pe_t. The upper end is where the LITERATURE form's
        // own cancellation first shows (it is already 3e-10 out at Pe_t = 1e4
        // and 4% out at 1e8), so past it the two forms disagree because the
        // reference is wrong, not the implementation - the next test measures
        // exactly that.
        for i in 0..=140 {
            let pe_t: f64 = 10.0_f64.powf(-4.0 + 0.05 * f64::from(i));
            let want = kays_crawford_prt_literature(pe_t, c, p_inf);
            let got =
                f64::from(kays_crawford_prt(pe_t as Scalar, c as Scalar, p_inf as Scalar));
            worst = worst.max((got - want).abs() / want);
        }
        assert!(worst < 1e-10, "worst relative disagreement {worst:e}");
    }

    /// Why the rearrangement is not cosmetic: at large `Pe_t` the literature
    /// form subtracts two numbers of order `C Pe_t/sqrt(Pr_t_inf)` to leave
    /// one of order `1/Pr_t_inf`, and the digits go with them. Measured here
    /// rather than asserted in a comment.
    #[test]
    fn the_literature_form_is_the_one_that_loses_the_digits() {
        let (c, p_inf) = (0.3_f64, 0.85_f64);
        // The true answer at Pe_t = 1e8 is p_inf to eight decimal places.
        let literature = kays_crawford_prt_literature(1e8, c, p_inf);
        let rearranged =
            f64::from(kays_crawford_prt(1e8 as Scalar, c as Scalar, p_inf as Scalar));
        assert!(
            (rearranged - p_inf).abs() < 1e-7,
            "the rearranged form should sit at the asymptote; it gave {rearranged}"
        );
        assert!(
            (literature - p_inf).abs() > 1e-3,
            "the literature form was expected to have lost its digits by \
             Pe_t = 1e8; it gave {literature}"
        );
    }

    /// The physical statement the correlation exists to make: `Pr_t` falls
    /// MONOTONICALLY from `2 Pr_t_inf` at the wall to `Pr_t_inf` in the free
    /// stream, and never leaves that interval. SPEC-LIT S37.5's own row.
    #[test]
    fn kays_crawford_is_monotone_between_its_two_limits() {
        let (c, p_inf) = (KAYS_CRAWFORD_C, 0.85 as Scalar);
        let mut prev = kays_crawford_prt(0.0, c, p_inf);
        for i in 0..=200 {
            let pe_t: Scalar = 10.0_f64.powf(-8.0 + 0.075 * f64::from(i)) as Scalar;
            let got = kays_crawford_prt(pe_t, c, p_inf);
            assert!(got.is_finite(), "Pe_t = {pe_t} gave {got}");
            assert!(
                got >= p_inf - 1e-12 && got <= 2.0 * p_inf + 1e-12,
                "Pe_t = {pe_t}: {got} is outside [{p_inf}, {}]",
                2.0 * p_inf
            );
            assert!(got <= prev + 1e-12, "Pe_t = {pe_t}: {got} rose above {prev}");
            prev = got;
        }
    }

    /// Nothing in the sweep - denormals, zero and infinity included - comes
    /// back as NaN or as a diffusivity a `k_eff` could not use.
    #[test]
    fn kays_crawford_is_finite_and_positive_everywhere_it_can_be_called() {
        let (c, p_inf) = (KAYS_CRAWFORD_C, 0.85 as Scalar);
        let inputs: [Scalar; 10] =
            [0.0, Scalar::MIN_POSITIVE, 1e-300, 1e-30, 1e-8, 1.0, 1e8, 1e30, 1e300, Scalar::MAX];
        for pe_t in inputs {
            let got = kays_crawford_prt(pe_t, c, p_inf);
            assert!(got.is_finite() && got > 0.0, "Pe_t = {pe_t:e} gave {got}");
        }
        assert_eq!(kays_crawford_prt(Scalar::INFINITY, c, p_inf), p_inf);
    }

    /// SPEC-LIT S13.4 on the selector itself: both spellings are recognised,
    /// anything else is an error that NAMES the menu.
    #[test]
    fn an_unrecognised_prt_model_is_a_13_4_error_naming_the_alternatives() {
        assert_eq!(PrtModel::parse("x", "constant").unwrap(), PrtModel::Constant);
        assert_eq!(PrtModel::parse("x", "KaysCrawford").unwrap(), PrtModel::KaysCrawford);

        let e = PrtModel::parse("physics.fluid.PrtModel", "kaysCrawfordJischa")
            .expect_err("an unknown spelling has to be refused");
        let msg = e.to_string();
        assert!(msg.contains("physics.fluid.PrtModel"), "{msg}");
        assert!(msg.contains("kaysCrawfordJischa"), "{msg}");
        assert!(msg.contains("constant") && msg.contains("KaysCrawford"), "{msg}");
    }

    /// The default has to stay `constant`: every measurement `ofgpu-fire`
    /// has recorded was made with one, and a default that moved would move
    /// all of them at once (SPEC-LIT S37.4).
    #[test]
    fn the_default_prt_model_is_the_constant_one() {
        assert_eq!(GasProperties::default().pr_t_model, PrtModel::Constant);
        assert_eq!(PrtModel::default(), PrtModel::Constant);
    }

    #[test]
    fn default_air_properties_validate_and_give_the_textbook_r_s() {
        let p = GasProperties::default();
        p.validate().expect("air properties");
        // R/W = 8.314462618/0.0289647 ~= 287.05 J/(kg K) - Incropera/ICAO air.
        assert!((p.r_s() - 287.05).abs() < 0.1, "R_s = {}", p.r_s());
    }

    #[test]
    fn a_non_positive_property_is_refused() {
        assert!(GasProperties { w: 0.0, ..GasProperties::default() }.validate().is_err());
        assert!(GasProperties { cp: -1.0, ..GasProperties::default() }.validate().is_err());
        assert!(GasProperties { gamma: 1.0, ..GasProperties::default() }.validate().is_err());
        assert!(GasProperties { k: 0.0, ..GasProperties::default() }.validate().is_err());
    }

    /// SPEC-LIT S25.3, "they coincide at constant p0 and W - show it in a
    /// test": `(rho(T) - rho(T_ref))*g/rho(T_ref)` and
    /// `crate::momentum::BuoyancyCoeffs::at(T) = g*(T_ref/T - 1)` are the SAME
    /// number at every `T`, not merely as `T -> T_ref`, because
    /// `rho(T)/rho(T_ref) = T_ref/T` exactly for an ideal gas at one `p0`.
    /// This is what licenses reusing `crate::momentum::Momentum` completely
    /// unmodified for the low-Mach solver's velocity/pressure system (the
    /// module doc's first *DESIGN* note).
    #[test]
    fn buoyancy_matches_the_density_ratio_at_any_deltat() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };
        let hm = tiny_box_mesh(2);
        let m = crate::GpuMesh::upload(&g, &hm)?;

        let props = GasProperties::default();
        let t_ref = 293.15;
        let p0 = props.r_s() * 300.0 * 1.2; // an arbitrary, plausible p0

        let gas = GasState::new(&g, &m, props, DomainKind::Open, p0)?;
        let rho_ref = gas.rho_at(t_ref);

        let buoy = BuoyancyCoeffs {
            g: crate::Vec3::new(0.0, 0.0, -9.81),
            t_ref,
            t_min: 1.0,
        };

        // Small, moderate and large delta-T/T, including the fire-plume-scale
        // ratio SPEC-LIT S9 itself uses (1173 K against 293 K).
        for t in [294.0, 305.0, 350.0, 600.0, 1173.15] {
            let rho = gas.rho_at(t);
            let density_ratio_force = buoy.g * ((rho - rho_ref) / rho_ref);
            let want = buoy.at(t);

            let diff = (density_ratio_force - want).mag();
            let scale = want.mag().max(1e-12);
            assert!(
                diff / scale < 1e-9,
                "T={t}: (rho-rho_ref)/rho_ref*g = {density_ratio_force:?}, \
                 g*(Tref/T-1) = {want:?}, relative diff {}",
                diff / scale
            );
        }
        Ok(())
    }

    /// SPEC-LIT S25.2, the decisive gate: a sealed box with a heater of known
    /// power `P` raises `p0` at exactly `dp0/dt = (gamma-1)P/V`.
    #[test]
    fn sealed_box_p0_ramp_matches_the_analytic_rate() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };
        let hm = tiny_box_mesh(4);
        let m = crate::GpuMesh::upload(&g, &hm)?;

        let props = GasProperties::default();
        let p0_init = props.r_s() * 300.0 * 1.2;
        let mut gas = GasState::new(&g, &m, props, DomainKind::Sealed, p0_init)?;

        let mut sources = EnergySources::new(&g, &m)?;
        // A uniform heater: q''' = P/V_dom everywhere, so integral(Q)dV = P
        // regardless of how the cells are sized.
        let p_watts: Scalar = 2500.0;
        let q_per_vol = p_watts / m.total_volume;
        let q_field = g.upload(&vec![q_per_vol; hm.n_cells])?;
        sources.register_explicit(&g, &q_field)?;

        let total_q = sources.total_q(&g, &m)?;
        assert!(
            (total_q - p_watts).abs() < 1e-6 * p_watts,
            "integral(Q)dV = {total_q}, expected {p_watts}"
        );

        let dt = 1e-3;
        gas.advance_p0(total_q, dt)?;

        let want_dp0dt = (props.gamma - 1.0) * p_watts / m.total_volume;
        let got = gas.dp0dt();
        let rel = (got - want_dp0dt).abs() / want_dp0dt.abs();
        assert!(
            rel < 1e-3,
            "dp0/dt = {got}, (gamma-1)*P/V = {want_dp0dt}, relative error {rel}"
        );

        // And an open domain with the identical heater does not move p0 at
        // all - S25.2's other half.
        let mut open = GasState::new(&g, &m, props, DomainKind::Open, p0_init)?;
        open.advance_p0(total_q, dt)?;
        assert_eq!(open.dp0dt(), 0.0);
        assert_eq!(open.p0(), p0_init);
        Ok(())
    }

    #[test]
    fn advance_p0_refuses_a_non_finite_input() {
        let Some(g) = gpu() else { return };
        let hm = tiny_box_mesh(2);
        let Ok(m) = crate::GpuMesh::upload(&g, &hm) else { return };
        let props = GasProperties::default();
        let Ok(mut gas) = GasState::new(&g, &m, props, DomainKind::Sealed, 101325.0) else { return };
        assert!(gas.advance_p0(Scalar::NAN, 1e-3).is_err());
        assert!(gas.advance_p0(1.0, -1.0).is_err());
    }

    // ----------------------------------------------------------------------
    //  A 1-D slab: two Dirichlet ends, empty everywhere else
    // ----------------------------------------------------------------------

    fn tiny_box_mesh(n: usize) -> HostMesh {
        let (mut m, points, faces) =
            crate::mesh::topology::tests::box_mesh([n, n, n], crate::Vec3::new(0.1, 0.1, 0.1));
        for p in m.patches.iter_mut() {
            p.kind = PatchKind::Generic;
            p.type_name = "patch".to_string();
        }
        m.compute_geometry(&points, &faces).expect("box geometry");
        m.build_cell_face_maps();
        m
    }

    /// `N` cells along `x`, a fixed condition at both ends, `empty` on every
    /// other face - the same construction `scalar_transport::tests::slab`
    /// uses, duplicated rather than shared because it lives in a `#[cfg(test)]`
    /// module private to that file.
    fn slab(n: usize, h: Scalar) -> HostMesh {
        let (mut m, points, faces) =
            crate::mesh::topology::tests::box_mesh([n, 1, 1], crate::Vec3::new(h, h, h));

        let kinds = [
            PatchKind::Generic,
            PatchKind::Generic,
            PatchKind::Empty,
            PatchKind::Empty,
            PatchKind::Empty,
            PatchKind::Empty,
        ];
        for (p, k) in m.patches.iter_mut().zip(kinds) {
            p.kind = k;
            p.type_name = if k == PatchKind::Empty { "empty" } else { "patch" }.to_string();
        }

        m.compute_geometry(&points, &faces).expect("slab geometry");
        m.build_cell_face_maps();
        m
    }

    /// Build an `Energy` over `slab(n, h)`, laminar (`nu_t = 0` everywhere,
    /// so `k_eff = k` exactly), no flow, steady `p0`.
    fn laminar_slab_energy<'m>(
        gpu: &crate::Gpu,
        hm: &HostMesh,
        m: &'m crate::GpuMesh,
        props: GasProperties,
        steady: bool,
        delta_t: Scalar,
    ) -> Result<(Energy<'m>, GpuScalarField, GpuSurfaceScalarField)> {
        let ctrl = EnergyControls {
            t_solver: SolverControls {
                tolerance: 1e-14,
                rel_tol: 0.0,
                max_iter: 2000,
                check_interval: 1,
                ..SolverControls::default()
            },
            t_relax: 1.0,
            steady,
            delta_t,
            sn_grad: SnGradScheme::Uncorrected,
            ddt: if steady { DdtScheme::SteadyState } else { DdtScheme::Euler },
            ..EnergyControls::default()
        };
        let e = Energy::new(gpu, m, ctrl, props)?;
        let nut = GpuScalarField::zeros(gpu, m, "nut")?;
        let phi = GpuSurfaceScalarField::zeros(gpu, m, "phi")?;
        let _ = hm;
        Ok((e, nut, phi))
    }

    fn set_dirichlet_ends(gpu: &crate::Gpu, hm: &HostMesh, t: &mut GpuScalarField, t0: Scalar, t1: Scalar) -> Result<()> {
        let nbf = hm.n_boundary_faces;
        let mut kind = vec![BcKind::Empty as Label; nbf];
        let mut fr = vec![0.0 as Scalar; nbf];
        let mut rv = vec![0.0 as Scalar; nbf];
        for (p, pi) in hm.patches.iter().enumerate() {
            if p < 2 {
                let v = if p == 0 { t0 } else { t1 };
                for k in 0..pi.size {
                    kind[pi.start + k] = BcKind::FixedValue as Label;
                    fr[pi.start + k] = 1.0;
                    rv[pi.start + k] = v;
                }
            }
        }
        gpu.write(&mut t.bc_kind, &kind)?;
        gpu.write(&mut t.fr, &fr)?;
        gpu.write(&mut t.ref_value, &rv)?;
        Ok(())
    }

    /// SPEC-LIT S37.3: `energyKEffKaysCrawford` has to reproduce the host
    /// [`kays_crawford_prt`] on every face, INCLUDING the two end branches -
    /// the same discipline `wallfunctions::thermal_wall_device_agrees_with_the_host_law`
    /// holds the thermal wall function to. `nu_t` is seeded across fourteen
    /// decades plus an exact zero, so both branches and the whole span
    /// between them are exercised on the device, not only on the host.
    #[test]
    fn kays_crawford_device_agrees_with_the_host_correlation() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        const N: usize = 16;
        let h: Scalar = 0.01;
        let hm = slab(N, h);
        let m = crate::GpuMesh::upload(&g, &hm)?;

        let nu: Scalar = 1.5e-5;
        let props = GasProperties {
            pr: 0.71,
            pr_t: 0.85,
            pr_t_model: PrtModel::KaysCrawford,
            ..GasProperties::default()
        };
        let (mut e, mut nut, _phi) = laminar_slab_energy(&g, &hm, &m, props, true, 1.0)?;

        // nu_t/nu from 0 (a resolved wall under a low-Re model) up through
        // the log layer and past it, one cell per decade.
        let nut_cells: Vec<Scalar> = (0..N)
            .map(|i| {
                if i == 0 {
                    0.0
                } else {
                    nu * (10.0 as Scalar).powf(-7.0 + i as Scalar)
                }
            })
            .collect();
        g.write(&mut nut.f, &nut_cells)?;
        field_ops::correct_boundary_conditions(&g, &FieldKernels::new(&g)?, &mut nut, &m)?;

        let gas = GasState::new(&g, &m, props, DomainKind::Open, 101325.0)?;
        e.initialise(&g)?;
        let mut gas = gas;
        gas.update_density(&g, e.field())?;

        e.update_k_eff(&g, &nut, &gas, nu)?;

        // Rebuild the face fields the kernel was handed, so the comparison is
        // against the SAME interpolated rho_f/nu_t_f rather than against a
        // second interpolation that could differ in the last bit.
        let rho_f = g.download(&e.rho_face.f)?;
        let nut_f = g.download(&e.nut_face.f)?;
        let k_eff_f = g.download(&e.k_eff_face.f)?;

        let mut worst: f64 = 0.0;
        let mut saw_sublayer_limit = false;
        let mut saw_free_stream = false;
        for i in 0..m.n_internal_faces {
            let pe_t = (nut_f[i] / nu) * props.pr;
            let prt = kays_crawford_prt(pe_t, KAYS_CRAWFORD_C, props.pr_t);
            saw_sublayer_limit |= (prt - 2.0 * props.pr_t).abs() < 1e-6;
            saw_free_stream |= (prt - props.pr_t).abs() < 1e-3;
            let want = props.k + rho_f[i] * nut_f[i] * props.cp / prt;
            let got = f64::from(k_eff_f[i]);
            worst = worst.max((got - f64::from(want)).abs() / f64::from(want).abs().max(1e-300));
        }
        assert!(worst < 1e-12, "worst relative host/device disagreement {worst:e}");
        assert!(saw_sublayer_limit, "the sweep never reached the 2*Pr_t_inf branch");
        assert!(saw_free_stream, "the sweep never reached the Pr_t_inf asymptote");

        // And the CONSTANT model on the identical field is the old formula,
        // exactly - the default has to be bit-for-bit what it always was.
        let props_c = GasProperties { pr_t_model: PrtModel::Constant, ..props };
        let (mut ec, _n2, _p2) = laminar_slab_energy(&g, &hm, &m, props_c, true, 1.0)?;
        ec.initialise(&g)?;
        ec.update_k_eff(&g, &nut, &gas, nu)?;
        let k_eff_c = g.download(&ec.k_eff_face.f)?;
        for i in 0..m.n_internal_faces {
            let want = props_c.k + rho_f[i] * nut_f[i] * props_c.cp / props_c.pr_t;
            assert_eq!(k_eff_c[i], want, "face {i}: the constant branch moved");
        }
        Ok(())
    }

    /// Steady slab conduction, fixed flux at `x=0`, fixed value at `x=L`:
    /// the exact solution is linear in `x`, and the discrete Gauss laplacian
    /// on a uniform orthogonal mesh reproduces a linear field exactly.
    #[test]
    fn steady_slab_fixed_flux_gives_an_exact_linear_profile() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        const N: usize = 10;
        let h: Scalar = 0.02;
        let hm = slab(N, h);
        let m = crate::GpuMesh::upload(&g, &hm)?;

        let props = GasProperties { k: 0.5, cp: 1000.0, ..GasProperties::default() };
        let (mut e, nut, phi) = laminar_slab_energy(&g, &hm, &m, props, true, 1.0)?;

        let gas = GasState::new(&g, &m, props, DomainKind::Open, 101325.0)?;

        let q_w: Scalar = 200.0; // W/m2, INTO the domain at x=0
        let t_l: Scalar = 300.0; // fixed value at x=L

        {
            let f = e.field_mut();
            let nbf = hm.n_boundary_faces;
            let mut kind = vec![BcKind::Empty as Label; nbf];
            let mut fr = vec![0.0 as Scalar; nbf];
            let mut rv = vec![0.0 as Scalar; nbf];
            let mut rg = vec![0.0 as Scalar; nbf];
            for (p, pi) in hm.patches.iter().enumerate() {
                match p {
                    0 => {
                        // x=0: fixed flux. `ref_grad` is the OUTWARD-normal
                        // derivative (SPEC-LIT S2.4's `Delta_b`, evaluated
                        // from the cell to the boundary face); at the xmin
                        // patch the outward normal is -x, so a profile that
                        // FALLS with +x (`dT/dx = -q_w/k`, heat flowing in
                        // +x) has a POSITIVE outward-normal derivative,
                        // `dT/dn = -dT/dx = +q_w/k`.
                        for k in 0..pi.size {
                            kind[pi.start + k] = BcKind::FixedValue as Label;
                            fr[pi.start + k] = 0.0;
                            rg[pi.start + k] = flux_to_grad(q_w, props.k);
                        }
                    }
                    1 => {
                        for k in 0..pi.size {
                            kind[pi.start + k] = BcKind::FixedValue as Label;
                            fr[pi.start + k] = 1.0;
                            rv[pi.start + k] = t_l;
                        }
                    }
                    _ => {}
                }
            }
            g.write(&mut f.bc_kind, &kind)?;
            g.write(&mut f.fr, &fr)?;
            g.write(&mut f.ref_value, &rv)?;
            g.write(&mut f.ref_grad, &rg)?;
            g.write(&mut f.f, &vec![t_l; hm.n_cells])?;
        }
        e.initialise(&g)?;
        let mut gas = gas;
        gas.update_density(&g, e.field())?;

        // No thermal wall function in this test - `set_thermal_wall` was
        // never called, so `k`/`nu` below are never read.
        let k = g.zeros::<Scalar>(hm.n_cells.max(1))?;
        for _ in 0..3 {
            e.correct(&g, &phi, &nut, &k, 0.0, &gas)?;
        }

        // T(x) = T_L + (q_w/k) * (L - x), so dT/dx = -q_w/k everywhere.
        let got = g.download(&e.field().f)?;
        let l = N as Scalar * h;
        let slope = -q_w / props.k;
        for i in 0..N {
            let x = (i as Scalar + 0.5) * h;
            let want = t_l + slope * (x - l);
            assert!(
                (got[i] - want).abs() < 1e-6 * (1.0 + want.abs()),
                "cell {i}: T={}, want {want} (x={x})",
                got[i]
            );
        }
        Ok(())
    }

    /// Build the same slab as [`steady_slab_fixed_flux_gives_an_exact_linear_profile`]
    /// but through [`BcKind::FixedFluxTemperature`]/[`Energy::set_fixed_flux_walls`]
    /// (SPEC-LIT §32.2) rather than a hand-set triple, with a UNIFORM but
    /// NONZERO eddy viscosity so `k_eff_wall != props.k` - the case that
    /// would expose a wall condition that silently used the molecular `k`
    /// (or a `k_eff_wall` computed once and never refreshed) instead of the
    /// CURRENT per-face `k_eff_wall` §32.2 asks for. `gas.update_density` is
    /// called once, before the loop, exactly as the test above - `rho` stays
    /// at its uniform initial value for the whole run, so `k_eff_wall` is a
    /// single known constant throughout and every check below is closed-form.
    #[allow(clippy::too_many_arguments)]
    fn fixed_flux_slab<'m>(
        gpu: &crate::Gpu,
        hm: &HostMesh,
        m: &'m crate::GpuMesh,
        props: GasProperties,
        q_w: Scalar,
        t_l: Scalar,
        nut_val: Scalar,
    ) -> Result<(Energy<'m>, GpuScalarField, GpuSurfaceScalarField, GasState<'m>, Scalar)> {
        let (mut e, mut nut, phi) = laminar_slab_energy(gpu, hm, m, props, true, 1.0)?;
        let nbf = hm.n_boundary_faces;

        gpu.write(&mut nut.f, &vec![nut_val; hm.n_cells])?;
        gpu.write(&mut nut.bf, &vec![nut_val; nbf])?;

        {
            let f = e.field_mut();
            let mut kind = vec![BcKind::Empty as Label; nbf];
            let mut fr = vec![0.0 as Scalar; nbf];
            let mut rv = vec![0.0 as Scalar; nbf];
            for (p, pi) in hm.patches.iter().enumerate() {
                match p {
                    // x=0: the fixed-flux patch - `q` lives in `ref_value`,
                    // exactly the `ThermalWallFunction`-style "seeded once,
                    // read every iteration" convention (SPEC-LIT §32.2).
                    0 => {
                        for k in 0..pi.size {
                            kind[pi.start + k] = BcKind::FixedFluxTemperature as Label;
                            fr[pi.start + k] = 0.0;
                            rv[pi.start + k] = q_w;
                        }
                    }
                    1 => {
                        for k in 0..pi.size {
                            kind[pi.start + k] = BcKind::FixedValue as Label;
                            fr[pi.start + k] = 1.0;
                            rv[pi.start + k] = t_l;
                        }
                    }
                    _ => {}
                }
            }
            gpu.write(&mut f.bc_kind, &kind)?;
            gpu.write(&mut f.fr, &fr)?;
            gpu.write(&mut f.ref_value, &rv)?;
            gpu.write(&mut f.f, &vec![t_l; hm.n_cells])?;
        }

        let mut faces = vec![false; nbf];
        for k in 0..hm.patches[0].size {
            faces[hm.patches[0].start + k] = true;
        }
        e.set_fixed_flux_walls(gpu, &faces)?;

        e.initialise(gpu)?;
        let mut gas = GasState::new(gpu, m, props, DomainKind::Open, 101325.0)?;
        gas.update_density(gpu, e.field())?;

        let cp_over_prt = props.cp / props.pr_t;
        let rho0 = props_rho_at(props, t_l);
        let k_eff_wall = props.k + rho0 * nut_val * cp_over_prt;

        Ok((e, nut, phi, gas, k_eff_wall))
    }

    /// `rho = p0/(R_s T)` at the SAME `p0`/`T` [`fixed_flux_slab`] seeds
    /// `GasState` with - a tiny host mirror so the tests above do not need to
    /// download `gas.rho()` just to predict `k_eff_wall` in closed form.
    fn props_rho_at(props: GasProperties, t: Scalar) -> Scalar {
        101325.0 / (props.r_s() * t)
    }

    /// SPEC-LIT §32.2's one-cell analytic check: after a SINGLE
    /// [`Energy::correct`] call (no need to iterate to convergence - the
    /// fixed-flux triple is exact at whatever `T_P`/`k_eff_wall` the matrix
    /// was assembled with, every single call), the flux face's `ref_grad`
    /// must equal `q_w / k_eff_wall` EXACTLY, with `k_eff_wall` the CURRENT
    /// per-face value (molecular `k` plus the eddy term from the uniform,
    /// nonzero `nut` this test sets) - not the molecular `props.k` alone,
    /// which is what a wall condition that ignored `k_eff_wall` entirely
    /// would produce instead.
    #[test]
    fn fixed_flux_triple_is_exact_after_one_iteration() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        const N: usize = 6;
        let h: Scalar = 0.02;
        let hm = slab(N, h);
        let m = crate::GpuMesh::upload(&g, &hm)?;

        let props = GasProperties { k: 0.5, cp: 1000.0, ..GasProperties::default() };
        let q_w: Scalar = 200.0;
        let t_l: Scalar = 300.0;
        let nut_val: Scalar = 0.01;

        let (mut e, nut, phi, gas, k_eff_wall) =
            fixed_flux_slab(&g, &hm, &m, props, q_w, t_l, nut_val)?;

        // Sanity: the eddy term genuinely dominates here, so a test that
        // passed with the molecular `props.k` instead would be caught.
        assert!(
            k_eff_wall > 5.0 * props.k,
            "k_eff_wall = {k_eff_wall} should be well above molecular k = {}",
            props.k
        );

        let k_cell = g.zeros::<Scalar>(hm.n_cells.max(1))?;
        e.correct(&g, &phi, &nut, &k_cell, 0.0, &gas)?;

        let fr = g.download(&e.field().fr)?;
        let rg = g.download(&e.field().ref_grad)?;
        let bf0 = hm.patches[0].start;
        assert_eq!(fr[bf0], 0.0, "fixedFluxTemperature must stay fr = 0");

        let want = flux_to_grad(q_w, k_eff_wall);
        assert!(
            (rg[bf0] - want).abs() < 1e-9 * want.abs(),
            "ref_grad = {}, want q_w/k_eff_wall = {want}",
            rg[bf0]
        );

        // And the identity §32.2 actually cares about: the flux the matrix
        // sees, k_eff_wall * ref_grad, is exactly q_w - whatever k_eff_wall
        // happens to be.
        let flux = k_eff_wall * rg[bf0];
        assert!(
            (flux - q_w).abs() < 1e-9 * q_w.abs(),
            "k_eff_wall * ref_grad = {flux}, want q_w = {q_w}"
        );
        Ok(())
    }

    /// SPEC-LIT §32.2: the imposed flux comes back out of the ASSEMBLED
    /// equation, not just out of the boundary triple in isolation - the
    /// steady conduction profile this converges to has EXACTLY the slope
    /// `-q_w/k_eff_wall`, with the turbulent `k_eff_wall` this test's
    /// nonzero `nut` implies, not the molecular `props.k` alone.
    #[test]
    fn fixed_flux_temperature_reproduces_q_through_the_assembled_equation() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        const N: usize = 10;
        let h: Scalar = 0.02;
        let hm = slab(N, h);
        let m = crate::GpuMesh::upload(&g, &hm)?;

        let props = GasProperties { k: 0.5, cp: 1000.0, ..GasProperties::default() };
        let q_w: Scalar = 200.0;
        let t_l: Scalar = 300.0;
        let nut_val: Scalar = 0.01;

        let (mut e, nut, phi, gas, k_eff_wall) =
            fixed_flux_slab(&g, &hm, &m, props, q_w, t_l, nut_val)?;

        let k_cell = g.zeros::<Scalar>(hm.n_cells.max(1))?;
        for _ in 0..5 {
            e.correct(&g, &phi, &nut, &k_cell, 0.0, &gas)?;
        }

        // T(x) = T_L + (q_w/k_eff_wall) * (L - x).
        let got = g.download(&e.field().f)?;
        let l = N as Scalar * h;
        let slope = -q_w / k_eff_wall;
        for i in 0..N {
            let x = (i as Scalar + 0.5) * h;
            let want = t_l + slope * (x - l);
            assert!(
                (got[i] - want).abs() < 1e-6 * (1.0 + want.abs()),
                "cell {i}: T={}, want {want} (x={x}, k_eff_wall={k_eff_wall})",
                got[i]
            );
        }
        Ok(())
    }

    /// Abramowitz & Stegun 7.1.26, |error| < 1.5e-7 - a public-domain
    /// rational approximation used HERE ONLY, to check the solver against the
    /// analytic erf solution; not part of the solver itself.
    fn erf(x: Scalar) -> Scalar {
        let x = x as f64;
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let pp = 0.3275911;
        let t = 1.0 / (1.0 + pp * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        (sign * y) as Scalar
    }

    /// 1-D transient conduction, fixed-T ends, against the erf solution for a
    /// semi-infinite solid (Incropera S5.7) - SPEC-LIT S26's decisive test,
    /// including the 2nd-order-in-space convergence check.
    ///
    /// The domain is long enough, and the run short enough, that the far end
    /// (fixed at the initial temperature) never feels the disturbance to
    /// better than the tolerance checked here - it stands in for the
    /// semi-infinite solid the closed form assumes.
    #[test]
    fn one_d_transient_conduction_matches_erf_at_second_order() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        let alpha: Scalar = 1.0e-4; // an arbitrary, convenient diffusivity
        let k_mol: Scalar = 1.0;
        let cp: Scalar = 1.0;
        // alpha = k/(rho cp)  =>  rho = k/(alpha cp)
        let rho_val = k_mol / (alpha * cp);
        let props = GasProperties { k: k_mol, cp, ..GasProperties::default() };
        let r_s = props.r_s();
        let t_far: Scalar = 300.0;
        let t_wall: Scalar = 400.0;
        let p0 = rho_val * r_s * t_far;

        let time: Scalar = 4.0;
        let l: Scalar = 12.0 * (alpha * time).sqrt(); // >> the diffusion length

        let mut errors = Vec::new();
        for &n in &[40usize, 80usize] {
            let h = l / n as Scalar;
            let hm = slab(n, h);
            let m = crate::GpuMesh::upload(&g, &hm)?;

            let ctrl = EnergyControls {
                t_solver: SolverControls {
                    tolerance: 1e-14,
                    rel_tol: 0.0,
                    max_iter: 2000,
                    check_interval: 1,
                    ..SolverControls::default()
                },
                t_relax: 1.0,
                steady: false,
                delta_t: 1.0, // overwritten per step below
                sn_grad: SnGradScheme::Uncorrected,
                ddt: DdtScheme::Euler,
                ..EnergyControls::default()
            };

            // dt small enough for O(dt) time error to stay well below the
            // spatial O(h^2) error being measured.
            let steps = 4000;
            let dt = time / steps as Scalar;
            let ctrl = EnergyControls { delta_t: dt, ..ctrl };

            let mut e = Energy::new(&g, &m, ctrl, props)?;
            let mut gas = GasState::new(&g, &m, props, DomainKind::Open, p0)?;
            let nut = GpuScalarField::zeros(&g, &m, "nut")?;
            let phi = GpuSurfaceScalarField::zeros(&g, &m, "phi")?;

            {
                let f = e.field_mut();
                set_dirichlet_ends(&g, &hm, f, t_wall, t_far)?;
                g.write(&mut f.f, &vec![t_far; hm.n_cells])?;
            }
            e.initialise(&g)?;
            // Seeded ONCE, from the uniform T = t_far the run starts at, and
            // held fixed for the whole transient: this test validates the
            // energy equation's discretisation order at CONSTANT properties,
            // decoupled from the ideal-gas rho(T) feedback the buoyancy and
            // p0-ramp tests cover separately. Letting rho track T here would
            // make alpha = k/(rho cp) vary by up to (t_wall-t_far)/t_far
            // across the domain, and the erf solution below assumes it does
            // not.
            gas.update_density(&g, e.field())?;

            // No thermal wall function in this test.
            let k = g.zeros::<Scalar>(hm.n_cells.max(1))?;
            for _ in 0..steps {
                e.correct(&g, &phi, &nut, &k, 0.0, &gas)?;
            }

            let got = g.download(&e.field().f)?;
            let mut max_err: Scalar = 0.0;
            for i in 0..n {
                let x = (i as Scalar + 0.5) * h;
                let want = t_far + (t_wall - t_far) * (1.0 - erf(x / (2.0 * (alpha * time).sqrt())));
                max_err = max_err.max((got[i] - want).abs());
            }
            errors.push(max_err);
        }

        // Halving h should quarter the error (2nd order); demand at least a
        // factor of 3 to leave room for the discrete-vs-continuous mismatch
        // right at the wall face, which this test does not try to correct
        // for.
        let ratio = errors[0] / errors[1];
        assert!(
            ratio > 3.0,
            "errors were {errors:?}; ratio {ratio} is not close to the \
             4x a 2nd-order scheme should give when h is halved"
        );
        Ok(())
    }

    // ----------------------------------------------------------------------
    //  SPEC-LIT §29.3 - the Jayatilleke thermal wall function's wiring
    // ----------------------------------------------------------------------

    /// [`Energy::set_thermal_wall`] plus one [`Energy::correct`] must rewrite
    /// the `thermalWallFunction` face's triple to exactly
    /// [`crate::wallfunctions::thermal_wall_ref_grad`]'s answer, evaluated at
    /// the cell temperature as it stood BEFORE this call (`update_thermal_wall`
    /// runs before `assemble`, so it reads the still-previous `T_P`) and at
    /// `k_eff_wall = props.k` exactly - `nut` is zero everywhere in this
    /// laminar test, so S26's `k_eff = k + rho cp nu_t/Pr_t` has nothing to
    /// add. This is the wall-BC WIRING test; [`thermal_wall_ref_grad`]'s own
    /// module in `src/wallfunctions.rs` is where the LAW itself is pinned
    /// down.
    #[test]
    fn set_thermal_wall_rewrites_the_triple_through_correct() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };

        const N: usize = 6;
        let h: Scalar = 0.05;
        let hm = slab(N, h);
        let m = crate::GpuMesh::upload(&g, &hm)?;

        let props = GasProperties { k: 0.03, cp: 1006.0, ..GasProperties::default() };
        let (mut e, nut, phi) = laminar_slab_energy(&g, &hm, &m, props, true, 1.0)?;

        let wc = WallFunctionCoeffs::default();
        let mut faces = vec![false; hm.n_boundary_faces];
        let wall_patch = &hm.patches[0]; // xmin, same convention every other slab test here uses
        for i in 0..wall_patch.size {
            faces[wall_patch.start + i] = true;
        }
        e.set_thermal_wall(&g, wc, &faces)?;

        let t_w: Scalar = 350.0;
        let t_init: Scalar = 300.0;
        {
            let f = e.field_mut();
            let nbf = hm.n_boundary_faces;
            let mut kind = vec![BcKind::Empty as Label; nbf];
            let mut fr = vec![0.0 as Scalar; nbf];
            let mut rv = vec![0.0 as Scalar; nbf];
            for (p, pi) in hm.patches.iter().enumerate() {
                if p == 0 {
                    for k in 0..pi.size {
                        kind[pi.start + k] = BcKind::ThermalWallFunction as Label;
                        fr[pi.start + k] = 1.0; // field_setup's own seed for this kind
                        rv[pi.start + k] = t_w;
                    }
                } else {
                    for k in 0..pi.size {
                        kind[pi.start + k] = BcKind::ZeroGradient as Label;
                    }
                }
            }
            g.write(&mut f.bc_kind, &kind)?;
            g.write(&mut f.fr, &fr)?;
            g.write(&mut f.ref_value, &rv)?;
            g.write(&mut f.f, &vec![t_init; hm.n_cells])?;
        }
        e.initialise(&g)?;

        let mut gas = GasState::new(&g, &m, props, DomainKind::Open, 101_325.0)?;
        gas.update_density(&g, e.field())?;

        let k_val: Scalar = 0.02;
        let nu: Scalar = 1.5e-5;
        let k_dev = g.upload(&vec![k_val; hm.n_cells])?;

        e.correct(&g, &phi, &nut, &k_dev, nu, &gas)?;

        let wall_face = wall_patch.start;
        let wall_cell = hm.b_face_cells[wall_face] as usize;
        let rho_host = g.download(&gas.rho().f)?;
        let y = hm.b_y[wall_face];

        let want = thermal_wall_ref_grad(
            t_w,
            t_init, // T_P as it stood before `correct` ran
            k_val,
            y,
            nu,
            rho_host[wall_cell],
            props.cp,
            props.pr,
            props.pr_t,
            wc.kappa,
            wc.e,
            wc.cmu,
            props.k, // k_eff_wall: nut = 0 everywhere, so k_eff = k exactly
            1e-15,
        )
        .expect("a positive standoff and k_eff must produce a ref_grad");

        let got_fr = g.download(&e.field().fr)?;
        let got_grad = g.download(&e.field().ref_grad)?;

        assert_eq!(
            got_fr[wall_face], 0.0,
            "ThermalWallFunction must rewrite fr to the fixedGradient degenerate form"
        );
        assert!(
            (got_grad[wall_face] - want).abs() <= 1e-9 * want.abs().max(1e-9),
            "wall face ref_grad {}, wanted {want}",
            got_grad[wall_face]
        );

        // Every OTHER boundary face is untouched: it never entered
        // `ThermalWallData`'s face list, so `fr` still holds whatever
        // `field_setup`/the test itself seeded (`zeroGradient`'s `fr = 0`,
        // coincidentally the same number - checked via `bc_kind` instead).
        let got_kind = g.download(&e.field().bc_kind)?;
        for (bf, &k) in got_kind.iter().enumerate() {
            if bf != wall_face {
                assert_ne!(k, BcKind::ThermalWallFunction as Label);
            }
        }

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  EnergySources
    // ----------------------------------------------------------------------

    #[test]
    fn registered_sources_sum_and_clear() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };
        let hm = tiny_box_mesh(2);
        let m = crate::GpuMesh::upload(&g, &hm)?;
        let mut s = EnergySources::new(&g, &m)?;

        let a = g.upload(&vec![10.0 as Scalar; hm.n_cells])?;
        let b = g.upload(&vec![5.0 as Scalar; hm.n_cells])?;
        s.register_explicit(&g, &a)?;
        s.register_explicit(&g, &b)?;

        let q = g.download(s.q())?;
        assert!(q.iter().all(|&v| (v - 15.0).abs() < 1e-12), "{q:?}");

        s.clear(&g)?;
        let q = g.download(s.q())?;
        assert!(q.iter().all(|&v| v == 0.0));
        Ok(())
    }

    #[test]
    fn a_mismatched_registration_is_refused() -> Result<()> {
        let Some(g) = gpu() else { return Ok(()) };
        let hm = tiny_box_mesh(3);
        let m = crate::GpuMesh::upload(&g, &hm)?;
        let mut s = EnergySources::new(&g, &m)?;
        let too_short = g.upload(&vec![1.0 as Scalar; 1])?;
        assert!(s.register_explicit(&g, &too_short).is_err());
        Ok(())
    }

    // ----------------------------------------------------------------------
    //  §13.4 contract
    // ----------------------------------------------------------------------

    #[test]
    fn an_unsupported_ddt_scheme_is_a_loud_error_naming_the_alternatives() {
        let _guard = crate::io::contract::permissive_test_guard();
        crate::io::contract::set_permissive(false);
        let e = reconcile_ddt(DdtScheme::LocalEuler).unwrap_err().to_string();
        assert!(e.contains("localEuler"), "{e}");
        assert!(e.contains("backward"), "{e}");
        assert!(e.contains("permissive"), "{e}");

        let e = reconcile_ddt(DdtScheme::CrankNicolson(0.9)).unwrap_err().to_string();
        assert!(e.contains("CrankNicolson"), "{e}");
    }

    #[test]
    fn steadystate_euler_and_backward_are_unaffected() {
        assert_eq!(reconcile_ddt(DdtScheme::SteadyState).unwrap(), DdtScheme::SteadyState);
        assert_eq!(reconcile_ddt(DdtScheme::Euler).unwrap(), DdtScheme::Euler);
        assert_eq!(reconcile_ddt(DdtScheme::Backward).unwrap(), DdtScheme::Backward);
    }

    #[test]
    fn flux_to_grad_is_the_plain_division() {
        assert!((flux_to_grad(200.0, 0.5) - 400.0).abs() < 1e-12);
    }
}
