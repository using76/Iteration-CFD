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
//! **3. The T-equation wall function is out of scope, and SAID so.**
//! `SPEC-LIT` S26: "the convective wall function for temperature
//! (Jayatilleke-type) is deferred". `crate::field::BcKind` has no
//! temperature-wall-function variant at all - there is structurally nothing
//! for a case to request - so a fixed-T or fixed-flux wall is exactly the
//! generic S4 Robin triple every other scalar in this crate already uses,
//! with [`flux_to_grad`] doing the one conversion a fixed-flux wall needs
//! (`g_ref = q_w/k_eff`). Whichever module parses `boundaryField/T/type` is
//! responsible for rejecting a request for the convective wall function by
//! name (S13.4); this module offers no silent substitute for one.

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
use crate::{Label, Scalar, Vec3};

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
    target_divergence: CudaFunction,
}

impl EnergyKernels {
    fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::ENERGY)?;
        Ok(Self {
            accumulate: k.func("energyAccumulate")?,
            k_eff: k.func("energyKEff")?,
            target_divergence: k.func("energyTargetDivergence")?,
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
    fn update_k_eff(&mut self, gpu: &Gpu, nut: &GpuScalarField, gas: &GasState) -> Result<()> {
        let m = self.m;
        fv::interpolate_linear(gpu, &self.fvk, &mut self.rho_face, gas.rho(), m)?;
        fv::interpolate_linear(gpu, &self.fvk, &mut self.nut_face, nut, m)?;

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

        field_ops::copy_field(gpu, &self.fldk, &mut self.k_eff_mag_sf.f, &self.k_eff_face.f, m.n_internal_faces)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.k_eff_mag_sf.f, &m.mag_sf, m.n_internal_faces)?;

        field_ops::copy_field(gpu, &self.fldk, &mut self.k_eff_mag_sf.bf, &self.k_eff_face.bf, m.n_boundary_faces)?;
        field_ops::multiply_field(gpu, &self.fldk, &mut self.k_eff_mag_sf.bf, &m.b_mag_sf, m.n_boundary_faces)
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
    /// this crate reads it with.
    pub fn correct(
        &mut self,
        gpu: &Gpu,
        phi: &GpuSurfaceScalarField,
        nut: &GpuScalarField,
        gas: &GasState,
    ) -> Result<SolverPerformance> {
        let m = self.m;
        let n = m.n_cells;
        if n == 0 {
            return Ok(SolverPerformance::default());
        }

        field_ops::store_old_time(gpu, &self.fldk, &mut self.t)?;

        self.refresh_rho_cp(gpu, gas)?;
        self.update_k_eff(gpu, nut, gas)?;
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
    use crate::mesh::{HostMesh, PatchKind};
    use crate::momentum::BuoyancyCoeffs;

    fn gpu() -> Option<crate::Gpu> {
        crate::Gpu::new(0).ok()
    }

    // ----------------------------------------------------------------------
    //  GasProperties / GasState
    // ----------------------------------------------------------------------

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

        for _ in 0..3 {
            e.correct(&g, &phi, &nut, &gas)?;
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

            for _ in 0..steps {
                e.correct(&g, &phi, &nut, &gas)?;
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
