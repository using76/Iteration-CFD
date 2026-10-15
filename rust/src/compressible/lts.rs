// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The local time step of a steady compressible run - SPEC-LIT §125.4, as
//! built in §125.7 (docs/17 CMP-13).
//!
//! ```text
//! dt_P = CFL * V_P^(1/3) / ( |u_P| + c_P )                                (125.2)
//! ```
//!
//! A steady run selects it by naming `localEuler` for the momentum AND the
//! energy equation. [`super::Compressible::new`] then builds both equations
//! on `Euler` and attaches a [`LocalStep`] to each: every ddt term of
//! (123.3), (124.1) and (125.1) takes the Euler coefficients `a_n = r_P`,
//! `a_0 = -r_P`, `a_00 = 0` with `r_P = 1/dt_P` its own cell's. The step is
//! formed once per pseudo step, at its top, from the cell `u` and `T` the
//! step starts from. At a converged fixed point every ddt difference is
//! zero, so the fixed point does not depend on `CFL` (§125.4).
//!
//! The four kernels are `cuda/compressible.cu`'s last: `cmpLtsRDeltaT`
//! forms `r_P`, and `cmpDdtRhoLocal`, `cmpPressureDdtLocal` and
//! `cmpEnergySourceLocal` are `tsDdtGeneralRho`, `cmpPressureDdt` and
//! `cmpEnergySource` with the per-cell coefficient, in the same expression
//! order, so a step field equal to `1/dt` everywhere is the Euler step.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §125.4 - (125.2) and the fixed-point argument
//!   ofgpu `SPEC-LIT.md` §125.7 - the local step as built, which is the
//!     contract of this file
//!   ofgpu `SPEC-LIT.md` §122.5 - (122.12)'s sound speed `c = sqrt(grs T)`
//!   Courant, Friedrichs & Lewy (1928), the acoustic Courant condition, and
//!     Blazek (2015) ch. 6, the spectral-radius form - both *not read*, as
//!     §125.4 states of them
//! No GPL-licensed source was consulted.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::ldu::GpuLduMatrix;
use crate::mesh::GpuMesh;
use crate::{Label, Scalar, Vec3};

use super::thermo::GasConstants;

#[cfg(test)]
mod tests {
    //! CMP-13's gate table of SPEC-LIT §125.7, written once in
    //! `tests/shared/compressible_lts_cases.rs` and included here and by
    //! `tests/compressible_lts.rs`, which runs the same tests in both builds.
    use crate as ofgpu;

    include!("../../tests/shared/compressible_lts_cases.rs");
}

/// A refusal in the one house format.
fn refuse(who: &str, msg: String) -> Error {
    Error::Refused(format!("compressible::lts::{who}: {msg}"))
}

/// The local step's one input: the acoustic Courant number of (125.2).
/// Read only when the momentum and energy equations name `localEuler`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LtsControls {
    /// `CFL` of (125.2), finite and above 0. *DESIGN* default 1, the
    /// acoustic Courant limit of an explicit scheme; the implicit pseudo
    /// steps tolerate more, and the fixed point does not depend on it.
    pub cfl: Scalar,
}

impl Default for LtsControls {
    fn default() -> Self {
        Self { cfl: 1.0 }
    }
}

impl LtsControls {
    /// `cfl` must be finite and above 0, or it is refused by name.
    pub fn validate(&self) -> Result<()> {
        if !self.cfl.is_finite() || !(self.cfl > 0.0) {
            return Err(refuse(
                "validate",
                format!(
                    "cfl is {}; the local step (125.2) needs a finite Courant number \
                     above 0 (SPEC-LIT §125.4). Alternative: 1",
                    self.cfl
                ),
            ));
        }
        Ok(())
    }
}

/// (125.2) in f64, the host twin of `cmpLtsRDeltaT`:
/// `dt = cfl V^(1/3) / (|u| + c)`.
pub fn host_delta_t(cfl: f64, v: f64, u: [f64; 3], c: f64) -> f64 {
    let um = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    cfl * v.cbrt() / (um + c)
}

/// The four kernels of §125.7, resolved once.
pub struct LtsKernels {
    r_delta_t: CudaFunction,
    ddt_rho: CudaFunction,
    pressure_ddt: CudaFunction,
    energy_source: CudaFunction,
}

impl LtsKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let c = KernelSet::new(gpu, crate::kernels::COMPRESSIBLE)?;
        Ok(Self {
            r_delta_t: c.func("cmpLtsRDeltaT")?,
            ddt_rho: c.func("cmpDdtRhoLocal")?,
            pressure_ddt: c.func("cmpPressureDdtLocal")?,
            energy_source: c.func("cmpEnergySourceLocal")?,
        })
    }
}

/// One equation's per-cell reciprocal step `r_P = 1/dt_P` and the kernels
/// that read it. Allocated once, when [`super::Compressible::new`] attaches
/// it; nothing below allocates.
pub struct LocalStep {
    /// `r_P = (|u_P| + c_P) / (CFL V_P^(1/3))` [1/s].
    pub r_delta_t: DevBuf<Scalar>,
    k: LtsKernels,
}

fn expect_len<T>(b: &DevBuf<T>, n: usize, who: &str, what: &str) -> Result<()> {
    if b.len() < n {
        return Err(Error::Config(format!(
            "compressible::lts::{who}: {what} holds {} entries and the mesh has {n} cells",
            b.len()
        )));
    }
    Ok(())
}

impl LocalStep {
    /// A step field of `n_cells` entries, zero until [`Self::update`].
    pub fn new(gpu: &Gpu, m: &GpuMesh) -> Result<Self> {
        Ok(Self { r_delta_t: gpu.zeros(m.n_cells.max(1))?, k: LtsKernels::new(gpu)? })
    }

    /// (125.2) on every cell, from the cell values of `u` and `T`.
    pub fn update(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        gas: &GasConstants,
        cfl: Scalar,
        u: &DevBuf<Vec3>,
        t: &DevBuf<Scalar>,
    ) -> Result<()> {
        let n = m.n_cells;
        if n == 0 {
            return Ok(());
        }
        expect_len(u, n, "update", "u")?;
        expect_len(t, n, "update", "T")?;
        let nl = n as Label;
        let grs = gas.grs;
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.r_delta_t)
                .arg(&mut self.r_delta_t)
                .arg(&m.v)
                .arg(u)
                .arg(t)
                .arg(&grs)
                .arg(&cfl)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// `r` copied from another equation's step field, so the equations of
    /// one pseudo step share one step.
    pub fn copy_from(&mut self, gpu: &Gpu, other: &LocalStep) -> Result<()> {
        gpu.stream().memcpy_dtod(&other.r_delta_t, &mut self.r_delta_t)?;
        Ok(())
    }

    /// `sign · d(rho psi)/dt`, Euler with the per-cell step: the local twin
    /// of `Ddt::add_rho`'s `tsDdtGeneralRho`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_ddt_rho(
        &self,
        gpu: &Gpu,
        a: &mut GpuLduMatrix,
        m: &GpuMesh,
        rho: &DevBuf<Scalar>,
        rho0: &DevBuf<Scalar>,
        psi0: &DevBuf<Scalar>,
        sign: Scalar,
    ) -> Result<()> {
        let n = m.n_cells;
        if n == 0 {
            return Ok(());
        }
        for (b, what) in [(rho, "rho"), (rho0, "rho0"), (psi0, "psi0")] {
            expect_len(b, n, "add_ddt_rho", what)?;
        }
        let nl = n as Label;
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.ddt_rho)
                .arg(&mut a.diag)
                .arg(&mut a.source)
                .arg(&m.v)
                .arg(&self.r_delta_t)
                .arg(rho)
                .arg(rho0)
                .arg(psi0)
                .arg(&sign)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// (124.7)-(124.8)'s time term with the per-cell step: the local twin of
    /// `cmpPressureDdt`, launched with the same `sign`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_pressure_ddt(
        &self,
        gpu: &Gpu,
        a: &mut GpuLduMatrix,
        m: &GpuMesh,
        psi: &DevBuf<Scalar>,
        rho: &DevBuf<Scalar>,
        p_star: &DevBuf<Scalar>,
        rho0: &DevBuf<Scalar>,
        sign: Scalar,
    ) -> Result<()> {
        let n = m.n_cells;
        if n == 0 {
            return Ok(());
        }
        for (b, what) in [(psi, "psi"), (rho, "rho"), (p_star, "pStar"), (rho0, "rho0")] {
            expect_len(b, n, "add_pressure_ddt", what)?;
        }
        let nl = n as Label;
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.pressure_ddt)
                .arg(&mut a.diag)
                .arg(&mut a.source)
                .arg(&m.v)
                .arg(&self.r_delta_t)
                .arg(psi)
                .arg(rho)
                .arg(p_star)
                .arg(rho0)
                .arg(&sign)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// (125.3)'s cell source `rcp V (dp/dt - d(rho K)/dt + rho g . u)` with
    /// the per-cell step: the local twin of `cmpEnergySource`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_energy_source(
        &self,
        gpu: &Gpu,
        source: &mut DevBuf<Scalar>,
        m: &GpuMesh,
        rho: (&DevBuf<Scalar>, &DevBuf<Scalar>),
        p: (&DevBuf<Scalar>, &DevBuf<Scalar>),
        u: (&DevBuf<Vec3>, &DevBuf<Vec3>),
        g: Vec3,
        rcp: Scalar,
    ) -> Result<()> {
        let n = m.n_cells;
        if n == 0 {
            return Ok(());
        }
        for (b, what) in [(rho.0, "rho"), (rho.1, "rho0"), (p.0, "p"), (p.1, "p0")] {
            expect_len(b, n, "add_energy_source", what)?;
        }
        expect_len(u.0, n, "add_energy_source", "u")?;
        expect_len(u.1, n, "add_energy_source", "u0")?;
        let nl = n as Label;
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.energy_source)
                .arg(source)
                .arg(&m.v)
                .arg(&self.r_delta_t)
                .arg(rho.0)
                .arg(rho.1)
                .arg(p.0)
                .arg(p.1)
                .arg(u.0)
                .arg(u.1)
                .arg(&g.x)
                .arg(&g.y)
                .arg(&g.z)
                .arg(&rcp)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }
}
