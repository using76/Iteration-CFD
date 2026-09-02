// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Time schemes: which one the case asked for, and what it puts in the matrix.
//!
//! Written from:
//!   Crank & Nicolson, *Proc. Camb. Phil. Soc.* 43 (1947) 50-67
//!   Ferziger & Perić, *Computational Methods for Fluid Dynamics*, §6.3
//!   Patankar, *Numerical Heat Transfer and Fluid Flow* (1980) §4.2
//!   ofgpu `SPEC-LIT.md` §13.1, §13.2, §13.3, §13.4. §13.2's smoothing ratio,
//!     sweep count and damping are marked *DESIGN* there and are ours; so is
//!     the reading of the `CrankNicolson <c>` coefficient documented on
//!     [`DdtScheme`].
//! No GPL-licensed source was consulted.
//!
//! # Why this module exists
//!
//! `ddtSchemes` used to be reduced to a single boolean - "does the entry
//! contain the word steadyState" - which turned `backward`,
//! `CrankNicolson 0.9`, `localEuler` and `bounded backward` all into
//! first-order Euler with nothing printed. That is the silent substitution
//! `SPEC-LIT` §13.4 forbids. [`DdtScheme::parse`] returns the scheme *and* its
//! coefficient, and refuses what it cannot honour.
//!
//! # The one implicit form
//!
//! Euler and BDF2, at constant or variable `Δt`, are all
//!
//! ```text
//! d(psi)/dt ≈ a_n·psi^n + a_0·psi^{n-1} + a_00·psi^{n-2}
//! ```
//!
//! so [`DdtCoeffs`] carries three numbers and `cuda/timescheme.cu` has one
//! kernel. The device cannot tell which scheme produced the coefficients,
//! which is exactly why an adaptive run cannot quietly fall back to the
//! constant-`Δt` BDF2 formula: there is no constant-`Δt` code path to fall
//! back to.
//!
//! The theta method is not of that form - it weights the *spatial* operator
//! rather than the time derivative - so it is a separate transformation
//! ([`apply_theta`]) applied to the assembled matrix before the Euler ddt goes
//! in.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::GpuSurfaceScalarField;
use crate::ldu::GpuLduMatrix;
use crate::ldu_ops::{self, LduKernels};
use crate::mesh::GpuMesh;
use crate::{Label, Scalar};

// ==========================================================================
//  §13.4  Which scheme, and its coefficient
// ==========================================================================

/// The `ddtSchemes` entry, parsed.
///
/// # `CrankNicolson <c>`
///
/// `SPEC-LIT` §13.1 parameterises the family by `theta`, says `theta = 1` is
/// Euler and `theta = 1/2` is Crank-Nicolson, and then recommends
/// "off-centring towards Euler (`theta ≈ 0.9`)" in the same breath as naming
/// the dictionary entry `CrankNicolson <c>`. So **`c` is read as `theta`
/// directly**. That is a reading of the specification, not something the
/// literature fixes, and it is printed at start-up
/// ([`crate::io::case::print_effective_settings`]) precisely so that nobody
/// has to guess which convention a run used.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DdtScheme {
    /// No time derivative at all; under-relaxation carries the iteration
    /// instead (Patankar 1980 §6.7).
    SteadyState,
    /// First-order backward difference (`SPEC-LIT` §3.3).
    Euler,
    /// Second-order backward difference, BDF2 (`SPEC-LIT` §3.3, §13.3).
    Backward,
    /// The theta method (`SPEC-LIT` §13.1); the payload is `theta`.
    CrankNicolson(Scalar),
    /// Local time stepping (`SPEC-LIT` §13.2): a steady solve driven by a
    /// per-cell pseudo time step.
    LocalEuler,
}

impl DdtScheme {
    /// Classify a raw `ddtSchemes` value, `SPEC-LIT` §13.4:
    ///
    /// ```text
    /// recognised and implemented   -> use it
    /// recognised, not implemented  -> Error naming the setting and what is available
    /// not recognised               -> Error naming the setting
    /// ```
    ///
    /// The `bounded` wrapper is accepted and ignored: it constrains the
    /// *convection* operator (`SPEC-LIT` §3.1's `- Sp(div(phi), psi)` term),
    /// not the time derivative, and `bounded backward` is still BDF2 in time.
    pub fn parse(raw: &str) -> Result<Self> {
        // The entry is a token list: an optional `bounded` wrapper, the scheme
        // name, and for CrankNicolson a coefficient.
        let toks: Vec<&str> = raw
            .split(|c: char| c.is_whitespace() || c == ';')
            .filter(|t| !t.is_empty() && *t != "bounded")
            .collect();

        let Some(name) = toks.first().copied() else {
            return Err(Error::Config(
                "ddtSchemes: the entry is empty; expected one of steadyState, \
                 Euler, backward, CrankNicolson <theta>, localEuler"
                    .to_string(),
            ));
        };

        match name {
            "steadyState" => Ok(Self::SteadyState),
            "Euler" | "euler" => Ok(Self::Euler),
            "backward" => Ok(Self::Backward),
            "localEuler" | "LocalEuler" => Ok(Self::LocalEuler),

            "CrankNicolson" | "CrankNicholson" => {
                let Some(c) = toks.get(1) else {
                    return Err(Error::Config(
                        "ddtSchemes: `CrankNicolson` needs its off-centring \
                         coefficient, as in `CrankNicolson 0.9`; theta = 1 is \
                         Euler and theta = 0.5 is pure Crank-Nicolson \
                         (SPEC-LIT 13.1)"
                            .to_string(),
                    ));
                };
                let theta: Scalar = c.parse::<f64>().map(|v| v as Scalar).map_err(|_| {
                    Error::Config(format!(
                        "ddtSchemes: `CrankNicolson {c}` - `{c}` is not a number"
                    ))
                })?;
                if !(theta > 0.0 && theta <= 1.0) {
                    return Err(Error::Config(format!(
                        "ddtSchemes: `CrankNicolson {theta}` - theta must lie \
                         in (0, 1]; 1 is Euler implicit and 0.5 is pure \
                         Crank-Nicolson (SPEC-LIT 13.1)"
                    )));
                }
                Ok(Self::CrankNicolson(theta))
            }

            // Recognised names of schemes this solver does not have. Naming
            // them separately is the whole point of SPEC-LIT 13.4: the user
            // gets told the request was understood and cannot be served,
            // rather than getting Euler and no message.
            "CoEuler" | "SLTS" | "bounded" => Err(Error::Config(format!(
                "ddtSchemes: `{name}` is a recognised OpenFOAM scheme that \
                 ofgpu does not implement. Available: steadyState, Euler, \
                 backward, CrankNicolson <theta>, localEuler. \
                 Run with -permissive to substitute Euler and carry on."
            ))),

            other => Err(Error::Config(format!(
                "ddtSchemes: `{other}` is not a time scheme this solver \
                 recognises. Available: steadyState, Euler, backward, \
                 CrankNicolson <theta>, localEuler. \
                 Run with -permissive to substitute Euler and carry on."
            ))),
        }
    }

    /// What to print in the start-up banner.
    pub fn describe(&self) -> String {
        match self {
            Self::SteadyState => "steadyState (no ddt term)".to_string(),
            Self::Euler => "Euler (1st order implicit)".to_string(),
            Self::Backward => "backward / BDF2 (2nd order implicit)".to_string(),
            Self::CrankNicolson(t) => {
                format!("CrankNicolson, theta = {t} (2nd order at theta = 0.5)")
            }
            Self::LocalEuler => "localEuler (local time stepping, steady)".to_string(),
        }
    }

    /// Does this scheme read `psi^{n-2}`?
    ///
    /// A field carrying only one old level cannot support BDF2 whatever the
    /// kernel can compute (`SPEC-LIT` §13.3), so this is what a driver checks
    /// before promising second order.
    pub fn needs_second_old_level(&self) -> bool {
        matches!(self, Self::Backward)
    }

    /// Is the run steady - no physical time, convergence by iteration?
    ///
    /// `localEuler` counts as steady: its time step is a preconditioner, not a
    /// time (`SPEC-LIT` §13.2).
    pub fn is_steady(&self) -> bool {
        matches!(self, Self::SteadyState | Self::LocalEuler)
    }

    /// Reconcile the scheme with the older `steady` boolean.
    ///
    /// `steady` is derived from the scheme by the case reader, so the two agree
    /// for anything that came out of an `fvSchemes`. They can still disagree
    /// for a caller that builds its controls in code and sets only `steady` -
    /// which is most of this crate's own tests, and any embedder written
    /// against the API as it was. In that case the explicit transient flag
    /// wins over the *default* scheme, because a caller who never touched
    /// `ddtSchemes` is still entitled to the first-order default rather than to
    /// no time derivative at all.
    ///
    /// A scheme that was named explicitly is never overridden: this only
    /// promotes `steadyState` to `Euler`, never the other way.
    pub fn reconciled(self, steady: bool) -> Self {
        match self {
            Self::SteadyState if !steady => Self::Euler,
            other => other,
        }
    }

    /// `theta`, for the schemes that weight the spatial operator. Every other
    /// scheme is fully implicit, i.e. `theta = 1`.
    pub fn theta(&self) -> Scalar {
        match self {
            Self::CrankNicolson(t) => *t,
            _ => 1.0,
        }
    }

    /// The three implicit coefficients of `SPEC-LIT` §13.3.
    ///
    /// `step` is the number of time steps ALREADY COMPLETED, so `step == 0` is
    /// the first step of a run. BDF2 degrades to Euler there because
    /// `psi^{n-2}` does not exist yet (`SPEC-LIT` §3.3) - which costs one step
    /// of first-order error and is the standard start-up, not a silent
    /// substitution: it is unavoidable and it is documented here.
    ///
    /// **Variable time step.** `SPEC-LIT` §13.3, with `r = dt_n/dt_{n-1}`:
    ///
    /// ```text
    /// d(psi)/dt = [ (1+2r)/(1+r) psi^n - (1+r) psi^{n-1}
    ///               + r²/(1+r) psi^{n-2} ] / dt_n
    /// ```
    ///
    /// which is `(3/2, -2, 1/2)/dt` at `r = 1`. The general form is what is
    /// implemented, so an adaptive run does not silently drop to first order.
    pub fn coeffs(&self, dt: Scalar, dt_old: Scalar, step: u64) -> Result<DdtCoeffs> {
        match self {
            // A steady scheme writes no ddt at all; the caller must not reach
            // here, but returning zeros is the honest answer if it does.
            Self::SteadyState | Self::LocalEuler => Ok(DdtCoeffs::ZERO),

            Self::Euler | Self::CrankNicolson(_) => {
                let r_dt = reciprocal(dt, "deltaT")?;
                Ok(DdtCoeffs {
                    a_n: r_dt,
                    a_0: -r_dt,
                    a_00: 0.0,
                })
            }

            Self::Backward => {
                let r_dt = reciprocal(dt, "deltaT")?;
                if step == 0 {
                    return Ok(DdtCoeffs {
                        a_n: r_dt,
                        a_0: -r_dt,
                        a_00: 0.0,
                    });
                }
                let dt_old = if dt_old > 0.0 { dt_old } else { dt };
                let r = dt / dt_old;
                let s = 1.0 + r;
                Ok(DdtCoeffs {
                    a_n: (1.0 + 2.0 * r) / s * r_dt,
                    a_0: -s * r_dt,
                    a_00: (r * r) / s * r_dt,
                })
            }
        }
    }
}

fn reciprocal(dt: Scalar, what: &str) -> Result<Scalar> {
    if !(dt > 0.0) || !dt.is_finite() {
        return Err(Error::Config(format!(
            "{what} is {dt}; a transient time scheme needs a positive, finite \
             time step"
        )));
    }
    Ok(1.0 / dt)
}

/// `d(psi)/dt ≈ a_n·psi^n + a_0·psi^{n-1} + a_00·psi^{n-2}`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DdtCoeffs {
    pub a_n: Scalar,
    pub a_0: Scalar,
    pub a_00: Scalar,
}

impl DdtCoeffs {
    pub const ZERO: Self = Self {
        a_n: 0.0,
        a_0: 0.0,
        a_00: 0.0,
    };

    /// A consistent time derivative annihilates a constant field: the three
    /// coefficients must sum to zero. Cheap, and it catches a mistyped
    /// variable-`Δt` formula immediately - which is the failure mode that
    /// otherwise shows up only as a convergence order of 1 in a study nobody
    /// ran.
    pub fn is_consistent(&self) -> bool {
        let scale = self.a_n.abs().max(self.a_0.abs()).max(self.a_00.abs());
        if scale == 0.0 {
            return true;
        }
        (self.a_n + self.a_0 + self.a_00).abs() <= 1e-12 * scale
    }
}

// ==========================================================================
//  Time-step bookkeeping
// ==========================================================================

/// `dt`, `dt_{n-1}` and how many steps have been taken.
///
/// Exists so the two facts BDF2 needs - the step ratio and whether
/// `psi^{n-2}` is real yet - live in one place rather than being rediscovered
/// by every driver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeState {
    pub dt: Scalar,
    pub dt_old: Scalar,
    /// Steps completed. Zero during the first step.
    pub step: u64,
    pub time: Scalar,
}

impl TimeState {
    pub fn new(dt: Scalar) -> Self {
        Self {
            dt,
            dt_old: dt,
            step: 0,
            time: 0.0,
        }
    }

    /// Close the step just taken and open the next one at `next_dt`.
    ///
    /// Call ONCE per time step, next to the field rotation
    /// ([`crate::field_ops::store_old_time`]) - the two are the same event and
    /// separating them is how BDF2 quietly becomes first order.
    pub fn advance(&mut self, next_dt: Scalar) {
        self.time += self.dt;
        self.dt_old = self.dt;
        self.dt = next_dt;
        self.step += 1;
    }

    pub fn coeffs(&self, scheme: DdtScheme) -> Result<DdtCoeffs> {
        scheme.coeffs(self.dt, self.dt_old, self.step)
    }
}

// ==========================================================================
//  Kernels
// ==========================================================================

pub struct TimeKernels {
    ddt_general: CudaFunction,
    ddt_general_rho: CudaFunction,
    ddt_rho_continuity: CudaFunction,
    ddt_local: CudaFunction,
    theta_cells: CudaFunction,
    scale: CudaFunction,
    lts_r_delta_t: CudaFunction,
    lts_smooth: CudaFunction,
    lts_damp: CudaFunction,
}

impl TimeKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::TIMESCHEME)?;
        Ok(Self {
            ddt_general: k.func("tsDdtGeneral")?,
            ddt_general_rho: k.func("tsDdtGeneralRho")?,
            ddt_rho_continuity: k.func("tsDdtRhoContinuity")?,
            ddt_local: k.func("tsDdtLocal")?,
            theta_cells: k.func("tsThetaCells")?,
            scale: k.func("tsScale")?,
            lts_r_delta_t: k.func("tsLtsRDeltaT")?,
            lts_smooth: k.func("tsLtsSmooth")?,
            lts_damp: k.func("tsLtsDamp")?,
        })
    }
}

fn check_matrix(a: &GpuLduMatrix, m: &GpuMesh, who: &str) -> Result<()> {
    if a.n_cells != m.n_cells || a.n_internal_faces != m.n_internal_faces {
        return Err(Error::Config(format!(
            "{who}: matrix is {}x{} but the mesh is {}x{}",
            a.n_cells, a.n_internal_faces, m.n_cells, m.n_internal_faces
        )));
    }
    Ok(())
}

fn expect_len<T>(b: &DevBuf<T>, n: usize, what: &str) -> Result<()> {
    if b.len() < n {
        return Err(Error::Config(format!(
            "timescheme: `{what}` holds {} values, {n} were needed",
            b.len()
        )));
    }
    Ok(())
}

// ==========================================================================
//  §3.3 / §13.3  The implicit ddt
// ==========================================================================

/// Add `sign · d(psi)/dt` to the matrix, for any of Euler and BDF2 at any
/// step ratio.
///
/// ```text
/// diag[P]   += sign · V_P · a_n
/// source[P] -= sign · V_P · (a_0·psi0_P + a_00·psi00_P)
/// ```
///
/// `psi00` is read only when `a_00 != 0`, but it is not optional in the
/// signature: a caller that has no second old level should not be able to ask
/// for BDF2 and get Euler without noticing. `SPEC-LIT` §13.3.
pub fn fvm_ddt(
    gpu: &Gpu,
    k: &TimeKernels,
    a: &mut GpuLduMatrix,
    m: &GpuMesh,
    psi0: &DevBuf<Scalar>,
    psi00: &DevBuf<Scalar>,
    c: DdtCoeffs,
    sign: Scalar,
) -> Result<()> {
    check_matrix(a, m, "fvm_ddt")?;
    let n = m.n_cells;
    if n == 0 || c == DdtCoeffs::ZERO {
        return Ok(());
    }
    expect_len(psi0, n, "psi0")?;
    expect_len(psi00, n, "psi00")?;
    if !c.is_consistent() {
        return Err(Error::Config(format!(
            "fvm_ddt: the coefficients {c:?} do not sum to zero, so the \
             discrete time derivative of a constant field is not zero"
        )));
    }
    let nl = n as Label;

    unsafe {
        gpu.stream()
            .launch_builder(&k.ddt_general)
            .arg(&mut a.diag)
            .arg(&mut a.source)
            .arg(&m.v)
            .arg(psi0)
            .arg(psi00)
            .arg(&c.a_n)
            .arg(&c.a_0)
            .arg(&c.a_00)
            .arg(&sign)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `sign · d(rho psi)/dt`. Each level carries its own density, which is what
/// makes the discrete form conserve `rho·psi` rather than `psi`.
#[allow(clippy::too_many_arguments)]
pub fn fvm_ddt_rho(
    gpu: &Gpu,
    k: &TimeKernels,
    a: &mut GpuLduMatrix,
    m: &GpuMesh,
    rho: &DevBuf<Scalar>,
    rho0: &DevBuf<Scalar>,
    rho00: &DevBuf<Scalar>,
    psi0: &DevBuf<Scalar>,
    psi00: &DevBuf<Scalar>,
    c: DdtCoeffs,
    sign: Scalar,
) -> Result<()> {
    check_matrix(a, m, "fvm_ddt_rho")?;
    let n = m.n_cells;
    if n == 0 || c == DdtCoeffs::ZERO {
        return Ok(());
    }
    for (b, what) in [
        (rho, "rho"),
        (rho0, "rho0"),
        (rho00, "rho00"),
        (psi0, "psi0"),
        (psi00, "psi00"),
    ] {
        expect_len(b, n, what)?;
    }
    let nl = n as Label;

    unsafe {
        gpu.stream()
            .launch_builder(&k.ddt_general_rho)
            .arg(&mut a.diag)
            .arg(&mut a.source)
            .arg(&m.v)
            .arg(rho)
            .arg(rho0)
            .arg(rho00)
            .arg(psi0)
            .arg(psi00)
            .arg(&c.a_n)
            .arg(&c.a_0)
            .arg(&c.a_00)
            .arg(&sign)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// Euler implicit with a per-cell reciprocal step - `SPEC-LIT` §13.2.
pub fn fvm_ddt_local(
    gpu: &Gpu,
    k: &TimeKernels,
    a: &mut GpuLduMatrix,
    m: &GpuMesh,
    r_delta_t: &DevBuf<Scalar>,
    psi0: &DevBuf<Scalar>,
    sign: Scalar,
) -> Result<()> {
    check_matrix(a, m, "fvm_ddt_local")?;
    let n = m.n_cells;
    if n == 0 {
        return Ok(());
    }
    expect_len(r_delta_t, n, "rDeltaT")?;
    expect_len(psi0, n, "psi0")?;
    let nl = n as Label;

    unsafe {
        gpu.stream()
            .launch_builder(&k.ddt_local)
            .arg(&mut a.diag)
            .arg(&mut a.source)
            .arg(&m.v)
            .arg(r_delta_t)
            .arg(psi0)
            .arg(&sign)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  §13.1  The theta method
// ==========================================================================

/// Turn an assembled spatial system into its theta-weighted form.
///
/// With `M` the spatial matrix and `b` its source, the semi-discrete system is
/// `V dpsi/dt = b - M·psi = L(psi)`, and
///
/// ```text
/// V(psi^n - psi^{n-1})/dt = theta·L(psi^n) + (1 - theta)·L(psi^{n-1})
/// ```
///
/// rearranges to
///
/// ```text
/// M' = theta·M
/// b' = b - (1 - theta)·M·psi^{n-1}
/// ```
///
/// after which the ordinary Euler [`fvm_ddt`] is added. `theta = 1` scales by
/// one and subtracts nothing, so asking this for Euler really gives Euler,
/// bit for bit.
///
/// `SPEC-LIT` §13.1 marks re-applying the *current* operator to `psi^{n-1}`
/// (rather than keeping the previous step's matrix) as the *DESIGN* choice,
/// because a second matrix would double the largest allocation in the solver.
/// The consequence, stated plainly: the explicit half uses `b^n`, not
/// `b^{n-1}`, so a source that varies inside a step is lagged to first order.
/// A source that does not vary - the usual case - is second order.
///
/// **Call order.** After the spatial operators and after
/// [`crate::ldu_ops::add_boundary_contributions`], before the ddt term. Called
/// before the fold it would scale a diagonal the boundary has not reached yet;
/// called after the ddt it would scale `V/dt` as well and integrate nothing at
/// all.
#[allow(clippy::too_many_arguments)]
pub fn apply_theta(
    gpu: &Gpu,
    k: &TimeKernels,
    lk: &LduKernels,
    a: &mut GpuLduMatrix,
    m: &GpuMesh,
    psi0: &DevBuf<Scalar>,
    scratch: &mut DevBuf<Scalar>,
    theta: Scalar,
) -> Result<()> {
    check_matrix(a, m, "apply_theta")?;
    if !(theta > 0.0 && theta <= 1.0) {
        return Err(Error::Config(format!(
            "apply_theta: theta = {theta}; SPEC-LIT 13.1 defines the family on \
             (0, 1] with 1 = Euler and 0.5 = Crank-Nicolson"
        )));
    }
    let n = m.n_cells;
    if n == 0 {
        return Ok(());
    }
    expect_len(psi0, n, "psi0")?;
    expect_len(scratch, n, "scratch")?;

    // M·psi^{n-1}, with the matrix as it stands - i.e. BEFORE any scaling.
    ldu_ops::amul(gpu, lk, scratch, psi0, a, m)?;

    let nl = n as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.theta_cells)
            .arg(&mut a.diag)
            .arg(&mut a.source)
            .arg(&*scratch)
            .arg(&theta)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }

    // The off-diagonals and the coupled-interface coefficients are part of M
    // and are scaled with it.
    scale_in_place(gpu, k, &mut a.upper, theta, m.n_internal_faces)?;
    scale_in_place(gpu, k, &mut a.lower, theta, m.n_internal_faces)?;
    scale_in_place(gpu, k, &mut a.boundary_coeffs, theta, m.n_boundary_faces)?;

    Ok(())
}

fn scale_in_place(
    gpu: &Gpu,
    k: &TimeKernels,
    x: &mut DevBuf<Scalar>,
    factor: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let nl = n as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.scale)
            .arg(&mut *x)
            .arg(&factor)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  §13.2  Local time stepping
// ==========================================================================

/// Controls for the local time step.
///
/// ***DESIGN.*** `SPEC-LIT` §13.2 fixes the formula for `rDeltaT` and says the
/// smoothing ratio, the sweep count and the damping are ours. They are:
///
/// * `smoothing_ratio = 1.1`. One sweep may raise a cell to within 10% of its
///   largest neighbour, so the field can vary by at most that factor per cell
///   after enough sweeps. Tighter than this and the whole mesh is dragged down
///   to the smallest step, which is a global time step with extra steps.
/// * `n_sweeps = 4`. A value propagates one cell per sweep, so four sweeps
///   smooths over a four-cell halo. More sweeps cost a kernel each and change
///   nothing about the converged answer.
/// * `damping = 1.0`, i.e. off. Damping only matters when `phi` is still
///   swinging between outer iterations; a run that needs it can set it.
///
/// None of this is physical. The converged steady answer must not depend on
/// any of it, and `two_courant_numbers_give_the_same_steady_state` in this
/// file is the test that says so.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LtsControls {
    /// Target Courant number.
    pub co_max: Scalar,
    /// Largest step any cell may take, i.e. the floor under `rDeltaT`.
    pub dt_max: Scalar,
    pub smoothing_ratio: Scalar,
    pub n_sweeps: usize,
    pub damping: Scalar,
}

impl Default for LtsControls {
    fn default() -> Self {
        Self {
            co_max: 1.0,
            dt_max: 1.0,
            smoothing_ratio: 1.1,
            n_sweeps: 4,
            damping: 1.0,
        }
    }
}

impl LtsControls {
    fn validate(&self) -> Result<()> {
        if !(self.co_max > 0.0 && self.co_max.is_finite()) {
            return Err(Error::Config(format!(
                "maxCo is {}; local time stepping needs a positive Courant \
                 number (SPEC-LIT 13.2)",
                self.co_max
            )));
        }
        if !(self.dt_max > 0.0 && self.dt_max.is_finite()) {
            return Err(Error::Config(format!(
                "maxDeltaT is {}; it is the floor under rDeltaT and must be \
                 positive",
                self.dt_max
            )));
        }
        if !(self.smoothing_ratio > 1.0) {
            return Err(Error::Config(format!(
                "the LTS smoothing ratio is {}; it must exceed 1, or the sweep \
                 drags every cell to the smallest step in the mesh",
                self.smoothing_ratio
            )));
        }
        if !(self.damping > 0.0 && self.damping <= 1.0) {
            return Err(Error::Config(format!(
                "the LTS damping is {}; it relaxes rDeltaT towards its new \
                 value and must lie in (0, 1]",
                self.damping
            )));
        }
        Ok(())
    }
}

/// The local time step field, and the scratch its smoothing needs.
pub struct Lts {
    pub ctrl: LtsControls,
    /// `[n_cells]` the reciprocal local time step.
    pub r_delta_t: DevBuf<Scalar>,
    /// `[n_cells]` the previous iteration's, for the damping.
    r_delta_t_old: DevBuf<Scalar>,
    /// `[n_cells]` ping-pong buffer for the smoothing sweeps.
    scratch: DevBuf<Scalar>,
    first: bool,
}

impl Lts {
    pub fn new(gpu: &Gpu, m: &GpuMesh, ctrl: LtsControls) -> Result<Self> {
        ctrl.validate()?;
        let n = m.n_cells.max(1);
        Ok(Self {
            ctrl,
            r_delta_t: gpu.zeros(n)?,
            r_delta_t_old: gpu.zeros(n)?,
            scratch: gpu.zeros(n)?,
            first: true,
        })
    }

    /// Recompute `rDeltaT` from the current flux - `SPEC-LIT` §13.2:
    ///
    /// ```text
    /// rDeltaT_P = max( 1/dt_max , (1/2) Σ_f |phi_f| / (Co_max V_P) )
    /// ```
    ///
    /// then smooth, then damp. The whole thing is device-resident and involves
    /// no host decision, so it captures into a CUDA graph like everything else
    /// in the loop.
    pub fn update(
        &mut self,
        gpu: &Gpu,
        k: &TimeKernels,
        phi: &GpuSurfaceScalarField,
        m: &GpuMesh,
    ) -> Result<()> {
        let n = m.n_cells;
        if n == 0 {
            return Ok(());
        }
        let nl = n as Label;
        let r_dt_max = 1.0 / self.ctrl.dt_max;

        unsafe {
            gpu.stream()
                .launch_builder(&k.lts_r_delta_t)
                .arg(&mut self.r_delta_t)
                .arg(&phi.f)
                .arg(&phi.bf)
                .arg(&m.v)
                .arg(&m.cf_offset)
                .arg(&m.cf_face)
                .arg(&m.cf_own)
                .arg(&m.bcf_offset)
                .arg(&m.bcf_face)
                .arg(&self.ctrl.co_max)
                .arg(&r_dt_max)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }

        // Ping-pong so no sweep reads a value another block is writing. An odd
        // sweep count would leave the answer in `scratch`, so the loop always
        // runs an even number of launches by copying back.
        for _ in 0..self.ctrl.n_sweeps {
            unsafe {
                gpu.stream()
                    .launch_builder(&k.lts_smooth)
                    .arg(&mut self.scratch)
                    .arg(&self.r_delta_t)
                    .arg(&m.owner)
                    .arg(&m.neighbour)
                    .arg(&m.cf_offset)
                    .arg(&m.cf_face)
                    .arg(&m.cf_own)
                    .arg(&self.ctrl.smoothing_ratio)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
            std::mem::swap(&mut self.r_delta_t, &mut self.scratch);
        }

        if self.ctrl.damping < 1.0 && !self.first {
            unsafe {
                gpu.stream()
                    .launch_builder(&k.lts_damp)
                    .arg(&mut self.r_delta_t)
                    .arg(&self.r_delta_t_old)
                    .arg(&self.ctrl.damping)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }

        if self.ctrl.damping < 1.0 {
            gpu.stream()
                .memcpy_dtod(&self.r_delta_t, &mut self.r_delta_t_old)?;
        }
        self.first = false;
        Ok(())
    }
}

// ==========================================================================
//  The ddt term an equation actually carries
// ==========================================================================

/// The time derivative of one equation: the scheme, the step history, the
/// kernels, and - for `localEuler` - the local step field.
///
/// This is what a model owns so that `ddtSchemes` reaches the matrix. Before
/// it existed, every equation in the crate called `fvm_ddt_euler` with a
/// scalar `1/dt` that was zero when `steady`, so `backward`,
/// `CrankNicolson <c>` and `localEuler` all produced first-order Euler and
/// nothing said so.
pub struct Ddt {
    pub scheme: DdtScheme,
    pub state: TimeState,
    kernels: TimeKernels,
    lts: Option<Lts>,
}

impl Ddt {
    /// Build the ddt term for `scheme`, for an equation that carries implicit
    /// under-relaxation (`SPEC-LIT` §5.2) at a factor below 1 - which is every
    /// equation in this crate that predates this constructor, so this is the
    /// conservative default. See [`Self::new_with_relax`] for the condition
    /// under which `CrankNicolson` is actually reachable.
    pub fn new(
        gpu: &Gpu,
        m: &GpuMesh,
        scheme: DdtScheme,
        dt: Scalar,
        lts_ctrl: LtsControls,
    ) -> Result<Self> {
        Self::new_with_relax(gpu, m, scheme, dt, lts_ctrl, true)
    }

    /// Build the ddt term for `scheme`, gating `CrankNicolson` on whether the
    /// equation it will sit in is implicitly under-relaxed.
    ///
    /// **`CrankNicolson` is refused when `relaxed`**, and that is deliberate
    /// rather than an oversight: the theta method weights the *spatial*
    /// operator ([`apply_theta`]), so it has to be applied between the
    /// boundary fold and the ddt, and this crate's implicit under-relaxation
    /// ([`crate::ldu_ops::relax`]) sits in exactly that gap and must see the
    /// unrelaxed, unfolded diagonal. Applying `apply_theta` first and relaxing
    /// afterwards would relax the theta-scaled matrix instead - a different,
    /// unpublished scheme - so the two cannot coexist in one equation.
    ///
    /// `relaxed = false` is exactly the equations `SPEC-LIT` §5.2's
    /// relaxation pair excludes: a PISO/PIMPLE equation with its relaxation
    /// factor left at 1 (Issa 1986 §2 runs the predictor and every corrector
    /// unrelaxed - that is what makes it non-iterative), or any other
    /// equation a case has explicitly set to `relaxationFactors { ... 1; }`.
    /// There `apply_theta` is the *only* thing touching the diagonal in that
    /// gap, so the theta method is fully reachable - and this constructor
    /// is what lets it through.
    ///
    /// Silently running Euler instead of a refused `CrankNicolson` would be
    /// the very substitution `SPEC-LIT` §13.4 forbids, so the request is
    /// refused with a message that names it, names the condition
    /// (`relaxed`), and says where the theta method *is* available.
    /// `-permissive` downgrades it to Euler, and says so.
    pub fn new_with_relax(
        gpu: &Gpu,
        m: &GpuMesh,
        scheme: DdtScheme,
        dt: Scalar,
        lts_ctrl: LtsControls,
        relaxed: bool,
    ) -> Result<Self> {
        let scheme = match scheme {
            DdtScheme::CrankNicolson(_) if relaxed => crate::io::contract::unsupported_note(
                "ddtSchemes/default",
                "CrankNicolson",
                &["steadyState", "Euler", "backward", "localEuler"],
                "the theta method IS implemented (timescheme::apply_theta, SPEC-LIT 13.1) and IS reachable from an equation with no implicit under-relaxation (relaxationFactors == 1, as PISO/PIMPLE run) - it cannot be reached from an implicitly under-relaxed equation, because the theta weighting and the relaxation want the same slot in the assembly and the relaxation has to see the unweighted diagonal",
                "Euler",
                DdtScheme::Euler,
            )?,
            other => other,
        };

        let lts = if scheme == DdtScheme::LocalEuler {
            Some(Lts::new(gpu, m, lts_ctrl)?)
        } else {
            None
        };

        Ok(Self {
            scheme,
            state: TimeState::new(if dt > 0.0 { dt } else { 1.0 }),
            kernels: TimeKernels::new(gpu)?,
            lts,
        })
    }

    /// Does this term write anything at all?
    pub fn is_active(&self) -> bool {
        self.scheme != DdtScheme::SteadyState
    }

    /// Recompute the local time step from the current flux. A no-op unless
    /// the scheme is `localEuler`, so a caller may always call it.
    pub fn update_local_step(
        &mut self,
        gpu: &Gpu,
        phi: &GpuSurfaceScalarField,
        m: &GpuMesh,
    ) -> Result<()> {
        match self.lts.as_mut() {
            None => Ok(()),
            Some(lts) => lts.update(gpu, &self.kernels, phi, m),
        }
    }

    /// The local step field, for a caller that wants to report it.
    pub fn local_step(&self) -> Option<&DevBuf<Scalar>> {
        self.lts.as_ref().map(|l| &l.r_delta_t)
    }

    /// Add `sign · d(psi)/dt` to `a`, whichever scheme this is.
    ///
    /// `psi0` and `psi00` are `psi^{n-1}` and `psi^{n-2}`; both are required
    /// even for Euler, so that a caller with only one old level cannot ask for
    /// `backward` and be given Euler without noticing.
    pub fn add(
        &self,
        gpu: &Gpu,
        a: &mut GpuLduMatrix,
        m: &GpuMesh,
        psi0: &DevBuf<Scalar>,
        psi00: &DevBuf<Scalar>,
        sign: Scalar,
    ) -> Result<()> {
        match self.scheme {
            DdtScheme::SteadyState => Ok(()),

            DdtScheme::LocalEuler => {
                let Some(lts) = self.lts.as_ref() else {
                    return Err(Error::Config(
                        "localEuler was selected but the local step field was \
                         never built"
                            .to_string(),
                    ));
                };
                fvm_ddt_local(gpu, &self.kernels, a, m, &lts.r_delta_t, psi0, sign)
            }

            other => {
                let c = other.coeffs(self.state.dt, self.state.dt_old, self.state.step)?;
                fvm_ddt(gpu, &self.kernels, a, m, psi0, psi00, c, sign)
            }
        }
    }

    /// [`Self::add`] for `sign · d(rho psi)/dt` - `SPEC-LIT` §86.3.
    ///
    /// The same three coefficients [`Self::add`] uses, so a mass-weighted
    /// equation cannot be integrated by a different time scheme from the one
    /// `ddtSchemes` named for the rest of the run: the scheme is read off
    /// `self`, not off the call site.
    ///
    /// `localEuler` is **refused by name** (§86.7): its step is a per-cell
    /// preconditioner rather than a time derivative, so `d(rho psi)/dt` with
    /// a different `dt` in every cell conserves nothing, and the discrete
    /// continuity residual [`Self::rho_continuity`] forms would be a
    /// statement about the preconditioner. `steadyState` is a no-op, exactly
    /// as it is for [`Self::add`].
    pub fn add_rho(
        &self,
        gpu: &Gpu,
        a: &mut GpuLduMatrix,
        m: &GpuMesh,
        rho: &DevBuf<Scalar>,
        rho0: &DevBuf<Scalar>,
        rho00: &DevBuf<Scalar>,
        psi0: &DevBuf<Scalar>,
        psi00: &DevBuf<Scalar>,
        sign: Scalar,
    ) -> Result<()> {
        match self.scheme {
            DdtScheme::SteadyState => Ok(()),

            DdtScheme::LocalEuler => Err(Error::Config(
                "ddtSchemes named `localEuler` and this equation is                  mass-weighted (SPEC-LIT §86.3). The local step is a per-cell                  preconditioner, not a time derivative, so d(rho psi)/dt with                  a different dt in every cell conserves nothing. Alternative:                  `Euler` or `backward` for a transient run, `steadyState` for                  a steady one - both are honoured here."
                    .to_string(),
            )),

            other => {
                let c = other.coeffs(self.state.dt, self.state.dt_old, self.state.step)?;
                fvm_ddt_rho(gpu, &self.kernels, a, m, rho, rho0, rho00, psi0, psi00, c, sign)
            }
        }
    }

    /// `a_N rho + a_0 rho^{n-1} + a_00 rho^{n-2}` per cell - the ddt half of
    /// the DISCRETE continuity residual, `SPEC-LIT` (86.4).
    ///
    /// This is exactly what [`Self::add_rho`] puts into row `P` when the
    /// transported field is `1` everywhere, divided by `V_P`. It is the term
    /// §3.1's bounded correction never had to know about, because on a
    /// constant-density equation it is identically zero.
    ///
    /// Zero under `steadyState`, which is the same thing [`Self::add_rho`]
    /// contributes there.
    pub fn rho_continuity(
        &self,
        gpu: &Gpu,
        out: &mut DevBuf<Scalar>,
        m: &GpuMesh,
        rho: &DevBuf<Scalar>,
        rho0: &DevBuf<Scalar>,
        rho00: &DevBuf<Scalar>,
    ) -> Result<()> {
        let n = m.n_cells;
        expect_len(out, n, "out")?;
        if n == 0 {
            return Ok(());
        }
        let c = match self.scheme {
            DdtScheme::SteadyState => DdtCoeffs::ZERO,
            DdtScheme::LocalEuler => {
                return Err(Error::Config(
                    "SPEC-LIT §86.3 refuses `localEuler` on a mass-weighted                      equation; this residual is not defined for it.                      Alternative: `Euler`, `backward` or `steadyState`."
                        .to_string(),
                ))
            }
            other => other.coeffs(self.state.dt, self.state.dt_old, self.state.step)?,
        };
        for (b, what) in [(rho, "rho"), (rho0, "rho0"), (rho00, "rho00")] {
            expect_len(b, n, what)?;
        }
        let nl = n as Label;
        unsafe {
            gpu.stream()
                .launch_builder(&self.kernels.ddt_rho_continuity)
                .arg(&mut *out)
                .arg(rho)
                .arg(rho0)
                .arg(rho00)
                .arg(&c.a_n)
                .arg(&c.a_0)
                .arg(&c.a_00)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }
        Ok(())
    }

    /// Close the step just taken and open the next one.
    ///
    /// Call ONCE per time step, next to
    /// [`crate::field_ops::advance_time_levels`] - the two are the same event,
    /// and separating them is how BDF2 comes out first order.
    pub fn advance(&mut self, next_dt: Scalar) {
        self.state.advance(next_dt);
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
#[allow(clippy::unnecessary_cast)]
mod tests {
    use super::*;
    use crate::field::{BcKind, GpuScalarField};
    use crate::fv::{self, FvKernels};
    use crate::ldu_ops::LduKernels;
    use crate::mesh::{HostMesh, PatchKind};
    use crate::solver::{self, SolverControls, SolverKernels, SolverWorkspace};
    use crate::types::Vec3;

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    // ----------------------------------------------------------------------
    //  §13.4  Parsing
    // ----------------------------------------------------------------------

    #[test]
    fn every_implemented_scheme_parses_and_the_rest_are_errors() {
        assert_eq!(DdtScheme::parse("steadyState").unwrap(), DdtScheme::SteadyState);
        assert_eq!(DdtScheme::parse("Euler").unwrap(), DdtScheme::Euler);
        assert_eq!(DdtScheme::parse("backward").unwrap(), DdtScheme::Backward);
        // The `bounded` wrapper constrains convection, not the ddt.
        assert_eq!(
            DdtScheme::parse("bounded backward").unwrap(),
            DdtScheme::Backward
        );
        assert_eq!(DdtScheme::parse("localEuler").unwrap(), DdtScheme::LocalEuler);
        assert_eq!(
            DdtScheme::parse("CrankNicolson 0.9").unwrap(),
            DdtScheme::CrankNicolson(0.9)
        );

        // This is the whole point: none of these may quietly become Euler.
        for bad in [
            "CoEuler",
            "SLTS",
            "vanLeer",
            "",
            "CrankNicolson",
            "CrankNicolson banana",
            "CrankNicolson 1.5",
            "CrankNicolson 0",
        ] {
            assert!(
                DdtScheme::parse(bad).is_err(),
                "`{bad}` parsed instead of failing"
            );
        }
    }

    #[test]
    fn the_error_names_the_setting_and_the_alternatives() {
        // SPEC-LIT 13.4: an unimplemented name must say what it was and what
        // is available, or the message is no better than silence.
        let e = DdtScheme::parse("CoEuler").unwrap_err().to_string();
        assert!(e.contains("CoEuler"), "{e}");
        assert!(e.contains("backward"), "{e}");
        assert!(e.contains("permissive"), "{e}");
    }

    // ----------------------------------------------------------------------
    //  §13.3  The coefficients
    // ----------------------------------------------------------------------

    #[test]
    fn bdf2_reduces_to_the_textbook_coefficients_at_constant_dt() {
        let dt = 0.25 as Scalar;
        let c = DdtScheme::Backward.coeffs(dt, dt, 7).unwrap();
        assert!((c.a_n - 1.5 / dt).abs() < 1e-14);
        assert!((c.a_0 + 2.0 / dt).abs() < 1e-14);
        assert!((c.a_00 - 0.5 / dt).abs() < 1e-14);
        assert!(c.is_consistent());
    }

    #[test]
    fn bdf2_degrades_to_euler_on_the_first_step_only() {
        let dt = 0.25 as Scalar;
        let first = DdtScheme::Backward.coeffs(dt, dt, 0).unwrap();
        assert_eq!(first, DdtScheme::Euler.coeffs(dt, dt, 0).unwrap());
        let second = DdtScheme::Backward.coeffs(dt, dt, 1).unwrap();
        assert_ne!(second, first);
    }

    /// A quadratic in time is differentiated EXACTLY by BDF2, at any step
    /// ratio. That is the sharpest statement of second order there is, and it
    /// fails immediately if the variable-`dt` formula is wrong.
    #[test]
    fn variable_step_bdf2_differentiates_a_quadratic_exactly() {
        for &r in &[1.0 as Scalar, 1.5, 0.4, 3.0] {
            let dt = 0.3 as Scalar;
            let dt_old = dt / r;
            let c = DdtScheme::Backward.coeffs(dt, dt_old, 5).unwrap();
            assert!(c.is_consistent(), "r = {r}: {c:?}");

            // psi(t) = a + b t + c t², sampled at t = 0 (new), -dt, -dt-dt_old.
            let (a, b, q) = (0.7 as Scalar, -1.3 as Scalar, 2.1 as Scalar);
            let psi = |t: Scalar| a + b * t + q * t * t;
            let got = c.a_n * psi(0.0) + c.a_0 * psi(-dt) + c.a_00 * psi(-dt - dt_old);
            // d(psi)/dt at t = 0 is b.
            assert!(
                (got - b).abs() < 1e-11 * b.abs().max(1.0),
                "r = {r}: BDF2 gave {got}, exact {b}"
            );
        }
    }

    #[test]
    fn euler_differentiates_a_line_exactly_and_a_quadratic_only_to_first_order() {
        let dt = 0.2 as Scalar;
        let c = DdtScheme::Euler.coeffs(dt, dt, 3).unwrap();
        let line = |t: Scalar| 1.0 + 2.0 * t;
        assert!((c.a_n * line(0.0) + c.a_0 * line(-dt) - 2.0).abs() < 1e-14);
    }

    // ----------------------------------------------------------------------
    //  A mesh, and the pieces every device test below shares
    // ----------------------------------------------------------------------

    fn box_mesh(n: [usize; 3], d: Vec3) -> HostMesh {
        let (mut m, pts, faces) = crate::mesh::topology::tests::box_mesh(n, d);
        for p in m.patches.iter_mut() {
            p.kind = PatchKind::Generic;
            p.type_name = "patch".to_string();
        }
        m.build_cell_face_maps();
        m.compute_geometry(&pts, &faces).expect("geometry");
        m
    }

    struct Fx {
        gpu: Gpu,
        tk: TimeKernels,
        fvk: FvKernels,
        lduk: LduKernels,
        solk: SolverKernels,
        hm: HostMesh,
        m: GpuMesh,
    }

    fn fixture(n: [usize; 3]) -> Option<Fx> {
        let hm = box_mesh(n, Vec3::new(1.0 / n[0] as Scalar, 1.0 / n[1] as Scalar, 0.25));
        let gpu = gpu()?;
        let tk = TimeKernels::new(&gpu).ok()?;
        let fvk = FvKernels::new(&gpu).ok()?;
        let lduk = LduKernels::new(&gpu).ok()?;
        let solk = SolverKernels::new(&gpu).ok()?;
        let m = GpuMesh::upload(&gpu, &hm).ok()?;
        Some(Fx { gpu, tk, fvk, lduk, solk, hm, m })
    }

    /// A Dirichlet field whose faces hold `bf`.
    fn dirichlet(
        gpu: &Gpu,
        m: &GpuMesh,
        hm: &HostMesh,
        f: &[Scalar],
        bf: &[Scalar],
    ) -> Result<GpuScalarField> {
        let mut psi = GpuScalarField::zeros(gpu, m, "psi")?;
        gpu.write(&mut psi.f, f)?;
        gpu.write(&mut psi.f0, f)?;
        gpu.write(&mut psi.f00, f)?;
        gpu.write(&mut psi.bf, bf)?;
        gpu.write(&mut psi.fr, &vec![1.0 as Scalar; hm.n_boundary_faces])?;
        gpu.write(&mut psi.ref_value, bf)?;
        gpu.write(&mut psi.ref_grad, &vec![0.0 as Scalar; hm.n_boundary_faces])?;
        gpu.write(
            &mut psi.bc_kind,
            &vec![BcKind::FixedValue as Label; hm.n_boundary_faces],
        )?;
        Ok(psi)
    }

    // ----------------------------------------------------------------------
    //  §22  MMS in time: fix the mesh, refine dt
    //
    //  The reference is NOT another CFD code and not a finer run of the scheme
    //  under test. The spatial matrix is downloaded and the semi-discrete
    //  system
    //
    //      V dpsi/dt = b - M·psi
    //
    //  is integrated on the host with classical RK4 at a step 4096 times
    //  smaller. RK4 is fourth order and explicit; it shares no arithmetic with
    //  the implicit schemes it is judging, and at that step its own error is
    //  below the round-off floor of the comparison.
    // ----------------------------------------------------------------------

    struct DenseSystem {
        n: usize,
        v: Vec<f64>,
        diag: Vec<f64>,
        upper: Vec<f64>,
        lower: Vec<f64>,
        source: Vec<f64>,
        owner: Vec<Label>,
        neighbour: Vec<Label>,
    }

    impl DenseSystem {
        /// `out = b - M·psi`, the `L(psi)` of SPEC-LIT §13.1.
        fn l(&self, psi: &[f64]) -> Vec<f64> {
            let mut ax = vec![0.0f64; self.n];
            for c in 0..self.n {
                ax[c] = self.diag[c] * psi[c];
            }
            for f in 0..self.upper.len() {
                let (o, nb) = (self.owner[f] as usize, self.neighbour[f] as usize);
                ax[o] += self.upper[f] * psi[nb];
                ax[nb] += self.lower[f] * psi[o];
            }
            (0..self.n).map(|c| self.source[c] - ax[c]).collect()
        }

        /// `d(psi)/dt = L(psi)/V`, integrated with classical RK4.
        fn rk4(&self, psi0: &[f64], t_end: f64, n_steps: usize) -> Vec<f64> {
            let h = t_end / n_steps as f64;
            let mut y = psi0.to_vec();
            let f = |y: &[f64]| -> Vec<f64> {
                let l = self.l(y);
                (0..self.n).map(|c| l[c] / self.v[c]).collect()
            };
            for _ in 0..n_steps {
                let k1 = f(&y);
                let y2: Vec<f64> = (0..self.n).map(|c| y[c] + 0.5 * h * k1[c]).collect();
                let k2 = f(&y2);
                let y3: Vec<f64> = (0..self.n).map(|c| y[c] + 0.5 * h * k2[c]).collect();
                let k3 = f(&y3);
                let y4: Vec<f64> = (0..self.n).map(|c| y[c] + h * k3[c]).collect();
                let k4 = f(&y4);
                for c in 0..self.n {
                    y[c] += h / 6.0 * (k1[c] + 2.0 * k2[c] + 2.0 * k3[c] + k4[c]);
                }
            }
            y
        }
    }

    /// Assemble `-laplacian(1, psi) = 0` with Dirichlet faces, fold the
    /// boundary in, and hand back both the device matrix and a host copy.
    ///
    /// The source is time-INDEPENDENT, which is what the *DESIGN* of §13.1
    /// requires of a second-order theta run: the explicit half re-applies the
    /// current operator to the old level, so a source that moves inside a step
    /// is lagged. Saying that out loud here is better than a test that quietly
    /// avoids it.
    fn diffusion_system(fx: &Fx) -> Result<(GpuLduMatrix, DenseSystem, Vec<Scalar>)> {
        let (gpu, hm, m) = (&fx.gpu, &fx.hm, &fx.m);
        let n = hm.n_cells;

        // A smooth initial field, and boundary values held at zero, so the
        // solution relaxes towards zero and there is something to integrate.
        let pi = std::f64::consts::PI as Scalar;
        let psi0: Vec<Scalar> = (0..n)
            .map(|c| {
                let p = hm.c[c];
                (pi * p.x).sin() * (pi * p.y).sin() + 0.3 * (2.0 * pi * p.x).cos()
            })
            .collect();
        let bf = vec![0.0 as Scalar; hm.n_boundary_faces];
        let psi = dirichlet(gpu, m, hm, &psi0, &bf)?;

        // A DELIBERATELY slow diffusivity. With gamma = 1 this mesh relaxes
        // with a time constant of about 5 ms, so any dt a convergence study
        // can afford is several time constants long, the solution has decayed
        // to nothing by the end, and what gets measured is the stiff-limit
        // behaviour of an L-stable scheme rather than its order. Scaled down
        // by 100 the time constant is about half a second and the study sits
        // where the asymptotic order actually lives.
        const GAMMA: Scalar = 0.01;
        let gamma_h: Vec<Scalar> = hm.mag_sf.iter().map(|s| GAMMA * *s).collect();
        let b_gamma_h: Vec<Scalar> = hm.b_mag_sf.iter().map(|s| GAMMA * *s).collect();
        let gamma = gpu.upload(&gamma_h)?;
        let b_gamma = gpu.upload(&b_gamma_h)?;

        let mut a = GpuLduMatrix::new(gpu, m)?;
        a.zero(gpu)?;
        // `- laplacian(gamma, psi)` on the LHS, i.e. M = -L_diff.
        fv::fvm_laplacian(gpu, &fx.fvk, &mut a, m, &gamma, &b_gamma, &psi, -1.0)?;
        ldu_ops::add_boundary_contributions(gpu, &fx.lduk, &mut a, m)?;
        gpu.sync()?;

        let dense = DenseSystem {
            n,
            v: hm.v.iter().map(|x| *x as f64).collect(),
            diag: gpu.download(&a.diag)?.iter().map(|x| *x as f64).collect(),
            upper: gpu.download(&a.upper)?.iter().map(|x| *x as f64).collect(),
            lower: gpu.download(&a.lower)?.iter().map(|x| *x as f64).collect(),
            source: gpu.download(&a.source)?.iter().map(|x| *x as f64).collect(),
            owner: hm.owner.clone(),
            neighbour: hm.neighbour.clone(),
        };

        Ok((a, dense, psi0))
    }

    /// March the device schemes and return `max|psi(T) - reference|`.
    ///
    /// `template` is the spatial system, assembled once: it does not depend on
    /// time here, which is precisely the condition the §13.1 *DESIGN* needs
    /// for the theta method to be second order.
    ///
    /// Goes through the real [`Ddt`] object rather than the bare [`fvm_ddt`]
    /// function, `apply_theta` applied by the caller exactly as a PISO
    /// momentum equation would: this is what proves `CrankNicolson` is not
    /// merely correct in isolation but actually *reachable* end to end at
    /// `relaxed = false`, and still refused at `relaxed = true`.
    #[allow(clippy::too_many_arguments)]
    fn march_error(
        fx: &Fx,
        template: &GpuLduMatrix,
        scheme: DdtScheme,
        relaxed: bool,
        psi0: &[Scalar],
        reference: &[f64],
        t_end: Scalar,
        n_steps: usize,
        ratio: Scalar,
    ) -> Result<f64> {
        let (gpu, m, hm) = (&fx.gpu, &fx.m, &fx.hm);
        let n = hm.n_cells;

        // A geometric dt sequence summing to t_end, so `ratio != 1` exercises
        // the variable-step coefficients on every step but the first.
        let dts = step_sequence(t_end, n_steps, ratio);

        let mut psi_f = gpu.upload(psi0)?;
        let mut f0 = gpu.upload(psi0)?;
        let mut f00 = gpu.upload(psi0)?;
        let mut scratch: DevBuf<Scalar> = gpu.zeros(n)?;

        let mut a = GpuLduMatrix::new(gpu, m)?;
        let mut ws = SolverWorkspace::for_mesh(gpu, m)?;
        let ctrl = SolverControls {
            tolerance: 1e-15,
            rel_tol: 0.0,
            max_iter: 4000,
            ..SolverControls::default()
        };

        let mut ddt = Ddt::new_with_relax(gpu, m, scheme, dts[0], LtsControls::default(), relaxed)?;
        let mut ts = TimeState::new(dts[0]);

        for (i, &dt) in dts.iter().enumerate() {
            ts.dt = dt;
            if i == 0 {
                ts.dt_old = dt;
            }

            // Rotate psi^{n-2} <- psi^{n-1} <- psi, in that order, ONCE.
            gpu.stream().memcpy_dtod(&f0, &mut f00)?;
            gpu.stream().memcpy_dtod(&psi_f, &mut f0)?;

            // Restore the spatial system; the ddt is added on top of it.
            a.zero(gpu)?;
            gpu.stream().memcpy_dtod(&template.diag, &mut a.diag)?;
            gpu.stream().memcpy_dtod(&template.upper, &mut a.upper)?;
            gpu.stream().memcpy_dtod(&template.lower, &mut a.lower)?;
            gpu.stream().memcpy_dtod(&template.source, &mut a.source)?;

            if let DdtScheme::CrankNicolson(theta) = ddt.scheme {
                apply_theta(gpu, &fx.tk, &fx.lduk, &mut a, m, &f0, &mut scratch, theta)?;
            }

            ddt.state = ts;
            ddt.add(gpu, &mut a, m, &f0, &f00, 1.0)?;

            let perf = solver::solve_pbicgstab(gpu, &fx.solk, &mut psi_f, &a, m, &mut ws, &ctrl)?;
            assert!(
                perf.converged,
                "the linear solve stagnated at {:e}; a convergence study cannot                  see past a solver error floor",
                perf.final_residual
            );

            ts.advance(dt);
        }

        gpu.sync()?;
        let got = gpu.download(&psi_f)?;
        Ok((0..n)
            .map(|c| (got[c] as f64 - reference[c]).abs())
            .fold(0.0f64, f64::max))
    }

    /// `n` steps summing to `t_end`, ALTERNATING between `h` and `ratio*h`.
    ///
    /// Alternating rather than geometric, so that doubling `n` really does
    /// halve every step. A geometric sequence with a fixed ratio changes shape
    /// as it is refined - the largest step shrinks by far less than a factor of
    /// two - and a convergence study run on one measures the shape change as
    /// much as the scheme. The alternation still puts `dt_n/dt_{n-1}` at
    /// `ratio` and then `1/ratio` on every step but the first, which is what
    /// exercises the general variable-step formula of SPEC-LIT 13.3.
    fn step_sequence(t_end: Scalar, n: usize, ratio: Scalar) -> Vec<Scalar> {
        let raw: Vec<Scalar> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { ratio })
            .collect();
        let total: Scalar = raw.iter().copied().sum();
        raw.iter().map(|d| d * t_end / total).collect()
    }

    /// The gate itself, with no march involved: `CrankNicolson` is refused
    /// (with the note pointing at `apply_theta` and at the unrelaxed
    /// condition) when the equation is relaxed, and constructs cleanly - with
    /// its `theta` preserved - when it is not.
    #[test]
    fn crank_nicolson_is_refused_relaxed_and_reachable_unrelaxed() -> Result<()> {
        let Some(fx) = fixture([2, 2, 1]) else { return Ok(()) };
        let (gpu, m) = (&fx.gpu, &fx.m);

        let e = match Ddt::new_with_relax(
            gpu,
            m,
            DdtScheme::CrankNicolson(0.5),
            0.1,
            LtsControls::default(),
            true,
        ) {
            Ok(_) => panic!("CrankNicolson at relaxed = true should have been refused"),
            Err(e) => e.to_string(),
        };
        assert!(e.contains("CrankNicolson"), "{e}");
        assert!(e.contains("apply_theta"), "{e}");
        assert!(e.contains("permissive"), "{e}");

        // The plain `Ddt::new` is the same conservative default: still
        // refused, because it assumes `relaxed = true`.
        assert!(Ddt::new(gpu, m, DdtScheme::CrankNicolson(0.5), 0.1, LtsControls::default())
            .is_err());

        let ddt = Ddt::new_with_relax(
            gpu,
            m,
            DdtScheme::CrankNicolson(0.5),
            0.1,
            LtsControls::default(),
            false,
        )?;
        assert_eq!(ddt.scheme, DdtScheme::CrankNicolson(0.5));

        // Every other scheme is unaffected by `relaxed` either way.
        for scheme in [
            DdtScheme::SteadyState,
            DdtScheme::Euler,
            DdtScheme::Backward,
            DdtScheme::LocalEuler,
        ] {
            assert!(Ddt::new_with_relax(gpu, m, scheme, 0.1, LtsControls::default(), true).is_ok());
            assert!(
                Ddt::new_with_relax(gpu, m, scheme, 0.1, LtsControls::default(), false).is_ok()
            );
        }
        Ok(())
    }

    /// `SPEC-LIT` §22: BDF2 order, MMS in TIME - fix the mesh, refine `dt`.
    ///
    /// Run at a NON-UNIT step ratio, so what is measured is the general
    /// variable-`Δt` formula of §13.3 and not the constant-`Δt` special case.
    #[test]
    fn bdf2_is_second_order_in_time() -> Result<()> {
        let Some(fx) = fixture([6, 6, 1]) else { return Ok(()) };
        let (a, dense, psi0) = diffusion_system(&fx)?;

        let t_end = 0.4 as Scalar;
        let psi0_f: Vec<f64> = psi0.iter().map(|x| *x as f64).collect();
        let reference = dense.rk4(&psi0_f, t_end as f64, 40_000);

        let mut errs = Vec::new();
        for &n_steps in &[16usize, 32, 64] {
            errs.push(march_error(
                &fx,
                &a,
                DdtScheme::Backward,
                true,
                &psi0,
                &reference,
                t_end,
                n_steps,
                1.5,
            )?);
        }

        for w in errs.windows(2) {
            let order = (w[0] / w[1]).log2();
            assert!(
                order > 1.8,
                "BDF2 converged at order {order} (errors {errs:?}); \
                 SPEC-LIT 22 wants 2"
            );
        }
        Ok(())
    }

    /// `SPEC-LIT` §22: theta = 1/2 order, the same way - and, unlike the
    /// low-level check above, through the actual gated [`Ddt::new_with_relax`]
    /// at `relaxed = false`. This is the PISO/PIMPLE condition
    /// (`relaxationFactors == 1`): the theta method must be reachable there
    /// and still be second order once it is reached, not merely accepted.
    #[test]
    fn crank_nicolson_is_reachable_and_second_order_for_an_unrelaxed_equation() -> Result<()> {
        let Some(fx) = fixture([6, 6, 1]) else { return Ok(()) };
        let (a, dense, psi0) = diffusion_system(&fx)?;

        let t_end = 0.4 as Scalar;
        let psi0_f: Vec<f64> = psi0.iter().map(|x| *x as f64).collect();
        let reference = dense.rk4(&psi0_f, t_end as f64, 40_000);

        let mut errs = Vec::new();
        for &n_steps in &[16usize, 32, 64] {
            errs.push(march_error(
                &fx,
                &a,
                DdtScheme::CrankNicolson(0.5),
                false,
                &psi0,
                &reference,
                t_end,
                n_steps,
                1.0,
            )?);
        }

        for w in errs.windows(2) {
            let order = (w[0] / w[1]).log2();
            assert!(
                order > 1.8,
                "Crank-Nicolson converged at order {order} (errors {errs:?})"
            );
        }
        Ok(())
    }

    /// Euler must be FIRST order on the same problem. Without this the two
    /// tests above could pass on a mesh where the temporal error is below the
    /// noise floor and nothing is being measured at all.
    #[test]
    fn euler_is_only_first_order_on_the_same_problem() -> Result<()> {
        let Some(fx) = fixture([6, 6, 1]) else { return Ok(()) };
        let (a, dense, psi0) = diffusion_system(&fx)?;

        let t_end = 0.4 as Scalar;
        let psi0_f: Vec<f64> = psi0.iter().map(|x| *x as f64).collect();
        let reference = dense.rk4(&psi0_f, t_end as f64, 40_000);

        let mut errs = Vec::new();
        for &n_steps in &[16usize, 32, 64] {
            errs.push(march_error(
                &fx,
                &a,
                DdtScheme::Euler,
                true,
                &psi0,
                &reference,
                t_end,
                n_steps,
                1.0,
            )?);
        }
        for w in errs.windows(2) {
            let order = (w[0] / w[1]).log2();
            assert!(
                order < 1.4,
                "Euler converged at order {order}; if it is second order then \
                 the reference is not measuring the time error"
            );
        }
        Ok(())
    }

    /// `theta = 1` must reproduce Euler to the last bit: a Crank-Nicolson code
    /// path that is asked for Euler and gives something else is exactly the
    /// silent substitution this work is about.
    #[test]
    fn theta_one_is_bitwise_euler() -> Result<()> {
        let Some(fx) = fixture([4, 4, 1]) else { return Ok(()) };
        let (a, dense, psi0) = diffusion_system(&fx)?;
        let reference = vec![0.0f64; dense.n];

        let e_theta = march_error(
            &fx,
            &a,
            DdtScheme::CrankNicolson(1.0),
            false,
            &psi0,
            &reference,
            0.2,
            4,
            1.0,
        )?;
        let e_euler = march_error(
            &fx,
            &a,
            DdtScheme::Euler,
            false,
            &psi0,
            &reference,
            0.2,
            4,
            1.0,
        )?;
        assert_eq!(
            e_theta, e_euler,
            "theta = 1 is not Euler; the theta transform is not a no-op at 1"
        );
        Ok(())
    }

    // ----------------------------------------------------------------------
    //  §13.2 / §22  Local time stepping
    // ----------------------------------------------------------------------

    /// `SPEC-LIT` §22: two `Co_max` values must reach the same steady state.
    ///
    /// The problem is steady convection-diffusion driven by fixed Dirichlet
    /// walls. The local time step is a preconditioner, so the converged answer
    /// may not depend on it - and if the smoothing or the `rDeltaT` formula
    /// leaked into the matrix in a way that does not vanish at convergence,
    /// this is where it shows.
    #[test]
    fn two_courant_numbers_give_the_same_steady_state() -> Result<()> {
        let Some(fx) = fixture([8, 8, 1]) else { return Ok(()) };

        let a = lts_steady_state(&fx, 0.5)?;
        let b = lts_steady_state(&fx, 20.0)?;

        let d = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0 as Scalar, Scalar::max);
        let scale = a.iter().fold(0.0 as Scalar, |m, v| m.max(v.abs()));
        assert!(
            d <= 1e-8 * scale.max(1.0),
            "Co_max 0.5 and 20 disagree by {d} (field scale {scale}); the \
             steady answer depends on the pseudo time step"
        );
        Ok(())
    }

    /// Iterate `div(phi,psi) - laplacian(gamma,psi) = 0` to steady state with
    /// a local Euler ddt at the given Courant number.
    fn lts_steady_state(fx: &Fx, co_max: Scalar) -> Result<Vec<Scalar>> {
        let (gpu, hm, m) = (&fx.gpu, &fx.hm, &fx.m);
        let n = hm.n_cells;

        // A uniform flux in +x, made from the mesh's own area vectors so it is
        // discretely conservative.
        let u = Vec3::new(1.0, 0.35, 0.0);
        let phi_h: Vec<Scalar> = (0..hm.n_internal_faces).map(|f| u.dot(hm.sf[f])).collect();
        let bphi_h: Vec<Scalar> = (0..hm.n_boundary_faces).map(|f| u.dot(hm.b_sf[f])).collect();
        let mut phi = crate::field::GpuSurfaceScalarField::zeros(gpu, m, "phi")?;
        gpu.write(&mut phi.f, &phi_h)?;
        gpu.write(&mut phi.bf, &bphi_h)?;

        // Dirichlet everywhere, with a value that varies over the boundary so
        // the interior solution is not constant.
        let bf: Vec<Scalar> = (0..hm.n_boundary_faces)
            .map(|i| {
                let p = hm.b_cf[i];
                p.x + 2.0 * p.y
            })
            .collect();
        let zeros = vec![0.0 as Scalar; n];
        let psi = dirichlet(gpu, m, hm, &zeros, &bf)?;

        let gamma_h: Vec<Scalar> = hm.mag_sf.iter().map(|s| 0.02 * *s).collect();
        let b_gamma_h: Vec<Scalar> = hm.b_mag_sf.iter().map(|s| 0.02 * *s).collect();
        let gamma = gpu.upload(&gamma_h)?;
        let b_gamma = gpu.upload(&b_gamma_h)?;

        let mut w: DevBuf<Scalar> = gpu.zeros(hm.n_internal_faces)?;
        let mut bw: DevBuf<Scalar> = gpu.zeros(hm.n_boundary_faces)?;
        fv::div_scheme_weights(
            gpu,
            &fx.fvk,
            Some(&mut w),
            Some(&mut bw),
            fv::DivScheme::Upwind,
            &phi,
            &psi,
            None,
            m,
        )?;

        let mut lts = Lts::new(
            gpu,
            m,
            LtsControls {
                co_max,
                dt_max: 1.0,
                ..LtsControls::default()
            },
        )?;
        lts.update(gpu, &fx.tk, &phi, m)?;

        let mut a = GpuLduMatrix::new(gpu, m)?;
        let mut x = gpu.upload(&zeros)?;
        let mut x_old: DevBuf<Scalar> = gpu.zeros(n)?;
        let mut ws = SolverWorkspace::for_mesh(gpu, m)?;
        let ctrl = SolverControls {
            tolerance: 1e-14,
            rel_tol: 0.0,
            max_iter: 2000,
            ..SolverControls::default()
        };

        for _ in 0..400 {
            gpu.stream().memcpy_dtod(&x, &mut x_old)?;

            a.zero(gpu)?;
            fv::fvm_div_gauss(gpu, &fx.fvk, &mut a, m, &phi, &w, &bw, &psi, 1.0)?;
            fv::fvm_laplacian(gpu, &fx.fvk, &mut a, m, &gamma, &b_gamma, &psi, -1.0)?;
            ldu_ops::add_boundary_contributions(gpu, &fx.lduk, &mut a, m)?;
            fvm_ddt_local(gpu, &fx.tk, &mut a, m, &lts.r_delta_t, &x_old, 1.0)?;

            solver::solve_pbicgstab(gpu, &fx.solk, &mut x, &a, m, &mut ws, &ctrl)?;
        }

        gpu.sync()?;
        gpu.download(&x)
    }

    #[test]
    fn the_local_step_honours_its_floor_and_the_courant_number() -> Result<()> {
        let Some(fx) = fixture([4, 4, 1]) else { return Ok(()) };
        let (gpu, hm, m) = (&fx.gpu, &fx.hm, &fx.m);

        // Zero flux: every cell must sit exactly on the 1/dt_max floor.
        let phi = crate::field::GpuSurfaceScalarField::zeros(gpu, m, "phi")?;
        let mut lts = Lts::new(
            gpu,
            m,
            LtsControls {
                co_max: 1.0,
                dt_max: 0.25,
                ..LtsControls::default()
            },
        )?;
        lts.update(gpu, &fx.tk, &phi, m)?;
        gpu.sync()?;
        for v in gpu.download(&lts.r_delta_t)? {
            assert!((v - 4.0).abs() < 1e-13, "zero flux gave rDeltaT = {v}");
        }

        // A uniform flux: halving Co_max must double every local step, because
        // the Courant term is exactly proportional to 1/Co_max above the floor.
        let u = Vec3::new(3.0, 0.0, 0.0);
        let phi_h: Vec<Scalar> = (0..hm.n_internal_faces).map(|f| u.dot(hm.sf[f])).collect();
        let bphi_h: Vec<Scalar> = (0..hm.n_boundary_faces).map(|f| u.dot(hm.b_sf[f])).collect();
        let mut phi = crate::field::GpuSurfaceScalarField::zeros(gpu, m, "phi")?;
        gpu.write(&mut phi.f, &phi_h)?;
        gpu.write(&mut phi.bf, &bphi_h)?;

        let mut one = Lts::new(
            gpu,
            m,
            LtsControls { co_max: 1.0, dt_max: 1e9, ..LtsControls::default() },
        )?;
        one.update(gpu, &fx.tk, &phi, m)?;
        let mut half = Lts::new(
            gpu,
            m,
            LtsControls { co_max: 0.5, dt_max: 1e9, ..LtsControls::default() },
        )?;
        half.update(gpu, &fx.tk, &phi, m)?;
        gpu.sync()?;

        let a = gpu.download(&one.r_delta_t)?;
        let b = gpu.download(&half.r_delta_t)?;
        for c in 0..hm.n_cells {
            assert!(
                (b[c] - 2.0 * a[c]).abs() <= 1e-12 * b[c],
                "cell {c}: Co 0.5 gave {}, Co 1 gave {}",
                b[c],
                a[c]
            );
        }
        Ok(())
    }

    #[test]
    fn smoothing_bounds_the_neighbour_ratio_and_never_lowers_a_cell() -> Result<()> {
        let Some(fx) = fixture([12, 1, 1]) else { return Ok(()) };
        let (gpu, hm, m) = (&fx.gpu, &fx.hm, &fx.m);

        // One cell with a huge flux through it, the rest quiescent: the worst
        // case the smoothing exists for.
        let mut phi_h = vec![0.0 as Scalar; hm.n_internal_faces];
        if !phi_h.is_empty() {
            phi_h[hm.n_internal_faces / 2] = 500.0;
        }
        let mut phi = crate::field::GpuSurfaceScalarField::zeros(gpu, m, "phi")?;
        gpu.write(&mut phi.f, &phi_h)?;

        let ctrl = LtsControls {
            co_max: 1.0,
            dt_max: 1.0,
            smoothing_ratio: 1.1,
            n_sweeps: 40,
            damping: 1.0,
        };
        let mut raw = Lts::new(gpu, m, LtsControls { n_sweeps: 0, ..ctrl })?;
        raw.update(gpu, &fx.tk, &phi, m)?;
        let mut smooth = Lts::new(gpu, m, ctrl)?;
        smooth.update(gpu, &fx.tk, &phi, m)?;
        gpu.sync()?;

        let r = gpu.download(&raw.r_delta_t)?;
        let s = gpu.download(&smooth.r_delta_t)?;

        for c in 0..hm.n_cells {
            assert!(s[c] >= r[c] - 1e-12, "smoothing lowered cell {c}");
        }
        for f in 0..hm.n_internal_faces {
            let (o, nb) = (hm.owner[f] as usize, hm.neighbour[f] as usize);
            let hi = s[o].max(s[nb]);
            let lo = s[o].min(s[nb]);
            assert!(
                hi <= lo * ctrl.smoothing_ratio * (1.0 + 1e-9),
                "face {f}: {hi} vs {lo} exceeds the ratio {}",
                ctrl.smoothing_ratio
            );
        }
        Ok(())
    }
}
