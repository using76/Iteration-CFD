// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Educational use is free; research,
// publication and commercial use require a licence - simul@msimul.com
// See LICENSE at the repository root.

//! The LES filter width - SPEC-LIT §16 - and the launchers for `cuda/les.cu`.
//!
//! Written from:
//!   Deardorff, *J. Fluid Mech.* 41 (1970) 453-480 - the cube root of the
//!     cell volume
//!   Scotti, Meneveau & Lilly, *Phys. Fluids A* 5 (1993) 2306-2308 - the
//!     anisotropy correction
//!   van Driest, *J. Aeronaut. Sci.* 23 (1956) 1007-1011 - the near-wall
//!     damping
//!   Smagorinsky, *Mon. Weather Rev.* 91 (1963) 99-164, Nicoud & Ducros,
//!     *Flow Turbul. Combust.* 62 (1999) 183-200, and Deardorff,
//!     *Boundary-Layer Meteorol.* 18 (1980) 495-527 - the three models these
//!     widths feed; the models themselves are in `src/models/les.rs`
//!   Tucker, *Applied Mathematical Modelling* 22 (1998) 293-305 - the wall
//!     distance van Driest damping needs
//!   ofgpu `SPEC-LIT.md` §16 and §6.5
//! No GPL-licensed source was consulted.
//!
//! # Why the filter width is its own object
//!
//! Every subgrid model is `nu_t = (C Delta)^a (something from grad U)`, so the
//! width is the one thing all three share and the one thing that is a property
//! of the *mesh* rather than of the model. Splitting it out means the three
//! models in `src/models/les.rs` are three lines of physics each, and it means
//! the width can be tested against SPEC-LIT §16 with no model present at all -
//! which is what `tests` below does.
//!
//! # The four steps, in order
//!
//! ```text
//! base       cubeRootVol | maxDeltaxyz            §16.1, §16.2
//!   * f      Scotti anisotropy correction         §16.3   (optional)
//!   min      van Driest damping                   §16.4   (optional)
//!   smooth   neighbour-ratio sweeps               §16.5   (optional)
//! ```
//!
//! The first two are geometry and are computed once, at construction. The
//! third depends on the flow through `y+` and is rebuilt every time
//! [`LesDelta::update`] is called. The fourth is applied last, so that what a
//! neighbour sees is the width that will actually be used, and it therefore
//! has to be redone whenever the third is - which is why [`LesDelta::update`]
//! always restarts from the stored geometric base rather than smoothing an
//! already-smoothed field, an operation that is not idempotent and would creep
//! upward one sweep at a time.

use cudarc::driver::{CudaFunction, PushKernelArg};

use crate::device::{cfg_for, DevBuf, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field_ops::{copy_field, FieldKernels};
use crate::mesh::GpuMesh;
use crate::{Label, Scalar, Tensor, Vec3};

// ==========================================================================
//  What a case asked for
// ==========================================================================

/// The geometric width before any correction - SPEC-LIT §16.1 and §16.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseDelta {
    /// `Delta = V^(1/3)`. The default, and correct for an isotropic cell.
    CubeRootVol,
    /// `Delta = max(dx1, dx2, dx3)`. Safer on a highly anisotropic cell, where
    /// the cube root underestimates the largest unresolved scale.
    MaxEdge,
}

impl BaseDelta {
    /// The name a case writes in `LES { delta ...; }`.
    pub fn name(self) -> &'static str {
        match self {
            Self::CubeRootVol => "cubeRootVol",
            Self::MaxEdge => "maxDeltaxyz",
        }
    }
}

/// Smoothing - SPEC-LIT §16.5, which marks the whole thing *DESIGN*.
///
/// # The numbers, and why they are these numbers
///
/// * `max_ratio = 1.15`. One sweep may raise a cell to within 15 % of its
///   largest neighbour. The corresponding local-time-step sweep (§13.2) uses
///   1.1; the filter width is allowed a little more slack because it enters
///   `nu_t` quadratically in two of the three models, so over-smoothing it
///   costs resolved-scale energy directly.
/// * `sweeps = 2`, which propagates a width outward over a two-cell halo.
///   More sweeps cost one kernel each and change less every time, because the
///   ratio compounds: after `n` sweeps a cell can have been raised by at most
///   `max_ratio^n` relative to a cell `n` away.
///
/// Both are ours and neither is in the literature. The smoothing is off by
/// default, because a uniform mesh does not need it and a case that has a
/// jump in cell size should be told to smooth rather than smoothed silently.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmoothSpec {
    pub max_ratio: Scalar,
    pub sweeps: usize,
}

impl Default for SmoothSpec {
    fn default() -> Self {
        Self {
            max_ratio: 1.15,
            sweeps: 2,
        }
    }
}

/// Everything §16 offers, as one description a case can name.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeltaSpec {
    pub base: BaseDelta,
    /// A plain multiplier on the geometric width. OpenFOAM spells it
    /// `deltaCoeff` and it is 1 unless a case says otherwise.
    pub delta_coeff: Scalar,
    /// Apply the Scotti anisotropy correction - SPEC-LIT §16.3.
    pub anisotropy: bool,
    /// Apply van Driest damping - SPEC-LIT §16.4. Needs the wall distance and
    /// therefore a mesh with a wall in it.
    pub van_driest: bool,
    /// Smooth the result - SPEC-LIT §16.5.
    pub smooth: Option<SmoothSpec>,

    /// van Driest's constants: `kappa = 0.41`, `A+ = 26`, `C_delta = 0.158`,
    /// exactly as SPEC-LIT §16.4 tabulates them. Carried here rather than
    /// hard-coded so that a case which moves `kappa` for the wall functions
    /// moves it here too - SPEC-LIT §15.6's rule, applied to the one other
    /// place `kappa` appears.
    pub kappa: Scalar,
    pub a_plus: Scalar,
    pub c_delta: Scalar,
}

impl Default for DeltaSpec {
    fn default() -> Self {
        Self {
            base: BaseDelta::CubeRootVol,
            delta_coeff: 1.0,
            anisotropy: false,
            van_driest: false,
            smooth: None,
            kappa: 0.41,
            a_plus: 26.0,
            c_delta: 0.158,
        }
    }
}

impl DeltaSpec {
    fn check(&self) -> Result<()> {
        if !(self.delta_coeff > 0.0) {
            return Err(Error::Config(format!(
                "LES delta: deltaCoeff = {}; a filter width must be positive",
                self.delta_coeff
            )));
        }
        if let Some(s) = self.smooth {
            if !(s.max_ratio > 1.0) {
                return Err(Error::Config(format!(
                    "LES delta: the smoothing ratio is {}; it must exceed 1, \
                     or the sweep raises every cell to its largest neighbour \
                     and never terminates",
                    s.max_ratio
                )));
            }
        }
        if self.van_driest && !(self.a_plus > 0.0 && self.c_delta > 0.0) {
            return Err(Error::Config(format!(
                "LES delta: van Driest damping needs A+ > 0 and C_delta > 0, \
                 got A+ = {}, C_delta = {}",
                self.a_plus, self.c_delta
            )));
        }
        Ok(())
    }

    /// How a run should print what it is doing.
    pub fn describe(&self) -> String {
        let mut s = self.base.name().to_string();
        if self.delta_coeff != 1.0 {
            s.push_str(&format!(" * {}", self.delta_coeff));
        }
        if self.anisotropy {
            s.push_str(" + Scotti");
        }
        if self.van_driest {
            s.push_str(&format!(
                " + vanDriest(kappa {}, A+ {}, Cdelta {})",
                self.kappa, self.a_plus, self.c_delta
            ));
        }
        if let Some(sm) = self.smooth {
            s.push_str(&format!(
                " + smooth(ratio {}, {} sweeps)",
                sm.max_ratio, sm.sweeps
            ));
        }
        s
    }
}

// ==========================================================================
//  Kernels
// ==========================================================================

/// Every entry point in `cuda/les.cu`, resolved once.
pub struct LesKernels {
    extents: CudaFunction,
    cube_root: CudaFunction,
    max_edge: CudaFunction,
    scotti: CudaFunction,
    local_y_plus: CudaFunction,
    van_driest: CudaFunction,
    smooth: CudaFunction,

    nut_smagorinsky: CudaFunction,
    nut_wale: CudaFunction,
    nut_deardorff: CudaFunction,
    test_filter: CudaFunction,
}

impl LesKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let k = KernelSet::new(gpu, crate::kernels::LES)?;
        Ok(Self {
            extents: k.func("lesCellExtents")?,
            cube_root: k.func("lesDeltaCubeRootVol")?,
            max_edge: k.func("lesDeltaMaxEdge")?,
            scotti: k.func("lesScottiFactor")?,
            local_y_plus: k.func("lesLocalYPlus")?,
            van_driest: k.func("lesVanDriest")?,
            smooth: k.func("lesSmoothDelta")?,

            nut_smagorinsky: k.func("lesNutSmagorinsky")?,
            nut_wale: k.func("lesNutWale")?,
            nut_deardorff: k.func("lesNutDeardorff")?,
            test_filter: k.func("lesTestFilterVector")?,
        })
    }
}

fn expect_len<T>(buf: &DevBuf<T>, want: usize, what: &str) -> Result<()> {
    if buf.len() == want {
        Ok(())
    } else {
        Err(Error::Config(format!(
            "LES: `{what}` has {} elements, expected {want}",
            buf.len()
        )))
    }
}

/// `dx_i = 2 max_f |Cf_i - C_i|` per cell - the input to §16.2 and §16.3.
///
/// See `cuda/les.cu` for why this measure and not the point bounding box.
pub fn cell_extents(
    gpu: &Gpu,
    k: &LesKernels,
    dx: &mut DevBuf<Vec3>,
    m: &GpuMesh,
) -> Result<()> {
    let n = m.n_cells;
    if n == 0 {
        return Ok(());
    }
    expect_len(dx, n, "dx")?;

    let nl = n as Label;
    let f = k.extents.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(dx)
            .arg(&m.c)
            .arg(&m.cf)
            .arg(&m.b_cf)
            .arg(&m.cf_offset)
            .arg(&m.cf_face)
            .arg(&m.cf_own)
            .arg(&m.bcf_offset)
            .arg(&m.bcf_face)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `y+` and `u_tau` from the local wall-normal shear - SPEC-LIT §16.4's input.
///
/// *DESIGN*; the derivation and its justification are at the kernel in
/// `cuda/les.cu`. A caller with a better `u_tau` may fill `y_plus` itself and
/// never call this.
#[allow(clippy::too_many_arguments)]
pub fn local_y_plus(
    gpu: &Gpu,
    k: &LesKernels,
    y_plus: &mut DevBuf<Scalar>,
    u_tau: &mut DevBuf<Scalar>,
    grad_u: &DevBuf<Tensor>,
    grad_y: &DevBuf<Vec3>,
    nut: &DevBuf<Scalar>,
    y: &DevBuf<Scalar>,
    nu: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(y_plus, n, "yPlus")?;
    expect_len(u_tau, n, "uTau")?;
    expect_len(grad_u, n, "grad U")?;
    expect_len(grad_y, n, "grad y")?;
    expect_len(nut, n, "nut")?;
    expect_len(y, n, "y")?;

    let nl = n as Label;
    let f = k.local_y_plus.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(y_plus)
            .arg(u_tau)
            .arg(grad_u)
            .arg(grad_y)
            .arg(nut)
            .arg(y)
            .arg(&nu)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `nu_t = (C_s Delta)² sqrt(2 S:S)` - Smagorinsky (1963), SPEC-LIT §6.5.
#[allow(clippy::too_many_arguments)]
pub fn nut_smagorinsky(
    gpu: &Gpu,
    k: &LesKernels,
    nut: &mut DevBuf<Scalar>,
    s: &DevBuf<Scalar>,
    delta: &DevBuf<Scalar>,
    cs: Scalar,
    nut_max: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(nut, n, "nut")?;
    expect_len(s, n, "S")?;
    expect_len(delta, n, "delta")?;

    let nl = n as Label;
    let f = k.nut_smagorinsky.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(nut)
            .arg(s)
            .arg(delta)
            .arg(&cs)
            .arg(&nut_max)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// WALE - Nicoud & Ducros (1999), SPEC-LIT §6.5.
#[allow(clippy::too_many_arguments)]
pub fn nut_wale(
    gpu: &Gpu,
    k: &LesKernels,
    nut: &mut DevBuf<Scalar>,
    grad_u: &DevBuf<Tensor>,
    delta: &DevBuf<Scalar>,
    cw: Scalar,
    nut_max: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(nut, n, "nut")?;
    expect_len(grad_u, n, "grad U")?;
    expect_len(delta, n, "delta")?;

    let nl = n as Label;
    let f = k.nut_wale.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(nut)
            .arg(grad_u)
            .arg(delta)
            .arg(&cw)
            .arg(&nut_max)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// The test filter Deardorff's subgrid energy estimate needs.
///
/// Adapted from FDS's `TEST_FILTER` (NIST, public domain); see `cuda/les.cu`
/// for the acknowledgement and for what is ours in it.
#[allow(clippy::too_many_arguments)]
pub fn test_filter(
    gpu: &Gpu,
    k: &LesKernels,
    u_hat: &mut DevBuf<Vec3>,
    u: &DevBuf<Vec3>,
    u_b: &DevBuf<Vec3>,
    m: &GpuMesh,
) -> Result<()> {
    let n = m.n_cells;
    if n == 0 {
        return Ok(());
    }
    expect_len(u_hat, n, "uHat")?;
    expect_len(u, n, "U")?;
    expect_len(u_b, m.n_boundary_faces, "U boundary")?;

    let nl = n as Label;
    let f = k.test_filter.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(u_hat)
            .arg(u)
            .arg(u_b)
            .arg(&m.owner)
            .arg(&m.neighbour)
            .arg(&m.cf_offset)
            .arg(&m.cf_face)
            .arg(&m.cf_own)
            .arg(&m.bcf_offset)
            .arg(&m.bcf_face)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

/// `k_sgs = |u - u_hat|²/2` and `nu_t = C_D Delta sqrt(k_sgs)` - Deardorff
/// (1980) in the algebraic form FDS uses; SPEC-LIT §6.5.
#[allow(clippy::too_many_arguments)]
pub fn nut_deardorff(
    gpu: &Gpu,
    k: &LesKernels,
    nut: &mut DevBuf<Scalar>,
    k_sgs: &mut DevBuf<Scalar>,
    u: &DevBuf<Vec3>,
    u_hat: &DevBuf<Vec3>,
    delta: &DevBuf<Scalar>,
    cd: Scalar,
    nut_max: Scalar,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    expect_len(nut, n, "nut")?;
    expect_len(k_sgs, n, "kSgs")?;
    expect_len(u, n, "U")?;
    expect_len(u_hat, n, "uHat")?;
    expect_len(delta, n, "delta")?;

    let nl = n as Label;
    let f = k.nut_deardorff.clone();
    unsafe {
        gpu.stream()
            .launch_builder(&f)
            .arg(nut)
            .arg(k_sgs)
            .arg(u)
            .arg(u_hat)
            .arg(delta)
            .arg(&cd)
            .arg(&nut_max)
            .arg(&nl)
            .launch(cfg_for(n))?;
    }
    Ok(())
}

// ==========================================================================
//  The filter width itself
// ==========================================================================

/// The filter width of SPEC-LIT §16, and the scratch its two flow-dependent
/// stages need.
///
/// Allocated once. [`Self::update`] launches kernels and nothing else, so an
/// LES time step stays capturable.
pub struct LesDelta {
    spec: DeltaSpec,
    n_cells: usize,

    k: LesKernels,
    fld: FieldKernels,

    /// `[n_cells]` the cell extents §16.2 and §16.3 read.
    dx: DevBuf<Vec3>,
    /// `[n_cells]` the geometric width - base times the Scotti factor, and
    /// nothing else. Computed once; never overwritten.
    base: DevBuf<Scalar>,
    /// `[n_cells]` the width in force, after damping and smoothing.
    delta: DevBuf<Scalar>,
    /// `[n_cells]` ping-pong buffer for the smoothing sweeps.
    scratch: DevBuf<Scalar>,

    /// `[n_cells]` the local `y+` and `u_tau` §16.4 is driven by.
    y_plus: DevBuf<Scalar>,
    u_tau: DevBuf<Scalar>,
}

impl LesDelta {
    /// Build the geometric width. The flow-dependent stages are
    /// [`Self::update`]'s job and have not run yet, so a caller that asks for
    /// van Driest damping and never calls `update` gets the undamped width -
    /// which is the conservative direction and is what a zero-velocity initial
    /// field would have produced anyway.
    pub fn new(gpu: &Gpu, m: &GpuMesh, spec: DeltaSpec) -> Result<Self> {
        spec.check()?;

        let nc = m.n_cells.max(1);
        let k = LesKernels::new(gpu)?;
        let fld = FieldKernels::new(gpu)?;

        let mut d = Self {
            spec,
            n_cells: m.n_cells,
            k,
            fld,
            dx: gpu.zeros(nc)?,
            base: gpu.zeros(nc)?,
            delta: gpu.zeros(nc)?,
            scratch: gpu.zeros(nc)?,
            y_plus: gpu.zeros(nc)?,
            u_tau: gpu.zeros(nc)?,
        };

        d.build_base(gpu, m)?;
        d.finish(gpu, m)?;

        Ok(d)
    }

    pub fn spec(&self) -> &DeltaSpec {
        &self.spec
    }
    /// The width in force - what every subgrid model multiplies.
    pub fn delta(&self) -> &DevBuf<Scalar> {
        &self.delta
    }
    /// The geometric width, before damping and smoothing. Kept public because
    /// it is what a van Driest test compares against.
    pub fn geometric(&self) -> &DevBuf<Scalar> {
        &self.base
    }
    pub fn cell_extents(&self) -> &DevBuf<Vec3> {
        &self.dx
    }
    pub fn y_plus(&self) -> &DevBuf<Scalar> {
        &self.y_plus
    }
    pub fn u_tau(&self) -> &DevBuf<Scalar> {
        &self.u_tau
    }

    /// §16.1/§16.2 and then §16.3 - pure geometry, computed once.
    fn build_base(&mut self, gpu: &Gpu, m: &GpuMesh) -> Result<()> {
        let n = self.n_cells;
        if n == 0 {
            return Ok(());
        }

        cell_extents(gpu, &self.k, &mut self.dx, m)?;

        let nl = n as Label;
        let coeff = self.spec.delta_coeff;

        match self.spec.base {
            BaseDelta::CubeRootVol => {
                let f = self.k.cube_root.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&mut self.base)
                        .arg(&m.v)
                        .arg(&coeff)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }
            BaseDelta::MaxEdge => {
                let f = self.k.max_edge.clone();
                unsafe {
                    gpu.stream()
                        .launch_builder(&f)
                        .arg(&mut self.base)
                        .arg(&self.dx)
                        .arg(&coeff)
                        .arg(&nl)
                        .launch(cfg_for(n))?;
                }
            }
        }

        if self.spec.anisotropy {
            let f = self.k.scotti.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.base)
                    .arg(&self.dx)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
        }

        Ok(())
    }

    /// Rebuild the width in force: the geometric base, then §16.4's damping if
    /// the flow supports it, then §16.5's smoothing.
    ///
    /// `grad_y` is the gradient of the wall distance, which near a wall is the
    /// unit wall normal - see [`crate::walldistance::WallDistance::grad_y`].
    /// `nut` is the PREVIOUS step's eddy viscosity, which is what makes
    /// `u_tau` the total shear velocity rather than the molecular one; the lag
    /// is the ordinary segregated one.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        gpu: &Gpu,
        m: &GpuMesh,
        grad_u: &DevBuf<Tensor>,
        grad_y: &DevBuf<Vec3>,
        nut: &DevBuf<Scalar>,
        y: &DevBuf<Scalar>,
        nu: Scalar,
    ) -> Result<()> {
        let n = self.n_cells;
        if n == 0 {
            return Ok(());
        }

        // Nothing flow-dependent to do, and `new` already left `delta` right.
        if !self.spec.van_driest {
            return Ok(());
        }

        copy_field(gpu, &self.fld, &mut self.delta, &self.base, n)?;

        local_y_plus(
            gpu,
            &self.k,
            &mut self.y_plus,
            &mut self.u_tau,
            grad_u,
            grad_y,
            nut,
            y,
            nu,
            n,
        )?;

        let nl = n as Label;
        let (kappa, a_plus, c_delta) =
            (self.spec.kappa, self.spec.a_plus, self.spec.c_delta);

        let f = self.k.van_driest.clone();
        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(&mut self.delta)
                .arg(y)
                .arg(&self.y_plus)
                .arg(&kappa)
                .arg(&a_plus)
                .arg(&c_delta)
                .arg(&nl)
                .launch(cfg_for(n))?;
        }

        self.smooth(gpu, m)
    }

    /// `delta <- base`, then §16.5. What `new` leaves behind, and what a spec
    /// with no van Driest term never has to redo.
    fn finish(&mut self, gpu: &Gpu, m: &GpuMesh) -> Result<()> {
        if self.n_cells == 0 {
            return Ok(());
        }
        copy_field(gpu, &self.fld, &mut self.delta, &self.base, self.n_cells)?;
        self.smooth(gpu, m)
    }

    /// §16.5's sweeps, applied to [`Self::delta`] in place through the
    /// ping-pong buffer.
    fn smooth(&mut self, gpu: &Gpu, m: &GpuMesh) -> Result<()> {
        let Some(sm) = self.spec.smooth else {
            return Ok(());
        };
        let n = self.n_cells;
        if n == 0 || sm.sweeps == 0 {
            return Ok(());
        }

        let nl = n as Label;
        let ratio = sm.max_ratio;

        for _ in 0..sm.sweeps {
            let f = self.k.smooth.clone();
            unsafe {
                gpu.stream()
                    .launch_builder(&f)
                    .arg(&mut self.scratch)
                    .arg(&self.delta)
                    .arg(&m.owner)
                    .arg(&m.neighbour)
                    .arg(&m.cf_offset)
                    .arg(&m.cf_face)
                    .arg(&m.cf_own)
                    .arg(&ratio)
                    .arg(&nl)
                    .launch(cfg_for(n))?;
            }
            copy_field(gpu, &self.fld, &mut self.delta, &self.scratch, n)?;
        }

        Ok(())
    }
}

impl LesDelta {
    /// The kernels this object resolved. Handed out so a model does not have
    /// to resolve a second copy of the ones it needs.
    pub fn kernels(&self) -> &LesKernels {
        &self.k
    }
}

// ==========================================================================
//  Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::HostMesh;

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// A box of `d`-sized hexahedra. `d` need not be isotropic, which is the
    /// point of half the tests here.
    fn hex_box(n: [usize; 3], d: Vec3) -> HostMesh {
        let (mut m, points, faces) = crate::mesh::topology::tests::box_mesh(n, d);
        m.compute_geometry(&points, &faces).expect("geometry");
        m.build_cell_face_maps();
        m
    }

    // ----------------------------------------------------------------------
    //  §16.1, §16.2 - the geometric widths
    // ----------------------------------------------------------------------

    /// On a cube the two bases are the same number, and both are the edge.
    /// They can only disagree once the cell is stretched, which is what
    /// SPEC-LIT §16.2 says the maximum-edge base exists for.
    #[test]
    fn on_a_cube_both_bases_are_the_edge_length() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let h: Scalar = 0.25;
        let hm = hex_box([4, 4, 4], Vec3::new(h, h, h));
        let m = GpuMesh::upload(&gpu, &hm)?;

        for base in [BaseDelta::CubeRootVol, BaseDelta::MaxEdge] {
            let d = LesDelta::new(
                &gpu,
                &m,
                DeltaSpec {
                    base,
                    ..Default::default()
                },
            )?;
            gpu.sync()?;
            for (c, &v) in gpu.download(d.delta())?.iter().enumerate() {
                assert!(
                    (v - h).abs() < 1e-13 * h,
                    "{}: cell {c} has Delta = {v}, edge {h}",
                    base.name()
                );
            }
        }

        Ok(())
    }

    /// The cell extents §16.2 and §16.3 read, on a cell whose three edges are
    /// all different.
    #[test]
    fn the_cell_extents_are_the_edges() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let d = Vec3::new(0.4, 0.1, 0.05);
        let hm = hex_box([3, 3, 3], d);
        let m = GpuMesh::upload(&gpu, &hm)?;

        let delta = LesDelta::new(&gpu, &m, DeltaSpec::default())?;
        gpu.sync()?;

        for (c, e) in gpu.download(delta.cell_extents())?.iter().enumerate() {
            assert!((e.x - d.x).abs() < 1e-13, "cell {c}: dx = {}", e.x);
            assert!((e.y - d.y).abs() < 1e-13, "cell {c}: dy = {}", e.y);
            assert!((e.z - d.z).abs() < 1e-13, "cell {c}: dz = {}", e.z);
        }

        // And the maximum-edge base is then the largest of the three, while
        // the cube root is the geometric mean - a factor of two apart here.
        let max_edge = LesDelta::new(
            &gpu,
            &m,
            DeltaSpec {
                base: BaseDelta::MaxEdge,
                ..Default::default()
            },
        )?;
        gpu.sync()?;
        let cube = gpu.download(delta.delta())?[0];
        let edge = gpu.download(max_edge.delta())?[0];

        let mean = (d.x * d.y * d.z).cbrt();
        assert!((cube - mean).abs() < 1e-13, "{cube} vs {mean}");
        assert!((edge - d.x).abs() < 1e-13, "{edge} vs {}", d.x);
        assert!(edge > cube, "the maximum edge must exceed the geometric mean");

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  §16.3 - Scotti
    // ----------------------------------------------------------------------

    /// SPEC-LIT §22: "LES delta, isotropic cell, Scotti `f = 1`".
    ///
    /// Exactly one, not approximately: every logarithm in the correction is
    /// `ln 1 = 0` and `cosh 0 = 1`, so a cube must come back bit for bit as
    /// the uncorrected width.
    #[test]
    fn the_scotti_factor_is_one_on_an_isotropic_cell() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let h: Scalar = 0.2;
        let hm = hex_box([3, 3, 3], Vec3::new(h, h, h));
        let m = GpuMesh::upload(&gpu, &hm)?;

        let plain = LesDelta::new(&gpu, &m, DeltaSpec::default())?;
        let scotti = LesDelta::new(
            &gpu,
            &m,
            DeltaSpec {
                anisotropy: true,
                ..Default::default()
            },
        )?;
        gpu.sync()?;

        let a = gpu.download(plain.delta())?;
        let b = gpu.download(scotti.delta())?;

        for (c, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(
                x, y,
                "cell {c}: the Scotti correction changed an isotropic cell, \
                 {x} -> {y}"
            );
        }

        Ok(())
    }

    /// And on a stretched cell it must grow, and grow by the amount Scotti,
    /// Meneveau & Lilly's expression says - which is computed here on the host
    /// from the aspect ratios, independently of the kernel.
    #[test]
    fn the_scotti_factor_grows_with_the_aspect_ratio() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let d = Vec3::new(0.8, 0.2, 0.05);
        let hm = hex_box([3, 3, 3], d);
        let m = GpuMesh::upload(&gpu, &hm)?;

        let plain = LesDelta::new(&gpu, &m, DeltaSpec::default())?;
        let scotti = LesDelta::new(
            &gpu,
            &m,
            DeltaSpec {
                anisotropy: true,
                ..Default::default()
            },
        )?;
        gpu.sync()?;

        let base = gpu.download(plain.delta())?[0];
        let corrected = gpu.download(scotti.delta())?[0];

        let dmax = d.x.max(d.y).max(d.z);
        let a1 = (d.y / dmax).ln();
        let a2 = (d.z / dmax).ln();
        let want = ((4.0 / 27.0) * (a1 * a1 - a1 * a2 + a2 * a2)).sqrt().cosh();

        assert!(want > 1.0, "the test cell is not anisotropic enough: f = {want}");
        assert!(
            (corrected / base - want).abs() < 1e-12,
            "f = {}, Scotti's expression gives {want}",
            corrected / base
        );

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  §16.4 - van Driest
    // ----------------------------------------------------------------------

    /// SPEC-LIT §22: "van Driest, `y -> infinity`, reduces to the geometric
    /// delta".
    ///
    /// Driven with a real shear so that `y+` is genuinely positive - a test
    /// that passed because `y+` came out zero would be measuring the guard
    /// rather than the limit. The damped length is `2.59 y [1 - exp(-y+/26)]`,
    /// which at `y = 1e4 m` is four orders of magnitude above any filter width
    /// on this mesh, so the `min` must pick the geometric one exactly.
    #[test]
    fn van_driest_reduces_to_the_geometric_delta_far_from_the_wall() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let h: Scalar = 0.1;
        let hm = hex_box([4, 4, 4], Vec3::new(h, h, h));
        let m = GpuMesh::upload(&gpu, &hm)?;
        let n = hm.n_cells;

        let mut delta = LesDelta::new(
            &gpu,
            &m,
            DeltaSpec {
                van_driest: true,
                ..Default::default()
            },
        )?;

        // A pure shear dU_x/dy = 10, a wall normal in +y, and the wall four
        // kilometres away.
        let mut g = crate::Tensor::default();
        g.yx = 10.0;
        let grad_u = gpu.upload(&vec![g; n])?;
        let grad_y = gpu.upload(&vec![Vec3::new(0.0, 1.0, 0.0); n])?;
        let nut = gpu.upload(&vec![0.0 as Scalar; n])?;
        let y = gpu.upload(&vec![1.0e4 as Scalar; n])?;

        delta.update(&gpu, &m, &grad_u, &grad_y, &nut, &y, 1e-5)?;
        gpu.sync()?;

        let yp = gpu.download(delta.y_plus())?;
        assert!(
            yp[0] > 1e3,
            "y+ came out {}, so the damping was never exercised",
            yp[0]
        );

        let base = gpu.download(delta.geometric())?;
        let damped = gpu.download(delta.delta())?;
        for (c, (a, b)) in base.iter().zip(&damped).enumerate() {
            assert_eq!(a, b, "cell {c}: the damping moved a far-field Delta");
        }

        Ok(())
    }

    /// And close to the wall it must bite, and land on van Driest's own
    /// expression - recomputed here on the host from the `y+` the kernel
    /// reported, so that the two halves of §16.4 are checked separately.
    #[test]
    fn van_driest_damps_the_delta_close_to_a_wall() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let h: Scalar = 0.1;
        let hm = hex_box([4, 4, 4], Vec3::new(h, h, h));
        let m = GpuMesh::upload(&gpu, &hm)?;
        let n = hm.n_cells;

        let spec = DeltaSpec {
            van_driest: true,
            ..Default::default()
        };
        let mut delta = LesDelta::new(&gpu, &m, spec)?;

        let mut g = crate::Tensor::default();
        g.yx = 10.0;
        let grad_u = gpu.upload(&vec![g; n])?;
        let grad_y = gpu.upload(&vec![Vec3::new(0.0, 1.0, 0.0); n])?;
        let nut = gpu.upload(&vec![0.0 as Scalar; n])?;

        let y_wall: Scalar = 1e-4;
        let y = gpu.upload(&vec![y_wall; n])?;
        let nu: Scalar = 1e-5;

        delta.update(&gpu, &m, &grad_u, &grad_y, &nut, &y, nu)?;
        gpu.sync()?;

        let u_tau = gpu.download(delta.u_tau())?[0];
        let yp = gpu.download(delta.y_plus())?[0];
        let got = gpu.download(delta.delta())?[0];
        let base = gpu.download(delta.geometric())?[0];

        // u_tau = sqrt(nu_eff |dU/dn|) with the wall-parallel shear 10 s^-1.
        assert!(
            (u_tau - (nu * 10.0 as Scalar).sqrt()).abs() < 1e-14,
            "u_tau = {u_tau}"
        );
        assert!((yp - y_wall * u_tau / nu).abs() < 1e-12 * yp, "y+ = {yp}");

        let want = (spec.kappa / spec.c_delta) * y_wall * (1.0 - (-yp / spec.a_plus).exp());
        assert!(want < base, "the test point is not close enough to the wall");
        assert!(
            (got - want).abs() < 1e-13 * want,
            "Delta = {got}, van Driest gives {want}"
        );

        Ok(())
    }

    /// A domain with no wall reports `y+ = 0`, and the damping must then leave
    /// the filter width alone rather than annihilate it - see the *DESIGN*
    /// note at `lesVanDriest` in `cuda/les.cu`.
    #[test]
    fn no_wall_means_no_damping() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let h: Scalar = 0.1;
        let hm = hex_box([3, 3, 3], Vec3::new(h, h, h));
        let m = GpuMesh::upload(&gpu, &hm)?;
        let n = hm.n_cells;

        let mut delta = LesDelta::new(
            &gpu,
            &m,
            DeltaSpec {
                van_driest: true,
                ..Default::default()
            },
        )?;

        // What `wall_distance` leaves behind when there is no wall: a huge
        // distance and a zero gradient, so there is no wall-normal direction
        // and no y+.
        let grad_u = gpu.upload(&vec![crate::Tensor::default(); n])?;
        let grad_y = gpu.upload(&vec![Vec3::ZERO; n])?;
        let nut = gpu.upload(&vec![0.0 as Scalar; n])?;
        let y = gpu.upload(&vec![crate::walldistance::NO_WALL; n])?;

        delta.update(&gpu, &m, &grad_u, &grad_y, &nut, &y, 1e-5)?;
        gpu.sync()?;

        for (c, &v) in gpu.download(delta.delta())?.iter().enumerate() {
            assert!(
                (v - h).abs() < 1e-13 * h,
                "cell {c}: Delta = {v} in a wall-free box, edge {h}"
            );
        }

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  §16.5 - smoothing
    // ----------------------------------------------------------------------

    /// On a uniform mesh the sweep must change nothing at all: every
    /// neighbour's width divided by a ratio above one is below the cell's own.
    #[test]
    fn smoothing_a_uniform_mesh_changes_nothing() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let h: Scalar = 0.1;
        let hm = hex_box([4, 4, 4], Vec3::new(h, h, h));
        let m = GpuMesh::upload(&gpu, &hm)?;

        let d = LesDelta::new(
            &gpu,
            &m,
            DeltaSpec {
                smooth: Some(SmoothSpec::default()),
                ..Default::default()
            },
        )?;
        gpu.sync()?;

        for (c, &v) in gpu.download(d.delta())?.iter().enumerate() {
            assert!((v - h).abs() < 1e-13 * h, "cell {c}: Delta = {v}");
        }

        Ok(())
    }

    /// A graded mesh, and the property the sweep is for: once it has settled,
    /// no cell's width exceeds a neighbour's by more than the ratio.
    ///
    /// The mesh grades as `x ~ i²`, which puts a factor of fifteen between the
    /// first cell and the last - far more than any ratio would allow, so the
    /// unsmoothed field fails the same check and the test has something to
    /// measure.
    #[test]
    fn smoothing_limits_the_ratio_between_neighbours() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };

        let nx = 8usize;
        let (mut hm, mut points, faces) =
            crate::mesh::topology::tests::box_mesh([nx, 2, 1], Vec3::new(1.0, 0.25, 0.25));

        // x <- nx (x/nx)^4, which grades the cells hard and leaves y and z
        // alone. The fourth power rather than the second because the width is
        // the CUBE ROOT of the volume: a factor of three between two adjacent
        // cell sizes is only 1.44 in Delta, which is not enough above the
        // ratio for the check below to be measuring anything.
        for p in points.iter_mut() {
            let t = p.x / nx as Scalar;
            p.x = nx as Scalar * t * t * t * t;
        }
        hm.compute_geometry(&points, &faces).expect("geometry");
        hm.build_cell_face_maps();

        let m = GpuMesh::upload(&gpu, &hm)?;
        let ratio: Scalar = 1.15;

        let rough = LesDelta::new(&gpu, &m, DeltaSpec::default())?;
        let smooth = LesDelta::new(
            &gpu,
            &m,
            DeltaSpec {
                smooth: Some(SmoothSpec {
                    max_ratio: ratio,
                    // Enough sweeps to reach the fixed point: the widths span a
                    // factor of fifteen and each sweep can close a factor of
                    // 1.15, so ln 15 / ln 1.15 = 20 would do and 32 is
                    // comfortable.
                    sweeps: 32,
                }),
                ..Default::default()
            },
        )?;
        gpu.sync()?;

        let a = gpu.download(rough.delta())?;
        let b = gpu.download(smooth.delta())?;

        let worst = |d: &[Scalar]| {
            let mut w: Scalar = 1.0;
            for f in 0..hm.n_internal_faces {
                let (p, q) = (hm.owner[f] as usize, hm.neighbour[f] as usize);
                w = w.max(d[p] / d[q]).max(d[q] / d[p]);
            }
            w
        };

        assert!(
            worst(&a) > ratio * 1.5,
            "the unsmoothed mesh is already smooth ({}), so this measures nothing",
            worst(&a)
        );
        assert!(
            worst(&b) <= ratio * (1.0 + 1e-12),
            "after smoothing the worst neighbour ratio is {}, limit {ratio}",
            worst(&b)
        );

        // Raising, never lowering: the smoothed width must remain an upper
        // bound on the geometric one, or the filter would claim to resolve
        // scales it does not.
        for (c, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(y >= x, "cell {c}: smoothing LOWERED Delta, {x} -> {y}");
        }

        Ok(())
    }

    // ----------------------------------------------------------------------
    //  The contract
    // ----------------------------------------------------------------------

    #[test]
    fn a_smoothing_ratio_of_one_is_refused() -> Result<()> {
        let Some(gpu) = gpu() else {
            return Ok(());
        };
        let hm = hex_box([2, 2, 2], Vec3::new(0.5, 0.5, 0.5));
        let m = GpuMesh::upload(&gpu, &hm)?;

        assert!(LesDelta::new(
            &gpu,
            &m,
            DeltaSpec {
                smooth: Some(SmoothSpec {
                    max_ratio: 1.0,
                    sweeps: 2
                }),
                ..Default::default()
            }
        )
        .is_err());

        assert!(LesDelta::new(
            &gpu,
            &m,
            DeltaSpec {
                delta_coeff: 0.0,
                ..Default::default()
            }
        )
        .is_err());

        Ok(())
    }

    #[test]
    fn the_description_names_every_stage() {
        let s = DeltaSpec {
            base: BaseDelta::MaxEdge,
            anisotropy: true,
            van_driest: true,
            smooth: Some(SmoothSpec::default()),
            ..Default::default()
        }
        .describe();

        for want in ["maxDeltaxyz", "Scotti", "vanDriest", "smooth"] {
            assert!(s.contains(want), "{s} does not mention {want}");
        }
    }
}
