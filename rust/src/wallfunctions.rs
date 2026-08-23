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
//!   ofgpu `SPEC-LIT.md` §6.4. The two items marked *DESIGN* there - the
//!     blending, and the treatment of the wall-adjacent CELL - are ours.
//!   ofgpu `SPEC-LIT.md` §15.2 - `nutLowRe` is `nu_t = 0`, and §15.5 - each
//!     field's OWN patch type decides what happens to it at the wall
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
    /// SPEC-LIT §15.5.
    pub fn build(gpu: &Gpu, m: &HostMesh, faces: &crate::field_setup::WallFaces) -> Result<Self> {
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

        // A zero-length device allocation is an error rather than an empty
        // buffer, so a case with no wall functions still gets one element -
        // which no kernel ever reads, because every launcher returns early on
        // `n_wall_cells == 0`.
        let pad = |v: Vec<Label>| if v.is_empty() { vec![0 as Label] } else { v };

        Ok(Self {
            n_wall_cells: n_cells_w,
            n_wall_faces: n_faces,
            wall_cells: gpu.upload(&pad(wall_cells))?,
            wf_offset: gpu.upload(&wf_offset)?,
            wf_face: gpu.upload(&pad(wf_face))?,

            n_nut_faces: n_nut,
            nut_face: gpu.upload(&pad(nut_faces))?,

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
    pub fn update_nut(
        &self,
        gpu: &Gpu,
        nut_bf: &mut DevBuf<Scalar>,
        k: &DevBuf<Scalar>,
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

        let cmu25 = wc.cmu.powf(0.25);
        let nl = n as Label;
        let f = self.k.nut_wall.clone();

        unsafe {
            gpu.stream()
                .launch_builder(&f)
                .arg(nut_bf)
                .arg(k)
                .arg(&m.b_face_cells)
                .arg(&m.b_y)
                .arg(&self.nut_face)
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

        let wd = WallData::build(&gpu, &m, &same_for_both(&flags))?;
        assert_eq!(wd.n_wall_faces, p.size);
        assert_eq!(wd.n_wall_cells, p.size);

        let nu: Scalar = 1e-5;
        let k_min: Scalar = 1e-15;
        let wc = WallFunctionCoeffs::default();

        // k chosen so y+ straddles y+_lam: y = 0.125, C_mu^0.25 = 0.5477.
        let k_host = vec![2.0e-6 as Scalar; m.n_cells];
        let k_dev = gpu.upload(&k_host)?;
        let mut nut_bf = gpu.zeros::<Scalar>(m.n_boundary_faces)?;

        wd.update_nut(&gpu, &mut nut_bf, &k_dev, &gm, &wc, nu, k_min)?;
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

        let mut wd = WallData::build(&gpu, &m, &same_for_both(&flags))?;
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

        wd.update_nut(&gpu, &mut nut_bf, &k_dev, &gm, &wc, nu, k_min)?;
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

        let mut wd = WallData::build(&gpu, &m, &same_for_both(&flags))?;
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
}
