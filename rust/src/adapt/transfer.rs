// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Moving the fields across an adapt, conservatively.
//!
//! Written from:
//!   ofgpu SPEC-LIT.md sections 1, 25, 26 and 75.6
//!   T. J. Barth and D. C. Jespersen, "The design and application of upwind
//!     schemes on unstructured meshes", AIAA paper 89-0366 (1989) - the
//!     monotone reconstruction limiter
//!   M. J. Berger and P. Colella, J. Comput. Phys. 82 (1989) 64-84 - the
//!     restriction/prolongation pair, and the flux-register apparatus
//!     SPEC-LIT section 74.1 argues a face-based code does not need
//!   D. S. Balsara, J. Comput. Phys. 174 (2001) 614-648 - locally
//!     divergence-free prolongation. NAMED AND NOT IMPLEMENTED; the face flux
//!     is not prolonged here and SPEC-LIT section 75.10 records that.
//! No GPL-licensed source was consulted.
//!
//! # What must be conserved, and why one formula does all three cases
//!
//! Every equation this crate solves is
//!
//! ```text
//! d(rho phi)/dt + div(rho u phi) = div(Gamma grad phi) + S,   phi in {1, u, h, Y}
//! ```
//!
//! so the conserved quantity per cell is `rho phi V` and an adapt must leave
//! `sum_P rho_P phi_P V_P` alone. Mass is the case `phi = 1`; energy is
//! `phi = h`.
//!
//! Write `C(p)` for the new cells old cell `p` feeds and `S(q)` for the old
//! cells new cell `q` draws from, and give every pair a weight
//!
//! ```text
//! w_qp = V_q / sum_{q' in C(p)} V_q'      so that   sum_{q in C(p)} w_qp = 1
//! ```
//!
//! which is exactly `1` for a kept or coarsened cell (the sum has one term)
//! and the child's volume share for a refined one. Then
//!
//! ```text
//! rho_q V_q          = sum_{p in S(q)} w_qp rho_p V_p
//! rho_q phi_q V_q    = sum_{p in S(q)} w_qp rho_p V_p phihat_qp
//! ```
//!
//! **and that single pair of gathers is restriction, prolongation and the
//! identity at once.** Summing the first over `q` gives
//! `sum_p rho_p V_p sum_q w_qp = sum_p rho_p V_p`: mass is conserved with no
//! rescale and no correction pass.
//!
//! # The reconstruction, and the rescale that is not needed
//!
//! ```text
//! phihat_qp = phi_p + Psi_p (grad phi)_p . (x_q - xbar_p)
//! xbar_p    = sum_{q in C(p)} w_qp x_q
//! ```
//!
//! The reconstruction is centred on `xbar_p`, the **weight-weighted centroid
//! of the children**, and not on the parent's own centre. That one change is
//! what makes the second sum telescope:
//!
//! ```text
//! sum_q w_qp phihat_qp = phi_p + Psi_p (grad phi)_p . ( sum_q w_qp x_q - xbar_p )
//!                      = phi_p                                    exactly
//! ```
//!
//! The design note of record instead prescribed a multiplicative rescale
//!
//! ```text
//! lambda_p = rho_p phi_p V_p / sum_c rho_c phihat_c V_c ,  phi_c = lambda_p phihat_c
//! ```
//!
//! **which is singular.** Any field whose volume-weighted mean over the parent
//! is zero - a velocity component in a recirculation, a temperature written as
//! a fluctuation, `p_rgh` itself - divides by zero, and near-zero means an
//! arbitrarily large `lambda` that destroys monotonicity. The recentred form
//! above is exact, is never singular, preserves the reconstructed gradient
//! exactly, and costs one extra gather over the children.
//! `tests::the_multiplicative_rescale_is_singular_where_the_recentred_form_is_not`
//! is the measurement of that difference on a field with zero mean.
//!
//! On an exact hexahedral split the children's volumes are equal and `xbar_p`
//! is the parent's own centroid, so the recentring is arithmetically inert;
//! it earns its place on the round-off that the geometry sweep leaves behind,
//! and on any future split whose children are not congruent.
//!
//! # Monotonicity
//!
//! `Psi_p` is the Barth-Jespersen limiter evaluated at the reconstruction
//! points, so `phihat_qp` lies between the minimum and maximum of `phi` over
//! `p` and its face neighbours - hence between the global minimum and maximum.
//! A prolongation cannot invent an extremum, which is what stops a refine
//! turning a bounded mass fraction into one outside `[0, 1]`.

use crate::adapt::{AdaptKernels, Map};
use cudarc::driver::PushKernelArg;

use crate::device::{cfg_for, Gpu};
use crate::error::{Error, Result};
use crate::mesh::{GpuMesh, HostMesh};
use crate::{DevBuf, Label, Scalar, Vec3};

/// How a refined child's value is reconstructed from its parent's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prolongation {
    /// `phi_c = phi_p`. Conservative, monotone, and first-order.
    Constant,
    /// `phi_c = phi_p + Psi_p grad(phi)_p . (x_c - xbar_p)` with the
    /// Barth-Jespersen limiter. Conservative, monotone, and second-order.
    LimitedLinear,
}

/// The per-old-cell quantities the transfer needs: the weight-weighted
/// centroid of the cells an old cell feeds, and the sum of the weights.
///
/// `wsum` is `1` by construction and is kept only so the gate can check it.
#[derive(Debug, Clone, Default)]
pub struct ParentTargets {
    pub xbar: Vec<Vec3>,
    pub wsum: Vec<Scalar>,
}

/// `xbar_p = sum_{q in C(p)} w_qp x_q`, one gather per old cell over the
/// transpose map.
pub fn parent_targets(map: &Map, c_new: &[Vec3]) -> Result<ParentTargets> {
    if c_new.len() != map.n_new {
        return Err(Error::Mesh(format!(
            "the new mesh has {} cell centres and the map {} new cells",
            c_new.len(),
            map.n_new
        )));
    }
    let mut xbar = vec![Vec3::ZERO; map.n_old];
    let mut wsum = vec![0.0 as Scalar; map.n_old];
    for p in 0..map.n_old {
        let (a, b) = (map.own_offset[p] as usize, map.own_offset[p + 1] as usize);
        let mut x = Vec3::ZERO;
        let mut w = 0.0;
        for i in a..b {
            let q = map.own_child[i] as usize;
            x += c_new[q] * map.own_w[i];
            w += map.own_w[i];
        }
        xbar[p] = x;
        wsum[p] = w;
    }
    Ok(ParentTargets { xbar, wsum })
}

/// The Barth-Jespersen limiter, evaluated at the reconstruction points an
/// adapt actually uses - the centres of the cells this old cell feeds.
///
/// ```text
/// phi_max = max(phi_p, phi over the face neighbours of p)
/// phi_min = min(...)
/// D_q     = grad(phi)_p . (x_q - xbar_p)
/// psi_q   = 1                            |D_q| below the floor
///         = min(1, (phi_max - phi_p)/D)  D > 0
///         = min(1, (phi_min - phi_p)/D)  D < 0
/// Psi_p   = min_q psi_q
/// ```
///
/// The neighbourhood extrema are gathered over the OLD mesh's cell -> face
/// CSR, and a boundary face contributes its evaluated face value, so a cell
/// against a Dirichlet wall is limited by the wall value rather than by
/// nothing.
#[allow(clippy::too_many_arguments)]
pub fn barth_jespersen(
    out: &mut Vec<Scalar>,
    phi: &[Scalar],
    bphi: &[Scalar],
    grad: &[Vec3],
    tgt: &ParentTargets,
    map: &Map,
    c_new: &[Vec3],
    m: &HostMesh,
) {
    out.clear();
    out.resize(map.n_old, 1.0);

    let mut lo: Vec<Scalar> = phi[..m.n_cells].to_vec();
    let mut hi: Vec<Scalar> = phi[..m.n_cells].to_vec();
    for f in 0..m.n_internal_faces {
        let (p, n) = (m.owner[f] as usize, m.neighbour[f] as usize);
        lo[p] = lo[p].min(phi[n]);
        hi[p] = hi[p].max(phi[n]);
        lo[n] = lo[n].min(phi[p]);
        hi[n] = hi[n].max(phi[p]);
    }
    // `bf` indexes the face-cell map and the patch lookup as well as `bphi`.
    #[allow(clippy::needless_range_loop)]
    for bf in 0..m.n_boundary_faces {
        if crate::reference::is_empty_face(m, bf) {
            continue;
        }
        let p = m.b_face_cells[bf] as usize;
        lo[p] = lo[p].min(bphi[bf]);
        hi[p] = hi[p].max(bphi[bf]);
    }

    for p in 0..map.n_old {
        let (a, b) = (map.own_offset[p] as usize, map.own_offset[p + 1] as usize);
        let mut psi = 1.0 as Scalar;
        for i in a..b {
            let q = map.own_child[i] as usize;
            let d = grad[p].dot(c_new[q] - tgt.xbar[p]);
            let s = if d > LIMITER_FLOOR {
                ((hi[p] - phi[p]) / d).min(1.0)
            } else if d < -LIMITER_FLOOR {
                ((lo[p] - phi[p]) / d).min(1.0)
            } else {
                1.0
            };
            psi = psi.min(s.max(0.0));
        }
        out[p] = psi;
    }
}

/// Below this the reconstruction offset is treated as zero and the limiter
/// leaves the cell alone. `1e-300` is far below any physical increment and
/// far above the point where `(phi_max - phi_p)/D` overflows.
pub const LIMITER_FLOOR: Scalar = 1e-300;

/// `rho_q V_q = sum_p w_qp rho_p V_p`, one gather per new cell.
pub fn transfer_density(
    out: &mut Vec<Scalar>,
    rho: &[Scalar],
    v_old: &[Scalar],
    v_new: &[Scalar],
    map: &Map,
) -> Result<()> {
    check(map, rho.len(), v_old.len(), v_new.len())?;
    out.clear();
    out.resize(map.n_new, 0.0);
    for q in 0..map.n_new {
        let (a, b) = (map.src_offset[q] as usize, map.src_offset[q + 1] as usize);
        let mut mass = 0.0;
        for i in a..b {
            let p = map.src_cell[i] as usize;
            mass += map.src_w[i] * rho[p] * v_old[p];
        }
        out[q] = mass / v_new[q];
    }
    Ok(())
}

/// The conserved transfer of a mass-weighted scalar: restriction, prolongation
/// and the identity, in one gather.
///
/// `rho_new` must be the output of [`transfer_density`] on the same map, so
/// that the mass this divides by is the mass that was transferred.
#[allow(clippy::too_many_arguments)]
pub fn transfer_scalar(
    out: &mut Vec<Scalar>,
    phi: &[Scalar],
    rho: &[Scalar],
    grad: &[Vec3],
    psi: &[Scalar],
    tgt: &ParentTargets,
    v_old: &[Scalar],
    v_new: &[Scalar],
    c_new: &[Vec3],
    map: &Map,
    mode: Prolongation,
) -> Result<()> {
    check(map, rho.len(), v_old.len(), v_new.len())?;
    if phi.len() < map.n_old {
        return Err(Error::Field {
            field: "phi".to_string(),
            msg: format!("has {} entries, the old mesh has {}", phi.len(), map.n_old),
        });
    }
    out.clear();
    out.resize(map.n_new, 0.0);
    for q in 0..map.n_new {
        let (a, b) = (map.src_offset[q] as usize, map.src_offset[q + 1] as usize);
        let mut mass = 0.0 as Scalar;
        let mut mom = 0.0 as Scalar;
        for i in a..b {
            let p = map.src_cell[i] as usize;
            let m = map.src_w[i] * rho[p] * v_old[p];
            let hat = match mode {
                Prolongation::Constant => phi[p],
                Prolongation::LimitedLinear => {
                    phi[p] + psi[p] * grad[p].dot(c_new[q] - tgt.xbar[p])
                }
            };
            mass += m;
            mom += m * hat;
        }
        out[q] = if mass != 0.0 { mom / mass } else { 0.0 };
    }
    Ok(())
}

fn check(map: &Map, n_rho: usize, n_vold: usize, n_vnew: usize) -> Result<()> {
    if n_rho < map.n_old || n_vold < map.n_old {
        return Err(Error::Mesh(format!(
            "the old fields have {n_rho}/{n_vold} entries, the old mesh has {}",
            map.n_old
        )));
    }
    if n_vnew != map.n_new {
        return Err(Error::Mesh(format!(
            "the new volume array has {n_vnew} entries, the new mesh has {}",
            map.n_new
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
//  The conservation audit
// ---------------------------------------------------------------------------

/// The three integrals an adapt must not move, and the two bounds it must not
/// widen.
///
/// The sums are computed with [`crate::exactsum::host_exact_sum`] rather than
/// by accumulation, so that a gate at `1e-14` is measuring the transfer and
/// not the summation order - which on 10^4 cells is itself worth `1e-14`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Integrals {
    pub volume: Scalar,
    pub mass: Scalar,
    pub energy: Scalar,
    pub min: Scalar,
    pub max: Scalar,
}

impl Integrals {
    pub fn of(rho: &[Scalar], phi: &[Scalar], v: &[Scalar]) -> Self {
        let n = v.len();
        let mass: Vec<Scalar> = (0..n).map(|c| rho[c] * v[c]).collect();
        let energy: Vec<Scalar> = (0..n).map(|c| rho[c] * phi[c] * v[c]).collect();
        Self {
            volume: crate::exactsum::host_exact_sum(v),
            mass: crate::exactsum::host_exact_sum(&mass),
            energy: crate::exactsum::host_exact_sum(&energy),
            min: phi[..n].iter().cloned().fold(Scalar::INFINITY, Scalar::min),
            max: phi[..n].iter().cloned().fold(Scalar::NEG_INFINITY, Scalar::max),
        }
    }

    /// Relative drift of each integral against a reference, as
    /// `(volume, mass, energy)`. A zero reference is reported as an absolute
    /// difference rather than as a division by zero.
    pub fn drift(&self, r: &Self) -> (Scalar, Scalar, Scalar) {
        let rel = |a: Scalar, b: Scalar| {
            let d = (a - b).abs();
            if b.abs() > 0.0 {
                d / b.abs()
            } else {
                d
            }
        };
        (rel(self.volume, r.volume), rel(self.mass, r.mass), rel(self.energy, r.energy))
    }
}

// ---------------------------------------------------------------------------
//  The device side
// ---------------------------------------------------------------------------

/// The map, on the device.
pub struct GpuMap {
    pub n_old: usize,
    pub n_new: usize,
    pub src_offset: DevBuf<Label>,
    pub src_cell: DevBuf<Label>,
    pub src_w: DevBuf<Scalar>,
    pub own_offset: DevBuf<Label>,
    pub own_child: DevBuf<Label>,
    pub own_w: DevBuf<Scalar>,
}

impl GpuMap {
    pub fn upload(gpu: &Gpu, m: &Map) -> Result<Self> {
        Ok(Self {
            n_old: m.n_old,
            n_new: m.n_new,
            src_offset: gpu.upload(&m.src_offset)?,
            src_cell: gpu.upload(&m.src_cell)?,
            src_w: gpu.upload(&m.src_w)?,
            own_offset: gpu.upload(&m.own_offset)?,
            own_child: gpu.upload(&m.own_child)?,
            own_w: gpu.upload(&m.own_w)?,
        })
    }
}

/// `xbar` and `wsum`, one thread per OLD cell.
pub fn gpu_parent_targets(
    gpu: &Gpu,
    k: &AdaptKernels,
    xbar: &mut DevBuf<Vec3>,
    wsum: &mut DevBuf<Scalar>,
    map: &GpuMap,
    c_new: &DevBuf<Vec3>,
) -> Result<()> {
    if map.n_old == 0 {
        return Ok(());
    }
    let n = map.n_old as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.parent_targets)
            .arg(&mut *xbar)
            .arg(&mut *wsum)
            .arg(&map.own_offset)
            .arg(&map.own_child)
            .arg(&map.own_w)
            .arg(c_new)
            .arg(&n)
            .launch(cfg_for(map.n_old))?;
    }
    Ok(())
}

/// The Barth-Jespersen limiter, one thread per OLD cell.
#[allow(clippy::too_many_arguments)]
pub fn gpu_barth_jespersen(
    gpu: &Gpu,
    k: &AdaptKernels,
    out: &mut DevBuf<Scalar>,
    phi: &DevBuf<Scalar>,
    bphi: &DevBuf<Scalar>,
    grad: &DevBuf<Vec3>,
    xbar: &DevBuf<Vec3>,
    map: &GpuMap,
    c_new: &DevBuf<Vec3>,
    m: &GpuMesh,
) -> Result<()> {
    if map.n_old == 0 {
        return Ok(());
    }
    let n = map.n_old as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.limiter)
            .arg(&mut *out)
            .arg(phi)
            .arg(bphi)
            .arg(grad)
            .arg(xbar)
            .arg(&map.own_offset)
            .arg(&map.own_child)
            .arg(c_new)
            .arg(&m.owner)
            .arg(&m.neighbour)
            .arg(&m.cf_offset)
            .arg(&m.cf_face)
            .arg(&m.cf_own)
            .arg(&m.bcf_offset)
            .arg(&m.bcf_face)
            .arg(&m.b_kind)
            .arg(&n)
            .launch(cfg_for(map.n_old))?;
    }
    Ok(())
}

/// `rho_q V_q = sum_p w_qp rho_p V_p`, one thread per NEW cell.
pub fn gpu_transfer_density(
    gpu: &Gpu,
    k: &AdaptKernels,
    out: &mut DevBuf<Scalar>,
    rho: &DevBuf<Scalar>,
    v_old: &DevBuf<Scalar>,
    v_new: &DevBuf<Scalar>,
    map: &GpuMap,
) -> Result<()> {
    if map.n_new == 0 {
        return Ok(());
    }
    let n = map.n_new as Label;
    unsafe {
        gpu.stream()
            .launch_builder(&k.transfer_density)
            .arg(&mut *out)
            .arg(rho)
            .arg(v_old)
            .arg(v_new)
            .arg(&map.src_offset)
            .arg(&map.src_cell)
            .arg(&map.src_w)
            .arg(&n)
            .launch(cfg_for(map.n_new))?;
    }
    Ok(())
}

/// The conserved scalar transfer, one thread per NEW cell.
#[allow(clippy::too_many_arguments)]
pub fn gpu_transfer_scalar(
    gpu: &Gpu,
    k: &AdaptKernels,
    out: &mut DevBuf<Scalar>,
    phi: &DevBuf<Scalar>,
    rho: &DevBuf<Scalar>,
    grad: &DevBuf<Vec3>,
    psi: &DevBuf<Scalar>,
    xbar: &DevBuf<Vec3>,
    v_old: &DevBuf<Scalar>,
    v_new: &DevBuf<Scalar>,
    c_new: &DevBuf<Vec3>,
    map: &GpuMap,
    mode: Prolongation,
) -> Result<()> {
    if map.n_new == 0 {
        return Ok(());
    }
    let n = map.n_new as Label;
    let use_grad: Label = match mode {
        Prolongation::Constant => 0,
        Prolongation::LimitedLinear => 1,
    };
    unsafe {
        gpu.stream()
            .launch_builder(&k.transfer_scalar)
            .arg(&mut *out)
            .arg(phi)
            .arg(rho)
            .arg(grad)
            .arg(psi)
            .arg(xbar)
            .arg(v_old)
            .arg(v_new)
            .arg(c_new)
            .arg(&map.src_offset)
            .arg(&map.src_cell)
            .arg(&map.src_w)
            .arg(&use_grad)
            .arg(&n)
            .launch(cfg_for(map.n_new))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapt::{plan, Forest, Mark};
    use crate::mesh::refined::RefinedBox;

    const CUBE: Vec3 = Vec3::new(0.125, 0.125, 0.125);

    /// A smooth blob and a linearly varying density, so that nothing in the
    /// transfer can hide behind a constant.
    fn fields(m: &HostMesh) -> (Vec<Scalar>, Vec<Scalar>, Vec<Scalar>, Vec<Scalar>) {
        let blob = |p: Vec3| {
            let r = ((p.x - 0.55).powi(2) + (p.y - 0.45).powi(2) + (p.z - 0.5).powi(2)).sqrt();
            0.2 + (-((r / 0.22).powi(2))).exp()
        };
        let dens = |p: Vec3| 1.0 + 0.3 * p.x - 0.2 * p.y + 0.1 * p.z;
        (
            m.c.iter().map(|&p| blob(p)).collect(),
            m.b_cf.iter().map(|&p| blob(p)).collect(),
            m.c.iter().map(|&p| dens(p)).collect(),
            m.b_cf.iter().map(|&p| dens(p)).collect(),
        )
    }

    struct Step {
        after: Forest,
        mesh: RefinedBox,
        rho: Vec<Scalar>,
        phi: Vec<Scalar>,
        before: Integrals,
        after_i: Integrals,
    }

    #[allow(clippy::too_many_arguments)]
    fn adapt_once(
        f: &Forest,
        r: &RefinedBox,
        rho: &[Scalar],
        phi: &[Scalar],
        bphi: &[Scalar],
        mark: &[Mark],
        l_max: u32,
        mode: Prolongation,
    ) -> Step {
        let m = &r.mesh;
        let p = plan(f, m, mark, l_max).unwrap();
        let nm = &p.mesh.mesh;
        let tgt = parent_targets(&p.map, &nm.c).unwrap();
        let mut grad = Vec::new();
        crate::reference::fvc_grad_scalar(&mut grad, phi, bphi, m);
        let mut psi = Vec::new();
        barth_jespersen(&mut psi, phi, bphi, &grad, &tgt, &p.map, &nm.c, m);

        let mut rho_new = Vec::new();
        transfer_density(&mut rho_new, rho, &m.v, &nm.v, &p.map).unwrap();
        let mut phi_new = Vec::new();
        transfer_scalar(
            &mut phi_new, phi, rho, &grad, &psi, &tgt, &m.v, &nm.v, &nm.c, &p.map, mode,
        )
        .unwrap();

        let before = Integrals::of(rho, phi, &m.v);
        let after_i = Integrals::of(&rho_new, &phi_new, &nm.v);
        Step { after: p.after, mesh: p.mesh, rho: rho_new, phi: phi_new, before, after_i }
    }

    /// THE GATE. Mass and energy across a refine, and across a coarsen, to
    /// round-off - a refine that loses heat is not a refine.
    #[test]
    fn mass_and_energy_survive_a_refine_and_a_coarsen() {
        for mode in [Prolongation::Constant, Prolongation::LimitedLinear] {
            let f = Forest::uniform([8, 8, 8], CUBE).unwrap();
            let r = f.build().unwrap();
            let (phi, bphi, rho, _) = fields(&r.mesh);

            // ---- refine the blob ------------------------------------------
            let mark: Vec<Mark> = (0..f.len())
                .map(|c| if phi[c] > 0.7 { Mark::Refine } else { Mark::Keep })
                .collect();
            assert!(mark.contains(&Mark::Refine));
            let up = adapt_once(&f, &r, &rho, &phi, &bphi, &mark, 2, mode);
            let (dv, dm, de) = up.after_i.drift(&up.before);
            assert!(dv < 1e-14, "{mode:?}: volume drifted {dv:e} on a refine");
            assert!(dm < 1e-14, "{mode:?}: mass drifted {dm:e} on a refine");
            assert!(de < 1e-14, "{mode:?}: energy drifted {de:e} on a refine");
            assert!(
                up.after_i.max <= up.before.max + 1e-14
                    && up.after_i.min >= up.before.min - 1e-14,
                "{mode:?}: a refine invented an extremum: [{}, {}] from [{}, {}]",
                up.after_i.min,
                up.after_i.max,
                up.before.min,
                up.before.max
            );
            assert!(up.after.len() > f.len());

            // ---- and coarsen it all the way back --------------------------
            let mark = vec![Mark::Coarsen; up.after.len()];
            let bphi2: Vec<Scalar> =
                up.mesh.mesh.b_cf.iter().map(|_| 0.2 as Scalar).collect();
            let down =
                adapt_once(&up.after, &up.mesh, &up.rho, &up.phi, &bphi2, &mark, 2, mode);
            assert!(down.after.len() < up.after.len(), "the coarsen must remove cells");
            let (dv, dm, de) = down.after_i.drift(&down.before);
            assert!(dv < 1e-14, "{mode:?}: volume drifted {dv:e} on a coarsen");
            assert!(dm < 1e-14, "{mode:?}: mass drifted {dm:e} on a coarsen");
            assert!(de < 1e-14, "{mode:?}: energy drifted {de:e} on a coarsen");
            assert!(
                down.after_i.max <= down.before.max + 1e-14
                    && down.after_i.min >= down.before.min - 1e-14,
                "{mode:?}: a coarsen invented an extremum"
            );
        }
    }

    /// The round trip the brief asks for, measured rather than assumed - and
    /// the measurement is the finding. A refine followed by the coarsen that
    /// undoes it returns the field to ROUND-OFF, not to the interpolation
    /// error, for either prolongation, because restriction is the exact left
    /// inverse of any conservative prolongation into a complete family.
    ///
    /// The direction with a real error is the other one, and
    /// `coarsen_then_refine_loses_information_at_the_order_the_prolongation_promises`
    /// measures it.
    #[test]
    fn refine_then_coarsen_returns_the_field_to_round_off() {
        for mode in [Prolongation::Constant, Prolongation::LimitedLinear] {
            let f = Forest::uniform([8, 8, 8], CUBE).unwrap();
            let r = f.build().unwrap();
            let (phi, bphi, rho, _) = fields(&r.mesh);

            let mark: Vec<Mark> = (0..f.len())
                .map(|c| if phi[c] > 0.7 { Mark::Refine } else { Mark::Keep })
                .collect();
            let up = adapt_once(&f, &r, &rho, &phi, &bphi, &mark, 2, mode);

            // Coarsen exactly what was refined, and nothing else.
            let lv = up.after.levels();
            let mark: Vec<Mark> = (0..up.after.len())
                .map(|c| if lv[c] > 0 { Mark::Coarsen } else { Mark::Keep })
                .collect();
            let bphi2: Vec<Scalar> =
                up.mesh.mesh.b_cf.iter().map(|_| 0.2 as Scalar).collect();
            let down =
                adapt_once(&up.after, &up.mesh, &up.rho, &up.phi, &bphi2, &mark, 2, mode);
            assert_eq!(down.after.len(), f.len(), "the round trip must restore the mesh");

            let mut worst_phi = 0.0 as Scalar;
            let mut worst_rho = 0.0 as Scalar;
            for c in 0..f.len() {
                worst_phi = worst_phi.max((down.phi[c] - phi[c]).abs() / phi[c].abs());
                worst_rho = worst_rho.max((down.rho[c] - rho[c]).abs() / rho[c].abs());
            }
            assert!(
                worst_phi < 1e-14 && worst_rho < 1e-14,
                "{mode:?}: the round trip is not round-off: phi {worst_phi:e}, \
                 rho {worst_rho:e}"
            );
        }
    }

    /// The direction that DOES lose information: coarsen a resolved field and
    /// refine it back. The error is the prolongation's own, and it converges
    /// at the order the prolongation promises - first for piecewise-constant,
    /// second for limited-linear. Measured on three meshes, not assumed.
    #[test]
    fn coarsen_then_refine_loses_information_at_the_order_the_prolongation_promises() {
        let smooth = |p: Vec3| 1.0 + (2.0 * p.x).sin() * (1.7 * p.y).cos() + 0.4 * p.z * p.z;
        let mut err: Vec<(Prolongation, Vec<Scalar>)> =
            vec![(Prolongation::Constant, Vec::new()), (Prolongation::LimitedLinear, Vec::new())];

        for n in [8usize, 16, 32] {
            let d = Vec3::new(1.0 / n as Scalar, 1.0 / n as Scalar, 1.0 / n as Scalar);
            // Start one level down everywhere, so that "coarsen then refine"
            // is expressible on the whole mesh.
            let f0 = Forest::from_base_levels([n / 2, n / 2, n / 2], d * 2.0, &vec![
                1u32;
                (n / 2)
                    * (n / 2)
                    * (n / 2)
            ])
            .unwrap();
            let r0 = f0.build().unwrap();
            let phi: Vec<Scalar> = r0.mesh.c.iter().map(|&p| smooth(p)).collect();
            let bphi: Vec<Scalar> = r0.mesh.b_cf.iter().map(|&p| smooth(p)).collect();
            let rho = vec![1.0 as Scalar; f0.len()];

            for (mode, e) in err.iter_mut() {
                let down = adapt_once(
                    &f0,
                    &r0,
                    &rho,
                    &phi,
                    &bphi,
                    &vec![Mark::Coarsen; f0.len()],
                    1,
                    *mode,
                );
                let bd: Vec<Scalar> =
                    down.mesh.mesh.b_cf.iter().map(|&p| smooth(p)).collect();
                let up = adapt_once(
                    &down.after,
                    &down.mesh,
                    &down.rho,
                    &down.phi,
                    &bd,
                    &vec![Mark::Refine; down.after.len()],
                    1,
                    *mode,
                );
                assert_eq!(up.after.len(), f0.len());
                let l2: Scalar = (up
                    .phi
                    .iter()
                    .zip(phi.iter())
                    .zip(r0.mesh.v.iter())
                    .map(|((a, b), v)| (a - b) * (a - b) * v)
                    .sum::<Scalar>())
                .sqrt();
                e.push(l2);
            }
        }

        for (mode, e) in &err {
            let orders: Vec<Scalar> =
                e.windows(2).map(|w| (w[0] / w[1]).ln() / (2.0 as Scalar).ln()).collect();
            let last = orders[orders.len() - 1];
            // Printed, not only asserted: SPEC-LIT S75.7 quotes these numbers
            // and they have to be reproducible from this test's own output.
            println!("  {mode:?}: L2 errors {e:?}, observed orders {orders:?}");
            match mode {
                Prolongation::Constant => assert!(
                    (0.7..1.4).contains(&last),
                    "piecewise-constant prolongation should be first order; \
                     errors {e:?}, orders {orders:?}"
                ),
                Prolongation::LimitedLinear => assert!(
                    last > 1.8,
                    "limited-linear prolongation should be second order; \
                     errors {e:?}, orders {orders:?}"
                ),
            }
        }
    }

    /// The reason this file does not use the design note's multiplicative
    /// rescale. On a field with zero volume-weighted mean over a parent, the
    /// rescale divides by (near) zero; the recentred reconstruction is exact
    /// and finite on the same data.
    #[test]
    fn the_multiplicative_rescale_is_singular_where_the_recentred_form_is_not() {
        let f = Forest::uniform([4, 4, 4], CUBE).unwrap();
        let r = f.build().unwrap();
        let m = &r.mesh;
        // A field whose value is exactly zero in the cell being refined.
        let c0 = 21usize;
        let phi: Vec<Scalar> = m.c.iter().map(|&p| p.x - m.c[c0].x).collect();
        let bphi: Vec<Scalar> = m.b_cf.iter().map(|&p| p.x - m.c[c0].x).collect();
        let rho = vec![1.0 as Scalar; m.n_cells];
        assert_eq!(phi[c0], 0.0);

        let mut mark = vec![Mark::Keep; f.len()];
        mark[c0] = Mark::Refine;
        let s = adapt_once(&f, &r, &rho, &phi, &bphi, &mark, 2, Prolongation::LimitedLinear);
        for (q, &v) in s.phi.iter().enumerate() {
            assert!(v.is_finite(), "new cell {q} came out {v}");
        }
        let (_, dm, de) = s.after_i.drift(&s.before);
        assert!(dm < 1e-14 && de < 1e-13, "mass {dm:e}, energy {de:e}");

        // What the rescale would have been: the denominator is the mass-
        // weighted sum over the children of a field that is antisymmetric
        // about the parent centre, so it is zero to round-off and lambda is
        // 1/0. Measured rather than argued.
        let p = plan(&f, m, &mark, 2).unwrap();
        let nm = &p.mesh.mesh;
        let (a, b) = (p.map.own_offset[c0] as usize, p.map.own_offset[c0 + 1] as usize);
        let mut grad = Vec::new();
        crate::reference::fvc_grad_scalar(&mut grad, &phi, &bphi, m);
        let mut den = 0.0 as Scalar;
        for i in a..b {
            let q = p.map.own_child[i] as usize;
            den += rho[c0] * (phi[c0] + grad[c0].dot(nm.c[q] - m.c[c0])) * nm.v[q];
        }
        let scale = rho[c0] * m.v[c0] * grad[c0].mag() * m.v[c0].cbrt();
        assert!(
            den.abs() < 1e-14 * scale,
            "the rescale's denominator should vanish here; it is {den:e} against a \
             scale of {scale:e}"
        );
    }

    /// The weights sum to one for every old cell. That single identity is the
    /// whole conservation argument, so it is asserted directly rather than
    /// only through its consequences.
    #[test]
    fn the_conservative_weights_sum_to_one() {
        let f = Forest::uniform([6, 6, 6], CUBE).unwrap();
        let r = f.build().unwrap();
        let mut mark = vec![Mark::Keep; f.len()];
        for (c, mk) in mark.iter_mut().enumerate() {
            if c % 5 == 0 {
                *mk = Mark::Refine;
            }
        }
        let p = plan(&f, &r.mesh, &mark, 2).unwrap();
        let t = parent_targets(&p.map, &p.mesh.mesh.c).unwrap();
        for (c, &w) in t.wsum.iter().enumerate() {
            assert!((w - 1.0).abs() < 1e-15, "cell {c}: weights sum to {w}");
        }
        // And xbar is the parent's own centre, to the round-off the geometry
        // sweep leaves - which is what makes the recentring inert on an exact
        // hexahedral split.
        let mut worst = 0.0 as Scalar;
        for c in 0..p.map.n_old {
            worst = worst.max((t.xbar[c] - r.mesh.c[c]).mag());
        }
        assert!(worst < 1e-15, "xbar is {worst:e} from the parent centre");
    }
}
