// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The block-coupled solid matrix of SPEC-LIT §109: a second storage format
//! beside §1's one-entry-per-face LDU - one `3x3` block per cell and per
//! internal face per direction, one `Vec3` source per cell - and the assembly
//! of (95.3)'s face sum into it with the normal-derivative row of the face
//! gradient two-point and implicit in all three terms of `sigma_f.Sf`
//! ((109.1)-(109.5)). Nothing here solves or iterates: the solve around this
//! matrix is §109's unwritten half.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §1 (the storage this sits beside), §2.4 (the
//!     over-relaxed correction, applied to all three components at once),
//!     §3.5 (the Green-Gauss gradient the tangential rows are read from),
//!     §4 and §95.2 (the per-component boundary statement, kept exactly),
//!     §95.1 (the split this operator moves half of into the matrix), §109
//!   P. Cardiff, Ž. Tuković, H. Jasak, A. Ivanković, *Comput. Struct.* 175
//!     (2016) 100-122, DOI 10.1016/j.compstruc.2016.07.004 - the IDEA of a
//!     block-coupled cell-centred finite-volume matrix for the three
//!     displacement components. Cited for that and for nothing else: the
//!     coefficients below are derived from (95.3) and (95.5) in §109.2, and
//!     the paper's own tangential stencil is not built (§109.4)
//!   I. Demirdžić, S. Muzaferija, *Int. J. Numer. Methods Eng.* 37 (1994)
//!     3751-3766, DOI 10.1002/nme.1620372110 - a traction face's contribution
//!     to the equilibrium sum IS the prescribed traction
//!   H. Jasak, H. G. Weller, *Int. J. Numer. Methods Eng.* 48 (2000) 267-287,
//!     DOI 10.1002/(SICI)1097-0207(20000520)48:2<267::AID-NME884>3.0.CO;2-Q -
//!     the split whose deferred half this matrix carries implicitly
//!   I. Demirdžić, D. Martinović, *Comput. Methods Appl. Mech. Eng.* 109
//!     (1993) 331-349, DOI 10.1016/0045-7825(93)90085-C - the thermal term
//!
//! The host half is written as scatter loops over faces in `src/reference.rs`'s
//! style; the device half (in this file, over `cuda/solidblock.cu`) gathers one
//! row per thread, so their agreement means something. OpenFOAM and solids4foam are GPL and
//! were not opened; no solid-mechanics solver of any licence was consulted.
//! No GPL-licensed source was consulted.

use crate::error::{Error, Result};
use crate::mesh::HostMesh;
use crate::reference;
use crate::{Label, Scalar, Tensor, Vec3};

use super::bc::CompBc;

/// SPEC-LIT §109.1: the block system on the host - one `3x3` per cell, one per internal face per
/// direction, one `Vec3` per cell. No boundary-coefficient pair: a boundary face folds into `diag`
/// and `source` at assembly, because the operator refuses every coupled patch.
#[derive(Debug, Clone, PartialEq)]
pub struct HostBlockLdu {
    pub n_cells: usize,
    pub n_internal_faces: usize,
    pub diag: Vec<Tensor>,
    pub upper: Vec<Tensor>,
    pub lower: Vec<Tensor>,
    pub source: Vec<Vec3>,
}

impl HostBlockLdu {
    pub fn zeros(m: &HostMesh) -> Self {
        Self {
            n_cells: m.n_cells,
            n_internal_faces: m.n_internal_faces,
            diag: vec![Tensor::ZERO; m.n_cells],
            upper: vec![Tensor::ZERO; m.n_internal_faces],
            lower: vec![Tensor::ZERO; m.n_internal_faces],
            source: vec![Vec3::ZERO; m.n_cells],
        }
    }

    pub fn zero(&mut self) {
        self.diag.fill(Tensor::ZERO);
        self.upper.fill(Tensor::ZERO);
        self.lower.fill(Tensor::ZERO);
        self.source.fill(Vec3::ZERO);
    }
}

/// Everything one assembly reads (§109.2), as host slices; `check` refuses every wrong length and
/// every coupled patch by name (`Error::Config`, text starting `solid block:`).
pub struct BlockState<'a> {
    pub u: &'a [Vec3],
    pub ub: &'a [Vec3],
    pub grad: &'a [Tensor],
    pub t: &'a [Scalar],
    pub bt: &'a [Scalar],
    pub mask: &'a [Label],
    pub ref_value: &'a [Vec3],
    pub traction: &'a [Vec3],
    pub mu: &'a [Scalar],
    pub lambda: &'a [Scalar],
    pub beta_alpha: &'a [Scalar],
    pub t_ref: &'a [Scalar],
}

impl<'a> BlockState<'a> {
    /// Refuses a slice whose length is not what its field's own shape says -
    /// `[n_cells]` or `[n_boundary_faces]`, both numbers in the message - and
    /// any coupled boundary face, naming its patch: the block format of
    /// §109.1 carries no boundary-coefficient pair, so nothing decomposes a
    /// couple.
    pub fn check(&self, m: &HostMesh) -> Result<()> {
        let want = [
            ("u", self.u.len(), m.n_cells),
            ("ub", self.ub.len(), m.n_boundary_faces),
            ("grad", self.grad.len(), m.n_cells),
            ("t", self.t.len(), m.n_cells),
            ("bt", self.bt.len(), m.n_boundary_faces),
            ("mask", self.mask.len(), m.n_boundary_faces),
            ("ref_value", self.ref_value.len(), m.n_boundary_faces),
            ("traction", self.traction.len(), m.n_boundary_faces),
            ("mu", self.mu.len(), m.n_cells),
            ("lambda", self.lambda.len(), m.n_cells),
            ("beta_alpha", self.beta_alpha.len(), m.n_cells),
            ("t_ref", self.t_ref.len(), m.n_cells),
        ];
        for (name, got, expect) in want {
            if got != expect {
                let which = if expect == m.n_cells {
                    "n_cells"
                } else {
                    "n_boundary_faces"
                };
                return Err(Error::Config(format!(
                    "solid block: `{name}` has length {got}, expected {expect} ({which})"
                )));
            }
        }
        for bf in 0..m.n_boundary_faces {
            if reference::is_coupled_face(m, bf) {
                return Err(Error::Config(format!(
                    "solid block: boundary face {bf} of patch `{}` is a coupled \
                     patch, and the block format of §109.1 has no pair to carry it",
                    m.patches[m.b_patch[bf] as usize].name
                )));
            }
        }
        Ok(())
    }
}

/// `SolidBcs::new`'s scatter loop without the upload: `(mask, ref_value, traction)` per boundary face.
pub fn scatter_bcs(m: &HostMesh, per_patch: &[[CompBc; 3]]) -> (Vec<Label>, Vec<Vec3>, Vec<Vec3>) {
    let n_bf = m.n_boundary_faces;
    let mut mask = vec![0 as Label; n_bf];
    let mut ref_value = vec![Vec3::ZERO; n_bf];
    let mut traction = vec![Vec3::ZERO; n_bf];
    for bf in 0..n_bf {
        let patch = per_patch[m.b_patch[bf] as usize];
        for i in 0..3 {
            match patch[i] {
                CompBc::Fixed(v) => {
                    mask[bf] |= 1 << i;
                    match i {
                        0 => ref_value[bf].x = v,
                        1 => ref_value[bf].y = v,
                        _ => ref_value[bf].z = v,
                    }
                }
                CompBc::Traction(t) => match i {
                    0 => traction[bf].x = t,
                    1 => traction[bf].y = t,
                    _ => traction[bf].z = t,
                },
            }
        }
    }
    (mask, ref_value, traction)
}

// ==========================================================================
//  The private contractions
// ==========================================================================

/// `(a^T G)_j = sum_i a_i G_ij`. Copied from `src/solid/prototype.rs`'s
/// private `dot_left` - the same bodies, made local here rather than made
/// `pub(super)` there, so that the prototype's own surface stays what it is.
#[inline]
fn dot_left(a: Vec3, g: Tensor) -> Vec3 {
    Vec3::new(
        a.x * g.xx + a.y * g.yx + a.z * g.zx,
        a.x * g.xy + a.y * g.yy + a.z * g.zy,
        a.x * g.xz + a.y * g.yz + a.z * g.zz,
    )
}

/// `(G^T.a)_j = sum_i G_ji a_i`. Copied from `src/solid/prototype.rs`'s
/// private `dot_right`, for the same reason.
#[inline]
fn dot_right(g: Tensor, a: Vec3) -> Vec3 {
    Vec3::new(
        g.xx * a.x + g.xy * a.y + g.xz * a.z,
        g.yx * a.x + g.yy * a.y + g.yz * a.z,
        g.zx * a.x + g.zy * a.y + g.zz * a.z,
    )
}

/// The ordinary product `(T v)_i = sum_j T_ij v_j` - a block acting on a
/// displacement (§109.1). The arithmetic is identical to [`dot_right`]'s; it
/// keeps its own name so a reader never has to ask whether a block or a
/// gradient is meant.
#[inline]
fn matvec(t: Tensor, v: Vec3) -> Vec3 {
    Vec3::new(
        t.xx * v.x + t.xy * v.y + t.xz * v.z,
        t.yx * v.x + t.yy * v.y + t.yz * v.z,
        t.zx * v.x + t.zy * v.y + t.zz * v.z,
    )
}

/// The ordinary product of two `3x3`s, `(A B)_ij = sum_k A_ik B_kj`. The only
/// use is (109.4)'s `P_b M_b P_b`.
#[inline]
fn matmul(a: Tensor, b: Tensor) -> Tensor {
    Tensor {
        xx: a.xx * b.xx + a.xy * b.yx + a.xz * b.zx,
        xy: a.xx * b.xy + a.xy * b.yy + a.xz * b.zy,
        xz: a.xx * b.xz + a.xy * b.yz + a.xz * b.zz,
        yx: a.yx * b.xx + a.yy * b.yx + a.yz * b.zx,
        yy: a.yx * b.xy + a.yy * b.yy + a.yz * b.zy,
        yz: a.yx * b.xz + a.yy * b.yz + a.yz * b.zz,
        zx: a.zx * b.xx + a.zy * b.yx + a.zz * b.zx,
        zy: a.zx * b.xy + a.zy * b.yy + a.zz * b.zy,
        zz: a.zx * b.xz + a.zy * b.yz + a.zz * b.zz,
    }
}

/// `n = Sf/|Sf|`, zero on a degenerate face - the same guard
/// `reference::boundary_corr_vector` applies.
#[inline]
fn normal_of(sf: Vec3, mag_sf: Scalar) -> Vec3 {
    if mag_sf > 0.0 {
        sf / mag_sf
    } else {
        Vec3::ZERO
    }
}

/// `M_b = mu I + (mu + lambda) n n^T` of (109.4) - which is also (109.1)'s
/// bracket and (109.3)'s, at whatever `|Sf| Delta` the caller multiplies in.
#[inline]
fn m_of(mu: Scalar, lambda: Scalar, n: Vec3) -> Tensor {
    let mut t = Vec3::outer(n, n) * (mu + lambda);
    t.xx += mu;
    t.yy += mu;
    t.zz += mu;
    t
}

/// (109.1): the face coefficient `B_f = |Sf| Delta [ mu I + (mu + lambda) n n^T ]`.
#[inline]
fn face_block(mu: Scalar, lambda: Scalar, sf: Vec3, mag_sf: Scalar, delta: Scalar) -> Tensor {
    m_of(mu, lambda, normal_of(sf, mag_sf)) * (mag_sf * delta)
}

/// `P_b`, the diagonal `0/1` matrix of the FIXED components (109.4): bit `i`
/// of the mask set means component `i` is `Fixed`.
#[inline]
fn mask_matrix(mask: Label) -> Tensor {
    Tensor {
        xx: (mask & 1) as Scalar,
        yy: ((mask >> 1) & 1) as Scalar,
        zz: ((mask >> 2) & 1) as Scalar,
        ..Tensor::ZERO
    }
}

/// The Frobenius norm of a `3x3`, the norm (109.6)'s scale is read with.
#[inline]
fn frobenius(t: Tensor) -> Scalar {
    t.ddot(t).sqrt()
}

/// Component `(i, j)` of a block, for [`dense_from_block`]'s expansion.
#[inline]
fn block_at(t: Tensor, i: usize, j: usize) -> Scalar {
    match (i, j) {
        (0, 0) => t.xx,
        (0, 1) => t.xy,
        (0, 2) => t.xz,
        (1, 0) => t.yx,
        (1, 1) => t.yy,
        (1, 2) => t.yz,
        (2, 0) => t.zx,
        (2, 1) => t.zy,
        _ => t.zz,
    }
}

/// (109.3)'s `q_f`, the explicit face vector for the ROW cell whose material
/// is given: the over-relaxed correction of the normal row - `Gbar_f.k_f` is
/// exactly `reference.rs`'s `corr`, applied to all three components at once -
/// plus the deferred tangential part `mu G'_f^T.Sf + lambda tr(G'_f) Sf`.
#[inline]
fn face_explicit(
    mu: Scalar,
    lambda: Scalar,
    mag_sf: Scalar,
    n: Vec3,
    k: Vec3,
    gbar: Tensor,
    gprime: Tensor,
    sf: Vec3,
) -> Vec3 {
    matvec(m_of(mu, lambda, n), dot_left(k, gbar)) * mag_sf
        + dot_right(gprime, sf) * mu
        + sf * (lambda * gprime.trace())
}

/// (109.1)-(109.5), scatter-shaped: `a` is zeroed first; runs `s.check(m)` first.
pub fn assemble_host(a: &mut HostBlockLdu, m: &HostMesh, s: &BlockState<'_>) -> Result<()> {
    s.check(m)?;
    a.zero();

    // (109.1) the face coefficient, and (109.2) its four entries; then
    // (109.3), the explicit face vector, added to the owner's row and
    // subtracted from the neighbour's - each row with ITS OWN material.
    for f in 0..m.n_internal_faces {
        let o = m.owner[f] as usize;
        let nb = m.neighbour[f] as usize;
        let sf = m.sf[f];
        let mag_sf = m.mag_sf[f];
        let b = face_block(s.mu[o], s.lambda[o], sf, mag_sf, m.delta_coeffs[f]);
        a.upper[f] = Tensor::ZERO - b;
        a.lower[f] = Tensor::ZERO - b;
        a.diag[o] += b;
        a.diag[nb] += b;

        let w = m.weights[f];
        let gbar = s.grad[o] * w + s.grad[nb] * (1.0 - w);
        let n = normal_of(sf, mag_sf);
        let gprime = gbar - Vec3::outer(n, dot_left(n, gbar));
        let kf = m.non_orth_corr[f];
        a.source[o] += face_explicit(s.mu[o], s.lambda[o], mag_sf, n, kf, gbar, gprime, sf);
        a.source[nb] -= face_explicit(s.mu[nb], s.lambda[nb], mag_sf, n, kf, gbar, gprime, sf);
    }

    // (109.4) a boundary face: the FIXED columns implicit into `diag`, the
    // full `(sigma_b . Sf_b)_i` on a fixed row, `t_i |Sf_b|` and nothing else
    // on a free row.
    for bf in 0..m.n_boundary_faces {
        if reference::is_empty_face(m, bf) {
            continue;
        }
        let c = m.b_face_cells[bf] as usize;
        let sf_b = m.b_sf[bf];
        let mag_sf = m.b_mag_sf[bf];
        let n = normal_of(sf_b, mag_sf);
        let delta = m.b_delta_coeffs[bf];
        let kb = reference::boundary_corr_vector(m, bf);
        let p = mask_matrix(s.mask[bf]);
        let m_b = m_of(s.mu[c], s.lambda[c], n);
        let g = s.grad[c];
        let gprime = g - Vec3::outer(n, dot_left(n, g));
        let pv = |v: Vec3| matvec(p, v);
        let fixed_in = s.ref_value[bf] * delta + dot_left(kb, g);
        let free_in = (s.ub[bf] - s.u[c]) * delta;
        let normal = mag_sf * matvec(m_b, pv(fixed_in) + (free_in - pv(free_in)));
        let tangential =
            dot_right(gprime, sf_b) * s.mu[c] + sf_b * (s.lambda[c] * gprime.trace());
        a.diag[c] += matmul(p, matmul(m_b, p)) * (mag_sf * delta);
        a.source[c] += pv(normal + tangential) + (s.traction[bf] - pv(s.traction[bf])) * mag_sf;
    }

    // (109.5) the thermal load, `solidThermalLoad`'s masked face gather with
    // no bond faces: a traction face's thermal share is excluded (its
    // prescribed traction IS the face's whole contribution), a fixed
    // component's is kept.
    for f in 0..m.n_internal_faces {
        let o = m.owner[f] as usize;
        let nb = m.neighbour[f] as usize;
        let w = m.weights[f];
        let t_f = s.t[o] * w + s.t[nb] * (1.0 - w);
        a.source[o] -= m.sf[f] * (s.beta_alpha[o] * (t_f - s.t_ref[o]));
        a.source[nb] += m.sf[f] * (s.beta_alpha[nb] * (t_f - s.t_ref[nb]));
    }
    for bf in 0..m.n_boundary_faces {
        if reference::is_empty_face(m, bf) {
            continue;
        }
        let c = m.b_face_cells[bf] as usize;
        let p = mask_matrix(s.mask[bf]);
        let q = matvec(p, (s.bt[bf] - s.t_ref[c]) * m.b_sf[bf]);
        a.source[c] -= q * s.beta_alpha[c];
    }
    Ok(())
}

/// (109.2): `out = A u`, diagonal then a face loop that scatters into both cells.
pub fn amul_host(out: &mut Vec<Vec3>, u: &[Vec3], a: &HostBlockLdu, m: &HostMesh) {
    out.clear();
    out.resize(a.n_cells, Vec3::ZERO);
    for c in 0..a.n_cells {
        out[c] = matvec(a.diag[c], u[c]);
    }
    for f in 0..a.n_internal_faces {
        let o = m.owner[f] as usize;
        let nb = m.neighbour[f] as usize;
        out[o] += matvec(a.upper[f], u[nb]);
        out[nb] += matvec(a.lower[f], u[o]);
    }
}

/// (109.6): `out = A u - source`.
pub fn residual_host(out: &mut Vec<Vec3>, u: &[Vec3], a: &HostBlockLdu, m: &HostMesh) {
    amul_host(out, u, a, m);
    for c in 0..a.n_cells {
        out[c] -= a.source[c];
    }
}

/// (109.6)'s `scale`.
pub fn residual_scale(a: &HostBlockLdu, u: &[Vec3]) -> Scalar {
    let mut diag = 0.0 as Scalar;
    for t in &a.diag {
        diag = diag.max(frobenius(*t));
    }
    let mut u_mag = 0.0 as Scalar;
    for v in u {
        u_mag = u_mag.max(v.mag());
    }
    diag * u_mag
}

/// Row-major `3n x 3n`, unknown `3c + i` for component `i` of cell `c`; `reference::solve_dense` takes it.
pub fn dense_from_block(a: &HostBlockLdu, m: &HostMesh) -> Vec<Scalar> {
    let n = a.n_cells * 3;
    let mut d = vec![0.0 as Scalar; n * n];
    for c in 0..a.n_cells {
        for i in 0..3 {
            for j in 0..3 {
                d[(3 * c + i) * n + (3 * c + j)] = block_at(a.diag[c], i, j);
            }
        }
    }
    for f in 0..m.n_internal_faces {
        let o = m.owner[f] as usize;
        let nb = m.neighbour[f] as usize;
        for i in 0..3 {
            for j in 0..3 {
                d[(3 * o + i) * n + (3 * nb + j)] += block_at(a.upper[f], i, j);
                d[(3 * nb + i) * n + (3 * o + j)] += block_at(a.lower[f], i, j);
            }
        }
    }
    d
}

pub fn flatten3(v: &[Vec3]) -> Vec<Scalar> {
    v.iter().flat_map(|p| [p.x, p.y, p.z]).collect()
}

pub fn unflatten3(v: &[Scalar]) -> Vec<Vec3> {
    v.chunks_exact(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solid::bc::{fixed_minus_x, free_expansion};
    use crate::solid::prototype;
    use crate::solid::tests::{graded_block, linear_state};
    use crate::solid::Material;

    const DT: Scalar = 100.0;
    const T_REF: Scalar = 300.0;

    fn max_abs(v: &[Scalar]) -> Scalar {
        v.iter().fold(0.0 as Scalar, |a, &x| a.max(x.abs()))
    }

    /// `Material::steel(0.3)`'s per-cell arrays, with the tests' uniform `T_ref`.
    fn steel_arrays(n: usize) -> (Vec<Scalar>, Vec<Scalar>, Vec<Scalar>, Vec<Scalar>) {
        let mat = Material::steel(0.3);
        (
            vec![mat.mu(); n],
            vec![mat.lambda(); n],
            vec![mat.three_lambda_two_mu() * mat.alpha; n],
            vec![T_REF; n],
        )
    }

    /// What one test's assembly reads besides the state itself: the scattered
    /// boundary statement and steel's per-cell arrays.
    struct Arrays {
        mask: Vec<Label>,
        ref_value: Vec<Vec3>,
        traction: Vec<Vec3>,
        mu: Vec<Scalar>,
        lambda: Vec<Scalar>,
        beta_alpha: Vec<Scalar>,
        t_ref: Vec<Scalar>,
    }

    fn arrays(hm: &HostMesh, per_patch: &[[CompBc; 3]]) -> Arrays {
        let (mask, ref_value, traction) = scatter_bcs(hm, per_patch);
        let (mu, lambda, beta_alpha, t_ref) = steel_arrays(hm.n_cells);
        Arrays {
            mask,
            ref_value,
            traction,
            mu,
            lambda,
            beta_alpha,
            t_ref,
        }
    }

    fn state<'a>(
        u: &'a [Vec3],
        ub: &'a [Vec3],
        grad: &'a [Tensor],
        t: &'a [Scalar],
        bt: &'a [Scalar],
        ar: &'a Arrays,
    ) -> BlockState<'a> {
        BlockState {
            u,
            ub,
            grad,
            t,
            bt,
            mask: &ar.mask,
            ref_value: &ar.ref_value,
            traction: &ar.traction,
            mu: &ar.mu,
            lambda: &ar.lambda,
            beta_alpha: &ar.beta_alpha,
            t_ref: &ar.t_ref,
        }
    }

    /// The exact Green-Gauss gradient of a state with exact face values -
    /// exact on an orthogonal block (§3.5).
    fn exact_grad(hm: &HostMesh, u: &[Vec3], ub: &[Vec3]) -> Vec<Tensor> {
        let mut g = Vec::new();
        reference::fvc_grad_vector(&mut g, u, ub, hm);
        g
    }

    /// The deterministic non-linear state the symmetry and product gates
    /// read: the linear state multiplied by a fixed trigonometric field of
    /// the cell / boundary-face centre, and a temperature gradient on top.
    /// No random numbers anywhere in this crate's tests.
    fn nonlinear_state(
        hm: &HostMesh,
    ) -> (Vec<Vec3>, Vec<Vec3>, Vec<Tensor>, Vec<Scalar>, Vec<Scalar>) {
        let (mut u, mut ub) = linear_state(hm);
        let bump = |x: Vec3| 1.0 + 0.3 * (7.0 * x.x).sin() * (5.0 * x.y).cos();
        for (uc, xc) in u.iter_mut().zip(&hm.c) {
            *uc = *uc * bump(*xc);
        }
        for (ub_b, xb) in ub.iter_mut().zip(&hm.b_cf) {
            *ub_b = *ub_b * bump(*xb);
        }
        let grad = exact_grad(hm, &u, &ub);
        let t: Vec<Scalar> = hm.c.iter().map(|x| T_REF + DT * x.x).collect();
        let bt: Vec<Scalar> = hm.b_cf.iter().map(|x| T_REF + DT * x.x).collect();
        (u, ub, grad, t, bt)
    }

    /// The free-expansion state: `u = alpha dT x` at cells and boundary-face
    /// centres, `grad = alpha dT I`, `T = T_REF + DT` everywhere.
    fn free_expansion_state(
        hm: &HostMesh,
    ) -> (Vec<Vec3>, Vec<Vec3>, Vec<Tensor>, Vec<Scalar>, Vec<Scalar>) {
        let a = Material::steel(0.3).alpha * DT;
        let u: Vec<Vec3> = hm.c.iter().map(|x| *x * a).collect();
        let ub: Vec<Vec3> = hm.b_cf.iter().map(|x| *x * a).collect();
        let grad = vec![
            Tensor {
                xx: a,
                yy: a,
                zz: a,
                ..Tensor::ZERO
            };
            hm.n_cells
        ];
        let t = vec![T_REF + DT; hm.n_cells];
        let bt = vec![T_REF + DT; hm.n_boundary_faces];
        (u, ub, grad, t, bt)
    }

    #[test]
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
    fn a_linear_displacement_is_an_exact_solution_of_the_block_system_on_the_graded_block() {
        let hm = graded_block();
        let mut ar = arrays(&hm, &[[CompBc::Fixed(0.0); 3]; 6]);
        let (u, ub) = linear_state(&hm);
        // The host twin of Gate 95-C's `set_fixed_values(&ub)`: the prescribed
        // value IS the linear field's own boundary value.
        ar.ref_value = ub.clone();
        let grad = exact_grad(&hm, &u, &ub);
        let t = vec![T_REF; hm.n_cells];
        let bt = vec![T_REF; hm.n_boundary_faces];
        let s = state(&u, &ub, &grad, &t, &bt, &ar);
        let mut a = HostBlockLdu::zeros(&hm);
        assemble_host(&mut a, &hm, &s).expect("assemble");
        let mut r = Vec::new();
        residual_host(&mut r, &u, &a, &hm);
        let ratio = max_abs(&flatten3(&r)) / residual_scale(&a, &u);
        println!("  109-A graded block: max|r|/scale = {ratio:e}");
        assert!(ratio <= 1e-12, "the linear patch test residual is {ratio:e} of the scale");
    }

    #[test]
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
    fn the_free_expansion_state_is_an_exact_solution_of_the_block_system() {
        let hm = prototype::block(10).expect("block");
        let ar = arrays(&hm, &free_expansion());
        let (u, ub, grad, t, bt) = free_expansion_state(&hm);
        let s = state(&u, &ub, &grad, &t, &bt, &ar);
        let mut a = HostBlockLdu::zeros(&hm);
        assemble_host(&mut a, &hm, &s).expect("assemble");
        let mut r = Vec::new();
        residual_host(&mut r, &u, &a, &hm);
        let ratio = max_abs(&flatten3(&r)) / residual_scale(&a, &u);
        println!("  109-A block(10): max|r|/scale = {ratio:e}");
        assert!(ratio <= 1e-12, "the free-expansion residual is {ratio:e} of the scale");
    }

    #[test]
    fn the_block_matrix_is_symmetric_to_the_bit() {
        for (name, hm) in [
            ("block(4)", prototype::block(4).expect("block")),
            (
                "jittered_block(4, 0.25)",
                prototype::jittered_block(4, 0.25).expect("jittered"),
            ),
        ] {
            let ar = arrays(&hm, &fixed_minus_x());
            let (u, ub, grad, t, bt) = nonlinear_state(&hm);
            let s = state(&u, &ub, &grad, &t, &bt, &ar);
            let mut a = HostBlockLdu::zeros(&hm);
            assemble_host(&mut a, &hm, &s).expect("assemble");
            let dense = dense_from_block(&a, &hm);
            let n = 3 * hm.n_cells;
            for i in 0..n {
                for j in (i + 1)..n {
                    assert_eq!(
                        dense[i * n + j],
                        dense[j * n + i],
                        "dense[{i}][{j}] != dense[{j}][{i}] on {name}"
                    );
                }
            }
            for (c, d) in a.diag.iter().enumerate() {
                assert!(
                    d.xx > 0.0 && d.yy > 0.0 && d.zz > 0.0,
                    "cell {c} lost a positive diagonal entry on {name}"
                );
            }
            println!("  §109.3 {name}: N = {n}, symmetric to the bit, diagonals positive");
        }
    }

    #[test]
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
    fn the_host_product_and_the_dense_expansion_agree() {
        let hm = prototype::jittered_block(4, 0.25).expect("jittered");
        let ar = arrays(&hm, &fixed_minus_x());
        let (u, ub, grad, t, bt) = nonlinear_state(&hm);
        let s = state(&u, &ub, &grad, &t, &bt, &ar);
        let mut a = HostBlockLdu::zeros(&hm);
        assemble_host(&mut a, &hm, &s).expect("assemble");
        let mut au = Vec::new();
        amul_host(&mut au, &u, &a, &hm);
        let dense = dense_from_block(&a, &hm);
        let x = flatten3(&u);
        let y = flatten3(&au);
        let n = 3 * hm.n_cells;
        let mut worst = 0.0 as Scalar;
        for i in 0..n {
            let mut row = 0.0 as Scalar;
            for (j, &xj) in x.iter().enumerate() {
                row += dense[i * n + j] * xj;
            }
            worst = worst.max((y[i] - row).abs());
        }
        // The (109.6) scale and NOT `rel_max`: the product cancels heavily and
        // `max|A u|` is the wrong yardstick.
        let ratio = worst / residual_scale(&a, &u);
        println!("  §109.3 max|A u - dense x|/scale = {ratio:e}");
        assert!(
            ratio <= 1e-14,
            "the product and the dense expansion differ by {ratio:e} of the scale"
        );
    }

    #[test]
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
    fn the_dense_block_system_solves_the_free_expansion_state_directly() {
        let hm = prototype::block(4).expect("block");
        let ar = arrays(&hm, &free_expansion());
        let (u, ub, grad, t, bt) = free_expansion_state(&hm);
        let s = state(&u, &ub, &grad, &t, &bt, &ar);
        let mut a = HostBlockLdu::zeros(&hm);
        assemble_host(&mut a, &hm, &s).expect("assemble");
        let dense = dense_from_block(&a, &hm);
        let rhs = flatten3(&a.source);
        let x = reference::solve_dense(dense, &rhs)
            .expect("the free-expansion block system is nonsingular");
        let want = flatten3(&u);
        let err: Vec<Scalar> = x.iter().zip(&want).map(|(&g, &w)| g - w).collect();
        let rel = max_abs(&err) / max_abs(&want);
        println!("  §109.3 dense solve, relative error = {rel:e}");
        assert!(rel <= 1e-10, "the dense solve misses the state by {rel:e}");
    }

    #[test]
    fn the_block_state_refuses_a_wrong_length_by_name() {
        let hm = prototype::block(4).expect("block");
        let (n, nbf) = (hm.n_cells, hm.n_boundary_faces);
        let ar = arrays(&hm, &fixed_minus_x());
        let (u, ub, grad, t, bt) = nonlinear_state(&hm);
        assert!(state(&u, &ub, &grad, &t, &bt, &ar).check(&hm).is_ok());

        let s = BlockState {
            u: &u[..n - 1],
            ..state(&u, &ub, &grad, &t, &bt, &ar)
        };
        let msg = s.check(&hm).expect_err("a short u").to_string();
        assert!(msg.contains("solid block"), "{msg}");
        assert!(msg.contains("`u`"), "{msg}");
        assert!(msg.contains(&(n - 1).to_string()) && msg.contains(&n.to_string()), "{msg}");

        let s = BlockState {
            ub: &ub[..nbf - 1],
            ..state(&u, &ub, &grad, &t, &bt, &ar)
        };
        let msg = s.check(&hm).expect_err("a short ub").to_string();
        assert!(msg.contains("`ub`"), "{msg}");
        assert!(
            msg.contains(&(nbf - 1).to_string()) && msg.contains(&nbf.to_string()),
            "{msg}"
        );

        let s = BlockState {
            t_ref: &ar.t_ref[..n - 1],
            ..state(&u, &ub, &grad, &t, &bt, &ar)
        };
        let msg = s.check(&hm).expect_err("a short t_ref").to_string();
        assert!(msg.contains("`t_ref`"), "{msg}");

        let s = BlockState {
            mu: &ar.mu[..n / 2],
            ..state(&u, &ub, &grad, &t, &bt, &ar)
        };
        let msg = s.check(&hm).expect_err("a short mu").to_string();
        assert!(msg.contains("`mu`"), "{msg}");
        assert!(msg.contains(&(n / 2).to_string()), "{msg}");
    }
}
