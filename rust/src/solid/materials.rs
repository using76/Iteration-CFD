// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Two materials bonded inside one solid region - the per-cell material map,
//! the series `(2 mu + lambda)` face coefficient, the traction-continuous
//! bond face and the host mirrors of the bond kernels. SPEC-LIT §95.8.
//!
//! Written from:
//!   Ž. Tuković, A. Ivanković, A. Karač, *Int. J. Numer. Methods Eng.* 93
//!     (2013) 400-419, DOI 10.1002/nme.4390 - the interface displacement from
//!     traction continuity with one-sided normal derivatives (S95.10)-(S95.11)
//!   S. V. Patankar, *Numerical Heat Transfer and Fluid Flow*, Hemisphere
//!     (1980) §4.2.3 - the series face coefficient, through SPEC-LIT §46.2
//!   I. Demirdžić, S. Muzaferija, *Int. J. Numer. Methods Eng.* 37 (1994)
//!     3751-3766, DOI 10.1002/nme.1620372110 - a face's contribution IS its traction
//!   ofgpu `SPEC-LIT.md` §1, §2.3, §2.4, §3.5, §46.2, §95.8
//!
//! OpenFOAM, solids4foam and foam-extend are GPL and were not opened; no
//! solid-mechanics solver of any licence was consulted. No GPL-licensed source was consulted.

use super::Material;
use crate::mesh::HostMesh;
use crate::{Error, Label, Result, Scalar, Tensor, Vec3};

/// What a bond face says. `Series` is the traction-continuous bond face of
/// SPEC-LIT §95.8; `Linear` treats the bond like any other face with the
/// constants interpolated linearly, which is the case-reachable foil the
/// second leg of Gate 95-E measures against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BondTreatment {
    #[default]
    Series,
    Linear,
}

/// One solid region carrying several bonded materials. The map is host-only:
/// the device sees [`PerCell`] arrays, never this struct.
#[derive(Debug, Clone)]
pub struct MaterialMap {
    /// One name per material, in the order the entries were given.
    pub names: Vec<String>,
    /// One validated [`Material`] per name.
    pub materials: Vec<Material>,
    /// Per material; `None` inherits the region's `T_ref`
    /// ([`super::displacement::Displacement::set_temperature`]'s argument).
    pub t_ref: Vec<Option<Scalar>>,
    /// `[n_cells]` material index of every cell.
    pub cell: Vec<Label>,
    pub bond: BondTreatment,
}

/// The per-cell material arrays the device kernels read. All `[n_cells]`.
#[derive(Debug, Clone)]
pub struct PerCell {
    pub mu: Vec<Scalar>,
    pub lambda: Vec<Scalar>,
    pub alpha: Vec<Scalar>,
    /// `(3 lambda + 2 mu) alpha`, the coefficient the thermal load carries,
    /// computed with the same expression the one-material operator passes
    /// its kernel, so that a one-material map hands the kernels the same bits.
    pub beta_alpha: Vec<Scalar>,
}

/// Which internal faces are bond faces, and the bond index of each.
#[derive(Debug, Clone)]
pub struct Bonds {
    /// `[n_internal_faces]`: the bond index, or `-1` on a same-material face.
    pub face_bond: Vec<Label>,
    /// `[n_bond]`: the internal face index of each bond face, ascending.
    pub bond_face: Vec<Label>,
}

impl Bonds {
    pub fn n_bond(&self) -> usize {
        self.bond_face.len()
    }
}

/// One cell's material, as the mirrors read it. `gamma()` is the implicit
/// `2 mu + lambda`.
#[derive(Clone, Copy, Debug)]
pub struct CellMaterial {
    pub mu: Scalar,
    pub lambda: Scalar,
    pub beta_alpha: Scalar,
    pub t_ref: Scalar,
}

impl CellMaterial {
    pub fn of(pc: &PerCell, t_ref: &[Scalar], c: usize) -> Self {
        Self {
            mu: pc.mu[c],
            lambda: pc.lambda[c],
            beta_alpha: pc.beta_alpha[c],
            t_ref: t_ref[c],
        }
    }

    /// `2 mu + lambda`, computed the way the kernels compute it from the two
    /// per-cell arrays.
    pub fn gamma(self) -> Scalar {
        2.0 * self.mu + self.lambda
    }
}

/// What one bond face says: the face displacement, the face traction, the
/// interpolated tangential gradient both sides share, and the geometry the
/// one-sided derivatives are divided by.
#[derive(Clone, Copy, Debug)]
pub struct BondFace {
    pub u_f: Vec3,
    pub t_f: Vec3,
    /// `G_t`, the face gradient with its normal row struck out.
    pub g_t: Tensor,
    pub n: Vec3,
    pub d_p: Scalar,
    pub d_n: Scalar,
    pub theta_p: Scalar,
    pub theta_n: Scalar,
}

/// A cell listed under no material, before the map is built.
const UNASSIGNED: Label = -1;

impl MaterialMap {
    /// One material everywhere: the map every existing constructor builds,
    /// kept bitwise on the one-material path.
    pub fn uniform(name: &str, m: Material, n_cells: usize) -> Self {
        Self {
            names: vec![name.to_string()],
            materials: vec![m],
            t_ref: vec![None],
            cell: vec![0; n_cells],
            bond: BondTreatment::Series,
        }
    }

    /// The map from per-material cell lists. Refuses, by name: a cell listed
    /// under two materials, a cell listed under none, an index past the end
    /// of the region, a material name used twice, and a material that does
    /// not validate (that message passes through).
    pub fn from_cell_lists(
        entries: &[(&str, Material, Option<Scalar>, Vec<Label>)],
        n_cells: usize,
        bond: BondTreatment,
    ) -> Result<Self> {
        let mut names: Vec<String> = Vec::new();
        let mut materials: Vec<Material> = Vec::new();
        let mut t_ref: Vec<Option<Scalar>> = Vec::new();
        let mut cell = vec![UNASSIGNED; n_cells];
        for (name, mat, tr, cells) in entries {
            mat.validate()?;
            if names.iter().any(|n| n == name) {
                return Err(Error::Config(format!(
                    "solid materials: the material name '{name}' is used by two \
                     entries - one region cannot carry two materials of one name"
                )));
            }
            let idx = names.len() as Label;
            names.push(name.to_string());
            materials.push(*mat);
            t_ref.push(*tr);
            for &c in cells {
                if c as usize >= n_cells {
                    return Err(Error::Config(format!(
                        "solid materials: material '{name}' lists cell {c}, but \
                         the region has {n_cells} cells"
                    )));
                }
                if cell[c as usize] != UNASSIGNED {
                    let other = &names[cell[c as usize] as usize];
                    return Err(Error::Config(format!(
                        "solid materials: cell {c} is listed under both '{other}' \
                         and '{name}' - a cell carries exactly one material"
                    )));
                }
                cell[c as usize] = idx;
            }
        }
        let unlisted: Vec<Label> = (0..n_cells as Label)
            .filter(|&c| cell[c as usize] == UNASSIGNED)
            .collect();
        if let Some(c) = unlisted.first() {
            return Err(Error::Config(format!(
                "solid materials: cell {c} is listed under no material (and \
                 {} more like it) - every cell of the region carries a material",
                unlisted.len() - 1
            )));
        }
        Ok(Self { names, materials, t_ref, cell, bond })
    }

    pub fn n_materials(&self) -> usize {
        self.names.len()
    }

    /// The cells of one material, ascending.
    pub fn cells_of(&self, material: usize) -> Vec<Label> {
        (0..self.cell.len() as Label)
            .filter(|&c| self.cell[c as usize] as usize == material)
            .collect()
    }

    /// The per-cell arrays the device reads. `beta_alpha` is
    /// `(3 lambda + 2 mu) alpha`, written exactly as the one-material
    /// operator writes its `thermalCoeff`, so a one-material map hands the
    /// kernels the same bits.
    pub fn per_cell(&self) -> PerCell {
        let n = self.cell.len();
        let mut pc = PerCell {
            mu: vec![0.0 as Scalar; n],
            lambda: vec![0.0 as Scalar; n],
            alpha: vec![0.0 as Scalar; n],
            beta_alpha: vec![0.0 as Scalar; n],
        };
        for (c, &m) in self.cell.iter().enumerate() {
            let mat = self.materials[m as usize];
            pc.mu[c] = mat.mu();
            pc.lambda[c] = mat.lambda();
            pc.alpha[c] = mat.alpha;
            pc.beta_alpha[c] = mat.three_lambda_two_mu() * mat.alpha;
        }
        pc
    }

    /// The per-cell `T_ref`: a material's own value where the map carries
    /// one, the region's scalar where it does not.
    pub fn t_ref_per_cell(&self, region_t_ref: Scalar) -> Vec<Scalar> {
        self.cell
            .iter()
            .map(|&c| self.t_ref[c as usize].unwrap_or(region_t_ref))
            .collect()
    }

    /// Which internal faces are bond faces: a face whose owner and
    /// neighbour carry different materials. A bond face whose one-sided
    /// normal distances are not both positive is refused - that is a
    /// degenerate mesh, not a material map.
    pub fn bonds(&self, m: &HostMesh) -> Result<Bonds> {
        let mut face_bond = vec![-1 as Label; m.n_internal_faces];
        let mut bond_face = Vec::new();
        for f in 0..m.n_internal_faces {
            let (o, n) = (m.owner[f] as usize, m.neighbour[f] as usize);
            if self.cell[o] == self.cell[n] {
                continue;
            }
            let d_p = m.sf[f].dot(m.cf[f] - m.c[o]).abs() / m.mag_sf[f];
            let d_n = m.sf[f].dot(m.c[n] - m.cf[f]).abs() / m.mag_sf[f];
            if !(d_p > 0.0) || !(d_n > 0.0) {
                return Err(Error::Mesh(format!(
                    "solid materials: internal face {f} between materials '{}' \
                     and '{}' has d_P = {d_p}, d_N = {d_n}: a cell centre lies \
                     on its own face: the mesh is degenerate here, not the \
                     material map",
                    self.names[self.cell[o] as usize],
                    self.names[self.cell[n] as usize]
                )));
            }
            face_bond[f] = bond_face.len() as Label;
            bond_face.push(f as Label);
        }
        Ok(Bonds { face_bond, bond_face })
    }

    /// `(gamma_mag_sf, b_gamma_mag_sf)`, the implicit coefficient of the
    /// split per face. A same-material face is the one-material operator's
    /// own number, coefficient times magnitude, to the bit; a bond face is
    /// the series value of SPEC-LIT §46.2's shape with `2 mu + lambda` in
    /// the place of `k`, the resistance the two cells present in series and
    /// not an average of the two conductivities; a boundary face belongs to
    /// its one cell.
    pub fn implicit_coefficients(&self, m: &HostMesh) -> (Vec<Scalar>, Vec<Scalar>) {
        let gammas: Vec<Scalar> =
            self.materials.iter().map(|mat| mat.implicit_gamma()).collect();
        let mut gamma_mag_sf = vec![0.0 as Scalar; m.n_internal_faces];
        for f in 0..m.n_internal_faces {
            let (o, n) = (m.owner[f] as usize, m.neighbour[f] as usize);
            let (cp, cn) = (self.cell[o] as usize, self.cell[n] as usize);
            let mag = m.mag_sf[f];
            gamma_mag_sf[f] = if cp == cn {
                gammas[cp] * mag
            } else {
                let w = m.weights[f];
                match self.bond {
                    BondTreatment::Series => {
                        mag / ((1.0 - w) / gammas[cp] + w / gammas[cn])
                    }
                    BondTreatment::Linear => {
                        mag * (w * gammas[cp] + (1.0 - w) * gammas[cn])
                    }
                }
            };
        }
        let mut b_gamma_mag_sf = vec![0.0 as Scalar; m.n_boundary_faces];
        for bf in 0..m.n_boundary_faces {
            let c = self.cell[m.b_face_cells[bf] as usize] as usize;
            b_gamma_mag_sf[bf] = gammas[c] * m.b_mag_sf[bf];
        }
        (gamma_mag_sf, b_gamma_mag_sf)
    }

    /// The one-line-per-material summary a coupled case's banner carries:
    /// each material's constants and cell count, then the bond face count
    /// and the treatment by name.
    pub fn describe(&self, n_bond: usize) -> String {
        let mut s = String::new();
        for (i, mat) in self.materials.iter().enumerate() {
            let t = self.t_ref[i].map(|v| v.to_string()).unwrap_or_else(|| "region".to_string());
            s.push_str(&format!(
                "{}  {}  {}  {}  {}  {}\n",
                self.names[i],
                mat.e,
                mat.nu,
                mat.alpha,
                t,
                self.cells_of(i).len()
            ));
        }
        let mode = match self.bond {
            BondTreatment::Series => "series",
            BondTreatment::Linear => "linear",
        };
        s.push_str(&format!("bond faces: {n_bond} ({mode})"));
        s
    }
}

// ==========================================================================
//  The bond face, and the host mirrors of the two bond kernels
// ==========================================================================

/// `(a^T G)_j = sum_i a_i G_ij`, the normal derivative - the host copy of
/// the kernels' contraction, written in SPEC-LIT §1's index convention.
#[inline]
fn dot_left(a: Vec3, g: Tensor) -> Vec3 {
    Vec3::new(
        a.x * g.xx + a.y * g.yx + a.z * g.zx,
        a.x * g.xy + a.y * g.yy + a.z * g.zy,
        a.x * g.xz + a.y * g.yz + a.z * g.zz,
    )
}

/// `(G a)_j = sum_i G_ji a_i`, the transposed contraction.
#[inline]
fn dot_right(g: Tensor, a: Vec3) -> Vec3 {
    Vec3::new(
        g.xx * a.x + g.xy * a.y + g.xz * a.z,
        g.yx * a.x + g.yy * a.y + g.yz * a.z,
        g.zx * a.x + g.zy * a.y + g.zz * a.z,
    )
}

/// The tangential part of a vector, the normal component struck out.
#[inline]
fn tangential(v: Vec3, n: Vec3) -> Vec3 {
    v - n * v.dot(n)
}

/// The mu/lambda half of the deferred surface integral, the thermal part
/// living in the thermal load:
///
/// ```text
/// [ mu grad(u)^T + lambda tr(grad u) I - (mu + lambda) grad(u) ] . Sf
/// ```
#[inline]
fn deferred_mu_lam(mu: Scalar, lam: Scalar, g: Tensor, s: Vec3) -> Vec3 {
    dot_right(g, s) * mu + s * (lam * g.trace()) - dot_left(s, g) * (mu + lam)
}

/// The traction one side of a bond face carries, from that side's own
/// one-sided normal derivative `g`: `sigma.n` of the side's state
/// `grad(u) = G_t + n (x) g` - the wall form the traction boundary solves,
/// with the other cell standing where the wall stood. SPEC-LIT §95.8.
#[inline]
pub fn side_traction(mat: CellMaterial, g_t: Tensor, g: Vec3, theta: Scalar, n: Vec3) -> Vec3 {
    let r = dot_right(g_t, n);
    mat.mu * g
        + (mat.mu + mat.lambda) * g.dot(n) * n
        + mat.mu * r
        + mat.lambda * g_t.trace() * n
        - theta * n
}

/// What one bond face says, both treatments. `Series` solves the face for
/// the displacement that makes the traction continuous across it with
/// one-sided normal derivatives (Tuković, Ivanković & Karač 2013) and takes
/// the traction from the owner side; `Linear` interpolates the displacement
/// and every coefficient like any other face. `w` is the mesh weight; the
/// two normal distances come from the face and cell centres.
pub fn bond_face_values(
    mode: BondTreatment,
    w: Scalar,
    sf: Vec3,
    cf: Vec3,
    c_p: Vec3,
    c_n: Vec3,
    u_p: Vec3,
    u_n: Vec3,
    g_p: Tensor,
    g_n: Tensor,
    t_p: Scalar,
    t_n: Scalar,
    mat_p: CellMaterial,
    mat_n: CellMaterial,
) -> BondFace {
    let mag = sf.mag();
    let n = sf.normalised();
    let d_p = sf.dot(cf - c_p).abs() / mag;
    let d_n = sf.dot(c_n - cf).abs() / mag;
    let g_f = g_p * w + g_n * (1.0 - w);
    let g_t = g_f - n.outer(dot_left(n, g_f));
    let r = dot_right(g_t, n);
    let t_face = w * t_p + (1.0 - w) * t_n;
    let theta_p = mat_p.beta_alpha * (t_face - mat_p.t_ref);
    let theta_n = mat_n.beta_alpha * (t_face - mat_n.t_ref);
    let u_f = match mode {
        BondTreatment::Series => {
            let a_p = mat_p.gamma() / d_p;
            let a_n = mat_n.gamma() / d_n;
            let b_p = mat_p.mu / d_p;
            let b_n = mat_n.mu / d_n;
            let u_f_n = (a_p * u_p.dot(n)
                + a_n * u_n.dot(n)
                + (mat_n.lambda - mat_p.lambda) * g_t.trace()
                - (theta_n - theta_p))
                / (a_p + a_n);
            let u_f_t = (tangential(u_p, n) * b_p
                + tangential(u_n, n) * b_n
                + r * (mat_n.mu - mat_p.mu))
                / (b_p + b_n);
            n * u_f_n + u_f_t
        }
        BondTreatment::Linear => w * u_p + (1.0 - w) * u_n,
    };
    let t_f = match mode {
        BondTreatment::Series => {
            let g_p_side = (u_f - u_p) / d_p;
            side_traction(mat_p, g_t, g_p_side, theta_p, n)
        }
        BondTreatment::Linear => {
            let mu_f = w * mat_p.mu + (1.0 - w) * mat_n.mu;
            let lam_f = w * mat_p.lambda + (1.0 - w) * mat_n.lambda;
            let theta_f = w * theta_p + (1.0 - w) * theta_n;
            mu_f * (dot_left(n, g_f) + dot_right(g_f, n))
                + n * (lam_f * g_f.trace())
                - n * theta_f
        }
    };
    BondFace { u_f, t_f, g_t, n, d_p, d_n, theta_p, theta_n }
}

/// The Green-Gauss correction at the two bond cells once the face
/// displacement is known: the face's share of the cell's gradient sum is
/// rewritten from the interpolated value the plain sweep used to what the
/// bond face says, scattered into both cells. SPEC-LIT §95.8.
pub fn bond_grad_correction(
    grad: &mut [Tensor],
    bonds: &Bonds,
    u: &[Vec3],
    bond_u: &[Vec3],
    m: &HostMesh,
) {
    for (b, &f) in bonds.bond_face.iter().enumerate() {
        let fu = f as usize;
        let p = m.owner[fu] as usize;
        let n = m.neighbour[fu] as usize;
        let w = m.weights[fu];
        let du = bond_u[b] - (u[p] * w + u[n] * (1.0 - w));
        let t = m.sf[fu].outer(du);
        grad[p] += t * (1.0 / m.v[p]);
        grad[n] = grad[n] - t * (1.0 / m.v[n]);
    }
}

/// The right-hand side the deferred gather and the thermal load build,
/// host-side, with the bond branches the device's two kernels carry: a
/// bond face contributes the explicit term `(S95.12)` - its own traction
/// times the face area, minus the matrix's implicit share and its
/// non-orthogonal correction - and nothing to the thermal sum; a plain
/// face contributes the deferred group with each side's own material and
/// each side's own thermal thrust; a non-empty boundary face contributes,
/// on its fixed components only, the deferred group and the thermal thrust
/// of its one cell. Scatter-shaped: loops faces, writes both cells.
pub fn rhs_mirror(
    pc: &PerCell,
    t_ref: &[Scalar],
    bonds: &Bonds,
    bond_t: &[Vec3],
    gamma_mag_sf: &[Scalar],
    u: &[Vec3],
    grad: &[Tensor],
    b_grad: &[Tensor],
    t: &[Scalar],
    bt: &[Scalar],
    fixed: &[[bool; 3]],
    m: &HostMesh,
) -> Vec<Vec3> {
    let mut rhs = vec![Vec3::ZERO; m.n_cells];
    for f in 0..m.n_internal_faces {
        let p = m.owner[f] as usize;
        let n = m.neighbour[f] as usize;
        let w = m.weights[f];
        let gf = grad[p] * w + grad[n] * (1.0 - w);
        let fb = bonds.face_bond[f];
        if fb >= 0 {
            let q = bond_t[fb as usize] * m.mag_sf[f]
                - gamma_mag_sf[f]
                    * (m.delta_coeffs[f] * (u[n] - u[p])
                        + dot_left(m.non_orth_corr[f], gf));
            rhs[p] += q;
            rhs[n] -= q;
        } else {
            rhs[p] += deferred_mu_lam(pc.mu[p], pc.lambda[p], gf, m.sf[f]);
            rhs[n] -= deferred_mu_lam(pc.mu[n], pc.lambda[n], gf, m.sf[f]);
            let tf = w * t[p] + (1.0 - w) * t[n];
            rhs[p] -= m.sf[f] * (pc.beta_alpha[p] * (tf - t_ref[p]));
            rhs[n] += m.sf[f] * (pc.beta_alpha[n] * (tf - t_ref[n]));
        }
    }
    for bf in 0..m.n_boundary_faces {
        if crate::reference::is_empty_face(m, bf) {
            continue;
        }
        let c = m.b_face_cells[bf] as usize;
        let q = deferred_mu_lam(pc.mu[c], pc.lambda[c], b_grad[bf], m.b_sf[bf])
            - m.b_sf[bf] * (pc.beta_alpha[c] * (bt[bf] - t_ref[c]));
        let mut rc = rhs[c];
        for i in 0..3 {
            if fixed[bf][i] {
                let v = rc.component(i) + q.component(i);
                rc = match i {
                    0 => Vec3::new(v, rc.y, rc.z),
                    1 => Vec3::new(rc.x, v, rc.z),
                    _ => Vec3::new(rc.x, rc.y, v),
                };
            }
        }
        rhs[c] = rc;
    }
    rhs
}

/// The solved normal gradient the traction boundary condition produces, per
/// boundary face, from the CELL gradient - the device's
/// `solidTractionRefGrad` with a per-cell material, written into the FREE
/// components only (a fixed component's value is left at zero here; the
/// device leaves its buffer's entry untouched). Empty faces stay zero.
pub fn traction_ref_grad_mirror(
    pc: &PerCell,
    t_ref: &[Scalar],
    grad: &[Tensor],
    traction: &[Vec3],
    bt: &[Scalar],
    fixed: &[[bool; 3]],
    m: &HostMesh,
) -> Vec<Vec3> {
    let mut out = vec![Vec3::ZERO; m.n_boundary_faces];
    for bf in 0..m.n_boundary_faces {
        if crate::reference::is_empty_face(m, bf) {
            continue;
        }
        let c = m.b_face_cells[bf] as usize;
        let g = grad[c];
        let n = m.b_sf[bf].normalised();
        let g_t = g - n.outer(dot_left(n, g));
        let dr = dot_right(g_t, n);
        let tr = traction[bf];
        let tn = tr.dot(n);
        let drn = dr.dot(n);
        let (mu, lam) = (pc.mu[c], pc.lambda[c]);
        let gamma = 2.0 * mu + lam;
        let thermal = pc.beta_alpha[c] * (bt[bf] - t_ref[c]);
        let normal = (tn - mu * drn - lam * g_t.trace() + thermal) / gamma;
        let tangential = (tr - n * tn) * (1.0 / mu) - (dr - n * drn);
        let r = n * normal + tangential;
        let mut row = Vec3::ZERO;
        for i in 0..3 {
            if !fixed[bf][i] {
                let v = r.component(i);
                row = match i {
                    0 => Vec3::new(v, row.y, row.z),
                    1 => Vec3::new(row.x, v, row.z),
                    _ => Vec3::new(row.x, row.y, v),
                };
            }
        }
        out[bf] = row;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockgen::{BlockSpec, GradedAxis};
    use crate::cht::{Conductivity, Conduction, PairingTolerances, RegionInput, RegionKind, ThermalMesh};
    use crate::solid::prototype;

    /// The deterministic draw the bond-face tests share: no crate, no
    /// thread-local, the same twenty faces on every machine.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn r(&mut self) -> Scalar {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / 9007199254740992.0
        }

        fn range(&mut self, lo: Scalar, hi: Scalar) -> Scalar {
            lo + (hi - lo) * self.r()
        }
    }

    /// One random bond-face geometry and field set, `C_P` at the origin and
    /// the face plane cutting the owner-centre line at `d_P`.
    #[derive(Clone, Copy)]
    struct FaceSet {
        n: Vec3,
        w: Scalar,
        sf: Vec3,
        cf: Vec3,
        c_p: Vec3,
        c_n: Vec3,
        u_p: Vec3,
        u_n: Vec3,
        g_p: Tensor,
        g_n: Tensor,
        t_p: Scalar,
        t_n: Scalar,
    }

    fn draw(rng: &mut Lcg) -> FaceSet {
        let mut n = Vec3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0));
        if n.mag() < 1e-3 {
            n = Vec3::new(1.0, 0.0, 0.0);
        }
        let n = n.normalised();
        let d_p = rng.range(0.5, 1.5);
        let d_n = rng.range(0.5, 1.5);
        let mag = rng.range(0.5, 2.0);
        let g = |rng: &mut Lcg| Tensor {
            xx: rng.range(-1.0, 1.0) * 1e-3,
            xy: rng.range(-1.0, 1.0) * 1e-3,
            xz: rng.range(-1.0, 1.0) * 1e-3,
            yx: rng.range(-1.0, 1.0) * 1e-3,
            yy: rng.range(-1.0, 1.0) * 1e-3,
            yz: rng.range(-1.0, 1.0) * 1e-3,
            zx: rng.range(-1.0, 1.0) * 1e-3,
            zy: rng.range(-1.0, 1.0) * 1e-3,
            zz: rng.range(-1.0, 1.0) * 1e-3,
        };
        FaceSet {
            w: d_n / (d_p + d_n),
            sf: n * mag,
            cf: n * d_p,
            c_p: Vec3::ZERO,
            c_n: n * (d_p + d_n),
            u_p: Vec3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)) * 1e-3,
            u_n: Vec3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)) * 1e-3,
            g_p: g(rng),
            g_n: g(rng),
            t_p: rng.range(290.0, 310.0),
            t_n: rng.range(290.0, 310.0),
            n,
        }
    }

    /// A random validated material and the cell form the mirrors read.
    fn mat_at(rng: &mut Lcg) -> (Material, CellMaterial) {
        let m = Material {
            e: rng.range(50.0e9, 300.0e9),
            nu: rng.range(0.2, 0.4),
            alpha: rng.range(1e-5, 3e-5),
        };
        let c = CellMaterial {
            mu: m.mu(),
            lambda: m.lambda(),
            beta_alpha: m.three_lambda_two_mu() * m.alpha,
            t_ref: rng.range(290.0, 300.0),
        };
        (m, c)
    }

    fn split_map(hm: &HostMesh, a: Material, b: Material, bond: BondTreatment) -> MaterialMap {
        let low: Vec<Label> = (0..hm.n_cells as Label)
            .filter(|&c| hm.c[c as usize].x < 0.5)
            .collect();
        let high: Vec<Label> = (0..hm.n_cells as Label)
            .filter(|&c| hm.c[c as usize].x >= 0.5)
            .collect();
        MaterialMap::from_cell_lists(
            &[("steel", a, None, low), ("brass", b, None, high)],
            hm.n_cells,
            bond,
        )
        .expect("map")
    }

    fn one_region(m: &HostMesh) -> ThermalMesh {
        ThermalMesh::build(
            &[RegionInput { name: "s".into(), kind: RegionKind::Solid, mesh: m }],
            &[],
            PairingTolerances::default(),
        )
        .expect("thermal mesh")
    }

    /// The five refusals, each by name.
    #[test]
    fn a_cell_in_two_lists_or_in_none_is_refused_by_name() {
        let a = Material::steel(0.3);
        let b = Material { e: 100e9, nu: 0.3, alpha: 2.0e-5 };
        let err = MaterialMap::from_cell_lists(
            &[("steel", a, None, vec![0, 1]), ("brass", b, None, vec![1, 2])],
            8,
            BondTreatment::Series,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("steel") && err.contains("brass") && err.contains("1"), "{err}");

        let err = MaterialMap::from_cell_lists(&[("steel", a, None, vec![0])], 8, BondTreatment::Series)
            .unwrap_err()
            .to_string();
        assert!(err.contains("cell 1") && err.contains("no material"), "{err}");

        let err = MaterialMap::from_cell_lists(&[("steel", a, None, vec![8])], 8, BondTreatment::Series)
            .unwrap_err()
            .to_string();
        assert!(err.contains("8") && err.contains("8 cells"), "{err}");

        let err = MaterialMap::from_cell_lists(
            &[("steel", a, None, vec![0]), ("steel", b, None, vec![1])],
            8,
            BondTreatment::Series,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("steel") && err.contains("two entries"), "{err}");

        let bad = Material { e: 200e9, nu: 0.45 + 0.05, alpha: 1.2e-5 };
        let err = MaterialMap::from_cell_lists(&[("steel", bad, None, vec![0])], 8, BondTreatment::Series)
            .unwrap_err()
            .to_string();
        assert!(err.contains("nu") && err.contains("0.5"), "{err}");
    }

    /// A one-material map is the uniform coefficient the one-material
    /// operator builds, coefficient times magnitude, on every face, to the
    /// bit - and makes no bond face.
    #[test]
    fn a_one_material_map_is_the_uniform_coefficient_to_the_bit() {
        let hm = prototype::jittered_block(8, 0.25).expect("block");
        let mat = Material::steel(0.3);
        let map = MaterialMap::uniform("steel", mat, hm.n_cells);
        let (gamma_mag_sf, b_gamma_mag_sf) = map.implicit_coefficients(&hm);
        let gamma = mat.implicit_gamma();
        for f in 0..hm.n_internal_faces {
            assert_eq!(
                gamma_mag_sf[f].to_bits(),
                (gamma * hm.mag_sf[f]).to_bits(),
                "internal face {f}"
            );
        }
        for bf in 0..hm.n_boundary_faces {
            assert_eq!(
                b_gamma_mag_sf[bf].to_bits(),
                (gamma * hm.b_mag_sf[bf]).to_bits(),
                "boundary face {bf}"
            );
        }
        assert_eq!(map.bonds(&hm).expect("bonds").n_bond(), 0);
    }

    /// At a bond face the coefficient is the series value, and it is the
    /// number the conduction solver's own series conductance produces on the
    /// same mesh with the same per-cell conductivity. The linear form
    /// over-predicts the face conductance by the two-material factor of
    /// SPEC-LIT 46.2's argument, which is what the last check measures.
    #[test]
    fn the_bond_coefficient_is_s46_2_and_conduction_agrees() {
        let hm = prototype::block(6).expect("block");
        let a = Material::steel(0.3);
        let b = Material { e: 100e9, nu: 0.3, alpha: 2.0e-5 };
        let map = split_map(&hm, a, b, BondTreatment::Series);
        let (gamma_mag_sf, b_gamma_mag_sf) = map.implicit_coefficients(&hm);

        let n = hm.n_cells;
        let k: Vec<Tensor> = (0..n)
            .map(|c| {
                let mat = if hm.c[c].x < 0.5 { a } else { b };
                Conductivity::Isotropic(mat.implicit_gamma()).tensor()
            })
            .collect();
        let cond = Conduction::build(&one_region(&hm), &k, vec![1.0; n]).expect("conduction");
        for f in 0..hm.n_internal_faces {
            let want = cond.gamma_mag_sf[f];
            assert!(
                ((gamma_mag_sf[f] - want) / want.abs().max(1e-300)).abs() <= 1e-12,
                "internal face {f}: {} vs {}",
                gamma_mag_sf[f],
                want
            );
        }
        for bf in 0..hm.n_boundary_faces {
            let want = cond.b_gamma_mag_sf[bf];
            assert!(
                ((b_gamma_mag_sf[bf] - want) / want.abs().max(1e-300)).abs() <= 1e-12,
                "boundary face {bf}"
            );
        }

        let gammas = [a.implicit_gamma(), b.implicit_gamma()];
        let bonds = map.bonds(&hm).expect("bonds");
        assert!(bonds.n_bond() > 0);
        for &f in &bonds.bond_face {
            let fu = f as usize;
            let (o, nb) = (hm.owner[fu] as usize, hm.neighbour[fu] as usize);
            let (cp, cn) = (map.cell[o] as usize, map.cell[nb] as usize);
            let (gp, gn) = (gammas[cp], gammas[cn]);
            let w = hm.weights[fu];
            let linear = hm.mag_sf[fu] * (w * gp + (1.0 - w) * gn);
            let series = gamma_mag_sf[fu];
            let r = gn / gp;
            let factor = (1.0 + r) * (1.0 + r) / (4.0 * r);
            assert!(
                ((linear / series - factor) / factor).abs() <= 1e-12,
                "bond face {f}: linear/series = {} vs {factor}",
                linear / series
            );
        }
    }

    /// The face traction evaluated from the owner side and from the
    /// neighbour side of the same face agree: that is what the face
    /// displacement was solved for.
    #[test]
    fn the_bond_traction_is_the_same_from_both_sides() {
        let mut rng = Lcg::new(0x5EED);
        for case in 0..20 {
            let fs = draw(&mut rng);
            let (_, mp) = mat_at(&mut rng);
            let (_, mn) = mat_at(&mut rng);
            let bf = bond_face_values(
                BondTreatment::Series,
                fs.w,
                fs.sf,
                fs.cf,
                fs.c_p,
                fs.c_n,
                fs.u_p,
                fs.u_n,
                fs.g_p,
                fs.g_n,
                fs.t_p,
                fs.t_n,
                mp,
                mn,
            );
            let g_p_side = (bf.u_f - fs.u_p) / bf.d_p;
            let g_n_side = (fs.u_n - bf.u_f) / bf.d_n;
            let t_p = side_traction(mp, bf.g_t, g_p_side, bf.theta_p, bf.n);
            let t_n = side_traction(mn, bf.g_t, g_n_side, bf.theta_n, bf.n);
            let rel = (t_p - t_n).mag() / t_p.mag().max(1e-300);
            assert!(rel <= 1e-12, "case {case}: {rel:e}");
        }
    }

    /// Equal materials make the solved face displacement the linear
    /// interpolate of the cell values: the resistance weights collapse onto
    /// the mesh weights, so the bond face adds nothing a plain face would
    /// not say.
    #[test]
    fn equal_materials_make_the_bond_face_the_linear_interpolate() {
        let mut rng = Lcg::new(0x5EED + 1);
        for case in 0..20 {
            let fs = draw(&mut rng);
            let (_, m) = mat_at(&mut rng);
            let bf = bond_face_values(
                BondTreatment::Series,
                fs.w,
                fs.sf,
                fs.cf,
                fs.c_p,
                fs.c_n,
                fs.u_p,
                fs.u_n,
                fs.g_p,
                fs.g_n,
                fs.t_p,
                fs.t_n,
                m,
                m,
            );
            let linear = fs.u_p * fs.w + fs.u_n * (1.0 - fs.w);
            let rel = (bf.u_f - linear).mag() / linear.mag().max(1e-300);
            assert!(rel <= 1e-14, "case {case}: {rel:e}");
        }
    }

    /// With equal moduli and a linear displacement field both treatments
    /// agree where the treatment cannot matter, and differ in the one place
    /// they must: the normal traction carries the resistance-weighted
    /// thermal thrust on the series side and the linearly weighted one on
    /// the linear side, and the solved face displacement carries the
    /// resistance-weighted balance.
    #[test]
    fn equal_moduli_and_a_linear_field_make_series_and_linear_agree() {
        let mut rng = Lcg::new(0x5EED + 2);
        for case in 0..20 {
            let mut fs = draw(&mut rng);
            let (_, mp) = mat_at(&mut rng);
            let mn = CellMaterial { beta_alpha: mp.beta_alpha * 2.0, ..mp };
            fs.g_n = fs.g_p;
            // u_N = u_P + (d_P + d_N) (n . G): the linear field's own difference
            let d_total = (fs.c_n - fs.c_p).mag();
            let ndg = {
                let n = fs.n;
                Vec3::new(
                    n.x * fs.g_p.xx + n.y * fs.g_p.yx + n.z * fs.g_p.zx,
                    n.x * fs.g_p.xy + n.y * fs.g_p.yy + n.z * fs.g_p.zy,
                    n.x * fs.g_p.xz + n.y * fs.g_p.yz + n.z * fs.g_p.zz,
                )
            };
            fs.u_n = fs.u_p + ndg * d_total;
            let d_p = fs.cf.mag();
            let d_n = (fs.c_n - fs.cf).mag();

            let series = bond_face_values(
                BondTreatment::Series, fs.w, fs.sf, fs.cf, fs.c_p, fs.c_n,
                fs.u_p, fs.u_n, fs.g_p, fs.g_n, fs.t_p, fs.t_n, mp, mn,
            );
            let linear = bond_face_values(
                BondTreatment::Linear, fs.w, fs.sf, fs.cf, fs.c_p, fs.c_n,
                fs.u_p, fs.u_n, fs.g_p, fs.g_n, fs.t_p, fs.t_n, mp, mn,
            );
            let scale = series.t_f.mag().max(1e-300);

            // (a) equal distances: the two treatments agree outright. The
            // linear field is rebuilt on the symmetric geometry so the
            // one-sided derivative stays the field's own normal derivative.
            let d_a = (fs.cf - fs.c_p).mag();
            let c_n_a = fs.cf + fs.n * d_a;
            let u_n_a = fs.u_p + ndg * (2.0 * d_a);
            let s_a = bond_face_values(
                BondTreatment::Series, 0.5, fs.sf, fs.cf, fs.c_p, c_n_a,
                fs.u_p, u_n_a, fs.g_p, fs.g_n, fs.t_p, fs.t_n, mp, mn,
            );
            let l_a = bond_face_values(
                BondTreatment::Linear, 0.5, fs.sf, fs.cf, fs.c_p, c_n_a,
                fs.u_p, u_n_a, fs.g_p, fs.g_n, fs.t_p, fs.t_n, mp, mn,
            );
            let rel_a = (s_a.t_f - l_a.t_f).mag() / s_a.t_f.mag().max(1e-300);
            assert!(rel_a <= 1e-12, "case {case} (a): {rel_a:e}");

            // (b) unequal distances: tangential parts equal, normal parts
            // differing by exactly the weighted-thrust difference.
            let tang = |v: Vec3, n: Vec3| v - n * v.dot(n);
            let rel_t = (tang(series.t_f, series.n) - tang(linear.t_f, linear.n)).mag()
                / scale;
            assert!(rel_t <= 1e-12, "case {case} (b) tangential: {rel_t:e}");
            let d_norm = series.t_f.dot(series.n) - linear.t_f.dot(series.n);
            let want = (2.0 * fs.w - 1.0) * (series.theta_p - series.theta_n);
            assert!(
                ((d_norm - want) / scale).abs() <= 1e-12,
                "case {case} (b) normal: {d_norm} vs {want}"
            );

            // (c) the solved face displacement carries the resistance-weighted
            // thermal balance in its normal component.
            let a_p = mp.gamma() / d_p;
            let a_n = mn.gamma() / d_n;
            let lhs = series.u_f.dot(series.n)
                - (fs.u_p * fs.w + fs.u_n * (1.0 - fs.w)).dot(series.n);
            let want = -(series.theta_n - series.theta_p) / (a_p + a_n);
            let u_scale = series.u_f.mag().max(1e-300);
            assert!(
                ((lhs - want) / u_scale).abs() <= 1e-12,
                "case {case} (c): {lhs} vs {want}"
            );
        }
    }

    /// A cell centre sitting on its own bond face is refused, and the
    /// message says the mesh is what is broken.
    #[test]
    fn a_degenerate_bond_face_is_refused() {
        let axis = |n: usize| GradedAxis { lo: 0.0, hi: 1.0, n, expansion: 1.0, two_sided: false };
        let spec = BlockSpec {
            x: axis(2),
            y: axis(1),
            z: axis(1),
            patch_type: ["patch", "patch", "patch", "patch", "patch", "patch"].map(String::from),
            ..Default::default()
        };
        let mut hm = crate::blockgen::build_mesh(&spec).expect("block");
        assert_eq!(hm.n_internal_faces, 1);
        hm.cf[0] = hm.c[hm.owner[0] as usize];
        let map = MaterialMap::uniform("steel", Material::steel(0.3), hm.n_cells);
        // One cell of the pair must carry a different material for the face
        // to be a bond face at all.
        let mut cell = map.cell.clone();
        cell[hm.neighbour[0] as usize] = 1;
        let map = MaterialMap {
            names: vec!["steel".into(), "brass".into()],
            materials: vec![Material::steel(0.3), Material { e: 100e9, nu: 0.3, alpha: 2.0e-5 }],
            t_ref: vec![None, None],
            cell,
            bond: BondTreatment::Series,
        };
        let err = map.bonds(&hm).unwrap_err().to_string();
        assert!(err.contains("face 0"), "{err}");
        assert!(err.contains("degenerate") && err.contains("not the material map"), "{err}");
    }

    /// The summary names every material and says what the bond faces are
    /// treated with.
    #[test]
    fn describe_names_every_material_and_the_bond_count() {
        let hm = prototype::block(4).expect("block");
        let a = Material::steel(0.3);
        let b = Material { e: 100e9, nu: 0.3, alpha: 2.0e-5 };
        let map = split_map(&hm, a, b, BondTreatment::Series);
        let bonds = map.bonds(&hm).expect("bonds");
        let s = map.describe(bonds.n_bond());
        assert!(s.contains("steel"), "{s}");
        assert!(s.contains("brass"), "{s}");
        assert!(s.contains("bond faces: 16"), "{s}");
        assert!(s.contains("series"), "{s}");
    }
}
