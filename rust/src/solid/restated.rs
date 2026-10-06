// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Gate 95-G's bodies, host-only (§95.11): NAFEMS LE1 (the elliptic
//! membrane), LE10 (the thick plate) and LE11 (the solid
//! cylinder/taper/sphere under a temperature field) as a restatement and
//! openly published text give them; the Lamé ring the boundary-point
//! read-out is gated on; and that read-out, the least-squares linear fit
//! (S95.27) that carries a cell-centred stress to a boundary point.
//! Nothing here launches a kernel, touches a `Gpu`, or iterates anything.
//!
//! THE PRIMARY WAS NOT READ. *The Standard NAFEMS Benchmarks*, NAFEMS ref.
//! P18 (TNSB Rev. 3, 1990) was not bought; every number of LE1, LE10 and
//! LE11 below is a RESTATEMENT's, and §95.11 names each page.
//!
//! Written from:
//!   ESRD, Inc., *Benchmarks Guide: The Standard NAFEMS Benchmarks, Linear
//!     Elastic Tests* (2018), www.esrd.com/wp-content/uploads/dlm_uploads/
//!     Benchmarks-Guide-Standard-NAFEMS-Benchmarks-Linear-Elastic-Tests.pdf
//!     - material, boundary conditions, loads and targets, read as text;
//!     (c) 2018 ESRD, Inc., not redistributed
//!   github.com/masteryol/FEA-NAFEMS-Benchmarks-ANSYS, README - LE1's two
//!     ellipses, its thickness and point D, as text
//!   SimScale, "Thick Plate Under Pressure", www.simscale.com/docs/
//!     validation-cases/thick-plate-under-pressure/ - LE10's point table,
//!     ellipses and thickness, as text
//!   LEAP Australia, "NAFEMS Discovery Benchmark Series - Part 2: Pressure
//!     Plates", www.leapaust.com.au/blog/fea/
//!     nafems-discovery-series-part-2-pressure-plates/ - LE10 is LE1's
//!     plate at 0.6 m
//!   CEA, "Test elas11 Description sheet", www-cast3m.cea.fr/html/
//!     CasTestsCastem/node9.html - LE11's section points, temperature and
//!     end conditions, as text
//!   Precise Simulation, "Temperature Loading of a Tapered Cylinder",
//!     www.featool.com/doc/Structural_Mechanics_07_temperature_loading1 -
//!     the same section from the same numbers
//!   S. P. Timoshenko, J. N. Goodier, *Theory of Elasticity*, 3rd ed.,
//!     McGraw-Hill (1970) ch. 2 - the plane-strain material with plane
//!     stress's in-plane law; ch. 4 - the thick-walled cylinder under
//!     internal pressure
//!   this crate's `src/solid/fixtures.rs` - the mapped-block pattern
//!
//! The pages above document proprietary codes' runs and were read as
//! text; no code's source and no input deck was opened, and the GPL
//! FeenoX/Fino examples of these tests were not opened. OpenFOAM and
//! solids4foam are GPL and were not opened.
//! No GPL-licensed source was consulted.

use crate::blockgen::{self, BlockSpec, GradedAxis, PatchWindow};
use crate::error::{Error, Result};
use crate::io::polymesh::{build_host_mesh, PolyMeshRaw};
use crate::mesh::HostMesh;
use crate::solid::bc::{CompBc, PatchBcs};
use crate::solid::fixtures::{annulus, Annulus};
use crate::solid::Material;
use crate::types::{to_dvec3, to_vec3};
use crate::{Scalar, Vec3};
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4, FRAC_PI_6};

/// LE1/LE10/LE11's material as the restatement gives it (ESRD 2018).
pub const NAFEMS_E: Scalar = 210.0e9;
pub const NAFEMS_NU: Scalar = 0.3;
/// LE1's thickness [m] and its outward pressure on the outer edge BC [Pa].
pub const LE1_THICKNESS: Scalar = 0.1;
pub const LE1_P: Scalar = 10.0e6;
/// LE10's thickness [m] and its normal pressure on the upper face [Pa].
pub const LE10_THICKNESS: Scalar = 0.6;
pub const LE10_P: Scalar = 1.0e6;
/// LE11's expansion coefficient [1/degC], its height and the taper's top.
pub const LE11_ALPHA: Scalar = 2.3e-4;
pub const LE11_HEIGHT: Scalar = 1.79;
pub const LE11_TAPER_TOP: Scalar = 1.39;
/// The Lamé ring: §95.10's quarter annulus under an internal pressure.
pub const LAME_R_IN: Scalar = 0.5;
pub const LAME_R_OUT: Scalar = 1.0;
pub const LAME_P: Scalar = 1.0e6;

/// A mapped block, the point a published stress is read at, and the
/// stencil (S95.27) reads it from. `n` is the cell count along the block's
/// own x, y, z axes; cell `(i, j, k)` is id `i + n[0] (j + n[1] k)`, the
/// order `blockgen` numbers cells in and `build_host_mesh` keeps.
pub struct RestatedBody {
    pub mesh: HostMesh,
    pub raw: PolyMeshRaw,
    pub n: [usize; 3],
    pub target: Vec3,
    pub stencil: Vec<usize>,
    pub fit_dims: [bool; 3],
}

const NAMES: [&str; 6] = ["inner", "outer", "cut0", "cut90", "zmin", "zmax"];

fn axis(lo: Scalar, hi: Scalar, n: usize) -> GradedAxis {
    GradedAxis { lo, hi, n, expansion: 1.0, two_sided: false }
}

fn block(x: GradedAxis, y: GradedAxis, z: GradedAxis, windows: Vec<PatchWindow>) -> BlockSpec {
    BlockSpec {
        x,
        y,
        z,
        patch_name: NAMES.map(String::from),
        patch_type: ["patch"; 6].map(String::from),
        windows,
        cyclic: Vec::new(),
    }
}

/// The cell ids of the index box `i[0]..i[1] x j[0]..j[1] x k[0]..k[1]`
/// (half-open), `k` outermost and `i` innermost.
pub fn corner_stencil(n: [usize; 3], i: [usize; 2], j: [usize; 2], k: [usize; 2]) -> Vec<usize> {
    let mut out = Vec::new();
    for kk in k[0]..k[1] {
        for jj in j[0]..j[1] {
            for ii in i[0]..i[1] {
                out.push(ii + n[0] * (jj + n[1] * kk));
            }
        }
    }
    out
}

/// (S95.27): the constant term of the least-squares linear fit of
/// `values` over `stencil`'s cell centres, in the offsets from `target`
/// scaled by the largest one, spanning the directions `dims` names. The
/// normal equations are solved by Gaussian elimination with partial
/// pivoting; a stencil that does not span the fit is refused by name.
pub fn extrapolate_linear(
    m: &HostMesh,
    values: &[Scalar],
    stencil: &[usize],
    target: Vec3,
    dims: [bool; 3],
) -> Result<Scalar> {
    if values.len() != m.n_cells {
        return Err(Error::Config(format!(
            "restated: {} values for {} cells",
            values.len(),
            m.n_cells
        )));
    }
    let q = 1 + dims.iter().filter(|&&d| d).count();
    if stencil.len() < q {
        return Err(Error::Config(format!(
            "restated: a {}-cell stencil cannot carry a {q}-term linear fit",
            stencil.len()
        )));
    }
    let mut s: Scalar = 0.0;
    for &c in stencil {
        if c >= m.n_cells {
            return Err(Error::Config(format!(
                "restated: stencil cell {c} is not one of the {} cells",
                m.n_cells
            )));
        }
        s = s.max((m.c[c] - target).mag());
    }
    if !(s > 0.0) {
        return Err(Error::Config("restated: the stencil sits on the point itself".to_string()));
    }
    let mut n = [[0.0 as Scalar; 4]; 4];
    let mut r = [0.0 as Scalar; 4];
    for &c in stencil {
        let d = (m.c[c] - target) / s;
        let comp = [d.x, d.y, d.z];
        let mut row = [0.0 as Scalar; 4];
        row[0] = 1.0;
        let mut k = 1;
        for a in 0..3 {
            if dims[a] {
                row[k] = comp[a];
                k += 1;
            }
        }
        for a in 0..q {
            r[a] += row[a] * values[c];
            for b in 0..q {
                n[a][b] += row[a] * row[b];
            }
        }
    }
    let scale = (0..q).map(|a| n[a][a].abs()).fold(0.0 as Scalar, Scalar::max);
    for col in 0..q {
        let mut piv = col;
        for row in col + 1..q {
            if n[row][col].abs() > n[piv][col].abs() {
                piv = row;
            }
        }
        if n[piv][col].abs() <= 1.0e-9 * scale {
            return Err(Error::Config(format!(
                "restated: the {}-cell stencil does not span the fit's directions {dims:?}",
                stencil.len()
            )));
        }
        n.swap(col, piv);
        r.swap(col, piv);
        for row in col + 1..q {
            let f = n[row][col] / n[col][col];
            for b in col..q {
                n[row][b] -= f * n[col][b];
            }
            r[row] -= f * r[col];
        }
    }
    let mut x = [0.0 as Scalar; 4];
    for a in (0..q).rev() {
        let mut acc = r[a];
        for b in a + 1..q {
            acc -= n[a][b] * x[b];
        }
        x[a] = acc / n[a][a];
    }
    Ok(x[0])
}

/// `t = -p n` on every face of the patch named `patch` and zero on every
/// other boundary face: a positive `p` pushes on the body, a negative one
/// pulls (LE1's "outward pressure" is `p = -LE1_P`).
pub fn patch_pressure(m: &HostMesh, patch: &str, p: Scalar) -> Result<Vec<Vec3>> {
    let info = m
        .patches
        .iter()
        .find(|q| q.name == patch)
        .ok_or_else(|| Error::Config(format!("restated: no patch named '{patch}'")))?;
    let mut t = vec![Vec3::new(0.0, 0.0, 0.0); m.n_boundary_faces];
    for bf in info.start..info.start + info.size {
        t[bf] = m.b_sf[bf] * (-p / m.b_mag_sf[bf]);
    }
    Ok(t)
}

/// (S95.25): the ellipse of the family between AD (`t = 0`) and BC (`t = 1`).
fn ellipse_family(t: Scalar, phi: Scalar) -> (Scalar, Scalar) {
    ((2.0 + 1.25 * t) * phi.cos(), (1.0 + 1.75 * t) * phi.sin())
}

/// The quarter elliptic annulus of LE1/LE10, `n_z` layers over `thickness`.
fn elliptic_annulus(
    n_t: usize,
    n_phi: usize,
    thickness: Scalar,
    n_z: usize,
    windows: Vec<PatchWindow>,
) -> Result<(HostMesh, PolyMeshRaw)> {
    let spec = block(
        axis(0.0, 1.0, n_t),
        axis(0.0, FRAC_PI_2 as Scalar, n_phi),
        axis(0.0, thickness, n_z),
        windows,
    );
    let mut raw = blockgen::raw_mesh(&spec)?;
    for p in raw.points.iter_mut() {
        let q = to_vec3(*p);
        let cut90 = (q.y - FRAC_PI_2 as Scalar).abs() < 1e-12;
        let cut0 = q.y.abs() < 1e-12;
        let (x, y) = ellipse_family(q.x, q.y);
        let mut q = Vec3::new(x, y, q.z);
        if cut90 {
            q.x = 0.0;
        }
        if cut0 {
            q.y = 0.0;
        }
        *p = to_dvec3(q);
    }
    let mesh = build_host_mesh(&raw)?;
    Ok((mesh, raw))
}

/// LE1: the membrane, one cell thick, `n_t` cells across the wall and
/// `n_phi` around it; the point is D = (2, 0) at mid-thickness.
pub fn le1_membrane(n_t: usize, n_phi: usize) -> Result<RestatedBody> {
    if n_t < 2 || n_phi < 2 {
        return Err(Error::Config(format!(
            "restated: LE1 needs two cells across the wall and around it at least, got {n_t} x {n_phi}"
        )));
    }
    let (mesh, raw) = elliptic_annulus(n_t, n_phi, LE1_THICKNESS, 1, Vec::new())?;
    Ok(RestatedBody {
        mesh,
        raw,
        n: [n_t, n_phi, 1],
        target: Vec3::new(2.0, 0.0, 0.5 * LE1_THICKNESS),
        stencil: corner_stencil([n_t, n_phi, 1], [0, 2], [0, 2], [0, 1]),
        fit_dims: [true, true, false],
    })
}

/// LE1 in plane strain (D3), slot order `inner outer cut0 cut90 zmin zmax`:
/// DC (`y = 0`) fixes `u_y`, AB (`x = 0`) fixes `u_x`, both faces fix `u_z`;
/// the outer edge's pull is a traction value set per face.
pub fn le1_bcs() -> PatchBcs {
    let mut b = [[CompBc::Traction(0.0); 3]; 6];
    b[2][1] = CompBc::Fixed(0.0);
    b[3][0] = CompBc::Fixed(0.0);
    b[4][2] = CompBc::Fixed(0.0);
    b[5][2] = CompBc::Fixed(0.0);
    b
}

/// (S95.24): the plane-strain material whose in-plane law is plane
/// stress's with `(NAFEMS_E, NAFEMS_NU)` (Timoshenko & Goodier ch. 2).
pub fn le1_plane_strain_material() -> Material {
    let nu = NAFEMS_NU;
    Material { e: NAFEMS_E * (1.0 + 2.0 * nu) / ((1.0 + nu) * (1.0 + nu)), nu: nu / (1.0 + nu), alpha: 0.0 }
}

/// LE10's material, isothermal.
pub fn nafems_material() -> Material {
    Material { e: NAFEMS_E, nu: NAFEMS_NU, alpha: 0.0 }
}

/// LE10: the plate, `n_z` (even) layers over its 0.6 m; the outer faces of
/// the two layers either side of the mid-plane are the patch `outer_mid`,
/// the band that stands for the line `u_z = 0` (D4). Patch order:
/// `inner outer_mid outer cut0 cut90 zmin zmax`. The point is D = (2, 0, 0.6).
pub fn le10_plate(n_t: usize, n_phi: usize, n_z: usize) -> Result<RestatedBody> {
    if n_t < 2 || n_phi < 2 || n_z < 2 || n_z % 2 != 0 {
        return Err(Error::Config(format!(
            "restated: LE10 needs n_t, n_phi >= 2 and an even n_z >= 2 - the band is the two \
             layers either side of the mid-plane - got {n_t} x {n_phi} x {n_z}"
        )));
    }
    let band = PatchWindow {
        slot: 1,
        lo: [0, n_z / 2 - 1],
        hi: [n_phi, n_z / 2 + 1],
        name: "outer_mid".to_string(),
        type_name: "patch".to_string(),
    };
    let (mesh, raw) = elliptic_annulus(n_t, n_phi, LE10_THICKNESS, n_z, vec![band])?;
    Ok(RestatedBody {
        mesh,
        raw,
        n: [n_t, n_phi, n_z],
        target: Vec3::new(2.0, 0.0, LE10_THICKNESS),
        stencil: corner_stencil([n_t, n_phi, n_z], [0, 2], [0, 2], [n_z - 2, n_z]),
        fit_dims: [true, true, true],
    })
}

/// LE10's conditions, by patch NAME: symmetry on `cut0` (`y = 0`) and
/// `cut90` (`x = 0`); `u_x = u_y = 0` on the outer face, and `u_z = 0` on
/// its band too; everything else traction (the upper face's pressure is a
/// value set per face).
pub fn le10_bcs(m: &HostMesh) -> Result<Vec<[CompBc; 3]>> {
    let t0 = CompBc::Traction(0.0);
    let f0 = CompBc::Fixed(0.0);
    m.patches
        .iter()
        .map(|p| match p.name.as_str() {
            "inner" | "zmin" | "zmax" => Ok([t0, t0, t0]),
            "outer" => Ok([f0, f0, t0]),
            "outer_mid" => Ok([f0, f0, f0]),
            "cut0" => Ok([t0, f0, t0]),
            "cut90" => Ok([f0, t0, t0]),
            other => Err(Error::Config(format!(
                "restated: LE10 has no condition for a patch named '{other}'"
            ))),
        })
        .collect()
}

/// LE11's inner boundary A-E-H at normalised arc length `s`, as `(r, z)`:
/// the unit sphere from A = (1, 0) up to E = (1/sqrt 2, 1/sqrt 2), then the
/// bore `r = 1/sqrt 2` up to H = (1/sqrt 2, 1.79).
pub fn le11_inner(s: Scalar) -> (Scalar, Scalar) {
    let arc = FRAC_PI_4 as Scalar;
    let bore = FRAC_1_SQRT_2 as Scalar;
    let l = s * (arc + (LE11_HEIGHT - bore));
    if l <= arc {
        (l.cos(), l.sin())
    } else {
        (bore, bore + (l - arc))
    }
}

/// LE11's outer boundary B-C-G-I at normalised arc length `s`, as `(r, z)`:
/// the sphere of radius 1.4 from B = (1.4, 0) up to C = (1.4 cos 30deg, 0.7),
/// the taper from C to G = (1, 1.39), the cylinder `r = 1` up to I = (1, 1.79).
pub fn le11_outer(s: Scalar) -> (Scalar, Scalar) {
    let rs = 1.4 as Scalar;
    let a = FRAC_PI_6 as Scalar;
    let arc = rs * a;
    let (cr, cz) = (rs * a.cos(), rs * a.sin());
    let (gr, gz) = (1.0 as Scalar, LE11_TAPER_TOP);
    let taper = ((gr - cr) * (gr - cr) + (gz - cz) * (gz - cz)).sqrt();
    let l = s * (arc + taper + (LE11_HEIGHT - LE11_TAPER_TOP));
    if l <= arc {
        (rs * (l / rs).cos(), rs * (l / rs).sin())
    } else if l <= arc + taper {
        let f = (l - arc) / taper;
        (cr + f * (gr - cr), cz + f * (gz - cz))
    } else {
        (gr, gz + (l - arc - taper))
    }
}

/// LE11: the quarter body, block `(t, theta, s)` mapped by
/// `(r, z) = (1 - t) inner(s) + t outer(s)`, `x = r cos theta`,
/// `y = r sin theta`; slots `inner outer cut0 cut90 zmin zmax` are the bore
/// side, the outside, `y = 0`, `x = 0`, `z = 0` and `z = 1.79`. The point is
/// A = (1, 0, 0).
pub fn le11_body(n_t: usize, n_theta: usize, n_s: usize) -> Result<RestatedBody> {
    if n_t < 2 || n_theta < 2 || n_s < 2 {
        return Err(Error::Config(format!(
            "restated: LE11 needs two cells in every direction at least, got {n_t} x {n_theta} x {n_s}"
        )));
    }
    let spec = block(
        axis(0.0, 1.0, n_t),
        axis(0.0, FRAC_PI_2 as Scalar, n_theta),
        axis(0.0, 1.0, n_s),
        Vec::new(),
    );
    let mut raw = blockgen::raw_mesh(&spec)?;
    for p in raw.points.iter_mut() {
        let q = to_vec3(*p);
        let cut90 = (q.y - FRAC_PI_2 as Scalar).abs() < 1e-12;
        let cut0 = q.y.abs() < 1e-12;
        let bottom = q.z.abs() < 1e-12;
        let top = (q.z - 1.0).abs() < 1e-12;
        let (ri, zi) = le11_inner(q.z);
        let (ro, zo) = le11_outer(q.z);
        let r = (1.0 - q.x) * ri + q.x * ro;
        let z = (1.0 - q.x) * zi + q.x * zo;
        let mut q = Vec3::new(r * q.y.cos(), r * q.y.sin(), z);
        if cut90 {
            q.x = 0.0;
        }
        if cut0 {
            q.y = 0.0;
        }
        if bottom {
            q.z = 0.0;
        }
        if top {
            q.z = LE11_HEIGHT;
        }
        *p = to_dvec3(q);
    }
    let mesh = build_host_mesh(&raw)?;
    Ok(RestatedBody {
        mesh,
        raw,
        n: [n_t, n_theta, n_s],
        target: Vec3::new(1.0, 0.0, 0.0),
        stencil: corner_stencil([n_t, n_theta, n_s], [0, 2], [0, 2], [0, 2]),
        fit_dims: [true, true, true],
    })
}

/// LE11's conditions in slot order: the bore side and the outside free,
/// symmetry on `y = 0` and `x = 0`, `u_z = 0` on both ends.
pub fn le11_bcs() -> PatchBcs {
    let mut b = [[CompBc::Traction(0.0); 3]; 6];
    b[2][1] = CompBc::Fixed(0.0);
    b[3][0] = CompBc::Fixed(0.0);
    b[4][2] = CompBc::Fixed(0.0);
    b[5][2] = CompBc::Fixed(0.0);
    b
}

/// LE11's material: the restatement's `E`, `nu` and `alpha = 2.3e-4 /degC`.
pub fn le11_material() -> Material {
    Material { e: NAFEMS_E, nu: NAFEMS_NU, alpha: LE11_ALPHA }
}

/// (S95.26): `T = sqrt(x^2 + y^2) + z` [degC], `T_ref = 0`.
pub fn le11_temperature(p: Vec3) -> Scalar {
    (p.x * p.x + p.y * p.y).sqrt() + p.z
}

/// (S95.26) at every cell centre and every boundary-face centre.
pub fn le11_temperatures(m: &HostMesh) -> (Vec<Scalar>, Vec<Scalar>) {
    (m.c.iter().map(|&p| le11_temperature(p)).collect(), m.b_cf.iter().map(|&p| le11_temperature(p)).collect())
}

/// The Lamé ring: `fixtures::annulus(nr, 2 nr, 2, 0.5, 1.0)`, plane strain
/// with `fixtures::quarter_annulus_plane_strain_bcs`; the point is the bore
/// on the symmetry plane `y = 0` at mid-length, `(r_in, 0, L/2)`.
pub fn lame_ring(nr: usize) -> Result<RestatedBody> {
    if nr < 2 {
        return Err(Error::Config(format!("restated: the Lamé ring needs nr >= 2, got {nr}")));
    }
    let Annulus { mesh, raw, length, .. } = annulus(nr, 2 * nr, 2, LAME_R_IN, LAME_R_OUT)?;
    Ok(RestatedBody {
        mesh,
        raw,
        n: [nr, 2 * nr, 2],
        target: Vec3::new(LAME_R_IN, 0.0, 0.5 * length),
        stencil: corner_stencil([nr, 2 * nr, 2], [0, 2], [0, 2], [0, 2]),
        fit_dims: [true, true, true],
    })
}

/// (S95.28): the hoop stress at the bore of a thick-walled cylinder under
/// an internal pressure `p` (Timoshenko & Goodier ch. 4).
pub fn lame_hoop_at_bore(p: Scalar, r_in: Scalar, r_out: Scalar) -> Scalar {
    p * (r_out * r_out + r_in * r_in) / (r_out * r_out - r_in * r_in)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
    fn every_restated_body_closes_and_has_positive_volume() {
        let le1 = le1_membrane(8, 16).unwrap();
        let r = le1.mesh.check();
        println!(
            "LE1 cells {} closure {:e} max_non_orth_deg {}",
            le1.mesh.n_cells, r.max_closure_error, r.max_non_orth_deg
        );
        assert!(r.max_closure_error <= 1e-12);
        assert!(r.min_volume > 0.0);
        let le10 = le10_plate(6, 12, 4).unwrap();
        let r = le10.mesh.check();
        println!(
            "LE10 cells {} closure {:e} max_non_orth_deg {}",
            le10.mesh.n_cells, r.max_closure_error, r.max_non_orth_deg
        );
        assert!(r.max_closure_error <= 1e-12);
        assert!(r.min_volume > 0.0);
        let le11 = le11_body(4, 8, 16).unwrap();
        let r = le11.mesh.check();
        println!(
            "LE11 cells {} closure {:e} max_non_orth_deg {}",
            le11.mesh.n_cells, r.max_closure_error, r.max_non_orth_deg
        );
        assert!(r.max_closure_error <= 1e-12);
        assert!(r.min_volume > 0.0);
        let lame = lame_ring(6).unwrap();
        let r = lame.mesh.check();
        println!(
            "Lame cells {} closure {:e} max_non_orth_deg {}",
            lame.mesh.n_cells, r.max_closure_error, r.max_non_orth_deg
        );
        assert!(r.max_closure_error <= 1e-12);
        assert!(r.min_volume > 0.0);
    }
    #[test]
    fn the_elliptic_bodies_carry_the_published_area_and_patches() {
        let v_ref = |th: Scalar| FRAC_PI_4 as Scalar * (3.25 * 2.75 - 2.0 * 1.0) * th;
        let mut e: Vec<Scalar> = Vec::new();
        for (nt, nphi) in [(8usize, 16usize), (16, 32)] {
            let b = le1_membrane(nt, nphi).unwrap();
            let r = b.mesh.check();
            e.push((r.total_volume - v_ref(0.1)).abs() / v_ref(0.1));
        }
        println!("LE1 volume errors: coarse {:e} fine {:e}", e[0], e[1]);
        assert!(e[1] <= 1e-2, "fine-mesh volume error {:e}", e[1]);
        assert!(e[0] / e[1] >= 3.0, "convergence ratio {}", e[0] / e[1]);
        let le1 = le1_membrane(8, 16).unwrap();
        let want = [
            ("inner", 16usize),
            ("outer", 16),
            ("cut0", 8),
            ("cut90", 8),
            ("zmin", 128),
            ("zmax", 128),
        ];
        assert_eq!(le1.mesh.patches.len(), want.len());
        for (p, w) in le1.mesh.patches.iter().zip(want.iter()) {
            assert_eq!(p.name, w.0);
            assert_eq!(p.size, w.1);
        }
        let le10 = le10_plate(6, 12, 4).unwrap();
        let want10 = [
            ("inner", 48usize),
            ("outer_mid", 24),
            ("outer", 24),
            ("cut0", 24),
            ("cut90", 24),
            ("zmin", 72),
            ("zmax", 72),
        ];
        assert_eq!(le10.mesh.patches.len(), want10.len());
        for (p, w) in le10.mesh.patches.iter().zip(want10.iter()) {
            assert_eq!(p.name, w.0);
            assert_eq!(p.size, w.1);
        }
        for p in le10.mesh.patches.iter() {
            if p.name == "outer_mid" {
                for bf in p.start..p.start + p.size {
                    let zc = le10.mesh.b_cf[bf].z;
                    assert!((zc - 0.3).abs() <= 0.15 + 1e-9, "outer_mid centre z {zc}");
                }
            }
            if p.name == "outer" {
                for bf in p.start..p.start + p.size {
                    assert!((le10.mesh.b_cf[bf].z - 0.3).abs() >= 0.15 - 1e-9);
                }
            }
        }
        for b in [&le1, &le10] {
            for p in b.mesh.patches.iter() {
                if p.name == "cut0" {
                    for bf in p.start..p.start + p.size {
                        assert_eq!(b.mesh.b_cf[bf].y, 0.0);
                    }
                }
                if p.name == "cut90" {
                    for bf in p.start..p.start + p.size {
                        assert_eq!(b.mesh.b_cf[bf].x, 0.0);
                    }
                }
            }
        }
        println!("cut0/cut90 face centres exact on y = 0 and x = 0 for LE1 and LE10");
    }
    #[test]
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
    fn le11_is_the_published_section_revolved() {
        let bore = FRAC_1_SQRT_2 as Scalar;
        let (r, z) = le11_inner(0.0);
        assert!((r - 1.0).abs() <= 1e-12 && z.abs() <= 1e-12, "A");
        let (r, z) = le11_inner(1.0);
        assert!((r - bore).abs() <= 1e-12 && (z - LE11_HEIGHT).abs() <= 1e-12, "H");
        let (r, z) = le11_outer(0.0);
        assert!((r - 1.4).abs() <= 1e-12 && z.abs() <= 1e-12, "B");
        let (r, z) = le11_outer(1.0);
        assert!((r - 1.0).abs() <= 1e-12 && (z - LE11_HEIGHT).abs() <= 1e-12, "I");
        let rs = 1.4 as Scalar;
        let a = FRAC_PI_6 as Scalar;
        let arc = rs * a;
        let (cr, cz) = (rs * a.cos(), rs * a.sin());
        let taper = ((1.0 - cr) * (1.0 - cr) + (LE11_TAPER_TOP - cz) * (LE11_TAPER_TOP - cz)).sqrt();
        let total = arc + taper + (LE11_HEIGHT - LE11_TAPER_TOP);
        let (r, z) = le11_outer(arc / total);
        assert!((r - cr).abs() <= 1e-12 && (z - cz).abs() <= 1e-12, "C");
        let (r, z) = le11_outer((arc + taper) / total);
        assert!((r - 1.0).abs() <= 1e-12 && (z - LE11_TAPER_TOP).abs() <= 1e-12, "G");
        println!("inner A-H and outer B-C-G-I endpoints, C and G within 1e-12");
        let b = le11_body(8, 16, 32).unwrap();
        for p in b.mesh.patches.iter() {
            let bad = match p.name.as_str() {
                "zmin" => (p.start..p.start + p.size).any(|f| b.mesh.b_cf[f].z != 0.0),
                "zmax" => {
                    (p.start..p.start + p.size).any(|f| (b.mesh.b_cf[f].z - LE11_HEIGHT).abs() > 1e-12)
                }
                "cut0" => (p.start..p.start + p.size).any(|f| b.mesh.b_cf[f].y != 0.0),
                "cut90" => (p.start..p.start + p.size).any(|f| b.mesh.b_cf[f].x != 0.0),
                _ => false,
            };
            assert!(!bad, "patch {} has a misplaced face centre", p.name);
        }
        let ro = |z: Scalar| -> Scalar {
            if z <= 0.7 {
                (1.96 - z * z).sqrt()
            } else if z <= LE11_TAPER_TOP {
                cr + (z - 0.7) * (1.0 - cr) / (LE11_TAPER_TOP - 0.7)
            } else {
                1.0
            }
        };
        let ri = |z: Scalar| -> Scalar {
            if z <= bore {
                (1.0 - z * z).sqrt()
            } else {
                bore
            }
        };
        let n_int = 20_000usize;
        let h = LE11_HEIGHT / n_int as Scalar;
        let f = |z: Scalar| ro(z) * ro(z) - ri(z) * ri(z);
        let mut v = f(0.0) + f(LE11_HEIGHT);
        for i in 1..n_int {
            v += f(h * i as Scalar) * if i % 2 == 1 { 4.0 } else { 2.0 };
        }
        v *= h / 3.0;
        let v_ref = FRAC_PI_4 as Scalar * v;
        let r = b.mesh.check();
        let gap = (r.total_volume - v_ref).abs() / v_ref;
        println!(
            "LE11 volume: mesh {:.6} reference {:.6} relative gap {:e}",
            r.total_volume, v_ref, gap
        );
        assert!(gap <= 1e-2, "LE11 volume off by {gap:e} relative");
    }
    #[test]
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
    fn a_linear_field_is_carried_to_the_point_exactly() {
        for name in ["LE1", "LE10", "LE11", "Lame"] {
            let b = match name {
                "LE1" => le1_membrane(8, 16).unwrap(),
                "LE10" => le10_plate(6, 12, 4).unwrap(),
                "LE11" => le11_body(4, 8, 16).unwrap(),
                _ => lame_ring(6).unwrap(),
            };
            let v: Vec<Scalar> = b
                .mesh
                .c
                .iter()
                .map(|&p| 3.0 + 2.0 * p.x - 5.0 * p.y + 7.0 * p.z)
                .collect();
            let got = extrapolate_linear(&b.mesh, &v, &b.stencil, b.target, b.fit_dims).unwrap();
            let want = 3.0 + 2.0 * b.target.x - 5.0 * b.target.y + 7.0 * b.target.z;
            println!("{name}: fit {got:.12} exact {want:.12}");
            assert!(
                (got - want).abs() <= 1e-10 * (1.0 + want.abs()),
                "{name}: {got} against {want}"
            );
        }
    }
    #[test]
    fn a_smooth_field_is_carried_at_second_order() {
        let mut e: Vec<Scalar> = Vec::new();
        for (nt, nphi) in [(8usize, 16usize), (16, 32), (32, 64)] {
            let b = le1_membrane(nt, nphi).unwrap();
            let v: Vec<Scalar> = b
                .mesh
                .c
                .iter()
                .map(|&p| p.x * p.x + p.x * p.y + p.y * p.y)
                .collect();
            let got = extrapolate_linear(&b.mesh, &v, &b.stencil, b.target, b.fit_dims).unwrap();
            e.push((got - 4.0).abs());
        }
        println!("errors {:e} {:e} {:e}", e[0], e[1], e[2]);
        println!(
            "log2 ratios {:.3} {:.3}",
            (e[0] / e[1]).log2(),
            (e[1] / e[2]).log2()
        );
        assert!(e[0] / e[1] >= 2.8, "first ratio {}", e[0] / e[1]);
        assert!(e[1] / e[2] >= 2.8, "second ratio {}", e[1] / e[2]);
    }
    #[test]
    fn the_stencil_holds_the_cell_nearest_the_point() {
        for name in ["LE1", "LE10", "LE11", "Lame"] {
            let b = match name {
                "LE1" => le1_membrane(8, 16).unwrap(),
                "LE10" => le10_plate(6, 12, 4).unwrap(),
                "LE11" => le11_body(4, 8, 16).unwrap(),
                _ => lame_ring(6).unwrap(),
            };
            let mut best = 0usize;
            let mut bd = Scalar::MAX;
            for c in 0..b.mesh.n_cells {
                let d = (b.mesh.c[c] - b.target).mag();
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            println!("{name}: nearest cell {best} at {bd:e}, stencil {:?}", b.stencil);
            assert!(
                b.stencil.contains(&best),
                "{name}: nearest cell {best} is not in the stencil"
            );
            let want = if name == "LE1" { 4 } else { 8 };
            assert_eq!(b.stencil.len(), want, "{name} stencil size");
        }
    }
    #[test]
    fn a_stencil_that_does_not_span_the_fit_is_refused() {
        let b = le1_membrane(8, 16).unwrap();
        let v: Vec<Scalar> = b.mesh.c.iter().map(|&p| p.x).collect();
        let e = extrapolate_linear(&b.mesh, &v, &b.stencil, b.target, [true, true, true])
            .unwrap_err();
        assert!(format!("{e:?}").contains("does not span"), "{e:?}");
        let two = corner_stencil([8, 16, 1], [0, 2], [0, 1], [0, 1]);
        let e = extrapolate_linear(&b.mesh, &v, &two, b.target, [true, true, false]).unwrap_err();
        assert!(format!("{e:?}").contains("cannot carry"), "{e:?}");
        assert!(extrapolate_linear(&b.mesh, &v[..3], &b.stencil, b.target, b.fit_dims).is_err());
        println!("refusals: does-not-span, cannot-carry, wrong-length values");
    }
    #[test]
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
    fn the_membrane_material_has_plane_stress_in_plane_law() {
        let m = le1_plane_strain_material();
        println!("E* {:.6e} nu* {:.6}", m.e, m.nu);
        assert!(
            (m.e / (1.0 - m.nu * m.nu) - NAFEMS_E).abs() <= 1e-12 * NAFEMS_E,
            "E*/(1 - nu*^2) = {} against {}",
            m.e / (1.0 - m.nu * m.nu),
            NAFEMS_E
        );
        assert!(
            (m.nu / (1.0 - m.nu) - NAFEMS_NU).abs() <= 1e-12,
            "nu*/(1 - nu*) = {} against {}",
            m.nu / (1.0 - m.nu),
            NAFEMS_NU
        );
    }
    #[test]
    fn the_pressure_is_minus_p_n_on_its_patch_only() {
        let le1 = le1_membrane(8, 16).unwrap();
        let t = patch_pressure(&le1.mesh, "outer", -LE1_P).unwrap();
        for p in le1.mesh.patches.iter() {
            let on = p.name == "outer";
            for bf in p.start..p.start + p.size {
                if !on {
                    assert_eq!(t[bf].mag(), 0.0, "face {bf} of {} must be free", p.name);
                } else {
                    let n = le1.mesh.b_sf[bf] / le1.mesh.b_mag_sf[bf];
                    let tn = t[bf].dot(n);
                    assert!(
                        (tn - LE1_P).abs() <= 1e-5 * LE1_P,
                        "face {bf}: t.n = {tn:e}"
                    );
                    assert!((t[bf] - n * tn).mag() <= 1e-5 * LE1_P, "face {bf} tangential");
                }
            }
        }
        println!("outer faces carry t = +P n_hat to 1e-5, every other face zero");
        let e = patch_pressure(&le1.mesh, "nowhere", 1.0).unwrap_err();
        assert!(format!("{e:?}").contains("nowhere"), "{e:?}");
        let le10 = le10_plate(6, 12, 4).unwrap();
        let rows = le10_bcs(&le10.mesh).unwrap();
        assert_eq!(rows.len(), 7);
        assert!(matches!(rows[1][0], CompBc::Fixed(_)), "outer_mid u_x");
        assert!(matches!(rows[1][1], CompBc::Fixed(_)), "outer_mid u_y");
        assert!(matches!(rows[1][2], CompBc::Fixed(_)), "outer_mid u_z");
    }
}
