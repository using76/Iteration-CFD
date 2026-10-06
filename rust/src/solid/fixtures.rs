// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Solid-mechanics fixtures, host-only: the quarter-annulus ring the
//! displacement solve runs on, and the thick-walled cylinder's closed forms
//! the stress readout is held against. Nothing here launches a kernel,
//! touches a `Gpu`, or iterates anything.
//!
//! The end-loaded cantilever of Timoshenko & Goodier ch. 3 - the beam, its
//! closed form in plane strain, the per-face load and clamp, and the error
//! measures of Gate 95-A (§109.6) - is here too, copied from the private
//! fixture `src/solid/prototype.rs` measured the segregated loop on.
//!
//! Written from:
//!   S. P. Timoshenko, J. N. Goodier, *Theory of Elasticity*, 3rd ed.,
//!     McGraw-Hill (1970), the thermal-stress chapter's long circular
//!     cylinder - the closed-form radial, hoop and axial stresses
//!   S. P. Timoshenko, J. N. Goodier, *Theory of Elasticity*, 3rd ed.,
//!     McGraw-Hill (1970) ch. 3 - the cantilever loaded at its free end
//!   B. A. Boley, J. H. Weiner, *Theory of Thermal Stresses*, Wiley (1960),
//!     ch. 9 - the same solution, as
//!     `docs/10-fsi-solid-mesh-plan.md` section I cites the two of them
//!   this crate's `src/walldistance.rs` quarter annulus - the point map
//!     `(i, j, k) -> (r cos th, r sin th, z)` whose positive Jacobian
//!     carries the polyMesh winding and the outward normals through
//!     unchanged, and whose two radial cuts are the symmetry planes of
//!     every axisymmetric field
//!
//! The ring is a QUARTER of the annulus, cut at `theta = 0` and
//! `theta = pi/2` and closed with symmetry planes; a full ring with a
//! theta cyclic pair needs a coupled-face path the displacement operator
//! does not have yet, and is not built here - it belongs with the region
//! and interface work, which is where the coupled-face path is.
//!
//! OpenFOAM and solids4foam are GPL and were not opened; no
//! solid-mechanics solver of any licence was consulted. The fixture and
//! its plane-strain patch statement are ORIGINAL to this file.
//! No GPL-licensed source was consulted.

use crate::blockgen::{self, BlockSpec, GradedAxis};
use crate::error::{Error, Result};
use crate::io::polymesh::{build_host_mesh, PolyMeshRaw};
use crate::mesh::HostMesh;
use crate::solid::bc::PatchBcs;
use crate::solid::bc::CompBc;
use crate::solid::Material;
use crate::solid::materials::Bonds;
use crate::types::{to_dvec3, to_vec3};
use crate::{Label, Scalar, Tensor, Vec3};
use std::f64::consts::FRAC_PI_2;

/// The quarter ring and the raw mesh it came from: `r in [r_in, r_out]`,
/// `theta in [0, pi/2]`, `z in [0, r_out - r_in]`. The `raw` points are
/// what [`annulus`] mapped, so the point-valued readers (the point
/// displacement among them) can run on the same case the solve ran on.
pub struct Annulus {
    pub mesh: HostMesh,
    pub raw: PolyMeshRaw,
    pub r_in: Scalar,
    pub r_out: Scalar,
    pub length: Scalar,
}

/// A QUARTER ring, `nr x n_theta x nz` cells, patches in slot order
/// `inner outer cut0 cut90 zmin zmax`, all six of type `patch`.
///
/// A quarter ring with symmetry cuts, not the full ring: the two radial
/// cuts are the symmetry planes of every axisymmetric field - the same
/// per-component `Fixed(0)` normal + `Traction(0)` tangential statement
/// the displacement operator's boundary table already carries - and are
/// axis-aligned (`y = 0`, `x = 0`) after the point map below. The full
/// ring with a cyclic pair is not built here and belongs with the region
/// and interface work, which is where the coupled-face path is.
///
/// The block is built in `(r, theta, z)` coordinates and every point is
/// then mapped `(x, y, z) -> (x cos y, x sin y, z)` - a map with a
/// positive Jacobian everywhere in `r > 0`, so the winding and the
/// outward normals survive it unchanged. The points of the `theta = pi/2`
/// plane are forced onto `x = 0` exactly and those of `theta = 0` onto
/// `y = 0` exactly, so that the two symmetry planes are the coordinate
/// planes the patch statement assumes.
pub fn annulus(nr: usize, n_theta: usize, nz: usize, r_in: Scalar, r_out: Scalar) -> Result<Annulus> {
    let axis = |lo, hi, n| GradedAxis { lo, hi, n, expansion: 1.0, two_sided: false };
    let spec = BlockSpec {
        x: axis(r_in, r_out, nr),
        y: axis(0.0, FRAC_PI_2 as Scalar, n_theta),
        z: axis(0.0, r_out - r_in, nz),
        patch_name: ["inner", "outer", "cut0", "cut90", "zmin", "zmax"].map(String::from),
        patch_type: ["patch"; 6].map(String::from),
        windows: Vec::new(),
        cyclic: Vec::new(),
    };
    let mut raw = blockgen::raw_mesh(&spec)?;
    for p in raw.points.iter_mut() {
        let q = to_vec3(*p);
        let cut90 = (q.y - FRAC_PI_2 as Scalar).abs() < 1e-12;
        let cut0 = q.y.abs() < 1e-12;
        let mut q = Vec3::new(q.x * q.y.cos(), q.x * q.y.sin(), q.z);
        if cut90 {
            q.x = 0.0;
        }
        if cut0 {
            q.y = 0.0;
        }
        *p = to_dvec3(q);
    }
    let mesh = build_host_mesh(&raw)?;
    Ok(Annulus { mesh, raw, r_in, r_out, length: r_out - r_in })
}

/// Plane strain on the quarter ring, per component and per patch, in slot
/// order `inner outer cut0 cut90 zmin zmax`: inner and outer faces free
/// (`Traction(0)`), `cut0` (the plane `y = 0`) fixes `u_y`, `cut90` (the
/// plane `x = 0`) fixes `u_x`, both z ends fix `u_z`, every other
/// component `Traction(0)`. One `Fixed` per displacement component - the
/// rigid modes are out and no strain is imposed: on a symmetry plane of
/// the exact solution the normal displacement is zero anyway, so the
/// statement constrains nothing the solution does not already satisfy.
pub fn quarter_annulus_plane_strain_bcs() -> PatchBcs {
    let mut b = [[CompBc::Traction(0.0); 3]; 6];
    b[2][1] = CompBc::Fixed(0.0); // cut0, y = 0: u_y = 0
    b[3][0] = CompBc::Fixed(0.0); // cut90, x = 0: u_x = 0
    b[4][2] = CompBc::Fixed(0.0); // zmin: u_z = 0
    b[5][2] = CompBc::Fixed(0.0); // zmax: u_z = 0
    b
}

/// The steady log profile between `t_in` at `r_in` and `t_out` at `r_out`,
/// uniform conductivity, no heat source:
///
/// ```text
/// T(r) = t_out + (t_in - t_out) ln(r_out/r) / ln(r_out/r_in)
/// ```
///
/// `ln(r_out/r)` and `ln(r_out/r_in)` are written with the same expression
/// shape, so at `r = r_in` the ratio is 1.0 bitwise and `T(r_in)` is
/// `t_in` to the last bit.
#[inline]
pub fn thick_cylinder_temperature(r: Scalar, r_in: Scalar, r_out: Scalar, t_in: Scalar, t_out: Scalar) -> Scalar {
    t_out + (t_in - t_out) * (r_out / r).ln() / (r_out / r_in).ln()
}

/// `(sigma_r, sigma_theta, sigma_z)` of the thick-walled cylinder under a
/// steady radial temperature, plane strain, stress-free reference
/// `T_ref = t_out` (Timoshenko & Goodier, the thermal-stress chapter's
/// long circular cylinder; Boley & Weiner ch. 9):
///
/// ```text
/// k        = ln(r_out/r_in)
/// C0       = alpha E d_t_inner / (2 (1 - nu) k)
/// sigma_r  = C0 [ -ln(r_out/r) - (r_in^2/(r_out^2-r_in^2)) (1 - r_out^2/r^2) k ]
/// sigma_t  = C0 [ 1 - ln(r_out/r) - (r_in^2/(r_out^2-r_in^2)) (1 + r_out^2/r^2) k ]
/// sigma_z  = nu (sigma_r + sigma_t) - alpha E d_t_inner ln(r_out/r) / k
/// ```
///
/// These are the log-profile evaluation of the integral forms; the
/// equilibrium test below is what says the transcription is the closed
/// form and not a neighbour of it. `d_t_inner = t_in - t_ref`.
#[inline]
pub fn thick_cylinder_stress(r: Scalar, r_in: Scalar, r_out: Scalar, e: Scalar, nu: Scalar,
                             alpha: Scalar, d_t_inner: Scalar) -> (Scalar, Scalar, Scalar) {
    let k = (r_out / r_in).ln();
    let c0 = alpha * e * d_t_inner / (2.0 * (1.0 - nu) * k);
    let a2 = r_in * r_in;
    let b2 = r_out * r_out;
    let lnr = (r_out / r).ln();
    let s_r = c0 * (-lnr - (a2 / (b2 - a2)) * (1.0 - b2 / (r * r)) * k);
    let s_t = c0 * (1.0 - lnr - (a2 / (b2 - a2)) * (1.0 + b2 / (r * r)) * k);
    let s_z = nu * (s_r + s_t) - alpha * e * d_t_inner * lnr / k;
    (s_r, s_t, s_z)
}

/// `S`, the largest |component| of the closed form over 2 000 radii
/// `r_in + (r_out - r_in)(i + 0.5)/2000` - the scale the gate's errors are
/// measured against, exact and not estimated.
pub fn thick_cylinder_scale(r_in: Scalar, r_out: Scalar, e: Scalar, nu: Scalar, alpha: Scalar,
                            d_t_inner: Scalar) -> Scalar {
    let mut s = 0.0 as Scalar;
    for i in 0..2000 {
        let r = r_in + (r_out - r_in) * (i as Scalar + 0.5) / 2000.0;
        let (s_r, s_t, s_z) = thick_cylinder_stress(r, r_in, r_out, e, nu, alpha, d_t_inner);
        s = s.max(s_r.abs()).max(s_t.abs()).max(s_z.abs());
    }
    s
}

/// The von Mises equivalent of a DIAGONAL stress, in the difference form
/// with no shear terms - bitwise what `stress::von_mises` computes on the
/// same diagonal, whose identity that file's split test checks.
#[inline]
fn vm_of_diag(s_r: Scalar, s_t: Scalar, s_z: Scalar) -> Scalar {
    let d_rt = s_r - s_t;
    let d_tz = s_t - s_z;
    let d_zr = s_z - s_r;
    (0.5 * (d_rt * d_rt + d_tz * d_tz + d_zr * d_zr)).sqrt()
}

/// The volume mean of von Mises over the full ring,
/// `F = (2/(r_out^2 - r_in^2)) INT vm(r) r dr`, composite Simpson with
/// 20 000 intervals - the error is far below 1e-10 relative, well under
/// any tolerance the gate reads the number against.
pub fn thick_cylinder_mean_von_mises(r_in: Scalar, r_out: Scalar, e: Scalar, nu: Scalar,
                                     alpha: Scalar, d_t_inner: Scalar) -> Scalar {
    let n = 20_000;
    let h = (r_out - r_in) / n as Scalar;
    let f = |r: Scalar| {
        let (s_r, s_t, s_z) = thick_cylinder_stress(r, r_in, r_out, e, nu, alpha, d_t_inner);
        vm_of_diag(s_r, s_t, s_z) * r
    };
    let mut sum = f(r_in) + f(r_out);
    for i in 1..n {
        let r = r_in + h * i as Scalar;
        sum += f(r) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * h / 3.0 * 2.0 / (r_out * r_out - r_in * r_in)
}

/// A cantilever strip `x in [0, l]`, `y in [0, h]`, ONE cell thick in `z`
/// (`z in [0, h/ny]`), square cells - `nx/l == ny/h` is the caller's job
/// and is not checked here. All six patches are real `patch`es, so the
/// boundary statement is the caller's to make. Returns the mesh, the cells
/// below the bond line (`c.y < a1`, exactly `nx` per cell row, the lower
/// material) and the cells above it. An `a1` that does not lie on a cell
/// face would cut cells in half; it is refused by name.
pub fn bimetal_strip(
    nx: usize,
    ny: usize,
    l: Scalar,
    h: Scalar,
    a1: Scalar,
) -> Result<(HostMesh, Vec<Label>, Vec<Label>)> {
    let frac = a1 / h * ny as Scalar;
    if (frac - frac.round()).abs() > 1e-9 {
        return Err(Error::Config(format!(
            "bimetal_strip: the bond height a1 = {a1} does not lie on a cell \
             face of ny = {ny} cells over h = {h} (a1/h*ny = {frac}); the two \
             materials would cut through cells instead of meeting at a face"
        )));
    }
    let axis = |lo, hi, n| GradedAxis { lo, hi, n, expansion: 1.0, two_sided: false };
    let spec = BlockSpec {
        x: axis(0.0, l, nx),
        y: axis(0.0, h, ny),
        z: axis(0.0, h / ny as Scalar, 1),
        patch_name: ["xMin", "xMax", "yMin", "yMax", "zMin", "zMax"].map(String::from),
        patch_type: ["patch"; 6].map(String::from),
        windows: Vec::new(),
        cyclic: Vec::new(),
    };
    let raw = blockgen::raw_mesh(&spec)?;
    let mesh = build_host_mesh(&raw)?;
    let lower: Vec<Label> =
        (0..mesh.n_cells as Label).filter(|&c| mesh.c[c as usize].y < a1).collect();
    let upper: Vec<Label> =
        (0..mesh.n_cells as Label).filter(|&c| mesh.c[c as usize].y > a1).collect();
    Ok((mesh, lower, upper))
}

/// The curvature of the deformed strip, read off the cell gradients along
/// the column `i = nx/2` - `l/2` itself is a cell FACE, so the column is
/// the one just past it: every cell with
/// `|c.x - (l/2 + 0.5 l/nx)| < 0.25 l/nx`, exactly `ny` cells for an even
/// `nx`. Along that column the axial strain `eps_xx = grad[c].xx` is
/// least-squares fitted as `eps_xx = e0 + kappa y_c`; the slope is the
/// curvature, the intercept the strain at the mid-plane. Returns
/// `(kappa, e0)`.
pub fn strip_curvature(m: &HostMesh, grad: &[Tensor], l: Scalar, nx: usize) -> (Scalar, Scalar) {
    let dx = l / nx as Scalar;
    let mid = l * 0.5 + 0.5 * dx;
    let (mut sy, mut syy, mut sg, mut syg, mut n) =
        (0.0, 0.0, 0.0, 0.0, 0.0 as Scalar);
    for c in 0..m.n_cells {
        if (m.c[c].x - mid).abs() < 0.25 * dx {
            let (y, g) = (m.c[c].y, grad[c].xx);
            n += 1.0;
            sy += y;
            syy += y * y;
            sg += g;
            syg += y * g;
        }
    }
    let nf = n;
    let kappa = (nf * syg - sy * sg) / (nf * syy - sy * sy);
    let e0 = (sg - kappa * sy) / nf;
    (kappa, e0)
}

/// The interface stress a bond treatment leaves behind, as a ratio
/// (SPEC-LIT (S95.20)): the largest `|sigma_yy|` over the BOND cells - the
/// owner and the neighbour of every bond face - with `|c.x - l/2| <= h`,
/// over the largest `|sigma_xx|` over ALL cells in the same window. The
/// exact state has `sigma_yy = 0` in the bond cells, so the ratio measures
/// exactly the stress the treatment invents there; the window sits `2 h`
/// clear of each end, where the clamp's and the free end's own reaction
/// has decayed to the Saint-Venant floor.
pub fn bond_stress_ratio(
    m: &HostMesh,
    sigma: &[Tensor],
    bonds: &Bonds,
    l: Scalar,
    h: Scalar,
) -> Scalar {
    let in_window = |x: Scalar| (x - l * 0.5).abs() <= h;
    let mut s_yy = 0.0 as Scalar;
    for &f in &bonds.bond_face {
        let fu = f as usize;
        for c in [m.owner[fu] as usize, m.neighbour[fu] as usize] {
            if in_window(m.c[c].x) {
                s_yy = s_yy.max(sigma[c].yy.abs());
            }
        }
    }
    let mut s_xx = 0.0 as Scalar;
    for c in 0..m.n_cells {
        if in_window(m.c[c].x) {
            s_xx = s_xx.max(sigma[c].xx.abs());
        }
    }
    s_yy / s_xx
}

/// Gate 95-A's material and load: `E`, `nu`, the end load `P`, the
/// half-depth `c`, and `alpha` (isothermal: unused).
pub const CANTILEVER_E: Scalar = 200.0e9;
pub const CANTILEVER_NU: Scalar = 0.3;
pub const CANTILEVER_P: Scalar = 1.0e5;
pub const CANTILEVER_C: Scalar = 0.1;

/// Gate 95-A's material: steel, the load of the gate, isothermal.
pub fn cantilever_material() -> Material {
    Material { e: CANTILEVER_E, nu: CANTILEVER_NU, alpha: 1.2e-5 }
}

/// A beam of span `l` along `x`, depth `2c` in `y` and ONE cell through
/// the thickness in `z`, whose cells are cubes when `n_x/l == n_y/(2c)`.
///
/// Six real patches, `blockgen`'s `-x +x -y +y -z +z` slot order: the
/// `-z`/`+z` pair is named rather than left empty, because an empty face
/// contributes to no surface integral and would delete two of the six
/// tractions of a three-dimensional stress problem.
pub fn cantilever_mesh(n_x: usize, n_y: usize, l: Scalar, c: Scalar) -> Result<HostMesh> {
    let axis = |lo: Scalar, hi: Scalar, n: usize| GradedAxis {
        lo,
        hi,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    blockgen::build_mesh(&BlockSpec {
        x: axis(0.0, l, n_x),
        y: axis(-c, c, n_y),
        z: axis(0.0, 2.0 * c / n_y as Scalar, 1),
        patch_type: ["patch", "patch", "patch", "patch", "patch", "patch"].map(String::from),
        ..Default::default()
    })
}

/// Timoshenko & Goodier, *Theory of Elasticity*, 3rd ed. (1970) ch. 3:
/// the cantilever loaded at its free end. Origin at the FREE end on the
/// neutral axis, `x` toward the built-in end at `x = l`, `y` in
/// `[-c, c]`, unit thickness, `I = 2c^3/3`, load `P` in `+y` at `x = 0`.
/// Plane strain through `E' = E/(1 - nu^2)`, `nu' = nu/(1 - nu)`, `G`
/// unchanged (ch. 2). The rigid-body constant is fixed by "a vertical
/// element of the cross-section at the built-in end remains vertical".
pub struct CantileverExact {
    pub p: Scalar,
    pub l: Scalar,
    pub c: Scalar,
    pub i: Scalar,
    pub e_p: Scalar,
    pub nu_p: Scalar,
    pub g: Scalar,
}

impl CantileverExact {
    pub fn new(e: Scalar, nu: Scalar, l: Scalar, c: Scalar, p: Scalar) -> Self {
        Self {
            p,
            l,
            c,
            i: 2.0 * c * c * c / 3.0,
            e_p: e / (1.0 - nu * nu),
            nu_p: nu / (1.0 - nu),
            g: e / (2.0 * (1.0 + nu)),
        }
    }

    /// `-P x y / I` - the bending stress at `(x, y)`, the stress the
    /// cell-centre error measure reads.
    pub fn sigma_xx(&self, x: Scalar, y: Scalar) -> Scalar {
        -self.p * x * y / self.i
    }

    /// `-P (c^2 - y^2)/(2I)`. Its depth integral is `-P`, so the parabolic
    /// shear the free end carries IS the end load and nothing is left to a
    /// Saint-Venant end effect.
    pub fn sigma_xy(&self, y: Scalar) -> Scalar {
        -self.p * (self.c * self.c - y * y) / (2.0 * self.i)
    }

    pub fn u(&self, x: Scalar, y: Scalar) -> Scalar {
        -self.p * x * x * y / (2.0 * self.e_p * self.i)
            - self.nu_p * self.p * y * y * y / (6.0 * self.e_p * self.i)
            + self.p * y * y * y / (6.0 * self.i * self.g)
            + self.p * self.l * self.l * y / (2.0 * self.e_p * self.i)
    }

    pub fn v(&self, x: Scalar, y: Scalar) -> Scalar {
        self.nu_p * self.p * x * y * y / (2.0 * self.e_p * self.i)
            + self.p * x * x * x / (6.0 * self.e_p * self.i)
            - (self.p * self.c * self.c / (2.0 * self.i * self.g)
                + self.p * self.l * self.l / (2.0 * self.e_p * self.i))
                * x
            + self.p * self.l * self.l * self.l / (3.0 * self.e_p * self.i)
            + self.p * self.c * self.c * self.l / (2.0 * self.i * self.g)
    }

    /// `v(0, y)`, which does not depend on `y`.
    pub fn tip(&self) -> Scalar {
        self.p * self.l * self.l * self.l / (3.0 * self.e_p * self.i)
            + self.p * self.c * self.c * self.l / (2.0 * self.i * self.g)
    }

    /// `P l c / I` - the peak bending stress of the gate's load, the scale
    /// both stress errors are read against.
    pub fn stress_scale(&self) -> Scalar {
        self.p * self.l * self.c / self.i
    }
}

/// The end-loaded cantilever, as the displacement operator's boundary
/// statement: `all_free()`, then patch 1 (`+x`, the built-in end)
/// `[Fixed(0.0); 3]`, and the z-component of patches 4 and 5 (`-z`, `+z`)
/// `Fixed(0.0)` - the two slab faces are symmetry planes.
pub fn cantilever_bcs() -> PatchBcs {
    let mut b = crate::solid::bc::all_free();
    b[1] = [CompBc::Fixed(0.0); 3];
    b[4][2] = CompBc::Fixed(0.0);
    b[5][2] = CompBc::Fixed(0.0);
    b
}

/// Per boundary face, in the flattened boundary-face order: (traction,
/// fixed value). Patch 0 (`-x`, the free end) carries the closed form's
/// parabolic shear `(0, -sigma_xy(y), 0)`; patch 1 carries its own
/// displacement `(u(l, y), v(l, y), 0)`; every other entry zero, which is
/// what the symmetry planes' `Fixed` components need. For
/// `SolidBcs::set_traction_values` / `set_fixed_values`.
pub fn cantilever_face_values(m: &HostMesh, exact: &CantileverExact) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut tr = vec![Vec3::ZERO; m.n_boundary_faces];
    let mut fv = vec![Vec3::ZERO; m.n_boundary_faces];
    for bf in 0..m.n_boundary_faces {
        let y = m.b_cf[bf].y;
        match m.b_patch[bf] as usize {
            0 => tr[bf] = Vec3::new(0.0, -exact.sigma_xy(y), 0.0),
            1 => fv[bf] = Vec3::new(exact.u(exact.l, y), exact.v(exact.l, y), 0.0),
            _ => {}
        }
    }
    (tr, fv)
}

/// The `|Sf|`-weighted mean of `ub.y` over patch 0 - the prototype's
/// `tip_deflection` on a boundary array, the discrete reading of the
/// closed form's `v(0, y)`.
pub fn cantilever_tip_deflection(m: &HostMesh, ub: &[Vec3]) -> Scalar {
    let (mut w, mut v) = (0.0 as Scalar, 0.0 as Scalar);
    for bf in 0..m.n_boundary_faces {
        if m.b_patch[bf] as usize == 0 {
            w += m.b_mag_sf[bf];
            v += m.b_mag_sf[bf] * ub[bf].y;
        }
    }
    v / w
}

/// Gate 95-A's three error measures: the relative inf-norm of the
/// displacement error, and the volume-weighted rms of each in-plane stress
/// error, both stress norms read against [`CantileverExact::stress_scale`].
pub struct CantileverErrors {
    pub e_u: Scalar,
    pub e_xx: Scalar,
    pub e_xy: Scalar,
}

/// `e_u  = max_c |u_c - (u(x_c,y_c), v(x_c,y_c), 0)| / max_c |(u,v,0)|`,
/// `e_xx = sqrt( sum_c V_c (sigma_c.xx - sigma_xx(x_c,y_c))^2 / sum_c V_c ) / stress_scale()`,
/// `e_xy = sqrt( sum_c V_c (sigma_c.xy - sigma_xy(y_c))^2   / sum_c V_c ) / stress_scale()`.
pub fn cantilever_errors(
    m: &HostMesh,
    u: &[Vec3],
    sigma: &[Tensor],
    exact: &CantileverExact,
) -> CantileverErrors {
    let mut num_u = 0.0 as Scalar;
    let mut den_u = 0.0 as Scalar;
    let mut num_xx = 0.0 as Scalar;
    let mut num_xy = 0.0 as Scalar;
    let mut vol = 0.0 as Scalar;
    for c in 0..m.n_cells {
        let x = m.c[c];
        let want = Vec3::new(exact.u(x.x, x.y), exact.v(x.x, x.y), 0.0);
        num_u = num_u.max((u[c] - want).mag());
        den_u = den_u.max(want.mag());
        let d_xx = sigma[c].xx - exact.sigma_xx(x.x, x.y);
        let d_xy = sigma[c].xy - exact.sigma_xy(x.y);
        num_xx += m.v[c] * d_xx * d_xx;
        num_xy += m.v[c] * d_xy * d_xy;
        vol += m.v[c];
    }
    let s = exact.stress_scale();
    CantileverErrors {
        e_u: num_u / den_u.max(crate::SCALAR_FLOOR),
        e_xx: (num_xx / vol).sqrt() / s,
        e_xy: (num_xy / vol).sqrt() / s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cantilever closed form is plane-strain Hooke-consistent at
    /// every sampled point (central differences of the cubics are exact),
    /// is in equilibrium, `tip()` is `v(0, y)` at any `y`, and the gate's
    /// smallest mesh is 160 cubic cells over six real patches. The mesh is
    /// the 2.5:1 ratio's own (`l = 0.5`, `n_x = (l * n_y / 2c)`), the only
    /// span whose `20 x 8` cells are the cubes the volume check names.
    #[test]
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
    fn the_cantilever_closed_form_is_hooke_consistent_and_in_equilibrium() {
        let exact = CantileverExact::new(CANTILEVER_E, CANTILEVER_NU, 1.0, CANTILEVER_C, CANTILEVER_P);
        let h = 1.0e-6;
        let s = exact.stress_scale();
        let mut hooke_xx = 0.0 as Scalar;
        let mut hooke_xy = 0.0 as Scalar;
        let mut equilibrium = 0.0 as Scalar;
        for k in 0..20 {
            let x = (k as Scalar + 0.5) / 20.0;
            let y = CANTILEVER_C * 0.9 * (k as Scalar).sin();
            let eps_x = (exact.u(x + h, y) - exact.u(x - h, y)) / (2.0 * h);
            let eps_y = (exact.v(x, y + h) - exact.v(x, y - h)) / (2.0 * h);
            let gamma = (exact.u(x, y + h) - exact.u(x, y - h)) / (2.0 * h)
                + (exact.v(x + h, y) - exact.v(x - h, y)) / (2.0 * h);
            let law = exact.e_p / (1.0 - exact.nu_p * exact.nu_p);
            hooke_xx = hooke_xx
                .max((law * (eps_x + exact.nu_p * eps_y) - exact.sigma_xx(x, y)).abs());
            hooke_xy = hooke_xy.max((exact.g * gamma - exact.sigma_xy(y)).abs());
            let dsxx_dx = (exact.sigma_xx(x + h, y) - exact.sigma_xx(x - h, y)) / (2.0 * h);
            let dsxy_dy = (exact.sigma_xy(y + h) - exact.sigma_xy(y - h)) / (2.0 * h);
            equilibrium = equilibrium.max((dsxx_dx + dsxy_dy).abs());
        }
        for y in [-CANTILEVER_C, 0.0, CANTILEVER_C / 2.0] {
            let d = (exact.v(0.0, y) - exact.tip()).abs();
            assert!(d <= 1.0e-12 * exact.tip().abs(), "tip at y = {y}: {d:e}");
        }
        println!(
            "cantilever closed form: |Hooke sigma_xx| = {hooke_xx:e}  \
             |Hooke sigma_xy| = {hooke_xy:e}  |div sigma| = {equilibrium:e}  (scale {s:e})"
        );
        assert!(hooke_xx <= 1.0e-6 * s, "Hooke sigma_xx: {hooke_xx:e} against {s:e}");
        assert!(hooke_xy <= 1.0e-6 * s, "Hooke sigma_xy: {hooke_xy:e} against {s:e}");
        assert!(
            equilibrium <= 1.0e-6 * s / 1.0,
            "equilibrium: {equilibrium:e} against {:#e}",
            1.0e-6 * s / 1.0
        );
        let m = cantilever_mesh(20, 8, 0.5, 0.1).expect("cantilever mesh");
        assert_eq!(m.n_cells, 160);
        let mut patches: Vec<Label> = m.b_patch.clone();
        patches.sort();
        patches.dedup();
        assert_eq!(patches.len(), 6, "six real patches, got {patches:?}");
        let want = (0.2 as Scalar / 8.0).powi(3);
        for c in 0..m.n_cells {
            let rel = ((m.v[c] - want) / want).abs();
            assert!(rel <= 1.0e-12, "cell {c}: volume {:+e} against {want:e}", m.v[c]);
        }
    }

    /// The ring is a closed quarter ring: it closes to round-off, every
    /// cell has volume, it is orthogonal to a hundredth of a degree, the
    /// total volume is the quarter annulus's times the length, and the six
    /// patches carry the sizes the three index directions dictate.
    #[test]
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
    fn the_annulus_is_a_closed_quarter_ring() {
        let a = annulus(6, 12, 2, 0.5, 1.0).expect("annulus");
        let r = a.mesh.check();
        println!(
            "annulus: closure={:.3e} minV={:.3e} nonorth={:.3e}deg V={:.6}",
            r.max_closure_error, r.min_volume, r.max_non_orth_deg, r.total_volume
        );
        assert!(r.max_closure_error <= 1e-12, "closure {:.3e}", r.max_closure_error);
        assert!(r.min_volume > 0.0, "negative cell volume");
        assert!(
            r.max_non_orth_deg <= 0.01,
            "non-orthogonality {} deg",
            r.max_non_orth_deg
        );
        let want = std::f64::consts::FRAC_PI_4 as Scalar * (1.0 - 0.25) * a.length;
        let dv = (r.total_volume - want).abs() / want;
        println!("annulus: V={:.6} exact={:.6} rel={:.3e}", r.total_volume, want, dv);
        assert!(dv <= 0.01, "total volume off by {dv:e} relative");
        let got: Vec<(&str, usize, &str)> = a
            .mesh
            .patches
            .iter()
            .map(|p| (p.name.as_str(), p.size, p.type_name.as_str()))
            .collect();
        assert_eq!(
            got,
            [
                ("inner", 24, "patch"),
                ("outer", 24, "patch"),
                ("cut0", 12, "patch"),
                ("cut90", 12, "patch"),
                ("zmin", 72, "patch"),
                ("zmax", 72, "patch"),
            ]
        );
    }

    /// The closed form is in equilibrium at the gate numbers - free inner
    /// and outer walls, radial equilibrium, and the end temperatures the
    /// profile is named for. A form that fails these is mis-transcribed,
    /// whatever the book says.
    #[test]
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
    fn the_thick_cylinder_closed_form_is_in_equilibrium() {
        let (ri, ro) = (0.5, 1.0);
        let (e, nu, alpha, dt_i) = (200.0e9, 0.3, 1.2e-5, 100.0);
        let s = thick_cylinder_scale(ri, ro, e, nu, alpha, dt_i);
        println!("thick cylinder: S = {s:.6e} Pa");

        // The two radial walls are traction-free.
        let (ra, _, _) = thick_cylinder_stress(ri, ri, ro, e, nu, alpha, dt_i);
        let (rb, _, _) = thick_cylinder_stress(ro, ri, ro, e, nu, alpha, dt_i);
        println!("sigma_r(a) = {ra:.3e}  sigma_r(b) = {rb:.3e}  (S = {s:.3e})");
        assert!(ra.abs() <= 1e-12 * s, "sigma_r(a) = {ra:e}");
        assert!(rb.abs() <= 1e-12 * s, "sigma_r(b) = {rb:e}");

        // d sigma_r/dr = (sigma_theta - sigma_r)/r, by central differences.
        let dr = 1.0e-4 * (ro - ri);
        let mut worst = 0.0 as Scalar;
        for i in 0..50 {
            let r = ri + (ro - ri) * (i as Scalar + 0.5) / 50.0;
            let (sr, st, _) = thick_cylinder_stress(r, ri, ro, e, nu, alpha, dt_i);
            let (sp, _, _) = thick_cylinder_stress(r + dr, ri, ro, e, nu, alpha, dt_i);
            let (sm, _, _) = thick_cylinder_stress(r - dr, ri, ro, e, nu, alpha, dt_i);
            let residual = ((sp - sm) / (2.0 * dr) - (st - sr) / r).abs();
            worst = worst.max(residual);
        }
        println!("equilibrium: max |d sr/dr - (st - sr)/r| = {worst:.3e} Pa (bound 1e-6 S/a = {:.3e})", 1e-6 * s / ri);
        assert!(
            worst <= 1e-6 * s / ri,
            "radial equilibrium residual {worst:e} Pa at the central-difference step"
        );

        // The end temperatures, exactly.
        assert_eq!(thick_cylinder_temperature(ri, ri, ro, 400.0, 300.0), 400.0);
        assert_eq!(thick_cylinder_temperature(ro, ri, ro, 400.0, 300.0), 300.0);
    }

    /// The strip splits exactly on the cell face at `a1`: 384 cells in two
    /// halves of 192, 48 internal faces joining the two materials, all six
    /// patches real `patch`es, the mesh closed with volume everywhere - and
    /// an `a1` between two faces is refused by name.
    #[test]
    #[cfg_attr(feature = "single", ignore = "fails at f32: SPEC-LIT 112.3")]
    fn the_bimetal_strip_splits_on_a_face() {
        let (m, low, high) = bimetal_strip(48, 8, 0.06, 0.01, 0.005).expect("strip");
        assert_eq!(low.len(), 192, "cells below the bond line");
        assert_eq!(high.len(), 192, "cells above the bond line");
        let r = m.check();
        println!(
            "strip: closure={:.3e} minV={:.3e} nonorth={:.3e}deg",
            r.max_closure_error, r.min_volume, r.max_non_orth_deg
        );
        assert!(r.max_closure_error <= 1e-12, "closure {:.3e}", r.max_closure_error);
        assert!(r.min_volume > 0.0, "negative cell volume");
        let got: Vec<(&str, usize, &str)> = m
            .patches
            .iter()
            .map(|p| (p.name.as_str(), p.size, p.type_name.as_str()))
            .collect();
        for (name, size, ty) in &got {
            println!("  {name} {size} {ty}");
        }
        assert!(got.iter().all(|(_, _, ty)| *ty == "patch"), "every patch a patch");
        let mut is_low = vec![false; m.n_cells];
        for &c in &low {
            is_low[c as usize] = true;
        }
        let bonds = (0..m.n_internal_faces)
            .filter(|&f| is_low[m.owner[f] as usize] != is_low[m.neighbour[f] as usize])
            .count();
        println!("strip: bond faces = {bonds}");
        assert_eq!(bonds, 48, "one bond face per x column");

        let err = bimetal_strip(48, 8, 0.06, 0.01, 0.0047);
        let msg = format!("{err:?}");
        println!("a1 = 0.0047: {msg}");
        assert!(msg.contains("0.0047"), "the refusal names the height: {msg}");
        assert!(msg.contains("cell face"), "the refusal says what is wrong: {msg}");
    }
}
