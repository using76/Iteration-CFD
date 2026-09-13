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
//! Written from:
//!   S. P. Timoshenko, J. N. Goodier, *Theory of Elasticity*, 3rd ed.,
//!     McGraw-Hill (1970), the thermal-stress chapter's long circular
//!     cylinder - the closed-form radial, hoop and axial stresses
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
//! does not have yet, and is the bonded-solid unit's to add.
//!
//! OpenFOAM and solids4foam are GPL and were not opened; no
//! solid-mechanics solver of any licence was consulted. The fixture and
//! its plane-strain patch statement are ORIGINAL to this file.
//! No GPL-licensed source was consulted.

use crate::blockgen::{self, BlockSpec, GradedAxis};
use crate::error::Result;
use crate::io::polymesh::{build_host_mesh, PolyMeshRaw};
use crate::mesh::HostMesh;
use crate::solid::bc::PatchBcs;
use crate::solid::bc::CompBc;
use crate::{Scalar, Vec3};
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
/// ring with a cyclic pair is the bonded-solid unit's, once the operator
/// couples faces.
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
        y: axis(0.0, FRAC_PI_2, n_theta),
        z: axis(0.0, r_out - r_in, nz),
        patch_name: ["inner", "outer", "cut0", "cut90", "zmin", "zmax"].map(String::from),
        patch_type: ["patch"; 6].map(String::from),
        windows: Vec::new(),
        cyclic: Vec::new(),
    };
    let mut raw = blockgen::raw_mesh(&spec)?;
    for p in raw.points.iter_mut() {
        let cut90 = (p.y - FRAC_PI_2).abs() < 1e-12;
        let cut0 = p.y.abs() < 1e-12;
        *p = Vec3::new(p.x * p.y.cos(), p.x * p.y.sin(), p.z);
        if cut90 {
            p.x = 0.0;
        }
        if cut0 {
            p.y = 0.0;
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The ring is a closed quarter ring: it closes to round-off, every
    /// cell has volume, it is orthogonal to a hundredth of a degree, the
    /// total volume is the quarter annulus's times the length, and the six
    /// patches carry the sizes the three index directions dictate.
    #[test]
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
        let want = std::f64::consts::FRAC_PI_4 * (1.0 - 0.25) * a.length;
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
}
