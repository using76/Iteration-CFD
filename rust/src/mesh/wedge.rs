// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The wedge derivation of SPEC-LIT section 119.1.
//!
//! An axisymmetric case is solved on a wedge of the full revolve: two planar
//! sides - conventionally `back` and `front` - that meet along the axis of
//! symmetry, every other boundary lying on the solid of revolution. The CASE
//! declares only the two patch names; this module DERIVES the wedge from the
//! geometry the mesh already carries - the patch pair, the axis line, the
//! full angle, the mid-plane - and refuses, by name, a mesh whose two wedge
//! patches are not a matched, planar, mirror-symmetric pair. All arithmetic
//! is f64 through `HostMesh::geom64`, in both builds; under `single` that is
//! the widened shadow (SPEC-LIT section 118.2), so the same derivation sees
//! the same f64 geometry whichever build runs.
//!
//! Until BC-04 every solver path treats a `Wedge` patch exactly as it treats
//! `Symmetry`; the kind and this record exist so the mesh reader can already
//! gate the mesh and the mesh report can already print it.
//!
//! Provenance: ORIGINAL - elementary vector geometry (plane normals, the
//! line where two planes meet, a mirror reflection). No GPL-licensed source
//! was consulted.

use crate::error::{Error, Result};
use crate::mesh::{Geom64, HostMesh, PatchKind};
use crate::types::DVec3;

/// Relative tolerance of every wedge check (planarity, pairing, mirror
/// symmetry).
pub const WEDGE_TOL: f64 = 1e-8;
/// Below this |n_back x n_front| the two planes are parallel: not a wedge.
pub const WEDGE_MIN_SIN: f64 = 1e-6;

/// The wedge a mesh's two `wedge` patches define - SPEC-LIT section 119.1.
#[derive(Debug, Clone, PartialEq)]
pub struct WedgeInfo {
    /// patch index of the wedge patch listed FIRST
    pub back: usize,
    /// patch index of the wedge patch listed second
    pub front: usize,
    /// faces per side (both sides have this many)
    pub n_faces: usize,
    /// unit axis direction
    pub axis: DVec3,
    /// a point on the axis
    pub axis_point: DVec3,
    /// FULL wedge angle, radians
    pub theta: f64,
    /// unit normal of the mid-plane
    pub mid_normal: DVec3,
}

/// The planar patch of section 119.1, in f64: unit outward normal, the
/// area-weighted centroid, the plane offset (`n . x = d`) and the length
/// scale every tolerance is made relative to.
struct PatchPlane {
    n: DVec3,
    c: DVec3,
    d: f64,
    len: f64,
}

impl PatchPlane {
    fn new(g: &Geom64<'_>, pi: &crate::mesh::PatchInfo) -> PatchPlane {
        let mut s = DVec3::ZERO;
        let mut c = DVec3::ZERO;
        let mut area = 0.0;
        for j in 0..pi.size {
            let bf = pi.start + j;
            s += g.b_sf[bf];
            c += g.b_mag_sf[bf] * g.b_cf[bf];
            area += g.b_mag_sf[bf];
        }
        let n = s.normalised();
        let c = c / area;
        PatchPlane { n, c, d: n.dot(c), len: area.sqrt() }
    }
}

/// One face too far off its patch plane - or one face whose normal points
/// away - is a wedge side that is not planar. The measured value and the
/// limit are both reported, in f64.
fn check_planar(g: &Geom64<'_>, pi: &crate::mesh::PatchInfo, pl: &PatchPlane) -> Result<()> {
    for j in 0..pi.size {
        let bf = pi.start + j;
        let dn = (g.b_sf[bf] / g.b_mag_sf[bf] - pl.n).mag();
        if dn > WEDGE_TOL {
            return Err(Error::Mesh(format!(
                "wedge: patch '{}' is not planar: face {j} of the patch has \
                 its normal {dn:.6e} from the patch normal, limit {WEDGE_TOL:.6e}",
                pi.name
            )));
        }
        let off = (g.b_cf[bf] - pl.c).dot(pl.n).abs();
        if off > WEDGE_TOL * pl.len {
            return Err(Error::Mesh(format!(
                "wedge: patch '{}' is not planar: face {j} of the patch sits \
                 {off:.6e} off the patch plane, limit {:.6e}",
                pi.name,
                WEDGE_TOL * pl.len
            )));
        }
    }
    Ok(())
}

/// Derive the wedge of SPEC-LIT section 119.1 from the mesh's `wedge`
/// patches, or refuse the mesh. `Ok(None)` when the mesh has no wedge patch
/// with faces.
pub fn derive_wedge(m: &HostMesh) -> Result<Option<WedgeInfo>> {
    // 1. Collect the wedge patches, in patch order.
    let idx: Vec<usize> = m
        .patches
        .iter()
        .enumerate()
        .filter(|(_, p)| p.kind == PatchKind::Wedge && p.size > 0)
        .map(|(i, _)| i)
        .collect();
    let quoted: Vec<String> =
        idx.iter().map(|&i| format!("'{}'", m.patches[i].name)).collect();
    match idx.len() {
        0 => return Ok(None),
        1 => {
            return Err(Error::Mesh(format!(
                "wedge: patch {} has no partner: a wedge needs exactly two \
                 wedge patches, the mesh lists 1",
                quoted[0]
            )))
        }
        2 => {}
        n => {
            return Err(Error::Mesh(format!(
                "wedge: patches {} have no partner: a wedge needs exactly two \
                 wedge patches, the mesh lists {n}",
                quoted.join(", ")
            )))
        }
    }

    let g = m.geom64();
    let (b, f) = (idx[0], idx[1]);
    let (pb, pf) = (PatchPlane::new(&g, &m.patches[b]), PatchPlane::new(&g, &m.patches[f]));

    // 2. Each side lies in ONE plane.
    check_planar(&g, &m.patches[b], &pb)?;
    check_planar(&g, &m.patches[f], &pf)?;

    // 3. The pair is written face for face, same owner cell each k.
    let (nb, nf) = (m.patches[b].size, m.patches[f].size);
    if nb != nf {
        return Err(Error::Mesh(format!(
            "wedge: '{}' and '{}' are not paired: {nb} faces against {nf}",
            m.patches[b].name, m.patches[f].name
        )));
    }
    for k in 0..nb {
        let (bb, bf) = (m.patches[b].start + k, m.patches[f].start + k);
        if m.b_face_cells[bb] != m.b_face_cells[bf] {
            return Err(Error::Mesh(format!(
                "wedge: '{}' and '{}' are not paired: face {k} of each side \
                 belongs to cells {} and {}",
                m.patches[b].name, m.patches[f].name, m.b_face_cells[bb],
                m.b_face_cells[bf]
            )));
        }
    }

    // 4. Two parallel planes are a slab, not a wedge.
    let u = pb.n.cross(pf.n);
    if u.mag() < WEDGE_MIN_SIN {
        return Err(Error::Mesh(format!(
            "wedge: '{}' and '{}' are parallel planes, not a wedge",
            m.patches[b].name, m.patches[f].name
        )));
    }
    // The axis is `n_front x n_back` (SPEC-LIT section 119.1): back y = -x t,
    // front y = +x t gives +z. `u` below feeds only a |u|^2, where its sign
    // is already gone.
    let axis = pf.n.cross(pb.n).normalised();
    derive_rest(m, &g, b, f, pb, pf, u, axis)
}

/// The axis, mid-plane, angle and mirror test - the geometry that only
/// exists once the pair is known to be two non-parallel planes.
#[allow(clippy::too_many_arguments)]
fn derive_rest(
    m: &HostMesh,
    g: &Geom64<'_>,
    b: usize,
    f: usize,
    pb: PatchPlane,
    pf: PatchPlane,
    u: DVec3,
    axis: DVec3,
) -> Result<Option<WedgeInfo>> {
    let mid_normal = (pf.n - pb.n).normalised();
    let theta = 2.0 * f64::atan2((pb.n + pf.n).mag(), (pf.n - pb.n).mag());

    // The two planes meet in a line; `p` is its point nearest the origin,
    // and the axis point is moved along that line to the mesh's middle.
    let nn = pb.n.dot(pf.n);
    let p = (pb.n * (pb.d - pf.d * nn) + pf.n * (pf.d - pb.d * nn)) / u.mag_sqr();
    let cbar = (pb.c + pf.c) / 2.0;
    let axis_point = p + axis * axis.dot(cbar - p);

    // 5. Front is back reflected in the mid-plane, face for face.
    let mirror = |x: DVec3| x - mid_normal * (2.0 * (x - axis_point).dot(mid_normal));
    let name = format!("'{}' and '{}' are not symmetric about the mid-plane", m.patches[b].name, m.patches[f].name);
    for k in 0..m.patches[b].size {
        let (bb, bf) = (m.patches[b].start + k, m.patches[f].start + k);
        let dm = (mirror(g.b_cf[bb]) - g.b_cf[bf]).mag();
        if dm > WEDGE_TOL * pb.len {
            return Err(Error::Mesh(format!(
                "wedge: {name}: face {k} of each side sits {dm:.6e} apart \
                 after reflection, limit {:.6e}",
                WEDGE_TOL * pb.len
            )));
        }
        let da = (g.b_mag_sf[bb] - g.b_mag_sf[bf]).abs();
        if da > WEDGE_TOL * g.b_mag_sf[bb] {
            return Err(Error::Mesh(format!(
                "wedge: {name}: face {k} of each side has areas \
                 {:.6e} against {:.6e}, limit {:.6e}",
                g.b_mag_sf[bb],
                g.b_mag_sf[bf],
                WEDGE_TOL * g.b_mag_sf[bb]
            )));
        }
    }

    Ok(Some(WedgeInfo {
        back: b,
        front: f,
        n_faces: m.patches[b].size,
        axis,
        axis_point,
        theta,
        mid_normal,
    }))
}

impl WedgeInfo {
    /// One report line for the mesh report - SPEC-LIT section 119.1. Every
    /// component gains `+ 0.0` so `-0` prints as `0`.
    pub fn report_line(&self, m: &HostMesh) -> String {
        let f = |v: DVec3| (v.x + 0.0, v.y + 0.0, v.z + 0.0);
        let (ax, ay, az) = f(self.axis);
        let (px, py, pz) = f(self.axis_point);
        let (mx, my, mz) = f(self.mid_normal);
        format!(
            "  wedge: '{}' / '{}', theta {:.6} deg, axis ({ax:.9}, {ay:.9}, \
             {az:.9}) through ({px:.6e}, {py:.6e}, {pz:.6e}), mid-plane \
             normal ({mx:.9}, {my:.9}, {mz:.9})",
            m.patches[self.back].name,
            m.patches[self.front].name,
            self.theta.to_degrees(),
        )
    }
}

/// The rotated area a wedge face's reconstruction reads - SPEC-LIT 119.2,
/// equation (119.2j): the flux of the chord-average boundary value
/// `(U_P + R(sigma theta) U_P)/2` through the face's own area `sf` is the
/// CELL value through `(sf + R(-sigma theta) sf)/2`. `sigma` is `+1` on the
/// front patch and `-1` on the back; the rotation is the FULL Rodrigues
/// formula `R(phi) v = v cos(phi) + (a x v) sin(phi)
/// + a (a . v)(1 - cos(phi))` about the wedge axis. On a 119.1 wedge this
/// is `+cos(theta/2) |sf| mid_normal` on the front and its negative on the
/// back, so the face constrains the swirl component and nothing else.
pub fn recon_area(sf: DVec3, axis: DVec3, theta: f64, sigma: f64) -> DVec3 {
    let phi = -sigma * theta;
    let c = phi.cos();
    let s = phi.sin();
    let rotated = sf * c + axis.cross(sf) * s + axis * (axis.dot(sf) * (1.0 - c));
    (sf + rotated) * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockgen::{GradedAxis, WedgeSpec, build_wedge_mesh, raw_wedge_mesh};
    use crate::io::polymesh::build_host_mesh;

    fn spec_of(r: GradedAxis, z: GradedAxis, theta: f64) -> WedgeSpec {
        WedgeSpec { r, z, theta_deg: theta, patch_type: WedgeSpec::default().patch_type }
    }

    /// Annulus A: r 0.5..1.5 nr 8, z 0..2 nz 5, theta 5 deg.
    fn annulus() -> WedgeSpec {
        spec_of(
            GradedAxis { lo: 0.5, hi: 1.5, n: 8, ..GradedAxis::default() },
            GradedAxis { lo: 0.0, hi: 2.0, n: 5, ..GradedAxis::default() },
            5.0,
        )
    }

    /// Annulus A at another angle.
    fn annulus_at(theta: f64) -> WedgeSpec {
        let mut s = annulus();
        s.theta_deg = theta;
        s
    }

    /// Pipe P: r 0..1 nr 6, z 0..3 nz 4.
    fn pipe(theta: f64) -> WedgeSpec {
        spec_of(
            GradedAxis { lo: 0.0, hi: 1.0, n: 6, ..GradedAxis::default() },
            GradedAxis { lo: 0.0, hi: 3.0, n: 4, ..GradedAxis::default() },
            theta,
        )
    }

    /// The derivation itself, precision class A - BOTH builds: the axis is
    /// +z, the mid-plane normal +y, the full angle the spec's, the axis
    /// point on the axis line at the mesh's mid-height.
    #[test]
    fn wedge_axis_angle_and_mid_plane_are_derived() {
        for theta_deg in [1.0, 5.0, 10.0] {
            for (spec, r_out, z_mid, l) in
                [(pipe(theta_deg), 1.0, 1.5, 3.0), (annulus_at(theta_deg), 1.5, 1.0, 2.0)]
            {
                let m = build_wedge_mesh(&spec).unwrap();
                let w = m.wedge.clone().unwrap();
                assert_eq!(m.patches[w.back].name, "back");
                assert_eq!(m.patches[w.front].name, "front");
                assert_eq!(w.n_faces, spec.r.n * spec.z.n);
                let axis_err = (w.axis - DVec3::new(0.0, 0.0, 1.0)).mag();
                assert!(axis_err <= 1e-12, "axis error {axis_err:e}");
                assert!(
                    (w.theta - theta_deg.to_radians()).abs() <= 1e-12,
                    "theta {} vs {}",
                    w.theta,
                    theta_deg.to_radians()
                );
                assert!((w.mid_normal - DVec3::new(0.0, 1.0, 0.0)).mag() <= 1e-12);
                assert!(w.axis_point.x.hypot(w.axis_point.y) <= 1e-12 * r_out);
                assert!((w.axis_point.z - z_mid).abs() <= 1e-12 * l);
                println!("theta {theta_deg:>5} deg: axis error {axis_err:e}");
            }
        }
    }

    /// A mesh whose two sides are typed `symmetry` has no wedge to derive -
    /// and nothing else about the build objects.
    #[test]
    fn a_mesh_without_wedge_patches_has_no_wedge() {
        let mut raw = raw_wedge_mesh(&pipe(5.0)).unwrap();
        for name in ["back", "front"] {
            let p = raw.patches.iter_mut().find(|p| p.name == name).unwrap();
            p.type_name = "symmetry".to_string();
            p.kind = PatchKind::Symmetry;
        }
        let m = build_host_mesh(&raw).unwrap();
        assert!(m.wedge.is_none());
    }

    /// One wedge patch alone is refused, by name.
    #[test]
    fn refuses_an_unpaired_wedge() {
        let mut raw = raw_wedge_mesh(&pipe(5.0)).unwrap();
        let p = raw.patches.iter_mut().find(|p| p.name == "front").unwrap();
        p.type_name = "symmetry".to_string();
        p.kind = PatchKind::Symmetry;
        let msg = match build_host_mesh(&raw) {
            Err(Error::Mesh(msg)) => msg,
            other => panic!("expected Error::Mesh, got {other:?}"),
        };
        assert!(msg.contains("'back'"), "{msg}");
        assert!(msg.contains("has no partner"), "{msg}");
    }

    /// One front-side corner point nudged out of its plane is refused. The
    /// point is located with the 1e-6 tolerances, never `==`.
    #[test]
    fn refuses_a_non_planar_wedge() {
        let mut raw = raw_wedge_mesh(&annulus()).unwrap();
        let h = 2.5f64.to_radians();
        let hits: Vec<usize> = raw
            .points
            .iter()
            .enumerate()
            .filter(|(_, p)| p.y > 0.0 && (p.x - 1.0).abs() < 1e-6 && (p.z - 0.8).abs() < 1e-6)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(hits.len(), 1, "{hits:?}");
        let d = DVec3::new(-h.sin(), h.cos(), 0.0);
        let p = &mut raw.points[hits[0]];
        *p = *p + d * 1e-4;
        let msg = match build_host_mesh(&raw) {
            Err(Error::Mesh(msg)) => msg,
            other => panic!("expected Error::Mesh, got {other:?}"),
        };
        assert!(msg.contains("'front'"), "{msg}");
        assert!(msg.contains("is not planar"), "{msg}");
    }

    /// A front side steeper than the back is refused: the mirror test fails
    /// face for face.
    #[test]
    fn refuses_an_asymmetric_wedge() {
        let mut raw = raw_wedge_mesh(&annulus()).unwrap();
        let t3 = 3.0f64.to_radians().tan();
        for p in raw.points.iter_mut() {
            if p.y > 0.0 {
                p.y = p.x * t3;
            }
        }
        let msg = match build_host_mesh(&raw) {
            Err(Error::Mesh(msg)) => msg,
            other => panic!("expected Error::Mesh, got {other:?}"),
        };
        assert!(msg.contains("'back'"), "{msg}");
        assert!(msg.contains("'front'"), "{msg}");
        assert!(msg.contains("not symmetric about the mid-plane"), "{msg}");
    }

    /// The report line names the pair, the angle and the axis.
    #[test]
    fn report_line_names_the_pair_axis_and_angle() {
        let m = build_wedge_mesh(&pipe(5.0)).unwrap();
        let w = m.wedge.clone().unwrap();
        let line = w.report_line(&m);
        assert!(line.starts_with("  wedge: "), "{line:?}");
        assert!(line.contains("'back'"), "{line:?}");
        assert!(line.contains("'front'"), "{line:?}");
        assert!(line.contains("theta 5.000000 deg"), "{line:?}");
        assert!(line.contains("axis ("), "{line:?}");
    }
}
