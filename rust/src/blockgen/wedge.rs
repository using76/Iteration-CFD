// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §119.0 - the wedge mesh `blockgen` builds (BC-02): an (r, z)
//! block of `nr` x `nz` cells extruded over a full wedge angle `theta_deg`
//! (0 < theta < 180) about the y = 0 mid-plane, the axis on z through the
//! origin. ORIGINAL - elementary geometry (trapezoid area, prism volume,
//! the Taylor series of the tangent); no GPL-licensed source was consulted.
//!
//! With `t = wedge_half_tan(theta)` every point is
//! `P_s(i, k) = (r_i, s r_i t, z_k)`, s = -1 back / +1 front, so every
//! radial face is the plane `x = r_i` and every finite-volume metric of the
//! mesh is the curved sector's metric times the one constant `2t/theta` -
//! the reason the tangent convention is taken and the chord is not (§119.0).
//! On a pipe (`r_in == 0`) the axis point is shared by both sides, the
//! zero-area axis-side face is not written, the i = 0 cells are 5-face
//! prisms and no `inner` patch exists. The points, `t`, the point map and
//! every gate are f64 in BOTH builds (§118.2): the nodes come from
//! `graded_nodes` in `Scalar` and are widened once with `f64::from`, and
//! nothing downstream ever narrows them.

use crate::blockgen::{graded_nodes, GradedAxis};
use crate::error::{Error, Result};
use crate::io::polymesh::{build_host_mesh, write_poly_mesh_raw, PolyMeshRaw};
use crate::mesh::{HostMesh, PatchInfo, PatchKind};
use crate::types::DVec3;
use crate::Label;

/// The patches in emission order. `inner` is absent on a pipe, where the
/// axis-side face would have zero area (§119.0).
pub const WEDGE_PATCH_NAMES: [&str; 6] = ["inner", "outer", "zmin", "zmax", "back", "front"];

/// The wedge specification: the radial and axial [`GradedAxis`] blocks and
/// the FULL wedge angle in degrees. `patch_type` carries the types of
/// `inner`, `outer`, `zmin`, `zmax` in that order; `back` and `front` are
/// always typed `wedge`. `zmin`/`zmax` may be a `cyclic` pair - both or
/// neither, a half-declared pair is refused.
#[derive(Debug, Clone, PartialEq)]
pub struct WedgeSpec {
    /// `lo` is `r_in >= 0` (0 makes the mesh a pipe), `hi` is `r_out > lo`,
    /// `n` is `nr >= 1`; any grading [`GradedAxis`] supports.
    pub r: GradedAxis,
    /// `lo` is `z_0`, `hi` is `z_nz > lo`, `n` is `nz >= 1`.
    pub z: GradedAxis,
    /// FULL wedge angle in degrees, strictly between 0 and 180.
    pub theta_deg: f64,
    /// Types of `inner`, `outer`, `zmin`, `zmax`; `inner` is ignored when
    /// `r_in == 0` (no such patch is emitted).
    pub patch_type: [String; 4],
}

impl Default for WedgeSpec {
    fn default() -> Self {
        Self {
            r: GradedAxis { n: 8, ..GradedAxis::default() },
            z: GradedAxis { n: 8, ..GradedAxis::default() },
            theta_deg: 5.0,
            patch_type: [
                "wall".to_string(),
                "wall".to_string(),
                "patch".to_string(),
                "patch".to_string(),
            ],
        }
    }
}

/// Exactly `(theta_deg.to_radians() * 0.5).tan()` - the `t` of §119.0, the
/// one constant by which every metric of the mesh is the curved sector's.
pub fn wedge_half_tan(theta_deg: f64) -> f64 {
    (theta_deg.to_radians() * 0.5).tan()
}

/// A refusal carrying the module prefix every `blockgen::wedge` error has.
fn refused(msg: String) -> Error {
    Error::Mesh(format!("blockgen::wedge: {msg}"))
}

/// Every refusal of §119.0, checked before anything is built - the angle,
/// the two axis blocks, and the four patch types (`wedge` is refused on
/// them; a half-declared `cyclic` pair on `zmin`/`zmax` is refused by
/// name). `wedge_nodes` calls this first, so every entry point refuses
/// alike.
fn check_spec(spec: &WedgeSpec) -> Result<()> {
    if !(spec.theta_deg.is_finite() && spec.theta_deg > 0.0 && spec.theta_deg < 180.0) {
        return Err(refused(format!(
            "theta_deg must be finite and in (0, 180) degrees, got {}",
            spec.theta_deg
        )));
    }
    if !(spec.r.lo.is_finite() && spec.r.lo >= 0.0) {
        return Err(refused(format!("r.lo must be finite and >= 0, got {}", spec.r.lo)));
    }
    if !(spec.r.hi > spec.r.lo) {
        return Err(refused(format!(
            "r.hi ({}) must be > r.lo ({})",
            spec.r.hi, spec.r.lo
        )));
    }
    if !(spec.z.hi > spec.z.lo) {
        return Err(refused(format!(
            "z.hi ({}) must be > z.lo ({})",
            spec.z.hi, spec.z.lo
        )));
    }
    if spec.r.n == 0 || spec.z.n == 0 {
        return Err(refused(format!(
            "r.n and z.n must both be >= 1, got r.n = {}, z.n = {}",
            spec.r.n, spec.z.n
        )));
    }
    for (name, ty) in ["inner", "outer", "zmin", "zmax"].iter().zip(spec.patch_type.iter()) {
        if ty == "wedge" {
            return Err(refused(format!(
                "patch '{name}' must not have type 'wedge' - the wedge type is \
                 the one this generator writes on back and front itself"
            )));
        }
    }
    let (zmin_c, zmax_c) = (spec.patch_type[2] == "cyclic", spec.patch_type[3] == "cyclic");
    if zmin_c != zmax_c {
        return Err(refused(format!(
            "zmin is '{}' but zmax is '{}' - the zmin/zmax cyclic pair must be \
             both \"cyclic\" or neither",
            spec.patch_type[2], spec.patch_type[3]
        )));
    }
    Ok(())
}

/// The widened `r` and `z` nodes the mesh is built from: `graded_nodes` in
/// `Scalar`, widened once with `f64::from` - f64 in BOTH builds (§119.0).
/// The nodes of an axis with `n == 0` would be the single node `lo`, but
/// `check_spec` has already refused that.
pub fn wedge_nodes(spec: &WedgeSpec) -> Result<(Vec<f64>, Vec<f64>)> {
    check_spec(spec)?;
    let widen = |a: &GradedAxis| graded_nodes(a).iter().map(|&x| f64::from(x)).collect();
    Ok((widen(&spec.r), widen(&spec.z)))
}

/// Register one patch over the `size` boundary faces appended since
/// `start`, in the order the patch table lists them.
fn patch_of(patches: &mut Vec<PatchInfo>, start: usize, name: &str, ty: &str, size: usize) {
    patches.push(PatchInfo {
        name: name.to_string(),
        type_name: ty.to_string(),
        kind: PatchKind::from_type(ty),
        start,
        size,
        nbr_patch: None,
    });
}

/// The wedge mesh as a [`PolyMeshRaw`]: points f64 in both builds, every
/// face wound so its area vector points from owner to neighbour (internal)
/// or out of the domain (boundary), the internal faces in the
/// upper-triangular order every `blockgen` mesh uses.
pub fn raw_wedge_mesh(spec: &WedgeSpec) -> Result<PolyMeshRaw> {
    let (r, z) = wedge_nodes(spec)?;
    let t = wedge_half_tan(spec.theta_deg);
    let nr = spec.r.n;
    let nz = spec.z.n;
    let pipe = r[0] == 0.0;
    let i0: usize = if pipe { 1 } else { 0 };
    // Points per axial layer: the back side i = 0..=nr, then the front side
    // i = i0..=nr. On a pipe the front side starts at 1, because F(0, k) IS
    // the point B(0, k) (§119.0).
    let np = (nr + 1) + (nr + 1 - i0);

    let mut points = Vec::with_capacity((nz + 1) * np);
    for k in 0..=nz {
        for i in 0..=nr {
            // `r[i] * t` computed once and negated, so F is the bitwise
            // mirror of B.
            let y = r[i] * t;
            // The axis point (r = 0) is written as y = +0.0, not -0.0.
            let yb = if y == 0.0 { 0.0 } else { -y };
            points.push(DVec3::new(r[i], yb, z[k]));
        }
        for i in i0..=nr {
            let y = r[i] * t;
            points.push(DVec3::new(r[i], y, z[k]));
        }
    }

    let b = |i: usize, k: usize| (k * np + i) as Label;
    let f = |i: usize, k: usize| (k * np + (nr + 1) + (i - i0)) as Label;
    let cell = |i: usize, k: usize| (i + nr * k) as Label;

    let mut faces: Vec<Vec<Label>> = Vec::new();
    let mut owner: Vec<Label> = Vec::new();
    let mut neighbour: Vec<Label> = Vec::new();

    // Internal faces, per cell in cell order: the +r face (neighbour c + 1)
    // before the +z face (neighbour c + nr) - the upper-triangular order.
    for k in 0..nz {
        for i in 0..nr {
            let c = cell(i, k);
            if i + 1 < nr {
                faces.push(vec![b(i + 1, k), f(i + 1, k), f(i + 1, k + 1), b(i + 1, k + 1)]);
                owner.push(c);
                neighbour.push(c + 1);
            }
            if k + 1 < nz {
                // On a pipe the i = 0 cell's z-face holds the axis label
                // twice; the duplicate dropped, it is a 3-point triangle.
                if pipe && i == 0 {
                    faces.push(vec![b(0, k + 1), b(1, k + 1), f(1, k + 1)]);
                } else {
                    faces.push(vec![b(i, k + 1), b(i + 1, k + 1), f(i + 1, k + 1), f(i, k + 1)]);
                }
                owner.push(c);
                neighbour.push(c + nr as Label);
            }
        }
    }

    // The boundary faces, patch by patch in WEDGE_PATCH_NAMES order.
    let mut patches: Vec<PatchInfo> = Vec::new();
    let mut bnd: Vec<Vec<Label>> = Vec::new();
    let mut b_owner: Vec<Label> = Vec::new();
    let pt = |j: usize| spec.patch_type[j].clone();

    if !pipe {
        // inner: the -r faces in ascending k. Absent on a pipe, where the
        // axis-side face of every i = 0 cell would have zero area.
        let start = bnd.len();
        for k in 0..nz {
            bnd.push(vec![b(0, k), b(0, k + 1), f(0, k + 1), f(0, k)]);
            b_owner.push(cell(0, k));
        }
        patch_of(&mut patches, start, "inner", &pt(0), nz);
    }
    let start = bnd.len();
    for k in 0..nz {
        bnd.push(vec![b(nr, k), f(nr, k), f(nr, k + 1), b(nr, k + 1)]);
        b_owner.push(cell(nr - 1, k));
    }
    patch_of(&mut patches, start, "outer", &pt(1), nz);

    let start = bnd.len();
    for i in 0..nr {
        let pts = if pipe && i == 0 {
            vec![b(0, 0), f(1, 0), b(1, 0)]
        } else {
            vec![b(i, 0), f(i, 0), f(i + 1, 0), b(i + 1, 0)]
        };
        bnd.push(pts);
        b_owner.push(cell(i, 0));
    }
    patch_of(&mut patches, start, "zmin", &pt(2), nr);

    let start = bnd.len();
    for i in 0..nr {
        let pts = if pipe && i == 0 {
            vec![b(0, nz), b(1, nz), f(1, nz)]
        } else {
            vec![b(i, nz), b(i + 1, nz), f(i + 1, nz), f(i, nz)]
        };
        bnd.push(pts);
        b_owner.push(cell(i, nz - 1));
    }
    patch_of(&mut patches, start, "zmax", &pt(3), nr);

    let start = bnd.len();
    for k in 0..nz {
        for i in 0..nr {
            bnd.push(vec![b(i, k), b(i + 1, k), b(i + 1, k + 1), b(i, k + 1)]);
            b_owner.push(cell(i, k));
        }
    }
    patch_of(&mut patches, start, "back", "wedge", nr * nz);

    let start = bnd.len();
    for k in 0..nz {
        for i in 0..nr {
            let pts = if pipe && i == 0 {
                // F(0, .) is the axis point B(0, .) on a pipe: the same
                // point, so the front face carries the back-side label.
                vec![b(0, k), b(0, k + 1), f(1, k + 1), f(1, k)]
            } else {
                vec![f(i, k), f(i, k + 1), f(i + 1, k + 1), f(i + 1, k)]
            };
            bnd.push(pts);
            b_owner.push(cell(i, k));
        }
    }
    patch_of(&mut patches, start, "front", "wedge", nr * nz);

    // check_spec has refused a half-declared pair, so both are cyclic here;
    // each names the other's index in the patch list.
    if spec.patch_type[2] == "cyclic" {
        let zi = patches.iter().position(|p| p.name == "zmin").unwrap();
        let xa = patches.iter().position(|p| p.name == "zmax").unwrap();
        patches[zi].nbr_patch = Some(xa);
        patches[xa].nbr_patch = Some(zi);
    }

    faces.extend(bnd);
    owner.extend(b_owner);

    Ok(PolyMeshRaw { points, faces, owner, neighbour, patches })
}

/// Build the wedge mesh as a [`HostMesh`] with the geometry computed: the
/// in-memory twin of [`write_wedge_mesh`] + `read_poly_mesh` +
/// `build_host_mesh`, with the same validation.
pub fn build_wedge_mesh(spec: &WedgeSpec) -> Result<HostMesh> {
    build_host_mesh(&raw_wedge_mesh(spec)?)
}

/// Write `constant/polyMesh` under `case_dir`.
pub fn write_wedge_mesh(case_dir: &std::path::Path, spec: &WedgeSpec) -> Result<()> {
    let raw = raw_wedge_mesh(spec)?;
    write_poly_mesh_raw(&case_dir.join("constant").join("polyMesh"), &raw)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Annulus G: A with the radial axis geometrically graded (ratio 4).
    fn annulus_graded() -> WedgeSpec {
        let mut s = annulus();
        s.r.expansion = 4.0;
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

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs()
    }

    /// The refusal message of a spec this generator must not build.
    fn refused(spec: &WedgeSpec) -> String {
        match raw_wedge_mesh(spec) {
            Err(Error::Mesh(m)) => m,
            other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
        }
    }

    /// §119.0's volume gate: the mesh holds `t (R2^2 - R1^2) L` and every
    /// cell its closed form, to 1e-12 relative. The printed ratio
    /// `tan(theta/2) / (theta/2)` is the one constant by which every metric
    /// of the straight-faced mesh is the curved sector's: 1.0000254 at
    /// 1 deg, 1.000635 at 5 deg, 1.002546 at 10 deg.
    #[test]
    fn wedge_annulus_volume() {
        for theta in [1.0, 5.0, 10.0, 45.0, 120.0] {
            let t = wedge_half_tan(theta);
            let half = theta.to_radians() * 0.5;
            println!("theta {theta:>5} deg: tan(theta/2)/(theta/2) = {:.7}", t / half);
            let mut a = annulus();
            a.theta_deg = theta;
            let mut gspec = annulus_graded();
            gspec.theta_deg = theta;
            for spec in [a, gspec] {
                let m = build_wedge_mesh(&spec).unwrap();
                let rpt = m.check();
                let want_total = t * (1.5 * 1.5 - 0.5 * 0.5) * 2.0;
                assert!(
                    rel(rpt.total_volume, want_total) <= 1e-12,
                    "theta {theta}: total volume {} != {want_total}",
                    rpt.total_volume
                );
                let g = m.geom64();
                let (r, z) = wedge_nodes(&spec).unwrap();
                for k in 0..spec.z.n {
                    for i in 0..spec.r.n {
                        let c = i + spec.r.n * k;
                        let want = t * (r[i + 1] * r[i + 1] - r[i] * r[i]) * (z[k + 1] - z[k]);
                        assert!(
                            rel(g.v[c], want) <= 1e-12,
                            "theta {theta} cell ({i},{k}): v = {} != {want}",
                            g.v[c]
                        );
                    }
                }
                assert!(rpt.max_closure_error <= 1e-12, "closure {}", rpt.max_closure_error);
                assert!(rpt.ldu_ordered);
                assert_eq!(rpt.n_regions, 1);
            }
        }
    }

    /// The pipe's axis: no `inner` patch, no degenerate face - the i = 0
    /// cells are 5-face prisms with two axis triangles, and the smallest
    /// face in the mesh is one of those triangles, of area `t r_1^2`.
    #[test]
    fn wedge_pipe_axis_has_no_degenerate_faces() {
        for theta in [1.0, 5.0, 10.0] {
            let spec = pipe(theta);
            let nr = spec.r.n;
            let nz = spec.z.n;
            let raw = raw_wedge_mesh(&spec).unwrap();
            let m = build_wedge_mesh(&spec).unwrap();
            let g = m.geom64();

            assert!(!m.patches.iter().any(|p| p.name == "inner"), "no inner patch on a pipe");
            assert_eq!(m.patches.len(), 5);
            assert_eq!(m.n_points, (nz + 1) * (2 * nr + 1));

            for fv in &raw.faces {
                assert!(fv.len() >= 3, "a face with {} points", fv.len());
                let mut s = fv.clone();
                s.sort_unstable();
                s.dedup();
                assert_eq!(s.len(), fv.len(), "repeated point label in a face");
            }

            // Each face counted once per cell it bounds.
            let n_pts: Vec<usize> = raw.faces.iter().map(|f| f.len()).collect();
            let mut n_faces = vec![0usize; m.n_cells];
            let mut n_tris = vec![0usize; m.n_cells];
            for (fi, o) in raw.owner.iter().enumerate() {
                n_faces[*o as usize] += 1;
                if n_pts[fi] == 3 {
                    n_tris[*o as usize] += 1;
                }
            }
            for (fi, n) in raw.neighbour.iter().enumerate() {
                n_faces[*n as usize] += 1;
                if n_pts[fi] == 3 {
                    n_tris[*n as usize] += 1;
                }
            }
            for c in 0..m.n_cells {
                if c % nr == 0 {
                    assert_eq!(n_faces[c], 5, "axis cell {c}");
                    assert_eq!(n_tris[c], 2, "axis cell {c}: two 3-point faces");
                } else {
                    assert_eq!(n_faces[c], 6, "cell {c}");
                    assert_eq!(n_tris[c], 0, "cell {c}: all faces 4-point");
                }
            }
            let n_tri_total = n_pts.iter().filter(|&&n| n == 3).count();
            assert_eq!(n_tri_total, nz + 1);

            let mut min_sf = f64::INFINITY;
            for x in g.mag_sf.iter().chain(g.b_mag_sf.iter()) {
                assert!(*x >= 1e-14, "face area {x}");
                min_sf = min_sf.min(*x);
            }
            let t = wedge_half_tan(theta);
            let (r, _) = wedge_nodes(&spec).unwrap();
            assert!(
                rel(min_sf, t * r[1] * r[1]) <= 1e-12,
                "smallest face area {min_sf} != axis triangle {}",
                t * r[1] * r[1]
            );
            let want_total = t * 1.0 * 3.0;
            assert!(rel(m.check().total_volume, want_total) <= 1e-12);
            assert!(m.check().max_closure_error <= 1e-12);
        }
    }

    /// Every face lies in the plane its topology says: the wedge faces in
    /// the two wedge planes with §119.0's normals and areas, the radial
    /// faces in `x = r_i` with §119.0's `2 t r dz` and a purely +x vector.
    #[test]
    fn wedge_faces_lie_in_the_wedge_planes() {
        let theta: f64 = 5.0;
        let h = theta.to_radians() * 0.5;
        let (sin_h, cos_h) = h.sin_cos();
        let t = wedge_half_tan(theta);
        for spec in [annulus(), pipe(theta)] {
            let m = build_wedge_mesh(&spec).unwrap();
            let g = m.geom64();
            let (r, z) = wedge_nodes(&spec).unwrap();
            let nr = spec.r.n;
            let nz = spec.z.n;

            let want_names: Vec<&str> = WEDGE_PATCH_NAMES
                .iter()
                .copied()
                .filter(|n| *n != "inner" || spec.r.lo != 0.0)
                .collect();
            let names: Vec<&str> = m.patches.iter().map(|p| p.name.as_str()).collect();
            assert_eq!(names, want_names);
            for p in &m.patches {
                let want = match p.name.as_str() {
                    "inner" | "outer" => "wall",
                    "zmin" | "zmax" => "patch",
                    "back" | "front" => "wedge",
                    other => panic!("unexpected patch {other}"),
                };
                assert_eq!(p.type_name, want, "type of {}", p.name);
            }
            for p in &m.patches {
                let want = match p.name.as_str() {
                    "back" => [-sin_h, -cos_h, 0.0],
                    "front" => [-sin_h, cos_h, 0.0],
                    _ => continue,
                };
                for j in 0..p.size {
                    let bf = p.start + j;
                    let n = [
                        g.b_sf[bf].x / g.b_mag_sf[bf],
                        g.b_sf[bf].y / g.b_mag_sf[bf],
                        g.b_sf[bf].z / g.b_mag_sf[bf],
                    ];
                    for (a, b) in n.iter().zip(want) {
                        assert!(
                            (a - b).abs() <= 1e-12,
                            "{} face {bf}: normal {n:?} != {want:?}",
                            p.name
                        );
                    }
                }
            }
            let back = m.patches.iter().find(|p| p.name == "back").unwrap();
            let front = m.patches.iter().find(|p| p.name == "front").unwrap();
            for k in 0..nz {
                for i in 0..nr {
                    let c = i + nr * k;
                    let want = (r[i + 1] - r[i]) * (z[k + 1] - z[k]) / cos_h;
                    assert!(rel(g.b_mag_sf[back.start + c], want) <= 1e-12, "back ({i},{k})");
                    assert!(rel(g.b_mag_sf[front.start + c], want) <= 1e-12, "front ({i},{k})");
                }
            }
            let mut fi = 0usize;
            for k in 0..nz {
                for i in 0..nr {
                    if i + 1 < nr {
                        let want = 2.0 * t * r[i + 1] * (z[k + 1] - z[k]);
                        assert!(rel(g.mag_sf[fi], want) <= 1e-12, "+r face ({i},{k})");
                        assert!(g.sf[fi].y.abs() <= 1e-12 * g.mag_sf[fi], "sf.y ({i},{k})");
                        assert!(g.sf[fi].z.abs() <= 1e-12 * g.mag_sf[fi], "sf.z ({i},{k})");
                        assert!(g.sf[fi].x > 0.0, "sf.x ({i},{k})");
                        fi += 1;
                    }
                    if k + 1 < nz {
                        fi += 1;
                    }
                }
            }
            assert_eq!(fi, m.n_internal_faces);
        }
    }

    /// Class D: every front point is the bitwise mirror of its back point.
    #[test]
    fn wedge_front_is_the_mirror_of_back() {
        for spec in [annulus(), pipe(5.0)] {
            let raw = raw_wedge_mesh(&spec).unwrap();
            let nr = spec.r.n;
            let nz = spec.z.n;
            let is_pipe = spec.r.lo == 0.0;
            let i0: usize = if is_pipe { 1 } else { 0 };
            let np = (nr + 1) + (nr + 1 - i0);
            assert_eq!(raw.points.len(), (nz + 1) * np);
            for k in 0..=nz {
                for i in i0..=nr {
                    let bp = &raw.points[k * np + i];
                    let fp = &raw.points[k * np + (nr + 1) + (i - i0)];
                    assert_eq!(fp.x.to_bits(), bp.x.to_bits(), "x of F({i},{k})");
                    assert_eq!(fp.z.to_bits(), bp.z.to_bits(), "z of F({i},{k})");
                    assert_eq!(fp.y.to_bits(), (-bp.y).to_bits(), "y of F({i},{k})");
                }
            }
            if is_pipe {
                for k in 0..=nz {
                    let axis = &raw.points[k * np];
                    assert_eq!(axis.x, 0.0, "axis x at k = {k}");
                    assert_eq!(axis.y.to_bits(), 0.0f64.to_bits(), "axis y at k = {k} is +0.0, not -0.0");
                }
            }
        }
    }

    /// The counts of the annulus: 40 cells, 108 points, 67 internal faces,
    /// patches 5/5/8/8/40/40.
    #[test]
    fn wedge_counts() {
        let spec = annulus();
        let raw = raw_wedge_mesh(&spec).unwrap();
        let m = build_wedge_mesh(&spec).unwrap();
        assert_eq!(m.n_cells, 40);
        assert_eq!(m.n_points, 108);
        assert_eq!(m.n_internal_faces, 67);
        assert_eq!(m.n_boundary_faces, 106);
        let names: Vec<&str> = raw.patches.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, WEDGE_PATCH_NAMES.to_vec());
        let sizes: Vec<usize> = raw.patches.iter().map(|p| p.size).collect();
        assert_eq!(sizes, [5, 5, 8, 8, 40, 40]);
        assert_eq!(raw.faces.len(), 67 + 106);
    }

    /// Every refusal of §119.0 fires, and its message names what is wrong.
    #[test]
    fn wedge_spec_refusals() {
        let mut s = annulus();
        for theta in [0.0, 180.0, -5.0, f64::NAN] {
            s.theta_deg = theta;
            let m = refused(&s);
            assert!(m.starts_with("blockgen::wedge:"), "{m}");
            assert!(m.contains("theta"), "theta {theta}: {m}");
        }
        s = annulus();
        s.r.lo = -0.1;
        assert!(refused(&s).contains("r.lo"));
        s = annulus();
        s.r.hi = s.r.lo;
        assert!(refused(&s).contains("r.hi"));
        s = annulus();
        s.r.n = 0;
        assert!(refused(&s).contains("r.n"));
        s = annulus();
        s.z.n = 0;
        assert!(refused(&s).contains("z.n"));
        s = annulus();
        s.patch_type[1] = "wedge".to_string();
        assert!(refused(&s).contains("outer"), "{}", refused(&s));
        s = annulus();
        s.patch_type[2] = "cyclic".to_string();
        let m = refused(&s);
        assert!(m.contains("zmin") && m.contains("zmax"), "{m}");
    }

    /// The declared `cyclic` pair on zmin/zmax is wired both ways, and each
    /// zmin face's neighbour cell is the zmax face's owner across the layer.
    #[test]
    fn wedge_cyclic_z_pair() {
        let mut spec = annulus();
        spec.patch_type[2] = "cyclic".to_string();
        spec.patch_type[3] = "cyclic".to_string();
        let m = build_wedge_mesh(&spec).unwrap();
        let zi = m.patches.iter().position(|p| p.name == "zmin").unwrap();
        let xa = m.patches.iter().position(|p| p.name == "zmax").unwrap();
        assert_eq!(m.patches[zi].nbr_patch, Some(xa));
        assert_eq!(m.patches[xa].nbr_patch, Some(zi));
        let nr = spec.r.n;
        let nz = spec.z.n;
        for i in 0..nr {
            assert_eq!(
                m.b_nbr_cell[m.patches[zi].start + i],
                (i + nr * (nz - 1)) as Label,
                "zmin face {i}"
            );
        }
    }

    /// The mesh written to disk reads back the same.
    #[test]
    fn wedge_mesh_round_trips_through_disk() {
        let spec = annulus();
        let dir =
            std::env::temp_dir().join(format!("ofgpu_wedge_{}_roundtrip", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        write_wedge_mesh(&dir, &spec).unwrap();
        let back = crate::io::polymesh::read_poly_mesh(&dir).unwrap();
        let raw = raw_wedge_mesh(&spec).unwrap();
        assert_eq!(back.points, raw.points);
        assert_eq!(back.faces, raw.faces);
        assert_eq!(back.owner, raw.owner);
        assert_eq!(back.neighbour, raw.neighbour);
        assert_eq!(back.patches.len(), raw.patches.len());
        for (a, b) in back.patches.iter().zip(raw.patches.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.type_name, b.type_name);
            assert_eq!(a.start, b.start);
            assert_eq!(a.size, b.size);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
