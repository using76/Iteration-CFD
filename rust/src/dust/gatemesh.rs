// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §157.3 - the REDUCED gates' 1-D spherical sector (DST-07): a
//! thin pyramid with its apex at the ignition point and four planar side
//! faces through it, carrying slip conditions. The radial faces are planar
//! squares, so face areas grow as r² exactly and the cell volumes are the
//! frusta's exact volumes; the apex cell is a pyramid. Every position,
//! area and volume here is host f64 ([`DVec3`], never [`crate::Scalar`])
//! so the two builds agree bit for bit, and the apex IS the coordinate
//! origin, so F32-10's local-origin API has nothing to shift.
//!
//! SPEC-LIT 158.10 (DST-11) adds the PLACEHOLDER vessel generator below the
//! sector code: the labelled cylinder of `cases/dust/rembeVented.geometry.json`
//! (H22 publishes no REMBE geometry, so the gate is NAMED and NOT RUN and
//! nothing built here is compared with data) - its dimensions, the closed
//! wall-shell STL with the vent cut through it, the membrane plane on an
//! automesher base-grid plane, the `ofgpu-automesher` castellation config and
//! the sealed-region walk.
//!
//! The specification was informed by a review of FSCDustFoam (Huang & Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use crate::error::{Error, Result};
use crate::io::polymesh::{build_host_mesh, PolyMeshRaw};
use crate::mesh::{HostMesh, PatchInfo, PatchKind};
use crate::types::DVec3;
use crate::Label;
use serde::{Deserialize, Serialize};

/// The 1-D spherical sector of SPEC-LIT §157.3: `radius` R is the outer
/// square's distance from the apex, `fine_radius` r_f the uniform spacing's
/// end, `n_fine` the cells in [0, r_f] (the apex pyramid is the first),
/// `growth` the ratio of successive radial spacings beyond r_f and
/// `angle_deg` the FULL angle between opposite side faces. The patch names
/// and their type strings ride in the order `outer, ymin, ymax, zmin,
/// zmax` - the emission order of [`raw_sector_mesh`].
#[derive(Debug, Clone, PartialEq)]
pub struct SectorSpec {
    /// R, m: the outer square's distance from the apex.
    pub radius: f64,
    /// r_f, m: `0 < r_f < R`; the uniform spacing runs out to here.
    pub fine_radius: f64,
    /// `>= 1` cells in `[0, r_f]`; `dr = r_f / n_fine`; the apex pyramid is
    /// the first of them.
    pub n_fine: usize,
    /// g in `[1, 1.2]`: the ratio of successive radial spacings beyond r_f.
    pub growth: f64,
    /// The FULL angle between opposite side faces, in `(0, 90]` degrees.
    pub angle_deg: f64,
    /// The five patches' names, in `outer, ymin, ymax, zmin, zmax` order.
    pub patch_name: [String; 5],
    /// Their type strings (`"wall"`, `"symmetry"`, `"patch"`, ...), mapped
    /// onto [`PatchKind`] the way `blockgen::wedge` maps its own.
    pub patch_type: [String; 5],
}

/// A refusal naming the field, every one of them SPEC-LIT §157.3's.
fn refused(field: &str, detail: String) -> Error {
    Error::Config(format!("gatemesh::sector.{field}: {detail} (SPEC-LIT 157.3)"))
}

/// Exactly `tan(angle_deg / 2)` - `a`, the constant of §157.3's geometry:
/// the square at radius x has side `2 a x`, area `4 a² x²`, and the frustum
/// `[x_a, x_b]` has volume `(4 a²/3)(x_b³ - x_a³)`.
fn half_angle_tan(angle_deg: f64) -> f64 {
    (angle_deg.to_radians() * 0.5).tan()
}

/// Every refusal of §157.3, checked before anything is built - the radius,
/// the fine radius, the cell count, the growth and the angle, each named.
/// Every entry point calls this first, so all of them refuse alike.
pub fn check_spec(spec: &SectorSpec) -> Result<()> {
    if !(spec.radius.is_finite() && spec.radius > 0.0) {
        return Err(refused(
            "radius",
            format!("must be finite and > 0, got {}", spec.radius),
        ));
    }
    if !(spec.fine_radius.is_finite() && spec.fine_radius > 0.0 && spec.fine_radius < spec.radius)
    {
        return Err(refused(
            "fine_radius",
            format!(
                "must be finite and in (0, radius) = (0, {}), got {}",
                spec.radius, spec.fine_radius
            ),
        ));
    }
    if spec.n_fine == 0 {
        return Err(refused(
            "n_fine",
            format!("must be >= 1 - the apex pyramid is a cell, got {}", spec.n_fine),
        ));
    }
    if !(spec.growth.is_finite() && (1.0..=1.2).contains(&spec.growth)) {
        return Err(refused(
            "growth",
            format!("must be finite and in [1, 1.2], got {}", spec.growth),
        ));
    }
    if !(spec.angle_deg.is_finite() && spec.angle_deg > 0.0 && spec.angle_deg <= 90.0) {
        return Err(refused(
            "angle_deg",
            format!("must be finite and in (0, 90] degrees, got {}", spec.angle_deg),
        ));
    }
    Ok(())
}

/// The radii `x_0 = 0, ..., x_N = R` - N + 1 values, N the cell count.
/// Uniform `dr = r_f / n_fine` out to r_f (`x_i = i * dr`, and `x_{n_fine}`
/// = r_f assigned exactly), then the smallest count `n_g >= 1` of spacings
/// `h_k = dr g^k` whose accumulated sum reaches `R - r_f`, scaled by
/// `s = (R - r_f) / sum` so the graded tail lands on R and `x_N = R`
/// assigned, never summed (SPEC-LIT §157.3).
pub fn sector_radii(spec: &SectorSpec) -> Result<Vec<f64>> {
    check_spec(spec)?;
    let dr = spec.fine_radius / spec.n_fine as f64;
    let mut x = Vec::with_capacity(spec.n_fine + 1);
    x.push(0.0);
    for i in 1..spec.n_fine {
        x.push(i as f64 * dr);
    }
    x.push(spec.fine_radius);
    let tail = spec.radius - spec.fine_radius;
    let (n_g, sum) = {
        let mut sum = 0.0;
        let mut n_g = 0usize;
        while sum < tail {
            n_g += 1;
            sum += dr * spec.growth.powi(n_g as i32);
        }
        (n_g, sum)
    };
    let s = tail / sum;
    let mut xi = spec.fine_radius;
    for k in 1..=n_g {
        xi += dr * spec.growth.powi(k as i32) * s;
        x.push(xi);
    }
    let n = x.len() - 1;
    x[n] = spec.radius;
    Ok(x)
}

/// The exact centroid x of the frustum `[x_a, x_b]`,
/// `x_c = (3/4)(x_b⁴ - x_a⁴)/(x_b³ - x_a³)`; the apex pyramid is the limit
/// `x_a -> 0`, `x_c = 3 x_b / 4`.
fn centroid_x(x_a: f64, x_b: f64) -> f64 {
    if x_a == 0.0 {
        0.75 * x_b
    } else {
        0.75 * (x_b.powi(4) - x_a.powi(4)) / (x_b.powi(3) - x_a.powi(3))
    }
}

/// The exact centroid x of each cell (SPEC-LIT §157.3) - the radius every
/// 1-D comparison of the gates is made at.
pub fn cell_radii(spec: &SectorSpec) -> Result<Vec<f64>> {
    Ok(sector_radii(spec)?
        .windows(2)
        .map(|w| centroid_x(w[0], w[1]))
        .collect())
}

/// One boundary patch registered over the `size` faces appended since
/// `start`, its kind from the type string exactly as `blockgen::wedge`'s
/// own `patch_of` maps it.
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

/// The sector mesh as a [`PolyMeshRaw`]. Point 0 is the apex (the origin);
/// layer i in 1..=N carries the four points `1 + 4 (i - 1) + j` at
/// `a = tan(angle/2)` round its square, j running counter-clockwise about
/// +x. The internal faces are the layers 1..N-1 squares wound +x, owner
/// i-1, neighbour i - the upper-triangular order every `blockgen` mesh
/// uses. The boundary is, in patch order, the outer square (layer N, +x)
/// then the four sides, each cell's face wound so its right-hand-rule area
/// vector points OUT of the cell (SPEC-LIT §157.3): a triangle
/// `[apex, two layer-1 points]` for the apex pyramid and a quad per frustum.
pub fn raw_sector_mesh(spec: &SectorSpec) -> Result<PolyMeshRaw> {
    check_spec(spec)?;
    let x = sector_radii(spec)?;
    let n = x.len() - 1;
    let a = half_angle_tan(spec.angle_deg);
    let layer = |i: usize, j: usize| (1 + 4 * (i - 1) + j) as Label;

    let mut points = Vec::with_capacity(1 + 4 * n);
    points.push(DVec3::ZERO);
    for &xi in &x[1..] {
        let y = a * xi;
        points.push(DVec3::new(xi, -y, -y));
        points.push(DVec3::new(xi, y, -y));
        points.push(DVec3::new(xi, y, y));
        points.push(DVec3::new(xi, -y, y));
    }

    let mut faces: Vec<Vec<Label>> = Vec::new();
    let mut owner: Vec<Label> = Vec::new();
    let mut neighbour: Vec<Label> = Vec::new();
    for i in 1..n {
        faces.push(vec![layer(i, 0), layer(i, 1), layer(i, 2), layer(i, 3)]);
        owner.push((i - 1) as Label);
        neighbour.push(i as Label);
    }

    let mut bnd: Vec<Vec<Label>> = Vec::new();
    let mut b_owner: Vec<Label> = Vec::new();
    bnd.push(vec![layer(n, 0), layer(n, 1), layer(n, 2), layer(n, 3)]);
    b_owner.push((n - 1) as Label);
    let mut patches: Vec<PatchInfo> = Vec::new();
    patch_of(&mut patches, 0, &spec.patch_name[0], &spec.patch_type[0], 1);

    // ymin, the plane y = -a x (j = 0 and 3): quads wound to -y.
    let start = bnd.len();
    bnd.push(vec![0, layer(1, 0), layer(1, 3)]);
    b_owner.push(0);
    for c in 1..n {
        bnd.push(vec![layer(c, 0), layer(c + 1, 0), layer(c + 1, 3), layer(c, 3)]);
        b_owner.push(c as Label);
    }
    patch_of(&mut patches, start, &spec.patch_name[1], &spec.patch_type[1], n);

    // ymax, the plane y = +a x (j = 1 and 2): wound to +y.
    let start = bnd.len();
    bnd.push(vec![0, layer(1, 2), layer(1, 1)]);
    b_owner.push(0);
    for c in 1..n {
        bnd.push(vec![layer(c, 1), layer(c, 2), layer(c + 1, 2), layer(c + 1, 1)]);
        b_owner.push(c as Label);
    }
    patch_of(&mut patches, start, &spec.patch_name[2], &spec.patch_type[2], n);

    // zmin, the plane z = -a x (j = 0 and 1): wound to -z.
    let start = bnd.len();
    bnd.push(vec![0, layer(1, 1), layer(1, 0)]);
    b_owner.push(0);
    for c in 1..n {
        bnd.push(vec![layer(c + 1, 0), layer(c, 0), layer(c, 1), layer(c + 1, 1)]);
        b_owner.push(c as Label);
    }
    patch_of(&mut patches, start, &spec.patch_name[3], &spec.patch_type[3], n);

    // zmax, the plane z = +a x (j = 2 and 3): wound to +z.
    let start = bnd.len();
    bnd.push(vec![0, layer(1, 3), layer(1, 2)]);
    b_owner.push(0);
    for c in 1..n {
        bnd.push(vec![layer(c, 2), layer(c, 3), layer(c + 1, 3), layer(c + 1, 2)]);
        b_owner.push(c as Label);
    }
    patch_of(&mut patches, start, &spec.patch_name[4], &spec.patch_type[4], n);

    faces.extend(bnd);
    owner.extend(b_owner);
    Ok(PolyMeshRaw { points, faces, owner, neighbour, patches })
}

/// The sector mesh as a [`HostMesh`]: `crate::io::polymesh`'s builder on
/// [`raw_sector_mesh`], with the same validation every mesh goes through.
/// The origin is ZERO because the apex IS the origin - F32-10's
/// local-origin API has nothing to shift (SPEC-LIT §157.3).
pub fn build_sector_mesh(spec: &SectorSpec) -> Result<HostMesh> {
    build_host_mesh(&raw_sector_mesh(spec)?)
}

/// What [`check_sector`] measured, host f64, from the raw points alone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SectorCheck {
    /// The worst `|sum of outward S_f| / sum |S_f|` over the cells.
    pub closure_max: f64,
    /// The worst `|A / (4 a² x²) - 1|` over the N radial squares.
    pub area_r2_max: f64,
    /// The cell count N.
    pub cells: usize,
}

/// The area vector of one face of `raw`, by the polygon formula
/// `0.5 Σ (p_k - p_0) x (p_{k+1} - p_0)` in f64 from the raw points.
fn face_area_vec(points: &[DVec3], f: &[Label]) -> DVec3 {
    let p0 = points[f[0] as usize];
    let mut s = DVec3::ZERO;
    for k in 0..f.len() {
        let u = points[f[k] as usize] - p0;
        let v = points[f[(k + 1) % f.len()] as usize] - p0;
        s += u.cross(v);
    }
    s * 0.5
}

/// The §157.3 gate, measured on [`raw_sector_mesh`]'s own output: the worst
/// relative closure of any cell's outward area-vector sum, and the worst
/// deviation of a radial square's area from `4 a² x²` (the N-1 internal
/// squares and the outer one - the sides are not radial faces).
pub fn check_sector(spec: &SectorSpec) -> Result<SectorCheck> {
    let raw = raw_sector_mesh(spec)?;
    let x = sector_radii(spec)?;
    let a = half_angle_tan(spec.angle_deg);
    let n = x.len() - 1;

    let mut sum_sf = vec![DVec3::ZERO; n];
    let mut sum_mag = vec![0.0f64; n];
    let n_if = raw.neighbour.len();
    for (fi, f) in raw.faces.iter().enumerate() {
        let s = face_area_vec(&raw.points, f);
        let o = raw.owner[fi] as usize;
        sum_sf[o] += s;
        sum_mag[o] += s.dot(s).sqrt();
        if fi < n_if {
            let nb = raw.neighbour[fi] as usize;
            sum_sf[nb] -= s;
            sum_mag[nb] += s.dot(s).sqrt();
        }
    }
    let mut closure_max = 0.0f64;
    for c in 0..n {
        let rel = sum_sf[c].dot(sum_sf[c]).sqrt() / sum_mag[c];
        closure_max = closure_max.max(rel);
    }

    let square = |fi: usize, xi: f64| {
        let s = face_area_vec(&raw.points, &raw.faces[fi]);
        (s.dot(s).sqrt() / (4.0 * a * a * xi * xi) - 1.0).abs()
    };
    let mut area_r2_max = 0.0f64;
    for i in 1..n {
        area_r2_max = area_r2_max.max(square(i - 1, x[i]));
    }
    // The outer square is the FIRST boundary face.
    area_r2_max = area_r2_max.max(square(n_if, x[n]));
    Ok(SectorCheck { closure_max, area_r2_max, cells: n })
}

// ==========================================================================
//  The placeholder vessel generator (docs/17 DST-11, SPEC-LIT 158.10)
// ==========================================================================
//
// DST-REMBE is NAMED and NOT RUN until the vessel's geometry is obtained
// (H22 publishes none, SPEC-LIT 151.7 item 9). Everything below builds the
// labelled PLACEHOLDER cylinder of `cases/dust/rembeVented.geometry.json`
// for the REDUCED smoke, which is never compared with data. The numbers are
// the placeholder's definitions, not physics:
//
//   D = (4 V / (pi lambda))^(1/3),   R = D/2,   L = lambda D
//     (158.10a) of SPEC-LIT 158.10
//   r_v = (A_v / pi)^(1/2)
//     (158.10b) of SPEC-LIT 158.10
//   x1 = X_m - t/2,  x0 = x1 - L;  in_cavity(p): x0 < p.x < x1 and
//     |(p.y - axisY, p.z - axisZ)| < R
//     (158.10c) of SPEC-LIT 158.10
//   n = max(1, round((x_hi - x_lo)/h_b)),  h0 = (x_hi - x_lo)/n,
//     i = round((X_m - x_lo)/h0),  plane = x_lo + i h0,  h_f = h0 / 2^level
//     (158.10d) of SPEC-LIT 158.10
//   A(rho) = (N/2) rho^2 sin(2 pi / N),
//     V_solid = A(R+t) (L+2t) - A(R) L - A(r_v) t
//     (158.10e) of SPEC-LIT 158.10
//
// Every position is host f64 ([`DVec3`], `f64`), as the sector code above,
// so the two builds agree bit for bit.

/// The `<key>: <value> - <why> (SPEC-LIT §158.10)` refusal of the placeholder
/// geometry's reader and generator.
fn geometry_refused(key: &str, value: impl std::fmt::Display, why: &str) -> Error {
    Error::Config(format!("{key}: {value} - {why} (SPEC-LIT §158.10)"))
}

/// `cases/dust/rembeVented.geometry.json` (SPEC-LIT 158.10): the placeholder
/// vessel's parameters, strict JSON, every key required.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VesselGeometry {
    /// `true`: this is not the REMBE vessel.
    pub placeholder: bool,
    /// The label that says so, carried into every artefact built from it.
    pub label: String,
    pub vessel: VesselSpec,
    pub vent: VesselVent,
    pub domain: VesselDomain,
    pub probes: Vec<VesselProbe>,
}

/// `vessel`: the cylinder's volume V [m3], `lengthOverDiameter` lambda, the
/// wall thickness t [m], the axis' y and z [m], the membrane plane `X_m` [m]
/// and the N of the N-gon rings (see (158.10a) of SPEC-LIT 158.10, (158.10c)
/// of SPEC-LIT 158.10 and (158.10e) of SPEC-LIT 158.10).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VesselSpec {
    pub volume: f64,
    pub length_over_diameter: f64,
    pub shell: f64,
    pub axis_y: f64,
    pub axis_z: f64,
    pub vent_plane_x: f64,
    pub segments: usize,
}

/// `vent`: the vent's area `A_v` [m2] ((158.10b) of SPEC-LIT 158.10).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VesselVent {
    pub area: f64,
}

/// `domain`: the automesher's extent `[x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]`,
/// base size `h_b` [m], the refinement `level` inside `band` [m] of the
/// shell and `outerLevel` inside `outerBand` [m] ((158.10d) of SPEC-LIT
/// 158.10).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VesselDomain {
    pub extent: [f64; 6],
    pub base_size: f64,
    pub level: u32,
    pub band: f64,
    pub outer_level: u32,
    pub outer_band: f64,
}

/// One named pressure probe of the placeholder's file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VesselProbe {
    pub name: String,
    pub point: [f64; 3],
}

/// The vessel's derived dimensions, by (158.10a) of SPEC-LIT 158.10,
/// (158.10b) of SPEC-LIT 158.10 and (158.10c) of SPEC-LIT 158.10: `d` and `r` the cavity's diameter and radius, `l` its length,
/// `r_vent` the vent's radius, `x0`/`x1` the inner faces of the back and the
/// vented front end, `t` the wall and the axis' `y` and `z`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VesselDims {
    pub d: f64,
    pub r: f64,
    pub l: f64,
    pub r_vent: f64,
    pub x0: f64,
    pub x1: f64,
    pub t: f64,
    pub axis_y: f64,
    pub axis_z: f64,
}

impl VesselDims {
    /// (158.10c) of SPEC-LIT 158.10: `x0 < p.x < x1` and
    /// `|(p.y - axisY, p.z - axisZ)| < R`, both strict.
    pub fn in_cavity(&self, p: DVec3) -> bool {
        let (dy, dz) = (p.y - self.axis_y, p.z - self.axis_z);
        self.x0 < p.x && p.x < self.x1 && (dy * dy + dz * dz).sqrt() < self.r
    }
}

impl VesselGeometry {
    /// Every refusal of the file's fields, each naming its key
    /// (SPEC-LIT §158.10): a non-positive or non-finite volume,
    /// lengthOverDiameter, shell, vent area, baseSize, band or outerBand;
    /// fewer than 8 segments; `level < 1` or `outerLevel > level`; an extent
    /// pair with hi <= lo; an empty or repeated probe name; and a vent that
    /// does not fit inside the cavity (r_v >= R, by (158.10a) of SPEC-LIT
    /// 158.10 and (158.10b) of SPEC-LIT 158.10).
    pub fn check(&self) -> Result<()> {
        let v = &self.vessel;
        let d = &self.domain;
        let positive = |key: &str, x: f64| -> Result<()> {
            if x.is_finite() && x > 0.0 {
                Ok(())
            } else {
                Err(geometry_refused(key, x, "must be finite and > 0"))
            }
        };
        let finite = |key: &str, x: f64| -> Result<()> {
            if x.is_finite() {
                Ok(())
            } else {
                Err(geometry_refused(key, x, "must be finite"))
            }
        };
        positive("vessel.volume", v.volume)?;
        positive("vessel.lengthOverDiameter", v.length_over_diameter)?;
        positive("vessel.shell", v.shell)?;
        finite("vessel.axisY", v.axis_y)?;
        finite("vessel.axisZ", v.axis_z)?;
        finite("vessel.ventPlaneX", v.vent_plane_x)?;
        if v.segments < 8 {
            return Err(geometry_refused(
                "vessel.segments",
                v.segments,
                "must be >= 8 - fewer segments is not a cylinder",
            ));
        }
        positive("vent.area", self.vent.area)?;
        positive("domain.baseSize", d.base_size)?;
        positive("domain.band", d.band)?;
        positive("domain.outerBand", d.outer_band)?;
        if d.level < 1 {
            return Err(geometry_refused("domain.level", d.level, "must be >= 1"));
        }
        if d.outer_level > d.level {
            return Err(geometry_refused(
                "domain.outerLevel",
                d.outer_level,
                &format!("must not exceed domain.level = {}", d.level),
            ));
        }
        for (axis, name) in ["x", "y", "z"].iter().enumerate() {
            let (lo, hi) = (d.extent[2 * axis], d.extent[2 * axis + 1]);
            if !(lo.is_finite() && hi.is_finite() && hi > lo) {
                return Err(geometry_refused(
                    "domain.extent",
                    format!("{lo}..{hi} on {name}"),
                    "each pair must be finite with hi > lo",
                ));
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        for (i, p) in self.probes.iter().enumerate() {
            if p.name.is_empty() {
                return Err(geometry_refused(&format!("probes[{i}].name"), "\"\"", "is empty"));
            }
            if !seen.insert(p.name.as_str()) {
                return Err(geometry_refused(
                    &format!("probes[{i}].name"),
                    format!("\"{}\"", p.name),
                    "is repeated - each probe has its own name",
                ));
            }
            if !p.point.iter().all(|c| c.is_finite()) {
                return Err(geometry_refused(
                    &format!("probes[{i}].point"),
                    format!("{:?}", p.point),
                    "is not a finite point",
                ));
            }
        }
        let dims = self.dims();
        if !(dims.r_vent < dims.r) {
            return Err(geometry_refused(
                "vent.area",
                self.vent.area,
                &format!(
                    "makes the vent radius r_v = {} reach the cavity radius R = {} \
                     (by (158.10a) of SPEC-LIT 158.10 and (158.10b) of SPEC-LIT 158.10)",
                    dims.r_vent, dims.r
                ),
            ));
        }
        Ok(())
    }

    /// The dimensions of (158.10a) of SPEC-LIT 158.10, (158.10b) of SPEC-LIT
    /// 158.10 and (158.10c) of SPEC-LIT 158.10. Pure
    /// arithmetic on the fields; [`Self::check`] says whether they make a
    /// vessel.
    pub fn dims(&self) -> VesselDims {
        let v = &self.vessel;
        let lambda = v.length_over_diameter;
        let d = (4.0 * v.volume / (std::f64::consts::PI * lambda)).cbrt();
        let l = lambda * d;
        let x1 = v.vent_plane_x - v.shell / 2.0;
        VesselDims {
            d,
            r: d / 2.0,
            l,
            r_vent: (self.vent.area / std::f64::consts::PI).sqrt(),
            x0: x1 - l,
            x1,
            t: v.shell,
            axis_y: v.axis_y,
            axis_z: v.axis_z,
        }
    }
}

/// Read the placeholder geometry from text through
/// [`crate::io::case_json::parse_jsonc_str`] - an unknown key is named with
/// its path - then check its fields ([`VesselGeometry::check`]).
pub fn parse_vessel_geometry(text: &str, what: &str) -> Result<VesselGeometry> {
    let g: VesselGeometry = crate::io::case_json::parse_jsonc_str(text, what)?;
    g.check()?;
    Ok(g)
}

/// [`parse_vessel_geometry`] on a file.
pub fn read_vessel_geometry(path: &std::path::Path) -> Result<VesselGeometry> {
    use crate::error::IoContext;
    let text = std::fs::read_to_string(path).path(path)?;
    parse_vessel_geometry(&text, &path.display().to_string())
}

/// One STL facet, three vertices.
type Facet = [[f64; 3]; 3];

/// The N-gon rings the shell is built from: ring vertex `k` of radius `rho`
/// at `x` sits at the angle `2 pi (k mod N) / N`, at
/// `y = axisY + rho cos`, `z = axisZ + rho sin`. EVERY vertex of the solid -
/// the fan centres included, at `rho = 0` - comes out of [`Rings::vertex`],
/// so a vertex shared by two pieces is the same f64 triple and the same text.
struct Rings {
    n: usize,
    axis_y: f64,
    axis_z: f64,
}

impl Rings {
    fn vertex(&self, x: f64, rho: f64, k: usize) -> [f64; 3] {
        let a = 2.0 * std::f64::consts::PI * ((k % self.n) as f64) / (self.n as f64);
        [x, self.axis_y + rho * a.cos(), self.axis_z + rho * a.sin()]
    }

    /// A cylindrical side, `rho` fixed, from `xa` to `xb`: facing outward
    /// (away from the axis) or toward the axis.
    fn side(&self, out: &mut Vec<Facet>, xa: f64, xb: f64, rho: f64, outward: bool) {
        for k in 0..self.n {
            let a = self.vertex(xa, rho, k);
            let b = self.vertex(xb, rho, k);
            let c = self.vertex(xb, rho, k + 1);
            let d = self.vertex(xa, rho, k + 1);
            if outward {
                out.push([a, d, c]);
                out.push([a, c, b]);
            } else {
                out.push([a, b, c]);
                out.push([a, c, d]);
            }
        }
    }

    /// A disc at `x`, a fan from the axis point, facing `+x` or `-x`.
    fn disc(&self, out: &mut Vec<Facet>, x: f64, rho: f64, plus_x: bool) {
        let centre = self.vertex(x, 0.0, 0);
        for k in 0..self.n {
            let a = self.vertex(x, rho, k);
            let b = self.vertex(x, rho, k + 1);
            out.push(if plus_x { [centre, a, b] } else { [centre, b, a] });
        }
    }

    /// An annulus at `x` between `ri` and `ro`, facing `+x` or `-x`.
    fn annulus(&self, out: &mut Vec<Facet>, x: f64, ri: f64, ro: f64, plus_x: bool) {
        for k in 0..self.n {
            let a = self.vertex(x, ri, k);
            let b = self.vertex(x, ro, k);
            let c = self.vertex(x, ro, k + 1);
            let d = self.vertex(x, ri, k + 1);
            if plus_x {
                out.push([a, b, c]);
                out.push([a, c, d]);
            } else {
                out.push([a, d, c]);
                out.push([a, c, b]);
            }
        }
    }
}

/// The wall shell with the vent hole as ONE closed ASCII STL solid,
/// `solid vessel` ... `endsolid vessel` (SPEC-LIT 158.10): seven pieces of
/// N-gon rings, N = `vessel.segments`, every facet normal pointing OUT of the
/// solid - the outer side (`rho = R + t`, `x0 - t .. x1 + t`), the back outer
/// disc at `x0 - t`, the front outer annulus at `x1 + t` (`r_v .. R + t`),
/// the inner side (`rho = R`, `x0 .. x1`), the inner back disc at `x0`, the
/// inner front annulus at `x1` (`r_v .. R`) and the vent hole's side
/// (`rho = r_v`, `x1 .. x1 + t`). Its volume is V_solid of (158.10e) of
/// SPEC-LIT 158.10. A vertex is printed `{:.17e}`, which round-trips f64
/// exactly; the facet normal is the unit cross product.
pub fn vessel_stl(g: &VesselGeometry) -> Result<String> {
    use std::fmt::Write as _;
    g.check()?;
    let dm = g.dims();
    let rings = Rings { n: g.vessel.segments, axis_y: dm.axis_y, axis_z: dm.axis_z };
    let (r, rv, t) = (dm.r, dm.r_vent, dm.t);
    let mut facets: Vec<Facet> = Vec::with_capacity(rings.n * 12);
    rings.side(&mut facets, dm.x0 - t, dm.x1 + t, r + t, true); // 1. the outer side
    rings.disc(&mut facets, dm.x0 - t, r + t, false); // 2. the back outer disc
    rings.annulus(&mut facets, dm.x1 + t, rv, r + t, true); // 3. the front outer annulus
    rings.side(&mut facets, dm.x0, dm.x1, r, false); // 4. the inner side
    rings.disc(&mut facets, dm.x0, r, true); // 5. the inner back disc
    rings.annulus(&mut facets, dm.x1, rv, r, false); // 6. the inner front annulus
    rings.side(&mut facets, dm.x1, dm.x1 + t, rv, false); // 7. the vent hole's side

    let floor = f64::from(crate::SCALAR_FLOOR);
    let mut s = String::with_capacity(facets.len() * 420 + 32);
    s.push_str("solid vessel\n");
    for f in &facets {
        let u = [f[1][0] - f[0][0], f[1][1] - f[0][1], f[1][2] - f[0][2]];
        let w = [f[2][0] - f[0][0], f[2][1] - f[0][1], f[2][2] - f[0][2]];
        let n = [
            u[1] * w[2] - u[2] * w[1],
            u[2] * w[0] - u[0] * w[2],
            u[0] * w[1] - u[1] * w[0],
        ];
        let m = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        let n = if m > floor { [n[0] / m, n[1] / m, n[2] / m] } else { [0.0; 3] };
        let _ = write!(
            s,
            " facet normal {:.17e} {:.17e} {:.17e}\n  outer loop\n",
            n[0], n[1], n[2]
        );
        for p in f {
            let _ = writeln!(s, "   vertex {:.17e} {:.17e} {:.17e}", p[0], p[1], p[2]);
        }
        s.push_str("  endloop\n endfacet\n");
    }
    s.push_str("endsolid vessel\n");
    Ok(s)
}

/// The membrane plane by (158.10d) of SPEC-LIT 158.10:
/// `n = max(1, round((x_hi - x_lo)/h_b))`, `h0 = (x_hi - x_lo)/n`,
/// `i = round((X_m - x_lo)/h0)`, `plane = x_lo + i h0`, `h_f = h0 / 2^level`;
/// returns `x_lo + i h0`. The checks run in this order, each refused by key:
///
/// 1. `vessel.ventPlaneX` unless `|X_m - (x_lo + i h0)| <= 1e-9 (x_hi - x_lo)`
///    - the plane is then an automesher base-grid plane, so no octree cell
///    crosses it;
/// 2. `vessel.ventPlaneX` unless `x1 + h_f/2 < plane < x1 + t - h_f/2`;
/// 3. `vessel.shell` when `t < 2 h_f`;
/// 4. `domain.extent` unless the shell's bounding box
///    `[x0-t, x1+t] x [axisY-(R+t), axisY+(R+t)] x [axisZ-(R+t), axisZ+(R+t)]`
///    lies inside the extent with one `baseSize` of margin on every side -
///    the automesher's stage-0 rule.
pub fn membrane_plane(g: &VesselGeometry) -> Result<f64> {
    let dm = g.dims();
    let e = g.domain.extent;
    let (x_lo, x_hi) = (e[0], e[1]);
    let hb = g.domain.base_size;
    let xm = g.vessel.vent_plane_x;
    let n = ((x_hi - x_lo) / hb).round().max(1.0);
    let h0 = (x_hi - x_lo) / n;
    let i = ((xm - x_lo) / h0).round();
    let plane = x_lo + i * h0;
    let h_f = h0 / 2.0f64.powi(g.domain.level as i32);
    if !((xm - plane).abs() <= 1.0e-9 * (x_hi - x_lo)) {
        return Err(geometry_refused(
            "vessel.ventPlaneX",
            xm,
            &format!(
                "is not an automesher base-grid plane: the nearest is {plane} (h0 = {h0}), \
                 and an octree cell would cross the membrane"
            ),
        ));
    }
    if !(dm.x1 + h_f / 2.0 < plane && plane < dm.x1 + dm.t - h_f / 2.0) {
        return Err(geometry_refused(
            "vessel.ventPlaneX",
            xm,
            &format!(
                "the plane {plane} must lie strictly inside the front wall, between \
                 x1 + h_f/2 = {} and x1 + t - h_f/2 = {} (h_f = {h_f})",
                dm.x1 + h_f / 2.0,
                dm.x1 + dm.t - h_f / 2.0
            ),
        ));
    }
    if dm.t < 2.0 * h_f {
        return Err(geometry_refused(
            "vessel.shell",
            dm.t,
            &format!("is below 2 h_f = {}: the wall needs two finest cells", 2.0 * h_f),
        ));
    }
    let ro = dm.r + dm.t;
    let lo = [dm.x0 - dm.t, dm.axis_y - ro, dm.axis_z - ro];
    let hi = [dm.x1 + dm.t, dm.axis_y + ro, dm.axis_z + ro];
    for axis in 0..3 {
        let (e_lo, e_hi) = (e[2 * axis], e[2 * axis + 1]);
        if lo[axis] - hb < e_lo || hi[axis] + hb > e_hi {
            return Err(geometry_refused(
                "domain.extent",
                format!("{e_lo}..{e_hi} on {}", ["x", "y", "z"][axis]),
                &format!(
                    "must contain the shell's bounding box {}..{} with one baseSize = {hb} \
                     of margin on every side",
                    lo[axis], hi[axis]
                ),
            ));
        }
    }
    Ok(plane)
}

/// The `ofgpu-automesher <cfg> -stopAfter castellate` config of the
/// placeholder (SPEC-LIT 158.10): castellation only, with staircase walls and
/// no snapped point, so the membrane plane's faces stay exactly planar. The
/// settings are the W4 supervisor's verified ones (29,662 cells): `stl_file`
/// as the one input surface, the extent and base size of the file, two
/// distance bands on the patch `vessel` (`band` at `level`, `outerBand` at
/// `outerLevel`), `keep_region` `seed` with the seed point
/// `x_lo + 0.9137 (x_hi - x_lo)`, `y_lo + 0.5137 (y_hi - y_lo)`,
/// `z_lo + 0.5137 (z_hi - z_lo)` - outside the vessel, in the open air - and
/// the case written to `case_dir` under the name `rembe`.
pub fn automesher_config(g: &VesselGeometry, stl_file: &str, case_dir: &str) -> serde_json::Value {
    let e = g.domain.extent;
    let d = &g.domain;
    let seed = [
        e[0] + 0.9137 * (e[1] - e[0]),
        e[2] + 0.5137 * (e[3] - e[2]),
        e[4] + 0.5137 * (e[5] - e[4]),
    ];
    serde_json::json!({
        "input": { "surfaces": [ { "path": stl_file } ] },
        "domain": { "extent": e, "base_size": d.base_size, "grading": [1.0, 1.0, 1.0] },
        "refinement": {
            "levels": [ {
                "patch": "vessel",
                "bands": [
                    { "distance": d.band, "level": d.level },
                    { "distance": d.outer_band, "level": d.outer_level }
                ]
            } ],
            "feature_angle_deg": 30.0,
            "max_level": d.level
        },
        "castellation": { "keep_region": "seed", "seed_point": seed, "min_faces": 4 },
        "snap": {
            "iterations": 30, "tolerance": 1e-3, "smoothing_passes": 3, "smoothing": 0.5,
            "undo_limit": 4, "max_area_ratio": 4.0, "feature_tolerance": 0.5
        },
        "layers": {
            "patches": [], "n": 0, "first_thickness": 0.01, "growth": 1.3,
            "min_thickness": 0.1, "medial_frac": 0.5
        },
        "quality": {
            "max_closure": 1e-10, "max_non_orth_deg": 70.0, "report_non_orth_deg": 60.0,
            "min_thickness_ratio": 0.05, "max_cond": 1e4
        },
        "output": { "case_dir": case_dir, "name": "rembe" }
    })
}

/// The cells reachable from `seed` through INTERNAL faces that are not in
/// `membrane` (face ids), breadth first, in ascending ids (SPEC-LIT 158.10):
/// the cells a sealed vessel's closed membrane keeps together. An out-of-range
/// `seed` reaches nothing.
pub fn seal_reach(hm: &HostMesh, membrane: &[usize], seed: usize) -> Vec<usize> {
    if seed >= hm.n_cells {
        return Vec::new();
    }
    let sealed: std::collections::HashSet<usize> = membrane.iter().copied().collect();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); hm.n_cells];
    for f in 0..hm.n_internal_faces {
        if sealed.contains(&f) {
            continue;
        }
        let (a, b) = (hm.owner[f] as usize, hm.neighbour[f] as usize);
        adj[a].push(b);
        adj[b].push(a);
    }
    let mut seen = vec![false; hm.n_cells];
    seen[seed] = true;
    let mut queue = std::collections::VecDeque::from([seed]);
    while let Some(c) = queue.pop_front() {
        for &q in &adj[c] {
            if !seen[q] {
                seen[q] = true;
                queue.push_back(q);
            }
        }
    }
    (0..hm.n_cells).filter(|&c| seen[c]).collect()
}

#[cfg(test)]
mod tests {
    use crate as ofgpu;
    include!("../../tests/shared/dust_gatemesh_cases.rs");
}
