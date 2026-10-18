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
//! The specification was informed by a review of FSCDustFoam (Huang & Lipatnikov, GPL) and no code was copied.
//! No GPL-licensed source was consulted.

use crate::error::{Error, Result};
use crate::io::polymesh::{build_host_mesh, PolyMeshRaw};
use crate::mesh::{HostMesh, PatchInfo, PatchKind};
use crate::types::DVec3;
use crate::Label;

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

#[cfg(test)]
mod tests {
    use crate as ofgpu;
    include!("../../tests/shared/dust_gatemesh_cases.rs");
}
