// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Finite-volume mesh geometry: face area vectors and centroids, cell volumes
//! and centroids, the linear interpolation weight, and the over-relaxed
//! non-orthogonal decomposition.
//!
//! Written from:
//!   H. Jasak, "Error Analysis and Estimation for the Finite Volume Method
//!     with Applications to Fluid Flows", PhD thesis, Imperial College London
//!     (1996), sections 3.2, 3.3.1 and 3.4.2
//!   F. Moukalled, L. Mangani and M. Darwish, "The Finite Volume Method in
//!     Computational Fluid Dynamics", Springer (2016), sections 6.4 and 8.6.4
//!   J. H. Ferziger and M. Peric, "Computational Methods for Fluid Dynamics",
//!     3rd ed., Springer (2002), section 8.6
//!   ofgpu SPEC-LIT.md section 2, which carries those citations
//! No GPL-licensed source was consulted.
//!
//! # Why the passes are in this order
//!
//! Each quantity needs the one before it, and nothing here can be reordered
//! without silently changing the answer:
//!
//! 1. **Faces.** Every face - internal and boundary alike - gets its area
//!    vector and centroid by triangulating about the vertex average
//!    (SPEC-LIT 2.1). Nothing else is known yet, and nothing else is needed.
//! 2. **Pyramid apex.** A pass over *all* faces averaging each cell's face
//!    centroids. This is only an estimate of the cell centre, but it must be a
//!    point inside the cell, and the exact centre is not available yet - which
//!    is the whole reason the decomposition is done about an estimate.
//! 3. **Volumes and centroids** by pyramid decomposition about that apex
//!    (SPEC-LIT 2.2). Exact for a polyhedron with planar faces, whatever apex
//!    is chosen, because the divergence theorem does not care where it sits.
//! 4. **Weights and delta coefficients** (SPEC-LIT 2.3, 2.4), which need the
//!    cell centroids from step 3 and so cannot be folded into the face pass.
//! 5. **Boundary metrics** last: the wall distance `nf . d` and, for a cyclic
//!    couple, the pairing that turns two boundary faces into one internal one.
//!
//! This sweep and `topology.rs` are the only places in the crate that scatter
//! into a per-cell array. They are allowed to because they run once, on the
//! host, at setup; every device kernel gathers instead.

use crate::error::{Error, Result};
use crate::mesh::{HostMesh, MeshReport, PatchKind};
use crate::{Label, Scalar, Vec3};

/// Lower bound on `nf . d`, as a fraction of `|d|`, before it is inverted.
///
/// SPEC-LIT section 2.4, after Jasak (1996) section 3.4.2: the floor bounds
/// the implicit part of the Laplacian on a mesh whose non-orthogonality
/// approaches (or exceeds) 90 degrees. `0.05` corresponds to about 87 degrees,
/// well past the point where the answer is trustworthy; it only keeps the
/// coefficient finite.
const NON_ORTH_FLOOR: Scalar = 0.05;

/// Smallest magnitude this module will divide by.
///
/// Roughly the square root of the smallest normal `Scalar`, so that `x/SMALL`
/// cannot overflow for any `x` a mesh coordinate can hold. Only genuinely
/// degenerate geometry - a zero-area face, a collapsed cell, two coincident
/// cell centres - ever reaches it, and every site that does says what it falls
/// back to.
#[cfg(feature = "single")]
const SMALL: Scalar = 1.0e-19;
#[cfg(not(feature = "single"))]
const SMALL: Scalar = 1.0e-150;

// ==========================================================================
//  2.1  Face centroid and area
// ==========================================================================

/// Area vector and centroid of one polygonal face, by decomposition into
/// triangles about the average of its vertices (SPEC-LIT 2.1).
///
/// A general face of a polyhedral mesh is *not* planar, so there is no single
/// exact answer; the median decomposition below is the standard one. For a
/// triangle it reduces to the exact centroid - the three sub-triangles about
/// the vertex average have equal area and their centroids average back to it -
/// and for a planar convex polygon it is exact, because the centroid of a
/// union is the area-weighted mean of the parts' centroids and a triangle's
/// centroid is exact.
///
/// `Sf` is the vector sum of the triangle normals, so it is exactly the flux
/// area even when the face is warped. The scalar area accumulated alongside is
/// unsigned and used only to weight the centroid.
///
/// Returns `(Sf, Cf)`. The caller has already checked every vertex index.
fn face_geometry(verts: &[Label], points: &[Vec3]) -> (Vec3, Vec3) {
    let n = verts.len();
    if n == 0 {
        return (Vec3::ZERO, Vec3::ZERO);
    }

    let mut x_avg = Vec3::ZERO;
    for &v in verts {
        x_avg += points[v as usize];
    }
    x_avg = x_avg / n as Scalar;

    // Fewer than three vertices spans no area at all: there is nothing to
    // triangulate, and the centroid is the vertex average by definition.
    if n < 3 {
        return (Vec3::ZERO, x_avg);
    }

    let mut sf = Vec3::ZERO;
    let mut cf = Vec3::ZERO;
    let mut area: Scalar = 0.0;

    for i in 0..n {
        let a = points[verts[i] as usize];
        let b = points[verts[(i + 1) % n] as usize];

        // Twice the sub-triangle's area vector, and its centroid.
        let t_n = (a - x_avg).cross(b - x_avg);
        let t_c = (x_avg + a + b) / 3.0;
        let t_a = t_n.mag() * 0.5;

        sf += t_n * 0.5;
        cf += t_c * t_a;
        area += t_a;
    }

    if area > SMALL {
        (sf, cf / area)
    } else {
        // Degenerate face: the area weighting is meaningless, so fall back to
        // the vertex average. `Sf` is still the (near-zero) triangle sum.
        (sf, x_avg)
    }
}

// ==========================================================================
//  Validation
// ==========================================================================

/// Everything indexed by the sweep is checked here, once, so that a corrupt
/// mesh becomes an `Error::Mesh` naming the offending entity rather than a
/// panic somewhere in the middle of a pass.
fn validate(m: &HostMesh, points: &[Vec3], faces: &[Vec<Label>]) -> Result<()> {
    let (n_cells, n_if, n_bf) = (m.n_cells, m.n_internal_faces, m.n_boundary_faces);
    let n_faces = n_if + n_bf;

    // Face and cell ids travel to the device as `Label`. A mesh that cannot be
    // addressed there cannot be solved, and the casts below would wrap.
    if n_faces > Label::MAX as usize || n_cells > Label::MAX as usize {
        return Err(Error::Mesh(format!(
            "compute_geometry: {n_cells} cells and {n_faces} faces do not fit \
             in a {}-bit label",
            8 * std::mem::size_of::<Label>()
        )));
    }

    if m.owner.len() != n_if || m.neighbour.len() != n_if {
        return Err(Error::Mesh(format!(
            "compute_geometry: owner/neighbour hold {}/{} entries, expected {}",
            m.owner.len(),
            m.neighbour.len(),
            n_if
        )));
    }
    if m.b_face_cells.len() != n_bf {
        return Err(Error::Mesh(format!(
            "compute_geometry: bFaceCells holds {} entries, expected {}",
            m.b_face_cells.len(),
            n_bf
        )));
    }
    if faces.len() != n_faces {
        return Err(Error::Mesh(format!(
            "compute_geometry: {} face vertex lists for {} internal + {} \
             boundary faces",
            faces.len(),
            n_if,
            n_bf
        )));
    }

    for f in 0..n_if {
        for (what, c) in [("owner", m.owner[f]), ("neighbour", m.neighbour[f])] {
            if c < 0 || c as usize >= n_cells {
                return Err(Error::Mesh(format!(
                    "compute_geometry: internal face {f} has {what} {c}, \
                     outside [0, {n_cells})"
                )));
            }
        }
    }
    for bf in 0..n_bf {
        let c = m.b_face_cells[bf];
        if c < 0 || c as usize >= n_cells {
            return Err(Error::Mesh(format!(
                "compute_geometry: boundary face {bf} belongs to cell {c}, \
                 outside [0, {n_cells})"
            )));
        }
    }
    for (f, fv) in faces.iter().enumerate() {
        for &v in fv.iter() {
            if v < 0 || v as usize >= points.len() {
                return Err(Error::Mesh(format!(
                    "compute_geometry: face {f} refers to point {v} but there \
                     are {} points",
                    points.len()
                )));
            }
        }
    }

    Ok(())
}

/// Boundary face -> boundary face across every cyclic couple; `-1` elsewhere.
///
/// The pairing is by position within the patch: face `k` of a cyclic patch
/// couples to face `k` of the patch it names. That ordering is the contract a
/// cyclic pair carries, and the polyMesh reader derives `b_nbr_cell` from the
/// same rule.
fn cyclic_pairing(m: &HostMesh) -> Result<Vec<Label>> {
    let n_bf = m.n_boundary_faces;
    let mut pair = vec![-1 as Label; n_bf];

    for (p, pi) in m.patches.iter().enumerate() {
        if pi.kind != PatchKind::Cyclic {
            continue;
        }
        // A cyclic patch that names no neighbour is not coupled yet; it
        // behaves as an ordinary boundary until something names its partner.
        let Some(np) = pi.nbr_patch else { continue };

        if np == p {
            return Err(Error::Mesh(format!(
                "compute_geometry: cyclic patch '{}' names itself as its \
                 neighbour",
                pi.name
            )));
        }
        let Some(pn) = m.patches.get(np) else {
            return Err(Error::Mesh(format!(
                "compute_geometry: cyclic patch '{}' names patch {} of {}",
                pi.name,
                np,
                m.patches.len()
            )));
        };
        if pn.size != pi.size {
            return Err(Error::Mesh(format!(
                "compute_geometry: cyclic patches '{}' ({} faces) and '{}' \
                 ({} faces) differ in size",
                pi.name, pi.size, pn.name, pn.size
            )));
        }
        if pi.start + pi.size > n_bf || pn.start + pn.size > n_bf {
            return Err(Error::Mesh(format!(
                "compute_geometry: cyclic patches '{}' and '{}' run past the \
                 {n_bf} boundary faces",
                pi.name, pn.name
            )));
        }

        for k in 0..pi.size {
            pair[pi.start + k] = (pn.start + k) as Label;
        }
    }

    Ok(pair)
}

// ==========================================================================
//  The sweep
// ==========================================================================

/// Fill every geometric array of `m` from raw `points` and face vertex lists.
///
/// `faces` is the polyMesh face list: `n_internal_faces` internal faces first,
/// then the boundary faces in flattened patch order, so boundary face `bf` is
/// global face `n_internal_faces + bf`.
///
/// The topological boundary arrays (`b_kind`, `b_patch`, `b_nbr_cell`,
/// `b_weights`) are derived from `patches` only when they are not already
/// sized, so a caller that has built them itself - the polyMesh reader does -
/// keeps its own. The cyclic *weights* are geometry, and are always written.
pub fn compute(m: &mut HostMesh, points: &[Vec3], faces: &[Vec<Label>]) -> Result<()> {
    validate(m, points, faces)?;

    let (n_cells, n_if, n_bf) = (m.n_cells, m.n_internal_faces, m.n_boundary_faces);
    let n_faces = n_if + n_bf;

    m.n_points = points.len();

    // ---- 1. every face's area vector and centroid -------------------------
    let mut f_sf = vec![Vec3::ZERO; n_faces];
    let mut f_cf = vec![Vec3::ZERO; n_faces];
    for f in 0..n_faces {
        let (sf, cf) = face_geometry(&faces[f], points);
        f_sf[f] = sf;
        f_cf[f] = cf;
    }

    // ---- 2. pyramid apex: the average of each cell's face centroids -------
    // Only an estimate of the cell centre. It has to lie inside the cell, and
    // for a convex polyhedron the mean of the face centroids does.
    let mut apex = vec![Vec3::ZERO; n_cells];
    let mut n_cell_faces = vec![0u32; n_cells];

    for f in 0..n_if {
        for c in [m.owner[f] as usize, m.neighbour[f] as usize] {
            apex[c] += f_cf[f];
            n_cell_faces[c] += 1;
        }
    }
    for bf in 0..n_bf {
        let c = m.b_face_cells[bf] as usize;
        apex[c] += f_cf[n_if + bf];
        n_cell_faces[c] += 1;
    }

    for c in 0..n_cells {
        if n_cell_faces[c] == 0 {
            return Err(Error::Mesh(format!(
                "compute_geometry: cell {c} has no faces, so it is not a cell"
            )));
        }
        apex[c] = apex[c] / n_cell_faces[c] as Scalar;
    }

    // ---- 3. volumes and centroids by pyramid decomposition ----------------
    // V = (1/3) sum_f (s Sf) . (Cf - apex) is the divergence theorem applied
    // face by face: exact for planar faces, and independent of the apex for a
    // closed cell. The pyramid centroid sits three quarters of the way from
    // the apex to the base centroid.
    let mut v = vec![0.0 as Scalar; n_cells];
    let mut c_acc = vec![Vec3::ZERO; n_cells];

    {
        let mut add_pyramid = |cell: usize, s: Scalar, sf: Vec3, cf: Vec3| {
            let a = apex[cell];
            let v_pyr = (sf * s).dot(cf - a) / 3.0;
            v[cell] += v_pyr;
            c_acc[cell] += (cf * 0.75 + a * 0.25) * v_pyr;
        };

        for f in 0..n_if {
            add_pyramid(m.owner[f] as usize, 1.0, f_sf[f], f_cf[f]);
            add_pyramid(m.neighbour[f] as usize, -1.0, f_sf[f], f_cf[f]);
        }
        for bf in 0..n_bf {
            add_pyramid(
                m.b_face_cells[bf] as usize,
                1.0,
                f_sf[n_if + bf],
                f_cf[n_if + bf],
            );
        }
    }

    let mut c = vec![Vec3::ZERO; n_cells];
    for cell in 0..n_cells {
        // A non-positive volume is a broken mesh, not an error to return here:
        // `check`/`print_report` exist to say so, and they cannot run if the
        // sweep refuses to finish. The apex is the best centre available.
        c[cell] = if v[cell] > SMALL {
            c_acc[cell] / v[cell]
        } else {
            apex[cell]
        };
    }

    m.v = v;
    m.c = c;

    // ---- 4. internal face weights and delta coefficients ------------------
    let mut mag_sf = vec![0.0 as Scalar; n_if];
    let mut weights = vec![0.0 as Scalar; n_if];
    let mut delta_coeffs = vec![0.0 as Scalar; n_if];
    let mut non_orth_corr = vec![Vec3::ZERO; n_if];

    for f in 0..n_if {
        let p = m.owner[f] as usize;
        let nb = m.neighbour[f] as usize;
        let sf = f_sf[f];
        let cf = f_cf[f];

        mag_sf[f] = sf.mag();
        weights[f] = interp_weight(sf, cf, m.c[p], m.c[nb]);

        let (delta, k) = non_orth_split(sf, m.c[nb] - m.c[p]);
        delta_coeffs[f] = delta;
        non_orth_corr[f] = k;
    }

    m.sf = f_sf[..n_if].to_vec();
    m.cf = f_cf[..n_if].to_vec();
    m.mag_sf = mag_sf;
    m.weights = weights;
    m.delta_coeffs = delta_coeffs;
    m.non_orth_corr = non_orth_corr;

    // ---- 5. boundary metrics, cyclic couples last -------------------------
    let pair = cyclic_pairing(m)?;

    if m.b_kind.len() != n_bf {
        m.b_kind = vec![PatchKind::Generic as Label; n_bf];
        for pi in m.patches.iter() {
            for k in 0..pi.size.min(n_bf.saturating_sub(pi.start)) {
                m.b_kind[pi.start + k] = pi.kind as Label;
            }
        }
    }
    if m.b_patch.len() != n_bf {
        m.b_patch = vec![-1; n_bf];
        for (p, pi) in m.patches.iter().enumerate() {
            for k in 0..pi.size.min(n_bf.saturating_sub(pi.start)) {
                m.b_patch[pi.start + k] = p as Label;
            }
        }
    }
    if m.b_nbr_cell.len() != n_bf {
        let nbr: Vec<Label> = (0..n_bf)
            .map(|bf| {
                if pair[bf] >= 0 {
                    m.b_face_cells[pair[bf] as usize]
                } else {
                    -1
                }
            })
            .collect();
        m.b_nbr_cell = nbr;
    }
    // An uncoupled boundary face interpolates to the boundary value itself, so
    // the owner's weight is 1 (SPEC-LIT section 2). Cyclic faces overwrite
    // theirs from the geometry below.
    if m.b_weights.len() != n_bf {
        m.b_weights = vec![1.0; n_bf];
    }

    let mut b_sf = vec![Vec3::ZERO; n_bf];
    let mut b_mag_sf = vec![0.0 as Scalar; n_bf];
    let mut b_cf = vec![Vec3::ZERO; n_bf];
    let mut b_delta_coeffs = vec![0.0 as Scalar; n_bf];
    let mut b_non_orth_corr = vec![Vec3::ZERO; n_bf];
    let mut b_y = vec![0.0 as Scalar; n_bf];

    for bf in 0..n_bf {
        let sf = f_sf[n_if + bf];
        let cf = f_cf[n_if + bf];
        let p = m.b_face_cells[bf] as usize;

        b_sf[bf] = sf;
        b_mag_sf[bf] = sf.mag();
        b_cf[bf] = cf;

        // The owner-side offset: cell centre to its own face centre. Its
        // projection on the face normal is the wall-normal distance a wall
        // function means by `y`, and it is defined the same way on every
        // patch, coupled or not.
        //
        // *DESIGN*: the projection carries the same `0.05 |d|` floor as
        // `Delta`, so that a wall function dividing by `y` cannot produce an
        // infinity. On any usable mesh the floor is inactive, and on an
        // uncoupled patch `b_y == 1/b_delta_coeffs` exactly.
        let d_own = cf - m.c[p];
        let nf = sf.normalised();
        b_y[bf] = floor_along(nf.dot(d_own), d_own);

        if pair[bf] >= 0 {
            // Cyclic. The couple is one internal face folded in half, so the
            // separation is measured through both halves:
            //
            //     d = (Cf_own - C_P) + (C_N - Cf_nbr)
            //
            // which equals `C_N + s - C_P` for the transform `s` that maps the
            // neighbour patch onto this one, without ever having to know `s`.
            let nbr = pair[bf] as usize;
            let n_cell = m.b_face_cells[nbr] as usize;
            let d_nbr = m.c[n_cell] - f_cf[n_if + nbr];
            let d = d_own + d_nbr;

            // SPEC-LIT 2.4's over-relaxed split, applied to the SAME `d` that
            // spans the periodic image: a cyclic pair is, geometrically, one
            // internal face folded in half, so it gets the identical
            // treatment an internal face gets and no more. Before this, only
            // `Delta` (`.0`) was kept and the explicit correction `k` (`.1`)
            // was silently dropped, so a cyclic face on a sheared mesh lost
            // its non-orthogonal correction even though the internal faces
            // right next to it kept theirs.
            let (delta, k) = non_orth_split(sf, d);
            b_delta_coeffs[bf] = delta;
            b_non_orth_corr[bf] = k;
            // Both offsets are projected on the OWNER-side `Sf`, which is the
            // face the weight is used on.
            m.b_weights[bf] = weight_from_offsets(sf.dot(d_own).abs(), sf.dot(d_nbr).abs());
        } else {
            b_delta_coeffs[bf] = non_orth_split(sf, d_own).0;
            // Left at zero: an uncoupled boundary face has no neighbour cell
            // to interpolate `psi` from, so `snGrad` there is not the
            // internal-face formula this correction belongs to - SPEC-LIT
            // section 4's `(fr, psi_ref, g_ref)` triple is evaluated directly
            // on the face instead.
        }
    }

    m.b_sf = b_sf;
    m.b_mag_sf = b_mag_sf;
    m.b_cf = b_cf;
    m.b_delta_coeffs = b_delta_coeffs;
    m.b_non_orth_corr = b_non_orth_corr;
    m.b_y = b_y;

    Ok(())
}

// ==========================================================================
//  2.3, 2.4  the per-face coefficients
// ==========================================================================

/// SPEC-LIT 2.3: the weight that places the interpolated value where the face
/// plane cuts the line `P-N` (Jasak 1996 section 3.3.1).
///
/// The absolute values are the stabilisation the specification calls for on a
/// mesh whose non-orthogonality exceeds 90 degrees, where the signed products
/// change sign and the weight would leave `[0, 1]`.
#[inline]
fn interp_weight(sf: Vec3, cf: Vec3, c_p: Vec3, c_n: Vec3) -> Scalar {
    weight_from_offsets(sf.dot(cf - c_p).abs(), sf.dot(c_n - cf).abs())
}

/// `w = d_N / (d_P + d_N)`, the owner's share.
///
/// Two coincident centres leave nothing to weight; a half-and-half split is
/// the only unbiased answer and keeps the interpolation a convex combination.
#[inline]
fn weight_from_offsets(d_p: Scalar, d_n: Scalar) -> Scalar {
    let sum = d_p + d_n;
    if sum > SMALL {
        d_n / sum
    } else {
        0.5
    }
}

/// Apply the SPEC-LIT 2.4 floor to a projection along `d`.
#[inline]
fn floor_along(proj: Scalar, d: Vec3) -> Scalar {
    proj.max(NON_ORTH_FLOOR * d.mag())
}

/// SPEC-LIT 2.4: the over-relaxed non-orthogonal split, returning
/// `(Delta, k)`.
///
/// ```text
/// Delta = 1 / max(nf . d, 0.05 |d|)
/// k     = nf - d Delta
/// snGrad(psi)|_f = Delta (psi_N - psi_P) + k . (grad psi)_f
/// ```
///
/// The split is over-relaxed in the sense of Jasak (1996) section 3.4.2: `Sf`
/// is written as `alpha d + k_vec` with `alpha = |Sf|^2 / (Sf . d)`, the
/// choice that makes the *correction* orthogonal to the area vector,
/// `k_vec . Sf = 0`. Dividing through by `|Sf|` gives the normalised pair
/// above, and indeed `k . nf = 1 - (nf . d) Delta = 0` whenever the floor is
/// inactive.
///
/// (SPEC-LIT annotates `k` with `k . d = 0`. That is the property of the
/// *minimum-correction* split, `alpha = (Sf . d)/(d . d)`, not of the formula
/// the same section writes down; the formula is implemented as written. Both
/// splits agree on the limit the section states - on an orthogonal mesh
/// `k = 0` and `Delta = 1/|d|` - and the over-relaxed one is the one that
/// grows the implicit coefficient as non-orthogonality rises, which is what
/// keeps the deferred correction of SPEC-LIT 3.2 convergent.)
#[inline]
fn non_orth_split(sf: Vec3, d: Vec3) -> (Scalar, Vec3) {
    let nf = sf.normalised();
    let denom = floor_along(nf.dot(d), d);

    // Coincident cell centres, or a zero-area face: there is no direction to
    // difference along. A zero coefficient drops the face from the operator,
    // which is the only finite thing to do, and the underlying breakage shows
    // up in `check` as a collapsed cell or a closure failure.
    if denom <= SMALL {
        return (0.0, Vec3::ZERO);
    }

    let delta = 1.0 / denom;
    (delta, nf - d * delta)
}

// ==========================================================================
//  Checking
// ==========================================================================

/// Measure the mesh. Infallible by contract - this is what reports a broken
/// mesh, so it has to survive one, including one whose arrays were never
/// filled at all.
pub fn check(m: &HostMesh) -> MeshReport {
    let n_cells = m.n_cells.min(m.v.len());

    // ---- volumes ----------------------------------------------------------
    let mut total_volume: Scalar = 0.0;
    let mut min_volume = Scalar::INFINITY;
    let mut max_volume = Scalar::NEG_INFINITY;
    let mut min_volume_cell = 0usize;

    for cell in 0..n_cells {
        let v = m.v[cell];
        total_volume += v;
        if v < min_volume {
            min_volume = v;
            min_volume_cell = cell;
        }
        max_volume = max_volume.max(v);
    }
    if n_cells == 0 {
        min_volume = 0.0;
        max_volume = 0.0;
    }

    // ---- non-orthogonality, internal faces --------------------------------
    // The angle between the face normal and the line joining the two cell
    // centres. *DESIGN*: boundary faces are left out - an uncoupled one has no
    // second cell, so there is no line to take an angle to, and including only
    // the cyclic ones would make the mean depend on how the domain was cut.
    let n_if = m
        .n_internal_faces
        .min(m.owner.len())
        .min(m.neighbour.len())
        .min(m.sf.len());

    let mut max_non_orth_deg: Scalar = 0.0;
    let mut sum_non_orth = 0.0f64;
    let mut n_non_orth = 0usize;

    for f in 0..n_if {
        let (p, nb) = (m.owner[f] as usize, m.neighbour[f] as usize);
        let (Some(&c_p), Some(&c_n)) = (m.c.get(p), m.c.get(nb)) else {
            continue;
        };
        let d = c_n - c_p;
        let mag_d = d.mag();
        let nf = m.sf[f].normalised();
        if mag_d <= SMALL || nf.mag_sqr() <= 0.0 {
            continue;
        }

        let deg = (nf.dot(d) / mag_d).clamp(-1.0, 1.0).acos().to_degrees();
        max_non_orth_deg = max_non_orth_deg.max(deg);
        sum_non_orth += deg as f64;
        n_non_orth += 1;
    }

    let mean_non_orth_deg = if n_non_orth > 0 {
        (sum_non_orth / n_non_orth as f64) as Scalar
    } else {
        0.0
    };

    // ---- face closure -----------------------------------------------------
    // `sum_f s Sf = 0` for a closed cell, exactly, in exact arithmetic. It is
    // the single best indicator that the face winding is right: one face wound
    // backwards flips one term and the residual jumps to O(|Sf|).
    let mut closure = vec![Vec3::ZERO; n_cells];

    for f in 0..n_if {
        let (p, nb) = (m.owner[f] as usize, m.neighbour[f] as usize);
        if let Some(s) = closure.get_mut(p) {
            *s += m.sf[f];
        }
        if let Some(s) = closure.get_mut(nb) {
            *s -= m.sf[f];
        }
    }
    let n_bf = m
        .n_boundary_faces
        .min(m.b_face_cells.len())
        .min(m.b_sf.len());
    for bf in 0..n_bf {
        if let Some(s) = closure.get_mut(m.b_face_cells[bf] as usize) {
            *s += m.b_sf[bf];
        }
    }

    let mut max_closure_error: Scalar = 0.0;
    let mut max_closure_cell = 0usize;
    for (cell, cl) in closure.iter().enumerate() {
        // V^(2/3) is the cell's natural area scale, which makes the ratio
        // dimensionless and comparable between a coarse mesh and a fine one.
        let scale = (m.v[cell].abs() as f64)
            .max(f64::MIN_POSITIVE)
            .powf(2.0 / 3.0);
        let e = ((cl.mag() as f64) / scale) as Scalar;
        if e > max_closure_error {
            max_closure_error = e;
            max_closure_cell = cell;
        }
    }

    // ---- upper-triangular ordering ----------------------------------------
    // owner < neighbour, and faces sorted by (owner, neighbour). The LDU
    // addressing and every gather kernel assume it.
    let mut ldu_ordered =
        m.owner.len() == m.n_internal_faces && m.neighbour.len() == m.n_internal_faces;
    for f in 0..n_if {
        let ascending = f == 0
            || m.owner[f] > m.owner[f - 1]
            || (m.owner[f] == m.owner[f - 1] && m.neighbour[f] > m.neighbour[f - 1]);
        if m.owner[f] >= m.neighbour[f] || !ascending {
            ldu_ordered = false;
            break;
        }
    }

    MeshReport {
        total_volume,
        min_volume,
        max_volume,
        min_volume_cell,
        max_non_orth_deg,
        mean_non_orth_deg,
        max_closure_error,
        max_closure_cell,
        ldu_ordered,
    }
}

/// Above this the mesh does not close and the run should not have started.
const CLOSURE_LIMIT: Scalar = 1.0e-10;
/// Above this the non-orthogonal correction dominates the implicit part, and
/// the deferred iteration of SPEC-LIT 3.2 converges slowly if at all.
const NON_ORTH_WARN_DEG: Scalar = 70.0;

/// Print the mesh summary, then anything wrong with it.
///
/// The report goes to stdout and the diagnoses to stderr, deliberately: the
/// report is data a run logs, while an error has to show up in the terminal
/// even when stdout has been redirected to a file.
pub fn print_report(m: &HostMesh) {
    let r = check(m);

    println!(
        "mesh: {} cells, {} internal faces, {} boundary faces, {} points",
        m.n_cells, m.n_internal_faces, m.n_boundary_faces, m.n_points
    );

    println!("  patches ({}):", m.patches.len());
    for (i, p) in m.patches.iter().enumerate() {
        println!(
            "    {:>3}  {:<24} {:<16} {:<9} {:>8} faces  [{}, {})",
            i,
            p.name,
            p.type_name,
            p.kind.as_str(),
            p.size,
            p.start,
            p.start + p.size
        );
    }

    println!(
        "  volume: total {:.6e}, min {:.6e} (cell {}), max {:.6e}",
        r.total_volume, r.min_volume, r.min_volume_cell, r.max_volume
    );
    println!(
        "  non-orthogonality: max {:.3} deg, mean {:.3} deg",
        r.max_non_orth_deg, r.mean_non_orth_deg
    );
    println!(
        "  face closure: max |sum Sf| / V^(2/3) = {:.3e} (cell {})",
        r.max_closure_error, r.max_closure_cell
    );
    println!(
        "  lduAddressing: {}",
        if r.ldu_ordered {
            "upper-triangular"
        } else {
            "NOT upper-triangular"
        }
    );

    if m.n_cells > 0 && r.min_volume <= 0.0 {
        eprintln!(
            "[ofgpu] mesh error: cell {} has volume {:.6e}; a non-positive \
             volume means the cell is inside out or collapsed",
            r.min_volume_cell, r.min_volume
        );
    }
    if r.max_closure_error > CLOSURE_LIMIT {
        eprintln!(
            "[ofgpu] mesh error: cell {} does not close, |sum Sf|/V^(2/3) = \
             {:.3e}; the face winding is wrong",
            r.max_closure_cell, r.max_closure_error
        );
    }
    if !r.ldu_ordered {
        eprintln!(
            "[ofgpu] mesh error: faces are not in upper-triangular \
             (owner, neighbour) order; every gather kernel mis-addresses"
        );
    }
    if r.max_non_orth_deg > NON_ORTH_WARN_DEG {
        eprintln!(
            "[ofgpu] mesh warning: maximum non-orthogonality {:.1} deg; the \
             explicit correction will dominate and the solution may need extra \
             non-orthogonal correctors",
            r.max_non_orth_deg
        );
    }
}

// ==========================================================================
//  Tests
//
//  Nothing here compares against another solver. Every expected number is
//  either an analytic property of a shape written out by hand (a hexahedron,
//  a tetrahedron), or an identity the discretisation must satisfy exactly
//  (closure, k . nf = 0, w = 0.5 by symmetry).
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::topology::tests::box_mesh;
    use crate::mesh::PatchInfo;

    // Tolerances are per precision: the identities below are exact in exact
    // arithmetic, so what is being allowed for is round-off and nothing else.
    #[cfg(feature = "single")]
    const TOL: Scalar = 1.0e-5;
    #[cfg(not(feature = "single"))]
    const TOL: Scalar = 1.0e-12;

    /// `|sum_f s Sf| / V^(2/3)`, which cancels catastrophically and so is
    /// judged against the working precision rather than against `TOL`.
    #[cfg(feature = "single")]
    const CLOSURE_TOL: Scalar = 1.0e-5;
    #[cfg(not(feature = "single"))]
    const CLOSURE_TOL: Scalar = 1.0e-13;

    /// Degrees. An angle goes through `acos`, which loses half the mantissa
    /// near zero.
    #[cfg(feature = "single")]
    const ANGLE_TOL: Scalar = 1.0e-3;
    #[cfg(not(feature = "single"))]
    const ANGLE_TOL: Scalar = 1.0e-9;

    fn close(a: Scalar, b: Scalar) -> bool {
        (a - b).abs() <= TOL * (1.0 as Scalar).max(b.abs())
    }

    /// `box_mesh` plus the geometry sweep, panicking on a mesh error - which
    /// in a test is exactly what should happen.
    fn built(n: [usize; 3], d: Vec3) -> HostMesh {
        let (mut m, points, faces) = box_mesh(n, d);
        m.compute_geometry(&points, &faces).expect("geometry");
        m
    }

    /// Stretch a box in x by `x -> L (x/L)^p`. Every face stays planar and
    /// every cell stays a rectangular hexahedron, just a different width, so
    /// the analytic answers stay closed-form while the weights stop being 1/2.
    fn grade_x(points: &mut [Vec3], l: Scalar, p: Scalar) {
        for q in points.iter_mut() {
            q.x = l * (q.x / l).powf(p);
        }
    }

    /// `x -> x + s y`. A linear map, so faces stay planar and the determinant
    /// is 1: volumes are unchanged and the mesh is uniformly non-orthogonal.
    fn shear_x_with_y(points: &mut [Vec3], s: Scalar) {
        for q in points.iter_mut() {
            q.x += s * q.y;
        }
    }

    // ---- 2.2  volumes and centroids --------------------------------------

    #[test]
    fn hexahedra_get_their_analytic_volume_and_centroid() {
        let (nx, ny, nz) = (3usize, 2usize, 2usize);
        let d = Vec3::new(0.5, 0.25, 2.0);
        let m = built([nx, ny, nz], d);

        let cell_v = d.x * d.y * d.z;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = i + nx * (j + ny * k);
                    assert!(close(m.v[c], cell_v), "cell {c} volume {}", m.v[c]);

                    let want = Vec3::new(
                        (i as Scalar + 0.5) * d.x,
                        (j as Scalar + 0.5) * d.y,
                        (k as Scalar + 0.5) * d.z,
                    );
                    assert!(
                        (m.c[c] - want).mag() <= TOL,
                        "cell {c} centre {} wanted {want}",
                        m.c[c]
                    );
                }
            }
        }

        let r = m.check();
        let total = nx as Scalar * d.x * ny as Scalar * d.y * nz as Scalar * d.z;
        assert!(close(r.total_volume, total), "total {}", r.total_volume);
        assert!(close(r.min_volume, cell_v));
        assert!(close(r.max_volume, cell_v));
    }

    /// A single tetrahedron: four triangular faces, no internal face at all.
    /// The volume and centroid are known exactly, and the triangulation of
    /// section 2.1 is exact on a triangle, so this pins the pyramid
    /// decomposition on something that is not a box.
    #[test]
    fn a_tetrahedron_is_exact() {
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        // Wound so that every Sf points out of the tet.
        let faces: Vec<Vec<Label>> = vec![
            vec![0, 2, 1],
            vec![0, 1, 3],
            vec![0, 3, 2],
            vec![1, 2, 3],
        ];

        let mut m = HostMesh {
            n_cells: 1,
            n_internal_faces: 0,
            n_boundary_faces: 4,
            b_face_cells: vec![0; 4],
            patches: vec![PatchInfo {
                name: "walls".to_string(),
                type_name: "wall".to_string(),
                kind: PatchKind::Wall,
                start: 0,
                size: 4,
                nbr_patch: None,
            }],
            ..Default::default()
        };
        m.compute_geometry(&points, &faces).expect("geometry");

        assert!(close(m.v[0], 1.0 / 6.0), "volume {}", m.v[0]);
        assert!(
            (m.c[0] - Vec3::new(0.25, 0.25, 0.25)).mag() <= TOL,
            "centroid {}",
            m.c[0]
        );

        // Face 3 is the slanted one; its centroid is the vertex average and
        // its outward normal is (1,1,1)/sqrt(3), so the cell centre stands
        // 0.25/sqrt(3) off it.
        assert!(close(m.b_y[3], 0.25 / (3.0 as Scalar).sqrt()), "y {}", m.b_y[3]);
        assert!(close(m.b_y[0], 0.25));

        let r = m.check();
        assert!(r.max_closure_error <= CLOSURE_TOL, "closure {}", r.max_closure_error);
        // No internal faces, so there is no angle to report and nothing to
        // order.
        assert_eq!(r.max_non_orth_deg, 0.0);
        assert!(r.ldu_ordered);
    }

    // ---- 2.3  weights ----------------------------------------------------

    #[test]
    fn a_uniform_mesh_weights_both_sides_equally() {
        let m = built([3, 2, 2], Vec3::new(0.5, 0.25, 2.0));

        for f in 0..m.n_internal_faces {
            assert!(close(m.weights[f], 0.5), "face {f} weight {}", m.weights[f]);
        }
    }

    /// The point of the weight is that it follows the geometry. On a graded
    /// mesh the owner's share is the *neighbour's* width over the sum of the
    /// two, because the face sits closer to the smaller cell.
    #[test]
    fn a_graded_mesh_gets_the_weight_the_geometry_demands() {
        let nx = 4usize;
        let d = Vec3::new(0.25, 0.7, 0.4);
        let l = nx as Scalar * d.x;

        let (mut m, mut points, faces) = box_mesh([nx, 1, 1], d);
        grade_x(&mut points, l, 2.0);
        m.compute_geometry(&points, &faces).expect("geometry");

        // Node positions and the widths they imply.
        let node = |i: usize| l * ((i as Scalar * d.x) / l).powf(2.0);
        let h = |i: usize| node(i + 1) - node(i);

        for i in 0..nx {
            assert!(close(m.v[i], h(i) * d.y * d.z), "cell {i} volume {}", m.v[i]);
            assert!(close(m.c[i].x, 0.5 * (node(i) + node(i + 1))));
        }

        // Only x-normal internal faces exist on a 4x1x1 block, and sorted by
        // (owner, neighbour) face f joins cell f to cell f+1.
        assert_eq!(m.n_internal_faces, nx - 1);
        for f in 0..m.n_internal_faces {
            let want = h(f + 1) / (h(f) + h(f + 1));
            assert!(close(m.weights[f], want), "face {f}: {} vs {want}", m.weights[f]);
            assert!(
                (m.weights[f] - 0.5).abs() > 0.05,
                "face {f} weight came out at 1/2 on a graded mesh"
            );
        }
        // Spelled out for the first face so a plausible-but-wrong formula
        // cannot pass by agreeing with itself: widths 1/16 and 3/16.
        assert!(close(m.weights[0], 0.75));
    }

    // ---- 2.4  the non-orthogonal split -----------------------------------

    #[test]
    fn an_orthogonal_mesh_needs_no_correction() {
        let d = Vec3::new(0.5, 0.25, 2.0);
        let m = built([3, 2, 2], d);

        for f in 0..m.n_internal_faces {
            let dd = m.c[m.neighbour[f] as usize] - m.c[m.owner[f] as usize];

            assert!(
                m.non_orth_corr[f].mag() <= TOL,
                "face {f}: k = {} on an orthogonal mesh",
                m.non_orth_corr[f]
            );
            // SPEC-LIT 2.4: Delta degenerates to 1/|d| when nf is along d.
            assert!(
                close(m.delta_coeffs[f], 1.0 / dd.mag()),
                "face {f}: Delta {} vs 1/|d| {}",
                m.delta_coeffs[f],
                1.0 / dd.mag()
            );
        }

        let r = m.check();
        assert!(r.max_non_orth_deg <= ANGLE_TOL, "angle {}", r.max_non_orth_deg);
        assert!(r.mean_non_orth_deg <= ANGLE_TOL);
    }

    /// A uniform shear is the cleanest non-orthogonal mesh there is: every
    /// internal face makes the same known angle, so the reported maximum and
    /// mean are both `atan(s)` and can be checked against the analytic value
    /// rather than against a previous run.
    #[test]
    fn a_sheared_mesh_reports_the_analytic_non_orthogonality() {
        let d = Vec3::new(0.5, 0.25, 2.0);
        let s: Scalar = 0.3;

        let (mut m, mut points, faces) = box_mesh([3, 3, 1], d);
        shear_x_with_y(&mut points, s);
        m.compute_geometry(&points, &faces).expect("geometry");

        // The map is linear with unit determinant, so the volumes survive it
        // and the centroids move with it.
        for j in 0..3 {
            for i in 0..3 {
                let c = i + 3 * j;
                assert!(close(m.v[c], d.x * d.y * d.z), "cell {c} volume {}", m.v[c]);
                let y = (j as Scalar + 0.5) * d.y;
                let want = Vec3::new((i as Scalar + 0.5) * d.x + s * y, y, 0.5 * d.z);
                assert!((m.c[c] - want).mag() <= TOL, "cell {c} centre {}", m.c[c]);
            }
        }

        let want_deg = s.atan().to_degrees();
        let r = m.check();
        assert!(
            (r.max_non_orth_deg - want_deg).abs() <= ANGLE_TOL,
            "max {} vs atan(s) {want_deg}",
            r.max_non_orth_deg
        );
        assert!(
            (r.mean_non_orth_deg - want_deg).abs() <= ANGLE_TOL,
            "mean {} vs {want_deg}",
            r.mean_non_orth_deg
        );
        assert!(r.max_closure_error <= CLOSURE_TOL, "closure {}", r.max_closure_error);

        for f in 0..m.n_internal_faces {
            let nf = m.sf[f].normalised();
            let dd = m.c[m.neighbour[f] as usize] - m.c[m.owner[f] as usize];
            let k = m.non_orth_corr[f];

            // The over-relaxed split puts the whole correction orthogonal to
            // the area vector and leaves the implicit part exact along it.
            assert!(k.dot(nf).abs() <= TOL, "face {f}: k . nf = {}", k.dot(nf));
            assert!(
                close(m.delta_coeffs[f] * nf.dot(dd), 1.0),
                "face {f}: Delta (nf.d) = {}",
                m.delta_coeffs[f] * nf.dot(dd)
            );
            assert!(k.mag() > 0.1, "face {f}: no correction on a sheared mesh");
        }
    }

    // ---- closure and ordering ---------------------------------------------

    #[test]
    fn cells_close_to_round_off() {
        for n in [[3usize, 2, 2], [1, 1, 1], [5, 1, 3]] {
            let m = built(n, Vec3::new(0.5, 0.25, 2.0));
            let r = m.check();
            assert!(
                r.max_closure_error <= CLOSURE_TOL,
                "{n:?}: closure {} at cell {}",
                r.max_closure_error,
                r.max_closure_cell
            );
            assert!(r.ldu_ordered);
        }
    }

    /// Closure is the winding check: reverse one face and the residual jumps
    /// from round-off to order one.
    #[test]
    fn a_face_wound_backwards_shows_up_in_the_closure_error() {
        let (mut m, points, mut faces) = box_mesh([3, 2, 2], Vec3::new(0.5, 0.25, 2.0));
        let nif = m.n_internal_faces;
        faces[nif].reverse();
        m.compute_geometry(&points, &faces).expect("geometry");

        let r = m.check();
        assert!(
            r.max_closure_error > 1.0e-3,
            "a reversed face went unnoticed: {}",
            r.max_closure_error
        );
        assert_eq!(r.max_closure_cell, m.b_face_cells[0] as usize);
    }

    #[test]
    fn swapped_addressing_is_reported_as_out_of_order() {
        let mut m = built([3, 2, 2], Vec3::new(0.5, 0.25, 2.0));
        assert!(m.check().ldu_ordered);

        m.owner.swap(0, 1);
        m.neighbour.swap(0, 1);
        assert!(!m.check().ldu_ordered, "an out-of-order pair went unnoticed");

        let mut m = built([3, 2, 2], Vec3::new(0.5, 0.25, 2.0));
        std::mem::swap(&mut m.owner[0], &mut m.neighbour[0]);
        assert!(!m.check().ldu_ordered, "owner > neighbour went unnoticed");
    }

    /// `check` is what reports a broken mesh, so it has to survive one that
    /// was never built at all rather than panicking on an empty array.
    #[test]
    fn check_survives_an_empty_mesh() {
        let r = HostMesh::default().check();
        assert_eq!(r.total_volume, 0.0);
        assert_eq!(r.min_volume, 0.0);
        assert_eq!(r.max_closure_error, 0.0);
        assert!(r.ldu_ordered);
    }

    // ---- boundary metrics -------------------------------------------------

    #[test]
    fn boundary_faces_carry_the_half_cell_distance() {
        let (nx, ny, nz) = (3usize, 2usize, 2usize);
        let d = Vec3::new(0.5, 0.25, 2.0);
        let m = built([nx, ny, nz], d);

        // box_mesh orders the patches xmin xmax ymin ymax zmin zmax.
        let span = [d.x, d.x, d.y, d.y, d.z, d.z];
        let area = [d.y * d.z, d.y * d.z, d.x * d.z, d.x * d.z, d.x * d.y, d.x * d.y];

        for (p, pi) in m.patches.iter().enumerate() {
            assert!(pi.size > 0);
            for bf in pi.start..pi.start + pi.size {
                assert!(close(m.b_y[bf], 0.5 * span[p]), "bf {bf}: y {}", m.b_y[bf]);
                assert!(
                    close(m.b_delta_coeffs[bf], 2.0 / span[p]),
                    "bf {bf}: Delta {}",
                    m.b_delta_coeffs[bf]
                );
                assert!(close(m.b_mag_sf[bf], area[p]));
                // Uncoupled: the face value is its own interpolation.
                assert!(close(m.b_weights[bf], 1.0));
                assert_eq!(m.b_nbr_cell[bf], -1);
                assert_eq!(m.b_patch[bf], p as Label);
                assert_eq!(m.b_kind[bf], pi.kind as Label);

                // Sf points out of the domain, so it lies along +/- one axis.
                let nf = m.b_sf[bf].normalised();
                let sign: Scalar = if p % 2 == 0 { -1.0 } else { 1.0 };
                let axis = match p / 2 {
                    0 => Vec3::new(sign, 0.0, 0.0),
                    1 => Vec3::new(0.0, sign, 0.0),
                    _ => Vec3::new(0.0, 0.0, sign),
                };
                assert!((nf - axis).mag() <= TOL, "bf {bf} normal {nf}");
            }
        }
    }

    /// A cyclic couple is one internal face folded in half: the separation has
    /// to be measured through both halves, or the coupled coefficient comes
    /// out twice what it should be and the weight 1 instead of 1/2.
    #[test]
    fn a_cyclic_couple_measures_through_both_halves() {
        let (nx, ny, nz) = (3usize, 2usize, 2usize);
        let d = Vec3::new(0.5, 0.25, 2.0);
        let (mut m, points, faces) = box_mesh([nx, ny, nz], d);

        for (p, nbr) in [(0usize, 1usize), (1, 0)] {
            m.patches[p].kind = PatchKind::Cyclic;
            m.patches[p].type_name = "cyclic".to_string();
            m.patches[p].nbr_patch = Some(nbr);
        }
        m.compute_geometry(&points, &faces).expect("geometry");

        let cell = |i: usize, j: usize, k: usize| (i + nx * (j + ny * k)) as Label;

        for k in 0..nz {
            for j in 0..ny {
                let idx = j + ny * k;
                let lo = m.patches[0].start + idx;
                let hi = m.patches[1].start + idx;

                assert_eq!(m.b_nbr_cell[lo], cell(nx - 1, j, k));
                assert_eq!(m.b_nbr_cell[hi], cell(0, j, k));

                for bf in [lo, hi] {
                    // d runs half a cell out of one side and half a cell into
                    // the other, so |d| is a whole cell.
                    assert!(
                        close(m.b_delta_coeffs[bf], 1.0 / d.x),
                        "bf {bf}: Delta {} vs {}",
                        m.b_delta_coeffs[bf],
                        1.0 / d.x
                    );
                    assert!(close(m.b_weights[bf], 0.5), "bf {bf}: w {}", m.b_weights[bf]);
                    // y stays the owner-side distance, which is half a cell.
                    assert!(close(m.b_y[bf], 0.5 * d.x));
                    assert_eq!(m.b_kind[bf], PatchKind::Cyclic as Label);
                    // The cyclic couple is axis-aligned, so `nf` and `d` are
                    // exactly collinear and `k = nf - d*Delta` cancels to
                    // exactly zero, not merely close to it - this is what an
                    // uncorrected cyclic face on a SHEARED mesh (below) fails
                    // to do.
                    assert_eq!(
                        m.b_non_orth_corr[bf],
                        Vec3::ZERO,
                        "bf {bf}: b_non_orth_corr should be exactly zero on an \
                         orthogonal cyclic couple, got {}",
                        m.b_non_orth_corr[bf]
                    );
                }
            }
        }

        // The uncoupled patches are untouched by the pairing.
        let ymin = &m.patches[2];
        for bf in ymin.start..ymin.start + ymin.size {
            assert_eq!(m.b_nbr_cell[bf], -1);
            assert!(close(m.b_weights[bf], 1.0));
            assert!(close(m.b_delta_coeffs[bf], 2.0 / d.y));
        }
    }

    /// The cyclic patches sit on `x = 0` and `x = Lx`, and `shear_x_with_y`
    /// moves a point's `x` by `s*y` only - identical for every point at a
    /// given `y`, including a face's own centroid and the cell centre behind
    /// it. So the shear tilts every x-normal face's `Sf`, exactly as it does
    /// for `a_sheared_mesh_reports_the_analytic_non_orthogonality`, while
    /// leaving `d` exactly axis-aligned - for the cyclic couple no less than
    /// for an ordinary internal face: `d_own`/`d_nbr` are each a face
    /// centroid minus a cell centre at the SAME `(j, k)`, so the shear
    /// cancels out of them exactly.
    ///
    /// The couple is therefore geometrically two mirrored copies of the same
    /// internal x-face translated along x - translation does not change a
    /// tilt - so its correction vector `k` must equal the internal face's,
    /// up to the sign flip that comes from the `x = 0` patch's `Sf` pointing
    /// the opposite way. Before the fix in this module, both patches simply
    /// read zero, having been skipped in the coupled branch entirely.
    #[test]
    fn a_sheared_cyclic_couple_gets_the_same_correction_as_an_internal_face() {
        let (nx, ny, nz) = (3usize, 2usize, 1usize);
        let d = Vec3::new(0.5, 0.25, 2.0);
        let s: Scalar = 0.3;

        let (mut m, mut points, faces) = box_mesh([nx, ny, nz], d);
        for (p, nbr) in [(0usize, 1usize), (1, 0)] {
            m.patches[p].kind = PatchKind::Cyclic;
            m.patches[p].type_name = "cyclic".to_string();
            m.patches[p].nbr_patch = Some(nbr);
        }
        shear_x_with_y(&mut points, s);
        m.compute_geometry(&points, &faces).expect("geometry");

        let cell = |i: usize, j: usize, k: usize| (i + nx * (j + ny * k)) as Label;

        for k in 0..nz {
            for j in 0..ny {
                let idx = j + ny * k;
                let lo = m.patches[0].start + idx;
                let hi = m.patches[1].start + idx;

                // Not the degenerate orthogonal case any more: shearing must
                // actually have moved the correction away from zero, or this
                // test is not exercising SPEC-LIT 2.4 at all.
                assert!(
                    m.b_non_orth_corr[hi].mag() > 1.0e-3,
                    "k, j={j}: b_non_orth_corr[{hi}] is {} - the shear left it \
                     at zero",
                    m.b_non_orth_corr[hi]
                );

                // An internal x-face between the two cells at the SAME (j, k)
                // - any one will do, since the tilt does not depend on i.
                let (io, in_) = (cell(0, j, k) as usize, cell(1, j, k) as usize);
                let f_internal = (0..m.n_internal_faces)
                    .find(|&f| m.owner[f] as usize == io && m.neighbour[f] as usize == in_)
                    .expect("an internal x-face joins cell 0 and cell 1 at this (j, k)");

                let want = m.non_orth_corr[f_internal];
                assert!(
                    (m.b_non_orth_corr[hi] - want).mag() <= TOL,
                    "j={j}: hi patch (x = Lx, Sf pointing +x like the internal \
                     face) got {}, internal face gave {}",
                    m.b_non_orth_corr[hi],
                    want
                );
                // The x = 0 patch's `Sf` points the other way, so its `nf`
                // and `d` are both the internal face's negated - and `k` is
                // linear in each, so it comes out negated twice, i.e. equal
                // to the internal face's `k` with ITS OWN sign flipped once.
                assert!(
                    (m.b_non_orth_corr[lo] + want).mag() <= TOL,
                    "j={j}: lo patch (x = 0, Sf pointing -x) got {}, wanted {}",
                    m.b_non_orth_corr[lo],
                    -want
                );
            }
        }
    }

    // ---- a corrupt mesh is named, not panicked on --------------------------

    #[test]
    fn a_short_face_list_is_an_error_not_a_panic() {
        let (mut m, points, mut faces) = box_mesh([2, 1, 1], Vec3::new(1.0, 1.0, 1.0));
        faces.pop();
        let e = m.compute_geometry(&points, &faces).unwrap_err();
        assert!(matches!(e, Error::Mesh(_)), "{e}");
    }

    #[test]
    fn a_point_index_past_the_end_is_an_error_not_a_panic() {
        let (mut m, points, mut faces) = box_mesh([2, 1, 1], Vec3::new(1.0, 1.0, 1.0));
        faces[0][0] = points.len() as Label;
        let e = m.compute_geometry(&points, &faces).unwrap_err();
        assert!(matches!(e, Error::Mesh(_)), "{e}");
    }

    #[test]
    fn addressing_outside_the_cell_range_is_an_error_not_a_panic() {
        let (mut m, points, faces) = box_mesh([2, 1, 1], Vec3::new(1.0, 1.0, 1.0));
        let mut bad = m.clone();
        bad.owner[0] = 99;
        assert!(bad.compute_geometry(&points, &faces).is_err());

        let mut bad = m.clone();
        bad.b_face_cells[0] = -1;
        assert!(bad.compute_geometry(&points, &faces).is_err());

        // The good one still works, so the failures above are the mutation.
        assert!(m.compute_geometry(&points, &faces).is_ok());
    }

    #[test]
    fn a_cyclic_patch_that_names_the_wrong_partner_is_an_error() {
        let (mut m, points, faces) = box_mesh([2, 2, 2], Vec3::new(1.0, 1.0, 1.0));
        m.patches[0].kind = PatchKind::Cyclic;
        m.patches[0].nbr_patch = Some(0);
        assert!(m.clone().compute_geometry(&points, &faces).is_err());

        // xmin has 4 faces, ymin has 4 as well on a 2x2x2 block, so pick a
        // pairing that genuinely differs in size: xmin against a patch that
        // has been shortened.
        m.patches[0].nbr_patch = Some(2);
        m.patches[2].size -= 1;
        assert!(m.compute_geometry(&points, &faces).is_err());
    }
}
