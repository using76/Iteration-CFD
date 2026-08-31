// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Inverting the LDU addressing into the two cell -> face CSR maps.
//!
//! Provenance: original. The cell -> face CSR inversion is this project's own
//! design; see SPEC-LIT.md section 1. No GPL-licensed source was consulted.
//!
//! The textbook way to assemble a matrix is to *scatter* over faces
//! (`diag[owner[f]] -= ...`). On a GPU that needs `atomicAdd` on `f64`, which
//! is both slow and non-deterministic in its rounding. Paying once, here, to
//! invert `owner`/`neighbour` lets every device kernel *gather* instead: one
//! thread per cell, no atomics, bitwise reproducible.
//!
//! This is the only file besides the geometry sweep that scatters
//! at all, and it is allowed to because it runs once, on the host, during
//! setup.
//!
//! Since SPEC-LIT §70 it also builds a THIRD map, `rf_offset`/`rf_face`/
//! `rf_flags`, which merges the two into one list per cell ordered by the
//! **global** face id. The two maps above order a row by the LOCAL id, which
//! is a property of how the mesh was cut up rather than of the mesh: cut an
//! internal face and it becomes a boundary face on both sides, moving its term
//! from one list to the other and renumbering the rest. Floating-point
//! addition is not associative, so that moves the bits of `A psi`. The merged
//! map is what the matrix-vector product, the relaxation and the value pinning
//! walk; on an undecomposed mesh it is bit-for-bit the two old lists
//! concatenated, and [`row_face_map`] carries the argument for why it must be.

use crate::mesh::HostMesh;
use crate::Label;

/// `Some(cell)` when `l` addresses a real cell.
///
/// Out-of-range addressing is a corrupt mesh; see the note on
/// [`build_cell_face_maps`] for why it is skipped rather than returned.
#[inline]
fn cell_of(l: Label, n_cells: usize) -> Option<usize> {
    if l >= 0 && (l as usize) < n_cells {
        Some(l as usize)
    } else {
        None
    }
}

/// Offsets go to the device as `Label`, so a mesh whose face count overflows
/// `i32` cannot be represented at all. Saturating keeps the CSR self-consistent
/// (the tail cells simply gather nothing) instead of wrapping negative, which
/// would index a device array out of bounds.
#[inline]
fn to_label(n: usize) -> Label {
    if n > Label::MAX as usize {
        Label::MAX
    } else {
        n as Label
    }
}

/// Build `cf_offset`/`cf_face`/`cf_own` and `bcf_offset`/`bcf_face`.
///
/// Counting pass, prefix sum, fill pass, for each of the two maps. The fill
/// pass walks faces in ascending index and appends, so every cell's slice
/// comes out sorted by face id without a separate sort - and that fixed order
/// is exactly what makes the gathered `Amul` bitwise reproducible run to run.
///
/// Out-of-range addressing is a hard error, not an abort. `HostMesh::build_cell_
/// face_maps` is declared infallible in `mesh.rs`, which this module may not
/// change, so a bad face is instead dropped from both passes (leaving the CSR
/// internally consistent, just short) and reported on stderr. The fallible
/// `geometry::compute` validates the same indices properly and returns
/// `Error::Mesh`, so on the normal setup path a corrupt mesh is still a hard
/// failure rather than a silent one.
pub fn build_cell_face_maps(m: &mut HostMesh) {
    let n_cells = m.n_cells;

    // Declared counts size the device-visible arrays; the effective counts
    // bound the loops. They differ only on a mesh that is already broken.
    let n_if_decl = m.n_internal_faces;
    let n_bf_decl = m.n_boundary_faces;
    let n_if = n_if_decl.min(m.owner.len()).min(m.neighbour.len());
    let n_bf = n_bf_decl.min(m.b_face_cells.len());

    if n_if != n_if_decl {
        eprintln!(
            "[ofgpu] mesh error: owner/neighbour hold {}/{} entries, expected {}",
            m.owner.len(),
            m.neighbour.len(),
            n_if_decl
        );
    }
    if n_bf != n_bf_decl {
        eprintln!(
            "[ofgpu] mesh error: bFaceCells has {} entries, expected {}",
            m.b_face_cells.len(),
            n_bf_decl
        );
    }

    // ---- cell -> internal face CSR ---------------------------------------
    // Every internal face appears twice, once for its owner and once for its
    // neighbour, so a complete CSR holds exactly 2*n_internal_faces entries.

    // Counted one slot to the right so the prefix sum below turns counts into
    // starts in place, the way the C++ does.
    let mut off = vec![0usize; n_cells + 1];
    let mut dropped = 0usize;

    for f in 0..n_if {
        match (cell_of(m.owner[f], n_cells), cell_of(m.neighbour[f], n_cells)) {
            (Some(o), Some(n)) => {
                off[o + 1] += 1;
                off[n + 1] += 1;
            }
            _ => dropped += 1,
        }
    }

    for c in 0..n_cells {
        let prev = off[c];
        off[c + 1] += prev;
    }

    m.cf_face = vec![-1; 2 * n_if_decl];
    m.cf_own = vec![0; 2 * n_if_decl];

    let mut cursor = off[..n_cells].to_vec();

    for f in 0..n_if {
        let (o, n) = match (cell_of(m.owner[f], n_cells), cell_of(m.neighbour[f], n_cells)) {
            (Some(o), Some(n)) => (o, n),
            _ => continue,
        };

        let io = cursor[o];
        m.cf_face[io] = f as Label;
        m.cf_own[io] = 1;
        cursor[o] = io + 1;

        let inb = cursor[n];
        m.cf_face[inb] = f as Label;
        m.cf_own[inb] = 0;
        cursor[n] = inb + 1;
    }

    if off[n_cells] > Label::MAX as usize {
        eprintln!(
            "[ofgpu] mesh error: cell->face map has {} entries, more than a label holds",
            off[n_cells]
        );
    }

    m.cf_offset = off.iter().map(|&x| to_label(x)).collect();

    // ---- cell -> boundary face CSR ---------------------------------------
    // A boundary face belongs to exactly one cell, so this map holds
    // n_boundary_faces entries.
    let mut boff = vec![0usize; n_cells + 1];

    for bf in 0..n_bf {
        match cell_of(m.b_face_cells[bf], n_cells) {
            Some(c) => boff[c + 1] += 1,
            None => dropped += 1,
        }
    }

    for c in 0..n_cells {
        let prev = boff[c];
        boff[c + 1] += prev;
    }

    m.bcf_face = vec![-1; n_bf_decl];

    let mut bcursor = boff[..n_cells].to_vec();

    for bf in 0..n_bf {
        let c = match cell_of(m.b_face_cells[bf], n_cells) {
            Some(c) => c,
            None => continue,
        };

        let i = bcursor[c];
        m.bcf_face[i] = bf as Label;
        bcursor[c] = i + 1;
    }

    m.bcf_offset = boff.iter().map(|&x| to_label(x)).collect();

    if dropped > 0 {
        eprintln!(
            "[ofgpu] mesh error: {dropped} face(s) address a cell outside \
             [0, {n_cells}) and were dropped from the cell->face maps"
        );
    }

    // ---- and the merged, global-face-ordered row map (SPEC-LIT §70) ------
    let (rf_offset, rf_face, rf_flags) = row_face_map(
        n_cells,
        n_if_decl,
        n_bf_decl,
        &m.owner,
        &m.neighbour,
        &m.b_face_cells,
        &m.global_face,
    );
    m.rf_offset = rf_offset;
    m.rf_face = rf_face;
    m.rf_flags = rf_flags;
}

// ==========================================================================
//  The merged, GLOBAL-face-ordered row map - SPEC-LIT §70
// ==========================================================================

/// `rf_flags` bit 0: this cell is the face's OWNER.
///
/// Always set on a boundary face - a boundary face has exactly one adjacent
/// cell and that cell is its owner - so the bit only discriminates on an
/// internal face, where it chooses `upper`/`neighbour` over `lower`/`owner`.
///
/// Mirrored on the device as `OFGPU_RF_OWNS` in `cuda/ofgpu_device.cuh`.
pub const RF_OWNS: Label = 1;

/// `rf_flags` bit 1: this entry addresses the BOUNDARY arrays
/// (`boundary_coeffs`, `internal_coeffs`, `b_nbr_cell`) and its `rf_face` is a
/// boundary-face index. Clear means the internal-face arrays and an internal
/// face index.
///
/// Mirrored on the device as `OFGPU_RF_BOUNDARY` in `cuda/ofgpu_device.cuh`.
pub const RF_BOUNDARY: Label = 2;

/// Build `rf_offset`/`rf_face`/`rf_flags`: for every cell, ONE list of its
/// incident faces - internal and boundary together - in ascending **global**
/// face id.
///
/// SPEC-LIT §70. [`build_cell_face_maps`] above orders each cell's slice by
/// LOCAL face id, in two separate lists. That is a property of the partition,
/// not of the mesh: cut an internal face and it becomes a boundary face on
/// both sides, which moves its term from the end of the first list to the end
/// of the second and renumbers everything after it. Because floating-point
/// addition is not associative, a row summed in a different order is a
/// different number - so `A psi` moves in its bits under decomposition before
/// any collective exists. Keyed on the global id instead, the order is a
/// property of the mesh alone.
///
/// `global_face` is indexed by SLOT: slot `f` for internal face `f`, slot
/// `n_if_decl + bf` for boundary face `bf`, which is the polyMesh face
/// numbering. An empty or wrong-length array means the identity, and the
/// identity is what every undecomposed mesh has.
///
/// **On an undecomposed mesh the result is bit-for-bit the two old lists
/// concatenated**, because the identity map puts every internal face's key
/// below every boundary face's. That is what makes this refactor provably
/// free, and `merged_row_is_the_two_old_lists_concatenated` asserts it rather
/// than leaving it as an argument.
///
/// Returned rather than written into the mesh so that [`crate::mesh::GpuMesh::upload`]
/// can build the map for a `HostMesh` that never called
/// [`build_cell_face_maps`] - every mesh written out by hand in a test -
/// without cloning the mesh to do it.
pub fn row_face_map(
    n_cells: usize,
    n_if_decl: usize,
    n_bf_decl: usize,
    owner: &[Label],
    neighbour: &[Label],
    b_face_cells: &[Label],
    global_face: &[Label],
) -> (Vec<Label>, Vec<Label>, Vec<Label>) {
    let n_if = n_if_decl.min(owner.len()).min(neighbour.len());
    let n_bf = n_bf_decl.min(b_face_cells.len());
    let n_slot = n_if_decl + n_bf_decl;

    // A wrong-length map is the identity, not an error: this function is on
    // the infallible path (see the note on `build_cell_face_maps`), and a
    // decomposition that got the length wrong is caught where the rest of the
    // mesh is validated, in `geometry::compute`.
    let identity = global_face.len() != n_slot;
    let key = |s: usize| -> Label {
        if identity {
            s as Label
        } else {
            global_face[s]
        }
    };

    // ---- counting pass ---------------------------------------------------
    // An internal face is counted twice, once for each side; a boundary face
    // once. The same arithmetic as the two maps above, into one array.
    let mut off = vec![0usize; n_cells + 1];
    for f in 0..n_if {
        if let (Some(o), Some(n)) = (cell_of(owner[f], n_cells), cell_of(neighbour[f], n_cells)) {
            off[o + 1] += 1;
            off[n + 1] += 1;
        }
    }
    for &l in b_face_cells.iter().take(n_bf) {
        if let Some(c) = cell_of(l, n_cells) {
            off[c + 1] += 1;
        }
    }
    for c in 0..n_cells {
        let prev = off[c];
        off[c + 1] += prev;
    }

    let mut rf_face = vec![-1 as Label; 2 * n_if_decl + n_bf_decl];
    let mut rf_flags = vec![0 as Label; 2 * n_if_decl + n_bf_decl];
    let mut cursor = off[..n_cells].to_vec();

    // ---- the visit order -------------------------------------------------
    // Appending slots in ascending global id is what sorts every cell's slice
    // at once, so there is no per-cell sort. When the ids are already
    // ascending in slot order - every mesh this crate can currently read -
    // there is no sort at all, and the fill is literally the two old passes
    // run back to back, which is §70.3's argument in code.
    //
    // The sort is STABLE. The two halves of one cut face carry the SAME
    // global id; they never share a row, so the order between them is
    // arbitrary, and a stable sort makes "arbitrary" mean "the same every
    // time" rather than "whatever the pivot happened to choose".
    let ascending = identity || (1..n_slot).all(|s| key(s - 1) <= key(s));
    let mut order: Vec<usize> = Vec::new();
    if !ascending {
        order = (0..n_slot).collect();
        order.sort_by_key(|&s| key(s));
    }

    // Appending one slot. The two loops below differ only in where the slot
    // number comes from; the ascending case materialises no permutation at
    // all, which on a 40 M-cell mesh is a gigabyte of host memory not spent.
    let mut emit = |s: usize| {
        if s < n_if_decl {
            let f = s;
            if f >= n_if {
                return;
            }
            let (o, nb) = match (cell_of(owner[f], n_cells), cell_of(neighbour[f], n_cells)) {
                (Some(o), Some(nb)) => (o, nb),
                _ => return,
            };

            let io = cursor[o];
            rf_face[io] = f as Label;
            rf_flags[io] = RF_OWNS;
            cursor[o] = io + 1;

            let inb = cursor[nb];
            rf_face[inb] = f as Label;
            rf_flags[inb] = 0;
            cursor[nb] = inb + 1;
        } else {
            let bf = s - n_if_decl;
            if bf >= n_bf {
                return;
            }
            let c = match cell_of(b_face_cells[bf], n_cells) {
                Some(c) => c,
                None => return,
            };

            let i = cursor[c];
            rf_face[i] = bf as Label;
            rf_flags[i] = RF_BOUNDARY | RF_OWNS;
            cursor[c] = i + 1;
        }
    };

    if ascending {
        for s in 0..n_slot {
            emit(s);
        }
    } else {
        for &s in &order {
            emit(s);
        }
    }

    (
        off.iter().map(|&x| to_label(x)).collect(),
        rf_face,
        rf_flags,
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::mesh::{PatchInfo, PatchKind};
    use crate::{Scalar, Vec3};

    /// A structured `nx * ny * nz` box of `d`-sized hexahedra, in the polyMesh
    /// layout: faces internal-first in upper-triangular (owner, neighbour)
    /// order, then the six boundary patches in the order
    /// `xmin xmax ymin ymax zmin zmax`.
    ///
    /// Written out by hand rather than taken from `blockgen` so that a bug in
    /// the generator cannot make these tests agree with a broken geometry
    /// sweep. Winding follows the polyMesh convention: an internal face's points run so
    /// that `Sf` points from owner to neighbour, a boundary face's so that
    /// `Sf` points out of the domain.
    pub(crate) fn box_mesh(
        n: [usize; 3],
        d: Vec3,
    ) -> (HostMesh, Vec<Vec3>, Vec<Vec<Label>>) {
        let (nx, ny, nz) = (n[0], n[1], n[2]);
        let pt = |i: usize, j: usize, k: usize| -> Label {
            (i + (nx + 1) * (j + (ny + 1) * k)) as Label
        };
        let cell = |i: usize, j: usize, k: usize| -> Label { (i + nx * (j + ny * k)) as Label };

        let mut points = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    points.push(Vec3::new(
                        i as Scalar * d.x,
                        j as Scalar * d.y,
                        k as Scalar * d.z,
                    ));
                }
            }
        }

        // (owner, neighbour, face points)
        let mut internal: Vec<(Label, Label, Vec<Label>)> = Vec::new();

        for k in 0..nz {
            for j in 0..ny {
                for i in 1..nx {
                    // +x normal: edge +y then edge +z
                    internal.push((
                        cell(i - 1, j, k),
                        cell(i, j, k),
                        vec![
                            pt(i, j, k),
                            pt(i, j + 1, k),
                            pt(i, j + 1, k + 1),
                            pt(i, j, k + 1),
                        ],
                    ));
                }
            }
        }
        for k in 0..nz {
            for j in 1..ny {
                for i in 0..nx {
                    // +y normal: edge +z then edge +x
                    internal.push((
                        cell(i, j - 1, k),
                        cell(i, j, k),
                        vec![
                            pt(i, j, k),
                            pt(i, j, k + 1),
                            pt(i + 1, j, k + 1),
                            pt(i + 1, j, k),
                        ],
                    ));
                }
            }
        }
        for k in 1..nz {
            for j in 0..ny {
                for i in 0..nx {
                    // +z normal: edge +x then edge +y
                    internal.push((
                        cell(i, j, k - 1),
                        cell(i, j, k),
                        vec![
                            pt(i, j, k),
                            pt(i + 1, j, k),
                            pt(i + 1, j + 1, k),
                            pt(i, j + 1, k),
                        ],
                    ));
                }
            }
        }

        internal.sort_by_key(|&(o, nb, _)| (o, nb));

        // (face cell, face points), patch by patch
        let mut patch_faces: Vec<Vec<(Label, Vec<Label>)>> = Vec::new();

        let mut xmin = Vec::new();
        let mut xmax = Vec::new();
        for k in 0..nz {
            for j in 0..ny {
                xmin.push((
                    cell(0, j, k),
                    vec![
                        pt(0, j, k),
                        pt(0, j, k + 1),
                        pt(0, j + 1, k + 1),
                        pt(0, j + 1, k),
                    ],
                ));
                xmax.push((
                    cell(nx - 1, j, k),
                    vec![
                        pt(nx, j, k),
                        pt(nx, j + 1, k),
                        pt(nx, j + 1, k + 1),
                        pt(nx, j, k + 1),
                    ],
                ));
            }
        }
        patch_faces.push(xmin);
        patch_faces.push(xmax);

        let mut ymin = Vec::new();
        let mut ymax = Vec::new();
        for k in 0..nz {
            for i in 0..nx {
                ymin.push((
                    cell(i, 0, k),
                    vec![
                        pt(i, 0, k),
                        pt(i + 1, 0, k),
                        pt(i + 1, 0, k + 1),
                        pt(i, 0, k + 1),
                    ],
                ));
                ymax.push((
                    cell(i, ny - 1, k),
                    vec![
                        pt(i, ny, k),
                        pt(i, ny, k + 1),
                        pt(i + 1, ny, k + 1),
                        pt(i + 1, ny, k),
                    ],
                ));
            }
        }
        patch_faces.push(ymin);
        patch_faces.push(ymax);

        let mut zmin = Vec::new();
        let mut zmax = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                zmin.push((
                    cell(i, j, 0),
                    vec![
                        pt(i, j, 0),
                        pt(i, j + 1, 0),
                        pt(i + 1, j + 1, 0),
                        pt(i + 1, j, 0),
                    ],
                ));
                zmax.push((
                    cell(i, j, nz - 1),
                    vec![
                        pt(i, j, nz),
                        pt(i + 1, j, nz),
                        pt(i + 1, j + 1, nz),
                        pt(i, j + 1, nz),
                    ],
                ));
            }
        }
        patch_faces.push(zmin);
        patch_faces.push(zmax);

        let names = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"];
        let kinds = [
            PatchKind::Generic,
            PatchKind::Generic,
            PatchKind::Wall,
            PatchKind::Wall,
            PatchKind::Empty,
            PatchKind::Empty,
        ];

        let mut faces: Vec<Vec<Label>> = Vec::new();
        let mut owner = Vec::new();
        let mut neighbour = Vec::new();
        for (o, nb, fp) in internal {
            owner.push(o);
            neighbour.push(nb);
            faces.push(fp);
        }

        let mut b_face_cells = Vec::new();
        let mut patches = Vec::new();
        for (p, pf) in patch_faces.into_iter().enumerate() {
            let start = b_face_cells.len();
            let size = pf.len();
            for (c, fp) in pf {
                b_face_cells.push(c);
                faces.push(fp);
            }
            patches.push(PatchInfo {
                name: names[p].to_string(),
                type_name: match kinds[p] {
                    PatchKind::Wall => "wall".to_string(),
                    PatchKind::Empty => "empty".to_string(),
                    _ => "patch".to_string(),
                },
                kind: kinds[p],
                start,
                size,
                nbr_patch: None,
            });
        }

        let m = HostMesh {
            n_cells: nx * ny * nz,
            n_internal_faces: owner.len(),
            n_boundary_faces: b_face_cells.len(),
            n_points: points.len(),
            owner,
            neighbour,
            b_face_cells,
            patches,
            ..Default::default()
        };

        (m, points, faces)
    }

    fn built(n: [usize; 3], d: Vec3) -> HostMesh {
        let (mut m, _, _) = box_mesh(n, d);
        m.build_cell_face_maps();
        m
    }

    #[test]
    fn every_internal_face_appears_once_as_owner_and_once_as_neighbour() {
        let m = built([3, 2, 2], Vec3::new(0.5, 0.25, 2.0));

        assert_eq!(m.cf_offset.len(), m.n_cells + 1);
        assert_eq!(m.cf_offset[0], 0);
        assert_eq!(m.cf_offset[m.n_cells] as usize, 2 * m.n_internal_faces);
        assert_eq!(m.cf_face.len(), 2 * m.n_internal_faces);

        let mut as_owner = vec![0usize; m.n_internal_faces];
        let mut as_nbr = vec![0usize; m.n_internal_faces];

        for c in 0..m.n_cells {
            let (a, b) = (m.cf_offset[c] as usize, m.cf_offset[c + 1] as usize);
            for j in a..b {
                let f = m.cf_face[j];
                assert!(f >= 0 && (f as usize) < m.n_internal_faces, "face id {f}");
                let f = f as usize;
                if m.cf_own[j] == 1 {
                    assert_eq!(m.owner[f] as usize, c, "cf_own set on the wrong side");
                    as_owner[f] += 1;
                } else {
                    assert_eq!(m.neighbour[f] as usize, c, "cf_own clear on the wrong side");
                    as_nbr[f] += 1;
                }
            }
        }

        assert!(as_owner.iter().all(|&n| n == 1));
        assert!(as_nbr.iter().all(|&n| n == 1));
    }

    /// The gather order is what makes a run bitwise reproducible, so the
    /// within-cell ordering is a contract, not an accident.
    #[test]
    fn faces_are_ascending_within_every_cell() {
        let m = built([3, 2, 2], Vec3::new(0.5, 0.25, 2.0));

        for c in 0..m.n_cells {
            let (a, b) = (m.cf_offset[c] as usize, m.cf_offset[c + 1] as usize);
            for j in a + 1..b {
                assert!(
                    m.cf_face[j - 1] < m.cf_face[j],
                    "cell {c} gathers faces out of order"
                );
            }

            let (a, b) = (m.bcf_offset[c] as usize, m.bcf_offset[c + 1] as usize);
            for j in a + 1..b {
                assert!(m.bcf_face[j - 1] < m.bcf_face[j]);
            }
        }
    }

    #[test]
    fn boundary_csr_covers_every_boundary_face_once() {
        let m = built([3, 2, 2], Vec3::new(0.5, 0.25, 2.0));

        assert_eq!(m.bcf_offset[m.n_cells] as usize, m.n_boundary_faces);

        let mut seen = vec![0usize; m.n_boundary_faces];
        for c in 0..m.n_cells {
            let (a, b) = (m.bcf_offset[c] as usize, m.bcf_offset[c + 1] as usize);
            for j in a..b {
                let bf = m.bcf_face[j];
                assert!(bf >= 0);
                let bf = bf as usize;
                assert_eq!(m.b_face_cells[bf] as usize, c);
                seen[bf] += 1;
            }
        }
        assert!(seen.iter().all(|&n| n == 1));

        // 3x2x2 box: 2*(2*2) + 2*(3*2) + 2*(3*2) = 32 boundary faces, and the
        // interior faces are 2*2*2 + 3*1*2 + 3*2*1 = 20.
        assert_eq!(m.n_boundary_faces, 32);
        assert_eq!(m.n_internal_faces, 20);
    }

    // ----------------------------------------------------------------------
    //  The merged, global-face-ordered row map - SPEC-LIT §70
    // ----------------------------------------------------------------------

    /// The global id of the face an `(rf_face, rf_flags)` entry names, read
    /// back the way [`row_face_map`] keyed it.
    fn merged_key(m: &HostMesh, face: Label, flags: Label) -> Label {
        let slot = if flags & RF_BOUNDARY != 0 {
            m.n_internal_faces + face as usize
        } else {
            face as usize
        };
        if m.global_face.len() == m.n_internal_faces + m.n_boundary_faces {
            m.global_face[slot]
        } else {
            slot as Label
        }
    }

    /// SPEC-LIT §70.3, the by-construction claim, checked rather than
    /// believed: under the identity global map the merged slice IS the two old
    /// slices concatenated, face for face and flag for flag. That equality is
    /// the entire reason the refactor cannot move a bit, so it is asserted on
    /// meshes rather than argued once in prose.
    fn assert_merge_is_the_two_lists_concatenated(m: &HostMesh) {
        for c in 0..m.n_cells {
            let (a, b) = (m.rf_offset[c] as usize, m.rf_offset[c + 1] as usize);
            let (ia, ib) = (m.cf_offset[c] as usize, m.cf_offset[c + 1] as usize);
            let (ba, bb) = (m.bcf_offset[c] as usize, m.bcf_offset[c + 1] as usize);

            assert_eq!(
                b - a,
                (ib - ia) + (bb - ba),
                "cell {c}: merged row is {} long, the two old rows are {} + {}",
                b - a,
                ib - ia,
                bb - ba
            );

            for (k, j) in (ia..ib).enumerate() {
                assert_eq!(m.rf_face[a + k], m.cf_face[j], "cell {c}, internal slot {k}");
                assert_eq!(
                    m.rf_flags[a + k] & RF_BOUNDARY,
                    0,
                    "cell {c}, internal slot {k} is flagged boundary"
                );
                assert_eq!(
                    m.rf_flags[a + k] & RF_OWNS,
                    m.cf_own[j],
                    "cell {c}, internal slot {k} owns the wrong side"
                );
            }
            for (k, j) in (ba..bb).enumerate() {
                let s = a + (ib - ia) + k;
                assert_eq!(m.rf_face[s], m.bcf_face[j], "cell {c}, boundary slot {k}");
                assert_eq!(
                    m.rf_flags[s],
                    RF_BOUNDARY | RF_OWNS,
                    "cell {c}, boundary slot {k} flags"
                );
            }
        }
    }

    #[test]
    fn merged_row_is_the_two_old_lists_concatenated() {
        assert_merge_is_the_two_lists_concatenated(&built([3, 2, 2], Vec3::new(0.5, 0.25, 2.0)));
        assert_merge_is_the_two_lists_concatenated(&built([1, 1, 1], Vec3::new(1.0, 1.0, 1.0)));
        assert_merge_is_the_two_lists_concatenated(&built([4, 1, 1], Vec3::new(0.3, 1.0, 1.0)));
    }

    #[test]
    fn every_face_appears_exactly_once_in_the_merged_map() {
        let m = built([3, 2, 2], Vec3::new(0.5, 0.25, 2.0));

        assert_eq!(m.rf_offset.len(), m.n_cells + 1);
        assert_eq!(m.rf_offset[0], 0);
        assert_eq!(
            m.rf_offset[m.n_cells] as usize,
            2 * m.n_internal_faces + m.n_boundary_faces
        );
        assert_eq!(m.rf_face.len(), 2 * m.n_internal_faces + m.n_boundary_faces);
        assert_eq!(m.rf_flags.len(), m.rf_face.len());

        let mut as_owner = vec![0usize; m.n_internal_faces];
        let mut as_nbr = vec![0usize; m.n_internal_faces];
        let mut as_bnd = vec![0usize; m.n_boundary_faces];

        for c in 0..m.n_cells {
            for j in m.rf_offset[c] as usize..m.rf_offset[c + 1] as usize {
                let f = m.rf_face[j];
                assert!(f >= 0, "a live merged slot still holds the -1 fill");
                let f = f as usize;
                let fl = m.rf_flags[j];

                if fl & RF_BOUNDARY != 0 {
                    assert_eq!(fl, RF_BOUNDARY | RF_OWNS);
                    assert!(f < m.n_boundary_faces);
                    assert_eq!(m.b_face_cells[f] as usize, c);
                    as_bnd[f] += 1;
                } else {
                    assert!(f < m.n_internal_faces);
                    if fl & RF_OWNS != 0 {
                        assert_eq!(m.owner[f] as usize, c, "RF_OWNS on the wrong side");
                        as_owner[f] += 1;
                    } else {
                        assert_eq!(m.neighbour[f] as usize, c, "RF_OWNS clear on the wrong side");
                        as_nbr[f] += 1;
                    }
                }
            }
        }

        assert!(as_owner.iter().all(|&n| n == 1));
        assert!(as_nbr.iter().all(|&n| n == 1));
        assert!(as_bnd.iter().all(|&n| n == 1));
    }

    /// The ordering is the contract, so it is asserted directly rather than
    /// inferred from the concatenation test: within a row, ascending GLOBAL
    /// face id, whatever the local numbering says.
    #[test]
    fn the_merged_row_is_ascending_in_global_face_id() {
        let m = built([3, 2, 2], Vec3::new(0.5, 0.25, 2.0));

        for c in 0..m.n_cells {
            let (a, b) = (m.rf_offset[c] as usize, m.rf_offset[c + 1] as usize);
            for j in a + 1..b {
                let prev = merged_key(&m, m.rf_face[j - 1], m.rf_flags[j - 1]);
                let this = merged_key(&m, m.rf_face[j], m.rf_flags[j]);
                assert!(prev < this, "cell {c} gathers global faces out of order");
            }
        }
    }

    /// The general path. A `global_face` that is not ascending in slot order
    /// sends the builder through the sort, and the rows come out in the
    /// PERMUTED order rather than the local one. Nothing this repository can
    /// read produces such a map yet, which is exactly why it needs a test of
    /// its own rather than being covered incidentally.
    #[test]
    fn a_permuted_global_face_map_reorders_the_row() {
        let (mut m, _, _) = box_mesh([3, 2, 2], Vec3::new(0.5, 0.25, 2.0));
        let n_slot = m.n_internal_faces + m.n_boundary_faces;

        // Reverse every id. Every row's order must reverse with them.
        m.global_face = (0..n_slot).map(|s| (n_slot - 1 - s) as Label).collect();
        m.build_cell_face_maps();

        let ident = built([3, 2, 2], Vec3::new(0.5, 0.25, 2.0));

        for c in 0..m.n_cells {
            let (a, b) = (m.rf_offset[c] as usize, m.rf_offset[c + 1] as usize);
            for j in a + 1..b {
                let prev = merged_key(&m, m.rf_face[j - 1], m.rf_flags[j - 1]);
                let this = merged_key(&m, m.rf_face[j], m.rf_flags[j]);
                assert!(prev < this, "cell {c} is not sorted by the permuted id");
            }

            let (ia, ib) = (ident.rf_offset[c] as usize, ident.rf_offset[c + 1] as usize);
            assert_eq!(b - a, ib - ia, "cell {c} changed length");
            for k in 0..(b - a) {
                assert_eq!(
                    m.rf_face[a + k],
                    ident.rf_face[ib - 1 - k],
                    "cell {c}, slot {k}"
                );
                assert_eq!(m.rf_flags[a + k], ident.rf_flags[ib - 1 - k]);
            }
        }
    }

    #[test]
    fn an_out_of_range_face_is_dropped_from_the_merged_map_too() {
        let (mut m, _, _) = box_mesh([2, 1, 1], Vec3::new(1.0, 1.0, 1.0));
        m.owner[0] = 99;
        m.build_cell_face_maps();

        assert_eq!(m.rf_face.len(), 2 * m.n_internal_faces + m.n_boundary_faces);
        assert_eq!(
            m.rf_offset[m.n_cells] as usize,
            2 * (m.n_internal_faces - 1) + m.n_boundary_faces
        );

        for c in 0..m.n_cells {
            for j in m.rf_offset[c] as usize..m.rf_offset[c + 1] as usize {
                assert!(
                    m.rf_face[j] >= 0,
                    "a live merged slot still holds the -1 fill"
                );
            }
        }
    }

    /// A corrupt addressing must not panic and must not produce a CSR that
    /// points at a face id no kernel can read.
    #[test]
    fn out_of_range_addressing_is_dropped_not_panicked_on() {
        let (mut m, _, _) = box_mesh([2, 1, 1], Vec3::new(1.0, 1.0, 1.0));
        m.owner[0] = 99;
        m.build_cell_face_maps();

        assert_eq!(m.cf_face.len(), 2 * m.n_internal_faces);
        assert_eq!(m.cf_offset[m.n_cells] as usize, 2 * (m.n_internal_faces - 1));

        for c in 0..m.n_cells {
            let (a, b) = (m.cf_offset[c] as usize, m.cf_offset[c + 1] as usize);
            for j in a..b {
                assert!(m.cf_face[j] >= 0, "a live CSR slot still holds the -1 fill");
            }
        }
    }
}
