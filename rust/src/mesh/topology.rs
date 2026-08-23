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
