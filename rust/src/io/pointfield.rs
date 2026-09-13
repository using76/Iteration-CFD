// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Cell-to-point inverse-distance interpolation and the point/cell
//! adjacency behind a `PointData` block, on the raw geometry the caller
//! keeps - SPEC-LIT §49.3's pattern: `HostMesh` retains `n_points` and
//! nothing else, so the point set and the face polygons arrive here as the
//! `PolyMeshRaw` that outlives the `HostMesh` built from it.
//!
//! Written from:
//!   H. Jasak, "Error Analysis and Estimation for the Finite Volume Method
//!     with Applications to Fluid Flows", PhD thesis, Imperial College (1996),
//!     Jasak §3.3.2 - the inverse-distance weight `w = 1/|d|` (SPEC-LIT §3.5)
//!   F. Moukalled, L. Mangani, M. Darwish, "The Finite Volume Method in
//!     Computational Fluid Dynamics", Springer (2016), §9.3 - the same weight
//!   ofgpu `SPEC-LIT.md` §1, §3.5, §49.3
//!
//! OpenFOAM's `volPointInterpolation` is GPL and was not opened; no other
//! cell-to-point implementation was consulted.
//!
//! No GPL-licensed source was consulted.

use std::ops::{Add, Mul};

use crate::error::{Error, Result};
use crate::io::polymesh::PolyMeshRaw;
use crate::mesh::HostMesh;
use crate::{Label, Scalar, Tensor, Vec3};

/// The point/cell adjacency of a raw polyMesh, with the inverse-distance
/// weights of SPEC-LIT §3.5 already normalised into it.
///
/// Two CSR tables, one per direction. `offset`/`cell`/`weight` map a POINT
/// to the cells that share it (ascending cell index, weights summing to
/// one); `cell_offset`/`cell_point` map a CELL to its distinct points
/// (ascending point index), which is the vertex-mean stencil
/// `point_to_cell` walks.
pub struct PointInterpolator {
    pub n_points: usize,
    pub n_cells: usize,
    /// `[n_points + 1]`  point -> cells CSR
    pub offset: Vec<Label>,
    /// ascending within a point
    pub cell: Vec<Label>,
    /// normalised, sums to 1 per point
    pub weight: Vec<Scalar>,
    /// `[n_cells + 1]`   cell -> distinct points CSR
    pub cell_offset: Vec<Label>,
    pub cell_point: Vec<Label>,
}

impl PointInterpolator {
    /// Builds the adjacency from the raw mesh, checking it against `m`
    /// instead of re-deriving anything: `n_cells` IS `m.n_cells` (which
    /// `build_host_mesh` derived as `max(owner ∪ neighbour) + 1`), and a
    /// point count or a cell label that disagrees with `m` is refused by
    /// name.
    pub fn new(raw: &PolyMeshRaw, m: &HostMesh) -> Result<Self> {
        if raw.points.len() != m.n_points {
            return Err(Error::Field {
                field: "points".to_string(),
                msg: format!(
                    "has {} value(s), mesh has {} point(s)",
                    raw.points.len(),
                    m.n_points
                ),
            });
        }
        let n_cells = m.n_cells;
        let n_points = raw.points.len();
        let n_if = raw.neighbour.len();

        let check_cell = |what: &str, c: Label| -> Result<()> {
            if c < 0 || c as usize >= n_cells {
                return Err(Error::Field {
                    field: what.to_string(),
                    msg: format!("has cell {c} value(s), mesh has {n_cells} cell(s)"),
                });
            }
            Ok(())
        };

        // ---- pass 1: (point, cell) incidences per point -------------------
        let mut cnt = vec![0usize; n_points];
        for (f, face) in raw.faces.iter().enumerate() {
            check_cell("owner", raw.owner[f])?;
            for &p in face {
                if p < 0 || p as usize >= n_points {
                    return Err(Error::Field {
                        field: "faces".to_string(),
                        msg: format!(
                            "has point {p} value(s), mesh has {n_points} point(s)"
                        ),
                    });
                }
                cnt[p as usize] += 1;
            }
            if f < n_if {
                check_cell("neighbour", raw.neighbour[f])?;
                for &p in face {
                    cnt[p as usize] += 1;
                }
            }
        }

        // ---- pass 2: fill, then sort + dedupe per point --------------------
        let mut offset = Vec::<Label>::with_capacity(n_points + 1);
        offset.push(0);
        for c in &cnt {
            offset.push(offset[offset.len() - 1] + *c as Label);
        }
        let mut cell = vec![0 as Label; offset[n_points] as usize];
        let mut fill: Vec<usize> = offset[..n_points].iter().map(|&x| x as usize).collect();
        for (f, face) in raw.faces.iter().enumerate() {
            for &p in face {
                let pi = p as usize;
                cell[fill[pi]] = raw.owner[f];
                fill[pi] += 1;
            }
            if f < n_if {
                for &p in face {
                    let pi = p as usize;
                    cell[fill[pi]] = raw.neighbour[f];
                    fill[pi] += 1;
                }
            }
        }

        // ---- dedupe, weight, normalise (SPEC-LIT §3.5: w = 1/|d|) ----------
        let mut off2 = Vec::<Label>::with_capacity(n_points + 1);
        let mut cells: Vec<Label> = Vec::new();
        let mut weights: Vec<Scalar> = Vec::new();
        for p in 0..n_points {
            off2.push(cells.len() as Label);
            let mut cs = cell[offset[p] as usize..offset[p + 1] as usize].to_vec();
            cs.sort_unstable();
            cs.dedup();
            let xp = raw.points[p];
            let ws: Vec<Scalar> = cs
                .iter()
                .map(|&c| 1.0 / (xp - m.c[c as usize]).mag().max(Scalar::MIN_POSITIVE))
                .collect();
            let wsum: Scalar = ws.iter().sum();
            for (c, w) in cs.into_iter().zip(ws) {
                cells.push(c);
                weights.push(w / wsum);
            }
        }
        off2.push(cells.len() as Label);

        // ---- the cell -> distinct points CSR (the inverse direction) ------
        let mut cell_cnt = vec![0usize; n_cells];
        for &c in &cells {
            cell_cnt[c as usize] += 1;
        }
        let mut cell_offset = Vec::<Label>::with_capacity(n_cells + 1);
        cell_offset.push(0);
        for k in 0..n_cells {
            cell_offset.push(cell_offset[k] + cell_cnt[k] as Label);
        }
        let mut cell_point = vec![0 as Label; cells.len()];
        let mut cfill: Vec<usize> =
            cell_offset[..n_cells].iter().map(|&x| x as usize).collect();
        for p in 0..n_points {
            for i in off2[p] as usize..off2[p + 1] as usize {
                let c = cells[i] as usize;
                cell_point[cfill[c]] = p as Label;
                cfill[c] += 1;
            }
        }

        Ok(Self {
            n_points,
            n_cells,
            offset: off2,
            cell: cells,
            weight: weights,
            cell_offset,
            cell_point,
        })
    }

    fn check_cell_len(&self, got: usize) -> Result<()> {
        if got != self.n_cells {
            return Err(Error::Field {
                field: "cell".to_string(),
                msg: format!("has {got} value(s), mesh has {} cell(s)", self.n_cells),
            });
        }
        Ok(())
    }

    /// The one gather every `cell_to_point*` runs: normalised weights mean
    /// the sum needs no second pass, and identical traversal order per point
    /// makes the vector and tensor results componentwise bit-identical to
    /// the scalar one.
    fn gather<T>(&self, cell: &[T]) -> Vec<T>
    where
        T: Copy + Default + Add<Output = T> + Mul<Scalar, Output = T>,
    {
        let mut out = Vec::with_capacity(self.n_points);
        for p in 0..self.n_points {
            let mut acc = T::default();
            for i in self.offset[p] as usize..self.offset[p + 1] as usize {
                acc = acc + cell[self.cell[i] as usize] * self.weight[i];
            }
            out.push(acc);
        }
        out
    }

    pub fn cell_to_point(&self, cell: &[Scalar]) -> Result<Vec<Scalar>> {
        self.check_cell_len(cell.len())?;
        Ok(self.gather(cell))
    }

    pub fn cell_to_point_vector(&self, cell: &[Vec3]) -> Result<Vec<Vec3>> {
        self.check_cell_len(cell.len())?;
        Ok(self.gather(cell))
    }

    pub fn cell_to_point_tensor(&self, cell: &[Tensor]) -> Result<Vec<Tensor>> {
        self.check_cell_len(cell.len())?;
        Ok(self.gather(cell))
    }

    /// The point-to-cell half of the cell -> point -> cell round trip: every
    /// cell takes the plain mean of the values at its distinct points.
    pub fn point_to_cell(&self, point: &[Scalar]) -> Result<Vec<Scalar>> {
        if point.len() != self.n_points {
            return Err(Error::Field {
                field: "point".to_string(),
                msg: format!(
                    "has {} value(s), mesh has {} point(s)",
                    point.len(),
                    self.n_points
                ),
            });
        }
        let mut out = Vec::with_capacity(self.n_cells);
        for c in 0..self.n_cells {
            let (s, e) =
                (self.cell_offset[c] as usize, self.cell_offset[c + 1] as usize);
            let n = e - s;
            if n == 0 {
                out.push(0.0);
                continue;
            }
            let mut acc = 0.0;
            for i in s..e {
                acc += point[self.cell_point[i] as usize];
            }
            out.push(acc / n as Scalar);
        }
        Ok(out)
    }
}

/// Cell-to-point interpolation by the inverse-distance weight of SPEC-LIT
/// §3.5 (`w = 1/|d|`), applied over the cells that share a point instead of
/// over a cell's face neighbours.
///
/// For a mesh point `p`, `C(p)` is the set of cells that own or neighbour a
/// face of the raw polyMesh which lists `p`, each cell counted once. The
/// point value is the weighted mean of the cell-centred values over `C(p)`:
///
/// ```text
/// phi_p = ( sum_{c in C(p)} w_pc phi_c ) / ( sum_{c in C(p)} w_pc )
/// w_pc  = 1 / max(|x_p - C_c|, tiny)
/// ```
///
/// with `x_p` the point coordinate, `C_c` the cell centre, and `tiny` the
/// smallest positive normal `f64`: it guards the division for a point that
/// coincides with a cell centre, where the plain weight would divide by
/// zero. The weights are stored NORMALISED, `sum_c w_pc = 1` for every
/// point, so a vector or a tensor field is gathered by exactly the same
/// loop applied componentwise, and the scalar, vector and tensor results
/// agree bit for bit.
///
/// On a uniform block of spacing `(hx, hy, hz)` an INTERIOR point has its
/// eight cells at mirror-image offsets `(+/-hx/2, +/-hy/2, +/-hz/2)`, equal
/// weights, so a linear field is reproduced to rounding. A BOUNDARY point's
/// cells all lie on one side of it, the interpolation is a convex
/// combination of one-sided values, and a linear field carries an `O(h)`
/// one-sided bias of at most `(|a| hx + |b| hy + |c| hz) / 2`; that bias is
/// accepted here. A point is a boundary point iff some boundary face lists
/// it - see [`boundary_points`].
pub fn cell_to_point(raw: &PolyMeshRaw, m: &HostMesh, cell: &[Scalar]) -> Result<Vec<Scalar>> {
    PointInterpolator::new(raw, m)?.cell_to_point(cell)
}

/// `[n_points]`, `true` iff the point is listed by some boundary face
/// (`raw.faces[raw.neighbour.len()..]`) - the points where the cell-to-point
/// interpolation above is one-sided.
pub fn boundary_points(raw: &PolyMeshRaw) -> Vec<bool> {
    let n_if = raw.neighbour.len();
    let mut b = vec![false; raw.points.len()];
    for face in &raw.faces[n_if..] {
        for &p in face {
            b[p as usize] = true;
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blockgen::{self, BlockSpec, GradedAxis};
    use crate::io::polymesh::build_host_mesh;

    /// A uniform 6x5x4 block of size (1.2, 1.0, 0.8): spacing (0.2, 0.2, 0.2).
    fn block_6x5x4() -> (PolyMeshRaw, HostMesh) {
        let axis = |lo: Scalar, hi: Scalar, n: usize| GradedAxis {
            lo,
            hi,
            n,
            expansion: 1.0,
            two_sided: false,
        };
        let spec = BlockSpec {
            x: axis(0.0, 1.2, 6),
            y: axis(0.0, 1.0, 5),
            z: axis(0.0, 0.8, 4),
            patch_type: ["patch", "patch", "patch", "patch", "patch", "patch"]
                .map(String::from),
            ..Default::default()
        };
        let raw = blockgen::raw_mesh(&spec).expect("raw_mesh");
        let m = build_host_mesh(&raw).expect("build_host_mesh");
        (raw, m)
    }

    /// The exact point-linear field the exactness bound is checked on:
    /// phi = 1 + 2x - 3y + 0.5z.
    fn phi(x: Scalar, y: Scalar, z: Scalar) -> Scalar {
        1.0 + 2.0 * x - 3.0 * y + 0.5 * z
    }

    #[test]
    fn a_linear_field_is_exact_at_interior_points_and_bounded_at_boundary_points() {
        let (raw, m) = block_6x5x4();
        let interp = PointInterpolator::new(&raw, &m).expect("new");
        let cell: Vec<Scalar> = m.c.iter().map(|c| phi(c.x, c.y, c.z)).collect();
        let point = interp.cell_to_point(&cell).expect("cell_to_point");
        let is_b = boundary_points(&raw);

        let (hx, hy, hz): (Scalar, Scalar, Scalar) = (1.2 / 6.0, 1.0 / 5.0, 0.8 / 4.0);
        let (a, b, c): (Scalar, Scalar, Scalar) = (2.0, -3.0, 0.5);
        let bound = (a.abs() * hx + b.abs() * hy + c.abs() * hz) / 2.0;

        let (mut max_int, mut max_bnd, mut max_phi): (Scalar, Scalar, Scalar) =
            (0.0, 0.0, 0.0);
        for p in 0..raw.points.len() {
            let x = raw.points[p];
            let exact = phi(x.x, x.y, x.z);
            max_phi = max_phi.max(exact.abs());
            let err = (point[p] - exact).abs();
            if is_b[p] {
                max_bnd = max_bnd.max(err);
            } else {
                max_int = max_int.max(err);
            }
        }
        println!("max interior error = {max_int:.3e} (tol 1e-12 * max|phi| = {:.3e})",
            1e-12 * max_phi);
        println!("max boundary error = {max_bnd:.3e} (bound {bound:.3e})");
        assert!(max_int <= 1e-12 * max_phi, "interior {max_int:.3e}");
        assert!(max_bnd <= bound + 1e-12, "boundary {max_bnd:.3e} > {bound:.3e}");
    }

    #[test]
    fn the_vertex_mean_of_a_linear_point_field_is_the_cell_centre_value() {
        let (raw, m) = block_6x5x4();
        let interp = PointInterpolator::new(&raw, &m).expect("new");
        let point: Vec<Scalar> = raw.points.iter().map(|x| phi(x.x, x.y, x.z)).collect();
        let back = interp.point_to_cell(&point).expect("point_to_cell");

        let mut max_rel = 0.0 as Scalar;
        for c in 0..m.n_cells {
            let want = phi(m.c[c].x, m.c[c].y, m.c[c].z);
            let err = (back[c] - want).abs() / want.abs().max(1.0);
            max_rel = max_rel.max(err);
        }
        println!("max relative vertex-mean error = {max_rel:.3e}");
        assert!(max_rel <= 1e-12, "vertex mean {max_rel:.3e}");
    }

    #[test]
    fn the_round_trip_is_exact_on_interior_cells_of_a_uniform_block() {
        let (raw, m) = block_6x5x4();
        let interp = PointInterpolator::new(&raw, &m).expect("new");
        let cell: Vec<Scalar> = m.c.iter().map(|c| phi(c.x, c.y, c.z)).collect();
        let point = interp.cell_to_point(&cell).expect("cell_to_point");
        let back = interp.point_to_cell(&point).expect("point_to_cell");
        let is_b = boundary_points(&raw);

        // A cell qualifies when every point of its point set - exactly what
        // the cell -> points CSR holds - is an interior point.
        let (mut qualifying, mut max_rel) = (0usize, 0.0 as Scalar);
        for c in 0..m.n_cells {
            let (s, e) =
                (interp.cell_offset[c] as usize, interp.cell_offset[c + 1] as usize);
            if !(s..e).all(|i| !is_b[interp.cell_point[i] as usize]) {
                continue;
            }
            qualifying += 1;
            let want = phi(m.c[c].x, m.c[c].y, m.c[c].z);
            max_rel = max_rel.max((back[c] - want).abs() / want.abs().max(1.0));
        }
        println!("qualifying cells = {qualifying}, max relative round-trip error = {max_rel:.3e}");
        assert_eq!(qualifying, 4 * 3 * 2);
        assert!(max_rel <= 1e-12, "round trip {max_rel:.3e}");
    }

    /// The k-th row-major component of a tensor, for the componentwise check.
    fn t_cmpt(t: &Tensor, k: usize) -> Scalar {
        [t.xx, t.xy, t.xz, t.yx, t.yy, t.yz, t.zx, t.zy, t.zz][k]
    }

    #[test]
    fn the_vector_and_tensor_gathers_agree_with_the_scalar_one_componentwise() {
        let (raw, m) = block_6x5x4();
        let interp = PointInterpolator::new(&raw, &m).expect("new");
        // Deliberately non-linear cell fields, so agreement is not vacuous.
        let s: Vec<Scalar> = (0..m.n_cells)
            .map(|c| ((7 * c) % 13) as Scalar * 0.5 - 3.0 + c as Scalar * 0.25)
            .collect();
        let v: Vec<Vec3> = s
            .iter()
            .enumerate()
            .map(|(c, &x)| Vec3::new(x, -2.0 * x + c as Scalar, 0.125 * x))
            .collect();
        let t: Vec<Tensor> = s
            .iter()
            .enumerate()
            .map(|(_, &x)| Tensor {
                xx: x, xy: x + 1.0, xz: 2.0 * x,
                yx: x - 1.0, yy: -x, yz: x * x * 0.01,
                zx: 0.5 * x, zy: x + 0.25, zz: 3.0 * x,
            })
            .collect();

        let pv = interp.cell_to_point_vector(&v).unwrap();
        let pt = interp.cell_to_point_tensor(&t).unwrap();
        let sx = interp.cell_to_point(
            &v.iter().map(|p| p.x).collect::<Vec<_>>(),
        ).unwrap();
        let sy = interp.cell_to_point(
            &v.iter().map(|p| p.y).collect::<Vec<_>>(),
        ).unwrap();
        let sz = interp.cell_to_point(
            &v.iter().map(|p| p.z).collect::<Vec<_>>(),
        ).unwrap();
        for p in 0..interp.n_points {
            assert_eq!(pv[p].x, sx[p], "vector x at point {p}");
            assert_eq!(pv[p].y, sy[p], "vector y at point {p}");
            assert_eq!(pv[p].z, sz[p], "vector z at point {p}");
        }
        let names = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"];
        for k in 0..9 {
            let col: Vec<Scalar> = t.iter().map(|x| t_cmpt(x, k)).collect();
            let sk = interp.cell_to_point(&col).unwrap();
            for p in 0..interp.n_points {
                assert_eq!(t_cmpt(&pt[p], k), sk[p], "tensor {} at point {p}", names[k]);
            }
        }
    }

    #[test]
    fn weights_sum_to_one_and_the_adjacency_of_a_block_is_what_geometry_says() {
        let (raw, m) = block_6x5x4();
        let interp = PointInterpolator::new(&raw, &m).expect("new");
        let mut tally = std::collections::BTreeMap::<usize, usize>::new();
        for p in 0..interp.n_points {
            let (s, e) = (interp.offset[p] as usize, interp.offset[p + 1] as usize);
            let n = e - s;
            *tally.entry(n).or_insert(0) += 1;
            let wsum: Scalar = interp.weight[s..e].iter().sum();
            assert!(
                (wsum - 1.0).abs() <= 1e-15 * n as Scalar,
                "point {p}: sum of {n} weights = {wsum:.3e}"
            );
        }
        let (nx, ny, nz) = (6usize, 5usize, 4usize);
        let want8 = (nx - 1) * (ny - 1) * (nz - 1);
        let want4 = 2 * ((ny - 1) * (nz - 1) + (nx - 1) * (nz - 1) + (nx - 1) * (ny - 1));
        let want2 = 4 * ((nx - 1) + (ny - 1) + (nz - 1));
        let want1 = 8;
        assert_eq!(tally.get(&8), Some(&want8), "interior points");
        assert_eq!(tally.get(&4), Some(&want4), "face points");
        assert_eq!(tally.get(&2), Some(&want2), "edge points");
        assert_eq!(tally.get(&1), Some(&want1), "corner points");
        let total: usize = tally.values().sum();
        assert_eq!(total, interp.n_points);
        println!("adjacency: 8 cells -> {want8}, 4 -> {want4}, 2 -> {want2}, 1 -> {want1}");
    }

    #[test]
    fn it_refuses_a_field_of_the_wrong_length_by_name() {
        let (raw, m) = block_6x5x4();
        let interp = PointInterpolator::new(&raw, &m).expect("new");

        let bad_cell = vec![0.0 as Scalar; interp.n_cells - 1];
        match interp.cell_to_point(&bad_cell) {
            Err(Error::Field { field, msg }) => {
                assert_eq!(field, "cell");
                assert!(msg.contains("cell"), "msg: {msg}");
            }
            other => panic!("expected Error::Field, got {other:?}"),
        }
        let bad_point = vec![0.0 as Scalar; interp.n_points - 1];
        match interp.point_to_cell(&bad_point) {
            Err(Error::Field { field, .. }) => assert_eq!(field, "point"),
            other => panic!("expected Error::Field, got {other:?}"),
        }
        // and the constructor refuses a raw whose point count disagrees
        // with the HostMesh it is checked against.
        let r = PointInterpolator::new(&raw, &HostMesh::default());
        match r {
            Err(Error::Field { field, .. }) => assert_eq!(field, "points"),
            Err(_) => panic!("expected Error::Field"),
            Ok(_) => panic!("expected Error::Field, got Ok"),
        }
    }
}
