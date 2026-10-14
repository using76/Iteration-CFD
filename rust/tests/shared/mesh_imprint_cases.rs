// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.
//
// BC-18's both-builds gate, written ONCE and included twice - by
// `tests/mesh_imprint.rs`, which runs it as an integration test in BOTH
// builds, and by the lib tests of `src/mesh/imprint.rs`, where the same
// names sit beside the code. The nine tests are SPEC-LIT §120.18's gate
// table, verbatim: matching grids, the 3:2 split counts, the 7-degree
// rotated grid's 28 pairs, the sub-face area sums, the one-to-one
// pairing, per-cell closure, the moved pair, and the refusals.
//
// An integration test's body cannot carry `//!` or `#![..]`, and neither
// can this file - it is included, not compiled standalone. Every
// comparison is made in f64 from the `PolyMeshRaw` points, which are f64
// in BOTH builds (SPEC-LIT §118.2); the areas, closure residuals and
// volumes come from an independent Newell area vector and a fan volume
// computed HERE, not from the crate's geometry.
//
// Provenance: ORIGINAL - the meshes are this tree's own two-block
// construction, the rotated grid is clipped by the crate's own
// `clip_convex`, and every expected count is SPEC-LIT §120.18's. No
// GPL-licensed source was consulted.

use std::collections::{BTreeSet, HashMap};

use ofgpu::io::polymesh::{build_host_mesh, PolyMeshRaw};
use ofgpu::mesh::imprint::{
    clip_convex, imprint, polygon_area_2d, ImprintOptions, Imprinted, ImprintTransform,
};
use ofgpu::mesh::{PatchInfo, PatchKind};
use ofgpu::types::DVec3;
use ofgpu::Label;

fn pt(p: &DVec3) -> [f64; 3] {
    [p.x, p.y, p.z]
}

/// The gate's own Newell area vector of a face loop: `1/2 sum x_k x
/// x_{k+1}`, in f64 from the mesh's points.
fn newell(points: &[DVec3], loop_: &[usize]) -> [f64; 3] {
    let n = loop_.len();
    let mut s = [0.0; 3];
    for k in 0..n {
        let p = pt(&points[loop_[k]]);
        let q = pt(&points[loop_[(k + 1) % n]]);
        s[0] += p[1] * q[2] - p[2] * q[1];
        s[1] += p[2] * q[0] - p[0] * q[2];
        s[2] += p[0] * q[1] - p[1] * q[0];
    }
    [0.5 * s[0], 0.5 * s[1], 0.5 * s[2]]
}

fn mag(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn face_loop(raw: &PolyMeshRaw, f: usize) -> Vec<usize> {
    raw.faces[f].iter().map(|&k| k as usize).collect()
}

/// A cell's faces, each oriented OUT of the cell (as stored when the cell
/// is the owner, reversed when it is the neighbour).
fn cell_face_loops(raw: &PolyMeshRaw, c: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    for f in 0..raw.faces.len() {
        if raw.owner[f] as usize == c {
            out.push(face_loop(raw, f));
        } else if f < raw.neighbour.len() && raw.neighbour[f] as usize == c {
            out.push(face_loop(raw, f).into_iter().rev().collect());
        }
    }
    out
}

/// The fan volume of a cell: sum over its out-oriented faces of
/// `(1/6)(p0 x p_k) . p_{k+1}`.
fn cell_volume(raw: &PolyMeshRaw, c: usize) -> f64 {
    let mut v = 0.0;
    for lp in cell_face_loops(raw, c) {
        let pts: Vec<[f64; 3]> = lp.iter().map(|&k| pt(&raw.points[k])).collect();
        let p0 = pts[0];
        for k in 1..pts.len() {
            let a = pts[k];
            let b = pts[(k + 1) % pts.len()];
            let cr = [
                p0[1] * a[2] - p0[2] * a[1],
                p0[2] * a[0] - p0[0] * a[2],
                p0[0] * a[1] - p0[1] * a[0],
            ];
            v += (cr[0] * b[0] + cr[1] * b[1] + cr[2] * b[2]) / 6.0;
        }
    }
    v
}

/// `|sum_f S_f|` and `sum_f |S_f|` of a cell, faces oriented out.
fn closure(raw: &PolyMeshRaw, c: usize) -> (f64, f64) {
    let mut s = [0.0; 3];
    let mut total = 0.0;
    for lp in cell_face_loops(raw, c) {
        let sf = newell(&raw.points, &lp);
        total += mag(sf);
        s[0] += sf[0];
        s[1] += sf[1];
        s[2] += sf[2];
    }
    (mag(s), total)
}

/// Every directed edge of a cell occurs exactly once, and its reverse
/// exactly once.
fn edges_paired(raw: &PolyMeshRaw, c: usize) -> bool {
    let mut count: HashMap<(usize, usize), usize> = HashMap::new();
    for lp in cell_face_loops(raw, c) {
        for i in 0..lp.len() {
            *count.entry((lp[i], lp[(i + 1) % lp.len()])).or_default() += 1;
        }
    }
    count.values().all(|&n| n == 1)
        && count.keys().all(|&(u, w)| count.contains_key(&(w, u)))
}

// ==========================================================================
//  The gate fixture: two one-layer blocks over the unit square
// ==========================================================================

/// Weld 2-D vertices across polygons within `tol`; the loops come back in
/// welded indices.
fn weld(polys: &[Vec<[f64; 2]>], tol: f64) -> (Vec<[f64; 2]>, Vec<Vec<usize>>) {
    let mut pts: Vec<[f64; 2]> = Vec::new();
    let mut loops = Vec::with_capacity(polys.len());
    for poly in polys {
        let mut lp = Vec::with_capacity(poly.len());
        for v in poly {
            let idx = pts
                .iter()
                .position(|w| (w[0] - v[0]).hypot(w[1] - v[1]) <= tol)
                .unwrap_or_else(|| {
                    pts.push(*v);
                    pts.len() - 1
                });
            lp.push(idx);
        }
        loops.push(lp);
    }
    (pts, loops)
}

/// One cell per polygon (CCW in x,y): every welded vertex at `z0` (index
/// `k`), then every one at `z1` (`k + n2`). An undirected 2-D edge used
/// by two polygons is an internal face, owner the lower cell index; used
/// by one polygon it is a boundary side face, whose loop for the edge
/// `p -> q` as the OWNER traverses it is `[p@z0, q@z0, q@z1, p@z1]`. Low
/// face = the polygon loop REVERSED at `z0`; high face = the polygon loop
/// at `z1`. Faces: internal sorted by (owner, neighbour), then `low`
/// (cell order), `high` (cell order), `side`. Patches `[low, high, side]`.
fn extrude(
    polys: &[Vec<[f64; 2]>],
    z0: f64,
    z1: f64,
    low: &str,
    high: &str,
    side: &str,
) -> PolyMeshRaw {
    let (pts2, loops) = weld(polys, 1e-12);
    let n2 = pts2.len();
    let mut points: Vec<DVec3> = Vec::with_capacity(2 * n2);
    for v in &pts2 {
        points.push(DVec3::new(v[0], v[1], z0));
    }
    for v in &pts2 {
        points.push(DVec3::new(v[0], v[1], z1));
    }
    let mut uses: HashMap<(usize, usize), Vec<(usize, usize, usize)>> = HashMap::new();
    for (c, lp) in loops.iter().enumerate() {
        for i in 0..lp.len() {
            let (u, w) = (lp[i], lp[(i + 1) % lp.len()]);
            uses.entry((u.min(w), u.max(w))).or_default().push((c, u, w));
        }
    }
    let mut internal: Vec<(usize, usize, Vec<usize>)> = Vec::new();
    let mut n_side = 0usize;
    for use_list in uses.values() {
        if use_list.len() == 2 {
            let (own, nbr, ou, ow) = if use_list[0].0 < use_list[1].0 {
                (use_list[0].0, use_list[1].0, use_list[0].1, use_list[0].2)
            } else {
                (use_list[1].0, use_list[0].0, use_list[1].1, use_list[1].2)
            };
            internal.push((own, nbr, vec![ou, ow, ow + n2, ou + n2]));
        } else {
            n_side += 1;
        }
    }
    internal.sort_by_key(|(o, n, _)| (*o, *n));
    let mut cells: Vec<Vec<usize>> = Vec::new();
    let mut owner: Vec<usize> = Vec::new();
    let mut neighbour: Vec<usize> = Vec::new();
    for (o, n, lp) in internal {
        cells.push(lp);
        owner.push(o);
        neighbour.push(n);
    }
    for (c, lp) in loops.iter().enumerate() {
        cells.push(lp.iter().rev().copied().collect());
        owner.push(c);
    }
    for (c, lp) in loops.iter().enumerate() {
        cells.push(lp.iter().map(|&k| k + n2).collect());
        owner.push(c);
    }
    for (c, lp) in loops.iter().enumerate() {
        for i in 0..lp.len() {
            let (u, w) = (lp[i], lp[(i + 1) % lp.len()]);
            if uses[&(u.min(w), u.max(w))].len() == 1 {
                cells.push(vec![u, w, w + n2, u + n2]);
                owner.push(c);
            }
        }
    }
    let mut patches = Vec::new();
    let mut start = 0usize;
    for (name, size) in [(low, loops.len()), (high, loops.len()), (side, n_side)] {
        patches.push(PatchInfo {
            name: name.to_string(),
            type_name: "patch".to_string(),
            kind: PatchKind::Generic,
            start,
            size,
            nbr_patch: None,
        });
        start += size;
    }
    PolyMeshRaw {
        points,
        faces: cells.into_iter().map(|lp| lp.iter().map(|&k| k as Label).collect()).collect(),
        owner: owner.iter().map(|&k| k as Label).collect(),
        neighbour: neighbour.iter().map(|&k| k as Label).collect(),
        patches,
    }
}

/// The `n x n` grid over the unit square: squares
/// `(i/n, j/n)-((i+1)/n, (j+1)/n)`, CCW, `j` the outer loop and `i` the
/// inner, coordinates written as `i as f64 / n`.
fn grid(n: usize) -> Vec<Vec<[f64; 2]>> {
    let mut polys = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            polys.push(vec![
                [i as f64 / n as f64, j as f64 / n as f64],
                [(i + 1) as f64 / n as f64, j as f64 / n as f64],
                [(i + 1) as f64 / n as f64, (j + 1) as f64 / n as f64],
                [i as f64 / n as f64, (j + 1) as f64 / n as f64],
            ]);
        }
    }
    polys
}

/// The grid of spacing 0.5 rotated +7 degrees about (0.5, 0.5), for
/// `i, j` in `-1..=2`, each square then clipped to the unit square with
/// the crate's own `clip_convex` and kept when its area exceeds 1e-12:
/// 12 polygons of total area 1.
fn rotated7() -> Vec<Vec<[f64; 2]>> {
    let (sn, cs) = 7.0_f64.to_radians().sin_cos();
    let rot = |p: [f64; 2]| -> [f64; 2] {
        let (dx, dy) = (p[0] - 0.5, p[1] - 0.5);
        [0.5 + dx * cs - dy * sn, 0.5 + dx * sn + dy * cs]
    };
    let unit: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut polys = Vec::new();
    for j in -1i64..=2 {
        for i in -1i64..=2 {
            let sq = vec![
                rot([0.5 * i as f64, 0.5 * j as f64]),
                rot([0.5 * (i + 1) as f64, 0.5 * j as f64]),
                rot([0.5 * (i + 1) as f64, 0.5 * (j + 1) as f64]),
                rot([0.5 * i as f64, 0.5 * (j + 1) as f64]),
            ];
            let clipped = clip_convex(&sq, &unit, 1e-12);
            if polygon_area_2d(&clipped) > 1e-12 {
                polys.push(clipped);
            }
        }
    }
    polys
}

/// The moved pair's point map: `Rz(+90 deg)` about (0.5, 0.5), then
/// `+ (0, 0, 5)` - `(x, y, z) -> (1 - y, x, z + 5)`.
fn moved_map(p: [f64; 3]) -> [f64; 3] {
    [1.0 - p[1], p[0], p[2] + 5.0]
}

/// The transform of the moved pair: `T` maps the stored `B` back onto the
/// grid `A` sees - `rot = [[0,1,0],[-1,0,0],[0,0,1]]`, `shift = [0,1,-5]`.
fn moved_options() -> ImprintOptions {
    ImprintOptions {
        transform: ImprintTransform {
            rot: [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            shift: [0.0, 1.0, -5.0],
        },
        rel_tol: 1e-9,
    }
}

/// `A` = the 3 x 3 grid extruded to `z in [-1/3, 0]`, `B` = `b_polys`
/// extruded to `z in [0, 0.5]` with its points mapped by `b_map` when
/// given, concatenated: `B`'s points/cells/faces after `A`'s, internal
/// faces `A` then `B`, then boundary `A`'s three patches then `B`'s
/// three; patch order `[bottomA, ifaceA, sidesA, ifaceB, topB, sidesB]`.
fn two_blocks(
    b_polys: &[Vec<[f64; 2]>],
    b_map: Option<fn([f64; 3]) -> [f64; 3]>,
) -> PolyMeshRaw {
    let a = extrude(&grid(3), -1.0 / 3.0, 0.0, "bottomA", "ifaceA", "sidesA");
    let mut b = extrude(b_polys, 0.0, 0.5, "ifaceB", "topB", "sidesB");
    if let Some(map) = b_map {
        for p in b.points.iter_mut() {
            let q = map([p.x, p.y, p.z]);
            *p = DVec3::new(q[0], q[1], q[2]);
        }
    }
    let na_cells = a.owner.iter().copied().max().unwrap_or(0) as usize + 1;
    let n_if_a = a.neighbour.len();
    let n_if_b = b.neighbour.len();
    let n_bf_a = a.faces.len() - n_if_a;
    let np_a = a.points.len() as Label;
    let dc = na_cells as Label;
    let mut points = a.points.clone();
    points.extend(b.points.clone());
    let map = |lp: &[Label]| lp.iter().map(|&k| k + np_a).collect();
    // internal faces A then B, then boundary A's three patches then B's
    // three - so the first n_if_a + n_if_b faces are the internal ones.
    let mut faces: Vec<Vec<Label>> = Vec::with_capacity(a.faces.len() + b.faces.len());
    faces.extend(a.faces[..n_if_a].iter().map(|lp| lp.clone()));
    faces.extend(b.faces[..n_if_b].iter().map(|lp| map(lp)));
    faces.extend(a.faces[n_if_a..].iter().map(|lp| lp.clone()));
    faces.extend(b.faces[n_if_b..].iter().map(|lp| map(lp)));
    let mut owner: Vec<Label> = Vec::with_capacity(faces.len());
    owner.extend(a.owner[..n_if_a].iter().copied());
    owner.extend(b.owner[..n_if_b].iter().map(|&o| o + dc));
    owner.extend(a.owner[n_if_a..].iter().copied());
    owner.extend(b.owner[n_if_b..].iter().map(|&o| o + dc));
    let mut neighbour: Vec<Label> = a.neighbour.clone();
    neighbour.extend(b.neighbour.iter().map(|&n| n + dc));
    let mut patches = a.patches.clone();
    for p in b.patches {
        patches.push(PatchInfo {
            start: p.start + n_bf_a,
            ..p
        });
    }
    PolyMeshRaw {
        points,
        faces,
        owner,
        neighbour,
        patches,
    }
}

/// The faces OUTSIDE the two imprinted patches that changed - internal
/// faces keep their indices; every other patch keeps its face order, so
/// face `j` of patch `p` is input face `n_if + p.start + j` against
/// output face `n_if + p.start + j` of the output's own patch table.
fn changed_outside(raw: &PolyMeshRaw, out: &Imprinted) -> Vec<(usize, usize)> {
    let n_if = raw.neighbour.len();
    let mut ch = Vec::new();
    for f in 0..n_if {
        if out.mesh.faces[f] != raw.faces[f] {
            ch.push((f, f));
        }
    }
    for (pi, pat) in raw.patches.iter().enumerate() {
        if pi == out.patch_a || pi == out.patch_b {
            continue;
        }
        for j in 0..pat.size {
            let f = n_if + pat.start + j;
            let g = n_if + out.mesh.patches[pi].start + j;
            if out.mesh.faces[g] != raw.faces[f] {
                ch.push((f, g));
            }
        }
    }
    ch
}

fn tally(parents: &[usize], n: usize) -> Vec<usize> {
    let mut t = vec![0usize; n];
    for &p in parents {
        t[p] += 1;
    }
    t
}

#[test]
fn matching_grids_imprint_to_themselves() {
    let raw = two_blocks(&grid(3), None);
    let out = imprint(&raw, "ifaceA", "ifaceB", &ImprintOptions::default()).expect("imprint");
    assert_eq!(out.parent_a.len(), 9);
    assert_eq!(out.parent_b.len(), 9);
    assert_eq!(out.n_new_points_a, 0);
    assert_eq!(out.n_new_points_b, 0);
    assert_eq!(out.n_faces_split, 0);
    assert_eq!(out.mesh.points.len(), raw.points.len());
    for (p, q) in out.mesh.points.iter().zip(raw.points.iter()) {
        assert_eq!(pt(p), pt(q));
    }
    let n_if = raw.neighbour.len();
    let oa = n_if + out.mesh.patches[out.patch_a].start;
    for (i, &parent) in out.parent_a.iter().enumerate() {
        let sub: BTreeSet<usize> = face_loop(&out.mesh, oa + i).into_iter().collect();
        let par: BTreeSet<usize> = face_loop(
            &raw,
            n_if + raw.patches[out.patch_a].start + parent,
        )
        .into_iter()
        .collect();
        assert_eq!(sub, par, "sub-face {i} of parent {parent}");
    }
}

#[test]
fn three_to_two_splits_into_sixteen_pairs() {
    let raw = two_blocks(&grid(2), None);
    let out = imprint(&raw, "ifaceA", "ifaceB", &ImprintOptions::default()).expect("imprint");
    assert_eq!(out.parent_a.len(), 16);
    assert_eq!(out.parent_b.len(), 16);
    assert_eq!(out.n_new_points_a, 9);
    assert_eq!(out.n_new_points_b, 16);
    assert_eq!(out.mesh.points.len(), raw.points.len() + 25);
    assert_eq!(tally(&out.parent_a, 9), vec![1, 2, 1, 2, 4, 2, 1, 2, 1]);
    assert!(tally(&out.parent_b, 4).iter().all(|&n| n == 4));
    let ch = changed_outside(&raw, &out);
    assert_eq!(ch.len(), 20);
    let (mut in_a, mut in_b) = (0usize, 0usize);
    for &(f, g) in &ch {
        assert_eq!(
            out.mesh.faces[g].len(),
            raw.faces[f].len() + 1,
            "face {f}"
        );
        if (raw.owner[f] as usize) < 9 {
            in_a += 1;
        } else {
            in_b += 1;
        }
    }
    assert_eq!(in_a, 8);
    assert_eq!(in_b, 12);
}

#[test]
fn rotated_seven_degrees_gives_twenty_eight_pairs() {
    let polys = rotated7();
    assert_eq!(polys.len(), 12);
    let total: f64 = polys.iter().map(|p| polygon_area_2d(p)).sum();
    assert!((total - 1.0).abs() <= 1e-14, "total area {total}");
    let raw = two_blocks(&polys, None);
    let out = imprint(&raw, "ifaceA", "ifaceB", &ImprintOptions::default()).expect("imprint");
    assert_eq!(out.parent_a.len(), 28);
    assert_eq!(out.parent_b.len(), 28);
    assert_eq!(tally(&out.parent_a, 9), vec![2, 4, 2, 4, 4, 4, 2, 4, 2]);
    assert_eq!(
        tally(&out.parent_b, 12),
        vec![1, 2, 2, 4, 4, 1, 1, 4, 4, 2, 2, 1]
    );
}

#[test]
fn sub_face_areas_sum_to_each_parent() {
    for (polys, n_pb) in [(grid(2), 4usize), (rotated7(), 12)] {
        let raw = two_blocks(&polys, None);
        let out = imprint(&raw, "ifaceA", "ifaceB", &ImprintOptions::default()).expect("imprint");
        let n_if = raw.neighbour.len();
        for (side, (parents, np)) in [
            (&out.parent_a, 9usize),
            (&out.parent_b, n_pb),
        ]
        .into_iter()
        .enumerate()
        {
            let patch = if side == 0 { out.patch_a } else { out.patch_b };
            let ostart = n_if + out.mesh.patches[patch].start;
            let pstart = n_if + raw.patches[patch].start;
            let mut sums = vec![0.0f64; np];
            for (i, &p) in parents.iter().enumerate() {
                sums[p] += mag(newell(&out.mesh.points, &face_loop(&out.mesh, ostart + i)));
            }
            for (pi, &s) in sums.iter().enumerate() {
                let sp = mag(newell(&raw.points, &face_loop(&raw, pstart + pi)));
                assert!(
                    (s - sp).abs() <= 1e-12 * sp,
                    "side {side} parent {pi}: sum {s} vs {sp}"
                );
            }
            let total: f64 = sums.iter().sum();
            assert!((total - 1.0).abs() <= 1e-12, "side {side} total area {total}");
        }
    }
}

#[test]
fn the_two_sides_pair_one_to_one() {
    let cases: [(Vec<Vec<[f64; 2]>>, Option<fn([f64; 3]) -> [f64; 3]>, ImprintOptions); 3] = [
        (grid(2), None, ImprintOptions::default()),
        (rotated7(), None, ImprintOptions::default()),
        (grid(2), Some(moved_map), moved_options()),
    ];
    for (polys, map, opts) in cases {
        let raw = two_blocks(&polys, map);
        let out = imprint(&raw, "ifaceA", "ifaceB", &opts).expect("imprint");
        let tr = opts.transform;
        let exact = tr == ImprintTransform::identity();
        let n_if = raw.neighbour.len();
        assert_eq!(
            out.mesh.patches[out.patch_a].size,
            out.mesh.patches[out.patch_b].size
        );
        let sa = n_if + out.mesh.patches[out.patch_a].start;
        let sb = n_if + out.mesh.patches[out.patch_b].start;
        for i in 0..out.parent_a.len() {
            let la: Vec<[f64; 3]> = face_loop(&out.mesh, sa + i)
                .iter()
                .map(|&k| pt(&out.mesh.points[k]))
                .collect();
            let lb: Vec<[f64; 3]> = face_loop(&out.mesh, sb + i)
                .iter()
                .map(|&k| pt(&out.mesh.points[k]))
                .collect();
            let n = la.len();
            assert_eq!(lb.len(), n, "pair {i}");
            let reversed = std::iter::once(la[0]).chain((1..n).rev().map(|k| la[k]));
            for (j, w) in reversed.enumerate() {
                let v = tr.apply(lb[j]);
                if exact {
                    assert_eq!(v, w, "pair {i} vertex {j} is not bit for bit");
                } else {
                    for d in 0..3 {
                        assert!(
                            (v[d] - w[d]).abs() <= 1e-12,
                            "pair {i} vertex {j} coord {d}: {} vs {}",
                            v[d],
                            w[d]
                        );
                    }
                }
            }
            let sab = newell(&out.mesh.points, &face_loop(&out.mesh, sa + i));
            let sbb = newell(&out.mesh.points, &face_loop(&out.mesh, sb + i));
            let rsb = tr.rotate(sbb);
            let sum = [sab[0] + rsb[0], sab[1] + rsb[1], sab[2] + rsb[2]];
            assert!(
                mag(sum) <= 1e-12 * mag(sab),
                "pair {i}: |S_A + R S_B| = {}",
                mag(sum)
            );
        }
    }
}

#[test]
fn imprinted_cells_close() {
    let cases: [(Vec<Vec<[f64; 2]>>, Option<fn([f64; 3]) -> [f64; 3]>, ImprintOptions); 3] = [
        (grid(2), None, ImprintOptions::default()),
        (rotated7(), None, ImprintOptions::default()),
        (grid(2), Some(moved_map), moved_options()),
    ];
    for (polys, map, opts) in cases {
        let raw = two_blocks(&polys, map);
        let out = imprint(&raw, "ifaceA", "ifaceB", &opts).expect("imprint");
        let hi = build_host_mesh(&raw).expect("input builds");
        let hc = build_host_mesh(&out.mesh).expect("build_host_mesh");
        assert_eq!(hc.n_cells, hi.n_cells);
        for c in 0..hi.n_cells {
            let (sum, total) = closure(&out.mesh, c);
            assert!(
                sum <= 1e-12 * total,
                "cell {c}: |sum S_f| = {sum}, sum |S_f| = {total}"
            );
            assert!(edges_paired(&out.mesh, c), "cell {c} edges not paired");
            let (v0, v1) = (cell_volume(&raw, c), cell_volume(&out.mesh, c));
            assert!(
                (v1 - v0).abs() <= 1e-12 * v0.abs(),
                "cell {c}: volume {v1} vs input {v0}"
            );
        }
    }
}

#[test]
fn a_moved_pair_imprints_like_the_fixed_one() {
    let raw = two_blocks(&grid(2), Some(moved_map));
    let out = imprint(&raw, "ifaceA", "ifaceB", &moved_options()).expect("imprint");
    assert_eq!(out.parent_a.len(), 16);
    assert_eq!(out.parent_b.len(), 16);
    assert_eq!(out.n_new_points_a, 9);
    assert_eq!(out.n_new_points_b, 16);
    assert_eq!(out.n_faces_split, 20);
}

#[test]
fn partial_overlap_is_refused_by_name() {
    let shift_x: fn([f64; 3]) -> [f64; 3] = |p| [p[0] + 0.1, p[1], p[2]];
    let raw = two_blocks(&grid(3), Some(shift_x));
    let err = imprint(&raw, "ifaceA", "ifaceB", &ImprintOptions::default()).expect_err("refused");
    let msg = err.to_string();
    assert!(msg.contains("cyclicACMI"), "{msg}");
    assert!(msg.contains("covered"), "{msg}");
    let shift_z: fn([f64; 3]) -> [f64; 3] = |p| [p[0], p[1], p[2] + 0.01];
    let raw = two_blocks(&grid(2), Some(shift_z));
    let err = imprint(&raw, "ifaceA", "ifaceB", &ImprintOptions::default()).expect_err("refused");
    assert!(err.to_string().contains("covered"), "{}", err);
}

#[test]
fn bad_requests_are_refused() {
    let raw = two_blocks(&grid(3), None);
    let err = imprint(&raw, "nope", "ifaceB", &ImprintOptions::default()).expect_err("refused");
    assert!(err.to_string().contains("nope"), "{}", err);
    let err = imprint(&raw, "ifaceA", "ifaceA", &ImprintOptions::default()).expect_err("refused");
    assert!(err.to_string().contains("ifaceA"), "{}", err);
    let bad = ImprintOptions {
        transform: ImprintTransform {
            rot: [[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            shift: [0.0; 3],
        },
        rel_tol: 1e-9,
    };
    let err = imprint(&raw, "ifaceA", "ifaceB", &bad).expect_err("refused");
    assert!(err.to_string().contains("orthonormal"), "{}", err);
    let mut raw2 = two_blocks(&grid(3), None);
    let pb = raw2.patches.iter().position(|p| p.name == "ifaceB").unwrap();
    let n_if = raw2.neighbour.len();
    for j in 0..raw2.patches[pb].size {
        raw2.faces[n_if + raw2.patches[pb].start + j].reverse();
    }
    let err = imprint(&raw2, "ifaceA", "ifaceB", &ImprintOptions::default()).expect_err("refused");
    assert!(err.to_string().contains("same way"), "{}", err);
}
