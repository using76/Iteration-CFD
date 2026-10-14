// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Deterministic double pairwise aggregation of one LDU level - AMG-01.
//!
//! Notay's Algorithm 2.1 (one pairwise pass) run twice, as his Algorithm
//! 2.2: the symmetric strengths of (141.4), every tie broken by the smaller
//! id, the aggregates numbered by (141.8) and the coarse addressing of the
//! fold table. Nothing iterates a hash map, so the same input gives the same
//! bytes on every run (SPEC-LIT §141.1).
//!
//! Provenance: ORIGINAL - written from Notay, ETNA 37 (2010) 123-146,
//! Algorithms 2.1 and 2.2 (SPEC-LIT §141.1). No GPL-licensed source was
//! consulted.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::error::{Error, Result};
use crate::ldu::HostLduMatrix;
use crate::mesh::HostMesh;
use crate::{Label, Scalar};

/// The strength threshold of (141.5): `S_i = { j : s_ij > BETA max s_ik }`.
pub const BETA: f64 = 0.25;

/// A fine face that folds into its aggregate's diagonal (the fold table).
pub const INTERIOR: Label = -1;

/// An uncoupled boundary face, which lands nowhere in the coarse matrix.
pub const UNCOUPLED: Label = -2;

/// One level of an LDU matrix. A coarse level passes empty boundary slices.
#[derive(Clone, Copy)]
pub struct LevelView<'a> {
    pub n_cells: usize,
    /// `[n_faces]`
    pub owner: &'a [Label],
    /// `[n_faces]`
    pub neighbour: &'a [Label],
    /// `[n_cells]`
    pub diag: &'a [Scalar],
    /// `[n_faces]`
    pub upper: &'a [Scalar],
    /// `[n_faces]`
    pub lower: &'a [Scalar],
    /// `[n_bf]`
    pub b_face_cells: &'a [Label],
    /// `[n_bf]`, `< 0` = uncoupled
    pub b_nbr_cell: &'a [Label],
    /// `[n_bf]`
    pub boundary_coeffs: &'a [Scalar],
}

/// The result of double pairwise aggregation (SPEC-LIT §141.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aggregation {
    pub n_fine: usize,
    /// Aggregates after pass 1.
    pub n_pass1: usize,
    /// Final aggregates.
    pub n_coarse: usize,
    /// Fine cell -> aggregate, numbered by (141.8).
    pub cell_agg: Vec<Label>,
    /// CSR offsets of the members, `[n_coarse + 1]`.
    pub agg_offset: Vec<Label>,
    /// Members, ascending within each aggregate.
    pub agg_cells: Vec<Label>,
    /// `= min(I, J)`, ascending by `(owner, neighbour)`.
    pub coarse_owner: Vec<Label>,
    /// `= max(I, J)`, ascending by `(owner, neighbour)`.
    pub coarse_neighbour: Vec<Label>,
    /// Coarse face index of each fine internal face, or [`INTERIOR`].
    pub face_coarse: Vec<Label>,
    /// Per the fold table; `false` on [`INTERIOR`].
    pub face_flip: Vec<bool>,
    /// Coarse face, [`INTERIOR`] or [`UNCOUPLED`], per boundary face.
    pub bface_coarse: Vec<Label>,
    /// Per the fold table.
    pub bface_flip: Vec<bool>,
}

impl Aggregation {
    /// Distinct coarse faces.
    pub fn n_coarse_faces(&self) -> usize {
        self.coarse_owner.len()
    }

    /// The members of one aggregate.
    pub fn size(&self, agg: usize) -> usize {
        (self.agg_offset[agg + 1] - self.agg_offset[agg]) as usize
    }

    /// The aggregation a given cell-to-aggregate map defines on `l`: the
    /// members CSR and the coarse addressing of the fold table, built by the
    /// code `aggregate_level` ends with, with no pass run (`n_pass1 = 0`).
    /// The map must already be numbered by (141.8), every id `0..n_c` used,
    /// and `l` passes the checks of §141.1. This is how the union of the
    /// block-local aggregations of the parts becomes one whole-level
    /// aggregation (SPEC-LIT §141.10).
    pub fn from_map(l: &LevelView, cell_agg: Vec<Label>) -> Result<Self> {
        check_level(l)?;
        if cell_agg.len() != l.n_cells {
            return Err(Error::Mesh(format!(
                "amg::aggregate: length mismatch: the map has {} entries, n_cells is {}",
                cell_agg.len(),
                l.n_cells
            )));
        }
        // (141.8): scanning the cells in order, each aggregate id first
        // appears in the order 0, 1, 2, ...
        let mut n_c: Label = 0;
        for (c, &g) in cell_agg.iter().enumerate() {
            if g < 0 || g > n_c {
                return Err(Error::Mesh(format!(
                    "amg::aggregate: the map is not numbered by (141.8): cell {} has aggregate {} \
                     where at most {} may appear",
                    c, g, n_c
                )));
            }
            if g == n_c {
                n_c += 1;
            }
        }
        addressing(l, cell_agg, 0, n_c as usize)
    }
}

/// The finest level: addressing and boundary faces from `m`, coefficients
/// from `a`. The shapes of `a` are checked against `m` (SPEC-LIT §141.1).
pub fn aggregate(m: &HostMesh, a: &HostLduMatrix) -> Result<Aggregation> {
    if a.diag.len() != m.n_cells {
        return Err(Error::Mesh(format!(
            "amg::aggregate: length mismatch: diag has {} entries, the mesh has {} cells",
            a.diag.len(),
            m.n_cells
        )));
    }
    if a.upper.len() != m.n_internal_faces {
        return Err(Error::Mesh(format!(
            "amg::aggregate: length mismatch: upper has {} entries, the mesh has {} internal faces",
            a.upper.len(),
            m.n_internal_faces
        )));
    }
    if a.lower.len() != m.n_internal_faces {
        return Err(Error::Mesh(format!(
            "amg::aggregate: length mismatch: lower has {} entries, the mesh has {} internal faces",
            a.lower.len(),
            m.n_internal_faces
        )));
    }
    if a.boundary_coeffs.len() != m.n_boundary_faces {
        return Err(Error::Mesh(format!(
            "amg::aggregate: length mismatch: boundary_coeffs has {} entries, the mesh has {} boundary faces",
            a.boundary_coeffs.len(),
            m.n_boundary_faces
        )));
    }
    aggregate_level(&LevelView {
        n_cells: m.n_cells,
        owner: &m.owner,
        neighbour: &m.neighbour,
        diag: &a.diag,
        upper: &a.upper,
        lower: &a.lower,
        b_face_cells: &m.b_face_cells,
        b_nbr_cell: &m.b_nbr_cell,
        boundary_coeffs: &a.boundary_coeffs,
    })
}

/// Aggregate one level: both passes, the (141.8) numbering and the coarse
/// addressing of the fold table (SPEC-LIT §141.1).
pub fn aggregate_level(l: &LevelView) -> Result<Aggregation> {
    check_level(l)?;
    let n = l.n_cells;
    let nf = l.owner.len();
    let nbf = l.b_face_cells.len();

    // The sign (141.3), summed in f64 in cell order.
    let mut dsum = 0.0f64;
    for c in 0..n {
        dsum += l.diag[c] as f64;
    }
    if !dsum.is_finite() || dsum == 0.0 {
        return Err(Error::Config(format!(
            "amg::aggregate: diagonal sum is {}, so the sign of the operator (141.3) is undefined",
            dsum
        )));
    }
    let sigma = if dsum > 0.0 { 1.0 } else { -1.0 };

    // The ordered entries of (141.1)/(141.2), internal faces ascending then
    // boundary faces ascending. Key = min * n + max; side 0 holds the
    // (min, max) entry, side 1 the (max, min) entry.
    let mut ent: Vec<(usize, u8, f64)> = Vec::with_capacity(2 * nf + nbf);
    for f in 0..nf {
        let (o, nb) = (l.owner[f] as usize, l.neighbour[f] as usize);
        let (key, a_lo, a_hi) = if o < nb {
            (o * n + nb, l.upper[f] as f64, l.lower[f] as f64)
        } else {
            (nb * n + o, l.lower[f] as f64, l.upper[f] as f64)
        };
        ent.push((key, 0, a_lo));
        ent.push((key, 1, a_hi));
    }
    for bf in 0..nbf {
        let d = l.b_nbr_cell[bf];
        if d < 0 {
            continue; // (141.2) covers coupled faces only
        }
        let (c, d) = (l.b_face_cells[bf] as usize, d as usize);
        let v = -(l.boundary_coeffs[bf] as f64);
        if c < d {
            ent.push((c * n + d, 0, v));
        } else {
            ent.push((d * n + c, 1, v));
        }
    }
    // Stable: face order survives inside each side of a pair.
    ent.sort_by_key(|e| e.0);

    // One strength per unordered pair (141.4); the edges come out ascending
    // by (min, max), which is also the summation order (141.7) needs.
    let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(ent.len() / 2);
    let mut t = 0usize;
    while t < ent.len() {
        let key = ent[t].0;
        let (mut a_lo, mut a_hi) = (0.0f64, 0.0f64);
        while t < ent.len() && ent[t].0 == key {
            if ent[t].1 == 0 {
                a_lo += ent[t].2;
            } else {
                a_hi += ent[t].2;
            }
            t += 1;
        }
        let s = -sigma * (a_lo + a_hi) / 2.0;
        edges.push((key / n, key % n, s));
    }

    // Pass 1 on the fine graph (Notay's Algorithm 2.1), its aggregates
    // numbered by (141.8).
    let pass1 = one_two_passes(n, &edges)?;
    let n1 = pass1.len();
    let p1 = pass_map(n, &pass1);

    // Pass 2 (Notay's Algorithm 2.2) on the pass-1 aggregates: the strength
    // s1_IJ of (141.7), summed in ascending fine-edge order - `edges`
    // already is in that order.
    let mut ent2: Vec<(usize, f64)> = Vec::with_capacity(edges.len());
    for &(lo, hi, s) in edges.iter() {
        let (a, b) = (p1[lo] as usize, p1[hi] as usize);
        if a == b {
            continue;
        }
        let key = if a < b { a * n1 + b } else { b * n1 + a };
        ent2.push((key, s));
    }
    ent2.sort_by_key(|e| e.0);
    let mut edges2: Vec<(usize, usize, f64)> = Vec::with_capacity(ent2.len());
    let mut t2 = 0usize;
    while t2 < ent2.len() {
        let key = ent2[t2].0;
        let mut s1 = 0.0f64;
        while t2 < ent2.len() && ent2[t2].0 == key {
            s1 += ent2[t2].1;
            t2 += 1;
        }
        edges2.push((key / n1, key % n1, s1));
    }
    let pass2 = one_two_passes(n1, &edges2)?;
    let n2 = pass2.len();
    let p2 = pass_map(n1, &pass2);

    // The final aggregate of a cell is its pass-2 aggregate.
    let mut cell_agg = vec![0 as Label; n];
    for i in 0..n {
        cell_agg[i] = p2[p1[i] as usize];
    }

    addressing(l, cell_agg, n1, n2)
}

/// The checks every level passes before it is aggregated or addressed:
/// the lengths, the finite coefficients and the addressing, each refused
/// by name (SPEC-LIT §141.1).
fn check_level(l: &LevelView) -> Result<()> {
    let n = l.n_cells;
    let nf = l.owner.len();
    let nbf = l.b_face_cells.len();

    if l.diag.len() != n {
        return Err(Error::Mesh(format!(
            "amg::aggregate: length mismatch: diag has {} entries, n_cells is {}",
            l.diag.len(),
            n
        )));
    }
    if l.neighbour.len() != nf {
        return Err(Error::Mesh(format!(
            "amg::aggregate: length mismatch: neighbour has {} entries, owner has {}",
            l.neighbour.len(),
            nf
        )));
    }
    if l.upper.len() != nf {
        return Err(Error::Mesh(format!(
            "amg::aggregate: length mismatch: upper has {} entries, owner has {}",
            l.upper.len(),
            nf
        )));
    }
    if l.lower.len() != nf {
        return Err(Error::Mesh(format!(
            "amg::aggregate: length mismatch: lower has {} entries, owner has {}",
            l.lower.len(),
            nf
        )));
    }
    if l.b_nbr_cell.len() != nbf {
        return Err(Error::Mesh(format!(
            "amg::aggregate: length mismatch: b_nbr_cell has {} entries, b_face_cells has {}",
            l.b_nbr_cell.len(),
            nbf
        )));
    }
    if l.boundary_coeffs.len() != nbf {
        return Err(Error::Mesh(format!(
            "amg::aggregate: length mismatch: boundary_coeffs has {} entries, b_face_cells has {}",
            l.boundary_coeffs.len(),
            nbf
        )));
    }

    // Non-finite coefficients, named by array and index (SPEC-LIT §141.1).
    for (c, &d) in l.diag.iter().enumerate() {
        if !(d as f64).is_finite() {
            return Err(Error::Config(format!(
                "amg::aggregate: non-finite diag[{}]",
                c
            )));
        }
    }
    for (f, &u) in l.upper.iter().enumerate() {
        if !(u as f64).is_finite() {
            return Err(Error::Config(format!(
                "amg::aggregate: non-finite upper[{}]",
                f
            )));
        }
    }
    for (f, &lw) in l.lower.iter().enumerate() {
        if !(lw as f64).is_finite() {
            return Err(Error::Config(format!(
                "amg::aggregate: non-finite lower[{}]",
                f
            )));
        }
    }
    for (bf, &bc) in l.boundary_coeffs.iter().enumerate() {
        if l.b_nbr_cell[bf] >= 0 && !(bc as f64).is_finite() {
            return Err(Error::Config(format!(
                "amg::aggregate: non-finite boundary_coeffs[{}]",
                bf
            )));
        }
    }

    // Addressing: every id inside the level, no self-couple, no halo couple
    // (SPEC-LIT §141.1, refused by name).
    for (f, &o) in l.owner.iter().enumerate() {
        if o < 0 || o as usize >= n {
            return Err(Error::Mesh(format!(
                "amg::aggregate: owner[{}] = {} out of 0..{}",
                f, o, n
            )));
        }
        if o == l.neighbour[f] {
            return Err(Error::Mesh(format!(
                "amg::aggregate: self-coupled internal face {} (owner == neighbour)",
                f
            )));
        }
    }
    for (f, &nb) in l.neighbour.iter().enumerate() {
        if nb < 0 || nb as usize >= n {
            return Err(Error::Mesh(format!(
                "amg::aggregate: neighbour[{}] = {} out of 0..{}",
                f, nb, n
            )));
        }
    }
    for (bf, &c) in l.b_face_cells.iter().enumerate() {
        if c < 0 || c as usize >= n {
            return Err(Error::Mesh(format!(
                "amg::aggregate: b_face_cells[{}] = {} out of 0..{}",
                bf, c, n
            )));
        }
    }
    for (bf, &d) in l.b_nbr_cell.iter().enumerate() {
        if d < 0 {
            continue; // uncoupled
        }
        if d as usize >= n {
            return Err(Error::Mesh(format!(
                "amg::aggregate: coupled face {} has b_nbr_cell[{}] = {} >= n_cells {}; block-local aggregation across a cut is AMG-07a's business (SPEC-LIT §141.1)",
                bf, bf, d, n
            )));
        }
        if d == l.b_face_cells[bf] {
            return Err(Error::Mesh(format!(
                "amg::aggregate: self-coupled boundary face {} (b_nbr_cell == b_face_cells)",
                bf
            )));
        }
    }
    Ok(())
}

/// The members CSR and the coarse addressing of the fold table, from a
/// final cell-to-aggregate map numbered by (141.8) (SPEC-LIT §141.1).
fn addressing(l: &LevelView, cell_agg: Vec<Label>, n1: usize, n2: usize) -> Result<Aggregation> {
    let n = l.n_cells;
    let nf = l.owner.len();
    let nbf = l.b_face_cells.len();

    // The members as a CSR, ascending inside each aggregate (SPEC-LIT
    // §141.1) - the aggregate-sorted order AMG-02's restriction reads.
    let mut agg_offset = vec![0 as Label; n2 + 1];
    for &g in cell_agg.iter() {
        agg_offset[g as usize + 1] += 1;
    }
    for g in 0..n2 {
        agg_offset[g + 1] += agg_offset[g];
    }
    let mut agg_cells = vec![0 as Label; n];
    {
        let mut cur = agg_offset.clone();
        cur.truncate(n2);
        for i in 0..n {
            let g = cell_agg[i] as usize;
            agg_cells[cur[g] as usize] = i as Label;
            cur[g] += 1;
        }
    }

    // The coarse addressing: where each fine entry lands (the fold table).
    let mut xface: Vec<usize> = Vec::new();
    let mut face_coarse = vec![INTERIOR; nf];
    let mut face_flip = vec![false; nf];
    for f in 0..nf {
        let io = cell_agg[l.owner[f] as usize];
        let inb = cell_agg[l.neighbour[f] as usize];
        if io == inb {
            continue; // into the coarse diagonal
        }
        face_flip[f] = io > inb;
        xface.push(f);
    }
    let mut xbf: Vec<usize> = Vec::new();
    let mut bface_coarse = vec![UNCOUPLED; nbf];
    let mut bface_flip = vec![false; nbf];
    for bf in 0..nbf {
        let d = l.b_nbr_cell[bf];
        if d < 0 {
            continue; // uncoupled: UNCOUPLED, and nothing in the coarse matrix
        }
        let ic = cell_agg[l.b_face_cells[bf] as usize];
        let idc = cell_agg[d as usize];
        if ic == idc {
            bface_coarse[bf] = INTERIOR;
            continue; // into the coarse diagonal
        }
        bface_flip[bf] = ic > idc;
        xbf.push(bf);
    }

    // Distinct coarse faces, ascending by (owner, neighbour); duplicate
    // edges - the same unordered pair from several faces - merge here.
    let mut pairs: Vec<(Label, Label)> = Vec::with_capacity(xface.len() + xbf.len());
    for &f in xface.iter() {
        let io = cell_agg[l.owner[f] as usize];
        let inb = cell_agg[l.neighbour[f] as usize];
        pairs.push((io.min(inb), io.max(inb)));
    }
    for &bf in xbf.iter() {
        let ic = cell_agg[l.b_face_cells[bf] as usize];
        let idc = cell_agg[l.b_nbr_cell[bf] as usize];
        pairs.push((ic.min(idc), ic.max(idc)));
    }
    pairs.sort();
    pairs.dedup();
    let coarse_owner: Vec<Label> = pairs.iter().map(|p| p.0).collect();
    let coarse_neighbour: Vec<Label> = pairs.iter().map(|p| p.1).collect();
    for &f in xface.iter() {
        let io = cell_agg[l.owner[f] as usize];
        let inb = cell_agg[l.neighbour[f] as usize];
        face_coarse[f] = face_index(&pairs, (io.min(inb), io.max(inb)))?;
    }
    for &bf in xbf.iter() {
        let ic = cell_agg[l.b_face_cells[bf] as usize];
        let idc = cell_agg[l.b_nbr_cell[bf] as usize];
        bface_coarse[bf] = face_index(&pairs, (ic.min(idc), ic.max(idc)))?;
    }

    Ok(Aggregation {
        n_fine: n,
        n_pass1: n1,
        n_coarse: n2,
        cell_agg,
        agg_offset,
        agg_cells,
        coarse_owner,
        coarse_neighbour,
        face_coarse,
        face_flip,
        bface_coarse,
        bface_flip,
    })
}

/// The index of a coarse face known to exist in the sorted list.
fn face_index(pairs: &[(Label, Label)], p: (Label, Label)) -> Result<Label> {
    match pairs.binary_search(&p) {
        Ok(i) => Ok(i as Label),
        Err(_) => Err(Error::Mesh(format!(
            "amg::aggregate: coarse face {:?} missing from the sorted list",
            p
        ))),
    }
}

/// One pairwise pass (Algorithm 2.1) over a strength list.
fn one_two_passes(n: usize, edges: &[(usize, usize, f64)]) -> Result<Vec<[Label; 2]>> {
    let (row_start, nbr_col, nbr_s, maxpos) = positive_csr(n, edges);
    let m = initial_m(n, &row_start, &nbr_col, &nbr_s, &maxpos);
    one_pass(n, &row_start, &nbr_col, &nbr_s, &maxpos, m)
}

/// CSR over the edges of positive strength, plus each node's strongest
/// positive strength (the max of (141.5)). Edges arrive ascending by
/// (min, max), so every row ascends by neighbour id.
fn positive_csr(
    n: usize,
    edges: &[(usize, usize, f64)],
) -> (Vec<usize>, Vec<Label>, Vec<f64>, Vec<f64>) {
    let mut row_start = vec![0usize; n + 1];
    for &(a, b, s) in edges {
        if s > 0.0 {
            row_start[a + 1] += 1;
            row_start[b + 1] += 1;
        }
    }
    for i in 0..n {
        row_start[i + 1] += row_start[i];
    }
    let mut cur = row_start.clone();
    let mut nbr_col = vec![0 as Label; row_start[n]];
    let mut nbr_s = vec![0.0f64; row_start[n]];
    let mut maxpos = vec![0.0f64; n];
    for &(a, b, s) in edges {
        if s <= 0.0 {
            continue;
        }
        nbr_col[cur[a]] = b as Label;
        nbr_s[cur[a]] = s;
        cur[a] += 1;
        nbr_col[cur[b]] = a as Label;
        nbr_s[cur[b]] = s;
        cur[b] += 1;
        if s > maxpos[a] {
            maxpos[a] = s;
        }
        if s > maxpos[b] {
            maxpos[b] = s;
        }
    }
    (row_start, nbr_col, nbr_s, maxpos)
}

/// `m_i = |{ j : i in S_j }|` (141.6) from the fixed S sets, computed once
/// before any node is marked.
fn initial_m(
    n: usize,
    row_start: &[usize],
    nbr_col: &[Label],
    nbr_s: &[f64],
    maxpos: &[f64],
) -> Vec<i64> {
    let mut m = vec![0i64; n];
    for k in 0..n {
        let thr = BETA * maxpos[k];
        for e in row_start[k]..row_start[k + 1] {
            if nbr_s[e] > thr {
                m[nbr_col[e] as usize] += 1;
            }
        }
    }
    m
}

/// Number one pass's aggregates by (141.8) - ascending smallest member - and
/// map every node to its numbered aggregate.
fn pass_map(n: usize, aggs: &[[Label; 2]]) -> Vec<Label> {
    let mut order: Vec<usize> = (0..aggs.len()).collect();
    order.sort_by_key(|&g| aggs[g][0].min(aggs[g][1]));
    let mut num = vec![0 as Label; aggs.len()];
    for (new, &g) in order.iter().enumerate() {
        num[g] = new as Label;
    }
    let mut map = vec![0 as Label; n];
    for (g, pair) in aggs.iter().enumerate() {
        for &k in pair {
            map[k as usize] = num[g];
        }
    }
    map
}

/// Algorithm 2.1 with `CheckDD = false` (SPEC-LIT §141.1): take the unmarked
/// node of smallest `m_i` (ties to the smaller id), pair it with its
/// strongest unmarked positive neighbour when that neighbour is in `S_i`,
/// mark the members, decrement the counters in their S sets, repeat.
fn one_pass(
    n: usize,
    row_start: &[usize],
    nbr_col: &[Label],
    nbr_s: &[f64],
    maxpos: &[f64],
    mut m: Vec<i64>,
) -> Result<Vec<[Label; 2]>> {
    let mut marked = vec![false; n];
    // Min-heap on (m_i, id); stale entries are discarded on pop.
    let mut heap: BinaryHeap<Reverse<(i64, Label)>> = BinaryHeap::with_capacity(n);
    for i in 0..n {
        heap.push(Reverse((m[i], i as Label)));
    }
    let mut aggs: Vec<[Label; 2]> = Vec::new();
    while let Some(Reverse((mv, i))) = heap.pop() {
        let iu = i as usize;
        if marked[iu] || m[iu] != mv {
            continue; // a stale or already-aggregated entry
        }
        // Step 2: the strongest unmarked positive neighbour, ties to the
        // smaller id.
        let mut j: Option<Label> = None;
        let mut best = 0.0f64;
        for e in row_start[iu]..row_start[iu + 1] {
            let k = nbr_col[e];
            let s = nbr_s[e];
            if marked[k as usize] || s <= 0.0 {
                continue;
            }
            let take = match j {
                None => true,
                Some(j) => s > best || (s == best && k < j),
            };
            if take {
                j = Some(k);
                best = s;
            }
        }
        // Step 3: pair only when j is in S_i, that is s_ij > BETA maxpos_i.
        let partner = j.filter(|_| best > BETA * maxpos[iu]);
        // Steps 4 and 5: mark the members, decrement their S sets.
        marked[iu] = true;
        decrement(iu, row_start, nbr_col, nbr_s, maxpos, &marked, &mut m, &mut heap);
        if let Some(j) = partner {
            let ju = j as usize;
            marked[ju] = true;
            decrement(ju, row_start, nbr_col, nbr_s, maxpos, &marked, &mut m, &mut heap);
            aggs.push([i, j]);
        } else {
            aggs.push([i, i]);
        }
    }
    let left = marked.iter().filter(|&&x| !x).count();
    if left != 0 {
        return Err(Error::Mesh(format!(
            "amg::aggregate: the pass left {} of {} nodes unmarked",
            left, n
        )));
    }
    Ok(aggs)
}

/// Step 5: every `l` in `S_k` loses one from its counter, and the fresh heap
/// entry supersedes the stale one (lazy deletion).
fn decrement(
    k: usize,
    row_start: &[usize],
    nbr_col: &[Label],
    nbr_s: &[f64],
    maxpos: &[f64],
    marked: &[bool],
    m: &mut [i64],
    heap: &mut BinaryHeap<Reverse<(i64, Label)>>,
) {
    let thr = BETA * maxpos[k];
    for e in row_start[k]..row_start[k + 1] {
        if nbr_s[e] > thr {
            let l = nbr_col[e] as usize;
            m[l] -= 1;
            if !marked[l] {
                heap.push(Reverse((m[l], l as Label)));
            }
        }
    }
}
