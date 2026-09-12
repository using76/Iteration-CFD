// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The true octree, and the real-point mesh it emits - SPEC-LIT §92.9.
//!
//! Over the stage-0 base grid a leaf is a key: a level and an index in that
//! level's own lattice, eq. (92.17); [`LeafKey::child`] is (92.17)'s `split`
//! and [`LeafKey::ancestor`] is (92.18). A leaf finds its neighbour by
//! lookup, not by a tree walk - the four cases of eq. (92.19) are
//! [`Neighbour`]. The 2:1 balance of §74.2, restated over the sparse leaf
//! set as eq. (92.21), is [`Octree::balance_2to1`]: levels only rise and
//! integer `max` is associative, so the fixed point does not depend on the
//! order the leaves are visited. Below the tree sits its emission: the
//! background block's node arrays, (92.20)'s point rule, `emit`'s real-point
//! `PolyMeshRaw`, and the level (92.1)/(92.22) asks of a leaf given the
//! surface. Nothing here classifies a cell against the surface or moves a
//! point - castellation and snapping are later units.
//!
//! Provenance: ORIGINAL - the key packing, the neighbour cases and the
//! balance sweep are this project's own, stated in SPEC-LIT §92.9 with
//! equations (92.17)-(92.19), (92.21) and (92.22), against the static
//! per-base-cell twin `mesh::refined::balance_2to1` of §74.2. The 2:1 face
//! convention and the face winding are NOT new: they are `mesh::refined`'s
//! (§74) and `blockgen`'s, reused so that §74's measured interface constants
//! stay true of these meshes and a level-0 tree reproduces
//! `blockgen::raw_mesh` point for point. No GPL-licensed source was
//! consulted.

use std::collections::HashSet;

use crate::error::{Error, Result};

/// Index bits per axis in a packed key (§92.9): a level-6 lattice index must
/// fit, so the base grid is capped at 2^20 / 2^max_level cells per axis.
pub const INDEX_BITS: u32 = 20;

/// The maximum level, SPEC-LIT §74.2's cap, quoted here as `refinement.max_level`
/// is validated against it in `AutomeshConfig::validate`.
pub const MAX_LEVEL: u32 = 6;

/// One leaf key of (92.17): a level and its index in that level's own lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LeafKey {
    pub level: u32,
    pub idx: [u32; 3],
}

impl LeafKey {
    /// The key as one 64-bit integer: `(level << 60) | (k << 40) | (j << 20) | i`.
    /// [`INDEX_BITS`] per axis and 4 bits of level are all a valid key needs;
    /// [`Octree::uniform`] refuses a base grid that would overrun them.
    pub fn pack(self) -> u64 {
        (u64::from(self.level) << 60)
            | (u64::from(self.idx[2]) << 40)
            | (u64::from(self.idx[1]) << 20)
            | u64::from(self.idx[0])
    }

    /// The inverse of [`LeafKey::pack`].
    pub fn unpack(v: u64) -> LeafKey {
        LeafKey {
            level: ((v >> 60) & 0xF) as u32,
            idx: [(v & 0xF_FFFF) as u32, ((v >> 20) & 0xF_FFFF) as u32, ((v >> 40) & 0xF_FFFF) as u32],
        }
    }

    /// (92.17)'s `split`: the eight children `(l+1, 2i+a, 2j+b, 2k+c)` for
    /// `a, b, c in {0, 1}`. `a` indexes axis 0, `b` axis 1, `c` axis 2.
    pub fn child(self, a: u32, b: u32, c: u32) -> LeafKey {
        LeafKey {
            level: self.level + 1,
            idx: [2 * self.idx[0] + a, 2 * self.idx[1] + b, 2 * self.idx[2] + c],
        }
    }

    /// (92.18): the ancestor of this key at `level`, the index shifted right
    /// by the level difference. Panic-free: `level >= self.level` returns
    /// `self` unchanged.
    pub fn ancestor(self, level: u32) -> LeafKey {
        if level >= self.level {
            return self;
        }
        let d = self.level - level;
        LeafKey { level, idx: [self.idx[0] >> d, self.idx[1] >> d, self.idx[2] >> d] }
    }

    /// This leaf's lower corner ON THE FINEST LATTICE:
    /// `idx[a] << (max_level - level)`. The sort key §92.9 fixes the cell
    /// numbering with - see [`Octree::leaves`].
    pub fn lower_corner(self, max_level: u32) -> [u64; 3] {
        let s = max_level.saturating_sub(self.level);
        [u64::from(self.idx[0]) << s, u64::from(self.idx[1]) << s, u64::from(self.idx[2]) << s]
    }

    /// This leaf's edge length on the finest lattice: `1 << (max_level - level)`.
    pub fn size_on_finest(self, max_level: u32) -> u64 {
        1u64 << (max_level - self.level)
    }
}

/// What lies across one face of a leaf - the four cases of (92.19).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Neighbour {
    /// The index left the domain: this face is on the box boundary.
    Outside,
    /// A leaf at the same level.
    Same(LeafKey),
    /// A leaf at a strictly coarser level (this leaf is the finer side).
    Coarser(LeafKey),
    /// The other side is refined past this leaf's level.
    Finer,
}

/// The tree: a sparse set of leaf keys over a `base_n` grid of base cells.
#[derive(Debug)]
pub struct Octree {
    base_n: [usize; 3],
    max_level: u32,
    keys: HashSet<u64>,
}

impl Octree {
    /// Every base cell one level-0 leaf - §92.9's stage-0 tree. Refuses a
    /// zero axis, a `max_level` past [`MAX_LEVEL`], and a base grid whose
    /// level-`max_level` lattice would not fit [`INDEX_BITS`] bits per axis
    /// (the packed key would overflow), naming the axis and both numbers.
    pub fn uniform(base_n: [usize; 3], max_level: u32) -> Result<Octree> {
        for a in 0..3 {
            if base_n[a] == 0 {
                return Err(Error::Mesh(format!(
                    "octree: base_n[{}] is 0; every axis needs at least one base cell",
                    a
                )));
            }
        }
        if max_level > MAX_LEVEL {
            return Err(Error::Mesh(format!(
                "octree: max_level {} exceeds MAX_LEVEL {} (SPEC-LIT §74.2's cap)",
                max_level, MAX_LEVEL
            )));
        }
        let limit = 1u64 << INDEX_BITS;
        for a in 0..3 {
            let n_l = (base_n[a] as u128) << max_level;
            if n_l >= u128::from(limit) {
                return Err(Error::Mesh(format!(
                    "octree: base_n[{}] = {} at max_level {} gives {} cells on the axis, \
                     past the {} = 2^INDEX_BITS a packed key holds",
                    a, base_n[a], max_level, n_l, limit
                )));
            }
        }
        let mut keys = HashSet::with_capacity(base_n[0] * base_n[1] * base_n[2]);
        for k in 0..base_n[2] {
            for j in 0..base_n[1] {
                for i in 0..base_n[0] {
                    keys.insert(LeafKey { level: 0, idx: [i as u32, j as u32, k as u32] }.pack());
                }
            }
        }
        Ok(Octree { base_n, max_level, keys })
    }

    /// The base grid this tree was built over.
    pub fn base_n(&self) -> [usize; 3] {
        self.base_n
    }

    /// The level cap this tree was built with.
    pub fn max_level(&self) -> u32 {
        self.max_level
    }

    /// The number of leaves.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether the tree has no leaves (only the empty `uniform` degenerate).
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Exact key membership.
    pub fn contains(&self, key: LeafKey) -> bool {
        self.keys.contains(&key.pack())
    }

    /// The present leaf covering that cell of that level's own lattice, if it
    /// is at `level` or COARSER: the first present key among the ancestors of
    /// `(level, idx)`, walked from `level` down to 0. `None` means the region
    /// is refined finer than `level` (or the index is outside the lattice).
    pub fn leaf_at(&self, level: u32, idx: [u32; 3]) -> Option<LeafKey> {
        for l2 in (0..=level.min(MAX_LEVEL)).rev() {
            let d = level - l2;
            let key = LeafKey { level: l2, idx: [idx[0] >> d, idx[1] >> d, idx[2] >> d] };
            if self.contains(key) {
                return Some(key);
            }
        }
        None
    }

    /// Remove `key`, insert its eight children - (92.17)'s `split`. Refuses a
    /// key that is not a present leaf, and a key already at `max_level`.
    pub fn split(&mut self, key: LeafKey) -> Result<()> {
        if !self.contains(key) {
            return Err(Error::Mesh(format!(
                "octree: cannot split {:?} - it is not a present leaf",
                key
            )));
        }
        if key.level == self.max_level {
            return Err(Error::Mesh(format!(
                "octree: cannot split {:?} - it is already at max_level {}",
                key, self.max_level
            )));
        }
        self.keys.remove(&key.pack());
        for a in 0..2u32 {
            for b in 0..2u32 {
                for c in 0..2u32 {
                    let ch = key.child(a, b, c);
                    self.keys.insert(ch.pack());
                }
            }
        }
        Ok(())
    }

    /// Who is across the face of `key` in direction `axis`/`positive` - the
    /// four cases of (92.19), each decided by a lookup, not a walk.
    pub fn neighbour(&self, key: LeafKey, axis: usize, positive: bool) -> Neighbour {
        let step: i64 = if positive { 1 } else { -1 };
        let m = key.idx[axis] as i64 + step;
        let n_l = (self.base_n[axis] as u64) << key.level;
        if m < 0 || m >= n_l as i64 {
            return Neighbour::Outside;
        }
        let mut idx = key.idx;
        idx[axis] = m as u32;
        match self.leaf_at(key.level, idx) {
            Some(q) if q.level == key.level => Neighbour::Same(q),
            Some(q) => Neighbour::Coarser(q),
            None => Neighbour::Finer,
        }
    }

    /// Every present key, sorted by packed value. Every iterating method goes
    /// through this, so no pass depends on the `HashSet`'s iteration order.
    fn sorted(&self) -> Vec<LeafKey> {
        let mut v: Vec<LeafKey> = self.keys.iter().map(|p| LeafKey::unpack(*p)).collect();
        v.sort_by_key(|k| k.pack());
        v
    }

    /// §92.2 stage 1's level-driven refinement with the criterion LEFT OUT (a
    /// later unit supplies the closure): repeat until nothing changes - for
    /// every present leaf whose `level < min(target(leaf), max_level)`, split
    /// it. `target` is asked per PRESENT leaf, so a criterion on a whole base
    /// cell must key its answer through `ancestor(0)`; a split leaf's children
    /// have their own keys. The leaves of a pass are collected and sorted
    /// before mutating, so the pass is order-independent. Returns the number
    /// of splits.
    pub fn refine(&mut self, target: impl Fn(LeafKey) -> u32) -> usize {
        let mut splits = 0usize;
        loop {
            let mut pass: Vec<LeafKey> = self
                .sorted()
                .into_iter()
                .filter(|k| k.level < target(*k).min(self.max_level))
                .collect();
            if pass.is_empty() {
                return splits;
            }
            for k in pass.drain(..) {
                // Every key here is a present leaf below the cap, so the split
                // cannot fail. Returning on a failure rather than retrying is
                // what keeps a future bug a wrong answer instead of a mesher
                // that spins on the same pass forever.
                if self.split(k).is_err() {
                    return splits;
                }
                splits += 1;
            }
        }
    }

    /// (92.21), read from the FINER side, which is what makes it a lookup:
    /// repeat until nothing changes - for every leaf `P` at level `l`, for
    /// each of the 6 directions, if the neighbour is `Coarser(q)` with
    /// `q.level < l - 1`, mark `q` to be split; then split all marked keys
    /// (sorted, so the pass is order-independent), and repeat. Returns the
    /// number of splits. A split here produces a level of at most
    /// `l - 1 < max_level`, so balance can never exceed the cap of §74.2 -
    /// no second check against `max_level` is needed.
    pub fn balance_2to1(&mut self) -> usize {
        let mut splits = 0usize;
        loop {
            let mut marked: Vec<LeafKey> = Vec::new();
            for p in self.sorted() {
                for axis in 0..3 {
                    for positive in [false, true] {
                        if let Neighbour::Coarser(q) = self.neighbour(p, axis, positive) {
                            if q.level + 1 < p.level {
                                marked.push(q);
                            }
                        }
                    }
                }
            }
            if marked.is_empty() {
                return splits;
            }
            marked.sort_by_key(|k| k.pack());
            marked.dedup();
            for q in marked {
                // As in `refine`: a marked key is a present leaf strictly
                // below the finer side's level, so this cannot fail, and a
                // failure must end the sweep rather than repeat it.
                if self.split(q).is_err() {
                    return splits;
                }
                splits += 1;
            }
        }
    }

    /// The largest `|level(P) - level(N)|` over face-adjacent leaf pairs,
    /// §92.9's table row for (92.21). `Same` gives 0; `Coarser(q)` gives
    /// `P.level - q.level`; `Finer` is covered when the finer side asks.
    pub fn max_level_jump(&self) -> u32 {
        let mut jump = 0u32;
        for p in self.sorted() {
            for axis in 0..3 {
                for positive in [false, true] {
                    if let Neighbour::Coarser(q) = self.neighbour(p, axis, positive) {
                        jump = jump.max(p.level - q.level);
                    }
                }
            }
        }
        jump
    }

    /// Every leaf, sorted by `(lower_corner[2], lower_corner[1],
    /// lower_corner[0])` on the finest lattice. This IS §92.9's cell
    /// numbering - lexicographic `(q_k, q_j, q_i)` order of the leaf's lower
    /// corner, stage 0's own cell order on an unrefined tree - and every
    /// later unit reads the leaves in exactly this order.
    pub fn leaves(&self) -> Vec<LeafKey> {
        let mut v = self.sorted();
        v.sort_by_key(|k| {
            let q = k.lower_corner(self.max_level);
            (q[2], q[1], q[0])
        });
        v
    }

    /// §92.9's partition invariant, asserted not assumed. (a) No present key
    /// has a present ancestor. (b) The leaves' volumes on the finest lattice
    /// sum to `nx ny nz 8^max_level` exactly. Together: the leaf set is a
    /// partition of the domain, and a refusal names both offending numbers.
    pub fn check_partition(&self) -> Result<()> {
        for p in self.sorted() {
            for l2 in (0..p.level).rev() {
                let anc = p.ancestor(l2);
                if self.contains(anc) {
                    return Err(Error::Mesh(format!(
                        "octree: {:?} and its ancestor {:?} are both present - \
                         the leaf set is not a partition",
                        p, anc
                    )));
                }
            }
        }
        let mut vol: u128 = 0;
        for p in self.sorted() {
            let s = u128::from(p.size_on_finest(self.max_level));
            vol += s * s * s;
        }
        let total = u128::from(
            self.base_n[0] as u128 * self.base_n[1] as u128 * self.base_n[2] as u128,
        ) << (3 * self.max_level);
        if vol != total {
            return Err(Error::Mesh(format!(
                "octree: leaf volumes sum to {} on the finest lattice, \
                 not the domain's {}",
                vol, total
            )));
        }
        Ok(())
    }
}


// ==========================================================================
//  The background block and the mesh a tree emits - SPEC-LIT §92.9
// ==========================================================================
//
// (92.20) is where a point's coordinate comes from and (92.19) is who emits
// each face; this section turns an [`Octree`] into a real-point
// `PolyMeshRaw` with no surface, no distance bands and no criterion - the
// criterion is the next unit's. It reads `blockgen`'s and `mesh::refined`'s
// conventions and rewrites neither. Provenance: ORIGINAL. No GPL-licensed
// source was consulted.

use std::collections::HashMap;

/// The stage-0 background block (§92.2 stage 0) as the octree needs it: the
/// per-axis graded node arrays (92.20) interpolates inside, and the base cell
/// counts (92.17)'s lattice is built over.
pub struct Background {
    pub axes: [crate::blockgen::GradedAxis; 3],
    pub nodes: [Vec<crate::Scalar>; 3],
}

impl Background {
    /// The block a `domain` asks for: `n_a = max(1, round((hi - lo)/base_size))`
    /// per axis, one-sided `grading` as the expansion ratio. Refuses an empty
    /// or reversed axis and a non-positive `base_size`, naming the field.
    pub fn from_domain(d: &super::DomainSpec) -> crate::error::Result<Background> {
        if !(d.base_size > 0.0) {
            return Err(Error::Mesh(format!(
                "background: base_size is {:?}, but stage 0 needs a positive cell size",
                d.base_size
            )));
        }
        let mut axes = [
            crate::blockgen::GradedAxis::default(),
            crate::blockgen::GradedAxis::default(),
            crate::blockgen::GradedAxis::default(),
        ];
        for a in 0..3 {
            let (lo, hi) = (d.extent[2 * a], d.extent[2 * a + 1]);
            if !(hi > lo) {
                return Err(Error::Mesh(format!(
                    "background: extent axis {} is [{:?}, {:?}] - empty or reversed",
                    a, lo, hi
                )));
            }
            let n = (((hi - lo) / d.base_size).round() as usize).max(1);
            axes[a] = crate::blockgen::GradedAxis {
                lo,
                hi,
                n,
                expansion: d.grading[a],
                two_sided: false,
            };
        }
        let nodes = [
            crate::blockgen::graded_nodes(&axes[0]),
            crate::blockgen::graded_nodes(&axes[1]),
            crate::blockgen::graded_nodes(&axes[2]),
        ];
        Ok(Background { axes, nodes })
    }

    /// The base cell counts, [`Octree::uniform`]'s argument.
    pub fn base_n(&self) -> [usize; 3] {
        [self.axes[0].n, self.axes[1].n, self.axes[2].n]
    }

    /// The same block as a [`crate::blockgen::BlockSpec`], with
    /// [`patch_names`]'s names and `"patch"` as every type - so stage 0's
    /// block and the octree's level-0 tree are provably the same mesh (the
    /// test below asserts it).
    pub fn block_spec(&self) -> crate::blockgen::BlockSpec {
        crate::blockgen::BlockSpec {
            x: self.axes[0].clone(),
            y: self.axes[1].clone(),
            z: self.axes[2].clone(),
            patch_name: patch_names(),
            patch_type: ["patch"; 6].map(String::from),
            windows: Vec::new(),
            cyclic: Vec::new(),
        }
    }

    /// (92.20): the coordinate of finest-lattice index `q` on `axis`, where
    /// `max_level` is the tree's. `t == 0` returns `nodes[axis][b]` by an
    /// array read and no arithmetic, which is what makes a level-0 tree
    /// reproduce `blockgen::raw_mesh`'s points bit for bit.
    pub fn coord(&self, axis: usize, q: u64, max_level: u32) -> crate::Scalar {
        let n = self.axes[axis].n;
        let b = (q >> max_level) as usize;
        let r = q - ((b as u64) << max_level);
        if r == 0 {
            // The guard: q may sit on the axis' far face, where b == n and
            // nodes[n] is the axis' last graded node.
            return self.nodes[axis][b.min(n)];
        }
        let t = r as crate::Scalar / (1u64 << max_level) as crate::Scalar;
        self.nodes[axis][b] + t * (self.nodes[axis][b + 1] - self.nodes[axis][b])
    }
}

/// The six box patch names, in the `-x +x -y +y -z +z` slot order every patch
/// list in this crate uses. `blockgen::BlockSpec::default`'s own names.
pub fn patch_names() -> [String; 6] {
    ["xMin", "xMax", "yMin", "yMax", "zMin", "zMax"].map(String::from)
}

/// One face (92.19) emits, before its points are numbered: the quad on the
/// finest lattice and, across an internal face, its two cells in `low`/`high`
/// order along `+axis`.
struct FaceRec {
    /// `Some((low, high))` across the face, `None` on the box boundary.
    int: Option<(usize, usize)>,
    /// The boundary patch slot `2*axis + positive`; meaningless when `int` is
    /// `Some`.
    patch: usize,
    /// The emitting cell - a boundary face's owner.
    owner: usize,
    axis: usize,
    /// The face's constant lattice coordinate on `axis`.
    qc: u64,
    /// `[a0, a1, b0, b1]`: the span on `(axis+1)%3`, then on `(axis+2)%3`.
    span: [u64; 4],
}

/// The face's four corners on the finest lattice, wound so `Sf` points along
/// `+axis`: `mesh::refined::build`'s `corners` closure and
/// `blockgen::internal_quad`'s winding on integer lattice coordinates -
/// `(a0,b0), (a1,b0), (a1,b1), (a0,b1)`.
fn quad(r: &FaceRec) -> [[u64; 3]; 4] {
    let t1 = (r.axis + 1) % 3;
    let t2 = (r.axis + 2) % 3;
    let mk = |a: u64, b: u64| {
        let mut p = [0u64; 3];
        p[r.axis] = r.qc;
        p[t1] = a;
        p[t2] = b;
        p
    };
    [
        mk(r.span[0], r.span[2]),
        mk(r.span[1], r.span[2]),
        mk(r.span[1], r.span[3]),
        mk(r.span[0], r.span[3]),
    ]
}

/// Every face (92.19) emits, handed to `sink` in a deterministic order: the
/// leaves in [`Octree::leaves`] order, each axis, `-` then `+`. The span is
/// always the EMITTING cell's face extent, which for `Same` and `Coarser`
/// alike is the finer of the two cells - (92.19)'s "every internal face is a
/// whole face of the finer of the two cells it separates".
fn walk(tree: &Octree, cell_of: &HashMap<u64, usize>, sink: &mut impl FnMut(FaceRec)) {
    for p in tree.leaves() {
        let id = cell_of[&p.pack()];
        let q0 = p.lower_corner(tree.max_level());
        let s = p.size_on_finest(tree.max_level());
        for axis in 0..3 {
            let t1 = (axis + 1) % 3;
            let t2 = (axis + 2) % 3;
            let span = [q0[t1], q0[t1] + s, q0[t2], q0[t2] + s];
            for positive in [false, true] {
                let qc = if positive { q0[axis] + s } else { q0[axis] };
                match tree.neighbour(p, axis, positive) {
                    Neighbour::Outside => sink(FaceRec {
                        int: None,
                        patch: 2 * axis + positive as usize,
                        owner: id,
                        axis,
                        qc,
                        span,
                    }),
                    // `Same` is emitted on the `+` side alone, so a pair is
                    // emitted once.
                    Neighbour::Same(q) => {
                        if positive {
                            sink(FaceRec {
                                int: Some((id, cell_of[&q.pack()])),
                                patch: 0,
                                owner: id,
                                axis,
                                qc,
                                span,
                            });
                        }
                    }
                    // This leaf is the finer side: it emits the whole face.
                    Neighbour::Coarser(q) => {
                        let other = cell_of[&q.pack()];
                        sink(FaceRec {
                            int: Some(if positive { (id, other) } else { (other, id) }),
                            patch: 0,
                            owner: id,
                            axis,
                            qc,
                            span,
                        });
                    }
                    // The finer side emits it.
                    Neighbour::Finer => {}
                }
            }
        }
    }
}

/// The finest-lattice coordinate of every point [`emit`] numbers, in
/// `emit`'s own point order: `point_lattice(tree)[i]` is the lattice
/// triple of `emit(tree, ..).points[i]`. §92.10's pinch test needs the
/// integer coordinates the float points were made from.
pub fn point_lattice(tree: &Octree) -> Vec<[u64; 3]> {
    let leaves = tree.leaves();
    let mut cell_of: HashMap<u64, usize> = HashMap::with_capacity(leaves.len());
    for (id, key) in leaves.iter().enumerate() {
        cell_of.insert(key.pack(), id);
    }
    lattice_points(tree, &cell_of)
}

/// [`emit`]'s pass 1, verbatim: the lattice triples of every face corner the
/// walk emits, welded into a set by integer identity and sorted into §92.9's
/// point order. One body, two callers - [`emit`] numbers its points from it
/// and [`point_lattice`] exposes it - so the two orderings cannot drift.
fn lattice_points(tree: &Octree, cell_of: &HashMap<u64, usize>) -> Vec<[u64; 3]> {
    let mut used: HashSet<[u64; 3]> = HashSet::new();
    walk(tree, cell_of, &mut |r| {
        for c in quad(&r) {
            used.insert(c);
        }
    });
    // §92.9's point order: (q_k, q_j, q_i), z slowest, x fastest - the order
    // stage 0's block numbers its own.
    let mut pts: Vec<[u64; 3]> = used.drain().collect();
    pts.sort_unstable_by_key(|q| (q[2], q[1], q[0]));
    pts
}

/// Emit the leaf mesh of `tree` over `bg` as a REAL-POINT `PolyMeshRaw`
/// (§92.9): unique points welded by integer lattice identity, one cell per
/// leaf in [`Octree::leaves`] order, internal faces by (92.19), the six box
/// patches in slot order. Runs [`Octree::check_partition`] first and refuses
/// if the tree is not a partition.
pub fn emit(
    tree: &Octree,
    bg: &Background,
    names: &[String; 6],
) -> crate::error::Result<crate::io::polymesh::PolyMeshRaw> {
    tree.check_partition()?;
    let tn = tree.base_n();
    let bn = bg.base_n();
    if tn != bn {
        return Err(Error::Mesh(format!(
            "octree::emit: the tree's base grid is {:?} but the block's is {:?} - \
             not the same block",
            tn, bn
        )));
    }
    let leaves = tree.leaves();
    let mut cell_of: HashMap<u64, usize> = HashMap::with_capacity(leaves.len());
    for (id, key) in leaves.iter().enumerate() {
        cell_of.insert(key.pack(), id);
    }
    let level = tree.max_level();

    // Pass 1: the point set. A point IS its lattice coordinate - welded by
    // integer identity, with no float comparison and no tolerance anywhere
    // in this function. `lattice_points` holds the body and `point_lattice`
    // is its public face, so §92.10's pinch test reads the same numbering.
    let pts = lattice_points(tree, &cell_of);
    let mut label_of: HashMap<[u64; 3], crate::Label> = HashMap::with_capacity(pts.len());
    for (n, q) in pts.iter().enumerate() {
        label_of.insert(*q, n as crate::Label);
    }

    // Pass 2: the same faces in the same order - the walk is a pure function
    // of the tree - their corners now numbered.
    let mut internal: Vec<(usize, usize, [crate::Label; 4])> = Vec::new();
    let mut boundary: Vec<Vec<(u64, u64, usize, [crate::Label; 4])>> = vec![Vec::new(); 6];
    walk(tree, &cell_of, &mut |r| {
        let qs = quad(&r);
        let mut ps = [0 as crate::Label; 4];
        for (n, c) in qs.iter().enumerate() {
            ps[n] = label_of[c];
        }
        match r.int {
            Some((low, high)) => internal.push((low, high, ps)),
            None => {
                // blockgen's boundary order: the lower-numbered tangential
                // direction fastest, so the sort key is (slow, fast).
                let t1 = (r.axis + 1) % 3;
                let t2 = (r.axis + 2) % 3;
                let (fast, slow) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
                // A `-` slot keeps `Sf` pointing out of the domain. blockgen's
                // own xMin/yMin/zMin winding lists the SAME cycle the `+axis`
                // quad runs, but from (a0,b0) the other way round - not the
                // bare reversal - so the emitted list is [P0, P3, P2, P1].
                let ps = if r.patch % 2 == 0 {
                    [ps[0], ps[3], ps[2], ps[1]]
                } else {
                    ps
                };
                boundary[r.patch].push((qs[0][slow], qs[0][fast], r.owner, ps));
            }
        }
    });

    // §92.9 "Order": a face built with owner > neighbour is flipped, its
    // point list reversed with the pair, so `Sf` keeps pointing from owner
    // to neighbour; then sorted by (owner, neighbour) - §2's upper-triangular
    // order, reached by construction and asserted here rather than repaired
    // afterwards.
    let mut internal: Vec<(usize, usize, Vec<crate::Label>)> = internal
        .into_iter()
        .map(|(low, high, ps)| {
            if low > high {
                (high, low, vec![ps[3], ps[2], ps[1], ps[0]])
            } else {
                (low, high, ps.to_vec())
            }
        })
        .collect();
    internal.sort_by_key(|(o, n, _)| (*o, *n));
    for (o, n, _) in &internal {
        if o >= n {
            return Err(Error::Mesh(format!(
                "octree::emit: face has owner {} >= neighbour {} - not upper-triangular",
                o, n
            )));
        }
    }
    for w in internal.windows(2) {
        if w[0].0 == w[1].0 && w[0].1 == w[1].1 {
            return Err(Error::Mesh(format!(
                "octree::emit: cells {} and {} share more than one face - \
                 the leaf set is not a partition",
                w[0].0, w[0].1
            )));
        }
    }

    // Internal faces first, then the six patches in slot order; a patch's
    // `start` counts from the FIRST boundary face.
    let n_internal = internal.len();
    let mut faces: Vec<Vec<crate::Label>> = Vec::new();
    let mut owner: Vec<crate::Label> = Vec::new();
    let mut neighbour: Vec<crate::Label> = Vec::with_capacity(n_internal);
    for (o, n, ps) in &internal {
        faces.push(ps.clone());
        owner.push(*o as crate::Label);
        neighbour.push(*n as crate::Label);
    }
    let mut patches: Vec<crate::mesh::PatchInfo> = Vec::with_capacity(6);
    for slot in 0..6 {
        let mut bf = std::mem::take(&mut boundary[slot]);
        bf.sort_by_key(|(slow, fast, _, _)| (*slow, *fast));
        let start = faces.len() - n_internal;
        for (_, _, o, ps) in &bf {
            faces.push(ps.to_vec());
            owner.push(*o as crate::Label);
        }
        patches.push(crate::mesh::PatchInfo {
            name: names[slot].clone(),
            type_name: "patch".to_string(),
            kind: crate::mesh::PatchKind::from_type("patch"),
            start,
            size: bf.len(),
            nbr_patch: None,
        });
    }

    // (92.20): the real coordinates, interpolated inside the base cell each
    // lattice point falls in.
    let points: Vec<crate::Vec3> = pts
        .iter()
        .map(|q| {
            crate::Vec3::new(
                bg.coord(0, q[0], level),
                bg.coord(1, q[1], level),
                bg.coord(2, q[2], level),
            )
        })
        .collect();

    Ok(crate::io::polymesh::PolyMeshRaw {
        points,
        faces,
        owner,
        neighbour,
        patches,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    #[test]
    fn a_uniform_tree_is_the_base_grid() {
        let t = Octree::uniform([4, 3, 2], 3).expect("uniform");
        assert_eq!(t.len(), 24);
        t.check_partition().expect("partition");
        assert_eq!(t.max_level_jump(), 0);
        let leaves = t.leaves();
        for (n, leaf) in leaves.iter().enumerate() {
            assert_eq!(leaf.level, 0, "leaf {} is not a base cell", n);
            let i = n % 4;
            let j = (n / 4) % 3;
            let k = n / 12;
            assert_eq!(leaf.idx, [i as u32, j as u32, k as u32], "leaf {} out of (k,j,i) order", n);
        }
    }

    #[test]
    fn splitting_one_leaf_gives_eight_children_and_keeps_the_partition() {
        let mut t = Octree::uniform([4, 3, 2], 3).expect("uniform");
        let parent = LeafKey { level: 0, idx: [1, 0, 0] };
        t.split(parent).expect("split");
        assert_eq!(t.len(), 24 + 7);
        t.check_partition().expect("partition");
        assert!(!t.contains(parent), "the parent is still present");
        for a in 0..2u32 {
            for b in 0..2u32 {
                for c in 0..2u32 {
                    assert!(t.contains(parent.child(a, b, c)), "child ({},{},{}) missing", a, b, c);
                }
            }
        }
        assert!(t.leaf_at(0, [1, 0, 0]).is_none(), "the region is finer than level 0 now");
    }

    #[test]
    fn a_key_and_its_pack_round_trip() {
        let max_idx = (1u32 << INDEX_BITS) - 1;
        for level in 0..=MAX_LEVEL {
            for idx in [
                [0, 0, 0],
                [1, 2, 3],
                [max_idx, max_idx, max_idx],
                [max_idx, 0, max_idx >> level],
            ] {
                let k = LeafKey { level, idx };
                assert_eq!(LeafKey::unpack(k.pack()), k, "round trip failed for {:?}", k);
            }
        }
        let k = LeafKey { level: 6, idx: [63, 41, 7] };
        assert_eq!(k.ancestor(2).ancestor(0), k.ancestor(0), "ancestor composes");
        assert_eq!(k.ancestor(5).ancestor(3), k.ancestor(3), "ancestor composes");
        assert_eq!(k.ancestor(6), k, "ancestor at own level is self");
        assert_eq!(k.ancestor(9), k, "ancestor past own level is self, panic-free");
    }

    /// `point_lattice` is `emit`'s point numbering on the integers: same
    /// length, and every entry lands exactly on the float point - on a tree
    /// with a 2:1 jump, where coarse and fine points interleave.
    #[test]
    fn point_lattice_matches_emit_on_a_two_to_one_tree() {
        use crate::automesher::DomainSpec;
        let bg = Background::from_domain(&DomainSpec {
            extent: [0.0, 4.0, 0.0, 4.0, 0.0, 4.0],
            base_size: 1.0,
            grading: [1.0; 3],
        })
        .expect("background");
        let mut tree = Octree::uniform(bg.base_n(), 1).expect("uniform");
        tree.refine(|k| if k.idx[0] % 2 == 0 { 1 } else { 0 });
        tree.balance_2to1();
        tree.check_partition().expect("partition");
        assert_eq!(tree.max_level_jump(), 1, "a band-refined, balanced tree");
        let mesh = emit(&tree, &bg, &patch_names()).expect("emit");
        let lat = point_lattice(&tree);
        assert_eq!(lat.len(), mesh.points.len());
        let l = tree.max_level();
        for (i, q) in lat.iter().enumerate() {
            let p = mesh.points[i];
            let xyz = [p.x, p.y, p.z];
            for a in 0..3 {
                assert_eq!(bg.coord(a, q[a], l), xyz[a], "axis {a} of point {i}");
            }
        }
    }

    /// A [3,3,3] tree, max_level 2: base cell [1,1,1] split to level 1, and
    /// its child (0,0,0) split again to level 2. Then one instance of each
    /// case of (92.19), by hand-computed key.
    #[test]
    fn the_neighbour_cases_are_the_four_of_92_19() {
        let mut t = Octree::uniform([3, 3, 3], 2).expect("uniform");
        let mid = LeafKey { level: 0, idx: [1, 1, 1] };
        t.split(mid).expect("split");
        let deep = mid.child(0, 0, 0); // (1, [2, 2, 2])
        t.split(deep).expect("split");

        // Outside: the -x face of base cell [0,0,0] leaves the domain.
        assert_eq!(t.neighbour(LeafKey { level: 0, idx: [0, 0, 0] }, 0, false), Neighbour::Outside);
        // Same: +x of [0,0,0] is the level-0 base cell [1,0,0].
        assert_eq!(
            t.neighbour(LeafKey { level: 0, idx: [0, 0, 0] }, 0, true),
            Neighbour::Same(LeafKey { level: 0, idx: [1, 0, 0] })
        );
        // Coarser: the level-2 leaf (2,[4,4,4]) looks -x at (2,[3,4,4]), whose
        // first present ancestor is the level-0 base cell [0,1,1].
        assert_eq!(
            t.neighbour(LeafKey { level: 2, idx: [4, 4, 4] }, 0, false),
            Neighbour::Coarser(LeafKey { level: 0, idx: [0, 1, 1] })
        );
        // Finer: +x of base cell [0,1,1] is the split base cell [1,1,1] -
        // the other side is refined past level 0, so nothing present is found.
        assert_eq!(t.neighbour(LeafKey { level: 0, idx: [0, 1, 1] }, 0, true), Neighbour::Finer);
    }

    /// §74.2's own balance test, on the sparse tree: one deep corner leaf in
    /// a [5,5,5] tree, max_level 3 - idempotent, and the same fixed point
    /// however the splits are ordered.
    #[test]
    fn balance_is_idempotent_and_order_independent() {
        let corner = LeafKey { level: 0, idx: [0, 0, 0] };
        let mut a = Octree::uniform([5, 5, 5], 3).expect("uniform");
        // target is asked per present leaf, so the corner's descendants are
        // keyed through ancestor(0) - a leaf's own key never equals `corner`
        // once it is split.
        let deep = |k: LeafKey| if k.ancestor(0) == corner { 3 } else { 0 };
        assert_eq!(a.refine(deep), 1 + 8 + 64);
        assert!(a.balance_2to1() > 0, "an unbalanced deep corner must force splits");
        assert_eq!(a.balance_2to1(), 0, "balancing again must split nothing");
        a.check_partition().expect("partition");
        assert_eq!(a.max_level_jump(), 1);

        // The same splits, deepest-available-first: the corner, its eight
        // children in reverse index order, then the 64 grandchildren reversed.
        let mut b = Octree::uniform([5, 5, 5], 3).expect("uniform");
        b.split(corner).expect("split");
        let mut l1: Vec<LeafKey> = Vec::new();
        for x in 0..2u32 {
            for y in 0..2u32 {
                for z in 0..2u32 {
                    l1.push(corner.child(x, y, z));
                }
            }
        }
        l1.reverse();
        for k in &l1 {
            b.split(*k).expect("split");
        }
        let mut l2: Vec<LeafKey> = Vec::new();
        for p in &l1 {
            for x in 0..2u32 {
                for y in 0..2u32 {
                    for z in 0..2u32 {
                        l2.push(p.child(x, y, z));
                    }
                }
            }
        }
        l2.reverse();
        for k in &l2 {
            b.split(*k).expect("split");
        }
        b.balance_2to1();
        assert_eq!(a.leaves(), b.leaves(), "balance reached a different fixed point");
    }

    /// With the cap at 2, a corner refined to 2 balances with the jump at 1
    /// and nothing anywhere past the cap.
    #[test]
    fn balance_never_passes_the_cap() {
        let corner = LeafKey { level: 0, idx: [0, 0, 0] };
        let mut t = Octree::uniform([3, 3, 3], 2).expect("uniform");
        t.refine(|k| if k.ancestor(0) == corner { 2 } else { 0 });
        assert!(t.balance_2to1() > 0);
        assert!(t.leaves().iter().all(|k| k.level <= 2), "a leaf sits past the cap");
        assert_eq!(t.max_level_jump(), 1);
        t.check_partition().expect("partition");
    }

    /// The closure's target is min'd with the cap: a corner asked for 5 gets
    /// exactly (2^2)^3 leaves and nothing else is touched.
    #[test]
    fn refine_uses_the_target_and_the_cap() {
        let corner = LeafKey { level: 0, idx: [0, 0, 0] };
        let mut t = Octree::uniform([2, 2, 2], 2).expect("uniform");
        // The subtree is keyed through ancestor(0): every leaf inside the
        // corner base cell answers 5, and the cap of 2 does the rest.
        let splits = t.refine(|k| if k.ancestor(0) == corner { 5 } else { 0 });
        assert_eq!(splits, 9, "1 split at level 0, 8 at level 1, the cap stops level 2");
        assert_eq!(t.len(), 8 - 1 + 64, "exactly 63 new leaves");
        t.check_partition().expect("partition");
        for leaf in t.leaves() {
            let q = leaf.lower_corner(2);
            let in_corner = q[0] < 4 && q[1] < 4 && q[2] < 4;
            assert_eq!(leaf.level, if in_corner { 2 } else { 0 }, "wrong level at {:?}", leaf);
        }
        assert!(t.contains(LeafKey { level: 0, idx: [1, 1, 1] }), "a far cell was refined");
    }

    /// 200_000 cells on an axis at level 6 is 12.8e6 on a lattice the packed
    /// key gives 2^20 of - refused, naming the axis and the limit.
    #[test]
    fn an_overflowing_base_grid_is_refused_by_name() {
        let err = Octree::uniform([200_000, 1, 1], 6).expect_err("must be refused");
        let msg = match err {
            Error::Mesh(m) => m,
            other => panic!("wrong error kind: {:?}", other),
        };
        assert!(msg.contains("base_n[0]"), "the axis is not named: {}", msg);
        let limit = 1u64 << INDEX_BITS;
        assert!(msg.contains(&limit.to_string()), "the limit is not named: {}", msg);
    }

    /// §92.9's first table row: a tree with every leaf at level 0 is
    /// `blockgen::raw_mesh`'s own block - the same points ARRAY bit for bit,
    /// the same cells, faces and patches.
    #[test]
    fn a_level_zero_tree_is_blockgens_own_block() {
        let bg = Background::from_domain(&crate::automesher::DomainSpec {
            extent: [0.0, 4.0, 0.0, 3.0, 0.0, 2.0],
            base_size: 1.0,
            grading: [1.0; 3],
        })
        .expect("background");
        assert_eq!(bg.base_n(), [4, 3, 2]);
        let tree = Octree::uniform(bg.base_n(), 2).expect("tree");
        let raw = emit(&tree, &bg, &patch_names()).expect("emit");
        let want = crate::blockgen::raw_mesh(&bg.block_spec()).expect("blockgen");

        assert_eq!(raw.points.len(), want.points.len(), "point count");
        for (n, (p, q)) in raw.points.iter().zip(want.points.iter()).enumerate() {
            assert!(
                p.x.to_bits() == q.x.to_bits()
                    && p.y.to_bits() == q.y.to_bits()
                    && p.z.to_bits() == q.z.to_bits(),
                "point {} differs: {:?} vs blockgen's {:?}",
                n,
                p,
                q
            );
        }
        assert_eq!(raw.owner, want.owner, "owner");
        assert_eq!(raw.neighbour, want.neighbour, "neighbour");
        for (f, (p, q)) in raw.faces.iter().zip(want.faces.iter()).enumerate() {
            assert_eq!(p, q, "face {}'s point list", f);
        }
        assert_eq!(raw.patches.len(), want.patches.len(), "patch count");
        for (slot, (a, b)) in raw.patches.iter().zip(want.patches.iter()).enumerate() {
            assert_eq!(a.name, b.name, "patch {} name", slot);
            assert_eq!(a.start, b.start, "patch {} start", slot);
            assert_eq!(a.size, b.size, "patch {} size", slot);
        }
    }

    /// (92.20)'s `t == 0` branch under grading: every point of a level-0
    /// tree is a graded node read from the array, so the arrays stay bit
    /// identical when the x axis is graded.
    #[test]
    fn a_graded_level_zero_tree_still_matches_blockgen() {
        let bg = Background::from_domain(&crate::automesher::DomainSpec {
            extent: [0.0, 4.0, 0.0, 3.0, 0.0, 2.0],
            base_size: 1.0,
            grading: [2.0, 1.0, 1.0],
        })
        .expect("background");
        let tree = Octree::uniform(bg.base_n(), 2).expect("tree");
        let raw = emit(&tree, &bg, &patch_names()).expect("emit");
        let want = crate::blockgen::raw_mesh(&bg.block_spec()).expect("blockgen");

        assert_eq!(raw.points.len(), want.points.len(), "point count");
        for (n, (p, q)) in raw.points.iter().zip(want.points.iter()).enumerate() {
            assert!(
                p.x.to_bits() == q.x.to_bits()
                    && p.y.to_bits() == q.y.to_bits()
                    && p.z.to_bits() == q.z.to_bits(),
                "point {} differs: {:?} vs blockgen's {:?}",
                n,
                p,
                q
            );
        }
    }

    /// A balanced refined tree emits a mesh that closes, carries the box's
    /// own volume, is one region, and passes §92.3's gate.
    #[test]
    fn a_refined_tree_emits_a_mesh_that_closes() {
        let bg = Background::from_domain(&crate::automesher::DomainSpec {
            extent: [0.0, 6.0, 0.0, 6.0, 0.0, 6.0],
            base_size: 1.0,
            grading: [1.0; 3],
        })
        .expect("background");
        let mut tree = Octree::uniform(bg.base_n(), 2).expect("tree");
        // The eight central base cells of a [6,6,6] grid: the {2,3}^3 block.
        tree.refine(|k| {
            if k.ancestor(0).idx.iter().all(|&c| c == 2 || c == 3) {
                2
            } else {
                0
            }
        });
        tree.balance_2to1();
        tree.check_partition().expect("partition");
        assert_eq!(tree.max_level_jump(), 1);
        let raw = emit(&tree, &bg, &patch_names()).expect("emit");

        let mut host = crate::io::polymesh::build_host_mesh(&raw).expect("host mesh");
        host.compute_geometry(&raw.points, &raw.faces).expect("geometry");
        let rep = host.check();
        assert!(
            rep.max_closure_error < 1e-12,
            "closure error {:?}",
            rep.max_closure_error
        );
        assert!(rep.min_volume > 0.0, "min volume {:?}", rep.min_volume);
        assert!(rep.ldu_ordered, "faces not in upper-triangular order");
        assert_eq!(rep.n_regions, 1, "{} regions", rep.n_regions);

        let box_volume = 6.0f64 * 6.0 * 6.0;
        let rel = ((rep.total_volume - box_volume) / box_volume).abs();
        assert!(
            rel < 1e-12,
            "total volume {:?} vs the box's {:?}",
            rep.total_volume,
            box_volume
        );

        let rep =
            crate::automesher::quality::check(&raw, &crate::automesher::QualitySpec::default().thresholds())
            .expect("the quality gate refused the mesh");
        assert!(rep.passed(), "the quality gate did not pass: {}", rep.summary());
    }

    /// §92.8's unit-2 row: the same half-refined block through
    /// `mesh::refined::refined_half` and through the octree gives the same
    /// cell count, face counts and total volume. The two number their cells
    /// differently, so the owner/neighbour arrays are NOT compared.
    #[test]
    fn the_octree_reproduces_the_static_generator() {
        let n = [4usize; 3];
        let half = crate::mesh::refined::refined_half(n, crate::Vec3::new(0.25, 0.25, 0.25), 1)
            .expect("refined_half");

        let bg = Background::from_domain(&crate::automesher::DomainSpec {
            extent: [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            base_size: 0.25,
            grading: [1.0; 3],
        })
        .expect("background");
        assert_eq!(bg.base_n(), n);
        let mut tree = Octree::uniform(n, 1).expect("tree");
        // The same half: base cells i >= 2, one level finer.
        tree.refine(|k| if k.ancestor(0).idx[0] >= 2 { 1 } else { 0 });
        tree.check_partition().expect("partition");
        let raw = emit(&tree, &bg, &patch_names()).expect("emit");

        let cells = raw.owner.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        assert_eq!(cells as usize, half.mesh.n_cells, "cell count");
        assert_eq!(raw.neighbour.len(), half.mesh.n_internal_faces, "internal faces");
        assert_eq!(
            raw.faces.len() - raw.neighbour.len(),
            half.mesh.n_boundary_faces,
            "boundary faces"
        );

        let mut host = crate::io::polymesh::build_host_mesh(&raw).expect("host mesh");
        host.compute_geometry(&raw.points, &raw.faces).expect("geometry");
        let v = host.check().total_volume;
        let w = half.mesh.check().total_volume;
        let rel = ((v - w) / w).abs();
        assert!(rel < 1e-12, "total volume {:?} vs {:?}", v, w);
    }

    /// §92.9's last table row, run by hand (`cargo test --release --lib
    /// automesher -- --ignored`): 10^6 leaves on one core, no thread pool.
    #[test]
    #[ignore]
    fn a_million_leaves_emit_in_under_ten_seconds() {
        let bg = Background::from_domain(&crate::automesher::DomainSpec {
            extent: [0.0, 100.0, 0.0, 100.0, 0.0, 100.0],
            base_size: 1.0,
            grading: [1.0; 3],
        })
        .expect("background");
        let tree = Octree::uniform(bg.base_n(), 1).expect("tree");
        assert_eq!(tree.len(), 1_000_000);
        let t0 = std::time::Instant::now();
        let raw = emit(&tree, &bg, &patch_names()).expect("emit");
        let dt = t0.elapsed();
        let cells = raw.owner.iter().copied().max().map(|m| m + 1).unwrap_or(0);
        assert_eq!(cells, 1_000_000, "cell count");
        assert!(dt.as_secs_f64() < 10.0, "emission took {:?}", dt);
    }

    // ======================================================================
    //  The surface-driven criterion - (92.1)/(92.22)
    // ======================================================================

    use crate::automesher::{DistanceBand, RefinementBand, RefinementSpec};
    use crate::surface::SoupTri;

    /// The 12 outward-wound triangles of an axis-aligned box, all on patch 0.
    fn box_soup(lo: [f64; 3], hi: [f64; 3]) -> Vec<SoupTri> {
        let v = crate::Vec3::new;
        let p = [
            v(lo[0], lo[1], lo[2]),
            v(hi[0], lo[1], lo[2]),
            v(hi[0], hi[1], lo[2]),
            v(lo[0], hi[1], lo[2]),
            v(lo[0], lo[1], hi[2]),
            v(hi[0], lo[1], hi[2]),
            v(hi[0], hi[1], hi[2]),
            v(lo[0], hi[1], hi[2]),
        ];
        let f: [[usize; 3]; 12] = [
            [0, 3, 2],
            [0, 2, 1], // -z
            [4, 5, 6],
            [4, 6, 7], // +z
            [0, 4, 7],
            [0, 7, 3], // -x
            [1, 2, 6],
            [1, 6, 5], // +x
            [0, 1, 5],
            [0, 5, 4], // -y
            [3, 7, 6],
            [3, 6, 2], // +y
        ];
        f.iter().map(|t| (0u32, [p[t[0]], p[t[1]], p[t[2]]])).collect()
    }

    /// A sphere as a twice-subdivided octahedron (8 -> 32 -> 128 triangles),
    /// vertices on the sphere, outward wound.
    fn sphere_soup(c: [f64; 3], r: f64) -> Vec<SoupTri> {
        let dir = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let faces: [[usize; 3]; 8] = [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ];
        let at = |d: [f64; 3]| [c[0] + r * d[0], c[1] + r * d[1], c[2] + r * d[2]];
        let mut tris: Vec<[[f64; 3]; 3]> =
            faces.iter().map(|f| [at(dir[f[0]]), at(dir[f[1]]), at(dir[f[2]])]).collect();
        for _ in 0..2 {
            let mid = |p: [f64; 3], q: [f64; 3]| {
                [0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]), 0.5 * (p[2] + q[2])]
            };
            // Re-project onto the sphere: the midpoint of two absolute
            // vertices, pulled back to radius r ABOUT THE CENTRE.
            let norm = |p: [f64; 3]| {
                let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
                let m = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                [c[0] + r * d[0] / m, c[1] + r * d[1] / m, c[2] + r * d[2] / m]
            };
            let mut next = Vec::with_capacity(tris.len() * 4);
            for [a, b, cc] in tris.drain(..) {
                let (ab, bc, ca) = (norm(mid(a, b)), norm(mid(b, cc)), norm(mid(cc, a)));
                next.push([a, ab, ca]);
                next.push([ab, b, bc]);
                next.push([ca, bc, cc]);
                next.push([ab, bc, ca]);
            }
            tris = next;
        }
        tris.into_iter().map(|t| (0u32, t.map(|p| crate::Vec3::new(p[0], p[1], p[2])))).collect()
    }

    /// One band over one patch, the cap matching the trees the tests build.
    fn band_spec(patch: &str, distance: f64, level: u32) -> RefinementSpec {
        RefinementSpec {
            levels: vec![RefinementBand {
                patch: patch.to_string(),
                bands: vec![DistanceBand { distance, level }],
                feature_level: 0,
            }],
            feature_angle_deg: 30.0,
            max_level: 1,
        }
    }

    /// The [0,8]^3 block, 1 m cells, and one one-patch surface over `soup`.
    fn block_and_surface(soup: Vec<SoupTri>, name: &str) -> (Background, Surface) {
        let bg = Background::from_domain(&crate::automesher::DomainSpec {
            extent: [0.0, 8.0, 0.0, 8.0, 0.0, 8.0],
            base_size: 1.0,
            grading: [1.0; 3],
        })
        .expect("background");
        assert_eq!(bg.base_n(), [8, 8, 8]);
        let surf = Surface::from_soup(soup, vec![name.to_string()]).expect("surface");
        (bg, surf)
    }

    /// §92.3's gate over an emitted tree, plus the partition the gate reads:
    /// one region, closed to 1e-12, upper-triangular, the block's own volume.
    fn assert_gate(tree: &Octree, bg: &Background, volume: f64) {
        let raw = emit(tree, bg, &patch_names()).expect("emit");
        let mut host = crate::io::polymesh::build_host_mesh(&raw).expect("host mesh");
        host.compute_geometry(&raw.points, &raw.faces).expect("geometry");
        let rep = host.check();
        assert_eq!(rep.n_regions, 1, "{} regions", rep.n_regions);
        assert!(rep.max_closure_error < 1e-12, "closure error {:?}", rep.max_closure_error);
        assert!(rep.ldu_ordered, "faces not in upper-triangular order");
        let rel = ((rep.total_volume - volume) / volume).abs();
        assert!(rel < 1e-12, "total volume {:?} vs the block's {:?}", rep.total_volume, volume);
        let gate = crate::automesher::quality::check(
            &raw,
            &crate::automesher::QualitySpec::default().thresholds(),
        )
        .expect("the quality gate refused the mesh");
        assert!(gate.passed(), "the quality gate did not pass: {}", gate.summary());
    }

    /// Test 1's hand count: which base cells' centres lie within `0.9` of the
    /// box [3.5,4.5]^3's surface, by the exact unsigned box distance.
    fn hand_count_within(distance: f64) -> Vec<[usize; 3]> {
        let (lo, hi) = ([3.5f64; 3], [4.5f64; 3]);
        let box_distance = |p: [f64; 3]| -> f64 {
            let inside = (0..3).all(|a| lo[a] <= p[a] && p[a] <= hi[a]);
            if inside {
                (0..3).map(|a| (p[a] - lo[a]).min(hi[a] - p[a])).fold(f64::INFINITY, f64::min)
            } else {
                (0..3)
                    .map(|a| match p[a] {
                        v if v < lo[a] => lo[a] - v,
                        v if v > hi[a] => v - hi[a],
                        _ => 0.0,
                    })
                    .map(|d| d * d)
                    .sum::<f64>()
                    .sqrt()
            }
        };
        let mut hits = Vec::new();
        for i in 0..8usize {
            for j in 0..8usize {
                for k in 0..8usize {
                    let p = [i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5];
                    if box_distance(p) <= distance {
                        hits.push([i, j, k]);
                    }
                }
            }
        }
        hits
    }

    #[test]
    fn a_small_cube_refines_exactly_the_cells_the_band_reaches() {
        let (bg, surf) = block_and_surface(box_soup([3.5; 3], [4.5; 3]), "box");
        let spec = band_spec("box", 0.9, 1);
        let mut tree = Octree::uniform(bg.base_n(), 1).expect("tree");

        // The hand count: the eight base cells whose centres are the box's
        // own corners sit ON the surface (d = 0); the next ring is a full
        // cell away (d = 1.0 > 0.9). 512 base cells, each hit cell split
        // into its eight children.
        let hits = hand_count_within(0.9);
        assert_eq!(hits.len(), 8, "the hand count: {:?}", hits);
        for q in &hits {
            assert!(q.iter().all(|&c| c == 3 || c == 4), "hand count picked {:?}", q);
        }
        let want_len = 512 - hits.len() + 8 * hits.len();

        let (splits, bal) = refine_to_surface(&mut tree, &bg, &surf, &spec).expect("refine");
        assert_eq!(tree.len(), want_len, "{} cells split", hits.len());
        assert_eq!((splits, bal), (hits.len(), 0), "one split per hit cell, none for balance");

        // 2 lattice units per base cell at max_level 1, 1 m base cells.
        let m_per_q = 1.0 / (1u32 << 1) as f64;
        for leaf in tree.leaves() {
            if leaf.level == 1 {
                let q = leaf.lower_corner(1);
                let lo = [q[0] as f64 * m_per_q, q[1] as f64 * m_per_q, q[2] as f64 * m_per_q];
                assert!(
                    lo.iter().all(|&v| (3.0..5.0).contains(&v)),
                    "level-1 leaf {:?} lower corner {:?} m outside [3,5)^3",
                    leaf, lo
                );
            } else {
                assert_eq!(leaf.level, 0, "leaf {:?} at an unexpected level", leaf);
            }
        }
        assert_eq!(tree.max_level_jump(), 1);
        tree.check_partition().expect("partition");
    }

    #[test]
    fn the_emitted_mesh_of_a_refined_tree_passes_the_gate() {
        let (bg, surf) = block_and_surface(box_soup([3.5; 3], [4.5; 3]), "box");
        let spec = band_spec("box", 0.9, 1);
        let mut tree = Octree::uniform(bg.base_n(), 1).expect("tree");
        let (splits, bal) = refine_to_surface(&mut tree, &bg, &surf, &spec).expect("refine");
        assert_eq!((splits, bal, tree.len()), (8, 0, 512 - 8 + 64));
        assert_gate(&tree, &bg, 8.0 * 8.0 * 8.0);
    }

    #[test]
    fn a_sphere_refines_only_around_itself() {
        let (bg, surf) = block_and_surface(sphere_soup([4.0; 3], 0.5), "sphere");
        let spec = band_spec("sphere", 0.9, 1);
        let mut tree = Octree::uniform(bg.base_n(), 1).expect("tree");
        refine_to_surface(&mut tree, &bg, &surf, &spec).expect("refine");

        let l = tree.max_level();
        let centre = crate::Vec3::new(4.0, 4.0, 4.0);
        let mut n1 = 0usize;
        for leaf in tree.leaves() {
            if leaf.level == 1 {
                n1 += 1;
                let (c, _) = leaf_centre_half_diag(&bg, l, leaf);
                let r = (c - centre).mag();
                assert!(r < 2.0, "level-1 leaf {:?} centre is {} from the sphere", leaf, r);
            }
        }
        assert!(n1 >= 64, "{} leaves at level 1", n1);
        assert!(tree.contains(LeafKey { level: 0, idx: [0, 0, 0] }), "corner [0,0,0] refined");
        assert!(tree.contains(LeafKey { level: 0, idx: [7, 7, 7] }), "corner [7,7,7] refined");
        assert_gate(&tree, &bg, 512.0);
    }

    #[test]
    fn a_band_naming_a_patch_the_surface_lacks_is_refused() {
        let (_, surf) = block_and_surface(box_soup([3.5; 3], [4.5; 3]), "box");
        let spec = band_spec("nosuchpatch", 0.9, 1);
        let err = band_surfaces(&surf, &spec).expect_err("must be refused");
        let msg = match err {
            Error::Mesh(m) => m,
            other => panic!("wrong error kind: {:?}", other),
        };
        assert!(msg.contains("nosuchpatch"), "the bad patch is not named: {}", msg);
        assert!(msg.contains("box"), "no real patch name is listed: {}", msg);
    }

    /// (92.22): a band narrower than the cell it is measured on cannot be
    /// allowed to let the surface slip through unrefined. The eight cells
    /// whose centres are the box's own corners carry the box on their
    /// boundary (d = 0 <= 0.01) and split; the ring beyond is 1.0 away,
    /// farther than both the band and the base cell's half-diagonal
    /// sqrt(3)/2, so neither (92.1) nor (92.22) asks for it.
    #[test]
    fn a_band_narrower_than_the_cell_still_refines_the_cells_the_surface_crosses() {
        let (bg, surf) = block_and_surface(box_soup([3.5; 3], [4.5; 3]), "box");
        let spec = band_spec("box", 0.01, 1);
        let mut tree = Octree::uniform(bg.base_n(), 1).expect("tree");
        let (splits, bal) = refine_to_surface(&mut tree, &bg, &surf, &spec).expect("refine");
        assert_eq!((splits, bal), (8, 0));
        assert_eq!(tree.len(), 512 - 8 + 64);

        let m_per_q = 1.0 / (1u32 << 1) as f64;
        for leaf in tree.leaves() {
            if leaf.level == 1 {
                let q = leaf.lower_corner(1);
                let lo = [q[0] as f64 * m_per_q, q[1] as f64 * m_per_q, q[2] as f64 * m_per_q];
                assert!(
                    lo.iter().all(|&v| (3.0..5.0).contains(&v)),
                    "level-1 leaf {:?} lower corner {:?} m outside [3,5)^3",
                    leaf, lo
                );
            } else {
                assert_eq!(leaf.level, 0, "leaf {:?} at an unexpected level", leaf);
            }
        }
        // The ring beyond stays unsplit.
        assert!(tree.contains(LeafKey { level: 0, idx: [2, 3, 3] }), "ring cell [2,3,3] split");
        assert!(tree.contains(LeafKey { level: 0, idx: [5, 4, 4] }), "ring cell [5,4,4] split");
        assert_eq!(tree.max_level_jump(), 1);
        tree.check_partition().expect("partition");
    }

    /// (92.22) ON ITS OWN, which the test above cannot isolate: there the
    /// box's corners sit exactly on the cell centres, so `d = 0` and (92.1)
    /// already refuses nothing. Here the box is [3.75, 4.25]^3, so the eight
    /// cells it sits in have their centres `sqrt(3)/4 = 0.433` m from it -
    /// far outside a `0.01` band, and well inside the `sqrt(3)/2 = 0.866`
    /// half-diagonal. (92.1) alone would refine NOTHING and the surface would
    /// pass through eight unrefined cells; (92.22) is what refines them, and
    /// `level_at` is asked both ways to say so in one assertion.
    ///
    /// Written by the supervising session, not by the coding agent.
    #[test]
    fn a_band_narrower_than_the_cell_refines_only_through_92_22() {
        let (bg, surf) = block_and_surface(box_soup([3.75; 3], [4.25; 3]), "box");
        let spec = band_spec("box", 0.01, 1);

        // The two terms, separated: `half_diag = 0` switches (92.22) off.
        // `h = 1.0` is the level-0 cell's own longest edge; this spec's
        // `feature_level` is 0, so (92.37) asks nothing whatever `h` is.
        let parts = band_surfaces(&surf, &spec).expect("band surfaces");
        let fs = extract(&surf, spec.feature_angle_deg).expect("features");
        let idx = BandIndex::new(&parts, &surf, &fs, &spec, 1.0).expect("band index");
        let centre = crate::Vec3::new(3.5, 3.5, 3.5);
        assert_eq!(idx.level_at(centre, 0.0, 1.0), 0, "(92.1) alone must not refine this cell");
        assert_eq!(
            idx.level_at(centre, 3.0f64.sqrt() / 2.0, 1.0),
            1,
            "(92.22) must refine a cell the surface passes through"
        );
        // And the ring is outside both: 1.299 m away, past the half-diagonal.
        let ring = crate::Vec3::new(2.5, 3.5, 3.5);
        assert_eq!(idx.level_at(ring, 3.0f64.sqrt() / 2.0, 1.0), 0, "the ring is not touched");

        // The whole stage then refines exactly the eight cells the box sits in.
        let mut tree = Octree::uniform(bg.base_n(), 1).expect("tree");
        let (splits, bal) = refine_to_surface(&mut tree, &bg, &surf, &spec).expect("refine");
        assert_eq!((splits, bal, tree.len()), (8, 0, 512 - 8 + 64));
        for leaf in tree.leaves() {
            let inside = leaf.ancestor(0).idx.iter().all(|&c| c == 3 || c == 4);
            assert_eq!(leaf.level, u32::from(inside), "leaf {:?} at the wrong level", leaf);
        }
        assert_gate(&tree, &bg, 8.0 * 8.0 * 8.0);
    }

    /// §92.12 (92.37): a leaf within its own longest edge of one of a
    /// patch's feature edges carries that patch's `feature_level`. The cube
    /// sits OFF the lattice on purpose, so no centre lands on a band
    /// boundary and no tie is being hidden.
    #[test]
    fn a_cube_s_edges_pull_the_octree_to_the_feature_level() {
        let bg = Background::from_domain(&crate::automesher::DomainSpec {
            extent: [0.0, 4.0, 0.0, 4.0, 0.0, 4.0],
            base_size: 1.0,
            grading: [1.0; 3],
        })
        .expect("background");
        assert_eq!(bg.base_n(), [4, 4, 4]);
        let surf = Surface::from_soup(
            box_soup([1.3, 1.3, 1.3], [2.3, 2.3, 2.3]),
            vec!["cube".to_string()],
        )
        .expect("surface");
        let spec = RefinementSpec {
            levels: vec![RefinementBand {
                patch: "cube".to_string(),
                bands: vec![],
                feature_level: 2,
            }],
            feature_angle_deg: 30.0,
            max_level: 2,
        };
        let mut tree = Octree::uniform(bg.base_n(), 2).expect("tree");
        let (splits, _bal) = refine_to_surface(&mut tree, &bg, &surf, &spec).expect("refine");
        assert!(splits > 0, "the cube's edges must refine something");
        assert_eq!(tree.max_level_jump(), 1);
        tree.check_partition().expect("partition");
        // The cube's 12 edges, as segments, and the distance to the nearest
        // one - point-to-segment arithmetic written here, sharing nothing
        // with BandIndex or the feature index.
        let (lo, hi) = ([1.3f64; 3], [2.3f64; 3]);
        let mut edges: Vec<([f64; 3], [f64; 3])> = Vec::new();
        for &x in &[lo[0], hi[0]] {
            for &y in &[lo[1], hi[1]] {
                edges.push(([x, y, lo[2]], [x, y, hi[2]]));
            }
        }
        for &y in &[lo[1], hi[1]] {
            for &z in &[lo[2], hi[2]] {
                edges.push(([lo[0], y, z], [hi[0], y, z]));
            }
        }
        for &z in &[lo[2], hi[2]] {
            for &x in &[lo[0], hi[0]] {
                edges.push(([x, lo[1], z], [x, hi[1], z]));
            }
        }
        assert_eq!(edges.len(), 12, "a cube carries 12 edges");
        let edge_dist = |p: [f64; 3]| -> f64 {
            let mut best = f64::INFINITY;
            for (a, b) in &edges {
                let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
                let t = (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2])
                    / (ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]);
                let t = t.clamp(0.0, 1.0);
                let q = [a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]];
                let d = ((p[0] - q[0]).powi(2)
                    + (p[1] - q[1]).powi(2)
                    + (p[2] - q[2]).powi(2))
                .sqrt();
                best = best.min(d);
            }
            best
        };
        // (92.37)'s promise, leaf by leaf: within its own longest edge of a
        // cube edge means at the feature level.
        let l = tree.max_level();
        for leaf in tree.leaves() {
            let (c, e) = leaf_centre_edges(&bg, l, leaf);
            let h = e[0].max(e[1]).max(e[2]);
            let d = edge_dist([c.x, c.y, c.z]);
            if d <= h {
                assert_eq!(
                    leaf.level, 2,
                    "leaf {:?} centre is {d} from an edge, inside its own h = {h}",
                    leaf
                );
            }
        }
        // Far from every cube edge, level 0: the [3,4]^3 corner, a full
        // metre past anything the band or the balance pass could pull.
        assert!(
            tree.contains(LeafKey { level: 0, idx: [3, 3, 3] }),
            "corner [3,3,3] split - the band reached a metre past the cube"
        );
        // The count the geometry implies, on the plain lattice: a level-2
        // cell exists exactly where its level-0 centre sits within 1.0 m of
        // an edge AND its level-1 centre within 0.5 m - refinement splits
        // down and never up, and balance (92.21) only ever splits BELOW the
        // level that triggered it, so it adds no level-2 leaf. A band
        // around 12 edges, not merely "some".
        let mut want2 = 0usize;
        for i in 0..16usize {
            for j in 0..16usize {
                for k in 0..16usize {
                    let c0 = [
                        ((i / 4) as f64 + 0.5) * 1.0,
                        ((j / 4) as f64 + 0.5) * 1.0,
                        ((k / 4) as f64 + 0.5) * 1.0,
                    ];
                    let c1 = [
                        ((i / 2) as f64 + 0.5) * 0.5,
                        ((j / 2) as f64 + 0.5) * 0.5,
                        ((k / 2) as f64 + 0.5) * 0.5,
                    ];
                    if edge_dist(c0) <= 1.0 && edge_dist(c1) <= 0.5 {
                        want2 += 1;
                    }
                }
            }
        }
        assert!(want2 >= 48, "the oracle saw only {} level-2 cells - 12 edges x 4 long", want2);
        let got2 = tree.leaves().iter().filter(|lf| lf.level == 2).count();
        assert_eq!(got2, want2, "level-2 leaves, the lattice oracle says {}", want2);
    }

    /// (92.37) at its default `feature_level: 0` is invisible: the same
    /// config with the feature entry present but zero yields exactly the
    /// leaves the empty `levels` list yields - the same run the code made
    /// before §92.12's refinement, element by element.
    #[test]
    fn feature_level_zero_changes_nothing() {
        let bg = Background::from_domain(&crate::automesher::DomainSpec {
            extent: [0.0, 4.0, 0.0, 4.0, 0.0, 4.0],
            base_size: 1.0,
            grading: [1.0; 3],
        })
        .expect("background");
        let surf = Surface::from_soup(
            box_soup([1.3, 1.3, 1.3], [2.3, 2.3, 2.3]),
            vec!["cube".to_string()],
        )
        .expect("surface");
        let base = RefinementSpec {
            levels: vec![RefinementBand {
                patch: "cube".to_string(),
                bands: vec![],
                feature_level: 0,
            }],
            feature_angle_deg: 30.0,
            max_level: 2,
        };
        let none = RefinementSpec { levels: vec![], feature_angle_deg: 30.0, max_level: 2 };
        let mut with_zero = Octree::uniform(bg.base_n(), 2).expect("tree");
        refine_to_surface(&mut with_zero, &bg, &surf, &base).expect("refine");
        let mut without = Octree::uniform(bg.base_n(), 2).expect("tree");
        refine_to_surface(&mut without, &bg, &surf, &none).expect("refine");
        let zero_leaves: Vec<_> =
            with_zero.leaves().into_iter().map(|lf| (lf.level, lf.idx)).collect();
        let none_leaves: Vec<_> =
            without.leaves().into_iter().map(|lf| (lf.level, lf.idx)).collect();
        assert_eq!(zero_leaves.len(), none_leaves.len(), "the same number of leaves");
        for (z, n) in zero_leaves.iter().zip(none_leaves.iter()) {
            assert_eq!(z, n, "feature_level: 0 changed the leaf set");
        }
    }

    /// A smooth sphere asks for nothing: its largest fold is 22.08 degrees,
    /// under the 30-degree default, so the feature set is empty and
    /// `feature_level: 2` refines not one cell.
    #[test]
    fn a_smooth_sphere_asks_for_no_feature_refinement() {
        let bg = Background::from_domain(&crate::automesher::DomainSpec {
            extent: [0.0, 4.0, 0.0, 4.0, 0.0, 4.0],
            base_size: 1.0,
            grading: [1.0; 3],
        })
        .expect("background");
        let surf = Surface::from_soup(sphere_soup([2.0; 3], 1.0), vec!["sphere".to_string()])
            .expect("surface");
        let fs = crate::automesher::features::extract(&surf, 30.0).expect("features");
        assert!(fs.is_empty(), "a smooth sphere has no feature edge: {}", fs.summary());
        let spec = RefinementSpec {
            levels: vec![RefinementBand {
                patch: "sphere".to_string(),
                bands: vec![],
                feature_level: 2,
            }],
            feature_angle_deg: 30.0,
            max_level: 2,
        };
        let mut tree = Octree::uniform(bg.base_n(), 2).expect("tree");
        let (splits, bal) = refine_to_surface(&mut tree, &bg, &surf, &spec).expect("refine");
        assert_eq!((splits, bal, tree.len()), (0, 0, 64), "not one cell refined");
    }
}

// ==========================================================================
//  The level a leaf is asked for - SPEC-LIT §92.2 stage 1, (92.1)/(92.22)
// ==========================================================================
//
// (92.1)'s `l_dist` is evaluated per patch that `refinement.levels` NAMES:
// one [`TriIndex`] (§23.4) over that patch's triangles alone, so the maximum
// is taken over the patches the config asked about and not over the
// surface's every solid. A patch the config does not name is not indexed and
// refines nothing. Provenance: ORIGINAL. No GPL-licensed source was
// consulted.

use super::features::{FeatureIndex, FeatureSet, extract};
use super::{DistanceBand, RefinementSpec};
use crate::surface::{SoupTri, Surface, TriIndex};

/// One sub-`Surface` per `refinement.levels[]` entry, in that order: the
/// triangles of the patch it names and nothing else. Refuses, naming the
/// patch and listing the surface's own patch names, a `patch` the surface
/// does not have - a silently ignored band is a mesh the user did not ask
/// for. A named patch the surface has but holds no triangle of is refused
/// the same way, for the same reason.
pub fn band_surfaces(surf: &Surface, spec: &RefinementSpec) -> Result<Vec<Surface>> {
    let mut parts = Vec::with_capacity(spec.levels.len());
    for band in &spec.levels {
        let Some(id) = surf.patch_names.iter().position(|n| n == &band.patch) else {
            return Err(Error::Mesh(format!(
                "band_surfaces: a refinement band names patch {:?}, but the surface's \
                 patches are {:?} - a patch the surface lacks refines nothing",
                band.patch, surf.patch_names
            )));
        };
        let id = id as u32;
        let soup: Vec<SoupTri> = surf
            .tris
            .iter()
            .enumerate()
            .filter(|(t, _)| surf.tri_patch[*t] == id)
            .map(|(_, tri)| {
                (
                    0u32,
                    [
                        surf.points[tri[0] as usize],
                        surf.points[tri[1] as usize],
                        surf.points[tri[2] as usize],
                    ],
                )
            })
            .collect();
        if soup.is_empty() {
            return Err(Error::Mesh(format!(
                "band_surfaces: patch {:?} holds no triangle of the surface, whose \
                 patches are {:?} - a band over an empty patch refines nothing",
                band.patch, surf.patch_names
            )));
        }
        parts.push(Surface::from_soup(soup, vec![band.patch.clone()])?);
    }
    Ok(parts)
}

/// The per-patch distance queries (92.1)/(92.22) need: one [`TriIndex`] over
/// each patch `refinement.levels` NAMES, over that patch's triangles alone.
/// A patch the config does not name is not indexed and refines nothing.
pub struct BandIndex<'s> {
    /// One entry per `refinement.levels[]` entry, in that order.
    parts: Vec<BandPart<'s>>,
}

/// One `refinement.levels[]` entry's queries: the patch's triangles for
/// (92.1)/(92.22), its bands, and - only when the entry asks for feature
/// refinement - the patch's own feature edges, bucketed out of the WHOLE
/// surface's feature set.
struct BandPart<'s> {
    tri: TriIndex<'s>,
    bands: Vec<DistanceBand>,
    /// The patch's feature edges, bucketed - `None` while `feat_level` is
    /// zero, the config's default.
    feat: Option<FeatureIndex<'s>>,
    /// This entry's `refinement.levels[].feature_level` - SPEC-LIT §92.12,
    /// eq. (92.37).
    feat_level: u32,
}

impl<'s> BandIndex<'s> {
    /// One [`BandPart`] per sub-surface of [`band_surfaces`], in that order.
    /// `parts` must be `band_surfaces(surf, spec)`'s output for the same
    /// `surf` and `spec`; refuse if the lengths disagree. `fs` is the WHOLE
    /// surface's feature set, extracted once by the caller; an entry whose
    /// `feature_level` is above zero indexes it down to its own patch
    /// (§92.12, eq. 92.37), an entry at the default zero indexes nothing.
    pub fn new(
        parts: &'s [Surface],
        surf: &Surface,
        fs: &'s FeatureSet,
        spec: &RefinementSpec,
        cell_hint: crate::Scalar,
    ) -> Result<BandIndex<'s>> {
        if parts.len() != spec.levels.len() {
            return Err(Error::Mesh(format!(
                "BandIndex::new: {} sub-surface(s) for {} refinement.levels entries - \
                 not the same list",
                parts.len(),
                spec.levels.len()
            )));
        }
        let mut indexed = Vec::with_capacity(parts.len());
        for (part, band) in parts.iter().zip(&spec.levels) {
            // The patch id the entry's name carries in the FULL surface -
            // `fs.edge_patches` are full-surface ids. A patch the surface
            // lacks is refused, the same refusal [`band_surfaces`] makes.
            let Some(patch_id) = surf.patch_names.iter().position(|n| n == &band.patch) else {
                return Err(Error::Mesh(format!(
                    "BandIndex::new: a refinement band names patch {:?}, but the surface's \
                     patches are {:?} - a patch the surface lacks refines nothing",
                    band.patch, surf.patch_names
                )));
            };
            let feat = if band.feature_level > 0 {
                Some(FeatureIndex::for_patch(fs, cell_hint, patch_id as u32)?)
            } else {
                None
            };
            indexed.push(BandPart {
                tri: TriIndex::new(part, cell_hint)?,
                bands: band.bands.clone(),
                feat_level: band.feature_level,
                feat,
            });
        }
        Ok(BandIndex { parts: indexed })
    }

    /// (92.1) + (92.22) + §92.12's (92.37) for one leaf: `centre` is the
    /// leaf's centre, `half_diag` half its diagonal, and `h` the leaf's
    /// longest edge - the cell size (92.37) measures with. Returns the level
    /// the bands and the feature edges ask for, UNCAPPED (`Octree::refine`
    /// applies `max_level`).
    pub fn level_at(
        &self,
        centre: crate::Vec3,
        half_diag: crate::Scalar,
        h: crate::Scalar,
    ) -> u32 {
        let mut asked = 0u32;
        for part in &self.parts {
            let (_, d) = part.tri.nearest_triangle(centre);
            // (92.1): the deepest band whose distance the cell centre falls
            // inside.
            let mut l = 0u32;
            for band in &part.bands {
                if d <= band.distance {
                    l = l.max(band.level);
                }
            }
            // (92.22): a leaf whose bounding sphere touches the patch is
            // refined to the deepest level that patch's bands ask for
            // anywhere - it errs toward refining, never away from it.
            if d <= half_diag {
                for band in &part.bands {
                    l = l.max(band.level);
                }
            }
            // §92.12 (92.37): a leaf within its own longest edge of one of
            // the patch's feature edges asks that patch's `feature_level`,
            // uncapped exactly like the band levels.
            if part.feat_level > 0 {
                if let Some((_, df, _)) =
                    part.feat.as_ref().and_then(|f| f.closest_edge_point(centre))
                {
                    if df <= h {
                        l = l.max(part.feat_level);
                    }
                }
            }
            asked = asked.max(l);
        }
        asked
    }
}

/// A leaf's centre and its three edge lengths, in metres - the same (92.20)
/// evaluation [`leaf_centre_half_diag`] reports, with the per-axis edges the
/// centre test's `h_min` needs. The one place this arithmetic is written.
pub(crate) fn leaf_centre_edges(
    bg: &Background,
    l: u32,
    k: LeafKey,
) -> (crate::Vec3, [crate::Scalar; 3]) {
    let lo = k.lower_corner(l);
    let s = k.size_on_finest(l);
    let x = [bg.coord(0, lo[0], l), bg.coord(0, lo[0] + s, l)];
    let y = [bg.coord(1, lo[1], l), bg.coord(1, lo[1] + s, l)];
    let z = [bg.coord(2, lo[2], l), bg.coord(2, lo[2] + s, l)];
    let c = crate::Vec3::new(0.5 * (x[0] + x[1]), 0.5 * (y[0] + y[1]), 0.5 * (z[0] + z[1]));
    (c, [x[1] - x[0], y[1] - y[0], z[1] - z[0]])
}

/// A leaf's centre and half its diagonal, in metres: (92.20) evaluated at the
/// two ends of the leaf's own span on the finest lattice - `l` is the TREE's
/// `max_level`, `bg.coord`'s third argument. These are the two numbers
/// (92.1) and (92.22) measure from; (92.37) reads its own `h` off
/// [`leaf_centre_edges`] instead, so this survives for the tests alone.
#[cfg(test)]
fn leaf_centre_half_diag(
    bg: &Background,
    l: u32,
    k: LeafKey,
) -> (crate::Vec3, crate::Scalar) {
    let (c, e) = leaf_centre_edges(bg, l, k);
    (c, 0.5 * crate::Vec3::new(e[0], e[1], e[2]).mag())
}

/// §92.2 stage 1: refine `tree` until every leaf carries the level
/// (92.1)/(92.22)/§92.12's (92.37) ask of it, then 2:1 balance it (92.21).
/// Returns `(refine_splits, balance_splits)`. The cap is the tree's own
/// `max_level` - [`Octree::refine`] applies it, and (92.21)'s splits sit
/// below the level that triggered them, so balance never passes it (§74.2).
pub fn refine_to_surface(
    tree: &mut Octree,
    bg: &Background,
    surf: &Surface,
    spec: &RefinementSpec,
) -> Result<(usize, usize)> {
    let parts = band_surfaces(surf, spec)?;
    // §92.12: the WHOLE surface's feature edges, extracted ONCE - the
    // per-entry indexes below filter it to their own patch (92.37). Not
    // extracted at all when no entry asks for feature refinement: the walk
    // is over every triangle of the surface, and a config that did not ask
    // for it must not pay for it.
    //
    // The `wants_features` guard is the supervising session's.
    let wants_features = spec.levels.iter().any(|b| b.feature_level > 0);
    let fs = if wants_features {
        extract(surf, spec.feature_angle_deg)?
    } else {
        FeatureSet::empty(spec.feature_angle_deg)
    };
    // §23.4's "~ the mesh spacing": the smallest base cell edge, the finest
    // spacing any query of this tree can be asked on.
    let mut cell_hint = crate::Scalar::INFINITY;
    for a in 0..3 {
        for w in bg.nodes[a].windows(2) {
            if w[1] - w[0] < cell_hint {
                cell_hint = w[1] - w[0];
            }
        }
    }
    if !(cell_hint > 0.0) {
        return Err(Error::Mesh(format!(
            "refine_to_surface: the background block's smallest cell edge is {cell_hint:?} - \
             stage 0 needs a positive cell size"
        )));
    }
    let idx = BandIndex::new(&parts, surf, &fs, spec, cell_hint)?;
    let l = tree.max_level();
    let splits = tree.refine(|k| {
        let (c, e) = leaf_centre_edges(bg, l, k);
        // (92.22) measures with the half-diagonal, (92.37) with the cell's
        // own longest edge - both straight off the same three edge lengths.
        let hd = 0.5 * crate::Vec3::new(e[0], e[1], e[2]).mag();
        let h = e[0].max(e[1]).max(e[2]);
        idx.level_at(c, hd, h)
    });
    let bal = tree.balance_2to1();
    tree.check_partition()?;
    Ok((splits, bal))
}
