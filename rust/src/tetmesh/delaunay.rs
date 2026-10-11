// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The Bowyer-Watson Delaunay kernel of SPEC-LIT §116.6.
//!
//! Incremental insertion (Bowyer 1981, Watson 1981): locate the tetrahedron
//! that contains the new point, grow the cavity of every tetrahedron whose
//! perturbed circumsphere contains it, rule (116.5a) for hull tets, then
//! star the cavity's boundary from the point. One infinite vertex `INF`
//! closes the hull, so every face has two tets; a hull tet stores INF in
//! slot 3, wound so the finite neighbour's apex has `orient3d < 0`, and a
//! new tet that gains INF is brought to slot 3 by an even permutation.
//! Point location is the visibility walk of Devillers, Pion & Teillaud
//! (2002), its face order rotated by a fixed-seed SplitMix64 stream owned
//! by the kernel, so results are deterministic. Every geometric decision is
//! the exact predicate of §116.4 or the perturbed `insphere` of §116.5.
//!
//! Provenance: ORIGINAL. Written from Bowyer (1981), Comput. J. 24(2):162-166, Watson (1981), Comput. J. 24(2):167-172, and Devillers, Pion & Teillaud (2002), Int. J. Found. Comput. Sci. 13(2):181-199. No GPL-licensed source was consulted.

use super::order::{brio_order, BRIO_SEED};
use super::perturb::insphere_perturbed;
use super::predicates::{in_exact_domain, orient3d};
use super::Point;
use crate::error::{Error, Result};
use std::collections::HashMap;

/// The infinite vertex id (SPEC-LIT §116.6).
pub const INF: u32 = u32::MAX;

/// "No tetrahedron".
pub const NONE: u32 = u32::MAX;

/// The kernel's insertion counters (SPEC-LIT §116.6). `inserted` counts
/// every inserted point including the first four; `walk_steps` is the total
/// faces crossed by the insertion walks; `cavity_tets` the total tets
/// removed; `coplanar_hull` the number of times rule (116.5a)'s `o = 0`
/// branch was evaluated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DelaunayStats {
    pub inserted: usize,
    pub walk_steps: u64,
    pub cavity_tets: u64,
    pub coplanar_hull: u64,
}

/// A point-location answer: the tetrahedron found and the faces the walk
/// crossed to reach it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Located {
    pub tet: u32,
    pub steps: u32,
}

/// The `check` report: counts of the FINITE simplices, plus the hull and
/// interior face counts (SPEC-LIT §116.6).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CheckReport {
    pub vertices: usize,
    pub edges: usize,
    pub faces: usize,
    pub tets: usize,
    pub hull_faces: usize,
    pub interior_faces: usize,
}

/// The Delaunay kernel's state (SPEC-LIT §116.6). Tets live in slot arrays
/// that grow and reuse dead slots through a LIFO free list. `rng` is the
/// walk's SplitMix64 state, seeded `0x5EED`, owned by the kernel so results
/// are deterministic; `last` is the last created tet, where the next
/// insertion's walk starts.
#[derive(Debug)]
pub struct Delaunay {
    points: Vec<Point>,
    verts: Vec<[u32; 4]>,
    neigh: Vec<[u32; 4]>,
    live: Vec<bool>,
    /// Per-slot epoch stamps for the cavity BFS (§116.7): a stamp equal
    /// to `epoch` means "in the current insertion's cavity", so no
    /// insertion allocates or clears a `slots()`-long array.
    cav_mark: Vec<u64>,
    /// Per-slot epoch stamps for "conflict-tested this insertion".
    chk_mark: Vec<u64>,
    free: Vec<u32>,
    rng: u64,
    last: u32,
    live_count: usize,
    /// The current insertion's epoch; `insert` advances it by one.
    epoch: u64,
    stats: DelaunayStats,
}

/// One step of SplitMix64 (Steele, Lea & Flood 2014), all operations
/// wrapping - the same stream the §116.5 tests use.
fn splitmix_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// The outward-wound face opposite slot `i`, from §116.4's table: opposite
/// slot 3 `(v0,v1,v2)`; slot 2 `(v0,v3,v1)`; slot 1 `(v0,v2,v3)`; slot 0
/// `(v1,v3,v2)`. A point strictly outside the tet through this face has
/// `orient3d(face, p) < 0`.
fn face_out(v: [u32; 4], i: usize) -> [u32; 3] {
    match i {
        0 => [v[1], v[3], v[2]],
        1 => [v[0], v[2], v[3]],
        2 => [v[0], v[3], v[1]],
        3 => [v[0], v[1], v[2]],
        _ => unreachable!(),
    }
}

/// The face's vertex set, sorted, for adjacency and star matching.
fn face_set(v: [u32; 4], i: usize) -> [u32; 3] {
    let mut f = face_out(v, i);
    f.sort_unstable();
    f
}

/// A face's vertex set, sorted, from an explicit winding.
fn sorted3(mut f: [u32; 3]) -> [u32; 3] {
    f.sort_unstable();
    f
}

/// Moves INF to slot 3 by an EVEN permutation of the creation order
/// (SPEC-LIT §116.6), which keeps the orientation: `(f0, f1, inf, p)`
/// becomes `(f1, f0, p, inf)`.
fn even_perm(c: [u32; 4]) -> [u32; 4] {
    let j = c
        .iter()
        .position(|&x| x == INF)
        .expect("even_perm without INF");
    match j {
        0 => [c[2], c[1], c[3], INF],
        1 => [c[0], c[2], c[3], INF],
        2 => [c[1], c[0], c[3], INF],
        _ => c,
    }
}

/// FNV-1a 64 over every id of every tet as 4 little-endian bytes, each
/// tet's ids sorted ascending and the list sorted lexicographically, so the
/// digest does not depend on the stored slot order.
pub fn tet_digest(tets: &[[u32; 4]]) -> u64 {
    let mut list: Vec<[u32; 4]> = tets
        .iter()
        .map(|t| {
            let mut s = *t;
            s.sort_unstable();
            s
        })
        .collect();
    list.sort_unstable();
    let mut h = 0xcbf29ce484222325_u64;
    for t in &list {
        for v in t {
            for b in v.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100000001b3);
            }
        }
    }
    h
}

impl Delaunay {
    /// Triangulates `points` in their given order (SPEC-LIT §116.6).
    pub fn build(points: &[Point]) -> Result<Delaunay> {
        let order: Vec<u32> = (0..points.len() as u32).collect();
        Self::build_ordered(points, &order)
    }

    /// Triangulates `points` in BRIO order (SPEC-LIT §116.7) - the
    /// mesher's entry point. The order is a function of the seed alone;
    /// the tetrahedron set does not depend on it (§116.6, Uniqueness),
    /// so BRIO changes the cost and nothing else.
    pub fn build_brio(points: &[Point]) -> Result<Delaunay> {
        let order = brio_order(points, BRIO_SEED);
        Self::build_ordered(points, &order)
    }

    /// Triangulates `points` inserting them in `order`, a permutation of
    /// `0..n`. The tetrahedron set is order-independent - the perturbed
    /// Delaunay triangulation is unique (§116.6, Uniqueness).
    pub fn build_ordered(points: &[Point], order: &[u32]) -> Result<Delaunay> {
        let n = points.len();
        if order.len() != n {
            return Err(Error::Refused(format!(
                "delaunay/order: length {} is not a permutation of 0..{}",
                order.len(),
                n
            )));
        }
        let mut seen = vec![false; n];
        for &i in order {
            let i = i as usize;
            if i >= n || seen[i] {
                return Err(Error::Refused(format!(
                    "delaunay/order: index {} is not a permutation of 0..{}",
                    i, n
                )));
            }
            seen[i] = true;
        }
        // Refusal order (SPEC-LIT §116.6): domain, then duplicate, then
        // flat. The domain test is (116.3f), NaN and infinities included.
        for (i, p) in points.iter().enumerate() {
            for (axis, x) in p.iter().enumerate() {
                if !in_exact_domain(*x) {
                    return Err(Error::Refused(format!(
                        "delaunay/domain: point {} axis {} = {}",
                        i,
                        ["x", "y", "z"][axis],
                        x
                    )));
                }
            }
        }
        // Duplicates: exactly equal coordinates, the lexicographically
        // smallest pair (i, j), i < j, over all coinciding pairs. The map
        // holds the first index per coordinate; -0.0 and 0.0 coincide.
        let canon = |x: f64| if x == 0.0 { 0.0f64.to_bits() } else { x.to_bits() };
        let mut first: HashMap<[u64; 3], usize> = HashMap::new();
        let mut dup: Option<(usize, usize)> = None;
        for (j, p) in points.iter().enumerate() {
            let key = [canon(p[0]), canon(p[1]), canon(p[2])];
            match first.get(&key) {
                Some(&i) => {
                    if dup.map_or(true, |b| (i, j) < b) {
                        dup = Some((i, j));
                    }
                }
                None => {
                    first.insert(key, j);
                }
            }
        }
        if let Some((i, j)) = dup {
            return Err(Error::Refused(format!(
                "delaunay/duplicate: points {} and {} coincide",
                i, j
            )));
        }
        if n < 4 {
            return Err(Error::Refused(
                "delaunay/flat: fewer than 4 points".to_string(),
            ));
        }
        // The first tetrahedron (SPEC-LIT §116.6, Start): scanning `order`,
        // the first point; the first that differs; the first not collinear
        // with those two; the first with `orient3d != 0` against the three.
        let pt = |i: u32| -> &Point { &points[i as usize] };
        let collinear = |a: &Point, b: &Point, c: &Point| -> bool {
            // Each coordinate-plane orientation: `orient3d` of the three
            // points with one coordinate set to 0 and a fourth point on
            // that axis. All three exactly 0 means collinear.
            for k in 0..3 {
                let mut pa = *a;
                let mut pb = *b;
                let mut pc = *c;
                pa[k] = 0.0;
                pb[k] = 0.0;
                pc[k] = 0.0;
                let d = [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ][k];
                if orient3d(&pa, &pb, &pc, &d) != 0.0 {
                    return false;
                }
            }
            true
        };
        let o0 = order[0];
        let mut q1 = None;
        for pos in 1..n {
            if pt(order[pos]) != pt(o0) {
                q1 = Some(pos);
                break;
            }
        }
        let q1 = match q1 {
            Some(q) => q,
            None => {
                return Err(Error::Refused(
                    "delaunay/flat: all points coincide".to_string(),
                ))
            }
        };
        let i1 = order[q1];
        let mut q2 = None;
        for pos in q1 + 1..n {
            if !collinear(pt(o0), pt(i1), pt(order[pos])) {
                q2 = Some(pos);
                break;
            }
        }
        let q2 = match q2 {
            Some(q) => q,
            None => {
                return Err(Error::Refused(
                    "delaunay/flat: all points collinear".to_string(),
                ))
            }
        };
        let i2 = order[q2];
        let mut q3 = None;
        for pos in q2 + 1..n {
            if orient3d(pt(o0), pt(i1), pt(i2), pt(order[pos])) != 0.0 {
                q3 = Some(pos);
                break;
            }
        }
        let q3 = match q3 {
            Some(q) => q,
            None => {
                return Err(Error::Refused(
                    "delaunay/flat: all points coplanar".to_string(),
                ))
            }
        };
        let i3 = order[q3];
        // Orient the first tet positively, (116.2).
        let mut t0 = [o0, i1, i2, i3];
        if orient3d(pt(t0[0]), pt(t0[1]), pt(t0[2]), pt(t0[3])) < 0.0 {
            t0.swap(2, 3);
        }
        let mut d = Delaunay {
            points: points.to_vec(),
            verts: Vec::with_capacity(2 * n + 8),
            neigh: Vec::with_capacity(2 * n + 8),
            live: Vec::with_capacity(2 * n + 8),
            cav_mark: Vec::with_capacity(2 * n + 8),
            chk_mark: Vec::with_capacity(2 * n + 8),
            free: Vec::new(),
            rng: 0x5EED,
            last: 0,
            live_count: 0,
            epoch: 0,
            stats: DelaunayStats::default(),
        };
        // Slot 0 is the first tet; slots 1..5 are its four hull tets, each
        // the REVERSE of §116.4's outward face with INF in slot 3, so the
        // finite neighbour's apex has `orient3d < 0` (rule 2 of §116.6).
        d.verts.push(t0);
        d.neigh.push([NONE; 4]);
        d.live.push(true);
        d.cav_mark.push(0);
        d.chk_mark.push(0);
        for i in 0..4 {
            let f = face_out(t0, i);
            d.verts.push([f[2], f[1], f[0], INF]);
            d.neigh.push([NONE; 4]);
            d.live.push(true);
            d.cav_mark.push(0);
            d.chk_mark.push(0);
        }
        d.live_count = 5;
        d.stats.inserted = 4;
        // Link the five by face sets: the first tet's face i borders hull
        // tet i's hull face, and every pair of hull tets shares one
        // INF-containing face.
        for t in 0..5u32 {
            for i in 0..4 {
                if d.neigh[t as usize][i] != NONE {
                    continue;
                }
                let fs = face_set(d.verts[t as usize], i);
                for u in 0..5u32 {
                    if u == t {
                        continue;
                    }
                    if let Some(j) =
                        (0..4).find(|&j| face_set(d.verts[u as usize], j) == fs)
                    {
                        d.neigh[t as usize][i] = u;
                        d.neigh[u as usize][j] = t;
                        break;
                    }
                }
            }
        }
        debug_assert!(d.neigh.iter().all(|n| n.iter().all(|&x| x != NONE)));
        // Insert the REMAINING points in `order` - every position but the
        // four the first tetrahedron consumed, which may skip over
        // collinear and coplanar points.
        for (pos, &pid) in order.iter().enumerate() {
            if pos == 0 || pos == q1 || pos == q2 || pos == q3 {
                continue;
            }
            d.insert(pid)?;
        }
        Ok(d)
    }

    /// The input points, in input order.
    pub fn points(&self) -> &[Point] {
        &self.points
    }

    /// The length of the tet arrays, live and dead.
    pub fn slots(&self) -> usize {
        self.verts.len()
    }

    /// Whether slot `t` holds a live tetrahedron.
    pub fn is_live(&self, t: u32) -> bool {
        self.live[t as usize]
    }

    /// The stored vertex order of slot `t`.
    pub fn tet(&self, t: u32) -> [u32; 4] {
        self.verts[t as usize]
    }

    /// `neighbours(t)[i]` is the tet across the face opposite slot `i`.
    pub fn neighbours(&self, t: u32) -> [u32; 4] {
        self.neigh[t as usize]
    }

    /// Whether `t` is a hull tet: INF in slot 3.
    pub fn is_hull(&self, t: u32) -> bool {
        self.verts[t as usize][3] == INF
    }

    /// The live finite tets, in slot order, stored vertex order.
    pub fn finite_tets(&self) -> Vec<[u32; 4]> {
        (0..self.slots() as u32)
            .filter(|&t| self.is_live(t) && !self.is_hull(t))
            .map(|t| self.verts[t as usize])
            .collect()
    }

    /// The hull face `(a, b, c)` of each live hull tet `(a, b, c, INF)`, in
    /// slot order.
    pub fn hull_faces(&self) -> Vec<[u32; 3]> {
        (0..self.slots() as u32)
            .filter(|&t| self.is_live(t) && self.is_hull(t))
            .map(|t| {
                let v = self.verts[t as usize];
                [v[0], v[1], v[2]]
            })
            .collect()
    }

    /// The visibility walk (SPEC-LIT §116.6) as a read-only query, from
    /// slot `start` - which may be any live slot. Each call uses its own
    /// SplitMix64 stream seeded `0x5EED`, so the answer does not depend on
    /// any other call.
    pub fn locate(&self, p: &Point, start: u32) -> Result<Located> {
        // The start rule of §116.6: a walk that starts on a hull
        // tetrahedron starts from its finite neighbour instead - the same
        // adjustment the insertion path applies to `last` before its own
        // walk. A walk that merely *enters* the hull still stops there, so
        // a query strictly beyond a hull face still answers that hull tet.
        let mut start = start;
        if self.is_live(start) && self.is_hull(start) {
            let n3 = self.neigh[start as usize][3];
            if n3 != NONE {
                start = n3;
            }
        }
        let mut rng = 0x5EED_u64;
        let (tet, steps) = self.walk_tet(p, start, &mut rng)?;
        Ok(Located { tet, steps })
    }

    /// The insertion counters.
    pub fn stats(&self) -> DelaunayStats {
        self.stats
    }

    /// The signed volume of the triangulated ball: the sum of (116.2) over
    /// the finite tets in slot order, divided by 6.
    pub fn finite_volume(&self) -> f64 {
        let mut s = 0.0;
        for t in 0..self.slots() as u32 {
            if !self.is_live(t) || self.is_hull(t) {
                continue;
            }
            let v = self.verts[t as usize];
            s += orient3d(
                &self.points[v[0] as usize],
                &self.points[v[1] as usize],
                &self.points[v[2] as usize],
                &self.points[v[3] as usize],
            );
        }
        s / 6.0
    }

    /// The visibility walk (SPEC-LIT §116.6). At each step it draws one
    /// SplitMix64 value and tests the four faces starting at slot
    /// `k = x >> 62`, then `k + 1 ..` modulo 4. It crosses the first face
    /// with `p` strictly outside (`orient3d < 0`), and stops in a finite tet
    /// with no such face, or on entering a hull tet, which then conflicts
    /// strictly. Returns the tet and the number of faces crossed. A walk
    /// longer than the number of live tets is refused as `delaunay/walk`.
    fn walk_tet(&self, p: &Point, start: u32, rng: &mut u64) -> Result<(u32, u32)> {
        let mut t = start;
        let mut crossed: u32 = 0;
        loop {
            debug_assert!(self.is_live(t), "walk on dead tet {}", t);
            if self.is_hull(t) {
                return Ok((t, crossed));
            }
            let k = (splitmix_next(rng) >> 62) as usize;
            let v = self.verts[t as usize];
            let mut next = NONE;
            for d in 0..4 {
                let i = (k + d) & 3;
                let f = face_out(v, i);
                if orient3d(
                    &self.points[f[0] as usize],
                    &self.points[f[1] as usize],
                    &self.points[f[2] as usize],
                    p,
                ) < 0.0
                {
                    crossed += 1;
                    if crossed as usize > self.live_count {
                        return Err(Error::Refused(format!(
                            "delaunay/walk: crossed {} faces, {} live tets",
                            crossed, self.live_count
                        )));
                    }
                    next = self.neigh[t as usize][i];
                    debug_assert!(next != NONE, "walk through unset adjacency");
                    break;
                }
            }
            if next == NONE {
                return Ok((t, crossed));
            }
            t = next;
        }
    }

    /// The conflict rule (116.5a): a finite tet conflicts with `p` iff the
    /// perturbed `insphere` of §116.5 is +1; a hull tet `(a, b, c, inf)`
    /// conflicts iff `orient3d(a, b, c, p) > 0`, with `o = 0` decided by its
    /// finite neighbour, which counts one `coplanar_hull`. `pid` is `p`'s
    /// input point index, the id the perturbation sees.
    fn conflicts(&mut self, t: u32, p: &Point, pid: usize) -> bool {
        let v = self.verts[t as usize];
        if v[3] != INF {
            insphere_perturbed(
                [
                    &self.points[v[0] as usize],
                    &self.points[v[1] as usize],
                    &self.points[v[2] as usize],
                    &self.points[v[3] as usize],
                    p,
                ],
                [
                    v[0] as usize,
                    v[1] as usize,
                    v[2] as usize,
                    v[3] as usize,
                    pid,
                ],
            ) > 0
        } else {
            let o = orient3d(
                &self.points[v[0] as usize],
                &self.points[v[1] as usize],
                &self.points[v[2] as usize],
                p,
            );
            if o > 0.0 {
                true
            } else if o < 0.0 {
                false
            } else {
                self.stats.coplanar_hull += 1;
                let n3 = self.neigh[t as usize][3];
                debug_assert!(!self.is_hull(n3), "hull neighbour 3 is finite");
                self.conflicts_finite(n3, p, pid)
            }
        }
    }

    /// The finite half of the conflict rule, for (116.5a)'s `o = 0` branch.
    fn conflicts_finite(&self, t: u32, p: &Point, pid: usize) -> bool {
        let v = self.verts[t as usize];
        insphere_perturbed(
            [
                &self.points[v[0] as usize],
                &self.points[v[1] as usize],
                &self.points[v[2] as usize],
                &self.points[v[3] as usize],
                p,
            ],
            [
                v[0] as usize,
                v[1] as usize,
                v[2] as usize,
                v[3] as usize,
                pid,
            ],
        ) > 0
    }

    /// A slot for a new tet: the LIFO free list first, else a fresh one.
    fn alloc(&mut self) -> u32 {
        if let Some(s) = self.free.pop() {
            s
        } else {
            self.verts.push([NONE; 4]);
            self.neigh.push([NONE; 4]);
            self.live.push(false);
            self.cav_mark.push(0);
            self.chk_mark.push(0);
            (self.verts.len() - 1) as u32
        }
    }

    /// One Bowyer-Watson insertion (SPEC-LIT §116.6): walk from the last
    /// created tet, grow the cavity, star its boundary from `p`, link all
    /// adjacency. Dead slots go on the LIFO free list for reuse. The
    /// cavity marks are per-slot epoch stamps, so nothing here allocates
    /// or clears anything proportional to `slots()` (§116.7).
    fn insert(&mut self, pid: u32) -> Result<()> {
        // Advance this insertion's epoch. On (unreachable) wrap-around,
        // clear every stamp so an old mark can never alias the new epoch.
        if self.epoch == u64::MAX {
            for s in self.cav_mark.iter_mut() {
                *s = 0;
            }
            for s in self.chk_mark.iter_mut() {
                *s = 0;
            }
            self.epoch = 0;
        }
        self.epoch += 1;
        let epoch = self.epoch;
        let p = self.points[pid as usize];
        let start = {
            let mut t = self.last;
            if self.is_hull(t) {
                t = self.neigh[t as usize][3];
            }
            t
        };
        // The stream is owned by the kernel; take it out so the read-only
        // walk can run against `&self`, then put it back.
        let mut rng = std::mem::replace(&mut self.rng, 0);
        let (t0, steps) = self.walk_tet(&p, start, &mut rng)?;
        self.rng = rng;
        self.stats.walk_steps += u64::from(steps);
        // Grow the cavity: BFS over the neighbours, rule (116.5a) on each
        // candidate once. The located tet always conflicts.
        let mut cav: Vec<u32> = vec![t0];
        self.cav_mark[t0 as usize] = epoch;
        self.chk_mark[t0 as usize] = epoch;
        debug_assert!(self.conflicts(t0, &p, pid as usize));
        let mut qi = 0;
        while qi < cav.len() {
            let t = cav[qi];
            qi += 1;
            for i in 0..4 {
                let u = self.neigh[t as usize][i];
                if !self.live[u as usize] || self.chk_mark[u as usize] == epoch {
                    continue;
                }
                self.chk_mark[u as usize] = epoch;
                if self.conflicts(u, &p, pid as usize) {
                    self.cav_mark[u as usize] = epoch;
                    cav.push(u);
                }
            }
        }
        // The cavity's boundary faces, each with the tet beyond it. Wound
        // outward from the cavity tet by §116.4's table. Recorded before
        // any slot is freed or overwritten.
        let mut bnd: Vec<([u32; 3], u32)> = Vec::new();
        for &t in &cav {
            let v = self.verts[t as usize];
            for i in 0..4 {
                let u = self.neigh[t as usize][i];
                if self.cav_mark[u as usize] != epoch {
                    bnd.push((face_out(v, i), u));
                }
            }
        }
        self.stats.cavity_tets += cav.len() as u64;
        for &t in &cav {
            self.live[t as usize] = false;
            self.free.push(t);
        }
        self.live_count -= cav.len();
        // Star every boundary face from p. A finite face is stored in the
        // creation order when that is already positive (116.2), else with
        // the last two slots swapped; a face with INF moves INF to slot 3
        // by an even permutation.
        let mut new_slots: Vec<u32> = Vec::with_capacity(bnd.len());
        let mut pending: Vec<(u32, u32, [u32; 3])> = Vec::with_capacity(bnd.len());
        for &(f, u) in &bnd {
            let creation = [f[0], f[1], f[2], pid];
            let stored = if f[0] == INF || f[1] == INF || f[2] == INF {
                even_perm(creation)
            } else {
                let o = orient3d(
                    &self.points[f[0] as usize],
                    &self.points[f[1] as usize],
                    &self.points[f[2] as usize],
                    &p,
                );
                debug_assert!(o != 0.0, "flat star face");
                if o > 0.0 {
                    creation
                } else {
                    [f[0], f[1], pid, f[2]]
                }
            };
            let s = self.alloc();
            self.verts[s as usize] = stored;
            self.neigh[s as usize] = [NONE; 4];
            self.live[s as usize] = true;
            new_slots.push(s);
            pending.push((s, u, sorted3(f)));
        }
        // Link each new tet to the live tet beyond its boundary face, both
        // directions, by face-set match.
        for (s, u, fs) in pending.iter().copied() {
            let m = (0..4)
                .find(|&i| face_set(self.verts[s as usize], i) == fs)
                .expect("new tet lost its boundary face");
            self.neigh[s as usize][m] = u;
            let j = (0..4)
                .find(|&i| face_set(self.verts[u as usize], i) == fs)
                .expect("boundary neighbour lost the face");
            self.neigh[u as usize][j] = s;
        }
        // Link the new tets among themselves: every remaining face of a new
        // tet is shared with exactly one other new tet. Lookups in the map
        // are fine - the stored order never comes from its iteration.
        let mut open: HashMap<[u32; 3], (u32, usize)> = HashMap::new();
        for &s in &new_slots {
            for i in 0..4 {
                if self.neigh[s as usize][i] != NONE {
                    continue;
                }
                let fs = face_set(self.verts[s as usize], i);
                match open.remove(&fs) {
                    Some((t, j)) => {
                        self.neigh[s as usize][i] = t;
                        self.neigh[t as usize][j] = s;
                    }
                    None => {
                        open.insert(fs, (s, i));
                    }
                }
            }
        }
        debug_assert!(open.is_empty(), "star left unmatched faces");
        self.live_count += new_slots.len();
        self.last = *new_slots.last().expect("empty star");
        self.stats.inserted += 1;
        Ok(())
    }

    /// Verifies the §116.6 gates: adjacency complete and symmetric, INF
    /// only in slot 3, every finite tet positively oriented, every hull
    /// tet's rule-2 sign, every face between two finite tets locally
    /// Delaunay under (116.4), hull convexity, and every point present.
    /// Returns the first violation by name, else the report.
    pub fn check(&self) -> Result<CheckReport> {
        for t in 0..self.slots() as u32 {
            if !self.is_live(t) {
                continue;
            }
            let v = self.verts[t as usize];
            for i in 0..4 {
                let u = self.neigh[t as usize][i];
                if u == NONE || !self.is_live(u) {
                    return Err(Error::Refused(format!(
                        "delaunay/check: neighbour {} (slot {}) of tet {} is not live",
                        u, i, t
                    )));
                }
                let fs = face_set(v, i);
                let mut matched = 0;
                for j in 0..4 {
                    if self.neigh[u as usize][j] == t
                        && face_set(self.verts[u as usize], j) == fs
                    {
                        matched += 1;
                    }
                }
                if matched != 1 {
                    return Err(Error::Refused(format!(
                        "delaunay/check: adjacency of tet {} slot {} is not symmetric",
                        t, i
                    )));
                }
            }
            if v[0] == INF || v[1] == INF || v[2] == INF {
                return Err(Error::Refused(format!(
                    "delaunay/check: INF not in slot 3 of tet {}",
                    t
                )));
            }
        }
        let n = self.points.len();
        let mut seen_v = vec![false; n];
        let mut edges: Vec<[u32; 2]> = Vec::new();
        let mut tris: Vec<[u32; 3]> = Vec::new();
        let mut report = CheckReport::default();
        for t in 0..self.slots() as u32 {
            if !self.is_live(t) {
                continue;
            }
            let v = self.verts[t as usize];
            if self.is_hull(t) {
                report.hull_faces += 1;
                // Rule 2: the finite neighbour 3's apex has orient3d < 0.
                let u = self.neigh[t as usize][3];
                if self.is_hull(u) {
                    return Err(Error::Refused(format!(
                        "delaunay/check: hull tet {} borders a hull tet through its hull face",
                        t
                    )));
                }
                let d = (0..4)
                    .map(|j| self.verts[u as usize][j])
                    .find(|&x| x != v[0] && x != v[1] && x != v[2])
                    .expect("hull neighbour 3 shares all finite vertices");
                let o = orient3d(
                    &self.points[v[0] as usize],
                    &self.points[v[1] as usize],
                    &self.points[v[2] as usize],
                    &self.points[d as usize],
                );
                if o >= 0.0 {
                    return Err(Error::Refused(format!(
                        "delaunay/check: hull tet {} violates rule 2 (sign {})",
                        t, o
                    )));
                }
                // Hull convexity across INF-containing faces.
                for i in 0..3 {
                    let w = self.neigh[t as usize][i];
                    if !self.is_hull(w) {
                        continue;
                    }
                    let wv = self.verts[w as usize];
                    let x = (0..3)
                        .map(|j| wv[j])
                        .find(|&q| q != v[0] && q != v[1] && q != v[2])
                        .expect("hull neighbours share all finite vertices");
                    if orient3d(
                        &self.points[v[0] as usize],
                        &self.points[v[1] as usize],
                        &self.points[v[2] as usize],
                        &self.points[x as usize],
                    ) > 0.0
                    {
                        return Err(Error::Refused(format!(
                            "delaunay/check: hull is not convex at tet {}",
                            t
                        )));
                    }
                }
                continue;
            }
            // A finite tet: (116.2) strictly positive, and its simplices
            // counted.
            let o = orient3d(
                &self.points[v[0] as usize],
                &self.points[v[1] as usize],
                &self.points[v[2] as usize],
                &self.points[v[3] as usize],
            );
            if o <= 0.0 {
                return Err(Error::Refused(format!(
                    "delaunay/check: finite tet {} has orient3d {}",
                    t, o
                )));
            }
            report.tets += 1;
            for &x in &v {
                seen_v[x as usize] = true;
            }
            for a in 0..4 {
                for b in (a + 1)..4 {
                    let mut e = [v[a], v[b]];
                    e.sort_unstable();
                    edges.push(e);
                }
            }
            for a in 0..4 {
                for b in (a + 1)..4 {
                    for c in (b + 1)..4 {
                        tris.push(sorted3([v[a], v[b], v[c]]));
                    }
                }
            }
            // Local Delaunay on every face between two finite tets: the
            // neighbour's apex must be OUTSIDE the circumsphere, == -1.
            for i in 0..4 {
                let u = self.neigh[t as usize][i];
                if self.is_hull(u) {
                    continue;
                }
                report.interior_faces += 1;
                if u < t {
                    continue;
                }
                let fs = face_set(v, i);
                let w = (0..4)
                    .map(|j| self.verts[u as usize][j])
                    .find(|&x| x != fs[0] && x != fs[1] && x != fs[2])
                    .expect("neighbour shares all face vertices");
                let s = insphere_perturbed(
                    [
                        &self.points[v[0] as usize],
                        &self.points[v[1] as usize],
                        &self.points[v[2] as usize],
                        &self.points[v[3] as usize],
                        &self.points[w as usize],
                    ],
                    [
                        v[0] as usize,
                        v[1] as usize,
                        v[2] as usize,
                        v[3] as usize,
                        w as usize,
                    ],
                );
                if s != -1 {
                    return Err(Error::Refused(format!(
                        "delaunay/check: face {} of tet {} is not locally Delaunay",
                        i, t
                    )));
                }
            }
        }
        report.interior_faces /= 2;
        if let Some(i) = seen_v.iter().position(|&s| !s) {
            return Err(Error::Refused(format!(
                "delaunay/check: point {} is in no finite tet",
                i
            )));
        }
        edges.sort_unstable();
        edges.dedup();
        tris.sort_unstable();
        tris.dedup();
        report.vertices = seen_v.iter().filter(|&&s| s).count();
        report.edges = edges.len();
        report.faces = tris.len();
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64 (Steele, Lea & Flood 2014), all operations wrapping - the
    /// same stream the §116.5 tests use.
    struct SplitMix64 {
        state: u64,
    }

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            SplitMix64 { state: seed }
        }
        fn next(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
    }

    /// `u() = (next() >> 11) * 2^-53`, uniform on `[0, 1)`.
    fn draw_u(r: &mut SplitMix64) -> f64 {
        (r.next() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    /// `R(n, seed)`: n points `[u(), u(), u()]`, x then y then z draw.
    fn r_points(n: usize, seed: u64) -> Vec<Point> {
        let mut r = SplitMix64::new(seed);
        (0..n).map(|_| [draw_u(&mut r), draw_u(&mut r), draw_u(&mut r)]).collect()
    }

    /// `S(n, seed)`: n points uniform on the unit sphere, rejecting the
    /// ball of radius 0.25.
    fn s_points_n(n: usize, seed: u64) -> Vec<Point> {
        let mut r = SplitMix64::new(seed);
        let mut out: Vec<Point> = Vec::new();
        while out.len() < n {
            let vx = 2.0 * draw_u(&mut r) - 1.0;
            let vy = 2.0 * draw_u(&mut r) - 1.0;
            let vz = 2.0 * draw_u(&mut r) - 1.0;
            let s2 = vx * vx + vy * vy + vz * vz;
            if s2 > 0.0625 && s2 <= 1.0 {
                let s = s2.sqrt();
                out.push([vx / s, vy / s, vz / s]);
            }
        }
        out
    }

    /// `S(2000, 2027)`.
    fn s_points() -> Vec<Point> {
        s_points_n(2000, 2027)
    }

    /// `L(m)`: the m x m x m lattice, index `i = x + m*(y + m*z)`.
    fn lattice(m: usize) -> Vec<Point> {
        let mut out = Vec::with_capacity(m * m * m);
        for z in 0..m {
            for y in 0..m {
                for x in 0..m {
                    out.push([x as f64, y as f64, z as f64]);
                }
            }
        }
        out
    }

    /// The insertion order `n-1 .. 0`.
    fn reversed(n: usize) -> Vec<u32> {
        (0..n as u32).rev().collect()
    }

    /// Each tet's ids sorted ascending, then the list sorted.
    fn sorted_tets(mut tets: Vec<[u32; 4]>) -> Vec<[u32; 4]> {
        for t in tets.iter_mut() {
            t.sort_unstable();
        }
        tets.sort_unstable();
        tets
    }

    fn first_live(d: &Delaunay) -> u32 {
        (0..d.slots() as u32).find(|&t| d.is_live(t)).expect("no live tets")
    }

    /// The generators self-check: exact shortest-repr literals.
    #[test]
    fn generators_match() {
        let r = r_points(10000, 2026);
        assert_eq!(
            r[0],
            [0.8578542230112182, 0.4716273839414571, 0.667344955216218]
        );
        let s = s_points();
        assert_eq!(
            s[0],
            [-0.3078313421051075, 0.7712631213557475, -0.5571293049682072]
        );
        assert_eq!(
            s[1999],
            [0.19493444655081146, 0.581288524928832, -0.7900026660296732]
        );
        assert_eq!(s.len(), 2000);
    }

    #[test]
    fn unit_tet_and_hull() {
        let pts: Vec<Point> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ];
        let d = Delaunay::build(&pts).unwrap();
        assert_eq!(d.finite_tets().len(), 1);
        assert_eq!(d.hull_faces().len(), 4);
        assert_eq!(d.slots(), 5);
        let rep = d.check().unwrap();
        assert_eq!(rep.vertices, 4);
        assert_eq!(rep.edges, 6);
        assert_eq!(rep.faces, 4);
        assert_eq!(rep.tets, 1);
        assert_eq!(rep.hull_faces, 4);
        assert_eq!(rep.interior_faces, 0);
        assert_eq!(d.finite_volume(), 1.0 / 6.0);
        assert_eq!(d.stats().inserted, 4);
    }

    #[test]
    fn cube_corners() {
        let cube = lattice(2);
        let d = Delaunay::build(&cube).unwrap();
        let rep = d.check().unwrap();
        assert_eq!(rep.tets, 6);
        assert_eq!(rep.edges, 19);
        assert_eq!(rep.faces, 18);
        assert_eq!(rep.vertices, 8);
        assert_eq!(d.hull_faces().len(), 12);
        assert_eq!(tet_digest(&d.finite_tets()), 0x8ecc7ce0071e1d85);
        assert_eq!(d.finite_volume(), 1.0);
        let r = Delaunay::build_ordered(&cube, &reversed(8)).unwrap();
        assert_eq!(tet_digest(&r.finite_tets()), 0x8ecc7ce0071e1d85);
    }

    #[test]
    fn lattice3_brute_force() {
        let pts = lattice(3);
        let d = Delaunay::build(&pts).unwrap();
        let rep = d.check().unwrap();
        assert_eq!(rep.tets, 48);
        assert_eq!(rep.edges, 98);
        assert_eq!(rep.faces, 120);
        assert_eq!(rep.vertices, 27);
        assert_eq!(d.hull_faces().len(), 48);
        assert_eq!(tet_digest(&d.finite_tets()), 0x1ab2235318ee986d);
        assert_eq!(d.finite_volume(), 8.0);
        let r = Delaunay::build_ordered(&pts, &reversed(27)).unwrap();
        assert_eq!(tet_digest(&r.finite_tets()), 0x1ab2235318ee986d);
    }

    #[test]
    fn coplanar_hull_rule() {
        let a: Vec<Point> = vec![
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [3.0, 3.0, 0.0],
        ];
        let b: Vec<Point> = vec![
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [5.0, 5.0, 0.0],
        ];
        let b2: Vec<Point> = vec![
            [5.0, 5.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
        ];
        let expect_a: Vec<Vec<[u32; 4]>> = vec![
            vec![[0, 1, 3, 4], [0, 2, 3, 4]],
            vec![[0, 1, 3, 4], [0, 2, 3, 4]],
        ];
        let expect_b: Vec<Vec<[u32; 4]>> = vec![
            vec![[0, 1, 2, 3], [1, 2, 3, 4]],
            vec![[0, 1, 2, 3], [1, 2, 3, 4]],
        ];
        for (pts, want) in [(&a, &expect_a), (&b, &expect_b), (&b2, &expect_b)] {
            for (k, order) in [
                (0..5u32).collect::<Vec<u32>>(),
                reversed(5),
            ]
            .into_iter()
            .enumerate()
            {
                let d = Delaunay::build_ordered(pts, &order).unwrap();
                d.check().unwrap();
                assert_eq!(sorted_tets(d.finite_tets()), want[k], "tets differ");
            }
        }
        assert!(Delaunay::build(&a).unwrap().stats().coplanar_hull > 0);
        assert!(Delaunay::build(&b).unwrap().stats().coplanar_hull > 0);
    }

    #[test]
    fn locally_delaunay_random() {
        let pts = r_points(10000, 2026);
        let d = Delaunay::build(&pts).unwrap();
        let rep = d.check().unwrap();
        assert_eq!(rep.vertices, 10000);
        assert_eq!(rep.edges, 76389);
        assert_eq!(rep.faces, 132663);
        assert_eq!(rep.tets, 66273);
        assert_eq!(d.hull_faces().len(), 234);
        assert_eq!(tet_digest(&d.finite_tets()), 0x62df92f27011a4d5);
    }

    #[test]
    fn locally_delaunay_lattice() {
        let pts = lattice(20);
        let d = Delaunay::build(&pts).unwrap();
        let rep = d.check().unwrap();
        assert_eq!(rep.vertices, 8000);
        assert_eq!(d.hull_faces().len(), 4332);
        let r = Delaunay::build_ordered(&pts, &reversed(8000)).unwrap();
        r.check().unwrap();
        assert_eq!(tet_digest(&d.finite_tets()), tet_digest(&r.finite_tets()));
    }

    #[test]
    fn locally_delaunay_sphere() {
        let pts = s_points();
        let d = Delaunay::build(&pts).unwrap();
        d.check().unwrap();
        let hf = d.hull_faces();
        assert_eq!(hf.len(), 3996);
        let mut seen = vec![false; 2000];
        for f in &hf {
            for &x in f {
                seen[x as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "a point is off the hull");
    }

    #[test]
    fn euler_ball() {
        for pts in [
            r_points(10000, 2026),
            lattice(20),
            s_points(),
            lattice(2),
            lattice(3),
        ] {
            let d = Delaunay::build(&pts).unwrap();
            let rep = d.check().unwrap();
            let e = rep.vertices as i64 - rep.edges as i64 + rep.faces as i64
                - rep.tets as i64;
            assert_eq!(e, 1, "Euler's relation fails");
        }
    }

    #[test]
    fn volume_closes() {
        let l20 = Delaunay::build(&lattice(20)).unwrap();
        let vol = l20.finite_volume();
        assert!((vol - 6859.0).abs() <= 1e-12 * 6859.0, "vol {}", vol);
        let r10k = Delaunay::build(&r_points(10000, 2026)).unwrap();
        let vol = r10k.finite_volume();
        assert!(
            (vol - 0.9875476112201466).abs() <= 1e-12 * 0.9875476112201466,
            "vol {}",
            vol
        );
        let s = Delaunay::build(&s_points()).unwrap();
        let vol = s.finite_volume();
        assert!(
            (vol - 4.163454815148943).abs() <= 1e-12 * 4.163454815148943,
            "vol {}",
            vol
        );
    }

    #[test]
    fn refusals() {
        let dup: Vec<Point> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ];
        let msg = Delaunay::build(&dup).unwrap_err().to_string();
        assert!(msg.contains("delaunay/duplicate"), "{}", msg);
        assert!(msg.contains("points 1 and 5"), "{}", msg);

        let five: Vec<Point> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, f64::NAN, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let msg = Delaunay::build(&five).unwrap_err().to_string();
        assert!(msg.contains("delaunay/domain: point 2 axis y"), "{}", msg);
        for bad in [1e-200_f64, 1e200, f64::INFINITY] {
            let pts: Vec<Point> = vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, bad],
            ];
            let msg = Delaunay::build(&pts).unwrap_err().to_string();
            assert!(msg.contains("delaunay/domain"), "{}", msg);
        }

        let three: Vec<Point> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let msg = Delaunay::build(&three).unwrap_err().to_string();
        assert!(msg.contains("delaunay/flat"), "{}", msg);
        let flat: Vec<Point> = (0..10)
            .map(|i| [i as f64, (i % 3) as f64, 0.0])
            .collect();
        let msg = Delaunay::build(&flat).unwrap_err().to_string();
        assert!(msg.contains("delaunay/flat"), "{}", msg);
        let line: Vec<Point> = (0..6).map(|i| [i as f64, i as f64, i as f64]).collect();
        let msg = Delaunay::build(&line).unwrap_err().to_string();
        assert!(msg.contains("delaunay/flat"), "{}", msg);

        let cube = lattice(2);
        let msg = Delaunay::build_ordered(&cube, &[0, 0, 1, 2, 3, 4, 5, 6])
            .unwrap_err()
            .to_string();
        assert!(msg.contains("delaunay/order"), "{}", msg);
    }

    #[test]
    fn locate_walk() {
        let pts = r_points(2000, 2026);
        let d = Delaunay::build(&pts).unwrap();
        let start = first_live(&d);
        for q in r_points(1000, 2028) {
            let loc = d.locate(&q, start).unwrap();
            let v = d.tet(loc.tet);
            if d.is_hull(loc.tet) {
                let o = orient3d(
                    &pts[v[0] as usize],
                    &pts[v[1] as usize],
                    &pts[v[2] as usize],
                    &q,
                );
                assert!(o > 0.0, "hull locate outside");
            } else {
                for i in 0..4 {
                    let f = face_out(v, i);
                    let o = orient3d(
                        &pts[f[0] as usize],
                        &pts[f[1] as usize],
                        &pts[f[2] as usize],
                        &q,
                    );
                    assert!(o >= 0.0, "locate outside face {}", i);
                }
            }
        }
        let outside = [2.0, 0.5, 0.5];
        let loc = d.locate(&outside, start).unwrap();
        assert!(d.is_hull(loc.tet), "query beyond the box must be hull");
    }

    /// FNV-1a 64 over the whole slot layout: per slot, the live flag as
    /// 4 LE bytes; then, when live, the 4 ids of `tet(t)` and the 4 of
    /// `neighbours(t)`, each u32 as 4 LE bytes.
    fn layout_digest(d: &Delaunay) -> u64 {
        let mut h = 0xcbf29ce484222325_u64;
        let mut byte = |b: u8| {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x100000001b3);
        };
        for t in 0..d.slots() as u32 {
            let live = d.is_live(t);
            for b in (live as u32).to_le_bytes() {
                byte(b);
            }
            if live {
                for v in d.tet(t) {
                    for b in v.to_le_bytes() {
                        byte(b);
                    }
                }
                for v in d.neighbours(t) {
                    for b in v.to_le_bytes() {
                        byte(b);
                    }
                }
            }
        }
        h
    }

    /// The kernel with the input order makes the same decisions as
    /// TET-03a's, in the same order: slots, statistics and slot layout
    /// are bit-identical (§116.7, The kernel's cost).
    #[test]
    fn input_order_layout_unchanged() {
        let d = Delaunay::build(&r_points(10000, 2026)).unwrap();
        assert_eq!(d.slots(), 66507);
        assert_eq!(
            d.stats(),
            DelaunayStats {
                inserted: 10000,
                walk_steps: 485991,
                cavity_tets: 194564,
                coplanar_hull: 0,
            }
        );
        assert_eq!(layout_digest(&d), 0xb0a266f6cb294888);

        let d = Delaunay::build(&lattice(20)).unwrap();
        assert_eq!(d.slots(), 45486);
        assert_eq!(
            d.stats(),
            DelaunayStats {
                inserted: 8000,
                walk_steps: 174917,
                cavity_tets: 931265,
                coplanar_hull: 237724,
            }
        );
        assert_eq!(layout_digest(&d), 0x36a122b209b2c959);

        let d = Delaunay::build(&r_points(2000, 2026)).unwrap();
        assert_eq!(d.slots(), 13123);
        assert_eq!(
            d.stats(),
            DelaunayStats {
                inserted: 2000,
                walk_steps: 55907,
                cavity_tets: 37510,
                coplanar_hull: 0,
            }
        );
        assert_eq!(layout_digest(&d), 0xc5129e51ec302dc8);
    }

    /// The §116.7 gate on 10^5 uniform points: the kernel's cost, the
    /// tetrahedron set and the volume, with BRIO order.
    #[test]
    fn brio_random_1e5() {
        let pts = r_points(100000, 2026);
        let d = Delaunay::build_brio(&pts).unwrap();
        let rep = d.check().unwrap();
        assert_eq!(rep.vertices, 100000);
        assert_eq!(rep.edges, 772540);
        assert_eq!(rep.faces, 1344865);
        assert_eq!(rep.tets, 672324);
        assert_eq!(rep.hull_faces, 434);
        assert_eq!(tet_digest(&d.finite_tets()), 0x789107d99e34d6cd);
        let e = rep.vertices as i64 - rep.edges as i64 + rep.faces as i64
            - rep.tets as i64;
        assert_eq!(e, 1, "Euler's relation fails");
        let vol = d.finite_volume();
        assert!(
            (vol - 0.9976647305767516).abs() <= 1e-12 * 0.9976647305767516,
            "vol {}",
            vol
        );
        let st = d.stats();
        assert_eq!(st.inserted, 100000);
        let mean_walk = st.walk_steps as f64 / (st.inserted - 4) as f64;
        let mean_cav = st.cavity_tets as f64 / (st.inserted - 4) as f64;
        eprintln!("brio_random_1e5: mean walk {:.2}, mean cavity {:.2}", mean_walk, mean_cav);
        assert!(mean_walk <= 20.0, "mean walk {:.2}", mean_walk);
    }

    /// The tetrahedron set does not depend on the insertion order:
    /// input order, BRIO order and a second seed's BRIO order all give
    /// the same triangulation of 10^5 uniform points.
    #[test]
    fn brio_order_independent_1e5() {
        let pts = r_points(100000, 2026);
        let a = Delaunay::build(&pts).unwrap();
        assert_eq!(tet_digest(&a.finite_tets()), 0x789107d99e34d6cd);
        let b = Delaunay::build_brio(&pts).unwrap();
        assert_eq!(tet_digest(&b.finite_tets()), 0x789107d99e34d6cd);
        let c = Delaunay::build_ordered(&pts, &brio_order(&pts, 7)).unwrap();
        assert_eq!(tet_digest(&c.finite_tets()), 0x789107d99e34d6cd);
    }

    /// 46^3 lattice points (97336) in BRIO order: gates, volume, and the
    /// same tetrahedron set as a second seed's BRIO order.
    #[test]
    fn brio_lattice_46() {
        let pts = lattice(46);
        let d = Delaunay::build_brio(&pts).unwrap();
        let rep = d.check().unwrap();
        assert_eq!(rep.vertices, 97336);
        assert_eq!(rep.hull_faces, 24300);
        let e = rep.vertices as i64 - rep.edges as i64 + rep.faces as i64
            - rep.tets as i64;
        assert_eq!(e, 1, "Euler's relation fails");
        let vol = d.finite_volume();
        assert!((vol - 91125.0).abs() <= 1e-12 * 91125.0, "vol {}", vol);
        let r = Delaunay::build_ordered(&pts, &brio_order(&pts, 7)).unwrap();
        assert_eq!(tet_digest(&d.finite_tets()), tet_digest(&r.finite_tets()));
    }

    /// 10^5 points on the unit sphere: the hull carries every point, and
    /// the ball's volume closes on 4 pi / 3.
    #[test]
    fn brio_sphere_1e5() {
        let pts = s_points_n(100000, 2027);
        assert_eq!(
            pts[99999],
            [0.9853909926425763, 0.1663419063113154, -0.0365371293836266]
        );
        let d = Delaunay::build_brio(&pts).unwrap();
        let rep = d.check().unwrap();
        assert_eq!(rep.hull_faces, 199996);
        let mut seen = vec![false; 100000];
        for f in d.hull_faces() {
            for &x in &f {
                seen[x as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "a point is off the hull");
        let e = rep.vertices as i64 - rep.edges as i64 + rep.faces as i64
            - rep.tets as i64;
        assert_eq!(e, 1, "Euler's relation fails");
        let vol = d.finite_volume();
        assert!(
            (vol - 4.188285937318097).abs() <= 1e-12 * 4.188285937318097,
            "vol {}",
            vol
        );
    }

    /// TET-03b's supervisor run (SPEC-LIT §116.18): 10^6 uniform points,
    /// release. The gate is 15 s or less for `build_brio` and a mean walk
    /// of 20 or less; the rest is the usual check.
    #[test]
    #[ignore = "TET-03b supervisor run (R): 10^6 points, release"]
    fn million_points_timed() {
        let pts = r_points(1000000, 2026);
        let t0 = std::time::Instant::now();
        let d = Delaunay::build_brio(&pts).unwrap();
        let secs = t0.elapsed().as_secs_f64();
        let st = d.stats();
        let mean_walk = st.walk_steps as f64 / (st.inserted - 4) as f64;
        let mean_cav = st.cavity_tets as f64 / (st.inserted - 4) as f64;
        let finite = (0..d.slots() as u32)
            .filter(|&t| d.is_live(t) && !d.is_hull(t))
            .count();
        eprintln!(
            "million_points_timed: {:.3} s, mean walk {:.2}, mean cavity {:.2}, slots {}, finite tets {}",
            secs, mean_walk, mean_cav, d.slots(), finite
        );
        assert!(secs <= 15.0, "build_brio took {:.3} s", secs);
        assert!(mean_walk <= 20.0, "mean walk {:.2}", mean_walk);
        let rep = d.check().unwrap();
        assert_eq!(rep.vertices, 1000000);
        assert_eq!(rep.edges, 7749181);
        assert_eq!(rep.faces, 13498055);
        assert_eq!(rep.tets, 6748873);
        assert_eq!(rep.hull_faces, 618);
        assert_eq!(tet_digest(&d.finite_tets()), 0x7fe01dd487275a08);
        let e = rep.vertices as i64 - rep.edges as i64 + rep.faces as i64
            - rep.tets as i64;
        assert_eq!(e, 1, "Euler's relation fails");
        let vol = d.finite_volume();
        assert!(
            (vol - 0.9996391098418097).abs() <= 1e-12 * 0.9996391098418097,
            "vol {}",
            vol
        );
    }
}
