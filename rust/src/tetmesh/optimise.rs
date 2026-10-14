// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The optimisation of SPEC-LIT §116.14: the topological half (TET-10a).
//!
//! The quality of a tetrahedron is its smallest dihedral angle, (116.21):
//! at edge `e`, `theta_e = pi - arccos(n_1 . n_2)` with `n_1`, `n_2` the
//! outward unit normals of the two faces that meet at `e`. Three moves
//! change the connectivity and never a coordinate:
//!
//! - the 2-3 flip: the two tetrahedra on an interior face become the three
//!   about the edge joining their apexes;
//! - edge removal: the `m` tetrahedra about an interior edge `ab` become
//!   `2 (m - 2)`, two per triangle of a triangulation of the edge's ring
//!   `u_0 .. u_{m-1}`, chosen by the dynamic programme (116.21a) to
//!   maximise the worst new quality; `m = 3` is the 3-2 flip.
//!
//! A move is kept only when every new tetrahedron has `orient3d > 0`
//! (the exact predicate of §116.4), none of its new faces or edges exists
//! already, and the worst quality of the changed star strictly improves,
//! so the mesh's worst quality never decreases (116.21b). A face on one
//! tetrahedron only (the boundary) is never interior to a move, and a
//! face named frozen is never removed, so the surface is untouched.
//!
//! Provenance: ORIGINAL. Written from Shewchuk (2002), "Two discrete
//! optimization algorithms for the topological improvement of tetrahedral
//! meshes" (edge removal, its ring triangulation by Klincsek's dynamic
//! programme, and the 2-3 and 3-2 flips), Klincsek (1980), Ann. Discrete
//! Math. 9:121-123 (the dynamic programme), and Freitag & Ollivier-Gooch
//! (1997), Int. J. Numer. Meth. Eng. 40(21):3979-4002 (the flips as a
//! quality move); the choices are derived in SPEC-LIT §116.14. No
//! GPL-licensed source was consulted.

use super::predicates::orient3d;
use super::Point;
use crate::error::{Error, Result};
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

/// The largest ring an edge removal triangulates (DESIGN, §116.14).
pub const MAX_RING: usize = 10;

fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Point, b: Point) -> Point {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sorted2(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

fn sorted3(mut f: [u32; 3]) -> [u32; 3] {
    f.sort_unstable();
    f
}

/// The faces of a tetrahedron `[v0, v1, v2, v3]` with `orient3d > 0`,
/// wound so the right-hand normal points out, opposite `v3, v2, v1, v0`.
const FACES: [[usize; 3]; 4] = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]];

/// The six edges, in the order of [`dihedral_angles`].
const EDGES: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];

/// The unit normal of `FACES[f]` (outward for `orient3d > 0`; both flip
/// together otherwise, which leaves (116.21) unchanged).
fn face_normal(p: &[Point; 4], f: usize) -> Point {
    let [a, b, c] = FACES[f].map(|k| p[k]);
    let n = cross(sub(b, a), sub(c, a));
    let l = dot(n, n).sqrt();
    if l == 0.0 {
        return [0.0; 3];
    }
    [n[0] / l, n[1] / l, n[2] / l]
}

/// (116.21) at the six edges of `p`, radians, in the order `01, 02, 03,
/// 12, 13, 23`. The two faces at edge `ij` are those opposite the other
/// two vertices. A face of zero area gives 0 at every edge.
pub fn dihedral_angles(p: [Point; 4]) -> [f64; 6] {
    // FACES[f] is opposite vertex 3 - f.
    let n: [Point; 4] = [0, 1, 2, 3].map(|f| face_normal(&p, f));
    if n.iter().any(|v| *v == [0.0; 3]) {
        return [0.0; 6];
    }
    EDGES.map(|[i, j]| {
        let mut o = (0..4).filter(|&k| k != i && k != j);
        let (k, l) = (o.next().unwrap(), o.next().unwrap());
        let c = dot(n[3 - k], n[3 - l]).clamp(-1.0, 1.0);
        PI - c.acos()
    })
}

/// The quality of §116.14: the smallest of (116.21)'s six angles, radians.
pub fn min_dihedral(p: [Point; 4]) -> f64 {
    dihedral_angles(p).into_iter().fold(f64::INFINITY, f64::min)
}

fn tet_points(points: &[Point], t: [u32; 4]) -> [Point; 4] {
    t.map(|v| points[v as usize])
}

fn orient(points: &[Point], t: [u32; 4]) -> f64 {
    let p = tet_points(points, t);
    orient3d(&p[0], &p[1], &p[2], &p[3])
}

/// `f` rotated so its smallest index leads, its cyclic order kept.
fn canonical(f: [u32; 3]) -> [u32; 3] {
    let k = (0..3).min_by_key(|&k| f[k]).unwrap();
    [f[k], f[(k + 1) % 3], f[(k + 2) % 3]]
}

/// The faces on exactly one of `tets`, each wound outward (for tetrahedra
/// with `orient3d > 0`), rotated to lead with its smallest index, sorted.
pub fn boundary_faces(tets: &[[u32; 4]]) -> Vec<[u32; 3]> {
    let mut seen: HashMap<[u32; 3], (usize, [u32; 3])> = HashMap::new();
    for t in tets {
        for f in FACES {
            let w = f.map(|k| t[k]);
            seen.entry(sorted3(w)).or_insert((0, w)).0 += 1;
        }
    }
    let mut out: Vec<[u32; 3]> =
        seen.into_values().filter(|(n, _)| *n == 1).map(|(_, w)| canonical(w)).collect();
    out.sort_unstable();
    out
}

/// (116.21b)'s static half: every index names a point, every tetrahedron
/// has `orient3d > 0`, and no face lies on three or more tetrahedra.
pub fn check_valid(points: &[Point], tets: &[[u32; 4]]) -> Result<()> {
    let mut faces: HashMap<[u32; 3], usize> = HashMap::new();
    for (k, t) in tets.iter().enumerate() {
        if let Some(&v) = t.iter().find(|&&v| v as usize >= points.len()) {
            return Err(Error::Refused(format!(
                "optimise/index: tetrahedron {k} names vertex {v}, but there are {} points",
                points.len()
            )));
        }
        if orient(points, *t) <= 0.0 {
            return Err(Error::Refused(format!(
                "optimise/invalid: tetrahedron {k} {t:?} has orient3d <= 0 (116.2)"
            )));
        }
        for f in FACES {
            let key = sorted3(f.map(|i| t[i]));
            let n = faces.entry(key).or_insert(0);
            *n += 1;
            if *n > 2 {
                return Err(Error::Refused(format!(
                    "optimise/manifold: the face {key:?} lies on three or more tetrahedra"
                )));
            }
        }
    }
    Ok(())
}

/// What the flips did (SPEC-LIT §116.14).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FlipStats {
    /// Passes run, the last one (with no move) included.
    pub passes: usize,
    /// 2-3 flips kept.
    pub flips23: usize,
    /// 3-2 flips kept (edge removals with `m = 3`).
    pub flips32: usize,
    /// Edge removals kept with `m >= 4`.
    pub edge_removals: usize,
    /// The worst quality of the mesh, degrees: before the first pass, then
    /// after each pass. Never decreasing (116.21b).
    pub worst: Vec<f64>,
    /// The smallest strict gain of a kept move, degrees: the new worst of
    /// its star minus the old. Infinite when no move was kept.
    pub min_gain: f64,
}

impl FlipStats {
    /// Every move kept.
    pub fn moves(&self) -> usize {
        self.flips23 + self.flips32 + self.edge_removals
    }
}

/// The flipped mesh: the same points, new tetrahedra (the untouched ones
/// first, in their input order, then the new ones in the order made).
#[derive(Clone, Debug)]
pub struct Flipped {
    /// The tetrahedra, each with `orient3d > 0`.
    pub tets: Vec<[u32; 4]>,
    /// What was done.
    pub stats: FlipStats,
}

/// A move: the tetrahedra it removes and the ones it makes, the worst
/// quality before and after.
struct Move {
    old: Vec<u32>,
    new: Vec<[u32; 4]>,
    before: f64,
    after: f64,
    /// The ring size for an edge removal, 0 for a 2-3 flip.
    ring: usize,
}

struct Mesh<'a> {
    pts: &'a [Point],
    tets: Vec<[u32; 4]>,
    alive: Vec<bool>,
    q: Vec<f64>,
    edges: HashMap<(u32, u32), Vec<u32>>,
    faces: HashMap<[u32; 3], Vec<u32>>,
    frozen: HashSet<[u32; 3]>,
}

impl<'a> Mesh<'a> {
    fn add(&mut self, t: [u32; 4]) {
        let id = self.tets.len() as u32;
        self.tets.push(t);
        self.alive.push(true);
        self.q.push(min_dihedral(tet_points(self.pts, t)));
        for [i, j] in EDGES {
            self.edges.entry(sorted2(t[i], t[j])).or_default().push(id);
        }
        for f in FACES {
            self.faces.entry(sorted3(f.map(|k| t[k]))).or_default().push(id);
        }
    }

    fn remove(&mut self, id: u32) {
        let t = self.tets[id as usize];
        self.alive[id as usize] = false;
        for [i, j] in EDGES {
            let key = sorted2(t[i], t[j]);
            if let Some(v) = self.edges.get_mut(&key) {
                v.retain(|&x| x != id);
                if v.is_empty() {
                    self.edges.remove(&key);
                }
            }
        }
        for f in FACES {
            let key = sorted3(f.map(|k| t[k]));
            if let Some(v) = self.faces.get_mut(&key) {
                v.retain(|&x| x != id);
                if v.is_empty() {
                    self.faces.remove(&key);
                }
            }
        }
    }

    fn has_edge(&self, a: u32, b: u32) -> bool {
        self.edges.contains_key(&sorted2(a, b))
    }

    /// The quality of `t` if it is a valid new tetrahedron, else `-inf`.
    fn new_q(&self, t: [u32; 4]) -> f64 {
        if orient(self.pts, t) > 0.0 {
            min_dihedral(tet_points(self.pts, t))
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Edge removal of `ab` (116.21a): the ring, then the dynamic
    /// programme over its triangulations.
    fn edge_removal(&self, a: u32, b: u32) -> Option<Move> {
        let mut ids = self.edges.get(&sorted2(a, b))?.clone();
        ids.sort_unstable();
        let m = ids.len();
        if !(3..=MAX_RING).contains(&m) {
            return None;
        }
        let mut next: HashMap<u32, u32> = HashMap::with_capacity(m);
        let mut start = u32::MAX;
        for &id in &ids {
            let t = self.tets[id as usize];
            let mut o = t.iter().copied().filter(|&v| v != a && v != b);
            let (c, d) = (o.next()?, o.next()?);
            let (c, d) = if orient(self.pts, [a, b, c, d]) > 0.0 { (c, d) } else { (d, c) };
            if self.frozen.contains(&sorted3([a, b, c])) || next.insert(c, d).is_some() {
                return None;
            }
            if start == u32::MAX {
                start = c;
            }
        }
        // The ring u_0 .. u_{m-1}: orient3d(a, b, u_i, u_{i+1}) > 0. An
        // open ring (the edge is on the boundary) has a vertex with no
        // successor; two closed fans about one edge (overlapping
        // tetrahedra, which `check_valid` does not test) close early.
        let mut u = Vec::with_capacity(m);
        let mut cur = start;
        for step in 0..m {
            u.push(cur);
            cur = *next.get(&cur)?;
            if (cur == start) != (step + 1 == m) {
                return None;
            }
        }
        let before = ids.iter().map(|&id| self.q[id as usize]).fold(f64::INFINITY, f64::min);
        let adjacent = |i: usize, j: usize| j == i + 1 || (i == 0 && j == m - 1);
        // q_ab(i, k, j), i < k < j: the pair of new tetrahedra on the
        // triangle (u_i, u_k, u_j), or -inf.
        let pair = |i: usize, k: usize, j: usize| -> f64 {
            for (x, y) in [(i, k), (k, j), (i, j)] {
                if !adjacent(x, y) && self.has_edge(u[x], u[y]) {
                    return f64::NEG_INFINITY;
                }
            }
            if self.faces.contains_key(&sorted3([u[i], u[k], u[j]])) {
                return f64::NEG_INFINITY;
            }
            let t1 = self.new_q([u[i], u[k], u[j], b]);
            if t1 == f64::NEG_INFINITY {
                return t1;
            }
            t1.min(self.new_q([u[k], u[i], u[j], a]))
        };
        let mut qv = vec![f64::NEG_INFINITY; m * m];
        let mut kv = vec![usize::MAX; m * m];
        for i in 0..m - 1 {
            qv[i * m + i + 1] = f64::INFINITY;
        }
        for gap in 2..m {
            for i in 0..m - gap {
                let j = i + gap;
                for k in i + 1..j {
                    let v = qv[i * m + k].min(qv[k * m + j]);
                    if v <= qv[i * m + j] {
                        continue;
                    }
                    let v = v.min(pair(i, k, j));
                    if v > qv[i * m + j] {
                        qv[i * m + j] = v;
                        kv[i * m + j] = k;
                    }
                }
            }
        }
        let after = qv[m - 1];
        if !(after > before) {
            return None;
        }
        let mut new = Vec::with_capacity(2 * (m - 2));
        let mut stack = vec![(0usize, m - 1)];
        while let Some((i, j)) = stack.pop() {
            if j < i + 2 {
                continue;
            }
            let k = kv[i * m + j];
            new.push([u[i], u[k], u[j], b]);
            new.push([u[k], u[i], u[j], a]);
            stack.push((i, k));
            stack.push((k, j));
        }
        Some(Move { old: ids, new, before, after, ring: m })
    }

    /// The 2-3 flip of the interior face `f` (sorted).
    fn flip23(&self, f: [u32; 3]) -> Option<Move> {
        if self.frozen.contains(&f) {
            return None;
        }
        let ids = self.faces.get(&f)?;
        if ids.len() != 2 {
            return None;
        }
        let mut ids = ids.clone();
        ids.sort_unstable();
        let apex = |id: u32| self.tets[id as usize].into_iter().find(|v| !f.contains(v));
        let (s, t) = (apex(ids[0])?, apex(ids[1])?);
        if s == t || self.has_edge(s, t) {
            return None;
        }
        let ring = if orient(self.pts, [s, t, f[0], f[1]]) > 0.0 {
            [f[0], f[1], f[2]]
        } else {
            [f[0], f[2], f[1]]
        };
        let new: Vec<[u32; 4]> = (0..3).map(|i| [s, t, ring[i], ring[(i + 1) % 3]]).collect();
        let after = new.iter().map(|&n| self.new_q(n)).fold(f64::INFINITY, f64::min);
        let before = self.q[ids[0] as usize].min(self.q[ids[1] as usize]);
        if !(after > before) {
            return None;
        }
        Some(Move { old: ids, new, before, after, ring: 0 })
    }

    fn worst(&self) -> f64 {
        self.q
            .iter()
            .zip(&self.alive)
            .filter(|(_, &a)| a)
            .map(|(&q, _)| q)
            .fold(f64::INFINITY, f64::min)
    }
}

/// Runs at most `passes` passes of the flips of §116.14 over `tets` (each
/// with `orient3d > 0`, no face on three or more), stopping after a pass
/// with no move. A pass takes the tetrahedra present at its start, worst
/// quality first (then by index); for each one still present it evaluates
/// the edge removal of its six edges and the 2-3 flip of its four faces
/// and keeps the candidate with the best new worst quality, if any
/// strictly improves its star. A face in `frozen` (any winding) is never
/// removed; a boundary face never is either. The points are not changed.
pub fn flip(
    points: &[Point],
    tets: &[[u32; 4]],
    frozen: &[[u32; 3]],
    passes: usize,
) -> Result<Flipped> {
    if passes == 0 {
        return Err(Error::Refused("optimise/passes: must be >= 1, got 0".to_string()));
    }
    check_valid(points, tets)?;
    let mut mesh = Mesh {
        pts: points,
        tets: Vec::with_capacity(tets.len() * 2),
        alive: Vec::new(),
        q: Vec::new(),
        edges: HashMap::new(),
        faces: HashMap::new(),
        frozen: frozen.iter().map(|&f| sorted3(f)).collect(),
    };
    for &t in tets {
        mesh.add(t);
    }
    let deg = 180.0 / PI;
    let mut stats = FlipStats { worst: vec![mesh.worst() * deg], min_gain: f64::INFINITY, ..Default::default() };
    for _ in 0..passes {
        stats.passes += 1;
        let mut order: Vec<u32> =
            (0..mesh.tets.len() as u32).filter(|&id| mesh.alive[id as usize]).collect();
        order.sort_by(|&x, &y| mesh.q[x as usize].total_cmp(&mesh.q[y as usize]).then(x.cmp(&y)));
        let mut moved = 0usize;
        for id in order {
            if !mesh.alive[id as usize] {
                continue;
            }
            let t = mesh.tets[id as usize];
            let mut best: Option<Move> = None;
            let mut consider = |c: Option<Move>| {
                if let Some(c) = c {
                    if best.as_ref().map_or(true, |b| c.after > b.after) {
                        best = Some(c);
                    }
                }
            };
            for [i, j] in EDGES {
                consider(mesh.edge_removal(t[i].min(t[j]), t[i].max(t[j])));
            }
            for f in FACES {
                consider(mesh.flip23(sorted3(f.map(|k| t[k]))));
            }
            let Some(mv) = best else { continue };
            for &o in &mv.old {
                mesh.remove(o);
            }
            for &n in &mv.new {
                mesh.add(n);
            }
            match mv.ring {
                0 => stats.flips23 += 1,
                3 => stats.flips32 += 1,
                _ => stats.edge_removals += 1,
            }
            stats.min_gain = stats.min_gain.min((mv.after - mv.before) * deg);
            moved += 1;
        }
        stats.worst.push(mesh.worst() * deg);
        if moved == 0 {
            break;
        }
    }
    let out: Vec<[u32; 4]> =
        mesh.tets.iter().zip(&mesh.alive).filter(|(_, &a)| a).map(|(&t, _)| t).collect();
    Ok(Flipped { tets: out, stats })
}

#[cfg(test)]
mod tests {
    use super::super::carve::CarveStart;
    use super::super::refine::tests::{case, small};
    use super::super::refine::refine;
    use super::super::{TetOptimiseSpec, TetRefineSpec};
    use super::*;

    fn volume(points: &[Point], tets: &[[u32; 4]]) -> f64 {
        let mut s = 0.0;
        for t in tets {
            let p = tet_points(points, *t);
            s += dot(sub(p[0], p[3]), cross(sub(p[1], p[3]), sub(p[2], p[3])));
        }
        s / 6.0
    }

    #[test]
    fn regular_tet_dihedrals() {
        let s = 1.0 / 2f64.sqrt();
        let p = [[1.0, 0.0, -s], [-1.0, 0.0, -s], [0.0, 1.0, s], [0.0, -1.0, s]];
        let p = if orient3d(&p[0], &p[1], &p[2], &p[3]) > 0.0 { p } else { [p[1], p[0], p[2], p[3]] };
        let want = (1.0f64 / 3.0).acos();
        for a in dihedral_angles(p) {
            assert!((a - want).abs() <= 1e-12, "{a} against {want}");
        }
        // The right-angled corner: three 90-degree edges at the origin.
        let c = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
        assert!(orient3d(&c[0], &c[1], &c[2], &c[3]) > 0.0);
        let d = dihedral_angles(c);
        for k in [0, 1, 2] {
            assert!((d[k] - PI / 2.0).abs() <= 1e-14, "{:?}", d);
        }
        let opp = (1.0f64 / 3.0).sqrt().acos();
        for k in [3, 4, 5] {
            assert!((d[k] - opp).abs() <= 1e-14, "{:?}", d);
        }
    }

    /// Three tetrahedra about the edge from `(0, 0, 1)` to `(0, 0, -1)`
    /// whose ring is a triangle in `z = 0`: the 3-2 flip makes two, and
    /// the 2-3 flip of their shared face gives the three back.
    #[test]
    fn three_two_and_back() {
        let pts: Vec<Point> =
            vec![[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [-0.5, 0.9, 0.0], [-0.5, -0.9, 0.0]];
        let mut three = Vec::new();
        for (c, d) in [(2, 3), (3, 4), (4, 2)] {
            let t = [0, 1, c, d];
            three.push(if orient(&pts, t) > 0.0 { t } else { [0, 1, d, c] });
        }
        let q3 = three.iter().map(|&t| min_dihedral(tet_points(&pts, t))).fold(f64::INFINITY, f64::min);
        let f = flip(&pts, &three, &[], 4).unwrap();
        println!("3-2: {:?}, worst {:.4} deg before", f.stats, q3 * 180.0 / PI);
        assert_eq!(f.stats.flips32, 1);
        assert_eq!(f.tets.len(), 2);
        check_valid(&pts, &f.tets).unwrap();
        assert_eq!(boundary_faces(&f.tets), boundary_faces(&three));
        assert!((volume(&pts, &f.tets) - volume(&pts, &three)).abs() <= 1e-15);
        // The two are the better side: no 2-3 flip brings the three back.
        let g = flip(&pts, &f.tets, &[], 4).unwrap();
        assert_eq!(g.stats.moves(), 0);
        assert_eq!(g.tets, f.tets);
        // Freezing a face about the edge refuses the 3-2 flip.
        let h = flip(&pts, &three, &[[1, 0, 3]], 4).unwrap();
        assert_eq!(h.stats.moves(), 0);
        assert_eq!(h.tets, three);
        // A flattened pair: the 2-3 flip of their face is better.
        let flat: Vec<Point> =
            vec![[0.0, 0.0, 0.05], [0.0, 0.0, -0.05], [1.0, 0.0, 0.0], [-0.5, 0.9, 0.0], [-0.5, -0.9, 0.0]];
        let two: Vec<[u32; 4]> = vec![[2, 3, 4, 1], [3, 2, 4, 0]];
        let two: Vec<[u32; 4]> =
            two.into_iter().map(|t| if orient(&flat, t) > 0.0 { t } else { [t[1], t[0], t[2], t[3]] }).collect();
        let k = flip(&flat, &two, &[], 4).unwrap();
        println!("2-3: {:?}", k.stats);
        assert_eq!(k.stats.flips23, 1);
        assert_eq!(k.tets.len(), 3);
        check_valid(&flat, &k.tets).unwrap();
        assert_eq!(boundary_faces(&k.tets), boundary_faces(&two));
        let z = flip(&flat, &two, &[[4, 3, 2]], 4).unwrap();
        assert_eq!(z.stats.moves(), 0, "a frozen face is never removed");
        assert_eq!(z.tets, two);
    }

    #[test]
    fn refusals() {
        let pts: Vec<Point> = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
        let e = flip(&pts, &[[0, 1, 2, 3]], &[], 0).unwrap_err().to_string();
        assert!(e.contains("optimise/passes: must be >= 1, got 0"), "{e}");
        let e = flip(&pts, &[[1, 0, 2, 3]], &[], 1).unwrap_err().to_string();
        assert!(e.contains("optimise/invalid: tetrahedron 0 [1, 0, 2, 3]"), "{e}");
        let e = flip(&pts, &[[0, 1, 2, 4]], &[], 1).unwrap_err().to_string();
        assert!(e.contains("optimise/index: tetrahedron 0 names vertex 4"), "{e}");
    }

    /// The gates' audit of one flipped TET-09 output: monotone worst
    /// quality, at least one move, every new tetrahedron valid, no face on
    /// three, the volume kept, the boundary faces unchanged.
    fn audit(name: &str, points: &[Point], tets: &[[u32; 4]], frozen: &[[u32; 3]]) -> (Flipped, Vec<[u32; 3]>) {
        let f = flip(points, tets, frozen, TetOptimiseSpec::default().passes).unwrap();
        let w = &f.stats.worst;
        let monotone = w.windows(2).all(|p| p[1] >= p[0]);
        let v0 = volume(points, tets);
        let v1 = volume(points, &f.tets);
        let rv = (v1 - v0).abs() / v0.abs();
        println!(
            "optimise flips {name}: {} -> {} tets, {:?}; worst min dihedral per pass {:?} deg, monotone {}; \
             volume rel {rv:.2e}",
            tets.len(),
            f.tets.len(),
            f.stats,
            w.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>(),
            if monotone { "PASS" } else { "MISSED" }
        );
        assert!(monotone, "{w:?}");
        assert!(f.stats.moves() > 0, "the flips kept no move on {name}");
        assert!(f.stats.min_gain > 0.0);
        check_valid(points, &f.tets).unwrap();
        assert!(rv <= 1e-12, "volume rel {rv:e}");
        let b0 = boundary_faces(tets);
        (f, b0)
    }

    #[test]
    fn flips_monotone() {
        let c = case();
        let r = &c.refined;
        audit("sphere in a box", &r.recovery.points, &r.carved.tets, &r.recovery.subtris);
        let h = |_: Point| 0.4;
        let (s, fr) = small();
        let r = refine(&s, fr, &CarveStart::Hull, &TetRefineSpec::default(), &h).unwrap();
        audit("cube in a box", &r.recovery.points, &r.carved.tets, &r.recovery.subtris);
    }

    #[test]
    fn boundary_frozen() {
        let c = case();
        let r = &c.refined;
        let pts = &r.recovery.points;
        let bits0: Vec<[u64; 3]> = pts.iter().map(|p| p.map(f64::to_bits)).collect();
        let (f, b0) = audit("sphere in a box (boundary)", pts, &r.carved.tets, &r.recovery.subtris);
        let b1 = boundary_faces(&f.tets);
        let mut carved: Vec<[u32; 3]> = r.carved.faces.iter().map(|&x| canonical(x)).collect();
        carved.sort_unstable();
        println!(
            "optimise boundary: {} faces before, {} after, carve's {}; identical {}",
            b0.len(),
            b1.len(),
            carved.len(),
            b0 == b1 && b0 == carved
        );
        assert_eq!(b0, carved, "boundary_faces reads the carve's outward faces");
        assert_eq!(b1, b0, "the boundary face set is unchanged");
        let bits1: Vec<[u64; 3]> = pts.iter().map(|p| p.map(f64::to_bits)).collect();
        assert_eq!(bits1, bits0);
        // Every subtriangle is still a face, and a frozen interior face
        // stays: freeze every interior face of the flipped mesh, and no
        // further move removes one.
        let faces: HashSet<[u32; 3]> =
            f.tets.iter().flat_map(|t| FACES.map(|k| sorted3(k.map(|i| t[i])))).collect();
        for s in &r.recovery.subtris {
            assert!(faces.contains(&sorted3(*s)), "subtriangle {s:?} is a face");
        }
        let all: Vec<[u32; 3]> = faces.iter().copied().collect();
        let g = flip(pts, &f.tets, &all, 2).unwrap();
        let mut faces_g: Vec<[u32; 3]> =
            g.tets.iter().flat_map(|t| FACES.map(|k| sorted3(k.map(|i| t[i])))).collect();
        faces_g.sort_unstable();
        faces_g.dedup();
        let mut want: Vec<[u32; 3]> = all.clone();
        want.sort_unstable();
        assert!(want.iter().all(|x| faces_g.binary_search(x).is_ok()), "a frozen face was removed");
    }
}
