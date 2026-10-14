// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The optimisation of SPEC-LIT §116.14: the topological half (TET-10a)
//! and the geometric half (TET-10b).
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
//! The geometric half moves points, never connectivity. Each point has a
//! freedom (116.21c): free inside, tangential on one plane of the input
//! surface, along a crease of two, or fixed at a corner. The smoothing
//! (116.21d) is a non-smooth ascent of the star's smallest dihedral. The
//! sliver perturbation (116.21e) tries seeded random moves on the
//! vertices of tetrahedra outside the targets. A move is kept only when
//! the star keeps `orient3d > 0` and its smallest dihedral rises strictly,
//! so (116.21b) still holds. [`optimise`] alternates the flips, the
//! smoothing and the perturbation (116.21f).
//!
//! Provenance: ORIGINAL. Written from Shewchuk (2002), "Two discrete
//! optimization algorithms for the topological improvement of tetrahedral
//! meshes" (edge removal, its ring triangulation by Klincsek's dynamic
//! programme, and the 2-3 and 3-2 flips), Klincsek (1980), Ann. Discrete
//! Math. 9:121-123 (the dynamic programme), Freitag & Ollivier-Gooch
//! (1997), Int. J. Numer. Meth. Eng. 40(21):3979-4002 (the flips as a
//! quality move, and the smoothing as the non-smooth ascent of the
//! smallest angle), and Gilbert (1966), SIAM J. Control 4(1):61-80 (the
//! smallest-norm point of a convex hull); the choices are derived in
//! SPEC-LIT §116.14. No GPL-licensed source was consulted.

use super::predicates::orient3d;
use super::recover::FaceRecovery;
use super::{Point, TetOptimiseSpec};
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

// ---------------------------------------------------------------------
// TET-10b: smoothing and sliver perturbation (§116.14, (116.21c)-(116.21f)).
// ---------------------------------------------------------------------

/// The distance within which an angle of the star counts as active in the
/// smoothing's ascent direction, radians (DESIGN, (116.21d)).
const ACTIVE: f64 = 1e-3;

/// The ascent steps tried per vertex and per sweep (DESIGN, (116.21d)).
const ASCENT_STEPS: usize = 8;

/// The halvings of one ascent step's line search (DESIGN, (116.21d)).
const HALVINGS: usize = 12;

/// The perturbation radii, as fractions of the shortest edge at the
/// vertex (DESIGN, (116.21e)).
const PERTURB_RADII: [f64; 3] = [0.2, 0.1, 0.05];

/// The perturbation trials at each radius (DESIGN, (116.21e)).
const PERTURB_TRIALS: usize = 4;

/// How a vertex may move in the smoothing and the perturbation of
/// §116.14 (116.21c). A moved vertex is always placed at
/// `anchor + P (x - anchor)`, the anchor being its input position, so it
/// stays on its plane or line to rounding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Freedom {
    /// An interior vertex: anywhere.
    Free,
    /// A surface vertex: in the plane through its anchor with this unit
    /// normal.
    Plane { normal: Point },
    /// A feature vertex: along the line through its anchor with this unit
    /// direction.
    Line { dir: Point },
    /// A corner: never.
    Fixed,
}

impl Freedom {
    /// `P g`: `g` with the components the freedom forbids removed.
    fn project(&self, g: Point) -> Point {
        match *self {
            Freedom::Free => g,
            Freedom::Plane { normal: n } => {
                let s = dot(g, n);
                [g[0] - s * n[0], g[1] - s * n[1], g[2] - s * n[2]]
            }
            Freedom::Line { dir: u } => {
                let s = dot(g, u);
                [s * u[0], s * u[1], s * u[2]]
            }
            Freedom::Fixed => [0.0; 3],
        }
    }

    /// `anchor + P (x - anchor)`: `x` projected back onto the freedom.
    fn place(&self, anchor: Point, x: Point) -> Point {
        match *self {
            Freedom::Free => x,
            Freedom::Fixed => anchor,
            _ => {
                let d = self.project(sub(x, anchor));
                [anchor[0] + d[0], anchor[1] + d[1], anchor[2] + d[2]]
            }
        }
    }
}

fn unit(a: Point) -> Option<Point> {
    let l = dot(a, a).sqrt();
    if l > 0.0 && l.is_finite() {
        Some([a[0] / l, a[1] / l, a[2] / l])
    } else {
        None
    }
}

/// (116.21c): each point's freedom. A point on input triangles has them
/// grouped into exact planes (every vertex of a triangle with
/// `orient3d == 0` against the group's first triangle): one plane of one
/// patch makes it `Plane`, two planes of one patch each make it `Line`
/// along `n_1 x n_2`, anything else `Fixed`. A point on no input triangle
/// counts its coordinates equal to a bound of the points' bounding box
/// (the domain box): 0 makes it `Free`, 1 `Plane` and 2 `Line` along the
/// box's axes, 3 `Fixed`. A point that would be `Free` but lies on a
/// boundary face of `tets`, and a point on no tetrahedron, are `Fixed`.
pub fn freedoms(fr: &FaceRecovery, tri_patch: &[u32], tets: &[[u32; 4]]) -> Vec<Freedom> {
    let pts = &fr.points;
    let on = fr.point_tris();
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in pts {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let mut used = vec![false; pts.len()];
    for t in tets {
        for &v in t {
            used[v as usize] = true;
        }
    }
    let mut on_boundary = vec![false; pts.len()];
    for f in boundary_faces(tets) {
        for v in f {
            on_boundary[v as usize] = true;
        }
    }
    let axis = |k: usize| {
        let mut e = [0.0; 3];
        e[k] = 1.0;
        e
    };
    let normal = |g: [u32; 3]| {
        let [a, b, c] = g.map(|i| pts[i as usize]);
        unit(cross(sub(b, a), sub(c, a)))
    };
    let mut out = Vec::with_capacity(pts.len());
    for v in 0..pts.len() {
        if !used[v] {
            out.push(Freedom::Fixed);
            continue;
        }
        let f = if !on[v].is_empty() {
            // (the plane's first triangle, its patch, every triangle of that patch)
            let mut planes: Vec<([u32; 3], u32, bool)> = Vec::new();
            for &ti in &on[v] {
                let t = fr.tris[ti as usize];
                let patch = tri_patch.get(ti as usize).copied().unwrap_or(u32::MAX);
                let hit = planes.iter_mut().find(|(g, _, _)| {
                    let [a, b, c] = g.map(|i| pts[i as usize]);
                    t.iter().all(|&i| orient3d(&a, &b, &c, &pts[i as usize]) == 0.0)
                });
                match hit {
                    Some(pl) => pl.2 &= pl.1 == patch,
                    None => planes.push((t, patch, true)),
                }
            }
            match planes.as_slice() {
                [(g, _, true)] => normal(*g).map_or(Freedom::Fixed, |n| Freedom::Plane { normal: n }),
                [(g1, _, true), (g2, _, true)] => match (normal(*g1), normal(*g2)) {
                    (Some(n1), Some(n2)) => {
                        unit(cross(n1, n2)).map_or(Freedom::Fixed, |u| Freedom::Line { dir: u })
                    }
                    _ => Freedom::Fixed,
                },
                _ => Freedom::Fixed,
            }
        } else {
            let pinned: Vec<usize> =
                (0..3).filter(|&k| pts[v][k] == lo[k] || pts[v][k] == hi[k]).collect();
            match pinned.len() {
                0 => Freedom::Free,
                1 => Freedom::Plane { normal: axis(pinned[0]) },
                2 => Freedom::Line { dir: axis(3 - pinned[0] - pinned[1]) },
                _ => Freedom::Fixed,
            }
        };
        out.push(if f == Freedom::Free && on_boundary[v] { Freedom::Fixed } else { f });
    }
    out
}

/// What the optimisation of §116.14 did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OptimiseStats {
    /// Rounds run (a flip pass, a smoothing sweep and a perturbation
    /// sweep each), the last one (with no move) included.
    pub rounds: usize,
    /// 2-3 flips kept, over every round.
    pub flips23: usize,
    /// 3-2 flips kept.
    pub flips32: usize,
    /// Edge removals kept with `m >= 4`.
    pub edge_removals: usize,
    /// Vertex moves kept by the smoothing.
    pub smoothed: usize,
    /// Vertex moves kept by the sliver perturbation.
    pub perturbed: usize,
    /// The smallest dihedral of the mesh, degrees: before the first round,
    /// then after each. Never decreasing (116.21b).
    pub worst: Vec<f64>,
    /// The largest dihedral of the mesh, degrees, beside `worst`.
    pub largest: Vec<f64>,
}

/// The optimised mesh: the points (the same numbering, the movable ones
/// moved) and the tetrahedra.
#[derive(Clone, Debug)]
pub struct Optimised {
    /// The points, `points[v]` moved within `freedom[v]` only.
    pub points: Vec<Point>,
    /// The tetrahedra, each with `orient3d > 0`.
    pub tets: Vec<[u32; 4]>,
    /// What was done.
    pub stats: OptimiseStats,
}

/// The splitmix64 generator of `optimise.perturb_seed` (DESIGN).
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)`.
    fn signed(&mut self) -> f64 {
        2.0 * ((self.next() >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    }

    /// Uniform in the unit ball, by rejection.
    fn ball(&mut self) -> Point {
        loop {
            let p = [self.signed(), self.signed(), self.signed()];
            if dot(p, p) <= 1.0 {
                return p;
            }
        }
    }
}

/// The tetrahedron `t` with vertex `v` at `x`.
fn moved(pts: &[Point], t: [u32; 4], v: u32, x: Point) -> [Point; 4] {
    t.map(|w| if w == v { x } else { pts[w as usize] })
}

/// (116.21c)'s `f_v(x)`: the smallest `q` over the star of `v` with `v`
/// at `x`, or `None` when a tetrahedron of the star has `orient3d <= 0`.
fn star_q(pts: &[Point], tets: &[[u32; 4]], star: &[u32], v: u32, x: Point) -> Option<f64> {
    let mut m = f64::INFINITY;
    for &t in star {
        let p = moved(pts, tets[t as usize], v, x);
        if orient3d(&p[0], &p[1], &p[2], &p[3]) <= 0.0 {
            return None;
        }
        m = m.min(min_dihedral(p));
    }
    Some(m)
}

/// The shortest edge at `v` over its star, with `v` at `x`.
fn shortest_edge(pts: &[Point], tets: &[[u32; 4]], star: &[u32], v: u32, x: Point) -> f64 {
    let mut l = f64::INFINITY;
    for &t in star {
        for &w in &tets[t as usize] {
            if w != v {
                let d = sub(pts[w as usize], x);
                l = l.min(dot(d, d).sqrt());
            }
        }
    }
    l
}

/// The point of smallest norm in the convex hull of `g` (Gilbert's
/// iteration), or `None` for an empty set.
fn min_norm(g: &[Point]) -> Option<Point> {
    let mut d = *g.iter().min_by(|a, b| dot(**a, **a).total_cmp(&dot(**b, **b)))?;
    for _ in 0..200 {
        let (i, m) = g
            .iter()
            .enumerate()
            .map(|(i, p)| (i, dot(d, *p)))
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))?;
        let dd = dot(d, d);
        if dd - m <= 1e-12 * dd {
            break;
        }
        let w = sub(g[i], d);
        let ww = dot(w, w);
        if ww == 0.0 {
            break;
        }
        let t = (-dot(d, w) / ww).clamp(0.0, 1.0);
        d = [d[0] + t * w[0], d[1] + t * w[1], d[2] + t * w[2]];
    }
    Some(d)
}

/// (116.21d)'s direction: the unit vector along the smallest-norm point of
/// the convex hull of the projected gradients of the star's active angles
/// (within `ACTIVE` of `f`), by central differences with step `1e-6 l`;
/// `None` when that point's norm is at most `1e-6 / l` (stationary).
#[allow(clippy::too_many_arguments)]
fn ascent(
    pts: &[Point],
    tets: &[[u32; 4]],
    star: &[u32],
    v: u32,
    x: Point,
    f: f64,
    free: Freedom,
    l: f64,
) -> Option<Point> {
    let h = 1e-6 * l;
    let mut grads: Vec<Point> = Vec::new();
    for &t in star {
        let tv = tets[t as usize];
        let a = dihedral_angles(moved(pts, tv, v, x));
        let act: Vec<usize> = (0..6).filter(|&k| a[k] <= f + ACTIVE).collect();
        if act.is_empty() {
            continue;
        }
        let mut g = vec![[0.0; 3]; act.len()];
        for j in 0..3 {
            let mut xp = x;
            xp[j] += h;
            let mut xm = x;
            xm[j] -= h;
            let ap = dihedral_angles(moved(pts, tv, v, xp));
            let am = dihedral_angles(moved(pts, tv, v, xm));
            for (i, &k) in act.iter().enumerate() {
                g[i][j] = (ap[k] - am[k]) / (2.0 * h);
            }
        }
        grads.extend(g.into_iter().map(|g| free.project(g)));
    }
    let d = min_norm(&grads)?;
    let n = dot(d, d).sqrt();
    if !(n * l > 1e-6) {
        return None;
    }
    Some([d[0] / n, d[1] / n, d[2] / n])
}

/// One vertex's smoothing (116.21d): the centroid of its star's other
/// vertices, placed on its freedom, if it raises `f_v`; then up to
/// `ASCENT_STEPS` ascent steps of length `0.25 l 2^-k`, `k < HALVINGS`,
/// each kept at the first `k` that raises `f_v` strictly. `Some` new
/// position when `f_v` rose.
fn smooth_vertex(
    pts: &[Point],
    tets: &[[u32; 4]],
    star: &[u32],
    v: u32,
    free: Freedom,
    anchor: Point,
) -> Option<Point> {
    let x0 = pts[v as usize];
    let f0 = star_q(pts, tets, star, v, x0)?;
    let (mut x, mut f) = (x0, f0);
    let mut others: Vec<u32> =
        star.iter().flat_map(|&t| tets[t as usize]).filter(|&w| w != v).collect();
    others.sort_unstable();
    others.dedup();
    let mut c = [0.0; 3];
    for &w in &others {
        for k in 0..3 {
            c[k] += pts[w as usize][k];
        }
    }
    let n = others.len() as f64;
    let xl = free.place(anchor, [c[0] / n, c[1] / n, c[2] / n]);
    if let Some(fl) = star_q(pts, tets, star, v, xl) {
        if fl > f {
            x = xl;
            f = fl;
        }
    }
    for _ in 0..ASCENT_STEPS {
        let l = shortest_edge(pts, tets, star, v, x);
        let Some(u) = ascent(pts, tets, star, v, x, f, free, l) else { break };
        let mut s = 0.25 * l;
        let mut stepped = false;
        for _ in 0..HALVINGS {
            let xn = free.place(anchor, [x[0] + s * u[0], x[1] + s * u[1], x[2] + s * u[2]]);
            if let Some(fnew) = star_q(pts, tets, star, v, xn) {
                if fnew > f {
                    x = xn;
                    f = fnew;
                    stepped = true;
                    break;
                }
            }
            s *= 0.5;
        }
        if !stepped {
            break;
        }
    }
    (f > f0).then_some(x)
}

/// One vertex's perturbation (116.21e): `PERTURB_TRIALS` random moves at
/// each radius `r l` of `PERTURB_RADII`, along `P u` normalised, with `u`
/// uniform in the unit ball; the move that raises `f_v` most, if any does.
fn perturb_vertex(
    pts: &[Point],
    tets: &[[u32; 4]],
    star: &[u32],
    v: u32,
    free: Freedom,
    anchor: Point,
    rng: &mut SplitMix,
) -> Option<Point> {
    let x0 = pts[v as usize];
    let f0 = star_q(pts, tets, star, v, x0)?;
    let l = shortest_edge(pts, tets, star, v, x0);
    let mut best: Option<(f64, Point)> = None;
    for r in PERTURB_RADII {
        for _ in 0..PERTURB_TRIALS {
            let Some(u) = unit(free.project(rng.ball())) else { continue };
            let s = r * l;
            let xn = free.place(anchor, [x0[0] + s * u[0], x0[1] + s * u[1], x0[2] + s * u[2]]);
            if let Some(fv) = star_q(pts, tets, star, v, xn) {
                if fv > best.map_or(f0, |b| b.0) {
                    best = Some((fv, xn));
                }
            }
        }
    }
    best.map(|b| b.1)
}

/// The smallest and the largest dihedral of the mesh, radians.
pub fn dihedral_extremes(pts: &[Point], tets: &[[u32; 4]]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &t in tets {
        for a in dihedral_angles(tet_points(pts, t)) {
            lo = lo.min(a);
            hi = hi.max(a);
        }
    }
    (lo, hi)
}

/// Runs the optimisation of §116.14 (116.21f): at most `spec.passes`
/// rounds of one flip pass ([`flip`], `frozen` as there), one smoothing
/// sweep and, with `spec.perturb_slivers`, one perturbation sweep,
/// stopping after a round with no move. The smoothing visits every vertex
/// that is not `Fixed`, worst `f_v` first (ties by index); the
/// perturbation visits the tetrahedra with `q` below
/// `spec.target_min_dihedral_deg` or a dihedral above
/// `spec.target_max_dihedral_deg`, worst `q` first (ties by index), and
/// each of their vertices that is not `Fixed`. Every move keeps
/// `orient3d > 0` on its star and raises the star's smallest `q`
/// strictly, so (116.21b) holds with the points moving within their
/// freedoms; the connectivity changes only by the flips.
pub fn optimise(
    points: &[Point],
    tets: &[[u32; 4]],
    frozen: &[[u32; 3]],
    freedom: &[Freedom],
    spec: &TetOptimiseSpec,
) -> Result<Optimised> {
    if freedom.len() != points.len() {
        return Err(Error::Refused(format!(
            "optimise/freedom: {} freedoms for {} points",
            freedom.len(),
            points.len()
        )));
    }
    let (tmin, tmax) = (spec.target_min_dihedral_deg, spec.target_max_dihedral_deg);
    if !(tmin.is_finite() && tmax.is_finite() && 0.0 < tmin && tmin < tmax && tmax < 180.0) {
        return Err(Error::Refused(format!(
            "tet.optimise.target_min_dihedral_deg, target_max_dihedral_deg: must satisfy \
             0 < min < max < 180, got {tmin} and {tmax}"
        )));
    }
    if spec.passes == 0 {
        return Err(Error::Refused("optimise/passes: must be >= 1, got 0".to_string()));
    }
    check_valid(points, tets)?;
    let deg = 180.0 / PI;
    let (qmin, qmax) = (tmin / deg, tmax / deg);
    let anchors = points.to_vec();
    let mut pts = points.to_vec();
    let mut tets = tets.to_vec();
    let mut rng = SplitMix(spec.perturb_seed);
    let (lo, hi) = dihedral_extremes(&pts, &tets);
    let mut stats =
        OptimiseStats { worst: vec![lo * deg], largest: vec![hi * deg], ..Default::default() };
    for _ in 0..spec.passes {
        stats.rounds += 1;
        let f = flip(&pts, &tets, frozen, 1)?;
        let mut moves = f.stats.moves();
        stats.flips23 += f.stats.flips23;
        stats.flips32 += f.stats.flips32;
        stats.edge_removals += f.stats.edge_removals;
        tets = f.tets;
        let mut stars: Vec<Vec<u32>> = vec![Vec::new(); pts.len()];
        for (k, t) in tets.iter().enumerate() {
            for &v in t {
                stars[v as usize].push(k as u32);
            }
        }
        // The smoothing sweep.
        let mut order: Vec<(f64, u32)> = (0..pts.len() as u32)
            .filter(|&v| freedom[v as usize] != Freedom::Fixed && !stars[v as usize].is_empty())
            .filter_map(|v| {
                star_q(&pts, &tets, &stars[v as usize], v, pts[v as usize]).map(|q| (q, v))
            })
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (_, v) in order {
            let vi = v as usize;
            if let Some(x) = smooth_vertex(&pts, &tets, &stars[vi], v, freedom[vi], anchors[vi]) {
                pts[vi] = x;
                stats.smoothed += 1;
                moves += 1;
            }
        }
        // The perturbation sweep.
        if spec.perturb_slivers {
            let mut bad: Vec<(f64, u32)> = tets
                .iter()
                .enumerate()
                .filter_map(|(k, &t)| {
                    let a = dihedral_angles(tet_points(&pts, t));
                    let q = a.iter().copied().fold(f64::INFINITY, f64::min);
                    let m = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    (q < qmin || m > qmax).then_some((q, k as u32))
                })
                .collect();
            bad.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            for (_, k) in bad {
                for v in tets[k as usize] {
                    let vi = v as usize;
                    if freedom[vi] == Freedom::Fixed {
                        continue;
                    }
                    let free = freedom[vi];
                    if let Some(x) =
                        perturb_vertex(&pts, &tets, &stars[vi], v, free, anchors[vi], &mut rng)
                    {
                        pts[vi] = x;
                        stats.perturbed += 1;
                        moves += 1;
                    }
                }
            }
        }
        let (lo, hi) = dihedral_extremes(&pts, &tets);
        stats.worst.push(lo * deg);
        stats.largest.push(hi * deg);
        if moves == 0 {
            break;
        }
    }
    Ok(Optimised { points: pts, tets, stats })
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

    // ----- TET-10b: smoothing and sliver perturbation -----

    use super::super::recover::recover_faces;
    use super::super::refine::Refined;
    use super::super::remesh::{remesh, RemeshOptions};
    use super::super::sizefield::{OctreeOptions, SizeField, SizeParams, SizeSource};
    use super::super::surfprep::{prepare, TetSurface};
    use super::super::TetRecoverSpec;
    use std::sync::OnceLock;

    /// The domain box `[lo, hi]^3` as its 8 corner points, on no triangle.
    fn add_box(s: &mut TetSurface, lo: f64, hi: f64) {
        for k in 0..8u32 {
            let c = |b: u32| if (k >> b) & 1 == 1 { hi } else { lo };
            s.points.push([c(0), c(1), c(2)]);
        }
    }

    /// The solid `[-a, a]^3` minus the cylinder `x^2 + y^2 < r^2` along `z`,
    /// faceted with 24 sides, one patch, wound outward. Each cube face's
    /// perimeter carries 6
    /// segments, matched to the 24 circle points at 45 + 15 k degrees.
    fn cube_minus_cylinder(a: f64, r: f64) -> TetSurface {
        let m = 6usize;
        let n = 4 * m;
        let corners = [[a, a], [-a, a], [-a, -a], [a, -a]];
        let mut ring = Vec::with_capacity(n);
        for i in 0..n {
            let (s, j) = (i / m, i % m);
            let (c0, c1) = (corners[s], corners[(s + 1) % 4]);
            let t = j as f64 / m as f64;
            ring.push([c0[0] + t * (c1[0] - c0[0]), c0[1] + t * (c1[1] - c0[1])]);
        }
        let mut points: Vec<Point> = Vec::new();
        for z in [a, -a] {
            for p in &ring {
                points.push([p[0], p[1], z]);
            }
            for i in 0..n {
                let th = (45.0 + 15.0 * i as f64).to_radians();
                points.push([r * th.cos(), r * th.sin(), z]);
            }
        }
        let (pt, ct, pb, cb) = (0u32, n as u32, 2 * n as u32, 3 * n as u32);
        let mut tris: Vec<[u32; 3]> = Vec::new();
        for i in 0..n as u32 {
            let j = (i + 1) % n as u32;
            // top (+z), bottom (-z), the wall (towards the axis), the sides
            tris.push([pt + i, pt + j, ct + j]);
            tris.push([pt + i, ct + j, ct + i]);
            tris.push([pb + i, cb + j, pb + j]);
            tris.push([pb + i, cb + i, cb + j]);
            tris.push([cb + i, ct + j, cb + j]);
            tris.push([cb + i, ct + i, ct + j]);
            tris.push([pb + i, pb + j, pt + j]);
            tris.push([pb + i, pt + j, pt + i]);
        }
        let tri_patch = vec![0; tris.len()];
        TetSurface { points, tris, tri_patch, patch_names: vec!["part".to_string()] }
    }

    /// The prism over the isosceles triangle with a 30-degree apex at
    /// `(-0.5, 0)` and its base at `x = 0.5`, for `z` in `[-0.5, 0.5]`, one
    /// patch, wound outward.
    fn wedge() -> TetSurface {
        let w = 15f64.to_radians().tan();
        let tri = [[-0.5, 0.0], [0.5, -w], [0.5, w]];
        let mut points: Vec<Point> = Vec::new();
        for z in [0.5, -0.5] {
            for p in tri {
                points.push([p[0], p[1], z]);
            }
        }
        let mut tris: Vec<[u32; 3]> = vec![[0, 1, 2], [3, 5, 4]];
        for i in 0..3u32 {
            let j = (i + 1) % 3;
            tris.push([3 + i, 3 + j, j]);
            tris.push([3 + i, j, i]);
        }
        let tri_patch = vec![0; tris.len()];
        TetSurface { points, tris, tri_patch, patch_names: vec!["wedge".to_string()] }
    }

    /// A TET-09 output: `surf` remeshed (feature angle 60 degrees) to the
    /// size field with one patch source on it, `h` on it, `H = 0.4` and
    /// `g = 1.2`, in the box `[-1.25, 1.25]^3`, recovered and refined from
    /// the hull (the flow outside, through the hole). With the remeshed
    /// surface's patches.
    struct Solid {
        refined: Refined,
        tri_patch: Vec<u32>,
    }

    fn solid(surf: &TetSurface, h: f64, cap: u64) -> Result<Solid> {
        let tris: Vec<[Point; 3]> = surf.tris.iter().map(|t| t.map(|v| surf.points[v as usize])).collect();
        let name = surf.patch_names[0].clone();
        let field = SizeField::build(
            SizeParams::new(0.4, 1.2, None).unwrap(),
            &[SizeSource::Patch { name, tris, distance: 0.0, size: h }],
            [-1.25; 3],
            [1.25; 3],
            OctreeOptions::default(),
        )
        .unwrap();
        let hx = |x: Point| field.h(x);
        let prep = prepare(surf, 60.0)?;
        let r = remesh(&prep, &hx, &RemeshOptions::default())?;
        let mut s = r.surface.clone();
        add_box(&mut s, -1.25, 1.25);
        let fr = recover_faces(&s, &r.feature_edges, &TetRecoverSpec::default())?;
        let spec = TetRefineSpec { max_inserted_points: cap, ..TetRefineSpec::default() };
        let refined = refine(&s, fr, &CarveStart::Hull, &spec, &hx)?;
        Ok(Solid { refined, tri_patch: s.tri_patch.clone() })
    }

    fn holed() -> &'static Solid {
        static S: OnceLock<Solid> = OnceLock::new();
        S.get_or_init(|| {
            let cap = TetRefineSpec::default().max_inserted_points;
            solid(&cube_minus_cylinder(0.5, 0.25), 0.0625, cap).unwrap()
        })
    }

    struct Run {
        free: Vec<Freedom>,
        out: Optimised,
    }

    /// Optimises a TET-09 output at the default `tet.optimise` and checks
    /// (116.21b) with the points moved: the smallest dihedral never
    /// decreases, every tetrahedron valid, the boundary faces and the
    /// volume kept. Prints the gate of §116.18: min >= 10, max <= 165.
    fn run(name: &str, r: &Refined, tri_patch: &[u32]) -> Run {
        let (pts, tets) = (&r.recovery.points, &r.carved.tets);
        let free = freedoms(&r.recovery, tri_patch, tets);
        let count = |f: fn(&Freedom) -> bool| free.iter().filter(|x| f(x)).count();
        let t0 = std::time::Instant::now();
        let out = optimise(pts, tets, &r.recovery.subtris, &free, &TetOptimiseSpec::default()).unwrap();
        let secs = t0.elapsed().as_secs_f64();
        let s = &out.stats;
        let monotone = s.worst.windows(2).all(|p| p[1] >= p[0]);
        let (lo, hi) = (*s.worst.last().unwrap(), *s.largest.last().unwrap());
        let pass = lo >= 10.0 && hi <= 165.0;
        let rv = (volume(&out.points, &out.tets) - volume(pts, tets)).abs() / volume(pts, tets);
        println!(
            "optimise {name}: {} points (free {}, plane {}, line {}, fixed {}), {} -> {} tets, {:.2} s; \
             rounds {}, flips 2-3 {} 3-2 {} removals {}, smoothed {}, perturbed {}; min dihedral per round {:?}, \
             max {:?} deg; volume rel {rv:.2e}; min >= 10 and max <= 165: {}",
            pts.len(),
            count(|f| *f == Freedom::Free),
            count(|f| matches!(f, Freedom::Plane { .. })),
            count(|f| matches!(f, Freedom::Line { .. })),
            count(|f| *f == Freedom::Fixed),
            tets.len(),
            out.tets.len(),
            secs,
            s.rounds,
            s.flips23,
            s.flips32,
            s.edge_removals,
            s.smoothed,
            s.perturbed,
            s.worst.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>(),
            s.largest.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>(),
            if pass { "PASS" } else { "MISSED" }
        );
        assert!(monotone, "{:?}", s.worst);
        check_valid(&out.points, &out.tets).unwrap();
        assert_eq!(boundary_faces(&out.tets), boundary_faces(tets), "the boundary faces are kept");
        assert!(rv <= 1e-12, "volume rel {rv:e}");
        let deg = 180.0 / PI;
        let (l2, h2) = dihedral_extremes(&out.points, &out.tets);
        assert_eq!((l2 * deg, h2 * deg), (lo, hi));
        assert!(pass, "{name}: min {lo:.4} deg, max {hi:.4} deg against 10 and 165");
        Run { free, out }
    }

    #[test]
    fn dihedral_bounds_sphere() {
        let c = case();
        let r = &c.refined;
        run("sphere in a box", r, &vec![0; r.recovery.tris.len()]);
    }

    #[test]
    fn dihedral_bounds_cube_minus_cylinder() {
        let s = holed();
        run("cube minus a cylinder", &s.refined, &s.tri_patch);
    }

    /// The 30-degree wedge is the recorded MISS of §116.18: TET-09's
    /// refinement does not terminate on it (the 30-degree angle between
    /// the facets at its apex edge breaks Shewchuk 1998's assumption of
    /// angles of at least 90 degrees), so there is no mesh to optimise. The
    /// test asserts that cause: at `h = 0.1` the refinement reaches a cap of
    /// 20 000 inserted points, against the 2 262 points of the whole sphere
    /// in a box.
    #[test]
    fn dihedral_bounds_wedge() {
        let (h, cap) = (0.1, 20_000);
        let t0 = std::time::Instant::now();
        let e = match solid(&wedge(), h, cap) {
            Ok(s) => panic!(
                "the wedge now refines ({} tets): restore its gate, run(\"30-degree wedge\", ..)",
                s.refined.carved.tets.len()
            ),
            Err(e) => e.to_string(),
        };
        println!(
            "optimise 30-degree wedge: MISSED - TET-09's refinement does not terminate at h = {h} on it ({:.1} s): \
             {e}; min >= 10 and max <= 165 cannot be measured (SPEC-LIT 116.18)",
            t0.elapsed().as_secs_f64()
        );
        assert!(e.contains("refine/cap: "), "{e}");
    }

    /// The distance from `p` to the triangle `abc` (Ericson 2005, 5.1.5).
    fn tri_distance(p: Point, a: Point, b: Point, c: Point) -> f64 {
        let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
        let at = |u: f64, v: f64| [a[0] + u * ab[0] + v * ac[0], a[1] + u * ab[1] + v * ac[1], a[2] + u * ab[2] + v * ac[2]];
        let dist = |q: Point| dot(sub(p, q), sub(p, q)).sqrt();
        let (d1, d2) = (dot(ab, ap), dot(ac, ap));
        if d1 <= 0.0 && d2 <= 0.0 {
            return dist(a);
        }
        let bp = sub(p, b);
        let (d3, d4) = (dot(ab, bp), dot(ac, bp));
        if d3 >= 0.0 && d4 <= d3 {
            return dist(b);
        }
        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            return dist(at(d1 / (d1 - d3), 0.0));
        }
        let cp = sub(p, c);
        let (d5, d6) = (dot(ab, cp), dot(ac, cp));
        if d6 >= 0.0 && d5 <= d6 {
            return dist(c);
        }
        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            return dist(at(0.0, d2 / (d2 - d6)));
        }
        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return dist([b[0] + w * (c[0] - b[0]), b[1] + w * (c[1] - b[1]), b[2] + w * (c[2] - b[2])]);
        }
        let den = 1.0 / (va + vb + vc);
        dist(at(vb * den, vc * den))
    }

    /// Every point on the input surface stays on it within 1e-12 (the
    /// distance to the nearest input triangle); a box point keeps its box
    /// coordinates bit for bit; a `Fixed` point does not move.
    fn stays(name: &str, r: &Refined, run: &Run) {
        let fr = &r.recovery;
        let on = fr.point_tris();
        let (p0, p1) = (&fr.points, &run.out.points);
        let tri: Vec<[Point; 3]> = fr.tris.iter().map(|t| t.map(|v| p0[v as usize])).collect();
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in p0 {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let (mut worst, mut surface_moved, mut box_moved, mut interior_moved) = (0.0f64, 0, 0, 0);
        for v in 0..p0.len() {
            if p1[v] != p0[v] {
                if !on[v].is_empty() {
                    surface_moved += 1;
                } else if matches!(run.free[v], Freedom::Free) {
                    interior_moved += 1;
                } else {
                    box_moved += 1;
                }
            }
            match run.free[v] {
                Freedom::Fixed => assert_eq!(p1[v], p0[v], "a fixed point moved"),
                _ if on[v].is_empty() => {
                    for k in 0..3 {
                        let pinned = p0[v][k] == lo[k] || p0[v][k] == hi[k];
                        if pinned {
                            assert_eq!(p1[v][k].to_bits(), p0[v][k].to_bits(), "box point {v} left its plane");
                        }
                    }
                }
                _ => {
                    let d = tri.iter().map(|t| tri_distance(p1[v], t[0], t[1], t[2])).fold(f64::INFINITY, f64::min);
                    worst = worst.max(d);
                }
            }
        }
        let ok = worst <= 1e-12;
        println!(
            "optimise surface {name}: moved {surface_moved} surface, {box_moved} box, {interior_moved} interior \
             points; the largest distance of a surface point to the input surface {worst:.3e} (bar 1e-12): {}",
            if ok { "PASS" } else { "MISSED" }
        );
        assert!(surface_moved + interior_moved > 0);
        assert!(ok, "{worst:e}");
    }

    #[test]
    fn surface_stays() {
        let c = case();
        let r = &c.refined;
        let k = run("sphere in a box (surface)", r, &vec![0; r.recovery.tris.len()]);
        stays("sphere in a box", r, &k);
        let s = holed();
        let k = run("cube minus a cylinder (surface)", &s.refined, &s.tri_patch);
        stays("cube minus a cylinder", &s.refined, &k);
    }

    #[test]
    fn optimise_refusals_and_freedoms() {
        let pts: Vec<Point> = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
        let t = [[0u32, 1, 2, 3]];
        let spec = TetOptimiseSpec::default();
        let e = optimise(&pts, &t, &[], &[Freedom::Free; 3], &spec).unwrap_err().to_string();
        assert!(e.contains("optimise/freedom: 3 freedoms for 4 points"), "{e}");
        let mut bad = spec.clone();
        bad.target_max_dihedral_deg = 5.0;
        let e = optimise(&pts, &t, &[], &[Freedom::Free; 4], &bad).unwrap_err().to_string();
        assert!(e.contains("must satisfy 0 < min < max < 180, got 10 and 5"), "{e}");
        let mut bad = spec.clone();
        bad.passes = 0;
        let e = optimise(&pts, &t, &[], &[Freedom::Free; 4], &bad).unwrap_err().to_string();
        assert!(e.contains("optimise/passes: must be >= 1, got 0"), "{e}");
        // The projections and the placement back on a plane and a line.
        let p = Freedom::Plane { normal: [0.0, 0.0, 1.0] };
        assert_eq!(p.place([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), [4.0, 5.0, 3.0]);
        let l = Freedom::Line { dir: [0.0, 1.0, 0.0] };
        assert_eq!(l.place([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), [1.0, 5.0, 3.0]);
        assert_eq!(Freedom::Fixed.place([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), [1.0, 2.0, 3.0]);
        // A flattened tetrahedron's free apex is lifted: the smallest
        // dihedral rises, and with the apex fixed nothing moves.
        let flat: Vec<Point> = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.3, 0.3, 0.05]];
        let t = [[0u32, 1, 2, 3]];
        let t = if orient(&flat, t[0]) > 0.0 { t } else { [[1, 0, 2, 3]] };
        let mut free = [Freedom::Fixed; 4];
        free[3] = Freedom::Free;
        let o = optimise(&flat, &t, &[], &free, &spec).unwrap();
        println!("optimise lift: {:?}", o.stats);
        assert!(o.stats.smoothed > 0 && o.stats.worst.last() > o.stats.worst.first());
        assert_eq!(&o.points[..3], &flat[..3]);
        let o = optimise(&flat, &t, &[], &[Freedom::Fixed; 4], &spec).unwrap();
        assert_eq!(o.points, flat);
        assert_eq!(o.stats.rounds, 1);
        // The cube-minus-cylinder classification: its corners are fixed,
        // and the points on the faces are on planes.
        let s = holed();
        let f = freedoms(&s.refined.recovery, &s.tri_patch, &s.refined.carved.tets);
        let on = s.refined.recovery.point_tris();
        let corner = s.refined.recovery.points.iter().position(|p| *p == [0.5, 0.5, 0.5]).unwrap();
        assert_eq!(f[corner], Freedom::Fixed);
        assert!(on.iter().zip(&f).any(|(o, x)| !o.is_empty() && matches!(x, Freedom::Plane { .. })));
        assert!(on.iter().zip(&f).any(|(o, x)| !o.is_empty() && matches!(x, Freedom::Line { .. })));
    }
}
