// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The static non-conformal interface imprint of SPEC-LIT section 120.18
//! (BC-18): two patches whose faces cover the same planar surface but do
//! not match are intersected and split, so every overlapping pair `(a, b)`
//! becomes one sub-face on each side, paired one to one like a conformal
//! `cyclic`, and every cell that owned a face of either patch stays a
//! closed polyhedron. The clip is the reentrant polygon clipper of
//! Sutherland & Hodgman, "Reentrant polygon clipping", Communications of
//! the ACM 17(1):32-42, 1974, doi:10.1145/360767.360802; the plane frame,
//! the Newell area vector and the edge splitting are elementary vector
//! geometry. All arithmetic is f64 in both builds.
//!
//! No GPL-licensed source was consulted.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::io::polymesh::PolyMeshRaw;
use crate::types::DVec3;
use crate::Label;

/// `R` is orthonormal when every entry of `R^T R - I` is this small
/// (120.18.1).
const ORTHO_TOL: f64 = 1e-12;
/// Coverage bar of (120.18.5), relative to the parent's `|S_f|`.
const COVER_TOL: f64 = 1e-8;

/// The rigid map of (120.18.1): `x_A = rot * y_B + shift`, with `rot`
/// orthonormal, that takes a point `y` of patch `B` into the frame of
/// patch `A`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImprintTransform {
    pub rot: [[f64; 3]; 3],
    pub shift: [f64; 3],
}

impl ImprintTransform {
    pub fn identity() -> Self {
        Self {
            rot: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            shift: [0.0; 3],
        }
    }

    /// `R y + t`, as a plain 3x3 matrix product; the identity maps every
    /// coordinate bit for bit.
    pub fn apply(&self, y: [f64; 3]) -> [f64; 3] {
        let r = &self.rot;
        let t = self.shift;
        [
            r[0][0] * y[0] + r[0][1] * y[1] + r[0][2] * y[2] + t[0],
            r[1][0] * y[0] + r[1][1] * y[1] + r[1][2] * y[2] + t[1],
            r[2][0] * y[0] + r[2][1] * y[1] + r[2][2] * y[2] + t[2],
        ]
    }

    /// `R^T (x - t)`.
    pub fn apply_inverse(&self, x: [f64; 3]) -> [f64; 3] {
        let r = &self.rot;
        let d = [x[0] - self.shift[0], x[1] - self.shift[1], x[2] - self.shift[2]];
        [
            r[0][0] * d[0] + r[1][0] * d[1] + r[2][0] * d[2],
            r[0][1] * d[0] + r[1][1] * d[1] + r[2][1] * d[2],
            r[0][2] * d[0] + r[1][2] * d[1] + r[2][2] * d[2],
        ]
    }

    /// `R v`: a direction or normal maps by the rotation alone.
    pub fn rotate(&self, v: [f64; 3]) -> [f64; 3] {
        let r = &self.rot;
        [
            r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
            r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
            r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
        ]
    }
}

/// The options of one imprint: the rigid map of (120.18.1) and the
/// relative tolerance `tau = rel_tol * L` (the default `1e-9`).
#[derive(Debug, Clone, Copy)]
pub struct ImprintOptions {
    pub transform: ImprintTransform,
    pub rel_tol: f64,
}

impl Default for ImprintOptions {
    fn default() -> Self {
        Self {
            transform: ImprintTransform::identity(),
            rel_tol: 1e-9,
        }
    }
}

/// What `imprint` produced: the new mesh with both patches replaced by
/// their sub-faces, the two patch indices, each sub-face's parent as a
/// LOCAL face index of the input patch, and the counts of new points on
/// each side and of faces outside `A` and `B` that gained a point.
#[derive(Debug, Clone)]
pub struct Imprinted {
    pub mesh: PolyMeshRaw,
    pub patch_a: usize,
    pub patch_b: usize,
    pub parent_a: Vec<usize>,
    pub parent_b: Vec<usize>,
    pub n_new_points_a: usize,
    pub n_new_points_b: usize,
    pub n_faces_split: usize,
}

// ==========================================================================
//  Small f64 vector helpers - `[f64; 3]` so the code is the same in both
//  builds (the points are DVec3 either way, SPEC-LIT 118.2).
// ==========================================================================

fn p3(p: &DVec3) -> [f64; 3] {
    [p.x, p.y, p.z]
}

fn v3(p: [f64; 3]) -> DVec3 {
    DVec3::new(p[0], p[1], p[2])
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn mag3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

fn unit3(a: [f64; 3]) -> [f64; 3] {
    scale3(a, 1.0 / mag3(a))
}

fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    mag3(sub3(a, b))
}

/// The Newell area vector of a 3-D loop, `S = 1/2 sum_k x_k x x_{k+1}`
/// (120.18.2).
fn newell3(loop_: &[[f64; 3]]) -> [f64; 3] {
    let n = loop_.len();
    let mut s = [0.0; 3];
    for k in 0..n {
        s = add3(s, cross3(loop_[k], loop_[(k + 1) % n]));
    }
    scale3(s, 0.5)
}

/// The plane frame of one face (120.18.2): the Newell area vector, the
/// centre, the orthonormal `e1, e2, e3` of the clip frame and the loop in
/// `(s, r)`. `pts` are the face's point indices in the input numbering.
/// A vertex farther than `tau` from the face's own plane is refused by
/// name; the clockwise check runs later, after the same-way refusal.
struct FaceFrame {
    local: usize,
    face: usize,
    pts: Vec<usize>,
    xs: Vec<[f64; 3]>,
    s: [f64; 3],
    c: [f64; 3],
    e1: [f64; 3],
    e2: [f64; 3],
    e3: [f64; 3],
    xy: Vec<[f64; 2]>,
}

fn frame_of(
    raw: &PolyMeshRaw,
    patch: &str,
    local: usize,
    face: usize,
    tau: f64,
) -> Result<FaceFrame> {
    let pts: Vec<usize> = raw.faces[face].iter().map(|&k| k as usize).collect();
    let xs: Vec<[f64; 3]> = pts.iter().map(|&k| p3(&raw.points[k])).collect();
    let n = xs.len();
    let mut s = [0.0; 3];
    let mut c = [0.0; 3];
    for k in 0..n {
        s = add3(s, cross3(xs[k], xs[(k + 1) % n]));
        c = add3(c, xs[k]);
    }
    let s = scale3(s, 0.5);
    let c = scale3(c, 1.0 / n as f64);
    if mag3(s) <= 0.0 {
        return Err(Error::Mesh(format!(
            "imprint: face {local} of patch '{patch}' (face {face}) is degenerate: \
             its Newell area vector is zero (SPEC-LIT 120.18)"
        )));
    }
    let e3 = unit3(s);
    let d01 = sub3(xs[1], xs[0]);
    let e1 = unit3(sub3(d01, scale3(e3, dot3(d01, e3))));
    let e2 = cross3(e3, e1);
    let xy: Vec<[f64; 2]> = xs
        .iter()
        .map(|x| {
            let r = sub3(*x, c);
            [dot3(r, e1), dot3(r, e2)]
        })
        .collect();
    for (k, x) in xs.iter().enumerate() {
        let h = dot3(sub3(*x, c), e3);
        if h.abs() > tau {
            return Err(Error::Mesh(format!(
                "imprint: face {local} of patch '{patch}' (face {face}) is not planar: \
                 vertex {k} is {h:.3e} from its plane, tau = {tau:.3e} (SPEC-LIT 120.18)"
            )));
        }
    }
    Ok(FaceFrame {
        local,
        face,
        pts,
        xs,
        s,
        c,
        e1,
        e2,
        e3,
        xy,
    })
}

/// The clockwise refusal of 120.18.2: the loop in `(s, r)` must not turn
/// clockwise anywhere (a cross product below `-tau |edge|`); the clipper
/// needs planar convex faces.
fn check_convex(fr: &FaceFrame, patch: &str, tau: f64) -> Result<()> {
    let n = fr.xy.len();
    for k in 0..n {
        let u = sub2(fr.xy[k], fr.xy[(k + n - 1) % n]);
        let w = sub2(fr.xy[(k + 1) % n], fr.xy[k]);
        let cr = u[0] * w[1] - u[1] * w[0];
        let scale = hypot2(u) * hypot2(w);
        if cr < -tau * scale {
            return Err(Error::Mesh(format!(
                "imprint: face {} of patch '{patch}' (face {}) turns clockwise at \
                 vertex {k}: the clipper needs planar convex faces (SPEC-LIT 120.18)",
                fr.local, fr.face
            )));
        }
    }
    Ok(())
}

fn sub2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn hypot2(a: [f64; 2]) -> f64 {
    (a[0] * a[0] + a[1] * a[1]).sqrt()
}

/// The signed shoelace area of a 2-D loop, positive for CCW.
pub fn polygon_area_2d(p: &[[f64; 2]]) -> f64 {
    let n = p.len();
    let mut a = 0.0;
    for k in 0..n {
        a += p[k][0] * p[(k + 1) % n][1] - p[(k + 1) % n][0] * p[k][1];
    }
    0.5 * a
}

/// Sutherland & Hodgman (120.18.3)-(120.18.4): the `subject` - which must
/// be passed CCW; any other orientation is NOT allowed - clipped by the
/// CCW convex `clip`, one clip edge `p -> q` at a time. With the signed
/// distance `d(x) = cross(q - p, x - p) / |q - p|` a point is inside when
/// `d >= -tol`; each subject edge `S -> E` emits `E` when both are inside,
/// `I` when only `S` is, `I` then `E` when only `E` is, nothing when both
/// are outside, with `I = S + (d_S / (d_S - d_E)) (E - S)`. Returns the
/// clipped loop (may be empty), CCW.
pub fn clip_convex(subject: &[[f64; 2]], clip: &[[f64; 2]], tol: f64) -> Vec<[f64; 2]> {
    let mut cur: Vec<[f64; 2]> = subject.to_vec();
    for e in 0..clip.len() {
        let p = clip[e];
        let q = clip[(e + 1) % clip.len()];
        let ex = q[0] - p[0];
        let ey = q[1] - p[1];
        let elen = (ex * ex + ey * ey).sqrt();
        if elen == 0.0 || cur.is_empty() {
            continue;
        }
        let inp = std::mem::take(&mut cur);
        for i in 0..inp.len() {
            let s = inp[i];
            let e2 = inp[(i + 1) % inp.len()];
            let d_s = (ex * (s[1] - p[1]) - ey * (s[0] - p[0])) / elen;
            let d_e = (ex * (e2[1] - p[1]) - ey * (e2[0] - p[0])) / elen;
            let s_in = d_s >= -tol;
            let e_in = d_e >= -tol;
            if s_in {
                if e_in {
                    cur.push(e2);
                } else {
                    let t = d_s / (d_s - d_e);
                    cur.push([
                        s[0] + t * (e2[0] - s[0]),
                        s[1] + t * (e2[1] - s[1]),
                    ]);
                }
            } else if e_in {
                let t = d_s / (d_s - d_e);
                cur.push([
                    s[0] + t * (e2[0] - s[0]),
                    s[1] + t * (e2[1] - s[1]),
                ]);
                cur.push(e2);
            }
        }
    }
    cur
}

// ==========================================================================
//  The imprint
// ==========================================================================

fn find_patch(raw: &PolyMeshRaw, name: &str) -> Result<usize> {
    raw.patches
        .iter()
        .position(|p| p.name == name)
        .ok_or_else(|| {
            Error::Mesh(format!("imprint: no patch named '{name}' (SPEC-LIT 120.18)"))
        })
}

/// Every entry of `R^T R - I` within `ORTHO_TOL` (120.18.1).
fn check_orthonormal(rot: &[[f64; 3]; 3]) -> Result<()> {
    let mut worst = 0.0f64;
    for i in 0..3 {
        for j in 0..3 {
            let mut rij = 0.0;
            for k in 0..3 {
                rij += rot[k][i] * rot[k][j];
            }
            worst = worst.max((rij - if i == j { 1.0 } else { 0.0 }).abs());
        }
    }
    if worst > ORTHO_TOL {
        return Err(Error::Mesh(format!(
            "imprint: the transform R is not orthonormal: max |R^T R - I| = {worst:.3e} \
             > {ORTHO_TOL:.0e} (SPEC-LIT 120.18)"
        )));
    }
    Ok(())
}

/// `tau = rel_tol * L` with `L` the diagonal of the bounding box of the
/// points of patch `A`'s faces (120.18, Tolerance).
fn tau_of(raw: &PolyMeshRaw, faces: &[usize], rel_tol: f64) -> f64 {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for &f in faces {
        for &k in &raw.faces[f] {
            let p = p3(&raw.points[k as usize]);
            for d in 0..3 {
                lo[d] = lo[d].min(p[d]);
                hi[d] = hi[d].max(p[d]);
            }
        }
    }
    let diag = (hi[0] - lo[0]).hypot((hi[1] - lo[1]).hypot(hi[2] - lo[2]));
    rel_tol * diag
}

/// The boundary faces of one patch, as global face indices.
fn patch_faces(raw: &PolyMeshRaw, pi: usize) -> Vec<usize> {
    let n_if = raw.neighbour.len();
    let p = &raw.patches[pi];
    (n_if + p.start..n_if + p.start + p.size).collect()
}

/// The candidate pairs of (120.18, Pairs): the boxes of `a` and `T(b)`,
/// each grown by `tau`, overlap, and every vertex of `T(b)` lies within
/// `tau` of `a`'s plane. A candidate whose normals agree is refused by
/// name - the two sides face the same way, so the transform is wrong.
fn find_candidates(
    fa: &[FaceFrame],
    fb: &[FaceFrame],
    tb: &[Vec<[f64; 3]>],
    tr: &ImprintTransform,
    tau: f64,
    pa: &str,
    pb: &str,
) -> Result<Vec<(usize, usize)>> {
    let mut cands = Vec::new();
    for (x, fra) in fa.iter().enumerate() {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in &fra.xs {
            for d in 0..3 {
                lo[d] = lo[d].min(p[d]);
                hi[d] = hi[d].max(p[d]);
            }
        }
        for (y, frb) in fb.iter().enumerate() {
            let mut ok = true;
            let mut blo = [f64::INFINITY; 3];
            let mut bhi = [f64::NEG_INFINITY; 3];
            for p in &tb[y] {
                for d in 0..3 {
                    blo[d] = blo[d].min(p[d]);
                    bhi[d] = bhi[d].max(p[d]);
                    if (dot3(sub3(*p, fra.c), fra.e3)).abs() > tau {
                        ok = false;
                    }
                }
            }
            if !ok {
                continue;
            }
            let overlap = (0..3).all(|d| lo[d] - tau <= bhi[d] + tau && blo[d] - tau <= hi[d] + tau);
            if !overlap {
                continue;
            }
            let nb = tr.rotate(frb.e3);
            let agreement = dot3(fra.e3, nb);
            if agreement > 0.0 {
                return Err(Error::Mesh(format!(
                    "imprint: patches '{pa}' and '{pb}' face the same way (face {} of \
                     '{pa}' against face {y} of '{pb}'): e3_a . R e3_b = {agreement:+.3} > 0, \
                     so the transform is wrong (SPEC-LIT 120.18)",
                    fra.local
                )));
            }
            cands.push((x, y));
        }
    }
    Ok(cands)
}

/// One surviving pair: the parents and the clipped loop in `A`'s frame,
/// back in world 3-D by (120.18.2).
struct Kept {
    xa: usize,
    yb: usize,
    loop3: Vec<[f64; 3]>,
}

/// The clip of (120.18.3)-(120.18.4) per candidate pair. In `a`'s frame
/// `a`'s loop is CCW; `T(b)`'s loop runs the other way and is reversed
/// first. A result with fewer than three distinct points left, or with
/// `2 |area| <= tau * perimeter` (a contact along an edge or at a
/// corner), is dropped.
fn clip_pairs(
    fa: &[FaceFrame],
    tb: &[Vec<[f64; 3]>],
    cands: &[(usize, usize)],
    tau: f64,
) -> Vec<Kept> {
    let mut kept = Vec::new();
    for &(x, y) in cands {
        let fra = &fa[x];
        let subj: Vec<[f64; 2]> = tb[y]
            .iter()
            .map(|p| {
                let r = sub3(*p, fra.c);
                [dot3(r, fra.e1), dot3(r, fra.e2)]
            })
            .collect();
        let subj: Vec<[f64; 2]> = subj.into_iter().rev().collect();
        let out2 = clip_convex(&subj, &fra.xy, tau);
        let loop3: Vec<[f64; 3]> = out2
            .iter()
            .map(|q| add3(fra.c, add3(scale3(fra.e1, q[0]), scale3(fra.e2, q[1]))))
            .collect();
        let lp = dedupe(&loop3, tau);
        if lp.len() < 3 {
            continue;
        }
        let area = mag3(newell3(&lp));
        let mut per = 0.0;
        for k in 0..lp.len() {
            per = per + dist3(lp[k], lp[(k + 1) % lp.len()]);
        }
        if 2.0 * area <= tau * per {
            continue;
        }
        kept.push(Kept {
            xa: x,
            yb: y,
            loop3: lp,
        });
    }
    kept
}

/// Consecutive repeats of a loop are removed (within `tau`, cyclically).
fn dedupe(loop_: &[[f64; 3]], tau: f64) -> Vec<[f64; 3]> {
    let mut out: Vec<[f64; 3]> = Vec::with_capacity(loop_.len());
    for p in loop_ {
        let keep = match out.last() {
            Some(w) => dist3(*w, *p) > tau,
            None => true,
        };
        if keep {
            out.push(*p);
        }
    }
    while out.len() > 1 && dist3(out[0], out[out.len() - 1]) <= tau {
        out.pop();
    }
    out
}

// ==========================================================================
//  Snap and weld (SPEC-LIT section 120.18, "Snap and weld")
// ==========================================================================

/// How the B side gets one sub-face vertex: its own point `m`, a new
/// point at `T^-1(x_k)` for `a`'s vertex `k`, or a new point at
/// `T^-1(v)` for the welded `A`-side point (an `a_new` slot).
enum BKey {
    Orig(usize),
    FromK(usize),
    FromA(usize),
}

/// The snap-and-weld result: the new points of each side and, per kept
/// pair, the parents (local indices) and the sub-face loops of both
/// sides. The `B` loop of a pair is the `A` loop reversed with the same
/// first vertex, written with the `B` side's indices.
struct Resolved {
    a_new: Vec<[f64; 3]>,
    b_new: Vec<[f64; 3]>,
    xa: Vec<usize>,
    yb: Vec<usize>,
    a_loops: Vec<Vec<Label>>,
    b_loops: Vec<Vec<Label>>,
}

fn resolve(
    raw: &PolyMeshRaw,
    fa: &[FaceFrame],
    fb: &[FaceFrame],
    tb: &[Vec<[f64; 3]>],
    kept: &[Kept],
    tr: &ImprintTransform,
    n0: usize,
    tau: f64,
) -> Resolved {
    let mut a_new: Vec<[f64; 3]> = Vec::new();
    let mut m_map: HashMap<usize, usize> = HashMap::new();
    let mut xa = Vec::with_capacity(kept.len());
    let mut yb = Vec::with_capacity(kept.len());
    let mut a_loops = Vec::with_capacity(kept.len());
    let mut b_keys: Vec<Vec<BKey>> = Vec::with_capacity(kept.len());
    for kp in kept {
        let fra = &fa[kp.xa];
        let frb = &fb[kp.yb];
        let mut a_loop: Vec<Label> = Vec::with_capacity(kp.loop3.len());
        let mut keys: Vec<BKey> = Vec::with_capacity(kp.loop3.len());
        for v in &kp.loop3 {
            let k = fra.pts.iter().find(|&&p| dist3(p3(&raw.points[p]), *v) <= tau);
            let m = (0..frb.pts.len()).find(|&j| dist3(tb[kp.yb][j], *v) <= tau);
            let (a_idx, key) = match (k, m) {
                (Some(&k), Some(j)) => (k, BKey::Orig(frb.pts[j])),
                (Some(&k), None) => (k, BKey::FromK(k)),
                (None, Some(j)) => {
                    let pt = frb.pts[j];
                    let e = *m_map.entry(pt).or_insert_with(|| {
                        a_new.push(tr.apply(p3(&raw.points[pt])));
                        a_new.len() - 1
                    });
                    (n0 + e, BKey::Orig(pt))
                }
                (None, None) => {
                    let e = a_new
                        .iter()
                        .position(|w| dist3(*w, *v) <= tau)
                        .unwrap_or_else(|| {
                            a_new.push(*v);
                            a_new.len() - 1
                        });
                    (n0 + e, BKey::FromA(e))
                }
            };
            a_loop.push(a_idx as Label);
            keys.push(key);
        }
        xa.push(kp.xa);
        yb.push(kp.yb);
        a_loops.push(a_loop);
        b_keys.push(keys);
    }
    let (b_new, b_loops) = resolve_b(raw, &a_new, &b_keys, tr, n0);
    Resolved {
        a_new,
        b_new,
        xa,
        yb,
        a_loops,
        b_loops,
    }
}

/// The `B` side of the snap and weld: `m` when there is one; else one new
/// point `T^-1(x_k)` per `k`; else one new point `T^-1(v)` per welded
/// `A`-side point. Matching grids add no point at all.
fn resolve_b(
    raw: &PolyMeshRaw,
    a_new: &[[f64; 3]],
    b_keys: &[Vec<BKey>],
    tr: &ImprintTransform,
    n0: usize,
) -> (Vec<[f64; 3]>, Vec<Vec<Label>>) {
    let base = n0 + a_new.len();
    let mut b_new: Vec<[f64; 3]> = Vec::new();
    let mut k_map: HashMap<usize, usize> = HashMap::new();
    let mut e_map: HashMap<usize, usize> = HashMap::new();
    let mut b_loops = Vec::with_capacity(b_keys.len());
    for keys in b_keys {
        let mut b_idx: Vec<usize> = Vec::with_capacity(keys.len());
        for key in keys {
            let idx = match key {
                BKey::Orig(pt) => *pt,
                BKey::FromK(k) => {
                    let e = *k_map.entry(*k).or_insert_with(|| {
                        b_new.push(tr.apply_inverse(p3(&raw.points[*k])));
                        b_new.len() - 1
                    });
                    base + e
                }
                BKey::FromA(e) => {
                    let e = *e_map.entry(*e).or_insert_with(|| {
                        b_new.push(tr.apply_inverse(a_new[*e]));
                        b_new.len() - 1
                    });
                    base + e
                }
            };
            b_idx.push(idx);
        }
        let n = b_idx.len();
        let mut loop_: Vec<Label> = Vec::with_capacity(n);
        loop_.push(b_idx[0] as Label);
        for j in (1..n).rev() {
            loop_.push(b_idx[j] as Label);
        }
        b_loops.push(loop_);
    }
    (b_new, b_loops)
}

// ==========================================================================
//  Coverage (120.18.5)
// ==========================================================================

/// Each original face must be covered by its sub-faces:
/// `| sum |S_i| - |S_f| | <= 1e-8 |S_f|` for every face of `A` and of
/// `B`; otherwise the interface only partly overlaps, which is
/// `cyclicACMI` and refused, naming the face and its covered fraction.
fn check_coverage(
    raw: &PolyMeshRaw,
    pa: &str,
    pb: &str,
    fa: &[FaceFrame],
    fb: &[FaceFrame],
    res: &Resolved,
) -> Result<()> {
    let mut pts: Vec<[f64; 3]> = raw.points.iter().map(p3).collect();
    pts.extend(res.a_new.iter().copied());
    pts.extend(res.b_new.iter().copied());
    let mut sa = vec![0.0; fa.len()];
    let mut sb = vec![0.0; fb.len()];
    for i in 0..res.a_loops.len() {
        let la: Vec<[f64; 3]> = res.a_loops[i].iter().map(|&k| pts[k as usize]).collect();
        sa[res.xa[i]] += mag3(newell3(&la));
        let lb: Vec<[f64; 3]> = res.b_loops[i].iter().map(|&k| pts[k as usize]).collect();
        sb[res.yb[i]] += mag3(newell3(&lb));
    }
    for (fr, sum) in fa.iter().zip(sa.iter()) {
        let sp = mag3(fr.s);
        if (sum - sp).abs() > COVER_TOL * sp {
            return Err(Error::Mesh(format!(
                "imprint: patch '{pa}' face {} (face {}) is covered to {:.6} of its area: \
                 |sum |S_i| - |S_f|| = {:.3e} > {COVER_TOL:.0e} |S_f| - a partly covered \
                 interface is cyclicACMI and refused (SPEC-LIT 120.18)",
                fr.local,
                fr.face,
                sum / sp,
                (sum - sp).abs()
            )));
        }
    }
    for (fr, sum) in fb.iter().zip(sb.iter()) {
        let sp = mag3(fr.s);
        if (sum - sp).abs() > COVER_TOL * sp {
            return Err(Error::Mesh(format!(
                "imprint: patch '{pb}' face {} (face {}) is covered to {:.6} of its area: \
                 |sum |S_i| - |S_f|| = {:.3e} > {COVER_TOL:.0e} |S_f| - a partly covered \
                 interface is cyclicACMI and refused (SPEC-LIT 120.18)",
                fr.local,
                fr.face,
                sum / sp,
                (sum - sp).abs()
            )));
        }
    }
    Ok(())
}

// ==========================================================================
//  Splitting the edges of the cells
// ==========================================================================

/// The insertion table: an original edge `p q` of an `A` face (or a `B`
/// face, with its own points) that now carries that side's new points
/// strictly inside it - `|(v - p) x (q - p)| <= tau |q - p|`, `lambda` in
/// `(0, 1)` - maps to those points. Keys are canonical undirected edges.
fn split_map_for(
    raw: &PolyMeshRaw,
    pts: &[[f64; 3]],
    faces: &[usize],
    news: &[[f64; 3]],
    first: usize,
    tau: f64,
) -> HashMap<(usize, usize), Vec<Label>> {
    let mut keys: Vec<(usize, usize)> = Vec::new();
    for &f in faces {
        let lp = &raw.faces[f];
        for i in 0..lp.len() {
            let u = lp[i] as usize;
            let w = lp[(i + 1) % lp.len()] as usize;
            let key = (u.min(w), u.max(w));
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
    }
    let mut map: HashMap<(usize, usize), Vec<Label>> = HashMap::new();
    for (u, w) in keys {
        let pu = pts[u];
        let pw = pts[w];
        let ew = sub3(pw, pu);
        let elen = mag3(ew);
        if elen == 0.0 {
            continue;
        }
        let mut vs: Vec<Label> = Vec::new();
        for (i, pv) in news.iter().enumerate() {
            let cr = cross3(sub3(*pv, pu), ew);
            if mag3(cr) <= tau * elen {
                let lam = dot3(sub3(*pv, pu), ew) / (elen * elen);
                if lam > 0.0 && lam < 1.0 {
                    vs.push((first + i) as Label);
                }
            }
        }
        if !vs.is_empty() {
            map.insert((u, w), vs);
        }
    }
    map
}

/// One face rebuilt with its edges' new points inserted, in increasing
/// `lambda` along the traversal direction; `None` when nothing goes in.
fn apply_split(
    pts: &[[f64; 3]],
    face: &[Label],
    map: &HashMap<(usize, usize), Vec<Label>>,
) -> Option<Vec<Label>> {
    let mut out: Vec<Label> = Vec::with_capacity(face.len() + 2);
    let mut changed = false;
    for i in 0..face.len() {
        let u = face[i];
        out.push(u);
        let w = face[(i + 1) % face.len()];
        let key = if (u as usize) < w as usize {
            (u as usize, w as usize)
        } else {
            (w as usize, u as usize)
        };
        if let Some(vs) = map.get(&key) {
            let pu = pts[u as usize];
            let ew = sub3(pts[w as usize], pu);
            let mut lam: Vec<(f64, Label)> = vs
                .iter()
                .map(|&v| {
                    let d = sub3(pts[v as usize], pu);
                    (dot3(d, ew), v)
                })
                .collect();
            lam.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("lambda NaN"));
            for (_, v) in lam {
                if !out.contains(&v) {
                    out.push(v);
                    changed = true;
                }
            }
        }
    }
    if changed {
        Some(out)
    } else {
        None
    }
}

// ==========================================================================
//  The output (120.18, The output)
// ==========================================================================

type SplitMap = HashMap<(usize, usize), Vec<Label>>;

/// Internal faces keep their order and owners (with inserted points);
/// patches keep their order, names, types and kinds; `A` and `B` hold
/// their sub-faces with the parents' owners; every `start` is recomputed.
fn assemble(
    pts: &[[f64; 3]],
    raw: &PolyMeshRaw,
    ia: usize,
    ib: usize,
    res: &Resolved,
    map: &SplitMap,
) -> (PolyMeshRaw, usize) {
    let n_if = raw.neighbour.len();
    let mut faces: Vec<Vec<Label>> = Vec::with_capacity(raw.faces.len());
    let mut owner: Vec<Label> = Vec::with_capacity(raw.faces.len());
    let neighbour: Vec<Label> = raw.neighbour.clone();
    let mut n_split = 0usize;
    for f in 0..n_if {
        match apply_split(pts, &raw.faces[f], map) {
            Some(l) => {
                n_split += 1;
                faces.push(l);
            }
            None => faces.push(raw.faces[f].clone()),
        }
        owner.push(raw.owner[f]);
    }
    let mut patches = raw.patches.clone();
    let mut cursor = 0usize;
    for (pi, pat) in raw.patches.iter().enumerate() {
        if pi == ia {
            for i in 0..res.a_loops.len() {
                faces.push(res.a_loops[i].clone());
                owner.push(raw.owner[n_if + pat.start + res.xa[i]]);
            }
        } else if pi == ib {
            for i in 0..res.b_loops.len() {
                faces.push(res.b_loops[i].clone());
                owner.push(raw.owner[n_if + pat.start + res.yb[i]]);
            }
        } else {
            for j in 0..pat.size {
                let f = n_if + pat.start + j;
                match apply_split(pts, &raw.faces[f], map) {
                    Some(l) => {
                        n_split += 1;
                        faces.push(l);
                    }
                    None => faces.push(raw.faces[f].clone()),
                }
                owner.push(raw.owner[f]);
            }
        }
        patches[pi].start = cursor;
        patches[pi].size = faces.len() - n_if - cursor;
        cursor = faces.len() - n_if;
    }
    let mut points: Vec<DVec3> = raw.points.clone();
    points.extend(res.a_new.iter().map(|p| v3(*p)));
    points.extend(res.b_new.iter().map(|p| v3(*p)));
    (
        PolyMeshRaw {
            points,
            faces,
            owner,
            neighbour,
            patches,
        },
        n_split,
    )
}

/// Imprint the non-conformal pair `patch_a` / `patch_b` of `raw`
/// (SPEC-LIT section 120.18): every overlapping pair of an `A` face and a
/// `T(b)` face becomes one sub-face on each side, paired one to one in
/// one pair order (by `A` face, then by `B` face), and every cell stays
/// closed.
pub fn imprint(
    raw: &PolyMeshRaw,
    patch_a: &str,
    patch_b: &str,
    opts: &ImprintOptions,
) -> Result<Imprinted> {
    let ia = find_patch(raw, patch_a)?;
    let ib = find_patch(raw, patch_b)?;
    if ia == ib {
        return Err(Error::Mesh(format!(
            "imprint: patches '{patch_a}' and '{patch_b}' are the same patch \
             (SPEC-LIT 120.18)"
        )));
    }
    for (pi, name) in [(ia, patch_a), (ib, patch_b)] {
        if raw.patches[pi].size == 0 {
            return Err(Error::Mesh(format!(
                "imprint: patch '{name}' is empty (SPEC-LIT 120.18)"
            )));
        }
    }
    check_orthonormal(&opts.transform.rot)?;
    let a_faces = patch_faces(raw, ia);
    let b_faces = patch_faces(raw, ib);
    let mut seen = std::collections::HashSet::new();
    for &f in &a_faces {
        seen.extend(raw.faces[f].iter().copied());
    }
    for &f in &b_faces {
        for &k in &raw.faces[f] {
            if seen.contains(&k) {
                return Err(Error::Mesh(format!(
                    "imprint: patches '{patch_a}' and '{patch_b}' share point {k}: \
                     the two maps would meet in one face (SPEC-LIT 120.18)"
                )));
            }
        }
    }
    let tau = tau_of(raw, &a_faces, opts.rel_tol);
    let fa: Vec<FaceFrame> = a_faces
        .iter()
        .enumerate()
        .map(|(l, &f)| frame_of(raw, patch_a, l, f, tau))
        .collect::<Result<_>>()?;
    let fb: Vec<FaceFrame> = b_faces
        .iter()
        .enumerate()
        .map(|(l, &f)| frame_of(raw, patch_b, l, f, tau))
        .collect::<Result<_>>()?;
    let tb: Vec<Vec<[f64; 3]>> = fb
        .iter()
        .map(|fr| fr.pts.iter().map(|&k| opts.transform.apply(p3(&raw.points[k]))).collect())
        .collect();
    let cands = find_candidates(&fa, &fb, &tb, &opts.transform, tau, patch_a, patch_b)?;
    for fr in &fa {
        check_convex(fr, patch_a, tau)?;
    }
    for fr in &fb {
        check_convex(fr, patch_b, tau)?;
    }
    let kept = clip_pairs(&fa, &tb, &cands, tau);
    let n0 = raw.points.len();
    let res = resolve(raw, &fa, &fb, &tb, &kept, &opts.transform, n0, tau);
    check_coverage(raw, patch_a, patch_b, &fa, &fb, &res)?;
    let na = res.a_new.len();
    let mut pts_all: Vec<[f64; 3]> = raw.points.iter().map(p3).collect();
    pts_all.extend(res.a_new.iter().copied());
    pts_all.extend(res.b_new.iter().copied());
    let mut map = split_map_for(raw, &pts_all, &a_faces, &res.a_new, n0, tau);
    let map_b = split_map_for(raw, &pts_all, &b_faces, &res.b_new, n0 + na, tau);
    for (k, vs) in map_b {
        map.entry(k).or_insert(vs);
    }
    let (mesh, n_split) = assemble(&pts_all, raw, ia, ib, &res, &map);
    Ok(Imprinted {
        mesh,
        patch_a: ia,
        patch_b: ib,
        parent_a: res.xa,
        parent_b: res.yb,
        n_new_points_a: na,
        n_new_points_b: res.b_new.len(),
        n_faces_split: n_split,
    })
}

#[cfg(test)]
mod tests {
    use crate as ofgpu;

    include!("../../tests/shared/mesh_imprint_cases.rs");

    #[test]
    fn clip_convex_overlap_gives_a_quarter() {
        let sq: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let moved: Vec<[f64; 2]> = sq.iter().map(|p| [p[0] + 0.5, p[1] + 0.5]).collect();
        let out = clip_convex(&sq, &moved, 1e-12);
        assert!((polygon_area_2d(&out) - 0.25).abs() <= 1e-15);
    }

    #[test]
    fn clip_convex_keeps_an_enclosed_triangle() {
        let sq: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let tri: Vec<[f64; 2]> = vec![[0.2, 0.2], [0.6, 0.2], [0.2, 0.6]];
        let out = clip_convex(&tri, &sq, 1e-12);
        assert!((polygon_area_2d(&out) - 0.08).abs() <= 1e-15);
    }

    #[test]
    fn clip_convex_of_a_disjoint_pair_is_empty() {
        let sq: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let far: Vec<[f64; 2]> = sq.iter().map(|p| [p[0] + 2.0, p[1]]).collect();
        let out = clip_convex(&sq, &far, 1e-12);
        assert!(polygon_area_2d(&out).abs() == 0.0);
    }

    #[test]
    fn clip_convex_of_identical_squares_is_the_square() {
        let sq: Vec<[f64; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let out = clip_convex(&sq, &sq, 1e-12);
        assert!((polygon_area_2d(&out) - 1.0).abs() <= 1e-15);
    }
}

