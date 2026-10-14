// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 120.11 - `timeVaryingMappedFixedValue`: a fixed value read from
//! an inflow database, `constant/boundaryData/<patch>/points` and one folder
//! per time, interpolated onto the patch's faces in space by a planar
//! Delaunay triangulation of the sample points with barycentric weights, and
//! linearly in time between the two folders that bracket the clock.
//!
//! Space is settled once, at setup, on the host: (120.11a) projects the
//! samples and the face centres onto the samples' plane, (120.11b) is the
//! Delaunay triangulation by Bowyer-Watson insertion (Bowyer 1981,
//! doi:10.1093/comjnl/24.2.162; Watson 1981, doi:10.1093/comjnl/24.2.167;
//! the empty-circle property is Delaunay 1934), and (120.11c) gives each face
//! at most three (sample, weight) pairs, exact for a linear field. Each time
//! folder becomes one f64 slab of face values. Time is (120.11d), the clamp
//! form of SPEC-LIT (120.2), evaluated on the device by `f1TvApplyScalar` /
//! `f1TvApplyVector` in `cuda/function1.cu` (f64, `-fmad=false`) against the
//! clock of SPEC-LIT 120.1; [`TvMappedRule::values64`] is its host twin,
//! bit for bit.
//!
//! No GPL-licensed source was consulted.

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::field_setup::F1Setup;
use crate::io::fields::PatchFieldSpec;
use crate::types::DVec3;
use crate::Scalar;

/// How a face takes its value from the samples (SPEC-LIT 120.11).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapMethod {
    /// (120.11a)-(120.11c): the barycentric weights of the Delaunay triangle
    /// holding the face centre (`planarInterpolation`, the default).
    Planar,
    /// The nearest sample point in 3-D, the lowest index on a tie (`nearest`).
    Nearest,
}

/// One face's stencil: `n` (1 to 3) samples `idx[..n]` with weights `w[..n]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceWeights {
    pub n: usize,
    pub idx: [usize; 3],
    pub w: [f64; 3],
}

impl FaceWeights {
    fn one(i: usize) -> Self {
        Self { n: 1, idx: [i, 0, 0], w: [1.0, 0.0, 0.0] }
    }
    fn two(a: usize, b: usize, s: f64) -> Self {
        Self { n: 2, idx: [a, b, 0], w: [1.0 - s, s, 0.0] }
    }
}

/// (120.11a): the samples' plane - centroid `c`, the in-plane unit axes `e1`
/// and (unless the samples are collinear) `e2`, and the scale `1/L`.
#[derive(Clone, Copy, Debug)]
struct Projection {
    c: DVec3,
    e1: DVec3,
    e2: Option<DVec3>,
    inv_l: f64,
}

/// The relative distance below which the samples count as collinear.
const COLLINEAR: f64 = 1e-9;
/// The barycentric slack within which a face counts as inside a triangle.
const INSIDE: f64 = 1e-10;
/// The super triangle's half-size, in units of the scaled samples.
const SUPER: f64 = 1e3;

impl Projection {
    fn new(p: &[DVec3]) -> Option<Self> {
        let n = p.len();
        let mut c = DVec3::new(0.0, 0.0, 0.0);
        for q in p {
            c = c + *q;
        }
        c = c * (1.0 / n as f64);
        let (mut a, mut la) = (0usize, -1.0f64);
        for (k, q) in p.iter().enumerate() {
            let d = (*q - c).mag();
            if d > la {
                la = d;
                a = k;
            }
        }
        if !(la > 0.0) {
            return None;
        }
        let e1 = (p[a] - c) * (1.0 / la);
        let (mut rb, mut lb) = (DVec3::new(0.0, 0.0, 0.0), -1.0f64);
        for q in p {
            let d = *q - c;
            let r = d - e1 * d.dot(e1);
            let m = r.mag();
            if m > lb {
                lb = m;
                rb = r;
            }
        }
        let e2 = if lb <= COLLINEAR * la { None } else { Some(rb * (1.0 / lb)) };
        Some(Self { c, e1, e2, inv_l: 1.0 / la })
    }

    fn uv(&self, x: DVec3) -> [f64; 2] {
        let d = x - self.c;
        let u = d.dot(self.e1) * self.inv_l;
        let v = match self.e2 {
            Some(e2) => d.dot(e2) * self.inv_l,
            None => 0.0,
        };
        [u, v]
    }
}

/// (120.11c)'s `orient(P, Q, R)`, twice the signed area of `P Q R`.
fn orient(p: [f64; 2], q: [f64; 2], r: [f64; 2]) -> f64 {
    (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
}

/// The circumcircle `(centre u, centre v, radius^2)` of a CCW triangle.
fn circumcircle(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> (f64, f64, f64) {
    let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
    let a2 = a[0] * a[0] + a[1] * a[1];
    let b2 = b[0] * b[0] + b[1] * b[1];
    let c2 = c[0] * c[0] + c[1] * c[1];
    let ux = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d;
    let uy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d;
    let r2 = (a[0] - ux) * (a[0] - ux) + (a[1] - uy) * (a[1] - uy);
    (ux, uy, r2)
}

fn tri_err(msg: String) -> Error {
    Error::Config(format!(
        "timeVaryingMappedFixedValue: the Delaunay triangulation of the sample \
         points failed: {msg} (SPEC-LIT 120.11)"
    ))
}

/// (120.11b): the Delaunay triangulation of `uv` (distinct points, not all
/// collinear), each triangle counter-clockwise, covering the convex hull.
/// Bowyer-Watson insertion inside a super triangle; the cavity of a point is
/// grown from the triangle holding it through edge neighbours whose
/// circumcircle strictly holds it, then shrunk until every cavity edge sees
/// the point counter-clockwise, so round-off can make the result
/// not-quite-Delaunay but never invalid. The super triangles are removed and
/// the boundary is completed to the convex hull.
pub fn delaunay(uv: &[[f64; 2]]) -> Result<Vec<[usize; 3]>> {
    let n = uv.len();
    let mut pts: Vec<[f64; 2]> = uv.to_vec();
    pts.push([-3.0 * SUPER, -3.0 * SUPER]);
    pts.push([3.0 * SUPER, -3.0 * SUPER]);
    pts.push([0.0, 3.0 * SUPER]);
    let mut tris: Vec<[usize; 3]> = vec![[n, n + 1, n + 2]];
    let mut circ: Vec<(f64, f64, f64)> = vec![circumcircle(pts[n], pts[n + 1], pts[n + 2])];
    let mut alive: Vec<bool> = vec![true];

    let shares_edge = |s: &[usize; 3], t: &[usize; 3]| -> bool {
        let mut k = 0;
        for a in s {
            if t.contains(a) {
                k += 1;
            }
        }
        k == 2
    };

    for ip in 0..n {
        let p = pts[ip];
        // The triangle holding p.
        let mut seed = None;
        for (t, tri) in tris.iter().enumerate() {
            if !alive[t] {
                continue;
            }
            let [a, b, c] = *tri;
            if orient(pts[a], pts[b], p) >= 0.0
                && orient(pts[b], pts[c], p) >= 0.0
                && orient(pts[c], pts[a], p) >= 0.0
            {
                seed = Some(t);
                break;
            }
        }
        let Some(seed) = seed else {
            return Err(tri_err(format!("sample {ip} lies in no triangle")));
        };
        // Candidates: circumcircle strictly holds p.
        let mut cand: Vec<usize> = Vec::new();
        for t in 0..tris.len() {
            if !alive[t] || t == seed {
                continue;
            }
            let (cx, cy, r2) = circ[t];
            let d2 = (p[0] - cx) * (p[0] - cx) + (p[1] - cy) * (p[1] - cy);
            if d2 < r2 * (1.0 - 1e-12) {
                cand.push(t);
            }
        }
        let mut excluded: Vec<usize> = Vec::new();
        let cavity = loop {
            // The connected component of the seed among the candidates.
            let mut cav = vec![seed];
            let mut grew = true;
            while grew {
                grew = false;
                for &t in &cand {
                    if cav.contains(&t) || excluded.contains(&t) {
                        continue;
                    }
                    if cav.iter().any(|&s| shares_edge(&tris[s], &tris[t])) {
                        cav.push(t);
                        grew = true;
                    }
                }
            }
            // Every cavity-boundary edge must see p counter-clockwise.
            let mut bad = None;
            'edges: for &t in &cav {
                let tri = tris[t];
                for e in 0..3 {
                    let (a, b) = (tri[e], tri[(e + 1) % 3]);
                    let inner = cav.iter().any(|&s| {
                        s != t && tris[s].contains(&a) && tris[s].contains(&b)
                    });
                    if !inner && !(orient(pts[a], pts[b], p) > 0.0) {
                        bad = Some(t);
                        break 'edges;
                    }
                }
            }
            match bad {
                None => break cav,
                Some(t) if t == seed => {
                    return Err(tri_err(format!(
                        "sample {ip} lies on an edge of the triangle that holds it and \
                         no neighbour can take it"
                    )));
                }
                Some(t) => excluded.push(t),
            }
        };
        // Fan p to the cavity's boundary.
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for &t in &cavity {
            let tri = tris[t];
            for e in 0..3 {
                let (a, b) = (tri[e], tri[(e + 1) % 3]);
                let inner = cavity
                    .iter()
                    .any(|&s| s != t && tris[s].contains(&a) && tris[s].contains(&b));
                if !inner {
                    edges.push((a, b));
                }
            }
        }
        for &t in &cavity {
            alive[t] = false;
        }
        for (a, b) in edges {
            tris.push([a, b, ip]);
            circ.push(circumcircle(pts[a], pts[b], p));
            alive.push(true);
        }
    }

    let mut out: Vec<[usize; 3]> = tris
        .iter()
        .zip(&alive)
        .filter(|(t, al)| **al && t.iter().all(|&v| v < n))
        .map(|(t, _)| *t)
        .collect();
    complete_hull(uv, &mut out);

    // The net: positive areas summing to the convex hull's.
    let mut area = 0.0;
    for t in &out {
        let a = orient(uv[t[0]], uv[t[1]], uv[t[2]]);
        if !(a > 0.0) {
            return Err(tri_err(format!("a triangle {t:?} is not counter-clockwise")));
        }
        area += 0.5 * a;
    }
    let hull = hull_area(uv);
    if !((area - hull).abs() <= 1e-9 * hull) {
        return Err(tri_err(format!(
            "the triangles cover {area:e} of the convex hull's {hull:e}"
        )));
    }
    Ok(out)
}

/// The boundary edges (`a -> b`, interior on the left) of a triangle set.
fn boundary_edges(tris: &[[usize; 3]]) -> Vec<(usize, usize)> {
    let mut count: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for t in tris {
        for e in 0..3 {
            let (a, b) = (t[e], t[(e + 1) % 3]);
            *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    let mut out = Vec::new();
    for t in tris {
        for e in 0..3 {
            let (a, b) = (t[e], t[(e + 1) % 3]);
            if count[&(a.min(b), a.max(b))] == 1 {
                out.push((a, b));
            }
        }
    }
    out
}

/// Fill every right turn of the boundary with the triangle `(a, c, b)`
/// until the triangulation is convex ((120.11b), SPEC-LIT 120.11): a super
/// triangle near a hull edge can take a thin hull triangle with it.
fn complete_hull(uv: &[[f64; 2]], tris: &mut Vec<[usize; 3]>) {
    loop {
        let edges = boundary_edges(tris);
        let mut added = false;
        'search: for &(a, b) in &edges {
            for &(b2, c) in &edges {
                if b2 != b || c == a {
                    continue;
                }
                if !(orient(uv[a], uv[b], uv[c]) < 0.0) {
                    continue;
                }
                // The new triangle (a, c, b) must hold no other sample.
                let t = [a, c, b];
                let blocked = (0..uv.len()).any(|k| {
                    !t.contains(&k)
                        && orient(uv[a], uv[c], uv[k]) > 0.0
                        && orient(uv[c], uv[b], uv[k]) > 0.0
                        && orient(uv[b], uv[a], uv[k]) > 0.0
                });
                if blocked {
                    continue;
                }
                tris.push(t);
                added = true;
                break 'search;
            }
        }
        if !added {
            return;
        }
    }
}

/// The convex hull's area (Andrew's monotone chain).
fn hull_area(uv: &[[f64; 2]]) -> f64 {
    let mut p: Vec<[f64; 2]> = uv.to_vec();
    p.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    let mut h: Vec<[f64; 2]> = Vec::new();
    for pass in 0..2 {
        let start = h.len();
        let it: Box<dyn Iterator<Item = &[f64; 2]>> =
            if pass == 0 { Box::new(p.iter()) } else { Box::new(p.iter().rev()) };
        for q in it {
            while h.len() >= start + 2 && orient(h[h.len() - 2], h[h.len() - 1], *q) <= 0.0 {
                h.pop();
            }
            h.push(*q);
        }
        h.pop();
    }
    let mut a = 0.0;
    for i in 0..h.len() {
        let (p, q) = (h[i], h[(i + 1) % h.len()]);
        a += p[0] * q[1] - q[0] * p[1];
    }
    0.5 * a
}

/// The closest point of segment `a b` to `x`: `(distance^2, s)`.
fn segment(x: [f64; 2], a: [f64; 2], b: [f64; 2]) -> (f64, f64) {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l2 = dx * dx + dy * dy;
    let s = if l2 > 0.0 {
        (((x[0] - a[0]) * dx + (x[1] - a[1]) * dy) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (px, py) = (a[0] + s * dx - x[0], a[1] + s * dy - x[1]);
    (px * px + py * py, s)
}

fn config(field: &str, patch: &str, msg: impl AsRef<str>) -> Error {
    Error::Field {
        field: field.to_string(),
        msg: format!(
            "{patch}: timeVaryingMappedFixedValue: {} (SPEC-LIT 120.11)",
            msg.as_ref()
        ),
    }
}

/// (120.11a)-(120.11c): each target point's stencil on the samples `p`.
/// Distinct samples are required (a repeat is refused by name).
pub fn weights(
    p: &[DVec3],
    x: &[DVec3],
    method: MapMethod,
    field: &str,
    patch: &str,
) -> Result<Vec<FaceWeights>> {
    if p.is_empty() {
        return Err(config(field, patch, "the `points` file holds no sample point"));
    }
    if method == MapMethod::Nearest {
        return Ok(x
            .iter()
            .map(|xf| {
                let mut best = (f64::INFINITY, 0usize);
                for (k, q) in p.iter().enumerate() {
                    let d = (*q - *xf).mag_sqr();
                    if d < best.0 {
                        best = (d, k);
                    }
                }
                FaceWeights::one(best.1)
            })
            .collect());
    }
    let Some(pr) = Projection::new(p) else {
        // One sample (or all at one place, refused below when there are two).
        if p.len() > 1 {
            return Err(config(field, patch, "samples 0 and 1 are the same point"));
        }
        return Ok(vec![FaceWeights::one(0); x.len()]);
    };
    let uv: Vec<[f64; 2]> = p.iter().map(|q| pr.uv(*q)).collect();
    // Distinct samples: sort by (u, v) and compare within a window in u.
    let mut order: Vec<usize> = (0..uv.len()).collect();
    order.sort_by(|&a, &b| uv[a][0].total_cmp(&uv[b][0]).then(uv[a][1].total_cmp(&uv[b][1])));
    let dup = 1e-12;
    for i in 0..order.len() {
        for j in i + 1..order.len() {
            let (a, b) = (uv[order[i]], uv[order[j]]);
            if b[0] - a[0] > dup {
                break;
            }
            if (b[1] - a[1]).abs() <= dup {
                return Err(config(
                    field,
                    patch,
                    format!(
                        "samples {} and {} are the same point",
                        order[i].min(order[j]),
                        order[i].max(order[j])
                    ),
                ));
            }
        }
    }

    if pr.e2.is_none() {
        // Collinear: 1-D along u between neighbours, clamped at the ends.
        return Ok(x
            .iter()
            .map(|xf| {
                let u = pr.uv(*xf)[0];
                let first = order[0];
                let last = order[order.len() - 1];
                if u <= uv[first][0] {
                    return FaceWeights::one(first);
                }
                if u >= uv[last][0] {
                    return FaceWeights::one(last);
                }
                let mut j = 0;
                while j + 2 < order.len() && uv[order[j + 1]][0] <= u {
                    j += 1;
                }
                let (a, b) = (order[j], order[j + 1]);
                let s = (u - uv[a][0]) / (uv[b][0] - uv[a][0]);
                FaceWeights::two(a, b, s)
            })
            .collect());
    }

    let tris = delaunay(&uv)?;
    let edges = boundary_edges(&tris);
    Ok(x
        .iter()
        .map(|xf| {
            let q = pr.uv(*xf);
            for t in &tris {
                let (a, b, c) = (uv[t[0]], uv[t[1]], uv[t[2]]);
                let d = orient(a, b, c);
                let la = orient(q, b, c) / d;
                let lb = orient(a, q, c) / d;
                let lc = orient(a, b, q) / d;
                if la >= -INSIDE && lb >= -INSIDE && lc >= -INSIDE {
                    return FaceWeights { n: 3, idx: *t, w: [la, lb, lc] };
                }
            }
            // Outside the hull: the nearest point of the nearest boundary edge.
            let mut best = (f64::INFINITY, 0usize, 0usize, 0.0);
            for &(a, b) in &edges {
                let (d2, s) = segment(q, uv[a], uv[b]);
                if d2 < best.0 {
                    best = (d2, a, b, s);
                }
            }
            FaceWeights::two(best.1, best.2, best.3)
        })
        .collect())
}

/// `phi_f = SUM_i w_i phi_{idx_i}` per face and component, f64 in stencil
/// order ((120.11c), SPEC-LIT 120.11); `data` is the samples' values, `rank`
/// per point.
pub fn apply_weights(w: &[FaceWeights], data: &[f64], rank: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(w.len() * rank);
    for fw in w {
        for c in 0..rank {
            let mut s = 0.0;
            for j in 0..fw.n {
                s += fw.w[j] * data[fw.idx[j] * rank + c];
            }
            out.push(s);
        }
    }
    out
}

/// One `timeVaryingMappedFixedValue` patch (SPEC-LIT 120.11), built at
/// setup: the face values of every time folder, one f64 slab per time.
#[derive(Clone, Debug, PartialEq)]
pub struct TvMappedRule {
    /// The patch's name, for diagnostics.
    pub patch: String,
    /// First flattened boundary face of the patch.
    pub start: usize,
    /// Its faces.
    pub n: usize,
    /// 1 (scalar field) or 3 (vector field).
    pub n_t: usize,
    /// The folders' times, strictly increasing.
    pub times: Vec<f64>,
    /// `[times.len() * n * n_t]`, slab `k` the face values of folder `k`,
    /// component-fastest.
    pub values: Vec<f64>,
}

impl TvMappedRule {
    /// (120.11d) in f64: the clamp form of (120.2) on the slabs - the host
    /// twin of `f1TvApplyScalar`/`f1TvApplyVector`, the same operations in
    /// the same order.
    pub fn values64(&self, t: f64) -> Vec<f64> {
        let m = self.n * self.n_t;
        let k = self.times.len();
        if k == 1 || t <= self.times[0] {
            return self.values[..m].to_vec();
        }
        if t >= self.times[k - 1] {
            return self.values[(k - 1) * m..k * m].to_vec();
        }
        let mut j = 0;
        while j + 1 < k && self.times[j + 1] <= t {
            j += 1;
        }
        let lam = (t - self.times[j]) / (self.times[j + 1] - self.times[j]);
        let a = &self.values[j * m..(j + 1) * m];
        let b = &self.values[(j + 1) * m..(j + 2) * m];
        a.iter().zip(b).map(|(&a, &b)| a + lam * (b - a)).collect()
    }

    /// [`Self::values64`] rounded ONCE into `Scalar` (SPEC-LIT 118).
    pub fn host_values(&self, t: f64) -> Vec<Scalar> {
        self.values64(t).into_iter().map(|v| v as Scalar).collect()
    }
}

/// The setup entry this field rank would need.
fn timed_entry(n_t: usize) -> &'static str {
    if n_t == 1 {
        "setup_scalar_field_timed"
    } else {
        "setup_vector_field_timed"
    }
}

/// Build a patch's [`TvMappedRule`] from its entries and the case's
/// `constant/boundaryData/<patch>` (SPEC-LIT 120.11). `bc_f` holds the
/// patch's WORLD face centres in f64; `n_t` is 1 or 3. Refused by name: a
/// setup path that does not advance the clock, a run with no case directory,
/// the entries 120.11 lists, and a database that does not cover the run.
#[allow(clippy::too_many_arguments)]
pub fn build_rule(
    spec: &PatchFieldSpec,
    start: usize,
    patch: &str,
    field: &str,
    n_t: usize,
    bc_f: &[DVec3],
    f1: Option<&F1Setup>,
) -> Result<TvMappedRule> {
    let Some(fs) = f1 else {
        return Err(config(
            field,
            patch,
            format!(
                "this setup path does not advance time-varying boundary conditions each \
                 time step, so the patch would silently run at its first time folder \
                 forever - call {}, which evaluates the patch at t_start and returns \
                 the TimedBcs to apply after each F1Clock::advance",
                timed_entry(n_t)
            ),
        ));
    };
    let Some(case_dir) = fs.case_dir else {
        return Err(config(
            field,
            patch,
            "the run carries no case directory (F1Setup::case_dir), and the inflow \
             database is read from <case>/constant/boundaryData/<patch>",
        ));
    };
    let base = format!("{field}: boundaryField/{patch}");
    let entries = crate::function1::patch_entries(spec, &base)?;
    let word = |k: &str| entries.get(k).map(|v| v.trim().trim_end_matches(';').trim().to_string());

    let method = match word("mapMethod").as_deref() {
        None | Some("planarInterpolation") => MapMethod::Planar,
        Some("nearest") => MapMethod::Nearest,
        Some(other) => {
            return Err(config(
                field,
                patch,
                format!(
                    "{base}/mapMethod: `{other}` is not built - `planarInterpolation` \
                     or `nearest`"
                ),
            ))
        }
    };
    match word("setAverage").as_deref() {
        None | Some("false") | Some("off") | Some("no") | Some("0") => {}
        Some(v) => {
            return Err(config(
                field,
                patch,
                format!(
                    "{base}/setAverage: `{v}` is not built - the interpolated values are \
                     used as they are, and the files' averages are read past"
                ),
            ))
        }
    }
    if entries.contains_key("offset") || entries.keys().any(|k| k.starts_with("offset/")) {
        return Err(config(
            field,
            patch,
            format!("{base}/offset: an offset added to the mapped value is not built"),
        ));
    }
    if let Some(v) = word("perturb") {
        if v.parse::<f64>().is_err() {
            return Err(config(field, patch, format!("{base}/perturb: `{v}` is not a number")));
        }
        // Accepted and unused: (120.11b) does not perturb the samples.
    }
    let table = word("fieldTable").unwrap_or_else(|| field.to_string());

    let dir = case_dir.join("constant").join("boundaryData").join(patch);
    // A reader's failure is re-raised naming the condition and the patch.
    let wrap = |e: Error| config(field, patch, e.to_string());
    let points = crate::io::boundary_data::read_points(&dir.join("points")).map_err(wrap)?;
    let folders = crate::io::boundary_data::time_folders(&dir).map_err(wrap)?;
    let w = weights(&points, bc_f, method, field, patch)?;

    let times: Vec<f64> = folders.iter().map(|(t, _)| *t).collect();
    if times.len() >= 2 && (fs.t_start < times[0] || fs.t_end > times[times.len() - 1]) {
        return Err(config(
            field,
            patch,
            format!(
                "the run [{}, {}] is not inside the database's times [{}, {}] in {}",
                fs.t_start,
                fs.t_end,
                times[0],
                times[times.len() - 1],
                dir.display()
            ),
        ));
    }
    let mut values = Vec::with_capacity(times.len() * bc_f.len() * n_t);
    for (_, f) in &folders {
        let path = f.join(&table);
        let data = crate::io::boundary_data::read_values(&path, n_t).map_err(wrap)?;
        if data.len() != points.len() * n_t {
            return Err(config(
                field,
                patch,
                format!(
                    "{} holds {} values and `points` {} samples",
                    path.display(),
                    data.len() / n_t,
                    points.len()
                ),
            ));
        }
        values.extend(apply_weights(&w, &data, n_t));
    }
    Ok(TvMappedRule {
        patch: patch.to_string(),
        start,
        n: bc_f.len(),
        n_t,
        times,
        values,
    })
}
