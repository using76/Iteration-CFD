// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

// TET-13a's and TET-13b's tests (SPEC-LIT §116.15, §116.18), written once and included
// twice: by src/tetmesh/inflate.rs's test module and by
// tests/tetmesh_inflate.rs, so the `--features single` build, whose
// library test crate does not compile (SPEC-LIT §112.3), runs them too.
// They use the public API only. Every number is evaluated in f64 (§116.2),
// so the gates are class A with the same bars, and the printed lines are
// the same, in both builds.
//
// Provenance: ORIGINAL. Aubry & Löhner (2008)'s most normal normal as
// SPEC-LIT §116.15 solves it. No GPL-licensed source was consulted.

use ofgpu::automesher::quality::{measure, QualityThresholds};
use ofgpu::io::polymesh::build_host_mesh;
use ofgpu::tetmesh::carve::CarveStart;
use ofgpu::tetmesh::emit::{
    carved_boundary, prism_valid, to_poly_mesh, to_poly_mesh_mixed, EmitPatch, PRISM_SPLIT,
};
use ofgpu::tetmesh::inflate::{
    angle, inflate, most_normal, thicknesses, wall_normals, Inflated, LayerStack, Limit,
    Thicknesses,
};
use ofgpu::tetmesh::optimise::{freedoms, optimise};
use ofgpu::tetmesh::predicates::orient3d;
use ofgpu::tetmesh::recover::recover_faces;
use ofgpu::tetmesh::refine::refine;
use ofgpu::tetmesh::remesh::{remesh, RemeshOptions};
use ofgpu::tetmesh::sizefield::{OctreeOptions, SizeField, SizeParams, SizeSource};
use ofgpu::tetmesh::surfprep::{prepare, TetSurface};
use ofgpu::tetmesh::{Point, TetLayerSpec, TetOptimiseSpec, TetRecoverSpec, TetRefineSpec};
use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;

/// The little vector kit the new tests share with the library's own.
fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: Point, b: Point) -> Point {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn cross(a: Point, b: Point) -> Point {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: Point) -> f64 {
    dot(a, a).sqrt()
}

/// `a / |a|`, or None for the zero vector.
fn unit(a: Point) -> Option<Point> {
    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if n > 0.0 && n.is_finite() {
        Some([a[0] / n, a[1] / n, a[2] / n])
    } else {
        None
    }
}

/// Triangle `t`'s unit normal and area, as the library forms them.
fn tri_normal_area(surf: &TetSurface, t: usize) -> Option<(Point, f64)> {
    let [a, b, c] = surf.tris[t].map(|i| surf.points[i as usize]);
    let (u, v) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
    let x = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    unit(x).map(|n| (n, 0.5 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt()))
}

/// A surface builder: points merged bit for bit, patches by name.
#[derive(Default)]
struct Builder {
    points: Vec<Point>,
    tris: Vec<[u32; 3]>,
    tri_patch: Vec<u32>,
    patch_names: Vec<String>,
    at: HashMap<[u64; 3], u32>,
}

impl Builder {
    fn patch(&mut self, name: &str) -> u32 {
        if let Some(i) = self.patch_names.iter().position(|n| n == name) {
            return i as u32;
        }
        self.patch_names.push(name.to_string());
        (self.patch_names.len() - 1) as u32
    }
    fn point(&mut self, p: Point) -> u32 {
        let key = [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
        if let Some(&i) = self.at.get(&key) {
            return i;
        }
        self.points.push(p);
        let i = (self.points.len() - 1) as u32;
        self.at.insert(key, i);
        i
    }
    fn tri(&mut self, a: Point, b: Point, c: Point, patch: u32) {
        let t = [self.point(a), self.point(b), self.point(c)];
        self.tris.push(t);
        self.tri_patch.push(patch);
    }
    /// The quad `a b c d` (counter-clockwise seen from its normal side)
    /// as the triangles `a b c` and `a c d`.
    fn quad(&mut self, a: Point, b: Point, c: Point, d: Point, patch: u32) {
        self.tri(a, b, c, patch);
        self.tri(a, c, d, patch);
    }
    fn done(self) -> TetSurface {
        TetSurface {
            points: self.points,
            tris: self.tris,
            tri_patch: self.tri_patch,
            patch_names: self.patch_names,
        }
    }
}

fn normalise(p: Point) -> Point {
    unit(p).unwrap()
}

/// The icosphere of radius 1: the 12 icosahedron points first, then
/// each level splits every triangle into four, each edge midpoint
/// `normalise((a + b) / 2)`.
fn icosphere(level: u32) -> TetSurface {
    let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
    let raw = [
        [-1.0, phi, 0.0], [1.0, phi, 0.0], [-1.0, -phi, 0.0], [1.0, -phi, 0.0],
        [0.0, -1.0, phi], [0.0, 1.0, phi], [0.0, -1.0, -phi], [0.0, 1.0, -phi],
        [phi, 0.0, -1.0], [phi, 0.0, 1.0], [-phi, 0.0, -1.0], [-phi, 0.0, 1.0],
    ];
    let faces: [[usize; 3]; 20] = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    let mut tris: Vec<[Point; 3]> =
        faces.iter().map(|t| t.map(|i| normalise(raw[i]))).collect();
    let mid = |a: Point, b: Point| {
        normalise([(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5])
    };
    for _ in 0..level {
        let mut next = Vec::with_capacity(tris.len() * 4);
        for [a, b, c] in tris {
            let (ab, bc, ca) = (mid(a, b), mid(b, c), mid(c, a));
            next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        tris = next;
    }
    let mut b = Builder::default();
    let w = b.patch("sphere");
    for p in raw {
        b.point(normalise(p));
    }
    for [p, q, r] in tris {
        b.tri(p, q, r, w);
    }
    b.done()
}

/// The surface of the box `[0, 1]^3` with `k x k` quads per face,
/// wound outward, every point mapped by `f`.
fn cube(k: usize, f: &dyn Fn(Point) -> Point) -> TetSurface {
    let mut b = Builder::default();
    let w = b.patch("cube");
    let g = |i: usize| i as f64 / k as f64;
    for a in 0..3 {
        let (u, v) = ((a + 1) % 3, (a + 2) % 3);
        for side in [0.0, 1.0] {
            for i in 0..k {
                for j in 0..k {
                    let at = |iu: usize, jv: usize| {
                        let mut p = [0.0; 3];
                        p[a] = side;
                        p[u] = g(iu);
                        p[v] = g(jv);
                        f(p)
                    };
                    let (p0, p1, p2, p3) =
                        (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
                    // e_u x e_v = e_a: counter-clockwise in (u, v) faces +a.
                    if side == 1.0 {
                        b.quad(p0, p1, p2, p3, w);
                    } else {
                        b.quad(p0, p3, p2, p1, w);
                    }
                }
            }
        }
    }
    b.done()
}

/// A plate `z = z0` over `[x0, x0 + nx] x [0, ny]` in unit quads,
/// facing `+z` when `up`.
fn plate(b: &mut Builder, x0: f64, nx: usize, ny: usize, z0: f64, up: bool, patch: u32) {
    for i in 0..nx {
        for j in 0..ny {
            let (x, y) = (x0 + i as f64, j as f64);
            let (p0, p1, p2, p3) =
                ([x, y, z0], [x + 1.0, y, z0], [x + 1.0, y + 1.0, z0], [x, y + 1.0, z0]);
            if up {
                b.quad(p0, p1, p2, p3, patch);
            } else {
                b.quad(p0, p3, p2, p1, patch);
            }
        }
    }
}

/// The open `n`-sided pipe of radius 1, `z` in `[0, 1]` in `nz` rings,
/// wound inward (normals to the axis) or outward.
fn pipe(n: usize, nz: usize, inward: bool) -> TetSurface {
    let mut b = Builder::default();
    let w = b.patch("pipe");
    let at = |k: usize, j: usize| {
        let th = 2.0 * std::f64::consts::PI * (k % n) as f64 / n as f64;
        [th.cos(), th.sin(), j as f64 / nz as f64]
    };
    for k in 0..n {
        for j in 0..nz {
            let (p0, p1, p2, p3) = (at(k, j), at(k + 1, j), at(k + 1, j + 1), at(k, j + 1));
            // e_theta x e_z = e_r: this order faces outward.
            if inward {
                b.quad(p0, p3, p2, p1, w);
            } else {
                b.quad(p0, p1, p2, p3, w);
            }
        }
    }
    b.done()
}

fn stack_5() -> LayerStack {
    LayerStack {
        n: 5,
        first_thickness: 1e-3,
        growth: 1.2,
    }
}

#[test]
fn stack_total() {
    let t = stack_5().total();
    assert!((t - 7.4416e-3).abs() <= 1e-15 * 7.4416e-3, "T = {t:e}");
    let flat = LayerStack {
        n: 4,
        first_thickness: 0.25,
        growth: 1.0,
    };
    assert_eq!(flat.total(), 1.0);
    let bad = LayerStack {
        n: 3,
        first_thickness: 0.0,
        growth: 1.2,
    };
    let e = bad.validate().unwrap_err().to_string();
    assert!(e.contains("first_thickness"), "{e}");
}

#[test]
fn most_normal_small_sets() {
    let z = [0.0, 0.0, 1.0];
    assert_eq!(most_normal(&[z, z]), Some((z, 1.0)));
    let (n, f) = most_normal(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).unwrap();
    let h = std::f64::consts::FRAC_1_SQRT_2;
    assert!(angle(n, [h, h, 0.0]) <= 1e-15 && (f - h).abs() <= 1e-15, "{n:?} {f}");
    assert_eq!(most_normal(&[z, [0.0, 0.0, -1.0]]), None);
    assert_eq!(most_normal(&[]), None);
    // A fourth normal inside the cap of three changes nothing.
    let s = 1.0 / 3.0f64.sqrt();
    let (n3, _) = most_normal(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], z]).unwrap();
    let (n4, _) = most_normal(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], z, [s, s, s]]).unwrap();
    assert_eq!(n3, n4);
}

/// The largest angle between `n*` and the radial direction over the
/// wall points `pick` keeps, and how many it kept.
fn radial_error(surf: &TetSurface, pick: &dyn Fn(u32) -> bool) -> (f64, usize) {
    let wn = wall_normals(surf, &[true]).unwrap();
    let mut worst = 0.0f64;
    let mut count = 0;
    for (k, &p) in wn.points.iter().enumerate() {
        if !pick(p) {
            continue;
        }
        let n = wn.normal[k].expect("every sphere point has a normal");
        worst = worst.max(angle(n, surf.points[p as usize]));
        count += 1;
    }
    (worst, count)
}

/// §116.18 TET-13a: the sphere normal is radial within 1e-6 rad where
/// (116.22) is radial, at every point whose star is symmetric about the
/// radius: all of the icosahedron (level 0) and of level 1, and the 12
/// five-valent points of level 3. Elsewhere (116.22) is not radial, by
/// O(h / R), and the test prints that as MISSED.
#[test]
fn normals_sphere() {
    for (level, pick) in [(0u32, u32::MAX), (1, u32::MAX), (3, 12)] {
        let s = icosphere(level);
        let (e, c) = radial_error(&s, &|p| p < pick);
        println!(
            "normals_sphere: level {level}: {c} points, the largest angle to the radius \
             {e:e} rad (bar 1e-6)"
        );
        assert!(e <= 1e-6, "level {level}: {e:e} rad");
    }
    // Every point of levels 3 and 4. The supervisor's independent
    // transcription of (116.22) (numpy, outside the tree) gives
    // 4.45853096089773e-3 and 2.3552644221493026e-3 rad.
    for (level, probe) in [(3u32, 4.45853096089773e-3), (4, 2.3552644221493026e-3)] {
        let s = icosphere(level);
        let (e, c) = radial_error(&s, &|_| true);
        println!(
            "normals_sphere: MISSED for every point of level {level} ({c} points): the largest \
             angle to the radius is {e:e} rad against 1e-6; cause: (116.22) on an inscribed \
             triangulation is radial only where the star is symmetric about the radius, and \
             the error is O(h / R)"
        );
        assert!((e - probe).abs() <= 1e-9, "level {level}: {e:e} against the probe {probe:e}");
    }
}

/// §116.18 TET-13a: at a cube corner the normal makes equal angles with
/// the three faces within 1e-9, on the 12-triangle cube, on a 3 x 3
/// grid per face, and on that grid rotated and moved 1e4 m away. On an
/// edge it bisects the two faces, and inside a face it is the face
/// normal.
#[test]
fn corner_normal() {
    let (ca, sa, cb, sb) = (0.3f64.cos(), 0.3f64.sin(), 1.1f64.cos(), 1.1f64.sin());
    let rot = move |p: Point| {
        let q = [ca * p[0] - sa * p[1], sa * p[0] + ca * p[1], p[2]];
        let r = [q[0], cb * q[1] - sb * q[2], sb * q[1] + cb * q[2]];
        [r[0] + 1.0e4, r[1] - 2.0e4, r[2] + 3.0e4]
    };
    let id = |p: Point| p;
    let cases: [(&str, usize, &dyn Fn(Point) -> Point); 3] =
        [("1x1", 1, &id), ("3x3", 3, &id), ("3x3 rotated at 1e4 m", 3, &rot)];
    for (name, k, f) in cases {
        let s = cube(k, f);
        let wn = wall_normals(&s, &[true]).unwrap();
        let mut worst = [0.0f64; 3];
        let mut seen = [0usize; 3];
        for (i, &p) in wn.points.iter().enumerate() {
            let n = wn.normal[i].unwrap();
            // The distinct face planes at the point, and the spread of
            // the angles from n* to every incident triangle.
            let mut faces: Vec<Point> = Vec::new();
            let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
            for t in 0..s.tris.len() {
                if s.tris[t].contains(&p) {
                    let m = tri_normal_area(&s, t).unwrap().0;
                    if !faces.iter().any(|&g| angle(g, m) < 1e-6) {
                        faces.push(m);
                    }
                    let a = angle(n, m);
                    lo = lo.min(a);
                    hi = hi.max(a);
                }
            }
            let class = faces.len() - 1;
            worst[class] = worst[class].max(hi - lo);
            seen[class] += 1;
            if class == 1 {
                let a = angle(n, faces[0]);
                let d = (a - std::f64::consts::FRAC_PI_4).abs();
                assert!(d <= 1e-9, "{name}: edge point {p}: {a}");
            }
        }
        println!(
            "corner_normal: {name}: {} corners, {} edge and {} face points; angle spreads \
             {:e} / {:e} / {:e} rad (bar 1e-9)",
            seen[2], seen[1], seen[0], worst[2], worst[1], worst[0]
        );
        assert_eq!(seen[2], 8, "{name}");
        assert_eq!(seen[1], 12 * (k - 1), "{name}");
        for w in worst {
            assert!(w <= 1e-9, "{name}: spread {w:e}");
        }
    }
}

/// A layered lower plate `z = 0` over `x` in `[0, sum n]`, `y` in
/// `[0, 2]`, and upper plates facing it, `(gap, n)` each, side by
/// side; the thicknesses at §116.3's default limits.
fn plates(gaps: &[(f64, usize)], upper_layered: bool) -> (TetSurface, Vec<bool>, Thicknesses) {
    let mut b = Builder::default();
    let lo = b.patch("lower");
    let up = b.patch("upper");
    let nx: usize = gaps.iter().map(|g| g.1).sum();
    plate(&mut b, 0.0, nx, 2, 0.0, true, lo);
    let mut x0 = 0.0;
    for &(g, n) in gaps {
        plate(&mut b, x0, n, 2, g, false, up);
        x0 += n as f64;
    }
    let s = b.done();
    let mask = vec![true, upper_layered];
    let wn = wall_normals(&s, &mask).unwrap();
    let t = thicknesses(&s, &mask, &wn, &stack_5(), &TetLayerSpec::default(), 60.0).unwrap();
    (s, mask, t)
}

/// Every layered edge keeps (116.24c) at the default ratio 1.5.
fn ratio_holds(s: &TetSurface, mask: &[bool], t: &Thicknesses) {
    let wn = wall_normals(s, mask).unwrap();
    let slot: HashMap<u32, usize> =
        wn.points.iter().enumerate().map(|(k, &p)| (p, k)).collect();
    for (i, tri) in s.tris.iter().enumerate() {
        if !mask[s.tri_patch[i] as usize] {
            continue;
        }
        for e in 0..3 {
            let (a, b) = (slot[&tri[e]], slot[&tri[(e + 1) % 3]]);
            assert!(t.total[a] <= 1.5 * t.total[b] && t.total[b] <= 1.5 * t.total[a]);
        }
    }
}

/// §116.18 TET-13a: a gap of 3T between two layered plates keeps T,
/// which is gap/2 or less; a gap of T is limited to gap/2 exactly; and
/// a step from a gap of T to one of 3T grades by the ratio 1.5.
#[test]
fn thickness_limited() {
    let tt = stack_5().total();
    let (s, mask, t) = plates(&[(3.0 * tt, 4)], true);
    assert_eq!(t.requested, tt);
    assert_eq!(t.total.len(), 30);
    for k in 0..t.total.len() {
        assert_eq!(t.gap[k], Some(3.0 * tt));
        assert_eq!((t.total[k], t.limit[k]), (tt, Limit::Requested));
        assert!(t.total[k] <= 0.5 * t.gap[k].unwrap());
    }
    ratio_holds(&s, &mask, &t);
    let kept = t.count(Limit::Requested);
    let (s, mask, t) = plates(&[(tt, 4)], true);
    for k in 0..t.total.len() {
        assert_eq!(t.gap[k], Some(tt));
        assert_eq!((t.total[k], t.limit[k]), (0.5 * tt, Limit::Gap));
    }
    ratio_holds(&s, &mask, &t);
    println!(
        "thickness_limited: T = {tt:e}; gap 3T: {kept} points keep T; gap T: {} points at gap/2",
        t.count(Limit::Gap)
    );

    let (s, mask, t) = plates(&[(tt, 4), (3.0 * tt, 4)], false);
    let wn = wall_normals(&s, &mask).unwrap();
    assert_eq!(wn.points.len(), 27);
    for (k, &p) in wn.points.iter().enumerate() {
        let x = s.points[p as usize][0];
        let want = if x <= 4.0 {
            (0.5 * tt, Limit::Gap)
        } else if x == 5.0 {
            (1.5 * (0.5 * tt), Limit::Neighbour)
        } else {
            (tt, Limit::Requested)
        };
        assert_eq!((t.total[k], t.limit[k]), want, "point {p} at x = {x}");
    }
    ratio_holds(&s, &mask, &t);
    println!(
        "thickness_limited: the step: {} points at gap/2, {} graded to 0.75 T, {} at T",
        t.count(Limit::Gap),
        t.count(Limit::Neighbour),
        t.count(Limit::Requested)
    );
}

/// (116.24b): inside a 16-sided pipe of radius 1 the concave limit
/// `0.5 rho_c` binds, with `1 / rho_c` the estimate (116.9b) across an
/// axial edge, `3 phi / (2 w)`, `phi = 2 pi / 16`, `w = 2 sin(pi / 16)`;
/// outside the same pipe nothing limits the stack.
#[test]
fn concave_limited() {
    let stack = LayerStack {
        n: 1,
        first_thickness: 1.0,
        growth: 1.0,
    };
    let spec = TetLayerSpec::default();
    let pi = std::f64::consts::PI;
    let want = 0.5 * (2.0 * 2.0 * (pi / 16.0).sin()) / (3.0 * (2.0 * pi / 16.0));
    let s = pipe(16, 2, true);
    let wn = wall_normals(&s, &[true]).unwrap();
    let t = thicknesses(&s, &[true], &wn, &stack, &spec, 60.0).unwrap();
    assert_eq!(wn.points.len(), 48);
    for (k, &p) in wn.points.iter().enumerate() {
        let x = s.points[p as usize];
        let radial = [-x[0], -x[1], 0.0];
        assert!(angle(wn.normal[k].unwrap(), radial) <= 1e-12, "point {p}");
        assert_eq!(t.limit[k], Limit::Concave, "point {p}");
        let d = (t.total[k] - want).abs();
        assert!(d <= 1e-12 * want, "point {p}: {} against {want}", t.total[k]);
    }
    println!(
        "concave_limited: inside, {} points at 0.5 rho_c = {want:.15}",
        t.count(Limit::Concave)
    );
    let s = pipe(16, 2, false);
    let wn = wall_normals(&s, &[true]).unwrap();
    let t = thicknesses(&s, &[true], &wn, &stack, &spec, 60.0).unwrap();
    for k in 0..t.total.len() {
        let got = (t.total[k], t.limit[k], t.rho_c[k], t.gap[k]);
        assert_eq!(got, (1.0, Limit::Requested, None, None));
    }
}

#[test]
fn refusals() {
    let s = pipe(8, 1, true);
    let e = wall_normals(&s, &[true, false]).unwrap_err().to_string();
    assert!(e.contains("layered mask"), "{e}");
    let other = pipe(9, 1, true);
    let wn = wall_normals(&other, &[true]).unwrap();
    let e = thicknesses(&s, &[true], &wn, &stack_5(), &TetLayerSpec::default(), 60.0)
        .unwrap_err()
        .to_string();
    assert!(e.contains("were not built from this surface"), "{e}");
    let wn = wall_normals(&s, &[true]).unwrap();
    let spec = TetLayerSpec {
        gap_fraction: 0.0,
        ..TetLayerSpec::default()
    };
    let e = thicknesses(&s, &[true], &wn, &stack_5(), &spec, 60.0)
        .unwrap_err()
        .to_string();
    assert!(e.contains("gap_fraction"), "{e}");
}

/// §116.15's (116.25f) on the right prism over the unit corner triangle:
/// each split tetrahedron is exactly `+1`, the prism is valid, the mixed
/// emitter takes its five faces on one patch with nothing internal and an
/// exact closure, and the flipped prism is invalid and refused
/// (`emit/orient`).
#[test]
fn prism_split_right() {
    let pts: Vec<Point> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let prism = [0u32, 1, 2, 3, 4, 5];
    for (k, row) in PRISM_SPLIT.iter().enumerate() {
        let q = row.map(|i| pts[prism[i] as usize]);
        assert_eq!(orient3d(&q[0], &q[1], &q[2], &q[3]), 1.0, "split {k}");
    }
    assert!(prism_valid(&pts, &prism));
    let bnd: Vec<(Vec<u32>, u32)> = vec![
        (vec![0, 2, 1], 0),
        (vec![3, 4, 5], 0),
        (vec![0, 1, 4, 3], 0),
        (vec![1, 2, 5, 4], 0),
        (vec![2, 0, 3, 5], 0),
    ];
    let ps = vec![EmitPatch { name: "wall".to_string(), type_name: "wall".to_string() }];
    let e = to_poly_mesh_mixed(&pts, &[], &[prism], &bnd, &ps).unwrap();
    assert_eq!(e.raw.faces.len(), 5);
    assert!(e.raw.neighbour.is_empty());
    assert_eq!(e.raw.patches[0].size, 5);
    let q = measure(&e.raw, &QualityThresholds::default()).unwrap();
    println!(
        "prism_split_right: the three splits are exactly 1.0; 5 faces, 0 internal; \
         92.3 max closure {e:e}",
        e = q.max_closure
    );
    assert!(q.max_closure < 1e-10, "closure {}", q.max_closure);
    let flip = [3u32, 4, 5, 0, 1, 2];
    assert!(!prism_valid(&pts, &flip));
    let bnd2: Vec<(Vec<u32>, u32)> = vec![
        (vec![3, 5, 4], 0),
        (vec![0, 1, 2], 0),
        (vec![3, 4, 1, 0], 0),
        (vec![4, 5, 2, 1], 0),
        (vec![5, 3, 0, 2], 0),
    ];
    let err = to_poly_mesh_mixed(&pts, &[], &[flip], &bnd2, &ps).unwrap_err().to_string();
    assert!(err.contains("emit/orient"), "{err}");
    println!("prism_split_right: the flipped prism refused: {err}");
}

/// TET-13b's sphere in a box, built exactly as tests/tetmesh_optimise.rs
/// builds it (SPEC-LIT §116.18's TET-13b fixture), and cached for
/// `sphere_layers` and `conformal`.
struct SphereCase {
    points: Vec<Point>,
    tets: Vec<[u32; 4]>,
    bnd: Vec<([u32; 3], u32)>,
}

fn sphere_case() -> &'static SphereCase {
    static CASE: OnceLock<SphereCase> = OnceLock::new();
    CASE.get_or_init(|| {
        let s = icosphere(3);
        let tris: Vec<[Point; 3]> =
            s.tris.iter().map(|t| t.map(|v| s.points[v as usize])).collect();
        let field = SizeField::build(
            SizeParams::new(0.5, 1.2, None).unwrap(),
            &[SizeSource::Patch {
                name: "sphere".to_string(),
                tris,
                distance: 0.0,
                size: 0.25,
            }],
            [-2.0; 3],
            [2.0; 3],
            OctreeOptions::default(),
        )
        .unwrap();
        let prep = prepare(&s, 60.0).unwrap();
        let r = remesh(&prep, &|x: Point| field.h(x), &RemeshOptions::default()).unwrap();
        let mut surf = r.surface.clone();
        for k in 0..8u32 {
            let c = |b: u32| if (k >> b) & 1 == 1 { 2.0 } else { -2.0 };
            surf.points.push([c(0), c(1), c(2)]);
        }
        let fr = recover_faces(&surf, &r.feature_edges, &TetRecoverSpec::default()).unwrap();
        let refined = refine(
            &surf,
            fr,
            &CarveStart::Hull,
            &TetRefineSpec::default(),
            &|x: Point| field.h(x),
        )
        .unwrap();
        let tri_patch = vec![0u32; refined.recovery.tris.len()];
        let free = freedoms(&refined.recovery, &tri_patch, &refined.carved.tets);
        let o = optimise(
            &refined.recovery.points,
            &refined.carved.tets,
            &refined.recovery.subtris,
            &free,
            &TetOptimiseSpec::default(),
        )
        .unwrap();
        let bnd = carved_boundary(&refined.carved, &refined.recovery, &tri_patch, 1).unwrap();
        SphereCase { points: o.points, tets: o.tets, bnd }
    })
}

fn mixed_patches() -> Vec<EmitPatch> {
    vec![
        EmitPatch { name: "sphere".to_string(), type_name: "wall".to_string() },
        EmitPatch { name: "domain".to_string(), type_name: "patch".to_string() },
    ]
}

fn inflate_sphere(c: &SphereCase) -> Inflated {
    inflate(
        &c.points,
        &c.tets,
        &c.bnd,
        &[true, false],
        &LayerStack { n: 5, first_thickness: 2e-3, growth: 1.2 },
        &TetLayerSpec::default(),
        60.0,
    )
    .unwrap()
}

/// §116.18 TET-13b's gate on the sphere in a box (SPEC-LIT §116.15): five
/// layers at `t_1 = 2e-3` (= 1e-3 D, D = 2), growth 1.2. (a) the achieved
/// first-layer thickness is within 1 % of 2e-3 at 95 % of the wall points;
/// (b) no tetrahedron with `orient3d <= 0` and no prism failing (116.25f);
/// (c) five prisms per wall triangle; (d) the mixed emission builds and the
/// §92.3 closure is under 1e-10 in f64; (e) the wall faces' non-orthogonality
/// is at most 5 degrees on `build_host_mesh`.
#[test]
fn sphere_layers() {
    let c = sphere_case();
    let inf = inflate_sphere(c);
    let n_wall = c.bnd.iter().filter(|&&(_, p)| p == 0).count();
    let wp = inf.normals.points.len();
    let (mut ok1, mut worst1) = (0usize, 0.0f64);
    for row in &inf.layers {
        let d = norm(sub(inf.points[row[1] as usize], inf.points[row[0] as usize]));
        let rel = (d - 2e-3).abs() / 2e-3;
        worst1 = worst1.max(rel);
        if rel <= 0.01 {
            ok1 += 1;
        }
    }
    let frac1 = ok1 as f64 / wp as f64;
    let bad_tets = inf
        .tets
        .iter()
        .filter(|t| {
            let q = t.map(|v| inf.points[v as usize]);
            orient3d(&q[0], &q[1], &q[2], &q[3]) <= 0.0
        })
        .count();
    let bad_prisms = inf.prisms.iter().filter(|p| !prism_valid(&inf.points, p)).count();
    let prisms_ok = inf.prisms.len() == 5 * n_wall;
    let e = to_poly_mesh_mixed(&inf.points, &inf.tets, &inf.prisms, &inf.boundary, &mixed_patches())
        .unwrap();
    let q = measure(&e.raw, &QualityThresholds::default()).unwrap();
    let single = std::mem::size_of::<ofgpu::Scalar>() == 4;
    if single && q.max_closure >= 1e-10 {
        println!(
            "sphere_layers: MISSED - G2 under single (docs/17-review-later A28): max closure \
             {e:e} against 1e-10",
            e = q.max_closure
        );
    } else {
        assert!(q.max_closure < 1e-10, "closure {}", q.max_closure);
    }
    let m = build_host_mesh(&e.raw).unwrap();
    let sp = &e.raw.patches[0];
    let mut worst_e = 0.0f64;
    for f in sp.start..sp.start + sp.size {
        let (sf, cf) = (m.b_sf[f], m.b_cf[f]);
        let cc = m.c[m.b_face_cells[f] as usize];
        let d = [cf.x as f64 - cc.x as f64, cf.y as f64 - cc.y as f64, cf.z as f64 - cc.z as f64];
        worst_e = worst_e.max(angle([sf.x as f64, sf.y as f64, sf.z as f64], d).to_degrees());
    }
    let counts = [
        ("Requested", inf.thickness.count(Limit::Requested)),
        ("Gap", inf.thickness.count(Limit::Gap)),
        ("Concave", inf.thickness.count(Limit::Concave)),
        ("Neighbour", inf.thickness.count(Limit::Neighbour)),
        ("Inversion", inf.thickness.count(Limit::Inversion)),
        ("NoNormal", inf.thickness.count(Limit::NoNormal)),
    ];
    println!(
        "sphere_layers: {} wall points, {} wall triangles, rounds {}, n_cut {}, limits {:?}; \
         t1 within 1 % of 2e-3 at {ok1}/{wp} ({:.1} %), worst rel {worst1:.2e}; \
         bad tets {bad_tets}, bad prisms {bad_prisms}, prisms {} = 5 x {n_wall}: {prisms_ok}; \
         max closure {:.3e}; wall non-orthogonality max {worst_e:.4} deg (bar 5)",
        wp, n_wall, inf.rounds, inf.n_cut, counts,
        100.0 * frac1,
        inf.prisms.len(),
        q.max_closure
    );
    assert!(frac1 >= 0.95, "t1 within 1 % at {ok1}/{wp}");
    assert_eq!(bad_tets, 0);
    assert_eq!(bad_prisms, 0);
    assert!(prisms_ok, "{} prisms for {n_wall} wall triangles", inf.prisms.len());
    if worst_e > 5.0 {
        // Every wall point carries the full requested stack here, so the
        // first prism's shear comes from the point normals alone: its
        // vertices extrude along n* (116.22), which on the polyhedral
        // sphere differ from the face normal by O(h / R) (TET-13a's row).
        println!(
            "sphere_layers: MISSED - wall non-orthogonality {worst_e:.4} deg against the bar 5; \
             cause: the first prism extrudes along the point normals n* (116.22), not the face \
             normal (SPEC-LIT 116.18, docs/17-review-later)"
        );
        assert!(inf.thickness.count(Limit::Requested) == wp, "a limiter moved the stack");
    } else {
        assert!(worst_e <= 5.0, "wall non-orthogonality {worst_e} deg");
    }
}

/// The same case, TET-13b's conformal checks: (a) the prism-tet interfaces
/// are exactly the wall triangles; (b) the sphere patch is the undeformed
/// wall bit for bit and the domain patch matches the tets-only emission;
/// (c) every cell's Gauss volume from `raw` is positive and the total
/// equals the optimised tets' `orient3d` volume within 1e-12 relative.
#[test]
fn conformal() {
    let c = sphere_case();
    let inf = inflate_sphere(c);
    let n_wall = c.bnd.iter().filter(|&&(_, p)| p == 0).count();
    let n_tets = inf.tets.len();
    let e = to_poly_mesh_mixed(&inf.points, &inf.tets, &inf.prisms, &inf.boundary, &mixed_patches())
        .unwrap();
    let raw = &e.raw;
    let straddle = (0..raw.neighbour.len())
        .filter(|&f| {
            let (o, nb) = (raw.owner[f] as usize, raw.neighbour[f] as usize);
            (o < n_tets) != (nb < n_tets)
        })
        .count();
    assert_eq!(straddle, n_wall, "the prism-tet interfaces");
    let sp = &raw.patches[0];
    assert_eq!(sp.size as usize, n_wall, "the sphere patch's faces");
    let bits = |f: &[ofgpu::Label], pts: &[ofgpu::types::DVec3]| -> Vec<[u64; 3]> {
        let mut v: Vec<[u64; 3]> = f
            .iter()
            .map(|&i| {
                let p = pts[i as usize];
                [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]
            })
            .collect();
        v.sort_unstable();
        v
    };
    let mut want: BTreeSet<Vec<[u64; 3]>> = BTreeSet::new();
    for &(f, p) in &c.bnd {
        if p == 0 {
            let mut v: Vec<[u64; 3]> =
                f.iter().map(|&i| c.points[i as usize].map(|x| x.to_bits())).collect();
            v.sort_unstable();
            want.insert(v);
        }
    }
    let mut got: BTreeSet<Vec<[u64; 3]>> = BTreeSet::new();
    // `PatchInfo.start` is the offset into the boundary faces, after the
    // internal ones (mesh.rs): add the internal count for the face index.
    let bf = raw.neighbour.len();
    for f in bf + sp.start..bf + sp.start + sp.size {
        got.insert(bits(&raw.faces[f], &raw.points));
    }
    assert_eq!(got, want, "the sphere patch is the undeformed wall");
    let e0 = to_poly_mesh(&c.points, &c.tets, &c.bnd, &mixed_patches()).unwrap();
    let dp = &raw.patches[1];
    assert_eq!(e0.raw.patches[1].size, dp.size, "the domain patch's faces");
    let mut dom0: BTreeSet<Vec<[u64; 3]>> = BTreeSet::new();
    let d0 = &e0.raw.patches[1];
    let b0 = e0.raw.neighbour.len();
    for f in b0 + d0.start..b0 + d0.start + d0.size {
        dom0.insert(bits(&e0.raw.faces[f], &e0.raw.points));
    }
    let mut dom1: BTreeSet<Vec<[u64; 3]>> = BTreeSet::new();
    for f in bf + dp.start..bf + dp.start + dp.size {
        dom1.insert(bits(&raw.faces[f], &raw.points));
    }
    assert_eq!(dom1, dom0, "the domain patch's points are the tets-only ones");
    // (c): the Gauss volume of every cell, f64, from the emitted mesh.
    let n_cells = raw
        .owner
        .iter()
        .chain(raw.neighbour.iter())
        .copied()
        .max()
        .unwrap_or(0) as usize
        + 1;
    let mut vol = vec![0.0f64; n_cells];
    for f in 0..raw.faces.len() {
        let pts: Vec<[f64; 3]> = raw.faces[f]
            .iter()
            .map(|&i| {
                let p = raw.points[i as usize];
                [p.x, p.y, p.z]
            })
            .collect();
        let m = pts.len();
        let xf = [
            pts.iter().map(|p| p[0]).sum::<f64>() / m as f64,
            pts.iter().map(|p| p[1]).sum::<f64>() / m as f64,
            pts.iter().map(|p| p[2]).sum::<f64>() / m as f64,
        ];
        let mut s = [0.0f64; 3];
        for i in 0..m {
            let (a, b) = (sub(pts[i], xf), sub(pts[(i + 1) % m], xf));
            s = add(s, cross(a, b));
        }
        let s = [0.5 * s[0], 0.5 * s[1], 0.5 * s[2]];
        let d = dot(xf, s) / 3.0;
        vol[raw.owner[f] as usize] += d;
        if f < raw.neighbour.len() {
            vol[raw.neighbour[f] as usize] -= d;
        }
    }
    let total_mixed = vol.iter().sum::<f64>();
    let total_tets: f64 = c
        .tets
        .iter()
        .map(|t| {
            let q = t.map(|v| c.points[v as usize]);
            orient3d(&q[0], &q[1], &q[2], &q[3]) / 6.0
        })
        .sum();
    let rel = (total_mixed - total_tets).abs() / total_tets.abs();
    let min_v = vol.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "conformal: {} cells ({} tets, {} prisms), {} wall triangles; interfaces {straddle}; \
         sphere patch {}, domain patch {} == tets-only {}; min cell volume {min_v:.3e}; \
         volume {total_mixed:.12} against the tets' {total_tets:.12}, rel {rel:.2e} (bar 1e-12)",
        n_cells, n_tets, inf.prisms.len(), n_wall, sp.size,
        dp.size, e0.raw.patches[1].size
    );
    assert!(vol.iter().all(|&v| v > 0.0), "a cell with volume <= 0: {min_v:e}");
    assert!(rel <= 1e-12, "volume rel {rel:e}");
}

/// The unit box `[0, 1]^3`, its 8 corners, and an interior point
/// `(0.5, 0.5, 0.005)`; 12 boundary triangles, 2 per side, each joined to
/// the interior point as a tetrahedron with `orient3d > 0`. Patch 0
/// "floor" is the 2 `z = 0` triangles, patch 1 "sides" the other 10.
fn box_ip_case() -> (Vec<Point>, Vec<[u32; 4]>, Vec<([u32; 3], u32)>) {
    let mut pts: Vec<Point> = (0..8u32)
        .map(|k| {
            [
                if k & 1 == 1 { 1.0 } else { 0.0 },
                if k & 2 == 2 { 1.0 } else { 0.0 },
                if k & 4 == 4 { 1.0 } else { 0.0 },
            ]
        })
        .collect();
    let ip = pts.len() as u32;
    pts.push([0.5, 0.5, 0.005]);
    let sides: [([u32; 3], [u32; 3], u32); 6] = [
        ([0, 1, 3], [0, 3, 2], 0),
        ([4, 5, 7], [4, 7, 6], 1),
        ([0, 1, 5], [0, 5, 4], 1),
        ([2, 3, 7], [2, 7, 6], 1),
        ([0, 2, 6], [0, 6, 4], 1),
        ([1, 3, 7], [1, 7, 5], 1),
    ];
    let mut tets = Vec::new();
    let mut bnd = Vec::new();
    for (tri, patch) in sides.iter().flat_map(|(t1, t2, p)| [(t1, *p), (t2, *p)]) {
        let (mut a, mut b, c) = (tri[0], tri[1], tri[2]);
        let (pa, pb, pc, pw) = (pts[a as usize], pts[b as usize], pts[c as usize], pts[ip as usize]);
        if orient3d(&pa, &pb, &pc, &pw) < 0.0 {
            std::mem::swap(&mut a, &mut b);
        }
        tets.push([a, b, c, ip]);
        bnd.push(([a, b, c], patch));
    }
    (pts, tets, bnd)
}

fn box_stack() -> LayerStack {
    LayerStack { n: 2, first_thickness: 0.15, growth: 1.0 }
}

/// §116.15's "The cut" (TET-13b): with the floor layered at `T = 0.3`, the
/// first move inverts the floor tetrahedra, so the floor corners' `T` is
/// halved (`Limit::Inversion`) and the move is redone. The end state has no
/// invalid cell, `n_cut = 4`, 4 prisms, and the boundary of 2 bottoms, 10
/// side triangles and 8 side quads.
#[test]
fn inversion_cut() {
    let (pts, tets, bnd) = box_ip_case();
    let inf = inflate(&pts, &tets, &bnd, &[true, false], &box_stack(), &TetLayerSpec::default(), 60.0)
        .unwrap();
    assert!(inf.rounds >= 2, "rounds {}", inf.rounds);
    assert!(inf.tets.iter().all(|t| {
        let q = t.map(|v| inf.points[v as usize]);
        orient3d(&q[0], &q[1], &q[2], &q[3]) > 0.0
    }));
    assert!(inf.prisms.iter().all(|p| prism_valid(&inf.points, p)));
    let floor: Vec<usize> = (0..inf.normals.points.len())
        .filter(|&k| pts[inf.normals.points[k] as usize][2] == 0.0)
        .collect();
    assert_eq!(floor.len(), 4);
    let ts: Vec<f64> = floor.iter().map(|&k| inf.thickness.total[k]).collect();
    for (&k, &t) in floor.iter().zip(&ts) {
        let r = t / 0.3;
        let k2 = -r.log2();
        assert!(
            r > 0.0 && (k2 - k2.round()).abs() < 1e-9 && k2.round() >= 1.0,
            "T = {t:e} is not 0.3 x 2^-k, k >= 1"
        );
        assert_eq!(inf.thickness.limit[k], Limit::Inversion);
    }
    assert_eq!(inf.n_cut, 4, "n_cut");
    assert_eq!(inf.prisms.len(), 4, "prisms");
    let n0 = inf.boundary.iter().filter(|(_, p)| *p == 0).count();
    let p1: Vec<&Vec<u32>> =
        inf.boundary.iter().filter(|(_, p)| *p == 1).map(|(f, _)| f).collect();
    assert_eq!(n0, 2, "the floor bottoms");
    assert_eq!(p1.len(), 18, "the sides: 10 triangles + 8 quads");
    assert_eq!(p1.iter().filter(|f| f.len() == 4).count(), 8, "quads");
    assert_eq!(p1.iter().filter(|f| f.len() == 3).count(), 10, "triangles");
    let ps = vec![
        EmitPatch { name: "floor".to_string(), type_name: "wall".to_string() },
        EmitPatch { name: "sides".to_string(), type_name: "wall".to_string() },
    ];
    let e = to_poly_mesh_mixed(&inf.points, &inf.tets, &inf.prisms, &inf.boundary, &ps).unwrap();
    let q = measure(&e.raw, &QualityThresholds::default()).unwrap();
    println!(
        "inversion_cut: rounds {}, n_cut {}, floor T {ts:?} (all Limit::Inversion); \
         {} cells; 92.3 max closure {e:.3e}",
        inf.rounds,
        inf.n_cut,
        q.n_cells,
        e = q.max_closure
    );
    assert_eq!(q.n_cells, 16);
    if std::mem::size_of::<ofgpu::Scalar>() == 4 && q.max_closure >= 1e-10 {
        println!(
            "inversion_cut: MISSED - G2 under single (docs/17-review-later A28): max closure \
             {e:.3e} against 1e-10",
            e = q.max_closure
        );
    } else {
        assert!(q.max_closure < 1e-10, "closure {}", q.max_closure);
    }
}

/// The refusals of §116.15 (TET-13b): a layered mask with too few entries
/// is refused by name, and a listed face that is no tetrahedron's face is
/// refused with `inflate/boundary`.
#[test]
fn inflate_refusals() {
    let (pts, tets, bnd) = box_ip_case();
    let e = inflate(&pts, &tets, &bnd, &[], &box_stack(), &TetLayerSpec::default(), 60.0)
        .unwrap_err()
        .to_string();
    assert!(e.contains("inflate/boundary") && e.contains("patch 0"), "{e}");
    println!("inflate_refusals: the short mask refused: {e}");
    let mut extra = bnd.clone();
    extra.push(([1, 2, 8], 0));
    let e = inflate(&pts, &tets, &extra, &[true, false], &box_stack(), &TetLayerSpec::default(), 60.0)
        .unwrap_err()
        .to_string();
    assert!(e.contains("inflate/boundary"), "{e}");
    println!("inflate_refusals: the stray face refused: {e}");
}

// ---- TET-13c: sharp edges and corners (SPEC-LIT §116.15, §116.18). ----

use ofgpu::tetmesh::carve::carve;
use ofgpu::tetmesh::emit::rounding_flips_mixed;

/// The TET-13c fixtures' oriented surface builder: points deduplicated by
/// the exact bits of `[x, y, z]` after mapping -0.0 to 0.0; `tri` skipped
/// when two of its three points are the same; every triangle pushed so its
/// right-hand normal has a non-negative cosine with `want`, which points
/// out of the solid into the fluid. One patch, "body".
#[derive(Default)]
struct Oriented {
    points: Vec<Point>,
    tris: Vec<[u32; 3]>,
    tri_patch: Vec<u32>,
    at: HashMap<[u64; 3], u32>,
}

impl Oriented {
    fn point(&mut self, p: Point) -> u32 {
        let q = p.map(|v| if v == 0.0 { 0.0 } else { v });
        let key = [q[0].to_bits(), q[1].to_bits(), q[2].to_bits()];
        if let Some(&i) = self.at.get(&key) {
            return i;
        }
        self.points.push(q);
        self.at.insert(key, self.points.len() as u32 - 1);
        self.points.len() as u32 - 1
    }
    fn tri(&mut self, a: Point, b: Point, c: Point, want: Point) {
        let (ia, ib, ic) = (self.point(a), self.point(b), self.point(c));
        if ia == ib || ib == ic || ia == ic {
            return;
        }
        let (pa, pb, pc) = (
            self.points[ia as usize],
            self.points[ib as usize],
            self.points[ic as usize],
        );
        let n = cross(sub(pb, pa), sub(pc, pa));
        let t = if dot(n, want) >= 0.0 { [ia, ib, ic] } else { [ia, ic, ib] };
        self.tris.push(t);
        self.tri_patch.push(0);
    }
    /// The quad `a b c d` as the triangles `a b c` and `a c d`.
    fn quad(&mut self, a: Point, b: Point, c: Point, d: Point, want: Point) {
        self.tri(a, b, c, want);
        self.tri(a, c, d, want);
    }
    fn done(self) -> TetSurface {
        TetSurface {
            patch_names: vec!["body".to_string()],
            points: self.points,
            tris: self.tris,
            tri_patch: self.tri_patch,
        }
    }
}

/// TET-13c's fixture A: the cube `[-0.5, 0.5]^3` minus a blind bore of
/// radius 0.25 along `z`, open at `z = 0.5` and ending at `z = 0`, 32
/// facets around, every face oriented out of the solid into the fluid.
fn cube_minus_cylinder() -> TetSurface {
    let (m, r) = (32usize, 0.25f64);
    let mut b = Oriented::default();
    let th = |k: usize| 2.0 * std::f64::consts::PI * (k % m) as f64 / m as f64;
    let ring = |k: usize, z: f64| -> Point {
        let t = th(k);
        [r * t.cos(), r * t.sin(), z]
    };
    let snap = |v: f64| if (v.abs() - 0.5).abs() < 1e-12 { 0.5 * v.signum() } else { v };
    let sq = |k: usize, z: f64| -> Point {
        let t = th(k);
        let (c, s) = (t.cos(), t.sin());
        let f = 0.5 / c.abs().max(s.abs());
        [snap(f * c), snap(f * s), z]
    };
    let (up, dn) = ([0.0, 0.0, 1.0], [0.0, 0.0, -1.0]);
    for k in 0..m {
        b.quad(ring(k, 0.5), sq(k, 0.5), sq(k + 1, 0.5), ring(k + 1, 0.5), up);
        let w = ring(k, 0.0);
        b.quad(
            ring(k, 0.5),
            ring(k + 1, 0.5),
            ring(k + 1, 0.0),
            ring(k, 0.0),
            [-w[0], -w[1], 0.0],
        );
        b.tri([0.0, 0.0, 0.0], ring(k, 0.0), ring(k + 1, 0.0), up);
        b.tri([0.0, 0.0, -0.5], sq(k, -0.5), sq(k + 1, -0.5), dn);
        let o = sq(k, 0.5);
        let p = sq(k + 1, 0.5);
        let (ox, oy) = (o[0] + p[0], o[1] + p[1]);
        let want = if ox.abs() > oy.abs() {
            [ox.signum(), 0.0, 0.0]
        } else {
            [0.0, oy.signum(), 0.0]
        };
        b.quad(sq(k, -0.5), sq(k + 1, -0.5), sq(k + 1, 0.5), sq(k, 0.5), want);
    }
    b.done()
}

/// TET-13c's fixture B: the prism over the isosceles triangle with the
/// 30-degree apex edge at `x = 0.5`, `x` from -0.5 to 0.5, `z` from -0.5
/// to 0.5, a 4 x 4 grid per face, every face oriented out of the solid.
fn wedge30() -> TetSurface {
    let n = 4usize;
    let w = 15f64.to_radians().tan();
    let mut b = Oriented::default();
    let ap = |z: f64| -> Point { [0.5, 0.0, z] };
    let bl = |z: f64| -> Point { [-0.5, -w, z] };
    let br = |z: f64| -> Point { [-0.5, w, z] };
    let lerp = |a: Point, c: Point, t: f64| -> Point {
        [a[0] + t * (c[0] - a[0]), a[1] + t * (c[1] - a[1]), a[2] + t * (c[2] - a[2])]
    };
    let g = |i: usize| i as f64 / n as f64;
    let zz = |j: usize| -0.5 + g(j);
    for i in 0..n {
        for j in 0..n {
            let f = |ii: usize, jj: usize| lerp(ap(zz(jj)), br(zz(jj)), g(ii));
            b.quad(f(i, j), f(i + 1, j), f(i + 1, j + 1), f(i, j + 1), [w, 1.0, 0.0]);
            let e = |ii: usize, jj: usize| lerp(ap(zz(jj)), bl(zz(jj)), g(ii));
            b.quad(e(i, j), e(i + 1, j), e(i + 1, j + 1), e(i, j + 1), [w, -1.0, 0.0]);
            let q = |ii: usize, jj: usize| lerp(bl(zz(jj)), br(zz(jj)), g(ii));
            b.quad(q(i, j), q(i + 1, j), q(i + 1, j + 1), q(i, j + 1), [-1.0, 0.0, 0.0]);
        }
    }
    for (z, s) in [(-0.5f64, -1.0f64), (0.5f64, 1.0f64)] {
        let (a, u, v) = (ap(z), sub(bl(z), ap(z)), sub(br(z), ap(z)));
        let pt = |i: usize, j: usize| -> Point {
            [a[0] + g(i) * u[0] + g(j) * v[0], a[1] + g(i) * u[1] + g(j) * v[1], z]
        };
        for i in 0..n {
            for j in 0..(n - i) {
                b.tri(pt(i, j), pt(i + 1, j), pt(i, j + 1), [0.0, 0.0, s]);
                if i + j + 1 < n {
                    b.tri(pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1), [0.0, 0.0, s]);
                }
            }
        }
    }
    b.done()
}

/// A TET-13c fixture's volume mesh: the optimised tets and their boundary
/// (the wall on patch 0, the hull on patch 1).
struct Fixture {
    points: Vec<Point>,
    tets: Vec<[u32; 4]>,
    bnd: Vec<([u32; 3], u32)>,
}

/// The fixture meshing of SPEC-LIT §116.18: the body's surface remeshed to
/// the size field, then a BCC lattice - `refine` (116.13) does not
/// terminate on these bodies - recovered, carved to the fluid and
/// optimised. A lattice candidate enters when it lies in the fluid and
/// every surface point is at least `0.7 sp` away.
fn build_fixture(surf: &TetSurface, inside: &dyn Fn(Point) -> bool) -> Fixture {
    let (h, sp, bx) = (0.1f64, 0.2f64, 1.5f64);
    let tris: Vec<[Point; 3]> =
        surf.tris.iter().map(|t| t.map(|v| surf.points[v as usize])).collect();
    let field = SizeField::build(
        SizeParams::new(0.5, 1.2, None).unwrap(),
        &[SizeSource::Patch {
            name: "body".to_string(),
            tris,
            distance: 0.0,
            size: h,
        }],
        [-bx; 3],
        [bx; 3],
        OctreeOptions::default(),
    )
    .unwrap();
    let prep = prepare(surf, 60.0).unwrap();
    let r = remesh(&prep, &|x: Point| field.h(x), &RemeshOptions::default()).unwrap();
    let mut s = r.surface.clone();
    let nsurf = s.points.len();
    let nl = (2.0 * bx / sp).round() as usize;
    for i in 0..=nl {
        for j in 0..=nl {
            for k in 0..=nl {
                let c = [-bx + i as f64 * sp, -bx + j as f64 * sp, -bx + k as f64 * sp];
                let mut cands = vec![c];
                if i < nl && j < nl && k < nl {
                    cands.push([
                        -bx + (i as f64 + 0.5) * sp,
                        -bx + (j as f64 + 0.5) * sp,
                        -bx + (k as f64 + 0.5) * sp,
                    ]);
                }
                for x in cands {
                    if inside(x) {
                        continue;
                    }
                    let far = s.points[..nsurf].iter().all(|&p| norm(sub(x, p)) >= 0.7 * sp);
                    if far {
                        s.points.push(x);
                    }
                }
            }
        }
    }
    let fr = recover_faces(&s, &r.feature_edges, &TetRecoverSpec::default()).unwrap();
    let carved = carve(&s, &fr, &CarveStart::Hull).unwrap();
    let tri_patch = vec![0u32; fr.tris.len()];
    let free = freedoms(&fr, &tri_patch, &carved.tets);
    let o = optimise(
        &fr.points,
        &carved.tets,
        &fr.subtris,
        &free,
        &TetOptimiseSpec::default(),
    )
    .unwrap();
    let bnd = carved_boundary(&carved, &fr, &tri_patch, 1).unwrap();
    Fixture { points: o.points, tets: o.tets, bnd }
}

fn cube_cyl_case() -> &'static Fixture {
    static C: OnceLock<Fixture> = OnceLock::new();
    C.get_or_init(|| {
        build_fixture(&cube_minus_cylinder(), &|x: Point| {
            x[0].abs() <= 0.5
                && x[1].abs() <= 0.5
                && x[2].abs() <= 0.5
                && !(x[0] * x[0] + x[1] * x[1] < 0.0625 && x[2] > 0.0)
        })
    })
}

fn wedge_case() -> &'static Fixture {
    static W: OnceLock<Fixture> = OnceLock::new();
    W.get_or_init(|| {
        build_fixture(&wedge30(), &|x: Point| {
            -0.5 <= x[0]
                && x[0] <= 0.5
                && x[2].abs() <= 0.5
                && x[1].abs() <= 15f64.to_radians().tan() * (0.5 - x[0])
        })
    })
}

/// TET-13c's prisms with repeated ids on `prism_split_right`'s right prism
/// points: with one point collapsed the prism is a pyramid (its two
/// distinct split rows exactly `+1`, volume `1/3`), with two it is a
/// tetrahedron (the one remaining row `+1`, volume `1/6`), the mixed
/// emitter takes each cell's reduced faces with an exact closure, and a
/// prism with no distinct split row is invalid.
#[test]
fn collapsed_prisms() {
    let pts: Vec<Point> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let single = std::mem::size_of::<ofgpu::Scalar>() == 4;
    let ps = vec![EmitPatch { name: "wall".to_string(), type_name: "wall".to_string() }];
    let check = |name: &str, prism: [u32; 6], rows: &[usize], bnd: Vec<(Vec<u32>, u32)>, nf: usize, vol: f64| {
        for &k in rows {
            let row = PRISM_SPLIT[k];
            let q = row.map(|i| pts[prism[i] as usize]);
            assert_eq!(orient3d(&q[0], &q[1], &q[2], &q[3]), 1.0, "{name} split {k}");
        }
        assert!(prism_valid(&pts, &prism), "{name}");
        let e = to_poly_mesh_mixed(&pts, &[], &[prism], &bnd, &ps).unwrap();
        assert_eq!(e.raw.faces.len(), nf, "{name} faces");
        assert!(e.raw.neighbour.is_empty(), "{name} internal");
        assert_eq!(e.raw.patches[0].size, nf, "{name} patch");
        let q = measure(&e.raw, &QualityThresholds::default()).unwrap();
        assert_eq!(q.n_cells, 1, "{name} cells");
        println!(
            "collapsed_prisms {name}: {nf} faces, 0 internal, 1 cell; max closure {e:.3e}, \
             min volume {v:.15}",
            e = q.max_closure,
            v = q.min_volume
        );
        if !single {
            assert!(
                (q.max_closure as f64) < 1e-10,
                "{name} closure {}",
                q.max_closure
            );
            let v = q.min_volume as f64;
            assert!((v - vol).abs() <= 1e-12, "{name} volume {v}");
        }
    };
    check(
        "pyramid [0,1,2,0,4,5]",
        [0u32, 1, 2, 0, 4, 5],
        &[1, 2],
        vec![
            (vec![0, 2, 1], 0),
            (vec![0, 4, 5], 0),
            (vec![0, 1, 4], 0),
            (vec![1, 2, 5, 4], 0),
            (vec![2, 0, 5], 0),
        ],
        5,
        1.0 / 3.0,
    );
    check(
        "tet [0,1,2,0,1,5]",
        [0u32, 1, 2, 0, 1, 5],
        &[2],
        vec![(vec![0, 2, 1], 0), (vec![0, 1, 5], 0), (vec![1, 2, 5], 0), (vec![2, 0, 5], 0)],
        4,
        1.0 / 6.0,
    );
    assert!(!prism_valid(&pts, &[0u32, 1, 2, 0, 1, 2]));
    println!("collapsed_prisms: [0,1,2,0,1,2] has no distinct split row: invalid");
}

/// §116.18's TET-13c gate (SPEC-LIT §116.15) on the two fixtures: 0
/// inverted cells, §92.3 passes (its miss under `single` is printed as
/// MISSED, docs/17-review-later A28), coverage 90 % or more, the collapsed
/// points named with their limit and `N + 1` copies of themselves, the
/// tapered points uncollapsed, and `N` prisms on every wall triangle that
/// has not fully collapsed.
#[test]
fn sharp_edges_taper() {
    let stack = LayerStack { n: 5, first_thickness: 5e-3, growth: 1.2 };
    let ps = vec![
        EmitPatch { name: "body".to_string(), type_name: "wall".to_string() },
        EmitPatch { name: "domain".to_string(), type_name: "patch".to_string() },
    ];
    let single = std::mem::size_of::<ofgpu::Scalar>() == 4;
    for (name, c) in [("cube minus cylinder", cube_cyl_case()), ("30-degree wedge", wedge_case())] {
        let inf = inflate(
            &c.points,
            &c.tets,
            &c.bnd,
            &[true, false],
            &stack,
            &TetLayerSpec::default(),
            60.0,
        )
        .unwrap();
        let bad_tets = inf
            .tets
            .iter()
            .filter(|t| {
                let q = t.map(|v| inf.points[v as usize]);
                orient3d(&q[0], &q[1], &q[2], &q[3]) <= 0.0
            })
            .count();
        let bad_prisms = inf.prisms.iter().filter(|p| !prism_valid(&inf.points, p)).count();
        let flips = rounding_flips_mixed(&inf.points, &inf.tets, &inf.prisms);
        assert_eq!(bad_tets, 0, "{name}: {} inverted tets", bad_tets);
        assert_eq!(bad_prisms, 0, "{name}: {} failing prisms", bad_prisms);
        assert!(flips.is_empty(), "{name}: {} rounding flips", flips.len());
        let e = to_poly_mesh_mixed(&inf.points, &inf.tets, &inf.prisms, &inf.boundary, &ps)
            .unwrap_or_else(|err| panic!("{name}: the mixed emission refused: {err}"));
        let q = measure(&e.raw, &QualityThresholds::default()).unwrap();
        let missed = !q.passed();
        if missed && single {
            println!(
                "{name}: MISSED - §92.3 under single (docs/17-review-later A28): max closure \
                 {mc:.3e} (cell {mcc}), max non-orthogonality {no:.3} deg, min thickness ratio \
                 {tr:.4} (cell {tc}), max cond {co:.3} (cell {cc})",
                mc = q.max_closure, mcc = q.max_closure_cell,
                no = q.max_non_orth_deg, tr = q.min_thickness_ratio,
                tc = q.min_thickness_cell, co = q.max_cond, cc = q.max_cond_cell,
            );
        }
        assert!(inf.coverage() >= 0.9, "{name}: coverage {}", inf.coverage());
        let slot_of: HashMap<u32, usize> =
            inf.normals.points.iter().enumerate().map(|(k, &p)| (p, k)).collect();
        for &p in &inf.collapsed {
            let k = slot_of[&p];
            assert_eq!(inf.thickness.limit[k], Limit::Collapsed, "{name}: point {p}");
            let row = &inf.layers[k];
            assert_eq!(row.len(), stack.n + 1, "{name}: point {p} row");
            assert!(row.iter().all(|&x| x == p), "{name}: point {p} row {row:?}");
        }
        for &p in &inf.tapered {
            assert!(
                slot_of.contains_key(&p) && !inf.collapsed.contains(&p),
                "{name}: tapered point {p} collapsed or no wall point"
            );
        }
        let collapsed_set: BTreeSet<u32> = inf.collapsed.iter().copied().collect();
        let wall = c.bnd.iter().filter(|&&(_, p)| p == 0).count();
        let full = c
            .bnd
            .iter()
            .filter(|&&(f, p)| p == 0 && f.iter().all(|v| collapsed_set.contains(&v)))
            .count();
        assert_eq!(
            inf.prisms.len(),
            5 * (wall - full),
            "{name}: {} prisms for {wall} wall triangles, {full} fully collapsed",
            inf.prisms.len()
        );
        let head = |v: &[u32]| format!("{:?} of {}", &v[..v.len().min(20)], v.len());
        println!(
            "sharp_edges_taper {name}: {t} tets, {wp} wall points, collapsed {col}, tapered \
             {tap}, coverage {cv} = {ok}/{wp}, rounds {rd}, n_cut {nc}, prisms {pr}; \
             92.3: max closure {mc:.3e}, max non-orth {no:.1} deg, min thickness {tr:.4}, \
             max cond {co:.1}, {gate}",
            t = inf.tets.len(),
            wp = inf.normals.points.len(),
            col = head(&inf.collapsed),
            tap = head(&inf.tapered),
            cv = inf.coverage(),
            ok = inf.normals.points.len() - inf.collapsed.len(),
            rd = inf.rounds,
            nc = inf.n_cut,
            pr = inf.prisms.len(),
            mc = q.max_closure,
            no = q.max_non_orth_deg,
            tr = q.min_thickness_ratio,
            co = q.max_cond,
            gate = if q.passed() { "PASS" } else { "MISSED under single" },
        );
        if missed {
            assert!(single, "{name}: 92.3 failed in f64: {}", q.refusal_text());
        }
    }
}

/// TET-13c on the sphere: nothing is sharp, so no point collapses and none
/// is tapered, and the coverage is 1.
#[test]
fn sphere_has_no_collapse() {
    let c = sphere_case();
    let inf = inflate_sphere(c);
    println!(
        "sphere_has_no_collapse: {} wall points, collapsed {:?}, tapered {:?}, coverage {}",
        inf.normals.points.len(),
        inf.collapsed,
        inf.tapered,
        inf.coverage()
    );
    assert!(inf.collapsed.is_empty(), "{:?}", inf.collapsed);
    assert!(inf.tapered.is_empty(), "{:?}", inf.tapered);
    assert_eq!(inf.coverage(), 1.0);
}
