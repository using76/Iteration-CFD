// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

// TET-13a's tests (SPEC-LIT §116.15, §116.18), written once and included
// twice: by src/tetmesh/inflate.rs's test module and by
// tests/tetmesh_inflate.rs, so the `--features single` build, whose
// library test crate does not compile (SPEC-LIT §112.3), runs them too.
// They use the public API only. Every number is evaluated in f64 (§116.2),
// so the gates are class A with the same bars, and the printed lines are
// the same, in both builds.
//
// Provenance: ORIGINAL. Aubry & Löhner (2008)'s most normal normal as
// SPEC-LIT §116.15 solves it. No GPL-licensed source was consulted.

use ofgpu::tetmesh::inflate::{
    angle, most_normal, thicknesses, wall_normals, LayerStack, Limit, Thicknesses,
};
use ofgpu::tetmesh::surfprep::TetSurface;
use ofgpu::tetmesh::{Point, TetLayerSpec};
use std::collections::HashMap;

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
