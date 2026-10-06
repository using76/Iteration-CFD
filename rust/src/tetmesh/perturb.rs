// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The symbolically perturbed `insphere` of SPEC-LIT §116.5, rule
//! (116.4). A point `p_i` with global vertex index `i` is lifted to
//! `|p_i|^2 + eps_i`, with `eps_i` larger for a larger index, so the
//! result is 0 only when all five points are coplanar; a finite
//! tetrahedron never asks such a query. This module calls the exact
//! predicates of §116.4 and does nothing else to the coordinates: it
//! reads the sign of (116.3), and when that is exactly 0 the signs of
//! the `orient3d` cofactors of (116.4a) in decreasing order of global
//! vertex index, which is the rule (116.4). The depth it reports is
//! the number of those terms, at most 2 for a query that is not
//! coplanar, by (116.4b).
//!
//! Provenance: ORIGINAL. Written from Edelsbrunner & Mücke (1990), ACM Trans. Graph. 9(1):66-104, and Devillers & Teillaud (2011), Comput. Geom. 44(3):160-168. No GPL-licensed source was consulted.

use super::predicates::{insphere, orient3d};
use super::Point;

/// The sign and the depth of (116.4), the perturbed `insphere` of
/// §116.5. `p[j]` is the argument at position `j`, `0` for `a` through
/// `4` for `e`, and `id[j]` is the global vertex index of `p[j]`. The
/// ids must be pairwise distinct, which is checked with
/// `debug_assert!` only.
///
/// The depth is the number of `orient3d` terms of (116.4) evaluated:
/// `0` when `insphere` of (116.3) is not 0, otherwise the position in
/// the decreasing-index scan of the first non-zero term, and `5` for a
/// coplanar query, whose answer is 0. A query that is not coplanar
/// never needs more than 2 terms, by (116.4b). No allocation.
pub fn insphere_perturbed_depth(p: [&Point; 5], id: [usize; 5]) -> (i32, u32) {
    debug_assert!(
        id[0] != id[1]
            && id[0] != id[2]
            && id[0] != id[3]
            && id[0] != id[4]
            && id[1] != id[2]
            && id[1] != id[3]
            && id[1] != id[4]
            && id[2] != id[3]
            && id[2] != id[4]
            && id[3] != id[4],
        "the five vertex ids must be pairwise distinct"
    );
    let s = insphere(p[0], p[1], p[2], p[3], p[4]);
    if s > 0.0 {
        return (1, 0);
    }
    if s < 0.0 {
        return (-1, 0);
    }
    // The five positions, in a fixed-size array, sorted into
    // decreasing order of global vertex index.
    let mut order = [0usize, 1, 2, 3, 4];
    for i in 1..5 {
        let key = order[i];
        let key_id = id[key];
        let mut j = i;
        while j > 0 && id[order[j - 1]] < key_id {
            order[j] = order[j - 1];
            j -= 1;
        }
        order[j] = key;
    }
    let mut depth = 0u32;
    for &j in order.iter() {
        depth += 1;
        // The cofactor C_j of (116.4a) is (-1)^(j+1) times the
        // `orient3d` of the four points other than `p[j]`, in their
        // argument order; (116.4) reads its sign.
        let o = match j {
            0 => orient3d(p[1], p[2], p[3], p[4]),
            1 => orient3d(p[0], p[2], p[3], p[4]),
            2 => orient3d(p[0], p[1], p[3], p[4]),
            3 => orient3d(p[0], p[1], p[2], p[4]),
            _ => orient3d(p[0], p[1], p[2], p[3]),
        };
        if o != 0.0 {
            let so = if o > 0.0 { 1i32 } else { -1i32 };
            let sign = if j % 2 == 0 { -so } else { so };
            return (sign, depth);
        }
    }
    (0, 5)
}

/// The sign of (116.4): the first element of
/// `insphere_perturbed_depth`. It is 0 only when all five points are
/// coplanar, and it is antisymmetric under any swap of two arguments
/// together with its id, because it is the sign of one perturbed
/// determinant of §116.5.
pub fn insphere_perturbed(p: [&Point; 5], id: [usize; 5]) -> i32 {
    insphere_perturbed_depth(p, id).0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An integer triple as a `Point`.
    fn pt(v: [i64; 3]) -> Point {
        [v[0] as f64, v[1] as f64, v[2] as f64]
    }

    /// The five `Point`s of a query.
    fn q5(p: &[[i64; 3]; 5]) -> [Point; 5] {
        [pt(p[0]), pt(p[1]), pt(p[2]), pt(p[3]), pt(p[4])]
    }

    /// The sign of an integer.
    fn sgn_i(v: i128) -> i32 {
        if v > 0 {
            1
        } else if v < 0 {
            -1
        } else {
            0
        }
    }

    /// The result of a query, with the sign alone re-checked through
    /// `insphere_perturbed`.
    fn result_of(p: [&Point; 5], id: [usize; 5]) -> (i32, u32) {
        let r = insphere_perturbed_depth(p, id);
        assert_eq!(insphere_perturbed(p, id), r.0);
        r
    }

    /// The result over integer points.
    fn result_i(p: &[[i64; 3]; 5], id: [usize; 5]) -> (i32, u32) {
        let f = q5(p);
        result_of([&f[0], &f[1], &f[2], &f[3], &f[4]], id)
    }

    /// SplitMix64 (Steele, Lea & Flood 2014), all operations wrapping.
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
        fn range(&mut self, lo: i64, hi: i64) -> i64 {
            lo + (self.next() % ((hi - lo) as u64)) as i64
        }
    }

    /// Draw `m` distinct values from `0..n`, in draw order: repeat
    /// `v = range(0, n)`, keep `v` if it is not already kept, until
    /// `m` are kept.
    fn draw_distinct(r: &mut SplitMix64, m: usize, n: i64) -> Vec<i64> {
        let mut kept: Vec<i64> = Vec::new();
        while kept.len() < m {
            let v = r.range(0, n);
            if !kept.contains(&v) {
                kept.push(v);
            }
        }
        kept
    }

    /// The lattice point of an index: `k` in `0..1000` is
    /// `(k % 10, (k / 10) % 10, k / 100)`.
    fn lat(k: i64) -> [i64; 3] {
        [k % 10, (k / 10) % 10, k / 100]
    }

    /// The global vertex index of a lattice point.
    fn lat_index(p: [i64; 3]) -> i64 {
        p[0] + 10 * p[1] + 100 * p[2]
    }

    /// U: five distinct lattice points; the ids are their lattice
    /// indices.
    fn gen_u(r: &mut SplitMix64) -> ([[i64; 3]; 5], [usize; 5]) {
        let ids = draw_distinct(r, 5, 1000);
        let mut p = [[0i64; 3]; 5];
        let mut id = [0usize; 5];
        for j in 0..5 {
            p[j] = lat(ids[j]);
            id[j] = ids[j] as usize;
        }
        (p, id)
    }

    /// C: five corners of one cube of the lattice, of side `s`; the
    /// ids are the lattice indices of the corners.
    fn gen_c(r: &mut SplitMix64) -> ([[i64; 3]; 5], [usize; 5]) {
        let s = r.range(1, 10);
        let x0 = r.range(0, 10 - s);
        let y0 = r.range(0, 10 - s);
        let z0 = r.range(0, 10 - s);
        let cs = draw_distinct(r, 5, 8);
        let mut p = [[0i64; 3]; 5];
        let mut id = [0usize; 5];
        for j in 0..5 {
            let c = cs[j];
            p[j] = [
                x0 + s * (c & 1),
                y0 + s * ((c >> 1) & 1),
                z0 + s * ((c >> 2) & 1),
            ];
            id[j] = lat_index(p[j]) as usize;
        }
        (p, id)
    }

    /// S: a U query or a C query from the same generator, then two
    /// distinct positions `i` and `k`.
    fn gen_s(r: &mut SplitMix64) -> ([[i64; 3]; 5], [usize; 5], usize, usize) {
        let (p, id) = if r.next() & 1 == 0 {
            gen_u(r)
        } else {
            gen_c(r)
        };
        let i = r.range(0, 5) as usize;
        let mut k = r.range(0, 4) as usize;
        if k >= i {
            k += 1;
        }
        (p, id, i, k)
    }

    fn sub3(a: [i64; 3], b: [i64; 3]) -> [i128; 3] {
        [
            a[0] as i128 - b[0] as i128,
            a[1] as i128 - b[1] as i128,
            a[2] as i128 - b[2] as i128,
        ]
    }

    fn det3_i128(r0: [i128; 3], r1: [i128; 3], r2: [i128; 3]) -> i128 {
        r0[0] * (r1[1] * r2[2] - r1[2] * r2[1]) - r0[1] * (r1[0] * r2[2] - r1[2] * r2[0])
            + r0[2] * (r1[0] * r2[1] - r1[1] * r2[0])
    }

    /// (116.1) on integer points: the 3x3 determinant of the rows
    /// `a - d`, `b - d`, `c - d`.
    fn o3_i128(a: [i64; 3], b: [i64; 3], c: [i64; 3], d: [i64; 3]) -> i128 {
        det3_i128(sub3(a, d), sub3(b, d), sub3(c, d))
    }

    fn lift2(w: [i128; 3]) -> i128 {
        w[0] * w[0] + w[1] * w[1] + w[2] * w[2]
    }

    /// (116.3) on integer points: the 4x4 determinant of the rows
    /// `[p - e, |p - e|^2]` for `p = a, b, c, d`, by cofactor
    /// expansion along its last column.
    fn is_i128(a: [i64; 3], b: [i64; 3], c: [i64; 3], d: [i64; 3], e: [i64; 3]) -> i128 {
        let ae = sub3(a, e);
        let be = sub3(b, e);
        let ce = sub3(c, e);
        let de = sub3(d, e);
        -lift2(ae) * det3_i128(be, ce, de)
            + lift2(be) * det3_i128(ae, ce, de)
            - lift2(ce) * det3_i128(ae, be, de)
            + lift2(de) * det3_i128(ae, be, ce)
    }

    /// True when all five points are coplanar: the `o3_i128` of the
    /// four points other than each of them, in argument order, is 0.
    fn coplanar5(p: &[[i64; 3]; 5]) -> bool {
        for j in 0..5 {
            let mut rest = [[0i64; 3]; 4];
            let mut m = 0usize;
            for t in 0..5 {
                if t != j {
                    rest[m] = p[t];
                    m += 1;
                }
            }
            if o3_i128(rest[0], rest[1], rest[2], rest[3]) != 0 {
                return false;
            }
        }
        true
    }

    fn det4_i128(m: [[i128; 4]; 4]) -> i128 {
        let mut acc = 0i128;
        for i in 0..4 {
            let mut sub = [[0i128; 3]; 3];
            let mut rr = 0usize;
            for t in 0..4 {
                if t != i {
                    sub[rr] = [m[t][0], m[t][1], m[t][2]];
                    rr += 1;
                }
            }
            let term = m[i][3] * det3_i128(sub[0], sub[1], sub[2]);
            acc += if (i + 3) % 2 == 0 { term } else { -term };
        }
        acc
    }

    /// (116.4c), the explicit perturbation, as an integer determinant:
    /// with `N = 2^14` and `r_j` the number of ids in `id` smaller
    /// than `id[j]`, row `j` is
    /// `[x_j, y_j, z_j, N^5 |p_j|^2 + N^(r_j), 1]`; its sign is the
    /// sign of the 5x5 determinant, evaluated exactly in `i128` by
    /// cofactor expansion along the last column. Exact for every
    /// coordinate in `-7..=9`, by the bound in §116.5.
    fn explicit_i128(p: &[[i64; 3]; 5], id: &[usize; 5]) -> i32 {
        const N: i128 = 1i128 << 14;
        let mut m = [[0i128; 5]; 5];
        for j in 0..5 {
            let mut r = 0usize;
            for t in 0..5 {
                if id[t] < id[j] {
                    r += 1;
                }
            }
            let sq = p[j][0] as i128 * p[j][0] as i128
                + p[j][1] as i128 * p[j][1] as i128
                + p[j][2] as i128 * p[j][2] as i128;
            m[j] = [
                p[j][0] as i128,
                p[j][1] as i128,
                p[j][2] as i128,
                N.pow(5) * sq + N.pow(r as u32),
                1,
            ];
        }
        let mut acc = 0i128;
        for j in 0..5 {
            let mut sub = [[0i128; 4]; 4];
            let mut rr = 0usize;
            for t in 0..5 {
                if t != j {
                    sub[rr] = [m[t][0], m[t][1], m[t][2], m[t][3]];
                    rr += 1;
                }
            }
            let term = det4_i128(sub);
            acc += if (j + 4) % 2 == 0 { term } else { -term };
        }
        sgn_i(acc)
    }

    /// The unit-cube corners, CUBE[i] = (i & 1, (i >> 1) & 1, (i >> 2) & 1).
    const CUBE: [[i64; 3]; 8] = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ];

    /// The eight points of the sphere of radius 7.
    const R7: [[i64; 3]; 8] = [
        [7, 0, 0],
        [0, 7, 0],
        [0, 0, 7],
        [2, 3, 6],
        [6, -2, 3],
        [-3, 6, -2],
        [-7, 0, 0],
        [2, -6, -3],
    ];

    /// The 56 five-subsets of the eight indices, lexicographic.
    fn combos5() -> Vec<[usize; 5]> {
        let mut out = Vec::new();
        for a in 0..8usize {
            for b in a + 1..8 {
                for c in b + 1..8 {
                    for d in c + 1..8 {
                        for e in d + 1..8 {
                            out.push([a, b, c, d, e]);
                        }
                    }
                }
            }
        }
        out
    }

    /// The 120 orderings of five positions, lexicographic.
    fn perms5() -> Vec<[usize; 5]> {
        fn rec(cur: &mut Vec<usize>, rest: &mut Vec<usize>, out: &mut Vec<[usize; 5]>) {
            if rest.is_empty() {
                out.push([cur[0], cur[1], cur[2], cur[3], cur[4]]);
                return;
            }
            for i in 0..rest.len() {
                let v = rest.remove(i);
                cur.push(v);
                rec(cur, rest, out);
                cur.pop();
                rest.insert(i, v);
            }
        }
        let mut out = Vec::new();
        let mut cur = Vec::new();
        let mut rest: Vec<usize> = (0..5).collect();
        rec(&mut cur, &mut rest, &mut out);
        out
    }

    /// One fixture: the result must be `want`, and the explicit
    /// perturbation must give the same sign.
    fn check_fixture(name: &str, p: &[[i64; 3]; 5], id: [usize; 5], want: (i32, u32)) {
        let r = result_i(p, id);
        assert_eq!(r, want, "fixture {}", name);
        assert_eq!(explicit_i128(p, &id), want.0, "fixture {} explicit", name);
    }

    #[test]
    fn fixtures_exact() {
        let f1 = [CUBE[0], CUBE[1], CUBE[2], CUBE[4], CUBE[7]];
        check_fixture("F1", &f1, [0, 1, 2, 4, 7], (1, 1));
        check_fixture("F2", &f1, [7, 1, 2, 4, 0], (1, 1));
        check_fixture("F3", &f1, [3, 9, 1, 5, 2], (-1, 1));
        let f4 = [CUBE[0], CUBE[1], CUBE[2], CUBE[4], CUBE[3]];
        check_fixture("F4", &f4, [10, 11, 12, 13, 14], (1, 1));
        let f5 = [CUBE[0], CUBE[1], CUBE[2], CUBE[3], CUBE[4]];
        check_fixture("F5", &f5, [0, 1, 2, 3, 4], (-1, 2));
        // F6 is coplanar, so its 0 is the contract.
        let f6 = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [3, 7, 0]];
        check_fixture("F6", &f6, [0, 1, 2, 3, 4], (0, 5));
        // F7 is decided by `insphere` alone.
        let f7 = [[0, 0, 0], [4, 0, 0], [0, 0, 4], [0, 4, 0], [1, 1, 1]];
        assert_eq!(is_i128(f7[0], f7[1], f7[2], f7[3], f7[4]), 576);
        check_fixture("F7", &f7, [0, 1, 2, 3, 4], (1, 0));
        let f8 = [R7[0], R7[1], R7[2], R7[3], R7[4]];
        check_fixture("F8", &f8, [0, 1, 2, 3, 4], (1, 1));
        check_fixture("F9", &f8, [4, 3, 2, 1, 0], (-1, 1));
    }

    /// One lattice query of test 3: a coplanar query answers 0 at
    /// depth 5; a non-coplanar one is never 0, its depth is at most 2
    /// by (116.4b), and when `insphere` is not 0 it is the answer at
    /// depth 0. Counts into `cnt`: coplanar, then depths 0, 1, 2.
    fn scan_lattice(
        p: &[[i64; 3]; 5],
        id: &[usize; 5],
        cnt: &mut (usize, usize, usize, usize),
        zero_noncop: &mut usize,
    ) {
        let cop = coplanar5(p);
        let res = result_i(p, *id);
        if cop {
            assert_eq!(res, (0, 5));
            cnt.0 += 1;
            return;
        }
        if res.0 == 0 {
            *zero_noncop += 1;
            return;
        }
        let is = is_i128(p[0], p[1], p[2], p[3], p[4]);
        if is != 0 {
            assert_eq!(res, (sgn_i(is), 0));
            cnt.1 += 1;
        } else {
            assert!(
                res.1 == 1 || res.1 == 2,
                "depth over a non-coplanar query is at most 2, by (116.4b)"
            );
            if res.1 == 1 {
                cnt.2 += 1;
            } else {
                cnt.3 += 1;
            }
        }
    }

    #[test]
    fn never_zero_on_lattice() {
        let mut u = (0usize, 0usize, 0usize, 0usize);
        let mut c = (0usize, 0usize, 0usize, 0usize);
        let mut zero_noncop = 0usize;
        let mut r = SplitMix64::new(21);
        let (p, id) = gen_u(&mut r);
        assert_eq!(p, [[3, 4, 5], [9, 1, 7], [5, 1, 4], [7, 5, 4], [1, 8, 4]]);
        assert_eq!(id, [543, 719, 415, 457, 481]);
        scan_lattice(&p, &id, &mut u, &mut zero_noncop);
        for _ in 1..500000 {
            let (p, id) = gen_u(&mut r);
            scan_lattice(&p, &id, &mut u, &mut zero_noncop);
        }
        let mut r = SplitMix64::new(22);
        let (p, id) = gen_c(&mut r);
        assert_eq!(p, [[2, 9, 8], [2, 3, 8], [8, 3, 8], [8, 9, 8], [8, 3, 2]]);
        assert_eq!(id, [892, 832, 838, 898, 238]);
        scan_lattice(&p, &id, &mut c, &mut zero_noncop);
        for _ in 1..500000 {
            let (p, id) = gen_c(&mut r);
            scan_lattice(&p, &id, &mut c, &mut zero_noncop);
        }
        println!(
            "TET02-LATTICE U cop={} d0={} d1={} d2={} | C cop={} d0={} d1={} d2={} | zero_noncoplanar={}",
            u.0, u.1, u.2, u.3, c.0, c.1, c.2, c.3, zero_noncop
        );
        assert_eq!(u, (351, 497975, 1515, 159));
        assert_eq!(c, (0, 0, 401380, 98620));
        assert_eq!(zero_noncop, 0);
    }

    /// One S query of test 4: a coplanar query is skipped; otherwise
    /// the swapped query, over `(i, k)` and over all ten swaps for the
    /// first 1000 non-skipped queries, must negate the sign at an
    /// unchanged depth.
    fn swap_query(
        p: &[[i64; 3]; 5],
        id: &[usize; 5],
        i: usize,
        k: usize,
        skipped: &mut usize,
        checked: &mut usize,
        seen: &mut usize,
    ) {
        if coplanar5(p) {
            *skipped += 1;
            return;
        }
        *checked += 1;
        *seen += 1;
        let res = result_i(p, *id);
        assert!(res.0 != 0);
        let mut pairs = [(0usize, 0usize); 10];
        let mut np = 1usize;
        pairs[0] = (i, k);
        if *seen <= 1000 {
            np = 0;
            for a in 0..5usize {
                for b in a + 1..5 {
                    pairs[np] = (a, b);
                    np += 1;
                }
            }
        }
        for t in 0..np {
            let (a, b) = pairs[t];
            let mut p2 = *p;
            let mut id2 = *id;
            p2.swap(a, b);
            id2.swap(a, b);
            let r2 = result_i(&p2, id2);
            assert_eq!(r2.0, -res.0, "a swap must negate the sign");
            assert_eq!(r2.1, res.1, "a swap must not change the depth");
        }
    }

    #[test]
    fn antisymmetric() {
        let mut r = SplitMix64::new(23);
        let mut skipped = 0usize;
        let mut checked = 0usize;
        let mut seen = 0usize;
        let (p, id, i, k) = gen_s(&mut r);
        assert_eq!(p, [[9, 7, 6], [0, 1, 9], [7, 4, 9], [6, 7, 1], [4, 3, 5]]);
        assert_eq!(id, [679, 910, 947, 176, 534]);
        assert_eq!(i, 1);
        assert_eq!(k, 2);
        swap_query(&p, &id, i, k, &mut skipped, &mut checked, &mut seen);
        for _ in 1..100000 {
            let (p, id, i, k) = gen_s(&mut r);
            swap_query(&p, &id, i, k, &mut skipped, &mut checked, &mut seen);
        }
        println!("TET02-SWAP skipped={} checked={}", skipped, checked);
        assert_eq!(skipped, 35);
        assert_eq!(checked, 99965);
    }

    #[test]
    fn matches_explicit_perturbation() {
        let perms = perms5();
        let combos = combos5();
        let both = [CUBE, R7];
        let want: [(usize, usize, usize); 2] = [(13440, 0, 0), (13440, 240, 240)];
        for si in 0..2 {
            let set = &both[si];
            let mut nq = 0usize;
            let mut nz = 0usize;
            let mut ncop = 0usize;
            for combo in combos.iter() {
                let mut sub = [[0i64; 3]; 5];
                for t in 0..5 {
                    sub[t] = set[combo[t]];
                }
                let cop = coplanar5(&sub);
                for map in 0..2 {
                    for perm in perms.iter() {
                        let mut qp = [[0i64; 3]; 5];
                        let mut id = [0usize; 5];
                        for pos in 0..5 {
                            let sidx = combo[perm[pos]];
                            qp[pos] = set[sidx];
                            id[pos] = if map == 0 { sidx } else { 7 - sidx };
                        }
                        let r = result_i(&qp, id);
                        assert_eq!(r.0, explicit_i128(&qp, &id), "vs explicit");
                        nq += 1;
                        if r.0 == 0 {
                            nz += 1;
                            assert!(cop, "a zero result must be coplanar");
                        }
                        if cop {
                            ncop += 1;
                            assert_eq!(r, (0, 5));
                        }
                    }
                }
            }
            assert_eq!((nq, nz, ncop), want[si], "set index {}", si);
        }
        // (b) The first 20000 queries of U and of C.
        for (seed, ugen) in [(21u64, true), (22u64, false)] {
            let mut r = SplitMix64::new(seed);
            for _ in 0..20000 {
                let (p, id) = if ugen { gen_u(&mut r) } else { gen_c(&mut r) };
                let res = result_i(&p, id);
                assert_eq!(res.0, explicit_i128(&p, &id), "vs explicit");
            }
        }
    }

    #[test]
    fn depends_only_on_index_order() {
        for (seed, ugen) in [(21u64, true), (22u64, false)] {
            let mut r = SplitMix64::new(seed);
            for _ in 0..10000 {
                let (p, id) = if ugen { gen_u(&mut r) } else { gen_c(&mut r) };
                let base = result_i(&p, id);
                let id2 = [
                    id[0] * 3 + 7,
                    id[1] * 3 + 7,
                    id[2] * 3 + 7,
                    id[3] * 3 + 7,
                    id[4] * 3 + 7,
                ];
                assert_eq!(result_i(&p, id2), base);
                let big = 1usize << 40;
                let id3 = [
                    id[0] + big,
                    id[1] + big,
                    id[2] + big,
                    id[3] + big,
                    id[4] + big,
                ];
                assert_eq!(result_i(&p, id3), base);
            }
        }
    }

    #[test]
    fn translation_and_scale_invariant() {
        const SHIFT: [f64; 3] = [123457.0, -654321.0, 999000.0];
        for (seed, ugen) in [(21u64, true), (22u64, false)] {
            let mut r = SplitMix64::new(seed);
            for _ in 0..10000 {
                let (p, id) = if ugen { gen_u(&mut r) } else { gen_c(&mut r) };
                let base = result_i(&p, id);
                let mut sp = [[0.0f64; 3]; 5];
                for j in 0..5 {
                    for t in 0..3 {
                        sp[j][t] = p[j][t] as f64 + SHIFT[t];
                    }
                }
                let r1 = result_of([&sp[0], &sp[1], &sp[2], &sp[3], &sp[4]], id);
                assert_eq!(r1, base);
                for sc in [2f64.powi(-40), 2f64.powi(40)] {
                    let mut cp = [[0.0f64; 3]; 5];
                    for j in 0..5 {
                        for t in 0..3 {
                            cp[j][t] = p[j][t] as f64 * sc;
                        }
                    }
                    let r2 = result_of([&cp[0], &cp[1], &cp[2], &cp[3], &cp[4]], id);
                    assert_eq!(r2, base);
                }
            }
        }
    }
}
