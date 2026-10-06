// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The exact predicates of SPEC-LIT §116.4: `orient3d` of (116.1) and
//! `insphere` of (116.3), with their signs.
//!
//! Stage A, the floating-point filter of (116.3b) and (116.3c), runs
//! first; the exact stage in expansion arithmetic, (116.3d) and
//! (116.3e), runs only when stage A cannot decide. The points are
//! `[f64; 3]` in both builds (§116.2). The input domain is (116.3f),
//! and `in_exact_domain` tests it.
//!
//! Provenance: ORIGINAL. Written from Shewchuk (1997), Discrete Comput. Geom. 18:305-363. No GPL-licensed source was consulted.

use super::Point;

/// Half an ulp of 1.0, the `eps` of (116.3a): `2^-53`, bits
/// 0x3CA0000000000000.
pub const EPS: f64 = f64::from_bits(0x3CA0000000000000);

/// The stage A filter bound of (116.3b), `(7 + 56 eps) eps`, multiplied
/// by the permanent `perm` of the cofactor terms.
pub const O3D_BOUND_A: f64 = (7.0 + 56.0 * EPS) * EPS;

/// The stage A filter bound of (116.3c), `(16 + 224 eps) eps`, multiplied
/// by the permanent `perm` of the cofactor terms.
pub const ISP_BOUND_A: f64 = (16.0 + 224.0 * EPS) * EPS;

/// Stage A of `orient3d`, (116.3b): returns `(det~, bound)`. `det~` is
/// computed in exactly the order (116.3b) writes it, with every
/// parenthesis kept, and `bound` is `O3D_BOUND_A * perm`. The filter
/// decides iff `|det~| > bound`, and then the sign of (116.1) is the
/// sign of `det~`. No allocation.
pub fn orient3d_stage_a(a: &Point, b: &Point, c: &Point, d: &Point) -> (f64, f64) {
    let adx = a[0] - d[0];
    let ady = a[1] - d[1];
    let adz = a[2] - d[2];
    let bdx = b[0] - d[0];
    let bdy = b[1] - d[1];
    let bdz = b[2] - d[2];
    let cdx = c[0] - d[0];
    let cdy = c[1] - d[1];
    let cdz = c[2] - d[2];
    let det = adz * (bdx * cdy - cdx * bdy)
        + bdz * (cdx * ady - adx * cdy)
        + cdz * (adx * bdy - bdx * ady);
    let perm = ((bdx * cdy).abs() + (cdx * bdy).abs()) * adz.abs()
        + ((cdx * ady).abs() + (adx * cdy).abs()) * bdz.abs()
        + ((adx * bdy).abs() + (bdx * ady).abs()) * cdz.abs();
    (det, O3D_BOUND_A * perm)
}

/// Stage A of `insphere`, (116.3c): returns `(det~, bound)`. `det~` is
/// computed in exactly the order (116.3c) writes it, with every
/// parenthesis kept, and `bound` is `ISP_BOUND_A * perm`. The filter
/// decides iff `|det~| > bound`, and then the sign of (116.3) is the
/// sign of `det~`. No allocation.
pub fn insphere_stage_a(a: &Point, b: &Point, c: &Point, d: &Point, e: &Point) -> (f64, f64) {
    let aex = a[0] - e[0];
    let aey = a[1] - e[1];
    let aez = a[2] - e[2];
    let bex = b[0] - e[0];
    let bey = b[1] - e[1];
    let bez = b[2] - e[2];
    let cex = c[0] - e[0];
    let cey = c[1] - e[1];
    let cez = c[2] - e[2];
    let dex = d[0] - e[0];
    let dey = d[1] - e[1];
    let dez = d[2] - e[2];
    let ab = aex * bey - bex * aey;
    let bc = bex * cey - cex * bey;
    let cd = cex * dey - dex * cey;
    let da = dex * aey - aex * dey;
    let ac = aex * cey - cex * aey;
    let bd = bex * dey - dex * bey;
    let abc = aez * bc - bez * ac + cez * ab;
    let bcd = bez * cd - cez * bd + dez * bc;
    let cda = cez * da + dez * ac + aez * cd;
    let dab = dez * ab + aez * bd + bez * da;
    let alift = aex * aex + aey * aey + aez * aez;
    let blift = bex * bex + bey * bey + bez * bez;
    let clift = cex * cex + cey * cey + cez * cez;
    let dlift = dex * dex + dey * dey + dez * dez;
    let det = (dlift * abc - clift * dab) + (blift * cda - alift * bcd);
    let perm = (((cex * dey).abs() + (dex * cey).abs()) * bez.abs()
        + ((dex * bey).abs() + (bex * dey).abs()) * cez.abs()
        + ((bex * cey).abs() + (cex * bey).abs()) * dez.abs())
        * alift
        + (((dex * aey).abs() + (aex * dey).abs()) * cez.abs()
            + ((aex * cey).abs() + (cex * aey).abs()) * dez.abs()
            + ((cex * dey).abs() + (dex * cey).abs()) * aez.abs())
            * blift
        + (((aex * bey).abs() + (bex * aey).abs()) * dez.abs()
            + ((bex * dey).abs() + (dex * bey).abs()) * aez.abs()
            + ((dex * aey).abs() + (aex * dey).abs()) * bez.abs())
            * clift
        + (((bex * cey).abs() + (cex * bey).abs()) * aez.abs()
            + ((cex * aey).abs() + (aex * cey).abs()) * bez.abs()
            + ((aex * bey).abs() + (bex * aey).abs()) * cez.abs())
            * dlift;
    (det, ISP_BOUND_A * perm)
}

/// Two-Sum, a block of (116.3d): `a + b = x + y` exactly.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let bv = x - a;
    let av = x - bv;
    let y = (a - av) + (b - bv);
    (x, y)
}

/// Two-Diff, a block of (116.3d): `a - b = x + y` exactly.
fn two_diff(a: f64, b: f64) -> (f64, f64) {
    let x = a - b;
    let bv = a - x;
    let av = x + bv;
    let y = (a - av) + (bv - b);
    (x, y)
}

/// Split, a block of (116.3d), with `2^27 + 1 = 134217729.0`.
fn split(a: f64) -> (f64, f64) {
    let c = 134217729.0 * a;
    let ahi = c - (c - a);
    let alo = a - ahi;
    (ahi, alo)
}

/// Two-Product (116.3d): `a * b = x + y` exactly. No FMA.
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let x = a * b;
    let (ahi, alo) = split(a);
    let (bhi, blo) = split(b);
    let y = alo * blo - (((x - ahi * bhi) - alo * bhi) - ahi * blo);
    (x, y)
}

/// Grow, a block of (116.3e): `q = b`; for each `e_i` in increasing
/// order, `(q, h_i) = Two-Sum(q, e_i)`, keep `h_i` if non-zero; finally
/// keep `q` if non-zero. The result sums exactly to `e + b` (his
/// Theorem 10 in section 2 of Shewchuk (1997)).
fn grow(e: &mut Vec<f64>, b: f64) {
    let mut q = b;
    let mut out: Vec<f64> = Vec::with_capacity(e.len() + 1);
    for &ei in e.iter() {
        let (nq, h) = two_sum(q, ei);
        if h != 0.0 {
            out.push(h);
        }
        q = nq;
    }
    if q != 0.0 {
        out.push(q);
    }
    *e = out;
}

/// Sum, a block of (116.3e): `h = e`; for each `f_j`, `h = Grow(h, f_j)`.
fn sum_exp(e: &[f64], f: &[f64]) -> Vec<f64> {
    let mut h = e.to_vec();
    for &fj in f.iter() {
        grow(&mut h, fj);
    }
    h
}

/// Product, a block of (116.3e): `h = 0`; for each `f_j`, for each
/// `e_i`: `(x, y) = Two-Product(e_i, f_j)`; `h = Grow(Grow(h, y), x)`.
fn product_exp(e: &[f64], f: &[f64]) -> Vec<f64> {
    let mut h: Vec<f64> = Vec::new();
    for &fj in f.iter() {
        for &ei in e.iter() {
            let (x, y) = two_product(ei, fj);
            grow(&mut h, y);
            grow(&mut h, x);
        }
    }
    h
}

/// Difference (116.3e): a difference of two inputs is the expansion of
/// Two-Diff, `y` first, then `x`, zeros dropped.
fn diff_exp(a: f64, b: f64) -> Vec<f64> {
    let (x, y) = two_diff(a, b);
    let mut e: Vec<f64> = Vec::with_capacity(2);
    if y != 0.0 {
        e.push(y);
    }
    if x != 0.0 {
        e.push(x);
    }
    e
}

/// The negation of an expansion: every component sign-flipped.
fn negate(e: &[f64]) -> Vec<f64> {
    e.iter().map(|v| -v).collect()
}

/// The difference of two expansions, as an expansion operation: the Sum
/// of `e` and the negation of `f` (116.3e).
fn sub_exp(e: &[f64], f: &[f64]) -> Vec<f64> {
    sum_exp(e, &negate(f))
}

/// The exact stage of `orient3d`: (116.1) evaluated in expansion
/// arithmetic by (116.3d) and (116.3e), with the same cofactor formula
/// as stage A. It never uses stage A. The result is the expansion's
/// last (largest-magnitude) component, or `0.0` if the expansion is
/// empty; its sign is the exact sign of the determinant.
pub fn orient3d_exact(a: &Point, b: &Point, c: &Point, d: &Point) -> f64 {
    let adx = diff_exp(a[0], d[0]);
    let ady = diff_exp(a[1], d[1]);
    let adz = diff_exp(a[2], d[2]);
    let bdx = diff_exp(b[0], d[0]);
    let bdy = diff_exp(b[1], d[1]);
    let bdz = diff_exp(b[2], d[2]);
    let cdx = diff_exp(c[0], d[0]);
    let cdy = diff_exp(c[1], d[1]);
    let cdz = diff_exp(c[2], d[2]);
    let t1 = sub_exp(&product_exp(&bdx, &cdy), &product_exp(&cdx, &bdy));
    let t2 = sub_exp(&product_exp(&cdx, &ady), &product_exp(&adx, &cdy));
    let t3 = sub_exp(&product_exp(&adx, &bdy), &product_exp(&bdx, &ady));
    let det = sum_exp(
        &sum_exp(&product_exp(&adz, &t1), &product_exp(&bdz, &t2)),
        &product_exp(&cdz, &t3),
    );
    det.last().copied().unwrap_or(0.0)
}

/// The exact stage of `insphere`: (116.3) evaluated in expansion
/// arithmetic by (116.3d) and (116.3e), with the same cofactor formula
/// as stage A. It never uses stage A. The result is the expansion's
/// last (largest-magnitude) component, or `0.0` if the expansion is
/// empty; its sign is the exact sign of the determinant.
pub fn insphere_exact(a: &Point, b: &Point, c: &Point, d: &Point, e: &Point) -> f64 {
    let aex = diff_exp(a[0], e[0]);
    let aey = diff_exp(a[1], e[1]);
    let aez = diff_exp(a[2], e[2]);
    let bex = diff_exp(b[0], e[0]);
    let bey = diff_exp(b[1], e[1]);
    let bez = diff_exp(b[2], e[2]);
    let cex = diff_exp(c[0], e[0]);
    let cey = diff_exp(c[1], e[1]);
    let cez = diff_exp(c[2], e[2]);
    let dex = diff_exp(d[0], e[0]);
    let dey = diff_exp(d[1], e[1]);
    let dez = diff_exp(d[2], e[2]);
    let ab = sub_exp(&product_exp(&aex, &bey), &product_exp(&bex, &aey));
    let bc = sub_exp(&product_exp(&bex, &cey), &product_exp(&cex, &bey));
    let cd = sub_exp(&product_exp(&cex, &dey), &product_exp(&dex, &cey));
    let da = sub_exp(&product_exp(&dex, &aey), &product_exp(&aex, &dey));
    let ac = sub_exp(&product_exp(&aex, &cey), &product_exp(&cex, &aey));
    let bd = sub_exp(&product_exp(&bex, &dey), &product_exp(&dex, &bey));
    let abc = sum_exp(
        &sub_exp(&product_exp(&aez, &bc), &product_exp(&bez, &ac)),
        &product_exp(&cez, &ab),
    );
    let bcd = sum_exp(
        &sub_exp(&product_exp(&bez, &cd), &product_exp(&cez, &bd)),
        &product_exp(&dez, &bc),
    );
    let cda = sum_exp(
        &sum_exp(&product_exp(&cez, &da), &product_exp(&dez, &ac)),
        &product_exp(&aez, &cd),
    );
    let dab = sum_exp(
        &sum_exp(&product_exp(&dez, &ab), &product_exp(&aez, &bd)),
        &product_exp(&bez, &da),
    );
    let alift = sum_exp(
        &sum_exp(&product_exp(&aex, &aex), &product_exp(&aey, &aey)),
        &product_exp(&aez, &aez),
    );
    let blift = sum_exp(
        &sum_exp(&product_exp(&bex, &bex), &product_exp(&bey, &bey)),
        &product_exp(&bez, &bez),
    );
    let clift = sum_exp(
        &sum_exp(&product_exp(&cex, &cex), &product_exp(&cey, &cey)),
        &product_exp(&cez, &cez),
    );
    let dlift = sum_exp(
        &sum_exp(&product_exp(&dex, &dex), &product_exp(&dey, &dey)),
        &product_exp(&dez, &dez),
    );
    let t = sub_exp(&product_exp(&dlift, &abc), &product_exp(&clift, &dab));
    let u = sub_exp(&product_exp(&blift, &cda), &product_exp(&alift, &bcd));
    let det = sum_exp(&t, &u);
    det.last().copied().unwrap_or(0.0)
}

/// The exact sign of (116.1): stage A, the filter of (116.3b), decides
/// when `det~ > bound || -det~ > bound`, and then `det~` itself is
/// returned unchanged; otherwise the exact stage of (116.3d) and
/// (116.3e) runs and its last component is returned.
pub fn orient3d(a: &Point, b: &Point, c: &Point, d: &Point) -> f64 {
    let (det, bound) = orient3d_stage_a(a, b, c, d);
    if det > bound || -det > bound {
        det
    } else {
        orient3d_exact(a, b, c, d)
    }
}

/// The exact sign of (116.3): stage A, the filter of (116.3c), decides
/// when `det~ > bound || -det~ > bound`, and then `det~` itself is
/// returned unchanged; otherwise the exact stage of (116.3d) and
/// (116.3e) runs and its last component is returned.
pub fn insphere(a: &Point, b: &Point, c: &Point, d: &Point, e: &Point) -> f64 {
    let (det, bound) = insphere_stage_a(a, b, c, d, e);
    if det > bound || -det > bound {
        det
    } else {
        insphere_exact(a, b, c, d, e)
    }
}

/// The domain (116.3f): `x == 0` or `2^-120 <= |x| <= 2^120`. NaN and
/// the two infinities are outside it.
pub fn in_exact_domain(x: f64) -> bool {
    if x == 0.0 {
        return true;
    }
    let m = x.abs();
    m >= f64::from_bits(0x3870000000000000) && m <= f64::from_bits(0x4770000000000000)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, y: f64, z: f64) -> Point {
        [x, y, z]
    }

    /// The sign of an f64, with 0 (and -0.0) its own sign.
    fn sgn(x: f64) -> i32 {
        if x > 0.0 {
            1
        } else if x < 0.0 {
            -1
        } else {
            0
        }
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
        fn u01(&mut self) -> f64 {
            (self.next() >> 11) as f64 * EPS
        }
    }

    /// The permutations of `[0, n)` in lexicographic order.
    fn perms(n: usize) -> Vec<Vec<usize>> {
        fn rec(cur: &mut Vec<usize>, rest: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
            if rest.is_empty() {
                out.push(cur.clone());
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
        let mut cur: Vec<usize> = Vec::new();
        let mut rest: Vec<usize> = (0..n).collect();
        rec(&mut cur, &mut rest, &mut out);
        out
    }

    fn perm4() -> Vec<[usize; 4]> {
        perms(4).into_iter().map(|v| [v[0], v[1], v[2], v[3]]).collect()
    }

    fn perm5() -> Vec<[usize; 5]> {
        perms(5)
            .into_iter()
            .map(|v| [v[0], v[1], v[2], v[3], v[4]])
            .collect()
    }

    /// The 6 axis permutations, lexicographic.
    const AXP: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    /// The 56 five-subsets of 8 corner indices, lexicographic.
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

    fn sub_i(a: &[i64; 3], b: &[i64; 3]) -> [i128; 3] {
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

    /// (116.1) on integer points, exactly in `i128`.
    fn o3_i128(a: &[i64; 3], b: &[i64; 3], c: &[i64; 3], d: &[i64; 3]) -> i128 {
        det3_i128(sub_i(a, d), sub_i(b, d), sub_i(c, d))
    }

    fn lift2(w: [i128; 3]) -> i128 {
        w[0] * w[0] + w[1] * w[1] + w[2] * w[2]
    }

    /// (116.3) on integer points, exactly in `i128`: the 4x4 determinant
    /// of the rows `[p - e, |p - e|^2]`, expanded by cofactors of its
    /// last column.
    fn is_i128(a: &[i64; 3], b: &[i64; 3], c: &[i64; 3], d: &[i64; 3], e: &[i64; 3]) -> i128 {
        let ae = sub_i(a, e);
        let be = sub_i(b, e);
        let ce = sub_i(c, e);
        let de = sub_i(d, e);
        let a2 = lift2(ae);
        let b2 = lift2(be);
        let c2 = lift2(ce);
        let d2 = lift2(de);
        -a2 * det3_i128(be, ce, de) + b2 * det3_i128(ae, ce, de)
            - c2 * det3_i128(ae, be, de) + d2 * det3_i128(ae, be, ce)
    }

    /// A minimal signed big integer: a sign and a little-endian `u32`
    /// magnitude, normalised (no high zero limbs; zero has an empty
    /// magnitude and a positive sign).
    #[derive(Clone)]
    struct Big {
        neg: bool,
        mag: Vec<u32>,
    }

    impl Big {
        fn zero() -> Big {
            Big {
                neg: false,
                mag: Vec::new(),
            }
        }
        fn norm(mut mag: Vec<u32>) -> Vec<u32> {
            while let Some(&0) = mag.last() {
                mag.pop();
            }
            mag
        }
        /// `m * 2^shift`, with sign `neg`; `m >= 0`.
        fn from_shift(m: u64, shift: u32, neg: bool) -> Big {
            let v = (m as u128) << (shift % 32);
            let mut mag = vec![0u32; (shift / 32) as usize];
            mag.push(v as u32);
            mag.push((v >> 32) as u32);
            mag.push((v >> 64) as u32);
            let mag = Big::norm(mag);
            if mag.is_empty() {
                Big::zero()
            } else {
                Big { neg, mag }
            }
        }
        fn cmp_mag(a: &[u32], b: &[u32]) -> std::cmp::Ordering {
            if a.len() != b.len() {
                return a.len().cmp(&b.len());
            }
            for i in (0..a.len()).rev() {
                if a[i] != b[i] {
                    return a[i].cmp(&b[i]);
                }
            }
            std::cmp::Ordering::Equal
        }

        fn add_mag(a: &[u32], b: &[u32]) -> Vec<u32> {
            let mut out = Vec::with_capacity(a.len().max(b.len()) + 1);
            let mut carry = 0u64;
            for i in 0..a.len().max(b.len()) {
                let s = carry
                    + *a.get(i).unwrap_or(&0) as u64
                    + *b.get(i).unwrap_or(&0) as u64;
                out.push(s as u32);
                carry = s >> 32;
            }
            if carry != 0 {
                out.push(carry as u32);
            }
            out
        }
        /// `a - b` with `a >= b`.
        fn sub_mag(a: &[u32], b: &[u32]) -> Vec<u32> {
            let mut out = Vec::with_capacity(a.len());
            let mut borrow = 0i64;
            for i in 0..a.len() {
                let d = a[i] as i64 - *b.get(i).unwrap_or(&0) as i64 - borrow;
                if d < 0 {
                    out.push((d + (1i64 << 32)) as u32);
                    borrow = 1;
                } else {
                    out.push(d as u32);
                    borrow = 0;
                }
            }
            Big::norm(out)
        }
        fn mul_mag(a: &[u32], b: &[u32]) -> Vec<u32> {
            let mut out = vec![0u32; a.len() + b.len()];
            for i in 0..a.len() {
                let mut carry = 0u64;
                for j in 0..b.len() {
                    let t = out[i + j] as u64 + a[i] as u64 * b[j] as u64 + carry;
                    out[i + j] = t as u32;
                    carry = t >> 32;
                }
                let mut k = i + b.len();
                while carry != 0 {
                    let t = out[k] as u64 + carry;
                    out[k] = t as u32;
                    carry = t >> 32;
                    k += 1;
                }
            }
            Big::norm(out)
        }

        fn add(&self, o: &Big) -> Big {
            if self.neg == o.neg {
                let mag = Big::add_mag(&self.mag, &o.mag);
                if mag.is_empty() {
                    Big::zero()
                } else {
                    Big { neg: self.neg, mag }
                }
            } else {
                match Big::cmp_mag(&self.mag, &o.mag) {
                    std::cmp::Ordering::Equal => Big::zero(),
                    std::cmp::Ordering::Greater => {
                        let mag = Big::sub_mag(&self.mag, &o.mag);
                        Big { neg: self.neg, mag }
                    }
                    std::cmp::Ordering::Less => {
                        let mag = Big::sub_mag(&o.mag, &self.mag);
                        Big { neg: o.neg, mag }
                    }
                }
            }
        }
        fn negated(&self) -> Big {
            if self.mag.is_empty() {
                Big::zero()
            } else {
                Big {
                    neg: !self.neg,
                    mag: self.mag.clone(),
                }
            }
        }
        fn sub(&self, o: &Big) -> Big {
            self.add(&o.negated())
        }
        fn mul(&self, o: &Big) -> Big {
            if self.mag.is_empty() || o.mag.is_empty() {
                return Big::zero();
            }
            let mag = Big::mul_mag(&self.mag, &o.mag);
            Big {
                neg: self.neg != o.neg,
                mag,
            }
        }
        fn signum(&self) -> i32 {
            if self.mag.is_empty() {
                0
            } else if self.neg {
                -1
            } else {
                1
            }
        }
    }

    /// `x = m * 2^e` with `m` a signed integer, `|m| < 2^53`, for finite
    /// `x` (zero gives `m = 0`).
    fn decompose(x: f64) -> (i64, i32) {
        let bits = x.to_bits();
        if bits & 0x7FFF_FFFF_FFFF_FFFF == 0 {
            return (0, 0);
        }
        let ef = ((bits >> 52) & 0x7ff) as i32;
        let f = (bits & ((1u64 << 52) - 1)) as i64;
        let (m, e) = if ef == 0 {
            (f, -1074)
        } else {
            (f | (1i64 << 52), ef - 1075)
        };
        if bits >> 63 == 1 {
            (-m, e)
        } else {
            (m, e)
        }
    }

    /// The smallest exponent over a query's nonzero coordinates.
    fn emin_of(coords: &[f64]) -> i32 {
        let mut emin = i32::MAX;
        for &x in coords {
            if x == 0.0 {
                continue;
            }
            let (_, e) = decompose(x);
            if e < emin {
                emin = e;
            }
        }
        emin
    }

    /// `x` as the integer `m * 2^(e - Emin)`: every coordinate of the
    /// query scaled by the same power of two, so every determinant keeps
    /// its sign.
    fn to_big(x: f64, emin: i32) -> Big {
        let (m, e) = decompose(x);
        Big::from_shift(m.unsigned_abs(), (e - emin) as u32, m < 0)
    }

    fn det3_big(r0: &[Big; 3], r1: &[Big; 3], r2: &[Big; 3]) -> Big {
        let t0 = r1[1].mul(&r2[2]).sub(&r1[2].mul(&r2[1]));
        let t1 = r1[0].mul(&r2[2]).sub(&r1[2].mul(&r2[0]));
        let t2 = r1[0].mul(&r2[1]).sub(&r1[1].mul(&r2[0]));
        r0[0].mul(&t0).sub(&r0[1].mul(&t1)).add(&r0[2].mul(&t2))
    }

    /// (116.1) on general f64 points, exactly, via the big integers of
    /// `decompose`/`to_big`.
    fn o3_big(pts: &[Point; 4]) -> Big {
        let flat: Vec<f64> = pts.iter().flat_map(|q| q.iter().copied()).collect();
        let emin = emin_of(&flat);
        let g: Vec<Big> = flat.iter().map(|&x| to_big(x, emin)).collect();
        let sub = |i: usize| -> [Big; 3] {
            [
                g[3 * i].sub(&g[9]),
                g[3 * i + 1].sub(&g[10]),
                g[3 * i + 2].sub(&g[11]),
            ]
        };
        det3_big(&sub(0), &sub(1), &sub(2))
    }

    /// (116.3) on general f64 points, exactly, via the big integers of
    /// `decompose`/`to_big`.
    fn is_big(pts: &[Point; 5]) -> Big {
        let flat: Vec<f64> = pts.iter().flat_map(|q| q.iter().copied()).collect();
        let emin = emin_of(&flat);
        let g: Vec<Big> = flat.iter().map(|&x| to_big(x, emin)).collect();
        let sub = |i: usize| -> [Big; 3] {
            [
                g[3 * i].sub(&g[12]),
                g[3 * i + 1].sub(&g[13]),
                g[3 * i + 2].sub(&g[14]),
            ]
        };
        let ae = sub(0);
        let be = sub(1);
        let ce = sub(2);
        let de = sub(3);
        let sq = |w: &[Big; 3]| -> Big {
            w[0].mul(&w[0]).add(&w[1].mul(&w[1])).add(&w[2].mul(&w[2]))
        };
        let a2 = sq(&ae);
        let b2 = sq(&be);
        let c2 = sq(&ce);
        let d2 = sq(&de);
        a2.mul(&det3_big(&be, &ce, &de))
            .negated()
            .add(&b2.mul(&det3_big(&ae, &ce, &de)))
            .sub(&c2.mul(&det3_big(&ae, &be, &de)))
            .add(&d2.mul(&det3_big(&ae, &be, &ce)))
    }

    /// O1, seed 1: 30000 orient3d queries — sheared, axis-permuted,
    /// argument-reordered.
    fn gen_o1(r: &mut SplitMix64, p4: &[[usize; 4]]) -> [[i64; 3]; 4] {
        let a = [
            r.range(-(1 << 18), 1 << 18),
            r.range(-(1 << 18), 1 << 18),
            r.range(-(1 << 18), 1 << 18),
        ];
        let pv = r.range(-512, 512);
        let rv = r.range(-512, 512);
        let q = r.range(-32, 32);
        let s = r.range(-16, 16);
        let (al, be, ga, de) = (q, 1i64, q * s - 1, s);
        let x = r.range(-256, 256);
        let y = r.range(-256, 256);
        let w = r.range(-1, 2);
        let e1 = [1i64, 0, pv];
        let e2 = [0i64, 1, rv];
        let mut b = [0i64; 3];
        let mut c = [0i64; 3];
        let mut d = [0i64; 3];
        for k in 0..3 {
            b[k] = a[k] + al * e1[k] + be * e2[k];
            c[k] = a[k] + ga * e1[k] + de * e2[k];
            d[k] = a[k] + x * e1[k] + y * e2[k];
        }
        d[2] += w;
        let k1 = 2 * r.range(0, 2) - 1;
        let k2 = 2 * r.range(0, 2) - 1;
        let mut pts = [a, b, c, d];
        for v in pts.iter_mut() {
            *v = [v[0] + k1 * v[2], v[1] + k2 * v[2], v[2]];
        }
        let pi = AXP[r.range(0, 6) as usize];
        for v in pts.iter_mut() {
            *v = [v[pi[0]], v[pi[1]], v[pi[2]]];
        }
        let sg = p4[r.range(0, 24) as usize];
        [pts[sg[0]], pts[sg[1]], pts[sg[2]], pts[sg[3]]]
    }

    /// O2, seed 2: 10000 orient3d queries — three points on a common
    /// line plus a free point.
    fn gen_o2(r: &mut SplitMix64, p4: &[[usize; 4]]) -> [[i64; 3]; 4] {
        let a = [
            r.range(-(1 << 18), 1 << 18),
            r.range(-(1 << 18), 1 << 18),
            r.range(-(1 << 18), 1 << 18),
        ];
        let u = [r.range(-256, 256), r.range(-256, 256), r.range(-256, 256)];
        let k1 = r.range(-512, 512);
        let k2 = r.range(-512, 512);
        let v = [r.range(-1, 2), r.range(-1, 2), r.range(-1, 2)];
        let b = [a[0] + k1 * u[0], a[1] + k1 * u[1], a[2] + k1 * u[2]];
        let c = [
            a[0] + k2 * u[0] + v[0],
            a[1] + k2 * u[1] + v[1],
            a[2] + k2 * u[2] + v[2],
        ];
        let d = [
            r.range(-(1 << 18), 1 << 18),
            r.range(-(1 << 18), 1 << 18),
            r.range(-(1 << 18), 1 << 18),
        ];
        let sg = p4[r.range(0, 24) as usize];
        let pts = [a, b, c, d];
        [pts[sg[0]], pts[sg[1]], pts[sg[2]], pts[sg[3]]]
    }

    /// I1, seed 3: 30000 insphere queries — signed axis permutations of
    /// one vector about a common origin.
    fn gen_i1(r: &mut SplitMix64) -> [[i64; 3]; 5] {
        let v = [
            r.range(1, 1 << 18),
            r.range(1, 1 << 18),
            r.range(1, 1 << 18),
        ];
        let o = [
            r.range(-(1 << 18), 1 << 18),
            r.range(-(1 << 18), 1 << 18),
            r.range(-(1 << 18), 1 << 18),
        ];
        let mut ids: Vec<i64> = Vec::with_capacity(5);
        while ids.len() < 5 {
            let k = r.range(0, 48);
            if !ids.contains(&k) {
                ids.push(k);
            }
        }
        let mut pts = [[0i64; 3]; 5];
        for i in 0..5 {
            let k = ids[i];
            let pi = AXP[(k / 8) as usize];
            let m = k % 8;
            let s0 = if (m >> 0) & 1 == 1 { -1i64 } else { 1 };
            let s1 = if (m >> 1) & 1 == 1 { -1i64 } else { 1 };
            let s2 = if (m >> 2) & 1 == 1 { -1i64 } else { 1 };
            pts[i] = [
                o[0] + s0 * v[pi[0]],
                o[1] + s1 * v[pi[1]],
                o[2] + s2 * v[pi[2]],
            ];
        }
        if r.next() & 1 == 1 {
            let w = [r.range(-1, 2), r.range(-1, 2), r.range(-1, 2)];
            for j in 0..3 {
                pts[4][j] += w[j];
            }
        }
        pts
    }

    /// I2, seed 4: 20000 insphere queries — an integer `L*U` lattice.
    fn gen_i2(r: &mut SplitMix64, p5: &[[usize; 5]]) -> [[i64; 3]; 5] {
        let t = [
            r.range(-(1 << 17), 1 << 17),
            r.range(-(1 << 17), 1 << 17),
            r.range(-(1 << 17), 1 << 17),
        ];
        let u12 = r.range(-512, 512);
        let u13 = r.range(-512, 512);
        let u23 = r.range(-512, 512);
        let l21 = r.range(-512, 512);
        let l31 = r.range(-512, 512);
        let l32 = r.range(-512, 512);
        let c1 = [1i64, l21, l31];
        let c2 = [u12, l21 * u12 + 1, l31 * u12 + l32];
        let c3 = [u13, l21 * u13 + u23, l31 * u13 + l32 * u23 + 1];
        let mut pts = [[0i64; 3]; 5];
        for i in 0..5 {
            let x = r.range(-1, 2);
            let y = r.range(-1, 2);
            pts[i] = [
                t[0] + x * c1[0] + y * c2[0],
                t[1] + x * c1[1] + y * c2[1],
                t[2] + x * c1[2] + y * c2[2],
            ];
        }
        let w = r.range(-1, 2);
        for j in 0..3 {
            pts[4][j] += w * c3[j];
        }
        let pi = AXP[r.range(0, 6) as usize];
        for v in pts.iter_mut() {
            *v = [v[pi[0]], v[pi[1]], v[pi[2]]];
        }
        let sg = p5[r.range(0, 120) as usize];
        [pts[sg[0]], pts[sg[1]], pts[sg[2]], pts[sg[3]], pts[sg[4]]]
    }

    /// I3, seed 5: 10000 insphere queries — five corners of one cube of
    /// side `s`.
    fn gen_i3(r: &mut SplitMix64) -> [[i64; 3]; 5] {
        let s = r.range(1, 1 << 18);
        let o = [
            r.range(-(1 << 18), 1 << 18),
            r.range(-(1 << 18), 1 << 18),
            r.range(-(1 << 18), 1 << 18),
        ];
        let mut ids: Vec<i64> = Vec::with_capacity(5);
        while ids.len() < 5 {
            let k = r.range(0, 8);
            if !ids.contains(&k) {
                ids.push(k);
            }
        }
        let mut pts = [[0i64; 3]; 5];
        for i in 0..5 {
            pts[i] = [
                o[0] + s * (ids[i] & 1),
                o[1] + s * ((ids[i] >> 1) & 1),
                o[2] + s * ((ids[i] >> 2) & 1),
            ];
        }
        pts
    }

    fn u3(r: &mut SplitMix64) -> f64 {
        2.0 * r.u01() - 1.0
    }

    /// G1, seed 6: 10000 general-f64 orient3d queries — a fourth point
    /// in the plane of a triangle, optionally shifted far away.
    fn gen_g1(r: &mut SplitMix64) -> [Point; 4] {
        let a = [u3(r), u3(r), u3(r)];
        let b = [u3(r), u3(r), u3(r)];
        let c = [u3(r), u3(r), u3(r)];
        let s = r.u01();
        let t = r.u01();
        let mut d = [0.0f64; 3];
        for k in 0..3 {
            d[k] = (a[k] + s * (b[k] - a[k])) + t * (c[k] - a[k]);
        }
        let mut pts = [a, b, c, d];
        if r.next() & 1 == 1 {
            let tt = [1e6 * r.u01(), 1e6 * r.u01(), 1e6 * r.u01()];
            for q in pts.iter_mut() {
                for k in 0..3 {
                    q[k] += tt[k];
                }
            }
        }
        pts
    }

    /// G2, seed 7: 10000 general-f64 insphere queries — five points on a
    /// common sphere, optionally shifted far away.
    fn gen_g2(r: &mut SplitMix64) -> [Point; 5] {
        let c0 = [u3(r), u3(r), u3(r)];
        let rad = 0.5 + r.u01();
        let mut pts = [[0.0f64; 3]; 5];
        for i in 0..5 {
            let g = [u3(r), u3(r), u3(r)];
            let n = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
            pts[i] = [
                c0[0] + rad * (g[0] / n),
                c0[1] + rad * (g[1] / n),
                c0[2] + rad * (g[2] / n),
            ];
        }
        if r.next() & 1 == 1 {
            let tt = [1e6 * r.u01(), 1e6 * r.u01(), 1e6 * r.u01()];
            for q in pts.iter_mut() {
                for k in 0..3 {
                    q[k] += tt[k];
                }
            }
        }
        pts
    }

    /// Integer query to f64, scaled by `sc` (a power of two, so exact).
    fn fp4(q: &[[i64; 3]; 4], sc: f64) -> [Point; 4] {
        let one = |v: &[i64; 3]| -> Point {
            [v[0] as f64 * sc, v[1] as f64 * sc, v[2] as f64 * sc]
        };
        [one(&q[0]), one(&q[1]), one(&q[2]), one(&q[3])]
    }

    /// Integer query to f64, scaled by `sc` (a power of two, so exact).
    fn fp5(q: &[[i64; 3]; 5], sc: f64) -> [Point; 5] {
        let one = |v: &[i64; 3]| -> Point {
            [v[0] as f64 * sc, v[1] as f64 * sc, v[2] as f64 * sc]
        };
        [one(&q[0]), one(&q[1]), one(&q[2]), one(&q[3]), one(&q[4])]
    }

    /// One orient3d query against the `i128` oracle at the three scales.
    /// Adds (zero, undecided, nz_undecided) at scale 1 to `cnt`.
    fn check_or4(q: &[[i64; 3]; 4], scales: &[f64; 3], cnt: &mut (usize, usize, usize)) {
        let os = o3_i128(&q[0], &q[1], &q[2], &q[3]);
        let osn = if os > 0 {
            1i32
        } else if os < 0 {
            -1
        } else {
            0
        };
        let mut dec1 = false;
        for si in 0..3 {
            let fp = fp4(q, scales[si]);
            let (det, bound) = orient3d_stage_a(&fp[0], &fp[1], &fp[2], &fp[3]);
            let dec = det > bound || -det > bound;
            if si == 0 {
                dec1 = dec;
            } else {
                assert_eq!(dec, dec1, "stage A must decide at 2^+-40 iff at 1");
            }
            if si == 0 {
                if dec {
                    assert_eq!(sgn(det), osn, "filter sign on an integer query");
                }
                let v = orient3d(&fp[0], &fp[1], &fp[2], &fp[3]);
                let ve = orient3d_exact(&fp[0], &fp[1], &fp[2], &fp[3]);
                assert_eq!(sgn(v), osn, "orient3d vs i128 oracle");
                assert_eq!(sgn(ve), osn, "orient3d_exact vs i128 oracle");
                if os == 0 {
                    assert_eq!(v, 0.0, "orient3d must be exactly 0.0");
                    assert_eq!(ve, 0.0, "orient3d_exact must be exactly 0.0");
                    cnt.0 += 1;
                }
                if !dec {
                    cnt.1 += 1;
                    if os != 0 {
                        cnt.2 += 1;
                    }
                }
            }
        }
    }

    /// One insphere query against the `i128` oracle at the three scales.
    /// Adds (zero, undecided, nz_undecided) at scale 1 to `cnt`.
    fn check_or5(q: &[[i64; 3]; 5], scales: &[f64; 3], cnt: &mut (usize, usize, usize)) {
        let os = is_i128(&q[0], &q[1], &q[2], &q[3], &q[4]);
        let osn = if os > 0 {
            1i32
        } else if os < 0 {
            -1
        } else {
            0
        };
        let mut dec1 = false;
        for si in 0..3 {
            let fp = fp5(q, scales[si]);
            let (det, bound) = insphere_stage_a(&fp[0], &fp[1], &fp[2], &fp[3], &fp[4]);
            let dec = det > bound || -det > bound;
            if si == 0 {
                dec1 = dec;
            } else {
                assert_eq!(dec, dec1, "stage A must decide at 2^+-40 iff at 1");
            }
            if si == 0 {
                if dec {
                    assert_eq!(sgn(det), osn, "filter sign on an integer query");
                }
                let v = insphere(&fp[0], &fp[1], &fp[2], &fp[3], &fp[4]);
                let ve = insphere_exact(&fp[0], &fp[1], &fp[2], &fp[3], &fp[4]);
                assert_eq!(sgn(v), osn, "insphere vs i128 oracle");
                assert_eq!(sgn(ve), osn, "insphere_exact vs i128 oracle");
                if os == 0 {
                    assert_eq!(v, 0.0, "insphere must be exactly 0.0");
                    assert_eq!(ve, 0.0, "insphere_exact must be exactly 0.0");
                    cnt.0 += 1;
                }
                if !dec {
                    cnt.1 += 1;
                    if os != 0 {
                        cnt.2 += 1;
                    }
                }
            }
        }
    }

    fn all_under_2p20_i(q: &[[i64; 3]]) -> bool {
        q.iter().all(|v| v.iter().all(|&x| x.abs() < (1i64 << 20)))
    }

    #[test]
    fn eps_and_bounds() {
        assert_eq!(EPS.to_bits(), 0x3CA0000000000000);
        assert_eq!(EPS, 2f64.powi(-53));
        assert_eq!(O3D_BOUND_A, (7.0 + 56.0 * EPS) * EPS);
        assert_eq!(ISP_BOUND_A, (16.0 + 224.0 * EPS) * EPS);
        assert_eq!(O3D_BOUND_A, 7.771561172376103e-16);
        assert_eq!(ISP_BOUND_A, 1.7763568394002532e-15);
    }

    #[test]
    fn primitives_are_exact() {
        let x = 1.0 + 2f64.powi(-30);
        let (p1, p2) = two_product(x, x);
        assert_eq!(p1, 1.0 + 2f64.powi(-29));
        assert_eq!(p2, 2f64.powi(-60));
        let (q1, q2) = two_product(0.1, 0.3);
        assert_eq!(q1, 0.03);
        assert_eq!(q2, 1.6653345369377347e-18);
        let (s1, s2) = two_sum(1.0, 2f64.powi(-60));
        assert_eq!(s1, 1.0);
        assert_eq!(s2, 2f64.powi(-60));
        let (d1, d2) = two_diff(1.0, 2f64.powi(-60));
        assert_eq!(d1, 1.0);
        assert_eq!(d2, -2f64.powi(-60));
        let (h1, h2) = split(x);
        assert_eq!(h1, 1.0);
        assert_eq!(h2, 2f64.powi(-30));
        let (g1, g2) = split(0.1);
        assert_eq!(g1, 0.09999999962747097);
        assert_eq!(g2, 3.7252903539730653e-10);
        let mut e = vec![2f64.powi(-60)];
        grow(&mut e, 1.0);
        assert_eq!(e, vec![2f64.powi(-60), 1.0]);
        let mut f = vec![1.0];
        grow(&mut f, -1.0);
        assert_eq!(f, Vec::<f64>::new());
    }

    #[test]
    fn convention_unit_tetrahedron() {
        let v0 = p(0.0, 0.0, 0.0);
        let v1 = p(1.0, 0.0, 0.0);
        let v2 = p(0.0, 0.0, 1.0);
        let v3 = p(0.0, 1.0, 0.0);
        assert_eq!(orient3d(&v0, &v1, &v2, &v3), 1.0);
        assert_eq!(
            orient3d(
                &p(0.0, 0.0, 0.0),
                &p(1.0, 0.0, 0.0),
                &p(0.0, 1.0, 0.0),
                &p(0.0, 0.0, 1.0)
            ),
            -1.0
        );
        assert_eq!(insphere(&v0, &v1, &v2, &v3, &p(0.25, 0.25, 0.25)), 0.5625);
        assert_eq!(insphere(&v0, &v1, &v2, &v3, &p(5.0, 5.0, 5.0)), -60.0);
        // The four outward-wound faces of the table in §116.4, each
        // against its opposite vertex.
        assert_eq!(orient3d(&v0, &v1, &v2, &v3), 1.0);
        assert_eq!(orient3d(&v0, &v3, &v1, &v2), 1.0);
        assert_eq!(orient3d(&v0, &v2, &v3, &v1), 1.0);
        assert_eq!(orient3d(&v1, &v3, &v2, &v0), 1.0);
        assert_eq!(orient3d_exact(&v0, &v1, &v2, &v3), 1.0);
        assert_eq!(orient3d_exact(&v0, &v3, &v1, &v2), 1.0);
        assert_eq!(orient3d_exact(&v0, &v2, &v3, &v1), 1.0);
        assert_eq!(orient3d_exact(&v1, &v3, &v2, &v0), 1.0);
        assert_eq!(insphere_exact(&v0, &v1, &v2, &v3, &p(0.25, 0.25, 0.25)), 0.5625);
        assert_eq!(insphere_exact(&v0, &v1, &v2, &v3, &p(5.0, 5.0, 5.0)), -60.0);
    }

    #[test]
    fn matches_exact_i128() {
        let p4 = perm4();
        let p5 = perm5();
        let scales: [f64; 3] = [1.0, 2f64.powi(-40), 2f64.powi(40)];
        let mut c = [(0usize, 0usize, 0usize); 5];
        // O1, seed 1.
        let mut r = SplitMix64::new(1);
        for i in 0..30000 {
            let q = gen_o1(&mut r, &p4);
            if i == 0 {
                assert_eq!(
                    q,
                    [
                        [293041, -208808, -101449],
                        [341570, -257021, -149811],
                        [300809, -216545, -109218],
                        [386012, -301408, -194432],
                    ]
                );
            }
            assert!(all_under_2p20_i(&q));
            check_or4(&q, &scales, &mut c[0]);
        }
        // O2, seed 2.
        let mut r = SplitMix64::new(2);
        for _ in 0..10000 {
            let q = gen_o2(&mut r, &p4);
            assert!(all_under_2p20_i(&q));
            check_or4(&q, &scales, &mut c[1]);
        }
        // I1, seed 3.
        let mut r = SplitMix64::new(3);
        for i in 0..30000 {
            let q = gen_i1(&mut r);
            if i == 0 {
                assert_eq!(
                    q,
                    [
                        [31743, 322474, 444279],
                        [31743, 8486, 130291],
                        [31743, 149378, 444279],
                        [31743, 8486, 303387],
                        [213567, 322474, 171223],
                    ]
                );
            }
            assert!(all_under_2p20_i(&q));
            check_or5(&q, &scales, &mut c[2]);
        }
        // I2, seed 4.
        let mut r = SplitMix64::new(4);
        for _ in 0..20000 {
            let q = gen_i2(&mut r, &p5);
            assert!(all_under_2p20_i(&q));
            check_or5(&q, &scales, &mut c[3]);
        }
        // I3, seed 5.
        let mut r = SplitMix64::new(5);
        for i in 0..10000 {
            let q = gen_i3(&mut r);
            if i == 0 {
                assert_eq!(
                    q,
                    [
                        [284786, -16057, 187071],
                        [210680, -16057, 187071],
                        [284786, -16057, 112965],
                        [284786, 58049, 112965],
                        [210680, -16057, 112965],
                    ]
                );
            }
            assert!(all_under_2p20_i(&q));
            check_or5(&q, &scales, &mut c[4]);
        }
        let zero = c[0].0 + c[1].0 + c[2].0 + c[3].0 + c[4].0;
        let und = c[0].1 + c[1].1 + c[2].1 + c[3].1 + c[4].1;
        let nz_und = c[0].2 + c[1].2 + c[2].2 + c[3].2 + c[4].2;
        println!(
            "TET01-I128 O1 zero={} und={} | O2 zero={} und={} | I1 zero={} und={} | I2 zero={} und={} | I3 zero={} und={} | total zero={} und={} nz_und={}",
            c[0].0, c[0].1, c[1].0, c[1].1, c[2].0, c[2].1, c[3].0, c[3].1, c[4].0, c[4].1,
            zero, und, nz_und
        );
        assert!(zero >= 45000);
        assert!(und >= 48000);
        assert!(nz_und >= 3000);
    }

    #[test]
    fn cospherical_is_zero() {
        let p5 = perm5();
        let cases: [([i64; 3], i64); 3] = [
            ([0, 0, 0], 1),
            ([524287, -524287, 300001], 1),
            ([-1000, 2000, -3000], 3),
        ];
        for (o, s) in cases.iter() {
            let mut corners: [Point; 8] = [[0.0; 3]; 8];
            for i in 0..8usize {
                corners[i] = [
                    (o[0] + s * (i as i64 & 1)) as f64,
                    (o[1] + s * ((i as i64 >> 1) & 1)) as f64,
                    (o[2] + s * ((i as i64 >> 2) & 1)) as f64,
                ];
            }
            for sub in combos5().iter() {
                let base = [
                    corners[sub[0]],
                    corners[sub[1]],
                    corners[sub[2]],
                    corners[sub[3]],
                    corners[sub[4]],
                ];
                for sg in p5.iter() {
                    let q = [
                        base[sg[0]],
                        base[sg[1]],
                        base[sg[2]],
                        base[sg[3]],
                        base[sg[4]],
                    ];
                    assert_eq!(insphere(&q[0], &q[1], &q[2], &q[3], &q[4]), 0.0);
                }
            }
        }
        // Eight points of the sphere of radius 7, each shifted.
        let sh = [123457.0f64, -654321.0, 999000.0];
        let raw: [[f64; 3]; 8] = [
            [7.0, 0.0, 0.0],
            [0.0, 7.0, 0.0],
            [0.0, 0.0, 7.0],
            [2.0, 3.0, 6.0],
            [6.0, -2.0, 3.0],
            [-3.0, 6.0, -2.0],
            [-7.0, 0.0, 0.0],
            [2.0, -6.0, -3.0],
        ];
        for sub in combos5().iter() {
            let q: Vec<Point> = sub
                .iter()
                .map(|&i| [raw[i][0] + sh[0], raw[i][1] + sh[1], raw[i][2] + sh[2]])
                .collect();
            assert_eq!(insphere(&q[0], &q[1], &q[2], &q[3], &q[4]), 0.0);
        }
        // Three collinear points, plain and shifted.
        assert_eq!(
            orient3d(
                &p(0.0, 0.0, 0.0),
                &p(1.0, 2.0, 3.0),
                &p(4.0, 5.0, 6.0),
                &p(7.0, 8.0, 9.0)
            ),
            0.0
        );
        let o = [1000003.0f64, -999999.0, 524288.0];
        assert_eq!(
            orient3d(
                &p(o[0], o[1], o[2]),
                &p(1.0 + o[0], 2.0 + o[1], 3.0 + o[2]),
                &p(4.0 + o[0], 5.0 + o[1], 6.0 + o[2]),
                &p(7.0 + o[0], 8.0 + o[1], 9.0 + o[2])
            ),
            0.0
        );
        assert_eq!(
            orient3d(
                &p(0.0, 0.0, 0.0),
                &p(1.0, 1.0, 1.0),
                &p(2.0, 2.0, 2.0),
                &p(5.0, -3.0, 8.0)
            ),
            0.0
        );
        // Five coplanar points, shifted.
        let co = [777777.0f64, -333333.0, 999999.0];
        assert_eq!(
            insphere(
                &p(co[0], co[1], co[2]),
                &p(1.0 + co[0], co[1], co[2]),
                &p(co[0], 1.0 + co[1], co[2]),
                &p(1.0 + co[0], 1.0 + co[1], co[2]),
                &p(3.0 + co[0], 7.0 + co[1], co[2])
            ),
            0.0
        );
    }

    #[test]
    fn f64_under_single() {
        let t40 = 2f64.powi(-40);
        assert_eq!(((1.0f64 + t40) as f32) as f64, 1.0);
        assert_eq!(((1.0f64 - t40) as f32) as f64, 1.0);
        let a = p(1.0, 1.0, 1.0);
        let b = p(2.0, 1.0, 1.0);
        let c = p(1.0, 2.0, 1.0);
        let d4 = p(1.0, 1.0, 2.0);
        assert_eq!(orient3d(&a, &b, &c, &p(1.5, 1.5, 1.0 + t40)), -t40);
        assert_eq!(orient3d(&a, &b, &c, &p(1.5, 1.5, 1.0 - t40)), t40);
        assert_eq!(orient3d(&a, &b, &c, &d4), -1.0);
        assert_eq!(insphere(&a, &b, &c, &d4, &p(2.0, 2.0, 1.0)), 0.0);
        assert!(insphere(&a, &b, &c, &d4, &p(2.0, 2.0, 1.0 + t40)) < 0.0);
        assert!(insphere(&a, &b, &c, &d4, &p(2.0, 2.0, 1.0 - t40)) > 0.0);
        assert_eq!(orient3d_exact(&a, &b, &c, &p(1.5, 1.5, 1.0 + t40)), -t40);
        assert_eq!(orient3d_exact(&a, &b, &c, &p(1.5, 1.5, 1.0 - t40)), t40);
        assert_eq!(orient3d_exact(&a, &b, &c, &d4), -1.0);
        assert_eq!(insphere_exact(&a, &b, &c, &d4, &p(2.0, 2.0, 1.0)), 0.0);
        assert!(insphere_exact(&a, &b, &c, &d4, &p(2.0, 2.0, 1.0 + t40)) < 0.0);
        assert!(insphere_exact(&a, &b, &c, &d4, &p(2.0, 2.0, 1.0 - t40)) > 0.0);
        assert_eq!(std::mem::size_of_val(&orient3d(&a, &b, &c, &d4)), 8);
    }

    #[test]
    fn matches_bigint_general_f64() {
        // G1, seed 6.
        let mut und1 = 0usize;
        let mut nw1 = 0usize;
        let mut r = SplitMix64::new(6);
        for i in 0..10000 {
            let q = gen_g1(&mut r);
            if i == 0 {
                assert_eq!(q[0][0].to_bits(), 0x4108450580C921BF);
            }
            for v in q.iter() {
                for &x in v.iter() {
                    assert!(in_exact_domain(x), "G1 coordinate outside (116.3f)");
                }
            }
            let os = o3_big(&q).signum();
            let (det, bound) = orient3d_stage_a(&q[0], &q[1], &q[2], &q[3]);
            let dec = det > bound || -det > bound;
            if dec {
                let v = orient3d(&q[0], &q[1], &q[2], &q[3]);
                assert_eq!(
                    v.to_bits(),
                    det.to_bits(),
                    "a decided filter must return det~ unchanged"
                );
            } else {
                und1 += 1;
            }
            if sgn(det) != os || (det == 0.0 && os != 0) {
                nw1 += 1;
            }
            let v = orient3d(&q[0], &q[1], &q[2], &q[3]);
            let ve = orient3d_exact(&q[0], &q[1], &q[2], &q[3]);
            assert_eq!(sgn(v), os, "orient3d vs big-integer oracle");
            assert_eq!(sgn(ve), os, "orient3d_exact vs big-integer oracle");
        }
        // G2, seed 7.
        let mut und2 = 0usize;
        let mut nw2 = 0usize;
        let mut r = SplitMix64::new(7);
        for _ in 0..10000 {
            let q = gen_g2(&mut r);
            for v in q.iter() {
                for &x in v.iter() {
                    assert!(in_exact_domain(x), "G2 coordinate outside (116.3f)");
                }
            }
            let os = is_big(&q).signum();
            let (det, bound) = insphere_stage_a(&q[0], &q[1], &q[2], &q[3], &q[4]);
            let dec = det > bound || -det > bound;
            if dec {
                let v = insphere(&q[0], &q[1], &q[2], &q[3], &q[4]);
                assert_eq!(
                    v.to_bits(),
                    det.to_bits(),
                    "a decided filter must return det~ unchanged"
                );
            } else {
                und2 += 1;
            }
            if sgn(det) != os || (det == 0.0 && os != 0) {
                nw2 += 1;
            }
            let v = insphere(&q[0], &q[1], &q[2], &q[3], &q[4]);
            let ve = insphere_exact(&q[0], &q[1], &q[2], &q[3], &q[4]);
            assert_eq!(sgn(v), os, "insphere vs big-integer oracle");
            assert_eq!(sgn(ve), os, "insphere_exact vs big-integer oracle");
        }
        println!(
            "TET01-BIG G1 und={} naive_wrong={} | G2 und={} naive_wrong={}",
            und1, nw1, und2, nw2
        );
        assert!(und1 >= 4000);
        assert!(nw1 >= 800);
        assert!(und2 >= 4000);
        assert!(nw2 >= 1400);
    }

    #[test]
    fn transpositions_negate() {
        // G1, first 2000 queries, all 6 swaps of two arguments.
        let mut r = SplitMix64::new(6);
        for _ in 0..2000 {
            let q = gen_g1(&mut r);
            let bs = sgn(orient3d(&q[0], &q[1], &q[2], &q[3]));
            for i in 0..4 {
                for j in i + 1..4 {
                    let mut s = q;
                    s.swap(i, j);
                    assert_eq!(
                        sgn(orient3d(&s[0], &s[1], &s[2], &s[3])),
                        -bs,
                        "a swap must negate orient3d"
                    );
                }
            }
        }
        // G2, first 2000 queries, all 10 swaps.
        let mut r = SplitMix64::new(7);
        for _ in 0..2000 {
            let q = gen_g2(&mut r);
            let bs = sgn(insphere(&q[0], &q[1], &q[2], &q[3], &q[4]));
            for i in 0..5 {
                for j in i + 1..5 {
                    let mut s = q;
                    s.swap(i, j);
                    assert_eq!(
                        sgn(insphere(&s[0], &s[1], &s[2], &s[3], &s[4])),
                        -bs,
                        "a swap must negate insphere"
                    );
                }
            }
        }
        // O1 and I2, the integer points as f64.
        let p4 = perm4();
        let p5 = perm5();
        let mut r = SplitMix64::new(1);
        for _ in 0..2000 {
            let q = gen_o1(&mut r, &p4);
            let f = fp4(&q, 1.0);
            let bs = sgn(orient3d(&f[0], &f[1], &f[2], &f[3]));
            for i in 0..4 {
                for j in i + 1..4 {
                    let mut s = f;
                    s.swap(i, j);
                    assert_eq!(
                        sgn(orient3d(&s[0], &s[1], &s[2], &s[3])),
                        -bs,
                        "a swap must negate orient3d"
                    );
                }
            }
        }
        let mut r = SplitMix64::new(4);
        for _ in 0..2000 {
            let q = gen_i2(&mut r, &p5);
            let f = fp5(&q, 1.0);
            let bs = sgn(insphere(&f[0], &f[1], &f[2], &f[3], &f[4]));
            for i in 0..5 {
                for j in i + 1..5 {
                    let mut s = f;
                    s.swap(i, j);
                    assert_eq!(
                        sgn(insphere(&s[0], &s[1], &s[2], &s[3], &s[4])),
                        -bs,
                        "a swap must negate insphere"
                    );
                }
            }
        }
    }

    #[test]
    fn exact_domain_bounds() {
        assert!(in_exact_domain(0.0));
        assert!(in_exact_domain(-0.0));
        assert!(in_exact_domain(2f64.powi(-120)));
        assert!(in_exact_domain(-2f64.powi(-120)));
        assert!(in_exact_domain(2f64.powi(120)));
        assert!(in_exact_domain(1.0));
        assert!(in_exact_domain(-1e6));
        assert!(!in_exact_domain(2f64.powi(-121)));
        assert!(!in_exact_domain(2f64.powi(121)));
        assert!(!in_exact_domain(f64::MIN_POSITIVE));
        assert!(!in_exact_domain(f64::NAN));
        assert!(!in_exact_domain(f64::INFINITY));
        assert!(!in_exact_domain(f64::NEG_INFINITY));
    }
}
