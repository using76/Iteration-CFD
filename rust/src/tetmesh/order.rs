// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The BRIO insertion order of SPEC-LIT §116.7.
//!
//! Provenance: ORIGINAL. Written from Amenta, Choi & Rote (2003), Proc. 19th SoCG 211-219, and Skilling (2004), AIP Conf. Proc. 707:381-387. No GPL-licensed source was consulted.

use super::Point;

/// The Hilbert grid's bits per axis, `B` of (116.6c): the key grid holds
/// `2^B` cells per axis, `8^B` cells in all, so a `u64` carries every key.
pub const BRIO_BITS: u32 = 16;

/// The first round's size, `c_1` of (116.6d).
pub const BRIO_FIRST: usize = 64;

/// The shuffle's SplitMix64 seed (§116.7, Rounds).
pub const BRIO_SEED: u64 = 2003;

/// One step of SplitMix64 (Steele, Lea & Flood 2014), all operations
/// wrapping - the same stream the kernel's walk draws.
fn splitmix_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// The Hilbert key of every point, (116.6c) then `H_b` of (116.6b): the
/// per-axis bounding box maps `x_k` to a cell of the `2^bits`-per-axis
/// grid, and [`crate::decompose::hilbert_index`] reads the key off the
/// cell. A non-finite coordinate cannot panic - the f64-to-u32 cast
/// saturates and maps NaN to 0 - it only makes the key meaningless;
/// `Delaunay::build_brio` refuses the set afterwards, exactly as `build`
/// does.
///
/// Panics when `bits` is outside `1..=21`, the transform's range.
pub fn hilbert_keys(points: &[Point], bits: u32) -> Vec<u64> {
    assert!(
        (1..=21).contains(&bits),
        "hilbert_keys: bits {} outside 1..=21",
        bits
    );
    let two_pow = 2_u64.pow(bits) as f64;
    let cap = (2_u64.pow(bits) - 1) as u32;
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in points {
        for k in 0..3 {
            lo[k] = f64::min(lo[k], p[k]);
            hi[k] = f64::max(hi[k], p[k]);
        }
    }
    points
        .iter()
        .map(|p| {
            let mut q = [0_u32; 3];
            for k in 0..3 {
                if hi[k] > lo[k] {
                    // (116.6c) in exactly this order: subtract, divide,
                    // multiply by 2^bits, cast - then the cap.
                    let v = ((p[k] - lo[k]) / (hi[k] - lo[k]) * two_pow) as u32;
                    q[k] = v.min(cap);
                }
            }
            crate::decompose::hilbert_index(q, bits)
        })
        .collect()
}

/// The round boundaries of (116.6d): `c_0 = 0`, `c_1 = min(n, 64)`, then
/// each `c_(k+1) = min(n, 2 c_k)` until `c_k = n`. `n = 0` gives `[0]`.
pub fn brio_rounds(n: usize) -> Vec<usize> {
    let mut rounds = vec![0_usize];
    if n > 0 {
        let mut c = n.min(BRIO_FIRST);
        rounds.push(c);
        while c < n {
            c = n.min(c.saturating_mul(2));
            rounds.push(c);
        }
    }
    rounds
}

/// The BRIO insertion order (§116.7): one Fisher-Yates shuffle of
/// `pi = (0, 1, ..., n-1)`, for `i = n-1` down to `1` swapping `pi_i`
/// with `pi_j`, `j = s mod (i + 1)` for the next SplitMix64 value `s`
/// of the stream seeded `seed`; the prefix blocks of [`brio_rounds`]
/// then sort by the pair `(key, index)` ascending, the key the
/// [`BRIO_BITS`]-bit Hilbert key of (116.6c), and concatenate. The
/// result is a permutation of `0..n` for every input, non-finite
/// coordinates included, and a function of the seed alone.
pub fn brio_order(points: &[Point], seed: u64) -> Vec<u32> {
    let n = points.len();
    let keys = hilbert_keys(points, BRIO_BITS);
    let mut pi: Vec<u32> = (0..n as u32).collect();
    let mut state = seed;
    for i in (1..n).rev() {
        let j = (splitmix_next(&mut state) % (i as u64 + 1)) as usize;
        pi.swap(i, j);
    }
    let mut order = Vec::with_capacity(n);
    for w in brio_rounds(n).windows(2) {
        pi[w[0]..w[1]].sort_unstable_by_key(|&i| (keys[i as usize], i));
        order.extend_from_slice(&pi[w[0]..w[1]]);
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompose::hilbert_index;
    use crate::tetmesh::delaunay::Delaunay;

    /// SplitMix64 (Steele, Lea & Flood 2014), all operations wrapping -
    /// the same stream the §116.5 tests use.
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
        (0..n)
            .map(|_| [draw_u(&mut r), draw_u(&mut r), draw_u(&mut r)])
            .collect()
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

    /// FNV-1a 64 over the bytes (offset 0xcbf29ce484222325, prime
    /// 0x100000001b3).
    fn fnv64(bytes: &[u8]) -> u64 {
        let mut h = 0xcbf29ce484222325_u64;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// FNV-1a 64 over the order, each index as 4 little-endian bytes.
    fn order_digest(order: &[u32]) -> u64 {
        let mut bytes = Vec::with_capacity(order.len() * 4);
        for i in order {
            bytes.extend_from_slice(&i.to_le_bytes());
        }
        fnv64(&bytes)
    }

    /// The transform §116.7 buys: a bijection cells -> [0, 8^bits) whose
    /// consecutive indices are face-adjacent cells (L1 distance 1).
    #[test]
    fn hilbert_is_bijective() {
        for bits in 1..=5_u32 {
            let side = 1_u32 << bits;
            let cells = 8_usize.pow(bits);
            let mut seen = vec![false; cells];
            let mut inv = vec![[0_u32; 3]; cells];
            for z in 0..side {
                for y in 0..side {
                    for x in 0..side {
                        let h = hilbert_index([x, y, z], bits) as usize;
                        assert!(h < cells, "bits {}: index {} out of range", bits, h);
                        assert!(!seen[h], "bits {}: index {} repeats", bits, h);
                        seen[h] = true;
                        inv[h] = [x, y, z];
                    }
                }
            }
            assert!(seen.iter().all(|&s| s), "bits {}: not onto", bits);
            assert_eq!(hilbert_index([0, 0, 0], bits), 0);
            assert_eq!(inv[cells - 1], [side - 1, 0, 0]);
            for h in 0..cells - 1 {
                let (a, b) = (inv[h], inv[h + 1]);
                let d = a[0].abs_diff(b[0]) + a[1].abs_diff(b[1]) + a[2].abs_diff(b[2]);
                assert_eq!(d, 1, "bits {}: cells {}, {} not unit steps", bits, h, h + 1);
            }
        }
        // bits 1, cells by index.
        let by_index: Vec<[u32; 3]> = (0..8)
            .map(|h| {
                let mut c = [0_u32; 3];
                for z in 0..2 {
                    for y in 0..2 {
                        for x in 0..2 {
                            if hilbert_index([x, y, z], 1) == h {
                                c = [x, y, z];
                            }
                        }
                    }
                }
                c
            })
            .collect();
        assert_eq!(
            by_index,
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 1],
                [0, 1, 0],
                [1, 1, 0],
                [1, 1, 1],
                [1, 0, 1],
                [1, 0, 0]
            ]
        );
        assert_eq!(hilbert_index([1, 2, 3], 2), 22);
        assert_eq!(hilbert_index([5, 0, 7], 3), 430);
        assert_eq!(hilbert_index([12345, 54321, 777], 16), 131844176086295);
        assert_eq!(hilbert_index([65535, 65535, 65535], 16), 201053554793325);
        assert_eq!(
            hilbert_index([1048575, 0, 1048575], 21),
            617636520325096594
        );
        assert_eq!(
            hilbert_index([2097151, 1234567, 1362865], 21),
            6783307986849049521
        );
    }

    /// The boundaries of (116.6d), pinned.
    #[test]
    fn brio_rounds_doubling() {
        let eq = |n: usize, want: &[usize]| assert_eq!(brio_rounds(n), want);
        eq(0, &[0]);
        eq(1, &[0, 1]);
        eq(4, &[0, 4]);
        eq(63, &[0, 63]);
        eq(64, &[0, 64]);
        eq(65, &[0, 64, 65]);
        eq(128, &[0, 64, 128]);
        eq(129, &[0, 64, 128, 129]);
        eq(1000, &[0, 64, 128, 256, 512, 1000]);
        let m = brio_rounds(1_000_000);
        assert_eq!(m.len(), 16);
        assert_eq!(m[0], 0);
        for k in 1..15 {
            assert_eq!(m[k], 1_usize << (k + 5), "round {}", k);
        }
        assert_eq!(m[15], 1_000_000);
    }

    /// The shuffle, the rounds and the keys, pinned against an
    /// independent reading of §116.7.
    #[test]
    fn brio_order_pinned() {
        let pts = r_points(1000, 2026);
        let o = brio_order(&pts, BRIO_SEED);
        assert_eq!(o[..8], [131, 894, 673, 190, 846, 977, 241, 885]);
        assert_eq!(o[o.len() - 4..], [538, 867, 726, 644]);
        assert_eq!(order_digest(&o), 0xed3e42d856fec369);
        assert_eq!(
            &hilbert_keys(&pts, 16)[..3],
            &[214692658940882, 91781408820783, 163476583310020]
        );

        let pts = r_points(10000, 2026);
        let o = brio_order(&pts, BRIO_SEED);
        assert_eq!(o[..8], [8052, 3438, 9860, 3528, 4039, 4364, 843, 7750]);
        assert_eq!(order_digest(&o), 0x085d3e9a151f63c5);
        let o7 = brio_order(&pts, 7);
        assert_eq!(o7[..8], [2496, 1713, 1961, 3871, 2337, 2386, 7611, 6954]);
        assert_eq!(order_digest(&o7), 0xdb164054b75f64dd);

        let l3 = lattice(3);
        let o = brio_order(&l3, BRIO_SEED);
        assert_eq!(
            o,
            [
                0, 9, 18, 12, 21, 24, 15, 6, 3, 4, 5, 8, 7, 13, 22, 25, 16, 17, 26, 23, 14, 10,
                19, 20, 11, 1, 2
            ]
        );
        assert_eq!(order_digest(&o), 0x4a2afd7c88e3665e);
        assert_eq!(
            &hilbert_keys(&l3, 16)[..5],
            &[0, 265139375383698, 281474976710655, 135711149485494, 140737488355328]
        );
    }

    /// Every order is a permutation, and inside each round the pairs
    /// `(key, index)` strictly increase.
    #[test]
    fn brio_is_permutation() {
        fn check(pts: &[Point]) {
            let n = pts.len();
            let o = brio_order(pts, 99);
            let mut seen = vec![false; n];
            for &i in &o {
                assert!((i as usize) < n, "index {} out of 0..{}", i, n);
                assert!(!seen[i as usize], "index {} repeats", i);
                seen[i as usize] = true;
            }
            assert!(seen.iter().all(|&s| s), "not a permutation of 0..{}", n);
            let keys = hilbert_keys(pts, BRIO_BITS);
            for w in brio_rounds(n).windows(2) {
                for p in w[0]..w[1].saturating_sub(1) {
                    let a = (keys[o[p] as usize], o[p]);
                    let b = (keys[o[p + 1] as usize], o[p + 1]);
                    assert!(a < b, "round {} not sorted at {}", w[0], p);
                }
            }
        }
        for n in 0..=10 {
            check(&r_points(n, 99));
        }
        for n in [63, 64, 65, 129] {
            check(&r_points(n, 99));
        }
        for m in 2..=4 {
            check(&lattice(m));
        }
    }

    /// Non-finite coordinates only make keys meaningless: no panic, still
    /// a permutation, and `build_brio` refuses the set exactly as `build`
    /// does.
    #[test]
    fn brio_nonfinite_no_panic() {
        let mut pts: Vec<Point> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        pts.push([f64::NAN, 0.5, 0.5]);
        pts.push([0.5, f64::INFINITY, 0.5]);
        pts.push([0.5, 0.5, f64::NEG_INFINITY]);
        let o = brio_order(&pts, BRIO_SEED);
        let mut seen = vec![false; pts.len()];
        for &i in &o {
            seen[i as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));

        // All x equal: q_x = 0, so the keys equal those of the x = 0 set.
        let a: Vec<Point> = (0..20).map(|i| [2.5, i as f64, (i % 7) as f64]).collect();
        let b: Vec<Point> = a.iter().map(|p| [0.0, p[1], p[2]]).collect();
        assert_eq!(hilbert_keys(&a, 16), hilbert_keys(&b, 16));

        // The NaN fixture of delaunay's `refusals` test.
        let five: Vec<Point> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, f64::NAN, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let want = Delaunay::build(&five).unwrap_err().to_string();
        let got = Delaunay::build_brio(&five).unwrap_err().to_string();
        assert_eq!(got, want);
        assert!(got.contains("delaunay/domain: point 2 axis y"), "{}", got);
    }
}
