// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! Tests of the `Comm` contract and of `ThreadComm` (SPEC-LIT §138.1.5).
//!
//! Every test is class D: exact bytes and exact counts, the same in the
//! default build and in `--features single`, because nothing here touches a
//! `Scalar`. Nothing here is a tolerance.
//!
//! The message bytes come from two public-domain generators, each written
//! from its published definition and checked against its published test
//! vectors in the first test:
//!
//! * SplitMix64, from Steele, Lea & Flood, *OOPSLA 2014* 453-472, DOI
//!   10.1145/2660193.2660195, constants as published, public domain;
//! * FNV-1a 64, Fowler/Noll/Vo's public-domain hash, from its published
//!   definition and test vectors.
//!
//! A payload is a function of `(src, dst, tag, len)` alone, so a rank can
//! compute what it should have received without asking anyone.
//!
//! This file uses only the public items of `comm` (`Comm`, `ThreadComm`,
//! `MAX_TAG`, `DEFAULT_TIMEOUT`) and `std`, and writes `std::thread::...` in
//! full: `use super::*;` brings the module `comm::thread` into scope, and a
//! bare `thread::scope` would resolve to that.
//!
//! Provenance: ORIGINAL - the tests are ours (docs/17 PAR-01); only the two
//! generators above are third-party, both public domain. No MPI
//! implementation's source was opened. No GPL-licensed source was consulted.

use super::*;
use crate::error::Error;
use crate::Result;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

// ==========================================================================
//  The generators
// ==========================================================================

/// SplitMix64: all arithmetic is wrapping `u64`.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// FNV-1a, 64 bit.
fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for &x in bytes {
        h = (h ^ u64::from(x)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The bytes rank `src` sends to rank `dst` under `tag`: SplitMix64 seeded
/// with `(src << 40) | (dst << 20) | tag`, little-endian, cut to `len`.
fn payload(src: usize, dst: usize, tag: u32, len: usize) -> Vec<u8> {
    let mut g = SplitMix64(((src as u64) << 40) | ((dst as u64) << 20) | u64::from(tag));
    let mut out = Vec::with_capacity(len + 8);
    while out.len() < len {
        out.extend_from_slice(&g.next_u64().to_le_bytes());
    }
    out.truncate(len);
    out
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The length of the message rank `src` sends rank `dst` in the asymmetric
/// graph.
fn msg_len(src: usize, dst: usize) -> usize {
    1 + (7 * src + 13 * dst) % 97
}

/// The asymmetric graph: the peers rank `r` sends to in a world of `n`.
/// Rank 0 has no peer at all, rank 1 has five, and the rest form a chain
/// that also points back at rank 1.
fn sends_of(r: usize, n: usize) -> Vec<usize> {
    match r {
        0 => Vec::new(),
        1 => (2..=6).filter(|&d| d < n).collect(),
        _ => {
            let mut v = Vec::new();
            if r <= 6 {
                v.push(1);
            }
            if r + 1 < n {
                v.push(r + 1);
            }
            v
        }
    }
}

/// The peers rank `r` receives from, ascending.
fn recvs_of(r: usize, n: usize) -> Vec<usize> {
    (0..n).filter(|&s| sends_of(s, n).contains(&r)).collect()
}

// ==========================================================================
//  Shared helpers
// ==========================================================================

/// One round of the asymmetric exchange under `tag`. Odd ranks list their
/// sends in descending peer order and their receives ascending; even ranks
/// the other way round, so the order of a list is shown not to matter.
/// Returns what arrived, as `(source, bytes)` in ascending source order, and
/// asserts every buffer against the payload it should hold.
fn asym_round(c: &dyn Comm, n: usize, tag: u32) -> Result<Vec<(usize, Vec<u8>)>> {
    let r = c.rank();
    let mut send_peers = sends_of(r, n);
    let mut recv_peers = recvs_of(r, n);
    if r % 2 == 1 {
        send_peers.reverse();
    } else {
        recv_peers.reverse();
    }
    let out: Vec<Vec<u8>> = send_peers
        .iter()
        .map(|&d| payload(r, d, tag, msg_len(r, d)))
        .collect();
    let sends: Vec<(usize, &[u8])> = send_peers
        .iter()
        .zip(&out)
        .map(|(&d, b)| (d, b.as_slice()))
        .collect();
    let mut bufs: Vec<Vec<u8>> = recv_peers
        .iter()
        .map(|&s| vec![0u8; msg_len(s, r)])
        .collect();
    let mut recvs: Vec<(usize, &mut [u8])> = recv_peers
        .iter()
        .zip(bufs.iter_mut())
        .map(|(&s, b)| (s, b.as_mut_slice()))
        .collect();
    c.exchange(tag, &sends, &mut recvs)?;
    let mut got: Vec<(usize, Vec<u8>)> = recv_peers.into_iter().zip(bufs).collect();
    got.sort_by_key(|(s, _)| *s);
    for (s, b) in &got {
        assert!(
            *b == payload(*s, r, tag, msg_len(*s, r)),
            "rank {r}: the buffer from rank {s} under tag {tag} is not its payload"
        );
    }
    Ok(got)
}

/// The message of an `Err(Error::Config(..))` that starts `comm: `. Anything
/// else fails the test.
fn config_message<T: std::fmt::Debug>(r: Result<T>) -> String {
    match r {
        Err(Error::Config(m)) => {
            assert!(
                m.starts_with("comm: "),
                "message does not start `comm: `: {m:?}"
            );
            m
        }
        other => panic!("expected Err(Error::Config(..)), got {other:?}"),
    }
}

fn assert_names(message: &str, needles: &[&str]) {
    for needle in needles {
        assert!(
            message.contains(needle),
            "message {message:?} does not contain {needle:?}"
        );
    }
}

/// Run `f` on every rank of a world of `n` and require every rank's `Err` to
/// be a `comm: ` message naming every needle. Returns the messages.
fn every_rank_refuses<F>(n: usize, needles: &[&str], f: F) -> Vec<String>
where
    F: Fn(&ThreadComm) -> Result<()> + Sync,
{
    let results = ThreadComm::run(n, f).expect("the world runs");
    assert_eq!(results.len(), n);
    results
        .into_iter()
        .map(|r| {
            let m = config_message(r);
            assert_names(&m, needles);
            m
        })
        .collect()
}

// ==========================================================================
//  T0
// ==========================================================================

#[test]
fn the_test_generators_are_the_published_ones() {
    let mut g = SplitMix64(0);
    assert_eq!(g.next_u64(), 0xe220_a839_7b1d_cdaf);
    assert_eq!(g.next_u64(), 0x6e78_9e6a_a1b9_65f4);
    assert_eq!(g.next_u64(), 0x06c4_5d18_8009_454f);

    assert_eq!(fnv(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv(b"foobar"), 0x8594_4171_f739_67e8);

    assert_eq!(
        hex(&payload(1, 2, 0, 16)),
        "429bd6e2aae4768a0f5bd1b80494bd3c"
    );
}

// ==========================================================================
//  T1
// ==========================================================================

#[test]
fn allgather_returns_rank_order() {
    // 1. Order, for every world size.
    for n in 1..=8usize {
        let res = ThreadComm::run(n, |c| {
            let mine = payload(c.rank(), 1000, 3, 24);
            let mut all = vec![0u8; 24 * n];
            c.allgather(&mine, &mut all).unwrap();
            all
        })
        .unwrap();
        let expect: Vec<u8> = (0..n).flat_map(|q| payload(q, 1000, 3, 24)).collect();
        for (r, all) in res.iter().enumerate() {
            assert!(
                *all == expect,
                "n {n}: rank {r}'s allgather is not rank order"
            );
        }
        if n == 8 {
            assert_eq!(res[0].len(), 192);
            assert_eq!(fnv(&res[0]), 0x1933_ebdb_fd35_1aab);
        }
    }

    // 2. Exact bytes.
    let res = ThreadComm::run(4, |c| {
        let mine = [c.rank() as u8; 3];
        let mut all = [0xffu8; 12];
        c.allgather(&mine, &mut all).unwrap();
        all
    })
    .unwrap();
    for all in &res {
        assert_eq!(all, &[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]);
    }

    // 3. Local refusal: every rank passes the same wrong `all`.
    every_rank_refuses(4, &["97", "96"], |c| {
        c.allgather(&[0u8; 24], &mut [0u8; 97])
    });

    // 4. Length mismatch: rank 1 passes 23 bytes, the others 24.
    let msgs = every_rank_refuses(3, &["23", "24"], |c| {
        if c.rank() == 1 {
            c.allgather(&[1u8; 23], &mut [0u8; 69])
        } else {
            c.allgather(&[2u8; 24], &mut [0u8; 72])
        }
    });
    assert_names(&msgs[0], &["rank 1"]);
    assert_names(&msgs[2], &["rank 1"]);
}

// ==========================================================================
//  T2
// ==========================================================================

#[test]
fn allgatherv_ragged_lengths() {
    for n in 1..=8usize {
        let res = ThreadComm::run(n, |c| {
            let r = c.rank();
            c.allgatherv(&payload(r, 1001, 4, (5 * r) % 7)).unwrap()
        })
        .unwrap();
        for (r, got) in res.iter().enumerate() {
            assert_eq!(got.len(), n, "n {n}: rank {r}");
            for (q, block) in got.iter().enumerate() {
                assert!(
                    *block == payload(q, 1001, 4, (5 * q) % 7),
                    "n {n}: rank {r}'s entry {q} is not rank {q}'s block"
                );
            }
            assert!(
                *got == res[0],
                "n {n}: rank {r}'s result differs from rank 0's"
            );
        }
        if n == 8 {
            let lens: Vec<usize> = res[0].iter().map(Vec::len).collect();
            assert_eq!(lens, [0, 5, 3, 1, 6, 4, 2, 0]);
            let all: Vec<u8> = res[0].concat();
            assert_eq!(all.len(), 21);
            assert_eq!(fnv(&all), 0x61c5_31f3_62ff_e81a);
        }
        if n == 1 {
            assert_eq!(res[0].len(), 1);
            assert!(res[0][0].is_empty());
        }
    }
}

// ==========================================================================
//  T3
// ==========================================================================

#[test]
fn exchange_asymmetric_neighbour_lists_never_deadlock() {
    let t0 = Instant::now();
    for n in 1..=8usize {
        let res = ThreadComm::run_with_timeout(n, Duration::from_secs(10), |c| {
            (0..3u32)
                .map(|tag| asym_round(c, n, tag).unwrap())
                .collect::<Vec<_>>()
        })
        .unwrap();
        // Digest order: rank, then tag, then ascending source.
        let mut digest = Vec::new();
        for (r, rounds) in res.iter().enumerate() {
            assert_eq!(rounds.len(), 3);
            for (tag, got) in rounds.iter().enumerate() {
                let sources: Vec<usize> = got.iter().map(|(s, _)| *s).collect();
                assert_eq!(sources, recvs_of(r, n), "n {n}: rank {r}, tag {tag}");
                for (s, b) in got {
                    assert!(
                        *b == payload(*s, r, tag as u32, msg_len(*s, r)),
                        "n {n}: rank {r} got wrong bytes from rank {s} under tag {tag}"
                    );
                    digest.extend_from_slice(b);
                }
            }
        }
        match n {
            6 => {
                assert_eq!(digest.len(), 1770);
                assert_eq!(fnv(&digest), 0xb24d_3d9d_5c5a_7090);
            }
            8 => {
                assert_eq!(digest.len(), 2358);
                assert_eq!(fnv(&digest), 0x39f2_e858_f877_cc65);
            }
            _ => {}
        }
        // The graph itself: rank 0 has no peer, rank 1 has five each way.
        if n == 7 || n == 8 {
            assert!(sends_of(0, n).is_empty() && recvs_of(0, n).is_empty());
            assert_eq!(sends_of(1, n).len(), 5);
            assert_eq!(recvs_of(1, n).len(), 5);
        }
    }
    assert!(
        t0.elapsed() < Duration::from_secs(10),
        "took {:?}",
        t0.elapsed()
    );
}

// ==========================================================================
//  T4
// ==========================================================================

/// What one rank saw in the three rounds of T4.
struct ByteRounds {
    all_to_all: Vec<Vec<u8>>,
    second: Option<Vec<u8>>,
    third: Option<Vec<u8>>,
}

fn byte_rounds(c: &dyn Comm, n: usize) -> ByteRounds {
    let r = c.rank();
    // Round 1, tag 7: every rank sends 100000 bytes to every rank, itself
    // included, and receives from every rank.
    let out: Vec<Vec<u8>> = (0..n).map(|q| payload(r, q, 7, 100_000)).collect();
    let sends: Vec<(usize, &[u8])> = out
        .iter()
        .enumerate()
        .map(|(q, b)| (q, b.as_slice()))
        .collect();
    let mut bufs: Vec<Vec<u8>> = (0..n).map(|_| vec![0u8; 100_000]).collect();
    let mut recvs: Vec<(usize, &mut [u8])> = bufs
        .iter_mut()
        .enumerate()
        .map(|(q, b)| (q, b.as_mut_slice()))
        .collect();
    c.exchange(7, &sends, &mut recvs).unwrap();

    // Round 2, tag 0: only rank 0 sends, only rank 1 receives.
    let mut second = None;
    if n >= 2 {
        match r {
            0 => {
                let p = payload(0, 1, 0, 100_000);
                c.exchange(0, &[(1, p.as_slice())], &mut []).unwrap();
            }
            1 => {
                let mut b = vec![0u8; 100_000];
                c.exchange(0, &[], &mut [(0, b.as_mut_slice())]).unwrap();
                second = Some(b);
            }
            _ => c.exchange(0, &[], &mut []).unwrap(),
        }
    }

    // Round 3, the largest tag: rank 7 to rank 6.
    let mut third = None;
    if n == 8 {
        match r {
            7 => {
                let p = payload(7, 6, MAX_TAG, 100_000);
                c.exchange(MAX_TAG, &[(6, p.as_slice())], &mut []).unwrap();
            }
            6 => {
                let mut b = vec![0u8; 100_000];
                c.exchange(MAX_TAG, &[], &mut [(7, b.as_mut_slice())])
                    .unwrap();
                third = Some(b);
            }
            _ => c.exchange(MAX_TAG, &[], &mut []).unwrap(),
        }
    }
    ByteRounds {
        all_to_all: bufs,
        second,
        third,
    }
}

#[test]
fn exchange_is_byte_identity() {
    for n in 1..=8usize {
        let res = ThreadComm::run(n, |c| byte_rounds(c, n)).unwrap();
        for (r, rounds) in res.iter().enumerate() {
            for (q, b) in rounds.all_to_all.iter().enumerate() {
                assert!(
                    *b == payload(q, r, 7, 100_000),
                    "n {n}: rank {r}'s buffer from rank {q} differs"
                );
            }
        }
        if n >= 6 {
            assert_eq!(fnv(&res[5].all_to_all[3]), 0xca96_5103_59a7_95e6);
        }
        if n >= 2 {
            let b = res[1].second.as_ref().expect("rank 1 received in round 2");
            assert!(*b == payload(0, 1, 0, 100_000));
            assert_eq!(fnv(b), 0x51b7_fdb4_04f5_73bd);
        }
        if n == 8 {
            let b = res[6].third.as_ref().expect("rank 6 received in round 3");
            assert!(*b == payload(7, 6, MAX_TAG, 100_000));
            assert_eq!(fnv(b), 0x4e6f_90c6_2bd2_4e7e);
        }
    }
}

// ==========================================================================
//  T5
// ==========================================================================

#[test]
fn bcast_from_every_root() {
    // 1. Every root, N sequential broadcasts in one world.
    for n in 1..=8usize {
        let res = ThreadComm::run(n, |c| {
            let r = c.rank();
            let mut seen = Vec::new();
            for root in 0..n {
                let mut buf = if r == root {
                    payload(root, 1002, 5, 1000)
                } else {
                    vec![0u8; 1000]
                };
                let before = buf.clone();
                c.bcast(root, &mut buf).unwrap();
                if r == root {
                    assert!(buf == before, "the root's own buffer changed");
                }
                seen.push(buf);
            }
            seen
        })
        .unwrap();
        for (r, seen) in res.iter().enumerate() {
            for (root, buf) in seen.iter().enumerate() {
                assert!(
                    *buf == payload(root, 1002, 5, 1000),
                    "n {n}: rank {r} did not get root {root}'s bytes"
                );
            }
        }
        assert_eq!(fnv(&res[0][0]), 0x415f_a11a_74cf_d84a);
        if n == 8 {
            assert_eq!(fnv(&res[3][7]), 0x271f_f184_2b60_ee20);
        }
    }

    // 2. Root out of range: every rank passes root 3 in a world of 3.
    every_rank_refuses(3, &["root 3", "size 3"], |c| c.bcast(3, &mut [0u8; 8]));

    // 3. Length mismatch: rank 2 passes 999 bytes, the others 1000.
    every_rank_refuses(3, &["999", "1000"], |c| {
        if c.rank() == 2 {
            c.bcast(0, &mut [0u8; 999])
        } else {
            c.bcast(0, &mut [0u8; 1000])
        }
    });
}

// ==========================================================================
//  T6
// ==========================================================================

/// The program of T6: three asymmetric exchange rounds, an allgather, an
/// allgatherv and a broadcast, with the scheduler perturbed before each of
/// the six operations. Returns everything this rank received, in order.
fn program(c: &dyn Comm, n: usize, rep: usize) -> Vec<u8> {
    let r = c.rank();
    let jitter = |op: usize| {
        for _ in 0..((rep * 7 + r * 3 + op) % 5) {
            std::thread::yield_now();
        }
    };
    let mut o = Vec::new();
    for tag in 0..3u32 {
        jitter(tag as usize);
        for (_, b) in asym_round(c, n, tag).unwrap() {
            o.extend_from_slice(&b);
        }
    }
    jitter(3);
    let mut all = vec![0u8; 24 * n];
    c.allgather(&payload(r, 1000, 3, 24), &mut all).unwrap();
    o.extend_from_slice(&all);
    jitter(4);
    for v in c.allgatherv(&payload(r, 1001, 4, (5 * r) % 7)).unwrap() {
        o.extend_from_slice(&v);
    }
    jitter(5);
    let root = n - 1;
    let mut buf = if r == root {
        payload(root, 1002, 5, 64)
    } else {
        vec![0u8; 64]
    };
    c.bcast(root, &mut buf).unwrap();
    o.extend_from_slice(&buf);
    o
}

#[test]
fn a_thousand_runs_produce_identical_bytes() {
    // (bytes, D(N)) for N = 1..=8, from the independent oracle.
    const EXPECT: [(usize, u64); 8] = [
        (88, 0x410f_8f0a_4284_d037),
        (234, 0xb991_fe33_bbce_0ec7),
        (618, 0x54c7_50f5_7eb6_de7f),
        (1270, 0x1394_b8a5_e535_6558),
        (2117, 0xca7a_ee5a_a3e1_b481),
        (3132, 0xf228_d3ea_2871_ed1a),
        (4018, 0xa609_6b18_ace7_f85e),
        (4574, 0xcaa0_cded_0010_ae47),
    ];
    let t0 = Instant::now();
    let (mut worlds, mut threads) = (0usize, 0usize);
    for rep in 0..1000usize {
        for n in 1..=8usize {
            let outs = ThreadComm::run(n, |c| program(c, n, rep)).unwrap();
            let all: Vec<u8> = outs.concat();
            let (bytes, digest) = EXPECT[n - 1];
            assert_eq!(all.len(), bytes, "n {n}, rep {rep}: byte count");
            assert_eq!(fnv(&all), digest, "n {n}, rep {rep}: digest");
            worlds += 1;
            threads += n;
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    println!("[PAR-01] {worlds} worlds, {threads} rank threads, {secs:.1} s");
    assert_eq!((worlds, threads), (8000, 36000));
    if !cfg!(debug_assertions) {
        assert!(
            secs < 60.0,
            "the release build took {secs:.1} s, the gate is 60 s"
        );
    }
}

// ==========================================================================
//  T7
// ==========================================================================

#[test]
fn refusals_are_named() {
    // A world needs a rank and a timeout.
    let m = config_message(ThreadComm::world(0));
    assert_names(&m, &["at least one rank"]);
    let m = config_message(ThreadComm::world_with_timeout(2, Duration::ZERO));
    assert_names(&m, &["timeout"]);

    // A tag above MAX_TAG.
    every_rank_refuses(2, &["32768", "32767"], |c| c.exchange(32768, &[], &mut []));

    // A peer out of range.
    every_rank_refuses(4, &["peer 4", "size 4"], |c| {
        c.exchange(1, &[(4, &[1u8][..])], &mut [])
    });

    // A peer twice in one list.
    every_rank_refuses(2, &["peer 1", "twice"], |c| {
        let b = [1u8, 2, 3];
        c.exchange(1, &[(1, &b[..]), (1, &b[..])], &mut [])
    });

    // A received length that differs: rank 0 sends 10 bytes, rank 1 expects 12.
    let res = ThreadComm::run(2, |c| {
        if c.rank() == 0 {
            c.exchange(2, &[(1, &[9u8; 10][..])], &mut [])
        } else {
            let mut buf = [0u8; 12];
            c.exchange(2, &[], &mut [(0, &mut buf[..])])
        }
    })
    .unwrap();
    let mut res = res.into_iter();
    res.next().unwrap().expect("rank 0 only sends");
    let m = config_message(res.next().unwrap());
    assert_names(&m, &["rank 0", "tag 2", "10", "12"]);
}

// ==========================================================================
//  T8
// ==========================================================================

/// One ring step under `t`: send 33 bytes to the next rank, receive 33 from
/// the previous one, and require them to be that rank's payload.
fn ring(c: &dyn Comm, t: u32) {
    let r = c.rank();
    let (next, prev) = ((r + 1) % 4, (r + 3) % 4);
    let out = payload(r, next, t, 33);
    let mut buf = vec![0u8; 33];
    c.exchange(
        t,
        &[(next, out.as_slice())],
        &mut [(prev, buf.as_mut_slice())],
    )
    .unwrap();
    assert!(
        buf == payload(prev, r, t, 33),
        "rank {r}: ring({t}) delivered the wrong bytes"
    );
}

#[test]
fn collectives_and_point_to_point_never_cross() {
    // The collectives use their own (dst, tag) constants in the payload rule,
    // none of which a ring payload can have (its dst is a rank, 0..4):
    //   bcast      root 2, payload(2, 2000, 11, 33)
    //   allgather  payload(q, 2001, 12, 33)
    //   allgatherv payload(q, 2002, 13, 30 + q)
    ThreadComm::run(4, |c| {
        let r = c.rank();
        ring(c, 0);

        let mut b = if r == 2 {
            payload(2, 2000, 11, 33)
        } else {
            vec![0u8; 33]
        };
        c.bcast(2, &mut b).unwrap();
        assert!(b == payload(2, 2000, 11, 33), "rank {r}: bcast bytes");

        ring(c, MAX_TAG);

        let mut all = vec![0u8; 4 * 33];
        c.allgather(&payload(r, 2001, 12, 33), &mut all).unwrap();
        let expect: Vec<u8> = (0..4).flat_map(|q| payload(q, 2001, 12, 33)).collect();
        assert!(all == expect, "rank {r}: allgather bytes");

        ring(c, 1);

        let got = c.allgatherv(&payload(r, 2002, 13, 30 + r)).unwrap();
        assert_eq!(got.len(), 4);
        for (q, block) in got.iter().enumerate() {
            assert!(
                *block == payload(q, 2002, 13, 30 + q),
                "rank {r}: allgatherv entry {q}"
            );
        }

        c.barrier().unwrap();
        ring(c, 0);
    })
    .unwrap();
}

// ==========================================================================
//  T9
// ==========================================================================

#[test]
fn repeated_exchanges_never_overtake() {
    ThreadComm::run(2, |c| {
        for k in 0..200u32 {
            if c.rank() == 0 {
                c.exchange(3, &[(1, &k.to_le_bytes()[..])], &mut [])
                    .unwrap();
            } else {
                // Rank 0 sends and moves on, so it runs ahead of this rank.
                for _ in 0..(k % 7) {
                    std::thread::yield_now();
                }
                let mut b = [0u8; 4];
                c.exchange(3, &[], &mut [(0, &mut b[..])]).unwrap();
                assert_eq!(
                    u32::from_le_bytes(b),
                    k,
                    "call {k} received another call's message"
                );
            }
        }
    })
    .unwrap();
}

// ==========================================================================
//  T10
// ==========================================================================

#[test]
fn mismatched_collectives_are_refused_by_name() {
    // 1. Different kinds: both ranks fail, naming both kinds and the call.
    every_rank_refuses(2, &["bcast", "allgather", "collective #0"], |c| {
        if c.rank() == 0 {
            c.bcast(0, &mut [0u8; 8])
        } else {
            c.allgather(&[0u8; 8], &mut [0u8; 16])
        }
    });

    // 2. Different roots: every rank fails.
    every_rank_refuses(3, &["root"], |c| {
        let root = if c.rank() == 0 { 0 } else { 1 };
        c.bcast(root, &mut [0u8; 8])
    });
}

// ==========================================================================
//  T11
// ==========================================================================

#[test]
fn a_panicking_rank_aborts_the_world_instead_of_hanging() {
    let t0 = Instant::now();
    // The default 120 s timeout: only the abort can end this quickly.
    let res = ThreadComm::run(4, |c| {
        if c.rank() == 2 {
            panic!("rank two stops here");
        }
        c.allgather(&[7u8; 8], &mut [0u8; 32])
    });
    let m = config_message(res);
    assert_names(&m, &["rank 2", "panicked", "rank two stops here"]);
    assert!(
        t0.elapsed() < Duration::from_secs(5),
        "took {:?}",
        t0.elapsed()
    );
}

// ==========================================================================
//  T12
// ==========================================================================

#[test]
fn a_missing_message_times_out_by_name() {
    let mut world = ThreadComm::world_with_timeout(2, Duration::from_millis(200)).unwrap();
    let c1 = world.pop().unwrap();
    let c0 = world.pop().unwrap();
    assert_eq!((c0.rank(), c1.rank()), (0, 1));
    let (r0, r1, waited) = std::thread::scope(|s| {
        let h0 = s.spawn(move || c0.exchange(4, &[], &mut []));
        let h1 = s.spawn(move || {
            let t0 = Instant::now();
            let mut buf8 = [0u8; 8];
            let r = c1.exchange(4, &[], &mut [(0, &mut buf8[..])]);
            (r, t0.elapsed())
        });
        let r0 = h0.join().unwrap();
        let (r1, waited) = h1.join().unwrap();
        (r0, r1, waited)
    });
    r0.expect("rank 0 has nothing to send or receive");
    let m = config_message(r1);
    assert_names(&m, &["rank 0", "tag 4", "200 ms"]);
    assert!(
        waited >= Duration::from_millis(200),
        "gave up after only {waited:?}"
    );
    assert!(waited < Duration::from_secs(5), "took {waited:?}");
}

// ==========================================================================
//  T13
// ==========================================================================

#[test]
fn barrier_holds_every_rank() {
    for n in 1..=8usize {
        let counter = AtomicUsize::new(0);
        ThreadComm::run(n, |c| {
            for k in 0..50usize {
                counter.fetch_add(1, Ordering::SeqCst);
                c.barrier().unwrap();
                assert_eq!(
                    counter.load(Ordering::SeqCst),
                    n * (k + 1),
                    "n {n}, round {k}: a rank passed the barrier before all had entered"
                );
                c.barrier().unwrap();
            }
        })
        .unwrap();
    }
}

// ==========================================================================
//  T14
// ==========================================================================

#[test]
fn a_world_knows_its_ranks() {
    fn assert_send<T: Send>() {}
    assert_send::<ThreadComm>();

    for n in 1..=8usize {
        let world = ThreadComm::world(n).unwrap();
        assert_eq!(world.len(), n);
        for (i, c) in world.iter().enumerate() {
            assert_eq!(c.rank(), i);
            assert_eq!(c.size(), n);
            assert_eq!(c.transport(), "threads");
        }
        // The trait is object-safe.
        let d: &dyn Comm = &world[n - 1];
        assert_eq!((d.rank(), d.size(), d.transport()), (n - 1, n, "threads"));

        let ranks = ThreadComm::run(n, |c| c.rank() * 10).unwrap();
        let expect: Vec<usize> = (0..n).map(|i| i * 10).collect();
        assert_eq!(ranks, expect);
    }
}
