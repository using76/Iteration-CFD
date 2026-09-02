// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
//! SPEC-LIT section 84.9, rows 5 to 7. No GPL-licensed source was consulted.
//!
//! Rows 1 and 2 - the bitwise gate against the host emitter, and the whole
//! resident route end to end - live in `adapt`'s own test module and not
//! here, because they need `Forest` and section 75.9's
//! `no_time_loop_reaches_the_adapt` requires that nothing outside `adapt`
//! name it at all. Section 84.7 records the same rule as the reason
//! `LeafGrid` carries its own `voxel_limit` rather than reaching for
//! `VOXEL_LIMIT`.

use super::*;

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// Every kernel is a gather, and there is no atomic of any width.
///
/// Section 84.7 rests on this: an `atomicMin` over touch ranks would give the
/// same answer - min is order-independent on integers - but the file's claim
/// is the stronger one, and a claim in a header comment that no test reads is
/// a claim that decays. `parcelsort.cu`'s scan is reused rather than copied
/// and carries the same property (section 67.2).
#[test]
fn the_emitter_uses_no_atomic_and_no_shared_scatter() {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("cuda/meshemit.cu");
    let text = std::fs::read_to_string(&p).expect("cuda/meshemit.cu must be readable");

    // The header says the words; the code must not - so both comment forms
    // come out first, block comments included, because the file's own header
    // is where "there is no atomic of any width in this file" is written.
    let mut code = String::with_capacity(text.len());
    let b = text.as_bytes();
    let (mut i, mut block, mut line) = (0usize, false, false);
    while i < b.len() {
        if block {
            if b[i] == b'*' && i + 1 < b.len() && b[i + 1] == b'/' {
                block = false;
                i += 2;
                continue;
            }
        } else if line {
            if b[i] == b'\n' {
                line = false;
                code.push('\n');
            }
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            block = true;
            i += 2;
            continue;
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            line = true;
        } else {
            code.push(b[i] as char);
        }
        i += 1;
    }
    assert!(
        !code.contains("atomic"),
        "cuda/meshemit.cu has grown an atomic. SPEC-LIT section 84.7: the point \
         numbering is a min over touch ranks and a min is order-independent, so \
         an atomic would still be deterministic - but the file's header claims \
         there is none, and a claim nothing checks is a claim that decays"
    );
    assert!(
        !code.contains("__shfl") && !code.contains("cub::"),
        "cuda/meshemit.cu reaches for a warp primitive or CUB; the scan it needs \
         is parcelsort.cu's, reused rather than copied (SPEC-LIT section 84.2)"
    );
}

/// The emitter refuses a leaf set with a gap in the SAME WORDS the host
/// emitter refuses it in.
///
/// Section 13.4's rule applied to a diagnosis: a caller that switches emitters
/// must not get a different account of the same broken mesh. The gap is
/// reached by handing the emitter a leaf list with one leaf removed, which
/// `Forest::from_leaves` would refuse - so the packed array is built directly.
#[test]
fn a_leaf_set_with_a_gap_is_refused_in_the_host_emitters_words() {
    let Some(g) = gpu() else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let k = MeshEmitKernels::new(&g).expect("meshemit kernels");

    // A 2x1x1 base grid with base cell 1 refined once, minus one of its eight
    // children: voxels of the finest grid in that octant belong to no leaf.
    let mut leaf: Vec<i32> = vec![0, 0, 0, 0, 0];
    for k2 in 0..2 {
        for j in 0..2 {
            for i in 0..2 {
                if (i, j, k2) == (1, 1, 1) {
                    continue;
                }
                leaf.extend_from_slice(&[1, 1, i, j, k2]);
            }
        }
    }
    let e = emit_device(
        &g,
        &k,
        &LeafGrid {
            n: [2, 1, 1],
            d: Vec3::new(0.1, 0.2, 0.3),
            lmax: 1,
            voxel_limit: 64 << 20,
            leaf: &leaf,
        },
    );
    let e = match e {
        Err(e) => e,
        Ok(_) => panic!("a leaf set with a gap must be refused"),
    };
    let s = e.to_string();
    assert!(
        s.contains("belongs to no leaf") && s.contains("leaves a gap here"),
        "the gap refusal is not the host emitter's: {s}"
    );
}

/// The scan this module borrows is the one it thinks it is borrowing.
///
/// `Scan` runs `parcelsort.cu`'s three kernels at a length chosen per call,
/// which [`crate::parcels::DeviceScan`] cannot do. If the two ever disagreed,
/// every count in the emitter would be wrong in a way the bitwise gate would
/// report as a mesh difference and not as a scan difference - so it is
/// cheaper to say here which one it is.
#[test]
fn the_borrowed_scan_is_an_exclusive_scan() {
    let Some(g) = gpu() else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let k = MeshEmitKernels::new(&g).expect("meshemit kernels");

    for n in [1usize, 7, 256, 1024, 1025, 4096, 100_003] {
        let inp: Vec<i32> = (0..n).map(|i| (i % 3) as i32).collect();
        let d_in = g.upload(&inp).expect("upload");
        let mut d_out: DevBuf<i32> = g.zeros(n).expect("alloc");
        k.scan.run(&g, &d_in, &mut d_out, n).expect("scan");
        let got = g.download(&d_out).expect("download");

        let mut acc = 0i32;
        for i in 0..n {
            assert_eq!(got[i], acc, "exclusive scan at {i} of {n}");
            acc += inp[i];
        }
    }
}
