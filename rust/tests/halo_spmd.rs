// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §138.3's gate — the SPMD halo exchange, in BOTH builds.
//!
//! An integration test because the library's own test target does not compile
//! under `--features single`; an integration test links the normal (non-test)
//! library, which does. Every rank is a [`ThreadComm`] thread with its own
//! [`Gpu`] on device 0, and the reference is the in-process [`HaloExchange`]
//! of §71.4 on the main thread's card: a program run over `P` ranks must
//! produce the bytes the one-process, `P`-part run produces.
//!
//! Class D of docs/17 §5.1: exact bytes and exact counts. Every comparison is
//! `to_bits()` per component, labels by `==`; the halo of every field is ZERO
//! before its exchange, and no value the owners hold is zero, so a forgotten
//! exchange cannot pass.
//!
//! Provenance: ORIGINAL - the SPMD form under test is this project's own
//! design (SPEC-LIT §138.3, PAR-02), on the exchange plan of §71.4. No
//! external source. No GPL-licensed source was consulted.

use ofgpu::blockgen::{build_case, CaseKind};
use ofgpu::comm::{Comm, ThreadComm};
use ofgpu::decompose::{Decomposition, PartitionMethod};
use ofgpu::device::{DevBuf, Gpu};
use ofgpu::halo::{HaloExchange, SpmdHalo, HALO_TAG};
use ofgpu::{Label, Result, Scalar, Tensor, Vec3};

/// A deterministic field: a function of the GLOBAL cell id, so splitting it
/// is a permutation and not a computation.
fn s_at(c: usize) -> Scalar {
    0.25 + 0.0625 * ((c * 37) % 23) as Scalar
}

/// `(s, -s, 2 s)` at cell `c`.
fn vec_at(c: usize) -> Vec3 {
    Vec3::new(s_at(c), -s_at(c), 2.0 * s_at(c))
}

/// The nine components, row-major (xx..zz), of the test tensor at cell `c`:
/// component `k` is `(k+1) * s_at(c)`, negated when `k` is odd.
fn tensor_at(c: usize) -> Tensor {
    let s = s_at(c);
    let v: [Scalar; 9] = std::array::from_fn(|k| {
        let x = (k + 1) as Scalar * s;
        if k % 2 == 1 {
            -x
        } else {
            x
        }
    });
    Tensor {
        xx: v[0], xy: v[1], xz: v[2],
        yx: v[3], yy: v[4], yz: v[5],
        zx: v[6], zy: v[7], zz: v[8],
    }
}

fn comps(t: Tensor) -> [Scalar; 9] {
    [
        t.xx, t.xy, t.xz, t.yx, t.yy, t.yz, t.zx, t.zy, t.zz,
    ]
}

/// `3 c - 1` at cell `c`.
fn label_at(c: usize) -> Label {
    3 * c as Label - 1
}

/// The local buffer of one part: its owned cells from the global field,
/// through `global_cell`; its halo ZERO, so a forgotten exchange cannot pass.
fn local_of<T: Copy>(
    global_cell: &[Label],
    n_cells: usize,
    n_local: usize,
    global: &[T],
    zero: T,
) -> Vec<T> {
    let mut h = vec![zero; n_local];
    for (i, hi) in h.iter_mut().enumerate().take(n_cells) {
        *hi = global[global_cell[i] as usize];
    }
    h
}

/// The round robin: cell `c` to part `c % P`, the worst partition possible,
/// which cuts nearly every face.
fn round_robin(n: usize, p: usize) -> PartitionMethod {
    PartitionMethod::Explicit((0..n).map(|c| (c % p) as Label).collect())
}

/// **The gate.** Two `blockgen` boxes, `P = 1..4`, three partitioners, four
/// payloads: every local cell of every rank, owned and ghost, equals bit for
/// bit the in-process `HaloExchange` result AND the owner's value. For
/// `P > 1` every part has a halo, and the number of ghost cells compared is
/// counted, positive, and equal to the sum of `n_halo` over every
/// decomposition visited times the four payloads.
#[test]
fn an_spmd_halo_is_bit_for_bit_the_in_process_halo() {
    let gpu0 = match Gpu::new(0) {
        Ok(g) => g,
        Err(_) => return,
    };
    assert_eq!(HALO_TAG, 71, "the tag is pinned by SPEC-LIT §138.3");

    let mut ghost_comparisons = 0usize;
    let mut expected_ghosts = 0usize;
    for (nx, ny, nz) in [(5usize, 4usize, 3usize), (12, 10, 8)] {
        let (m, _) = build_case(CaseKind::Big, nx, ny, nz).expect("mesh");
        let n = m.n_cells;
        let scalars: Vec<Scalar> = (0..n).map(s_at).collect();
        let vecs: Vec<Vec3> = (0..n).map(vec_at).collect();
        let tensors: Vec<Tensor> = (0..n).map(tensor_at).collect();
        let labels: Vec<Label> = (0..n).map(label_at).collect();
        for pcount in 1..=4usize {
            for method in [
                PartitionMethod::Hilbert,
                PartitionMethod::Linear,
                round_robin(n, pcount),
            ] {
                let d = Decomposition::build(&m, pcount, &method).expect("cut");
                // A gate that passed because nothing was actually cut would
                // be worse than no gate.
                if pcount > 1 {
                    assert!(
                        d.n_cut_faces > 0,
                        "P={pcount} {} cut no face",
                        method.as_str()
                    );
                    assert!(
                        d.parts.iter().all(|p| p.n_halo > 0),
                        "P={pcount} {} left a part with no halo",
                        method.as_str()
                    );
                }
                expected_ghosts += d.parts.iter().map(|p| p.n_halo).sum::<usize>() * 4;

                // ---- the reference: ONE in-process exchange, all four
                // payloads, on the main thread's card ----------------------
                let mut ex = HaloExchange::new(&gpu0, &d).expect("exchange");
                let mut sf: Vec<DevBuf<Scalar>> = d
                    .parts
                    .iter()
                    .map(|p| {
                        gpu0.upload(&local_of(
                            &p.global_cell,
                            p.mesh.n_cells,
                            p.n_local(),
                            &scalars,
                            0.0,
                        ))
                        .expect("upload")
                    })
                    .collect();
                let mut vf: Vec<DevBuf<Vec3>> = d
                    .parts
                    .iter()
                    .map(|p| {
                        gpu0.upload(&local_of(
                            &p.global_cell,
                            p.mesh.n_cells,
                            p.n_local(),
                            &vecs,
                            Vec3::ZERO,
                        ))
                        .expect("upload")
                    })
                    .collect();
                let mut tf: Vec<DevBuf<Tensor>> = d
                    .parts
                    .iter()
                    .map(|p| {
                        gpu0.upload(&local_of(
                            &p.global_cell,
                            p.mesh.n_cells,
                            p.n_local(),
                            &tensors,
                            Tensor::ZERO,
                        ))
                        .expect("upload")
                    })
                    .collect();
                let mut lf: Vec<DevBuf<Label>> = d
                    .parts
                    .iter()
                    .map(|p| {
                        gpu0.upload(&local_of(
                            &p.global_cell,
                            p.mesh.n_cells,
                            p.n_local(),
                            &labels,
                            0,
                        ))
                        .expect("upload")
                    })
                    .collect();
                ex.scalar(&gpu0, &mut sf).expect("exchange");
                ex.vector(&gpu0, &mut vf).expect("exchange");
                ex.tensor(&gpu0, &mut tf).expect("exchange");
                ex.label(&gpu0, &mut lf).expect("exchange");
                gpu0.sync().expect("sync");
                let ref_s: Vec<Vec<Scalar>> = sf
                    .iter()
                    .map(|b| gpu0.download(b).expect("download"))
                    .collect();
                let ref_v: Vec<Vec<Vec3>> = vf
                    .iter()
                    .map(|b| gpu0.download(b).expect("download"))
                    .collect();
                let ref_t: Vec<Vec<Tensor>> = tf
                    .iter()
                    .map(|b| gpu0.download(b).expect("download"))
                    .collect();
                let ref_l: Vec<Vec<Label>> = lf
                    .iter()
                    .map(|b| gpu0.download(b).expect("download"))
                    .collect();

                // ---- the SPMD run: one thread per rank, each with its own
                // `Gpu` on device 0. A rank's error must fail the test with
                // its text, so the closure returns `Result`. ----------------
                let got = ThreadComm::run(
                    pcount,
                    |comm| -> Result<(Vec<Scalar>, Vec<Vec3>, Vec<Tensor>, Vec<Label>)> {
                        let gpu = Gpu::new(0)?;
                        let part = &d.parts[comm.rank()];
                        let mut halo = SpmdHalo::new(&gpu, part, pcount)?;
                        let mut s = gpu.upload(&local_of(
                            &part.global_cell,
                            part.mesh.n_cells,
                            part.n_local(),
                            &scalars,
                            0.0,
                        ))?;
                        let mut v = gpu.upload(&local_of(
                            &part.global_cell,
                            part.mesh.n_cells,
                            part.n_local(),
                            &vecs,
                            Vec3::ZERO,
                        ))?;
                        let mut t = gpu.upload(&local_of(
                            &part.global_cell,
                            part.mesh.n_cells,
                            part.n_local(),
                            &tensors,
                            Tensor::ZERO,
                        ))?;
                        let mut l = gpu.upload(&local_of(
                            &part.global_cell,
                            part.mesh.n_cells,
                            part.n_local(),
                            &labels,
                            0,
                        ))?;
                        halo.scalar(&gpu, comm, &mut s)?;
                        halo.vector(&gpu, comm, &mut v)?;
                        halo.tensor(&gpu, comm, &mut t)?;
                        halo.label(&gpu, comm, &mut l)?;
                        gpu.sync()?;
                        Ok((
                            gpu.download(&s)?,
                            gpu.download(&v)?,
                            gpu.download(&t)?,
                            gpu.download(&l)?,
                        ))
                    },
                )
                .expect("spmd run");

                // ---- every rank's every local cell: SPMD == in-process ==
                // == the owner's value, four payloads ------------------------
                for (r, part) in d.parts.iter().enumerate() {
                    let (gs, gv, gt, gl) = got[r]
                        .as_ref()
                        .unwrap_or_else(|e| panic!("rank {r} failed: {e}"));
                    for i in 0..part.n_local() {
                        let g = part.global_cell[i] as usize;
                        if i >= part.mesh.n_cells {
                            ghost_comparisons += 4;
                        }
                        let at = format!(
                            "mesh {nx}x{ny}x{nz} P={pcount} {} rank {r} \
                             local {i} (global {g})",
                            method.as_str()
                        );
                        assert_eq!(
                            gs[i].to_bits(),
                            scalars[g].to_bits(),
                            "scalar vs owner, {at}"
                        );
                        assert_eq!(
                            gs[i].to_bits(),
                            ref_s[r][i].to_bits(),
                            "scalar vs in-process, {at}"
                        );
                        let got_v = [gv[i].x, gv[i].y, gv[i].z];
                        let want_v = [vecs[g].x, vecs[g].y, vecs[g].z];
                        let ref_v3 = [ref_v[r][i].x, ref_v[r][i].y, ref_v[r][i].z];
                        for k in 0..3 {
                            assert_eq!(
                                got_v[k].to_bits(),
                                want_v[k].to_bits(),
                                "vector component {k} vs owner, {at}"
                            );
                            assert_eq!(
                                got_v[k].to_bits(),
                                ref_v3[k].to_bits(),
                                "vector component {k} vs in-process, {at}"
                            );
                        }
                        let got_t = comps(gt[i]);
                        let want_t = comps(tensors[g]);
                        let ref_t9 = comps(ref_t[r][i]);
                        for k in 0..9 {
                            assert_eq!(
                                got_t[k].to_bits(),
                                want_t[k].to_bits(),
                                "tensor component {k} vs owner, {at}"
                            );
                            assert_eq!(
                                got_t[k].to_bits(),
                                ref_t9[k].to_bits(),
                                "tensor component {k} vs in-process, {at}"
                            );
                        }
                        assert_eq!(gl[i], labels[g], "label vs owner, {at}");
                        assert_eq!(gl[i], ref_l[r][i], "label vs in-process, {at}");
                    }
                }
            }
        }
    }
    assert!(
        ghost_comparisons > 0,
        "no ghost cell was compared - the gate passed on nothing"
    );
    assert_eq!(
        ghost_comparisons, expected_ghosts,
        "ghost comparisons must be the sum of n_halo over every \
         decomposition visited times the four payloads"
    );
    eprintln!(
        "an_spmd_halo_is_bit_for_bit_the_in_process_halo: {ghost_comparisons} \
         ghost cells compared across the four payloads"
    );
}

/// Two different scalar fields exchanged one after the other with NOTHING
/// between them: each field's ghosts hold THAT field's owner values, so the
/// second exchange neither read the first's bytes nor left them behind. This
/// is the hazard §138.3's second synchronisation closes: the next exchange
/// must not overwrite a host buffer the stream is still reading.
#[test]
fn two_spmd_exchanges_back_to_back_do_not_share_bytes() {
    match Gpu::new(0) {
        Ok(_) => {}
        Err(_) => return,
    }
    let (m, _) = build_case(CaseKind::Big, 12, 10, 8).expect("mesh");
    let n = m.n_cells;
    let d = Decomposition::build(&m, 4, &round_robin(n, 4)).expect("cut");

    let got = ThreadComm::run(4, |comm| -> Result<(Vec<Scalar>, Vec<Scalar>)> {
        let gpu = Gpu::new(0)?;
        let part = &d.parts[comm.rank()];
        let mut halo = SpmdHalo::new(&gpu, part, 4)?;
        let mut a = vec![0.0 as Scalar; part.n_local()];
        let mut b = vec![0.0 as Scalar; part.n_local()];
        for i in 0..part.mesh.n_cells {
            let c = part.global_cell[i] as usize;
            a[i] = s_at(c);
            b[i] = -s_at(c) - 1.0;
        }
        let mut da = gpu.upload(&a)?;
        let mut db = gpu.upload(&b)?;
        halo.scalar(&gpu, comm, &mut da)?;
        halo.scalar(&gpu, comm, &mut db)?;
        gpu.sync()?;
        Ok((gpu.download(&da)?, gpu.download(&db)?))
    })
    .expect("spmd run");

    for (r, part) in d.parts.iter().enumerate() {
        let (a, b) = got[r]
            .as_ref()
            .unwrap_or_else(|e| panic!("rank {r} failed: {e}"));
        for i in 0..part.n_local() {
            let g = part.global_cell[i] as usize;
            assert_eq!(
                a[i].to_bits(),
                s_at(g).to_bits(),
                "A, rank {r} local {i} (global {g})"
            );
            assert_eq!(
                b[i].to_bits(),
                (-s_at(g) - 1.0).to_bits(),
                "B, rank {r} local {i} (global {g})"
            );
        }
    }
}

/// The eight refusals of §138.3's table, in order, each `Error::Config`
/// starting `halo: `. No threads: every refusal happens BEFORE any message is
/// sent, so handles from `ThreadComm::world` on this thread are enough.
#[test]
fn an_spmd_halo_refuses_by_name() {
    let gpu = match Gpu::new(0) {
        Ok(g) => g,
        Err(_) => return,
    };
    let (m, _) = build_case(CaseKind::Big, 5, 4, 3).expect("mesh");
    let n = m.n_cells;
    let d3 = Decomposition::build(&m, 3, &round_robin(n, 3)).expect("cut");

    // Row 1: the part's number is not below the rank count.
    let msg = SpmdHalo::new(&gpu, &d3.parts[2], 2)
        .err().expect("row 1 must refuse")
        .to_string();
    assert!(msg.starts_with("halo: "), "{msg}");
    assert!(msg.contains("part 2"), "{msg}");
    assert!(msg.contains("2 ranks"), "{msg}");

    // Row 2: a neighbour is not below the rank count.
    let msg = SpmdHalo::new(&gpu, &d3.parts[0], 2)
        .err().expect("row 2 must refuse")
        .to_string();
    assert!(msg.starts_with("halo: "), "{msg}");
    assert!(msg.contains("neighbour 2"), "{msg}");
    assert!(msg.contains("2 ranks"), "{msg}");

    // Row 3, left: the offsets do not fit the neighbour list.
    let mut bad = d3.parts[0].clone();
    bad.send_offset.pop();
    let want = bad.send_offset.len();
    let msg = SpmdHalo::new(&gpu, &bad, 3)
        .err().expect("row 3 (offsets) must refuse")
        .to_string();
    assert!(msg.starts_with("halo: "), "{msg}");
    assert!(msg.contains(&format!("{want}")), "{msg}");
    assert!(msg.contains(&format!("not {}", d3.parts[0].nbr_parts.len() + 1)), "{msg}");

    // Row 3, right: the send list's length is not where its offsets end.
    let mut bad = d3.parts[0].clone();
    let end = *bad.send_offset.last().expect("offsets");
    bad.send_index.push(0);
    let msg = SpmdHalo::new(&gpu, &bad, 3)
        .err().expect("row 3 (send list) must refuse")
        .to_string();
    assert!(msg.starts_with("halo: "), "{msg}");
    assert!(msg.contains(&format!("holds {} entries", end + 1)), "{msg}");
    assert!(msg.contains(&format!("end at {end}")), "{msg}");

    // Row 4: the offsets' VALUES, each a clone of `d3.parts[0]`, which has
    // two neighbours and so three offsets: they start at 0, they never
    // decrease, and the receive offsets end at the part's `n_halo`.
    let nh = d3.parts[0].n_halo;

    // Row 4, receive end: the receive offsets end past `n_halo`.
    let mut bad = d3.parts[0].clone();
    let end = bad.recv_offset.last().copied().expect("offsets");
    *bad.recv_offset.last_mut().expect("offsets") = end + 1;
    let msg = SpmdHalo::new(&gpu, &bad, 3)
        .err().expect("row 4 receive end must refuse")
        .to_string();
    assert!(msg.starts_with("halo: "), "{msg}");
    assert!(msg.contains("receive offsets"), "{msg}");
    assert!(msg.contains(&(end + 1).to_string()), "{msg}");
    assert!(msg.contains("halo cells"), "{msg}");
    assert!(msg.contains(&nh.to_string()), "{msg}");

    // Row 4, receive start: the receive offsets' first entry is not 0.
    let mut bad = d3.parts[0].clone();
    bad.recv_offset[0] = 1;
    let msg = SpmdHalo::new(&gpu, &bad, 3)
        .err().expect("row 4 receive start must refuse")
        .to_string();
    assert!(msg.starts_with("halo: "), "{msg}");
    assert!(msg.contains("receive offsets"), "{msg}");
    assert!(msg.contains("start at 0"), "{msg}");

    // Row 4, send start: the send offsets' first entry is not 0.
    let mut bad = d3.parts[0].clone();
    bad.send_offset[0] = 1;
    let msg = SpmdHalo::new(&gpu, &bad, 3)
        .err().expect("row 4 send start must refuse")
        .to_string();
    assert!(msg.starts_with("halo: "), "{msg}");
    assert!(msg.contains("send offsets"), "{msg}");
    assert!(msg.contains("start at 0"), "{msg}");

    // Row 4, send decrease: `send_offset[1]` is past `send_offset[2]`.
    let mut bad = d3.parts[0].clone();
    let last = *bad.send_offset.last().expect("offsets");
    bad.send_offset[1] = last + 1;
    let msg = SpmdHalo::new(&gpu, &bad, 3)
        .err().expect("row 4 send decrease must refuse")
        .to_string();
    assert!(msg.starts_with("halo: "), "{msg}");
    assert!(msg.contains("send offsets"), "{msg}");
    assert!(msg.contains("decrease"), "{msg}");
    assert!(msg.contains(&format!("entry 1 is {}", last + 1)), "{msg}");
    assert!(msg.contains(&format!("entry 2 is {last}")), "{msg}");

    // Row 4, receive decrease: `recv_offset[1]` is past `recv_offset[2]`.
    let mut bad = d3.parts[0].clone();
    let last = *bad.recv_offset.last().expect("offsets");
    bad.recv_offset[1] = last + 1;
    let msg = SpmdHalo::new(&gpu, &bad, 3)
        .err().expect("row 4 receive decrease must refuse")
        .to_string();
    assert!(msg.starts_with("halo: "), "{msg}");
    assert!(msg.contains("receive offsets"), "{msg}");
    assert!(msg.contains("decrease"), "{msg}");
    assert!(msg.contains(&format!("entry 1 is {}", last + 1)), "{msg}");
    assert!(msg.contains(&format!("entry 2 is {last}")), "{msg}");

    // Row 5: an exchange inside a capture. Afterwards, the same call outside
    // the capture is `Ok` - the refusal left nothing behind.
    let d1 = Decomposition::build(&m, 1, &round_robin(n, 1)).expect("cut");
    let mut halo = SpmdHalo::new(&gpu, &d1.parts[0], 1).expect("plan");
    let comm1 = &ThreadComm::world(1).expect("world")[0];
    let mut f = gpu.zeros::<Scalar>(halo.n_local()).expect("alloc");
    let msg = gpu
        .capture(|_| halo.scalar(&gpu, comm1, &mut f))
        .err().expect("row 5 must refuse")
        .to_string();
    assert!(msg.starts_with("halo: "), "{msg}");
    assert!(msg.contains("capture"), "{msg}");
    halo.scalar(&gpu, comm1, &mut f).expect("after the refusal");

    // Row 6: `comm.size()` differs from the rank count given to `new`.
    let w2 = ThreadComm::world(2).expect("world");
    let mut halo = SpmdHalo::new(&gpu, &d3.parts[0], 3).expect("plan");
    let mut f = gpu.zeros::<Scalar>(halo.n_local()).expect("alloc");
    let msg = halo
        .scalar(&gpu, &w2[0], &mut f)
        .err().expect("row 6 must refuse")
        .to_string();
    assert!(msg.starts_with("halo: "), "{msg}");
    assert!(msg.contains("3"), "{msg}");
    assert!(msg.contains("2"), "{msg}");
    assert!(msg.contains("ranks"), "{msg}");

    // Row 7: `comm.rank()` is not the part.
    let w3 = ThreadComm::world(3).expect("world");
    let mut halo = SpmdHalo::new(&gpu, &d3.parts[1], 3).expect("plan");
    let mut f = gpu.zeros::<Scalar>(halo.n_local()).expect("alloc");
    let msg = halo
        .scalar(&gpu, &w3[0], &mut f)
        .err().expect("row 7 must refuse")
        .to_string();
    assert!(msg.starts_with("halo: "), "{msg}");
    assert!(msg.contains("rank 0"), "{msg}");
    assert!(msg.contains("part 1"), "{msg}");

    // Row 8: a field buffer shorter than `n_cells + n_halo`, all four
    // payloads. Rank 1 IS part 1 here, so only the buffer is wrong.
    let mut halo = SpmdHalo::new(&gpu, &d3.parts[1], 3).expect("plan");
    let nc = halo.n_cells();
    let nl = halo.n_local();
    let mut fs = gpu.zeros::<Scalar>(nc).expect("alloc");
    let msg = halo
        .scalar(&gpu, &w3[1], &mut fs)
        .err().expect("row 8 scalar must refuse")
        .to_string();
    assert!(msg.starts_with("halo: "), "{msg}");
    assert!(msg.contains("halo cells"), "{msg}");
    assert!(msg.contains("must be"), "{msg}");
    assert!(msg.contains(&nc.to_string()), "{msg}");
    assert!(msg.contains(&nl.to_string()), "{msg}");
    for name in ["vector", "tensor", "label"] {
        let msg = match name {
            "vector" => {
                let mut f = gpu.zeros::<Vec3>(nc).expect("alloc");
                halo.vector(&gpu, &w3[1], &mut f)
                    .err().expect("row 8 vector must refuse")
                    .to_string()
            }
            "tensor" => {
                let mut f = gpu.zeros::<Tensor>(nc).expect("alloc");
                halo.tensor(&gpu, &w3[1], &mut f)
                    .err().expect("row 8 tensor must refuse")
                    .to_string()
            }
            _ => {
                let mut f = gpu.zeros::<Label>(nc).expect("alloc");
                halo.label(&gpu, &w3[1], &mut f)
                    .err().expect("row 8 label must refuse")
                    .to_string()
            }
        };
        assert!(msg.starts_with("halo: "), "{name}: {msg}");
        assert!(msg.contains("halo cells"), "{name}: {msg}");
        assert!(msg.contains("must be"), "{name}: {msg}");
        assert!(msg.contains(&nc.to_string()), "{name}: {msg}");
        assert!(msg.contains(&nl.to_string()), "{name}: {msg}");
    }
}
