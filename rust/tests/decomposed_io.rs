// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 139.7 (PAR-15)'s gate - decomposed I/O: per-rank writes under
//! `processorN/`, the reconstruct to the whole mesh, and the restart.
//!
//! Everything is compared `to_bits()` equality, zero differing values, in
//! f64 AND under `--features single`: a written set read back is the set
//! written; a restart fed a set and run one step lands where the
//! uninterrupted run landed; a reconstruct is the gathered fields; a
//! `processorN` set from a different P (or another cut) is refused by name.
//!
//! The restart leg is [`cavity_part_steps`]'s cavity at
//! [`RESTART_N`] cells per side, [`RESTART_ITERS`] steps before the write,
//! on `ThreadComm` ranks - P = 1 Linear, P = 2 and 4 Hilbert - one `Gpu`
//! per rank, the convention of the crate's other decomposed tests.
//!
//! Provenance: ORIGINAL - the gate of the I/O this module exercises
//! (SPEC-LIT 139.7, PAR-15); the awkward values are this test's own. No
//! external source. No GPL-licensed source was consulted.

use std::path::PathBuf;

use ofgpu::comm::ThreadComm;
use ofgpu::decompose::{Decomposition, PartitionMethod};
use ofgpu::distflow::{
    cavity_mesh, cavity_part_steps, CavitySteps, RESTART_ITERS, RESTART_N,
};
use ofgpu::io::decomposed::{
    check_processor_set, gather_written, read_rank, reconstruct, write_rank, RankFields,
};
use ofgpu::io::fields::{read_scalar_field, read_vector_field};
use ofgpu::io::output_types::OutputField;
use ofgpu::io::vtu::write_vtu;
use ofgpu::{Error, Gpu, Label, Result, Scalar, Vec3};

/// A machine without a card makes every device test pass vacuously, the
/// convention the rest of the crate's tests follow.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// Each test's own temp case directory, removed at the end.
struct CaseDir {
    path: PathBuf,
}

impl CaseDir {
    fn new(tag: &str) -> Self {
        let path = std::env::temp_dir()
            .join(format!("it_decomposed_io_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("temp case dir");
        CaseDir { path }
    }
}

impl Drop for CaseDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

// ==========================================================================
//  The comparisons - to_bits everywhere
// ==========================================================================

fn count_scalars(a: &[Scalar], b: &[Scalar]) -> usize {
    if a.len() != b.len() {
        return a.len().max(b.len());
    }
    a.iter()
        .zip(b)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count()
}

fn count_vectors(a: &[Vec3], b: &[Vec3]) -> usize {
    if a.len() != b.len() {
        return a.len().max(b.len());
    }
    a.iter()
        .zip(b)
        .filter(|(p, q)| {
            p.x.to_bits() != q.x.to_bits()
                || p.y.to_bits() != q.y.to_bits()
                || p.z.to_bits() != q.z.to_bits()
        })
        .count()
}

/// Every array of one rank's set, against the same rank's; 0 is bitwise.
fn diff(a: &RankFields, b: &RankFields) -> usize {
    count_vectors(&a.u, &b.u)
        + count_vectors(&a.u_bf, &b.u_bf)
        + count_scalars(&a.p, &b.p)
        + count_scalars(&a.p_bf, &b.p_bf)
        + count_scalars(&a.phi_f, &b.phi_f)
        + count_scalars(&a.phi_bf, &b.phi_bf)
}

/// The owned cells of `written`, gathered to whole-mesh order by
/// `Decomposition::gather_field` - PAR-01's gather, which does NOT run
/// through [`gather_written`], the gather `reconstruct` itself calls, so a
/// wrong cell gather rule cannot grade its own homework. U one component at
/// a time: x, then y, then z.
fn gather_cells(dec: &Decomposition, written: &[RankFields]) -> (Vec<Vec3>, Vec<Scalar>) {
    let axis = |pick: fn(&Vec3) -> Scalar| -> Vec<Scalar> {
        let cols: Vec<Vec<Scalar>> = written
            .iter()
            .map(|w| w.u.iter().map(pick).collect())
            .collect();
        dec.gather_field(&cols).expect("gather_field")
    };
    let (xs, ys, zs) = (axis(|v| v.x), axis(|v| v.y), axis(|v| v.z));
    let u: Vec<Vec3> = xs
        .iter()
        .zip(&ys)
        .zip(&zs)
        .map(|((x, y), z)| Vec3::new(*x, *y, *z))
        .collect();
    let cols: Vec<Vec<Scalar>> = written.iter().map(|w| w.p.clone()).collect();
    let p = dec.gather_field(&cols).expect("gather_field");
    (u, p)
}

// ==========================================================================
//  The synthetic sets
// ==========================================================================

/// Awkward values on purpose: a full mantissa, negative zero, the smallest
/// normal, the largest finite, and a repeating fraction - the values a
/// six-digit writer would spoil and this one must not.
fn awkward(i: usize) -> Scalar {
    match i % 5 {
        0 => ((i as Scalar) + 1.0).sqrt() / 7.0,
        1 => -0.0,
        2 => Scalar::MIN_POSITIVE,
        3 => Scalar::MAX,
        _ => -1.0 / 3.0,
    }
}

/// A full synthetic set for one part, every array filled from `awkward`.
fn synthetic(dec: &Decomposition, part: usize) -> RankFields {
    let m = &dec.parts[part].mesh;
    let (nc, nbf, nif) = (m.n_cells, m.n_boundary_faces, m.n_internal_faces);
    let mut f = RankFields {
        u: Vec::with_capacity(nc),
        u_bf: Vec::with_capacity(nbf),
        p: Vec::with_capacity(nc),
        p_bf: Vec::with_capacity(nbf),
        phi_f: Vec::with_capacity(nif),
        phi_bf: Vec::with_capacity(nbf),
    };
    for i in 0..nc {
        f.u.push(Vec3::new(awkward(3 * i), awkward(3 * i + 1), awkward(3 * i + 2)));
        f.p.push(awkward(i + 1));
    }
    for i in 0..nbf {
        f.u_bf.push(Vec3::new(awkward(nbf + i), -awkward(i), awkward(2 * i + 3)));
        f.p_bf.push(awkward(i + 2));
        f.phi_bf.push(-awkward(i + 4));
    }
    for i in 0..nif {
        f.phi_f.push(awkward(i + 3));
    }
    f
}

// ==========================================================================
//  Tests 1-4: the host-only legs
// ==========================================================================

/// **Test 1.** A synthetic set of every part of a P = 3 Hilbert cut of the
/// 8 x 8 cavity writes and reads back bit for bit - the awkward values
/// included, negative zero staying negative zero.
#[test]
fn the_rank_files_round_trip_bit_for_bit() {
    let hm = cavity_mesh(8).expect("cavity mesh");
    let dec = Decomposition::build(&hm, 3, &PartitionMethod::Hilbert).expect("cut");
    let dir = CaseDir::new("roundtrip");
    assert!(
        awkward(1).is_sign_negative() && awkward(1) == 0.0,
        "the synthetic values must include a negative zero"
    );
    for part in 0..dec.n_parts {
        write_rank(&dir.path, &dec, part, "0.5", &synthetic(&dec, part))
            .expect("write_rank");
    }
    // `read_rank` checks the WHOLE set first, so every part is on disk
    // before the first one is read back.
    for part in 0..dec.n_parts {
        let f = synthetic(&dec, part);
        let g = read_rank(&dir.path, &dec, part, "0.5").expect("read_rank");
        let bad = diff(&f, &g);
        assert_eq!(bad, 0, "part {part}: {bad} value(s) did not round trip");
    }
}

/// **Test 2.** A P = 4 set, checked, read and reconstructed with a P = 2
/// decomposition: refused, naming the set's 4 parts, this run's 2 ranks and
/// the choice.
#[test]
fn a_processor_set_from_a_different_p_is_refused() {
    let hm = cavity_mesh(8).expect("cavity mesh");
    let dec4 = Decomposition::build(&hm, 4, &PartitionMethod::Hilbert).expect("cut");
    let dir = CaseDir::new("wrongp");
    for part in 0..dec4.n_parts {
        write_rank(&dir.path, &dec4, part, "0.5", &synthetic(&dec4, part))
            .expect("write_rank");
    }
    let dec2 = Decomposition::build(&hm, 2, &PartitionMethod::Hilbert).expect("cut");

    let m = check_processor_set(&dir.path, &dec2, "0.5")
        .err()
        .expect("check_processor_set must refuse")
        .to_string();
    assert!(m.contains("processor set of 4 parts"), "{m}");
    assert!(m.contains("2 ranks"), "{m}");
    assert!(m.contains("from a different P"), "{m}");

    let m = read_rank(&dir.path, &dec2, 0, "0.5")
        .err()
        .expect("read_rank must refuse")
        .to_string();
    assert!(m.contains("from a different P"), "{m}");

    let m = reconstruct(&dir.path, &hm, &dec2, "0.5")
        .err()
        .expect("reconstruct must refuse")
        .to_string();
    assert!(m.contains("from a different P"), "{m}");
}

/// **Test 3.** A set cut by Hilbert, checked with a decomposition cut
/// another way at the SAME P: the cellProcAddressing is named, with the
/// first cell the two cuts disagree on. (Linear happens to be a different
/// cut of the 8 x 8 mesh; if the two ever coincided, round robin cannot.)
#[test]
fn a_processor_set_from_another_cut_is_refused() {
    let hm = cavity_mesh(8).expect("cavity mesh");
    let dec_h = Decomposition::build(&hm, 2, &PartitionMethod::Hilbert).expect("cut");
    let dir = CaseDir::new("othercut");
    for part in 0..dec_h.n_parts {
        write_rank(&dir.path, &dec_h, part, "0.5", &synthetic(&dec_h, part))
            .expect("write_rank");
    }
    let dec_l = Decomposition::build(&hm, 2, &PartitionMethod::Linear).expect("cut");
    let same_cut = (0..2).all(|p| {
        let n = dec_h.parts[p].mesh.n_cells;
        dec_h.parts[p].global_cell[..n] == dec_l.parts[p].global_cell[..n]
    });
    let other = if same_cut {
        PartitionMethod::Explicit((0..hm.n_cells).map(|c| (c % 2) as Label).collect())
    } else {
        PartitionMethod::Linear
    };
    let dec2 = Decomposition::build(&hm, 2, &other).expect("cut");
    let m = check_processor_set(&dir.path, &dec2, "0.5")
        .err()
        .expect("check_processor_set must refuse")
        .to_string();
    assert!(m.contains("cellProcAddressing"), "{m}");
}

/// **Test 4.** A time the set does not hold; a set with a rank's directory
/// removed; an empty directory. Each refused by name.
#[test]
fn a_missing_time_or_rank_is_refused() {
    let hm = cavity_mesh(8).expect("cavity mesh");
    let dec = Decomposition::build(&hm, 3, &PartitionMethod::Hilbert).expect("cut");
    let dir = CaseDir::new("missing");
    for part in 0..dec.n_parts {
        write_rank(&dir.path, &dec, part, "0.5", &synthetic(&dec, part))
            .expect("write_rank");
    }

    let m = check_processor_set(&dir.path, &dec, "9")
        .err()
        .expect("a missing time must refuse")
        .to_string();
    assert!(m.contains("does not exist"), "{m}");

    std::fs::remove_dir_all(dir.path.join("processor1")).expect("remove processor1");
    let m = check_processor_set(&dir.path, &dec, "0.5")
        .err()
        .expect("a gapped set must refuse")
        .to_string();
    assert!(m.contains("processor1 is missing"), "{m}");

    let empty = CaseDir::new("empty");
    let m = check_processor_set(&empty.path, &dec, "0.5")
        .err()
        .expect("an empty directory must refuse")
        .to_string();
    assert!(m.contains("no processor0"), "{m}");
}

// ==========================================================================
//  Tests 5-7: the restart gate, P = 1, 2, 4
// ==========================================================================

/// **The restart leg.** World A runs `RESTART_ITERS + 1` steps with the
/// fields after step `RESTART_ITERS` written to a `processorN` set; world B
/// restarts from that set and runs one step. B's end state is A's end
/// state, bit for bit, on every array of every rank; the reconstruct is
/// checked against a reference gathered INDEPENDENTLY of the code under
/// test - the cells by [`gather_cells`], the boundary faces by
/// [`gather_written`] - bit for bit; the reconstructed case fields read
/// back as the independently gathered internals; and the reconstructed VTU
/// is byte for byte `write_vtu` of those gathered fields.
fn restart_gate(np: usize) {
    if gpu().is_none() {
        println!("decomposed_io: no CUDA device - the gate passes vacuously");
        return;
    }
    let hm = cavity_mesh(RESTART_N).expect("cavity mesh");
    let method = if np == 1 {
        PartitionMethod::Linear
    } else {
        PartitionMethod::Hilbert
    };
    let name = if np == 1 { "linear" } else { "hilbert" };
    let dec = Decomposition::build(&hm, np, &method).expect("cut");
    let dir = CaseDir::new(&format!("gate{np}"));
    let dir_p = dir.path.join(format!("P{np}"));
    std::fs::create_dir_all(&dir_p).expect("case dir");
    let tname = RESTART_ITERS.to_string();

    let a = ThreadComm::run(np, |comm| -> Result<CavitySteps> {
        let gpu = Gpu::new(0)?;
        cavity_part_steps(
            &gpu,
            comm,
            &dec,
            RESTART_ITERS + 1,
            None,
            Some((&dir_p, RESTART_ITERS, tname.as_str())),
        )
    })
    .unwrap_or_else(|e| panic!("P = {np} {name}: {e}"));
    let a = a
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap_or_else(|e| panic!("P = {np} {name}: {e}"));

    let b = ThreadComm::run(np, |comm| -> Result<CavitySteps> {
        let gpu = Gpu::new(0)?;
        cavity_part_steps(
            &gpu,
            comm,
            &dec,
            1,
            Some((&dir_p, tname.as_str())),
            None,
        )
    })
    .unwrap_or_else(|e| panic!("P = {np} {name}: {e}"));
    let b = b
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap_or_else(|e| panic!("P = {np} {name}: {e}"));

    let mut restart_bad = 0usize;
    for (x, y) in b.iter().zip(a.iter()) {
        restart_bad += diff(&x.end, &y.end);
    }

    let written: Vec<RankFields> = a
        .iter()
        .map(|s| {
            s.written
                .clone()
                .ok_or_else(|| Error::Config("the write step wrote nothing".to_string()))
        })
        .collect::<Result<_>>()
        .expect("written fields");
    let rec = reconstruct(&dir_p, &hm, &dec, &tname).expect("reconstruct");
    // The cells against the INDEPENDENT gather, the boundary faces against
    // `gather_written`.
    let (want_u, want_p) = gather_cells(&dec, &written);
    let want = gather_written(&hm, &dec, &written).expect("gather");
    let rec_bad = count_vectors(&rec.u, &want_u)
        + count_scalars(&rec.p, &want_p)
        + count_vectors(&rec.u_bf, &want.u_bf)
        + count_scalars(&rec.p_bf, &want.p_bf);

    // The reconstructed case fields, read back at the whole cell count.
    let gu = read_vector_field(&dir_p.join(&tname).join("U"), hm.n_cells)
        .expect("read U")
        .internal;
    let gp = read_scalar_field(&dir_p.join(&tname).join("p"), hm.n_cells)
        .expect("read p")
        .internal;
    let case_bad = count_vectors(&gu, &want_u) + count_scalars(&gp, &want_p);

    // The VTU on disk is write_vtu of the independently gathered fields,
    // byte for byte.
    let t = tname.parse::<Scalar>().expect("numeric time");
    let twin = dir.path.join("vtu_twin.vtu");
    write_vtu(
        &twin,
        &hm,
        &[
            OutputField::vector("U", &want_u),
            OutputField::scalar("p", &want_p),
        ],
        Some(t),
    )
    .expect("write the twin vtu");
    let on_disk = std::fs::read(dir_p.join("VTK").join(format!("reconstructed_{tname}.vtu")))
        .expect("the reconstructed vtu");
    let twin_bytes = std::fs::read(&twin).expect("the twin vtu");
    let vtu_ok = on_disk == twin_bytes;

    println!(
        "[139.7] P = {np}: restart differing values {restart_bad}, \
         reconstruct differing values {rec_bad}"
    );
    assert_eq!(restart_bad, 0, "P = {np}: the restart is not bitwise");
    assert_eq!(rec_bad, 0, "P = {np}: the reconstruct is not bitwise");
    assert_eq!(case_bad, 0, "P = {np}: the case fields are not bitwise");
    assert!(vtu_ok, "P = {np}: the VTU is not write_vtu's bytes");
}

/// **Test 5.** The restart gate on one rank.
#[test]
fn restart_and_reconstruct_on_one_rank() {
    restart_gate(1);
}

/// **Test 6.** The restart gate on two ranks.
#[test]
fn restart_and_reconstruct_on_two_ranks() {
    restart_gate(2);
}

/// **Test 7.** The restart gate on four ranks.
#[test]
fn restart_and_reconstruct_on_four_ranks() {
    restart_gate(4);
}

/// **Test 8.** A P = 4 set on disk, a P = 2 world restarting from it: EVERY
/// rank refuses with "from a different P", before anything collective, and
/// the world returns - no rank is left waiting inside a collective.
#[test]
fn a_restart_from_a_different_p_refuses_on_every_rank() {
    if gpu().is_none() {
        println!("decomposed_io: no CUDA device - the gate passes vacuously");
        return;
    }
    let hm = cavity_mesh(RESTART_N).expect("cavity mesh");
    let dec4 = Decomposition::build(&hm, 4, &PartitionMethod::Hilbert).expect("cut");
    let dec2 = Decomposition::build(&hm, 2, &PartitionMethod::Hilbert).expect("cut");
    let dir = CaseDir::new("refuse");
    let dir_p = dir.path.join("P4");
    std::fs::create_dir_all(&dir_p).expect("case dir");
    let tname = "1";

    // World A: one step, written after it - the P = 4 set on disk.
    let a = ThreadComm::run(4, |comm| -> Result<CavitySteps> {
        let gpu = Gpu::new(0)?;
        cavity_part_steps(&gpu, comm, &dec4, 1, None, Some((&dir_p, 1, tname)))
    })
    .unwrap_or_else(|e| panic!("P = 4: {e}"));
    let a = a
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap_or_else(|e| panic!("P = 4: {e}"));
    assert!(
        a.iter().all(|s| s.written.is_some()),
        "P = 4: every rank must have written its set"
    );

    // World B: a P = 2 restart refuses on every rank, and returns.
    let b = ThreadComm::run(2, |comm| -> Result<CavitySteps> {
        let gpu = Gpu::new(0)?;
        cavity_part_steps(&gpu, comm, &dec2, 1, Some((&dir_p, tname)), None)
    })
    .unwrap_or_else(|e| panic!("the restart world did not return: {e}"));
    assert_eq!(b.len(), 2, "the restart world must return both ranks");
    for (r, x) in b.iter().enumerate() {
        let m = match x {
            Err(e) => e.to_string(),
            Ok(_) => "NO ERROR - the restart was accepted".to_string(),
        };
        assert!(
            m.contains("from a different P"),
            "rank {r}: the refusal is not by name: {m}"
        );
    }
}
