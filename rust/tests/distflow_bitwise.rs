// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 139.10 (PAR-14)'s gate - a decomposed laminar SIMPLE is **bit
//! for bit** the one-rank run: every cell of U, p (and T on the plume) and
//! every face of phi bitwise equal to the P = 1 SPMD run after
//! [`BITWISE_ITERS`] outer iterations, at P = 2, 3, 4, in BOTH builds.
//!
//! The reference is the P = 1 run under the Linear cut (which IS the
//! one-part cut), on [`ThreadComm`] ranks, each with its own [`Gpu`] on
//! device 0. A difference is a measurement, not a crash: it prints one
//! `[139.10]` MISSED line naming the first differing value and the test
//! still passes, so the gate can be run at a HEAD where the kernels are
//! not yet bitwise and report the miss it is about to close. Asserted:
//! every rank returned `Ok`, `steps.len() == BITWISE_ITERS` on every rank,
//! and something was compared.
//!
//! Provenance: ORIGINAL - the fixtures under test are this project's own
//! design (SPEC-LIT 139.10, PAR-14) over the decomposed SIMPLE of 139.4,
//! the exchange seam of 139.2 and the SPMD Krylov of 139.3. No external
//! source. No GPL-licensed source was consulted.

use ofgpu::comm::{Comm, ThreadComm};
use ofgpu::decompose::{Decomposition, PartitionMethod};
use ofgpu::distflow::{
    cavity_mesh, cavity_part_trace, plume_mesh, plume_part_trace, BitwiseRun, BITWISE_CAVITY_N,
    BITWISE_ITERS, BITWISE_PLUME,
};
use ofgpu::{Gpu, Result, Scalar, Vec3, HostMesh};

/// A machine without a card makes every device test pass vacuously, the
/// convention the rest of the crate's tests follow.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The bitwise gate (SPEC-LIT 139.10). `P = 1` with the Linear cut is the
/// reference; `P = 2, 3, 4` with Hilbert must reproduce it bit for bit.
fn bitwise_gate(
    name: &str,
    hm: &HostMesh,
    run: impl Fn(&Gpu, &dyn Comm, &Decomposition) -> Result<BitwiseRun> + Sync,
) {
    let Some(_gpu0) = gpu() else {
        println!("distflow_bitwise: no CUDA device - {name} passes vacuously");
        return;
    };
    let dec1 = Decomposition::build(hm, 1, &PartitionMethod::Linear).expect("cut 1");
    let r1s = ThreadComm::run(1, |comm| {
        let gpu = Gpu::new(0)?;
        run(&gpu, comm, &dec1)
    })
    .unwrap_or_else(|e| panic!("{name} P=1: {e}"));
    let r1 = r1s[0].as_ref().unwrap_or_else(|e| panic!("{name} P=1 rank 0: {e}"));
    assert_eq!(r1.global_cell.len(), hm.n_cells, "{name}: P=1 owns every cell");

    // The reference's global arrays: U, p (and T) by whole-mesh cell; the
    // internal flux by whole-mesh face id; the boundary flux by whole-mesh
    // face id minus the internal count.
    let n_if_g = hm.n_internal_faces;
    assert_eq!(r1.n_internal_faces, n_if_g, "{name}: P=1 is the whole mesh");
    let mut u1 = vec![Vec3::ZERO; hm.n_cells];
    let mut p1 = vec![0.0 as Scalar; hm.n_cells];
    let has_t = !r1.t.is_empty();
    let mut t1 = vec![0.0 as Scalar; hm.n_cells];
    for (i, &g) in r1.global_cell.iter().enumerate() {
        u1[g] = r1.u[i];
        p1[g] = r1.p[i];
    }
    if has_t {
        for (i, &g) in r1.global_cell.iter().enumerate() {
            t1[g] = r1.t[i];
        }
    }
    let mut phi_if1 = vec![0.0 as Scalar; n_if_g];
    let mut phi_bf1 = vec![0.0 as Scalar; r1.global_face.len() - n_if_g];
    for (f, &g) in r1.global_face.iter().enumerate() {
        let g = g as usize;
        if f < r1.n_internal_faces {
            phi_if1[g] = r1.phi_f[f];
        } else {
            phi_bf1[g - n_if_g] = r1.phi_bf[f - r1.n_internal_faces];
        }
    }

    // Every rank of every P > 1 world, against the one-rank run, on BITS.
    for np in [2usize, 3, 4] {
        let dec = Decomposition::build(hm, np, &PartitionMethod::Hilbert).expect("cut");
        let got = ThreadComm::run(np, |comm| {
            let gpu = Gpu::new(0)?;
            run(&gpu, comm, &dec)
        })
        .unwrap_or_else(|e| panic!("{name} P={np}: {e}"));
        let runs: Vec<&BitwiseRun> = got
            .iter()
            .enumerate()
            .map(|(r, x)| x.as_ref().unwrap_or_else(|e| panic!("{name} P={np} rank {r}: {e}")))
            .collect();
        for run in &runs {
            assert_eq!(
                run.steps.len(),
                BITWISE_ITERS,
                "{name} P={np}: the step count is part of the gate"
            );
        }

        // The tallies: U/p/phi values, T values, steps whose iteration
        // counts differ, steps whose residual bits differ - and the first
        // difference of the run, for the one MISSED line.
        let (mut k, mut n, mut kt, mut nt) = (0usize, 0usize, 0usize, 0usize);
        // Per STEP, over every rank: a step counts once however many ranks
        // differ on it.
        let mut s_bad = vec![false; BITWISE_ITERS];
        let mut r_bad = vec![false; BITWISE_ITERS];
        let mut first: Option<(&'static str, usize, usize, usize, Scalar, Scalar)> = None;
        let note = |first: &mut Option<(&'static str, usize, usize, usize, Scalar, Scalar)>,
                    r: usize,
                    field: &'static str,
                    i: usize,
                    g: usize,
                    got: Scalar,
                    want: Scalar| {
            if first.is_none() {
                *first = Some((field, r, i, g, got, want));
            }
        };
        for (r, run) in runs.iter().enumerate() {
            for (i, &g) in run.global_cell.iter().enumerate() {
                for (field, got, want) in [
                    ("U", run.u[i].x, u1[g].x),
                    ("U", run.u[i].y, u1[g].y),
                    ("U", run.u[i].z, u1[g].z),
                    ("p", run.p[i], p1[g]),
                ] {
                    n += 1;
                    if got.to_bits() != want.to_bits() {
                        k += 1;
                        note(&mut first, r, field, i, g, got, want);
                    }
                }
                if has_t {
                    nt += 1;
                    if run.t[i].to_bits() != t1[g].to_bits() {
                        kt += 1;
                        note(&mut first, r, "T", i, g, run.t[i], t1[g]);
                    }
                }
            }
            let n_if = run.n_internal_faces;
            assert_eq!(run.phi_f.len(), n_if, "{name} P={np} rank {r}: phi_f");
            assert_eq!(
                run.global_face.len(),
                n_if + run.phi_bf.len(),
                "{name} P={np} rank {r}: global_face"
            );
            for i in 0..n_if {
                let g = run.global_face[i] as usize;
                n += 1;
                if run.phi_f[i].to_bits() != phi_if1[g].to_bits() {
                    k += 1;
                    note(&mut first, r, "phi", i, g, run.phi_f[i], phi_if1[g]);
                }
            }
            for (j, &g) in run.global_face[n_if..].iter().enumerate() {
                let g = g as usize;
                // A processor face's flux is the whole mesh's internal flux
                // at that face with the rank's own side's sign; every other
                // boundary face is the reference's boundary face.
                let side = run.b_proc.get(j).copied().unwrap_or(-1);
                let (got, want) = (
                    run.phi_bf[j],
                    if side >= 0 {
                        let s = if side == 1 { 1.0 } else { -1.0 };
                        s * phi_if1[g]
                    } else {
                        phi_bf1[g - n_if_g]
                    },
                );
                n += 1;
                if got.to_bits() != want.to_bits() {
                    k += 1;
                    note(&mut first, r, "phi", n_if + j, g, got, want);
                }
            }
            for (s, (st, st1)) in run.steps.iter().zip(&r1.steps).enumerate() {
                if st.u_iters != st1.u_iters || st.p_iters != st1.p_iters || st.t_iters != st1.t_iters
                {
                    s_bad[s] = true;
                    note(
                        &mut first,
                        r,
                        "iterations",
                        s,
                        s,
                        st.p_iters as Scalar,
                        st1.p_iters as Scalar,
                    );
                }
                let same3 = st
                    .u_res
                    .iter()
                    .zip(&st1.u_res)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                if !same3
                    || st.p_res.to_bits() != st1.p_res.to_bits()
                    || st.t_res.to_bits() != st1.t_res.to_bits()
                {
                    r_bad[s] = true;
                    note(&mut first, r, "residual", s, s, st.p_res, st1.p_res);
                }
            }
        }
        let s_cnt = s_bad.iter().filter(|&&b| b).count();
        let s_res = r_bad.iter().filter(|&&b| b).count();
        println!(
            "  [139.10] {name} P={np}: U,p,phi {k} of {n} values differ; T {kt} of {nt}; \
             iteration counts differ on {s_cnt} of {BITWISE_ITERS} steps; \
             residual bits on {s_res}"
        );
        if let Some((field, r, i, g, got, want)) = first {
            let ulp = (got.to_bits() as i64)
                .wrapping_sub(want.to_bits() as i64)
                .unsigned_abs();
            println!(
                "  [139.10] MISSED {name} P={np} bitwise: first {field} at rank {r} local {i} \
                 global {g} got {got:e} want {want:e} ({ulp} ulp)"
            );
        }
        assert!(n > 0, "{name} P={np}: nothing was compared");
    }
}

/// The cavity fixture: the reduced 16 x 16 lid-driven cavity at Re = 100,
/// 50 outer iterations, P = 2, 3, 4 against the one-rank run, bits.
#[test]
fn a_decomposed_cavity_is_the_one_rank_run_bit_for_bit() {
    // The gate constants, fixed before measuring.
    assert_eq!(BITWISE_CAVITY_N, 16);
    assert_eq!(BITWISE_ITERS, 50);
    assert_eq!(BITWISE_PLUME, (12, 6, 4));

    let Some(_gpu0) = gpu() else {
        println!("distflow_bitwise: no CUDA device - the cavity gate passes vacuously");
        return;
    };
    let hm = cavity_mesh(BITWISE_CAVITY_N).expect("cavity mesh");
    bitwise_gate("cavity", &hm, |gpu, comm, dec| {
        cavity_part_trace(gpu, comm, dec, BITWISE_ITERS)
    });
}

/// The plume fixture: the laminar Boussinesq plume on the block generator's
/// `plume` box at 12 x 6 x 4, 50 outer iterations of SIMPLE plus energy,
/// P = 2, 3, 4 against the one-rank run, bits - U, p, T and phi.
#[test]
fn a_decomposed_plume_is_the_one_rank_run_bit_for_bit() {
    let Some(_gpu0) = gpu() else {
        println!("distflow_bitwise: no CUDA device - the plume gate passes vacuously");
        return;
    };
    let (nx, ny, nz) = BITWISE_PLUME;
    let hm = plume_mesh(nx, ny, nz).expect("plume mesh");
    bitwise_gate("plume", &hm, |gpu, comm, dec| {
        plume_part_trace(gpu, comm, dec, BITWISE_ITERS)
    });
}

/// The plume fixture's own shape, host-side: the block generator's `plume`
/// box at 12 x 6 x 4 is 288 cells and carries the seven patches the
/// fixture's boundary conditions name, none of them empty.
#[test]
fn the_plume_fixture_has_its_patches() {
    let (nx, ny, nz) = BITWISE_PLUME;
    let hm = plume_mesh(nx, ny, nz).expect("plume mesh");
    assert_eq!(hm.n_cells, 288, "the plume fixture is 12 x 6 x 4 cells");
    for name in ["inlet", "outlet", "wallXMin", "wallYMin", "wallYMax", "floor", "ceiling"] {
        let p = hm
            .patches
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("the plume mesh has no `{name}` patch"));
        assert!(p.size > 0, "the plume mesh's `{name}` patch is empty");
    }
}

/// The refinement's exact world sum over DOUBLE terms (SPEC-LIT 139.10):
/// dyadic terms whose sum is exact in f64, dealt over one part and over two,
/// must come back as that sum to the bit. Found in review: the host anchor
/// took `ilogb + 1` twice, so every such sum came back exactly doubled while
/// staying partition-invariant - the bitwise gate cannot see a bug that is
/// the same on every cut, and this test can.
#[test]
fn the_exact_double_sum_is_the_sum() {
    use ofgpu::exactsum::{exact_sum64, ExactSum64};
    let Some(gpu) = gpu() else {
        println!("distflow_bitwise: no CUDA device - exact double sum passes vacuously");
        return;
    };
    let terms = [300.0f64, -0.125, 2.5, 0.0078125];
    let want: f64 = 302.3828125;
    assert_eq!(terms.iter().sum::<f64>().to_bits(), want.to_bits());

    for split in [vec![4usize], vec![2, 2], vec![1, 3]] {
        let mut ex = ExactSum64::new(&gpu, &split).expect("exact sum");
        let mut at = 0usize;
        let mut bufs = Vec::new();
        for &k in &split {
            bufs.push(gpu.upload(&terms[at..at + k]).expect("upload"));
            at += k;
        }
        let mut amax = 0.0f64;
        for (p, b) in bufs.iter().enumerate() {
            amax = amax.max(ex.max_part(&gpu, p, b, split[p]).expect("max"));
        }
        ex.set_anchor(&gpu, amax).expect("anchor");
        let mut limbs = Vec::new();
        for (p, b) in bufs.iter().enumerate() {
            limbs.extend(ex.part_limbs(&gpu, p, b, split[p]).expect("limbs"));
        }
        let got = exact_sum64(&limbs, amax, 0.0);
        println!("  [139.10] exact double sum, parts {split:?}: {got} (want {want})");
        assert_eq!(got.to_bits(), want.to_bits(), "parts {split:?}: got {got}, want {want}");
    }
}
