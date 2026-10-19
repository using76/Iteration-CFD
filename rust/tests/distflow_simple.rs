// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 139.4 (PAR-10c)'s gate - `Simple`'s exchange seam: a laminar
//! lid-driven cavity (Ghia, Ghia & Shin 1982, DOI 10.1016/0021-9991(82)90058-4)
//! at Re = 100, 64 x 64 (and the reduced 32 x 32 verify run), on P = 1, 2, 4 [`ThreadComm`] ranks, converging
//! to the serial answer.
//!
//! An integration test for the same reason `tests/distflow_krylov.rs` is:
//! it links the normal (non-test) library, so it compiles under
//! `--features single` too. Every rank is a thread with its own [`Gpu`] on
//! device 0; the reference is `cavity_serial` on the main thread's card.
//!
//! The gate constants are the ones `distflow` fixed BEFORE measuring -
//! `assert_eq!`ed against their literals at the top of the first test, so
//! nobody loosens them silently.
//!
//! Provenance: ORIGINAL - the exchange seam under test is this project's
//! own design (SPEC-LIT 139.4, PAR-10c) over the exchanges of 139.2 and the
//! SPMD Krylov routing of 139.3; the reference profile is the published
//! table of Ghia, Ghia & Shin (1982). No external source. No GPL-licensed
//! source was consulted.

use ofgpu::comm::{Comm, ThreadComm};
use ofgpu::decompose::{Decomposition, PartitionMethod};
use ofgpu::distflow::{
    cavity_ghia_errors, cavity_mesh, cavity_part, cavity_serial, CAVITY_ITER_BAND, CAVITY_N,
    CAVITY_N_REDUCED, CAVITY_TOL_CENTRE, CAVITY_TOL_FIELD,
};
use ofgpu::{Gpu, Result, Scalar, Vec3};

/// A machine without a card makes every device test pass vacuously, the
/// convention the rest of the crate's tests follow.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The rank counts and their cutters: P = 1 is the Linear cut (which IS the
/// one-part cut), P = 2 and 4 are Hilbert's.
fn cuts(hm: &ofgpu::HostMesh, np: usize) -> (String, Decomposition) {
    let method = if np == 1 {
        PartitionMethod::Linear
    } else {
        PartitionMethod::Hilbert
    };
    let name = if np == 1 { "linear" } else { "hilbert" };
    (
        name.to_string(),
        Decomposition::build(hm, np, &method).expect("cut"),
    )
}

/// **The gate.** The serial cavity converges; the decomposed one, on
/// P = 1, 2, 4 ranks with `Simple`'s exchange set, reaches the serial
/// answer: every rank `Ok` and `converged`, every rank the same
/// `iterations` and the same `u_residual` bits (the world's max is what
/// makes those bits equal), every owned cell within
/// [`CAVITY_TOL_FIELD`] of the serial U and p, the SIMPLE iteration count
/// within [`CAVITY_ITER_BAND`] of the serial count, and the centreline
/// within the serial run's own error plus [`CAVITY_TOL_CENTRE`] at every
/// Ghia station.
///
/// The verify run is the REDUCED cavity, [`CAVITY_N_REDUCED`] cells per side
/// with the gate's constants unchanged (the size is reduced, never a
/// tolerance): the full 64 x 64 gate takes 20-46 min per rank count on the
/// shared card (SPEC-LIT 139.4's measured table), so it is
/// [`the_full_size_cavity_gate`], ignored and run as a deferred item.
#[test]
fn a_decomposed_cavity_converges_to_the_serial_answer() {
    assert_eq!(CAVITY_N_REDUCED, 32);
    gate(CAVITY_N_REDUCED);
}

/// The plan's gate at its own size, 64 x 64 (docs/17 §12, PAR-10c).
/// Deferred: over the 10-minute verify budget on one shared card.
#[test]
#[ignore = "deferred: the 64 x 64 gate runs 20-46 min per rank count (SPEC-LIT 139.4)"]
fn the_full_size_cavity_gate() {
    gate(CAVITY_N);
}

fn gate(n: usize) {
    // The gate constants, fixed before measuring - literal values for THIS
    // build, so a silent loosening fails here and not in a paper.
    #[cfg(feature = "single")]
    {
        assert_eq!(CAVITY_TOL_FIELD, 1e-3);
        assert_eq!(CAVITY_TOL_CENTRE, 1e-3);
    }
    #[cfg(not(feature = "single"))]
    {
        assert_eq!(CAVITY_TOL_FIELD, 1e-6);
        assert_eq!(CAVITY_TOL_CENTRE, 1e-6);
    }
    assert_eq!(CAVITY_ITER_BAND, 0.02);
    assert_eq!(CAVITY_N, 64);

    let Some(gpu0) = gpu() else {
        println!("distflow_simple: no CUDA device - the gate passes vacuously");
        return;
    };

    let hm = cavity_mesh(n).expect("cavity mesh");
    let t0 = std::time::Instant::now();
    let serial = cavity_serial(&gpu0, &hm).expect("serial cavity");
    println!(
        "n={n} serial : converged {} in {} iterations, worst U residual {:.3e}, \
         continuity {:.3e}, {:.1} s",
        serial.converged,
        serial.iterations,
        serial.u_residual,
        serial.continuity,
        t0.elapsed().as_secs_f64()
    );
    assert!(serial.converged, "the serial cavity must converge");
    let errs_s = cavity_ghia_errors(&hm, &serial.u);
    let u_scale = serial.u.iter().fold(0.0 as Scalar, |a, v| {
        a.max(v.x.abs()).max(v.y.abs()).max(v.z.abs())
    });
    let p_scale = serial.p.iter().fold(0.0 as Scalar, |a, &v| a.max(v.abs()));
    assert!(u_scale > 0.0 && p_scale >= 0.0);
    let it_s = serial.iterations as f64;

    for np in [1usize, 2, 4] {
        let (pname, dec) = cuts(&hm, np);
        let t0 = std::time::Instant::now();
        let got = ThreadComm::run(np, |comm| -> Result<ofgpu::distflow::CavityRun> {
            let gpu = Gpu::new(0)?;
            cavity_part(&gpu, comm, &dec)
        })
        .unwrap_or_else(|e| panic!("P={np} {pname}: {e}"));

        // Every rank Ok, and the world agreeing about what happened.
        let first = got[0]
            .as_ref()
            .unwrap_or_else(|e| panic!("P={np} {pname} rank 0: {e}"));
        let runs: Vec<&ofgpu::distflow::CavityRun> = got
            .iter()
            .enumerate()
            .map(|(r, x)| x.as_ref().unwrap_or_else(|e| panic!("P={np} {pname} rank {r}: {e}")))
            .collect();
        for (r, run) in runs.iter().enumerate() {
            assert!(run.converged, "P={np} {pname} rank {r}: must converge");
            assert_eq!(
                run.iterations,
                first.iterations,
                "P={np} {pname} rank {r}: iterations differ from rank 0"
            );
            assert_eq!(
                run.u_residual.to_bits(),
                first.u_residual.to_bits(),
                "P={np} {pname} rank {r}: u_residual bits differ from rank 0"
            );
        }
        assert_eq!(
            runs.iter().map(|r| r.global_cell.len()).sum::<usize>(),
            hm.n_cells,
            "P={np} {pname}: the parts must cover the mesh exactly once"
        );

        // Scatter the owned cells to whole-mesh order.
        let mut up = vec![Vec3::ZERO; hm.n_cells];
        let mut pp = vec![0.0 as Scalar; hm.n_cells];
        for run in &runs {
            for (i, &g) in run.global_cell.iter().enumerate() {
                up[g] = run.u[i];
                pp[g] = run.p[i];
            }
        }
        let u_rel = up
            .iter()
            .zip(&serial.u)
            .fold(0.0 as Scalar, |a, (x, w)| {
                a.max((x.x - w.x).abs())
                    .max((x.y - w.y).abs())
                    .max((x.z - w.z).abs())
            })
            / u_scale;
        let p_rel = pp
            .iter()
            .zip(&serial.p)
            .fold(0.0 as Scalar, |a, (&x, &w)| a.max((x - w).abs()))
            / p_scale;
        let errs = cavity_ghia_errors(&hm, &up);
        let worst_e = errs.iter().fold(0.0 as Scalar, |a, &x| a.max(x));
        let worst_s = errs_s.iter().fold(0.0 as Scalar, |a, &x| a.max(x));
        let band_ok = (first.iterations as f64 - it_s).abs() <= CAVITY_ITER_BAND * it_s;
        let field_ok = u_rel <= CAVITY_TOL_FIELD && p_rel <= CAVITY_TOL_FIELD;
        let centre_ok = errs
            .iter()
            .zip(&errs_s)
            .all(|(&ep, &es)| ep <= es + CAVITY_TOL_CENTRE);
        println!(
            "n={n} P={np:<2} {pname:<8}: iters {} on every rank (serial {}), \
             U rel {u_rel:.3e} <= {CAVITY_TOL_FIELD:e}, p rel {p_rel:.3e}, \
             worst centreline |e| {worst_e:.4} vs serial {worst_s:.4}, \
             band ok {band_ok}, centre ok {centre_ok}, {:.1} s",
            first.iterations,
            serial.iterations,
            t0.elapsed().as_secs_f64()
        );
        assert!(field_ok, "P={np} {pname}: U rel {u_rel:.3e}, p rel {p_rel:.3e}");
        assert!(band_ok, "P={np} {pname}: iterations {} against serial {it_s}", first.iterations);
        assert!(
            centre_ok,
            "P={np} {pname}: worst centreline |e| {worst_e:.4} vs serial {worst_s:.4}"
        );
    }
}

/// `cavity_bcs` writes the four walls and nothing else: on every part of a
/// 2-part cut, every processor face of `U` is still `Cyclic` (the coupled
/// branch every kernel tests) and every `empty` face still `Empty`, as
/// `GpuVectorField::zeros` seeded them.
#[test]
fn the_cavity_boundary_keeps_the_empty_and_processor_kinds() {
    use ofgpu::distflow::{cavity_bcs, cavity_controls};
    use ofgpu::field::BcKind;
    use ofgpu::mesh::{GpuMesh, PatchKind};
    use ofgpu::momentum::BuoyancyCoeffs;
    use ofgpu::simple::Simple;

    let Some(gpu) = gpu() else {
        println!("distflow_simple: no CUDA device - the kinds test passes vacuously");
        return;
    };
    let hm = cavity_mesh(8).expect("8x8 cavity");
    let dec = Decomposition::build(&hm, 2, &PartitionMethod::Linear).expect("cut 2");
    let mut checked = [0usize; 2];
    for part in &dec.parts {
        let gm = GpuMesh::upload_with_halo(&gpu, &part.mesh, part.n_halo).expect("part mesh");
        let mut s = Simple::new(&gpu, &part.mesh, &gm, cavity_controls(), BuoyancyCoeffs::default())
            .expect("simple");
        cavity_bcs(&gpu, &part.mesh, &mut s).expect("bcs");
        let kinds: Vec<ofgpu::Label> = gpu.download(&s.u().bc_kind).expect("kinds");
        for p in &part.mesh.patches {
            let want = match (p.kind, p.type_name.as_str()) {
                (PatchKind::Cyclic, "processor") => BcKind::Cyclic,
                (PatchKind::Empty, _) => BcKind::Empty,
                _ => continue,
            };
            for i in p.start..p.start + p.size {
                assert_eq!(
                    kinds[i],
                    want as ofgpu::Label,
                    "part {} patch {} face {i}: U kind {} is not {want:?}",
                    part.part,
                    p.name,
                    kinds[i]
                );
                checked[usize::from(want == BcKind::Empty)] += 1;
            }
        }
    }
    assert!(checked[0] > 0 && checked[1] > 0, "checked {checked:?} faces");
}

/// The refusal table: `set_exchange` names what a decomposed run cannot
/// carry, and `owner_value` refuses a world in which no rank owns the
/// reference cell. Both messages start `distflow: `.
#[test]
fn the_decomposed_simple_refuses_by_name() {
    use ofgpu::distflow::RankCtx;
    use ofgpu::fv::SnGradScheme;
    use ofgpu::io::case::DivScheme;
    use ofgpu::mesh::GpuMesh;
    use ofgpu::momentum::{BuoyancyCoeffs, MomentumControls};
    use ofgpu::simple::{Simple, SimpleControls};

    let Some(_gpu0) = gpu() else {
        println!("distflow_simple: no CUDA device - the refusals pass vacuously");
        return;
    };
    let hm = cavity_mesh(4).expect("4x4 cavity");

    // The three `set_exchange` refusals, each on its own solver in a 1-rank
    // world, each control otherwise clean so the named row is the one that
    // fires.
    let dec1 = Decomposition::build(&hm, 1, &PartitionMethod::Linear).expect("cut 1");
    let refused = |div: DivScheme, sn: SnGradScheme, vvs: bool| -> String {
        ThreadComm::run(1, |comm| -> String {
            let gpu = Gpu::new(0).expect("gpu");
            let part = &dec1.parts[comm.rank()];
            let gm = GpuMesh::upload_with_halo(&gpu, &part.mesh, part.n_halo)
                .expect("part mesh");
            let ctrl = SimpleControls {
                momentum: MomentumControls {
                    nu: 1e-3,
                    div_scheme: div,
                    sn_grad: sn,
                    variable_viscosity_stress: vvs,
                    ..MomentumControls::default()
                },
                ..SimpleControls::default()
            };
            let mut s = Simple::new(&gpu, &part.mesh, &gm, ctrl, BuoyancyCoeffs::default())
                .expect("simple");
            let ctx = RankCtx::new(&gpu, comm, part, &gm).expect("rank ctx");
            match s.set_exchange(Some(Box::new(ctx))) {
                Err(e) => format!("{e}"),
                Ok(()) => "NO REFUSAL".to_string(),
            }
        })
        .expect("1-rank world")
        .remove(0)
    };

    let msg = refused(DivScheme::LinearUpwind, SnGradScheme::Uncorrected, false);
    assert!(msg.contains("distflow: "), "div: {msg}");
    assert!(msg.contains("LinearUpwind"), "div: {msg}");

    let msg = refused(DivScheme::Central, SnGradScheme::Corrected, false);
    assert!(msg.contains("distflow: "), "snGrad: {msg}");
    assert!(msg.contains("snGrad"), "snGrad: {msg}");

    let msg = refused(DivScheme::Central, SnGradScheme::Uncorrected, true);
    assert!(msg.contains("distflow: "), "vvs: {msg}");
    assert!(msg.contains("variable-viscosity stress"), "vvs: {msg}");

    // `owner_value` with NO owner: both ranks of a 2-rank world refuse
    // alike, after the gather.
    use ofgpu::distflow::FlowExchange;
    let dec2 = Decomposition::build(&hm, 2, &PartitionMethod::Linear).expect("cut 2");
    let got = ThreadComm::run(2, |comm| -> Result<String> {
        let gpu = Gpu::new(0)?;
        let part = &dec2.parts[comm.rank()];
        let gm = GpuMesh::upload_with_halo(&gpu, &part.mesh, part.n_halo)?;
        let mut ctx = RankCtx::new(&gpu, comm, part, &gm)?;
        match ctx.owner_value(None) {
            Err(e) => Ok(format!("{e}")),
            Ok(_) => Ok("NO REFUSAL".to_string()),
        }
    })
    .expect("2-rank world");
    for (r, res) in got.iter().enumerate() {
        let msg = res.as_ref().expect("rank result");
        assert!(
            msg.contains("distflow: ") && msg.contains("owner_value"),
            "rank {r}: {msg}"
        );
    }
}
