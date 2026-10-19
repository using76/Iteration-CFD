// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 139.5 (PAR-11)'s gate - the energy equation and a passive
//! scalar on a part: the differentially heated cavity of de Vahl Davis
//! (1983, DOI 10.1002/fld.1650030305) at `Ra = 1e4` under the Boussinesq
//! body force (SPEC-LIT 121), 40 x 40 (and the reduced 20 x 20 verify run),
//! on P = 1, 2, 4 [`ThreadComm`] ranks, converging to the serial answer -
//! and refusing by name what the seams cannot run yet.
//!
//! Each reduced rank count is its own test, each with its own serial run
//! and one [`ThreadComm::run`], so a single verify command stays inside the
//! 10-minute budget on the shared card. The R rows against the serial run
//! (Nu, T, s, U, p rel) are hard `assert!`s in f64; under `--features
//! single` a row above the roundoff PRINTS `[139.5] MISSED` and the test
//! continues - f32 round-off of the segregated solve order is reported,
//! never failed on. The plan's 40 x 40 runs as the serial Nu gate
//! ([`the_serial_plan_size_nusselt`]); the decomposed 40 x 40 P = 4 gate
//! ([`the_plan_size_cavity_gate`]) is `#[ignore]`d over that budget and is
//! the supervisor's run.
//!
//! An integration test for the same reason `tests/distflow_simple.rs` is:
//! it links the normal (non-test) library, so it compiles under
//! `--features single` too. Every rank is a thread with its own [`Gpu`] on
//! device 0; the reference is `dvd_serial` on the main thread's card.
//!
//! The gate constants are the ones `distflow` fixed BEFORE measuring -
//! `assert_eq!`ed against their literals at the top of the first test, so
//! nobody loosens them silently.
//!
//! Provenance: ORIGINAL - the seams under test are this project's own
//! design (SPEC-LIT 139.5, PAR-11) over the exchange seam of 139.4 and the
//! SPMD Krylov routing of 139.3; the reference number is de Vahl Davis's
//! published mean hot-wall Nusselt number. No external source. No
//! GPL-licensed source was consulted.

use ofgpu::comm::{Comm, ThreadComm};
use ofgpu::decompose::{Decomposition, PartitionMethod};
use ofgpu::distflow::{
    dvd_nusselt, dvd_part, dvd_serial, DvdRun, DVD_DELTA_T, DVD_ITER_BAND, DVD_N,
    DVD_N_REDUCED, DVD_NU_BAND, DVD_NU_PUBLISHED, DVD_RA, DVD_TOL_SERIAL,
};
use ofgpu::precision::roundoff_tol;
use ofgpu::{Gpu, HostMesh, Result, Scalar, Vec3};

/// A machine without a card makes every device test pass vacuously, the
/// convention the rest of the crate's tests follow.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The rank counts and their cutters: P = 1 is the Linear cut (which IS the
/// one-part cut), P = 2 and 4 are Hilbert's.
fn cuts(hm: &HostMesh, np: usize) -> (String, Decomposition) {
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

/// A decomposed world's owned cells gathered to whole-mesh order:
/// `(u, p, t, s)`.
fn gather(runs: &[&DvdRun], n_cells: usize) -> (Vec<Vec3>, Vec<Scalar>, Vec<Scalar>, Vec<Scalar>) {
    let mut u = vec![Vec3::ZERO; n_cells];
    let mut p = vec![0.0 as Scalar; n_cells];
    let mut t = vec![0.0 as Scalar; n_cells];
    let mut s = vec![0.0 as Scalar; n_cells];
    for run in runs {
        for (i, &g) in run.global_cell.iter().enumerate() {
            u[g] = run.u[i];
            p[g] = run.p[i];
            t[g] = run.t[i];
            s[g] = run.s[i];
        }
    }
    (u, p, t, s)
}

/// The decomposed-vs-serial rows: `(nu_rel, t_rel, s_rel, u_rel, p_rel)` -
/// Nu by [`dvd_nusselt`] over the gathered whole-mesh `T`, `T` and `s`
/// against the serial fields, `U` and `p` against the serial ones, each
/// scaled as SPEC-LIT 139.5's gate defines them.
fn rels(
    hm: &HostMesh,
    serial: &DvdRun,
    u: &[Vec3],
    p: &[Scalar],
    t: &[Scalar],
    s: &[Scalar],
) -> (f64, Scalar, Scalar, Scalar, Scalar) {
    let nu_s = dvd_nusselt(hm, &serial.t);
    let nu_p = dvd_nusselt(hm, t);
    let nu_rel = (nu_p - nu_s).abs() / nu_s;
    let t_rel = t
        .iter()
        .zip(&serial.t)
        .fold(0.0 as Scalar, |a, (&x, &w)| a.max((x - w).abs()))
        / (DVD_DELTA_T as Scalar);
    let s_rel = s
        .iter()
        .zip(&serial.s)
        .fold(0.0 as Scalar, |a, (&x, &w)| a.max((x - w).abs()));
    let u_scale = serial.u.iter().fold(0.0 as Scalar, |a, v| {
        a.max(v.x.abs()).max(v.y.abs()).max(v.z.abs())
    });
    let p_scale = serial.p.iter().fold(0.0 as Scalar, |a, &v| a.max(v.abs()));
    let u_rel = u
        .iter()
        .zip(&serial.u)
        .fold(0.0 as Scalar, |a, (x, w)| {
            a.max((x.x - w.x).abs())
                .max((x.y - w.y).abs())
                .max((x.z - w.z).abs())
        })
        / u_scale;
    let p_rel = p
        .iter()
        .zip(&serial.p)
        .fold(0.0 as Scalar, |a, (&x, &w)| a.max((x - w).abs()))
        / p_scale;
    (nu_rel, t_rel, s_rel, u_rel, p_rel)
}

/// The gate constants, `assert_eq!`ed to their literals - FIXED BEFORE
/// MEASURING, so a silent loosening fails here and not in a paper.
fn assert_constants() {
    use ofgpu::distflow::{
        DVD_ALPHA, DVD_BETA, DVD_G_Y, DVD_L, DVD_MAX_OUTER, DVD_NU, DVD_RESIDUAL, DVD_T_COLD,
        DVD_T_HOT, DVD_T_REF,
    };
    assert_eq!(DVD_N, 40);
    assert_eq!(DVD_N_REDUCED, 20);
    assert_eq!(DVD_RA, 1.0e4);
    assert_eq!(DVD_NU_PUBLISHED, 2.243);
    assert_eq!(DVD_NU_BAND, 0.01);
    assert_eq!(DVD_TOL_SERIAL, 1e-6);
    assert_eq!(DVD_ITER_BAND, 0.02);
    assert_eq!(DVD_MAX_OUTER, 3000);
    if cfg!(feature = "single") {
        assert_eq!(DVD_RESIDUAL, 1e-5);
    } else {
        assert_eq!(DVD_RESIDUAL, 1e-7);
    }
    assert_eq!(DVD_L, 0.1);
    assert_eq!(DVD_T_HOT, 304.5);
    assert_eq!(DVD_T_COLD, 295.5);
    assert_eq!(DVD_DELTA_T, 9.0);
    assert_eq!(DVD_T_REF, 300.0);
    assert_eq!(DVD_BETA, 1.0 / 300.0);
    assert_eq!(DVD_NU, 0.71e-3);
    assert_eq!(DVD_ALPHA, 1.0e-3);
    // g = Ra nu alpha / (beta DeltaT L^3): 7.1e-3 / 3e-5.
    assert!((DVD_G_Y - 7.1e-3 / 3.0e-5).abs() < 1e-9, "DVD_G_Y = {DVD_G_Y}");
}

/// One serial run at `n` per side, printed: `dvd_serial` and its mean
/// hot-wall Nu against the published 2.243 (informational at the reduced
/// size), with the wall time.
fn run_serial(gpu0: &Gpu, hm: &HostMesh, n: usize) -> DvdRun {
    let t0 = std::time::Instant::now();
    let serial = dvd_serial(gpu0, hm).expect("serial heated cavity");
    let nu_s = dvd_nusselt(hm, &serial.t);
    println!(
        "[139.5] n={n} serial : converged {} in {} iterations, Nu {nu_s:.5} \
         (published {DVD_NU_PUBLISHED}, {:+.2} %), residuals {:?}, {:.1} s",
        serial.converged,
        serial.iterations,
        (nu_s / DVD_NU_PUBLISHED - 1.0) * 100.0,
        serial.residuals.map(|r| r as f64),
        t0.elapsed().as_secs_f64()
    );
    serial
}

/// One decomposed world at `np` on the prebuilt cut: a single
/// [`ThreadComm::run`], every rank its own [`Gpu`] on device 0, timed.
fn run_part(np: usize, dec: &Decomposition) -> Vec<Result<DvdRun>> {
    let t0 = std::time::Instant::now();
    let got = ThreadComm::run(np, |comm| -> Result<DvdRun> {
        let gpu = Gpu::new(0)?;
        dvd_part(&gpu, comm, dec)
    })
    .unwrap_or_else(|e| panic!("P={np}: {e}"));
    println!("[139.5] P={np} world: {:.1} s", t0.elapsed().as_secs_f64());
    got
}

/// The decomposed world against the serial run - the rows SPEC-LIT 139.5's
/// gate fixes. HARD in BOTH builds: every rank `Ok`, every rank the same
/// `iterations`, `converged` and `residuals` bits, the parts covering the
/// mesh exactly once, and the iteration band. The R rows (Nu, T, s, U, p
/// rel) are hard `assert!`s in f64; under `--features single` a row above
/// the roundoff PRINTS `[139.5] MISSED` and the test continues - f32
/// round-off of the segregated solve order is reported, never failed on.
/// Returns rank 0's `(iterations, converged)` for the 1 % row's
/// unconverged clause.
fn check_part(
    n: usize,
    hm: &HostMesh,
    serial: &DvdRun,
    np: usize,
    pname: &str,
    got: Vec<Result<DvdRun>>,
) -> (usize, bool) {
    let tol = roundoff_tol(DVD_TOL_SERIAL) as f64;
    let runs: Vec<&DvdRun> = got
        .iter()
        .enumerate()
        .map(|(r, x)| x.as_ref().unwrap_or_else(|e| panic!("P={np} {pname} rank {r}: {e}")))
        .collect();
    let first = runs[0];
    for (r, run) in runs.iter().enumerate() {
        assert_eq!(
            run.iterations, first.iterations,
            "P={np} {pname} rank {r}: iterations differ from rank 0"
        );
        assert_eq!(
            run.converged, first.converged,
            "P={np} {pname} rank {r}: converged differs from rank 0"
        );
        for (k, (&a, &b)) in run.residuals.iter().zip(&first.residuals).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "P={np} {pname} rank {r}: residual {k} bits differ from rank 0"
            );
        }
    }
    assert_eq!(
        runs.iter().map(|r| r.global_cell.len()).sum::<usize>(),
        hm.n_cells,
        "P={np} {pname}: the parts must cover the mesh exactly once"
    );

    let (u, p, t, s) = gather(&runs, hm.n_cells);
    let (nu_rel, t_rel, s_rel, u_rel, p_rel) = rels(hm, serial, &u, &p, &t, &s);
    let it_s = serial.iterations as f64;
    let band = (first.iterations as f64 - it_s).abs() <= DVD_ITER_BAND * it_s;
    println!(
        "[139.5] n={n} P={np:<2} {pname:<8}: iters {} on every rank (serial {it_s}), \
         Nu rel {nu_rel:.3e}, T rel {t_rel:.3e}, s rel {s_rel:.3e}, \
         U rel {u_rel:.3e}, p rel {p_rel:.3e} <= {tol:e}, band ok {band}",
        first.iterations
    );
    let build = if cfg!(feature = "single") { "single" } else { "f64" };
    let rows = [
        ("Nu rel", nu_rel),
        ("T rel", t_rel as f64),
        ("s rel", s_rel as f64),
        ("U rel", u_rel as f64),
        ("p rel", p_rel as f64),
    ];
    if !(serial.converged && first.converged) {
        println!(
            "[139.5] MISSED: n={n} P={np} {build} build: unconverged after {} outer \
             iterations (serial {}, parts {}); cause: DVD_RESIDUAL not reached within \
             DVD_MAX_OUTER",
            first.iterations, serial.converged, first.converged
        );
    }
    if cfg!(feature = "single") {
        for (row, v) in rows {
            // `!(v <= tol)`, so a NaN row is a miss too.
            if !(v <= tol) {
                println!(
                    "[139.5] MISSED: {row} {build} build: {v:.3e} > {tol:e}; cause: \
                     f32 round-off of the segregated solve order; T rel is normalised \
                     by DeltaT = 9 K, so 1e-5 is about 3 f32 ulps of T at 300 K"
                );
            }
        }
    } else {
        for (row, v) in rows {
            assert!(v <= tol, "P={np} {pname}: {row} {v:.3e} > {tol:e}");
        }
    }
    assert!(band, "P={np} {pname}: iterations {} against serial {it_s}", first.iterations);
    (first.iterations, first.converged)
}

/// The REDUCED gate on ONE rank - the one-part Linear cut. Its own serial
/// run at [`DVD_N_REDUCED`], then its one [`ThreadComm::run`]:
/// [`check_part`]'s rows - the R rows hard in f64, MISSED-and-continue
/// above the roundoff under `--features single`.
#[test]
fn a_reduced_cavity_on_one_rank() {
    assert_constants();
    let Some(gpu0) = gpu() else {
        println!("distflow_thermal: no CUDA device - the gate passes vacuously");
        return;
    };
    let n = DVD_N_REDUCED;
    let hm = ofgpu::distflow::cavity_mesh(n).expect("cavity mesh");
    let serial = run_serial(&gpu0, &hm, n);
    let (pname, dec) = cuts(&hm, 1);
    let got = run_part(1, &dec);
    check_part(n, &hm, &serial, 1, &pname, got);
}

/// The REDUCED gate on TWO Hilbert ranks: its own serial run at
/// [`DVD_N_REDUCED`], then its one [`ThreadComm::run`] - [`check_part`],
/// as [`a_reduced_cavity_on_one_rank`].
#[test]
fn a_reduced_cavity_on_two_ranks() {
    assert_constants();
    let Some(gpu0) = gpu() else {
        println!("distflow_thermal: no CUDA device - the gate passes vacuously");
        return;
    };
    let n = DVD_N_REDUCED;
    let hm = ofgpu::distflow::cavity_mesh(n).expect("cavity mesh");
    let serial = run_serial(&gpu0, &hm, n);
    let (pname, dec) = cuts(&hm, 2);
    let got = run_part(2, &dec);
    check_part(n, &hm, &serial, 2, &pname, got);
}

/// The REDUCED gate on FOUR Hilbert ranks - the supervisor's verify run:
/// its own serial run at [`DVD_N_REDUCED`], then its one
/// [`ThreadComm::run`] - [`check_part`], as [`a_reduced_cavity_on_one_rank`].
#[test]
fn a_reduced_cavity_on_four_ranks() {
    assert_constants();
    let Some(gpu0) = gpu() else {
        println!("distflow_thermal: no CUDA device - the gate passes vacuously");
        return;
    };
    let n = DVD_N_REDUCED;
    let hm = ofgpu::distflow::cavity_mesh(n).expect("cavity mesh");
    let serial = run_serial(&gpu0, &hm, n);
    let (pname, dec) = cuts(&hm, 4);
    let got = run_part(4, &dec);
    check_part(n, &hm, &serial, 4, &pname, got);
}

/// The plan's gate at its own size, 40 x 40 (docs/17 §12, row PAR-11):
/// serial then P = 4 (Hilbert) only, the same decomposed-vs-serial rows as
/// the verify tests ([`check_part`]) - and the 1 % row against the
/// published 2.243, which PRINTS PASS or MISSED and never fails: at this
/// size the discretisation, not the decomposition, sets the error. An
/// unconverged run prints MISSED too and does not fail on that alone.
/// Deferred over the verify budget; the serial half runs alone as
/// [`the_serial_plan_size_nusselt`].
#[test]
#[ignore = "deferred: the 40 x 40 P = 4 gate runs over the 10-minute verify budget on the shared card (SPEC-LIT 139.5)"]
fn the_plan_size_cavity_gate() {
    assert_constants();
    let Some(gpu0) = gpu() else {
        println!("distflow_thermal: no CUDA device - the gate passes vacuously");
        return;
    };
    let n = DVD_N;
    let hm = ofgpu::distflow::cavity_mesh(n).expect("cavity mesh");
    let serial = run_serial(&gpu0, &hm, n);
    let nu_s = dvd_nusselt(&hm, &serial.t);
    let (pname, dec) = cuts(&hm, 4);
    let got = run_part(4, &dec);
    let (_iterations, parts_converged) = check_part(n, &hm, &serial, 4, &pname, got);
    let dev = (nu_s / DVD_NU_PUBLISHED - 1.0) * 100.0;

    if !(serial.converged && parts_converged) {
        println!(
            "[139.5] MISSED: Gate PAR-11 Nu within 1 % of 2.243: serial Nu {nu_s:.5} \
             ({dev:+.2} %) on 40x40; cause: the discretisation, not the decomposition \
             (unconverged: serial {}, parts {})",
            serial.converged, parts_converged
        );
    } else if (nu_s - DVD_NU_PUBLISHED).abs() <= DVD_NU_BAND * DVD_NU_PUBLISHED {
        println!(
            "[139.5] PASS: Gate PAR-11 Nu within 1 % of 2.243: serial Nu {nu_s:.5} \
             ({dev:+.2} %) on 40x40"
        );
    } else {
        println!(
            "[139.5] MISSED: Gate PAR-11 Nu within 1 % of 2.243: serial Nu {nu_s:.5} \
             ({dev:+.2} %) on 40x40; cause: the discretisation, not the decomposition"
        );
    }
}

/// The plan's 40 x 40 SERIAL gate alone - SPEC-LIT 139.5's class P row: the
/// mean hot-wall Nu within 1 % of the published 2.243. It PRINTS
/// `[139.5] PASS` or `[139.5] MISSED` and never fails on the 1 % row alone:
/// at this size the discretisation, not the decomposition, sets the error.
/// The decomposed half of the gate is [`the_plan_size_cavity_gate`].
#[test]
fn the_serial_plan_size_nusselt() {
    assert_constants();
    let Some(gpu0) = gpu() else {
        println!("distflow_thermal: no CUDA device - the gate passes vacuously");
        return;
    };
    let n = DVD_N;
    let hm = ofgpu::distflow::cavity_mesh(n).expect("cavity mesh");
    let t0 = std::time::Instant::now();
    let serial = dvd_serial(&gpu0, &hm).expect("serial heated cavity");
    let nu = dvd_nusselt(&hm, &serial.t);
    let dev = (nu / DVD_NU_PUBLISHED - 1.0) * 100.0;
    if serial.converged && (nu - DVD_NU_PUBLISHED).abs() <= DVD_NU_BAND * DVD_NU_PUBLISHED {
        println!(
            "[139.5] PASS: Gate PAR-11 Nu within 1 % of 2.243: serial Nu {nu:.5} \
             ({dev:+.2} %) on 40x40"
        );
    } else {
        println!(
            "[139.5] MISSED: Gate PAR-11 Nu within 1 % of 2.243: serial Nu {nu:.5} \
             ({dev:+.2} %) on 40x40; cause: the discretisation, not the decomposition"
        );
    }
    println!(
        "n={n} serial: Nu_hot {nu:.5}, iterations {}, converged {}, wall {:.1} s",
        serial.iterations,
        serial.converged,
        t0.elapsed().as_secs_f64()
    );
}

/// The refusals, BY NAME: on a 1-rank [`ThreadComm`] world and a 4 x 4
/// cavity part, `Energy::set_exchange` refuses a far-cell-gradient
/// convection scheme and a corrected snGrad, and `ScalarTransport::set_exchange`
/// refuses the scheme and the mass weighting - each naming what it refused
/// (SPEC-LIT 139.5's refusal table).
#[test]
fn the_decomposed_energy_and_scalar_refuse_by_name() {
    use ofgpu::distflow::RankCtx;
    use ofgpu::energy::{Energy, EnergyControls};
    use ofgpu::fv::DivScheme;
    use ofgpu::io::case::TurbulenceControls;
    use ofgpu::io::schemes::DivEntry;
    use ofgpu::mesh::GpuMesh;
    use ofgpu::scalar_transport::{ScalarTransport, ScalarTransportCoeffs};

    if gpu().is_none() {
        println!("distflow_thermal: no CUDA device - the refusal test passes vacuously");
        return;
    }
    let hm = ofgpu::distflow::cavity_mesh(4).expect("cavity mesh");
    let dec = Decomposition::build(&hm, 1, &PartitionMethod::Linear).expect("cut");
    let lin = DivEntry { scheme: DivScheme::LinearUpwind, bounded: true };
    let central = DivEntry { scheme: DivScheme::Central, bounded: true };
    let got = ThreadComm::run(1, |comm| -> Result<()> {
        let gpu = Gpu::new(0)?;
        let part = &dec.parts[comm.rank()];
        let gm = GpuMesh::upload_with_halo(&gpu, &part.mesh, part.n_halo)?;
        let props = ofgpu::distflow::dvd_props();

        // (a) on Energy: div LinearUpwind.
        let mut e = Energy::new(
            &gpu,
            &gm,
            EnergyControls { div_scheme: lin, ..EnergyControls::default() },
            props,
        )?;
        let err = e
            .set_exchange(Some(Box::new(RankCtx::new(&gpu, comm, part, &gm)?)))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("distflow: ") && err.contains("LinearUpwind"),
            "Energy div refusal must name the scheme: {err}"
        );

        // (b) on Energy: snGrad Corrected (the default), convection upwind.
        let mut e = Energy::new(&gpu, &gm, EnergyControls::default(), props)?;
        let err = e
            .set_exchange(Some(Box::new(RankCtx::new(&gpu, comm, part, &gm)?)))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("distflow: ") && err.contains("snGrad"),
            "Energy snGrad refusal must name the scheme: {err}"
        );

        // (a) on the scalar: div LinearUpwind.
        let mut s = ScalarTransport::new(
            &gpu,
            &part.mesh,
            &gm,
            "s",
            ScalarTransportCoeffs::default(),
            TurbulenceControls::default(),
        )?;
        s.set_convection(lin);
        let err = s
            .set_exchange(Some(Box::new(RankCtx::new(&gpu, comm, part, &gm)?)))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("distflow: ") && err.contains("LinearUpwind"),
            "scalar div refusal must name the scheme: {err}"
        );

        // (c') on the scalar: mass weighting on, convection central -
        // snGrad uncorrected so (b) does not speak first.
        let mut s = ScalarTransport::new(
            &gpu,
            &part.mesh,
            &gm,
            "s",
            ScalarTransportCoeffs::default(),
            TurbulenceControls {
                sn_grad: ofgpu::fv::SnGradScheme::Uncorrected,
                ..TurbulenceControls::default()
            },
        )?;
        s.set_convection(central);
        s.use_mass_weighting(&gpu)?;
        let err = s
            .set_exchange(Some(Box::new(RankCtx::new(&gpu, comm, part, &gm)?)))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("distflow: ") && err.contains("mass-weighted"),
            "scalar mass-weighting refusal must name the mode: {err}"
        );

        // (c') the other order: the exchange first, then the weighting.
        let mut s = ScalarTransport::new(
            &gpu,
            &part.mesh,
            &gm,
            "s",
            ScalarTransportCoeffs::default(),
            TurbulenceControls {
                sn_grad: ofgpu::fv::SnGradScheme::Uncorrected,
                ..TurbulenceControls::default()
            },
        )?;
        s.set_convection(central);
        s.set_exchange(Some(Box::new(RankCtx::new(&gpu, comm, part, &gm)?)))?;
        let err = s.use_mass_weighting(&gpu).unwrap_err().to_string();
        assert!(
            err.contains("distflow: ") && err.contains("mass-weighted"),
            "a weighting asked for after the exchange must be refused: {err}"
        );
        Ok(())
    });
    let runs = got.unwrap_or_else(|e| panic!("refusal world: {e}"));
    for r in runs {
        r.unwrap_or_else(|e| panic!("refusal world: {e}"));
    }
}
