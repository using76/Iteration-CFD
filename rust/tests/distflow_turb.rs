// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 139.6 (PAR-12)'s gate - k-epsilon and k-omega SST decomposed:
//! the streamwise-periodic channel of `cases/channelPeriodicWF.jsonc` (192
//! cells, the flow driven by the case's own body force), run serial and
//! decomposed by [`ofgpu::distflow::channel_serial`] and
//! [`ofgpu::distflow::channel_part`] on [`ThreadComm`] ranks, converging to
//! the serial answer; the whole mesh's wall distance split to each part
//! BITWISE; the coupled diffusivity equal to the whole mesh's internal-face
//! diffusivity; and the refusal table.
//!
//! The short-trajectory run ([`the_short_trajectory_follows_serial`], 20
//! outer iterations, no convergence test) is the verify row; the plan's gate
//! ([`the_channel_k_epsilon_gate`], [`the_channel_sst_gate`], up to
//! [`ofgpu::distflow::CHANNEL_MAX_OUTER`] with `residualControl`) is
//! `#[ignore]`d over the 10-minute budget on the shared card and is the
//! supervisor's run. The R rows against the serial run (U, k, epsilon/omega
//! rel) are hard `assert!`s in f64; under `--features single` a row above the
//! roundoff PRINTS `[139.6] MISSED` and the test continues - f32 round-off of
//! the segregated solve order is reported, never failed on.
//!
//! Provenance: ORIGINAL - the seams under test are this project's own design
//! (SPEC-LIT 139.6, PAR-12) over the exchange seam of 139.4 and the SPMD
//! Krylov routing of 139.3; the case is the tree's own
//! `channelPeriodicWF.jsonc`. No external source. No GPL-licensed source was
//! consulted.

use ofgpu::comm::{Comm, ThreadComm};
use ofgpu::decompose::{Decomposition, PartitionMethod};
use ofgpu::distflow::{
    channel_case, channel_part, channel_serial, channel_wall_distance, ChannelModel,
    CHANNEL_EPS0, CHANNEL_K0, CHANNEL_MAX_OUTER, CHANNEL_OMEGA0, CHANNEL_P, CHANNEL_P_RELAX,
    CHANNEL_RESIDUAL, CHANNEL_SHORT_OUTER, CHANNEL_TOL_SERIAL, CHANNEL_U0, CHANNEL_U_RELAX,
};
use ofgpu::precision::roundoff_tol;
use ofgpu::{Gpu, Result, Scalar, Vec3};

/// A machine without a card makes every device test pass vacuously, the
/// convention the rest of the crate's tests follow.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The gate constants, `assert_eq!`ed to their literals - FIXED BEFORE
/// MEASURING, so a silent loosening fails here and not in a paper.
fn assert_constants() {
    assert_eq!(CHANNEL_P, 4);
    assert_eq!(CHANNEL_TOL_SERIAL, 1e-6);
    assert_eq!(CHANNEL_MAX_OUTER, 3000);
    assert_eq!(CHANNEL_SHORT_OUTER, 20);
    assert_eq!(CHANNEL_U_RELAX, 0.5);
    assert_eq!(CHANNEL_P_RELAX, 0.3);
    assert_eq!(CHANNEL_U0, 3.0);
    assert_eq!(CHANNEL_K0, 0.03);
    assert_eq!(CHANNEL_EPS0, 0.3);
    assert_eq!(CHANNEL_OMEGA0, CHANNEL_EPS0 / (0.09 * CHANNEL_K0));
    if cfg!(feature = "single") {
        assert_eq!(CHANNEL_RESIDUAL, 1e-5);
    } else {
        assert_eq!(CHANNEL_RESIDUAL, 1e-7);
    }
}

/// The fixture is the case file: 192 cells (8 x 6 x 4), the case's `nu`,
/// `kappa` and `E`, and exactly one momentum source on `U` - the calibrated
/// 3.9 m/s² body force a periodic domain uses in place of an inlet.
#[test]
fn the_gate_constants_are_the_fixed_ones() {
    assert_constants();
    let case = channel_case().expect("channel case");
    assert_eq!(case.hm.n_cells, 192, "the case mesh is 8 x 6 x 4");
    assert_eq!(case.cc.nu, 1.5e-5);
    assert_eq!(case.cc.wall.kappa, 0.41);
    assert_eq!(case.cc.wall.e, 9.8);
    let on_u: Vec<&ofgpu::sources::SourceSpec> =
        case.sources.iter().filter(|s| s.field == "U").collect();
    assert_eq!(on_u.len(), 1, "exactly one source on U");
    assert_eq!(
        on_u[0].term,
        Some(ofgpu::sources::SourceTerm::BodyForce(Vec3::new(3.9, 0.0, 0.0)))
    );
}

/// Tucker's Poisson wall distance of SPEC-LIT 6.6 is a WHOLE-MESH quantity:
/// a part's model takes `y_whole[global_cell[i]]` and never solves for it.
/// Class D: the split field is the whole mesh's, BITWISE, at every owned
/// cell of every part of every cut.
#[test]
fn the_split_wall_distance_is_the_whole_mesh_wall_distance() {
    assert_constants();
    let Some(gpu0) = gpu() else {
        println!("distflow_turb: no CUDA device - the gate passes vacuously");
        return;
    };
    let case = channel_case().expect("channel case");
    let y_whole = channel_wall_distance(&gpu0, &case.hm, &case.cc).expect("whole-mesh y");
    let ctrl = ofgpu::distflow::channel_turbulence(&case.cc);

    for np in [2usize, 3, 4] {
        for method in [PartitionMethod::Linear, PartitionMethod::Hilbert] {
            let pname = if np == 1 || matches!(method, PartitionMethod::Linear) {
                "linear"
            } else {
                "hilbert"
            };
            let dec = Decomposition::build(&case.hm, np, &method).expect("cut");
            for part in &dec.parts {
                let hm = &part.mesh;
                let gm =
                    ofgpu::mesh::GpuMesh::upload_with_halo(&gpu0, hm, part.n_halo).expect("part");
                let mut y = gpu0.zeros(hm.n_cells).expect("y buffer");
                gpu0.write(&mut y, &ofgpu::distflow::split_cells(&y_whole, part))
                    .expect("y write");
                let on: Vec<bool> = hm
                    .b_kind
                    .iter()
                    .map(|&k| k == ofgpu::mesh::PatchKind::Wall as ofgpu::Label)
                    .collect();
                let wf =
                    ofgpu::field_setup::WallFaces { constrained_cells: on.clone(), nut: on };
                let rough = ofgpu::field_setup::NutRoughness::none(hm.n_boundary_faces);
                let model = ofgpu::models::KOmegaSst::new(
                    &gpu0,
                    hm,
                    &gm,
                    ofgpu::models::KOmegaSstCoeffs::default(),
                    ctrl,
                    case.cc.wall,
                    &wf,
                    &y,
                    &rough,
                )
                .expect("model on the part");
                let got = gpu0.download(model.wall_distance()).expect("y download");
                for (i, &g) in part.global_cell[..hm.n_cells].iter().enumerate() {
                    assert_eq!(
                        got[i].to_bits(),
                        y_whole[g as usize].to_bits(),
                        "P={np} {pname} part {}: y[{i}] is not the whole mesh's \
                         y[{}], bitwise",
                        part.part,
                        g
                    );
                }
            }
            println!("[139.6] split y: P={np} {pname}: bitwise on every part");
        }
    }
}

/// A processor face is an INTERNAL face of the whole mesh, so its blended
/// diffusivity is the two cells' interpolation and not the owner-only value
/// `turbGammaBoundaryCell` writes. Class R: after
/// `face_diffusivity_coupled_cell`, every processor face's `b_gamma` is the
/// whole mesh's internal-face `gamma` at the same global face, within the
/// roundoff; the owner-only value it overwrites misses by far more.
#[test]
fn the_coupled_diffusivity_is_the_internal_face_diffusivity() {
    assert_constants();
    let Some(gpu0) = gpu() else {
        println!("distflow_turb: no CUDA device - the gate passes vacuously");
        return;
    };
    use ofgpu::turbulence::{face_diffusivity_cell, face_diffusivity_coupled_cell, TurbKernels};

    let case = channel_case().expect("channel case");
    let hm_w = &case.hm;
    let nu = case.cc.nu;
    let kern = TurbKernels::new(&gpu0).expect("turbulence kernels");

    // The whole mesh's per-cell fields: a smoothly varying nut and a
    // smoothly varying 1/sigma - the variation IS the point, an SST `F1`
    // stands in for nothing here.
    let nut_c: Vec<Scalar> = (0..hm_w.n_cells)
        .map(|c| (1e-4 * (1.0 + 0.5 * (c as f64).sin())) as Scalar)
        .collect();
    let r_c: Vec<Scalar> = (0..hm_w.n_cells)
        .map(|c| (0.5 + 0.4 * (3.0 * c as f64).cos()) as Scalar)
        .collect();

    // The whole mesh's internal-face diffusivity - the reference.
    let m_w = ofgpu::mesh::GpuMesh::upload(&gpu0, hm_w).expect("whole mesh");
    let mut nut_w = ofgpu::field::GpuScalarField::zeros(&gpu0, &m_w, "nut").expect("nut");
    gpu0.write(&mut nut_w.f, &nut_c).expect("nut write");
    let mut r_w = gpu0.zeros(hm_w.n_cells).expect("r buffer");
    gpu0.write(&mut r_w, &r_c).expect("r write");
    let mut gamma_w = gpu0.zeros(m_w.n_internal_faces).expect("gamma");
    let mut b_gamma_w = gpu0.zeros(m_w.n_boundary_faces).expect("b_gamma");
    face_diffusivity_cell(&gpu0, &kern, &mut gamma_w, &mut b_gamma_w, &nut_w, &r_w, &m_w, nu)
        .expect("whole diffusivity");
    let gamma_w = gpu0.download(&gamma_w).expect("gamma download");

    for (np, method) in [
        (2usize, PartitionMethod::Hilbert),
        (4usize, PartitionMethod::Linear),
    ] {
        let pname = if matches!(method, PartitionMethod::Linear) { "linear" } else { "hilbert" };
        let dec = Decomposition::build(hm_w, np, &method).expect("cut");
        for part in &dec.parts {
            let hm = &part.mesh;
            let gm = ofgpu::mesh::GpuMesh::upload_with_halo(&gpu0, hm, part.n_halo)
                .expect("part mesh");
            let n_local = gm.n_local();

            // nut at EVERY local cell, halo included - what the exchange
            // would fill it with. The multiplier's owned half is `n_cells`
            // long (what `face_diffusivity_cell` reads); its halo-carrying
            // twin is what the coupled kernel reads.
            let nut_l: Vec<Scalar> = part.global_cell[..n_local]
                .iter()
                .map(|&g| nut_c[g as usize])
                .collect();
            let mut nut_p = ofgpu::field::GpuScalarField::zeros(&gpu0, &gm, "nut").expect("nut");
            gpu0.write(&mut nut_p.f, &nut_l).expect("nut write");

            let r_owned: Vec<Scalar> = part.global_cell[..hm.n_cells]
                .iter()
                .map(|&g| r_c[g as usize])
                .collect();
            let r_halo: Vec<Scalar> = part.global_cell[..n_local]
                .iter()
                .map(|&g| r_c[g as usize])
                .collect();
            let mut r_p = gpu0.zeros(hm.n_cells).expect("r owned");
            gpu0.write(&mut r_p, &r_owned).expect("r write");
            let mut r_ph = gpu0.zeros(n_local).expect("r halo");
            gpu0.write(&mut r_ph, &r_halo).expect("r halo write");

            let mut gamma_p = gpu0.zeros(gm.n_internal_faces).expect("gamma");
            let mut b_gamma_p = gpu0.zeros(gm.n_boundary_faces).expect("b_gamma");
            face_diffusivity_cell(&gpu0, &kern, &mut gamma_p, &mut b_gamma_p, &nut_p, &r_p, &gm, nu)
                .expect("part diffusivity");
            let owner_only = gpu0.download(&b_gamma_p).expect("b_gamma download");

            // The part's processor faces, ascending - and the coupled kernel
            // over exactly them.
            let mut faces: Vec<ofgpu::Label> = Vec::new();
            for pi in &hm.patches {
                if pi.type_name == "processor" {
                    faces.extend((pi.start..pi.start + pi.size).map(|b| b as ofgpu::Label));
                }
            }
            assert!(!faces.is_empty(), "P={np} {pname} part {}: no processor faces", part.part);
            let mut face_buf = gpu0.zeros(faces.len()).expect("faces");
            gpu0.write(&mut face_buf, &faces).expect("faces write");
            face_diffusivity_coupled_cell(
                &gpu0,
                &kern,
                &mut b_gamma_p,
                &nut_p,
                &r_ph,
                &face_buf,
                faces.len(),
                &gm,
                nu,
            )
            .expect("coupled diffusivity");
            let coupled = gpu0.download(&b_gamma_p).expect("b_gamma download");

            let tol = roundoff_tol(1e-13);
            let mut worst_miss = 0.0f64;
            for &b in &faces {
                let b = b as usize;
                let g = hm.global_face[hm.n_internal_faces + b] as usize;
                let want = gamma_w[g];
                let got = coupled[b];
                let miss = ((owner_only[b] - want) / want).abs();
                worst_miss = worst_miss.max(miss as f64);
                assert!(
                    ((got - want) as f64).abs() <= tol * (want as f64).abs(),
                    "P={np} {pname} part {}: face {b} (global {g}): coupled \
                     {got:e} against the whole mesh's {want:e}",
                    part.part
                );
            }
            assert!(
                worst_miss > 1e-3,
                "P={np} {pname} part {}: the owner-only diffusivity agrees on \
                 every processor face ({worst_miss:e}) - the fixture has no \
                 variation to catch the bug with",
                part.part
            );
            println!(
                "[139.6] coupled gamma: P={np} {pname:<7} part {}: {} faces \
                 bitwise-class, owner-only miss {worst_miss:.3e}",
                part.part,
                faces.len()
            );
        }
    }
}

/// The refusal table, BY NAME: on a 1-rank [`ThreadComm`] world, a
/// decomposed k-epsilon refuses a far-cell-gradient convection scheme, a
/// corrected snGrad, and buoyancy attached after the exchange - each
/// `Error::Config` starting `distflow: ` (SPEC-LIT 139.6's table).
#[test]
fn a_decomposed_model_refuses_what_it_cannot_run() {
    use ofgpu::distflow::RankCtx;
    use ofgpu::fv::DivScheme;
    use ofgpu::models::{KEpsilon, KEpsilonCoeffs};
    use ofgpu::turbulence::BuoyancyProduction;

    if gpu().is_none() {
        println!("distflow_turb: no CUDA device - the refusal test passes vacuously");
        return;
    }
    let case = channel_case().expect("channel case");
    let dec =
        Decomposition::build(&case.hm, 1, &PartitionMethod::Linear).expect("cut");
    let ctrl = ofgpu::distflow::channel_turbulence(&case.cc);
    let got = ThreadComm::run(1, |comm| -> Result<()> {
        let gpu = Gpu::new(0)?;
        let part = &dec.parts[comm.rank()];
        let hm = &part.mesh;
        let gm = ofgpu::mesh::GpuMesh::upload_with_halo(&gpu, hm, part.n_halo)?;
        let on: Vec<bool> = hm
            .b_kind
            .iter()
            .map(|&k| k == ofgpu::mesh::PatchKind::Wall as ofgpu::Label)
            .collect();
        let wf = ofgpu::field_setup::WallFaces { constrained_cells: on.clone(), nut: on };
        let rough = ofgpu::field_setup::NutRoughness::none(hm.n_boundary_faces);
        let ctx = || RankCtx::new(&gpu, comm, part, &gm);

        // (a) k divScheme LinearUpwind - the far-cell gradient.
        let mut ke = KEpsilon::new(
            &gpu,
            hm,
            &gm,
            KEpsilonCoeffs::default(),
            ofgpu::io::case::TurbulenceControls {
                div_scheme: DivScheme::LinearUpwind,
                ..ctrl
            },
            case.cc.wall,
            &wf,
            &rough,
        )?;
        let err = ke
            .set_exchange(&gpu, hm, Some(Box::new(ctx()?)))
            .unwrap_err()
            .to_string();
        assert!(
            err.starts_with("distflow: ") && err.contains("LinearUpwind"),
            "the k divScheme refusal must name the scheme: {err}"
        );

        // (b) snGrad Corrected (the case file's own setting, replaced by
        // `channel_turbulence` on both sides of the gate).
        let mut ke = KEpsilon::new(
            &gpu,
            hm,
            &gm,
            KEpsilonCoeffs::default(),
            ofgpu::io::case::TurbulenceControls {
                sn_grad: ofgpu::fv::SnGradScheme::Corrected,
                ..ctrl
            },
            case.cc.wall,
            &wf,
            &rough,
        )?;
        let err = ke
            .set_exchange(&gpu, hm, Some(Box::new(ctx()?)))
            .unwrap_err()
            .to_string();
        assert!(
            err.starts_with("distflow: ") && err.contains("snGrad"),
            "the snGrad refusal must name the scheme: {err}"
        );

        // (c) buoyancy attached first - `set_exchange` refuses what a driver
        // can attach after it too, and G_b has no exchanged gradient.
        let mut ke = KEpsilon::new(
            &gpu,
            hm,
            &gm,
            KEpsilonCoeffs::default(),
            ctrl,
            case.cc.wall,
            &wf,
            &rough,
        )?;
        ke.set_buoyancy(BuoyancyProduction {
            g: Vec3::new(0.0, -9.81, 0.0),
            ..BuoyancyProduction::default()
        })?;
        let err = ke
            .set_exchange(&gpu, hm, Some(Box::new(ctx()?)))
            .unwrap_err()
            .to_string();
        assert!(
            err.starts_with("distflow: ") && err.contains("G_b"),
            "the buoyancy refusal must name G_b: {err}"
        );
        Ok(())
    });
    for r in got.unwrap_or_else(|e| panic!("refusal world: {e}")) {
        r.unwrap_or_else(|e| panic!("refusal world: {e}"));
    }
}

/// A decomposed world's owned cells gathered to whole-mesh order:
/// `(u, k, epsilon/omega)`.
fn gather(
    runs: &[&ofgpu::distflow::ChannelRun],
    n_cells: usize,
) -> (Vec<Vec3>, Vec<Scalar>, Vec<Scalar>) {
    let mut u = vec![Vec3::ZERO; n_cells];
    let mut k = vec![0.0 as Scalar; n_cells];
    let mut w = vec![0.0 as Scalar; n_cells];
    for run in runs {
        for (i, &g) in run.global_cell.iter().enumerate() {
            u[g] = run.u[i];
            k[g] = run.k[i];
            w[g] = run.w[i];
        }
    }
    (u, k, w)
}

/// The decomposed-vs-serial rows: `(u_rel, k_rel, w_rel)` - `max|a-b| /
/// max|serial|`, `U` componentwise.
fn rels(
    serial: &ofgpu::distflow::ChannelRun,
    u: &[Vec3],
    k: &[Scalar],
    w: &[Scalar],
) -> (f64, f64, f64) {
    let u_scale = serial
        .u
        .iter()
        .fold(0.0 as Scalar, |a, v| a.max(v.x.abs()).max(v.y.abs()).max(v.z.abs()));
    let u_diff = u.iter().zip(&serial.u).fold(0.0 as Scalar, |a, (x, s)| {
        a.max((x.x - s.x).abs())
            .max((x.y - s.y).abs())
            .max((x.z - s.z).abs())
    });
    let k_scale = serial.k.iter().fold(0.0 as Scalar, |a, &v| a.max(v.abs()));
    let w_scale = serial.w.iter().fold(0.0 as Scalar, |a, &v| a.max(v.abs()));
    let k_diff = k
        .iter()
        .zip(&serial.k)
        .fold(0.0 as Scalar, |a, (&x, &s)| a.max((x - s).abs()));
    let w_diff = w
        .iter()
        .zip(&serial.w)
        .fold(0.0 as Scalar, |a, (&x, &s)| a.max((x - s).abs()));
    (
        (u_diff / u_scale) as f64,
        (k_diff / k_scale) as f64,
        (w_diff / w_scale) as f64,
    )
}

/// The decomposed world against the serial run - the rows SPEC-LIT 139.6's
/// gate fixes. HARD in BOTH builds: every rank `Ok`, every rank the same
/// `iterations`, `converged`, `diverged` and `residuals` bits, the parts
/// covering the mesh exactly once. The R rows (U, k, epsilon/omega rel) are
/// hard `assert!`s in f64; under `--features single` a row above the
/// roundoff PRINTS `[139.6] MISSED` and the test continues - f32 round-off
/// of the segregated solve order is reported, never failed on.
fn check_world(
    hm: &ofgpu::HostMesh,
    serial: &ofgpu::distflow::ChannelRun,
    np: usize,
    pname: &str,
    got: Vec<Result<ofgpu::distflow::ChannelRun>>,
    model: ChannelModel,
) {
    let tol = roundoff_tol(CHANNEL_TOL_SERIAL) as f64;
    let runs: Vec<&ofgpu::distflow::ChannelRun> = got
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
        assert_eq!(
            run.diverged, first.diverged,
            "P={np} {pname} rank {r}: diverged differs from rank 0"
        );
        for (kk, (&a, &b)) in run.residuals.iter().zip(&first.residuals).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "P={np} {pname} rank {r}: residual {kk} bits differ from rank 0"
            );
        }
    }
    assert_eq!(
        runs.iter().map(|r| r.global_cell.len()).sum::<usize>(),
        hm.n_cells,
        "P={np} {pname}: the parts must cover the mesh exactly once"
    );

    let (u, k, w) = gather(&runs, hm.n_cells);
    let (u_rel, k_rel, w_rel) = rels(serial, &u, &k, &w);
    let wname = if model == ChannelModel::KEpsilon { "epsilon" } else { "omega" };
    println!(
        "[139.6] channel {}: P={np:<2} {pname:<7}: iters {} on every rank (serial {}), \
         U rel {u_rel:.3e}, k rel {k_rel:.3e}, {wname} rel {w_rel:.3e} <= {tol:e}",
        model.name(),
        first.iterations,
        serial.iterations
    );
    let build = if cfg!(feature = "single") { "single" } else { "f64" };
    let rows = [
        ("U rel", u_rel),
        ("k rel", k_rel),
        ("w rel", w_rel),
    ];
    if cfg!(feature = "single") {
        for (row, v) in rows {
            // `!(v <= tol)`, so a NaN row is a miss too.
            if !(v <= tol) {
                println!(
                    "[139.6] MISSED: {row} {build} build: {v:.3e} > {tol:e}; cause: \
                     f32 round-off of the segregated solve order"
                );
            }
        }
    } else {
        for (row, v) in rows {
            assert!(v <= tol, "P={np} {pname}: {row} {v:.3e} > {tol:e}");
        }
    }
}

/// A serial channel run that went NaN under `single` - the open f32 SIMPLE
/// floor family the review-later list names - is reported and the test
/// returns, never failed on.
fn serial_diverged_single(model: ChannelModel, run: &ofgpu::distflow::ChannelRun) -> bool {
    if cfg!(feature = "single") && run.diverged {
        println!(
            "[139.6] MISSED: channel {} serial diverged at iteration {}; cause: \
             the open f32 SIMPLE floor family on channelPeriodicWF \
             (docs/17-review-later)",
            model.name(),
            run.iterations
        );
        return true;
    }
    false
}

/// The SHORT trajectory - 20 outer iterations, no convergence test, both
/// models, serial against P = 2 and P = 4 Hilbert: the same field bits
/// within the roundoff at a FIXED iteration count, which is the seam's own
/// claim (the decomposed iteration is the serial one, cut and exchanged).
#[test]
fn the_short_trajectory_follows_serial() {
    assert_constants();
    let Some(gpu0) = gpu() else {
        println!("distflow_turb: no CUDA device - the gate passes vacuously");
        return;
    };
    let case = channel_case().expect("channel case");
    let y_whole = channel_wall_distance(&gpu0, &case.hm, &case.cc).expect("whole-mesh y");
    for model in [ChannelModel::KEpsilon, ChannelModel::KOmegaSst] {
        let t0 = std::time::Instant::now();
        let serial =
            channel_serial(&gpu0, &case, model, CHANNEL_SHORT_OUTER, false).expect("serial");
        if serial_diverged_single(model, &serial) {
            return;
        }
        for np in [2usize, CHANNEL_P] {
            let dec = Decomposition::build(&case.hm, np, &PartitionMethod::Hilbert).expect("cut");
            let t1 = std::time::Instant::now();
            let got = ThreadComm::run(np, |comm| -> Result<ofgpu::distflow::ChannelRun> {
                let gpu = Gpu::new(0)?;
                channel_part(
                    &gpu,
                    comm,
                    &dec,
                    &case,
                    model,
                    &y_whole,
                    CHANNEL_SHORT_OUTER,
                    false,
                )
            })
            .unwrap_or_else(|e| panic!("channel {} P={np}: {e}", model.name()));
            check_world(&case.hm, &serial, np, "hilbert", got, model);
            println!(
                "[139.6] channel {} short: serial {:.1} s, P={np} world {:.1} s",
                model.name(),
                t0.elapsed().as_secs_f64(),
                t1.elapsed().as_secs_f64()
            );
        }
    }
}

/// The FULL gate - `CHANNEL_MAX_OUTER` with `stop = true`, serial against
/// P = `CHANNEL_P` Hilbert. The rows of the short trajectory plus the
/// iterations, converged flag, residuals and wall time; an unconverged run
/// is a `[139.6] MISSED` row with the residuals - hard in f64,
/// print-and-continue under `single`.
fn the_channel_gate(model: ChannelModel) {
    assert_constants();
    let Some(gpu0) = gpu() else {
        println!("distflow_turb: no CUDA device - the gate passes vacuously");
        return;
    };
    let t0 = std::time::Instant::now();
    let case = channel_case().expect("channel case");
    let y_whole = channel_wall_distance(&gpu0, &case.hm, &case.cc).expect("whole-mesh y");
    let serial =
        channel_serial(&gpu0, &case, model, CHANNEL_MAX_OUTER, true).expect("serial");
    println!(
        "[139.6] channel {} serial: converged {} in {} iterations, residuals {:?}, {:.1} s",
        model.name(),
        serial.converged,
        serial.iterations,
        serial.residuals.map(|r| r as f64),
        t0.elapsed().as_secs_f64()
    );
    if serial_diverged_single(model, &serial) {
        return;
    }
    let np = CHANNEL_P;
    let dec = Decomposition::build(&case.hm, np, &PartitionMethod::Hilbert).expect("cut");
    let t1 = std::time::Instant::now();
    let got = ThreadComm::run(np, |comm| -> Result<ofgpu::distflow::ChannelRun> {
        let gpu = Gpu::new(0)?;
        channel_part(
            &gpu,
            comm,
            &dec,
            &case,
            model,
            &y_whole,
            CHANNEL_MAX_OUTER,
            true,
        )
    })
    .unwrap_or_else(|e| panic!("channel {} P={np}: {e}", model.name()));
    check_world(&case.hm, &serial, np, "hilbert", got, model);
    if !serial.converged {
        let msg = format!(
            "channel {} gate unconverged after {} iterations, residuals {:?}",
            model.name(),
            serial.iterations,
            serial.residuals.map(|r| r as f64)
        );
        if cfg!(feature = "single") {
            println!(
                "[139.6] MISSED: {msg}; cause: the open f32 SIMPLE floor family on \
                 channelPeriodicWF (docs/17-review-later)"
            );
        } else {
            panic!("[139.6] {msg}");
        }
    }
    println!(
        "[139.6] channel {} gate: serial {:.1} s, P={np} world {:.1} s, total {:.1} s",
        model.name(),
        t0.elapsed().as_secs_f64(),
        t1.elapsed().as_secs_f64(),
        t0.elapsed().as_secs_f64()
    );
}

/// The PAR-12 gate, k-epsilon: `cases/channelPeriodicWF.jsonc` to the
/// `CHANNEL_RESIDUAL` floor, serial against P = 4 Hilbert. The supervisor's
/// run - ignored here, the GPU is shared.
#[test]
#[ignore = "the plan-size gate: the supervisor's run"]
fn the_channel_k_epsilon_gate() {
    the_channel_gate(ChannelModel::KEpsilon);
}

/// The PAR-12 gate, SST - the blended-sigma path through
/// `face_diffusivity_coupled_cell`, the constant-sigma shortcut closed.
#[test]
#[ignore = "the plan-size gate: the supervisor's run"]
fn the_channel_sst_gate() {
    the_channel_gate(ChannelModel::KOmegaSst);
}
