// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The capture gates themselves: one per module the registry marks `Gate`.
//!
//! `SPEC-LIT` 81.9. A gate may live anywhere in the tree - the registry looks
//! its name up across `src/`, and `src/soot/tests.rs` deliberately keeps its
//! own, to show that it can. Most live here, because most of them want the
//! same three lines of scaffolding and there is no reason to write those
//! nineteen times.
//!
//! Every gate here is the same shape and it is not an accident: build,
//! iterate, read back. What differs between two of them is the physics, and
//! that is the only thing that should differ.
//!
//! Provenance: ORIGINAL. The meshes are this crate's own generator, the
//! coefficients are each module's published defaults, and nothing here
//! computes physics - it compares this crate against itself. No GPL-licensed
//! source was consulted.

use crate::capture::{buf, capture_replays_bitwise, field};
use crate::error::Result;
use crate::field::{GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use crate::field_setup::{NutRoughness, WallFaces};
use crate::turbulence::{FlowState, TurbulenceControls};
use crate::wallfunctions::WallFunctionCoeffs;
use crate::{DevBuf, Gpu, GpuMesh, HostMesh, Scalar, Vec3};

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// A closed 4x4x4 box: the operators need a mesh, and nothing here is a
/// statement about a mesh.
fn box4() -> HostMesh {
    let (mut m, points, faces) =
        crate::mesh::topology::tests::box_mesh([4, 4, 4], Vec3::new(0.25, 0.25, 0.25));
    m.compute_geometry(&points, &faces).expect("geometry");
    m.build_cell_face_maps();
    m
}

/// Turbulence controls with **`fixed_iters` on**.
///
/// This is the whole reason the fixed-iteration mode exists. The adaptive
/// residual test DMAs a convergence flag to the host every `check_interval`
/// sweeps, and `SPEC-LIT` 81.3 now refuses that inside a capture by name
/// rather than letting the driver return an unattributed error. A gate that
/// forgot this line would fail with that message, which is the point.
fn fixed(dt: Scalar) -> TurbulenceControls {
    let mut ctrl = TurbulenceControls {
        steady: false,
        delta_t: dt,
        k_relax: 1.0,
        eps_relax: 1.0,
        nut_max_coeff: 1e8,
        ..Default::default()
    };
    ctrl.k_solver.fixed_iters = true;
    ctrl.k_solver.max_iter = 4;
    ctrl.k_solver.report_residuals = false;
    ctrl.epsilon_solver = ctrl.k_solver;
    ctrl
}

/// Wall distance and its gradient, for the models that need one. Values are
/// arbitrary and positive; a bitwise-replay gate is a statement about
/// determinism, not about a boundary layer.
fn wall_distance(gpu: &Gpu, n: usize) -> Result<(DevBuf<Scalar>, DevBuf<Vec3>)> {
    let y: Vec<Scalar> = (0..n).map(|i| 0.01 + 0.002 * (i % 7) as Scalar).collect();
    let g: Vec<Vec3> = (0..n).map(|_| Vec3::new(0.0, 1.0, 0.0)).collect();
    Ok((gpu.upload(&y)?, gpu.upload(&g)?))
}

/// A quiescent flow state: `U = 0`, `phi = 0`. Every cell then evolves by the
/// model's own equations, which is the cleanest thing to replay.
struct Quiet {
    u: GpuVectorField,
    phi: GpuSurfaceScalarField,
}

impl Quiet {
    fn new(gpu: &Gpu, m: &GpuMesh) -> Result<Self> {
        Ok(Self {
            u: GpuVectorField::zeros(gpu, m, "U")?,
            phi: GpuSurfaceScalarField::zeros(gpu, m, "phi")?,
        })
    }
    fn state(&self) -> FlowState<'_> {
        FlowState::new(&self.u, &self.phi, 1e-3)
    }
}

// ==========================================================================
//  The RAS two-equation family
// ==========================================================================

// ==========================================================================
//  81.9  The gates
// ==========================================================================

/// `SPEC-LIT` 39: Wilcox k-omega.
#[test]
fn the_k_omega_correction_replays_bitwise() {
    let Some(gpu) = gpu() else { return };
    let hm = box4();
    let mesh = GpuMesh::upload(&gpu, &hm).expect("mesh");
    let ctrl = fixed(1e-3);
    let wf = WallFaces::none(hm.n_boundary_faces);
    let rough = NutRoughness::none(hm.n_boundary_faces);
    let quiet = Quiet::new(&gpu, &mesh).expect("flow");
    let flow = quiet.state();

    let report = capture_replays_bitwise(
        &gpu,
        "k-omega (SPEC-LIT 39)",
        || {
            let mut m = crate::models::k_omega::KOmega::new(
                &gpu,
                &hm,
                &mesh,
                Default::default(),
                ctrl,
                WallFunctionCoeffs::default(),
                &wf,
                &rough,
            )?;
            gpu.write(&mut m.k_mut().f, &vec![0.05 as Scalar; hm.n_cells])?;
            gpu.write(&mut m.omega_mut().f, &vec![50.0 as Scalar; hm.n_cells])?;
            m.initialise(&gpu, &flow)?;
            Ok(m)
        },
        |m| m.correct(&gpu, &flow).map(|_| ()),
        |m| {
            Ok(vec![
                field(&gpu, "k", m.k())?,
                field(&gpu, "omega", m.omega())?,
                field(&gpu, "nut", m.nut())?,
            ])
        },
    )
    .expect("SPEC-LIT 81.7: k-omega must capture and replay bitwise");
    println!("  k-omega: {report}");
}

/// `SPEC-LIT` 41: Launder-Sharma low-Reynolds k-epsilon.
#[test]
fn the_launder_sharma_correction_replays_bitwise() {
    let Some(gpu) = gpu() else { return };
    let hm = box4();
    let mesh = GpuMesh::upload(&gpu, &hm).expect("mesh");
    let ctrl = fixed(1e-3);
    let wf = WallFaces::none(hm.n_boundary_faces);
    let rough = NutRoughness::none(hm.n_boundary_faces);
    let quiet = Quiet::new(&gpu, &mesh).expect("flow");
    let flow = quiet.state();

    let report = capture_replays_bitwise(
        &gpu,
        "Launder-Sharma (SPEC-LIT 41)",
        || {
            let mut m = crate::models::launder_sharma::LaunderSharmaKE::new(
                &gpu,
                &hm,
                &mesh,
                Default::default(),
                ctrl,
                WallFunctionCoeffs::default(),
                &wf,
                &rough,
            )?;
            gpu.write(&mut m.k_mut().f, &vec![0.05 as Scalar; hm.n_cells])?;
            gpu.write(&mut m.epsilon_mut().f, &vec![0.2 as Scalar; hm.n_cells])?;
            m.initialise(&gpu, &flow)?;
            Ok(m)
        },
        |m| m.correct(&gpu, &flow).map(|_| ()),
        |m| {
            Ok(vec![
                field(&gpu, "k", m.k())?,
                field(&gpu, "epsilon", m.epsilon())?,
                field(&gpu, "nut", m.nut())?,
            ])
        },
    )
    .expect("SPEC-LIT 81.7: Launder-Sharma must capture and replay bitwise");
    println!("  Launder-Sharma: {report}");
}

/// `SPEC-LIT` 44: both k-epsilon variants, RNG and realizable. Two models in
/// one gate because the registry names one file.
#[test]
fn both_ke_variants_replay_bitwise() {
    let Some(gpu) = gpu() else { return };
    let hm = box4();
    let mesh = GpuMesh::upload(&gpu, &hm).expect("mesh");
    let ctrl = fixed(1e-3);
    let wf = WallFaces::none(hm.n_boundary_faces);
    let rough = NutRoughness::none(hm.n_boundary_faces);
    let quiet = Quiet::new(&gpu, &mesh).expect("flow");
    let flow = quiet.state();

    let r = capture_replays_bitwise(
        &gpu,
        "RNG k-epsilon (SPEC-LIT 44)",
        || {
            let mut m = crate::models::ke_variants::RngKe::new(
                &gpu,
                &hm,
                &mesh,
                Default::default(),
                ctrl,
                WallFunctionCoeffs::default(),
                &wf,
                &rough,
            )?;
            gpu.write(&mut m.k_mut().f, &vec![0.05 as Scalar; hm.n_cells])?;
            gpu.write(&mut m.epsilon_mut().f, &vec![0.2 as Scalar; hm.n_cells])?;
            m.initialise(&gpu, &flow)?;
            Ok(m)
        },
        |m| m.correct(&gpu, &flow).map(|_| ()),
        |m| {
            Ok(vec![
                field(&gpu, "k", m.k())?,
                field(&gpu, "epsilon", m.epsilon())?,
                field(&gpu, "nut", m.nut())?,
            ])
        },
    )
    .expect("SPEC-LIT 81.7: RNG k-epsilon must capture and replay bitwise");
    println!("  RNG k-epsilon: {r}");

    let r = capture_replays_bitwise(
        &gpu,
        "realizable k-epsilon (SPEC-LIT 44)",
        || {
            let mut m = crate::models::ke_variants::RealizableKe::new(
                &gpu,
                &hm,
                &mesh,
                Default::default(),
                ctrl,
                WallFunctionCoeffs::default(),
                &wf,
                &rough,
            )?;
            gpu.write(&mut m.k_mut().f, &vec![0.05 as Scalar; hm.n_cells])?;
            gpu.write(&mut m.epsilon_mut().f, &vec![0.2 as Scalar; hm.n_cells])?;
            m.initialise(&gpu, &flow)?;
            Ok(m)
        },
        |m| m.correct(&gpu, &flow).map(|_| ()),
        |m| {
            Ok(vec![
                field(&gpu, "k", m.k())?,
                field(&gpu, "epsilon", m.epsilon())?,
                field(&gpu, "nut", m.nut())?,
            ])
        },
    )
    .expect("SPEC-LIT 81.7: realizable k-epsilon must capture and replay bitwise");
    println!("  realizable k-epsilon: {r}");
}

/// `SPEC-LIT` 40: Menter k-omega SST.
#[test]
fn the_sst_correction_replays_bitwise() {
    let Some(gpu) = gpu() else { return };
    let hm = box4();
    let mesh = GpuMesh::upload(&gpu, &hm).expect("mesh");
    let ctrl = fixed(1e-3);
    let wf = WallFaces::none(hm.n_boundary_faces);
    let quiet = Quiet::new(&gpu, &mesh).expect("flow");
    let flow = quiet.state();
    let (y, _gy) = wall_distance(&gpu, hm.n_cells).expect("y");

    let report = capture_replays_bitwise(
        &gpu,
        "k-omega SST (SPEC-LIT 40)",
        || {
            let mut m = crate::models::k_omega_sst::KOmegaSst::new(
                &gpu,
                &hm,
                &mesh,
                Default::default(),
                ctrl,
                WallFunctionCoeffs::default(),
                &wf,
                &y,
            )?;
            gpu.write(&mut m.k_mut().f, &vec![0.05 as Scalar; hm.n_cells])?;
            gpu.write(&mut m.omega_mut().f, &vec![50.0 as Scalar; hm.n_cells])?;
            m.initialise(&gpu, &flow)?;
            Ok(m)
        },
        |m| m.correct(&gpu, &flow).map(|_| ()),
        |m| {
            Ok(vec![
                field(&gpu, "k", m.k())?,
                field(&gpu, "omega", m.omega())?,
                field(&gpu, "nut", m.nut())?,
            ])
        },
    )
    .expect("SPEC-LIT 81.7: k-omega SST must capture and replay bitwise");
    println!("  k-omega SST: {report}");
}

/// `SPEC-LIT` 56: Spalart-Allmaras.
#[test]
fn the_spalart_allmaras_correction_replays_bitwise() {
    let Some(gpu) = gpu() else { return };
    let hm = box4();
    let mesh = GpuMesh::upload(&gpu, &hm).expect("mesh");
    let ctrl = fixed(1e-3);
    let wf = WallFaces::none(hm.n_boundary_faces);
    let rough = NutRoughness::none(hm.n_boundary_faces);
    let quiet = Quiet::new(&gpu, &mesh).expect("flow");
    let flow = quiet.state();
    let (y, _gy) = wall_distance(&gpu, hm.n_cells).expect("y");

    let report = capture_replays_bitwise(
        &gpu,
        "Spalart-Allmaras (SPEC-LIT 56)",
        || {
            let mut m = crate::models::spalart_allmaras::SpalartAllmaras::new(
                &gpu,
                &hm,
                &mesh,
                Default::default(),
                ctrl,
                WallFunctionCoeffs::default(),
                &wf,
                &rough,
                &y,
            )?;
            gpu.write(&mut m.nu_tilda_mut().f, &vec![1e-3 as Scalar; hm.n_cells])?;
            m.initialise(&gpu, &flow)?;
            Ok(m)
        },
        |m| m.correct(&gpu, &flow).map(|_| ()),
        |m| {
            Ok(vec![
                field(&gpu, "nuTilda", m.nu_tilda())?,
                field(&gpu, "nut", m.nut())?,
            ])
        },
    )
    .expect("SPEC-LIT 81.7: Spalart-Allmaras must capture and replay bitwise");
    println!("  Spalart-Allmaras: {report}");
}

/// `SPEC-LIT` 88: the Langtry-Menter transition model, on SST.
///
/// The one thing here that a capture could have broken is §88.4's fixed
/// point: a loop whose trip count depended on a floating-point convergence
/// test would be a data-dependent loop inside the recorded region, and a
/// graph cannot hold one. The trip count is a launch parameter instead, so
/// the sequence is identical every replay - and this is the measurement that
/// says so rather than the argument.
#[test]
fn the_transition_correction_replays_bitwise() {
    let Some(gpu) = gpu() else { return };
    let hm = box4();
    let mesh = GpuMesh::upload(&gpu, &hm).expect("mesh");
    let ctrl = fixed(1e-3);
    let wf = WallFaces::none(hm.n_boundary_faces);
    let quiet = Quiet::new(&gpu, &mesh).expect("flow");
    let flow = quiet.state();
    let (y, _gy) = wall_distance(&gpu, hm.n_cells).expect("y");

    let report = capture_replays_bitwise(
        &gpu,
        "kOmegaSSTLM (SPEC-LIT 88)",
        || {
            let mut m = crate::models::k_omega_sst::KOmegaSst::new(
                &gpu,
                &hm,
                &mesh,
                Default::default(),
                ctrl,
                WallFunctionCoeffs::default(),
                &wf,
                &y,
            )?;
            gpu.write(&mut m.k_mut().f, &vec![0.05 as Scalar; hm.n_cells])?;
            gpu.write(&mut m.omega_mut().f, &vec![50.0 as Scalar; hm.n_cells])?;
            // The two new equations get the SAME fixed-iteration solver the
            // other two have. Not a detail: a checking solve calls
            // `read_flag`, which synchronises on an event, and §81.3's guard
            // catches exactly that - the capture fails with
            // CUDA_ERROR_CAPTURED_EVENT rather than silently recording a
            // stale flag. It is how this gate found that `LmControls`
            // defaults to a checking solver, which is right for a run and
            // wrong for a capture.
            let lmc = crate::models::transition::LmControls {
                gamma_solver: ctrl.k_solver,
                gamma_relax: 1.0,
                gamma_conv: ctrl.k_conv(),
                re_thetat_solver: ctrl.k_solver,
                re_thetat_relax: 1.0,
                re_thetat_conv: ctrl.k_conv(),
            };
            let mut lm = crate::models::transition::LangtryMenter::new(
                &gpu,
                &mesh,
                Default::default(),
                lmc,
                &y,
            )?;
            gpu.write(&mut lm.gamma_mut().f, &vec![0.3 as Scalar; hm.n_cells])?;
            gpu.write(&mut lm.re_thetat_mut().f, &vec![250.0 as Scalar; hm.n_cells])?;
            lm.initialise(&gpu, &mesh)?;
            m.set_transition(Some(lm))?;
            m.initialise(&gpu, &flow)?;
            Ok(m)
        },
        |m| m.correct(&gpu, &flow).map(|_| ()),
        |m| {
            let lm = m.transition().expect("attached");
            Ok(vec![
                field(&gpu, "k", m.k())?,
                field(&gpu, "omega", m.omega())?,
                field(&gpu, "nut", m.nut())?,
                field(&gpu, "gamma", lm.gamma())?,
                field(&gpu, "ReThetat", lm.re_thetat())?,
            ])
        },
    )
    .expect("SPEC-LIT 81.7: kOmegaSSTLM must capture and replay bitwise");
    println!("  kOmegaSSTLM: {report}");
}

/// `SPEC-LIT` 43: the LES subgrid models.
#[test]
fn the_les_correction_replays_bitwise() {
    let Some(gpu) = gpu() else { return };
    let hm = box4();
    let mesh = GpuMesh::upload(&gpu, &hm).expect("mesh");
    let ctrl = fixed(1e-3);
    let wf = WallFaces::none(hm.n_boundary_faces);
    let quiet = Quiet::new(&gpu, &mesh).expect("flow");
    let flow = quiet.state();
    let (y, gy) = wall_distance(&gpu, hm.n_cells).expect("y");

    let report = capture_replays_bitwise(
        &gpu,
        "LES subgrid (SPEC-LIT 43)",
        || {
            crate::models::les::Les::new(
                &gpu,
                &hm,
                &mesh,
                crate::models::les::LesModel::Smagorinsky,
                Default::default(),
                Default::default(),
                ctrl,
                &wf,
                &y,
                &gy,
            )
        },
        |m| m.correct(&gpu, &flow),
        |m| Ok(vec![field(&gpu, "nut", m.nut())?]),
    )
    .expect("SPEC-LIT 81.7: LES must capture and replay bitwise");
    println!("  LES: {report}");
}

/// `SPEC-LIT` 57: the DES length scale, on the SA background.
#[test]
fn the_des_correction_replays_bitwise() {
    let Some(gpu) = gpu() else { return };
    let hm = box4();
    let mesh = GpuMesh::upload(&gpu, &hm).expect("mesh");
    let n = hm.n_cells;
    let (y, gy) = wall_distance(&gpu, n).expect("y");
    let nut = gpu.upload(&vec![1e-4 as Scalar; n]).expect("nut");
    let grad_frob = gpu.upload(&vec![3.0 as Scalar; n]).expect("S");

    let report = capture_replays_bitwise(
        &gpu,
        "DES length scale, SA background (SPEC-LIT 57)",
        || {
            crate::models::des::DesLengthScale::new(
                &gpu,
                &mesh,
                &y,
                &gy,
                crate::models::des::DesBranch::Des97,
                crate::models::des::HybridDelta::MaxEdge,
                crate::models::des::HybridBackground::Sa,
                crate::models::des::DesCoeffs::sa(),
            )
        },
        |m| m.update_sa(&gpu, &nut, &grad_frob, &y, 1e-5, n),
        |m| {
            Ok(vec![
                buf(&gpu, "length", m.length())?,
                buf(&gpu, "shielding", m.shielding())?,
                buf(&gpu, "filterWidth", m.filter_width())?,
            ])
        },
    )
    .expect("SPEC-LIT 81.7: DES must capture and replay bitwise");
    println!("  DES: {report}");
}

/// `SPEC-LIT` 54: the psychrometric update.
#[test]
fn the_psychrometric_update_replays_bitwise() {
    let Some(gpu) = gpu() else { return };
    let hm = box4();
    let mesh = GpuMesh::upload(&gpu, &hm).expect("mesh");
    let n = hm.n_cells;

    let mut t = GpuScalarField::zeros(&gpu, &mesh, "T").expect("T");
    gpu.write(&mut t.f, &vec![300.0 as Scalar; n]).expect("T");
    let mut yv = GpuScalarField::zeros(&gpu, &mesh, "Yv").expect("Yv");
    gpu.write(&mut yv.f, &vec![0.008 as Scalar; n]).expect("Yv");

    let report = capture_replays_bitwise(
        &gpu,
        "psychrometrics (SPEC-LIT 54)",
        || crate::psychro::Psychrometrics::new(&gpu, &mesh, 101325.0),
        |p| p.update(&gpu, &t, &yv),
        |p| {
            Ok(vec![
                buf(&gpu, "w", &p.w)?,
                buf(&gpu, "rh", &p.rh)?,
                buf(&gpu, "h", &p.h)?,
                buf(&gpu, "v", &p.v)?,
                field(&gpu, "Tv", p.virtual_temperature_field())?,
            ])
        },
    )
    .expect("SPEC-LIT 81.7: psychrometrics must capture and replay bitwise");
    println!("  psychrometrics: {report}");
}

/// `SPEC-LIT` 24: momentum, one predictor.
#[test]
fn the_momentum_predictor_replays_bitwise() {
    let Some(gpu) = gpu() else { return };
    let hm = box4();
    let mesh = GpuMesh::upload(&gpu, &hm).expect("mesh");

    let mut ctrl = crate::momentum::MomentumControls {
        nu: 1e-3,
        u_relax: 1.0,
        ..Default::default()
    };
    ctrl.u_solver.fixed_iters = true;
    ctrl.u_solver.max_iter = 4;
    ctrl.u_solver.report_residuals = false;

    let phi = GpuSurfaceScalarField::zeros(&gpu, &mesh, "phi").expect("phi");
    let nut = GpuScalarField::zeros(&gpu, &mesh, "nut").expect("nut");

    let report = capture_replays_bitwise(
        &gpu,
        "momentum predictor (SPEC-LIT 24)",
        || {
            let m = crate::momentum::Momentum::new(&gpu, &mesh, ctrl, Default::default())?;
            let mut u = GpuVectorField::zeros(&gpu, &mesh, "U")?;
            let seed: Vec<Vec3> = (0..hm.n_cells)
                .map(|i| Vec3::new(0.1 + 0.01 * (i % 5) as Scalar, 0.0, 0.0))
                .collect();
            gpu.write(&mut u.f, &seed)?;
            Ok((m, u))
        },
        |(m, u): &mut (_, GpuVectorField)| m.solve(&gpu, u, &phi, &nut).map(|_| ()),
        |(_, u): &(_, GpuVectorField)| {
            let v = gpu.download(&u.f)?;
            let mut flat = Vec::with_capacity(v.len() * 3);
            for c in &v {
                flat.push(c.x);
                flat.push(c.y);
                flat.push(c.z);
            }
            Ok(vec![("U", flat)])
        },
    )
    .expect("SPEC-LIT 81.7: the momentum predictor must capture and replay bitwise");
    println!("  momentum: {report}");
}

/// `SPEC-LIT` 28: the P-1 radiation correction.
#[test]
fn the_p1_radiation_correction_replays_bitwise() {
    let Some(gpu) = gpu() else { return };
    let hm = box4();
    let mesh = GpuMesh::upload(&gpu, &hm).expect("mesh");
    let n = hm.n_cells;

    let mut t = GpuScalarField::zeros(&gpu, &mesh, "T").expect("T");
    gpu.write(&mut t.f, &vec![1200.0 as Scalar; n]).expect("T");
    gpu.write(&mut t.bf, &vec![600.0 as Scalar; hm.n_boundary_faces]).expect("Tb");

    let props = crate::radiation::RadiationProps::new(0.5).expect("props");
    let sc = crate::solver::SolverControls {
        tolerance: 1e-14,
        rel_tol: 0.0,
        max_iter: 4,
        fixed_iters: true,
        report_residuals: false,
        ..Default::default()
    };

    let report = capture_replays_bitwise(
        &gpu,
        "P-1 radiation (SPEC-LIT 28)",
        || crate::radiation::Radiation::new(&gpu, &mesh, props),
        |r: &mut crate::radiation::Radiation| r.correct(&gpu, &t, None, &sc, 0).map(|_| ()),
        |r: &crate::radiation::Radiation| {
            Ok(vec![
                field(&gpu, "G", r.field())?,
                buf(&gpu, "su", r.su())?,
                buf(&gpu, "sp", r.sp())?,
            ])
        },
    )
    .expect("SPEC-LIT 81.7: P-1 radiation must capture and replay bitwise");
    println!("  P-1 radiation: {report}");
}

/// `SPEC-LIT` 62: the WSGG band coefficients, recomputed each iteration from
/// the local composition.
#[test]
fn the_wsgg_update_replays_bitwise() {
    let Some(gpu) = gpu() else { return };
    let hm = box4();
    let mesh = GpuMesh::upload(&gpu, &hm).expect("mesh");
    let n = hm.n_cells;

    let mut t = GpuScalarField::zeros(&gpu, &mesh, "T").expect("T");
    gpu.write(&mut t.f, &vec![1400.0 as Scalar; n]).expect("T");
    gpu.write(&mut t.bf, &vec![900.0 as Scalar; hm.n_boundary_faces]).expect("Tb");
    let y_p = gpu.upload(&vec![0.1 as Scalar; n]).expect("yP");
    let medium = crate::wsgg::MediumState {
        y_products: Some(&y_p),
        ..Default::default()
    };

    let props = crate::wsgg::SpectralProps {
        model: crate::wsgg::SpectralModel::Wsgg,
        ..Default::default()
    };

    let report = capture_replays_bitwise(
        &gpu,
        "WSGG band properties (SPEC-LIT 62)",
        || {
            let mut b = crate::wsgg::Bands::new(&gpu, &mesh, props, 0.5, true)?;
            // The one host round-trip in the WSGG update is the diagnostic
            // count of floored cells; the floor itself is a kernel. Off, the
            // update is pure device work - see Bands::set_count_floored.
            b.set_count_floored(false);
            Ok(b)
        },
        |b: &mut crate::wsgg::Bands| b.update(&gpu, &mesh, &t, &medium),
        |b: &crate::wsgg::Bands| {
            // Every solved band, not just the first: a capture that dropped
            // one would otherwise pass.
            let mut out = Vec::new();
            for j in b.solved() {
                out.push(("kappa_j", gpu.download(b.kappa(j))?));
                out.push(("weight_j", gpu.download(b.weight(j))?));
            }
            Ok(out)
        },
    )
    .expect("SPEC-LIT 81.7: the WSGG update must capture and replay bitwise");
    println!("  WSGG: {report}");
}

// ==========================================================================
//  81.10  The refusals, measured
// ==========================================================================

/// **Two modules cannot be captured, and this runs the capture to prove it.**
///
/// A refusal written only in prose decays: the module is fixed, or made
/// worse, and the sentence beside it stays the same. So the two refusals in
/// [`crate::capture::registry::REGISTRY`] are executed. Each must fail, and
/// each must fail *naming the call that makes it impossible* - if either
/// starts to succeed, this test fails and the registry row is out of date in
/// the good direction.
///
/// * `src/vof.rs` - `Vof::step` computes an alpha Courant number on the
///   device, **downloads it**, and divides the time step by it to get a
///   sub-cycle count it then loops over on the host. That is the
///   data-dependent trip count a graph cannot hold: the graph would record
///   whatever count the capture happened to see and replay that count for
///   ever. `SPEC-LIT` 13.4's alternative is a **prescribed**
///   `nAlphaSubCycles`, given by the case rather than derived from the flux -
///   which is what removes the read-back. It is not implemented;
/// * `src/fvdom.rs` - `FvDom::correct` sweeps the ordinates and carries each
///   ordinate's boundary intensity to the next **through the host**
///   (`bf_cache`), and downloads the wall temperature once per correction.
///   The alternative is a device-resident inflow coupling, which is a
///   rewrite of the sweep and not a flag.
#[test]
fn the_two_refusals_are_measured_and_not_asserted() {
    let Some(gpu) = gpu() else { return };
    let hm = box4();
    let mesh = GpuMesh::upload(&gpu, &hm).expect("mesh");
    let n = hm.n_cells;

    // ---- VOF ------------------------------------------------------------
    let props = crate::vof::VofProperties {
        rho1: 1000.0,
        rho2: 1.0,
        mu1: 1e-3,
        mu2: 1.8e-5,
        sigma: 0.0,
        g: Vec3::new(0.0, 0.0, -9.81),
        c_alpha: 1.0,
    };
    let mut vctrl = crate::vof::VofControls {
        delta_t: 1e-4,
        adjust_time_step: false,
        max_sub_cycles: 1,
        report_continuity: false,
        ..Default::default()
    };
    vctrl.u_solver.fixed_iters = true;
    vctrl.u_solver.max_iter = 2;
    vctrl.u_solver.report_residuals = false;
    vctrl.p_solver = vctrl.u_solver;

    let mut v = crate::vof::Vof::new(&gpu, &hm, &mesh, props, vctrl).expect("vof");
    let a: Vec<Scalar> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
    gpu.write(&mut v.alpha_mut().f, &a).expect("alpha");
    v.initialise(&gpu).expect("init");
    v.step(&gpu, 1e-4).expect("one eager step");
    gpu.sync().expect("sync");

    let err = match gpu.capture(|_| v.step(&gpu, 1e-4).map(|_| ())) {
        Ok(_) => panic!(
            "Vof::step CAPTURED. If the sub-cycle count no longer comes back \
             to the host, VOF is capturable and src/vof.rs should be promoted \
             from Refused to Gate in the capture registry - SPEC-LIT 81.10"
        ),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("Gpu::download"),
        "VOF must be refused by the download guard and not by something else \
         - got: {err}"
    );

    // ---- fvDOM ----------------------------------------------------------
    let mut t = GpuScalarField::zeros(&gpu, &mesh, "T").expect("T");
    gpu.write(&mut t.f, &vec![1200.0 as Scalar; n]).expect("T");
    gpu.write(&mut t.bf, &vec![600.0 as Scalar; hm.n_boundary_faces]).expect("Tb");
    let dprops = crate::fvdom::FvDomProps::new(0.5, 0.0).expect("props");
    let sc = crate::solver::SolverControls {
        tolerance: 1e-14,
        rel_tol: 0.0,
        max_iter: 2,
        fixed_iters: true,
        report_residuals: false,
        ..Default::default()
    };
    let mut d = crate::fvdom::FvDom::new(&gpu, &mesh, dprops).expect("fvdom");
    d.correct(&gpu, &t, None, &sc, 1).expect("one eager sweep");
    gpu.sync().expect("sync");

    let err = match gpu.capture(|_| d.correct(&gpu, &t, None, &sc, 1).map(|_| ())) {
        Ok(_) => panic!(
            "FvDom::correct CAPTURED. If the ordinate coupling no longer goes \
             through the host, src/fvdom.rs should be promoted from Refused \
             to Gate in the capture registry - SPEC-LIT 81.10"
        ),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("Gpu::download"),
        "fvDOM must be refused by the download guard and not by something \
         else - got: {err}"
    );
}

/// **A `PCG` or `DIC` solve cannot be captured, and this says which call.**
///
/// `solver::solve` verifies that the matrix really is symmetric before it
/// runs conjugate gradients or an incomplete Cholesky on it - `SPEC-LIT` 8.2,
/// and the right thing to do, because CG on an asymmetric matrix does not
/// converge slowly, it converges to the wrong answer. The check ends in a
/// host read-back.
///
/// `solve`'s own doc comment has said so since it was written. Nothing held
/// it: `solver.rs`'s capture gate calls `solve_pbicgstab` directly and
/// `models/k_epsilon`'s runs the default, so no test in this crate had ever
/// captured a `PCG` solve. This one does, and requires it to fail naming the
/// call - so if the check ever moves onto the device, this test says so.
///
/// The consequence for a case is concrete and worth stating plainly: choosing
/// `PCG` in `fvSolution` costs that equation its CUDA graph, whatever
/// `fixed_iters` says.
#[test]
fn a_pcg_solve_is_not_capturable_and_says_which_call() {
    use crate::solver::{LinearSolverKind, Preconditioner, SolverControls};

    let Some(gpu) = gpu() else { return };
    let hm = box4();
    let mesh = GpuMesh::upload(&gpu, &hm).expect("mesh");
    let n = hm.n_cells;

    let k = crate::solver::SolverKernels::new(&gpu).expect("kernels");
    let mut w = crate::solver::SolverWorkspace::for_mesh(&gpu, &mesh).expect("workspace");
    let mut a = crate::ldu::GpuLduMatrix::new(&gpu, &mesh).expect("matrix");
    // A symmetric, diagonally dominant system, so the refusal below cannot be
    // the asymmetry refusal wearing the wrong name.
    gpu.write(&mut a.diag, &vec![4.0 as Scalar; n]).expect("diag");
    gpu.write(&mut a.source, &vec![1.0 as Scalar; n]).expect("b");
    let mut psi: crate::DevBuf<Scalar> = gpu.zeros(n).expect("psi");

    let base = SolverControls {
        tolerance: 1e-14,
        rel_tol: 0.0,
        max_iter: 4,
        fixed_iters: true,
        report_residuals: false,
        ..Default::default()
    };

    for (what, ctrl) in [
        (
            "PCG",
            SolverControls { solver: LinearSolverKind::PCG, precon: Preconditioner::Diagonal, ..base },
        ),
        (
            "DIC",
            SolverControls {
                solver: LinearSolverKind::PBiCGStab,
                precon: Preconditioner::Dic,
                ..base
            },
        ),
    ] {
        // Warm, outside the capture, so this is not the first launch.
        crate::solver::solve(&gpu, &k, &mut psi, &a, &mesh, &mut w, &ctrl).expect("warm");
        gpu.sync().expect("sync");

        let err = match gpu.capture(|_| {
            crate::solver::solve(&gpu, &k, &mut psi, &a, &mesh, &mut w, &ctrl).map(|_| ())
        }) {
            Ok(_) => panic!(
                "a {what} solve CAPTURED. If the symmetry check no longer \
                 reads back to the host, say so in SPEC-LIT 81.11 and in \
                 solve()'s own doc comment, both of which currently claim it \
                 does"
            ),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("Gpu::download"),
            "{what} must be refused by the download guard, naming the call - \
             got: {err}"
        );
    }
}
