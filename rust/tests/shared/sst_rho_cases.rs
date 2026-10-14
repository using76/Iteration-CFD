// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

// The DST-05 gates of SPEC-LIT §156.6 - the rho-weighted SST-2003 and
// SST-sust - written once and included both by src/models/k_omega_sst.rs (as
// k_omega_sst::tests) and by tests/sst_rho.rs, so both builds run them.
// Written from SPEC-LIT §156 (the NASA Turbulence Modeling Resource SST page:
// the conservation form, SST-2003, SST-sust). Tolerances are the build's:
// class R is 1e-12 in f64 and 1e-5 under `single` (D-F32-1); class D is bit
// for bit in both. A GPU test returns early, by name, on a box without a
// device. Helper names carry a `rho_` prefix and the two field types are
// named by path, because the lib copy shares a module with the older tests.
// The specification was informed by a review of FSCDustFoam (Huang &
// Lipatnikov, GPL) and no code was copied.
// No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::device::{DevBuf, Gpu};
use ofgpu::field_ops::{correct_boundary_conditions_vector, FieldKernels};
use ofgpu::field_setup::{NutRoughness, WallFaces};
use ofgpu::fv::DivScheme;
use ofgpu::models::k_omega_sst::{
    KOmegaSst, KOmegaSstCoeffs, RhoSstOptions, SstDensity, SstSust,
};
use ofgpu::timescheme::DdtScheme;
use ofgpu::turbulence::{FlowState, TurbulenceControls};
use ofgpu::wallfunctions::WallFunctionCoeffs;
use ofgpu::{BcKind, GpuMesh, HostMesh, Label, Result, Scalar, Vec3};

/// Class R (§156.6): 1e-12 in f64, 1e-5 under `single`.
const RHO_R: f64 = if cfg!(feature = "single") { 1e-5 } else { 1e-12 };
/// The `∫rho k dV` budget: summed on the host in f64, but k is stored in the
/// build's precision, so it is class R (as §124.2's imbalance row, §125).
const RHO_A: f64 = RHO_R;

fn rho_gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

#[allow(clippy::unnecessary_cast, clippy::useless_conversion)]
fn rho_w(x: Scalar) -> f64 {
    x as f64
}

/// A uniform block, `nz = 1` with `empty` front and back, x patches and y
/// patches of the given types.
fn rho_mesh(gpu: &Gpu, n: [usize; 2], x_type: &str, y_type: &str) -> Result<(HostMesh, GpuMesh)> {
    let axis = |hi: Scalar, n: usize| GradedAxis {
        lo: 0.0,
        hi,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    let spec = BlockSpec {
        x: axis(1.0, n[0]),
        y: axis(1.0, n[1]),
        z: axis(0.1, 1),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: [x_type, x_type, y_type, y_type, "empty", "empty"].map(String::from),
        cyclic: Vec::new(),
    };
    let hm = blockgen::build_mesh(&spec)?;
    let gm = GpuMesh::upload(gpu, &hm)?;
    Ok((hm, gm))
}

/// Transient, unrelaxed, solved to the floor: `decay_controls` of the older
/// tests with the time scheme and the convection named.
fn rho_controls(dt: Scalar, ddt: DdtScheme, scheme: DivScheme, bounded: bool) -> TurbulenceControls {
    let mut ctrl = TurbulenceControls {
        steady: false,
        delta_t: dt,
        k_relax: 1.0,
        eps_relax: 1.0,
        nut_max_coeff: 1e8,
        ..Default::default()
    };
    ctrl.ddt = ddt;
    ctrl.div_scheme = scheme;
    ctrl.eps_div_scheme = scheme;
    ctrl.bounded_convection = bounded;
    ctrl.eps_bounded_convection = bounded;
    ctrl.k_solver.tolerance = if cfg!(feature = "single") { 1e-12 } else { 1e-14 };
    ctrl.k_solver.rel_tol = 0.0;
    ctrl.k_solver.report_residuals = false;
    ctrl.epsilon_solver = ctrl.k_solver;
    ctrl
}

/// Everything one run needs on the device: the flow and the density.
struct RhoFlow {
    u: ofgpu::field::GpuVectorField,
    phi_v: ofgpu::field::GpuSurfaceScalarField,
    phi_m: ofgpu::field::GpuSurfaceScalarField,
    rho: DevBuf<Scalar>,
    rho0: DevBuf<Scalar>,
    rho00: DevBuf<Scalar>,
    rho_face: ofgpu::field::GpuSurfaceScalarField,
}

impl RhoFlow {
    fn density(&self) -> SstDensity<'_> {
        SstDensity {
            rho: &self.rho,
            rho0: &self.rho0,
            rho00: &self.rho00,
            rho_face: &self.rho_face,
            phi_volume: &self.phi_v,
        }
    }
}

/// `u(x)` in the cells and as fixedValue on every non-empty face, the volume
/// flux `u(c_f).S_f` (zero on the boundary when `closed`), the cell densities
/// `rho(x)` (new level) and `rho_old(x)` (both old levels), the face density
/// by the mesh's interpolation weights, and the mass flux `rho_f phi`.
fn rho_flow(
    gpu: &Gpu,
    hm: &HostMesh,
    gm: &GpuMesh,
    u_at: &dyn Fn(Vec3) -> Vec3,
    rho_at: &dyn Fn(Vec3) -> Scalar,
    rho_old_at: &dyn Fn(Vec3) -> Scalar,
    closed: bool,
) -> Result<RhoFlow> {
    let dot = |a: Vec3, b: Vec3| a.x * b.x + a.y * b.y + a.z * b.z;
    let (nc, nif, nbf) = (hm.n_cells, hm.n_internal_faces, hm.n_boundary_faces);

    let mut u = ofgpu::field::GpuVectorField::zeros(gpu, gm, "U")?;
    let uc: Vec<Vec3> = hm.c.iter().map(|&c| u_at(c)).collect();
    gpu.write(&mut u.f, &uc)?;
    gpu.write(&mut u.f0, &uc)?;
    gpu.write(&mut u.f00, &uc)?;
    let mut kind: Vec<Label> = gpu.download(&u.bc_kind)?;
    let mut fr = vec![0.0 as Scalar; nbf];
    let mut rv = vec![Vec3::ZERO; nbf];
    for i in 0..nbf {
        if kind[i] != BcKind::Empty as Label {
            kind[i] = BcKind::FixedValue as Label;
            fr[i] = 1.0;
            rv[i] = u_at(hm.b_cf[i]);
        }
    }
    gpu.write(&mut u.bc_kind, &kind)?;
    gpu.write(&mut u.fr, &fr)?;
    gpu.write(&mut u.ref_value, &rv)?;
    let fld = FieldKernels::new(gpu)?;
    correct_boundary_conditions_vector(gpu, &fld, &mut u, gm)?;

    let rho_c: Vec<Scalar> = hm.c.iter().map(|&c| rho_at(c)).collect();
    let rho_o: Vec<Scalar> = hm.c.iter().map(|&c| rho_old_at(c)).collect();

    let phi_i: Vec<Scalar> = (0..nif).map(|f| dot(u_at(hm.cf[f]), hm.sf[f])).collect();
    let phi_b: Vec<Scalar> = (0..nbf)
        .map(|i| {
            if closed || kind[i] == BcKind::Empty as Label {
                0.0
            } else {
                dot(u_at(hm.b_cf[i]), hm.b_sf[i])
            }
        })
        .collect();
    let rf_i: Vec<Scalar> = (0..nif)
        .map(|f| {
            let (p, q) = (hm.owner[f] as usize, hm.neighbour[f] as usize);
            let w = hm.weights[f];
            // A uniform density stays exact on the faces (the rho == 1 row).
            if rho_c[p] == rho_c[q] {
                rho_c[p]
            } else {
                w * rho_c[p] + (1.0 - w) * rho_c[q]
            }
        })
        .collect();
    let rf_b: Vec<Scalar> = (0..nbf).map(|i| rho_c[hm.b_face_cells[i] as usize]).collect();

    let mut phi_v = ofgpu::field::GpuSurfaceScalarField::zeros(gpu, gm, "phi")?;
    gpu.write(&mut phi_v.f, &phi_i)?;
    gpu.write(&mut phi_v.bf, &phi_b)?;
    let mut phi_m = ofgpu::field::GpuSurfaceScalarField::zeros(gpu, gm, "phi_m")?;
    let pm_i: Vec<Scalar> = (0..nif).map(|f| rf_i[f] * phi_i[f]).collect();
    let pm_b: Vec<Scalar> = (0..nbf).map(|i| rf_b[i] * phi_b[i]).collect();
    gpu.write(&mut phi_m.f, &pm_i)?;
    gpu.write(&mut phi_m.bf, &pm_b)?;
    let mut rho_face = ofgpu::field::GpuSurfaceScalarField::zeros(gpu, gm, "rho_f")?;
    gpu.write(&mut rho_face.f, &rf_i)?;
    gpu.write(&mut rho_face.bf, &rf_b)?;

    let _ = nc;
    Ok(RhoFlow {
        u,
        phi_v,
        phi_m,
        rho: gpu.upload(&rho_c)?,
        rho0: gpu.upload(&rho_o)?,
        rho00: gpu.upload(&rho_o)?,
        rho_face,
    })
}

/// A model with the given initial `k(x)`, `omega(x)`, wall faces and wall
/// distance, initialised against `flow`.
#[allow(clippy::too_many_arguments)]
fn rho_model<'m>(
    gpu: &Gpu,
    hm: &HostMesh,
    gm: &'m GpuMesh,
    ctrl: TurbulenceControls,
    walls: &WallFaces,
    y: &[Scalar],
    f1: Option<Scalar>,
    k_at: &dyn Fn(Vec3) -> Scalar,
    w_at: &dyn Fn(Vec3) -> Scalar,
    flow: &FlowState,
) -> Result<KOmegaSst<'m>> {
    let yb = gpu.upload(y)?;
    let rough = NutRoughness::none(hm.n_boundary_faces);
    let mut m = KOmegaSst::new(
        gpu,
        hm,
        gm,
        KOmegaSstCoeffs::default(),
        ctrl,
        WallFunctionCoeffs::default(),
        walls,
        &yb,
        &rough,
    )?;
    m.force_f1(f1);
    let k: Vec<Scalar> = hm.c.iter().map(|&c| k_at(c)).collect();
    let w: Vec<Scalar> = hm.c.iter().map(|&c| w_at(c)).collect();
    gpu.write(&mut m.k_mut().f, &k)?;
    gpu.write(&mut m.omega_mut().f, &w)?;
    m.initialise(gpu, flow)?;
    Ok(m)
}

fn rho_no_walls(hm: &HostMesh) -> (WallFaces, Vec<Scalar>) {
    (
        WallFaces::none(hm.n_boundary_faces),
        vec![ofgpu::walldistance::NO_WALL; hm.n_cells],
    )
}

/// Every state the bitwise row compares: k, omega, nu_t, and the matrix the
/// correction assembled last (the k equation's).
fn rho_snapshot(gpu: &Gpu, m: &KOmegaSst) -> Result<Vec<(&'static str, Vec<Scalar>)>> {
    gpu.sync()?;
    let a = &m.core().a;
    Ok(vec![
        ("k", gpu.download(&m.k().f)?),
        ("omega", gpu.download(&m.omega().f)?),
        ("nut", gpu.download(&m.nut().f)?),
        ("nut boundary", gpu.download(&m.nut().bf)?),
        ("diag", gpu.download(&a.diag)?),
        ("upper", gpu.download(&a.upper)?),
        ("lower", gpu.download(&a.lower)?),
        ("source", gpu.download(&a.source)?),
    ])
}

fn rho_bitwise(what: &str, step: usize, a: &[(&str, Vec<Scalar>)], b: &[(&str, Vec<Scalar>)]) {
    for ((name, x), (_, y)) in a.iter().zip(b.iter()) {
        assert_eq!(x.len(), y.len(), "{what}, step {step}: {name} lengths");
        for (i, (p, q)) in x.iter().zip(y.iter()).enumerate() {
            assert!(
                p.to_bits() == q.to_bits(),
                "{what}, step {step}: {name}[{i}] kinematic {p:e} vs rho form {q:e}"
            );
        }
    }
}

/// One scenario of the rho == 1 row: the kinematic `correct` and
/// `correct_rho` with every density 1 and both fluxes the kinematic one,
/// side by side, compared bit for bit after every step.
#[allow(clippy::too_many_arguments)]
fn rho_one_scenario(
    gpu: &Gpu,
    what: &str,
    hm: &HostMesh,
    gm: &GpuMesh,
    ctrl: TurbulenceControls,
    walls: &WallFaces,
    y: &[Scalar],
    f1: Option<Scalar>,
    u_at: &dyn Fn(Vec3) -> Vec3,
    k_at: &dyn Fn(Vec3) -> Scalar,
    w_at: &dyn Fn(Vec3) -> Scalar,
    steps: usize,
) -> Result<()> {
    let one = |_: Vec3| 1.0 as Scalar;
    let fl = rho_flow(gpu, hm, gm, u_at, &one, &one, false)?;
    let nu: Scalar = 1e-5;
    let kin_flow = FlowState::new(&fl.u, &fl.phi_v, nu);
    let rho_flow_state = FlowState::new(&fl.u, &fl.phi_m, nu);

    let mut kin = rho_model(gpu, hm, gm, ctrl, walls, y, f1, k_at, w_at, &kin_flow)?;
    let mut rho = rho_model(gpu, hm, gm, ctrl, walls, y, f1, k_at, w_at, &rho_flow_state)?;
    rho_bitwise(what, 0, &rho_snapshot(gpu, &kin)?, &rho_snapshot(gpu, &rho)?);

    let dens = fl.density();
    for step in 1..=steps {
        kin.correct(gpu, &kin_flow)?;
        rho.correct_rho(gpu, &rho_flow_state, &dens)?;
        rho_bitwise(what, step, &rho_snapshot(gpu, &kin)?, &rho_snapshot(gpu, &rho)?);
    }
    Ok(())
}

/// §156.6 row 1 (D, both builds): with rho == 1 the rho form's matrices and
/// fields are the kinematic SST's bit for bit - the two decays the older
/// tests run (F1 forced to 1 and to 0), and a sheared, dilating channel with
/// wall functions, computed F1/F2 and the production limiter active, under
/// Euler with `bounded linearUpwind` and under `backward`.
#[test]
fn rho_one_is_bitwise_kinematic() -> Result<()> {
    let Some(gpu) = rho_gpu() else {
        eprintln!("rho_one_is_bitwise_kinematic: no GPU, skipped");
        return Ok(());
    };

    // The two decays: a closed 4x4 box, u = 0.
    let (hm, gm) = rho_mesh(&gpu, [4, 4], "patch", "patch")?;
    let (walls, y) = rho_no_walls(&hm);
    let still = |_: Vec3| Vec3::ZERO;
    let k0 = |_: Vec3| 1e-2 as Scalar;
    let w0 = |_: Vec3| 10.0 as Scalar;
    for f1 in [1.0 as Scalar, 0.0] {
        let ctrl = rho_controls(1e-2, DdtScheme::Euler, DivScheme::Upwind, false);
        let what = format!("decay, F1 forced to {f1}");
        rho_one_scenario(&gpu, &what, &hm, &gm, ctrl, &walls, &y, Some(f1), &still, &k0, &w0, 20)?;
    }

    // The channel: walls at y = 0 and 1 with wall functions, a parabolic
    // shear plus a dilatation, k and omega varying so grad k . grad omega and
    // F1 are not trivial, omega small at small x so G > 10 beta* k omega
    // there (the k limiter active).
    let (hm, gm) = rho_mesh(&gpu, [6, 6], "patch", "wall")?;
    let nbf = hm.n_boundary_faces;
    let wall: Vec<bool> = (0..nbf).map(|i| hm.b_patch[i] == 2 || hm.b_patch[i] == 3).collect();
    let walls = WallFaces {
        constrained_cells: wall.clone(),
        nut: wall,
    };
    let y: Vec<Scalar> = hm.c.iter().map(|c| c.y.min(1.0 - c.y)).collect();
    let u_at = |c: Vec3| Vec3::new(4.0 * c.y * (1.0 - c.y) + 0.3 * c.x, 0.1 * c.x * c.y, 0.0);
    let k_at = |c: Vec3| 1e-3 * (1.0 + 0.5 * c.x + 0.3 * c.y * c.y);
    let w_at = |c: Vec3| 2.0 + 48.0 * c.x + 5.0 * c.y;
    // dt = 2^-10: backward's a_N + a_0 + a_00 is then exactly zero in both
    // builds, which fvm_ddt requires (at dt = 1e-3 it is not, in float).
    let dt: Scalar = 1.0 / 1024.0;
    let euler = rho_controls(dt, DdtScheme::Euler, DivScheme::LinearUpwind, true);
    rho_one_scenario(
        &gpu, "channel, Euler, bounded linearUpwind", &hm, &gm, euler, &walls, &y, None, &u_at, &k_at,
        &w_at, 8,
    )?;
    let bdf2 = rho_controls(dt, DdtScheme::Backward, DivScheme::Upwind, false);
    rho_one_scenario(
        &gpu, "channel, backward, upwind", &hm, &gm, bdf2, &walls, &y, None, &u_at, &k_at, &w_at, 8,
    )?;
    Ok(())
}

/// Homogeneous decay of the rho form at uniform rho = 1.3: N steps of dt to
/// t = 5/(beta2 omega0), at F1 = 0.
fn rho_decay(gpu: &Gpu, n_steps: usize) -> Result<(f64, f64)> {
    let (hm, gm) = rho_mesh(gpu, [2, 2], "patch", "patch")?;
    let (walls, y) = rho_no_walls(&hm);
    let c = KOmegaSstCoeffs::default();
    let w0: Scalar = 10.0;
    let t_end = 5.0 / (rho_w(c.beta_2) * rho_w(w0));
    let dt = (t_end / n_steps as f64) as Scalar;
    let rho13 = |_: Vec3| 1.3 as Scalar;
    let fl = rho_flow(gpu, &hm, &gm, &|_| Vec3::ZERO, &rho13, &rho13, true)?;
    let flow = FlowState::new(&fl.u, &fl.phi_m, 1e-5);
    let ctrl = rho_controls(dt, DdtScheme::Euler, DivScheme::Upwind, false);
    let mut m = rho_model(
        gpu, &hm, &gm, ctrl, &walls, &y, Some(0.0), &|_| 1e-2, &|_| w0, &flow,
    )?;
    let dens = fl.density();
    for _ in 0..n_steps {
        m.correct_rho(gpu, &flow, &dens)?;
    }
    gpu.sync()?;
    let k = gpu.download(&m.k().f)?;
    let w = gpu.download(&m.omega().f)?;
    Ok((rho_w(k[0]), rho_w(w[0])))
}

/// §156.6 row 2 (P): (156.11) at t = 5/(beta2 omega0), uniform rho = 1.3,
/// F1 = 0. The k error is first order in dt - measured between N = 1000 and
/// 2000 steps, where the truncation error (about 4.7/N) is far above the
/// round-off the run accumulates (about N u, which under `single` reaches
/// the truncation error near N = 10^4) - and the finest run, N = 10^4, is
/// within 1e-3 of (156.11) in k and omega. The semi-implicit omega update is
/// exact for this Riccati equation - `1/omega` grows by exactly `beta2 dt`
/// per step - so omega carries round-off only (§156.7).
#[test]
fn homogeneous_decay_closed_form() -> Result<()> {
    let Some(gpu) = rho_gpu() else {
        eprintln!("homogeneous_decay_closed_form: no GPU, skipped");
        return Ok(());
    };
    let c = KOmegaSstCoeffs::default();
    let (b2, bs) = (rho_w(c.beta_2), rho_w(c.beta_star));
    let (k0, w0) = (1e-2_f64, 10.0_f64);
    let tau = 1.0 + b2 * w0 * (5.0 / (b2 * w0));
    let k_exact = k0 * tau.powf(-bs / b2);
    let w_exact = w0 / tau;

    let (k1, _) = rho_decay(&gpu, 1000)?;
    let (k2, _) = rho_decay(&gpu, 2000)?;
    let (k3, w3) = rho_decay(&gpu, 10000)?;
    let err = |k: f64| (k / k_exact - 1.0).abs();
    let (e1, e2, e3) = (err(k1), err(k2), err(k3));
    let ew = (w3 / w_exact - 1.0).abs();
    let ratio = e1 / e2;
    eprintln!(
        "homogeneous decay: k err {e1:.4e} (N=1000) {e2:.4e} (N=2000) ratio {ratio:.4}; \
         N=10000: k err {e3:.4e}, omega err {ew:.3e}"
    );
    assert!((1.8..=2.2).contains(&ratio), "k error ratio {ratio} is not first order");
    assert!(e3 <= 1e-3, "k at N = 10000 is {e3:e} from (156.11)");
    assert!(ew <= 1e-3, "omega at N = 10000 is {ew:e} from (156.11)");
    Ok(())
}

/// §156.6 row 3 (R): SST-sust with k_amb = k0 and omega_amb = omega0 holds
/// the homogeneous state, at rho = 1.3 and the computed F1.
#[test]
fn sust_holds_ambient() -> Result<()> {
    let Some(gpu) = rho_gpu() else {
        eprintln!("sust_holds_ambient: no GPU, skipped");
        return Ok(());
    };
    let (hm, gm) = rho_mesh(&gpu, [3, 3], "patch", "patch")?;
    let (walls, y) = rho_no_walls(&hm);
    let (k0, w0): (Scalar, Scalar) = (1e-2, 10.0);
    let rho13 = |_: Vec3| 1.3 as Scalar;
    let fl = rho_flow(&gpu, &hm, &gm, &|_| Vec3::ZERO, &rho13, &rho13, true)?;
    let flow = FlowState::new(&fl.u, &fl.phi_m, 1e-5);
    let ctrl = rho_controls(1e-2, DdtScheme::Euler, DivScheme::Upwind, false);
    let mut m = rho_model(&gpu, &hm, &gm, ctrl, &walls, &y, None, &|_| k0, &|_| w0, &flow)?;
    m.set_rho_options(RhoSstOptions {
        limit_omega_production: true,
        sust: Some(SstSust {
            k_amb: k0,
            omega_amb: w0,
        }),
    })?;
    let dens = fl.density();
    let mut worst = 0.0_f64;
    for _ in 0..50 {
        m.correct_rho(&gpu, &flow, &dens)?;
        gpu.sync()?;
        for v in gpu.download(&m.k().f)? {
            worst = worst.max((rho_w(v) / rho_w(k0) - 1.0).abs());
        }
        for v in gpu.download(&m.omega().f)? {
            worst = worst.max((rho_w(v) / rho_w(w0) - 1.0).abs());
        }
    }
    eprintln!("sust: worst relative drift of k and omega over 50 steps {worst:.3e}");
    assert!(worst <= RHO_R, "SST-sust drifted {worst:e} from the ambient state");

    // Without the sustaining terms the same state decays - the row tests
    // the terms, not a frozen solver.
    m.set_rho_options(RhoSstOptions::default())?;
    m.correct_rho(&gpu, &flow, &dens)?;
    gpu.sync()?;
    let k = gpu.download(&m.k().f)?;
    assert!(rho_w(k[0]) < rho_w(k0) * (1.0 - 1e-4), "k did not decay without sust: {}", k[0]);
    Ok(())
}

/// §156.6 row 4 (named A; class R, see `RHO_A`): a closed box with
/// zero-gradient k and omega on every patch, an internal mass flux, rho
/// varying in space and between the time levels. After each Euler step
/// `sum V (rho k' - rho0 k)/dt` equals
/// `sum V [rho min(G, ..) - rho beta* omega' k' - (2/3) rho div(u) k*]`
/// (k* the level fvm_susp put it on) - the convection and diffusion
/// telescope - summed on the host in f64.
#[test]
fn rho_k_budget_closes() -> Result<()> {
    let Some(gpu) = rho_gpu() else {
        eprintln!("rho_k_budget_closes: no GPU, skipped");
        return Ok(());
    };
    let (hm, gm) = rho_mesh(&gpu, [5, 4], "patch", "patch")?;
    let (walls, y) = rho_no_walls(&hm);
    let pi = std::f64::consts::PI;
    let u_at = move |c: Vec3| {
        let (x, yy) = (rho_w(c.x), rho_w(c.y));
        Vec3::new((0.5 * (pi * x).sin()) as Scalar, (0.3 * (pi * yy).sin()) as Scalar, 0.0)
    };
    let rho_new = |c: Vec3| 1.0 + 0.4 * c.x + 0.2 * c.y;
    let rho_old = |c: Vec3| 0.9 * (1.0 + 0.3 * c.x);
    let fl = rho_flow(&gpu, &hm, &gm, &u_at, &rho_new, &rho_old, true)?;
    let flow = FlowState::new(&fl.u, &fl.phi_m, 1e-5);
    let dt: Scalar = 1e-3;
    let ctrl = rho_controls(dt, DdtScheme::Euler, DivScheme::Upwind, false);
    let k_at = |c: Vec3| 1e-2 * (1.0 + 0.5 * c.x * c.y);
    let w_at = |c: Vec3| 10.0 + 4.0 * c.y;
    let mut m = rho_model(&gpu, &hm, &gm, ctrl, &walls, &y, None, &k_at, &w_at, &flow)?;
    let dens = fl.density();

    let v = &hm.v;
    let rho_n: Vec<Scalar> = gpu.download(&fl.rho)?;
    let rho_o: Vec<Scalar> = gpu.download(&fl.rho0)?;
    let c = KOmegaSstCoeffs::default();
    for step in 1..=3 {
        gpu.sync()?;
        let k_old = gpu.download(&m.k().f)?;
        m.correct_rho(&gpu, &flow, &dens)?;
        gpu.sync()?;
        let k_new = gpu.download(&m.k().f)?;
        let w_new = gpu.download(&m.omega().f)?;
        let prod = gpu.download(m.k_production())?;
        let susp = gpu.download(&m.core().susp)?;
        // The scale is the gross size of every term, storage included
        // (sum V (rho k' + rho0 k)/dt): a stored k carries its round-off
        // into the storage term at that size, whatever the net change is.
        let (mut lhs, mut rhs, mut scale) = (0.0_f64, 0.0_f64, 0.0_f64);
        for i in 0..hm.n_cells {
            let vi = rho_w(v[i]);
            let (st_new, st_old) = (
                vi * rho_w(rho_n[i]) * rho_w(k_new[i]) / rho_w(dt),
                vi * rho_w(rho_o[i]) * rho_w(k_old[i]) / rho_w(dt),
            );
            let sink = vi * rho_w(rho_n[i]) * rho_w(c.beta_star) * rho_w(w_new[i]) * rho_w(k_new[i]);
            let s = rho_w(susp[i]);
            let dil = vi * s * if s > 0.0 { rho_w(k_new[i]) } else { rho_w(k_old[i]) };
            let p = vi * rho_w(prod[i]);
            lhs += st_new - st_old;
            rhs += p - sink - dil;
            scale += st_new + st_old + p.abs() + sink.abs() + dil.abs();
        }
        let err = (lhs - rhs).abs() / scale;
        eprintln!("rho k budget, step {step}: lhs {lhs:.6e} rhs {rhs:.6e} rel {err:.3e}");
        assert!(err <= RHO_A, "step {step}: the rho k budget misses by {err:e} of its gross terms");
    }
    Ok(())
}

/// §156.6 row 5 (R): u = (a x, 0, 0), uniform rho = 1.3: the k equation's
/// production `rho min(G, ..) - (2/3) rho div(u) k` is
/// `mu_t (2 S:S - (2/3)(div u)^2) - (2/3) rho k div u = rho nu_t (4/3) a^2
/// - (2/3) rho k a`, both dilatation terms present.
#[test]
fn dilatation_production_terms() -> Result<()> {
    let Some(gpu) = rho_gpu() else {
        eprintln!("dilatation_production_terms: no GPU, skipped");
        return Ok(());
    };
    let (hm, gm) = rho_mesh(&gpu, [4, 4], "patch", "patch")?;
    let (walls, y) = rho_no_walls(&hm);
    let a: Scalar = 0.2;
    let rho: Scalar = 1.3;
    let u_at = move |c: Vec3| Vec3::new(a * c.x, 0.0, 0.0);
    let fl = rho_flow(&gpu, &hm, &gm, &u_at, &move |_| rho, &move |_| rho, false)?;
    let flow = FlowState::new(&fl.u, &fl.phi_m, 1e-5);
    let ctrl = rho_controls(1e-3, DdtScheme::Euler, DivScheme::Upwind, false);
    let k0: Scalar = 1e-2;
    let mut m = rho_model(&gpu, &hm, &gm, ctrl, &walls, &y, Some(0.0), &|_| k0, &|_| 10.0, &flow)?;
    gpu.sync()?;
    let nut = gpu.download(&m.nut().f)?;
    m.correct_rho(&gpu, &flow, &fl.density())?;
    gpu.sync()?;
    let prod = gpu.download(m.k_production())?;
    let susp = gpu.download(&m.core().susp)?;
    let (a, r, k) = (rho_w(a), rho_w(rho), rho_w(k0));
    let mut worst = 0.0_f64;
    for i in 0..hm.n_cells {
        let host = r * rho_w(nut[i]) * (2.0 * a * a - (2.0 / 3.0) * a * a) - (2.0 / 3.0) * r * k * a;
        let dev = rho_w(prod[i]) - rho_w(susp[i]) * k;
        let err = (dev - host).abs() / host.abs();
        worst = worst.max(err);
        assert!(err <= RHO_R, "cell {i}: production {dev:e} vs (156.3) {host:e}, rel {err:e}");
    }
    eprintln!("dilatation production: worst rel {worst:.3e} over {} cells", hm.n_cells);
    Ok(())
}

/// §156.6 row 6: `limitOmegaProduction`. At false the omega Su is the
/// kinematic one times rho bit for bit (D); at true it is
/// `blend(gamma) rho min(G, 10 beta* k omega)/nu_t` in the limited cells
/// (R) and the false value bit for bit elsewhere. u = (s y, 0, 0),
/// s = 20, k = 1e-3, omega = 1 + 99 x, F1 = 0, rho = 1.3: the column at
/// omega = 13.4 is limited (G/lim = 2.5), the other three are not (<= 0.31).
#[test]
fn omega_limiter_setting() -> Result<()> {
    let Some(gpu) = rho_gpu() else {
        eprintln!("omega_limiter_setting: no GPU, skipped");
        return Ok(());
    };
    let (hm, gm) = rho_mesh(&gpu, [4, 4], "patch", "patch")?;
    let (walls, y) = rho_no_walls(&hm);
    let s: Scalar = 20.0;
    let rho: Scalar = 1.3;
    let u_at = move |c: Vec3| Vec3::new(s * c.y, 0.0, 0.0);
    let fl = rho_flow(&gpu, &hm, &gm, &u_at, &move |_| rho, &move |_| rho, false)?;
    let kin_flow = FlowState::new(&fl.u, &fl.phi_v, 1e-5);
    let flow = FlowState::new(&fl.u, &fl.phi_m, 1e-5);
    let ctrl = rho_controls(1e-3, DdtScheme::Euler, DivScheme::Upwind, false);
    let k_at = |_: Vec3| 1e-3 as Scalar;
    let w_at = |c: Vec3| 1.0 + 99.0 * c.x;
    let mk = |f: &FlowState| rho_model(&gpu, &hm, &gm, ctrl, &walls, &y, Some(0.0), &k_at, &w_at, f);

    let mut kin = mk(&kin_flow)?;
    kin.correct(&gpu, &kin_flow)?;
    let mut off = mk(&flow)?;
    gpu.sync()?;
    let nut = gpu.download(&off.nut().f)?;
    off.correct_rho(&gpu, &flow, &fl.density())?;
    let mut on = mk(&flow)?;
    on.set_rho_options(RhoSstOptions {
        limit_omega_production: true,
        sust: None,
    })?;
    on.correct_rho(&gpu, &flow, &fl.density())?;
    gpu.sync()?;
    let su_kin = gpu.download(&kin.core().su)?;
    let su_off = gpu.download(&off.core().su)?;
    let su_on = gpu.download(&on.core().su)?;

    let c = KOmegaSstCoeffs::default();
    let mut limited = 0usize;
    let mut worst = 0.0_f64;
    for i in 0..hm.n_cells {
        assert!(
            su_off[i].to_bits() == (su_kin[i] * rho).to_bits(),
            "cell {i}: rho form Su {:e} is not rho times the kinematic {:e}",
            su_off[i],
            su_kin[i]
        );
        let (k, w) = (rho_w(k_at(hm.c[i])), rho_w(w_at(hm.c[i])));
        let lim = rho_w(c.c1) * rho_w(c.beta_star) * k * w;
        let g = rho_w(nut[i]) * rho_w(s) * rho_w(s);
        if g > 2.0 * lim {
            limited += 1;
            let host = rho_w(c.gamma_2) * rho_w(rho) * lim / rho_w(nut[i]);
            let err = (rho_w(su_on[i]) - host).abs() / host;
            worst = worst.max(err);
            assert!(err <= RHO_R, "cell {i}: limited Su {:e} vs {host:e}, rel {err:e}", su_on[i]);
        } else {
            assert!(g < 0.5 * lim, "cell {i} sits too near the limiter: G/lim = {}", g / lim);
            assert!(
                su_on[i].to_bits() == su_off[i].to_bits(),
                "cell {i}: unlimited Su moved: {:e} vs {:e}",
                su_on[i],
                su_off[i]
            );
        }
    }
    assert!(limited > 0 && limited < hm.n_cells, "{limited} of {} cells limited", hm.n_cells);
    eprintln!("omega limiter: {limited} of {} cells limited, worst rel {worst:.3e}", hm.n_cells);
    Ok(())
}
