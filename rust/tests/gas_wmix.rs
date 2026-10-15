// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CHR-11's gates (SPEC-LIT 133.10).
//!
//! * `gas_constant_w_default_bitwise` (class D): a `GasState` without a
//!   mixture - one gas of constant `W`, §25 as it was - gives the same
//!   `rho`, target divergence, `T` and `p0`, bit for bit, as the parent
//!   commit 41d449a did. The pins were captured there, one per build.
//! * `gas_wmix_sealed_box_p0_closed_form` (class A): a sealed isothermal box
//!   with a mole-changing, heat-neutral test reaction gives `dp0/dt` equal to
//!   the closed form (133.23) to 1e-12 relative in both builds.
//!
//! An integration test, so it runs in both builds; a machine without a card
//! passes vacuously.

use ofgpu::blockgen::{build_mesh, BlockSpec, GradedAxis};
use ofgpu::device::{DevBuf, Gpu};
use ofgpu::energy::{
    ConstraintIntegrals, DomainKind, Energy, EnergyControls, GasMixture, GasProperties, GasState,
};
use ofgpu::field::{BcKind, GpuScalarField, GpuSurfaceScalarField};
use ofgpu::fv::SnGradScheme;
use ofgpu::mesh::{GpuMesh, HostMesh};
use ofgpu::timescheme::DdtScheme;
use ofgpu::{Label, Scalar};

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// An `n^3` box of side `side` m, walls on all six sides.
fn box_mesh(n: usize, side: f64) -> HostMesh {
    let axis = || GradedAxis { lo: 0.0, hi: side as Scalar, n, expansion: 1.0, two_sided: false };
    let b = BlockSpec {
        x: axis(),
        y: axis(),
        z: axis(),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall"; 6].map(String::from),
        cyclic: Vec::new(),
    };
    build_mesh(&b).expect("box mesh")
}

/// The laminar transient `Energy`: Euler, tolerance 1e-14, relTol 0.
fn energy_on<'m>(g: &Gpu, m: &'m GpuMesh, dt: f64) -> Energy<'m> {
    let ctrl = EnergyControls {
        t_solver: ofgpu::solver::SolverControls {
            tolerance: 1e-14,
            rel_tol: 0.0,
            max_iter: 2000,
            check_interval: 1,
            ..ofgpu::solver::SolverControls::default()
        },
        t_relax: 1.0,
        steady: false,
        delta_t: dt as Scalar,
        sn_grad: SnGradScheme::Uncorrected,
        ddt: DdtScheme::Euler,
        ..EnergyControls::default()
    };
    Energy::new(g, m, ctrl, GasProperties::default()).expect("energy")
}

fn calculated_nut(g: &Gpu, m: &GpuMesh) -> GpuScalarField {
    let mut nut = GpuScalarField::zeros(g, m, "nut").expect("nut");
    g.write(&mut nut.bc_kind, &vec![BcKind::Calculated as Label; m.n_boundary_faces])
        .expect("nut kinds");
    nut
}

fn bytes_of(g: &Gpu, b: &DevBuf<Scalar>, out: &mut Vec<u8>) {
    for v in g.download(b).expect("download") {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

// ==========================================================================
//  gas_constant_w_default_bitwise
// ==========================================================================

/// The pins: SHA-256 of the bytes `default_run` collects, captured at the
/// parent commit 41d449a on this machine's card and driver. On another card,
/// re-capture both from 41d449a, never from CHR-11 or later.
#[cfg(not(feature = "single"))]
const DEFAULT_PIN: &str = "a43cff96df6ab44703aeae9445fd1e1a81611a383841160f0616c46397558258";
#[cfg(feature = "single")]
const DEFAULT_PIN: &str = "0e6783841dd598eadfbabfa6dbfc3deb7d025beb516b9f3a8b02af6006db4061";

/// §25 as it ran before CHR-11, through the API it had: a sealed 4^3 walled
/// box (adiabatic walls, no flow), `T` 300-330 K by cell, a uniform 1e3 W/m3
/// heater, Euler at dt 1e-3 s, three steps of `update_density`,
/// `update_target_divergence`, `correct` and `advance_p0` in
/// `ofgpu-lowmach`'s order. Returns every `rho` cell and face value, every
/// target divergence, every `T` and each step's `p0` and `dp0/dt`, as bytes.
fn default_run(g: &Gpu) -> Vec<u8> {
    let hm = box_mesh(4, 0.01);
    let m = GpuMesh::upload(g, &hm).expect("mesh");
    let n = hm.n_cells;
    let dt = 1.0e-3;
    let mut e = energy_on(g, &m, dt);
    let tv: Vec<Scalar> = (0..n).map(|c| (300.0 + 30.0 * c as f64 / n as f64) as Scalar).collect();
    g.write(&mut e.field_mut().f, &tv).expect("T");
    e.initialise(g).expect("T initialise");
    let props = GasProperties::default();
    let p0_init = props.r_s() * 300.0 * 1.2;
    let mut gas = GasState::new(g, &m, props, DomainKind::Sealed, p0_init).expect("gas");
    gas.update_density(g, e.field()).expect("rho");
    gas.seed_time_levels();
    let heater = g.upload(&vec![1.0e3 as Scalar; n]).expect("heater");
    let phi = GpuSurfaceScalarField::zeros(g, &m, "phi").expect("phi");
    let nut = calculated_nut(g, &m);
    let mut out = Vec::new();
    for s in 0..3 {
        if s > 0 {
            e.advance_time_levels(g).expect("T levels");
            e.advance_time_step(dt as Scalar);
            gas.advance_time_levels();
        }
        gas.update_density(g, e.field()).expect("rho");
        e.sources_mut().clear(g).expect("clear");
        e.sources_mut().register_explicit(g, &heater).expect("heater");
        e.update_target_divergence(g, &gas, &nut, &nut.f, 1.5e-5).expect("target");
        e.correct(g, &phi, &nut, &nut.f, 1.5e-5, &gas).expect("energy");
        let total_q = e.sources_mut().total_q(g, &m).expect("total q")
            + e.total_conduction_q(g).expect("conduction q");
        gas.advance_p0(total_q, dt as Scalar).expect("p0");
        for b in [&gas.rho().f, &gas.rho().bf, &gas.rho().f0, e.target_divergence(), &e.field().f] {
            bytes_of(g, b, &mut out);
        }
        out.extend_from_slice(&gas.p0().to_le_bytes());
        out.extend_from_slice(&gas.dp0dt().to_le_bytes());
    }
    out
}

/// **SPEC-LIT 133.10, row D.** With no mixture, `GasState` and the target
/// divergence run every launch they ran before CHR-11: the bytes of
/// `default_run` hash to the parent commit's pin in each build.
#[test]
fn gas_constant_w_default_bitwise() {
    let Some(g) = gpu() else { return };
    let bytes = default_run(&g);
    let got = ofgpu::radiation::wsgg::sha256_hex(&bytes);
    println!("  [133.10] default path: {} bytes, sha256 {got}", bytes.len());
    assert_eq!(got, DEFAULT_PIN, "the constant-W path moved against 41d449a");
}

// ==========================================================================
//  gas_wmix_sealed_box_p0_closed_form
// ==========================================================================

/// The test reaction's members, kg/mol: `A` with the molar mass of NO2 and
/// an inert `N` of exactly twice it, so `A -> N` halves the moles at fixed
/// mass, the shape of the dimerisation 2 NO2 -> N2O4. Nothing registers its
/// heat: the reaction is heat-neutral by construction.
const W_A: f64 = 46.0055e-3;
const W_N: f64 = 2.0 * W_A;
/// `S_A`, kg/(m3 s): `A`'s net mass source, uniform.
const S_A: f64 = -2.0;
/// The temperature the isothermal leg holds, K.
const T_BOX: f64 = 300.0;

#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    f64::from(x)
}

/// D-F32-1's class-R tolerance: `max(tol64, min(tol64 2^29, 1e-5))` under
/// single, `tol64` otherwise (docs/17 §5.1 rule 4).
fn tol_r(t: f64) -> f64 {
    if cfg!(feature = "single") {
        t.max((t * 536870912.0).min(1e-5))
    } else {
        t
    }
}

fn mixture() -> GasMixture {
    GasMixture::new(&[("A", W_A)], ("N", W_N)).expect("mixture")
}

/// Every level and every boundary face of `y` set to `v`.
fn set_all(g: &Gpu, y: &mut GpuScalarField, v: f64, n: usize, nbf: usize) {
    let c = vec![v as Scalar; n];
    g.write(&mut y.f, &c).expect("Y");
    g.write(&mut y.f0, &c).expect("Y0");
    g.write(&mut y.f00, &c).expect("Y00");
    g.write(&mut y.bf, &vec![v as Scalar; nbf]).expect("Yb");
}

/// `Σ_c rho_c V_c`, kg.
fn mass_of(g: &Gpu, gas: &GasState, m: &GpuMesh) -> f64 {
    let rho = g.download(&gas.rho().f).expect("rho");
    let v = g.download(&m.v).expect("V");
    rho.iter().zip(&v).map(|(a, b)| w64(*a) * w64(*b)).sum()
}

/// **SPEC-LIT 133.10, row A.** A sealed 2^3 walled box, `T` held at 300 K
/// (the energy equation is not solved: isothermal), no flow, no heat source,
/// a mixture of `A` and the inert `N = 2 A` with `S_A = -2 kg/(m3 s)`. At
/// three states `Y_A` in {0.9, 0.6, 0.3}, `dp0/dt` from (133.21)-(133.22)
/// through `Energy::constraint_integrals` and `GasState::advance_p0_mixture`
/// equals the closed form (133.23)
///
/// ```text
/// dp0/dt = γ R T S_A (1/W_A - 1/W_N),   γ = cp/(cp - R/W),   1/W from (133.19)
/// ```
///
/// evaluated in f64 from the stored inputs, to 1e-12 relative in both builds
/// (class A: the device forms it in f64 too). `p0` steps by exactly
/// `dt dp0/dt` (D); `rho` is (133.20) and, once `p0` has moved, the target
/// divergence of the same state is zero against its own `D` (R, D-F32-1).
/// Leg (b) solves `T` (adiabatic) and prints the equation-of-state drift
/// §133.2 names; the refusals are named.
#[test]
fn gas_wmix_sealed_box_p0_closed_form() {
    let Some(g) = gpu() else { return };
    let hm = box_mesh(2, 0.01);
    let m = GpuMesh::upload(&g, &hm).expect("mesh");
    let (n, nbf) = (hm.n_cells, hm.n_boundary_faces);
    let dt = 1.0e-4;
    let props = GasProperties::default();
    let (r, cp) = (w64(props.r_universal), w64(props.cp));
    let dinv = 1.0 / W_A - 1.0 / W_N;
    let phi = GpuSurfaceScalarField::zeros(&g, &m, "phi").expect("phi");
    let nut = calculated_nut(&g, &m);
    let src = g.upload(&vec![S_A as Scalar; n]).expect("S_A");
    let s_a = w64(S_A as Scalar);

    // ---- (a) the isothermal box: three states --------------------------
    let mut e = energy_on(&g, &m, dt);
    g.write(&mut e.field_mut().f, &vec![T_BOX as Scalar; n]).expect("T");
    e.initialise(&g).expect("T initialise");
    let mut gas = GasState::new(&g, &m, props, DomainKind::Sealed, 1.0e5).expect("gas");
    gas.use_mixture(&g, mixture()).expect("use mixture");
    let mut ya = GpuScalarField::zeros(&g, &m, "A").expect("Y_A");
    let err = e.update_target_divergence(&g, &gas, &nut, &nut.f, 1.5e-5).unwrap_err().to_string();
    assert!(err.contains("update_density_mixture"), "an unformed mixture: {err}");
    let (mut worst_rate, mut worst_rho, mut worst_div) = (0.0f64, 0.0f64, 0.0f64);
    for y0 in [0.9, 0.6, 0.3] {
        set_all(&g, &mut ya, y0, n, nbf);
        gas.update_density_mixture(&g, e.field(), &[&ya]).expect("rho mixture");
        gas.set_molar_change(&g, &[&src]).expect("molar change");
        e.update_target_divergence(&g, &gas, &nut, &nut.f, 1.5e-5).expect("target");
        let ci = e.constraint_integrals(&g).expect("integrals");
        let p0_before = w64(gas.p0());

        let y = w64(y0 as Scalar);
        let t = w64(T_BOX as Scalar);
        let inv_w = mixture().inverse_molar_mass(&[y]);
        let gamma = cp / (cp - r * inv_w);
        let want = gamma * r * t * s_a * dinv;
        for (c, v) in g.download(&gas.rho().f).expect("rho").into_iter().enumerate() {
            let host = p0_before / (r * t * inv_w);
            let rel = (w64(v) - host).abs() / host;
            assert!(rel <= tol_r(1e-12), "cell {c}: rho {v} against (133.20)'s {host}");
            worst_rho = worst_rho.max(rel);
        }

        let rate = gas.advance_p0_mixture(ci, dt as Scalar).expect("p0");
        let rel = (rate - want).abs() / want.abs();
        println!("  [133.10] Y_A {y0}: dp0/dt {rate:.15e} Pa/s, (133.23) {want:.15e}, relative {rel:.2e}");
        assert!(rel <= 1e-12, "Y_A {y0}: dp0/dt {rate} against the closed form {want} ({rel:.3e})");
        worst_rate = worst_rate.max(rel);
        assert_eq!(gas.p0(), (p0_before + rate * dt) as Scalar, "p0 is not one Euler step of (133.22)");

        // The same state at the new p0, with dp0/dt now the constraint's own:
        // the box is uniform and sealed, so div u is zero in every cell.
        gas.update_density_mixture(&g, e.field(), &[&ya]).expect("rho mixture");
        e.update_target_divergence(&g, &gas, &nut, &nut.f, 1.5e-5).expect("target");
        let d = r * t * s_a * dinv / w64(gas.p0());
        for v in g.download(e.target_divergence()).expect("target") {
            let rel = w64(v).abs() / d.abs();
            assert!(rel <= tol_r(1e-12), "Y_A {y0}: the target divergence {v} against D {d}");
            worst_div = worst_div.max(rel);
        }
    }
    println!(
        "  [133.10] isothermal box: dp0/dt {worst_rate:.2e}, rho {worst_rho:.2e}, \
         target divergence {worst_div:.2e} (relative)"
    );

    // ---- refusals, by name -------------------------------------------
    let e1 = gas.update_density(&g, e.field()).unwrap_err().to_string();
    assert!(e1.contains("update_density_mixture"), "{e1}");
    let e2 = gas.advance_p0(0.0, dt as Scalar).unwrap_err().to_string();
    assert!(e2.contains("advance_p0_mixture"), "{e2}");
    let e3 = gas.update_density_mixture(&g, e.field(), &[&ya, &ya]).unwrap_err().to_string();
    assert!(e3.contains("2 mass-fraction fields") && e3.contains("\"A\""), "{e3}");
    let e4 = gas.set_molar_change(&g, &[]).unwrap_err().to_string();
    assert!(e4.contains("0 sources"), "{e4}");
    let e5 = GasMixture::new(&[("A", W_A)], ("N", 0.0)).unwrap_err().to_string();
    assert!(e5.contains("\"N\""), "{e5}");
    let mut plain = GasState::new(&g, &m, props, DomainKind::Sealed, 1.0e5).expect("gas");
    plain.update_density(&g, e.field()).expect("rho");
    e.update_target_divergence(&g, &plain, &nut, &nut.f, 1.5e-5).expect("target");
    let e6 = e.constraint_integrals(&g).unwrap_err().to_string();
    assert!(e6.contains("was not a") && e6.contains("mixture"), "{e6}");
    let ci = ConstraintIntegrals { dilatation: 0.0, inv_gamma_volume: 1.0 };
    let e7 = plain.advance_p0_mixture(ci, dt as Scalar).unwrap_err().to_string();
    assert!(e7.contains("without a mixture"), "{e7}");

    // ---- (b) adiabatic: T solved, the equation-of-state drift -----------
    // The sealed uniform box keeps rho; with rho cp dT/dt = dp0/dt and no
    // heat released, p0 = rho R Z (T0 - p0_0/(rho cp))/(1 - R Z/cp), with
    // Z = 1/W(t) and Y_A(t) = Y_A0 + S_A t/rho. Explicit Euler on a lagged
    // dp0/dt is first order: printed at two steps, with the mass drift.
    let (y_a0, p0_init, t_end) = (0.9, 1.0e5, 0.05);
    let rho0 = p0_init / (r * T_BOX * mixture().inverse_molar_mass(&[y_a0]));
    let mut prev: Option<f64> = None;
    for steps in [10usize, 20] {
        let h = t_end / steps as f64;
        let mut e = energy_on(&g, &m, h);
        g.write(&mut e.field_mut().f, &vec![T_BOX as Scalar; n]).expect("T");
        e.initialise(&g).expect("T initialise");
        let mut gas = GasState::new(&g, &m, props, DomainKind::Sealed, p0_init as Scalar).expect("gas");
        gas.use_mixture(&g, mixture()).expect("use mixture");
        let yat = |s: usize| y_a0 + S_A * (s as f64) * h / rho0;
        let mut ya = GpuScalarField::zeros(&g, &m, "A").expect("Y_A");
        set_all(&g, &mut ya, y_a0, n, nbf);
        gas.update_density_mixture(&g, e.field(), &[&ya]).expect("rho");
        gas.seed_time_levels();
        let mass0 = mass_of(&g, &gas, &m);
        for s in 0..steps {
            if s > 0 {
                e.advance_time_levels(&g).expect("T levels");
                e.advance_time_step(h as Scalar);
                gas.advance_time_levels();
            }
            // `ofgpu-lowmach`'s order: the species are solved after `T`, so
            // at the energy solve `Y` still holds the step's opening value
            // on both of its first two levels.
            let lv = |k: usize| vec![yat(s.saturating_sub(k)) as Scalar; n];
            g.write(&mut ya.f, &lv(0)).expect("Y");
            g.write(&mut ya.f0, &lv(0)).expect("Y0");
            g.write(&mut ya.f00, &lv(1)).expect("Y00");
            g.write(&mut ya.bf, &vec![yat(s) as Scalar; nbf]).expect("Yb");
            gas.update_density_mixture(&g, e.field(), &[&ya]).expect("rho");
            gas.set_molar_change(&g, &[&src]).expect("molar change");
            e.sources_mut().clear(&g).expect("clear");
            e.update_target_divergence(&g, &gas, &nut, &nut.f, 1.5e-5).expect("target");
            e.correct(&g, &phi, &nut, &nut.f, 1.5e-5, &gas).expect("energy");
            let ci = e.constraint_integrals(&g).expect("integrals");
            gas.advance_p0_mixture(ci, h as Scalar).expect("p0");
        }
        let z = mixture().inverse_molar_mass(&[yat(steps)]);
        let exact = rho0 * r * z * (T_BOX - p0_init / (rho0 * cp)) / (1.0 - r * z / cp);
        let got = w64(gas.p0());
        let rel = (got - exact).abs() / (p0_init - exact).abs();
        set_all(&g, &mut ya, yat(steps), n, nbf);
        gas.update_density_mixture(&g, e.field(), &[&ya]).expect("rho");
        let drift = (mass_of(&g, &gas, &m) - mass0).abs() / mass0;
        let t_end = w64(g.download(&e.field().f).expect("T")[0]);
        let t_exact = T_BOX + (exact - p0_init) / (rho0 * cp);
        let order = prev.map(|p| (p / rel).log2());
        println!(
            "  [133.10] adiabatic box, {steps} steps: p0 {got:.6} Pa against {exact:.6} \
             (error {rel:.3e} of the change), T {t_end:.6} K against {t_exact:.6}, mass \
             drift {drift:.3e}, order {order:?}"
        );
        assert!(got.is_finite() && drift.is_finite(), "the adiabatic leg is not finite");
        prev = Some(rel);
    }
}
