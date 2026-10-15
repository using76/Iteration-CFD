// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CHR-10b's gates (SPEC-LIT 133.9): the split step's heat release (133.16)
//! and the species-enthalpy diffusion (133.17) enter the temperature
//! equation through `EnergySources`, metered, and the mixture `cp` (133.18)
//! fills CHR-10a's field.
//!
//! * `energy_box_matches_reactor` (class P, 0.5 K): CHR-09's 0-D box, with
//!   `T` now the energy equation's, stays on the host twin
//!   `ConstPressureReactor::advance` (CHR-06) over 200 Lie steps.
//! * `energy_budget_meters_heat_release` (class R, D-F32-1): the assembly
//!   budget closes with the heat release and the enthalpy diffusion
//!   metered as their own entries, and the heat-release entry is the one
//!   the split step released.
//! * `energy_species_enthalpy_matches_host` (class R): (133.18) and
//!   (133.17) on the device against an f64 host evaluation of the same
//!   formulas, and the refusals by name.
//!
//! The chemistry is CHR-06's fifteen-reaction fixture of §130.7 over the
//! ten species of `reference/mcbride-gordon-reno1993/table_II.csv` (NASA
//! TM-4513, a US-Government work), whose rate numbers are PLACEHOLDERS.
//! An integration test, so it runs in both builds; a machine without a card
//! passes vacuously. No GPL-licensed source was consulted.

#![allow(clippy::excessive_precision)]

use ofgpu::blockgen::{build_mesh, BlockSpec, GradedAxis};
use ofgpu::chem::source::{ChemSource, SpeciesEnthalpy};
use ofgpu::chem::{
    ConstPressureReactor, Mechanism, ReactorOptions, SpeciesThermo, ThermoSet, P_ATM, P_BAR,
};
use ofgpu::device::{DevBuf, Gpu};
use ofgpu::energy::{DomainKind, Energy, EnergyBudget, EnergyControls, EnergySources, GasProperties, GasState};
use ofgpu::field::{BcKind, GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use ofgpu::fv::{fvc_grad_scalar, FvKernels, SnGradScheme};
use ofgpu::mesh::{GpuMesh, HostMesh};
use ofgpu::species::{Species, SpeciesCoeffs};
use ofgpu::timescheme::DdtScheme;
use ofgpu::turbulence::{FlowState, TurbulenceControls};
use ofgpu::{Label, Scalar, Vec3};

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// D-F32-1's class-R tolerance: `max(tol64, min(tol64 2^29, 1e-5))` under
/// single, `tol64` otherwise (docs/17 §5.1 rule 4).
const fn tol_r(t: f64) -> f64 {
    if cfg!(feature = "single") {
        let scaled = t * 536870912.0;
        let capped = if scaled < 1e-5 { scaled } else { 1e-5 };
        if t > capped { t } else { capped }
    } else {
        t
    }
}

#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    f64::from(x)
}

fn down(g: &Gpu, b: &DevBuf<Scalar>) -> Vec<f64> {
    g.download(b).expect("download").into_iter().map(w64).collect()
}

/// An `n^3` box of side `side` m, walls on all six sides, or cyclic on
/// every axis when `cyclic` (CHR-09's `tests/chem_species.rs` box).
fn box_mesh(n: usize, side: f64, cyclic: bool) -> HostMesh {
    let axis = || GradedAxis { lo: 0.0, hi: side as Scalar, n, expansion: 1.0, two_sided: false };
    let mut b = BlockSpec {
        x: axis(),
        y: axis(),
        z: axis(),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall"; 6].map(String::from),
        cyclic: Vec::new(),
    };
    if cyclic {
        for a in 0..3 {
            b.set_cyclic_axis(a).expect("cyclic axis");
        }
    }
    build_mesh(&b).expect("box mesh")
}

// ==========================================================================
//  The chemistry fixture (CHR-06's, as tests/chem_species.rs has it)
// ==========================================================================

const TABLE_II: &str = include_str!("../../reference/mcbride-gordon-reno1993/table_II.csv");

#[allow(clippy::type_complexity)]
const CODES: &[(&str, &str, &[(&str, u32)])] = &[
    ("1", "N2", &[("N", 2)]),
    ("2", "O2", &[("O", 2)]),
    ("5", "CH4", &[("C", 1), ("H", 4)]),
    ("6", "CO2", &[("C", 1), ("O", 2)]),
    ("7", "H2O", &[("H", 2), ("O", 1)]),
    ("8", "CO", &[("C", 1), ("O", 1)]),
    ("9", "H2", &[("H", 2)]),
    ("10", "OH", &[("O", 1), ("H", 1)]),
    ("11", "O", &[("O", 1)]),
    ("12", "H", &[("H", 1)]),
];

/// The TM-4513 records as a set at `P_BAR`, bounds 200/1000/6000 K.
fn tm_set() -> ThermoSet {
    let mut rows: Vec<(String, f64, f64, [f64; 7])> = Vec::new();
    for line in TABLE_II.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("species") {
            continue;
        }
        let c: Vec<&str> = line.split(',').collect();
        let mut a = [0.0f64; 7];
        for (j, v) in a.iter_mut().enumerate() {
            *v = c[4 + j].parse().unwrap();
        }
        rows.push((c[0].to_string(), c[1].parse().unwrap(), c[3].parse().unwrap(), a));
    }
    let mut v = Vec::new();
    for (code, name, els) in CODES {
        let pick = |t_lo: f64| {
            rows.iter().find(|(c, t, _, _)| c == code && *t == t_lo).map(|r| (r.2, r.3)).unwrap()
        };
        let (mol_weight, upper) = pick(1000.0);
        let (_, lower) = pick(200.0);
        v.push(SpeciesThermo {
            name: name.to_string(),
            elements: els.iter().map(|(s, n)| (s.to_string(), *n)).collect(),
            molar_mass: mol_weight / 1000.0,
            t_low: 200.0,
            t_mid: 1000.0,
            t_high: 6000.0,
            upper,
            lower,
            line: 0,
        });
    }
    ThermoSet::from_species(v, P_BAR, "TM-4513").unwrap()
}

/// CHR-06's fifteen-reaction fixture of §130.7 (placeholder rates).
fn fixture() -> Mechanism {
    let rx = [
        "H + O2 <=> O + OH                 1.0E+14  -0.5  15000.0",
        "O + H2 <=> H + OH                 1.0E+05   2.5   6000.0",
        "OH + H2 <=> H + H2O               2.0E+08   1.5   3000.0",
        "H + OH + M <=> H2O + M            2.0E+22  -2.0      0.0",
        "  H2O/6.0/ H2/2.0/",
        "O + O + M <=> O2 + M              1.0E+17  -1.0      0.0",
        "H + H (+M) <=> H2 (+M)            1.0E+13   0.0      0.0",
        "  LOW / 5.0E+18 -1.0 0.0 /",
        "  H2O/5.0/",
        "CO + O (+M) <=> CO2 (+M)          2.0E+10   0.0   2500.0",
        "  LOW / 5.0E+14 0.0 3000.0 /",
        "  TROE / 0.6 100.0 1500.0 5000.0 /",
        "O + H2 (+N2) <=> H2O (+N2)        1.0E+12   0.0   1000.0",
        "  LOW / 1.0E+18 -1.0 0.0 /",
        "  TROE / 0.7 200.0 1000.0 /",
        "H + O (+M) <=> OH (+M)            5.0E+12   0.0      0.0",
        "  LOW / 4.0E+18 -1.0 0.0 /",
        "  SRI / 0.5 800.0 200.0 1.2 0.1 /",
        "CO + OH <=> CO2 + H               4.0E+06   1.5   -500.0",
        "OH + OH <=> O + H2O               3.0E+04   2.4  -2000.0",
        "  DUP",
        "OH + OH <=> O + H2O               1.0E+12   0.0   1000.0",
        "  DUP",
        "H2 + O2 <=> OH + OH               1.7E+13   0.0  47780.0",
        "CH4 + O2 <=> CO + H2O + H2        1.0E+12   0.0  30000.0",
        "CH4 + O <=> CO + 2 H2             1.0E+13   0.0   8000.0",
    ];
    let mut text: Vec<&str> =
        vec!["ELEMENTS H O C N END", "SPECIES H2 O2 H2O OH O H CO CO2 CH4 N2 END", "REACTIONS"];
    text.extend_from_slice(&rx);
    text.push("END");
    Mechanism::read_chemkin(&text.join("\n"), "kin-fixture", Some(&tm_set()))
        .unwrap_or_else(|e| panic!("the fixture does not read: {e}"))
}

/// Stoichiometric H2-air (2 : 1 : 3.76 by moles) with H, O and OH at
/// `trace` each, as mass fractions in the mechanism's species order.
fn h2_air(mech: &Mechanism, trace: f64) -> Vec<f64> {
    let set = tm_set();
    let xw: Vec<f64> = mech
        .species()
        .iter()
        .map(|s| {
            let x = match s.as_str() {
                "H2" => 2.0,
                "O2" => 1.0,
                "N2" => 3.76,
                "H" | "O" | "OH" => trace,
                _ => 0.0,
            };
            x * set.species()[set.index_of(s).expect("in set")].molar_mass
        })
        .collect();
    let sum: f64 = xw.iter().sum();
    xw.iter().map(|v| v / sum).collect()
}

/// Transient species controls at `dt`: Euler, no relaxation.
fn species_controls(dt: f64) -> TurbulenceControls {
    let mut c = TurbulenceControls {
        k_relax: 1.0,
        steady: false,
        delta_t: dt as Scalar,
        ddt: DdtScheme::Euler,
        sn_grad: SnGradScheme::Uncorrected,
        ..TurbulenceControls::default()
    };
    c.k_solver.report_residuals = false;
    c
}

/// The laminar transient `Energy`: Euler, tolerance 1e-14, relTol 0, no
/// relaxation, uncorrected snGrad.
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

/// Everything one coupled step touches, built on one mesh.
struct Rig<'m> {
    sp: Species<'m>,
    cs: ChemSource,
    se: SpeciesEnthalpy,
    e: Energy<'m>,
    gas: GasState<'m>,
    pf: DevBuf<Scalar>,
    cp_c: DevBuf<Scalar>,
    cp_b: DevBuf<Scalar>,
    w: DevBuf<Scalar>,
    q: DevBuf<Scalar>,
    qhd: DevBuf<Scalar>,
}

/// The set (mechanism order, inert N2), seeded with `y(k, c)`, `T` seeded
/// with `t(c)`; the energy registry metered; the density from `T`.
#[allow(clippy::too_many_arguments)]
fn rig<'m>(
    g: &Gpu,
    hm: &HostMesh,
    m: &'m GpuMesh,
    mech: &Mechanism,
    coeffs: &[SpeciesCoeffs],
    weighted: bool,
    dt: f64,
    y: impl Fn(usize, usize) -> f64,
    t: impl Fn(usize) -> f64,
) -> Rig<'m> {
    let n = hm.n_cells;
    let names: Vec<String> = mech.species().to_vec();
    let mut sp =
        Species::new(g, hm, m, &names, coeffs, "N2", 1.5e-5, species_controls(dt)).expect("species");
    if weighted {
        sp.use_mass_weighting(g).expect("mass weighting");
    }
    for (k, name) in names.iter().enumerate() {
        if name != "N2" {
            let v: Vec<Scalar> = (0..n).map(|c| y(k, c) as Scalar).collect();
            g.write(&mut sp.by_name_mut(name).expect("solved").field_mut().f, &v).expect("seed");
        }
    }
    sp.initialise(g).expect("initialise");
    let cs = ChemSource::new(g, mech, &mut sp, &ReactorOptions::default(), n).expect("chemistry");
    let se = SpeciesEnthalpy::new(g, m, mech.thermo(), &sp).expect("enthalpy table");
    let mut e = energy_on(g, m, dt);
    let tv: Vec<Scalar> = (0..n).map(|c| t(c) as Scalar).collect();
    g.write(&mut e.field_mut().f, &tv).expect("T");
    e.initialise(g).expect("T initialise");
    e.sources_mut().use_meters(g).expect("meters");
    let mut gas = GasState::new(g, m, GasProperties::default(), DomainKind::Open, P_ATM as Scalar)
        .expect("gas");
    gas.update_density(g, e.field()).expect("rho");
    let z = |len: usize| g.upload(&vec![0.0 as Scalar; len.max(1)]).expect("buffer");
    Rig {
        sp,
        cs,
        se,
        e,
        gas,
        pf: g.upload(&vec![P_ATM as Scalar; n]).expect("p"),
        cp_c: z(n),
        cp_b: z(m.n_boundary_faces),
        w: z(n),
        q: z(n),
        qhd: z(n),
    }
}

/// One coupled step in `ofgpu-lowmach`'s order (its `open_time_step` and
/// `outer_iteration`, with no flow solve): open the step, rho from `T`, the
/// mixture cp field from `(T°, Y°)`, the split chemistry step, its heat
/// release and the enthalpy diffusion registered, the energy equation, the
/// species equations. The step's `w` is left in `r.w`, its heat release in
/// `r.q` and its enthalpy diffusion in `r.qhd`.
fn step(g: &Gpu, m: &GpuMesh, r: &mut Rig<'_>, flow: &FlowState, nut: &GpuScalarField, dt: f64, s: usize) {
    if s > 0 {
        r.e.advance_time_levels(g).expect("T levels");
        r.e.advance_time_step(dt as Scalar);
        r.gas.advance_time_levels();
    }
    r.gas.update_density(g, r.e.field()).expect("rho");
    r.se.mixture_cp(g, &r.sp, r.e.field(), &mut r.cp_c, &mut r.cp_b).expect("mixture cp");
    r.e.set_cp_field(g, &r.cp_c, &r.cp_b).expect("cp field");
    let rho_arg = r.sp.is_mass_weighted().then_some(r.gas.rho());
    r.cs.step(g, &mut r.sp, &r.e.field().f, &r.pf, rho_arg, dt).expect("chemistry step");
    let w = r.e.heat_release_weight(g, &r.gas).expect("weight");
    g.stream().memcpy_dtod(w, &mut r.w).expect("w");
    r.cs.heat_release(g, &r.w, &mut r.q).expect("heat release");
    r.se.diffusion_source(g, m, &r.sp, r.e.field(), &r.gas.rho().f, &mut r.qhd).expect("Q_hd");
    let es: &mut EnergySources = r.e.sources_mut();
    es.clear(g).expect("clear");
    es.register_heat_release(g, &r.q).expect("register Q_chem");
    es.register_enthalpy_diffusion(g, &r.qhd).expect("register Q_hd");
    let phi = flow.phi;
    r.e.correct(g, phi, nut, &nut.f, 1.5e-5, &r.gas).expect("energy");
    let rho_arg = r.sp.is_mass_weighted().then_some(r.gas.rho());
    r.sp.correct_with_density(g, flow, nut, rho_arg).expect("species");
}

fn calculated_nut(g: &Gpu, m: &GpuMesh) -> GpuScalarField {
    let mut nut = GpuScalarField::zeros(g, m, "nut").expect("nut");
    g.write(&mut nut.bc_kind, &vec![BcKind::Calculated as Label; m.n_boundary_faces])
        .expect("nut kinds");
    nut
}

// ==========================================================================
//  energy_box_matches_reactor
// ==========================================================================

/// The flow step of the split, s (CHR-09's, §132.11).
const DT: f64 = 1.0e-6;
/// The steps the gate runs (CHR-09's reduced count).
const STEPS: usize = 200;

/// **SPEC-LIT 133.9, row P.** CHR-09's 0-D box - 2^3 cells, cyclic on all
/// three axes, uniform, no flow, open (`p0` held at 1 atm) - with `T` the
/// energy equation's: each step registers the split step's heat release
/// (133.16) and the (here zero) enthalpy diffusion (133.17) on
/// `EnergySources`, sets the mixture-cp field (133.18), and solves `T`. Over
/// 200 Lie steps at dt = 1e-6 s the energy equation's `T` stays on the host
/// twin `ConstPressureReactor::advance` chained over the same steps to
/// 0.5 K, on §19's constant-density species equation and on §86's
/// mass-weighted one.
#[test]
fn energy_box_matches_reactor() {
    let Some(g) = gpu() else { return };
    let hm = box_mesh(2, 0.01, true);
    let m = GpuMesh::upload(&g, &hm).expect("mesh");
    let mech = fixture();
    let ks = mech.species().len();
    let coeffs = vec![SpeciesCoeffs::default(); ks];
    let (t0, p) = (1400.0f64, P_ATM);
    let y0 = h2_air(&mech, 1.0e-4);

    let reactor = ConstPressureReactor::new(&mech, p).expect("reactor");
    let mut opts = ReactorOptions::default();
    let mut host_t = Vec::with_capacity(STEPS);
    let (mut y, mut t) = (y0.clone(), t0);
    for _ in 0..STEPS {
        let r = reactor.advance(&y, t, DT, &opts).expect("host advance");
        opts.h0 = r.stats.h_next;
        y = r.y;
        t = r.temperature;
        host_t.push(t);
    }
    println!("  [133.9] host reactor: T {t0} -> {t:.3} K over {STEPS} dt");
    assert!(t - t0 > 10.0, "the reactor barely heated ({} K): the gate would measure nothing", t - t0);

    let u = GpuVectorField::zeros(&g, &m, "U").expect("U");
    let phi = GpuSurfaceScalarField::zeros(&g, &m, "phi").expect("phi");
    let nut = calculated_nut(&g, &m);
    let flow = FlowState::new(&u, &phi, 1.5e-5);

    for weighted in [false, true] {
        let mut r = rig(&g, &hm, &m, &mech, &coeffs, weighted, DT, |k, _| y0[k], |_| t0);
        let (mut worst, mut at, mut hd_max) = (0.0f64, 0usize, 0.0f64);
        for (s, ht) in host_t.iter().enumerate() {
            step(&g, &m, &mut r, &flow, &nut, DT, s);
            for v in down(&g, &r.e.field().f) {
                assert!(v.is_finite(), "step {s}: T is not finite");
                if (v - ht).abs() > worst {
                    worst = (v - ht).abs();
                    at = s;
                }
            }
            hd_max = down(&g, &r.qhd).iter().fold(hd_max, |a, v| a.max(v.abs()));
        }
        assert!(r.cs.refused_cells(&g).expect("status").is_empty(), "a cell was refused");
        let leg = if weighted { "mass-weighted" } else { "constant-density" };
        println!(
            "  [133.9] box T vs reactor, {leg}: {worst:.3e} K (step {at}), max |Q_hd| {hd_max:.3e} W/m3 \
             (a uniform box has none)"
        );
        assert!(worst <= 0.5, "{leg}: the energy equation's T left the reactor by {worst} K at step {at}");
    }
}

// ==========================================================================
//  energy_budget_meters_heat_release
// ==========================================================================

/// **SPEC-LIT 133.9, row R.** A closed 4^3 box (adiabatic zero-gradient
/// walls, no flow), with `T` and `Y` varying from cell to cell, runs three
/// coupled steps at dt = 1e-5 s on the constant-density species equation
/// with five different diffusivities. At the end state `assembly_budget`:
/// (a) closes - its residual over the gross `Σ w T° V/dt` the rows hold;
/// (b) meters the heat release - `heat_release_w` against the host's
/// `-Σ w (T* - T°)/dt V` from the same `w`, `T°` and `T*`, relative;
/// (c) splits `sources_w` into the two meters, over the gross; and the
/// meters measure something: the heat release above 1e-4 of the gross and
/// the enthalpy diffusion nonzero.
#[test]
fn energy_budget_meters_heat_release() {
    let Some(g) = gpu() else { return };
    let hm = box_mesh(4, 0.004, false);
    let m = GpuMesh::upload(&g, &hm).expect("mesh");
    let n = hm.n_cells;
    let mech = fixture();
    let ks = mech.species().len();
    let coeffs: Vec<SpeciesCoeffs> = (0..ks)
        .map(|k| SpeciesCoeffs { d: (2.0e-5 * (1.0 + 0.3 * k as f64)) as Scalar, ..SpeciesCoeffs::default() })
        .collect();
    let base = h2_air(&mech, 1.0e-4);
    let h2 = mech.species().iter().position(|s| s == "H2").expect("H2");
    let oh = mech.species().iter().position(|s| s == "OH").expect("OH");
    let y = |k: usize, c: usize| {
        let wave = (c as f64 * 0.61).sin();
        if k == h2 {
            base[k] * (1.0 + 0.2 * wave)
        } else if k == oh {
            base[k] * (1.0 + 0.5 * (c as f64 * 0.37).cos())
        } else {
            base[k]
        }
    };
    let t = |c: usize| 1700.0 + 60.0 * (c as f64 * 0.43).sin();
    let dt = 1.0e-5;

    let u = GpuVectorField::zeros(&g, &m, "U").expect("U");
    let phi = GpuSurfaceScalarField::zeros(&g, &m, "phi").expect("phi");
    let nut = calculated_nut(&g, &m);
    let flow = FlowState::new(&u, &phi, 1.5e-5);
    let mut r = rig(&g, &hm, &m, &mech, &coeffs, false, dt, y, t);
    for s in 0..3 {
        step(&g, &m, &mut r, &flow, &nut, dt, s);
    }
    let b: EnergyBudget = r.e.assembly_budget(&g, &r.gas).expect("budget");

    let (w, t0, ts) = (down(&g, &r.w), down(&g, r.cs.start_temperature()), down(&g, r.cs.temperature()));
    let vol = down(&g, &m.v);
    let gross: f64 = (0..n).map(|c| w[c] * t0[c] * vol[c] / dt).sum();
    let host_hr: f64 = -(0..n).map(|c| w[c] * (ts[c] - t0[c]) / dt * vol[c]).sum::<f64>();
    let closes = b.residual_w.abs() / gross;
    let metered = (b.heat_release_w - host_hr).abs() / host_hr.abs();
    let split = (b.sources_w - b.heat_release_w - b.enthalpy_diffusion_w).abs() / gross;
    println!(
        "  [133.9] budget: ddt {:.6e} W, laplacian {:.6e} W, sources {:.6e} W = heat release \
         {:.6e} W + enthalpy diffusion {:.6e} W, residual {:.3e} W; gross {gross:.6e} W",
        b.ddt_w, b.laplacian_w, b.sources_w, b.heat_release_w, b.enthalpy_diffusion_w, b.residual_w
    );
    println!(
        "  [133.9] closes {closes:.3e}, heat release vs host {metered:.3e}, split {split:.3e} \
         (tolerance {:.1e}); heat release / gross {:.3e}",
        tol_r(1e-12),
        host_hr.abs() / gross
    );
    assert!(r.cs.refused_cells(&g).expect("status").is_empty(), "a cell was refused");
    assert!(closes <= tol_r(1e-12), "the budget does not close: {closes:e} of the gross");
    assert!(metered <= tol_r(1e-12), "the heat-release meter is not the step's heat: {metered:e}");
    assert!(split <= tol_r(1e-12), "the meters are not a split of sources_w: {split:e}");
    assert!(host_hr.abs() > 1e-4 * gross, "the heat release is too small to meter ({host_hr:e} W)");
    assert!(b.enthalpy_diffusion_w != 0.0, "the enthalpy diffusion is zero: (b) would be vacuous");
}

// ==========================================================================
//  energy_species_enthalpy_matches_host
// ==========================================================================

/// **SPEC-LIT 133.9, row R.** On a 4^3 walled box with `T` and `Y`
/// varying, (133.18)'s mixture `cp` on every cell and boundary face equals
/// the host `ThermoSet::cp_mass` of the same downloaded state, and
/// (133.17)'s `Q_hd` equals an f64 host evaluation from the device's own
/// Gauss gradients, the TM-4513 `cp_k` and the set's `D_k`, both relative
/// to the largest value. And the refusals are named.
#[test]
fn energy_species_enthalpy_matches_host() {
    let Some(g) = gpu() else { return };
    let hm = box_mesh(4, 0.004, false);
    let m = GpuMesh::upload(&g, &hm).expect("mesh");
    let (n, nb) = (hm.n_cells, hm.n_boundary_faces);
    let mech = fixture();
    let thermo = mech.thermo();
    let names: Vec<String> = mech.species().to_vec();
    let ks = names.len();
    let coeffs: Vec<SpeciesCoeffs> = (0..ks)
        .map(|k| SpeciesCoeffs { d: (1.5e-5 * (1.0 + 0.4 * k as f64)) as Scalar, ..SpeciesCoeffs::default() })
        .collect();
    let base = h2_air(&mech, 1.0e-3);
    let y = |k: usize, c: usize| base[k] * (1.0 + 0.3 * ((c * (k + 1)) as f64 * 0.29).sin());
    let mut r = rig(&g, &hm, &m, &mech, &coeffs, false, 1e-6, y, |c| 900.0 + 15.0 * c as f64);

    // (133.18) against ThermoSet::cp_mass, cells and boundary faces.
    r.se.mixture_cp(&g, &r.sp, r.e.field(), &mut r.cp_c, &mut r.cp_b).expect("mixture cp");
    let member = |k: usize| -> (Vec<f64>, Vec<f64>) {
        let f = if names[k] == "N2" { r.sp.inert() } else { r.sp.by_name(&names[k]).expect("m").field() };
        (down(&g, &f.f), down(&g, &f.bf))
    };
    let ys: Vec<(Vec<f64>, Vec<f64>)> = (0..ks).map(member).collect();
    let (tc, tb) = (down(&g, &r.e.field().f), down(&g, &r.e.field().bf));
    let order: Vec<usize> = thermo.species().iter().map(|s| names.iter().position(|nm| nm == &s.name).unwrap()).collect();
    let host_cp = |t: f64, at: &dyn Fn(usize) -> f64| {
        let yv: Vec<f64> = order.iter().map(|&k| at(k)).collect();
        thermo.cp_mass(t, &yv).expect("host cp")
    };
    let (dc, db) = (down(&g, &r.cp_c), down(&g, &r.cp_b));
    let mut cp_err = 0.0f64;
    for c in 0..n {
        let h = host_cp(tc[c], &|k| ys[k].0[c]);
        cp_err = cp_err.max((dc[c] - h).abs() / h);
    }
    for f in 0..nb {
        let h = host_cp(tb[f], &|k| ys[k].1[f]);
        cp_err = cp_err.max((db[f] - h).abs() / h);
    }

    // (133.17) against the host, on the device's own gradients.
    r.se.diffusion_source(&g, &m, &r.sp, r.e.field(), &r.gas.rho().f, &mut r.qhd).expect("Q_hd");
    let fvk = FvKernels::new(&g).expect("fv");
    let mut gbuf: DevBuf<Vec3> = g.zeros(n).expect("grad");
    let mut grad = |f: &GpuScalarField| -> Vec<Vec3> {
        fvc_grad_scalar(&g, &fvk, &mut gbuf, f, &m).expect("grad");
        g.download(&gbuf).expect("grad")
    };
    let gt = grad(r.e.field());
    let rho = down(&g, &r.gas.rho().f);
    let cp_of = |name: &str, t: f64| {
        let sp = thermo.get(name).expect("thermo");
        sp.cp(t).expect("cp") / sp.molar_mass
    };
    let mut host = vec![0.0f64; n];
    for (i, name) in r.sp.names().to_vec().iter().enumerate() {
        let gy = grad(r.sp.get(i).expect("solved").field());
        let d = w64(r.sp.diffusivity(i).expect("D"));
        for c in 0..n {
            let dot = w64(gy[c].x) * w64(gt[c].x) + w64(gy[c].y) * w64(gt[c].y) + w64(gy[c].z) * w64(gt[c].z);
            host[c] += rho[c] * d * (cp_of(name, tc[c]) - cp_of("N2", tc[c])) * dot;
        }
    }
    let dev = down(&g, &r.qhd);
    let scale = host.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let hd_err = (0..n).map(|c| (dev[c] - host[c]).abs()).fold(0.0, f64::max) / scale;
    println!(
        "  [133.9] mixture cp vs host {cp_err:.3e}; Q_hd vs host {hd_err:.3e} of max |Q_hd| \
         {scale:.3e} W/m3 (tolerance {:.1e})",
        tol_r(1e-12)
    );
    assert!(cp_err <= tol_r(1e-12), "(133.18) left the host by {cp_err:e}");
    assert!(scale > 1.0, "Q_hd is too small to measure ({scale:e} W/m3)");
    assert!(hd_err <= tol_r(1e-12), "(133.17) left the host by {hd_err:e}");

    // The refusals, by name.
    let mut bad = names.clone();
    let j = bad.iter().position(|s| s == "CH4").expect("CH4");
    bad[j] = "XX".into();
    let sp_bad = Species::new(&g, &hm, &m, &bad, &coeffs, "N2", 1.5e-5, species_controls(1e-6)).expect("set");
    let e1 = SpeciesEnthalpy::new(&g, &m, thermo, &sp_bad).err().expect("refused").to_string();
    assert!(e1.contains("\"XX\"") && e1.contains("133.9"), "{e1}");
    let mut sp2 = Species::new(&g, &hm, &m, &names, &coeffs, "N2", 1.5e-5, species_controls(1e-6)).expect("set");
    let cs2 = ChemSource::new(&g, &mech, &mut sp2, &ReactorOptions::default(), n).expect("chemistry");
    let e2 = cs2.heat_release(&g, &r.w, &mut r.q).err().expect("refused").to_string();
    assert!(e2.contains("before any split step") && e2.contains("133.9"), "{e2}");
    let mut bare = EnergySources::new(&g, &m).expect("registry");
    let e3 = bare.register_heat_release(&g, &r.q).err().expect("refused").to_string();
    assert!(e3.contains("use_meters") && e3.contains("register_heat_release"), "{e3}");
    println!("  [133.9] refusals: {e1} | {e2} | {e3}");
}
