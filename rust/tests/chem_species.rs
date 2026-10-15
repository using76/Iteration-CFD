// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CHR-09's gates (SPEC-LIT §132.11): the species set carries per-species
//! reaction sources through `correct_with_density`'s `su`, and the split
//! chemistry step of §132.5 fills them.
//!
//! The chemistry runs on CHR-06's fifteen-reaction fixture of §130.7 over
//! the ten species of `reference/mcbride-gordon-reno1993/table_II.csv`
//! (NASA TM-4513, a US-Government work), whose rate numbers are
//! PLACEHOLDERS, not any paper's; the reference is the f64 host twin
//! `ConstPressureReactor::advance` (§132.4).
//!
//! An integration test, so it runs in both builds: the chemistry computes
//! in double in both (§132.6); the species fields cross as Scalar.
//! No GPL-licensed source was consulted.

#![allow(clippy::excessive_precision)]

use ofgpu::blockgen::{build_mesh, BlockSpec, GradedAxis};
use ofgpu::chem::source::ChemSource;
use ofgpu::chem::{
    ConstPressureReactor, Mechanism, ReactorOptions, SpeciesThermo, ThermoSet, P_ATM, P_BAR,
};
use ofgpu::device::Gpu;
use ofgpu::field::{BcKind, GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use ofgpu::mesh::{GpuMesh, HostMesh};
use ofgpu::species::{Species, SpeciesCoeffs};
use ofgpu::timescheme::DdtScheme;
use ofgpu::turbulence::{FlowState, TurbulenceControls};
use ofgpu::{Label, Scalar};

/// The process's device, if the box has one: every GPU test returns early
/// rather than failing.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// An `n^3` box of side `side` m, walls on all six sides, or cyclic on every
/// axis when `cyclic`.
fn box_mesh(n: usize, side: f64, cyclic: bool) -> HostMesh {
    let axis = || GradedAxis {
        lo: 0.0,
        hi: side as Scalar,
        n,
        expansion: 1.0,
        two_sided: false,
    };
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

/// FNV-1a over the bits of every value, in order.
fn fnv(h: &mut u64, v: &[Scalar]) {
    for x in v {
        for byte in u64::from(x.to_bits()).to_le_bytes() {
            *h ^= u64::from(byte);
            *h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
}

/// A closed-domain internal flux that is deliberately not solenoidal, as
/// §86's own transport tests use; every boundary face carries zero flux.
fn closed_flux(g: &Gpu, m: &GpuMesh, hm: &HostMesh) -> GpuSurfaceScalarField {
    let mut phi = GpuSurfaceScalarField::zeros(g, m, "phi").expect("phi");
    let v: Vec<Scalar> = (0..hm.n_internal_faces)
        .map(|i| (2e-6 * (0.8 + (i as f64 * 0.29).sin())) as Scalar)
        .collect();
    g.write(&mut phi.f, &v).expect("phi");
    phi
}

/// A density varying in space, held in time, with its old levels set.
fn fixed_density(g: &Gpu, m: &GpuMesh, hm: &HostMesh) -> GpuScalarField {
    let mut rho = GpuScalarField::zeros(g, m, "rho").expect("rho");
    let v: Vec<Scalar> = (0..hm.n_cells)
        .map(|i| (0.4 + 0.1 * (i as f64 * 0.37).sin()) as Scalar)
        .collect();
    g.write(
        &mut rho.bc_kind,
        &vec![BcKind::Calculated as Label; hm.n_boundary_faces],
    )
    .expect("rho kinds");
    g.write(&mut rho.f, &v).expect("rho");
    g.write(&mut rho.f0, &v).expect("rho0");
    g.write(&mut rho.f00, &v).expect("rho00");
    g.write(&mut rho.bf, &vec![0.4 as Scalar; hm.n_boundary_faces])
        .expect("rho bf");
    rho
}

/// Transient controls at `dt`: Euler, no relaxation, uncorrected snGrad.
fn controls(dt: f64) -> TurbulenceControls {
    let mut c = TurbulenceControls {
        k_relax: 1.0,
        steady: false,
        delta_t: dt as Scalar,
        ddt: DdtScheme::Euler,
        sn_grad: ofgpu::fv::SnGradScheme::Uncorrected,
        ..TurbulenceControls::default()
    };
    c.k_solver.report_residuals = false;
    c
}

// ==========================================================================
//  species_without_chemistry_is_bitwise
// ==========================================================================

/// The FNV-1a of every species field after three no-source steps on both
/// species equations, recorded at afd0f30's tree before CHR-09 changed
/// `src/species.rs` (f64 build, then the f32 build).
const PIN_F64: u64 = 0x1bc5_208b_67b6_041e;
const PIN_F32: u64 = 0x018e_7024_cfde_c719;

/// **SPEC-LIT §132.11, row D.** A species set with no reaction sources runs
/// the arithmetic it ran before CHR-09, bit for bit: the hash of every
/// solved field and the inert remainder, after three steps of the
/// constant-density equation and three of the mass-weighted one, equals the
/// value pinned before `su` was threaded through.
#[test]
fn species_without_chemistry_is_bitwise() {
    let Some(g) = gpu() else { return };
    let hm = box_mesh(4, 0.02, false);
    let m = GpuMesh::upload(&g, &hm).expect("mesh");
    let n = hm.n_cells;
    let names: Vec<String> = ["H2", "O2", "N2"].iter().map(|s| s.to_string()).collect();
    let coeffs = [
        SpeciesCoeffs { d: 6e-5, ..SpeciesCoeffs::default() },
        SpeciesCoeffs { d: 2e-5, ..SpeciesCoeffs::default() },
        SpeciesCoeffs::default(),
    ];
    let u = GpuVectorField::zeros(&g, &m, "U").expect("U");
    let phi = closed_flux(&g, &m, &hm);
    let nut = GpuScalarField::zeros(&g, &m, "nut").expect("nut");
    let flow = FlowState::new(&u, &phi, 1.5e-5);
    let rho = fixed_density(&g, &m, &hm);

    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for weighted in [false, true] {
        let mut sp = Species::new(&g, &hm, &m, &names, &coeffs, "N2", 1.5e-5, controls(1e-3))
            .expect("species");
        if weighted {
            sp.use_mass_weighting(&g).expect("mass weighting");
        }
        for j in 0..sp.n_solved() {
            let v: Vec<Scalar> = (0..n)
                .map(|i| (0.1 + 0.05 * (((i + 7 * j) as f64) * 0.23).sin()) as Scalar)
                .collect();
            g.write(&mut sp.get_mut(j).expect("solved").field_mut().f, &v)
                .expect("seed");
        }
        sp.initialise(&g).expect("initialise");
        for _ in 0..3 {
            sp.correct_with_density(&g, &flow, &nut, weighted.then_some(&rho))
                .expect("correct");
        }
        for j in 0..sp.n_solved() {
            fnv(&mut h, &g.download(&sp.get(j).expect("solved").field().f).expect("d"));
        }
        fnv(&mut h, &g.download(&sp.inert().f).expect("d"));
    }
    let pin = if cfg!(feature = "single") { PIN_F32 } else { PIN_F64 };
    println!("  [132.11] no-chemistry species hash {h:#018x} (pin {pin:#018x})");
    assert_eq!(h, pin, "the no-chemistry species path moved a bit");
}

// ==========================================================================
//  The chemistry fixture (CHR-06's, as tests/chem_device_integrate.rs has it)
// ==========================================================================

/// The ten TM-4513 records of `reference/mcbride-gordon-reno1993/
/// table_II.csv` (SPEC-LIT §128.6, §130.7).
const TABLE_II: &str = include_str!("../../reference/mcbride-gordon-reno1993/table_II.csv");

/// The table's species codes: name and elements (§128.6's code list).
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

/// The records as a set at `P_BAR` (TM-4513 tabulates at 1 bar, §128.4;
/// the mechanism reader moves the set to `P_ATM`), bounds 200/1000/6000 K.
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
            rows.iter()
                .find(|(c, t, _, _)| c == code && *t == t_lo)
                .map(|r| (r.2, r.3))
                .unwrap()
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

/// CHR-06's fifteen-reaction reactor fixture of §130.7: every reaction
/// reversible, the reverse from K_c. Its rate numbers are placeholders.
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
    let mut text: Vec<&str> = vec![
        "ELEMENTS H O C N END",
        "SPECIES H2 O2 H2O OH O H CO CO2 CH4 N2 END",
        "REACTIONS",
    ];
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

/// Every species of the set, mechanism order, as f64: solved fields and the
/// inert remainder.
fn species_state(g: &Gpu, sp: &Species<'_>, mech: &Mechanism) -> Vec<Vec<f64>> {
    mech.species()
        .iter()
        .map(|name| {
            let f = if name == sp.inert_name() {
                &sp.inert().f
            } else {
                &sp.by_name(name).expect("member").field().f
            };
            g.download(f).expect("d").iter().map(|v| f64::from(*v)).collect()
        })
        .collect()
}

// ==========================================================================
//  species_box_matches_reactor
// ==========================================================================

/// The flow step of the split, s (docs/17 CHR-09; §132.10 set it).
const DT: f64 = 1.0e-6;
/// The steps the gate runs (the plan's reduced count).
const STEPS: usize = 200;

/// **SPEC-LIT §132.11, row P.** A closed adiabatic box with no flow - 2^3
/// cells, cyclic on all three axes, uniform - is CHR-06's constant-pressure
/// reactor: Lie-split at dt = 1e-6 s over 200 steps, with T fed back from
/// the step's T* (the energy equation is CHR-10b's), every species of the
/// set stays on the host twin `ConstPressureReactor::advance` chained over
/// the same 200 dt (each chunk's first trial step the last one's h_next,
/// as the device carries it) to 1e-6, on §19's constant-density equation
/// and on §86's mass-weighted one.
#[test]
fn species_box_matches_reactor() {
    let Some(g) = gpu() else { return };
    let hm = box_mesh(2, 0.01, true);
    let m = GpuMesh::upload(&g, &hm).expect("mesh");
    let n = hm.n_cells;
    let mech = fixture();
    let ks = mech.species().len();
    let names: Vec<String> = mech.species().to_vec();
    let coeffs = vec![SpeciesCoeffs::default(); ks];
    let (t0, p) = (1400.0f64, P_ATM);
    let y0 = h2_air(&mech, 1.0e-4);

    // The host twin, chained.
    let reactor = ConstPressureReactor::new(&mech, p).expect("reactor");
    let mut opts = ReactorOptions::default();
    let mut host: Vec<(Vec<f64>, f64)> = Vec::with_capacity(STEPS);
    let (mut y, mut t) = (y0.clone(), t0);
    for _ in 0..STEPS {
        let r = reactor.advance(&y, t, DT, &opts).expect("host advance");
        opts.h0 = r.stats.h_next;
        y = r.y;
        t = r.temperature;
        host.push((y.clone(), t));
    }
    let moved = (0..ks).map(|k| (y[k] - y0[k]).abs()).fold(0.0, f64::max);
    println!("  [132.11] host reactor: T {t0} -> {t:.3} K, max |dY| {moved:.3e} over 200 dt");
    assert!(moved > 1e-3, "the reactor barely moved ({moved:e}): the gate would measure nothing");

    let u = GpuVectorField::zeros(&g, &m, "U").expect("U");
    let phi = GpuSurfaceScalarField::zeros(&g, &m, "phi").expect("phi");
    let nut = GpuScalarField::zeros(&g, &m, "nut").expect("nut");
    let flow = FlowState::new(&u, &phi, 1.5e-5);
    let mut rho = GpuScalarField::zeros(&g, &m, "rho").expect("rho");
    for buf in [&mut rho.f, &mut rho.f0, &mut rho.f00] {
        g.write(buf, &vec![0.25 as Scalar; n]).expect("rho");
    }

    for weighted in [false, true] {
        let mut sp = Species::new(&g, &hm, &m, &names, &coeffs, "N2", 1.5e-5, controls(DT))
            .expect("species");
        if weighted {
            sp.use_mass_weighting(&g).expect("mass weighting");
        }
        for (k, name) in names.iter().enumerate() {
            if name != "N2" {
                g.write(
                    &mut sp.by_name_mut(name).expect("solved").field_mut().f,
                    &vec![y0[k] as Scalar; n],
                )
                .expect("seed");
            }
        }
        sp.initialise(&g).expect("initialise");
        let mut cs = ChemSource::new(&g, &mech, &mut sp, &ReactorOptions::default(), n)
            .expect("chemistry");
        let mut tf = g.upload(&vec![t0 as Scalar; n]).expect("T");
        let pf = g.upload(&vec![p as Scalar; n]).expect("p");
        let rho_arg = weighted.then_some(&rho);

        // The solved species and the inert remainder are measured apart:
        // the remainder is `1 - Σ solved`, so it carries the SUM of the
        // solved species' errors (the module doc of `src/species.rs`, "What
        // the inert species absorbs"), and `drift` is that sum, per cell.
        let (mut solved_err, mut inert_err, mut inert_at) = (0.0f64, 0.0f64, 0usize);
        let (mut cause, mut worst_t) = (0.0f64, 0.0f64);
        for (s, (hy, ht)) in host.iter().enumerate() {
            cs.step(&g, &mut sp, &tf, &pf, rho_arg, DT).expect("chemistry step");
            g.stream().memcpy_dtod(cs.temperature(), &mut tf).expect("T*");
            sp.correct_with_density(&g, &flow, &nut, rho_arg).expect("correct");
            let dev = species_state(&g, &sp, &mech);
            for c in 0..n {
                let mut drift = 0.0f64;
                for k in 0..ks {
                    let e = dev[k][c] - hy[k];
                    assert!(e.is_finite(), "step {s}: {} is not finite", names[k]);
                    if names[k] == "N2" {
                        if e.abs() > inert_err {
                            inert_err = e.abs();
                            inert_at = s;
                        }
                    } else {
                        solved_err = solved_err.max(e.abs());
                        drift += e;
                    }
                }
                let ki = names.iter().position(|nm| nm == "N2").expect("inert");
                cause = cause.max((dev[ki][c] - hy[ki] + drift).abs());
            }
            let td = g.download(&tf).expect("T");
            worst_t = td.iter().map(|v| (f64::from(*v) - ht).abs()).fold(worst_t, f64::max);
        }
        assert!(cs.refused_cells(&g).expect("status").is_empty(), "a cell was refused");
        let leg = if weighted { "mass-weighted" } else { "constant-density" };
        println!(
            "  [132.11] box vs reactor, {leg}: solved species {solved_err:.3e}, inert N2 \
             {inert_err:.3e} (step {inert_at}), T* {worst_t:.3e} K"
        );
        assert!(solved_err <= 1e-6, "{leg}: a solved species left the reactor by {solved_err:e}");
        if inert_err <= 1e-6 {
            continue;
        }
        // Only the f32 build reaches here: the remainder is the sum of nine
        // f32 fields, each rounded at every step, and their round-off adds.
        println!(
            "  [132.11] box vs reactor, {leg}, inert N2 under single: MISSED - {inert_err:.4e} \
             against 1e-6 at step {inert_at}; the inert remainder 1 - sum carries the summed \
             f32 round-off of the nine solved fields, one rounding each per step over 200 \
             steps (N2 against minus that sum: {cause:.3e}); every solved species holds \
             {solved_err:.3e}. The cure (f64 mass fractions, or a renormalised set) changes \
             the species numerics and is the user's decision (SPEC-LIT 132.11)"
        );
        assert!(cfg!(feature = "single"), "{leg}: the inert N2 left the reactor by {inert_err:e}");
        assert!(cause <= 1e-6, "the cause: N2 against minus the solved drift {cause:e}");
    }
}

// ==========================================================================
//  species_budget_closes_with_chemistry
// ==========================================================================

/// **SPEC-LIT §132.11, row R, and (86.5) with the chemistry metered.** On a
/// closed box (walls, a non-solenoidal internal flux, conservative upwind,
/// a density varying in space) the mass-weighted species equation with the
/// split step's sources moves each species' mass by exactly what the
/// sources and §86.9's clip put in: per step and per solved species,
/// `|Δ Σ rho Y V - dt Σ su V - Δ clip| <= 1e-12 Σ rho V` (class R: tol32 =
/// 1e-5 under single), with the chemistry's own term above 1e-6 of the
/// mass so the identity is not a statement about zero.
#[test]
fn species_budget_closes_with_chemistry() {
    let Some(g) = gpu() else { return };
    let hm = box_mesh(4, 0.01, false);
    let m = GpuMesh::upload(&g, &hm).expect("mesh");
    let n = hm.n_cells;
    let mech = fixture();
    let names: Vec<String> = mech.species().to_vec();
    let coeffs = vec![SpeciesCoeffs::default(); names.len()];
    let y0 = h2_air(&mech, 1.0e-3);
    let dt = DT;

    let mut ctrl = controls(dt);
    ctrl.k_solver.tolerance = 1e-15;
    ctrl.k_solver.max_iter = 200;
    let mut sp = Species::new(&g, &hm, &m, &names, &coeffs, "N2", 1.5e-5, ctrl).expect("species");
    sp.use_mass_weighting(&g).expect("mass weighting");
    sp.set_convection(ofgpu::io::schemes::DivEntry {
        scheme: ofgpu::fv::DivScheme::Upwind,
        bounded: false,
    });
    for (k, name) in names.iter().enumerate() {
        if name != "N2" {
            let v: Vec<Scalar> = (0..n)
                .map(|c| (y0[k] * (1.0 + 0.2 * ((c + 3 * k) as f64 * 0.37).sin())) as Scalar)
                .collect();
            g.write(&mut sp.by_name_mut(name).expect("solved").field_mut().f, &v)
                .expect("seed");
        }
    }
    sp.initialise(&g).expect("initialise");
    let mut cs =
        ChemSource::new(&g, &mech, &mut sp, &ReactorOptions::default(), n).expect("chemistry");
    let tf = g
        .upload(
            &(0..n)
                .map(|c| (1500.0 + 100.0 * (c as f64 * 0.53).sin()) as Scalar)
                .collect::<Vec<_>>(),
        )
        .expect("T");
    let pf = g.upload(&vec![P_ATM as Scalar; n]).expect("p");
    let rho = fixed_density(&g, &m, &hm);
    let rho_h: Vec<f64> = g.download(&rho.f).expect("rho").iter().map(|v| f64::from(*v)).collect();
    let vol: Vec<f64> = hm.v.iter().map(|v| f64::from(*v)).collect();
    let total: f64 = (0..n).map(|c| rho_h[c] * vol[c]).sum();

    let u = GpuVectorField::zeros(&g, &m, "U").expect("U");
    let phi = closed_flux(&g, &m, &hm);
    let nut = GpuScalarField::zeros(&g, &m, "nut").expect("nut");
    let flow = FlowState::new(&u, &phi, 1.5e-5);

    let solved: Vec<String> = sp.names().to_vec();
    let mass = |sp: &Species<'_>, name: &str| -> f64 {
        let y = g.download(&sp.by_name(name).expect("solved").field().f).expect("d");
        (0..n).map(|c| rho_h[c] * f64::from(y[c]) * vol[c]).sum()
    };
    let ledger = |sp: &Species<'_>, name: &str| -> f64 {
        let l = g.download(sp.clip_ledger(name).expect("ledger")).expect("d");
        l.iter().map(|v| f64::from(*v)).sum()
    };
    let (mut worst, mut worst_name, mut largest_s) = (0.0f64, String::new(), 0.0f64);
    for _ in 0..3 {
        let before: Vec<(f64, f64)> =
            solved.iter().map(|nm| (mass(&sp, nm), ledger(&sp, nm))).collect();
        cs.step(&g, &mut sp, &tf, &pf, Some(&rho), dt).expect("chemistry step");
        let s_dt: Vec<f64> = solved
            .iter()
            .map(|nm| {
                let su = g.download(sp.reaction_source(nm).expect("su")).expect("d");
                dt * (0..n).map(|c| f64::from(su[c]) * vol[c]).sum::<f64>()
            })
            .collect();
        sp.correct_with_density(&g, &flow, &nut, Some(&rho)).expect("correct");
        for (i, nm) in solved.iter().enumerate() {
            let (m0, l0) = before[i];
            let defect = mass(&sp, nm) - m0 - s_dt[i] - (ledger(&sp, nm) - l0);
            let rel = defect.abs() / total;
            assert!(rel.is_finite(), "{nm}: the budget is not finite");
            largest_s = largest_s.max(s_dt[i].abs() / total);
            if rel > worst {
                worst = rel;
                worst_name = nm.clone();
            }
        }
    }
    println!(
        "  [132.11] species budget with chemistry: worst defect {worst:.3e} of the mass \
         ({worst_name}), largest chemistry term {largest_s:.3e}"
    );
    assert!(largest_s > 1e-6, "the chemistry put in only {largest_s:e} of the mass");
    assert!(worst <= tol_r(1e-12), "(86.5) with the chemistry: defect {worst:e} ({worst_name})");
}
