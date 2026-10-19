// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §137.10's gates for `ofgpu-lowmach`'s parcels step (CHR-26), in
//! both builds: the 25-step evaporating pool through the case against the
//! library calls made by hand (row D, bitwise), the vapour's two entries
//! into a reacting step - (137.8) into the named species' reaction source
//! and (137.9) into the constraint's target divergence (rows D and R) -
//! Gate 68-B through the whole driver (row D), the coupled spray's (68.4)
//! momentum balance (VAL-72a, row R), and the refusals by name (row D).
//!
//! One fixture deviation, recorded: the flow cases solve `p` with PCG, not
//! [`NUMERICS`]'s PBiCGStab. The all-wall box leaves the pressure system
//! Neumann-singular, and in the single build PBiCGStab's biorthogonal
//! iterates walk the null mode until the run breaks (`|p| res` stuck at
//! 7.6e-2, then `T [inf, -inf]`), where PCG holds 1.1e-6. Tolerances are
//! untouched.
//!
//! Written from SPEC-LIT §68, §76, §77 and §137.10, over the CHR-06
//! chemistry fixture of `lowmach_chem::tests`. No GPL-licensed source was
//! consulted.

use ofgpu::energy::{
    DomainKind, Energy, EnergyControls, EnergySources, GasMixture, GasProperties, GasState,
};
use ofgpu::field::{GpuScalarField, GpuVectorField};
use ofgpu::field_setup::{setup_scalar_field_with, BcInputs};
use ofgpu::io::case_json::read_case_jsonc;
use ofgpu::momentum::MomentumSources;
use ofgpu::parcels::couple::{
    CouplingControls, CouplingMode, CouplingSnapshot, MassCoupling, ParcelCoupling,
};
use ofgpu::parcels::deposit::ParcelDeposition;
use ofgpu::parcels::{
    DragModel, Injector, ParcelControls, ParcelPhysics, ParcelSnapshot, Parcels,
};
use ofgpu::timescheme::DdtScheme;
use ofgpu::{DVec3, Error, Gpu, GpuMesh, Scalar, Vec3};

use super::{refuse_settings, vapour_member, CaseParcels, ParcelRunShape};
use crate::common::CaseNumerics;
use crate::lowmach_chem::tests::{run_case, write_case, NAMES, NUMERICS};
use crate::lowmach_chem::{Reacting, VapourSource};

// ==========================================================================
//  The cases
// ==========================================================================

/// The reacting blocks of `evap_case`: the fixture species over the TM-4513
/// thermo, air's O2 with H2O at 30 % rh and every other solved member dry.
const EVAP_CHEM: &str = r#",
  "species": {
    "names": ["H2", "O2", "H2O", "OH", "O", "H", "CO", "CO2", "CH4", "N2"],
    "inert": "N2",
    "initial": { "H2": 0.0, "O2": 0.2287, "H2O": 5.86e-3, "OH": 0.0, "O": 0.0,
        "H": 0.0, "CO": 0.0, "CO2": 0.0, "CH4": 0.0 },
    "patches": []
  },
  "chemistry": { "mechanism": "mech.ck", "thermo": "therm.dat", "rtol": 1e-6 },
  "combustion": { "model": "laminar" }"#;

/// `evap_case`'s coupling: the vapour goes to H2O.
const EVAP_COUPLING: &str = r#"{ "momentum": "semiImplicit", "energy": "explicit",
    "mass": "evaporation", "vapour": "H2O" }"#;

/// A 500 um water droplet in still 25 C air at 30 % rh (gate 76-A's shape)
/// over the chem box's fluid, every boundary a wall. `chem` carries the
/// reacting blocks and `coupling` the coupling object, so the steady refusal
/// can run the same case with neither ahead of the parcels one.
pub(crate) fn evap_with(run: &str, chem: &str, coupling: &str) -> String {
    let head = format!(
        r#"{{
  "name": "parcelsEvap",
  "mesh": {{
    "kind": "cartesian",
    "bounds": {{ "min": [0, 0, 0], "max": [0.04, 0.04, 0.04] }},
    "cells": [4, 4, 4],
    "boundaries": {{ "xmin": "x0", "xmax": "x1", "ymin": "y0", "ymax": "y1", "zmin": "z0", "zmax": "z1" }}
  }},
  "physics": {{
    "gravity": [0, 0, -9.81],
    "fluid": {{ "nu": 1.5e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 293.15 }},
    "buoyancy": "densityRatio"
  }},
  "patches": [
    {{
      "match": ".*", "kind": "wall",
      "U": {{ "type": "fixedValue", "value": [0, 0, 0] }},
      "p": {{ "type": "zeroGradient" }},
      "T": {{ "type": "zeroGradient" }}
    }}
  ],
  "initial": {{ "U": [0, 0, 0], "T": 298.15, "p": 0.0 }},{NUMERICS},"#
    );
    let tail = format!(
        r#"
  "run": {run},
  "parcels": {{
    "capacity": 256, "physics": "evaporating", "drag": "schillerNaumann",
    "coupling": {coupling},
    "injectors": [
      {{
        "position": [0.02, 0.02, 0.03], "axis": [0, 0, -1], "speed": 0.5,
        "diameter": 5e-4, "temperature": 285.0, "massFlow": 1e-5,
        "parcelsPerEvent": 2, "interval": 0.005
      }}
    ]
  }}{chem}
}}"#
    );
    format!("{head}{tail}")
}

pub(crate) fn evap_case(run: &str) -> String {
    evap_with(run, EVAP_CHEM, EVAP_COUPLING)
}

/// `evap_case` for the driver's steady refusal: no reacting blocks to be
/// refused ahead of the parcels one, mass coupling none, and the steady
/// numerics - so `refuse_settings`'s own "parcels" line is what comes back.
fn evap_case_steady() -> String {
    evap_with(
        r#"{ "endTime": 0.025, "deltaT": 1e-3 }"#,
        "",
        r#"{ "momentum": "semiImplicit", "energy": "explicit", "mass": "none" }"#,
    )
    .replace(r#""ddt": "Euler""#, r#""ddt": "steadyState""#)
    .replace(r#"{ "kind": "PIMPLE", "correctors": 2 }"#, r#"{ "kind": "SIMPLE" }"#)
}

/// `flow_case`'s parcels: the heating model over the box's centre, coupling
/// as stated, one injector whose interval is every step.
fn flow_parcels(momentum: &str, mass_flow: &str) -> String {
    format!(
        r#",
  "parcels": {{
    "capacity": 64, "physics": "heating",
    "coupling": {{ "momentum": "{momentum}", "energy": "explicit" }},
    "injectors": [
      {{
        "position": [0.05, 0.05, 0.05], "axis": [1, 0, 0], "speed": 1.0,
        "diameter": 1e-4, "temperature": 290.0, "massFlow": {mass_flow},
        "parcelsPerEvent": 4
      }}
    ]
  }}"#
    )
}

/// [`NUMERICS`] with the pressure solver on PCG - the flow fixtures' one
/// deviation, recorded in this file's header.
fn flow_numerics() -> String {
    NUMERICS.replace(
        r#""match": "p", "solver": "PBiCGStab""#,
        r#""match": "p", "solver": "PCG""#,
    )
}

/// The non-reacting box: drifting gas over a heated, sealed, all-wall box.
/// `parcels` is `""` or the [`flow_parcels`] fragment.
fn flow_case(parcels: &str) -> String {
    format!(
        r#"{{
  "name": "parcelsFlow",
  "mesh": {{
    "kind": "cartesian",
    "bounds": {{ "min": [0, 0, 0], "max": [0.1, 0.1, 0.1] }},
    "cells": [4, 4, 4],
    "boundaries": {{ "xmin": "x0", "xmax": "x1", "ymin": "y0", "ymax": "y1", "zmin": "z0", "zmax": "z1" }}
  }},
  "physics": {{
    "gravity": [0.3, -9.81, 0.2],
    "fluid": {{ "nu": 1.5e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 293.15 }},
    "buoyancy": "densityRatio"
  }},
  "patches": [
    {{
      "match": ".*", "kind": "wall",
      "U": {{ "type": "fixedValue", "value": [0, 0, 0] }},
      "p": {{ "type": "zeroGradient" }},
      "T": {{ "type": "zeroGradient" }}
    }}
  ],
  "initial": {{ "U": [0.01, 0.02, -0.01], "T": 300.0, "p": 0.0 }},{numerics},
  "run": {{ "endTime": 0.01, "deltaT": 1e-3 }}{parcels}
}}"#,
        numerics = flow_numerics()
    )
}

const FLOW_ARGS: &[&str] =
    &["-endTime", "0.01", "-deltaT", "1e-3", "-sealed", "-heaterPower", "50", "-check", "1000"];

// ==========================================================================
//  Bit-for-bit comparison, field by field
// ==========================================================================

fn bit_f(a: &[Scalar], b: &[Scalar]) -> usize {
    a.len().abs_diff(b.len()) + a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

fn bit_v(a: &[Vec3], b: &[Vec3]) -> usize {
    a.len().abs_diff(b.len())
        + a.iter().zip(b).filter(|(p, q)| trine(p, q)).count()
}

fn bit_d(a: &[DVec3], b: &[DVec3]) -> usize {
    a.len().abs_diff(b.len())
        + a.iter().zip(b).filter(|(p, q)| trine64(p, q)).count()
}

fn bit_i<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    a.len().abs_diff(b.len()) + a.iter().zip(b).filter(|(x, y)| x != y).count()
}

fn trine(p: &Vec3, q: &Vec3) -> bool {
    p.x.to_bits() != q.x.to_bits() || p.y.to_bits() != q.y.to_bits() || p.z.to_bits() != q.z.to_bits()
}

fn trine64(p: &DVec3, q: &DVec3) -> bool {
    p.x.to_bits() != q.x.to_bits() || p.y.to_bits() != q.y.to_bits() || p.z.to_bits() != q.z.to_bits()
}

/// Every field of the two pool snapshots, x64 included.
fn parcel_mismatches(a: &ParcelSnapshot, b: &ParcelSnapshot) -> usize {
    bit_d(&a.x64, &b.x64)
        + bit_v(&a.x, &b.x)
        + bit_v(&a.u, &b.u)
        + bit_f(&a.d, &b.d)
        + bit_f(&a.temperature, &b.temperature)
        + bit_f(&a.n_p, &b.n_p)
        + bit_i(&a.cell, &b.cell)
        + bit_i(&a.uid, &b.uid)
        + bit_i(&a.flags, &b.flags)
        + bit_v(&a.impulse, &b.impulse)
        + bit_f(&a.exchange, &b.exchange)
        + bit_f(&a.heat, &b.heat)
        + bit_f(&a.heat_exchange, &b.heat_exchange)
        + bit_f(&a.mass_lost, &b.mass_lost)
        + bit_f(&a.latent, &b.latent)
        + (a.n_slots != b.n_slots) as usize
}

/// Every field of the two coupling snapshots.
fn coupling_mismatches(a: &CouplingSnapshot, b: &CouplingSnapshot) -> usize {
    bit_v(&a.force, &b.force)
        + bit_f(&a.exchange, &b.exchange)
        + bit_f(&a.heat, &b.heat)
        + bit_f(&a.heat_exchange, &b.heat_exchange)
        + bit_v(&a.momentum_su, &b.momentum_su)
        + bit_f(&a.momentum_sp, &b.momentum_sp)
        + bit_f(&a.energy_q, &b.energy_q)
        + bit_f(&a.energy_sp, &b.energy_sp)
        + bit_f(&a.vapour, &b.vapour)
        + bit_f(&a.vapour_enthalpy, &b.vapour_enthalpy)
        + bit_f(&a.species_su, &b.species_su)
        + bit_f(&a.divergence, &b.divergence)
}

// ==========================================================================
//  lowmach_parcels_equal_library
// ==========================================================================

const PAIR_STEPS: usize = 25;
const PAIR_DT: Scalar = 1e-3;

/// **SPEC-LIT 137.10, row D.** The 500 um water droplet of gate 76-A, run
/// through `CaseParcels` from a lowered `parcels` block for 25 steps, is
/// bitwise the library's own `Parcels`/`ParcelDeposition`/`ParcelCoupling`
/// driven by hand from the same settings - over every field of the pool and
/// coupling snapshots and both registries, x64 positions included.
#[test]
fn lowmach_parcels_equal_library() {
    let Ok(g) = Gpu::new(0) else { return };
    let path = write_case("pair", &evap_case(r#"{ "endTime": 0.025, "deltaT": 1e-3 }"#));
    let lowered =
        read_case_jsonc(&path).expect("the case parses").lower().expect("the case lowers");
    let lp = lowered.parcels.clone().expect("the case states parcels");

    // The library's objects, from HAND-WRITTEN controls.
    let ctrl = ParcelControls {
        capacity: 256,
        physics: ParcelPhysics::Evaporating,
        drag: DragModel::SchillerNaumann,
        gravity: Vec3::new(0.0, 0.0, -9.81),
        ..ParcelControls::default()
    };
    let inj = Injector {
        position: Vec3::new(0.02, 0.02, 0.03),
        axis: Vec3::new(0.0, 0.0, -1.0),
        speed: 0.5,
        diameter: 5e-4,
        temperature: 285.0,
        mass_flow: 1e-5,
        parcels_per_event: 2,
        interval: 0.005,
        ..Injector::default()
    };
    let cpl = CouplingControls {
        momentum: CouplingMode::SemiImplicit,
        energy: CouplingMode::Explicit,
        mass: MassCoupling::Evaporation,
    };
    assert_eq!(lp.controls, ctrl, "the lowering's controls");
    assert_eq!(lp.injectors, vec![inj], "the lowering's injector");
    assert_eq!(lp.coupling, cpl, "the lowering's coupling");
    assert_eq!(lp.vapour.as_deref(), Some("H2O"), "the lowering's vapour");

    // The vapour's place in the reacting mixture, as the pair test of
    // 133.11 builds it: molar masses straight from the thermo file.
    let ch = lowered.chem.clone().expect("the case states chemistry");
    let mech = ch.load(path.parent().expect("the case's directory")).expect("the mechanism");
    let th = mech.thermo();
    let w = |s: &str| th.species()[th.index_of(s).expect("a thermo record")].molar_mass;
    let (wv, wn) = (w("H2O"), w("N2"));
    let solved: Vec<(&str, f64)> =
        NAMES.iter().filter(|n| **n != "N2").map(|n| (*n, w(n))).collect();
    let mix = GasMixture::new(&solved, ("N2", wn)).expect("the mixture");
    let (k, f) = vapour_member(&mix, "H2O").expect("the vapour member");
    assert_eq!(k, solved.iter().position(|(n, _)| *n == "H2O").expect("H2O's index"));
    assert_eq!(
        f.to_bits(),
        (((1.0 / wn) / (1.0 / wv - 1.0 / wn)) as Scalar).to_bits(),
        "(137.9)'s factor"
    );

    // The two pools, on the same mesh and the same dt.
    let hm = lowered.build_host_mesh().expect("the mesh");
    let m = GpuMesh::upload(&g, &hm).expect("upload");
    let n = hm.n_cells;
    let mut pc = CaseParcels::new(&g, &hm, &m, &lp, PAIR_DT, Some((k, f))).expect("the driver's pool");
    let mut p = Parcels::new(&g, &hm, &m, ctrl, &[inj], PAIR_DT).expect("the library's pool");
    let mut dep = ParcelDeposition::new(&g, &p).expect("the deposit");
    let mut cp = ParcelCoupling::new(&g, &p, cpl).expect("the coupling");

    // The gas, fixed for every step: still 1.18 kg/m3 air at 25 C, 30 % rh.
    let u = GpuVectorField::zeros(&g, &m, "U").expect("U");
    let rho = g.upload(&vec![1.18 as Scalar; n]).expect("rho");
    let t = g.upload(&vec![298.15 as Scalar; n]).expect("T");
    let y = g.upload(&vec![5.86e-3 as Scalar; n]).expect("Y_v");
    let mut ms_d = MomentumSources::new(&g, &m).expect("the momentum registry");
    let mut ms_l = MomentumSources::new(&g, &m).expect("the momentum registry");
    let mut es_d = EnergySources::new(&g, &m).expect("the energy registry");
    let mut es_l = EnergySources::new(&g, &m).expect("the energy registry");
    let mut bad = 0usize;
    let (mut su_max, mut q_max, mut vap_sum) = (0.0f64, 0.0f64, 0.0f64);
    for _ in 0..PAIR_STEPS {
        pc.step(&g, &u, &rho, &t, Some(&y), PAIR_DT).expect("the driver's step");
        p.step(&g, &u, &rho, Some(&t), Some(&y), PAIR_DT).expect("the pool's step");
        dep.update(&g, &p).expect("deposit");
        cp.update(&g, &p, &dep, &rho, &u, Some(&t), Some(&y), PAIR_DT).expect("gather");
        let (ps_d, ps_l) =
            (pc.parcels().snapshot(&g).expect("pool snap"), p.snapshot(&g).expect("pool snap"));
        let (cs_d, cs_l) = (
            pc.coupling().snapshot(&g).expect("coupling snap"),
            cp.snapshot(&g).expect("coupling snap"),
        );
        bad += parcel_mismatches(&ps_d, &ps_l) + coupling_mismatches(&cs_d, &cs_l);
        for v in &cs_d.momentum_su {
            su_max = su_max
                .max(f64::from(v.x.abs()))
                .max(f64::from(v.y.abs()))
                .max(f64::from(v.z.abs()));
        }
        q_max = q_max.max(cs_d.energy_q.iter().fold(0.0f64, |a, &q| a.max(f64::from(q.abs()))));
        vap_sum = cs_d.vapour.iter().map(|v| f64::from(*v)).sum();

        pc.register_momentum(&g, &mut ms_d).expect("momentum register");
        ms_l.clear(&g).expect("clear");
        cp.register_momentum(&g, &mut ms_l).expect("momentum register");
        bad += bit_v(&g.download(ms_d.su()).expect("su"), &g.download(ms_l.su()).expect("su"));
        bad += bit_f(&g.download(ms_d.sp()).expect("sp"), &g.download(ms_l.sp()).expect("sp"));
        es_d.clear(&g).expect("clear");
        es_l.clear(&g).expect("clear");
        pc.register_energy(&g, &mut es_d).expect("energy register");
        cp.register_energy(&g, &mut es_l).expect("energy register");
        bad += bit_f(&g.download(es_d.q()).expect("q"), &g.download(es_l.q()).expect("q"));
        bad += bit_f(&g.download(es_d.sp()).expect("sp"), &g.download(es_l.sp()).expect("sp"));
    }
    assert_eq!(bad, 0, "{bad} bits differ between the driver and the library");
    assert!(pc.parcels().stats(&g).expect("stats").n_injected > 0, "nothing was injected");
    assert!(vap_sum > 0.0, "the pool produced no vapour: the gate is vacuous");
    assert!(su_max > 0.0, "no momentum was registered: the gate is vacuous");
    assert!(q_max > 0.0, "no heat was registered: the gate is vacuous");
    println!(
        "  [137.10] pair: {PAIR_STEPS} steps bitwise driver == library over both snapshots and \
         both registries; vapour {vap_sum:.3e} kg/(m3 s), max |su| {su_max:.3e} m/s2, \
         max |q| {q_max:.3e} W/m3"
    );
}

// ==========================================================================
//  lowmach_parcels_vapour_enters_species
// ==========================================================================

/// **SPEC-LIT 137.10, rows D and R.** The vapour's two entries into a
/// reacting step, on the pieces `run()` builds: (137.8) puts the pool's
/// `mdot'''` on the named species' reaction source, bitwise, and leaves
/// every other member's untouched; (137.9) moves the constraint's target
/// divergence by `R T mdot''' / (p0 W_v)`.
#[test]
fn lowmach_parcels_vapour_enters_species() {
    let Ok(g) = Gpu::new(0) else { return };
    let path = write_case("vapour", &evap_case(r#"{ "endTime": 0.025, "deltaT": 1e-3 }"#));
    let lowered =
        read_case_jsonc(&path).expect("the case parses").lower().expect("the case lowers");
    let ch = lowered.chem.clone().expect("the case states chemistry");
    let cc = lowered.to_case_controls();
    let base = path.parent().expect("the case's directory");
    let n = lowered.build_host_mesh().expect("the mesh").n_cells;
    let dt = f64::from(1e-3 as Scalar);
    let hm = lowered.build_host_mesh().expect("the mesh");
    let m = GpuMesh::upload(&g, &hm).expect("upload");

    // What run() builds for a reacting case, twice, identically.
    macro_rules! build {
        () => {{
            let props = GasProperties {
                pr: lowered.fluid.pr,
                pr_t: lowered.fluid.prt,
                pr_t_model: lowered.prt_model,
                ..GasProperties::default()
            };
            let ectrl = EnergyControls {
                t_solver: lowered.solver_for("T").expect("T solver"),
                t_relax: lowered.relax_for("T", EnergyControls::default().t_relax),
                div_scheme: lowered.div_for("div(phi,T)"),
                grad_scheme: lowered.grad,
                sn_grad: lowered.laplacian_sn_grad,
                n_non_orth_correctors: cc.algorithm.n_non_orth_correctors,
                ddt: DdtScheme::Euler,
                steady: false,
                delta_t: 1e-3 as Scalar,
            };
            let mut e = Energy::new(&g, &m, ectrl, props).expect("energy");
            let raw_t = lowered.t_field.as_ref().expect("T").to_raw(n);
            setup_scalar_field_with(
                &g,
                e.field_mut(),
                &raw_t,
                &hm,
                BcInputs { thermal: true, ..BcInputs::default() },
            )
            .expect("T setup");
            e.initialise(&g).expect("T initialise");
            let mut gas = GasState::new(&g, &m, props, DomainKind::Open, 101325.0).expect("gas");
            let num = CaseNumerics::read(&path, &cc, Some(&lowered)).expect("numerics");
            let (y_ctrl, y_conv) = crate::species_controls(&num, &cc.turb).expect("species ctrl");
            let raws = ch.species.fields.iter().map(|fl| fl.to_raw(n)).collect();
            let r = Reacting::new(
                &g, &hm, &m, &ch, base, raws, cc.nu, props.pr, y_ctrl, y_conv, dt,
            )
            .expect("reacting");
            gas.use_mixture(&g, r.mixture.clone()).expect("mixture");
            e.sources_mut().use_meters(&g).expect("meters");
            r.update_density(&g, &mut gas, e.field()).expect("rho");
            (e, gas, r)
        }};
    }
    let (mut e_a, mut gas_a, mut r_a) = build!();
    let (mut e_b, mut gas_b, mut r_b) = build!();

    // The vapour's place and factor, and the pool's mdot''' field.
    let mech = ch.load(base).expect("the mechanism");
    let th = mech.thermo();
    let w = |s: &str| th.species()[th.index_of(s).expect("a thermo record")].molar_mass;
    let (wv, wn) = (w("H2O"), w("N2"));
    let solved: Vec<(&str, f64)> =
        NAMES.iter().filter(|nm| **nm != "N2").map(|nm| (*nm, w(nm))).collect();
    let mix = GasMixture::new(&solved, ("N2", wn)).expect("the mixture");
    assert_eq!(mix, r_a.mixture, "the hand-built mixture is the run's");
    let (k, f) = vapour_member(&mix, "H2O").expect("the vapour member");
    let mdot_host: Vec<Scalar> =
        (0..n).map(|c| (1e-3 * (1.0 + c as f64 / n as f64)) as Scalar).collect();
    let mdot = g.upload(&mdot_host).expect("mdot");
    let nut0 = GpuScalarField::zeros(&g, &m, "nut").expect("nut");
    let k0 = g.upload(&vec![0.0 as Scalar; n]).expect("k");

    // A: no vapour. B: the same step with it.
    e_a.sources_mut().clear(&g).expect("clear");
    r_a.before_flow_with(&g, &m, &mut e_a, &mut gas_a, None).expect("before_flow");
    e_a.update_target_divergence(&g, &gas_a, &nut0, &k0, cc.nu).expect("target");
    let target_a = g.download(e_a.target_divergence()).expect("download");
    let su_a: Vec<Vec<Scalar>> = r_a
        .species
        .names()
        .iter()
        .map(|nm| g.download(r_a.species.reaction_source(nm).expect("su")).expect("download"))
        .collect();

    e_b.sources_mut().clear(&g).expect("clear");
    r_b
        .before_flow_with(
            &g,
            &m,
            &mut e_b,
            &mut gas_b,
            Some(VapourSource { index: k, mdot: &mdot, factor: f }),
        )
        .expect("before_flow");
    e_b.update_target_divergence(&g, &gas_b, &nut0, &k0, cc.nu).expect("target");
    let target_b = g.download(e_b.target_divergence()).expect("download");
    let su_b: Vec<Vec<Scalar>> = r_b
        .species
        .names()
        .iter()
        .map(|nm| g.download(r_b.species.reaction_source(nm).expect("su")).expect("download"))
        .collect();

    // (137.8), class D: H2O's reaction source gained exactly mdot''', the
    // addition formed in Scalar on the host; no other member moved.
    let names = r_a.species.names();
    let (mut bad_v, mut bad_o) = (0usize, 0usize);
    for (i, _nm) in names.iter().enumerate() {
        for c in 0..n {
            if i == k {
                if su_b[i][c].to_bits() != (su_a[i][c] + mdot_host[c]).to_bits() {
                    bad_v += 1;
                }
            } else if su_a[i][c].to_bits() != su_b[i][c].to_bits() {
                bad_o += 1;
            }
        }
    }
    assert_eq!(bad_v, 0, "(137.8): H2O's source is not su_A + mdot''' bitwise");
    assert_eq!(bad_o, 0, "(137.8): another member's reaction source moved");

    // (137.9), class R: the target divergence moved by R T mdot'''/(p0 W_v).
    let r_univ = f64::from(gas_b.props().r_universal);
    let p0 = 101325.0f64;
    let t_a = g.download(&e_a.field().f).expect("T");
    let (mut max_err, mut max_want) = (0.0f64, 0.0f64);
    for c in 0..n {
        let delta = f64::from(target_b[c]) - f64::from(target_a[c]);
        let want = r_univ * f64::from(t_a[c]) * f64::from(mdot_host[c]) / (p0 * wv);
        max_err = max_err.max((delta - want).abs());
        max_want = max_want.max(want.abs());
    }
    let tol = if cfg!(feature = "single") { 1e-5 } else { 1e-12 };
    assert!(
        max_err <= tol * max_want,
        "(137.9): max |delta - want| {max_err:e} against max |want| {max_want:e} (tol {tol:e})"
    );
    println!(
        "  [137.10] vapour: (137.8) bitwise on H2O (k = {k}), other members untouched; \
         (137.9) max |delta - want| {max_err:.3e}, max |want| {max_want:.3e}, ratio {:.3e} \
         (tol {tol:e})",
        max_err / max_want
    );
}

// ==========================================================================
//  lowmach_no_parcels_no_change
// ==========================================================================

/// **SPEC-LIT 137.10, row D - Gate 68-B through the driver.** A full
/// `ofgpu-lowmach` run of a case with no parcels block, with the model
/// present but every parcel of zero weight, and with the coupled spray:
/// zero weight moves not one bit of the final flow, and the spray does.
#[test]
fn lowmach_no_parcels_no_change() {
    let Ok(g) = Gpu::new(0) else { return };
    let _ = &g;
    let a = run_case(&write_case("dry", &flow_case("")), FLOW_ARGS).expect("A runs");
    let b = run_case(
        &write_case("zero", &flow_case(&flow_parcels("explicit", "0.0"))),
        FLOW_ARGS,
    )
    .expect("B runs");
    let c = run_case(
        &write_case("spray", &flow_case(&flow_parcels("explicit", "1e-4"))),
        FLOW_ARGS,
    )
    .expect("C runs");

    let flow_gap = |x: &crate::RunEnd, y: &crate::RunEnd| -> usize {
        x.final_flow
            .iter()
            .map(|(name, vx)| {
                let vy = &y
                    .final_flow
                    .iter()
                    .find(|(n, _)| n == name)
                    .unwrap_or_else(|| panic!("{name}"))
                    .1;
                vx.len().abs_diff(vy.len())
                    + vx.iter().zip(vy).filter(|(u, v)| u.to_bits() != v.to_bits()).count()
            })
            .sum()
    };
    let ab = flow_gap(&a, &b);
    let ab_ux = (a.ux_mean.to_bits() != b.ux_mean.to_bits()) as usize;
    let ab_t = (a.t_mean.to_bits() != b.t_mean.to_bits()) as usize;
    assert_eq!(ab, 0, "the zero-weight model moved the final flow");
    assert_eq!(ab_ux, 0, "the zero-weight model moved ux_mean");
    assert_eq!(ab_t, 0, "the zero-weight model moved t_mean");
    let pe = b.parcels.as_ref().expect("B carried its parcels end");
    assert!(pe.stats.n_injected > 0, "B injected nothing: the gate is vacuous");
    let ca = flow_gap(&a, &c);
    assert!(ca > 0, "the coupled spray moved no bit of the flow: the gate is vacuous");
    println!(
        "  [137.10] gate 68-B through the driver: A vs B final_flow {ab} / ux {ab_ux} / \
         t {ab_t} bit mismatches; C vs A {ca} values differ (the spray moves the gas)"
    );
}

// ==========================================================================
//  lowmach_parcels_momentum_balance
// ==========================================================================

/// **SPEC-LIT 137.10, row R - VAL-72a's row through the driver.** On the
/// coupled spray's run, the momentum the gas was given by the last step,
/// (68.4), is the momentum the live parcels lost, to the build's own
/// round-off.
#[test]
fn lowmach_parcels_momentum_balance() {
    let Ok(g) = Gpu::new(0) else { return };
    let _ = &g;
    let end = run_case(
        &write_case("balance", &flow_case(&flow_parcels("semiImplicit", "1e-4"))),
        FLOW_ARGS,
    )
    .expect("the coupled run");
    let pe = end.parcels.as_ref().expect("the run carried its parcels end");
    let (ga, lo) = (&pe.gained, &pe.lost);
    let defect = f64::from(ga.x + lo.x).hypot(f64::from(ga.y + lo.y)).hypot(f64::from(ga.z + lo.z));
    let scale = f64::from(lo.x).hypot(f64::from(lo.y)).hypot(f64::from(lo.z));
    let tol = if cfg!(feature = "single") { 1e-5 } else { 1e-12 };
    assert!(scale > 0.0, "the live parcels lost no momentum: the gate is vacuous");
    assert!(
        defect <= tol * scale,
        "(68.4): defect {defect:e} against scale {scale:e} (tol {tol:e})"
    );
    println!(
        "  [137.10] (68.4) on the run: defect {defect:.3e} N s, lost {scale:.3e} N s, \
         ratio {:.3e} (tol {tol:e}), {} parcels injected",
        defect / scale,
        pe.stats.n_injected
    );
}

// ==========================================================================
//  lowmach_parcels_refusals_name_the_setting
// ==========================================================================

/// **SPEC-LIT 137.10, row D.** What the parcels step cannot honour is
/// refused by name before a field is set up: the run settings, the
/// physics/coupling pairings, the vapour's species, the one-molar-mass
/// factor, and both driver seams - the steady run and the non-lowmach
/// `load_case`.
#[test]
fn lowmach_parcels_refusals_name_the_setting() {
    let Ok(g) = Gpu::new(0) else { return };
    let _ = &g;
    let path = write_case("refuse", &evap_case(r#"{ "endTime": 0.025, "deltaT": 1e-3 }"#));
    let lowered = read_case_jsonc(&path).expect("the case parses").lower().expect("the case lowers");
    let mut lp = lowered.parcels.clone().expect("the case states parcels");
    let solved: Vec<String> =
        NAMES.iter().filter(|n| **n != "N2").map(|s| s.to_string()).collect();
    fn shape<'a>(
        transient: bool,
        restart: bool,
        motion: bool,
        species: Option<(&'a [String], &'a str)>,
    ) -> ParcelRunShape<'a> {
        ParcelRunShape { transient, restart, motion, species }
    }
    let run = Some((solved.as_slice(), "N2"));
    refuse_settings(&lp, &shape(true, false, false, run))
        .expect("the evaporating shape is runnable");

    let says = |tag: &str, e: Error, key: &str| {
        let m = e.to_string();
        assert!(m.contains(key) && m.contains("parcels") && m.contains("137.10"), "{tag}: {m}");
    };
    let e = refuse_settings(&lp, &shape(false, false, false, run)).expect_err("(a) refused");
    says("(a) steady", e, "parcels:");
    let e = refuse_settings(&lp, &shape(true, true, false, run)).expect_err("(b) refused");
    says("(b) restart", e, "parcels:");
    let e = refuse_settings(&lp, &shape(true, false, true, run)).expect_err("(c) refused");
    says("(c) motion", e, "parcels:");

    lp.coupling.mass = MassCoupling::None;
    lp.vapour = None;
    let e = refuse_settings(&lp, &shape(true, false, false, run)).expect_err("(d) refused");
    says("(d) mass none", e, "parcels.physics");

    lp.coupling.mass = MassCoupling::Evaporation;
    lp.vapour = Some("H2O".to_string());
    let e = refuse_settings(&lp, &shape(true, false, false, None)).expect_err("(e) refused");
    says("(e) no species", e, "parcels.coupling.vapour");
    lp.vapour = Some("N2".to_string());
    let e = refuse_settings(&lp, &shape(true, false, false, run)).expect_err("(f) refused");
    says("(f) the inert", e, "parcels.coupling.vapour");
    lp.vapour = Some("Ar".to_string());
    let e = refuse_settings(&lp, &shape(true, false, false, run)).expect_err("(g) refused");
    says("(g) not named", e, "parcels.coupling.vapour");

    let mix = GasMixture::new(&[("A", 0.028)], ("N", 0.028)).expect("the mixture");
    let e = vapour_member(&mix, "A").expect_err("(h) refused");
    says("(h) one molar mass", e, "molar mass");

    // Through the driver: a steady run of the parcels case, and the seam
    // every other driver loads its cases through.
    let steady = write_case("steady", &evap_case_steady());
    let e = run_case(&steady, &["-iters", "5"])
        .expect_err("a steady parcels run is refused")
        .to_string();
    assert!(e.contains("parcels") && e.contains("137.10"), "steady: {e}");
    let e =
        crate::common::load_case(&path).err().expect("the seam refuses parcels").to_string();
    assert!(e.contains("137.10"), "seam: {e}");
    println!(
        "  [137.10] refusals: (a)-(h) name their settings; the steady run says \"parcels\"; \
         load_case says 137.10"
    );
}

// ==========================================================================
//  The landing ledger - SPEC-LIT 147.2
// ==========================================================================

/// **SPEC-LIT 147.2, row D.** `plane_crossing`'s exact contract: the
/// interpolated x of a downward crossing, and `None` for a start on the
/// plane or a step that never comes down through it.
#[test]
fn plane_crossing_is_exact() {
    assert_eq!(super::plane_crossing(2.0, 1.0, -2.0, 5.0, 0.0), Some(3.0));
    assert_eq!(super::plane_crossing(1.0, 0.0, 0.0, 4.0, 0.0), Some(4.0));
    assert_eq!(super::plane_crossing(0.0, 0.0, -1.0, 1.0, 0.0), None);
    assert_eq!(super::plane_crossing(2.0, 0.0, 1.0, 1.0, 0.0), None);
}

/// **SPEC-LIT 147.2, row D.** `mass_percentile`'s exact contract: the
/// smallest x whose running mass reaches q of the total, `None` with no
/// hits or no mass.
#[test]
fn mass_percentile_is_exact() {
    let hits = [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (4.0, 1.0)];
    assert_eq!(super::mass_percentile(&hits, 0.25), Some(1.0));
    assert_eq!(super::mass_percentile(&hits, 0.5), Some(2.0));
    assert_eq!(super::mass_percentile(&hits, 0.9), Some(4.0));
    assert_eq!(super::mass_percentile(&hits, 0.99), Some(4.0));
    assert_eq!(super::mass_percentile(&[], 0.5), None);
    assert_eq!(super::mass_percentile(&[(1.0, 0.0), (2.0, 0.0)], 0.5), None);
}

/// The 1 m x 0.25 m x 1 m all-wall box the landing tests walk in.
fn landing_box() -> ofgpu::HostMesh {
    let axis = |lo: Scalar, hi: Scalar, n: usize| ofgpu::blockgen::GradedAxis {
        lo,
        hi,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    let spec = |x: ofgpu::blockgen::GradedAxis| ofgpu::blockgen::BlockSpec {
        x,
        y: axis(0.0, 0.25, 2),
        z: axis(0.0, 1.0, 10),
        windows: Vec::new(),
        patch_name: ofgpu::blockgen::BlockSpec::default().patch_name,
        patch_type: ["patch"; 6].map(String::from),
        cyclic: Vec::new(),
    };
    ofgpu::blockgen::build_mesh(&spec(axis(0.0, 1.0, 10))).expect("the box")
}

/// The same box widened to 1.6 m in x (the 64-parcel throw reaches
/// ~1.17 m) and with a WALL floor: `patch_type[4]`, the `-z` face in
/// blockgen's `-x +x -y +y -z +z` order. The floor is the box's only
/// wall and is the landing plane, so every wall end is a landing.
fn landing_floor_box() -> ofgpu::HostMesh {
    let axis = |lo: Scalar, hi: Scalar, n: usize| ofgpu::blockgen::GradedAxis {
        lo,
        hi,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    let mut patch_type = ["patch"; 6].map(String::from);
    patch_type[4] = "wall".to_string();
    let spec = |x: ofgpu::blockgen::GradedAxis| ofgpu::blockgen::BlockSpec {
        x,
        y: axis(0.0, 0.25, 2),
        z: axis(0.0, 1.0, 10),
        windows: Vec::new(),
        patch_name: ofgpu::blockgen::BlockSpec::default().patch_name,
        patch_type,
        cyclic: Vec::new(),
    };
    ofgpu::blockgen::build_mesh(&spec(axis(0.0, 1.6, 16))).expect("the box")
}

/// **SPEC-LIT 147.2, row R.** One seeded ballistic parcel - drag none, the
/// gas at rest, no coupling - crossing the recorded plane: the ledger
/// counts exactly one landing and its x is the closed-form parabola's,
/// 0.1 + sqrt(2 * 0.5 / 9.81) m, to 2e-3 m.
#[test]
fn a_landing_ledger_finds_the_ballistic_throw() {
    use ofgpu::io::case_parcels::LoweredParcels;
    use ofgpu::parcels::{SeedParcel, WallAction};
    let Ok(g) = Gpu::new(0) else { return };
    let hm = landing_box();
    let m = GpuMesh::upload(&g, &hm).expect("upload");
    let lp = LoweredParcels {
        controls: ParcelControls {
            capacity: 16,
            drag: DragModel::None,
            physics: ParcelPhysics::Inert,
            wall: WallAction::Remove,
            gravity: Vec3::new(0.0, 0.0, -9.81),
            ..ParcelControls::default()
        },
        injectors: Vec::new(),
        coupling: CouplingControls {
            momentum: CouplingMode::Off,
            energy: CouplingMode::Off,
            mass: MassCoupling::None,
        },
        vapour: None,
        landing: Some(0.25),
    };
    let dt = 1e-3 as Scalar;
    let mut pc = CaseParcels::new(&g, &hm, &m, &lp, dt, None).expect("the pool");
    pc.seed(
        &g,
        &hm,
        &[SeedParcel {
            position: Vec3::new(0.1, 0.125, 0.75),
            velocity: Vec3::new(1.0, 0.0, 0.0),
            diameter: 1e-3,
            temperature: 293.15,
            n_p: 1.0,
            uid: Some(1),
        }],
    )
    .expect("the seed");
    let u = GpuVectorField::zeros(&g, &m, "U").expect("U");
    let rho = g.upload(&vec![1.2 as Scalar; hm.n_cells]).expect("rho");
    let t = g.upload(&vec![293.15 as Scalar; hm.n_cells]).expect("T");
    let want = 0.1 + (2.0 * 0.5_f64 / 9.81).sqrt();
    let mut got = None;
    for _ in 0..1000 {
        pc.step(&g, &u, &rho, &t, None, dt).expect("the step");
        match pc.landing_summary() {
            Some(s) if s.crossings > 0 => {
                got = Some(s);
                break;
            }
            _ => {}
        }
    }
    let s = got.expect("the parcel never crossed the plane");
    assert_eq!(s.crossings, 1, "one parcel, one landing: {s:?}");
    let err = (s.farthest.expect("a farthest") - want).abs();
    println!(
        "  [147.2] ballistic landing: farthest {} m against the closed form {want} m, \
         error {err:.3e} m",
        s.farthest.unwrap()
    );
    assert!(err <= 2e-3, "farthest {} against {want}, error {err:.3e} m", s.farthest.unwrap());
}

/// **SPEC-LIT 147.2, row D.** A pool whose case states no landing plane:
/// `landing_summary` is `None` and the report line is the one this driver
/// has always printed - the word "landing" appears nowhere in it.
#[test]
fn no_landing_no_report_change() {
    use ofgpu::io::case_parcels::LoweredParcels;
    use ofgpu::parcels::{SeedParcel, WallAction};
    let Ok(g) = Gpu::new(0) else { return };
    let hm = landing_box();
    let m = GpuMesh::upload(&g, &hm).expect("upload");
    let lp = LoweredParcels {
        controls: ParcelControls {
            capacity: 16,
            drag: DragModel::None,
            physics: ParcelPhysics::Inert,
            wall: WallAction::Remove,
            gravity: Vec3::new(0.0, 0.0, -9.81),
            ..ParcelControls::default()
        },
        injectors: Vec::new(),
        coupling: CouplingControls {
            momentum: CouplingMode::Off,
            energy: CouplingMode::Off,
            mass: MassCoupling::None,
        },
        vapour: None,
        landing: None,
    };
    let dt = 1e-3 as Scalar;
    let mut pc = CaseParcels::new(&g, &hm, &m, &lp, dt, None).expect("the pool");
    pc.seed(
        &g,
        &hm,
        &[SeedParcel {
            position: Vec3::new(0.1, 0.125, 0.75),
            velocity: Vec3::new(1.0, 0.0, 0.0),
            diameter: 1e-3,
            temperature: 293.15,
            n_p: 1.0,
            uid: Some(1),
        }],
    )
    .expect("the seed");
    let u = GpuVectorField::zeros(&g, &m, "U").expect("U");
    let rho = g.upload(&vec![1.2 as Scalar; hm.n_cells]).expect("rho");
    let t = g.upload(&vec![293.15 as Scalar; hm.n_cells]).expect("T");
    pc.step(&g, &u, &rho, &t, None, dt).expect("the step");
    assert!(pc.landing_summary().is_none(), "no plane, no summary");
    let line = pc.report_line(&g).expect("the report line");
    assert!(!line.contains("landing"), "{line}");
}

/// **SPEC-LIT 147.2, row R.** Sixty-four seeded ballistic parcels fall onto
/// a floor that IS the landing plane (`zmin` is the box's only wall and
/// the plane is z = 0): the ledger must count every wall end as a landing.
/// A parcel removed at the wall stops at the face point, a few ulps either
/// side of the plane in position precision, and the ledger takes a stop
/// within `ON_PLANE` above the plane as ON it. Each crossing x is held to
/// the closed form `x0 + vx sqrt(2 z0 / 9.81)` to 2e-3 m.
#[test]
fn a_floor_at_the_plane_counts_every_landing() {
    use ofgpu::io::case_parcels::LoweredParcels;
    use ofgpu::parcels::{SeedParcel, WallAction};
    let Ok(g) = Gpu::new(0) else { return };
    let hm = landing_floor_box();
    let m = GpuMesh::upload(&g, &hm).expect("upload");
    let lp = LoweredParcels {
        controls: ParcelControls {
            capacity: 128,
            drag: DragModel::None,
            physics: ParcelPhysics::Inert,
            wall: WallAction::Remove,
            gravity: Vec3::new(0.0, 0.0, -9.81),
            ..ParcelControls::default()
        },
        injectors: Vec::new(),
        coupling: CouplingControls {
            momentum: CouplingMode::Off,
            energy: CouplingMode::Off,
            mass: MassCoupling::None,
        },
        vapour: None,
        landing: Some(0.0),
    };
    let dt = 1e-3 as Scalar;
    let mut pc = CaseParcels::new(&g, &hm, &m, &lp, dt, None).expect("the pool");
    let seeds: Vec<SeedParcel> = (0..64u32)
        .map(|k| {
            let k = f64::from(k);
            SeedParcel {
                position: Vec3::new(
                    (0.05 + 0.0113 * k) as Scalar,
                    0.125,
                    (0.61 + 0.0047 * k) as Scalar,
                ),
                velocity: Vec3::new((0.37 + 0.0091 * k) as Scalar, 0.0, 0.0),
                diameter: 1e-3,
                temperature: 293.15,
                n_p: 1.0,
                uid: None,
            }
        })
        .collect();
    pc.seed(&g, &hm, &seeds).expect("the seed");
    let u = GpuVectorField::zeros(&g, &m, "U").expect("U");
    let rho = g.upload(&vec![1.2 as Scalar; hm.n_cells]).expect("rho");
    let t = g.upload(&vec![293.15 as Scalar; hm.n_cells]).expect("T");
    let mut n_wall = 0;
    for _ in 0..2000 {
        pc.step(&g, &u, &rho, &t, None, dt).expect("the step");
        n_wall = pc.parcels().stats(&g).expect("the stats").n_wall;
        if n_wall == 64 {
            break;
        }
    }
    let sum = pc.landing_summary().expect("the summary");
    println!(
        "  [147.2] floor at the plane: wall {n_wall} against crossings {}",
        sum.crossings
    );
    assert_eq!(n_wall, 64, "every parcel reaches the floor");
    assert_eq!(sum.crossings, 64, "the ledger counts every landing: {sum:?}");
    assert_eq!(sum.crossings as i64, n_wall, "crossings = wall at the end");
    // The closed-form throw of each seed, in the order the ledger's hits
    // come back in: x order - here also k order, the throw grows with k.
    let mut want: Vec<(u32, f64)> = (0..64u32)
        .map(|k| {
            let kf = f64::from(k);
            let z0 = 0.61 + 0.0047 * kf;
            let x0 = 0.05 + 0.0113 * kf;
            let vx = 0.37 + 0.0091 * kf;
            (k, x0 + vx * (2.0 * z0 / 9.81).sqrt())
        })
        .collect();
    want.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("a finite throw"));
    let hits = pc.landing_hits();
    assert_eq!(hits.len(), 64, "one hit per parcel, none twice: {sum:?}");
    for ((x, _), (k, wx)) in hits.iter().zip(&want) {
        let err = (x - wx).abs();
        assert!(err <= 2e-3, "parcel {k}: x {x} against {wx}, error {err:.3e} m");
    }
}
