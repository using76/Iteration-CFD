// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §133.11's gates for `ofgpu-lowmach`'s reacting step (CHR-12b),
//! in both builds: the 0-D box through the case against the library calls
//! made by hand (class D, bitwise), the element content of a 1-D reacting
//! tube (class R), and the driver's refusals by name (class D).
//!
//! The chemistry is CHR-06's fifteen-reaction fixture of §130.7 over the ten
//! NASA TM-4513 records of `reference/mcbride-gordon-reno1993/table_II.csv`
//! (a US-Government work); its rate numbers are PLACEHOLDERS, no paper's.
//! `cases/mech/jl4.ck` does not exist (its paper is unread), so the plan's
//! "reduced 1-D JL flame" runs this fixture.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use ofgpu::chem::source::{ChemSource, SpeciesEnthalpy};
use ofgpu::energy::{DomainKind, Energy, EnergyControls, GasMixture, GasProperties, GasState};
use ofgpu::field::{BcKind, GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use ofgpu::field_ops::{self, update_inlet_outlet_scalar, FieldKernels};
use ofgpu::field_setup::{setup_scalar_field, setup_scalar_field_with, BcInputs};
use ofgpu::io::case_json::read_case_jsonc;
use ofgpu::io::fields::read_scalar_field;
use ofgpu::species::{Species, SpeciesCoeffs};
use ofgpu::timescheme::DdtScheme;
use ofgpu::turbulence::FlowState;
use ofgpu::{Error, Gpu, GpuMesh, Label, Scalar};

const TABLE_II: &str = include_str!("../../../../reference/mcbride-gordon-reno1993/table_II.csv");

/// The ten TM-4513 species: the table's code, the name, the elements.
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

/// The ten records as a CHEMKIN THERMO file (SAND89-8009 Table III's
/// columns), each coefficient written in the record's own E15.8 digits.
pub(crate) fn thermo_text() -> String {
    let mut rows: Vec<(String, f64, [f64; 7])> = Vec::new();
    for line in TABLE_II.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("species") {
            continue;
        }
        let c: Vec<&str> = line.split(',').collect();
        let mut a = [0.0f64; 7];
        for (j, v) in a.iter_mut().enumerate() {
            *v = c[4 + j].parse().expect("a coefficient");
        }
        rows.push((c[0].to_string(), c[1].parse().expect("t_lo"), a));
    }
    let f = |x: f64| format!("{x:>15.8E}");
    let mut t = String::from("THERMO ALL\n   200.000  1000.000  6000.000\n");
    for (code, name, els) in CODES {
        let pick = |t_lo: f64| rows.iter().find(|(c, t, _)| c == code && *t == t_lo).expect("row").2;
        let (up, lo) = (pick(1000.0), pick(200.0));
        let mut c1 = format!("{name:<18}TEST  ");
        for (s, n) in *els {
            c1.push_str(&format!("{s:<2}{n:>3}"));
        }
        while c1.len() < 44 {
            c1.push(' ');
        }
        c1.push_str(&format!("G{:>10.3}{:>10.3}{:>8.2}      1", 200.0, 6000.0, 1000.0));
        let c2 = format!("{}{}{}{}{}    2", f(up[0]), f(up[1]), f(up[2]), f(up[3]), f(up[4]));
        let c3 = format!("{}{}{}{}{}    3", f(up[5]), f(up[6]), f(lo[0]), f(lo[1]), f(lo[2]));
        let c4 = format!("{}{}{}{}                   4", f(lo[3]), f(lo[4]), f(lo[5]), f(lo[6]));
        t.push_str(&format!("{c1}\n{c2}\n{c3}\n{c4}\n"));
    }
    t.push_str("END\n");
    t
}

/// CHR-06's fifteen-reaction fixture of §130.7 (placeholder rates).
pub(crate) const MECHANISM: &str = "ELEMENTS H O C N END
SPECIES H2 O2 H2O OH O H CO CO2 CH4 N2 END
REACTIONS
H + O2 <=> O + OH                 1.0E+14  -0.5  15000.0
O + H2 <=> H + OH                 1.0E+05   2.5   6000.0
OH + H2 <=> H + H2O               2.0E+08   1.5   3000.0
H + OH + M <=> H2O + M            2.0E+22  -2.0      0.0
  H2O/6.0/ H2/2.0/
O + O + M <=> O2 + M              1.0E+17  -1.0      0.0
H + H (+M) <=> H2 (+M)            1.0E+13   0.0      0.0
  LOW / 5.0E+18 -1.0 0.0 /
  H2O/5.0/
CO + O (+M) <=> CO2 (+M)          2.0E+10   0.0   2500.0
  LOW / 5.0E+14 0.0 3000.0 /
  TROE / 0.6 100.0 1500.0 5000.0 /
O + H2 (+N2) <=> H2O (+N2)        1.0E+12   0.0   1000.0
  LOW / 1.0E+18 -1.0 0.0 /
  TROE / 0.7 200.0 1000.0 /
H + O (+M) <=> OH (+M)            5.0E+12   0.0      0.0
  LOW / 4.0E+18 -1.0 0.0 /
  SRI / 0.5 800.0 200.0 1.2 0.1 /
CO + OH <=> CO2 + H               4.0E+06   1.5   -500.0
OH + OH <=> O + H2O               3.0E+04   2.4  -2000.0
  DUP
OH + OH <=> O + H2O               1.0E+12   0.0   1000.0
  DUP
H2 + O2 <=> OH + OH               1.7E+13   0.0  47780.0
CH4 + O2 <=> CO + H2O + H2        1.0E+12   0.0  30000.0
CH4 + O <=> CO + 2 H2             1.0E+13   0.0   8000.0
END
";

/// The mechanism's species, in its own order; N2 is the inert one.
pub(crate) const NAMES: [&str; 10] = ["H2", "O2", "H2O", "OH", "O", "H", "CO", "CO2", "CH4", "N2"];

/// The molar mass of `name`, kg/mol, from the table's record.
fn molar_mass(name: &str) -> f64 {
    let code = CODES.iter().find(|c| c.1 == name).expect("a fixture species").0;
    for line in TABLE_II.lines() {
        let c: Vec<&str> = line.trim().split(',').collect();
        if c.len() > 3 && c[0] == code {
            return c[3].parse::<f64>().expect("mol_weight") / 1000.0;
        }
    }
    panic!("no row for {name}");
}

/// Stoichiometric H2-air (2 : 1 : 3.76 by moles) with H, O and OH at 1e-4
/// moles each, as mass fractions in [`NAMES`] order.
fn h2_air() -> Vec<f64> {
    let xw: Vec<f64> = NAMES
        .iter()
        .map(|s| {
            let x = match *s {
                "H2" => 2.0,
                "O2" => 1.0,
                "N2" => 3.76,
                "H" | "O" | "OH" => 1.0e-4,
                _ => 0.0,
            };
            x * molar_mass(s)
        })
        .collect();
    let sum: f64 = xw.iter().sum();
    xw.iter().map(|v| v / sum).collect()
}

/// The `species`, `chemistry` and `combustion` blocks, with the solved
/// species held at [`h2_air`] on the patches matching `inlet` (when given).
/// The reactor's `rtol` is 1e-6 rather than §132.8's 1e-8: these gates
/// compare the driver with the library and an element balance, neither of
/// which depends on it, and the looser tolerance takes a third of the time.
fn chem_blocks(inlet: Option<&str>) -> String {
    let y = h2_air();
    let solved: Vec<(usize, &str)> =
        NAMES.iter().copied().enumerate().filter(|(_, n)| *n != "N2").collect();
    let initial: Vec<String> = solved.iter().map(|(i, n)| format!("\"{n}\": {:e}", y[*i])).collect();
    let patches = match inlet {
        Some(m) => {
            let ys: Vec<String> = solved
                .iter()
                .map(|(i, n)| format!("\"{n}\": {{ \"type\": \"fixedValue\", \"value\": {:e} }}", y[*i]))
                .collect();
            format!("[ {{ \"match\": \"{m}\", \"Y\": {{ {} }} }} ]", ys.join(", "))
        }
        None => "[]".to_string(),
    };
    let names: Vec<String> = NAMES.iter().map(|n| format!("\"{n}\"")).collect();
    format!(
        r#",
  "species": {{ "names": [{}], "inert": "N2", "initial": {{ {} }}, "patches": {patches} }},
  "chemistry": {{ "mechanism": "mech.ck", "thermo": "therm.dat", "rtol": 1e-6 }},
  "combustion": {{ "model": "laminar" }}"#,
        names.join(", "),
        initial.join(", ")
    )
}

pub(crate) const NUMERICS: &str = r#"
  "numerics": {
    "algorithm": { "kind": "PIMPLE", "correctors": 2 },
    "ddt": "Euler",
    "div": { "default": "Gauss upwind", "div(phi,U)": "Gauss upwind", "div(phi,T)": "bounded Gauss upwind" },
    "grad": "Gauss linear",
    "laplacian": { "snGrad": "corrected", "nonOrthogonalCorrectors": 0 },
    "relaxation": { "U": 1.0, "p": 1.0, "T": 1.0, "Yi": 1.0 },
    "solvers": [
      { "match": "p", "solver": "PBiCGStab", "preconditioner": "DIC", "tolerance": 1e-13, "relTol": 0.0, "maxIter": 2000 },
      { "match": "U", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": 1e-13, "relTol": 0.0, "maxIter": 2000 },
      { "match": "T", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": 1e-14, "relTol": 0.0, "maxIter": 2000 },
      { "match": "Yi", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": 1e-14, "relTol": 0.0, "maxIter": 2000 }
    ]
  }"#;

/// CHR-09's 0-D box as one cell, 1 cm, with adiabatic walls, at rest, `T`
/// 1400 K. One cell, because the open box's pressure source is then removed
/// to an exact zero (`Simple`'s mean of one value is that value): the box
/// stays at rest bitwise. On CHR-09's 2^3 cyclic box the uniform target
/// divergence leaves a round-off residue that the pressure solve turns
/// into a flow (measured: 2 m/s after 200 steps), so the flow solve is not
/// inert there and the pair would compare a different problem.
fn box_case(extra: &str, run: &str) -> String {
    format!(
        r#"{{
  "name": "reactingBox",
  "mesh": {{
    "kind": "cartesian",
    "bounds": {{ "min": [0, 0, 0], "max": [0.01, 0.01, 0.01] }},
    "cells": [2, 1, 1],
    "boundaries": {{ "xmin": "x0", "xmax": "x1", "ymin": "y0", "ymax": "y1", "zmin": "z0", "zmax": "z1" }}
  }},
  "physics": {{
    "gravity": [0, 0, 0],
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
  "initial": {{ "U": [0, 0, 0], "T": 1400.0, "p": 0.0 }},{NUMERICS},
  "run": {run}{extra}
}}"#
    )
}

/// The 1-D tube: 40 x 1 x 1 cells over 1 cm, premixed H2-air entering at
/// 1 m/s and 1400 K into gas at 1000 K, an open outlet, symmetric sides.
fn tube_case() -> String {
    format!(
        r#"{{
  "name": "reactingTube",
  "mesh": {{
    "kind": "cartesian",
    "bounds": {{ "min": [0, 0, 0], "max": [0.01, 0.00025, 0.00025] }},
    "cells": [40, 1, 1],
    "boundaries": {{ "xmin": "inlet", "xmax": "outlet", "ymin": "sideA", "ymax": "sideB", "zmin": "sideC", "zmax": "sideD" }}
  }},
  "physics": {{
    "gravity": [0, 0, 0],
    "fluid": {{ "nu": 1.5e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 293.15 }},
    "buoyancy": "densityRatio"
  }},
  "patches": [
    {{
      "match": "inlet", "kind": "inlet",
      "U": {{ "type": "fixedValue", "value": [1.0, 0, 0] }},
      "p": {{ "type": "zeroGradient" }},
      "T": {{ "type": "fixedValue", "value": 1400.0 }}
    }},
    {{
      "match": "outlet", "kind": "open",
      "U": {{ "type": "zeroGradient" }},
      "p": {{ "type": "fixedValue", "value": 0.0 }},
      "T": {{ "type": "inletOutlet", "inletValue": 1000.0 }}
    }},
    {{ "match": "side.*", "kind": "symmetry" }}
  ],
  "initial": {{ "U": [1.0, 0, 0], "T": 1000.0, "p": 0.0 }},{NUMERICS},
  "run": {{ "endTime": 0.002, "deltaT": 1e-5 }}{}
}}"#,
        chem_blocks(Some("inlet"))
    )
}

pub(crate) fn scratch_dir(tag: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let d = std::env::temp_dir().join(format!(
        "ofgpu_lowmach_chem_{}_{}_{tag}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("scratch dir");
    d
}

/// Write `text` as `case.jsonc` beside the two chemistry files; its path.
pub(crate) fn write_case(tag: &str, text: &str) -> PathBuf {
    let dir = scratch_dir(tag);
    std::fs::write(dir.join("mech.ck"), MECHANISM).expect("mechanism");
    std::fs::write(dir.join("therm.dat"), thermo_text()).expect("thermo");
    let path = dir.join("case.jsonc");
    std::fs::write(&path, text).expect("case");
    path
}

pub(crate) fn run_case(path: &Path, args: &[&str]) -> Result<crate::RunEnd, Error> {
    let mut a = vec!["ofgpu-lowmach".to_string(), path.to_string_lossy().to_string()];
    a.extend(args.iter().map(|s| (*s).to_string()));
    let o = crate::parse(&a)?;
    crate::run(&o)
}

/// The one time directory the run wrote (its final state, no schedule).
pub(crate) fn final_dir(case: &Path) -> PathBuf {
    let root = crate::common::json_case_output_dir(case);
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .expect("the output root")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.parse::<f64>().is_ok_and(|t| t > 0.0))
        })
        .collect();
    assert_eq!(dirs.len(), 1, "one final time directory expected under {}: {dirs:?}", root.display());
    dirs.pop().expect("one")
}

pub(crate) fn written(dir: &Path, name: &str, n: usize) -> Vec<Scalar> {
    read_scalar_field(&dir.join(name), n)
        .unwrap_or_else(|e| panic!("{name} was not written: {e}"))
        .internal
}

// ==========================================================================
//  lowmach_reacting_case_equals_library
// ==========================================================================

/// The flow step and the step count of CHR-09's box.
const DT: f64 = 1.0e-6;
const STEPS: usize = 200;

/// **SPEC-LIT 133.11, row D.** CHR-09's 0-D box (open, `p0` 1 atm) run by
/// `ofgpu-lowmach` from a case for 200 Lie steps at dt = 1e-6 s writes
/// every mass fraction, `T`, `rho` and `HRR` bitwise equal to §133.11's
/// step made by hand from the library's calls on the same lowered case,
/// without the flow solve: the box stays at rest (the written `U` is
/// exactly zero), so that solve changes nothing the step reads.
#[test]
fn lowmach_reacting_case_equals_library() {
    let Ok(g) = Gpu::new(0) else { return };
    let dt_s = DT as Scalar;
    let dt = f64::from(dt_s);
    let path = write_case("pair", &box_case(&chem_blocks(None), r#"{ "endTime": 2e-4, "deltaT": 1e-6 }"#));
    let end =
        run_case(&path, &["-endTime", "2e-4", "-deltaT", "1e-6", "-check", "1000"]).expect("the reacting box runs");
    let out = final_dir(&path);

    // ---- the library path ------------------------------------------------
    let l = read_case_jsonc(&path).expect("parse").lower().expect("lower");
    let ch = l.chem.clone().expect("the case states chemistry");
    let hm = l.build_host_mesh().expect("mesh");
    let mut cc = l.to_case_controls();
    cc.turb.delta_t = dt_s;
    let m = GpuMesh::upload(&g, &hm).expect("upload");
    let n = hm.n_cells;
    let props = GasProperties {
        pr: l.fluid.pr,
        pr_t: l.fluid.prt,
        pr_t_model: l.prt_model,
        ..GasProperties::default()
    };
    let ectrl = EnergyControls {
        t_solver: l.solver_for("T").expect("T solver"),
        t_relax: l.relax_for("T", EnergyControls::default().t_relax),
        div_scheme: l.div_for("div(phi,T)"),
        grad_scheme: l.grad,
        sn_grad: l.laplacian_sn_grad,
        n_non_orth_correctors: cc.algorithm.n_non_orth_correctors,
        ddt: DdtScheme::Euler,
        steady: false,
        delta_t: dt_s,
    };
    let mut e = Energy::new(&g, &m, ectrl, props).expect("energy");
    let raw_t = l.t_field.as_ref().expect("T").to_raw(n);
    setup_scalar_field_with(&g, e.field_mut(), &raw_t, &hm, BcInputs { thermal: true, ..BcInputs::default() })
        .expect("T setup");
    e.initialise(&g).expect("T initialise");
    let mut gas = GasState::new(&g, &m, props, DomainKind::Open, 101325.0).expect("gas");

    let mech = ch.load(path.parent().expect("dir")).expect("mechanism");
    let sp_l = &ch.species;
    let coeffs: Vec<SpeciesCoeffs> = sp_l
        .le
        .iter()
        .map(|&le| SpeciesCoeffs { d: cc.nu / (props.pr * le as Scalar), sc_t: sp_l.sct as Scalar })
        .collect();
    let conv = ofgpu::io::DivEntry { scheme: ofgpu::fv::DivScheme::Upwind, bounded: true };
    let mut yc = cc.turb;
    yc.k_solver = l.solver_for("Yi").expect("Yi solver");
    yc.k_relax = 1.0;
    yc.grad_scheme = l.grad;
    yc.sn_grad = l.laplacian_sn_grad;
    yc.div_scheme = conv.scheme;
    yc.bounded_convection = conv.bounded;
    let mut sp = Species::new(&g, &hm, &m, &sp_l.names, &coeffs, &sp_l.inert, cc.nu, yc).expect("species");
    sp.set_convection(conv);
    sp.use_mass_weighting(&g).expect("mass weighting");
    let fk = FieldKernels::new(&g).expect("kernels");
    for (i, f) in sp_l.fields.iter().enumerate() {
        setup_scalar_field(&g, sp.get_mut(i).expect("solved").field_mut(), &f.to_raw(n), &hm).expect("Y setup");
    }
    sp.initialise(&g).expect("species initialise");
    let ns = sp.n_solved();
    for i in 0..ns {
        field_ops::seed_old_time(&g, &fk, sp.get_mut(i).expect("solved").field_mut()).expect("seed");
    }
    let mut cs = ChemSource::new(&g, &mech, &mut sp, &ch.chemistry.reactor, n).expect("chemistry");
    let mut se = SpeciesEnthalpy::new(&g, &m, mech.thermo(), &sp).expect("enthalpy");
    let th = mech.thermo();
    let w = |s: &str| th.species()[th.index_of(s).expect("thermo")].molar_mass;
    let solved: Vec<(&str, f64)> = sp.names().iter().map(|s| (s.as_str(), w(s))).collect();
    gas.use_mixture(&g, GasMixture::new(&solved, ("N2", w("N2"))).expect("mixture")).expect("use mixture");
    e.sources_mut().use_meters(&g).expect("meters");
    {
        let fs: Vec<&GpuScalarField> = (0..ns).map(|i| sp.get(i).expect("solved").field()).collect();
        gas.update_density_mixture(&g, e.field(), &fs).expect("rho");
    }
    let z = |len: usize| g.upload(&vec![0.0 as Scalar; len.max(1)]).expect("buffer");
    let (mut pbuf, mut cp_c, mut cp_b, mut q, mut qhd) = (z(n), z(n), z(m.n_boundary_faces), z(n), z(n));
    let u0 = GpuVectorField::zeros(&g, &m, "U").expect("U");
    let phi0 = GpuSurfaceScalarField::zeros(&g, &m, "phi").expect("phi");
    let mut nut = GpuScalarField::zeros(&g, &m, "nut").expect("nut");
    g.write(&mut nut.bc_kind, &vec![BcKind::Calculated as Label; m.n_boundary_faces]).expect("nut kinds");
    for s in 0..STEPS {
        if s > 0 {
            e.advance_time_levels(&g).expect("T levels");
            e.advance_time_step(dt_s);
            gas.advance_time_levels();
            for i in 0..ns {
                field_ops::advance_time_levels(&g, &fk, sp.get_mut(i).expect("solved").field_mut())
                    .expect("Y levels");
            }
        }
        e.advance_lumped_masses(&g).expect("lumps");
        update_inlet_outlet_scalar(&g, &fk, e.field_mut(), &phi0).expect("T inletOutlet");
        {
            let fs: Vec<&GpuScalarField> = (0..ns).map(|i| sp.get(i).expect("solved").field()).collect();
            gas.update_density_mixture(&g, e.field(), &fs).expect("rho");
        }
        e.sources_mut().clear(&g).expect("clear");
        se.mixture_cp(&g, &sp, e.field(), &mut cp_c, &mut cp_b).expect("cp");
        e.set_cp_field(&g, &cp_c, &cp_b).expect("cp field");
        field_ops::set_field(&g, &fk, &mut pbuf, gas.p0(), n).expect("p");
        cs.step(&g, &mut sp, &e.field().f, &pbuf, Some(gas.rho()), dt).expect("split step");
        let wt = e.heat_release_weight(&g, &gas).expect("weight");
        cs.heat_release(&g, wt, &mut q).expect("heat release");
        se.diffusion_source(&g, &m, &sp, e.field(), &gas.rho().f, &mut qhd).expect("Q_hd");
        e.sources_mut().register_heat_release(&g, &q).expect("register");
        e.sources_mut().register_enthalpy_diffusion(&g, &qhd).expect("register");
        e.correct(&g, &phi0, &nut, &nut.f, cc.nu, &gas).expect("energy");
        let flow = FlowState::new(&u0, &phi0, cc.nu);
        sp.correct_with_density(&g, &flow, &nut, Some(gas.rho())).expect("species");
    }
    assert!(cs.refused_cells(&g).expect("status").is_empty(), "the reactor refused a cell");

    // ---- the comparison --------------------------------------------------
    let host = |b: &ofgpu::device::DevBuf<Scalar>| g.download(b).expect("download");
    let mut rows: Vec<(String, Vec<Scalar>)> = vec![
        ("T".into(), host(&e.field().f)),
        ("rho".into(), host(&gas.rho().f)),
        (crate::lowmach_chem::HRR.into(), host(&q)),
        ("N2".into(), host(&sp.inert().f)),
    ];
    for (i, name) in sp.names().iter().enumerate() {
        rows.push((name.clone(), host(&sp.get(i).expect("solved").field().f)));
    }
    let t_end = host(&e.field().f)[0];
    // Diagnostic (printed, not gated): the box is uniform and at rest, so
    // any change of its element content is the split step's own.
    let y0 = h2_air();
    for el in ["H", "O"] {
        let (mut n0, mut n1) = (0.0f64, 0.0f64);
        for (k, name) in NAMES.iter().enumerate() {
            let (_, _, els) = CODES.iter().find(|c| c.1 == *name).expect("fixture");
            let a = els.iter().find(|(s, _)| *s == el).map_or(0, |x| x.1);
            if a == 0 {
                continue;
            }
            let c = f64::from(a) / molar_mass(name);
            let yk = rows.iter().find(|(r, _)| r == name).expect("row").1[0];
            n0 += c * y0[k];
            n1 += c * f64::from(yk);
        }
        println!("  [133.11] box element {el}: relative change {:.3e} over {STEPS} steps", (n1 - n0).abs() / n0);
    }
    let mut worst = 0usize;
    for (name, lib) in &rows {
        // Written (six digits) and held at full precision by `RunEnd`.
        let _ = written(&out, name, n);
        let drv = &end.final_state.iter().find(|(r, _)| r == name).unwrap_or_else(|| panic!("{name}")).1;
        let diff = drv.iter().zip(lib).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let gap = drv.iter().zip(lib).map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs()).fold(0.0, f64::max);
        println!("  [133.11] {name:>4}: {diff} of {n} cells differ, largest gap {gap:.3e}");
        worst = worst.max(diff);
    }
    let u = ofgpu::io::fields::read_vector_field(&out.join("U"), n).expect("U").internal;
    assert!(u.iter().all(|v| v.x == 0.0 && v.y == 0.0 && v.z == 0.0), "the box moved: {:?}", u[0]);
    println!("  [133.11] box T 1400 -> {t_end} K over {STEPS} steps; the written U is exactly zero");
    assert!(f64::from(t_end) - 1400.0 > 10.0, "the box barely heated: the pair would compare nothing");
    assert_eq!(worst, 0, "the driver's reacting step is not the library's, bitwise");
}

// ==========================================================================
//  lowmach_reacting_elements_conserved
// ==========================================================================

/// **SPEC-LIT 133.11, row R.** The 1-D tube runs 200 Lie steps at dt = 1e-5 s
/// through `ofgpu-lowmach` without a NaN, and every cell's element content
/// `n_e = Σ_k a_ek Y_k / W_k` (mol/kg) of H, O and N stays at the premixed
/// value it entered with: equal diffusivities, the reactor's element balance
/// and §19's bounded convection keep a uniform composition uniform. C,
/// absent, stays absent. Tolerance `1e-10` relative, `1e-5` in f32 (D-F32-1).
#[test]
fn lowmach_reacting_elements_conserved() {
    if Gpu::new(0).is_err() {
        return;
    }
    let path = write_case("tube", &tube_case());
    let end = run_case(&path, &["-endTime", "0.002", "-deltaT", "1e-5", "-check", "50"]).expect("the tube runs");
    assert_eq!(end.steps, 200, "the tube ran {} steps", end.steps);
    let out = final_dir(&path);
    let n = 40;
    let y0 = h2_air();
    // (index into NAMES, a_ek / W_k) of every species carrying element `e`.
    let elements = |e: &str| -> Vec<(usize, f64)> {
        CODES
            .iter()
            .filter_map(|(_, name, els)| {
                let a = els.iter().find(|(s, _)| *s == e).map_or(0, |x| x.1);
                (a > 0).then(|| {
                    (NAMES.iter().position(|x| x == name).expect("named"), f64::from(a) / molar_mass(name))
                })
            })
            .collect()
    };
    // Written (six digits) and held at full precision by `RunEnd`.
    let state = |name: &str| -> Vec<Scalar> {
        let _ = written(&out, name, n);
        end.final_state.iter().find(|(r, _)| r == name).unwrap_or_else(|| panic!("{name}")).1.clone()
    };
    let ys: Vec<Vec<Scalar>> = NAMES.iter().map(|s| state(s)).collect();
    let t = state("T");
    let (t_lo, t_hi) =
        t.iter().fold((f64::INFINITY, 0.0f64), |(a, b), v| (a.min(f64::from(*v)), b.max(f64::from(*v))));
    let tol = if cfg!(feature = "single") { 1.0e-5 } else { 1.0e-10 };
    let mut worst = 0.0f64;
    for e in ["H", "O", "N", "C"] {
        let rows = elements(e);
        let n0: f64 = rows.iter().map(|(k, c)| c * y0[*k]).sum();
        let mut err = 0.0f64;
        for c in 0..n {
            let nc: f64 = rows.iter().map(|(k, a)| a * f64::from(ys[*k][c])).sum();
            err = err.max(if n0 > 0.0 { (nc - n0).abs() / n0 } else { nc.abs() });
        }
        println!("  [133.11] element {e}: n0 {n0:.6e} mol/kg, max relative change {err:.3e}");
        worst = worst.max(err);
    }
    let hrr = state(crate::lowmach_chem::HRR);
    let hrr_max = hrr.iter().fold(0.0f64, |a, v| a.max(f64::from(*v).abs()));
    println!(
        "  [133.11] tube T {t_lo:.2} .. {t_hi:.2} K, max |HRR| {hrr_max:.3e} W/m3, worst {worst:.3e} \
         (tol {tol:e})"
    );
    assert!(hrr_max > 0.0, "nothing reacted: the gate would measure nothing");
    assert!(worst <= tol, "an element left its premixed content by {worst:.3e} (tol {tol:e})");
}

// ==========================================================================
//  lowmach_reacting_refusals_name_the_setting
// ==========================================================================

/// **SPEC-LIT 133.11, row D.** What the reacting step cannot honour is
/// refused by name before the time loop: a steady run, Strang splitting, a
/// checkpoint, and a ddt other than Euler.
#[test]
fn lowmach_reacting_refusals_name_the_setting() {
    if Gpu::new(0).is_err() {
        return;
    }
    let base = box_case(&chem_blocks(None), r#"{ "endTime": 1e-5, "deltaT": 1e-6 }"#);
    let refuse = |tag: &str, text: &str, args: &[&str], key: &str| {
        let path = write_case(tag, text);
        let err = run_case(&path, args).map(drop).expect_err("must be refused").to_string();
        assert!(err.contains(key) && err.contains("133.11"), "{tag}: {err}");
        println!("  [133.11] {tag}: refused: {err}");
    };
    let tr = ["-endTime", "1e-5", "-deltaT", "1e-6"];
    let strang = base.replace(r#""thermo": "therm.dat""#, r#""thermo": "therm.dat", "splitting": "strang""#);
    let backward = base.replace(r#""ddt": "Euler""#, r#""ddt": "backward""#);
    let steady = base
        .replace(r#""ddt": "Euler""#, r#""ddt": "steadyState""#)
        .replace(r#"{ "kind": "PIMPLE", "correctors": 2 }"#, r#"{ "kind": "SIMPLE" }"#);
    refuse("steady", &steady, &["-iters", "3"], "chemistry");
    refuse("strang", &strang, &tr, "chemistry.splitting");
    refuse("restart", &base, &["-endTime", "1e-5", "-deltaT", "1e-6", "-restartWrite", "2"], "restart");
    refuse("backward", &backward, &tr, "numerics.ddt");
}

// ==========================================================================
//  lowmach_reacting_refused_cell_is_named
// ==========================================================================

/// **SPEC-LIT 133.11, row D (CHR-13).** A cell the reactor refuses ends the
/// run NAMING it: the error holds `the reactor refused`, the count, the
/// first cell, its status code and its name. The case is the 0-D box with
/// an initial `T` of 150 K, below the thermo's 200 K floor, so every cell
/// is refused on the first split step with OUTSIDE_THERMO_RANGE; before the
/// diagnostic the cell's NaN surfaced later as a non-finite pressure
/// residual and the run ended without naming the cause.
#[test]
fn lowmach_reacting_refused_cell_is_named() {
    if Gpu::new(0).is_err() {
        return;
    }
    let cold =
        box_case(&chem_blocks(None), r#"{ "endTime": 1e-5, "deltaT": 1e-6 }"#)
            .replace(r#""T": 1400.0"#, r#""T": 150.0"#);
    let path = write_case("refused", &cold);
    let err = run_case(&path, &["-endTime", "1e-5", "-deltaT", "1e-6"])
        .expect_err("the cold box must be refused")
        .to_string();
    println!("  [133.11] cold box: run ended: {err}");
    assert!(err.contains("the reactor refused"), "the error must name the reactor: {err}");
    assert!(
        err.contains("OUTSIDE_THERMO_RANGE"),
        "the error must carry the status NAME: {err}"
    );
    assert!(err.contains("133.11"), "the error must name its section: {err}");
}
