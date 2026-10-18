// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

// DST-10's gate table (SPEC-LIT §158.6): the Phase A dust model in the
// compressible solver. Written once here and included by
// `src/compressible/tests.rs` (module `dust`, the lib tests) and by
// `tests/compressible_dust.rs`, which runs the same tests in BOTH builds.
// `ofgpu` must name the crate in the including scope (`use crate as ofgpu;`
// in the lib); this file is included, not compiled standalone. Every
// comparison is made in f64 from the downloaded `Scalar` values; "bitwise"
// is `to_bits()` on the downloaded `Scalar`.
//
// `no_dust_no_change` was written and its two pins taken by the supervisor
// on the tree BEFORE DST-10 changed any file (feat/t17-S at 8b13cb2): it is
// the "with no dust block the solver is bitwise unchanged" row (D) and is
// never re-pinned.
//
// Provenance: ORIGINAL - the meshes and fields below are this tree's own
// construction; the references are SPEC-LIT (152.11), (158.2) and (158.3).
// The specification was informed by a review of FSCDustFoam (Huang &
// Lipatnikov, GPL) and no code was copied. No GPL-licensed source was
// consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::compressible::bc::{self as cbc, Lowered};
use ofgpu::compressible::pressure::Retarded;
use ofgpu::compressible::thermo::PerfectGas;
use ofgpu::compressible::{CompressibleControls, CompressibleState};
use ofgpu::device::DevBuf;
use ofgpu::dust::vent::{build_solver, setup_state};
use ofgpu::fv::{DivScheme, Limiter};
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::io::polymesh::{build_host_mesh, PolyMeshRaw};
use ofgpu::solver::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::timescheme::DdtScheme;
use ofgpu::{Gpu, GpuMesh, HostMesh, Result, Scalar, Vec3};

/// The process's device, if the box has one: every GPU test returns early
/// rather than failing.
fn dst10_gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// A `Scalar` widened to f64.
fn dst10_w64(x: Scalar) -> f64 {
    f64::from(x)
}

/// `n` cells on [0, len] in x, one cell of `side` in y and z; the x ends
/// `patch`, the four sides `symmetry`.
fn dst10_tube_raw(n: usize, len: f64, side: f64) -> PolyMeshRaw {
    let ax = |hi: f64, n: usize| GradedAxis {
        lo: 0.0,
        hi: hi as Scalar,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    let spec = BlockSpec {
        x: ax(len, n),
        y: ax(side, 1),
        z: ax(side, 1),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["patch", "patch", "symmetry", "symmetry", "symmetry", "symmetry"]
            .map(String::from),
        cyclic: Vec::new(),
    };
    blockgen::raw_mesh(&spec).expect("dst10_tube_raw")
}

/// A patch entry that states only a type.
fn dst10_entry(t: &str) -> PatchFieldSpec {
    PatchFieldSpec { type_name: t.into(), ..PatchFieldSpec::default() }
}

/// `U` fixed at rest on a patch.
fn dst10_fixed_u() -> PatchFieldSpec {
    PatchFieldSpec { type_name: "fixedValue".into(), value_v: vec![Vec3::ZERO], ..PatchFieldSpec::default() }
}

/// The tube's lowered fields: `U` at rest, gauge `p` and `T` from the cell
/// centre's x through `state(x) = (p_abs, T)`, about `gas.p_op`; the x ends
/// `U` fixed at rest, `p` and `T` zeroGradient (closed ends); the four sides
/// symmetry and zeroGradient.
fn dst10_lowered(hm: &HostMesh, gas: &PerfectGas, state: &dyn Fn(f64) -> (f64, f64)) -> Result<Lowered> {
    let mut u = RawVectorField { name: "U".into(), internal: vec![Vec3::ZERO; hm.n_cells], ..RawVectorField::default() };
    let pick = |k: usize| -> Vec<Scalar> {
        (0..hm.n_cells)
            .map(|i| {
                let s = state(dst10_w64(hm.c[i].x));
                if k == 0 { (s.0 - gas.p_op) as Scalar } else { s.1 as Scalar }
            })
            .collect()
    };
    let mut p = RawScalarField { name: "p".into(), internal: pick(0), ..RawScalarField::default() };
    let mut t = RawScalarField { name: "T".into(), internal: pick(1), ..RawScalarField::default() };
    for key in ["xMin", "xMax"] {
        u.boundary.insert(key.into(), dst10_fixed_u());
        p.boundary.insert(key.into(), dst10_entry("zeroGradient"));
        t.boundary.insert(key.into(), dst10_entry("zeroGradient"));
    }
    for key in ["yMin", "yMax", "zMin", "zMax"] {
        u.boundary.insert(key.into(), dst10_entry("symmetry"));
        p.boundary.insert(key.into(), dst10_entry("zeroGradient"));
        t.boundary.insert(key.into(), dst10_entry("zeroGradient"));
    }
    cbc::lower(&u, &p, &t, hm, gas)
}

/// Transient controls at `dt`: `gas`, van Leer on `U`, `T` and the retarded
/// density, Euler, PBiCGStab/DILU at 1e-12 on all three equations, no
/// relaxation, `n_outer` outer iterations, 2 correctors, `alpha_p` 1.
fn dst10_controls(gas: PerfectGas, dt: f64, n_outer: usize) -> CompressibleControls {
    let s = SolverControls {
        solver: LinearSolverKind::PBiCGStab,
        precon: Preconditioner::Dilu,
        tolerance: 1.0e-12,
        rel_tol: 0.0,
        max_iter: 200,
        ..SolverControls::default()
    };
    let mut c = CompressibleControls { gas, ..CompressibleControls::default() };
    c.momentum.u_solver = s;
    c.pressure.p_solver = s;
    c.energy.t_solver = s;
    c.momentum.ddt = DdtScheme::Euler;
    c.energy.ddt = DdtScheme::Euler;
    c.momentum.delta_t = dt as Scalar;
    c.energy.delta_t = dt as Scalar;
    c.momentum.div_scheme = DivScheme::Limited(Limiter::VanLeer);
    c.energy.div_scheme = DivScheme::Limited(Limiter::VanLeer);
    c.pressure.retarded = Retarded::Limited(Limiter::VanLeer);
    c.momentum.u_relax = 1.0;
    c.energy.t_relax = 1.0;
    c.n_outer = n_outer;
    c.n_corr = 2;
    c.p_relax = 1.0;
    c
}

/// FNV-1a 64 over the little-endian bytes of `to_bits() as u64` of every
/// value, in the order given.
fn dst10_fnv(h: &mut u64, vals: &[Scalar]) {
    for v in vals {
        for b in (v.to_bits() as u64).to_le_bytes() {
            *h ^= u64::from(b);
            *h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

/// The state's hash: `U` (x, y, z per value), `p`, `T`, `rho`, `psi`, `mu`,
/// `kappa` - each cells then boundary faces - then `mdot` internal then
/// boundary faces.
fn dst10_state_hash(g: &Gpu, s: &CompressibleState) -> Result<u64> {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let flat = |b: &DevBuf<Vec3>| -> Result<Vec<Scalar>> {
        Ok(g.download(b)?.into_iter().flat_map(|v| [v.x, v.y, v.z]).collect())
    };
    dst10_fnv(&mut h, &flat(&s.u.f)?);
    dst10_fnv(&mut h, &flat(&s.u.bf)?);
    for f in [&s.p, &s.t, &s.rho, &s.psi, &s.mu, &s.kappa] {
        dst10_fnv(&mut h, &g.download(&f.f)?);
        dst10_fnv(&mut h, &g.download(&f.bf)?);
    }
    dst10_fnv(&mut h, &g.download(&s.mdot.f)?);
    dst10_fnv(&mut h, &g.download(&s.mdot.bf)?);
    Ok(h)
}

/// The pin of `no_dust_no_change`, taken on 8b13cb2 (before DST-10).
#[cfg(not(feature = "single"))]
const DST10_PIN: u64 = 0xac0f_1e9c_5966_b505;
#[cfg(feature = "single")]
const DST10_PIN: u64 = 0x5340_40ad_c523_b30a;

/// **No dust, no change (D, SPEC-LIT §158.6):** the solver without a dust
/// attachment runs as it did before DST-10, bit for bit: a viscous,
/// conducting Sod tube (100 cells on 1 m, Sutherland air about 1e5 Pa,
/// p_abs 1e5 / 1e4 Pa and rho 1 / 0.125 kg/m3 either side of x = 0.5),
/// 40 Euler steps of 1e-5 s with 3 outer iterations; the FNV-1a 64 hash of
/// every state array after the last step equals the pin.
#[test]
fn no_dust_no_change() -> Result<()> {
    let Some(g) = dst10_gpu() else { return Ok(()) };
    let raw = dst10_tube_raw(100, 1.0, 0.01);
    let hm = build_host_mesh(&raw)?;
    let gm = GpuMesh::upload(&g, &hm)?;
    let gas = PerfectGas::air(1.0e5);
    let rs = gas.r_s();
    let sod = move |x: f64| -> (f64, f64) {
        let (p, rho) = if x < 0.5 { (1.0e5, 1.0) } else { (1.0e4, 0.125) };
        (p, p / (rho * rs))
    };
    let low = dst10_lowered(&hm, &gas, &sod)?;
    let mut solver = build_solver(&g, &gm, dst10_controls(gas, 1.0e-5, 3), &low)?;
    let mut st = setup_state(&g, &gm, &hm, &low)?;
    solver.initialise(&g, &mut st)?;
    for _ in 0..40 {
        solver.step(&g, &mut st)?;
    }
    let h = dst10_state_hash(&g, &st)?;
    println!("no_dust_no_change: hash {h:#018x}, pin {DST10_PIN:#018x}");
    assert_eq!(h, DST10_PIN, "no_dust_no_change: the hash moved");
    Ok(())
}

// ==========================================================================
//  DST-10's added rows (SPEC-LIT §158.6): the hConst pair, the sealed box's
//  AICC, and the SST's rho == 1 identity. Everything below carries the
//  `dst10_` prefix, like the pinned helpers above.
// ==========================================================================

use ofgpu::chem::thermo::R_GAS;
use ofgpu::compressible::thermo::DustHConst;
use ofgpu::dust::source::{compressible_sst_inputs, CompressibleDust, DustSetup};
use ofgpu::dust::thermo::{EquivalentGas, FormulaInputs, PrescribedInputs, tm4513_gas_set};
use ofgpu::field::{GpuScalarField, GpuSurfaceScalarField, GpuVectorField};
use ofgpu::field_ops::{FieldKernels, correct_boundary_conditions_vector};
use ofgpu::field_setup::{NutRoughness, WallFaces};
use ofgpu::fv::{FvKernels, SnGradScheme};
use ofgpu::io::case_dust::LoweredDust;
use ofgpu::io::case_json::read_case_jsonc_str;
use ofgpu::models::k_omega_sst::{KOmegaSst, KOmegaSstCoeffs, SstDensity};
use ofgpu::turbulence::{FlowState, TurbulenceControls};
use ofgpu::wallfunctions::WallFunctionCoeffs;
use ofgpu::{BcKind, Label};

/// A sealed 2 x 2 x 2 cube of 1 cm sides, every patch a wall.
fn dst10_box_raw() -> PolyMeshRaw {
    let ax = || GradedAxis {
        lo: 0.0,
        hi: 0.01 as Scalar,
        n: 2,
        expansion: 1.0,
        two_sided: false,
    };
    let spec = BlockSpec {
        x: ax(),
        y: ax(),
        z: ax(),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall"; 6].map(String::from),
        cyclic: Vec::new(),
    };
    blockgen::raw_mesh(&spec).expect("dst10_box_raw")
}

/// The box's lowered fields: `U` fixed at rest, gauge `p` and `T` uniform
/// (`p_abs`, `t`), `zeroGradient` on all six walls.
fn dst10_box_lowered(hm: &HostMesh, gas: &PerfectGas, p_abs: f64, t: f64) -> Result<Lowered> {
    let mut u = RawVectorField { name: "U".into(), internal: vec![Vec3::ZERO; hm.n_cells], ..RawVectorField::default() };
    let mut p = RawScalarField {
        name: "p".into(),
        internal: vec![(p_abs - gas.p_op) as Scalar; hm.n_cells],
        ..RawScalarField::default()
    };
    let mut tf = RawScalarField {
        name: "T".into(),
        internal: vec![t as Scalar; hm.n_cells],
        ..RawScalarField::default()
    };
    for key in ["xMin", "xMax", "yMin", "yMax", "zMin", "zMax"] {
        u.boundary.insert(key.into(), dst10_fixed_u());
        p.boundary.insert(key.into(), dst10_entry("zeroGradient"));
        tf.boundary.insert(key.into(), dst10_entry("zeroGradient"));
    }
    cbc::lower(&u, &p, &tf, hm, gas)
}

/// The uniform-ignition TFC pair of `src/bin/lowmach_dust/tests.rs`
/// (`TFC_DUST`): TFC at Pr_t 0.28 with the centre ignition, the formula
/// thermo at the REMBE charge phi = 0.62.
const DST10_TFC: &str = r#", "combustion": { "model": "TFC", "PrT": 0.28,
    "ignition": { "point": [0.005, 0.005, 0.005], "sigmaR": 1000, "W0": 1000, "t0": 1e-4 } },
  "dust": { "thermo": "formula", "phi": 0.62 }"#;

/// The sealed box as a JSONC text, lowered: the dust blocks above on the
/// same 2 x 2 x 2 cube at initial T 273 K, gauge p 0.
fn dst10_dust() -> LoweredDust {
    let text = format!(
        r#"{{
  "name": "dst10Box",
  "mesh": {{ "kind": "cartesian", "bounds": {{ "min": [0, 0, 0], "max": [0.01, 0.01, 0.01] }},
    "cells": [2, 2, 2], "boundaries": {{ "xmin": "wallX0", "xmax": "wallX1", "ymin": "wallY0",
      "ymax": "wallY1", "zmin": "wallZ0", "zmax": "wallZ1" }} }},
  "physics": {{ "gravity": [0, 0, 0], "fluid": {{ "nu": 1.5e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 293.15 }},
    "buoyancy": "densityRatio" }},
  "patches": [ {{ "match": ".*", "kind": "wall",
      "U": {{ "type": "fixedValue", "value": [0, 0, 0] }},
      "p": {{ "type": "zeroGradient" }}, "T": {{ "type": "zeroGradient" }} }} ],
  "initial": {{ "U": [0, 0, 0], "T": 273, "p": 0.0 }},
  "numerics": {{
    "algorithm": {{ "kind": "PIMPLE", "correctors": 2 }},
    "ddt": "Euler",
    "div": {{ "default": "Gauss upwind", "div(phi,U)": "Gauss upwind", "div(phi,T)": "bounded Gauss upwind" }},
    "grad": "Gauss linear",
    "laplacian": {{ "snGrad": "corrected", "nonOrthogonalCorrectors": 0 }},
    "relaxation": {{ "U": 1.0, "p": 1.0, "T": 1.0 }},
    "solvers": [
      {{ "match": "p", "solver": "PBiCGStab", "preconditioner": "DIC", "tolerance": 1e-13, "relTol": 0.0, "maxIter": 2000 }},
      {{ "match": "U", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": 1e-13, "relTol": 0.0, "maxIter": 2000 }},
      {{ "match": "T", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": 1e-13, "relTol": 0.0, "maxIter": 2000 }},
      {{ "match": "c", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": 1e-13, "relTol": 0.0, "maxIter": 2000 }}
    ]
  }},
  "run": {{ "endTime": 0.02, "deltaT": 1e-4 }}{DST10_TFC}
}}"#
    );
    read_case_jsonc_str(&text, "dst10Box")
        .and_then(|c| c.lower())
        .expect("lower")
        .dust
        .expect("dust config")
}

/// The c~ equation's controls at `dt`: Euler, tight diagonal solve, no
/// relaxation, uncorrected snGrad, van Leer convection (as T's) - after `progress_controls` of
/// `dust_source_cases.rs`, copied here under the dst10_ name.
fn dst10_progress_controls(dt: f64) -> TurbulenceControls {
    TurbulenceControls {
        k_solver: SolverControls {
            tolerance: 1e-14,
            rel_tol: 0.0,
            max_iter: 2000,
            check_interval: 1,
            ..SolverControls::default()
        },
        k_relax: 1.0,
        steady: false,
        delta_t: dt as Scalar,
        ddt: DdtScheme::Euler,
        sn_grad: SnGradScheme::Uncorrected,
        div_scheme: DivScheme::Limited(Limiter::VanLeer),
        ..TurbulenceControls::default()
    }
}

/// What one sealed-box run reports at its end, downloaded in f64.
struct Dst10BoxEnd {
    /// Pa, the volume mean of `p + p_op`.
    p_end: f64,
    /// K, the volume mean of `T`.
    t_end: f64,
    /// kg, `sum rho V` at the end.
    mass_end: f64,
    /// kg, `sum rho V` at the start.
    mass_start: f64,
    /// The smallest c~ on the cells.
    c_min: f64,
    /// The last solve's out-of-bounds count.
    out_of_bounds: u64,
    /// The hConst AICC's `P_max/p_0` (273 K).
    p_ratio_ref: f64,
    /// The hConst AICC's `T_v` (273 K).
    t_v_ref: f64,
}

/// One sealed-box run from scratch at `dt`: the TFC pair attached to a
/// `dst10_controls(gas, dt, 5)` solver, `steps` Euler steps, the end state
/// downloaded.
fn dst10_box_run(g: &Gpu, dt: f64, steps: usize) -> Result<Dst10BoxEnd> {
    let raw = dst10_box_raw();
    let hm = build_host_mesh(&raw)?;
    let gm = GpuMesh::upload(g, &hm)?;
    let gas = PerfectGas::air(101325.0);
    let low = dst10_box_lowered(&hm, &gas, 101325.0, 273.0)?;
    // The dust model at the energy equation's molecular conductivity.
    let kc = gas.constants();
    let k_mol = dst10_w64(kc.kappa(273.0 as Scalar));
    let d = dst10_dust();
    let setup = DustSetup::new(&d, 101325.0, k_mol)?;
    let nu_ref = dst10_w64(kc.mu(273.0 as Scalar)) / setup.rho_u0;
    // c~ seeded at 0, zeroGradient on every wall.
    let mut raw_c = RawScalarField {
        name: "c".into(),
        dimensions: "[0 0 0 0 0 0 0]".into(),
        internal: vec![0.0 as Scalar; hm.n_cells],
        ..RawScalarField::default()
    };
    for p in &hm.patches {
        raw_c
            .boundary
            .insert(p.name.clone(), dst10_entry("zeroGradient"));
    }
    let dust = CompressibleDust::new(
        g,
        &hm,
        &gm,
        setup,
        &raw_c,
        dst10_progress_controls(dt),
        None,
        nu_ref,
    )?;
    let mut solver = build_solver(g, &gm, dst10_controls(gas, dt, 5), &low)?;
    solver.attach_dust(g, dust)?;
    let mut st = setup_state(g, &gm, &hm, &low)?;
    solver.initialise(g, &mut st)?;
    let v: Vec<Scalar> = g.download(&gm.v)?;
    let r0: Vec<Scalar> = g.download(&st.rho.f)?;
    let mass_start: f64 =
        (0..hm.n_cells).map(|i| dst10_w64(v[i]) * dst10_w64(r0[i])).sum();
    for _ in 0..steps {
        solver.step(g, &mut st)?;
    }
    g.sync()?;
    let pf: Vec<Scalar> = g.download(&st.p.f)?;
    let tf: Vec<Scalar> = g.download(&st.t.f)?;
    let rf: Vec<Scalar> = g.download(&st.rho.f)?;
    let cf: Vec<Scalar> = g.download(&solver.dust().expect("dust").c().f)?;
    let (mut p_end, mut t_end, mut mass_end, mut vol, mut c_min) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, f64::INFINITY);
    for i in 0..hm.n_cells {
        let vi = dst10_w64(v[i]);
        p_end += (dst10_w64(pf[i]) + gas.p_op) * vi;
        t_end += dst10_w64(tf[i]) * vi;
        mass_end += dst10_w64(rf[i]) * vi;
        vol += vi;
        c_min = c_min.min(dst10_w64(cf[i]));
    }
    let dust = solver.dust().expect("dust");
    let (t_v_ref, p_ratio_ref) = dust.hconst().aicc(273.0);
    let out_of_bounds = dust.last_solve().map(|sv| sv.out_of_bounds).unwrap_or(0);
    Ok(Dst10BoxEnd {
        p_end: p_end / vol,
        t_end: t_end / vol,
        mass_end,
        mass_start,
        c_min,
        out_of_bounds,
        p_ratio_ref,
        t_v_ref,
    })
}

/// **The sealed box ends at the AICC (P, SPEC-LIT §158.6 row 1):** the box
/// of `dst10_box_raw` with the TFC pair attached burns uniformly to c~ = 1,
/// and `p_end/p_init` is the hConst AICC ratio within 0.5 %, first order in
/// dt. MISSED rows print and still pass.
#[test]
fn dust_box_ends_at_aicc() -> Result<()> {
    let Some(g) = dst10_gpu() else { return Ok(()) };
    let coarse = dst10_box_run(&g, 1.0e-4, 200)?;
    let fine = dst10_box_run(&g, 5.0e-5, 400)?;
    for (tag, end) in [("dt=1e-4", &coarse), ("dt=5e-5", &fine)] {
        assert!(
            end.c_min >= 1.0 - 1e-6,
            "{tag}: c~ min {} did not burn through",
            end.c_min
        );
        assert_eq!(
            end.out_of_bounds, 0,
            "{tag}: c~ left [0, 1] on {} cells",
            end.out_of_bounds
        );
    }
    let e = |end: &Dst10BoxEnd| (end.p_end / 101325.0 / end.p_ratio_ref - 1.0).abs();
    let (e1, e2) = (e(&coarse), e(&fine));
    let ratio = e1 / e2;
    println!(
        "dust_box_ends_at_aicc: measured e(dt=1e-4) {e1:.3e}, e(dt=5e-5) {e2:.3e}, bar 0.005, \
         ratio {ratio:.3}, bar [1.8, 2.2]"
    );
    println!(
        "dust_box_ends_at_aicc: T_end {:.4} K against T_v {:.4} K, mass {:.6e} -> {:.6e} kg, \
         AICC p_ratio {:.9}",
        coarse.t_end, coarse.t_v_ref, coarse.mass_start, fine.mass_end, coarse.p_ratio_ref
    );
    if e1 > 0.005 || e2 > 0.005 {
        println!(
            "MISSED box AICC (P): measured e(dt=1e-4) {e1:.3e}, e(dt=5e-5) {e2:.3e}, bar 0.005, "
        );
    }
    if !(1.8..=2.2).contains(&ratio) {
        println!(
            "MISSED box AICC first order in dt (P): measured e(dt=1e-4) {e1:.3e}, \
             e(dt=5e-5) {e2:.3e}, ratio {ratio:.3}, bar [1.8, 2.2], "
        );
    }
    Ok(())
}

/// **The hConst pair is the gas's own (host f64, both builds):** at the
/// prescribed (constant c_p) Leeds gas the hConst AICC ratio and isobaric
/// `T_b` are the NASA-7 ones to 1e-12 relative; at the formula REMBE gas
/// `cp_u` is exact, the isobaric `T_b` is kept, Δh_c is positive and the
/// closed-form AICC satisfies its own definition (152.11).
#[test]
fn dust_hconst_matches_gas() -> Result<()> {
    // (a) The prescribed gas: one constant c_p, so the hConst identity is
    // the NASA-7 one to round-off.
    let gas = EquivalentGas::prescribed(&PrescribedInputs::huang2020_leeds(1200.0))?;
    let hc = DustHConst::from_gas(&gas, 328.0)?;
    let (_, aicc_r) = hc.aicc(328.0);
    let gas_r = gas.aicc(328.0)?.p_ratio;
    let gas_tb = gas.isobaric_burned(328.0)?.t_b;
    let e_r = (aicc_r - gas_r).abs() / gas_r.abs();
    let e_tb = (hc.isobaric_tb(328.0) - gas_tb).abs() / gas_tb.abs();
    println!(
        "dust_hconst_matches_gas: prescribed AICC ratio err {e_r:.3e}, isobaric T_b err \
         {e_tb:.3e}, bar 1e-12"
    );
    assert!(e_r <= 1e-12, "prescribed: AICC ratio {aicc_r} vs {gas_r}");
    assert!(e_tb <= 1e-12, "prescribed: T_b {} vs {}", hc.isobaric_tb(328.0), gas_tb);

    // (b) The formula REMBE gas at T_u = 273 K: cp_u exact, the NASA-7
    // isobaric T_b kept, Δh_c positive, the closed-form AICC self-consistent.
    let gas = EquivalentGas::formula(&FormulaInputs::huang2022_rembe()?, &tm4513_gas_set()?)?;
    let hc = DustHConst::from_gas(&gas, 273.0)?;
    assert_eq!(hc.cp_u, gas.cp_u(273.0)?, "cp_u is the gas's own cp_u(273)");
    let e_tb = (hc.isobaric_tb(273.0) - gas.isobaric_burned(273.0)?.t_b).abs()
        / gas.isobaric_burned(273.0)?.t_b.abs();
    assert!(e_tb <= 1e-12, "REMBE: T_b {} off by {e_tb:.3e}", hc.isobaric_tb(273.0));
    assert!(hc.delta_h_c() > 0.0, "delta_h_c = {} is not positive", hc.delta_h_c());
    let (t_v, p_ratio) = hc.aicc(273.0);
    let u_u = hc.a_u + (hc.cp_u - R_GAS / hc.w_u) * 273.0;
    let u_b = hc.a_b + (hc.cp_b - R_GAS / hc.w_b) * t_v;
    assert!(
        (u_b - u_u).abs() <= 1e-9 * u_u.abs(),
        "REMBE: u_B(t_v) = {u_b} vs u_U(273) = {u_u}"
    );
    println!(
        "dust_hconst_matches_gas: REMBE at 273 K, AICC p_ratio {p_ratio:.9}, t_v {t_v:.4} K, \
         isobaric T_b err {e_tb:.3e}, bar 1e-12"
    );
    Ok(())
}

/// A uniform 2-D block, `nz = 1` with `empty` front and back, every x and y
/// patch `patch`: the interpolation weights are exactly 1/2.
fn dst10_sst_mesh(g: &Gpu, n: [usize; 2]) -> Result<(HostMesh, GpuMesh)> {
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
        patch_type: ["patch", "patch", "patch", "patch", "empty", "empty"].map(String::from),
        cyclic: Vec::new(),
    };
    let hm = blockgen::build_mesh(&spec)?;
    let gm = GpuMesh::upload(g, &hm)?;
    Ok((hm, gm))
}

/// Transient, unrelaxed, solved to the floor, Euler and upwind.
fn dst10_sst_controls(dt: Scalar) -> TurbulenceControls {
    let mut ctrl = TurbulenceControls {
        steady: false,
        delta_t: dt,
        k_relax: 1.0,
        eps_relax: 1.0,
        nut_max_coeff: 1e8,
        ..Default::default()
    };
    ctrl.ddt = DdtScheme::Euler;
    ctrl.div_scheme = DivScheme::Upwind;
    ctrl.eps_div_scheme = DivScheme::Upwind;
    ctrl.bounded_convection = false;
    ctrl.eps_bounded_convection = false;
    ctrl.k_solver.tolerance = if cfg!(feature = "single") { 1e-12 } else { 1e-14 };
    ctrl.k_solver.rel_tol = 0.0;
    ctrl.k_solver.report_residuals = false;
    ctrl.epsilon_solver = ctrl.k_solver;
    ctrl
}

/// The sheared channel velocity in the cells and as fixedValue on every
/// non-empty face, its volume flux `phi`, and `mdot` - here the same flux,
/// the rho form's `FlowState` input at rho = 1.
fn dst10_sst_flow(
    g: &Gpu,
    hm: &HostMesh,
    gm: &GpuMesh,
) -> Result<(GpuVectorField, GpuSurfaceScalarField, GpuSurfaceScalarField)> {
    let dot = |a: Vec3, b: Vec3| a.x * b.x + a.y * b.y + a.z * b.z;
    let u_at = |c: Vec3| Vec3::new(4.0 * c.y * (1.0 - c.y) + 0.3 * c.x, 0.1 * c.x * c.y, 0.0);
    let (nif, nbf) = (hm.n_internal_faces, hm.n_boundary_faces);
    let mut u = GpuVectorField::zeros(g, gm, "U")?;
    let uc: Vec<Vec3> = hm.c.iter().map(|&c| u_at(c)).collect();
    g.write(&mut u.f, &uc)?;
    g.write(&mut u.f0, &uc)?;
    g.write(&mut u.f00, &uc)?;
    let mut kind: Vec<Label> = g.download(&u.bc_kind)?;
    let mut fr = vec![0.0 as Scalar; nbf];
    let mut rv = vec![Vec3::ZERO; nbf];
    for i in 0..nbf {
        if kind[i] != BcKind::Empty as Label {
            kind[i] = BcKind::FixedValue as Label;
            fr[i] = 1.0;
            rv[i] = u_at(hm.b_cf[i]);
        }
    }
    g.write(&mut u.bc_kind, &kind)?;
    g.write(&mut u.fr, &fr)?;
    g.write(&mut u.ref_value, &rv)?;
    let fld = FieldKernels::new(g)?;
    correct_boundary_conditions_vector(g, &fld, &mut u, gm)?;
    let phi_i: Vec<Scalar> = (0..nif).map(|f| dot(u_at(hm.cf[f]), hm.sf[f])).collect();
    let phi_b: Vec<Scalar> = (0..nbf).map(|i| dot(u_at(hm.b_cf[i]), hm.b_sf[i])).collect();
    let mut phi = GpuSurfaceScalarField::zeros(g, gm, "phi")?;
    g.write(&mut phi.f, &phi_i)?;
    g.write(&mut phi.bf, &phi_b)?;
    let mut mdot = GpuSurfaceScalarField::zeros(g, gm, "mdot")?;
    g.write(&mut mdot.f, &phi_i)?;
    g.write(&mut mdot.bf, &phi_b)?;
    Ok((u, phi, mdot))
}

/// A model with the given initial `k(x)`, `omega(x)`, no wall faces, the
/// given wall distance and flow state.
fn dst10_sst_model<'m>(
    g: &Gpu,
    hm: &HostMesh,
    gm: &'m GpuMesh,
    ctrl: TurbulenceControls,
    y: &[Scalar],
    flow: &FlowState<'_>,
) -> Result<KOmegaSst<'m>> {
    let yb = g.upload(y)?;
    let rough = NutRoughness::none(hm.n_boundary_faces);
    let mut m = KOmegaSst::new(
        g,
        hm,
        gm,
        KOmegaSstCoeffs::default(),
        ctrl,
        WallFunctionCoeffs::default(),
        &WallFaces::none(hm.n_boundary_faces),
        &yb,
        &rough,
    )?;
    let k: Vec<Scalar> = hm.c.iter().map(|c| 1.0e-3 * (1.0 + 0.5 * c.x + 0.3 * c.y * c.y)).collect();
    let w: Vec<Scalar> = hm.c.iter().map(|c| 2.0 + 48.0 * c.x + 5.0 * c.y).collect();
    g.write(&mut m.k_mut().f, &k)?;
    g.write(&mut m.omega_mut().f, &w)?;
    m.initialise(g, flow)?;
    Ok(m)
}

/// Every field the bitwise row compares: k, omega and nu_t, cells and
/// boundary faces.
fn dst10_sst_snapshot(g: &Gpu, m: &KOmegaSst<'_>) -> Result<Vec<(&'static str, Vec<Scalar>)>> {
    g.sync()?;
    Ok(vec![
        ("k", g.download(&m.k().f)?),
        ("omega", g.download(&m.omega().f)?),
        ("nut", g.download(&m.nut().f)?),
        ("nut boundary", g.download(&m.nut().bf)?),
    ])
}

/// **kOmegaSST at rho == 1 is the kinematic SST, bit for bit (D, SPEC-LIT
/// §158.6 row 5):** two identical models on a uniform mesh (every
/// interpolation weight exactly 1/2), the same sheared velocity; model A
/// runs the kinematic `correct`, model B `correct_rho` on the
/// `SstDensity` `compressible_sst_inputs` builds from rho = 1 and
/// `mdot = phi`. After each of 3 steps k, omega and nu_t (cells and
/// boundary faces) are equal bitwise.
#[test]
fn sst_rho_one_bitwise() -> Result<()> {
    let Some(g) = dst10_gpu() else { return Ok(()) };
    let (hm, gm) = dst10_sst_mesh(&g, [6, 6])?;
    let y: Vec<Scalar> = hm.c.iter().map(|c| c.y.min(1.0 - c.y)).collect();
    let (u, phi, mdot) = dst10_sst_flow(&g, &hm, &gm)?;
    let nu: Scalar = 1e-5;
    let kin_flow = FlowState::new(&u, &phi, nu);
    let rho_flow = FlowState::new(&u, &mdot, nu);
    let dt: Scalar = 1.0 / 1024.0;
    let ctrl = dst10_sst_controls(dt);
    let mut kin = dst10_sst_model(&g, &hm, &gm, ctrl, &y, &kin_flow)?;
    let mut rhom = dst10_sst_model(&g, &hm, &gm, ctrl, &y, &rho_flow)?;
    // The rho form's density: 1 on every cell, boundary face and old level.
    let one_c = vec![1.0 as Scalar; hm.n_cells];
    let one_b = vec![1.0 as Scalar; hm.n_boundary_faces];
    let mut rho = GpuScalarField::zeros(&g, &gm, "rho")?;
    g.write(&mut rho.f, &one_c)?;
    g.write(&mut rho.f0, &one_c)?;
    g.write(&mut rho.f00, &one_c)?;
    g.write(&mut rho.bf, &one_b)?;
    let mut rho_face = GpuSurfaceScalarField::zeros(&g, &gm, "rhoF")?;
    let mut phi_volume = GpuSurfaceScalarField::zeros(&g, &gm, "phiV")?;
    {
        let fvk = FvKernels::new(&g)?;
        let fldk = FieldKernels::new(&g)?;
        compressible_sst_inputs(&g, &fvk, &fldk, &gm, &rho, &mdot, &mut rho_face, &mut phi_volume)?;
    }
    let dens = SstDensity {
        rho: &rho.f,
        rho0: &rho.f0,
        rho00: &rho.f00,
        rho_face: &rho_face,
        phi_volume: &phi_volume,
    };
    dst10_sst_bitwise(&g, &kin, &rhom, 0)?;
    for step in 1..=3 {
        kin.correct(&g, &kin_flow)?;
        rhom.correct_rho(&g, &rho_flow, &dens)?;
        dst10_sst_bitwise(&g, &kin, &rhom, step)?;
    }
    println!(
        "sst_rho_one_bitwise: measured k, omega, nu_t bitwise equal over 3 steps, bar bitwise, \
         {} cells",
        hm.n_cells
    );
    Ok(())
}

/// The bitwise comparison of one step's two models.
fn dst10_sst_bitwise(
    g: &Gpu,
    a: &KOmegaSst<'_>,
    b: &KOmegaSst<'_>,
    step: usize,
) -> Result<()> {
    let sa = dst10_sst_snapshot(g, a)?;
    let sb = dst10_sst_snapshot(g, b)?;
    for ((name, x), (_, yv)) in sa.iter().zip(sb.iter()) {
        assert_eq!(x.len(), yv.len(), "step {step}: {name} lengths");
        for (i, (p, q)) in x.iter().zip(yv.iter()).enumerate() {
            assert!(
                p.to_bits() == q.to_bits(),
                "step {step}: {name}[{i}] kinematic {p:e} vs rho form {q:e}"
            );
        }
    }
    Ok(())
}

// ==========================================================================
//  DST-10 part 2 (SPEC-LIT §158.6 rows 2-3): the 1-D flame tube - the
//  deflagration jump conditions (158.2) and the acoustic precursor
//  (158.3). Both rows read ONE run per build, cached below.
// ==========================================================================

use std::sync::OnceLock;

use ofgpu::dust::vessel0d::BurningLaw;

/// What the tube run reports at its end, every field downloaded in f64 per
/// cell, with the c~ bookkeeping of the steps and the constants the rows
/// read.
#[derive(Clone)]
struct Dst10TubeEnd {
    /// m, the cell centres' x.
    x: Vec<f64>,
    /// Pa, the gauge p.
    p: Vec<f64>,
    /// m/s, u_x.
    ux: Vec<f64>,
    /// kg/m3.
    rho: Vec<f64>,
    /// K.
    t: Vec<f64>,
    /// The progress variable c~.
    c: Vec<f64>,
    /// The max c~ over every step.
    c_max: f64,
    /// The sum of the c~ bounds counts over every step.
    bounds: u64,
    /// kg/m3, rho_u0 of (152.9) at p_init.
    rho_u0: f64,
    /// m/s, S_L (152.13).
    s_l: f64,
    /// The hConst pair's gamma_u, (155.10)'s exponent.
    gamma_u: f64,
    /// J/(kg K), the hConst pair's R_U = R0/W_U.
    r_u: f64,
    /// K, T_u.
    t_u: f64,
}

/// A downloaded value must be finite: a NaN or an inf panics with its name.
fn dst10_finite(v: f64, what: &str) -> f64 {
    assert!(v.is_finite(), "dst10_flame_tube: {what} is not finite: {v:e}");
    v
}

/// The tube's dust case as a JSONC text, lowered: the formula REMBE gas at
/// phi 0.62 with TFC at Pr_t 0.28 and NO ignition, initial T 300 K, on the
/// same 400 x 1 x 1 tube the run itself meshes. The prescribed constant
/// `U_t = S_L` is set on the lowered struct by `dst10_flame_tube`, not in
/// the case text.
fn dst10_tube_dust() -> LoweredDust {
    let text = format!(
        r#"{{
  "name": "dst10Tube",
  "mesh": {{ "kind": "cartesian", "bounds": {{ "min": [0, 0, 0], "max": [0.4, 0.001, 0.001] }},
    "cells": [400, 1, 1], "boundaries": {{ "xmin": "tubeX0", "xmax": "tubeX1", "ymin": "tubeY0",
      "ymax": "tubeY1", "zmin": "tubeZ0", "zmax": "tubeZ1" }} }},
  "physics": {{ "gravity": [0, 0, 0], "fluid": {{ "nu": 1.5e-5, "Pr": 0.71, "Prt": 0.85, "TRef": 300 }},
    "buoyancy": "densityRatio" }},
  "patches": [ {{ "match": ".*", "kind": "wall",
      "U": {{ "type": "fixedValue", "value": [0, 0, 0] }},
      "p": {{ "type": "zeroGradient" }}, "T": {{ "type": "zeroGradient" }} }} ],
  "initial": {{ "U": [0, 0, 0], "T": 300, "p": 0.0 }},
  "numerics": {{
    "algorithm": {{ "kind": "PIMPLE", "correctors": 2 }},
    "ddt": "Euler",
    "div": {{ "default": "Gauss upwind", "div(phi,U)": "Gauss upwind", "div(phi,T)": "bounded Gauss upwind" }},
    "grad": "Gauss linear",
    "laplacian": {{ "snGrad": "corrected", "nonOrthogonalCorrectors": 0 }},
    "relaxation": {{ "U": 1.0, "p": 1.0, "T": 1.0 }},
    "solvers": [
      {{ "match": "p", "solver": "PBiCGStab", "preconditioner": "DIC", "tolerance": 1e-13, "relTol": 0.0, "maxIter": 2000 }},
      {{ "match": "U", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": 1e-13, "relTol": 0.0, "maxIter": 2000 }},
      {{ "match": "T", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": 1e-13, "relTol": 0.0, "maxIter": 2000 }},
      {{ "match": "c", "solver": "PBiCGStab", "preconditioner": "diagonal", "tolerance": 1e-13, "relTol": 0.0, "maxIter": 2000 }}
    ]
  }},
  "run": {{ "endTime": 1.5e-3, "deltaT": 1e-6 }},
  "combustion": {{ "model": "TFC", "PrT": 0.28 }},
  "dust": {{ "thermo": "formula", "phi": 0.62 }}
}}"#
    );
    read_case_jsonc_str(&text, "dst10Tube")
        .and_then(|c| c.lower())
        .expect("lower")
        .dust
        .expect("dust config")
}

/// The one tube run both rows read (SPEC-LIT §158.6 rows 2-3): 800 cells on
/// [0, 0.8] m, BOTH x ends closed (U fixed at rest, p and T zeroGradient),
/// the sides symmetry; Sutherland air for mu and kappa; the formula REMBE
/// gas at phi 0.62 with TFC at Pr_t 0.28 and the prescribed constant
/// `U_t = S_L`, no ignition; `dst10_controls(gas, dt, 10)` (10 outer
/// iterations: 3 leave the expansion 9 % short, 10 and 20 agree, §158.9),
/// the c~ controls of the box (van Leer, as T); Euler steps of `dt` to
/// t_end = 1.5e-3 s. The initial state is
/// the smooth flame already on the hConst adiabatic line: c~ = 1 - (3 xi^2
/// - 2 xi^3) about x = 0.03 m, T on h(T, c~) = h_U(T_u) of (152.10), at
/// rest, gauge p 0. Cached per build - the run is well above the 30 s line.
fn dst10_flame_tube(g: &Gpu, dt: f64) -> Result<Dst10TubeEnd> {
    static CACHE: OnceLock<Dst10TubeEnd> = OnceLock::new();
    if let Some(hit) = CACHE.get() {
        println!("dst10_flame_tube: cache hit, the run is not repeated");
        return Ok(hit.clone());
    }
    let started = std::time::Instant::now();
    let raw = dst10_tube_raw(800, 0.8, 0.001);
    let hm = build_host_mesh(&raw)?;
    let gm = GpuMesh::upload(g, &hm)?;
    let gas = PerfectGas::air(101325.0);
    let kc = gas.constants();
    let k_mol = dst10_w64(kc.kappa(300.0 as Scalar));
    let mut d = dst10_tube_dust();
    d.prescribed_ut = Some(BurningLaw::constant(d.s_l));
    let setup = DustSetup::new(&d, 101325.0, k_mol)?;
    let (rho_u0, s_l, t_u) = (setup.rho_u0, setup.s_l, setup.t_u);
    let hc = DustHConst::from_gas(&setup.gas, setup.t_u)?;
    let (gamma_u, r_u) = (hc.gamma_u(), hc.r_u());
    // The smoothstep front about x = 0.03 m, and T on the adiabatic line.
    let smooth = |x: f64| -> f64 {
        let xi = ((x - 0.02) / 0.02).clamp(0.0, 1.0);
        3.0 * xi * xi - 2.0 * xi * xi * xi
    };
    let state = |x: f64| -> (f64, f64) {
        let c = 1.0 - smooth(x);
        let t = (hc.cp_u * t_u + c * (hc.a_u - hc.a_b)) / (hc.cp_u + c * (hc.cp_b - hc.cp_u));
        (101325.0, t)
    };
    let low = dst10_lowered(&hm, &gas, &state)?;

    let nu_ref = dst10_w64(kc.mu(300.0 as Scalar)) / rho_u0;
    // c~ seeded with the same smoothstep, zeroGradient on every patch.
    let mut raw_c = RawScalarField {
        name: "c".into(),
        dimensions: "[0 0 0 0 0 0 0]".into(),
        internal: (0..hm.n_cells)
            .map(|i| (1.0 - smooth(dst10_w64(hm.c[i].x))) as Scalar)
            .collect(),
        ..RawScalarField::default()
    };
    for p in &hm.patches {
        raw_c
            .boundary
            .insert(p.name.clone(), dst10_entry("zeroGradient"));
    }
    let dust = CompressibleDust::new(
        g,
        &hm,
        &gm,
        setup,
        &raw_c,
        dst10_progress_controls(dt),
        None,
        nu_ref,
    )?;
    let mut solver = build_solver(g, &gm, dst10_controls(gas, dt, 10), &low)?;
    solver.attach_dust(g, dust)?;
    let mut st = setup_state(g, &gm, &hm, &low)?;
    solver.initialise(g, &mut st)?;
    // The c~ budget's datum on the host initial state (information): the
    // mass per area and M_c = sum rho c~ dx, both per the mixture law, to
    // be compared with the same integrals of the end state - with both x
    // ends closed and u = 0 there, d M_c/dt is the burning rate.
    let (r_u_c, r_b_c) = (hc.r_u(), hc.r_b());
    let line = |c: f64| -> f64 {
        let t = (hc.cp_u * t_u + c * (hc.a_u - hc.a_b)) / (hc.cp_u + c * (hc.cp_b - hc.cp_u));
        101325.0 / (r_u_c + c * (r_b_c - r_u_c)) / t
    };
    let dx = dst10_w64(hm.c[1].x - hm.c[0].x);
    let (mut mass0, mut mc0) = (0.0_f64, 0.0_f64);
    for i in 0..hm.n_cells {
        let c = 1.0 - smooth(dst10_w64(hm.c[i].x));
        mass0 += line(c);
        mc0 += line(c) * c;
    }
    let (mass0, mc0) = (mass0 * dx, mc0 * dx);
    let steps = (1.5e-3 / dt).round() as usize;
    let (mut c_max, mut bounds) = (0.0_f64, 0_u64);
    for _ in 0..steps {
        solver.step(g, &mut st)?;
        let dust = solver.dust().expect("dust attached");
        if let Some(sv) = dust.last_solve() {
            bounds += sv.out_of_bounds;
        }
        let cf: Vec<Scalar> = g.download(&dust.c().f)?;
        for v in cf {
            let v = dst10_finite(dst10_w64(v), "c~");
            c_max = c_max.max(v);
        }
    }

    g.sync()?;
    let pf: Vec<Scalar> = g.download(&st.p.f)?;
    let tf: Vec<Scalar> = g.download(&st.t.f)?;
    let rf: Vec<Scalar> = g.download(&st.rho.f)?;
    let uf: Vec<Vec3> = g.download(&st.u.f)?;
    let cf: Vec<Scalar> = g.download(&solver.dust().expect("dust attached").c().f)?;
    let end = Dst10TubeEnd {
        x: (0..hm.n_cells).map(|i| dst10_finite(dst10_w64(hm.c[i].x), "x")).collect(),
        p: pf.iter().map(|&v| dst10_finite(dst10_w64(v), "p")).collect(),
        ux: uf.iter().map(|v| dst10_finite(dst10_w64(v.x), "u_x")).collect(),
        rho: rf.iter().map(|&v| dst10_finite(dst10_w64(v), "rho")).collect(),
        t: tf.iter().map(|&v| dst10_finite(dst10_w64(v), "T")).collect(),
        c: cf.iter().map(|&v| dst10_finite(dst10_w64(v), "c~")).collect(),
        c_max,
        bounds,
        rho_u0,
        s_l,
        gamma_u,
        r_u,
        t_u,
    };
    let ran = started.elapsed().as_secs_f64();
    let (mass1, mc1): (f64, f64) =
        end.rho.iter().zip(end.c.iter()).map(|(&r, &c)| (r, r * c)).fold(
            (0.0, 0.0),
            |(m, mc), (r, rc)| (m + r, mc + rc),
        );
    let (mass1, mc1) = (mass1 * dx, mc1 * dx);
    let tv: f64 = (1..end.c.len()).map(|i| (end.c[i] - end.c[i - 1]).abs()).sum();
    let burn = (mc1 - mc0) / (steps as f64 * dt);
    println!(
        "dst10_budget: mass/area {mass0:.9e} -> {mass1:.9e} kg/m2, M_c {mc0:.9e} -> \
         {mc1:.9e} kg/m2, dM_c/dt {burn:.4e} against rho_u0 S_L {:.4e} kg/m2/s, \
         TV(c~) {tv:.4}",
        rho_u0 * s_l
    );
    println!(
        "dst10_flame_tube: {} cells, {steps} steps of dt {dt:e} to 1.5e-3 s in {ran:.1} s, \
         max c~ over the steps {c_max:.6}, sum of the c~ bounds counts {bounds}",
        end.x.len(),
    );
    Ok(CACHE.get_or_init(|| end).clone())
}

/// The zones of both rows, from the run's arrays: the wave front `x_w` the
/// LAST cell centre with p > 0.5 max p; the plateau P = [0.06, 0.25] (x_w must pass 0.45 m);
/// the burned zone B = [0.002, 0.015]. Arithmetic means over each zone's
/// cells (uniform mesh); a zone below five cells panics - the run did not
/// develop as designed.
struct Dst10Zones {
    /// m, the wave front.
    x_w: f64,
    /// Pa, the mean gauge p over P - dp_wave, the rise over the initial.
    p_u: f64,
    /// m/s, the mean u_x over P.
    u_u: f64,
    /// kg/m3, the mean rho over P.
    rho_u: f64,
    /// Pa, the mean gauge p over B.
    p_b: f64,
    /// kg/m3, the mean rho over B.
    rho_b: f64,
}

fn dst10_zones(e: &Dst10TubeEnd) -> Dst10Zones {
    let n = e.x.len();
    let p_max = e.p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(p_max > 0.0, "dst10_zones: max p {p_max} is not above 0");
    let mut x_w = e.x[0];
    for i in 0..n {
        if e.p[i] > 0.5 * p_max {
            x_w = e.x[i];
        }
    }
    let zone = |lo: f64, hi: f64, what: &str| -> (f64, f64, f64, usize) {
        let (mut sp, mut su, mut sr, mut k) = (0.0_f64, 0.0_f64, 0.0_f64, 0_usize);
        for i in 0..n {
            if e.x[i] >= lo && e.x[i] <= hi {
                sp += e.p[i];
                su += e.ux[i];
                sr += e.rho[i];
                k += 1;
            }
        }
        assert!(
            k >= 5,
            "dst10_zones: {what} holds {k} cells on [{lo}, {hi}], fewer than 5"
        );
        (sp / k as f64, su / k as f64, sr / k as f64, k)
    };
    // The front is spread over about 0.2 m by 1.5 ms (measured, SPEC-LIT
    // §158.9), so the plateau P ends 0.25 m behind the 50 % point: P =
    // [0.06, 0.25] m, and that point must have passed 0.45 m.
    assert!(x_w >= 0.45, "dst10_zones: the wave front x_w {x_w} m has not passed 0.45 m");
    let (p_u, u_u, rho_u, n_p) = zone(0.06, 0.25, "the plateau P");
    let (p_b, _, rho_b, _) = zone(0.002, 0.015, "the burned zone B");
    println!(
        "dst10_zones: x_w {x_w:.4} m, P holds {n_p} cells on [0.06, 0.25], B holds its \
         cells on [0.002, 0.015]; c~ max over the run {:.6}, c~ bounds counts {}",
        e.c_max, e.bounds
    );
    // Information only, naming the cell: the state every 10th cell, and
    // every cell of the front brush, so a missed bar can be read against
    // the profile that produced it.
    for i in 0..n {
        if i % 10 == 0 || (e.x[i] >= 0.018 && e.x[i] <= 0.048) {
            println!(
                "dst10_profile: i {i:3}, x {:.4}, p {:9.3}, u_x {:8.4}, rho {:.5}, \
                 T {:9.3}, c~ {:+.6}",
                e.x[i], e.p[i], e.ux[i], e.rho[i], e.t[i], e.c[i]
            );
        }
    }
    Dst10Zones { x_w, p_u, u_u, rho_u, p_b, rho_b }
}

/// **The deflagration jump conditions (P, SPEC-LIT §158.6 row 2, (158.2)):**
/// with sigma = rho_u/rho_b from the plateau P and the burned zone B, the
/// fresh gas moves at u_u = (sigma - 1) S_L within 1 % and p_u - p_b =
/// rho_u S_L^2 (sigma - 1) within 2 %. A missed bar prints MISSED and the
/// test still passes.
#[test]
fn deflagration_jump_conditions() -> Result<()> {
    let Some(g) = dst10_gpu() else { return Ok(()) };
    let e = dst10_flame_tube(&g, 1.0e-6)?;
    let z = dst10_zones(&e);
    let sigma = z.rho_u / z.rho_b;
    let u_pred = (sigma - 1.0) * e.s_l;
    let dp = z.p_u - z.p_b;
    let dp_pred = z.rho_u * e.s_l * e.s_l * (sigma - 1.0);
    let (e_u, e_dp) = ((z.u_u / u_pred - 1.0).abs(), (dp / dp_pred - 1.0).abs());
    println!(
        "deflagration_jump_conditions: x_w {:.4} m, sigma {:.4}, u_u {:.4} m/s against \
         (sigma-1) S_L {:.4} m/s, p_u - p_b {:.4} Pa against rho_u S_L^2 (sigma-1) \
         {:.4} Pa, mass error {e_u:.3e} (bar 0.01), momentum error {e_dp:.3e} (bar 0.02)",
        z.x_w, sigma, z.u_u, u_pred, dp, dp_pred
    );
    if e_u > 0.01 {
        println!(
            "MISSED jump mass (P): measured u_u {:.4} m/s, prediction {:.4} m/s, bar 0.01, ",
            z.u_u, u_pred
        );
    }
    if e_dp > 0.02 {
        println!(
            "MISSED jump momentum (P): measured p_u - p_b {:.4} Pa, prediction {:.4} Pa, \
             bar 0.02, ",
            dp, dp_pred
        );
    }
    Ok(())
}

/// **The acoustic precursor (P, SPEC-LIT §158.6 row 3, (158.3)):** the
/// plateau's gauge rise dp_wave = p_u against `rho_u0 c_u0 (sigma - 1)
/// S_L` with `c_u0 = sqrt(gamma_u R_U T_u)` of the hConst pair and sigma
/// from row 2's zones, within 2 %. The measured wave speed
/// `(x_w - 0.03)/t_end` against c_u0 is information, no bar.
#[test]
fn acoustic_precursor() -> Result<()> {
    let Some(g) = dst10_gpu() else { return Ok(()) };
    let e = dst10_flame_tube(&g, 1.0e-6)?;
    let z = dst10_zones(&e);
    let sigma = z.rho_u / z.rho_b;
    let c_u0 = (e.gamma_u * e.r_u * e.t_u).sqrt();
    let pred = e.rho_u0 * c_u0 * (sigma - 1.0) * e.s_l;
    let err = (z.p_u / pred - 1.0).abs();
    let wave_speed = (z.x_w - 0.03) / 1.5e-3;
    println!(
        "acoustic_precursor: dp_wave {:.4} Pa against rho_u0 c_u0 (sigma-1) S_L {:.4} Pa, \
         error {err:.3e} (bar 0.02), c_u0 {c_u0:.2} m/s, measured wave speed \
         (x_w - 0.03)/t_end {wave_speed:.2} m/s",
        z.p_u, pred
    );
    if err > 0.02 {
        println!(
            "MISSED precursor (P): measured dp_wave {:.4} Pa, prediction {:.4} Pa, \
             bar 0.02, ",
            z.p_u, pred
        );
    }
    Ok(())
}
