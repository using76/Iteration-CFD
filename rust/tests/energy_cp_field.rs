// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 133.8 (CHR-10a): `Energy` carries a per-cell `cp` field in its
//! ddt row (133.13) and its convection row (133.14), in BOTH builds.
//!
//! * `energy_cp_field_default_bitwise` (class D): a case that sets no field
//!   is the constant-`cp` path in every bit. The shipped
//!   `cases/channelPeriodicFluxWF.jsonc` runs as `ofgpu-lowmach` itself for
//!   20 steps and writes its full-precision restart checkpoint, whose
//!   SHA-256 is pinned to what the parent commit's binaries wrote (0777b1d,
//!   one pin per build, each reproduced twice on this machine's card). A
//!   different GPU or driver may compile other SASS: re-capture both pins
//!   from the parent commit there, never from this one.
//! * `energy_uniform_cp_field_equals_constant` (class R, D-F32-1): a uniform
//!   field equal to `GasProperties::cp` reproduces the constant path to
//!   round-off, under Euler and backward, and so does its assembly budget.
//!   A uniform field `2 cp` with conductivity `k` reproduces the constant
//!   `cp` with `k/2`: the same
//!   equation times two, which holds only if BOTH rows read the field.
//! * `energy_cp_field_refuses_by_name`: a zero, negative or NaN value and a
//!   short buffer are refused by name.
//!
//! A machine without a card passes vacuously, the convention of
//! `tests/bc_thermal.rs`. No GPL-licensed source was consulted.

use std::path::PathBuf;
use std::process::Command;

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::energy::{DomainKind, Energy, EnergyBudget, EnergyControls, GasProperties, GasState};
use ofgpu::field_setup::{setup_scalar_field_with, BcInputs};
use ofgpu::fv::SnGradScheme;
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField};
use ofgpu::timescheme::DdtScheme;
use ofgpu::{BcKind, Gpu, GpuMesh, GpuScalarField, GpuSurfaceScalarField, HostMesh, Result, Scalar};

/// `t` in the f64 build; under `single` (class R, D-F32-1)
/// `max(t, min(t*2^29, 1e-5))`.
const fn tol(t: f64) -> f64 {
    if cfg!(feature = "single") {
        let scaled = t * 536870912.0;
        let capped = if scaled < 1e-5 { scaled } else { 1e-5 };
        if t > capped {
            t
        } else {
            capped
        }
    } else {
        t
    }
}

/// The build's `Scalar` widened to f64.
#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    x as f64
}

/// The parent commit's restart checkpoint, f64 build.
#[cfg(not(feature = "single"))]
const DEFAULT_PIN: &str = "ca1682374dda61c7e50158cd90759ad9c062bc3a5c645c5ea275cc903f7d01e1";
/// The parent commit's restart checkpoint, `single` build.
#[cfg(feature = "single")]
const DEFAULT_PIN: &str = "eaffceeed13a09fc9febebc731e5c6a67fd7de2051d81fb08adce3a5585a748b";

#[test]
fn energy_cp_field_default_bitwise() {
    if Gpu::new(0).is_err() {
        eprintln!("no CUDA device: the hash-pinned run needs one; skipped");
        return;
    }
    let shipped = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cases")
        .join("channelPeriodicFluxWF.jsonc");
    let dir = std::env::temp_dir().join(format!("ofgpu-chr10a-pin-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let case = dir.join("channelPeriodicFluxWF.jsonc");
    std::fs::copy(&shipped, &case).expect("copy the shipped case");

    let out = Command::new(env!("CARGO_BIN_EXE_ofgpu-lowmach"))
        .arg(&case)
        .args(["-iters", "20", "-restartWrite", "20"])
        .output()
        .expect("run ofgpu-lowmach");
    assert!(
        out.status.success(),
        "ofgpu-lowmach failed: {}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let mcr = dir.join("channelPeriodicFluxWF_jsonc").join("restart.mcr");
    let bytes = std::fs::read(&mcr).expect("restart.mcr beside the run");
    let got = ofgpu::radiation::wsgg::sha256_hex(&bytes);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(got, DEFAULT_PIN, "the default (no cp field) run moved a bit");
}

/// The [16,1,1] channel over [1, 0.1]^2: xMin/xMax patches, four empties.
fn channel() -> BlockSpec {
    let axis = |hi: Scalar, n: usize| GradedAxis {
        lo: 0.0,
        hi,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    BlockSpec {
        x: axis(1.0, 16),
        y: axis(0.1, 1),
        z: axis(0.1, 1),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["patch", "patch", "empty", "empty", "empty", "empty"].map(String::from),
        cyclic: Vec::new(),
    }
}

fn patch_of(hm: &HostMesh, name: &str) -> usize {
    hm.patches
        .iter()
        .position(|p| p.name == name)
        .unwrap_or_else(|| panic!("no patch {name}"))
}

/// A +x plug flow of `speed` m/s: `speed |S_f|` owner to neighbour inside,
/// `-speed |S|` in at xMin, `+speed |S|` out at xMax.
fn plug_phi(gpu: &Gpu, m: &GpuMesh, hm: &HostMesh, speed: f64) -> Result<GpuSurfaceScalarField> {
    let mut phi = GpuSurfaceScalarField::zeros(gpu, m, "phi")?;
    let f_int: Vec<Scalar> = hm
        .mag_sf
        .iter()
        .zip(&hm.sf)
        .map(|(&s, v)| {
            let s = w64(s);
            if v.x > 0.0 {
                (speed * s) as Scalar
            } else if v.x < 0.0 {
                (-speed * s) as Scalar
            } else {
                0.0
            }
        })
        .collect();
    gpu.write(&mut phi.f, &f_int)?;
    let mut bf = vec![0.0 as Scalar; hm.n_boundary_faces];
    for (name, sign) in [("xMin", -1.0), ("xMax", 1.0)] {
        let pi = patch_of(hm, name);
        for f in hm.patches[pi].start..hm.patches[pi].start + hm.patches[pi].size {
            bf[f] = (sign * speed * w64(hm.b_mag_sf[f])) as Scalar;
        }
    }
    gpu.write(&mut phi.bf, &bf)?;
    Ok(phi)
}

fn ps(ty: &str, v: Option<f64>) -> PatchFieldSpec {
    PatchFieldSpec {
        type_name: ty.into(),
        value: v.map(|x| vec![x as Scalar]).unwrap_or_default(),
        ..Default::default()
    }
}

/// 310 K in at xMin, zeroGradient out at xMax, 300 K inside.
fn raw_t() -> RawScalarField {
    let mut raw = RawScalarField {
        name: "T".into(),
        ..Default::default()
    };
    raw.boundary.insert("xMin".into(), ps("fixedValue", Some(310.0)));
    raw.boundary.insert("xMax".into(), ps("zeroGradient", None));
    raw.internal = vec![300.0 as Scalar];
    raw
}

/// The laminar transient `Energy` the two legs compare: tolerance 1e-14,
/// relTol 0, no relaxation, Uncorrected snGrad.
fn energy_on<'m>(gpu: &Gpu, m: &'m GpuMesh, props: GasProperties, ddt: DdtScheme, dt: f64) -> Result<Energy<'m>> {
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
        ddt,
        ..EnergyControls::default()
    };
    Energy::new(gpu, m, ctrl, props)
}

/// `steps` transient steps with the density following `T`; `cp_field` is a
/// uniform value set on every cell and boundary face, or none. Returns `T`
/// and the assembly budget at the end state.
fn run(
    gpu: &Gpu,
    hm: &HostMesh,
    gm: &GpuMesh,
    props: &GasProperties,
    ddt: DdtScheme,
    cp_field: Option<Scalar>,
) -> Result<(Vec<f64>, EnergyBudget)> {
    let (dt, steps) = (2.0, 10);
    let mut e = energy_on(gpu, gm, props.clone(), ddt, dt)?;
    let mut gas = GasState::new(gpu, gm, props.clone(), DomainKind::Open, 101325.0)?;
    let mut nut = GpuScalarField::zeros(gpu, gm, "nut")?;
    gpu.write(&mut nut.bc_kind, &vec![BcKind::Calculated as i32; gm.n_boundary_faces])?;
    let phi = plug_phi(gpu, gm, hm, 0.01)?;
    setup_scalar_field_with(gpu, e.field_mut(), &raw_t(), hm, BcInputs::default())?;
    e.initialise(gpu)?;
    gas.update_density(gpu, e.field())?;
    if let Some(c) = cp_field {
        let cells = gpu.upload(&vec![c; gm.n_cells])?;
        let faces = gpu.upload(&vec![c; gm.n_boundary_faces])?;
        e.set_cp_field(gpu, &cells, &faces)?;
        assert!(e.cp_field().is_some(), "the field is held");
    }
    for step in 0..steps {
        if step > 0 {
            e.advance_time_levels(gpu)?;
            e.advance_time_step(dt as Scalar);
            gas.advance_time_levels();
        }
        e.correct(gpu, &phi, &nut, &nut.f, 1.5e-5, &gas)?;
        gas.update_density(gpu, e.field())?;
    }
    let budget = e.assembly_budget(gpu, &gas)?;
    Ok((gpu.download(&e.field().f)?.into_iter().map(w64).collect(), budget))
}

/// The budget entries the field branch of `assembly_budget` forms its own
/// way (SPEC-LIT 133.8): `|a - b|` over the larger magnitude, per entry.
fn budget_rel(a: &EnergyBudget, b: &EnergyBudget) -> f64 {
    let pairs = [
        (a.ddt_w, b.ddt_w),
        (a.convection_w, b.convection_w),
        (a.bounded_correction_direct_w, b.bounded_correction_direct_w),
        (a.bounded_correction_prescribed_w, b.bounded_correction_prescribed_w),
        (a.target_divergence_w, b.target_divergence_w),
        (a.net_mass_flux_kg_per_s, b.net_mass_flux_kg_per_s),
        (a.mass_flux_divergence_l1_kg_per_s, b.mass_flux_divergence_l1_kg_per_s),
        (a.mass_flux_divergence_max_kg_per_s, b.mass_flux_divergence_max_kg_per_s),
    ];
    let mut w = 0.0f64;
    for (x, y) in pairs {
        let scale = x.abs().max(y.abs());
        let r = if scale == 0.0 { 0.0 } else { (x - y).abs() / scale };
        if r.is_nan() || r > w {
            w = r;
        }
    }
    w
}

/// The worst `|a/b - 1|` over the cells; NaN propagates.
fn worst_rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut w = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let r = (x / y - 1.0).abs();
        if r.is_nan() || r > w {
            w = r;
        }
        if w.is_nan() {
            return w;
        }
    }
    w
}

#[test]
fn energy_uniform_cp_field_equals_constant() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = blockgen::build_mesh(&channel())?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let props = GasProperties { k: 1.0, ..GasProperties::default() };
    let cp = props.cp;
    let half_k = GasProperties { k: props.k / 2.0, ..props.clone() };

    for ddt in [DdtScheme::Euler, DdtScheme::Backward] {
        let (constant, b_const) = run(&gpu, &hm, &gm, &props, ddt, None)?;
        // Non-vacuous: the front has entered the channel.
        let moved = constant.iter().map(|&t| (t - 300.0).abs()).fold(0.0f64, f64::max);
        assert!(moved > 1.0, "{ddt:?}: T moved only {moved} K");

        // (a) a uniform field equal to the constant.
        let (same, b_same) = run(&gpu, &hm, &gm, &props, ddt, Some(cp))?;
        let ra = worst_rel(&same, &constant);
        // ... and its budget, whose field branch divides by `cp_f` and
        // weighs by `cp_c`.
        let rbud = budget_rel(&b_same, &b_const);
        assert!(b_const.mass_flux_divergence_l1_kg_per_s > 0.0, "{ddt:?}: the mass rows measure something");
        // (b) the field 2 cp with k is the constant cp with k/2, times two.
        let (doubled, _) = run(&gpu, &hm, &gm, &props, ddt, Some(2.0 * cp))?;
        let (halved, _) = run(&gpu, &hm, &gm, &half_k, ddt, None)?;
        let rb = worst_rel(&doubled, &halved);
        // ... and it is a different temperature from (a)'s, so (b) tests
        // something.
        let rd = worst_rel(&doubled, &constant);
        eprintln!("  [133.8] {ddt:?}: moved {moved:e} K, (a) {ra:e}, budget {rbud:e}, (b) {rb:e}, 2cp vs cp {rd:e}");
        assert!(ra <= tol(1e-12), "{ddt:?} (a) uniform cp field vs constant: {ra:e}");
        assert!(rbud <= tol(1e-12), "{ddt:?} (a) budget, uniform cp field vs constant: {rbud:e}");
        assert!(rb <= tol(1e-12), "{ddt:?} (b) 2cp field, k vs cp, k/2: {rb:e}");
        assert!(rd > 1e-4, "{ddt:?}: 2cp did not change T ({rd:e})");
    }
    Ok(())
}

#[test]
fn energy_cp_field_refuses_by_name() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = blockgen::build_mesh(&channel())?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let props = GasProperties::default();
    let mut e = energy_on(&gpu, &gm, props, DdtScheme::Euler, 1.0)?;
    let (n, nb) = (gm.n_cells, gm.n_boundary_faces);
    let good = gpu.upload(&vec![1006.0 as Scalar; nb])?;

    for (what, bad) in [("zero", 0.0 as Scalar), ("negative", -1.0), ("NaN", Scalar::NAN)] {
        let mut v = vec![1006.0 as Scalar; n];
        v[5] = bad;
        let cells = gpu.upload(&v)?;
        let err = e.set_cp_field(&gpu, &cells, &good).expect_err(what).to_string();
        assert!(err.contains("cp field") && err.contains("cell 5"), "{what}: {err}");
        assert!(e.cp_field().is_none(), "{what}: a refused field is not held");
    }
    let mut faces = vec![1006.0 as Scalar; nb];
    faces[3] = -2.0;
    let bad_faces = gpu.upload(&faces)?;
    let cells = gpu.upload(&vec![1006.0 as Scalar; n])?;
    let err = e.set_cp_field(&gpu, &cells, &bad_faces).expect_err("face").to_string();
    assert!(err.contains("cp field") && err.contains("boundary face 3"), "{err}");

    let short = gpu.upload(&vec![1006.0 as Scalar; n - 1])?;
    let err = e.set_cp_field(&gpu, &short, &good).expect_err("short").to_string();
    assert!(err.contains("cp field") && err.contains("cells"), "{err}");

    e.set_cp_field(&gpu, &cells, &good)?;
    assert!(e.cp_field().is_some());
    e.clear_cp_field();
    assert!(e.cp_field().is_none(), "cleared back to the constant path");
    Ok(())
}
