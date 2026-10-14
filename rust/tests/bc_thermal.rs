// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 120.8 - the thermal family through the public API, in BOTH
//! builds: `externalWallHeatFluxTemperature`'s coefficient mode closed
//! against the series-resistance form and its radiating surface against the
//! quartic's root, the device triple against its host twin
//! `cht::ambient::ThermalFace::triple`, the flux/power modes against
//! SPEC-LIT 32.2's `fixedFluxTemperature` bit for bit, `lumpedMassTemperature`
//! against the exponential decay of the lumped-capacitance balance at second
//! order, and `totalFlowRateAdvectiveDiffusive` against Danckwerts' 1-D
//! `u c_in = u c - D dc/dx` - plus the refusals by name and the two alphat
//! aliases. The tolerances follow docs/17 §5.1 class R, D-F32-1 - the f64
//! tolerance in the f64 build, `max(t, min(t*2^29, 1e-5))` under `single` -
//! which the `tol` const fn below encodes; everything is compared in f64. A
//! machine without a card passes vacuously, the convention of
//! `tests/bc_inlets.rs`.
//!
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::energy::{DomainKind, Energy, EnergyControls, GasProperties, GasState};
use ofgpu::field_setup::{
    faces_where, scalar_boundary_with, setup_scalar_field_with, thermal_walls,
    vector_boundary_with, BcInputs,
};
use ofgpu::fv::SnGradScheme;
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::timescheme::DdtScheme;
use ofgpu::{
    BcKind, Error, Gpu, GpuMesh, GpuScalarField, GpuSurfaceScalarField, HostMesh, Result, Scalar,
    Vec3,
};

/// `t` in the f64 build; under `single` (class R, D-F32-1)
/// `max(t, min(t*2^29, 1e-5))` - i.e. 1e-10 -> 1e-5, 1e-13 -> 1e-5.
const fn tol(t: f64) -> f64 {
    if cfg!(feature = "single") {
        let scaled = t * 536870912.0;
        let capped = if scaled < 1e-5 { scaled } else { 1e-5 };
        if t > capped { t } else { capped }
    } else {
        t
    }
}

/// The build's `Scalar` widened to the f64 the answer keys are computed in.
#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    x as f64
}

/// A NaN-propagating max: `f64::max` swallows NaN and `f64::maximum` is
/// unstable on this toolchain, so fold by hand - NaN in, NaN out.
fn nmax(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// A checked term for the folds: a non-finite term panics here, naming
/// itself, instead of vanishing through `f64::max`.
fn fin(what: &str, x: f64) -> f64 {
    assert!(x.is_finite(), "{what} is non-finite: {x:e}");
    x
}

/// A uniform block, every patch typed by the caller - the `spec` shape of
/// `tests/bc_inlets.rs`, sized by the caller.
fn box_spec(n: [usize; 3], lo: [Scalar; 3], hi: [Scalar; 3], types: [&str; 6]) -> BlockSpec {
    let axis = |lo: Scalar, hi: Scalar, n: usize| GradedAxis {
        lo,
        hi,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    BlockSpec {
        x: axis(lo[0], hi[0], n[0]),
        y: axis(lo[1], hi[1], n[1]),
        z: axis(lo[2], hi[2], n[2]),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: types.map(String::from),
        cyclic: Vec::new(),
    }
}

/// The index of a patch by name.
fn patch_of(hm: &HostMesh, name: &str) -> usize {
    hm.patches
        .iter()
        .position(|p| p.name == name)
        .unwrap_or_else(|| panic!("no patch {name}"))
}

/// A patch entry carrying nothing but its type.
fn ps(ty: &str) -> PatchFieldSpec {
    PatchFieldSpec {
        type_name: ty.into(),
        ..Default::default()
    }
}

/// A patch entry of one of the thermal names, with its entries.
fn th_bc(name: &str, extra: &[(&str, &str)]) -> PatchFieldSpec {
    let mut s = ps(name);
    for (k, v) in extra {
        s.extra.insert((*k).into(), (*v).into());
    }
    s
}

/// A `lumpedMassTemperature` entry: `mass`/`Cp` ride the free-form `extra`
/// map, while the lump's `T_m^0` is the structured `value` list - `None`
/// writes no `value` at all, which the parser refuses.
fn lump_bc(extra: &[(&str, &str)], t0: Option<f64>) -> PatchFieldSpec {
    let mut s = PatchFieldSpec {
        type_name: "lumpedMassTemperature".into(),
        value: t0.map(|t| vec![t as Scalar]).unwrap_or_default(),
        ..Default::default()
    };
    for (k, v) in extra {
        s.extra.insert((*k).into(), (*v).into());
    }
    s
}

/// A Danckwerts entry: `inletValue` is the structured list, uniform - the
/// parser refuses a spread one.
fn danck_bc(t_in: f64) -> PatchFieldSpec {
    PatchFieldSpec {
        type_name: "totalFlowRateAdvectiveDiffusive".into(),
        inlet_value: vec![t_in as Scalar],
        ..Default::default()
    }
}

/// A `fixedValue` entry at one uniform temperature.
fn fixed_value(v: f64) -> PatchFieldSpec {
    PatchFieldSpec {
        type_name: "fixedValue".into(),
        value: vec![v as Scalar],
        ..Default::default()
    }
}

/// A raw scalar field named T, one uniform internal value, the given patch
/// entries.
fn raw_t(boundary: Vec<(&str, PatchFieldSpec)>, internal: f64) -> RawScalarField {
    let mut raw = RawScalarField {
        name: "T".into(),
        ..Default::default()
    };
    for (p, s) in boundary {
        raw.boundary.insert(p.into(), s);
    }
    raw.internal = vec![internal as Scalar];
    raw
}

/// The [4,1,1] slab over [0, 0.1] x [0.01]^2: xMin/xMax patches, four
/// empties. k = 0.05 gives C_b = k Delta_b = 4.0 on the x ends.
fn slab() -> BlockSpec {
    box_spec(
        [4, 1, 1],
        [0.0, 0.0, 0.0],
        [0.1, 0.01, 0.01],
        ["patch", "patch", "empty", "empty", "empty", "empty"],
    )
}

/// The [16,1,1] channel over [1, 0.1]^2: xMin/xMax patches, four empties.
fn channel() -> BlockSpec {
    box_spec(
        [16, 1, 1],
        [0.0, 0.0, 0.0],
        [1.0, 0.1, 0.1],
        ["patch", "patch", "empty", "empty", "empty", "empty"],
    )
}

/// The laminar `Energy` of `src/energy.rs`'s `laminar_slab_energy`:
/// tolerance 1e-14, relTol 0, 2000 sweeps, no relaxation, Uncorrected
/// snGrad, the caller's ddt.
fn energy_on<'m>(
    gpu: &Gpu,
    m: &'m GpuMesh,
    props: GasProperties,
    steady: bool,
    ddt: DdtScheme,
    delta_t: f64,
) -> Result<(Energy<'m>, GpuScalarField, GpuSurfaceScalarField)> {
    let ctrl = EnergyControls {
        t_solver: ofgpu::solver::SolverControls {
            tolerance: 1e-14,
            rel_tol: 0.0,
            max_iter: 2000,
            check_interval: 1,
            ..ofgpu::solver::SolverControls::default()
        },
        t_relax: 1.0,
        steady,
        delta_t: delta_t as Scalar,
        sn_grad: SnGradScheme::Uncorrected,
        ddt,
        ..EnergyControls::default()
    };
    let e = Energy::new(gpu, m, ctrl, props)?;
    // Laminar: zero eddy viscosity with `calculated` boundary values.
    let mut nut = GpuScalarField::zeros(gpu, m, "nut")?;
    gpu.write(
        &mut nut.bc_kind,
        &vec![BcKind::Calculated as i32; m.n_boundary_faces],
    )?;
    let phi = GpuSurfaceScalarField::zeros(gpu, m, "phi")?;
    Ok((e, nut, phi))
}

/// Seed `T` from `raw` with the thermal opt-in, wire the thermal walls and
/// whatever `fixedFluxTemperature` faces the raw names, initialise, and
/// freeze the gas at the initial field - the whole per-run preamble.
fn wire_t(
    gpu: &Gpu,
    hm: &HostMesh,
    raw: &RawScalarField,
    e: &mut Energy<'_>,
    gas: &mut GasState<'_>,
) -> Result<()> {
    setup_scalar_field_with(
        gpu,
        e.field_mut(),
        raw,
        hm,
        BcInputs {
            thermal: true,
            ..BcInputs::default()
        },
    )?;
    let walls = thermal_walls(raw, hm)?;
    e.set_thermal_walls(gpu, &walls)?;
    let ffq = faces_where(raw, hm, BcKind::is_fixed_flux_temperature)?;
    e.set_fixed_flux_walls(gpu, &ffq)?;
    e.initialise(gpu)?;
    gas.update_density(gpu, e.field())
}

/// Steady `correct` passes until max |dT_b| on patch `pi` is at most
/// `tol(1e-13)` of its own level - or `cap` passes. Returns the pass count,
/// the patch's face values and the internal field as they stand.
fn settle(
    gpu: &Gpu,
    hm: &HostMesh,
    e: &mut Energy<'_>,
    phi: &GpuSurfaceScalarField,
    nut: &GpuScalarField,
    gas: &GasState<'_>,
    pi: usize,
    cap: usize,
) -> Result<(usize, Vec<Scalar>, Vec<Scalar>)> {
    let at = hm.patches[pi].start;
    let np = hm.patches[pi].size;
    let mut prev = vec![f64::NAN; np];
    let mut passes = 0usize;
    let mut tb = vec![0.0 as Scalar; hm.n_boundary_faces];
    for pass in 0..cap {
        e.correct(gpu, phi, nut, &nut.f, 1.5e-5, gas)?;
        passes = pass + 1;
        tb = gpu.download(&e.field().bf)?;
        let (mut change, mut scale) = (0.0f64, 0.0f64);
        for i in 0..np {
            change = nmax(change, (w64(tb[at + i]) - prev[i]).abs());
            scale = nmax(scale, w64(tb[at + i]).abs());
        }
        prev = tb[at..at + np].iter().map(|&v| w64(v)).collect();
        // Pass 0 has no previous level to differ from - the criterion is
        // live from pass 1 on.
        if pass > 0 && fin("dT_b", change) <= tol(1e-13) * fin("T_b", scale) {
            break;
        }
    }
    let tf = gpu.download(&e.field().f)?;
    Ok((passes, tb, tf))
}

// ==========================================================================
//  (120.8a)-(120.8f) - the coefficient mode on a slab
// ==========================================================================

/// The x ends' face conductances and areas, widened - `hm`'s f64 geometry,
/// widened once like the kernel widens the stored arrays.
fn end_conductances(hm: &HostMesh, pi: usize) -> (Vec<f64>, Vec<f64>) {
    let at = hm.patches[pi].start;
    let np = hm.patches[pi].size;
    let c = (at..at + np).map(|f| w64(hm.b_delta_coeffs[f])).collect();
    let a = (at..at + np).map(|f| w64(hm.b_mag_sf[f])).collect();
    (c, a)
}

/// THE gate, row 1 (SPEC-LIT 120.8): `mode coefficient` with two wall layers
/// in series with the outer film closes against Incropera's series
/// resistance, `q = (T_hot - Ta)/(L/k + R_l + 1/h)`, on EVERY xMax face and
/// on the xMin face too; the control run without the layers is the plain
/// two-resistance form.
#[test]
fn coefficient_mode_slab_is_the_series_resistance_closed_form() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = blockgen::build_mesh(&slab())?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let props = GasProperties { k: 0.05, ..GasProperties::default() };
    let xmn = patch_of(&hm, "xMin");
    let xmx = patch_of(&hm, "xMax");
    let (cb_x, _) = end_conductances(&hm, xmx);
    let (cb_in, _) = end_conductances(&hm, xmn);
    let npx = hm.patches[xmx].size;
    let npn = hm.patches[xmn].size;
    let atx = hm.patches[xmx].start;
    let atn = hm.patches[xmn].start;

    let entry = |extra: &[(&str, &str)]| th_bc("externalWallHeatFluxTemperature", extra);
    for (layers, want_q) in [
        (true, 100.0 / (0.1 / 0.05 + 0.03 + 0.1)),
        (false, 100.0 / (0.1 / 0.05 + 0.1)),
    ] {
        let extra: Vec<(&str, &str)> = if layers {
            vec![
                ("mode", "coefficient"),
                ("h", "10"),
                ("Ta", "300"),
                ("thicknessLayers", "(0.01 0.02)"),
                ("kappaLayers", "(0.5 2.0)"),
            ]
        } else {
            vec![("mode", "coefficient"), ("h", "10"), ("Ta", "300")]
        };
        let raw = raw_t(
            vec![("xMin", fixed_value(400.0)), ("xMax", entry(&extra))],
            300.0,
        );
        let (mut e, nut, phi) =
            energy_on(&gpu, &gm, props.clone(), true, DdtScheme::SteadyState, 0.0)?;
        let mut gas = GasState::new(&gpu, &gm, props.clone(), DomainKind::Open, 101325.0)?;
        wire_t(&gpu, &hm, &raw, &mut e, &mut gas)?;
        let (passes, tb, tf) = settle(&gpu, &hm, &mut e, &phi, &nut, &gas, xmx, 200)?;
        eprintln!("  [120.8] coefficient mode: settled in {passes} passes (layers {layers})");

        // The measured flux on EVERY xMax face: q_m = C_b (T_P - T_b),
        // `C_b = k_eff Delta_b` - the k_eff at the face, downloaded.
        let keff = gpu.download(e.k_eff_wall())?;
        let mut qmax = 0.0f64;
        for i in 0..npx {
            let f = atx + i;
            let tp = fin("T_P", w64(tf[hm.b_face_cells[f] as usize]));
            let qm = fin("q_m", w64(keff[f]) * cb_x[i] * (tp - w64(tb[f])));
            qmax = nmax(qmax, (qm / want_q - 1.0).abs());
        }
        eprintln!("  [120.8] coefficient mode: q_m xMax {want_q:.12} rel err {qmax:e}");
        assert!(qmax <= tol(1e-10), "q_m on xMax: rel err {qmax:e} of {want_q}");

        // ... and on the xMin face: q_m = C_b (T_b - T_P).
        let mut qmin = 0.0f64;
        for i in 0..npn {
            let f = atn + i;
            let tp = fin("T_P", w64(tf[hm.b_face_cells[f] as usize]));
            let qm = fin("q_m", w64(keff[f]) * cb_in[i] * (w64(tb[f]) - tp));
            qmin = nmax(qmin, (qm / want_q - 1.0).abs());
        }
        eprintln!("  [120.8] coefficient mode: q_m xMin {want_q:.12} rel err {qmin:e}");
        assert!(qmin <= tol(1e-10), "q_m on xMin: rel err {qmin:e} of {want_q}");

        // The face temperature: T_o = T_b - R_l q_c closed against the
        // analytic outer surface, i.e. T_b itself at 306.1032863849765.
        if layers {
            let want_tb = 300.0 + want_q * 0.13;
            let mut dtb = 0.0f64;
            for i in 0..npx {
                dtb = nmax(dtb, (w64(tb[atx + i]) / want_tb - 1.0).abs());
            }
            eprintln!("  [120.8] coefficient mode: T_b {want_tb:.12} rel err {dtb:e}");
            assert!(dtb <= tol(1e-10), "T_b on xMax: rel err {dtb:e} of {want_tb}");
        }
    }
    Ok(())
}

/// SPEC-LIT 120.8 (120.8d): a radiating outer surface is the quartic's root.
/// `q = k/L (T_hot - T_b)` through the slab (plus the layers) balances
/// `h (T_o - Ta) + eps sigma (T_o^4 - Ta^4)` at the outer surface `T_o =
/// T_b - R_l q`; each case settles inside 50 Newton-tangent passes.
#[test]
fn a_radiating_outer_surface_is_the_root_of_the_quartic() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = blockgen::build_mesh(&slab())?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let props = GasProperties { k: 0.05, ..GasProperties::default() };
    let xmx = patch_of(&hm, "xMax");
    let npx = hm.patches[xmx].size;
    let atx = hm.patches[xmx].start;

    for (layers, want_tb) in [
        (false, 309.59128377473974_f64),
        (true, 313.7498487803651),
    ] {
        let extra: Vec<(&str, &str)> = if layers {
            vec![
                ("mode", "coefficient"),
                ("h", "10"),
                ("Ta", "300"),
                ("emissivity", "0.8"),
                ("thicknessLayers", "(0.01 0.02)"),
                ("kappaLayers", "(0.5 2.0)"),
            ]
        } else {
            vec![
                ("mode", "coefficient"),
                ("h", "10"),
                ("Ta", "300"),
                ("emissivity", "0.8"),
            ]
        };
        let raw = raw_t(
            vec![
                ("xMin", fixed_value(600.0)),
                ("xMax", th_bc("externalWallHeatFluxTemperature", &extra)),
            ],
            300.0,
        );
        let (mut e, nut, phi) =
            energy_on(&gpu, &gm, props.clone(), true, DdtScheme::SteadyState, 0.0)?;
        let mut gas = GasState::new(&gpu, &gm, props.clone(), DomainKind::Open, 101325.0)?;
        wire_t(&gpu, &hm, &raw, &mut e, &mut gas)?;
        let (passes, tb, _) = settle(&gpu, &hm, &mut e, &phi, &nut, &gas, xmx, 200)?;
        eprintln!("  [120.8] radiating: T_b {want_tb:.12} in {passes} passes (layers {layers})");
        assert!(passes <= 50, "the quartic's tangent took {passes} passes");
        let mut dtb = 0.0f64;
        for i in 0..npx {
            dtb = nmax(dtb, (w64(tb[atx + i]) / want_tb - 1.0).abs());
        }
        eprintln!("  [120.8] radiating: T_b {want_tb:.12} rel err {dtb:e}");
        assert!(dtb <= tol(1e-10), "T_b: rel err {dtb:e} of {want_tb}");
    }
    Ok(())
}

/// SPEC-LIT 120.8: the kernel `energyThermalWall` is
/// `cht::ambient::ThermalFace::triple`'s device twin - same operation order,
/// every input widened, one rounding per stored value. Three passes into the
/// layered radiating case, a fresh `refresh_thermal_walls` after the
/// downloads lands on the same numbers.
#[test]
fn the_device_triple_is_its_host_twin() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = blockgen::build_mesh(&slab())?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let props = GasProperties { k: 0.05, ..GasProperties::default() };
    let xmx = patch_of(&hm, "xMax");
    let npx = hm.patches[xmx].size;
    let atx = hm.patches[xmx].start;
    let raw = raw_t(
        vec![
            ("xMin", fixed_value(600.0)),
            (
                "xMax",
                th_bc(
                    "externalWallHeatFluxTemperature",
                    &[
                        ("mode", "coefficient"),
                        ("h", "10"),
                        ("Ta", "300"),
                        ("emissivity", "0.8"),
                        ("thicknessLayers", "(0.01 0.02)"),
                        ("kappaLayers", "(0.5 2.0)"),
                    ],
                ),
            ),
        ],
        300.0,
    );
    let (mut e, nut, phi) =
        energy_on(&gpu, &gm, props.clone(), true, DdtScheme::SteadyState, 0.0)?;
    let mut gas = GasState::new(&gpu, &gm, props.clone(), DomainKind::Open, 101325.0)?;
    wire_t(&gpu, &hm, &raw, &mut e, &mut gas)?;
    let walls = thermal_walls(&raw, &hm)?;
    let face_of = |i: usize| walls.faces[i];

    for _ in 0..3 {
        e.correct(&gpu, &phi, &nut, &nut.f, 1.5e-5, &gas)?;
    }
    // The state the twin is judged on, downloaded BEFORE the refresh.
    let tb = gpu.download(&e.field().bf)?;
    let tf = gpu.download(&e.field().f)?;
    let keff = gpu.download(e.k_eff_wall())?;
    let pcv = gpu.download(&e.conv_flux().bf)?;

    e.refresh_thermal_walls(&gpu)?;
    let fr = gpu.download(&e.field().fr)?;
    let rv = gpu.download(&e.field().ref_value)?;
    let rg = gpu.download(&e.field().ref_grad)?;

    for i in 0..npx {
        let bf = atx + i;
        let f = face_of(i);
        let tp = fin("T_P", w64(tf[hm.b_face_cells[bf] as usize]));
        let want = f
            .triple(
                fin("k_eff", w64(keff[bf])),
                fin("Delta_b", w64(hm.b_delta_coeffs[bf])),
                fin("T_b", w64(tb[bf])),
                tp,
                w64(pcv[bf]),
                fin("|S_f|", w64(hm.b_mag_sf[bf])),
            )
            .expect("a live wall face has a triple");
        let want = [want.0, want.1, want.2];
        let got = [w64(fr[bf]), w64(rv[bf]), w64(rg[bf])];
        let names = ["fr", "refValue", "refGrad"];
        for (c, (&w, &g)) in want.iter().zip(&got).enumerate() {
            fin(names[c], g);
            if w == 0.0 {
                assert_eq!(g, 0.0, "{} face {i}: {} must be exactly zero", names[c], names[c]);
            } else {
                let d = (g / w - 1.0).abs();
                assert!(
                    d <= tol(1e-14),
                    "face {i} {} = {g:e}, twin {w:e}, rel {d:e}",
                    names[c]
                );
            }
        }
    }
    eprintln!("  [120.8] device triple = host twin on {npx} xMax faces, 3 numbers each");
    Ok(())
}

// ==========================================================================
//  (120.8f) with H = 0 - the flux and power modes are 32.2's condition
// ==========================================================================

/// SPEC-LIT 120.8: with `H = 0` the triple is `fr = 0`,
/// `refGrad = q_in/k_eff` - SPEC-LIT 32.2's fixed flux in every bit. The
/// flux-spelled runs agree with `fixedFluxTemperature q -50` BIT FOR BIT
/// (class D); the power-spelled runs differ only by the one f64 `Q/A`
/// rounding, and land within 1e-12 relative of the flux run.
#[test]
fn flux_and_power_are_the_fixed_flux_condition() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = blockgen::build_mesh(&slab())?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let props = GasProperties { k: 0.05, ..GasProperties::default() };
    let xmx = patch_of(&hm, "xMax");
    let atx = hm.patches[xmx].start;
    let npx = hm.patches[xmx].size;
    let area: f64 = (atx..atx + npx).map(|f| w64(hm.b_mag_sf[f])).sum();

    let run = |xmax: PatchFieldSpec| -> Result<(Vec<Scalar>, Vec<Scalar>)> {
        let raw = raw_t(vec![("xMin", fixed_value(400.0)), ("xMax", xmax)], 300.0);
        let (mut e, nut, phi) =
            energy_on(&gpu, &gm, props.clone(), true, DdtScheme::SteadyState, 0.0)?;
        let mut gas = GasState::new(&gpu, &gm, props.clone(), DomainKind::Open, 101325.0)?;
        wire_t(&gpu, &hm, &raw, &mut e, &mut gas)?;
        for _ in 0..20 {
            e.correct(&gpu, &phi, &nut, &nut.f, 1.5e-5, &gas)?;
        }
        Ok((gpu.download(&e.field().f)?, gpu.download(&e.field().bf)?))
    };

    let flux = run(th_bc(
        "fixedFluxTemperature",
        &[("q", "-50")],
    ))?;
    let ext = run(th_bc(
        "externalWallHeatFluxTemperature",
        &[("mode", "flux"), ("q", "-50")],
    ))?;
    let turb = run(th_bc(
        "turbulentHeatFluxTemperature",
        &[("heatSource", "flux"), ("q", "-50")],
    ))?;
    for (name, other) in [("externalWallHeatFluxTemperature", &ext), ("turbulentHeatFluxTemperature", &turb)] {
        for (c, (a, b)) in flux.0.iter().zip(&other.0).enumerate() {
            assert_eq!(w64(*a), w64(*b), "{name}: T cell {c} differs from fixedFluxTemperature");
        }
        for (c, (a, b)) in flux.1.iter().zip(&other.1).enumerate() {
            assert_eq!(w64(*a), w64(*b), "{name}: T face {c} differs from fixedFluxTemperature");
        }
        eprintln!("  [120.8] {name}: flux mode is bitwise fixedFluxTemperature");
    }

    // The power spellings: the whole patch gets -50 W/m^2 worth of area.
    let q_w = -50.0 * area;
    let extp = run(th_bc(
        "externalWallHeatFluxTemperature",
        &[("mode", "power"), ("Q", &format!("{q_w:e}"))],
    ))?;
    let turbp = run(th_bc(
        "turbulentHeatFluxTemperature",
        &[("heatSource", "power"), ("q", &format!("{q_w:e}"))],
    ))?;
    for (name, other) in [("mode power", &extp), ("heatSource power", &turbp)] {
        let mut worst = 0.0f64;
        for (a, b) in flux.0.iter().zip(&other.0).chain(flux.1.iter().zip(&other.1)) {
            worst = nmax(worst, (w64(*b) / w64(*a) - 1.0).abs());
        }
        eprintln!("  [120.8] {name}: worst rel deviation from the flux run {worst:e}");
        assert!(worst <= tol(1e-12), "{name}: rel err {worst:e}");
    }
    Ok(())
}

// ==========================================================================
//  (120.8g) - the lumped mass
// ==========================================================================

/// THE gate, row 2 (SPEC-LIT 120.8): a lumped mass behind the slab decays
/// like Incropera's `T - T_inf = (T_0 - T_inf) exp(-t/tau)`,
/// `tau = m cp L/(k A) = 2.4e8 s`, and `backward` converges at second
/// order; Euler does not. One `correct` per step, the lump's level
/// recorded by (120.8g) after the step is opened.
#[test]
fn a_lumped_mass_decays_exponentially_at_second_order() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = blockgen::build_mesh(&slab())?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let props = GasProperties { k: 0.05, ..GasProperties::default() };
    let xmx = patch_of(&hm, "xMax");
    let atx = hm.patches[xmx].start;
    let npx = hm.patches[xmx].size;
    let area: f64 = (atx..atx + npx).map(|f| w64(hm.b_mag_sf[f])).sum();
    let tau = 12.0 * 1000.0 * 0.1 / (0.05 * area);
    eprintln!("  [120.8] lump: A {area:e}, tau {tau:e} s");

    let run = |ddt: DdtScheme, n: usize| -> Result<f64> {
        let dt = tau / n as f64;
        let raw = raw_t(
            vec![
                ("xMin", fixed_value(300.0)),
                ("xMax", lump_bc(&[("mass", "12"), ("Cp", "1000")], Some(400.0))),
            ],
            300.0,
        );
        let (mut e, nut, phi) = energy_on(&gpu, &gm, props.clone(), false, ddt, dt)?;
        let mut gas = GasState::new(&gpu, &gm, props.clone(), DomainKind::Open, 101325.0)?;
        // The lump names its faces by flattened boundary-face index - the
        // index `ThermalFace::bf` carries - not by position in the face list.
        let walls = thermal_walls(&raw, &hm)?;
        assert_eq!(walls.lumps.len(), 1, "one lump");
        assert_eq!(
            walls.lumps[0].faces,
            (atx..atx + npx).collect::<Vec<_>>(),
            "the lump's faces are xMax's boundary faces"
        );
        wire_t(&gpu, &hm, &raw, &mut e, &mut gas)?;
        for step in 0..n {
            // `open_time_step`: from step 1 on, the levels and the ddt
            // counter - the FIRST step opens nothing and takes the Euler row.
            if step > 0 {
                e.advance_time_levels(&gpu)?;
                e.advance_time_step(dt as Scalar);
                gas.advance_time_levels();
            }
            e.advance_lumped_masses(&gpu)?;
            e.correct(&gpu, &phi, &nut, &nut.f, 1.5e-5, &gas)?;
        }
        e.lumped_mass_temperatures(&gpu).map(|v| v[0])
    };

    let exact = 300.0 + 100.0 * (-1.0_f64).exp();
    let err = |tm: f64| {
        fin("T_m", tm);
        (tm - exact).abs() / 100.0
    };
    let e50 = err(run(DdtScheme::Backward, 50)?);
    let e25 = err(run(DdtScheme::Backward, 25)?);
    let eu = err(run(DdtScheme::Euler, 50)?);
    let ratio = e25 / e50;
    eprintln!("  [120.8] lump: err(N=50) {e50:e}, err(N=25) {e25:e}, ratio {ratio:e}, Euler {eu:e}");
    assert!(e50 <= 1e-4, "second order at N=50: {e50:e}");
    assert!(
        (3.5..=4.5).contains(&ratio),
        "second order halves the error fourfold: {ratio}"
    );
    assert!(eu > 1e-3, "Euler control is first order: {eu:e}");
    Ok(())
}

// ==========================================================================
//  (120.8h) - Danckwerts' totalFlowRateAdvectiveDiffusive
// ==========================================================================

/// The channel's `phi`: a +x plug flow of `speed` m/s - `speed |S_f|` on the
/// internal faces in the owner-to-neighbour direction, `-speed |S|` in at
/// xMin, `+speed |S|` out at xMax, zero everywhere else.
fn plug_phi(
    gpu: &Gpu,
    m: &GpuMesh,
    hm: &HostMesh,
    speed: f64,
) -> Result<GpuSurfaceScalarField> {
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

/// THE gate, row 3 (SPEC-LIT 120.8): Danckwerts' inlet
/// `u c_in = u c - D dc/dx` is exactly the (120.8h) Robin triple at the
/// face - `a T_b + C_b (T_b - T_P) = a T_in` at EVERY pass - so a uniform
/// `T_in` fills the channel exactly, the converged inlet face sits below
/// `T_in` where the 300 K outflow pulls on it, and whatever comes in leaves
/// xMax whole. On an OUTFLOW face `H = max(-phi_conv, 0)/|S| = 0`, which is
/// `fr = 0, refGrad = 0` - zeroGradient - to the last bit.
#[test]
fn danckwerts_inlet_carries_the_advected_heat() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = blockgen::build_mesh(&channel())?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let props = GasProperties { k: 12.0, ..GasProperties::default() };
    let xmn = patch_of(&hm, "xMin");
    let xmx = patch_of(&hm, "xMax");
    let atn = hm.patches[xmn].start;
    let atx = hm.patches[xmx].start;
    let npn = hm.patches[xmn].size;
    let npx = hm.patches[xmx].size;
    let phi = plug_phi(&gpu, &gm, &hm, 0.01)?;

    // (a) uniform 310 in, zeroGradient out: 310 fills every cell.
    {
        let raw = raw_t(
            vec![("xMin", danck_bc(310.0)), ("xMax", ps("zeroGradient"))],
            300.0,
        );
        let (mut e, nut, _) =
            energy_on(&gpu, &gm, props.clone(), true, DdtScheme::SteadyState, 0.0)?;
        let mut gas = GasState::new(&gpu, &gm, props.clone(), DomainKind::Open, 101325.0)?;
        wire_t(&gpu, &hm, &raw, &mut e, &mut gas)?;
        let (passes, _, tf) = settle(&gpu, &hm, &mut e, &phi, &nut, &gas, xmn, 300)?;
        let mut worst = 0.0f64;
        for c in 0..hm.n_cells {
            worst = nmax(worst, (w64(tf[c]) / 310.0 - 1.0).abs());
        }
        eprintln!("  [120.8] Danckwerts (a): {passes} passes, worst cell rel err {worst:e}");
        assert!(passes <= 300, "(a) took {passes} passes");
        assert!(worst <= tol(1e-10), "(a) cell rel err {worst:e}");
    }

    // (b) uniform 301 in against a fixed 300 outflow: the triple IS
    // Danckwerts' condition at every pass, the converged inlet face sits
    // below T_in, and the outflow carries the advected inflow out whole.
    {
        let raw = raw_t(
            vec![("xMin", danck_bc(301.0)), ("xMax", fixed_value(300.0))],
            300.0,
        );
        let (mut e, nut, _) =
            energy_on(&gpu, &gm, props.clone(), true, DdtScheme::SteadyState, 0.0)?;
        let mut gas = GasState::new(&gpu, &gm, props.clone(), DomainKind::Open, 101325.0)?;
        wire_t(&gpu, &hm, &raw, &mut e, &mut gas)?;
        let (mut worst_id, mut prev) = (0.0f64, vec![f64::NAN; npn]);
        let mut passes = 0usize;
        for pass in 0..300 {
            e.correct(&gpu, &phi, &nut, &nut.f, 1.5e-5, &gas)?;
            passes = pass + 1;
            let tb = gpu.download(&e.field().bf)?;
            let tf = gpu.download(&e.field().f)?;
            let pcv = gpu.download(&e.conv_flux().bf)?;
            let keff = gpu.download(e.k_eff_wall())?;
            for i in 0..npn {
                let f = atn + i;
                let (tbv, tp) = (w64(tb[f]), w64(tf[hm.b_face_cells[f] as usize]));
                let a = -w64(pcv[f]) / fin("|S_f|", w64(hm.b_mag_sf[f]));
                let cb = w64(keff[f]) * w64(hm.b_delta_coeffs[f]);
                fin("a T_b + C_b (T_b - T_P)", a * tbv + cb * (tbv - tp));
                worst_id = nmax(
                    worst_id,
                    ((a * tbv + cb * (tbv - tp)) / (a * 301.0) - 1.0).abs(),
                );
            }
            let (mut change, mut scale) = (0.0f64, 0.0f64);
            for i in 0..npn {
                change = nmax(change, (w64(tb[atn + i]) - prev[i]).abs());
                scale = nmax(scale, w64(tb[atn + i]).abs());
            }
            prev = tb[atn..atn + npn].iter().map(|&v| w64(v)).collect();
            if pass > 0 && fin("dT_b", change) <= tol(1e-13) * fin("T_b", scale) {
                break;
            }
        }
        eprintln!("  [120.8] Danckwerts (b): {passes} passes, per-pass identity worst {worst_id:e}");
        assert!(passes <= 300, "(b) took {passes} passes");
        assert!(worst_id <= tol(1e-12), "(b) identity rel err {worst_id:e}");

        // Converged: the inlet face sits below T_in, the outflow carries
        // exactly the advected inflow, and the fixedValue control pins 301.
        let tb = gpu.download(&e.field().bf)?;
        let tf = gpu.download(&e.field().f)?;
        let pcv = gpu.download(&e.conv_flux().bf)?;
        let keff = gpu.download(e.k_eff_wall())?;
        let tb_in = (0..npn).map(|i| w64(tb[atn + i])).fold(f64::INFINITY, f64::min);
        eprintln!("  [120.8] Danckwerts (b): converged xMin T_b {tb_in:.12} (< 300.9)");
        assert!(tb_in < 300.9, "(b) inlet face T_b {tb_in}");
        let out_xmax: f64 = (0..npx)
            .map(|i| {
                let f = atx + i;
                let tp = w64(tf[hm.b_face_cells[f] as usize]);
                let cb = w64(keff[f]) * w64(hm.b_delta_coeffs[f]);
                w64(pcv[f]) * w64(tb[f]) + cb * (tp - w64(tb[f])) * w64(hm.b_mag_sf[f])
            })
            .sum();
        let in_xmin: f64 = (0..npn).map(|i| -w64(pcv[atn + i]) * 301.0).sum();
        let bal = (out_xmax / in_xmin - 1.0).abs();
        eprintln!(
            "  [120.8] Danckwerts (b): out xMax {out_xmax:.9} vs a|S|T_in {in_xmin:.9}, rel {bal:e}"
        );
        assert!(bal <= tol(1e-10), "(b) outflow balance rel err {bal:e}");

        let rawc = raw_t(
            vec![("xMin", fixed_value(301.0)), ("xMax", fixed_value(300.0))],
            300.0,
        );
        let (mut ec, nutc, _) =
            energy_on(&gpu, &gm, props.clone(), true, DdtScheme::SteadyState, 0.0)?;
        let mut gasc = GasState::new(&gpu, &gm, props.clone(), DomainKind::Open, 101325.0)?;
        wire_t(&gpu, &hm, &rawc, &mut ec, &mut gasc)?;
        let (_, tbc, _) = settle(&gpu, &hm, &mut ec, &phi, &nutc, &gasc, xmn, 300)?;
        for i in 0..npn {
            assert_eq!(w64(tbc[atn + i]), 301.0, "(b) control xMin face {i}");
        }
        eprintln!("  [120.8] Danckwerts (b): fixedValue control pins T_b = 301 exactly");

        // (c) Danckwerts on the OUTFLOW too: H = 0 gives fr = 0/(0 + C_b)
        // and refGrad = 0/k_eff - both EXACT zeros, zeroGradient to the
        // last bit.
        let raw = raw_t(
            vec![("xMin", danck_bc(301.0)), ("xMax", danck_bc(300.0))],
            300.0,
        );
        let (mut e2, nut2, _) =
            energy_on(&gpu, &gm, props.clone(), true, DdtScheme::SteadyState, 0.0)?;
        let mut gas2 = GasState::new(&gpu, &gm, props.clone(), DomainKind::Open, 101325.0)?;
        wire_t(&gpu, &hm, &raw, &mut e2, &mut gas2)?;
        e2.correct(&gpu, &phi, &nut2, &nut2.f, 1.5e-5, &gas2)?;
        let fr = gpu.download(&e2.field().fr)?;
        let rg = gpu.download(&e2.field().ref_grad)?;
        for i in 0..npx {
            assert_eq!(w64(fr[atx + i]), 0.0, "(c) xMax fr face {i}");
            assert_eq!(w64(rg[atx + i]), 0.0, "(c) xMax ref_grad face {i}");
        }
        eprintln!("  [120.8] Danckwerts (c): fr = ref_grad = 0 exactly on {npx} outflow faces");
    }
    Ok(())
}

// ==========================================================================
//  Refusals by name, and the two aliases
// ==========================================================================

/// SPEC-LIT 120.8's refusal list: the three Mixed-kind names are refused
/// without the `BcInputs::thermal` opt-in, every entry-level refusal names
/// the offending key or mode, all four names are scalar conditions on a
/// vector field - and on the device a lump is refused on a steady `Energy`,
/// and `correct` is refused before `advance_lumped_masses`.
#[test]
fn thermal_refusals_are_by_name() -> Result<()> {
    let hm = blockgen::build_mesh(&slab())?;
    let internal = vec![300.0 as Scalar; hm.n_cells];
    let refused = |name: &str, spec: PatchFieldSpec, thermal: bool| -> String {
        let raw = raw_t(vec![("xMax", spec)], 300.0);
        match scalar_boundary_with(
            &raw,
            &hm,
            &internal,
            BcInputs { thermal, ..BcInputs::default() },
        ) {
            Ok(_) => panic!("{name} was accepted"),
            Err(Error::Field { msg, .. }) => msg,
            Err(other) => panic!("{name}: wrong error {other:?}"),
        }
    };
    let has = |msg: &str, parts: &[&str]| {
        for p in parts {
            assert!(msg.contains(p), "message is missing `{p}`: {msg}");
        }
        assert!(msg.contains("SPEC-LIT 120.8"), "no section: {msg}");
    };

    // The opt-in gate: the three names whose triple only `Energy` can
    // rewrite, with `BcInputs::thermal = false`.
    for name in [
        "externalWallHeatFluxTemperature",
        "lumpedMassTemperature",
        "totalFlowRateAdvectiveDiffusive",
    ] {
        let msg = refused(name, ps(name), false);
        has(&msg, &[name, "BcInputs::thermal = false"]);
    }

    let ext = |extra: &[(&str, &str)]| th_bc("externalWallHeatFluxTemperature", extra);
    let msg = refused("modeless", ext(&[]), true);
    has(&msg, &["needs a `mode` entry"]);
    let msg = refused("bad mode", ext(&[("mode", "banana")]), true);
    has(&msg, &["`mode banana` is not one of flux, power, coefficient"]);
    let msg = refused("flux", ext(&[("mode", "flux")]), true);
    has(&msg, &["needs a `q` entry"]);
    let msg = refused(
        "flux reads h",
        ext(&[("mode", "flux"), ("q", "-50"), ("h", "10")]),
        true,
    );
    has(&msg, &["does not read `h`"]);
    let msg = refused("power", ext(&[("mode", "power")]), true);
    has(&msg, &["needs a `Q` entry"]);
    let msg = refused(
        "coefficient reads q",
        ext(&[("mode", "coefficient"), ("h", "10"), ("Ta", "300"), ("q", "-50")]),
        true,
    );
    has(&msg, &["does not read `q`"]);
    let msg = refused("coefficient", ext(&[("mode", "coefficient")]), true);
    has(&msg, &["needs a `h` entry"]);
    let msg = refused("Ta", ext(&[("mode", "coefficient"), ("h", "10")]), true);
    has(&msg, &["needs a `Ta` entry"]);
    let msg = refused(
        "h",
        ext(&[("mode", "coefficient"), ("h", "-1"), ("Ta", "300")]),
        true,
    );
    has(&msg, &["`h = -1` is not finite and positive"]);
    let msg = refused(
        "emissivity",
        ext(&[
            ("mode", "coefficient"),
            ("h", "10"),
            ("Ta", "300"),
            ("emissivity", "1.5"),
        ]),
        true,
    );
    has(&msg, &["`emissivity = 1.5` is outside [0, 1]"]);
    let msg = refused(
        "layers",
        ext(&[
            ("mode", "coefficient"),
            ("h", "10"),
            ("Ta", "300"),
            ("thicknessLayers", "(0.01 0.02)"),
            ("kappaLayers", "(0.5)"),
        ]),
        true,
    );
    has(&msg, &["the layers are refused"]);

    let msg = refused("heatSourceless", th_bc("turbulentHeatFluxTemperature", &[]), true);
    has(&msg, &["needs a `heatSource` entry"]);
    let msg = refused(
        "heatSource",
        th_bc("turbulentHeatFluxTemperature", &[("heatSource", "radiation")]),
        true,
    );
    has(&msg, &["`heatSource radiation` is not one of flux, power"]);
    let msg = refused(
        "turbulent q",
        th_bc("turbulentHeatFluxTemperature", &[("heatSource", "flux")]),
        true,
    );
    has(&msg, &["needs a `q` entry"]);

    let lump = |extra: &[(&str, &str)], t0: Option<f64>| refused(
        "lumpedMassTemperature",
        lump_bc(extra, t0),
        true,
    );
    let msg = lump(&[("Cp", "1000")], Some(400.0));
    has(&msg, &["needs a `mass` entry"]);
    let msg = lump(&[("mass", "0"), ("Cp", "1000")], Some(400.0));
    has(&msg, &["`mass = 0` is not finite and positive"]);
    let msg = lump(&[("mass", "12")], Some(400.0));
    has(&msg, &["needs a `Cp` entry"]);
    let msg = lump(&[("mass", "12"), ("Cp", "1000")], None);
    has(&msg, &["needs a `value` entry"]);

    let msg = refused("inletValueless", ps("totalFlowRateAdvectiveDiffusive"), true);
    has(&msg, &["needs an `inletValue` entry"]);
    let two = PatchFieldSpec {
        type_name: "totalFlowRateAdvectiveDiffusive".into(),
        inlet_value: vec![300.0 as Scalar, 310.0],
        ..Default::default()
    };
    let msg = refused("two inletValues", two, true);
    has(&msg, &["`inletValue` is a uniform entry"]);

    // All four names are scalar conditions: refused on a vector field by
    // NAME, before any kind is derived.
    let iv = vec![Vec3 { x: 0.0, y: 0.0, z: 0.0 }; hm.n_cells];
    for name in [
        "externalWallHeatFluxTemperature",
        "turbulentHeatFluxTemperature",
        "lumpedMassTemperature",
        "totalFlowRateAdvectiveDiffusive",
    ] {
        let rv = RawVectorField {
            name: "U".into(),
            internal: iv.clone(),
            boundary: [("xMax".to_string(), ps(name))].into_iter().collect(),
            ..Default::default()
        };
        let msg = match vector_boundary_with(&rv, &hm, &iv, BcInputs::default()) {
            Err(Error::Field { msg, .. }) => msg,
            Err(other) => panic!("{name}: wrong error {other:?}"),
            Ok(_) => panic!("{name} on U was accepted"),
        };
        has(&msg, &[name]);
    }

    // The device part: a lump on a steady Energy has no `a_n` to grow by,
    // and `correct` before `advance_lumped_masses` would solve with the
    // seed instead of the lump's level.
    if let Some(gpu) = Gpu::new(0).ok() {
        let gm = GpuMesh::upload(&gpu, &hm)?;
        let props = GasProperties { k: 0.05, ..GasProperties::default() };
        let lump_raw = raw_t(
            vec![
                ("xMin", fixed_value(300.0)),
                ("xMax", lump_bc(&[("mass", "12"), ("Cp", "1000")], Some(400.0))),
            ],
            300.0,
        );
        let (mut es, _, _) =
            energy_on(&gpu, &gm, props.clone(), true, DdtScheme::SteadyState, 0.0)?;
        let walls = thermal_walls(&lump_raw, &hm)?;
        let msg = match es.set_thermal_walls(&gpu, &walls) {
            Err(Error::Config(m)) => m,
            Err(other) => panic!("wrong error {other:?}"),
            Ok(()) => panic!("a steady Energy took a lump"),
        };
        has(&msg, &["lumpedMassTemperature"]);

        // A lump whose face is not one of the thermal faces (xMin's bf, while
        // only xMax is listed) has nowhere to write its film: refused by name.
        let (mut eb, _, _) =
            energy_on(&gpu, &gm, props.clone(), false, DdtScheme::Backward, 1.0e6)?;
        let mut bad = thermal_walls(&lump_raw, &hm)?;
        let xmn = hm.patches[patch_of(&hm, "xMin")].start;
        bad.lumps[0].faces = vec![xmn];
        let msg = match eb.set_thermal_walls(&gpu, &bad) {
            Err(Error::Config(m)) => m,
            Err(other) => panic!("wrong error {other:?}"),
            Ok(()) => panic!("a lump face outside the thermal faces was taken"),
        };
        has(&msg, &["lumpedMassTemperature", "not a thermal face"]);

        let (mut et, nut, phi) =
            energy_on(&gpu, &gm, props.clone(), false, DdtScheme::Backward, 1.0e6)?;
        let mut gas = GasState::new(&gpu, &gm, props.clone(), DomainKind::Open, 101325.0)?;
        wire_t(&gpu, &hm, &lump_raw, &mut et, &mut gas)?;
        let msg = match et.correct(&gpu, &phi, &nut, &nut.f, 1.5e-5, &gas) {
            Err(Error::Config(m)) => m,
            Err(other) => panic!("wrong error {other:?}"),
            Ok(_) => panic!("correct ran before advance_lumped_masses"),
        };
        has(&msg, &["advance_lumped_masses"]);
        eprintln!("  [120.8] refusals: steady lump and unadvanced correct both refused by name");
    }
    Ok(())
}

/// SPEC-LIT 120.8: the two alphat spellings read as `calculated` on the
/// `alphat` field - this solver folds the turbulent thermal diffusivity
/// into `k_eff` and carries no alphat - and on any other field they are
/// refused, naming `alphat`.
#[test]
fn the_alphat_aliases_read_as_calculated_on_alphat_only() -> Result<()> {
    for name in ["alphatWallFunction", "compressible::alphatWallFunction"] {
        let kind = BcKind::from_name(name, "alphat", "wall")
            .unwrap_or_else(|e| panic!("{name} on alphat was refused: {e:?}"));
        assert_eq!(kind, BcKind::Calculated, "{name} on alphat");
        let msg = match BcKind::from_name(name, "T", "wall") {
            Err(Error::Refused(m)) => m,
            other => panic!("{name} on T: wrong outcome {other:?}"),
        };
        assert!(msg.contains("alphat"), "no alphat in: {msg}");
    }
    eprintln!("  [120.8] alphat aliases: calculated on alphat, refused elsewhere naming alphat");
    Ok(())
}







