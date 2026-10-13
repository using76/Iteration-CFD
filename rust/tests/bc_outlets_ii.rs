// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 120.6 - the Outlets II family through the public API, in BOTH
//! builds: `fixedMean` holding its patch mean exactly ((120.6a)),
//! `fixedMeanOutletInlet` holding it on outflow only ((120.6b)),
//! `outletMappedUniformInlet` carrying the source patch's mixing-cup mean
//! ((120.6c)) and the loop duct closing on `offset/(1 - fraction)`
//! ((120.6d)), `matchedFlowRateOutletVelocity` carrying the inlet's flow
//! rate ((120.6e)), and `fixedFluxExtrapolatedPressure` exact on a linear
//! field ((120.6f)) - each against its host twin, plus the refusals by
//! name. The tolerances follow docs/17 §5.1 class R, D-F32-1 - the f64
//! tolerance in the f64 build, `max(t, min(t*2^29, 1e-5))` under `single` -
//! which the `tol` const fn below encodes; everything is compared in f64.
//! A machine without a card passes vacuously, the convention of
//! `tests/bc_inlets.rs`.
//!
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::field::patch_mean::{
    extrapolated_sn_grad, patch_mean_values_scalar, patch_mean_values_vector,
};
use ofgpu::field_ops::{
    correct_boundary_conditions, correct_boundary_conditions_vector, update_patch_means_scalar,
    update_patch_means_vector, FieldKernels,
};
use ofgpu::field_setup::{
    scalar_boundary_with, setup_scalar_field_with, setup_vector_field_with, vector_boundary_with,
    BcInputs,
};
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::momentum::{BuoyancyCoeffs, MomentumControls};
use ofgpu::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use ofgpu::simple::{Simple, SimpleControls};
use ofgpu::{BcKind, Gpu, GpuMesh, GpuScalarField, GpuSurfaceScalarField, GpuVectorField, HostMesh, Result, Scalar, Vec3};

/// `t` in the f64 build; under `single` (class R, D-F32-1)
/// `max(t, min(t*2^29, 1e-5))` - i.e. 1e-12 -> 1e-5, 1e-14 -> 5.36870912e-6.
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

/// A patch entry of one of the five Outlets II names, with its entries.
fn outlet_bc(name: &str, extra: &[(&str, &str)]) -> PatchFieldSpec {
    let mut s = ps(name);
    for (k, v) in extra {
        s.extra.insert((*k).into(), (*v).into());
    }
    s
}

/// Tight enough that what the tests measure is not the linear solver's
/// stopping criterion.
fn tight() -> ofgpu::solver::SolverControls {
    ofgpu::solver::SolverControls {
        tolerance: 1e-14,
        rel_tol: 0.0,
        max_iter: 500,
        check_interval: 1,
        ..ofgpu::solver::SolverControls::default()
    }
}

/// Laminar: zero eddy viscosity with `calculated` boundary values.
fn laminar_nut(gpu: &Gpu, hm: &HostMesh, gm: &GpuMesh) -> Result<GpuScalarField> {
    let mut nut = GpuScalarField::zeros(gpu, gm, "nut")?;
    gpu.write(
        &mut nut.bc_kind,
        &vec![BcKind::Calculated as i32; hm.n_boundary_faces],
    )?;
    Ok(nut)
}

/// A uniform T at rest, zeroGradient everywhere - nothing reads it while the
/// buoyancy vector is zero, but `correct` takes one.
fn uniform_t(gpu: &Gpu, hm: &HostMesh, gm: &GpuMesh) -> Result<GpuScalarField> {
    let mut t = GpuScalarField::zeros(gpu, gm, "T")?;
    gpu.write(
        &mut t.bc_kind,
        &vec![BcKind::ZeroGradient as i32; hm.n_boundary_faces],
    )?;
    Ok(t)
}

/// A raw vector field named U, one uniform internal value, the given patch
/// entries.
fn raw_u(boundary: Vec<(&str, PatchFieldSpec)>, internal: Vec3) -> RawVectorField {
    let mut raw = RawVectorField {
        name: "U".into(),
        ..Default::default()
    };
    for (p, s) in boundary {
        raw.boundary.insert(p.into(), s);
    }
    raw.internal = vec![internal];
    raw
}

/// A raw scalar field named p, uniform zero inside, the given patch entries.
fn raw_p(boundary: Vec<(&str, PatchFieldSpec)>) -> RawScalarField {
    let mut raw = RawScalarField {
        name: "p".into(),
        ..Default::default()
    };
    for (p, s) in boundary {
        raw.boundary.insert(p.into(), s);
    }
    raw.internal = vec![0.0];
    raw
}

/// A raw scalar field named T, uniform zero inside, the given patch entries.
fn raw_t(boundary: Vec<(&str, PatchFieldSpec)>) -> RawScalarField {
    let mut raw = RawScalarField {
        name: "T".into(),
        ..Default::default()
    };
    for (p, s) in boundary {
        raw.boundary.insert(p.into(), s);
    }
    raw.internal = vec![0.0];
    raw
}

/// All six patches zeroGradient, in patch order.
fn zg_all() -> Vec<(&'static str, PatchFieldSpec)> {
    vec![
        ("xMin", ps("zeroGradient")),
        ("xMax", ps("zeroGradient")),
        ("yMin", ps("zeroGradient")),
        ("yMax", ps("zeroGradient")),
        ("zMin", ps("zeroGradient")),
        ("zMax", ps("zeroGradient")),
    ]
}

/// Mesh B: 4 x 3 x 2 cells over [0,2] x [0,1] x [0,1], all six patches
/// `patch` - the xMin/xMax faces 1/6 each, the xMin patch 6 faces.
fn mesh_b(gpu: &Gpu) -> Result<(HostMesh, GpuMesh)> {
    let hm = blockgen::build_mesh(&box_spec(
        [4, 3, 2],
        [0.0, 0.0, 0.0],
        [2.0, 1.0, 1.0],
        ["patch", "patch", "patch", "patch", "patch", "patch"],
    ))?;
    let gm = GpuMesh::upload(gpu, &hm)?;
    Ok((hm, gm))
}

/// T_c = 1 + x + 2y^2 + 0.5z at the f64 cell centres, cast once.
fn scalar_cells_t(hm: &HostMesh) -> Vec<Scalar> {
    hm.geom64()
        .c
        .iter()
        .map(|p| (1.0 + p.x + 2.0 * p.y * p.y + 0.5 * p.z) as Scalar)
        .collect()
}

/// p_c = 3 + 2x - y + 0.5z at the f64 cell centres, cast once.
fn scalar_cells_p(hm: &HostMesh) -> Vec<Scalar> {
    hm.geom64()
        .c
        .iter()
        .map(|p| (3.0 + 2.0 * p.x - p.y + 0.5 * p.z) as Scalar)
        .collect()
}

/// U_c = (1 + x, y - z, 2z) at the f64 cell centres, cast once.
fn vector_cells_u(hm: &HostMesh) -> Vec<Vec3> {
    hm.geom64()
        .c
        .iter()
        .map(|p| {
            Vec3::new(
                (1.0 + p.x) as Scalar,
                (p.y - p.z) as Scalar,
                (2.0 * p.z) as Scalar,
            )
        })
        .collect()
}

// ==========================================================================
//  (120.6a) - fixedMean holds its patch mean
// ==========================================================================

#[test]
fn fixed_mean_holds_the_patch_mean() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let (hm, gm) = mesh_b(&gpu)?;
    let k = FieldKernels::new(&gpu)?;
    let xout = patch_of(&hm, "xMax");
    let at = hm.patches[xout].start;
    let np = hm.patches[xout].size;
    let n = hm.n_boundary_faces;

    let mut bounds = zg_all();
    bounds[1] = ("xMax", outlet_bc("fixedMean", &[("meanValue", "uniform 5")]));
    let mut t = GpuScalarField::zeros(&gpu, &gm, "T")?;
    setup_scalar_field_with(
        &gpu,
        &mut t,
        &raw_t(bounds),
        &hm,
        BcInputs {
            patch_means: true,
            ..BcInputs::default()
        },
    )?;
    let tc = scalar_cells_t(&hm);
    gpu.write(&mut t.f, &tc)?;
    let phi = GpuSurfaceScalarField::zeros(&gpu, &gm, "phi")?;
    correct_boundary_conditions(&gpu, &k, &mut t, &gm)?;
    update_patch_means_scalar(&gpu, &k, &mut t, &phi, None, &gm)?;
    correct_boundary_conditions(&gpu, &k, &mut t, &gm)?;
    gpu.sync()?;

    let bf = gpu.download(&t.bf)?;
    let fr = gpu.download(&t.fr)?;
    let cells = gpu.download(&t.f)?;
    let rv = gpu.download(&t.ref_value)?;
    let phi_b = gpu.download(&phi.bf)?;

    let mut max_t = 0.0f64;
    for v in &tc {
        max_t = nmax(max_t, w64(*v).abs());
    }
    for v in &bf {
        max_t = nmax(max_t, w64(*v).abs());
    }

    // (a): the area mean is the target, to round-off.
    let (mut sa, mut sb) = (0.0f64, 0.0f64);
    for i in 0..np {
        let f = at + i;
        sa += fin("|S_f|", w64(hm.b_mag_sf[f]));
        sb += fin("|S_f| psi_b", w64(hm.b_mag_sf[f]) * w64(bf[f]));
    }
    let mean = sb / sa;
    eprintln!("  [120.6] fixedMean: area mean {mean:.12} (want 5)");
    assert!(
        (mean - 5.0).abs() <= tol(1e-12) * 5.0,
        "area mean {mean:e}, want 5"
    );

    // (b): one constant shift - bf_f - T_P,f equal on every face of the
    // patch.
    let d0 = w64(bf[at]) - w64(cells[hm.b_face_cells[at] as usize]);
    for i in 0..np {
        let f = at + i;
        let d = w64(bf[f]) - w64(cells[hm.b_face_cells[f] as usize]);
        fin("bf - T_P", d);
        assert!(
            (d - d0).abs() <= tol(1e-14) * max_t,
            "face {i}: shift {d:e} vs {d0:e}"
        );
    }

    // (c): fr exactly 1 on the patch.
    for i in 0..np {
        assert_eq!(w64(fr[at + i]), 1.0, "fr face {i}");
    }

    // (d): the downloaded ref_value equals the host twin, to round-off.
    assert_eq!(t.mean_rules.len(), 1, "one rule on xMax");
    let mut fr2 = vec![0.0 as Scalar; n];
    let mut rv2 = vec![0.0 as Scalar; n];
    let mut rg2 = vec![0.0 as Scalar; n];
    patch_mean_values_scalar(
        &t.mean_rules[0],
        &cells,
        &bf,
        &phi_b,
        &hm.b_mag_sf,
        &hm.b_face_cells,
        &mut fr2,
        &mut rv2,
        &mut rg2,
    );
    for i in 0..np {
        let dd = (w64(rv[at + i]) - w64(rv2[at + i])).abs();
        fin("ref_value twin", dd);
        assert!(dd <= tol(1e-14) * max_t, "face {i}: twin gap {dd:e}");
    }

    // The vector twin: U_c = (1 + x, y - z, 2z), mean (1 2 3).
    let mut vb = zg_all();
    vb[1] = (
        "xMax",
        outlet_bc("fixedMean", &[("meanValue", "uniform (1 2 3)")]),
    );
    let mut u = GpuVectorField::zeros(&gpu, &gm, "U")?;
    setup_vector_field_with(
        &gpu,
        &mut u,
        &raw_u(vb, Vec3::ZERO),
        &hm,
        BcInputs {
            patch_means: true,
            ..BcInputs::default()
        },
    )?;
    let uc = vector_cells_u(&hm);
    gpu.write(&mut u.f, &uc)?;
    let phi2 = GpuSurfaceScalarField::zeros(&gpu, &gm, "phi")?;
    correct_boundary_conditions_vector(&gpu, &k, &mut u, &gm)?;
    update_patch_means_vector(&gpu, &k, &mut u, &phi2, &gm)?;
    correct_boundary_conditions_vector(&gpu, &k, &mut u, &gm)?;
    gpu.sync()?;

    let ubf = gpu.download(&u.bf)?;
    for c in 0..3 {
        let (mut sa, mut sb) = (0.0f64, 0.0f64);
        for i in 0..np {
            let f = at + i;
            let b = ubf[f];
            let v = [w64(b.x), w64(b.y), w64(b.z)][c];
            sa += fin("|S_f|", w64(hm.b_mag_sf[f]));
            sb += fin("|S_f| U_f", w64(hm.b_mag_sf[f]) * v);
        }
        let mean = sb / sa;
        let want = (c + 1) as f64;
        eprintln!(
            "  [120.6] fixedMean vector: component {c} mean {mean:.12} (want {want})"
        );
        assert!(
            (mean - want).abs() <= tol(1e-12) * 3.0,
            "component {c}: mean {mean:e}, want {want}"
        );
    }
    assert_eq!(u.mean_rules.len(), 1, "one vector rule");
    Ok(())
}

// ==========================================================================
//  (120.6b) - fixedMeanOutletInlet, switched by the flux
// ==========================================================================

#[test]
fn fixed_mean_outlet_inlet_holds_the_mean_on_a_two_signed_flux() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let (hm, gm) = mesh_b(&gpu)?;
    let k = FieldKernels::new(&gpu)?;
    let xout = patch_of(&hm, "xMax");
    let at = hm.patches[xout].start;
    let np = hm.patches[xout].size;
    let n = hm.n_boundary_faces;

    let mut bounds = zg_all();
    bounds[1] = (
        "xMax",
        outlet_bc("fixedMeanOutletInlet", &[("meanValue", "uniform 5")]),
    );
    let mut t = GpuScalarField::zeros(&gpu, &gm, "T")?;
    setup_scalar_field_with(
        &gpu,
        &mut t,
        &raw_t(bounds),
        &hm,
        BcInputs {
            patch_means: true,
            ..BcInputs::default()
        },
    )?;
    let tc = scalar_cells_t(&hm);
    gpu.write(&mut t.f, &tc)?;
    let mut phi = GpuSurfaceScalarField::zeros(&gpu, &gm, "phi")?;
    let mut phib = vec![0.0 as Scalar; n];
    for i in 0..np {
        let f = at + i;
        phib[f] = if i % 2 == 0 {
            hm.b_mag_sf[f]
        } else {
            -hm.b_mag_sf[f]
        };
    }
    gpu.write(&mut phi.bf, &phib)?;
    correct_boundary_conditions(&gpu, &k, &mut t, &gm)?;
    update_patch_means_scalar(&gpu, &k, &mut t, &phi, None, &gm)?;
    correct_boundary_conditions(&gpu, &k, &mut t, &gm)?;
    gpu.sync()?;

    let bf = gpu.download(&t.bf)?;
    let fr = gpu.download(&t.fr)?;
    let cells = gpu.download(&t.f)?;
    let rv = gpu.download(&t.ref_value)?;
    let phi_b = gpu.download(&phi.bf)?;

    let mut max_t = 0.0f64;
    for v in &tc {
        max_t = nmax(max_t, w64(*v).abs());
    }
    for v in &bf {
        max_t = nmax(max_t, w64(*v).abs());
    }

    // fr exactly 1 on outflow, exactly 0 on inflow; the inflow faces carry
    // psi_b = psi_P bit for bit.
    for i in 0..np {
        let f = at + i;
        let want = if w64(phib[f]) > 0.0 { 1.0 } else { 0.0 };
        assert_eq!(w64(fr[f]), want, "fr face {i}");
        if want == 0.0 {
            let c = hm.b_face_cells[f] as usize;
            assert_eq!(w64(bf[f]), w64(cells[c]), "inflow bf face {i}");
        }
    }

    // The area mean over the WHOLE patch is still the target: the inflow
    // faces contribute psi_P exactly, the outflow faces the shifted
    // profile.
    let (mut sa, mut sb) = (0.0f64, 0.0f64);
    for i in 0..np {
        let f = at + i;
        sa += fin("|S_f|", w64(hm.b_mag_sf[f]));
        sb += fin("|S_f| psi_b", w64(hm.b_mag_sf[f]) * w64(bf[f]));
    }
    let mean = sb / sa;
    eprintln!("  [120.6] fixedMeanOutletInlet: area mean {mean:.12} (want 5)");
    assert!(
        (mean - 5.0).abs() <= tol(1e-12) * 5.0,
        "area mean {mean:e}, want 5"
    );

    // The host twin, on the switched rule.
    assert_eq!(t.mean_rules.len(), 1, "one rule on xMax");
    let mut fr2 = vec![0.0 as Scalar; n];
    let mut rv2 = vec![0.0 as Scalar; n];
    let mut rg2 = vec![0.0 as Scalar; n];
    patch_mean_values_scalar(
        &t.mean_rules[0],
        &cells,
        &bf,
        &phi_b,
        &hm.b_mag_sf,
        &hm.b_face_cells,
        &mut fr2,
        &mut rv2,
        &mut rg2,
    );
    for i in 0..np {
        let dd = (w64(rv[at + i]) - w64(rv2[at + i])).abs();
        fin("ref_value twin", dd);
        assert!(dd <= tol(1e-14) * max_t, "face {i}: twin gap {dd:e}");
    }

    // All inflow: A_D = 0, s = 0, the patch is zero-gradient throughout -
    // bf = psi_P bit for bit, fr all 0.
    for i in 0..np {
        phib[at + i] = -hm.b_mag_sf[at + i];
    }
    gpu.write(&mut phi.bf, &phib)?;
    update_patch_means_scalar(&gpu, &k, &mut t, &phi, None, &gm)?;
    correct_boundary_conditions(&gpu, &k, &mut t, &gm)?;
    gpu.sync()?;
    let bf2 = gpu.download(&t.bf)?;
    let fr2d = gpu.download(&t.fr)?;
    for i in 0..np {
        let f = at + i;
        assert_eq!(w64(fr2d[f]), 0.0, "fr face {i}");
        assert_eq!(
            w64(bf2[f]),
            w64(cells[hm.b_face_cells[f] as usize]),
            "bf face {i}"
        );
    }
    Ok(())
}

// ==========================================================================
//  (120.6c) - outletMappedUniformInlet carries the source patch's mean
// ==========================================================================

#[test]
fn outlet_mapped_uniform_inlet_takes_the_outlet_mean() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let (hm, gm) = mesh_b(&gpu)?;
    let k = FieldKernels::new(&gpu)?;
    let xin = patch_of(&hm, "xMin");
    let atin = hm.patches[xin].start;
    let nin = hm.patches[xin].size;
    let xout = patch_of(&hm, "xMax");
    let atout = hm.patches[xout].start;
    let nout = hm.patches[xout].size;
    let n = hm.n_boundary_faces;

    let mut bounds = zg_all();
    bounds[0] = (
        "xMin",
        outlet_bc(
            "outletMappedUniformInlet",
            &[("outletPatch", "xMax"), ("fraction", "0.5"), ("offset", "0.25")],
        ),
    );
    let mut t = GpuScalarField::zeros(&gpu, &gm, "T")?;
    setup_scalar_field_with(
        &gpu,
        &mut t,
        &raw_t(bounds),
        &hm,
        BcInputs {
            patch_means: true,
            ..BcInputs::default()
        },
    )?;
    let tc = scalar_cells_t(&hm);
    gpu.write(&mut t.f, &tc)?;
    let mut phi = GpuSurfaceScalarField::zeros(&gpu, &gm, "phi")?;
    let mut phib = vec![0.0 as Scalar; n];
    for i in 0..nout {
        let f = atout + i;
        phib[f] = (0.1 * (1.0 + i as f64) * w64(hm.b_mag_sf[f])) as Scalar;
    }
    gpu.write(&mut phi.bf, &phib)?;
    correct_boundary_conditions(&gpu, &k, &mut t, &gm)?;
    update_patch_means_scalar(&gpu, &k, &mut t, &phi, None, &gm)?;
    correct_boundary_conditions(&gpu, &k, &mut t, &gm)?;
    gpu.sync()?;

    let bf = gpu.download(&t.bf)?;
    let fr = gpu.download(&t.fr)?;
    let cells = gpu.download(&t.f)?;
    let rv = gpu.download(&t.ref_value)?;
    let phi_b = gpu.download(&phi.bf)?;

    let mut max_t = 0.0f64;
    for v in &tc {
        max_t = nmax(max_t, w64(*v).abs());
    }
    for v in &bf {
        max_t = nmax(max_t, w64(*v).abs());
    }

    // The one uniform value, from the DOWNLOADED xMax face values: the
    // mixing-cup mean w = max(phi, 0), times fraction, plus offset.
    let (mut wn, mut wb) = (0.0f64, 0.0f64);
    for i in 0..nout {
        let f = atout + i;
        let w = w64(phi_b[f]).max(0.0);
        wn += fin("w_g", w);
        wb += fin("w_g bf_g", w * w64(bf[f]));
    }
    let want = 0.5 * (wb / wn) + 0.25;
    eprintln!("  [120.6] outletMapped: uniform value {want:.12}");
    for i in 0..nin {
        let d = (w64(bf[atin + i]) - want).abs();
        fin("bf", d);
        assert!(
            d <= tol(1e-12) * max_t,
            "inlet face {i}: {d:e} vs {want:e} (bound {})",
            tol(1e-12) * max_t
        );
        assert_eq!(w64(fr[atin + i]), 1.0, "fr face {i}");
    }

    // The host twin agrees, to round-off.
    assert_eq!(t.mean_rules.len(), 1, "one rule on xMin");
    let mut fr2 = vec![0.0 as Scalar; n];
    let mut rv2 = vec![0.0 as Scalar; n];
    let mut rg2 = vec![0.0 as Scalar; n];
    patch_mean_values_scalar(
        &t.mean_rules[0],
        &cells,
        &bf,
        &phi_b,
        &hm.b_mag_sf,
        &hm.b_face_cells,
        &mut fr2,
        &mut rv2,
        &mut rg2,
    );
    for i in 0..nin {
        let dd = (w64(rv[atin + i]) - w64(rv2[atin + i])).abs();
        fin("ref_value twin", dd);
        assert!(dd <= tol(1e-14) * max_t, "face {i}: twin gap {dd:e}");
    }

    // No outflow anywhere: the mixing-cup sum is empty, the rule falls
    // back to the source patch's area mean.
    gpu.write(&mut phi.bf, &vec![0.0 as Scalar; n])?;
    update_patch_means_scalar(&gpu, &k, &mut t, &phi, None, &gm)?;
    correct_boundary_conditions(&gpu, &k, &mut t, &gm)?;
    gpu.sync()?;
    let bf2 = gpu.download(&t.bf)?;
    let phi0 = gpu.download(&phi.bf)?;
    let (mut sa, mut sb) = (0.0f64, 0.0f64);
    for i in 0..nout {
        let f = atout + i;
        fin("phi zero", w64(phi0[f]));
        sa += w64(hm.b_mag_sf[f]);
        sb += w64(hm.b_mag_sf[f]) * w64(bf2[f]);
    }
    let want2 = 0.5 * (sb / sa) + 0.25;
    eprintln!(
        "  [120.6] outletMapped, no outflow: area-mean value {want2:.12}"
    );
    for i in 0..nin {
        let d = (w64(bf2[atin + i]) - want2).abs();
        fin("bf", d);
        assert!(d <= tol(1e-12) * max_t, "inlet face {i}: {d:e} vs {want2:e}");
    }

    // The vector twin: U as in test 1, xMin takes fraction 0.5 of the xMax
    // mixing-cup mean plus offset (0.5 0 0), componentwise.
    let mut vb = zg_all();
    vb[0] = (
        "xMin",
        outlet_bc(
            "outletMappedUniformInlet",
            &[
                ("outletPatch", "xMax"),
                ("fraction", "0.5"),
                ("offset", "(0.5 0 0)"),
            ],
        ),
    );
    let mut u = GpuVectorField::zeros(&gpu, &gm, "U")?;
    setup_vector_field_with(
        &gpu,
        &mut u,
        &raw_u(vb, Vec3::ZERO),
        &hm,
        BcInputs {
            patch_means: true,
            ..BcInputs::default()
        },
    )?;
    let uc = vector_cells_u(&hm);
    gpu.write(&mut u.f, &uc)?;
    gpu.write(&mut phi.bf, &phib)?;
    correct_boundary_conditions_vector(&gpu, &k, &mut u, &gm)?;
    update_patch_means_vector(&gpu, &k, &mut u, &phi, &gm)?;
    correct_boundary_conditions_vector(&gpu, &k, &mut u, &gm)?;
    gpu.sync()?;

    let ubf = gpu.download(&u.bf)?;
    let mut max_u = 0.0f64;
    for v in uc.iter().chain(ubf.iter()) {
        max_u = nmax(max_u, w64(v.x).abs());
        max_u = nmax(max_u, w64(v.y).abs());
        max_u = nmax(max_u, w64(v.z).abs());
    }
    let offs = [0.5, 0.0, 0.0];
    for c in 0..3 {
        let (mut wn, mut wb) = (0.0f64, 0.0f64);
        for i in 0..nout {
            let f = atout + i;
            let w = w64(phi_b[f]).max(0.0);
            let b = [w64(ubf[f].x), w64(ubf[f].y), w64(ubf[f].z)][c];
            wn += fin("w_g", w);
            wb += fin("w_g bf_g", w * b);
        }
        let want3 = 0.5 * (wb / wn) + offs[c];
        eprintln!(
            "  [120.6] outletMapped vector: component {c} value {want3:.12}"
        );
        for i in 0..nin {
            let b = ubf[atin + i];
            let v = [w64(b.x), w64(b.y), w64(b.z)][c];
            let d = (v - want3).abs();
            fin("bf", d);
            assert!(d <= tol(1e-12) * max_u, "component {c} face {i}: {d:e}");
        }
    }
    assert_eq!(u.mean_rules.len(), 1, "one vector rule");
    Ok(())
}

// ==========================================================================
//  (120.6d) - the loop duct closes on offset/(1 - fraction)
// ==========================================================================

#[test]
fn a_loop_duct_closes_on_the_offset() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    // 16 x 4 x 1 cells over [0,4] x [0,1] x [0,0.25]: xMin draws in
    // `fraction 0.5` of xMax's mean plus (0.5 0 0), xMax lets it out, y
    // symmetric, z empty. The fixed point (120.6d) is U = 0.5/(1 - 0.5) = 1
    // through a 0.25 area, so Q = 0.25.
    let hm = blockgen::build_mesh(&box_spec(
        [16, 4, 1],
        [0.0, 0.0, 0.0],
        [4.0, 1.0, 0.25],
        [
            "patch", "patch", "symmetry", "symmetry", "empty", "empty",
        ],
    ))?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let xin = patch_of(&hm, "xMin");
    let atin = hm.patches[xin].start;
    let nin = hm.patches[xin].size;
    let xout = patch_of(&hm, "xMax");
    let atout = hm.patches[xout].start;
    let nout = hm.patches[xout].size;

    let ctrl = SimpleControls {
        momentum: MomentumControls {
            nu: 0.1,
            u_solver: tight(),
            ..MomentumControls::default()
        },
        p_solver: tight(),
        ..SimpleControls::default()
    };
    let mut sim = Simple::new(
        &gpu,
        &hm,
        &gm,
        ctrl,
        BuoyancyCoeffs {
            g: Vec3::ZERO,
            ..BuoyancyCoeffs::default()
        },
    )?;
    setup_vector_field_with(
        &gpu,
        sim.u_mut(),
        &raw_u(
            vec![
                (
                    "xMin",
                    outlet_bc(
                        "outletMappedUniformInlet",
                        &[
                            ("outletPatch", "xMax"),
                            ("fraction", "0.5"),
                            ("offset", "(0.5 0 0)"),
                        ],
                    ),
                ),
                ("xMax", ps("zeroGradient")),
                ("yMin", ps("symmetry")),
                ("yMax", ps("symmetry")),
                ("zMin", ps("empty")),
                ("zMax", ps("empty")),
            ],
            Vec3::ZERO,
        ),
        &hm,
        BcInputs {
            patch_means: true,
            ..BcInputs::default()
        },
    )?;

    setup_scalar_field_with(
        &gpu,
        sim.p_mut(),
        &raw_p(vec![
            ("xMin", ps("fixedFluxExtrapolatedPressure")),
            ("xMax", outlet_bc("fixedMean", &[("meanValue", "uniform 0")])),
            ("yMin", ps("symmetry")),
            ("yMax", ps("symmetry")),
            ("zMin", ps("empty")),
            ("zMax", ps("empty")),
        ]),
        &hm,
        BcInputs {
            patch_means: true,
            ..BcInputs::default()
        },
    )?;
    sim.initialise(&gpu)?;
    let nut = laminar_nut(&gpu, &hm, &gm)?;
    let t = uniform_t(&gpu, &hm, &gm)?;
    let mut backend = PbicgstabBackend::new(ctrl.p_solver);
    backend.setup(&gpu, &hm, &gm, &SystemProbe::default())?;

    // SIMPLE iterations until the change of U between iterations drops
    // below 1e-14, at most 3000 of them. The f64 build has to break on the
    // tolerance BEFORE the cap - a silent 3000 would mean the gate checks
    // nothing - while under `single` the f32 SIMPLE floor parks the loop at
    // the cap (a known finding, SPEC-LIT 118), so there the cap stays and
    // whether it was reached is printed.
    let mut prev = gpu.download(&sim.u().f)?;
    let mut iterations = 0usize;
    let mut converged = false;
    for it in 0..3000 {
        sim.correct(&gpu, &mut backend, &nut, &t)?;
        iterations = it + 1;
        let now = gpu.download(&sim.u().f)?;
        let nonfinite = now.iter().enumerate().find(|(_, v)| {
            !w64(v.x).is_finite() || !w64(v.y).is_finite() || !w64(v.z).is_finite()
        });
        if let Some((c, v)) = nonfinite {
            return Err(ofgpu::Error::Diverged {
                iteration: it,
                what: format!("U cell {c} went non-finite: {v:?}"),
            });
        }
        let change = prev.iter().zip(&now).fold(0.0f64, |m, (a, b)| {
            m.max(nmax(nmax((w64(a.x) - w64(b.x)).abs(), (w64(a.y) - w64(b.y)).abs()),
                (w64(a.z) - w64(b.z)).abs()))
        });
        prev = now;
        if change < 1e-14 {
            converged = true;
            break;
        }
    }
    if cfg!(not(feature = "single")) {
        assert!(
            converged,
            "the loop duct hit the 3000-iteration cap instead of closing \
             below 1e-14 in {iterations} iterations"
        );
    } else if !converged {
        eprintln!(
            "  [120.6] loop duct: the 3000-iteration cap was reached (the \
             f32 SIMPLE floor)"
        );
    }
    eprintln!("  [120.6] loop duct: {iterations} iterations to its fixed point");

    // (120.6d): U = 0.5/(1 - 0.5) = 1 at the inlet, purely normal.
    let ubf = gpu.download(&sim.u().bf)?;
    for i in 0..nin {
        let b = ubf[atin + i];
        fin("U_b.x", w64(b.x));
        assert!(
            (w64(b.x) - 1.0).abs() <= tol(1e-10),
            "inlet face {i}: U_b.x = {}, want 1",
            w64(b.x)
        );
        fin("U_b.y", w64(b.y));
        fin("U_b.z", w64(b.z));
        assert!(
            w64(b.y).abs() <= tol(1e-10) && w64(b.z).abs() <= tol(1e-10),
            "inlet face {i}: tangential ({}, {})",
            w64(b.y),
            w64(b.z)
        );
    }

    // The rate: Q = -SUM_xMin phi, the outlet carries it back, and the
    // inlet value is fraction * (the mixing-cup mean of xMax) + offset.
    let phib = gpu.download(&sim.phi().bf)?;
    let mut q = 0.0f64;
    for i in 0..nin {
        q -= fin("phi_b", w64(phib[atin + i]));
    }
    eprintln!("  [120.6] loop duct: Q = {q:.9} (want 0.25)");
    assert!(
        (q - 0.25).abs() <= tol(1e-10) * 0.25,
        "Q = {q}, want 0.25"
    );
    let mut qout = 0.0f64;
    for i in 0..nout {
        qout += fin("phi_b", w64(phib[atout + i]));
    }
    assert!(
        (qout - q).abs() <= tol(1e-12),
        "outlet sum {qout} vs Q {q}"
    );
    let (mut wn, mut wb) = (0.0f64, 0.0f64);
    for i in 0..nout {
        let f = atout + i;
        let w = w64(phib[f]).max(0.0);
        wn += fin("w_g", w);
        wb += fin("w_g U_g.x", w * w64(ubf[f].x));
    }
    let want = 0.5 * (wb / wn) + 0.5;
    for i in 0..nin {
        let d = (w64(ubf[atin + i].x) - want).abs();
        fin("U_in - f U_out - c", d);
        assert!(d <= tol(1e-12), "inlet face {i}: {d:e} vs {want:e}");
    }

    // The pressure level is held by the fixedMean outlet: its area mean is
    // the target 0.
    let pbf = gpu.download(&sim.p().bf)?;
    let (mut sa, mut sb) = (0.0f64, 0.0f64);
    for i in 0..nout {
        let f = atout + i;
        sa += fin("|S_f|", w64(hm.b_mag_sf[f]));
        sb += fin("|S_f| p_b", w64(hm.b_mag_sf[f]) * w64(pbf[f]));
    }
    let pmean = sb / sa;
    eprintln!("  [120.6] loop duct: xMax p area mean {pmean:.12} (want 0)");
    assert!(
        pmean.abs() <= tol(1e-12),
        "p area mean {pmean:e}, want 0"
    );
    Ok(())
}

// ==========================================================================
//  (120.6e) - matchedFlowRateOutletVelocity carries the inlet's flux
// ==========================================================================

#[test]
fn matched_flow_rate_outlet_carries_the_inlet_flux() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let (hm, gm) = mesh_b(&gpu)?;
    let k = FieldKernels::new(&gpu)?;
    let xin = patch_of(&hm, "xMin");
    let atin = hm.patches[xin].start;
    let nin = hm.patches[xin].size;
    let xout = patch_of(&hm, "xMax");
    let atout = hm.patches[xout].start;
    let np = hm.patches[xout].size;
    let n = hm.n_boundary_faces;

    let mut vb = zg_all();
    vb[1] = (
        "xMax",
        outlet_bc("matchedFlowRateOutletVelocity", &[("inletPatch", "xMin")]),
    );
    let mut u = GpuVectorField::zeros(&gpu, &gm, "U")?;
    setup_vector_field_with(
        &gpu,
        &mut u,
        &raw_u(vb, Vec3::ZERO),
        &hm,
        BcInputs {
            patch_means: true,
            ..BcInputs::default()
        },
    )?;
    let uc = vector_cells_u(&hm);
    gpu.write(&mut u.f, &uc)?;
    let mut phi = GpuSurfaceScalarField::zeros(&gpu, &gm, "phi")?;
    // phi_g = -(1 + 0.5 i)|S_g| on xMin: Q = (1/6) SUM (1 + 0.5 i) over six
    // faces = 2.25 exactly.
    let mut phib = vec![0.0 as Scalar; n];
    for i in 0..nin {
        let f = atin + i;
        phib[f] = (-(1.0 + 0.5 * i as f64) * w64(hm.b_mag_sf[f])) as Scalar;
    }
    gpu.write(&mut phi.bf, &phib)?;
    correct_boundary_conditions_vector(&gpu, &k, &mut u, &gm)?;
    update_patch_means_vector(&gpu, &k, &mut u, &phi, &gm)?;
    correct_boundary_conditions_vector(&gpu, &k, &mut u, &gm)?;
    gpu.sync()?;

    let ubf = gpu.download(&u.bf)?;
    let cells = gpu.download(&u.f)?;
    let fr = gpu.download(&u.fr)?;
    let rv = gpu.download(&u.ref_value)?;
    let phi_b = gpu.download(&phi.bf)?;
    let mut max_u = 0.0f64;
    for v in uc.iter().chain(ubf.iter()) {
        max_u = nmax(max_u, w64(v.x).abs());
        max_u = nmax(max_u, w64(v.y).abs());
        max_u = nmax(max_u, w64(v.z).abs());
    }

    // (a): SUM_xMax U_b . S_f = Q = 2.25, to round-off.
    let mut flux = 0.0f64;
    for i in 0..np {
        let f = atout + i;
        let s = hm.b_sf[f];
        flux += fin("U_b . S_f",
            w64(ubf[f].x) * w64(s.x)
                + w64(ubf[f].y) * w64(s.y)
                + w64(ubf[f].z) * w64(s.z));
    }
    eprintln!("  [120.6] matchedFlowRate: outlet flux {flux:.12} (want 2.25)");
    assert!(
        (flux - 2.25).abs() <= tol(1e-12) * 2.25,
        "outlet flux {flux:e}, want 2.25"
    );

    // (b): fr exactly 1, and the shift is purely normal - the tangential
    // part of U_b - U_P vanishes.
    for i in 0..np {
        let f = atout + i;
        assert_eq!(w64(fr[f]), 1.0, "fr face {i}");
        let s = hm.b_sf[f];
        let m = (w64(s.x) * w64(s.x) + w64(s.y) * w64(s.y) + w64(s.z) * w64(s.z)).sqrt();
        let nx = w64(s.x) / m;
        let ny = w64(s.y) / m;
        let nz = w64(s.z) / m;
        let c = hm.b_face_cells[f] as usize;
        let dx = w64(ubf[f].x) - w64(cells[c].x);
        let dy = w64(ubf[f].y) - w64(cells[c].y);
        let dz = w64(ubf[f].z) - w64(cells[c].z);
        let dn = dx * nx + dy * ny + dz * nz;
        let tx = (dx - dn * nx).abs();
        let ty = (dy - dn * ny).abs();
        let tz = (dz - dn * nz).abs();
        let t = nmax(nmax(tx, ty), tz);
        fin("tangential shift", t);
        assert!(t <= tol(1e-14) * max_u, "face {i}: tangential {t:e}");
    }

    // (c): the host twin agrees, to round-off.
    assert_eq!(u.mean_rules.len(), 1, "one rule on xMax");
    let mut fr2 = vec![0.0 as Scalar; n];
    let mut rv2 = vec![Vec3::ZERO; n];
    let mut rg2 = vec![Vec3::ZERO; n];
    patch_mean_values_vector(
        &u.mean_rules[0],
        &cells,
        &ubf,
        &phi_b,
        &hm.b_sf,
        &hm.b_mag_sf,
        &hm.b_face_cells,
        &mut fr2,
        &mut rv2,
        &mut rg2,
    );
    for i in 0..np {
        let f = atout + i;
        let dd = nmax(
            nmax(
                (w64(rv[f].x) - w64(rv2[f].x)).abs(),
                (w64(rv[f].y) - w64(rv2[f].y)).abs(),
            ),
            (w64(rv[f].z) - w64(rv2[f].z)).abs(),
        );
        fin("ref_value twin", dd);
        assert!(dd <= tol(1e-14) * max_u, "face {i}: twin gap {dd:e}");
    }

    // At rest, the outlet is the pure normal profile U_b = n Q/A with
    // A = 1 - every face 2.25 along x, and the patch sum still Q.
    gpu.write(&mut u.f, &vec![Vec3::ZERO; hm.n_cells])?;
    correct_boundary_conditions_vector(&gpu, &k, &mut u, &gm)?;
    update_patch_means_vector(&gpu, &k, &mut u, &phi, &gm)?;
    correct_boundary_conditions_vector(&gpu, &k, &mut u, &gm)?;
    gpu.sync()?;
    let bf2 = gpu.download(&u.bf)?;
    for i in 0..np {
        let b = bf2[atout + i];
        fin("U_b.x", w64(b.x));
        assert!(
            (w64(b.x) - 2.25).abs() <= tol(1e-12) * 2.25,
            "face {i}: U_b.x = {}, want 2.25",
            w64(b.x)
        );
        assert!(
            w64(b.y).abs() <= tol(1e-12) * 2.25
                && w64(b.z).abs() <= tol(1e-12) * 2.25,
            "face {i}: tangential ({}, {})",
            w64(b.y),
            w64(b.z)
        );
    }
    let mut flux2 = 0.0f64;
    for i in 0..np {
        let f = atout + i;
        let s = hm.b_sf[f];
        flux2 += w64(bf2[f].x) * w64(s.x)
            + w64(bf2[f].y) * w64(s.y)
            + w64(bf2[f].z) * w64(s.z);
    }
    eprintln!(
        "  [120.6] matchedFlowRate, U = 0: outlet flux {flux2:.12} (want 2.25)"
    );
    assert!(
        (flux2 - 2.25).abs() <= tol(1e-12) * 2.25,
        "outlet flux {flux2:e}, want 2.25"
    );
    Ok(())
}

// ==========================================================================
//  (120.6f) - fixedFluxExtrapolatedPressure, exact on a linear field
// ==========================================================================

#[test]
fn fixed_flux_extrapolated_pressure_is_exact_on_a_linear_field() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let (hm, gm) = mesh_b(&gpu)?;
    let k = FieldKernels::new(&gpu)?;
    let n = hm.n_boundary_faces;

    let mut bounds = zg_all();
    for b in bounds.iter_mut() {
        b.1 = ps("fixedFluxExtrapolatedPressure");
    }
    let mut p = GpuScalarField::zeros(&gpu, &gm, "p")?;
    setup_scalar_field_with(
        &gpu,
        &mut p,
        &raw_t(bounds),
        &hm,
        BcInputs {
            patch_means: true,
            ..BcInputs::default()
        },
    )?;
    let pc = scalar_cells_p(&hm);
    gpu.write(&mut p.f, &pc)?;
    // The cell gradient, uploaded as a plain DevBuf<Vec3> - what the
    // momentum predictor would hold.
    let grad: Vec<Vec3> = vec![Vec3::new(2.0, -1.0, 0.5); hm.n_cells];
    let gbuf = gpu.upload(&grad)?;
    let phi = GpuSurfaceScalarField::zeros(&gpu, &gm, "phi")?;
    correct_boundary_conditions(&gpu, &k, &mut p, &gm)?;
    update_patch_means_scalar(&gpu, &k, &mut p, &phi, Some(&gbuf), &gm)?;
    correct_boundary_conditions(&gpu, &k, &mut p, &gm)?;
    gpu.sync()?;

    let bf = gpu.download(&p.bf)?;
    let fr = gpu.download(&p.fr)?;
    let rg = gpu.download(&p.ref_grad)?;

    let mut max_p = 0.0f64;
    for v in &pc {
        max_p = nmax(max_p, w64(*v).abs());
    }
    for v in &bf {
        max_p = nmax(max_p, w64(*v).abs());
    }

    // Every boundary face is the linear field at its face centre, to
    // round-off; fr exactly 0.
    let bcf = hm.geom64().b_cf;
    let mut worst = 0.0f64;
    for f in 0..n {
        let want = 3.0 + 2.0 * bcf[f].x - bcf[f].y + 0.5 * bcf[f].z;
        let d = (w64(bf[f]) - want).abs();
        fin("p_b", d);
        worst = worst.max(d);
        assert!(
            d <= tol(1e-12) * max_p,
            "face {f}: {d:e} vs {want:e} (bound {})",
            tol(1e-12) * max_p
        );
        assert_eq!(w64(fr[f]), 0.0, "fr face {f}");
    }
    eprintln!(
        "  [120.6] extrapolated pressure: worst face gap {worst:.3e} (bound {})",
        tol(1e-12) * max_p
    );

    // ref_grad equals the host twin on every face, and the update without
    // a gradient is refused by name.
    assert_eq!(p.mean_rules.len(), 6, "one rule per patch");
    let mut fr2 = vec![0.0 as Scalar; n];
    let mut rv2 = vec![0.0 as Scalar; n];
    let mut rg2 = vec![0.0 as Scalar; n];
    for rule in &p.mean_rules {
        extrapolated_sn_grad(
            rule,
            &grad,
            &hm.b_sf,
            &hm.b_mag_sf,
            &hm.b_face_cells,
            &mut fr2,
            &mut rv2,
            &mut rg2,
        );
    }
    for f in 0..n {
        let d = (w64(rg[f]) - w64(rg2[f])).abs();
        fin("ref_grad twin", d);
        assert!(d <= tol(1e-14) * 3.0, "face {f}: twin gap {d:e}");
    }

    let err = update_patch_means_scalar(&gpu, &k, &mut p, &phi, None, &gm)
        .err()
        .unwrap_or_else(|| panic!("the kind-48 rule built without a gradient"));
    let msg = err.to_string();
    assert!(
        msg.contains("120.6"),
        "missing-gradient refusal: {msg}"
    );
    Ok(())
}

// ==========================================================================
//  (120.6f) - a patch past the first launch block
// ==========================================================================

#[test]
fn extrapolated_gradient_reaches_a_patch_past_the_first_block() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    // 16 x 16 x 2 cells: the flattened boundary is xMin, xMax, yMin, yMax
    // (32 faces each), zMin (256) and zMax (256), so zMax starts at face
    // 384 - past the first launch block of 256 threads. The kernel has to
    // reach it, or (120.6f) silently holds only the first block's patches.
    let hm = blockgen::build_mesh(&box_spec(
        [16, 16, 2],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.25],
        ["patch", "patch", "patch", "patch", "patch", "patch"],
    ))?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let k = FieldKernels::new(&gpu)?;
    let zmax = patch_of(&hm, "zMax");
    let at = hm.patches[zmax].start;
    let np = hm.patches[zmax].size;
    let n = hm.n_boundary_faces;

    let mut bounds = zg_all();
    bounds[5] = ("zMax", ps("fixedFluxExtrapolatedPressure"));
    let mut p = GpuScalarField::zeros(&gpu, &gm, "p")?;
    setup_scalar_field_with(
        &gpu,
        &mut p,
        &raw_t(bounds),
        &hm,
        BcInputs {
            patch_means: true,
            ..BcInputs::default()
        },
    )?;
    let pc = scalar_cells_p(&hm);
    gpu.write(&mut p.f, &pc)?;
    let grad: Vec<Vec3> = vec![Vec3::new(2.0, -1.0, 0.5); hm.n_cells];
    let gbuf = gpu.upload(&grad)?;
    let phi = GpuSurfaceScalarField::zeros(&gpu, &gm, "phi")?;
    correct_boundary_conditions(&gpu, &k, &mut p, &gm)?;
    update_patch_means_scalar(&gpu, &k, &mut p, &phi, Some(&gbuf), &gm)?;
    correct_boundary_conditions(&gpu, &k, &mut p, &gm)?;
    gpu.sync()?;

    let bf = gpu.download(&p.bf)?;
    let fr = gpu.download(&p.fr)?;
    let rg = gpu.download(&p.ref_grad)?;

    let mut max_p = 0.0f64;
    for v in &pc {
        max_p = nmax(max_p, w64(*v).abs());
    }
    for v in &bf {
        max_p = nmax(max_p, w64(*v).abs());
    }

    // Every zMax face carries the linear field at its face centre - which
    // needs the refGrad the kernel never wrote while the patch started
    // past the first launch block: bf stayed p_P, a gap of 0.5 * 0.0625.
    let bcf = hm.geom64().b_cf;
    for i in 0..np {
        let f = at + i;
        let want = 3.0 + 2.0 * bcf[f].x - bcf[f].y + 0.5 * bcf[f].z;
        let d = (w64(bf[f]) - want).abs();
        fin("p_b", d);
        assert!(
            d <= tol(1e-12) * max_p,
            "zMax face {i} (flat {f}): {d:e} vs {want:e} (bound {})",
            tol(1e-12) * max_p
        );
        assert_eq!(w64(fr[f]), 0.0, "fr face {i}");
    }

    // ref_grad equals the host twin on the patch's faces.
    assert_eq!(p.mean_rules.len(), 1, "one rule on zMax");
    let mut fr2 = vec![0.0 as Scalar; n];
    let mut rv2 = vec![0.0 as Scalar; n];
    let mut rg2 = vec![0.0 as Scalar; n];
    extrapolated_sn_grad(
        &p.mean_rules[0],
        &grad,
        &hm.b_sf,
        &hm.b_mag_sf,
        &hm.b_face_cells,
        &mut fr2,
        &mut rv2,
        &mut rg2,
    );
    for i in 0..np {
        let f = at + i;
        let d = (w64(rg[f]) - w64(rg2[f])).abs();
        fin("ref_grad twin", d);
        assert!(d <= tol(1e-14) * 3.0, "zMax face {i}: twin gap {d:e}");
    }
    eprintln!(
        "  [120.6] extrapolated pressure, zMax at face {at}: {np} faces reached"
    );
    Ok(())
}

// ==========================================================================
//  The refusals, by name
// ==========================================================================

#[test]
fn outlets_ii_refusals_are_by_name() {
    let hm = blockgen::build_mesh(&box_spec(
        [4, 3, 2],
        [0.0, 0.0, 0.0],
        [2.0, 1.0, 1.0],
        ["patch", "patch", "patch", "patch", "patch", "patch"],
    ))
    .unwrap();
    let internal_s = vec![0.0 as Scalar; hm.n_cells];
    let internal_v = vec![Vec3::ZERO; hm.n_cells];
    let on = BcInputs {
        patch_means: true,
        ..BcInputs::default()
    };
    let off = BcInputs::default();

    // The five names on a scalar field without the opt-in.
    let names = [
        "fixedMean",
        "fixedMeanOutletInlet",
        "fixedFluxExtrapolatedPressure",
        "outletMappedUniformInlet",
        "matchedFlowRateOutletVelocity",
    ];
    for name in names {
        let mut bounds = zg_all();
        bounds[1] = ("xMax", outlet_bc(name, &[]));
        let err = scalar_boundary_with(&raw_t(bounds), &hm, &internal_s, off)
            .err()
            .unwrap_or_else(|| panic!("{name} built without the opt-in"));
        let msg = err.to_string();
        assert!(msg.contains(name), "{name}: {msg}");
        assert!(msg.contains("BcInputs::patch_means"), "{name}: {msg}");
        assert!(msg.contains("120.6"), "{name}: {msg}");
    }

    // The rank refusals, with the opt-in on: the flow-rate outlet is a
    // vector condition, the extrapolated pressure a scalar one.
    let mut bounds = zg_all();
    bounds[1] = (
        "xMax",
        outlet_bc("matchedFlowRateOutletVelocity", &[("inletPatch", "xMin")]),
    );
    let err = scalar_boundary_with(&raw_t(bounds), &hm, &internal_s, on)
        .err()
        .unwrap_or_else(|| panic!("the flow-rate outlet built on a scalar"));
    let msg = err.to_string();
    assert!(msg.contains("vector condition"), "{msg}");
    assert!(msg.contains("120.6"), "{msg}");

    let mut vb = zg_all();
    vb[1] = ("xMax", ps("fixedFluxExtrapolatedPressure"));
    let err = vector_boundary_with(&raw_u(vb, Vec3::ZERO), &hm, &internal_v, on)
        .err()
        .unwrap_or_else(|| panic!("the extrapolated pressure built on a vector"));
    let msg = err.to_string();
    assert!(msg.contains("scalar condition"), "{msg}");
    assert!(msg.contains("120.6"), "{msg}");

    // An `outletPatch` that names no patch of the mesh, and one that names
    // the patch itself.
    let mut bounds = zg_all();
    bounds[0] = (
        "xMin",
        outlet_bc(
            "outletMappedUniformInlet",
            &[("outletPatch", "nowhere"), ("fraction", "0.5")],
        ),
    );
    let err = scalar_boundary_with(&raw_t(bounds), &hm, &internal_s, on)
        .err()
        .unwrap_or_else(|| panic!("outletPatch nowhere built"));
    let msg = err.to_string();
    assert!(msg.contains("nowhere"), "{msg}");
    assert!(msg.contains("120.6"), "{msg}");

    let mut bounds = zg_all();
    bounds[0] = (
        "xMin",
        outlet_bc(
            "outletMappedUniformInlet",
            &[("outletPatch", "xMin")],
        ),
    );
    let err = scalar_boundary_with(&raw_t(bounds), &hm, &internal_s, on)
        .err()
        .unwrap_or_else(|| panic!("a self-naming outletPatch built"));
    let msg = err.to_string();
    assert!(msg.contains("xMin"), "{msg}");
    assert!(msg.contains("120.6"), "{msg}");

    // The required entries, by key: outletPatch, inletPatch, meanValue.
    let mut bounds = zg_all();
    bounds[0] = (
        "xMin",
        outlet_bc("outletMappedUniformInlet", &[("fraction", "0.5")]),
    );
    let err = scalar_boundary_with(&raw_t(bounds), &hm, &internal_s, on)
        .err()
        .unwrap_or_else(|| panic!("outletMappedUniformInlet built without outletPatch"));
    let msg = err.to_string();
    assert!(msg.contains("outletPatch"), "{msg}");
    assert!(msg.contains("120.6"), "{msg}");

    let mut vb = zg_all();
    vb[1] = ("xMax", outlet_bc("matchedFlowRateOutletVelocity", &[]));
    let err = vector_boundary_with(&raw_u(vb, Vec3::ZERO), &hm, &internal_v, on)
        .err()
        .unwrap_or_else(|| panic!("matchedFlowRateOutletVelocity built without inletPatch"));
    let msg = err.to_string();
    assert!(msg.contains("inletPatch"), "{msg}");
    assert!(msg.contains("120.6"), "{msg}");

    let mut bounds = zg_all();
    bounds[1] = ("xMax", outlet_bc("fixedMean", &[]));
    let err = scalar_boundary_with(&raw_t(bounds), &hm, &internal_s, on)
        .err()
        .unwrap_or_else(|| panic!("fixedMean built without meanValue"));
    let msg = err.to_string();
    assert!(msg.contains("meanValue"), "{msg}");
    assert!(msg.contains("120.6"), "{msg}");

    // A time-varying meanValue is refused by name - no driver runs the
    // timed path for these names.
    let mut bounds = zg_all();
    bounds[1] = (
        "xMax",
        outlet_bc("fixedMean", &[("meanValue", "table ((0 1) (1 2))")]),
    );
    let err = scalar_boundary_with(&raw_t(bounds), &hm, &internal_s, on)
        .err()
        .unwrap_or_else(|| panic!("a time-varying meanValue built"));
    let msg = err.to_string();
    assert!(msg.contains("time-varying"), "{msg}");
    assert!(msg.contains("meanValue"), "{msg}");
    assert!(msg.contains("120.6"), "{msg}");
}
