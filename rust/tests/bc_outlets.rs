// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 120.4 - the two outlets through the public API, in BOTH builds.
//! `outletInlet` is (120.4a)'s reverse flux switch - fixed on outflow, bit
//! for bit, device and host twin agreeing. `advective` is (120.4b)-(120.4d),
//! Orlanski (1976), doi:10.1016/0021-9991(76)90023-1, with the phase speed
//! GIVEN: its face relaxes by the closed form (120.4e) under Euler, reaches
//! the same fixed point under `backward` along a host f64 march of
//! [`ofgpu::field::advective::advective_triple`], and - the gate - lets a
//! Gaussian pulse leave through a 200-cell channel with under 1 % reflected
//! peak where `zeroGradient` reflects more than 1 %. Precision classes per
//! docs/17 §5.1: **R** the device against the closed form and the host
//! march (1e-12 in the f64 build, `max(t, min(t*2^29, 1e-5))` under
//! `single`, the `tol` const fn below), **P** the pulse ratios, the same
//! band in both builds. A machine without a card passes vacuously, the
//! convention of `tests/f32_delta_form.rs`.
//!
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::field::advective::{advective_triple, AdvectiveScalar};
use ofgpu::field_ops::{
    correct_boundary_conditions, correct_boundary_conditions_vector, update_inlet_outlet_scalar,
    update_inlet_outlet_vector, FieldKernels,
};
use ofgpu::fv::{self, DivScheme, FvKernels};
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField};
use ofgpu::ldu_ops::{add_boundary_contributions, LduKernels};
use ofgpu::solver::{solve, SolverControls, SolverKernels, SolverWorkspace};
use ofgpu::timescheme::{Ddt, DdtCoeffs, DdtScheme, LtsControls};
use ofgpu::{
    field_setup, BcKind, Gpu, GpuLduMatrix, GpuMesh, GpuScalarField, GpuSurfaceScalarField,
    GpuVectorField, HostMesh, Label, Result, Scalar, Vec3,
};

/// `t` in the f64 build; under `single` (class R, D-F32-1)
/// `max(t, min(t*2^29, 1e-5))` - i.e. 1e-12 -> 1e-5, 1e-13 -> 1e-5.
const fn tol(t: f64) -> f64 {
    if cfg!(feature = "single") {
        let scaled = t * 536870912.0;
        let capped = if scaled < 1e-5 { scaled } else { 1e-5 };
        if t > capped { t } else { capped }
    } else {
        t
    }
}

/// An f64 value stored as the build's `Scalar`: a no-op in the f64 build,
/// the one rounding under `single`.
#[allow(clippy::unnecessary_cast, clippy::useless_conversion)]
fn s(x: f64) -> Scalar {
    x as Scalar
}

/// The build's `Scalar` widened to the f64 the answer keys are computed in.
#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    x as f64
}

/// A checked term for the folds below: a non-finite term panics here,
/// naming itself, instead of vanishing through a comparison - the gate's
/// "every measured value checked finite first".
fn fin(what: &str, x: f64) -> f64 {
    assert!(x.is_finite(), "{what} is non-finite: {x:e}");
    x
}

/// The index of a patch by name.
fn patch_of(hm: &HostMesh, name: &str) -> usize {
    hm.patches
        .iter()
        .position(|p| p.name == name)
        .unwrap_or_else(|| panic!("no patch {name}"))
}

/// A uniform block, every side a plain `patch` unless the caller says
/// otherwise - the `box_spec` shape of `tests/bc_wall_slip.rs`.
fn box_spec(n: [usize; 3], lo: [f64; 3], hi: [f64; 3], types: [&str; 6]) -> BlockSpec {
    let axis = |lo: f64, hi: f64, n: usize| GradedAxis {
        lo: s(lo),
        hi: s(hi),
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

/// Every device test needs a card; without one the test passes vacuously.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

// ==========================================================================
//  outletInlet - (120.4a), the reverse switch
// ==========================================================================

/// 1 x 4 x 1: the zMax patch carries four faces, one over each cell.
fn outlet_mesh() -> Result<HostMesh> {
    blockgen::build_mesh(&box_spec(
        [1, 4, 1],
        [0.0, 0.0, 0.0],
        [1.0, 4.0, 1.0],
        ["patch"; 6],
    ))
}

/// The scalar field of the (120.4a) gate: cells 5, 6, 7, 8 behind the four
/// zMax faces, `outletInlet` at `outletValue` 3 there, zeroGradient
/// elsewhere. Returns the field and the zMax faces' start index.
fn outlet_field(gpu: &Gpu, hm: &HostMesh, gm: &GpuMesh) -> Result<(GpuScalarField, usize)> {
    let top = hm.patches[patch_of(hm, "zMax")].start;
    let n = hm.n_boundary_faces;

    let mut psi = GpuScalarField::zeros(gpu, gm, "phi")?;
    gpu.write(&mut psi.f, &[s(5.0), s(6.0), s(7.0), s(8.0)])?;
    let mut kind = vec![BcKind::ZeroGradient as Label; n];
    let fr = vec![0.0 as Scalar; n];
    let mut rv = vec![0.0 as Scalar; n];
    for i in 0..4 {
        kind[top + i] = BcKind::OutletInlet as Label;
        rv[top + i] = s(3.0);
    }
    gpu.write(&mut psi.bc_kind, &kind)?;
    gpu.write(&mut psi.fr, &fr)?;
    gpu.write(&mut psi.ref_value, &rv)?;
    Ok((psi, top))
}

/// The flux of the gate: `+1, -1, 0, +2` on the four zMax faces.
fn outlet_flux(gpu: &Gpu, hm: &HostMesh, gm: &GpuMesh, negated: bool) -> Result<GpuSurfaceScalarField> {
    let top = hm.patches[patch_of(hm, "zMax")].start;
    let mut phi = GpuSurfaceScalarField::zeros(gpu, gm, "phi")?;
    let mut phib = vec![0.0 as Scalar; hm.n_boundary_faces];
    let q = |i: usize| {
        let v = [1.0, -1.0, 0.0, 2.0][i];
        s(if negated { -v } else { v })
    };
    for i in 0..4 {
        phib[top + i] = q(i);
    }
    gpu.write(&mut phi.bf, &phib)?;
    Ok(phi)
}

/// (120.4a), the gate: faces with `phi_b = +1, -1, 0, +2` and
/// `outletValue 3` evaluate to `3, phi_P, phi_P, 3` - bit for bit - the
/// switch swaps with the flux, the host twin's fraction equals the
/// device's, and the same holds for a vector field.
#[test]
fn outlet_inlet_switches_on_the_flux_sign() {
    let Some(gpu) = gpu() else { return };
    let hm = outlet_mesh().unwrap();
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let k = FieldKernels::new(&gpu).unwrap();
    let top = hm.patches[patch_of(&hm, "zMax")].start;
    let n = hm.n_boundary_faces;

    // Outflow, back-flow, exactly zero flux, outflow again.
    let mut phi = outlet_flux(&gpu, &hm, &gm, false).unwrap();
    let (mut psi, _) = outlet_field(&gpu, &hm, &gm).unwrap();
    update_inlet_outlet_scalar(&gpu, &k, &mut psi, &phi).unwrap();
    correct_boundary_conditions(&gpu, &k, &mut psi, &gm).unwrap();
    gpu.sync().unwrap();
    let bf = gpu.download(&psi.bf).unwrap();
    for (i, want) in [3.0, 6.0, 7.0, 3.0].iter().enumerate() {
        fin("phi_b", w64(bf[top + i]));
        assert_eq!(w64(bf[top + i]), *want, "face {i}: outflow case");
    }

    // And the host twin's fraction is the device's, bit for bit.
    let (mut psi_host, _) = outlet_field(&gpu, &hm, &gm).unwrap();
    field_setup::update_inlet_outlet(&gpu, &mut psi_host, &phi, &hm).unwrap();
    let fr_dev = gpu.download(&psi.fr).unwrap();
    let fr_host = gpu.download(&psi_host.fr).unwrap();
    for f in 0..n {
        assert_eq!(fr_dev[f].to_bits(), fr_host[f].to_bits(), "fr on face {f}");
    }

    // The flux reverses; the switch swaps with it.
    phi = outlet_flux(&gpu, &hm, &gm, true).unwrap();
    update_inlet_outlet_scalar(&gpu, &k, &mut psi, &phi).unwrap();
    correct_boundary_conditions(&gpu, &k, &mut psi, &gm).unwrap();
    gpu.sync().unwrap();
    let bf = gpu.download(&psi.bf).unwrap();
    for (i, want) in [5.0, 3.0, 7.0, 8.0].iter().enumerate() {
        fin("phi_b", w64(bf[top + i]));
        assert_eq!(w64(bf[top + i]), *want, "face {i}: reversed case");
    }

    // The vector field shares the fraction across its components.
    let mut u = GpuVectorField::zeros(&gpu, &gm, "U").unwrap();
    gpu.write(
        &mut u.f,
        &[
            Vec3::new(s(5.0), s(5.0), s(5.0)),
            Vec3::new(s(6.0), s(6.0), s(6.0)),
            Vec3::new(s(7.0), s(7.0), s(7.0)),
            Vec3::new(s(8.0), s(8.0), s(8.0)),
        ],
    )
    .unwrap();
    let mut kind = vec![BcKind::ZeroGradient as Label; n];
    let mut rv = vec![Vec3::new(s(3.0), s(3.0), s(3.0)); n];
    for i in 0..4 {
        kind[top + i] = BcKind::OutletInlet as Label;
        rv[top + i] = Vec3::new(s(3.0), s(3.0), s(3.0));
    }
    gpu.write(&mut u.bc_kind, &kind).unwrap();
    gpu.write(&mut u.ref_value, &rv).unwrap();

    phi = outlet_flux(&gpu, &hm, &gm, false).unwrap();
    update_inlet_outlet_vector(&gpu, &k, &mut u, &phi).unwrap();
    correct_boundary_conditions_vector(&gpu, &k, &mut u, &gm).unwrap();
    gpu.sync().unwrap();
    let bf = gpu.download(&u.bf).unwrap();
    for (i, want) in [3.0, 6.0, 7.0, 3.0].iter().enumerate() {
        let v = bf[top + i];
        fin("U_b", w64(v.x));
        fin("U_b", w64(v.y));
        fin("U_b", w64(v.z));
        assert_eq!(w64(v.x), *want, "vector face {i} x");
        assert_eq!(w64(v.y), *want, "vector face {i} y");
        assert_eq!(w64(v.z), *want, "vector face {i} z");
    }
}

// ==========================================================================
//  advective - (120.4b)-(120.4e)
// ==========================================================================

/// The one-face rig of the relaxation gates. One cell of size 8, so
/// `Delta_b = 4` on the +x face; `w = 2` and `K = 0.5` pinned by writing
/// the buffers the wrappers read (`K = w*inv_l_inf`, `inv_l_inf = 0.25`);
/// `phi_inf = 1`, `phi_P = 3` frozen, `phi_b^0 = 10`.
struct RelaxFace {
    gm: GpuMesh,
    k: FieldKernels,
    f: GpuScalarField,
    adv: AdvectiveScalar,
    face: usize,
    n_bf: usize,
}

fn relax_face(gpu: &Gpu) -> Result<RelaxFace> {
    let hm = blockgen::build_mesh(&box_spec(
        [1, 1, 1],
        [0.0, 0.0, 0.0],
        [0.5, 1.0, 1.0],
        ["patch"; 6],
    ))?;
    let gm = GpuMesh::upload(gpu, &hm)?;
    let face = hm.patches[patch_of(&hm, "xMax")].start;
    let n_bf = hm.n_boundary_faces;
    // The gate's Delta_b: one cell of size 8 gives exactly 4.
    assert_eq!(hm.geom64().b_delta_coeffs[face], 4.0, "Delta_b");

    let mut f = GpuScalarField::zeros(gpu, &gm, "phi")?;
    gpu.write(&mut f.f, &[s(3.0)])?;
    let mut kind = vec![BcKind::ZeroGradient as Label; n_bf];
    kind[face] = BcKind::Advective as Label;
    let mut bf = vec![0.0 as Scalar; n_bf];
    bf[face] = s(10.0);
    gpu.write(&mut f.bc_kind, &kind)?;
    gpu.write(&mut f.fr, &vec![0.0 as Scalar; n_bf])?;
    gpu.write(&mut f.ref_value, &vec![0.0 as Scalar; n_bf])?;
    gpu.write(&mut f.ref_grad, &vec![0.0 as Scalar; n_bf])?;
    gpu.write(&mut f.bf, &bf)?;

    let mut inv = vec![0.0 as Scalar; n_bf];
    inv[face] = s(0.25);
    let finf = vec![s(1.0); n_bf];
    let mut adv = AdvectiveScalar::from_host(gpu, &inv, &finf, &f)?;
    // phi_b^o = phi_b^oo = phi_b^0 = 10, what `from_host` seeded; the speed
    // and the rate are pinned, not read off a flux.
    let mut w = vec![0.0 as Scalar; n_bf];
    w[face] = s(2.0);
    let mut kk = vec![0.0 as Scalar; n_bf];
    kk[face] = s(0.5);
    gpu.write(&mut adv.faces.w, &w)?;
    gpu.write(&mut adv.faces.k_relax, &kk)?;

    Ok(RelaxFace {
        gm,
        k: FieldKernels::new(gpu)?,
        f,
        adv,
        face,
        n_bf,
    })
}

/// March the face: per step rotate, rewrite the triple from (120.4d) with
/// `Ddt::face_coeffs`'s coefficients, correct, and record the face value.
fn march_relax(
    gpu: &Gpu,
    r: &mut RelaxFace,
    scheme: DdtScheme,
    steps: usize,
) -> Result<Vec<f64>> {
    let mut ddt = Ddt::new(gpu, &r.gm, scheme, s(0.125), LtsControls::default())?;
    let mut trace = Vec::with_capacity(steps);
    for _ in 0..steps {
        r.adv.store_old_time(gpu, &r.k, &r.f)?;
        let c = ddt.face_coeffs("the advective outlet on phi")?;
        ofgpu::field_ops::advective_robin_scalar(
            gpu,
            &r.k,
            &mut r.f.fr,
            &mut r.f.ref_value,
            &mut r.f.ref_grad,
            &r.f.bc_kind,
            BcKind::Advective,
            &r.adv.faces.w,
            &r.adv.faces.k_relax,
            &r.adv.phi_inf,
            &r.adv.b_old,
            &r.adv.b_old_old,
            &r.gm.b_delta_coeffs,
            c,
            r.n_bf,
        )?;
        correct_boundary_conditions(gpu, &r.k, &mut r.f, &r.gm)?;
        gpu.sync()?;
        let bf = gpu.download(&r.f.bf)?;
        trace.push(fin("phi_b", w64(bf[r.face])));
        ddt.advance(s(0.125));
    }
    Ok(trace)
}

/// The closed form (120.4e) of the relaxation, in f64: with `w`, `K`,
/// `phi_P`, `Delta_b` and `dt` frozen, `phi_b^n = phi* + (phi_b^0 - phi*)
/// r^n`.
fn closed_form(step: usize) -> f64 {
    let (w, k, delta, phi_p, phi_inf, b0) = (2.0f64, 0.5f64, 4.0f64, 3.0f64, 1.0f64, 10.0f64);
    let dt = 0.125f64;
    let phi_star = (k * phi_inf + w * delta * phi_p) / (k + w * delta);
    let alpha = w * dt * delta;
    let r = 1.0 / (1.0 + alpha + k * dt);
    phi_star + (b0 - phi_star) * r.powi(step as i32)
}

/// (120.4e), the gate: 20 Euler steps against the closed form, every step,
/// to 1e-12 relative (1e-5 under `single`).
#[test]
fn advective_face_relaxes_by_the_closed_form() {
    let Some(gpu) = gpu() else { return };
    let mut r = relax_face(&gpu).unwrap();
    let trace = march_relax(&gpu, &mut r, DdtScheme::Euler, 20).unwrap();
    for (n, got) in trace.iter().enumerate() {
        let want = closed_form(n + 1);
        let err = (got - want).abs() / want.abs();
        assert!(
            err <= tol(1e-12),
            "step {}: device {got:e} against (120.4e) {want:e}, rel {err:e}",
            n + 1
        );
    }
}

/// `backward` reaches the same fixed point `phi* = 24.5/8.5` - `a_n + a_0 +
/// a_00 = 0` - and every step is the host f64 march of `advective_triple`
/// to 1e-13 (1e-5 under `single`).
#[test]
fn advective_backward_reaches_the_fixed_point() {
    let Some(gpu) = gpu() else { return };
    let mut r = relax_face(&gpu).unwrap();
    let trace = march_relax(&gpu, &mut r, DdtScheme::Backward, 60).unwrap();

    // The host march: the SAME discrete scheme, no device in the loop -
    // Euler on the first step (psi^{n-2} does not exist yet), then the
    // constant-dt BDF2 coefficients (12, -16, 4).
    let (w, k, delta, phi_inf, phi_p, b0) = (2.0f64, 0.5f64, 4.0f64, 1.0f64, 3.0f64, 10.0f64);
    let (mut b_o, mut b_oo) = (b0, b0);
    let mut host = Vec::with_capacity(60);
    for step in 0..60 {
        let c = if step == 0 {
            DdtCoeffs { a_n: 8.0, a_0: -8.0, a_00: 0.0 }
        } else {
            DdtCoeffs { a_n: 12.0, a_0: -16.0, a_00: 4.0 }
        };
        let (fr, rv) = advective_triple(c, s(w), s(k), s(delta), s(phi_inf), s(b_o), s(b_oo));
        let face = w64(fr) * w64(rv) + (1.0 - w64(fr)) * phi_p;
        host.push(fin("host phi_b", face));
        b_oo = b_o;
        b_o = face;
    }

    for (n, (got, want)) in trace.iter().zip(&host).enumerate() {
        let err = (got - want).abs() / want.abs().max(1.0);
        assert!(
            err <= tol(1e-13),
            "step {}: device {got:e} against the host march {want:e}, rel {err:e}",
            n + 1
        );
    }

    let phi_star = 24.5 / 8.5;
    let got = trace[59];
    let err = (got - phi_star).abs() / phi_star;
    assert!(
        err <= tol(1e-12),
        "60 steps: device {got:e} against phi* {phi_star:e}, rel {err:e}"
    );
}

// ==========================================================================
//  The gate: a Gaussian pulse leaves through `advective`
// ==========================================================================

/// The channel: x [0, 1] in 200 cells, y and z one cell over [0, 1].
fn pulse_mesh() -> Result<HostMesh> {
    blockgen::build_mesh(&box_spec(
        [200, 1, 1],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        ["patch"; 6],
    ))
}

/// `dt = 0.25 dx`: the Courant number of the gate.
const DT: f64 = 0.001_25;
const STEPS: usize = 800;

/// One pulse run: `u = 1`, `phi = exp(-(x - 0.5)^2/(2 0.05^2))`, the inlet
/// `fixedValue 0`, the sides `zeroGradient`, and the outlet `advective`
/// (refreshed through `field::advective` before every assembly) or
/// `zeroGradient`. Returns `(incident, reflected)`: the peak at `t = 0.4`
/// and the largest magnitude left over `x < 0.9` at `t = 1.0`.
fn pulse(scheme: DdtScheme, advective_outlet: bool) -> Result<(f64, f64)> {
    let Some(gpu) = gpu() else { return Ok((f64::NAN, f64::NAN)) };
    let hm = pulse_mesh()?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let n = hm.n_cells;
    let xmin = patch_of(&hm, "xMin");
    let xmax = patch_of(&hm, "xMax");

    // The case. `advective` reaches the setup through the opt-in - the
    // promise this test keeps by refreshing it every step below.
    let mut raw = RawScalarField {
        name: "phi".into(),
        ..Default::default()
    };
    raw.internal = (0..n)
        .map(|c| {
            let x = f64::from(hm.c[c].x);
            s(-((x - 0.5) * (x - 0.5)) / (2.0 * 0.05 * 0.05)).exp()
        })
        .collect();
    let mut inlet = PatchFieldSpec {
        type_name: "fixedValue".into(),
        ..Default::default()
    };
    inlet.value = vec![s(0.0)];
    raw.boundary.insert("xMin".into(), inlet);
    let outlet_type = if advective_outlet { "advective" } else { "zeroGradient" };
    raw.boundary.insert(
        "xMax".into(),
        PatchFieldSpec {
            type_name: outlet_type.into(),
            ..Default::default()
        },
    );
    for p in ["yMin", "yMax", "zMin", "zMax"] {
        raw.boundary.insert(
            p.into(),
            PatchFieldSpec {
                type_name: "zeroGradient".into(),
                ..Default::default()
            },
        );
    }

    let mut psi = GpuScalarField::zeros(&gpu, &gm, "phi")?;
    field_setup::setup_scalar_field_with(
        &gpu,
        &mut psi,
        &raw,
        &hm,
        ofgpu::field_setup::BcInputs {
            advective: advective_outlet,
            ..Default::default()
        },
    )?;
    let mut adv = AdvectiveScalar::build(&gpu, &raw, &hm, &psi)?;

    // phi = +1 on every x-face (all of them, internal and outlet), -1 on
    // the inlet, 0 on the side faces.
    let mut phi = GpuSurfaceScalarField::zeros(&gpu, &gm, "phi")?;
    gpu.write(&mut phi.f, &vec![s(1.0); hm.n_internal_faces])?;
    let mut phib = vec![0.0 as Scalar; hm.n_boundary_faces];
    phib[hm.patches[xmin].start] = s(-1.0);
    phib[hm.patches[xmax].start] = s(1.0);
    gpu.write(&mut phi.bf, &phib)?;

    let fld = FieldKernels::new(&gpu)?;
    let fvk = FvKernels::new(&gpu)?;
    let lduk = LduKernels::new(&gpu)?;
    let solk = SolverKernels::new(&gpu)?;
    let mut ws = SolverWorkspace::for_mesh(&gpu, &gm)?;
    let mut a = GpuLduMatrix::new(&gpu, &gm)?;
    let mut w = gpu.zeros(hm.n_internal_faces)?;
    let mut bw = gpu.zeros(hm.n_boundary_faces)?;
    let ctrl = SolverControls {
        tolerance: s(if cfg!(feature = "single") { 1e-6 } else { 1e-12 }),
        rel_tol: 0.0,
        max_iter: 2000,
        check_interval: 1,
        ..Default::default()
    };
    let mut ddt = Ddt::new(&gpu, &gm, scheme, s(DT), LtsControls::default())?;

    let mut incident = 0.0f64;
    for step in 0..STEPS {
        if advective_outlet {
            // The step's old face levels, then the triple, BEFORE assembly.
            adv.store_old_time(&gpu, &fld, &psi)?;
            let c = ddt.face_coeffs("the advective outlet on phi")?;
            adv.update(&gpu, &fld, &mut psi, &phi, &gm, c)?;
        }

        // psi^{n-2} <- psi^{n-1} <- psi, in that order, once per step.
        ofgpu::field_ops::advance_time_levels(&gpu, &fld, &mut psi)?;

        a.zero(&gpu)?;
        ddt.add(&gpu, &mut a, &gm, &psi.f0, &psi.f00, s(1.0))?;
        fv::div_scheme_weights(
            &gpu,
            &fvk,
            Some(&mut w),
            Some(&mut bw),
            DivScheme::Central,
            &phi,
            &psi,
            None,
            &gm,
        )?;
        fv::fvm_div_gauss(&gpu, &fvk, &mut a, &gm, &phi, &w, &bw, &psi, s(1.0))?;
        add_boundary_contributions(&gpu, &lduk, &mut a, &gm)?;

        let perf = solve(&gpu, &solk, &mut psi.f, &a, &gm, &mut ws, &ctrl)?;
        assert!(
            perf.converged,
            "step {step}: the linear solve stagnated at {:e}",
            perf.final_residual
        );

        correct_boundary_conditions(&gpu, &fld, &mut psi, &gm)?;
        ddt.advance(s(DT));

        if step + 1 == 320 {
            gpu.sync()?;
            let f = gpu.download(&psi.f)?;
            incident = f.iter().fold(0.0f64, |m, v| m.max(w64(*v).abs()));
            fin("incident peak", incident);
        }
    }
    gpu.sync()?;

    let f = gpu.download(&psi.f)?;
    let reflected = (0..n)
        .filter(|c| f64::from(hm.c[*c].x) < 0.9)
        .fold(0.0f64, |m, c| m.max(w64(f[c]).abs()));
    fin("reflected peak", reflected);
    Ok((incident, reflected))
}

/// The gate itself: with `advective` the reflected peak is at most 1 % of
/// the incident one under Euler and `backward`; with `zeroGradient` it
/// exceeds 1 %, so the gate is not vacuous. All four ratios are printed.
#[test]
fn a_gaussian_pulse_leaves_through_advective() {
    for scheme in [DdtScheme::Euler, DdtScheme::Backward] {
        let (ai, ar) = pulse(scheme, true).unwrap();
        let (zi, zr) = pulse(scheme, false).unwrap();
        let a_ratio = ar / ai;
        let z_ratio = zr / zi;
        println!(
            "{scheme:?} ratio advective = {a_ratio:e} (incident {ai:e}, reflected {ar:e}), \
             ratio zeroGradient = {z_ratio:e} (incident {zi:e}, reflected {zr:e})"
        );
        assert!(
            a_ratio <= 0.01,
            "{scheme:?}: the advective outlet reflects {a_ratio:e} of the incident peak"
        );
        assert!(
            z_ratio > 0.01,
            "{scheme:?}: the zeroGradient baseline reflects only {z_ratio:e} - the gate is vacuous"
        );
    }
}
