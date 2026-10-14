// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 120.10 - the recycled inflow through the public API, in BOTH
//! builds: `mapped`/`mappedFixedValue` copying the cells of an offset plane
//! ((120.10a)-(120.10b), bit for bit), `setAverage` holding the patch mean
//! and the flux ((120.10c)-(120.10d)), a recycled laminar channel against
//! the cyclic channel through the whole SIMPLE loop, `turbulentInlet`'s
//! RMS and memory ((120.10e)-(120.10h)) over 1e4 seeded samples, and the
//! refusals by name. Tolerances follow docs/17 §5.1: round-off rows are
//! class R (D-F32-1, the `tol` const fn below), the channel and the RMS
//! bands are class P, the same in both builds; everything is compared in
//! f64. A machine without a card passes the GPU tests vacuously, the
//! convention of `tests/f32_boussinesq.rs`.
//!
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::field::mapped::{mapped_values, turbulent_advance, turbulent_seed};
use ofgpu::field_ops::{
    advance_turbulent_inlet_scalar, advance_turbulent_inlet_vector, correct_boundary_conditions,
    correct_boundary_conditions_vector, update_mapped_scalar, update_mapped_vector, FieldKernels,
};
use ofgpu::field_setup::{
    scalar_boundary_with, setup_scalar_field_with, setup_vector_field_with, vector_boundary_with,
    BcInputs,
};
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::momentum::{BuoyancyCoeffs, BuoyancyForm, MomentumControls};
use ofgpu::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use ofgpu::simple::{Simple, SimpleControls};
use ofgpu::{
    BcKind, Error, Gpu, GpuMesh, GpuScalarField, GpuVectorField, HostMesh, Label, Result, Scalar,
    Vec3,
};

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

/// A NaN-propagating max: `f64::max` swallows NaN.
fn nmax(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// A checked term: a non-finite value panics here, naming itself.
fn fin(what: &str, x: f64) -> f64 {
    assert!(x.is_finite(), "{what} is non-finite: {x:e}");
    x
}

/// A uniform block with the six patch types given.
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

/// A patch entry with its entries, as the reader captures them.
fn bc(name: &str, extra: &[(&str, &str)]) -> PatchFieldSpec {
    let mut s = ps(name);
    for (k, v) in extra {
        s.extra.insert((*k).into(), (*v).into());
    }
    s
}

/// A fixed value at a scalar.
fn fixed(v: Scalar) -> PatchFieldSpec {
    let mut s = ps("fixedValue");
    s.value = vec![v];
    s
}

/// Tight solvers: what the tests measure is not the stopping criterion.
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
    gpu.write(&mut nut.bc_kind, &vec![BcKind::Calculated as Label; hm.n_boundary_faces])?;
    Ok(nut)
}

/// A uniform temperature `t0`, zeroGradient everywhere.
fn uniform_t(gpu: &Gpu, hm: &HostMesh, gm: &GpuMesh, t0: Scalar) -> Result<GpuScalarField> {
    let mut t = GpuScalarField::zeros(gpu, gm, "T")?;
    gpu.write(&mut t.bc_kind, &vec![BcKind::ZeroGradient as Label; hm.n_boundary_faces])?;
    gpu.write(&mut t.f, &vec![t0; hm.n_cells])?;
    gpu.write(&mut t.bf, &vec![t0; hm.n_boundary_faces])?;
    Ok(t)
}

fn raw_vector(name: &str, boundary: Vec<(&str, PatchFieldSpec)>, internal: Vec3) -> RawVectorField {
    let mut raw = RawVectorField {
        name: name.into(),
        ..Default::default()
    };
    for (p, s) in boundary {
        raw.boundary.insert(p.into(), s);
    }
    raw.internal = vec![internal];
    raw
}

fn raw_scalar(name: &str, boundary: Vec<(&str, PatchFieldSpec)>, internal: Scalar) -> RawScalarField {
    let mut raw = RawScalarField {
        name: name.into(),
        ..Default::default()
    };
    for (p, s) in boundary {
        raw.boundary.insert(p.into(), s);
    }
    raw.internal = vec![internal];
    raw
}

/// Six patches, xMin carrying `inlet`, the rest zeroGradient.
fn with_inlet(inlet: PatchFieldSpec) -> Vec<(&'static str, PatchFieldSpec)> {
    vec![
        ("xMin", inlet),
        ("xMax", ps("zeroGradient")),
        ("yMin", ps("zeroGradient")),
        ("yMax", ps("zeroGradient")),
        ("zMin", ps("zeroGradient")),
        ("zMax", ps("zeroGradient")),
    ]
}

/// Mesh B: 4 x 3 x 2 cells over [0,2] x [0,1] x [0,1], all six `patch` -
/// cell centres at x = 0.25, 0.75, 1.25, 1.75, so `offset (1.25 0 0)` from
/// xMin lands on the centres of the third column.
fn mesh_b() -> Result<HostMesh> {
    blockgen::build_mesh(&box_spec(
        [4, 3, 2],
        [0.0, 0.0, 0.0],
        [2.0, 1.0, 1.0],
        ["patch", "patch", "patch", "patch", "patch", "patch"],
    ))
}

/// T_c = 1 + x + 2y^2 + 0.5z at the f64 cell centres, cast once.
fn cells_t(hm: &HostMesh) -> Vec<Scalar> {
    hm.geom64()
        .c
        .iter()
        .map(|p| (1.0 + p.x + 2.0 * p.y * p.y + 0.5 * p.z) as Scalar)
        .collect()
}

/// U_c = (1 + x, y - z, 2z) at the f64 cell centres, cast once.
fn cells_u(hm: &HostMesh) -> Vec<Vec3> {
    hm.geom64()
        .c
        .iter()
        .map(|p| Vec3::new((1.0 + p.x) as Scalar, (p.y - p.z) as Scalar, (2.0 * p.z) as Scalar))
        .collect()
}

/// The cell whose f64 centre is the f64 face centre of `f` plus `d`, by a
/// direct search on the centres - the answer key for (120.10a).
fn centre_cell(hm: &HostMesh, f: usize, d: [f64; 3]) -> usize {
    let g = hm.geom64();
    let x = g.b_cf[f];
    let want = [x.x + d[0], x.y + d[1], x.z + d[2]];
    let hits: Vec<usize> = (0..hm.n_cells)
        .filter(|&c| {
            let p = g.c[c];
            (p.x - want[0]).abs() < 1e-9 && (p.y - want[1]).abs() < 1e-9 && (p.z - want[2]).abs() < 1e-9
        })
        .collect();
    assert_eq!(hits.len(), 1, "face {f}: {} cells centred at the sample point", hits.len());
    hits[0]
}

fn flat(v: &[Vec3]) -> Vec<Scalar> {
    v.iter().flat_map(|u| [u.x, u.y, u.z]).collect()
}

fn comp(v: Vec3, k: usize) -> f64 {
    match k {
        0 => w64(v.x),
        1 => w64(v.y),
        _ => w64(v.z),
    }
}

/// `BcInputs` with the mapped opt-in.
fn mapped_on() -> BcInputs<'static> {
    BcInputs {
        mapped: true,
        ..BcInputs::default()
    }
}

/// `BcInputs` with the turbulentInlet opt-in.
fn turbulent_on() -> BcInputs<'static> {
    BcInputs {
        turbulent_inlet: true,
        ..BcInputs::default()
    }
}

// ==========================================================================
//  (120.10a)-(120.10b): the offset plane's cells, copied
// ==========================================================================

#[test]
fn mapped_copies_the_sampled_cell() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = mesh_b()?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let k = FieldKernels::new(&gpu)?;
    let xin = patch_of(&hm, "xMin");
    let at = hm.patches[xin].start;
    let np = hm.patches[xin].size;
    let d = [1.25, 0.0, 0.0];

    // Scalar, `mappedFixedValue`.
    let mut t = GpuScalarField::zeros(&gpu, &gm, "T")?;
    let raw = raw_scalar("T", with_inlet(bc("mappedFixedValue", &[("offset", "(1.25 0 0)")])), 0.0);
    setup_scalar_field_with(&gpu, &mut t, &raw, &hm, mapped_on())?;
    assert_eq!(t.mapped.len(), 1, "one mapped rule on xMin");
    let tc = cells_t(&hm);
    gpu.write(&mut t.f, &tc)?;
    update_mapped_scalar(&gpu, &k, &mut t, &gm)?;
    correct_boundary_conditions(&gpu, &k, &mut t, &gm)?;
    gpu.sync()?;
    let bf = gpu.download(&t.bf)?;
    let fr = gpu.download(&t.fr)?;
    let kinds = gpu.download(&t.bc_kind)?;
    for i in 0..np {
        let f = at + i;
        let c = centre_cell(&hm, f, d);
        assert_eq!(t.mapped[0].rule.cells[i] as usize, c, "face {i}: sampled cell");
        assert_eq!(bf[f].to_bits(), tc[c].to_bits(), "face {i}: bf {} vs T_c {}", bf[f], tc[c]);
        assert_eq!(fr[f].to_bits(), (1.0 as Scalar).to_bits(), "face {i}: fr");
        assert_eq!(kinds[f], BcKind::FixedValue as Label, "face {i}: kind");
    }
    eprintln!("  [120.10] mappedFixedValue: {np} faces copied bit for bit from column 3");

    // Vector, the `mapped` spelling.
    let mut u = GpuVectorField::zeros(&gpu, &gm, "U")?;
    let raw = raw_vector("U", with_inlet(bc("mapped", &[("offset", "(1.25 0 0)")])), Vec3::ZERO);
    setup_vector_field_with(&gpu, &mut u, &raw, &hm, mapped_on())?;
    assert_eq!(u.mapped.len(), 1, "one mapped rule on xMin");
    let uc = cells_u(&hm);
    gpu.write(&mut u.f, &uc)?;
    update_mapped_vector(&gpu, &k, &mut u, &gm)?;
    correct_boundary_conditions_vector(&gpu, &k, &mut u, &gm)?;
    gpu.sync()?;
    let bf = gpu.download(&u.bf)?;
    for i in 0..np {
        let f = at + i;
        let c = centre_cell(&hm, f, d);
        for kk in 0..3 {
            assert_eq!(
                comp(bf[f], kk).to_bits(),
                comp(uc[c], kk).to_bits(),
                "face {i} component {kk}: bf {:?} vs U_c {:?}",
                bf[f],
                uc[c]
            );
        }
    }
    eprintln!("  [120.10] mapped (vector): {np} faces copied bit for bit");
    Ok(())
}

// ==========================================================================
//  (120.10c)-(120.10d): setAverage holds the mean and the flux
// ==========================================================================

#[test]
fn set_average_holds_the_mean_and_the_flux() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = mesh_b()?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let k = FieldKernels::new(&gpu)?;
    let xin = patch_of(&hm, "xMin");
    let at = hm.patches[xin].start;
    let np = hm.patches[xin].size;
    let d = [1.25, 0.0, 0.0];
    let tc = cells_t(&hm);
    let area: f64 = (0..np).map(|i| w64(hm.b_mag_sf[at + i])).sum();

    // The ratio branch (average 5 over a sample mean of about 3.15) and the
    // shift branch (average 100, the sample mean below half of it).
    for (avg, ratio) in [(5.0f64, true), (100.0f64, false)] {
        let mut t = GpuScalarField::zeros(&gpu, &gm, "T")?;
        let a = format!("{avg}");
        let raw = raw_scalar(
            "T",
            with_inlet(bc(
                "mappedFixedValue",
                &[("offset", "(1.25 0 0)"), ("setAverage", "true"), ("average", &a)],
            )),
            0.0,
        );
        setup_scalar_field_with(&gpu, &mut t, &raw, &hm, mapped_on())?;
        gpu.write(&mut t.f, &tc)?;
        update_mapped_scalar(&gpu, &k, &mut t, &gm)?;
        correct_boundary_conditions(&gpu, &k, &mut t, &gm)?;
        gpu.sync()?;
        let bf = gpu.download(&t.bf)?;
        let rv = gpu.download(&t.ref_value)?;

        let mut sb = 0.0f64;
        for i in 0..np {
            sb += fin("|S| bf", w64(hm.b_mag_sf[at + i]) * w64(bf[at + i]));
        }
        let mean = sb / area;
        eprintln!("  [120.10] setAverage {avg}: area mean {mean:.14} (ratio branch {ratio})");
        assert!((mean - avg).abs() <= tol(1e-12) * avg, "area mean {mean:e}, want {avg}");

        // The branch: one factor (bf/psi* constant) or one shift (bf - psi*).
        let c0 = centre_cell(&hm, at, d);
        let key0 = if ratio { w64(bf[at]) / w64(tc[c0]) } else { w64(bf[at]) - w64(tc[c0]) };
        let mut worst = 0.0f64;
        for i in 0..np {
            let c = centre_cell(&hm, at + i, d);
            let key = if ratio {
                w64(bf[at + i]) / w64(tc[c])
            } else {
                w64(bf[at + i]) - w64(tc[c])
            };
            worst = nmax(worst, fin("branch key", (key - key0).abs()));
        }
        let scale = if ratio { key0.abs() } else { avg };
        eprintln!("  [120.10] setAverage {avg}: branch key {key0:.12}, spread {worst:.3e}");
        assert!(worst <= tol(1e-14) * scale, "branch spread {worst:e} (key {key0})");

        // The host twin.
        let mut host = vec![0.0 as Scalar; hm.n_boundary_faces];
        mapped_values(&t.mapped[0].rule, 1, &tc, &hm.b_mag_sf, &mut host);
        let mut tw = 0.0f64;
        for i in 0..np {
            tw = nmax(tw, fin("twin", (w64(host[at + i]) - w64(rv[at + i])).abs()));
        }
        eprintln!("  [120.10] setAverage {avg}: device vs host twin {tw:.3e}");
        assert!(tw <= tol(1e-14) * avg, "device vs twin {tw:e}");
    }

    // Vector: average (2 0 0) over a sample mean of (2.25, 0, 1) - the
    // ratio branch scales by 2/|m| and the shift removes the cross part.
    let mut u = GpuVectorField::zeros(&gpu, &gm, "U")?;
    let raw = raw_vector(
        "U",
        with_inlet(bc(
            "mapped",
            &[("offset", "(1.25 0 0)"), ("setAverage", "yes"), ("average", "(2 0 0)")],
        )),
        Vec3::ZERO,
    );
    setup_vector_field_with(&gpu, &mut u, &raw, &hm, mapped_on())?;
    let uc = cells_u(&hm);
    gpu.write(&mut u.f, &uc)?;
    update_mapped_vector(&gpu, &k, &mut u, &gm)?;
    correct_boundary_conditions_vector(&gpu, &k, &mut u, &gm)?;
    gpu.sync()?;
    let bf = gpu.download(&u.bf)?;
    let rv = gpu.download(&u.ref_value)?;
    let want = [2.0f64, 0.0, 0.0];
    let mut flux = 0.0f64;
    for (kk, w) in want.iter().enumerate() {
        let mut s = 0.0f64;
        for i in 0..np {
            s += fin("|S| bf", w64(hm.b_mag_sf[at + i]) * comp(bf[at + i], kk));
        }
        let mean = s / area;
        eprintln!("  [120.10] setAverage (2 0 0): component {kk} mean {mean:.14}");
        assert!((mean - w).abs() <= tol(1e-12) * 2.0, "component {kk} mean {mean:e}, want {w}");
    }
    for i in 0..np {
        let sf = hm.b_sf[at + i];
        flux -= w64(bf[at + i].x) * w64(sf.x) + w64(bf[at + i].y) * w64(sf.y) + w64(bf[at + i].z) * w64(sf.z);
    }
    eprintln!("  [120.10] setAverage (2 0 0): inflow {flux:.14} (want 2 A = {})", 2.0 * area);
    assert!((flux - 2.0 * area).abs() <= tol(1e-12) * 2.0 * area, "inflow {flux:e}");
    let mut host = vec![0.0 as Scalar; 3 * hm.n_boundary_faces];
    mapped_values(&u.mapped[0].rule, 3, &flat(&uc), &hm.b_mag_sf, &mut host);
    let mut tw = 0.0f64;
    for i in 0..np {
        for kk in 0..3 {
            tw = nmax(tw, fin("twin", (w64(host[3 * (at + i) + kk]) - comp(rv[at + i], kk)).abs()));
        }
    }
    eprintln!("  [120.10] setAverage (2 0 0): device vs host twin {tw:.3e}");
    // 4 bounds |U| on the patch (|U| <= 2.75 at x = 1.25).
    assert!(tw <= tol(1e-14) * 4.0, "device vs twin {tw:e}");
    Ok(())
}

// ==========================================================================
//  THE gate: a recycled laminar channel is the cyclic channel
// ==========================================================================

/// SIMPLE until the cell change of U is below 1e-14, at most `cap`
/// iterations; non-finite U is an `Error::Diverged`. Returns U and the
/// count.
fn run(
    sim: &mut Simple,
    gpu: &Gpu,
    backend: &mut PbicgstabBackend,
    nut: &GpuScalarField,
    t: &GpuScalarField,
    cap: usize,
) -> Result<(Vec<Vec3>, usize)> {
    let mut prev = gpu.download(&sim.u().f)?;
    for it in 0..cap {
        sim.correct(gpu, backend, nut, t)?;
        let now = gpu.download(&sim.u().f)?;
        let mut change = 0.0f64;
        for (a, b) in prev.iter().zip(&now) {
            for kk in 0..3 {
                let x = comp(*b, kk);
                if !x.is_finite() {
                    return Err(Error::Diverged {
                        iteration: it,
                        what: format!("U went non-finite: {b:?}"),
                    });
                }
                change = change.max((x - comp(*a, kk)).abs());
            }
        }
        prev = now;
        if change < 1e-14 {
            return Ok((prev, it + 1));
        }
    }
    Ok((prev, cap))
}

/// The u.x profile of the column of cells centred at `x`, by rising y,
/// and its bulk mean.
fn column(hm: &HostMesh, u: &[Vec3], x: f64) -> (Vec<f64>, f64) {
    let g = hm.geom64();
    let mut cells: Vec<usize> = (0..hm.n_cells).filter(|&c| (g.c[c].x - x).abs() < 1e-9).collect();
    cells.sort_by(|a, b| g.c[*a].y.total_cmp(&g.c[*b].y));
    let prof: Vec<f64> = cells.iter().map(|&c| fin("u.x", w64(u[c].x))).collect();
    let bulk = prof.iter().sum::<f64>() / prof.len() as f64;
    (prof, bulk)
}

#[test]
fn a_recycled_channel_is_the_cyclic_channel() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    // 16 x 16 x 1 cells over [0,2] x [0,1] x [0,1/16], nu 0.01: Re = 100 on
    // the height at a bulk speed of 1, so a channel's entry length (about
    // 0.05 Re H) is well past the sample plane x = 1.0625, the centre of the
    // ninth column - a UNIFORM inlet there is 21 % off the developed
    // profile (measured once, SPEC-LIT 120.10), so the 1 % band sees the
    // recycling and not the entry flow.
    let (n, lo, hi) = ([16usize, 16, 1], [0.0, 0.0, 0.0], [2.0, 1.0, 0.0625]);
    let nu: Scalar = 0.01;
    let xs = 1.0625f64;
    let ctrl = SimpleControls {
        momentum: MomentumControls {
            nu,
            u_solver: tight(),
            ..MomentumControls::default()
        },
        p_solver: tight(),
        ..SimpleControls::default()
    };
    let cap = 4000usize;
    let walls = |u: bool| -> Vec<(&'static str, PatchFieldSpec)> {
        vec![
            ("yMin", if u { ps("noSlip") } else { ps("zeroGradient") }),
            ("yMax", if u { ps("noSlip") } else { ps("zeroGradient") }),
            ("zMin", ps("empty")),
            ("zMax", ps("empty")),
        ]
    };

    // The cyclic channel, driven by a uniform body force b = 12 nu U_b/H^2
    // = 0.12 along x: the density-ratio form g (T_ref/T - 1) with T_ref 2
    // and T 1 is g itself. p is pinned at 0 on yMin, the level the
    // developed flow has everywhere.
    let mut b = box_spec(n, lo, hi, ["wall", "wall", "wall", "wall", "empty", "empty"]);
    b.set_cyclic_axis(0)?;
    let hc = blockgen::build_mesh(&b)?;
    let gc = GpuMesh::upload(&gpu, &hc)?;
    let buoy = BuoyancyCoeffs {
        g: Vec3::new(0.12, 0.0, 0.0),
        t_ref: 2.0,
        t_min: 0.5,
        form: BuoyancyForm::DensityRatio,
    };
    let mut sc = Simple::new(&gpu, &hc, &gc, ctrl, buoy)?;
    let mut ub = walls(true);
    ub.push(("xMin", ps("cyclic")));
    ub.push(("xMax", ps("cyclic")));
    setup_vector_field_with(&gpu, sc.u_mut(), &raw_vector("U", ub, Vec3::ZERO), &hc, BcInputs::default())?;
    let mut pb = walls(false);
    pb[0] = ("yMin", fixed(0.0));
    pb.push(("xMin", ps("cyclic")));
    pb.push(("xMax", ps("cyclic")));
    setup_scalar_field_with(&gpu, sc.p_mut(), &raw_scalar("p", pb, 0.0), &hc, BcInputs::default())?;
    sc.initialise(&gpu)?;
    let nut = laminar_nut(&gpu, &hc, &gc)?;
    let t1 = uniform_t(&gpu, &hc, &gc, 1.0)?;
    let mut be = PbicgstabBackend::new(ctrl.p_solver);
    be.setup(&gpu, &hc, &gc, &SystemProbe::default())?;
    let (u_cyc, it_cyc) = run(&mut sc, &gpu, &mut be, &nut, &t1, cap)?;
    let (p_cyc, bulk_cyc) = column(&hc, &u_cyc, xs);
    eprintln!("  [120.10] cyclic channel: {it_cyc} iterations, bulk {bulk_cyc:.10}");

    // The recycled channel: xMin samples the plane x = 1.0625 and holds the
    // bulk at 1 by setAverage; xMax lets it out at p = 0.
    let hr = blockgen::build_mesh(&box_spec(n, lo, hi, ["patch", "patch", "wall", "wall", "empty", "empty"]))?;
    let gr = GpuMesh::upload(&gpu, &hr)?;
    let mut sr = Simple::new(
        &gpu,
        &hr,
        &gr,
        ctrl,
        BuoyancyCoeffs {
            g: Vec3::ZERO,
            ..BuoyancyCoeffs::default()
        },
    )?;
    let mut ub = walls(true);
    ub.push((
        "xMin",
        bc(
            "mappedFixedValue",
            &[("offset", "(1.0625 0 0)"), ("setAverage", "true"), ("average", "(1 0 0)")],
        ),
    ));
    ub.push(("xMax", ps("zeroGradient")));
    setup_vector_field_with(&gpu, sr.u_mut(), &raw_vector("U", ub, Vec3::new(1.0, 0.0, 0.0)), &hr, mapped_on())?;
    let mut pb = walls(false);
    pb.push(("xMin", ps("zeroGradient")));
    pb.push(("xMax", fixed(0.0)));
    setup_scalar_field_with(&gpu, sr.p_mut(), &raw_scalar("p", pb, 0.0), &hr, mapped_on())?;
    sr.initialise(&gpu)?;
    let nut = laminar_nut(&gpu, &hr, &gr)?;
    let t0 = uniform_t(&gpu, &hr, &gr, 300.0)?;
    let mut be = PbicgstabBackend::new(ctrl.p_solver);
    be.setup(&gpu, &hr, &gr, &SystemProbe::default())?;
    let (u_rec, it_rec) = run(&mut sr, &gpu, &mut be, &nut, &t0, cap)?;
    let (p_rec, bulk_rec) = column(&hr, &u_rec, xs);
    eprintln!("  [120.10] recycled channel: {it_rec} iterations, bulk {bulk_rec:.10}");
    if !cfg!(feature = "single") {
        assert!(it_cyc < cap && it_rec < cap, "the f64 loops must stop on the change, not the cap");
    }

    // The profiles, each by its own bulk, at the sample plane.
    let peak = p_cyc.iter().fold(0.0f64, |m, v| nmax(m, v / bulk_cyc));
    let mut worst = 0.0f64;
    for (j, (a, c)) in p_rec.iter().zip(&p_cyc).enumerate() {
        let d = (a / bulk_rec - c / bulk_cyc).abs();
        let y = (j as f64 + 0.5) / 16.0;
        let exact = 1.5 * (1.0 - (2.0 * y - 1.0) * (2.0 * y - 1.0));
        eprintln!(
            "  [120.10]   y {y:.5}: recycled {:.8} cyclic {:.8} (Poiseuille {exact:.8})",
            a / bulk_rec,
            c / bulk_cyc
        );
        worst = nmax(worst, fin("profile difference", d));
    }
    eprintln!("  [120.10] worst profile difference {:.4e} of the peak {peak:.6}", worst);
    assert!(worst <= 0.01 * peak, "recycled vs cyclic: {worst:e} against 1 % of {peak}");

    // setAverage holds the flux: the inflow is the bulk 1 through the area.
    let xin = patch_of(&hr, "xMin");
    let (at, np) = (hr.patches[xin].start, hr.patches[xin].size);
    let phi = gpu.download(&sr.phi().bf)?;
    let area: f64 = (0..np).map(|i| w64(hr.b_mag_sf[at + i])).sum();
    let q: f64 = -(0..np).map(|i| fin("phi", w64(phi[at + i]))).sum::<f64>();
    eprintln!("  [120.10] recycled inflow {q:.15} (want {area})");
    assert!((q - area).abs() <= tol(1e-12) * area, "inflow {q:e}, want {area}");
    Ok(())
}

// ==========================================================================
//  (120.10e)-(120.10h): turbulentInlet's RMS and memory
// ==========================================================================

/// Mesh T: 1 x 10 x 10 cells over [0,1]^3 - 100 xMin faces.
fn mesh_t() -> Result<HostMesh> {
    blockgen::build_mesh(&box_spec(
        [1, 10, 10],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        ["patch", "patch", "patch", "patch", "patch", "patch"],
    ))
}

/// A vector turbulentInlet on xMin: reference (2 0 0), scale (0.1 0.2 0.05)
/// - sigma (0.2 0.4 0.1) - with the given alpha and seed.
fn turbulent_u(gpu: &Gpu, gm: &GpuMesh, hm: &HostMesh, alpha: &str, seed: &str) -> Result<GpuVectorField> {
    let mut u = GpuVectorField::zeros(gpu, gm, "U")?;
    let raw = raw_vector(
        "U",
        with_inlet(bc(
            "turbulentInlet",
            &[
                ("referenceField", "uniform (2 0 0)"),
                ("fluctuationScale", "(0.1 0.2 0.05)"),
                ("alpha", alpha),
                ("seed", seed),
            ],
        )),
        Vec3::ZERO,
    );
    setup_vector_field_with(gpu, &mut u, &raw, hm, turbulent_on())?;
    Ok(u)
}

#[test]
fn turbulent_inlet_rms_is_the_prescribed_scale() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = mesh_t()?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let k = FieldKernels::new(&gpu)?;
    let xin = patch_of(&hm, "xMin");
    let (at, np) = (hm.patches[xin].start, hm.patches[xin].size);
    assert_eq!(np, 100);
    let sigma = [0.2f64, 0.4, 0.1];
    let reference = [2.0f64, 0.0, 0.0];

    // alpha 1: white, the seed plus 99 steps = 1e4 samples a component.
    let mut u = turbulent_u(&gpu, &gm, &hm, "1", "7")?;
    assert_eq!(u.turbulent.len(), 1, "one turbulentInlet rule");
    let rule = u.turbulent[0].rule.clone();
    let mut host = vec![0.0 as Scalar; 3 * hm.n_boundary_faces];
    turbulent_seed(&rule, &mut host);
    let (mut s1, mut s2) = ([0.0f64; 3], [0.0f64; 3]);
    let mut tw = 0.0f64;
    let mut count = 0usize;
    for n in 0..100u64 {
        if n > 0 {
            advance_turbulent_inlet_vector(&gpu, &k, &mut u)?;
            turbulent_advance(&rule, n, &mut host);
        }
        let rv = gpu.download(&u.ref_value)?;
        for i in 0..np {
            for kk in 0..3 {
                let v = comp(rv[at + i], kk);
                let z = fin("fluctuation", (v - reference[kk]) / sigma[kk]);
                s1[kk] += z;
                s2[kk] += z * z;
                tw = nmax(tw, (w64(host[3 * (at + i) + kk]) - v).abs());
            }
        }
        count += np;
    }
    assert_eq!(count, 10_000);
    for kk in 0..3 {
        let mean = s1[kk] / count as f64;
        let rms = (s2[kk] / count as f64).sqrt();
        eprintln!("  [120.10] turbulentInlet alpha 1, component {kk}: RMS/sigma {rms:.5}, mean/sigma {mean:.5}");
        assert!((rms - 1.0).abs() <= 0.05, "component {kk}: RMS/sigma {rms}");
        assert!(mean.abs() <= 0.05, "component {kk}: mean/sigma {mean}");
    }
    eprintln!("  [120.10] turbulentInlet: device vs host twin over 1e4 x 3 draws {tw:.3e}");
    assert!(tw <= tol(1e-14) * 2.0, "device vs twin {tw:e}");

    // The face value is the fixed value it carries.
    correct_boundary_conditions_vector(&gpu, &k, &mut u, &gm)?;
    let bf = gpu.download(&u.bf)?;
    let rv = gpu.download(&u.ref_value)?;
    for i in 0..np {
        assert_eq!(flat(&bf[at + i..at + i + 1]), flat(&rv[at + i..at + i + 1]), "face {i}: bf is refValue");
    }

    // The same seed draws the same field, bit for bit; another seed does not.
    let mut a = turbulent_u(&gpu, &gm, &hm, "1", "7")?;
    let mut c = turbulent_u(&gpu, &gm, &hm, "1", "8")?;
    for _ in 0..5 {
        advance_turbulent_inlet_vector(&gpu, &k, &mut a)?;
        advance_turbulent_inlet_vector(&gpu, &k, &mut c)?;
    }
    let mut u5 = turbulent_u(&gpu, &gm, &hm, "1", "7")?;
    for _ in 0..5 {
        advance_turbulent_inlet_vector(&gpu, &k, &mut u5)?;
    }
    let (ra, rc, r5) = (gpu.download(&a.ref_value)?, gpu.download(&c.ref_value)?, gpu.download(&u5.ref_value)?);
    let bits = |v: &[Vec3]| -> Vec<u64> { flat(v).iter().map(|x| w64(*x).to_bits()).collect() };
    assert_eq!(bits(&ra), bits(&r5), "seed 7 twice must draw the same field");
    assert_ne!(bits(&ra), bits(&rc), "seeds 7 and 8 must differ");

    // alpha 0.1: the memory of (120.10g) - 1000 steps, RMS still sigma and
    // the lag-1 autocorrelation mu = 0.9.
    let mut u = turbulent_u(&gpu, &gm, &hm, "0.1", "3")?;
    let (mut s2, mut s11) = ([0.0f64; 3], [0.0f64; 3]);
    let mut prev = gpu.download(&u.ref_value)?;
    let mut s2_all = [0.0f64; 3];
    let steps = 1000usize;
    for i in 0..np {
        for kk in 0..3 {
            let z = (comp(prev[at + i], kk) - reference[kk]) / sigma[kk];
            s2_all[kk] += z * z;
        }
    }
    for _ in 0..steps {
        advance_turbulent_inlet_vector(&gpu, &k, &mut u)?;
        let now = gpu.download(&u.ref_value)?;
        for i in 0..np {
            for kk in 0..3 {
                let z0 = (comp(prev[at + i], kk) - reference[kk]) / sigma[kk];
                let z1 = fin("fluctuation", (comp(now[at + i], kk) - reference[kk]) / sigma[kk]);
                s2[kk] += z0 * z0;
                s11[kk] += z0 * z1;
                s2_all[kk] += z1 * z1;
            }
        }
        prev = now;
    }
    for kk in 0..3 {
        let rms = (s2_all[kk] / ((steps + 1) * np) as f64).sqrt();
        let rho = s11[kk] / s2[kk];
        eprintln!("  [120.10] turbulentInlet alpha 0.1, component {kk}: RMS/sigma {rms:.5}, lag-1 {rho:.5}");
        assert!((rms - 1.0).abs() <= 0.05, "component {kk}: RMS/sigma {rms}");
        assert!((rho - 0.9).abs() <= 0.05, "component {kk}: lag-1 autocorrelation {rho}");
    }

    // A scalar: reference 300, scale 0.02 - sigma 6.
    let mut t = GpuScalarField::zeros(&gpu, &gm, "T")?;
    let raw = raw_scalar(
        "T",
        with_inlet(bc(
            "turbulentInlet",
            &[("referenceField", "uniform 300"), ("fluctuationScale", "0.02"), ("alpha", "1")],
        )),
        300.0,
    );
    setup_scalar_field_with(&gpu, &mut t, &raw, &hm, turbulent_on())?;
    let (mut s2, mut cnt) = (0.0f64, 0usize);
    for n in 0..100 {
        if n > 0 {
            advance_turbulent_inlet_scalar(&gpu, &k, &mut t)?;
        }
        let rv = gpu.download(&t.ref_value)?;
        for i in 0..np {
            let z = fin("T fluctuation", (w64(rv[at + i]) - 300.0) / 6.0);
            s2 += z * z;
        }
        cnt += np;
    }
    let rms = (s2 / cnt as f64).sqrt();
    eprintln!("  [120.10] turbulentInlet on T: RMS/sigma {rms:.5} over {cnt} samples");
    assert!((rms - 1.0).abs() <= 0.05, "T: RMS/sigma {rms}");
    Ok(())
}

// ==========================================================================
//  Refusals by name
// ==========================================================================

#[test]
fn mapped_and_turbulent_inlet_refusals_are_by_name() -> Result<()> {
    let hm = mesh_b()?;
    let tc = cells_t(&hm);
    let uc = cells_u(&hm);
    for name in ["mapped", "mappedFixedValue", "turbulentInlet"] {
        assert_eq!(BcKind::from_name(name, "U", "p0")?, BcKind::FixedValue, "{name}");
    }
    let scalar = |spec: PatchFieldSpec, inputs: BcInputs| -> String {
        match scalar_boundary_with(&raw_scalar("T", with_inlet(spec), 0.0), &hm, &tc, inputs) {
            Ok(_) => String::from("(accepted)"),
            Err(e) => e.to_string(),
        }
    };
    let vector = |spec: PatchFieldSpec, inputs: BcInputs| -> String {
        match vector_boundary_with(&raw_vector("U", with_inlet(spec), Vec3::ZERO), &hm, &uc, inputs) {
            Ok(_) => String::from("(accepted)"),
            Err(e) => e.to_string(),
        }
    };
    let off = ("offset", "(1.25 0 0)");
    let cases: Vec<(String, &str)> = vec![
        (scalar(bc("mappedFixedValue", &[off]), BcInputs::default()), "BcInputs::mapped"),
        (vector(bc("mapped", &[off]), BcInputs::default()), "BcInputs::mapped"),
        (scalar(bc("mapped", &[]), mapped_on()), "offset"),
        (scalar(bc("mapped", &[("offset", "(5 0 0)")]), mapped_on()), "lies in no cell"),
        (scalar(bc("mapped", &[off, ("setAverage", "true")]), mapped_on()), "average"),
        (scalar(bc("mapped", &[off, ("setAverage", "maybe")]), mapped_on()), "setAverage"),
        (scalar(bc("mapped", &[off, ("interpolationScheme", "cellPoint")]), mapped_on()), "interpolationScheme"),
        (scalar(bc("mapped", &[off, ("sampleMode", "nearestPatchFace")]), mapped_on()), "sampleMode"),
        (scalar(bc("mapped", &[off, ("samplePatch", "xMax")]), mapped_on()), "samplePatch"),
        (scalar(bc("mapped", &[off, ("sampleRegion", "solid")]), mapped_on()), "sampleRegion"),
        (scalar(bc("mapped", &[off, ("fieldName", "p")]), mapped_on()), "fieldName"),
        (
            scalar(bc("turbulentInlet", &[("referenceField", "uniform 1"), ("fluctuationScale", "0.1")]), BcInputs::default()),
            "BcInputs::turbulent_inlet",
        ),
        (scalar(bc("turbulentInlet", &[("fluctuationScale", "0.1")]), turbulent_on()), "referenceField"),
        (scalar(bc("turbulentInlet", &[("referenceField", "uniform 1")]), turbulent_on()), "fluctuationScale"),
        (
            scalar(bc("turbulentInlet", &[("referenceField", "uniform 1"), ("fluctuationScale", "-0.1")]), turbulent_on()),
            "fluctuationScale",
        ),
        (
            vector(
                bc("turbulentInlet", &[("referenceField", "uniform (1 0 0)"), ("fluctuationScale", "(0.1 0.1 0.1)"), ("alpha", "0")]),
                turbulent_on(),
            ),
            "alpha",
        ),
        (
            vector(
                bc("turbulentInlet", &[("referenceField", "uniform (1 0 0)"), ("fluctuationScale", "(0.1 0.1 0.1)"), ("alpha", "1.5")]),
                turbulent_on(),
            ),
            "alpha",
        ),
        (
            vector(
                bc("turbulentInlet", &[("referenceField", "uniform (1 0 0)"), ("fluctuationScale", "(0.1 0.1 0.1)"), ("seed", "-3")]),
                turbulent_on(),
            ),
            "seed",
        ),
    ];
    for (msg, key) in &cases {
        eprintln!("  [120.10] refusal: {msg}");
        assert!(msg.contains(key), "the refusal must name `{key}`: {msg}");
        assert!(msg.contains("120.10"), "the refusal must cite SPEC-LIT 120.10: {msg}");
    }
    eprintln!("  [120.10] {} refusals, each by name", cases.len());

    // The accepted forms build: the seed of a mapped patch is its sample,
    // a turbulentInlet's its reference plus x^0.
    let ok = scalar_boundary_with(&raw_scalar("T", with_inlet(bc("mapped", &[off])), 0.0), &hm, &tc, mapped_on())?;
    let xin = patch_of(&hm, "xMin");
    let at = hm.patches[xin].start;
    for i in 0..hm.patches[xin].size {
        let c = centre_cell(&hm, at + i, [1.25, 0.0, 0.0]);
        assert_eq!(ok.ref_value[at + i].to_bits(), tc[c].to_bits(), "face {i}: the seed is the sample");
        assert_eq!(ok.value[at + i].to_bits(), tc[c].to_bits(), "face {i}: the seed value");
    }
    Ok(())
}
