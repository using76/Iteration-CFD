// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 120.12 - the synthetic LES inflow through the public API, in
//! BOTH builds: the digital filter reproduces the prescribed Reynolds stress
//! within 5 % and each prescribed integral length (both in-plane directions
//! and, through Taylor's hypothesis, time) within 10 % ((120.12c)-(120.12f));
//! the divergence-free eddies reproduce the Reynolds stress within 5 %
//! ((120.12i)-(120.12j)) and their analytic divergence is below 1e-10 of the
//! velocity gradient ((120.12h)); the device kernels agree with the host
//! twins; and the refusals by name. The statistics run on the f64 host twins
//! and are class P, the same in both builds; the device-vs-twin rows are
//! class R (D-F32-1, the `tol` const fn below). A machine without a card
//! passes the GPU tests vacuously, the convention of `tests/f32_boussinesq.rs`.
//!
//! No GPL-licensed source was consulted.

use std::collections::BTreeMap;

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::field::synthetic::{
    build_dfsem, build_digital_filter, convect, df_values, dfsem_eddies, dfsem_gradient,
    dfsem_values, dfsem_velocity, patch_frame, DfsemRule, PatchFrame,
    SyntheticRule,
};
use ofgpu::field_ops::{advance_synthetic_inlet_vector, FieldKernels};
use ofgpu::field_setup::{setup_scalar_field_with, setup_vector_field_with, BcInputs};
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::types::DVec3;
use ofgpu::{Gpu, GpuMesh, GpuScalarField, GpuVectorField, HostMesh, Result, Scalar, Vec3};

/// `t` in the f64 build; under `single` (class R, D-F32-1)
/// `max(t, min(t*2^29, 1e-5))` - i.e. 1e-12 -> 1e-5.
const fn tol(t: f64) -> f64 {
    if cfg!(feature = "single") {
        let scaled = t * 536870912.0;
        let capped = if scaled < 1e-5 { scaled } else { 1e-5 };
        if t > capped { t } else { capped }
    } else {
        t
    }
}

/// The build's `Scalar` widened to f64.
#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    x as f64
}

fn comp(v: Vec3, k: usize) -> f64 {
    match k {
        0 => w64(v.x),
        1 => w64(v.y),
        _ => w64(v.z),
    }
}

fn entries(kv: &[(&str, &str)]) -> BTreeMap<String, String> {
    kv.iter().map(|(k, v)| ((*k).to_string(), (*v).to_string())).collect()
}

/// An `n x n` plane of unit faces at x = 0, centres `((j + 1/2) h, (k + 1/2) h)`,
/// outward normal -x: the inflow points along +x.
fn plane(n: usize, h: f64) -> PatchFrame {
    let mut c = Vec::new();
    for k in 0..n {
        for j in 0..n {
            c.push(DVec3::new(0.0, (j as f64 + 0.5) * h, (k as f64 + 0.5) * h));
        }
    }
    let nrm = vec![DVec3::new(-1.0, 0.0, 0.0); c.len()];
    let a = vec![h * h; c.len()];
    patch_frame(&c, &nrm, &a, "U", "U: boundaryField/inlet").expect("a planar patch")
}

/// The stated Reynolds stress `(xx xy xz yy yz zz)` the two statistics tests
/// prescribe, and its 5 % band `0.05 sqrt(R_ii R_jj)`.
fn check_r(what: &str, r: [f64; 6], s: &[[f64; 3]; 3], count: usize) {
    let full = [[r[0], r[1], r[2]], [r[1], r[3], r[4]], [r[2], r[4], r[5]]];
    for i in 0..3 {
        for j in i..3 {
            let got = s[i][j] / count as f64;
            let band = 0.05 * (full[i][i] * full[j][j]).sqrt();
            eprintln!(
                "  [120.12] {what} R_{i}{j}: {got:.5} against {:.5} (band {band:.4}, off by {:.2} %)",
                full[i][j],
                100.0 * (got - full[i][j]).abs() / (full[i][i] * full[j][j]).sqrt()
            );
            assert!(got.is_finite() && (got - full[i][j]).abs() <= band, "{what} R_{i}{j}: {got} against {}", full[i][j]);
        }
    }
}

// ==========================================================================
//  (120.12c)-(120.12f): the digital filter's stress and lengths
// ==========================================================================

#[test]
fn digital_filter_reproduces_r_and_l() -> Result<()> {
    let n = 40usize;
    let frame = plane(n, 1.0);
    // u: (L_x, L_y, L_z) = (2, 1.5, 2.5); v: 1.5 each; w: (1.5, 2.5, 1.5) -
    // lattice cells of h = 1, and Ubulk dt = 1 in time.
    let r = [1.0, 0.3, 0.0, 0.5, 0.0, 0.25];
    let e = entries(&[
        ("R", "(1.0 0.3 0.0 0.5 0.0 0.25)"),
        ("UMean", "(1 0 0)"),
        ("L", "(2 1.5 2.5 1.5 1.5 1.5 1.5 2.5 1.5)"),
        ("planeDivisions", "(40 40)"),
        ("seed", "11"),
    ]);
    let (rule, seed) = build_digital_filter(&e, &frame, 0, "U", "U: boundaryField/inlet")?;
    assert!(seed.iter().all(|v| *v == Vec3::new(1.0, 0.0, 0.0)), "the seed is UMean");
    let dt = 1.0f64;
    let steps = 400usize;
    let mut uu: Vec<Vec<[f64; 3]>> = Vec::with_capacity(steps);
    let mut buf = vec![Vec3::new(0.0, 0.0, 0.0); rule.size];
    let mut s = [[0.0f64; 3]; 3];
    for step in 1..=steps {
        df_values(&rule, step as i64, dt, &mut buf)?;
        let plane: Vec<[f64; 3]> = buf
            .iter()
            .map(|v| [comp(*v, 0) - 1.0, comp(*v, 1), comp(*v, 2)])
            .collect();
        for u in &plane {
            for i in 0..3 {
                for j in 0..3 {
                    s[i][j] += u[i] * u[j];
                }
            }
        }
        uu.push(plane);
    }
    check_r("digital filter", r, &s, steps * rule.size);

    // (120.12f): the integral length of u (component 0) along e_y, e_z and
    // time, and of w (component 2), from the sample autocorrelation summed
    // over the filter's support, 2N lags.
    let at = |j: usize, k: usize| k * n + j;
    let length = |c: usize, dir: usize, lag_max: usize| -> f64 {
        let mut var = 0.0f64;
        let mut cnt = 0usize;
        for p in &uu {
            for u in p {
                var += u[c] * u[c];
                cnt += 1;
            }
        }
        var /= cnt as f64;
        let mut sum = 0.5f64;
        for lag in 1..=lag_max {
            let (mut acc, mut m) = (0.0f64, 0usize);
            match dir {
                0 => {
                    for t in 0..steps - lag {
                        for i in 0..n * n {
                            acc += uu[t][i][c] * uu[t + lag][i][c];
                            m += 1;
                        }
                    }
                }
                1 => {
                    for p in &uu {
                        for k in 0..n {
                            for j in 0..n - lag {
                                acc += p[at(j, k)][c] * p[at(j + lag, k)][c];
                                m += 1;
                            }
                        }
                    }
                }
                _ => {
                    for p in &uu {
                        for k in 0..n - lag {
                            for j in 0..n {
                                acc += p[at(j, k)][c] * p[at(j, k + lag)][c];
                                m += 1;
                            }
                        }
                    }
                }
            }
            sum += acc / m as f64 / var;
        }
        sum
    };
    let lens: [[f64; 3]; 3] = [[2.0, 1.5, 2.5], [1.5, 1.5, 1.5], [1.5, 2.5, 1.5]];
    for c in [0usize, 2] {
        for dir in 0..3 {
            let want = lens[c][dir];
            let lag_max = 2 * (2.0f64 * want).ceil() as usize;
            let got = length(c, dir, lag_max);
            eprintln!(
                "  [120.12] digital filter: component {c}, direction {dir}: L {got:.4} against {want} ({:.2} %)",
                100.0 * (got - want).abs() / want
            );
            assert!((got - want).abs() <= 0.10 * want, "component {c} direction {dir}: L {got} against {want}");
        }
    }
    Ok(())
}

// ==========================================================================
//  (120.12g)-(120.12k): the eddies' stress and divergence
// ==========================================================================

fn dfsem_rule(seed: &str) -> Result<DfsemRule> {
    let frame = plane(32, 1.0 / 16.0);
    let e = entries(&[
        ("R", "(0.8 0.1 0.05 0.6 0.05 0.5)"),
        ("UMean", "(2 0 0)"),
        ("Ubulk", "1"),
        ("sigma", "0.25"),
        ("seed", seed),
    ]);
    Ok(build_dfsem(&e, &frame, 0, "U", "U: boundaryField/inlet")?.0)
}

#[test]
fn dfsem_reproduces_r() -> Result<()> {
    let rule = dfsem_rule("5")?;
    let r = [0.8, 0.1, 0.05, 0.6, 0.05, 0.5];
    // Ubulk dt = 0.05: an eddy takes 10 steps to cross the plane.
    let (dt, steps) = (0.05f64, 6000usize);
    let mut x = 0.0f64;
    let mut buf = vec![Vec3::new(0.0, 0.0, 0.0); rule.size];
    let mut s = [[0.0f64; 3]; 3];
    for _ in 0..steps {
        x = convect(x, rule.ubulk, dt);
        dfsem_values(&rule, x, &mut buf);
        for v in &buf {
            let u = [comp(*v, 0) - 2.0, comp(*v, 1), comp(*v, 2)];
            for i in 0..3 {
                for j in 0..3 {
                    s[i][j] += u[i] * u[j];
                }
            }
        }
    }
    eprintln!("  [120.12] DFSEM: {} eddies, sigma {}, scale {:.5}", rule.n_eddy, rule.sigma, rule.scale);
    check_r("DFSEM", r, &s, steps * rule.size);
    Ok(())
}

#[test]
fn dfsem_is_divergence_free() -> Result<()> {
    let rule = dfsem_rule("9")?;
    let mut worst = 0.0f64;
    let mut fd_worst = 0.0f64;
    let mut scale_seen = 0.0f64;
    let mut points = 0usize;
    for t in 0..20 {
        let x = 0.37 * t as f64;
        let eddies = dfsem_eddies(&rule, x);
        for i in 0..100 {
            // Points on and off the plane, inside the box.
            let f = (i * 37 + t * 11) % rule.size;
            let p = [
                rule.rel[f][0] + 0.1 * ((i % 7) as f64 - 3.0) * rule.sigma / 3.0,
                rule.rel[f][1] + 0.013 * (i % 5) as f64,
                rule.rel[f][2] - 0.011 * (i % 3) as f64,
            ];
            let g = dfsem_gradient(&rule, &eddies, p);
            let gmax = g.iter().flatten().fold(0.0f64, |a, v| a.max(v.abs()));
            if gmax == 0.0 {
                continue;
            }
            scale_seen = scale_seen.max(gmax);
            let div = g[0][0] + g[1][1] + g[2][2];
            worst = worst.max(div.abs() / gmax);
            // The gradient is the velocity's: central differences, h = 1e-6 sigma.
            let h = 1e-6 * rule.sigma;
            for m in 0..3 {
                let (mut a, mut b) = (p, p);
                a[m] += h;
                b[m] -= h;
                let (ua, ub) = (dfsem_velocity(&rule, &eddies, a), dfsem_velocity(&rule, &eddies, b));
                for k in 0..3 {
                    let fd = (ua[k] - ub[k]) / (2.0 * h);
                    fd_worst = fd_worst.max((fd - g[k][m]).abs() / gmax);
                }
            }
            points += 1;
        }
    }
    eprintln!(
        "  [120.12] DFSEM: |div u'| / max|grad u'| <= {worst:.3e} over {points} points; \
         gradient vs central differences {fd_worst:.3e}"
    );
    assert!(points >= 1500, "{points} points with a non-zero gradient");
    assert!(scale_seen > 0.0);
    assert!(worst <= 1e-10, "DFSEM divergence {worst:e}");
    assert!(fd_worst <= 1e-5, "the analytic gradient is not the velocity's: {fd_worst:e}");
    Ok(())
}

// ==========================================================================
//  The device kernels against the host twins, and the refusals
// ==========================================================================

fn box_spec(n: [usize; 3]) -> BlockSpec {
    let axis = |n: usize| GradedAxis {
        lo: 0.0,
        hi: 1.0,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    BlockSpec {
        x: axis(n[0]),
        y: axis(n[1]),
        z: axis(n[2]),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["patch", "patch", "patch", "patch", "patch", "patch"].map(String::from),
        cyclic: Vec::new(),
    }
}

fn ps(ty: &str) -> PatchFieldSpec {
    PatchFieldSpec {
        type_name: ty.into(),
        ..Default::default()
    }
}

fn bc(name: &str, extra: &[(&str, &str)]) -> PatchFieldSpec {
    let mut s = ps(name);
    for (k, v) in extra {
        s.extra.insert((*k).into(), (*v).into());
    }
    s
}

fn raw_u(inlet: PatchFieldSpec) -> RawVectorField {
    let mut raw = RawVectorField {
        name: "U".into(),
        ..Default::default()
    };
    raw.boundary.insert("xMin".into(), inlet);
    for p in ["xMax", "yMin", "yMax", "zMin", "zMax"] {
        raw.boundary.insert(p.into(), ps("zeroGradient"));
    }
    raw.internal = vec![Vec3::new(1.0, 0.0, 0.0)];
    raw
}

fn on() -> BcInputs<'static> {
    BcInputs {
        turbulent_inlet: true,
        ..BcInputs::default()
    }
}

fn df_spec() -> PatchFieldSpec {
    bc(
        "turbulentDigitalFilterInlet",
        &[
            ("R", "(0.04 0.01 0 0.02 0 0.01)"),
            ("UMean", "(1 0 0)"),
            ("L", "(0.2 0.25 0.2 0.1 0.1 0.1 0.1 0.2 0.1)"),
            ("planeDivisions", "(8 8)"),
            ("seed", "3"),
        ],
    )
}

fn dfsem_spec(r: &str) -> PatchFieldSpec {
    bc(
        "turbulentDFSEMInlet",
        &[("R", r), ("UMean", "(1 0 0)"), ("sigma", "0.2"), ("seed", "4")],
    )
}

fn hm() -> Result<HostMesh> {
    blockgen::build_mesh(&box_spec([1, 8, 8]))
}

#[test]
fn synthetic_device_matches_host() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = hm()?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let k = FieldKernels::new(&gpu)?;
    let dt: Scalar = 0.025;
    let dt64 = w64(dt);
    for spec in [df_spec(), dfsem_spec("(0.03 0.005 0 0.02 0 0.02)")] {
        let name = spec.type_name.clone();
        let mut u = GpuVectorField::zeros(&gpu, &gm, "U")?;
        setup_vector_field_with(&gpu, &mut u, &raw_u(spec), &hm, on())?;
        assert_eq!(u.synthetic.len(), 1, "{name}: one rule");
        let rule = u.synthetic[0].rule.clone();
        let (start, size) = (rule.start(), rule.size());
        assert_eq!(size, 64, "{name}: the xMin faces");
        let mut host = vec![Vec3::new(0.0, 0.0, 0.0); hm.n_boundary_faces];
        // The seed: UMean for the filter, (120.12g) at X = 0 for the eddies.
        let seed = gpu.download(&u.ref_value)?;
        match &rule {
            SyntheticRule::DigitalFilter(_) => {
                for i in 0..size {
                    assert_eq!(seed[start + i], Vec3::new(1.0, 0.0, 0.0), "{name}: the seed is UMean");
                }
            }
            SyntheticRule::Dfsem(r) => {
                dfsem_values(r, 0.0, &mut host);
                for i in 0..size {
                    assert_eq!(seed[start + i], host[start + i], "{name}: the seed is the X = 0 field");
                }
            }
        }
        let (mut worst, mut umax) = (0.0f64, 0.0f64);
        let mut x = 0.0f64;
        for step in 1..=6i64 {
            advance_synthetic_inlet_vector(&gpu, &k, &mut u, dt)?;
            match &rule {
                SyntheticRule::DigitalFilter(r) => df_values(r, step, dt64, &mut host)?,
                SyntheticRule::Dfsem(r) => {
                    x = convect(x, r.ubulk, dt64);
                    dfsem_values(r, x, &mut host);
                }
            }
            let dev = gpu.download(&u.ref_value)?;
            for i in 0..size {
                for c in 0..3 {
                    let d = comp(dev[start + i], c);
                    let h = comp(host[start + i], c);
                    assert!(d.is_finite(), "{name}: face {i} is non-finite");
                    worst = worst.max((d - h).abs());
                    umax = umax.max((h - if c == 0 { 1.0 } else { 0.0 }).abs());
                }
            }
        }
        eprintln!("  [120.12] {name}: device vs host twin over 6 steps {:.3e} of max|u'| {umax:.4}", worst / umax);
        assert!(umax > 0.0, "{name}: no fluctuation");
        assert!(worst <= tol(1e-12) * umax, "{name}: device vs twin {worst:e}");
    }
    Ok(())
}

#[test]
fn synthetic_refusals_name_their_reason() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = hm()?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let msg = |r: Result<()>| -> String {
        match r {
            Ok(()) => panic!("expected a refusal"),
            Err(e) => e.to_string(),
        }
    };
    // No opt-in: refused, naming the entry point (BC-14's rule for turbulentInlet).
    for spec in [df_spec(), dfsem_spec("(0.03 0.005 0 0.02 0 0.02)")] {
        let mut u = GpuVectorField::zeros(&gpu, &gm, "U")?;
        let m = msg(setup_vector_field_with(&gpu, &mut u, &raw_u(spec), &hm, BcInputs::default()));
        assert!(m.contains("advance_synthetic_inlet_vector") && m.contains("120.12"), "{m}");
    }
    // A scalar field: a vector condition.
    let mut t = GpuScalarField::zeros(&gpu, &gm, "T")?;
    let mut raw = RawScalarField {
        name: "T".into(),
        ..Default::default()
    };
    raw.boundary.insert("xMin".into(), df_spec());
    for p in ["xMax", "yMin", "yMax", "zMin", "zMax"] {
        raw.boundary.insert(p.into(), ps("zeroGradient"));
    }
    raw.internal = vec![300.0];
    let m = msg(setup_scalar_field_with(&gpu, &mut t, &raw, &hm, on()));
    assert!(m.contains("vector condition"), "{m}");
    // (120.12j): one principal stress above the sum of the other two.
    let mut u = GpuVectorField::zeros(&gpu, &gm, "U")?;
    let m = msg(setup_vector_field_with(&gpu, &mut u, &raw_u(dfsem_spec("(1 0 0 0.1 0 0.1)")), &hm, on()));
    assert!(m.contains("not realisable") && m.contains("120.12j"), "{m}");
    // The filter takes that stress, but not one that is no stress at all.
    let mut s = df_spec();
    s.extra.insert("R".into(), "(1 0 0 0.1 0 0.1)".into());
    setup_vector_field_with(&gpu, &mut u, &raw_u(s.clone()), &hm, on())?;
    s.extra.insert("R".into(), "(1 2 0 1 0 1)".into());
    let m = msg(setup_vector_field_with(&gpu, &mut u, &raw_u(s.clone()), &hm, on()));
    assert!(m.contains("not positive semi-definite"), "{m}");
    // Missing entries, by name.
    s.extra.remove("L");
    s.extra.insert("R".into(), "(1 0 0 1 0 1)".into());
    let m = msg(setup_vector_field_with(&gpu, &mut u, &raw_u(s), &hm, on()));
    assert!(m.contains("`L`"), "{m}");
    // A bulk velocity out of the domain.
    let mut d = dfsem_spec("(0.03 0.005 0 0.02 0 0.02)");
    d.extra.insert("UMean".into(), "(-1 0 0)".into());
    let m = msg(setup_vector_field_with(&gpu, &mut u, &raw_u(d), &hm, on()));
    assert!(m.contains("Ubulk") && m.contains("not positive"), "{m}");
    Ok(())
}
