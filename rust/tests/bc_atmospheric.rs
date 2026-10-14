// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 120.9 - the atmospheric family through the public API, in BOTH
//! builds: the four Richards & Hoxey (1993) inlet profiles against
//! (120.9a)-(120.9e), `nutkAtmRoughWallFunction` against its host twin and
//! against the wall shear `u*^2` it is written to hold on the profile
//! ((120.9g)), `nutUBlendedWallFunction` against its host twin and its two
//! limits ((120.9h)), the refusals by name, and the gate: a 5 km fetch of
//! the neutral surface layer through `Simple` and k-epsilon whose outlet
//! keeps the inlet profile (horizontal homogeneity, Hargreaves & Wright
//! 2007). The twin tolerances follow docs/17 §5.1 class R, D-F32-1 - the
//! f64 tolerance in the f64 build, `max(t, min(t*2^29, 1e-5))` under
//! `single` - which `tol` encodes; the fetch's 3 %/10 %/3 % are class P,
//! the same in both builds. Everything is compared in f64. A machine
//! without a card passes vacuously, the convention of `tests/bc_inlets.rs`.
//!
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::field_setup::{
    setup_scalar_field_with, setup_vector_field_with, BcInputs, NutFamily, NutRoughness,
    WallFaces,
};
use ofgpu::io::case::{LinearSolverKind, Preconditioner, TurbulenceControls, WallFunctionCoeffs};
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::models::{KEpsilon, KEpsilonCoeffs};
use ofgpu::momentum::{BuoyancyCoeffs, MomentumControls};
use ofgpu::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use ofgpu::simple::{Simple, SimpleControls};
use ofgpu::wallfunctions::{nut_wall_atm_rough, nut_wall_blended, u_tau_blended, u_tau_log, WallData};
use ofgpu::{BcKind, Gpu, GpuMesh, GpuScalarField, GpuVectorField, HostMesh, Result, Scalar, Vec3};

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

/// A checked term: a non-finite value panics here, naming itself.
fn fin(what: &str, x: f64) -> f64 {
    assert!(x.is_finite(), "{what} is non-finite: {x:e}");
    x
}

fn axis(lo: Scalar, hi: Scalar, n: usize, expansion: Scalar) -> GradedAxis {
    GradedAxis { lo, hi, n, expansion, two_sided: false }
}

/// A block whose patches the caller types.
fn block(x: GradedAxis, y: GradedAxis, z: GradedAxis, types: [&str; 6]) -> BlockSpec {
    BlockSpec {
        x,
        y,
        z,
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: types.map(String::from),
        cyclic: Vec::new(),
    }
}

fn patch_of(hm: &HostMesh, name: &str) -> usize {
    hm.patches
        .iter()
        .position(|p| p.name == name)
        .unwrap_or_else(|| panic!("no patch {name}"))
}

fn ps(ty: &str) -> PatchFieldSpec {
    PatchFieldSpec {
        type_name: ty.into(),
        ..Default::default()
    }
}

/// A patch entry with its extra entries.
fn bc(ty: &str, extra: &[(&str, &str)]) -> PatchFieldSpec {
    let mut s = ps(ty);
    for (k, v) in extra {
        s.extra.insert((*k).into(), (*v).into());
    }
    s
}

fn raw_s(name: &str, boundary: Vec<(&str, PatchFieldSpec)>, internal: Scalar) -> RawScalarField {
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

fn raw_v(boundary: Vec<(&str, PatchFieldSpec)>, internal: Vec3) -> RawVectorField {
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

/// One profile's inputs, the f64 answer key of (120.9a)-(120.9e).
#[derive(Clone, Copy)]
struct Abl {
    u_ref: f64,
    z_ref: f64,
    z0: f64,
    z_ground: f64,
    kappa: f64,
    cmu: f64,
}

impl Abl {
    /// (120.9a).
    fn u_star(&self) -> f64 {
        self.kappa * self.u_ref / (self.z_ref / self.z0).ln_1p()
    }
    /// (120.9b)-(120.9e) at height `x_z` (world z), `z = max(x_z - zGround, 0)`.
    fn at(&self, x_z: f64) -> (f64, f64, f64, f64) {
        let us = self.u_star();
        let z = (x_z - self.z_ground).max(0.0);
        let u = us / self.kappa * (z / self.z0).ln_1p();
        let k = us * us / self.cmu.sqrt();
        let eps = us * us * us / (self.kappa * (z + self.z0));
        let omega = us / (self.kappa * self.cmu.sqrt() * (z + self.z0));
        (u, k, eps, omega)
    }
    /// The patch entries of the family, flow along +x, z up.
    fn entries(&self) -> Vec<(&'static str, String)> {
        let mut v = vec![
            ("flowDir", "(1 0 0)".to_string()),
            ("zDir", "(0 0 1)".to_string()),
            ("Uref", format!("{}", self.u_ref)),
            ("Zref", format!("{}", self.z_ref)),
            ("z0", format!("uniform {}", self.z0)),
        ];
        if self.z_ground != 0.0 {
            v.push(("zGround", format!("uniform {}", self.z_ground)));
        }
        if self.kappa != 0.41 {
            v.push(("kappa", format!("{}", self.kappa)));
        }
        if self.cmu != 0.09 {
            v.push(("Cmu", format!("{}", self.cmu)));
        }
        v
    }
    fn spec(&self, ty: &str) -> PatchFieldSpec {
        let mut s = ps(ty);
        for (k, v) in self.entries() {
            s.extra.insert(k.into(), v);
        }
        s
    }
}

// ==========================================================================
//  (120.9a)-(120.9e) - the four inlet profiles
// ==========================================================================

#[test]
fn the_inlet_profiles_are_richards_and_hoxey() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    // 2 x 1 x 12 cells over [0,2] x [0,1] x [0,60], graded towards the
    // ground; the profile sits on xMin.
    let hm = blockgen::build_mesh(&block(
        axis(0.0, 2.0, 2, 1.0),
        axis(0.0, 1.0, 1, 1.0),
        axis(0.0, 60.0, 12, 8.0),
        ["patch", "patch", "empty", "empty", "wall", "patch"],
    ))?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let abl = Abl { u_ref: 8.0, z_ref: 20.0, z0: 0.05, z_ground: -2.0, kappa: 0.4, cmu: 0.085 };
    let xin = patch_of(&hm, "xMin");
    let (at, n) = (hm.patches[xin].start, hm.patches[xin].size);
    let cf = hm.geom64();

    let others = |t: &str| -> Vec<(&'static str, PatchFieldSpec)> {
        vec![
            ("xMax", ps("zeroGradient")),
            ("yMin", ps("empty")),
            ("yMax", ps("empty")),
            ("zMin", ps(t)),
            ("zMax", ps("zeroGradient")),
        ]
    };

    let mut u = GpuVectorField::zeros(&gpu, &gm, "U")?;
    let mut b = others("noSlip");
    b.push(("xMin", abl.spec("atmBoundaryLayerInletVelocity")));
    setup_vector_field_with(&gpu, &mut u, &raw_v(b, Vec3::ZERO), &hm, BcInputs::default())?;
    let ub = gpu.download(&u.bf)?;
    let kinds = gpu.download(&u.bc_kind)?;
    let mut worst = 0.0f64;
    for i in 0..n {
        let f = at + i;
        assert_eq!(kinds[f], BcKind::FixedValue as i32, "face {i}: the kind");
        let (want, _, _, _) = abl.at(cf.b_cf[f].z);
        let got = fin("U_b.x", w64(ub[f].x));
        worst = worst.max((got - want).abs() / want.abs().max(1e-300));
        assert!(w64(ub[f].y) == 0.0 && w64(ub[f].z) == 0.0, "face {i}: off-axis {:?}", ub[f]);
    }
    eprintln!("  [120.9] U profile: worst relative {worst:.3e} over {n} faces");
    assert!(worst <= tol(1e-14), "U profile: {worst:e}");

    for (ty, which) in [
        ("atmBoundaryLayerInletK", 1usize),
        ("atmBoundaryLayerInletEpsilon", 2),
        ("atmBoundaryLayerInletOmega", 3),
    ] {
        let mut s = GpuScalarField::zeros(&gpu, &gm, "psi")?;
        let mut b = others("zeroGradient");
        b.push(("xMin", abl.spec(ty)));
        setup_scalar_field_with(&gpu, &mut s, &raw_s("psi", b, 0.0), &hm, BcInputs::default())?;
        let sb = gpu.download(&s.bf)?;
        let mut worst = 0.0f64;
        for i in 0..n {
            let f = at + i;
            let (_, k, e, o) = abl.at(cf.b_cf[f].z);
            let want = [0.0, k, e, o][which];
            let got = fin(ty, w64(sb[f]));
            worst = worst.max((got - want).abs() / want.abs());
        }
        eprintln!("  [120.9] {ty}: worst relative {worst:.3e}");
        assert!(worst <= tol(1e-14), "{ty}: {worst:e}");
    }

    // (120.9a): the reference point is ON the profile, U(Zref + zGround) = Uref.
    let (u_at_ref, _, _, _) = abl.at(abl.z_ref + abl.z_ground);
    assert!((u_at_ref - abl.u_ref).abs() <= 1e-12 * abl.u_ref, "U(Zref) = {u_at_ref}");
    Ok(())
}

// ==========================================================================
//  (120.9g), (120.9h) - the two wall functions against their host twins
// ==========================================================================

/// 8 x 1 x 2 cells over [0,8] x [0,1] x [0,0.2], the ground `zMin` a wall:
/// eight ground faces at y = 0.05.
fn ground_mesh(gpu: &Gpu) -> Result<(HostMesh, GpuMesh)> {
    let hm = blockgen::build_mesh(&block(
        axis(0.0, 8.0, 8, 1.0),
        axis(0.0, 1.0, 1, 1.0),
        axis(0.0, 0.2, 2, 1.0),
        ["patch", "patch", "empty", "empty", "wall", "patch"],
    ))?;
    let gm = GpuMesh::upload(gpu, &hm)?;
    Ok((hm, gm))
}

/// `nut` with `ground` on zMin and `calculated` elsewhere.
fn raw_nut(ground: PatchFieldSpec) -> RawScalarField {
    raw_s(
        "nut",
        vec![
            ("xMin", ps("calculated")),
            ("xMax", ps("calculated")),
            ("yMin", ps("empty")),
            ("yMax", ps("empty")),
            ("zMin", ground),
            ("zMax", ps("calculated")),
        ],
        0.0,
    )
}

/// The device `nu_t,w` of `wfNutWall` on every ground face, for the given
/// cell `k` and cell `U` (the ground is at rest).
fn device_nut(
    gpu: &Gpu,
    hm: &HostMesh,
    gm: &GpuMesh,
    nut: &RawScalarField,
    k: &[Scalar],
    u: &[Vec3],
    nu: Scalar,
) -> Result<Vec<Scalar>> {
    let ground = patch_of(hm, "zMin");
    let p = &hm.patches[ground];
    let mut flags = vec![false; hm.n_boundary_faces];
    for i in 0..p.size {
        flags[p.start + i] = true;
    }
    let faces = WallFaces { constrained_cells: flags.clone(), nut: flags };
    let rough = NutRoughness::from_case(Some(nut), hm)?;
    let wd = WallData::build(gpu, hm, &faces, &rough)?;
    assert_eq!(wd.n_nut_faces, p.size);
    let k_dev = gpu.upload(k)?;
    let mut uf = GpuVectorField::zeros(gpu, gm, "U")?;
    gpu.write(&mut uf.f, u)?;
    let mut nut_bf = gpu.zeros::<Scalar>(hm.n_boundary_faces)?;
    wd.update_nut(gpu, &mut nut_bf, &k_dev, &uf, gm, &WallFunctionCoeffs::default(), nu, 1e-15)?;
    gpu.sync()?;
    let all = gpu.download(&nut_bf)?;
    Ok(all[p.start..p.start + p.size].to_vec())
}

#[test]
fn nutk_atm_rough_holds_the_profile_shear() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let (hm, gm) = ground_mesh(&gpu)?;
    let wc = WallFunctionCoeffs::default();
    let nu: Scalar = 1.5e-5;
    let cmu25 = wc.cmu.powf(0.25);
    // One z0 a face, from far below the first cell (y = 0.05) to far above
    // it, where ln(1 + y/z0) is small and ln_1p keeps its digits.
    let z0s: [f64; 8] = [1e-4, 1e-3, 0.01, 0.05, 0.2, 1.0, 5.0, 30.0];
    let list = z0s.iter().map(|z| format!("{z}")).collect::<Vec<_>>().join(" ");
    let nut = raw_nut(bc("nutkAtmRoughWallFunction", &[("z0", &format!("nonuniform List<scalar> 8({list})"))]));
    let r = NutRoughness::from_case(Some(&nut), &hm)?;
    let ground = patch_of(&hm, "zMin");
    let p = &hm.patches[ground];
    assert!((0..p.size).all(|i| r.family[p.start + i] == NutFamily::AtmRough));

    // k on the Richards & Hoxey profile, u* = 0.5: k = u*^2/sqrt(C_mu).
    let u_star = 0.5f64;
    let k_val = (u_star * u_star / (wc.cmu as f64).sqrt()) as Scalar;
    let k = vec![k_val; hm.n_cells];
    let u = vec![Vec3::ZERO; hm.n_cells];
    let dev = device_nut(&gpu, &hm, &gm, &nut, &k, &u, nu)?;

    let (mut worst_twin, mut worst_shear) = (0.0f64, 0.0f64);
    for i in 0..p.size {
        let bf = p.start + i;
        let y = hm.b_y[bf];
        let z0 = z0s[i] as Scalar;
        let twin = nut_wall_atm_rough(k_val, y, nu, wc.kappa, cmu25, z0);
        let d = fin("nut_w", w64(dev[i]));
        assert!(w64(twin) > 0.0, "face {i}: nut_w = {}", w64(twin));
        worst_twin = worst_twin.max((d - w64(twin)).abs() / w64(twin));
        // (120.9g)'s purpose: (nu + nu_t,w) U(y)/y = u*^2 on the profile,
        // with U(y) = (u*/kappa) ln(1 + y/z0) and the u* k implies.
        let us = (wc.cmu as f64).powf(0.25) * (k_val as f64).sqrt();
        let uy = us / (wc.kappa as f64) * (w64(y) / z0s[i]).ln_1p();
        let tau = (w64(nu) + d) * uy / w64(y);
        worst_shear = worst_shear.max((tau - us * us).abs() / (us * us));
    }
    eprintln!(
        "  [120.9] nutkAtmRough: device vs twin {worst_twin:.3e}, wall shear vs u*^2 {worst_shear:.3e}"
    );
    assert!(worst_twin <= tol(1e-12), "device vs twin {worst_twin:e}");
    assert!(worst_shear <= tol(1e-12), "wall shear vs u*^2 {worst_shear:e}");
    Ok(())
}

#[test]
fn nutu_blended_meets_its_twin_and_its_limits() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let (hm, gm) = ground_mesh(&gpu)?;
    let wc = WallFunctionCoeffs::default();
    let nu: Scalar = 1.5e-5;
    let ground = patch_of(&hm, "zMin");
    let p = hm.patches[ground].clone();
    let speeds: Vec<Scalar> = (0..p.size).map(|i| (10.0f64).powi(i as i32 - 6) as Scalar).collect();
    // The wall-parallel cell speed of each ground face's cell.
    let mut u = vec![Vec3::ZERO; hm.n_cells];
    for i in 0..p.size {
        let c = hm.b_face_cells[p.start + i] as usize;
        u[c] = Vec3::new(speeds[i], 0.0, 0.0);
    }
    let k = vec![1.0 as Scalar; hm.n_cells];

    for n_blend in [4.0 as Scalar, 3.0] {
        let nut = if n_blend == 4.0 {
            raw_nut(ps("nutUBlendedWallFunction"))
        } else {
            raw_nut(bc("nutUBlendedWallFunction", &[("n", "3")]))
        };
        let dev = device_nut(&gpu, &hm, &gm, &nut, &k, &u, nu)?;
        let mut worst = 0.0f64;
        for i in 0..p.size {
            let bf = p.start + i;
            let y = hm.b_y[bf];
            let twin = nut_wall_blended(speeds[i], y, nu, wc.kappa, wc.e, n_blend);
            let d = fin("nut_w", w64(dev[i]));
            worst = worst.max((d - w64(twin)).abs() / w64(twin).max(w64(nu)));
        }
        eprintln!("  [120.9] nutUBlended n = {n_blend}: device vs twin {worst:.3e}");
        assert!(worst <= tol(1e-12), "n = {n_blend}: device vs twin {worst:e}");
    }

    // The limits, on the host twin (the device was just held to it).
    let y = hm.b_y[p.start];
    let (kappa, e) = (wc.kappa, wc.e);
    // Viscous: Re_y = 1e-6 * 0.05/1.5e-5 = 3.3e-3, so u_tau -> u_v and
    // nu_t,w -> nu ((1 + (kappa/E)^2)^{1/2} - 1) ~ 9e-4 nu.
    let slow = nut_wall_blended(1e-6, y, nu, kappa, e, 4.0);
    eprintln!("  [120.9] nutUBlended viscous limit: nu_t,w/nu = {:.3e}", w64(slow) / w64(nu));
    assert!(w64(slow) <= 1e-3 * w64(nu), "viscous limit nu_t,w = {}", w64(slow));
    // Log: Re_y = 3.3e4, the blend is the log-law root to 1e-4.
    let ul = u_tau_log(10.0, y, nu, kappa, e);
    let ub = u_tau_blended(10.0, y, nu, kappa, e, 4.0);
    let g = w64(ul) * (w64(e) * w64(y) * w64(ul) / w64(nu)).ln_1p() - w64(kappa) * 10.0;
    eprintln!(
        "  [120.9] nutUBlended log limit: u_l = {:.6}, blend/u_l - 1 = {:.3e}, |G(u_l)| = {:.3e}",
        w64(ul),
        w64(ub) / w64(ul) - 1.0,
        g.abs()
    );
    assert!(g.abs() <= tol(1e-12) * w64(kappa) * 10.0, "G(u_l) = {g:e}");
    assert!((w64(ub) / w64(ul) - 1.0).abs() <= 1e-4, "blend {} vs u_l {}", w64(ub), w64(ul));
    Ok(())
}

// ==========================================================================
//  Refusals by name
// ==========================================================================

#[test]
fn atmospheric_refusals_are_by_name() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let (hm, gm) = ground_mesh(&gpu)?;
    let abl = Abl { u_ref: 10.0, z_ref: 10.0, z0: 0.1, z_ground: 0.0, kappa: 0.41, cmu: 0.09 };
    let rest = || -> Vec<(&'static str, PatchFieldSpec)> {
        vec![
            ("xMax", ps("zeroGradient")),
            ("yMin", ps("empty")),
            ("yMax", ps("empty")),
            ("zMin", ps("zeroGradient")),
            ("zMax", ps("zeroGradient")),
        ]
    };
    let scalar_err = |s: PatchFieldSpec| -> String {
        let mut f = GpuScalarField::zeros(&gpu, &gm, "k").expect("field");
        let mut b = rest();
        b.push(("xMin", s));
        match setup_scalar_field_with(&gpu, &mut f, &raw_s("k", b, 0.0), &hm, BcInputs::default()) {
            Ok(()) => panic!("accepted"),
            Err(e) => format!("{e}"),
        }
    };
    let vector_err = |s: PatchFieldSpec| -> String {
        let mut f = GpuVectorField::zeros(&gpu, &gm, "U").expect("field");
        let mut b = rest();
        b.push(("xMin", s));
        match setup_vector_field_with(&gpu, &mut f, &raw_v(b, Vec3::ZERO), &hm, BcInputs::default()) {
            Ok(()) => panic!("accepted"),
            Err(e) => format!("{e}"),
        }
    };
    let with = |ty: &str, key: &str, val: Option<&str>| -> PatchFieldSpec {
        let mut s = abl.spec(ty);
        match val {
            Some(v) => {
                s.extra.insert(key.into(), v.into());
            }
            None => {
                s.extra.remove(key);
            }
        }
        s
    };

    let cases: Vec<(String, &str)> = vec![
        (scalar_err(with("atmBoundaryLayerInletK", "z0", None)), "z0"),
        (scalar_err(with("atmBoundaryLayerInletK", "z0", Some("uniform -0.1"))), "positive"),
        (scalar_err(with("atmBoundaryLayerInletEpsilon", "Uref", None)), "Uref"),
        (scalar_err(with("atmBoundaryLayerInletOmega", "Zref", Some("0"))), "Zref"),
        (scalar_err(with("atmBoundaryLayerInletK", "zDir", Some("(0 0 0)"))), "zDir"),
        (scalar_err(with("atmBoundaryLayerInletK", "d", Some("uniform 1"))), "`d`"),
        (scalar_err(with("atmBoundaryLayerInletK", "C1", Some("0.1"))), "`C1`"),
        (
            scalar_err(with("atmBoundaryLayerInletEpsilon", "Uref", Some("table ((0 1) (10 2))"))),
            "time-varying",
        ),
        (scalar_err(abl.spec("atmBoundaryLayerInletVelocity")), "vector condition"),
        (vector_err(abl.spec("atmBoundaryLayerInletK")), "scalar condition"),
        (vector_err(with("atmBoundaryLayerInletVelocity", "flowDir", None)), "flowDir"),
        (vector_err(with("atmBoundaryLayerInletVelocity", "flowDir", Some("(0 0 0)"))), "flowDir"),
    ];
    for (msg, want) in &cases {
        assert!(msg.contains(want), "refusal does not name {want:?}: {msg}");
        assert!(msg.contains("120.9") || *want == "flowDir" || want.contains("condition"), "{msg}");
    }
    eprintln!("  [120.9] {} refusals by name", cases.len());
    Ok(())
}

// ==========================================================================
//  The gate - a 5 km fetch of the neutral surface layer
// ==========================================================================

/// The fetch: 50 x 1 x 24 cells over [0, 5000] x [0, 10] x [0, 500] m, the
/// vertical graded 50:1 towards the ground (first cell 1.59 m). xMin and
/// zMax carry the profile (the top's fixed values hold the shear `u*^2`
/// that drives the layer), xMax is the outlet, zMin the rough ground with
/// `nutkAtmRoughWallFunction`, y empty. Richards & Hoxey's consistent
/// `sigma_eps` (120.9f). The outlet column's U and k against the inlet
/// profile, and the ground's u* against (120.9a) - class P, the same bands
/// in both builds.
#[test]
fn a_five_km_fetch_keeps_the_inlet_profile() -> Result<()> {
    let Some(gpu) = Gpu::new(0).ok() else {
        return Ok(());
    };
    let hm = blockgen::build_mesh(&block(
        axis(0.0, 5000.0, 50, 1.0),
        axis(0.0, 10.0, 1, 1.0),
        axis(0.0, 500.0, 24, 50.0),
        ["patch", "patch", "empty", "empty", "wall", "patch"],
    ))?;
    let gm = GpuMesh::upload(&gpu, &hm)?;
    let abl = Abl { u_ref: 10.0, z_ref: 10.0, z0: 0.01, z_ground: 0.0, kappa: 0.41, cmu: 0.09 };
    let u_star = abl.u_star();
    let nu: Scalar = 1.5e-5;
    let kc = KEpsilonCoeffs::default();
    // (120.9f): the sigma_eps under which (120.9b)-(120.9d) solve the
    // epsilon equation exactly (Richards & Hoxey 1993).
    let sigma_eps =
        (abl.kappa * abl.kappa / ((kc.c2 as f64 - kc.c1 as f64) * (kc.cmu as f64).sqrt())) as Scalar;
    let coeffs = KEpsilonCoeffs { sigma_eps, ..kc };
    let wc = WallFunctionCoeffs::default();
    assert_eq!(wc.cmu, coeffs.cmu, "one C_mu for the model and the wall (S15.6)");

    let solver = ofgpu::solver::SolverControls {
        tolerance: 1e-12,
        rel_tol: 0.01,
        max_iter: 200,
        ..ofgpu::solver::SolverControls::default()
    };
    let ctrl = SimpleControls {
        momentum: MomentumControls { nu, u_solver: solver, ..MomentumControls::default() },
        // PCG with DIC on the symmetric pressure matrix. PBiCGStab with the
        // diagonal preconditioner diverges on this system under `single`
        // from the first outer iteration (a finding for SPEC-LIT 118); PCG
        // converges in both builds.
        p_solver: ofgpu::solver::SolverControls {
            solver: LinearSolverKind::PCG,
            precon: Preconditioner::Dic,
            rel_tol: 0.01,
            max_iter: 1000,
            ..solver
        },
        ..SimpleControls::default()
    };
    let mut sim = Simple::new(&gpu, &hm, &gm, ctrl, BuoyancyCoeffs { g: Vec3::ZERO, ..BuoyancyCoeffs::default() })?;
    let tctrl = TurbulenceControls {
        k_solver: solver,
        epsilon_solver: solver,
        // nu_t reaches kappa u* (H + z0) = 122 m^2/s at the top; the default
        // cap of 1e5 nu = 1.5 m^2/s would bound epsilon off the profile.
        nut_max_coeff: 1e8,
        ..TurbulenceControls::default()
    };

    let ab = |ty: &str| abl.spec(ty);
    setup_vector_field_with(
        &gpu,
        sim.u_mut(),
        &raw_v(
            vec![
                ("xMin", ab("atmBoundaryLayerInletVelocity")),
                ("xMax", ps("zeroGradient")),
                ("yMin", ps("empty")),
                ("yMax", ps("empty")),
                ("zMin", ps("noSlip")),
                ("zMax", ab("atmBoundaryLayerInletVelocity")),
            ],
            Vec3::ZERO,
        ),
        &hm,
        BcInputs::default(),
    )?;
    setup_scalar_field_with(
        &gpu,
        sim.p_mut(),
        &raw_s(
            "p",
            vec![
                ("xMin", ps("zeroGradient")),
                ("xMax", bc("fixedValue", &[])),
                ("yMin", ps("empty")),
                ("yMax", ps("empty")),
                ("zMin", ps("zeroGradient")),
                ("zMax", ps("zeroGradient")),
            ],
            0.0,
        ),
        &hm,
        BcInputs::default(),
    )?;

    // The initial state: the profile in every cell, at the f64 cell centre.
    let cz: Vec<f64> = hm.geom64().c.iter().map(|c| c.z).collect();
    let u0: Vec<Vec3> = cz.iter().map(|&z| Vec3::new(abl.at(z).0 as Scalar, 0.0, 0.0)).collect();
    gpu.write(&mut sim.u_mut().f, &u0)?;
    sim.initialise(&gpu)?;

    let raw_eps = raw_s(
        "epsilon",
        vec![
            ("xMin", ab("atmBoundaryLayerInletEpsilon")),
            ("xMax", ps("zeroGradient")),
            ("yMin", ps("empty")),
            ("yMax", ps("empty")),
            ("zMin", ps("epsilonWallFunction")),
            ("zMax", ab("atmBoundaryLayerInletEpsilon")),
        ],
        1.0,
    );
    let raw_k = raw_s(
        "k",
        vec![
            ("xMin", ab("atmBoundaryLayerInletK")),
            ("xMax", ps("zeroGradient")),
            ("yMin", ps("empty")),
            ("yMax", ps("empty")),
            ("zMin", ps("kqRWallFunction")),
            ("zMax", ab("atmBoundaryLayerInletK")),
        ],
        1.0,
    );
    let nut_ground = bc("nutkAtmRoughWallFunction", &[("z0", &format!("{}", abl.z0))]);
    let raw_nu_t = raw_nut(nut_ground);
    let faces = WallFaces::from_case(&raw_eps, Some(&raw_nu_t), &hm)?;
    let rough = NutRoughness::from_case(Some(&raw_nu_t), &hm)?;
    let mut model = KEpsilon::new(&gpu, &hm, &gm, coeffs, tctrl, wc, &faces, &rough)?;
    setup_scalar_field_with(&gpu, model.k_mut(), &raw_k, &hm, BcInputs::default())?;
    setup_scalar_field_with(&gpu, model.epsilon_mut(), &raw_eps, &hm, BcInputs::default())?;
    setup_scalar_field_with(&gpu, model.nut_mut(), &raw_nu_t, &hm, BcInputs::default())?;
    let k0: Vec<Scalar> = cz.iter().map(|&z| abl.at(z).1 as Scalar).collect();
    let e0: Vec<Scalar> = cz.iter().map(|&z| abl.at(z).2 as Scalar).collect();
    gpu.write(&mut model.k_mut().f, &k0)?;
    gpu.write(&mut model.epsilon_mut().f, &e0)?;
    model.initialise(&gpu, &sim.flow_state())?;

    let t = {
        let mut t = GpuScalarField::zeros(&gpu, &gm, "T")?;
        gpu.write(&mut t.bc_kind, &vec![BcKind::ZeroGradient as i32; hm.n_boundary_faces])?;
        t
    };
    let mut backend = PbicgstabBackend::new(ctrl.p_solver);
    backend.setup(&gpu, &hm, &gm, &SystemProbe::default())?;

    // SIMPLE + k-epsilon to a fixed point: the largest change of U.x over
    // 25 iterations below 1e-10 of Uref, at most 1000 iterations. The f64
    // build must stop on the change; under `single` the f32 SIMPLE floor
    // (SPEC-LIT 118) may park it at the cap, which is printed.
    let mut prev = gpu.download(&sim.u().f)?;
    let (mut iterations, mut converged, mut change) = (0usize, false, f64::NAN);
    for it in 0..1000 {
        sim.correct(&gpu, &mut backend, model.nut(), &t)?;
        model.correct(&gpu, &sim.flow_state())?;
        iterations = it + 1;
        if iterations % 25 != 0 {
            continue;
        }
        let now = gpu.download(&sim.u().f)?;
        change = prev.iter().zip(&now).fold(0.0f64, |m, (a, b)| {
            let d = (w64(a.x) - w64(b.x)).abs();
            assert!(d.is_finite(), "U went non-finite at iteration {iterations}");
            m.max(d)
        });
        prev = now;
        if change < 1e-10 * abl.u_ref {
            converged = true;
            break;
        }
    }
    eprintln!(
        "  [120.9] fetch: {iterations} iterations, last 25-iteration change of U.x {change:.3e} m/s{}",
        if converged { "" } else { " (the cap)" }
    );
    if cfg!(not(feature = "single")) {
        assert!(converged, "the fetch hit the 1000-iteration cap, change {change:e}");
    }

    // The outlet column: the cells of the last x layer, against the inlet
    // profile at their own heights.
    let uc = gpu.download(&sim.u().f)?;
    let kcell = gpu.download(&model.k().f)?;
    let xs: Vec<f64> = hm.geom64().c.iter().map(|c| c.x).collect();
    let x_last = xs.iter().cloned().fold(f64::MIN, f64::max);
    let (mut worst_u, mut worst_k, mut n_col) = (0.0f64, 0.0f64, 0usize);
    let (mut z_u, mut z_k) = (0.0f64, 0.0f64);
    for c in 0..hm.n_cells {
        if (xs[c] - x_last).abs() > 1e-6 {
            continue;
        }
        n_col += 1;
        let (u_want, k_want, _, _) = abl.at(cz[c]);
        let du = (fin("U.x", w64(uc[c].x)) - u_want).abs() / u_want;
        let dk = (fin("k", w64(kcell[c])) - k_want).abs() / k_want;
        if du > worst_u {
            (worst_u, z_u) = (du, cz[c]);
        }
        if dk > worst_k {
            (worst_k, z_k) = (dk, cz[c]);
        }
    }
    assert_eq!(n_col, 24, "the outlet column");

    // The ground's u*: (nu + nu_t,w) |U_P|/y = u*^2 at every ground face.
    let ground = patch_of(&hm, "zMin");
    let p = &hm.patches[ground];
    let nutb = gpu.download(&model.nut().bf)?;
    let mut worst_us = 0.0f64;
    for i in 0..p.size {
        let bf = p.start + i;
        let c = hm.b_face_cells[bf] as usize;
        let tau = (w64(nu) + fin("nu_t,w", w64(nutb[bf]))) * w64(uc[c].x).abs() / w64(hm.b_y[bf]);
        worst_us = worst_us.max((tau.sqrt() - u_star).abs() / u_star);
    }
    eprintln!(
        "  [120.9] fetch outlet: U worst {:.3} % (z = {z_u:.2} m), k worst {:.3} % (z = {z_k:.2} m); \
         ground u* worst {:.3} % (u* = {u_star:.6})",
        100.0 * worst_u,
        100.0 * worst_k,
        100.0 * worst_us
    );
    assert!(worst_u <= 0.03, "outlet U off the inlet profile by {:.3} %", 100.0 * worst_u);
    assert!(worst_k <= 0.10, "outlet k off the inlet profile by {:.3} %", 100.0 * worst_k);
    assert!(worst_us <= 0.03, "ground u* off by {:.3} %", 100.0 * worst_us);
    Ok(())
}
