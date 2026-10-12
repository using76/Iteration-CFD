// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CHR-18's gates for the discrete P1 (SPEC-LIT §135.5-§135.7): the slab's
//! closed form (135.11), which is the exact solution of the SAME equation
//! the solve discretises, so the measured error is pure discretisation and
//! must show order 2.0 ± 0.1 on the two meshes of each case; §94's
//! observed-order reporting on three meshes; the Marshak wall's sign; the
//! (137.2) energy-source identity at `T = T*`; and every refusal. The SAME
//! assertions in BOTH builds - the tolerances are the discretisation's, and
//! the f32 rule of §135.6/§135.7 (the fine error is at least `1e3·2^-24`)
//! is asserted, not assumed.
//!
//! An integration test, because the library's own test target does not
//! compile under `single`; an integration test links the normal library,
//! which does. A machine without a card passes vacuously, the convention of
//! `tests/f32_refinement.rs`.
//!
//! Provenance: ORIGINAL - the meshes, the cases and the assertions are this
//! tree's own construction, run against the closed form (135.11) derived in
//! SPEC-LIT §135.5 from Siegel & Howell NASA SP-164 (US Government work,
//! public domain).
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::energy::EnergySources;
use ofgpu::radiation::p1::{
    MarshakPatch, P1, P1Medium, WallTemperature,
};
use ofgpu::radiation::SIGMA_SB;
use ofgpu::solver::LinearSolverKind::PCG;
use ofgpu::solver::SolverControls;
use ofgpu::vv::{observed_order, Behaviour, Level};
use ofgpu::{Gpu, GpuMesh, GpuScalarField, HostMesh, Result, Scalar};

/// The f32 rule of docs/17 §5.1 D-F32-1: `tol32 = max(tol64,
/// min(tol64·2^29, 1e-5))`, which for a `1e-12` f64 tolerance is `1e-5`.
const TOL: f64 = if cfg!(feature = "single") { 1e-5 } else { 1e-12 };

/// Every device test needs a card; without one the test passes vacuously.
fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

/// The build's `Scalar` widened to the f64 the closed form lives in: a
/// no-op in the f64 build, the exact widening under `single`.
#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    x as f64
}

/// The slab: x in [0, 1] m with `n` cells, one cell each in y and z over
/// [0, 0.1]; Marshak walls at xMin and xMax, the four other slots empty.
fn slab_spec(n: usize) -> BlockSpec {
    let axis = |lo: f64, hi: f64, cells: usize| GradedAxis {
        lo: lo as Scalar,
        hi: hi as Scalar,
        n: cells,
        expansion: 1.0,
        two_sided: false,
    };
    BlockSpec {
        x: axis(0.0, 1.0, n),
        // 0.125, not 0.1: the cross-section's every coordinate is then a
        // binary fraction, exactly representable under `single`, so the
        // f32 build's operator is the f64 build's and the gate measures the
        // discretisation, not Scalar rounding of the geometry.
        y: axis(0.0, 0.125, 1),
        z: axis(0.0, 0.125, 1),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall", "wall", "empty", "empty", "empty", "empty"].map(String::from),
        windows: Vec::new(),
        cyclic: Vec::new(),
    }
}

/// The mesh is uploaded on the CALLER's device: a second `Gpu::new(0)` has
/// its own stream, and stream-ordered allocations (cudarc's
/// `cudaMallocAsync`) are only ordered within one stream - a mesh uploaded
/// on another stream races every kernel launched here.
fn slab(g: &Gpu, n: usize) -> Result<(HostMesh, GpuMesh)> {
    let hm = blockgen::build_mesh(&slab_spec(n))?;
    let m = GpuMesh::upload(g, &hm)?;
    Ok((hm, m))
}

/// PCG, tolerance 1e-12, rel_tol 0, max_iter 5000, in BOTH builds - under
/// `single` the solve is SPEC-LIT §118.4-refined and this is its target.
fn controls() -> SolverControls {
    SolverControls {
        solver: PCG,
        tolerance: 1e-12,
        rel_tol: 0.0,
        max_iter: 5000,
        ..SolverControls::default()
    }
}

/// `κ = τ(1-ω)/L`, `σ_s = τω/L` with L = 1 - the uniform grey medium of
/// (135.11).
fn medium(tau: f64, omega: f64, a1: f64) -> P1Medium {
    P1Medium { kappa: tau * (1.0 - omega), sigma_s: tau * omega, a1 }
}

fn walls(emissivity: f64, tw: f64) -> Vec<MarshakPatch> {
    let t = if tw == 0.0 { WallTemperature::Fixed(0.0) } else { WallTemperature::Fixed(tw) };
    vec![
        MarshakPatch { patch: "xMin".into(), emissivity, temperature: t },
        MarshakPatch { patch: "xMax".into(), emissivity, temperature: t },
    ]
}

/// T uniform in every cell and on every face, written through `t.f`/`t.bf`.
fn uniform_t(g: &Gpu, m: &GpuMesh, t_k: f64) -> Result<GpuScalarField> {
    let mut t = GpuScalarField::zeros(g, m, "T")?;
    g.write(&mut t.f, &vec![t_k as Scalar; m.n_cells])?;
    g.write(&mut t.bf, &vec![t_k as Scalar; m.n_boundary_faces])?;
    Ok(t)
}

/// (135.11), computed HERE in f64: `a = ε/(2(2-ε))`,
/// `g = √((1-ω)/(3-ωA₁))`, `h = √((1-ω)(3-ωA₁))·τ/2`.
fn psi_p1(emissivity: f64, tau: f64, omega: f64, a1: f64) -> f64 {
    let a = emissivity / (2.0 * (2.0 - emissivity));
    let g = ((1.0 - omega) / (3.0 - omega * a1)).sqrt();
    let h = ((1.0 - omega) * (3.0 - omega * a1)).sqrt() * tau / 2.0;
    4.0 * a * g * h.tanh() / (a + g * h.tanh())
}

/// Ψ_disc per wall: the mean of (q·n)_b over that patch's faces, divided by
/// `σ (T^4 - T_w^4)` - (135.15) against the slab's driving. Both walls are
/// returned; the case asserts on both.
fn psi_walls(g: &Gpu, hm: &HostMesh, p1: &P1, t_k: f64, tw: f64) -> Result<(f64, f64)> {
    let q = g.download(p1.wall_flux())?;
    let side = |name: &str| -> f64 {
        let p = hm.patches.iter().find(|p| p.name == name).expect("xMin/xMax");
        let sum: f64 = (p.start..p.start + p.size).map(|bf| w64(q[bf])).sum();
        let n = p.size.max(1);
        sum / n as f64 / (SIGMA_SB as f64 * (t_k * t_k * t_k * t_k - tw * tw * tw * tw))
    };
    Ok((side("xMin"), side("xMax")))
}

/// Solve one slab case and return Ψ on both walls.
fn solve_slab(
    g: &Gpu,
    n: usize,
    emissivity: f64,
    tau: f64,
    omega: f64,
    a1: f64,
    tw: f64,
) -> Result<(P1, f64, f64)> {
    let (hm, m) = slab(g, n)?;
    let mut p1 = P1::new(g, &hm, &m, &medium(tau, omega, a1), &walls(emissivity, tw), controls())?;
    let t = uniform_t(g, &m, 1000.0)?;
    let perf = p1.solve(g, &m, &t)?;
    assert!(perf.converged, "the PCG solve must converge (tolerance 1e-12)");
    let psi = psi_walls(g, &hm, &p1, 1000.0, tw)?;
    Ok((p1, psi.0, psi.1))
}

// --------------------------------------------------------------------------
//  1. The slab's closed form - SPEC-LIT §135.5's table, §135.7's meshes
// --------------------------------------------------------------------------

/// The seven cases of §135.5's table on two meshes each. The discrete Ψ is
/// compared with (135.11) - the exact solution of the SAME PDE - so the
/// error is pure discretisation: same sign coarse and fine, shrinking,
/// order 2.0 ± 0.1, and the fine error at least `1e3·2^-24` (§135.6's f32
/// rule: the error is discretisation, not round-off - asserted here, in
/// both builds).
#[test]
fn p1_slab_closed_form() {
    let Some(g) = gpu() else { return };

    // (ε, τ, ω, A₁, N_coarse, N_fine). The two thin cases use N = 2, 4:
    // under `single` the assembled operator carries the build's own f32
    // unit round-off (2^-24), amplified by the operator's condition number,
    // and that floor
    // FALLS as the mesh coarsens (the absorption term that fixes G's level
    // gains on the face conductances) while the discretisation error GROWS
    // - so the coarse pair keeps the error an order of magnitude above the
    // floor on BOTH meshes, which is what §135.6's class-P rule asks. The
    // meshes are powers of two so Δx is a binary fraction: exact under
    // `single`, the f32 geometry IS the f64 geometry.
    const CASES: [(f64, f64, f64, f64, usize, usize); 7] = [
        (1.0, 0.1, 0.0, 0.0, 2, 4),
        (1.0, 1.0, 0.0, 0.0, 8, 16),
        (1.0, 10.0, 0.0, 0.0, 40, 80),
        (0.5, 0.1, 0.0, 0.0, 2, 4),
        (0.5, 1.0, 0.0, 0.0, 8, 16),
        (0.5, 10.0, 0.0, 0.0, 40, 80),
        (1.0, 1.0, 0.5, 0.5, 8, 16),
    ];
    // §135.5's Ψ_exact for the black non-scattering cases (135.6), REPORTED
    // beside P1's own error, never gated: P1 vs transport is the method's
    // error, not this discretisation's.
    const EXACT: [f64; 3] = [0.167417084183, 0.780616065604, 0.999992902475];
    // §135.5's Ψ_P1 per case, in CASES' order - the closed form's own table.
    const TABLE: [f64; 7] = [
        0.181406101221,
        0.893523053720,
        1.071796739833,
        0.153551010980,
        0.471883905488,
        0.517327167876,
        0.620331780315,
    ];

    for (ci, &(eps, tau, om, a1, nc, nf)) in CASES.iter().enumerate() {
        let want = psi_p1(eps, tau, om, a1);
        // (135.11) as coded here against SPEC-LIT §135.5's printed table:
        // f64 arithmetic in both builds, so twelve digits in both.
        assert!(
            (want - TABLE[ci]).abs() < 5e-13,
            "eps {eps} tau {tau} omega {om}: (135.11) gives {want}, SPEC-LIT \
             135.5's table says {}",
            TABLE[ci]
        );
        let mut es = [0.0f64; 2];
        let mut psis = [0.0f64; 2];
        for (mi, &n) in [nc, nf].iter().enumerate() {
            let (_p1, x_min, x_max) =
                solve_slab(&g, n, eps, tau, om, a1, 0.0).expect("the slab solves");
            let e_min = x_min / want - 1.0;
            let e_max = x_max / want - 1.0;
            // Both walls carry the same discretisation; the reported error
            // is the larger of the two, which is the conservative one for
            // the floor rule below.
            psis[mi] = x_min;
            es[mi] = if e_min.abs() >= e_max.abs() { e_min } else { e_max };
        }
        let p = (es[0].abs() / es[1].abs()).log2();
        println!(
            "eps {eps} tau {tau} omega {om} N ({nc},{nf}): psi_P1 {want:.12} \
             psi {:.9} -> {:.9} e_c {:.3e} e_f {:.3e} p {p:.4} |e_f|/2^-24 {:.1e}",
            psis[0],
            psis[1],
            es[0],
            es[1],
            es[1].abs() / 2.0f64.powi(-24)
        );
        assert!(
            es[0].signum() == es[1].signum() && es[1].abs() < es[0].abs(),
            "eps {eps} tau {tau}: the error must shrink, same sign: {es:?}"
        );
        assert!(
            (1.9..=2.1).contains(&p),
            "eps {eps} tau {tau}: observed order {p} is not 2.0 +- 0.1"
        );
        assert!(
            es[1].abs() >= 1e3 * 2.0f64.powi(-24),
            "eps {eps} tau {tau}: fine error {} is round-off, not \
             discretisation (SPEC-LIT 135.6)",
            es[1].abs()
        );
        if ci < 3 {
            println!(
                "  black, tau {tau}: psi_exact {} psi_P1/psi_exact - 1 {:+.2}%",
                EXACT[ci],
                (want / EXACT[ci] - 1.0) * 100.0
            );
        }
    }
}

// --------------------------------------------------------------------------
//  2. The observed order, reported SPEC-LIT §94's way
// --------------------------------------------------------------------------

/// ε = 1, τ = 1, ω = 0 on N = 8, 16, 32: `vv::observed_order` on
/// `Levels { h = 1/N, value = Ψ_disc }`, finest first. Monotone, p in
/// [1.9, 2.1], and the exact (135.11) answer inside §94's GCI band.
#[test]
fn p1_observed_order() {
    let Some(g) = gpu() else { return };
    let want = psi_p1(1.0, 1.0, 0.0, 0.0);

    let mut levels = [Level { h: 0.0, value: 0.0 }; 3];
    for (li, &n) in [32usize, 16, 8].iter().enumerate() {
        let (_p1, x_min, x_max) = solve_slab(&g, n, 1.0, 1.0, 0.0, 0.0, 0.0)
            .expect("the slab solves");
        levels[li] = Level {
            h: (1.0 / n as f64) as Scalar,
            value: ((x_min + x_max) / 2.0) as Scalar,
        };
    }
    let t = observed_order(&levels).expect("three finite levels");
    println!(
        "observed order: psi {} -> {} -> {} against psi_P1 {want:.12}: \
         r21 {:.4} r32 {:.4} ratio {:.4} {:?} p {:?} gci_fine {:?}",
        w64(levels[0].value),
        w64(levels[1].value),
        w64(levels[2].value),
        w64(t.r21),
        w64(t.r32),
        w64(t.ratio),
        t.behaviour,
        t.p.map(w64),
        t.gci_fine.map(w64)
    );
    assert_eq!(t.behaviour, Behaviour::Monotone, "the mesh sequence must converge monotonically");
    let p = t.p.expect("a monotone triplet carries p");
    assert!(
        (1.9..=2.1).contains(&w64(p)),
        "observed order {} is not 2.0 +- 0.1",
        w64(p)
    );
    let gci = t.gci_fine.expect("a monotone triplet carries a GCI");
    assert!(
        (w64(levels[0].value) - want).abs() <= w64(gci) * w64(levels[0].value).abs(),
        "the exact answer must sit inside the GCI band: |{} - {want}| <= {}",
        w64(levels[0].value),
        w64(gci)
    );
}

// --------------------------------------------------------------------------
//  3. The Marshak wall's sign - SPEC-LIT §135.4's correction
// --------------------------------------------------------------------------

/// N = 16, τ = 1, ε = 1 and 0.5. A cold wall must take energy OUT of the
/// medium: every Marshak face's `(q·n)_b > 0`, and `q_r` points at the
/// wall - positive-x in the half towards xMax, negative-x in the half
/// towards xMin. With the walls at 2000 K the flux reverses, both the
/// faces' and the field's.
#[test]
fn p1_marshak_sign() {
    let Some(g) = gpu() else { return };

    for eps in [1.0, 0.5] {
        let (hm, m) = slab(&g, 16).expect("mesh");
        let mut p1 =
            P1::new(&g, &hm, &m, &medium(1.0, 0.0, 0.0), &walls(eps, 0.0), controls())
                .expect("P1");
        let t = uniform_t(&g, &m, 1000.0).expect("T");
        p1.solve(&g, &m, &t).expect("solve");

        let q = g.download(p1.wall_flux()).expect("wall flux");
        for (bf, &v) in q.iter().enumerate() {
            if hm.b_kind[bf] == ofgpu::PatchKind::Empty as ofgpu::Label {
                continue;
            }
            assert!(w64(v) > 0.0, "eps {eps}: face {bf} flux {} is not out of a \
                medium hotter than its wall", w64(v));
        }
        let qr = g.download(p1.qr()).expect("qr");
        let xc = g.download(&m.c).expect("centres");
        for (c, (&v, &x)) in qr.iter().zip(&xc).enumerate() {
            if x.x > 0.5 {
                assert!(w64(v.x) > 0.0, "eps {eps}: cell {c} at x = {} must radiate \
                    towards xMax, q_r.x = {}", w64(x.x), w64(v.x));
            } else {
                assert!(w64(v.x) < 0.0, "eps {eps}: cell {c} at x = {} must radiate \
                    towards xMin, q_r.x = {}", w64(x.x), w64(v.x));
            }
        }

        // The same slab with the walls at 2000 K: everything reverses.
        let mut hot =
            P1::new(&g, &hm, &m, &medium(1.0, 0.0, 0.0), &walls(eps, 2000.0), controls())
                .expect("P1 hot");
        hot.solve(&g, &m, &t).expect("solve hot");
        let q = g.download(hot.wall_flux()).expect("wall flux");
        for (bf, &v) in q.iter().enumerate() {
            if hm.b_kind[bf] == ofgpu::PatchKind::Empty as ofgpu::Label {
                continue;
            }
            assert!(w64(v) < 0.0, "eps {eps}: a 2000 K wall must HEAT the medium, \
                face {bf} flux {}", w64(v));
        }
        let qr = g.download(hot.qr()).expect("qr");
        for (c, (&v, &x)) in qr.iter().zip(&xc).enumerate() {
            if x.x > 0.5 {
                assert!(w64(v.x) < 0.0, "eps {eps}: hot wall flips the sign, cell \
                    {c} q_r.x = {}", w64(v.x));
            } else {
                assert!(w64(v.x) > 0.0, "eps {eps}: hot wall flips the sign, cell \
                    {c} q_r.x = {}", w64(v.x));
            }
        }
    }
}

// --------------------------------------------------------------------------
//  4. The energy-source identity - (137.2) at T = T* is (135.4)
// --------------------------------------------------------------------------

/// N = 16, τ = 1, T_c = 400 + 100 c. Solve, register (137.2)'s pair on a
/// fresh `EnergySources`, and check per cell, in f64 from the downloaded
/// values, that `q + sp·T - κ(G - 4σT^4)` vanishes against the explicit
/// term - (135.4)'s right-hand side, reproduced exactly in real arithmetic.
/// `sp <= 0` every cell (Patankar's sign). docs/17 §5.1's tol32 under
/// `single`.
#[test]
fn p1_energy_source_identity() {
    let Some(g) = gpu() else { return };
    let (hm, m) = slab(&g, 16).expect("mesh");
    let mut p1 =
        P1::new(&g, &hm, &m, &medium(1.0, 0.0, 0.0), &walls(1.0, 0.0), controls())
            .expect("P1");

    let tv: Vec<Scalar> = (0..m.n_cells).map(|c| (400.0 + 100.0 * c as f64) as Scalar).collect();
    let mut t = GpuScalarField::zeros(&g, &m, "T").expect("T");
    g.write(&mut t.f, &tv).expect("T cells");
    // Every face takes its owner's value - zeroGradient's own answer - so
    // a FromField wall would read something honest too.
    let owners: Vec<usize> =
        hm.b_face_cells.iter().map(|&c| c as usize).collect();
    let tb: Vec<Scalar> = owners.iter().map(|&c| tv[c]).collect();
    g.write(&mut t.bf, &tb).expect("T faces");

    p1.solve(&g, &m, &t).expect("solve");

    let mut sources = EnergySources::new(&g, &m).expect("sources");
    sources.clear(&g).expect("clear");
    p1.register_energy_source(&g, &t, &mut sources).expect("register");

    let q = g.download(sources.q()).expect("q");
    let sp = g.download(sources.sp()).expect("sp");
    let gf = g.download(&p1.g().f).expect("G");
    let tf = g.download(&t.f).expect("T");
    let kappa = 1.0; // tau (1 - omega), L = 1
    let sigma = SIGMA_SB as f64;

    let mut worst = 0.0f64;
    for c in 0..m.n_cells {
        let (qc, spc, gc, tc) =
            (w64(q[c]), w64(sp[c]), w64(gf[c]), w64(tf[c]));
        assert!(spc <= 0.0, "cell {c}: the implicit sink {spc} must be <= 0");
        let num = (qc + spc * tc - kappa * (gc - 4.0 * sigma * tc * tc * tc * tc)).abs();
        let den = kappa * (gc + 12.0 * sigma * tc * tc * tc * tc);
        let r = num / den;
        worst = worst.max(r);
        assert!(
            r <= TOL,
            "cell {c}: the (137.2) split is off by r = {r} against tol {TOL}"
        );
    }
    println!("p1_energy_source_identity: max r_c = {worst:.3e} (tol {TOL})");
}

// --------------------------------------------------------------------------
//  5. The refusals
// --------------------------------------------------------------------------

/// Every refusal `P1Medium::from_config` and `P1::new` make, each checked
/// for `Err` AND for the entry name in the message - a refusal that does
/// not say what it refused is a substitution with extra steps.
#[test]
fn p1_refusals() {
    let Some(g) = gpu() else { return };
    let (hm, m) = slab(&g, 8).expect("mesh");
    use ofgpu::radiation::props::{Absorption, ParticipatingConfig, Scatter};
    use ofgpu::radiation::RadiationModel;

    let cfg = |model, absorption, scatter| ParticipatingConfig {
        model,
        absorption,
        scatter,
    };
    fn refused<T>(what: &str, r: ofgpu::Result<T>) {
        let e = r.err().unwrap_or_else(|| panic!("{what}: must be refused"));
        let msg = format!("{e}");
        assert!(msg.contains(what), "the refusal must name the entry: {msg}");
    }

    // from_config: a model that is not P1.
    refused(
        "fvDOM",
        P1Medium::from_config(&cfg(
            RadiationModel::FvDom,
            Absorption::Constant { kappa: 1.0 },
            Scatter::None,
        )),
    );
    refused(
        "viewFactor",
        P1Medium::from_config(&cfg(
            RadiationModel::ViewFactor,
            Absorption::Constant { kappa: 1.0 },
            Scatter::None,
        )),
    );
    // The WSGG sets are CHR-22's; this grey solve refuses them by name.
    refused(
        "wsgg",
        P1Medium::from_config(&cfg(
            RadiationModel::P1,
            Absorption::Wsgg {
                file: "bordbar.dat".into(),
                pressure_atm: 1.0,
            },
            Scatter::None,
        )),
    );
    refused(
        "wsggGrey",
        P1Medium::from_config(&cfg(
            RadiationModel::P1,
            Absorption::WsggGrey {
                file: "bordbar.dat".into(),
                pressure_atm: 1.0,
            },
            Scatter::None,
        )),
    );
    refused(
        "absorptionCoefficient",
        P1Medium::from_config(&cfg(
            RadiationModel::P1,
            Absorption::Constant { kappa: -0.5 },
            Scatter::None,
        )),
    );
    refused(
        "absorptionCoefficient",
        P1Medium::from_config(&cfg(
            RadiationModel::P1,
            Absorption::Constant { kappa: f64::NAN },
            Scatter::None,
        )),
    );

    // new(): the medium's numbers.
    let build = |med: &P1Medium, wl: &[MarshakPatch]| {
        P1::new(&g, &hm, &m, med, wl, controls())
    };
    let ok_walls = walls(1.0, 0.0);
    refused("kappa", build(&P1Medium { kappa: f64::NAN, sigma_s: 0.0, a1: 0.0 }, &ok_walls));
    refused(
        "sigma_s",
        build(&P1Medium { kappa: 1.0, sigma_s: -0.1, a1: 0.0 }, &ok_walls),
    );
    refused(
        "sigma_s",
        build(&P1Medium { kappa: 1.0, sigma_s: f64::NAN, a1: 0.0 }, &ok_walls),
    );
    refused("kappa", build(&P1Medium { kappa: 0.0, sigma_s: 0.0, a1: 0.0 }, &ok_walls));
    refused(
        "a1",
        build(&P1Medium { kappa: 1.0, sigma_s: 0.5, a1: 1.5 }, &ok_walls),
    );

    // new(): the walls.
    let mut bad_e = walls(0.0, 0.0);
    refused("emissivity", build(&medium(1.0, 0.0, 0.0), &bad_e));
    bad_e = walls(1.5, 0.0);
    refused("emissivity", build(&medium(1.0, 0.0, 0.0), &bad_e));
    bad_e = walls(f64::NAN, 0.0);
    refused("emissivity", build(&medium(1.0, 0.0, 0.0), &bad_e));
    refused("T_w", build(&medium(1.0, 0.0, 0.0), &walls(1.0, -1.0)));
    refused("T_w", build(&medium(1.0, 0.0, 0.0), &walls(1.0, f64::NAN)));

    let unknown = vec![MarshakPatch {
        patch: "hotWall".into(),
        emissivity: 1.0,
        temperature: WallTemperature::Fixed(0.0),
    }];
    refused("hotWall", build(&medium(1.0, 0.0, 0.0), &unknown));

    let empty = vec![MarshakPatch {
        patch: "zMin".into(),
        emissivity: 1.0,
        temperature: WallTemperature::Fixed(0.0),
    }];
    refused("zMin", build(&medium(1.0, 0.0, 0.0), &empty));

    let twice = vec![
        MarshakPatch {
            patch: "xMin".into(),
            emissivity: 1.0,
            temperature: WallTemperature::Fixed(0.0),
        },
        MarshakPatch {
            patch: "xMin".into(),
            emissivity: 0.5,
            temperature: WallTemperature::Fixed(0.0),
        },
    ];
    refused("xMin", build(&medium(1.0, 0.0, 0.0), &twice));

    // No Marshak patch at all: the equation is underdetermined without one.
    refused("no Marshak patch", build(&medium(1.0, 0.0, 0.0), &[]));
}

// --------------------------------------------------------------------------
//  6. The Marshak data pairs PATCH-with-wall
// --------------------------------------------------------------------------

/// The N = 16 slab, tau = 1, T = 1000 K, with the same two walls listed in
/// both orders - [xMin eps 1 at 2000 K, xMax eps 0.5 at 0 K] and the
/// reverse. Every face of a patch takes THAT wall's `a`, `T_w` and flag
/// whatever the order, so G, the wall fluxes and q_r are bitwise equal
/// across the listings; and the hot xMin takes flux in while the cold xMax
/// lets it out in BOTH listings, so a swapped pairing cannot pass by
/// symmetry.
#[test]
fn p1_wall_order_is_irrelevant() {
    let Some(g) = gpu() else { return };
    let hot = || MarshakPatch {
        patch: "xMin".into(),
        emissivity: 1.0,
        temperature: WallTemperature::Fixed(2000.0),
    };
    let cold = || MarshakPatch {
        patch: "xMax".into(),
        emissivity: 0.5,
        temperature: WallTemperature::Fixed(0.0),
    };
    let run = |order: Vec<MarshakPatch>| -> Result<(Vec<u64>, Vec<u64>, Vec<u64>)> {
        let (hm, m) = slab(&g, 16)?;
        let mut p1 = P1::new(&g, &hm, &m, &medium(1.0, 0.0, 0.0), &order, controls())?;
        let t = uniform_t(&g, &m, 1000.0)?;
        p1.solve(&g, &m, &t)?;
        let q = g.download(p1.wall_flux())?;
        for name in ["xMin", "xMax"] {
            let p = hm.patches.iter().find(|p| p.name == name).expect("xMin/xMax");
            for bf in p.start..p.start + p.size {
                assert!(
                    if name == "xMax" { w64(q[bf]) > 0.0 } else { w64(q[bf]) < 0.0 },
                    "{name}: face {bf} flux {} contradicts its own wall", w64(q[bf])
                );
            }
        }
        let bits = |v: &[Scalar]| v.iter().map(|&x| w64(x).to_bits()).collect::<Vec<u64>>();
        let mut qr = Vec::new();
        for v in g.download(p1.qr())? {
            qr.push(w64(v.x).to_bits());
            qr.push(w64(v.y).to_bits());
            qr.push(w64(v.z).to_bits());
        }
        Ok((bits(&g.download(&p1.g().f)?), bits(&q), qr))
    };
    let first = run(vec![hot(), cold()]).expect("walls in patch order");
    let second = run(vec![cold(), hot()]).expect("walls reversed");
    assert_eq!(first.0, second.0, "G depends on the wall listing order");
    assert_eq!(first.1, second.1, "the wall flux depends on the wall listing order");
    assert_eq!(first.2, second.2, "q_r depends on the wall listing order");
    println!("p1_wall_order_is_irrelevant: both listings agree bitwise");
}

/// A slab two cells tall in y (y in [0, 0.25]), one in z: xMin and xMax are
/// walls with TWO faces each, yMin/yMax plain zeroGradient patches (a
/// symmetry plane for G), so the problem is still the 1-D slab. Every face
/// of BOTH wall patches carries the wall's data: all four Marshak fluxes
/// come out positive, and each equals the one-cell-in-y N = 8 slab's xMin
/// flux to solver noise - 1e-10 relative in f64, 1e-5 under `single` - the
/// y direction carrying no gradient at all.
#[test]
fn p1_multi_face_patch() {
    let Some(g) = gpu() else { return };
    let axis = |lo: f64, hi: f64, cells: usize| GradedAxis {
        lo: lo as Scalar,
        hi: hi as Scalar,
        n: cells,
        expansion: 1.0,
        two_sided: false,
    };
    let spec = BlockSpec {
        x: axis(0.0, 1.0, 8),
        y: axis(0.0, 0.25, 2),
        z: axis(0.0, 0.125, 1),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall", "wall", "patch", "patch", "empty", "empty"].map(String::from),
        windows: Vec::new(),
        cyclic: Vec::new(),
    };
    let hm = blockgen::build_mesh(&spec).expect("mesh");
    let m = GpuMesh::upload(&g, &hm).expect("upload");
    let mut p1 =
        P1::new(&g, &hm, &m, &medium(1.0, 0.0, 0.0), &walls(1.0, 0.0), controls()).expect("P1");
    let t = uniform_t(&g, &m, 1000.0).expect("T");
    p1.solve(&g, &m, &t).expect("solve");

    // The reference: the same 1-D slab one cell tall in y, whose xMin patch
    // carries a single face.
    let (hm1, m1) = slab(&g, 8).expect("reference mesh");
    let mut ref1 =
        P1::new(&g, &hm1, &m1, &medium(1.0, 0.0, 0.0), &walls(1.0, 0.0), controls())
            .expect("P1 reference");
    let t1 = uniform_t(&g, &m1, 1000.0).expect("T reference");
    ref1.solve(&g, &m1, &t1).expect("reference solve");
    let q1 = g.download(ref1.wall_flux()).expect("reference flux");
    let px = hm1.patches.iter().find(|p| p.name == "xMin").expect("xMin");
    let reference = w64(q1[px.start]);

    let tol = if cfg!(feature = "single") { 1e-5 } else { 1e-10 };
    let q = g.download(p1.wall_flux()).expect("wall flux");
    for name in ["xMin", "xMax"] {
        let p = hm.patches.iter().find(|p| p.name == name).expect("xMin/xMax");
        for bf in p.start..p.start + p.size {
            assert!(w64(q[bf]) > 0.0, "{name}: face {bf} flux {} is not out of a \
                medium hotter than its wall", w64(q[bf]));
            let r = w64(q[bf]) / reference - 1.0;
            assert!(r.abs() <= tol, "{name}: face {bf} flux {} against the 1-D \
                slab's {reference} is r = {r}, tol {tol}", w64(q[bf]));
        }
    }
    println!("p1_multi_face_patch: all four Marshak faces match the 1-D slab to {tol}");
}

/// `FromField` walls: `T.f = 1000` in every
/// cell and `T.bf = 1500` on every face, the walls `FromField` against the
/// walls `Fixed(1500)` - the SAME solve, bitwise, on G, the wall fluxes and
/// q_r, because the kernel reads the stated `T_w` and the wall's own `T`
/// patch value at the same face. And the walls, hotter than the 1000 K
/// medium, take flux in: every Marshak face's flux is negative.
#[test]
fn p1_from_field_wall_equals_fixed() {
    let Some(g) = gpu() else { return };
    let from_walls = || {
        vec![
            MarshakPatch {
                patch: "xMin".into(),
                emissivity: 0.5,
                temperature: WallTemperature::FromField,
            },
            MarshakPatch {
                patch: "xMax".into(),
                emissivity: 0.5,
                temperature: WallTemperature::FromField,
            },
        ]
    };
    let fixed_walls = || {
        vec![
            MarshakPatch {
                patch: "xMin".into(),
                emissivity: 0.5,
                temperature: WallTemperature::Fixed(1500.0),
            },
            MarshakPatch {
                patch: "xMax".into(),
                emissivity: 0.5,
                temperature: WallTemperature::Fixed(1500.0),
            },
        ]
    };
    let (hm, m) = slab(&g, 16).expect("mesh");
    let mut t = GpuScalarField::zeros(&g, &m, "T").expect("T");
    g.write(&mut t.f, &vec![1000.0 as Scalar; m.n_cells]).expect("T cells");
    g.write(&mut t.bf, &vec![1500.0 as Scalar; m.n_boundary_faces]).expect("T faces");

    let mut from =
        P1::new(&g, &hm, &m, &medium(1.0, 0.0, 0.0), &from_walls(), controls()).expect("P1");
    from.solve(&g, &m, &t).expect("solve FromField");
    let mut fix =
        P1::new(&g, &hm, &m, &medium(1.0, 0.0, 0.0), &fixed_walls(), controls()).expect("P1");
    fix.solve(&g, &m, &t).expect("solve Fixed");

    let bits = |v: &[Scalar]| v.iter().map(|&x| w64(x).to_bits()).collect::<Vec<u64>>();
    let qr_bits = |p: &P1| -> Result<Vec<u64>> {
        let mut out = Vec::new();
        for v in g.download(p.qr())? {
            out.push(w64(v.x).to_bits());
            out.push(w64(v.y).to_bits());
            out.push(w64(v.z).to_bits());
        }
        Ok(out)
    };
    assert_eq!(
        bits(&g.download(&from.g().f).expect("G FromField")),
        bits(&g.download(&fix.g().f).expect("G Fixed")),
        "a FromField wall reading 1500 K is not the Fixed(1500) solve (G)"
    );
    assert_eq!(
        bits(&g.download(from.wall_flux()).expect("flux FromField")),
        bits(&g.download(fix.wall_flux()).expect("flux Fixed")),
        "a FromField wall reading 1500 K is not the Fixed(1500) solve (wall flux)"
    );
    assert_eq!(
        qr_bits(&from).expect("qr FromField"),
        qr_bits(&fix).expect("qr Fixed"),
        "a FromField wall reading 1500 K is not the Fixed(1500) solve (q_r)"
    );
    let q = g.download(from.wall_flux()).expect("wall flux");
    for name in ["xMin", "xMax"] {
        let p = hm.patches.iter().find(|p| p.name == name).expect("xMin/xMax");
        for bf in p.start..p.start + p.size {
            assert!(w64(q[bf]) < 0.0, "{name}: face {bf} flux {} - a 1500 K wall \
                must HEAT the 1000 K medium", w64(q[bf]));
        }
    }
    println!("p1_from_field_wall_equals_fixed: FromField is Fixed(1500) bitwise");
}
