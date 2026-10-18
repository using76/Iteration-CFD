// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate CHR-SLAB (SPEC-LIT §135.5, §136.7): the isothermal grey slab, run
//! live - P1 against its closed form (135.11) and the S8 fvDOM slab against
//! `1 - 2 E_3(tau)` (135.6).**
//!
//! A live port of this repository's own tests into `ofgpu-validate` rows,
//! one row per assertion, the tests staying as they are:
//! `tests/radiation_p1.rs::p1_slab_closed_form` - the seven (eps, tau,
//! omega, A1) cases of §135.5's table on two meshes each, the measured
//! order 2.0 ± 0.1, the shrinking same-sign error and §135.6's f32 floor -
//! and `tests/radiation_dom.rs::dom_slab_exact` - the E_3 oracle, the S8
//! slab at N = 256 inside 2 % of (135.6) at tau = 1 and 10, both walls,
//! every solve the step recurrence (136.12) at round-off, and tau = 0.1's
//! recorded miss asserted with its cause: S8's own angular limit (136.13)
//! lies above 2 %, accepted by the user on 2026-10-14. Under `single` the
//! N = 256 slab does not converge at all - the f32 PBiCGStab breaks down
//! on the 256-cell upwind chain (§136.7) - and the gate rows show it: one
//! sweep, every direction solve unconverged, the P1 and oracle rows
//! unchanged.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §135.5-§135.7 (the P1 slab, its table and its
//!     meshes), §136.7 (the DOM slab gate, (136.12), (136.13) and its
//!     recorded miss)
//!   tests/radiation_p1.rs (`p1_slab_closed_form` and its helpers 45-175),
//!     tests/radiation_dom.rs (`dom_slab_exact` and its helpers) - the
//!     tests stay as they are
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::radiation::dom::{DomCondition, DomControls, DomMedium, DomPatch, DomSolve};
use ofgpu::radiation::dom_quad::{Quadrature, SnOrder};
use ofgpu::radiation::p1::{MarshakPatch, P1, P1Medium, WallTemperature};
use ofgpu::radiation::SIGMA_SB;
use ofgpu::solver::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::{Error, Gpu, GpuMesh, GpuScalarField, HostMesh, Result, Scalar};

use super::chr_fixture::{psi_exact, w64};
use crate::{Checks, GateReport, How, Uncertainty, Verdict};

/// The gate's scope and registry name.
pub(crate) const GATE: &str = "SPEC-LIT 136.7 Gate CHR-SLAB the isothermal grey slab";

const PI: f64 = std::f64::consts::PI;

// ---------------------------------------------------------------------------
//  The P1 slab's helpers (tests/radiation_p1.rs 45-175)
// ---------------------------------------------------------------------------

/// §135.7's P1 slab: x in [0, 1] m with `n` cells, one cell each in y and
/// z over [0, 0.125] - binary fractions, so the f32 build's geometry is the
/// f64 build's; Marshak walls at xMin and xMax, the four other slots empty.
fn p1_spec(n: usize) -> BlockSpec {
    let axis = |lo: f64, hi: f64, cells: usize| GradedAxis {
        lo: lo as Scalar,
        hi: hi as Scalar,
        n: cells,
        expansion: 1.0,
        two_sided: false,
    };
    BlockSpec {
        x: axis(0.0, 1.0, n),
        y: axis(0.0, 0.125, 1),
        z: axis(0.0, 0.125, 1),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall", "wall", "empty", "empty", "empty", "empty"].map(String::from),
        windows: Vec::new(),
        cyclic: Vec::new(),
    }
}

/// The mesh is uploaded on the CALLER's device: a second `Gpu::new(0)` has
/// its own stream, and stream-ordered allocations are only ordered within
/// one stream.
fn p1_slab(g: &Gpu, n: usize) -> Result<(HostMesh, GpuMesh)> {
    let hm = blockgen::build_mesh(&p1_spec(n))?;
    let m = GpuMesh::upload(g, &hm)?;
    Ok((hm, m))
}

/// PCG, tolerance 1e-12, rel_tol 0, max_iter 5000, in BOTH builds - under
/// `single` the solve is SPEC-LIT §118.4-refined and this is its target.
fn p1_controls() -> SolverControls {
    SolverControls {
        solver: LinearSolverKind::PCG,
        tolerance: 1e-12,
        rel_tol: 0.0,
        max_iter: 5000,
        ..SolverControls::default()
    }
}

/// `kappa = tau (1-omega)/L`, `sigma_s = tau omega/L` with L = 1 - the
/// uniform grey medium of (135.11).
fn p1_medium(tau: f64, omega: f64, a1: f64) -> P1Medium {
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

/// (135.11), computed HERE in f64: `a = eps/(2(2-eps))`,
/// `g = sqrt((1-omega)/(3-omega A1))`, `h = sqrt((1-omega)(3-omega A1)) tau/2`.
fn psi_p1(emissivity: f64, tau: f64, omega: f64, a1: f64) -> f64 {
    let a = emissivity / (2.0 * (2.0 - emissivity));
    let g = ((1.0 - omega) / (3.0 - omega * a1)).sqrt();
    let h = ((1.0 - omega) * (3.0 - omega * a1)).sqrt() * tau / 2.0;
    4.0 * a * g * h.tanh() / (a + g * h.tanh())
}

/// Psi_disc per wall: the mean of `(q.n)_b` over that patch's faces,
/// divided by `sigma (T^4 - T_w^4)` - (135.15) against the slab's driving.
fn psi_walls(g: &Gpu, hm: &HostMesh, p1: &P1, t_k: f64, tw: f64) -> Result<(f64, f64)> {
    let q = g.download(p1.wall_flux())?;
    let side = |name: &str| -> Result<f64> {
        let p = hm
            .patches
            .iter()
            .find(|p| p.name == name)
            .ok_or_else(|| Error::Config(format!("the P1 slab has no {name} patch")))?;
        let sum: f64 = (p.start..p.start + p.size).map(|bf| w64(q[bf])).sum();
        Ok(sum / p.size.max(1) as f64
            / (SIGMA_SB as f64 * (t_k * t_k * t_k * t_k - tw * tw * tw * tw)))
    };
    Ok((side("xMin")?, side("xMax")?))
}

/// Solve one slab case at T = 1000 K with cold walls and return
/// `(converged, Psi_xMin, Psi_xMax)`.
fn solve_slab(
    g: &Gpu,
    n: usize,
    emissivity: f64,
    tau: f64,
    omega: f64,
    a1: f64,
) -> Result<(bool, f64, f64)> {
    let (hm, m) = p1_slab(g, n)?;
    let mut p1 =
        P1::new(g, &hm, &m, &p1_medium(tau, omega, a1), &walls(emissivity, 0.0), p1_controls())?;
    let t = uniform_t(g, &m, 1000.0)?;
    let perf = p1.solve(g, &m, &t)?;
    let psi = psi_walls(g, &hm, &p1, 1000.0, 0.0)?;
    Ok((perf.converged, psi.0, psi.1))
}

// ---------------------------------------------------------------------------
//  The DOM slab's helpers (tests/radiation_dom.rs 1280-1545)
// ---------------------------------------------------------------------------

/// The sweep tolerance of every DOM solve: 1e-12 in f64 - well below the
/// 1e-10 rows - and §136.7's `tol32` of 1e-5 under `single`.
const SWEEP_TOL: f64 = if cfg!(feature = "single") { 1e-5 } else { 1e-12 };

/// Every DOM solve's controls: PBiCGStab + diagonal at 1e-12 absolute, in
/// BOTH builds (under `single` the solve is §118.4-refined, as P1's), the
/// sweep at `SWEEP_TOL`, 200 sweeps at most.
fn dom_controls() -> DomControls {
    DomControls {
        linear: SolverControls {
            solver: LinearSolverKind::PBiCGStab,
            precon: Preconditioner::Diagonal,
            tolerance: 1e-12,
            rel_tol: 0.0,
            max_iter: 5000,
            ..SolverControls::default()
        },
        tol: SWEEP_TOL,
        max_sweeps: 200,
    }
}

/// The non-scattering DOM medium: `kappa = tau`, `sigma_s = 0`, `A_1 = 0`.
fn dom_medium(kappa: f64) -> DomMedium {
    DomMedium { kappa, sigma_s: 0.0, a1: 0.0 }
}

/// §136.7's slab: x in [0, 1] with `n` cells, y and z in [0, 1] with one
/// each; xMin and xMax walls, the four sides symmetry planes.
fn dom_slab(g: &Gpu, n: usize) -> Result<(HostMesh, GpuMesh)> {
    let axis = |n: usize| GradedAxis {
        lo: 0.0 as Scalar,
        hi: 1.0 as Scalar,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    let spec = BlockSpec {
        x: axis(n),
        y: axis(1),
        z: axis(1),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall", "wall", "symmetry", "symmetry", "symmetry", "symmetry"]
            .map(String::from),
        windows: Vec::new(),
        cyclic: Vec::new(),
    };
    let hm = blockgen::build_mesh(&spec)?;
    let m = GpuMesh::upload(g, &hm)?;
    Ok((hm, m))
}

/// One patch's mean wall flux over `sigma T^4` - `Psi` of (135.6).
fn patch_psi(hm: &HostMesh, wf: &[Scalar], name: &str, sigma_t4: f64) -> Result<f64> {
    let p = hm
        .patches
        .iter()
        .find(|p| p.name == name)
        .ok_or_else(|| Error::Config(format!("the slab names no {name} patch")))?;
    let sum: f64 = wf[p.start..p.start + p.size].iter().map(|&v| w64(v)).sum();
    Ok(sum / p.size as f64 / sigma_t4)
}

/// §136.7's slab as a [`DomSolve`] with `ctrl`: black walls at 0 K,
/// `kappa = tau`, T = 1000 K.
fn slab_case(
    g: &Gpu,
    order: SnOrder,
    med: &DomMedium,
    n: usize,
    ctrl: DomControls,
) -> Result<(HostMesh, GpuMesh, GpuScalarField, DomSolve)> {
    let (hm, m) = dom_slab(g, n)?;
    let quad = Quadrature::new(order)?;
    let black = || DomCondition::GreyDiffuse {
        emissivity: 1.0,
        temperature: WallTemperature::Fixed(0.0),
    };
    let ps = vec![
        DomPatch { patch: "xMin".into(), condition: black() },
        DomPatch { patch: "xMax".into(), condition: black() },
        DomPatch { patch: "yMin".into(), condition: DomCondition::Symmetry },
        DomPatch { patch: "yMax".into(), condition: DomCondition::Symmetry },
        DomPatch { patch: "zMin".into(), condition: DomCondition::Symmetry },
        DomPatch { patch: "zMax".into(), condition: DomCondition::Symmetry },
    ];
    let t = uniform_t(g, &m, 1000.0)?;
    let dom = DomSolve::new(g, &hm, &m, med, &quad, &ps, ctrl)?;
    Ok((hm, m, t, dom))
}

/// (136.12) of SPEC-LIT 136.7: the step scheme's slab on `n` cells with
/// this set, in f64, over the directions into xMin;
/// `-expm1(-n ln1p(.))` (§136.7).
fn psi_n(quad: &Quadrature, tau: f64, n: usize) -> f64 {
    let nf = n as f64;
    let mut s = 0.0f64;
    for (l, d) in quad.dirs().iter().enumerate() {
        if d[0] < 0.0 {
            let mu = -d[0];
            s += quad.weights()[l] * mu * -(-nf * (tau / (nf * mu)).ln_1p()).exp_m1();
        }
    }
    s / PI
}

/// (136.13): the set's own spatially converged slab value.
fn psi_limit(quad: &Quadrature, tau: f64) -> f64 {
    let mut s = 0.0f64;
    for (l, d) in quad.dirs().iter().enumerate() {
        if d[0] < 0.0 {
            let mu = -d[0];
            s += quad.weights()[l] * mu * -(-tau / mu).exp_m1();
        }
    }
    s / PI
}

/// Solve §136.7's slab with `order` at `tau` on `n` cells and emit its
/// rows: the sweeps converge with every direction solve converged
/// (`require`), and BOTH walls are the step recurrence (136.12) of
/// SPEC-LIT 136.7 at round-off (`check_roundoff`, 1e-10). Returns
/// `(Psi_xMin, Psi_xMax, sweeps)`.
fn slab_psi(
    c: &mut Checks,
    g: &Gpu,
    order: SnOrder,
    tau: f64,
    n: usize,
) -> Result<(f64, f64, usize)> {
    let (hm, m, t, mut dom) = slab_case(g, order, &dom_medium(tau), n, dom_controls())?;
    let quad = dom.rows().quadrature().clone();
    let rep = dom.solve(g, &m, &t)?;
    let what = format!("{order:?} tau {tau} N {n}");
    c.require(
        &format!("Gate CHR-SLAB: {what} - the sweeps converge and every direction solve converges (SPEC-LIT 136.7)"),
        rep.converged && rep.unconverged_solves == 0,
    );
    let wf = g.download(dom.wall_flux())?;
    let sigma_t4 = SIGMA_SB as f64 * 1000f64.powi(4);
    let lo = patch_psi(&hm, &wf, "xMin", sigma_t4)?;
    let hi = patch_psi(&hm, &wf, "xMax", sigma_t4)?;
    let want = psi_n(&quad, tau, n);
    for (wall, v) in [("xMin", lo), ("xMax", hi)] {
        let r = (v / want - 1.0).abs();
        c.check_roundoff(
            &format!("Gate CHR-SLAB: {what} {wall} against (136.12) of SPEC-LIT 136.7"),
            r as Scalar,
            1e-10,
        );
    }
    Ok((lo, hi, rep.sweeps))
}

/// One sweep of the N = 256 slab, returning the unconverged direction
/// solves: under `single` this is the f32 PBiCGStab breakdown of
/// SPEC-LIT 136.7 - BiCGStab needs about N iterations on the 256-cell
/// upwind chain, and in f32 it ends in NaN or diverges from N = 64 up.
fn slab_f32_sweep(g: &Gpu, tau: f64) -> Result<usize> {
    let ctrl = DomControls { max_sweeps: 1, ..dom_controls() };
    let (_hm, m, t, mut dom) = slab_case(g, SnOrder::S8, &dom_medium(tau), 256, ctrl)?;
    let rep = dom.solve(g, &m, &t)?;
    Ok(rep.unconverged_solves)
}

/// **Gate CHR-SLAB** (SPEC-LIT §135.5, §136.7): the isothermal grey slab,
/// live, in both builds.
pub(crate) fn check_chr_slab(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    // ---- P1 against its closed form (tests/radiation_p1.rs) ----------------
    // (eps, tau, omega, A1, N_coarse, N_fine): §135.5's seven cases. The
    // meshes are powers of two so dx is a binary fraction - exact under
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
    // §135.5's Psi_P1 per case, in CASES' order - the closed form's table.
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
        c.check_a(
            &format!("Gate CHR-SLAB: (135.11) at eps {eps} tau {tau} omega {om} against SPEC-LIT 135.5's table value"),
            (want - TABLE[ci]).abs() as Scalar,
            5e-13,
        );
        let mut es = [0.0f64; 2];
        let mut converged = true;
        for (mi, &n) in [nc, nf].iter().enumerate() {
            let (ok, x_min, x_max) = solve_slab(gpu, n, eps, tau, om, a1)?;
            let e_min = x_min / want - 1.0;
            let e_max = x_max / want - 1.0;
            // The reported error is the larger of the two walls', the
            // conservative one for the floor rule below.
            es[mi] = if e_min.abs() >= e_max.abs() { e_min } else { e_max };
            converged &= ok;
        }
        let p = (es[0].abs() / es[1].abs()).log2();
        c.require(
            &format!("Gate CHR-SLAB: eps {eps} tau {tau} omega {om} - the PCG solve converges on N {nc} and N {nf} (SPEC-LIT 135.7)"),
            converged,
        );
        c.require(
            &format!("Gate CHR-SLAB: eps {eps} tau {tau} - the error shrinks from N {nc} to N {nf}, same sign (SPEC-LIT 135.7)"),
            es[0].signum() == es[1].signum() && es[1].abs() < es[0].abs(),
        );
        c.check_p(
            &format!("Gate CHR-SLAB: eps {eps} tau {tau} omega {om} - the observed order against 2.0 (SPEC-LIT 135.7)"),
            (p - 2.0).abs() as Scalar,
            0.1,
        );
        c.require(
            &format!("Gate CHR-SLAB: eps {eps} tau {tau} - the fine error is at least 1e3 2^-24, discretisation not round-off (SPEC-LIT 135.6)"),
            es[1].abs() >= 1e3 * 2.0f64.powi(-24),
        );
    }

    // ---- the E_3 oracle's own check (both builds, host f64) ----------------
    for (tau, want) in
        [(0.1, 0.167417084183), (1.0, 0.780616065604), (10.0, 0.999992902475)]
    {
        let v = psi_exact(tau);
        c.check_a(
            &format!("Gate CHR-SLAB: 1 - 2 E_3({tau}) against SPEC-LIT 136.7's oracle value"),
            (v - want).abs() as Scalar,
            1e-11,
        );
    }

    if cfg!(feature = "single") {
        // Under `single` the N = 256 slab is the recorded f32 miss of
        // SPEC-LIT 136.7: the crate's float PBiCGStab breaks down on the
        // 256-cell upwind chain, so the direction solves report no
        // convergence. One sweep shows it; the rows carry the counts.
        let mut counts = [0usize; 3];
        for (i, tau) in [0.1, 1.0, 10.0].into_iter().enumerate() {
            counts[i] = slab_f32_sweep(gpu, tau)?;
            c.check_d(
                &format!("Gate CHR-SLAB: S8 N 256 tau {tau} direction solves converge under single (SPEC-LIT 136.7)"),
                counts[i] as Scalar,
                0.0,
            );
        }
        c.report(GateReport {
            verdict: Verdict::Misses,
            how: How::Live,
            gate: GATE,
            against: "the exact slab 1 - 2 E_3(tau), (135.6) of SPEC-LIT 135.2",
            headline: format!(
                "under single the N = 256 slab does not converge: {} of the direction \
                 solves unconverged in one sweep at tau 0.1, {} at tau 1, {} at tau 10 - \
                 the f32 PBiCGStab breaks down on the 256-cell upwind chain",
                counts[0], counts[1], counts[2]
            ),
            detail: vec![
                "the f64 build holds every N = 256 row; BiCGStab needs about N iterations \
                 on the upwind chain and in f32 it ends in NaN or diverges from N = 64 up \
                 (SPEC-LIT 136.7)"
                    .to_string(),
            ],
            uncertainty: Some(Uncertainty::SingleMesh(
                "N = 256, one sweep, the f32 miss shown as it stands; no mesh study under single",
            )),
        });
        return Ok(());
    }

    // ---- the f64 DOM slab: S8 N = 256 against (135.6) at tau 1 and 10 ------
    let quad = Quadrature::new(SnOrder::S8)?;
    for tau in [1.0, 10.0] {
        let (lo, hi, _sweeps) = slab_psi(c, gpu, SnOrder::S8, tau, 256)?;
        let ex = psi_exact(tau);
        for (wall, v) in [("xMin", lo), ("xMax", hi)] {
            let e = v / ex - 1.0;
            c.check_p(
                &format!("Gate CHR-SLAB: S8 N 256 tau {tau} {wall} within 2 % of 1 - 2 E_3 (SPEC-LIT 136.7)"),
                e.abs() as Scalar,
                0.02,
            );
        }
    }

    // ---- tau = 0.1: the recorded miss, asserted with its cause -------------
    let tau = 0.1;
    let ex = psi_exact(tau);
    let (lo256, hi256, _s256) = slab_psi(c, gpu, SnOrder::S8, tau, 256)?;
    let (lo32, hi32, _s32) = slab_psi(c, gpu, SnOrder::S8, tau, 32)?;
    let e_lim = psi_limit(&quad, tau) / ex - 1.0;
    // The cause row that does not depend on the wall: S8's angular limit
    // alone exceeds 2 %.
    c.require(
        &format!("Gate CHR-SLAB: tau {tau} - the cause: S8's angular limit (136.13) exceeds 2 % (SPEC-LIT 136.7)"),
        e_lim > 0.02,
    );
    let mut worst256 = 0.0f64;
    let mut e32_worst = 0.0f64;
    for (wall, v256, v32) in [("xMin", lo256, lo32), ("xMax", hi256, hi32)] {
        let (e256, e32) = (v256 / ex - 1.0, v32 / ex - 1.0);
        c.note(&format!(
            "Gate CHR-SLAB S8 tau {tau} {wall}: e_256 {e256:+.4e}, e_32 {e32:+.4e}, e_limit (136.13) {e_lim:+.4e}"
        ));
        // The band row itself - it fails, and that is the recorded miss.
        c.check_p(
            &format!("Gate CHR-SLAB: S8 N 256 tau {tau} {wall} within 2 % of 1 - 2 E_3 (SPEC-LIT 136.7)"),
            e256.abs() as Scalar,
            0.02,
        );
        // The cause rows: the step scheme approaches S8's limit from below
        // without passing it.
        c.require(
            &format!("Gate CHR-SLAB: S8 N 256 tau {tau} {wall} e_256 is the recorded miss, outside 2 % (SPEC-LIT 136.7)"),
            e256 > 0.02,
        );
        c.require(
            &format!("Gate CHR-SLAB: S8 tau {tau} {wall} - the step scheme approaches the limit from below, e_32 < e_256 (SPEC-LIT 136.7)"),
            e32 < e256,
        );
        c.require(
            &format!("Gate CHR-SLAB: S8 tau {tau} {wall} - e_256 must not pass e_limit (SPEC-LIT 136.7)"),
            e256 <= e_lim,
        );
        worst256 = worst256.max(e256);
        e32_worst = e32_worst.max(e32);
    }
    c.report(GateReport {
        verdict: Verdict::Misses,
        how: How::Live,
        gate: GATE,
        against: "the exact slab 1 - 2 E_3(tau), (135.6) of SPEC-LIT 135.2",
        headline: format!(
            "S8 N 256 tau 0.1: e_256 {worst256:+.4e} outside 2 % on both walls, S8's own \
             angular limit (136.13) e_lim {e_lim:+.4e} above 2 %",
        ),
        detail: vec![
            format!(
                "the cause: S8's angular error alone exceeds 2 % at tau 0.1 - the step \
                 scheme approaches that limit from below (e_32 {e32_worst:+.4e} < e_256 <= \
                 e_limit), so refining the mesh cannot bring the row inside 2 % (SPEC-LIT \
                 136.7); accepted by the user on 2026-10-14"
            ),
        ],
        uncertainty: Some(Uncertainty::SingleMesh(
            "N = 256 and N = 32 as the slab runs them; no three-mesh study",
        )),
    });
    Ok(())
}
