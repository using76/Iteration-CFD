// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate CHR-WSGG (SPEC-LIT §136.9): the weighted sum of grey gases on the
//! homogeneous isothermal slab, and the grey mean, run live.**
//!
//! A live port of this repository's own tests into `ofgpu-validate` rows,
//! one row per assertion, the tests staying as they are:
//! `tests/radiation_wsgg.rs::wsgg_slab_sum_of_grey_gases` - on the
//! synthetic four-gas set, the non-grey fvDOM loop's wall flux is
//! `sum_j a_j Psi_N(kappa_j)` (136.12) at round-off and lands within the
//! method's own grey-slab error of `sum_j a_j (1 - 2 E_3(kappa_j))`, and
//! the non-grey P1 loop's `G`, `q_r`, wall flux and source pair are
//! `sum_j a_j` times the fresh grey P1 solve at `kappa_j` to round-off,
//! its wall flux within the grey solves' own error of `sum_j a_j`
//! (135.11) - and `wsgg_grey_mean` - `L_m = 3.6 V/A` on the cube and the
//! two slabs, the grey `kappa` reproducing `eps(L_m)` exactly, and the
//! refusals. Smith 1982 and Bordbar 2014 are NAMED, NOT RUN: their tables
//! are not vendored, the gate is the synthetic set (DESIGN, §136.4).
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §136.4 (the WSGG model and its synthetic-set
//!     licence rule), §136.5 (the rows), §136.9 (the per-gas solve and
//!     this gate)
//!   tests/radiation_wsgg.rs (`wsgg_slab_sum_of_grey_gases`,
//!     `wsgg_grey_mean` and their helpers) - the tests stay as they are
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::energy::EnergySources;
use ofgpu::radiation::dom::{DomCondition, DomControls, DomPatch};
use ofgpu::radiation::dom_quad::{Quadrature, SnOrder};
use ofgpu::radiation::p1::{MarshakPatch, P1, P1Medium, WallTemperature};
use ofgpu::radiation::wsgg::{mean_beam_length, GreyGas, GreyGasSet, WsggDom, WsggP1};
use ofgpu::radiation::SIGMA_SB;
use ofgpu::solver::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::{Gpu, GpuMesh, GpuScalarField, HostMesh, Result, Scalar};

use super::chr_fixture::{e3, w64};
use crate::Checks;

/// The gate's scope and registry name.
pub(crate) const GATE: &str = "SPEC-LIT 136.9 Gate CHR-WSGG weighted sum of grey gases";

const PI: f64 = std::f64::consts::PI;

/// The stated synthetic four-gas set of SPEC-LIT §136.9 (DESIGN, not
/// Bordbar's): weights 0.30, 0.25, 0.20, 0.15 (window 0.10) and kappa 0.1,
/// 0.5, 2, 10 1/m - optically thin to thick on the unit slab.
fn synthetic_set() -> Result<GreyGasSet> {
    GreyGasSet::new(
        [(0.30, 0.1), (0.25, 0.5), (0.20, 2.0), (0.15, 10.0)]
            .iter()
            .map(|&(a, kappa)| GreyGas { a, kappa })
            .collect(),
    )
}

/// (136.12) of SPEC-LIT 136.7: the step scheme's slab on `n` cells with
/// this set, in f64.
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

/// (135.11), P1's slab closed form: black walls (`a = 1/2`), no scattering.
fn psi_p1(tau: f64) -> f64 {
    let a = 0.5f64;
    let g = (1.0f64 / 3.0).sqrt();
    let h = 3f64.sqrt() * tau / 2.0;
    4.0 * a * g * h.tanh() / (a + g * h.tanh())
}

fn axis(hi: f64, n: usize) -> GradedAxis {
    GradedAxis { lo: 0.0 as Scalar, hi: hi as Scalar, n, expansion: 1.0, two_sided: false }
}

fn mesh(g: &Gpu, spec: &BlockSpec) -> Result<(HostMesh, GpuMesh)> {
    let hm = blockgen::build_mesh(spec)?;
    let m = GpuMesh::upload(g, &hm)?;
    Ok((hm, m))
}

/// CHR-20's DOM slab: x in [0, 1] with `n` cells, y and z in [0, 1] with
/// one each; xMin and xMax walls, the four sides symmetry planes.
fn dom_slab_spec(n: usize) -> BlockSpec {
    BlockSpec {
        x: axis(1.0, n),
        y: axis(1.0, 1),
        z: axis(1.0, 1),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall", "wall", "symmetry", "symmetry", "symmetry", "symmetry"]
            .map(String::from),
        windows: Vec::new(),
        cyclic: Vec::new(),
    }
}

/// CHR-18's P1 slab: x in [0, 1] with `n` cells, y and z in [0, 0.125]
/// (binary fractions, exact under `single`); Marshak walls at xMin and
/// xMax, the other four slots empty.
fn p1_slab_spec(n: usize) -> BlockSpec {
    BlockSpec {
        x: axis(1.0, n),
        y: axis(0.125, 1),
        z: axis(0.125, 1),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall", "wall", "empty", "empty", "empty", "empty"].map(String::from),
        windows: Vec::new(),
        cyclic: Vec::new(),
    }
}

fn uniform_t(g: &Gpu, m: &GpuMesh, t_k: f64) -> Result<GpuScalarField> {
    let mut t = GpuScalarField::zeros(g, m, "T")?;
    g.write(&mut t.f, &vec![t_k as Scalar; m.n_cells])?;
    g.write(&mut t.bf, &vec![t_k as Scalar; m.n_boundary_faces])?;
    Ok(t)
}

/// One patch's mean wall flux over `scale`.
fn patch_mean(hm: &HostMesh, wf: &[Scalar], name: &str, scale: f64) -> Result<f64> {
    let p = hm
        .patches
        .iter()
        .find(|p| p.name == name)
        .ok_or_else(|| ofgpu::Error::Config(format!("the slab names no {name} patch")))?;
    let sum: f64 = wf[p.start..p.start + p.size].iter().map(|&v| w64(v)).sum();
    Ok(sum / p.size as f64 / scale)
}

/// `max_k |x_k - want_k| / max_k |want_k|`.
fn rel_max(x: &[f64], want: &[f64]) -> f64 {
    let d = x.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
    let s = want.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    if s == 0.0 {
        d
    } else {
        d / s
    }
}

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
        tol: if cfg!(feature = "single") { 1e-5 } else { 1e-12 },
        max_sweeps: 200,
    }
}

fn p1_controls() -> SolverControls {
    SolverControls {
        solver: LinearSolverKind::PCG,
        tolerance: 1e-12,
        rel_tol: 0.0,
        max_iter: 5000,
        ..SolverControls::default()
    }
}

/// **Gate CHR-WSGG** (SPEC-LIT §136.9): the weighted sum of grey gases,
/// live, in both builds.
pub(crate) fn check_chr_wsgg(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    c.note(
        "Smith 1982 and Bordbar 2014 are NAMED, NOT RUN: their tables are not vendored (SPEC-LIT 136.9); the gate is the synthetic four-gas set",
    );
    let set = synthetic_set()?;
    c.check_a(
        "Gate CHR-WSGG: the synthetic set's window is 0.10 exactly (SPEC-LIT 136.9)",
        (set.window() - 0.10).abs() as Scalar,
        1e-15,
    );
    let t_k = 1000.0f64;
    // The test's own row tolerances, as its consts have them - used only
    // inside these two band bounds; the pure round-off rows are
    // `check_roundoff` at their f64 value and take the class's f32 rule.
    let tol = if cfg!(feature = "single") { 1e-5 } else { 1e-12 };
    let tol_r10 = if cfg!(feature = "single") { 1e-5 } else { 1e-10 };

    // ---- fvDOM: the loop is the per-gas identity ---------------------------
    let n = 32;
    let (hm, m) = mesh(gpu, &dom_slab_spec(n))?;
    let quad = Quadrature::new(SnOrder::S8)?;
    let black = || DomCondition::GreyDiffuse {
        emissivity: 1.0,
        temperature: WallTemperature::Fixed(0.0),
    };
    let mut ps = vec![
        DomPatch { patch: "xMin".into(), condition: black() },
        DomPatch { patch: "xMax".into(), condition: black() },
    ];
    for s in ["yMin", "yMax", "zMin", "zMax"] {
        ps.push(DomPatch { patch: s.into(), condition: DomCondition::Symmetry });
    }
    let t = uniform_t(gpu, &m, t_k)?;
    let mut wd = WsggDom::new(gpu, &hm, &m, set.clone(), 0.0, 0.0, &quad, &ps, dom_controls())?;
    let reps = wd.solve(gpu, &m, &t)?;
    for (j, r) in reps.iter().enumerate() {
        c.require(
            &format!("Gate CHR-WSGG: DOM gas {} - the sweeps converge and every direction solve converges (SPEC-LIT 136.9)", j + 1),
            r.converged && r.unconverged_solves == 0,
        );
    }
    let wf = gpu.download(wd.wall_flux())?;
    let sigma_t4 = w64(SIGMA_SB) * t_k.powi(4);
    let want_disc: f64 = set.gases().iter().map(|gs| gs.a * psi_n(&quad, gs.kappa, n)).sum();
    let want_exact: f64 = set.gases().iter().map(|gs| gs.a * (1.0 - 2.0 * e3(gs.kappa))).sum();
    let method: f64 = set
        .gases()
        .iter()
        .map(|gs| gs.a * (psi_n(&quad, gs.kappa, n) - (1.0 - 2.0 * e3(gs.kappa))).abs())
        .sum();
    for wall in ["xMin", "xMax"] {
        let psi = patch_mean(&hm, &wf, wall, sigma_t4)?;
        let r = (psi / want_disc - 1.0).abs();
        let e = psi - want_exact;
        c.check_roundoff(
            &format!("Gate CHR-WSGG: DOM S8 N {n} {wall} against sum_j a_j Psi_N(kappa_j), (136.12) per gas (SPEC-LIT 136.9)"),
            r as Scalar,
            1e-10,
        );
        // The band is the method's own grey-slab error at this mesh and
        // set, plus the round-off of the identity - the test's own bound.
        c.check_p(
            &format!("Gate CHR-WSGG: DOM S8 N {n} {wall} within the method's own grey-slab error of sum_j a_j (1 - 2 E_3(kappa_j)) (SPEC-LIT 136.9)"),
            e.abs() as Scalar,
            (method + tol_r10 * want_exact) as Scalar,
        );
    }

    // ---- P1: the loop is sum_j a_j times the fresh grey solve --------------
    let tw = 500.0f64;
    let (hm, m) = mesh(gpu, &p1_slab_spec(n))?;
    let walls: Vec<MarshakPatch> = ["xMin", "xMax"]
        .iter()
        .map(|p| MarshakPatch {
            patch: (*p).into(),
            emissivity: 1.0,
            temperature: WallTemperature::Fixed(tw),
        })
        .collect();
    let t = uniform_t(gpu, &m, t_k)?;
    let mut wp = WsggP1::new(gpu, &hm, &m, set.clone(), 0.0, 0.0, &walls, p1_controls())?;
    let mut src = EnergySources::new(gpu, &m)?;
    src.clear(gpu)?;
    let perfs = wp.solve(gpu, &m, &t, Some(&mut src))?;
    for (j, p) in perfs.iter().enumerate() {
        c.require(
            &format!("Gate CHR-WSGG: P1 gas {} - the PCG solve converges (SPEC-LIT 136.9)", j + 1),
            p.converged,
        );
    }
    let nc = hm.n_cells;
    let nb = hm.n_boundary_faces;
    let (mut rg, mut rq, mut rw) = (vec![0.0; nc], vec![0.0; 3 * nc], vec![0.0; nb]);
    let (mut rs_q, mut rs_sp) = (vec![0.0; nc], vec![0.0; nc]);
    let drive = w64(SIGMA_SB) * (t_k.powi(4) - tw.powi(4));
    let (mut psi_disc, mut psi_ex, mut p1_method) = (0.0f64, 0.0f64, 0.0f64);
    for gs in set.gases() {
        let med = P1Medium { kappa: gs.kappa, sigma_s: 0.0, a1: 0.0 };
        let mut p1 = P1::new(gpu, &hm, &m, &med, &walls, p1_controls())?;
        let grey_ok = p1.solve(gpu, &m, &t)?.converged;
        c.require(
            &format!("Gate CHR-WSGG: the grey P1 at kappa {} converges (SPEC-LIT 136.9)", gs.kappa),
            grey_ok,
        );
        let mut s1 = EnergySources::new(gpu, &m)?;
        s1.clear(gpu)?;
        p1.register_energy_source(gpu, &t, &mut s1)?;
        let gg = gpu.download(&p1.g().f)?;
        let qq = gpu.download(p1.qr())?;
        let ww = gpu.download(p1.wall_flux())?;
        let sq = gpu.download(s1.q())?;
        let ssp = gpu.download(s1.sp())?;
        for c_i in 0..nc {
            rg[c_i] += gs.a * w64(gg[c_i]);
            rq[3 * c_i] += gs.a * w64(qq[c_i].x);
            rq[3 * c_i + 1] += gs.a * w64(qq[c_i].y);
            rq[3 * c_i + 2] += gs.a * w64(qq[c_i].z);
            rs_q[c_i] += gs.a * w64(sq[c_i]);
            rs_sp[c_i] += gs.a * w64(ssp[c_i]);
        }
        for f in 0..nb {
            rw[f] += gs.a * w64(ww[f]);
        }
        let pj = patch_mean(&hm, &ww, "xMin", drive)?;
        psi_disc += gs.a * pj;
        psi_ex += gs.a * psi_p1(gs.kappa);
        p1_method += gs.a * (pj - psi_p1(gs.kappa)).abs();
    }
    let gl: Vec<f64> = gpu.download(wp.g())?.iter().map(|&v| w64(v)).collect();
    let ql: Vec<f64> = gpu
        .download(wp.qr())?
        .iter()
        .flat_map(|v| [w64(v.x), w64(v.y), w64(v.z)])
        .collect();
    let wl: Vec<f64> = gpu.download(wp.wall_flux())?.iter().map(|&v| w64(v)).collect();
    let sql: Vec<f64> = gpu.download(src.q())?.iter().map(|&v| w64(v)).collect();
    let sspl: Vec<f64> = gpu.download(src.sp())?.iter().map(|&v| w64(v)).collect();
    for (what, x, want) in [
        ("G", &gl, &rg),
        ("q_r", &ql, &rq),
        ("wall flux", &wl, &rw),
        ("source q", &sql, &rs_q),
        ("source sp", &sspl, &rs_sp),
    ] {
        let r = rel_max(x, want);
        c.check_roundoff(
            &format!("Gate CHR-WSGG: P1 N {n} {what} against sum_j a_j x the grey P1 solve (SPEC-LIT 136.9)"),
            r as Scalar,
            1e-12,
        );
    }
    let wf = gpu.download(wp.wall_flux())?;
    for wall in ["xMin", "xMax"] {
        let psi = patch_mean(&hm, &wf, wall, drive)?;
        let e = psi - psi_ex;
        c.note(&format!(
            "Gate CHR-WSGG P1 N {n} {wall}: Psi {psi:.12}, sum a_j Psi_grey {psi_disc:.12}, sum a_j (135.11) \
             {psi_ex:.12}, e {e:+.4e} against the grey solves' own {p1_method:.4e}"
        ));
        c.check_p(
            &format!("Gate CHR-WSGG: P1 N {n} {wall} within the grey solves' own error of sum_j a_j (135.11) (SPEC-LIT 136.9)"),
            e.abs() as Scalar,
            (p1_method + tol * psi_ex) as Scalar,
        );
    }

    // ---- the grey mean (host only) -----------------------------------------
    let cube = BlockSpec {
        x: axis(1.0, 2),
        y: axis(1.0, 2),
        z: axis(1.0, 2),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall", "wall", "wall", "wall", "patch", "patch"].map(String::from),
        windows: Vec::new(),
        cyclic: Vec::new(),
    };
    let lm = mean_beam_length(&blockgen::build_mesh(&cube)?)?;
    c.check_a(
        "Gate CHR-WSGG: the unit cube's mean beam length is 0.6 m (SPEC-LIT 136.9)",
        (lm - 0.6).abs() as Scalar,
        1e-15,
    );
    let lm_slab = mean_beam_length(&blockgen::build_mesh(&p1_slab_spec(4))?)?;
    c.check_a(
        "Gate CHR-WSGG: the P1 slab's (empty sides) mean beam length is 1.8 m (SPEC-LIT 136.9)",
        (lm_slab - 1.8).abs() as Scalar,
        1e-14,
    );
    let lm_sym = mean_beam_length(&blockgen::build_mesh(&dom_slab_spec(4))?)?;
    c.check_a(
        "Gate CHR-WSGG: the DOM slab's (symmetry sides) mean beam length is 1.8 m (SPEC-LIT 136.9)",
        (lm_sym - 1.8).abs() as Scalar,
        1e-14,
    );
    for l in [0.6, 1.8] {
        let k = set.grey_kappa(l)?;
        let eps = set.emissivity(l);
        let back = -(-k * l).exp_m1();
        c.check_a(
            &format!("Gate CHR-WSGG: L_m {l} - the grey kappa reproduces eps(L_m) exactly, 1 - exp(-kappa L_m) against eps (SPEC-LIT 136.9)"),
            (back / eps - 1.0).abs() as Scalar,
            1e-14,
        );
    }
    c.require(
        "Gate CHR-WSGG: L_m = 0 is refused, naming the mean beam length (SPEC-LIT 136.9)",
        refusal_text(set.grey_kappa(0.0)).contains("mean beam length"),
    );
    let over = GreyGasSet::new(vec![GreyGas { a: 1.2, kappa: 10.0 }]);
    c.require(
        "Gate CHR-WSGG: an eps outside (0, 1) is refused, naming the range (SPEC-LIT 136.9)",
        refusal_text(over.and_then(|s| s.grey_kappa(1.0))).contains("outside (0, 1)"),
    );
    c.require(
        "Gate CHR-WSGG: an empty set is refused, naming no grey gas (SPEC-LIT 136.9)",
        refusal_text(GreyGasSet::new(Vec::new())).contains("no grey gas"),
    );
    c.require(
        "Gate CHR-WSGG: a window as a gas is refused, naming the window (SPEC-LIT 136.9)",
        refusal_text(GreyGasSet::new(vec![GreyGas { a: 0.5, kappa: 0.0 }])).contains("window"),
    );
    Ok(())
}

/// A refusal's message, or empty when the call succeeded - the gate's
/// `require` rows match on the name the refusal gives, the way
/// `tests/radiation_wsgg.rs`'s `refused` does.
fn refusal_text<T: std::fmt::Debug>(r: Result<T>) -> String {
    match r {
        Ok(_) => String::new(),
        Err(e) => format!("{e}"),
    }
}
