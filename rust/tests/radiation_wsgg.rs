// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! CHR-22's gates for the weighted sum of grey gases (SPEC-LIT §136.4,
//! §136.5, §136.9): the non-grey fvDOM and P1 loops on the homogeneous
//! isothermal slab against the per-gas identity `sum_j a_j (1 - 2E_3(kappa_j
//! L))` (`wsgg_slab_sum_of_grey_gases`), the H2O/CO2 refusals
//! (`wsgg_requires_h2o_co2`), the fitted-range refusals
//! (`wsgg_refuses_out_of_range`), the user file's digest check
//! (`wsgg_user_file_digest`), and the grey mean (`wsgg_grey_mean`).
//!
//! The gas set and the coefficient files written here are SYNTHETIC
//! (DESIGN, §136.4): Bordbar 2021's data may not be distributed, so the
//! real-data rows run only when `OFGPU_WSGG_DIR` names a folder holding the
//! user's own copy of the ten `.dat` files, and are skipped by name
//! otherwise.
//!
//! The SAME assertions in BOTH builds - the host evaluation is f64 in both,
//! the device rows run at the R class of §136.5 (`tol32` under `single`).
//! An integration test, because the library's own test target does not
//! compile under `single`. A machine without a card passes the device rows
//! vacuously, the convention of `tests/radiation_p1.rs`.
//!
//! Provenance: ORIGINAL - the meshes, the sets, the files and the
//! assertions are this tree's own construction, run against (136.7)-(136.10),
//! (136.12), (135.6) and (135.11) with Abramowitz & Stegun 5.1.11, 5.1.14 and
//! 5.1.22 for `E_n`, and the FIPS 180-4 example digests (NIST, US Government
//! work), via SPEC-LIT §135-§136.
//! No GPL-licensed source was consulted.

use std::path::PathBuf;

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::energy::EnergySources;
use ofgpu::radiation::dom::{DomCondition, DomControls, DomPatch};
use ofgpu::radiation::dom_quad::{Quadrature, SnOrder};
use ofgpu::radiation::p1::{MarshakPatch, P1Medium, WallTemperature, P1};
use ofgpu::radiation::wsgg::{
    mean_beam_length, require_h2o_co2, sha256_hex, GreyGas, GreyGasSet, WsggCoefficients,
    WsggDom, WsggP1, WSGG_DIGESTS,
};
use ofgpu::radiation::SIGMA_SB;
use ofgpu::solver::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::{Gpu, GpuMesh, GpuScalarField, HostMesh, Result, Scalar};

/// The R-class tolerance of docs/17 §5.1 D-F32-1 for a `1e-12` f64 row.
const TOL: f64 = if cfg!(feature = "single") { 1e-5 } else { 1e-12 };

/// The DOM slab's row against (136.12): 1e-10, as CHR-20's
/// `dom_slab_is_the_step_recurrence` (`tol32` under `single`).
const TOL_R10: f64 = if cfg!(feature = "single") { 1e-5 } else { 1e-10 };

const PI: f64 = std::f64::consts::PI;

fn gpu() -> Option<Gpu> {
    Gpu::new(0).ok()
}

#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    x as f64
}

/// The stated synthetic four-gas set of SPEC-LIT §136.9 (DESIGN, not
/// Bordbar's): weights 0.30, 0.25, 0.20, 0.15 (window 0.10) and kappa 0.1,
/// 0.5, 2, 10 1/m - optically thin to thick on the unit slab.
fn synthetic_set() -> GreyGasSet {
    GreyGasSet::new(
        [(0.30, 0.1), (0.25, 0.5), (0.20, 2.0), (0.15, 10.0)]
            .iter()
            .map(|&(a, kappa)| GreyGas { a, kappa })
            .collect(),
    )
    .expect("the synthetic set is valid")
}

/// `E_1(x)`, x > 0: A&S 5.1.11's series for x <= 1, the continued fraction
/// 5.1.22 (backward, 200 levels) for x > 1 - `tests/radiation_dom.rs`'s.
fn e1(x: f64) -> f64 {
    if x <= 1.0 {
        let euler = 0.577_215_664_901_532_9_f64;
        let (mut sum, mut term) = (0.0f64, 1.0f64);
        for k in 1..60 {
            let kf = k as f64;
            term *= -x / kf;
            sum -= term / kf;
        }
        -euler - x.ln() + sum
    } else {
        let mut f = x;
        for k in (1..=200).rev() {
            let kf = k as f64;
            f = x + kf / (1.0 + kf / f);
        }
        (-x).exp() / f
    }
}

/// `E_3(x)` by A&S 5.1.14 twice.
fn e3(x: f64) -> f64 {
    let ex = (-x).exp();
    let e2 = ex - x * e1(x);
    (ex - x * e2) / 2.0
}

/// (136.12): the step scheme's slab on `n` cells with this set, in f64.
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
fn patch_mean(hm: &HostMesh, wf: &[Scalar], name: &str, scale: f64) -> f64 {
    let p = hm.patches.iter().find(|p| p.name == name).expect("the slab names its walls");
    let sum: f64 = wf[p.start..p.start + p.size].iter().map(|&v| w64(v)).sum();
    sum / p.size as f64 / scale
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

/// `wsgg_slab_sum_of_grey_gases` (§136.5, class P / R) - the homogeneous
/// isothermal slab, the synthetic set, T = 1000 K, `L = 1`:
///
/// * fvDOM S8 on N = 32, cold black walls: the loop's wall flux is
///   `sum_j a_j Psi_N(kappa_j)` (136.12) at 1e-10, and its error against
///   `sum_j a_j (1 - 2E_3(kappa_j))` is within the method's own grey-slab
///   error at the same mesh and set, `sum_j a_j |Psi_N(kappa_j) - (1 -
///   2E_3(kappa_j))|`.
/// * P1 on N = 32, black walls at 500 K, each gas registering its energy
///   source: `G`, `q_r`, the wall flux and the source pair are `sum_j a_j`
///   times the fresh grey P1 solve at `kappa_j` to round-off (`TOL`), and
///   the wall flux's error against `sum_j a_j` (135.11) is within the
///   same sum of the grey solves' own errors.
#[test]
fn wsgg_slab_sum_of_grey_gases() -> Result<()> {
    let set = synthetic_set();
    assert!((set.window() - 0.10).abs() <= 1e-15, "window {}", set.window());
    let Some(g) = gpu() else { return Ok(()) };
    let t_k = 1000.0f64;

    // ---- fvDOM ----
    let n = 32;
    let (hm, m) = mesh(&g, &dom_slab_spec(n))?;
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
    let t = uniform_t(&g, &m, t_k)?;
    let mut wd = WsggDom::new(&g, &hm, &m, set.clone(), 0.0, 0.0, &quad, &ps, dom_controls())?;
    let reps = wd.solve(&g, &m, &t)?;
    for (j, r) in reps.iter().enumerate() {
        assert!(r.converged, "DOM gas {}: the sweeps must converge: {r:?}", j + 1);
        assert_eq!(r.unconverged_solves, 0, "DOM gas {}: every direction solve converges", j + 1);
    }
    let wf = g.download(wd.wall_flux())?;
    let sigma_t4 = w64(SIGMA_SB) * t_k.powi(4);
    let want_disc: f64 = set.gases().iter().map(|gs| gs.a * psi_n(&quad, gs.kappa, n)).sum();
    let want_exact: f64 = set.gases().iter().map(|gs| gs.a * (1.0 - 2.0 * e3(gs.kappa))).sum();
    let method: f64 = set
        .gases()
        .iter()
        .map(|gs| gs.a * (psi_n(&quad, gs.kappa, n) - (1.0 - 2.0 * e3(gs.kappa))).abs())
        .sum();
    for wall in ["xMin", "xMax"] {
        let psi = patch_mean(&hm, &wf, wall, sigma_t4);
        let r = (psi / want_disc - 1.0).abs();
        let e = psi - want_exact;
        println!(
            "  wsgg DOM S8 N {n} {wall}: Psi {psi:.12}, sum a_j Psi_N {want_disc:.12} (r {r:.3e}), \
             exact sum {want_exact:.12}, e {e:+.4e} against the method's own {method:.4e}"
        );
        assert!(r <= TOL_R10, "{wall}: Psi {psi:.12} against sum a_j Psi_N {want_disc:.12}, r {r:e}");
        assert!(
            e.abs() <= method + TOL_R10 * want_exact,
            "{wall}: |e| {e:e} exceeds the method's own grey-slab error {method:e}"
        );
    }

    // ---- P1 ----
    let tw = 500.0f64;
    let (hm, m) = mesh(&g, &p1_slab_spec(n))?;
    let walls: Vec<MarshakPatch> = ["xMin", "xMax"]
        .iter()
        .map(|p| MarshakPatch {
            patch: (*p).into(),
            emissivity: 1.0,
            temperature: WallTemperature::Fixed(tw),
        })
        .collect();
    let t = uniform_t(&g, &m, t_k)?;
    let mut wp = WsggP1::new(&g, &hm, &m, set.clone(), 0.0, 0.0, &walls, p1_controls())?;
    let mut src = EnergySources::new(&g, &m)?;
    src.clear(&g)?;
    let perfs = wp.solve(&g, &m, &t, Some(&mut src))?;
    for (j, p) in perfs.iter().enumerate() {
        assert!(p.converged, "P1 gas {}: the PCG solve must converge", j + 1);
    }
    let nc = hm.n_cells;
    let nb = hm.n_boundary_faces;
    let (mut rg, mut rq, mut rw) = (vec![0.0; nc], vec![0.0; 3 * nc], vec![0.0; nb]);
    let (mut rs_q, mut rs_sp) = (vec![0.0; nc], vec![0.0; nc]);
    let drive = w64(SIGMA_SB) * (t_k.powi(4) - tw.powi(4));
    let (mut psi_disc, mut psi_ex, mut p1_method) = (0.0f64, 0.0f64, 0.0f64);
    for gs in set.gases() {
        let med = P1Medium { kappa: gs.kappa, sigma_s: 0.0, a1: 0.0 };
        let mut p1 = P1::new(&g, &hm, &m, &med, &walls, p1_controls())?;
        assert!(p1.solve(&g, &m, &t)?.converged, "grey P1 at kappa {}", gs.kappa);
        let mut s1 = EnergySources::new(&g, &m)?;
        s1.clear(&g)?;
        p1.register_energy_source(&g, &t, &mut s1)?;
        let gg = g.download(&p1.g().f)?;
        let qq = g.download(p1.qr())?;
        let ww = g.download(p1.wall_flux())?;
        let sq = g.download(s1.q())?;
        let ssp = g.download(s1.sp())?;
        for c in 0..nc {
            rg[c] += gs.a * w64(gg[c]);
            rq[3 * c] += gs.a * w64(qq[c].x);
            rq[3 * c + 1] += gs.a * w64(qq[c].y);
            rq[3 * c + 2] += gs.a * w64(qq[c].z);
            rs_q[c] += gs.a * w64(sq[c]);
            rs_sp[c] += gs.a * w64(ssp[c]);
        }
        for f in 0..nb {
            rw[f] += gs.a * w64(ww[f]);
        }
        let pj = patch_mean(&hm, &ww, "xMin", drive);
        psi_disc += gs.a * pj;
        psi_ex += gs.a * psi_p1(gs.kappa);
        p1_method += gs.a * (pj - psi_p1(gs.kappa)).abs();
    }
    let gl: Vec<f64> = g.download(wp.g())?.iter().map(|&v| w64(v)).collect();
    let ql: Vec<f64> = g.download(wp.qr())?.iter().flat_map(|v| [w64(v.x), w64(v.y), w64(v.z)]).collect();
    let wl: Vec<f64> = g.download(wp.wall_flux())?.iter().map(|&v| w64(v)).collect();
    let sql: Vec<f64> = g.download(src.q())?.iter().map(|&v| w64(v)).collect();
    let sspl: Vec<f64> = g.download(src.sp())?.iter().map(|&v| w64(v)).collect();
    for (what, x, want) in [
        ("G", &gl, &rg),
        ("q_r", &ql, &rq),
        ("wall flux", &wl, &rw),
        ("source q", &sql, &rs_q),
        ("source sp", &sspl, &rs_sp),
    ] {
        let r = rel_max(x, want);
        println!("  wsgg P1 N {n} {what}: against sum a_j x grey P1, r {r:.3e}");
        assert!(r <= TOL, "P1 {what}: r {r:e} against sum a_j x grey P1 exceeds {TOL:e}");
    }
    let wf = g.download(wp.wall_flux())?;
    for wall in ["xMin", "xMax"] {
        let psi = patch_mean(&hm, &wf, wall, drive);
        let e = psi - psi_ex;
        println!(
            "  wsgg P1 N {n} {wall}: Psi {psi:.12}, sum a_j Psi_grey {psi_disc:.12}, sum a_j \
             (135.11) {psi_ex:.12}, e {e:+.4e} against the grey solves' own {p1_method:.4e}"
        );
        assert!(
            e.abs() <= p1_method + TOL * psi_ex,
            "P1 {wall}: |e| {e:e} exceeds the grey solves' own error {p1_method:e}"
        );
    }
    Ok(())
}

/// A synthetic coefficient file in the supplement's row layout (DESIGN,
/// not Bordbar's numbers): gas i's weight `a_i = a0_i + (0.01 M_r + 0.02)
/// T̂` for gas 1 and `a0_i` for the others, `kappa_p,i = k_i + 0.001 M_r^4`
/// for gas 1 and `k_i` for the others. `skip`/`dup` drop or repeat one
/// row `(i, j)` to make it malformed.
fn synthetic_file(pt: &str, skip: Option<(usize, usize)>, dup: Option<(usize, usize)>) -> String {
    let a0 = [0.30, 0.25, 0.20, 0.15];
    let k = [0.1, 0.5, 2.0, 10.0];
    let mut s = String::from(
        "SYNTHETIC test file - not Bordbar's coefficients\n  Pt (atm)    a(i,j)or k(i)  b4 b3 b2 b1 b0\n",
    );
    for i in 1..=4usize {
        for j in [1usize, 2, 3, 4, 5, 0] {
            let mut row = [0.0f64; 5];
            if j == 1 {
                row[4] = a0[i - 1];
            }
            if i == 1 && j == 2 {
                row[3] = 0.01;
                row[4] = 0.02;
            }
            if j == 0 {
                row[4] = k[i - 1];
                if i == 1 {
                    row[0] = 0.001;
                }
            }
            let line = format!(
                " {pt}  {i} {j}  {:e}  {:e}  {:e}  {:e}  {:e}\n",
                row[0], row[1], row[2], row[3], row[4]
            );
            if skip != Some((i, j)) {
                s.push_str(&line);
            }
            if dup == Some((i, j)) {
                s.push_str(&line);
            }
        }
    }
    s
}

fn refused<T: std::fmt::Debug>(what: &str, r: Result<T>, needle: &str) {
    match r {
        Ok(v) => panic!("{what}: must be refused, got {v:?}"),
        Err(e) => {
            let msg = e.to_string();
            assert!(msg.contains(needle), "{what}: the refusal must name '{needle}': {msg}");
            println!("  {what}: refused - {msg}");
        }
    }
}

/// `wsgg_requires_h2o_co2` (§136.5, class D): `wsgg` without H2O or CO2
/// among the species is refused naming the missing one; `X_CO2 = 0` is
/// refused, with or without water; the synthetic file evaluates (136.8)-
/// (136.9) exactly as written.
#[test]
fn wsgg_requires_h2o_co2() -> Result<()> {
    assert_eq!(require_h2o_co2(&["N2", "CO2", "O2", "H2O"])?, (3, 1));
    refused("no CO2", require_h2o_co2(&["N2", "O2", "H2O"]), "no CO2");
    refused("no H2O", require_h2o_co2(&["N2", "CO2"]), "no H2O");
    refused("neither", require_h2o_co2(&["N2"]), "no H2O and no CO2");

    let c = WsggCoefficients::parse(&synthetic_file("1.000000", None, None), 1.0)?;
    refused("X_CO2 = 0", c.gases(1500.0, 0.2, 0.0), "X_CO2 = 0");
    refused("X_H2O = X_CO2 = 0", c.gases(1500.0, 0.0, 0.0), "X_CO2 = 0");
    refused("X_H2O = 1.5", c.gases(1500.0, 1.5, 0.1), "X_H2O");

    // T̂ = 1, M_r = 2, p_H2O + p_CO2 = 0.3 atm.
    let s = c.gases(1500.0, 0.2, 0.1)?;
    let want = [
        (0.30 + (0.01 * 2.0 + 0.02), (0.1 + 0.001 * 16.0) * 0.3),
        (0.25, 0.5 * 0.3),
        (0.20, 2.0 * 0.3),
        (0.15, 10.0 * 0.3),
    ];
    for (j, (gs, (a, k))) in s.gases().iter().zip(want).enumerate() {
        assert!((gs.a - a).abs() <= 1e-15 * a, "gas {}: a {} want {a}", j + 1, gs.a);
        assert!((gs.kappa - k).abs() <= 1e-15 * k, "gas {}: kappa {} want {k}", j + 1, gs.kappa);
    }
    // T̂ = 0.4 at 600 K: gas 1's weight carries T̂ once.
    let s = c.gases(600.0, 0.2, 0.1)?;
    let a1 = 0.30 + (0.01 * 2.0 + 0.02) * 0.4;
    assert!((s.gases()[0].a - a1).abs() <= 1e-15, "gas 1 at 600 K: a {} want {a1}", s.gases()[0].a);
    Ok(())
}

/// `wsgg_refuses_out_of_range` (§136.5, class D): `T` and `M_r` outside
/// the fitted range are refused naming the range; the range's own ends
/// are accepted.
#[test]
fn wsgg_refuses_out_of_range() -> Result<()> {
    let c = WsggCoefficients::parse(&synthetic_file("1.000000", None, None), 1.0)?;
    refused("T = 299.9 K", c.gases(299.9, 0.1, 0.1), "300 <= T <= 3000");
    refused("T = 3000.1 K", c.gases(3000.1, 0.1, 0.1), "300 <= T <= 3000");
    refused("T = NaN", c.gases(f64::NAN, 0.1, 0.1), "300 <= T <= 3000");
    refused("M_r = 0.09", c.gases(1500.0, 0.009, 0.1), "0.1 <= M_r <= 4");
    refused("M_r = 4.01", c.gases(1500.0, 0.401, 0.1), "0.1 <= M_r <= 4");
    refused("X_H2O = 0 (M_r = 0)", c.gases(1500.0, 0.0, 0.1), "0.1 <= M_r <= 4");
    for (t, xh, xc) in [(300.0, 0.1, 0.1), (3000.0, 0.1, 0.1), (1500.0, 0.1, 1.0), (1500.0, 0.5, 0.125)] {
        c.gases(t, xh, xc)?;
    }
    Ok(())
}

/// `wsgg_user_file_digest` (§136.5, class D): FIPS 180-4's example digests;
/// a synthetic file is accepted with a warning naming its digest; a
/// malformed file and a pressure outside the ten are refused; with
/// `OFGPU_WSGG_DIR` the ten published files are accepted silently and a file
/// given at another pressure is refused - skipped by name otherwise.
#[test]
fn wsgg_user_file_digest() -> Result<()> {
    for (msg, want) in [
        (&b""[..], "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        (&b"abc"[..], "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        (
            &b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"[..],
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        ),
    ] {
        assert_eq!(sha256_hex(msg), want, "FIPS 180-4 example of {} bytes", msg.len());
    }
    let million = vec![b'a'; 1_000_000];
    assert_eq!(
        sha256_hex(&million),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );

    let dir = std::env::temp_dir().join(format!("ofgpu_wsgg_{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| ofgpu::Error::Config(e.to_string()))?;
    let write = |name: &str, text: &str| -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, text).expect("the temp dir is writable");
        p
    };
    let text = synthetic_file("1.000000", None, None);
    let p = write("synthetic.dat", &text);
    let (c, warn) = WsggCoefficients::read(&p, 1.0)?;
    assert_eq!(c.pressure_atm(), 1.0);
    let digest = sha256_hex(text.as_bytes());
    let w = warn.expect("an unknown digest is warned");
    assert!(w.contains(&digest), "the warning names the digest {digest}: {w}");
    println!("  synthetic file: accepted, warned - {w}");
    refused("pressure 0.7 atm", WsggCoefficients::read(&p, 0.7), "ten pressures");
    refused("Pt column 1 atm read at 0.5", WsggCoefficients::read(&p, 0.5), "another pressure");
    let p = write("missing.dat", &synthetic_file("1.000000", Some((3, 0)), None));
    refused("missing row", WsggCoefficients::read(&p, 1.0), "gas 3 row 0 is missing");
    let p = write("dup.dat", &synthetic_file("1.000000", None, Some((2, 4))));
    refused("duplicate row", WsggCoefficients::read(&p, 1.0), "gas 2 row 4 appears twice");
    refused("no such file", WsggCoefficients::read(&dir.join("absent.dat"), 1.0), "absent.dat");
    let _ = std::fs::remove_dir_all(&dir);

    match std::env::var_os("OFGPU_WSGG_DIR") {
        None => println!(
            "  wsgg_user_file_digest real-data rows: SKIPPED by name - OFGPU_WSGG_DIR is not \
             set (Bordbar 2021's files are not distributed, SPEC-LIT 136.4)"
        ),
        Some(d) => {
            let d = PathBuf::from(d);
            for (pt, name, _) in WSGG_DIGESTS {
                let (c, warn) = WsggCoefficients::read(&d.join(name), pt)?;
                assert!(warn.is_none(), "{name}: the published file is accepted silently");
                let s = c.gases(1500.0, 0.1, 0.1)?;
                let eps = s.emissivity(1.0);
                assert!(eps > 0.0 && eps < 1.0, "{name}: eps(1 m) = {eps}");
                println!("  {name} ({pt} atm): digest known; eps(1500 K, 0.1/0.1, 1 m) = {eps:.15e}");
            }
            refused(
                "the 1 atm file at 0.5 atm",
                WsggCoefficients::read(&d.join(WSGG_DIGESTS[2].1), 0.5),
                "the file for 1 atm",
            );
        }
    }
    Ok(())
}

/// `wsgg_grey_mean` (§136.4, class D): `L_m = 3.6 V/A` counts the wall and
/// plain faces only; the grey `kappa` reproduces `eps(L_m)` exactly; a
/// non-positive `L_m` and an `eps` outside (0, 1) are refused; an empty or
/// window-only set is refused.
#[test]
fn wsgg_grey_mean() -> Result<()> {
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
    assert!((lm - 0.6).abs() <= 1e-15, "the unit cube: L_m {lm}, want 0.6");
    let lm_slab = mean_beam_length(&blockgen::build_mesh(&p1_slab_spec(4))?)?;
    assert!((lm_slab - 1.8).abs() <= 1e-14, "the P1 slab (empty sides): L_m {lm_slab}, want 1.8");
    let lm_sym = mean_beam_length(&blockgen::build_mesh(&dom_slab_spec(4))?)?;
    assert!((lm_sym - 1.8).abs() <= 1e-14, "the DOM slab (symmetry sides): L_m {lm_sym}, want 1.8");

    let set = synthetic_set();
    for l in [0.6, 1.8] {
        let k = set.grey_kappa(l)?;
        let eps = set.emissivity(l);
        let back = -(-k * l).exp_m1();
        assert!((back / eps - 1.0).abs() <= 1e-14, "L_m {l}: 1 - exp(-kappa L_m) {back} against eps {eps}");
        println!("  wsgg grey mean L_m {l}: eps {eps:.15}, kappa {k:.15} 1/m");
    }
    refused("L_m = 0", set.grey_kappa(0.0), "mean beam length");
    let over = GreyGasSet::new(vec![GreyGas { a: 1.2, kappa: 10.0 }])?;
    refused("eps > 1", over.grey_kappa(1.0), "outside (0, 1)");
    refused("empty set", GreyGasSet::new(Vec::new()), "no grey gas");
    refused("a window as a gas", GreyGasSet::new(vec![GreyGas { a: 0.5, kappa: 0.0 }]), "window");
    Ok(())
}
