// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate CHR-CUBE (SPEC-LIT §136.8): fvDOM with scattering on the slab,
//! and the isothermal cube at 24^3, run live.**
//!
//! A live port of this repository's own tests into `ofgpu-validate` rows,
//! one row per assertion, the tests staying as they are:
//! `tests/radiation_dom.rs::scatter_oracle_reduces_to_e3` - (136.5) solved
//! in the gate by product integration, at `omega = 0` reducing to (135.6)
//! and at `omega = 0.5` reproducing §136.3's three A1 values;
//! `dom_scatter_slab` - `omega = 0.5`, `tau = 1`, `A_1 = 0, +-0.5`, S8 on
//! N = 32 within 2 % of (136.5), both walls, every solve the host source
//! iteration (136.14) at round-off; and `dom_cube_exact_integral` - the
//! wall-centre flux of the cold black cube against (136.6) by adaptive
//! quadrature, S8 within 3 % and S4 within 6 % at `kappa L = 0.1, 1, 10`,
//! every solve the host step scheme face by face at round-off.
//! `kappa L = 0.1` is the recorded miss, asserted with its causes: S8 lies
//! outside 3 % by the step scheme's spatial error (its own limit (136.17)
//! is inside, 32^3 is closer) and S4 passes 6 % only by cancellation (its
//! own limit is outside). Jamaluddin & Smith (1988) is NAMED, NOT RUN.
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §136.3 (the scattering-slab and cube references),
//!     §136.5 (the rows), §136.8 (scattering by source iteration, the
//!     cube, (136.14), (136.17), the recorded misses)
//!   tests/radiation_dom.rs (`scatter_oracle_reduces_to_e3`,
//!     `dom_scatter_slab`, `dom_cube_exact_integral` and their helpers) -
//!     the tests stay as they are
//! No GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::radiation::dom::{DomCondition, DomControls, DomMedium, DomPatch, DomSolve};
use ofgpu::radiation::dom_quad::{Quadrature, SnOrder};
use ofgpu::radiation::p1::WallTemperature;
use ofgpu::radiation::SIGMA_SB;
use ofgpu::solver::{LinearSolverKind, Preconditioner, SolverControls};
use ofgpu::{Error, Gpu, GpuMesh, GpuScalarField, HostMesh, Result, Scalar};

use super::chr_fixture::{e1, psi_exact, w64};
use crate::{key, Checks, GateReport, How, Uncertainty, Verdict};

/// The gate's scope and registry name.
pub(crate) const GATE: &str = "SPEC-LIT 136.8 Gate CHR-CUBE scattering slab and cube";

const PI: f64 = std::f64::consts::PI;

// ---------------------------------------------------------------------------
//  The DOM slab helpers the scattering rows share (tests/radiation_dom.rs)
// ---------------------------------------------------------------------------

/// The sweep tolerance of every DOM solve: 1e-12 in f64, §136.7's `tol32`
/// of 1e-5 under `single`.
const SWEEP_TOL: f64 = if cfg!(feature = "single") { 1e-5 } else { 1e-12 };

/// Every DOM solve's controls: PBiCGStab + diagonal at 1e-12 absolute, in
/// BOTH builds, the sweep at `SWEEP_TOL`, 200 sweeps at most.
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

/// The non-scattering DOM medium: `kappa = kl`, `sigma_s = 0`, `A_1 = 0`.
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

/// T uniform in every cell and on every face, written through `t.f`/`t.bf`.
fn uniform_t(g: &Gpu, m: &GpuMesh, t_k: f64) -> Result<GpuScalarField> {
    let mut t = GpuScalarField::zeros(g, m, "T")?;
    g.write(&mut t.f, &vec![t_k as Scalar; m.n_cells])?;
    g.write(&mut t.bf, &vec![t_k as Scalar; m.n_boundary_faces])?;
    Ok(t)
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
/// T = 1000 K, any medium `med`.
fn slab_case_in(
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

// ---------------------------------------------------------------------------
//  The scattering oracle and its host step scheme (tests/radiation_dom.rs
//  1727-1990)
// ---------------------------------------------------------------------------

/// `E_2(x) .. E_5(x)` for `x >= 0`: `E_1` as the fixture's [`e1`], then
/// A&S 5.1.14 upward, `E_(k+1) = (e^-x - x E_k)/k`; `E_n(0) = 1/(n - 1)`.
/// Only `x <= 1` is used (the oracle's slab is `tau = 1`), where the
/// upward recurrence loses no digit that matters.
fn e2to5(x: f64) -> [f64; 4] {
    if x == 0.0 {
        return [1.0, 0.5, 1.0 / 3.0, 0.25];
    }
    let ex = (-x).exp();
    let mut e = e1(x);
    let mut out = [0.0f64; 4];
    for (k, o) in out.iter_mut().enumerate() {
        e = (ex - x * e) / (k + 1) as f64;
        *o = e;
    }
    out
}

/// Dense Gaussian elimination with partial pivoting, in place: `a` is
/// `n x n` row-major, `b` the right-hand side; returns the solution.
fn dense_solve(mut a: Vec<f64>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for k in 0..n {
        let piv = (k..n)
            .max_by(|&i, &j| a[i * n + k].abs().total_cmp(&a[j * n + k].abs()))
            .expect("a non-empty column");
        if piv != k {
            for c in 0..n {
                a.swap(k * n + c, piv * n + c);
            }
            b.swap(k, piv);
        }
        let d = a[k * n + k];
        assert!(d != 0.0, "the oracle's system is singular at column {k}");
        for i in k + 1..n {
            let f = a[i * n + k] / d;
            if f != 0.0 {
                for c in k..n {
                    a[i * n + c] -= f * a[k * n + c];
                }
                b[i] -= f * b[k];
            }
        }
    }
    let mut x = vec![0.0f64; n];
    for k in (0..n).rev() {
        let mut s = b[k];
        for c in k + 1..n {
            s -= a[k * n + c] * x[c];
        }
        x[k] = s / a[k * n + k];
    }
    x
}

/// (136.5) solved by product integration (SPEC-LIT §136.3, (136.16)):
/// piecewise-linear `a` and `b` on `panels` cosine-graded panels of
/// `[0, tau]`, the kernel moments in closed form. In units of `I_b = 1`,
/// so `Psi = -q(0)/pi`.
fn scatter_oracle(tau: f64, om: f64, a1: f64, panels: usize) -> f64 {
    let n = panels + 1;
    let s: Vec<f64> = (0..n)
        .map(|j| tau * (1.0 - (PI * j as f64 / panels as f64).cos()) / 2.0)
        .collect();
    // E_2..E_5 at every node distance, once.
    let mut et = vec![[0.0f64; 4]; n * n];
    for i in 0..n {
        for j in 0..n {
            et[i * n + j] = e2to5((s[i] - s[j]).abs());
        }
    }
    // k1 = int E1 phi_j, k2s = int sgn(t_i - s) E2 phi_j, k3 = int E3 phi_j.
    let (mut k1, mut k2s, mut k3) = (vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n]);
    for i in 0..n {
        for p in 0..panels {
            let h = s[p + 1] - s[p];
            // (near node, far node, sign of t_i - s on the panel).
            let (near, far, sg) = if s[p] >= s[i] { (p, p + 1, -1.0) } else { (p + 1, p, 1.0) };
            let (x0, x1) = ((s[near] - s[i]).abs(), (s[far] - s[i]).abs());
            let (e0, e1v) = (et[i * n + near], et[i * n + far]);
            for (kn, sign, k) in [(1usize, 1.0, &mut k1), (2, sg, &mut k2s), (3, 1.0, &mut k3)] {
                // E_(kn+1) is e?[kn - 1], E_(kn+2) is e?[kn].
                let m0 = e0[kn - 1] - e1v[kn - 1];
                let m1 = x0 * e0[kn - 1] + e0[kn] - x1 * e1v[kn - 1] - e1v[kn];
                k[i * n + near] += sign * (x1 * m0 - m1) / h;
                k[i * n + far] += sign * (m1 - x0 * m0) / h;
            }
        }
    }
    // a - (om/2) (K1 a + K2s b) = (1 - om);  b - (om A1/2) (K2s a + K3 b) = 0.
    let c = om / 2.0;
    let nn = 2 * n;
    let mut a = vec![0.0f64; nn * nn];
    let mut rhs = vec![0.0f64; nn];
    for i in 0..n {
        a[i * nn + i] += 1.0;
        a[(n + i) * nn + n + i] += 1.0;
        for j in 0..n {
            a[i * nn + j] -= c * k1[i * n + j];
            a[i * nn + n + j] -= c * k2s[i * n + j];
            a[(n + i) * nn + j] -= a1 * c * k2s[i * n + j];
            a[(n + i) * nn + n + j] -= a1 * c * k3[i * n + j];
        }
        rhs[i] = 1.0 - om;
    }
    let x = dense_solve(a, rhs);
    // Psi = -q(0)/pi = -2 (K2s[0] . a + K3[0] . b).
    let q0: f64 = (0..n).map(|j| k2s[j] * x[j] + k3[j] * x[n + j]).sum();
    -2.0 * q0
}

/// The oracle at 200 and 400 panels, Richardson-extrapolated at second
/// order (SPEC-LIT §136.3's construction).
fn scatter_reference(tau: f64, om: f64, a1: f64) -> f64 {
    let (p200, p400) = (scatter_oracle(tau, om, a1, 200), scatter_oracle(tau, om, a1, 400));
    (4.0 * p400 - p200) / 3.0
}

/// The three references of the scattering slab, `(A_1, Psi)` at
/// `omega = 0.5`, `tau = 1`, computed once per run.
fn scatter_refs() -> &'static [(f64, f64); 3] {
    static REFS: std::sync::OnceLock<[(f64, f64); 3]> = std::sync::OnceLock::new();
    REFS.get_or_init(|| [0.0, 0.5, -0.5].map(|a1| (a1, scatter_reference(1.0, 0.5, a1))))
}

/// The step scheme's scattering slab on the host in f64 (SPEC-LIT §136.8,
/// (136.14)): §136.7's recurrence per direction, with (136.3)'s lagged
/// in-scattering sum `(sigma_s/4pi)(G_k + A_1 Omega_x q_k)` - the sides'
/// mirror faces cancel at the fixed point, so `q` is along x - iterated
/// until `G` changes by less than 1e-15. `tau = beta L`, `omega =
/// sigma_s/beta`, `I_b = 1`; returns `Psi` at xMin.
fn psi_scatter_n(quad: &Quadrature, tau: f64, om: f64, a1: f64, n: usize) -> f64 {
    let (beta, dx) = (tau, 1.0 / n as f64);
    let (sig, kap) = (om * beta, (1.0 - om) * beta);
    let nd = quad.len();
    let mut inten = vec![vec![0.0f64; n]; nd];
    let (mut gk, mut qk) = (vec![0.0f64; n], vec![0.0f64; n]);
    for _ in 0..2000 {
        for (l, d) in quad.dirs().iter().enumerate() {
            let mu = d[0].abs();
            let mut prev = 0.0f64;
            for step in 0..n {
                let k = if d[0] > 0.0 { step } else { n - 1 - step };
                let src = kap + sig / (4.0 * PI) * (gk[k] + a1 * d[0] * qk[k]);
                prev = (mu * prev + dx * src) / (mu + beta * dx);
                inten[l][k] = prev;
            }
        }
        let (mut gn, mut qn) = (vec![0.0f64; n], vec![0.0f64; n]);
        for (l, d) in quad.dirs().iter().enumerate() {
            let w = quad.weights()[l];
            for k in 0..n {
                gn[k] += w * inten[l][k];
                qn[k] += w * d[0] * inten[l][k];
            }
        }
        let dmax = gn.iter().zip(&gk).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let gmax = gn.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        gk = gn;
        qk = qn;
        if dmax < 1e-15 * gmax {
            break;
        }
    }
    let mut psi = 0.0f64;
    for (l, d) in quad.dirs().iter().enumerate() {
        if d[0] < 0.0 {
            psi += quad.weights()[l] * -d[0] * inten[l][0];
        }
    }
    psi / PI
}

/// The sweep tolerance of the scattering slab solves. The mirror lag and
/// the scattering lag together contract by about 0.6 per sweep here, so
/// stopping at `change < tol` leaves an iteration error of up to about
/// `1.5 tol` - above the R row's `tol32` of 1e-5 if `tol` were §136.7's
/// 1e-5 (measured under `single`). Under `single` the control is
/// therefore 1e-6, about 17 f32 ulps of `G`; in f64 it is `SWEEP_TOL`. A
/// control, not a gate tolerance: the R rows stay at round-off.
const SCATTER_SWEEP_TOL: f64 = if cfg!(feature = "single") { 1e-6 } else { SWEEP_TOL };

/// The scattering slab's medium: `beta = tau`, `sigma_s = omega tau`.
fn scatter_medium(tau: f64, om: f64, a1: f64) -> DomMedium {
    DomMedium { kappa: (1.0 - om) * tau, sigma_s: om * tau, a1 }
}

/// Solve the scattering slab with `order` on `n` cells at `tau = 1` and
/// emit its rows: the sweeps converge with every direction solve converged
/// (`require`), and BOTH walls are the host source iteration (136.14) of
/// SPEC-LIT 136.8 at round-off (`check_roundoff`, 1e-10). Returns
/// `(Psi_xMin, Psi_xMax, sweeps)`.
fn scatter_slab_psi(
    c: &mut Checks,
    g: &Gpu,
    order: SnOrder,
    om: f64,
    a1: f64,
    n: usize,
) -> Result<(f64, f64, usize)> {
    let tau = 1.0;
    let ctrl = DomControls { tol: SCATTER_SWEEP_TOL, ..dom_controls() };
    let (hm, m, t, mut dom) = slab_case_in(g, order, &scatter_medium(tau, om, a1), n, ctrl)?;
    let quad = dom.rows().quadrature().clone();
    let rep = dom.solve(g, &m, &t)?;
    let what = format!("{order:?} omega {om} A1 {a1} N {n}");
    c.require(
        &format!("Gate CHR-CUBE: scattering slab {what} - the sweeps converge and every direction solve converges (SPEC-LIT 136.8)"),
        rep.converged && rep.unconverged_solves == 0,
    );
    let wf = g.download(dom.wall_flux())?;
    let sigma_t4 = SIGMA_SB as f64 * 1000f64.powi(4);
    let lo = patch_psi(&hm, &wf, "xMin", sigma_t4)?;
    let hi = patch_psi(&hm, &wf, "xMax", sigma_t4)?;
    let want = psi_scatter_n(&quad, tau, om, a1, n);
    for (wall, v) in [("xMin", lo), ("xMax", hi)] {
        let r = (v / want - 1.0).abs();
        c.check_roundoff(
            &format!("Gate CHR-CUBE: scattering slab {what} {wall} against (136.14) of SPEC-LIT 136.8"),
            r as Scalar,
            1e-10,
        );
    }
    Ok((lo, hi, rep.sweeps))
}

// ---------------------------------------------------------------------------
//  The isothermal cube (tests/radiation_dom.rs 2070-2316)
// ---------------------------------------------------------------------------

/// Adaptive Simpson on `[a, b]` to `tol` absolute (Lyness's criterion,
/// with the Richardson correction), at most 40 halvings deep.
fn simpson<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, tol: f64) -> f64 {
    /// One interval `[a, b]` with `f` at `a`, the midpoint and `b`, and
    /// its whole-interval Simpson value.
    fn rec<F: Fn(f64) -> f64>(
        f: &F,
        ab: (f64, f64),
        fs: [f64; 3],
        whole: f64,
        tol: f64,
        depth: u32,
    ) -> f64 {
        let ((a, b), [fa, fm, fb]) = (ab, fs);
        let m = 0.5 * (a + b);
        let (flm, frm) = (f(0.5 * (a + m)), f(0.5 * (m + b)));
        let left = (m - a) / 6.0 * (fa + 4.0 * flm + fm);
        let right = (b - m) / 6.0 * (fm + 4.0 * frm + fb);
        let d = left + right - whole;
        if depth == 0 || d.abs() <= 15.0 * tol {
            return left + right + d / 15.0;
        }
        rec(f, (a, m), [fa, flm, fm], left, 0.5 * tol, depth - 1)
            + rec(f, (m, b), [fm, frm, fb], right, 0.5 * tol, depth - 1)
    }
    let (fa, fm, fb) = (f(a), f(0.5 * (a + b)), f(b));
    let whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
    rec(f, (a, b), [fa, fm, fb], whole, tol, 40)
}

/// (136.6) at the centre of one wall of the unit cube, over `sigma T^4`:
/// `(8/pi) int_0^(pi/4) dphi int_0^(pi/2) (1 - e^(-kappa L s)) cos(theta)
/// sin(theta) dtheta`, with `s = min(1/cos(theta), 0.5/(sin(theta)
/// cos(phi)))` - the 8-fold symmetry folds `phi` into `[0, pi/4]`, and the
/// polar integral is split at the edge direction `tan(theta_e) =
/// 0.5/cos(phi)`, where `s` has its kink (SPEC-LIT §136.3).
fn cube_exact(kl: f64) -> f64 {
    let inner = |phi: f64| {
        let te = (0.5 / phi.cos()).atan();
        let top = |t: f64| -(-kl / t.cos()).exp_m1() * t.cos() * t.sin();
        let side = |t: f64| -(-kl * 0.5 / (t.sin() * phi.cos())).exp_m1() * t.cos() * t.sin();
        simpson(&top, 0.0, te, 1e-14) + simpson(&side, te, PI / 2.0, 1e-14)
    };
    8.0 / PI * simpson(&inner, 0.0, PI / 4.0, 1e-13)
}

/// (136.17): the ordinates' own value at the wall centre - each direction
/// into the zMin wall integrated exactly along its ray, which is what the
/// step scheme approaches as the mesh is refined (`kappa L` per unit side).
fn cube_ordinate_limit(quad: &Quadrature, kl: f64) -> f64 {
    let mut psi = 0.0f64;
    for (l, d) in quad.dirs().iter().enumerate() {
        if d[2] < 0.0 {
            let s = (1.0 / d[2].abs()).min(0.5 / d[0].abs()).min(0.5 / d[1].abs());
            psi += quad.weights()[l] * d[2].abs() * -(-kl * s).exp_m1();
        }
    }
    psi / PI
}

/// The unit cube on `n^3` cells, all six patches walls.
fn cube_mesh(g: &Gpu, n: usize) -> Result<(HostMesh, GpuMesh)> {
    let axis = || GradedAxis {
        lo: 0.0 as Scalar,
        hi: 1.0 as Scalar,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    let spec = BlockSpec {
        x: axis(),
        y: axis(),
        z: axis(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall"; 6].map(String::from),
        windows: Vec::new(),
        cyclic: Vec::new(),
    };
    let hm = blockgen::build_mesh(&spec)?;
    let m = GpuMesh::upload(g, &hm)?;
    Ok((hm, m))
}

/// The step scheme of the non-scattering cube on the host in f64 (SPEC-LIT
/// §136.8): with black walls at 0 K nothing enters, so each direction's
/// rows are lower-triangular in the order of `Omega . c` and one upwind
/// pass over the cells in that order solves them exactly, from the mesh's
/// own faces. Returns the zMin wall flux of (136.4) per face of the patch,
/// with `I_b = sigma T^4/pi`.
fn cube_host_zmin(hm: &HostMesh, quad: &Quadrature, kl: f64, ib: f64) -> Result<Vec<f64>> {
    let n = hm.n_cells;
    // Per-cell internal faces: (face, other cell, sign of S_f out of the cell).
    let mut faces: Vec<Vec<(usize, usize, f64)>> = vec![Vec::new(); n];
    for f in 0..hm.n_internal_faces {
        let (o, nb) = (hm.owner[f] as usize, hm.neighbour[f] as usize);
        faces[o].push((f, nb, 1.0));
        faces[nb].push((f, o, -1.0));
    }
    let mut bout = vec![0.0f64; n];
    let pz = hm
        .patches
        .iter()
        .find(|p| p.name == "zMin")
        .ok_or_else(|| Error::Config("the cube has no zMin patch".to_string()))?;
    let mut flux = vec![0.0f64; pz.size];
    for (l, d) in quad.dirs().iter().enumerate() {
        let key = |c: usize| {
            let x = hm.c[c];
            d[0] * w64(x.x) + d[1] * w64(x.y) + d[2] * w64(x.z)
        };
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| key(a).total_cmp(&key(b)));
        bout.iter_mut().for_each(|v| *v = 0.0);
        for bf in 0..hm.n_boundary_faces {
            let s = hm.b_sf[bf];
            let dot = d[0] * w64(s.x) + d[1] * w64(s.y) + d[2] * w64(s.z);
            if dot > 0.0 {
                bout[hm.b_face_cells[bf] as usize] += dot;
            }
        }
        let mut inten = vec![0.0f64; n];
        for &c in &order {
            let (mut diag, mut rhs) = (bout[c] + kl * w64(hm.v[c]), kl * w64(hm.v[c]) * ib);
            for &(f, other, sg) in &faces[c] {
                let s = hm.sf[f];
                let dot = sg * (d[0] * w64(s.x) + d[1] * w64(s.y) + d[2] * w64(s.z));
                if dot > 0.0 {
                    diag += dot;
                } else {
                    rhs += -dot * inten[other];
                }
            }
            inten[c] = rhs / diag;
        }
        if d[2] < 0.0 {
            for (i, fl) in flux.iter_mut().enumerate() {
                let bf = pz.start + i;
                *fl += quad.weights()[l] * d[2].abs() * inten[hm.b_face_cells[bf] as usize];
            }
        }
    }
    Ok(flux)
}

/// Solve the cube with `order` at `kappa L = kl` on `n^3` cells and emit
/// its rows: black walls at 0 K, T = 1000 K. The solve converges with
/// every direction solve converged (`require`), the device zMin wall flux
/// is the host step scheme face by face at round-off (`check_roundoff`,
/// 1e-10, worst `|got - want| / scale` over the faces), and four faces
/// meet at the centre of zMin (`require`). Returns `(Psi at the wall
/// centre, sweeps)`: the mean of those four faces over `sigma T^4`.
fn cube_psi(c: &mut Checks, g: &Gpu, order: SnOrder, kl: f64, n: usize) -> Result<(f64, usize)> {
    if n % 2 != 0 {
        return Err(Error::Config(format!(
            "cube n {n} is odd: the wall centre must be a meeting point of four faces"
        )));
    }
    let (hm, m) = cube_mesh(g, n)?;
    let quad = Quadrature::new(order)?;
    let black = || DomCondition::GreyDiffuse {
        emissivity: 1.0,
        temperature: WallTemperature::Fixed(0.0),
    };
    let ps: Vec<DomPatch> = ["xMin", "xMax", "yMin", "yMax", "zMin", "zMax"]
        .into_iter()
        .map(|p| DomPatch { patch: p.into(), condition: black() })
        .collect();
    let t = uniform_t(g, &m, 1000.0)?;
    let mut dom = DomSolve::new(g, &hm, &m, &dom_medium(kl), &quad, &ps, dom_controls())?;
    let rep = dom.solve(g, &m, &t)?;
    let what = format!("cube {order:?} kL {kl} n {n}");
    c.require(
        &format!("Gate CHR-CUBE: {what} - the sweeps converge and every direction solve converges (SPEC-LIT 136.8)"),
        rep.converged && rep.unconverged_solves == 0,
    );
    let wf = g.download(dom.wall_flux())?;
    let sigma_t4 = SIGMA_SB as f64 * 1000f64.powi(4);
    let host = cube_host_zmin(&hm, &quad, kl, sigma_t4 / PI)?;
    let pz = hm
        .patches
        .iter()
        .find(|p| p.name == "zMin")
        .ok_or_else(|| Error::Config("the cube has no zMin patch".to_string()))?;
    let scale = host.iter().fold(0.0f64, |a, &b| a.max(b));
    let h = 1.0 / n as f64;
    let (mut centre, mut count) = (0.0f64, 0usize);
    let mut worst = 0.0f64;
    for (i, &want) in host.iter().enumerate() {
        let bf = pz.start + i;
        let got = w64(wf[bf]);
        worst = worst.max((got - want).abs() / scale);
        let cf = hm.b_cf[bf];
        if (w64(cf.x) - 0.5).abs() < h && (w64(cf.y) - 0.5).abs() < h {
            centre += got;
            count += 1;
        }
    }
    c.check_roundoff(
        &format!("Gate CHR-CUBE: {what} zMin faces against the host step scheme, worst |got - want| / scale (SPEC-LIT 136.8)"),
        worst as Scalar,
        1e-10,
    );
    c.require(
        &format!("Gate CHR-CUBE: {what} - four faces meet at the centre of zMin (SPEC-LIT 136.8)"),
        count == 4,
    );
    Ok((centre / 4.0 / sigma_t4, rep.sweeps))
}

/// **Gate CHR-CUBE** (SPEC-LIT §136.8): fvDOM with scattering on the slab,
/// and the isothermal cube, live, in both builds.
pub(crate) fn check_chr_cube(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    // ---- the scattering oracle's own check (host f64, both builds) ---------
    let v = scatter_oracle(1.0, 0.0, 0.0, 200);
    let ex = psi_exact(1.0);
    c.check_a(
        "Gate CHR-CUBE: the (136.5) oracle at omega 0 against 1 - 2 E_3(1) (SPEC-LIT 136.8)",
        (v / ex - 1.0).abs() as Scalar,
        1e-12,
    );
    for ((a1, got), want) in scatter_refs().iter().zip([0.5591260, 0.5621486, 0.5561464]) {
        c.check_a(
            &format!("Gate CHR-CUBE: the (136.5) oracle at omega 0.5 A1 {a1:+} against SPEC-LIT 136.3's table value"),
            (got - want).abs() as Scalar,
            1e-7,
        );
    }

    // ---- the scattering slab: omega 0.5, tau 1, three A1, S8 N = 32 --------
    for &(a1, want) in scatter_refs().iter() {
        let (lo, hi, _sweeps) = scatter_slab_psi(c, gpu, SnOrder::S8, 0.5, a1, 32)?;
        for (wall, v) in [("xMin", lo), ("xMax", hi)] {
            let e = v / want - 1.0;
            c.check_p(
                &format!("Gate CHR-CUBE: S8 N 32 omega 0.5 A1 {a1:+} {wall} within 2 % of (136.5) (SPEC-LIT 136.8)"),
                e.abs() as Scalar,
                0.02,
            );
        }
    }

    // ---- the isothermal cube at 24^3 against (136.6) -----------------------
    let table = [(0.1, 0.0791532440), (1.0, 0.5537277888), (10.0, 0.9989394816)];
    let refs: Vec<f64> = table.iter().map(|&(kl, _)| cube_exact(kl)).collect();
    for (&(kl, want), &got) in table.iter().zip(&refs) {
        c.check_a(
            &format!("Gate CHR-CUBE: (136.6) at kL {kl} against SPEC-LIT 136.3's table value"),
            (got / want - 1.0).abs() as Scalar,
            1e-9,
        );
    }
    let mut s8_miss: Option<(f64, f64, f64, f64)> = None;
    let mut s4_limit = 0.0f64;
    for (&(kl, _), &ex) in table.iter().zip(&refs) {
        for (order, band) in [(SnOrder::S4, 0.06f64), (SnOrder::S8, 0.03f64)] {
            let quad = Quadrature::new(order)?;
            let (psi, _sweeps) = cube_psi(c, gpu, order, kl, 24)?;
            let e = psi / ex - 1.0;
            let e_lim = cube_ordinate_limit(&quad, kl) / ex - 1.0;
            c.check_p(
                &format!("Gate CHR-CUBE: {order:?} kL {kl} 24^3 wall-centre flux within {band} of (136.6) (SPEC-LIT 136.8)"),
                e.abs() as Scalar,
                band as Scalar,
            );
            c.note(&format!(
                "Gate CHR-CUBE {order:?} kL {kl} 24^3: e {e:+.4e}, the set's own limit (136.17) {e_lim:+.4e}"
            ));
            if kl > 0.5 {
                continue;
            }
            if order == SnOrder::S8 {
                // The miss by the step scheme's spatial error, asserted
                // with its causes.
                let (psi32, _) = cube_psi(c, gpu, order, kl, 32)?;
                let e32 = psi32 / ex - 1.0;
                c.note(&format!("Gate CHR-CUBE S8 kL {kl} 32^3: e {e32:+.4e}"));
                c.require(
                    &format!("Gate CHR-CUBE: S8 kL {kl} 24^3 e lies below -0.03 - the recorded miss (SPEC-LIT 136.8)"),
                    e < -band,
                );
                c.require(
                    &format!("Gate CHR-CUBE: the cause: S8's own limit (136.17) at kL {kl} sits inside 3 % (SPEC-LIT 136.8)"),
                    e_lim.abs() <= band,
                );
                c.require(
                    &format!("Gate CHR-CUBE: S8 kL {kl}: refining to 32^3 brings the step scheme closer (SPEC-LIT 136.8)"),
                    e32.abs() < e.abs(),
                );
                s8_miss = Some((kl, e, e_lim, e32));
            } else {
                // Inside 6 % only by cancellation: S4's own limit is
                // outside - a passing class A row with the violation as
                // its err.
                c.check_a(
                    &format!("Gate CHR-CUBE: S4 kL {kl} 24^3 is inside 6 % only by cancellation - S4's own limit (136.17) exceeds 6 % (SPEC-LIT 136.8)"),
                    (band - e_lim).max(0.0) as Scalar,
                    0.0,
                );
                s4_limit = e_lim;
            }
        }
    }

    // Jamaluddin & Smith (1988): NAMED, NOT RUN - the paper is closed and
    // was not read, so there is no table to check (136.6) against (SPEC-LIT
    // §136.3; docs/17 §5.3 rule 3). The day `reference/jamaluddin-smith1988`
    // appears, the transcription check must be built - the require below
    // fails then, on purpose.
    c.note(
        "Jamaluddin & Smith (1988) NAMED, NOT RUN: closed and not read (SPEC-LIT 136.3); (136.6) is the reference",
    );
    let dir = key::reference_dir().join("jamaluddin-smith1988");
    c.require(
        "Gate CHR-CUBE: reference/jamaluddin-smith1988 does not exist - when it appears, the (136.6) transcription check must be built (SPEC-LIT 136.8)",
        !dir.exists(),
    );

    let (kl, e, e_lim, e32) = s8_miss
        .ok_or_else(|| Error::Config("Gate CHR-CUBE: no kL 0.1 S8 miss measured".to_string()))?;
    c.report(GateReport {
        verdict: Verdict::Misses,
        how: How::Live,
        gate: GATE,
        against: "(136.6), the cube's wall-centre flux by the exact angular integral, SPEC-LIT 136.3",
        headline: format!(
            "S8 kL {kl} 24^3: e {e:+.4e} outside 3 %, by the step scheme's spatial error"
        ),
        detail: vec![
            format!(
                "32^3 gives {e32:+.4e} and S8's own limit (136.17) is {e_lim:+.4e}, inside \
                 3 %: the angular set is not what fails (SPEC-LIT 136.8)"
            ),
            format!(
                "S4 kL {kl} 24^3 passes its 6 % band only because the step scheme's spatial \
                 error cancels part of S4's angular error - S4's own limit (136.17) is \
                 {s4_limit:+.4e}, outside 6 % (SPEC-LIT 136.8)"
            ),
        ],
        uncertainty: Some(Uncertainty::SingleMesh(
            "24^3, reduced, with the 32^3 comparison beside it; no three-mesh study",
        )),
    });
    Ok(())
}
