// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT §143.1 - the 1-D RANS oracle.
//!
//! Fully developed turbulent pipe flow on the radial cell set of the TG0
//! wedge case, solved on the host in `f64` only. Written from: Menter 1994
//! and Menter, Kuntz & Langtry 2003 (SPEC-LIT §6.3); Wilcox 1988 via the
//! NASA TMR page https://tmbwg.github.io/turbmodels/wilcox.html,
//! DOI 10.2514/3.10041 (SPEC-LIT §6.2); Wilcox 2006 from the same page's
//! "Wilcox (2006) k-omega" section, DOI 10.2514/1.36541 (cited, not read);
//! Launder & Sharma 1974 (SPEC-LIT
//! §33); Spalart-Allmaras via https://tmbwg.github.io/turbmodels/spalart.html
//! (SPEC-LIT §56); Tucker 1998 (SPEC-LIT §6.6); wall-cell treatment
//! SPEC-LIT §6.4; pipe metrics from tools/cad/post.py `pipe_block`; the
//! Richardson extrapolation is Celik 2008 (SPEC-LIT §94).
//!
//! No GPL-licensed source was consulted.

use crate::{Error, Result};

/// TG0 pipe radius [m].
pub const TG0_R: f64 = 0.025;
/// TG0 wedge opening angle [deg].
pub const TG0_THETA_DEG: f64 = 5.0;
/// TG0 kinematic viscosity [m2/s].
pub const TG0_NU: f64 = 1.516e-5;
/// The largest TG0 friction Reynolds number the §114 tables reach.
pub const TG0_RE_TAU_MAX: f64 = 2358.0;

const KAPPA: f64 = 0.41;

// kOmegaSST constants, SPEC-LIT §6.3.
const SK1: f64 = 0.85;
const SW1: f64 = 0.5;
const BS1: f64 = 0.075;
const GK1: f64 = 5.0 / 9.0;
const SK2: f64 = 1.0;
const SW2: f64 = 0.856;
const BS2: f64 = 0.0828;
const GK2: f64 = 0.44;
const BSTAR: f64 = 0.09;
const A_SST: f64 = 0.31;
const B_SST: f64 = 1.0;
const C_SST: f64 = 10.0;

/// The radial one-dimensional pipe mesh of SPEC-LIT §143.1: `n` cells graded
/// from the wall, `faces` ordered axis -> wall, every radius already scaled
/// by `cos(theta/2)` so the TG0 wedge is a slice of the pipe.
#[derive(Debug, Clone)]
pub struct PipeMesh {
    pub r: f64,
    pub theta_deg: f64,
    pub faces: Vec<f64>,
    pub centres: Vec<f64>,
    pub volumes: Vec<f64>,
}

/// Viscosity and body force of a fully developed pipe case (eq 143.5).
#[derive(Debug, Clone, Copy)]
pub struct PipeCase {
    pub nu: f64,
    pub re_tau: f64,
    pub g: f64,
}

/// Wall distance for the turbulence models (SPEC-LIT §6.4, Tucker 1998).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WallDistance {
    /// Tucker's PDE distance solved on the same cells (§6.6).
    Tucker,
    /// `rho_wall - centre`, exact on this mesh.
    Exact,
}

/// The six oracle models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    Laminar,
    KOmegaSst,
    KOmega1988,
    Wilcox2006,
    LaunderSharma,
    SpalartAllmaras,
}

/// Sweep controls.
#[derive(Debug, Clone, Copy)]
pub struct Controls {
    pub tol: f64,
    pub max_iter: usize,
    pub relax: f64,
    pub wall_distance: WallDistance,
}

impl Controls {
    pub fn for_model(m: Model) -> Controls {
        let relax = match m {
            Model::Laminar => 1.0,
            Model::KOmegaSst | Model::KOmega1988 | Model::Wilcox2006 => 0.7,
            Model::LaunderSharma => 0.5,
            Model::SpalartAllmaras => 0.1,
        };
        Controls { tol: 1e-11, max_iter: 50000, relax, wall_distance: WallDistance::Tucker }
    }
}

/// k-omega 1988 (Wilcox) constants. `Default` is the TMR page set
/// (0.09, 3/40, 5/9, 0.5, 0.5); `spec_lit_6_2` the variant with beta = 0.072.
#[derive(Debug, Clone, Copy)]
pub struct KOmega1988Coeffs {
    pub beta_star: f64,
    pub beta: f64,
    pub gamma: f64,
    pub sigma_k: f64,
    pub sigma_w: f64,
}

impl Default for KOmega1988Coeffs {
    fn default() -> Self {
        KOmega1988Coeffs {
            beta_star: 0.09,
            beta: 3.0 / 40.0,
            gamma: 5.0 / 9.0,
            sigma_k: 0.5,
            sigma_w: 0.5,
        }
    }
}

impl KOmega1988Coeffs {
    pub fn spec_lit_6_2() -> Self {
        KOmega1988Coeffs { beta: 0.072, ..Default::default() }
    }
}

/// Spalart-Allmaras (no ft2) constants; `Default` is the standard set
/// (SPEC-LIT §56).
#[derive(Debug, Clone, Copy)]
pub struct SaCoeffs {
    pub cb1: f64,
    pub sigma: f64,
    pub cb2: f64,
    pub kappa: f64,
    pub cw2: f64,
    pub cw3: f64,
    pub cv1: f64,
    pub cv2: f64,
    pub cv3: f64,
    pub rlim: f64,
}

impl Default for SaCoeffs {
    fn default() -> Self {
        SaCoeffs {
            cb1: 0.1355,
            sigma: 2.0 / 3.0,
            cb2: 0.622,
            kappa: 0.41,
            cw2: 0.3,
            cw3: 2.0,
            cv1: 7.1,
            cv2: 0.7,
            cv3: 0.9,
            rlim: 10.0,
        }
    }
}

impl SaCoeffs {
    /// `cb1/kappa^2 + (1 + cb2)/sigma` = 3.2390678... (SPEC-LIT §56.4).
    pub fn cw1(&self) -> f64 {
        self.cb1 / (self.kappa * self.kappa) + (1.0 + self.cb2) / self.sigma
    }
}

/// One solved case. Fields a model does not use stay empty; `nut` is always
/// filled (zeros for the laminar solve).
#[derive(Debug, Clone)]
pub struct Solution {
    pub model: Model,
    pub u: Vec<f64>,
    pub nut: Vec<f64>,
    pub k: Vec<f64>,
    pub omega: Vec<f64>,
    pub eps_tilde: Vec<f64>,
    pub nu_tilde: Vec<f64>,
    pub iterations: usize,
    pub last_change: f64,
}

/// Integral pipe quantities, eq 143.13, as tools/cad/post.py `pipe_block`.
#[derive(Debug, Clone, Copy)]
pub struct PipeMetrics {
    pub u_b: f64,
    pub u_tau: f64,
    pub f: f64,
    pub u_axis: f64,
    pub defect: f64,
    pub nut_axis_over_utau_r: f64,
}

/// The grading ratio of eq 143.1: `q` with `h1 (q^n - 1)/(q - 1) = length`,
/// bisected exactly 200 steps on (1, 4).
pub fn solve_q(h1: f64, length: f64, n: usize) -> Result<f64> {
    let hn = h1 * n as f64;
    if hn >= length {
        return Err(Error::Config(format!(
            "rans1d solve_q: h1 * n = {hn:e} >= length = {length:e}"
        )));
    }
    let nf = n as f64;
    let mut lo = 1.0f64;
    let mut hi = 4.0f64;
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if h1 * (mid.powf(nf) - 1.0) / (mid - 1.0) > length {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok((lo + hi) / 2.0)
}

impl PipeMesh {
    /// The graded mesh of eqs 143.1-143.2: `h1` the first (wall) spacing,
    /// vertex radii counted FROM THE WALL, stored axis -> wall, the axis
    /// entry exactly 0, everything scaled by `cos(theta/2)`.
    pub fn graded(r: f64, theta_deg: f64, n: usize, h1: f64) -> Result<PipeMesh> {
        let q = solve_q(h1, r, n)?;
        let total = q.powf(n as f64) - 1.0;
        let cosf = (theta_deg.to_radians() / 2.0).cos();
        let mut faces: Vec<f64> = (0..=n)
            .rev()
            .map(|i| r * (1.0 - (q.powf(i as f64) - 1.0) / total) * cosf)
            .collect();
        faces[0] = 0.0;
        let mut volumes = vec![0.0f64; n];
        let mut centres = vec![0.0f64; n];
        for j in 0..n {
            let (a, b) = (faces[j], faces[j + 1]);
            volumes[j] = (b * b - a * a) / 2.0;
            centres[j] = 2.0 * (b * b * b - a * a * a) / (3.0 * (b * b - a * a));
        }
        Ok(PipeMesh { r, theta_deg, faces, centres, volumes })
    }

    /// The TG0 levels of SPEC-LIT §143.1: `n = 40 * 2^level`,
    /// `h1 = (R/2358) * 2^-level`.
    pub fn tg0(level: u32) -> Result<PipeMesh> {
        PipeMesh::graded(
            TG0_R,
            TG0_THETA_DEG,
            40usize << level,
            TG0_R / TG0_RE_TAU_MAX * 2.0f64.powi(-(level as i32)),
        )
    }

    /// The cell count.
    pub fn n(&self) -> usize {
        self.faces.len() - 1
    }

    /// `rho_n = R cos(theta/2)`, the wall face radius.
    pub fn rho_wall(&self) -> f64 {
        self.faces[self.n()]
    }

    /// Wall face to wall-cell-centre distance.
    pub fn delta_wall(&self) -> f64 {
        self.rho_wall() - self.centres[self.n() - 1]
    }
}

impl PipeCase {
    /// The TG0 pipe case: `u_tau = re_tau nu / R`, `g = 2 u_tau^2 / R`.
    pub fn tg0(re_tau: f64) -> PipeCase {
        let u = re_tau * TG0_NU / TG0_R;
        PipeCase { nu: TG0_NU, re_tau, g: 2.0 * u * u / TG0_R }
    }
}

/// Exact tridiagonal solve (Thomas algorithm). `di` the diagonal, `lo[i]`
/// the sub-diagonal entry of row i, `up[i]` the super-diagonal entry.
fn thomas(lo: &[f64], di: &[f64], up: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = di.len();
    let mut c = vec![0.0f64; n];
    let mut d = vec![0.0f64; n];
    c[0] = up[0] / di[0];
    d[0] = rhs[0] / di[0];
    for i in 1..n {
        let m = di[i] - lo[i] * c[i - 1];
        d[i] = (rhs[i] - lo[i] * d[i - 1]) / m;
        if i < n - 1 {
            c[i] = up[i] / m;
        }
    }
    let mut x = vec![0.0f64; n];
    x[n - 1] = d[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d[i] - c[i] * x[i + 1];
    }
    x
}

/// Coefficient of the internal face between cells j and j+1: the CELL
/// diffusivities interpolated at the face (eq 143.3).
fn face_coef(mesh: &PipeMesh, j: usize, gamma: &[f64]) -> f64 {
    let a = mesh.faces[j + 1];
    let d = mesh.centres[j + 1] - mesh.centres[j];
    let w = (mesh.centres[j + 1] - a) / d;
    (w * gamma[j] + (1.0 - w) * gamma[j + 1]) * a / d
}

/// The diffusion operator of eq 143.3: diagonal = sum of face coefficients,
/// off-diagonals their negation. The axis face has zero area and drops out.
fn diff_matrix(mesh: &PipeMesh, gamma: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = mesh.n();
    let (mut lo, mut di, mut up) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
    for j in 0..n - 1 {
        let c = face_coef(mesh, j, gamma);
        di[j] += c;
        di[j + 1] += c;
        up[j] -= c;
        lo[j + 1] -= c;
    }
    (lo, di, up)
}

/// Dirichlet wall face: `gamma_wall rho_wall / delta_wall` onto the wall
/// cell's diagonal, `gamma_wall rho_wall phi_wall / delta_wall` onto the rhs.
fn add_wall(mesh: &PipeMesh, gamma_wall: f64, phi_wall: f64, di: &mut [f64], rhs: &mut [f64]) {
    let n = mesh.n();
    let c = gamma_wall * mesh.rho_wall() / mesh.delta_wall();
    di[n - 1] += c;
    rhs[n - 1] += c * phi_wall;
}

/// The Gauss gradient of eq 143.4 for every cell; `phi_wall` is the wall
/// face value of `phi` (the axis face value is not needed: its radius is 0).
pub fn gauss_grad(mesh: &PipeMesh, phi: &[f64], phi_wall: f64) -> Vec<f64> {
    let n = mesh.n();
    let mut fv = vec![0.0f64; n + 1];
    for (f, ((&rho, (cp, cn)), (p0, p1))) in mesh.faces[1..n]
        .iter()
        .zip(mesh.centres.windows(2).map(|cc| (cc[0], cc[1])))
        .zip(phi.windows(2).map(|pp| (pp[0], pp[1])))
        .enumerate()
    {
        let w = (cn - rho) / (cn - cp);
        fv[f + 1] = w * p0 + (1.0 - w) * p1;
    }
    fv[n] = phi_wall;
    let mut g = vec![0.0f64; n];
    for j in 0..n {
        let (a, b) = (mesh.faces[j], mesh.faces[j + 1]);
        g[j] = 2.0 * (fv[j + 1] * b - fv[j] * a - phi[j] * (b - a)) / (b * b - a * a);
    }
    g
}

/// Cell wall distances (SPEC-LIT §6.4): Tucker's PDE distance, or the exact
/// `rho_wall - centre`.
pub fn wall_distance(mesh: &PipeMesh, kind: WallDistance) -> Vec<f64> {
    let n = mesh.n();
    match kind {
        WallDistance::Exact => (0..n).map(|j| mesh.rho_wall() - mesh.centres[j]).collect(),
        WallDistance::Tucker => {
            let (mut lo, mut di, mut up) =
                (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
            let mut rhs = vec![0.0f64; n];
            for j in 0..n - 1 {
                let a = mesh.faces[j + 1];
                let dd = mesh.centres[j + 1] - mesh.centres[j];
                let coef = a / dd;
                di[j] += coef;
                di[j + 1] += coef;
                up[j] -= coef;
                lo[j + 1] -= coef;
            }
            for (r, v) in rhs.iter_mut().zip(&mesh.volumes) {
                *r = *v;
            }
            add_wall(mesh, 1.0, 0.0, &mut di, &mut rhs);
            let phi = thomas(&lo, &di, &up, &rhs);
            let g = gauss_grad(mesh, &phi, 0.0);
            (0..n).map(|j| -g[j].abs() + (g[j] * g[j] + 2.0 * phi[j]).sqrt()).collect()
        }
    }
}

/// The momentum solve of eq 143.5: diffusion of U with `Gamma = nu + nut`,
/// `U_wall = 0`, body force `g V_P`; solved directly, never relaxed.
fn momentum(mesh: &PipeMesh, case: &PipeCase, nut: &[f64]) -> Vec<f64> {
    let n = mesh.n();
    let gamma: Vec<f64> = (0..n).map(|j| case.nu + nut[j]).collect();
    let (lo, mut di, up) = diff_matrix(mesh, &gamma);
    let mut rhs: Vec<f64> = mesh.volumes.iter().map(|v| case.g * v).collect();
    add_wall(mesh, case.nu, 0.0, &mut di, &mut rhs);
    thomas(&lo, &di, &up, &rhs)
}

/// `max_j |a_j - b_j|`.
fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max)
}

/// `max_j |a_j|` (the fields here are non-negative up to the solver floors).
fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0f64, |acc, v| acc.max(*v))
}

/// The common turbulent initial state: 0.12 of the laminar solution.
fn start_u(mesh: &PipeMesh, case: &PipeCase) -> Vec<f64> {
    let n = mesh.n();
    let mut u = momentum(mesh, case, &vec![0.0; n]);
    for v in u.iter_mut() {
        *v *= 0.12;
    }
    u
}

/// The laminar oracle: one direct momentum solve with `nut = 0`.
fn solve_laminar(mesh: &PipeMesh, case: &PipeCase) -> Solution {
    let n = mesh.n();
    let u = momentum(mesh, case, &vec![0.0; n]);
    Solution {
        model: Model::Laminar,
        u,
        nut: vec![0.0; n],
        k: Vec::new(),
        omega: Vec::new(),
        eps_tilde: Vec::new(),
        nu_tilde: Vec::new(),
        iterations: 1,
        last_change: 0.0,
    }
}

/// The SST/1988 wall cell (SPEC-LIT §6.4): `omega_wall =
/// sqrt(omega_log^2 + omega_vis^2)` and the laminar switch
/// `lw = exp(1/Gam)` with `Gam = -0.01 yp^4/(1 + 5 yp)` (0 when Gam >= -1e-30).
fn wall_omega(kc: f64, d: f64, nu: f64, beta_star: f64) -> (f64, f64) {
    let rt = beta_star.powf(0.25);
    let yp = rt * d * kc.sqrt() / nu;
    let om_log = kc.sqrt() / (rt * KAPPA * d);
    let om_vis = 6.0 * nu / (0.075 * d * d);
    let omega_wall = (om_log * om_log + om_vis * om_vis).sqrt();
    let gam = -0.01 * yp.powi(4) / (1.0 + 5.0 * yp);
    let lw = if gam < -1e-30 { (1.0 / gam).exp() } else { 0.0 };
    (omega_wall, lw)
}

/// The wall-cell production override `G[n-1]`.
fn wall_production(lw: f64, nu: f64, u_wall: f64, d: f64, kc: f64, beta_star: f64) -> f64 {
    lw * nu * (u_wall.abs() / d) * beta_star.powf(0.25) * kc.sqrt() / (KAPPA * d)
}

/// kOmegaSST, eqs 143.7-143.8, coefficients of SPEC-LIT §6.3.
fn solve_sst(mesh: &PipeMesh, case: &PipeCase, ctl: &Controls) -> Result<Solution> {
    let n = mesh.n();
    let nu = case.nu;
    let vol = &mesh.volumes;
    let y = wall_distance(mesh, ctl.wall_distance);
    let mut u = start_u(mesh, case);
    let mut k = vec![0.01f64; n];
    let mut om = vec![100.0f64; n];
    let mut nut = vec![0.0f64; n];
    let mut it = 0usize;
    let mut change = f64::INFINITY;
    loop {
        if it >= ctl.max_iter {
            return Err(Error::Config(format!(
                "rans1d KOmegaSst did not converge: {} sweeps, last change {:+.3e}",
                it, change
            )));
        }
        it += 1;
        let gu = gauss_grad(mesh, &u, 0.0);
        let s: Vec<f64> = gu.iter().map(|g| g.abs()).collect();
        let pr: Vec<f64> = gu.iter().map(|g| g * g).collect();
        let gk = gauss_grad(mesh, &k, 0.0);
        let gw = gauss_grad(mesh, &om, om[n - 1]);
        let mut f1 = vec![0.0f64; n];
        let mut f2 = vec![0.0f64; n];
        let mut sk = vec![0.0f64; n];
        let mut sw = vec![0.0f64; n];
        let mut gc = vec![0.0f64; n];
        let mut bc = vec![0.0f64; n];
        let mut gpr = vec![0.0f64; n];
        for j in 0..n {
            let kc = k[j].max(0.0);
            let w = om[j].max(1e-300);
            let cd = 2.0 * SW2 * gk[j] * gw[j] / w;
            let tl = kc.sqrt() / (BSTAR * w * y[j]);
            let vis = 500.0 * nu / (y[j] * y[j] * w);
            let a1 = tl.max(vis).min(4.0 * SW2 * kc / (cd.max(1e-10) * y[j] * y[j])).min(10.0);
            let a2 = (2.0 * tl).max(vis).min(100.0);
            f1[j] = a1.powi(4).tanh();
            f2[j] = a2.powi(2).tanh();
            sk[j] = f1[j] * SK1 + (1.0 - f1[j]) * SK2;
            sw[j] = f1[j] * SW1 + (1.0 - f1[j]) * SW2;
            gc[j] = f1[j] * GK1 + (1.0 - f1[j]) * GK2;
            bc[j] = f1[j] * BS1 + (1.0 - f1[j]) * BS2;
            nut[j] = A_SST * kc / (A_SST * w).max(B_SST * f2[j] * s[j]);
            gpr[j] = nut[j] * pr[j];
        }
        let d = mesh.delta_wall();
        let (om_wall, lw) = wall_omega(k[n - 1].max(0.0), d, nu, BSTAR);
        gpr[n - 1] = wall_production(lw, nu, u[n - 1], d, k[n - 1].max(0.0), BSTAR);
        // omega, eq 143.8 first line: no wall-face term, wall row pinned.
        let gamma: Vec<f64> = (0..n).map(|j| nu + sw[j] * nut[j]).collect();
        let (mut lo, mut di, up) = diff_matrix(mesh, &gamma);
        let mut rhs = vec![0.0f64; n];
        for j in 0..n {
            let w = om[j].max(1e-300);
            di[j] += bc[j] * om[j] * vol[j];
            rhs[j] = gc[j] * pr[j] * vol[j];
            let cross = 2.0 * (1.0 - f1[j]) * SW2 * gk[j] * gw[j] / (w * w);
            if cross > 0.0 {
                rhs[j] += cross * om[j] * vol[j];
            } else {
                di[j] -= cross * vol[j];
            }
        }
        lo[n - 1] = 0.0;
        di[n - 1] = 1.0;
        rhs[n - 1] = om_wall;
        let os = thomas(&lo, &di, &up, &rhs);
        let om_new: Vec<f64> =
            (0..n).map(|j| om[j] + ctl.relax * (os[j].max(1e-12) - om[j])).collect();
        // k, eq 143.8 second line: wall value 0, sink beta* omega, limiter.
        let gamma: Vec<f64> = (0..n).map(|j| nu + sk[j] * nut[j]).collect();
        let (lo, mut di, up) = diff_matrix(mesh, &gamma);
        let mut rhs = vec![0.0f64; n];
        for j in 0..n {
            di[j] += BSTAR * om[j] * vol[j];
            rhs[j] = gpr[j].min(C_SST * BSTAR * k[j].max(0.0) * om[j]) * vol[j];
        }
        add_wall(mesh, nu, 0.0, &mut di, &mut rhs);
        let ks = thomas(&lo, &di, &up, &rhs);
        let k_new: Vec<f64> =
            (0..n).map(|j| k[j] + ctl.relax * (ks[j].max(0.0) - k[j])).collect();
        // nut and U from the new pair.
        let nut_new: Vec<f64> = (0..n)
            .map(|j| A_SST * k_new[j].max(0.0) / (A_SST * om_new[j]).max(B_SST * f2[j] * s[j]))
            .collect();
        let u_new = momentum(mesh, case, &nut_new);
        change = (max_diff(&u_new, &u) / max_abs(&u_new))
            .max(max_diff(&k_new, &k) / max_abs(&k_new))
            .max((0..n).map(|j| (om_new[j] - om[j]).abs() / om_new[j]).fold(0.0f64, f64::max));
        u = u_new;
        k = k_new;
        om = om_new;
        nut = nut_new;
        let finite = u.iter().chain(&nut).chain(&k).chain(&om).all(|v| v.is_finite());
        if !finite {
            return Err(Error::Diverged {
                iteration: it,
                what: format!("rans1d KOmegaSst went non-finite, last change {:+.3e}", change),
            });
        }
        if change < ctl.tol {
            break;
        }
    }
    Ok(Solution {
        model: Model::KOmegaSst,
        u,
        nut,
        k,
        omega: om,
        eps_tilde: Vec::new(),
        nu_tilde: Vec::new(),
        iterations: it,
        last_change: change,
    })
}

/// k-omega 1988 (Wilcox), eq 143.9: the SST sweep without blending, the
/// limiter or the cross-diffusion; coefficients from `c`.
pub fn solve_k_omega_1988(
    mesh: &PipeMesh,
    case: &PipeCase,
    c: &KOmega1988Coeffs,
    ctl: &Controls,
) -> Result<Solution> {
    let n = mesh.n();
    let nu = case.nu;
    let vol = &mesh.volumes;
    let mut u = start_u(mesh, case);
    let mut k = vec![0.01f64; n];
    let mut om = vec![100.0f64; n];
    let mut nut = vec![0.0f64; n];
    let mut it = 0usize;
    let mut change = f64::INFINITY;
    loop {
        if it >= ctl.max_iter {
            return Err(Error::Config(format!(
                "rans1d KOmega1988 did not converge: {} sweeps, last change {:+.3e}",
                it, change
            )));
        }
        it += 1;
        let gu = gauss_grad(mesh, &u, 0.0);
        let pr: Vec<f64> = gu.iter().map(|g| g * g).collect();
        // nut = kc/omega, G = nut P, with the wall-cell G override.
        let mut gpr = vec![0.0f64; n];
        for j in 0..n {
            nut[j] = k[j].max(0.0) / om[j].max(1e-300);
            gpr[j] = nut[j] * pr[j];
        }
        let d = mesh.delta_wall();
        let (om_wall, lw) = wall_omega(k[n - 1].max(0.0), d, nu, c.beta_star);
        gpr[n - 1] = wall_production(lw, nu, u[n - 1], d, k[n - 1].max(0.0), c.beta_star);
        // omega: no wall-face term, wall row pinned to omega_wall.
        let gamma: Vec<f64> = (0..n).map(|j| nu + c.sigma_w * nut[j]).collect();
        let (mut lo, mut di, up) = diff_matrix(mesh, &gamma);
        let mut rhs = vec![0.0f64; n];
        for j in 0..n {
            di[j] += c.beta * om[j] * vol[j];
            rhs[j] = c.gamma * pr[j] * vol[j];
        }
        lo[n - 1] = 0.0;
        di[n - 1] = 1.0;
        rhs[n - 1] = om_wall;
        let os = thomas(&lo, &di, &up, &rhs);
        let om_new: Vec<f64> =
            (0..n).map(|j| om[j] + ctl.relax * (os[j].max(1e-12) - om[j])).collect();
        // k: wall value 0, sink beta* omega, source G.
        let gamma: Vec<f64> = (0..n).map(|j| nu + c.sigma_k * nut[j]).collect();
        let (lo, mut di, up) = diff_matrix(mesh, &gamma);
        let mut rhs = vec![0.0f64; n];
        for j in 0..n {
            di[j] += c.beta_star * om[j] * vol[j];
            rhs[j] = gpr[j] * vol[j];
        }
        add_wall(mesh, nu, 0.0, &mut di, &mut rhs);
        let ks = thomas(&lo, &di, &up, &rhs);
        let k_new: Vec<f64> =
            (0..n).map(|j| k[j] + ctl.relax * (ks[j].max(0.0) - k[j])).collect();
        let nut_new: Vec<f64> =
            (0..n).map(|j| k_new[j].max(0.0) / om_new[j].max(1e-300)).collect();
        let u_new = momentum(mesh, case, &nut_new);
        change = (max_diff(&u_new, &u) / max_abs(&u_new))
            .max(max_diff(&k_new, &k) / max_abs(&k_new))
            .max((0..n).map(|j| (om_new[j] - om[j]).abs() / om_new[j]).fold(0.0f64, f64::max));
        u = u_new;
        k = k_new;
        om = om_new;
        nut = nut_new;
        let finite = u.iter().chain(&nut).chain(&k).chain(&om).all(|v| v.is_finite());
        if !finite {
            return Err(Error::Diverged {
                iteration: it,
                what: format!("rans1d KOmega1988 went non-finite, last change {:+.3e}", change),
            });
        }
        if change < ctl.tol {
            break;
        }
    }
    Ok(Solution {
        model: Model::KOmega1988,
        u,
        nut,
        k,
        omega: om,
        eps_tilde: Vec::new(),
        nu_tilde: Vec::new(),
        iterations: it,
        last_change: change,
    })
}

/// Wilcox 2006 k-omega constants (TMR "Wilcox (2006) k-omega"; DOI
/// 10.2514/1.36541, cited, not read). `f_beta = (1 + f_beta_num chi)/
/// (1 + f_beta_den chi)`, the stress-limiter coefficient `c_lim` and the
/// cross-diffusion coefficient `sigma_d` (used where grad k . grad omega > 0).
#[derive(Debug, Clone, Copy)]
pub struct Wilcox2006Coeffs {
    pub beta_star: f64,
    pub beta_0: f64,
    pub gamma: f64,
    pub sigma_k: f64,
    pub sigma_w: f64,
    pub sigma_d: f64,
    pub c_lim: f64,
    pub f_beta_num: f64,
    pub f_beta_den: f64,
}

impl Default for Wilcox2006Coeffs {
    fn default() -> Self {
        Wilcox2006Coeffs {
            beta_star: 0.09,
            beta_0: 0.0708,
            gamma: 13.0 / 25.0,
            sigma_k: 0.6,
            sigma_w: 0.5,
            sigma_d: 1.0 / 8.0,
            c_lim: 7.0 / 8.0,
            f_beta_num: 85.0,
            f_beta_den: 100.0,
        }
    }
}

/// Wilcox 2006 `f_beta = (1 + 85 chi)/(1 + 100 chi)` with
/// `chi = |Omega_ij Omega_jk Shat_ki|/(beta* omega)^3` (sum over i, j, k):
/// `S_ij = (du_i/dx_j + du_j/dx_i)/2`, `Omega_ij = (du_i/dx_j -
/// du_j/dx_i)/2`, `Shat_ki = S_ki - (1/2)(du_m/dx_m) delta_ki`, with
/// `grad_u[i][j] = du_i/dx_j`. The plain triple loop; no FMA, no shortcuts.
pub fn wilcox2006_f_beta(c: &Wilcox2006Coeffs, grad_u: &[[f64; 3]; 3], omega: f64) -> f64 {
    let mut s = [[0.0f64; 3]; 3];
    let mut o = [[0.0f64; 3]; 3];
    let mut shat = [[0.0f64; 3]; 3];
    let mut div = 0.0f64;
    for (m, row) in grad_u.iter().enumerate() {
        div += row[m];
    }
    for i in 0..3 {
        for j in 0..3 {
            s[i][j] = (grad_u[i][j] + grad_u[j][i]) / 2.0;
            o[i][j] = (grad_u[i][j] - grad_u[j][i]) / 2.0;
            shat[i][j] = s[i][j];
        }
        shat[i][i] -= div / 2.0;
    }
    let mut sum = 0.0f64;
    for i in 0..3 {
        for (j, oij) in o[i].iter().enumerate() {
            for k in 0..3 {
                sum += oij * o[j][k] * shat[k][i];
            }
        }
    }
    let chi = sum.abs() / (c.beta_star * omega).powi(3);
    (1.0 + c.f_beta_num * chi) / (1.0 + c.f_beta_den * chi)
}

/// The Wilcox 2006 cross-diffusion term `(sigma_d/omega) grad k . grad
/// omega`: exactly positive zero unless the dot product is strictly
/// positive.
pub fn wilcox2006_cross_diffusion(
    c: &Wilcox2006Coeffs,
    grad_k: &[f64; 3],
    grad_w: &[f64; 3],
    omega: f64,
) -> f64 {
    let dot = grad_k[0] * grad_w[0] + grad_k[1] * grad_w[1] + grad_k[2] * grad_w[2];
    if dot > 0.0 {
        c.sigma_d / omega * dot
    } else {
        0.0
    }
}

/// The stress limiter's `omega_hat = max(omega, C_lim sqrt(2 Sbar_ij
/// Sbar_ij / beta*))`: in the pipe `2 S_ij S_ij = (dU/drho)^2`, so the
/// second argument is `|dU/drho|`.
pub fn wilcox2006_omega_hat(c: &Wilcox2006Coeffs, omega: f64, du_drho: f64) -> f64 {
    omega.max(c.c_lim * du_drho.abs() / c.beta_star.sqrt())
}

/// Wilcox 2006 k-omega, the TMR page's "Wilcox (2006) k-omega" equations in
/// the same discretisation as `solve_k_omega_1988`: the `k/omega`
/// diffusivities, `beta = beta_0 f_beta`, the `omega_hat` stress limiter and
/// the `sigma_d/omega` cross-diffusion source (positive zero when
/// `grad k . grad omega <= 0`).
pub fn solve_wilcox_2006(
    mesh: &PipeMesh,
    case: &PipeCase,
    c: &Wilcox2006Coeffs,
    ctl: &Controls,
) -> Result<Solution> {
    let n = mesh.n();
    let nu = case.nu;
    let vol = &mesh.volumes;
    let mut u = start_u(mesh, case);
    let mut k = vec![0.01f64; n];
    let mut om = vec![100.0f64; n];
    let mut nut = vec![0.0f64; n];
    let mut it = 0usize;
    let mut change = f64::INFINITY;
    loop {
        if it >= ctl.max_iter {
            return Err(Error::Config(format!(
                "rans1d Wilcox2006 did not converge: {} sweeps, last change {:+.3e}",
                it, change
            )));
        }
        it += 1;
        let gu = gauss_grad(mesh, &u, 0.0);
        let gk = gauss_grad(mesh, &k, 0.0);
        let gw = gauss_grad(mesh, &om, om[n - 1]);
        let pr: Vec<f64> = gu.iter().map(|g| g * g).collect();
        // nut = kc/omega_hat, G = nut P, with the wall-cell G override.
        let ohat: Vec<f64> = (0..n).map(|j| wilcox2006_omega_hat(c, om[j], gu[j])).collect();
        let mut gpr = vec![0.0f64; n];
        for j in 0..n {
            nut[j] = k[j].max(0.0) / ohat[j];
            gpr[j] = nut[j] * pr[j];
        }
        let d = mesh.delta_wall();
        let (om_wall, lw) = wall_omega(k[n - 1].max(0.0), d, nu, c.beta_star);
        gpr[n - 1] = wall_production(lw, nu, u[n - 1], d, k[n - 1].max(0.0), c.beta_star);
        // omega: beta = beta_0 f_beta, the sigma_d cross-diffusion source,
        // no wall-face term, wall row pinned to omega_wall.
        let gamma: Vec<f64> = (0..n).map(|j| nu + c.sigma_w * k[j].max(0.0) / om[j]).collect();
        let (mut lo, mut di, up) = diff_matrix(mesh, &gamma);
        let mut rhs = vec![0.0f64; n];
        for j in 0..n {
            let beta_j = c.beta_0
                * wilcox2006_f_beta(c, &[[0.0, gu[j], 0.0], [0.0; 3], [0.0; 3]], om[j]);
            di[j] += beta_j * om[j] * vol[j];
            rhs[j] = c.gamma * (om[j] / ohat[j]) * pr[j] * vol[j]
                + wilcox2006_cross_diffusion(c, &[gk[j], 0.0, 0.0], &[gw[j], 0.0, 0.0], om[j])
                    * vol[j];
        }
        lo[n - 1] = 0.0;
        di[n - 1] = 1.0;
        rhs[n - 1] = om_wall;
        let os = thomas(&lo, &di, &up, &rhs);
        let om_new: Vec<f64> =
            (0..n).map(|j| om[j] + ctl.relax * (os[j].max(1e-12) - om[j])).collect();
        // k: wall value 0, sink beta* omega, source G.
        let gamma: Vec<f64> = (0..n).map(|j| nu + c.sigma_k * k[j].max(0.0) / om[j]).collect();
        let (lo, mut di, up) = diff_matrix(mesh, &gamma);
        let mut rhs = vec![0.0f64; n];
        for j in 0..n {
            di[j] += c.beta_star * om[j] * vol[j];
            rhs[j] = gpr[j] * vol[j];
        }
        add_wall(mesh, nu, 0.0, &mut di, &mut rhs);
        let ks = thomas(&lo, &di, &up, &rhs);
        let k_new: Vec<f64> =
            (0..n).map(|j| k[j] + ctl.relax * (ks[j].max(0.0) - k[j])).collect();
        let nut_new: Vec<f64> = (0..n)
            .map(|j| k_new[j].max(0.0) / wilcox2006_omega_hat(c, om_new[j], gu[j]))
            .collect();
        let u_new = momentum(mesh, case, &nut_new);
        change = (max_diff(&u_new, &u) / max_abs(&u_new))
            .max(max_diff(&k_new, &k) / max_abs(&k_new))
            .max((0..n).map(|j| (om_new[j] - om[j]).abs() / om_new[j]).fold(0.0f64, f64::max));
        u = u_new;
        k = k_new;
        om = om_new;
        nut = nut_new;
        let finite = u.iter().chain(&nut).chain(&k).chain(&om).all(|v| v.is_finite());
        if !finite {
            return Err(Error::Diverged {
                iteration: it,
                what: format!("rans1d Wilcox2006 went non-finite, last change {:+.3e}", change),
            });
        }
        if change < ctl.tol {
            break;
        }
    }
    Ok(Solution {
        model: Model::Wilcox2006,
        u,
        nut,
        k,
        omega: om,
        eps_tilde: Vec::new(),
        nu_tilde: Vec::new(),
        iterations: it,
        last_change: change,
    })
}

/// Launder-Sharma low-Re k-epsilon, eq 143.10 (SPEC-LIT §33.1).
fn solve_launder_sharma(mesh: &PipeMesh, case: &PipeCase, ctl: &Controls) -> Result<Solution> {
    const CMU: f64 = 0.09;
    const C1E: f64 = 1.44;
    const C2E: f64 = 1.92;
    const SIGK: f64 = 1.0;
    const SIGE: f64 = 1.3;
    let n = mesh.n();
    let nu = case.nu;
    let vol = &mesh.volumes;
    let nut_of = |kk: &[f64], ee: &[f64]| -> Vec<f64> {
        kk.iter().zip(ee).map(|(kv, ev)| {
            let kc = kv.max(1e-30);
            let ec = ev.max(1e-30);
            let rt = kc * kc / (nu * ec);
            let fmu = (-3.4 / (1.0 + rt / 50.0).powi(2)).exp();
            CMU * fmu * kc * kc / ec
        }).collect()
    };
    let mut u = start_u(mesh, case);
    let mut k = vec![0.01f64; n];
    let mut eps = vec![1.0f64; n];
    let mut nut = nut_of(&k, &eps);
    let mut it = 0usize;
    let mut change = f64::INFINITY;
    loop {
        if it >= ctl.max_iter {
            return Err(Error::Config(format!(
                "rans1d LaunderSharma did not converge: {} sweeps, last change {:+.3e}",
                it, change
            )));
        }
        it += 1;
        let gu = gauss_grad(mesh, &u, 0.0);
        let pr: Vec<f64> = gu.iter().map(|g| g * g).collect();
        let kc: Vec<f64> = k.iter().map(|v| v.max(1e-30)).collect();
        let ec: Vec<f64> = eps.iter().map(|v| v.max(1e-30)).collect();
        let f2: Vec<f64> = kc.iter().zip(&ec).map(|(kv, ev)| {
            let rt = kv * kv / (nu * ev);
            1.0 - 0.3 * (-rt * rt).exp()
        }).collect();
        let gpr: Vec<f64> = (0..n).map(|j| nut[j] * pr[j]).collect();
        // D and E, eq 143.10: wall damping terms; c is the cell centre.
        let sq: Vec<f64> = kc.iter().map(|v| v.sqrt()).collect();
        let gs = gauss_grad(mesh, &sq, 0.0);
        let dterm: Vec<f64> = gs.iter().map(|g| 2.0 * nu * g * g).collect();
        let gg = gauss_grad(mesh, &gu, gu[n - 1]);
        let eterm: Vec<f64> = (0..n)
            .map(|j| {
                2.0 * nu * nut[j]
                    * (gg[j] * gg[j] + (gu[j] / mesh.centres[j]) * (gu[j] / mesh.centres[j]))
            })
            .collect();
        // eps_tilde: wall value 0, sink C2 f2 ec/kc, sources C1 (ec/kc) G and E.
        let gamma: Vec<f64> = (0..n).map(|j| nu + nut[j] / SIGE).collect();
        let (lo, mut di, up) = diff_matrix(mesh, &gamma);
        let mut rhs = vec![0.0f64; n];
        for j in 0..n {
            di[j] += C2E * f2[j] * (ec[j] / kc[j]) * vol[j];
            rhs[j] = (C1E * (ec[j] / kc[j]) * gpr[j] + eterm[j]) * vol[j];
        }
        add_wall(mesh, nu, 0.0, &mut di, &mut rhs);
        let es = thomas(&lo, &di, &up, &rhs);
        let eps_new: Vec<f64> =
            (0..n).map(|j| eps[j] + ctl.relax * (es[j].max(1e-30) - eps[j])).collect();
        // k: wall value 0, sink (ec + D)/kc, source G.
        let gamma: Vec<f64> = (0..n).map(|j| nu + nut[j] / SIGK).collect();
        let (lo, mut di, up) = diff_matrix(mesh, &gamma);
        let mut rhs = vec![0.0f64; n];
        for j in 0..n {
            di[j] += (ec[j] + dterm[j]) / kc[j] * vol[j];
            rhs[j] = gpr[j] * vol[j];
        }
        add_wall(mesh, nu, 0.0, &mut di, &mut rhs);
        let ks = thomas(&lo, &di, &up, &rhs);
        let k_new: Vec<f64> =
            (0..n).map(|j| k[j] + ctl.relax * (ks[j].max(1e-30) - k[j])).collect();
        let nut_new = nut_of(&k_new, &eps_new);
        let u_new = momentum(mesh, case, &nut_new);
        change = (max_diff(&u_new, &u) / max_abs(&u_new))
            .max(max_diff(&k_new, &k) / max_abs(&k_new))
            .max(max_diff(&eps_new, &eps) / max_abs(&eps_new));
        u = u_new;
        k = k_new;
        eps = eps_new;
        nut = nut_new;
        let finite = u.iter().chain(&nut).chain(&k).chain(&eps).all(|v| v.is_finite());
        if !finite {
            return Err(Error::Diverged {
                iteration: it,
                what: format!("rans1d LaunderSharma went non-finite, last change {:+.3e}", change),
            });
        }
        if change < ctl.tol {
            break;
        }
    }
    Ok(Solution {
        model: Model::LaunderSharma,
        u,
        nut,
        k,
        omega: Vec::new(),
        eps_tilde: eps,
        nu_tilde: Vec::new(),
        iterations: it,
        last_change: change,
    })
}

/// Spalart-Allmaras noft2, eq 143.11 (SPEC-LIT §56.1-56.3).
fn solve_sa(mesh: &PipeMesh, case: &PipeCase, ctl: &Controls) -> Result<Solution> {
    let c = SaCoeffs::default();
    let n = mesh.n();
    let nu = case.nu;
    let vol = &mesh.volumes;
    let d = wall_distance(mesh, ctl.wall_distance);
    let mut u = start_u(mesh, case);
    let mut nt = vec![50.0 * nu; n];
    let mut nut;
    let mut it = 0usize;
    let mut change = f64::INFINITY;
    loop {
        if it >= ctl.max_iter {
            return Err(Error::Config(format!(
                "rans1d SpalartAllmaras did not converge: {} sweeps, last change {:+.3e}",
                it, change
            )));
        }
        it += 1;
        let gu = gauss_grad(mesh, &u, 0.0);
        let gn = gauss_grad(mesh, &nt, 0.0);
        let (mut prod, mut dest) = (vec![0.0f64; n], vec![0.0f64; n]);
        for j in 0..n {
            let (p, dq) = sa_point_terms(&c, nu, nt[j], gu[j].abs(), d[j]);
            prod[j] = p;
            dest[j] = dq;
        }
        // Diffusion with the face nu~ interpolated: Gamma_f = (nu + nu~_f)/sigma.
        let (mut lo, mut di, mut up) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]);
        let mut rhs = vec![0.0f64; n];
        for j in 0..n - 1 {
            let a = mesh.faces[j + 1];
            let dd = mesh.centres[j + 1] - mesh.centres[j];
            let w = (mesh.centres[j + 1] - a) / dd;
            let coef = (nu + w * nt[j] + (1.0 - w) * nt[j + 1]) / c.sigma * a / dd;
            di[j] += coef;
            di[j + 1] += coef;
            up[j] -= coef;
            lo[j + 1] -= coef;
        }
        // Implicit destruction: the diagonal gets cw1 fw nu~_old/d^2 * V, the
        // coefficient that multiplies the unknown nu~, so diagonal * nu~
        // equals the destruction cw1 fw (nu~/d)^2 * V.
        for j in 0..n {
            di[j] += dest[j] / nt[j] * vol[j];
            rhs[j] = (prod[j] + c.cb2 / c.sigma * gn[j] * gn[j]) * vol[j];
        }
        add_wall(mesh, nu / c.sigma, 0.0, &mut di, &mut rhs);
        let ns = thomas(&lo, &di, &up, &rhs);
        let nt_new: Vec<f64> =
            (0..n).map(|j| nt[j] + ctl.relax * (ns[j].max(0.0) - nt[j])).collect();
        let nut_new: Vec<f64> =
            (0..n).map(|j| nt_new[j] * fv1(nt_new[j] / nu, c.cv1)).collect();
        let u_new = momentum(mesh, case, &nut_new);
        change = (max_diff(&u_new, &u) / max_abs(&u_new))
            .max(max_diff(&nt_new, &nt) / max_abs(&nt_new));
        u = u_new;
        nt = nt_new;
        nut = nut_new;
        let finite = u.iter().chain(&nut).chain(&nt).all(|v| v.is_finite());
        if !finite {
            return Err(Error::Diverged {
                iteration: it,
                what: format!("rans1d SpalartAllmaras went non-finite, last change {:+.3e}", change),
            });
        }
        if change < ctl.tol {
            break;
        }
    }
    Ok(Solution {
        model: Model::SpalartAllmaras,
        u,
        nut,
        k: Vec::new(),
        omega: Vec::new(),
        eps_tilde: Vec::new(),
        nu_tilde: nt,
        iterations: it,
        last_change: change,
    })
}

/// `fv1(chi) = chi^3/(chi^3 + cv1^3)`.
fn fv1(chi: f64, cv1: f64) -> f64 {
    let c3 = cv1 * cv1 * cv1;
    chi * chi * chi / (chi * chi * chi + c3)
}

/// The pointwise SA terms of eq 143.11 (SPEC-LIT §56.1-56.3):
/// `(production, destruction)` = `(cb1 Stil nu~, cw1 fw (nu~/d)^2)`.
pub fn sa_point_terms(c: &SaCoeffs, nu: f64, nu_tilde: f64, vort: f64, d: f64) -> (f64, f64) {
    let chi = nu_tilde / nu;
    let f2v = 1.0 - chi / (1.0 + chi * fv1(chi, c.cv1));
    let sbar = nu_tilde * f2v / (c.kappa * c.kappa * d * d);
    let stil = if sbar >= -c.cv2 * vort {
        vort + sbar
    } else {
        vort + vort * (c.cv2 * c.cv2 * vort + c.cv3 * sbar)
            / ((c.cv3 - 2.0 * c.cv2) * vort - sbar)
    };
    let r = if stil > 0.0 {
        (nu_tilde / (stil * c.kappa * c.kappa * d * d)).min(c.rlim)
    } else {
        c.rlim
    };
    let g = r + c.cw2 * (r.powi(6) - r);
    let fw = g * ((1.0 + c.cw3.powi(6)) / (g.powi(6) + c.cw3.powi(6))).powf(1.0 / 6.0);
    (c.cb1 * stil * nu_tilde, c.cw1() * fw * (nu_tilde / d) * (nu_tilde / d))
}

/// Solve one model on the pipe mesh (SPEC-LIT §143.1). Not converging
/// within `Controls::max_iter` is `Error::Config`; a non-finite field is
/// `Error::Diverged`.
pub fn solve(mesh: &PipeMesh, case: &PipeCase, model: Model, ctl: &Controls) -> Result<Solution> {
    match model {
        Model::Laminar => Ok(solve_laminar(mesh, case)),
        Model::KOmegaSst => solve_sst(mesh, case, ctl),
        Model::KOmega1988 => solve_k_omega_1988(mesh, case, &KOmega1988Coeffs::default(), ctl),
        Model::Wilcox2006 => solve_wilcox_2006(mesh, case, &Wilcox2006Coeffs::default(), ctl),
        Model::LaunderSharma => solve_launder_sharma(mesh, case, ctl),
        Model::SpalartAllmaras => solve_sa(mesh, case, ctl),
    }
}

/// Eq 143.13, as tools/cad/post.py `pipe_block`.
pub fn metrics(mesh: &PipeMesh, case: &PipeCase, sol: &Solution) -> PipeMetrics {
    let n = mesh.n();
    let vol: f64 = mesh.volumes.iter().sum();
    let u_b: f64 = mesh.volumes.iter().zip(&sol.u).map(|(v, x)| v * x).sum::<f64>() / vol;
    let u_tau = (case.nu * sol.u[n - 1] / mesh.delta_wall()).sqrt();
    let f = 8.0 * (u_tau / u_b) * (u_tau / u_b);
    let (c0, c1) = (mesh.centres[0], mesh.centres[1]);
    let u_axis = sol.u[0] - (sol.u[1] - sol.u[0]) * c0 * c0 / (c1 * c1 - c0 * c0);
    PipeMetrics {
        u_b,
        u_tau,
        f,
        u_axis,
        defect: (u_axis - u_b) / u_tau,
        nut_axis_over_utau_r: sol.nut[0] / (u_tau * TG0_R),
    }
}

/// Total shear stress on every face: the `n - 1` internal faces then the
/// wall face, each `-Gamma_f (U_N - U_P)/d_f`.
pub fn face_total_stress(mesh: &PipeMesh, case: &PipeCase, sol: &Solution) -> Vec<f64> {
    let n = mesh.n();
    let mut out = Vec::with_capacity(n);
    for j in 0..n - 1 {
        let d = mesh.centres[j + 1] - mesh.centres[j];
        let w = (mesh.centres[j + 1] - mesh.faces[j + 1]) / d;
        let gam = w * (case.nu + sol.nut[j]) + (1.0 - w) * (case.nu + sol.nut[j + 1]);
        out.push(-gam * (sol.u[j + 1] - sol.u[j]) / d);
    }
    out.push(case.nu * sol.u[n - 1] / mesh.delta_wall());
    out
}

/// Celik 2008 (94.3)-(94.4) at a constant refinement `ratio`:
/// `(p, extrapolated)`.
pub fn richardson_constant_ratio(fine: f64, medium: f64, coarse: f64, ratio: f64) -> (f64, f64) {
    let p = ((coarse - medium) / (medium - fine)).ln() / ratio.ln();
    let ext = fine + (fine - medium) / (ratio.powf(p) - 1.0);
    (p, ext)
}

/// The Prandtl smooth-pipe friction line `1/sqrt(f) = 2 log10(2 sqrt(8)
/// re_tau) - 0.8` (tools/cad/turb_integral.py `f_prandtl`, TM 1218 eq 16.26
/// in closed form from `re_tau`).
pub fn f_prandtl(re_tau: f64) -> f64 {
    (2.0 * (2.0 * 8.0f64.sqrt() * re_tau).log10() - 0.8).powi(-2)
}

/// Blasius `f = 0.3164 re_d^-0.25` (tools/cad/turb_integral.py
/// `f_blasius`, TM 1218 eq 16.4).
pub fn f_blasius(re_d: f64) -> f64 {
    0.3164 * re_d.powf(-0.25)
}

/// The four TG0 band metrics of SPEC-LIT §143.2 for one solved pipe case.
#[derive(Debug, Clone, Copy)]
pub struct Tg0Bands {
    pub re_d: f64,
    pub re_tau: f64,
    pub f_vs_prandtl: f64,
    pub f_vs_blasius: f64,
    pub loglaw_dev: f64,
    pub loglaw_cells: usize,
    pub defect_vs_4_07: f64,
}

/// The band metrics, as tools/cad/post.py `pipe_block`: `re_d` and the
/// MEASURED `re_tau`; `f` against the Prandtl line at the INPUT `re_tau`
/// and against Blasius; the worst log-law deviation `|u+ - (2.5 ln y+ +
/// 5.5)|` over cells with `30 <= y+ <= 0.2 re_tau` (NaN, with 0 cells, if
/// there is no such cell); and the wake defect against 4.07.
pub fn tg0_bands(mesh: &PipeMesh, case: &PipeCase, sol: &Solution, m: &PipeMetrics) -> Tg0Bands {
    let rho_w = mesh.rho_wall();
    let re_d = m.u_b * 2.0 * rho_w / case.nu;
    let re_tau = m.u_tau * rho_w / case.nu;
    let f_vs_prandtl = m.f / f_prandtl(case.re_tau) - 1.0;
    let f_vs_blasius = m.f / f_blasius(re_d) - 1.0;
    let (mut dev, mut cells) = (f64::NAN, 0usize);
    for j in 0..mesh.n() {
        let yplus = (rho_w - mesh.centres[j]) * m.u_tau / case.nu;
        if 30.0 <= yplus && yplus <= 0.2 * re_tau {
            let d = (sol.u[j] / m.u_tau - (2.5 * yplus.ln() + 5.5)).abs();
            dev = dev.max(d);
            cells += 1;
        }
    }
    Tg0Bands {
        re_d,
        re_tau,
        f_vs_prandtl,
        f_vs_blasius,
        loglaw_dev: dev,
        loglaw_cells: cells,
        defect_vs_4_07: m.defect / 4.07 - 1.0,
    }
}
