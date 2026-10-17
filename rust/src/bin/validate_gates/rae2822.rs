// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! **Gate CMP-RAE (SPEC-LIT §127.6, §127.6.8): RAE 2822 Case 6 with
//! Spalart-Allmaras against the AGARD AR-138 experiment, a live coarse
//! smoke and the three recorded C-meshes, replayed.**
//!
//! §127.5.1's C-mesh of `rae2822_stations` carries air (`gamma = 1.4`,
//! Sutherland) at Case 6's corrected conditions - `M = 0.729`, `alpha =
//! 2.31` degrees, `Re_c = 6.5e6`, the NPARC archive's restatement of Cook,
//! McDonald & Firmin (1979) - marched steady with SIMPLE (five outer
//! iterations per step) and LTS CFL 0.1 on the span-1e-5 C-meshes until
//! no cell's `M_i` moves by more than `1e-5` across a chunk of 200
//! iterations (capped at 20000), the
//! Spalart-Allmaras model of §127.1 attached as `ofgpu-compressible`
//! attaches it. The key is the archive's measured surface `Cp`
//! (`rae2822-cp-case6`, 103 points), read through the answer-key reader
//! in `f64` in BOTH builds. Three rows, class P: the upper-surface shock
//! `x/c` (the sonic crossing (127.33)) within `+-0.04` of the
//! experiment's, the lower-surface `Cp` RMS within `0.05`, and `Cl`
//! within 6 % of the experiment's own integrated `Cl` - the key's `Cp`
//! integrated once, `Cl_exp = Cn cos(alpha)` with the `-Ca sin(alpha)`
//! term dropped (8.3e-4, 0.11 %). §94's `vv::grid_study` refuses two
//! levels, so the gate runs THREE C-meshes L0/L1/L2 - L1 is CMP-26's
//! default gate mesh - and the section runs a live coarse smoke (2000
//! iterations on `CMeshSpec::coarse()` at the gate's span: no bound, a converging residual,
//! a shock present) beside the replayed records, which is what "beyond
//! 10 min" looks like here (docs/17 5.3 rule 4).
//!
//! Written from:
//!   ofgpu `SPEC-LIT.md` §127.6 (the gate table), §127.6.8 (this gate as
//!     built), §127.5.1 (the C-mesh), §127.1 (SA in conservative form,
//!     via CMP-25's coupling), §122 (the gas), §125.4 (the local time
//!     step), §126.10 (the characteristic far field), §94 (the mesh
//!     study) and §69 (the verdict registry)
//!   P. H. Cook, M. A. McDonald and M. C. P. Firmin, "Aerofoil RAE 2822 -
//!     Pressure Distributions, and Boundary Layer and Wake Measurements",
//!     in Experimental Data Base for Computer Program Assessment, AGARD
//!     AR-138 (1979), Case 6, as the NASA NPARC Alliance Validation
//!     Archive reproduces it - read through the answer-key reader with
//!     its sha256 printed; AGARD-AR-138 itself was NOT read
//!   NACA Report 1135 (1953), eq. (44), the isentropic `p/p_t` taken at
//!     `M = 1` and at `M_inf`, for the sonic pressure coefficient (127.31,
//!     derived), as §127.2 cites the report
//!   the TMR SA page, for the free-stream `nu~ = 3 nu`, as §127.1 cites
//!     it
//! No GPL-licensed source was consulted.

use std::collections::BTreeMap;
use std::time::Instant;

use ofgpu::compressible::bc as cbc;
use ofgpu::compressible::pressure::Retarded;
use ofgpu::compressible::thermo::PerfectGas;
use ofgpu::compressible::{Compressible, CompressibleControls, CompressibleState};
use ofgpu::compressible::gatemesh::{airfoil_cmesh, rae2822_stations, CMeshSpec};
use ofgpu::field::{GpuScalarField, GpuVectorField};
use ofgpu::field_setup::{setup_scalar_field, setup_vector_field, NutRoughness, WallFaces};
use ofgpu::fv::{DivScheme, Limiter};
use ofgpu::io::case::{
    LinearSolverKind, Preconditioner, SolverControls, TurbulenceControls, WallFunctionCoeffs,
};
use ofgpu::io::fields::{PatchFieldSpec, RawScalarField, RawVectorField};
use ofgpu::io::polymesh::build_host_mesh;
use ofgpu::models::{SpalartAllmaras, SaCoeffs};
use ofgpu::timescheme::DdtScheme;
use ofgpu::walldistance;
use ofgpu::vv::{self, GridStudy, Level};
use ofgpu::{Error, Gpu, GpuMesh, Result, Scalar, Vec3};

use super::nozzle::{mach_field, max_change};
use crate::{key, Checks, GateReport, How, Uncertainty, Verdict};

/// The gate's scope and registry name.
pub(crate) const GATE: &str = "SPEC-LIT 127.6 Gate CMP-RAE RAE 2822";

const AGAINST: &str = "RAE 2822 Case 6 (the corrected conditions M 0.729, alpha 2.31 degrees, \
                       Re 6.5e6) of Cook, McDonald & Firmin, AGARD AR-138 (1979), as the NPARC \
                       archive reproduces it - the answer key rae2822-cp-case6";

/// Case 6's corrected conditions (decision TAKEN, SPEC-LIT 127.6.8):
/// NPARC's data at M 0.729, alpha 2.31 degrees, Re 6.5e6 is AR-138's
/// **Case 6**, not Case 9 - the supervisor confirmed the conditions
/// against the NPARC page before the key was written.
pub(crate) const M_INF: f64 = 0.729;
pub(crate) const ALPHA_DEG: f64 = 2.31;
pub(crate) const RE_C: f64 = 6.5e6;
pub(crate) const CHORD: f64 = 1.0;
/// NPARC's 460 R in kelvin.
pub(crate) const T_INF: f64 = 460.0 / 1.8;
/// Sutherland's constants for air (§122.1), restated here so the free
/// stream is host `f64` arithmetic in BOTH builds.
const A_S: f64 = 1.458e-6;
const T_S: f64 = 110.4;
/// `R_UNIVERSAL` and the molar mass of air, §122.1/§122.2.
const R_UNIVERSAL: f64 = 8.314462618;
const W_AIR: f64 = 0.0289647;
/// The fully turbulent free stream, TMR's low end (§127.1).
const NU_TILDA_RATIO: f64 = 3.0;
/// Air's `gamma` (§122.1), restated for the host `f64` free stream.
pub(crate) const GAMMA: f64 = 1.4;

/// The march: chunks of [`CHUNK`] iterations until no cell's `M_i` moves
/// by more than [`FIELD_STOP`] across a chunk, capped per level by
/// [`MAX_ITERS`].
/// LTS CFL 0.1 and the span [`SPAN`] are INPUTS (decision TAKEN by the
/// supervisor, SPEC-LIT 127.6.8, measured on L0): §125.7's local step is
/// `CFL V^(1/3)/(|u| + c)`, so on a one-cell-thick mesh the span sets the
/// step of the `y+ = 1` wall cells. At span 1 the first outer iteration
/// stops on a `T <= 0` bound at every CFL tried (1, 0.1), in a wall cell
/// near the trailing edge; at span 1e-5 it holds at CFL 0.1 and 0.3 and
/// not at 1 or 3.
pub(crate) const LTS_CFL: f64 = 0.1;
/// The C-mesh's span [m] (the `CMeshSpec::span` input of every level and
/// of the smoke), see [`LTS_CFL`].
pub(crate) const SPAN: f64 = 1.0e-5;
pub(crate) const N_OUTER: usize = 5;
pub(crate) const CHUNK: usize = 200;
pub(crate) const FIELD_STOP: f64 = 1.0e-5;
pub(crate) const MAX_ITERS: [usize; 3] = [20000, 20000, 20000];
/// The live smoke's budget: 2000 iterations on the coarse mesh, inside
/// the plan's 10 minutes (docs/17 5.3 rule 4).
pub(crate) const SMOKE_ITERS: usize = 2000;

/// The rows' bars, the plan's, fixed before measurement (class P).
pub(crate) const SHOCK_BAR: f64 = 0.04;
pub(crate) const RMS_BAR: f64 = 0.05;
pub(crate) const CL_BAR: f64 = 0.06;
/// The smoke's residual bar: the last chunk's `p` residual below a tenth
/// of the first chunk's.
pub(crate) const SMOKE_RESID_RATIO: f64 = 0.1;

fn w(x: Scalar) -> f64 {
    f64::from(x)
}

fn s(x: f64) -> Scalar {
    x as Scalar
}

/// The free stream, all in host `f64` whatever `Scalar` is - computed
/// from the constants above, never from `gas.mu()` (a `Scalar`).
#[allow(dead_code)] // RECORDED fields: a run fills the whole record; the row reads what it reads
pub(crate) struct FreeStream {
    pub r_s: f64,
    pub mu: f64,
    pub u: f64,
    pub rho: f64,
    pub p: f64,
    pub nu: f64,
    pub nu_tilda: f64,
    pub q: f64,
}

/// Case 6's free stream (127.30)'s inputs, in the order the relations
/// chain: `R_s = R/W`; Sutherland's `mu_inf = a_s T^1.5/(T + t_s)`;
/// `U_inf = M sqrt(gamma R_s T)`; the chord Reynolds number fixes the
/// density, `rho_inf = Re mu_inf/(U_inf c)`; the datum `p_inf = rho R_s
/// T` is BOTH the absolute static pressure and the gauge datum
/// (`PerfectGas::air(p_inf)`), so the solver's `p` field is `Cp q_inf`
/// directly; and `nu_tilda_inf = 3 nu_inf`.
pub(crate) fn free_stream() -> FreeStream {
    let r_s = R_UNIVERSAL / W_AIR;
    let mu = A_S * T_INF.powf(1.5) / (T_INF + T_S);
    let u = M_INF * (GAMMA * r_s * T_INF).sqrt();
    let rho = RE_C * mu / (u * CHORD);
    let p = rho * r_s * T_INF;
    let nu = mu / rho;
    FreeStream {
        r_s,
        mu,
        u,
        rho,
        p,
        nu,
        nu_tilda: NU_TILDA_RATIO * nu,
        q: 0.5 * rho * u * u,
    }
}

/// (127.31), derived: the sonic pressure coefficient - the `Cp` of the
/// isentropic `M = 1` point: NACA 1135's eq. (44), the isentropic
/// `p/p_t`, taken at `M = 1` and at `M`, with `Cp = (p - p_inf)/q_inf`
/// and `q_inf = gamma p_inf M^2/2`:
///
/// `Cp* = (2/(gamma M^2)) [ ((2 + (gamma-1) M^2)/(gamma+1))^(gamma/(gamma-1)) - 1 ]`
pub(crate) fn cp_star(gamma: f64, m: f64) -> f64 {
    (2.0 / (gamma * m * m))
        * (((2.0 + (gamma - 1.0) * m * m) / (gamma + 1.0)).powf(gamma / (gamma - 1.0)) - 1.0)
}

/// (127.33)'s shock locator, the same function for the experiment and the
/// simulation: on points sorted by `x`, the first pair with `x_k >
/// x_min` and `Cp_k < cp* <= Cp_(k+1)`, crossed linearly -
/// `x_s = x_k + (x_(k+1) - x_k)(cp* - Cp_k)/(Cp_(k+1) - Cp_k)`. None
/// when no pair crosses (a shock-free surface, or `Cp` already below
/// `cp*` at `x_min`).
pub(crate) fn sonic_crossing(pts: &[(f64, f64)], cp_star: f64, x_min: f64) -> Option<f64> {
    for k in 1..pts.len() {
        let (x0, c0) = pts[k - 1];
        let (x1, c1) = pts[k];
        if x0 > x_min && c0 < cp_star && cp_star <= c1 {
            return Some(x0 + (x1 - x0) * (cp_star - c0) / (c1 - c0));
        }
    }
    None
}

/// (127.33)'s lower-surface RMS: over the KEY's points whose `x` sits
/// inside the simulated range `[sim first x, sim last x]`, the RMS of
/// `Cp_sim(x) - Cp_exp` with `Cp_sim` linear in `x`. Returns the RMS and
/// the count; NaN and 0 when no point is inside.
pub(crate) fn lower_rms(sim: &[(f64, f64)], exp: &[(f64, f64)]) -> (f64, usize) {
    let (x_lo, x_hi) = match (sim.first(), sim.last()) {
        (Some(a), Some(b)) => (a.0, b.0),
        _ => return (f64::NAN, 0),
    };
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for &(x, c) in exp {
        if !(x_lo..=x_hi).contains(&x) {
            continue;
        }
        // Cp_sim linear in x between the bracketing simulated points.
        let cs = match sim.windows(2).find(|pr| pr[0].0 <= x && x <= pr[1].0) {
            Some(pr) => {
                let t = (x - pr[0].0) / (pr[1].0 - pr[0].0);
                pr[0].1 + t * (pr[1].1 - pr[0].1)
            }
            None => continue,
        };
        let d = cs - c;
        sum += d * d;
        n += 1;
    }
    if n == 0 {
        return (f64::NAN, 0);
    }
    ((sum / n as f64).sqrt(), n)
}

/// The trapezoid integral of `y` over `x` along sorted points.
pub(crate) fn trapz(pts: &[(f64, f64)]) -> f64 {
    pts.windows(2)
        .map(|pr| (pr[1].0 - pr[0].0) * (pr[0].1 + pr[1].1) / 2.0)
        .sum()
}

/// (127.32), derived: the pressure lift, `Cl = F . (-sin alpha, cos
/// alpha, 0)/(q c span)` with `F = sum_airfoil p_gauge,f Sf_f` - `Sf`
/// points out of the fluid, INTO the aerofoil, so `sum p Sf` is the
/// force ON the aerofoil. (The dropped `-Ca sin(alpha)` term is the
/// experiment's own integral's, [`exp_cl`]'s, not this function's: this
/// function takes both components of `F`.)
pub(crate) fn pressure_lift(
    p_gauge: &[f64],
    sf: &[[f64; 3]],
    alpha_deg: f64,
    q: f64,
    chord: f64,
    span: f64,
) -> f64 {
    let (sa, ca) = (alpha_deg.to_radians().sin(), alpha_deg.to_radians().cos());
    let mut f = [0.0f64; 3];
    for (p, s3) in p_gauge.iter().zip(sf) {
        for d in 0..3 {
            f[d] += p * s3[d];
        }
    }
    (f[0] * -sa + f[1] * ca) / (q * chord * span)
}

/// The experiment's `Cp` key, `Cp = -minus_cp`, split by surface and
/// sorted by `x`: `upper` the archive's surface 1 (TE to LE as written,
/// ascending here), `lower` its surface -1, `le` the shared leading-edge
/// point (surface 0).
pub(crate) struct CpKey {
    pub upper: Vec<(f64, f64)>,
    pub lower: Vec<(f64, f64)>,
    pub le: (f64, f64),
}

/// `rae2822-cp-case6` through the answer-key reader, columns `x_c`,
/// `minus_cp`, `surface`.
pub(crate) fn cp_key() -> Result<CpKey> {
    let kf = key::load("rae2822-cp-case6")?;
    let x = kf.column("x_c")?;
    let mcp = kf.column("minus_cp")?;
    let surf = kf.column("surface")?;
    let (mut upper, mut lower) = (Vec::new(), Vec::new());
    let mut le = (f64::NAN, f64::NAN);
    for k in 0..x.len() {
        let pt = (x[k], -mcp[k]);
        match surf[k] as i64 {
            1 => upper.push(pt),
            -1 => lower.push(pt),
            _ => le = pt,
        }
    }
    upper.sort_by(|a, b| a.0.total_cmp(&b.0));
    lower.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(CpKey { upper, lower, le })
}

impl CpKey {
    /// The upper surface with the leading-edge point prepended, as the
    /// shock locator and the `Cn` integral read it.
    pub fn upper_le(&self) -> Vec<(f64, f64)> {
        let mut v = vec![self.le];
        v.extend(self.upper.iter().copied());
        v
    }

    /// The lower surface with the leading-edge point prepended.
    pub fn lower_le(&self) -> Vec<(f64, f64)> {
        let mut v = vec![self.le];
        v.extend(self.lower.iter().copied());
        v
    }
}

/// The experiment's own shock position (127.33) on the key's upper
/// surface with its leading-edge point.
pub(crate) fn exp_shock(k: &CpKey) -> Option<f64> {
    sonic_crossing(&k.upper_le(), cp_star(GAMMA, M_INF), 0.1)
}

/// The experiment's integrated lift, decision TAKEN (SPEC-LIT 127.6.8):
/// the key's own `Cp` integrated, `Cl_exp = Cn cos(alpha)` with
/// `Cn = trapz(lower Cp) - trapz(upper Cp)` over `x`, the leading-edge
/// point in both surfaces. The dropped `-Ca sin(alpha)` term is 8.3e-4
/// (0.11 %), measured once.
pub(crate) fn exp_cl(k: &CpKey) -> f64 {
    (trapz(&k.lower_le()) - trapz(&k.upper_le())) * ALPHA_DEG.to_radians().cos()
}

/// The three gate meshes (SPEC-LIT 127.6.8, decision TAKEN: §94's
/// `vv::grid_study` refuses two levels, so the gate runs three C-meshes).
/// L1 is CMP-26's default gate mesh; L0 and L2 halve and double the wake
/// columns and the `eta` lines with `refine`. Level 0, 1, 2 = 8448,
/// 33 792, 135 168 cells; the smoke runs `CMeshSpec::coarse()` (5 632).
/// The smoke's mesh: `CMeshSpec::coarse()` at the gate's [`SPAN`].
pub(crate) fn smoke_spec() -> CMeshSpec {
    CMeshSpec { span: SPAN, ..CMeshSpec::coarse() }
}

#[allow(dead_code)] // the recorder tests' entry (the supervisor runs them)
pub(crate) fn level_spec(l: usize) -> CMeshSpec {
    match l {
        0 => CMeshSpec { refine: 1, n_wake: 24, n_normal: 48, span: SPAN, ..CMeshSpec::default() },
        2 => CMeshSpec { refine: 4, n_wake: 96, n_normal: 192, span: SPAN, ..CMeshSpec::default() },
        _ => CMeshSpec { span: SPAN, ..CMeshSpec::default() },
    }
}

/// The flow controls, oblique.rs's except decision TAKEN's two inputs
/// (SPEC-LIT 127.6.8, measured on L0): PBiCGStab + DILU (`U` and `T` at
/// `1e-9` relTol 0.01, `p` at `1e-10` relTol 0.001, maxIter 1000),
/// `LocalEuler` on momentum and energy, LTS CFL `LTS_CFL`, `u_relax
/// 0.7`, `t_relax 0.7`, `p_relax 0.3`, five outer iterations,
/// `residualControl` off, `vanLeer` on momentum, energy and the
/// retarded density. The inputs sit in one place.
pub(crate) fn controls(gas: PerfectGas) -> (CompressibleControls, TurbulenceControls) {
    let mut ctrl = CompressibleControls { gas, ..CompressibleControls::default() };
    let lin = |tolerance: Scalar, rel_tol: Scalar| SolverControls {
        solver: LinearSolverKind::PBiCGStab,
        precon: Preconditioner::Dilu,
        tolerance,
        rel_tol,
        max_iter: 1000,
        ..Default::default()
    };
    ctrl.momentum.u_solver = lin(1.0e-9, 0.01);
    ctrl.pressure.p_solver = lin(1.0e-10, 0.001);
    ctrl.energy.t_solver = lin(1.0e-9, 0.01);
    ctrl.lts.cfl = s(LTS_CFL);
    ctrl.momentum.ddt = DdtScheme::LocalEuler;
    ctrl.energy.ddt = DdtScheme::LocalEuler;
    ctrl.momentum.u_relax = 0.7;
    ctrl.energy.t_relax = 0.7;
    ctrl.p_relax = 0.3;
    ctrl.n_outer = N_OUTER;
    ctrl.residual_control = None;
    ctrl.momentum.div_scheme = DivScheme::Limited(Limiter::VanLeer);
    ctrl.energy.div_scheme = DivScheme::Limited(Limiter::VanLeer);
    ctrl.pressure.retarded = Retarded::Limited(Limiter::VanLeer);

    // The SA inputs, §127.1.2's steady shape: a steady run takes
    // `steadyState` and the nuTilda relaxation, upwind on
    // `div(phi,nuTilda)` with NO `bounded` prefix (§127.1's refusal).
    let mut turb = TurbulenceControls::default();
    turb.steady = true;
    turb.ddt = DdtScheme::SteadyState;
    turb.eps_div_scheme = DivScheme::Upwind;
    turb.eps_bounded_convection = false;
    turb.eps_relax = 0.7;
    turb.epsilon_solver = lin(1.0e-9, 0.01);
    (ctrl, turb)
}

fn entry(t: &str) -> PatchFieldSpec {
    PatchFieldSpec { type_name: t.into(), ..PatchFieldSpec::default() }
}

/// A `characteristicFarfield` entry stating its free-stream scalar.
fn ff_s(v: f64) -> PatchFieldSpec {
    PatchFieldSpec {
        type_name: "characteristicFarfield".into(),
        value: vec![s(v)],
        ..PatchFieldSpec::default()
    }
}

/// A `characteristicFarfield` entry stating its free-stream vector.
fn ff_v(v: Vec3) -> PatchFieldSpec {
    PatchFieldSpec {
        type_name: "characteristicFarfield".into(),
        value_v: vec![v],
        ..PatchFieldSpec::default()
    }
}

/// The four patches on one field: `farfield` AND `outlet` both
/// `characteristicFarfield` (§126.10), `airfoil` the wall's entry,
/// `frontAndBack` `empty`.
fn flow_bcs(ff: PatchFieldSpec, wall: PatchFieldSpec) -> BTreeMap<String, PatchFieldSpec> {
    let mut b = BTreeMap::new();
    b.insert("farfield".into(), ff.clone());
    b.insert("outlet".into(), ff);
    b.insert("airfoil".into(), wall);
    b.insert("frontAndBack".into(), entry("empty"));
    b
}

/// One airfoil face's measure: its centre's `x`, its outward `Sf` (out of
/// the fluid, INTO the aerofoil), and its owner cell, whose `p` the
/// zeroGradient wall reads.
struct WallFace {
    x: f64,
    sf: [f64; 3],
    owner: usize,
}

/// One run's field and bookkeeping, after its march stopped.
#[allow(dead_code)] // RECORDED fields: a run fills the whole record; the row reads what it reads
pub(crate) struct RaeRun {
    pub n_cells: usize,
    /// (94.1)'s cell size, `sqrt(sum V / (span n_cells))` [m].
    pub h: f64,
    pub iterations: usize,
    /// The field stop was met.
    pub converged: bool,
    /// The last chunk's max cell-M change.
    pub last_change: f64,
    /// The first chunk's and the last chunk's last outer `p` residual.
    pub first_p_res: f64,
    pub last_p_res: f64,
    /// Wall-clock, solve and mesh.
    pub seconds: f64,
    pub mesh_seconds: f64,
    /// (127.32)'s pressure lift on the airfoil faces.
    pub cl: f64,
    /// (127.33)'s sonic crossing on the upper surface; None when the
    /// surface never crosses `Cp*`.
    pub shock_x: Option<f64>,
    /// (127.33)'s lower-surface RMS against the key, and the count.
    pub rms_lower: f64,
    pub n_rms: usize,
    /// The largest cell Mach number.
    pub m_max: f64,
    /// The airfoil faces' `Cp` (owner cell's `p` over `q_inf`), `x` the
    /// face centre's, upper and lower sorted by `x`.
    pub upper: Vec<(f64, f64)>,
    pub lower: Vec<(f64, f64)>,
    /// Every downloaded flow field (`U`, `p`, `T`) finite.
    pub all_finite: bool,
    /// The solver's last printed line.
    pub last_line: String,
}

/// The airfoil patch's faces, the wall bookkeeping one run reads.
fn wall_faces(hm: &ofgpu::HostMesh) -> Result<(Vec<WallFace>, usize)> {
    let p = hm
        .patches
        .iter()
        .find(|p| p.name == "airfoil")
        .ok_or_else(|| Error::Mesh("gatemesh: no airfoil patch".into()))?;
    let faces: Vec<WallFace> = (p.start..p.start + p.size)
        .map(|fi| WallFace {
            x: w(hm.b_cf[fi].x),
            sf: [w(hm.b_sf[fi].x), w(hm.b_sf[fi].y), w(hm.b_sf[fi].z)],
            owner: hm.b_face_cells[fi] as usize,
        })
        .collect();
    Ok((faces, p.size))
}

/// One march of one level (SPEC-LIT 127.6.8's setup): the C-mesh, the
/// uniform free-stream start, SA attached as `ofgpu-compressible`
/// attaches it, the chunked march to the field stop. A solver bound
/// (`Error::Refused`, §122.3) is the caller's, as `Err`, carrying the
/// iterations reached.
pub(crate) fn run_level(
    gpu: &Gpu,
    spec: &CMeshSpec,
    max_iters: usize,
    progress: Option<&str>,
) -> Result<RaeRun> {
    let t_all = Instant::now();
    let t_mesh = Instant::now();
    let fs = free_stream();
    let gas = PerfectGas::air(fs.p);
    let cm = airfoil_cmesh(&rae2822_stations()?, spec)?;
    let hm = build_host_mesh(&cm.raw)?;
    let gm = GpuMesh::upload(gpu, &hm)?;
    let mesh_seconds = t_mesh.elapsed().as_secs_f64();
    let n = hm.n_cells;
    let (walls, n_s) = wall_faces(&hm)?;
    let (ctrl, tctrl) = controls(gas);
    let alpha = ALPHA_DEG.to_radians();
    let u_inf = Vec3::new(
        s(fs.u * alpha.cos()),
        s(fs.u * alpha.sin()),
        s(0.0),
    );

    // The uniform free stream in every cell: (127.30)'s `U`, gauge
    // `p = 0` (the datum IS `p_inf`, so the field reads `Cp` directly
    // through `q_inf`), `T_inf`.
    let mut ru = RawVectorField { name: "U".into(), ..RawVectorField::default() };
    ru.internal = vec![u_inf; n];
    ru.boundary = flow_bcs(ff_v(u_inf), PatchFieldSpec {
        type_name: "fixedValue".into(),
        value_v: vec![Vec3::ZERO],
        ..PatchFieldSpec::default()
    });
    let mut rp = RawScalarField { name: "p".into(), ..RawScalarField::default() };
    rp.internal = vec![s(0.0); n];
    rp.boundary = flow_bcs(ff_s(0.0), entry("zeroGradient"));
    let mut rt = RawScalarField { name: "T".into(), ..RawScalarField::default() };
    rt.internal = vec![s(T_INF); n];
    rt.boundary = flow_bcs(ff_s(T_INF), entry("zeroGradient"));

    let low = cbc::lower(&ru, &rp, &rt, &hm, &ctrl.gas)?;
    assert_eq!(low.farfields.len(), 2, "farfield and outlet are far-field patches");
    let mut uf = GpuVectorField::zeros(gpu, &gm, "U")?;
    let mut pf = GpuScalarField::zeros(gpu, &gm, "p")?;
    let mut tf = GpuScalarField::zeros(gpu, &gm, "T")?;
    setup_vector_field(gpu, &mut uf, &low.u, &hm)?;
    setup_scalar_field(gpu, &mut pf, &low.p, &hm)?;
    setup_scalar_field(gpu, &mut tf, &low.t, &hm)?;
    let mut st = CompressibleState::new(gpu, &gm, uf, pf, tf)?;
    let mut solver = Compressible::new(gpu, &gm, ctrl)?;
    solver.set_boundary_rules(gpu, low.rules)?;
    solver.set_outlet_rules(gpu, low.outlets)?;
    // Without the far-field rules the far-field faces stay fixed
    // placeholders (§126.10).
    solver.set_farfield_rules(gpu, low.farfields)?;

    // SA (§127.1, as compressible.rs 1120-1148 attaches it): nuTilda
    // fixed 0 on the wall (the model's own wall treatment), the free
    // stream at the far field, zeroGradient out; nut carries the wall
    // FUNCTION types.
    let mut raw_nt = RawScalarField { name: "nuTilda".into(), ..RawScalarField::default() };
    raw_nt.internal = vec![s(fs.nu_tilda); n];
    raw_nt.boundary = flow_bcs(
        PatchFieldSpec {
            type_name: "fixedValue".into(),
            value: vec![s(fs.nu_tilda)],
            ..PatchFieldSpec::default()
        },
        PatchFieldSpec {
            type_name: "fixedValue".into(),
            value: vec![s(0.0)],
            ..PatchFieldSpec::default()
        },
    );
    // `outlet` zeroGradient, not the far field: flow_bcs' far-field entry
    // is replaced below.
    raw_nt.boundary.insert("outlet".into(), entry("zeroGradient"));
    let mut raw_nut = RawScalarField { name: "nut".into(), ..RawScalarField::default() };
    raw_nut.internal = vec![s(0.0); n];
    raw_nut.boundary = flow_bcs(entry("calculated"), PatchFieldSpec {
        type_name: "nutLowReWallFunction".into(),
        value: vec![s(0.0)],
        ..PatchFieldSpec::default()
    });
    let wd = walldistance::wall_distance(gpu, &hm, &gm, &ctrl.pressure.p_solver, 0)?;
    let wf = WallFaces::from_case(&raw_nt, Some(&raw_nut), &hm)?;
    let rough = NutRoughness::from_case(Some(&raw_nut), &hm)?;
    let mut model = SpalartAllmaras::new(
        gpu,
        &hm,
        &gm,
        SaCoeffs::default(),
        tctrl,
        WallFunctionCoeffs::default(),
        &wf,
        &rough,
        &wd.y.f,
    )?;
    setup_scalar_field(gpu, model.nu_tilda_mut(), &raw_nt, &hm)?;
    setup_scalar_field(gpu, model.nut_mut(), &raw_nut, &hm)?;
    solver.set_turbulence(gpu, model)?;
    solver.initialise(gpu, &mut st)?;

    // The stop is the field's own: chunks of CHUNK iterations until no
    // cell's `M_i` moves by more than FIELD_STOP across a chunk.
    let machs = |st: &CompressibleState| -> Result<Vec<f64>> {
        Ok(mach_field(&gpu.download(&st.u.f)?, &gpu.download(&st.t.f)?, GAMMA, fs.r_s))
    };
    let mut prev = machs(&st)?;
    let mut iterations = 0usize;
    let mut change = f64::INFINITY;
    let (mut first_p_res, mut last_p_res) = (f64::NAN, f64::NAN);
    let mut last_line = "no iteration".to_string();
    while iterations < max_iters && !(change <= FIELD_STOP) {
        let k = CHUNK.min(max_iters - iterations);
        // A bound (T, p_abs or rho <= 0, §122.3) stops `run` with
        // Error::Refused: the gate's miss, carrying the iterations.
        let rep = match solver.run(gpu, &mut st, k) {
            Ok(rep) => rep,
            Err(Error::Refused(why)) => {
                return Err(Error::Refused(format!("after {iterations} iterations: {why}")));
            }
            Err(e) => return Err(e),
        };
        iterations += rep.steps;
        if let Some(o) = rep.last.as_ref().and_then(|l| l.outers.last()) {
            last_line = o.line();
            if let Some(pp) = o.p.first() {
                last_p_res = w(pp.initial_residual);
                if first_p_res.is_nan() {
                    first_p_res = last_p_res;
                }
            }
        }
        let next = machs(&st)?;
        let p_g: Vec<f64> = gpu.download(&st.p.f)?.iter().map(|v| w(*v)).collect();
        let pg: Vec<f64> = walls.iter().map(|f| p_g[f.owner]).collect();
        let sfs: Vec<[f64; 3]> = walls.iter().map(|f| f.sf).collect();
        let cl = pressure_lift(&pg, &sfs, ALPHA_DEG, fs.q, CHORD, spec.span);
        change = max_change(&prev, &next);
        prev = next;
        if let Some(tag) = progress {
            let pct = 100.0 * iterations as f64 / max_iters as f64;
            eprintln!(
                "{tag}: iter {iterations}/{max_iters} ({pct:.1} %) dM {change:.3e} \
                 p-res {last_p_res:.3e} Cl {cl:.5} {} s",
                t_all.elapsed().as_secs_f64()
            );
        }
    }

    // The measured field: the wall `Cp`s (airfoil face m is LOWER iff
    // m < n_s/2 - the patch runs lower TE->LE first, then upper LE->TE,
    // SPEC-LIT 127.5.1), the lift, the shock, the RMS.
    let u: Vec<Vec3> = gpu.download(&st.u.f)?;
    let t: Vec<Scalar> = gpu.download(&st.t.f)?;
    let p_g: Vec<f64> = gpu.download(&st.p.f)?.iter().map(|v| w(*v)).collect();
    let all_finite = u.iter().all(|v| w(v.x).is_finite() && w(v.y).is_finite() && w(v.z).is_finite())
        && t.iter().all(|v| w(*v).is_finite())
        && p_g.iter().all(|v| v.is_finite());
    let (mut upper, mut lower) = (Vec::new(), Vec::new());
    for (m, f) in walls.iter().enumerate() {
        let pt = (f.x, p_g[f.owner] / fs.q);
        if m < n_s / 2 {
            lower.push(pt);
        } else {
            upper.push(pt);
        }
    }
    upper.sort_by(|a, b| a.0.total_cmp(&b.0));
    lower.sort_by(|a, b| a.0.total_cmp(&b.0));
    let pg: Vec<f64> = walls.iter().map(|f| p_g[f.owner]).collect();
    let sfs: Vec<[f64; 3]> = walls.iter().map(|f| f.sf).collect();
    let cl = pressure_lift(&pg, &sfs, ALPHA_DEG, fs.q, CHORD, spec.span);
    let shock_x = sonic_crossing(&upper, cp_star(GAMMA, M_INF), 0.1);
    let kexp = cp_key()?;
    let (rms_lower, n_rms) = lower_rms(&lower, &kexp.lower);
    let mach = mach_field(&u, &t, GAMMA, fs.r_s);
    let m_max = mach
        .iter()
        .copied()
        .fold(0.0f64, |m, v| if v.is_nan() || m.is_nan() { f64::NAN } else { m.max(v) });
    let volume: f64 = hm.v.iter().map(|v| w(*v)).sum();
    let h = (volume / (spec.span * n as f64)).sqrt();
    Ok(RaeRun {
        n_cells: n,
        h,
        iterations,
        converged: change <= FIELD_STOP,
        last_change: change,
        first_p_res,
        last_p_res,
        seconds: t_all.elapsed().as_secs_f64(),
        mesh_seconds,
        cl,
        shock_x,
        rms_lower,
        n_rms,
        m_max,
        upper,
        lower,
        all_finite,
        last_line,
    })
}

/// One recorded run of SPEC-LIT 127.6.8's recipe: the mesh's own numbers,
/// the solve's, and the three measures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RaeRecord {
    /// `"f64"` or `"single"`, and the commit the binary was built at.
    pub build: &'static str,
    pub commit: &'static str,
    pub level: usize,
    pub n_cells: usize,
    pub h: f64,
    pub iterations: usize,
    pub converged: bool,
    pub last_change: f64,
    pub seconds: f64,
    pub cl: f64,
    /// NaN when the surface never crosses `Cp*`.
    pub shock_x: f64,
    pub rms_lower: f64,
    pub n_rms: usize,
    pub m_max: f64,
}

/// A [`RaeRecord`] literal from one [`RaeRun`], for the recorder tests to
/// print and the consts below to hold - build `"f64"` or `"single"` by
/// the feature, the commit this gate landed at.
#[allow(dead_code)] // the recorder tests' entry (the supervisor runs them)
pub(crate) fn record_literal(r: &RaeRun, level: usize) -> String {
    let build = if cfg!(feature = "single") { "single" } else { "f64" };
    format!(
        "RaeRecord {{ build: {build:?}, commit: \"HEAD+CMP-27\", level: {level}, \
         n_cells: {}, h: {:?}, iterations: {}, converged: {}, last_change: {:?}, \
         seconds: {:?}, cl: {:?}, shock_x: {:?}, rms_lower: {:?}, n_rms: {}, m_max: {:?} }}",
        r.n_cells,
        r.h,
        r.iterations,
        r.converged,
        r.last_change,
        r.seconds,
        r.cl,
        r.shock_x.unwrap_or(f64::NAN),
        r.rms_lower,
        r.n_rms,
        r.m_max
    )
}

/// The f64 build's recorded runs, one slot per level (index = level),
/// filled by CMP-27's recording step. Every run was attempted and none
/// reached a field: each level's refusal text is in [`F64_STOPPED`]; a
/// level never attempted would carry its reason in [`F64_NOT_RUN`].
pub(crate) const F64_LEVELS: [Option<&RaeRecord>; 3] = [None, None, None];
/// The three f64 recorded runs' refusal texts, EXACTLY as
/// `record_rae_l0/l1/l2` printed them (SPEC-LIT 127.6.8's **Measured**
/// paragraph): every recorded run stopped on a bound (§122.3) before any
/// field existed.
pub(crate) const F64_STOPPED: [Option<&str>; 3] = [
    Some("after 5800 iterations: compressible::Compressible::step: 2 cell(s) out of bounds at the end of outer iteration 4 of step 5968 (T <= 0: 1, p_abs <= 0: 0, rho <= 0: 1); the first is T at cell 6312, T = -5.441498878025827"),
    Some("after 5400 iterations: compressible::Compressible::step: 2 cell(s) out of bounds at the end of outer iteration 5 of step 5454 (T <= 0: 1, p_abs <= 0: 0, rho <= 0: 1); the first is T at cell 21072, T = -5.194727211491283"),
    Some("after 5200 iterations: compressible::Compressible::step: 2 cell(s) out of bounds at the end of outer iteration 5 of step 5376 (T <= 0: 1, p_abs <= 0: 0, rho <= 0: 1); the first is T at cell 75232, T = -4.433695285638386"),
];
/// A level that was never attempted would carry its reason here; every
/// f64 run was attempted, so all three are `None`.
pub(crate) const F64_NOT_RUN: [Option<&str>; 3] = [None; 3];
/// The single build's recorded run on L1 (the plan's gate mesh).
pub(crate) const SINGLE_L1: Option<&RaeRecord> = None;
/// The single L1 recorded run's refusal text, EXACTLY as
/// `record_rae_l1` printed it in the single build (SPEC-LIT 127.6.8's
/// **Measured** paragraph).
pub(crate) const SINGLE_STOPPED: Option<&str> = Some("after 200 iterations: compressible::Compressible::step: 4 cell(s) out of bounds at the end of outer iteration 4 of step 210 (T <= 0: 2, p_abs <= 0: 2, rho <= 0: 0); the first is T at cell 151, T = -47663.30078125");
/// The single build attempted its run, so this is `None`.
pub(crate) const SINGLE_NOT_RUN: Option<&str> = None;

/// The finest recorded level in `levels`, from the top down.
fn finest_level<'a>(levels: &'a [Option<&'a RaeRecord>; 3]) -> Option<(usize, &'a RaeRecord)> {
    (0..3).rev().find_map(|l| levels[l].map(|r| (l, r)))
}

/// The LIVE smoke (SPEC-LIT 127.6.8, docs/17 5.3 rule 4): 2000
/// iterations on `smoke_spec()` (`coarse()` at [`SPAN`]), no bound, a converging residual,
/// a shock present. Returns `Some(why)` when a row read against the bar.
fn check_rae2822_smoke(c: &mut Checks, gpu: &Gpu) -> Result<Option<String>> {
    const BOUND: &str = "Gate CMP-RAE smoke: the coarse C-mesh run stops on no bound and every \
                         field is finite (SPEC-LIT 127.6.8)";
    const RESID: &str = "Gate CMP-RAE smoke: the residual converges, the last chunk's p residual \
                         below a tenth of the first chunk's (SPEC-LIT 127.6.8)";
    const SHOCK: &str = "Gate CMP-RAE smoke: a shock is present, the upper-surface Cp crosses \
                         Cp* past x/c 0.1 (SPEC-LIT 127.6.8)";
    let r = match run_level(gpu, &smoke_spec(), SMOKE_ITERS, None) {
        Ok(r) => r,
        Err(Error::Refused(why)) => {
            c.note(&format!("Gate CMP-RAE smoke: the march at LTS CFL {LTS_CFL} stopped on a \
                             bound, so there is no field to compare: {why}"));
            for label in [BOUND, RESID, SHOCK] {
                c.require(label, false);
            }
            return Ok(Some(format!("the smoke stopped on a bound: {why}")));
        }
        Err(e) => return Err(e),
    };
    let bound_ok = r.all_finite;
    let resid_ok = r.last_p_res <= SMOKE_RESID_RATIO * r.first_p_res;
    let shock_ok = r.shock_x.is_some();
    c.require(BOUND, bound_ok);
    c.require(RESID, resid_ok);
    c.require(SHOCK, shock_ok);
    c.note(&format!(
        "Gate CMP-RAE smoke: {} cells, {} iterations, {:.1} s (mesh {:.1} s); the first chunk's \
         p residual {:.3e}, the last chunk's {:.3e}; the last cell-M change {:.3e}, field stop {}",
        r.n_cells,
        r.iterations,
        r.seconds,
        r.mesh_seconds,
        r.first_p_res,
        r.last_p_res,
        r.last_change,
        if r.converged { "met" } else { "NOT met" }
    ));
    c.note(&format!(
        "Gate CMP-RAE smoke: M_max {:.4}, Cl {:.6} against the experiment's integrated \
         {:.6}, shock x/c {}, lower-surface RMS {:.5} over {} points; last: {}",
        r.m_max,
        r.cl,
        exp_cl(&cp_key()?),
        r.shock_x.map_or("none".to_string(), |v| format!("{v:.4}")),
        r.rms_lower,
        r.n_rms,
        r.last_line
    ));
    let why = [
        (bound_ok, "the smoke's field is not finite"),
        (resid_ok, "the smoke's residual did not fall to a tenth of its first chunk's"),
        (shock_ok, "the smoke's upper surface never crosses Cp* past x/c 0.1"),
    ]
    .into_iter()
    .filter(|(ok, _)| !ok)
    .map(|(_, why)| why)
    .collect::<Vec<_>>()
    .join("; ");
    Ok((!why.is_empty()).then_some(why))
}

/// The labels of one build's three replay rows.
fn row_labels(build: &str) -> [String; 3] {
    [
        format!("Gate CMP-RAE {build}: upper-surface shock x/c (127.33) against the \
                 experiment's (bar 0.04)"),
        format!("Gate CMP-RAE {build}: lower-surface Cp RMS against the experiment (bar 0.05)"),
        format!("Gate CMP-RAE {build}: Cl against the experiment's integrated Cl (bar 6 %)"),
    ]
}

/// One recorded run's three errors against the experiment: the shock's
/// `|x_s - x_s,exp|` (a NaN shock fails), the lower-surface RMS, and
/// `|cl/cl_exp - 1|`.
fn row_errors(r: &RaeRecord, exp_shock: f64, exp_cl: f64) -> [f64; 3] {
    [(r.shock_x - exp_shock).abs(), r.rms_lower, (r.cl / exp_cl - 1.0).abs()]
}

/// One recorded run's three rows, each inside `c.replaying` (pipe_tet.rs's
/// shape); `out` collects every reason a row read against the bar.
fn replay_one(
    c: &mut Checks,
    build: &str,
    r: &RaeRecord,
    exp_shock: f64,
    exp_cl: f64,
    bars: [f64; 3],
    out: &mut Vec<String>,
) {
    let labels = row_labels(build);
    let errs = row_errors(r, exp_shock, exp_cl);
    for (k, label) in labels.iter().enumerate() {
        let (err, bar) = (errs[k], bars[k]);
        // `!(err <= bar)`, so that a NaN measure counts as the miss it is.
        if !(err <= bar) {
            c.note(&format!(
                "Gate CMP-RAE: {label}: {err:.4e} against the bar {bar} (level {})",
                r.level
            ));
            out.push(format!("{label}: {err:.4e} against the bar {bar}"));
        }
        c.replaying(|c| c.check_p(label, s(err), s(bar)));
    }
}

/// The REPLAY of the recorded runs (SPEC-LIT 127.6.8): every recorded
/// level's numbers in the notes, every run that stopped on a bound its
/// refusal text and one failed row, the three rows on the FINEST recorded
/// f64 level and on the single build's L1, the (94.9) studies when all
/// three f64 levels are recorded, and a skip per row with neither a
/// record nor a stop. A build with stop texts but no record reads its
/// three measure rows at NaN - they fail, there is no field to compare.
/// Returns whether any row failed.
pub(crate) fn check_rae2822_replay(
    c: &mut Checks,
    f64_levels: &[Option<&RaeRecord>; 3],
    f64_not_run: &[Option<&str>; 3],
    f64_stopped: &[Option<&str>; 3],
    single: Option<&RaeRecord>,
    single_not_run: Option<&str>,
    single_stopped: Option<&str>,
) -> bool {
    let exp = match cp_key() {
        Ok(k) => k,
        Err(why) => {
            c.note(&format!("Gate CMP-RAE: the answer key did not read: {why}"));
            for build in ["f64", "single"] {
                for label in row_labels(build) {
                    c.skip(&label, "the answer key is absent");
                }
            }
            return false;
        }
    };
    let exp_shock = exp_shock(&exp).unwrap_or(f64::NAN);
    let exp_cl = exp_cl(&exp);
    c.note(&format!(
        "Gate CMP-RAE: the experiment (rae2822-cp-case6): {} upper, {} lower points plus the \
         leading edge at x {:.4}, Cp {:.4}; Cp* (127.31) {:.6}, the experiment's shock x/c \
         {:.6}; Cn {:.9}, Cl_exp = Cn cos(alpha) {:.9} (the dropped -Ca sin(alpha) term 8.3e-4)",
        exp.upper.len(),
        exp.lower.len(),
        exp.le.0,
        exp.le.1,
        cp_star(GAMMA, M_INF),
        exp_shock,
        trapz(&exp.lower_le()) - trapz(&exp.upper_le()),
        exp_cl
    ));
    let mut failed = Vec::new();
    for (l, rec) in f64_levels.iter().enumerate() {
        match rec {
            Some(r) => c.note(&format!(
                "Gate CMP-RAE f64 level {l} recorded ({} at {}): {} cells, h {:.6e}, {} \
                 iterations, field stop {}, last change {:.3e}, {:.1} s; Cl {:.6}, shock x/c \
                 {}, RMS {:.5} over {} points, M_max {:.4}",
                r.build, r.commit, r.n_cells, r.h, r.iterations,
                if r.converged { "met" } else { "NOT met" },
                r.last_change, r.seconds, r.cl,
                if r.shock_x.is_nan() { "none".to_string() } else { format!("{:.4}", r.shock_x) },
                r.rms_lower, r.n_rms, r.m_max
            )),
            None => match f64_stopped[l] {
                Some(why) => {
                    c.note(&format!(
                        "Gate CMP-RAE f64 level {l}: the recorded run stopped on a bound, so \
                         there is no field to compare: {why}"
                    ));
                    c.replaying(|c| {
                        c.require(
                            &format!("Gate CMP-RAE f64 L{l}: the recorded run reaches its field \
                                      stop or its cap without a bound (SPEC-LIT 127.6.8)"),
                            false,
                        );
                    });
                    failed.push(format!("f64 L{l} stopped on a bound"));
                }
                None => c.note(&format!(
                    "Gate CMP-RAE f64 level {l} NOT recorded - {}",
                    f64_not_run[l].unwrap_or("no reason given")
                )),
            },
        }
    }
    let bars = [SHOCK_BAR, RMS_BAR, CL_BAR];
    match finest_level(f64_levels) {
        Some((l, r)) => {
            c.note(&format!(
                "Gate CMP-RAE: the f64 rows read the FINEST recorded level, L{l}"
            ));
            replay_one(c, "f64", r, exp_shock, exp_cl, bars, &mut failed);
        }
        None => match f64_stopped.iter().flatten().next() {
            Some(_) => {
                c.note(
                    "Gate CMP-RAE f64: no recorded run reached a field, so the three rows read \
                     NaN against their bars",
                );
                for (k, label) in row_labels("f64").into_iter().enumerate() {
                    c.replaying(|c| c.check_p(&label, s(f64::NAN), s(bars[k])));
                    failed.push(label);
                }
            }
            None => {
                let why = format!(
                    "DEFERRED (docs/17 5.3 rule 4): NOT RUN - {}",
                    f64_not_run[2].unwrap_or("no reason given")
                );
                for label in row_labels("f64") {
                    c.skip(&label, &why);
                }
            }
        },
    }
    match single {
        Some(r) => replay_one(c, "single", r, exp_shock, exp_cl, bars, &mut failed),
        None => match single_stopped {
            Some(why) => {
                c.note(&format!(
                    "Gate CMP-RAE single L1: the recorded run stopped on a bound, so there is \
                     no field to compare: {why}"
                ));
                c.replaying(|c| {
                    c.require(
                        "Gate CMP-RAE single L1: the recorded run reaches its field stop or \
                         its cap without a bound (SPEC-LIT 127.6.8)",
                        false,
                    );
                });
                failed.push("single L1 stopped on a bound".to_string());
                c.note(
                    "Gate CMP-RAE single: no recorded run reached a field, so the three rows \
                     read NaN against their bars",
                );
                for (k, label) in row_labels("single").into_iter().enumerate() {
                    c.replaying(|c| c.check_p(&label, s(f64::NAN), s(bars[k])));
                    failed.push(label);
                }
            }
            None => {
                let why = format!(
                    "DEFERRED (docs/17 5.3 rule 4): NOT RUN - {}",
                    single_not_run.unwrap_or("no reason given")
                );
                for label in row_labels("single") {
                    c.skip(&label, &why);
                }
            }
        },
    }
    // The (94.9) studies, on all three f64 levels finest first; on fewer,
    // a note saying how many are recorded.
    let n_recorded = f64_levels.iter().filter(|r| r.is_some()).count();
    let named = [("Cl", (|r: &RaeRecord| r.cl) as fn(&RaeRecord) -> f64),
        ("shock x/c", (|r: &RaeRecord| r.shock_x) as fn(&RaeRecord) -> f64),
        ("lower-surface RMS", (|r: &RaeRecord| r.rms_lower) as fn(&RaeRecord) -> f64)];
    if n_recorded == 3 {
        for (name, f) in named {
            match study_of(f64_levels, f) {
                Some(Ok(st)) => {
                    c.note(&format!("Gate CMP-RAE: 94 study of {name}: {}", st.one_line()))
                }
                Some(Err(e)) => {
                    c.note(&format!("Gate CMP-RAE: 94 study of {name} refused: {e}"))
                }
                None => c.note(&format!("Gate CMP-RAE: 94 study of {name}: fewer than three levels")),
            }
        }
    } else {
        c.note(&format!(
            "Gate CMP-RAE: {n_recorded} of the three f64 levels are recorded, so no 94 study \
             runs (three are the minimum, SPEC-LIT 94.1)"
        ));
    }
    !failed.is_empty()
}

/// One recorded run's bits that read against their bars, for the
/// verdict's headline; empty when the record sits inside all three.
fn against_bar(r: &RaeRecord, exp_shock: f64, exp_cl: f64) -> String {
    let mut bits = Vec::new();
    let e = (r.shock_x - exp_shock).abs();
    if !(e <= SHOCK_BAR) {
        bits.push(format!(
            "shock x/c {:.4} against the experiment's {exp_shock:.4} (bar 0.04)",
            r.shock_x
        ));
    }
    if !(r.rms_lower <= RMS_BAR) {
        bits.push(format!("lower-surface RMS {:.4} (bar 0.05)", r.rms_lower));
    }
    let e = r.cl / exp_cl - 1.0;
    if !(e.abs() <= CL_BAR) {
        bits.push(format!(
            "Cl {:+.2} % against the experiment's integrated (bar 6 %)",
            100.0 * e
        ));
    }
    bits.join(", ")
}

/// The step and cell a refusal text names, for the headline: the first
/// "step N" and "cell N" in the text.
fn stop_headline(why: &str) -> String {
    let num = |tag: &str| -> Option<u64> {
        let i = why.find(tag)? + tag.len();
        why[i..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse::<u64>()
            .ok()
    };
    match (num("step "), num("cell ")) {
        (Some(st), Some(cell)) => {
            format!("stopped on a bound at step {st} (the first out-of-bounds cell {cell})")
        }
        _ => "stopped on a bound".to_string(),
    }
}

/// The replay half of the verdict's headline: every stop's level, step
/// and cell, then the records' bits that read against their bars (the
/// rows' own reasons are in the notes above it).
fn replay_headline(
    f64_levels: &[Option<&RaeRecord>; 3],
    f64_stopped: &[Option<&str>; 3],
    single: Option<&RaeRecord>,
    single_stopped: Option<&str>,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    for (l, why) in f64_stopped.iter().enumerate() {
        if let Some(why) = *why {
            parts.push(format!("f64 L{l} {}", stop_headline(why)));
        }
    }
    if let Some(why) = single_stopped {
        parts.push(format!("single L1 {}", stop_headline(why)));
    }
    match cp_key() {
        Err(why) => {
            let bit = format!("the answer key did not read: {why}");
            if parts.is_empty() {
                return bit;
            }
            parts.push(bit);
        }
        Ok(k) => {
            let (xs, cl) = (exp_shock(&k).unwrap_or(f64::NAN), exp_cl(&k));
            if let Some((l, r)) = finest_level(f64_levels) {
                let bits = against_bar(r, xs, cl);
                if !bits.is_empty() {
                    parts.push(format!("f64 L{l}: {bits}"));
                }
            }
            if let Some(r) = single {
                let bits = against_bar(r, xs, cl);
                if !bits.is_empty() {
                    parts.push(format!("single L{}: {bits}", r.level));
                }
            }
        }
    }
    parts.join("; ")
}


/// **Gate CMP-RAE** (SPEC-LIT §127.6, §127.6.8): the live coarse smoke,
/// then the replayed records; the one verdict through the registry.
pub(crate) fn check_rae2822(c: &mut Checks, gpu: &Gpu) -> Result<()> {
    c.note(&format!(
        "Gate CMP-RAE: RAE 2822 Case 6 (the corrected conditions M {M_INF}, alpha {ALPHA_DEG} \
         degrees, Re_c {RE_C:.3e}, T_inf {T_INF:.4} K) with Spalart-Allmaras on \
         SPEC-LIT 127.5.1's C-mesh at three levels, L0 8448, L1 33 792, L2 135 168 cells \
         (the smoke `CMeshSpec::coarse()`, {SMOKE_ITERS} iterations); nuTilda inf \
         3 nu_inf, SA steady with `steadyState` and upwind `div(phi,nuTilda)`; LTS CFL \
         {LTS_CFL}, {N_OUTER} outer iterations, chunks of {CHUNK} to a cell-M change of \
         {FIELD_STOP:.1e}, caps {MAX_ITERS:?}; the rows are the replayed records' (SPEC-LIT \
         127.6.8)"
    ));
    let smoke_why = check_rae2822_smoke(c, gpu)?;
    let replay_failed = check_rae2822_replay(
        c,
        &F64_LEVELS,
        &F64_NOT_RUN,
        &F64_STOPPED,
        SINGLE_L1,
        SINGLE_NOT_RUN,
        SINGLE_STOPPED,
    );
    let cl_study = study_of(&F64_LEVELS, (|r: &RaeRecord| r.cl) as fn(&RaeRecord) -> f64)
        .and_then(|res| res.ok());
    let uncertainty = match cl_study {
        Some(st) => Uncertainty::Study(st),
        None => Uncertainty::SingleMesh(
            "no recorded C-mesh reached a field: every run stopped on a bound (SPEC-LIT 127.6.8)",
        ),
    };
    if smoke_why.is_some() || replay_failed {
        let mut parts: Vec<String> = Vec::new();
        if let Some(why) = &smoke_why {
            parts.push(why.clone());
        }
        if replay_failed {
            let h = replay_headline(&F64_LEVELS, &F64_STOPPED, SINGLE_L1, SINGLE_STOPPED);
            if !h.is_empty() {
                parts.push(h);
            }
        }
        let how = if replay_failed { How::Replayed } else { How::Live };
        c.report(GateReport {
            verdict: Verdict::Misses,
            how,
            gate: GATE,
            against: AGAINST,
            headline: parts.join("; "),
            detail: vec![],
            uncertainty: Some(uncertainty),
        });
    } else if finest_level(&F64_LEVELS).is_none() {
        report_open(c);
    }
    Ok(())
}

/// The Open verdict with no f64 level recorded and nothing failed
/// (pipe_tet.rs's shape).
fn report_open(c: &mut Checks) {
    c.report(GateReport {
        verdict: Verdict::Open,
        how: How::Replayed,
        gate: GATE,
        against: AGAINST,
        headline: "the three C-meshes are NOT RUN (DEFERRED, docs/17 5.3 rule 4); the live \
                   coarse smoke ran"
            .to_string(),
        detail: vec![],
        uncertainty: Some(Uncertainty::SingleMesh(
            "fewer than three recorded C-meshes (SPEC-LIT 127.6.8)",
        )),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close_abs(got: f64, want: f64, tol: f64, what: &str) {
        assert!((got - want).abs() <= tol, "{what}: {got} is not {want}");
    }

    fn close_rel(got: f64, want: f64, tol: f64, what: &str) {
        assert!((got - want).abs() <= tol * want.abs(), "{what}: {got} is not {want}");
    }

    /// The free stream at Case 6's corrected conditions, to the digit.
    #[test]
    fn the_free_stream() {
        let fs = free_stream();
        close_rel(fs.r_s, 287.0550227690948, 1.0e-12, "r_s");
        close_rel(fs.mu, 1.627635558889652e-05, 1.0e-12, "mu");
        close_rel(fs.u, 233.62363914459755, 1.0e-12, "u");
        close_rel(fs.rho, 0.45284934228058354, 1.0e-12, "rho");
        close_rel(fs.p, 33220.351110715754, 1.0e-12, "p");
        close_rel(fs.nu, 3.594209832993809e-05, 1.0e-12, "nu");
        close_rel(fs.nu_tilda, 1.0782629498981425e-04, 1.0e-12, "nu_tilda");
        close_rel(fs.q, 12358.259630240922, 1.0e-12, "q");
    }

    /// (127.31), the sonic pressure coefficient at Case 6's Mach number.
    #[test]
    fn the_sonic_cp_star() {
        close_abs(cp_star(1.4, 0.729), -0.6657815128683363, 1.0e-14, "cp_star");
    }

    /// The shock locator: the first crossing past x_min, interpolated;
    /// None when the surface never crosses.
    #[test]
    fn the_shock_locator() {
        let pts = [(0.05, -2.0), (0.08, 0.5), (0.1, -1.0), (0.5, -1.2), (0.6, -0.5), (0.8, -0.2)];
        let xs = sonic_crossing(&pts, -0.6657815128683363, 0.1).expect("the crossing");
        close_abs(xs, 0.5763169267330948, 1.0e-14, "x_s");
        assert_eq!(
            sonic_crossing(&[(0.1, -0.2), (0.5, -0.3), (0.9, -0.4)], -0.6657815128683363, 0.1),
            None,
            "every Cp above Cp* is a shock-free surface"
        );
    }

    /// The lower-surface RMS: the key's points inside the simulated
    /// range, `Cp_sim` linear between the bracketing points.
    #[test]
    fn the_lower_rms() {
        let sim = [(0.0, 0.0), (1.0, 1.0)];
        let exp = [(0.25, 0.3), (0.5, 0.5), (1.5, 9.0)];
        let (rms, n) = lower_rms(&sim, &exp);
        close_abs(rms, 0.03535533905932738, 1.0e-15, "the RMS");
        assert_eq!(n, 2, "the point past the trailing edge is skipped");
        assert!(lower_rms(&[], &exp).0.is_nan(), "no simulated points, NaN");
    }

    /// (127.32)'s pressure lift, its hand value at alpha 2.31, and the
    /// trapezoid it integrates.
    #[test]
    fn the_pressure_lift() {
        let p = [-2.0];
        let sf = [[0.0, -0.5, 0.0]];
        close_abs(pressure_lift(&p, &sf, 0.0, 1.0, 1.0, 1.0), 1.0, 0.0, "lift at alpha 0");
        close_abs(pressure_lift(&p, &sf, 90.0, 1.0, 1.0, 1.0), 0.0, 1.0e-15, "lift at alpha 90");
        let p = [-1.0];
        let sf = [[0.3, -0.4, 0.0]];
        close_abs(
            pressure_lift(&p, &sf, 2.31, 1.0, 1.0, 1.0),
            0.4117668055030109,
            1.0e-15,
            "lift at alpha 2.31",
        );
        close_abs(trapz(&[(0.0, 1.0), (0.5, 2.0), (1.0, 4.0)]), 2.25, 1.0e-14, "trapz");
    }

    /// The experiment's own numbers, read from the key on disk (the other
    /// key tests' shape): the surfaces' counts, the leading-edge point,
    /// the shock, the integrated `Cn` and `Cl`.
    #[test]
    fn the_experiment() {
        let k = cp_key().expect("load the answer key rae2822-cp-case6");
        assert_eq!(k.upper.len(), 53, "the upper surface's points");
        assert_eq!(k.lower.len(), 49, "the lower surface's points");
        close_abs(k.le.0, 0.0, 0.0, "the leading edge's x");
        close_abs(k.le.1, 0.8328, 1.0e-15, "the leading edge's Cp");
        let xs = exp_shock(&k).expect("the experiment's shock");
        close_abs(xs, 0.5778626382934252, 1.0e-12, "the experiment's x_s");
        let cn = trapz(&k.lower_le()) - trapz(&k.upper_le());
        close_abs(cn, 0.743602165, 1.0e-12, "Cn");
        close_abs(exp_cl(&k), 0.7429978957195293, 1.0e-12, "Cl_exp");
    }

    /// The three gate meshes and the smoke mesh, by their controls and
    /// their cell counts, no mesh built.
    #[test]
    fn the_level_specs() {
        let cells = |sp: &CMeshSpec| (2 * sp.n_wake + 128 * sp.refine) * sp.n_normal;
        let specs: [CMeshSpec; 3] = [level_spec(0), level_spec(1), level_spec(2)];
        assert_eq!((specs[0].refine, specs[0].n_wake, specs[0].n_normal), (1, 24, 48));
        assert_eq!((specs[1].refine, specs[1].n_wake, specs[1].n_normal), (2, 48, 96));
        assert_eq!((specs[2].refine, specs[2].n_wake, specs[2].n_normal), (4, 96, 192));
        let counts: Vec<usize> = specs.iter().map(cells).collect();
        assert_eq!(counts, vec![8448, 33792, 135168]);
        let co = CMeshSpec::coarse();
        assert_eq!((co.refine, co.n_wake, co.n_normal), (1, 24, 32));
        assert_eq!(cells(&co), 5632, "the smoke mesh");
    }

    /// The shipped consts carry no record: every row skips, nothing
    /// fails, and the gate's verdict is the Open NOT RUN one - with no
    /// unregistered verdict word in any note (§69.2).
    #[test]
    fn replay_with_no_records_is_open_not_failed() {
        let no_rec: [Option<&RaeRecord>; 3] = [None; 3];
        let no_why: [Option<&str>; 3] = [None; 3];
        let mut c = Checks::new();
        let failed = check_rae2822_replay(&mut c, &no_rec, &no_why, &no_why, None, None, None);
        assert!(!failed, "no record, no row failed");
        assert_eq!((c.total, c.failures, c.skipped), (0, 0, 6), "six skips, no rows");
        c.enter_gate(GATE);
        report_open(&mut c);
        c.leave_gate();
        assert_eq!(c.gates.len(), 1, "one gate in the registry");
        assert_eq!(c.gates[0].gate, GATE);
        assert_eq!(c.gates[0].verdict, Verdict::Open);
        let before = c.failures;
        c.audit_and_summarise();
        assert_eq!(c.failures, before, "an unregistered verdict in the notes (§69.2)");
    }

    /// A record outside two of the three bars: exactly two rows fail and
    /// the replay says so; the row inside its bar does not.
    #[test]
    fn replay_flags_a_record_outside_the_bars() {
        let rec = RaeRecord {
            build: "f64",
            commit: "test",
            level: 1,
            n_cells: 33792,
            h: 5.9e-3,
            iterations: 20000,
            converged: true,
            last_change: 8.0e-6,
            seconds: 900.0,
            cl: 0.60,
            shock_x: 0.70,
            rms_lower: 0.02,
            n_rms: 49,
            m_max: 0.75,
        };
        let levels: [Option<&RaeRecord>; 3] = [None, Some(&rec), None];
        let not_run: [Option<&str>; 3] = [Some("not recorded yet"); 3];
        let mut c = Checks::new();
        let no_stops: [Option<&str>; 3] = [None; 3];
        let failed = check_rae2822_replay(
            &mut c,
            &levels,
            &not_run,
            &no_stops,
            None,
            Some("not recorded yet"),
            None,
        );
        assert!(failed, "two rows read against their bars");
        assert_eq!(c.failures, 2, "the shock's 0.12 and Cl's -19 %; the RMS's 0.02 is inside");
        assert_eq!(c.skipped, 3, "the single build's three rows skip");
    }

    /// The shipped consts: every recorded run stopped on a bound, so each
    /// of the four stops is one failed row, and with no record each
    /// build's three measure rows read NaN and fail - 10 failures, no
    /// skip, and the replay says a row read against its bar.
    #[test]
    fn the_recorded_stops_read_as_misses() {
        let mut c = Checks::new();
        let failed = check_rae2822_replay(
            &mut c,
            &F64_LEVELS,
            &F64_NOT_RUN,
            &F64_STOPPED,
            SINGLE_L1,
            SINGLE_NOT_RUN,
            SINGLE_STOPPED,
        );
        assert!(failed, "every stop is a failed row");
        assert_eq!(
            (c.failures, c.skipped),
            (10, 0),
            "4 stop rows + 3 f64 NaN rows + 3 single NaN rows"
        );
        assert_eq!(c.total, 10, "every failed row counted toward the total");
    }

    // ---- the recorded runs and their price (the supervisor runs these) ----

    /// One level's price: 200 iterations on level `l`, one PRICE line.
    fn price(l: usize) {
        let gpu = Gpu::new(0).expect("a GPU");
        let r = run_level(&gpu, &level_spec(l), 200, Some("PRICE rae")).expect("run");
        println!(
            "PRICE level {l} cells {} mesh_s {:.1} run_s {:.1} per_iter_s {:.4e}",
            r.n_cells,
            r.mesh_seconds,
            r.seconds,
            r.seconds / r.iterations.max(1) as f64
        );
    }

    #[test]
    #[ignore = "a recorded run of SPEC-LIT 127.6.8 (GPU minutes)"]
    fn price_rae_l0() {
        price(0);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 127.6.8 (GPU minutes)"]
    fn price_rae_l1() {
        price(1);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 127.6.8 (GPU minutes)"]
    fn price_rae_l2() {
        price(2);
    }

    /// One recorded run of one level, printed as the record literal its
    /// slot above holds.
    fn record(l: usize) {
        let gpu = Gpu::new(0).expect("a GPU");
        let r = run_level(&gpu, &level_spec(l), MAX_ITERS[l], Some("RECORD rae")).expect("run");
        println!("RECORD {}", record_literal(&r, l));
    }

    #[test]
    #[ignore = "a recorded run of SPEC-LIT 127.6.8 (GPU minutes)"]
    fn record_rae_l0() {
        record(0);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 127.6.8 (GPU minutes)"]
    fn record_rae_l1() {
        record(1);
    }
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 127.6.8 (GPU minutes)"]
    fn record_rae_l2() {
        record(2);
    }

    /// The live smoke alone, on a fresh Checks, printing its rows.
    #[test]
    #[ignore = "a recorded run of SPEC-LIT 127.6.8 (GPU minutes)"]
    fn smoke_rae() {
        let gpu = Gpu::new(0).expect("a GPU");
        let mut c = Checks::new();
        let why = check_rae2822_smoke(&mut c, &gpu).expect("the smoke");
        println!(
            "rows {} failures {} against the bar: {why:?}",
            c.total, c.failures
        );
    }
}










/// The (94.9) study of one measure across the three recorded f64 levels,
/// finest first; None with fewer than three.
fn study_of(
    levels: &[Option<&RaeRecord>; 3],
    value: fn(&RaeRecord) -> f64,
) -> Option<std::result::Result<GridStudy, String>> {
    let lv: Vec<Level> = levels
        .iter()
        .rev()
        .filter_map(|r| r.as_ref())
        .map(|r| Level { h: s(r.h), value: s(value(r)) })
        .collect();
    if lv.len() < 3 {
        return None;
    }
    Some(vv::grid_study(&lv).map_err(|e| e.to_string()))
}
