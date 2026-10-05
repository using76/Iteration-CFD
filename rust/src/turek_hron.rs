// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Turek-Hron CFD1-CFD3 on the card, and Gate 105-C - SPEC-LIT 105.13 and
//! 105.14.
//!
//! Written from:
//!   S. Turek and J. Hron, LNCSE 53 (2006) 371-385,
//!     DOI 10.1007/3-540-34596-5_15 - the geometry, the fluid, the inflow
//!     profile and its ramp, the quantities and their reduction;
//!   ofgpu SPEC-LIT.md 32.5.6 - the two wall-force integrators this module
//!     calls and does not repeat;
//!   ofgpu SPEC-LIT.md 94.1 - the grid study;
//!   ofgpu SPEC-LIT.md 105.7-105.10 - the moving mesh.
//!
//! No GPL-licensed source was consulted.
//!
//! NOT here: no driver runs these cases and no JSONC case key carries them
//! (the JSONC case holds a blockgen box only, so the lib module reads the
//! polyMesh itself), and no kernel is launched from this file - every step
//! is `Simple`'s, orchestrated from the host exactly as a driver would.

use crate::ale_flow::{buoyancy_off, laminar_fields};
use crate::device::Gpu;
use crate::error::{Error, Result};
use crate::field::BcKind;
use crate::field_ops::{self, FieldKernels};
use crate::io::case::{DivScheme, LinearSolverKind, Preconditioner, SolverControls};
use crate::io::polymesh::{build_host_mesh, read_poly_mesh, PolyMeshRaw};
use crate::mesh::ale::AleMesh;
use crate::mesh::gpugeom::{flatten_faces, FacePointCsr};
use crate::mesh::{GpuMesh, HostMesh, PatchKind};
use crate::momentum::MomentumControls;
use crate::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use crate::simple::{Simple, SimpleControls};
use crate::timescheme::DdtScheme;
use crate::vv::{self, GridStudy, Level};
use crate::wallfunctions::{pressure_force, wall_shear};
use crate::{Label, Scalar, Vec3};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The channel's length, `x` in `[0, L]` (Turek and Hron 2006).
pub const CHANNEL_L: Scalar = 2.5;
/// The channel's height, `y` in `[0, H]`.
pub const CHANNEL_H: Scalar = 0.41;
/// The fluid's density, `kg/m^3` - the integrators' `rho_bf` and nothing
/// else; the momentum equation itself carries no density.
pub const RHO: Scalar = 1000.0;
/// The fluid's kinematic viscosity, `m^2/s` - Reynolds 20, 100 and 200 at
/// the three tests' mean inflows on the 0.1 m diameter.
pub const NU: Scalar = 1e-3;
/// The eight patches of the fluid region, in no particular order.
pub const PATCHES: [&str; 8] = [
    "empty_back",
    "empty_front",
    "wall_top",
    "cylinder",
    "wall_bottom",
    "inlet",
    "outlet",
    "fluid_to_flap",
];
/// The body the forces are taken on: the paper's S1 (cylinder) and S2
/// (flap) together.
pub const BODY: [&str; 2] = ["cylinder", "fluid_to_flap"];
/// `cases/turekHron/<dir>/fluid/polyMesh` for levels 1, 2, 3 and 4.
pub const LEVEL_DIRS: [&str; 4] = ["mesh", "mesh_L2", "mesh_L3", "mesh_L4"];
/// The steady runs' iteration budgets along the levels, L1 to L4.
pub const STEADY_MAX_ITERS: [usize; 4] = [4000, 8000, 16000, 40000];
/// How many of the finest levels the grid study takes - L4, L3 and L2.
/// L1 is run and printed but is not in the study: its CFD2 lift lies far
/// outside the asymptotic range (SPEC-LIT 105.13).
pub const STUDY_LEVELS: usize = 3;
/// How often the stopping rule's forces are folded between iterations.
pub const STEADY_CHECK_EVERY: usize = 100;
/// No stopping decision before this many iterations.
pub const STEADY_MIN_ITERS: usize = 300;
/// The stopping rule: drag and lift both move less than this fraction of
/// the drag between consecutive checks.
pub const STEADY_FORCE_TOL: Scalar = 1e-7;
/// The gate on the EXTRAPOLATED steady value, not on any one level's.
pub const STEADY_TOL: Scalar = 0.02;
/// CFD3's mesh level - the gate's comparison is same-mesh, static against
/// wobbling.
pub const CFD3_LEVEL: usize = 1;
/// CFD3's time step, s.
pub const CFD3_DT: Scalar = 1e-3;
/// CFD3's static spin-up, in steps: 8,000 at `dt = 1e-3` is `t = 8` s,
/// past the ramp's two seconds and into the periodic state.
pub const CFD3_SPIN_STEPS: usize = 8000;
/// Each continuation's window, in steps: 1.5 s, about six and a half lift
/// periods at the published frequency.
pub const CFD3_WINDOW_STEPS: usize = 1500;
/// How many lift periods the reduction averages over.
pub const CFD3_PERIODS: usize = 4;
/// The gate on the wobbling run against the static run, per statistic.
pub const CFD3_TOL: Scalar = 0.005;
/// The inflow ramp's duration, s: `(1 - cos(pi t / RAMP_T)) / 2` for
/// `t < RAMP_T`, else 1.
pub const RAMP_T: Scalar = 2.0;
/// The wobble box's left edge, m: every point left of it stays put.
pub const WOBBLE_X0: Scalar = 0.7;
/// The wobble box's right edge, m.
pub const WOBBLE_X1: Scalar = 2.3;
/// The wobble's displacement amplitude, m.
pub const WOBBLE_AMPLITUDE: Scalar = 0.004;
/// The wobble's period, s.
pub const WOBBLE_PERIOD: Scalar = 0.3;

/// Which of the paper's three CFD tests a run carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Case {
    Cfd1,
    Cfd2,
    Cfd3,
}

impl Case {
    /// The parabolic profile's mean, `m/s`: 0.2, 1, 2 (Reynolds 20, 100,
    /// 200 on the diameter).
    pub fn u_bar(self) -> Scalar {
        match self {
            Case::Cfd1 => 0.2,
            Case::Cfd2 => 1.0,
            Case::Cfd3 => 2.0,
        }
    }

    /// The paper's own name for the test.
    pub fn name(self) -> &'static str {
        match self {
            Case::Cfd1 => "CFD1",
            Case::Cfd2 => "CFD2",
            Case::Cfd3 => "CFD3",
        }
    }
}

/// The paper's parabolic inflow, `1.5 u_bar y (H - y) / (H/2)^2`, in `m/s`
/// along `x`; zero outside `0 <= y <= H`.
pub fn inlet_velocity(y: Scalar, u_bar: Scalar) -> Scalar {
    if !(0.0..=CHANNEL_H).contains(&y) {
        return 0.0;
    }
    1.5 * u_bar * y * (CHANNEL_H - y) / ((CHANNEL_H / 2.0) * (CHANNEL_H / 2.0))
}

/// The paper's start ramp: `(1 - cos(pi t / RAMP_T)) / 2` for
/// `t < RAMP_T`, else 1.
pub fn ramp(t: Scalar) -> Scalar {
    if t < RAMP_T {
        ((1.0 - (std::f64::consts::PI * f64::from(t) / f64::from(RAMP_T)).cos()) / 2.0) as Scalar
    } else {
        1.0
    }
}

/// One level's fluid mesh, read and made ready.
#[derive(Debug, Clone)]
pub struct Rig {
    pub level: usize,
    pub raw: PolyMeshRaw,
    pub hm: HostMesh,
    pub csr: FacePointCsr,
    /// The one-cell depth, `max z - min z` over the points.
    pub dz: Scalar,
    /// Indices into `hm.patches` of `BODY`, in `BODY`'s order.
    pub body: Vec<usize>,
    /// `vv::h_of(sum(hm.v) / dz, hm.n_cells, 2)` - §94.1's 2-D typical size.
    pub h: Scalar,
}

/// `<CARGO_MANIFEST_DIR>/../cases/turekHron/<LEVEL_DIRS[level-1]>/fluid/polyMesh`.
///
/// `level` outside 1..=4 panics: it is a programming error, not an input.
pub fn mesh_dir(level: usize) -> PathBuf {
    assert!(
        (1..=LEVEL_DIRS.len()).contains(&level),
        "turek_hron: level {level} is not 1, 2, 3 or 4"
    );
    // The crate's directory as compiled in, never the runtime environment:
    // `ofgpu-validate` run as a plain executable has no CARGO_MANIFEST_DIR.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("cases")
        .join("turekHron")
        .join(LEVEL_DIRS[level - 1])
        .join("fluid")
        .join("polyMesh")
}

/// `load_rig_from(&mesh_dir(level), level)`.
pub fn load_rig(level: usize) -> Result<Rig> {
    load_rig_from(&mesh_dir(level), level)
}

/// Read `dir`, build the host mesh, check the patch set, make the body a
/// wall - the rig of SPEC-LIT 105.13.
pub fn load_rig_from(dir: &Path, level: usize) -> Result<Rig> {
    if !dir.join("points").is_file() {
        return Err(Error::Config(format!(
            "Turek-Hron level {level}: no mesh at {}; generate it from the \
             repository root with `python tools/mesh/examples/turek_hron.py --level {level}`",
            dir.display()
        )));
    }
    let raw = read_poly_mesh(dir)?;
    let mut hm = build_host_mesh(&raw)?;
    let mut found: Vec<&str> = hm.patches.iter().map(|p| p.name.as_str()).collect();
    found.sort_unstable();
    let mut wanted: Vec<&str> = PATCHES.to_vec();
    wanted.sort_unstable();
    if found != wanted || found.windows(2).any(|w| w[0] == w[1]) {
        let names: Vec<String> = hm.patches.iter().map(|p| p.name.clone()).collect();
        return Err(Error::Config(format!(
            "Turek-Hron level {level}: the fluid region names the patches \
             [{}], and this benchmark needs exactly [{}]",
            names.join(", "),
            PATCHES.join(", ")
        )));
    }
    for name in BODY {
        let p = hm
            .patches
            .iter_mut()
            .find(|p| p.name == name)
            .expect("the patch set was checked above");
        p.kind = PatchKind::Wall;
        p.type_name = "wall".to_string();
    }
    hm.compute_geometry(&raw.points, &raw.faces)?;

    let z_min = raw.points.iter().map(|p| p.z).fold(Scalar::INFINITY, Scalar::min);
    let z_max = raw.points.iter().map(|p| p.z).fold(Scalar::NEG_INFINITY, Scalar::max);
    let dz = z_max - z_min;
    if !dz.is_finite() || dz <= 0.0 {
        return Err(Error::Mesh(format!(
            "Turek-Hron level {level}: the mesh's depth is {dz}; one cell of \
             it is what the per-metre forces are divided by"
        )));
    }
    let body = BODY
        .iter()
        .map(|name| hm.patches.iter().position(|p| p.name == *name).expect("checked"))
        .collect();
    let csr = flatten_faces(&raw.faces);
    let total_area = hm.v.iter().sum::<Scalar>() / dz;
    let h = vv::h_of(total_area, hm.n_cells, 2)?;
    Ok(Rig { level, raw, hm, csr, dz, body, h })
}

/// Drag and lift on the body, per unit depth (`N/m`), and their two parts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Forces {
    /// `(pressure + viscous).x`
    pub drag: Scalar,
    /// `(pressure + viscous).y`
    pub lift: Scalar,
    /// The pressure part, per unit depth.
    pub pressure: Vec3,
    /// The viscous part, per unit depth.
    pub viscous: Vec3,
}

/// The force on the body patches, per unit depth, from R4's two integrators
/// - §32.5.6's `pressure_force` and `wall_shear`, called and never repeated.
///
/// `p_bf` is this crate's KINEMATIC pressure; `pressure_force` multiplies by
/// `rho_bf` itself, so nothing here does.
pub fn body_forces(
    hm: &HostMesh,
    body: &[usize],
    dz: Scalar,
    p_bf: &[Scalar],
    u_cells: &[Vec3],
    u_bf: &[Vec3],
) -> Forces {
    let nbf = hm.n_boundary_faces;
    let rho_bf = vec![RHO; nbf];
    let nut_bf = vec![0.0 as Scalar; nbf];
    let kb = vec![false; nbf];
    let pf = pressure_force(hm, p_bf, &rho_bf);
    let ws = wall_shear(
        hm,
        Vec3::new(1.0, 0.0, 0.0),
        u_cells,
        u_bf,
        &rho_bf,
        &nut_bf,
        None,
        &kb,
        NU,
        0.09,
    );
    let mut pressure = Vec3::ZERO;
    let mut viscous = Vec3::ZERO;
    for &pi in body {
        let prow = pf
            .by_patch
            .iter()
            .find(|r| r.patch == pi)
            .unwrap_or_else(|| panic!("body_forces: patch {} has no pressure row; it is not wall-kind", hm.patches[pi].name));
        let wrow = ws
            .by_patch
            .iter()
            .find(|r| r.patch == pi)
            .unwrap_or_else(|| panic!("body_forces: patch {} has no wall-shear row; it is not wall-kind", hm.patches[pi].name));
        pressure += prow.force;
        viscous += wrow.force;
    }
    let pressure = pressure / dz;
    let viscous = viscous / dz;
    Forces {
        drag: pressure.x + viscous.x,
        lift: pressure.y + viscous.y,
        pressure,
        viscous,
    }
}

/// Which condition a patch gets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VelocityBc {
    /// No-slip: `FixedValue`, value zero.
    Wall,
    /// The parabolic profile at the face centre, scaled.
    Inlet { u_bar: Scalar },
    /// A uniform fixed value.
    Uniform(Vec3),
    /// `ZeroGradient`.
    ZeroGradient,
    /// `empty`.
    Empty,
}

/// Which condition the pressure gets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PressureBc {
    /// `ZeroGradient`.
    ZeroGradient,
    /// `FixedValue` at the value.
    Fixed(Scalar),
    /// `empty`.
    Empty,
}

/// Write U's and p's boundary arrays patch by patch, by NAME. Every patch
/// of `hm` must be named exactly once (`Error::Config` otherwise, naming
/// it) - SPEC-LIT 105.13.
pub fn set_bcs(
    gpu: &Gpu,
    s: &mut Simple<'_>,
    hm: &HostMesh,
    rules: &[(&str, VelocityBc, PressureBc)],
) -> Result<()> {
    let nbf = hm.n_boundary_faces;
    let mut u_kind = vec![BcKind::ZeroGradient as Label; nbf];
    let mut u_fr = vec![0.0 as Scalar; nbf];
    let mut u_rv = vec![Vec3::ZERO; nbf];
    let mut p_kind = vec![BcKind::ZeroGradient as Label; nbf];
    let mut p_fr = vec![0.0 as Scalar; nbf];
    let mut p_rv = vec![0.0 as Scalar; nbf];
    let rg = vec![Vec3::ZERO; nbf];
    let prg = vec![0.0 as Scalar; nbf];
    let mut named = vec![false; hm.patches.len()];
    for (name, ub, pb) in rules {
        let Some((pi, p)) = hm
            .patches
            .iter()
            .enumerate()
            .find(|(_, p)| p.name == *name)
        else {
            return Err(Error::Config(format!(
                "turek_hron::set_bcs: the rules name patch {name}, and the mesh has no such patch"
            )));
        };
        if named[pi] {
            return Err(Error::Config(format!(
                "turek_hron::set_bcs: patch {name} is named more than once"
            )));
        }
        named[pi] = true;
        for k in 0..p.size {
            let i = p.start + k;
            match ub {
                VelocityBc::Wall => {
                    u_kind[i] = BcKind::FixedValue as Label;
                    u_fr[i] = 1.0;
                }
                VelocityBc::Inlet { u_bar } => {
                    u_kind[i] = BcKind::FixedValue as Label;
                    u_fr[i] = 1.0;
                    u_rv[i] = Vec3::new(inlet_velocity(hm.b_cf[i].y, *u_bar), 0.0, 0.0);
                }
                VelocityBc::Uniform(v) => {
                    u_kind[i] = BcKind::FixedValue as Label;
                    u_fr[i] = 1.0;
                    u_rv[i] = *v;
                }
                VelocityBc::ZeroGradient => {}
                VelocityBc::Empty => u_kind[i] = BcKind::Empty as Label,
            }
            match pb {
                PressureBc::Fixed(v) => {
                    p_kind[i] = BcKind::FixedValue as Label;
                    p_fr[i] = 1.0;
                    p_rv[i] = *v;
                }
                PressureBc::ZeroGradient => {}
                PressureBc::Empty => p_kind[i] = BcKind::Empty as Label,
            }
        }
    }
    for (p, done) in hm.patches.iter().zip(&named) {
        if !done {
            return Err(Error::Config(format!(
                "turek_hron::set_bcs: patch {} is not named by any rule",
                p.name
            )));
        }
    }
    let u = s.u_mut();
    gpu.write(&mut u.bc_kind, &u_kind)?;
    gpu.write(&mut u.fr, &u_fr)?;
    gpu.write(&mut u.ref_value, &u_rv)?;
    gpu.write(&mut u.ref_grad, &rg)?;
    let p = s.p_mut();
    gpu.write(&mut p.bc_kind, &p_kind)?;
    gpu.write(&mut p.fr, &p_fr)?;
    gpu.write(&mut p.ref_value, &p_rv)?;
    gpu.write(&mut p.ref_grad, &prg)
}

/// U's ref_value array over every boundary face: the inlet faces carry
/// `inlet_velocity(b_cf.y, u_bar) * scale` along `x`, every other face
/// zero. The ramp writes through `scale` - `ramp(0)` is 0, so the spin-up
/// seats a fully closed inlet and each step opens it.
pub fn inlet_ref_values(hm: &HostMesh, inlet: &str, u_bar: Scalar, scale: Scalar) -> Vec<Vec3> {
    let mut rv = vec![Vec3::ZERO; hm.n_boundary_faces];
    let Some(p) = hm.patches.iter().find(|p| p.name == inlet) else {
        return rv;
    };
    for k in 0..p.size {
        let i = p.start + k;
        rv[i] = Vec3::new(inlet_velocity(hm.b_cf[i].y, u_bar) * scale, 0.0, 0.0);
    }
    rv
}

/// The Turek-Hron rule table for `u_bar`: walls and body `Wall` /
/// `ZeroGradient`, inlet `Inlet`/`ZeroGradient`, outlet
/// `ZeroGradient`/`Fixed(0)` (the paper's do-nothing), the empties
/// `Empty`/`Empty`.
pub fn turek_rules(u_bar: Scalar) -> Vec<(&'static str, VelocityBc, PressureBc)> {
    vec![
        ("empty_back", VelocityBc::Empty, PressureBc::Empty),
        ("empty_front", VelocityBc::Empty, PressureBc::Empty),
        ("wall_top", VelocityBc::Wall, PressureBc::ZeroGradient),
        (
            "cylinder",
            VelocityBc::Wall,
            PressureBc::ZeroGradient,
        ),
        ("wall_bottom", VelocityBc::Wall, PressureBc::ZeroGradient),
        (
            "inlet",
            VelocityBc::Inlet { u_bar },
            PressureBc::ZeroGradient,
        ),
        ("outlet", VelocityBc::ZeroGradient, PressureBc::Fixed(0.0)),
        (
            "fluid_to_flap",
            VelocityBc::Wall,
            PressureBc::ZeroGradient,
        ),
    ]
}

/// Steady SIMPLE: `alpha_U` 0.7, `alpha_p` 0.3 (SPEC-LIT 5.2's pairing).
pub fn steady_controls() -> SimpleControls {
    SimpleControls {
        momentum: MomentumControls {
            nu: NU,
            u_solver: SolverControls {
                solver: LinearSolverKind::PBiCGStab,
                tolerance: 1e-10,
                rel_tol: 0.1,
                max_iter: 200,
                precon: Preconditioner::Dilu,
                check_interval: 5,
                ..SolverControls::default()
            },
            u_relax: 0.7,
            div_scheme: DivScheme::LinearUpwind,
            bounded_convection: true,
            sn_grad: crate::fv::SnGradScheme::Corrected,
            variable_viscosity_stress: false,
            ..MomentumControls::default()
        },
        p_solver: SolverControls {
            solver: LinearSolverKind::PCG,
            tolerance: 1e-10,
            rel_tol: 0.05,
            max_iter: 2000,
            precon: Preconditioner::Dic,
            check_interval: 5,
            ..SolverControls::default()
        },
        p_relax: 0.3,
        n_non_orth_correctors: 1,
        n_correctors: 1,
        n_outer_correctors: 1,
        momentum_predictor: true,
        report_continuity: true,
    }
}

/// Transient PISO for CFD3 and for the wobble's uniform-flow test:
/// `backward`, `dt`, one outer corrector of two pressure correctors, no
/// non-orthogonal corrector, no relaxation - the pressure solve is the
/// cost of a step, and this is four of them fewer than two outer
/// correctors with a non-orthogonal corrector each (SPEC-LIT 105.13).
pub fn transient_controls(dt: Scalar) -> SimpleControls {
    SimpleControls {
        momentum: MomentumControls {
            nu: NU,
            u_solver: SolverControls {
                solver: LinearSolverKind::PBiCGStab,
                tolerance: 1e-10,
                rel_tol: 0.0,
                max_iter: 500,
                precon: Preconditioner::Dilu,
                check_interval: 5,
                ..SolverControls::default()
            },
            u_relax: 1.0,
            div_scheme: DivScheme::LinearUpwind,
            bounded_convection: false,
            sn_grad: crate::fv::SnGradScheme::Corrected,
            variable_viscosity_stress: false,
            steady: false,
            ddt: DdtScheme::Backward,
            delta_t: dt,
            ..MomentumControls::default()
        },
        p_solver: SolverControls {
            solver: LinearSolverKind::PCG,
            tolerance: 1e-10,
            rel_tol: 0.0,
            max_iter: 3000,
            precon: Preconditioner::Dic,
            check_interval: 5,
            ..SolverControls::default()
        },
        p_relax: 1.0,
        n_non_orth_correctors: 0,
        n_correctors: 2,
        n_outer_correctors: 1,
        momentum_predictor: true,
        report_continuity: true,
    }
}

/// One steady run.
#[derive(Debug, Clone)]
pub struct SteadyRun {
    pub case: Case,
    pub level: usize,
    pub n_cells: usize,
    pub h: Scalar,
    pub iterations: usize,
    /// The stopping rule was met before the budget ran out.
    pub converged: bool,
    /// max of the Ux, Uy and p initial residuals of the last iteration.
    pub residual: Scalar,
    pub forces: Forces,
    pub seconds: f64,
}

/// The stopping rule's two forces at one check.
fn fold_forces(rig: &Rig, s: &Simple<'_>, gpu: &Gpu) -> Result<Forces> {
    let u_cells = gpu.download(&s.u().f)?;
    let u_bf = gpu.download(&s.u().bf)?;
    let p_bf = gpu.download(&s.p().bf)?;
    Ok(body_forces(
        &rig.hm, &rig.body, rig.dz, &p_bf, &u_cells, &u_bf,
    ))
}

/// CFD1 or CFD2 on one rig, steady, from the seeded state (SPEC-LIT
/// 105.13). A run that exhausts its budget without meeting the stopping
/// rule returns `converged: false` - a finding the gate reports, not an
/// error to hide.
pub fn steady(gpu: &Gpu, rig: &Rig, case: Case, max_iters: usize) -> Result<SteadyRun> {
    if case == Case::Cfd3 {
        return Err(Error::Config(
            "turek_hron::steady: CFD3 is periodic; it has no steady run".to_string(),
        ));
    }
    let u_bar = case.u_bar();
    let started = Instant::now();
    let gm = GpuMesh::upload(gpu, &rig.hm)?;
    let ctrl = steady_controls();
    let mut s = Simple::new(gpu, &rig.hm, &gm, ctrl, buoyancy_off())?;
    set_bcs(gpu, &mut s, &rig.hm, &turek_rules(u_bar))?;

    // Seed: the inflow profile over the cells, its exact flux through the
    // faces - except where the velocity condition is a wall or an empty,
    // whose flux is zero - and a level pressure. The shape of
    // ale_flow::seat.
    let u_cells: Vec<Vec3> = rig
        .hm
        .c
        .iter()
        .map(|c| Vec3::new(inlet_velocity(c.y, u_bar), 0.0, 0.0))
        .collect();
    gpu.write(&mut s.u_mut().f, &u_cells)?;
    gpu.write(&mut s.p_mut().f, &vec![0.0 as Scalar; rig.hm.n_cells])?;
    let mut phi_f = vec![0.0 as Scalar; s.phi().f.len()];
    for (f, sfi) in rig.hm.sf.iter().enumerate() {
        phi_f[f] = Vec3::new(inlet_velocity(rig.hm.cf[f].y, u_bar), 0.0, 0.0).dot(*sfi);
    }
    gpu.write(&mut s.phi_mut().f, &phi_f)?;
    let rules = turek_rules(u_bar);
    let mut phi_b = vec![0.0 as Scalar; s.phi().bf.len()];
    for p in &rig.hm.patches {
        let flows = rules
            .iter()
            .find(|(n, _, _)| *n == p.name)
            .map(|(_, ub, _)| matches!(ub, VelocityBc::Inlet { .. } | VelocityBc::ZeroGradient))
            .unwrap_or(false);
        for k in 0..p.size {
            let i = p.start + k;
            if flows {
                phi_b[i] =
                    Vec3::new(inlet_velocity(rig.hm.b_cf[i].y, u_bar), 0.0, 0.0).dot(rig.hm.b_sf[i]);
            }
        }
    }
    gpu.write(&mut s.phi_mut().bf, &phi_b)?;
    s.initialise(gpu)?;

    let (nut, t) = laminar_fields(gpu, &rig.hm, &gm)?;
    let mut backend = PbicgstabBackend::new(ctrl.p_solver);
    backend.setup(gpu, &rig.hm, &gm, &SystemProbe::default())?;

    let mut last_forces = fold_forces(rig, &s, gpu)?;
    let mut last_res = 0.0 as Scalar;
    let mut converged = false;
    let mut iterations = max_iters;
    for it in 1..=max_iters {
        let perf = s.solve_step(gpu, &mut backend, &nut, &t)?;
        let res = crate::simple::nan_propagating_max(
            crate::simple::nan_propagating_max(
                perf.last.u[0].initial_residual,
                perf.last.u[1].initial_residual,
            ),
            perf.last.p.initial_residual,
        );
        last_res = res;
        if !res.is_finite() {
            return Err(Error::Config(format!(
                "turek_hron::steady: {} on level {} diverged at iteration {it} \
                 (initial residual {res})",
                case.name(),
                rig.level
            )));
        }
        if it % STEADY_CHECK_EVERY == 0 {
            let f = fold_forces(rig, &s, gpu)?;
            let drag_ok = (f.drag - last_forces.drag).abs() <= STEADY_FORCE_TOL * f.drag.abs();
            let lift_ok = (f.lift - last_forces.lift).abs() <= STEADY_FORCE_TOL * f.drag.abs();
            last_forces = f;
            if it >= STEADY_MIN_ITERS && drag_ok && lift_ok {
                converged = true;
                iterations = it;
                break;
            }
        }
    }
    if !converged {
        // The budget ended between checks: the reported forces are the last
        // check's, so take one more.
        last_forces = fold_forces(rig, &s, gpu)?;
    }
    Ok(SteadyRun {
        case,
        level: rig.level,
        n_cells: rig.hm.n_cells,
        h: rig.h,
        iterations,
        converged,
        residual: last_res,
        forces: last_forces,
        seconds: started.elapsed().as_secs_f64(),
    })
}

/// CFD1 or CFD2 on all four levels, and its two grid studies over the
/// finest three.
#[derive(Debug, Clone)]
pub struct SteadyStudy {
    pub case: Case,
    /// Finest first: L4, L3, L2, L1.
    pub runs: Vec<SteadyRun>,
    /// `vv::grid_study` over (h, drag) of the `STUDY_LEVELS` finest runs,
    /// finest first; `Err` carries the refusal's text.
    pub drag: std::result::Result<GridStudy, String>,
    pub lift: std::result::Result<GridStudy, String>,
}

/// Levels 4, 3, 2, 1 each loaded ONCE, CFD1 and CFD2 run on each with
/// `STEADY_MAX_ITERS[level - 1]`; returns `[CFD1 study, CFD2 study]`.
pub fn steady_studies(gpu: &Gpu) -> Result<Vec<SteadyStudy>> {
    let mut runs_cfd1 = Vec::new();
    let mut runs_cfd2 = Vec::new();
    for level in (1..=LEVEL_DIRS.len()).rev() {
        let rig = load_rig(level)?;
        runs_cfd1.push(steady(gpu, &rig, Case::Cfd1, STEADY_MAX_ITERS[level - 1])?);
        runs_cfd2.push(steady(gpu, &rig, Case::Cfd2, STEADY_MAX_ITERS[level - 1])?);
    }
    let (d1, l1) = studies_of(Case::Cfd1, &runs_cfd1);
    let (d2, l2) = studies_of(Case::Cfd2, &runs_cfd2);
    Ok(vec![
        SteadyStudy { drag: d1, lift: l1, case: Case::Cfd1, runs: runs_cfd1 },
        SteadyStudy { drag: d2, lift: l2, case: Case::Cfd2, runs: runs_cfd2 },
    ])
}

/// The drag study and the lift study over the `STUDY_LEVELS` finest of
/// `runs` (finest first) - the same levels, one quantity each, whichever
/// case the runs carry.
pub(crate) fn studies_of(
    _case: Case,
    runs: &[SteadyRun],
) -> (std::result::Result<GridStudy, String>, std::result::Result<GridStudy, String>) {
    let study = |value: fn(&Forces) -> Scalar| -> std::result::Result<GridStudy, String> {
        let levels: Vec<Level> = runs
            .iter()
            .take(STUDY_LEVELS)
            .map(|r| Level { h: r.h, value: value(&r.forces) })
            .collect();
        vv::grid_study(&levels).map_err(|e| e.to_string())
    };
    (study(|f| f.drag), study(|f| f.lift))
}

/// `|study.phi_ext - published| / |published|`, NaN when the study was
/// refused.
pub fn extrapolated_error(
    study: &std::result::Result<GridStudy, String>,
    published: Scalar,
) -> Scalar {
    match study {
        Ok(s) => (s.phi_ext - published).abs() / published.abs(),
        Err(_) => f64::NAN as Scalar,
    }
}

/// The analytic wake wobble of SPEC-LIT 105.13 - a displacement field, not
/// a smoother.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WobbleLaw {
    pub x0: Scalar,
    pub x1: Scalar,
    pub y0: Scalar,
    pub y1: Scalar,
    pub amplitude: Scalar,
    /// Unit, and in the x-y plane.
    pub direction: Vec3,
    pub period: Scalar,
}

impl WobbleLaw {
    /// x in `[WOBBLE_X0, WOBBLE_X1]`, y in `[0, CHANNEL_H]`,
    /// WOBBLE_AMPLITUDE along `(1, 1, 0)/sqrt 2`, WOBBLE_PERIOD.
    pub fn turek() -> Self {
        let r2 = (2.0 as Scalar).sqrt();
        WobbleLaw {
            x0: WOBBLE_X0,
            x1: WOBBLE_X1,
            y0: 0.0,
            y1: CHANNEL_H,
            amplitude: WOBBLE_AMPLITUDE,
            direction: Vec3::new(1.0 / r2, 1.0 / r2, 0.0),
            period: WOBBLE_PERIOD,
        }
    }

    /// 0 when `x <= x0`, `x >= x1`, `y <= y0` or `y >= y1`; otherwise
    /// `sin^2(pi (x - x0)/(x1 - x0)) * sin^2(pi (y - y0)/(y1 - y0))`.
    pub fn shape(&self, p: Vec3) -> Scalar {
        if p.x <= self.x0 || p.x >= self.x1 || p.y <= self.y0 || p.y >= self.y1 {
            return 0.0;
        }
        let sx =
            (std::f64::consts::PI * f64::from(p.x - self.x0) / f64::from(self.x1 - self.x0)).sin();
        let sy =
            (std::f64::consts::PI * f64::from(p.y - self.y0) / f64::from(self.y1 - self.y0)).sin();
        (sx * sx * sy * sy) as Scalar
    }

    /// `amplitude * shape(p) * sin(2 pi t / period) * direction`.
    pub fn displacement(&self, p: Vec3, t: Scalar) -> Vec3 {
        let s = (2.0 * std::f64::consts::PI * f64::from(t) / f64::from(self.period)).sin();
        self.amplitude * self.shape(p) * (s as Scalar) * self.direction
    }
}

/// A law bound to one mesh's rest points.
#[derive(Debug, Clone)]
pub struct Wobble {
    pub law: WobbleLaw,
    rest: Vec<Vec3>,
    /// false for every point of every face of every non-Empty patch.
    moves: Vec<bool>,
}

impl Wobble {
    /// Bind `law` to one mesh's rest points, refusing by name anything
    /// that could move a wall or break the bookkeeping - SPEC-LIT 105.13.
    pub fn new(hm: &HostMesh, rest: &[Vec3], faces: &[Vec<Label>], law: WobbleLaw) -> Result<Self> {
        if !law.amplitude.is_finite() || law.amplitude <= 0.0 {
            return Err(Error::Config(format!(
                "Wobble::new: amplitude is {}; it must be finite and positive",
                law.amplitude
            )));
        }
        if !law.period.is_finite() || law.period <= 0.0 {
            return Err(Error::Config(format!(
                "Wobble::new: period is {}; it must be finite and positive",
                law.period
            )));
        }
        if law.x1 <= law.x0 {
            return Err(Error::Config(format!(
                "Wobble::new: the box's x span is [{}, {}]; x1 must exceed x0",
                law.x0, law.x1
            )));
        }
        if law.y1 <= law.y0 {
            return Err(Error::Config(format!(
                "Wobble::new: the box's y span is [{}, {}]; y1 must exceed y0",
                law.y0, law.y1
            )));
        }
        let mag = (f64::from(law.direction.x * law.direction.x)
            + f64::from(law.direction.y * law.direction.y)
            + f64::from(law.direction.z * law.direction.z))
        .sqrt();
        if (mag - 1.0).abs() > 1e-12 {
            return Err(Error::Config(format!(
                "Wobble::new: direction's magnitude is {mag}; it must be unit"
            )));
        }
        if law.direction.z != 0.0 {
            return Err(Error::Config(format!(
                "Wobble::new: direction's z is {}; the empty planes' points may \
                 move only in their own plane",
                law.direction.z
            )));
        }
        if hm.n_points != 0 && rest.len() != hm.n_points {
            return Err(Error::Config(format!(
                "Wobble::new: {} rest points, and the mesh has {}",
                rest.len(),
                hm.n_points
            )));
        }
        if faces.len() != hm.n_internal_faces + hm.n_boundary_faces {
            return Err(Error::Config(format!(
                "Wobble::new: {} face point lists, and the mesh has {} faces \
                 ({} internal, {} boundary)",
                faces.len(),
                hm.n_internal_faces + hm.n_boundary_faces,
                hm.n_internal_faces,
                hm.n_boundary_faces
            )));
        }
        let mut moves = vec![true; rest.len()];
        for p in &hm.patches {
            if p.kind == PatchKind::Empty {
                continue;
            }
            for k in 0..p.size {
                let fi = hm.n_internal_faces + p.start + k;
                for &pt in &faces[fi] {
                    if law.shape(rest[pt as usize]) > 1e-12 {
                        return Err(Error::Config(format!(
                            "the wobble box reaches patch {} at point {}; a wall must not move",
                            p.name, pt
                        )));
                    }
                    moves[pt as usize] = false;
                }
            }
        }
        Ok(Wobble { law, rest: rest.to_vec(), moves })
    }

    /// Rest points, with `displacement(rest, t)` added to the points that
    /// move; a point that does not move is copied bitwise.
    pub fn points_at(&self, t: Scalar) -> Vec<Vec3> {
        self.rest
            .iter()
            .enumerate()
            .map(|(p, r)| {
                if self.moves[p] {
                    *r + self.law.displacement(*r, t)
                } else {
                    *r
                }
            })
            .collect()
    }

    /// The rest points the law is bound to.
    pub fn rest(&self) -> &[Vec3] {
        &self.rest
    }

    /// How many points the law actually moves.
    pub fn n_moving(&self) -> usize {
        self.moves.iter().filter(|m| **m).count()
    }
}

/// A periodic signal's reduction over whole periods (the paper's: mean =
/// `(max + min)/2`, amplitude = `(max - min)/2`, per period, averaged
/// here).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Periodic {
    pub mean: Scalar,
    pub amplitude: Scalar,
    pub frequency: Scalar,
    pub periods: usize,
}

/// Upward crossings of `v - mean(v)`, each linearly interpolated between
/// the two samples that bracket it.
pub fn upward_crossings(t: &[Scalar], v: &[Scalar]) -> Vec<Scalar> {
    if t.len() != v.len() || v.len() < 2 {
        return Vec::new();
    }
    let n = v.len();
    let mean = v.iter().sum::<Scalar>() / n as Scalar;
    let mut out = Vec::new();
    for i in 0..n - 1 {
        let a = v[i] - mean;
        let b = v[i + 1] - mean;
        if a < 0.0 && b >= 0.0 && b != a {
            let w = (0.0 - a) / (b - a);
            out.push(t[i] + w * (t[i + 1] - t[i]));
        }
    }
    out
}

/// One extreme's parabolic refinement: the vertex of the parabola through
/// the extreme sample and its two neighbours, uniform spacing. Falls back
/// to the sample when a neighbour is missing or the parabola degenerates.
fn refined_extreme(v: &[Scalar], i: usize, max: bool) -> Scalar {
    if i == 0 || i + 1 >= v.len() {
        return v[i];
    }
    let denom = v[i + 1] - 2.0 * v[i] + v[i - 1];
    let slopes = v[i + 1] - v[i - 1];
    if denom == 0.0 || (max && denom >= 0.0) || (!max && denom <= 0.0) {
        return v[i];
    }
    v[i] - slopes * slopes / (8.0 * denom)
}

/// Reduce `v` over the LAST `periods` intervals between consecutive
/// entries of `bounds` - the reduction of SPEC-LIT 105.13.
pub fn reduce_periods(
    t: &[Scalar],
    v: &[Scalar],
    bounds: &[Scalar],
    periods: usize,
) -> Result<Periodic> {
    if periods == 0 {
        return Err(Error::Config(format!(
            "reduce_periods: periods is 0; at least one full period is needed \
             ({} samples, {} bounds)",
            t.len(),
            bounds.len()
        )));
    }
    if t.len() != v.len() {
        return Err(Error::Config(format!(
            "reduce_periods: {} times and {} samples",
            t.len(),
            v.len()
        )));
    }
    if bounds.len() < periods + 1 {
        return Err(Error::Config(format!(
            "reduce_periods: {} bounds cannot carry {} periods - at least {} \
             crossings are needed",
            bounds.len(),
            periods,
            periods + 1
        )));
    }
    let last = &bounds[bounds.len() - (periods + 1)..];
    let mut sum_mean = 0.0f64;
    let mut sum_amp = 0.0f64;
    for k in 0..periods {
        let (b0, b1) = (last[k], last[k + 1]);
        let inside: Vec<Scalar> = (0..t.len())
            .filter(|&i| t[i] >= b0 && t[i] < b1)
            .map(|i| v[i])
            .collect();
        if inside.is_empty() {
            return Err(Error::Config(format!(
                "reduce_periods: no samples in [{b0}, {b1})"
            )));
        }
        let (mut imax, mut imin) = (0usize, 0usize);
        for (j, &x) in inside.iter().enumerate() {
            if x > inside[imax] {
                imax = j;
            }
            if x < inside[imin] {
                imin = j;
            }
        }
        // The parabola's neighbours are the SAMPLES' neighbours, so the
        // indices must be read back into t/v's own numbering.
        let first = (0..t.len())
            .find(|&i| t[i] >= b0 && t[i] < b1)
            .expect("the interval was checked non-empty");
        let vmax = refined_extreme(v, first + imax, true);
        let vmin = refined_extreme(v, first + imin, false);
        sum_mean += f64::from(vmax + vmin) / 2.0;
        sum_amp += f64::from(vmax - vmin) / 2.0;
    }
    let span = f64::from(last[periods]) - f64::from(last[0]);
    Ok(Periodic {
        mean: (sum_mean / periods as f64) as Scalar,
        amplitude: (sum_amp / periods as f64) as Scalar,
        frequency: (periods as f64 / span) as Scalar,
        periods,
    })
}

/// One continuation's record.
#[derive(Debug, Clone)]
pub struct Trace {
    pub moving: bool,
    pub t: Vec<Scalar>,
    pub drag: Vec<Scalar>,
    pub lift: Vec<Scalar>,
    /// min over steps and cells of `gm.v / hm.v`; 1.0 on the static run.
    pub min_volume_ratio: Scalar,
    /// max over steps of `|x - rest|` over points of non-Empty patches.
    pub max_boundary_displacement: Scalar,
    /// max over steps of `|x - rest|` over points with `rest x <= WOBBLE_X0`.
    pub max_upstream_displacement: Scalar,
    /// max over steps of `|x - rest|` over all points.
    pub max_displacement: Scalar,
    /// max over steps of `perf.last.continuity_error`.
    pub worst_continuity: Scalar,
    pub seconds: f64,
}

#[derive(Debug, Clone)]
pub struct Cfd3Traces {
    pub spin_seconds: f64,
    pub static_run: Trace,
    pub moving: Trace,
}

/// The spin-up's four state arrays, copied to the host.
struct Snapshot {
    u: Vec<Vec3>,
    p: Vec<Scalar>,
    phi_f: Vec<Scalar>,
    phi_bf: Vec<Scalar>,
}

/// One continuation from the snapshot, static or wobbling - SPEC-LIT
/// 105.13's step order: `begin_time_step` from the second step on, the
/// wobble's points, `move_mesh`, then the outer correctors.
#[allow(clippy::too_many_arguments)]
fn run_continuation(
    gpu: &Gpu,
    rig: &Rig,
    ctrl: SimpleControls,
    snap: &Snapshot,
    moving: bool,
    wobble: Option<&Wobble>,
    dt: Scalar,
    spin_steps: usize,
    window_steps: usize,
) -> Result<Trace> {
    let started = Instant::now();
    let gm = GpuMesh::upload(gpu, &rig.hm)?;
    let mut s = Simple::new(gpu, &rig.hm, &gm, ctrl, buoyancy_off())?;
    set_bcs(gpu, &mut s, &rig.hm, &turek_rules(2.0))?;
    gpu.write(&mut s.u_mut().f, &snap.u)?;
    gpu.write(&mut s.p_mut().f, &snap.p)?;
    gpu.write(&mut s.phi_mut().f, &snap.phi_f)?;
    gpu.write(&mut s.phi_mut().bf, &snap.phi_bf)?;
    s.initialise(gpu)?;
    let (nut, t) = laminar_fields(gpu, &rig.hm, &gm)?;
    let mut backend = PbicgstabBackend::new(ctrl.p_solver);
    backend.setup(gpu, &rig.hm, &gm, &SystemProbe::default())?;
    if moving {
        let ale = AleMesh::new(gpu, &rig.hm, &gm, &rig.raw.points, &rig.csr)?;
        s.attach_motion(gpu, ale, &[])?;
    }

    let mut tr = Trace {
        moving,
        t: Vec::with_capacity(window_steps),
        drag: Vec::with_capacity(window_steps),
        lift: Vec::with_capacity(window_steps),
        min_volume_ratio: if moving {
            f64::INFINITY as Scalar
        } else {
            1.0
        },
        max_boundary_displacement: 0.0,
        max_upstream_displacement: 0.0,
        max_displacement: 0.0,
        worst_continuity: 0.0,
        seconds: 0.0,
    };
    for k in 0..window_steps {
        if k > 0 {
            s.begin_time_step(gpu, dt)?;
        }
        if let Some(w) = wobble {
            let pts = w.points_at((k + 1) as Scalar * dt);
            for (p, r) in pts.iter().zip(w.rest()) {
                let d = (p.x - r.x) * (p.x - r.x)
                    + (p.y - r.y) * (p.y - r.y)
                    + (p.z - r.z) * (p.z - r.z);
                let d = (f64::from(d)).sqrt();
                tr.max_displacement = tr.max_displacement.max(d as Scalar);
                if r.x <= WOBBLE_X0 {
                    tr.max_upstream_displacement = tr.max_upstream_displacement.max(d as Scalar);
                }
            }
            for ((p, r), mv) in pts.iter().zip(w.rest()).zip(&w.moves) {
                if !*mv {
                    let d = ((p.x - r.x).powi(2) + (p.y - r.y).powi(2) + (p.z - r.z).powi(2)).sqrt();
                    tr.max_boundary_displacement = tr.max_boundary_displacement.max(d);
                }
            }
            s.motion_mut()
                .ok_or_else(|| Error::Config("cfd3: no motion is attached".to_string()))?
                .set_points(gpu, &pts)?;
            s.move_mesh(gpu)?;
        }
        let perf = s.solve_step(gpu, &mut backend, &nut, &t)?;
        tr.worst_continuity = tr.worst_continuity.max(perf.last.continuity_error);
        let u_cells = gpu.download(&s.u().f)?;
        let u_bf = gpu.download(&s.u().bf)?;
        let p_bf = gpu.download(&s.p().bf)?;
        let f = body_forces(&rig.hm, &rig.body, rig.dz, &p_bf, &u_cells, &u_bf);
        tr.t.push(spin_steps as Scalar * dt + (k + 1) as Scalar * dt);
        tr.drag.push(f.drag);
        tr.lift.push(f.lift);
        if moving {
            let v = gpu.download(&gm.v)?;
            for (vc, v0) in v.iter().zip(rig.hm.v.iter()).take(rig.hm.n_cells) {
                tr.min_volume_ratio = tr.min_volume_ratio.min(vc / v0);
            }
        }
    }
    tr.seconds = started.elapsed().as_secs_f64();
    Ok(tr)
}

/// The spin-up and the two continuations (SPEC-LIT 105.13): a static run
/// to `t = spin_steps * dt` through the ramp, its state copied to the
/// host, and from that one state two 1,500-step continuations that differ
/// only in the wobble.
pub fn cfd3_traces(
    gpu: &Gpu,
    rig: &Rig,
    dt: Scalar,
    spin_steps: usize,
    window_steps: usize,
    law: WobbleLaw,
) -> Result<Cfd3Traces> {
    let wobble = Wobble::new(&rig.hm, &rig.raw.points, &rig.raw.faces, law)?;

    // ---- the spin-up, static, through the paper's ramp --------------------
    let started = Instant::now();
    let gm = GpuMesh::upload(gpu, &rig.hm)?;
    let ctrl = transient_controls(dt);
    let mut s = Simple::new(gpu, &rig.hm, &gm, ctrl, buoyancy_off())?;
    set_bcs(gpu, &mut s, &rig.hm, &turek_rules(2.0))?;
    gpu.write(
        &mut s.u_mut().ref_value,
        &inlet_ref_values(&rig.hm, "inlet", 2.0, ramp(0.0)),
    )?;
    gpu.write(&mut s.u_mut().f, &vec![Vec3::ZERO; rig.hm.n_cells])?;
    gpu.write(&mut s.p_mut().f, &vec![0.0 as Scalar; rig.hm.n_cells])?;
    let nf = s.phi().f.len();
    let nbf = s.phi().bf.len();
    gpu.write(&mut s.phi_mut().f, &vec![0.0 as Scalar; nf])?;
    gpu.write(&mut s.phi_mut().bf, &vec![0.0 as Scalar; nbf])?;
    s.initialise(gpu)?;
    let (nut, t) = laminar_fields(gpu, &rig.hm, &gm)?;
    let mut backend = PbicgstabBackend::new(ctrl.p_solver);
    backend.setup(gpu, &rig.hm, &gm, &SystemProbe::default())?;
    let fk = FieldKernels::new(gpu)?;
    for k in 0..spin_steps {
        let t_now = (k + 1) as Scalar * dt;
        if k > 0 {
            s.begin_time_step(gpu, dt)?;
        }
        if t_now <= RAMP_T + dt {
            gpu.write(
                &mut s.u_mut().ref_value,
                &inlet_ref_values(&rig.hm, "inlet", 2.0, ramp(t_now)),
            )?;
            field_ops::correct_boundary_conditions_vector(gpu, &fk, s.u_mut(), &gm)?;
        }
        let perf = s.solve_step(gpu, &mut backend, &nut, &t)?;
        let res = crate::simple::nan_propagating_max(
            crate::simple::nan_propagating_max(
                perf.last.u[0].initial_residual,
                perf.last.u[1].initial_residual,
            ),
            perf.last.p.initial_residual,
        );
        if !res.is_finite() || !perf.last.continuity_error.is_finite() {
            return Err(Error::Config(format!(
                "turek_hron::cfd3: the spin-up went non-finite at step {k} \
                 (residual {res}, continuity {})",
                perf.last.continuity_error
            )));
        }
    }
    let spin_seconds = started.elapsed().as_secs_f64();
    let snap = Snapshot {
        u: gpu.download(&s.u().f)?,
        p: gpu.download(&s.p().f)?,
        phi_f: gpu.download(&s.phi().f)?,
        phi_bf: gpu.download(&s.phi().bf)?,
    };
    drop(s);
    drop(gm);

    let static_run = run_continuation(
        gpu, rig, transient_controls(dt), &snap, false, None, dt, spin_steps, window_steps,
    )?;
    let moving = run_continuation(
        gpu, rig, transient_controls(dt), &snap, true, Some(&wobble), dt, spin_steps, window_steps,
    )?;
    Ok(Cfd3Traces { spin_seconds, static_run, moving })
}

/// CFD3's whole record: the two traces, each one's reductions over its own
/// last four lift periods, and the four relative differences.
#[derive(Debug, Clone)]
pub struct Cfd3 {
    pub level: usize,
    pub dt: Scalar,
    pub law: WobbleLaw,
    pub traces: Cfd3Traces,
    pub static_drag: std::result::Result<Periodic, String>,
    pub static_lift: std::result::Result<Periodic, String>,
    pub moving_drag: std::result::Result<Periodic, String>,
    pub moving_lift: std::result::Result<Periodic, String>,
    /// `|moving - static| / |static|` for drag mean, drag amplitude, lift
    /// mean, lift amplitude, in that order; NaN where a reduction was
    /// refused.
    pub rel: [Scalar; 4],
}

fn reduce_over_lift(
    tr: &Trace,
    periods: usize,
) -> (
    std::result::Result<Periodic, String>,
    std::result::Result<Periodic, String>,
) {
    let bounds = upward_crossings(&tr.t, &tr.lift);
    let drag = reduce_periods(&tr.t, &tr.drag, &bounds, periods).map_err(|e| e.to_string());
    let lift = reduce_periods(&tr.t, &tr.lift, &bounds, periods).map_err(|e| e.to_string());
    (drag, lift)
}

fn rel_diff(
    moving: &std::result::Result<Periodic, String>,
    stat: &std::result::Result<Periodic, String>,
    field: fn(&Periodic) -> Scalar,
) -> Scalar {
    match (moving, stat) {
        (Ok(m), Ok(s)) => (field(m) - field(s)).abs() / field(s).abs(),
        _ => f64::NAN as Scalar,
    }
}

/// CFD3 at CFD3_LEVEL with CFD3_DT, CFD3_SPIN_STEPS, CFD3_WINDOW_STEPS,
/// `WobbleLaw::turek()`, each trace reduced over its own last
/// CFD3_PERIODS lift periods.
pub fn cfd3(gpu: &Gpu) -> Result<Cfd3> {
    let rig = load_rig(CFD3_LEVEL)?;
    let law = WobbleLaw::turek();
    let traces = cfd3_traces(gpu, &rig, CFD3_DT, CFD3_SPIN_STEPS, CFD3_WINDOW_STEPS, law)?;
    let (static_drag, static_lift) = reduce_over_lift(&traces.static_run, CFD3_PERIODS);
    let (moving_drag, moving_lift) = reduce_over_lift(&traces.moving, CFD3_PERIODS);
    let rel = [
        rel_diff(&moving_drag, &static_drag, |p| p.mean),
        rel_diff(&moving_drag, &static_drag, |p| p.amplitude),
        rel_diff(&moving_lift, &static_lift, |p| p.mean),
        rel_diff(&moving_lift, &static_lift, |p| p.amplitude),
    ];
    Ok(Cfd3 {
        level: CFD3_LEVEL,
        dt: CFD3_DT,
        law,
        traces,
        static_drag,
        static_lift,
        moving_drag,
        moving_lift,
        rel,
    })
}

#[cfg(test)]
mod tests;
