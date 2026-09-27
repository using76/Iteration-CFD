// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The SIMPLE loop on a moving mesh, and Gate 105-B - SPEC-LIT 105.10.
//!
//! Written from:
//!   ofgpu SPEC-LIT.md 105.7-105.10 - which consumer reads which flux, the
//!     moving wall, the smoother, and the gate this module runs
//!   ofgpu SPEC-LIT.md 94.1 - the grid study the stroke's observed order
//!     reads, and the extrapolation it reports beside it
//! No GPL-licensed source was consulted.
//!
//! The piston is the OPEN piston, not the plan document's closed one. A gas
//! compressed uniformly in a closed cylinder moves as `u = x (dx_w/dt) / x_w`,
//! whose divergence is `(dx_w/dt) / x_w`, and an incompressible solver cannot
//! carry it. The open piston pushes fluid out through an open end at
//! `U = dx_w/dt` everywhere, which `div U = 0`, and the mesh velocity between
//! the two ends is whatever the smoother makes it - so the relative flux is
//! genuinely non-zero in the interior and the gate exercises it, not just the
//! boundary condition.
//!
//! The time order is measured on a STROKING OUTLET rather than on an
//! accelerating piston, because at a face whose flux the velocity condition
//! prescribes the momentum force flux is written zero, so the reconstructed
//! pressure force carries half of a wall-normal pressure gradient; an
//! accelerating piston has one, and its wall cell would stay wrong by an
//! amount that does not fall with `dt`. The stroke has no such face: `p = P0`
//! at the fixed end and `p = 0` at the moving end leave `U(t)` uniform with
//! `dU/dt = P0 / L(t)` exactly, and the scheme's order shows.
//!
//! The first step of every gate loop does NOT call `begin_time_step`, so the
//! time state's step counter is 0 during it and `backward` takes its Euler
//! row - for the momentum derivative and for `AleMesh::advance` alike, both
//! reading the same state. Starting BDF2 on two equal old levels instead is a
//! first-order error whenever `dU/dt(0)` is not zero, and the stroke's is
//! `P0 / L0`.
//!
//! NOT here: the case block that prescribes a motion and any driver that runs
//! one - the next unit's. Nothing in this module puts work on the device; it
//! orchestrates the solvers that do.

use crate::device::Gpu;
use crate::error::{Error, Result};
use crate::field::{BcKind, GpuScalarField, GpuVectorField};
use crate::field_ops::{self, FieldKernels};
use crate::io::case::{DivScheme, SolverControls};
use crate::mesh::ale::AleMesh;
use crate::mesh::gpugeom::{FacePointCsr, flatten_faces};
use crate::mesh::motion::{Displacement, MeshMotion, PatchMotion};
use crate::mesh::refined::{self, RefinedBox};
use crate::mesh::{GpuMesh, HostMesh, PatchKind};
use crate::momentum::{BuoyancyCoeffs, MomentumControls};
use crate::pressure::{PbicgstabBackend, PressureBackend, SystemProbe};
use crate::simple::{PimplePerformance, Simple, SimpleControls};
use crate::timescheme::DdtScheme;
use crate::vv::{self, GridStudy, Level};
use crate::{Label, Scalar, Vec3};

/// The piston's speed, metres per second (SPEC-LIT 105.10 (a)).
pub const PISTON_C: Scalar = 0.5;
/// The piston's time step.
pub const PISTON_DT: Scalar = 0.005;
/// How many steps the piston runs.
pub const PISTON_STEPS: usize = 40;
/// The piston's base grid: 8x3x3 cells of `0.125 x 0.25 x 0.25`.
pub const PISTON_N: [usize; 3] = [8, 3, 3];
/// The stroke's base grid: 16x1x1 cells of `0.0625 x 0.1 x 0.1`.
pub const STROKE_N: [usize; 3] = [16, 1, 1];
/// The stroke channel's rest length.
pub const STROKE_L0: Scalar = 1.0;
/// The stroke's end amplitude.
pub const STROKE_A: Scalar = 0.2;
/// The stroke's period.
pub const STROKE_PERIOD: Scalar = 1.0;
/// The time the stroke gate stops at.
pub const STROKE_T: Scalar = 0.25;
/// The stroke's initial velocity.
pub const STROKE_U0: Scalar = 0.5;
/// The stroke's fixed-end pressure.
pub const STROKE_P0: Scalar = 1.0;
/// The stroke's three step counts, finest first.
pub const STROKE_STEPS: [usize; 3] = [320, 160, 80];
/// The uniform state the piston must keep (SPEC-LIT 105.10).
pub const UNIFORM_TOL: Scalar = 1e-10;
/// The relative flux through the moving wall, against its mesh flux.
pub const WALL_FLUX_TOL: Scalar = 1e-12;
/// The absolute flux's cell sum, against `c |Sf_wall|`.
pub const CONTINUITY_TOL: Scalar = 1e-10;
/// How far the observed order may sit from the scheme's own.
pub const ORDER_BAND: Scalar = 0.1;

/// What the piston measured - SPEC-LIT 105.10 (a). Every `worst_*` is a
/// maximum over all steps.
#[derive(Debug, Clone)]
pub struct PistonRun {
    pub scheme: DdtScheme,
    pub bounded: bool,
    pub steps: usize,
    pub n_cells: usize,
    /// max_c |U_c - (-c, 0, 0)| / c
    pub worst_u: Scalar,
    /// max_c |p_c| / c^2
    pub worst_p: Scalar,
    /// max over the piston's faces of |phi_b - phi_mesh,b| / |phi_mesh,b|
    pub worst_wall_flux: Scalar,
    /// max_c |sum_f phi_f| / (c |Sf_wall|)
    pub worst_continuity: Scalar,
    /// min_c V_c / V_c(rest)
    pub min_volume_ratio: Scalar,
    /// how far the piston travelled, as a fraction of the box length
    pub stroke: Scalar,
}

/// One stroking-outlet run - SPEC-LIT 105.10 (b).
#[derive(Debug, Clone)]
pub struct StrokeRun {
    pub scheme: DdtScheme,
    pub steps: usize,
    pub dt: Scalar,
    /// the volume-weighted mean of U_x at STROKE_T
    pub value: Scalar,
    /// max_c U_x - min_c U_x at STROKE_T
    pub spread: Scalar,
    pub min_volume_ratio: Scalar,
}

/// The three runs, finest first, and what `vv::grid_study` made of them.
#[derive(Debug, Clone)]
pub struct StrokeStudy {
    pub scheme: DdtScheme,
    pub runs: Vec<StrokeRun>,
    pub exact: Scalar,
    pub study: GridStudy,
    /// `study.p`, NaN when the triplet had no order
    pub p: Scalar,
    /// |finest - exact| / exact
    pub err_fine: Scalar,
    /// |study.phi_ext - exact| / exact
    pub err_ext: Scalar,
}

/// Gate 105-B for one scheme.
#[derive(Debug, Clone)]
pub struct Gate105b {
    pub conservative: PistonRun,
    pub bounded: PistonRun,
    pub stroke: StrokeStudy,
}

/// The host side of a gate fixture: the refined box, the motion built on it,
/// the moving wall's faces, and the face-point CSR the `AleMesh` wants.
pub(crate) struct Rig {
    pub rb: RefinedBox,
    pub motion: MeshMotion,
    pub wall_faces: Vec<Label>,
    pub csr: FacePointCsr,
}

/// Name the six patches' kinds and re-run the geometry, which refills
/// `b_kind` from them - the fixtures' patch setup (SPEC-LIT 105.10).
pub(crate) fn set_kinds(rb: &mut RefinedBox, kinds: [PatchKind; 6]) -> Result<()> {
    for (p, k) in rb.mesh.patches.iter_mut().zip(kinds) {
        p.kind = k;
        p.type_name = match k {
            PatchKind::Wall => "wall",
            PatchKind::Symmetry => "symmetry",
            _ => "patch",
        }
        .to_string();
    }
    rb.mesh.compute_geometry(&rb.points, &rb.faces)
}

/// The piston fixture: `xmin` at rest, `xmax` the moving wall at constant
/// speed `-c`, the four sides sliding in their planes (SPEC-LIT 105.10 (a)).
pub(crate) fn piston_rig() -> Result<Rig> {
    let mut rb = refined::build(PISTON_N, Vec3::new(0.125, 0.25, 0.25), &[0u32; 72])?;
    set_kinds(
        &mut rb,
        [
            PatchKind::Generic,
            PatchKind::Wall,
            PatchKind::Symmetry,
            PatchKind::Symmetry,
            PatchKind::Symmetry,
            PatchKind::Symmetry,
        ],
    )?;
    let motion = MeshMotion::new(
        &rb.mesh,
        &rb.points,
        &rb.faces,
        &[
            ("xmin", PatchMotion::Fixed),
            (
                "xmax",
                PatchMotion::Move(Displacement::Linear {
                    velocity: Vec3::new(-PISTON_C, 0.0, 0.0),
                }),
            ),
            ("ymin", PatchMotion::Slide),
            ("ymax", PatchMotion::Slide),
            ("zmin", PatchMotion::Slide),
            ("zmax", PatchMotion::Slide),
        ],
    )?;
    let wall_faces = motion.wall_faces(&rb.mesh, &["xmax"])?;
    let csr = flatten_faces(&rb.faces);
    Ok(Rig { rb, motion, wall_faces, csr })
}

/// The stroke fixture: both ends prescribed, the four sides sliding - the
/// outlet law is the caller's, `Fixed` for the never-moving run
/// (SPEC-LIT 105.10 (b)).
pub(crate) fn stroke_rig(outlet: PatchMotion) -> Result<Rig> {
    let mut rb = refined::build(STROKE_N, Vec3::new(0.0625, 0.1, 0.1), &[0u32; 16])?;
    set_kinds(
        &mut rb,
        [
            PatchKind::Generic,
            PatchKind::Generic,
            PatchKind::Symmetry,
            PatchKind::Symmetry,
            PatchKind::Symmetry,
            PatchKind::Symmetry,
        ],
    )?;
    let motion = MeshMotion::new(
        &rb.mesh,
        &rb.points,
        &rb.faces,
        &[
            ("xmin", PatchMotion::Fixed),
            ("xmax", outlet),
            ("ymin", PatchMotion::Slide),
            ("ymax", PatchMotion::Slide),
            ("zmin", PatchMotion::Slide),
            ("zmax", PatchMotion::Slide),
        ],
    )?;
    let csr = flatten_faces(&rb.faces);
    Ok(Rig { rb, motion, wall_faces: Vec::new(), csr })
}

/// The stroke's outlet law: the end of the channel oscillates along `x`.
pub fn stroke_law() -> PatchMotion {
    PatchMotion::Move(Displacement::Sine {
        amplitude: Vec3::new(STROKE_A, 0.0, 0.0),
        period: STROKE_PERIOD,
    })
}

/// The gate's solver settings (SPEC-LIT 105.10): tight linear solves, no
/// relaxation, three outer correctors of two PISO passes each.
pub(crate) fn flow_controls(scheme: DdtScheme, dt: Scalar, bounded: bool) -> SimpleControls {
    let solver = SolverControls {
        tolerance: 1e-13,
        rel_tol: 0.0,
        max_iter: 2000,
        min_iter: 0,
        check_interval: 1,
        ..SolverControls::default()
    };
    SimpleControls {
        momentum: MomentumControls {
            nu: 0.01,
            u_solver: solver,
            u_relax: 1.0,
            steady: false,
            delta_t: dt,
            ddt: scheme,
            div_scheme: DivScheme::Upwind,
            bounded_convection: bounded,
            sn_grad: crate::fv::SnGradScheme::Uncorrected,
            variable_viscosity_stress: false,
            ..MomentumControls::default()
        },
        p_solver: solver,
        p_relax: 1.0,
        n_non_orth_correctors: 0,
        n_correctors: 2,
        n_outer_correctors: 3,
        momentum_predictor: true,
        report_continuity: true,
    }
}

/// Buoyancy switched off: zero gravity at the fixtures' reference
/// temperature, so the body force is zero to the last bit.
pub(crate) fn buoyancy_off() -> BuoyancyCoeffs {
    BuoyancyCoeffs {
        g: Vec3::ZERO,
        t_ref: 293.15,
        t_min: 1.0,
    }
}

/// One patch's boundary condition, the gate fixtures' vocabulary.
#[derive(Clone, Copy)]
pub(crate) enum Bc<T> {
    ZeroGradient,
    Value(T),
    Symmetry,
}

/// Write a velocity boundary condition patch by patch, the way the fixtures
/// say it: host arrays over the boundary faces, then one write per buffer.
pub(crate) fn set_vector_bcs(
    gpu: &Gpu,
    u: &mut GpuVectorField,
    hm: &HostMesh,
    per_patch: &[Bc<Vec3>; 6],
) -> Result<()> {
    let nbf = hm.n_boundary_faces;
    let mut kind = vec![BcKind::ZeroGradient as Label; nbf];
    let mut fr = vec![0.0 as Scalar; nbf];
    let mut rv = vec![Vec3::ZERO; nbf];
    let rg = vec![Vec3::ZERO; nbf];
    for (p, bc) in hm.patches.iter().zip(per_patch) {
        for k in 0..p.size {
            let i = p.start + k;
            match *bc {
                Bc::ZeroGradient => {}
                Bc::Value(v) => {
                    kind[i] = BcKind::FixedValue as Label;
                    fr[i] = 1.0;
                    rv[i] = v;
                }
                Bc::Symmetry => kind[i] = BcKind::Symmetry as Label,
            }
        }
    }
    gpu.write(&mut u.bc_kind, &kind)?;
    gpu.write(&mut u.fr, &fr)?;
    gpu.write(&mut u.ref_value, &rv)?;
    gpu.write(&mut u.ref_grad, &rg)
}

/// The same for a scalar field.
pub(crate) fn set_scalar_bcs(
    gpu: &Gpu,
    s: &mut GpuScalarField,
    hm: &HostMesh,
    per_patch: &[Bc<Scalar>; 6],
) -> Result<()> {
    let nbf = hm.n_boundary_faces;
    let mut kind = vec![BcKind::ZeroGradient as Label; nbf];
    let mut fr = vec![0.0 as Scalar; nbf];
    let mut rv = vec![0.0 as Scalar; nbf];
    let rg = vec![0.0 as Scalar; nbf];
    for (p, bc) in hm.patches.iter().zip(per_patch) {
        for k in 0..p.size {
            let i = p.start + k;
            match *bc {
                Bc::ZeroGradient => {}
                Bc::Value(v) => {
                    kind[i] = BcKind::FixedValue as Label;
                    fr[i] = 1.0;
                    rv[i] = v;
                }
                Bc::Symmetry => kind[i] = BcKind::Symmetry as Label,
            }
        }
    }
    gpu.write(&mut s.bc_kind, &kind)?;
    gpu.write(&mut s.fr, &fr)?;
    gpu.write(&mut s.ref_value, &rv)?;
    gpu.write(&mut s.ref_grad, &rg)
}

/// A zero eddy viscosity with calculated boundary values, and a uniform
/// temperature at the reference value with zero-gradient boundaries, already
/// evaluated - a laminar isothermal run's two model inputs.
pub(crate) fn laminar_fields(
    gpu: &Gpu,
    hm: &HostMesh,
    gm: &GpuMesh,
) -> Result<(GpuScalarField, GpuScalarField)> {
    let mut nut = GpuScalarField::zeros(gpu, gm, "nut")?;
    let kinds = vec![BcKind::Calculated as Label; hm.n_boundary_faces];
    gpu.write(&mut nut.bc_kind, &kinds)?;
    let cells = vec![293.15 as Scalar; gm.n_cells];
    let mut t = GpuScalarField::zeros(gpu, gm, "T")?;
    gpu.write(&mut t.f, &cells)?;
    set_scalar_bcs(
        gpu,
        &mut t,
        hm,
        &[
            Bc::ZeroGradient,
            Bc::ZeroGradient,
            Bc::ZeroGradient,
            Bc::ZeroGradient,
            Bc::ZeroGradient,
            Bc::ZeroGradient,
        ],
    )?;
    field_ops::correct_boundary_conditions(gpu, &FieldKernels::new(gpu)?, &mut t, gm)?;
    Ok((nut, t))
}

/// Seed a seated `Simple`: the two boundary conditions, a uniform velocity,
/// zero pressure, and the exact uniform flux. SPEC-LIT 105.10's fixtures all
/// start from a state the exact solution never leaves.
pub(crate) fn seat(
    gpu: &Gpu,
    s: &mut Simple<'_>,
    hm: &HostMesh,
    gm: &GpuMesh,
    u_bc: &[Bc<Vec3>; 6],
    p_bc: &[Bc<Scalar>; 6],
    u0: Vec3,
) -> Result<()> {
    set_vector_bcs(gpu, s.u_mut(), hm, u_bc)?;
    set_scalar_bcs(gpu, s.p_mut(), hm, p_bc)?;

    let u_cells = vec![u0; gm.n_cells];
    gpu.write(&mut s.u_mut().f, &u_cells)?;

    let p_cells = vec![0.0 as Scalar; gm.n_cells];
    gpu.write(&mut s.p_mut().f, &p_cells)?;

    let sf = gpu.download(&gm.sf)?;
    let mut phi_f = vec![0.0 as Scalar; s.phi().f.len()];
    for (f, sfi) in sf.iter().enumerate() {
        phi_f[f] = u0.dot(*sfi);
    }
    gpu.write(&mut s.phi_mut().f, &phi_f)?;

    let b_sf = gpu.download(&gm.b_sf)?;
    let mut phi_b = vec![0.0 as Scalar; s.phi().bf.len()];
    for (i, sfi) in b_sf.iter().enumerate() {
        phi_b[i] = u0.dot(*sfi);
    }
    gpu.write(&mut s.phi_mut().bf, &phi_b)?;

    s.initialise(gpu)
}

/// One step of a moving-mesh run, in the order SPEC-LIT 105.10 prescribes:
/// `begin_time_step` from the SECOND step on - the first step keeps the time
/// state's step counter at 0, so `backward` takes its Euler row - then the
/// law's points, the mesh advance with the wall's value and the refreshed
/// relative flux, and only then the outer correctors.
pub(crate) fn take_moving_step(
    gpu: &Gpu,
    s: &mut Simple<'_>,
    backend: &mut dyn PressureBackend,
    motion: &MeshMotion,
    k: usize,
    dt: Scalar,
    nut: &GpuScalarField,
    t: &GpuScalarField,
) -> Result<PimplePerformance> {
    if k > 0 {
        s.begin_time_step(gpu, dt)?;
    }
    let pts = motion.points_at((k + 1) as Scalar * dt);
    s.motion_mut()
        .ok_or_else(|| Error::Config("ale_flow: no motion is attached".to_string()))?
        .set_points(gpu, &pts)?;
    s.move_mesh(gpu)?;
    s.solve_step(gpu, backend, nut, t)
}

/// The piston of SPEC-LIT 105.10 (a): `PISTON_STEPS` steps at constant speed,
/// every `worst_*` folded over the whole run. The exact state
/// `U = (-c, 0, 0)`, `p = 0` must survive the whole SIMPLE loop unchanged.
pub fn piston(gpu: &Gpu, scheme: DdtScheme, bounded: bool) -> Result<PistonRun> {
    let rig = piston_rig()?;
    let hm = &rig.rb.mesh;
    let gm = GpuMesh::upload(gpu, hm)?;
    let ale = AleMesh::new(gpu, hm, &gm, &rig.rb.points, &rig.csr)?;
    let ctrl = flow_controls(scheme, PISTON_DT, bounded);
    let mut s = Simple::new(gpu, hm, &gm, ctrl, buoyancy_off())?;
    seat(
        gpu,
        &mut s,
        hm,
        &gm,
        &[
            Bc::ZeroGradient,
            Bc::Value(Vec3::new(-PISTON_C, 0.0, 0.0)),
            Bc::Symmetry,
            Bc::Symmetry,
            Bc::Symmetry,
            Bc::Symmetry,
        ],
        &[
            Bc::Value(0.0),
            Bc::ZeroGradient,
            Bc::Symmetry,
            Bc::Symmetry,
            Bc::Symmetry,
            Bc::Symmetry,
        ],
        Vec3::new(-PISTON_C, 0.0, 0.0),
    )?;
    s.attach_motion(gpu, ale, &rig.wall_faces)?;
    let mut backend = PbicgstabBackend::new(ctrl.p_solver);
    backend.setup(gpu, hm, &gm, &SystemProbe::default())?;
    let (nut, t) = laminar_fields(gpu, hm, &gm)?;

    let n_cells = hm.n_cells;
    let wall0 = hm.b_mag_sf[rig.wall_faces[0] as usize];
    let mut worst_u = 0.0 as Scalar;
    let mut worst_p = 0.0 as Scalar;
    let mut worst_wall_flux = 0.0 as Scalar;
    let mut worst_continuity = 0.0 as Scalar;
    let mut min_volume_ratio = Scalar::INFINITY;
    for k in 0..PISTON_STEPS {
        let perf = take_moving_step(gpu, &mut s, &mut backend, &rig.motion, k, PISTON_DT, &nut, &t)?;
        let u = gpu.download(&s.u().f)?;
        let p = gpu.download(&s.p().f)?;
        let phi_b = gpu.download(&s.phi().bf)?;
        let pm_b = gpu.download(&s.motion().expect("attached").phi_mesh.bf)?;
        let v = gpu.download(&gm.v)?;
        for c in 0..n_cells {
            worst_u = worst_u.max((u[c] - Vec3::new(-PISTON_C, 0.0, 0.0)).mag() / PISTON_C);
            worst_p = worst_p.max(p[c].abs() / (PISTON_C * PISTON_C));
        }
        for &f in &rig.wall_faces {
            let i = f as usize;
            let pm = pm_b[i];
            worst_wall_flux = worst_wall_flux.max(((phi_b[i] - pm) / pm).abs());
        }
        worst_continuity =
            worst_continuity.max(perf.last.continuity_error / (PISTON_C * wall0));
        for c in 0..n_cells {
            min_volume_ratio = min_volume_ratio.min(v[c] / hm.v[c]);
        }
    }
    Ok(PistonRun {
        scheme,
        bounded,
        steps: PISTON_STEPS,
        n_cells,
        worst_u,
        worst_p,
        worst_wall_flux,
        worst_continuity,
        min_volume_ratio,
        stroke: PISTON_C * PISTON_DT * PISTON_STEPS as Scalar / 1.0,
    })
}

/// The stroking outlet of SPEC-LIT 105.10 (b): `steps` steps to `STROKE_T`,
/// the volume-weighted mean of `U_x` there, and how far apart the cells sit.
pub fn stroke(gpu: &Gpu, scheme: DdtScheme, steps: usize) -> Result<StrokeRun> {
    if steps == 0 {
        return Err(Error::Config(
            "ale_flow::stroke: steps must be at least one".to_string(),
        ));
    }
    let rig = stroke_rig(stroke_law())?;
    let hm = &rig.rb.mesh;
    let gm = GpuMesh::upload(gpu, hm)?;
    let ale = AleMesh::new(gpu, hm, &gm, &rig.rb.points, &rig.csr)?;
    let dt = STROKE_T / steps as Scalar;
    let ctrl = flow_controls(scheme, dt, false);
    let mut s = Simple::new(gpu, hm, &gm, ctrl, buoyancy_off())?;
    seat(
        gpu,
        &mut s,
        hm,
        &gm,
        &[
            Bc::ZeroGradient,
            Bc::ZeroGradient,
            Bc::Symmetry,
            Bc::Symmetry,
            Bc::Symmetry,
            Bc::Symmetry,
        ],
        &[
            Bc::Value(STROKE_P0),
            Bc::Value(0.0),
            Bc::Symmetry,
            Bc::Symmetry,
            Bc::Symmetry,
            Bc::Symmetry,
        ],
        Vec3::new(STROKE_U0, 0.0, 0.0),
    )?;
    s.attach_motion(gpu, ale, &[])?;
    let mut backend = PbicgstabBackend::new(ctrl.p_solver);
    backend.setup(gpu, hm, &gm, &SystemProbe::default())?;
    let (nut, t) = laminar_fields(gpu, hm, &gm)?;

    let n_cells = hm.n_cells;
    let mut min_volume_ratio = Scalar::INFINITY;
    for k in 0..steps {
        take_moving_step(gpu, &mut s, &mut backend, &rig.motion, k, dt, &nut, &t)?;
        let v = gpu.download(&gm.v)?;
        for c in 0..n_cells {
            min_volume_ratio = min_volume_ratio.min(v[c] / hm.v[c]);
        }
    }
    let u = gpu.download(&s.u().f)?;
    let v = gpu.download(&gm.v)?;
    let mut num = 0.0 as Scalar;
    let mut den = 0.0 as Scalar;
    let mut lo = Scalar::INFINITY;
    let mut hi = Scalar::NEG_INFINITY;
    for c in 0..n_cells {
        num += v[c] * u[c].x;
        den += v[c];
        lo = lo.min(u[c].x);
        hi = hi.max(u[c].x);
    }
    Ok(StrokeRun {
        scheme,
        steps,
        dt,
        value: num / den,
        spread: hi - lo,
        min_volume_ratio,
    })
}

/// The exact answer the stroke gate measures against: `U0 + P0` times the
/// integral of `1 / L(s)` over `[0, T]`, composite Simpson to round-off.
pub fn stroke_exact() -> Scalar {
    let n = 20_000usize;
    let h = STROKE_T / n as Scalar;
    let pi = std::f64::consts::PI as Scalar;
    let f = |s: Scalar| 1.0 / (STROKE_L0 + STROKE_A * (2.0 * pi * s / STROKE_PERIOD).sin());
    let mut acc = f(0.0) + f(STROKE_T);
    for i in 1..n {
        let w = if i % 2 == 1 { 4.0 } else { 2.0 };
        acc += w * f(i as Scalar * h);
    }
    STROKE_U0 + STROKE_P0 * acc * h / 3.0
}

/// The stroke at all three step counts with the grid study over them
/// (SPEC-LIT 94.1): the observed order, the finest level against the exact
/// value, and the extrapolation against both.
pub fn stroke_study(gpu: &Gpu, scheme: DdtScheme) -> Result<StrokeStudy> {
    let mut runs = Vec::new();
    for &n in &STROKE_STEPS {
        runs.push(stroke(gpu, scheme, n)?);
    }
    let levels = runs
        .iter()
        .map(|r| Level { h: r.dt, value: r.value })
        .collect::<Vec<_>>();
    let study = vv::grid_study(&levels)?;
    let exact = stroke_exact();
    let p = study.p.unwrap_or(Scalar::NAN);
    let err_fine = ((runs[0].value - exact) / exact).abs();
    let err_ext = ((study.phi_ext - exact) / exact).abs();
    Ok(StrokeStudy {
        scheme,
        runs,
        exact,
        study,
        p,
        err_fine,
        err_ext,
    })
}

/// Gate 105-B for one scheme: the piston in both convective forms and the
/// stroke's time-order study (SPEC-LIT 105.10).
pub fn gate_105b(gpu: &Gpu, scheme: DdtScheme) -> Result<Gate105b> {
    let conservative = piston(gpu, scheme, false)?;
    let bounded = piston(gpu, scheme, true)?;
    let stroke = stroke_study(gpu, scheme)?;
    Ok(Gate105b {
        conservative,
        bounded,
        stroke,
    })
}

/// The order the scheme promises on this gate: 1 for euler, 2 for backward.
pub fn expected_order(s: DdtScheme) -> Scalar {
    match s {
        DdtScheme::Euler => 1.0,
        DdtScheme::Backward => 2.0,
        _ => Scalar::NAN,
    }
}

/// The finest-step tolerance the gate asks for, per scheme: backward's second
/// order buys a hundred times the euler one.
pub fn fine_tolerance(s: DdtScheme) -> Scalar {
    match s {
        DdtScheme::Euler => 5e-4,
        DdtScheme::Backward => 5e-6,
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests;
