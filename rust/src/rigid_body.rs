// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! Six-degree-of-freedom rigid-body motion of the WHOLE mesh, driven by the
//! flow and by jet thrusters - SPEC-LIT section 161.
//!
//! Written from:
//!   ofgpu SPEC-LIT.md 161 (the body, the point map, the integrator, the
//!     wall and jet condition, the force, the refusals and the limits)
//!   ofgpu SPEC-LIT.md 105 (the moving mesh this module drives: `AleMesh`,
//!     `Simple::attach_motion`, `Simple::move_mesh`)
//!   ofgpu SPEC-LIT.md 32.5.1 (the viscous wall traction the force reuses)
//!   H. Goldstein, C. Poole, J. Safko, *Classical Mechanics*, 3rd ed.
//!     (Addison-Wesley, 2002), ch. 4 and 5 - the rigid displacement, the
//!     velocity field of a rigid body and Euler's equations
//!     `I w' + w x (I w) = M` in principal body axes
//!   J. Diebel, *Representing Attitude: Euler Angles, Unit Quaternions, and
//!     Rotation Vectors*, Stanford University technical report (2006) - the
//!     unit quaternion, its rotation matrix, its kinematics
//!     `q' = q (0, w)/2` for a body-frame `w`, and the z-y-x Euler angles
//!   J. C. Butcher, *Numerical Methods for Ordinary Differential Equations*
//!     (Wiley, 2003) - the classical fourth-order Runge-Kutta method
//! No GPL-licensed source was consulted.
//!
//! Provenance: ORIGINAL.
//!
//! # What it is
//!
//! [`RigidBodySpec`] is the `-rigidBody` JSON file; [`RigidBody`] the host
//! integrator (f64 in both builds); [`RigidBodyDevice`] the three device
//! operations of `cuda/rigidbody.cu` - the rigid point map written straight
//! into the [`AleMesh`]'s points, the body wall's velocity with the jets, and
//! the force and moment reduction; [`RigidBodyCoupling`] what a driver calls
//! once per time step. The coupling is LOOSE: the body advances over a step
//! with the force of the step before, extrapolated linearly (SPEC-LIT 161.3).
//!
//! The rest points live on the device once; a time step uploads no point.
//! It downloads twelve numbers (the force reduction) and uploads one small
//! array (the jets' world velocities) - that host round trip is why the
//! capture registry lists this module as refused rather than gated.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use serde::Deserialize;

use crate::device::{cfg_for, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::{BcKind, GpuSurfaceScalarField};
use crate::field_ops::{self, FieldKernels};
use crate::io::polymesh::PolyMeshRaw;
use crate::mesh::ale::AleMesh;
use crate::mesh::gpugeom::{flatten_faces, geo_points, GeoVec3};
use crate::mesh::{GpuMesh, HostMesh, PatchKind};
use crate::simple::Simple;
use crate::types::{DVec3, GeomPoint};
use crate::{DevBuf, Label, Scalar, Vec3};

/// A vector in f64, whatever the build.
pub type V3 = [f64; 3];
/// A 3x3 matrix in f64, row major.
pub type M3 = [[f64; 3]; 3];

/// Threads of the one block `rbForceMoment` runs on - `RB_THREADS` in
/// `cuda/rigidbody.cu`.
const RB_THREADS: u32 = 256;

// ==========================================================================
//  Small f64 vector algebra
// ==========================================================================

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn mat_vec(r: &M3, a: V3) -> V3 {
    [dot(r[0], a), dot(r[1], a), dot(r[2], a)]
}
fn mat_t_vec(r: &M3, a: V3) -> V3 {
    [
        r[0][0] * a[0] + r[1][0] * a[1] + r[2][0] * a[2],
        r[0][1] * a[0] + r[1][1] * a[1] + r[2][1] * a[2],
        r[0][2] * a[0] + r[1][2] * a[1] + r[2][2] * a[2],
    ]
}
fn finite(a: V3) -> bool {
    a.iter().all(|v| v.is_finite())
}

/// The Hamilton product `a b` of two quaternions `(w, x, y, z)`.
fn quat_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// The rotation matrix, body to world, of a unit quaternion `(w, x, y, z)`
/// (SPEC-LIT 161.2).
pub fn quat_to_matrix(q: [f64; 4]) -> M3 {
    let [w, x, y, z] = q;
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

/// The exact rotation by the rotation vector `phi` (angle `|phi|` about its
/// direction), as a unit quaternion.
fn quat_exp(phi: V3) -> [f64; 4] {
    let a = norm(phi);
    if a == 0.0 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let h = 0.5 * a;
    let s = h.sin() / a;
    [h.cos(), phi[0] * s, phi[1] * s, phi[2] * s]
}

// ==========================================================================
//  161.1  The -rigidBody file
// ==========================================================================

/// Which degrees of freedom the body has (SPEC-LIT 161.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum Dof {
    /// All six.
    #[serde(rename = "free")]
    Free,
    /// Translation in x and z, rotation about y: a flip in the x-z plane.
    #[serde(rename = "planarXZ")]
    PlanarXz,
}

impl Dof {
    pub fn name(self) -> &'static str {
        match self {
            Dof::Free => "free",
            Dof::PlanarXz => "planarXZ",
        }
    }
}

fn dof_free() -> Dof {
    Dof::Free
}

/// One thruster: a patch of the body that blows fluid out at `speed`
/// relative to the body, along `direction` in BODY axes, while the time
/// lies in one of the `on` intervals `[t0, t1)`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThrusterSpec {
    pub patch: String,
    pub direction: V3,
    pub speed: f64,
    #[serde(default)]
    pub on: Vec<[f64; 2]>,
}

impl ThrusterSpec {
    /// Is the jet on at time `t`? `t0 <= t < t1` for some interval.
    pub fn is_on(&self, t: f64) -> bool {
        self.on.iter().any(|iv| iv[0] <= t && t < iv[1])
    }

    /// The unit exit direction in body axes.
    pub fn unit_direction(&self) -> V3 {
        let n = norm(self.direction);
        if n > 0.0 {
            scale(self.direction, 1.0 / n)
        } else {
            [0.0; 3]
        }
    }
}

fn write_every_default() -> usize {
    1
}
fn write_file_default() -> String {
    "rigidBody.csv".to_string()
}

/// The CSV the run writes: every `every` steps, to `file` (relative to the
/// case's output root).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WriteSpec {
    #[serde(default = "write_every_default")]
    pub every: usize,
    #[serde(default = "write_file_default")]
    pub file: String,
}

impl Default for WriteSpec {
    fn default() -> Self {
        Self { every: write_every_default(), file: write_file_default() }
    }
}

/// The `-rigidBody` file - SPEC-LIT 161.1. `inertia` is the principal
/// moments about the centre of mass in body axes, which are the mesh axes
/// at `t = 0`; `velocity` and `angularVelocity` are the initial values in
/// those same axes; `rho` turns the kinematic pressure and viscosity into
/// newtons.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct RigidBodySpec {
    pub mass: f64,
    pub inertia: V3,
    pub centre_of_mass: V3,
    #[serde(default)]
    pub velocity: V3,
    #[serde(default)]
    pub angular_velocity: V3,
    #[serde(default)]
    pub gravity: V3,
    #[serde(default = "dof_free")]
    pub dof: Dof,
    pub body_patches: Vec<String>,
    pub rho: f64,
    #[serde(default)]
    pub thrusters: Vec<ThrusterSpec>,
    #[serde(default)]
    pub write: WriteSpec,
}

impl RigidBodySpec {
    /// Parse and validate the file's text; `origin` names it in messages.
    pub fn parse(text: &str, origin: &str) -> Result<Self> {
        let spec: RigidBodySpec = serde_json::from_str(text).map_err(|e| Error::Parse {
            path: origin.to_string(),
            msg: format!("-rigidBody: {e} (SPEC-LIT 161.1)"),
        })?;
        spec.validate()?;
        Ok(spec)
    }

    /// Read, parse and validate the file.
    pub fn read(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .map_err(|source| Error::Io { path: path.display().to_string(), source })?;
        Self::parse(&text, &path.display().to_string())
    }

    /// Everything the integrator and the coupling assume, refused by name.
    pub fn validate(&self) -> Result<()> {
        let bad = |what: String| -> Result<()> { Err(Error::Config(format!("-rigidBody: {what} (SPEC-LIT 161.1)"))) };
        if !(self.mass > 0.0 && self.mass.is_finite()) {
            return bad(format!("mass is {}; it must be positive", self.mass));
        }
        if !self.inertia.iter().all(|&i| i > 0.0 && i.is_finite()) {
            return bad(format!("inertia {:?} must be three positive principal moments", self.inertia));
        }
        let [a, b, c] = self.inertia;
        let tol = 1e-12 * (a + b + c);
        if a + b < c - tol || b + c < a - tol || a + c < b - tol {
            return bad(format!(
                "inertia {:?} breaks the triangle inequality every rigid body's principal moments obey",
                self.inertia
            ));
        }
        if !(finite(self.centre_of_mass) && finite(self.velocity) && finite(self.angular_velocity) && finite(self.gravity)) {
            return bad("centreOfMass, velocity, angularVelocity and gravity must be finite".to_string());
        }
        if !(self.rho > 0.0 && self.rho.is_finite()) {
            return bad(format!("rho is {}; it must be positive", self.rho));
        }
        if self.body_patches.is_empty() {
            return bad("bodyPatches is empty; the force has no faces to be summed over".to_string());
        }
        for (i, p) in self.body_patches.iter().enumerate() {
            if self.body_patches[..i].contains(p) {
                return bad(format!("bodyPatches names {p} twice"));
            }
        }
        if self.dof == Dof::PlanarXz {
            if self.velocity[1] != 0.0 {
                return bad(format!("dof planarXZ with a y velocity {}; the plane has none", self.velocity[1]));
            }
            if self.angular_velocity[0] != 0.0 || self.angular_velocity[2] != 0.0 {
                return bad(format!(
                    "dof planarXZ with an angular velocity {:?}; the plane turns about y only",
                    self.angular_velocity
                ));
            }
        }
        for (i, th) in self.thrusters.iter().enumerate() {
            if !self.body_patches.contains(&th.patch) {
                return bad(format!("thruster {} names patch {}, which bodyPatches does not list", i, th.patch));
            }
            if self.thrusters[..i].iter().any(|o| o.patch == th.patch) {
                return bad(format!("two thrusters name patch {}", th.patch));
            }
            if !(finite(th.direction) && norm(th.direction) > 0.0) {
                return bad(format!("thruster {} has direction {:?}; it needs a non-zero one", th.patch, th.direction));
            }
            if !(th.speed >= 0.0 && th.speed.is_finite()) {
                return bad(format!("thruster {} has speed {}; it must be zero or positive", th.patch, th.speed));
            }
            for iv in &th.on {
                if !(iv[0].is_finite() && iv[1].is_finite() && iv[0] < iv[1]) {
                    return bad(format!("thruster {} has the interval {:?}; it needs t0 < t1", th.patch, iv));
                }
            }
        }
        if self.write.every == 0 {
            return bad("write.every is 0; it counts steps and must be at least 1".to_string());
        }
        Ok(())
    }
}

// ==========================================================================
//  161.3  The integrator
// ==========================================================================

/// The body's state: the centre of mass `x` and its velocity `v` (world),
/// the orientation `q` (unit quaternion, body to world, scalar first), the
/// angular velocity `w` in BODY axes, the time, and `rot_y`, the rotation
/// about world `y` integrated without wrapping (radians).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyState {
    pub t: f64,
    pub x: V3,
    pub v: V3,
    pub q: [f64; 4],
    pub w: V3,
    pub rot_y: f64,
}

/// The host integrator of SPEC-LIT 161.3, f64 in both builds.
#[derive(Debug, Clone)]
pub struct RigidBody {
    pub mass: f64,
    /// Principal moments about the centre of mass, body axes.
    pub inertia: V3,
    pub gravity: V3,
    pub dof: Dof,
    /// The centre of mass at `t = 0`, which the rest points are measured
    /// from in the point map.
    pub c0: V3,
    pub state: BodyState,
    /// The previous step's linear acceleration (world) and moment (body),
    /// the second point of the linear extrapolation. `None` before the
    /// first step: the first step holds the force constant.
    prev: Option<(V3, V3)>,
}

impl RigidBody {
    /// The body of a validated spec, at rest orientation.
    pub fn new(spec: &RigidBodySpec) -> Result<Self> {
        spec.validate()?;
        Ok(Self::from_parts(
            spec.mass,
            spec.inertia,
            spec.centre_of_mass,
            spec.velocity,
            spec.angular_velocity,
            spec.gravity,
            spec.dof,
        ))
    }

    /// A body from its parts, unvalidated - the tests' constructor.
    pub fn from_parts(mass: f64, inertia: V3, c0: V3, v0: V3, w0: V3, gravity: V3, dof: Dof) -> Self {
        Self {
            mass,
            inertia,
            gravity,
            dof,
            c0,
            state: BodyState { t: 0.0, x: c0, v: v0, q: [1.0, 0.0, 0.0, 0.0], w: w0, rot_y: 0.0 },
            prev: None,
        }
    }

    /// `R(q)`, body to world.
    pub fn rotation(&self) -> M3 {
        quat_to_matrix(self.state.q)
    }

    /// The angular velocity in world axes, `R w`.
    pub fn omega_world(&self) -> V3 {
        mat_vec(&self.rotation(), self.state.w)
    }

    /// The point map `x(t) = X(t) + R(q) (x0 - c0)` - SPEC-LIT 161.2.
    pub fn map_point(&self, x0: V3) -> V3 {
        add(self.state.x, mat_vec(&self.rotation(), sub(x0, self.c0)))
    }

    /// The rigid velocity at a world point, `V + (R w) x (x - X)`.
    pub fn point_velocity(&self, x: V3) -> V3 {
        add(self.state.v, cross(self.omega_world(), sub(x, self.state.x)))
    }

    /// Roll, pitch and yaw in degrees: the z-y-x Euler angles of `R`
    /// (`R = Rz(yaw) Ry(pitch) Rx(roll)`). Pitch is `asin`, so it folds back
    /// at +-90 degrees; `rot_y` is the unwrapped angle about world `y`.
    pub fn euler_zyx_deg(&self) -> V3 {
        let r = self.rotation();
        let roll = r[2][1].atan2(r[2][2]);
        let pitch = (-r[2][0]).clamp(-1.0, 1.0).asin();
        let yaw = r[1][0].atan2(r[0][0]);
        [roll.to_degrees(), pitch.to_degrees(), yaw.to_degrees()]
    }

    /// `(m |V|^2 + w . I w)/2`.
    pub fn kinetic_energy(&self) -> f64 {
        let s = &self.state;
        let iw = [self.inertia[0] * s.w[0], self.inertia[1] * s.w[1], self.inertia[2] * s.w[2]];
        0.5 * (self.mass * dot(s.v, s.v) + dot(s.w, iw))
    }

    /// The angular momentum about the centre of mass in world axes, `R I w`.
    pub fn angular_momentum_world(&self) -> V3 {
        let s = &self.state;
        let iw = [self.inertia[0] * s.w[0], self.inertia[1] * s.w[1], self.inertia[2] * s.w[2]];
        mat_vec(&self.rotation(), iw)
    }

    /// The force with the components the degrees of freedom forbid removed.
    fn constrain_force(&self, f: V3) -> V3 {
        match self.dof {
            Dof::Free => f,
            Dof::PlanarXz => [f[0], 0.0, f[2]],
        }
    }

    /// The world moment with the components the degrees of freedom forbid
    /// removed.
    fn constrain_moment(&self, m: V3) -> V3 {
        match self.dof {
            Dof::Free => m,
            Dof::PlanarXz => [0.0, m[1], 0.0],
        }
    }

    /// `dw/dt` in body axes: `I^-1 (M - w x I w)`, with the planar
    /// constraint's x and z components held at zero.
    fn omega_rate(&self, m_body: V3, w: V3) -> V3 {
        let i = self.inertia;
        let iw = [i[0] * w[0], i[1] * w[1], i[2] * w[2]];
        let g = cross(w, iw);
        let mut d = [(m_body[0] - g[0]) / i[0], (m_body[1] - g[1]) / i[1], (m_body[2] - g[2]) / i[2]];
        if self.dof == Dof::PlanarXz {
            d[0] = 0.0;
            d[2] = 0.0;
        }
        d
    }

    /// One classical RK4 step of `dw/dt` from `tau` to `tau + h`, the moment
    /// `m0 + slope tau` linear in time.
    fn rk4_omega(&self, m0: V3, slope: V3, tau: f64, w: V3, h: f64) -> V3 {
        let m_at = |s: f64| add(m0, scale(slope, s));
        let k1 = self.omega_rate(m_at(tau), w);
        let k2 = self.omega_rate(m_at(tau + 0.5 * h), add(w, scale(k1, 0.5 * h)));
        let k3 = self.omega_rate(m_at(tau + 0.5 * h), add(w, scale(k2, 0.5 * h)));
        let k4 = self.omega_rate(m_at(tau + h), add(w, scale(k3, h)));
        add(w, scale(add(add(k1, scale(add(k2, k3), 2.0)), k4), h / 6.0))
    }

    /// Advance the body by `dt` under the fluid's force and moment about the
    /// centre of mass, both in world axes, measured at the start of the step
    /// - SPEC-LIT 161.3.
    ///
    /// Translation integrates the acceleration extrapolated linearly from
    /// this step's and the last step's values EXACTLY (the second-order
    /// Adams-Bashforth velocity and its exact position). Rotation integrates
    /// Euler's equations with the moment extrapolated the same way, by two
    /// classical RK4 half steps, and turns `q` by the exact rotation of the
    /// Simpson mean of `w` over the step. Gravity is added to the
    /// acceleration; the constraint removes what the degrees of freedom
    /// forbid.
    pub fn advance(&mut self, dt: f64, force: V3, moment: V3) {
        let f = self.constrain_force(force);
        let m_world = self.constrain_moment(moment);
        let mut a = add(scale(f, 1.0 / self.mass), self.gravity);
        if self.dof == Dof::PlanarXz {
            a[1] = 0.0;
        }
        let r = self.rotation();
        let m_body = mat_t_vec(&r, m_world);
        let (a_prev, m_prev) = self.prev.unwrap_or((a, m_body));

        // Translation: a(s) = a + (a - a_prev) s/dt, integrated exactly.
        let da = sub(a, a_prev);
        let s = &mut self.state;
        let x_new = add(
            add(s.x, scale(s.v, dt)),
            add(scale(a, 0.5 * dt * dt), scale(da, dt * dt / 6.0)),
        );
        let v_new = add(s.v, add(scale(a, dt), scale(da, 0.5 * dt)));
        s.x = x_new;
        s.v = v_new;

        // Rotation: Euler's equations, RK4 in two half steps.
        let slope = scale(sub(m_body, m_prev), 1.0 / dt);
        let w0 = self.state.w;
        let h = 0.5 * dt;
        let w_mid = self.rk4_omega(m_body, slope, 0.0, w0, h);
        let w_end = self.rk4_omega(m_body, slope, h, w_mid, h);
        let omega = scale(add(add(w0, scale(w_mid, 4.0)), w_end), 1.0 / 6.0);
        let mut q = quat_mul(self.state.q, quat_exp(scale(omega, dt)));
        let qn = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        for c in &mut q {
            *c /= qn;
        }
        let s = &mut self.state;
        s.q = q;
        s.w = w_end;
        s.rot_y += mat_vec(&quat_to_matrix(q), omega)[1] * dt;
        s.t += dt;
        self.prev = Some((a, m_body));
    }
}

// ==========================================================================
//  161.2, 161.4, 161.5  The device side
// ==========================================================================

/// The four kernels of `cuda/rigidbody.cu`.
pub struct RigidBodyKernels {
    pub rigid_points: CudaFunction,
    pub wall_velocity: CudaFunction,
    pub scatter_flux: CudaFunction,
    pub force_moment: CudaFunction,
}

impl RigidBodyKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let ks = KernelSet::new(gpu, crate::kernels::RIGIDBODY)?;
        Ok(Self {
            rigid_points: ks.func("rbRigidPoints")?,
            wall_velocity: ks.func("rbWallVelocity")?,
            scatter_flux: ks.func("rbScatterFlux")?,
            force_moment: ks.func("rbForceMoment")?,
        })
    }
}

/// The force and moment on the body, world axes, newtons and newton metres
/// (SPEC-LIT 161.5). The moment is about the centre of mass.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Wrench {
    pub pressure: V3,
    pub viscous: V3,
    pub jet: V3,
    pub moment: V3,
}

impl Wrench {
    pub fn force(&self) -> V3 {
        add(add(self.pressure, self.viscous), self.jet)
    }
}

/// The rest points, the body faces and the scratch the three device
/// operations need, resident.
pub struct RigidBodyDevice {
    pub n_points: usize,
    pub n_faces: usize,
    pub n_jets: usize,
    /// The world position of the mesh's local origin (SPEC-LIT 118.7).
    pub origin: V3,
    rest: DevBuf<GeoVec3>,
    faces: DevBuf<Label>,
    face_jet: DevBuf<Label>,
    jet_vel: DevBuf<Vec3>,
    phi_mesh_b: DevBuf<Scalar>,
    wall_flux: DevBuf<Scalar>,
    out: DevBuf<f64>,
    k: RigidBodyKernels,
}

fn dvec(p: DVec3) -> V3 {
    [p.x, p.y, p.z]
}

fn upload_labels(gpu: &Gpu, v: &[Label]) -> Result<DevBuf<Label>> {
    if v.is_empty() {
        gpu.zeros(1)
    } else {
        gpu.upload(v)
    }
}

impl RigidBodyDevice {
    /// `rest_world` the rest points, `origin` the mesh's local origin,
    /// `body_faces` boundary face ids, `face_jet` one thruster index (or -1)
    /// per body face, `n_bf` the mesh's boundary face count.
    pub fn new<P: GeomPoint>(
        gpu: &Gpu,
        rest_world: &[P],
        origin: DVec3,
        body_faces: &[Label],
        face_jet: &[Label],
        n_jets: usize,
        n_bf: usize,
    ) -> Result<Self> {
        if face_jet.len() != body_faces.len() {
            return Err(Error::Config(format!(
                "RigidBodyDevice::new: {} body faces and {} jet labels",
                body_faces.len(),
                face_jet.len()
            )));
        }
        if let Some(&f) = body_faces.iter().find(|&&f| f < 0 || f as usize >= n_bf) {
            return Err(Error::Config(format!(
                "RigidBodyDevice::new: body face {f} is not one of the mesh's {n_bf} boundary faces"
            )));
        }
        if let Some(&j) = face_jet.iter().find(|&&j| j >= n_jets as Label || j < -1) {
            return Err(Error::Config(format!("RigidBodyDevice::new: jet label {j} for {n_jets} thrusters")));
        }
        let geo = geo_points(rest_world, origin);
        Ok(Self {
            n_points: rest_world.len(),
            n_faces: body_faces.len(),
            n_jets,
            origin: dvec(origin),
            rest: if geo.is_empty() { gpu.zeros(1)? } else { gpu.upload(&geo)? },
            faces: upload_labels(gpu, body_faces)?,
            face_jet: upload_labels(gpu, face_jet)?,
            jet_vel: gpu.zeros(n_jets.max(1))?,
            phi_mesh_b: gpu.zeros(n_bf.max(1))?,
            wall_flux: gpu.zeros(body_faces.len().max(1))?,
            out: gpu.zeros(12)?,
            k: RigidBodyKernels::new(gpu)?,
        })
    }

    /// The centre of mass and its rest position in the mesh's local frame.
    fn local(&self, body: &RigidBody) -> (V3, V3) {
        (sub(body.state.x, self.origin), sub(body.c0, self.origin))
    }

    /// Write the body's points into `points` (an [`AleMesh`]'s, in its
    /// local frame) on the device - SPEC-LIT 161.2. No point is uploaded.
    pub fn write_points(&self, gpu: &Gpu, points: &mut DevBuf<GeoVec3>, body: &RigidBody) -> Result<()> {
        if points.len() < self.n_points {
            return Err(Error::Config(format!(
                "RigidBodyDevice::write_points: {} points to write into a buffer of {}",
                self.n_points,
                points.len()
            )));
        }
        if self.n_points == 0 {
            return Ok(());
        }
        let r = body.rotation();
        let (xl, cl) = self.local(body);
        let n = self.n_points as Label;
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.rigid_points)
                .arg(&mut *points)
                .arg(&self.rest)
                .arg(&r[0][0])
                .arg(&r[0][1])
                .arg(&r[0][2])
                .arg(&r[1][0])
                .arg(&r[1][1])
                .arg(&r[1][2])
                .arg(&r[2][0])
                .arg(&r[2][1])
                .arg(&r[2][2])
                .arg(&xl[0])
                .arg(&xl[1])
                .arg(&xl[2])
                .arg(&cl[0])
                .arg(&cl[1])
                .arg(&cl[2])
                .arg(&n)
                .launch(cfg_for(self.n_points))?;
        }
        Ok(())
    }

    /// Copy the mesh flux's boundary half - what [`Self::wall_velocity`]
    /// reads - into this struct's own buffer.
    pub fn stage_mesh_flux(&mut self, gpu: &Gpu, phi_mesh_b: &DevBuf<Scalar>) -> Result<()> {
        if phi_mesh_b.len() > self.phi_mesh_b.len() {
            return Err(Error::Config("RigidBodyDevice::stage_mesh_flux: the mesh flux is longer than the mesh".to_string()));
        }
        let mut dst = self.phi_mesh_b.slice_mut(0..phi_mesh_b.len());
        gpu.stream().memcpy_dtod(phi_mesh_b, &mut dst)?;
        Ok(())
    }

    /// The body wall's velocity into `ref_value` - SPEC-LIT 161.4. `jets` is
    /// one world velocity per thruster (zero while it is off); `full_rigid`
    /// is the t = 0 seed, before the mesh has a mesh flux.
    pub fn wall_velocity(
        &mut self,
        gpu: &Gpu,
        gm: &GpuMesh,
        ref_value: &mut DevBuf<Vec3>,
        body: &RigidBody,
        jets: &[V3],
        full_rigid: bool,
    ) -> Result<()> {
        if jets.len() != self.n_jets {
            return Err(Error::Config(format!(
                "RigidBodyDevice::wall_velocity: {} jet velocities for {} thrusters",
                jets.len(),
                self.n_jets
            )));
        }
        if self.n_faces == 0 {
            return Ok(());
        }
        if self.n_jets > 0 {
            let jv: Vec<Vec3> =
                jets.iter().map(|j| Vec3::new(j[0] as Scalar, j[1] as Scalar, j[2] as Scalar)).collect();
            gpu.write(&mut self.jet_vel, &jv)?;
        }
        let v = body.state.v;
        let w = body.omega_world();
        let (xl, _) = self.local(body);
        let full: i32 = i32::from(full_rigid);
        let n = self.n_faces as Label;
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.wall_velocity)
                .arg(&mut *ref_value)
                .arg(&mut self.wall_flux)
                .arg(&self.phi_mesh_b)
                .arg(&gm.b_sf)
                .arg(&gm.b_cf)
                .arg(&self.faces)
                .arg(&self.face_jet)
                .arg(&self.jet_vel)
                .arg(&v[0])
                .arg(&v[1])
                .arg(&v[2])
                .arg(&w[0])
                .arg(&w[1])
                .arg(&w[2])
                .arg(&xl[0])
                .arg(&xl[1])
                .arg(&xl[2])
                .arg(&full)
                .arg(&n)
                .launch(cfg_for(self.n_faces))?;
        }
        Ok(())
    }

    /// Write the flux [`Self::wall_velocity`] computed into the body faces of
    /// `phi_b`.
    pub fn scatter_flux(&self, gpu: &Gpu, phi_b: &mut DevBuf<Scalar>) -> Result<()> {
        if self.n_faces == 0 {
            return Ok(());
        }
        let n = self.n_faces as Label;
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.scatter_flux)
                .arg(&mut *phi_b)
                .arg(&self.wall_flux)
                .arg(&self.faces)
                .arg(&n)
                .launch(cfg_for(self.n_faces))?;
        }
        Ok(())
    }

    /// The force and moment on the body - SPEC-LIT 161.5. `p_b` the KINEMATIC
    /// pressure's boundary values, `u`/`u_b` the velocity's cell and
    /// boundary values, `phi_rel_b` the relative flux's boundary half, `rho`
    /// and `nu` the density and kinematic viscosity, `g_hyd` the gravity
    /// whose hydrostatic part `p` leaves out. One block, twelve numbers
    /// downloaded.
    #[allow(clippy::too_many_arguments)]
    pub fn force_moment(
        &mut self,
        gpu: &Gpu,
        gm: &GpuMesh,
        p_b: &DevBuf<Scalar>,
        u: &DevBuf<Vec3>,
        u_b: &DevBuf<Vec3>,
        phi_rel_b: &DevBuf<Scalar>,
        rho: f64,
        nu: f64,
        g_hyd: V3,
        body: &RigidBody,
    ) -> Result<Wrench> {
        let v = body.state.v;
        let w = body.omega_world();
        let (xl, _) = self.local(body);
        let n = self.n_faces as Label;
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (RB_THREADS, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.force_moment)
                .arg(&mut self.out)
                .arg(&self.faces)
                .arg(p_b)
                .arg(u)
                .arg(u_b)
                .arg(phi_rel_b)
                .arg(&gm.b_face_cells)
                .arg(&gm.b_sf)
                .arg(&gm.b_cf)
                .arg(&gm.b_delta_coeffs)
                .arg(&rho)
                .arg(&nu)
                .arg(&g_hyd[0])
                .arg(&g_hyd[1])
                .arg(&g_hyd[2])
                .arg(&v[0])
                .arg(&v[1])
                .arg(&v[2])
                .arg(&w[0])
                .arg(&w[1])
                .arg(&w[2])
                .arg(&xl[0])
                .arg(&xl[1])
                .arg(&xl[2])
                .arg(&n)
                .launch(cfg)?;
        }
        let o = gpu.download(&self.out)?;
        Ok(Wrench {
            pressure: [o[0], o[1], o[2]],
            viscous: [o[3], o[4], o[5]],
            jet: [o[6], o[7], o[8]],
            moment: [o[9], o[10], o[11]],
        })
    }
}

// ==========================================================================
//  161.6  The coupling a driver calls once per step
// ==========================================================================

/// The rigid body attached to one `Simple`: the body, its device state, the
/// last force, the CSV and the progress clock.
pub struct RigidBodyCoupling {
    pub body: RigidBody,
    pub thrusters: Vec<ThrusterSpec>,
    pub wrench: Wrench,
    pub csv_path: PathBuf,
    pub n_body_faces: usize,
    dev: RigidBodyDevice,
    fldk: FieldKernels,
    rho: f64,
    nu: f64,
    g_hyd: V3,
    rest_world: Vec<V3>,
    csv: BufWriter<File>,
    every: usize,
    jets_on: Vec<bool>,
    clock: Instant,
    clock_step: usize,
}

/// The body faces and their thruster labels, from the spec's patch names.
pub fn body_faces(hm: &HostMesh, spec: &RigidBodySpec) -> Result<(Vec<Label>, Vec<Label>)> {
    let mut faces = Vec::new();
    let mut jets = Vec::new();
    for name in &spec.body_patches {
        let p = hm.patches.iter().find(|p| &p.name == name).ok_or_else(|| {
            Error::Config(format!(
                "-rigidBody: bodyPatches names {name}, which this mesh does not have; its patches are {} (SPEC-LIT 161.1)",
                hm.patches.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(", ")
            ))
        })?;
        if matches!(p.kind, PatchKind::Cyclic | PatchKind::Empty | PatchKind::Symmetry) {
            return Err(Error::Config(format!(
                "-rigidBody: body patch {name} is a {:?} patch; a body face is a wall or a jet (SPEC-LIT 161.7)",
                p.kind
            )));
        }
        let jet = spec.thrusters.iter().position(|t| &t.patch == name).map_or(-1, |j| j as Label);
        for k in 0..p.size {
            faces.push((p.start + k) as Label);
            jets.push(jet);
        }
    }
    Ok((faces, jets))
}

impl RigidBodyCoupling {
    /// Build the moving mesh from the case's own polyMesh, seed the body
    /// walls' velocity and the flux from it, attach the motion to `s`, and
    /// measure the force the first step will use - SPEC-LIT 161.6.
    ///
    /// Call it once `U`, `p` and `phi` are seated and `s` is initialised.
    #[allow(clippy::too_many_arguments)]
    pub fn attach(
        gpu: &Gpu,
        hm: &HostMesh,
        gm: &GpuMesh,
        raw: &PolyMeshRaw,
        spec: &RigidBodySpec,
        s: &mut Simple<'_>,
        nu: f64,
        g_hyd: V3,
        csv_path: &Path,
    ) -> Result<Self> {
        let body = RigidBody::new(spec)?;
        if (hm.n_points != 0 && raw.points.len() != hm.n_points)
            || raw.faces.len() != hm.n_internal_faces + hm.n_boundary_faces
        {
            return Err(Error::Mesh(format!(
                "-rigidBody: the polyMesh has {} points and {} faces, and the mesh being solved {} and {}",
                raw.points.len(),
                raw.faces.len(),
                hm.n_points,
                hm.n_internal_faces + hm.n_boundary_faces
            )));
        }
        let (faces, face_jet) = body_faces(hm, spec)?;

        // Every body face must carry a fixed velocity: that is what the wall
        // and jet condition writes (SPEC-LIT 161.4).
        let kinds = gpu.download(&s.u().bc_kind)?;
        let mut bad: Vec<String> = Vec::new();
        for p in &hm.patches {
            if !spec.body_patches.contains(&p.name) {
                continue;
            }
            if (0..p.size).any(|k| kinds[p.start + k] != BcKind::FixedValue as Label) {
                bad.push(p.name.clone());
            }
        }
        if !bad.is_empty() {
            return Err(Error::Config(format!(
                "-rigidBody: body patch(es) {} do not carry a fixedValue U; the body wall's and the jets' \
                 velocity is written as a fixed value - give them `type fixedValue; value uniform (0 0 0);` \
                 in 0/U (SPEC-LIT 161.4)",
                bad.join(", ")
            )));
        }

        let mut dev = RigidBodyDevice::new(
            gpu,
            &raw.points,
            hm.origin,
            &faces,
            &face_jet,
            spec.thrusters.len(),
            hm.n_boundary_faces,
        )?;
        let fldk = FieldKernels::new(gpu)?;

        // The t = 0 seed: the whole rigid velocity on the body faces, the jets
        // that are on at t = 0, and the flux built from that U, before the
        // motion is attached (whose relative flux is then phi itself).
        let r = body.rotation();
        let jets: Vec<V3> = spec
            .thrusters
            .iter()
            .map(|t| if t.is_on(0.0) { scale(mat_vec(&r, t.unit_direction()), t.speed) } else { [0.0; 3] })
            .collect();
        dev.wall_velocity(gpu, gm, &mut s.u_mut().ref_value, &body, &jets, true)?;
        field_ops::correct_boundary_conditions_vector(gpu, &fldk, s.u_mut(), gm)?;
        let mut phi = GpuSurfaceScalarField::zeros(gpu, gm, "phi")?;
        crate::field_setup::compute_phi_from_u(gpu, &mut phi, s.u(), hm)?;
        *s.phi_mut() = phi;

        let csr = flatten_faces(&raw.faces);
        let ale = AleMesh::new(gpu, hm, gm, &raw.points, &csr)?;
        s.attach_motion(gpu, ale, &faces)?;

        let rho = spec.rho;
        let wrench = dev.force_moment(
            gpu,
            gm,
            &s.p().bf,
            &s.u().f,
            &s.u().bf,
            &s.convective_flux().bf,
            rho,
            nu,
            g_hyd,
            &body,
        )?;

        if let Some(dir) = csv_path.parent() {
            if !dir.as_os_str().is_empty() {
                fs::create_dir_all(dir).map_err(|source| Error::Io { path: dir.display().to_string(), source })?;
            }
        }
        let file = File::create(csv_path).map_err(|source| Error::Io { path: csv_path.display().to_string(), source })?;
        let mut csv = BufWriter::new(file);
        let names: Vec<&str> = spec.thrusters.iter().map(|t| t.patch.as_str()).collect();
        writeln!(
            csv,
            "step,time,X,Y,Z,Vx,Vy,Vz,roll_deg,pitch_deg,yaw_deg,rotY_deg,wx_body,wy_body,wz_body,\
             Fx,Fy,Fz,Mx,My,Mz,Fpx,Fpy,Fpz,Fvx,Fvy,Fvz,Fjx,Fjy,Fjz,thrusters_on[{}]",
            names.join("|")
        )
        .map_err(|source| Error::Io { path: csv_path.display().to_string(), source })?;

        let rest_world: Vec<V3> = raw.points.iter().map(|p| dvec(p.to_dvec3())).collect();
        println!(
            "rigid body (SPEC-LIT 161): mass {} kg, inertia ({} {} {}) kg m2, dof {}, centre of mass ({} {} {}), \
             V0 ({} {} {}), w0 ({} {} {}), gravity ({} {} {})",
            spec.mass,
            spec.inertia[0],
            spec.inertia[1],
            spec.inertia[2],
            spec.dof.name(),
            spec.centre_of_mass[0],
            spec.centre_of_mass[1],
            spec.centre_of_mass[2],
            spec.velocity[0],
            spec.velocity[1],
            spec.velocity[2],
            spec.angular_velocity[0],
            spec.angular_velocity[1],
            spec.angular_velocity[2],
            spec.gravity[0],
            spec.gravity[1],
            spec.gravity[2],
        );
        println!(
            "  {} body faces on {} patch(es) [{}], {} thruster(s) [{}]; rho {} kg/m3 converts the kinematic p, \
             mu = rho nu = {} Pa s; the fluid's hydrostatic part g = ({} {} {}) is added back to p",
            faces.len(),
            spec.body_patches.len(),
            spec.body_patches.join(", "),
            spec.thrusters.len(),
            names.join(", "),
            rho,
            rho * nu,
            g_hyd[0],
            g_hyd[1],
            g_hyd[2],
        );
        println!(
            "  t = 0 force ({:.4e} {:.4e} {:.4e}) N, moment ({:.4e} {:.4e} {:.4e}) N m; CSV {} every {} step(s)",
            wrench.force()[0],
            wrench.force()[1],
            wrench.force()[2],
            wrench.moment[0],
            wrench.moment[1],
            wrench.moment[2],
            csv_path.display(),
            spec.write.every,
        );

        Ok(Self {
            n_body_faces: faces.len(),
            jets_on: vec![false; spec.thrusters.len()],
            thrusters: spec.thrusters.clone(),
            body,
            wrench,
            csv_path: csv_path.to_path_buf(),
            dev,
            fldk,
            rho,
            nu,
            g_hyd,
            rest_world,
            csv,
            every: spec.write.every,
            clock: Instant::now(),
            clock_step: 0,
        })
    }

    /// Before the step's solve, after the time step is opened: advance the
    /// body over `dt` with the last force, move the points on the device,
    /// advance the mesh, and write the body walls' and jets' velocity and
    /// flux - SPEC-LIT 161.6. `t_new` is the step's end time.
    pub fn pre_step(&mut self, gpu: &Gpu, gm: &GpuMesh, s: &mut Simple<'_>, t_new: f64, dt: f64) -> Result<()> {
        self.body.advance(dt, self.wrench.force(), self.wrench.moment);
        if !(finite(self.body.state.x) && finite(self.body.state.w) && self.body.state.q.iter().all(|v| v.is_finite())) {
            return Err(Error::Diverged {
                iteration: 0,
                what: format!("the rigid body's state went non-finite at t = {t_new}"),
            });
        }
        {
            let ale = s.motion_mut().ok_or_else(|| {
                Error::Config("RigidBodyCoupling::pre_step: no motion is attached".to_string())
            })?;
            self.dev.write_points(gpu, &mut ale.points, &self.body)?;
        }
        s.move_mesh(gpu)?;
        {
            let ale = s.motion().ok_or_else(|| {
                Error::Config("RigidBodyCoupling::pre_step: no motion is attached".to_string())
            })?;
            self.dev.stage_mesh_flux(gpu, &ale.phi_mesh.bf)?;
        }
        let t_mid = t_new - 0.5 * dt;
        let r = self.body.rotation();
        let mut jets = Vec::with_capacity(self.thrusters.len());
        for (i, t) in self.thrusters.iter().enumerate() {
            let on = t.is_on(t_mid);
            self.jets_on[i] = on;
            jets.push(if on { scale(mat_vec(&r, t.unit_direction()), t.speed) } else { [0.0; 3] });
        }
        self.dev.wall_velocity(gpu, gm, &mut s.u_mut().ref_value, &self.body, &jets, false)?;
        field_ops::correct_boundary_conditions_vector(gpu, &self.fldk, s.u_mut(), gm)?;
        self.dev.scatter_flux(gpu, &mut s.phi_mut().bf)
    }

    /// After the step's solve: the force and moment the next step will use,
    /// the CSV row and the progress line every `write.every` steps.
    pub fn post_step(
        &mut self,
        gpu: &Gpu,
        gm: &GpuMesh,
        s: &Simple<'_>,
        step: usize,
        n_steps: usize,
        end_time: f64,
    ) -> Result<()> {
        self.wrench = self.dev.force_moment(
            gpu,
            gm,
            &s.p().bf,
            &s.u().f,
            &s.u().bf,
            &s.convective_flux().bf,
            self.rho,
            self.nu,
            self.g_hyd,
            &self.body,
        )?;
        let last = step + 1 == n_steps;
        if step % self.every == 0 || last {
            self.write_row(step)?;
            let st = &self.body.state;
            let e = self.body.euler_zyx_deg();
            let f = self.wrench.force();
            let w = self.body.omega_world();
            let steps = (step + 1).saturating_sub(self.clock_step).max(1);
            let per_step = self.clock.elapsed().as_secs_f64() / steps as f64;
            self.clock = Instant::now();
            self.clock_step = step + 1;
            let pct = if end_time > 0.0 { 100.0 * st.t / end_time } else { 0.0 };
            println!(
                "rigidBody {:>5.1}% step {}/{} t {:.5} | X ({:.4} {:.4} {:.4}) V ({:.3} {:.3} {:.3}) | \
                 rotY {:.2} deg rpy ({:.1} {:.1} {:.1}) w ({:.3} {:.3} {:.3}) rad/s | F ({:.3e} {:.3e} {:.3e}) \
                 My {:.3e} | jets {} | {:.3} s/step",
                pct,
                step + 1,
                n_steps,
                st.t,
                st.x[0],
                st.x[1],
                st.x[2],
                st.v[0],
                st.v[1],
                st.v[2],
                st.rot_y.to_degrees(),
                e[0],
                e[1],
                e[2],
                w[0],
                w[1],
                w[2],
                f[0],
                f[1],
                f[2],
                self.wrench.moment[1],
                self.jets_string(),
                per_step,
            );
        }
        if last {
            self.csv.flush().map_err(|source| Error::Io { path: self.csv_path.display().to_string(), source })?;
        }
        Ok(())
    }

    fn jets_string(&self) -> String {
        if self.jets_on.is_empty() {
            return "-".to_string();
        }
        self.jets_on.iter().map(|&o| if o { '1' } else { '0' }).collect()
    }

    fn write_row(&mut self, step: usize) -> Result<()> {
        let st = self.body.state;
        let e = self.body.euler_zyx_deg();
        let f = self.wrench.force();
        let m = self.wrench.moment;
        let (p, v, j) = (self.wrench.pressure, self.wrench.viscous, self.wrench.jet);
        let jets = self.jets_string();
        let path = self.csv_path.display().to_string();
        writeln!(
            self.csv,
            "{},{:.9e},{:.12e},{:.12e},{:.12e},{:.9e},{:.9e},{:.9e},{:.6},{:.6},{:.6},{:.6},{:.9e},{:.9e},{:.9e},\
             {:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{}",
            step + 1,
            st.t,
            st.x[0],
            st.x[1],
            st.x[2],
            st.v[0],
            st.v[1],
            st.v[2],
            e[0],
            e[1],
            e[2],
            st.rot_y.to_degrees(),
            st.w[0],
            st.w[1],
            st.w[2],
            f[0],
            f[1],
            f[2],
            m[0],
            m[1],
            m[2],
            p[0],
            p[1],
            p[2],
            v[0],
            v[1],
            v[2],
            j[0],
            j[1],
            j[2],
            jets
        )
        .map_err(|source| Error::Io { path: path.clone(), source })?;
        self.csv.flush().map_err(|source| Error::Io { path, source })
    }

    /// Write the moved points as `<dir>/polyMesh/points`, the OpenFOAM
    /// layout of a moving mesh's time directory, from the host point map.
    pub fn write_points(&self, dir: &Path) -> Result<()> {
        let pm = dir.join("polyMesh");
        fs::create_dir_all(&pm).map_err(|source| Error::Io { path: pm.display().to_string(), source })?;
        let path = pm.join("points");
        let mut out = String::with_capacity(self.rest_world.len() * 48 + 1024);
        let loc = dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        out.push_str("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       vectorField;\n");
        out.push_str(&format!("    location    \"{loc}/polyMesh\";\n    object      points;\n}}\n\n"));
        out.push_str(&format!("{}\n(\n", self.rest_world.len()));
        for &p in &self.rest_world {
            let x = self.body.map_point(p);
            out.push_str(&format!("({:.12e} {:.12e} {:.12e})\n", x[0], x[1], x[2]));
        }
        out.push_str(")\n");
        fs::write(&path, out).map_err(|source| Error::Io { path: path.display().to_string(), source })
    }
}

#[cfg(test)]
mod tests;
