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
//!
//! # Version 2 (SPEC-LIT 161.10-161.16)
//!
//! The same file grows the keys of the roadster contract: an initial
//! orientation and position, `dof planarYZ`, a kinematic `release`, nozzle
//! thrusters ([`thrust`]) whose thrust is the nozzle model's and whose plume
//! is momentum-equivalent, a controller ([`control`]) that commands their
//! valves, the propellant they burn, a ground stop on contact points, and the
//! RAS models a moving mesh has been verified with. A version-1 file runs
//! the version-1 arithmetic: every new path is behind a key it does not
//! carry. Per step the host still downloads only the twelve force numbers
//! and uploads only a few numbers per thruster.

pub mod control;
pub mod thrust;

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use serde::Deserialize;

use crate::device::{cfg_for, Gpu, KernelSet};
use crate::error::{Error, Result};
use crate::field::{BcKind, GpuScalarField, GpuSurfaceScalarField};
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
    /// Translation in y and z, rotation about x: a roll in the y-z plane
    /// (SPEC-LIT 161.13).
    #[serde(rename = "planarYZ")]
    PlanarYz,
}

impl Dof {
    pub fn name(self) -> &'static str {
        match self {
            Dof::Free => "free",
            Dof::PlanarXz => "planarXZ",
            Dof::PlanarYz => "planarYZ",
        }
    }

    /// The one axis a planar body turns about, `None` for `free`.
    pub fn axis(self) -> Option<usize> {
        match self {
            Dof::Free => None,
            Dof::PlanarXz => Some(1),
            Dof::PlanarYz => Some(0),
        }
    }
}

fn dof_free() -> Dof {
    Dof::Free
}

/// The nozzle of a version-2 thruster - SPEC-LIT 161.10.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NozzleSpec {
    /// Pa, the regulated chamber's stagnation pressure.
    pub p0: f64,
    /// K, its stagnation temperature.
    #[serde(rename = "T0")]
    pub t0: f64,
    /// m^2.
    #[serde(rename = "exitArea")]
    pub exit_area: f64,
    #[serde(default = "air_gamma")]
    pub gamma: f64,
    /// J/(kg K).
    #[serde(rename = "R", default = "air_r")]
    pub r: f64,
    /// `A_e/A*` on the supersonic branch; absent is ideally expanded.
    #[serde(rename = "areaRatio", default)]
    pub area_ratio: Option<f64>,
}

fn air_gamma() -> f64 {
    thrust::AIR_GAMMA
}
fn air_r() -> f64 {
    thrust::AIR_R
}

/// One thruster: a patch of the body that blows fluid out along `direction`
/// in BODY axes. Version 1: at `speed` relative to the body while the time
/// lies in one of the `on` intervals `[t0, t1)`. Version 2 (SPEC-LIT
/// 161.10): a `nozzle`, at full valve inside the `on` intervals, or valved
/// by the controller when `on` is absent.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThrusterSpec {
    pub patch: String,
    pub direction: V3,
    #[serde(default)]
    pub speed: Option<f64>,
    #[serde(default)]
    pub nozzle: Option<NozzleSpec>,
    #[serde(default)]
    pub on: Option<Vec<[f64; 2]>>,
}

impl ThrusterSpec {
    /// Is the jet on at time `t`? `t0 <= t < t1` for some interval.
    pub fn is_on(&self, t: f64) -> bool {
        self.on.as_deref().unwrap_or(&[]).iter().any(|iv| iv[0] <= t && t < iv[1])
    }

    /// The version-1 jet speed, zero for a nozzle thruster.
    pub fn jet_speed(&self) -> f64 {
        self.speed.unwrap_or(0.0)
    }

    /// A nozzle with no schedule: the controller commands it.
    pub fn closed_loop(&self) -> bool {
        self.nozzle.is_some() && self.on.is_none()
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

/// The equivalent plume's settings - SPEC-LIT 161.11.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct PlumeSpec {
    #[serde(default = "plume_model_default")]
    pub model: String,
    #[serde(default = "plume_max_mach_default")]
    pub max_mach: f64,
    #[serde(default = "plume_intensity_default")]
    pub turbulence_intensity: f64,
    #[serde(default = "plume_lfrac_default")]
    pub mixing_length_fraction: f64,
}

fn plume_model_default() -> String {
    "momentumEquivalent".to_string()
}
fn plume_max_mach_default() -> f64 {
    0.55
}
fn plume_intensity_default() -> f64 {
    0.05
}
fn plume_lfrac_default() -> f64 {
    0.07
}

impl Default for PlumeSpec {
    fn default() -> Self {
        Self {
            model: plume_model_default(),
            max_mach: plume_max_mach_default(),
            turbulence_intensity: plume_intensity_default(),
            mixing_length_fraction: plume_lfrac_default(),
        }
    }
}

/// One attitude waypoint, `rpy` in degrees (`R = Rz(yaw) Ry(pitch) Rx(roll)`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WaypointSpec {
    pub t: f64,
    pub rpy: V3,
}

fn profile_default() -> String {
    "minJerk".to_string()
}

/// `control.attitude` - SPEC-LIT 161.14.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct AttitudeSpec {
    pub waypoints: Vec<WaypointSpec>,
    #[serde(default = "profile_default")]
    pub profile: String,
    pub natural_frequency: f64,
    pub damping: f64,
}

/// `control.descent` - SPEC-LIT 161.14.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct DescentSpec {
    #[serde(default)]
    pub ground_z: f64,
    pub touchdown_speed: f64,
    pub brake_accel: f64,
    pub velocity_gain: f64,
    #[serde(default = "min_upright_default")]
    pub min_upright: f64,
}

fn min_upright_default() -> f64 {
    0.5
}
fn allocation_default() -> String {
    "boundedLeastSquares".to_string()
}

/// `control` - SPEC-LIT 161.14.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct ControlSpec {
    #[serde(default)]
    pub attitude: Option<AttitudeSpec>,
    #[serde(default)]
    pub descent: Option<DescentSpec>,
    #[serde(default = "allocation_default")]
    pub allocation: String,
}

/// `propellant` - SPEC-LIT 161.15.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropellantSpec {
    /// kg at t = 0, part of `mass`.
    pub mass: f64,
}

fn stop_default() -> bool {
    true
}

/// `ground` - SPEC-LIT 161.15.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct GroundSpec {
    #[serde(default)]
    pub z: f64,
    /// Body-frame (rest-mesh) points, e.g. the wheel bottoms.
    pub contact_points: Vec<V3>,
    #[serde(default = "stop_default")]
    pub stop: bool,
}

/// The `-rigidBody` file - SPEC-LIT 161.1 and, for its version-2 keys,
/// 161.13. `inertia` is the three principal moments about the centre of mass
/// in body axes (the rest mesh's axes), or six tensor components `[xx, yy,
/// zz, xy, xz, yz]`; `centreOfMass` is in the rest mesh's frame;
/// `velocity` is world and `angularVelocity` world at `t = 0`; `rho` turns
/// the kinematic pressure and viscosity into newtons.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct RigidBodySpec {
    pub mass: f64,
    pub inertia: Vec<f64>,
    pub centre_of_mass: V3,
    /// The world centre of mass at `t = 0`; absent is `centreOfMass`.
    #[serde(default)]
    pub position: Option<V3>,
    /// `[roll, pitch, yaw]` degrees, body to world at `t = 0` about the
    /// centre of mass; absent is the rest orientation.
    #[serde(default)]
    pub orientation: Option<V3>,
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
    /// Pa; the nozzles expand to it and the plume runs at it.
    #[serde(default)]
    pub p_ambient: Option<f64>,
    /// s: the body moves at its t = 0 velocity until then, jets shut.
    #[serde(default)]
    pub release: Option<f64>,
    #[serde(default)]
    pub thrusters: Vec<ThrusterSpec>,
    #[serde(default)]
    pub plume: Option<PlumeSpec>,
    #[serde(default)]
    pub control: Option<ControlSpec>,
    #[serde(default)]
    pub propellant: Option<PropellantSpec>,
    #[serde(default)]
    pub ground: Option<GroundSpec>,
    #[serde(default)]
    pub write: WriteSpec,
}

/// The ambient pressure a nozzle expands to when the file names none.
pub const P_AMBIENT_DEFAULT: f64 = 101325.0;

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
        if self.inertia.len() != 3 && self.inertia.len() != 6 {
            return bad(format!(
                "inertia has {} values; give three principal moments [Ixx, Iyy, Izz] or six tensor components \
                 [xx, yy, zz, xy, xz, yz] (SPEC-LIT 161.13)",
                self.inertia.len()
            ));
        }
        if !self.inertia.iter().all(|i| i.is_finite()) || !self.inertia[..3].iter().all(|&i| i > 0.0) {
            return bad(format!("inertia {:?} must be three positive principal moments", self.inertia));
        }
        let [a, b, c] = [self.inertia[0], self.inertia[1], self.inertia[2]];
        let tol = 1e-12 * (a + b + c);
        if a + b < c - tol || b + c < a - tol || a + c < b - tol {
            return bad(format!(
                "inertia {:?} breaks the triangle inequality every rigid body's principal moments obey",
                self.inertia
            ));
        }
        if self.inertia.len() == 6 {
            let j = inertia_tensor(&self.inertia);
            let m2 = j[0][0] * j[1][1] - j[0][1] * j[1][0];
            if !(m2 > 0.0 && det3(&j) > 0.0) {
                return bad(format!(
                    "inertia {:?} is not positive definite; a rigid body's inertia tensor is (SPEC-LIT 161.13)",
                    self.inertia
                ));
            }
        }
        if !(finite(self.centre_of_mass) && finite(self.velocity) && finite(self.angular_velocity) && finite(self.gravity)) {
            return bad("centreOfMass, velocity, angularVelocity and gravity must be finite".to_string());
        }
        if !self.position.is_none_or(finite) || !self.orientation.is_none_or(finite) {
            return bad("position and orientation must be finite (SPEC-LIT 161.13)".to_string());
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
        if self.dof == Dof::PlanarYz {
            if self.velocity[0] != 0.0 {
                return bad(format!("dof planarYZ with an x velocity {}; the plane has none", self.velocity[0]));
            }
            if self.angular_velocity[1] != 0.0 || self.angular_velocity[2] != 0.0 {
                return bad(format!(
                    "dof planarYZ with an angular velocity {:?}; the plane turns about x only",
                    self.angular_velocity
                ));
            }
        }
        if let Some(ax) = self.dof.axis() {
            // The turning axis must start on the world's and be principal:
            // then it stays on it and the gyroscopic term vanishes (SPEC-LIT 161.3).
            let r0 = quat_to_matrix(self.initial_quaternion());
            let col = [r0[0][ax], r0[1][ax], r0[2][ax]];
            if (col[ax] - 1.0).abs() > 1e-12 {
                return bad(format!(
                    "dof {} with orientation {:?}: the body's turning axis must start on the world's \
                     (SPEC-LIT 161.13)",
                    self.dof.name(),
                    self.orientation.unwrap_or_default()
                ));
            }
            if self.inertia.len() == 6 {
                let j = inertia_tensor(&self.inertia);
                let off = (0..3).filter(|&k| k != ax).map(|k| j[ax][k].abs()).fold(0.0f64, f64::max);
                if off != 0.0 {
                    return bad(format!(
                        "dof {} with inertia {:?}: the turning axis must be a principal axis (its products \
                         of inertia zero) (SPEC-LIT 161.13)",
                        self.dof.name(),
                        self.inertia
                    ));
                }
            }
        }
        if let Some(p) = self.p_ambient {
            if !(p > 0.0 && p.is_finite()) {
                return bad(format!("pAmbient is {p}; it must be positive (SPEC-LIT 161.10)"));
            }
        }
        if let Some(r) = self.release {
            if !(r >= 0.0 && r.is_finite()) {
                return bad(format!("release is {r}; it must be zero or positive (SPEC-LIT 161.13)"));
            }
        }
        let pa = self.p_amb();
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
            match (&th.speed, &th.nozzle) {
                (Some(_), Some(_)) => {
                    return bad(format!(
                        "thruster {} has both speed and nozzle; a jet is version 1 (speed) or version 2 \
                         (nozzle) (SPEC-LIT 161.10)",
                        th.patch
                    ))
                }
                (None, None) => {
                    return bad(format!(
                        "thruster {} has neither speed nor nozzle (SPEC-LIT 161.10)",
                        th.patch
                    ))
                }
                (Some(s), None) => {
                    if !(*s >= 0.0 && s.is_finite()) {
                        return bad(format!("thruster {} has speed {}; it must be zero or positive", th.patch, s));
                    }
                }
                (None, Some(n)) => {
                    thrust::nozzle(&th.patch, n.p0, n.t0, n.exit_area, n.gamma, n.r, pa, n.area_ratio)?;
                }
            }
            for iv in th.on.as_deref().unwrap_or(&[]) {
                if !(iv[0].is_finite() && iv[1].is_finite() && iv[0] < iv[1]) {
                    return bad(format!("thruster {} has the interval {:?}; it needs t0 < t1", th.patch, iv));
                }
            }
            if th.closed_loop() && self.control.is_none() {
                return bad(format!(
                    "thruster {} has a nozzle and no `on` schedule, so the controller valves it, but the file \
                     has no control block (SPEC-LIT 161.14)",
                    th.patch
                ));
            }
        }
        if let Some(p) = &self.plume {
            if p.model != "momentumEquivalent" {
                return bad(format!(
                    "plume.model {:?}; the one model is \"momentumEquivalent\" (SPEC-LIT 161.11)",
                    p.model
                ));
            }
            if !(p.max_mach > 0.0 && p.max_mach <= 1.0) {
                return bad(format!("plume.maxMach {}; it must lie in (0, 1] (SPEC-LIT 161.11)", p.max_mach));
            }
            if !(p.turbulence_intensity >= 0.0 && p.turbulence_intensity.is_finite()) {
                return bad(format!("plume.turbulenceIntensity {} (SPEC-LIT 161.11)", p.turbulence_intensity));
            }
            if !(p.mixing_length_fraction > 0.0 && p.mixing_length_fraction.is_finite()) {
                return bad(format!("plume.mixingLengthFraction {} (SPEC-LIT 161.11)", p.mixing_length_fraction));
            }
        }
        if let Some(c) = &self.control {
            if c.allocation != "boundedLeastSquares" {
                return bad(format!(
                    "control.allocation {:?}; the one allocation is \"boundedLeastSquares\" (SPEC-LIT 161.14)",
                    c.allocation
                ));
            }
            if let Some(a) = &c.attitude {
                if a.profile != "minJerk" {
                    return bad(format!(
                        "control.attitude.profile {:?}; the one profile is \"minJerk\" (SPEC-LIT 161.14)",
                        a.profile
                    ));
                }
                if !(a.natural_frequency > 0.0 && a.natural_frequency.is_finite() && a.damping >= 0.0 && a.damping.is_finite()) {
                    return bad(format!(
                        "control.attitude naturalFrequency {} and damping {}: positive and non-negative \
                         (SPEC-LIT 161.14)",
                        a.natural_frequency, a.damping
                    ));
                }
                let wp: Vec<(f64, V3)> = a.waypoints.iter().map(|w| (w.t, w.rpy)).collect();
                control::AttitudeProfile::new(&wp)?;
            }
            if let Some(d) = &c.descent {
                let ok = d.ground_z.is_finite()
                    && d.touchdown_speed >= 0.0
                    && d.touchdown_speed.is_finite()
                    && d.brake_accel > 0.0
                    && d.brake_accel.is_finite()
                    && d.velocity_gain > 0.0
                    && d.velocity_gain.is_finite()
                    && (-1.0..=1.0).contains(&d.min_upright);
                if !ok {
                    return bad(format!(
                        "control.descent {d:?}: touchdownSpeed >= 0, brakeAccel > 0, velocityGain > 0 and \
                         minUpright in [-1, 1] (SPEC-LIT 161.14)"
                    ));
                }
                if self.ground.as_ref().is_none_or(|g| g.contact_points.is_empty()) {
                    return bad(
                        "control.descent needs ground.contactPoints: its height is the lowest contact point's \
                         (SPEC-LIT 161.14)"
                            .to_string(),
                    );
                }
            }
        }
        if let Some(p) = &self.propellant {
            if !(p.mass >= 0.0 && p.mass < self.mass) {
                return bad(format!(
                    "propellant.mass {} kg must be zero or more and below the body's mass {} kg, which \
                     includes it (SPEC-LIT 161.15)",
                    p.mass, self.mass
                ));
            }
        }
        if let Some(g) = &self.ground {
            if !g.z.is_finite() || !g.contact_points.iter().all(|&p| finite(p)) {
                return bad("ground.z and ground.contactPoints must be finite (SPEC-LIT 161.15)".to_string());
            }
            if g.stop && g.contact_points.is_empty() {
                return bad("ground.stop with no contactPoints: nothing can touch down (SPEC-LIT 161.15)".to_string());
            }
        }
        if self.write.every == 0 {
            return bad("write.every is 0; it counts steps and must be at least 1".to_string());
        }
        Ok(())
    }

    /// The controller of the `control` block (SPEC-LIT 161.14), its height
    /// read off `ground.contactPoints`.
    pub fn controller(&self) -> Result<Option<control::Controller>> {
        let Some(c) = &self.control else { return Ok(None) };
        let attitude = match &c.attitude {
            Some(a) => {
                let wp: Vec<(f64, V3)> = a.waypoints.iter().map(|w| (w.t, w.rpy)).collect();
                Some((control::AttitudeProfile::new(&wp)?, a.natural_frequency, a.damping))
            }
            None => None,
        };
        Ok(Some(control::Controller {
            attitude,
            descent: c.descent.as_ref().map(|d| control::DescentLaw {
                ground_z: d.ground_z,
                touchdown_speed: d.touchdown_speed,
                brake_accel: d.brake_accel,
                velocity_gain: d.velocity_gain,
                min_upright: d.min_upright,
            }),
            contact: self.ground.as_ref().map(|g| g.contact_points.clone()).unwrap_or_default(),
            braking: false,
        }))
    }

    /// The ambient pressure, `pAmbient` or 101325 Pa.
    pub fn p_amb(&self) -> f64 {
        self.p_ambient.unwrap_or(P_AMBIENT_DEFAULT)
    }

    /// The orientation at `t = 0`.
    pub fn initial_quaternion(&self) -> [f64; 4] {
        match self.orientation {
            Some(rpy) => control::quat_from_rpy_deg(rpy),
            None => [1.0, 0.0, 0.0, 0.0],
        }
    }

    /// The principal moments, the first three values of `inertia`.
    pub fn principal(&self) -> V3 {
        [self.inertia[0], self.inertia[1], self.inertia[2]]
    }

    /// `[Ixy, Ixz, Iyz]`, zero for three values.
    pub fn products(&self) -> V3 {
        if self.inertia.len() == 6 {
            [self.inertia[3], self.inertia[4], self.inertia[5]]
        } else {
            [0.0; 3]
        }
    }

    /// Any nozzle thruster: thrust from the nozzle model (SPEC-LIT 161.10).
    pub fn has_nozzles(&self) -> bool {
        self.thrusters.iter().any(|t| t.nozzle.is_some())
    }

    /// Does the file use a key of SPEC-LIT 161.13-161.15? Then the CSV
    /// carries the version-2 columns and the run ends with a summary.
    pub fn is_v2(&self) -> bool {
        self.inertia.len() == 6
            || self.position.is_some()
            || self.orientation.is_some()
            || self.dof == Dof::PlanarYz
            || self.p_ambient.is_some()
            || self.release.is_some()
            || self.has_nozzles()
            || self.plume.is_some()
            || self.control.is_some()
            || self.propellant.is_some()
            || self.ground.is_some()
    }
}

/// The symmetric tensor of `[xx, yy, zz, xy, xz, yz]`.
fn inertia_tensor(v: &[f64]) -> M3 {
    [[v[0], v[3], v[4]], [v[3], v[1], v[5]], [v[4], v[5], v[2]]]
}

fn det3(m: &M3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn inverse3(m: &M3) -> M3 {
    let d = det3(m);
    let c = |a: usize, b: usize, cc: usize, dd: usize| m[a][b] * m[cc][dd] - m[a][dd] * m[cc][b];
    [
        [c(1, 1, 2, 2) / d, -c(0, 1, 2, 2) / d, c(0, 1, 1, 2) / d],
        [-c(1, 0, 2, 2) / d, c(0, 0, 2, 2) / d, -c(0, 0, 1, 2) / d],
        [c(1, 0, 2, 1) / d, -c(0, 0, 2, 1) / d, c(0, 0, 1, 1) / d],
    ]
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
    /// kg; a body that burns propellant loses it here (SPEC-LIT 161.15).
    pub mass: f64,
    /// The diagonal of the inertia tensor about the centre of mass, body
    /// axes - the principal moments when [`Self::products`] is zero.
    pub inertia: V3,
    /// The off-diagonal tensor components `[Ixy, Ixz, Iyz]` (SPEC-LIT
    /// 161.13); zero for principal body axes, the version-1 case.
    pub products: V3,
    /// `J^-1` when the products are not all zero.
    inv: Option<M3>,
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
    /// The body of a validated spec, at its initial position and orientation
    /// (SPEC-LIT 161.13; the rest orientation for a version-1 file).
    pub fn new(spec: &RigidBodySpec) -> Result<Self> {
        spec.validate()?;
        let mut b = Self::from_parts(
            spec.mass,
            spec.principal(),
            spec.centre_of_mass,
            spec.velocity,
            spec.angular_velocity,
            spec.gravity,
            spec.dof,
        );
        if spec.inertia.len() == 6 {
            b = b.with_products(spec.products());
        }
        if let Some(x) = spec.position {
            b.state.x = x;
        }
        if spec.orientation.is_some() {
            let q = spec.initial_quaternion();
            b.state.q = q;
            // angularVelocity is world at t = 0; the state holds body axes.
            b.state.w = mat_t_vec(&quat_to_matrix(q), spec.angular_velocity);
        }
        Ok(b)
    }

    /// A body from its parts, unvalidated - the tests' constructor.
    pub fn from_parts(mass: f64, inertia: V3, c0: V3, v0: V3, w0: V3, gravity: V3, dof: Dof) -> Self {
        Self {
            mass,
            inertia,
            products: [0.0; 3],
            inv: None,
            gravity,
            dof,
            c0,
            state: BodyState { t: 0.0, x: c0, v: v0, q: [1.0, 0.0, 0.0, 0.0], w: w0, rot_y: 0.0 },
            prev: None,
        }
    }

    /// The same body with the tensor's off-diagonal components `[Ixy, Ixz,
    /// Iyz]`; all zero leaves the principal-axes arithmetic of 161.3.
    pub fn with_products(mut self, products: V3) -> Self {
        self.products = products;
        self.inv = if products == [0.0; 3] { None } else { Some(inverse3(&self.inertia_matrix())) };
        self
    }

    /// The inertia tensor about the centre of mass, body axes.
    pub fn inertia_matrix(&self) -> M3 {
        let (i, p) = (self.inertia, self.products);
        [[i[0], p[0], p[1]], [p[0], i[1], p[2]], [p[1], p[2], i[2]]]
    }

    /// `J w`.
    fn j_mul(&self, w: V3) -> V3 {
        match self.inv {
            None => [self.inertia[0] * w[0], self.inertia[1] * w[1], self.inertia[2] * w[2]],
            Some(_) => mat_vec(&self.inertia_matrix(), w),
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

    /// `(m |V|^2 + w . J w)/2`.
    pub fn kinetic_energy(&self) -> f64 {
        let s = &self.state;
        let iw = self.j_mul(s.w);
        0.5 * (self.mass * dot(s.v, s.v) + dot(s.w, iw))
    }

    /// The angular momentum about the centre of mass in world axes, `R J w`.
    pub fn angular_momentum_world(&self) -> V3 {
        let s = &self.state;
        let iw = self.j_mul(s.w);
        mat_vec(&self.rotation(), iw)
    }

    /// The force with the components the degrees of freedom forbid removed.
    fn constrain_force(&self, f: V3) -> V3 {
        match self.dof {
            Dof::Free => f,
            Dof::PlanarXz => [f[0], 0.0, f[2]],
            Dof::PlanarYz => [0.0, f[1], f[2]],
        }
    }

    /// The world moment with the components the degrees of freedom forbid
    /// removed.
    fn constrain_moment(&self, m: V3) -> V3 {
        match self.dof {
            Dof::Free => m,
            Dof::PlanarXz => [0.0, m[1], 0.0],
            Dof::PlanarYz => [m[0], 0.0, 0.0],
        }
    }

    /// The acceleration with the component normal to the plane removed.
    fn constrain_accel(&self, a: &mut V3) {
        match self.dof {
            Dof::Free => {}
            Dof::PlanarXz => a[1] = 0.0,
            Dof::PlanarYz => a[0] = 0.0,
        }
    }

    /// `dw/dt` in body axes: `J^-1 (M - w x J w)`, with the planar
    /// constraint's other two components held at zero.
    fn omega_rate(&self, m_body: V3, w: V3) -> V3 {
        let mut d = match &self.inv {
            None => {
                let i = self.inertia;
                let iw = [i[0] * w[0], i[1] * w[1], i[2] * w[2]];
                let g = cross(w, iw);
                [(m_body[0] - g[0]) / i[0], (m_body[1] - g[1]) / i[1], (m_body[2] - g[2]) / i[2]]
            }
            Some(inv) => {
                let g = cross(w, mat_vec(&self.inertia_matrix(), w));
                mat_vec(inv, sub(m_body, g))
            }
        };
        match self.dof {
            Dof::Free => {}
            Dof::PlanarXz => {
                d[0] = 0.0;
                d[2] = 0.0;
            }
            Dof::PlanarYz => {
                d[1] = 0.0;
                d[2] = 0.0;
            }
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
        self.constrain_accel(&mut a);
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

    /// [`Self::advance`] with the thrusters' force and moment in BODY axes,
    /// held over the step - SPEC-LIT 161.15. The fluid's part is
    /// extrapolated as in 161.3; the thrust is not (the valves are set at the
    /// step's start and the nozzles turn with the body), so a valve that
    /// opens adds no jump to the extrapolation. The thrust moment is
    /// constant in body axes and enters Euler's equations as it is; the
    /// thrust force turns with the body and is integrated by Simpson's rule
    /// on `R` at the step's start, middle and end:
    ///
    /// ```text
    /// V += dt/6 (a0 + 4 a_mid + a1),  X += dt^2/6 (a0 + 2 a_mid),  a(s) = R(s) F_b / m
    /// ```
    pub fn advance_thrust(&mut self, dt: f64, force: V3, moment: V3, thrust_f_body: V3, thrust_m_body: V3) {
        let f = self.constrain_force(force);
        let m_world = self.constrain_moment(moment);
        let mut a = add(scale(f, 1.0 / self.mass), self.gravity);
        self.constrain_accel(&mut a);
        let r0 = self.rotation();
        let m_body = mat_t_vec(&r0, m_world);
        let m_thr = mat_t_vec(&r0, self.constrain_moment(mat_vec(&r0, thrust_m_body)));
        let (a_prev, m_prev) = self.prev.unwrap_or((a, m_body));

        // Rotation first: the thrust force needs R over the step.
        let slope = scale(sub(m_body, m_prev), 1.0 / dt);
        let m0 = add(m_body, m_thr);
        let w0 = self.state.w;
        let h = 0.5 * dt;
        let w_mid = self.rk4_omega(m0, slope, 0.0, w0, h);
        let w_end = self.rk4_omega(m0, slope, h, w_mid, h);
        let omega = scale(add(add(w0, scale(w_mid, 4.0)), w_end), 1.0 / 6.0);
        let q0 = self.state.q;
        let q_end = control::normalise(quat_mul(q0, quat_exp(scale(omega, dt))));
        let q_mid = control::normalise(quat_mul(q0, quat_exp(scale(add(w0, w_mid), 0.25 * dt))));

        let thr = |q: [f64; 4]| -> V3 {
            let mut at = scale(mat_vec(&quat_to_matrix(q), thrust_f_body), 1.0 / self.mass);
            at = self.constrain_force(at);
            self.constrain_accel(&mut at);
            at
        };
        let (t0, tm, t1) = (thr(q0), thr(q_mid), thr(q_end));

        let da = sub(a, a_prev);
        let s = &mut self.state;
        let x_new = add(
            add(add(s.x, scale(s.v, dt)), add(scale(a, 0.5 * dt * dt), scale(da, dt * dt / 6.0))),
            scale(add(t0, scale(tm, 2.0)), dt * dt / 6.0),
        );
        let v_new = add(
            add(s.v, add(scale(a, dt), scale(da, 0.5 * dt))),
            scale(add(add(t0, scale(tm, 4.0)), t1), dt / 6.0),
        );
        s.x = x_new;
        s.v = v_new;
        s.q = q_end;
        s.w = w_end;
        s.rot_y += mat_vec(&quat_to_matrix(q_end), omega)[1] * dt;
        s.t += dt;
        self.prev = Some((a, m_body));
    }

    /// The kinematic hold of `release` (SPEC-LIT 161.13): the body moves at
    /// its velocity and turns at its angular velocity, unchanged, and no
    /// force is integrated. The extrapolation restarts after it.
    pub fn advance_kinematic(&mut self, dt: f64) {
        let s = &mut self.state;
        s.x = add(s.x, scale(s.v, dt));
        let q = control::normalise(quat_mul(s.q, quat_exp(scale(s.w, dt))));
        s.q = q;
        s.rot_y += mat_vec(&quat_to_matrix(q), s.w)[1] * dt;
        s.t += dt;
        self.prev = None;
    }
}

// ==========================================================================
//  161.2, 161.4, 161.5  The device side
// ==========================================================================

/// The six kernels of `cuda/rigidbody.cu`.
pub struct RigidBodyKernels {
    pub rigid_points: CudaFunction,
    pub wall_velocity: CudaFunction,
    pub scatter_flux: CudaFunction,
    pub force_moment: CudaFunction,
    pub force_moment_v2: CudaFunction,
    pub jet_scalar: CudaFunction,
}

impl RigidBodyKernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let ks = KernelSet::new(gpu, crate::kernels::RIGIDBODY)?;
        Ok(Self {
            rigid_points: ks.func("rbRigidPoints")?,
            wall_velocity: ks.func("rbWallVelocity")?,
            scatter_flux: ks.func("rbScatterFlux")?,
            force_moment: ks.func("rbForceMoment")?,
            force_moment_v2: ks.func("rbForceMomentV2")?,
            jet_scalar: ks.func("rbJetScalar")?,
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

/// How many per-thruster scalar blocks [`RigidBodyDevice::jet_scalar`]
/// reads from: `T`, `k`, `omega`, `epsilon`, `nuTilda` (SPEC-LIT 161.11).
pub const JET_SCALAR_FIELDS: usize = 5;
/// The block of each of those, in that order.
pub const JET_T: usize = 0;
pub const JET_K: usize = 1;
pub const JET_OMEGA: usize = 2;
pub const JET_EPSILON: usize = 3;
pub const JET_NU_TILDA: usize = 4;

/// The rest points, the body faces and the scratch the device operations
/// need, resident.
pub struct RigidBodyDevice {
    pub n_points: usize,
    pub n_faces: usize,
    pub n_jets: usize,
    /// The world position of the mesh's local origin (SPEC-LIT 118.7).
    pub origin: V3,
    rest: DevBuf<GeoVec3>,
    faces: DevBuf<Label>,
    face_jet: DevBuf<Label>,
    /// `[n_faces]` 1 where the face counts in the fluid's force, 0 on a
    /// nozzle thruster's exit patch (SPEC-LIT 161.12).
    face_aero: DevBuf<Label>,
    jet_vel: DevBuf<Vec3>,
    /// `[JET_SCALAR_FIELDS * n_jets]` the per-thruster boundary scalars.
    jet_vals: DevBuf<Scalar>,
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
            face_aero: upload_labels(gpu, &vec![1 as Label; body_faces.len()])?,
            jet_vel: gpu.zeros(n_jets.max(1))?,
            jet_vals: gpu.zeros((JET_SCALAR_FIELDS * n_jets).max(1))?,
            phi_mesh_b: gpu.zeros(n_bf.max(1))?,
            wall_flux: gpu.zeros(body_faces.len().max(1))?,
            out: gpu.zeros(12)?,
            k: RigidBodyKernels::new(gpu)?,
        })
    }

    /// Leave the faces of the thrusters `excluded[j]` out of the fluid's force
    /// (SPEC-LIT 161.12) - a setup write.
    pub fn exclude_jets(&mut self, gpu: &Gpu, face_jet: &[Label], excluded: &[bool]) -> Result<()> {
        if face_jet.len() != self.n_faces {
            return Err(Error::Config("RigidBodyDevice::exclude_jets: one label per body face".to_string()));
        }
        let aero: Vec<Label> =
            face_jet.iter().map(|&j| if j >= 0 && excluded[j as usize] { 0 } else { 1 }).collect();
        if !aero.is_empty() {
            gpu.write(&mut self.face_aero, &aero)?;
        }
        Ok(())
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

    /// Stage the per-thruster scalars, `JET_SCALAR_FIELDS` blocks of
    /// `n_jets` - a few numbers per step, the one upload SPEC-LIT 161.11 adds.
    pub fn stage_jet_scalars(&mut self, gpu: &Gpu, vals: &[Scalar]) -> Result<()> {
        if vals.len() != JET_SCALAR_FIELDS * self.n_jets {
            return Err(Error::Config(format!(
                "RigidBodyDevice::stage_jet_scalars: {} values for {} thrusters",
                vals.len(),
                self.n_jets
            )));
        }
        if vals.is_empty() {
            return Ok(());
        }
        gpu.write(&mut self.jet_vals, vals)
    }

    /// Write block `block` of the staged scalars into `ref_value` on every
    /// jet face - SPEC-LIT 161.11.
    pub fn jet_scalar(&self, gpu: &Gpu, ref_value: &mut DevBuf<Scalar>, block: usize) -> Result<()> {
        if self.n_faces == 0 || self.n_jets == 0 {
            return Ok(());
        }
        let n = self.n_faces as Label;
        let off = (block * self.n_jets) as Label;
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.jet_scalar)
                .arg(&mut *ref_value)
                .arg(&self.faces)
                .arg(&self.face_jet)
                .arg(&self.jet_vals)
                .arg(&off)
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
        self.read_wrench(gpu)
    }

    /// [`Self::force_moment`] with the eddy viscosity's boundary values
    /// `nut_b` in the wall shear and the excluded jet faces left out -
    /// SPEC-LIT 161.12. `nut_b` `None` is the laminar shear.
    #[allow(clippy::too_many_arguments)]
    pub fn force_moment_v2(
        &mut self,
        gpu: &Gpu,
        gm: &GpuMesh,
        p_b: &DevBuf<Scalar>,
        u: &DevBuf<Vec3>,
        u_b: &DevBuf<Vec3>,
        phi_rel_b: &DevBuf<Scalar>,
        nut_b: Option<&DevBuf<Scalar>>,
        rho: f64,
        nu: f64,
        g_hyd: V3,
        body: &RigidBody,
    ) -> Result<Wrench> {
        let v = body.state.v;
        let w = body.omega_world();
        let (xl, _) = self.local(body);
        let n = self.n_faces as Label;
        let use_nut: i32 = i32::from(nut_b.is_some());
        let nut = nut_b.unwrap_or(p_b);
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (RB_THREADS, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            gpu.stream()
                .launch_builder(&self.k.force_moment_v2)
                .arg(&mut self.out)
                .arg(&self.faces)
                .arg(&self.face_aero)
                .arg(p_b)
                .arg(u)
                .arg(u_b)
                .arg(phi_rel_b)
                .arg(nut)
                .arg(&gm.b_face_cells)
                .arg(&gm.b_sf)
                .arg(&gm.b_cf)
                .arg(&gm.b_delta_coeffs)
                .arg(&rho)
                .arg(&nu)
                .arg(&use_nut)
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
        self.read_wrench(gpu)
    }

    fn read_wrench(&self, gpu: &Gpu) -> Result<Wrench> {
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

/// What the driver tells [`RigidBodyCoupling::attach`] beyond the file.
#[derive(Debug, Clone, Copy, Default)]
pub struct AttachOptions {
    /// `-permissive`: the plume's Mach-area refusal becomes a warning.
    pub permissive: bool,
    /// A RAS model runs (SPEC-LIT 161.16): the force's wall shear carries
    /// `nu_t`, and the jets carry `k` and the model's second variable in.
    pub turbulent: bool,
}

/// How the run ended at a touchdown - SPEC-LIT 161.15.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Touchdown {
    pub t: f64,
    pub velocity: V3,
    pub rpy_deg: V3,
    pub upright: f64,
    pub peak_load_g: f64,
    /// The peak of `|F_thrust|/(m |g|)` alone: the fluid's one-step pressure
    /// transients when a valve opens are in `peak_load_g`, not here.
    pub peak_thrust_g: f64,
    pub propellant_used: f64,
}

/// The rigid body attached to one `Simple`: the body, its device state, the
/// last force, the CSV and the progress clock - and, for a file with the
/// keys of SPEC-LIT 161.13-161.15, the nozzles, the controller and the
/// propellant.
pub struct RigidBodyCoupling {
    pub body: RigidBody,
    pub thrusters: Vec<ThrusterSpec>,
    /// The fluid's force and moment (world, about the centre of mass); with
    /// nozzle thrusters it leaves their exit patches out.
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

    // ---- SPEC-LIT 161.10-161.16 ----------------------------------------
    /// The file uses a version-2 key: the CSV carries the version-2 columns.
    csv_v2: bool,
    /// Some thruster has a nozzle: its thrust is the model's (SPEC-LIT 161.12) and
    /// the body advances by [`RigidBody::advance_thrust`].
    nozzle_mode: bool,
    turbulent: bool,
    release: f64,
    releasing: bool,
    p_amb: f64,
    plume: PlumeSpec,
    nozzles: Vec<Option<thrust::NozzleDesign>>,
    geoms: Vec<control::ThrusterGeom>,
    patches: Vec<thrust::PatchGeometry>,
    /// The full-valve plume speed (nozzle) or the jet speed (version 1): the
    /// reference of the inflow turbulence's floor.
    u_ref: Vec<f64>,
    controller: Option<control::Controller>,
    ground: Option<GroundSpec>,
    mass0: f64,
    prop0: Option<f64>,
    prop_used: f64,
    empty_warned: bool,
    valve: Vec<f64>,
    thrust: Vec<f64>,
    plumes: Vec<thrust::PlumeState>,
    thrust_body: (V3, V3),
    reference: Option<control::AttitudeReference>,
    load_g: f64,
    peak_load_g: f64,
    peak_thrust_g: f64,
    h_contact: Option<f64>,
    /// Which of `T`, `k`, `omega`, `epsilon`, `nuTilda` the jets write.
    jet_fields: [bool; JET_SCALAR_FIELDS],
    face_jet: Vec<Label>,
    /// The body faces' boundary ids, `body_faces`'s order.
    face_ids: Vec<Label>,
    pub touchdown: Option<Touchdown>,
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

/// The rest-frame geometry of every thruster's exit patch (SPEC-LIT 161.11).
fn thruster_patches(hm: &HostMesh, raw: &PolyMeshRaw, spec: &RigidBodySpec, rest: &[V3]) -> Result<Vec<thrust::PatchGeometry>> {
    spec.thrusters
        .iter()
        .map(|t| {
            let p = hm.patches.iter().find(|p| p.name == t.patch).ok_or_else(|| {
                Error::Config(format!("-rigidBody: thruster patch {} is not in the mesh (SPEC-LIT 161.1)", t.patch))
            })?;
            let faces: Vec<Vec<usize>> = (0..p.size)
                .map(|k| raw.faces[hm.n_internal_faces + p.start + k].iter().map(|&v| v as usize).collect())
                .collect();
            let refs: Vec<&[usize]> = faces.iter().map(|f| f.as_slice()).collect();
            Ok(thrust::patch_geometry(rest, &refs))
        })
        .collect()
}

fn csv_err(path: &Path) -> impl Fn(std::io::Error) -> Error + '_ {
    move |source| Error::Io { path: path.display().to_string(), source }
}

impl RigidBodyCoupling {
    /// Build the moving mesh from the case's own polyMesh, place it at the
    /// body's initial position (SPEC-LIT 161.13), seed the body walls'
    /// velocity and the flux from it, attach the motion to `s`, and measure
    /// the force the first step will use - SPEC-LIT 161.6.
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
        opts: AttachOptions,
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
        let rest_world: Vec<V3> = raw.points.iter().map(|p| dvec(p.to_dvec3())).collect();

        // ---- SPEC-LIT 161.10-161.11: the nozzles, their patches, the plume.
        let p_amb = spec.p_amb();
        let plume = spec.plume.clone().unwrap_or_default();
        let patches = thruster_patches(hm, raw, spec, &rest_world)?;
        let mut nozzles = Vec::with_capacity(spec.thrusters.len());
        let mut geoms = Vec::with_capacity(spec.thrusters.len());
        let mut u_ref = Vec::with_capacity(spec.thrusters.len());
        for (t, pg) in spec.thrusters.iter().zip(&patches) {
            let nz = match &t.nozzle {
                Some(n) => Some(thrust::nozzle(&t.patch, n.p0, n.t0, n.exit_area, n.gamma, n.r, p_amb, n.area_ratio)?),
                None => None,
            };
            if let Some(d) = &nz {
                if let Some(w) = thrust::check_patch_area(
                    &t.patch,
                    d.f_max,
                    pg.area,
                    d.t0,
                    d.gamma,
                    d.r,
                    p_amb,
                    plume.max_mach,
                    opts.permissive,
                )? {
                    eprintln!("WARNING (-permissive): {w}");
                }
                u_ref.push(thrust::equivalent_plume(d.f_max, pg.area, d.t0, d.gamma, d.r, p_amb).u_eq);
            } else {
                u_ref.push(t.jet_speed());
            }
            geoms.push(control::ThrusterGeom {
                r_body: sub(pg.centroid, spec.centre_of_mass),
                dir_body: t.unit_direction(),
                f_max: nz.map_or(0.0, |d| d.f_max),
                closed_loop: t.closed_loop(),
            });
            nozzles.push(nz);
        }
        let nozzle_mode = nozzles.iter().any(|n| n.is_some());
        if nozzle_mode {
            let excluded: Vec<bool> = nozzles.iter().map(|n| n.is_some()).collect();
            dev.exclude_jets(gpu, &face_jet, &excluded)?;
        }
        let controller = spec.controller()?;

        // ---- SPEC-LIT 161.13: the mesh placed at the body's t = 0 pose.
        let csr = flatten_faces(&raw.faces);
        let mut ale = AleMesh::new(gpu, hm, gm, &raw.points, &csr)?;
        let placed = spec.position.is_some() || spec.orientation.is_some();
        if placed {
            dev.write_points(gpu, &mut ale.points, &body)?;
            ale.reseat(gpu, gm)?;
        }

        // The t = 0 seed: the whole rigid velocity on the body faces, the jets
        // that are on at t = 0, and the flux built from that U, before the
        // motion is attached (whose relative flux is then phi itself). A
        // nozzle starts shut; its first step's valve is the controller's or
        // the schedule's.
        let release = spec.release.unwrap_or(0.0);
        let r = body.rotation();
        let jets: Vec<V3> = spec
            .thrusters
            .iter()
            .map(|t| {
                if t.nozzle.is_none() && 0.0 >= release && t.is_on(0.0) {
                    scale(mat_vec(&r, t.unit_direction()), t.jet_speed())
                } else {
                    [0.0; 3]
                }
            })
            .collect();
        dev.wall_velocity(gpu, gm, &mut s.u_mut().ref_value, &body, &jets, true)?;
        field_ops::correct_boundary_conditions_vector(gpu, &fldk, s.u_mut(), gm)?;
        let mut phi = GpuSurfaceScalarField::zeros(gpu, gm, "phi")?;
        if placed {
            // The host mesh holds the REST geometry; the placed one is on the
            // device only.
            let fvk = crate::fv::FvKernels::new(gpu)?;
            crate::fv::interpolate_vector_flux(gpu, &fvk, &mut phi, s.u(), gm)?;
        } else {
            crate::field_setup::compute_phi_from_u(gpu, &mut phi, s.u(), hm)?;
        }
        *s.phi_mut() = phi;

        s.attach_motion(gpu, ale, &faces)?;

        let rho = spec.rho;
        let kernel_v2 = nozzle_mode || opts.turbulent;
        let wrench = if kernel_v2 {
            dev.force_moment_v2(
                gpu,
                gm,
                &s.p().bf,
                &s.u().f,
                &s.u().bf,
                &s.convective_flux().bf,
                None,
                rho,
                nu,
                g_hyd,
                &body,
            )?
        } else {
            dev.force_moment(
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
            )?
        };

        if let Some(dir) = csv_path.parent() {
            if !dir.as_os_str().is_empty() {
                fs::create_dir_all(dir).map_err(|source| Error::Io { path: dir.display().to_string(), source })?;
            }
        }
        let file = File::create(csv_path).map_err(csv_err(csv_path))?;
        let mut csv = BufWriter::new(file);
        let names: Vec<&str> = spec.thrusters.iter().map(|t| t.patch.as_str()).collect();
        let csv_v2 = spec.is_v2();
        let mut header = format!(
            "step,time,X,Y,Z,Vx,Vy,Vz,roll_deg,pitch_deg,yaw_deg,rotY_deg,wx_body,wy_body,wz_body,\
             Fx,Fy,Fz,Mx,My,Mz,Fpx,Fpy,Fpz,Fvx,Fvy,Fvz,Fjx,Fjy,Fjz,thrusters_on[{}]",
            names.join("|")
        );
        if csv_v2 {
            for n in &names {
                header.push_str(&format!(",u_{n},F_{n},Ueq_{n},Meq_{n}"));
            }
            header.push_str(
                ",propellant,load_g,h_contact,ref_roll_deg,ref_pitch_deg,ref_yaw_deg,quat_w,quat_x,quat_y,quat_z",
            );
        }
        writeln!(csv, "{header}").map_err(csv_err(csv_path))?;

        println!(
            "rigid body (SPEC-LIT 161): mass {} kg, inertia ({}) kg m2, dof {}, centre of mass ({} {} {}), \
             V0 ({} {} {}), w0 ({} {} {}), gravity ({} {} {})",
            spec.mass,
            spec.inertia.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "),
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
        if placed {
            let e = body.euler_zyx_deg();
            println!(
                "  placed at t = 0 (SPEC-LIT 161.13): centre of mass ({} {} {}), rpy ({:.4} {:.4} {:.4}) deg",
                body.state.x[0], body.state.x[1], body.state.x[2], e[0], e[1], e[2]
            );
        }
        if release > 0.0 {
            println!("  release at t = {release} s: until then the body moves at its t = 0 velocity, jets shut");
        }
        for ((t, nz), (pg, g)) in spec.thrusters.iter().zip(&nozzles).zip(patches.iter().zip(&geoms)) {
            match nz {
                Some(d) => {
                    let full = thrust::equivalent_plume(d.f_max, pg.area, d.t0, d.gamma, d.r, p_amb);
                    println!(
                        "  nozzle {} (SPEC-LIT 161.10): p0 {} Pa, T0 {} K, A_e {} m2, A_e/A* {:.4}: M_e {:.4}, \
                         U_e {:.2} m/s, p_e {:.1} Pa, F_max {:.2} N, mdot_max {:.4} kg/s; exit patch {:.5} m2 \
                         (D_h {:.4} m) at ({:.4} {:.4} {:.4}): U_eq {:.2} m/s, T_j {:.2} K, M_eq {:.4} at full \
                         valve; {}",
                        t.patch,
                        d.p0,
                        d.t0,
                        d.exit_area,
                        d.area_ratio,
                        d.mach_exit,
                        d.u_exit,
                        d.p_exit,
                        d.f_max,
                        d.mdot_max,
                        pg.area,
                        pg.hydraulic_diameter,
                        pg.centroid[0],
                        pg.centroid[1],
                        pg.centroid[2],
                        full.u_eq,
                        full.t_j,
                        full.mach_eq,
                        if g.closed_loop { "valved by the controller" } else { "full valve in its `on` windows" },
                    );
                }
                None => println!(
                    "  jet {} (SPEC-LIT 161.4): speed {} m/s, exit patch {:.5} m2",
                    t.patch,
                    t.jet_speed(),
                    pg.area
                ),
            }
        }
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

        let n_jets = spec.thrusters.len();
        let mass0 = spec.mass;
        let plumes: Vec<thrust::PlumeState> = nozzles
            .iter()
            .map(|n| match n {
                Some(d) => thrust::equivalent_plume(0.0, 1.0, d.t0, d.gamma, d.r, p_amb),
                None => thrust::PlumeState::default(),
            })
            .collect();
        let h_contact = spec
            .ground
            .as_ref()
            .and_then(|g| control::lowest_point(&body, &g.contact_points).map(|z| z - g.z));
        Ok(Self {
            n_body_faces: faces.len(),
            jets_on: vec![false; n_jets],
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
            csv_v2,
            nozzle_mode,
            turbulent: opts.turbulent,
            release,
            releasing: release > 0.0,
            p_amb,
            plume,
            nozzles,
            geoms,
            patches,
            u_ref,
            controller,
            ground: spec.ground.clone(),
            mass0,
            prop0: spec.propellant.as_ref().map(|p| p.mass),
            prop_used: 0.0,
            empty_warned: false,
            valve: vec![0.0; n_jets],
            thrust: vec![0.0; n_jets],
            plumes,
            thrust_body: ([0.0; 3], [0.0; 3]),
            reference: None,
            load_g: 0.0,
            peak_load_g: 0.0,
            peak_thrust_g: 0.0,
            h_contact,
            jet_fields: [false; JET_SCALAR_FIELDS],
            face_jet,
            face_ids: faces.clone(),
            touchdown: None,
        })
    }


    /// Check and remember which boundary scalars the jets write - SPEC-LIT
    /// 161.11 - once the energy and turbulence fields exist, and write their
    /// t = 0 values. A nozzle's exit patch must carry a fixed-value `T`;
    /// under a RAS model every thruster's patch must carry a fixed value of
    /// `k` and of the model's second variable. Refused by name otherwise.
    pub fn bind_jet_fields(
        &mut self,
        gpu: &Gpu,
        gm: &GpuMesh,
        t: &mut GpuScalarField,
        mut turb: Vec<(&str, &mut GpuScalarField)>,
    ) -> Result<()> {
        let face_jet = self.face_jet.clone();
        let face_ids = self.face_ids.clone();
        let is_nozzle: Vec<bool> = self.nozzles.iter().map(|n| n.is_some()).collect();
        let patch_names: Vec<String> = self.thrusters.iter().map(|t| t.patch.clone()).collect();
        let not_fixed = |field: &GpuScalarField, only_nozzles: bool| -> Result<String> {
            let kinds = gpu.download(&field.bc_kind)?;
            let mut bad: Vec<usize> = Vec::new();
            for (k, &j) in face_jet.iter().enumerate() {
                if j < 0 || (only_nozzles && !is_nozzle[j as usize]) {
                    continue;
                }
                if kinds[face_ids[k] as usize] != BcKind::FixedValue as Label && !bad.contains(&(j as usize)) {
                    bad.push(j as usize);
                }
            }
            Ok(bad.iter().map(|&j| patch_names[j].clone()).collect::<Vec<_>>().join(", "))
        };
        if self.nozzle_mode {
            let bad = not_fixed(t, true)?;
            if !bad.is_empty() {
                return Err(Error::Config(format!(
                    "-rigidBody: nozzle thruster patch(es) {bad} do not carry a fixedValue T; the plume's \
                     temperature T_j is written as a fixed value (SPEC-LIT 161.11)"
                )));
            }
            self.jet_fields[JET_T] = true;
        }
        if self.turbulent && !self.thrusters.is_empty() {
            for (name, f) in turb.iter() {
                let block = match *name {
                    "k" => JET_K,
                    "omega" => JET_OMEGA,
                    "epsilon" => JET_EPSILON,
                    "nuTilda" => JET_NU_TILDA,
                    _ => continue,
                };
                let bad = not_fixed(f, false)?;
                if !bad.is_empty() {
                    return Err(Error::Config(format!(
                        "-rigidBody: thruster patch(es) {bad} do not carry a fixedValue {name}; a jet's inflow \
                         turbulence is written as a fixed value (SPEC-LIT 161.11)"
                    )));
                }
                self.jet_fields[block] = true;
            }
        }
        let fields: Vec<(&str, &mut GpuScalarField)> = turb.iter_mut().map(|(n, f)| (*n, &mut **f)).collect();
        self.write_jet_scalars(gpu, gm, t, fields)
    }

    /// The jet's speed relative to the body this step: the plume's `U_eq`
    /// for a nozzle, `speed` for a version-1 jet that is on, else zero.
    fn jet_speed_now(&self, i: usize) -> f64 {
        match self.nozzles[i] {
            Some(_) => self.plumes[i].u_eq,
            None => {
                if self.jets_on[i] {
                    self.thrusters[i].jet_speed()
                } else {
                    0.0
                }
            }
        }
    }

    /// Write this step's per-thruster boundary scalars - the plume's `T_j`
    /// and its inflow turbulence (SPEC-LIT 161.11) - into the fields
    /// [`Self::bind_jet_fields`] bound, and re-evaluate their boundaries.
    /// Uploads `5 n_jets` numbers.
    pub fn write_jet_scalars(
        &mut self,
        gpu: &Gpu,
        gm: &GpuMesh,
        t: &mut GpuScalarField,
        turb: Vec<(&str, &mut GpuScalarField)>,
    ) -> Result<()> {
        let n = self.thrusters.len();
        if n == 0 || !self.jet_fields.iter().any(|&b| b) {
            return Ok(());
        }
        let mut vals = vec![0.0 as Scalar; JET_SCALAR_FIELDS * n];
        for i in 0..n {
            let pg = &self.patches[i];
            let length = (self.plume.mixing_length_fraction * pg.hydraulic_diameter).max(f64::MIN_POSITIVE);
            let floor = (1e-3 * self.u_ref[i]).max(1e-6);
            let jt = thrust::jet_turbulence(self.jet_speed_now(i), floor, self.plume.turbulence_intensity, length);
            // A version-1 jet's T stays the case's: NaN leaves it (SPEC-LIT 161.11).
            let t_j = match self.nozzles[i] {
                Some(_) => self.plumes[i].t_j,
                None => f64::NAN,
            };
            vals[JET_T * n + i] = t_j as Scalar;
            vals[JET_K * n + i] = jt.k as Scalar;
            vals[JET_OMEGA * n + i] = jt.omega as Scalar;
            vals[JET_EPSILON * n + i] = jt.epsilon as Scalar;
            vals[JET_NU_TILDA * n + i] = jt.nu_tilda as Scalar;
        }
        self.dev.stage_jet_scalars(gpu, &vals)?;
        if self.jet_fields[JET_T] {
            self.dev.jet_scalar(gpu, &mut t.ref_value, JET_T)?;
            field_ops::correct_boundary_conditions(gpu, &self.fldk, t, gm)?;
        }
        for (name, f) in turb {
            let block = match name {
                "k" => JET_K,
                "omega" => JET_OMEGA,
                "epsilon" => JET_EPSILON,
                "nuTilda" => JET_NU_TILDA,
                _ => continue,
            };
            if self.jet_fields[block] {
                self.dev.jet_scalar(gpu, &mut f.ref_value, block)?;
                field_ops::correct_boundary_conditions(gpu, &self.fldk, f, gm)?;
            }
        }
        Ok(())
    }

    /// The valves of this step, from the controller or the schedule, limited
    /// by the propellant left - SPEC-LIT 161.14-161.15. Returns the thrust
    /// per thruster (N).
    fn command_thrust(&mut self, t_start: f64, t_mid: f64, dt: f64) -> Vec<f64> {
        let open: Vec<f64> = self
            .thrusters
            .iter()
            .zip(&self.nozzles)
            .map(|(t, nz)| match nz {
                Some(d) if !t.closed_loop() && t.is_on(t_mid) => d.f_max,
                _ => 0.0,
            })
            .collect();
        let mut thrust = match self.controller.as_mut() {
            Some(c) => c.command(&self.body, &self.geoms, &open, t_start).thrust,
            None => open,
        };
        for (i, f) in thrust.iter_mut().enumerate() {
            *f = match self.nozzles[i] {
                Some(d) => f.clamp(0.0, d.f_max),
                None => 0.0,
            };
        }
        // The propellant: mdot = sum (F/F_max) mdot_max; a step that would
        // overdraw the tank is scaled to empty it exactly.
        if let Some(p0) = self.prop0 {
            let mdot: f64 = thrust
                .iter()
                .zip(&self.nozzles)
                .map(|(f, nz)| nz.map_or(0.0, |d| f / d.f_max * d.mdot_max))
                .sum();
            let left = (p0 - self.prop_used).max(0.0);
            let want = mdot * dt;
            let k = if want > left { if want > 0.0 { left / want } else { 0.0 } } else { 1.0 };
            if k < 1.0 {
                for f in thrust.iter_mut() {
                    *f *= k;
                }
                if !self.empty_warned {
                    self.empty_warned = true;
                    eprintln!(
                        "WARNING: -rigidBody: the propellant ({p0} kg) runs out at t = {t_start:.6} s; the \
                         thrust is zero from here on (SPEC-LIT 161.15)"
                    );
                }
            }
            self.prop_used = (self.prop_used + k * want).min(p0);
        }
        thrust
    }

    /// Before the step's solve, after the time step is opened: advance the
    /// body over `dt` with the last force, move the points on the device,
    /// advance the mesh, and write the body walls' and jets' velocity and
    /// flux - SPEC-LIT 161.6. `t_new` is the step's end time.
    ///
    /// With nozzle thrusters (SPEC-LIT 161.15): the valves are set from the
    /// state at the step's start, the propellant is drawn, and the body
    /// advances with the thrust held in body axes. During `release` the body
    /// moves kinematically and every jet is shut.
    pub fn pre_step(&mut self, gpu: &Gpu, gm: &GpuMesh, s: &mut Simple<'_>, t_new: f64, dt: f64) -> Result<()> {
        let t_start = t_new - dt;
        let t_mid = t_new - 0.5 * dt;
        self.releasing = t_mid < self.release;
        if self.releasing {
            self.body.advance_kinematic(dt);
            self.thrust.iter_mut().for_each(|f| *f = 0.0);
            self.thrust_body = ([0.0; 3], [0.0; 3]);
        } else if self.nozzle_mode {
            let thrust = self.command_thrust(t_start, t_mid, dt);
            let (fb, mb) = control::thrust_wrench_body(&self.geoms, &thrust);
            self.body.advance_thrust(dt, self.wrench.force(), self.wrench.moment, fb, mb);
            self.body.mass = self.mass0 - self.prop_used;
            self.thrust = thrust;
            self.thrust_body = (fb, mb);
        } else {
            self.body.advance(dt, self.wrench.force(), self.wrench.moment);
        }
        if !(finite(self.body.state.x) && finite(self.body.state.w) && self.body.state.q.iter().all(|v| v.is_finite())) {
            return Err(Error::Diverged {
                iteration: 0,
                what: format!("the rigid body's state went non-finite at t = {t_new}"),
            });
        }
        if let Some(c) = &self.controller {
            self.reference = c.attitude.as_ref().map(|(p, _, _)| p.at(t_new));
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
        let r = self.body.rotation();
        let mut jets = Vec::with_capacity(self.thrusters.len());
        for i in 0..self.thrusters.len() {
            let t = &self.thrusters[i];
            match self.nozzles[i] {
                Some(d) => {
                    let pg = &self.patches[i];
                    self.plumes[i] = thrust::equivalent_plume(self.thrust[i], pg.area, d.t0, d.gamma, d.r, self.p_amb);
                    self.valve[i] = self.thrust[i] / d.f_max;
                    self.jets_on[i] = self.thrust[i] > 0.0;
                    jets.push(scale(mat_vec(&r, t.unit_direction()), self.plumes[i].u_eq));
                }
                None => {
                    let on = !self.releasing && t.is_on(t_mid);
                    self.jets_on[i] = on;
                    self.valve[i] = if on { 1.0 } else { 0.0 };
                    jets.push(if on { scale(mat_vec(&r, t.unit_direction()), t.jet_speed()) } else { [0.0; 3] });
                }
            }
        }
        self.dev.wall_velocity(gpu, gm, &mut s.u_mut().ref_value, &self.body, &jets, false)?;
        field_ops::correct_boundary_conditions_vector(gpu, &self.fldk, s.u_mut(), gm)?;
        self.dev.scatter_flux(gpu, &mut s.phi_mut().bf)
    }

    /// The thrust's force and moment in world axes at the current
    /// orientation.
    fn thrust_world(&self) -> (V3, V3) {
        let r = self.body.rotation();
        (mat_vec(&r, self.thrust_body.0), mat_vec(&r, self.thrust_body.1))
    }

    /// After the step's solve: the force and moment the next step will use,
    /// the CSV row and the progress line every `write.every` steps.
    /// `nut_b` is the eddy viscosity's boundary values under a RAS model.
    /// Returns `true` when the body has touched the ground and the file
    /// asked the run to stop there (SPEC-LIT 161.15).
    #[allow(clippy::too_many_arguments)]
    pub fn post_step(
        &mut self,
        gpu: &Gpu,
        gm: &GpuMesh,
        s: &Simple<'_>,
        nut_b: Option<&DevBuf<Scalar>>,
        step: usize,
        n_steps: usize,
        end_time: f64,
    ) -> Result<bool> {
        self.wrench = if self.nozzle_mode || self.turbulent {
            self.dev.force_moment_v2(
                gpu,
                gm,
                &s.p().bf,
                &s.u().f,
                &s.u().bf,
                &s.convective_flux().bf,
                if self.turbulent { nut_b } else { None },
                self.rho,
                self.nu,
                self.g_hyd,
                &self.body,
            )?
        } else {
            self.dev.force_moment(
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
            )?
        };
        let g_mag = {
            let g = norm(self.body.gravity);
            if g > 0.0 {
                g
            } else {
                9.80665
            }
        };
        let (ft, _) = self.thrust_world();
        self.load_g = norm(add(self.wrench.force(), ft)) / (self.body.mass * g_mag);
        if !self.releasing {
            self.peak_load_g = self.peak_load_g.max(self.load_g);
            self.peak_thrust_g = self.peak_thrust_g.max(norm(ft) / (self.body.mass * g_mag));
        }
        let mut stop = false;
        if let Some(g) = &self.ground {
            self.h_contact = control::lowest_point(&self.body, &g.contact_points).map(|z| z - g.z);
            if g.stop && self.h_contact.is_some_and(|h| h <= 0.0) {
                stop = true;
                let r = self.body.rotation();
                self.touchdown = Some(Touchdown {
                    t: self.body.state.t,
                    velocity: self.body.state.v,
                    rpy_deg: self.body.euler_zyx_deg(),
                    upright: control::body_z_world(&r)[2],
                    peak_load_g: self.peak_load_g,
                    peak_thrust_g: self.peak_thrust_g,
                    propellant_used: self.prop_used,
                });
            }
        }
        let last = step + 1 == n_steps || stop;
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
            let extra = if self.csv_v2 {
                let mut x = format!(" | load {:.3} g", self.load_g);
                if let Some(h) = self.h_contact {
                    x.push_str(&format!(" h {h:.3} m"));
                }
                if let Some(p0) = self.prop0 {
                    x.push_str(&format!(" prop {:.2} kg", p0 - self.prop_used));
                }
                if self.nozzle_mode {
                    let us: Vec<String> = self.valve.iter().map(|u| format!("{u:.2}")).collect();
                    x.push_str(&format!(" u [{}]", us.join(" ")));
                }
                if self.releasing {
                    x.push_str(" (release hold)");
                }
                x
            } else {
                String::new()
            };
            println!(
                "rigidBody {:>5.1}% step {}/{} t {:.5} | X ({:.4} {:.4} {:.4}) V ({:.3} {:.3} {:.3}) | \
                 rotY {:.2} deg rpy ({:.1} {:.1} {:.1}) w ({:.3} {:.3} {:.3}) rad/s | F ({:.3e} {:.3e} {:.3e}) \
                 My {:.3e} | jets {} | {:.3} s/step{}",
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
                extra,
            );
        }
        if last {
            self.csv.flush().map_err(|source| Error::Io { path: self.csv_path.display().to_string(), source })?;
        }
        if stop {
            if let Some(td) = &self.touchdown {
                println!("{}", self.touchdown_line(td));
            }
        }
        Ok(stop)
    }

    /// The one line a touchdown ends the run with (SPEC-LIT 161.15).
    pub fn touchdown_line(&self, td: &Touchdown) -> String {
        format!(
            "run ended: touchdown at t = {:.5} s: velocity ({:.4} {:.4} {:.4}) m/s (|v_z| {:.4} m/s), attitude \
             rpy ({:.2} {:.2} {:.2}) deg (body z . world z {:.4}), peak load {:.3} g (thrust alone {:.3} g), propellant used {:.3} kg{}",
            td.t,
            td.velocity[0],
            td.velocity[1],
            td.velocity[2],
            td.velocity[2].abs(),
            td.rpy_deg[0],
            td.rpy_deg[1],
            td.rpy_deg[2],
            td.upright,
            td.peak_load_g,
            td.peak_thrust_g,
            td.propellant_used,
            match self.prop0 {
                Some(p) => format!(" of {p} kg"),
                None => " (no propellant block)".to_string(),
            }
        )
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
        let mut extra = String::new();
        if self.csv_v2 {
            for i in 0..self.thrusters.len() {
                let (u, fi, ueq, meq) = match self.nozzles[i] {
                    Some(_) => (self.valve[i], self.thrust[i], self.plumes[i].u_eq, self.plumes[i].mach_eq),
                    None => {
                        let sp = if self.jets_on[i] { self.thrusters[i].jet_speed() } else { 0.0 };
                        (self.valve[i], self.rho * sp * sp * self.patches[i].area, sp, f64::NAN)
                    }
                };
                extra.push_str(&format!(",{u:.6},{fi:.9e},{ueq:.9e},{meq:.6}"));
            }
            let prop = self.prop0.map_or(f64::NAN, |p0| p0 - self.prop_used);
            let h = self.h_contact.unwrap_or(f64::NAN);
            let rr = self.reference.map_or([f64::NAN; 3], |r| control::rpy_deg_of_quat(r.q));
            let q = st.q;
            extra.push_str(&format!(
                ",{prop:.9e},{:.9e},{h:.9e},{:.6},{:.6},{:.6},{:.12e},{:.12e},{:.12e},{:.12e}",
                self.load_g, rr[0], rr[1], rr[2], q[0], q[1], q[2], q[3]
            ));
        }
        writeln!(
            self.csv,
            "{},{:.9e},{:.12e},{:.12e},{:.12e},{:.9e},{:.9e},{:.9e},{:.6},{:.6},{:.6},{:.6},{:.9e},{:.9e},{:.9e},\
             {:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{}{}",
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
            jets,
            extra
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
#[cfg(test)]
mod tests_v2;
