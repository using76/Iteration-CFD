// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! The thruster controller of SPEC-LIT 161.14: the attitude reference, the
//! PD on the quaternion error, the descent law and the bounded
//! least-squares allocation of the commanded wrench to the thrusters.
//!
//! Written from:
//!   ofgpu SPEC-LIT.md 161.14 (the reference, the law, the allocation and
//!     their stated limits)
//!   T. Flash, N. Hogan, "The coordination of arm movements: an
//!     experimentally confirmed mathematical model", J. Neurosci. 5 (1985)
//!     1688-1703 - the minimum-jerk profile s = 10 tau^3 - 15 tau^4 + 6 tau^5
//!   J. Diebel, *Representing Attitude: Euler Angles, Unit Quaternions, and
//!     Rotation Vectors*, Stanford University technical report (2006) - the
//!     quaternion product, conjugate, rotation vector and z-y-x angles
//!   B. Wie, *Space Vehicle Dynamics and Control*, 2nd ed. (AIAA, 2008),
//!     ch. 7 - the quaternion-error feedback with rate feedback and the
//!     computed-torque (gyroscopic) term
//!   P. B. Stark, R. L. Parker, "Bounded-variable least-squares: an
//!     algorithm and applications", Comput. Stat. 10 (1995) 129-141 - the
//!     active-set method for least squares with bounds on the variables
//!   G. H. Golub, C. F. Van Loan, *Matrix Computations*, 4th ed. (Johns
//!     Hopkins, 2013), sec. 5.2 and 5.3 - Householder QR and the
//!     least-squares solve on it
//! No GPL-licensed source was consulted.
//!
//! Provenance: ORIGINAL.
//!
//! Host only, f64 in both builds, evaluated once per time step from the
//! state the integrator already holds. Nothing here launches a kernel.

use crate::error::{Error, Result};

use super::{add, cross, mat_t_vec, mat_vec, norm, quat_exp, quat_mul, quat_to_matrix, scale, sub, RigidBody, M3, V3};

/// The unit quaternion of `R = Rz(yaw) Ry(pitch) Rx(roll)`, angles in degrees.
pub fn quat_from_rpy_deg(rpy: V3) -> [f64; 4] {
    let (hr, hp, hy) = (0.5 * rpy[0].to_radians(), 0.5 * rpy[1].to_radians(), 0.5 * rpy[2].to_radians());
    let qx = [hr.cos(), hr.sin(), 0.0, 0.0];
    let qy = [hp.cos(), 0.0, hp.sin(), 0.0];
    let qz = [hy.cos(), 0.0, 0.0, hy.sin()];
    normalise(quat_mul(quat_mul(qz, qy), qx))
}

/// The z-y-x angles of a unit quaternion in degrees, `[roll, pitch, yaw]`.
pub fn rpy_deg_of_quat(q: [f64; 4]) -> V3 {
    let r = quat_to_matrix(q);
    [
        r[2][1].atan2(r[2][2]).to_degrees(),
        (-r[2][0]).clamp(-1.0, 1.0).asin().to_degrees(),
        r[1][0].atan2(r[0][0]).to_degrees(),
    ]
}

pub(crate) fn normalise(q: [f64; 4]) -> [f64; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

pub(crate) fn conj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

/// The rotation vector (axis times angle) of a unit quaternion, taken on its
/// short side: `q` and `-q` are the same rotation, and the one with `w >= 0`
/// turns through at most 180 degrees.
pub fn rotation_vector(q: [f64; 4]) -> V3 {
    let q = if q[0] < 0.0 { [-q[0], -q[1], -q[2], -q[3]] } else { q };
    let v = [q[1], q[2], q[3]];
    let s = norm(v);
    if s == 0.0 {
        return [0.0; 3];
    }
    scale(v, 2.0 * s.atan2(q[0]) / s)
}

/// The minimum-jerk profile `s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5` on
/// `[0, 1]` (clamped outside), with `ds/dtau` and `d2s/dtau2`.
pub fn min_jerk(tau: f64) -> (f64, f64, f64) {
    let t = tau.clamp(0.0, 1.0);
    let (t2, t3) = (t * t, t * t * t);
    (
        t3 * (10.0 - 15.0 * t + 6.0 * t2),
        t2 * (30.0 - 60.0 * t + 30.0 * t2),
        t * (60.0 - 180.0 * t + 120.0 * t2),
    )
}

/// The attitude reference at one instant: the orientation (reference to
/// world), and its angular velocity and acceleration in its OWN axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttitudeReference {
    pub q: [f64; 4],
    pub w: V3,
    pub a: V3,
}

/// The waypoints of SPEC-LIT 161.14, each `(t, q)`, held before the first
/// and after the last.
#[derive(Debug, Clone, PartialEq)]
pub struct AttitudeProfile {
    pub points: Vec<(f64, [f64; 4])>,
}

impl AttitudeProfile {
    /// From `(t, [roll, pitch, yaw] deg)`, `t` strictly increasing.
    pub fn new(waypoints: &[(f64, V3)]) -> Result<Self> {
        if waypoints.is_empty() {
            return Err(Error::Config(
                "-rigidBody: control.attitude.waypoints is empty (SPEC-LIT 161.14)".to_string(),
            ));
        }
        for (i, w) in waypoints.iter().enumerate() {
            if !(w.0.is_finite() && w.1.iter().all(|v| v.is_finite())) {
                return Err(Error::Config(format!(
                    "-rigidBody: control.attitude.waypoints[{i}] is not finite (SPEC-LIT 161.14)"
                )));
            }
            if i > 0 && !(w.0 > waypoints[i - 1].0) {
                return Err(Error::Config(format!(
                    "-rigidBody: control.attitude.waypoints[{i}].t {} does not follow {}; the times must increase \
                     (SPEC-LIT 161.14)",
                    w.0,
                    waypoints[i - 1].0
                )));
            }
        }
        Ok(Self { points: waypoints.iter().map(|&(t, rpy)| (t, quat_from_rpy_deg(rpy))).collect() })
    }

    /// The reference at `t`: on segment `[t_k, t_{k+1}]` the orientation turns
    /// about the fixed axis of the SHORTEST rotation from `q_k` to `q_{k+1}`,
    /// through the angle `theta s(tau)`, `tau = (t - t_k)/T`; its rate and
    /// acceleration are `phi s'(tau)/T` and `phi s''(tau)/T^2` with `phi` the
    /// rotation vector of `q_k^-1 q_{k+1}` - the same vector in the
    /// reference's own axes all along the segment, because a rotation leaves
    /// its axis where it is.
    pub fn at(&self, t: f64) -> AttitudeReference {
        let p = &self.points;
        let hold = |q: [f64; 4]| AttitudeReference { q, w: [0.0; 3], a: [0.0; 3] };
        if t <= p[0].0 {
            return hold(p[0].1);
        }
        let last = p.len() - 1;
        if t >= p[last].0 {
            return hold(p[last].1);
        }
        let k = p.iter().rposition(|w| w.0 <= t).unwrap_or(0).min(last - 1);
        let (t0, q0) = p[k];
        let (t1, q1) = p[k + 1];
        let span = t1 - t0;
        let phi = rotation_vector(quat_mul(conj(q0), q1));
        let (s, sd, sdd) = min_jerk((t - t0) / span);
        AttitudeReference {
            q: normalise(quat_mul(q0, quat_exp(scale(phi, s)))),
            w: scale(phi, sd / span),
            a: scale(phi, sdd / (span * span)),
        }
    }
}

/// The angular acceleration (body axes) the PD on the quaternion error asks
/// for, with the reference's feed-forward - SPEC-LIT 161.14:
///
/// ```text
/// q_e = q_ref^-1 q (short side),  phi_e its rotation vector,  R_e = R(q_e)
/// w_e = w - R_e^T w_ref
/// alpha = R_e^T a_ref - 2 zeta w_n w_e - w_n^2 phi_e
/// ```
pub fn attitude_acceleration(q: [f64; 4], w: V3, r: &AttitudeReference, wn: f64, zeta: f64) -> V3 {
    let qe = quat_mul(conj(r.q), q);
    let qe = if qe[0] < 0.0 { [-qe[0], -qe[1], -qe[2], -qe[3]] } else { qe };
    let phi = rotation_vector(qe);
    let re = quat_to_matrix(qe);
    let we = sub(w, mat_t_vec(&re, r.w));
    sub(sub(mat_t_vec(&re, r.a), scale(we, 2.0 * zeta * wn)), scale(phi, wn * wn))
}

/// The descent law of SPEC-LIT 161.14.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DescentLaw {
    pub ground_z: f64,
    pub touchdown_speed: f64,
    pub brake_accel: f64,
    pub velocity_gain: f64,
    pub min_upright: f64,
}

impl DescentLaw {
    /// `v_ref(h) = -sqrt(v_td^2 + 2 a_b h)`.
    pub fn v_ref(&self, h: f64) -> f64 {
        -(self.touchdown_speed * self.touchdown_speed + 2.0 * self.brake_accel * h.max(0.0)).sqrt()
    }

    /// `F_z` in body axes - SPEC-LIT 161.14, with the contract's section 9
    /// change of 2026-10-17. Braking starts the first time the body falls
    /// faster than `v_ref` and then stays on (`braking` latches); while it
    /// is on and `uz = body z . world z >= minUpright`,
    ///
    /// ```text
    /// a_ff  = dv_ref/dt along the fall = a_b v_z / v_ref      (= a_b on the reference)
    /// a_cmd = clamp(g + a_ff + K (v_ref - v_z), 0, g + 1.3 a_b)
    /// F_z   = m a_cmd / uz
    /// ```
    ///
    /// and zero otherwise. Without the feed-forward the proportional law
    /// trails a reference that decelerates at `a_b` by `a_b/K`, and touches
    /// down at several times `v_td` (161.14 records the measurement).
    pub fn force_z_body(&self, m: f64, g: f64, h: f64, vz: f64, uz: f64, braking: &mut bool) -> f64 {
        let vr = self.v_ref(h);
        if !*braking && vz < vr {
            *braking = true;
        }
        if !*braking || !(uz >= self.min_upright) || !(uz > 0.0) {
            return 0.0;
        }
        let a_max = g + 1.3 * self.brake_accel;
        let a_ff = if vr < 0.0 { (self.brake_accel * vz / vr).clamp(0.0, a_max) } else { self.brake_accel };
        let a = (g + a_ff + self.velocity_gain * (vr - vz)).clamp(0.0, a_max);
        m * a / uz
    }
}

/// One thruster as the allocation sees it: where it sits relative to the
/// centre of mass and which way its gas leaves, both in body axes, and its
/// full-valve thrust.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThrusterGeom {
    pub r_body: V3,
    pub dir_body: V3,
    pub f_max: f64,
    /// The controller commands it (no `on` schedule).
    pub closed_loop: bool,
}

impl ThrusterGeom {
    /// The body-axes force and moment of a unit thrust: `f = -d`, `m = r x f`.
    pub fn unit_wrench(&self) -> (V3, V3) {
        let f = scale(self.dir_body, -1.0);
        (f, cross(self.r_body, f))
    }

    /// `[F_z, M_x, M_y]` per newton of thrust - one column of the allocation.
    pub fn column(&self) -> [f64; 3] {
        let (f, m) = self.unit_wrench();
        [f[2], m[0], m[1]]
    }
}

/// The weight of the two moment rows against the force row, after both are
/// put in newtons by the largest thruster arm: the moment's priority.
pub const MOMENT_PRIORITY: f64 = 1000.0;
/// The first stage's regularisation, relative to its largest weighted
/// column: it only makes the problem strictly convex, far below anything a
/// residual can feel.
pub const ALLOCATION_RIDGE: f64 = 1e-12;
/// The second stage's regularisation, relative to its largest column: large
/// enough that the thrusts' norm is resolved above the round-off of the
/// residual, small enough to leave the first stage's wrench to about 1e-10.
pub const LEAST_NORM_RIDGE: f64 = 1e-5;

/// The allocation of SPEC-LIT 161.14, in two stages. First the wrench the
/// thrusters can give, moment first: [`bounded_least_squares`] with the
/// moment rows in newtons (divided by the largest arm `arm`) and weighted
/// [`MOMENT_PRIORITY`]. Then the smallest thrusts that give that same
/// wrench: the same method, unweighted, its target the first stage's
/// wrench, with the ridge [`LEAST_NORM_RIDGE`]. A one-stage ridge small
/// enough not to move the moment-first residual is too small to be seen
/// above that residual's round-off, and leaves the split among thrusters
/// that give the same wrench to the path the active set happened to take.
pub fn allocate(cols: &[[f64; 3]], b: [f64; 3], arm: f64, lo: &[f64], hi: &[f64]) -> Vec<f64> {
    let arm = if arm > 0.0 { arm } else { 1.0 };
    let w1 = [1.0, MOMENT_PRIORITY / arm, MOMENT_PRIORITY / arm];
    let x1 = bounded_least_squares(cols, b, w1, lo, hi, ALLOCATION_RIDGE);
    let mut got = [0.0; 3];
    for (c, xj) in cols.iter().zip(&x1) {
        for k in 0..3 {
            got[k] += c[k] * xj;
        }
    }
    let w2 = [1.0, 1.0 / arm, 1.0 / arm];
    bounded_least_squares(cols, got, w2, lo, hi, LEAST_NORM_RIDGE)
}

/// `min ||W (A x - b)||^2 + eps ||x||^2` over `lo <= x <= hi`, `A`'s columns
/// `cols` - the bounded least squares of SPEC-LIT 161.14, by the active-set
/// method of Stark & Parker on the augmented system `[W A; sqrt(eps) I] x =
/// [W b; 0]`, each free-set solve a Householder QR (never the normal
/// equations, whose condition is the square). `eps > 0` makes the problem
/// strictly convex, so the method ends, at the unique minimiser.
pub fn bounded_least_squares(cols: &[[f64; 3]], b: [f64; 3], w: [f64; 3], lo: &[f64], hi: &[f64], ridge: f64) -> Vec<f64> {
    let n = cols.len();
    if n == 0 {
        return Vec::new();
    }
    let rows = 3 + n;
    let colmax = cols
        .iter()
        .map(|c| ((w[0] * c[0]).powi(2) + (w[1] * c[1]).powi(2) + (w[2] * c[2]).powi(2)).sqrt())
        .fold(0.0f64, f64::max)
        .max(f64::MIN_POSITIVE);
    let se = ridge * colmax;
    // C, row major, and d.
    let mut c = vec![vec![0.0; n]; rows];
    for (j, col) in cols.iter().enumerate() {
        for r in 0..3 {
            c[r][j] = w[r] * col[r];
        }
        c[3 + j][j] = se;
    }
    let mut d = vec![0.0; rows];
    for r in 0..3 {
        d[r] = w[r] * b[r];
    }

    #[derive(Clone, Copy, PartialEq)]
    enum St {
        Lo,
        Hi,
        Free,
    }
    let mut x: Vec<f64> = (0..n).map(|j| 0.0f64.clamp(lo[j], hi[j])).collect();
    let mut st: Vec<St> = (0..n)
        .map(|j| if x[j] <= lo[j] { St::Lo } else if x[j] >= hi[j] { St::Hi } else { St::Free })
        .collect();
    let scale_g = d.iter().map(|v| v.abs()).fold(0.0f64, f64::max).max(1.0) * colmax;
    let tol = 1e-14 * scale_g;

    let grad = |x: &[f64]| -> Vec<f64> {
        let mut res = vec![0.0; rows];
        for r in 0..rows {
            res[r] = (0..n).map(|j| c[r][j] * x[j]).sum::<f64>() - d[r];
        }
        (0..n).map(|j| (0..rows).map(|r| c[r][j] * res[r]).sum()).collect()
    };

    for _outer in 0..(20 * n + 50) {
        // The free-set minimiser, stepping back to the first bound it crosses.
        for _inner in 0..(n + 2) {
            let free: Vec<usize> = (0..n).filter(|&j| st[j] == St::Free).collect();
            if free.is_empty() {
                break;
            }
            let mut rhs = d.clone();
            for j in 0..n {
                if st[j] != St::Free {
                    for r in 0..rows {
                        rhs[r] -= c[r][j] * x[j];
                    }
                }
            }
            let sub_c: Vec<Vec<f64>> = (0..rows).map(|r| free.iter().map(|&j| c[r][j]).collect()).collect();
            let z = householder_lsq(&sub_c, &rhs);
            let inside = free.iter().zip(&z).all(|(&j, &zj)| zj >= lo[j] && zj <= hi[j]);
            if inside {
                for (&j, &zj) in free.iter().zip(&z) {
                    x[j] = zj;
                }
                break;
            }
            let mut alpha = 1.0f64;
            for (&j, &zj) in free.iter().zip(&z) {
                if zj < lo[j] {
                    let den = x[j] - zj;
                    if den > 0.0 {
                        alpha = alpha.min((x[j] - lo[j]) / den);
                    }
                } else if zj > hi[j] {
                    let den = zj - x[j];
                    if den > 0.0 {
                        alpha = alpha.min((hi[j] - x[j]) / den);
                    }
                }
            }
            let alpha = alpha.clamp(0.0, 1.0);
            for (&j, &zj) in free.iter().zip(&z) {
                x[j] += alpha * (zj - x[j]);
                let span = (hi[j] - lo[j]).abs().max(1.0);
                if x[j] <= lo[j] + 1e-14 * span && zj < lo[j] {
                    x[j] = lo[j];
                    st[j] = St::Lo;
                } else if x[j] >= hi[j] - 1e-14 * span && zj > hi[j] {
                    x[j] = hi[j];
                    st[j] = St::Hi;
                }
            }
        }
        // The multipliers of the fixed variables: free the worst violator.
        let g = grad(&x);
        let mut worst: Option<(usize, f64)> = None;
        for j in 0..n {
            let v = match st[j] {
                St::Lo if g[j] < -tol => -g[j],
                St::Hi if g[j] > tol => g[j],
                _ => continue,
            };
            if worst.is_none_or(|(_, w)| v > w) {
                worst = Some((j, v));
            }
        }
        match worst {
            Some((j, _)) => st[j] = St::Free,
            None => break,
        }
    }
    x
}

/// The least-squares solution of `A z = b` (`A` row major, full column rank)
/// by Householder QR (Golub & Van Loan 5.2.1, 5.3.3).
fn householder_lsq(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let m = a.len();
    let k = if m == 0 { 0 } else { a[0].len() };
    let mut r: Vec<Vec<f64>> = a.to_vec();
    let mut y: Vec<f64> = b.to_vec();
    for col in 0..k.min(m) {
        let alpha = (col..m).map(|i| r[i][col] * r[i][col]).sum::<f64>().sqrt();
        if alpha == 0.0 {
            continue;
        }
        let sign = if r[col][col] >= 0.0 { 1.0 } else { -1.0 };
        let mut v: Vec<f64> = (col..m).map(|i| r[i][col]).collect();
        v[0] += sign * alpha;
        let vv: f64 = v.iter().map(|t| t * t).sum();
        if vv == 0.0 {
            continue;
        }
        for j in col..k {
            let s: f64 = (col..m).map(|i| v[i - col] * r[i][j]).sum::<f64>() * 2.0 / vv;
            for i in col..m {
                r[i][j] -= s * v[i - col];
            }
        }
        let s: f64 = (col..m).map(|i| v[i - col] * y[i]).sum::<f64>() * 2.0 / vv;
        for i in col..m {
            y[i] -= s * v[i - col];
        }
    }
    let mut z = vec![0.0; k];
    for i in (0..k).rev() {
        let s: f64 = ((i + 1)..k).map(|j| r[i][j] * z[j]).sum();
        z[i] = if r[i][i] != 0.0 { (y[i] - s) / r[i][i] } else { 0.0 };
    }
    z
}

/// The controller of SPEC-LIT 161.14.
#[derive(Debug, Clone)]
pub struct Controller {
    /// The waypoints and the PD's natural frequency and damping.
    pub attitude: Option<(AttitudeProfile, f64, f64)>,
    pub descent: Option<DescentLaw>,
    /// Body-frame (rest-mesh) points whose lowest world z is the height `h`.
    pub contact: Vec<V3>,
    /// The descent law's latch: braking has begun.
    pub braking: bool,
}

/// What the controller decided for one step.
#[derive(Debug, Clone, PartialEq)]
pub struct Command {
    /// Thrust per thruster (N): the open-loop thrusters' unchanged.
    pub thrust: Vec<f64>,
    pub reference: Option<AttitudeReference>,
    /// The wrench asked of the closed-loop thrusters, `[F_z, M_x, M_y]` body.
    pub wrench: [f64; 3],
    /// The height of the lowest contact point over the descent law's ground.
    pub h: Option<f64>,
}

/// The lowest world z of `contact` (rest frame) on `body` - `None` without
/// contact points.
pub fn lowest_point(body: &RigidBody, contact: &[V3]) -> Option<f64> {
    contact.iter().map(|&p| body.map_point(p)[2]).reduce(f64::min)
}

impl Controller {
    /// The thrust of every thruster at time `t` for the body's current state.
    /// `open_thrust[i]` is the scheduled thrust of an open-loop thruster;
    /// the closed-loop ones share what is left of the wrench.
    pub fn command(&mut self, body: &RigidBody, thr: &[ThrusterGeom], open_thrust: &[f64], t: f64) -> Command {
        let st = &body.state;
        let r = body.rotation();
        let j = body.inertia_matrix();
        let mut moment = [0.0; 3];
        let mut reference = None;
        if let Some((prof, wn, zeta)) = &self.attitude {
            let rf = prof.at(t);
            let alpha = attitude_acceleration(st.q, st.w, &rf, *wn, *zeta);
            moment = add(mat_vec(&j, alpha), cross(st.w, mat_vec(&j, st.w)));
            reference = Some(rf);
        }
        let h = lowest_point(body, &self.contact).map(|z| z - self.descent.map_or(0.0, |d| d.ground_z));
        let mut fz = 0.0;
        if let (Some(d), Some(h)) = (&self.descent, h) {
            let uz = r[2][2];
            fz = d.force_z_body(body.mass, norm(body.gravity), h, st.v[2], uz, &mut self.braking);
        }
        let mut want = [fz, moment[0], moment[1]];
        for (i, th) in thr.iter().enumerate() {
            if !th.closed_loop {
                let c = th.column();
                for k in 0..3 {
                    want[k] -= c[k] * open_thrust[i];
                }
            }
        }
        let idx: Vec<usize> = (0..thr.len()).filter(|&i| thr[i].closed_loop).collect();
        let mut thrust: Vec<f64> = thr.iter().enumerate().map(|(i, t)| if t.closed_loop { 0.0 } else { open_thrust[i] }).collect();
        if !idx.is_empty() {
            let arm = idx.iter().map(|&i| norm(thr[i].r_body)).fold(0.0f64, f64::max);
            let cols: Vec<[f64; 3]> = idx.iter().map(|&i| thr[i].column()).collect();
            let lo = vec![0.0; idx.len()];
            let hi: Vec<f64> = idx.iter().map(|&i| thr[i].f_max).collect();
            let x = allocate(&cols, want, arm, &lo, &hi);
            for (k, &i) in idx.iter().enumerate() {
                thrust[i] = x[k];
            }
        }
        Command { thrust, reference, wrench: want, h }
    }
}

/// The body-axes force and moment of the thrusts `thrust` - what the nozzle
/// model applies at each exit patch's centroid (SPEC-LIT 161.12).
pub fn thrust_wrench_body(thr: &[ThrusterGeom], thrust: &[f64]) -> (V3, V3) {
    let mut f = [0.0; 3];
    let mut m = [0.0; 3];
    for (t, &tt) in thr.iter().zip(thrust) {
        let (uf, um) = t.unit_wrench();
        f = add(f, scale(uf, tt));
        m = add(m, scale(um, tt));
    }
    (f, m)
}

/// The body's z axis in world axes, `R e_z`; its z component is the
/// uprightness `body z . world z` the descent law reads.
pub fn body_z_world(r: &M3) -> V3 {
    [r[0][2], r[1][2], r[2][2]]
}
