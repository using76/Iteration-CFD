// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! The segregated thermo-elastic outer loop on the device: the Picard
//! iteration that drives the displacement operator to its fixed point, with
//! Aitken delta-squared relaxation, the observed contraction printed beside
//! its prediction on every run, and divergence named when it happens - the
//! loop `src/solid/prototype.rs` measured host-side, transcribed.
//!
//! Written from:
//!   U. Küttler, W. A. Wall, *Comput. Mech.* 43 (2008) 61-72, DOI
//!     10.1007/s00466-008-0255-5 - Aitken delta-squared dynamic relaxation of
//!     a partitioned fixed point, §3.2, in the vector form used here
//!   H. Jasak, H. G. Weller, *Int. J. Numer. Methods Eng.* 48 (2000) 267-287,
//!     DOI 10.1002/(SICI)1097-0207(20000520)48:2<267::AID-NME884>3.0.CO;2-Q -
//!     the `(2 mu + lambda)` implicit split whose deferred part is what the
//!     loop contracts on
//!   I. Demirdžić, S. Muzaferija, *Int. J. Numer. Methods Eng.* 37 (1994)
//!     3751-3766, DOI 10.1002/nme.1620372110 - the segregated formulation
//!     the loop iterates
//!   P. Cardiff, I. Demirdžić, *Arch. Comput. Methods Eng.* 28 (2021)
//!     3721-3780, DOI 10.1007/s11831-020-09523-0 - the review; the
//!     degradation of the segregated loop as `nu -> 0.5` is a published
//!     observation, and the refusal threshold drawn from this loop is read
//!     off the sweep, not off the algebra
//!   P. Cardiff, Ž. Tuković, H. Jasak, A. Ivanković, *Comput. Struct.* 175
//!     (2016) 100-122, DOI 10.1016/j.compstruc.2016.07.004 - block-coupled
//!     elasticity, the route a near-incompressible case must take; named and
//!     NOT implemented
//!   S. P. Timoshenko, J. N. Goodier, *Theory of Elasticity*, 3rd ed.,
//!     McGraw-Hill (1970) ch. 3 - the end-loaded cantilever, the closed form
//!     the validation binary's cantilever gate measures against
//!   `docs/09-thermal-structural-plan.md` §F.1a - the sweep whose counts and
//!     contractions the twin test is held to
//!   ofgpu `SPEC-LIT.md` §1, §2.4, §3.2, §3.5, §4, §8.2, §8.4, §21, §46.4,
//!     §69, §94
//!
//! OpenFOAM and solids4foam are GPL and were not opened; no solid-mechanics
//! solver of any licence was consulted. No GPL-licensed source was consulted.
//!
//! # Design: what runs where
//!
//! The three linear solves, the boundary sub-passes and the source assembly
//! are the displacement operator's device kernels. The five vector
//! operations of the outer loop itself - `r = F(u) - u`, its norm, the two
//! Aitken dot products, `u += omega r` - run on the HOST, on a downloaded
//! copy of `F(u)` and on the loop's own copy of `u`, in the prototype's
//! summation order (sequential sums over cells, `Vec3::dot` and
//! `Vec3::mag_sqr` per cell, exactly as the prototype writes them). That
//! costs one download and one upload of the displacement per outer
//! iteration - well under a millisecond beside three conjugate-gradient
//! solves - and buys a twin test with no second reduction order in the way.
//! The device reduction - norm, dot products and update as kernels, omega
//! chosen without a download - is the later optimisation.
//!
//! # The loop
//!
//! ```text
//! u = u_0 (zero, or what the caller left in the displacement field)
//! omega = 1;  first = 0;  prev_norm = 0
//! for k in 1..=max_outer:
//!     next = F(u)     # boundary_passes sub-passes, source, three solves
//!     r    = next - u ;  norm = ||r||_2 = sqrt(sum_c |r_c|^2)    (host)
//!     norms.push(norm);  iterations = k
//!     if !norm.is_finite() || (k > 1 && norm > first * 1e6):
//!         Err(Diverged { iteration: k, what })     # named, never a flag
//!     if k == 1:  first = norm;  omega = 1
//!     else:
//!         ratios.push(norm / prev_norm  if prev_norm > 0 else 0)
//!         if aitken:  omega = aitken_omega(omega, r_prev, r)
//!         if norm <= first * 10^(-decades):
//!             converged = true;  u += omega * r;  break
//!     if aitken:  omegas.push(omega)
//!     u += omega * r
//!     r_prev = r;  prev_norm = norm
//! then: correct_boundary once - the gradient the caller reads stress from
//!       belongs to the accepted u;  observed = geometric mean of the last
//!       10 ratios;  motion_ratio = max|u| / min_c V_c^(1/3);  print one line
//! ```
//!
//! The stopping quantity is `r = F(u) - u`, the residual of the fixed-point
//! map, NOT the relaxed increment applied - with Aitken the two differ, and
//! comparing them would be calling the difference an improvement.

use crate::device::Gpu;
use crate::error::{Error, Result};
use crate::solid::displacement::Displacement;
use crate::{Scalar, Vec3};

use super::motion_ratio_of;

// ==========================================================================
//  Controls and report
// ==========================================================================

/// The outer loop's controls: the relaxation, the decades of fixed-point
/// residual to drop, the cap, and the boundary sub-passes per application.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OuterControls {
    /// Aitken delta-squared relaxation. `false` gives bare Picard - which is
    /// what the divergence measurement and its twin test want.
    pub aitken: bool,
    /// Decades of `||r||` below the first residual that mean convergence.
    pub decades: Scalar,
    /// Outer iterations never exceeded.
    pub max_outer: usize,
    /// Boundary sub-passes per application, set on the displacement before
    /// the first one.
    pub boundary_passes: usize,
}

impl Default for OuterControls {
    fn default() -> Self {
        Self { aitken: true, decades: 6.0, max_outer: 2000, boundary_passes: 3 }
    }
}

/// What one run of the outer loop cost, and what it contracted at: the
/// prototype's report with divergence gone (it is `Err(Error::Diverged)`
/// here) and `motion_ratio` added for the driver's reach-the-fluid decision.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OuterReport {
    /// Outer iterations taken.
    pub iterations: usize,
    /// `true` if `decades` decades were reached inside the cap.
    pub converged: bool,
    /// `||r_k||` for every `k` - the whole history, because a rate read off
    /// a single pair of iterations is not a rate.
    pub norms: Vec<Scalar>,
    /// `||r_k|| / ||r_{k-1}||` for every `k >= 2`.
    pub ratios: Vec<Scalar>,
    /// Geometric mean of the last ten ratios - the observed contraction.
    pub observed_contraction: Scalar,
    /// `(mu + lambda)/(2 mu + lambda)` from the Lame constants, the
    /// prediction the observed number is printed beside.
    pub predicted_contraction: Scalar,
    /// Total linear-solver iterations, all components, all outer iterations.
    pub linear_iterations: usize,
    /// The Aitken factors used, if any. Pushed after the convergence test:
    /// the iteration that converges pushes none, the first pushes one.
    pub omegas: Vec<Scalar>,
    /// `max|u| / min_c V_c^(1/3)`: how far the solid has moved in units of
    /// its own smallest cell. Decided by the driver, not here.
    pub motion_ratio: Scalar,
}

// ==========================================================================
//  The pieces the loop is made of
// ==========================================================================

/// The geometric mean of the last `last` ratios - the observed contraction,
/// the tail and not the whole history: the first ratios carry the transient
/// of a zero initial guess. `NaN` when there are no ratios.
pub(crate) fn observed_contraction(ratios: &[Scalar], last: usize) -> Scalar {
    let tail = &ratios[ratios.len().saturating_sub(last)..];
    if tail.is_empty() {
        return Scalar::NAN;
    }
    let s: Scalar = tail.iter().map(|r| r.ln()).sum();
    (s / (tail.len() as Scalar)).exp()
}

/// Aitken delta-squared dynamic relaxation, Küttler & Wall (2008) §3.2 in
/// its vector form, copied from `src/solid/prototype.rs`:
///
/// ```text
/// omega_k = -omega_{k-1} (r_{k-1} . (r_k - r_{k-1})) / |r_k - r_{k-1}|^2
/// ```
///
/// There is NO cap on omega: the prototype caps nothing and the twin test
/// is held to its count, and the cap in the paper is for the first omega of
/// a new time step, which a quasi-static run does not have. A non-positive
/// denominator - the residuals identical - keeps the previous factor.
pub(crate) fn aitken_omega(prev: Scalar, r_prev: &[Vec3], r: &[Vec3]) -> Scalar {
    let mut num = 0.0 as Scalar;
    let mut den = 0.0 as Scalar;
    for (a, b) in r_prev.iter().zip(r.iter()) {
        let d = *b - *a;
        num += a.dot(d);
        den += d.mag_sqr();
    }
    if den <= 0.0 {
        return prev;
    }
    let w = -prev * num / den;
    if w.is_finite() && w != 0.0 {
        w
    } else {
        prev
    }
}

/// The Euclidean norm of a vector field, sequential over cells - the
/// prototype's own reduction order.
fn l2(v: &[Vec3]) -> Scalar {
    v.iter().map(|a| a.mag_sqr()).sum::<Scalar>().sqrt()
}

// ==========================================================================
//  The loop
// ==========================================================================

/// Drive the displacement operator to its fixed point. `F` is one
/// [`Displacement::apply`]; everything inside one application is device
/// work, the five vector operations between applications are host work on a
/// downloaded copy of `F(u)`. Sets `d.passes` first, ends with one
/// `correct_boundary`, so the gradient the caller reads stress from belongs
/// to the accepted `d.u`. Divergence is [`Error::Diverged`], by name.
pub fn solve(gpu: &Gpu, d: &mut Displacement<'_>, ctrl: &OuterControls) -> Result<OuterReport> {
    d.passes = ctrl.boundary_passes;
    let n_c = d.mesh().n_cells;

    // Cell volumes, once: motion_ratio needs them; the mesh does not move.
    let vols: Vec<Scalar> = gpu.download(&d.mesh().v)?;
    let mut report = OuterReport {
        predicted_contraction: d.material.predicted_contraction(),
        ..Default::default()
    };
    let mut u: Vec<Vec3> = gpu.download(&d.u.f)?;
    let mut next = gpu.zeros(n_c)?;
    let mut r: Vec<Vec3> = vec![Vec3::ZERO; n_c];
    let mut r_prev: Vec<Vec3> = vec![Vec3::ZERO; n_c];
    let mut omega = 1.0 as Scalar;
    let mut first = 0.0 as Scalar;
    let mut prev_norm = 0.0 as Scalar;

    for k in 1..=ctrl.max_outer {
        let perf = d.apply(gpu, &mut next)?;
        report.linear_iterations += perf.iter().map(|p| p.n_iterations).sum::<usize>();
        let f: Vec<Vec3> = gpu.download(&next)?;
        for c in 0..n_c {
            r[c] = f[c] - u[c];
        }
        let norm = l2(&r);
        report.norms.push(norm);
        report.iterations = k;
        if !norm.is_finite() || (k > 1 && norm > first * 1.0e6) {
            return Err(Error::Diverged {
                iteration: k,
                what: format!(
                    "the segregated displacement fixed point grew from {first:.3e} to \
                     {norm:.3e} at nu = {:.2} (a million-fold): the Picard loop does not \
                     contract here; aitken = {}, boundary_passes = {}",
                    d.material.nu, ctrl.aitken, ctrl.boundary_passes
                ),
            });
        }

        if k == 1 {
            first = norm;
            omega = 1.0;
        } else {
            report.ratios.push(if prev_norm > 0.0 { norm / prev_norm } else { 0.0 });
            if ctrl.aitken {
                omega = aitken_omega(omega, &r_prev, &r);
            }
            if norm <= first * (10.0 as Scalar).powf(-ctrl.decades) {
                report.converged = true;
                for c in 0..n_c {
                    u[c] += r[c] * omega;
                }
                gpu.write(&mut d.u.f, &u)?;
                break;
            }
        }
        if ctrl.aitken {
            report.omegas.push(omega);
        }
        for c in 0..n_c {
            u[c] += r[c] * omega;
        }
        gpu.write(&mut d.u.f, &u)?;
        r_prev.copy_from_slice(&r);
        prev_norm = norm;
    }

    // The boundary values and the gradient the caller will read stress from
    // have to belong to the displacement that was just accepted.
    d.correct_boundary(gpu)?;
    report.observed_contraction = observed_contraction(&report.ratios, 10);
    report.motion_ratio = motion_ratio_of(&u, &vols);
    println!(
        "solid outer: nu={:.3} outer={} converged={} observed={:.4} predicted={:.4} \
         omega_last={:.3} linear_iters={} motion_ratio={:.3e}",
        d.material.nu,
        report.iterations,
        report.converged,
        report.observed_contraction,
        report.predicted_contraction,
        omega,
        report.linear_iterations,
        report.motion_ratio
    );
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Gpu;
    use crate::io::case::{LinearSolverKind, Preconditioner, SolverControls};
    use crate::mesh::{GpuMesh, HostMesh};
    use crate::solid::bc::fixed_minus_x;
    use crate::solid::displacement::Displacement;
    use crate::solid::prototype::{self, Prototype};
    use crate::solid::Material;
    use crate::Scalar;

    fn gpu() -> Option<Gpu> {
        Gpu::new(0).ok()
    }

    /// The comparison solve: PCG + DIC at the prototype's tolerance and cap,
    /// the literal the displacement unit's tests also use.
    fn tight() -> SolverControls {
        SolverControls {
            solver: LinearSolverKind::PCG,
            precon: Preconditioner::Dic,
            tolerance: 1e-14,
            rel_tol: 0.0,
            max_iter: 5000,
            min_iter: 0,
            check_interval: 1,
            fixed_iters: false,
            report_residuals: true,
        }
    }

    /// The twin's device side: the block uploaded, the operator on it, the
    /// sweep's thermal state (dT = 100 against T_REF = 300) written in.
    fn device_displacement<'a>(
        gpu: &Gpu,
        gm: &'a GpuMesh,
        hm: &HostMesh,
        nu: Scalar,
    ) -> Displacement<'a> {
        let mut d = Displacement::new(gpu, gm, hm, Material::steel(nu), &fixed_minus_x(), tight())
            .expect("the block is a lawful displacement problem");
        let (n, n_bf) = (hm.c.len(), hm.b_cf.len());
        d.set_temperature(gpu, &vec![400.0; n], &vec![400.0; n_bf], 300.0)
            .expect("temperature");
        d
    }

    /// The device outer loop is the prototype's `run` on the device: same
    /// stopping rule, same Aitken, same first step at `omega = 1`, same
    /// final boundary correction - and so the same count to one and the
    /// same norm history. The observed contraction is held to its MEASURED
    /// twin gap 1e-5, not the drafted 1e-6 (see the assertion); `omegas`'
    /// length is a transcription fact asserted so it stays one.
    #[test]
    fn the_device_outer_loop_is_the_prototypes_twin() {
        let Some(gpu) = gpu() else { return };
        for nu in [0.3, 0.45] {
            let hm_h = prototype::block(20).expect("block");
            let mut p = Prototype::new(hm_h, Material::steel(nu), 100.0, &fixed_minus_x());
            let host = p.run(true, 6.0, 300);
            assert!(host.converged, "nu = {nu}: prototype did not converge");
            let hm = prototype::block(20).expect("block");
            let gm = GpuMesh::upload(&gpu, &hm).expect("upload");
            let mut d = device_displacement(&gpu, &gm, &hm, nu);
            let rep = solve(
                &gpu,
                &mut d,
                &OuterControls { aitken: true, decades: 6.0, max_outer: 300, boundary_passes: 3 },
            )
            .expect("the device twin converges wherever the prototype does");
            assert!(rep.converged, "nu = {nu}: device run did not converge");
            assert!(
                (rep.iterations as i64 - host.iterations as i64).abs() <= 1,
                "nu = {nu}: device outer {} vs prototype {}",
                rep.iterations,
                host.iterations
            );
            let common = rep.norms.len().min(host.norms.len());
            let mut dev_all = 0.0 as Scalar;
            for k in 0..common {
                let rel = (rep.norms[k] - host.norms[k]).abs() / host.norms[k];
                dev_all = dev_all.max(rel);
            }
            println!("solid outer (twin): nu={nu:.2} norm-dev all={dev_all:.3e}");
            // The count and the norm history are the transcription's proof.
            // The two inner PCG solves stop at their own nearest tolerance,
            // and the near-critical map at nu = 0.45 amplifies that noise
            // through 71 iterations: measured, the histories agree to 4e-9
            // at nu = 0.3 and 6e-5 at 0.45, and the observed contraction to
            // 2.4e-6 - hence the 1e-4 and 1e-5 bars below, the measured
            // floor of the comparison, not the drafted 1e-6.
            assert!(
                dev_all <= 1.0e-4,
                "nu = {nu}: norm histories differ by {dev_all:.3e} - the loop \
                 is not the prototype's"
            );
            assert!(
                (rep.observed_contraction - host.observed_contraction).abs() <= 1.0e-5,
                "nu = {nu}: observed {:.6} device vs {:.6} prototype",
                rep.observed_contraction,
                host.observed_contraction
            );
            assert_eq!(rep.predicted_contraction, Material::steel(nu).predicted_contraction());
            assert_eq!(rep.ratios.len(), rep.iterations - 1);
            assert_eq!(rep.omegas.len(), rep.iterations - 1);
            let u_d = gpu.download(&d.u.f).expect("download");
            let max_u = p.u.iter().map(|a| a.mag()).fold(0.0 as Scalar, f64::max);
            let dev = p.u.iter().zip(u_d.iter()).map(|(h, dv)| (*h - *dv).mag()).fold(0.0, f64::max);
            assert!(dev <= 1.0e-5 * max_u, "nu = {nu}: |du| = {dev:e}");
            assert!(rep.motion_ratio > 0.0 && rep.motion_ratio.is_finite());
            println!(
                "solid outer (prototype): nu={nu:.3} outer={} converged={} observed={:.4} \
                 predicted={:.4} omega_last={:.3} linear_iters={} motion_ratio={:.3e}",
                host.iterations, host.converged, host.observed_contraction,
                host.predicted_contraction, host.omegas.last().copied().unwrap_or(1.0),
                host.linear_iterations, motion_ratio_of(&p.u, &p.mesh.v)
            );
        }
    }

    /// Bare Picard at nu = 0.45 diverges on the device too - and named:
    /// `Err(Error::Diverged)` saying what grew; or, if the residual reaches
    /// the cap before the detector trips, the prototype's own disjunction.
    #[test]
    fn bare_picard_diverges_on_the_device_too() {
        let Some(gpu) = gpu() else { return };
        let hm = prototype::block(20).expect("block");
        let gm = GpuMesh::upload(&gpu, &hm).expect("upload");
        let mut d = device_displacement(&gpu, &gm, &hm, 0.45);
        match solve(
            &gpu,
            &mut d,
            &OuterControls { aitken: false, decades: 6.0, max_outer: 300, boundary_passes: 3 },
        ) {
            Err(Error::Diverged { what, .. }) => assert!(
                what.contains("does not contract") && what.contains("nu = 0.45"),
                "the divergence message does not name the failure: {what}"
            ),
            Ok(r) => assert!(
                !r.converged && r.norms.last() > r.norms.first(),
                "bare Picard at nu = 0.45 converged in {} outer iterations - it did \
                 not when this was measured",
                r.iterations
            ),
            Err(e) => panic!("bare Picard failed some other way: {e}"),
        }
    }
}
