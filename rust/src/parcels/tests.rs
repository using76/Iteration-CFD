// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! SPEC-LIT S66.12's table: the three gates (terminal velocity against the
//! analytic balance, a parcel crossing a known mesh into the cell arithmetic
//! says it should, and two identical runs producing bitwise identical parcel
//! state), the CUDA-graph claim, the identity bijection, and the S13.4/S13.4.1
//! contract.
//!
//! Written from the same sources as `src/parcels.rs`; see that module's
//! header. No GPL-licensed source was consulted.

use std::collections::HashSet;
use std::path::PathBuf;

use super::*;
use crate::blockgen::{self, BlockSpec, GradedAxis};
use crate::device::Gpu;
use crate::field::GpuVectorField;
use crate::mesh::GpuMesh;

// ----------------------------------------------------------------------
//  Fixtures
// ----------------------------------------------------------------------

/// A uniform Cartesian block, `n` cells over `[0, hi]`, with the six patches
/// typed as given in `-x +x -y +y -z +z` order.
///
/// Cell `(i, j, k)` is index `i + nx*(j + ny*k)` - `blockgen`'s own ordering,
/// which is what makes S66.12's mesh-walk gate an ARITHMETIC statement about
/// the destination cell rather than a comparison against another search.
fn block(n: [usize; 3], hi: [Scalar; 3], types: [&str; 6]) -> HostMesh {
    let axis = |i: usize| GradedAxis {
        lo: 0.0,
        hi: hi[i],
        n: n[i],
        expansion: 1.0,
        two_sided: false,
    };
    let b = BlockSpec {
        x: axis(0),
        y: axis(1),
        z: axis(2),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: types.map(String::from),
        cyclic: Vec::new(),
    };
    blockgen::build_mesh(&b).expect("block mesh")
}

/// Still gas of uniform density - the state every verification case in S66.12
/// is posed in.
fn still_gas(gpu: &Gpu, m: &GpuMesh, rho: Scalar) -> Result<(GpuVectorField, DevBuf<Scalar>)> {
    let u = GpuVectorField::zeros(gpu, m, "U")?;
    let r = gpu.upload(&vec![rho; m.n_cells])?;
    Ok((u, r))
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join("ofgpuParcels");
    let _ = std::fs::create_dir_all(&d);
    d.join(format!("{tag}.vtp"))
}

/// Bitwise equality of two scalar arrays. `==` on `f64` is bitwise for
/// non-NaN, but `-0.0 == 0.0` and `NaN != NaN`, and a reproducibility claim
/// has to be about the BITS.
fn bits_eq(a: &[Scalar], b: &[Scalar]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn vec_bits_eq(a: &[Vec3], b: &[Vec3]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.x.to_bits() == y.x.to_bits()
                && x.y.to_bits() == y.y.to_bits()
                && x.z.to_bits() == y.z.to_bits()
        })
}

fn snapshot_bits_eq(a: &ParcelSnapshot, b: &ParcelSnapshot) -> bool {
    a.n_slots == b.n_slots
        && vec_bits_eq(&a.x, &b.x)
        && vec_bits_eq(&a.u, &b.u)
        && bits_eq(&a.d, &b.d)
        && bits_eq(&a.temperature, &b.temperature)
        && bits_eq(&a.n_p, &b.n_p)
        && a.cell == b.cell
        && a.uid == b.uid
        && a.flags == b.flags
}

// ======================================================================
//  SPEC-LIT (66.9): the identity
// ======================================================================

/// The inverse of `x ^= x >> k`, by fixed-point iteration. Converges in
/// `ceil(64/k)` rounds.
fn unxorshr(y: u64, k: u32) -> u64 {
    let mut x = y;
    for _ in 0..(64 / k + 1) {
        x = y ^ (x >> k);
    }
    x
}

/// The multiplicative inverse of an odd `u64` modulo `2^64`, by Newton
/// iteration (each step doubles the number of correct bits; `x = a` is
/// already correct modulo 8 for odd `a`).
fn inv_odd(a: u64) -> u64 {
    let mut x = a;
    for _ in 0..6 {
        x = x.wrapping_mul(2u64.wrapping_sub(a.wrapping_mul(x)));
    }
    x
}

fn unmix(mut z: u64) -> u64 {
    z = unxorshr(z, 31);
    z = z.wrapping_mul(inv_odd(0x94d0_49bb_1331_11eb));
    z = unxorshr(z, 27);
    z = z.wrapping_mul(inv_odd(0xbf58_476d_1ce4_e5b9));
    z = unxorshr(z, 30);
    z
}

/// **The load-bearing property of (66.9).** The mix is a bijection, so two
/// parcels can never share a `uid`, whatever the injector, event or index.
/// Uniqueness by construction rather than by a birthday argument - and the
/// birthday argument is what a 32-bit identity would have to rest on, which
/// at `10^6` parcels is not a rest at all.
#[test]
fn the_uid_mix_is_a_bijection() {
    assert_eq!(inv_odd(0x94d0_49bb_1331_11eb).wrapping_mul(0x94d0_49bb_1331_11eb), 1);
    assert_eq!(inv_odd(0xbf58_476d_1ce4_e5b9).wrapping_mul(0xbf58_476d_1ce4_e5b9), 1);

    for injector in [0u64, 1, 7, 4094, SEEDED_INJECTOR_ID] {
        for event in [0u64, 1, 2, 1000, 65_535, (1 << UID_EVENT_BITS) - 1] {
            for index in [0u64, 1, 2, 999, (1 << UID_INDEX_BITS) - 1] {
                let packed = (injector << (UID_EVENT_BITS + UID_INDEX_BITS))
                    | (event << UID_INDEX_BITS)
                    | index;
                let uid = parcel_uid(injector, event, index);
                assert_eq!(
                    unmix(uid),
                    packed,
                    "the mix must be invertible at ({injector}, {event}, {index})"
                );
            }
        }
    }
}

/// The consequence, checked directly on a dense sweep: no collisions.
#[test]
fn no_two_parcels_can_share_an_identity() {
    let mut seen = HashSet::new();
    for injector in 0..8u64 {
        for event in 0..128u64 {
            for index in 0..128u64 {
                assert!(
                    seen.insert(parcel_uid(injector, event, index)),
                    "collision at ({injector}, {event}, {index})"
                );
            }
        }
    }
    assert_eq!(seen.len(), 8 * 128 * 128);

    // And a seeded parcel can never collide with an injected one, because
    // the injector count is capped below the reserved id.
    let seeded: HashSet<u64> = (0..4096u64).map(|i| parcel_uid(SEEDED_INJECTOR_ID, 0, i)).collect();
    for injector in 0..SEEDED_INJECTOR_ID {
        for index in 0..4u64 {
            assert!(!seeded.contains(&parcel_uid(injector, 0, index)));
        }
    }
}

// ======================================================================
//  SPEC-LIT (66.3)/(66.4): drag and the terminal balance, on the host
// ======================================================================

/// The `Re = 1` continuity fix does what it is named for - and the `Re = 1000`
/// join does NOT, which is a fact about Schiller-Naumann and is recorded
/// rather than smoothed over.
#[test]
fn the_drag_law_joins_exactly_at_re_1_and_only_nearly_at_re_1000() {
    let (rho, mu, d) = (1.2 as Scalar, 1.8e-5 as Scalar, 1e-4 as Scalar);
    // |u| that puts Re exactly at the branch point.
    let u_at = |re: Scalar| re * mu / (rho * d);

    let below = drag_k(DragModel::SchillerNaumann, rho, mu, d, u_at(1.0) * (1.0 - 1e-12));
    let above = drag_k(DragModel::SchillerNaumann, rho, mu, d, u_at(1.0) * (1.0 + 1e-12));
    assert!(
        (above - below).abs() <= 1e-9 * below,
        "the Re = 1 join must be exact: {below} vs {above}"
    );

    let below = drag_k(DragModel::SchillerNaumann, rho, mu, d, u_at(1000.0) * (1.0 - 1e-12));
    let above = drag_k(DragModel::SchillerNaumann, rho, mu, d, u_at(1000.0) * (1.0 + 1e-12));
    let jump = (above - below).abs() / below;
    assert!(
        (0.005..0.02).contains(&jump),
        "Schiller-Naumann's 0.44 branch joins with a small step, published as such; \
         measured {jump}"
    );
}

/// The removable singularity, checked at the limit itself. `C_d = 24/Re` is
/// infinite at `Re = 0`; `K = rho C_d |u|` is `24 mu/d`, exactly, and the
/// kernel computes the second.
#[test]
fn the_drag_rate_is_finite_at_zero_relative_velocity() {
    let (rho, mu, d) = (1.2 as Scalar, 1.8e-5 as Scalar, 1e-4 as Scalar);
    let k0 = drag_k(DragModel::SchillerNaumann, rho, mu, d, 0.0);
    assert!(k0.is_finite() && k0 > 0.0);
    assert!((k0 - 24.0 * mu / d).abs() < 1e-18);
    assert_eq!(drag_k(DragModel::None, rho, mu, d, 5.0), 0.0);
}

/// **Gate 66-A, the analytic half.** The fixed point the host solves must
/// satisfy the force balance it was derived from, to round-off, over four
/// decades of diameter.
#[test]
fn the_terminal_velocity_satisfies_the_force_balance_it_came_from() {
    let (rho, rho_l, mu, g) = (1.2 as Scalar, 1000.0 as Scalar, 1.8e-5 as Scalar, 9.81 as Scalar);
    for d in [1e-5 as Scalar, 1e-4, 3e-4, 1e-3, 3e-3] {
        let ut = terminal_velocity(DragModel::SchillerNaumann, rho, rho_l, mu, d, g);
        // (1/2) rho C_d A_pc u^2 == m_p g (1 - rho/rho_l), written through K
        // so that the Stokes branch has no division by u:
        //   K(u) u * (3/4) / (rho_l d) == g (1 - rho/rho_l)
        let k = drag_k(DragModel::SchillerNaumann, rho, mu, d, ut);
        let lhs = k * ut * 0.75 / (rho_l * d);
        let rhs = g * (1.0 - rho / rho_l);
        assert!(
            (lhs - rhs).abs() <= 1e-12 * rhs,
            "d = {d}: balance residual {} at u_t = {ut}",
            (lhs - rhs).abs() / rhs
        );
    }
}

/// The design note that preceded S66 quotes `u_t = sqrt(4 rho_l g d/(3 rho
/// C_d))`, which drops the buoyancy factor `(1 - rho/rho_l)`. This records
/// the size of that omission rather than letting it be inherited silently.
#[test]
fn the_buoyancy_factor_the_design_note_dropped_is_measured() {
    let (rho, rho_l, mu, g, d) =
        (1.2 as Scalar, 1000.0 as Scalar, 1.8e-5 as Scalar, 9.81 as Scalar, 1e-4 as Scalar);
    let with = terminal_velocity(DragModel::SchillerNaumann, rho, rho_l, mu, d, g);
    // The note's form is this one with rho_l in place of (rho_l - rho).
    let k = drag_k(DragModel::SchillerNaumann, rho, mu, d, with);
    let without = g * (4.0 / 3.0) * rho_l * d / k;
    let err = (without - with).abs() / with;
    assert!(
        (1e-4..1e-2).contains(&err),
        "water in air: dropping buoyancy is a {err} relative error in u_t"
    );

    // For a droplet in a LIQUID carrier it is first order, which is why the
    // factor is in the kernel and not in a comment.
    let heavy = terminal_velocity(DragModel::SchillerNaumann, 800.0, 1000.0, 1e-3, 1e-3, g);
    let k = drag_k(DragModel::SchillerNaumann, 800.0, 1e-3, 1e-3, heavy);
    let heavy_without = g * (4.0 / 3.0) * 1000.0 * 1e-3 / k;
    assert!(
        (heavy_without - heavy).abs() / heavy > 1.0,
        "in a liquid carrier the buoyancy factor is not a correction, it is the answer"
    );
}

// ======================================================================
//  SPEC-LIT S13.4: the refusal contract
// ======================================================================

#[test]
fn every_unsupported_setting_is_refused_by_name_with_the_menu() {
    let _g = contract::permissive_test_guard();
    contract::set_permissive(false);

    let e = DragModel::from_name("putnamStokes").unwrap_err().to_string();
    assert!(e.contains("putnamStokes"), "{e}");
    assert!(e.contains("schillerNaumann"), "the menu must be printed: {e}");

    let e = WallAction::from_name("stick").unwrap_err().to_string();
    assert!(e.contains("stick") && e.contains("film"), "{e}");
    let e = WallAction::from_name("splash").unwrap_err().to_string();
    assert!(e.contains("splash") && e.contains("population growth"), "{e}");

    // The one this unit is defined by NOT having. `heating` was refused here
    // too until S68.5 implemented it - the sensible-heat half of the same two
    // Ranz & Marshall papers - so the menu grew by one and the refusal list
    // shrank by one. Evaporation is still refused, and still by name.
    for name in ["evaporating", "evaporation", "heatAndMassTransfer"] {
        let e = ParcelPhysics::from_name(name).unwrap_err().to_string();
        assert!(e.contains(name), "{e}");
        assert!(
            e.contains("evaporation") || e.contains("evaporating"),
            "the refusal must name what is missing: {e}"
        );
        assert!(e.contains("inert"), "the alternative must be printed: {e}");
    }
    assert_eq!(
        ParcelPhysics::from_name("heating").unwrap(),
        ParcelPhysics::Heating,
        "S68.5 supports `heating`; if this ever fails, the refusal above has to \
         grow it back"
    );

    assert_eq!(DragModel::from_name("stokes").unwrap(), DragModel::Stokes);
    assert_eq!(WallAction::from_name("rebound").unwrap(), WallAction::Rebound);
    assert_eq!(ParcelPhysics::from_name("inert").unwrap(), ParcelPhysics::Inert);
}

#[test]
fn validate_names_every_number_it_rejects() {
    let base = ParcelControls::default();
    let cases: [(ParcelControls, &str); 6] = [
        (ParcelControls { capacity: 0, ..base }, "capacity"),
        (ParcelControls { rho_liquid: 0.0, ..base }, "rhoLiquid"),
        (ParcelControls { mu_gas: -1.0, ..base }, "muGas"),
        (ParcelControls { cfl: 0.0, ..base }, "cfl"),
        (ParcelControls { max_walk: 0, ..base }, "maxWalk"),
        (ParcelControls { restitution: 1.5, ..base }, "restitution"),
    ];
    for (c, what) in cases {
        let e = c.validate().unwrap_err().to_string();
        assert!(e.contains(what), "the refusal must name {what}: {e}");
    }
    assert!(base.validate().is_ok());
    assert!(base.describe().contains("schillerNaumann"));
    assert!(base.describe().contains("inert"));
}

// ======================================================================
//  Host geometry
// ======================================================================

#[test]
fn locate_cell_finds_the_cell_the_index_arithmetic_names() {
    let hm = block([5, 4, 3], [1.0, 1.0, 1.0], ["patch"; 6]);
    let (hx, hy, hz) = (1.0 / 5.0, 1.0 / 4.0, 1.0 / 3.0);
    for k in 0..3 {
        for j in 0..4 {
            for i in 0..5 {
                let p = Vec3::new(
                    (i as Scalar + 0.5) * hx,
                    (j as Scalar + 0.5) * hy,
                    (k as Scalar + 0.5) * hz,
                );
                assert_eq!(locate_cell(&hm, p), Some(i + 5 * (j + 4 * k)), "at {p:?}");
            }
        }
    }
    assert_eq!(locate_cell(&hm, Vec3::new(-0.1, 0.5, 0.5)), None);
    assert_eq!(locate_cell(&hm, Vec3::new(0.5, 0.5, 1.7)), None);
}

#[test]
fn the_injection_weight_makes_the_emitted_mass_exact() {
    // (66.8): n_p = mdot dt stride / (n_per_event m_droplet), so
    // n_per_event * n_p * m_droplet is mdot dt stride identically - which is
    // what makes the discharged mass independent of how many parcels the
    // case chose to represent it with.
    let rho_l: Scalar = 1000.0;
    let dt: Scalar = 1e-3;
    for n in [1u32, 7, 64, 1000] {
        for d in [1e-5 as Scalar, 2e-4, 1e-3] {
            let mdot: Scalar = 0.037;
            let stride = 5 as Scalar;
            let m_droplet = rho_l * std::f64::consts::FRAC_PI_6 as Scalar * d * d * d;
            let n_p = mdot * dt * stride / (n as Scalar * m_droplet);
            let emitted = n as Scalar * n_p * m_droplet;
            assert!(
                (emitted - mdot * dt * stride).abs() <= 1e-15 * mdot * dt * stride,
                "n = {n}, d = {d}: emitted {emitted}"
            );
        }
    }
}

// ======================================================================
//  GPU: the three gates of S66.12
// ======================================================================

/// A one-parcel run: seed at rest, integrate to `t_end` (but never fewer than
/// `min_steps`), return the final speed.
///
/// `max_substeps = 1` on purpose - the point of Gate 66-A is that the
/// EXPONENTIAL update is `dt`-independent, and sub-stepping would resolve the
/// transient for the large steps and hide exactly what is being tested.
///
/// `min_steps` is there because of (66.5)'s honest small print. When
/// `dt >> tau_p`, `exp(-dt/tau_p)` underflows and the update collapses to
/// `u^{n+1} = a_g tau_p(u^n)` - a FIXED-POINT ITERATION for the terminal
/// velocity, not a one-step answer. It contracts at
/// `|d ln K/d ln u| = 0.147` for this droplet, so it converges quickly, but
/// "run for eight seconds" is eight iterations at `dt = 1 s` and that is not
/// the same statement as "run to steady state". Measured: eight steps leaves
/// `1.1e-7` relative error, twenty-four leaves less than `1e-15`.
fn fall_to_terminal(
    gpu: &Gpu,
    hm: &HostMesh,
    gm: &GpuMesh,
    d: Scalar,
    dt: Scalar,
    t_end: Scalar,
    min_steps: usize,
    drag: DragModel,
) -> Result<Scalar> {
    let ctrl = ParcelControls {
        capacity: 4,
        drag,
        physics: ParcelPhysics::Inert,
        wall: WallAction::Remove,
        restitution: 1.0,
        tangential_loss: 0.0,
        gravity: Vec3::new(0.0, 0.0, -9.81),
        rho_liquid: 1000.0,
        mu_gas: 1.8e-5,
        c_liquid: 4182.0,
        k_gas: 0.026,
        cp_gas: 1005.0,
        added_mass: false,
        cfl: 0.9,
        max_substeps: 1,
        max_walk: 16,
        persistent_blocks: None,
    };
    let mut p = Parcels::new(gpu, hm, gm, ctrl, &[], dt)?;
    p.seed(
        gpu,
        hm,
        &[SeedParcel {
            position: Vec3::new(0.5, 0.5, 9.0),
            velocity: Vec3::ZERO,
            diameter: d,
            temperature: 293.15,
            n_p: 1.0,
            uid: None,
        }],
    )?;
    let (u, rho) = still_gas(gpu, gm, 1.2)?;
    let n = ((t_end / dt).round() as usize).max(min_steps);
    for _ in 0..n {
        p.step(gpu, &u, &rho, None, dt)?;
    }
    let s = p.snapshot(gpu)?;
    assert!(s.cell[0] >= 0, "the droplet left the box before the run ended");
    assert_eq!(p.stats(gpu)?.n_lost, 0, "no parcel may be lost on a Cartesian mesh");
    Ok(s.u[0].mag())
}

/// **Gate 66-A.** A droplet released in still air reaches the terminal
/// velocity of the analytic force balance, and reaches the SAME one at every
/// time step from `1e-3 s` to `1 s` - four decades, the largest of which is
/// thirty-two response times. That is what the exponential integration of
/// (66.5) buys; an explicit Euler step at `dt = 1 s` has an amplification
/// factor of `1 - dt/tau_p = -31` and diverges on the first step.
#[test]
fn gate_66a_terminal_velocity_is_the_analytic_one_at_every_time_step() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([2, 2, 20], [1.0, 1.0, 10.0], ["patch"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();

    let d: Scalar = 1e-4;
    let analytic = terminal_velocity(DragModel::SchillerNaumann, 1.2, 1000.0, 1.8e-5, d, 9.81);
    assert!((0.2..0.4).contains(&analytic), "u_t = {analytic}");

    let mut worst: Scalar = 0.0;
    for dt in [1e-3 as Scalar, 1e-2, 1e-1, 1.0] {
        let u =
            fall_to_terminal(&gpu, &hm, &gm, d, dt, 8.0, 24, DragModel::SchillerNaumann).unwrap();
        let err = (u - analytic).abs() / analytic;
        worst = worst.max(err);
        assert!(
            err < 1e-9,
            "dt = {dt}: terminal speed {u} against the analytic {analytic} (rel {err})"
        );
    }
    assert!(worst < 1e-9, "worst relative error over the dt sweep: {worst}");
}

/// The same gate in the branch where the drag law is NONLINEAR in the speed,
/// so the terminal velocity is a genuine fixed point rather than a one-step
/// answer: a 300 um droplet sits at `Re ~ 25`, inside Schiller-Naumann's
/// `24(0.85 + 0.15 Re^0.687)/Re` range.
#[test]
fn gate_66a_holds_in_the_intermediate_reynolds_branch() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([2, 2, 20], [1.0, 1.0, 10.0], ["patch"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();

    let d: Scalar = 3e-4;
    let analytic = terminal_velocity(DragModel::SchillerNaumann, 1.2, 1000.0, 1.8e-5, d, 9.81);
    let re = 1.2 * analytic * d / 1.8e-5;
    assert!((5.0..500.0).contains(&re), "this case must exercise the middle branch: Re = {re}");

    let u =
        fall_to_terminal(&gpu, &hm, &gm, d, 1e-3, 2.0, 1, DragModel::SchillerNaumann).unwrap();
    let err = (u - analytic).abs() / analytic;
    assert!(err < 1e-6, "u = {u} against the analytic {analytic} (rel {err})");
}

/// **Gate 66-B.** A ballistic parcel crossing a known Cartesian mesh lands in
/// the cell the index arithmetic says it should, and at the position a
/// straight line says it should.
///
/// `dragModel none` is what makes this a test of the WALK: with no drag and
/// no gravity the trajectory is a straight line whose endpoint is computed
/// without the solver, so a disagreement is the walk's and nothing else's.
#[test]
fn gate_66b_a_parcel_lands_in_the_cell_the_arithmetic_names() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let n = 10usize;
    let hm = block([n, n, n], [1.0, 1.0, 1.0], ["patch"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let h = 1.0 / n as Scalar;

    // Start points and velocities chosen so no endpoint sits on a face
    // plane; a point exactly on a face belongs to two cells and the question
    // "which cell" has no arithmetic answer to compare against.
    let seeds: Vec<SeedParcel> = [
        (Vec3::new(0.05, 0.05, 0.05), Vec3::new(0.31, 0.52, 0.73)),
        (Vec3::new(0.55, 0.35, 0.15), Vec3::new(-0.41, 0.23, 0.61)),
        (Vec3::new(0.95, 0.95, 0.95), Vec3::new(-0.77, -0.83, -0.67)),
        (Vec3::new(0.15, 0.85, 0.45), Vec3::new(0.63, -0.71, 0.09)),
        (Vec3::new(0.45, 0.45, 0.45), Vec3::new(0.0, 0.0, 0.37)),
    ]
    .iter()
    .map(|&(position, velocity)| SeedParcel {
        position,
        velocity,
        diameter: 1e-4,
        temperature: 293.15,
        n_p: 1.0,
        uid: None,
    })
    .collect();

    let ctrl = ParcelControls {
        capacity: 16,
        drag: DragModel::None,
        physics: ParcelPhysics::Inert,
        wall: WallAction::Remove,
        restitution: 1.0,
        tangential_loss: 0.0,
        gravity: Vec3::ZERO,
        rho_liquid: 1000.0,
        mu_gas: 1.8e-5,
        c_liquid: 4182.0,
        k_gas: 0.026,
        cp_gas: 1005.0,
        added_mass: false,
        cfl: 0.9,
        max_substeps: 64,
        max_walk: 16,
        persistent_blocks: None,
    };
    let dt: Scalar = 1.0;
    let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[], dt).unwrap();
    p.seed(&gpu, &hm, &seeds).unwrap();
    let (u, rho) = still_gas(&gpu, &gm, 1.2).unwrap();
    p.step(&gpu, &u, &rho, None, dt).unwrap();

    let s = p.snapshot(&gpu).unwrap();
    let stats = p.stats(&gpu).unwrap();
    assert_eq!(stats.n_lost, 0, "the walk lost a parcel on a Cartesian mesh");
    assert_eq!(stats.n_escaped, 0, "no seed was aimed out of the domain");

    for (i, sd) in seeds.iter().enumerate() {
        let expect_x = sd.position + sd.velocity * dt;
        let idx = |v: Scalar| (v / h).floor() as usize;
        let expect_cell = idx(expect_x.x) + n * (idx(expect_x.y) + n * idx(expect_x.z));
        assert_eq!(
            s.cell[i] as usize, expect_cell,
            "parcel {i}: ended in cell {} at {:?}, arithmetic says {expect_cell}",
            s.cell[i], s.x[i]
        );
        // The walk must not move the parcel: it only decides which cell the
        // straight line ended in.
        assert!(
            (s.x[i] - expect_x).mag() < 1e-13,
            "parcel {i}: position {:?} against the straight line {expect_x:?}",
            s.x[i]
        );
        assert!((s.u[i] - sd.velocity).mag() == 0.0, "no drag means no change of velocity");
    }
}

#[test]
fn a_parcel_aimed_out_of_the_domain_escapes_at_the_face_and_is_counted() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([10, 10, 10], [1.0, 1.0, 1.0], ["patch"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let ctrl = ParcelControls {
        capacity: 4,
        drag: DragModel::None,
        physics: ParcelPhysics::Inert,
        wall: WallAction::Remove,
        restitution: 1.0,
        tangential_loss: 0.0,
        gravity: Vec3::ZERO,
        rho_liquid: 1000.0,
        mu_gas: 1.8e-5,
        c_liquid: 4182.0,
        k_gas: 0.026,
        cp_gas: 1005.0,
        added_mass: false,
        cfl: 0.9,
        max_substeps: 64,
        max_walk: 32,
        persistent_blocks: None,
    };
    let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[], 1.0).unwrap();
    p.seed(
        &gpu,
        &hm,
        &[SeedParcel {
            position: Vec3::new(0.5, 0.35, 0.45),
            velocity: Vec3::new(2.0, 0.0, 0.0),
            diameter: 1e-4,
            temperature: 293.15,
            n_p: 1.0,
            uid: None,
        }],
    )
    .unwrap();
    let (u, rho) = still_gas(&gpu, &gm, 1.2).unwrap();
    p.step(&gpu, &u, &rho, None, 1.0).unwrap();

    let s = p.snapshot(&gpu).unwrap();
    let st = p.stats(&gpu).unwrap();
    assert_eq!(s.cell[0], -1, "an escaped parcel is marked dead by its cell");
    assert_eq!(s.flags[0] & 1, 0, "and its active bit is cleared");
    assert_eq!(st.n_escaped, 1);
    assert_eq!(st.n_wall, 0);
    assert!(
        (s.x[0].x - 1.0).abs() < 1e-13,
        "it stops ON the face it left through, at x = {}",
        s.x[0].x
    );
    // Nothing was written to the .vtp for it.
    assert!(s.live().is_empty());
}

/// A wall under [`WallAction::Rebound`] with `e = 1` is a mirror, and the
/// trajectory of a parcel bouncing between two of them is the triangle-wave
/// fold of the straight line it would otherwise have flown. That is an
/// analytic statement, so the reflection can be checked without trusting the
/// solver twice.
#[test]
fn a_rebounding_parcel_follows_the_folded_straight_line() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([10, 4, 4], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let ctrl = ParcelControls {
        capacity: 4,
        drag: DragModel::None,
        physics: ParcelPhysics::Inert,
        wall: WallAction::Rebound,
        restitution: 1.0,
        tangential_loss: 0.0,
        gravity: Vec3::ZERO,
        rho_liquid: 1000.0,
        mu_gas: 1.8e-5,
        c_liquid: 4182.0,
        k_gas: 0.026,
        cp_gas: 1005.0,
        added_mass: false,
        cfl: 0.9,
        max_substeps: 64,
        max_walk: 32,
        persistent_blocks: None,
    };
    let dt: Scalar = 0.05;
    let (x0, ux): (Scalar, Scalar) = (0.37, 1.3);
    let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[], dt).unwrap();
    p.seed(
        &gpu,
        &hm,
        &[SeedParcel {
            position: Vec3::new(x0, 0.375, 0.625),
            velocity: Vec3::new(ux, 0.0, 0.0),
            diameter: 1e-4,
            temperature: 293.15,
            n_p: 1.0,
            uid: None,
        }],
    )
    .unwrap();
    let (u, rho) = still_gas(&gpu, &gm, 1.2).unwrap();

    for n in 1..=40usize {
        p.step(&gpu, &u, &rho, None, dt).unwrap();
        let s = p.snapshot(&gpu).unwrap();
        assert!(s.cell[0] >= 0, "a rebounding parcel never leaves the domain");
        let t = n as Scalar * dt;
        let far = x0 + ux * t;
        let m = far % 2.0;
        let folded = if m > 1.0 { 2.0 - m } else { m };
        assert!(
            (s.x[0].x - folded).abs() < 1e-11,
            "step {n}: x = {} against the folded straight line {folded}",
            s.x[0].x
        );
    }
    let st = p.stats(&gpu).unwrap();
    assert_eq!(st.n_wall, 0, "rebound removes nothing");
    assert_eq!(st.n_lost, 0);
}

// ======================================================================
//  A run, and the things two of them must and must not share
// ======================================================================

/// The base case every reproducibility and pair test below is built on: one
/// hollow-cone injector low enough that its parcels reach the floor within
/// the run, in a box whose six patches are walls.
fn base_controls() -> ParcelControls {
    ParcelControls {
        capacity: 4096,
        drag: DragModel::SchillerNaumann,
        physics: ParcelPhysics::Inert,
        wall: WallAction::Remove,
        restitution: 1.0,
        tangential_loss: 0.0,
        gravity: Vec3::new(0.0, 0.0, -9.81),
        rho_liquid: 1000.0,
        mu_gas: 1.8e-5,
        c_liquid: 4182.0,
        k_gas: 0.026,
        cp_gas: 1005.0,
        added_mass: false,
        cfl: 0.9,
        max_substeps: 64,
        max_walk: 16,
        persistent_blocks: None,
    }
}

fn base_injector() -> Injector {
    Injector {
        position: Vec3::new(0.5, 0.5, 0.25),
        axis: Vec3::new(0.0, 0.0, -1.0),
        cone_half_angle: std::f64::consts::FRAC_PI_6 as Scalar,
        standoff: 0.02,
        speed: 3.0,
        diameter: 2e-4,
        temperature: 300.0,
        mass_flow: 1e-3,
        parcels_per_event: 8,
        interval: 0.0,
    }
}

const BASE_DT: Scalar = 0.05;
const BASE_STEPS: usize = 20;

/// Run the base case with the given settings and return the pool.
///
/// **Both sides of every pair below call THIS function**, never a
/// re-derivation of the control struct - which is exactly the shortcut
/// SPEC-LIT S13.4.1 records five instances of.
fn run_case(
    gpu: &Gpu,
    hm: &HostMesh,
    gm: &GpuMesh,
    ctrl: ParcelControls,
    inj: &[Injector],
    steps: usize,
) -> Result<ParcelSnapshot> {
    let mut p = Parcels::new(gpu, hm, gm, ctrl, inj, BASE_DT)?;
    let (u, rho) = still_gas(gpu, gm, 1.2)?;
    for _ in 0..steps {
        p.step(gpu, &u, &rho, None, BASE_DT)?;
    }
    p.snapshot(gpu)
}

/// **Gate 66-C.** Two runs of the same case, from the same input, produce
/// bitwise identical parcel state - every position, every velocity, every
/// identity, bit for bit.
#[test]
fn gate_66c_two_identical_runs_are_bitwise_identical() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([10, 10, 10], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let inj = [base_injector()];

    let a = run_case(&gpu, &hm, &gm, base_controls(), &inj, BASE_STEPS).unwrap();
    let b = run_case(&gpu, &hm, &gm, base_controls(), &inj, BASE_STEPS).unwrap();
    assert!(a.n_slots > 50, "the case must actually inject: {} slots", a.n_slots);
    assert!(
        snapshot_bits_eq(&a, &b),
        "two identical runs differ - the whole reproducibility claim rests on this"
    );

    // And the file that is written from them is byte-identical too.
    let pa = scratch("repro_a");
    let pb = scratch("repro_b");
    crate::io::vtu::write_parcels_vtp(&pa, &a, Some(1.0)).unwrap();
    crate::io::vtu::write_parcels_vtp(&pb, &b, Some(1.0)).unwrap();
    assert_eq!(std::fs::read(&pa).unwrap(), std::fs::read(&pb).unwrap());
}

/// The stronger form S66.12 can support in this section, and the exact limit
/// of it.
///
/// Permuting the initial pool permutes the answer and changes nothing else:
/// parcel `i` in one run and parcel `perm[i]` in the other end at the same
/// position and velocity, **bit for bit**. That proves there is no
/// parcel-to-parcel coupling and no dependence on slot order in the physics.
///
/// It is NOT the canonicalisation test of the deposition sort, which needs
/// the sort to exist, and it deliberately excludes `uid`: a SEEDED parcel's
/// identity is `mix(SEEDED, 0, slot)` and therefore does move with the
/// permutation. An INJECTED parcel's identity is a function of
/// `(injector, event, index)` and does not - which is the property the sort
/// will need.
#[test]
fn a_permutation_of_the_pool_permutes_the_answer_and_nothing_else() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([10, 10, 10], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();

    let seeds: Vec<SeedParcel> = (0..12)
        .map(|i| {
            let f = i as Scalar;
            SeedParcel {
                position: Vec3::new(0.13 + 0.06 * f, 0.17 + 0.05 * f, 0.83 - 0.05 * f),
                velocity: Vec3::new(0.7 - 0.11 * f, -0.3 + 0.07 * f, 0.2 * f - 1.1),
                diameter: 1e-4 + 2e-5 * f,
                temperature: 290.0 + f,
                n_p: 1.0 + f,
                uid: None,
            }
        })
        .collect();
    // A fixed shuffle, not a random one: the test has to be reproducible too.
    let perm: [usize; 12] = [7, 0, 11, 3, 5, 1, 9, 2, 10, 4, 8, 6];
    let shuffled: Vec<SeedParcel> = perm.iter().map(|&i| seeds[i]).collect();

    let ctrl = base_controls();
    let run = |sd: &[SeedParcel]| -> ParcelSnapshot {
        let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[], BASE_DT).unwrap();
        p.seed(&gpu, &hm, sd).unwrap();
        let (u, rho) = still_gas(&gpu, &gm, 1.2).unwrap();
        for _ in 0..BASE_STEPS {
            p.step(&gpu, &u, &rho, None, BASE_DT).unwrap();
        }
        p.snapshot(&gpu).unwrap()
    };

    let a = run(&seeds);
    let b = run(&shuffled);
    for (slot, &orig) in perm.iter().enumerate() {
        assert_eq!(a.cell[orig], b.cell[slot], "parcel {orig} landed in a different cell");
        assert!(
            vec_bits_eq(&[a.x[orig]], &[b.x[slot]]) && vec_bits_eq(&[a.u[orig]], &[b.u[slot]]),
            "parcel {orig}: {:?}/{:?} against {:?}/{:?}",
            a.x[orig],
            a.u[orig],
            b.x[slot],
            b.u[slot]
        );
    }
}

/// **The CUDA-graph claim of S66.7, tested directly.** The graph is captured
/// ONCE, before any step has run, and replayed twenty times while the working
/// set grows from zero to a hundred and sixty parcels underneath it. It
/// reproduces the eager path bit for bit.
///
/// If `n_active` or the step counter were kernel arguments rather than device
/// memory, or if the launch geometry depended on the parcel count, the replay
/// would inject the same first event twenty times over and this would fail on
/// the second launch.
#[test]
fn the_graph_is_captured_once_and_replayed() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([10, 10, 10], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let inj = [base_injector()];

    let eager = run_case(&gpu, &hm, &gm, base_controls(), &inj, BASE_STEPS).unwrap();

    let mut p = Parcels::new(&gpu, &hm, &gm, base_controls(), &inj, BASE_DT).unwrap();
    let (u, rho) = still_gas(&gpu, &gm, 1.2).unwrap();

    let captured = gpu
        .capture(|_| p.step(&gpu, &u, &rho, None, BASE_DT))
        .expect("capture must not fail: nothing in the step allocates, synchronises or reads back");
    let Some(mut graph) = captured else {
        panic!("the capture produced an empty graph - the step launched nothing");
    };
    graph.upload().unwrap();

    for _ in 0..BASE_STEPS {
        graph.launch().unwrap();
    }
    gpu.sync().unwrap();

    let replayed = p.snapshot(&gpu).unwrap();
    let st = p.stats(&gpu).unwrap();
    assert_eq!(
        st.n_injected,
        (BASE_STEPS * base_injector().parcels_per_event as usize) as i64,
        "the replay must inject once per launch, which it can only do by reading the step \
         counter from device memory"
    );
    assert_eq!(st.n_dropped, 0);
    assert!(
        snapshot_bits_eq(&eager, &replayed),
        "the graph replay differs from the eager path"
    );
}

/// The launch geometry is a performance knob whose contract is the OPPOSITE
/// of S13.4.1's: it is required NOT to change the answer. This is the
/// admissible-exception treatment that subsection sets out - a setting whose
/// effect must be identically zero is asserted to be inert, and the reason is
/// stated rather than left as a missing row in the pair table.
#[test]
fn the_persistent_grid_geometry_does_not_change_the_answer() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([10, 10, 10], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let inj = [base_injector()];

    let mut prev: Option<ParcelSnapshot> = None;
    for blocks in [1u32, 3, 17, 64] {
        let ctrl = ParcelControls { persistent_blocks: Some(blocks), ..base_controls() };
        let s = run_case(&gpu, &hm, &gm, ctrl, &inj, BASE_STEPS).unwrap();
        if let Some(p) = &prev {
            assert!(
                snapshot_bits_eq(p, &s),
                "{blocks} blocks gave a different answer - a grid-stride kernel whose result \
                 depends on gridDim is not order-independent"
            );
        }
        prev = Some(s);
    }
}

/// The device's drag branches are the host's. `DragModel`'s discriminants are
/// mirrored by the `OFP_DRAG_*` defines in `cuda/parcels.cu`, and a mismatch
/// would silently run a different correlation from the one the case named -
/// the S13.4.1 defect in its purest form.
///
/// Measured on the first step from rest, where the velocity change is
/// `a_g dt q(beta)` with `beta = dt K/inertia`, so `K` is recoverable from the
/// answer and can be compared with the host's [`drag_k`].
#[test]
fn the_device_enumerations_match_the_host() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([2, 2, 20], [1.0, 1.0, 10.0], ["patch"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (u, rho) = still_gas(&gpu, &gm, 1.2).unwrap();

    let dt: Scalar = 1e-3;
    let d: Scalar = 3e-4;
    let u0: Scalar = 5.0;
    for model in [DragModel::None, DragModel::Stokes, DragModel::SchillerNaumann] {
        let ctrl = ParcelControls {
            capacity: 2,
            drag: model,
            gravity: Vec3::ZERO,
            max_substeps: 1,
            ..base_controls()
        };
        let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[], dt).unwrap();
        p.seed(
            &gpu,
            &hm,
            &[SeedParcel {
                position: Vec3::new(0.5, 0.5, 5.0),
                velocity: Vec3::new(0.0, 0.0, -u0),
                diameter: d,
                temperature: 293.15,
                n_p: 1.0,
                uid: None,
            }],
        )
        .unwrap();
        p.step(&gpu, &u, &rho, None, dt).unwrap();
        let s = p.snapshot(&gpu).unwrap();

        // u_new = u + (0 - u) w, so w = 1 - u_new/u and beta = -ln(1 - w).
        let w = 1.0 - s.u[0].z / (-u0);
        let expect_k = drag_k(model, 1.2, ctrl.mu_gas, d, u0);
        let expect_beta = dt * expect_k / ((4.0 / 3.0) * ctrl.rho_liquid * d);
        let expect_w = -(-expect_beta).exp_m1();
        assert!(
            (w - expect_w).abs() <= 1e-12 * expect_w.max(1e-12),
            "{}: the kernel relaxed by {w}, the host says {expect_w}",
            model.name()
        );
    }
}

/// The identity the kernel wrote is the identity the host computes. If the
/// two mixes ever drift apart, the reproducibility argument still holds but
/// the RESTART one does not, because a restart reconstructs `uid` on the host.
#[test]
fn the_device_identity_matches_the_host_identity() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([10, 10, 10], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let inj = [base_injector(), Injector { position: Vec3::new(0.3, 0.7, 0.6), ..base_injector() }];

    let mut p = Parcels::new(&gpu, &hm, &gm, base_controls(), &inj, BASE_DT).unwrap();
    let (u, rho) = still_gas(&gpu, &gm, 1.2).unwrap();
    for _ in 0..4 {
        p.step(&gpu, &u, &rho, None, BASE_DT).unwrap();
    }
    let s = p.snapshot(&gpu).unwrap();

    let per = base_injector().parcels_per_event as usize;
    let mut slot = 0usize;
    for event in 0..4u64 {
        for (j, _) in inj.iter().enumerate() {
            for index in 0..per {
                assert_eq!(
                    s.uid[slot],
                    parcel_uid(j as u64, event, index as u64),
                    "slot {slot}: injector {j}, event {event}, index {index}"
                );
                slot += 1;
            }
        }
    }
    assert_eq!(slot, s.n_slots);
    // Distinct, which is the property that matters downstream.
    let set: HashSet<u64> = s.uid[..s.n_slots].iter().copied().collect();
    assert_eq!(set.len(), s.n_slots);
}

/// (66.8): the discharged mass is `mdot * t`, exactly, however many parcels
/// the case chose to represent it with - and it is exact, not approximate,
/// because `n_p` is derived from the flow rate rather than the other way
/// round.
#[test]
fn the_injector_discharges_exactly_the_mass_the_flow_rate_asks_for() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([10, 10, 10], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let mdot: Scalar = 1e-3;
    let steps = 6usize;

    for per_event in [1u32, 8, 37] {
        for d in [1e-4 as Scalar, 4e-4] {
            let inj = [Injector {
                parcels_per_event: per_event,
                diameter: d,
                mass_flow: mdot,
                // Upward, so nothing reaches a wall and the mass stays in
                // the pool for the whole run.
                axis: Vec3::new(0.0, 0.0, 1.0),
                position: Vec3::new(0.5, 0.5, 0.15),
                speed: 0.2,
                cone_half_angle: 0.1,
                standoff: 0.01,
                ..base_injector()
            }];
            let ctrl = ParcelControls { gravity: Vec3::ZERO, ..base_controls() };
            let s = run_case(&gpu, &hm, &gm, ctrl, &inj, steps).unwrap();
            let expect = mdot * BASE_DT * steps as Scalar;
            let got = s.liquid_mass(ctrl.rho_liquid);
            assert!(
                (got - expect).abs() <= 1e-12 * expect,
                "per_event = {per_event}, d = {d}: discharged {got} kg, asked for {expect}"
            );
        }
    }
}

/// S66.11: a full pool is refused by name, outside the step loop, with the
/// three things a user can do about it.
#[test]
fn the_pool_refuses_by_name_when_it_overflows() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([10, 10, 10], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let inj = [base_injector()];
    let ctrl = ParcelControls { capacity: 20, ..base_controls() };

    let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &inj, BASE_DT).unwrap();
    let (u, rho) = still_gas(&gpu, &gm, 1.2).unwrap();
    for _ in 0..10 {
        p.step(&gpu, &u, &rho, None, BASE_DT).unwrap();
    }
    let st = p.stats(&gpu).unwrap();
    assert!(st.n_dropped > 0, "capacity 20 with 8 parcels a step for 10 steps must overflow");
    assert!(st.n_slots <= ctrl.capacity, "the pool never writes past its capacity");
    let e = st.check_capacity().unwrap_err().to_string();
    assert!(e.contains("capacity"), "{e}");
    assert!(e.contains("parcelsPerEvent"), "the way out must be printed: {e}");
}

/// S66.6: a coupled patch is refused at setup, by name, because parcel
/// transport across one needs a transform this section has not got.
#[test]
fn a_cyclic_mesh_is_refused_by_name() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let mut b = BlockSpec {
        x: GradedAxis { lo: 0.0, hi: 1.0, n: 6, expansion: 1.0, two_sided: false },
        y: GradedAxis { lo: 0.0, hi: 1.0, n: 6, expansion: 1.0, two_sided: false },
        z: GradedAxis { lo: 0.0, hi: 1.0, n: 6, expansion: 1.0, two_sided: false },
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["patch", "patch", "wall", "wall", "wall", "wall"].map(String::from),
        cyclic: Vec::new(),
    };
    b.set_cyclic_axis(0).unwrap();
    let hm = blockgen::build_mesh(&b).unwrap();
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let e = match Parcels::new(&gpu, &hm, &gm, base_controls(), &[], BASE_DT) {
        Ok(_) => panic!("a cyclic mesh must be refused, not silently accepted"),
        Err(e) => e.to_string(),
    };
    assert!(e.contains("cyclic"), "{e}");
    assert!(e.contains("transform"), "the refusal must say WHY: {e}");
}

// ======================================================================
//  SPEC-LIT S13.4.1: the pair test
// ======================================================================

/// One turned setting. `pre` is an enabling edit applied to BOTH sides, for
/// entries that bite only through another - the discipline S13.4.1 records,
/// so that the two sides still differ in exactly one setting.
struct Knob {
    name: &'static str,
    pre: fn(&mut ParcelControls, &mut Injector),
    turn: fn(&mut ParcelControls, &mut Injector),
}

fn nothing(_: &mut ParcelControls, _: &mut Injector) {}

/// **Every setting this module claims to honour owes a pair, and here they
/// are.** Two runs of the same function, differing in exactly one field of
/// the same control struct, must write DIFFERENT output. If they are
/// byte-identical, the setting never reached the kernel, and the test names
/// which one.
///
/// Two settings are not in the table and say why here rather than by being
/// absent:
///
/// * `persistent_blocks` is launch geometry and is REQUIRED to be inert;
///   `the_persistent_grid_geometry_does_not_change_the_answer` asserts that
///   instead. This is S13.4.1's admissible exception.
/// * `physics` has a one-item menu. Its whole job is to refuse the others, so
///   there is no second value to turn; `every_unsupported_setting_is_refused_
///   by_name_with_the_menu` is its test.
#[test]
fn every_wired_setting_changes_what_the_run_writes() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([10, 10, 10], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();

    let rebound: fn(&mut ParcelControls, &mut Injector) =
        |c, _| c.wall = WallAction::Rebound;
    let no_substeps: fn(&mut ParcelControls, &mut Injector) = |c, _| c.cfl = 100.0;

    let knobs: &[Knob] = &[
        Knob { name: "dragModel", pre: nothing, turn: |c, _| c.drag = DragModel::Stokes },
        Knob { name: "wallInteraction", pre: nothing, turn: |c, _| c.wall = WallAction::Rebound },
        Knob { name: "restitution", pre: rebound, turn: |c, _| c.restitution = 0.4 },
        Knob { name: "tangentialLoss", pre: rebound, turn: |c, _| c.tangential_loss = 0.6 },
        Knob {
            name: "gravity",
            pre: nothing,
            turn: |c, _| c.gravity = Vec3::new(0.0, 0.0, -3.0),
        },
        Knob { name: "rhoLiquid", pre: nothing, turn: |c, _| c.rho_liquid = 800.0 },
        Knob { name: "muGas", pre: nothing, turn: |c, _| c.mu_gas = 3.0e-5 },
        Knob { name: "addedMass", pre: nothing, turn: |c, _| c.added_mass = true },
        Knob { name: "cfl", pre: nothing, turn: |c, _| c.cfl = 0.15 },
        Knob { name: "maxSubSteps", pre: nothing, turn: |c, _| c.max_substeps = 1 },
        Knob { name: "maxWalk", pre: no_substeps, turn: |c, _| c.max_walk = 1 },
        Knob { name: "capacity", pre: nothing, turn: |c, _| c.capacity = 24 },
        Knob { name: "injector/speed", pre: nothing, turn: |_, i| i.speed = 1.5 },
        Knob { name: "injector/diameter", pre: nothing, turn: |_, i| i.diameter = 1e-4 },
        Knob { name: "injector/temperature", pre: nothing, turn: |_, i| i.temperature = 350.0 },
        Knob { name: "injector/massFlow", pre: nothing, turn: |_, i| i.mass_flow = 2e-3 },
        Knob {
            name: "injector/parcelsPerEvent",
            pre: nothing,
            turn: |_, i| i.parcels_per_event = 4,
        },
        Knob { name: "injector/interval", pre: nothing, turn: |_, i| i.interval = 3.0 * BASE_DT },
        Knob {
            name: "injector/position",
            pre: nothing,
            turn: |_, i| i.position = Vec3::new(0.4, 0.6, 0.3),
        },
        Knob {
            name: "injector/axis",
            pre: nothing,
            turn: |_, i| i.axis = Vec3::new(0.0, 0.4, -1.0),
        },
        Knob { name: "injector/coneHalfAngle", pre: nothing, turn: |_, i| i.cone_half_angle = 0.2 },
        Knob { name: "injector/standoff", pre: nothing, turn: |_, i| i.standoff = 0.06 },
    ];

    for k in knobs {
        let mut ctrl_a = base_controls();
        let mut inj_a = base_injector();
        (k.pre)(&mut ctrl_a, &mut inj_a);
        let mut ctrl_b = ctrl_a;
        let mut inj_b = inj_a;
        (k.turn)(&mut ctrl_b, &mut inj_b);

        assert!(
            ctrl_a != ctrl_b || inj_a != inj_b,
            "{}: the knob turned nothing - it would compare two identical runs and pass",
            k.name
        );

        let a = run_case(&gpu, &hm, &gm, ctrl_a, &[inj_a], BASE_STEPS).unwrap();
        let b = run_case(&gpu, &hm, &gm, ctrl_b, &[inj_b], BASE_STEPS).unwrap();

        let pa = scratch(&format!("knob_a_{}", k.name.replace('/', "_")));
        let pb = scratch(&format!("knob_b_{}", k.name.replace('/', "_")));
        crate::io::vtu::write_parcels_vtp(&pa, &a, Some(1.0)).unwrap();
        crate::io::vtu::write_parcels_vtp(&pb, &b, Some(1.0)).unwrap();

        assert!(
            !a.live().is_empty(),
            "{}: the base run wrote no parcels at all, so the comparison measures nothing",
            k.name
        );
        assert_ne!(
            std::fs::read(&pa).unwrap(),
            std::fs::read(&pb).unwrap(),
            "{} is INERT: two runs differing only in it wrote byte-identical output \
             (SPEC-LIT S13.4.1)",
            k.name
        );
    }
}

// ======================================================================
//  Output
// ======================================================================

/// S66.13: the `.vtp` carries as many points as there are live parcels, and
/// the identity survives the round trip through two `Float64` halves.
#[test]
fn the_parcel_output_writes_one_point_per_live_parcel() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([10, 10, 10], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let inj = [base_injector()];
    let s = run_case(&gpu, &hm, &gm, base_controls(), &inj, BASE_STEPS).unwrap();

    let path = scratch("output");
    crate::io::vtu::write_parcels_vtp(&path, &s, Some(1.0)).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let head = String::from_utf8_lossy(&bytes[..bytes.len().min(4096)]).to_string();

    let n = s.live().len();
    assert!(n > 0);
    assert!(head.contains(&format!("NumberOfPoints=\"{n}\"")), "{head}");
    assert!(head.contains(&format!("NumberOfVerts=\"{n}\"")), "{head}");
    assert!(head.contains("type=\"PolyData\""));
    for name in ["U", "d", "T", "nP", "uidHigh", "uidLow", "cell"] {
        assert!(head.contains(&format!("Name=\"{name}\"")), "missing {name}:\n{head}");
    }

    // Both halves of the identity are exactly representable in f64, so the
    // full 64 bits are recoverable from the file.
    for &i in s.live().iter().take(8) {
        let hi = (s.uid[i] >> 32) as f64;
        let lo = (s.uid[i] & 0xffff_ffff) as f64;
        assert_eq!(((hi as u64) << 32) | lo as u64, s.uid[i]);
    }
}
