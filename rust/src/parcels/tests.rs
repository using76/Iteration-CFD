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

    // S78.11: `stick` was refused here until S78 built the deposit and the
    // ledger that accounts for it. It is supported now, and the line that
    // used to refuse it is a record of what changed rather than a deletion.
    assert_eq!(WallAction::from_name("stick").unwrap(), WallAction::Stick);
    let e = WallAction::from_name("film").unwrap_err().to_string();
    assert!(e.contains("film") && e.contains("no film transport"), "{e}");
    let e = WallAction::from_name("splash").unwrap_err().to_string();
    assert!(e.contains("splash") && e.contains("population growth"), "{e}");

    // The refusal list here has now shrunk TWICE, and the shape of this test
    // is the record of it. `heating` was refused until S68.5 implemented the
    // sensible-heat half of the two Ranz & Marshall papers; `evaporating` was
    // refused until S76 implemented the other half. Both are supported now,
    // and what is still refused is what genuinely does not exist.
    for name in ["evaporating", "evaporation", "heatAndMassTransfer"] {
        assert_eq!(
            ParcelPhysics::from_name(name).unwrap(),
            ParcelPhysics::Evaporating,
            "S76 supports `{name}`"
        );
    }
    assert_eq!(
        ParcelPhysics::from_name("heating").unwrap(),
        ParcelPhysics::Heating,
        "S68.5 supports `heating`; if this ever fails, the refusal above has to \
         grow it back"
    );
    for name in ["reacting", "combusting"] {
        let e = ParcelPhysics::from_name(name).unwrap_err().to_string();
        assert!(e.contains(name), "{e}");
        assert!(e.contains("species source"), "the refusal must name what is missing: {e}");
        assert!(e.contains("evaporating"), "the alternative must be printed: {e}");
    }

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
        evaporation: EvaporationControls::default(),
        impact: WallImpactControls::default(),
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
        p.step(gpu, &u, &rho, None, None, dt)?;
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
        evaporation: EvaporationControls::default(),
        impact: WallImpactControls::default(),
        persistent_blocks: None,
    };
    let dt: Scalar = 1.0;
    let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[], dt).unwrap();
    p.seed(&gpu, &hm, &seeds).unwrap();
    let (u, rho) = still_gas(&gpu, &gm, 1.2).unwrap();
    p.step(&gpu, &u, &rho, None, None, dt).unwrap();

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
        evaporation: EvaporationControls::default(),
        impact: WallImpactControls::default(),
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
    p.step(&gpu, &u, &rho, None, None, 1.0).unwrap();

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
        evaporation: EvaporationControls::default(),
        impact: WallImpactControls::default(),
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
        p.step(&gpu, &u, &rho, None, None, dt).unwrap();
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
        evaporation: EvaporationControls::default(),
        impact: WallImpactControls::default(),
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
        p.step(gpu, &u, &rho, None, None, BASE_DT)?;
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
            p.step(&gpu, &u, &rho, None, None, BASE_DT).unwrap();
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
        .capture(|_| p.step(&gpu, &u, &rho, None, None, BASE_DT))
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
        p.step(&gpu, &u, &rho, None, None, dt).unwrap();
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
        p.step(&gpu, &u, &rho, None, None, BASE_DT).unwrap();
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
        p.step(&gpu, &u, &rho, None, None, BASE_DT).unwrap();
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

// ======================================================================
//  SPEC-LIT S76: droplet heating and evaporation, on the device
// ======================================================================

/// One droplet, alone, in a box of still gas at a fixed state - the fixture
/// every S76 gate below is posed on, because a single suspended droplet is
/// what Ranz & Marshall measured and it isolates the closure from the mesh,
/// the walk and the injector all at once.
struct Drop {
    hm: HostMesh,
    gm: GpuMesh,
    u: GpuVectorField,
    rho: DevBuf<Scalar>,
    tg: DevBuf<Scalar>,
    yv: DevBuf<Scalar>,
    gas: GasState,
}

fn droplet_box(gpu: &Gpu, t_gas: Scalar, rh: Scalar, u_rel: Scalar) -> Drop {
    let hm = block([4, 4, 4], [1.0, 1.0, 1.0], ["patch"; 6]);
    let gm = GpuMesh::upload(gpu, &hm).unwrap();
    // The GAS moves and the droplet is released at rest, which is how the
    // suspended-droplet experiments were run.
    let mut u = GpuVectorField::zeros(gpu, &gm, "U").unwrap();
    u.f = gpu.upload(&vec![Vec3::new(u_rel, 0.0, 0.0); gm.n_cells]).unwrap();
    let rho_gas = 101_325.0 * 28.966e-3 / (crate::parcels::evaporation::R_UNIVERSAL * t_gas);
    let y = crate::psychro::yv_from_t_rh_p(t_gas, rh, 101_325.0);
    Drop {
        rho: gpu.upload(&vec![rho_gas; gm.n_cells]).unwrap(),
        tg: gpu.upload(&vec![t_gas; gm.n_cells]).unwrap(),
        yv: gpu.upload(&vec![y; gm.n_cells]).unwrap(),
        gas: GasState {
            t: t_gas,
            y_vapour: y,
            rho: rho_gas,
            mu: 1.8e-5,
            k: 0.026,
            cp: 1005.0,
            u_rel,
        },
        hm,
        gm,
        u,
    }
}

fn evaporating_controls(ev: EvaporationControls) -> ParcelControls {
    ParcelControls {
        capacity: 4,
        drag: DragModel::None,
        physics: ParcelPhysics::Evaporating,
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
        evaporation: ev,
        impact: WallImpactControls::default(),
        persistent_blocks: None,
    }
}

fn one_droplet<'m>(
    gpu: &Gpu,
    b: &'m Drop,
    ctrl: ParcelControls,
    d0: Scalar,
    t0: Scalar,
    n_p: Scalar,
    dt: Scalar,
) -> Parcels<'m> {
    let mut p = Parcels::new(gpu, &b.hm, &b.gm, ctrl, &[], dt).unwrap();
    p.seed(
        gpu,
        &b.hm,
        &[SeedParcel {
            position: Vec3::new(0.5, 0.5, 0.5),
            velocity: Vec3::ZERO,
            diameter: d0,
            temperature: t0,
            n_p,
            uid: None,
        }],
    )
    .unwrap();
    p
}

/// The kernel's closure and the host's are one closure. One sub-step from a
/// known state, with the mass and the heat read back and compared against
/// what `evaporation::droplet_rate` says they should be.
///
/// This is the test that would catch a coefficient typed differently on the
/// two sides, which is the only kind of error a `d^2` gate posed against the
/// host closed form could not see.
#[test]
fn the_device_closure_is_the_host_closure() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let dt: Scalar = 1e-4;
    let d0: Scalar = 5e-4;
    let t0: Scalar = 290.0;
    for (t_gas, rh, u_rel, transfer, sat) in [
        (298.15, 0.30, 0.0, MassTransfer::AbramzonSirignano, SaturationCurve::HylandWexler),
        (298.15, 0.30, 2.5, MassTransfer::AbramzonSirignano, SaturationCurve::HylandWexler),
        (350.00, 0.05, 1.0, MassTransfer::Spalding, SaturationCurve::HylandWexler),
        (320.00, 0.60, 0.0, MassTransfer::RanzMarshall, SaturationCurve::ClausiusClapeyron),
    ] {
        let b = droplet_box(&gpu, t_gas, rh, u_rel);
        let ev = EvaporationControls {
            transfer,
            saturation: sat,
            ..EvaporationControls::default()
        };
        let ctrl = ParcelControls { max_substeps: 1, ..evaporating_controls(ev) };
        let mut p = one_droplet(&gpu, &b, ctrl, d0, t0, 1.0, dt);
        let t_boil = p.boiling_temperature();
        p.step(&gpu, &b.u, &b.rho, Some(&b.tg), Some(&b.yv), dt).unwrap();
        let s = p.snapshot(&gpu).unwrap();

        // The host closure, integrated by the SAME one-sub-step rule (76.10)
        // - written out here rather than called, so that this is a check on
        //   the kernel and not a second call to the same function.
        let r = droplet_rate(&ev, t_boil, d0, t0, &b.gas);
        let mp = 1000.0 * std::f64::consts::FRAC_PI_6 * d0 * d0 * d0;
        let cap = mp * ctrl.c_liquid;
        let lam = (r.conductance + r.d_cooling_d_t) / cap;
        let w_t = -(-lam * dt).exp_m1();
        let tau = w_t / lam;
        let teq = t0 + (r.conductance * (b.gas.t - t0) + r.mdot * r.h_v)
            / (r.conductance + r.d_cooling_d_t);
        let d_e = cap * w_t * (teq - t0);
        let qc = r.conductance * ((b.gas.t - teq) * dt + (teq - t0) * tau);
        let dm = (qc - d_e) / r.h_v;
        let d_new = d0 * (1.0 - dm / mp).cbrt();

        let rel = (s.d[0] - d_new).abs() / (d0 - d_new).abs().max(1e-30);
        assert!(
            rel < 1e-9,
            "{} / {}, T_g {t_gas}: device d {}, host {d_new} (change {} vs {})",
            transfer.name(),
            sat.name(),
            s.d[0],
            d0 - s.d[0],
            d0 - d_new
        );
        // ... and the accumulators are the state change, not a second model
        // of it.
        // The accumulator is `rho_l (pi/6)(d_0^3 - d^3)` - ONE expression
        // over the two endpoint diameters, which is what makes the sum over a
        // run telescope. It is NOT `m(d_0) - m(d)` evaluated separately:
        // those two masses agree to fifteen digits and their difference does
        // not, so the grouping is the claim, and it is compared here in the
        // grouping the kernel uses.
        let want = 1000.0
            * std::f64::consts::FRAC_PI_6
            * (d0 - s.d[0])
            * (d0 * d0 + d0 * s.d[0] + s.d[0] * s.d[0]);
        assert!(
            (s.mass_lost[0] - want).abs() <= 1e-14 * want.abs(),
            "the accumulator {} is not the state change {want}",
            s.mass_lost[0]
        );
    }
}

/// **Gate 76-A.** The `d^2` law: an isolated droplet in a quiescent gas,
/// released AT its steady temperature, must shrink so that `d^2` falls
/// linearly at exactly the closed-form slope `K`.
///
/// The droplet is seeded at `evaporation::steady_temperature`, which is a
/// bracketed root find on the host; the kernel never solves that balance, it
/// relaxes towards it. So "the temperature does not move" and "the slope is
/// the closed form" are two independent statements and both are asserted.
#[test]
fn gate_76a_the_d2_law_holds_against_its_closed_form() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let b = droplet_box(&gpu, 298.15, 0.30, 0.0);
    let ev = EvaporationControls::default();
    let ctrl = evaporating_controls(ev);
    let dt: Scalar = 2e-3;
    let d0: Scalar = 5e-4;

    let t_boil = ev.boiling_temperature().unwrap();
    let t_wet = steady_temperature(&ev, t_boil, d0, &b.gas).unwrap();
    let k_closed = d2_law_slope(&ev, t_boil, ctrl.rho_liquid, d0, t_wet, &b.gas);

    let mut p = one_droplet(&gpu, &b, ctrl, d0, t_wet, 1.0, dt);
    let steps = 400;
    let mut worst_d2: Scalar = 0.0;
    let mut worst_t: Scalar = 0.0;
    for n in 1..=steps {
        p.step(&gpu, &b.u, &b.rho, Some(&b.tg), Some(&b.yv), dt).unwrap();
        let s = p.snapshot(&gpu).unwrap();
        let t = n as Scalar * dt;
        let want = d0 * d0 - k_closed * t;
        let got = s.d[0] * s.d[0];
        worst_d2 = worst_d2.max((got - want).abs() / (d0 * d0 - want).abs());
        worst_t = worst_t.max((s.temperature[0] - t_wet).abs());
    }
    let s = p.snapshot(&gpu).unwrap();
    println!(
        "[76-A] 500 um water in still 25 C air at 30 % rh: K {k_closed:.6e} m2/s, \
         T_wet {t_wet:.4} K; after {:.2} s, d {:.3} um, d2 error {worst_d2:.3e}, \
         T drift {worst_t:.2e} K",
        steps as Scalar * dt,
        1e6 * s.d[0],
    );
    assert!(worst_t < 1e-3, "the droplet drifted {worst_t} K off its steady temperature");
    assert!(
        worst_d2 < 2e-3,
        "d^2 departed the closed-form line by {worst_d2} of the drop so far"
    );
}

/// ... and the departure is the TIME-STEP error and nothing else: halving
/// `dt` halves it. Without this, gate 76-A could be passed by an integrator
/// that is wrong in a way the tolerance happens to admit.
#[test]
fn the_d2_law_error_is_first_order_in_the_step() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let b = droplet_box(&gpu, 298.15, 0.30, 0.0);
    let ev = EvaporationControls::default();
    let d0: Scalar = 5e-4;
    let t_boil = ev.boiling_temperature().unwrap();
    let t_wet = steady_temperature(&ev, t_boil, d0, &b.gas).unwrap();
    let k_closed = d2_law_slope(&ev, t_boil, 1000.0, d0, t_wet, &b.gas);
    let horizon: Scalar = 0.8;

    let run = |dt: Scalar, cfl: Scalar| -> Scalar {
        let ctrl = evaporating_controls(EvaporationControls { cfl, ..ev });
        let mut p = one_droplet(&gpu, &b, ctrl, d0, t_wet, 1.0, dt);
        let n = (horizon / dt).round() as usize;
        for _ in 0..n {
            p.step(&gpu, &b.u, &b.rho, Some(&b.tg), Some(&b.yv), dt).unwrap();
        }
        let s = p.snapshot(&gpu).unwrap();
        let want = d0 * d0 - k_closed * (n as Scalar * dt);
        (s.d[0] * s.d[0] - want).abs() / (d0 * d0 - want)
    };
    // `cfl = 1` disables the (76.10) sub-step bound in all but name, so the
    // step being refined is the one the case asked for.
    let e1 = run(1e-2, 1.0);
    let e2 = run(5e-3, 1.0);
    let e4 = run(2.5e-3, 1.0);
    println!("[76.10] d^2 error against dt: {e1:.3e} -> {e2:.3e} -> {e4:.3e}");
    assert!(e2 < 0.7 * e1, "halving dt did not halve the error: {e1} -> {e2}");
    assert!(e4 < 0.7 * e2, "halving dt did not halve the error: {e2} -> {e4}");
}

/// **Gate 76-B.** The temperature an evaporating droplet settles at.
///
/// Two comparisons, and they are different in kind. The first is against
/// this crate's own closed form, and it must close to round-off: the kernel's
/// exponential relaxation and the host's bracketed root find are two ways of
/// finding the same fixed point. The second is against S54's ASHRAE
/// psychrometric wet bulb, which is a DIFFERENT quantity - it assumes a
/// psychrometric ratio of one, and a droplet's balance carries the Lewis
/// number instead. The gap is reported.
#[test]
fn gate_76b_the_droplet_settles_at_its_wet_bulb_temperature() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let ev = EvaporationControls::default();
    let t_boil = ev.boiling_temperature().unwrap();
    // A 100 um droplet, because the thermal time constant is
    // rho c_l d^2/(6 Nu k) and at 500 um that is 3.3 s - a gate that ran for
    // four of those would be reporting how far the relaxation had got, not
    // where it was going. At 100 um it is 0.13 s and four seconds is thirty
    // time constants. This was measured, not assumed: the first version of
    // this gate used 500 um and missed the balance by 0.62 K.
    let dt: Scalar = 2e-3;
    let d0: Scalar = 1e-4;
    println!("[76-B] a 100 um droplet released at 25 C, relaxed for 4 s:");
    println!("        rh    T_device(C)  T_closed(C)  ASHRAE T_wb(C)   gap(K)");
    let mut worst_closed: Scalar = 0.0;
    let mut worst_ashrae: Scalar = 0.0;
    for &rh in &[0.10, 0.30, 0.50, 0.90] {
        let b = droplet_box(&gpu, 298.15, rh, 0.0);
        let ctrl = evaporating_controls(ev);
        let mut p = one_droplet(&gpu, &b, ctrl, d0, 298.15, 1.0, dt);
        for _ in 0..2000 {
            p.step(&gpu, &b.u, &b.rho, Some(&b.tg), Some(&b.yv), dt).unwrap();
        }
        let s = p.snapshot(&gpu).unwrap();
        let t_dev = s.temperature[0];
        // The closed form is re-evaluated at the diameter the droplet has
        // NOW, which is the honest comparison - though (76.11) says it
        // cannot matter, and the assertion below is what says so.
        let t_closed = steady_temperature(&ev, t_boil, s.d[0], &b.gas).unwrap();
        let w = crate::psychro::w_from_t_rh_p(298.15, rh, 101_325.0);
        let t_wb = crate::psychro::t_wb(298.15, w, 101_325.0).unwrap();
        worst_closed = worst_closed.max((t_dev - t_closed).abs());
        worst_ashrae = worst_ashrae.max((t_dev - 273.15 - t_wb).abs());
        println!(
            "        {rh:4.2}   {:9.4}    {:9.4}    {t_wb:11.4}   {:+7.4}",
            t_dev - 273.15,
            t_closed - 273.15,
            t_dev - 273.15 - t_wb
        );
    }
    println!(
        "        against this crate's own balance: {worst_closed:.3e} K; \
         against ASHRAE: {worst_ashrae:.3} K"
    );
    assert!(
        worst_closed < 1e-6,
        "the kernel settled {worst_closed} K away from the balance it is relaxing to"
    );
    // The published bar. The design note's own figure for the wet-bulb
    // plateau is +/- 2 K, and this is the number held against it.
    assert!(
        worst_ashrae < 2.0,
        "the droplet settled {worst_ashrae} K from the psychrometric wet bulb"
    );
}

/// **Gate 76-C.** The parcel's own mass conservation: what the pool holds
/// plus what it says it gave up is what it started with, to round-off, and
/// over a run long enough to remove most of the liquid.
///
/// It is an identity rather than a tolerance by construction - (76.10)
/// forms the accumulator as the difference of the step's two endpoint
/// masses, one subtraction - so the only error here is the summation of the
/// per-step numbers on the host.
#[test]
fn gate_76c_the_parcel_conserves_its_own_mass() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    // Still gas and no drag, so the droplet does not move: an escaping
    // parcel would leave with mass this pool never sees again, and the
    // conservation statement being made here is about the POOL. The gap an
    // escape opens is real and is S68.9 row 4's; it is not what this gate is
    // measuring, so the fixture removes it rather than tolerating it.
    let b = droplet_box(&gpu, 340.0, 0.05, 0.0);
    let ev = EvaporationControls::default();
    let ctrl = evaporating_controls(ev);
    let dt: Scalar = 2e-3;
    let d0: Scalar = 2e-4;
    let n_p: Scalar = 1.7e5;
    let mut p = one_droplet(&gpu, &b, ctrl, d0, 293.15, n_p, dt);

    let m0 = p.snapshot(&gpu).unwrap().liquid_mass(ctrl.rho_liquid);
    let mut given: Scalar = 0.0;
    let mut worst: Scalar = 0.0;
    let mut steps = 0;
    for _ in 0..4000 {
        p.step(&gpu, &b.u, &b.rho, Some(&b.tg), Some(&b.yv), dt).unwrap();
        let s = p.snapshot(&gpu).unwrap();
        given += s.total_mass_lost() + s.dead_mass_lost();
        steps += 1;
        let held = s.liquid_mass(ctrl.rho_liquid);
        worst = worst.max((held + given - m0).abs() / m0);
        if s.live().is_empty() {
            break;
        }
    }
    let s = p.snapshot(&gpu).unwrap();
    let st = p.stats(&gpu).unwrap();
    println!(
        "[76-C] a 200 um droplet standing for {n_p:.1e} of them, in 340 K air: \
         {} steps, {:.4} % of the liquid evaporated, worst mass defect {worst:.3e}",
        steps,
        100.0 * given / m0
    );
    assert!(given > 0.5 * m0, "only {} of the mass evaporated", given / m0);
    assert!(worst < 1e-13, "the parcel lost {worst} of its mass to bookkeeping");
    // A droplet that runs out is REMOVED and counted, not left as a
    // zero-diameter parcel that every later kernel divides by.
    if s.live().is_empty() {
        assert_eq!(st.n_evaporated, 1, "the spent parcel was not counted");
        assert_eq!(st.n_escaped, 0);
    }
}

/// (76.9): a droplet released above the boiling point flashes down to it and
/// then evaporates at Godsave's heat-limited rate - a `d^2` law with a
/// DIFFERENT closed form from gate 76-A's, which is the point.
#[test]
fn a_superheated_droplet_is_capped_at_the_boiling_point_and_boils() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let b = droplet_box(&gpu, 700.0, 0.0, 0.0);
    let ev = EvaporationControls::default();
    let ctrl = evaporating_controls(ev);
    let dt: Scalar = 2e-4;
    let d0: Scalar = 3e-4;
    let t_boil = ev.boiling_temperature().unwrap();
    let mut p = one_droplet(&gpu, &b, ctrl, d0, t_boil + 40.0, 1.0, dt);

    p.step(&gpu, &b.u, &b.rho, Some(&b.tg), Some(&b.yv), dt).unwrap();
    let first = p.snapshot(&gpu).unwrap();
    assert!(
        first.temperature[0] <= t_boil + 1e-12,
        "a superheated droplet stayed at {} K, above T_boil {t_boil}",
        first.temperature[0]
    );

    // Godsave's slope, written out from the correlation.
    let hv = crate::parcels::evaporation::latent_heat(&ev.liquid, t_boil);
    let b_t = ev.liquid.cp_vapour * (b.gas.t - t_boil) / hv;
    let k_godsave =
        8.0 * (b.gas.k / ev.liquid.cp_vapour) * (1.0 + b_t).ln() / ctrl.rho_liquid;

    let d1 = first.d[0];
    let mut worst: Scalar = 0.0;
    for n in 1..=200 {
        p.step(&gpu, &b.u, &b.rho, Some(&b.tg), Some(&b.yv), dt).unwrap();
        let s = p.snapshot(&gpu).unwrap();
        if s.live().is_empty() {
            break;
        }
        let want = d1 * d1 - k_godsave * (n as Scalar * dt);
        worst = worst.max((s.d[0] * s.d[0] - want).abs() / (d1 * d1 - want).abs());
        assert!(s.temperature[0] <= t_boil + 1e-12);
    }
    println!(
        "[76.9] boiling in 700 K air: T capped at {t_boil:.3} K, B_T {b_t:.4}, \
         K {k_godsave:.4e} m2/s, worst departure {worst:.3e}"
    );
    assert!(worst < 2e-3, "the boiling d^2 slope departed Godsave's by {worst}");
}

/// (76.7): a droplet in a saturated gas neither shrinks nor grows, and one in
/// a supersaturated gas GROWS - both on the device, and neither is a NaN.
#[test]
fn a_droplet_in_wet_air_stops_and_in_wetter_air_grows() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let ev = EvaporationControls::default();
    let dt: Scalar = 1e-3;
    let d0: Scalar = 3e-4;
    let t_boil = ev.boiling_temperature().unwrap();

    // Saturated: seeded at the gas temperature, so Y_s == Y_g to the accuracy
    // of the two correlations and the droplet barely moves.
    let sat = droplet_box(&gpu, 293.15, 1.0, 0.0);
    let mut p = one_droplet(&gpu, &sat, evaporating_controls(ev), d0, 293.15, 1.0, dt);
    for _ in 0..200 {
        p.step(&gpu, &sat.u, &sat.rho, Some(&sat.tg), Some(&sat.yv), dt).unwrap();
    }
    let s = p.snapshot(&gpu).unwrap();
    assert!(s.d[0].is_finite());
    assert!(
        (s.d[0] - d0).abs() < 1e-3 * d0,
        "a droplet in saturated air moved from {d0} to {}",
        s.d[0]
    );

    // Supersaturated: 20 % above saturation. The droplet grows, and the mass
    // accumulator goes negative with it.
    let mut wet = droplet_box(&gpu, 293.15, 1.0, 0.0);
    let y_hi = 1.2 * wet.gas.y_vapour;
    wet.yv = gpu.upload(&vec![y_hi; wet.gm.n_cells]).unwrap();
    wet.gas.y_vapour = y_hi;
    let mut q = one_droplet(&gpu, &wet, evaporating_controls(ev), d0, 293.15, 1.0, dt);
    for _ in 0..200 {
        q.step(&gpu, &wet.u, &wet.rho, Some(&wet.tg), Some(&wet.yv), dt).unwrap();
    }
    let s = q.snapshot(&gpu).unwrap();
    println!(
        "[76.7] in air 20 % above saturation a 300 um droplet grew to {:.3} um; \
         T_p {:.3} C against a T_wb of {:.3} C",
        1e6 * s.d[0],
        s.temperature[0] - 273.15,
        steady_temperature(&ev, t_boil, s.d[0], &wet.gas).unwrap() - 273.15
    );
    assert!(s.d[0] > d0, "a droplet in supersaturated air shrank to {}", s.d[0]);
    assert!(s.mass_lost[0] < 0.0, "condensation did not show as a negative loss");
}

/// SPEC-LIT S76.10, the bitwise claim, MEASURED.
///
/// The by-construction half is that every evaporation statement in
/// `cuda/parcels.cu` is inside `if (evaporating)` and the inert/heating
/// arithmetic is textually unchanged. This is the other half: a heating pool
/// run twice, once with the shipped evaporation settings and once with every
/// one of them set to something else - a different liquid, a different
/// saturation curve, a different blowing model, a different sub-step bound -
/// and the parcel state and both coupling accumulators must be bit for bit
/// the same.
#[test]
fn the_evaporation_settings_cannot_move_a_heating_parcel() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let b = droplet_box(&gpu, 340.0, 0.2, 1.5);
    let dt: Scalar = 1e-3;

    let run = |ev: EvaporationControls, physics: ParcelPhysics| -> ParcelSnapshot {
        let ctrl = ParcelControls {
            physics,
            drag: DragModel::SchillerNaumann,
            gravity: Vec3::new(0.0, 0.0, -9.81),
            ..evaporating_controls(ev)
        };
        let mut p = one_droplet(&gpu, &b, ctrl, 3e-4, 293.15, 3.0, dt);
        let t = if physics == ParcelPhysics::Inert { None } else { Some(&b.tg) };
        for _ in 0..40 {
            p.step(&gpu, &b.u, &b.rho, t, None, dt).unwrap();
        }
        p.snapshot(&gpu).unwrap()
    };

    let shipped = EvaporationControls::default();
    let other = EvaporationControls {
        saturation: SaturationCurve::ClausiusClapeyron,
        transfer: MassTransfer::RanzMarshall,
        liquid: LiquidProperties::benzene(),
        w_carrier: 0.044,
        p_ambient: 50_000.0,
        cfl: 0.9,
    };
    for physics in [ParcelPhysics::Inert, ParcelPhysics::Heating] {
        let a = run(shipped, physics);
        let c = run(other, physics);
        assert!(
            snapshot_bits_eq(&a, &c),
            "{}: the evaporation settings moved the parcel state",
            physics.name()
        );
        assert!(
            vec_bits_eq(&a.impulse, &c.impulse) && bits_eq(&a.exchange, &c.exchange),
            "{}: the evaporation settings moved the S68 momentum accumulators",
            physics.name()
        );
        assert!(
            bits_eq(&a.heat, &c.heat) && bits_eq(&a.heat_exchange, &c.heat_exchange),
            "{}: the evaporation settings moved the S68 energy accumulators",
            physics.name()
        );
        assert!(a.mass_lost.is_empty() && a.latent.is_empty());
        // ... and the two mass accessors say "nothing" rather than index an
        // array the pool never allocated.
        assert_eq!(a.total_mass_lost(), 0.0);
        assert_eq!(a.dead_mass_lost(), 0.0);
    }
}

/// SPEC-LIT S76.10's other half: the accumulators an EVAPORATING parcel
/// writes are the change in its own state, and the two energies it reports
/// add up to what its temperature and mass actually did.
#[test]
fn the_accumulators_are_the_change_in_the_parcels_own_state() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let b = droplet_box(&gpu, 330.0, 0.1, 0.5);
    let ev = EvaporationControls::default();
    let ctrl = evaporating_controls(ev);
    let dt: Scalar = 1e-3;
    let d0: Scalar = 4e-4;
    let mut p = one_droplet(&gpu, &b, ctrl, d0, 293.15, 1.0, dt);

    let mut prev = p.snapshot(&gpu).unwrap();
    let mut worst_mass: Scalar = 0.0;
    let mut worst_energy: Scalar = 0.0;
    for _ in 0..300 {
        p.step(&gpu, &b.u, &b.rho, Some(&b.tg), Some(&b.yv), dt).unwrap();
        let s = p.snapshot(&gpu).unwrap();
        // Mass: BITWISE, because the accumulator is the difference of the
        // two endpoint masses and nothing else.
        let m_before = 1000.0 * std::f64::consts::FRAC_PI_6 * prev.d[0].powi(3);
        let m_after = 1000.0 * std::f64::consts::FRAC_PI_6 * s.d[0].powi(3);
        let (a, c) = (prev.d[0], s.d[0]);
        let want =
            1000.0 * std::f64::consts::FRAC_PI_6 * (a - c) * (a * a + a * c + c * c);
        worst_mass = worst_mass.max((s.mass_lost[0] - want).abs() / want.abs());
        // Energy: the droplet's own budget. `m c_l dT = Q_conv - dm h_v` is
        // closed sub-step by sub-step by construction, so what is left here
        // is the change of `m` WITHIN the step, which is second order in dt
        // and is the number this reports rather than hides.
        let d_int = m_after * ctrl.c_liquid * s.temperature[0]
            - m_before * ctrl.c_liquid * prev.temperature[0];
        let budget = s.heat[0] - s.latent[0] - want * ctrl.c_liquid * s.temperature[0];
        let scale = s.heat[0].abs().max(s.latent[0].abs()).max(1e-30);
        worst_energy = worst_energy.max((d_int - budget).abs() / scale);
        prev = s;
    }
    println!(
        "[76.10] mass accumulator against the state change: {worst_mass:.3e}; \
         the droplet energy budget closes to {worst_energy:.3e}"
    );
    assert!(worst_mass < 1e-14, "the mass accumulator is off by {worst_mass}");
    assert!(worst_energy < 1e-5, "the droplet energy budget is off by {worst_energy}");
}

/// SPEC-LIT S13.4.1: every setting S76 adds changes the answer, and is shown
/// to.
#[test]
fn every_evaporation_setting_changes_the_answer() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let b = droplet_box(&gpu, 340.0, 0.10, 1.0);
    let dt: Scalar = 1e-3;
    let d0: Scalar = 3e-4;
    let after = |ev: EvaporationControls| -> (Scalar, Scalar) {
        let mut p = one_droplet(&gpu, &b, evaporating_controls(ev), d0, 293.15, 1.0, dt);
        for _ in 0..200 {
            p.step(&gpu, &b.u, &b.rho, Some(&b.tg), Some(&b.yv), dt).unwrap();
        }
        let s = p.snapshot(&gpu).unwrap();
        (s.d[0], s.temperature[0])
    };
    let base = EvaporationControls::default();
    let (d_base, t_base) = after(base);
    for (name, ev) in [
        (
            "saturation",
            EvaporationControls { saturation: SaturationCurve::ClausiusClapeyron, ..base },
        ),
        ("transfer=spalding", EvaporationControls { transfer: MassTransfer::Spalding, ..base }),
        (
            "transfer=ranzMarshall",
            EvaporationControls { transfer: MassTransfer::RanzMarshall, ..base },
        ),
        ("pAmbient", EvaporationControls { p_ambient: 80_000.0, ..base }),
        ("wCarrier", EvaporationControls { w_carrier: 0.030, ..base }),
        (
            "liquid",
            EvaporationControls {
                saturation: SaturationCurve::ClausiusClapeyron,
                liquid: LiquidProperties {
                    h_v_boil: 2.0e6,
                    ..LiquidProperties::water()
                },
                ..base
            },
        ),
    ] {
        let (d, t) = after(ev);
        assert!(
            (d - d_base).abs() > 1e-12 * d0 || (t - t_base).abs() > 1e-9,
            "{name} did not move the answer: d {d} vs {d_base}, T {t} vs {t_base}"
        );
    }
    // ... and `cfl` is a sub-step BOUND, so it only moves the answer when it
    // actually bites: at `dt = 1 ms` a 300 um droplet loses `1e-4` of its
    // `d^2` in a step and no setting in `(0, 1]` divides that further. So the
    // leg that demonstrates it uses a step coarse enough for the bound to
    // engage, which is also the only regime in which it is worth having.
    let coarse = |cfl: Scalar| -> Scalar {
        let dt: Scalar = 0.05;
        let ev = EvaporationControls { cfl, ..base };
        let mut p = one_droplet(&gpu, &b, evaporating_controls(ev), 1e-4, 293.15, 1.0, dt);
        for _ in 0..20 {
            p.step(&gpu, &b.u, &b.rho, Some(&b.tg), Some(&b.yv), dt).unwrap();
        }
        p.snapshot(&gpu).unwrap().d[0]
    };
    let (whole, split) = (coarse(1.0), coarse(0.02));
    println!(
        "[76.10] a 100 um droplet, 20 steps of 50 ms: cfl 1.0 leaves {:.4} um, \
         cfl 0.02 leaves {:.4} um",
        1e6 * whole,
        1e6 * split
    );
    assert!(
        (whole - split).abs() > 1e-12 * 1e-4,
        "evaporation cfl did not move an evaporating parcel: {whole} against {split}"
    );
}

/// SPEC-LIT S13.4, in both directions, on the field the parcels read.
#[test]
fn the_vapour_field_is_required_when_it_is_read_and_refused_when_it_is_not() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let b = droplet_box(&gpu, 300.0, 0.3, 0.0);
    let dt: Scalar = 1e-3;

    let mut ev = one_droplet(
        &gpu,
        &b,
        evaporating_controls(EvaporationControls::default()),
        3e-4,
        293.15,
        1.0,
        dt,
    );
    let e = ev.step(&gpu, &b.u, &b.rho, Some(&b.tg), None, dt).unwrap_err().to_string();
    assert!(e.contains("vapour mass fraction"), "{e}");
    assert!(e.contains("S76.2"), "{e}");
    let e = ev.step(&gpu, &b.u, &b.rho, None, Some(&b.yv), dt).unwrap_err().to_string();
    assert!(e.contains("no gas temperature"), "{e}");

    let ctrl = ParcelControls {
        physics: ParcelPhysics::Heating,
        ..evaporating_controls(EvaporationControls::default())
    };
    let mut h = one_droplet(&gpu, &b, ctrl, 3e-4, 293.15, 1.0, dt);
    let e = h.step(&gpu, &b.u, &b.rho, Some(&b.tg), Some(&b.yv), dt).unwrap_err().to_string();
    assert!(e.contains("not \"evaporating\""), "{e}");
    assert!(e.contains("S13.4"), "{e}");
    // ... and the supported combination runs.
    h.step(&gpu, &b.u, &b.rho, Some(&b.tg), None, dt).unwrap();
}

/// SPEC-LIT S76.14, as S77.7 leaves it: coupling the energy of an evaporating
/// pool with the MASS coupling off is still refused by name - but the reason
/// has changed and so has the fix.
///
/// S76 said the refusal was because "the latent half has nowhere to go". S77
/// found that half a step off: the latent heat is not a second transfer the
/// gas owes at all, because (76.10)'s budget already puts it inside the
/// convective heat S68 deposits. What was really missing is the vapour and
/// the sensible enthalpy it carries, and the refusal now says which setting
/// supplies them.
#[test]
fn energy_coupling_is_refused_for_an_evaporating_pool() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let b = droplet_box(&gpu, 300.0, 0.3, 0.0);
    let p = one_droplet(
        &gpu,
        &b,
        evaporating_controls(EvaporationControls::default()),
        3e-4,
        293.15,
        1.0,
        1e-3,
    );
    let e = match crate::parcels::couple::ParcelCoupling::new(
        &gpu,
        &p,
        crate::parcels::couple::CouplingControls {
            momentum: crate::parcels::couple::CouplingMode::Explicit,
            energy: crate::parcels::couple::CouplingMode::Explicit,
            mass: crate::parcels::couple::MassCoupling::None,
        },
    ) {
        Ok(_) => panic!("energy coupling was accepted for an evaporating pool"),
        Err(e) => e.to_string(),
    };
    assert!(e.contains("mass evaporation"), "{e}");
    assert!(e.contains("S77"), "{e}");
    // The old message's diagnosis is gone with the old reason: the gate is
    // no longer "the latent half has nowhere to go".
    assert!(!e.contains("S76.14"), "{e}");

    // ... and with the mass coupling ON the same pool is accepted, which is
    // the whole of S77.
    crate::parcels::couple::ParcelCoupling::new(
        &gpu,
        &p,
        crate::parcels::couple::CouplingControls {
            momentum: crate::parcels::couple::CouplingMode::Explicit,
            energy: crate::parcels::couple::CouplingMode::Explicit,
            mass: crate::parcels::couple::MassCoupling::Evaporation,
        },
    )
    .map(|_| ())
    .unwrap_or_else(|e| panic!("the S77 combination was refused: {e}"));

    // Momentum coupling IS allowed and remains what S68 gated: the drag
    // impulse is exact for the update that was applied whatever the diameter
    // did on the way.
    crate::parcels::couple::ParcelCoupling::new(
        &gpu,
        &p,
        crate::parcels::couple::CouplingControls {
            momentum: crate::parcels::couple::CouplingMode::Explicit,
            energy: crate::parcels::couple::CouplingMode::Off,
            mass: crate::parcels::couple::MassCoupling::None,
        },
    )
    .map(|_| ())
    .unwrap_or_else(|e| panic!("momentum coupling was refused: {e}"));
}

/// The device codes and the host enums are one mapping. `evaporation::tests`
/// pins the host half; this pins the device half by selecting each value and
/// showing the kernel behaved as that value and not as its neighbour.
#[test]
fn the_device_evaporation_enumerations_match_the_host() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let b = droplet_box(&gpu, 330.0, 0.10, 0.0);
    let dt: Scalar = 1e-4;
    let d0: Scalar = 4e-4;
    let t0: Scalar = 300.0;
    for sat in [SaturationCurve::ClausiusClapeyron, SaturationCurve::HylandWexler] {
        for transfer in [
            MassTransfer::RanzMarshall,
            MassTransfer::Spalding,
            MassTransfer::AbramzonSirignano,
        ] {
            let ev = EvaporationControls { saturation: sat, transfer, ..Default::default() };
            let ctrl = ParcelControls { max_substeps: 1, ..evaporating_controls(ev) };
            let mut p = one_droplet(&gpu, &b, ctrl, d0, t0, 1.0, dt);
            let t_boil = p.boiling_temperature();
            p.step(&gpu, &b.u, &b.rho, Some(&b.tg), Some(&b.yv), dt).unwrap();
            let s = p.snapshot(&gpu).unwrap();
            let r = droplet_rate(&ev, t_boil, d0, t0, &b.gas);
            // One sub-step at the fixed point of neither: the mass removed is
            // dominated by the modelled rate, so it identifies the model.
            let mp = 1000.0 * std::f64::consts::FRAC_PI_6 * d0 * d0 * d0;
            let cap = mp * ctrl.c_liquid;
            let lam = (r.conductance + r.d_cooling_d_t) / cap;
            let w_t = -(-lam * dt).exp_m1();
            let teq = t0
                + (r.conductance * (b.gas.t - t0) + r.mdot * r.h_v)
                    / (r.conductance + r.d_cooling_d_t);
            let qc = r.conductance * ((b.gas.t - teq) * dt + (teq - t0) * w_t / lam);
            let want = (qc - cap * w_t * (teq - t0)) / r.h_v;
            assert!(
                (s.mass_lost[0] - want).abs() < 1e-9 * want.abs(),
                "{}/{}: device removed {}, host says {want}",
                sat.name(),
                transfer.name(),
                s.mass_lost[0]
            );
        }
    }
}


// ======================================================================
//  SPEC-LIT S78 - the droplet-wall impact regime map
// ======================================================================

/// A cube of `wall` on all six sides, four cells a side. Every parcel below
/// is fired at the `+z` face of it.
fn impact_box() -> HostMesh {
    block([4, 4, 4], [1.0, 1.0, 1.0], ["wall"; 6])
}

/// A parcel that feels nothing but its own inertia: no drag, no gravity, no
/// heating. That is what makes S78's gates ANALYTIC - the impact velocity is
/// the seeded velocity, exactly, so the impact Weber number is a closed form
/// of the seed and not an output of the integrator.
fn impact_controls(wall: WallAction, im: WallImpactControls, capacity: usize) -> ParcelControls {
    ParcelControls {
        capacity,
        drag: DragModel::None,
        physics: ParcelPhysics::Inert,
        wall,
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
        evaporation: EvaporationControls::default(),
        impact: im,
        persistent_blocks: None,
    }
}

/// Fire `speeds.len()` parcels straight at the `+z` wall, one per column of
/// the mesh, and return the pool after they have all reached it.
fn fire_at_the_wall(
    gpu: &Gpu,
    hm: &HostMesh,
    gm: &GpuMesh,
    ctrl: ParcelControls,
    d: Scalar,
    t: Scalar,
    speeds: &[Scalar],
) -> (ParcelSnapshot, ParcelStats) {
    let dt: Scalar = 0.05;
    let seeds: Vec<SeedParcel> = speeds
        .iter()
        .enumerate()
        .map(|(i, &u)| SeedParcel {
            // Spread over the 4x4 columns so that no two share a cell - not
            // because they would interact (they cannot; S66 is one-way and
            // collisionless) but so that a failure names one parcel.
            position: Vec3::new(
                0.125 + 0.25 * ((i % 4) as Scalar),
                0.125 + 0.25 * (((i / 4) % 4) as Scalar),
                // Close to the +z wall, so that the SLOWEST speed a gate
                // sweeps still reaches it inside the run. A parcel that
                // never landed would report "did not deposit" and pass a
                // deposit test for the wrong reason.
                0.9,
            ),
            velocity: Vec3::new(0.0, 0.0, u),
            diameter: d,
            temperature: t,
            n_p: 1.0,
            uid: None,
        })
        .collect();
    let mut p = Parcels::new(gpu, hm, gm, ctrl, &[], dt).unwrap();
    p.seed(gpu, hm, &seeds).unwrap();
    let (ug, rho) = still_gas(gpu, gm, 1.2).unwrap();
    for _ in 0..40 {
        p.step(gpu, &ug, &rho, None, None, dt).unwrap();
    }
    (p.snapshot(gpu).unwrap(), p.stats(gpu).unwrap())
}

/// **Gate 78-A, the device half.** The regime the KERNEL takes is the regime
/// [`WallImpactControls::regime`] says, at every impact speed - checked where
/// it is hardest, one part in a million either side of the two boundaries
/// that separate a rebound from a deposit.
///
/// The outcome is read from the parcel's own state and not from a counter, so
/// each of the 4x2x2 impacts is identified individually: a deposited parcel
/// is dead, flagged and stopped; a rebounding one is alive and moving.
#[test]
fn gate_78a_the_device_regime_boundaries_are_the_published_ones() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = impact_box();
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (rho_l, d, t) = (1000.0 as Scalar, 1e-4 as Scalar, 293.15 as Scalar);

    for splash in [SplashCriterion::Mundo, SplashCriterion::BaiGosman] {
        for tension in [SurfaceTension::Constant, SurfaceTension::IapwsR176] {
            let im = WallImpactControls { splash, tension, ..Default::default() };
            // The two boundaries a parcel's own state can resolve: below
            // `weStick` it sticks, above it bounces; below `weSpread` it
            // bounces, above it spreads.
            let mut speeds = Vec::new();
            for to in [WallRegime::Rebound, WallRegime::Spread] {
                let u = im.boundary_speed(rho_l, d, t, to).unwrap();
                speeds.push(u * (1.0 - 1e-6));
                speeds.push(u * (1.0 + 1e-6));
            }
            let ctrl = impact_controls(WallAction::Weber, im, speeds.len());
            let (s, st) = fire_at_the_wall(&gpu, &hm, &gm, ctrl, d, t, &speeds);

            for (i, &u) in speeds.iter().enumerate() {
                let want = im.classify(rho_l, d, t, u).1;
                let deposited = s.flags[i] & FLAG_DEPOSITED != 0;
                assert_eq!(
                    deposited,
                    want.deposits(),
                    "{splash:?}/{tension:?} at {u} m/s: the host says {}, the device {}",
                    want.name(),
                    if deposited { "deposited" } else { "did not" }
                );
                if deposited {
                    assert!(s.cell[i] < 0, "a deposited parcel is out of the working set");
                    assert_eq!(s.u[i].z.to_bits(), (0.0 as Scalar).to_bits());
                } else {
                    assert!(s.cell[i] >= 0, "a rebounding parcel stays in it");
                }
            }
            // Two of the four stuck or spread and two bounced, so the
            // histogram is not vacuously right.
            assert!(st.n_stick + st.n_spread + st.n_splash > 0);
            assert!(st.n_rebound > 0);
            st.check_wall_histogram(WallAction::Weber).unwrap();
        }
    }
}

/// **Gate 78-A, the splash boundary.** Spread and splash are both deposits,
/// so the parcel's own state cannot tell them apart and the counter has to.
/// One parcel per run, so the histogram names exactly one impact.
#[test]
fn gate_78a_the_splash_threshold_is_where_the_criterion_puts_it() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = impact_box();
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (rho_l, d, t) = (1000.0 as Scalar, 1e-4 as Scalar, 293.15 as Scalar);

    for splash in [SplashCriterion::Mundo, SplashCriterion::BaiGosman] {
        let im = WallImpactControls { splash, ..Default::default() };
        let u0 = im.boundary_speed(rho_l, d, t, WallRegime::Splash).unwrap();
        for (f, want) in [(1.0 - 1e-6, WallRegime::Spread), (1.0 + 1e-6, WallRegime::Splash)] {
            let u = u0 * f;
            assert_eq!(im.classify(rho_l, d, t, u).1, want, "the host's own map");
            let ctrl = impact_controls(WallAction::Weber, im, 1);
            let (_, st) = fire_at_the_wall(&gpu, &hm, &gm, ctrl, d, t, &[u]);
            let got = [st.n_stick, st.n_rebound, st.n_spread, st.n_splash];
            let hit: Vec<usize> = (0..4).filter(|&j| got[j] > 0).collect();
            assert_eq!(
                hit,
                vec![want.code() as usize],
                "{splash:?} at {u} m/s ({f} of the threshold): the histogram is {got:?} and \
                 the map says {}",
                want.name()
            );
        }
        // ... and the threshold is a real, finite speed a spray reaches.
        assert!(u0 > 1.0 && u0 < 100.0, "{splash:?}: splash at {u0} m/s");
    }
}

/// **Gate 78-A, the sweep.** The two maps agree on every one of sixteen
/// impact speeds spanning all four regimes, not only at the boundaries. A
/// boundary test alone would pass a device map that was right at four points
/// and wrong between them.
#[test]
fn gate_78a_the_device_and_host_maps_agree_over_the_whole_sweep() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = impact_box();
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (rho_l, d, t) = (1000.0 as Scalar, 1e-4 as Scalar, 293.15 as Scalar);
    let im = WallImpactControls::default();
    // Geometric, from well inside `stick` to well inside `splash`.
    let speeds: Vec<Scalar> = (0..16)
        .map(|i| 0.4 * (10.0 as Scalar).powf((i as Scalar) / 10.0))
        .collect();
    let ctrl = impact_controls(WallAction::Weber, im, speeds.len());
    let (s, st) = fire_at_the_wall(&gpu, &hm, &gm, ctrl, d, t, &speeds);

    let mut seen = [0usize; 4];
    for (i, &u) in speeds.iter().enumerate() {
        let want = im.classify(rho_l, d, t, u).1;
        seen[want.code() as usize] += 1;
        assert_eq!(
            s.flags[i] & FLAG_DEPOSITED != 0,
            want.deposits(),
            "at {u} m/s the host says {}",
            want.name()
        );
    }
    // All four regimes were actually visited, so the sweep is a sweep.
    for (r, n) in seen.iter().enumerate() {
        assert!(*n > 0, "{} was never reached by the sweep", WallRegime::NAMES[r]);
    }
    // The three depositing counters are exact: a deposit ends the parcel, so
    // each contributes exactly one impact.
    assert_eq!(st.n_stick as usize, seen[WallRegime::Stick.code() as usize]);
    assert_eq!(st.n_spread as usize, seen[WallRegime::Spread.code() as usize]);
    assert_eq!(st.n_splash as usize, seen[WallRegime::Splash.code() as usize]);
    // `n_rebound` is not, and the reason is a property worth stating: a
    // rebounding parcel is RE-CLASSIFIED at every impact, and at `e = 1` the
    // normal speed is preserved, so its regime is invariant and it bounces
    // between the two walls for the rest of the run. The counter therefore
    // counts impacts and not parcels, which is what a regime histogram
    // should count.
    assert!(
        st.n_rebound as usize >= seen[WallRegime::Rebound.code() as usize],
        "{} rebound impacts from {} rebounding parcels",
        st.n_rebound,
        seen[WallRegime::Rebound.code() as usize]
    );
    assert_eq!(
        st.n_rebound > 0,
        seen[WallRegime::Rebound.code() as usize] > 0
    );
    st.check_wall_histogram(WallAction::Weber).unwrap();
}

/// **Gate 78-B.** Nothing vanishes at a wall, and the exact form of that
/// claim is per parcel: the mass sitting on the wall is BITWISE the mass that
/// arrived, because `d` and `n_p` are not written by the deposit and the pool
/// reclaims no slot.
///
/// The aggregate follows and is measured too - it closes to round-off rather
/// than exactly, because the three sub-sums re-associate the one sum over the
/// pool, and saying which of the two statements is exact is the point.
#[test]
fn gate_78b_every_gram_that_hits_a_wall_is_still_accounted_for() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([10, 10, 10], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let inj = Injector { mass_flow: 1e-3, ..base_injector() };

    for wall in [WallAction::Stick, WallAction::Weber] {
        let ctrl = ParcelControls { wall, ..base_controls() };
        let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[inj], BASE_DT).unwrap();
        let (ug, rho) = still_gas(&gpu, &gm, 1.2).unwrap();
        for _ in 0..BASE_STEPS {
            p.step(&gpu, &ug, &rho, None, None, BASE_DT).unwrap();
        }
        let s = p.snapshot(&gpu).unwrap();
        let st = p.stats(&gpu).unwrap();
        st.check_capacity().unwrap();
        st.check_wall_histogram(wall).unwrap();

        // The partition, first: every slot is in exactly one of the three
        // buckets. This is what "nothing vanishes" MEANS.
        let live = s.live();
        let dep = s.deposited();
        let gone = s.gone();
        assert_eq!(
            live.len() + dep.len() + gone.len(),
            s.n_slots,
            "{}: the three buckets do not cover the pool",
            wall.name()
        );
        let mut all: Vec<usize> = live.iter().chain(&dep).chain(&gone).copied().collect();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), s.n_slots, "{}: the buckets overlap", wall.name());
        assert!(!dep.is_empty(), "{}: nothing reached a wall", wall.name());
        assert!(!live.is_empty(), "{}: nothing was still flying", wall.name());

        // The exact claim: each deposited droplet is bitwise the droplet the
        // injector emitted. An inert parcel's `d` and `n_p` are written once,
        // at injection, and the deposit does not touch them.
        let m_droplet = ctrl.rho_liquid
            * std::f64::consts::FRAC_PI_6 as Scalar
            * inj.diameter
            * inj.diameter
            * inj.diameter;
        let want_np = inj.mass_flow * BASE_DT / ((inj.parcels_per_event as Scalar) * m_droplet);
        for &i in &dep {
            assert_eq!(
                s.d[i].to_bits(),
                inj.diameter.to_bits(),
                "{}: slot {i} lost diameter at the wall",
                wall.name()
            );
            assert_eq!(
                s.n_p[i].to_bits(),
                want_np.to_bits(),
                "{}: slot {i} lost weight at the wall",
                wall.name()
            );
            assert_eq!(s.u[i].x.to_bits(), (0.0 as Scalar).to_bits());
            assert_eq!(s.u[i].y.to_bits(), (0.0 as Scalar).to_bits());
            assert_eq!(s.u[i].z.to_bits(), (0.0 as Scalar).to_bits());
        }

        // The aggregate: airborne + on the wall + gone = the pool.
        let rl = ctrl.rho_liquid;
        let sum = s.liquid_mass(rl) + s.deposited_mass(rl) + s.escaped_mass(rl);
        let pool = s.pool_mass(rl);
        assert!(
            (sum - pool).abs() <= 8.0 * Scalar::EPSILON * pool,
            "{}: {sum} against {pool}, a relative gap of {}",
            wall.name(),
            (sum - pool).abs() / pool
        );
        assert!(s.deposited_mass(rl) > 0.0);

        // ... and the pool is the mass the injector says it emitted, which is
        // what stops the ledger from balancing around a leak upstream of it.
        let injected = (st.n_injected as Scalar) * want_np * m_droplet;
        assert!(
            (pool - injected).abs() <= 1e-12 * injected,
            "{}: the pool holds {pool} and the injector emitted {injected}",
            wall.name()
        );
    }
}

/// **Gate 78-B, the other half.** `stick` is `remove` plus exactly two
/// statements - the velocity is zeroed and the flag is set - and nothing else
/// about the run moves. So the mass the ledger now accounts for is precisely
/// the mass `remove` was throwing away without saying so.
#[test]
fn gate_78b_stick_is_remove_plus_two_statements_and_nothing_else() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([10, 10, 10], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let inj = [Injector { mass_flow: 1e-3, ..base_injector() }];

    let r = run_case(&gpu, &hm, &gm, base_controls(), &inj, BASE_STEPS).unwrap();
    let ctrl = ParcelControls { wall: WallAction::Stick, ..base_controls() };
    let k = run_case(&gpu, &hm, &gm, ctrl, &inj, BASE_STEPS).unwrap();

    assert_eq!(r.n_slots, k.n_slots);
    assert!(vec_bits_eq(&r.x, &k.x), "the trajectory moved");
    assert!(bits_eq(&r.d, &k.d), "a diameter moved");
    assert!(bits_eq(&r.n_p, &k.n_p), "a weight moved");
    assert_eq!(r.cell, k.cell, "a parcel lived or died differently");
    assert_eq!(r.uid, k.uid);

    // The two statements, and only on the parcels that reached a wall.
    let dep = k.deposited();
    assert!(!dep.is_empty(), "nothing reached a wall, so this compares nothing");
    for i in 0..k.n_slots {
        let on_wall = dep.contains(&i);
        assert_eq!(
            k.flags[i],
            r.flags[i] | if on_wall { FLAG_DEPOSITED } else { 0 },
            "slot {i}: the flag word differs by more than the deposit bit"
        );
        if on_wall {
            assert_eq!(k.u[i].z.to_bits(), (0.0 as Scalar).to_bits());
        } else {
            assert_eq!(k.u[i].x.to_bits(), r.u[i].x.to_bits(), "slot {i}");
            assert_eq!(k.u[i].y.to_bits(), r.u[i].y.to_bits(), "slot {i}");
            assert_eq!(k.u[i].z.to_bits(), r.u[i].z.to_bits(), "slot {i}");
        }
    }
    // `remove` never set the bit, so its ledger is empty and its mass is
    // unaccounted for - which is the defect S78 exists to fix.
    assert_eq!(r.deposited().len(), 0);
    assert!(k.deposited_mass(1000.0) > 0.0);
}

/// **Gate 78-C.** A run that never touches a wall is BITWISE what it was
/// before S78 existed - every position, velocity, diameter, flag and
/// identity - whatever the impact map is set to.
///
/// This is the construction claim rather than a measurement of one: the map
/// is read inside `if (wallAction == OFP_WALL_WEBER)` at a `wall` face and
/// nowhere else, so a domain with no wall in it cannot reach a single
/// expression of it. The test is here because "cannot reach" is exactly the
/// kind of claim that stops being true when somebody hoists a computation.
#[test]
fn gate_78c_a_run_with_no_wall_impact_is_bitwise_unmoved_by_the_map() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    // Every patch open: the parcels leave rather than land, so no impact of
    // any kind happens and the map has nothing to classify.
    let hm = block([10, 10, 10], [1.0, 1.0, 1.0], ["patch"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let inj = [Injector { mass_flow: 1e-3, ..base_injector() }];

    let a = run_case(&gpu, &hm, &gm, base_controls(), &inj, BASE_STEPS).unwrap();
    assert!(a.n_slots > 50, "the case must actually inject");
    assert!(a.deposited().is_empty(), "something deposited; there is no wall to deposit on");

    for im in [
        WallImpactControls::default(),
        WallImpactControls {
            splash: SplashCriterion::BaiGosman,
            tension: SurfaceTension::IapwsR176,
            sigma: 0.03,
            mu_liquid: 5e-3,
            we_stick: 7.0,
            we_spread: 33.0,
            k_crit: 12.0,
            splash_a: 900.0,
        },
    ] {
        for wall in [WallAction::Remove, WallAction::Stick, WallAction::Weber] {
            let ctrl = ParcelControls { wall, impact: im, ..base_controls() };
            let b = run_case(&gpu, &hm, &gm, ctrl, &inj, BASE_STEPS).unwrap();
            assert!(
                snapshot_bits_eq(&a, &b),
                "{}: the impact model moved a run that never met a wall",
                wall.name()
            );
        }
    }
}

/// **Gate 78-C, the default path.** A run under `remove` or `rebound` that
/// DOES meet walls is bitwise identical whatever the impact model is set to -
/// including a deliberately absurd one - because neither action reads it.
///
/// This is the claim that matters for "the defaults do not move": the
/// no-wall test above shows the map cannot reach a run with nothing to
/// classify, and this one shows it does not reach a run with plenty to
/// classify either, so long as the case did not ask for it. It is a
/// measurement of a construction - every expression of the map is inside
/// `if (wallAction == OFP_WALL_WEBER)` - and the construction is what makes
/// it true; the measurement is what would catch a hoist.
#[test]
fn gate_78c_the_default_wall_actions_do_not_read_the_impact_model() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([10, 10, 10], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let inj = [Injector { mass_flow: 1e-3, ..base_injector() }];

    let absurd = WallImpactControls {
        splash: SplashCriterion::BaiGosman,
        tension: SurfaceTension::IapwsR176,
        sigma: 1e-4,
        mu_liquid: 5e-3,
        we_stick: 1e6,
        we_spread: 1e7,
        k_crit: 1e-3,
        splash_a: 1.0,
    };
    for wall in [WallAction::Remove, WallAction::Rebound] {
        let a = run_case(
            &gpu,
            &hm,
            &gm,
            ParcelControls { wall, ..base_controls() },
            &inj,
            BASE_STEPS,
        )
        .unwrap();
        let b = run_case(
            &gpu,
            &hm,
            &gm,
            ParcelControls { wall, impact: absurd, ..base_controls() },
            &inj,
            BASE_STEPS,
        )
        .unwrap();
        assert!(
            snapshot_bits_eq(&a, &b),
            "{}: the impact model moved a run that never asked for it",
            wall.name()
        );
        // Non-vacuity: parcels really did reach walls under `remove`, and
        // really did bounce off them under `rebound`.
        let s = if wall == WallAction::Remove {
            a.n_slots - a.live().len()
        } else {
            a.live().len()
        };
        assert!(s > 0, "{}: nothing met a wall, so this compares nothing", wall.name());
        assert!(a.deposited().is_empty(), "{}: nothing may be flagged", wall.name());
    }
}

/// **Gate 78-C, the empty pool.** With no parcels at all, every counter this
/// section added is exactly zero and every ledger term is `+0.0` - not
/// `-0.0`, and not a NaN from a `0/0` Weber number nobody formed.
#[test]
fn gate_78c_an_empty_pool_deposits_exactly_nothing() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = impact_box();
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let ctrl = impact_controls(WallAction::Weber, WallImpactControls::default(), 8);
    let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[], 0.05).unwrap();
    let (ug, rho) = still_gas(&gpu, &gm, 1.2).unwrap();
    for _ in 0..4 {
        p.step(&gpu, &ug, &rho, None, None, 0.05).unwrap();
    }
    let s = p.snapshot(&gpu).unwrap();
    let st = p.stats(&gpu).unwrap();
    assert_eq!((st.n_stick, st.n_rebound, st.n_spread, st.n_splash), (0, 0, 0, 0));
    assert_eq!(st.n_wall, 0);
    st.check_wall_histogram(WallAction::Weber).unwrap();
    assert!(s.deposited().is_empty());
    for m in [s.deposited_mass(1000.0), s.escaped_mass(1000.0), s.pool_mass(1000.0)] {
        assert_eq!(m.to_bits(), (0.0 as Scalar).to_bits(), "an empty sum that is not +0.0");
    }
}

/// S13.4.1: every setting S78 added owes a pair, and here they are. Each one
/// is turned against the SAME four impacts - one in each regime under the
/// defaults - so a knob that never reaches the kernel is named.
///
/// `surfaceTension` and `splashA` need their own bases, for the reason S66's
/// own knob table records: a pair has to be posed where the setting bites.
/// The IAPWS curve at 293.15 K is within 0.01 % of the default `sigma`, so it
/// is turned against a `sigma` that is visibly not water's; and `splashA` is
/// read only by the criterion that has an `A`.
#[test]
fn every_impact_setting_changes_what_the_run_writes() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = impact_box();
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (d, t) = (1e-4 as Scalar, 293.15 as Scalar);
    // One impact in each of the four regimes, plus two placed where a
    // single threshold bites: 1.5 m/s is `We = 3.0`, between the default
    // `weStick` of 2 and the 3.5 the pair moves it to; 15 m/s is `We = 309`,
    // above Mundo's splash threshold (about 111) and below Bai & Gosman's
    // (about 531), which is the only place the two criteria disagree.
    let speeds: [Scalar; 5] = [0.8, 1.5, 2.5, 6.0, 15.0];

    let base = WallImpactControls::default();
    let bai = WallImpactControls { splash: SplashCriterion::BaiGosman, ..base };
    let cases: &[(&str, WallImpactControls, WallImpactControls)] = &[
        ("splashCriterion", base, bai),
        (
            "surfaceTension",
            WallImpactControls { sigma: 0.03, ..base },
            WallImpactControls { sigma: 0.03, tension: SurfaceTension::IapwsR176, ..base },
        ),
        ("sigma", base, WallImpactControls { sigma: 0.02, ..base }),
        ("muLiquid", base, WallImpactControls { mu_liquid: 1e-4, ..base }),
        ("weStick", base, WallImpactControls { we_stick: 3.5, ..base }),
        ("weSpread", base, WallImpactControls { we_spread: 60.0, ..base }),
        ("kCrit", base, WallImpactControls { k_crit: 20.0, ..base }),
        ("splashA", bai, WallImpactControls { splash_a: 200.0, ..bai }),
    ];

    for (name, a, b) in cases {
        assert_ne!(a, b, "{name}: the knob turned nothing");
        let run = |im: WallImpactControls| {
            let ctrl = impact_controls(WallAction::Weber, im, speeds.len());
            fire_at_the_wall(&gpu, &hm, &gm, ctrl, d, t, &speeds)
        };
        let (sa, ta) = run(*a);
        let (sb, tb) = run(*b);
        let hist = |t: &ParcelStats| (t.n_stick, t.n_rebound, t.n_spread, t.n_splash);
        assert!(
            hist(&ta) != hist(&tb) || !vec_bits_eq(&sa.u, &sb.u) || sa.flags != sb.flags,
            "{name} is INERT: two runs differing only in it classified all four impacts \
             identically, {:?} both times (SPEC-LIT S13.4.1)",
            hist(&ta)
        );
    }
}

/// The four values of `wallInteraction` are four different runs, and the two
/// S78 added are refused-no-longer. `spread`, `film` and `splash` are still
/// refused, each with a note naming what is missing and what to ask for
/// instead - S13.4's contract, and the reason the refusal list is a list
/// rather than a shrug.
#[test]
fn the_wall_interaction_contract_names_what_it_gained_and_what_it_still_refuses() {
    for s in WallAction::NAMES {
        assert_eq!(WallAction::from_name(s).unwrap().name(), *s);
    }
    assert_eq!(WallAction::from_name("deposit").unwrap(), WallAction::Stick);
    assert_eq!(WallAction::from_name("baiGosman").unwrap(), WallAction::Weber);
    assert_eq!(WallAction::from_name("regime").unwrap(), WallAction::Weber);

    for (bad, must) in [
        ("spread", "film transport"),
        ("film", "no film transport"),
        ("splash", "population growth"),
    ] {
        let m = format!("{}", WallAction::from_name(bad).unwrap_err());
        assert!(m.contains(must), "{bad}: {m}");
        assert!(m.contains("S78"), "{bad} does not carry a section number: {m}");
        assert!(m.contains("weber"), "{bad} does not print the menu: {m}");
    }
    // The three that used to be refused together are no longer one refusal:
    // `stick` is supported and the other two are refused for DIFFERENT
    // reasons, which is the whole content of S78.11.
    let spread = format!("{}", WallAction::from_name("spread").unwrap_err());
    let film = format!("{}", WallAction::from_name("film").unwrap_err());
    assert_ne!(spread, film);
    assert!(spread.contains("stick"), "{spread}");

    // S13.4.2: the banner prints the impact model when it is in force and
    // not when it is not.
    let c = base_controls();
    assert!(!ParcelControls { wall: WallAction::Remove, ..c }.describe().contains("S78"));
    assert!(!ParcelControls { wall: WallAction::Stick, ..c }.describe().contains("S78"));
    let w = ParcelControls { wall: WallAction::Weber, ..c }.describe();
    assert!(w.contains("parcels/impact:"), "{w}");
    assert!(w.contains("e=1 ft=0"), "{w}");
}

/// The device's regime codes are the host's, and they are also the offsets
/// into the counter array. Both facts are load-bearing and neither is visible
/// from either side alone, so a run with one impact of each kind pins them.
#[test]
fn the_device_regime_codes_are_the_host_regime_codes() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = impact_box();
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let im = WallImpactControls::default();
    let (rho_l, d, t) = (1000.0 as Scalar, 1e-4 as Scalar, 293.15 as Scalar);
    let ub = |r: WallRegime| im.boundary_speed(rho_l, d, t, r).unwrap();

    for want in [
        WallRegime::Stick,
        WallRegime::Rebound,
        WallRegime::Spread,
        WallRegime::Splash,
    ] {
        // A speed squarely inside the band: the geometric mean of its two
        // boundaries, or a factor either side of the one boundary it has.
        let u = match want {
            WallRegime::Stick => ub(WallRegime::Rebound) * 0.5,
            WallRegime::Rebound => (ub(WallRegime::Rebound) * ub(WallRegime::Spread)).sqrt(),
            WallRegime::Spread => (ub(WallRegime::Spread) * ub(WallRegime::Splash)).sqrt(),
            WallRegime::Splash => ub(WallRegime::Splash) * 2.0,
        };
        assert_eq!(im.classify(rho_l, d, t, u).1, want, "the host's own map at {u} m/s");
        let ctrl = impact_controls(WallAction::Weber, im, 1);
        let (_, st) = fire_at_the_wall(&gpu, &hm, &gm, ctrl, d, t, &[u]);
        // Exactly one slot of the histogram moved, and it is the one the
        // host's code indexes. A rebounding parcel bounces for the rest of
        // the run, so the COUNT is not one - the point being pinned is which
        // slot, not how many.
        let got = [st.n_stick, st.n_rebound, st.n_spread, st.n_splash];
        let hit: Vec<usize> = (0..4).filter(|&j| got[j] > 0).collect();
        assert_eq!(
            hit,
            vec![want.code() as usize],
            "{} at {u} m/s gave the histogram {got:?}",
            want.name()
        );
    }
}

/// A droplet impacts at the size and temperature it ACTUALLY has, not the
/// ones it was injected at. The map reads `d` and `T_p` as locals inside the
/// sub-step loop, and a droplet that shrank on the way to the wall has a
/// smaller Weber number for it - which can move it a whole regime, and does
/// here.
#[test]
fn a_droplet_impacts_at_the_size_and_temperature_it_arrives_with() {
    let im = WallImpactControls::default();
    let (rho_l, t) = (1000.0 as Scalar, 293.15 as Scalar);
    let u: Scalar = 2.0;
    let big = im.classify(rho_l, 2e-4, t, u).1;
    let small = im.classify(rho_l, 1e-5, t, u).1;
    assert_ne!(big, small, "the classification has to depend on the diameter at all");
    assert_eq!(big, WallRegime::Rebound);
    assert_eq!(small, WallRegime::Stick);
    // ... and on the temperature, through the IAPWS surface tension: a hot
    // droplet has less of it, so the same impact is a higher Weber number.
    let hot = WallImpactControls { tension: SurfaceTension::IapwsR176, ..im };
    let cold_we = hot.classify(rho_l, 1e-4, 293.15, u).0.we;
    let hot_we = hot.classify(rho_l, 1e-4, 360.0, u).0.we;
    assert!(hot_we > cold_we * 1.05, "{cold_we} against {hot_we}");
}
