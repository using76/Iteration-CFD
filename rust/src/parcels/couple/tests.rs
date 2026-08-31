// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! SPEC-LIT S68.11's table: the two conservation gates (what the parcels
//! lost is what the gas is given, in momentum and in energy), the
//! bitwise-unchanged gate (no parcels, no change), the sign contract the
//! implicit halves satisfy by construction, the S13.4.1 pairs and the
//! refusals.
//!
//! Written from the same sources as `src/parcels/couple.rs`; see that
//! module's header. No GPL-licensed source was consulted.

use super::*;
use crate::blockgen::{self, BlockSpec, GradedAxis};
use crate::device::Gpu;
use crate::energy::EnergySources;
use crate::field::GpuVectorField;
use crate::ldu::GpuLduMatrix;
use crate::mesh::{GpuMesh, HostMesh};
use crate::momentum::{BuoyancyCoeffs, Momentum, MomentumControls, MomentumSources};
use crate::parcels::{
    DragModel, Injector, ParcelControls, ParcelDeposition, ParcelSnapshot, SeedParcel, WallAction,
};

// ----------------------------------------------------------------------
//  Fixtures
// ----------------------------------------------------------------------

fn block(n: [usize; 3], hi: [Scalar; 3], types: [&str; 6]) -> HostMesh {
    let axis = |i: usize| GradedAxis {
        lo: 0.0,
        hi: hi[i],
        n: n[i],
        expansion: 1.0,
        two_sided: false,
    };
    blockgen::build_mesh(&BlockSpec {
        x: axis(0),
        y: axis(1),
        z: axis(2),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: types.map(String::from),
        cyclic: Vec::new(),
    })
    .expect("block mesh")
}

fn base_controls() -> ParcelControls {
    ParcelControls {
        capacity: 512,
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

fn momentum_only() -> CouplingControls {
    CouplingControls {
        momentum: CouplingMode::Explicit,
        energy: CouplingMode::Off,
        mass: MassCoupling::None,
    }
}

/// SplitMix64's finaliser, as everywhere else in this crate: a deterministic
/// scrambler, never a source of randomness. It scatters the fixtures off the
/// lattice a broken gather could still get right.
fn mix(i: u64) -> u64 {
    let mut z = i.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z ^= z >> 30;
    z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    z
}

fn unit(i: u64) -> Scalar {
    (mix(i) >> 11) as Scalar / (1u64 << 53) as Scalar
}

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

fn mag(v: Vec3) -> Scalar {
    (v.x * v.x + v.y * v.y + v.z * v.z).sqrt()
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

fn add(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x + b.x, a.y + b.y, a.z + b.z)
}

/// A gas at rest but for a uniform sweep, so that a parcel injected at rest
/// exchanges real momentum in a known direction.
fn gas(gpu: &Gpu, m: &GpuMesh, u: Vec3, rho: Scalar) -> (GpuVectorField, DevBuf<Scalar>) {
    let mut f = GpuVectorField::zeros(gpu, m, "U").unwrap();
    f.f = gpu.upload(&vec![u; m.n_cells]).unwrap();
    let r = gpu.upload(&vec![rho; m.n_cells]).unwrap();
    (f, r)
}

fn seed(position: Vec3, velocity: Vec3, d: Scalar, n_p: Scalar, uid: u64) -> SeedParcel {
    SeedParcel {
        position,
        velocity,
        diameter: d,
        temperature: 293.15,
        n_p,
        uid: Some(uid),
    }
}

/// One step of pool, sort and coupling, in the order S68.3 requires.
#[allow(clippy::too_many_arguments)]
fn one_step<'a>(
    gpu: &Gpu,
    p: &mut Parcels<'a>,
    dep: &mut ParcelDeposition<'a>,
    cp: &mut ParcelCoupling<'a>,
    u: &GpuVectorField,
    rho: &DevBuf<Scalar>,
    t: Option<&DevBuf<Scalar>>,
    dt: Scalar,
) {
    p.step(gpu, u, rho, t, dt).unwrap();
    dep.update(gpu, p).unwrap();
    cp.update(gpu, p, dep, rho, u, t, dt).unwrap();
}

// ======================================================================
//  The device/host mirror
// ======================================================================

/// The three coupling modes and the two physics codes are `#define`s in
/// `cuda/parcelcouple.cu` and `cuda/parcels.cu` and matched integers here.
/// Nothing in the type system connects them, so the connection is a test
/// that reads the kernel source - the same pinning `parcels::tests` does for
/// the drag and wall enumerations.
#[test]
fn the_device_modes_match_the_host() {
    let couple = include_str!("../../../cuda/parcelcouple.cu");
    for (name, want) in [
        ("OFC_MODE_OFF", CouplingMode::Off.code()),
        ("OFC_MODE_EXPLICIT", CouplingMode::Explicit.code()),
        ("OFC_MODE_SEMIIMPLICIT", CouplingMode::SemiImplicit.code()),
    ] {
        let line = couple
            .lines()
            .find(|l| l.starts_with(&format!("#define {name} ")))
            .unwrap_or_else(|| panic!("cuda/parcelcouple.cu does not define {name}"));
        let got: i32 = line.split_whitespace().nth(2).unwrap().parse().unwrap();
        assert_eq!(got, want, "{name} is {got} on the device and {want} here");
    }

    let parcels = include_str!("../../../cuda/parcels.cu");
    for (name, want) in [
        ("OFP_PHYS_INERT", ParcelPhysics::Inert.code()),
        ("OFP_PHYS_HEATING", ParcelPhysics::Heating.code()),
    ] {
        let line = parcels
            .lines()
            .find(|l| l.starts_with(&format!("#define {name} ")))
            .unwrap_or_else(|| panic!("cuda/parcels.cu does not define {name}"));
        let got: i32 = line.split_whitespace().nth(2).unwrap().parse().unwrap();
        assert_eq!(got, want, "{name} is {got} on the device and {want} here");
    }

    // And there is still no f64 atomic anywhere in the coupling.
    assert!(
        !couple.contains("atomicAdd"),
        "cuda/parcelcouple.cu has grown an atomic; S68.3's whole argument is that it \
         has none"
    );
}

// ======================================================================
//  (68.5): the impulse the integrator applied
// ======================================================================

/// The accumulator is the closed form of (68.5), not an approximation of
/// it: one parcel, one sub-step, frozen gas, compared against
/// [`drag_impulse`] evaluated on the host from the same `beta`.
#[test]
fn the_accumulated_impulse_is_the_closed_form() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([1, 1, 8], [1.0, 1.0, 8.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let rho_g: Scalar = 1.2;
    let (u, rho) = gas(&gpu, &gm, Vec3::ZERO, rho_g);

    let dt: Scalar = 1e-3;
    let d: Scalar = 5e-4;
    let u0 = Vec3::new(0.0, 0.0, -3.0);
    let ctrl = ParcelControls { capacity: 4, max_substeps: 1, ..base_controls() };
    let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[], dt).unwrap();
    p.seed(&gpu, &hm, &[seed(Vec3::new(0.5, 0.5, 4.5), u0, d, 1.0, 7)])
        .unwrap();
    p.step(&gpu, &u, &rho, None, dt).unwrap();
    let s = p.snapshot(&gpu).unwrap();

    // The host closed form, from (66.3)/(66.4) with the same numbers.
    let k = crate::parcels::drag_k(DragModel::SchillerNaumann, rho_g, ctrl.mu_gas, d, mag(u0));
    let inertia = (4.0 / 3.0) * ctrl.rho_liquid * d;
    let beta = dt * k / inertia;
    let m_eff = ctrl.rho_liquid * std::f64::consts::FRAC_PI_6 as Scalar * d * d * d;
    let rr = rho_g / ctrl.rho_liquid;
    let a_g = Vec3::new(0.0, 0.0, ctrl.gravity.z * (1.0 - rr));
    let want = drag_impulse(m_eff, beta, dt, sub(Vec3::ZERO, u0), a_g);

    let got = s.impulse[0];
    let err = mag(sub(got, want)) / mag(want);
    assert!(
        err < 1e-14,
        "the accumulated impulse is ({} {} {}), the closed form ({} {} {}); relative {err:e}",
        got.x,
        got.y,
        got.z,
        want.x,
        want.y,
        want.z
    );

    // ... and the exchange rate is m_eff (1 - e^-beta)/dt, positive.
    let want_a = m_eff * (-(-beta).exp_m1()) / dt;
    assert!(
        (s.exchange[0] - want_a).abs() <= 1e-14 * want_a,
        "exchange rate {} against {want_a}",
        s.exchange[0]
    );
    assert!(s.exchange[0] > 0.0);
}

/// At terminal velocity the drag impulse is exactly minus the weight the gas
/// is holding up, `-m a_g dt`, whatever the drag law says. This is the check
/// that the `-dt(1 - q) a_g` term - the one a re-linearised source would
/// drop - is there and is right.
#[test]
fn at_terminal_velocity_the_impulse_is_the_weight() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([1, 1, 8], [1.0, 1.0, 8.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let rho_g: Scalar = 1.2;
    let (u, rho) = gas(&gpu, &gm, Vec3::ZERO, rho_g);

    let dt: Scalar = 1e-3;
    let d: Scalar = 2e-4;
    let ctrl = base_controls();
    let u_t = crate::parcels::terminal_velocity(
        ctrl.drag,
        rho_g,
        ctrl.rho_liquid,
        ctrl.mu_gas,
        d,
        mag(ctrl.gravity),
    );

    let mut p = Parcels::new(
        &gpu,
        &hm,
        &gm,
        ParcelControls { capacity: 4, ..ctrl },
        &[],
        dt,
    )
    .unwrap();
    p.seed(
        &gpu,
        &hm,
        &[seed(
            Vec3::new(0.5, 0.5, 6.5),
            Vec3::new(0.0, 0.0, -u_t),
            d,
            1.0,
            3,
        )],
    )
    .unwrap();
    // A few steps, so any transient has relaxed.
    for _ in 0..8 {
        p.step(&gpu, &u, &rho, None, dt).unwrap();
    }
    let s = p.snapshot(&gpu).unwrap();

    let m_p = ctrl.rho_liquid * std::f64::consts::FRAC_PI_6 as Scalar * d * d * d;
    let rr = rho_g / ctrl.rho_liquid;
    // The drag holds the droplet UP against gravity, so the impulse is
    // positive where `g` is negative, and the gas is pushed down by exactly
    // as much - which is why a raining cloud drags air with it.
    let want = -m_p * ctrl.gravity.z * (1.0 - rr) * dt;
    let got = s.impulse[0].z;
    assert!(
        (got - want).abs() <= 1e-6 * want,
        "at terminal velocity the drag impulse is {got}, the weight impulse {want}"
    );
}

// ======================================================================
//  GATE 68-A: momentum conservation between the two phases
// ======================================================================

/// **Gate 68-A.** The impulse the gas is given is exactly minus the impulse
/// the parcels took, summed over the mesh and over the pool. Not to a
/// modelling tolerance: to round-off, because the deposited number IS the
/// accumulated one.
///
/// Run over a spread of cases that each break a different assumption a
/// re-linearised source would make: gravity on and off, one sub-step and
/// many, drag laws, added mass, parcels that cross cells, and weights that
/// span four decades.
#[test]
fn gate_68a_what_the_parcels_took_is_what_the_gas_is_given() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([6, 6, 6], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let rho_g: Scalar = 1.2;
    let (u, rho) = gas(&gpu, &gm, Vec3::new(2.0, -1.0, 0.5), rho_g);
    let dt: Scalar = 2e-3;

    let cases: [(&str, DragModel, bool, Vec3, u32); 5] = [
        ("Schiller-Naumann, gravity", DragModel::SchillerNaumann, false, Vec3::new(0.0, 0.0, -9.81), 64),
        ("Stokes, gravity", DragModel::Stokes, false, Vec3::new(0.0, 0.0, -9.81), 64),
        ("Schiller-Naumann, no gravity", DragModel::SchillerNaumann, false, Vec3::ZERO, 64),
        ("added mass", DragModel::SchillerNaumann, true, Vec3::new(0.0, 0.0, -9.81), 64),
        ("one sub-step", DragModel::SchillerNaumann, false, Vec3::new(0.0, 0.0, -9.81), 1),
    ];

    for (name, drag, added_mass, g, max_substeps) in cases {
        let ctrl = ParcelControls {
            capacity: 256,
            drag,
            added_mass,
            gravity: g,
            max_substeps,
            ..base_controls()
        };
        let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[], dt).unwrap();
        let seeds: Vec<SeedParcel> = (0..200u64)
            .map(|i| {
                seed(
                    Vec3::new(unit(i), unit(i + 1000), unit(i + 2000)),
                    Vec3::new(
                        6.0 * unit(i + 3000) - 3.0,
                        6.0 * unit(i + 4000) - 3.0,
                        6.0 * unit(i + 5000) - 3.0,
                    ),
                    1e-4 + 9e-4 * unit(i + 6000),
                    (10.0 as Scalar).powf(4.0 * unit(i + 7000)),
                    i + 1,
                )
            })
            .collect();
        p.seed(&gpu, &hm, &seeds).unwrap();
        let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
        let mut cp = ParcelCoupling::new(&gpu, &p, momentum_only()).unwrap();

        for step in 0..5 {
            one_step(&gpu, &mut p, &mut dep, &mut cp, &u, &rho, None, dt);
            let gained = cp.total_impulse(&gpu).unwrap();
            let lost = live_parcel_impulse(&p.snapshot(&gpu).unwrap());
            let defect = mag(add(gained, lost));
            let scale = mag(lost).max(mag(gained));
            assert!(
                scale > 0.0,
                "{name}: nothing was exchanged at all, so the gate is vacuous"
            );
            assert!(
                defect <= 1e-14 * scale,
                "{name}, step {step}: the gas gained ({:e} {:e} {:e}) and the parcels \
                 lost ({:e} {:e} {:e}); defect {defect:e} against {scale:e}",
                gained.x,
                gained.y,
                gained.z,
                -lost.x,
                -lost.y,
                -lost.z,
            );
        }
    }
}

/// The same claim on the path a case uses - an INJECTED spray, whose parcels
/// are born mid-step, cross cells and leave through a patch. Escapes are
/// what make this different from the seeded gate: a parcel that left is no
/// longer in the CSR, so the sum is over the live set on both sides, and the
/// difference between "all parcels" and "live parcels" is measured rather
/// than defined away.
#[test]
fn the_gate_holds_for_an_injected_spray_that_loses_parcels() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([8, 8, 8], [1.0, 1.0, 1.0], ["patch"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (u, rho) = gas(&gpu, &gm, Vec3::new(0.0, 0.0, 1.0), 1.2);
    let dt: Scalar = 1e-3;

    let inj = Injector {
        position: Vec3::new(0.5, 0.5, 0.12),
        axis: Vec3::new(0.0, 0.0, -1.0),
        cone_half_angle: 0.6,
        standoff: 0.0,
        speed: 12.0,
        diameter: 3e-4,
        temperature: 293.15,
        mass_flow: 1e-3,
        parcels_per_event: 16,
        interval: 0.0,
    };
    let ctrl = ParcelControls { capacity: 2048, ..base_controls() };
    let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[inj], dt).unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    let mut cp = ParcelCoupling::new(&gpu, &p, momentum_only()).unwrap();

    let mut escaped_any = false;
    for step in 0..60 {
        one_step(&gpu, &mut p, &mut dep, &mut cp, &u, &rho, None, dt);
        let gained = cp.total_impulse(&gpu).unwrap();
        let s = p.snapshot(&gpu).unwrap();
        let lost = live_parcel_impulse(&s);
        let defect = mag(add(gained, lost));
        let scale = mag(lost);
        if scale > 0.0 {
            assert!(
                defect <= 1e-13 * scale,
                "step {step}: defect {defect:e} against {scale:e}"
            );
        }
        escaped_any |= p.stats(&gpu).unwrap().n_escaped > 0;
    }
    assert!(
        escaped_any,
        "no parcel ever left the domain, so the live-set half of the claim went untested"
    );
    let st = p.stats(&gpu).unwrap();
    assert_eq!(st.n_lost, 0, "the walk lost a parcel on a Cartesian mesh");
}

// ======================================================================
//  GATE 68-B: no parcels, no change
// ======================================================================

/// **Gate 68-B, by construction.** With nothing registered the momentum
/// assembly launches no kernel over the registry, so the matrix is bit for
/// bit the matrix of a build without it. Asserted here on the assembled
/// matrix rather than argued: the two runs differ only in that one owns a
/// `MomentumSources` that was cleared and never written to.
#[test]
fn gate_68b_an_unregistered_registry_moves_not_one_bit() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([5, 4, 3], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();

    let assemble = |clear_and_register: bool| -> (Vec<Scalar>, Vec<Scalar>) {
        let mut mom = Momentum::new(&gpu, &gm, MomentumControls::default(), BuoyancyCoeffs::default()).unwrap();
        let mut u = GpuVectorField::zeros(&gpu, &gm, "U").unwrap();
        let cells: Vec<Vec3> = (0..gm.n_cells)
            .map(|i| Vec3::new(unit(i as u64), unit(i as u64 + 99), unit(i as u64 + 555)))
            .collect();
        u.f = gpu.upload(&cells).unwrap();
        let phi = crate::field::GpuSurfaceScalarField::zeros(&gpu, &gm, "phi").unwrap();
        let nut = crate::field::GpuScalarField::zeros(&gpu, &gm, "nut").unwrap();
        if clear_and_register {
            // A registry that exists, is cleared every iteration, and has
            // nothing pushed into it - which is what a run with the parcel
            // model present and no parcels injected does.
            mom.field_sources_mut().clear(&gpu).unwrap();
        }
        mom.assemble_only(&gpu, &u, &phi, &nut).unwrap();
        (
            gpu.download(&mom.matrix().diag).unwrap(),
            gpu.download(&mom.matrix().source).unwrap(),
        )
    };

    let (d0, s0) = assemble(false);
    let (d1, s1) = assemble(true);
    assert!(
        bits_eq(&d0, &d1) && bits_eq(&s0, &s1),
        "an empty registry moved the momentum matrix"
    );
}

/// **Gate 68-B, measured.** A pool that exists and has never held a parcel
/// deposits exactly `+0.0` into every one of the eight fields, and
/// registering those zeros leaves the assembled matrix bit for bit what it
/// was. The by-construction argument above covers the case where nothing
/// registers; this covers the case where something registers nothing.
#[test]
fn gate_68b_an_empty_pool_couples_exactly_zero() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([4, 4, 4], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (u, rho) = gas(&gpu, &gm, Vec3::new(1.0, 2.0, 3.0), 1.2);
    let dt: Scalar = 1e-3;

    let mut p = Parcels::new(&gpu, &hm, &gm, base_controls(), &[], dt).unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    let mut cp = ParcelCoupling::new(&gpu, &p, momentum_only()).unwrap();
    one_step(&gpu, &mut p, &mut dep, &mut cp, &u, &rho, None, dt);

    let s = cp.snapshot(&gpu).unwrap();
    for (name, v) in [
        ("beta", &s.exchange),
        ("q", &s.heat),
        ("alpha_T", &s.heat_exchange),
        ("momentum sp", &s.momentum_sp),
        ("energy q", &s.energy_q),
        ("energy sp", &s.energy_sp),
    ] {
        assert!(
            v.iter().all(|x| *x == 0.0),
            "{name} is not zero in every cell of an empty pool"
        );
    }
    for v in [&s.force, &s.momentum_su] {
        assert!(v.iter().all(|x| x.x == 0.0 && x.y == 0.0 && x.z == 0.0));
    }
    assert_eq!(cp.total_impulse(&gpu).unwrap().z, 0.0);

    // ... and it is a NEGATIVE zero, because the deposit is `-(sum)/(V dt)`
    // and negating `+0.0` gives `-0.0`. That is not a defect to be tidied
    // away: `-0.0` is the additive identity that leaves EVERY accumulator
    // bitwise unmoved, `x + (-0.0) == x` for every `x` including `-0.0`
    // itself, where `+0.0` would flip a stored negative zero to positive.
    // S68.11's finding 2.
    assert!(
        s.force.iter().all(|f| f.z.is_sign_negative()),
        "the empty deposit is +0.0, which is a WEAKER additive identity than \
         the -0.0 this arithmetic produces"
    );

    // And the matrix does not move when those zeros are registered.
    let assemble = |register: bool| -> (Vec<Scalar>, Vec<Scalar>) {
        let mut mom = Momentum::new(&gpu, &gm, MomentumControls::default(), BuoyancyCoeffs::default()).unwrap();
        let mut uu = GpuVectorField::zeros(&gpu, &gm, "U").unwrap();
        let cells: Vec<Vec3> = (0..gm.n_cells)
            .map(|i| Vec3::new(unit(i as u64), unit(i as u64 + 7), unit(i as u64 + 13)))
            .collect();
        uu.f = gpu.upload(&cells).unwrap();
        let phi = crate::field::GpuSurfaceScalarField::zeros(&gpu, &gm, "phi").unwrap();
        let nut = crate::field::GpuScalarField::zeros(&gpu, &gm, "nut").unwrap();
        if register {
            mom.field_sources_mut().clear(&gpu).unwrap();
            cp.register_momentum(&gpu, mom.field_sources_mut()).unwrap();
        }
        mom.assemble_only(&gpu, &uu, &phi, &nut).unwrap();
        (
            gpu.download(&mom.matrix().diag).unwrap(),
            gpu.download(&mom.matrix().source).unwrap(),
        )
    };
    let (d0, s0) = assemble(false);
    let (d1, s1) = assemble(true);
    assert!(
        bits_eq(&d0, &d1) && bits_eq(&s0, &s1),
        "coupling an empty pool moved the momentum matrix"
    );
}

// ======================================================================
//  The registry itself
// ======================================================================

/// The new registry lands in the same place, with the same units and the
/// same sign, as the S18 zone source it sits beside: a uniform body force
/// registered as a field must assemble the matrix a
/// [`crate::sources::SourceTerm::BodyForce`] over the whole mesh assembles.
/// Bitwise - both end as `source[P] += V_P * b`, in the same order, from the
/// same two kernels.
#[test]
fn a_field_body_force_is_the_zone_body_force_it_should_be() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([4, 5, 3], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let b = Vec3::new(0.3, -1.7, 9.81);

    let assemble = |as_field: bool| -> Vec<Scalar> {
        let mut mom = Momentum::new(&gpu, &gm, MomentumControls::default(), BuoyancyCoeffs::default()).unwrap();
        let u = GpuVectorField::zeros(&gpu, &gm, "U").unwrap();
        let phi = crate::field::GpuSurfaceScalarField::zeros(&gpu, &gm, "phi").unwrap();
        let nut = crate::field::GpuScalarField::zeros(&gpu, &gm, "nut").unwrap();
        if as_field {
            let field = gpu.upload(&vec![b; gm.n_cells]).unwrap();
            mom.field_sources_mut().clear(&gpu).unwrap();
            mom.field_sources_mut().register_explicit(&gpu, &field).unwrap();
        } else {
            let s = crate::sources::Source::new(
                &gpu,
                &hm,
                "everywhere",
                crate::sources::CellSelector::All,
                crate::sources::SourceTerm::BodyForce(b),
            )
            .unwrap();
            mom.sources_mut().push(s);
        }
        mom.assemble_only(&gpu, &u, &phi, &nut).unwrap();
        gpu.download(&mom.matrix().source).unwrap()
    };

    let zone = assemble(false);
    let field = assemble(true);
    assert!(
        bits_eq(&zone, &field),
        "the whole-field registry does not agree with the zone source it sits beside"
    );
}

/// Two registrations accumulate, `clear` forgets both, and `is_active` is
/// the host count the assembly branches on.
#[test]
fn the_registry_accumulates_and_clears() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([2, 2, 2], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let mut s = MomentumSources::new(&gpu, &gm).unwrap();
    assert!(!s.is_active());

    let a = gpu.upload(&vec![Vec3::new(1.0, 0.0, 0.0); gm.n_cells]).unwrap();
    let c = gpu.upload(&vec![Vec3::new(0.0, 2.0, 0.0); gm.n_cells]).unwrap();
    s.register_explicit(&gpu, &a).unwrap();
    s.register_explicit(&gpu, &c).unwrap();
    assert!(s.is_active());
    let got = gpu.download(s.su()).unwrap();
    assert!(got.iter().all(|v| v.x == 1.0 && v.y == 2.0 && v.z == 0.0));

    s.clear(&gpu).unwrap();
    assert!(!s.is_active());
    let got = gpu.download(s.su()).unwrap();
    assert!(got.iter().all(|v| v.x == 0.0 && v.y == 0.0 && v.z == 0.0));

    let short = gpu.zeros::<Vec3>(1).unwrap();
    let e = s.register_explicit(&gpu, &short).unwrap_err().to_string();
    assert!(e.contains("elements"), "{e}");
}

// ======================================================================
//  (68.10): the split, and the sign contract
// ======================================================================

/// At the linearisation point the semi-implicit split is the explicit source
/// exactly: `S_u + S_p u^n = f/rho`. The split changes what the matrix looks
/// like, never what was exchanged.
#[test]
fn the_semi_implicit_split_is_the_explicit_source_at_the_linearisation_point() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([4, 4, 4], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let u_gas = Vec3::new(1.5, -0.5, 2.0);
    let rho_g: Scalar = 1.2;
    let (u, rho) = gas(&gpu, &gm, u_gas, rho_g);
    let dt: Scalar = 1e-3;

    let seeds: Vec<SeedParcel> = (0..64u64)
        .map(|i| {
            seed(
                Vec3::new(unit(i), unit(i + 11), unit(i + 22)),
                Vec3::new(4.0 * unit(i + 33) - 2.0, 0.0, -5.0),
                4e-4,
                1.0 + 100.0 * unit(i + 44),
                i + 1,
            )
        })
        .collect();

    let run = |mode: CouplingMode| -> CouplingSnapshot {
        let mut p = Parcels::new(&gpu, &hm, &gm, base_controls(), &[], dt).unwrap();
        p.seed(&gpu, &hm, &seeds).unwrap();
        let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
        let mut cp = ParcelCoupling::new(
            &gpu,
            &p,
            CouplingControls { momentum: mode, energy: CouplingMode::Off, mass: MassCoupling::None },
        )
        .unwrap();
        one_step(&gpu, &mut p, &mut dep, &mut cp, &u, &rho, None, dt);
        cp.snapshot(&gpu).unwrap()
    };

    let ex = run(CouplingMode::Explicit);
    let si = run(CouplingMode::SemiImplicit);

    // The DEPOSIT is identical in both modes, bit for bit: the mode is a
    // linearisation, not a model.
    assert!(vec_bits_eq(&ex.force, &si.force));
    assert!(bits_eq(&ex.exchange, &si.exchange));

    let mut worst: Scalar = 0.0;
    let mut scale: Scalar = 0.0;
    let mut any_sink = false;
    for c in 0..gm.n_cells {
        let net = Vec3::new(
            si.momentum_su[c].x + si.momentum_sp[c] * u_gas.x,
            si.momentum_su[c].y + si.momentum_sp[c] * u_gas.y,
            si.momentum_su[c].z + si.momentum_sp[c] * u_gas.z,
        );
        worst = worst.max(mag(sub(net, ex.momentum_su[c])));
        scale = scale.max(mag(ex.momentum_su[c]));
        assert!(
            si.momentum_sp[c] <= 0.0,
            "cell {c}: S_p is {} and Patankar S4.2 needs it non-positive",
            si.momentum_sp[c]
        );
        assert!(si.exchange[c] >= 0.0);
        any_sink |= si.momentum_sp[c] < 0.0;
    }
    assert!(any_sink, "no cell had a sink, so the sign claim is vacuous");
    assert!(
        worst <= 1e-12 * scale,
        "the split is not the explicit source at u = u^n: {worst:e} against {scale:e}"
    );
}

/// **S13.4.1**: `momentum explicit` and `momentum semiImplicit` are two runs
/// identical in every byte but one, and they are REQUIRED to assemble a
/// different matrix. If they do not, the setting is doing nothing and the
/// test says so by name.
#[test]
fn the_momentum_coupling_mode_changes_the_matrix() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([4, 4, 4], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    // A gas velocity with a non-zero Z: `assemble_only` leaves `a.source`
    // holding the LAST component, and `beta u_z` is what the split adds to
    // it. With `u_z = 0` the two modes would agree on that one component and
    // the pair test would pass for the wrong reason.
    let (u, rho) = gas(&gpu, &gm, Vec3::new(1.0, 0.0, 2.0), 1.2);
    let dt: Scalar = 1e-3;
    let seeds: Vec<SeedParcel> = (0..32u64)
        .map(|i| {
            seed(
                Vec3::new(unit(i), unit(i + 5), unit(i + 9)),
                Vec3::new(0.0, 0.0, -8.0),
                5e-4,
                1000.0,
                i + 1,
            )
        })
        .collect();

    let run = |mode: CouplingMode| -> (Vec<Scalar>, Vec<Scalar>) {
        let mut p = Parcels::new(&gpu, &hm, &gm, base_controls(), &[], dt).unwrap();
        p.seed(&gpu, &hm, &seeds).unwrap();
        let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
        let mut cp = ParcelCoupling::new(
            &gpu,
            &p,
            CouplingControls { momentum: mode, energy: CouplingMode::Off, mass: MassCoupling::None },
        )
        .unwrap();
        one_step(&gpu, &mut p, &mut dep, &mut cp, &u, &rho, None, dt);

        let mut mom = Momentum::new(&gpu, &gm, MomentumControls::default(), BuoyancyCoeffs::default()).unwrap();
        let phi = crate::field::GpuSurfaceScalarField::zeros(&gpu, &gm, "phi").unwrap();
        let nut = crate::field::GpuScalarField::zeros(&gpu, &gm, "nut").unwrap();
        mom.field_sources_mut().clear(&gpu).unwrap();
        cp.register_momentum(&gpu, mom.field_sources_mut()).unwrap();
        mom.assemble_only(&gpu, &u, &phi, &nut).unwrap();
        (
            gpu.download(&mom.matrix().diag).unwrap(),
            gpu.download(&mom.matrix().source).unwrap(),
        )
    };

    let (d_ex, s_ex) = run(CouplingMode::Explicit);
    let (d_si, s_si) = run(CouplingMode::SemiImplicit);
    assert!(
        !bits_eq(&d_ex, &d_si),
        "`momentum semiImplicit` left the diagonal exactly as `explicit` did, so the \
         setting is inert - S13.4.1 requires it to bite"
    );
    assert!(
        !bits_eq(&s_ex, &s_si),
        "`momentum semiImplicit` left the source exactly as `explicit` did"
    );
}

/// (68.10)'s cost, measured rather than assumed. At the linearisation point the
/// defect is exactly zero; away from it, it is `sum_P V_P beta_P (u - u^n) dt`,
/// which is the momentum the split moved relative to the explicit source. And
/// under `explicit` there is nothing to move, so it is zero by definition
/// rather than by arithmetic.
#[test]
fn the_linearisation_defect_is_what_the_split_moved() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([3, 3, 3], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let u_lin_value = Vec3::new(1.0, -0.5, 2.0);
    let (u, rho) = gas(&gpu, &gm, u_lin_value, 1.2);
    let dt: Scalar = 1e-3;

    let seeds: Vec<SeedParcel> = (0..24u64)
        .map(|i| {
            seed(
                Vec3::new(unit(i), unit(i + 41), unit(i + 82)),
                Vec3::new(0.0, 0.0, -6.0),
                4e-4,
                100.0,
                i + 1,
            )
        })
        .collect();

    for mode in [CouplingMode::Explicit, CouplingMode::SemiImplicit] {
        let mut p = Parcels::new(&gpu, &hm, &gm, base_controls(), &[], dt).unwrap();
        p.seed(&gpu, &hm, &seeds).unwrap();
        let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
        let mut cp = ParcelCoupling::new(
            &gpu,
            &p,
            CouplingControls { momentum: mode, energy: CouplingMode::Off, mass: MassCoupling::None },
        )
        .unwrap();
        one_step(&gpu, &mut p, &mut dep, &mut cp, &u, &rho, None, dt);

        let u_lin = vec![u_lin_value; gm.n_cells];
        let at_point = cp.linearisation_defect(&gpu, &u, &u_lin).unwrap();
        assert_eq!(
            (at_point.x, at_point.y, at_point.z),
            (0.0, 0.0, 0.0),
            "`{}`: the defect at the linearisation point is not zero",
            mode.name()
        );

        // Now move the gas away from where the split was linearised.
        let du = Vec3::new(0.25, 0.0, -0.75);
        let mut moved = GpuVectorField::zeros(&gpu, &gm, "U").unwrap();
        moved.f = gpu
            .upload(&vec![add(u_lin_value, du); gm.n_cells])
            .unwrap();
        let got = cp.linearisation_defect(&gpu, &moved, &u_lin).unwrap();

        let beta = gpu.download(cp.exchange_rate()).unwrap();
        let vol = gpu.download(&gm.v).unwrap();
        let mut want = Vec3::ZERO;
        if mode == CouplingMode::SemiImplicit {
            for c in 0..gm.n_cells {
                let w = vol[c] * beta[c] * dt;
                want.x += w * du.x;
                want.y += w * du.y;
                want.z += w * du.z;
            }
            assert!(mag(want) > 0.0, "the fixture exchanged nothing");
        }
        let err = mag(sub(got, want));
        let scale = mag(want).max(1e-30);
        assert!(
            err <= 1e-14 * scale,
            "`{}`: defect ({:e} {:e} {:e}) against ({:e} {:e} {:e})",
            mode.name(),
            got.x,
            got.y,
            got.z,
            want.x,
            want.y,
            want.z
        );
    }
}

/// **S13.4.1**: `momentum off` and `momentum explicit` differ, and `off`
/// registers nothing at all.
#[test]
fn momentum_off_registers_nothing() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([3, 3, 3], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (u, rho) = gas(&gpu, &gm, Vec3::new(1.0, 0.0, 0.0), 1.2);
    let dt: Scalar = 1e-3;

    let mut p = Parcels::new(&gpu, &hm, &gm, base_controls(), &[], dt).unwrap();
    p.seed(
        &gpu,
        &hm,
        &[seed(Vec3::new(0.5, 0.5, 0.5), Vec3::new(0.0, 0.0, -9.0), 5e-4, 1e4, 1)],
    )
    .unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();

    for (mode, want_active) in [
        (CouplingMode::Off, false),
        (CouplingMode::Explicit, true),
        (CouplingMode::SemiImplicit, true),
    ] {
        let mut cp = ParcelCoupling::new(
            &gpu,
            &p,
            CouplingControls { momentum: mode, energy: CouplingMode::Off, mass: MassCoupling::None },
        )
        .unwrap();
        one_step(&gpu, &mut p, &mut dep, &mut cp, &u, &rho, None, dt);
        let mut s = MomentumSources::new(&gpu, &gm).unwrap();
        s.clear(&gpu).unwrap();
        cp.register_momentum(&gpu, &mut s).unwrap();
        assert_eq!(
            s.is_active(),
            want_active,
            "`momentum {}` registered the wrong thing",
            mode.name()
        );
        // The DEPOSIT is written whatever the mode - `off` suppresses the
        // registration, not the measurement, so a case can still write the
        // force density it chose not to apply.
        let snap = cp.snapshot(&gpu).unwrap();
        assert!(
            snap.force.iter().any(|f| f.z != 0.0),
            "`momentum {}` did not even deposit the force density",
            mode.name()
        );
    }
}

// ======================================================================
//  S68.5: heating, and its conservation gate
// ======================================================================

/// **Gate 68-A, energy.** The heat the gas gives up is exactly the heat the
/// droplets gained, to round-off - the same claim as the momentum gate, on
/// the same construction.
#[test]
fn gate_68a_energy_is_conserved_between_the_phases() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([5, 5, 5], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (u, rho) = gas(&gpu, &gm, Vec3::new(0.5, 0.0, 0.0), 1.2);
    let t_gas = gpu.upload(&vec![600.0 as Scalar; gm.n_cells]).unwrap();
    let dt: Scalar = 1e-3;

    let ctrl = ParcelControls {
        capacity: 256,
        physics: ParcelPhysics::Heating,
        ..base_controls()
    };
    let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[], dt).unwrap();
    let seeds: Vec<SeedParcel> = (0..120u64)
        .map(|i| SeedParcel {
            position: Vec3::new(unit(i), unit(i + 31), unit(i + 62)),
            velocity: Vec3::new(0.0, 0.0, -2.0 * unit(i + 93)),
            diameter: 1e-4 + 4e-4 * unit(i + 124),
            temperature: 280.0 + 20.0 * unit(i + 155),
            n_p: 1.0 + 1000.0 * unit(i + 186),
            uid: Some(i + 1),
        })
        .collect();
    p.seed(&gpu, &hm, &seeds).unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    let mut cp = ParcelCoupling::new(
        &gpu,
        &p,
        CouplingControls {
            momentum: CouplingMode::Explicit,
            energy: CouplingMode::Explicit,
            mass: MassCoupling::None,
        },
    )
    .unwrap();

    let vol = gpu.download(&gm.v).unwrap();
    for step in 0..5 {
        one_step(&gpu, &mut p, &mut dep, &mut cp, &u, &rho, Some(&t_gas), dt);
        let s = cp.snapshot(&gpu).unwrap();
        let given: Scalar = (0..gm.n_cells).map(|c| vol[c] * s.heat[c] * dt).sum();
        let taken = live_parcel_heat(&p.snapshot(&gpu).unwrap());
        assert!(taken > 0.0, "the droplets took no heat, so the gate is vacuous");
        assert!(
            (given + taken).abs() <= 1e-13 * taken,
            "step {step}: the gas gave {given:e} J and the droplets took {taken:e} J"
        );
        // The gas is hotter than the droplets, so the gas must be LOSING
        // heat: the sign is the first thing to get wrong and the first
        // thing to check.
        assert!(given < 0.0, "step {step}: the hot gas gained heat from cold droplets");
    }

    // The droplets warmed towards the gas and none overshot it.
    let s = p.snapshot(&gpu).unwrap();
    for i in 0..s.n_slots {
        assert!(
            s.temperature[i] > 279.0 && s.temperature[i] <= 600.0,
            "droplet {i} is at {} K",
            s.temperature[i]
        );
    }
    assert!(s.temperature[0] > 280.0, "no droplet warmed at all");
}

/// A droplet in still gas relaxes towards the gas temperature with the
/// lumped-capacity time constant `tau_T = rho_l c_l d^2 / (6 Nu k_g)`, and
/// at `Re = 0` the Ranz-Marshall Nusselt number is exactly 2. So this is a
/// closed form with no correlation left in it, and it is what (68.9) must
/// reproduce.
#[test]
fn a_still_droplet_relaxes_at_the_lumped_capacity_rate() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([1, 1, 4], [1.0, 1.0, 4.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (u, rho) = gas(&gpu, &gm, Vec3::ZERO, 1.2);
    let t_g: Scalar = 500.0;
    let t_gas = gpu.upload(&vec![t_g; gm.n_cells]).unwrap();

    let d: Scalar = 2e-4;
    let t0: Scalar = 300.0;
    let dt: Scalar = 1e-4;
    let ctrl = ParcelControls {
        capacity: 2,
        physics: ParcelPhysics::Heating,
        drag: DragModel::None,
        gravity: Vec3::ZERO,
        ..base_controls()
    };
    let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[], dt).unwrap();
    p.seed(
        &gpu,
        &hm,
        &[SeedParcel {
            position: Vec3::new(0.5, 0.5, 2.0),
            velocity: Vec3::ZERO,
            diameter: d,
            temperature: t0,
            n_p: 1.0,
            uid: Some(1),
        }],
    )
    .unwrap();

    let n = 200;
    for _ in 0..n {
        p.step(&gpu, &u, &rho, Some(&t_gas), dt).unwrap();
    }
    let got = p.snapshot(&gpu).unwrap().temperature[0];

    let tau = ctrl.rho_liquid * ctrl.c_liquid * d * d / (6.0 * 2.0 * ctrl.k_gas);
    let want = t_g + (t0 - t_g) * (-(n as Scalar) * dt / tau).exp();
    assert!(
        (got - want).abs() < 1e-9 * (t_g - t0),
        "after {n} steps the droplet is at {got} K, the closed form says {want} K \
         (tau_T = {tau} s)"
    );
}

/// **S13.4.1**, three rows at once: `cLiquid`, `kGas` and `cpGas` each change
/// what a heating run deposits. `cpGas` enters only through `Pr`, so it bites
/// only when `Re > 0` - which is why this fixture has the droplet moving.
#[test]
fn every_heating_property_changes_what_is_deposited() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([2, 2, 2], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (u, rho) = gas(&gpu, &gm, Vec3::new(3.0, 0.0, 0.0), 1.2);
    let t_gas = gpu.upload(&vec![700.0 as Scalar; gm.n_cells]).unwrap();
    let dt: Scalar = 1e-4;

    let run = |ctrl: ParcelControls| -> Vec<Scalar> {
        let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[], dt).unwrap();
        p.seed(
            &gpu,
            &hm,
            &[SeedParcel {
                position: Vec3::new(0.25, 0.25, 0.25),
                velocity: Vec3::new(0.0, 0.0, -1.0),
                diameter: 3e-4,
                temperature: 300.0,
                n_p: 100.0,
                uid: Some(1),
            }],
        )
        .unwrap();
        let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
        let mut cp = ParcelCoupling::new(
            &gpu,
            &p,
            CouplingControls {
                momentum: CouplingMode::Explicit,
                energy: CouplingMode::Explicit,
                mass: MassCoupling::None,
            },
        )
        .unwrap();
        one_step(&gpu, &mut p, &mut dep, &mut cp, &u, &rho, Some(&t_gas), dt);
        cp.snapshot(&gpu).unwrap().heat
    };

    let heating = ParcelControls {
        capacity: 4,
        physics: ParcelPhysics::Heating,
        ..base_controls()
    };
    let base = run(heating);
    for (name, ctrl) in [
        ("cLiquid", ParcelControls { c_liquid: 4183.0, ..heating }),
        ("kGas", ParcelControls { k_gas: 0.027, ..heating }),
        ("cpGas", ParcelControls { cp_gas: 1006.0, ..heating }),
    ] {
        let other = run(ctrl);
        assert!(
            !bits_eq(&base, &other),
            "changing `{name}` did not change one bit of the deposited heat; S13.4.1 \
             requires the pair to differ"
        );
    }

    // ... and `physics inert` deposits no heat at all, which is the fourth
    // pair and the one that says the enum is doing the work.
    let inert = ParcelControls { physics: ParcelPhysics::Inert, ..heating };
    let mut p = Parcels::new(&gpu, &hm, &gm, inert, &[], dt).unwrap();
    p.seed(
        &gpu,
        &hm,
        &[seed(Vec3::new(0.25, 0.25, 0.25), Vec3::new(0.0, 0.0, -1.0), 3e-4, 100.0, 1)],
    )
    .unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    let mut cp = ParcelCoupling::new(&gpu, &p, momentum_only()).unwrap();
    one_step(&gpu, &mut p, &mut dep, &mut cp, &u, &rho, None, dt);
    let s = cp.snapshot(&gpu).unwrap();
    assert!(s.heat.iter().all(|q| *q == 0.0));
    assert!(base.iter().any(|q| *q != 0.0));
}

/// The energy split satisfies the same sign contract as the momentum one,
/// and `EnergySources` takes it without complaint.
#[test]
fn the_energy_sink_is_non_positive_and_registers() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([3, 3, 3], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (u, rho) = gas(&gpu, &gm, Vec3::ZERO, 1.2);
    let t_gas = gpu.upload(&vec![800.0 as Scalar; gm.n_cells]).unwrap();
    let dt: Scalar = 1e-4;

    let ctrl = ParcelControls {
        capacity: 32,
        physics: ParcelPhysics::Heating,
        ..base_controls()
    };
    let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[], dt).unwrap();
    let seeds: Vec<SeedParcel> = (0..16u64)
        .map(|i| SeedParcel {
            position: Vec3::new(unit(i), unit(i + 3), unit(i + 6)),
            velocity: Vec3::ZERO,
            diameter: 2e-4,
            temperature: 300.0,
            n_p: 50.0,
            uid: Some(i + 1),
        })
        .collect();
    p.seed(&gpu, &hm, &seeds).unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    let mut cp = ParcelCoupling::new(
        &gpu,
        &p,
        CouplingControls {
            momentum: CouplingMode::Off,
            energy: CouplingMode::SemiImplicit,
            mass: MassCoupling::None,
        },
    )
    .unwrap();
    one_step(&gpu, &mut p, &mut dep, &mut cp, &u, &rho, Some(&t_gas), dt);

    let s = cp.snapshot(&gpu).unwrap();
    assert!(s.energy_sp.iter().all(|x| *x <= 0.0));
    assert!(s.energy_sp.iter().any(|x| *x < 0.0));
    // At T = T_gas the split is the explicit source.
    let mut worst: Scalar = 0.0;
    for c in 0..gm.n_cells {
        worst = worst.max((s.energy_q[c] + s.energy_sp[c] * 800.0 - s.heat[c]).abs());
    }
    let scale = s.heat.iter().fold(0.0 as Scalar, |a, b| a.max(b.abs()));
    assert!(worst <= 1e-12 * scale, "{worst:e} against {scale:e}");

    let mut es = EnergySources::new(&gpu, &gm).unwrap();
    es.clear(&gpu).unwrap();
    cp.register_energy(&gpu, &mut es).unwrap();
    // Numeric, not bitwise: the accumulator starts at `+0.0` and an empty
    // cell contributes `-0.0`, whose sum is `+0.0`. The VALUES are equal
    // everywhere and that is the claim; the sign of a zero is not.
    let q = gpu.download(es.q()).unwrap();
    assert!(q.iter().zip(&s.energy_q).all(|(a, b)| a == b));
    let sp = gpu.download(es.sp()).unwrap();
    assert!(sp.iter().zip(&s.energy_sp).all(|(a, b)| a == b));
    assert!(s.energy_q.iter().any(|x| *x != 0.0));
}

// ======================================================================
//  Reproducibility
// ======================================================================

/// Two identical runs couple identical bits, and a run whose parcels were
/// shifted into different slots - the S67 canonicalisation claim, carried
/// through the coupling - couples the same bits as well.
#[test]
fn the_coupling_is_bitwise_reproducible_under_a_slot_permutation() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([4, 4, 4], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (u, rho) = gas(&gpu, &gm, Vec3::new(1.0, 1.0, 0.0), 1.2);
    let dt: Scalar = 1e-3;

    let make = |i: u64| {
        seed(
            Vec3::new(unit(i), unit(i + 17), unit(i + 34)),
            Vec3::new(2.0 * unit(i + 51) - 1.0, 0.0, -4.0),
            3e-4 + 2e-4 * unit(i + 68),
            1.0 + 10.0 * unit(i + 85),
            i + 1,
        )
    };
    let forward: Vec<SeedParcel> = (0..48u64).map(make).collect();
    let mut reversed = forward.clone();
    reversed.reverse();

    let run = |seeds: &[SeedParcel]| -> CouplingSnapshot {
        let mut p = Parcels::new(&gpu, &hm, &gm, base_controls(), &[], dt).unwrap();
        p.seed(&gpu, &hm, seeds).unwrap();
        let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
        let mut cp = ParcelCoupling::new(&gpu, &p, momentum_only()).unwrap();
        for _ in 0..3 {
            one_step(&gpu, &mut p, &mut dep, &mut cp, &u, &rho, None, dt);
        }
        cp.snapshot(&gpu).unwrap()
    };

    let a = run(&forward);
    let b = run(&forward);
    let c = run(&reversed);
    assert!(vec_bits_eq(&a.force, &b.force) && bits_eq(&a.exchange, &b.exchange));
    assert!(
        vec_bits_eq(&a.force, &c.force) && vec_bits_eq(&a.momentum_su, &c.momentum_su),
        "reversing the seed order moved the coupled force; the S67 canonicalisation \
         does not reach the coupling"
    );
    assert!(a.force.iter().any(|f| f.z != 0.0), "nothing was coupled at all");
}

/// S68.9 row 19: the whole step - pool, sort, CSR, gather and coupling - captures
/// ONCE and replays bitwise while the working set grows. The launch geometry
/// of the gather is a setup constant and reads nothing back.
#[test]
fn the_coupling_captures_once_and_replays() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([4, 4, 4], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (u, rho) = gas(&gpu, &gm, Vec3::new(0.0, 0.0, 2.0), 1.2);
    let dt: Scalar = 5e-4;
    let inj = Injector {
        position: Vec3::new(0.5, 0.5, 0.5),
        axis: Vec3::new(0.0, 0.0, -1.0),
        cone_half_angle: 0.4,
        standoff: 0.0,
        speed: 4.0,
        diameter: 3e-4,
        temperature: 293.15,
        mass_flow: 1e-4,
        parcels_per_event: 8,
        interval: 0.0,
    };

    let eager = {
        let mut p = Parcels::new(&gpu, &hm, &gm, base_controls(), &[inj], dt).unwrap();
        let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
        let mut cp = ParcelCoupling::new(&gpu, &p, momentum_only()).unwrap();
        for _ in 0..12 {
            one_step(&gpu, &mut p, &mut dep, &mut cp, &u, &rho, None, dt);
        }
        cp.snapshot(&gpu).unwrap()
    };

    let mut p = Parcels::new(&gpu, &hm, &gm, base_controls(), &[inj], dt).unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    let mut cp = ParcelCoupling::new(&gpu, &p, momentum_only()).unwrap();
    let captured = gpu
        .capture(|_| {
            p.step(&gpu, &u, &rho, None, dt)?;
            dep.update(&gpu, &p)?;
            cp.update(&gpu, &p, &dep, &rho, &u, None, dt)
        })
        .expect("capture must not fail: nothing in the sequence allocates or reads back");
    let Some(mut graph) = captured else {
        panic!("the capture produced an empty graph");
    };
    graph.upload().unwrap();
    for _ in 0..12 {
        graph.launch().unwrap();
    }
    gpu.sync().unwrap();
    let replayed = cp.snapshot(&gpu).unwrap();

    assert!(
        vec_bits_eq(&eager.force, &replayed.force)
            && bits_eq(&eager.exchange, &replayed.exchange)
            && vec_bits_eq(&eager.momentum_su, &replayed.momentum_su),
        "the captured graph did not reproduce the eager coupling bit for bit"
    );
    assert!(eager.force.iter().any(|f| f.z != 0.0));
}

// ======================================================================
//  Refusals - SPEC-LIT S13.4
// ======================================================================

#[test]
fn energy_coupling_on_inert_parcels_is_refused_by_name() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([2, 2, 2], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let p = Parcels::new(&gpu, &hm, &gm, base_controls(), &[], 1e-3).unwrap();
    let e = match ParcelCoupling::new(
        &gpu,
        &p,
        CouplingControls {
            momentum: CouplingMode::Explicit,
            energy: CouplingMode::Explicit,
            mass: MassCoupling::None,
        },
    ) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("energy coupling on inert parcels was accepted"),
    };
    assert!(e.contains("INFINITE heat bath"), "{e}");
    assert!(e.contains("physics heating"), "{e}");
}

#[test]
fn the_gas_temperature_contract_is_refused_in_both_directions() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([2, 2, 2], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (u, rho) = gas(&gpu, &gm, Vec3::ZERO, 1.2);
    let t = gpu.upload(&vec![400.0 as Scalar; gm.n_cells]).unwrap();

    let mut inert = Parcels::new(&gpu, &hm, &gm, base_controls(), &[], 1e-3).unwrap();
    let e = inert
        .step(&gpu, &u, &rho, Some(&t), 1e-3)
        .unwrap_err()
        .to_string();
    assert!(e.contains("read and ignored"), "{e}");

    let heating = ParcelControls { physics: ParcelPhysics::Heating, ..base_controls() };
    let mut hot = Parcels::new(&gpu, &hm, &gm, heating, &[], 1e-3).unwrap();
    let e = hot.step(&gpu, &u, &rho, None, 1e-3).unwrap_err().to_string();
    assert!(e.contains("no gas temperature"), "{e}");
}

#[test]
fn a_coupling_handed_the_wrong_pool_or_dt_is_refused_by_name() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([2, 2, 2], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (u, rho) = gas(&gpu, &gm, Vec3::ZERO, 1.2);
    let dt: Scalar = 1e-3;

    let mut p = Parcels::new(&gpu, &hm, &gm, base_controls(), &[], dt).unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    let mut cp = ParcelCoupling::new(&gpu, &p, momentum_only()).unwrap();
    p.step(&gpu, &u, &rho, None, dt).unwrap();
    dep.update(&gpu, &p).unwrap();

    let e = cp
        .update(&gpu, &p, &dep, &rho, &u, None, 2e-3)
        .unwrap_err()
        .to_string();
    assert!(e.contains("dt = 0.002"), "{e}");

    let other = Parcels::new(
        &gpu,
        &hm,
        &gm,
        ParcelControls { capacity: 99, ..base_controls() },
        &[],
        dt,
    )
    .unwrap();
    let e = cp
        .update(&gpu, &other, &dep, &rho, &u, None, dt)
        .unwrap_err()
        .to_string();
    assert!(e.contains("slots"), "{e}");
}

#[test]
fn evaporation_is_refused_by_name_with_what_it_would_need() {
    let e = MassCoupling::from_name("evaporation").unwrap_err().to_string();
    assert!(e.contains("3x3"), "{e}");
    assert!(e.contains("target_divergence"), "{e}");
    assert!(e.contains("sprinkler"), "{e}");
    assert!(MassCoupling::from_name("none").is_ok());

    let e = ParcelPhysics::from_name("evaporating").unwrap_err().to_string();
    assert!(e.contains("heating"), "{e}");
    assert_eq!(ParcelPhysics::from_name("heating").unwrap(), ParcelPhysics::Heating);
    assert_eq!(ParcelPhysics::from_name("inert").unwrap(), ParcelPhysics::Inert);

    let e = CouplingMode::from_name("magic").unwrap_err().to_string();
    assert!(e.contains("semiImplicit"), "{e}");
    assert_eq!(CouplingMode::from_name("off").unwrap(), CouplingMode::Off);
    assert_eq!(
        CouplingMode::from_name("semiImplicit").unwrap(),
        CouplingMode::SemiImplicit
    );
}

/// The banner says every setting the run will use - S13.4.2 - so that a log
/// records what was in force rather than leaving it to be inferred.
#[test]
fn the_banner_names_every_setting() {
    let d = CouplingControls::default().describe();
    for want in ["momentum=explicit", "energy=off", "mass=none", "S68"] {
        assert!(d.contains(want), "{d}");
    }
    let hot = ParcelControls { physics: ParcelPhysics::Heating, ..base_controls() }.describe();
    for want in ["physics=heating", "cLiquid=4182", "kGas=0.026", "cpGas=1005"] {
        assert!(hot.contains(want), "{hot}");
    }
    // ... and an inert run does not print three numbers nothing will read.
    let cold = base_controls().describe();
    assert!(!cold.contains("cLiquid"), "{cold}");
}

/// A property that will be read has to be a number. All three are checked at
/// setup whatever the physics, because the error is more use there.
#[test]
fn a_nonsense_heating_property_is_refused_at_setup() {
    for (name, ctrl) in [
        ("cLiquid", ParcelControls { c_liquid: 0.0, ..base_controls() }),
        ("kGas", ParcelControls { k_gas: -1.0, ..base_controls() }),
        ("cpGas", ParcelControls { cp_gas: Scalar::NAN, ..base_controls() }),
    ] {
        let e = ctrl.validate().unwrap_err().to_string();
        assert!(e.contains(name), "{e}");
    }
}

/// What the coupling costs, reported before it is spent - S68.8.
#[test]
fn the_memory_it_costs_is_reportable() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([4, 4, 4], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let p = Parcels::new(&gpu, &hm, &gm, base_controls(), &[], 1e-3).unwrap();
    let cp = ParcelCoupling::new(&gpu, &p, momentum_only()).unwrap();
    assert_eq!(cp.device_bytes(), gm.n_cells * 15 * 8);
    assert_eq!(cp.controls().momentum, CouplingMode::Explicit);
}

/// The matrix a `GpuLduMatrix` sees is the one the registry wrote: a sanity
/// check that `fvm_sp(-1)` on a non-positive `S_p` STRENGTHENS the diagonal,
/// which is the whole point of the implicit half.
#[test]
fn the_implicit_half_strengthens_the_diagonal() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([3, 3, 3], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let mut a = GpuLduMatrix::new(&gpu, &gm).unwrap();
    a.zero(&gpu).unwrap();
    let sp = gpu.upload(&vec![-2.0 as Scalar; gm.n_cells]).unwrap();
    let fvk = crate::fv::FvKernels::new(&gpu).unwrap();
    crate::fv::fvm_sp(&gpu, &fvk, &mut a, &gm, &sp, -1.0).unwrap();
    let diag = gpu.download(&a.diag).unwrap();
    let vol = gpu.download(&gm.v).unwrap();
    for c in 0..gm.n_cells {
        assert!(
            (diag[c] - 2.0 * vol[c]).abs() <= 1e-15 * 2.0 * vol[c],
            "cell {c}: diag {} against {}",
            diag[c],
            2.0 * vol[c]
        );
    }
}

/// The host reference the gates are posed on sums only the live parcels, and
/// a dead one is excluded rather than silently included with a stale value.
#[test]
fn the_host_reference_counts_only_live_parcels() {
    let mut s = ParcelSnapshot {
        x: vec![Vec3::ZERO; 3],
        u: vec![Vec3::ZERO; 3],
        d: vec![1e-4; 3],
        temperature: vec![300.0; 3],
        n_p: vec![2.0, 3.0, 5.0],
        cell: vec![0, -1, 2],
        uid: vec![1, 2, 3],
        flags: vec![1, 0, 1],
        impulse: vec![Vec3::new(1.0, 0.0, 0.0); 3],
        exchange: vec![0.0; 3],
        heat: vec![10.0, 100.0, 1000.0],
        heat_exchange: vec![0.0; 3],
        n_slots: 3,
    };
    assert_eq!(live_parcel_impulse(&s).x, 7.0);
    assert_eq!(live_parcel_heat(&s), 2.0 * 10.0 + 5.0 * 1000.0);
    s.heat.clear();
    assert_eq!(live_parcel_heat(&s), 0.0);
}
