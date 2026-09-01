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
use crate::parcels::EvaporationControls;
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
        evaporation: EvaporationControls::default(),
        impact: crate::parcels::WallImpactControls::default(),
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
    p.step(gpu, u, rho, t, None, dt).unwrap();
    dep.update(gpu, p).unwrap();
    cp.update(gpu, p, dep, rho, u, t, None, dt).unwrap();
}

/// [`one_step`] for an evaporating pool: the gas carries a vapour field, so
/// both the parcel step and the S77 gather read it.
#[allow(clippy::too_many_arguments)]
fn one_wet_step<'a>(
    gpu: &Gpu,
    p: &mut Parcels<'a>,
    dep: &mut ParcelDeposition<'a>,
    cp: &mut ParcelCoupling<'a>,
    u: &GpuVectorField,
    rho: &DevBuf<Scalar>,
    t: &DevBuf<Scalar>,
    y: &DevBuf<Scalar>,
    dt: Scalar,
) {
    p.step(gpu, u, rho, Some(t), Some(y), dt).unwrap();
    dep.update(gpu, p).unwrap();
    cp.update(gpu, p, dep, rho, u, Some(t), Some(y), dt).unwrap();
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
        ("OFC_MASS_NONE", MassCoupling::None.code()),
        ("OFC_MASS_EVAPORATION", MassCoupling::Evaporation.code()),
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
        ("OFP_PHYS_EVAPORATING", ParcelPhysics::Evaporating.code()),
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
    p.step(&gpu, &u, &rho, None, None, dt).unwrap();
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
        p.step(&gpu, &u, &rho, None, None, dt).unwrap();
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
        p.step(&gpu, &u, &rho, Some(&t_gas), None, dt).unwrap();
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
            p.step(&gpu, &u, &rho, None, None, dt)?;
            dep.update(&gpu, &p)?;
            cp.update(&gpu, &p, &dep, &rho, &u, None, None, dt)
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
        .step(&gpu, &u, &rho, Some(&t), None, 1e-3)
        .unwrap_err()
        .to_string();
    assert!(e.contains("read and ignored"), "{e}");

    let heating = ParcelControls { physics: ParcelPhysics::Heating, ..base_controls() };
    let mut hot = Parcels::new(&gpu, &hm, &gm, heating, &[], 1e-3).unwrap();
    let e = hot.step(&gpu, &u, &rho, None, None, 1e-3).unwrap_err().to_string();
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
    p.step(&gpu, &u, &rho, None, None, dt).unwrap();
    dep.update(&gpu, &p).unwrap();

    let e = cp
        .update(&gpu, &p, &dep, &rho, &u, None, None, 2e-3)
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
        .update(&gpu, &other, &dep, &rho, &u, None, None, dt)
        .unwrap_err()
        .to_string();
    assert!(e.contains("slots"), "{e}");
}

#[test]
fn the_menu_takes_evaporation_now_and_still_refuses_the_rest() {
    // S68 refused every mass coupling because there was nothing to give;
    // S76 made the droplets evaporate and the reason became "there is
    // nowhere to put it"; S77 built the somewhere, so the name is accepted.
    // What is NOT accepted is a name for something that does not exist.
    assert_eq!(
        MassCoupling::from_name("evaporation").unwrap(),
        MassCoupling::Evaporation
    );
    assert_eq!(MassCoupling::from_name("vapour").unwrap(), MassCoupling::Evaporation);
    assert_eq!(MassCoupling::from_name("none").unwrap(), MassCoupling::None);

    let e = MassCoupling::from_name("breakup").unwrap_err().to_string();
    assert!(e.contains("evaporation"), "{e}");
    assert!(e.contains("none"), "{e}");

    assert_eq!(
        ParcelPhysics::from_name("evaporating").unwrap(),
        ParcelPhysics::Evaporating
    );
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
    // S77's value has a name of its own and the banner prints it.
    let wet = CouplingControls {
        momentum: CouplingMode::Explicit,
        energy: CouplingMode::Explicit,
        mass: MassCoupling::Evaporation,
    }
    .describe();
    assert!(wet.contains("mass=evaporation"), "{wet}");
    assert!(wet.contains("energy=explicit"), "{wet}");
    let hot = ParcelControls { physics: ParcelPhysics::Heating, ..base_controls() }.describe();
    for want in ["physics=heating", "cLiquid=4182", "kGas=0.026", "cpGas=1005"] {
        assert!(hot.contains(want), "{hot}");
    }
    // ... and an inert run does not print three numbers nothing will read.
    let cold = base_controls().describe();
    assert!(!cold.contains("cLiquid"), "{cold}");
    assert!(!cold.contains("saturation="), "{cold}");
    assert!(!hot.contains("saturation="), "{hot}");
    // S76.2: an evaporating run prints the twelve numbers and two menus that
    // ONLY it reads, on a second line, and the thermal three as well.
    let wet =
        ParcelControls { physics: ParcelPhysics::Evaporating, ..base_controls() }.describe();
    for want in [
        "physics=evaporating",
        "cLiquid=4182",
        "saturation=hylandWexler",
        "transfer=abramzonSirignano",
        "pAmbient=101325",
        "tCrit=647.096",
        "S76",
    ] {
        assert!(wet.contains(want), "{wet}");
    }
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
        mass_lost: vec![0.0; 3],
        latent: vec![0.0; 3],
        n_slots: 3,
    };
    assert_eq!(live_parcel_impulse(&s).x, 7.0);
    assert_eq!(live_parcel_heat(&s), 2.0 * 10.0 + 5.0 * 1000.0);
    s.heat.clear();
    assert_eq!(live_parcel_heat(&s), 0.0);
}

// ======================================================================
//  SPEC-LIT S77 - the vapour, the enthalpy it carries, and the volume
//  it makes
// ======================================================================

/// An evaporating pool: S76's physics, S66's everything else.
fn wet_controls() -> ParcelControls {
    ParcelControls {
        capacity: 256,
        physics: ParcelPhysics::Evaporating,
        ..base_controls()
    }
}

/// All three couplings on - the only combination S77.7 allows for an
/// evaporating pool.
fn full_coupling() -> CouplingControls {
    CouplingControls {
        momentum: CouplingMode::Explicit,
        energy: CouplingMode::Explicit,
        mass: MassCoupling::Evaporation,
    }
}

/// A gas at rest with a temperature and a vapour fraction, uniform, so that
/// the host reference of S77.9's gates can carry one number for each.
fn wet_gas(
    gpu: &Gpu,
    m: &GpuMesh,
    t: Scalar,
    yv: Scalar,
    rho: Scalar,
) -> (GpuVectorField, DevBuf<Scalar>, DevBuf<Scalar>, DevBuf<Scalar>) {
    let (u, r) = gas(gpu, m, Vec3::ZERO, rho);
    let tg = gpu.upload(&vec![t; m.n_cells]).unwrap();
    let yg = gpu.upload(&vec![yv; m.n_cells]).unwrap();
    (u, r, tg, yg)
}

/// A spread of droplets over the box, weights over three decades so a
/// dropped `n_p` cannot hide.
fn wet_seeds(n: u64) -> Vec<SeedParcel> {
    (0..n)
        .map(|i| SeedParcel {
            position: Vec3::new(unit(i), unit(i + 31), unit(i + 62)),
            velocity: Vec3::new(0.0, 0.0, -0.2 * unit(i + 93)),
            diameter: 6e-5 + 2.4e-4 * unit(i + 124),
            temperature: 285.0 + 8.0 * unit(i + 155),
            n_p: 1.0 + 1000.0 * unit(i + 186),
            uid: Some(i + 1),
        })
        .collect()
}

/// **Gate 77-A.** The vapour the gas is given is the vapour the parcels
/// lost, to round-off, and it is an IDENTITY rather than a tolerance for
/// S68's reason: `pdmv` is the difference of the step's two endpoint masses
/// (76.11), the gather multiplies it by `n_p` and divides by `V dt`, and the
/// only way for the two sides to differ is a dropped weight, a missing
/// volume or a `dt` that is not the one the mass was lost over.
///
/// The SIGN convention is the one place this differs from (68.4). S68's
/// `f_P` is minus what the parcel took, so its identity reads
/// `given + taken = 0`; `mass_lost` is ALREADY a loss, so `mdot_P` is plus
/// what the parcel gave up and the identity reads `given = taken`. Writing
/// it the other way round is the first mistake available here and this test
/// is what would catch it.
#[test]
fn gate_77a_the_vapour_the_parcels_lost_is_the_vapour_the_gas_is_given() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([5, 5, 5], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (u, rho, t_gas, y_gas) = wet_gas(&gpu, &gm, 340.0, 0.004, 1.0);
    let dt: Scalar = 2e-3;

    let mut p = Parcels::new(&gpu, &hm, &gm, wet_controls(), &[], dt).unwrap();
    p.seed(&gpu, &hm, &wet_seeds(120)).unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    let mut cp = ParcelCoupling::new(&gpu, &p, full_coupling()).unwrap();

    let vol = gpu.download(&gm.v).unwrap();
    let mut worst: Scalar = 0.0;
    let mut total: Scalar = 0.0;
    for step in 0..6 {
        one_wet_step(&gpu, &mut p, &mut dep, &mut cp, &u, &rho, &t_gas, &y_gas, dt);
        let s = cp.snapshot(&gpu).unwrap();
        let given: Scalar = (0..gm.n_cells).map(|c| vol[c] * s.vapour[c] * dt).sum();
        let taken = p.snapshot(&gpu).unwrap().total_mass_lost();
        assert!(taken > 0.0, "step {step}: nothing evaporated, so the gate is vacuous");
        assert!(given > 0.0, "step {step}: the gas was given {given:e} kg of vapour");
        worst = worst.max((given - taken).abs() / taken);
        total += taken;
        // The reduction over cells is the same number the object reports.
        let reported = cp.total_vapour_mass(&gpu).unwrap();
        assert!(
            (reported - given).abs() <= 1e-15 * given,
            "step {step}: total_vapour_mass {reported:e} against the cell sum {given:e}"
        );
    }
    assert!(
        worst <= 1e-13,
        "worst relative mass defect {worst:e} over {total:e} kg evaporated"
    );
}

/// The deposit is exactly what S77.4 says: S68's convective exchange plus
/// the enthalpy the arriving mass carries, `cp_g mdot (T_p - T_g)`, and
/// **nothing else**.
///
/// This is the test that fixes the section's one real physics decision. The
/// obvious reading of "couple the latent heat back" is to deposit
/// `-q_lat` as a second sink beside `-Q_c`. It is wrong, and it is wrong by
/// a factor this test prints: the droplet's own budget is
/// `Q_c = C dT_p + dm h_v` (76.10), so **the convective heat the gas has
/// already given up contains every joule the phase change consumed**. A
/// second latent sink counts them twice.
#[test]
fn the_energy_deposit_is_the_convective_heat_plus_the_vapour_enthalpy() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([4, 4, 4], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let t_g: Scalar = 330.0;
    let (u, rho, t_gas, y_gas) = wet_gas(&gpu, &gm, t_g, 0.006, 1.05);
    let dt: Scalar = 2e-3;

    let mut p = Parcels::new(&gpu, &hm, &gm, wet_controls(), &[], dt).unwrap();
    p.seed(&gpu, &hm, &wet_seeds(80)).unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    let mut cp = ParcelCoupling::new(&gpu, &p, full_coupling()).unwrap();

    let vol = gpu.download(&gm.v).unwrap();
    let cp_gas = base_controls().cp_gas;
    let mut worst: Scalar = 0.0;
    let mut ratio: Scalar = 0.0;
    for _ in 0..4 {
        one_wet_step(&gpu, &mut p, &mut dep, &mut cp, &u, &rho, &t_gas, &y_gas, dt);
        let s = cp.snapshot(&gpu).unwrap();
        let ps = p.snapshot(&gpu).unwrap();

        let given: Scalar = (0..gm.n_cells).map(|c| vol[c] * s.energy_q[c] * dt).sum();
        let conv = live_parcel_heat(&ps);
        let vap = live_parcel_vapour_enthalpy(&ps, cp_gas, t_g);
        // -Q_c + cp mdot (T_p - T_g), and both halves are real: the droplets
        // are colder than the gas so `vap` is a further cooling.
        let want = -conv + vap;
        worst = worst.max((given - want).abs() / want.abs());
        assert!(conv > 0.0 && vap < 0.0, "conv {conv:e} vap {vap:e}");

        // What the WRONG deposit would have been: a second latent sink.
        let latent: Scalar = ps.live().iter().map(|&i| ps.n_p[i] * ps.latent[i]).sum();
        ratio = ratio.max(latent / conv);
    }
    assert!(worst <= 1e-13, "worst relative deposit error {worst:e}");
    // The latent heat is a large fraction of the convective heat - 0.398
    // measured here, where the droplets are still warming towards their wet
    // bulb and so keep some of the heat, and approaching one once they are
    // on it. Depositing it a second time would cool the gas by that much
    // again. The number is asserted so that a future change which quietly
    // adds it back fails here with the size of its own mistake printed.
    assert!(
        ratio > 0.3 && ratio < 1.05,
        "the latent heat is {ratio:.4} of the convective heat; a second sink would \
         have cooled the gas by that much again"
    );
}

/// **Gate 77-B.** The energy ledger, closed to the accuracy of the droplet's
/// own budget (S76.12 row 7's `4.8e-12`, and measured at `1.3e-12` here):
///
/// ```text
///   dE_gas + dE_liquid + E_vapour = 0
/// ```
///
/// with every pool referred to absolute zero, which is the reference S26's
/// `rho cp T` actually carries.
///
/// Two things in it are worth stating because both are traps.
///
/// **1. The registry integral is NOT the gas's energy change.** (77.2)
/// deposits the NON-conservative source `cp mdot (T_p - T_g)`, because
/// S26's equation is `rho cp DT/Dt = Q`. The gas also gains the mass itself,
/// so the conservative change is `integral(q) dt + cp T_g dm`. Forgetting
/// that second term leaves the ledger short by about 12 % of the latent
/// heat - which is large, and which is exactly the size of error a
/// tolerance-shaped gate would have shrugged at.
///
/// **2. `E_vapour` is not `h_v dm`.** It is [`live_parcel_vapour_energy`],
/// the latent heat plus `dm (c_l - cp_g) T_p`, and the second term is the
/// offset between two sensible pools whose enthalpy data are `c_l T` and
/// `cp_g T`. S77.11 says what that means physically.
#[test]
fn gate_77b_the_energy_ledger_closes_across_the_phase_change() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([4, 4, 4], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let t_g: Scalar = 350.0;
    let (u, rho, t_gas, y_gas) = wet_gas(&gpu, &gm, t_g, 0.003, 1.0);
    let dt: Scalar = 2e-3;

    let ctrl = wet_controls();
    let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[], dt).unwrap();
    p.seed(&gpu, &hm, &wet_seeds(96)).unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    let mut cp = ParcelCoupling::new(&gpu, &p, full_coupling()).unwrap();

    let vol = gpu.download(&gm.v).unwrap();
    let (rl, cl, cg) = (ctrl.rho_liquid, ctrl.c_liquid, ctrl.cp_gas);
    let mut before = live_parcel_liquid_energy(&p.snapshot(&gpu).unwrap(), rl, cl);
    let mut worst: Scalar = 0.0;
    let mut worst_naive: Scalar = 0.0;
    for _ in 0..5 {
        one_wet_step(&gpu, &mut p, &mut dep, &mut cp, &u, &rho, &t_gas, &y_gas, dt);
        let s = cp.snapshot(&gpu).unwrap();
        let ps = p.snapshot(&gpu).unwrap();

        let registry: Scalar = (0..gm.n_cells).map(|c| vol[c] * s.energy_q[c] * dt).sum();
        let dm: Scalar = (0..gm.n_cells).map(|c| vol[c] * s.vapour[c] * dt).sum();
        let d_gas = registry + cg * t_g * dm;

        let after = live_parcel_liquid_energy(&ps, rl, cl);
        let d_liq = after - before;
        let e_vap = live_parcel_vapour_energy(&ps, cl, cg);
        before = after;

        let scale = d_liq.abs().max(e_vap.abs()).max(d_gas.abs());
        worst = worst.max((d_gas + d_liq + e_vap).abs() / scale);
        // And the ledger a reader who integrates the registry alone would
        // have written, which is the trap this test names.
        worst_naive = worst_naive.max((registry + d_liq + e_vap).abs() / scale);
    }
    // NOT round-off, and the difference matters: the two accumulators are
    // endpoint differences while the droplet's budget is a sum over
    // sub-steps whose mass changes between them (S76.10). The ledger
    // inherits that gap exactly and nothing else.
    assert!(worst <= 1e-11, "worst relative energy defect {worst:e}");
    assert!(
        worst_naive > 0.05,
        "the naive ledger closed too ({worst_naive:e}); either the fixture stopped \
         transferring mass or the deposit stopped being the non-conservative one"
    );
}

/// **Gate 77-C.** The divergence source is the volume the added mass
/// occupies, `mdot/rho`, cell by cell - and the species source is that same
/// number diluted by `1 - Y_v`.
///
/// Both are exact algebra on the deposit, so the tolerance is round-off and
/// what the test is really checking is that no factor was dropped: a `V`,
/// an `n_p`, a `dt`, or the density that turns a mass rate into a volume
/// rate. Getting `dSrc = mdot` instead of `mdot/rho` is off by three orders
/// of magnitude and dimensionally wrong, and nothing downstream could tell.
#[test]
fn gate_77c_the_divergence_and_species_sources_are_the_deposit_divided_through() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([4, 4, 4], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (rho_g, y_v): (Scalar, Scalar) = (0.98, 0.012);
    let (u, rho, t_gas, y_gas) = wet_gas(&gpu, &gm, 345.0, y_v, rho_g);
    let dt: Scalar = 2e-3;

    let mut p = Parcels::new(&gpu, &hm, &gm, wet_controls(), &[], dt).unwrap();
    p.seed(&gpu, &hm, &wet_seeds(64)).unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    let mut cp = ParcelCoupling::new(&gpu, &p, full_coupling()).unwrap();

    one_wet_step(&gpu, &mut p, &mut dep, &mut cp, &u, &rho, &t_gas, &y_gas, dt);
    let s = cp.snapshot(&gpu).unwrap();
    let mut occupied = 0;
    for c in 0..gm.n_cells {
        let want_d = s.vapour[c] / rho_g;
        let want_y = want_d * (1.0 - y_v);
        assert!(
            (s.divergence[c] - want_d).abs() <= 1e-15 * want_d.abs().max(1e-30),
            "cell {c}: dSrc {} against mdot/rho {want_d}",
            s.divergence[c]
        );
        assert!(
            (s.species_su[c] - want_y).abs() <= 1e-15 * want_y.abs().max(1e-30),
            "cell {c}: ySu {} against mdot(1-Y)/rho {want_y}",
            s.species_su[c]
        );
        if s.vapour[c] > 0.0 {
            occupied += 1;
        }
    }
    assert!(occupied > 4, "only {occupied} cells held vapour; the gate is thin");

    // The dilution factor BITES: 1.2 % of the source is small but it is not
    // zero, and S13.4.1 wants the pair rather than the assertion.
    let bare: Scalar = (0..gm.n_cells).map(|c| s.divergence[c]).sum();
    let diluted: Scalar = (0..gm.n_cells).map(|c| s.species_su[c]).sum();
    let moved = (bare - diluted) / bare;
    assert!(
        moved > 0.011 && moved < 0.013,
        "the dilution factor moved the species source by {moved}"
    );
}

/// **Gate 77-E, the by-construction half.** With mass coupling off nothing
/// is registered on the species equation and `None` is passed for the
/// divergence, so the arithmetic is the arithmetic it always was. This half
/// is a property of the source and cannot be measured; what CAN be measured
/// is that the four S77 arrays are then not even allocated at mesh size, so
/// there is nothing for a stray write to land in.
///
/// **The measured half** is here: a pool that has never held a parcel,
/// coupled and registered with mass coupling ON, deposits exactly zero into
/// all four arrays.
#[test]
fn gate_77e_an_empty_pool_couples_exactly_zero_vapour() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([4, 4, 4], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (u, rho, t_gas, y_gas) = wet_gas(&gpu, &gm, 330.0, 0.005, 1.1);
    let dt: Scalar = 1e-3;

    let mut p = Parcels::new(&gpu, &hm, &gm, wet_controls(), &[], dt).unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    let mut cp = ParcelCoupling::new(&gpu, &p, full_coupling()).unwrap();
    one_wet_step(&gpu, &mut p, &mut dep, &mut cp, &u, &rho, &t_gas, &y_gas, dt);

    let s = cp.snapshot(&gpu).unwrap();
    let zeros = vec![0.0 as Scalar; gm.n_cells];
    assert!(bits_eq(&s.vapour, &zeros), "an empty pool made vapour");
    assert!(bits_eq(&s.species_su, &zeros), "an empty pool sourced Y_v");
    assert!(bits_eq(&s.divergence, &zeros), "an empty pool expanded the gas");
    assert!(bits_eq(&s.vapour_enthalpy, &zeros), "an empty pool carried enthalpy");
    assert_eq!(cp.total_vapour_mass(&gpu).unwrap(), 0.0);

    // And the four arrays are the only thing S77 costs, so the memory report
    // moves by exactly them.
    let dry = ParcelCoupling::new(
        &gpu,
        &p,
        CouplingControls {
            momentum: CouplingMode::Explicit,
            energy: CouplingMode::Off,
            mass: MassCoupling::None,
        },
    )
    .unwrap();
    assert_eq!(cp.device_bytes() - dry.device_bytes(), gm.n_cells * 4 * 8);
    assert_eq!(dry.device_bytes(), gm.n_cells * 15 * 8);
}

/// **Gate 77-E, end to end.** S77's deposits are `+0.0` for an empty pool
/// (above); this is the claim a case actually cares about, which is one
/// consumer further on: an empty pool with `mass evaporation` ON leaves the
/// GAS ANSWER - the solved vapour field and the target divergence - bit for
/// bit where a run with no coupling at all left it.
///
/// The two are not the same statement. "The deposit is zero" is about this
/// module; "the answer is unmoved" is about `fvm_su` and `energyAccumulate`,
/// and it holds because the deposit is `+0.0` and `+0.0` is the ADDITIVE
/// IDENTITY - the same argument S68.3 makes for the momentum deposit, one
/// section over and now measured through the two S77 seams rather than
/// asserted about them.
///
/// It is what a reader of S77 wants to know before turning the setting on:
/// every case in this repository without a spray is unmoved, and not
/// "within tolerance" - `to_bits()`.
#[test]
fn gate_77e_a_coupled_empty_pool_leaves_the_gas_bitwise_where_it_was() {
    use crate::energy::{DomainKind, Energy, EnergyControls, GasProperties, GasState};
    use crate::field::{BcKind, GpuScalarField};
    use crate::io::case::TurbulenceControls as TCtrl;
    use crate::scalar_transport::{ScalarTransport, ScalarTransportCoeffs};

    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([3, 3, 3], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let dt: Scalar = 1e-3;
    let t_g: Scalar = 335.0;
    let (u, rho, t_gas, y_gas) = wet_gas(&gpu, &gm, t_g, 0.006, 1.05);

    // An evaporating pool with the whole of S77 switched on, and NOT ONE
    // PARCEL in it. Every S77 array is `n_cells` long and every entry of it
    // is written by the kernel this step.
    let mut p = Parcels::new(&gpu, &hm, &gm, wet_controls(), &[], dt).unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    let mut cp = ParcelCoupling::new(&gpu, &p, full_coupling()).unwrap();
    one_wet_step(&gpu, &mut p, &mut dep, &mut cp, &u, &rho, &t_gas, &y_gas, dt);

    // ---- 1. the species field ----------------------------------------
    //
    // (77.1) handed to a `ScalarTransport` called `Yv` - S54's own humidity
    // object - against the same object solved with no source at all.
    let ctrl = TCtrl {
        k_solver: crate::io::case::SolverControls {
            tolerance: 1e-15,
            rel_tol: 0.0,
            max_iter: 500,
            check_interval: 1,
            ..crate::io::case::SolverControls::default()
        },
        k_relax: 1.0,
        steady: false,
        delta_t: dt,
        sn_grad: crate::fv::SnGradScheme::Uncorrected,
        ..TCtrl::default()
    };
    let nut = GpuScalarField::zeros(&gpu, &gm, "nut").unwrap();
    let phi = crate::field::GpuSurfaceScalarField::zeros(&gpu, &gm, "phi").unwrap();
    let uz = GpuVectorField::zeros(&gpu, &gm, "U").unwrap();
    let flow = crate::turbulence::FlowState::new(&uz, &phi, 1.5e-5);
    // A LUMPY initial field: a uniform one would be unmoved by a wrong
    // source that happened to be uniform too.
    let y0: Vec<Scalar> = (0..gm.n_cells)
        .map(|c| 0.004 + 0.002 * ((c % 7) as Scalar) / 7.0)
        .collect();
    let solve = |su: Option<&DevBuf<Scalar>>| -> Vec<Scalar> {
        let mut yv = ScalarTransport::new(
            &gpu,
            &hm,
            &gm,
            "Yv",
            ScalarTransportCoeffs { pr: 0.6, prt: 0.7 },
            ctrl,
        )
        .unwrap();
        {
            let f = yv.field_mut();
            gpu.write(&mut f.f, &y0).unwrap();
        }
        yv.initialise(&gpu).unwrap();
        yv.correct_with_source(&gpu, &flow, &nut, su).unwrap();
        gpu.download(&yv.field().f).unwrap()
    };
    let bare = solve(None);
    let coupled = solve(Some(cp.vapour_source()));
    assert!(
        bits_eq(&bare, &coupled),
        "an empty pool's `+0.0` vapour source moved the solved Y_v"
    );
    // And the solve is not a no-op it would be trivially unmoved by: the
    // equation transported the lumpy field it was given.
    assert!(
        !bits_eq(&bare, &y0),
        "the species equation did not move the field at all, so the gate is vacuous"
    );

    // ---- 2. the target divergence ------------------------------------
    //
    // Both S77 halves at once: the energy deposit registered on
    // `EnergySources` (which is how the phase change's ENERGY reaches
    // `(div u)_target`, S77.6) and (77.3) passed as `d_mass` (which is how
    // its VOLUME does). Neither may move a bit when the pool is empty.
    let props = GasProperties { k: 0.026, cp: 1005.0, ..GasProperties::default() };
    let ectrl = EnergyControls { steady: true, delta_t: 1.0, ..EnergyControls::default() };
    let div = |register: bool| -> Vec<Scalar> {
        let mut e = Energy::new(&gpu, &gm, ectrl, props).unwrap();
        {
            let f = e.field_mut();
            let lumpy: Vec<Scalar> = (0..gm.n_cells)
                .map(|c| 320.0 + 8.0 * ((c % 5) as Scalar))
                .collect();
            gpu.write(&mut f.f, &lumpy).unwrap();
            gpu.write(
                &mut f.bc_kind,
                &vec![BcKind::ZeroGradient as crate::Label; hm.n_boundary_faces],
            )
            .unwrap();
            gpu.write(&mut f.fr, &vec![0.0 as Scalar; hm.n_boundary_faces]).unwrap();
            gpu.write(&mut f.ref_value, &vec![0.0 as Scalar; hm.n_boundary_faces])
                .unwrap();
            gpu.write(&mut f.ref_grad, &vec![0.0 as Scalar; hm.n_boundary_faces])
                .unwrap();
        }
        e.initialise(&gpu).unwrap();
        let mut gas = GasState::new(&gpu, &gm, props, DomainKind::Open, 101_325.0).unwrap();
        gas.update_density(&gpu, e.field()).unwrap();
        let k_cell = gpu.zeros::<Scalar>(gm.n_cells.max(1)).unwrap();
        let d = if register {
            cp.register_energy(&gpu, e.sources_mut()).unwrap();
            Some(cp.divergence_source())
        } else {
            None
        };
        e.update_target_divergence_with(&gpu, &gas, &nut, &k_cell, 1.5e-5, d)
            .unwrap();
        gpu.download(e.target_divergence()).unwrap()
    };
    let bare_div = div(false);
    let coupled_div = div(true);
    assert!(
        bits_eq(&bare_div, &coupled_div),
        "an empty pool's `+0.0` deposits moved (div u)_target"
    );
    // And the fixture is not vacuous: the divergence it is unmoved AT is a
    // real, non-uniform field, so "bitwise unmoved" is not "both are zero".
    assert!(
        bare_div.iter().any(|&d| d != 0.0),
        "the target divergence was zero everywhere, so the gate is vacuous"
    );
}

/// S13.4.1's pair for `coupling/mass`: the same pool, the same step, the two
/// values - and the ENERGY the gas is handed differs, bitwise, because the
/// vapour's enthalpy rides the same registry field.
///
/// This is what makes `mass evaporation` a setting rather than a label: not
/// only does the vapour appear, the temperature source changes with it.
#[test]
fn the_mass_coupling_mode_changes_the_answer() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([3, 3, 3], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (u, rho, t_gas, y_gas) = wet_gas(&gpu, &gm, 340.0, 0.004, 1.0);
    let dt: Scalar = 2e-3;

    let run = |mass: MassCoupling| {
        let mut p = Parcels::new(&gpu, &hm, &gm, wet_controls(), &[], dt).unwrap();
        p.seed(&gpu, &hm, &wet_seeds(48)).unwrap();
        let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
        // `energy off` is the only legal partner for `mass none` on an
        // evaporating pool (S77.7), so the pair is exercised at the two
        // points the contract allows and not at an illegal third.
        let energy = if mass.is_on() { CouplingMode::Explicit } else { CouplingMode::Off };
        let mut cp = ParcelCoupling::new(
            &gpu,
            &p,
            CouplingControls { momentum: CouplingMode::Explicit, energy, mass },
        )
        .unwrap();
        let (t, y) = if mass.is_on() {
            (Some(&t_gas), Some(&y_gas))
        } else {
            (None, None)
        };
        p.step(&gpu, &u, &rho, Some(&t_gas), Some(&y_gas), dt).unwrap();
        dep.update(&gpu, &p).unwrap();
        cp.update(&gpu, &p, &dep, &rho, &u, t, y, dt).unwrap();
        (cp.snapshot(&gpu).unwrap(), p.snapshot(&gpu).unwrap())
    };

    let (off, poff) = run(MassCoupling::None);
    let (on, pon) = run(MassCoupling::Evaporation);

    // The PARCELS are bit for bit the same in both: what the gas is told
    // does not reach back into the droplet within a step, which is the
    // read-only-gas property S76 was scoped around and S77 does not break.
    assert!(bits_eq(&poff.d, &pon.d), "the mass coupling moved the droplets");
    assert!(
        bits_eq(&poff.temperature, &pon.temperature),
        "the mass coupling moved the droplet temperature"
    );
    assert!(bits_eq(&poff.mass_lost, &pon.mass_lost));

    // The GAS is not.
    assert!(
        !bits_eq(&off.energy_q, &on.energy_q),
        "the mass coupling left the energy source alone"
    );
    let vap: Scalar = on.vapour.iter().sum();
    assert!(vap > 0.0, "nothing evaporated, so the pair is vacuous");
    assert_eq!(off.vapour.len(), 1, "`mass none` allocated the S77 arrays");
}

/// S77.7's contract, in every direction it has. Six refusals, each by name,
/// each naming the setting that would fix it.
#[test]
fn the_half_coupled_evaporating_pool_is_refused_by_name() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([2, 2, 2], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let dt: Scalar = 1e-3;
    let wet = Parcels::new(&gpu, &hm, &gm, wet_controls(), &[], dt).unwrap();
    let hot = Parcels::new(
        &gpu,
        &hm,
        &gm,
        ParcelControls { physics: ParcelPhysics::Heating, ..base_controls() },
        &[],
        dt,
    )
    .unwrap();

    // 1. mass coupling on a pool that cannot evaporate.
    let e = ParcelCoupling::new(
        &gpu,
        &hot,
        CouplingControls {
            momentum: CouplingMode::Explicit,
            energy: CouplingMode::Explicit,
            mass: MassCoupling::Evaporation,
        },
    )
    .err()
    .map(|e| e.to_string())
    .unwrap_or_default();
    assert!(e.contains("no vapour to give"), "{e}");
    assert!(e.contains("evaporating"), "{e}");

    // 2. the vapour without the heat.
    let e = ParcelCoupling::new(
        &gpu,
        &wet,
        CouplingControls {
            momentum: CouplingMode::Explicit,
            energy: CouplingMode::Off,
            mass: MassCoupling::Evaporation,
        },
    )
    .err()
    .map(|e| e.to_string())
    .unwrap_or_default();
    assert!(e.contains("humidifies without cooling"), "{e}");

    // 3. the heat without the vapour - S76.14's refusal, with its reason
    //    rewritten now that the other half exists.
    let e = ParcelCoupling::new(
        &gpu,
        &wet,
        CouplingControls {
            momentum: CouplingMode::Explicit,
            energy: CouplingMode::Explicit,
            mass: MassCoupling::None,
        },
    )
    .err()
    .map(|e| e.to_string())
    .unwrap_or_default();
    assert!(e.contains("mass evaporation"), "{e}");
    assert!(e.contains("S77"), "{e}");

    // 4. and the two legal combinations are legal.
    assert!(ParcelCoupling::new(&gpu, &wet, full_coupling()).is_ok());
    assert!(ParcelCoupling::new(
        &gpu,
        &wet,
        CouplingControls {
            momentum: CouplingMode::Explicit,
            energy: CouplingMode::Off,
            mass: MassCoupling::None,
        }
    )
    .is_ok());
    // 5. an inert pool is still refused energy, with S68's own reason.
    let cold = Parcels::new(&gpu, &hm, &gm, base_controls(), &[], dt).unwrap();
    let e = ParcelCoupling::new(
        &gpu,
        &cold,
        CouplingControls {
            momentum: CouplingMode::Off,
            energy: CouplingMode::Explicit,
            mass: MassCoupling::None,
        },
    )
    .err()
    .map(|e| e.to_string())
    .unwrap_or_default();
    assert!(e.contains("INFINITE heat bath"), "{e}");
    // 6. and the menu itself takes the new name and refuses a nonsense one.
    assert_eq!(
        MassCoupling::from_name("evaporation").unwrap(),
        MassCoupling::Evaporation
    );
    let e = MassCoupling::from_name("condensation").unwrap_err().to_string();
    assert!(e.contains("evaporation"), "{e}");
}

/// S13.4 on the vapour field, in both directions: `update` needs `Y_v`
/// exactly when (77.1) reads it.
#[test]
fn the_vapour_field_contract_is_refused_in_both_directions() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([2, 2, 2], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (u, rho, t_gas, y_gas) = wet_gas(&gpu, &gm, 330.0, 0.005, 1.1);
    let dt: Scalar = 1e-3;

    let mut p = Parcels::new(&gpu, &hm, &gm, wet_controls(), &[], dt).unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    p.step(&gpu, &u, &rho, Some(&t_gas), Some(&y_gas), dt).unwrap();
    dep.update(&gpu, &p).unwrap();

    let mut on = ParcelCoupling::new(&gpu, &p, full_coupling()).unwrap();
    let e = on
        .update(&gpu, &p, &dep, &rho, &u, Some(&t_gas), None, dt)
        .unwrap_err()
        .to_string();
    assert!(e.contains("dilution"), "{e}");

    let mut off = ParcelCoupling::new(
        &gpu,
        &p,
        CouplingControls {
            momentum: CouplingMode::Explicit,
            energy: CouplingMode::Off,
            mass: MassCoupling::None,
        },
    )
    .unwrap();
    let e = off
        .update(&gpu, &p, &dep, &rho, &u, None, Some(&y_gas), dt)
        .unwrap_err()
        .to_string();
    assert!(e.contains("read and ignored"), "{e}");

    // A short field is refused too, and named.
    let short = gpu.upload(&vec![0.0 as Scalar; 1]).unwrap();
    let e = on
        .update(&gpu, &p, &dep, &rho, &u, Some(&t_gas), Some(&short), dt)
        .unwrap_err()
        .to_string();
    assert!(e.contains("vapour mass fraction has 1 cells"), "{e}");
}

/// A droplet in SUPERSATURATED air grows (S76.12 row 11), and every sign in
/// S77 reverses with it: the gas loses vapour, the gas is WARMED by the
/// condensation, and the divergence source turns negative - the mixture
/// contracts because mass is leaving it.
///
/// This is also why neither S77 source can be made an implicit sink: with
/// `mdot` free to change sign, `-mdot cp` on the diagonal would sometimes be
/// positive, and S68's Patankar guarantee is "by construction, with no
/// clamp".
#[test]
fn condensation_reverses_every_sign_and_is_why_the_sources_are_explicit() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([2, 2, 2], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    // 20 C air holding far more vapour than it can: Y_s(293 K) is about
    // 0.0146, so 0.05 is heavily supersaturated and the droplets grow.
    let (u, rho, t_gas, y_gas) = wet_gas(&gpu, &gm, 293.15, 0.05, 1.2);
    let dt: Scalar = 1e-3;

    let mut p = Parcels::new(&gpu, &hm, &gm, wet_controls(), &[], dt).unwrap();
    p.seed(
        &gpu,
        &hm,
        &[SeedParcel {
            position: Vec3::new(0.5, 0.5, 0.5),
            velocity: Vec3::ZERO,
            diameter: 2e-4,
            temperature: 293.15,
            n_p: 1e5,
            uid: Some(1),
        }],
    )
    .unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    let mut cp = ParcelCoupling::new(&gpu, &p, full_coupling()).unwrap();

    let d0 = p.snapshot(&gpu).unwrap().d[0];
    for _ in 0..5 {
        one_wet_step(&gpu, &mut p, &mut dep, &mut cp, &u, &rho, &t_gas, &y_gas, dt);
    }
    let s = cp.snapshot(&gpu).unwrap();
    let ps = p.snapshot(&gpu).unwrap();
    assert!(ps.d[0] > d0, "the droplet did not grow: {} against {d0}", ps.d[0]);

    let vap: Scalar = s.vapour.iter().sum();
    let div: Scalar = s.divergence.iter().sum();
    let ysu: Scalar = s.species_su.iter().sum();
    assert!(vap < 0.0, "condensing droplets still made vapour: {vap:e}");
    assert!(div < 0.0, "condensing droplets still expanded the gas: {div:e}");
    assert!(ysu < 0.0, "condensing droplets still sourced Y_v: {ysu:e}");
    // And the sign of the implicit coefficient a semi-implicit split WOULD
    // have wanted is the wrong one, which is the whole argument.
    assert!(
        -vap > 0.0,
        "with mdot < 0 the sink coefficient -mdot cp is POSITIVE, which is what \
         S77.5 refuses to put on a diagonal"
    );
}

/// **Gate 77-D, the cheap half.** A sealed adiabatic box of air with water
/// sprayed into it moves along the ADIABATIC SATURATION LINE: the gas cools
/// and humidifies together, and ASHRAE's wet-bulb temperature of the mixture
/// - which is the adiabatic-saturation temperature - stays put while it
/// does. That is the defining property of the process, and it is a property
/// of the whole trajectory rather than of one endpoint, so it is worth more
/// than the endpoint alone.
///
/// The gas is advanced ON THE HOST from the three S77 deposits, which is the
/// point: this is the first test in this crate where a cell gets wetter,
/// and it is what S76.14 said the read-only gas could not do. The parcels
/// still see one frozen state per step, so the order-independence of S76
/// survives - what changed is that the state is refreshed BETWEEN steps.
///
/// The drift is the model gap, not the integrator's error, and S77.11 names
/// it: ASHRAE's relation carries `1.006 + 1.86 W` kJ/(kg K) for the moist
/// mixture where S26's energy equation has one constant `cp`.
#[test]
fn the_gas_moves_along_the_adiabatic_saturation_line() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([2, 2, 2], [0.5, 0.5, 0.5], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let dt: Scalar = 5e-4;
    let p_atm: Scalar = 101_325.0;

    let ctrl = ParcelControls { capacity: 8, ..wet_controls() };
    let cp_gas = ctrl.cp_gas;

    // 40 C at 12 % relative humidity - dry enough that the depression is
    // large and the drift, if there is one, has room to show.
    let mut t_g: Scalar = 313.15;
    let mut y_v: Scalar = crate::psychro::yv_from_t_rh_p(t_g, 0.12, p_atm);
    let w0 = crate::psychro::w_from_yv(y_v);
    let t_star = crate::psychro::t_wb(t_g, w0, p_atm).unwrap();

    let vol = gpu.download(&gm.v).unwrap();
    let v_box: Scalar = vol.iter().sum();
    let mut rho_g: Scalar = 1.15;
    let mut m_gas = rho_g * v_box;

    let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[], dt).unwrap();
    // 50 um droplets released AT the adiabatic-saturation temperature, which
    // is where a droplet on this line sits: they should barely change
    // temperature while the gas comes to meet them.
    p.seed(
        &gpu,
        &hm,
        &[SeedParcel {
            position: Vec3::new(0.25, 0.25, 0.25),
            velocity: Vec3::ZERO,
            diameter: 5e-5,
            temperature: t_star + 273.15,
            n_p: 6.0e7,
            uid: Some(1),
        }],
    )
    .unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    let mut cp = ParcelCoupling::new(&gpu, &p, full_coupling()).unwrap();

    let mut u = GpuVectorField::zeros(&gpu, &gm, "U").unwrap();
    u.f = gpu.upload(&vec![Vec3::ZERO; gm.n_cells]).unwrap();
    let mut rho = gpu.upload(&vec![rho_g; gm.n_cells]).unwrap();
    let mut t_f = gpu.upload(&vec![t_g; gm.n_cells]).unwrap();
    let mut y_f = gpu.upload(&vec![y_v; gm.n_cells]).unwrap();

    let t0 = t_g;
    let mut drift: Scalar = 0.0;
    for _ in 0..400 {
        p.step(&gpu, &u, &rho, Some(&t_f), Some(&y_f), dt).unwrap();
        dep.update(&gpu, &p).unwrap();
        cp.update(&gpu, &p, &dep, &rho, &u, Some(&t_f), Some(&y_f), dt)
            .unwrap();

        let s = cp.snapshot(&gpu).unwrap();
        let dm: Scalar = (0..gm.n_cells).map(|c| vol[c] * s.vapour[c] * dt).sum();
        let dq: Scalar = (0..gm.n_cells).map(|c| vol[c] * s.energy_q[c] * dt).sum();

        // The CONSERVATIVE enthalpy change: the registry carries the
        // non-conservative source, so the mass the gas gained brings
        // `cp T_g dm` with it (gate 77-B's first trap).
        let h = m_gas * cp_gas * t_g + dq + cp_gas * t_g * dm;
        let m_new = m_gas + dm;
        y_v = (m_gas * y_v + dm) / m_new;
        t_g = h / (m_new * cp_gas);
        m_gas = m_new;
        rho_g = m_gas / v_box;

        gpu.write(&mut rho, &vec![rho_g; gm.n_cells]).unwrap();
        gpu.write(&mut t_f, &vec![t_g; gm.n_cells]).unwrap();
        gpu.write(&mut y_f, &vec![y_v; gm.n_cells]).unwrap();

        let w = crate::psychro::w_from_yv(y_v);
        let t_wb = crate::psychro::t_wb(t_g, w, p_atm).unwrap();
        drift = drift.max((t_wb - t_star).abs());
    }

    let w = crate::psychro::w_from_yv(y_v);
    assert!(t_g < t0 - 2.0, "the gas did not cool: {t_g} from {t0}");
    assert!(w > 1.5 * w0, "the gas did not humidify: {w} from {w0}");
    assert!(
        crate::psychro::rh_from_t_w_p(t_g, w, p_atm) > 0.4,
        "the box did not get near saturation, so the drift bound is thin"
    );
    // The drift is the property-data gap, and 0.5 K is the bar S76.13 set
    // for the same comparison against the same reference.
    assert!(
        drift < 0.5,
        "ASHRAE's adiabatic-saturation temperature drifted {drift:.4} K along the \
         process; it should be an invariant of it"
    );
}

/// The species seam, end to end: (77.1)'s source handed to the very object
/// S54's humidity is - a `ScalarTransport` called `Yv` - and the vapour
/// arriving in the field.
///
/// The fixture is deliberately UNIFORM: one identical droplet at every cell
/// centre of a 2x2x2 block, adiabatic walls, no flow. Then `Y_v` stays
/// uniform, the convection and diffusion terms are identically zero, and
/// Euler makes the step exact - `Y_v' = Y_v + S dt` with no solver tolerance
/// in the way. Anything that goes wrong here is the SEAM and not the
/// equation: a missing `V`, a source applied after relaxation, or the
/// registry being read at a different iteration from the one it was written
/// in.
#[test]
fn the_vapour_source_reaches_the_species_field_it_is_handed_to() {
    use crate::io::case::TurbulenceControls as TCtrl;
    use crate::scalar_transport::{ScalarTransport, ScalarTransportCoeffs};

    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block([2, 2, 2], [1.0, 1.0, 1.0], ["wall"; 6]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let dt: Scalar = 1e-3;
    let (rho_g, y0): (Scalar, Scalar) = (1.0, 0.005);
    let (u, rho, t_gas, _) = wet_gas(&gpu, &gm, 340.0, y0, rho_g);

    let mut p = Parcels::new(&gpu, &hm, &gm, wet_controls(), &[], dt).unwrap();
    let mut seeds = Vec::new();
    let mut uid = 1u64;
    for k in 0..2 {
        for j in 0..2 {
            for i in 0..2 {
                seeds.push(SeedParcel {
                    position: Vec3::new(
                        0.25 + 0.5 * i as Scalar,
                        0.25 + 0.5 * j as Scalar,
                        0.25 + 0.5 * k as Scalar,
                    ),
                    velocity: Vec3::ZERO,
                    diameter: 1.5e-4,
                    temperature: 300.0,
                    n_p: 5.0e4,
                    uid: Some(uid),
                });
                uid += 1;
            }
        }
    }
    p.seed(&gpu, &hm, &seeds).unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    let mut cp = ParcelCoupling::new(&gpu, &p, full_coupling()).unwrap();

    let ctrl = TCtrl {
        k_solver: crate::io::case::SolverControls {
            tolerance: 1e-15,
            rel_tol: 0.0,
            max_iter: 500,
            check_interval: 1,
            ..crate::io::case::SolverControls::default()
        },
        k_relax: 1.0,
        steady: false,
        delta_t: dt,
        sn_grad: crate::fv::SnGradScheme::Uncorrected,
        ..TCtrl::default()
    };
    let mut yv = ScalarTransport::new(
        &gpu,
        &hm,
        &gm,
        "Yv",
        ScalarTransportCoeffs { pr: 0.6, prt: 0.7 },
        ctrl,
    )
    .unwrap();
    {
        let f = yv.field_mut();
        gpu.write(&mut f.f, &vec![y0; gm.n_cells]).unwrap();
    }
    yv.initialise(&gpu).unwrap();

    let nut = crate::field::GpuScalarField::zeros(&gpu, &gm, "nut").unwrap();
    let phi = crate::field::GpuSurfaceScalarField::zeros(&gpu, &gm, "phi").unwrap();
    let uz = crate::field::GpuVectorField::zeros(&gpu, &gm, "U").unwrap();
    let flow = crate::turbulence::FlowState::new(&uz, &phi, 1.5e-5);

    let vol = gpu.download(&gm.v).unwrap();
    let mut y_now = y0;
    let mut worst: Scalar = 0.0;
    for step in 0..4 {
        let y_field = gpu.upload(&vec![y_now; gm.n_cells]).unwrap();
        p.step(&gpu, &u, &rho, Some(&t_gas), Some(&y_field), dt).unwrap();
        dep.update(&gpu, &p).unwrap();
        cp.update(&gpu, &p, &dep, &rho, &u, Some(&t_gas), Some(&y_field), dt)
            .unwrap();

        let s = cp.snapshot(&gpu).unwrap();
        yv.correct_with_source(&gpu, &flow, &nut, Some(cp.vapour_source()))
            .unwrap();
        let got = gpu.download(&yv.field().f).unwrap();

        // Uniform source, uniform field, no flux: Euler is exact.
        let want = y_now + s.species_su[0] * dt;
        for c in 0..gm.n_cells {
            worst = worst.max((got[c] - want).abs() / want);
        }
        assert!(s.species_su[0] > 0.0, "step {step}: no vapour was sourced");
        y_now = got[0];
    }
    assert!(worst <= 1e-11, "worst relative Y_v error {worst:e}");
    assert!(y_now > y0, "Y_v did not rise: {y_now} from {y0}");

    // The vapour that arrived in the FIELD is the vapour the parcels lost,
    // divided through by the mixture the crate's unit-density species
    // equation carries. Not an identity - the equation is `DY/Dt = S` with
    // `rho` outside it - but it is the right order and the right sign, and
    // it is the check that catches a source handed to the wrong equation.
    let dm: Scalar = (0..gm.n_cells).map(|c| vol[c] * s_total(&cp, &gpu, c)).sum::<Scalar>() * dt;
    let v_dom: Scalar = vol.iter().sum();
    let want_dy = dm * (1.0 - y0) / (rho_g * v_dom);
    let got_dy = y_now - y0;
    assert!(
        (got_dy / (4.0 * want_dy) - 1.0).abs() < 0.35,
        "Y_v rose by {got_dy:e}, four steps of {want_dy:e} would have been {:e}",
        4.0 * want_dy
    );
}

/// `mdot'''` in one cell, read back - a one-line helper so the test above
/// reads as arithmetic rather than as three downloads.
fn s_total(cp: &ParcelCoupling<'_>, gpu: &Gpu, c: usize) -> Scalar {
    gpu.download(cp.vapour_production()).unwrap()[c]
}
