// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 147.1 - an injector can draw droplet diameters from a
//! Rosin-Rammler distribution, deterministically. The host closed forms
//! (147.1)-(147.3) are checked against each other; (147.4)'s counter-based
//! stratified draw is checked against the parcel identity it is keyed on;
//! the device draw is checked against the host twin in the same operation
//! order, the mass-weighted sample against (147.1) as a Kolmogorov-Smirnov
//! distance, the mass median against (147.2), the event mass against
//! `mdot dt stride` through (147.5), the same seed against itself bit for
//! bit, and every refusal by name. Every test runs in BOTH builds; an
//! integration test because the library's own test target does not compile
//! under `single`.
//!
//! Provenance: ORIGINAL - the meshes are this tree's own generators; no
//! GPL-licensed source was consulted.

use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::field::GpuVectorField;
use ofgpu::io::polymesh::build_host_mesh;
use ofgpu::parcels::inject::{event_mass_coefficient, mix64, stratified_u, stratum_offset};
use ofgpu::parcels::{
    parcel_uid, DragModel, Injector, ParcelControls, ParcelSnapshot, Parcels, RosinRammler,
    SizeDistribution, WallAction,
};
use ofgpu::{DevBuf, Gpu, GpuMesh, HostMesh, Scalar, Vec3};

/// The build's `Scalar` widened to the f64 the answers are compared in: a
/// no-op in the f64 build, the exact widening under `single`.
#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    f64::from(x)
}

/// `tests/f32_parcels.rs`'s `box_spec`: `[lo, hi]` axes, uniform, the given
/// patch types in `(xMin, xMax, yMin, yMax, zMin, zMax)` order.
fn box_spec(n: [usize; 3], lo: [Scalar; 3], hi: [Scalar; 3], types: [&str; 6]) -> BlockSpec {
    let axis = |lo: Scalar, hi: Scalar, n: usize| GradedAxis {
        lo,
        hi,
        n,
        expansion: 1.0,
        two_sided: false,
    };
    BlockSpec {
        x: axis(lo[0], hi[0], n[0]),
        y: axis(lo[1], hi[1], n[1]),
        z: axis(lo[2], hi[2], n[2]),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: types.map(String::from),
        cyclic: Vec::new(),
    }
}

/// A uniform box mesh built by `build_host_mesh` - origin-free in both builds.
fn box_mesh(n: [usize; 3], lo: [Scalar; 3], hi: [Scalar; 3]) -> HostMesh {
    build_host_mesh(&blockgen::raw_mesh(&box_spec(n, lo, hi, ["wall"; 6])).expect("raw"))
        .expect("box mesh")
}

/// The controls every test here runs: ballistic parcels in still gas, a
/// perfectly elastic wall - the verification setting, nothing else.
fn controls(capacity: usize) -> ParcelControls {
    ParcelControls {
        capacity,
        drag: DragModel::None,
        gravity: Vec3::ZERO,
        wall: WallAction::Rebound,
        restitution: 1.0,
        ..Default::default()
    }
}

/// Still gas of uniform density - the state every test here is posed in.
fn still_gas(gpu: &Gpu, m: &GpuMesh, rho: Scalar) -> (GpuVectorField, DevBuf<Scalar>) {
    (
        GpuVectorField::zeros(gpu, m, "U").expect("U"),
        gpu.upload(&vec![rho; m.n_cells]).expect("rho"),
    )
}

/// The pool's liquid density: `controls` leaves [`ParcelControls`]'s
/// default standing, and (147.5) divides the flow rate by exactly it.
fn liquid_density() -> Scalar {
    ParcelControls::default().rho_liquid
}

/// Setup S's injector: centre of the box, the default axis, one liquid
/// kilogramme per cubic metre scale of flow - only the size distribution
/// and the event shape move between tests.
fn rr_injector(per_event: u32, interval: Scalar, size: SizeDistribution) -> Injector {
    Injector {
        position: Vec3::new(0.5, 0.5, 0.5),
        speed: 1.0,
        mass_flow: 1e-3,
        parcels_per_event: per_event,
        interval,
        size,
        ..Default::default()
    }
}

/// The 147.1 draw every test here uses: X = 0.1 mm, q = 2.5, seed 7.
fn the_rr() -> RosinRammler {
    RosinRammler {
        x: 1e-4,
        q: 2.5,
        seed: 7,
    }
}

/// Run a pool in setup S for `steps` steps and hand back the snapshot.
/// One step of `interval = 0` fires every step, so the event count IS the
/// step count.
fn run_case(
    gpu: &Gpu,
    hm: &HostMesh,
    gm: &GpuMesh,
    capacity: usize,
    inj: Injector,
    steps: usize,
) -> ParcelSnapshot {
    let (u, rho) = still_gas(gpu, gm, 1.2);
    let dt: Scalar = 1e-3;
    let mut p = Parcels::new(gpu, hm, gm, controls(capacity), &[inj], dt).unwrap();
    for _ in 0..steps {
        p.step(gpu, &u, &rho, None, None, dt).unwrap();
    }
    p.stats(gpu).unwrap().check_capacity().unwrap();
    p.snapshot(gpu).unwrap()
}

/// The live parcels as `(d, n_p rho_l (pi/6) d^3)` rows in f64, sorted by
/// diameter - the sample the CDF and median tests both read.
fn mass_rows(s: &ParcelSnapshot, live: &[usize], rho_l: Scalar) -> Vec<(f64, f64)> {
    let mut rows: Vec<(f64, f64)> = live
        .iter()
        .map(|&i| {
            let d = w64(s.d[i]);
            (
                d,
                w64(s.n_p[i]) * w64(rho_l) * std::f64::consts::FRAC_PI_6 * d * d * d,
            )
        })
        .collect();
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    rows
}

/// The mass-weighted Kolmogorov-Smirnov distance of the sample against
/// (147.1): at each sorted diameter the cumulative mass is a step `F`, and
/// `D` is the largest gap to `Q` on either side of the step.
fn ks_distance(rows: &[(f64, f64)], rr: &RosinRammler) -> f64 {
    let total: f64 = rows.iter().map(|r| r.1).sum();
    let mut cum = 0.0;
    let mut dmax = 0.0f64;
    for (d, m) in rows {
        let f_below = cum / total;
        cum += *m;
        let f_at = cum / total;
        let q = rr.cdf(*d);
        dmax = dmax.max(f_at - q).max(q - f_below);
    }
    dmax
}

/// Neumaier's compensated sum - the mass total must measure (147.5)'s own
/// round-off, not the summation's.
fn exact_sum(vals: &[f64]) -> f64 {
    let mut s = 0.0;
    let mut c = 0.0;
    for &v in vals {
        let t = s + v;
        c += if s.abs() >= v.abs() {
            s - t + v
        } else {
            v - t + s
        };
        s = t;
    }
    s + c
}

// ==========================================================================
//  The closed forms, on the host (D/A)
// ==========================================================================

/// (147.1) and (147.3) are inverses of each other to 1e-14 on a sweep;
/// (147.2) is where (147.1) crosses 1/2, in closed form; (147.4)'s offset
/// lives strictly inside (0,1) and moves with the seed; and `mix64` here is
/// (66.9)'s mix bit for bit.
#[test]
fn rosin_rammler_closed_forms() {
    let rr = the_rr();
    for u in [1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0 - 1e-6] {
        let back = rr.cdf(rr.quantile(u));
        assert!(
            (back - u).abs() <= 1e-14,
            "Q(d({u})) = {back}, not {u} to 1e-14"
        );
    }
    assert!(
        (rr.cdf(rr.mass_median()) - 0.5).abs() <= 1e-14,
        "Q(MMD) is not 1/2 to 1e-14"
    );
    // The closed form at the build's OWN widened X: in the f64 build that is
    // the literal 1e-4, under `single` it is the f32 the device holds - the
    // same number `mass_median` reads.
    let x64 = w64(rr.x);
    let mmd = x64 * std::f64::consts::LN_2.powf(0.4);
    assert!(
        (rr.mass_median() - mmd).abs() <= 1e-14 * x64,
        "MMD {} differs from the closed form {mmd}",
        rr.mass_median()
    );
    for uid in [0u64, 1, u64::MAX] {
        let r = stratum_offset(uid, 7);
        assert!(
            r > 0.0 && r < 1.0,
            "uid {uid}: stratum offset {r} not in (0,1)"
        );
    }
    assert_ne!(
        stratum_offset(5, 7),
        stratum_offset(5, 8),
        "the seed does not move the offset"
    );
    assert_eq!(
        mix64((3u64 << 52) | (9u64 << 20) | 11),
        parcel_uid(3, 9, 11),
        "inject::mix64 is not (66.9)'s mix"
    );
}

// ==========================================================================
//  The sample, on the device, against the closed forms (P, R)
// ==========================================================================

/// The mass-weighted empirical CDF of ONE event of 100000 and of 100 events
/// of 1000, against (147.1), as a KS distance. Stratification puts each
/// parcel inside its own stratum of `u`, so `D` is bounded by 1/n of the
/// event - orders of magnitude inside the classical 1.36/sqrt(N) a random
/// sample would earn.
#[test]
fn the_sampled_cdf_is_rosin_rammler() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = box_mesh([10, 10, 10], [0.0; 3], [1.0; 3]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let rr = the_rr();
    // Q of the comparison: the same law, read at the build's own X and q
    // (RosinRammler::cdf widens them).
    let q_law = rr;
    let bound = 1.36 / (100000.0f64).sqrt();

    let one = run_case(
        &gpu,
        &hm,
        &gm,
        100000,
        rr_injector(100000, 0.0, SizeDistribution::RosinRammler(rr)),
        1,
    );
    let live = one.live();
    assert_eq!(live.len(), 100000, "the one-event pool is not full");
    let d = ks_distance(&mass_rows(&one, &live, liquid_density()), &q_law);
    println!("KS distance, one event of 100000: {d:.3e} (bound {bound:.3e})");
    assert!(d <= bound, "KS distance {d} above {bound}");

    let hundred = run_case(
        &gpu,
        &hm,
        &gm,
        100000,
        rr_injector(1000, 1e-3, SizeDistribution::RosinRammler(rr)),
        100,
    );
    let live = hundred.live();
    assert_eq!(live.len(), 100000, "the 100-event pool is not full");
    let d = ks_distance(&mass_rows(&hundred, &live, liquid_density()), &q_law);
    println!("KS distance, 100 events of 1000: {d:.3e} (bound {bound:.3e})");
    assert!(d <= bound, "KS distance {d} above {bound}");
}

/// The diameter at which the sorted cumulative mass first reaches half of
/// the total, against (147.2)'s closed form.
#[test]
fn the_mass_median_is_the_closed_form() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = box_mesh([10, 10, 10], [0.0; 3], [1.0; 3]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let rr = the_rr();
    let one = run_case(
        &gpu,
        &hm,
        &gm,
        100000,
        rr_injector(100000, 0.0, SizeDistribution::RosinRammler(rr)),
        1,
    );
    let live = one.live();
    let rows = mass_rows(&one, &live, liquid_density());
    let total: f64 = rows.iter().map(|r| r.1).sum();
    let mut cum = 0.0;
    let mut d50 = f64::NAN;
    for (d, m) in &rows {
        cum += *m;
        if cum >= 0.5 * total {
            d50 = *d;
            break;
        }
    }
    let mmd = rr.mass_median();
    println!("sample mass median {d50:.6e} m, (147.2) says {mmd:.6e} m");
    assert!(
        (d50 / mmd - 1.0).abs() <= 5e-3,
        "sample median {d50} m is not (147.2)'s {mmd} m to 0.5 %"
    );
}

/// Every live slot carries the identity (66.9) gives it, and its stored
/// diameter is the f64 host twin of (147.3)-(147.4) - the SAME operation
/// order in the kernel - to 1e-12 in f64, 1e-5 under `single` where the one
/// final rounding to f32 is the difference.
#[test]
fn the_device_draw_is_the_host_draw() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = box_mesh([10, 10, 10], [0.0; 3], [1.0; 3]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let rr = the_rr();
    let one = run_case(
        &gpu,
        &hm,
        &gm,
        100000,
        rr_injector(100000, 0.0, SizeDistribution::RosinRammler(rr)),
        1,
    );
    let live = one.live();
    assert_eq!(live.len(), 100000, "the pool is not full");
    let tol = if cfg!(feature = "single") {
        (1e-12 * 2f64.powi(29)).min(1e-5).max(1e-12)
    } else {
        1e-12
    };
    let mut worst = 0.0f64;
    for &i in &live {
        let uid = one.uid[i];
        assert_eq!(
            uid,
            parcel_uid(0, 0, i as u64),
            "slot {i} carries the wrong identity"
        );
        let host = rr.quantile(stratified_u(i as u32, 100000, uid, 7));
        let rel = (w64(one.d[i]) - host).abs() / host;
        worst = worst.max(rel);
    }
    println!("max |d_device/d_host - 1|: {worst:.3e} (tolerance {tol:.1e})");
    assert!(
        worst <= tol,
        "the device draw leaves the host draw by {worst}, above {tol}"
    );
}

/// (147.5): every parcel of the event carries `c / d_i^3` droplets, so the
/// event's mass is `mdot dt stride` to round-off - (66.13) per parcel.
#[test]
fn the_event_mass_is_exact() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = box_mesh([10, 10, 10], [0.0; 3], [1.0; 3]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let rr = the_rr();
    let one = run_case(
        &gpu,
        &hm,
        &gm,
        100000,
        rr_injector(100000, 0.0, SizeDistribution::RosinRammler(rr)),
        1,
    );
    let live = one.live();
    let masses: Vec<f64> = mass_rows(&one, &live, liquid_density())
        .iter()
        .map(|r| r.1)
        .collect();
    let dt: Scalar = 1e-3;
    let want = w64(1e-3) * w64(dt) * 1.0;
    let got = exact_sum(&masses);
    let rel = (got - want).abs() / want;
    let tol = if cfg!(feature = "single") {
        (1e-12 * 2f64.powi(29)).min(1e-5).max(1e-12)
    } else {
        1e-12
    };
    println!("event mass {got:.9e} kg against mdot dt {want:.9e} kg (rel {rel:.3e})");
    assert!(rel <= tol, "event mass off by {rel:.3e}, above {tol:.1e}");
}

// ==========================================================================
//  Determinism, and the refusals (D)
// ==========================================================================

/// Two pools, one seed: the draw is a pure function of
/// `(injector, event, index, seed)`, so `d` and `n_p` are THE SAME BITS. A
/// different seed moves nearly every diameter.
#[test]
fn same_seed_same_bits() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = box_mesh([10, 10, 10], [0.0; 3], [1.0; 3]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let mk = |seed: u64| {
        rr_injector(
            20000,
            0.0,
            SizeDistribution::RosinRammler(RosinRammler {
                x: 1e-4,
                q: 2.5,
                seed,
            }),
        )
    };
    let a = run_case(&gpu, &hm, &gm, 20000, mk(7), 1);
    let b = run_case(&gpu, &hm, &gm, 20000, mk(7), 1);
    let c = run_case(&gpu, &hm, &gm, 20000, mk(8), 1);
    let la = a.live();
    let lb = b.live();
    let lc = c.live();
    assert_eq!(la.len(), 20000, "the pool is not full");
    let mut moved = 0;
    for k in 0..la.len() {
        let (i, j, l) = (la[k], lb[k], lc[k]);
        assert_eq!(a.d[i].to_bits(), b.d[j].to_bits(), "slot {i}: d changed");
        assert_eq!(
            a.n_p[i].to_bits(),
            b.n_p[j].to_bits(),
            "slot {i}: n_p changed"
        );
        if a.d[i].to_bits() != c.d[l].to_bits() {
            moved += 1;
        }
    }
    println!("{moved} of {} diameters moved with the seed", la.len());
    assert!(
        moved > 19000,
        "only {moved} of {} diameters moved with the seed",
        la.len()
    );
}

/// SPEC-LIT 147.1's refusals, by name, at `Parcels::new`: X and q not
/// finite and positive, a largest draw that overflows `Scalar`, and - by
/// `validate` directly - a smallest draw so fine it rounds to 0 or overflows
/// `n_p`.
#[test]
fn rosin_rammler_refusals() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = box_mesh([10, 10, 10], [0.0; 3], [1.0; 3]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let dt: Scalar = 1e-3;
    let cases: [(Scalar, Scalar); 6] = [
        (0.0, 2.5),
        (Scalar::NAN, 2.5),
        (1e-4, 0.0),
        (1e-4, -1.0),
        (1e-4, Scalar::INFINITY),
        (1e-4, 1e-3),
    ];
    for (x, q) in cases {
        let inj = Injector {
            position: Vec3::new(0.5, 0.5, 0.5),
            parcels_per_event: 8,
            size: SizeDistribution::RosinRammler(RosinRammler { x, q, seed: 7 }),
            ..Default::default()
        };
        let msg = match Parcels::new(&gpu, &hm, &gm, controls(16), &[inj], dt) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("injector X {x} q {q} was accepted"),
        };
        assert!(
            msg.contains("Rosin-Rammler"),
            "X {x} q {q}: the refusal is \"{msg}\""
        );
    }
    let c = event_mass_coefficient(1e-3, dt, 1, 8, 1.2);
    let err = RosinRammler {
        x: 1e-30,
        q: 0.05,
        seed: 7,
    }
    .validate(0, 8, c)
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("Rosin-Rammler"),
        "the smallest-draw refusal is \"{err}\""
    );
}

// ==========================================================================
//  The default, checked by measurement
// ==========================================================================

/// An injector without a distribution reads `diameter`: the kernel's other
/// branch is the §66.8 store unchanged, so every diameter is the given one
/// EXACTLY and every weight is (66.13) computed in `Scalar`, bit for bit.
#[test]
fn a_monodisperse_injector_is_unchanged() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = box_mesh([10, 10, 10], [0.0; 3], [1.0; 3]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let ctrl = controls(64);
    let dt: Scalar = 1e-3;
    let inj = Injector {
        position: Vec3::new(0.5, 0.5, 0.5),
        speed: 1.0,
        mass_flow: 1e-3,
        parcels_per_event: 64,
        diameter: 1e-4,
        size: SizeDistribution::Monodisperse,
        ..Default::default()
    };
    let d0 = inj.diameter;
    let mdot = inj.mass_flow;
    let n = inj.parcels_per_event;
    let (u, rho) = still_gas(&gpu, &gm, 1.2);
    let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[inj], dt).unwrap();
    p.step(&gpu, &u, &rho, None, None, dt).unwrap();
    p.stats(&gpu).unwrap().check_capacity().unwrap();
    let s = p.snapshot(&gpu).unwrap();
    let live = s.live();
    assert_eq!(live.len(), 64, "the pool is not full");
    // (66.13), computed in Scalar exactly as `Parcels::new` computes it.
    let m_droplet = ctrl.rho_liquid * std::f64::consts::FRAC_PI_6 as Scalar * d0 * d0 * d0;
    let expect = mdot * dt * 1.0 as Scalar / (n as Scalar * m_droplet);
    for &i in &live {
        assert_eq!(s.d[i], d0, "slot {i}: the monodisperse diameter changed");
        assert_eq!(s.n_p[i], expect, "slot {i}: the weight is not (66.13)'s");
    }
}
