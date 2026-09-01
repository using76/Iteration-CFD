// meteor-cfd - Copyright (c) 2026 주식회사 메테오시뮬레이션 (Meteo Simulation Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.

//! SPEC-LIT S67.10's table: the scan against an exact host prefix sum, the
//! three gates (the CSR is a permutation of the live set in `(cell, uid)`
//! order; the deposited totals are exactly what was put in; permuting the
//! slots moves no bit), the CUDA-graph claim, and the refusals.
//!
//! Written from the same sources as `src/parcels/deposit.rs`; see that
//! module's header. No GPL-licensed source was consulted.

use super::*;
use crate::blockgen::{self, BlockSpec, GradedAxis};
use crate::device::Gpu;
use crate::mesh::{GpuMesh, HostMesh};
use crate::parcels::{
    DragModel, EvaporationControls, ParcelControls, ParcelPhysics, ParcelSnapshot, Parcels,
    SeedParcel, WallAction,
};
use crate::types::Vec3;

// ----------------------------------------------------------------------
//  Fixtures
// ----------------------------------------------------------------------

/// A uniform Cartesian block of `n` cells over the unit cube, all patches
/// walls. Cell `(i, j, k)` is index `i + nx*(j + ny*k)`.
fn block(n: usize) -> HostMesh {
    let axis = || GradedAxis { lo: 0.0, hi: 1.0, n, expansion: 1.0, two_sided: false };
    blockgen::build_mesh(&BlockSpec {
        x: axis(),
        y: axis(),
        z: axis(),
        windows: Vec::new(),
        patch_name: BlockSpec::default().patch_name,
        patch_type: ["wall"; 6].map(String::from),
        cyclic: Vec::new(),
    })
    .expect("block mesh")
}

/// Controls that hold every parcel exactly where it was put: no drag, no
/// gravity, so a step is the identity on position and the deposition is a
/// statement about the sort alone.
fn still(capacity: usize) -> ParcelControls {
    ParcelControls {
        capacity,
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
        persistent_blocks: None,
    }
}

/// A QUARTER of an ulp of 1.0, which is what makes gate 67-C's fixture
/// order-sensitive without depending on the scalar width:
///
/// * `1 + TINY` rounds back to `1`, a quarter ulp being below the half-ulp tie;
/// * `TINY + TINY + TINY` is `0.75` ulp, so `1 + 3*TINY` rounds UP to
///   `1 + 1` ulp.
///
/// The identity-ascending sum and the slot-order sum therefore differ by
/// exactly one ulp, in `f64` and in `f32` alike. Writing `1e-16` instead would
/// be an `f64` constant that silently stops discriminating under
/// `--features single`, and the gate would pass vacuously on precisely the
/// build nobody runs.
const TINY: Scalar = Scalar::EPSILON / 4.0;

fn seed_at(position: Vec3, n_p: Scalar, diameter: Scalar, uid: u64) -> SeedParcel {
    SeedParcel {
        position,
        velocity: Vec3::ZERO,
        diameter,
        temperature: 293.15,
        n_p,
        uid: Some(uid),
    }
}

/// The host mirror of `parcelDeposit`, summing in the same order over the
/// same CSR. Used to check the device gather, and to state what "the
/// canonical sum" is independently of the kernel that produced it.
fn host_gather(
    csr: &ParcelCsrSnapshot,
    pool: &ParcelSnapshot,
    vol: &[Scalar],
    rho_l: Scalar,
) -> DepositSnapshot {
    let pi6 = std::f64::consts::FRAC_PI_6 as Scalar;
    let n = csr.n_cells;
    let mut out = DepositSnapshot {
        count: vec![0; n],
        weight: vec![0.0; n],
        volume_fraction: vec![0.0; n],
        mass: vec![0.0; n],
    };
    for (c, &vc) in vol.iter().enumerate().take(n) {
        let (k0, k1) = (csr.offset[c] as usize, csr.offset[c + 1] as usize);
        let mut w = 0.0 as Scalar;
        let mut v = 0.0 as Scalar;
        for k in k0..k1 {
            let p = csr.index[k] as usize;
            let np = pool.n_p[p];
            let d = pool.d[p];
            w += np;
            v += np * pi6 * d * d * d;
        }
        out.count[c] = (k1 - k0) as i32;
        out.weight[c] = w;
        out.volume_fraction[c] = if vc > 0.0 { v / vc } else { 0.0 };
        out.mass[c] = rho_l * v;
    }
    out
}

/// Everything S67.10 row 1 claims about the CSR, in one place: the offsets
/// are a valid row pointer, every entry lands in the cell it says it does,
/// each segment is strictly increasing in identity, and the whole thing is a
/// permutation of the live set. Returns the reason on failure so the test
/// says what was wrong rather than which line it was on.
fn csr_defects(csr: &ParcelCsrSnapshot, pool: &ParcelSnapshot) -> Vec<String> {
    let mut bad = Vec::new();
    if csr.offset[0] != 0 {
        bad.push(format!("offset[0] is {} and must be 0", csr.offset[0]));
    }
    for c in 0..csr.n_cells {
        if csr.offset[c + 1] < csr.offset[c] {
            bad.push(format!("offset is not monotone at cell {c}"));
        }
    }

    let live: std::collections::BTreeSet<usize> = (0..pool.cell.len())
        .filter(|&i| pool.cell[i] >= 0)
        .collect();
    if csr.n_live != live.len() {
        bad.push(format!(
            "offset[n_cells] says {} live parcels, the pool has {}",
            csr.n_live,
            live.len()
        ));
    }

    let mut seen = std::collections::BTreeSet::new();
    for c in 0..csr.n_cells {
        let (k0, k1) = (csr.offset[c] as usize, csr.offset[c + 1] as usize);
        let mut prev: Option<u64> = None;
        for k in k0..k1 {
            let p = csr.index[k] as usize;
            if p >= pool.cell.len() {
                bad.push(format!("CSR entry {k} names slot {p}, outside the pool"));
                continue;
            }
            if pool.cell[p] as usize != c {
                bad.push(format!(
                    "cell {c}'s segment holds slot {p}, whose own cell is {}",
                    pool.cell[p]
                ));
            }
            if !seen.insert(p) {
                bad.push(format!("slot {p} appears twice in the CSR"));
            }
            let uid = pool.uid[p];
            if let Some(q) = prev {
                if uid <= q {
                    bad.push(format!(
                        "cell {c} is not in ascending identity order: {q} then {uid}"
                    ));
                }
            }
            prev = Some(uid);
        }
    }
    for p in &live {
        if !seen.contains(p) {
            bad.push(format!("live slot {p} is in no segment of the CSR"));
        }
    }
    bad
}

// ======================================================================
//  SPEC-LIT (67.2): the scan
// ======================================================================

/// A deterministic pseudo-random small non-negative integer, so the scan is
/// tested on something other than a constant and the test is still exactly
/// reproducible. SplitMix64's finaliser again, and again not as a generator.
fn spread(i: usize) -> i32 {
    (mix(i as u64) % 17) as i32
}

/// SplitMix64's finaliser, used - as everywhere else in this crate - as a
/// deterministic scrambler and never as a source of randomness. It is what
/// scatters the test fixtures over the mesh instead of leaving them on a
/// lattice that a broken sort could still get right.
fn mix(i: u64) -> u64 {
    let mut z = i.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z ^= z >> 30;
    z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    z
}

/// The scan is EXACT, at every size that straddles a tile boundary, and at a
/// size large enough that the single-block pass over the tile sums has to
/// loop. Integer addition is associative, so "exact" here means equal to the
/// host prefix sum element by element - there is no tolerance to state.
#[test]
fn the_scan_is_exact_at_every_size_that_straddles_a_tile() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    for n in [1usize, 2, 255, 256, 1023, 1024, 1025, 4096, 100_000, 2_000_000] {
        let host: Vec<i32> = (0..n).map(spread).collect();
        let mut want = vec![0i32; n];
        let mut acc = 0i32;
        for i in 0..n {
            want[i] = acc;
            acc += host[i];
        }

        let inp = gpu.upload(&host).unwrap();
        let mut out: DevBuf<i32> = gpu.zeros(n).unwrap();
        let mut scan = DeviceScan::new(&gpu, n).unwrap();
        scan.run(&gpu, &inp, &mut out).unwrap();
        let got = gpu.download(&out).unwrap();
        assert_eq!(
            got, want,
            "the exclusive scan is wrong at n = {n} (tile = {SORT_TILE})"
        );
    }
}

/// A scan over zero elements is refused rather than launching a zero-block
/// grid, which is an invalid configuration and not a no-op.
#[test]
fn a_scan_of_nothing_is_refused_by_name() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let e = match DeviceScan::new(&gpu, 0) {
        Ok(_) => panic!("a zero-length scan was accepted"),
        Err(e) => e.to_string(),
    };
    assert!(e.contains("absence of a scan"), "{e}");
}

// ======================================================================
//  SPEC-LIT (67.3): the pass count
// ======================================================================

/// The cell key is sorted over exactly the bits it can occupy, and no more.
/// The sentinel is `n_cells` itself, so a 1000-cell mesh needs ten bits and
/// therefore two passes - not the four a blind 32-bit key would cost.
#[test]
fn the_cell_key_costs_only_the_passes_its_own_bits_need() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    // 8, 64 and 216 cells all fit the 8 bits of one digit; 343 and 1000 need
    // ten, so two. The sentinel is `n_cells` itself, which is why 216 - not
    // 255 - is still one pass and 256 would not be.
    for (n, want) in [(2usize, 1u32), (4, 1), (6, 1), (7, 2), (10, 2)] {
        let hm = block(n);
        let gm = GpuMesh::upload(&gpu, &hm).unwrap();
        let p = Parcels::new(&gpu, &hm, &gm, still(1024), &[], 0.1).unwrap();
        let dep = ParcelDeposition::new(&gpu, &p).unwrap();
        assert_eq!(
            dep.cell_passes(),
            want,
            "{} cells: bits({}) = {}",
            gm.n_cells,
            gm.n_cells,
            usize::BITS - gm.n_cells.leading_zeros()
        );
        assert_eq!(dep.passes(), UID_PASSES + want);
        // Both ping-pong parities are exercised by the two mesh sizes above,
        // which is the point of testing more than one.
        assert_eq!(dep.final_is_a, want % 2 == 0);
    }
}

// ======================================================================
//  Gate 67-A: the CSR is a permutation of the live set
// ======================================================================

/// **Gate 67-A.** Every live parcel appears exactly once, in the segment of
/// the cell it is actually in, and each segment ascends in identity. That is
/// the whole contract of (67.5), and it is what makes the gather in (67.6)
/// both complete and non-double-counting.
///
/// Posed on a mesh whose cell count needs two radix passes and on one that
/// needs one, so both ping-pong parities are checked.
#[test]
fn gate_67a_the_csr_is_a_permutation_of_the_live_set() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    for n in [4usize, 10] {
        let hm = block(n);
        let gm = GpuMesh::upload(&gpu, &hm).unwrap();
        let h = 1.0 / n as Scalar;

        // 300 parcels sprayed deterministically over the mesh, several per
        // cell in places and none at all in others, with identities in an
        // order deliberately unrelated to their slots.
        let seeds: Vec<SeedParcel> = (0..300u64)
            .map(|i| {
                let f = |a: u64| (mix(i * 3 + a) % (n as u64)) as Scalar;
                let pos = Vec3::new((f(0) + 0.5) * h, (f(1) + 0.5) * h, (f(2) + 0.5) * h);
                seed_at(pos, 1.0, 1e-4, crate::parcels::parcel_uid(1, 5, 299 - i))
            })
            .collect();

        // 1024 slots is ONE radix tile and 8192 is eight, so the two mesh
        // sizes below cover a single-block sort - where the digit-major
        // global scan of (67.6) is trivial - and a multi-block one, where it
        // is the thing that makes the scatter stable across blocks.
        let cap = if n == 4 { 1024 } else { 8192 };
        let mut p = Parcels::new(&gpu, &hm, &gm, still(cap), &[], 0.1).unwrap();
        p.seed(&gpu, &hm, &seeds).unwrap();
        let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
        assert_eq!(dep.padded_capacity() / SORT_TILE, cap / SORT_TILE);
        dep.build(&gpu, &p).unwrap();

        let csr = dep.csr_snapshot(&gpu).unwrap();
        let pool = p.snapshot(&gpu).unwrap();
        let bad = csr_defects(&csr, &pool);
        assert!(bad.is_empty(), "{n}^3 mesh: {}", bad.join("; "));
        assert_eq!(csr.n_live, 300);
        assert_eq!(dep.live_count(&gpu).unwrap(), 300);
    }
}

/// An empty pool is a valid pool: every segment is empty, every deposited
/// field is zero, and nothing has to be special-cased on the host to make it
/// so. (A host branch on "are there any parcels" is exactly what a captured
/// graph cannot contain - S66.7.)
#[test]
fn an_empty_pool_deposits_zero_and_needs_no_special_case() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block(4);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let p = Parcels::new(&gpu, &hm, &gm, still(1024), &[], 0.1).unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    dep.update(&gpu, &p).unwrap();

    let csr = dep.csr_snapshot(&gpu).unwrap();
    assert_eq!(csr.n_live, 0);
    assert!(csr.offset.iter().all(|&o| o == 0));
    let d = dep.snapshot(&gpu).unwrap();
    assert_eq!(d.total_count(), 0);
    assert_eq!(d.total_weight().to_bits(), (0.0 as Scalar).to_bits());
    assert!(d.mass.iter().all(|&m| m == 0.0));
}

// ======================================================================
//  Gate 67-B: what went in comes out
// ======================================================================

/// **Gate 67-B.** The deposited totals are exactly what was put in.
///
/// Two exact claims and one to round-off, and the difference between them is
/// stated rather than blurred:
///
/// * `sum_P count[P]` is an INTEGER sum and equals the live parcel count
///   with no tolerance at all;
/// * `sum_P weight[P]` is a sum of `n_p`, and with dyadic weights every
///   partial sum is exactly representable, so it equals `sum_p n_p` **bit
///   for bit** whatever order the cells were visited in;
/// * `volume_fraction` and `mass` carry a `n_p (pi/6) d^3` product, and the
///   device is free to contract `acc + a*b` into an FMA where the host is
///   not, so those are checked against the host mirror to round-off. The
///   measured gap is asserted at `1e-15` relative and reported by
///   `ofgpu-validate`.
#[test]
fn gate_67b_what_was_deposited_is_exactly_what_was_put_in() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let n = 4usize;
    let hm = block(n);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let h = 1.0 / n as Scalar;

    // Dyadic weights, so the running sum is exact at every step: the claim
    // "the total is exactly what went in" is then about the gather and not
    // about how lucky the rounding was.
    let dyadic = [1.0 as Scalar, 2.0, 0.5, 0.25, 8.0, 0.125, 4.0, 16.0];
    let seeds: Vec<SeedParcel> = (0..120u64)
        .map(|i| {
            let f = |a: u64| (mix(i * 3 + a) % (n as u64)) as Scalar;
            let pos = Vec3::new((f(0) + 0.5) * h, (f(1) + 0.5) * h, (f(2) + 0.5) * h);
            seed_at(
                pos,
                dyadic[(i % 8) as usize],
                1e-4 + 1e-5 * (i % 5) as Scalar,
                crate::parcels::parcel_uid(2, 9, (i * 37) % 4096),
            )
        })
        .collect();
    let put_in: Scalar = {
        let mut s = 0.0 as Scalar;
        for sd in &seeds {
            s += sd.n_p;
        }
        s
    };

    let ctrl = still(1024);
    let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[], 0.1).unwrap();
    p.seed(&gpu, &hm, &seeds).unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    dep.update(&gpu, &p).unwrap();

    let got = dep.snapshot(&gpu).unwrap();
    assert_eq!(got.total_count(), seeds.len() as i64, "a parcel was lost or double-counted");
    assert_eq!(
        got.total_weight().to_bits(),
        put_in.to_bits(),
        "the deposited weight {} is not the {} that went in",
        got.total_weight(),
        put_in
    );

    // And every cell is the canonical sum the host computes over the same
    // CSR, to round-off.
    let csr = dep.csr_snapshot(&gpu).unwrap();
    let pool = p.snapshot(&gpu).unwrap();
    let want = host_gather(&csr, &pool, &hm.v, ctrl.rho_liquid);
    assert_eq!(got.count, want.count);
    for c in 0..gm.n_cells {
        assert_eq!(
            got.weight[c].to_bits(),
            want.weight[c].to_bits(),
            "cell {c}: weight is a sum of pure additions and must match bit for bit"
        );
        for (name, a, b) in [
            ("alphaP", got.volume_fraction[c], want.volume_fraction[c]),
            ("mass", got.mass[c], want.mass[c]),
        ] {
            let e = if b == 0.0 { (a - b).abs() } else { (a - b).abs() / b.abs() };
            assert!(e <= 1e-15, "cell {c} {name}: {a} against {b}, relative {e}");
        }
    }

    // The physical statement, on a cell whose contents are known by hand:
    // alpha_p is the droplet volume in the cell over the cell volume.
    let pi6 = std::f64::consts::FRAC_PI_6 as Scalar;
    let cell_v = h * h * h;
    for c in 0..gm.n_cells {
        if got.count[c] == 0 {
            continue;
        }
        let mut v = 0.0 as Scalar;
        for k in csr.offset[c] as usize..csr.offset[c + 1] as usize {
            let p = csr.index[k] as usize;
            let d = pool.d[p];
            v += pool.n_p[p] * pi6 * d * d * d;
        }
        let e = (got.volume_fraction[c] - v / cell_v).abs() / (v / cell_v);
        assert!(e <= 1e-15, "cell {c}: alpha_p {} against {}", got.volume_fraction[c], v / cell_v);
        let em = (got.mass[c] - ctrl.rho_liquid * v).abs() / (ctrl.rho_liquid * v);
        assert!(em <= 1e-15, "cell {c}: mass");
    }
}

/// Injected parcels, not seeded ones: after twenty steps of a spray into a
/// box of walls, the CSR holds exactly the parcels that are still alive -
/// injected, less escaped, less removed at a wall - and every one of them
/// exactly once. This is the conservation statement in the form a running
/// case can check, with no dyadic weights to help it.
#[test]
fn a_spray_deposits_every_live_parcel_exactly_once() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block(10);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let u = crate::field::GpuVectorField::zeros(&gpu, &gm, "U").unwrap();
    let rho = gpu.upload(&vec![1.2 as Scalar; gm.n_cells]).unwrap();

    let ctrl = ParcelControls {
        drag: DragModel::SchillerNaumann,
        gravity: Vec3::new(0.0, 0.0, -9.81),
        ..still(4096)
    };
    let inj = crate::parcels::Injector {
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
    };
    let dt: Scalar = 0.05;
    let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[inj], dt).unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    for _ in 0..20 {
        p.step(&gpu, &u, &rho, None, None, dt).unwrap();
    }
    dep.update(&gpu, &p).unwrap();

    let st = p.stats(&gpu).unwrap();
    let pool = p.snapshot(&gpu).unwrap();
    let csr = dep.csr_snapshot(&gpu).unwrap();
    let got = dep.snapshot(&gpu).unwrap();

    assert_eq!(st.n_injected, 160);
    assert_eq!(st.n_lost, 0, "the walk lost a parcel on a Cartesian mesh");
    let alive = st.n_injected - st.n_escaped - st.n_wall;
    assert!(alive > 0, "the case must leave something alive: {st:?}");
    assert_eq!(
        got.total_count() as i64,
        alive,
        "{} injected, {} escaped, {} at a wall, but {} deposited",
        st.n_injected,
        st.n_escaped,
        st.n_wall,
        got.total_count()
    );
    assert!(csr_defects(&csr, &pool).is_empty(), "{:?}", csr_defects(&csr, &pool));

    // And the deposited liquid mass is the mass the live parcels carry.
    let want = pool.liquid_mass(ctrl.rho_liquid);
    let e = (got.total_mass() - want).abs() / want;
    assert!(e <= 1e-14, "deposited {} kg against {} kg carried", got.total_mass(), want);
}

/// The sort over **thirty-two tiles and twenty thousand parcels**, checked the
/// same way gate 67-A checks three hundred.
///
/// Everything else here runs at one, four or eight tiles, where a radix pass
/// touches a handful of blocks and every digit bucket is nearly empty. A sort
/// can be correct there and wrong at scale in two specific ways - the
/// digit-major global scan (67.6) mis-ordering blocks, and the per-warp rank of
/// (67.7) mis-accumulating its running block offset - and neither would show up
/// in a small case. Twenty thousand parcels put every one of the 256 identity
/// digit buckets across all thirty-two blocks, and land about two hundred
/// parcels in each occupied cell, which is what loads the scatter's
/// same-digit ranking.
///
/// **How few cells they occupy is a finding about S66, not about the sort.**
/// (66.8)'s cone is a deterministic ring, so one injector's parcels sit on
/// concentric circles for ever and reach about sixteen cells of a thousand
/// however many of them there are. Eight injectors reach ninety-four. Nothing
/// here is wrong; it is what an unrandomised spray looks like, and it is the
/// clearest argument yet for the counter-based RNG S66.14 names.
///
/// It is filled by an **injector**, not by [`Parcels::seed`]: seeding is
/// `O(n_seeds x n_cells)` on the host because S66.6's `locate_cell` is a linear
/// scan, which is the right answer for a handful of injectors at setup and
/// hopeless for a pool. That is a real limit of S66, recorded here where it was
/// found.
#[test]
fn the_sort_holds_at_twenty_thousand_parcels_and_thirty_two_tiles() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block(10);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let u = crate::field::GpuVectorField::zeros(&gpu, &gm, "U").unwrap();
    let rho = gpu.upload(&vec![1.2 as Scalar; gm.n_cells]).unwrap();

    let ctrl = ParcelControls {
        drag: DragModel::SchillerNaumann,
        wall: WallAction::Rebound,
        gravity: Vec3::new(0.0, 0.0, -9.81),
        ..still(32768)
    };
    // EIGHT injectors, not one, and the reason is a property of S66 worth
    // recording: (66.8)'s cone is a deterministic RING - azimuth (i + 1/2)/n
    // of a turn at a fixed half-angle - so every parcel of an event leaves on
    // one circle at one speed and stays on it. One injector for twenty events
    // therefore paints twenty concentric circles and occupies about SIXTEEN
    // cells of a thousand, however many parcels it emits. That is the ring,
    // not the sort; eight injectors at different points and angles give the
    // cell key something to sort.
    let inj: Vec<crate::parcels::Injector> = (0..8u32)
        .map(|i| {
            let f = i as Scalar;
            crate::parcels::Injector {
                position: Vec3::new(0.2 + 0.08 * f, 0.25 + 0.06 * f, 0.85 - 0.09 * f),
                axis: Vec3::new(0.2 * f - 0.7, 0.15 * f - 0.5, -1.0),
                cone_half_angle: 0.25 + 0.12 * f,
                standoff: 0.02,
                speed: 1.5 + 0.5 * f,
                diameter: 2e-4,
                temperature: 300.0,
                mass_flow: 1e-3,
                parcels_per_event: 125,
                interval: 0.0,
            }
        })
        .collect();
    let dt: Scalar = 5e-3;

    let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &inj, dt).unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    assert_eq!(dep.padded_capacity() / SORT_TILE, 32);
    for _ in 0..20 {
        p.step(&gpu, &u, &rho, None, None, dt).unwrap();
    }
    dep.update(&gpu, &p).unwrap();

    let st = p.stats(&gpu).unwrap();
    let pool = p.snapshot(&gpu).unwrap();
    let csr = dep.csr_snapshot(&gpu).unwrap();
    let got = dep.snapshot(&gpu).unwrap();

    assert_eq!(st.n_injected, 20_000);
    assert_eq!(st.n_dropped, 0);
    assert_eq!(st.n_lost, 0);
    let bad = csr_defects(&csr, &pool);
    assert!(bad.is_empty(), "{} defects, first: {}", bad.len(), bad[0]);
    assert_eq!(
        got.total_count() as i64,
        st.n_injected - st.n_escaped - st.n_wall,
        "a parcel was lost or double-counted at scale"
    );
    // Every digit bucket has to have been busy for this to be the test it
    // claims to be: with 20000 parcels over 1000 cells the CSR should be
    // occupied nearly everywhere.
    // Measured: 94 of 1000 cells, so roughly 200 parcels per occupied cell.
    // That is the number this fixture is really about - long segments and
    // heavy digit collisions - and it is capped by (66.8)'s ring, not by the
    // parcel count.
    let occupied = got.count.iter().filter(|&&k| k > 0).count();
    assert!(occupied > 80, "only {occupied} of {} cells hold a parcel", gm.n_cells);

    // And it is still the canonical answer: a second run of the same case
    // deposits the same bits.
    let mut q = Parcels::new(&gpu, &hm, &gm, ctrl, &inj, dt).unwrap();
    let mut dep2 = ParcelDeposition::new(&gpu, &q).unwrap();
    for _ in 0..20 {
        q.step(&gpu, &u, &rho, None, None, dt).unwrap();
    }
    dep2.update(&gpu, &q).unwrap();
    let again = dep2.snapshot(&gpu).unwrap();
    assert_eq!(got.count, again.count);
    for i in 0..got.mass.len() {
        assert_eq!(got.mass[i].to_bits(), again.mass[i].to_bits(), "cell {i}");
    }
}

// ======================================================================
//  Gate 67-C: the canonicalisation
// ======================================================================

/// The pool used by gate 67-C: four parcels in one cell whose weights make
/// floating-point addition visibly non-associative, plus two elsewhere so the
/// CSR is not a single segment.
///
/// [`TINY`] is a quarter of an ulp of 1.0, so `1 + TINY` rounds back to `1`
/// while `1 + 3*TINY` rounds up by one ulp, and the canonical
/// identity-ascending order and the slot order give **different bits**. That
/// is what gives the gate its teeth: a sort that merely grouped by cell,
/// stable on the input order, would pass every other test in this file and
/// fail this one.
fn crowded_cell(order: [usize; 4]) -> Vec<SeedParcel> {
    let n_p = [1.0 as Scalar, TINY, TINY, TINY];
    // Identities ascending as 1, 2, 3 then 0: the heavy parcel sorts LAST.
    let uid = [400u64, 100, 200, 300];
    let pos = [
        Vec3::new(0.10, 0.10, 0.10),
        Vec3::new(0.12, 0.11, 0.13),
        Vec3::new(0.09, 0.14, 0.08),
        Vec3::new(0.15, 0.15, 0.15),
    ];
    let mut v: Vec<SeedParcel> = order
        .iter()
        .map(|&i| seed_at(pos[i], n_p[i], 1e-4, uid[i]))
        .collect();
    v.push(seed_at(Vec3::new(0.6, 0.6, 0.6), 3.0, 2e-4, 900));
    v.push(seed_at(Vec3::new(0.9, 0.3, 0.7), 5.0, 3e-4, 901));
    v
}

/// **Gate 67-C.** Two runs whose parcel SET is identical but whose slot order
/// is not deposit the same bits.
///
/// This is the load-bearing claim of the whole section. The sorted order is a
/// function of `{(cell_p, uid_p)}` alone - not of which slot a parcel
/// occupies, not of the order it was injected in, not of how the pool was
/// compacted - so the gather sums the same numbers in the same order, and the
/// answer is bitwise identical. Without the identity in the key this would
/// hold only for a sort that happened to be stable on this particular input,
/// which is exactly the false pass the third assertion below rules out.
#[test]
fn gate_67c_permuting_the_slots_moves_not_one_bit() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block(4);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let ctrl = still(1024);

    let run = |seeds: &[SeedParcel]| -> (DepositSnapshot, ParcelCsrSnapshot) {
        let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[], 0.1).unwrap();
        p.seed(&gpu, &hm, seeds).unwrap();
        let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
        dep.update(&gpu, &p).unwrap();
        (dep.snapshot(&gpu).unwrap(), dep.csr_snapshot(&gpu).unwrap())
    };

    let (a, csr_a) = run(&crowded_cell([0, 1, 2, 3]));
    let (b, csr_b) = run(&crowded_cell([3, 2, 1, 0]));
    let (c, _) = run(&crowded_cell([2, 0, 3, 1]));

    for (name, x, y) in [("reversed", &b, &a), ("shuffled", &c, &a)] {
        assert_eq!(x.count, y.count, "{name}: the counts moved");
        for i in 0..y.weight.len() {
            assert_eq!(
                x.weight[i].to_bits(),
                y.weight[i].to_bits(),
                "{name}: cell {i} weight {} against {}",
                x.weight[i],
                y.weight[i]
            );
            assert_eq!(x.mass[i].to_bits(), y.mass[i].to_bits(), "{name}: cell {i} mass");
            assert_eq!(
                x.volume_fraction[i].to_bits(),
                y.volume_fraction[i].to_bits(),
                "{name}: cell {i} alphaP"
            );
        }
    }

    // The CSR itself is the same object, not merely the sums: the same slots
    // in the same order would be too strong (the slots moved), but the same
    // IDENTITIES in the same order is exactly the claim.
    let ids = |csr: &ParcelCsrSnapshot, seeds: &[SeedParcel]| -> Vec<u64> {
        csr.live_order().iter().map(|&s| seeds[s as usize].uid.unwrap()).collect()
    };
    assert_eq!(
        ids(&csr_a, &crowded_cell([0, 1, 2, 3])),
        ids(&csr_b, &crowded_cell([3, 2, 1, 0])),
        "the canonical order is not the same sequence of identities"
    );

    // **The gate has teeth.** Summing in SLOT order instead gives different
    // bits for the two arrangements, so a sort that grouped by cell and was
    // merely stable on the input would have failed here. The canonical answer
    // is neither run's slot-order sum by accident: it is the identity-ordered
    // one.
    let slot_sum = |order: [usize; 4]| -> Scalar {
        let n_p = [1.0 as Scalar, TINY, TINY, TINY];
        let mut s = 0.0 as Scalar;
        for &i in &order {
            s += n_p[i];
        }
        s
    };
    assert_ne!(
        slot_sum([0, 1, 2, 3]).to_bits(),
        slot_sum([3, 2, 1, 0]).to_bits(),
        "the fixture stopped being order-sensitive; the gate would pass vacuously"
    );
    let canonical = slot_sum([1, 2, 3, 0]);
    assert_eq!(
        a.weight[0].to_bits(),
        canonical.to_bits(),
        "cell 0 deposited {} where the identity-ascending sum is {}",
        a.weight[0],
        canonical
    );
}

/// **Gate 67-C, in its second form: a real spray, every parcel in a different
/// slot.**
///
/// The permutation above is posed on seeded parcels, which is the cleanest way
/// to hold a *set* fixed while moving slots. This is the same claim on the
/// path a case actually uses. Both runs inject from the same injector, so the
/// injected parcels are the same set at the same positions with the same
/// identities - `uid = mix64(injector, event, index)` names none of them by
/// slot. What differs is where they land: the second run first seeds `k`
/// parcels aimed straight through the floor, which the wall removes on the
/// first step, so every injected parcel afterwards sits at slot `k + i`
/// instead of `i`.
///
/// The dead seeds are keyed `nCells` by (67.5) and drop out of the CSR
/// entirely, so what is left to compare is the same physical spray in a pool
/// that has been shifted underneath it. Every deposited bit must be unmoved.
#[test]
fn shifting_every_injected_parcel_into_a_different_slot_moves_no_bit() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block(10);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let u = crate::field::GpuVectorField::zeros(&gpu, &gm, "U").unwrap();
    let rho = gpu.upload(&vec![1.2 as Scalar; gm.n_cells]).unwrap();
    let ctrl = ParcelControls {
        drag: DragModel::SchillerNaumann,
        gravity: Vec3::new(0.0, 0.0, -9.81),
        ..still(4096)
    };
    let inj = crate::parcels::Injector {
        position: Vec3::new(0.5, 0.5, 0.55),
        axis: Vec3::new(0.0, 0.0, -1.0),
        cone_half_angle: std::f64::consts::FRAC_PI_6 as Scalar,
        standoff: 0.02,
        speed: 3.0,
        diameter: 2e-4,
        temperature: 300.0,
        mass_flow: 1e-3,
        parcels_per_event: 8,
        interval: 0.0,
    };
    let dt: Scalar = 0.05;

    let run = |k: usize| -> (DepositSnapshot, crate::parcels::ParcelStats) {
        let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[inj], dt).unwrap();
        if k > 0 {
            let seeds: Vec<SeedParcel> = (0..k)
                .map(|i| SeedParcel {
                    position: Vec3::new(0.05 + 0.01 * i as Scalar, 0.05, 0.02),
                    velocity: Vec3::new(0.0, 0.0, -100.0),
                    diameter: 3e-4,
                    temperature: 300.0,
                    n_p: 7.0,
                    uid: Some(1_000_000 + i as u64),
                })
                .collect();
            p.seed(&gpu, &hm, &seeds).unwrap();
        }
        let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
        for _ in 0..20 {
            p.step(&gpu, &u, &rho, None, None, dt).unwrap();
        }
        dep.update(&gpu, &p).unwrap();
        (dep.snapshot(&gpu).unwrap(), p.stats(&gpu).unwrap())
    };

    let (a, sa) = run(0);
    let (b, sb) = run(7);
    assert!(
        sb.n_wall >= sa.n_wall + 7,
        "the seeds were meant to die at the floor on the first step: {sa:?} against {sb:?}"
    );
    assert!(a.total_count() > 0, "the spray must leave something alive");
    assert_eq!(
        a.count, b.count,
        "the same spray in shifted slots deposited a different number of parcels"
    );
    for i in 0..a.weight.len() {
        assert_eq!(
            a.weight[i].to_bits(),
            b.weight[i].to_bits(),
            "cell {i}: {} against {}",
            a.weight[i],
            b.weight[i]
        );
        assert_eq!(a.mass[i].to_bits(), b.mass[i].to_bits(), "cell {i} mass");
        assert_eq!(
            a.volume_fraction[i].to_bits(),
            b.volume_fraction[i].to_bits(),
            "cell {i} alphaP"
        );
    }
}

/// SPEC-LIT S13.4.1's pair test, for the one new *input* this section adds.
///
/// §67 adds no case setting - everything in it is unconditional machinery, and
/// a knob that turned the canonicalisation off would be exactly the flag
/// §67.8(c) refuses to have. What it does add is `SeedParcel::uid`, and that
/// is an input like any other: two runs identical in **every byte but one
/// identity** are REQUIRED to deposit different bits, and this fails by name
/// if they do not.
///
/// The pair is the crowded cell of gate 67-C with the heavy parcel's identity
/// moved from last to first. The set of positions, diameters and weights is
/// untouched; only the order the four are summed in changes, and that is worth
/// exactly one ulp of 1.0 - which is the whole reason gate 67-C can see
/// anything at all.
#[test]
fn changing_one_identity_changes_the_deposited_bits() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block(4);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();

    let run = |heavy_uid: u64| -> DepositSnapshot {
        let mut seeds = crowded_cell([0, 1, 2, 3]);
        seeds[0].uid = Some(heavy_uid);
        let mut p = Parcels::new(&gpu, &hm, &gm, still(1024), &[], 0.1).unwrap();
        p.seed(&gpu, &hm, &seeds).unwrap();
        let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
        dep.update(&gpu, &p).unwrap();
        dep.snapshot(&gpu).unwrap()
    };

    let last = run(400);
    let first = run(50);
    assert_ne!(
        last.weight[0].to_bits(),
        first.weight[0].to_bits(),
        "moving the heavy parcel's identity from last to first in cell 0 left the deposited \
         weight at {} - so either the identity is not in the sort key, or the fixture \
         stopped being order-sensitive",
        last.weight[0]
    );
    // And each is the sum its own order names, so the difference is the sort
    // and not some other effect of the identity.
    let w = [1.0 as Scalar, TINY, TINY, TINY];
    let sum = |order: [usize; 4]| order.iter().fold(0.0 as Scalar, |a, &i| a + w[i]);
    assert_eq!(last.weight[0].to_bits(), sum([1, 2, 3, 0]).to_bits());
    assert_eq!(first.weight[0].to_bits(), sum([0, 1, 2, 3]).to_bits());
}

/// Two identical runs are identical, which is the weaker half of 67-C and is
/// still worth its own name: it fails differently from the permutation gate,
/// and a failure here is a non-determinism in the sort itself rather than a
/// key that is not a total order.
#[test]
fn two_identical_runs_deposit_identical_bits() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block(10);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let h = 0.1 as Scalar;
    let seeds: Vec<SeedParcel> = (0..500u64)
        .map(|i| {
            let f = |a: u64| (mix(i * 3 + a) % 10) as Scalar;
            seed_at(
                Vec3::new((f(0) + 0.5) * h, (f(1) + 0.5) * h, (f(2) + 0.5) * h),
                1.0 + (i % 7) as Scalar * 1e-15,
                1e-4,
                crate::parcels::parcel_uid(3, 1, i * 11 % 8192),
            )
        })
        .collect();

    let run = || -> DepositSnapshot {
        let mut p = Parcels::new(&gpu, &hm, &gm, still(1024), &[], 0.1).unwrap();
        p.seed(&gpu, &hm, &seeds).unwrap();
        let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
        dep.update(&gpu, &p).unwrap();
        dep.snapshot(&gpu).unwrap()
    };
    let a = run();
    let b = run();
    assert_eq!(a.count, b.count);
    for i in 0..a.weight.len() {
        assert_eq!(a.weight[i].to_bits(), b.weight[i].to_bits(), "cell {i}");
        assert_eq!(a.mass[i].to_bits(), b.mass[i].to_bits(), "cell {i}");
    }
}

/// The pool's launch geometry is required to be inert (S66.12 row 15), and
/// that has to keep being true once the deposition reads the pool: a
/// different persistent grid must not move a deposited bit either.
#[test]
fn the_pool_grid_geometry_does_not_move_the_deposition() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block(10);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let u = crate::field::GpuVectorField::zeros(&gpu, &gm, "U").unwrap();
    let rho = gpu.upload(&vec![1.2 as Scalar; gm.n_cells]).unwrap();
    let inj = crate::parcels::Injector {
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
    };
    let dt: Scalar = 0.05;

    let mut prev: Option<DepositSnapshot> = None;
    for blocks in [1u32, 5, 64] {
        let ctrl = ParcelControls {
            drag: DragModel::SchillerNaumann,
            gravity: Vec3::new(0.0, 0.0, -9.81),
            evaporation: EvaporationControls::default(),
            persistent_blocks: Some(blocks),
            ..still(4096)
        };
        let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[inj], dt).unwrap();
        let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
        for _ in 0..15 {
            p.step(&gpu, &u, &rho, None, None, dt).unwrap();
        }
        dep.update(&gpu, &p).unwrap();
        let s = dep.snapshot(&gpu).unwrap();
        if let Some(q) = &prev {
            assert_eq!(s.count, q.count, "{blocks} blocks moved the counts");
            for i in 0..s.weight.len() {
                assert_eq!(s.mass[i].to_bits(), q.mass[i].to_bits(), "{blocks} blocks, cell {i}");
            }
        }
        prev = Some(s);
    }
}

// ======================================================================
//  SPEC-LIT (67.7): CUDA-graph capture
// ======================================================================

/// The sort, the CSR build and the gather capture into a graph ONCE and
/// replay, while the working set grows underneath them. Nothing in the
/// sequence allocates, synchronises or reads back, and every launch geometry
/// (the padded item count, the radix block count, the cell count) is a setup
/// constant, which is what makes that true. The ping-pong parity is a setup
/// constant for the same reason: a ping-pong that rotated between rebuilds
/// would leave the host pointing at the wrong buffer the moment a graph froze
/// the other one.
#[test]
fn the_sort_and_the_gather_capture_once_and_replay() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block(10);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let u = crate::field::GpuVectorField::zeros(&gpu, &gm, "U").unwrap();
    let rho = gpu.upload(&vec![1.2 as Scalar; gm.n_cells]).unwrap();
    let ctrl = ParcelControls {
        drag: DragModel::SchillerNaumann,
        gravity: Vec3::new(0.0, 0.0, -9.81),
        ..still(4096)
    };
    let inj = crate::parcels::Injector {
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
    };
    let dt: Scalar = 0.05;
    let steps = 15usize;

    let eager = {
        let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[inj], dt).unwrap();
        let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
        for _ in 0..steps {
            p.step(&gpu, &u, &rho, None, None, dt).unwrap();
            dep.update(&gpu, &p).unwrap();
        }
        dep.snapshot(&gpu).unwrap()
    };

    let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[inj], dt).unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    let captured = gpu
        .capture(|_| {
            p.step(&gpu, &u, &rho, None, None, dt)?;
            dep.update(&gpu, &p)
        })
        .expect("capture must not fail: nothing in the sequence allocates or reads back");
    let Some(mut g) = captured else {
        panic!("the capture produced an empty graph");
    };
    g.upload().unwrap();
    for _ in 0..steps {
        g.launch().unwrap();
    }
    gpu.sync().unwrap();

    let replayed = dep.snapshot(&gpu).unwrap();
    assert_eq!(replayed.count, eager.count, "the graph replay deposited different counts");
    for i in 0..eager.weight.len() {
        assert_eq!(replayed.weight[i].to_bits(), eager.weight[i].to_bits(), "cell {i}");
        assert_eq!(replayed.mass[i].to_bits(), eager.mass[i].to_bits(), "cell {i}");
    }
    assert!(dep.live_count(&gpu).unwrap() > 0);
}

// ======================================================================
//  SPEC-LIT S13.4: the refusals
// ======================================================================

/// Two seeds sharing an identity is refused **at setup, by name**, because
/// `(cell, uid)` is a total order only if `uid` is unique - and a duplicate
/// would not crash anything, it would quietly make the deposition depend on
/// slot order again, which is the one thing this section exists to prevent.
#[test]
fn a_duplicate_seed_identity_is_refused_by_name() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block(4);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let mut p = Parcels::new(&gpu, &hm, &gm, still(64), &[], 0.1).unwrap();
    let seeds = [
        seed_at(Vec3::new(0.1, 0.1, 0.1), 1.0, 1e-4, 77),
        seed_at(Vec3::new(0.6, 0.6, 0.6), 1.0, 1e-4, 77),
    ];
    let e = p.seed(&gpu, &hm, &seeds).unwrap_err().to_string();
    assert!(e.contains("identity 77"), "{e}");
    assert!(e.contains("TOTAL order"), "{e}");
}

/// A deposition handed a pool it was not sized for is refused by name rather
/// than indexing past its own scratch.
#[test]
fn a_deposition_sized_for_another_pool_is_refused_by_name() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block(4);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let small = Parcels::new(&gpu, &hm, &gm, still(64), &[], 0.1).unwrap();
    let big = Parcels::new(&gpu, &hm, &gm, still(4096), &[], 0.1).unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &small).unwrap();
    let e = dep.build(&gpu, &big).unwrap_err().to_string();
    assert!(e.contains("sized for capacity 64"), "{e}");

    let other = block(6);
    let gm2 = GpuMesh::upload(&gpu, &other).unwrap();
    let elsewhere = Parcels::new(&gpu, &other, &gm2, still(64), &[], 0.1).unwrap();
    let e = dep.build(&gpu, &elsewhere).unwrap_err().to_string();
    assert!(e.contains("different"), "{e}");
}

/// The scratch this costs is reported before it is spent, not discovered as
/// an out-of-memory at hour three. Roughly 26 bytes per slot plus 32 per
/// cell: for a million parcels on a million cells, about 60 MB.
#[test]
fn the_scratch_it_costs_is_reportable() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block(10);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let p = Parcels::new(&gpu, &hm, &gm, still(100_000), &[], 0.1).unwrap();
    let dep = ParcelDeposition::new(&gpu, &p).unwrap();
    let per_slot = dep.device_bytes() as f64 / dep.padded_capacity() as f64;
    assert!(
        (20.0..40.0).contains(&per_slot),
        "the sort scratch is {per_slot} bytes per slot, which is not what S67.9 published"
    );
}

// ======================================================================
//  SPEC-LIT (67.9): what it costs, measured rather than modelled
// ======================================================================

/// The published cost of S67, on this machine, at a realistic size.
///
/// Ignored by default because it allocates a million-slot pool on a
/// 262144-cell mesh and because a wall time is not a correctness claim. Run
/// it with
/// `cargo test --release --lib -- --ignored --nocapture the_cost_of_the_sort`
/// when the numbers in S67.9 need re-measuring.
///
/// The design note that preceded this section estimated the sort from a
/// bandwidth model and said so: *"these are bandwidth-model estimates, not
/// measurements, and they should be treated as a hypothesis to be tested."*
/// This is the test.
///
/// The pool is filled by an **injector** and not by [`Parcels::seed`],
/// because seeding is `O(n_seeds x n_cells)` on the host - `locate_cell`
/// (S66.6) is a linear scan, which is the right answer for a handful of
/// injectors at setup and hopeless for a million parcels. That is a real
/// limit of S66 and it is recorded here rather than discovered again.
#[test]
#[ignore = "allocates ~130 MB and measures a wall time, which is not a correctness claim"]
fn the_cost_of_the_sort_at_a_million_slots() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = block(64);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let u = crate::field::GpuVectorField::zeros(&gpu, &gm, "U").unwrap();
    let rho = gpu.upload(&vec![1.2 as Scalar; gm.n_cells]).unwrap();

    let capacity = 1_000_000usize;
    let per_event = 4000u32;
    let dt: Scalar = 2e-3;

    // Rebound at every wall, so nothing is removed and the live count is
    // exactly what was injected.
    let ctrl = ParcelControls {
        drag: DragModel::SchillerNaumann,
        wall: WallAction::Rebound,
        gravity: Vec3::new(0.0, 0.0, -9.81),
        ..still(capacity)
    };
    let inj = crate::parcels::Injector {
        position: Vec3::new(0.5, 0.5, 0.5),
        axis: Vec3::new(0.0, 0.0, -1.0),
        cone_half_angle: 1.2,
        standoff: 0.02,
        speed: 4.0,
        diameter: 2e-4,
        temperature: 300.0,
        mass_flow: 1e-3,
        parcels_per_event: per_event,
        interval: 0.0,
    };

    // TWO occupancies of the SAME pool, because the load-bearing statement is
    // that the sort is priced by `capacity` and not by how many parcels are
    // alive: the launch geometry has to be a setup constant (S67.7), so every
    // pass covers every slot whether or not it holds anything.
    let mut p = Parcels::new(&gpu, &hm, &gm, ctrl, &[inj], dt).unwrap();
    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    println!(
        "S67.9  {} slots, {} cells, {} radix passes ({} over the cell key), scratch {:.1} MB",
        dep.padded_capacity(),
        gm.n_cells,
        dep.passes(),
        dep.cell_passes(),
        dep.device_bytes() as f64 / 1048576.0
    );

    let reps = 20;
    let mut done = 0usize;
    for target in [50usize, 250] {
        while done < target {
            p.step(&gpu, &u, &rho, None, None, dt).unwrap();
            done += 1;
        }
        gpu.sync().unwrap();
        dep.update(&gpu, &p).unwrap();
        gpu.sync().unwrap();
        let live = dep.live_count(&gpu).unwrap();

        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            dep.build(&gpu, &p).unwrap();
        }
        gpu.sync().unwrap();
        let build = t0.elapsed().as_secs_f64() / reps as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            dep.deposit(&gpu, &p).unwrap();
        }
        gpu.sync().unwrap();
        let gather = t0.elapsed().as_secs_f64() / reps as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            p.step(&gpu, &u, &rho, None, None, dt).unwrap();
            done += 1;
        }
        gpu.sync().unwrap();
        let integrate = t0.elapsed().as_secs_f64() / reps as f64;

        let csr = dep.csr_snapshot(&gpu).unwrap();
        let longest =
            (0..csr.n_cells).map(|c| csr.offset[c + 1] - csr.offset[c]).max().unwrap_or(0);
        // Timing something wrong is not a measurement, so the million-slot
        // case is checked as well as clocked.
        let bad = csr_defects(&csr, &p.snapshot(&gpu).unwrap());
        assert!(bad.is_empty(), "{} defects at {live} live, first: {}", bad.len(), bad[0]);

        println!(
            "S67.9  {live} live: sort + CSR {:.3} ms, gather {:.3} ms, \
             one parcel step {:.3} ms, longest segment {longest}",
            build * 1e3,
            gather * 1e3,
            integrate * 1e3
        );
        assert!(build > 0.0 && gather > 0.0);
    }
}
