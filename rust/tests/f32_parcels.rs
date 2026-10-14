// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! SPEC-LIT 118.9 - parcel positions and the face-crossing walk in f64, and
//! the exchange sums in `ofacc`/`Acc` (F32-12). Under `single` a parcel's
//! position is an f64 on the device, stored LOCAL to the pool mesh's origin
//! (118.7) with seeds and injectors given in WORLD coordinates and the
//! snapshot answering in WORLD coordinates; the per-cell sums of the
//! deposition and the coupling gather, and every host total, accumulate in
//! f64 and round ONCE to `Scalar`. An integration test because the library's
//! own test target does not compile under `single`; every test runs in BOTH
//! builds with the same assertions.
//!
//! Provenance: ORIGINAL - the meshes are this tree's own generators; no
//! GPL-licensed source was consulted.

use ofgpu::adapt::Forest;
use ofgpu::blockgen::{self, BlockSpec, GradedAxis};
use ofgpu::field::GpuVectorField;
use ofgpu::io::polymesh::{build_host_mesh, build_host_mesh_local};
use ofgpu::mesh::PatchKind;
use ofgpu::parcels::{
    locate_cell64, DragModel, Injector, ParcelControls, ParcelDeposition, Parcels, SeedParcel,
    WallAction,
};
use ofgpu::types::{to_vec3, DVec3};
use ofgpu::{DevBuf, Gpu, GpuMesh, HostMesh, Label, Scalar, Vec3};

/// The build's `Scalar` widened to the f64 the answers are compared in: a
/// no-op in the f64 build, the exact widening under `single`.
#[allow(clippy::useless_conversion)]
fn w64(x: Scalar) -> f64 {
    f64::from(x)
}

/// `tests/f32_local_origin.rs`'s `box_spec`: `[lo, hi]` axes, uniform, the
/// given patch types in `(xMin, xMax, yMin, yMax, zMin, zMax)` order.
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

/// A fixed hash of the seed index into a unit direction at speed 0.5..1.5.
/// Any deterministic mix will do; this is SplitMix64's finaliser, the same
/// bijection the parcel identity uses, read here as three 16-bit slices.
fn spray_direction(id: usize) -> Vec3 {
    let mut z = (id as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z ^= z >> 30;
    z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let u01 = |w: u64| (w & 0xFFFF) as f64 / 65536.0;
    let speed = (0.5 + u01(z)) as Scalar;
    let th = (std::f64::consts::TAU * u01(z >> 16)) as Scalar;
    let cz = (2.0 * u01(z >> 32) - 1.0) as Scalar;
    let sz = (1.0 - cz * cz).sqrt();
    Vec3::new(speed * sz * th.cos(), speed * sz * th.sin(), speed * cz)
}

// ==========================================================================
//  T1 - (118.9a): the displacement is widened exactly as it is added
// ==========================================================================

/// A parcel moving 1e-9 m per step at x = 0.4 - below half an ulp of the
/// f32 position - must still move: the step displacement is formed in float
/// exactly as before and ADDED in f64, so 1000 steps accumulate to a
/// billionth of a metre per step times 1000 without losing one.
#[test]
fn a_billionth_of_a_metre_per_step_is_not_lost() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = box_mesh([4, 4, 4], [0.0; 3], [1.0; 3]);
    assert_eq!(hm.origin, DVec3::ZERO, "the unit box has no local origin");
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (u, rho) = still_gas(&gpu, &gm, 1.2);

    let dt: Scalar = 1e-3;
    let v: Scalar = 1e-6;
    let mut p = Parcels::new(&gpu, &hm, &gm, controls(8), &[], dt).unwrap();
    p.seed(
        &gpu,
        &hm,
        &[SeedParcel {
            position: Vec3::new(0.4, 0.5, 0.5),
            velocity: Vec3::new(v, 0.0, 0.0),
            ..Default::default()
        }],
    )
    .unwrap();
    for _ in 0..1000 {
        p.step(&gpu, &u, &rho, None, None, dt).unwrap();
    }
    let s = p.snapshot(&gpu).unwrap();

    let e = w64(0.4) + 1000.0 * w64(dt) * w64(v);
    assert!(
        (s.x64[0].x - e).abs() <= 1e-12,
        "the f64 position {} is not the accumulated displacement {e}",
        s.x64[0].x
    );
    assert!(
        (w64(s.x[0].x) - e).abs() <= 1e-12 + w64(Scalar::EPSILON) * e,
        "the Scalar position {} is not the accumulated displacement {e}",
        s.x[0].x
    );
    assert_eq!(s.x64[0].y, w64(0.5), "y does not move");
    assert_eq!(s.x64[0].z, w64(0.5), "z does not move");
    assert_eq!(
        s.cell[0] as usize,
        locate_cell64(&hm, DVec3::new(e, 0.5, 0.5)).expect("inside"),
        "the walk's cell is the cell the position is in"
    );
}

// ==========================================================================
//  T2 - SPEC-LIT 118.7: stored LOCAL, seeded and answered in WORLD
// ==========================================================================

/// The pool is built on a mesh stored about a snapped local origin 2000 m
/// from the axis - the case `Parcels::new` refused by name. Seeds and
/// injectors are given in WORLD coordinates, the device stores LOCAL
/// positions, and the snapshot answers in WORLD coordinates, so a parcel
/// carried on this mesh moves in world space and sits in the cell its world
/// position names.
#[test]
fn a_local_origin_mesh_carries_parcels_in_world_coordinates() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = build_host_mesh_local(
        &blockgen::raw_mesh(&box_spec(
            [4, 4, 4],
            [2000.0, 2000.0, 0.0],
            [2001.0, 2001.0, 1.0],
            ["wall"; 6],
        ))
        .expect("raw"),
    )
    .expect("local mesh");
    if cfg!(feature = "single") {
        assert!(hm.origin != DVec3::ZERO, "the single build shifts to a local origin");
    } else {
        assert!(hm.origin == DVec3::ZERO, "the f64 build never shifts");
    }
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (u, rho) = still_gas(&gpu, &gm, 1.2);

    // `Parcels::new` must ACCEPT the local-origin mesh now - at HEAD it
    // refused it with "parcels are F32-12".
    let dt: Scalar = 1e-3;
    let seed = Vec3::new(2000.375, 2000.5, 0.5);
    let vel = Vec3::new(1e-3, -5e-4, 0.0);
    let mut p = Parcels::new(&gpu, &hm, &gm, controls(8), &[], dt).unwrap();
    p.seed(
        &gpu,
        &hm,
        &[SeedParcel {
            position: seed,
            velocity: vel,
            ..Default::default()
        }],
    )
    .unwrap();
    for _ in 0..100 {
        p.step(&gpu, &u, &rho, None, None, dt).unwrap();
    }
    let s = p.snapshot(&gpu).unwrap();

    let e = DVec3::new(
        w64(seed.x) + 100.0 * w64(dt) * w64(vel.x),
        w64(seed.y) + 100.0 * w64(dt) * w64(vel.y),
        w64(seed.z) + 100.0 * w64(dt) * w64(vel.z),
    );
    assert!(
        (s.x64[0].x - e.x).abs() <= 1e-10
            && (s.x64[0].y - e.y).abs() <= 1e-10
            && (s.x64[0].z - e.z).abs() <= 1e-10,
        "the world position {:?} is not the accumulated displacement {e:?}",
        s.x64[0]
    );
    assert_eq!(s.x[0], to_vec3(s.x64[0]), "x is to_vec3(x64[i]) bit for bit");
    assert_eq!(
        s.cell[0] as usize,
        locate_cell64(&hm, hm.local64(e)).expect("inside"),
        "the walk's cell is the cell the world position is in"
    );

    // And an INJECTOR on the same mesh: built Ok, it injects, and every
    // live parcel sits in the cell its own world position names.
    let inj = Injector {
        position: Vec3::new(2000.5, 2000.5, 0.5),
        mass_flow: 1e-6,
        diameter: 1e-4,
        ..Default::default()
    };
    let mut q = Parcels::new(&gpu, &hm, &gm, controls(8), &[inj], dt).unwrap();
    q.step(&gpu, &u, &rho, None, None, dt).unwrap();
    let st = q.stats(&gpu).unwrap();
    assert!(st.n_injected >= 1, "the injector injected nothing: {st:?}");
    let sq = q.snapshot(&gpu).unwrap();
    assert!(sq.live().len() >= 1, "no live parcel after one step: {st:?}");
    for i in sq.live() {
        let cell = locate_cell64(&hm, hm.local64(sq.x64[i])).expect("inside");
        assert_eq!(sq.cell[i] as usize, cell, "live parcel {i} is mislocated");
    }
}

// ==========================================================================
//  T3 - docs/17 row F32-12's gate: nobody is lost on the 2:1 adapted mesh
// ==========================================================================

/// 4096 ballistic parcels on a 2:1 adapted box - the refinement the static
/// generator cannot express - bounce elastically for 300 steps. The walk
/// must not lose, wall-count or escape one of them: every boundary face is a
/// `wall` patch under a restitution-1 rebound, so a parcel that leaves a cell
/// always has another cell to be in.
///
/// What this unit does NOT fix is reported, not asserted: the number of live
/// parcels whose cell is not the cell their own position names - a known
/// defect of the plane-crossing walk at 2:1 faces (380 at HEAD, in BOTH
/// builds), whose fix is barycentric tracking (Macpherson et al. 2009).
#[test]
fn the_2to1_adapted_walk_loses_no_parcel() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let mut lev = vec![0u32; 8 * 8 * 8];
    for i in 3..5 {
        for j in 3..5 {
            for k in 3..5 {
                lev[i + 8 * (j + 8 * k)] = 1;
            }
        }
    }
    let mut hm = Forest::from_base_levels([8, 8, 8], Vec3::new(0.125, 0.125, 0.125), &lev)
        .expect("forest")
        .build()
        .expect("adapted box")
        .mesh;
    // A wall on every boundary face, so nothing can escape: under a
    // restitution-1 rebound the boundary reflects and counts nothing.
    for p in hm.patches.iter_mut() {
        p.kind = PatchKind::Wall;
    }
    for k in hm.b_kind.iter_mut() {
        *k = PatchKind::Wall as Label;
    }
    assert_eq!(hm.origin, DVec3::ZERO, "the adapted box has no local origin");
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();
    let (u, rho) = still_gas(&gpu, &gm, 1.2);

    let mut seeds = Vec::with_capacity(4096);
    for a in 0..16usize {
        for b in 0..16usize {
            for c in 0..16usize {
                seeds.push(SeedParcel {
                    position: Vec3::new(
                        (a as Scalar + 0.37) / 16.0,
                        (b as Scalar + 0.41) / 16.0,
                        (c as Scalar + 0.43) / 16.0,
                    ),
                    velocity: spray_direction(a + 16 * (b + 16 * c)),
                    ..Default::default()
                });
            }
        }
    }
    let dt: Scalar = 0.01;
    let mut p = Parcels::new(&gpu, &hm, &gm, controls(seeds.len()), &[], dt).unwrap();
    p.seed(&gpu, &hm, &seeds).unwrap();
    for _ in 0..300 {
        p.step(&gpu, &u, &rho, None, None, dt).unwrap();
    }

    let st = p.stats(&gpu).unwrap();
    assert_eq!(st.n_lost, 0, "the walk lost a parcel on the adapted mesh");
    assert_eq!(st.n_wall, 0, "a rebounding parcel never ends at the wall");
    assert_eq!(st.n_escaped, 0, "every patch is a wall; nothing escapes");
    let s = p.snapshot(&gpu).unwrap();
    assert_eq!(s.live().len(), 4096, "every parcel is still alive");

    let mut misplaced = 0usize;
    for i in s.live() {
        if locate_cell64(&hm, hm.local64(s.x64[i])) != Some(s.cell[i] as usize) {
            misplaced += 1;
        }
    }
    eprintln!(
        "f32_parcels T3: {misplaced} of {} live parcels sit in a cell their own position \
         is not in - the known walk defect at 2:1 faces this unit does not fix",
        s.live().len()
    );
}

// ==========================================================================
//  T4 - (118.9d): the exchange sums accumulate in f64 and round ONCE
// ==========================================================================

/// 65536 parcels of dyadic `n_p` in one cell: the deposited weight is the
/// f64 sum narrowed once, which is EXACTLY 98272 - at HEAD under `single`
/// the f32 accumulation gives 98271.5703 - and the deposited mass and the
/// snapshot's `liquid_mass` are the f64 sums narrowed once, inside one f32
/// ulp of the exact f64 value.
#[test]
fn the_exchange_sums_round_once() {
    let Some(gpu) = Gpu::new(0).ok() else { return };
    let hm = box_mesh([2, 1, 1], [0.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
    let gm = GpuMesh::upload(&gpu, &hm).unwrap();

    let n = 65536usize;
    let d: Scalar = 1e-4;
    let np = |i: usize| 1.0 + (i % 1024) as Scalar / 1024.0;
    let seeds: Vec<SeedParcel> = (0..n)
        .map(|i| SeedParcel {
            position: Vec3::new(0.5, 0.5, 0.5),
            velocity: Vec3::ZERO,
            diameter: d,
            temperature: 293.15,
            n_p: np(i),
            uid: None,
        })
        .collect();
    let mut p = Parcels::new(&gpu, &hm, &gm, controls(n), &[], 1e-3).unwrap();
    p.seed(&gpu, &hm, &seeds).unwrap();

    let mut dep = ParcelDeposition::new(&gpu, &p).unwrap();
    dep.update(&gpu, &p).unwrap();
    let ds = dep.snapshot(&gpu).unwrap();

    assert_eq!(
        ds.weight[0].to_bits(),
        (98272.0 as Scalar).to_bits(),
        "the deposited weight is the f64 sum narrowed once, which is exact here"
    );
    assert_eq!(ds.total_weight(), 98272.0);

    // outMass = rhoL * volSum: the sum in f64, rhoL multiplied once, ONE
    // narrowing - the expected value forms the same product order.
    let rho_l: Scalar = 1000.0;
    let pi6 = w64(std::f64::consts::FRAC_PI_6 as Scalar);
    let mut vol_sum = 0.0f64;
    for i in 0..n {
        vol_sum += w64(np(i)) * pi6 * w64(d) * w64(d) * w64(d);
    }
    let em = w64(rho_l) * vol_sum;
    assert!(
        (w64(ds.mass[0]) - em).abs() <= 2.4e-7 * em,
        "deposited mass {} against the exact sum {em}",
        ds.mass[0]
    );

    // liquid_mass: the same sum, rhoL inside the term, as the snapshot
    // forms it.
    let s = p.snapshot(&gpu).unwrap();
    let el: f64 = (0..n)
        .map(|i| w64(np(i)) * w64(rho_l) * pi6 * w64(d) * w64(d) * w64(d))
        .sum();
    assert!(
        (w64(s.liquid_mass(rho_l)) - el).abs() <= 2.4e-7 * el,
        "liquid_mass {} against the exact sum {el}",
        s.liquid_mass(rho_l)
    );
}
