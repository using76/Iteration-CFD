// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! VIO-03's integration test, in BOTH builds: the volume writers' bytes.
//!
//! 1. `cartesian_bytes_unchanged` pins the SHA-256 of the four Cartesian
//!    files (`.vdb` and `.nvdb`, `fp32` and `fp16`) of one fixture whose
//!    values are exact in f32, so the f64 and the `single` build write the
//!    same bytes. The pins were taken from the tree before VIO-03 touched
//!    the writers (97f3d6a, after VIO-08's 224 / bufferCount / name change
//!    of SPEC-LIT 117.8), so the Cartesian path is byte-identical (117.6).
//! 2. `lattice_bytes_same_in_both_builds` pins the SHA-256 of lattice files
//!    (117.6, `dense` and `fluid`) written from a `VoxelMap` on the 2:1
//!    mesh of `mesh/refined.rs`. The map and the samples are bit-identical
//!    across the builds (117.2 items 1-2), so one literal holds in both. The
//!    files' contents are checked voxel by voxel by the f64 lib tests
//!    `io::vdb::tests::voxmap_round_trip_*` and
//!    `io::nvdb::tests::voxmap_round_trip`.
//!
//! 3. `volume_target_lays_the_world_lattice_on_a_local_mesh` (VIO-05a):
//!    `output_plan::volume_target` on a mesh the `single` build stores about
//!    a local origin lays the world mesh's map, owner for owner (117.2
//!    item 4).
//!
//! Written from SPEC-LIT section 117 (117.2, 117.6, 117.8).
//! Provenance: ORIGINAL.
//! No GPL-licensed source was consulted.

use std::path::PathBuf;

use ofgpu::io::nvdb::{self, Precision, UniformGrid};
use ofgpu::io::output_types::OutputField;
use ofgpu::io::output_plan::volume_target;
use ofgpu::io::polymesh::{build_host_mesh, build_host_mesh_local, PolyMeshRaw};
use ofgpu::io::vdb;
use ofgpu::io::voxelmap::{LatticeActivation, LatticeSample, LatticeSpec, VoxelMap};
use ofgpu::mesh::refined::refined_core;
use ofgpu::mesh::HostMesh;
use ofgpu::radiation::wsgg::sha256_hex;
use ofgpu::types::{DVec3, Tensor, Vec3};
use ofgpu::Scalar;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ofgpu_volume_lattice_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

// ==========================================================================
//  1. The Cartesian bytes
// ==========================================================================

/// 130 x 9 x 3: crosses a Lower-node boundary at x = 128, so the file has
/// two Lower nodes and partial leaves on every axis.
fn grid() -> UniformGrid {
    UniformGrid {
        nx: 130,
        ny: 9,
        nz: 3,
        origin: Vec3::new(0.25, -0.5, 1.0),
        spacing: Vec3::new(0.125, 0.25, 0.5),
    }
}

/// Three fields whose every value, and every vector magnitude, is exact in
/// f32: a ramp in quarters, vectors from the Pythagorean quadruples
/// (1,2,2,3), (2,3,6,7), (1,4,8,9), (2,6,9,11) scaled by halves, and a
/// symmetric tensor of the same kind of numbers.
fn fields(g: &UniformGrid) -> (Vec<Scalar>, Vec<Vec3>, Vec<Tensor>) {
    const Q: [[f64; 4]; 4] = [[1.0, 2.0, 2.0, 3.0], [2.0, 3.0, 6.0, 7.0], [1.0, 4.0, 8.0, 9.0], [2.0, 6.0, 9.0, 11.0]];
    let mut s = Vec::with_capacity(g.n());
    let mut u = Vec::with_capacity(g.n());
    let mut r = Vec::with_capacity(g.n());
    for k in 0..g.nz {
        for j in 0..g.ny {
            for i in 0..g.nx {
                let idx = g.idx(i, j, k);
                let sv = (i as f64 + 2.0 * j as f64 + 3.0 * k as f64) * 0.25 - 7.0;
                s.push(sv as Scalar);
                let q = Q[idx % 4];
                let t = ((idx % 7) as f64 + 1.0) * 0.5;
                let v = Vec3::new((q[0] * t) as Scalar, (-q[1] * t) as Scalar, (q[2] * t) as Scalar);
                assert_eq!(v.mag() as f64, q[3] * t, "the fixture's magnitudes are exact");
                u.push(v);
                let off = 0.5 * sv;
                let yz = 0.125 * (idx % 5) as f64;
                r.push(Tensor {
                    xx: sv as Scalar, xy: off as Scalar, xz: 1.0,
                    yx: off as Scalar, yy: -sv as Scalar, yz: yz as Scalar,
                    zx: 1.0, zy: yz as Scalar, zz: 2.0,
                });
            }
        }
    }
    (s, u, r)
}

/// The SHA-256 of the four Cartesian files of this fixture as the tree wrote
/// them at 97f3d6a, before VIO-03 (the baseline after VIO-08, SPEC-LIT
/// 117.8). The same literals hold in both builds.
const CARTESIAN_PINS: [(&str, &str); 4] = [
    ("vdb fp32", "5f897e756a3c33ef0ba357a138f5dbc7bfe641fde706a3840e47cda031e12642"),
    ("vdb fp16", "da492a7dc02935b3d7a57e96b9d736179ff849435f72d54832b898e33364af91"),
    ("nvdb fp32", "6ede35cead8ad02a904a0b138ff50a4a6cd80fca6e0fb8519c71404fcf87aa35"),
    ("nvdb fp16", "45251abc3f0b783d3c07d55fb3735590df3a41c8a32336c359bb01e22254f90f"),
];

#[test]
fn cartesian_bytes_unchanged() {
    let g = grid();
    let (s, u, r) = fields(&g);
    let f = [OutputField::scalar("p", &s), OutputField::vector("U", &u), OutputField::tensor("R", &r)];
    let dir = scratch("cartesian");
    let mut got = Vec::new();
    for (label, p) in [("fp32", Precision::F32), ("fp16", Precision::F16)] {
        let path = dir.join(format!("c_{label}.vdb"));
        vdb::write(&path, &g, &f, p).expect("vdb::write");
        got.push((format!("vdb {label}"), sha256_hex(&std::fs::read(&path).unwrap())));
    }
    for (label, p) in [("fp32", Precision::F32), ("fp16", Precision::F16)] {
        let path = dir.join(format!("c_{label}.nvdb"));
        nvdb::write(&path, &g, &f, p).expect("nvdb::write");
        got.push((format!("nvdb {label}"), sha256_hex(&std::fs::read(&path).unwrap())));
    }
    let _ = std::fs::remove_dir_all(&dir);
    let want: Vec<(String, String)> =
        CARTESIAN_PINS.iter().map(|(l, h)| (l.to_string(), h.to_string())).collect();
    assert_eq!(got, want, "the Cartesian bytes changed (left: written now, right: the 97f3d6a pins)");
}

// ==========================================================================
//  2. The lattice bytes, the same in both builds
// ==========================================================================

/// The 2:1 refined box of `mesh/refined.rs` on `[0, 1]^3`: 120 cells.
fn refined() -> (PolyMeshRaw, HostMesh) {
    let rb = refined_core([4, 4, 4], Vec3::new(0.25, 0.25, 0.25), 0.3, 1).expect("refined_core");
    let points: Vec<DVec3> =
        rb.points.iter().map(|&p| DVec3::new(p.x as f64, p.y as f64, p.z as f64)).collect();
    let mut owner = rb.mesh.owner.clone();
    owner.extend(rb.mesh.b_face_cells.iter().copied());
    let raw = PolyMeshRaw {
        points,
        faces: rb.faces.clone(),
        owner,
        neighbour: rb.mesh.neighbour.clone(),
        patches: rb.mesh.patches.clone(),
    };
    let hm = build_host_mesh(&raw).expect("build_host_mesh");
    (raw, hm)
}

/// `h = 1/24` over `[-1/2, 6] x [-1/2, 3/2]^2` (156 x 48 x 48 voxels, two
/// Lower nodes along x, the fluid in one), background -7.5.
fn spec(sample: LatticeSample, activation: LatticeActivation) -> LatticeSpec {
    LatticeSpec {
        voxel: Some(1.0 / 24.0),
        bbox: Some([-0.5, -0.5, -0.5, 6.0, 1.5, 1.5]),
        sample,
        activation,
        background: -7.5,
        ..Default::default()
    }
}

/// The SHA-256 of the lattice files, the same literals in both builds:
/// cell values exact in f32 (`((37 c) mod 101) / 8 - 3`, and `U = (3t, -4t,
/// 0)` with `t = c/16`, whose magnitude 5t is exact in both builds), so the
/// samples - f64 from the same f64 geometry - are the same bits.
const LATTICE_PINS: [(&str, &str); 4] = [
    ("vdb fluid linear fp32", "122b3656b5696c2db4c9304fe2f645ec3a636cd0b4d263dd731fc0dbaf6fed26"),
    ("vdb dense cell fp16", "eae7c546e9d25f9f4aec1f1e97af33ec9d863b027143919ed7ec8f52f97f9b69"),
    ("nvdb fluid linear fp32", "847301c6e6622837d72d96be4717f7f4496b72533a7379de9c2dd1b9611c42f8"),
    ("nvdb dense cell fp16", "d8f6421de0b834c35c31dd14169a31f01a2d85c01bb03d19f0dcdb612975e630"),
];

#[test]
fn lattice_bytes_same_in_both_builds() {
    let (raw, hm) = refined();
    let phi: Vec<Scalar> = (0..hm.n_cells).map(|c| ((37 * c) % 101) as Scalar * 0.125 - 3.0).collect();
    let u: Vec<Vec3> = (0..hm.n_cells)
        .map(|c| {
            let t = c as Scalar / 16.0;
            Vec3::new(3.0 * t, -4.0 * t, 0.0)
        })
        .collect();
    let f = [OutputField::scalar("p", &phi), OutputField::vector("U", &u)];
    let dir = scratch("lattice");
    let cases = [
        ("fluid linear fp32", LatticeSample::Linear, LatticeActivation::Fluid, Precision::F32),
        ("dense cell fp16", LatticeSample::Cell, LatticeActivation::Dense, Precision::F16),
    ];
    let mut got = Vec::new();
    for fmt in ["vdb", "nvdb"] {
        for (label, sample, activation, precision) in cases {
            let vm = VoxelMap::build(&raw, &hm, &spec(sample, activation)).expect("VoxelMap::build");
            assert_eq!(vm.n, [156, 48, 48]);
            let path = dir.join(format!("l.{fmt}"));
            if fmt == "vdb" {
                vdb::write_lattice(&path, &vm, &f, precision).expect("vdb::write_lattice");
            } else {
                nvdb::write_lattice(&path, &vm, &f, precision).expect("nvdb::write_lattice");
            }
            got.push((format!("{fmt} {label}"), sha256_hex(&std::fs::read(&path).unwrap())));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    let want: Vec<(String, String)> =
        LATTICE_PINS.iter().map(|(l, h)| (l.to_string(), h.to_string())).collect();
    assert_eq!(got, want, "the lattice bytes (left: written now, right: the pins)");
}

// ==========================================================================
//  3. VIO-05a: the lattice is laid in world coordinates on a local mesh
// ==========================================================================

/// SPEC-LIT 117.2 item 4 through `output_plan::volume_target`: the 2:1 mesh
/// moved to `(1000, -2000, 500)`, which the `single` build stores about a
/// local origin (F32-10, `build_host_mesh_local`). The map laid from that
/// mesh is the map of the world mesh, owner for owner, with the same world
/// transform - in both builds, so the `single` map is the f64 one.
#[test]
fn volume_target_lays_the_world_lattice_on_a_local_mesh() {
    let (raw, _) = refined();
    let shift = DVec3::new(1000.0, -2000.0, 500.0);
    let raw = PolyMeshRaw { points: raw.points.iter().map(|&p| p + shift).collect(), ..raw };
    let local = build_host_mesh_local(&raw).expect("build_host_mesh_local");
    #[cfg(feature = "single")]
    assert_ne!(local.origin, DVec3::ZERO, "the single build shifts this mesh");
    #[cfg(not(feature = "single"))]
    assert_eq!(local.origin, DVec3::ZERO, "the f64 build never shifts");
    let spec = LatticeSpec { voxel: Some(1.0 / 24.0), ..LatticeSpec::default() };
    let t = volume_target(Some(&raw), &local, None, &spec).expect("volume_target");
    let vm = t.vox().expect("a lattice, not a Cartesian box");
    let world = VoxelMap::build(&raw, &build_host_mesh(&raw).expect("build_host_mesh"), &spec).expect("build");
    assert_eq!(vm.n, [24, 24, 24]);
    assert_eq!(vm.n, world.n);
    assert_eq!(vm.origin(), world.origin(), "the world transform");
    assert_eq!(vm.origin(), DVec3::new(1000.0 + 1.0 / 48.0, -2000.0 + 1.0 / 48.0, 500.0 + 1.0 / 48.0));
    let mut owned = 0;
    for k in 0..24 {
        for j in 0..24 {
            for i in 0..24 {
                assert_eq!(vm.owner(i, j, k), world.owner(i, j, k), "voxel ({i}, {j}, {k})");
                owned += usize::from(vm.owner(i, j, k).is_some());
            }
        }
    }
    assert_eq!(owned, 24 * 24 * 24, "the default box is the mesh's own");
    assert_eq!(vm.report.n_fluid, world.report.n_fluid);
}
