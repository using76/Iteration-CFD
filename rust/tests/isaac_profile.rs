// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
// Source-available, not Open Source. Teaching and academic research are
// free; commercial and non-academic research require a licence.
// Enquiries: simul@msimul.com
// See LICENSE at the repository root.
// Provenance: see PROVENANCE.md. No GPL-licensed source was consulted.

//! VIO-09's integration test, in BOTH builds: the Isaac profile's gate and
//! its scene bytes (SPEC-LIT 117.9).
//!
//! 1. `optical_depth_gate_in_this_build` - Max 1995's gate AT THE FILE'S
//!    precision, through the public API alone: the densest column's optical
//!    depth of the f32-rounded `sigma` is `tau` within 1 %, and `D` is the
//!    f64 (117.11)/(117.12) answer. The measured relative error is printed
//!    for the 117.13 table.
//! 2. `physical_k_exact_in_this_build` - `K_m = 8.7 m^2/g . rho_soot`
//!    exactly, in this build too (Mulholland & Croarkin 2000, 117.14).
//! 3. `isaac_scene_fixtures` - both profiles' `.usda` scene, written by
//!    `build_writers_isaac` itself, compared byte for byte with
//!    `tools/isaac/fixtures/<profile>/isaac.usda`. The same bytes must come
//!    out of the f64 build and the `single` build - the fixture values and
//!    the box coordinates are exact in f32, so they do. Set
//!    `OFGPU_BLESS_ISAAC=1` to (re)write the fixtures instead of comparing.
//!
//! Written from SPEC-LIT section 117 (117.9, 117.13, 117.14).
//! Provenance: ORIGINAL.
//! No GPL-licensed source was consulted.

use std::path::{Path, PathBuf};

use ofgpu::io::nvdb::Precision;
use ofgpu::io::output_plan::{build_writers_isaac, IsaacKind, IsaacScaling, IsaacSpec};
use ofgpu::io::output_types::OutputField;
use ofgpu::io::polymesh::{build_host_mesh, PolyMeshRaw};
use ofgpu::io::usda::{IsaacProfile, StageMeta};
use ofgpu::io::voxelmap::{
    fluid_range, normalise_for_optical_depth, optical_depth_target, physical_extinction,
    LatticeSpec, SMOKE_SPECIFIC_EXTINCTION, VoxelMap,
};
use ofgpu::io::writer::WriteCtx;
use ofgpu::mesh::refined::refined_core;
use ofgpu::types::{DVec3, Vec3};
use ofgpu::Scalar;

/// The 2:1 mesh of `mesh/refined.rs`, exactly as `rust/tests/volume_lattice.rs`
/// builds it: `refined_core([4,4,4], 0.25, 0.3, 1)` on the world origin - the
/// coordinates stay exact in f32, so both builds hold the same geometry.
fn refined() -> (PolyMeshRaw, ofgpu::mesh::HostMesh) {
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

/// The fixture directory, from the workspace root this crate sits in.
fn fixtures_dir(profile: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate has a parent")
        .join("tools/isaac/fixtures")
        .join(profile)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ofgpu_isaac_profile_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// The Isaac spec of 117.9's table, every key at its default but the
/// profile: `opticalDepth`, `tau` 2.0, the default (fluid) range, `K_m`
/// 8700, kind `density`, stage Z / 1.0.
fn isaac_spec(profile: IsaacProfile) -> IsaacSpec {
    IsaacSpec {
        profile,
        field: "smoke".to_string(),
        scaling: IsaacScaling::OpticalDepth,
        tau: 2.0,
        range: None,
        km: SMOKE_SPECIFIC_EXTINCTION,
        kind: IsaacKind::Density,
        stage: StageMeta::default(),
    }
}

/// (117.11) written out again, so the gate's check is independent of the
/// function under test - the same independent copy the lib test uses.
fn own_max_column(n: [usize; 3], h: [f64; 3], s: &[f64]) -> f64 {
    let at = |i: usize, j: usize, k: usize| i + n[0] * (j + n[1] * k);
    let mut best = 0.0f64;
    for k in 0..n[2] {
        for j in 0..n[1] {
            let mut sum = 0.0f64;
            for i in 0..n[0] {
                sum += s[at(i, j, k)];
            }
            best = best.max(h[0] * sum);
        }
    }
    for k in 0..n[2] {
        for i in 0..n[0] {
            let mut sum = 0.0f64;
            for j in 0..n[1] {
                sum += s[at(i, j, k)];
            }
            best = best.max(h[1] * sum);
        }
    }
    for j in 0..n[1] {
        for i in 0..n[0] {
            let mut sum = 0.0f64;
            for k in 0..n[2] {
                sum += s[at(i, j, k)];
            }
            best = best.max(h[2] * sum);
        }
    }
    best
}

/// VIO-09's gate, IN THIS BUILD, through the public API: the densest
/// column's optical depth equals `tau` exactly in f64 (1e-12 relative) and
/// within 1 % after `sigma` is rounded to the file's f32. The measured
/// errors go to stdout for the 117.13 table.
#[test]
fn optical_depth_gate_in_this_build() {
    let (raw, hm) = refined();
    let spec = LatticeSpec { voxel: Some(1.0 / 24.0), ..Default::default() };
    let vm = VoxelMap::build(&raw, &hm, &spec).expect("VoxelMap::build");
    let g = hm.geom64();
    let centre = DVec3::new(0.5, 0.5, 0.5);
    let f: Vec<Scalar> = (0..hm.n_cells)
        .map(|c| {
            let d = g.c[c] - centre;
            (-(d.dot(d)) / 0.05).exp() as Scalar
        })
        .collect();
    let vox = vm.sample(&f).expect("sample");
    let mask = vm.fluid_mask();
    let n = vm.n;
    let h = [vm.voxel; 3];
    let range = fluid_range(&vox, Some(&mask)).expect("the plume has fluid voxels");
    let f_hat = normalise_for_optical_depth(&vox, Some(&mask), range);
    for tau in [2.0, 0.5] {
        let (sigma, d) = optical_depth_target(n, h, &f_hat, tau);
        assert_eq!(d, own_max_column(n, h, &f_hat), "D is the public (117.11)");
        let got = own_max_column(n, h, &sigma);
        let rel = ((got - tau) / tau).abs();
        assert!(rel < 1e-12, "tau {tau}: densest column {got} vs tau, rel {rel}");
        assert!(got <= tau * (1.0 + 1e-12), "tau {tau}: a column reaches {got}");
        let s32: Vec<f64> = sigma.iter().map(|&s| s as f32 as f64).collect();
        let got32 = own_max_column(n, h, &s32);
        let rel32 = ((got32 - tau) / tau).abs();
        println!(
            "optical_depth_gate_in_this_build: tau {tau}, f32-rounded densest column {got32}, \
             relative error {rel32:.3e}"
        );
        assert!(rel32 < 1e-2, "tau {tau}: f32 relative error {rel32} >= 1 %");
    }
}

/// (117.14) in this build: `K_m = 8.7 m^2/g = 8700 m^2/kg` exactly, and
/// `sigma = km . rho_s` bit for bit - the f32 build multiplies the same
/// f64 constants.
#[test]
fn physical_k_exact_in_this_build() {
    assert_eq!(SMOKE_SPECIFIC_EXTINCTION, 8.7 * 1000.0, "K_m is 8.7 m^2/g in m^2/kg");
    let rho_s = [0.0, 1e-3, 2.5e-4, -1e-5];
    let (sigma, negatives) = physical_extinction(&rho_s, None, SMOKE_SPECIFIC_EXTINCTION);
    assert_eq!(sigma[0], 0.0);
    assert_eq!(sigma[1], 8700.0 * 1e-3, "sigma = K_m rho_s bit for bit");
    assert_eq!(sigma[2], 8700.0 * 2.5e-4);
    assert_eq!(sigma[3], 0.0, "a negative rho_s is written 0 and counted");
    assert_eq!(negatives, 1);
}

/// Both profiles' scene, written by `build_writers_isaac` itself on the 2:1
/// mesh - lattice `h = 1/8` on the default box [0,1]^3 - and compared byte
/// for byte with the fixture `tools/isaac/fixtures/<profile>/isaac.usda`.
/// The field values `((37 c) mod 101) / 128` and the box coordinates are
/// exact in f32, so the f64 build and the `single` build must print the
/// same bytes; a difference is a failure, reported line by line.
#[test]
fn isaac_scene_fixtures() {
    let (raw, hm) = refined();
    let spec = LatticeSpec { voxel: Some(1.0 / 8.0), ..Default::default() };
    let vm = VoxelMap::build(&raw, &hm, &spec).expect("VoxelMap::build");
    assert_eq!(vm.n, [8, 8, 8], "h = 1/8 on the default box [0,1]^3");
    let smoke: Vec<Scalar> =
        (0..hm.n_cells).map(|c| ((37 * c) % 101) as Scalar / 128.0).collect();
    let fields = [OutputField::scalar("smoke", &smoke)];
    let foam: [ofgpu::io::writer::FoamField; 0] = [];
    let bless = std::env::var("OFGPU_BLESS_ISAAC").ok().as_deref() == Some("1");

    for (profile_name, profile) in
        [("index", IsaacProfile::Index), ("pathtrace", IsaacProfile::Pathtrace)]
    {
        let dir = scratch(profile_name);
        // D2: the formats the caller keeps are the case's own; the scene is
        // named with them - [Vdb, Usda] here, as isaac_formats resolves it.
        let mut writers = build_writers_isaac(
            &dir,
            "isaac",
            &[
                ofgpu::io::output_plan::OutputFormat::Vdb,
                ofgpu::io::output_plan::OutputFormat::Usda,
            ],
            Precision::F32,
            Some(&isaac_spec(profile)),
        )
        .expect("build_writers_isaac");
        let ctx = WriteCtx {
            time: 0.0,
            step: 0,
            name: "0",
            mesh: &hm,
            cart: None,
            vox: Some(&vm),
            fields: &fields,
            foam: &foam,
        };
        for w in &mut writers {
            w.write_step(&ctx).expect("write");
        }
        let got_path = dir.join("isaac.usda");
        let vdb_path = dir.join("VDB/isaac_000000.vdb");
        let fixture = fixtures_dir(profile_name);
        if bless {
            // Only the scene is a fixture: the .vdb it references is binary
            // and usda_check.py's selftest gives its copies a placeholder.
            std::fs::create_dir_all(&fixture).expect("fixtures dir");
            std::fs::copy(&got_path, fixture.join("isaac.usda")).expect("bless the scene");
            println!(
                "isaac_scene_fixtures: blessed {} (scene {} bytes)",
                profile_name,
                std::fs::metadata(fixture.join("isaac.usda")).unwrap().len(),
            );
        } else {
            let got = std::fs::read_to_string(&got_path).expect("the scene was written");
            let want = std::fs::read_to_string(fixture.join("isaac.usda"))
                .expect("the fixture - run with OFGPU_BLESS_ISAAC=1 to write it");
            for (line, (a, b)) in
                got.lines().zip(want.lines()).enumerate().filter(|(_, (a, b))| a != b)
            {
                panic!(
                    "isaac_scene_fixtures ({profile_name}): line {} differs:\n  got  {a:?}\n  want {b:?}",
                    line + 1
                );
            }
            assert_eq!(
                got.len(),
                want.len(),
                "isaac_scene_fixtures ({profile_name}): the byte lengths differ"
            );
            let vdb = std::fs::metadata(&vdb_path).expect("the .vdb was written").len();
            println!(
                "isaac_scene_fixtures: {profile_name} matches the fixture \
                 (scene {} bytes, vdb {vdb} bytes)",
                got.len()
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
